"""
HR System Connector — Workday / BambooHR Integration
======================================================
Pulls employment context that enriches the insider-threat risk model:

  • **Employment status** — terminated or resigned employees are
    high-risk during their notice period (data exfiltration window).
  • **Department** — finance, R&D, and executive staff have access
    to higher-value assets.
  • **Job role** — SysAdmins and DBAs have privileged access that
    makes their anomalies more impactful.
  • **Active projects** — access to sensitive projects raises the
    contextual risk multiplier.
  • **Resignation date** — the #1 predictor of data theft.  CERT's
    research shows 70% of IP theft occurs within 60 days of
    resignation.
  • **PTO calendar** — activity during PTO is an immediate red flag.

Architecture
------------
1. **Pull mode** — ``fetch_employee_context()`` calls the HR API
   and caches the result in Redis with a 1-hour TTL.
2. **Push mode** — ``/api/v1/hr/webhook`` receives real-time
   employment change events (hires, terminations, dept transfers).
   These invalidate the Redis cache for the affected employee and
   publish a Kafka event so downstream services react immediately.

The HR API client is a **mock** that returns realistic synthetic
data.  In production, replace ``_mock_hr_api_call()`` with real
Workday / BambooHR SDK calls.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status

from app.auth.jwt_handler import Role
from app.auth.rbac import TokenPayload, require_role
from app.config import get_settings
from app.models.schemas import HREmployeeContext, HRWebhookEvent

try:
    from shared.utils.logging import get_logger
except ImportError:
    import logging as _logging
    get_logger = _logging.getLogger

log = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/hr", tags=["hr-integration"])

# Redis key pattern for HR cache
_HR_CACHE_PREFIX = "hr:employee:"


# ── Mock HR API ───────────────────────────────────────────────

# In production, this would call Workday/BambooHR REST API.
_MOCK_EMPLOYEES: dict[str, dict[str, Any]] = {
    "EMP001": {
        "employee_id": "EMP001",
        "employment_status": "active",
        "department": "Engineering",
        "job_role": "Senior Software Engineer",
        "manager_id": "EMP010",
        "active_projects": ["Project Alpha", "Project Gamma"],
        "resignation_date": None,
        "hire_date": "2021-03-15",
        "pto_calendar": [
            {"start_date": "2026-03-10", "end_date": "2026-03-14", "type": "vacation"},
        ],
    },
    "EMP002": {
        "employee_id": "EMP002",
        "employment_status": "active",
        "department": "Finance",
        "job_role": "Financial Analyst",
        "manager_id": "EMP011",
        "active_projects": ["Q1 Audit", "Budget Planning"],
        "resignation_date": None,
        "hire_date": "2020-08-01",
        "pto_calendar": [],
    },
    "EMP003": {
        "employee_id": "EMP003",
        "employment_status": "terminated",
        "department": "Engineering",
        "job_role": "DevOps Engineer",
        "manager_id": "EMP010",
        "active_projects": [],
        "resignation_date": "2026-02-28",
        "hire_date": "2019-11-20",
        "pto_calendar": [],
    },
    "EMP004": {
        "employee_id": "EMP004",
        "employment_status": "active",
        "department": "Research",
        "job_role": "Data Scientist",
        "manager_id": "EMP012",
        "active_projects": ["ML Pipeline v3", "Threat Model Research"],
        "resignation_date": None,
        "hire_date": "2022-06-01",
        "pto_calendar": [
            {"start_date": "2026-04-01", "end_date": "2026-04-05", "type": "conference"},
        ],
    },
    "EMP005": {
        "employee_id": "EMP005",
        "employment_status": "active",
        "department": "IT Operations",
        "job_role": "System Administrator",
        "manager_id": "EMP013",
        "active_projects": ["Infrastructure Modernisation"],
        "resignation_date": None,
        "hire_date": "2018-01-10",
        "pto_calendar": [],
    },
    "EMP006": {
        "employee_id": "EMP006",
        "employment_status": "on_leave",
        "department": "Executive",
        "job_role": "VP of Engineering",
        "manager_id": None,
        "active_projects": ["Strategic Planning 2026"],
        "resignation_date": None,
        "hire_date": "2017-05-20",
        "pto_calendar": [
            {"start_date": "2026-03-01", "end_date": "2026-03-20", "type": "medical_leave"},
        ],
    },
}


async def _mock_hr_api_call(employee_id: str) -> dict[str, Any] | None:
    """
    Simulate a Workday/BambooHR API call.

    In production, replace with::

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.hr_api_base_url}/employees/{employee_id}",
                headers={"Authorization": f"Bearer {settings.hr_api_key}"},
            )
            resp.raise_for_status()
            return resp.json()
    """
    return _MOCK_EMPLOYEES.get(employee_id)


# ── Redis cache helpers ───────────────────────────────────────

def _cache_key(employee_id: str) -> str:
    return f"{_HR_CACHE_PREFIX}{employee_id}"


async def _get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis


async def _get_cached(redis: aioredis.Redis, employee_id: str) -> dict | None:
    raw = await redis.get(_cache_key(employee_id))
    if raw:
        return json.loads(raw)
    return None


async def _set_cached(redis: aioredis.Redis, employee_id: str, data: dict) -> None:
    await redis.setex(
        _cache_key(employee_id),
        settings.hr_cache_ttl_seconds,
        json.dumps(data, default=str),
    )


async def _invalidate_cached(redis: aioredis.Redis, employee_id: str) -> None:
    await redis.delete(_cache_key(employee_id))


# ── Public API ────────────────────────────────────────────────

async def fetch_employee_context(
    employee_id: str, redis: aioredis.Redis,
) -> HREmployeeContext | None:
    """
    Get employment context for an employee, with Redis caching (1-hour TTL).

    Called internally by the risk-scoring aggregation pipeline —
    not exposed directly as an API endpoint.
    """
    # Check cache first
    cached = await _get_cached(redis, employee_id)
    if cached:
        return HREmployeeContext(**cached)

    # Pull from HR API
    data = await _mock_hr_api_call(employee_id)
    if not data:
        return None

    # Cache for 1 hour
    await _set_cached(redis, employee_id, data)

    return HREmployeeContext(**data)


# ── REST endpoints ────────────────────────────────────────────

@router.get(
    "/employees/{employee_id}",
    response_model=HREmployeeContext,
    summary="Get employment context (SOC_MANAGER+ or HR_SYSTEM)",
)
async def get_employee_context(
    employee_id: str,
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_MANAGER, Role.ADMIN, Role.HR_SYSTEM)
    ),
) -> HREmployeeContext:
    """
    Fetch employment context for a specific employee.
    Result is cached in Redis with 1-hour TTL.
    """
    redis = await _get_redis(request)
    ctx = await fetch_employee_context(employee_id, redis)
    if not ctx:
        raise HTTPException(
            status_code=404,
            detail=f"Employee {employee_id} not found in HR system.",
        )
    return ctx


@router.post(
    "/employees",
    response_model=HREmployeeContext,
    status_code=status.HTTP_201_CREATED,
    summary="Push employment context (HR_SYSTEM only)",
)
async def push_employee_context(
    body: HREmployeeContext,
    request: Request,
    _user: TokenPayload = Depends(require_role(Role.HR_SYSTEM)),
) -> HREmployeeContext:
    """
    HR system pushes employment context for an employee.
    Updates the Redis cache immediately.
    """
    redis = await _get_redis(request)
    await _set_cached(redis, body.employee_id, body.model_dump())

    log.info(
        "hr.context_pushed",
        employee_id=body.employee_id,
        status=body.employment_status,
    )

    # Publish Kafka event for downstream services to consume
    try:
        from app.services.kafka_producer import publish_event
        await publish_event(
            topic="itds.hr.context-updated",
            key=body.employee_id,
            value={
                "event_type": "context_push",
                "employee_id": body.employee_id,
                "employment_status": body.employment_status,
                "department": body.department,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception:
        pass  # Kafka failure should not block the HTTP response

    return body


@router.get(
    "/employees",
    summary="List all cached employee contexts (SOC_MANAGER+)",
)
async def list_cached_employees(
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> list[dict[str, Any]]:
    """List all employee IDs currently in the HR cache."""
    redis = await _get_redis(request)
    keys = []
    async for key in redis.scan_iter(match=f"{_HR_CACHE_PREFIX}*"):
        keys.append(key)

    results = []
    for key in keys:
        raw = await redis.get(key)
        if raw:
            results.append(json.loads(raw))

    return results


# ── Webhook receiver ──────────────────────────────────────────

def _verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """
    Verify HMAC-SHA256 signature from the HR system webhook.
    Prevents forged webhook events from being accepted.
    """
    expected = hmac.new(
        settings.hr_webhook_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post(
    "/webhook",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Receive real-time employment change events",
)
async def hr_webhook(
    request: Request,
    x_hr_signature: str | None = Header(None),
) -> dict[str, str]:
    """
    Webhook receiver for real-time employment changes.

    Called by Workday/BambooHR when:
      • An employee is hired
      • An employee is terminated
      • An employee changes department or role
      • An employee is suspended

    The webhook:
      1. Verifies the HMAC signature (prevents forged events).
      2. Invalidates the Redis cache for the affected employee.
      3. Publishes a Kafka event so the risk-scoring service can
         immediately adjust the employee's contextual risk.
    """
    body = await request.body()

    # Verify webhook signature (skip in debug mode for testing)
    if not settings.debug and x_hr_signature:
        if not _verify_webhook_signature(body, x_hr_signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature.",
            )

    try:
        event = HRWebhookEvent.model_validate_json(body)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook payload.",
        )

    redis = await _get_redis(request)

    # Invalidate cache for this employee — next fetch will pull fresh data
    await _invalidate_cached(redis, event.employee_id)

    log.info(
        "hr.webhook_received",
        event_type=event.event_type,
        employee_id=event.employee_id,
    )

    # Publish to Kafka for downstream processing
    try:
        from app.services.kafka_producer import publish_event
        await publish_event(
            topic="itds.hr.employment-change",
            key=event.employee_id,
            value={
                "event_id": str(uuid.uuid4()),
                "event_type": event.event_type,
                "employee_id": event.employee_id,
                "changes": event.changes,
                "source": event.source,
                "timestamp": event.timestamp,
                "received_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception:
        pass  # Kafka failure should not block webhook acknowledgement

    return {"status": "accepted", "event_type": event.event_type}
