"""
Analytics Routes — Dashboard Aggregation + Entity Proxy
========================================================
Implements the **request aggregation** pattern for the dashboard:
a single ``GET /analytics/dashboard`` fans out to three downstream
services in parallel and merges the results.

Endpoints:
  GET /analytics/dashboard              — Aggregated SOC dashboard
  GET /analytics/entities               — Local entity list (DB)
  GET /analytics/entities/{entity_id}   — Entity detail (DB + risk proxy)
  GET /analytics/model-health           — Proxy to ML engine / alert FP/FN
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.api_gateway.app.auth_components.jwt_handler import Role
from services.api_gateway.app.auth_components.rbac import TokenPayload, require_role
from services.api_gateway.app.models.database import Alert, MonitoredEntity, get_db
from services.api_gateway.app.models.schemas import (
    DashboardMetrics,
    EntityDetailResponse,
    EntityResponse,
    PaginatedResponse,
    RiskDistribution,
)
from services.api_gateway.app.services.service_client import aggregate_dashboard, proxy_get

try:
    from shared.utils.logging import get_logger
except ImportError:
    import logging as _logging
    get_logger = _logging.getLogger

router = APIRouter(prefix="/analytics", tags=["analytics"])
log = get_logger(__name__)

DASHBOARD_CACHE_TTL = 30  # seconds


# ── Dashboard (aggregated) ────────────────────────────────────

@router.get(
    "/dashboard",
    summary="Aggregated SOC dashboard (single call → 3 services)",
)
async def get_dashboard(
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    **Request aggregation endpoint.**

    The frontend calls this *once* and receives data from:
      1. Risk-scoring service (current adaptive thresholds)
      2. Alert service (stats + recent open alerts)
      3. Ingestion service (health status)
      4. Local database (entity counts, risk distribution)

    The result is cached in Redis for 30 s to prevent N analysts
    refreshing simultaneously from hammering the backend.
    """
    redis = request.app.state.redis
    request_id = request.headers.get("X-Request-ID")

    # Check Redis cache first
    cached = await redis.get("dashboard:aggregated")
    if cached:
        return json.loads(cached)

    # ── Local DB metrics (always available) ───────────────────
    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(hours=24)

    total_entities = (await db.execute(
        select(func.count(MonitoredEntity.id)).where(
            MonitoredEntity.is_active == True  # noqa: E712
        )
    )).scalar() or 0

    active_alerts = (await db.execute(
        select(func.count(Alert.id)).where(
            Alert.is_resolved == False  # noqa: E712
        )
    )).scalar() or 0

    critical_alerts = (await db.execute(
        select(func.count(Alert.id)).where(
            Alert.is_resolved == False,  # noqa: E712
            Alert.severity == "critical",
        )
    )).scalar() or 0

    avg_risk = (await db.execute(
        select(func.avg(MonitoredEntity.risk_score)).where(
            MonitoredEntity.is_active == True  # noqa: E712
        )
    )).scalar() or 0.0

    alerts_24h = (await db.execute(
        select(func.count(Alert.id)).where(Alert.created_at >= day_ago)
    )).scalar() or 0

    # Risk distribution
    dist: dict[str, int] = {}
    for level in ("low", "medium", "high", "critical"):
        count = (await db.execute(
            select(func.count(MonitoredEntity.id)).where(
                MonitoredEntity.risk_level == level,
                MonitoredEntity.is_active == True,  # noqa: E712
            )
        )).scalar() or 0
        dist[level] = count

    # Top risk entities
    top_entities = (await db.execute(
        select(MonitoredEntity)
        .where(MonitoredEntity.is_active == True)  # noqa: E712
        .order_by(MonitoredEntity.risk_score.desc())
        .limit(10)
    )).scalars().all()

    # ── Downstream aggregation (parallel fan-out) ─────────────
    downstream = await aggregate_dashboard(request_id=request_id)

    # ── Merge local + downstream ──────────────────────────────
    result = {
        "total_entities": total_entities,
        "active_alerts": active_alerts,
        "critical_alerts": critical_alerts,
        "average_risk_score": round(float(avg_risk), 2),
        "risk_distribution": dist,
        "alerts_last_24h": alerts_24h,
        "top_risk_entities": [
            EntityResponse.model_validate(e).model_dump(mode="json")
            for e in top_entities
        ],
        # Downstream data (may be partially null if a service is down)
        "thresholds": downstream.get("thresholds"),
        "alert_stats": downstream.get("alert_stats"),
        "recent_alerts": downstream.get("recent_alerts"),
        "ingestion_status": downstream.get("ingestion_status"),
    }

    # Cache for 30 s
    await redis.setex("dashboard:aggregated", DASHBOARD_CACHE_TTL, json.dumps(result, default=str))

    return result


# ── Entity list (local DB) ────────────────────────────────────

@router.get("/entities", response_model=PaginatedResponse)
async def list_entities(
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
    risk_level: str | None = Query(None, pattern=r"^(low|medium|high|critical)$"),
    department: str | None = None,
    search: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse:
    query = select(MonitoredEntity).where(
        MonitoredEntity.is_active == True  # noqa: E712
    )
    count_q = select(func.count(MonitoredEntity.id)).where(
        MonitoredEntity.is_active == True  # noqa: E712
    )

    if risk_level:
        query = query.where(MonitoredEntity.risk_level == risk_level)
        count_q = count_q.where(MonitoredEntity.risk_level == risk_level)
    if department:
        query = query.where(MonitoredEntity.department == department)
        count_q = count_q.where(MonitoredEntity.department == department)
    if search:
        like = f"%{search}%"
        query = query.where(
            MonitoredEntity.display_name.ilike(like)
            | MonitoredEntity.employee_id.ilike(like)
        )
        count_q = count_q.where(
            MonitoredEntity.display_name.ilike(like)
            | MonitoredEntity.employee_id.ilike(like)
        )

    total = (await db.execute(count_q)).scalar() or 0
    offset = (page - 1) * page_size
    entities = (await db.execute(
        query.order_by(MonitoredEntity.risk_score.desc())
        .offset(offset)
        .limit(page_size)
    )).scalars().all()

    return PaginatedResponse(
        items=[EntityResponse.model_validate(e).model_dump(mode="json") for e in entities],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


# ── Entity detail ─────────────────────────────────────────────

@router.get("/entities/{entity_id}", response_model=EntityDetailResponse)
async def get_entity_detail(
    entity_id: UUID,
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
    db: AsyncSession = Depends(get_db),
) -> EntityDetailResponse:
    entity = await db.get(MonitoredEntity, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found.")

    recent_alerts = (await db.execute(
        select(Alert)
        .where(Alert.entity_id == entity_id)
        .order_by(Alert.created_at.desc())
        .limit(20)
    )).scalars().all()

    from services.api_gateway.app.models.schemas import AlertResponse

    # Try to fetch risk trend from risk-scoring-service
    risk_history: list[dict[str, Any]] = []
    try:
        request_id = request.headers.get("X-Request-ID")
        trend_data = await proxy_get(
            "risk_scoring",
            f"/api/v1/trends/{entity.employee_id}/history",
            request_id=request_id,
        )
        if isinstance(trend_data, list):
            risk_history = trend_data
        elif isinstance(trend_data, dict) and "scores" in trend_data:
            risk_history = trend_data["scores"]
    except Exception:
        pass  # Graceful degradation — trend is optional

    return EntityDetailResponse(
        **EntityResponse.model_validate(entity).model_dump(),
        recent_alerts=[AlertResponse.model_validate(a) for a in recent_alerts],
        risk_history=risk_history,
    )


# ── Model health (proxy) ─────────────────────────────────────

@router.get(
    "/model-health",
    summary="ML model health metrics (SOC_MANAGER+)",
)
async def model_health(
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    """
    Aggregates model health from:
      1. Alert service → FP/FN analysis report
      2. Risk-scoring service → current thresholds + PID history
    """
    from services.api_gateway.app.services.service_client import _safe_get_factory
    import asyncio

    request_id = request.headers.get("X-Request-ID")

    async def _safe(service: str, path: str) -> Any:
        try:
            return await proxy_get(service, path, request_id=request_id)
        except Exception:
            return None

    fpfn_report, thresholds, pid_history = await asyncio.gather(
        _safe("alert", "/api/v1/analysis/latest"),
        _safe("risk_scoring", "/api/v1/thresholds/current"),
        _safe("risk_scoring", "/api/v1/thresholds/history"),
    )

    return {
        "fpfn_report": fpfn_report,
        "thresholds": thresholds,
        "pid_history": pid_history,
    }


# ── Risk scoring proxy ───────────────────────────────────────

@router.get(
    "/thresholds",
    summary="Current adaptive thresholds (SOC_MANAGER+)",
)
async def get_thresholds(
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    from httpx import HTTPStatusError
    try:
        return await proxy_get(
            "risk_scoring", "/api/v1/thresholds/current",
            request_id=request.headers.get("X-Request-ID"),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text,
        )


@router.post(
    "/thresholds/update",
    summary="Feed FP data into PID controller (SOC_MANAGER+)",
)
async def update_thresholds(
    body: dict,
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    from httpx import HTTPStatusError
    from services.api_gateway.app.services.service_client import proxy_post
    try:
        return await proxy_post(
            "risk_scoring", "/api/v1/thresholds/update",
            json_body=body, request_id=request.headers.get("X-Request-ID"),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text,
        )


@router.get(
    "/trends/{entity_id}",
    summary="30-day risk trend for entity",
)
async def get_risk_trend(
    entity_id: str,
    request: Request,
    days: int = Query(30, ge=1, le=365),
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    from httpx import HTTPStatusError
    try:
        return await proxy_get(
            "risk_scoring", f"/api/v1/trends/{entity_id}",
            params={"days": days},
            request_id=request.headers.get("X-Request-ID"),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text,
        )

