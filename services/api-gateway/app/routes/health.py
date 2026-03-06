"""
Health-check & Readiness Probes
================================
Kubernetes liveness / readiness + aggregated downstream health.

``/health``       → lightweight liveness (is the gateway process alive?)
``/health/ready`` → deep readiness (PG + Redis + all downstream services)
"""

from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from app.models.database import engine

try:
    from shared.utils.logging import get_logger
except ImportError:
    import logging as _logging
    get_logger = _logging.getLogger

log = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    return {"status": "ok", "service": "api-gateway"}


@router.get("/health/ready", summary="Readiness probe (checks all dependencies)")
async def readiness(request: Request) -> JSONResponse:
    """
    Verify that PostgreSQL, Redis, and all downstream services are
    reachable.  Returns 200 if all pass, 503 if any fail.
    """
    checks: dict[str, str] = {}

    # PostgreSQL
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        log.error("health.postgres_failed", error=str(exc))
        checks["postgres"] = "unavailable"

    # Redis
    try:
        redis = request.app.state.redis
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        log.error("health.redis_failed", error=str(exc))
        checks["redis"] = "unavailable"

    # Downstream services (fire in parallel, 3 s timeout each)
    from app.config import get_settings
    settings = get_settings()

    service_urls = {
        "ingestion": f"{settings.ingestion_service_url}/healthz",
        "risk_scoring": f"{settings.risk_scoring_service_url}/healthz",
        "alert_service": f"{settings.alert_service_url}/healthz",
        "ml_engine": f"{settings.ml_engine_url}/healthz",
    }

    async def _check_service(name: str, url: str) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                resp = await c.get(url)
                return name, "ok" if resp.status_code == 200 else "degraded"
        except Exception:
            return name, "unavailable"

    results = await asyncio.gather(
        *[_check_service(n, u) for n, u in service_urls.items()]
    )
    for name, status in results:
        checks[name] = status

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ready" if all_ok else "degraded",
            "checks": checks,
        },
    )
