"""
Risk-scoring-service FastAPI application.

Orchestrates four core subsystems:
    1. **ML detector fan-out** — calls all 4 detectors via asyncio.gather
    2. **Context engine**      — YAML-rule-based HR-aware score adjustment
    3. **Threshold manager**   — PID-controller adaptive thresholds
    4. **Trend analyzer**      — 30-day TimescaleDB history + slow-burn detection

Start-up creates the async DB engine, initialises the context engine &
threshold manager, and mounts all route groups under ``/api/v1``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.context_engine import ContextEngine
from app.database import engine as db_engine
from app.models import Base
from app.thresholds import ThresholdManager

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # ── Startup ─────────────────────────────────────────────────
    logger.info("risk_scoring_starting", extra={"port": settings.PORT})

    # Create TimescaleDB tables (hypertable creation needs migration)
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")

    # Initialise shared singletons
    context_engine = ContextEngine()
    threshold_mgr = ThresholdManager()

    # Inject into route modules
    from app.routes import scoring as scoring_mod
    from app.routes import thresholds as threshold_mod

    scoring_mod.set_context_engine(context_engine)
    scoring_mod.set_threshold_manager(threshold_mgr)
    threshold_mod.set_threshold_manager(threshold_mgr)

    logger.info("risk_scoring_started")

    yield

    # ── Shutdown ────────────────────────────────────────────────
    await db_engine.dispose()
    logger.info("risk_scoring_stopped")


app = FastAPI(
    title="ITDS – Risk Scoring Service",
    version="2.0.0",
    description=(
        "Context-aware insider-threat risk scoring with adaptive "
        "PID-tuned thresholds, 4-detector ML ensemble, and 30-day "
        "slow-burn trend detection."
    ),
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ── Route registration ──────────────────────────────────────────────
from app.routes import scoring as scoring_routes     # noqa: E402
from app.routes import thresholds as threshold_routes # noqa: E402
from app.routes import trends as trend_routes         # noqa: E402

app.include_router(scoring_routes.router, prefix="/api/v1", tags=["risk"])
app.include_router(threshold_routes.router, prefix="/api/v1/thresholds", tags=["thresholds"])
app.include_router(trend_routes.router, prefix="/api/v1/trends", tags=["trends"])


@app.get("/healthz")
async def health():
    return {"status": "ok", "service": settings.SERVICE_NAME}
