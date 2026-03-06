"""
Alert-service FastAPI application.

Manages the full alert lifecycle:
    1. **Alert Engine**     — Kafka consumer → dedup → structured alerts
    2. **XAI Reports**      — auto-generated explainability reports
    3. **Multi-channel**    — Slack, Email, SIEM notifications
    4. **Feedback Loop**    — analyst verdicts → ensemble retraining
    5. **FP/FN Analyzer**   — weekly precision/recall/F1 reports

Start-up creates PostgreSQL tables, launches the Kafka consumer as a
background task, and mounts all route groups under ``/api/v1``.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.database import engine as db_engine
from app.models import Base

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)

_kafka_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _kafka_task

    # ── Startup ─────────────────────────────────────────────────
    logger.info("alert_service_starting", extra={"port": settings.PORT})

    # Create PostgreSQL tables
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")

    # Launch Kafka consumer as background task
    from app.alert_engine import kafka_consumer_loop
    _kafka_task = asyncio.create_task(kafka_consumer_loop())

    logger.info("alert_service_started")
    yield

    # ── Shutdown ────────────────────────────────────────────────
    if _kafka_task and not _kafka_task.done():
        _kafka_task.cancel()
        try:
            await _kafka_task
        except asyncio.CancelledError:
            pass

    await db_engine.dispose()
    logger.info("alert_service_stopped")


app = FastAPI(
    title="ITDS – Alert Service",
    version="2.0.0",
    description=(
        "Alert lifecycle management with Kafka-driven ingestion, "
        "deduplication, XAI reporting, multi-channel notifications, "
        "analyst feedback loop, and FP/FN analysis."
    ),
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ── Route registration ──────────────────────────────────────────────
from app.routes import alerts as alert_routes       # noqa: E402
from app.routes import feedback as feedback_routes   # noqa: E402
from app.routes import analysis as analysis_routes   # noqa: E402

app.include_router(alert_routes.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(feedback_routes.router, prefix="/api/v1/feedback", tags=["feedback"])
app.include_router(analysis_routes.router, prefix="/api/v1/analysis", tags=["analysis"])


@app.get("/healthz")
async def health():
    return {"status": "ok", "service": settings.SERVICE_NAME}
