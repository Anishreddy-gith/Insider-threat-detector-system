"""
Ingestion-service FastAPI application.

Responsibilities
────────────────
1. Expose REST endpoints for syslog, EDR/DLP, network-flow, and
   generic event ingestion (``POST /ingest/event``)
2. Validate & normalise payloads using Pydantic v2 strict schemas
3. Persist events to PostgreSQL via SQLAlchemy 2.0 async ORM
4. Maintain 30-day rolling baselines per user (Welford's algorithm / Redis)
5. Consume from Kafka topics: ``login-events``, ``file-access-events``,
   ``network-events``, ``app-events``
6. Publish normalised + anomaly-enriched events to ``itds.raw-events``
7. Expose /healthz and /metrics for K8s probes & Prometheus scraping
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.baseline import BaselineEngine
from app.config import get_settings
from app.routes import collectors

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)

# Shared baseline engine instance
baseline_engine = BaselineEngine()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Lifecycle: start Kafka producer + consumer, DB engine, and
    the Redis-backed BaselineEngine.
    """
    from app.consumer import start_consumer
    from app.database import engine as db_engine
    from app.kafka import producer
    from app.models import Base
    from app.routes.ingest import set_baseline_engine

    # 1) Create tables (dev convenience — use Alembic in production)
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2) Start Kafka producer
    await producer.start()

    # 3) Start baseline engine (Redis)
    await baseline_engine.start()
    set_baseline_engine(baseline_engine)

    # 4) Launch Kafka consumer as a background task
    consumer_task = asyncio.create_task(start_consumer(baseline_engine))

    logger.info("ingestion_service_started", extra={"port": settings.PORT})
    yield

    # Teardown
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    await baseline_engine.stop()
    await producer.stop()
    await db_engine.dispose()
    logger.info("ingestion_service_stopped")


app = FastAPI(
    title="ITDS – Ingestion Service",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Legacy syslog / endpoint / network collectors
app.include_router(collectors.router, prefix="/api/v1/ingest", tags=["ingest"])

# New unified event endpoint
from app.routes.ingest import router as ingest_router  # noqa: E402
app.include_router(ingest_router, prefix="/ingest", tags=["ingest-v2"])


@app.get("/healthz")
async def health():
    return {"status": "ok", "service": settings.SERVICE_NAME}
