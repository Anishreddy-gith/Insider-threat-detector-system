"""
Data Ingestion Service — Main
==============================
Receives raw telemetry from endpoint agents, SIEM connectors, and network
taps, normalises it, and publishes canonical events to Kafka.

Architecture notes:
  • This service is the **only** write path into Kafka for raw events.
  • It validates schemas before publishing (schema-on-write) so downstream
    consumers can trust the data shape.
  • Batch ingestion endpoint for bulk historical imports.
  • Individual event endpoint for real-time streaming.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.collectors.syslog_collector import router as syslog_router
from app.collectors.endpoint_collector import router as endpoint_router
from app.collectors.network_collector import router as network_router
from app.processors.normalizer import EventNormalizer
from shared.utils.logging import configure_logging, get_logger

log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(level=settings.log_level, json_output=(settings.log_format == "json"))
    log.info("data_ingestion.starting")

    # Initialise the normalizer (pre-compile regex patterns, load lookup tables)
    app.state.normalizer = EventNormalizer()

    yield

    log.info("data_ingestion.stopped")


app = FastAPI(
    title="ITDS — Data Ingestion Service",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

Instrumentator(excluded_handlers=["/health"]).instrument(app).expose(app, endpoint="/metrics")

app.include_router(syslog_router, prefix="/api/v1/ingest")
app.include_router(endpoint_router, prefix="/api/v1/ingest")
app.include_router(network_router, prefix="/api/v1/ingest")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
