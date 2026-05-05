"""
Behavioral Analytics — Main
=============================
Consumes normalised events from Kafka, extracts features, runs ML models,
and publishes anomaly detections back to Kafka.

This service is the ML brain of the platform: it owns all model inference
and training orchestration.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.pipeline.inference import InferencePipeline
from shared.utils.logging import configure_logging, get_logger

log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(level=settings.log_level, json_output=(settings.log_format == "json"))
    log.info("behavioral_analytics.starting")

    # Redis for feature caching
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    await app.state.redis.ping()

    # Initialise the ML inference pipeline (loads models from registry)
    app.state.pipeline = InferencePipeline(
        model_path=settings.model_registry_path,
        anomaly_threshold=settings.anomaly_threshold,
        enable_gnn=settings.enable_gnn_model,
        enable_xai=settings.enable_explainability,
    )
    await app.state.pipeline.load_models()

    yield

    await app.state.redis.aclose()
    log.info("behavioral_analytics.stopped")


app = FastAPI(
    title="ITDS — Behavioral Analytics Service",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator(excluded_handlers=["/health"]).instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def list_models() -> dict:
    """List loaded models and their versions."""
    pipeline: InferencePipeline = app.state.pipeline
    return {
        "models": pipeline.get_model_info(),
        "anomaly_threshold": settings.anomaly_threshold,
    }
