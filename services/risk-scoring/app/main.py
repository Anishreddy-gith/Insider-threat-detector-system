"""
Risk Scoring Service — Main
=============================
Consumes anomaly detections from Kafka, computes composite risk scores,
manages risk decay, and emits alerts when thresholds are breached.

Risk Model:
  • **Exponential decay**: Risk scores decay towards 0 over time if no
    new anomalies are detected.  This prevents stale alerts from
    permanently stigmatising reformed users.
  • **Additive anomaly impact**: Each new anomaly increases the risk score
    proportionally to its severity × model confidence.
  • **Alert deduplication**: Cooldown period prevents alert fatigue from
    rapid re-scoring of the same user.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.scoring.risk_engine import RiskEngine
from shared.utils.logging import configure_logging, get_logger

log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(level=settings.log_level, json_output=(settings.log_format == "json"))
    log.info("risk_scoring.starting")

    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    await app.state.redis.ping()

    app.state.risk_engine = RiskEngine(
        redis_client=app.state.redis,
        decay_hours=settings.risk_score_decay_hours,
        alert_cooldown_minutes=settings.alert_cooldown_minutes,
    )

    yield

    await app.state.redis.aclose()
    log.info("risk_scoring.stopped")


app = FastAPI(
    title="ITDS — Risk Scoring Service",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator(excluded_handlers=["/health"]).instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/risk/{user_id}")
async def get_risk_score(user_id: str) -> dict:
    """Get current risk score for a user."""
    engine: RiskEngine = app.state.risk_engine
    return await engine.get_current_risk(user_id)
