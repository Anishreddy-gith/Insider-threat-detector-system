from __future__ import annotations

import importlib.util
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from kafka import KafkaProducer
from pydantic import BaseModel, Field

from services.api_gateway.app.rate_limit import RateLimitMiddleware
from services.api_gateway.app.redis_client import RiskRedisStore

TOPICS = {
    "user_activity": "user_activity",
    "processed_features": "processed_features",
    "anomaly_scores": "anomaly_scores",
    "risk_events": "risk_events",
}


def _json_serializer(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _load_jwt_auth_middleware():
    import pathlib

    auth_path = pathlib.Path(__file__).with_name("auth.py")
    spec = importlib.util.spec_from_file_location("gateway_auth_file", auth_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.JWTAuthMiddleware


JWTAuthMiddleware = _load_jwt_auth_middleware()


class IngestEvent(BaseModel):
    user_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    activity_type: str = Field(min_length=1)
    metadata: dict[str, Any]


class RiskResponse(BaseModel):
    user_id: str
    risk_level: str
    final_score: float


app = FastAPI(title="ITDS API Gateway", version="1.0.0")
app.add_middleware(JWTAuthMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "120")))


@app.on_event("startup")
def startup() -> None:
    app.state.producer = KafkaProducer(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        value_serializer=_json_serializer,
        linger_ms=10,
        acks="all",
    )
    app.state.redis_store = RiskRedisStore(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    app.state.risk_cache: dict[str, dict[str, Any]] = {}


@app.on_event("shutdown")
def shutdown() -> None:
    producer = getattr(app.state, "producer", None)
    if producer is not None:
        producer.flush()
        producer.close()


@app.post("/ingest")
def ingest(event: IngestEvent) -> dict[str, Any]:
    payload = {
        "user_id": event.user_id,
        "timestamp": event.timestamp,
        "activity_type": event.activity_type,
        "metadata": event.metadata,
    }
    app.state.producer.send(TOPICS["user_activity"], value=payload)
    app.state.producer.flush()
    return {"status": "accepted", "topic": TOPICS["user_activity"]}


@app.get("/risk/{user_id}", response_model=RiskResponse)
def get_risk(user_id: str) -> RiskResponse:
    user_id = str(user_id).strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid user_id")

    latest = app.state.redis_store.get_latest(user_id)
    if latest is None:
        latest = app.state.risk_cache.get(user_id)

    if latest is None:
        raise HTTPException(status_code=404, detail="Risk score not found")

    try:
        return RiskResponse(
            user_id=user_id,
            risk_level=str(latest["risk_level"]),
            final_score=float(latest["final_score"]),
        )
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=500, detail="Corrupt risk payload")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
