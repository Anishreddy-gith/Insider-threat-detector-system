from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from kafka import KafkaProducer
from pydantic import BaseModel, Field

from services.api_gateway.app.auth import JWTAuthMiddleware
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


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _demo_risk_for(event: dict[str, Any]) -> dict[str, Any]:
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    activity_type = str(event.get("activity_type", "")).lower()
    score = 0.18

    if activity_type in {"file_download", "data_exfiltration", "privilege_escalation"}:
        score += 0.42
    if metadata.get("after_hours") is True:
        score += 0.18
    if metadata.get("failed_logins", 0) and int(metadata.get("failed_logins", 0)) >= 3:
        score += 0.14

    final_score = round(min(score, 0.99), 2)
    if final_score >= 0.75:
        risk_level = "critical"
    elif final_score >= 0.5:
        risk_level = "high"
    elif final_score >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "user_id": str(event["user_id"]),
        "risk_level": risk_level,
        "final_score": final_score,
        "updated_at": int(time.time()),
        "demo_mode": True,
    }


class InMemoryRiskStore:
    def __init__(self) -> None:
        self._values: dict[str, dict[str, Any]] = {}

    def get_latest(self, user_id: str) -> dict[str, Any] | None:
        return self._values.get(user_id)

    def set_latest(self, user_id: str, payload: dict[str, Any]) -> None:
        self._values[user_id] = payload


class DemoProducer:
    def __init__(self, risk_store: InMemoryRiskStore) -> None:
        self.risk_store = risk_store
        self.events: list[dict[str, Any]] = []

    def send(self, topic: str, value: dict[str, Any]) -> None:
        self.events.append({"topic": topic, "value": value})
        if topic == TOPICS["user_activity"]:
            self.risk_store.set_latest(str(value["user_id"]), _demo_risk_for(value))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    app.state.demo_mode = _env_flag("DEMO_MODE") or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip().lower() in {
        "",
        "demo",
        "mock",
        "memory",
    }
    if app.state.demo_mode:
        app.state.redis_store = InMemoryRiskStore()
        app.state.producer = DemoProducer(app.state.redis_store)
        app.state.risk_cache = app.state.redis_store._values
        return

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


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return health()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return "# HELP http_requests_total Total HTTP requests\n# TYPE http_requests_total counter\nhttp_requests_total 0\n"
