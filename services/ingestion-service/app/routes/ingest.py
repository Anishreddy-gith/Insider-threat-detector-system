"""
Direct HTTP ingestion route – fallback when Kafka producers aren't available.

POST /ingest/event   – accept a single event via the discriminated
                       ``IngestEvent`` union and process it identically
                       to the Kafka consumer path (persist → baseline → publish).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.baseline import BaselineEngine
from app.config import get_settings
from app.consumer import _event_to_row, _extract_metrics
from app.database import get_session
from app.kafka import publish_event
from app.models import UserEvent
from app.schemas import IngestEvent, IngestResponse

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)

router = APIRouter()

# The BaselineEngine instance is injected at app startup (see main.py)
_baseline_engine: BaselineEngine | None = None


def set_baseline_engine(engine: BaselineEngine) -> None:
    global _baseline_engine
    _baseline_engine = engine


@router.post(
    "/event",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a single event via HTTP",
    description=(
        "Fallback endpoint for direct HTTP ingestion. Accepts any "
        "event type (login, logout, file_access, network_request, app_usage) "
        "via the discriminated union schema. Events are persisted, "
        "baseline-checked, and published to Kafka."
    ),
)
async def ingest_event(event: IngestEvent):
    """
    Process a single event through the full ingestion pipeline:
      1. Validate (Pydantic strict mode)
      2. Persist to PostgreSQL
      3. Update rolling baseline (Welford)
      4. Check for anomalies (z-score ≥ 3σ)
      5. Publish to Kafka
    """
    if _baseline_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Baseline engine not initialised",
        )

    # 1) Persist
    row_data = _event_to_row(event)
    async with get_session() as session:
        session.add(UserEvent(**row_data))

    # 2) Baseline + anomaly detection
    metrics = _extract_metrics(event)
    anomalies: list[dict] = []
    for metric_name, value in metrics:
        await _baseline_engine.update(
            event.user_id, metric_name, value, event.timestamp,
        )
        check = await _baseline_engine.check_anomaly(
            event.user_id, metric_name, value,
        )
        if check["is_anomalous"]:
            anomalies.append(check)

    # 3) Publish to Kafka (enriched if anomalous)
    payload = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "user_id": event.user_id,
        "timestamp": event.timestamp.isoformat(),
        "data": event.model_dump(mode="json"),
    }
    if anomalies:
        payload["anomalies"] = anomalies

    await publish_event(
        settings.KAFKA_RAW_EVENTS_TOPIC,
        key=event.user_id,
        payload=payload,
    )

    logger.info(
        "http_event_ingested",
        extra={
            "user_id": event.user_id,
            "event_type": event.event_type,
            "anomalous": len(anomalies) > 0,
        },
    )

    return IngestResponse(
        event_id=event.event_id,
        event_type=event.event_type,
    )
