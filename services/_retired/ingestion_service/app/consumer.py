"""
Kafka multi-topic consumer for the ingestion service.

Subscribes to:
  • login-events
  • file-access-events
  • network-events
  • app-events

Each incoming record is parsed into the appropriate Pydantic schema,
persisted to PostgreSQL, fed through the BaselineEngine, and – if the
baseline flags an anomaly – re-published to ``itds.raw-events`` with
an anomaly annotation so downstream services can react.

The consumer runs as a background asyncio task started during the
FastAPI lifespan.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from aiokafka import AIOKafkaConsumer

from app.baseline import BaselineEngine
from app.config import get_settings
from app.database import get_session
from app.kafka import publish_event
from app.models import UserEvent
from app.schemas import (
    AppUsageEvent,
    FileAccessEvent,
    LoginEvent,
    LogoutEvent,
    NetworkRequestEvent,
)

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)

# ── Topic → schema mapping ──────────────────────────────────────────

TOPIC_SCHEMA_MAP: dict[str, type] = {
    "login-events":       LoginEvent,
    "file-access-events": FileAccessEvent,
    "network-events":     NetworkRequestEvent,
    "app-events":         AppUsageEvent,
}

CONSUME_TOPICS = list(TOPIC_SCHEMA_MAP.keys())


# ── Metric extraction per event type ────────────────────────────────

def _extract_metrics(event) -> list[tuple[str, float]]:
    """
    Return ``(metric_name, value)`` pairs that should be fed into the
    BaselineEngine for the given event.
    """
    metrics: list[tuple[str, float]] = []

    if isinstance(event, LoginEvent):
        metrics.append(("login_count", 1.0))
        hour = event.timestamp.hour
        is_off_hours = 1.0 if hour < 6 or hour >= 22 else 0.0
        metrics.append(("off_hours_login", is_off_hours))

    elif isinstance(event, LogoutEvent):
        metrics.append(("session_duration", event.session_duration_seconds))

    elif isinstance(event, FileAccessEvent):
        metrics.append(("file_access_count", 1.0))
        metrics.append(("file_bytes_total", float(event.file_size_bytes)))
        if event.sensitivity_label in ("confidential", "restricted"):
            metrics.append(("sensitive_file_access", 1.0))

    elif isinstance(event, NetworkRequestEvent):
        total_bytes = float(event.bytes_sent + event.bytes_received)
        metrics.append(("network_bytes_total", total_bytes))
        metrics.append(("network_request_count", 1.0))

    elif isinstance(event, AppUsageEvent):
        metrics.append(("app_duration_total", event.duration_seconds))

    return metrics


def _event_to_row(event) -> dict:
    """Map a Pydantic event to the columns of the ``user_events`` table."""
    base = {
        "user_id": event.user_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
    }

    if isinstance(event, LoginEvent):
        base["source_ip"] = event.source_ip
        base["action"] = event.auth_method
        base["metadata_"] = {"user_agent": event.user_agent, "geo": event.geo_location}

    elif isinstance(event, LogoutEvent):
        base["source_ip"] = event.source_ip
        base["duration_seconds"] = event.session_duration_seconds

    elif isinstance(event, FileAccessEvent):
        base["file_path"] = event.file_path
        base["file_size_bytes"] = event.file_size_bytes
        base["action"] = event.action.value
        base["sensitivity_label"] = event.sensitivity_label
        base["metadata_"] = {"source_device": event.source_device}

    elif isinstance(event, NetworkRequestEvent):
        base["source_ip"] = event.source_ip
        base["destination_ip"] = event.destination_ip
        base["destination_port"] = event.destination_port
        base["bytes_transferred"] = event.bytes_sent + event.bytes_received
        base["metadata_"] = {
            "protocol": event.protocol,
            "domain": event.domain,
            "encrypted": event.is_encrypted,
        }

    elif isinstance(event, AppUsageEvent):
        base["application_name"] = event.application_name
        base["duration_seconds"] = event.duration_seconds
        base["metadata_"] = {
            "window_title": event.window_title,
            "category": event.category,
            "foreground": event.foreground,
        }

    return base


# ── Consumer loop ───────────────────────────────────────────────────

async def start_consumer(baseline_engine: BaselineEngine) -> None:
    """
    Continuously consume from the four event topics, persist to PG,
    update baselines, and flag anomalies.
    """
    consumer = AIOKafkaConsumer(
        *CONSUME_TOPICS,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id=settings.KAFKA_CONSUMER_GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode()),
    )

    await consumer.start()
    logger.info(
        "kafka_consumer_started",
        extra={"topics": CONSUME_TOPICS, "group": settings.KAFKA_CONSUMER_GROUP},
    )

    try:
        async for msg in consumer:
            try:
                await _process_message(msg, baseline_engine)
            except Exception:
                logger.exception(
                    "consumer_message_error",
                    extra={"topic": msg.topic, "offset": msg.offset},
                )
    finally:
        await consumer.stop()
        logger.info("kafka_consumer_stopped")


async def _process_message(msg, baseline_engine: BaselineEngine) -> None:
    """Parse, persist, baseline-check, and optionally re-publish one message."""
    schema_cls = TOPIC_SCHEMA_MAP.get(msg.topic)
    if schema_cls is None:
        logger.warning("unknown_topic", extra={"topic": msg.topic})
        return

    event = schema_cls.model_validate(msg.value)
    logger.debug(
        "event_consumed",
        extra={"topic": msg.topic, "user": event.user_id, "type": event.event_type},
    )

    # 1) Persist to PostgreSQL
    row_data = _event_to_row(event)
    async with get_session() as session:
        session.add(UserEvent(**row_data))

    # 2) Update baselines + anomaly checks
    metrics = _extract_metrics(event)
    anomalies: list[dict] = []
    for metric_name, value in metrics:
        await baseline_engine.update(
            event.user_id, metric_name, value, event.timestamp,
        )
        check = await baseline_engine.check_anomaly(
            event.user_id, metric_name, value,
        )
        if check["is_anomalous"]:
            anomalies.append(check)

    # 3) If anomalous, publish enriched event to itds.raw-events
    if anomalies:
        enriched_payload = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "user_id": event.user_id,
            "timestamp": event.timestamp.isoformat(),
            "anomalies": anomalies,
            "original_event": msg.value,
        }
        await publish_event(
            settings.KAFKA_RAW_EVENTS_TOPIC,
            key=event.user_id,
            payload=enriched_payload,
        )
        logger.info(
            "anomaly_detected",
            extra={
                "user_id": event.user_id,
                "anomaly_count": len(anomalies),
                "metrics": [a["metric"] for a in anomalies],
            },
        )
