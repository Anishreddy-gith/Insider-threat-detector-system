"""
Alert Engine — Kafka consumer + deduplication + lifecycle management.

Consumes risk-score events from ``itds.risk-scores``. For scores > 0.7,
generates a structured alert with deduplication: the same (user_id,
anomaly_type) within a 1-hour window produces a single alert whose
score is updated to the maximum observed value.

Alert lifecycle:
    NEW → INVESTIGATING → RESOLVED | FALSE_POSITIVE

The engine also triggers XAI report generation and multi-channel
notification for every HIGH / CRITICAL alert.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session
from app.models import Alert
from app.schemas import AlertStatus, RiskScoreEvent

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Deduplication ───────────────────────────────────────────────────

def _dedup_key(user_id: str, anomaly_type: str) -> str:
    """
    Generate a deterministic dedup key from user + anomaly type.

    The key is checked against existing alerts within the dedup window;
    if a matching NEW/INVESTIGATING alert exists, we update it rather
    than creating a duplicate.
    """
    raw = f"{user_id}::{anomaly_type}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _derive_anomaly_type(event: RiskScoreEvent) -> str:
    """
    Derive a human-readable anomaly type from detector breakdown.

    Picks the detector with the highest anomaly score as the primary
    anomaly type, then appends the top feature if available.
    """
    if not event.detector_breakdown:
        return "unknown_anomaly"

    top_detector = max(
        event.detector_breakdown,
        key=lambda d: d.anomaly_score,
    )
    base = top_detector.detector

    # Enrich with top feature
    if event.top_anomalous_features:
        top_feat = event.top_anomalous_features[0].get("feature", "")
        if top_feat:
            return f"{base}:{top_feat}"
    return base


def _build_summary(event: RiskScoreEvent, anomaly_type: str) -> str:
    """Build a one-line alert summary."""
    return (
        f"[{event.risk_level}] User {event.user_id} — "
        f"risk score {event.overall_score:.2f} "
        f"(anomaly: {anomaly_type})"
    )


# ── Core alert creation / dedup ─────────────────────────────────────

async def process_risk_score(
    event: RiskScoreEvent,
    session: AsyncSession,
) -> Alert | None:
    """
    Process a single risk-score event.

    Returns the created/updated Alert, or None if below threshold.
    """
    if event.overall_score < settings.ALERT_SCORE_THRESHOLD:
        return None

    anomaly_type = _derive_anomaly_type(event)
    key = _dedup_key(event.user_id, anomaly_type)

    # Check for existing alert within dedup window
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=settings.DEDUP_WINDOW_SECONDS
    )

    result = await session.execute(
        select(Alert)
        .where(
            Alert.dedup_key == key,
            Alert.status.in_(["NEW", "INVESTIGATING"]),
            Alert.created_at >= cutoff,
        )
        .order_by(Alert.created_at.desc())
        .limit(1)
    )
    existing = result.scalar_one_or_none()

    if existing:
        # Update existing alert with higher score
        if event.overall_score > existing.risk_score:
            existing.risk_score = event.overall_score
            existing.risk_level = event.risk_level
            existing.summary = _build_summary(event, anomaly_type)
            existing.detector_breakdown = _detector_breakdown_dict(event)
            existing.top_features = event.top_anomalous_features
            existing.updated_at = datetime.now(timezone.utc)
            logger.info(
                "alert_dedup_updated",
                extra={
                    "alert_id": str(existing.id),
                    "user_id": event.user_id,
                    "new_score": event.overall_score,
                },
            )
        else:
            logger.info(
                "alert_dedup_skipped",
                extra={
                    "alert_id": str(existing.id),
                    "user_id": event.user_id,
                    "existing_score": existing.risk_score,
                },
            )
        return existing

    # Create new alert
    alert = Alert(
        user_id=event.user_id,
        risk_score=event.overall_score,
        risk_level=event.risk_level,
        anomaly_type=anomaly_type,
        summary=_build_summary(event, anomaly_type),
        status="NEW",
        detector_breakdown=_detector_breakdown_dict(event),
        top_features=event.top_anomalous_features,
        recommended_actions=event.recommended_actions,
        business_context=event.business_context,
        dedup_key=key,
    )
    session.add(alert)
    await session.flush()

    logger.info(
        "alert_created",
        extra={
            "alert_id": str(alert.id),
            "user_id": event.user_id,
            "risk_score": event.overall_score,
            "risk_level": event.risk_level,
            "anomaly_type": anomaly_type,
        },
    )
    return alert


def _detector_breakdown_dict(event: RiskScoreEvent) -> dict[str, Any]:
    """Flatten detector breakdown to a serialisable dict."""
    return {
        d.detector: {
            "anomaly_score": d.anomaly_score,
            "is_anomaly": d.is_anomaly,
            "detail": d.detail,
        }
        for d in event.detector_breakdown
    }


# ── Alert lifecycle transitions ─────────────────────────────────────

async def transition_status(
    alert_id: str,
    new_status: AlertStatus,
    session: AsyncSession,
    analyst: str | None = None,
    notes: str | None = None,
) -> Alert:
    """
    Transition an alert to a new lifecycle state.

    Valid transitions:
        NEW → INVESTIGATING
        INVESTIGATING → RESOLVED | FALSE_POSITIVE
        NEW → RESOLVED | FALSE_POSITIVE (analyst skip)
    """
    from uuid import UUID

    result = await session.execute(
        select(Alert).where(Alert.id == UUID(alert_id))
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise ValueError(f"Alert {alert_id} not found")

    valid_transitions = {
        "NEW": {"INVESTIGATING", "RESOLVED", "FALSE_POSITIVE"},
        "INVESTIGATING": {"RESOLVED", "FALSE_POSITIVE"},
    }
    allowed = valid_transitions.get(alert.status, set())
    if new_status.value not in allowed:
        raise ValueError(
            f"Cannot transition from {alert.status} to {new_status.value}"
        )

    alert.status = new_status.value
    if analyst:
        alert.assigned_to = analyst
    alert.updated_at = datetime.now(timezone.utc)

    logger.info(
        "alert_status_changed",
        extra={
            "alert_id": alert_id,
            "new_status": new_status.value,
            "analyst": analyst,
        },
    )
    return alert


# ── Kafka consumer loop ─────────────────────────────────────────────

async def kafka_consumer_loop() -> None:
    """
    Long-running Kafka consumer that processes risk-score events.

    Runs as a background task during the FastAPI lifespan.
    Imports aiokafka lazily so the service can start without Kafka
    during development.
    """
    try:
        from aiokafka import AIOKafkaConsumer
    except ImportError:
        logger.warning("aiokafka_not_installed — Kafka consumer disabled")
        return

    consumer = AIOKafkaConsumer(
        settings.KAFKA_RISK_SCORES_TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id=settings.KAFKA_CONSUMER_GROUP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    retry_delay = 5
    while True:
        try:
            await consumer.start()
            logger.info("kafka_consumer_started")
            break
        except Exception as exc:
            logger.warning(
                "kafka_connect_retry",
                extra={"error": str(exc), "retry_in": retry_delay},
            )
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

    try:
        async for msg in consumer:
            try:
                event = RiskScoreEvent(**msg.value)
                async with async_session() as session:
                    alert = await process_risk_score(event, session)
                    if alert and alert.risk_level in ("HIGH", "CRITICAL"):
                        # Generate XAI report + notify
                        from app.xai_reports import XAIReportGenerator
                        from app.notifiers.dispatcher import dispatch_notifications

                        xai = XAIReportGenerator()
                        report = xai.generate(event, str(alert.id))
                        alert.xai_report = report
                        await session.commit()
                        await dispatch_notifications(alert, report)
                    elif alert:
                        await session.commit()
            except Exception:
                logger.exception("kafka_message_processing_error")
    finally:
        await consumer.stop()
        logger.info("kafka_consumer_stopped")
