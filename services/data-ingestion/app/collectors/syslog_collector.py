"""
Syslog Collector
=================
Accepts syslog-formatted events via REST (acts as an HTTP syslog relay).

Real deployments would also have a raw UDP/TCP syslog listener, but the
REST endpoint is easier to integrate with cloud-native SIEM connectors
(Splunk HEC, Elastic Agent, CrowdStrike Falcon).
"""

from __future__ import annotations

import json
from typing import Any

from aiokafka import AIOKafkaProducer
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from shared.constants import TOPIC_AUTH_EVENT, TOPIC_USER_ACTIVITY
from shared.models.events import EventEnvelope, EventType, UserActivityPayload
from shared.utils.logging import get_logger

router = APIRouter(tags=["syslog"])
log = get_logger(__name__)
settings = get_settings()

_producer: AIOKafkaProducer | None = None


async def _get_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is None:
        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            enable_idempotence=True,
            compression_type="lz4",
        )
        await _producer.start()
    return _producer


class SyslogEvent(BaseModel):
    """Raw syslog-shaped event from a SIEM connector."""
    facility: int = Field(ge=0, le=23)
    severity: int = Field(ge=0, le=7)
    hostname: str
    app_name: str
    proc_id: str | None = None
    msg_id: str | None = None
    structured_data: dict[str, Any] = Field(default_factory=dict)
    message: str


class SyslogBatch(BaseModel):
    events: list[SyslogEvent] = Field(max_length=5000)


@router.post("/syslog", status_code=202)
async def ingest_syslog(event: SyslogEvent, request: Request) -> dict[str, str]:
    """
    Ingest a single syslog event.

    The normalizer converts it to a canonical ``UserActivityPayload``
    before publishing to Kafka.
    """
    normalizer = request.app.state.normalizer
    normalised = normalizer.normalise_syslog(event.model_dump())

    envelope = EventEnvelope(
        event_type=EventType.USER_ACTIVITY,
        source_service="data-ingestion",
        payload=normalised,
    )

    producer = await _get_producer()
    await producer.send_and_wait(
        TOPIC_USER_ACTIVITY,
        value=envelope.model_dump(mode="json"),
        key=normalised.get("user_id", "unknown").encode("utf-8"),
    )

    log.debug("syslog.ingested", hostname=event.hostname, app=event.app_name)
    return {"status": "accepted"}


@router.post("/syslog/batch", status_code=202)
async def ingest_syslog_batch(batch: SyslogBatch, request: Request) -> dict[str, Any]:
    """
    Bulk ingest — used for historical backfills from SIEM exports.

    WHY batch endpoint?
      Reduces HTTP overhead for large imports.  Each event is still
      individually published to Kafka (preserving per-user partitioning).
    """
    normalizer = request.app.state.normalizer
    producer = await _get_producer()
    ingested = 0

    for event in batch.events:
        normalised = normalizer.normalise_syslog(event.model_dump())
        envelope = EventEnvelope(
            event_type=EventType.USER_ACTIVITY,
            source_service="data-ingestion",
            payload=normalised,
        )
        await producer.send(
            TOPIC_USER_ACTIVITY,
            value=envelope.model_dump(mode="json"),
            key=normalised.get("user_id", "unknown").encode("utf-8"),
        )
        ingested += 1

    await producer.flush()
    log.info("syslog.batch_ingested", count=ingested)
    return {"status": "accepted", "ingested": ingested}
