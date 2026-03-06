"""
Endpoint Collector
===================
Receives telemetry from host-based agents (EDR, DLP, UEBA agents).

Covers:
  • File access events (read, write, copy to USB, upload)
  • Process execution events
  • USB device insertions
  • Print jobs
"""

from __future__ import annotations

import json
from typing import Any

from aiokafka import AIOKafkaProducer
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from shared.constants import TOPIC_FILE_ACCESS, TOPIC_USER_ACTIVITY
from shared.models.events import EventEnvelope, EventType
from shared.utils.logging import get_logger

router = APIRouter(tags=["endpoint"])
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


class EndpointEvent(BaseModel):
    """Telemetry from a host-based agent."""
    agent_id: str
    user_id: str
    hostname: str
    event_type: str  # "file_access" | "process_exec" | "usb_insert" | "print_job"
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | None = None


class FileAccessEvent(BaseModel):
    """Specific schema for file-system access events."""
    user_id: str
    hostname: str
    file_path: str
    action: str  # "read" | "write" | "delete" | "rename" | "copy"
    file_size_bytes: int | None = None
    sensitivity_label: str | None = None
    destination: str | None = None  # non-null for copy/move operations
    timestamp: str | None = None


@router.post("/endpoint", status_code=202)
async def ingest_endpoint_event(event: EndpointEvent, request: Request) -> dict[str, str]:
    """Ingest a generic endpoint telemetry event."""
    normalizer = request.app.state.normalizer
    normalised = normalizer.normalise_endpoint(event.model_dump())

    envelope = EventEnvelope(
        event_type=EventType.USER_ACTIVITY,
        source_service="data-ingestion",
        payload=normalised,
    )

    producer = await _get_producer()
    await producer.send_and_wait(
        TOPIC_USER_ACTIVITY,
        value=envelope.model_dump(mode="json"),
        key=event.user_id.encode("utf-8"),
    )

    log.debug("endpoint.event.ingested", user=event.user_id, type=event.event_type)
    return {"status": "accepted"}


@router.post("/endpoint/file-access", status_code=202)
async def ingest_file_access(event: FileAccessEvent, request: Request) -> dict[str, str]:
    """
    Ingest a file-access event with full metadata.

    WHY a separate endpoint for file access?
      File access events carry sensitivity labels and destination info
      that are critical for detecting data exfiltration.  A dedicated
      schema ensures we never lose these fields during normalisation.
    """
    envelope = EventEnvelope(
        event_type=EventType.FILE_ACCESS,
        source_service="data-ingestion",
        payload=event.model_dump(),
    )

    producer = await _get_producer()
    await producer.send_and_wait(
        TOPIC_FILE_ACCESS,
        value=envelope.model_dump(mode="json"),
        key=event.user_id.encode("utf-8"),
    )

    log.debug("file_access.ingested", user=event.user_id, action=event.action)
    return {"status": "accepted"}
