"""
Ingestion collector routes.

Three collector families:
- **syslog**   – authentication logs, sudo events, SSH sessions
- **endpoint** – EDR/DLP agent telemetry (process spawns, USB, file access)
- **network**  – NetFlow / DNS / HTTP proxy logs

Each endpoint validates, normalises timestamps to UTC, wraps the payload
in an ``EventEnvelope``, and publishes to Kafka.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.kafka import publish_event

settings = get_settings()
router = APIRouter()


# ── Shared envelope ─────────────────────────────────────────────────────

class EventEnvelope(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    entity_id: str
    timestamp: datetime
    source: str
    data: dict


# ── Syslog ──────────────────────────────────────────────────────────────

class SyslogEvent(BaseModel):
    entity_id: str
    hostname: str
    facility: str
    severity: str
    message: str
    timestamp: datetime | None = None


@router.post("/syslog", status_code=status.HTTP_202_ACCEPTED)
async def ingest_syslog(event: SyslogEvent):
    envelope = EventEnvelope(
        event_type="syslog",
        entity_id=event.entity_id,
        timestamp=event.timestamp or datetime.now(timezone.utc),
        source=event.hostname,
        data=event.model_dump(),
    )
    await publish_event(
        settings.KAFKA_RAW_EVENTS_TOPIC,
        key=event.entity_id,
        payload=envelope.model_dump(),
    )
    return {"accepted": True, "event_id": envelope.event_id}


@router.post("/syslog/batch", status_code=status.HTTP_202_ACCEPTED)
async def ingest_syslog_batch(events: list[SyslogEvent]):
    if len(events) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch exceeds maximum of {settings.MAX_BATCH_SIZE} events",
        )
    ids = []
    for event in events:
        envelope = EventEnvelope(
            event_type="syslog",
            entity_id=event.entity_id,
            timestamp=event.timestamp or datetime.now(timezone.utc),
            source=event.hostname,
            data=event.model_dump(),
        )
        await publish_event(
            settings.KAFKA_RAW_EVENTS_TOPIC,
            key=event.entity_id,
            payload=envelope.model_dump(),
        )
        ids.append(envelope.event_id)
    return {"accepted": len(ids), "event_ids": ids}


# ── Endpoint / EDR ──────────────────────────────────────────────────────

class EndpointEvent(BaseModel):
    entity_id: str
    hostname: str
    event_subtype: str          # process_spawn | usb_mount | file_access | dlp_alert
    process_name: str | None = None
    file_path: str | None = None
    action: str | None = None
    sensitivity_label: str | None = None
    timestamp: datetime | None = None


@router.post("/endpoint", status_code=status.HTTP_202_ACCEPTED)
async def ingest_endpoint(event: EndpointEvent):
    envelope = EventEnvelope(
        event_type="endpoint",
        entity_id=event.entity_id,
        timestamp=event.timestamp or datetime.now(timezone.utc),
        source=event.hostname,
        data=event.model_dump(),
    )
    await publish_event(
        settings.KAFKA_RAW_EVENTS_TOPIC,
        key=event.entity_id,
        payload=envelope.model_dump(),
    )
    return {"accepted": True, "event_id": envelope.event_id}


# ── Network flow ────────────────────────────────────────────────────────

class NetworkEvent(BaseModel):
    entity_id: str
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str = "TCP"
    bytes_sent: int = 0
    bytes_received: int = 0
    domain: str | None = None
    timestamp: datetime | None = None


@router.post("/network", status_code=status.HTTP_202_ACCEPTED)
async def ingest_network(event: NetworkEvent):
    envelope = EventEnvelope(
        event_type="network",
        entity_id=event.entity_id,
        timestamp=event.timestamp or datetime.now(timezone.utc),
        source=event.src_ip,
        data=event.model_dump(),
    )
    await publish_event(
        settings.KAFKA_RAW_EVENTS_TOPIC,
        key=event.entity_id,
        payload=envelope.model_dump(),
    )
    return {"accepted": True, "event_id": envelope.event_id}


@router.post("/network/batch", status_code=status.HTTP_202_ACCEPTED)
async def ingest_network_batch(events: list[NetworkEvent]):
    if len(events) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch exceeds maximum of {settings.MAX_BATCH_SIZE} events",
        )
    ids = []
    for event in events:
        envelope = EventEnvelope(
            event_type="network",
            entity_id=event.entity_id,
            timestamp=event.timestamp or datetime.now(timezone.utc),
            source=event.src_ip,
            data=event.model_dump(),
        )
        await publish_event(
            settings.KAFKA_RAW_EVENTS_TOPIC,
            key=event.entity_id,
            payload=envelope.model_dump(),
        )
        ids.append(envelope.event_id)
    return {"accepted": len(ids), "event_ids": ids}
