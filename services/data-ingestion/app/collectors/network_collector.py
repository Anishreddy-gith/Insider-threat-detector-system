"""
Network Collector
==================
Receives network flow metadata from DNS/proxy logs, NetFlow, or PCAP
summary feeds.

These events are critical for detecting:
  • Data exfiltration to external IPs / domains
  • C2 beaconing patterns (regular intervals to unusual destinations)
  • Shadow IT usage (connections to unsanctioned cloud services)
"""

from __future__ import annotations

import json
from typing import Any

from aiokafka import AIOKafkaProducer
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from shared.constants import TOPIC_NETWORK_EVENT
from shared.models.events import EventEnvelope, EventType
from shared.utils.logging import get_logger

router = APIRouter(tags=["network"])
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


class NetworkFlowEvent(BaseModel):
    """Network flow summary event."""
    user_id: str | None = None
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str = Field(pattern=r"^(TCP|UDP|ICMP)$")
    bytes_sent: int = 0
    bytes_received: int = 0
    domain: str | None = None
    is_encrypted: bool = True
    geo_country: str | None = None
    timestamp: str | None = None


class NetworkFlowBatch(BaseModel):
    flows: list[NetworkFlowEvent] = Field(max_length=10000)


@router.post("/network", status_code=202)
async def ingest_network_flow(flow: NetworkFlowEvent) -> dict[str, str]:
    """Ingest a single network flow event."""
    envelope = EventEnvelope(
        event_type=EventType.NETWORK_EVENT,
        source_service="data-ingestion",
        payload=flow.model_dump(),
    )

    producer = await _get_producer()
    key = (flow.user_id or flow.src_ip).encode("utf-8")
    await producer.send_and_wait(
        TOPIC_NETWORK_EVENT,
        value=envelope.model_dump(mode="json"),
        key=key,
    )

    log.debug("network.flow.ingested", src=flow.src_ip, dst=flow.dst_ip)
    return {"status": "accepted"}


@router.post("/network/batch", status_code=202)
async def ingest_network_batch(batch: NetworkFlowBatch) -> dict[str, Any]:
    """Bulk ingest network flows (e.g. from NetFlow export)."""
    producer = await _get_producer()
    count = 0

    for flow in batch.flows:
        envelope = EventEnvelope(
            event_type=EventType.NETWORK_EVENT,
            source_service="data-ingestion",
            payload=flow.model_dump(),
        )
        key = (flow.user_id or flow.src_ip).encode("utf-8")
        await producer.send(
            TOPIC_NETWORK_EVENT,
            value=envelope.model_dump(mode="json"),
            key=key,
        )
        count += 1

    await producer.flush()
    log.info("network.batch_ingested", count=count)
    return {"status": "accepted", "ingested": count}
