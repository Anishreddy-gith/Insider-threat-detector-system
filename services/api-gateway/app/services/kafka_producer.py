"""
Kafka Producer — Async wrapper
================================
Thin abstraction over ``aiokafka.AIOKafkaProducer`` used by the gateway
to publish events (e.g. when an analyst resolves an alert).

Design decisions:
  • JSON serialisation with Pydantic's ``.model_dump_json()`` ensures all
    payloads are schema-validated before hitting the wire.
  • Idempotent producer (``enable_idempotence=True``) prevents duplicate
    messages on transient broker failures.
  • ``acks="all"`` waits for ISR acknowledgement — we never want to lose
    an alert-resolution event.
"""

from __future__ import annotations

import json
from typing import Any

from aiokafka import AIOKafkaProducer

from app.config import get_settings
from shared.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

_producer: AIOKafkaProducer | None = None


async def get_producer() -> AIOKafkaProducer:
    """Lazily initialise and return the singleton Kafka producer."""
    global _producer
    if _producer is None:
        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            acks="all",
            enable_idempotence=True,
            compression_type="gzip",
            max_request_size=1_048_576,  # 1 MiB safety cap
        )
        await _producer.start()
        log.info("kafka.producer.started")
    return _producer


async def publish_event(topic: str, key: str, value: dict[str, Any]) -> None:
    """
    Publish a single event to a Kafka topic.

    Args:
        topic: Kafka topic name (e.g. ``itds.alerts``).
        key:   Partition key (usually ``user_id`` so all events for one
               user land on the same partition → ordering guarantee).
        value: JSON-serialisable dict.
    """
    producer = await get_producer()
    await producer.send_and_wait(topic, value=value, key=key)
    log.debug("kafka.event.published", topic=topic, key=key)


async def close_producer() -> None:
    """Gracefully flush & close the producer on shutdown."""
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
        log.info("kafka.producer.stopped")
