"""
Kafka producer singleton for the ingestion service.

WHY idempotent?  Network glitches between the ingestion pod and the Kafka
broker can cause duplicate sends.  Setting ``enable_idempotence=True``
guarantees exactly-once delivery at the producer level (Kafka ≥ 0.11).
"""

from __future__ import annotations

import json
import logging

from aiokafka import AIOKafkaProducer

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)

producer = AIOKafkaProducer(
    bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v, default=str).encode(),
    compression_type=settings.KAFKA_COMPRESSION,
    linger_ms=settings.KAFKA_LINGER_MS,
    max_batch_size=settings.KAFKA_BATCH_SIZE,
    enable_idempotence=True,
)


async def publish_event(topic: str, key: str, payload: dict) -> None:
    """Publish a single event to Kafka, keyed by entity-id for ordering."""
    await producer.send_and_wait(
        topic,
        key=key.encode(),
        value=payload,
    )
    logger.debug("event_published", extra={"topic": topic, "key": key})
