"""Canonical event envelope shared by all microservices."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class EventEnvelope(BaseModel):
    """
    Wire format transported via Kafka.

    Every event emitted or consumed by any ITDS service MUST be
    wrapped in this envelope so that downstream consumers can
    deserialise uniformly.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str                          # e.g. syslog, endpoint, anomaly
    source_service: str                      # e.g. ingestion-service
    entity_id: str                           # user / device / host identifier
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    payload: dict[str, Any]                  # service-specific body
    metadata: dict[str, Any] = Field(default_factory=dict)

    def partition_key(self) -> bytes:
        """Kafka partition key – guarantees per-entity ordering."""
        return self.entity_id.encode()
