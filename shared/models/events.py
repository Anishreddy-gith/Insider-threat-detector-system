"""
Shared Event Schemas
====================
Canonical Pydantic models for Kafka event payloads.

Design decisions:
  • We use a single envelope (``EventEnvelope``) for ALL Kafka messages so
    consumers can deserialise without knowing the concrete type upfront.
  • ``event_type`` is a string enum that maps 1-to-1 to a Pydantic model —
    this lets us do schema-validated routing in consumers.
  • ``correlation_id`` enables distributed tracing across microservices.
  • Timestamps are always UTC ISO-8601 to avoid timezone ambiguity.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# ── Event Type Registry ───────────────────────────────────────
class EventType(StrEnum):
    """
    Centralised registry of all event types flowing through Kafka.
    Each constant maps to exactly one Kafka topic.
    """
    USER_ACTIVITY = "user.activity"
    FILE_ACCESS = "file.access"
    NETWORK_EVENT = "network.event"
    AUTH_EVENT = "auth.event"
    ANOMALY_DETECTED = "anomaly.detected"
    RISK_SCORE_UPDATED = "risk.score.updated"
    ALERT_CREATED = "alert.created"
    ALERT_RESOLVED = "alert.resolved"


# ── Envelope ──────────────────────────────────────────────────
class EventEnvelope(BaseModel):
    """
    Universal Kafka message wrapper.

    Why an envelope?
      1. Consumers can peek at ``event_type`` before deserialising ``payload``.
      2. ``correlation_id`` threads distributed traces without coupling to
         a specific tracing backend (Jaeger / Zipkin / OTLP).
      3. ``source_service`` identifies the producer for debugging.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Propagated across services for distributed tracing.",
    )
    source_service: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any]

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# ── Payload Models ────────────────────────────────────────────
class UserActivityPayload(BaseModel):
    """Raw user-behaviour event ingested from endpoint agents."""
    user_id: str
    session_id: str
    action: str  # e.g. "login", "file_download", "usb_insert"
    resource: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FileAccessPayload(BaseModel):
    """File-system access telemetry."""
    user_id: str
    file_path: str
    action: str  # "read" | "write" | "delete" | "rename" | "copy"
    file_size_bytes: int | None = None
    sensitivity_label: str | None = None  # "public" | "internal" | "confidential" | "restricted"
    destination: str | None = None  # non-null for copy/move
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NetworkEventPayload(BaseModel):
    """Network flow metadata captured at perimeter / DNS / proxy."""
    user_id: str | None = None
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str  # "TCP" | "UDP" | "ICMP"
    bytes_sent: int = 0
    bytes_received: int = 0
    domain: str | None = None
    is_encrypted: bool = True
    geo_country: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnomalyPayload(BaseModel):
    """Emitted by the Behavioral Analytics service when a model flags unusual activity."""
    user_id: str
    anomaly_score: float = Field(ge=0.0, le=1.0)
    model_name: str  # e.g. "lstm_autoencoder", "gnn_v2"
    model_version: str
    feature_contributions: dict[str, float] = Field(
        default_factory=dict,
        description="SHAP / feature-attribution values for explainability.",
    )
    raw_features: dict[str, Any] = Field(default_factory=dict)
    explanation: str | None = None  # human-readable XAI summary
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RiskScorePayload(BaseModel):
    """Composite risk score aggregating multiple anomaly signals."""
    user_id: str
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: str  # "low" | "medium" | "high" | "critical"
    contributing_factors: list[dict[str, Any]] = Field(default_factory=list)
    previous_score: float | None = None
    score_delta: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AlertPayload(BaseModel):
    """Security alert requiring analyst attention."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    severity: str  # "info" | "low" | "medium" | "high" | "critical"
    title: str
    description: str
    risk_score: float = Field(ge=0.0, le=100.0)
    indicators: list[dict[str, Any]] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    is_resolved: bool = False
    assigned_to: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
