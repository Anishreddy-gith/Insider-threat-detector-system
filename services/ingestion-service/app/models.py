"""
SQLAlchemy 2.0 ORM models – async-native, mapped-column style.

Tables
──────
user_events      –  Every raw event ingested (login, logout, file_access,
                    network_request, app_usage).
user_baselines   –  Rolling 30-day statistics per user per behaviour type,
                    maintained via Welford's online algorithm.
risk_scores      –  Time-series risk score per user, written by the
                    risk-scoring-service but readable here for enrichment.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ── Base ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared declarative base for all ingestion-service models."""
    pass


# ── Enumerations ────────────────────────────────────────────────────────

class EventType(str, PyEnum):
    LOGIN            = "login"
    LOGOUT           = "logout"
    FILE_ACCESS      = "file_access"
    NETWORK_REQUEST  = "network_request"
    APP_USAGE        = "app_usage"


class RiskLevel(str, PyEnum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ── UserEvent ───────────────────────────────────────────────────────────

class UserEvent(Base):
    """
    Canonical store for every ingested user-activity event.

    Partitioned logically by ``event_type``; a composite index on
    ``(user_id, event_type, timestamp)`` supports the baseline engine's
    range-scan queries.
    """

    __tablename__ = "user_events"
    __table_args__ = (
        Index("ix_userevents_user_type_ts", "user_id", "event_type", "timestamp"),
        Index("ix_userevents_timestamp", "timestamp"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # ── event-type-specific payload ──
    source_ip: Mapped[str | None] = mapped_column(String(45))
    destination_ip: Mapped[str | None] = mapped_column(String(45))
    destination_port: Mapped[int | None] = mapped_column(Integer)
    file_path: Mapped[str | None] = mapped_column(Text)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    application_name: Mapped[str | None] = mapped_column(String(256))
    action: Mapped[str | None] = mapped_column(String(64))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    bytes_transferred: Mapped[int | None] = mapped_column(Integer)
    sensitivity_label: Mapped[str | None] = mapped_column(String(32))

    # ── flexible overflow bucket ──
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)

    # ── audit ──
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


# ── UserBaseline ────────────────────────────────────────────────────────

class UserBaseline(Base):
    """
    Rolling 30-day behavioural baseline per user per behaviour metric.

    The ``count``, ``mean``, and ``m2`` columns store Welford's online
    algorithm state so that mean & variance can be updated in *O(1)* per
    event without re-reading historical data.

    Variance = m2 / (count - 1)   (Bessel-corrected sample variance)
    """

    __tablename__ = "user_baselines"
    __table_args__ = (
        Index("ix_baselines_user_metric", "user_id", "metric_name", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)

    # ── Welford state ──
    count: Mapped[int] = mapped_column(Integer, default=0)
    mean: Mapped[float] = mapped_column(Float, default=0.0)
    m2: Mapped[float] = mapped_column(Float, default=0.0)     # sum of (x - mean)^2

    # ── convenience snapshots ──
    min_value: Mapped[float] = mapped_column(Float, default=0.0)
    max_value: Mapped[float] = mapped_column(Float, default=0.0)
    last_value: Mapped[float] = mapped_column(Float, default=0.0)

    # ── window management ──
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


# ── RiskScore ───────────────────────────────────────────────────────────

class RiskScore(Base):
    """
    Time-series risk score per user.

    Written by risk-scoring-service; queryable here so the ingestion
    service can enrich events with the latest risk context before
    publishing to Kafka.
    """

    __tablename__ = "risk_scores"
    __table_args__ = (
        Index("ix_risk_user_ts", "user_id", "calculated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    contributing_factors: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
