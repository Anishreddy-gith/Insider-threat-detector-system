"""
SQLAlchemy 2.0 ORM models for risk-scoring-service.

The ``score_history`` table is designed for TimescaleDB: it uses
``scored_at`` as the time column so that TimescaleDB can partition
into hypertables automatically.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ScoreHistory(Base):
    """
    Per-user risk score history.

    TimescaleDB hypertable on ``scored_at`` for fast time-range queries.
    """

    __tablename__ = "score_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    detector_breakdown: Mapped[dict] = mapped_column(JSON, nullable=True)
    context_adjustments: Mapped[dict] = mapped_column(JSON, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_score_history_user_time", "user_id", "scored_at"),
    )


class ThresholdState(Base):
    """
    Persists the adaptive threshold PID controller state.
    """

    __tablename__ = "threshold_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    threshold_low: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_medium: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_high: Mapped[float] = mapped_column(Float, nullable=False)
    fp_rate: Mapped[float] = mapped_column(Float, nullable=False)
    integral_error: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_error: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
