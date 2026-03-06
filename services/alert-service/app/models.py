"""
SQLAlchemy ORM models for the alert service.

Tables:
    alerts       — structured alerts with lifecycle status
    feedback     — analyst verdicts on alerts
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Alert(Base):
    """Persisted alert with full lifecycle tracking."""

    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)  # HIGH | CRITICAL
    anomaly_type: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    # Lifecycle: NEW → INVESTIGATING → RESOLVED | FALSE_POSITIVE
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="NEW", index=True
    )

    # Detector breakdown & XAI data
    detector_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    xai_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    top_features: Mapped[list | None] = mapped_column(JSON, nullable=True)
    recommended_actions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    business_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Notification tracking
    notified_slack: Mapped[bool] = mapped_column(Boolean, default=False)
    notified_email: Mapped[bool] = mapped_column(Boolean, default=False)
    notified_siem: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    assigned_to: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dedup_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_alerts_user_status", "user_id", "status"),
        Index("ix_alerts_created_at", "created_at"),
    )


class Feedback(Base):
    """Analyst feedback on an alert — feeds the retraining loop."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    analyst_id: Mapped[str] = mapped_column(String(128), nullable=False)

    # Verdict
    verdict: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # TRUE_POSITIVE | FALSE_POSITIVE | NEEDS_REVIEW
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Snapshot of alert data at feedback time (for retraining)
    risk_score_snapshot: Mapped[float] = mapped_column(Float, nullable=False)
    detector_scores_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    department: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_feedback_verdict", "verdict"),
        Index("ix_feedback_created_at", "created_at"),
    )
