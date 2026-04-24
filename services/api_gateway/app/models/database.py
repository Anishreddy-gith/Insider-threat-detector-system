"""
Database Models & Session Factory
===================================
SQLAlchemy 2.0 async ORM models for the API Gateway's read-model.

Tables
------
* ``users``              â€” platform users (analysts, managers, admins)
* ``monitored_entities`` â€” employees whose behaviour is monitored
* ``alerts``             â€” risk alerts raised by the ML pipeline
* ``audit_logs``         â€” **immutable** append-only compliance log

The ``audit_logs`` table deserves special attention:

  **Why immutable?**
  GDPR Article 30 and NIST SP 800-53 (AU-3, AU-9) require organisations
  to maintain tamper-proof records of data-processing activities.  An
  insider threat investigation is itself a sensitive data-processing
  activity â€” if an attacker (or a rogue analyst) could modify audit
  records, they could cover their tracks.  By making the table
  append-only (no UPDATE, no DELETE), we guarantee an unbroken chain
  of evidence that is admissible in legal proceedings.

  The table has additional columns compared to the v1 schema:
    â€¢ ``user_role`` â€” captures the role at the time of action (in case
      the role is later changed, we preserve what they *were* when the
      action occurred).
    â€¢ ``status_code`` â€” HTTP response code, useful for detecting probing
      attacks (e.g.  floods of 403s).
    â€¢ ``response_time_ms`` â€” latency tracking for performance forensics.
    â€¢ ``query_params`` â€” what filters were applied (e.g.  which entity
      was searched).
    â€¢ ``user_agent`` â€” browser/tool fingerprint for correlation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from services.api_gateway.app.config import get_settings

settings = get_settings()


# â”€â”€ Engine & Session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency â€” yields a scoped async session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# â”€â”€ Base â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class Base(DeclarativeBase):
    pass


# â”€â”€ ORM Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class User(Base):
    """
    Internal user record (SOC analyst / manager / admin / HR service account).

    NOT the monitored employee â€” that's ``MonitoredEntity``.
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(
        String(50), nullable=False, default="SOC_ANALYST",
    )  # SOC_ANALYST | SOC_MANAGER | ADMIN | HR_SYSTEM
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class MonitoredEntity(Base):
    """An employee / service-account whose behaviour is being analysed."""
    __tablename__ = "monitored_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(255))
    email = Column(String(255))
    department = Column(String(100))
    role_title = Column(String(100))
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String(20), default="low")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    alerts = relationship("Alert", back_populates="entity", lazy="selectin")

    __table_args__ = (
        Index("ix_entity_risk", "risk_score", "risk_level"),
    )


class Alert(Base):
    """Security alert raised by the risk-scoring pipeline."""
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(
        UUID(as_uuid=True), ForeignKey("monitored_entities.id"), nullable=False,
    )
    severity = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    risk_score = Column(Float, nullable=False)
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    assigned_to = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    entity = relationship("MonitoredEntity", back_populates="alerts")

    __table_args__ = (
        Index("ix_alert_severity_created", "severity", "created_at"),
        Index("ix_alert_resolved", "is_resolved"),
    )


class AuditLog(Base):
    """
    **Immutable** audit trail â€” append-only, no UPDATE / DELETE.

    Every API call is recorded with:
      â€¢ Who â€” authenticated user id + role at time of request
      â€¢ What â€” HTTP method + path + query parameters
      â€¢ When â€” server-side UTC timestamp
      â€¢ Where â€” client IP address
      â€¢ Outcome â€” HTTP status code + response latency

    Compliance references:
      â€¢ GDPR Article 30 (Records of Processing Activities)
      â€¢ NIST SP 800-53 AU-3 (Content of Audit Records)
      â€¢ NIST SP 800-53 AU-9 (Protection of Audit Information)
      â€¢ SOX Section 802 (Alteration of Documents)

    This table is both a **compliance requirement** and an
    **investigative asset**:

    As a *compliance requirement*, it demonstrates to regulators that
    every access to sensitive employee data is logged, attributable,
    and time-stamped.  Without it, the organisation cannot prove it
    processed personal data lawfully.

    As an *investigative asset*, during an active insider threat
    incident the audit log answers critical questions:
      â€¢ Did the analyst only view the alerts they were assigned?
      â€¢ Did anyone export bulk entity data before the breach?
      â€¢ Were thresholds lowered suspiciously before an alert was
        suppressed?
      â€¢ What was the exact sequence of API calls the suspect made?

    The append-only design ensures that even if the insider *is* a
    SOC analyst with database access, they cannot erase the evidence
    of their own actions.
    """
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=True)      # UUID string, nullable for anon
    user_role = Column(String(50), nullable=True)      # role at time of request
    action = Column(String(200), nullable=False)       # "GET /api/v1/alerts"
    resource_path = Column(String(500), nullable=False)
    query_params = Column(Text, nullable=True)         # raw query string
    ip_address = Column(String(45), nullable=True)     # IPv4 or IPv6
    user_agent = Column(String(500), nullable=True)
    status_code = Column(Integer, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_audit_user_time", "user_id", "timestamp"),
        Index("ix_audit_action_time", "action", "timestamp"),
        Index("ix_audit_ip_time", "ip_address", "timestamp"),
    )

