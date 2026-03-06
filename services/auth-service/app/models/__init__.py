"""
Auth-service ORM models.

Design decisions:
- UUID primary keys prevent sequential enumeration attacks.
- ``failed_login_attempts`` + ``locked_until`` implement account-lockout
  without external state (Redis) — keeps the authority in the DB.
- ``refresh_token_family`` supports refresh-token rotation; if a previously
  used token is replayed, the entire family is revoked (prevents token theft
  replay chains — see RFC 6749 §10.4).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    """Core user identity – passwords stored as bcrypt hashes only."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String(320), unique=True, nullable=False, index=True
    )
    username: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default="viewer"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # ── Account lockout ─────────────────────────────────────
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Refresh-token rotation ──────────────────────────────
    refresh_token_family: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )

    # ── Timestamps ──────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<User {self.username} role={self.role}>"
