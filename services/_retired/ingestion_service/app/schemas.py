"""
Pydantic v2 schemas – strict validation for every event type.

Each schema enforces:
  • type-annotated fields with constrained ranges
  • ISO-8601 timestamps (defaulting to UTC now)
  • discriminated-union ``IngestEvent`` so a single endpoint can accept
    any event type via ``event_type`` literal discrimination
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    IPvAnyAddress,
    field_validator,
)


# ── Shared helpers ──────────────────────────────────────────────────────

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


EventId = Annotated[str, Field(default_factory=lambda: str(uuid.uuid4()))]


class _EventBase(BaseModel):
    """Fields common to every event type."""
    model_config = ConfigDict(strict=True, frozen=True)

    event_id: EventId = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., min_length=1, max_length=128, description="Unique user identifier")
    timestamp: datetime = Field(default_factory=_utc_now)

    @field_validator("timestamp", mode="after")
    @classmethod
    def _coerce_utc(cls, v: datetime) -> datetime:
        """Normalise every timestamp to UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


# ── Login / Logout ──────────────────────────────────────────────────────

class LoginEvent(_EventBase):
    event_type: Literal["login"] = "login"
    source_ip: str = Field(..., description="Client IP address")
    user_agent: str = Field("", max_length=512)
    auth_method: str = Field("password", pattern=r"^(password|sso|mfa|certificate)$")
    success: bool = True
    geo_location: str | None = Field(None, max_length=128)


class LogoutEvent(_EventBase):
    event_type: Literal["logout"] = "logout"
    source_ip: str = Field(..., description="Client IP address")
    session_duration_seconds: float = Field(..., ge=0)


# ── File access ─────────────────────────────────────────────────────────

class FileAccessAction(str, Enum):
    READ     = "read"
    WRITE    = "write"
    DELETE   = "delete"
    COPY     = "copy"
    MOVE     = "move"
    DOWNLOAD = "download"
    UPLOAD   = "upload"


class FileAccessEvent(_EventBase):
    event_type: Literal["file_access"] = "file_access"
    file_path: str = Field(..., min_length=1, max_length=4096)
    file_size_bytes: int = Field(..., ge=0)
    action: FileAccessAction
    sensitivity_label: str = Field("unclassified", pattern=r"^(unclassified|internal|confidential|restricted)$")
    source_device: str = Field("", max_length=256)


# ── Network request ────────────────────────────────────────────────────

class NetworkRequestEvent(_EventBase):
    event_type: Literal["network_request"] = "network_request"
    source_ip: str = Field(...)
    destination_ip: str = Field(...)
    destination_port: int = Field(..., ge=1, le=65535)
    protocol: str = Field("TCP", pattern=r"^(TCP|UDP|ICMP|SCTP)$")
    bytes_sent: int = Field(0, ge=0)
    bytes_received: int = Field(0, ge=0)
    domain: str | None = Field(None, max_length=253)
    is_encrypted: bool = True


# ── Application usage ──────────────────────────────────────────────────

class AppUsageEvent(_EventBase):
    event_type: Literal["app_usage"] = "app_usage"
    application_name: str = Field(..., min_length=1, max_length=256)
    window_title: str = Field("", max_length=512)
    duration_seconds: float = Field(..., ge=0)
    foreground: bool = True
    category: str = Field("uncategorised", max_length=64)


# ── Discriminated union ────────────────────────────────────────────────

IngestEvent = Annotated[
    Union[
        LoginEvent,
        LogoutEvent,
        FileAccessEvent,
        NetworkRequestEvent,
        AppUsageEvent,
    ],
    Field(discriminator="event_type"),
]


# ── Response models ─────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    accepted: bool = True
    event_id: str
    event_type: str


class IngestBatchResponse(BaseModel):
    accepted: int
    event_ids: list[str]
