"""
Pydantic request / response schemas for the auth service.

Separation from ORM models keeps API contracts independent of storage
implementation (data-transfer-object pattern).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Requests ─────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """New-user registration payload."""
    email: EmailStr
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(min_length=12, max_length=128)
    full_name: str | None = Field(default=None, max_length=128)

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        """
        NIST 800-63B §5.1.1.2 – no composition rules, but require minimum
        length and block known-bad passwords.  Composition rules (upper/lower/
        special) actually *reduce* entropy because users follow predictable
        patterns.  Length is the primary security control.
        """
        if len(set(v)) < 4:
            raise ValueError("Password must contain at least 4 unique characters")
        return v


class LoginRequest(BaseModel):
    """Login with email or username + password."""
    username: str            # accepts email or username
    password: str


class RefreshRequest(BaseModel):
    """Exchange a valid refresh token for a new access + refresh pair."""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Authenticated password change."""
    current_password: str
    new_password: str = Field(min_length=12, max_length=128)

    @field_validator("new_password")
    @classmethod
    def new_password_complexity(cls, v: str) -> str:
        if len(set(v)) < 4:
            raise ValueError("Password must contain at least 4 unique characters")
        return v


# ── Responses ────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int          # seconds until access token expires


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    username: str
    full_name: str | None
    role: str
    is_active: bool
    created_at: datetime


class MessageResponse(BaseModel):
    message: str
