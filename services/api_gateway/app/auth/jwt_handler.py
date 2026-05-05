"""
JWT Token Handler
==================
Creates and verifies JWT access / refresh tokens using ``python-jose``.

Design decisions
----------------
* **HS256 symmetric signing** â€” acceptable for a single-gateway topology
  where the same process creates and verifies tokens.  If the gateway is
  ever split into multiple services that each need to verify tokens
  independently, migrate to RS256 with a shared public key.
* **Role embedded in payload** â€” so the gateway can enforce RBAC without a
  DB round-trip on every request.  Trade-off: role changes only take effect
  when the current token expires.  30-minute access token TTL limits the
  staleness window.
* **Separate refresh token** â€” long-lived (7 days), used only to obtain
  new access tokens.  Stored in ``httpOnly`` cookie or frontend secure
  storage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from services.api_gateway.app.config import get_settings

settings = get_settings()


# â”€â”€ Password hashing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# â”€â”€ RBAC roles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class Role(StrEnum):
    """
    A user's security role determines what they can see and do.

    Hierarchy (most-permissive â†’ least):
      ADMIN > SOC_MANAGER > SOC_ANALYST > HR_SYSTEM

    * **SOC_ANALYST** â€” front-line SOC operator.  Can view alerts,
      submit feedback (true-positive / false-positive verdicts),
      and view entity risk profiles.
    * **SOC_MANAGER** â€” team lead.  Everything an analyst can do,
      plus configure adaptive thresholds, view model health, and
      access the FP/FN analysis reports.
    * **ADMIN** â€” platform administrator.  Full access including
      user management, system configuration, and audit log export.
    * **HR_SYSTEM** â€” machine-to-machine service account.  Can only
      push employment context (hires, terminations, dept changes)
      and read nothing else.  Minimal-privilege principle.
    """
    SOC_ANALYST = "SOC_ANALYST"
    SOC_MANAGER = "SOC_MANAGER"
    ADMIN = "ADMIN"
    HR_SYSTEM = "HR_SYSTEM"


# â”€â”€ Token helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def create_access_token(
    user_id: str,
    role: str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """
    Create a short-lived JWT access token.

    Payload includes:
      â€¢ ``sub``  â€” user UUID (subject)
      â€¢ ``role`` â€” RBAC role string
      â€¢ ``iat``  â€” issued-at timestamp
      â€¢ ``exp``  â€” expiry timestamp
      â€¢ ``type`` â€” "access" (distinguishes from refresh tokens)
    """
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_access_token_expire_minutes),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(user_id: str) -> str:
    """Create a long-lived refresh token (7-day default)."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_refresh_token_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and verify a JWT.

    Raises ``JWTError`` on invalid/expired tokens â€” callers must handle.
    """
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )


def verify_access_token(token: str) -> dict[str, Any]:
    """Decode an access token and ensure it is not a refresh token."""
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise JWTError("Token is not an access token.")
    return payload


def verify_refresh_token(token: str) -> dict[str, Any]:
    """Decode a refresh token and ensure it is not an access token."""
    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise JWTError("Token is not a refresh token.")
    return payload

