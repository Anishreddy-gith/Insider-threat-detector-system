"""
Security utilities â€“ password hashing, JWT token lifecycle, account lockout.

Design rationale:
- **bcrypt** is chosen over argon2 because passlib's bcrypt backend is
  battle-tested, widely deployed, and sufficient for our password volume
  (~low thousands of hashes/day).  Argon2 is superior for very high-volume
  scenarios but adds a C-library build dependency.
- **HS256** is used for JWTs because the auth-service is the sole issuer
  *and* validator.  RS256 asymmetric signing would be needed if we wanted
  other services to verify tokens without possessing the secret â€“ our API
  Gateway forwards the already-validated claims, so symmetric is fine.
- Refresh token rotation: each refresh issues a *new* refresh token and
  invalidates the old one, limiting the window for stolen-token replay.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from services.auth_service.app.config import get_settings

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)

# â”€â”€ Password hashing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.BCRYPT_ROUNDS,
)


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*.  Never store or log the plaintext."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time comparison to prevent timing side-channels."""
    return pwd_context.verify(plain, hashed)


# â”€â”€ JWT lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def create_access_token(
    user_id: str,
    role: str,
    username: str,
) -> tuple[str, int]:
    """
    Create a short-lived access token.

    Returns:
        (encoded_jwt, expires_in_seconds)
    """
    expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_in = int(expires_delta.total_seconds())
    now = datetime.now(timezone.utc)
    claims = {
        "sub": user_id,
        "role": role,
        "username": username,
        "iat": now,
        "exp": now + expires_delta,
        "type": "access",
        "jti": str(uuid.uuid4()),       # unique token id for revocation lists
    }
    token = jwt.encode(claims, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, expires_in


def create_refresh_token(user_id: str, family: str | None = None) -> tuple[str, str]:
    """
    Create a long-lived refresh token with a family identifier.

    The family links all tokens in a rotation chain.  If a previously-used
    token is replayed, the entire family can be revoked.

    Returns:
        (encoded_jwt, family_id)
    """
    family = family or str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    claims = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        "type": "refresh",
        "family": family,
        "jti": str(uuid.uuid4()),
    }
    token = jwt.encode(claims, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, family


def decode_token(token: str) -> dict | None:
    """
    Decode and validate a JWT.  Returns claims dict or None on failure.

    We let python-jose handle exp validation automatically.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError as exc:
        logger.warning("jwt_decode_failed", extra={"error": str(exc)})
        return None


# â”€â”€ Account lockout helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def is_account_locked(locked_until: datetime | None) -> bool:
    """Check whether the account lockout window is still active."""
    if locked_until is None:
        return False
    return datetime.now(timezone.utc) < locked_until


def get_lockout_until() -> datetime:
    """Calculate the lockout expiration timestamp from now."""
    return datetime.now(timezone.utc) + timedelta(
        minutes=settings.LOCKOUT_DURATION_MINUTES
    )

