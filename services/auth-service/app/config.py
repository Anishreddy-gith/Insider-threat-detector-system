"""
Auth-service configuration – single source of truth for every tuneable parameter.

Security rationale:
- JWT secrets MUST come from environment variables, never hardcoded.
- bcrypt rounds default to 12 (OWASP recommendation ≥ 10).
- Access tokens are short-lived (15 min); refresh tokens longer (7 days)
  to limit the blast radius of a leaked JWT while keeping UX smooth.
"""

from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Server ────────────────────────────────────────────────
    SERVICE_NAME: str = "auth-service"
    HOST: str = "0.0.0.0"
    PORT: int = 8004
    DEBUG: bool = False

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://itds:itds@localhost:5432/itds"

    # ── Redis ─────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── JWT ───────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Password Policy (NIST 800-63B) ───────────────────────
    PASSWORD_MIN_LENGTH: int = 12          # NIST recommends ≥ 8; we default to 12
    BCRYPT_ROUNDS: int = 12                # ~250 ms per hash on modern hardware

    # ── Rate Limiting ─────────────────────────────────────────
    LOGIN_RATE_LIMIT: int = 10             # max login attempts per window
    LOGIN_RATE_WINDOW_SECONDS: int = 300   # 5-minute window

    # ── Account Lockout ───────────────────────────────────────
    MAX_FAILED_ATTEMPTS: int = 5           # lock after N consecutive failures
    LOCKOUT_DURATION_MINUTES: int = 30     # auto-unlock after this period


@lru_cache
def get_settings() -> Settings:
    return Settings()
