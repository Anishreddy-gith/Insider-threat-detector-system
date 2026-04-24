"""
API Gateway — Configuration
============================
Centralised, validated settings loaded from environment variables.

Every config value is type-checked at startup via ``pydantic-settings``
so misconfigurations fail fast rather than silently corrupting runtime
behaviour.

Sections:
  • Service identity & debug
  • PostgreSQL (gateway read-model + immutable audit log)
  • Redis  (rate-limiting, HR cache, dashboard cache)
  • Kafka  (event publishing)
  • JWT / Auth  (secret, algorithm, expiry, refresh)
  • CORS
  • Inter-service URLs  (aligned with docker-compose env vars)
  • Per-role rate limits
  • HR integration
  • SIEM integration (Splunk HEC + Microsoft Sentinel)
  • Logging
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Service identity ──────────────────────────────────────
    service_name: str = "api-gateway"
    debug: bool = False

    # ── PostgreSQL ────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "insider_threat_db"
    postgres_user: str = "itds_admin"
    postgres_password: str = "changeme"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ─────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "changeme"

    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"

    # ── Kafka ─────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"

    # ── JWT / Auth ────────────────────────────────────────────
    jwt_secret_key: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_minutes: int = 10080  # 7 days

    # ── CORS ──────────────────────────────────────────────────
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:5173"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v

    # ── Inter-service URLs (aligned with docker-compose) ──────
    ingestion_service_url: str = Field(
        default="http://ingestion-service:8001",
        alias="INGESTION_URL",
    )
    ml_engine_url: str = Field(
        default="http://ml-engine:8002",
        alias="ML_ENGINE_URL",
    )
    risk_scoring_service_url: str = Field(
        default="http://risk-scoring-service:8003",
        alias="RISK_SCORING_URL",
    )
    alert_service_url: str = Field(
        default="http://alert-service:8004",
        alias="ALERT_URL",
    )

    # ── Per-role rate limits (requests per minute) ────────────
    rate_limit_default: int = 60
    rate_limit_soc_analyst: int = 120
    rate_limit_soc_manager: int = 200
    rate_limit_admin: int = 300
    rate_limit_hr_system: int = 30
    rate_limit_window_seconds: int = 60

    def rate_limit_for_role(self, role: str) -> int:
        """Return the per-minute request cap for a given RBAC role."""
        return {
            "SOC_ANALYST": self.rate_limit_soc_analyst,
            "SOC_MANAGER": self.rate_limit_soc_manager,
            "ADMIN": self.rate_limit_admin,
            "HR_SYSTEM": self.rate_limit_hr_system,
        }.get(role, self.rate_limit_default)

    # ── HR Integration ────────────────────────────────────────
    hr_api_base_url: str = "https://api.workday.example.com/v1"
    hr_api_key: str = ""
    hr_cache_ttl_seconds: int = 3600  # 1 hour
    hr_webhook_secret: str = "hr-webhook-secret"

    # ── SIEM Integration ──────────────────────────────────────
    # Splunk HTTP Event Collector
    splunk_hec_url: str = "https://splunk.example.com:8088/services/collector/event"
    splunk_hec_token: str = ""
    splunk_index: str = "insider_threats"
    splunk_source: str = "itds-api-gateway"
    splunk_sourcetype: str = "_json"

    # Microsoft Sentinel Data Collector API
    sentinel_workspace_id: str = ""
    sentinel_shared_key: str = ""
    sentinel_log_type: str = "InsiderThreatAlert"
    sentinel_api_url: str = ""

    @property
    def sentinel_data_collector_url(self) -> str:
        if self.sentinel_api_url:
            return self.sentinel_api_url
        return (
            f"https://{self.sentinel_workspace_id}"
            f".ods.opinsights.azure.com/api/logs?api-version=2016-04-01"
        )

    # ── Logging ───────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor — settings are immutable after boot."""
    return Settings()
