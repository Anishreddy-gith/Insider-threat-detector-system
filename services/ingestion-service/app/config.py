"""
Ingestion-service configuration.

Every tuneable knob lives here.  Values are loaded from environment
variables (or .env) via pydantic-settings; the lru_cache ensures a
single Settings instance for the process lifetime.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    SERVICE_NAME: str = "ingestion-service"
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = False

    # ── PostgreSQL ──────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://itds:itds_secret@postgres:5432/itds"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # ── Redis ───────────────────────────────────────────────────────
    REDIS_URL: str = "redis://:itds_redis@redis:6379/0"

    # ── Kafka (producer) ────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"
    KAFKA_RAW_EVENTS_TOPIC: str = "itds.raw-events"
    KAFKA_COMPRESSION: str = "lz4"
    KAFKA_LINGER_MS: int = 10
    KAFKA_BATCH_SIZE: int = 65536

    # ── Kafka (consumer) ────────────────────────────────────────────
    KAFKA_CONSUMER_GROUP: str = "ingestion-service-cg"

    # ── Rate limiting ──────────────────────────────────────────────
    INGEST_RATE_LIMIT: int = 5000          # events per second
    MAX_BATCH_SIZE: int = 1000             # max events in a single batch POST


@lru_cache
def get_settings() -> Settings:
    return Settings()
