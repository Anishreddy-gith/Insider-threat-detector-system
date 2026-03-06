"""
Data Ingestion Service — Configuration
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    service_name: str = "data-ingestion"
    kafka_bootstrap_servers: str = "localhost:9092"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "insider_threat_db"
    postgres_user: str = "itds_admin"
    postgres_password: str = "changeme"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "changeme"

    batch_size: int = 1000
    flush_interval_seconds: int = 5

    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
