"""
Behavioral Analytics — Configuration
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    service_name: str = "behavioral-analytics"
    kafka_bootstrap_servers: str = "localhost:9092"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "insider_threat_db"
    postgres_user: str = "itds_admin"
    postgres_password: str = "changeme"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "changeme"

    model_registry_path: str = "/models"
    anomaly_threshold: float = 0.85
    enable_gnn_model: bool = True
    enable_explainability: bool = True

    # Differential privacy budget (ε, δ).
    # Lower ε → stronger privacy, noisier model.
    # ε = 1.0 is a common starting point balancing utility vs. privacy.
    differential_privacy_epsilon: float = 1.0
    differential_privacy_delta: float = 1e-5

    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
