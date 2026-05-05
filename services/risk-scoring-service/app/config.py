"""
Risk-scoring-service configuration.

Centralises all tunables for the risk scorer: ML-engine connection,
database, Redis, thresholds, context rules, and PID controller gains.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    SERVICE_NAME: str = "risk-scoring-service"
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    DEBUG: bool = False

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"
    KAFKA_ANOMALIES_TOPIC: str = "itds.anomalies"
    KAFKA_RISK_SCORES_TOPIC: str = "itds.risk-scores"
    KAFKA_CONSUMER_GROUP: str = "risk-scoring-cg"

    # Redis (score state + threshold cache)
    REDIS_URL: str = "redis://redis:6379/2"

    # PostgreSQL / TimescaleDB (score history)
    DATABASE_URL: str = (
        "postgresql+asyncpg://itds:itds@timescaledb:5432/risk_scores"
    )

    # ML-engine connection
    ML_ENGINE_URL: str = "http://ml-engine:8002"
    ML_ENGINE_TIMEOUT: float = 10.0

    # HR API (mock or real)
    HR_API_URL: str = "http://hr-api:8080"
    HR_API_TIMEOUT: float = 5.0

    # Risk engine tunables
    RISK_DECAY_LAMBDA: float = 0.01        # exponential decay rate (per hour)
    SEVERITY_WEIGHT_LOW: float = 5.0
    SEVERITY_WEIGHT_MEDIUM: float = 15.0
    SEVERITY_WEIGHT_HIGH: float = 30.0
    SEVERITY_WEIGHT_CRITICAL: float = 50.0
    ALERT_COOLDOWN_SECONDS: int = 3600     # 1-hour dedup window

    # Context-aware scoring
    CONTEXT_RULES_PATH: str = "app/rules/context_rules.yaml"

    # Adaptive threshold PID controller
    PID_KP: float = 0.5                    # proportional gain
    PID_KI: float = 0.1                    # integral gain
    PID_KD: float = 0.05                   # derivative gain
    FP_TARGET_RATE: float = 0.05           # target false positive rate (5%)
    THRESHOLD_LOW: float = 25.0
    THRESHOLD_MEDIUM: float = 50.0
    THRESHOLD_HIGH: float = 75.0
    THRESHOLD_MIN: float = 10.0            # minimum allowed threshold
    THRESHOLD_MAX: float = 95.0            # maximum allowed threshold

    # Trend analyzer
    TREND_WINDOW_DAYS: int = 30
    SLOW_BURN_LOOKBACK_DAYS: int = 14
    SLOW_BURN_SLOPE_THRESHOLD: float = 0.02  # daily slope that triggers alert


@lru_cache
def get_settings() -> Settings:
    return Settings()
