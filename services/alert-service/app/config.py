"""
Alert-service configuration.

Centralises all tunables: database, Kafka, notification channels,
alert thresholds, deduplication, feedback loop, and FP/FN analysis.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    SERVICE_NAME: str = "alert-service"
    HOST: str = "0.0.0.0"
    PORT: int = 8004
    DEBUG: bool = False

    # ── PostgreSQL ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://itds:itds@postgres:5432/itds_alerts"

    # ── Kafka ───────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"
    KAFKA_RISK_SCORES_TOPIC: str = "itds.risk-scores"
    KAFKA_ALERTS_TOPIC: str = "itds.alerts"
    KAFKA_CONSUMER_GROUP: str = "alert-service-cg"

    # ── Alert thresholds (on [0, 1] scale) ──────────────────────
    ALERT_SCORE_THRESHOLD: float = 0.7       # generate alert above this
    THRESHOLD_HIGH: float = 0.7
    THRESHOLD_CRITICAL: float = 0.85

    # ── Deduplication ───────────────────────────────────────────
    DEDUP_WINDOW_SECONDS: int = 3600         # 1 hour dedup window

    # ── Risk-scoring service ────────────────────────────────────
    RISK_SCORING_URL: str = "http://risk-scoring-service:8003"

    # ── Notification: Slack ─────────────────────────────────────
    SLACK_WEBHOOK_URL: str = ""
    SLACK_CHANNEL: str = "#insider-threat-alerts"
    SLACK_ENABLED: bool = False

    # ── Notification: Email (SMTP) ──────────────────────────────
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_USE_TLS: bool = True
    EMAIL_FROM: str = "itds-alerts@company.com"
    EMAIL_TO: list[str] = ["soc@company.com"]
    EMAIL_ENABLED: bool = False

    # ── Notification: SIEM webhook ──────────────────────────────
    SIEM_WEBHOOK_URL: str = ""
    SIEM_FORMAT: str = "sentinel"            # "sentinel" | "splunk"
    SIEM_AUTH_TOKEN: str = ""
    SIEM_ENABLED: bool = False

    # ── Feedback loop ───────────────────────────────────────────
    FEEDBACK_RETRAIN_CRON: str = "0 2 * * 0"  # weekly Sun 02:00
    ENSEMBLE_WEIGHTS_PATH: str = "data/ensemble_weights.json"
    MIN_FEEDBACK_FOR_RETRAIN: int = 50       # minimum samples to retrain

    # ── FP/FN analyzer ──────────────────────────────────────────
    ANALYSIS_WINDOW_DAYS: int = 7
    ANALYSIS_REPORT_DIR: str = "data/reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
