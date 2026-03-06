"""
ML-engine configuration.

Separates ML hyper-parameters from infrastructure knobs.
The SHAP_THRESHOLD gates expensive explanation generation – only events
whose ensemble anomaly score exceeds this value trigger full SHAP analysis.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    SERVICE_NAME: str = "ml-engine"
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    DEBUG: bool = False

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"
    KAFKA_RAW_EVENTS_TOPIC: str = "itds.raw-events"
    KAFKA_ANOMALIES_TOPIC: str = "itds.anomalies"
    KAFKA_CONSUMER_GROUP: str = "ml-engine-cg"

    # Redis (feature cache)
    REDIS_URL: str = "redis://redis:6379/1"

    # ML hyper-parameters
    AUTOENCODER_HIDDEN_DIM: int = 128
    AUTOENCODER_NUM_LAYERS: int = 2
    AUTOENCODER_SEQ_LEN: int = 24
    FEATURE_DIM: int = 45
    GNN_HIDDEN_DIM: int = 64
    GNN_NUM_LAYERS: int = 3
    ENSEMBLE_WEIGHTS_AUTOENCODER: float = 0.6
    ENSEMBLE_WEIGHTS_GNN: float = 0.4

    # Explainability
    SHAP_THRESHOLD: float = 0.65           # only explain scores above this
    SHAP_MAX_BACKGROUND_SAMPLES: int = 100

    # Differential privacy
    DP_EPSILON: float = 1.0
    DP_DELTA: float = 1e-5
    DP_NOISE_MULTIPLIER: float = 0.5
    DP_MAX_GRAD_NORM: float = 1.0
    DP_DAILY_BUDGET: float = 10.0
    DP_ALERT_THRESHOLD: float = 0.80

    # Federated learning
    FL_NUM_ROUNDS: int = 10
    FL_LOCAL_EPOCHS: int = 5
    FL_MIN_CLIENTS: int = 2

    # Model paths
    MODEL_DIR: str = "/app/models/weights"


@lru_cache
def get_settings() -> Settings:
    return Settings()
