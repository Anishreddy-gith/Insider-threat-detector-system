"""
Shared Constants
================
Single source of truth for magic numbers, topic names, and status enums
used across multiple microservices.
"""

from __future__ import annotations

from typing import Final


class KafkaTopics:
    """Backward-compatible topic namespace used by existing tests."""

    RAW_EVENTS: Final[str] = "itds.user.activity"
    ANOMALIES: Final[str] = "itds.anomaly.detected"
    RISK_EVENTS: Final[str] = "itds.risk.score"


class RiskLevel:
    """Backward-compatible risk-level namespace used by existing tests."""

    LOW: Final[str] = "low"
    MEDIUM: Final[str] = "medium"
    HIGH: Final[str] = "high"
    CRITICAL: Final[str] = "critical"

# ── Kafka Topics ──────────────────────────────────────────────
# Topic names follow the pattern: itds.<domain>.<action>
TOPIC_USER_ACTIVITY: Final[str] = "itds.user.activity"
TOPIC_FILE_ACCESS: Final[str] = "itds.file.access"
TOPIC_NETWORK_EVENT: Final[str] = "itds.network.event"
TOPIC_AUTH_EVENT: Final[str] = "itds.auth.event"
TOPIC_ANOMALY_DETECTED: Final[str] = "itds.anomaly.detected"
TOPIC_RISK_SCORE: Final[str] = "itds.risk.score"
TOPIC_ALERTS: Final[str] = "itds.alerts"

# ── Risk Levels ───────────────────────────────────────────────
RISK_LOW: Final[str] = "low"
RISK_MEDIUM: Final[str] = "medium"
RISK_HIGH: Final[str] = "high"
RISK_CRITICAL: Final[str] = "critical"

RISK_THRESHOLDS: Final[dict[str, tuple[float, float]]] = {
    RISK_LOW: (0.0, 25.0),
    RISK_MEDIUM: (25.0, 50.0),
    RISK_HIGH: (50.0, 75.0),
    RISK_CRITICAL: (75.0, 100.0),
}

# ── Alert Severities ─────────────────────────────────────────
SEVERITY_INFO: Final[str] = "info"
SEVERITY_LOW: Final[str] = "low"
SEVERITY_MEDIUM: Final[str] = "medium"
SEVERITY_HIGH: Final[str] = "high"
SEVERITY_CRITICAL: Final[str] = "critical"

# ── RBAC Roles ────────────────────────────────────────────────
ROLE_ADMIN: Final[str] = "admin"
ROLE_ANALYST: Final[str] = "analyst"
ROLE_VIEWER: Final[str] = "viewer"
ROLE_AUDITOR: Final[str] = "auditor"

# ── Data Sensitivity Labels (aligns with Microsoft MIP) ──────
SENSITIVITY_PUBLIC: Final[str] = "public"
SENSITIVITY_INTERNAL: Final[str] = "internal"
SENSITIVITY_CONFIDENTIAL: Final[str] = "confidential"
SENSITIVITY_RESTRICTED: Final[str] = "restricted"

# ── Model Names ───────────────────────────────────────────────
MODEL_LSTM_AUTOENCODER: Final[str] = "lstm_autoencoder"
MODEL_GNN: Final[str] = "gnn_entity_graph"
MODEL_ENSEMBLE: Final[str] = "ensemble_v1"

# ── Redis Key Prefixes ───────────────────────────────────────
REDIS_PREFIX_SESSION: Final[str] = "session:"
REDIS_PREFIX_RISK: Final[str] = "risk:"
REDIS_PREFIX_RATE_LIMIT: Final[str] = "rl:"
REDIS_PREFIX_FEATURE_CACHE: Final[str] = "feat:"

# ── Pagination Defaults ──────────────────────────────────────
DEFAULT_PAGE_SIZE: Final[int] = 50
MAX_PAGE_SIZE: Final[int] = 500
