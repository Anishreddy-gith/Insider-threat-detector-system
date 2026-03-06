"""
Project-wide constants.

Import these from shared-libs instead of hard-coding strings or magic
numbers across services.
"""

from __future__ import annotations

# ── Kafka Topics ─────────────────────────────────────────────────────
TOPIC_RAW_EVENTS   = "itds.raw-events"
TOPIC_ANOMALIES    = "itds.anomalies"
TOPIC_RISK_SCORES  = "itds.risk-scores"
TOPIC_ALERTS       = "itds.alerts"

# ── Risk Levels ──────────────────────────────────────────────────────
RISK_LOW      = "low"
RISK_MEDIUM   = "medium"
RISK_HIGH     = "high"
RISK_CRITICAL = "critical"

# ── Alert Statuses ───────────────────────────────────────────────────
ALERT_OPEN          = "open"
ALERT_ACKNOWLEDGED  = "acknowledged"
ALERT_ESCALATED     = "escalated"
ALERT_RESOLVED      = "resolved"

# ── RBAC Roles ───────────────────────────────────────────────────────
ROLE_ADMIN      = "admin"
ROLE_ANALYST    = "analyst"
ROLE_VIEWER     = "viewer"
ROLE_SYSTEM     = "system"

# ── Service Ports (for Docker/Compose inter-service references) ──────
SERVICE_PORTS = {
    "api-gateway":         8000,
    "ingestion-service":   8001,
    "ml-engine":           8002,
    "risk-scoring-service": 8003,
    "alert-service":       8004,
    "frontend":            3000,
}
