"""
Risk Engine
============
Core logic for computing, decaying, and alerting on composite risk scores.

The risk score for each user is a value in [0, 100] that represents the
probability-weighted severity of insider-threat indicators.

Architecture:
  • State is stored in Redis (not in-memory) so the service can be
    horizontally scaled without shared state.
  • Each anomaly event increases the score; exponential decay pulls it
    back down over time.
  • Alerts are deduplicated with a cooldown window to prevent alert
    fatigue for analysts.

Score formula:
  new_score = decayed_old_score + (anomaly_score × severity_weight × 100)
  where decayed_old_score = old_score × exp(-λt)
        λ = ln(2) / half_life_hours
        t = hours since last score update
"""

from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from shared.constants import (
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_THRESHOLDS,
)
from shared.utils.logging import get_logger

log = get_logger(__name__)


class RiskEngine:
    """
    Stateful risk scoring engine backed by Redis.

    Args:
        redis_client:          Async Redis connection.
        decay_hours:           Half-life for exponential risk decay (hours).
        alert_cooldown_minutes: Minimum time between alerts for the same user.
    """

    def __init__(
        self,
        redis_client: Any,
        decay_hours: int = 24,
        alert_cooldown_minutes: int = 15,
    ) -> None:
        self.redis = redis_client
        # λ for exponential decay: score halves every ``decay_hours`` hours.
        self.decay_lambda = math.log(2) / decay_hours
        self.alert_cooldown_seconds = alert_cooldown_minutes * 60

        # Severity weights — critical anomalies have disproportionate impact.
        self._severity_weights: dict[str, float] = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2,
            "info": 0.05,
        }

    async def process_anomaly(
        self, user_id: str, anomaly: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process an anomaly detection event and update the user's risk score.

        Steps:
          1. Fetch current score from Redis.
          2. Apply exponential decay based on time elapsed.
          3. Add anomaly impact.
          4. Determine risk level.
          5. Check if an alert should be generated.
          6. Persist new score.

        Returns:
            dict with updated risk info + optional alert.
        """
        now = time.time()

        # ── 1. Fetch current state ────────────────────────────
        key = f"risk:{user_id}"
        state_raw = await self.redis.get(key)

        if state_raw:
            state = json.loads(state_raw)
            old_score = state["score"]
            last_update = state["last_update"]
        else:
            old_score = 0.0
            last_update = now

        # ── 2. Exponential decay ──────────────────────────────
        hours_elapsed = (now - last_update) / 3600.0
        decayed_score = old_score * math.exp(-self.decay_lambda * hours_elapsed)

        # ── 3. Add anomaly impact ─────────────────────────────
        anomaly_score = anomaly.get("anomaly_score", 0.0)
        model_name = anomaly.get("model_name", "unknown")

        # Higher severity weight for more confident, higher-scoring anomalies.
        impact = anomaly_score * self._get_impact_multiplier(anomaly) * 100
        new_score = min(decayed_score + impact, 100.0)

        # ── 4. Determine risk level ───────────────────────────
        risk_level = self._score_to_level(new_score)

        # ── 5. Check alert generation ─────────────────────────
        alert = None
        if new_score >= RISK_THRESHOLDS[RISK_HIGH][0]:
            alert = await self._maybe_create_alert(
                user_id, new_score, risk_level, anomaly
            )

        # ── 6. Persist ────────────────────────────────────────
        new_state = {
            "score": new_score,
            "level": risk_level,
            "last_update": now,
            "previous_score": old_score,
            "last_anomaly": anomaly_score,
            "last_model": model_name,
        }
        await self.redis.setex(key, 86400 * 7, json.dumps(new_state))  # 7-day TTL

        log.info(
            "risk.updated",
            user_id=user_id,
            old_score=round(old_score, 2),
            new_score=round(new_score, 2),
            risk_level=risk_level,
            alert_generated=alert is not None,
        )

        return {
            "user_id": user_id,
            "risk_score": round(new_score, 2),
            "risk_level": risk_level,
            "previous_score": round(old_score, 2),
            "score_delta": round(new_score - old_score, 2),
            "alert": alert,
        }

    async def get_current_risk(self, user_id: str) -> dict[str, Any]:
        """Fetch the current risk score for a user (with decay applied)."""
        key = f"risk:{user_id}"
        state_raw = await self.redis.get(key)

        if not state_raw:
            return {
                "user_id": user_id,
                "risk_score": 0.0,
                "risk_level": RISK_LOW,
                "last_update": None,
            }

        state = json.loads(state_raw)
        now = time.time()
        hours_elapsed = (now - state["last_update"]) / 3600.0
        current_score = state["score"] * math.exp(-self.decay_lambda * hours_elapsed)

        return {
            "user_id": user_id,
            "risk_score": round(current_score, 2),
            "risk_level": self._score_to_level(current_score),
            "last_update": datetime.fromtimestamp(
                state["last_update"], tz=timezone.utc
            ).isoformat(),
        }

    def _get_impact_multiplier(self, anomaly: dict[str, Any]) -> float:
        """
        Calculate impact multiplier based on anomaly characteristics.

        Considers:
          • Model confidence (higher confidence → higher impact)
          • Contributing factors (more factors → more concerning)
        """
        confidence = anomaly.get("confidence", 0.5)
        n_factors = len(anomaly.get("feature_contributions", {}))

        # More contributing factors = more systemic anomaly = higher multiplier.
        factor_bonus = min(n_factors * 0.05, 0.3)

        return confidence * (1.0 + factor_bonus)

    async def _maybe_create_alert(
        self,
        user_id: str,
        score: float,
        level: str,
        anomaly: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Create an alert if the cooldown has elapsed.

        Uses Redis SETNX with TTL as a distributed lock — prevents
        duplicate alerts even with multiple service replicas.
        """
        cooldown_key = f"alert_cooldown:{user_id}"
        is_new = await self.redis.set(
            cooldown_key,
            "1",
            ex=self.alert_cooldown_seconds,
            nx=True,  # only set if not exists
        )

        if not is_new:
            log.debug("alert.cooldown_active", user_id=user_id)
            return None

        severity = "critical" if score >= 75 else "high"

        # Build explanation from anomaly data.
        explanation = anomaly.get("explanation", {})
        nl_explanation = explanation.get(
            "natural_language_explanation",
            f"Anomaly score {anomaly.get('anomaly_score', 0):.2f} exceeded threshold.",
        )

        alert = {
            "alert_id": str(uuid.uuid4()),
            "user_id": user_id,
            "severity": severity,
            "title": f"Insider Threat Alert — {level.upper()} Risk",
            "description": nl_explanation,
            "risk_score": score,
            "indicators": explanation.get("top_features", []),
            "recommended_actions": self._get_recommended_actions(level, anomaly),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log.warning("alert.created", alert_id=alert["alert_id"], user_id=user_id, severity=severity)
        return alert

    @staticmethod
    def _get_recommended_actions(level: str, anomaly: dict[str, Any]) -> list[str]:
        """
        Generate recommended response actions based on risk level.

        These are suggestions for the SOC analyst — the system never
        takes autonomous blocking actions (human-in-the-loop principle).
        """
        actions = ["Review user's recent activity timeline"]

        if level == RISK_CRITICAL:
            actions.extend([
                "Immediately escalate to incident response team",
                "Consider temporary access restriction pending investigation",
                "Preserve all relevant logs for forensic analysis",
                "Contact user's direct supervisor",
            ])
        elif level == RISK_HIGH:
            actions.extend([
                "Schedule investigation within 4 hours",
                "Review file access patterns for data exfiltration indicators",
                "Check for concurrent sessions from unusual locations",
            ])
        else:
            actions.append("Monitor for continued anomalous behaviour")

        return actions

    @staticmethod
    def _score_to_level(score: float) -> str:
        """Map a numeric risk score to a categorical level."""
        for level, (low, high) in RISK_THRESHOLDS.items():
            if low <= score < high:
                return level
        return RISK_CRITICAL if score >= 75 else RISK_LOW
