"""
Adaptive Threshold Manager — PID-controller-based auto-tuning.

Problem
───────
Static thresholds (LOW < 25, MEDIUM < 50, HIGH < 75) cause FP spikes
when the organisation's behavioural baseline shifts (e.g. during
quarter-end, audit season, or M&A due diligence).

Solution
────────
A discrete PID controller continuously adjusts the three thresholds
based on the observed false-positive rate:

    error(t) = fp_rate(t) − target_fp_rate
    P = Kp · error
    I = Ki · Σ error(τ), τ=0..t
    D = Kd · (error(t) − error(t−1))

    adjustment = P + I + D

If the FP rate exceeds the target, thresholds are *raised* (tightened)
so fewer alerts fire.  If FP rate drops below target, thresholds are
*lowered* (loosened) so genuine threats aren't missed.

All three thresholds move together by the same adjustment, clamped
to [THRESHOLD_MIN, THRESHOLD_MAX].
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ThresholdManager:
    """
    PID-controller-based adaptive threshold manager.

    Parameters
    ----------
    kp, ki, kd : float
        PID gains.
    target_fp_rate : float
        Desired FP rate (default 5 %).
    initial_thresholds : tuple[float, float, float]
        (low, medium, high) — starting thresholds on the 0-100 scale.
    """

    def __init__(
        self,
        kp: float | None = None,
        ki: float | None = None,
        kd: float | None = None,
        target_fp_rate: float | None = None,
        initial_thresholds: tuple[float, float, float] | None = None,
    ) -> None:
        self.kp = kp if kp is not None else settings.PID_KP
        self.ki = ki if ki is not None else settings.PID_KI
        self.kd = kd if kd is not None else settings.PID_KD
        self.target_fp_rate = (
            target_fp_rate
            if target_fp_rate is not None
            else settings.FP_TARGET_RATE
        )

        t = initial_thresholds or (
            settings.THRESHOLD_LOW,
            settings.THRESHOLD_MEDIUM,
            settings.THRESHOLD_HIGH,
        )
        self.threshold_low: float = t[0]
        self.threshold_medium: float = t[1]
        self.threshold_high: float = t[2]

        # PID state
        self._integral_error: float = 0.0
        self._last_error: float = 0.0
        self._history: list[dict[str, Any]] = []

    # ── core update ─────────────────────────────────────────────────

    def update(
        self,
        fp_count: int,
        total_alerts: int,
    ) -> dict[str, Any]:
        """
        Feed observed false-positive counts and get updated thresholds.

        Parameters
        ----------
        fp_count : int
            Number of confirmed false positives in the last window.
        total_alerts : int
            Total alerts fired in the last window.

        Returns
        -------
        dict with new thresholds + diagnostics.
        """
        if total_alerts == 0:
            fp_rate = 0.0
        else:
            fp_rate = fp_count / total_alerts

        error = fp_rate - self.target_fp_rate

        # PID terms
        p_term = self.kp * error
        self._integral_error += error
        # Anti-windup: clamp integral
        self._integral_error = max(-10.0, min(10.0, self._integral_error))
        i_term = self.ki * self._integral_error
        d_term = self.kd * (error - self._last_error)
        self._last_error = error

        adjustment = p_term + i_term + d_term

        # Positive adjustment → raise thresholds (fewer alerts)
        # Negative adjustment → lower thresholds (more alerts)
        old = (self.threshold_low, self.threshold_medium, self.threshold_high)

        self.threshold_low = self._clamp(
            self.threshold_low + adjustment * 100
        )
        self.threshold_medium = self._clamp(
            self.threshold_medium + adjustment * 100
        )
        self.threshold_high = self._clamp(
            self.threshold_high + adjustment * 100
        )

        # Maintain ordering invariant
        if self.threshold_medium <= self.threshold_low:
            self.threshold_medium = self.threshold_low + 5.0
        if self.threshold_high <= self.threshold_medium:
            self.threshold_high = self.threshold_medium + 5.0

        adjustment_made = (
            abs(self.threshold_low - old[0]) > 0.01
            or abs(self.threshold_medium - old[1]) > 0.01
            or abs(self.threshold_high - old[2]) > 0.01
        )

        result = {
            "threshold_low": round(self.threshold_low, 2),
            "threshold_medium": round(self.threshold_medium, 2),
            "threshold_high": round(self.threshold_high, 2),
            "fp_rate": round(fp_rate, 4),
            "error": round(error, 4),
            "p_term": round(p_term, 4),
            "i_term": round(i_term, 4),
            "d_term": round(d_term, 4),
            "adjustment": round(adjustment, 4),
            "adjustment_made": adjustment_made,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._history.append(result)

        if adjustment_made:
            logger.info(
                "thresholds_adjusted",
                extra={
                    "old": old,
                    "new": (
                        self.threshold_low,
                        self.threshold_medium,
                        self.threshold_high,
                    ),
                    "fp_rate": fp_rate,
                    "adjustment": adjustment,
                },
            )

        return result

    # ── scoring helper ──────────────────────────────────────────────

    def classify(self, score_0_1: float) -> str:
        """
        Classify a [0, 1] score into a risk level using current
        adaptive thresholds (which are on a 0-100 scale internally).
        """
        score_100 = score_0_1 * 100
        if score_100 >= self.threshold_high:
            return "CRITICAL"
        if score_100 >= self.threshold_medium:
            return "HIGH"
        if score_100 >= self.threshold_low:
            return "MEDIUM"
        return "LOW"

    # ── persistence helpers ─────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Serialise PID state for database / Redis persistence."""
        return {
            "threshold_low": self.threshold_low,
            "threshold_medium": self.threshold_medium,
            "threshold_high": self.threshold_high,
            "integral_error": self._integral_error,
            "last_error": self._last_error,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore PID state from database / Redis."""
        self.threshold_low = state["threshold_low"]
        self.threshold_medium = state["threshold_medium"]
        self.threshold_high = state["threshold_high"]
        self._integral_error = state.get("integral_error", 0.0)
        self._last_error = state.get("last_error", 0.0)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    # ── internal ────────────────────────────────────────────────────

    @staticmethod
    def _clamp(val: float) -> float:
        return max(settings.THRESHOLD_MIN, min(settings.THRESHOLD_MAX, val))
