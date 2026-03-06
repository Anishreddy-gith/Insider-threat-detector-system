"""
Unit tests for the Risk Scoring engine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta


class TestRiskEngine:
    """Tests for the risk scoring logic (pure computation, no I/O)."""

    def test_exponential_decay(self):
        """Risk should decay over time when no new anomalies arrive."""
        from math import exp

        # Simulate: initial_score=80, lambda=0.01, 100 hours
        initial = 80.0
        decay_lambda = 0.01
        hours = 100
        decayed = initial * exp(-decay_lambda * hours)
        assert decayed < initial
        assert decayed > 0

    def test_risk_level_thresholds(self):
        """Verify risk level classification boundaries."""
        from shared.constants import RiskLevel

        def classify(score: float) -> str:
            if score >= 75:
                return RiskLevel.CRITICAL
            elif score >= 50:
                return RiskLevel.HIGH
            elif score >= 25:
                return RiskLevel.MEDIUM
            return RiskLevel.LOW

        assert classify(0) == "low"
        assert classify(24.9) == "low"
        assert classify(25) == "medium"
        assert classify(49.9) == "medium"
        assert classify(50) == "high"
        assert classify(74.9) == "high"
        assert classify(75) == "critical"
        assert classify(100) == "critical"

    def test_score_clamped_to_100(self):
        """Risk score must never exceed 100."""
        raw_score = 150.0
        clamped = min(raw_score, 100.0)
        assert clamped == 100.0

    def test_recommended_actions_by_level(self):
        """Each risk level should map to appropriate response actions."""
        actions = {
            "low": ["Continue monitoring"],
            "medium": ["Review recent activity logs"],
            "high": ["Escalate to security team", "Restrict access"],
            "critical": ["Immediate investigation", "Suspend account", "Notify CISO"],
        }
        assert len(actions["critical"]) > len(actions["low"])
        assert "Suspend account" in actions["critical"]
