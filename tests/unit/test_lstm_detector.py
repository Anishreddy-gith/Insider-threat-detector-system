"""
Unit tests — LSTM Temporal Detector
=====================================

Tests temporal sequence anomaly detection:
  ✓ Temporal anomalies (spiked last 4 hours) are detected
  ✓ Normal sequences pass through mostly unflagged
  ✓ Output schema with temporal-specific fields
  ✓ Property-based tests for output invariants
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from services.ml_engine.app.detectors.lstm_detector import LSTMDetector


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_lstm() -> LSTMDetector:
    """
    Pre-trained LSTM on 100 normal (24, 20) sequences.
    Reduced epochs for test speed.
    """
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, size=(100, 24, 20)).astype(np.float32)
    det = LSTMDetector(
        input_dim=20,
        seq_len=24,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        lr=1e-3,
        epochs=15,
        batch_size=32,
        sigma_threshold=2.0,
    )
    det.train(X_train)
    return det


# ── Temporal Anomaly Detection ───────────────────────────────────

class TestTemporalAnomalyDetection:
    """Verify detection of sequences with spiked tail."""

    def test_spiked_sequences_detected(
        self,
        trained_lstm: LSTMDetector,
    ) -> None:
        """
        Sequences where the last 4 hours spike by +5σ should be
        detected at ≥70% (temporal anomalies are harder).
        """
        rng = np.random.default_rng(123)
        X_anom = rng.normal(0, 1, size=(30, 24, 20)).astype(np.float32)
        X_anom[:, -4:, :] += 5.0  # Spike last 4 hours

        results = trained_lstm.predict(X_anom)
        detected = sum(1 for r in results if r["is_anomaly"])
        recall = detected / len(X_anom)

        assert recall >= 0.70, (
            f"Expected ≥70% detection of temporal anomalies, "
            f"got {recall:.1%} ({detected}/{len(X_anom)})"
        )

    def test_normal_sequences_low_fp_rate(
        self,
        trained_lstm: LSTMDetector,
    ) -> None:
        """Normal sequences should have low false positive rate."""
        rng = np.random.default_rng(456)
        X_normal = rng.normal(0, 1, size=(50, 24, 20)).astype(np.float32)
        results = trained_lstm.predict(X_normal)

        flagged = sum(1 for r in results if r["is_anomaly"])
        fp_rate = flagged / len(X_normal)

        assert fp_rate <= 0.20, f"Normal FP rate {fp_rate:.1%} too high"

    def test_anomalous_scores_higher_than_normal(
        self,
        trained_lstm: LSTMDetector,
    ) -> None:
        rng = np.random.default_rng(789)
        X_normal = rng.normal(0, 1, size=(50, 24, 20)).astype(np.float32)
        X_anom = rng.normal(0, 1, size=(30, 24, 20)).astype(np.float32)
        X_anom[:, -4:, :] += 5.0

        normal_scores = [
            r["anomaly_score"] for r in trained_lstm.predict(X_normal)
        ]
        anom_scores = [
            r["anomaly_score"] for r in trained_lstm.predict(X_anom)
        ]

        assert np.mean(anom_scores) > np.mean(normal_scores)


# ── Output Schema ────────────────────────────────────────────────

class TestOutputSchema:
    """LSTM predictions must include temporal-specific fields."""

    REQUIRED_KEYS = {
        "anomaly_score",
        "is_anomaly",
        "detector",
        "sequence_error",
        "last_step_error",
        "last_step_z",
        "temporal_features",
    }

    def test_all_keys_present(
        self,
        trained_lstm: LSTMDetector,
    ) -> None:
        rng = np.random.default_rng(111)
        X = rng.normal(0, 1, size=(3, 24, 20)).astype(np.float32)
        results = trained_lstm.predict(X)

        for r in results:
            missing = self.REQUIRED_KEYS - set(r.keys())
            assert not missing, f"Missing keys: {missing}"

    def test_detector_name(
        self,
        trained_lstm: LSTMDetector,
    ) -> None:
        rng = np.random.default_rng(222)
        X = rng.normal(0, 1, size=(1, 24, 20)).astype(np.float32)
        assert trained_lstm.predict(X)[0]["detector"] == "lstm_temporal"

    def test_scores_bounded(
        self,
        trained_lstm: LSTMDetector,
    ) -> None:
        rng = np.random.default_rng(333)
        X = rng.normal(0, 1, size=(20, 24, 20)).astype(np.float32)
        results = trained_lstm.predict(X)
        for r in results:
            assert 0.0 <= r["anomaly_score"] <= 1.0


# ── Property-Based (Hypothesis) ─────────────────────────────────

class TestPropertyBased:
    """Hypothesis tests for LSTM detector invariants."""

    @given(
        scale=st.floats(min_value=0.1, max_value=3.0),
    )
    @h_settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_output_count_matches_batch_size(
        self,
        trained_lstm: LSTMDetector,
        scale: float,
    ) -> None:
        """Output length must equal input batch size."""
        rng = np.random.default_rng(444)
        batch = rng.normal(0, scale, size=(5, 24, 20)).astype(np.float32)
        results = trained_lstm.predict(batch)
        assert len(results) == 5


# ── Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:

    def test_unfitted_raises(self) -> None:
        det = LSTMDetector(input_dim=20, seq_len=24)
        X = np.zeros((5, 24, 20), dtype=np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            det.predict(X)

    def test_wrong_ndim_raises(self) -> None:
        det = LSTMDetector(input_dim=20, seq_len=24)
        X = np.zeros((5, 20), dtype=np.float32)
        with pytest.raises(ValueError, match="3-D"):
            det.train(X)

    def test_single_sequence(
        self,
        trained_lstm: LSTMDetector,
    ) -> None:
        """Single sequence prediction should work."""
        X = np.zeros((1, 24, 20), dtype=np.float32)
        results = trained_lstm.predict(X)
        assert len(results) == 1
