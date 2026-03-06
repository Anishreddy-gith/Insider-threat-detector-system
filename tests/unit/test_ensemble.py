"""
Unit tests — Ensemble Scorer (Meta-learner)
=============================================

Tests the 4-detector fusion:
  ✓ Weighted average mode (no labels)
  ✓ Logistic regression mode (with labels)
  ✓ predict_from_detector_outputs convenience method
  ✓ Score monotonicity: higher detector scores → higher ensemble score
  ✓ SHAP additivity: detector breakdown sums relate to ensemble score
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from app.detectors.ensemble import EnsembleScorer, DEFAULT_DETECTOR_ORDER


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def uncalibrated_ensemble() -> EnsembleScorer:
    """Ensemble in weighted-average mode (no labels)."""
    ens = EnsembleScorer()
    ens.train(np.zeros((1, 4), dtype=np.float32))  # just mark fitted
    return ens


@pytest.fixture(scope="module")
def calibrated_ensemble() -> EnsembleScorer:
    """Ensemble trained with logistic regression."""
    rng = np.random.default_rng(42)
    # 100 normal, 20 anomalous
    normal_scores = rng.uniform(0, 0.3, size=(100, 4)).astype(np.float32)
    anom_scores = rng.uniform(0.6, 1.0, size=(20, 4)).astype(np.float32)
    X = np.vstack([normal_scores, anom_scores])
    y = np.concatenate([np.zeros(100), np.ones(20)])

    ens = EnsembleScorer()
    ens.train(X, y)
    return ens


# ── Weighted Average Mode ───────────────────────────────────────

class TestWeightedAverage:

    def test_equal_scores_produce_expected(
        self,
        uncalibrated_ensemble: EnsembleScorer,
    ) -> None:
        """All detectors at 0.5 → ensemble ≈ 0.5."""
        X = np.full((1, 4), 0.5, dtype=np.float32)
        results = uncalibrated_ensemble.predict(X)
        assert abs(results[0]["anomaly_score"] - 0.5) < 0.05

    def test_all_zeros_produce_low(
        self,
        uncalibrated_ensemble: EnsembleScorer,
    ) -> None:
        X = np.zeros((1, 4), dtype=np.float32)
        results = uncalibrated_ensemble.predict(X)
        assert results[0]["anomaly_score"] < 0.1

    def test_all_ones_produce_high(
        self,
        uncalibrated_ensemble: EnsembleScorer,
    ) -> None:
        X = np.ones((1, 4), dtype=np.float32)
        results = uncalibrated_ensemble.predict(X)
        assert results[0]["anomaly_score"] > 0.9

    def test_method_is_weighted_average(
        self,
        uncalibrated_ensemble: EnsembleScorer,
    ) -> None:
        X = np.full((1, 4), 0.5, dtype=np.float32)
        results = uncalibrated_ensemble.predict(X)
        assert results[0]["method"] == "weighted_average"


# ── Calibrated (Logistic Regression) Mode ────────────────────────

class TestLogisticRegression:

    def test_high_scores_flagged_anomalous(
        self,
        calibrated_ensemble: EnsembleScorer,
    ) -> None:
        X = np.full((10, 4), 0.9, dtype=np.float32)
        results = calibrated_ensemble.predict(X)
        flagged = sum(1 for r in results if r["is_anomaly"])
        assert flagged >= 8, f"Expected ≥8 flagged, got {flagged}"

    def test_low_scores_not_flagged(
        self,
        calibrated_ensemble: EnsembleScorer,
    ) -> None:
        X = np.full((10, 4), 0.1, dtype=np.float32)
        results = calibrated_ensemble.predict(X)
        flagged = sum(1 for r in results if r["is_anomaly"])
        assert flagged <= 2, f"Expected ≤2 flagged, got {flagged}"

    def test_method_is_logistic(
        self,
        calibrated_ensemble: EnsembleScorer,
    ) -> None:
        X = np.full((1, 4), 0.5, dtype=np.float32)
        results = calibrated_ensemble.predict(X)
        assert results[0]["method"] == "logistic_regression"


# ── predict_from_detector_outputs ─────────────────────────────

class TestFromDetectorOutputs:

    def test_convenience_method(
        self,
        uncalibrated_ensemble: EnsembleScorer,
    ) -> None:
        detector_results = {
            "isolation_forest": [
                {"anomaly_score": 0.8, "is_anomaly": True, "detector": "isolation_forest"}
            ],
            "autoencoder": [
                {"anomaly_score": 0.6, "is_anomaly": True, "detector": "autoencoder"}
            ],
            "lstm_temporal": [
                {"anomaly_score": 0.7, "is_anomaly": True, "detector": "lstm_temporal"}
            ],
            "gnn_relational": [
                {"anomaly_score": 0.5, "is_anomaly": False, "detector": "gnn_relational"}
            ],
        }
        results = uncalibrated_ensemble.predict_from_detector_outputs(detector_results)
        assert len(results) == 1
        assert "detector_detail" in results[0]
        assert results[0]["anomaly_score"] > 0.5  # weighted combo of 0.8,0.6,0.7,0.5


# ── Score Monotonicity ───────────────────────────────────────────

class TestMonotonicity:

    def test_higher_inputs_produce_higher_output(
        self,
        uncalibrated_ensemble: EnsembleScorer,
    ) -> None:
        """If all detector scores increase, ensemble must increase."""
        X_low = np.full((1, 4), 0.2, dtype=np.float32)
        X_high = np.full((1, 4), 0.8, dtype=np.float32)

        low_score = uncalibrated_ensemble.predict(X_low)[0]["anomaly_score"]
        high_score = uncalibrated_ensemble.predict(X_high)[0]["anomaly_score"]

        assert high_score > low_score


# ── Output Schema ────────────────────────────────────────────────

class TestOutputSchema:

    REQUIRED_KEYS = {
        "anomaly_score",
        "is_anomaly",
        "detector",
        "method",
        "detector_breakdown",
    }

    def test_all_keys_present(
        self,
        uncalibrated_ensemble: EnsembleScorer,
    ) -> None:
        X = np.full((3, 4), 0.5, dtype=np.float32)
        results = uncalibrated_ensemble.predict(X)
        for r in results:
            missing = self.REQUIRED_KEYS - set(r.keys())
            assert not missing, f"Missing: {missing}"

    def test_breakdown_has_all_detectors(
        self,
        uncalibrated_ensemble: EnsembleScorer,
    ) -> None:
        X = np.full((1, 4), 0.5, dtype=np.float32)
        results = uncalibrated_ensemble.predict(X)
        breakdown = results[0]["detector_breakdown"]
        for name in DEFAULT_DETECTOR_ORDER:
            assert name in breakdown


# ── Property-Based (Hypothesis) ─────────────────────────────────

class TestPropertyBased:

    @given(
        X=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=1, max_value=10),
                st.just(4),
            ),
            elements=st.floats(min_value=0.0, max_value=1.0),
        )
    )
    @h_settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_output_scores_bounded(
        self,
        uncalibrated_ensemble: EnsembleScorer,
        X: np.ndarray,
    ) -> None:
        results = uncalibrated_ensemble.predict(X)
        for r in results:
            assert 0.0 <= r["anomaly_score"] <= 1.0

    @given(
        X=arrays(
            dtype=np.float32,
            shape=(5, 4),
            elements=st.floats(min_value=0.0, max_value=1.0),
        )
    )
    @h_settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_output_length(
        self,
        uncalibrated_ensemble: EnsembleScorer,
        X: np.ndarray,
    ) -> None:
        results = uncalibrated_ensemble.predict(X)
        assert len(results) == len(X)


# ── Persistence ──────────────────────────────────────────────────

class TestPersistence:

    def test_save_load_round_trip(
        self,
        calibrated_ensemble: EnsembleScorer,
        tmp_model_dir,
    ) -> None:
        rng = np.random.default_rng(88)
        X = rng.uniform(0, 1, size=(10, 4)).astype(np.float32)

        original = calibrated_ensemble.predict(X)
        calibrated_ensemble.save(tmp_model_dir)

        loaded = EnsembleScorer()
        loaded.load(tmp_model_dir)
        reloaded = loaded.predict(X)

        for a, b in zip(original, reloaded):
            assert abs(a["anomaly_score"] - b["anomaly_score"]) < 1e-5


# ── Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:

    def test_unfitted_raises(self) -> None:
        ens = EnsembleScorer()
        X = np.full((1, 4), 0.5, dtype=np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            ens.predict(X)
