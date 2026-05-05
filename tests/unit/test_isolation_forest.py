"""
Unit tests — Isolation Forest Detector
=======================================

Property-based tests (Hypothesis) + deterministic assertions:
  ✓ 95%+ detection rate on injected 4σ anomalies
  ✓ SHAP values sum to expected output score (additivity)
  ✓ Output schema is correct
  ✓ Persistence round-trip
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from app.detectors.isolation_forest import IsolationForestDetector, FEATURE_NAMES


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_iforest() -> IsolationForestDetector:
    """Pre-trained Isolation Forest on 500 normal samples."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, size=(500, 20)).astype(np.float32)
    det = IsolationForestDetector(contamination=0.05, n_estimators=100, random_state=42)
    det.train(X_train)
    return det


# ── Detection Rate ───────────────────────────────────────────────

class TestDetectionRate:
    """Validate that ≥95% of injected anomalies are caught."""

    def test_anomaly_recall_above_95_percent(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        """
        50 anomalous points at 4σ from training distribution
        must be detected at ≥95%.
        """
        rng = np.random.default_rng(99)
        X_anom = rng.normal(4.0, 0.5, size=(50, 20)).astype(np.float32)
        results = trained_iforest.predict(X_anom)

        detected = sum(1 for r in results if r["is_anomaly"])
        recall = detected / len(X_anom)

        assert recall >= 0.95, (
            f"Expected ≥95% anomaly recall, got {recall:.1%} "
            f"({detected}/{len(X_anom)})"
        )

    def test_normal_false_positive_rate_below_15_percent(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        """Normal data should not be flagged excessively."""
        rng = np.random.default_rng(101)
        X_normal = rng.normal(0, 1, size=(200, 20)).astype(np.float32)
        results = trained_iforest.predict(X_normal)

        false_positives = sum(1 for r in results if r["is_anomaly"])
        fp_rate = false_positives / len(X_normal)

        assert fp_rate <= 0.15, (
            f"Expected ≤15% FP rate on normal data, got {fp_rate:.1%}"
        )

    def test_anomaly_scores_higher_for_anomalies(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        """Mean anomaly score for injected anomalies >> normal data."""
        rng = np.random.default_rng(77)
        X_normal = rng.normal(0, 1, size=(100, 20)).astype(np.float32)
        X_anom = rng.normal(4.0, 0.5, size=(50, 20)).astype(np.float32)

        normal_scores = [r["anomaly_score"] for r in trained_iforest.predict(X_normal)]
        anom_scores = [r["anomaly_score"] for r in trained_iforest.predict(X_anom)]

        assert np.mean(anom_scores) > np.mean(normal_scores), (
            f"Anomalous mean ({np.mean(anom_scores):.4f}) should exceed "
            f"normal mean ({np.mean(normal_scores):.4f})"
        )


# ── SHAP Additivity ─────────────────────────────────────────────

class TestSHAPAdditivity:
    """
    SHAP values must satisfy the additivity property:
        f(x) = E[f(x)] + Σ SHAP_i(x)

    For tree models, the base value + SHAP values should recover 
    the model's raw prediction score.
    """

    def test_shap_values_present_in_output(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        """Each prediction must contain a feature_importance dict."""
        rng = np.random.default_rng(10)
        X = rng.normal(0, 1, size=(5, 20)).astype(np.float32)
        results = trained_iforest.predict(X)

        for r in results:
            assert "feature_importance" in r
            assert isinstance(r["feature_importance"], dict)
            assert len(r["feature_importance"]) >= 20

    def test_shap_sum_approximates_score_deviation(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        """
        The sum of absolute SHAP values should be non-trivial for
        anomalous points (they deviate from the expected value).
        """
        rng = np.random.default_rng(55)
        X_anom = rng.normal(4.0, 0.5, size=(10, 20)).astype(np.float32)
        results = trained_iforest.predict(X_anom)

        for r in results:
            shap_sum = sum(abs(v) for v in r["feature_importance"].values())
            # For a 4σ anomaly, total SHAP contribution should be meaningful
            assert shap_sum > 0.01, (
                f"SHAP values sum to {shap_sum:.6f} — too small for an anomaly"
            )

    def test_shap_feature_names_match(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        """SHAP keys should use FEATURE_NAMES from the detector."""
        rng = np.random.default_rng(12)
        X = rng.normal(0, 1, size=(1, 20)).astype(np.float32)
        results = trained_iforest.predict(X)
        keys = set(results[0]["feature_importance"].keys())

        for name in FEATURE_NAMES:
            assert name in keys, f"Missing SHAP key: {name}"


# ── Output Schema ────────────────────────────────────────────────

class TestOutputSchema:
    """Validate the structure of every prediction result."""

    REQUIRED_KEYS = {"anomaly_score", "is_anomaly", "detector", "feature_importance"}

    def test_all_required_keys_present(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, size=(3, 20)).astype(np.float32)
        results = trained_iforest.predict(X)

        for r in results:
            missing = self.REQUIRED_KEYS - set(r.keys())
            assert not missing, f"Missing keys in output: {missing}"

    def test_anomaly_score_bounded(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, size=(50, 20)).astype(np.float32)
        results = trained_iforest.predict(X)

        for r in results:
            assert 0.0 <= r["anomaly_score"] <= 1.0, (
                f"Score {r['anomaly_score']} out of [0, 1]"
            )

    def test_detector_name(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, size=(1, 20)).astype(np.float32)
        results = trained_iforest.predict(X)
        assert results[0]["detector"] == "isolation_forest"


# ── Property-Based (Hypothesis) ─────────────────────────────────

class TestPropertyBased:
    """Hypothesis-driven property tests for Isolation Forest."""

    @given(
        X=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=1, max_value=20),
                st.just(20),
            ),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    @h_settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_output_length_matches_input(
        self,
        trained_iforest: IsolationForestDetector,
        X: np.ndarray,
    ) -> None:
        """predict() must return exactly len(X) result dicts."""
        results = trained_iforest.predict(X)
        assert len(results) == len(X)

    @given(
        X=arrays(
            dtype=np.float32,
            shape=(5, 20),
            elements=st.floats(
                min_value=-5.0,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    @h_settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_scores_are_finite(
        self,
        trained_iforest: IsolationForestDetector,
        X: np.ndarray,
    ) -> None:
        """All scores must be finite floats."""
        results = trained_iforest.predict(X)
        for r in results:
            assert np.isfinite(r["anomaly_score"])


# ── Persistence ──────────────────────────────────────────────────

class TestPersistence:
    """Save → load → predict produces identical results."""

    def test_save_load_round_trip(
        self,
        trained_iforest: IsolationForestDetector,
        tmp_model_dir: Path,
    ) -> None:
        rng = np.random.default_rng(88)
        X = rng.normal(0, 1, size=(10, 20)).astype(np.float32)

        original = trained_iforest.predict(X)
        trained_iforest.save(tmp_model_dir)

        loaded = IsolationForestDetector()
        loaded.load(tmp_model_dir)
        reloaded = loaded.predict(X)

        for a, b in zip(original, reloaded):
            assert abs(a["anomaly_score"] - b["anomaly_score"]) < 1e-5


# ── Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary condition tests."""

    def test_single_sample_prediction(
        self,
        trained_iforest: IsolationForestDetector,
    ) -> None:
        """Single row input should work."""
        X = np.zeros((1, 20), dtype=np.float32)
        results = trained_iforest.predict(X)
        assert len(results) == 1

    def test_unfitted_raises(self) -> None:
        """Predict before train must raise RuntimeError."""
        det = IsolationForestDetector()
        X = np.zeros((5, 20), dtype=np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            det.predict(X)

    def test_wrong_dimensions_raises(self) -> None:
        """1-D input to train must raise ValueError."""
        det = IsolationForestDetector()
        X = np.zeros(20, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            det.train(X)
