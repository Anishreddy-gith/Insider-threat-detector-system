"""
Unit tests — Autoencoder Detector
==================================

Key properties tested:
  ✓ Reconstruction error distribution is statistically different
    for anomalous vs normal inputs (two-sample KS test)
  ✓ Anomalous inputs produce higher z-scores
  ✓ Output schema correctness
  ✓ Persistence round-trip
  ✓ Hypothesis property-based tests
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats

from app.detectors.autoencoder import AutoencoderDetector


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_autoencoder() -> AutoencoderDetector:
    """Pre-trained Autoencoder on 500 normal 32-dim samples."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, size=(500, 32)).astype(np.float32)
    det = AutoencoderDetector(
        input_dim=32,
        lr=1e-3,
        epochs=30,
        batch_size=64,
        sigma_threshold=2.0,
    )
    det.train(X_train)
    return det


# ── Statistical Separation ───────────────────────────────────────

class TestReconstructionErrorDistribution:
    """
    The reconstruction error distribution for anomalous inputs must
    be statistically different from normal inputs.
    """

    def test_ks_test_rejects_null(
        self,
        trained_autoencoder: AutoencoderDetector,
    ) -> None:
        """
        Kolmogorov-Smirnov test: H₀ = "anomalous and normal 
        reconstruction errors come from the same distribution".
        We require p < 0.01 (reject H₀).
        """
        rng = np.random.default_rng(100)
        X_normal = rng.normal(0, 1, size=(200, 32)).astype(np.float32)
        X_anom = rng.normal(4.0, 0.5, size=(100, 32)).astype(np.float32)

        normal_results = trained_autoencoder.predict(X_normal)
        anom_results = trained_autoencoder.predict(X_anom)

        normal_errors = [r["reconstruction_error"] for r in normal_results]
        anom_errors = [r["reconstruction_error"] for r in anom_results]

        ks_stat, p_value = stats.ks_2samp(normal_errors, anom_errors)

        assert p_value < 0.01, (
            f"KS test failed to reject H₀ (p={p_value:.4f}). "
            f"Reconstruction error distributions are NOT statistically "
            f"different — the autoencoder is not learning."
        )

    def test_mean_error_anomalous_exceeds_normal(
        self,
        trained_autoencoder: AutoencoderDetector,
    ) -> None:
        """Mean reconstruction error must be higher for anomalies."""
        rng = np.random.default_rng(200)
        X_normal = rng.normal(0, 1, size=(200, 32)).astype(np.float32)
        X_anom = rng.normal(4.0, 0.5, size=(100, 32)).astype(np.float32)

        normal_errors = [
            r["reconstruction_error"]
            for r in trained_autoencoder.predict(X_normal)
        ]
        anom_errors = [
            r["reconstruction_error"]
            for r in trained_autoencoder.predict(X_anom)
        ]

        assert np.mean(anom_errors) > np.mean(normal_errors) * 1.5, (
            f"Anomalous mean error ({np.mean(anom_errors):.4f}) should be "
            f"≥1.5× the normal mean error ({np.mean(normal_errors):.4f})"
        )

    def test_welch_t_test_significant(
        self,
        trained_autoencoder: AutoencoderDetector,
    ) -> None:
        """Welch's t-test (unequal variance) for error means."""
        rng = np.random.default_rng(300)
        X_normal = rng.normal(0, 1, size=(200, 32)).astype(np.float32)
        X_anom = rng.normal(4.0, 0.5, size=(100, 32)).astype(np.float32)

        normal_errors = [
            r["reconstruction_error"]
            for r in trained_autoencoder.predict(X_normal)
        ]
        anom_errors = [
            r["reconstruction_error"]
            for r in trained_autoencoder.predict(X_anom)
        ]

        t_stat, p_value = stats.ttest_ind(
            anom_errors, normal_errors, equal_var=False
        )

        assert p_value < 0.01, (
            f"Welch t-test p={p_value:.6f} — not significant"
        )
        assert t_stat > 0, "Anomalous errors should be higher (positive t)"


# ── Z-Score and Anomaly Flagging ────────────────────────────────

class TestZScoreAndFlagging:
    """Verify that the sigma threshold correctly flags anomalies."""

    def test_anomalies_have_high_z_scores(
        self,
        trained_autoencoder: AutoencoderDetector,
    ) -> None:
        rng = np.random.default_rng(400)
        X_anom = rng.normal(4.0, 0.5, size=(50, 32)).astype(np.float32)
        results = trained_autoencoder.predict(X_anom)

        high_z = sum(1 for r in results if r["z_score"] > 2.0)
        assert high_z / len(results) >= 0.80, (
            f"Expected ≥80% of anomalies to have z > 2.0, "
            f"got {high_z}/{len(results)}"
        )

    def test_normal_data_mostly_not_flagged(
        self,
        trained_autoencoder: AutoencoderDetector,
    ) -> None:
        rng = np.random.default_rng(500)
        X_normal = rng.normal(0, 1, size=(200, 32)).astype(np.float32)
        results = trained_autoencoder.predict(X_normal)

        flagged = sum(1 for r in results if r["is_anomaly"])
        fp_rate = flagged / len(results)
        assert fp_rate <= 0.15, f"FP rate {fp_rate:.1%} too high"


# ── Output Schema ────────────────────────────────────────────────

class TestOutputSchema:
    """Validate prediction output structure."""

    REQUIRED_KEYS = {
        "anomaly_score",
        "is_anomaly",
        "detector",
        "reconstruction_error",
        "z_score",
        "feature_errors",
    }

    def test_all_keys_present(
        self,
        trained_autoencoder: AutoencoderDetector,
    ) -> None:
        rng = np.random.default_rng(600)
        X = rng.normal(0, 1, size=(3, 32)).astype(np.float32)
        results = trained_autoencoder.predict(X)

        for r in results:
            missing = self.REQUIRED_KEYS - set(r.keys())
            assert not missing, f"Missing keys: {missing}"

    def test_score_in_unit_interval(
        self,
        trained_autoencoder: AutoencoderDetector,
    ) -> None:
        rng = np.random.default_rng(700)
        X = rng.normal(0, 1, size=(50, 32)).astype(np.float32)
        results = trained_autoencoder.predict(X)

        for r in results:
            assert 0.0 <= r["anomaly_score"] <= 1.0

    def test_detector_name(
        self,
        trained_autoencoder: AutoencoderDetector,
    ) -> None:
        rng = np.random.default_rng(800)
        X = rng.normal(0, 1, size=(1, 32)).astype(np.float32)
        assert trained_autoencoder.predict(X)[0]["detector"] == "autoencoder"


# ── Property-Based (Hypothesis) ─────────────────────────────────

class TestPropertyBased:
    """Hypothesis property tests for the Autoencoder."""

    @given(
        X=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=1, max_value=20),
                st.just(32),
            ),
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
    def test_output_count_matches_input(
        self,
        trained_autoencoder: AutoencoderDetector,
        X: np.ndarray,
    ) -> None:
        results = trained_autoencoder.predict(X)
        assert len(results) == len(X)

    @given(
        X=arrays(
            dtype=np.float32,
            shape=(3, 32),
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
    def test_reconstruction_error_non_negative(
        self,
        trained_autoencoder: AutoencoderDetector,
        X: np.ndarray,
    ) -> None:
        """Reconstruction error (MSE) must be ≥ 0."""
        results = trained_autoencoder.predict(X)
        for r in results:
            assert r["reconstruction_error"] >= 0.0


# ── Persistence ──────────────────────────────────────────────────

class TestPersistence:
    """Save/load round-trip verification."""

    def test_save_load_round_trip(
        self,
        trained_autoencoder: AutoencoderDetector,
        tmp_model_dir: Path,
    ) -> None:
        rng = np.random.default_rng(900)
        X = rng.normal(0, 1, size=(10, 32)).astype(np.float32)

        original = trained_autoencoder.predict(X)
        trained_autoencoder.save(tmp_model_dir)

        loaded = AutoencoderDetector(input_dim=32)
        loaded.load(tmp_model_dir)
        reloaded = loaded.predict(X)

        for a, b in zip(original, reloaded):
            assert abs(a["anomaly_score"] - b["anomaly_score"]) < 1e-4


# ── Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary conditions."""

    def test_unfitted_raises(self) -> None:
        det = AutoencoderDetector(input_dim=32)
        X = np.zeros((5, 32), dtype=np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            det.predict(X)

    def test_wrong_input_dim_raises(self) -> None:
        det = AutoencoderDetector(input_dim=32)
        X = np.zeros((5, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="32 features"):
            det.train(X)

    def test_wrong_ndim_raises(self) -> None:
        det = AutoencoderDetector(input_dim=32)
        X = np.zeros(32, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            det.train(X)
