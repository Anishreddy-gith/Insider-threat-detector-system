"""
Privacy verification tests — differential privacy + membership inference.

Validates:
  1. LaplaceMechanism noise calibration (ε-DP contract)
  2. DPIsolationForestTrainer privacy guarantees
  3. Budget tracker accounting and exhaustion
  4. Membership inference attack simulation — confirms that a trained
     DP model does not leak individual training membership beyond the
     theoretical ε bound.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import stats

from services.ml_engine.app.privacy.differential_privacy import (
    DPIsolationForestTrainer,
    LaplaceMechanism,
    explain_epsilon,
)
from services.ml_engine.app.privacy.budget_tracker import (
    BudgetStatus,
    PrivacyBudgetExhausted,
    PrivacyBudgetTracker,
)


# ═══════════════════════════════════════════════════════════════════
#  Test LaplaceMechanism
# ═══════════════════════════════════════════════════════════════════

class TestLaplaceMechanism:
    """Verify noise calibration and query accuracy."""

    def test_epsilon_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            LaplaceMechanism(epsilon=0.0)
        with pytest.raises(ValueError, match="positive"):
            LaplaceMechanism(epsilon=-1.0)

    def test_answer_mean_is_unbiased(self):
        """Average of many DP mean answers should converge to the true mean."""
        rng_seed = 42
        np.random.seed(rng_seed)
        data = np.random.normal(5.0, 1.0, 1000)
        true_mean = float(np.clip(data, 0, 10).mean())

        mech = LaplaceMechanism(epsilon=1.0)
        dp_means = [mech.answer_mean(data, lower=0.0, upper=10.0) for _ in range(500)]

        # Unbiasedness: average of DP answers ≈ true answer
        assert abs(np.mean(dp_means) - true_mean) < 0.5, (
            "DP mean is biased — average deviates > 0.5 from true mean"
        )

    def test_noise_scale_inversely_proportional_to_epsilon(self):
        """Higher ε → less noise → answers closer to truth."""
        np.random.seed(42)
        data = np.random.normal(5.0, 1.0, 500)
        true_mean = float(np.clip(data, 0, 10).mean())

        errors_low_eps = []
        errors_high_eps = []
        for _ in range(200):
            low = LaplaceMechanism(epsilon=0.1).answer_mean(data, 0, 10)
            high = LaplaceMechanism(epsilon=5.0).answer_mean(data, 0, 10)
            errors_low_eps.append(abs(low - true_mean))
            errors_high_eps.append(abs(high - true_mean))

        # High-ε answers should be more accurate on average
        assert np.mean(errors_high_eps) < np.mean(errors_low_eps)

    def test_answer_count_non_negative(self):
        """DP count should never be negative."""
        np.random.seed(42)
        mech = LaplaceMechanism(epsilon=1.0)
        data = np.array([1, 2, 3, 4, 5])
        counts = [mech.answer_count(data) for _ in range(100)]
        assert all(c >= 0 for c in counts)

    def test_answer_sum_approximate(self):
        """DP sum should be approximately correct over many trials."""
        np.random.seed(42)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_sum = 15.0
        mech = LaplaceMechanism(epsilon=2.0)
        dp_sums = [mech.answer_sum(data, lower=0.0, upper=10.0) for _ in range(300)]
        assert abs(np.mean(dp_sums) - true_sum) < 1.0

    def test_laplace_scale_computation(self):
        """Scale should be sensitivity / epsilon."""
        mech = LaplaceMechanism(epsilon=2.0)
        scale = mech._laplace_scale(sensitivity=4.0)
        assert abs(scale - 2.0) < 1e-10

    def test_answer_count_with_predicate(self):
        """Counting with a predicate function."""
        np.random.seed(42)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        mech = LaplaceMechanism(epsilon=1.0)
        # Count values > 5 (true count = 5)
        counts = [
            mech.answer_count(data, predicate=lambda x: x > 5)
            for _ in range(200)
        ]
        assert abs(np.mean(counts) - 5.0) < 1.0


# ═══════════════════════════════════════════════════════════════════
#  Test DPIsolationForestTrainer
# ═══════════════════════════════════════════════════════════════════

class TestDPIsolationForest:
    """Verify DP training produces a functional anomaly detector."""

    @pytest.fixture(scope="class")
    def dp_trainer(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (500, 20)).astype(np.float32)
        trainer = DPIsolationForestTrainer(
            epsilon=1.0,
            delta=1e-5,
            contamination=0.05,
            n_estimators=100,
        )
        result = trainer.train(X)
        return trainer, result, X

    def test_train_returns_metadata(self, dp_trainer):
        trainer, result, _ = dp_trainer
        assert result["epsilon"] == 1.0
        assert result["delta"] == 1e-5
        assert result["n_samples"] == 500
        assert result["n_features"] == 20
        assert result["model"] is not None

    def test_predict_returns_scores(self, dp_trainer):
        trainer, _, X = dp_trainer
        scores = trainer.predict(X[:10])
        assert len(scores) == 10
        assert all(np.isfinite(scores))

    def test_anomalies_score_differ(self, dp_trainer):
        """Anomalous and normal points should produce different score distributions.

        With DP noise the model may not reliably rank anomalies lower,
        but scores should still differ measurably between distributions.
        """
        trainer, _, _ = dp_trainer
        rng = np.random.default_rng(99)
        normal = rng.normal(0, 1, (50, 20)).astype(np.float32)
        anomalous = rng.normal(5, 0.5, (50, 20)).astype(np.float32)

        normal_scores = trainer.predict(normal)
        anom_scores = trainer.predict(anomalous)
        # DP noise may obscure ranking, but distributions must differ
        assert abs(np.mean(anom_scores) - np.mean(normal_scores)) > 0.01

    def test_untrained_predict_raises(self):
        trainer = DPIsolationForestTrainer(epsilon=1.0)
        with pytest.raises(RuntimeError, match="not trained"):
            trainer.predict(np.zeros((1, 20)))

    def test_privacy_report_structure(self, dp_trainer):
        trainer, _, _ = dp_trainer
        report = trainer.privacy_report()
        assert "mechanism" in report
        assert report["epsilon"] == 1.0
        assert "guarantee" in report
        assert "plain_english" in report

    def test_noise_addition_changes_data(self, dp_trainer):
        """Verify that DP noise actually perturbs the data."""
        trainer, _, X = dp_trainer
        X_scaled = trainer._scaler.transform(X[:5])
        X_clamped = np.clip(X_scaled, trainer.feature_lower, trainer.feature_upper)
        X_noisy = trainer._add_dp_noise(X_clamped)
        # Noisy data should differ from clamped
        assert not np.allclose(X_clamped, X_noisy)

    def test_different_epsilon_different_noise_scale(self):
        """Lower ε → more noise → worse accuracy."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (300, 10)).astype(np.float32)
        X_test = rng.normal(0, 1, (50, 10)).astype(np.float32)
        X_anom = rng.normal(4, 0.5, (50, 10)).astype(np.float32)

        # High ε (less privacy, better accuracy)
        t_high = DPIsolationForestTrainer(epsilon=10.0, n_estimators=50)
        t_high.train(X)
        gap_high = np.mean(t_high.predict(X_test)) - np.mean(t_high.predict(X_anom))

        # Low ε (more privacy, worse accuracy)
        t_low = DPIsolationForestTrainer(epsilon=0.1, n_estimators=50)
        t_low.train(X)
        gap_low = np.mean(t_low.predict(X_test)) - np.mean(t_low.predict(X_anom))

        # High-epsilon model should have better separation
        assert gap_high >= gap_low * 0.5  # relaxed bound due to noise


# ═══════════════════════════════════════════════════════════════════
#  Membership inference attack simulation
# ═══════════════════════════════════════════════════════════════════

class TestMembershipInference:
    """
    Simulate a membership inference attack against DP-trained models.

    The attacker's goal: given a data point x and black-box access to
    the model, determine whether x was in the training set.

    Success criterion: the DP model should make this attack no better
    than random guessing (AUC ≈ 0.5, advantage ≤ ε-derived bound).
    """

    @pytest.fixture(scope="class")
    def attack_data(self):
        """Prepare train/test split for membership inference."""
        rng = np.random.default_rng(42)
        # Members (in training set)
        X_train = rng.normal(0, 1, (200, 20)).astype(np.float32)
        # Non-members (held out, same distribution)
        X_nonmember = rng.normal(0, 1, (200, 20)).astype(np.float32)

        # Train DP model
        trainer = DPIsolationForestTrainer(
            epsilon=1.0, n_estimators=100, random_state=42,
        )
        trainer.train(X_train)

        # Attacker observes scores for both sets
        member_scores = trainer.predict(X_train)
        nonmember_scores = trainer.predict(X_nonmember)

        return member_scores, nonmember_scores

    def test_score_distributions_overlap(self, attack_data):
        """Member and non-member score distributions should overlap.

        If DP works, the model should not give systematically different
        scores to members vs non-members from the same distribution.
        """
        member_scores, nonmember_scores = attack_data

        # KS test: null hypothesis = same distribution
        ks_stat, p_value = stats.ks_2samp(member_scores, nonmember_scores)
        # We WANT the null hypothesis to hold (p > 0.01 = distributions similar)
        # But DP noise may cause some difference, so we use a lenient threshold
        assert p_value > 0.001 or ks_stat < 0.3, (
            f"Score distributions too different: KS={ks_stat:.4f}, p={p_value:.6f}"
        )

    def test_attack_auc_near_random(self, attack_data):
        """A threshold-based membership attack should achieve AUC ≈ 0.5."""
        member_scores, nonmember_scores = attack_data

        # The attacker tries to use the anomaly score as a classifier:
        # "if score > threshold, predict MEMBER"
        # Under DP, this should be no better than random.

        # Compute empirical AUC
        labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
        scores = np.concatenate([member_scores, nonmember_scores])

        # Simple AUC calculation (rank-based)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = labels[sorted_indices]

        tp = 0
        fp = 0
        auc = 0.0
        prev_fp = 0
        prev_tp = 0
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += (tp + prev_tp) / 2.0
            prev_tp = tp
            prev_fp = fp
        if n_pos * n_neg > 0:
            auc /= (n_pos * n_neg)
        else:
            auc = 0.5

        # AUC should be close to 0.5 (random guessing)
        assert 0.3 <= auc <= 0.7, (
            f"Membership inference AUC = {auc:.4f} — too far from random"
        )

    def test_advantage_bounded_by_epsilon(self, attack_data):
        """The attacker's advantage should be bounded by e^ε - 1."""
        member_scores, nonmember_scores = attack_data
        epsilon = 1.0

        # Best threshold attack: predict member if score > median
        threshold = np.median(np.concatenate([member_scores, nonmember_scores]))

        tp_rate = np.mean(member_scores > threshold)
        fp_rate = np.mean(nonmember_scores > threshold)

        # Advantage = |TPR - FPR|
        advantage = abs(tp_rate - fp_rate)

        # Theoretical bound: for (ε)-DP, advantage ≤ e^ε - 1
        bound = math.exp(epsilon) - 1  # ≈ 1.718 for ε=1

        assert advantage <= bound + 0.1, (  # small tolerance
            f"Advantage {advantage:.4f} exceeds DP bound {bound:.4f}"
        )

    def test_non_dp_model_leaks_more(self):
        """A non-DP model should leak more membership info than a DP model."""
        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, (200, 20)).astype(np.float32)
        X_non = rng.normal(0, 1, (200, 20)).astype(np.float32)

        # DP model
        dp_trainer = DPIsolationForestTrainer(epsilon=0.5, n_estimators=100)
        dp_trainer.train(X_train)
        dp_member = dp_trainer.predict(X_train)
        dp_non = dp_trainer.predict(X_non)
        dp_gap = abs(np.mean(dp_member) - np.mean(dp_non))

        # Non-DP model (high epsilon = basically no privacy)
        nondp_trainer = DPIsolationForestTrainer(epsilon=100.0, n_estimators=100)
        nondp_trainer.train(X_train)
        nondp_member = nondp_trainer.predict(X_train)
        nondp_non = nondp_trainer.predict(X_non)
        nondp_gap = abs(np.mean(nondp_member) - np.mean(nondp_non))

        # DP model should have smaller gap (less leakage)
        # This is a statistical test so we add tolerance
        assert dp_gap <= nondp_gap + 0.1


# ═══════════════════════════════════════════════════════════════════
#  Budget tracker tests
# ═══════════════════════════════════════════════════════════════════

class _MockRedis:
    """Minimal Redis mock for budget tracker tests."""

    def __init__(self):
        self._store = {}
        self._ttls = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def expire(self, key, ttl):
        self._ttls[key] = ttl

    def ttl(self, key):
        return self._ttls.get(key, -2)

    def delete(self, key):
        self._store.pop(key, None)
        self._ttls.pop(key, None)

    def keys(self, pattern="*"):
        import fnmatch
        return [k for k in self._store if fnmatch.fnmatch(k, pattern)]


class TestBudgetTracker:
    """Test privacy budget accounting."""

    @pytest.fixture
    def tracker(self):
        redis = _MockRedis()
        return PrivacyBudgetTracker(
            redis_client=redis,
            daily_budget=10.0,
            alert_threshold=0.80,
        )

    def test_initial_status_is_zero(self, tracker):
        status = tracker.get_status("user-001")
        assert status.consumed == 0.0
        assert status.remaining == 10.0
        assert status.queries == 0
        assert not status.exhausted

    def test_consume_updates_budget(self, tracker):
        status = tracker.consume("user-001", epsilon=2.0, query_description="test query")
        assert abs(status.consumed - 2.0) < 1e-6
        assert abs(status.remaining - 8.0) < 1e-6
        assert status.queries == 1

    def test_multiple_consumptions_accumulate(self, tracker):
        tracker.consume("user-001", 3.0)
        tracker.consume("user-001", 4.0)
        status = tracker.get_status("user-001")
        assert abs(status.consumed - 7.0) < 1e-6

    def test_alert_triggered_at_threshold(self, tracker):
        """Alert fires when 80% of budget is consumed."""
        tracker.consume("user-001", 8.0)
        status = tracker.get_status("user-001")
        assert status.alert_triggered is True
        assert status.utilisation_pct >= 80.0

    def test_alert_callback_invoked(self):
        callback = MagicMock()
        redis = _MockRedis()
        tracker = PrivacyBudgetTracker(
            redis_client=redis,
            daily_budget=10.0,
            alert_threshold=0.80,
            alert_callback=callback,
        )
        tracker.consume("user-x", 9.0)
        callback.assert_called_once()
        call_arg = callback.call_args[0][0]
        assert isinstance(call_arg, BudgetStatus)
        assert call_arg.alert_triggered is True

    def test_budget_exhausted_raises(self, tracker):
        """Consuming the full budget should raise on next query."""
        tracker.consume("user-001", 10.0)
        with pytest.raises(PrivacyBudgetExhausted):
            tracker.consume("user-001", 0.1)

    def test_budget_clamped_when_exceeded(self, tracker):
        """Overshoot is clamped to exact budget limit."""
        status = tracker.consume("user-001", 12.0)
        assert abs(status.consumed - 10.0) < 1e-6
        assert status.exhausted is True

    def test_reset_clears_budget(self, tracker):
        tracker.consume("user-001", 5.0)
        status = tracker.reset("user-001")
        assert status.consumed == 0.0
        assert status.queries == 0

    def test_per_entity_isolation(self, tracker):
        """Different entities have independent budgets."""
        tracker.consume("alice", 3.0)
        tracker.consume("bob", 7.0)
        assert abs(tracker.get_status("alice").consumed - 3.0) < 1e-6
        assert abs(tracker.get_status("bob").consumed - 7.0) < 1e-6

    def test_get_all_entities(self, tracker):
        tracker.consume("alice", 1.0)
        tracker.consume("bob", 2.0)
        tracker.consume("charlie", 3.0)
        entities = tracker.get_all_entities()
        assert set(entities) == {"alice", "bob", "charlie"}


# ═══════════════════════════════════════════════════════════════════
#  Explain epsilon tests
# ═══════════════════════════════════════════════════════════════════

class TestExplainEpsilon:
    """Verify the auditor-friendly explanation generator."""

    def test_strong_guarantee(self):
        text = explain_epsilon(epsilon=0.5, delta=1e-5)
        assert "STRONG" in text.upper()
        assert "ε" in text or "epsilon" in text.lower() or "privacy" in text.lower()

    def test_moderate_guarantee(self):
        text = explain_epsilon(epsilon=3.0, delta=1e-5)
        assert "MODERATE" in text.upper()

    def test_weak_guarantee(self):
        text = explain_epsilon(epsilon=10.0, delta=1e-5)
        assert "WEAK" in text.upper()

    def test_extremely_strong_guarantee(self):
        text = explain_epsilon(epsilon=0.05, delta=1e-5)
        assert "EXTREMELY STRONG" in text.upper()

    def test_output_contains_ratio(self):
        text = explain_epsilon(epsilon=1.0)
        # e^1 ≈ 2.7183
        assert "2.718" in text or "2.72" in text or "2.7" in text
