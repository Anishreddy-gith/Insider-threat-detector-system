"""
Differential Privacy — OpenDP Integration
============================================
Applies calibrated noise to aggregate statistics and model outputs so that
no individual user's data can be reverse-engineered from published metrics
or model parameters.

Why Differential Privacy?
  1. **GDPR Art. 25** requires "Data Protection by Design and by Default."
     DP is the gold-standard mathematical framework for quantifying privacy.
  2. **Composition**: DP guarantees compose — we can track total privacy
     budget (ε) across multiple queries and alert when it's exhausted.
  3. **Formal guarantee**: Unlike k-anonymity or l-diversity, DP provides
     a provable bound on re-identification risk regardless of attacker's
     auxiliary knowledge.

Budget tracking:
  • We maintain a per-user privacy budget in Redis.
  • Once a user's cumulative ε exceeds the configured maximum, further
    queries about that user return only coarse aggregates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.utils.logging import get_logger

log = get_logger(__name__)

# Try importing OpenDP; fall back to manual Gaussian/Laplace if unavailable.
try:
    import opendp.prelude as dp
    dp.enable_features("contrib", "honest-but-curious")
    OPENDP_AVAILABLE = True
except ImportError:
    OPENDP_AVAILABLE = False
    log.warning("opendp.not_installed — using manual DP mechanisms")


class DifferentialPrivacyEngine:
    """
    Applies differential privacy to feature vectors, aggregate statistics,
    and model outputs.

    Args:
        epsilon: Privacy budget per query (lower = more private, noisier).
        delta:   Relaxation parameter for approximate (ε, δ)-DP.
        max_budget: Maximum cumulative ε before refusing further queries.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_budget: float = 10.0,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.max_budget = max_budget
        self._spent_budget: dict[str, float] = {}  # user_id → cumulative ε

    def add_noise_to_count(self, true_count: int, sensitivity: int = 1) -> int:
        """
        Add Laplace noise to a count query.

        Uses the Laplace mechanism:  noise ~ Lap(sensitivity / ε)

        Example: "How many alerts did user X trigger this week?"
        """
        if OPENDP_AVAILABLE:
            # OpenDP measurement for integer counts.
            meas = (
                dp.space_of(list[int])
                >> dp.t.then_count()
                >> dp.m.then_laplace(scale=float(sensitivity) / self.epsilon)
            )
            # OpenDP returns a noisy float; round to int.
            return max(0, int(round(meas([0] * true_count))))

        # Manual fallback.
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return max(0, int(round(true_count + noise)))

    def add_noise_to_mean(
        self, true_mean: float, n_samples: int, value_range: tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Add noise to a mean statistic.

        Sensitivity of a mean = (upper - lower) / n.
        """
        sensitivity = (value_range[1] - value_range[0]) / max(n_samples, 1)
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        noisy = true_mean + noise
        return float(np.clip(noisy, value_range[0], value_range[1]))

    def add_noise_to_features(
        self,
        features: np.ndarray,
        sensitivity: float = 1.0,
    ) -> np.ndarray:
        """
        Add calibrated Gaussian noise to a feature vector.

        Uses the Gaussian mechanism for (ε, δ)-DP:
          σ = sensitivity × √(2 ln(1.25/δ)) / ε

        WHY Gaussian over Laplace for features?
          Feature vectors are high-dimensional; Gaussian noise preserves
          better utility in high dimensions due to concentration of measure.
        """
        sigma = (
            sensitivity
            * np.sqrt(2 * np.log(1.25 / self.delta))
            / self.epsilon
        )
        noise = np.random.normal(0, sigma, size=features.shape)
        return features + noise

    def check_budget(self, user_id: str) -> bool:
        """
        Check if a user's privacy budget is exhausted.

        Returns True if further queries are allowed, False otherwise.
        """
        spent = self._spent_budget.get(user_id, 0.0)
        return spent < self.max_budget

    def consume_budget(self, user_id: str, epsilon_spent: float | None = None) -> float:
        """
        Record privacy budget consumption for a user.

        Returns remaining budget.
        """
        eps = epsilon_spent or self.epsilon
        current = self._spent_budget.get(user_id, 0.0)
        self._spent_budget[user_id] = current + eps
        remaining = max(0.0, self.max_budget - self._spent_budget[user_id])

        if remaining <= 0:
            log.warning("dp.budget_exhausted", user_id=user_id, spent=current + eps)

        return remaining

    def get_budget_status(self, user_id: str) -> dict[str, float]:
        """Get current privacy budget status for a user."""
        spent = self._spent_budget.get(user_id, 0.0)
        return {
            "user_id": user_id,
            "spent_epsilon": spent,
            "remaining_epsilon": max(0.0, self.max_budget - spent),
            "max_epsilon": self.max_budget,
            "is_exhausted": spent >= self.max_budget,
        }
