"""
Ensemble Model — Multi-Signal Anomaly Fusion
==============================================
Combines scores from the LSTM autoencoder, GNN, and rule-based detectors
into a single anomaly probability.

Why an ensemble?
  1. **Complementary signals**: The autoencoder captures temporal anomalies;
     the GNN captures structural anomalies; rules capture known-bad patterns.
     No single model catches everything.
  2. **Robustness**: An adversary who learns to evade one model still gets
     caught by the others.
  3. **Calibration**: Individual model scores have different distributions;
     the ensemble normalises and weights them.

Fusion strategy:
  • Learned weighted average (meta-learner) trained on analyst feedback
    (resolved alerts = ground truth labels).
  • Falls back to simple average if the meta-learner hasn't been trained yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalySignal:
    """A single anomaly score from one detector."""
    model_name: str
    score: float  # [0, 1]
    confidence: float = 1.0  # model self-assessed confidence
    feature_contributions: dict[str, float] = field(default_factory=dict)


class EnsembleScorer:
    """
    Meta-learner that fuses multiple anomaly signals into a single score.

    Training:
        After analysts resolve alerts (true positive / false positive),
        call ``fit()`` with the corresponding signals + labels.

    Inference:
        Call ``predict()`` with a list of ``AnomalySignal`` objects.
    """

    def __init__(self, model_names: list[str] | None = None) -> None:
        self.model_names = model_names or [
            "lstm_autoencoder",
            "gnn_entity_graph",
            "rule_engine",
        ]
        self._scaler = StandardScaler()
        self._meta_model = LogisticRegression(
            class_weight="balanced",  # insider threats are rare → class imbalance
            max_iter=1000,
        )
        self._is_fitted = False

        # Default weights when meta-learner hasn't been trained.
        # Autoencoder gets highest weight because temporal patterns
        # are the strongest signal in insider-threat literature.
        self._default_weights: dict[str, float] = {
            "lstm_autoencoder": 0.45,
            "gnn_entity_graph": 0.30,
            "rule_engine": 0.25,
        }

    def fit(
        self,
        signals_batch: list[list[AnomalySignal]],
        labels: list[int],
    ) -> None:
        """
        Train the meta-learner on historical analyst decisions.

        Args:
            signals_batch: List of signal-lists (one per sample).
            labels: 1 = true positive (real threat), 0 = false positive.
        """
        X = self._signals_to_features(signals_batch)
        X_scaled = self._scaler.fit_transform(X)
        self._meta_model.fit(X_scaled, labels)
        self._is_fitted = True

    def predict(self, signals: list[AnomalySignal]) -> dict[str, Any]:
        """
        Produce a fused anomaly score from multiple detector outputs.

        Returns:
            dict with 'score', 'confidence', 'model_contributions'.
        """
        if self._is_fitted:
            return self._predict_with_meta_model(signals)
        return self._predict_with_defaults(signals)

    def _predict_with_meta_model(self, signals: list[AnomalySignal]) -> dict[str, Any]:
        """Use the trained logistic regression meta-learner."""
        X = self._signals_to_features([signals])
        X_scaled = self._scaler.transform(X)
        prob = self._meta_model.predict_proba(X_scaled)[0, 1]  # P(threat)

        contributions = {}
        if hasattr(self._meta_model, "coef_"):
            coefs = self._meta_model.coef_[0]
            for i, name in enumerate(self.model_names):
                # Contribution = coefficient × standardised score
                contributions[name] = float(coefs[i] * X_scaled[0, i])

        return {
            "score": float(prob),
            "confidence": float(np.mean([s.confidence for s in signals])),
            "model_contributions": contributions,
            "method": "meta_learner",
        }

    def _predict_with_defaults(self, signals: list[AnomalySignal]) -> dict[str, Any]:
        """Weighted average fallback when meta-learner isn't trained."""
        total_weight = 0.0
        weighted_score = 0.0
        contributions: dict[str, float] = {}

        for signal in signals:
            w = self._default_weights.get(signal.model_name, 0.1)
            weighted_score += w * signal.score * signal.confidence
            total_weight += w
            contributions[signal.model_name] = signal.score * w

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0

        return {
            "score": float(min(final_score, 1.0)),
            "confidence": float(np.mean([s.confidence for s in signals])),
            "model_contributions": contributions,
            "method": "weighted_average",
        }

    def _signals_to_features(
        self, signals_batch: list[list[AnomalySignal]]
    ) -> np.ndarray:
        """Convert signal lists to a feature matrix for the meta-learner."""
        rows = []
        for signals in signals_batch:
            signal_map = {s.model_name: s for s in signals}
            row = []
            for name in self.model_names:
                s = signal_map.get(name)
                row.append(s.score * s.confidence if s else 0.0)
            rows.append(row)
        return np.array(rows, dtype=np.float64)
