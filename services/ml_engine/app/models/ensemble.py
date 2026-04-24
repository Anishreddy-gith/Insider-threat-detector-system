"""
Ensemble scorer â€“ combines autoencoder and GNN anomaly signals.

Two modes:
1. **Weighted average** (default before analyst feedback is available)
2. **Meta-learner** (LogisticRegression trained on analyst-labelled
   true/false positives once enough labels accumulate)

The meta-learner calibrates raw model scores against human judgement,
reducing false-positive fatigue in the SOC.
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression

from services.ml_engine.app.config import get_settings

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)


class EnsembleScorer:
    def __init__(self):
        self.meta_learner: LogisticRegression | None = None
        self.weights = {
            "autoencoder": settings.ENSEMBLE_WEIGHTS_AUTOENCODER,
            "gnn": settings.ENSEMBLE_WEIGHTS_GNN,
        }

    def score(self, model_scores: dict[str, float]) -> float:
        """Combine sub-model scores into a single anomaly probability."""
        if self.meta_learner is not None:
            features = np.array(
                [[model_scores.get("autoencoder", 0.0), model_scores.get("gnn", 0.0)]]
            )
            return float(self.meta_learner.predict_proba(features)[0, 1])

        # Weighted average fallback
        total = sum(
            self.weights.get(k, 0.5) * v for k, v in model_scores.items()
        )
        norm = sum(self.weights.get(k, 0.5) for k in model_scores)
        return float(np.clip(total / norm, 0.0, 1.0))

    def train_meta_learner(
        self, X: np.ndarray, y: np.ndarray
    ) -> None:
        """Train the meta-learner on analyst-labelled data."""
        self.meta_learner = LogisticRegression(max_iter=500, class_weight="balanced")
        self.meta_learner.fit(X, y)
        logger.info("meta_learner_trained", extra={"samples": len(y)})

