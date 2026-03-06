"""
Ensemble Scorer — logistic regression meta-learner.

Combines the anomaly scores from all four detectors into a single
calibrated risk score in [0, 1] with per-detector breakdown.

Why logistic regression?
────────────────────────
Logistic regression is a *linear* meta-learner, which means:
  1. It learns a **weight per detector** that is directly interpretable:
     "the GNN contributes 35 % of the final risk score".
  2. It is **fast** (one matrix multiply) — latency-critical for
     real-time alerting.
  3. It **regularises** (C parameter) so that an over-confident
     detector does not dominate.
  4. It outputs calibrated probabilities via sigmoid, so the risk
     score is meaningful as a probability of insider activity.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

# Expected column order for reproducibility
DEFAULT_DETECTOR_ORDER = [
    "isolation_forest",
    "autoencoder",
    "lstm_temporal",
    "gnn_relational",
]


class EnsembleScorer(BaseDetector):
    """
    Meta-learner ensemble that fuses anomaly scores from N detectors.

    Modes of operation
    ──────────────────
    1. **Before calibration (no labels):**
       Weighted average using configurable per-detector weights.

    2. **After calibration (with labels):**
       Logistic regression learns optimal weights from a small set of
       analyst-confirmed incidents (semi-supervised).

    Parameters
    ----------
    detector_names : list[str]
        Ordered names of the constituent detectors.
    default_weights : dict[str, float] | None
        Weights for the un-calibrated fallback.
    C : float
        Logistic regression regularisation strength.
    """

    name = "ensemble"

    def __init__(
        self,
        detector_names: list[str] | None = None,
        default_weights: dict[str, float] | None = None,
        C: float = 1.0,
    ) -> None:
        self.detector_names = detector_names or list(DEFAULT_DETECTOR_ORDER)
        self.default_weights = default_weights or {
            "isolation_forest": 0.20,
            "autoencoder": 0.25,
            "lstm_temporal": 0.25,
            "gnn_relational": 0.30,
        }
        self.C = C

        self._meta_learner: LogisticRegression | None = None
        self._scaler = StandardScaler()
        self._calibrated = False

    # ── training (calibration) ──────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Calibrate the ensemble on stacked detection scores.

        Parameters
        ----------
        X : (N, K) – each row is [iso_score, ae_score, lstm_score, gnn_score]
        y : (N,)   – binary labels (1 = confirmed insider, 0 = benign)
        """
        if y is None:
            # No labels — mark fitted with default weights only
            self._mark_fitted()
            logger.info("ensemble_uncalibrated", extra={"mode": "weighted_average"})
            return {"mode": "weighted_average", "weights": self.default_weights}

        if X.ndim != 2 or X.shape[1] != len(self.detector_names):
            raise ValueError(
                f"Expected (N, {len(self.detector_names)}) array, "
                f"got shape {X.shape}"
            )

        X_scaled = self._scaler.fit_transform(X)

        self._meta_learner = LogisticRegression(
            C=self.C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
            class_weight="balanced",  # handle label imbalance
        )
        self._meta_learner.fit(X_scaled, y)
        self._calibrated = True

        # Extract learned weights
        coefs = self._meta_learner.coef_[0]
        learned_weights = {
            name: round(float(c), 4)
            for name, c in zip(self.detector_names, coefs)
        }

        self._mark_fitted()
        logger.info(
            "ensemble_calibrated",
            extra={
                "n_samples": len(X),
                "learned_weights": learned_weights,
                "accuracy": round(
                    float(self._meta_learner.score(X_scaled, y)), 4
                ),
            },
        )
        return {
            "mode": "logistic_regression",
            "n_samples": len(X),
            "learned_weights": learned_weights,
            "training_accuracy": float(self._meta_learner.score(X_scaled, y)),
        }

    # ── prediction ──────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> list[dict[str, Any]]:
        """
        Fuse detector scores into a unified risk score.

        Parameters
        ----------
        X : (N, K) – stacked anomaly scores from each detector
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted – call train() first")

        results: list[dict[str, Any]] = []
        for i in range(len(X)):
            row = X[i]
            breakdown = {
                name: round(float(row[j]), 6)
                for j, name in enumerate(self.detector_names)
                if j < len(row)
            }

            if self._calibrated and self._meta_learner is not None:
                row_scaled = self._scaler.transform(row.reshape(1, -1))
                prob = float(self._meta_learner.predict_proba(row_scaled)[0, 1])
                is_anomalous = prob > 0.5
                method = "logistic_regression"
            else:
                # Weighted average fallback
                weighted_sum = sum(
                    self.default_weights.get(name, 0.25) * float(row[j])
                    for j, name in enumerate(self.detector_names)
                    if j < len(row)
                )
                total_weight = sum(
                    self.default_weights.get(name, 0.25)
                    for j, name in enumerate(self.detector_names)
                    if j < len(row)
                )
                prob = weighted_sum / (total_weight + 1e-10)
                is_anomalous = prob > 0.5
                method = "weighted_average"

            results.append(
                {
                    "anomaly_score": round(prob, 6),
                    "is_anomaly": bool(is_anomalous),
                    "detector": self.name,
                    "method": method,
                    "detector_breakdown": breakdown,
                }
            )

        return results

    def predict_from_detector_outputs(
        self,
        detector_results: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """
        Convenience method: accept raw detector outputs and fuse them.

        Parameters
        ----------
        detector_results : dict[detector_name, list[prediction_dicts]]
            Keyed by detector name, each value is the list returned by
            that detector's ``predict()`` method.
        """
        # Determine N from the first detector
        first_key = next(iter(detector_results))
        n = len(detector_results[first_key])

        # Stack scores
        score_matrix = np.zeros((n, len(self.detector_names)), dtype=np.float32)
        for j, name in enumerate(self.detector_names):
            if name in detector_results:
                for i, pred in enumerate(detector_results[name]):
                    score_matrix[i, j] = pred["anomaly_score"]

        results = self.predict(score_matrix)

        # Augment with per-detector detail
        for i, result in enumerate(results):
            detail: dict[str, dict] = {}
            for name in self.detector_names:
                if name in detector_results and i < len(detector_results[name]):
                    pred = detector_results[name][i]
                    detail[name] = {
                        k: v for k, v in pred.items() if k != "detector"
                    }
            result["detector_detail"] = detail

        return results

    # ── persistence ─────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "ensemble_scorer.pkl"
        artefact = {
            "meta_learner": self._meta_learner,
            "scaler": self._scaler,
            "calibrated": self._calibrated,
            "detector_names": self.detector_names,
            "default_weights": self.default_weights,
            "C": self.C,
            "meta": self._meta(),
        }
        with open(path, "wb") as f:
            pickle.dump(artefact, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("ensemble_saved", extra={"path": str(path)})
        return path

    def load(self, directory: str | Path) -> None:
        path = Path(directory) / "ensemble_scorer.pkl"
        with open(path, "rb") as f:
            artefact = pickle.load(f)
        self._meta_learner = artefact["meta_learner"]
        self._scaler = artefact["scaler"]
        self._calibrated = artefact["calibrated"]
        self.detector_names = artefact["detector_names"]
        self.default_weights = artefact["default_weights"]
        self.C = artefact["C"]
        self._mark_fitted()
        logger.info("ensemble_loaded", extra={"path": str(path)})
