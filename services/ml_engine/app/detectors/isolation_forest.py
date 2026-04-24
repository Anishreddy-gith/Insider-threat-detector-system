"""
Isolation Forest detector with SHAP explainability.

Best for: **point anomalies** â€” single events that are statistically
unusual in the feature space (e.g. a login at 3 AM from a country the
user has never connected from).

How it works
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Isolation Forest builds an ensemble of random binary trees.  "Normal"
points require many splits to isolate; anomalies are isolated in few
splits, producing a shorter average path length â‡’ higher anomaly score.

SHAP integration
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
After each prediction we run ``shap.TreeExplainer`` to produce a
per-feature importance dict.  This tells the SOC analyst *which*
behavioural feature (login_hour, bytes_transferred, â€¦) drove the
anomaly flag, eliminating black-box frustration.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from services.ml_engine.app.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

# â”€â”€ Feature schema (matches ingestion-service behavioural vectors) â”€â”€
FEATURE_NAMES = [
    "login_hour",
    "login_count_24h",
    "session_duration_mean",
    "session_duration_std",
    "bytes_sent_total",
    "bytes_received_total",
    "files_accessed_count",
    "sensitive_files_ratio",
    "unique_apps_count",
    "app_switches_per_hour",
    "off_hours_ratio",
    "new_destination_ips",
    "failed_login_ratio",
    "avg_file_size",
    "network_request_count",
    "unique_domains",
    "download_upload_ratio",
    "peer_group_deviation",
    "days_since_last_activity",
    "cumulative_risk_score",
]


class IsolationForestDetector(BaseDetector):
    """
    Scikit-learn 1.5 Isolation Forest with SHAP-based explainability.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in the training set.
    n_estimators : int
        Number of isolation trees.
    random_state : int
        Seed for reproducibility.
    """

    name = "isolation_forest"

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            warm_start=False,
        )
        self._scaler = StandardScaler()
        self._explainer: shap.TreeExplainer | None = None
        self._background: np.ndarray | None = None

    # â”€â”€ training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Fit the Isolation Forest on (N, D) behavioural feature vectors.

        ``y`` is ignored â€” Isolation Forest is unsupervised.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}")

        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)

        # Keep a background sample for SHAP
        n_bg = min(100, len(X_scaled))
        idx = np.random.default_rng(self.random_state).choice(
            len(X_scaled), size=n_bg, replace=False
        )
        self._background = X_scaled[idx]
        self._explainer = shap.TreeExplainer(self._model, self._background)

        self._mark_fitted()
        logger.info(
            "isolation_forest_trained",
            extra={"n_samples": len(X), "n_features": X.shape[1]},
        )
        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "contamination": self.contamination,
        }

    # â”€â”€ prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def predict(self, X: np.ndarray) -> list[dict[str, Any]]:
        """
        Score each row in X.

        Returns a list of dicts with:
            anomaly_score   â€“ float in [0, 1]  (1 = most anomalous)
            is_anomaly      â€“ bool
            detector        â€“ "isolation_forest"
            feature_importance â€“ dict[feature_name, shap_value]
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted â€“ call train() first")

        X_scaled = self._scaler.transform(X)

        # sklearn score_samples: lower = more anomalous; range â‰ˆ [-1, 0]
        raw_scores = self._model.score_samples(X_scaled)
        # Normalise to [0, 1]: map min_possible â†’ 1, max_possible â†’ 0
        norm_scores = 1.0 - (raw_scores - raw_scores.min()) / (
            (raw_scores.max() - raw_scores.min()) + 1e-10
        )

        labels = self._model.predict(X_scaled)  # 1 = normal, -1 = anomaly

        # SHAP values
        shap_values = self._explainer.shap_values(X_scaled)

        results: list[dict[str, Any]] = []
        for i in range(len(X)):
            # Build feature importance dict
            feat_imp: dict[str, float] = {}
            sv = shap_values[i]
            n_feats = min(len(FEATURE_NAMES), len(sv))
            for j in range(n_feats):
                feat_imp[FEATURE_NAMES[j]] = round(float(sv[j]), 6)
            # Also handle extra unnamed features
            for j in range(n_feats, len(sv)):
                feat_imp[f"feature_{j}"] = round(float(sv[j]), 6)

            # Sort by absolute importance descending
            feat_imp = dict(
                sorted(feat_imp.items(), key=lambda kv: abs(kv[1]), reverse=True)
            )

            results.append(
                {
                    "anomaly_score": round(float(norm_scores[i]), 6),
                    "is_anomaly": bool(labels[i] == -1),
                    "detector": self.name,
                    "feature_importance": feat_imp,
                }
            )

        return results

    # â”€â”€ persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "isolation_forest.pkl"
        artefact = {
            "model": self._model,
            "scaler": self._scaler,
            "background": self._background,
            "meta": self._meta(),
        }
        with open(path, "wb") as f:
            pickle.dump(artefact, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("isolation_forest_saved", extra={"path": str(path)})
        return path

    def load(self, directory: str | Path) -> None:
        path = Path(directory) / "isolation_forest.pkl"
        with open(path, "rb") as f:
            artefact = pickle.load(f)
        self._model = artefact["model"]
        self._scaler = artefact["scaler"]
        self._background = artefact["background"]
        self._explainer = shap.TreeExplainer(self._model, self._background)
        self._mark_fitted()
        logger.info("isolation_forest_loaded", extra={"path": str(path)})

