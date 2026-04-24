"""
End-to-end inference pipeline (v2 â€” four-detector ensemble).

Orchestrates:
    feature vector â†’ IsolationForest + Autoencoder + LSTM + GNN â†’ Ensemble â†’ XAI

The pipeline owns detector lifecycle (load / predict / save).  Each
detector implements :class:`BaseDetector` and is therefore hot-swappable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from services.ml_engine.app.config import get_settings
from services.ml_engine.app.detectors import (
    AutoencoderDetector,
    EnsembleScorer,
    GNNDetector,
    IsolationForestDetector,
    LSTMDetector,
)

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.50:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


class InferencePipeline:
    """Load-once, predict-many pipeline using four pluggable detectors."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(settings.MODEL_DIR)

        # â”€â”€ Detectors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.isolation_forest = IsolationForestDetector(
            contamination=0.05,
            n_estimators=200,
        )
        self.autoencoder = AutoencoderDetector(
            input_dim=32,
            lr=1e-3,
            epochs=50,
            sigma_threshold=2.0,
        )
        self.lstm = LSTMDetector(
            input_dim=settings.FEATURE_DIM,
            seq_len=settings.AUTOENCODER_SEQ_LEN,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            max_grad_norm=1.0,
        )
        self.gnn = GNNDetector(
            input_dim=settings.FEATURE_DIM,
            hidden_dim=settings.GNN_HIDDEN_DIM,
            dropout=0.3,
        )
        self.ensemble = EnsembleScorer(
            detector_names=[
                "isolation_forest",
                "autoencoder",
                "lstm_temporal",
                "gnn_relational",
            ],
        )

    # â”€â”€ Model loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def load_models(self) -> None:
        """Load persisted weights for every detector that has them."""
        for det in (
            self.isolation_forest,
            self.autoencoder,
            self.lstm,
            self.gnn,
            self.ensemble,
        ):
            try:
                det.load(self.model_dir)
                logger.info(
                    "detector_loaded",
                    extra={"detector": det.name, "dir": str(self.model_dir)},
                )
            except FileNotFoundError:
                logger.warning(
                    "detector_weights_missing",
                    extra={"detector": det.name, "dir": str(self.model_dir)},
                )

    # â”€â”€ Single-entity prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def predict_single(
        self, entity_id: str, features: list[float]
    ) -> dict:
        """
        Run all available detectors on a single feature vector.

        Detectors that haven't been trained yet are skipped gracefully.
        """
        arr = np.array(features, dtype=np.float32)
        detector_results: dict[str, list[dict]] = {}

        # â”€â”€ Isolation Forest (point anomaly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.isolation_forest.is_fitted:
            iso_input = arr.reshape(1, -1)
            # Pad/truncate to match expected dimensionality
            if iso_input.shape[1] < 20:
                iso_input = np.pad(
                    iso_input,
                    ((0, 0), (0, 20 - iso_input.shape[1])),
                )
            detector_results["isolation_forest"] = (
                self.isolation_forest.predict(iso_input[:, :20])
            )

        # â”€â”€ Autoencoder (distributional anomaly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.autoencoder.is_fitted:
            ae_input = arr[:32].reshape(1, -1)
            if ae_input.shape[1] < 32:
                ae_input = np.pad(
                    ae_input,
                    ((0, 0), (0, 32 - ae_input.shape[1])),
                )
            detector_results["autoencoder"] = self.autoencoder.predict(ae_input)

        # â”€â”€ LSTM (temporal anomaly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.lstm.is_fitted:
            # Build a (1, T, D) sequence by tiling for single-vector input
            seq = np.tile(arr, (settings.AUTOENCODER_SEQ_LEN, 1))
            seq = seq[np.newaxis, :, :]   # (1, T, D)
            detector_results["lstm_temporal"] = self.lstm.predict(seq)

        # â”€â”€ GNN (relational anomaly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.gnn.is_fitted:
            gnn_input = arr.reshape(1, -1)
            detector_results["gnn_relational"] = self.gnn.predict(gnn_input)

        # â”€â”€ Ensemble fusion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if detector_results:
            if self.ensemble.is_fitted:
                ensemble_out = self.ensemble.predict_from_detector_outputs(
                    detector_results
                )
            else:
                # Fallback: average available scores
                scores = [
                    r[0]["anomaly_score"]
                    for r in detector_results.values()
                ]
                avg = float(np.mean(scores))
                ensemble_out = [
                    {
                        "anomaly_score": round(avg, 6),
                        "is_anomaly": avg > 0.5,
                        "detector": "ensemble",
                        "method": "simple_average",
                        "detector_breakdown": {
                            k: r[0]["anomaly_score"]
                            for k, r in detector_results.items()
                        },
                    }
                ]
            result = ensemble_out[0]
        else:
            result = {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "detector": "ensemble",
                "method": "no_detectors_available",
                "detector_breakdown": {},
            }

        # â”€â”€ Build response â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ensemble_score = result["anomaly_score"]

        # Explanation from Isolation Forest SHAP (if available)
        top_features = None
        explanation = None
        if (
            ensemble_score >= settings.SHAP_THRESHOLD
            and "isolation_forest" in detector_results
        ):
            feat_imp = detector_results["isolation_forest"][0].get(
                "feature_importance", {}
            )
            top_features = [
                {"feature": k, "contribution": v}
                for k, v in list(feat_imp.items())[:5]
            ]
            explanation = (
                f"Entity {entity_id} scored {ensemble_score:.2f}. "
                f"Top anomalous features: "
                f"{', '.join(f['feature'] for f in top_features)}."
            )

        return {
            "entity_id": entity_id,
            "anomaly_score": round(ensemble_score, 4),
            "risk_level": _risk_level(ensemble_score),
            "top_features": top_features,
            "explanation": explanation,
            "detector_breakdown": result.get("detector_breakdown", {}),
        }

