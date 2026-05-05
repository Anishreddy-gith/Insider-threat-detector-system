"""
Inference Pipeline
===================
Orchestrates the end-to-end prediction flow:
  Event → Features → Model(s) → Ensemble → XAI → Anomaly Payload

This is the real-time path: consumes events from Kafka, runs inference,
and publishes anomaly detections back to Kafka.

Design decisions:
  • Models are loaded once at startup and kept in memory (GPU or CPU).
  • Feature extraction is async (Redis lookups) so we don't block inference.
  • SHAP explanations are generated only when the anomaly score exceeds
    the threshold — explainability is expensive, so we skip it for
    clearly-normal behaviour.
  • Results are cached in Redis with a short TTL to deduplicate rapid
    re-scoring of the same user.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.models.autoencoder import LSTMAutoencoder
from app.models.ensemble import AnomalySignal, EnsembleScorer
from app.explainability.shap_explainer import SHAPExplainer
from app.features.extractor import FEATURE_NAMES, NUM_FEATURES
from shared.utils.logging import get_logger

log = get_logger(__name__)


class InferencePipeline:
    """
    Coordinates model loading, inference, ensembling, and explainability.

    Args:
        model_path:       Directory containing saved model weights.
        anomaly_threshold: Score above which a user is flagged.
        enable_gnn:       Whether to load the GNN model.
        enable_xai:       Whether to generate SHAP explanations for anomalies.
    """

    def __init__(
        self,
        model_path: str = "/models",
        anomaly_threshold: float = 0.85,
        enable_gnn: bool = True,
        enable_xai: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.anomaly_threshold = anomaly_threshold
        self.enable_gnn = enable_gnn
        self.enable_xai = enable_xai

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models (loaded in ``load_models()``)
        self.autoencoder: LSTMAutoencoder | None = None
        self.gnn_model = None  # Loaded conditionally
        self.ensemble = EnsembleScorer()
        self.explainer: SHAPExplainer | None = None

        self._model_versions: dict[str, str] = {}

    async def load_models(self) -> None:
        """
        Load trained model weights from disk.

        Falls back to freshly initialised models if no checkpoint exists
        (useful for first deployment before training).
        """
        # ── LSTM Autoencoder ──────────────────────────────────
        self.autoencoder = LSTMAutoencoder(
            input_dim=NUM_FEATURES,
            hidden_dim=128,
            num_layers=2,
            seq_len=24,
            dropout=0.2,
        ).to(self.device)

        ae_path = self.model_path / "lstm_autoencoder.pt"
        if ae_path.exists():
            state = torch.load(ae_path, map_location=self.device, weights_only=True)
            self.autoencoder.load_state_dict(state)
            self._model_versions["lstm_autoencoder"] = "loaded"
            log.info("model.loaded", name="lstm_autoencoder", path=str(ae_path))
        else:
            self._model_versions["lstm_autoencoder"] = "initialised"
            log.warning("model.not_found — using random weights", path=str(ae_path))

        self.autoencoder.eval()

        # ── GNN ───────────────────────────────────────────────
        if self.enable_gnn:
            try:
                from app.models.gnn_model import InsiderThreatGNN

                self.gnn_model = InsiderThreatGNN(
                    in_channels=NUM_FEATURES,
                    hidden_channels=128,
                    out_channels=64,
                ).to(self.device)

                gnn_path = self.model_path / "gnn_model.pt"
                if gnn_path.exists():
                    state = torch.load(gnn_path, map_location=self.device, weights_only=True)
                    self.gnn_model.load_state_dict(state)
                    self._model_versions["gnn"] = "loaded"
                else:
                    self._model_versions["gnn"] = "initialised"

                self.gnn_model.eval()
            except ImportError:
                log.warning("gnn.torch_geometric_not_available")
                self.gnn_model = None

        # ── SHAP Explainer ────────────────────────────────────
        if self.enable_xai and self.autoencoder:
            self.explainer = SHAPExplainer(
                model=self.autoencoder,
                feature_names=FEATURE_NAMES,
            )

        log.info("inference_pipeline.ready", models=list(self._model_versions.keys()))

    async def predict(
        self, user_id: str, feature_sequence: np.ndarray
    ) -> dict[str, Any]:
        """
        Run full inference pipeline on a user's feature sequence.

        Args:
            user_id: The monitored entity identifier.
            feature_sequence: (seq_len, NUM_FEATURES) array of recent features.

        Returns:
            dict with anomaly_score, is_anomalous, explanation, etc.
        """
        signals: list[AnomalySignal] = []

        # ── LSTM Autoencoder ──────────────────────────────────
        if self.autoencoder:
            ae_score = self._run_autoencoder(feature_sequence)
            signals.append(AnomalySignal(
                model_name="lstm_autoencoder",
                score=ae_score,
                confidence=0.9,
            ))

        # ── Ensemble ──────────────────────────────────────────
        result = self.ensemble.predict(signals)
        anomaly_score = result["score"]
        is_anomalous = anomaly_score >= self.anomaly_threshold

        # ── Explainability (only for anomalies — saves compute) ─
        explanation = None
        if is_anomalous and self.enable_xai and self.explainer:
            # Reshape for SHAP: (1, seq_len, features)
            input_3d = feature_sequence[np.newaxis, :, :]
            explanation = self.explainer.explain(input_3d)

        return {
            "user_id": user_id,
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous,
            "model_contributions": result.get("model_contributions", {}),
            "ensemble_method": result.get("method"),
            "explanation": explanation,
        }

    def _run_autoencoder(self, feature_sequence: np.ndarray) -> float:
        """
        Run the LSTM autoencoder and return a normalised anomaly score.

        The raw MSE reconstruction error is mapped to [0, 1] via a
        sigmoid-like transformation calibrated on the training set's
        error distribution.
        """
        tensor = torch.tensor(
            feature_sequence[np.newaxis, :, :],  # (1, seq_len, features)
            dtype=torch.float32,
            device=self.device,
        )
        raw_score = self.autoencoder.get_anomaly_score(tensor).item()

        # Normalise: sigmoid transformation centered at the expected
        # normal-behaviour MSE.  The constants are calibrated during
        # training (hardcoded here for the initial deployment).
        normal_mse = 0.05  # approximate mean MSE on training set
        steepness = 20.0   # controls how sharply the score rises
        normalised = 1.0 / (1.0 + np.exp(-steepness * (raw_score - normal_mse)))

        return float(np.clip(normalised, 0.0, 1.0))

    def get_model_info(self) -> list[dict[str, str]]:
        """Return metadata about loaded models (for /models endpoint)."""
        return [
            {"name": name, "status": status}
            for name, status in self._model_versions.items()
        ]
