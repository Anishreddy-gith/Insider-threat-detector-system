"""
Feed-forward Autoencoder detector (PyTorch 2.3).

Best for: **volumetric / distributional anomalies** — detecting when a
user's *overall behavioural profile* deviates from their personal
baseline (e.g. a finance analyst whose file-access volume jumps 10×
while their login pattern stays normal).

Architecture
────────────
    Encoder:  32 → 16 → 8  (bottleneck)
    Decoder:   8 → 16 → 32 (reconstruction)

Trained **per-user** on their 30-day baseline window with MSE loss +
Adam optimiser.  At inference the reconstruction error is compared to
the user's training-set error distribution; anything > 2σ above the
mean is flagged anomalous.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from app.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


# ── Network definition ──────────────────────────────────────────────

class _Autoencoder(nn.Module):
    """Symmetric feed-forward autoencoder: 32 → 16 → 8 → 16 → 32."""

    def __init__(self, input_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            # No activation — reconstruction targets are real-valued
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── Detector class ──────────────────────────────────────────────────

class AutoencoderDetector(BaseDetector):
    """
    Per-user feed-forward autoencoder.

    Parameters
    ----------
    input_dim : int
        Feature vector dimensionality (default 32 to match the
        encoder architecture; padded / projected from the raw
        feature set if necessary).
    lr : float
        Adam learning rate.
    epochs : int
        Training epochs per user baseline.
    batch_size : int
        Mini-batch size.
    sigma_threshold : float
        Number of standard deviations above mean training error
        that triggers an anomaly flag.
    """

    name = "autoencoder"

    def __init__(
        self,
        input_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        sigma_threshold: float = 2.0,
    ) -> None:
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.sigma_threshold = sigma_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _Autoencoder(input_dim).to(self.device)
        self._optimizer = Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss(reduction="none")

        # Fitted statistics from training data
        self._train_error_mean: float = 0.0
        self._train_error_std: float = 1.0

    # ── training ────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train on (N, 32) normal-behaviour feature vectors.

        After training, compute the mean + std of per-sample
        reconstruction error to set the anomaly threshold.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {X.shape[1]}"
            )

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        epoch_losses: list[float] = []

        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self._model(batch)
                loss = self._loss_fn(recon, batch).mean()
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                running_loss += loss.item() * len(batch)
            epoch_losses.append(running_loss / len(X))

        # Compute per-sample training error statistics
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon = self._model(X_t)
            per_sample_error = self._loss_fn(recon, X_t).mean(dim=1).cpu().numpy()
            self._train_error_mean = float(per_sample_error.mean())
            self._train_error_std = float(per_sample_error.std()) + 1e-10

        self._mark_fitted()
        logger.info(
            "autoencoder_trained",
            extra={
                "n_samples": len(X),
                "final_loss": round(epoch_losses[-1], 6),
                "error_mean": round(self._train_error_mean, 6),
                "error_std": round(self._train_error_std, 6),
            },
        )
        return {
            "n_samples": len(X),
            "epochs": self.epochs,
            "final_loss": epoch_losses[-1],
            "error_mean": self._train_error_mean,
            "error_std": self._train_error_std,
        }

    # ── prediction ──────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> list[dict[str, Any]]:
        """
        Compute reconstruction error and flag anomalies > 2σ.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted – call train() first")

        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon = self._model(X_t)
            # Per-sample MSE
            errors = self._loss_fn(recon, X_t).mean(dim=1).cpu().numpy()
            # Per-feature errors for explainability
            feat_errors = self._loss_fn(recon, X_t).cpu().numpy()

        results: list[dict[str, Any]] = []
        for i in range(len(X)):
            z = (errors[i] - self._train_error_mean) / self._train_error_std
            # Map z-score through sigmoid for a [0,1] anomaly score
            anomaly_score = float(1.0 / (1.0 + np.exp(-z)))
            is_anomalous = z > self.sigma_threshold

            # Per-feature reconstruction error (top contributors)
            fe = feat_errors[i]
            top_k = min(5, len(fe))
            top_indices = np.argsort(fe)[-top_k:][::-1]
            feature_errors = {
                f"feature_{j}": round(float(fe[j]), 6) for j in top_indices
            }

            results.append(
                {
                    "anomaly_score": round(anomaly_score, 6),
                    "is_anomaly": bool(is_anomalous),
                    "detector": self.name,
                    "reconstruction_error": round(float(errors[i]), 6),
                    "z_score": round(float(z), 4),
                    "feature_errors": feature_errors,
                }
            )
        return results

    # ── persistence ─────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        model_path = d / "autoencoder_detector.pt"
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "input_dim": self.input_dim,
                "train_error_mean": self._train_error_mean,
                "train_error_std": self._train_error_std,
                "meta": self._meta(),
            },
            model_path,
        )
        logger.info("autoencoder_saved", extra={"path": str(model_path)})
        return model_path

    def load(self, directory: str | Path) -> None:
        model_path = Path(directory) / "autoencoder_detector.pt"
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.input_dim = ckpt["input_dim"]
        self._model = _Autoencoder(self.input_dim).to(self.device)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._train_error_mean = ckpt["train_error_mean"]
        self._train_error_std = ckpt["train_error_std"]
        self._mark_fitted()
        logger.info("autoencoder_loaded", extra={"path": str(model_path)})
