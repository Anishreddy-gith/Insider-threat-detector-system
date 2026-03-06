"""
LSTM Temporal Detector (PyTorch 2.3).

Best for: **temporal / sequential anomalies** — behavioural patterns
that are individually normal but form an unusual *sequence*.  Example:
a user who always logs in → emails → browses → logs out, but one day
follows login → mass-download → USB-mount → logout.

Architecture
────────────
    LSTM(input_dim, hidden=128, layers=2, dropout=0.3)
    → Linear(128, input_dim)

The model is trained as a **next-step predictor**: given the first T−1
hourly feature vectors of a 24-hour window, predict hour T.  High
prediction error on a new observation means the sequence deviates
from learned temporal structure.

Gradient clipping (max_norm=1.0) prevents exploding gradients during
backpropagation through time.
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

class _LSTMPredictor(nn.Module):
    """Next-step LSTM predictor for hourly behavioural sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, D) – batch of hourly feature sequences

        Returns
        -------
        (B, T, D) – predicted next-step features for each time step
        """
        out, _ = self.lstm(x)        # (B, T, H)
        return self.fc(out)           # (B, T, D)


# ── Detector class ──────────────────────────────────────────────────

class LSTMDetector(BaseDetector):
    """
    Sequence-based anomaly detector using LSTM next-step prediction.

    Parameters
    ----------
    input_dim : int
        Per-hour feature vector dimensionality.
    seq_len : int
        Hourly window length (default 24 — one day).
    hidden_dim : int
        LSTM hidden state size.
    num_layers : int
        Stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers.
    lr : float
        Adam learning rate.
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    max_grad_norm : float
        Gradient clipping threshold.
    sigma_threshold : float
        Anomaly threshold in standard deviations of training error.
    """

    name = "lstm_temporal"

    def __init__(
        self,
        input_dim: int = 20,
        seq_len: int = 24,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 32,
        max_grad_norm: float = 1.0,
        sigma_threshold: float = 2.0,
    ) -> None:
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.sigma_threshold = sigma_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _LSTMPredictor(
            input_dim, hidden_dim, num_layers, dropout
        ).to(self.device)
        self._optimizer = Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss(reduction="none")

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
        Train on (N, T, D) sequences of hourly feature vectors.

        The model learns to predict ``X[:, 1:, :]`` from ``X[:, :-1, :]``
        (teacher-forced next-step prediction).
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D array (N, T, D), got shape {X.shape}")

        inputs = torch.tensor(X[:, :-1, :], dtype=torch.float32)   # (N, T-1, D)
        targets = torch.tensor(X[:, 1:, :], dtype=torch.float32)   # (N, T-1, D)

        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        epoch_losses: list[float] = []

        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            for batch_in, batch_tgt in loader:
                batch_in = batch_in.to(self.device)
                batch_tgt = batch_tgt.to(self.device)

                pred = self._model(batch_in)
                loss = self._loss_fn(pred, batch_tgt).mean()

                self._optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — critical for LSTM stability
                nn.utils.clip_grad_norm_(
                    self._model.parameters(), self.max_grad_norm
                )
                self._optimizer.step()
                running_loss += loss.item() * len(batch_in)

            epoch_losses.append(running_loss / len(X))

        # ── Compute training error distribution ─────────────────────
        self._model.eval()
        with torch.no_grad():
            all_in = inputs.to(self.device)
            all_tgt = targets.to(self.device)
            all_pred = self._model(all_in)
            # Per-sequence mean prediction error
            per_seq = self._loss_fn(all_pred, all_tgt).mean(dim=(1, 2)).cpu().numpy()
            self._train_error_mean = float(per_seq.mean())
            self._train_error_std = float(per_seq.std()) + 1e-10

        self._mark_fitted()
        logger.info(
            "lstm_trained",
            extra={
                "n_sequences": len(X),
                "seq_len": X.shape[1],
                "final_loss": round(epoch_losses[-1], 6),
                "error_mean": round(self._train_error_mean, 6),
            },
        )
        return {
            "n_sequences": len(X),
            "seq_len": X.shape[1],
            "epochs": self.epochs,
            "final_loss": epoch_losses[-1],
            "error_mean": self._train_error_mean,
            "error_std": self._train_error_std,
        }

    # ── prediction ──────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> list[dict[str, Any]]:
        """
        Score (N, T, D) sequences.

        The last time-step's prediction error is used as the primary
        anomaly signal (most recent hour deviation).
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted – call train() first")

        self._model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X[:, :-1, :], dtype=torch.float32).to(self.device)
            targets = torch.tensor(X[:, 1:, :], dtype=torch.float32).to(self.device)
            predictions = self._model(inputs)

            # Per-sequence MSE over all time steps
            per_seq = self._loss_fn(predictions, targets).mean(dim=(1, 2)).cpu().numpy()

            # Per-step error for the last step (most recent hour)
            last_step_error = (
                self._loss_fn(predictions[:, -1, :], targets[:, -1, :])
                .mean(dim=1)
                .cpu()
                .numpy()
            )

            # Per-feature error at the last step (for explainability)
            per_feat_last = (
                self._loss_fn(predictions[:, -1, :], targets[:, -1, :])
                .cpu()
                .numpy()
            )

        results: list[dict[str, Any]] = []
        for i in range(len(X)):
            z = (per_seq[i] - self._train_error_mean) / self._train_error_std
            anomaly_score = float(1.0 / (1.0 + np.exp(-z)))
            is_anomalous = z > self.sigma_threshold

            z_last = (last_step_error[i] - self._train_error_mean) / self._train_error_std

            # Top contributing features at the last time step
            fe = per_feat_last[i]
            top_k = min(5, len(fe))
            top_idx = np.argsort(fe)[-top_k:][::-1]
            temporal_features = {
                f"feature_{j}": round(float(fe[j]), 6) for j in top_idx
            }

            results.append(
                {
                    "anomaly_score": round(anomaly_score, 6),
                    "is_anomaly": bool(is_anomalous),
                    "detector": self.name,
                    "sequence_error": round(float(per_seq[i]), 6),
                    "last_step_error": round(float(last_step_error[i]), 6),
                    "last_step_z": round(float(z_last), 4),
                    "temporal_features": temporal_features,
                }
            )
        return results

    # ── persistence ─────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "lstm_detector.pt"
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "train_error_mean": self._train_error_mean,
                "train_error_std": self._train_error_std,
                "meta": self._meta(),
            },
            path,
        )
        logger.info("lstm_saved", extra={"path": str(path)})
        return path

    def load(self, directory: str | Path) -> None:
        path = Path(directory) / "lstm_detector.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = ckpt["input_dim"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.num_layers = ckpt["num_layers"]
        self.dropout = ckpt["dropout"]
        self._model = _LSTMPredictor(
            self.input_dim, self.hidden_dim, self.num_layers, self.dropout
        ).to(self.device)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._train_error_mean = ckpt["train_error_mean"]
        self._train_error_std = ckpt["train_error_std"]
        self._mark_fitted()
        logger.info("lstm_loaded", extra={"path": str(path)})
