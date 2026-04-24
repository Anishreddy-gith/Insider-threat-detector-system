"""
Bidirectional LSTM Autoencoder for behavioural anomaly detection.

Architecture
────────────
Encoder:  Input(45) → BiLSTM(128) × 2 layers → latent(128)
Decoder:  latent → RepeatVector(24) → LSTM(128) × 2 → Linear(45)

The model learns "normal" behavioural patterns.  At inference, a high
reconstruction error signals a deviation from the learned baseline.

WHY bidirectional?
──────────────────
Insider threat sequences exhibit temporal dependencies in *both*
directions.  A file exfiltration at t=20 often correlates with
credential abuse at t=5 AND lateral movement at t=22.  Bidirectional
encoding captures both forward and backward context.

WHY reconstruction error (not classification)?
──────────────────────────────────────────────
Insider threat events are severely class-imbalanced (<0.1% positive).
Training a classifier would over-fit to the majority class.  Autoencoders
learn the normal manifold in an unsupervised manner, sidestepping the
label scarcity problem entirely.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 45,
        hidden_dim: int = 128,
        num_layers: int = 2,
        seq_len: int = 24,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # ── Encoder ─────────────────────────────────────────
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project bidirectional output (2×hidden) → hidden
        self.encoder_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Decoder ─────────────────────────────────────────
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sequences → fixed-size latent vector."""
        out, _ = self.encoder(x)                      # (B, T, 2H)
        latent = self.encoder_proj(out[:, -1, :])      # (B, H) – last step
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent → reconstructed sequence."""
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, H)
        decoded, _ = self.decoder(repeated)
        return self.output_proj(decoded)               # (B, T, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample anomaly score in [0, 1] via sigmoid of MSE."""
        with torch.no_grad():
            recon = self.forward(x)
            mse = ((x - recon) ** 2).mean(dim=(1, 2))
            return torch.sigmoid(mse)

    def per_feature_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature reconstruction error for interpretability."""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=1)      # (B, input_dim)


class TabularAutoencoder(nn.Module):
    """Feed-forward autoencoder for tabular feature vectors."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))

        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(x)
            return F.mse_loss(recon, x, reduction="none").mean(dim=1)
