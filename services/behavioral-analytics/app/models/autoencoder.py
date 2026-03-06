"""
LSTM Autoencoder for Behavioural Anomaly Detection
====================================================
Learns the "normal" behaviour patterns of each user by reconstructing
sequences of activity features.  High reconstruction error → anomaly.

Architecture:
  ┌─────────┐     ┌────────────┐     ┌─────────┐     ┌────────────┐     ┌──────────┐
  │ Input   │────▶│ LSTM       │────▶│ Latent  │────▶│ LSTM       │────▶│ Output   │
  │ Seq     │     │ Encoder    │     │ Space   │     │ Decoder    │     │ Recon    │
  └─────────┘     └────────────┘     └─────────┘     └────────────┘     └──────────┘

Why LSTM Autoencoder?
  1. **Temporal patterns**: Insider threats unfold over days/weeks.  LSTMs
     capture long-range dependencies (e.g. user starts downloading files
     late at night, a pattern emerging over two weeks).
  2. **Unsupervised**: We don't have labelled insider-threat datasets in
     production — autoencoders train on "normal" and flag deviations.
  3. **Per-user baselines**: Each user's reconstruction error distribution
     becomes their personal baseline, avoiding the "noisy neighbour"
     problem of a single global threshold.

Implementation notes:
  • Bidirectional encoder captures both forward and backward context.
  • Dropout between layers for regularisation (insider threat datasets
    are small → overfitting risk).
  • ``RepeatVector``-style approach: the encoder's final hidden state is
    repeated ``seq_len`` times and fed to the decoder.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM that compresses a feature sequence into a fixed-size
    latent vector.

    Args:
        input_dim:  Number of input features per timestep.
        hidden_dim: LSTM hidden size.
        num_layers: Stacked LSTM depth (2-3 is typical).
        dropout:    Inter-layer dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # captures context from both directions
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project bidirectional output (2 * hidden_dim) to latent_dim.
        self.fc_latent = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            latent: (batch, hidden_dim) — compressed representation.
        """
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        # Concatenate last forward & backward hidden states.
        h_forward = h_n[-2]  # last layer, forward
        h_backward = h_n[-1]  # last layer, backward
        h_cat = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden)
        latent = torch.relu(self.fc_latent(h_cat))
        return latent


class LSTMDecoder(nn.Module):
    """
    LSTM decoder that reconstructs the original sequence from the latent
    vector.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        seq_len: int = 24,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (batch, hidden_dim) — from encoder.
        Returns:
            reconstruction: (batch, seq_len, output_dim)
        """
        # Repeat the latent vector across the sequence length.
        # This is the "RepeatVector" approach from Keras — simple and effective.
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden)
        decoded, _ = self.lstm(repeated)
        reconstruction = self.fc_out(decoded)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM autoencoder for sequential anomaly detection.

    Usage:
        model = LSTMAutoencoder(input_dim=45, hidden_dim=128, seq_len=24)
        recon = model(x)  # x: (batch, 24, 45)
        loss = F.mse_loss(recon, x)  # reconstruction error = anomaly score

    The per-sample reconstruction error (MSE) serves as the anomaly score:
      • Low error → behaviour matches learned "normal" patterns.
      • High error → behaviour deviates — potential insider threat.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        seq_len: int = 24,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, num_layers, seq_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode → decode → reconstruct."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample anomaly scores (MSE reconstruction error).

        Returns:
            scores: (batch,) — one scalar per sample, higher = more anomalous.
        """
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            # Mean squared error per sample (average over seq_len & features).
            mse = torch.mean((x - recon) ** 2, dim=(1, 2))
        return mse

    def get_feature_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-feature reconstruction error — used for explainability.

        Returns which features contributed most to the anomaly score,
        enabling analysts to understand *why* a user was flagged.

        Returns:
            errors: (batch, input_dim) — average error per feature.
        """
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            # Average over the time dimension to get per-feature error.
            feature_errors = torch.mean((x - recon) ** 2, dim=1)
        return feature_errors
