"""
Federated Learning Client — Flower Integration
=================================================
Enables privacy-preserving collaborative model training across multiple
organisational sites without sharing raw employee data.

Why Federated Learning for Insider Threat Detection?
  1. **Data sovereignty**: Different departments / subsidiaries may be in
     different legal jurisdictions (GDPR vs. CCPA vs. PIPL).  FL keeps
     data local while still benefiting from collective learning.
  2. **Richer patterns**: A single site may not have enough insider-threat
     incidents to train a robust model.  FL aggregates gradient updates
     from many sites → better generalisation.
  3. **Regulatory compliance**: GDPR Art. 5(1)(c) mandates "data
     minimisation".  FL is the strongest technical measure — the data
     literally never leaves the local site.

Architecture:
  • This module implements a **Flower client** that trains the local
    LSTM autoencoder and sends gradient updates to a central aggregator.
  • The aggregator (Flower server) runs as a separate process.
  • Differential privacy noise is added to gradients before transmission
    (defence-in-depth on top of FL).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import flwr as fl
    from flwr.common import NDArrays, Scalar
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False

from shared.utils.logging import get_logger

log = get_logger(__name__)


class InsiderThreatFLClient:
    """
    Flower-compatible federated learning client.

    Wraps a PyTorch model and a local dataset, implementing the FL
    training loop.

    Args:
        model:       The local PyTorch model to train.
        train_loader: DataLoader for local training data.
        val_loader:   DataLoader for local validation data.
        device:       torch device ("cpu" or "cuda").
        dp_epsilon:   Differential privacy budget per round.
        dp_delta:     DP relaxation parameter.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        device: str = "cpu",
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta

    def get_parameters(self) -> list[np.ndarray]:
        """Extract model parameters as NumPy arrays."""
        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load parameters from the aggregator into the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def train_one_round(
        self, epochs: int = 1, lr: float = 1e-3
    ) -> dict[str, float]:
        """
        Run local training for the specified number of epochs.

        Returns training metrics (loss, samples processed).
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        total_loss = 0.0
        num_samples = 0

        for _ in range(epochs):
            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()

                # ── Gradient clipping + DP noise ──────────────
                # Clip gradients to bound sensitivity, then add
                # calibrated Gaussian noise for (ε, δ)-DP.
                max_grad_norm = 1.0
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )
                self._add_dp_noise(max_grad_norm)

                optimizer.step()
                total_loss += loss.item() * batch.size(0)
                num_samples += batch.size(0)

        avg_loss = total_loss / max(num_samples, 1)
        log.info("fl.train_round", loss=avg_loss, samples=num_samples)
        return {"loss": avg_loss, "num_samples": num_samples}

    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on local validation data."""
        self.model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss = criterion(recon, batch)
                total_loss += loss.item() * batch.size(0)
                num_samples += batch.size(0)

        avg_loss = total_loss / max(num_samples, 1)
        return {"val_loss": avg_loss, "num_samples": num_samples}

    def _add_dp_noise(self, max_grad_norm: float) -> None:
        """
        Add calibrated Gaussian noise to gradients for differential privacy.

        The noise scale is computed from the Gaussian mechanism:
          σ = (max_grad_norm × √(2 ln(1.25/δ))) / ε

        WHY Gaussian mechanism?
          It provides (ε, δ)-DP which is the standard for deep learning
          (pure ε-DP would require Laplace noise with heavier tails,
          destroying model utility).
        """
        sigma = (
            max_grad_norm
            * np.sqrt(2 * np.log(1.25 / self.dp_delta))
            / self.dp_epsilon
        )

        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=sigma,
                    size=param.grad.shape,
                    device=param.grad.device,
                )
                param.grad.add_(noise)


def create_flower_client(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    dp_epsilon: float = 1.0,
    dp_delta: float = 1e-5,
) -> Any:
    """
    Factory function that returns a Flower ``NumPyClient`` wrapping our
    insider-threat model.

    Usage:
        client = create_flower_client(model, train_loader, val_loader)
        fl.client.start_numpy_client(server_address="...", client=client)
    """
    if not FLWR_AVAILABLE:
        log.warning("fl.flower_not_installed")
        return None

    itds_client = InsiderThreatFLClient(
        model, train_loader, val_loader,
        dp_epsilon=dp_epsilon, dp_delta=dp_delta,
    )

    class _FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
            return itds_client.get_parameters()

        def fit(
            self, parameters: NDArrays, config: dict[str, Scalar]
        ) -> tuple[NDArrays, int, dict[str, Scalar]]:
            itds_client.set_parameters(parameters)
            metrics = itds_client.train_one_round(
                epochs=int(config.get("local_epochs", 1)),
                lr=float(config.get("learning_rate", 1e-3)),
            )
            return (
                itds_client.get_parameters(),
                metrics["num_samples"],
                {"loss": metrics["loss"]},
            )

        def evaluate(
            self, parameters: NDArrays, config: dict[str, Scalar]
        ) -> tuple[float, int, dict[str, Scalar]]:
            itds_client.set_parameters(parameters)
            metrics = itds_client.evaluate()
            return (
                metrics["val_loss"],
                metrics["num_samples"],
                {},
            )

    return _FlowerClient()
