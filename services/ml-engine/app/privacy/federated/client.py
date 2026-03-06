"""
Flower client — wraps the ITDS Autoencoder for federated training.

Each client represents one department (e.g. Engineering, Finance, HR).
The client:
  1. Receives the current global model parameters from the server.
  2. Trains locally on its own behavioural data for ``local_epochs``.
  3. Returns the updated weights + the number of training examples
     (used by FedAvg for weighted aggregation).

No raw data is transmitted — only model weight deltas.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ── Lightweight autoencoder (matches detectors/autoencoder.py) ──────

class _FederatedAutoencoder(nn.Module):
    """32 → 16 → 8 → 16 → 32  symmetric autoencoder."""

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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ── Flower NumPyClient ──────────────────────────────────────────────

class ITDSFlowerClient(NumPyClient):
    """
    Flower client for federated autoencoder training.

    Parameters
    ----------
    data : np.ndarray
        (N, 32) normal-behaviour feature vectors for this department.
    client_id : str
        Human-readable label (e.g. "engineering", "finance").
    input_dim : int
        Feature dimensionality.
    local_epochs : int
        Number of local SGD epochs per federated round.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    """

    def __init__(
        self,
        data: np.ndarray,
        client_id: str = "client-0",
        input_dim: int = 32,
        local_epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _FederatedAutoencoder(input_dim).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Store data
        self.data = data
        self.dataset = TensorDataset(
            torch.tensor(data, dtype=torch.float32)
        )
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )

    # ── Flower interface ────────────────────────────────────────────

    def get_parameters(self, config: dict[str, Any] | None = None) -> list[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Replace model parameters with those received from the server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """
        Train locally and return updated weights.

        Returns
        -------
        (updated_parameters, num_examples, metrics_dict)
        """
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0.0
        n_batches = 0
        for _epoch in range(self.local_epochs):
            for (batch,) in self.loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss = self.loss_fn(recon, batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(
            "federated_client_trained",
            extra={
                "client_id": self.client_id,
                "local_epochs": self.local_epochs,
                "avg_loss": round(avg_loss, 6),
                "n_samples": len(self.data),
            },
        )
        return (
            self.get_parameters(),
            len(self.data),
            {"loss": avg_loss, "client_id": self.client_id},
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the global model on local data."""
        self.set_parameters(parameters)
        self.model.eval()

        with torch.no_grad():
            X_t = torch.tensor(self.data, dtype=torch.float32).to(self.device)
            recon = self.model(X_t)
            loss = self.loss_fn(recon, X_t).item()

        return (
            loss,
            len(self.data),
            {"loss": loss, "client_id": self.client_id},
        )


# ── Client factory ──────────────────────────────────────────────────

def make_client_fn(
    datasets: dict[str, np.ndarray],
    input_dim: int = 32,
    local_epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    """
    Return a ``client_fn`` callable suitable for ``flwr.simulation``.

    Parameters
    ----------
    datasets : dict[client_id, (N, D) array]
        Pre-partitioned data keyed by department name.
    """
    client_ids = list(datasets.keys())

    def client_fn(cid: str) -> ITDSFlowerClient:
        # Flower simulation passes string CIDs "0", "1", …
        idx = int(cid) if cid.isdigit() else 0
        name = client_ids[idx % len(client_ids)]
        return ITDSFlowerClient(
            data=datasets[name],
            client_id=name,
            input_dim=input_dim,
            local_epochs=local_epochs,
            lr=lr,
            batch_size=batch_size,
        )

    return client_fn
