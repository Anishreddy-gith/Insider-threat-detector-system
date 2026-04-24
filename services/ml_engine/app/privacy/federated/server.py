"""
Flower server â€” FedAvg aggregation strategy for the ITDS autoencoder.

The server never sees raw data.  It:
  1. Initialises the global model.
  2. Broadcasts parameters to selected clients each round.
  3. Receives updated weights + dataset sizes.
  4. Aggregates via **FedAvg** (weighted by ``num_examples``).
  5. Optionally evaluates the global model on a held-out server set.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


# Re-use the same architecture so initial parameters are compatible
from services.ml_engine.app.privacy.federated.client import _FederatedAutoencoder


class ITDSFedAvgStrategy(FedAvg):
    """
    FedAvg with ITDS-specific logging and optional server-side
    evaluation on a global validation set.

    Parameters
    ----------
    input_dim : int
        Autoencoder feature dimensionality.
    val_data : np.ndarray | None
        Optional (N, D) array for server-side evaluation.
    min_fit_clients : int
        Minimum number of clients to sample per round.
    min_available_clients : int
        Wait for at least this many clients before starting a round.
    num_rounds : int
        Total federated rounds (informational â€” actual count is in
        ``ServerConfig``).
    """

    def __init__(
        self,
        input_dim: int = 32,
        val_data: np.ndarray | None = None,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs: Any,
    ) -> None:
        # Build initial parameters from a fresh model
        model = _FederatedAutoencoder(input_dim)
        initial_params = [
            val.cpu().numpy() for val in model.state_dict().values()
        ]

        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=ndarrays_to_parameters(initial_params),
            **kwargs,
        )
        self.input_dim = input_dim
        self.val_data = val_data
        self._round_metrics: list[dict[str, Any]] = []

    # â”€â”€ Override: log per-round metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, FitRes]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate and log per-round training metrics."""
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Collect per-client losses
        client_losses: dict[str, float] = {}
        total_examples = 0
        weighted_loss = 0.0
        for _, fit_res in results:
            cid = fit_res.metrics.get("client_id", "unknown")
            loss = fit_res.metrics.get("loss", 0.0)
            client_losses[str(cid)] = float(loss)
            weighted_loss += float(loss) * fit_res.num_examples
            total_examples += fit_res.num_examples

        avg_loss = weighted_loss / max(total_examples, 1)
        round_info = {
            "round": server_round,
            "avg_loss": round(avg_loss, 6),
            "client_losses": client_losses,
            "total_examples": total_examples,
            "failures": len(failures),
        }
        self._round_metrics.append(round_info)

        logger.info(
            "fedavg_round_complete",
            extra=round_info,
        )

        # Server-side evaluation if val_data provided
        if aggregated_params is not None and self.val_data is not None:
            ndarrays = parameters_to_ndarrays(aggregated_params)
            val_loss = self._evaluate_global(ndarrays)
            round_info["val_loss"] = val_loss
            logger.info(
                "fedavg_server_eval",
                extra={"round": server_round, "val_loss": round(val_loss, 6)},
            )

        return aggregated_params, metrics

    def _evaluate_global(self, ndarrays: list[np.ndarray]) -> float:
        """Evaluate aggregated weights on the server validation set."""
        model = _FederatedAutoencoder(self.input_dim)
        params_dict = zip(model.state_dict().keys(), ndarrays)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        with torch.no_grad():
            X_t = torch.tensor(self.val_data, dtype=torch.float32)
            recon = model(X_t)
            loss = torch.nn.functional.mse_loss(recon, X_t).item()
        return loss

    def get_round_metrics(self) -> list[dict[str, Any]]:
        """Return collected per-round metrics for analysis."""
        return list(self._round_metrics)


def create_server_config(num_rounds: int = 10) -> ServerConfig:
    """
    Create a Flower ``ServerConfig``.

    Parameters
    ----------
    num_rounds : int
        Number of federated averaging rounds.
    """
    return ServerConfig(num_rounds=num_rounds)

