"""
federated â€“ Flower-based federated learning for the ITDS autoencoder.

Each department/team trains its autoencoder locally and only shares
model *weights* with the central server.  Raw behavioural data never
leaves the client boundary.

Strategy: **FedAvg** (McMahan et al., 2017) â€” each client trains for
``local_epochs`` steps, then sends updated weights.  The server
averages them proportional to dataset size.
"""

from services.ml_engine.app.privacy.federated.client import ITDSFlowerClient, make_client_fn
from services.ml_engine.app.privacy.federated.server import (
    ITDSFedAvgStrategy,
    create_server_config,
)

__all__ = [
    "ITDSFlowerClient",
    "make_client_fn",
    "ITDSFedAvgStrategy",
    "create_server_config",
]

