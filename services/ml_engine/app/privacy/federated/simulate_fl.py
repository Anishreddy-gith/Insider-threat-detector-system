#!/usr/bin/env python3
"""
Federated learning simulation â€” 5 virtual clients with different
behavioural distributions.

Each client represents one department:
    0 â€“ Engineering   (high file-access, high app-switch rates)
    1 â€“ Finance       (high bytes-transferred, tight login hours)
    2 â€“ HR            (moderate everything, high sensitive-file ratio)
    3 â€“ IT-Ops        (very high network traffic, off-hours logins)
    4 â€“ Executives    (low volume, erratic hours)

Usage
â”€â”€â”€â”€â”€
    python -m app.privacy.federated.simulate_fl

The script uses ``flwr.simulation`` to run ``NUM_ROUNDS`` federated
rounds locally without requiring any network infrastructure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# â”€â”€ Make sure the project root is importable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import flwr as fl
from flwr.simulation import start_simulation

from services.ml_engine.app.privacy.federated.client import make_client_fn
from services.ml_engine.app.privacy.federated.server import ITDSFedAvgStrategy, create_server_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("fl_simulation")

# â”€â”€ Simulation parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
INPUT_DIM = 32
NUM_CLIENTS = 5
SAMPLES_PER_CLIENT = 500
NUM_ROUNDS = 10
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
SEED = 42


def _generate_department_data(
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Generate synthetic per-department behavioural distributions.

    Each department has a different mean vector and covariance diagonal,
    simulating real-world distributional heterogeneity.
    """
    departments: dict[str, dict] = {
        "engineering": {
            "mean_shift": np.array(
                [0.0] * 6 + [2.0, 0.5] + [3.0, 2.5] + [0.0] * 22,
                dtype=np.float32,
            ),
            "std_scale": 1.2,
        },
        "finance": {
            "mean_shift": np.array(
                [0.5, 1.0, -0.5, 0.2] + [3.0, 2.0] + [0.0] * 26,
                dtype=np.float32,
            ),
            "std_scale": 0.8,
        },
        "hr": {
            "mean_shift": np.array(
                [0.0] * 7 + [2.5] + [0.0] * 24,
                dtype=np.float32,
            ),
            "std_scale": 1.0,
        },
        "it_ops": {
            "mean_shift": np.array(
                [1.5] + [0.0] * 3 + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
                + [2.0] + [0.0] * 20,
                dtype=np.float32,
            ),
            "std_scale": 1.5,
        },
        "executives": {
            "mean_shift": np.array(
                [-0.5, -1.0] + [0.0] * 30,
                dtype=np.float32,
            ),
            "std_scale": 1.8,
        },
    }

    datasets: dict[str, np.ndarray] = {}
    for name, params in departments.items():
        mean = params["mean_shift"][:INPUT_DIM]
        # Pad if shorter than INPUT_DIM
        if len(mean) < INPUT_DIM:
            mean = np.pad(mean, (0, INPUT_DIM - len(mean)))
        std = params["std_scale"]
        data = rng.normal(loc=mean, scale=std, size=(SAMPLES_PER_CLIENT, INPUT_DIM))
        datasets[name] = data.astype(np.float32)

    return datasets


def main() -> None:
    rng = np.random.default_rng(SEED)
    datasets = _generate_department_data(rng)

    logger.info(
        "Generated %d department datasets: %s",
        len(datasets),
        {k: v.shape for k, v in datasets.items()},
    )

    # Hold out 10% from each department for server-side validation
    val_parts = []
    train_datasets: dict[str, np.ndarray] = {}
    for name, data in datasets.items():
        split = int(0.9 * len(data))
        train_datasets[name] = data[:split]
        val_parts.append(data[split:])
    val_data = np.vstack(val_parts)

    # Build strategy with server-side validation
    strategy = ITDSFedAvgStrategy(
        input_dim=INPUT_DIM,
        val_data=val_data,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    # Client factory
    client_fn = make_client_fn(
        datasets=train_datasets,
        input_dim=INPUT_DIM,
        local_epochs=LOCAL_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
    )

    # Run simulation
    logger.info(
        "Starting federated simulation: %d clients, %d rounds",
        NUM_CLIENTS,
        NUM_ROUNDS,
    )

    start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=create_server_config(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  FEDERATED LEARNING SIMULATION COMPLETE")
    print("=" * 60)
    for m in strategy.get_round_metrics():
        val = m.get("val_loss", "N/A")
        val_str = f"{val:.6f}" if isinstance(val, float) else val
        print(
            f"  Round {m['round']:>2d}  |  "
            f"avg_loss={m['avg_loss']:.6f}  |  "
            f"val_loss={val_str}  |  "
            f"clients={len(m['client_losses'])}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

