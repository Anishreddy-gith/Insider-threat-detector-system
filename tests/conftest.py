"""
Root conftest — shared fixtures for every test layer.

Provides:
  • Synthetic data generators for normal and anomalous feature vectors
  • Path helpers for model persistence tests
  • Reusable numpy random state for reproducibility
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ["DEBUG"] = "false"


# ── Reproducibility ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic numpy random generator."""
    return np.random.default_rng(seed=42)


# ── Normal behaviour generator ───────────────────────────────────

@pytest.fixture(scope="session")
def normal_data_20d(rng: np.random.Generator) -> np.ndarray:
    """(500, 20) normal-behaviour feature matrix — moderate variance."""
    return rng.normal(loc=0.0, scale=1.0, size=(500, 20)).astype(np.float32)


@pytest.fixture(scope="session")
def normal_data_32d(rng: np.random.Generator) -> np.ndarray:
    """(500, 32) normal-behaviour feature matrix for Autoencoder."""
    return rng.normal(loc=0.0, scale=1.0, size=(500, 32)).astype(np.float32)


@pytest.fixture(scope="session")
def anomalous_data_20d(rng: np.random.Generator) -> np.ndarray:
    """(50, 20) anomalous feature vectors — shifted 4σ from normal."""
    return rng.normal(loc=4.0, scale=0.5, size=(50, 20)).astype(np.float32)


@pytest.fixture(scope="session")
def anomalous_data_32d(rng: np.random.Generator) -> np.ndarray:
    """(50, 32) anomalous feature vectors for Autoencoder."""
    return rng.normal(loc=4.0, scale=0.5, size=(50, 32)).astype(np.float32)


@pytest.fixture(scope="session")
def mixed_data_20d(
    normal_data_20d: np.ndarray,
    anomalous_data_20d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    (550, 20) mixed dataset and (550,) labels.
    Labels: 0=normal, 1=anomalous.
    """
    X = np.vstack([normal_data_20d, anomalous_data_20d])
    y = np.concatenate([
        np.zeros(len(normal_data_20d)),
        np.ones(len(anomalous_data_20d)),
    ])
    return X, y


@pytest.fixture(scope="session")
def normal_sequences(rng: np.random.Generator) -> np.ndarray:
    """(100, 24, 20) normal temporal sequences for LSTM."""
    return rng.normal(loc=0.0, scale=1.0, size=(100, 24, 20)).astype(np.float32)


@pytest.fixture(scope="session")
def anomalous_sequences(rng: np.random.Generator) -> np.ndarray:
    """(20, 24, 20) anomalous temporal sequences — last 4 hours spiked."""
    seqs = rng.normal(loc=0.0, scale=1.0, size=(20, 24, 20)).astype(np.float32)
    seqs[:, -4:, :] += 5.0  # Spike in the last 4 hours
    return seqs


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Temporary directory for model persistence tests."""
    d = tmp_path / "models"
    d.mkdir()
    return d
