"""
BaseDetector – abstract interface shared by every detection module.

Every detector in the ITDS pipeline **must** subclass ``BaseDetector``
and implement three methods:

    train(X, …)         – learn a model from normal data
    predict(X)          – return a dict per sample with at least
                          ``anomaly_score`` and ``is_anomaly``
    save / load         – persist / restore weights

The common interface makes the ensemble scorer agnostic to the
underlying algorithm – it simply calls ``predict()`` on each detector
and merges the results.
"""

from __future__ import annotations

import abc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BaseDetector(abc.ABC):
    """Abstract base class for all ITDS anomaly detectors."""

    # Human-readable tag used in logs and ensemble output
    name: str = "base"

    # ── training ────────────────────────────────────────────────────

    @abc.abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Fit the model.

        Parameters
        ----------
        X : (N, D) or (N, T, D) feature array – layout depends on the
            concrete detector.
        y : optional labels (only used by supervised meta-learners).

        Returns
        -------
        A dict of training metrics (loss, duration, etc.).
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> list[dict[str, Any]]:
        """
        Score one or more samples.

        Returns
        -------
        A list of dicts, each containing **at least**::

            {
                "anomaly_score": float,  # 0–1, higher = more anomalous
                "is_anomaly": bool,
                "detector": str,         # self.name
            }

        Detectors may add extra keys (e.g. ``feature_importance``).
        """

    # ── persistence ─────────────────────────────────────────────────

    @abc.abstractmethod
    def save(self, directory: str | Path) -> Path:
        """Persist model artefacts to *directory*.  Return the main file path."""

    @abc.abstractmethod
    def load(self, directory: str | Path) -> None:
        """Restore model artefacts from *directory*."""

    # ── helpers ─────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""
        return getattr(self, "_fitted", False)

    def _mark_fitted(self) -> None:
        self._fitted = True

    def _meta(self) -> dict[str, Any]:
        """Standard metadata appended to every saved artefact."""
        return {
            "detector": self.name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} fitted={self.is_fitted}>"
