"""
detectors â€“ pluggable anomaly-detection modules for the ITDS ML engine.

Every detector implements the :class:`BaseDetector` ABC, making them
interchangeable in the ensemble scorer.
"""

from services.ml_engine.app.detectors.autoencoder import AutoencoderDetector
from services.ml_engine.app.detectors.base import BaseDetector
from services.ml_engine.app.detectors.ensemble import EnsembleScorer
from services.ml_engine.app.detectors.gnn_detector import GNNDetector
from services.ml_engine.app.detectors.isolation_forest import IsolationForestDetector
from services.ml_engine.app.detectors.lstm_detector import LSTMDetector

__all__ = [
    "BaseDetector",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "LSTMDetector",
    "GNNDetector",
    "EnsembleScorer",
]

