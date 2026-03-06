"""
detectors – pluggable anomaly-detection modules for the ITDS ML engine.

Every detector implements the :class:`BaseDetector` ABC, making them
interchangeable in the ensemble scorer.
"""

from app.detectors.autoencoder import AutoencoderDetector
from app.detectors.base import BaseDetector
from app.detectors.ensemble import EnsembleScorer
from app.detectors.gnn_detector import GNNDetector
from app.detectors.isolation_forest import IsolationForestDetector
from app.detectors.lstm_detector import LSTMDetector

__all__ = [
    "BaseDetector",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "LSTMDetector",
    "GNNDetector",
    "EnsembleScorer",
]
