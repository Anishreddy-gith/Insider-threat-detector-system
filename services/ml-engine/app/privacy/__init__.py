"""
privacy – Privacy-preserving ML primitives for the ITDS ML engine.

Modules
───────
federated/          Flower-based federated learning (FedAvg)
differential_privacy  (ε, δ)-DP wrappers around detectors
budget_tracker      Per-user cumulative ε tracking with Redis
pia_generator       Automated Privacy Impact Assessment reports
"""

from app.privacy.budget_tracker import PrivacyBudgetTracker
from app.privacy.differential_privacy import (
    DPIsolationForestTrainer,
    LaplaceMechanism,
    explain_epsilon,
)
from app.privacy.pia_generator import PIAGenerator

__all__ = [
    "DPIsolationForestTrainer",
    "LaplaceMechanism",
    "explain_epsilon",
    "PrivacyBudgetTracker",
    "PIAGenerator",
]
