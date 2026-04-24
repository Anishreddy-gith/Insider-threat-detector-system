"""
privacy â€“ Privacy-preserving ML primitives for the ITDS ML engine.

Modules
â”€â”€â”€â”€â”€â”€â”€
federated/          Flower-based federated learning (FedAvg)
differential_privacy  (Îµ, Î´)-DP wrappers around detectors
budget_tracker      Per-user cumulative Îµ tracking with Redis
pia_generator       Automated Privacy Impact Assessment reports
"""

from services.ml_engine.app.privacy.budget_tracker import PrivacyBudgetTracker
from services.ml_engine.app.privacy.differential_privacy import (
    DPIsolationForestTrainer,
    LaplaceMechanism,
    explain_epsilon,
)
from services.ml_engine.app.privacy.pia_generator import PIAGenerator

__all__ = [
    "DPIsolationForestTrainer",
    "LaplaceMechanism",
    "explain_epsilon",
    "PrivacyBudgetTracker",
    "PIAGenerator",
]

