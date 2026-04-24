from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    y_pred = (y_score >= float(threshold)).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "threshold": float(threshold),
    }
