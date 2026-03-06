"""
Feedback API + async retraining loop.

Security analysts submit verdicts (TRUE_POSITIVE, FALSE_POSITIVE,
NEEDS_REVIEW) on alerts. The feedback is stored in PostgreSQL and
used in a weekly async job to fine-tune ensemble weights.

The retraining loop:
    1. Collect all feedback since last retrain
    2. Build (detector_scores, label) pairs from confirmed verdicts
    3. Fit a logistic regression meta-learner on the new data
    4. Export updated weights to a JSON file
    5. Publish a Kafka message so the ML engine can hot-reload

This loop is the core of the *continuously improving system* — it
closes the gap between the ML model's view and the analyst's
ground-truth judgement.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session
from app.models import Alert, Feedback
from app.schemas import FeedbackCreate, FeedbackResponse

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Feedback storage ────────────────────────────────────────────────

async def store_feedback(
    payload: FeedbackCreate,
    session: AsyncSession,
) -> Feedback:
    """
    Persist analyst feedback and update the parent alert's status.
    """
    # Fetch the parent alert
    result = await session.execute(
        select(Alert).where(Alert.id == UUID(payload.alert_id))
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise ValueError(f"Alert {payload.alert_id} not found")

    # Map verdict to alert status
    status_map = {
        "TRUE_POSITIVE": "RESOLVED",
        "FALSE_POSITIVE": "FALSE_POSITIVE",
        "NEEDS_REVIEW": "INVESTIGATING",
    }
    alert.status = status_map.get(payload.verdict, alert.status)
    alert.assigned_to = payload.analyst_id
    alert.updated_at = datetime.now(timezone.utc)

    # Store feedback record
    fb = Feedback(
        alert_id=UUID(payload.alert_id),
        analyst_id=payload.analyst_id,
        verdict=payload.verdict,
        confidence=payload.confidence,
        notes=payload.notes,
        risk_score_snapshot=alert.risk_score,
        detector_scores_snapshot=alert.detector_breakdown,
        user_id=alert.user_id,
        department=(alert.business_context or {}).get("department"),
    )
    session.add(fb)
    await session.flush()

    logger.info(
        "feedback_stored",
        extra={
            "feedback_id": str(fb.id),
            "alert_id": payload.alert_id,
            "verdict": payload.verdict,
            "analyst": payload.analyst_id,
        },
    )
    return fb


async def get_feedback_for_alert(
    alert_id: str,
    session: AsyncSession,
) -> list[Feedback]:
    """Fetch all feedback records for a given alert."""
    result = await session.execute(
        select(Feedback)
        .where(Feedback.alert_id == UUID(alert_id))
        .order_by(Feedback.created_at.desc())
    )
    return list(result.scalars().all())


async def get_feedback_stats(session: AsyncSession) -> dict[str, Any]:
    """Summary statistics of all feedback."""
    result = await session.execute(
        select(
            Feedback.verdict,
            func.count(Feedback.id).label("count"),
        ).group_by(Feedback.verdict)
    )
    rows = result.all()
    stats = {row.verdict: row.count for row in rows}
    total = sum(stats.values())
    stats["total"] = total
    if total > 0:
        stats["fp_rate"] = round(stats.get("FALSE_POSITIVE", 0) / total, 4)
        stats["tp_rate"] = round(stats.get("TRUE_POSITIVE", 0) / total, 4)
    return stats


# ── Async retraining loop ──────────────────────────────────────────

async def retrain_ensemble_weights() -> dict[str, Any]:
    """
    Weekly batch job that retrains ensemble weights from analyst feedback.

    Steps:
        1. Query confirmed TP + FP feedback from the last week
        2. Build feature matrix: [iso_forest_score, ae_score, lstm_score, gnn_score]
        3. Build label vector: 1 = TRUE_POSITIVE, 0 = FALSE_POSITIVE
        4. Fit logistic regression → extract coefficients as weights
        5. Normalize weights to sum to 1.0
        6. Save to JSON file for the ML engine to consume
    """
    async with async_session() as session:
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=settings.ANALYSIS_WINDOW_DAYS
        )

        result = await session.execute(
            select(Feedback)
            .where(
                Feedback.created_at >= cutoff,
                Feedback.verdict.in_(["TRUE_POSITIVE", "FALSE_POSITIVE"]),
            )
            .order_by(Feedback.created_at.asc())
        )
        feedbacks = list(result.scalars().all())

    if len(feedbacks) < settings.MIN_FEEDBACK_FOR_RETRAIN:
        logger.info(
            "retrain_skipped_insufficient_data",
            extra={"count": len(feedbacks), "min": settings.MIN_FEEDBACK_FOR_RETRAIN},
        )
        return {
            "status": "skipped",
            "reason": "insufficient_feedback",
            "count": len(feedbacks),
            "minimum": settings.MIN_FEEDBACK_FOR_RETRAIN,
        }

    # Build training data
    detector_names = ["isolation_forest", "autoencoder", "lstm_temporal", "gnn_relational"]
    X: list[list[float]] = []
    y: list[int] = []

    for fb in feedbacks:
        scores = fb.detector_scores_snapshot or {}
        row = []
        for name in detector_names:
            detector_data = scores.get(name, {})
            if isinstance(detector_data, dict):
                row.append(detector_data.get("anomaly_score", 0.0))
            elif isinstance(detector_data, (int, float)):
                row.append(float(detector_data))
            else:
                row.append(0.0)
        X.append(row)
        y.append(1 if fb.verdict == "TRUE_POSITIVE" else 0)

    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.int32)

    # Fit logistic regression
    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )
        clf.fit(X_arr, y_arr)

        # Extract and normalize weights
        raw_weights = np.abs(clf.coef_[0])
        weights = raw_weights / raw_weights.sum()
        weight_dict = {
            name: round(float(w), 4)
            for name, w in zip(detector_names, weights)
        }

        # Training metrics
        from sklearn.metrics import accuracy_score, roc_auc_score

        y_pred = clf.predict(X_arr)
        y_prob = clf.predict_proba(X_arr)[:, 1]
        accuracy = float(accuracy_score(y_arr, y_pred))
        try:
            auc = float(roc_auc_score(y_arr, y_prob))
        except ValueError:
            auc = None  # single class

    except ImportError:
        logger.warning("sklearn_not_available — using simple weight update")
        # Fallback: weight by TP-rate per detector
        weight_dict = _simple_weight_update(feedbacks, detector_names)
        accuracy = None
        auc = None

    # Save weights
    output = {
        "weights": weight_dict,
        "trained_on": len(feedbacks),
        "accuracy": accuracy,
        "auc_roc": auc,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "class_distribution": {
            "true_positive": int(y_arr.sum()),
            "false_positive": int(len(y_arr) - y_arr.sum()),
        },
    }

    path = Path(settings.ENSEMBLE_WEIGHTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "ensemble_weights_retrained",
        extra={
            "weights": weight_dict,
            "samples": len(feedbacks),
            "accuracy": accuracy,
            "auc": auc,
        },
    )

    return {
        "status": "success",
        "weights": weight_dict,
        "samples": len(feedbacks),
        "accuracy": accuracy,
        "auc_roc": auc,
    }


def _simple_weight_update(
    feedbacks: list[Feedback],
    detector_names: list[str],
) -> dict[str, float]:
    """Fallback: weight detectors by their TP discrimination power."""
    tp_scores: dict[str, list[float]] = {n: [] for n in detector_names}
    fp_scores: dict[str, list[float]] = {n: [] for n in detector_names}

    for fb in feedbacks:
        scores = fb.detector_scores_snapshot or {}
        for name in detector_names:
            detector_data = scores.get(name, {})
            if isinstance(detector_data, dict):
                s = detector_data.get("anomaly_score", 0.0)
            elif isinstance(detector_data, (int, float)):
                s = float(detector_data)
            else:
                s = 0.0
            if fb.verdict == "TRUE_POSITIVE":
                tp_scores[name].append(s)
            else:
                fp_scores[name].append(s)

    # Discrimination = mean(TP scores) - mean(FP scores)
    disc: dict[str, float] = {}
    for name in detector_names:
        tp_mean = float(np.mean(tp_scores[name])) if tp_scores[name] else 0.5
        fp_mean = float(np.mean(fp_scores[name])) if fp_scores[name] else 0.5
        disc[name] = max(0.01, tp_mean - fp_mean)

    total = sum(disc.values())
    return {name: round(v / total, 4) for name, v in disc.items()}
