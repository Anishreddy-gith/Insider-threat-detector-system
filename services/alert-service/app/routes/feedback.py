"""
Feedback routes — analyst verdicts + retraining trigger.

Endpoints:
  POST  /feedback           – submit a verdict on an alert
  GET   /feedback/{alert_id} – list feedback for an alert
  GET   /feedback/stats     – aggregate feedback statistics
  POST  /feedback/retrain   – trigger ensemble weight retraining
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.feedback_api import (
    get_feedback_for_alert,
    get_feedback_stats,
    retrain_ensemble_weights,
    store_feedback,
)
from app.schemas import FeedbackCreate, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    body: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit analyst feedback on an alert.

    Marks the alert as TRUE_POSITIVE → RESOLVED,
    FALSE_POSITIVE → FALSE_POSITIVE, or NEEDS_REVIEW → INVESTIGATING.
    """
    try:
        fb = await store_feedback(body, db)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return FeedbackResponse(
        id=str(fb.id),
        alert_id=str(fb.alert_id),
        analyst_id=fb.analyst_id,
        verdict=fb.verdict,
        confidence=fb.confidence,
        notes=fb.notes,
        created_at=fb.created_at,
    )


@router.get("/stats")
async def feedback_stats(db: AsyncSession = Depends(get_db)):
    """Aggregate feedback statistics (TP/FP/NR counts + rates)."""
    return await get_feedback_stats(db)


@router.get("/{alert_id}", response_model=list[FeedbackResponse])
async def list_feedback(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List all feedback entries for a specific alert."""
    feedbacks = await get_feedback_for_alert(alert_id, db)
    return [
        FeedbackResponse(
            id=str(fb.id),
            alert_id=str(fb.alert_id),
            analyst_id=fb.analyst_id,
            verdict=fb.verdict,
            confidence=fb.confidence,
            notes=fb.notes,
            created_at=fb.created_at,
        )
        for fb in feedbacks
    ]


@router.post("/retrain")
async def trigger_retrain():
    """
    Manually trigger ensemble weight retraining from feedback data.

    Normally runs as a weekly cron job, but can be triggered on-demand
    by SOC leads when sufficient new feedback has accumulated.
    """
    result = await retrain_ensemble_weights()
    return result
