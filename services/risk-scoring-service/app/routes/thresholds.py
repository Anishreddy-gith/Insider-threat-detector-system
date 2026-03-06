"""
Threshold management routes.

Provides endpoints to:
    - Feed false-positive data into the PID controller
    - Read current adaptive thresholds
    - View adjustment history
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from app.schemas import ThresholdResponse, ThresholdUpdate
from app.thresholds import ThresholdManager

logger = logging.getLogger(__name__)
router = APIRouter()

_threshold_mgr: ThresholdManager | None = None


def set_threshold_manager(mgr: ThresholdManager) -> None:
    global _threshold_mgr
    _threshold_mgr = mgr


def _mgr() -> ThresholdManager:
    return _threshold_mgr or ThresholdManager()


@router.post("/update", response_model=ThresholdResponse)
async def update_thresholds(body: ThresholdUpdate):
    """
    Feed observed FP data into the PID controller.

    Call this periodically (e.g. every hour) with the number of
    false positives and total alerts in the last window.
    """
    mgr = _mgr()
    result = mgr.update(fp_count=body.fp_count, total_alerts=body.total_alerts)
    return ThresholdResponse(
        threshold_low=result["threshold_low"],
        threshold_medium=result["threshold_medium"],
        threshold_high=result["threshold_high"],
        fp_rate=result["fp_rate"],
        adjustment_made=result["adjustment_made"],
    )


@router.get("/current", response_model=ThresholdResponse)
async def get_current_thresholds():
    """Return current adaptive threshold values."""
    mgr = _mgr()
    return ThresholdResponse(
        threshold_low=mgr.threshold_low,
        threshold_medium=mgr.threshold_medium,
        threshold_high=mgr.threshold_high,
        fp_rate=0.0,
        adjustment_made=False,
    )


@router.get("/history")
async def get_threshold_history() -> list[dict[str, Any]]:
    """Return the full PID adjustment history for this session."""
    return _mgr().history
