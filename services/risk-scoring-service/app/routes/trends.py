"""
Trend & history routes.

Provides read access to per-user score trajectories and the
slow-burn detection output.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import TrendInfo
from app.trend_analyzer import TrendAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{user_id}", response_model=TrendInfo)
async def get_user_trend(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Analyse a user's 30-day score trajectory.

    Returns trend classification (``stable | rising | falling | slow_burn``)
    plus the OLS slope and daily score series.
    """
    analyzer = TrendAnalyzer(db)
    return await analyzer.analyze(user_id)


@router.get("/{user_id}/history")
async def get_user_history(
    user_id: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return raw score history entries for a user."""
    analyzer = TrendAnalyzer(db)
    return await analyzer.get_history(user_id, days=days)
