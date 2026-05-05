"""
Trend Analyzer — 30-day score history + "slow burn" detection.

Stores per-user risk scores in TimescaleDB and detects two patterns:

1. **Spike** — a single score jump ≥ 2× the user's rolling mean.
2. **Slow burn** — a persistent upward trend where the daily slope
   exceeds ``SLOW_BURN_SLOPE_THRESHOLD`` for ≥ 7 consecutive days.

Slow-burn detection is vital because sophisticated insiders rarely
trigger a spike; instead they incrementally escalate (access slightly
more files each day, connect to slightly more unusual hosts) hoping
to stay beneath static thresholds.

The slope is estimated via ordinary least squares (OLS) on the last
``SLOW_BURN_LOOKBACK_DAYS`` of daily-aggregated scores.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import ScoreHistory
from app.schemas import TrendInfo

logger = logging.getLogger(__name__)
settings = get_settings()


class TrendAnalyzer:
    """
    Analyse a user's risk-score trajectory over the last 30 days.

    Usage::

        analyzer = TrendAnalyzer(db_session)
        trend = await analyzer.analyze("user-42")
        if trend.score_trend == "slow_burn":
            ... escalate ...
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── public API ──────────────────────────────────────────────────

    async def record_score(
        self,
        user_id: str,
        overall_score: float,
        risk_level: str,
        detector_breakdown: dict[str, Any] | None = None,
        context_adjustments: dict[str, Any] | None = None,
    ) -> ScoreHistory:
        """Persist a new score entry in TimescaleDB."""
        entry = ScoreHistory(
            user_id=user_id,
            overall_score=overall_score,
            risk_level=risk_level,
            detector_breakdown=detector_breakdown,
            context_adjustments=context_adjustments,
            scored_at=datetime.now(timezone.utc),
        )
        self.session.add(entry)
        await self.session.flush()
        logger.info(
            "score_recorded",
            extra={
                "user_id": user_id,
                "score": round(overall_score, 4),
                "risk_level": risk_level,
            },
        )
        return entry

    async def analyze(self, user_id: str) -> TrendInfo:
        """
        Fetch 30-day history and compute trend classification.

        Returns a ``TrendInfo`` with:
            score_trend   – "rising" | "falling" | "stable" | "slow_burn"
            trend_slope   – OLS-estimated daily slope
            scores_30d    – list of daily-averaged scores
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=settings.TREND_WINDOW_DAYS
        )

        result = await self.session.execute(
            select(ScoreHistory)
            .where(
                ScoreHistory.user_id == user_id,
                ScoreHistory.scored_at >= cutoff,
            )
            .order_by(ScoreHistory.scored_at.asc())
        )
        rows = result.scalars().all()

        if not rows:
            return TrendInfo(score_trend="stable", trend_slope=0.0, scores_30d=[])

        # ── Aggregate to daily means ────────────────────────────────
        daily: dict[str, list[float]] = {}
        for row in rows:
            day_key = row.scored_at.strftime("%Y-%m-%d")
            daily.setdefault(day_key, []).append(row.overall_score)

        daily_means = [
            float(np.mean(scores)) for scores in daily.values()
        ]

        if len(daily_means) < 2:
            return TrendInfo(
                score_trend="stable",
                trend_slope=0.0,
                scores_30d=[round(s, 4) for s in daily_means],
            )

        # ── OLS slope on the lookback window ────────────────────────
        lookback = min(settings.SLOW_BURN_LOOKBACK_DAYS, len(daily_means))
        recent = daily_means[-lookback:]
        slope = self._ols_slope(recent)

        # ── Classify trend ──────────────────────────────────────────
        trend = self._classify_trend(daily_means, slope)

        if trend == "slow_burn":
            logger.warning(
                "slow_burn_detected",
                extra={
                    "user_id": user_id,
                    "slope": round(slope, 6),
                    "window_days": lookback,
                    "recent_scores": [round(s, 4) for s in recent[-5:]],
                },
            )

        return TrendInfo(
            score_trend=trend,
            trend_slope=round(slope, 6),
            scores_30d=[round(s, 4) for s in daily_means],
        )

    async def get_history(
        self,
        user_id: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Return raw score history entries as dicts."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self.session.execute(
            select(ScoreHistory)
            .where(
                ScoreHistory.user_id == user_id,
                ScoreHistory.scored_at >= cutoff,
            )
            .order_by(ScoreHistory.scored_at.desc())
        )
        return [
            {
                "score": row.overall_score,
                "risk_level": row.risk_level,
                "scored_at": row.scored_at.isoformat(),
                "detector_breakdown": row.detector_breakdown,
            }
            for row in result.scalars().all()
        ]

    # ── internal ────────────────────────────────────────────────────

    @staticmethod
    def _ols_slope(values: list[float]) -> float:
        """
        Ordinary least squares slope: Δscore / Δday.

        Uses the normal equation:  β = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
        """
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(values, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)
        if den == 0:
            return 0.0
        return float(num / den)

    def _classify_trend(
        self,
        daily_means: list[float],
        slope: float,
    ) -> str:
        """
        Classify the score trajectory.

        slow_burn : persistent upward slope above threshold
        rising    : upward slope but below slow-burn threshold
        falling   : downward slope
        stable    : near-zero slope
        """
        if slope >= settings.SLOW_BURN_SLOPE_THRESHOLD:
            # Confirm it's persistent: check that the slope over
            # at least 7 consecutive days is positive
            if len(daily_means) >= 7:
                last_7 = daily_means[-7:]
                sub_slope = self._ols_slope(last_7)
                if sub_slope > 0:
                    return "slow_burn"
            return "rising"
        if slope > 0.005:
            return "rising"
        if slope < -0.005:
            return "falling"
        return "stable"
