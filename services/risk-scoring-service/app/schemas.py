"""
Pydantic v2 request / response schemas for the risk-scoring API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────

class EventPayload(BaseModel):
    """Raw event data sent alongside the user_id."""
    event_type: str = Field(..., examples=["login", "file_access", "network"])
    features: list[float] = Field(
        ...,
        description="Pre-extracted feature vector (dimensionality matches ML engine)",
    )
    source_ip: str | None = None
    resource: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskScoreRequest(BaseModel):
    user_id: str
    event: EventPayload


# ── Response ─────────────────────────────────────────────────────────

class DetectorResult(BaseModel):
    detector: str
    anomaly_score: float
    is_anomaly: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class ContextAdjustment(BaseModel):
    rule: str
    factor: float
    reason: str


class BusinessContext(BaseModel):
    employment_status: str = "active"
    department: str = "unknown"
    is_on_pto: bool = False
    recently_promoted: bool = False
    submitted_resignation: bool = False
    current_projects: list[str] = Field(default_factory=list)
    adjustments_applied: list[ContextAdjustment] = Field(default_factory=list)


class TrendInfo(BaseModel):
    score_trend: str = "stable"            # rising | falling | stable | slow_burn
    trend_slope: float = 0.0
    scores_30d: list[float] = Field(default_factory=list)


class RiskScoreResponse(BaseModel):
    user_id: str
    overall_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str                        # LOW | MEDIUM | HIGH | CRITICAL
    detector_breakdown: list[DetectorResult]
    top_anomalous_features: list[dict[str, Any]]
    recommended_actions: list[str]
    business_context: BusinessContext
    trend: TrendInfo | None = None
    scored_at: datetime
    raw_ml_score: float                    # pre-context-adjustment score
    context_adjusted_score: float          # post-context-adjustment score


# ── Threshold ────────────────────────────────────────────────────────

class ThresholdUpdate(BaseModel):
    fp_count: int
    total_alerts: int


class ThresholdResponse(BaseModel):
    threshold_low: float
    threshold_medium: float
    threshold_high: float
    fp_rate: float
    adjustment_made: bool
