"""
Pydantic v2 request / response schemas for the alert service.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class AlertStatus(str, Enum):
    NEW = "NEW"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FeedbackVerdict(str, Enum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


# ── Risk score payload (from Kafka / risk-scoring-service) ───────────

class DetectorResult(BaseModel):
    detector: str
    anomaly_score: float
    is_anomaly: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class RiskScoreEvent(BaseModel):
    """Payload consumed from the itds.risk-scores Kafka topic."""
    user_id: str
    overall_score: float
    risk_level: str
    detector_breakdown: list[DetectorResult]
    top_anomalous_features: list[dict[str, Any]] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    business_context: dict[str, Any] = Field(default_factory=dict)
    raw_ml_score: float = 0.0
    context_adjusted_score: float = 0.0
    scored_at: datetime | None = None


# ── Alert schemas ────────────────────────────────────────────────────

class AlertCreate(BaseModel):
    user_id: str
    risk_score: float
    risk_level: RiskLevel
    anomaly_type: str
    summary: str
    detector_breakdown: dict[str, Any] | None = None
    top_features: list[dict[str, Any]] | None = None
    recommended_actions: list[str] | None = None
    business_context: dict[str, Any] | None = None


class AlertResponse(BaseModel):
    id: str
    user_id: str
    risk_score: float
    risk_level: str
    anomaly_type: str
    status: str
    summary: str
    detector_breakdown: dict[str, Any] | None = None
    xai_report: dict[str, Any] | None = None
    top_features: list[dict[str, Any]] | None = None
    recommended_actions: list[str] | None = None
    business_context: dict[str, Any] | None = None
    notified_slack: bool = False
    notified_email: bool = False
    notified_siem: bool = False
    assigned_to: str | None = None
    created_at: datetime
    updated_at: datetime


class AlertStatusUpdate(BaseModel):
    status: AlertStatus
    analyst: str | None = None
    notes: str | None = None


# ── Feedback schemas ─────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    alert_id: str
    analyst_id: str
    verdict: FeedbackVerdict
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str | None = None


class FeedbackResponse(BaseModel):
    id: str
    alert_id: str
    analyst_id: str
    verdict: str
    confidence: float
    notes: str | None
    created_at: datetime


# ── XAI report schema ───────────────────────────────────────────────

class XAIReportResponse(BaseModel):
    alert_id: str
    user_id: str
    report_markdown: str
    generated_at: datetime


# ── FP/FN analysis ──────────────────────────────────────────────────

class AnalysisMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    auc_roc: float | None = None
    total_alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int


class AnalysisReport(BaseModel):
    period_start: datetime
    period_end: datetime
    overall: AnalysisMetrics
    per_detector: dict[str, AnalysisMetrics]
    per_department: dict[str, AnalysisMetrics]
    report_markdown: str
    generated_at: datetime
