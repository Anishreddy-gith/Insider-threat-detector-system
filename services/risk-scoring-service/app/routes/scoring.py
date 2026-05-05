"""
Risk scoring route — ``POST /score``.

Orchestrates the full scoring pipeline:
    1. Fan-out to all 4 ML detectors via :func:`asyncio.gather`
    2. Combine detector outputs through the ensemble meta-learner
    3. Apply context-aware adjustments from YAML rules
    4. Classify risk level using adaptive PID-tuned thresholds
    5. Persist score to TimescaleDB & compute 30-day trend
    6. Return a rich :class:`RiskScoreResponse`
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.context_engine import ContextEngine
from app.database import get_db
from app.schemas import (
    DetectorResult,
    RiskScoreRequest,
    RiskScoreResponse,
)
from app.thresholds import ThresholdManager
from app.trend_analyzer import TrendAnalyzer

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singletons (initialised in main.py lifespan)
_context_engine: ContextEngine | None = None
_threshold_mgr: ThresholdManager | None = None

DETECTOR_NAMES = [
    "isolation_forest",
    "autoencoder",
    "lstm_temporal",
    "gnn_relational",
]


def set_context_engine(engine: ContextEngine) -> None:
    global _context_engine
    _context_engine = engine


def set_threshold_manager(mgr: ThresholdManager) -> None:
    global _threshold_mgr
    _threshold_mgr = mgr


# ── ML-engine async fan-out ─────────────────────────────────────────

async def _call_detector(
    client: httpx.AsyncClient,
    detector_name: str,
    user_id: str,
    features: list[float],
    event_metadata: dict[str, Any],
) -> DetectorResult:
    """
    Call a single detector on the ML engine.

    Expected ML-engine endpoint:
        POST /api/v1/detect/{detector_name}
        Body: {"user_id": ..., "features": [...], "metadata": {...}}
        Resp: {"anomaly_score": 0.72, "is_anomaly": true, "detail": {...}}
    """
    try:
        resp = await client.post(
            f"{settings.ML_ENGINE_URL}/api/v1/detect/{detector_name}",
            json={
                "user_id": user_id,
                "features": features,
                "metadata": event_metadata,
            },
            timeout=settings.ML_ENGINE_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            return DetectorResult(
                detector=detector_name,
                anomaly_score=data.get("anomaly_score", 0.0),
                is_anomaly=data.get("is_anomaly", False),
                detail=data.get("detail", {}),
            )
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.warning(
            "detector_call_failed",
            extra={"detector": detector_name, "error": str(exc)},
        )

    # Graceful fallback — detector treated as "no anomaly"
    return DetectorResult(
        detector=detector_name,
        anomaly_score=0.0,
        is_anomaly=False,
        detail={"error": "detector_unavailable"},
    )


async def call_all_detectors(
    user_id: str,
    features: list[float],
    event_metadata: dict[str, Any],
) -> list[DetectorResult]:
    """Fan-out to all 4 detectors using asyncio.gather."""
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[
                _call_detector(client, name, user_id, features, event_metadata)
                for name in DETECTOR_NAMES
            ],
            return_exceptions=False,
        )
    return list(results)


# ── Ensemble combiner ──────────────────────────────────────────────

def ensemble_score(results: list[DetectorResult]) -> float:
    """
    Combine detector scores using a weighted average.

    In production this would call the ML-engine's ensemble endpoint;
    here we use simple weights aligned with detector trust levels.
    """
    weights = {
        "isolation_forest": 0.30,
        "autoencoder": 0.25,
        "lstm_temporal": 0.25,
        "gnn_relational": 0.20,
    }
    total_w = 0.0
    total_s = 0.0
    for r in results:
        w = weights.get(r.detector, 0.25)
        total_w += w
        total_s += w * r.anomaly_score
    return total_s / total_w if total_w else 0.0


# ── Feature explanations & actions ──────────────────────────────────

def extract_top_features(
    results: list[DetectorResult],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Pull the most anomalous features from detector details.

    Prefers SHAP-based importance from Isolation Forest, then
    feature errors from the autoencoder, etc.
    """
    features: list[dict[str, Any]] = []
    for r in results:
        detail = r.detail
        # Isolation Forest: feature_importance
        if "feature_importance" in detail:
            for name, score in detail["feature_importance"].items():
                features.append(
                    {"feature": name, "importance": score, "source": r.detector}
                )
        # Autoencoder: feature_errors
        if "feature_errors" in detail:
            for i, err in enumerate(detail["feature_errors"]):
                features.append(
                    {"feature": f"dim_{i}", "importance": err, "source": r.detector}
                )
        # LSTM: temporal_features
        if "temporal_features" in detail:
            for k, v in detail["temporal_features"].items():
                features.append(
                    {"feature": k, "importance": v, "source": r.detector}
                )
    features.sort(key=lambda x: x["importance"], reverse=True)
    return features[:top_k]


def recommended_actions(level: str) -> list[str]:
    """Context-sensitive recommended actions per risk level."""
    return {
        "LOW": ["Continue standard monitoring"],
        "MEDIUM": [
            "Review recent activity logs",
            "Check access patterns against peer group",
        ],
        "HIGH": [
            "Escalate to security team",
            "Restrict access to sensitive resources",
            "Review peer-group deviation report",
        ],
        "CRITICAL": [
            "Immediate SOC investigation",
            "Suspend account pending review",
            "Notify CISO and legal counsel",
            "Preserve forensic evidence chain",
        ],
    }.get(level, ["Continue monitoring"])


# ── Main endpoint ──────────────────────────────────────────────────

@router.post("/score", response_model=RiskScoreResponse)
async def compute_risk(
    body: RiskScoreRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Full risk-scoring pipeline.

    1. Call 4 ML detectors via ``asyncio.gather``
    2. Ensemble-combine into a single raw score
    3. Adjust for business context (YAML rules + HR data)
    4. Classify via adaptive PID-tuned thresholds
    5. Record in TimescaleDB, compute 30-day trend
    6. Return structured ``RiskScoreResponse``
    """
    user_id = body.user_id
    event = body.event
    event_meta = {
        "event_type": event.event_type,
        "source_ip": event.source_ip,
        "resource": event.resource,
        "timestamp": event.timestamp.isoformat() if event.timestamp else None,
        "hour": event.timestamp.hour if event.timestamp else None,
        **event.metadata,
    }

    # ① Fan-out to detectors
    detector_results = await call_all_detectors(
        user_id, event.features, event_meta
    )

    # ② Ensemble score (0-1)
    raw_ml = ensemble_score(detector_results)

    # ③ Context-aware adjustment
    ctx = _context_engine or ContextEngine()
    ctx_result = await ctx.adjust(
        user_id=user_id,
        raw_score=raw_ml,
        event_metadata=event_meta,
    )
    adjusted = ctx_result.adjusted_score

    # ④ Classify with adaptive thresholds
    tm = _threshold_mgr or ThresholdManager()
    risk_level = tm.classify(adjusted)

    # ⑤ Features & actions
    top_features = extract_top_features(detector_results)
    actions = recommended_actions(risk_level)

    # ⑥ Persist + trend
    trend_analyzer = TrendAnalyzer(db)
    await trend_analyzer.record_score(
        user_id=user_id,
        overall_score=adjusted,
        risk_level=risk_level,
        detector_breakdown={r.detector: r.anomaly_score for r in detector_results},
        context_adjustments={
            a.rule: {"factor": a.factor, "reason": a.reason}
            for a in ctx_result.adjustments
        },
    )
    trend = await trend_analyzer.analyze(user_id)

    scored_at = datetime.now(timezone.utc)

    return RiskScoreResponse(
        user_id=user_id,
        overall_score=round(adjusted, 4),
        risk_level=risk_level,
        detector_breakdown=detector_results,
        top_anomalous_features=top_features,
        recommended_actions=actions,
        business_context=ctx_result.context,
        trend=trend,
        scored_at=scored_at,
        raw_ml_score=round(raw_ml, 4),
        context_adjusted_score=round(adjusted, 4),
    )
