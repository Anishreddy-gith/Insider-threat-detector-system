"""
Alert CRUD + lifecycle management routes.

Endpoints:
  POST   /                    – create a new alert (API-driven)
  GET    /                    – list alerts (filterable)
  GET    /{alert_id}          – retrieve a single alert
  GET    /{alert_id}/xai      – get the XAI report
  PATCH  /{alert_id}/status   – transition lifecycle state
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Alert
from app.schemas import (
    AlertCreate,
    AlertResponse,
    AlertStatus,
    AlertStatusUpdate,
    RiskLevel,
    XAIReportResponse,
)
from app.alert_engine import _dedup_key, transition_status
from app.xai_reports import XAIReportGenerator

logger = logging.getLogger(__name__)
router = APIRouter()


def _alert_to_response(alert: Alert) -> AlertResponse:
    return AlertResponse(
        id=str(alert.id),
        user_id=alert.user_id,
        risk_score=alert.risk_score,
        risk_level=alert.risk_level,
        anomaly_type=alert.anomaly_type,
        status=alert.status,
        summary=alert.summary,
        detector_breakdown=alert.detector_breakdown,
        xai_report=alert.xai_report,
        top_features=alert.top_features,
        recommended_actions=alert.recommended_actions,
        business_context=alert.business_context,
        notified_slack=alert.notified_slack,
        notified_email=alert.notified_email,
        notified_siem=alert.notified_siem,
        assigned_to=alert.assigned_to,
        created_at=alert.created_at,
        updated_at=alert.updated_at,
    )


@router.post("/", response_model=AlertResponse, status_code=201)
async def create_alert(
    body: AlertCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new alert (API-driven, bypasses Kafka)."""
    alert = Alert(
        user_id=body.user_id,
        risk_score=body.risk_score,
        risk_level=body.risk_level.value,
        anomaly_type=body.anomaly_type,
        summary=body.summary,
        status="NEW",
        detector_breakdown=body.detector_breakdown,
        top_features=body.top_features,
        recommended_actions=body.recommended_actions,
        business_context=body.business_context,
        dedup_key=_dedup_key(body.user_id, body.anomaly_type),
    )
    db.add(alert)
    await db.flush()
    return _alert_to_response(alert)


@router.get("/", response_model=list[AlertResponse])
async def list_alerts(
    status: str | None = Query(None),
    risk_level: str | None = Query(None),
    user_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List alerts with optional filters."""
    q = select(Alert).order_by(Alert.created_at.desc())
    if status:
        q = q.where(Alert.status == status)
    if risk_level:
        q = q.where(Alert.risk_level == risk_level)
    if user_id:
        q = q.where(Alert.user_id == user_id)
    q = q.offset(offset).limit(limit)

    result = await db.execute(q)
    return [_alert_to_response(a) for a in result.scalars().all()]


@router.get("/stats")
async def alert_stats(db: AsyncSession = Depends(get_db)):
    """Summary statistics of current alerts."""
    result = await db.execute(
        select(Alert.status, func.count(Alert.id).label("count"))
        .group_by(Alert.status)
    )
    return {row.status: row.count for row in result.all()}


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve a single alert by ID."""
    result = await db.execute(
        select(Alert).where(Alert.id == UUID(alert_id))
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found")
    return _alert_to_response(alert)


@router.get("/{alert_id}/xai", response_model=XAIReportResponse)
async def get_xai_report(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve the XAI report for an alert."""
    result = await db.execute(
        select(Alert).where(Alert.id == UUID(alert_id))
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found")
    if not alert.xai_report:
        raise HTTPException(404, "XAI report not yet generated for this alert")

    return XAIReportResponse(
        alert_id=str(alert.id),
        user_id=alert.user_id,
        report_markdown=alert.xai_report.get("markdown", ""),
        generated_at=alert.xai_report.get("generated_at", alert.created_at),
    )


@router.patch("/{alert_id}/status", response_model=AlertResponse)
async def update_status(
    alert_id: str,
    body: AlertStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Transition an alert to a new lifecycle state."""
    try:
        alert = await transition_status(
            alert_id=alert_id,
            new_status=body.status,
            session=db,
            analyst=body.analyst,
            notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return _alert_to_response(alert)
