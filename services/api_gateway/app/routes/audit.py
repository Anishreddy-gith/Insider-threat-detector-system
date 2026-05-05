"""
Audit Log Routes
=================
Read-only access to the immutable audit trail.

Only ADMIN users can query audit logs â€” this prevents analysts
from checking whether their actions are being monitored (which
could be relevant during an active insider threat investigation).

The audit log is NEVER deletable or modifiable via API.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.api_gateway.app.auth.jwt_handler import Role
from services.api_gateway.app.auth.rbac import TokenPayload, require_role
from services.api_gateway.app.models.database import AuditLog, get_db
from services.api_gateway.app.models.schemas import AuditLogResponse, PaginatedResponse

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get(
    "/logs",
    response_model=PaginatedResponse,
    summary="Query audit logs (ADMIN only)",
)
async def list_audit_logs(
    _user: TokenPayload = Depends(require_role(Role.ADMIN)),
    user_id: str | None = Query(None, description="Filter by user UUID"),
    action: str | None = Query(None, description="Filter by action prefix"),
    ip_address: str | None = Query(None),
    hours: int = Query(24, ge=1, le=8760, description="Look-back window in hours"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse:
    """
    Query the immutable audit log.

    Supports filtering by user, action prefix, IP, and time window.
    Results are always ordered newest-first.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    query = select(AuditLog).where(AuditLog.timestamp >= cutoff)
    count_q = select(func.count(AuditLog.id)).where(AuditLog.timestamp >= cutoff)

    if user_id:
        query = query.where(AuditLog.user_id == user_id)
        count_q = count_q.where(AuditLog.user_id == user_id)
    if action:
        query = query.where(AuditLog.action.ilike(f"{action}%"))
        count_q = count_q.where(AuditLog.action.ilike(f"{action}%"))
    if ip_address:
        query = query.where(AuditLog.ip_address == ip_address)
        count_q = count_q.where(AuditLog.ip_address == ip_address)

    total = (await db.execute(count_q)).scalar() or 0
    offset = (page - 1) * page_size

    results = (await db.execute(
        query.order_by(AuditLog.timestamp.desc()).offset(offset).limit(page_size)
    )).scalars().all()

    return PaginatedResponse(
        items=[AuditLogResponse.model_validate(r).model_dump(mode="json") for r in results],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get(
    "/logs/summary",
    summary="Audit log summary statistics (ADMIN only)",
)
async def audit_summary(
    _user: TokenPayload = Depends(require_role(Role.ADMIN)),
    hours: int = Query(24, ge=1, le=8760),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get aggregate statistics about API usage over the given window.
    Useful for detecting anomalous access patterns.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    total = (await db.execute(
        select(func.count(AuditLog.id)).where(AuditLog.timestamp >= cutoff)
    )).scalar() or 0

    unique_users = (await db.execute(
        select(func.count(func.distinct(AuditLog.user_id))).where(
            AuditLog.timestamp >= cutoff,
            AuditLog.user_id.isnot(None),
        )
    )).scalar() or 0

    unique_ips = (await db.execute(
        select(func.count(func.distinct(AuditLog.ip_address))).where(
            AuditLog.timestamp >= cutoff
        )
    )).scalar() or 0

    error_count = (await db.execute(
        select(func.count(AuditLog.id)).where(
            AuditLog.timestamp >= cutoff,
            AuditLog.status_code >= 400,
        )
    )).scalar() or 0

    return {
        "window_hours": hours,
        "total_requests": total,
        "unique_users": unique_users,
        "unique_ips": unique_ips,
        "error_requests": error_count,
        "error_rate": round(error_count / max(total, 1) * 100, 2),
    }

