"""
Alert Routes — Proxy to Alert Service + Local RBAC
====================================================
The gateway proxies alert CRUD to the downstream alert-service while
enforcing RBAC and writing audit log entries.

Endpoints:
  GET    /alerts              — List alerts (SOC_ANALYST+)
  GET    /alerts/stats        — Alert stats (SOC_ANALYST+)
  GET    /alerts/{id}         — Single alert (SOC_ANALYST+)
  GET    /alerts/{id}/xai     — XAI report (SOC_ANALYST+)
  PATCH  /alerts/{id}/status  — Update lifecycle (SOC_ANALYST+)
  POST   /feedback            — Submit verdict (SOC_ANALYST+)
  GET    /feedback/stats      — Feedback stats (SOC_MANAGER+)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from httpx import HTTPStatusError

from app.auth.jwt_handler import Role
from app.auth.rbac import TokenPayload, require_role
from app.services.service_client import proxy_get, proxy_patch, proxy_post

router = APIRouter(tags=["alerts"])


def _request_id(request: Request) -> str | None:
    return request.headers.get("X-Request-ID")


# ── Alerts ────────────────────────────────────────────────────

@router.get("/alerts", summary="List alerts (proxied to alert-service)")
async def list_alerts(
    request: Request,
    status: str | None = Query(None),
    risk_level: str | None = Query(None),
    user_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    if risk_level:
        params["risk_level"] = risk_level
    if user_id:
        params["user_id"] = user_id

    try:
        return await proxy_get(
            "alert", "/api/v1/alerts/",
            params=params, request_id=_request_id(request),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text,
        )


@router.get("/alerts/stats", summary="Alert statistics")
async def alert_stats(
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    try:
        return await proxy_get(
            "alert", "/api/v1/alerts/stats",
            request_id=_request_id(request),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text,
        )


@router.get("/alerts/{alert_id}", summary="Get single alert with details")
async def get_alert(
    alert_id: str,
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    try:
        return await proxy_get(
            "alert", f"/api/v1/alerts/{alert_id}",
            request_id=_request_id(request),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text,
        )


@router.get(
    "/alerts/{alert_id}/xai",
    summary="Get XAI report for alert",
)
async def get_alert_xai(
    alert_id: str,
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    try:
        return await proxy_get(
            "alert", f"/api/v1/alerts/{alert_id}/xai",
            request_id=_request_id(request),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text,
        )


@router.patch(
    "/alerts/{alert_id}/status",
    summary="Update alert lifecycle status (SOC_ANALYST+)",
)
async def update_alert_status(
    alert_id: str,
    body: dict,
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    try:
        return await proxy_patch(
            "alert", f"/api/v1/alerts/{alert_id}/status",
            json_body=body, request_id=_request_id(request),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text,
        )


# ── Feedback ──────────────────────────────────────────────────

@router.post("/feedback", summary="Submit analyst feedback (TP/FP/Escalate)")
async def submit_feedback(
    body: dict,
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    try:
        return await proxy_post(
            "alert", "/api/v1/feedback/",
            json_body=body, request_id=_request_id(request),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text,
        )


@router.get("/feedback/stats", summary="Feedback aggregate statistics")
async def feedback_stats(
    request: Request,
    _user: TokenPayload = Depends(
        require_role(Role.SOC_MANAGER, Role.ADMIN)
    ),
) -> Any:
    try:
        return await proxy_get(
            "alert", "/api/v1/feedback/stats",
            request_id=_request_id(request),
        )
    except HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text,
        )
