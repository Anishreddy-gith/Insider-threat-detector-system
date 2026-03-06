"""
Pydantic Response / Request Schemas
====================================
DTOs exposed via the REST API.

Separation from ORM models lets us:
  1. Control exactly which fields are exposed.
  2. Add computed / virtual properties.
  3. Version the API without touching the DB schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ── Pagination ────────────────────────────────────────────────

class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)


class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── User / Auth ───────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, description="Minimum 12 chars — NIST 800-63B.")
    full_name: str = Field(min_length=1, max_length=255)
    role: str = Field(
        default="SOC_ANALYST",
        pattern=r"^(SOC_ANALYST|SOC_MANAGER|ADMIN|HR_SYSTEM)$",
    )


class UserResponse(BaseModel):
    id: UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ── Monitored Entity ─────────────────────────────────────────

class EntityResponse(BaseModel):
    id: UUID
    employee_id: str
    display_name: str | None = None
    department: str | None = None
    role_title: str | None = None
    risk_score: float
    risk_level: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class EntityDetailResponse(EntityResponse):
    recent_alerts: list[AlertResponse] = Field(default_factory=list)
    risk_history: list[dict[str, Any]] = Field(default_factory=list)


# ── Alerts ────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    id: UUID
    entity_id: UUID
    severity: str
    title: str
    description: str | None = None
    risk_score: float
    is_resolved: bool
    resolved_by: UUID | None = None
    resolved_at: datetime | None = None
    assigned_to: UUID | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertResolveRequest(BaseModel):
    resolution_note: str = Field(
        min_length=10,
        description="Analyst must explain resolution for audit trail.",
    )


class AlertAssignRequest(BaseModel):
    analyst_id: UUID


# ── Analytics ─────────────────────────────────────────────────

class RiskDistribution(BaseModel):
    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0


class DashboardMetrics(BaseModel):
    total_entities: int
    active_alerts: int
    critical_alerts: int
    average_risk_score: float
    risk_distribution: RiskDistribution
    alerts_last_24h: int
    top_risk_entities: list[EntityResponse]


class AnomalyExplanation(BaseModel):
    model_name: str
    anomaly_score: float
    top_features: list[dict[str, float]]
    natural_language_explanation: str
    confidence: float


# ── HR Integration ────────────────────────────────────────────

class HREmployeeContext(BaseModel):
    """Employment context pushed by Workday / BambooHR connector."""
    employee_id: str
    employment_status: str = Field(
        ..., description="active | on_leave | terminated | suspended"
    )
    department: str
    job_role: str
    manager_id: str | None = None
    active_projects: list[str] = Field(default_factory=list)
    resignation_date: str | None = None
    hire_date: str | None = None
    pto_calendar: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {start_date, end_date, type} dicts",
    )


class HRWebhookEvent(BaseModel):
    """Real-time employment change event from HR webhook."""
    event_type: str = Field(
        ..., description="hire | termination | department_change | role_change | suspension"
    )
    employee_id: str
    timestamp: str
    changes: dict[str, Any] = Field(default_factory=dict)
    source: str = "workday"


# ── SIEM ──────────────────────────────────────────────────────

class SIEMAlertPayload(BaseModel):
    """Alert payload formatted for SIEM ingestion."""
    alert_id: str
    entity_id: str
    severity: str
    risk_score: float
    title: str
    description: str | None = None
    xai_report: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: str


# ── Audit Log ─────────────────────────────────────────────────

class AuditLogResponse(BaseModel):
    id: UUID
    user_id: str | None = None
    user_role: str | None = None
    action: str
    resource_path: str
    query_params: str | None = None
    ip_address: str | None = None
    status_code: int | None = None
    response_time_ms: float | None = None
    timestamp: datetime

    model_config = {"from_attributes": True}
