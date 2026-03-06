"""
Privacy API routes — budget status, PIA generation, ε explanation.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.config import get_settings
from app.privacy.differential_privacy import explain_epsilon
from app.privacy.pia_generator import PIAGenerator

router = APIRouter()
settings = get_settings()


# ── Schemas ──────────────────────────────────────────────────────────

class BudgetStatusResponse(BaseModel):
    entity_id: str
    daily_budget: float
    consumed: float
    remaining: float
    queries: int
    utilisation_pct: float
    alert_triggered: bool
    exhausted: bool


class EpsilonExplainRequest(BaseModel):
    epsilon: float = 1.0
    delta: float = 1e-5


class PIARequest(BaseModel):
    organisation_name: str = "ITDS Deployment"
    system_name: str = "Insider Threat Detection System"
    dpo_contact: str = "dpo@example.com"


# ── Endpoints ────────────────────────────────────────────────────────

@router.get("/budget/{entity_id}", response_model=BudgetStatusResponse)
async def get_budget_status(entity_id: str, request: Request):
    """Get current privacy budget status for an entity."""
    tracker = getattr(request.app.state, "budget_tracker", None)
    if tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Privacy budget tracker not initialised (Redis unavailable)",
        )
    status = tracker.get_status(entity_id)
    return BudgetStatusResponse(**status.to_dict())


@router.post("/explain-epsilon")
async def explain_epsilon_endpoint(body: EpsilonExplainRequest):
    """Get a plain-English explanation of an (ε, δ) guarantee."""
    return {"explanation": explain_epsilon(body.epsilon, body.delta)}


@router.post("/pia")
async def generate_pia(body: PIARequest):
    """Auto-generate a Privacy Impact Assessment report."""
    gen = PIAGenerator(
        organisation_name=body.organisation_name,
        system_name=body.system_name,
        dpo_contact=body.dpo_contact,
    )
    return gen.generate()
