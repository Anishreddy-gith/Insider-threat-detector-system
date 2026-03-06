"""
Synchronous prediction endpoint.

Used by the API-Gateway when a SOC analyst requests an ad-hoc entity
risk assessment.  The normal flow is asynchronous via Kafka; this route
is the "pull" complement to the Kafka "push" pipeline.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class PredictRequest(BaseModel):
    entity_id: str
    features: list[float]        # pre-extracted 45-dim vector


class PredictResponse(BaseModel):
    entity_id: str
    anomaly_score: float
    risk_level: str
    top_features: list[dict] | None = None
    explanation: str | None = None


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    """Run inference synchronously and return the anomaly assessment."""
    pipeline = request.app.state.pipeline
    result = await pipeline.predict_single(body.entity_id, body.features)
    return PredictResponse(**result)
