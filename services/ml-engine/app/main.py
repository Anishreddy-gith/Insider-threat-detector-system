"""
ML-engine FastAPI application.

This service is the heaviest compute node in the pipeline.  It:
1. Consumes ``itds.raw-events`` from Kafka via an async consumer
2. Extracts a 45-dimensional feature vector per entity window
3. Runs inference through the LSTM Autoencoder → GNN → Ensemble pipeline
4. Generates SHAP explanations for high-anomaly detections (cost gating)
5. Publishes ``itds.anomalies`` back to Kafka
6. Exposes /healthz, /metrics, and an POST /api/v1/predict endpoint for
   synchronous ad-hoc predictions (used by the API Gateway).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load ML models into GPU/CPU memory at startup; release on shutdown."""
    from app.pipeline.inference import InferencePipeline

    application.state.pipeline = InferencePipeline()
    application.state.pipeline.load_models()
    logger.info(
        "ml_models_loaded",
        extra={"device": str(application.state.pipeline.device)},
    )
    yield
    logger.info("ml_engine_stopped")


app = FastAPI(
    title="ITDS – ML Engine",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ── Routes ───────────────────────────────────────────────────────────────

from app.routes import predict as predict_routes  # noqa: E402
from app.routes import privacy as privacy_routes  # noqa: E402

app.include_router(predict_routes.router, prefix="/api/v1", tags=["ml"])
app.include_router(privacy_routes.router, prefix="/api/v1/privacy", tags=["privacy"])


@app.get("/healthz")
async def health():
    return {"status": "ok", "service": settings.SERVICE_NAME}
