"""
Auth-service FastAPI application entry-point.

Responsibilities:
- Bootstrap DB connection pool (async SQLAlchemy)
- Expose authentication endpoints (login, register, refresh, me)
- Prometheus instrumentation for SRE dashboards
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from services.auth_service.app.config import get_settings
from shared.utils.logging import setup_logging

settings = get_settings()
setup_logging(settings.SERVICE_NAME)
logger = logging.getLogger(settings.SERVICE_NAME)

# â”€â”€ Async SQLAlchemy engine & session factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,           # detect stale connections before use
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,       # avoid lazy-load pitfalls in async code
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown hooks â€“ mirrors the pattern in all other services."""
    logger.info("auth_service_starting", extra={"port": settings.PORT})
    yield
    await engine.dispose()
    logger.info("auth_service_stopped")


app = FastAPI(
    title="ITDS â€“ Auth Service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

# â”€â”€ Prometheus metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# â”€â”€ Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from services.auth_service.app.routes import auth as auth_routes  # noqa: E402 â€“ after app init

app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["auth"])


@app.get("/healthz")
async def health():
    return {"status": "ok", "service": settings.SERVICE_NAME}

