"""
API Gateway — FastAPI Application Entry-point
==============================================
This is the **single entry-point** for all external HTTP traffic.

Architecture rationale
----------------------
*  Every internal microservice is **only** reachable on the Docker/k8s
   overlay network.  The gateway is the sole ingress — reducing the attack
   surface.
*  Cross-cutting concerns (auth, rate-limiting, CORS, request-id
   propagation, privacy headers, audit logging) are applied here once,
   so individual services stay focused on business logic.
*  The gateway does **not** implement business logic itself; it proxies
   or aggregates responses from downstream services.

Middleware execution order (outermost → innermost)
--------------------------------------------------
1. **CORS** — Pre-flight OPTIONS must get correct headers before
   anything else touches the request.
2. **PrivacyHeaders** — Inject GDPR/CCPA security response headers.
3. **JWTAuth** — Decode the ``Authorization: Bearer <token>`` header,
   inject ``request.state.user`` / ``.role`` for downstream middleware
   and route handlers.
4. **RateLimiter** — Per-role sliding-window rate limiting backed by
   Redis.  Runs *after* auth so it knows the caller's role.
5. **AuditLog** — Append-only PostgreSQL audit record for every API
   call.  Runs innermost so it captures the response status code and
   elapsed time.

Lifespan hooks
--------------
*  **Startup** — configure structured logging, create DB tables (dev),
   warm up Redis pool, initialise inter-service HTTP clients.
*  **Shutdown** — close Redis, dispose SQLAlchemy engine, close HTTP
   client pools, stop Kafka producer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.middleware.auth import JWTAuthMiddleware
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.privacy import PrivacyHeadersMiddleware
from app.middleware.audit import AuditLogMiddleware
from app.models.database import engine, Base
from app.routes import alerts, analytics, health, users, audit
from app.auth import routes as auth_routes
from app.integrations import hr_connector
from app.services.service_client import init_clients, close_clients

try:
    from shared.utils.logging import configure_logging, get_logger
except ImportError:
    import logging as _logging

    def configure_logging(**kwargs):  # type: ignore[misc]
        _logging.basicConfig(level=kwargs.get("level", "INFO"))

    get_logger = _logging.getLogger

log = get_logger(__name__)
settings = get_settings()


# ── Lifespan (startup / shutdown hooks) ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage long-lived resources.

    WHY lifespan instead of on_event?
      FastAPI ≥ 0.109 deprecated ``@app.on_event`` in favour of the lifespan
      context manager, which is cleaner and makes testing easier (you can
      override the lifespan in test fixtures).
    """
    # ── Startup ───────────────────────────────────────────────
    configure_logging(
        level=settings.log_level,
        json_output=(settings.log_format == "json"),
    )
    log.info("api_gateway.starting", port=8000, debug=settings.debug)

    # Create tables (dev convenience — use Alembic in production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Warm up Redis connection pool
    app.state.redis = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=20,
    )
    await app.state.redis.ping()
    log.info("redis.connected")

    # Initialise httpx connection pools for downstream services
    await init_clients()
    log.info("service_clients.initialised")

    # Start Kafka producer (best-effort — gateway doesn't critically need it)
    try:
        from app.services.kafka_producer import producer
        await producer.start()
        log.info("kafka_producer.started")
    except Exception as exc:
        log.warning("kafka_producer.start_failed", error=str(exc))

    yield  # ← application serves requests here

    # ── Shutdown ──────────────────────────────────────────────
    # Close Kafka producer
    try:
        from app.services.kafka_producer import producer
        await producer.stop()
    except Exception:
        pass

    # Close inter-service HTTP clients
    await close_clients()

    # Close Redis
    await app.state.redis.aclose()

    # Dispose SQLAlchemy engine
    await engine.dispose()

    log.info("api_gateway.stopped")


# ── Application factory ──────────────────────────────────────
app = FastAPI(
    title="Insider Threat Detection System — API Gateway",
    description=(
        "Unified REST API for the ITDS platform.  "
        "Routes requests to internal microservices and enforces "
        "authentication, rate-limiting, privacy controls, and "
        "immutable audit logging."
    ),
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# ── Middleware stack (order matters — outermost runs first) ────

# 1. CORS — must be outermost so pre-flight OPTIONS get correct headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Role"],
)

# 2. Privacy headers (GDPR/CCPA compliance signals).
app.add_middleware(PrivacyHeadersMiddleware)

# 3. JWT authentication — decodes bearer token, populates request.state.
app.add_middleware(JWTAuthMiddleware)

# 4. Rate limiter — per-role Redis-backed sliding window.
app.add_middleware(RateLimiterMiddleware)

# 5. Audit logger — append-only PostgreSQL record (innermost).
app.add_middleware(AuditLogMiddleware)

# ── Prometheus metrics ────────────────────────────────────────
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics")

# ── Route registration ────────────────────────────────────────

# Health check — no prefix, available at /health and /health/detailed
app.include_router(health.router)

# Auth routes — /api/v1/auth/*
app.include_router(auth_routes.router, prefix="/api/v1")

# User management — /api/v1/users/*
app.include_router(users.router, prefix="/api/v1")

# Alert proxy — /api/v1/alerts/*
app.include_router(alerts.router, prefix="/api/v1")

# Analytics & dashboard — /api/v1/dashboard, /api/v1/entities, etc.
app.include_router(analytics.router, prefix="/api/v1")

# Audit log query — /api/v1/audit/*
app.include_router(audit.router, prefix="/api/v1")

# HR integration — /api/v1/hr/*
app.include_router(hr_connector.router, prefix="/api/v1")
