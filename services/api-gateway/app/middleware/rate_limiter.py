"""
Per-Role Redis Rate Limiter
=============================
Enforces different request-rate caps for each RBAC role.

Why per-role?
  • **HR_SYSTEM** accounts push bulk employment data — they should be
    heavily throttled (30 req/min) to prevent a compromised HR connector
    from flooding the gateway.
  • **SOC_ANALYST** dashboards poll frequently — 120 req/min is generous
    but avoids runaway browser tabs from overwhelming the backend.
  • **ADMIN** accounts may run batch operations — 300 req/min.
  • Unauthenticated (anonymous) traffic gets the default cap (60 req/min)
    applied by IP.

Algorithm: fixed-window counter in Redis.
  Key pattern: ``rl:<role>:<identifier>:<window_epoch>``
  The *identifier* is the user UUID for authenticated requests, or the
  client IP for anonymous ones.
"""

from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from app.config import get_settings

settings = get_settings()

# Paths exempt from rate limiting.
_EXEMPT: frozenset[str] = frozenset({"/health", "/metrics"})


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Redis-backed per-role fixed-window rate limiter.

    After the JWT middleware has run, ``request.state.role`` is available.
    If it is not set (unauthenticated), the limiter falls back to IP-based
    keying with the default cap.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in _EXEMPT:
            return await call_next(request)

        redis = getattr(request.app.state, "redis", None)
        if redis is None:
            return await call_next(request)

        # Determine role + identifier ──────────────────────────
        role: str = getattr(request.state, "role", "anonymous")
        user_id: str | None = getattr(request.state, "user_id", None)
        client_ip = request.client.host if request.client else "unknown"
        identifier = user_id or client_ip

        max_requests = settings.rate_limit_for_role(role)
        window = settings.rate_limit_window_seconds

        # Fixed-window counter ─────────────────────────────────
        epoch_window = int(time.time()) // window
        key = f"rl:{role}:{identifier}:{epoch_window}"

        pipe = redis.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, window)
        count, _ = await pipe.execute()

        remaining = max(0, max_requests - count)

        if count > max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Try again later.",
                    "role": role,
                    "limit": max_requests,
                    "window_seconds": window,
                },
                headers={
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Role": role,
                    "Retry-After": str(window),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Role"] = role
        return response
