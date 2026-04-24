"""
JWT Authentication Middleware
==============================
Validates Bearer tokens on every request (except public endpoints).

Responsibilities:
  1. Generate / propagate ``X-Request-ID`` for distributed tracing.
  2. Verify ``Authorization: Bearer <token>`` on protected routes.
  3. Inject decoded user context into ``request.state`` so downstream
     handlers and middleware (rate-limiter, audit-logger) can read
     ``user_id``, ``role``, etc.

This middleware MUST execute before the rate-limiter and audit-logger
so those layers have access to user identity.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Request, Response
from jose import JWTError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from services.api_gateway.app.auth.jwt_handler import verify_access_token

try:
    from shared.utils.logging import correlation_id_ctx, get_logger
except ImportError:
    # Fallback if shared module isn't on the path during tests.
    import logging as _logging
    from contextvars import ContextVar
    correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")
    def get_logger(name: str):  # type: ignore[misc]
        return _logging.getLogger(name)

log = get_logger(__name__)

# Paths that do NOT require authentication.
_PUBLIC_PREFIXES: frozenset[str] = frozenset({
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/refresh",
    "/api/v1/hr/webhook",
})


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that verifies JWT access tokens and injects
    user claims into the request state.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # â”€â”€ Request-ID propagation (distributed tracing) â”€â”€â”€â”€â”€â”€
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        correlation_id_ctx.set(request_id)

        # â”€â”€ Skip auth for public endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        path = request.url.path
        if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # â”€â”€ Extract Bearer token â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        auth_header: str | None = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or malformed Authorization header."},
                headers={"X-Request-ID": request_id},
            )

        token = auth_header.split(" ", 1)[1]

        # â”€â”€ Verify token â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            payload: dict[str, Any] = verify_access_token(token)
        except JWTError as exc:
            log.warning("auth.token_invalid", error=str(exc))
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token."},
                headers={"X-Request-ID": request_id},
            )

        # â”€â”€ Inject user context for downstream layers â”€â”€â”€â”€â”€â”€â”€â”€â”€
        request.state.user = payload
        request.state.user_id = payload.get("sub")
        request.state.role = payload.get("role", "SOC_ANALYST")

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

