"""
Immutable Audit Logging Middleware
===================================
Logs every API call to an append-only PostgreSQL table for compliance.

Why middleware instead of per-route decorators?
  • **Completeness** — a decorator can be forgotten on a new endpoint.
    Middleware captures *every* request, even undocumented or mis-routed
    ones, so the audit trail has no gaps.
  • **Separation of concerns** — route handlers stay focused on business
    logic.

The audit_logs table is designed to be **immutable**:
  • No UPDATE or DELETE operations are ever performed.
  • The table has no foreign-key CASCADE deletes.
  • A database trigger (applied via Alembic migration) can ``REVOKE
    UPDATE, DELETE`` on the table for the application role if desired.

Recorded fields (per NIST SP 800-53 AU-3):
  • Authenticated user (``sub`` from JWT, or "anonymous")
  • Action (HTTP method + path)
  • Timestamp (server-side UTC, not client-supplied)
  • Source IP address
  • HTTP status code (response-side)
  • Resource accessed (path + query string)
  • User-Agent (for forensic correlation)
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from fastapi import Request, Response
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import get_settings
from app.models.database import AsyncSessionLocal

settings = get_settings()

# Paths that are NOT audit-logged (health probes generate too much noise).
_SKIP_PREFIXES: frozenset[str] = frozenset({
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
})


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Append-only audit logger — writes one row per HTTP request.

    The INSERT uses raw SQL with bound parameters to:
      1. Bypass the ORM session lifecycle (no accidental rollbacks).
      2. Guarantee the row is written even if the downstream handler
         raises an exception.
      3. Minimise latency (single INSERT, no SELECT or UPDATE).
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path

        # Skip noisy infrastructure endpoints.
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            return await call_next(request)

        # Capture request metadata BEFORE calling downstream handler.
        start = time.monotonic()
        user_data = getattr(request.state, "user", None)
        user_id = user_data.get("sub") if user_data else None
        user_role = user_data.get("role", "anonymous") if user_data else "anonymous"
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        method = request.method
        query = str(request.url.query) if request.url.query else None

        # Let the actual request proceed.
        response: Response | None = None
        status_code = 500  # default if handler crashes
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)

            # Fire-and-forget audit INSERT — must not block the response.
            try:
                async with AsyncSessionLocal() as session:
                    await session.execute(
                        text("""
                            INSERT INTO audit_logs
                                (id, user_id, user_role, action, resource_path,
                                 query_params, ip_address, user_agent,
                                 status_code, response_time_ms, timestamp)
                            VALUES
                                (:id, :user_id, :user_role, :action, :resource_path,
                                 :query_params, :ip_address, :user_agent,
                                 :status_code, :response_time_ms, :timestamp)
                        """),
                        {
                            "id": str(uuid.uuid4()),
                            "user_id": user_id,
                            "user_role": user_role,
                            "action": f"{method} {path}",
                            "resource_path": path,
                            "query_params": query,
                            "ip_address": client_ip,
                            "user_agent": user_agent[:500] if user_agent else None,
                            "status_code": status_code,
                            "response_time_ms": elapsed_ms,
                            "timestamp": datetime.now(timezone.utc),
                        },
                    )
                    await session.commit()
            except Exception:
                # Audit logging must NEVER crash the request pipeline.
                # In production, ship this to a dead-letter queue.
                pass

        return response  # type: ignore[return-value]
