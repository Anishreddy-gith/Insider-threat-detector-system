"""
Privacy Response Headers Middleware
====================================
Adds HTTP headers that signal GDPR/CCPA compliance to browsers and proxies.

These headers don't *enforce* privacy on their own, but they:
  1. Instruct compliant browsers to limit tracking (``Permissions-Policy``).
  2. Signal to intermediary caches that responses contain private data
     (``Cache-Control: private``).
  3. Prevent the page from being embedded in third-party frames
     (``X-Frame-Options``) — mitigating click-jacking that could trick
     an analyst into approving a false negative.
"""

from __future__ import annotations

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class PrivacyHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security & privacy headers on every response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        # Prevent MIME type sniffing (XSS vector).
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Block framing to mitigate click-jacking.
        response.headers["X-Frame-Options"] = "DENY"

        # Strict transport security (preloaded in production via LB).
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )

        # Referrer policy — don't leak URLs to third parties.
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy — disable APIs we don't need.
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), interest-cohort=()"
        )

        # Private cache to prevent shared proxies from caching user data.
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "private, no-store"

        return response
