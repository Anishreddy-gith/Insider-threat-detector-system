from __future__ import annotations

import os
from typing import Any

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, protected_prefixes: tuple[str, ...] = ("/ingest", "/risk")) -> None:
        super().__init__(app)
        self.protected_prefixes = protected_prefixes
        self.jwt_secret = os.getenv("JWT_SECRET", "change_me_in_production")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.demo_mode = os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

    def _is_protected(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.protected_prefixes)

    async def dispatch(self, request: Request, call_next):
        if self.demo_mode:
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        if not self._is_protected(request.url.path):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing bearer token"})

        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            return JSONResponse(status_code=401, content={"detail": "Invalid bearer token"})

        try:
            payload: dict[str, Any] = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                options={"verify_aud": False},
            )
        except jwt.PyJWTError:
            return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})

        request.state.user = payload.get("sub") or payload.get("user_id") or "unknown"
        request.state.jwt_payload = payload
        return await call_next(request)
