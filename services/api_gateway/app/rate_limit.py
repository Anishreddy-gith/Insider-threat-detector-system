from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 120) -> None:
        super().__init__(app)
        self.requests_per_minute = max(1, requests_per_minute)
        self.window_seconds = 60.0
        self.buckets: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in {"/ingest"} and not request.url.path.startswith("/risk"):
            return await call_next(request)

        client_host = request.client.host if request.client else "unknown"
        now = time.time()
        q = self.buckets[client_host]

        while q and (now - q[0]) > self.window_seconds:
            q.popleft()

        if len(q) >= self.requests_per_minute:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        q.append(now)
        return await call_next(request)
