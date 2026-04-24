from __future__ import annotations

import json
import os
from typing import Any

import redis


class RiskRedisStore:
    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.prefix = "risk:latest:"
        self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)

    def _key(self, user_id: str) -> str:
        return f"{self.prefix}{user_id}"

    def get_latest(self, user_id: str) -> dict[str, Any] | None:
        value = self._client.get(self._key(user_id))
        if not value:
            return None
        try:
            payload = json.loads(value)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    def set_latest(self, user_id: str, payload: dict[str, Any]) -> None:
        self._client.set(self._key(user_id), json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
