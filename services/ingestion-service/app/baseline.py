"""
BaselineEngine – 30-day rolling behavioural baseline per user.

Uses **Welford's online algorithm** to maintain streaming mean and
variance in O(1) per event, backed by Redis for sub-millisecond reads.

Redis key layout
────────────────
    baseline:{user_id}:{metric_name}   →  HASH {
        count, mean, m2, min, max, last, window_start, window_end
    }

Why Welford?
────────────
Batch statistics require re-reading the full 30-day window on every
update.  With 5 000 events/s flowing through ingestion, that would
mean ~13 billion reads/day.  Welford's algorithm instead keeps three
running accumulators (count, mean, M₂) and updates them in constant
time per sample:

    δ   = x − meanₙ₋₁
    meanₙ = meanₙ₋₁ + δ / n
    M₂ₙ  = M₂ₙ₋₁ + δ × (x − meanₙ)

Variance = M₂ / (n − 1)  (Bessel-corrected).

This is numerically stable even for large n — unlike the naive formula
Var = E[X²] − (E[X])² which suffers catastrophic cancellation.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import redis.asyncio as aioredis

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)

BASELINE_WINDOW_DAYS = 30
_KEY_PREFIX = "baseline"


@dataclass
class BaselineState:
    """In-memory snapshot of a single user×metric baseline."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0            # sum of squared deviations
    min_value: float = math.inf
    max_value: float = -math.inf
    last_value: float = 0.0
    window_start: str = ""
    window_end: str = ""

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, value: float) -> float:
        """How many σ is *value* from the current mean?"""
        if self.std_dev == 0:
            return 0.0
        return (value - self.mean) / self.std_dev

    def to_dict(self) -> dict:
        return {
            "count": str(self.count),
            "mean": str(self.mean),
            "m2": str(self.m2),
            "min_value": str(self.min_value) if self.min_value != math.inf else "0",
            "max_value": str(self.max_value) if self.max_value != -math.inf else "0",
            "last_value": str(self.last_value),
            "window_start": self.window_start,
            "window_end": self.window_end,
        }

    @classmethod
    def from_redis(cls, raw: dict[bytes, bytes]) -> "BaselineState":
        """Rehydrate from a Redis HASH."""
        if not raw:
            return cls()
        d = {k.decode(): v.decode() for k, v in raw.items()}
        return cls(
            count=int(d.get("count", 0)),
            mean=float(d.get("mean", 0.0)),
            m2=float(d.get("m2", 0.0)),
            min_value=float(d.get("min_value", 0.0)),
            max_value=float(d.get("max_value", 0.0)),
            last_value=float(d.get("last_value", 0.0)),
            window_start=d.get("window_start", ""),
            window_end=d.get("window_end", ""),
        )


class BaselineEngine:
    """
    Maintains per-user, per-metric rolling baselines using Welford's
    online algorithm with Redis-backed state.

    Typical metrics:
        login_count, file_access_count, file_bytes_total,
        network_bytes_total, app_duration_total, off_hours_ratio
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis: aioredis.Redis | None = None
        self._redis_url = redis_url or settings.REDIS_URL

    # ── lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=False,   # we handle decoding manually
        )
        logger.info("baseline_engine_started")

    async def stop(self) -> None:
        if self._redis:
            await self._redis.aclose()
            logger.info("baseline_engine_stopped")

    # ── core API ────────────────────────────────────────────────────

    def _key(self, user_id: str, metric: str) -> str:
        return f"{_KEY_PREFIX}:{user_id}:{metric}"

    async def get_baseline(self, user_id: str, metric: str) -> BaselineState:
        """Fetch the current baseline state from Redis."""
        assert self._redis is not None
        raw = await self._redis.hgetall(self._key(user_id, metric))
        return BaselineState.from_redis(raw)

    async def update(
        self,
        user_id: str,
        metric: str,
        value: float,
        event_time: datetime | None = None,
    ) -> BaselineState:
        """
        Welford update: O(1) streaming mean + variance.

        Returns the *new* baseline state after incorporating ``value``.
        """
        assert self._redis is not None
        now = event_time or datetime.now(timezone.utc)
        now_iso = now.isoformat()
        key = self._key(user_id, metric)

        state = await self.get_baseline(user_id, metric)

        # ── Window expiry: reset if last observation > 30 days ago ──
        if state.window_end:
            last_end = datetime.fromisoformat(state.window_end)
            if (now - last_end) > timedelta(days=BASELINE_WINDOW_DAYS):
                state = BaselineState()     # full reset

        # ── Welford step ──
        state.count += 1
        delta = value - state.mean
        state.mean += delta / state.count
        delta2 = value - state.mean
        state.m2 += delta * delta2

        # ── Min / max / last ──
        if value < state.min_value:
            state.min_value = value
        if value > state.max_value:
            state.max_value = value
        state.last_value = value

        # ── Window bounds ──
        if not state.window_start:
            state.window_start = now_iso
        state.window_end = now_iso

        # ── Persist to Redis ──
        await self._redis.hset(key, mapping=state.to_dict())
        await self._redis.expire(key, BASELINE_WINDOW_DAYS * 86400)

        return state

    async def check_anomaly(
        self,
        user_id: str,
        metric: str,
        value: float,
        z_threshold: float = 3.0,
    ) -> dict:
        """
        Compare *value* against the user's baseline and return an
        anomaly assessment.

        An observation is flagged anomalous when |z-score| ≥ z_threshold
        (default 3σ) AND the baseline has at least 10 observations.
        """
        state = await self.get_baseline(user_id, metric)
        z = state.z_score(value)
        is_anomalous = abs(z) >= z_threshold and state.count >= 10

        return {
            "user_id": user_id,
            "metric": metric,
            "value": value,
            "baseline_mean": round(state.mean, 4),
            "baseline_std": round(state.std_dev, 4),
            "z_score": round(z, 4),
            "is_anomalous": is_anomalous,
            "observation_count": state.count,
        }

    async def get_all_baselines(self, user_id: str) -> dict[str, BaselineState]:
        """Return every metric baseline for a given user."""
        assert self._redis is not None
        pattern = f"{_KEY_PREFIX}:{user_id}:*"
        result: dict[str, BaselineState] = {}
        async for key in self._redis.scan_iter(match=pattern, count=100):
            metric = key.decode().split(":")[-1]
            raw = await self._redis.hgetall(key)
            result[metric] = BaselineState.from_redis(raw)
        return result
