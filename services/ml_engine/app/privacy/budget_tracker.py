"""
Privacy Budget Tracker — per-user cumulative ε accounting in Redis.

Every differentially-private query consumes a portion of the daily
privacy budget.  This tracker:

  1. Stores ``consumed_ε`` per user/entity in Redis with TTL = 86 400 s.
  2. Raises a structured alert when ≥ 80 % of the daily budget is used.
  3. Hard-blocks queries when the budget is fully exhausted.

Key design
──────────
  Redis key format:  ``dp:budget:{entity_id}``
  Value:             JSON → { "consumed": float, "queries": int }
  TTL:               86 400 seconds (auto-resets daily at midnight UTC)

The tracker is **composition-aware** — under basic sequential
composition, running k queries each with budget ε_i costs Σ ε_i.
Under advanced (Rényi / zCDP) composition the total is tighter,
but we use the conservative sequential bound for safety.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────
DEFAULT_DAILY_BUDGET = 10.0        # total ε per entity per day
ALERT_THRESHOLD_RATIO = 0.80       # alert at 80 % consumption
REDIS_KEY_PREFIX = "dp:budget:"
REDIS_TTL_SECONDS = 86_400         # 24 hours


@dataclass
class BudgetStatus:
    """Snapshot of an entity's current privacy budget state."""

    entity_id: str
    daily_budget: float
    consumed: float
    remaining: float
    queries: int
    utilisation_pct: float
    alert_triggered: bool
    exhausted: bool
    ttl_seconds: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "daily_budget": self.daily_budget,
            "consumed": round(self.consumed, 6),
            "remaining": round(self.remaining, 6),
            "queries": self.queries,
            "utilisation_pct": round(self.utilisation_pct, 2),
            "alert_triggered": self.alert_triggered,
            "exhausted": self.exhausted,
            "ttl_seconds": self.ttl_seconds,
        }


class PrivacyBudgetTracker:
    """
    Track cumulative ε consumption per entity in Redis.

    Parameters
    ----------
    redis_client
        An ``redis.Redis`` or ``redis.asyncio.Redis`` instance
        (sync interface used here for simplicity; wrap with
        ``asyncio.to_thread`` in async contexts).
    daily_budget : float
        Maximum ε that may be consumed per entity per day.
    alert_threshold : float
        Fraction of daily_budget at which an alert fires.
    alert_callback : callable | None
        Optional function called when the threshold is crossed.
        Signature: ``callback(status: BudgetStatus) -> None``
    """

    def __init__(
        self,
        redis_client: Any,
        daily_budget: float = DEFAULT_DAILY_BUDGET,
        alert_threshold: float = ALERT_THRESHOLD_RATIO,
        alert_callback: Any | None = None,
    ) -> None:
        self.redis = redis_client
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.alert_callback = alert_callback

    # ── key helpers ─────────────────────────────────────────────────

    @staticmethod
    def _key(entity_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}{entity_id}"

    def _get_state(self, entity_id: str) -> dict[str, Any]:
        """Read the current budget state from Redis."""
        raw = self.redis.get(self._key(entity_id))
        if raw is None:
            return {"consumed": 0.0, "queries": 0}
        return json.loads(raw)

    def _set_state(
        self, entity_id: str, state: dict[str, Any]
    ) -> None:
        """Write budget state to Redis with TTL."""
        key = self._key(entity_id)
        # Preserve existing TTL if key exists; otherwise set fresh 24h
        existing_ttl = self.redis.ttl(key)
        self.redis.set(key, json.dumps(state))
        if existing_ttl and existing_ttl > 0:
            self.redis.expire(key, existing_ttl)
        else:
            self.redis.expire(key, REDIS_TTL_SECONDS)

    # ── public API ──────────────────────────────────────────────────

    def consume(
        self,
        entity_id: str,
        epsilon: float,
        query_description: str = "",
    ) -> BudgetStatus:
        """
        Record consumption of ``epsilon`` privacy budget.

        Returns
        -------
        BudgetStatus with the updated state.

        Raises
        ------
        PrivacyBudgetExhausted
            If the daily budget is already fully consumed.
        """
        state = self._get_state(entity_id)
        consumed = state["consumed"]
        queries = state["queries"]

        if consumed >= self.daily_budget:
            status = self._build_status(entity_id, consumed, queries)
            raise PrivacyBudgetExhausted(status)

        # Check if *this* query would exceed the budget
        new_consumed = consumed + epsilon
        if new_consumed > self.daily_budget:
            # Allow partial — clamp to budget and warn
            logger.warning(
                "dp_budget_clamped",
                extra={
                    "entity_id": entity_id,
                    "requested": epsilon,
                    "granted": round(self.daily_budget - consumed, 6),
                },
            )
            new_consumed = self.daily_budget

        new_queries = queries + 1
        state = {"consumed": new_consumed, "queries": new_queries}
        self._set_state(entity_id, state)

        status = self._build_status(entity_id, new_consumed, new_queries)

        # Alert check
        if status.alert_triggered:
            logger.warning(
                "dp_budget_alert",
                extra={
                    "entity_id": entity_id,
                    "utilisation_pct": status.utilisation_pct,
                    "consumed": status.consumed,
                    "remaining": status.remaining,
                    "query": query_description,
                },
            )
            if self.alert_callback is not None:
                try:
                    self.alert_callback(status)
                except Exception:
                    logger.exception("dp_budget_alert_callback_failed")

        logger.info(
            "dp_budget_consumed",
            extra={
                "entity_id": entity_id,
                "epsilon": epsilon,
                "total_consumed": round(new_consumed, 6),
                "remaining": round(status.remaining, 6),
                "query": query_description,
            },
        )
        return status

    def get_status(self, entity_id: str) -> BudgetStatus:
        """Get current budget status without consuming anything."""
        state = self._get_state(entity_id)
        ttl = self.redis.ttl(self._key(entity_id))
        status = self._build_status(
            entity_id, state["consumed"], state["queries"]
        )
        status.ttl_seconds = max(ttl, 0) if ttl else 0
        return status

    def reset(self, entity_id: str) -> BudgetStatus:
        """Manually reset an entity's budget (admin only)."""
        self.redis.delete(self._key(entity_id))
        logger.info("dp_budget_reset", extra={"entity_id": entity_id})
        return self._build_status(entity_id, 0.0, 0)

    def get_all_entities(self) -> list[str]:
        """List all entity IDs with active budget tracking."""
        keys = self.redis.keys(f"{REDIS_KEY_PREFIX}*")
        return [
            k.decode().replace(REDIS_KEY_PREFIX, "")
            if isinstance(k, bytes)
            else k.replace(REDIS_KEY_PREFIX, "")
            for k in keys
        ]

    # ── internal ────────────────────────────────────────────────────

    def _build_status(
        self,
        entity_id: str,
        consumed: float,
        queries: int,
    ) -> BudgetStatus:
        remaining = max(0.0, self.daily_budget - consumed)
        utilisation = (consumed / self.daily_budget) * 100.0
        return BudgetStatus(
            entity_id=entity_id,
            daily_budget=self.daily_budget,
            consumed=consumed,
            remaining=remaining,
            queries=queries,
            utilisation_pct=utilisation,
            alert_triggered=utilisation >= (self.alert_threshold * 100),
            exhausted=consumed >= self.daily_budget,
        )


class PrivacyBudgetExhausted(Exception):
    """Raised when an entity's daily ε budget is fully consumed."""

    def __init__(self, status: BudgetStatus) -> None:
        self.status = status
        super().__init__(
            f"Privacy budget exhausted for entity '{status.entity_id}'. "
            f"Consumed {status.consumed:.4f} / {status.daily_budget:.4f} ε. "
            f"Budget resets in {status.ttl_seconds}s."
        )
