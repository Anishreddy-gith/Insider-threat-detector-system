"""
Context-Aware Scoring Engine.

Adjusts raw ML anomaly scores based on business context fetched from
an HR API (or mock).  Rules are loaded from a YAML file and applied
as multiplicative factors in priority order.

Why context matters
───────────────────
A raw ML score treats every user identically.  But a finance analyst
downloading 500 spreadsheets during quarter-end close is *normal*,
while the same behaviour from a departing engineer is *critical*.
The context engine bridges this gap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
import yaml

from app.config import get_settings
from app.schemas import BusinessContext, ContextAdjustment

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Rule loader ─────────────────────────────────────────────────────

def load_rules(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load context rules from a YAML file, sorted by priority."""
    p = Path(path or settings.CONTEXT_RULES_PATH)
    if not p.exists():
        logger.warning("context_rules_not_found", extra={"path": str(p)})
        return []
    with open(p) as f:
        data = yaml.safe_load(f)
    rules = data.get("rules", [])
    rules.sort(key=lambda r: r.get("priority", 100))
    return rules


_RULES: list[dict[str, Any]] | None = None


def _get_rules() -> list[dict[str, Any]]:
    global _RULES
    if _RULES is None:
        _RULES = load_rules()
    return _RULES


def reload_rules(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Force-reload rules (e.g. after YAML file update)."""
    global _RULES
    _RULES = load_rules(path)
    return _RULES


# ── Mock HR API client ──────────────────────────────────────────────

async def fetch_hr_context(user_id: str) -> dict[str, Any]:
    """
    Fetch business context from the HR API.

    Falls back to a mock response if the HR service is unreachable.
    """
    try:
        async with httpx.AsyncClient(timeout=settings.HR_API_TIMEOUT) as client:
            resp = await client.get(
                f"{settings.HR_API_URL}/api/v1/employees/{user_id}"
            )
            if resp.status_code == 200:
                return resp.json()
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.warning(
            "hr_api_unavailable",
            extra={"user_id": user_id, "url": settings.HR_API_URL},
        )

    # ── Mock fallback ────────────────────────────────────────────
    return _mock_hr_context(user_id)


def _mock_hr_context(user_id: str) -> dict[str, Any]:
    """
    Deterministic mock HR context for development / testing.

    Uses the user_id hash to generate varied but reproducible contexts.
    """
    h = hash(user_id)
    return {
        "user_id": user_id,
        "employment_status": "active",
        "department": ["engineering", "finance", "hr", "it_ops", "executive"][
            h % 5
        ],
        "is_on_pto": (h % 7 == 0),
        "recently_promoted": (h % 11 == 0),
        "submitted_resignation": (h % 23 == 0),
        "is_new_hire": (h % 13 == 0),
        "current_projects": [f"project-{(h >> i) % 100}" for i in range(3)],
        "manager": f"mgr-{h % 50}",
        "office_hours": {"start": 9, "end": 17},
    }


# ── Context engine ──────────────────────────────────────────────────

class ContextEngine:
    """
    Applies YAML-defined rules to adjust raw ML scores.

    Usage::

        engine = ContextEngine()
        result = await engine.adjust(
            user_id="u-123",
            raw_score=0.72,
            event_metadata={"resource": "/data/project-42/report.xlsx"},
        )
        # result.adjusted_score, result.context, result.adjustments
    """

    def __init__(self, rules: list[dict[str, Any]] | None = None) -> None:
        self.rules = rules if rules is not None else _get_rules()

    async def adjust(
        self,
        user_id: str,
        raw_score: float,
        event_metadata: dict[str, Any] | None = None,
    ) -> ContextResult:
        """
        Fetch HR context → evaluate rules → return adjusted score.
        """
        hr_data = await fetch_hr_context(user_id)
        context = self._build_business_context(hr_data, event_metadata)
        adjustments: list[ContextAdjustment] = []
        adjusted = raw_score

        for rule in self.rules:
            condition = rule["condition"]
            expected = rule["value"]
            factor = rule["factor"]

            # Evaluate condition against context
            actual = self._evaluate_condition(
                condition, context, hr_data, event_metadata
            )
            if actual == expected:
                adjusted *= factor
                adjustments.append(
                    ContextAdjustment(
                        rule=rule["name"],
                        factor=factor,
                        reason=rule["reason"],
                    )
                )
                logger.info(
                    "context_rule_applied",
                    extra={
                        "user_id": user_id,
                        "rule": rule["name"],
                        "factor": factor,
                        "score_before": round(raw_score, 4),
                        "score_after": round(adjusted, 4),
                    },
                )

        # Clamp to [0, 1]
        adjusted = max(0.0, min(1.0, adjusted))
        context.adjustments_applied = adjustments

        return ContextResult(
            adjusted_score=adjusted,
            context=context,
            adjustments=adjustments,
        )

    # ── internal helpers ────────────────────────────────────────────

    @staticmethod
    def _build_business_context(
        hr_data: dict[str, Any],
        event_metadata: dict[str, Any] | None,
    ) -> BusinessContext:
        return BusinessContext(
            employment_status=hr_data.get("employment_status", "active"),
            department=hr_data.get("department", "unknown"),
            is_on_pto=hr_data.get("is_on_pto", False),
            recently_promoted=hr_data.get("recently_promoted", False),
            submitted_resignation=hr_data.get("submitted_resignation", False),
            current_projects=hr_data.get("current_projects", []),
        )

    @staticmethod
    def _evaluate_condition(
        condition: str,
        context: BusinessContext,
        hr_data: dict[str, Any],
        event_metadata: dict[str, Any] | None,
    ) -> Any:
        """
        Evaluate a rule condition against the available context.

        Supports direct field lookups on BusinessContext / hr_data,
        plus special composite conditions like ``resource_matches_project``.
        """
        # Special composite conditions
        if condition == "resource_matches_project":
            if not event_metadata or "resource" not in event_metadata:
                return False
            resource = event_metadata["resource"].lower()
            return any(
                proj.lower() in resource
                for proj in context.current_projects
            )

        if condition == "after_hours_non_ops":
            if not event_metadata or "timestamp" not in event_metadata:
                return False
            if context.department in ("it_ops", "security"):
                return False
            hour = event_metadata.get("hour")
            if hour is None:
                ts = event_metadata.get("timestamp")
                if hasattr(ts, "hour"):
                    hour = ts.hour
                else:
                    return False
            hours = hr_data.get("office_hours", {"start": 9, "end": 17})
            return hour < hours["start"] or hour >= hours["end"]

        # Direct field lookup
        if hasattr(context, condition):
            return getattr(context, condition)
        return hr_data.get(condition, False)


class ContextResult:
    """Container for context-adjustment output."""

    __slots__ = ("adjusted_score", "context", "adjustments")

    def __init__(
        self,
        adjusted_score: float,
        context: BusinessContext,
        adjustments: list[ContextAdjustment],
    ) -> None:
        self.adjusted_score = adjusted_score
        self.context = context
        self.adjustments = adjustments
