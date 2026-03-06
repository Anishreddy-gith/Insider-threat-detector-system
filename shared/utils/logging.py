"""
Structured Logging — Shared Utility
====================================
Provides a single ``get_logger`` factory that every service imports.

Design decisions:
  • JSON-structured logs so they can be parsed by ELK / Loki / Datadog
    without custom grok patterns.
  • ``structlog`` wraps stdlib ``logging`` for zero-dependency compatibility
    with uvicorn's log capture.
  • Sensitive fields (passwords, tokens, PII) are redacted by the
    ``_redact_sensitive`` processor — defence-in-depth even if a developer
    accidentally logs a secret.
  • ``correlation_id`` is injected from context-vars set by the API-Gateway
    middleware so every downstream log line is traceable.
"""

from __future__ import annotations

import logging
import re
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# ── Context-propagated correlation ID ─────────────────────────
correlation_id_ctx: ContextVar[str | None] = ContextVar(
    "correlation_id", default=None
)

# Fields whose *values* must be masked in log output.
_SENSITIVE_KEYS: re.Pattern[str] = re.compile(
    r"(password|secret|token|api_key|authorization|ssn|credit_card)",
    re.IGNORECASE,
)


def _inject_correlation_id(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add the current correlation ID to every log entry."""
    cid = correlation_id_ctx.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def _redact_sensitive(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Mask values of keys that look like secrets.

    WHY: Even well-intentioned code can accidentally log credentials during
    error dumps.  This is a safety net — not a replacement for careful coding.
    """
    for key in list(event_dict.keys()):
        if _SENSITIVE_KEYS.search(key):
            event_dict[key] = "***REDACTED***"
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """
    Bootstrap structlog + stdlib logging.

    Call once at service startup (in ``main.py``).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_correlation_id,
        _redact_sensitive,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()  # pretty for local dev

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a context-aware structured logger bound to *name*.

    Usage::

        from shared.utils.logging import get_logger
        log = get_logger(__name__)
        log.info("user.login", user_id="u-42", ip="10.0.0.1")
    """
    return structlog.get_logger(name)
