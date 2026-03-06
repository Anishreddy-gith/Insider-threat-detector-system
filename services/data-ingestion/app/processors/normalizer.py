"""
Event Normaliser
=================
Converts heterogeneous raw events into canonical payloads that match our
shared Pydantic schemas.

Design decisions:
  • Normalisation happens *before* Kafka publishing (schema-on-write)
    rather than in consumers (schema-on-read).  This pushes validation
    cost to the edge, keeping the analytics pipeline cleaner.
  • Missing fields get sensible defaults rather than causing rejection,
    because insider-threat detection benefits from partial data
    (better some signal than none).
  • Timestamps are coerced to UTC ISO-8601 regardless of source format.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from shared.utils.logging import get_logger

log = get_logger(__name__)

# Regex patterns for extracting known fields from syslog messages.
_USER_PATTERN = re.compile(r"user[=:\s]+(\S+)", re.IGNORECASE)
_IP_PATTERN = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")
_ACTION_PATTERN = re.compile(
    r"\b(login|logout|failed_login|file_download|file_upload|usb_insert|"
    r"print_job|email_send|sudo|ssh|rdp|vpn_connect|vpn_disconnect)\b",
    re.IGNORECASE,
)


class EventNormalizer:
    """
    Stateless normaliser.  Pre-compiles regex patterns at construction time
    so we pay the compilation cost once (amortised across millions of events).
    """

    def __init__(self) -> None:
        log.info("normalizer.initialized")

    def normalise_syslog(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Transform a raw syslog event into a canonical user-activity dict.

        Extraction strategy:
          1. Check ``structured_data`` first (RFC 5424 compliant sources).
          2. Fall back to regex parsing of the ``message`` field.
        """
        structured = raw.get("structured_data", {})
        message = raw.get("message", "")

        # Attempt structured extraction first.
        user_id = structured.get("user") or structured.get("uid")
        ip_address = structured.get("src_ip") or structured.get("ip")
        action = structured.get("action") or structured.get("event")

        # Fall back to regex.
        if not user_id:
            match = _USER_PATTERN.search(message)
            user_id = match.group(1) if match else "unknown"

        if not ip_address:
            match = _IP_PATTERN.search(message)
            ip_address = match.group(1) if match else None

        if not action:
            match = _ACTION_PATTERN.search(message)
            action = match.group(1).lower() if match else "unknown"

        return {
            "user_id": user_id,
            "session_id": structured.get("session_id", ""),
            "action": action,
            "resource": structured.get("resource"),
            "ip_address": ip_address,
            "user_agent": structured.get("user_agent"),
            "metadata": {
                "facility": raw.get("facility"),
                "severity": raw.get("severity"),
                "hostname": raw.get("hostname"),
                "app_name": raw.get("app_name"),
                "raw_message": message[:500],  # truncate to prevent oversized payloads
            },
            "timestamp": self._normalise_timestamp(
                structured.get("timestamp") or raw.get("timestamp")
            ),
        }

    def normalise_endpoint(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalise an endpoint agent event."""
        return {
            "user_id": raw.get("user_id", "unknown"),
            "session_id": raw.get("agent_id", ""),
            "action": raw.get("event_type", "unknown"),
            "resource": raw.get("details", {}).get("resource"),
            "ip_address": raw.get("details", {}).get("ip_address"),
            "user_agent": None,
            "metadata": {
                "hostname": raw.get("hostname"),
                "agent_id": raw.get("agent_id"),
                **raw.get("details", {}),
            },
            "timestamp": self._normalise_timestamp(raw.get("timestamp")),
        }

    @staticmethod
    def _normalise_timestamp(ts: Any) -> str:
        """
        Coerce various timestamp formats to ISO-8601 UTC.

        Handles:
          • ISO-8601 strings (with/without timezone)
          • Unix epoch (int/float)
          • None → now()
        """
        if ts is None:
            return datetime.now(timezone.utc).isoformat()

        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if isinstance(ts, str):
            # Try ISO parse; if it fails, return current time.
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except ValueError:
                log.warning("normalizer.timestamp_parse_failed", raw=ts)
                return datetime.now(timezone.utc).isoformat()

        return datetime.now(timezone.utc).isoformat()
