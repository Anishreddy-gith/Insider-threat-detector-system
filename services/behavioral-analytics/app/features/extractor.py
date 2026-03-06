"""
Feature Extractor
==================
Transforms raw events into ML-ready feature vectors.

Feature categories (aligned with CERT insider threat research):
  1. **Temporal** — time-of-day, day-of-week, session duration, inter-event intervals
  2. **Volumetric** — event counts per window, bytes transferred, file counts
  3. **Behavioural** — new resources accessed, deviation from peer group
  4. **Access pattern** — sensitivity level distribution, after-hours ratio
  5. **Network** — unique destinations, unusual ports, encrypted ratio

All features are designed to be computable in real-time (no batch-only features)
so we can serve live inference.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from shared.utils.logging import get_logger

log = get_logger(__name__)

# Feature names in canonical order — order must match model input_dim.
FEATURE_NAMES: list[str] = [
    # Temporal (8)
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_business_hours",
    "session_duration_min",
    "time_since_last_event_sec",
    "events_in_last_hour",
    "events_in_last_24h",
    # Volumetric (6)
    "total_events_today",
    "file_access_count",
    "file_download_count",
    "bytes_transferred",
    "email_count",
    "print_count",
    # Access pattern (7)
    "unique_resources_accessed",
    "new_resources_ratio",
    "sensitivity_public_ratio",
    "sensitivity_internal_ratio",
    "sensitivity_confidential_ratio",
    "sensitivity_restricted_ratio",
    "after_hours_ratio",
    # Behavioural (6)
    "login_failures",
    "privilege_escalations",
    "usb_insertions",
    "vpn_connections",
    "unusual_app_usage_count",
    "peer_deviation_score",
    # Network (8)
    "unique_dst_ips",
    "unique_dst_ports",
    "unique_domains",
    "external_connection_ratio",
    "encrypted_traffic_ratio",
    "total_bytes_sent",
    "total_bytes_received",
    "new_destination_ratio",
    # Derived / risk indicators (10)
    "off_hours_sensitive_access",
    "bulk_download_indicator",
    "data_exfil_score",
    "lateral_movement_score",
    "credential_abuse_score",
    "policy_violation_count",
    "risk_score_ewma",
    "anomaly_score_ewma",
    "days_since_last_alert",
    "cumulative_alert_count",
]

NUM_FEATURES: int = len(FEATURE_NAMES)  # 45


class FeatureExtractor:
    """
    Stateful feature extractor that maintains per-user running statistics
    in Redis for real-time feature computation.

    Usage:
        extractor = FeatureExtractor(redis_client)
        features = await extractor.extract(user_id, event)
        # features is a np.ndarray of shape (NUM_FEATURES,)
    """

    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client

    async def extract(
        self, user_id: str, event: dict[str, Any]
    ) -> np.ndarray:
        """
        Extract a feature vector from a single event + user history.

        Steps:
          1. Parse temporal features from the event timestamp.
          2. Look up running counters from Redis.
          3. Compute derived features.
          4. Update Redis counters for the next event.
        """
        ts = self._parse_timestamp(event.get("timestamp"))
        features = np.zeros(NUM_FEATURES, dtype=np.float32)

        # ── Temporal features ─────────────────────────────────
        features[0] = ts.hour / 23.0  # normalised [0, 1]
        features[1] = ts.weekday() / 6.0
        features[2] = float(ts.weekday() >= 5)
        features[3] = float(9 <= ts.hour <= 17)

        # Session & inter-event timing from Redis.
        cache_key = f"feat:{user_id}"
        user_state = await self.redis.hgetall(cache_key)

        last_event_ts = user_state.get("last_event_ts")
        if last_event_ts:
            delta = (ts - datetime.fromisoformat(last_event_ts)).total_seconds()
            features[5] = min(delta / 3600.0, 24.0)  # cap at 24h

        features[6] = float(user_state.get("events_last_hour", 0))
        features[7] = float(user_state.get("events_last_24h", 0))

        # ── Volumetric features ───────────────────────────────
        features[8] = float(user_state.get("total_events_today", 0))
        features[9] = float(user_state.get("file_access_count", 0))
        features[10] = float(user_state.get("file_download_count", 0))
        features[11] = float(user_state.get("bytes_transferred", 0)) / 1e9  # normalise to GB
        features[12] = float(user_state.get("email_count", 0))
        features[13] = float(user_state.get("print_count", 0))

        # ── Access pattern features ───────────────────────────
        features[14] = float(user_state.get("unique_resources", 0))
        total_resources = max(float(user_state.get("total_resource_accesses", 1)), 1)
        features[15] = float(user_state.get("new_resources", 0)) / total_resources

        # Sensitivity distribution
        sens_total = max(sum(
            float(user_state.get(f"sens_{l}", 0))
            for l in ("public", "internal", "confidential", "restricted")
        ), 1)
        features[16] = float(user_state.get("sens_public", 0)) / sens_total
        features[17] = float(user_state.get("sens_internal", 0)) / sens_total
        features[18] = float(user_state.get("sens_confidential", 0)) / sens_total
        features[19] = float(user_state.get("sens_restricted", 0)) / sens_total
        features[20] = float(user_state.get("after_hours_events", 0)) / max(
            float(user_state.get("total_events_today", 1)), 1
        )

        # ── Behavioural features ──────────────────────────────
        features[21] = float(user_state.get("login_failures", 0))
        features[22] = float(user_state.get("privilege_escalations", 0))
        features[23] = float(user_state.get("usb_insertions", 0))
        features[24] = float(user_state.get("vpn_connections", 0))
        features[25] = float(user_state.get("unusual_apps", 0))
        features[26] = float(user_state.get("peer_deviation", 0))

        # ── Network features ─────────────────────────────────
        features[27] = float(user_state.get("unique_dst_ips", 0))
        features[28] = float(user_state.get("unique_dst_ports", 0))
        features[29] = float(user_state.get("unique_domains", 0))
        net_total = max(float(user_state.get("total_connections", 1)), 1)
        features[30] = float(user_state.get("external_connections", 0)) / net_total
        features[31] = float(user_state.get("encrypted_connections", 0)) / net_total
        features[32] = float(user_state.get("total_bytes_sent", 0)) / 1e9
        features[33] = float(user_state.get("total_bytes_recv", 0)) / 1e9
        features[34] = float(user_state.get("new_destinations", 0)) / net_total

        # ── Derived risk indicators ──────────────────────────
        # Off-hours access to sensitive files = compound risk signal.
        features[35] = features[18] * (1.0 - features[3])  # confidential × !business_hours
        # Bulk download: normalized file count × bytes.
        features[36] = min(features[10] * features[11], 1.0)
        # Placeholder scores (computed by specialised sub-models).
        features[37] = float(user_state.get("data_exfil_score", 0))
        features[38] = float(user_state.get("lateral_movement_score", 0))
        features[39] = float(user_state.get("credential_abuse_score", 0))
        features[40] = float(user_state.get("policy_violations", 0))

        # EWMA (exponentially weighted moving average) scores.
        alpha = 0.3  # smoothing factor — higher = more weight on recent events
        prev_risk_ewma = float(user_state.get("risk_score_ewma", 0))
        features[41] = alpha * features[35] + (1 - alpha) * prev_risk_ewma
        features[42] = float(user_state.get("anomaly_score_ewma", 0))
        features[43] = float(user_state.get("days_since_last_alert", 30)) / 30.0
        features[44] = float(user_state.get("cumulative_alert_count", 0))

        # ── Update Redis state for next event ─────────────────
        await self._update_state(cache_key, ts, event)

        return features

    async def _update_state(
        self, cache_key: str, ts: datetime, event: dict[str, Any]
    ) -> None:
        """Increment Redis counters for the current event."""
        pipe = self.redis.pipeline()
        pipe.hset(cache_key, "last_event_ts", ts.isoformat())
        pipe.hincrby(cache_key, "total_events_today", 1)
        pipe.hincrby(cache_key, "events_last_hour", 1)
        pipe.hincrby(cache_key, "events_last_24h", 1)

        action = event.get("action", "")
        if "file" in action:
            pipe.hincrby(cache_key, "file_access_count", 1)
        if action in ("file_download", "file_copy"):
            pipe.hincrby(cache_key, "file_download_count", 1)
        if action == "failed_login":
            pipe.hincrby(cache_key, "login_failures", 1)
        if action == "usb_insert":
            pipe.hincrby(cache_key, "usb_insertions", 1)

        # TTL: auto-expire after 48 hours to respect data retention.
        pipe.expire(cache_key, 172800)
        await pipe.execute()

    @staticmethod
    def _parse_timestamp(ts: Any) -> datetime:
        """Parse event timestamp to datetime."""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc)
