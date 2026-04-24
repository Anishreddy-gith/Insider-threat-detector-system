"""
SIEM Integration Connector â€” Splunk & Microsoft Sentinel
==========================================================
Pushes high-severity insider-threat alerts to enterprise SIEM
platforms in their native ingestion formats.

Why push to SIEM?
  â€¢ SOC teams already live in Splunk / Sentinel â€” meeting them where
    they work reduces the mean time to acknowledge (MTTA).
  â€¢ SIEM correlation engines can cross-reference our alerts with
    network IDS, EDR, and DLP signals for richer context.
  â€¢ Compliance frameworks (SOC 2, ISO 27001) require centralised
    logging â€” the SIEM is the system of record.

Supported targets
-----------------
1. **Splunk HTTP Event Collector (HEC)** â€” pushes JSON events to
   ``/services/collector/event`` with an HEC token.
2. **Microsoft Sentinel Data Collector API** â€” pushes JSON to the
   Log Analytics workspace using shared-key HMAC-SHA256 auth.

Both include the full XAI metadata (model attributions, SHAP
features, baseline comparisons) so the SIEM analyst can triage
without switching tools.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
from typing import Any

import httpx

from services.api_gateway.app.config import get_settings

try:
    from shared.utils.logging import get_logger
except ImportError:
    import logging as _logging
    get_logger = _logging.getLogger

log = get_logger(__name__)
settings = get_settings()


# â”€â”€ Splunk HEC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _format_splunk_event(alert: dict[str, Any]) -> dict[str, Any]:
    """
    Format an alert into a Splunk HEC event payload.

    Splunk HEC expects::
        {
            "event": { ... },
            "source": "itds-api-gateway",
            "sourcetype": "_json",
            "index": "insider_threats",
            "time": <epoch_seconds>
        }

    We include the full XAI report, evidence items, and entity
    metadata in the ``event`` body so Splunk search can drill
    into ``event.xai_report.model_attributions[].model_name``.
    """
    return {
        "event": {
            "alert_id": alert.get("alert_id", alert.get("id")),
            "entity_id": alert.get("entity_id"),
            "severity": alert.get("severity"),
            "risk_score": alert.get("risk_score"),
            "title": alert.get("title"),
            "description": alert.get("description"),
            "xai_report": alert.get("xai_report"),
            "evidence": alert.get("evidence", []),
            "context_factors": alert.get("context_factors", []),
            "source_system": "itds",
            "alert_type": "insider_threat",
        },
        "source": settings.splunk_source,
        "sourcetype": settings.splunk_sourcetype,
        "index": settings.splunk_index,
        "time": _to_epoch(alert.get("timestamp")),
    }


def _to_epoch(ts: str | datetime.datetime | None) -> float:
    """Convert timestamp to Unix epoch for Splunk."""
    if ts is None:
        return datetime.datetime.now(datetime.timezone.utc).timestamp()
    if isinstance(ts, str):
        try:
            dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return datetime.datetime.now(datetime.timezone.utc).timestamp()
    return ts.timestamp()


async def push_to_splunk(alert: dict[str, Any]) -> bool:
    """
    Push a single alert to Splunk via HTTP Event Collector.

    Returns ``True`` on success, ``False`` on failure.
    Failures are logged but never raise â€” SIEM push is best-effort.
    """
    if not settings.splunk_hec_token:
        log.debug("siem.splunk.skipped", reason="no HEC token configured")
        return False

    payload = _format_splunk_event(alert)

    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            resp = await client.post(
                settings.splunk_hec_url,
                json=payload,
                headers={
                    "Authorization": f"Splunk {settings.splunk_hec_token}",
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code == 200:
                log.info(
                    "siem.splunk.pushed",
                    alert_id=alert.get("alert_id", alert.get("id")),
                )
                return True
            else:
                log.warning(
                    "siem.splunk.failed",
                    status=resp.status_code,
                    body=resp.text[:500],
                )
                return False
    except Exception as exc:
        log.error("siem.splunk.error", error=str(exc))
        return False


# â”€â”€ Microsoft Sentinel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_sentinel_signature(
    date: str,
    content_length: int,
    method: str = "POST",
    content_type: str = "application/json",
    resource: str = "/api/logs",
) -> str:
    """
    Build the HMAC-SHA256 Authorization header for the Sentinel
    Data Collector API.

    The signing algorithm is documented at:
    https://learn.microsoft.com/en-us/azure/azure-monitor/logs/data-collector-api
    """
    x_headers = f"x-ms-date:{date}"
    string_to_hash = (
        f"{method}\n{content_length}\n{content_type}\n{x_headers}\n{resource}"
    )
    decoded_key = base64.b64decode(settings.sentinel_shared_key)
    encoded_hash = base64.b64encode(
        hmac.new(
            decoded_key,
            string_to_hash.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    return f"SharedKey {settings.sentinel_workspace_id}:{encoded_hash}"


def _format_sentinel_payload(alert: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Format an alert into the Sentinel Data Collector API payload.

    Sentinel expects an array of JSON objects.  Each object becomes
    a row in the custom log table (``InsiderThreatAlert_CL``).
    Field names are automatically suffixed with ``_s`` (string),
    ``_d`` (double), etc.

    We flatten the XAI report into top-level fields for easier KQL
    querying::

        InsiderThreatAlert_CL
        | where severity_s == "critical"
        | extend top_model = parse_json(xai_model_attributions_s)[0].model_name
    """
    xai = alert.get("xai_report") or {}

    return [{
        "AlertId": alert.get("alert_id", alert.get("id")),
        "EntityId": alert.get("entity_id"),
        "Severity": alert.get("severity"),
        "RiskScore": alert.get("risk_score"),
        "Title": alert.get("title"),
        "Description": alert.get("description"),
        "SourceSystem": "ITDS",
        "AlertType": "InsiderThreat",
        # XAI metadata â€” JSON-encoded for KQL parse_json()
        "XAI_ExecutiveSummary": xai.get("executive_summary", ""),
        "XAI_ModelAttributions": json.dumps(xai.get("model_attributions", [])),
        "XAI_TopSHAPFeatures": json.dumps(xai.get("shap_features", [])[:10]),
        "XAI_BaselineComparison": json.dumps(xai.get("baseline_comparison", {})),
        "XAI_PeerGroupAnalysis": json.dumps(xai.get("peer_group_analysis", {})),
        # Evidence
        "Evidence": json.dumps(alert.get("evidence", [])),
        "ContextFactors": json.dumps(alert.get("context_factors", [])),
        # Timestamp
        "TimeGenerated": alert.get("timestamp", datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()),
    }]


async def push_to_sentinel(alert: dict[str, Any]) -> bool:
    """
    Push a single alert to Microsoft Sentinel via the Data Collector API.

    Returns ``True`` on success, ``False`` on failure.
    """
    if not settings.sentinel_workspace_id or not settings.sentinel_shared_key:
        log.debug("siem.sentinel.skipped", reason="no workspace credentials configured")
        return False

    payload = _format_sentinel_payload(alert)
    body = json.dumps(payload)
    content_length = len(body)

    rfc1123_date = datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%a, %d %b %Y %H:%M:%S GMT")

    signature = _build_sentinel_signature(rfc1123_date, content_length)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                settings.sentinel_data_collector_url,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": signature,
                    "Log-Type": settings.sentinel_log_type,
                    "x-ms-date": rfc1123_date,
                    "time-generated-field": "TimeGenerated",
                },
            )
            if resp.status_code in (200, 202):
                log.info(
                    "siem.sentinel.pushed",
                    alert_id=alert.get("alert_id", alert.get("id")),
                )
                return True
            else:
                log.warning(
                    "siem.sentinel.failed",
                    status=resp.status_code,
                    body=resp.text[:500],
                )
                return False
    except Exception as exc:
        log.error("siem.sentinel.error", error=str(exc))
        return False


# â”€â”€ Unified push â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def push_alert_to_siems(alert: dict[str, Any]) -> dict[str, bool]:
    """
    Push a high-severity alert to all configured SIEM platforms.

    Returns a dict of ``{"splunk": bool, "sentinel": bool}`` indicating
    which pushes succeeded.

    Called by the alert-service webhook handler or the gateway's alert
    processing pipeline when ``alert.severity in ("high", "critical")``.
    """
    results: dict[str, bool] = {}

    results["splunk"] = await push_to_splunk(alert)
    results["sentinel"] = await push_to_sentinel(alert)

    log.info(
        "siem.push_complete",
        alert_id=alert.get("alert_id", alert.get("id")),
        severity=alert.get("severity"),
        results=results,
    )

    return results

