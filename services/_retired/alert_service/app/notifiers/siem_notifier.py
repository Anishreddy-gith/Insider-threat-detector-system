"""
SIEM webhook notifier — pushes alerts in Splunk HEC or Microsoft
Sentinel format to an external SIEM webhook endpoint.

The payload includes the full XAI report summary so that analysts
can triage directly from their SIEM dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import get_settings
from app.models import Alert

logger = logging.getLogger(__name__)
settings = get_settings()


def _sentinel_payload(alert: Alert, xai_report: dict[str, Any]) -> dict[str, Any]:
    """
    Format alert as a Microsoft Sentinel custom log (Data Collection Rules).

    Schema follows the CommonSecurityLog format with custom fields.
    """
    sections = xai_report.get("sections", [])
    exec_summary = sections[0].get("content", "") if sections else ""

    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "data": {
            "TimeGenerated": datetime.now(timezone.utc).isoformat(),
            "SourceSystem": "ITDS",
            "Category": "InsiderThreat",
            "Activity": alert.anomaly_type,
            "SourceUserId": alert.user_id,
            "LogSeverity": _severity_to_cef(alert.risk_level),
            "DeviceCustomNumber1": alert.risk_score,
            "DeviceCustomNumber1Label": "RiskScore",
            "DeviceCustomString1": alert.risk_level,
            "DeviceCustomString1Label": "RiskLevel",
            "DeviceCustomString2": alert.status,
            "DeviceCustomString2Label": "AlertStatus",
            "DeviceCustomString3": str(alert.id),
            "DeviceCustomString3Label": "AlertId",
            "Message": exec_summary,
            "AdditionalExtensions": {
                "detector_breakdown": alert.detector_breakdown,
                "top_features": alert.top_features,
                "recommended_actions": alert.recommended_actions,
                "xai_summary": exec_summary,
            },
        },
    }


def _splunk_payload(alert: Alert, xai_report: dict[str, Any]) -> dict[str, Any]:
    """
    Format alert as a Splunk HTTP Event Collector (HEC) event.
    """
    sections = xai_report.get("sections", [])
    exec_summary = sections[0].get("content", "") if sections else ""

    return {
        "time": int(datetime.now(timezone.utc).timestamp()),
        "host": "itds-alert-service",
        "source": "itds:alert",
        "sourcetype": "itds:insider_threat",
        "index": "insider_threats",
        "event": {
            "alert_id": str(alert.id),
            "user_id": alert.user_id,
            "risk_score": alert.risk_score,
            "risk_level": alert.risk_level,
            "anomaly_type": alert.anomaly_type,
            "status": alert.status,
            "summary": alert.summary,
            "xai_summary": exec_summary,
            "detector_breakdown": alert.detector_breakdown,
            "top_features": alert.top_features,
            "recommended_actions": alert.recommended_actions,
            "business_context": alert.business_context,
        },
    }


def _severity_to_cef(risk_level: str) -> int:
    """Convert risk level to CEF severity (0-10 scale)."""
    return {"LOW": 3, "MEDIUM": 5, "HIGH": 7, "CRITICAL": 10}.get(risk_level, 5)


async def send_siem(
    alert: Alert,
    xai_report: dict[str, Any],
) -> bool:
    """
    Post alert to the configured SIEM webhook.

    Returns True on success, False on failure.
    """
    if not settings.SIEM_WEBHOOK_URL:
        logger.warning("siem_webhook_not_configured")
        return False

    formatter = (
        _sentinel_payload
        if settings.SIEM_FORMAT == "sentinel"
        else _splunk_payload
    )
    payload = formatter(alert, xai_report)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.SIEM_AUTH_TOKEN:
        if settings.SIEM_FORMAT == "splunk":
            headers["Authorization"] = f"Splunk {settings.SIEM_AUTH_TOKEN}"
        else:
            headers["Authorization"] = f"Bearer {settings.SIEM_AUTH_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                settings.SIEM_WEBHOOK_URL,
                json=payload,
                headers=headers,
            )
            if resp.status_code in (200, 201, 202):
                logger.info(
                    "siem_notification_sent",
                    extra={
                        "alert_id": str(alert.id),
                        "format": settings.SIEM_FORMAT,
                    },
                )
                return True
            logger.warning(
                "siem_notification_failed",
                extra={
                    "alert_id": str(alert.id),
                    "status": resp.status_code,
                    "body": resp.text[:200],
                },
            )
    except Exception:
        logger.exception("siem_notification_error", extra={"alert_id": str(alert.id)})

    return False
