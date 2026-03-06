"""
Slack notifier — posts rich alert cards via Incoming Webhook.

Uses Block Kit for structured formatting with the XAI report
summary embedded as a collapsible section.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings
from app.models import Alert

logger = logging.getLogger(__name__)
settings = get_settings()


def _build_blocks(alert: Alert, xai_report: dict[str, Any]) -> list[dict]:
    """Construct Slack Block Kit blocks for the alert."""
    severity_emoji = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢",
    }
    emoji = severity_emoji.get(alert.risk_level, "⚪")

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Insider Threat Alert — {alert.risk_level}",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*User:*\n{alert.user_id}"},
                {"type": "mrkdwn", "text": f"*Risk Score:*\n{alert.risk_score:.2f}"},
                {"type": "mrkdwn", "text": f"*Anomaly Type:*\n{alert.anomaly_type}"},
                {"type": "mrkdwn", "text": f"*Status:*\n{alert.status}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Summary:* {alert.summary}",
            },
        },
    ]

    # XAI report summary
    sections = xai_report.get("sections", [])
    if sections:
        # Executive summary
        exec_summary = sections[0].get("content", "")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*XAI Summary:*\n{exec_summary}"},
        })

    # Top features (compact)
    feat_sec = next((s for s in sections if "features" in s), None)
    if feat_sec:
        features = feat_sec["features"][:5]
        feat_text = "\n".join(
            f"• `{f['feature']}` — importance: {f['importance']:.4f}"
            for f in features
        )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Top Anomalous Features:*\n{feat_text}"},
        })

    # Investigation link
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Investigate"},
                "url": f"https://itds.internal/alerts/{alert.id}",
                "style": "danger" if alert.risk_level == "CRITICAL" else "primary",
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Mark False Positive"},
                "url": f"https://itds.internal/alerts/{alert.id}/feedback",
            },
        ],
    })

    return blocks


async def send_slack(
    alert: Alert,
    xai_report: dict[str, Any],
) -> bool:
    """
    Post an alert to Slack via Incoming Webhook.

    Returns True on success, False on failure.
    """
    if not settings.SLACK_WEBHOOK_URL:
        logger.warning("slack_webhook_not_configured")
        return False

    payload = {
        "channel": settings.SLACK_CHANNEL,
        "username": "ITDS Alert Bot",
        "icon_emoji": ":shield:",
        "blocks": _build_blocks(alert, xai_report),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(settings.SLACK_WEBHOOK_URL, json=payload)
            if resp.status_code == 200:
                logger.info(
                    "slack_notification_sent",
                    extra={"alert_id": str(alert.id)},
                )
                return True
            logger.warning(
                "slack_notification_failed",
                extra={
                    "alert_id": str(alert.id),
                    "status": resp.status_code,
                    "body": resp.text[:200],
                },
            )
    except Exception:
        logger.exception("slack_notification_error", extra={"alert_id": str(alert.id)})

    return False
