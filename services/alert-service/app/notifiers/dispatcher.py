"""
Multi-channel notification dispatcher.

Routes alerts + XAI reports to configured channels:
    - Slack (webhook)
    - Email (SMTP via aiosmtplib)
    - SIEM webhook (Splunk / Microsoft Sentinel format)
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.models import Alert

logger = logging.getLogger(__name__)
settings = get_settings()


async def dispatch_notifications(
    alert: Alert,
    xai_report: dict[str, Any],
) -> dict[str, bool]:
    """
    Fan-out notifications to all enabled channels.

    Updates the alert's notification flags in-place (caller commits).
    Returns a dict of {channel: success_bool}.
    """
    results: dict[str, bool] = {}

    if settings.SLACK_ENABLED:
        from app.notifiers.slack_notifier import send_slack
        ok = await send_slack(alert, xai_report)
        alert.notified_slack = ok
        results["slack"] = ok

    if settings.EMAIL_ENABLED:
        from app.notifiers.email_notifier import send_email
        ok = await send_email(alert, xai_report)
        alert.notified_email = ok
        results["email"] = ok

    if settings.SIEM_ENABLED:
        from app.notifiers.siem_notifier import send_siem
        ok = await send_siem(alert, xai_report)
        alert.notified_siem = ok
        results["siem"] = ok

    if not results:
        logger.info(
            "no_notification_channels_enabled",
            extra={"alert_id": str(alert.id)},
        )

    return results
