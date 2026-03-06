"""
Explainable AI (XAI) Report Generator.

For every HIGH / CRITICAL alert, produces a human-readable report that
a security analyst can understand *without* ML expertise.

Report sections:
    1. Executive summary — what happened, when, how serious
    2. Model attribution — which detector(s) flagged, individual scores
    3. Feature importance — SHAP-ranked feature list
    4. Baseline comparison — "normally X, today Y"
    5. Peer group comparison — how this user compares to department peers
    6. Investigation playbook — step-by-step recommended actions
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.schemas import RiskScoreEvent

logger = logging.getLogger(__name__)


class XAIReportGenerator:
    """
    Generates structured + markdown XAI reports from risk-score events.

    Usage::

        gen = XAIReportGenerator()
        report = gen.generate(event, alert_id="abc-123")
        md = report["markdown"]
    """

    # ── Public API ──────────────────────────────────────────────────

    def generate(
        self,
        event: RiskScoreEvent,
        alert_id: str,
        baseline: dict[str, Any] | None = None,
        peer_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a full XAI report.

        Parameters
        ----------
        event : RiskScoreEvent
            The risk-score event that triggered the alert.
        alert_id : str
            UUID of the parent alert.
        baseline : dict, optional
            User's historical baseline stats (e.g. avg_files_per_day).
        peer_stats : dict, optional
            Aggregate stats for the user's department peer group.

        Returns
        -------
        dict with keys: alert_id, sections (list of dicts), markdown (str),
        generated_at (ISO timestamp).
        """
        baseline = baseline or self._mock_baseline(event.user_id)
        peer_stats = peer_stats or self._mock_peer_stats(event)

        sections = [
            self._executive_summary(event),
            self._model_attribution(event),
            self._feature_importance(event),
            self._baseline_comparison(event, baseline),
            self._peer_group_comparison(event, peer_stats),
            self._investigation_playbook(event),
        ]

        md = self._render_markdown(event, alert_id, sections)

        report = {
            "alert_id": alert_id,
            "user_id": event.user_id,
            "risk_level": event.risk_level,
            "sections": sections,
            "markdown": md,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "xai_report_generated",
            extra={"alert_id": alert_id, "user_id": event.user_id},
        )
        return report

    # ── Section builders ────────────────────────────────────────────

    def _executive_summary(self, event: RiskScoreEvent) -> dict[str, Any]:
        scored_at = (
            event.scored_at.strftime("%Y-%m-%d %H:%M UTC")
            if event.scored_at
            else "N/A"
        )
        detector_names = [d.detector for d in event.detector_breakdown if d.is_anomaly]
        flagged_by = ", ".join(detector_names) if detector_names else "ensemble"

        summary_text = (
            f"User **{event.user_id}** triggered a **{event.risk_level}** "
            f"risk alert at {scored_at} with an overall risk score of "
            f"**{event.overall_score:.2f}** (raw ML: {event.raw_ml_score:.2f}, "
            f"context-adjusted: {event.context_adjusted_score:.2f}). "
            f"Flagged by: {flagged_by}."
        )
        return {"title": "Executive Summary", "content": summary_text}

    def _model_attribution(self, event: RiskScoreEvent) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for d in event.detector_breakdown:
            rows.append({
                "detector": d.detector,
                "anomaly_score": round(d.anomaly_score, 4),
                "is_anomaly": d.is_anomaly,
                "key_detail": self._summarise_detail(d.detail),
            })
        return {
            "title": "Model Attribution",
            "content": (
                "Each ML detector independently scored this event. "
                "Scores closer to 1.0 indicate higher anomaly."
            ),
            "detectors": rows,
        }

    def _feature_importance(self, event: RiskScoreEvent) -> dict[str, Any]:
        features = event.top_anomalous_features or []
        ranked: list[dict[str, Any]] = []
        for i, f in enumerate(features[:10], 1):
            ranked.append({
                "rank": i,
                "feature": f.get("feature", "unknown"),
                "importance": round(f.get("importance", 0.0), 4),
                "source": f.get("source", "unknown"),
            })
        return {
            "title": "SHAP Feature Importances",
            "content": (
                "Features ranked by their contribution to the anomaly "
                "score (SHAP / reconstruction error)."
            ),
            "features": ranked,
        }

    def _baseline_comparison(
        self,
        event: RiskScoreEvent,
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        comparisons: list[str] = []
        for metric, stats in baseline.items():
            if isinstance(stats, dict):
                normal = stats.get("mean", "?")
                today = stats.get("today", "?")
                unit = stats.get("unit", "")
                comparisons.append(
                    f"- **{metric}**: normally {normal} {unit}; "
                    f"today **{today}** {unit}"
                )
        return {
            "title": "Historical Baseline Comparison",
            "content": (
                "How today's behaviour compares to this user's "
                "established baseline (30-day rolling average)."
            ),
            "comparisons": comparisons,
        }

    def _peer_group_comparison(
        self,
        event: RiskScoreEvent,
        peer_stats: dict[str, Any],
    ) -> dict[str, Any]:
        dept = event.business_context.get("department", "unknown")
        peer_mean = peer_stats.get("peer_mean_score", 0.0)
        peer_std = peer_stats.get("peer_std_score", 0.0)
        z_score = (
            (event.overall_score - peer_mean) / peer_std
            if peer_std > 0
            else 0.0
        )
        return {
            "title": "Peer Group Comparison",
            "content": (
                f"Compared to the **{dept}** department peer group "
                f"(mean score: {peer_mean:.2f}, std: {peer_std:.2f}), "
                f"this user's score is **{z_score:+.1f}σ** from the mean."
            ),
            "department": dept,
            "peer_mean": peer_mean,
            "peer_std": peer_std,
            "z_score": round(z_score, 2),
        }

    def _investigation_playbook(self, event: RiskScoreEvent) -> dict[str, Any]:
        base_steps = [
            "1. Review the flagged user's activity logs for the last 24 hours",
            "2. Cross-reference accessed resources with role-based access policies",
            "3. Check if the user has any approved change requests or tickets",
        ]

        level_steps = {
            "HIGH": [
                "4. Notify the user's manager for context verification",
                "5. Review DLP logs for any data exfiltration indicators",
                "6. Check VPN/network logs for unusual connection patterns",
                "7. Document findings and update alert status",
            ],
            "CRITICAL": [
                "4. **IMMEDIATE**: Notify SOC lead and CISO",
                "5. Consider temporary account suspension pending investigation",
                "6. Preserve all forensic evidence (disk images, network captures)",
                "7. Review DLP, email gateway, and USB device logs",
                "8. Interview user's manager and peers for context",
                "9. Engage legal/HR if data exfiltration is confirmed",
                "10. Document full incident timeline for post-mortem",
            ],
        }

        steps = base_steps + level_steps.get(event.risk_level, [
            "4. Continue standard monitoring",
            "5. Document findings and close if benign",
        ])

        return {
            "title": "Recommended Investigation Steps",
            "content": "Follow these steps to investigate this alert.",
            "steps": steps,
        }

    # ── Markdown renderer ───────────────────────────────────────────

    def _render_markdown(
        self,
        event: RiskScoreEvent,
        alert_id: str,
        sections: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = [
            f"# XAI Report — Alert {alert_id[:8]}",
            f"**User**: {event.user_id} | "
            f"**Risk Level**: {event.risk_level} | "
            f"**Score**: {event.overall_score:.2f}",
            "",
        ]

        for sec in sections:
            lines.append(f"## {sec['title']}")
            lines.append("")
            lines.append(sec.get("content", ""))
            lines.append("")

            # Detector table
            if "detectors" in sec:
                lines.append("| Detector | Score | Anomaly? | Key Detail |")
                lines.append("|----------|-------|----------|------------|")
                for d in sec["detectors"]:
                    flag = "✓" if d["is_anomaly"] else "✗"
                    lines.append(
                        f"| {d['detector']} | {d['anomaly_score']:.4f} "
                        f"| {flag} | {d['key_detail']} |"
                    )
                lines.append("")

            # Feature ranking
            if "features" in sec:
                lines.append("| Rank | Feature | Importance | Source |")
                lines.append("|------|---------|------------|--------|")
                for f in sec["features"]:
                    lines.append(
                        f"| {f['rank']} | {f['feature']} "
                        f"| {f['importance']:.4f} | {f['source']} |"
                    )
                lines.append("")

            # Baseline comparisons
            if "comparisons" in sec:
                for c in sec["comparisons"]:
                    lines.append(c)
                lines.append("")

            # Investigation steps
            if "steps" in sec:
                for s in sec["steps"]:
                    lines.append(s)
                lines.append("")

        lines.append("---")
        lines.append(
            f"*Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
            f"by ITDS XAI Engine*"
        )
        return "\n".join(lines)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _summarise_detail(detail: dict[str, Any]) -> str:
        """Extract a one-line summary from detector detail dict."""
        if not detail:
            return "—"
        if "error" in detail:
            return detail["error"]
        # Autoencoder
        if "z_score" in detail:
            return f"z-score: {detail['z_score']:.2f}"
        # LSTM
        if "sequence_error" in detail:
            return f"seq-error: {detail['sequence_error']:.4f}"
        # GNN
        if "embedding_norm" in detail:
            return f"emb-norm: {detail['embedding_norm']:.4f}"
        # Isolation Forest
        if "feature_importance" in detail:
            top = sorted(
                detail["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:2]
            return ", ".join(f"{k}={v:.3f}" for k, v in top)
        return str(list(detail.keys())[:3])

    @staticmethod
    def _mock_baseline(user_id: str) -> dict[str, Any]:
        """Generate mock baseline for development."""
        h = hash(user_id)
        return {
            "files_accessed_per_day": {
                "mean": 12 + (h % 20),
                "today": 47 + (h % 800),
                "unit": "files",
            },
            "login_locations": {
                "mean": 1,
                "today": 1 + (h % 3),
                "unit": "distinct IPs",
            },
            "data_download_mb": {
                "mean": round(5.2 + (h % 10) * 0.5, 1),
                "today": round(5.2 + (h % 10) * 0.5 + (h % 200), 1),
                "unit": "MB",
            },
            "after_hours_sessions": {
                "mean": 0.3,
                "today": 1 + (h % 4),
                "unit": "sessions",
            },
        }

    @staticmethod
    def _mock_peer_stats(event: RiskScoreEvent) -> dict[str, Any]:
        """Generate mock peer-group stats for development."""
        dept = event.business_context.get("department", "engineering")
        dept_baselines = {
            "engineering": {"peer_mean_score": 0.15, "peer_std_score": 0.08},
            "finance": {"peer_mean_score": 0.12, "peer_std_score": 0.06},
            "hr": {"peer_mean_score": 0.10, "peer_std_score": 0.05},
            "it_ops": {"peer_mean_score": 0.18, "peer_std_score": 0.10},
            "executive": {"peer_mean_score": 0.08, "peer_std_score": 0.04},
        }
        return dept_baselines.get(
            dept, {"peer_mean_score": 0.14, "peer_std_score": 0.07}
        )
