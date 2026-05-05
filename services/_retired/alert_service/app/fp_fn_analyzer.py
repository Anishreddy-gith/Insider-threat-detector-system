"""
False Positive / False Negative Analyzer — weekly batch job.

Computes precision, recall, F1, and AUC-ROC broken down by:
    - Overall
    - Per detector
    - Per department / user group

Generates a comprehensive Markdown report and saves it to disk.

Requires resolved feedback data (TRUE_POSITIVE / FALSE_POSITIVE)
to function. NEEDS_REVIEW verdicts are excluded from metrics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session
from app.models import Alert, Feedback
from app.schemas import AnalysisMetrics, AnalysisReport

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Core metrics computation ────────────────────────────────────────

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float] | None = None,
) -> AnalysisMetrics:
    """
    Compute classification metrics.

    y_true : 1 = actual positive (confirmed threat)
    y_pred : 1 = predicted positive (alert was raised)
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    auc_roc = None
    if y_scores and len(set(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auc_roc = float(roc_auc_score(y_true, y_scores))
        except (ImportError, ValueError):
            pass

    return AnalysisMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        auc_roc=round(auc_roc, 4) if auc_roc is not None else None,
        total_alerts=len(y_true),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


# ── Data collection ─────────────────────────────────────────────────

async def _collect_data(
    session: AsyncSession,
    days: int,
) -> list[dict[str, Any]]:
    """
    Join Feedback + Alert data for the analysis window.

    Only includes confirmed verdicts (TRUE_POSITIVE / FALSE_POSITIVE).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    result = await session.execute(
        select(Feedback, Alert)
        .join(Alert, Alert.id == Feedback.alert_id)
        .where(
            Feedback.created_at >= cutoff,
            Feedback.verdict.in_(["TRUE_POSITIVE", "FALSE_POSITIVE"]),
        )
        .order_by(Feedback.created_at.asc())
    )

    rows: list[dict[str, Any]] = []
    for fb, alert in result.all():
        rows.append({
            "feedback_id": str(fb.id),
            "alert_id": str(alert.id),
            "user_id": fb.user_id,
            "department": fb.department or "unknown",
            "verdict": fb.verdict,
            "risk_score": fb.risk_score_snapshot,
            "detector_scores": fb.detector_scores_snapshot or {},
            "risk_level": alert.risk_level,
            "anomaly_type": alert.anomaly_type,
        })

    return rows


# ── Analysis engine ─────────────────────────────────────────────────

async def run_analysis(days: int | None = None) -> AnalysisReport:
    """
    Run the full FP/FN analysis and generate a Markdown report.

    Parameters
    ----------
    days : int, optional
        Analysis window in days. Defaults to settings.ANALYSIS_WINDOW_DAYS.
    """
    window = days or settings.ANALYSIS_WINDOW_DAYS

    async with async_session() as session:
        data = await _collect_data(session, window)

    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=window)

    if not data:
        empty_metrics = AnalysisMetrics(
            precision=0.0, recall=0.0, f1=0.0, auc_roc=None,
            total_alerts=0, true_positives=0, false_positives=0, false_negatives=0,
        )
        return AnalysisReport(
            period_start=period_start,
            period_end=period_end,
            overall=empty_metrics,
            per_detector={},
            per_department={},
            report_markdown="# FP/FN Analysis Report\n\nNo feedback data available.",
            generated_at=period_end,
        )

    # ── Overall metrics ─────────────────────────────────────────
    y_true = [1 if r["verdict"] == "TRUE_POSITIVE" else 0 for r in data]
    y_pred = [1] * len(data)  # All were predicted positive (alerts were raised)
    y_scores = [r["risk_score"] for r in data]
    overall = compute_metrics(y_true, y_pred, y_scores)

    # ── Per-detector metrics ────────────────────────────────────
    detector_names = ["isolation_forest", "autoencoder", "lstm_temporal", "gnn_relational"]
    per_detector: dict[str, AnalysisMetrics] = {}

    for det_name in detector_names:
        det_y_true: list[int] = []
        det_y_pred: list[int] = []
        det_scores: list[float] = []

        for r in data:
            det_data = r["detector_scores"].get(det_name, {})
            if isinstance(det_data, dict):
                score = det_data.get("anomaly_score", 0.0)
                is_anomaly = det_data.get("is_anomaly", False)
            elif isinstance(det_data, (int, float)):
                score = float(det_data)
                is_anomaly = score > 0.5
            else:
                continue

            det_y_true.append(1 if r["verdict"] == "TRUE_POSITIVE" else 0)
            det_y_pred.append(1 if is_anomaly else 0)
            det_scores.append(score)

        if det_y_true:
            per_detector[det_name] = compute_metrics(det_y_true, det_y_pred, det_scores)

    # ── Per-department metrics ──────────────────────────────────
    dept_groups: dict[str, list[dict]] = defaultdict(list)
    for r in data:
        dept_groups[r["department"]].append(r)

    per_department: dict[str, AnalysisMetrics] = {}
    for dept, items in dept_groups.items():
        d_true = [1 if r["verdict"] == "TRUE_POSITIVE" else 0 for r in items]
        d_pred = [1] * len(items)
        d_scores = [r["risk_score"] for r in items]
        per_department[dept] = compute_metrics(d_true, d_pred, d_scores)

    # ── Markdown report ─────────────────────────────────────────
    md = _render_markdown(
        period_start, period_end, overall, per_detector, per_department, data
    )

    # Save to disk
    report_dir = Path(settings.ANALYSIS_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"fpfn_report_{period_end.strftime('%Y%m%d')}.md"
    report_path.write_text(md, encoding="utf-8")

    logger.info(
        "fpfn_analysis_complete",
        extra={
            "period": f"{period_start.date()} → {period_end.date()}",
            "total_feedback": len(data),
            "precision": overall.precision,
            "recall": overall.recall,
            "f1": overall.f1,
            "report_path": str(report_path),
        },
    )

    return AnalysisReport(
        period_start=period_start,
        period_end=period_end,
        overall=overall,
        per_detector=per_detector,
        per_department=per_department,
        report_markdown=md,
        generated_at=period_end,
    )


# ── Markdown renderer ───────────────────────────────────────────────

def _render_markdown(
    start: datetime,
    end: datetime,
    overall: AnalysisMetrics,
    per_detector: dict[str, AnalysisMetrics],
    per_department: dict[str, AnalysisMetrics],
    data: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# ITDS — FP/FN Analysis Report",
        "",
        f"**Period:** {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}  ",
        f"**Total Feedback Entries:** {len(data)}",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Precision | {overall.precision:.4f} |",
        f"| Recall | {overall.recall:.4f} |",
        f"| F1 Score | {overall.f1:.4f} |",
        f"| AUC-ROC | {overall.auc_roc if overall.auc_roc is not None else 'N/A'} |",
        f"| True Positives | {overall.true_positives} |",
        f"| False Positives | {overall.false_positives} |",
        f"| False Negatives | {overall.false_negatives} |",
        "",
        "---",
        "",
        "## Per-Detector Breakdown",
        "",
        "| Detector | Precision | Recall | F1 | AUC-ROC | TP | FP | FN |",
        "|----------|-----------|--------|-----|---------|----|----|-----|",
    ]

    for name, m in per_detector.items():
        auc = f"{m.auc_roc:.4f}" if m.auc_roc is not None else "N/A"
        lines.append(
            f"| {name} | {m.precision:.4f} | {m.recall:.4f} | "
            f"{m.f1:.4f} | {auc} | {m.true_positives} | "
            f"{m.false_positives} | {m.false_negatives} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Per-Department Breakdown",
        "",
        "| Department | Precision | Recall | F1 | Total Alerts | TP | FP |",
        "|------------|-----------|--------|-----|-------------|----|----|",
    ])

    for dept, m in sorted(per_department.items()):
        lines.append(
            f"| {dept} | {m.precision:.4f} | {m.recall:.4f} | "
            f"{m.f1:.4f} | {m.total_alerts} | "
            f"{m.true_positives} | {m.false_positives} |"
        )

    # Key findings
    lines.extend([
        "",
        "---",
        "",
        "## Key Findings & Recommendations",
        "",
    ])

    if overall.precision < 0.7:
        lines.append(
            "- ⚠️ **Low precision** — too many false positives. Consider "
            "raising alert thresholds or improving context-aware scoring rules."
        )
    if overall.recall < 0.8:
        lines.append(
            "- ⚠️ **Low recall** — potential missed threats. Review detector "
            "sensitivity and lower anomaly thresholds cautiously."
        )

    # Worst detector
    if per_detector:
        worst = min(per_detector.items(), key=lambda x: x[1].f1)
        lines.append(
            f"- 📊 **Weakest detector**: `{worst[0]}` (F1={worst[1].f1:.4f}) — "
            f"consider re-training or adjusting its ensemble weight."
        )
        best = max(per_detector.items(), key=lambda x: x[1].f1)
        lines.append(
            f"- 📊 **Strongest detector**: `{best[0]}` (F1={best[1].f1:.4f})"
        )

    # Worst department
    if per_department:
        worst_dept = min(per_department.items(), key=lambda x: x[1].precision)
        if worst_dept[1].precision < 0.6:
            lines.append(
                f"- 🏢 **{worst_dept[0]}** has the highest false-positive rate "
                f"(precision={worst_dept[1].precision:.4f}). Review context rules "
                f"for this department."
            )

    lines.extend([
        "",
        "---",
        f"*Generated at {end.strftime('%Y-%m-%d %H:%M UTC')} by ITDS FP/FN Analyzer*",
    ])

    return "\n".join(lines)
