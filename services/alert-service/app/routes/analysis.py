"""
FP/FN analysis routes.

Endpoints:
  POST  /analysis/run   – trigger a batch analysis
  GET   /analysis/latest – get the latest report
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Query

from app.config import get_settings
from app.fp_fn_analyzer import run_analysis
from app.schemas import AnalysisReport

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


@router.post("/run", response_model=AnalysisReport)
async def trigger_analysis(
    days: int = Query(default=7, ge=1, le=90),
):
    """
    Run the FP/FN analysis for the specified window.

    Returns precision, recall, F1, and AUC-ROC per detector and
    per department, plus a full Markdown report.
    """
    return await run_analysis(days=days)


@router.get("/latest")
async def get_latest_report():
    """Return the most recently generated Markdown report."""
    report_dir = Path(settings.ANALYSIS_REPORT_DIR)
    if not report_dir.exists():
        return {"report": None, "message": "No reports generated yet"}

    reports = sorted(report_dir.glob("fpfn_report_*.md"), reverse=True)
    if not reports:
        return {"report": None, "message": "No reports generated yet"}

    latest = reports[0]
    return {
        "filename": latest.name,
        "report": latest.read_text(encoding="utf-8"),
    }
