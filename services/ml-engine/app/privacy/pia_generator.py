"""
Privacy Impact Assessment (PIA) Generator.

Auto-generates a structured PIA report as a Python dict / JSON
covering the six mandatory sections from NIST SP 800-122 and
GDPR Article 35:

    1. Data Collected
    2. Retention Periods
    3. ε Budget & Differential Privacy
    4. Data Minimisation Measures
    5. User / Data Subject Rights
    6. Risk Mitigations

The report is deterministic for a given configuration — it reads
system settings and produces a point-in-time snapshot.  It is
designed to be served via an API endpoint or exported to PDF.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app.privacy.differential_privacy import explain_epsilon


def generate_pia(
    *,
    organisation_name: str = "ITDS Deployment",
    system_name: str = "Insider Threat Detection System",
    dpo_contact: str = "dpo@example.com",
    extra_data_categories: list[dict[str, str]] | None = None,
    extra_mitigations: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate a complete Privacy Impact Assessment report.

    Parameters
    ----------
    organisation_name : str
        Legal entity operating the system.
    system_name : str
        Name of the system under assessment.
    dpo_contact : str
        Data Protection Officer email.
    extra_data_categories : list[dict]
        Additional data categories beyond the defaults.
    extra_mitigations : list[str]
        Additional risk mitigations to include.

    Returns
    -------
    dict — The full PIA report, serialisable to JSON.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc).isoformat()

    return {
        "meta": {
            "report_type": "Privacy Impact Assessment",
            "system_name": system_name,
            "organisation": organisation_name,
            "dpo_contact": dpo_contact,
            "generated_at": now,
            "framework_references": [
                "NIST SP 800-122 (Guide to Protecting PII)",
                "GDPR Article 35 (Data Protection Impact Assessment)",
                "ISO/IEC 27701:2019 (Privacy Information Management)",
            ],
        },
        "section_1_data_collected": _section_data_collected(
            extra_data_categories
        ),
        "section_2_retention_periods": _section_retention(),
        "section_3_privacy_budget": _section_privacy_budget(settings),
        "section_4_data_minimisation": _section_data_minimisation(),
        "section_5_user_rights": _section_user_rights(),
        "section_6_risk_mitigations": _section_risk_mitigations(
            settings, extra_mitigations
        ),
    }


# ── Section builders ────────────────────────────────────────────────


def _section_data_collected(
    extra: list[dict[str, str]] | None,
) -> dict[str, Any]:
    categories = [
        {
            "category": "Authentication Events",
            "description": (
                "Login timestamps, source IP addresses, "
                "authentication method, success/failure status."
            ),
            "pii_level": "medium",
            "legal_basis": "Legitimate interest (security monitoring)",
        },
        {
            "category": "File Access Logs",
            "description": (
                "File paths accessed, read/write/delete operations, "
                "file sensitivity labels, timestamps."
            ),
            "pii_level": "high",
            "legal_basis": "Legitimate interest (data loss prevention)",
        },
        {
            "category": "Network Activity",
            "description": (
                "Destination IPs/domains, bytes transferred, "
                "protocol types, connection durations."
            ),
            "pii_level": "medium",
            "legal_basis": "Legitimate interest (network security)",
        },
        {
            "category": "Application Usage",
            "description": (
                "Application names, session durations, "
                "switching frequency. No keystroke or content logging."
            ),
            "pii_level": "low",
            "legal_basis": "Legitimate interest (behavioural baseline)",
        },
        {
            "category": "Behavioural Feature Vectors",
            "description": (
                "Derived statistical aggregates (means, counts, ratios) "
                "computed from the above categories. Raw events are not "
                "stored beyond the retention window."
            ),
            "pii_level": "low",
            "legal_basis": "Legitimate interest (anomaly detection)",
        },
    ]
    if extra:
        categories.extend(extra)

    return {
        "total_categories": len(categories),
        "categories": categories,
        "note": (
            "The system does NOT collect: email content, chat messages, "
            "keystrokes, screenshots, webcam feeds, or any biometric data."
        ),
    }


def _section_retention() -> dict[str, Any]:
    return {
        "raw_events": {
            "period": "30 days",
            "justification": (
                "Required for baseline window computation. "
                "Events older than 30 days are permanently deleted."
            ),
            "deletion_method": "Automated Kafka topic retention + DB cascade delete",
        },
        "feature_vectors": {
            "period": "90 days",
            "justification": (
                "Supports seasonal pattern detection and model retraining. "
                "Stored in aggregated, pseudonymised form."
            ),
            "deletion_method": "Scheduled batch deletion job",
        },
        "model_weights": {
            "period": "Until model replacement",
            "justification": (
                "Trained models contain no raw PII — only statistical "
                "patterns. Replaced upon retraining."
            ),
            "deletion_method": "Overwritten on each training cycle",
        },
        "anomaly_alerts": {
            "period": "1 year",
            "justification": (
                "Retained for audit trail and incident response. "
                "Contains entity IDs but no raw behavioural data."
            ),
            "deletion_method": "Annual archival → encrypted cold storage → 7-year purge",
        },
        "privacy_budget_state": {
            "period": "24 hours (Redis TTL)",
            "justification": "Per-day accounting; auto-expires via Redis TTL.",
            "deletion_method": "Redis key expiration (TTL = 86400s)",
        },
    }


def _section_privacy_budget(settings: Any) -> dict[str, Any]:
    epsilon = settings.DP_EPSILON
    delta = settings.DP_DELTA

    return {
        "mechanism": "Laplace mechanism (input perturbation)",
        "epsilon": epsilon,
        "delta": delta,
        "daily_budget_per_entity": 10.0,
        "alert_threshold_pct": 80,
        "composition_theorem": "Sequential composition (conservative bound)",
        "guarantee_summary": (
            f"Each query consumes ε from the entity's daily budget. "
            f"The total guarantee degrades linearly: k queries of ε each "
            f"yield (k·ε)-DP. Budget resets every 24 hours."
        ),
        "plain_english_explanation": explain_epsilon(epsilon, delta),
        "privacy_amplification": (
            "Federated learning provides additional privacy amplification: "
            "raw data never leaves the client boundary. The server only "
            "sees aggregated model weight updates. Combined with DP noise "
            "on local training, this yields a stronger effective guarantee."
        ),
    }


def _section_data_minimisation() -> dict[str, Any]:
    return {
        "measures": [
            {
                "measure": "Feature aggregation",
                "description": (
                    "Raw events are immediately reduced to fixed-dimension "
                    "statistical feature vectors. The ML models never see "
                    "raw log lines, filenames, or IP addresses."
                ),
            },
            {
                "measure": "Pseudonymisation",
                "description": (
                    "Entity identifiers are replaced with opaque UUIDs. "
                    "The mapping table is stored separately with "
                    "restricted access."
                ),
            },
            {
                "measure": "Federated learning",
                "description": (
                    "Department-level data never leaves the department's "
                    "compute boundary. Only model weight updates are "
                    "sent to the central server."
                ),
            },
            {
                "measure": "Differential privacy",
                "description": (
                    "Calibrated noise is added during training and query "
                    "answering, ensuring no single employee's data can "
                    "be reconstructed from model outputs."
                ),
            },
            {
                "measure": "Purpose limitation",
                "description": (
                    "Collected data is used exclusively for insider threat "
                    "detection. No secondary uses (performance evaluation, "
                    "HR decisions) are permitted without separate DPIA."
                ),
            },
            {
                "measure": "Minimum necessary features",
                "description": (
                    "Only the 20-45 behavioural features required for "
                    "anomaly detection are retained. No content-level "
                    "data (email bodies, file contents) is processed."
                ),
            },
        ],
    }


def _section_user_rights() -> dict[str, Any]:
    return {
        "rights": [
            {
                "right": "Right of Access (GDPR Art. 15)",
                "implementation": (
                    "Employees may request a copy of their behavioural "
                    "feature vectors and any anomaly scores via the "
                    "API Gateway /api/v1/subject-access-request endpoint."
                ),
            },
            {
                "right": "Right to Erasure (GDPR Art. 17)",
                "implementation": (
                    "Upon request, all raw events, feature vectors, and "
                    "Redis state for the entity are purged. Model weights "
                    "are retrained without the deleted data."
                ),
            },
            {
                "right": "Right to Rectification (GDPR Art. 16)",
                "implementation": (
                    "If events were ingested incorrectly (e.g. wrong "
                    "entity attribution), the ingestion service supports "
                    "event amendment and baseline recalculation."
                ),
            },
            {
                "right": "Right to Object (GDPR Art. 21)",
                "implementation": (
                    "Employees may object to automated profiling. The "
                    "system supports entity-level opt-out, at which point "
                    "monitoring ceases and existing data is deleted."
                ),
            },
            {
                "right": "Right to Explanation (GDPR Recital 71)",
                "implementation": (
                    "All anomaly alerts include SHAP-based feature "
                    "importance explanations and per-detector score "
                    "breakdowns so subjects can understand and contest "
                    "findings."
                ),
            },
        ],
        "contact": (
            "Requests should be directed to the Data Protection Officer "
            "via the organisation's privacy portal."
        ),
    }


def _section_risk_mitigations(
    settings: Any,
    extra: list[str] | None,
) -> dict[str, Any]:
    mitigations = [
        {
            "risk": "Model inversion attack",
            "description": (
                "An adversary queries the model repeatedly to reconstruct "
                "training data."
            ),
            "mitigation": (
                f"Differential privacy (ε={settings.DP_EPSILON}) bounds "
                f"information leakage. The Privacy Budget Tracker limits "
                f"daily queries, preventing sustained attacks."
            ),
            "residual_risk": "low",
        },
        {
            "risk": "Membership inference attack",
            "description": (
                "An adversary determines whether a specific employee's "
                "data was in the training set."
            ),
            "mitigation": (
                f"(ε={settings.DP_EPSILON}, δ={settings.DP_DELTA})-DP "
                f"makes membership inference probabilistically bounded "
                f"to at most e^ε ≈ {math.exp(settings.DP_EPSILON):.2f}× "
                f"advantage."
            ),
            "residual_risk": "low",
        },
        {
            "risk": "Data exfiltration from ML pipeline",
            "description": (
                "Compromise of the ML engine could expose training data."
            ),
            "mitigation": (
                "Federated learning ensures raw data stays on department "
                "boundaries. The central server only stores aggregated "
                "model weights, not data."
            ),
            "residual_risk": "medium",
        },
        {
            "risk": "False positive discrimination",
            "description": (
                "Anomaly scores could disproportionately affect certain "
                "roles or departments."
            ),
            "mitigation": (
                "Per-department federated training accounts for "
                "role-specific baselines. The ensemble scorer normalises "
                "across detectors. All alerts require human SOC analyst "
                "review before action."
            ),
            "residual_risk": "medium",
        },
        {
            "risk": "Privacy budget exhaustion denial of service",
            "description": (
                "An attacker intentionally exhausts an entity's ε budget "
                "to prevent legitimate monitoring."
            ),
            "mitigation": (
                "Budget consumption is authenticated and rate-limited. "
                "Admin-level budget resets are available. The system "
                "continues to monitor using non-DP models as fallback."
            ),
            "residual_risk": "low",
        },
    ]

    if extra:
        for m in extra:
            mitigations.append({
                "risk": "Custom mitigation",
                "description": m,
                "mitigation": "See organisation-specific policy.",
                "residual_risk": "tbd",
            })

    return {
        "total_risks_assessed": len(mitigations),
        "mitigations": mitigations,
        "overall_risk_rating": "LOW-MEDIUM",
        "review_schedule": "Quarterly, or upon significant system changes",
    }


# ── Convenience class ──────────────────────────────────────────────

class PIAGenerator:
    """
    Stateful PIA generator that caches the last report.

    Usage::

        gen = PIAGenerator(organisation_name="Acme Corp")
        report = gen.generate()
        json_str = gen.to_json(indent=2)
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._last_report: dict[str, Any] | None = None

    def generate(self, **overrides: Any) -> dict[str, Any]:
        merged = {**self._kwargs, **overrides}
        self._last_report = generate_pia(**merged)
        return self._last_report

    def to_json(self, indent: int = 2) -> str:
        if self._last_report is None:
            self.generate()
        return json.dumps(self._last_report, indent=indent, default=str)

    @property
    def last_report(self) -> dict[str, Any] | None:
        return self._last_report
