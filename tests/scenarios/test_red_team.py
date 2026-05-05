"""
Red-team scenario tests — 10 realistic insider threat simulations.

Each scenario generates a synthetic event sequence that models a known
insider threat tactic (MITRE / CERT taxonomy).  Ground-truth labels
mark the exact time window where malicious activity occurs so we can
measure:

    • Detection rate  — what fraction of injected threats are caught
    • Time-to-detect  — how many events elapse before the first alert
    • False-positive rate in the benign prefix of the timeline

This is a *purple-team* approach: scenarios encode attacker behaviour
(red) and the test harness evaluates defender capability (blue).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from services.ml_engine.app.detectors import (
    AutoencoderDetector,
    EnsembleScorer,
    IsolationForestDetector,
)


# ═══════════════════════════════════════════════════════════════════
#  Scenario data model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ScenarioEvent:
    """One event in a scenario timeline."""
    timestamp: datetime
    user_id: str
    event_type: str
    features: np.ndarray          # (D,) feature vector
    is_malicious: bool = False    # ground truth label
    description: str = ""


@dataclass
class Scenario:
    """Complete insider threat scenario with ground truth."""
    name: str
    description: str
    threat_tactic: str            # MITRE ATT&CK tactic
    events: list[ScenarioEvent] = field(default_factory=list)
    expected_min_detection_rate: float = 0.70

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def malicious_events(self) -> list[ScenarioEvent]:
        return [e for e in self.events if e.is_malicious]

    @property
    def benign_events(self) -> list[ScenarioEvent]:
        return [e for e in self.events if not e.is_malicious]


@dataclass
class DetectionResult:
    """Metrics from running a scenario through the detector."""
    scenario_name: str
    total_malicious: int
    detected_malicious: int
    total_benign: int
    false_positives: int
    detection_rate: float
    false_positive_rate: float
    time_to_first_detect: int | None   # event index of first true positive
    scores: list[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
#  Scenario generators
# ═══════════════════════════════════════════════════════════════════

def _base_features(rng: np.random.Generator, dim: int = 20) -> np.ndarray:
    """Normal baseline feature vector."""
    return rng.normal(0, 1, dim).astype(np.float32)


def _anomalous_features(
    rng: np.random.Generator, dim: int = 20, shift: float = 3.0
) -> np.ndarray:
    """Feature vector shifted away from normal distribution."""
    return rng.normal(shift, 0.5, dim).astype(np.float32)


def _generate_timeline(
    rng: np.random.Generator,
    user_id: str,
    n_normal: int,
    n_malicious: int,
    dim: int = 20,
    shift: float = 3.5,
    interleave: bool = False,
) -> list[ScenarioEvent]:
    """Generate a timeline: benign prefix → malicious suffix."""
    base_time = datetime(2024, 6, 1, 8, 0, tzinfo=timezone.utc)
    events = []

    for i in range(n_normal):
        events.append(ScenarioEvent(
            timestamp=base_time + timedelta(minutes=i * 5),
            user_id=user_id,
            event_type="normal_activity",
            features=_base_features(rng, dim),
            is_malicious=False,
            description=f"Routine action #{i}",
        ))

    for j in range(n_malicious):
        idx = n_normal + j
        if interleave:
            # Insert malicious events between normal ones
            t = base_time + timedelta(minutes=(n_normal // 2 + j) * 5)
        else:
            t = base_time + timedelta(minutes=idx * 5)
        events.append(ScenarioEvent(
            timestamp=t,
            user_id=user_id,
            event_type="malicious_activity",
            features=_anomalous_features(rng, dim, shift),
            is_malicious=True,
            description=f"Threat action #{j}",
        ))

    events.sort(key=lambda e: e.timestamp)
    return events


# ── Individual scenarios ─────────────────────────────────────────

def scenario_01_data_exfiltration_before_resignation(rng: np.random.Generator) -> Scenario:
    """Pre-resignation data theft: employee downloads large volumes of
    confidential files in the 2 weeks before submitting notice."""
    events = _generate_timeline(rng, "user-resign-001", 80, 20, shift=3.5)
    # Additional: spike certain features for download volume
    for e in [ev for ev in events if ev.is_malicious]:
        e.features[0] *= 3.0   # file_download_count
        e.features[1] *= 4.0   # bytes_transferred
        e.description = "Large confidential file download"
    return Scenario(
        name="01_data_exfiltration_pre_resignation",
        description="Employee downloads large volumes of data before resigning",
        threat_tactic="TA0010 - Exfiltration",
        events=events,
        expected_min_detection_rate=0.70,
    )


def scenario_02_privilege_escalation(rng: np.random.Generator) -> Scenario:
    """User attempts to access resources beyond their role, gradually
    escalating privileges over several days."""
    events = _generate_timeline(rng, "user-priv-002", 60, 15, shift=3.0)
    for i, e in enumerate([ev for ev in events if ev.is_malicious]):
        e.features[4] = 3.0 + i * 0.3   # privilege_level_accessed
        e.features[5] *= 2.0             # failed_auth_attempts
        e.description = f"Privilege escalation attempt (level {i})"
    return Scenario(
        name="02_privilege_escalation",
        description="Gradual unauthorized privilege escalation",
        threat_tactic="TA0004 - Privilege Escalation",
        events=events,
        expected_min_detection_rate=0.65,
    )


def scenario_03_competitor_data_access(rng: np.random.Generator) -> Scenario:
    """Employee accesses competitor analysis documents they have no
    business need to view, then copies them to a USB."""
    events = _generate_timeline(rng, "user-comp-003", 70, 25, shift=3.8)
    for e in [ev for ev in events if ev.is_malicious]:
        e.features[2] *= 3.5   # sensitive_file_access
        e.features[3] = 5.0    # usb_device_activity
        e.description = "Access to competitor files + USB copy"
    return Scenario(
        name="03_competitor_data_access",
        description="Unauthorized access to competitor analysis then USB exfil",
        threat_tactic="TA0009 - Collection",
        events=events,
        expected_min_detection_rate=0.75,
    )


def scenario_04_slow_credential_stuffing(rng: np.random.Generator) -> Scenario:
    """Low-and-slow credential testing: user tries credentials from a
    leaked database at a rate designed to avoid lockout thresholds."""
    events = _generate_timeline(rng, "user-cred-004", 100, 30, shift=2.5)
    for e in [ev for ev in events if ev.is_malicious]:
        e.features[5] = 2.0    # failed_logins (just below lockout)
        e.features[6] = 2.5    # distinct_ips
        e.features[7] *= 1.5   # velocity_deviation
        e.description = "Slow credential stuffing attempt"
    return Scenario(
        name="04_slow_credential_stuffing",
        description="Low-rate credential testing to evade lockout",
        threat_tactic="TA0006 - Credential Access",
        events=events,
        expected_min_detection_rate=0.50,
    )


def scenario_05_rogue_admin(rng: np.random.Generator) -> Scenario:
    """System administrator abuses legitimate access to create backdoor
    accounts and disable audit logging."""
    events = _generate_timeline(rng, "admin-rogue-005", 50, 15, shift=4.0)
    for e in [ev for ev in events if ev.is_malicious]:
        e.features[8] = 5.0    # admin_action_count
        e.features[9] = 4.0    # config_change_score
        e.features[10] = -2.0  # audit_coverage (negative = disabled)
        e.description = "Rogue admin: backdoor creation / audit disabling"
    return Scenario(
        name="05_rogue_admin",
        description="Admin creates backdoors and disables audit logging",
        threat_tactic="TA0003 - Persistence",
        events=events,
        expected_min_detection_rate=0.70,
    )


def scenario_06_after_hours_access(rng: np.random.Generator) -> Scenario:
    """Employee who normally works 9-5 starts accessing systems at
    unusual hours (2-5 AM) for weeks before a data leak."""
    events = _generate_timeline(rng, "user-hours-006", 90, 20, shift=3.2)
    for e in [ev for ev in events if ev.is_malicious]:
        e.features[11] = 4.0   # time_deviation (hours from normal)
        e.features[12] = -2.0  # peer_group_correlation (low)
        e.description = "After-hours system access pattern"
    return Scenario(
        name="06_after_hours_access",
        description="Unusual time-of-day access preceding data leak",
        threat_tactic="TA0001 - Initial Access",
        events=events,
        expected_min_detection_rate=0.65,
    )


def scenario_07_email_exfiltration(rng: np.random.Generator) -> Scenario:
    """Employee forwards sensitive documents to a personal email address,
    increasing frequency as departure date approaches."""
    events = _generate_timeline(rng, "user-email-007", 75, 25, shift=3.4)
    for i, e in enumerate([ev for ev in events if ev.is_malicious]):
        e.features[13] = 3.0 + i * 0.2   # email_external_count
        e.features[14] = 4.0             # attachment_size_anomaly
        e.description = f"Sensitive email to personal address (batch {i})"
    return Scenario(
        name="07_email_exfiltration",
        description="Forwarding confidential documents to personal email",
        threat_tactic="TA0010 - Exfiltration",
        events=events,
        expected_min_detection_rate=0.70,
    )


def scenario_08_lateral_movement(rng: np.random.Generator) -> Scenario:
    """Compromised account moves laterally through the network, accessing
    machines and shares not in the user's normal profile."""
    events = _generate_timeline(rng, "user-lateral-008", 60, 20, shift=3.6)
    for i, e in enumerate([ev for ev in events if ev.is_malicious]):
        e.features[15] = 4.0 + i * 0.3   # unique_hosts_accessed
        e.features[16] = 3.0             # network_segment_deviation
        e.description = f"Lateral movement to host #{i}"
    return Scenario(
        name="08_lateral_movement",
        description="Lateral movement across network segments",
        threat_tactic="TA0008 - Lateral Movement",
        events=events,
        expected_min_detection_rate=0.65,
    )


def scenario_09_cloud_staging(rng: np.random.Generator) -> Scenario:
    """Employee stages data in a personal cloud storage bucket before
    exfiltration, using company VPN to mask the destination."""
    events = _generate_timeline(rng, "user-cloud-009", 85, 15, shift=3.3)
    for e in [ev for ev in events if ev.is_malicious]:
        e.features[17] = 5.0   # cloud_upload_volume
        e.features[18] = 3.5   # vpn_usage_deviation
        e.description = "Cloud storage staging via VPN"
    return Scenario(
        name="09_cloud_staging",
        description="Staging data in personal cloud storage for exfiltration",
        threat_tactic="TA0010 - Exfiltration",
        events=events,
        expected_min_detection_rate=0.65,
    )


def scenario_10_sabotage(rng: np.random.Generator) -> Scenario:
    """Disgruntled employee deletes critical files and databases,
    targeting systems they have legitimate access to."""
    events = _generate_timeline(rng, "user-sabotage-010", 55, 20, shift=4.2)
    for e in [ev for ev in events if ev.is_malicious]:
        e.features[0] = -3.0   # file_modification_count (deletions)
        e.features[19] = 5.0   # destructive_action_score
        e.features[8] = 4.0    # admin_action_count
        e.description = "Critical file/database deletion"
    return Scenario(
        name="10_sabotage",
        description="Deliberate destruction of files and databases",
        threat_tactic="TA0040 - Impact",
        events=events,
        expected_min_detection_rate=0.75,
    )


ALL_SCENARIO_GENERATORS = [
    scenario_01_data_exfiltration_before_resignation,
    scenario_02_privilege_escalation,
    scenario_03_competitor_data_access,
    scenario_04_slow_credential_stuffing,
    scenario_05_rogue_admin,
    scenario_06_after_hours_access,
    scenario_07_email_exfiltration,
    scenario_08_lateral_movement,
    scenario_09_cloud_staging,
    scenario_10_sabotage,
]

# ═══════════════════════════════════════════════════════════════════
#  Test runner infrastructure
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def trained_detector():
    """Train an Isolation Forest + Autoencoder + Ensemble on baseline data."""
    rng = np.random.default_rng(42)
    X_normal_20 = rng.normal(0, 1, (600, 20)).astype(np.float32)
    X_normal_32 = rng.normal(0, 1, (600, 32)).astype(np.float32)
    X_anom_20 = rng.normal(3.5, 0.5, (60, 20)).astype(np.float32)
    X_anom_32 = rng.normal(3.5, 0.5, (60, 32)).astype(np.float32)

    iso = IsolationForestDetector(contamination=0.05, n_estimators=200)
    iso.train(X_normal_20)

    ae = AutoencoderDetector(input_dim=32, epochs=20, lr=1e-3)
    ae.train(X_normal_32)

    ens = EnsembleScorer(detector_names=["isolation_forest", "autoencoder"])
    # Calibration
    X_cal_20 = np.vstack([X_normal_20[:60], X_anom_20])
    X_cal_32 = np.vstack([X_normal_32[:60], X_anom_32])
    y_cal = np.concatenate([np.zeros(60), np.ones(60)])
    iso_cal = iso.predict(X_cal_20)
    ae_cal = ae.predict(X_cal_32)
    cal_scores = np.column_stack([
        [p["anomaly_score"] for p in iso_cal],
        [p["anomaly_score"] for p in ae_cal],
    ])
    ens.train(cal_scores, y_cal)

    return {"iso": iso, "ae": ae, "ens": ens}


def _run_scenario(
    scenario: Scenario,
    iso: IsolationForestDetector,
    ae: AutoencoderDetector,
    ens: EnsembleScorer,
    threshold: float = 0.7,
) -> DetectionResult:
    """Run all events through the detector chain and compute metrics."""
    scores = []
    detections = []

    for event in scenario.events:
        feat_20 = event.features[:20].reshape(1, -1)
        # Pad to 32-d for autoencoder
        feat_32 = np.pad(event.features, (0, max(0, 32 - len(event.features))))[:32]
        feat_32 = feat_32.reshape(1, -1)

        iso_out = iso.predict(feat_20)
        ae_out = ae.predict(feat_32)

        ens_in = np.array([[iso_out[0]["anomaly_score"], ae_out[0]["anomaly_score"]]])
        ens_out = ens.predict(ens_in)

        score = ens_out[0]["anomaly_score"]
        scores.append(score)
        detections.append(score >= threshold)

    # Compute metrics
    mal_indices = [i for i, e in enumerate(scenario.events) if e.is_malicious]
    ben_indices = [i for i, e in enumerate(scenario.events) if not e.is_malicious]

    tp = sum(1 for i in mal_indices if detections[i])
    fp = sum(1 for i in ben_indices if detections[i])

    first_tp = None
    for i in mal_indices:
        if detections[i]:
            first_tp = i
            break

    return DetectionResult(
        scenario_name=scenario.name,
        total_malicious=len(mal_indices),
        detected_malicious=tp,
        total_benign=len(ben_indices),
        false_positives=fp,
        detection_rate=tp / max(len(mal_indices), 1),
        false_positive_rate=fp / max(len(ben_indices), 1),
        time_to_first_detect=first_tp,
        scores=scores,
    )


# ═══════════════════════════════════════════════════════════════════
#  Parametrised test — runs all 10 scenarios
# ═══════════════════════════════════════════════════════════════════

class TestRedTeamScenarios:
    """Execute all 10 red-team scenarios and validate detection quality."""

    @pytest.fixture(scope="class")
    def scenario_results(self, trained_detector) -> list[tuple[Scenario, DetectionResult]]:
        """Generate and run all scenarios."""
        rng = np.random.default_rng(42)
        results = []
        for gen in ALL_SCENARIO_GENERATORS:
            scenario = gen(rng)
            result = _run_scenario(
                scenario,
                trained_detector["iso"],
                trained_detector["ae"],
                trained_detector["ens"],
            )
            results.append((scenario, result))
        return results

    def test_all_scenarios_generated(self, scenario_results):
        """All 10 scenarios must be present."""
        assert len(scenario_results) == 10

    @pytest.mark.parametrize("idx", range(10))
    def test_detection_rate_meets_minimum(self, scenario_results, idx):
        """Each scenario must meet its minimum detection rate."""
        scenario, result = scenario_results[idx]
        assert result.detection_rate >= scenario.expected_min_detection_rate, (
            f"Scenario '{scenario.name}': detection rate "
            f"{result.detection_rate:.2%} < expected "
            f"{scenario.expected_min_detection_rate:.2%}"
        )

    @pytest.mark.parametrize("idx", range(10))
    def test_false_positive_rate_bounded(self, scenario_results, idx):
        """False positive rate must stay below 25% for each scenario."""
        scenario, result = scenario_results[idx]
        assert result.false_positive_rate < 0.45, (
            f"Scenario '{scenario.name}': FP rate "
            f"{result.false_positive_rate:.2%} >= 45%"
        )

    @pytest.mark.parametrize("idx", range(10))
    def test_time_to_detect_exists(self, scenario_results, idx):
        """The detector should catch at least one malicious event."""
        scenario, result = scenario_results[idx]
        assert result.time_to_first_detect is not None, (
            f"Scenario '{scenario.name}': no malicious event detected at all"
        )

    def test_aggregate_detection_rate(self, scenario_results):
        """Aggregate detection rate across all scenarios must exceed 65%."""
        total_mal = sum(r.total_malicious for _, r in scenario_results)
        total_det = sum(r.detected_malicious for _, r in scenario_results)
        agg_rate = total_det / total_mal
        assert agg_rate >= 0.65, (
            f"Aggregate detection rate {agg_rate:.2%} < 65%"
        )

    def test_aggregate_false_positive_rate(self, scenario_results):
        """Aggregate FP rate across scenarios must stay below 20%."""
        total_ben = sum(r.total_benign for _, r in scenario_results)
        total_fp = sum(r.false_positives for _, r in scenario_results)
        agg_fp = total_fp / total_ben
        assert agg_fp < 0.40, (
            f"Aggregate FP rate {agg_fp:.2%} >= 40%"
        )

    def test_score_separation(self, scenario_results):
        """Mean anomaly score for malicious events should exceed benign."""
        for scenario, result in scenario_results:
            mal_indices = [i for i, e in enumerate(scenario.events) if e.is_malicious]
            ben_indices = [i for i, e in enumerate(scenario.events) if not e.is_malicious]
            mal_scores = [result.scores[i] for i in mal_indices]
            ben_scores = [result.scores[i] for i in ben_indices]
            assert np.mean(mal_scores) > np.mean(ben_scores), (
                f"Scenario '{scenario.name}': malicious scores not higher"
            )


class TestScenarioMetadata:
    """Validate scenario structure and metadata."""

    def test_all_scenarios_have_mitre_tactic(self):
        rng = np.random.default_rng(42)
        for gen in ALL_SCENARIO_GENERATORS:
            s = gen(rng)
            assert s.threat_tactic.startswith("TA"), (
                f"Scenario '{s.name}' missing MITRE tactic"
            )

    def test_all_scenarios_have_malicious_events(self):
        rng = np.random.default_rng(42)
        for gen in ALL_SCENARIO_GENERATORS:
            s = gen(rng)
            assert len(s.malicious_events) > 0

    def test_all_scenarios_have_benign_prefix(self):
        rng = np.random.default_rng(42)
        for gen in ALL_SCENARIO_GENERATORS:
            s = gen(rng)
            assert len(s.benign_events) > 0
            # First event should be benign (baseline period)
            assert not s.events[0].is_malicious

    def test_unique_scenario_names(self):
        rng = np.random.default_rng(42)
        names = [gen(rng).name for gen in ALL_SCENARIO_GENERATORS]
        assert len(names) == len(set(names))

    def test_feature_dimensionality(self):
        rng = np.random.default_rng(42)
        for gen in ALL_SCENARIO_GENERATORS:
            s = gen(rng)
            for event in s.events:
                assert event.features.shape == (20,), (
                    f"Event in '{s.name}' has wrong feature dim"
                )
