/**
 * Mock data for standalone demo mode.
 * Provides realistic insider-threat data for every API endpoint
 * so the frontend renders fully without a backend.
 */

import type {
  DashboardMetrics,
  AlertDetail,
  Alert,
  UserProfile,
  ModelHealthOverview,
  RiskScore,
} from "./types";

// ── helpers ──────────────────────────────────────────────────

const now = new Date();
function daysAgo(n: number) {
  const d = new Date(now);
  d.setDate(d.getDate() - n);
  return d.toISOString();
}
function hoursAgo(n: number) {
  const d = new Date(now);
  d.setHours(d.getHours() - n);
  return d.toISOString();
}

// ── Users / Entities ─────────────────────────────────────────

export const MOCK_USERS = [
  { id: "u-001", username: "jsmith", department: "Engineering", title: "Senior Engineer", manager: "M. Chen", risk_score: 82.4, alert_count: 5, last_activity: hoursAgo(1), hire_date: "2021-03-15" },
  { id: "u-002", username: "agarcia", department: "Finance", title: "Financial Analyst", manager: "R. Patel", risk_score: 67.3, alert_count: 3, last_activity: hoursAgo(2), hire_date: "2020-08-20" },
  { id: "u-003", username: "kwilson", department: "HR", title: "HR Manager", manager: "S. Lee", risk_score: 45.1, alert_count: 2, last_activity: hoursAgo(4), hire_date: "2019-06-10" },
  { id: "u-004", username: "mzhang", department: "Engineering", title: "DevOps Lead", manager: "M. Chen", risk_score: 91.7, alert_count: 8, last_activity: hoursAgo(0.5), hire_date: "2022-01-05" },
  { id: "u-005", username: "rjohnson", department: "Sales", title: "Account Executive", manager: "D. Park", risk_score: 33.2, alert_count: 1, last_activity: hoursAgo(6), hire_date: "2023-02-28" },
  { id: "u-006", username: "lbrown", department: "Legal", title: "Compliance Officer", manager: "J. Adams", risk_score: 22.8, alert_count: 0, last_activity: hoursAgo(8), hire_date: "2018-11-12" },
  { id: "u-007", username: "tkim", department: "Engineering", title: "ML Engineer", manager: "M. Chen", risk_score: 55.9, alert_count: 2, last_activity: hoursAgo(3), hire_date: "2022-07-19" },
  { id: "u-008", username: "dpatel", department: "Finance", title: "Controller", manager: "R. Patel", risk_score: 71.6, alert_count: 4, last_activity: hoursAgo(1.5), hire_date: "2020-04-01" },
  { id: "u-009", username: "slee", department: "IT", title: "Sys Admin", manager: "A. Rivera", risk_score: 88.3, alert_count: 7, last_activity: hoursAgo(0.3), hire_date: "2021-09-15" },
  { id: "u-010", username: "nwang", department: "Research", title: "Data Scientist", manager: "T. Nguyen", risk_score: 39.5, alert_count: 1, last_activity: hoursAgo(5), hire_date: "2023-05-20" },
  { id: "u-011", username: "cmartin", department: "Sales", title: "VP Sales", manager: "CEO", risk_score: 28.1, alert_count: 0, last_activity: hoursAgo(12), hire_date: "2017-03-01" },
  { id: "u-012", username: "jdoe", department: "IT", title: "Network Engineer", manager: "A. Rivera", risk_score: 62.4, alert_count: 3, last_activity: hoursAgo(2), hire_date: "2022-11-08" },
];

// ── Alerts ───────────────────────────────────────────────────

export const MOCK_ALERTS: AlertDetail[] = [
  {
    id: "a-001",
    entity_id: "u-004",
    entity_name: "mzhang",
    dedup_key: "exfil-cloud-u004",
    title: "Mass data exfiltration via cloud storage",
    description: "User mzhang uploaded 2.3 GB of source code to personal Google Drive over a 4-hour window outside business hours. Pattern matches pre-resignation data staging.",
    severity: "critical",
    risk_score: 91.7,
    status: "INVESTIGATING",
    created_at: hoursAgo(2),
    updated_at: hoursAgo(1),
    context_factors: [
      { factor: "After-hours activity", multiplier: 1.8, description: "Activity at 2:30 AM local time" },
      { factor: "Resignation notice", multiplier: 2.1, description: "2-week notice filed 3 days ago" },
      { factor: "Volume spike", multiplier: 1.5, description: "450% above normal data transfer" },
    ],
    xai_report: {
      executive_summary: "High-confidence insider threat detection. User mzhang shows strong indicators of pre-departure data exfiltration, uploading proprietary source code to unauthorized cloud storage during off-hours. The combination of resignation notice, after-hours activity, and anomalous data volume produces a strong signal across all four detectors.",
      model_attribution: [
        { detector: "Isolation Forest", contribution_pct: 35.2, raw_score: 0.94, anomaly_type: "Statistical outlier in file access volume" },
        { detector: "Autoencoder", contribution_pct: 28.7, raw_score: 0.89, anomaly_type: "Reconstruction error on cloud upload pattern" },
        { detector: "LSTM Temporal", contribution_pct: 22.1, raw_score: 0.82, anomaly_type: "Temporal anomaly: off-hours bulk transfer" },
        { detector: "GNN Relational", contribution_pct: 14.0, raw_score: 0.71, anomaly_type: "Unusual resource access pattern" },
      ],
      shap_features: [
        { feature: "cloud_upload_volume", shap_value: 0.234, feature_value: 2300, direction: "increases_risk" as const },
        { feature: "after_hours_ratio", shap_value: 0.189, feature_value: 0.92, direction: "increases_risk" as const },
        { feature: "file_access_count", shap_value: 0.156, feature_value: 487, direction: "increases_risk" as const },
        { feature: "resignation_flag", shap_value: 0.134, feature_value: 1, direction: "increases_risk" as const },
        { feature: "unique_repos_accessed", shap_value: 0.098, feature_value: 12, direction: "increases_risk" as const },
        { feature: "email_external_ratio", shap_value: -0.045, feature_value: 0.15, direction: "decreases_risk" as const },
        { feature: "login_frequency", shap_value: -0.023, feature_value: 3, direction: "decreases_risk" as const },
        { feature: "badge_anomaly", shap_value: 0.067, feature_value: 0.8, direction: "increases_risk" as const },
      ],
      baseline_comparison: { current_score: 91.7, baseline_score: 34.2, deviation_pct: 168.1 },
      peer_group_analysis: { peer_group: "Engineering - DevOps", peer_count: 18, user_zscore: 3.42, peer_mean: 31.5, peer_std: 12.4 },
      investigation_playbook: [
        "Review DLP logs for cloud storage uploads in the past 72 hours",
        "Check if uploaded files contain proprietary code or trade secrets",
        "Interview direct manager M. Chen about project handoff status",
        "Review badge access logs for after-hours physical access",
        "Verify if resignation exit process includes IP review",
        "Consider temporary revocation of cloud storage access pending review",
      ],
    },
    evidence: [
      { type: "file_access", timestamp: hoursAgo(4), description: "Google Drive upload initiated: 487 files, 2.3 GB", risk_contribution: 0.35, raw_data: {} },
      { type: "login", timestamp: hoursAgo(4.5), description: "VPN login from home IP at 2:30 AM", risk_contribution: 0.15, raw_data: {} },
      { type: "file_access", timestamp: hoursAgo(3.5), description: "Accessed 12 restricted Git repositories", risk_contribution: 0.25, raw_data: {} },
      { type: "network", timestamp: daysAgo(3), description: "Resignation notice filed — last day in 11 days", risk_contribution: 0.20, raw_data: {} },
      { type: "badge", timestamp: hoursAgo(5), description: "No physical badge swipe — remote only session", risk_contribution: 0.05, raw_data: {} },
    ],
    audit_trail: [
      { action: "CREATED", actor: "system", timestamp: hoursAgo(2), details: "Alert auto-generated by ensemble scorer" },
      { action: "STATUS_CHANGE", actor: "analyst_1", timestamp: hoursAgo(1), details: "Status changed to INVESTIGATING" },
    ],
    recommended_actions: [
      "Suspend cloud storage sync access immediately",
      "Engage legal for IP theft assessment",
      "Preserve forensic image of workstation",
      "Review all file access in past 30 days",
    ],
  },
  {
    id: "a-002",
    entity_id: "u-009",
    entity_name: "slee",
    dedup_key: "privesc-db-u009",
    title: "Privilege escalation: unauthorized admin access",
    description: "System administrator slee accessed 6 production databases outside their assigned scope, creating new admin accounts with elevated privileges.",
    severity: "critical",
    risk_score: 88.3,
    status: "NEW",
    created_at: hoursAgo(0.5),
    updated_at: hoursAgo(0.5),
    context_factors: [
      { factor: "Scope violation", multiplier: 2.3, description: "Accessed resources outside assigned scope" },
      { factor: "Privilege creation", multiplier: 1.9, description: "Created admin accounts without approval" },
    ],
    xai_report: {
      executive_summary: "Sys admin slee accessed production databases outside their authorized scope and created unauthorized admin accounts. The GNN relational detector flagged unusual cross-system access patterns not seen among peer sys admins.",
      model_attribution: [
        { detector: "GNN Relational", contribution_pct: 42.5, raw_score: 0.91, anomaly_type: "Unusual cross-system resource access" },
        { detector: "Isolation Forest", contribution_pct: 25.3, raw_score: 0.85, anomaly_type: "Statistical outlier in privilege operations" },
        { detector: "LSTM Temporal", contribution_pct: 18.8, raw_score: 0.78, anomaly_type: "Rapid successive database access" },
        { detector: "Autoencoder", contribution_pct: 13.4, raw_score: 0.72, anomaly_type: "Reconstruction error on admin operations" },
      ],
      shap_features: [
        { feature: "cross_system_access", shap_value: 0.312, feature_value: 6, direction: "increases_risk" as const },
        { feature: "admin_account_creation", shap_value: 0.267, feature_value: 3, direction: "increases_risk" as const },
        { feature: "scope_violation_count", shap_value: 0.198, feature_value: 6, direction: "increases_risk" as const },
        { feature: "velocity_db_queries", shap_value: 0.134, feature_value: 245, direction: "increases_risk" as const },
        { feature: "login_frequency", shap_value: -0.015, feature_value: 5, direction: "decreases_risk" as const },
      ],
      baseline_comparison: { current_score: 88.3, baseline_score: 28.7, deviation_pct: 207.7 },
      peer_group_analysis: { peer_group: "IT - System Admins", peer_count: 8, user_zscore: 4.12, peer_mean: 25.3, peer_std: 9.8 },
      investigation_playbook: [
        "Disable newly created admin accounts immediately",
        "Audit all database queries executed in the session",
        "Review change management tickets for authorization",
        "Check if slee had legitimate business justification",
      ],
    },
    evidence: [
      { type: "network", timestamp: hoursAgo(1), description: "Created admin account 'svc_backup_admin' in prod-db-03", risk_contribution: 0.30, raw_data: {} },
      { type: "file_access", timestamp: hoursAgo(1.5), description: "245 queries against finance-prod database (unauthorized)", risk_contribution: 0.35, raw_data: {} },
      { type: "login", timestamp: hoursAgo(2), description: "Accessed 6 production DB clusters in 90 minutes", risk_contribution: 0.25, raw_data: {} },
    ],
    audit_trail: [
      { action: "CREATED", actor: "system", timestamp: hoursAgo(0.5), details: "Alert auto-generated by ensemble scorer" },
    ],
    recommended_actions: [
      "Lock created admin accounts",
      "Review all DB queries for data exfiltration",
      "Rotate affected database credentials",
    ],
  },
  {
    id: "a-003",
    entity_id: "u-001",
    entity_name: "jsmith",
    dedup_key: "afterhours-u001",
    title: "Anomalous after-hours access pattern",
    description: "jsmith has shown a sustained pattern of late-night system access over the past week, accessing sensitive engineering documents.",
    severity: "high",
    risk_score: 72.4,
    status: "INVESTIGATING",
    created_at: hoursAgo(12),
    updated_at: hoursAgo(6),
    context_factors: [
      { factor: "After-hours pattern", multiplier: 1.6, description: "7 consecutive days of late-night access" },
    ],
    xai_report: {
      executive_summary: "Sustained after-hours access pattern detected for jsmith over 7 consecutive days. While individual sessions are not alarming, the pattern deviates significantly from both personal baseline and peer group norms.",
      model_attribution: [
        { detector: "LSTM Temporal", contribution_pct: 45.0, raw_score: 0.78, anomaly_type: "Temporal pattern shift" },
        { detector: "Isolation Forest", contribution_pct: 30.0, raw_score: 0.72, anomaly_type: "Outlier in access timing" },
        { detector: "Autoencoder", contribution_pct: 15.0, raw_score: 0.65, anomaly_type: "Deviation from normal behavior" },
        { detector: "GNN Relational", contribution_pct: 10.0, raw_score: 0.42, anomaly_type: "Unchanged peer relationships" },
      ],
      shap_features: [
        { feature: "after_hours_ratio", shap_value: 0.312, feature_value: 0.78, direction: "increases_risk" as const },
        { feature: "consecutive_late_nights", shap_value: 0.201, feature_value: 7, direction: "increases_risk" as const },
        { feature: "document_sensitivity_avg", shap_value: 0.156, feature_value: 0.85, direction: "increases_risk" as const },
      ],
      baseline_comparison: { current_score: 72.4, baseline_score: 25.1, deviation_pct: 188.4 },
      peer_group_analysis: { peer_group: "Engineering - Backend", peer_count: 24, user_zscore: 2.87, peer_mean: 29.6, peer_std: 11.2 },
      investigation_playbook: [
        "Check with manager if there is a legitimate project deadline",
        "Review documents accessed during off-hours sessions",
        "Compare with historical after-hours patterns",
      ],
    },
    evidence: [
      { type: "login", timestamp: hoursAgo(14), description: "VPN login at 11:45 PM (7th consecutive late-night session)", risk_contribution: 0.40, raw_data: {} },
      { type: "file_access", timestamp: hoursAgo(13), description: "Accessed 34 sensitive engineering docs", risk_contribution: 0.35, raw_data: {} },
    ],
    audit_trail: [
      { action: "CREATED", actor: "system", timestamp: hoursAgo(12), details: "Alert auto-generated" },
      { action: "STATUS_CHANGE", actor: "analyst_2", timestamp: hoursAgo(6), details: "Started investigation" },
    ],
    recommended_actions: ["Verify with manager", "Review access logs"],
  },
  {
    id: "a-004", entity_id: "u-002", entity_name: "agarcia",
    dedup_key: "email-exfil-u002",
    title: "Unusual external email volume with attachments",
    description: "agarcia sent 47 emails with financial spreadsheet attachments to external addresses in a single day.",
    severity: "high", risk_score: 67.3, status: "NEW", created_at: hoursAgo(5), updated_at: hoursAgo(5),
    context_factors: [{ factor: "Email volume spike", multiplier: 1.7, description: "380% above daily average" }],
    xai_report: {
      executive_summary: "Anomalous email exfiltration pattern detected for finance analyst agarcia.",
      model_attribution: [
        { detector: "Isolation Forest", contribution_pct: 40.0, raw_score: 0.81, anomaly_type: "Volume outlier" },
        { detector: "Autoencoder", contribution_pct: 30.0, raw_score: 0.74, anomaly_type: "Pattern deviation" },
        { detector: "LSTM Temporal", contribution_pct: 20.0, raw_score: 0.68, anomaly_type: "Burst activity" },
        { detector: "GNN Relational", contribution_pct: 10.0, raw_score: 0.45, anomaly_type: "New external contacts" },
      ],
      shap_features: [
        { feature: "email_external_ratio", shap_value: 0.287, feature_value: 0.89, direction: "increases_risk" as const },
        { feature: "attachment_count", shap_value: 0.234, feature_value: 47, direction: "increases_risk" as const },
      ],
      baseline_comparison: { current_score: 67.3, baseline_score: 18.5, deviation_pct: 263.8 },
      peer_group_analysis: { peer_group: "Finance - Analysts", peer_count: 12, user_zscore: 3.15, peer_mean: 15.2, peer_std: 8.5 },
      investigation_playbook: ["Review email recipients", "Check attachment contents for PII/financial data"],
    },
    evidence: [
      { type: "email", timestamp: hoursAgo(6), description: "47 emails with .xlsx attachments to external domains", risk_contribution: 0.50, raw_data: {} },
    ],
    audit_trail: [{ action: "CREATED", actor: "system", timestamp: hoursAgo(5), details: "Auto-generated" }],
    recommended_actions: ["Review DLP email logs", "Check if recipients are approved vendors"],
  },
  {
    id: "a-005", entity_id: "u-008", entity_name: "dpatel",
    dedup_key: "competitor-research-u008",
    title: "Competitor website research pattern",
    description: "dpatel conducted extensive research on competitor financial filings and hiring pages.",
    severity: "medium", risk_score: 51.6, status: "RESOLVED", created_at: daysAgo(2), updated_at: daysAgo(1),
    context_factors: [],
    xai_report: {
      executive_summary: "Moderate risk: competitive intelligence gathering detected but within acceptable parameters after investigation.",
      model_attribution: [
        { detector: "Isolation Forest", contribution_pct: 50.0, raw_score: 0.62, anomaly_type: "Unusual web browsing" },
        { detector: "Autoencoder", contribution_pct: 25.0, raw_score: 0.48, anomaly_type: "Deviation" },
        { detector: "LSTM Temporal", contribution_pct: 15.0, raw_score: 0.41, anomaly_type: "Sustained pattern" },
        { detector: "GNN Relational", contribution_pct: 10.0, raw_score: 0.35, anomaly_type: "Normal" },
      ],
      shap_features: [{ feature: "competitor_site_visits", shap_value: 0.198, feature_value: 23, direction: "increases_risk" as const }],
      baseline_comparison: { current_score: 51.6, baseline_score: 22.0, deviation_pct: 134.5 },
      peer_group_analysis: { peer_group: "Finance", peer_count: 15, user_zscore: 1.8, peer_mean: 20.0, peer_std: 7.2 },
      investigation_playbook: ["Confirmed legitimate market research"],
    },
    evidence: [],
    audit_trail: [
      { action: "CREATED", actor: "system", timestamp: daysAgo(2), details: "Auto-generated" },
      { action: "STATUS_CHANGE", actor: "analyst_1", timestamp: daysAgo(1), details: "Resolved — legitimate research" },
    ],
    recommended_actions: [],
  },
  {
    id: "a-006", entity_id: "u-007", entity_name: "tkim",
    dedup_key: "usb-exfil-u007",
    title: "USB device data transfer detected",
    description: "tkim connected an unauthorized USB storage device and transferred 800 MB of ML model weights.",
    severity: "high", risk_score: 65.9, status: "ESCALATED", created_at: daysAgo(1), updated_at: hoursAgo(8),
    context_factors: [{ factor: "Unauthorized USB", multiplier: 2.0, description: "USB storage not in approved list" }],
    xai_report: {
      executive_summary: "USB exfiltration of ML model artifacts detected. Device not in approved hardware list.",
      model_attribution: [
        { detector: "Isolation Forest", contribution_pct: 35.0, raw_score: 0.76, anomaly_type: "USB usage outlier" },
        { detector: "Autoencoder", contribution_pct: 30.0, raw_score: 0.71, anomaly_type: "Data flow anomaly" },
        { detector: "LSTM Temporal", contribution_pct: 20.0, raw_score: 0.63, anomaly_type: "Unusual timing" },
        { detector: "GNN Relational", contribution_pct: 15.0, raw_score: 0.55, anomaly_type: "IP access" },
      ],
      shap_features: [
        { feature: "usb_usage", shap_value: 0.345, feature_value: 1, direction: "increases_risk" as const },
        { feature: "data_transfer_volume", shap_value: 0.267, feature_value: 800, direction: "increases_risk" as const },
      ],
      baseline_comparison: { current_score: 65.9, baseline_score: 30.0, deviation_pct: 119.7 },
      peer_group_analysis: { peer_group: "Engineering - ML", peer_count: 10, user_zscore: 2.5, peer_mean: 28.0, peer_std: 10.5 },
      investigation_playbook: ["Confiscate USB device", "Verify model IP classification"],
    },
    evidence: [
      { type: "usb", timestamp: daysAgo(1), description: "Unauthorized USB device SanDisk-32GB connected, 800 MB transferred", risk_contribution: 0.60, raw_data: {} },
    ],
    audit_trail: [
      { action: "CREATED", actor: "system", timestamp: daysAgo(1), details: "Auto-generated" },
      { action: "STATUS_CHANGE", actor: "analyst_1", timestamp: hoursAgo(10), details: "Investigating" },
      { action: "STATUS_CHANGE", actor: "analyst_1", timestamp: hoursAgo(8), details: "Escalated to security ops" },
    ],
    recommended_actions: ["Retrieve USB device", "Assess IP exposure risk"],
  },
];

// ── Dashboard Metrics ────────────────────────────────────────

export const MOCK_DASHBOARD: DashboardMetrics = {
  total_alerts: 23,
  critical_entities: 3,
  average_risk_score: 52.7,
  active_entities: MOCK_USERS.length,
  risk_distribution: [
    { level: "low", count: 3 },
    { level: "medium", count: 4 },
    { level: "high", count: 3 },
    { level: "critical", count: 2 },
  ],
  alert_trend: [
    { date: daysAgo(6).slice(0, 10), count: 2 },
    { date: daysAgo(5).slice(0, 10), count: 5 },
    { date: daysAgo(4).slice(0, 10), count: 3 },
    { date: daysAgo(3).slice(0, 10), count: 7 },
    { date: daysAgo(2).slice(0, 10), count: 4 },
    { date: daysAgo(1).slice(0, 10), count: 6 },
    { date: daysAgo(0).slice(0, 10), count: 3 },
  ],
  top_entities: MOCK_USERS.sort((a, b) => b.risk_score - a.risk_score).slice(0, 8).map((u) => ({
    entity_id: u.id,
    username: u.username,
    department: u.department,
    risk_score: u.risk_score,
    risk_level: (u.risk_score >= 75 ? "critical" : u.risk_score >= 50 ? "high" : u.risk_score >= 25 ? "medium" : "low") as import("./types").RiskLevel,
    timestamp: hoursAgo(0),
    context_multiplier: 1.0,
    detector_scores: [
      { detector: "isolation_forest", raw_score: u.risk_score / 100, weight: 0.30, weighted_score: u.risk_score * 0.003 },
      { detector: "autoencoder", raw_score: u.risk_score / 100, weight: 0.25, weighted_score: u.risk_score * 0.0025 },
      { detector: "lstm_temporal", raw_score: u.risk_score / 100, weight: 0.25, weighted_score: u.risk_score * 0.0025 },
      { detector: "gnn_relational", raw_score: u.risk_score / 100, weight: 0.20, weighted_score: u.risk_score * 0.002 },
    ],
  })),
  department_heatmap: [
    { department: "Engineering", entity_count: 45, avg_risk: 62.3, critical_count: 2 },
    { department: "Finance", entity_count: 28, avg_risk: 48.7, critical_count: 1 },
    { department: "IT", entity_count: 22, avg_risk: 71.5, critical_count: 2 },
    { department: "HR", entity_count: 18, avg_risk: 32.1, critical_count: 0 },
    { department: "Sales", entity_count: 35, avg_risk: 25.8, critical_count: 0 },
    { department: "Legal", entity_count: 12, avg_risk: 19.4, critical_count: 0 },
    { department: "Research", entity_count: 20, avg_risk: 38.9, critical_count: 1 },
  ],
};

// ── Model Health ─────────────────────────────────────────────

export const MOCK_MODEL_HEALTH: ModelHealthOverview = {
  detectors: [
    {
      detector: "isolation_forest",
      precision: 0.89,
      recall: 0.94,
      f1_score: 0.91,
      auc_roc: 0.962,
      true_positives: 847,
      false_positives: 104,
      false_negatives: 54,
      last_trained: daysAgo(1),
      training_samples: 125000,
    },
    {
      detector: "autoencoder",
      precision: 0.85,
      recall: 0.91,
      f1_score: 0.88,
      auc_roc: 0.945,
      true_positives: 819,
      false_positives: 144,
      false_negatives: 81,
      last_trained: daysAgo(1),
      training_samples: 125000,
    },
    {
      detector: "lstm_temporal",
      precision: 0.82,
      recall: 0.88,
      f1_score: 0.85,
      auc_roc: 0.931,
      true_positives: 792,
      false_positives: 173,
      false_negatives: 108,
      last_trained: daysAgo(2),
      training_samples: 98000,
    },
    {
      detector: "gnn_relational",
      precision: 0.87,
      recall: 0.83,
      f1_score: 0.85,
      auc_roc: 0.928,
      true_positives: 747,
      false_positives: 112,
      false_negatives: 153,
      last_trained: daysAgo(2),
      training_samples: 98000,
    },
  ],
  ensemble_weights: {
    isolation_forest: 0.30,
    autoencoder: 0.25,
    lstm_temporal: 0.25,
    gnn_relational: 0.20,
  },
  privacy_budget: {
    total_epsilon: 10.0,
    consumed_epsilon: 3.47,
    remaining_epsilon: 6.53,
    total_delta: 1e-5,
    consumed_delta: 1.2e-6,
    queries_today: 142,
    budget_refresh_at: new Date(
      now.getFullYear(), now.getMonth(), now.getDate() + 1
    ).toISOString(),
  },
  federated_learning: {
    status: "training",
    current_round: 47,
    total_rounds: 100,
    participating_clients: 5,
    global_accuracy: 0.912,
    convergence_delta: 2.3e-4,
    last_aggregation: hoursAgo(1),
  },
  fp_fn_analysis: {
    overall_precision: 0.87,
    overall_recall: 0.91,
    overall_f1: 0.89,
    overall_auc_roc: 0.952,
    by_detector: [
      { detector: "isolation_forest", precision: 0.89, recall: 0.94, f1_score: 0.91, auc_roc: 0.962, true_positives: 847, false_positives: 104, false_negatives: 54, last_trained: daysAgo(1), training_samples: 125000 },
      { detector: "autoencoder", precision: 0.85, recall: 0.91, f1_score: 0.88, auc_roc: 0.945, true_positives: 819, false_positives: 144, false_negatives: 81, last_trained: daysAgo(1), training_samples: 125000 },
      { detector: "lstm_temporal", precision: 0.82, recall: 0.88, f1_score: 0.85, auc_roc: 0.931, true_positives: 792, false_positives: 173, false_negatives: 108, last_trained: daysAgo(2), training_samples: 98000 },
      { detector: "gnn_relational", precision: 0.87, recall: 0.83, f1_score: 0.85, auc_roc: 0.928, true_positives: 747, false_positives: 112, false_negatives: 153, last_trained: daysAgo(2), training_samples: 98000 },
    ],
    by_department: [
      { department: "Engineering", precision: 0.91, recall: 0.93, f1_score: 0.92, alert_count: 34, fp_count: 3 },
      { department: "Finance", precision: 0.85, recall: 0.89, f1_score: 0.87, alert_count: 22, fp_count: 4 },
      { department: "IT", precision: 0.88, recall: 0.95, f1_score: 0.91, alert_count: 28, fp_count: 3 },
      { department: "HR", precision: 0.82, recall: 0.86, f1_score: 0.84, alert_count: 12, fp_count: 2 },
      { department: "Sales", precision: 0.79, recall: 0.84, f1_score: 0.81, alert_count: 18, fp_count: 6 },
      { department: "Research", precision: 0.90, recall: 0.88, f1_score: 0.89, alert_count: 15, fp_count: 2 },
    ],
  },
};

// ── User Profile builder ─────────────────────────────────────

export function buildUserProfile(userId: string): UserProfile | null {
  const user = MOCK_USERS.find((u) => u.id === userId);
  if (!user) return null;

  const riskHistory = Array.from({ length: 30 }, (_, i) => ({
    timestamp: daysAgo(29 - i),
    score: Math.max(5, Math.min(95, user.risk_score + (Math.random() - 0.5) * 30 - (29 - i) * 0.3)),
  }));

  return {
    id: user.id,
    username: user.username,
    email: `${user.username}@company.com`,
    department: user.department,
    title: user.title,
    manager: user.manager,
    hire_date: user.hire_date,
    risk_score: user.risk_score,
    risk_level: (user.risk_score >= 75 ? "critical" : user.risk_score >= 50 ? "high" : user.risk_score >= 25 ? "medium" : "low") as import("./types").RiskLevel,
    risk_history: riskHistory,
    behavior_features: {
      login_frequency: 0.3 + Math.random() * 0.4,
      after_hours_ratio: user.risk_score > 60 ? 0.5 + Math.random() * 0.4 : 0.1 + Math.random() * 0.2,
      file_access_volume: 0.2 + Math.random() * 0.5,
      email_external_ratio: user.risk_score > 50 ? 0.4 + Math.random() * 0.4 : 0.05 + Math.random() * 0.15,
      usb_usage: Math.random() * 0.3,
      network_anomaly: user.risk_score > 70 ? 0.5 + Math.random() * 0.4 : 0.05 + Math.random() * 0.15,
      badge_anomaly: Math.random() * 0.3,
      data_exfiltration_score: user.risk_score > 80 ? 0.6 + Math.random() * 0.3 : 0.05 + Math.random() * 0.2,
    },
    peer_comparison: {
      peer_group: `${user.department} peers`,
      peer_count: 12 + Math.floor(Math.random() * 20),
      metrics: [
        { feature: "Login Freq", user_value: 5.2, peer_mean: 4.1, peer_p95: 8.3 },
        { feature: "File Access", user_value: user.risk_score > 60 ? 120 : 45, peer_mean: 52, peer_p95: 95 },
        { feature: "Email Vol", user_value: user.risk_score > 50 ? 87 : 32, peer_mean: 41, peer_p95: 78 },
        { feature: "After Hours", user_value: user.risk_score > 60 ? 35 : 8, peer_mean: 12, peer_p95: 28 },
        { feature: "USB Usage", user_value: user.risk_score > 70 ? 4 : 0, peer_mean: 0.5, peer_p95: 2 },
      ],
    },
    context_factors: user.risk_score > 60
      ? [
          { factor: "Elevated access", multiplier: 1.4, description: "Above-normal resource access" },
          { factor: "Behavioural shift", multiplier: 1.3, description: "Recent pattern change detected" },
        ]
      : [],
    xai_report: user.risk_score > 50
      ? {
          executive_summary: `${user.username} shows ${user.risk_score > 75 ? "significant" : "moderate"} deviation from baseline behavior patterns. Key drivers include file access volume and temporal anomalies.`,
          model_attribution: [
            { detector: "Isolation Forest", contribution_pct: 35, raw_score: 0.72, anomaly_type: "Statistical outlier" },
            { detector: "Autoencoder", contribution_pct: 28, raw_score: 0.65, anomaly_type: "Reconstruction error" },
            { detector: "LSTM Temporal", contribution_pct: 22, raw_score: 0.58, anomaly_type: "Temporal deviation" },
            { detector: "GNN Relational", contribution_pct: 15, raw_score: 0.45, anomaly_type: "Relational anomaly" },
          ],
          shap_features: [
            { feature: "file_access_volume", shap_value: 0.23, feature_value: 120, direction: "increases_risk" as const },
            { feature: "after_hours_ratio", shap_value: 0.18, feature_value: 0.7, direction: "increases_risk" as const },
            { feature: "email_external_ratio", shap_value: 0.12, feature_value: 0.45, direction: "increases_risk" as const },
          ],
          baseline_comparison: {
            current_score: user.risk_score,
            baseline_score: user.risk_score * 0.4,
            deviation_pct: 150,
          },
          peer_group_analysis: {
            peer_group: user.department,
            peer_count: 15,
            user_zscore: 2.5,
            peer_mean: user.risk_score * 0.5,
            peer_std: 10.0,
          },
          investigation_playbook: [
            "Review recent access patterns",
            "Compare with peer behavior baseline",
            "Check for any HR events (PIP, resignation, etc.)",
          ],
        }
      : undefined,
    recent_alerts: MOCK_ALERTS.filter((a) => a.entity_id === userId)
      .slice(0, 5)
      .map((a) => ({
        id: a.id,
        entity_id: a.entity_id,
        entity_name: a.entity_name,
        title: a.title,
        description: a.description,
        severity: a.severity,
        risk_score: a.risk_score,
        status: a.status,
        dedup_key: a.dedup_key,
        created_at: a.created_at,
        updated_at: a.updated_at,
      })),
  };
}

// ── Risk Scores ──────────────────────────────────────────────

export const MOCK_RISK_SCORES: RiskScore[] = MOCK_USERS.map((u) => {
  const base = u.risk_score / 100;
  return {
    entity_id: u.id,
    username: u.username,
    risk_score: u.risk_score,
    risk_level: (u.risk_score >= 75 ? "critical" : u.risk_score >= 50 ? "high" : u.risk_score >= 25 ? "medium" : "low") as import("./types").RiskLevel,
    timestamp: hoursAgo(Math.random() * 2),
    department: u.department,
    context_multiplier: u.risk_score > 60 ? 1.3 + Math.random() * 0.5 : 1.0,
    detector_scores: [
      { detector: "isolation_forest", raw_score: base * (0.8 + Math.random() * 0.4), weight: 0.30, weighted_score: base * 0.30 },
      { detector: "autoencoder", raw_score: base * (0.7 + Math.random() * 0.4), weight: 0.25, weighted_score: base * 0.25 },
      { detector: "lstm_temporal", raw_score: base * (0.6 + Math.random() * 0.5), weight: 0.25, weighted_score: base * 0.25 },
      { detector: "gnn_relational", raw_score: base * (0.5 + Math.random() * 0.5), weight: 0.20, weighted_score: base * 0.20 },
    ],
  };
});
