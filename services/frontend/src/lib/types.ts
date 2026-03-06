/**
 * Shared TypeScript types for the Insider Threat Detection System frontend.
 *
 * These mirror the Pydantic schemas from:
 *   - risk-scoring-service
 *   - alert-service
 *   - ingestion-service
 *   - ml-engine
 */

/* ═══════════════════════════════════════════════════════════════════════
   Risk & Scoring
   ═══════════════════════════════════════════════════════════════════════ */

export type RiskLevel = "low" | "medium" | "high" | "critical";

export interface RiskScore {
  entity_id: string;
  username: string;
  department: string;
  risk_score: number;
  risk_level: RiskLevel;
  detector_scores: DetectorScore[];
  context_multiplier: number;
  timestamp: string;
}

export interface DetectorScore {
  detector: string;           // isolation_forest | autoencoder | lstm | gnn
  raw_score: number;
  weight: number;
  weighted_score: number;
}

export interface RiskTrendPoint {
  timestamp: string;
  score: number;
}

export interface ThresholdConfig {
  low: number;
  medium: number;
  high: number;
  critical: number;
  updated_at: string;
}

export interface TrendAnalysis {
  entity_id: string;
  slope: number;
  direction: "rising" | "falling" | "stable";
  slow_burn: boolean;
  data_points: number;
  window_days: number;
}

/* ═══════════════════════════════════════════════════════════════════════
   Alerts
   ═══════════════════════════════════════════════════════════════════════ */

export type AlertStatus =
  | "NEW"
  | "INVESTIGATING"
  | "RESOLVED"
  | "FALSE_POSITIVE"
  | "ESCALATED";

export interface Alert {
  id: string;
  entity_id: string;
  entity_name: string;
  severity: RiskLevel;
  risk_score: number;
  title: string;
  description: string;
  status: AlertStatus;
  dedup_key: string;
  created_at: string;
  updated_at: string;
}

export interface AlertDetail extends Alert {
  recommended_actions: string[];
  xai_report: XAIReport | null;
  evidence: EvidenceItem[];
  audit_trail: AuditEntry[];
  context_factors: ContextFactor[];
}

export interface PaginatedAlerts {
  items: Alert[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

/* ═══════════════════════════════════════════════════════════════════════
   XAI (Explainable AI) Reports
   ═══════════════════════════════════════════════════════════════════════ */

export interface XAIReport {
  executive_summary: string;
  model_attribution: ModelAttribution[];
  shap_features: SHAPFeature[];
  baseline_comparison: BaselineComparison;
  peer_group_analysis: PeerGroupAnalysis;
  investigation_playbook: string[];
}

export interface ModelAttribution {
  detector: string;
  contribution_pct: number;
  raw_score: number;
  anomaly_type: string;
}

export interface SHAPFeature {
  feature: string;
  shap_value: number;
  feature_value: number;
  direction: "increases_risk" | "decreases_risk";
}

export interface BaselineComparison {
  current_score: number;
  baseline_score: number;
  deviation_pct: number;
  period: string;
}

export interface PeerGroupAnalysis {
  peer_group: string;
  peer_count: number;
  user_zscore: number;
  peer_mean: number;
  peer_std: number;
}

/* ═══════════════════════════════════════════════════════════════════════
   Evidence & Audit
   ═══════════════════════════════════════════════════════════════════════ */

export interface EvidenceItem {
  type: "login" | "file_access" | "email" | "usb" | "network" | "badge";
  timestamp: string;
  description: string;
  risk_contribution: number;
  raw_data: Record<string, unknown>;
}

export interface AuditEntry {
  action: string;
  actor: string;
  timestamp: string;
  details: string;
}

export interface ContextFactor {
  factor: string;           // e.g. "pto_overlap", "resignation_notice"
  multiplier: number;
  description: string;
}

/* ═══════════════════════════════════════════════════════════════════════
   Feedback
   ═══════════════════════════════════════════════════════════════════════ */

export type FeedbackVerdict =
  | "TRUE_POSITIVE"
  | "FALSE_POSITIVE"
  | "ESCALATE"
  | "NEEDS_REVIEW";

export interface FeedbackPayload {
  alert_id: string;
  verdict: FeedbackVerdict;
  analyst_notes: string;
}

export interface FeedbackResponse {
  id: string;
  alert_id: string;
  verdict: FeedbackVerdict;
  analyst_id: string;
  analyst_notes: string;
  created_at: string;
}

/* ═══════════════════════════════════════════════════════════════════════
   User / Entity Profile
   ═══════════════════════════════════════════════════════════════════════ */

export interface UserProfile {
  id: string;
  username: string;
  email: string;
  department: string;
  title: string;
  manager: string;
  hire_date: string;
  risk_score: number;
  risk_level: RiskLevel;
  risk_history: RiskTrendPoint[];
  behavior_features: BehaviorFeatures;
  peer_comparison: PeerComparison;
  recent_alerts: Alert[];
  context_factors: ContextFactor[];
  xai_report: XAIReport | null;
}

export interface BehaviorFeatures {
  login_frequency: number;
  after_hours_ratio: number;
  file_access_volume: number;
  email_external_ratio: number;
  usb_usage: number;
  network_anomaly: number;
  badge_anomaly: number;
  data_exfiltration_score: number;
}

export interface PeerComparison {
  peer_group: string;
  peer_count: number;
  metrics: PeerMetric[];
}

export interface PeerMetric {
  feature: string;
  user_value: number;
  peer_mean: number;
  peer_p95: number;
  z_score: number;
}

/* ═══════════════════════════════════════════════════════════════════════
   Dashboard
   ═══════════════════════════════════════════════════════════════════════ */

export interface DashboardMetrics {
  total_alerts: number;
  critical_entities: number;
  average_risk_score: number;
  active_entities: number;
  risk_distribution: { level: string; count: number }[];
  alert_trend: { date: string; count: number }[];
  top_entities: RiskScore[];
  department_heatmap: DepartmentHeatmapEntry[];
}

export interface DepartmentHeatmapEntry {
  department: string;
  entity_count: number;
  avg_risk: number;
  critical_count: number;
}

/* ═══════════════════════════════════════════════════════════════════════
   Model Health & Privacy
   ═══════════════════════════════════════════════════════════════════════ */

export interface DetectorHealth {
  detector: string;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  last_trained: string;
  training_samples: number;
}

export interface PrivacyBudget {
  total_epsilon: number;
  consumed_epsilon: number;
  remaining_epsilon: number;
  total_delta: number;
  consumed_delta: number;
  queries_today: number;
  budget_refresh_at: string;
}

export interface FederatedLearningStatus {
  current_round: number;
  total_rounds: number;
  participating_clients: number;
  global_accuracy: number;
  last_aggregation: string;
  convergence_delta: number;
  status: "training" | "aggregating" | "idle" | "converged";
}

export interface ModelHealthOverview {
  detectors: DetectorHealth[];
  ensemble_weights: Record<string, number>;
  privacy_budget: PrivacyBudget;
  federated_learning: FederatedLearningStatus;
  fp_fn_analysis: FPFNReport;
}

export interface FPFNReport {
  overall_precision: number;
  overall_recall: number;
  overall_f1: number;
  overall_auc_roc: number;
  by_detector: DetectorHealth[];
  by_department: DepartmentMetrics[];
}

export interface DepartmentMetrics {
  department: string;
  precision: number;
  recall: number;
  f1_score: number;
  alert_count: number;
  fp_count: number;
}

/* ═══════════════════════════════════════════════════════════════════════
   WebSocket Events
   ═══════════════════════════════════════════════════════════════════════ */

export interface WSRiskUpdate {
  event: "risk_update";
  payload: RiskScore;
}

export interface WSAlertCreated {
  event: "alert_created";
  payload: Alert;
}

export interface WSAlertUpdated {
  event: "alert_updated";
  payload: Alert;
}

export type WSEvent = WSRiskUpdate | WSAlertCreated | WSAlertUpdated;

/* ═══════════════════════════════════════════════════════════════════════
   Live Feed
   ═══════════════════════════════════════════════════════════════════════ */

export interface LiveFeedItem {
  id: string;
  type: "alert" | "risk_change" | "status_change";
  entity_id: string;
  entity_name: string;
  message: string;
  severity: RiskLevel;
  timestamp: string;
}
