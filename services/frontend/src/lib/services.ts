/**
 * API service functions – typed wrappers around the Axios instance.
 *
 * All calls return the Axios response data directly (unwrapped),
 * so React Query hooks can consume them without `.data` chaining.
 */

import api from "./api";
import type {
  Alert,
  AlertDetail,
  DashboardMetrics,
  FeedbackPayload,
  FeedbackResponse,
  FPFNReport,
  ModelHealthOverview,
  PaginatedAlerts,
  RiskScore,
  ThresholdConfig,
  TrendAnalysis,
  UserProfile,
} from "./types";

/* ═══════════════════════════════════════════════════════════════════════
   Dashboard
   ═══════════════════════════════════════════════════════════════════════ */

export async function fetchDashboard(): Promise<DashboardMetrics> {
  const { data } = await api.get("/analytics/dashboard");
  return data;
}

/* ═══════════════════════════════════════════════════════════════════════
   Risk Scores
   ═══════════════════════════════════════════════════════════════════════ */

export async function fetchRiskScores(params?: {
  department?: string;
  min_score?: number;
  limit?: number;
  offset?: number;
}): Promise<RiskScore[]> {
  const { data } = await api.get("/risk/scores", { params });
  return data;
}

export async function fetchEntityRiskScore(
  entityId: string
): Promise<RiskScore> {
  const { data } = await api.get(`/risk/scores/${entityId}`);
  return data;
}

export async function fetchRiskTrend(
  entityId: string,
  days = 30
): Promise<TrendAnalysis> {
  const { data } = await api.get(`/risk/trends/${entityId}`, {
    params: { days },
  });
  return data;
}

export async function fetchThresholds(): Promise<ThresholdConfig> {
  const { data } = await api.get("/risk/thresholds");
  return data;
}

/* ═══════════════════════════════════════════════════════════════════════
   Alerts
   ═══════════════════════════════════════════════════════════════════════ */

export async function fetchAlerts(params?: {
  page?: number;
  size?: number;
  severity?: string;
  status?: string;
}): Promise<PaginatedAlerts> {
  const { data } = await api.get("/alerts", { params });
  return data;
}

export async function fetchAlert(alertId: string): Promise<AlertDetail> {
  const { data } = await api.get(`/alerts/${alertId}`);
  return data;
}

export async function updateAlertStatus(
  alertId: string,
  status: string
): Promise<Alert> {
  const { data } = await api.patch(`/alerts/${alertId}`, { status });
  return data;
}

/* ═══════════════════════════════════════════════════════════════════════
   Feedback
   ═══════════════════════════════════════════════════════════════════════ */

export async function submitFeedback(
  payload: FeedbackPayload
): Promise<FeedbackResponse> {
  const { data } = await api.post("/alerts/feedback", payload);
  return data;
}

/* ═══════════════════════════════════════════════════════════════════════
   User / Entity Profile
   ═══════════════════════════════════════════════════════════════════════ */

export async function fetchUserProfile(
  entityId: string
): Promise<UserProfile> {
  const { data } = await api.get(`/analytics/entities/${entityId}`);
  return data;
}

/* ═══════════════════════════════════════════════════════════════════════
   Model Health & Privacy
   ═══════════════════════════════════════════════════════════════════════ */

export async function fetchModelHealth(): Promise<ModelHealthOverview> {
  const { data } = await api.get("/ml/health");
  return data;
}

export async function fetchFPFNReport(): Promise<FPFNReport> {
  const { data } = await api.get("/alerts/analysis");
  return data;
}
