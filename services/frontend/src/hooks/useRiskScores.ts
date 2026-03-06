/**
 * useRiskScores – React Query hook for risk score data.
 *
 * Provides:
 * - All entity risk scores (with optional department/min_score filters)
 * - Single entity risk score
 * - Threshold configuration
 * - 10s auto-refetch (supplemented by WebSocket invalidation)
 */

import { useQuery } from "@tanstack/react-query";
import {
  fetchRiskScores,
  fetchEntityRiskScore,
  fetchThresholds,
  fetchRiskTrend,
} from "@/lib/services";

export function useRiskScores(params?: {
  department?: string;
  min_score?: number;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: ["riskScores", params],
    queryFn: () => fetchRiskScores(params),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

export function useEntityRiskScore(entityId: string | undefined) {
  return useQuery({
    queryKey: ["riskScore", entityId],
    queryFn: () => fetchEntityRiskScore(entityId!),
    enabled: !!entityId,
    staleTime: 5_000,
  });
}

export function useThresholds() {
  return useQuery({
    queryKey: ["thresholds"],
    queryFn: fetchThresholds,
    staleTime: 60_000,
  });
}

export function useRiskTrend(entityId: string | undefined, days = 30) {
  return useQuery({
    queryKey: ["riskTrend", entityId, days],
    queryFn: () => fetchRiskTrend(entityId!, days),
    enabled: !!entityId,
    staleTime: 30_000,
  });
}
