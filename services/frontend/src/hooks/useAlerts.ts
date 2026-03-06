/**
 * useAlerts – React Query hooks for alert CRUD and feedback.
 *
 * Provides:
 * - Paginated alert list with severity/status filters
 * - Single alert detail
 * - Optimistic status update mutation
 * - Feedback submission mutation
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchAlerts,
  fetchAlert,
  updateAlertStatus,
  submitFeedback,
} from "@/lib/services";
import type { FeedbackPayload } from "@/lib/types";

export function useAlerts(params?: {
  page?: number;
  size?: number;
  severity?: string;
  status?: string;
}) {
  return useQuery({
    queryKey: ["alerts", params],
    queryFn: () => fetchAlerts(params),
    staleTime: 5_000,
    refetchInterval: 15_000,
  });
}

export function useAlert(alertId: string | undefined) {
  return useQuery({
    queryKey: ["alert", alertId],
    queryFn: () => fetchAlert(alertId!),
    enabled: !!alertId,
    staleTime: 5_000,
  });
}

export function useUpdateAlertStatus() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      alertId,
      status,
    }: {
      alertId: string;
      status: string;
    }) => updateAlertStatus(alertId, status),

    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["alert", variables.alertId] });
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
}

export function useSubmitFeedback() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: FeedbackPayload) => submitFeedback(payload),

    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["alert", variables.alert_id],
      });
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      queryClient.invalidateQueries({ queryKey: ["modelHealth"] });
    },
  });
}
