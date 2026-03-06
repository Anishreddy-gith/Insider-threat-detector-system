/**
 * useUserProfile – React Query hook for entity / user profile data.
 *
 * Provides the full user profile including:
 * - 30-day risk history
 * - Behavior radar features
 * - Peer comparison
 * - Recent alerts
 * - XAI report
 */

import { useQuery } from "@tanstack/react-query";
import { fetchUserProfile } from "@/lib/services";

export function useUserProfile(entityId: string | undefined) {
  return useQuery({
    queryKey: ["userProfile", entityId],
    queryFn: () => fetchUserProfile(entityId!),
    enabled: !!entityId,
    staleTime: 15_000,
  });
}
