/// <reference types="vite/client" />
/**
 * WebSocket hook for real-time risk score updates and alert notifications.
 *
 * Connects to the Socket.IO server and emits typed events that
 * React Query caches and UI components can consume.
 */

import { useEffect, useRef, useCallback } from "react";
import { io, type Socket } from "socket.io-client";
import { useQueryClient } from "@tanstack/react-query";
import type {
  RiskScore,
  Alert,
  LiveFeedItem,
} from "@/lib/types";

type SocketEventHandler = {
  onRiskUpdate?: (score: RiskScore) => void;
  onAlertCreated?: (alert: Alert) => void;
  onAlertUpdated?: (alert: Alert) => void;
  onLiveFeed?: (item: LiveFeedItem) => void;
};

const WS_URL = import.meta.env.VITE_WS_URL ?? "http://localhost:8000";

export function useWebSocket(handlers?: SocketEventHandler) {
  const socketRef = useRef<Socket | null>(null);
  const queryClient = useQueryClient();

  useEffect(() => {
    const socket = io(WS_URL, {
      path: "/ws/socket.io",
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });

    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("[WS] Connected:", socket.id);
    });

    socket.on("disconnect", (reason) => {
      console.log("[WS] Disconnected:", reason);
    });

    // ── Risk score updates → invalidate risk queries ──────────────
    socket.on("risk_update", (payload: RiskScore) => {
      // Invalidate dashboard and risk-related queries
      queryClient.invalidateQueries({ queryKey: ["dashboard"] });
      queryClient.invalidateQueries({ queryKey: ["riskScores"] });
      queryClient.invalidateQueries({
        queryKey: ["userProfile", payload.entity_id],
      });
      handlers?.onRiskUpdate?.(payload);
    });

    // ── New alert → invalidate alert queries ──────────────────────
    socket.on("alert_created", (payload: Alert) => {
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard"] });
      handlers?.onAlertCreated?.(payload);
    });

    // ── Alert status change ───────────────────────────────────────
    socket.on("alert_updated", (payload: Alert) => {
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      queryClient.invalidateQueries({
        queryKey: ["alert", payload.id],
      });
      handlers?.onAlertUpdated?.(payload);
    });

    // ── Generic live-feed event ───────────────────────────────────
    socket.on("live_feed", (item: LiveFeedItem) => {
      handlers?.onLiveFeed?.(item);
    });

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const emit = useCallback(
    (event: string, data?: unknown) => {
      socketRef.current?.emit(event, data);
    },
    []
  );

  return { socket: socketRef, emit };
}
