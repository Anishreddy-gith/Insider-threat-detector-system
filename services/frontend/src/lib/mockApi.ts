/**
 * Mock API interceptor — intercepts all axios requests and returns
 * realistic demo data so the frontend works fully without a backend.
 *
 * Activated when VITE_MOCK_API=true or when the backend is unreachable.
 */

import type { InternalAxiosRequestConfig, AxiosResponse } from "axios";
import api from "./api";
import {
  MOCK_DASHBOARD,
  MOCK_ALERTS,
  MOCK_USERS,
  MOCK_MODEL_HEALTH,
  MOCK_RISK_SCORES,
  buildUserProfile,
} from "./mockData";

function mockResponse<T>(data: T, delay = 150): Promise<AxiosResponse<T>> {
  return new Promise((resolve) =>
    setTimeout(
      () =>
        resolve({
          data,
          status: 200,
          statusText: "OK",
          headers: {},
          config: {} as InternalAxiosRequestConfig,
        }),
      delay
    )
  );
}

/** Install mock request interceptor on the shared axios instance. */
export function enableMockApi(): void {
  // Eject the 401 response interceptor — we won't need refresh logic
  api.interceptors.response.handlers?.forEach((_h, i) => {
    api.interceptors.response.eject(i);
  });

  api.interceptors.request.use(async (config) => {
    const url = config.url ?? "";
    const method = (config.method ?? "get").toLowerCase();

    // ── Auth ──
    if (url.includes("/auth/login") && method === "post") {
      return Promise.reject({
        response: await mockResponse({
          access_token: "mock-jwt-access-token",
          refresh_token: "mock-jwt-refresh-token",
        }),
      });
    }
    if (url.includes("/auth/me")) {
      return Promise.reject({
        response: await mockResponse({
          id: "admin-001",
          username: "demo_admin",
          email: "admin@itds.demo",
          role: "admin",
          full_name: "Demo Administrator",
        }),
      });
    }

    // ── Dashboard ──
    if (url.includes("/analytics/dashboard")) {
      return Promise.reject({ response: await mockResponse(MOCK_DASHBOARD) });
    }

    // ── Entity detail ──
    const entityDetailMatch = url.match(/\/analytics\/entities\/(.+)/);
    if (entityDetailMatch) {
      const profile = buildUserProfile(entityDetailMatch[1]);
      if (profile) {
        return Promise.reject({ response: await mockResponse(profile) });
      }
    }

    // ── Entities list ──
    if (url.includes("/analytics/entities")) {
      const params = config.params ?? {};
      const page = Number(params.page ?? 1);
      const size = Number(params.size ?? 24);
      const search = (params.search ?? "").toLowerCase();
      let filtered = MOCK_USERS;
      if (search) {
        filtered = MOCK_USERS.filter(
          (u) =>
            u.username.toLowerCase().includes(search) ||
            u.department.toLowerCase().includes(search)
        );
      }
      const total = filtered.length;
      const items = filtered.slice((page - 1) * size, page * size);
      return Promise.reject({
        response: await mockResponse({
          items,
          total,
          page,
          size,
          pages: Math.ceil(total / size),
        }),
      });
    }

    // ── Alert detail ──
    const alertDetailMatch = url.match(/\/alerts\/([a-z]-\d+)$/);
    if (alertDetailMatch && method === "get") {
      const alert = MOCK_ALERTS.find((a) => a.id === alertDetailMatch[1]);
      if (alert) {
        return Promise.reject({ response: await mockResponse(alert) });
      }
    }

    // ── Alert status update ──
    if (url.match(/\/alerts\/[a-z]-\d+$/) && method === "patch") {
      const id = url.split("/").pop();
      const alert = MOCK_ALERTS.find((a) => a.id === id);
      if (alert) {
        const body =
          typeof config.data === "string"
            ? JSON.parse(config.data)
            : config.data;
        alert.status = body.status ?? alert.status;
        return Promise.reject({ response: await mockResponse(alert) });
      }
    }

    // ── Feedback ──
    if (url.includes("/alerts/feedback") && method === "post") {
      return Promise.reject({
        response: await mockResponse({
          id: "fb-001",
          alert_id: "a-001",
          verdict: "TRUE_POSITIVE",
          status: "accepted",
        }),
      });
    }

    // ── Alerts list ──
    if (url.includes("/alerts")) {
      const params = config.params ?? {};
      const page = Number(params.page ?? 1);
      const size = Number(params.size ?? 20);
      const severity = params.severity;
      let filtered = MOCK_ALERTS;
      if (severity) {
        filtered = MOCK_ALERTS.filter((a) => a.severity === severity);
      }
      const total = filtered.length;
      const items = filtered.slice((page - 1) * size, page * size);
      return Promise.reject({
        response: await mockResponse({
          items,
          total,
          page,
          size,
          pages: Math.max(1, Math.ceil(total / size)),
        }),
      });
    }

    // ── Risk scores ──
    if (url.includes("/risk/scores")) {
      return Promise.reject({
        response: await mockResponse(MOCK_RISK_SCORES),
      });
    }
    if (url.includes("/risk/thresholds")) {
      return Promise.reject({
        response: await mockResponse({
          low: 25,
          medium: 50,
          high: 75,
          critical: 90,
        }),
      });
    }

    // ── Model health ──
    if (url.includes("/ml/health")) {
      return Promise.reject({
        response: await mockResponse(MOCK_MODEL_HEALTH),
      });
    }

    // Fallback
    return Promise.reject({
      response: await mockResponse({ error: "Not found" }, 50),
    });
  });

  // Re-add a response interceptor that extracts mock responses from our
  // "rejected" promises. We reject with { response } so we can intercept
  // it here and turn it into a successful response.
  api.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error?.response?.status === 200 || error?.response?.data) {
        return Promise.resolve(error.response);
      }
      return Promise.reject(error);
    }
  );

  console.log(
    "%c[ITDS] Mock API enabled — running in demo mode",
    "color: #22c55e; font-weight: bold;"
  );
}
