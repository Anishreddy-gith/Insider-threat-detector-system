import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

import { enableMockApi } from "./lib/mockApi";
import { useAuthStore } from "./store/authStore";

// ── Demo mode: enable mock API and seed auth state ──────────
const DEMO_MODE =
  import.meta.env.VITE_MOCK_API === "true" ||
  !import.meta.env.VITE_API_URL;

if (DEMO_MODE) {
  enableMockApi();
  // Seed auth so we bypass the login screen and land on the dashboard
  const { accessToken } = useAuthStore.getState();
  if (!accessToken) {
    useAuthStore.getState().setTokens("mock-jwt-access-token", "mock-jwt-refresh-token");
    useAuthStore.getState().setUser({
      id: "admin-001",
      username: "demo_admin",
      email: "admin@itds.demo",
      role: "admin",
      fullName: "Demo Administrator",
    });
  }
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: DEMO_MODE ? 0 : 2,
      refetchOnWindowFocus: !DEMO_MODE,
      staleTime: 5_000,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
