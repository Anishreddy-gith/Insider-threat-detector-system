/**
 * Zustand auth store – lightweight state management for JWT lifecycle.
 *
 * WHY Zustand over Redux?  Insider threat dashboards have modest client
 * state (auth tokens, cached dashboard data).  Zustand has ~1 KB gzipped
 * vs Redux Toolkit's ~12 KB, with zero boilerplate.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  user: {
    id: string;
    username: string;
    email: string;
    role: string;
    fullName: string | null;
  } | null;
  setTokens: (access: string, refresh: string) => void;
  setUser: (user: AuthState["user"]) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      accessToken: null,
      refreshToken: null,
      user: null,
      setTokens: (access, refresh) =>
        set({ accessToken: access, refreshToken: refresh }),
      setUser: (user) => set({ user }),
      logout: () =>
        set({ accessToken: null, refreshToken: null, user: null }),
    }),
    {
      name: "itds-auth",
      // Only persist tokens & user, not functions
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        user: state.user,
      }),
    }
  )
);
