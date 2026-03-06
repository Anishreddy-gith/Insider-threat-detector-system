/**
 * Live-feed Zustand store – ring-buffer of the last 50 real-time events.
 *
 * The WebSocket hook pushes items here; the Dashboard live-feed panel
 * subscribes to this store for O(1) re-renders.
 */

import { create } from "zustand";
import type { LiveFeedItem } from "@/lib/types";

const MAX_ITEMS = 50;

interface LiveFeedState {
  items: LiveFeedItem[];
  push: (item: LiveFeedItem) => void;
  clear: () => void;
}

export const useLiveFeedStore = create<LiveFeedState>((set) => ({
  items: [],
  push: (item) =>
    set((state) => ({
      items: [item, ...state.items].slice(0, MAX_ITEMS),
    })),
  clear: () => set({ items: [] }),
}));
