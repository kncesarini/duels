// The single piece of client-side state for the whole app. It never
// computes a rule: it only remembers the most recent `StatePayload` the
// server sent and forwards `Action`s the server told us were legal.

import { create } from "zustand";

import type { Action } from "./generated/Action";
import type { Catalog } from "./generated/Catalog";
import type { Event } from "./generated/Event";
import type { StatePayload } from "./generated/StatePayload";
import { connectRoomSocket, createRoom, fetchCatalog, sendAction } from "./lib/api";

export type ConnectionStatus = "idle" | "connecting" | "connected" | "closed" | "error";

const MAX_EVENT_LOG = 40;

interface GameStore {
  catalog: Catalog | null;
  catalogError: string | null;

  roomId: string | null;
  mode: "bot" | "hotseat" | null;
  status: ConnectionStatus;
  payload: StatePayload | null;
  errorMessage: string | null;
  eventLog: Event[];

  loadCatalog: () => Promise<void>;
  startVsBot: (seed?: number) => Promise<void>;
  startHotSeat: (seed?: number) => Promise<void>;
  submitAction: (action: Action) => void;
  leaveGame: () => void;
}

let socket: WebSocket | null = null;

function closeSocket() {
  if (socket) {
    socket.onclose = null;
    socket.close();
    socket = null;
  }
}

function connect(roomId: string, mode: "bot" | "hotseat") {
  closeSocket();
  useGameStore.setState({ roomId, mode, status: "connecting", errorMessage: null, payload: null, eventLog: [] });
  socket = connectRoomSocket(roomId, {
    onOpen: () => useGameStore.setState({ status: "connected" }),
    onClose: () => useGameStore.setState({ status: "closed" }),
    onError: () => useGameStore.setState({ status: "error", errorMessage: "WebSocket connection error" }),
    onMessage: (msg) => {
      if (msg.type === "State") {
        const state = msg as { type: "State" } & StatePayload;
        useGameStore.setState((s) => ({
          payload: state,
          eventLog: [...s.eventLog, ...state.events].slice(-MAX_EVENT_LOG),
        }));
      } else if (msg.type === "Error") {
        useGameStore.setState({ errorMessage: msg.message });
      }
    },
  });
}

export const useGameStore = create<GameStore>((set, get) => ({
  catalog: null,
  catalogError: null,

  roomId: null,
  mode: null,
  status: "idle",
  payload: null,
  errorMessage: null,
  eventLog: [],

  loadCatalog: async () => {
    if (get().catalog) return;
    try {
      const catalog = await fetchCatalog();
      set({ catalog, catalogError: null });
    } catch (e) {
      set({ catalogError: e instanceof Error ? e.message : String(e) });
    }
  },

  startVsBot: async (seed) => {
    set({ status: "connecting", errorMessage: null });
    try {
      const res = await createRoom({
        seats: [{ kind: "human" }, { kind: "agent", name: "random" }],
        seed: seed ?? null,
      });
      connect(res.room_id, "bot");
    } catch (e) {
      set({ status: "error", errorMessage: e instanceof Error ? e.message : String(e) });
    }
  },

  startHotSeat: async (seed) => {
    set({ status: "connecting", errorMessage: null });
    try {
      const res = await createRoom({
        seats: [{ kind: "human" }, { kind: "human" }],
        seed: seed ?? null,
      });
      connect(res.room_id, "hotseat");
    } catch (e) {
      set({ status: "error", errorMessage: e instanceof Error ? e.message : String(e) });
    }
  },

  submitAction: (action) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    sendAction(socket, action);
  },

  leaveGame: () => {
    closeSocket();
    set({ roomId: null, mode: null, status: "idle", payload: null, errorMessage: null, eventLog: [] });
  },
}));
