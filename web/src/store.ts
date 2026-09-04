// The single piece of client-side state for the whole app. It never
// computes a rule: it only remembers the most recent `StatePayload` the
// server sent and forwards `Action`s the server told us were legal.

import { create } from "zustand";

import type { Action } from "./generated/Action";
import type { Catalog } from "./generated/Catalog";
import type { Event } from "./generated/Event";
import type { StatePayload } from "./generated/StatePayload";
import { connectRoomSocket, createRoom, fetchAgents, fetchCatalog, sendAction } from "./lib/api";

export type ConnectionStatus = "idle" | "connecting" | "connected" | "reconnecting" | "closed" | "error";

const MAX_EVENT_LOG = 40;
const MAX_RECONNECT_ATTEMPTS = 5;

interface GameStore {
  catalog: Catalog | null;
  catalogError: string | null;

  /** Agent names `POST /rooms` will accept, from `GET /agents`. `["random"]`
   * until `loadAgents` resolves, so the opponent picker always has at least
   * the one opponent the e2e suite (and every existing save) relies on. */
  agents: string[];
  agentsError: string | null;

  roomId: string | null;
  mode: "bot" | "hotseat" | null;
  status: ConnectionStatus;
  payload: StatePayload | null;
  errorMessage: string | null;
  eventLog: Event[];
  /** True from the moment `submitAction` sends an action until the server's
   * reply (a new `State` or an `Error`) arrives, or the connection drops.
   * Every action-submitting control reads this to disable itself while a
   * round trip is in flight, so a slow reply (a network hiccup - the app
   * talks to `duels-server` over a real, if usually fast, connection) reads
   * as "still working" rather than "my click did nothing", which is
   * indistinguishable from a dead client to a player with no other signal. */
  pending: boolean;

  loadCatalog: () => Promise<void>;
  loadAgents: () => Promise<void>;
  startVsBot: (seed?: number, agent?: string) => Promise<void>;
  startHotSeat: (seed?: number) => Promise<void>;
  submitAction: (action: Action) => void;
  leaveGame: () => void;
}

// `socket` is the one currently "live" connection; `socketEpoch` disambiguates
// it from any earlier one whose close/message events are still in flight
// (e.g. a `leaveGame()` or reconnect racing with the socket it's replacing).
// Every handler below closes over the epoch it was created for and ignores
// itself if a newer connection has since taken over - without this, a stale
// socket's belated `close` (or, in principle, a last in-flight `message`)
// could otherwise clobber the state of the room the app has since moved to.
let socket: WebSocket | null = null;
let socketEpoch = 0;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function clearReconnectTimer() {
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function closeSocket() {
  clearReconnectTimer();
  socketEpoch += 1;
  if (socket) {
    socket.close();
    socket = null;
  }
}

/**
 * Open the WebSocket for `roomId` and wire it into the store. On an
 * unexpected drop (not one caused by `leaveGame()`/`closeSocket()`), the
 * connection is transparently re-established against the same room - the
 * server always answers a fresh connection with its current snapshot (see
 * `duels-server`'s `room_ws`), so a transient network hiccup self-heals
 * instead of leaving the player stuck on a frozen board with no feedback and
 * every further click silently swallowed by `submitAction`'s `readyState`
 * check.
 */
function connect(roomId: string, mode: "bot" | "hotseat", reconnectAttempt = 0) {
  clearReconnectTimer();
  const epoch = ++socketEpoch;
  if (socket) {
    socket.close();
    socket = null;
  }
  if (reconnectAttempt === 0) {
    useGameStore.setState({
      roomId,
      mode,
      status: "connecting",
      errorMessage: null,
      payload: null,
      eventLog: [],
      pending: false,
    });
  } else {
    useGameStore.setState({ status: "reconnecting", pending: false });
  }
  socket = connectRoomSocket(roomId, {
    onOpen: () => {
      if (epoch !== socketEpoch) return;
      useGameStore.setState({ status: "connected" });
    },
    onClose: () => {
      if (epoch !== socketEpoch) return;
      useGameStore.setState({ status: "closed", pending: false });
      if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
        useGameStore.setState({
          errorMessage: "Lost connection to the game server and could not reconnect. Reload the page to try again.",
        });
        return;
      }
      reconnectTimer = setTimeout(
        () => connect(roomId, mode, reconnectAttempt + 1),
        500 * (reconnectAttempt + 1),
      );
    },
    onError: () => {
      if (epoch !== socketEpoch) return;
      useGameStore.setState({ status: "error", errorMessage: "WebSocket connection error" });
    },
    onMessage: (msg) => {
      if (epoch !== socketEpoch) return;
      if (msg.type === "State") {
        const state = msg as { type: "State" } & StatePayload;
        useGameStore.setState((s) => ({
          payload: state,
          status: "connected",
          pending: false,
          eventLog: [...s.eventLog, ...state.events].slice(-MAX_EVENT_LOG),
        }));
      } else if (msg.type === "Error") {
        useGameStore.setState({ errorMessage: msg.message, pending: false });
      }
    },
  });
}

export const useGameStore = create<GameStore>((set, get) => ({
  catalog: null,
  catalogError: null,

  agents: ["random"],
  agentsError: null,

  roomId: null,
  mode: null,
  status: "idle",
  payload: null,
  errorMessage: null,
  eventLog: [],
  pending: false,

  loadCatalog: async () => {
    if (get().catalog) return;
    try {
      const catalog = await fetchCatalog();
      set({ catalog, catalogError: null });
    } catch (e) {
      set({ catalogError: e instanceof Error ? e.message : String(e) });
    }
  },

  loadAgents: async () => {
    try {
      const agents = await fetchAgents();
      set({ agents: agents.length > 0 ? agents : ["random"], agentsError: null });
    } catch (e) {
      // Keep the `["random"]` default so the picker still works if `GET
      // /agents` is unreachable; just surface the error alongside it.
      set({ agentsError: e instanceof Error ? e.message : String(e) });
    }
  },

  startVsBot: async (seed, agent = "random") => {
    set({ status: "connecting", errorMessage: null });
    try {
      const res = await createRoom({
        seats: [{ kind: "human" }, { kind: "agent", name: agent }],
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
    // Guard against a second submission (a real double-click, or a repeat
    // tap while a slow reply is still on its way) landing before the first
    // one's outcome is known: the server would just reject it as no longer
    // legal, but with no visible round trip in between that reads as "my
    // first click did nothing" just as easily as a genuine bug would.
    if (get().pending) return;
    set({ pending: true });
    sendAction(socket, action);
  },

  leaveGame: () => {
    closeSocket();
    set({ roomId: null, mode: null, status: "idle", payload: null, errorMessage: null, eventLog: [], pending: false });
  },
}));
