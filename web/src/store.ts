// The single piece of client-side state for the whole app.
//
// It never computes a rule. It remembers what the server sent, decides *when*
// to show each part of it (the move-playback sequence), accumulates the step
// history the log and review mode read, and forwards `Action`s the server
// told us were legal.

import { create } from "zustand";

import type { Action } from "./generated/Action";
import type { AnalysisPayload } from "./generated/AnalysisPayload";
import type { Catalog } from "./generated/Catalog";
import type { Player } from "./generated/Player";
import type { StatePayload } from "./generated/StatePayload";
import type { StepPayload } from "./generated/StepPayload";
import { buildEntries, type LogEntry } from "./lib/log";
import { seatIndex } from "./lib/cost";
import {
  applySettings,
  DEFAULT_SETTINGS,
  loadSettings,
  saveSettings,
  SPEED_FACTOR,
  type Settings,
} from "./lib/settings";
import {
  connectRoomSocket,
  createRoom,
  fetchAgents,
  fetchAnalysis,
  fetchCatalog,
  fetchExport,
  sendAction,
} from "./lib/api";

export type ConnectionStatus = "idle" | "connecting" | "connected" | "reconnecting" | "closed" | "error";

/** Which part of the take → resolve → reveal → hand-over sequence is on
 * screen. `thinking` is the deliberate minimum hold before an opponent's move
 * is shown, so a five-millisecond agent reply is paced like a human one. */
export type PlaybackPhase = "thinking" | "take" | "resolve" | "reveal" | "settling";

export interface Playback {
  /** Index into `history` of the step being played. */
  stepIndex: number;
  phase: PlaybackPhase;
  /** How many of the step's consequence chips have appeared so far. */
  effects: number;
}

const MAX_RECONNECT_ATTEMPTS = 5;

/** Base durations in ms, before the animation-speed multiplier. */
const T = { think: 400, take: 600, effect: 200, reveal: 150, settle: 400, handover: 1200 } as const;

interface GameStore {
  catalog: Catalog | null;
  catalogError: string | null;

  /** Agent names `POST /rooms` will accept, from `GET /agents`. `["random"]`
   * until `loadAgents` resolves, so the opponent picker always has at least
   * the one opponent the e2e suite relies on. */
  agents: string[];
  agentsError: string | null;

  roomId: string | null;
  mode: "bot" | "hotseat" | null;
  status: ConnectionStatus;
  /** The newest snapshot. Always authoritative; the table may be showing an
   * earlier step while a move plays back. */
  latest: StatePayload | null;
  errorMessage: string | null;
  pending: boolean;
  /** When the last action was submitted, for the thinking indicator's
   * elapsed-time readout. */
  pendingSince: number | null;

  /** Every action applied to this room, oldest first. */
  history: StepPayload[];
  entries: LogEntry[];
  /** Index into `history` of the position the table is drawing. `-1` before
   * anything has happened. */
  displayedIndex: number;
  playback: Playback | null;
  /** Non-null while an earlier position is being reviewed. */
  reviewIndex: number | null;
  /** True during the hot-seat hand-over pause. */
  handover: boolean;

  /** Which player's costs every card shows. Follows the player to move
   * unless overridden, and the override is dropped at each hand-over. */
  lensOverride: Player | null;
  /** True while `Alt` is held, which peeks the other player's costs. */
  peek: boolean;
  selectedSlot: number | null;

  settings: Settings;

  /** Advanced mode only: `duels-eval`'s read of the live position, refreshed
   * whenever the position changes. Null while off, in flight, or failed. */
  analysis: AnalysisPayload | null;
  analysisError: string | null;

  loadCatalog: () => Promise<void>;
  loadAgents: () => Promise<void>;
  startVsBot: (seed?: number, agent?: string) => Promise<void>;
  startHotSeat: (seed?: number) => Promise<void>;
  submitAction: (action: Action) => void;
  leaveGame: () => void;

  selectSlot: (slot: number | null) => void;
  setLens: (p: Player | null) => void;
  setPeek: (on: boolean) => void;
  review: (index: number | null) => void;
  stepReview: (delta: number) => void;
  updateSettings: (patch: Partial<Settings>) => void;
  loadAnalysis: () => Promise<void>;
  /** Build the flag-a-position bundle: the export (seed + full move list), the
   * analysis as it currently reads, and the typed reasoning. Returns the JSON
   * text to copy or download. */
  buildFlagBundle: (notes: string) => Promise<string>;
}

/** The JSON a flagged position is exported as. Not a generated type: its
 * parts come from generated ones (`ExportPayload`, `AnalysisPayload`), but the
 * bundle itself is assembled here, and its shape is the contract with whatever
 * later reads a directory of these. */
export interface FlagBundle {
  /** The room's setup seed. `duels_server::room::replay(seed, moves)`
   * reconstructs the exact position this was flagged in. */
  seed: number;
  moves: Action[];
  /** What the position read as when it was flagged. */
  current_win_probability: number;
  /** The same, for every action legal at the time. */
  action_win_probabilities: Array<{ action: Action; value: number; win_probability: number }>;
  /** `duels_eval::Config::params_string()` of the generation that produced
   * those numbers, so a later round of tuning cannot be mistaken for this one. */
  eval_generation: string;
  /** Free text: what is wrong with the evaluation here, and why. */
  notes: string;
  /** Context that costs nothing to record and saves reconstructing it. */
  captured: { room_id: string; turn: number; age: number; current_player: Player; value: number };
}

let socket: WebSocket | null = null;
let socketEpoch = 0;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let playTimer: ReturnType<typeof setTimeout> | null = null;
let playQueue: number[] = [];
let pendingSince = 0;
/** Guards against a slow analysis response landing after a newer one (or after
 * the game was left) and overwriting it. */
let analysisEpoch = 0;

function clearReconnectTimer() {
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function clearPlayTimer() {
  if (playTimer !== null) {
    clearTimeout(playTimer);
    playTimer = null;
  }
}

function closeSocket() {
  clearReconnectTimer();
  clearPlayTimer();
  playQueue = [];
  socketEpoch += 1;
  if (socket) {
    socket.close();
    socket = null;
  }
}

/** Which seat this browser is playing. Against an agent that is the one human
 * seat; in hot-seat both seats are this browser's, so it is whoever is on
 * move. */
export function localSeat(payload: StatePayload): Player {
  const humanSeats = payload.seats
    .map((s, i) => (s.kind === "human" ? i : -1))
    .filter((i) => i >= 0);
  if (humanSeats.length === 1) return humanSeats[0] === 0 ? "one" : "two";
  return payload.observation.current_player;
}

/** The seat the opponent occupies, from this browser's point of view. */
export function opponentSeat(payload: StatePayload): Player {
  return localSeat(payload) === "one" ? "two" : "one";
}

function speed(): number {
  return SPEED_FACTOR[useGameStore.getState().settings.animation];
}

function chipCountFor(stepIndex: number): number {
  const { entries } = useGameStore.getState();
  return entries.find((e) => e.stepIndex === stepIndex && e.text === null)?.chips.length ?? 0;
}

function revealCountFor(stepIndex: number): number {
  const { history } = useGameStore.getState();
  return history[stepIndex]?.events.filter((e) => e.type === "SlotRevealed").length ?? 0;
}

/** Play the queued steps one after another. Every step gets the same ordered
 * treatment regardless of how fast the engine produced it. */
function playNext() {
  clearPlayTimer();
  const next = playQueue.shift();
  const state = useGameStore.getState();
  if (next === undefined) {
    useGameStore.setState({ playback: null, displayedIndex: state.history.length - 1 });
    maybeHandover();
    return;
  }
  const s = speed();
  useGameStore.setState({
    playback: { stepIndex: next, phase: "take", effects: 0 },
    displayedIndex: next,
    selectedSlot: null,
  });
  playTimer = setTimeout(() => resolveEffects(next, 0), T.take * s);
}

function resolveEffects(stepIndex: number, shown: number) {
  const s = speed();
  const total = chipCountFor(stepIndex);
  if (shown >= total) {
    useGameStore.setState({ playback: { stepIndex, phase: "reveal", effects: total } });
    const reveals = revealCountFor(stepIndex);
    playTimer = setTimeout(() => settle(stepIndex), Math.max(1, reveals) * T.reveal * s);
    return;
  }
  useGameStore.setState({ playback: { stepIndex, phase: "resolve", effects: shown + 1 } });
  playTimer = setTimeout(() => resolveEffects(stepIndex, shown + 1), T.effect * s);
}

function settle(stepIndex: number) {
  const s = speed();
  useGameStore.setState({
    playback: { stepIndex, phase: "settling", effects: chipCountFor(stepIndex) },
  });
  playTimer = setTimeout(playNext, T.settle * s);
}

/** In hot-seat, pause on a "the other player is up" band instead of the
 * thinking indicator: there is no hidden information to screen, only pacing. */
function maybeHandover() {
  const st = useGameStore.getState();
  if (st.mode !== "hotseat" || !st.latest || st.latest.observation.result) return;
  const last = st.history[st.history.length - 1];
  if (!last || last.actor === st.latest.observation.current_player) return;
  useGameStore.setState({ handover: true });
  playTimer = setTimeout(() => useGameStore.setState({ handover: false }), T.handover * speed());
}

function onState(payload: StatePayload) {
  const st = useGameStore.getState();
  const seatNames = seatNamesFor(payload, st.mode);
  const startIndex = payload.replay ? 0 : st.history.length;
  const history = payload.replay ? payload.steps : [...st.history, ...payload.steps];
  const entries = buildEntries(history, st.catalog, seatNames);

  // The lens follows whoever is about to act, unless this browser pinned it
  // during the current player's turn.
  const changedPlayer = st.latest?.observation.current_player !== payload.observation.current_player;

  useGameStore.setState({
    latest: payload,
    status: "connected",
    pending: false,
    pendingSince: payload.replay ? null : st.pendingSince,
    history,
    entries,
    lensOverride: changedPlayer ? null : st.lensOverride,
  });

  // The position changed, so any analysis on screen is now about a position
  // that is no longer live. Refetch (advanced mode only) - after your own
  // move, after the opponent's, and on the replay a reconnect delivers.
  if (st.settings.advanced) void useGameStore.getState().loadAnalysis();

  if (payload.replay || payload.steps.length === 0) {
    clearPlayTimer();
    playQueue = [];
    useGameStore.setState({ playback: null, displayedIndex: history.length - 1 });
    return;
  }

  playQueue = payload.steps.map((_, i) => startIndex + i);

  // Hold the thinking indicator so an agent that answered instantly is paced
  // like one that thought about it.
  const opponent = opponentSeat(payload);
  const agentMoved =
    st.mode === "bot" && payload.steps.some((s) => s.actor === opponent);
  const elapsed = Date.now() - pendingSince;
  const hold = agentMoved ? Math.max(0, T.think * speed() - elapsed) : 0;
  if (hold > 0) {
    useGameStore.setState({
      playback: { stepIndex: playQueue[0], phase: "thinking", effects: 0 },
    });
    clearPlayTimer();
    playTimer = setTimeout(playNext, hold);
  } else {
    playNext();
  }
}

/** The names the log and prompts use for each seat. */
export function seatNamesFor(payload: StatePayload, mode: "bot" | "hotseat" | null): [string, string] {
  if (mode === "hotseat") return ["Player 1", "Player 2"];
  const me = localSeat(payload);
  const names: [string, string] = ["", ""];
  names[seatIndex(me)] = "You";
  names[seatIndex(me) === 0 ? 1 : 0] = "Opponent";
  return names;
}

function connect(roomId: string, mode: "bot" | "hotseat", reconnectAttempt = 0) {
  clearReconnectTimer();
  const epoch = ++socketEpoch;
  if (socket) {
    socket.close();
    socket = null;
  }
  if (reconnectAttempt === 0) {
    clearPlayTimer();
    playQueue = [];
    useGameStore.setState({
      roomId,
      mode,
      status: "connecting",
      errorMessage: null,
      latest: null,
      history: [],
      entries: [],
      displayedIndex: -1,
      playback: null,
      reviewIndex: null,
      handover: false,
      selectedSlot: null,
      lensOverride: null,
      pending: false,
      analysis: null,
      analysisError: null,
    });
    analysisEpoch += 1;
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
          errorMessage:
            "Lost connection to the game server and could not reconnect. Reload the page to try again.",
        });
        return;
      }
      reconnectTimer = setTimeout(() => connect(roomId, mode, reconnectAttempt + 1), 500 * (reconnectAttempt + 1));
    },
    onError: () => {
      if (epoch !== socketEpoch) return;
      useGameStore.setState({ status: "error", errorMessage: "WebSocket connection error" });
    },
    onMessage: (msg) => {
      if (epoch !== socketEpoch) return;
      if (msg.type === "State") {
        onState(msg as { type: "State" } & StatePayload);
      } else if (msg.type === "Error") {
        useGameStore.setState({ errorMessage: msg.message, pending: false });
      }
    },
  });
}

const initialSettings = typeof window === "undefined" ? DEFAULT_SETTINGS : loadSettings();
if (typeof document !== "undefined") applySettings(initialSettings);

export const useGameStore = create<GameStore>((set, get) => ({
  catalog: null,
  catalogError: null,

  agents: ["random"],
  agentsError: null,

  roomId: null,
  mode: null,
  status: "idle",
  latest: null,
  errorMessage: null,
  pending: false,
  pendingSince: null,

  history: [],
  entries: [],
  displayedIndex: -1,
  playback: null,
  reviewIndex: null,
  handover: false,

  lensOverride: null,
  peek: false,
  selectedSlot: null,

  settings: initialSettings,

  analysis: null,
  analysisError: null,

  loadCatalog: async () => {
    if (get().catalog) return;
    try {
      const catalog = await fetchCatalog();
      set({ catalog, catalogError: null });
      // Entries built before the catalog arrived show raw ids; rebuild once.
      const st = get();
      if (st.history.length > 0 && st.latest) {
        set({ entries: buildEntries(st.history, catalog, seatNamesFor(st.latest, st.mode)) });
      }
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
    // Guard against a second submission (a real double-click, or a repeat tap
    // while a slow reply is still on its way) landing before the first one's
    // outcome is known.
    if (get().pending || get().playback) return;
    pendingSince = Date.now();
    set({ pending: true, pendingSince, selectedSlot: null, reviewIndex: null });
    sendAction(socket, action);
  },

  leaveGame: () => {
    closeSocket();
    analysisEpoch += 1;
    set({
      analysis: null,
      analysisError: null,
      roomId: null,
      mode: null,
      status: "idle",
      latest: null,
      errorMessage: null,
      history: [],
      entries: [],
      displayedIndex: -1,
      playback: null,
      reviewIndex: null,
      handover: false,
      selectedSlot: null,
      pending: false,
    });
  },

  selectSlot: (slot) => set({ selectedSlot: slot, reviewIndex: null }),
  setLens: (p) => set({ lensOverride: p }),
  setPeek: (on) => set({ peek: on }),
  review: (index) => set({ reviewIndex: index }),
  stepReview: (delta) => {
    const st = get();
    if (st.reviewIndex === null) return;
    const next = Math.max(0, Math.min(st.history.length - 1, st.reviewIndex + delta));
    set({ reviewIndex: next });
  },
  updateSettings: (patch) => {
    const next = { ...get().settings, ...patch };
    saveSettings(next);
    applySettings(next);
    set({ settings: next });
    // Turning advanced mode on mid-game should show the current position
    // straight away rather than waiting for the next move; turning it off
    // should drop what is on screen rather than leaving it to go stale.
    if (patch.advanced === true) void get().loadAnalysis();
    if (patch.advanced === false) {
      analysisEpoch += 1;
      set({ analysis: null, analysisError: null });
    }
  },

  loadAnalysis: async () => {
    const { roomId } = get();
    if (!roomId) return;
    const epoch = ++analysisEpoch;
    try {
      const analysis = await fetchAnalysis(roomId);
      if (epoch !== analysisEpoch) return;
      set({ analysis, analysisError: null });
    } catch (e) {
      if (epoch !== analysisEpoch) return;
      // Leave the last good analysis on screen and label it: a failed refresh
      // is much less confusing than numbers silently vanishing mid-game.
      set({ analysisError: e instanceof Error ? e.message : String(e) });
    }
  },

  buildFlagBundle: async (notes) => {
    const { roomId, analysis } = get();
    if (!roomId) throw new Error("not in a game");
    // Both fetched fresh, so the move list and the numbers describe the same
    // position rather than one being however stale the last refresh left it.
    // They cannot drift apart between the two calls: the flag control is only
    // reachable while this browser is the seat on move, and nothing else
    // advances a room. The cached analysis is the fallback for a failed
    // refresh - a bundle with slightly older numbers still replays exactly,
    // and is far better than losing the reasoning already typed.
    const exported = await fetchExport(roomId);
    const current = await fetchAnalysis(roomId).catch((e: unknown) => {
      if (analysis) return analysis;
      throw e;
    });
    const bundle: FlagBundle = {
      seed: exported.seed,
      moves: exported.moves,
      current_win_probability: current.win_probability,
      action_win_probabilities: current.actions.map((a) => ({
        action: a.action,
        value: a.value,
        win_probability: a.win_probability,
      })),
      eval_generation: current.eval_generation,
      notes,
      captured: {
        room_id: exported.room_id,
        turn: current.turn,
        age: current.age,
        current_player: current.current_player,
        value: current.value,
      },
    };
    return JSON.stringify(bundle, null, 2);
  },
}));
