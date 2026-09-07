import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useGameStore } from "./store";

type Listener = (event: { data: string }) => void;

/** A controllable stand-in for the browser `WebSocket`, so the reconnect
 * logic in `store.ts` can be exercised without a real server. Every
 * `connectRoomSocket` call in the module under test produces one of these;
 * tests drive it via `triggerOpen`/`triggerClose`/`triggerMessage`. */
class FakeWebSocket {
  static readonly OPEN = 1;
  static readonly instances: FakeWebSocket[] = [];

  readyState = 0;
  sentCount = 0;
  private readonly listeners = new Map<string, Listener[]>();

  constructor(public readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, listener: Listener) {
    const list = this.listeners.get(type) ?? [];
    list.push(listener);
    this.listeners.set(type, list);
  }

  send() {
    this.sentCount += 1;
  }

  close() {
    this.triggerClose();
  }

  triggerOpen() {
    this.readyState = FakeWebSocket.OPEN;
    for (const cb of this.listeners.get("open") ?? []) cb({ data: "" });
  }

  triggerClose() {
    this.readyState = 3;
    for (const cb of this.listeners.get("close") ?? []) cb({ data: "" });
  }

  triggerMessage(data: unknown) {
    for (const cb of this.listeners.get("message") ?? []) cb({ data: JSON.stringify(data) });
  }
}

describe("useGameStore", () => {
  beforeEach(() => {
    useGameStore.setState({
      catalog: null,
      catalogError: null,
      roomId: null,
      mode: null,
      status: "idle",
      latest: null,
      errorMessage: null,
      history: [],
      entries: [],
      displayedIndex: -1,
      playback: null,
      pending: false,
    });
  });

  it("starts idle with no room", () => {
    const s = useGameStore.getState();
    expect(s.status).toBe("idle");
    expect(s.roomId).toBeNull();
    expect(s.latest).toBeNull();
  });

  it("leaveGame resets connection state back to idle", () => {
    useGameStore.setState({
      roomId: "room-1",
      mode: "bot",
      status: "connected",
      errorMessage: "boom",
    });

    useGameStore.getState().leaveGame();

    const s = useGameStore.getState();
    expect(s.roomId).toBeNull();
    expect(s.mode).toBeNull();
    expect(s.status).toBe("idle");
    expect(s.errorMessage).toBeNull();
  });

  it("submitAction is a no-op when there is no open connection", () => {
    // No socket has been opened in this test, so this must not throw.
    expect(() => useGameStore.getState().submitAction({ type: "Build", slot: 0 })).not.toThrow();
  });
});

describe("useGameStore reconnect behavior", () => {
  const realWebSocket = globalThis.WebSocket;
  const realFetch = globalThis.fetch;

  beforeEach(() => {
    FakeWebSocket.instances.length = 0;
    vi.useFakeTimers();
    globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ room_id: "room-1" }),
    })) as unknown as typeof fetch;
    useGameStore.setState({
      catalog: null,
      catalogError: null,
      roomId: null,
      mode: null,
      status: "idle",
      latest: null,
      errorMessage: null,
      history: [],
      entries: [],
      displayedIndex: -1,
      playback: null,
      pending: false,
    });
  });

  afterEach(() => {
    useGameStore.getState().leaveGame();
    vi.useRealTimers();
    globalThis.WebSocket = realWebSocket;
    globalThis.fetch = realFetch;
  });

  // This is the regression test for the "click a wonder and nothing
  // happens" bug: an unexpected WebSocket drop (a network hiccup, a
  // container restart, ...) used to leave the player staring at a frozen
  // board with no feedback, and every further click silently swallowed by
  // `submitAction`'s `readyState` check. The store must instead notice the
  // drop and transparently reconnect to the same room.
  it("re-establishes the connection to the same room after an unexpected drop", async () => {
    await useGameStore.getState().startVsBot();
    expect(FakeWebSocket.instances).toHaveLength(1);
    const first = FakeWebSocket.instances[0];
    first.triggerOpen();
    expect(useGameStore.getState().status).toBe("connected");

    first.triggerClose();
    expect(useGameStore.getState().status).toBe("closed");
    expect(useGameStore.getState().roomId).toBe("room-1");

    await vi.advanceTimersByTimeAsync(1000);

    expect(FakeWebSocket.instances).toHaveLength(2);
    const second = FakeWebSocket.instances[1];
    expect(second.url).toContain("room-1");
    expect(useGameStore.getState().status).toBe("reconnecting");

    second.triggerOpen();
    expect(useGameStore.getState().status).toBe("connected");
  });

  it("does not let a stale socket's close event reconnect a room the player already left", async () => {
    await useGameStore.getState().startVsBot();
    const first = FakeWebSocket.instances[0];
    first.triggerOpen();

    useGameStore.getState().leaveGame();
    // A belated close from the abandoned socket (this fires once already,
    // synchronously, inside `leaveGame()` -> `closeSocket()`; trigger it
    // again to simulate a redundant duplicate) must not resurrect the room.
    first.triggerClose();

    await vi.advanceTimersByTimeAsync(10_000);

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(useGameStore.getState().status).toBe("idle");
    expect(useGameStore.getState().roomId).toBeNull();
  });

  // Regression test for the other half of the "click a wonder and nothing
  // happens" bug: with no feedback between clicking and the server's reply,
  // a slow round trip (this app talks to `duels-server` over a real
  // connection, so nothing guarantees it's instant) is indistinguishable
  // from a dead client. `submitAction` must mark the action pending
  // immediately, refuse a second one while the first is still in flight, and
  // clear the flag as soon as a reply - success or `Error` - comes back.
  it("marks an action pending until the reply arrives and refuses a second submission meanwhile", async () => {
    await useGameStore.getState().startVsBot();
    const ws = FakeWebSocket.instances[0];
    ws.triggerOpen();

    useGameStore.getState().submitAction({ type: "PickWonder", wonder: "the-pyramids" });
    expect(ws.sentCount).toBe(1);
    expect(useGameStore.getState().pending).toBe(true);

    // A second click before the reply arrives must not reach the socket.
    useGameStore.getState().submitAction({ type: "PickWonder", wonder: "the-sphinx" });
    expect(ws.sentCount).toBe(1);

    ws.triggerMessage({ type: "Error", message: "that action is not currently legal" });
    expect(useGameStore.getState().pending).toBe(false);

    // Now that the reply landed, a fresh click is allowed through again.
    useGameStore.getState().submitAction({ type: "PickWonder", wonder: "the-sphinx" });
    expect(ws.sentCount).toBe(2);
  });
});

describe("advanced mode: analysis and the flag bundle", () => {
  const realFetch = globalThis.fetch;

  const ANALYSIS = {
    room_id: "room-1",
    turn: 9,
    age: 1,
    current_player: "one",
    game_over: false,
    value: 15.9,
    win_probability: 0.583,
    actions: [
      { action: { type: "Build", slot: 14 }, value: -6.07, win_probability: 0.564 },
      { action: { type: "Discard", slot: 14 }, value: -11.2, win_probability: 0.481 },
    ],
    eval_generation: "sci=0.50/dead=0.00",
  };
  const EXPORT = {
    room_id: "room-1",
    seed: 424242,
    moves: [{ type: "PickWonder", wonder: "piraeus" }, { type: "Discard", slot: 19 }],
  };

  /** Route each endpoint to its payload, or to a rejection for the ones named
   * in `fail`, so a partial outage can be exercised. */
  function mockApi(fail: string[] = []) {
    globalThis.fetch = vi.fn(async (url: unknown) => {
      const u = String(url);
      const which = u.endsWith("/analysis") ? "analysis" : u.endsWith("/export") ? "export" : "other";
      if (fail.includes(which)) return { ok: false, status: 503, statusText: "unavailable", json: async () => ({}) };
      return { ok: true, json: async () => (which === "analysis" ? ANALYSIS : EXPORT) };
    }) as unknown as typeof fetch;
  }

  beforeEach(() => {
    useGameStore.setState({ roomId: "room-1", analysis: null, analysisError: null });
  });

  afterEach(() => {
    globalThis.fetch = realFetch;
    useGameStore.setState({ roomId: null, analysis: null, analysisError: null });
  });

  it("does nothing outside a game rather than fetching an analysis of nothing", async () => {
    mockApi();
    useGameStore.setState({ roomId: null });
    await useGameStore.getState().loadAnalysis();
    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect(useGameStore.getState().analysis).toBeNull();
  });

  it("stores the analysis and clears any previous error", async () => {
    mockApi();
    useGameStore.setState({ analysisError: "an old failure" });
    await useGameStore.getState().loadAnalysis();
    const s = useGameStore.getState();
    expect(s.analysis?.win_probability).toBe(0.583);
    expect(s.analysisError).toBeNull();
  });

  // A failed refresh mid-game must not blank the panel: stale numbers plus a
  // label beat numbers that silently vanish.
  it("keeps the last good analysis on screen when a refresh fails", async () => {
    mockApi();
    await useGameStore.getState().loadAnalysis();
    mockApi(["analysis"]);
    await useGameStore.getState().loadAnalysis();
    const s = useGameStore.getState();
    expect(s.analysis?.win_probability).toBe(0.583);
    expect(s.analysisError).not.toBeNull();
  });

  it("builds a bundle carrying the seed, the moves, every score and the notes", async () => {
    mockApi();
    const json = await useGameStore.getState().buildFlagBundle("the military term is asleep here");
    const bundle = JSON.parse(json) as Record<string, unknown>;
    expect(bundle.seed).toBe(424242);
    expect(bundle.moves).toEqual(EXPORT.moves);
    expect(bundle.current_win_probability).toBe(0.583);
    expect(bundle.action_win_probabilities).toEqual(ANALYSIS.actions);
    expect(bundle.eval_generation).toBe("sci=0.50/dead=0.00");
    expect(bundle.notes).toBe("the military term is asleep here");
    expect(bundle.captured).toEqual({
      room_id: "room-1",
      turn: 9,
      age: 1,
      current_player: "one",
      value: 15.9,
    });
  });

  // The reasoning is the expensive part of a flag - it was typed by hand. A
  // failed analysis refresh must fall back to the numbers already on screen
  // rather than throw it away; the export is what makes the bundle useful and
  // it succeeded.
  it("falls back to the analysis on screen if the refresh fails while flagging", async () => {
    mockApi();
    await useGameStore.getState().loadAnalysis();
    mockApi(["analysis"]);
    const json = await useGameStore.getState().buildFlagBundle("still worth recording");
    expect((JSON.parse(json) as { current_win_probability: number }).current_win_probability).toBe(0.583);
  });

  it("gives up rather than emitting a bundle with no numbers in it at all", async () => {
    mockApi(["analysis"]);
    await expect(useGameStore.getState().buildFlagBundle("nothing to attach")).rejects.toThrow();
  });

  it("refuses to build a bundle outside a game", async () => {
    mockApi();
    useGameStore.setState({ roomId: null });
    await expect(useGameStore.getState().buildFlagBundle("x")).rejects.toThrow("not in a game");
  });
});
