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
      payload: null,
      errorMessage: null,
      eventLog: [],
      pending: false,
    });
  });

  it("starts idle with no room", () => {
    const s = useGameStore.getState();
    expect(s.status).toBe("idle");
    expect(s.roomId).toBeNull();
    expect(s.payload).toBeNull();
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
      payload: null,
      errorMessage: null,
      eventLog: [],
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
