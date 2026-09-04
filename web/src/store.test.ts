import { beforeEach, describe, expect, it } from "vitest";
import { useGameStore } from "./store";

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
