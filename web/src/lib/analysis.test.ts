import { describe, expect, it } from "vitest";
import type { Action } from "../generated/Action";
import type { ActionAnalysis } from "../generated/ActionAnalysis";
import type { AnalysisPayload } from "../generated/AnalysisPayload";
import { actionKey, bestForSlot, byAction, deltaPct, ranked, tone, winPct } from "./analysis";

function scored(action: Action, win_probability: number): ActionAnalysis {
  return { action, value: 0, win_probability };
}

function payload(actions: ActionAnalysis[], win_probability = 0.5): AnalysisPayload {
  return {
    room_id: "room-1",
    turn: 12,
    age: 2,
    current_player: "one",
    game_over: false,
    value: 0,
    win_probability,
    actions,
    eval_generation: "test",
  };
}

describe("winPct / deltaPct", () => {
  it("keeps a decimal, because adjacent options often differ by well under a point", () => {
    expect(winPct(0.6234)).toBe("62.3%");
    expect(winPct(0.6244)).toBe("62.4%");
  });

  it("signs a delta explicitly, so a small loss is not mistaken for a small gain", () => {
    expect(deltaPct(0.62, 0.5)).toBe("+12.0");
    expect(deltaPct(0.44, 0.5)).toBe("−6.0");
    expect(deltaPct(0.5, 0.5)).toBe("±0.0");
  });
});

describe("tone", () => {
  it("bands a probability so it reads without the number", () => {
    expect(tone(0.9)).toBe("good");
    expect(tone(0.6)).toBe("good");
    expect(tone(0.5)).toBe("even");
    expect(tone(0.4)).toBe("bad");
    expect(tone(0.05)).toBe("bad");
  });
});

describe("actionKey / byAction", () => {
  it("looks an action's score up by the action a component already holds", () => {
    const build: Action = { type: "Build", slot: 3 };
    const m = byAction(payload([scored(build, 0.7)]));
    expect(m.get(actionKey({ type: "Build", slot: 3 }))?.win_probability).toBe(0.7);
  });

  it("does not confuse two actions on the same slot", () => {
    const m = byAction(
      payload([
        scored({ type: "Build", slot: 3 }, 0.7),
        scored({ type: "Discard", slot: 3 }, 0.2),
        scored({ type: "BuildWonder", slot: 3, wonder: "piraeus" }, 0.55),
      ]),
    );
    expect(m.get(actionKey({ type: "Build", slot: 3 }))?.win_probability).toBe(0.7);
    expect(m.get(actionKey({ type: "Discard", slot: 3 }))?.win_probability).toBe(0.2);
    expect(m.get(actionKey({ type: "BuildWonder", slot: 3, wonder: "piraeus" }))?.win_probability).toBe(0.55);
  });

  it("is empty rather than throwing before the first analysis arrives", () => {
    expect(byAction(null).size).toBe(0);
    expect(bestForSlot(null, 0)).toBeNull();
    expect(ranked(null)).toEqual([]);
  });
});

describe("bestForSlot", () => {
  const p = payload([
    scored({ type: "Build", slot: 3 }, 0.41),
    scored({ type: "Discard", slot: 3 }, 0.62),
    scored({ type: "BuildWonder", slot: 3, wonder: "piraeus" }, 0.55),
    scored({ type: "Build", slot: 9 }, 0.71),
  ]);

  it("is the best of every way of taking that slot, not just the build", () => {
    expect(bestForSlot(p, 3)?.action).toEqual({ type: "Discard", slot: 3 });
    expect(bestForSlot(p, 9)?.win_probability).toBe(0.71);
  });

  it("is null for a slot with nothing legal on it", () => {
    expect(bestForSlot(p, 17)).toBeNull();
  });

  it("ignores actions that carry no slot at all", () => {
    const tokens = payload([scored({ type: "ChooseProgressToken", token: "law" }, 0.8)]);
    expect(bestForSlot(tokens, 0)).toBeNull();
  });
});

describe("ranked", () => {
  it("orders best first — an analysis is read as a ranking", () => {
    const p = payload([
      scored({ type: "Build", slot: 1 }, 0.3),
      scored({ type: "Build", slot: 2 }, 0.8),
      scored({ type: "Build", slot: 3 }, 0.5),
    ]);
    expect(ranked(p).map((a) => a.win_probability)).toEqual([0.8, 0.5, 0.3]);
  });

  it("does not reorder the payload it was given", () => {
    const actions = [scored({ type: "Build", slot: 1 }, 0.3), scored({ type: "Build", slot: 2 }, 0.8)];
    const p = payload(actions);
    ranked(p);
    expect(p.actions.map((a) => a.win_probability)).toEqual([0.3, 0.8]);
  });
});
