import { describe, expect, it } from "vitest";
import type { Breakdown } from "../generated/Breakdown";
import type { Observation } from "../generated/Observation";
import type { PlayerView } from "../generated/PlayerView";
import type { PublicPlayer } from "../generated/PublicPlayer";
import type { SlotView } from "../generated/SlotView";
import type { StepPayload } from "../generated/StepPayload";
import { buildEntries, exportText } from "./log";

const NAMES: [string, string] = ["You", "Opponent"];

function player(over: Partial<PublicPlayer> = {}): PublicPlayer {
  return {
    coins: 7,
    built: [],
    wonders: [],
    wonders_built: [],
    tokens: [],
    shields: 0,
    science: [0, 0, 0, 0, 0, 0, 0],
    pairs_awarded: [],
    ...over,
  };
}

function breakdown(total: number): Breakdown {
  return {
    civilian: 0,
    scientific: 0,
    commercial: 0,
    guilds: 0,
    wonders: 0,
    progress_tokens: 0,
    military: 0,
    coins: 0,
    total,
  };
}

function view(over: Partial<PlayerView> = {}): PlayerView {
  return {
    production: { wood: 1, clay: 0, stone: 0, glass: 0, papyrus: 0 },
    trade_prices: { wood: 2, clay: 2, stone: 2, glass: 2, papyrus: 3 },
    distinct_science: 0,
    vp_now: breakdown(0),
    discard_reward: 2,
    slot_costs: [],
    wonder_costs: [],
    ...over,
  };
}

const EMPTY_SLOT: SlotView = { state: "empty" };

function observation(over: Partial<Observation> = {}): Observation {
  return {
    phase: "turn",
    age: 1,
    current_player: "two",
    turn: 1,
    conflict: 0,
    loot_taken: [
      [false, false],
      [false, false],
    ],
    extra_turn: false,
    pending: null,
    last_card_taker: "one",
    players: [player(), player()],
    slots: Array.from({ length: 20 }, () => EMPTY_SLOT) as Observation["slots"],
    discard: [],
    wonder_fodder: [],
    board_tokens: [],
    set_aside_tokens: [],
    offered_wonders: [],
    undrafted_wonder_pool: [],
    draft_step: 8,
    draft_first: "one",
    unknown_slot_pool: [],
    hidden_guild_count: 0,
    result: null,
    ...over,
  };
}

function step(over: Partial<StepPayload> = {}): StepPayload {
  return {
    actor: "one",
    action: null,
    events: [],
    observation: observation(),
    views: [view(), view()],
    accessible_slots: [],
    ...over,
  };
}

describe("buildEntries", () => {
  it("turns one build into an entry with its payment, consequences and reveals", () => {
    const before = step({
      actor: "one",
      action: null,
      observation: observation({ turn: 4, players: [player({ coins: 7 }), player()] }),
      views: [
        view({
          slot_costs: [
            {
              slot: 3,
              card: "temple",
              plan: {
                lines: [
                  {
                    resource: "papyrus",
                    required: 1,
                    produced: 0,
                    from_choice: 0,
                    from_discount: 0,
                    bought: 1,
                    unit_price: 3,
                  },
                ],
                coin_cost: 0,
                coins: 3,
                trade: 3,
                via_chain: false,
                affordable: true,
              },
            },
          ],
        }),
        view(),
      ],
    });
    const built = step({
      actor: "one",
      action: { type: "Build", slot: 3 },
      events: [
        { type: "CardTaken", player: "one", slot: 3, card: "temple" },
        { type: "CoinsLost", player: "one", amount: 3, reason: "trade" },
        { type: "CardBuilt", player: "one", card: "temple", via_chain: false },
        { type: "SlotRevealed", slot: 9, card: "library" },
      ],
      observation: observation({ turn: 5, players: [player({ coins: 4 }), player()] }),
      views: [view({ vp_now: breakdown(3) }), view()],
    });

    const [entry] = buildEntries([before, built], null, NAMES);
    expect(entry.actor).toBe("one");
    expect(entry.verb).toBe("built");
    expect(entry.subject?.id).toBe("temple");
    expect(entry.turn).toBe(5);
    // The payment is read off the plan the tray showed *before* the move, so
    // the log and the tray can never tell different stories.
    expect(entry.payment).toBe("paid 3¢ (1 papyrus at 3¢)");
    expect(entry.detail[entry.detail.length - 1]).toBe("Coins 7 → 4.");
    // The victory-point delta is the difference between two server-scored
    // totals, not a client-side count.
    expect(entry.chips[0]).toEqual({ text: "+3 VP", kind: "plain" });
    expect(entry.reveals).toEqual(["library"]);
    // A construction payment is the payment line, never also a chip.
    expect(entry.chips.some((c) => c.text.includes("−"))).toBe(false);
  });

  it("adds a system entry when an age starts, and marks it a key event", () => {
    const entries = buildEntries(
      [
        step({ observation: observation({ age: 1, turn: 20 }) }),
        step({
          actor: "two",
          action: { type: "Build", slot: 0 },
          events: [
            { type: "CardTaken", player: "two", slot: 0, card: "altar" },
            { type: "CardBuilt", player: "two", card: "altar", via_chain: false },
            { type: "AgeEnded", age: 1 },
            { type: "AgeStarted", age: 2 },
          ],
          observation: observation({ age: 2, turn: 21 }),
        }),
      ],
      null,
      NAMES,
    );
    expect(entries).toHaveLength(2);
    expect(entries[1].text).toBe("Age II begins.");
    expect(entries[1].key).toBe(true);
    // The move itself is filed under the age it was made in, not the new one.
    expect(entries[0].age).toBe(1);
    expect(entries[1].age).toBe(2);
  });

  it("names a claimed progress token and treats it as a key event", () => {
    const [entry] = buildEntries(
      [
        step({
          actor: "one",
          action: { type: "ChooseProgressToken", token: "agriculture" },
          events: [{ type: "ProgressTokenTaken", player: "one", token: "agriculture", from_great_library: false }],
        }),
      ],
      null,
      NAMES,
    );
    expect(entry.verb).toBe("claimed");
    expect(entry.subject?.id).toBe("agriculture");
    expect(entry.key).toBe(true);
  });

  it("reads back as plain text, one line per entry", () => {
    const entries = buildEntries(
      [
        step({
          actor: "one",
          action: { type: "Discard", slot: 2 },
          events: [
            { type: "CardTaken", player: "one", slot: 2, card: "tavern" },
            { type: "CardDiscarded", player: "one", card: "tavern" },
            { type: "CoinsGained", player: "one", amount: 2, reason: "discarded_card" },
          ],
        }),
      ],
      null,
      NAMES,
    );
    const text = exportText(entries, NAMES);
    expect(text).toContain("You discarded tavern");
    expect(text).toContain("+2¢");
  });
});
