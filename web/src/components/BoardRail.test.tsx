// Regression test for a reported bug: the military track's "already taken"
// loot indicator was rendered on the wrong side of the track. The root cause
// was a swapped ternary picking which player's `loot_taken` entry a given
// track cell reads (see the fix in BoardRail.tsx). This test pins the
// correct side/perspective mapping so it can't silently invert again.

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { Catalog } from "../generated/Catalog";
import type { MilitaryCatalog } from "../generated/MilitaryCatalog";
import type { Observation } from "../generated/Observation";
import type { PublicPlayer } from "../generated/PublicPlayer";
import type { SlotView } from "../generated/SlotView";
import BoardRail from "./BoardRail";

afterEach(cleanup);

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
    hidden_guild_slots: 0,
    result: null,
    ...over,
  };
}

const military: MilitaryCatalog = {
  capital_distance: 6,
  loot: [
    [2, 2],
    [5, 5],
  ],
  victory_points_by_distance: [0, 0, 2, 5, 5, 10, 10],
};

const catalog: Catalog = {
  cards: [],
  wonders: [],
  tokens: [],
  military,
  layouts: [
    { rows: [] } as unknown as Catalog["layouts"][0],
    { rows: [] } as unknown as Catalog["layouts"][0],
    { rows: [] } as unknown as Catalog["layouts"][0],
  ],
  science_order: [],
};

/** The `.cell` at track position `d` is at array index `d + capital_distance`
 * — the loop that builds them runs `d` from `-cap` to `cap` in order. */
function cellAt(container: HTMLElement, d: number): Element {
  const cells = container.querySelectorAll(".track .cells > .cell");
  return cells[d + military.capital_distance];
}

describe("BoardRail's military track", () => {
  it("shows the taken loot marker on the side the pushing player collected it, not the mirror side, from Player One's own perspective", () => {
    // Player One pushed the pawn 4 steps towards Player Two and collected the
    // distance-2 loot token on that side.
    const obs = observation({
      conflict: 4,
      loot_taken: [
        [true, false],
        [false, false],
      ],
    });
    const { container } = render(
      <BoardRail
        observation={obs}
        catalog={catalog}
        bottom="one"
        seatNames={["You", "Opponent"]}
        ghosted={null}
      />,
    );

    // Player Two's capital is at the top when bottom="one", so the loot on
    // the side approaching it is at d = -2.
    expect(cellAt(container, -2).className).toMatch(/\bloot\b/);
    expect(cellAt(container, -2).className).toMatch(/\btaken\b/);

    // Its mirror on the opposite side (approaching Player One's own capital)
    // must NOT read as taken — that loot token was never touched.
    expect(cellAt(container, 2).className).toMatch(/\bloot\b/);
    expect(cellAt(container, 2).className).not.toMatch(/\btaken\b/);
  });

  it("keeps the same side/perspective mapping when the viewer flips (spectating as Player Two)", () => {
    // Identical game fact as above — Player One pushed toward Player Two and
    // collected that side's loot — viewed with Player Two drawn at the
    // bottom instead.
    const obs = observation({
      conflict: 4,
      loot_taken: [
        [true, false],
        [false, false],
      ],
    });
    const { container } = render(
      <BoardRail
        observation={obs}
        catalog={catalog}
        bottom="two"
        seatNames={["Opponent", "You"]}
        ghosted={null}
      />,
    );

    // Now Player Two's capital is at the bottom, so the collected loot is on
    // the d = +2 side.
    expect(cellAt(container, 2).className).toMatch(/\bloot\b/);
    expect(cellAt(container, 2).className).toMatch(/\btaken\b/);
    expect(cellAt(container, -2).className).toMatch(/\bloot\b/);
    expect(cellAt(container, -2).className).not.toMatch(/\btaken\b/);
  });
});
