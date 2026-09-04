import { describe, expect, it } from "vitest";
import { describeCardEffects, describeTokenEffects, describeWonderEffects } from "./effectText";
import type { WonderCatalogEntry } from "../generated/WonderCatalogEntry";
import type { CardCatalogEntry } from "../generated/CardCatalogEntry";
import type { TokenCatalogEntry } from "../generated/TokenCatalogEntry";
import type { Catalog } from "../generated/Catalog";

const baseWonder: WonderCatalogEntry = {
  id: "circus-maximus",
  name: "Circus Maximus",
  coin_cost: 0,
  resource_cost: { wood: 1, clay: 0, stone: 2, glass: 1, papyrus: 0 },
  victory_points: 3,
  shields: 1,
  coins: 0,
  opponent_loses_coins: 0,
  produces_choice: null,
  play_again: false,
  destroy: null,
  build_discarded_free: false,
  choose_progress_token: false,
};

describe("describeWonderEffects", () => {
  it("states exactly which building type a destroy effect targets", () => {
    const wonder: WonderCatalogEntry = { ...baseWonder, destroy: "manufactured_good" };
    const lines = describeWonderEffects(wonder);
    expect(lines.some((l) => /destroy.*manufactured good/i.test(l))).toBe(true);
    // Never a bare, unexplained "destroy" with no target.
    expect(lines.every((l) => l !== "destroy")).toBe(true);
  });

  it("states the shield amount, not just that a shield exists", () => {
    const lines = describeWonderEffects(baseWonder);
    expect(lines).toContain("+1 military shield");
  });

  it("pluralises multiple shields", () => {
    const lines = describeWonderEffects({ ...baseWonder, shields: 2 });
    expect(lines).toContain("+2 military shields");
  });

  it("describes coin-taking and opponent coin loss", () => {
    const wonder: WonderCatalogEntry = {
      ...baseWonder,
      shields: 0,
      coins: 3,
      opponent_loses_coins: 3,
      play_again: true,
    };
    const lines = describeWonderEffects(wonder);
    expect(lines.some((l) => l.includes("+3 coins"))).toBe(true);
    expect(lines.some((l) => /opponent immediately loses 3 coins/i.test(l))).toBe(true);
    expect(lines.some((l) => /extra turn/i.test(l))).toBe(true);
  });

  it("says which private pool the Great Library draws from", () => {
    const lines = describeWonderEffects({ ...baseWonder, shields: 0, choose_progress_token: true });
    expect(lines.some((l) => /set aside at setup/i.test(l))).toBe(true);
  });
});

const baseCard: CardCatalogEntry = {
  id: "lumber-yard",
  name: "Lumber Yard",
  age: 1,
  kind: "raw_material",
  coin_cost: 0,
  resource_cost: { wood: 0, clay: 0, stone: 0, glass: 0, papyrus: 0 },
  chain_from: null,
  chain_to: null,
  produces: { wood: 1, clay: 0, stone: 0, glass: 0, papyrus: 0 },
  produces_choice: null,
  victory_points: 0,
  science: null,
  shields: 0,
  coins: 0,
  fixed_trade: [],
  coins_per_own: null,
  coins_by_majority: null,
  points_by_majority: null,
  is_guild: false,
};

describe("describeCardEffects", () => {
  it("describes a guild's majority-based coin and point effects in plain English", () => {
    const guild: CardCatalogEntry = {
      ...baseCard,
      id: "merchants-guild",
      kind: "guild",
      is_guild: true,
      coins_by_majority: ["raw material cards", 1],
      points_by_majority: ["raw material cards", 1],
    };
    const lines = describeCardEffects(guild);
    expect(lines.some((l) => /gain 1 coin per raw material card.*whichever player/i.test(l))).toBe(true);
    expect(lines.some((l) => /game end.*1 victory point per raw material card/i.test(l))).toBe(true);
  });

  it("names the linked card instead of just noting a chain exists", () => {
    const card: CardCatalogEntry = { ...baseCard, chain_to: "theater" };
    const catalog = { cards: [{ ...baseCard, id: "theater", name: "Theater" }] } as unknown as Catalog;
    const lines = describeCardEffects(card, catalog);
    expect(lines.some((l) => l.includes("Theater"))).toBe(true);
  });
});

const baseToken: TokenCatalogEntry = {
  id: "strategy",
  name: "Strategy",
  coins: 0,
  victory_points: 0,
  vp_per_token: 0,
  science: null,
  discount: null,
  gain_trade_costs: false,
  shield_bonus: false,
  wonder_play_again: false,
  chain_build_coins: 0,
};

describe("describeTokenEffects", () => {
  it("spells out the shield bonus token instead of leaving it implicit", () => {
    const lines = describeTokenEffects({ ...baseToken, shield_bonus: true });
    expect(lines.some((l) => /shield.*military.*building/i.test(l))).toBe(true);
  });
});
