// Turns the structured effect fields on catalog entries (see
// `data/README.md`'s effect-type vocabulary) into short, unambiguous
// plain-English sentences. Used to back the hover tooltips on card/wonder/
// token chips so a player never has to guess what a bare icon or enum name
// means (e.g. "destroy" -> "Destroy one of your opponent's raw material
// buildings", "shield" -> "+1 military shield").
//
// Kept separate from `catalogHelpers.ts` (which only maps ids to short
// display labels/colours) because these functions assemble full sentences
// from several fields at once.

import type { Catalog } from "../generated/Catalog";
import type { CardCatalogEntry } from "../generated/CardCatalogEntry";
import type { WonderCatalogEntry } from "../generated/WonderCatalogEntry";
import type { TokenCatalogEntry } from "../generated/TokenCatalogEntry";
import { CARD_TYPE_LABEL, cardById, resourceEntries } from "./catalogHelpers";

const SCIENCE_LABEL: Record<string, string> = {
  mortar: "Mortar",
  pendulum: "Pendulum",
  inkwell: "Inkwell",
  wheel: "Wheel",
  sundial: "Sundial",
  gyroscope: "Gyroscope",
  balance: "Balance",
};

const RESOURCE_GROUP_LABEL: Record<string, string> = {
  raw_material: "raw material",
  manufactured_good: "manufactured good",
};

const DISCOUNT_LABEL: Record<string, string> = {
  wonders: "Wonders",
  civilian_buildings: "Civilian buildings",
};

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function shieldLine(n: number): string {
  return `+${plural(n, "military shield")}`;
}

/** Lowercase just the first letter, for splicing a label into mid-sentence. */
function lower(label: string): string {
  return label.charAt(0).toLowerCase() + label.slice(1);
}

export function describeCoinsPerOwn([label, n]: [string, number]): string {
  return `+${plural(n, "coin")} for every ${lower(label)} you own.`;
}

export function describeCoinsByMajority([label, n]: [string, number]): string {
  return `Immediately gain ${plural(n, "coin")} per ${lower(label)}, counted for whichever player (you or your opponent) owns more.`;
}

export function describePointsByMajority([label, n]: [string, number]): string {
  return `At game end, ${plural(n, "victory point")} per ${lower(label)}, counted for whichever player owns more.`;
}

/** Every effect an age/guild card has, as short plain-English sentences. */
export function describeCardEffects(card: CardCatalogEntry, catalog?: Catalog): string[] {
  const lines: string[] = [];

  if (card.chain_from) {
    const from = catalog ? cardById(catalog, card.chain_from) : undefined;
    lines.push(`Free to build if you already own ${from?.name ?? card.chain_from}.`);
  }

  const produced = resourceEntries(card.produces);
  if (produced.length > 0) {
    lines.push(`Produces ${produced.map(([r, n]) => `${n} ${r}`).join(", ")}.`);
  }
  if (card.produces_choice) {
    lines.push(`Produces 1 ${RESOURCE_GROUP_LABEL[card.produces_choice]} resource of your choice.`);
  }
  if (card.fixed_trade.length > 0) {
    lines.push(
      `Buy ${card.fixed_trade.join(", ")} from the bank for 1 coin each, no matter how much your opponent produces.`,
    );
  }
  if (card.coins > 0) lines.push(`+${plural(card.coins, "coin")} when built.`);
  if (card.shields > 0) lines.push(shieldLine(card.shields));
  if (card.science) lines.push(`Grants the ${SCIENCE_LABEL[card.science]} science symbol.`);
  if (card.coins_per_own) lines.push(describeCoinsPerOwn(card.coins_per_own));
  if (card.coins_by_majority) lines.push(describeCoinsByMajority(card.coins_by_majority));
  if (card.points_by_majority) lines.push(describePointsByMajority(card.points_by_majority));
  if (card.victory_points > 0) lines.push(`Worth ${plural(card.victory_points, "victory point")}.`);
  if (card.chain_to) {
    const to = catalog ? cardById(catalog, card.chain_to) : undefined;
    lines.push(`Owning this lets you build ${to?.name ?? card.chain_to} for free.`);
  }

  return lines;
}

/** Every effect a wonder has, as short plain-English sentences. */
export function describeWonderEffects(wonder: WonderCatalogEntry): string[] {
  const lines: string[] = [];

  if (wonder.produces_choice) {
    lines.push(`Produces 1 ${RESOURCE_GROUP_LABEL[wonder.produces_choice]} resource of your choice.`);
  }
  if (wonder.coins > 0) lines.push(`+${plural(wonder.coins, "coin")} when built.`);
  if (wonder.opponent_loses_coins > 0) {
    lines.push(`Your opponent immediately loses ${plural(wonder.opponent_loses_coins, "coin")}.`);
  }
  if (wonder.shields > 0) lines.push(shieldLine(wonder.shields));
  if (wonder.destroy) {
    lines.push(`Destroy one of your opponent's ${lower(CARD_TYPE_LABEL[wonder.destroy])} buildings.`);
  }
  if (wonder.build_discarded_free) lines.push("Build one card from the discard pile for free.");
  if (wonder.choose_progress_token) {
    lines.push("Draw 3 progress tokens from the ones set aside at setup and keep one.");
  }
  if (wonder.play_again) lines.push("Take an extra turn immediately.");
  if (wonder.victory_points > 0) lines.push(`Worth ${plural(wonder.victory_points, "victory point")}.`);

  return lines;
}

/** Every effect a progress token has, as short plain-English sentences. */
export function describeTokenEffects(token: TokenCatalogEntry): string[] {
  const lines: string[] = [];

  if (token.coins > 0) lines.push(`+${plural(token.coins, "coin")} when taken.`);
  if (token.science) lines.push(`Grants the ${SCIENCE_LABEL[token.science]} science symbol.`);
  if (token.discount) {
    lines.push(`${DISCOUNT_LABEL[token.discount]} cost 2 fewer resources of your choice, for the rest of the game.`);
  }
  if (token.gain_trade_costs) {
    lines.push("Your opponent's trade payments come to you instead of the bank.");
  }
  if (token.shield_bonus) lines.push("+1 shield every time you build a military (red) building.");
  if (token.wonder_play_again) lines.push("Take an extra turn every time you build a wonder.");
  if (token.chain_build_coins > 0) {
    lines.push(`+${plural(token.chain_build_coins, "coin")} every time you build using a chain symbol.`);
  }
  if (token.vp_per_token > 0) {
    lines.push(`At game end, +${plural(token.vp_per_token, "victory point")} for every progress token you hold.`);
  }
  if (token.victory_points > 0) lines.push(`Worth ${plural(token.victory_points, "victory point")} at game end.`);

  return lines;
}
