// Small, purely presentational lookups over the static `Catalog` the server
// sends from `GET /catalog`. Nothing here computes a rule or a cost - it
// only turns ids into display strings/colours.

import type { Catalog } from "../generated/Catalog";
import type { CardCatalogEntry } from "../generated/CardCatalogEntry";
import type { WonderCatalogEntry } from "../generated/WonderCatalogEntry";
import type { TokenCatalogEntry } from "../generated/TokenCatalogEntry";
import type { CardType } from "../generated/CardType";
import type { Science } from "../generated/Science";
import type { ResourceAmounts } from "../generated/ResourceAmounts";

export function cardById(catalog: Catalog, id: string): CardCatalogEntry | undefined {
  return catalog.cards.find((c) => c.id === id);
}

export function wonderById(catalog: Catalog, id: string): WonderCatalogEntry | undefined {
  return catalog.wonders.find((w) => w.id === id);
}

export function tokenById(catalog: Catalog, id: string): TokenCatalogEntry | undefined {
  return catalog.tokens.find((t) => t.id === id);
}

export const CARD_TYPE_LABEL: Record<CardType, string> = {
  raw_material: "Raw material",
  manufactured_good: "Manufactured good",
  civilian: "Civilian",
  scientific: "Scientific",
  commercial: "Commercial",
  military: "Military",
  guild: "Guild",
};

/** Tailwind classes for a card colour, approximating the physical card backs. */
export const CARD_TYPE_COLOR: Record<CardType, string> = {
  raw_material: "bg-amber-800 text-amber-50 border-amber-950",
  manufactured_good: "bg-stone-400 text-stone-900 border-stone-600",
  civilian: "bg-sky-700 text-sky-50 border-sky-950",
  scientific: "bg-emerald-700 text-emerald-50 border-emerald-950",
  commercial: "bg-yellow-400 text-yellow-950 border-yellow-600",
  military: "bg-red-700 text-red-50 border-red-950",
  guild: "bg-purple-800 text-purple-50 border-purple-950",
};

export const SCIENCE_SYMBOL: Record<Science, string> = {
  mortar: "⚖", // scales-ish stand-in
  pendulum: "⏱",
  inkwell: "✒",
  wheel: "⚙",
  sundial: "◔",
  gyroscope: "ἰ",
  balance: "⚜",
};

export const RESOURCE_SYMBOL: Record<keyof ResourceAmounts, string> = {
  wood: "W",
  clay: "C",
  stone: "S",
  glass: "G",
  papyrus: "P",
};

export function resourceEntries(amounts: ResourceAmounts): Array<[keyof ResourceAmounts, number]> {
  return (Object.keys(amounts) as Array<keyof ResourceAmounts>)
    .map((k) => [k, amounts[k]] as [keyof ResourceAmounts, number])
    .filter(([, n]) => n > 0);
}
