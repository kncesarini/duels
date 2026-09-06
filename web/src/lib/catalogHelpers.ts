// Small, purely presentational lookups over the static `Catalog` the server
// sends from `GET /catalog`. Nothing here computes a rule or a cost - it
// only turns ids into display strings/colours.

import type { Catalog } from "../generated/Catalog";
import type { CardCatalogEntry } from "../generated/CardCatalogEntry";
import type { WonderCatalogEntry } from "../generated/WonderCatalogEntry";
import type { TokenCatalogEntry } from "../generated/TokenCatalogEntry";
import type { CardType } from "../generated/CardType";
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

/** A card's display name, falling back to its id if the catalog is missing it. */
export function cardName(catalog: Catalog | null, id: string): string {
  return (catalog && cardById(catalog, id)?.name) ?? id;
}

export function wonderName(catalog: Catalog | null, id: string): string {
  return (catalog && wonderById(catalog, id)?.name) ?? id;
}

export function tokenName(catalog: Catalog | null, id: string): string {
  return (catalog && tokenById(catalog, id)?.name) ?? id;
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

/** The CSS custom property carrying each card colour (see `index.css`). Type
 * is never colour alone: every place this is used also prints the type's
 * glyph and, outside the card face itself, its name. */
export function typeColorVar(kind: CardType): string {
  return `var(--${kind})`;
}

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
