// The icon set: one inline SVG sprite mounted once, plus tiny components
// that reference it. Purely presentational - nothing here knows a rule.
//
// Every icon is paired with a text label or an adjacent number wherever it
// appears (see the accessibility notes in the design spec, 9.4): the sprite
// exists so type/resource/science identity is *fast* to read, never so it is
// the only way to read it.

import type { CardType } from "../generated/CardType";
import type { Resource } from "../generated/Resource";
import type { Science } from "../generated/Science";

export const RESOURCES: Resource[] = ["wood", "stone", "clay", "glass", "papyrus"];

/** Fixed display order for the science row, matching the physical board. */
export const SCIENCES: Science[] = [
  "wheel",
  "mortar",
  "sundial",
  "gyroscope",
  "pendulum",
  "inkwell",
  "balance",
];

export const SCIENCE_LABEL: Record<Science, string> = {
  mortar: "Mortar",
  pendulum: "Pendulum",
  inkwell: "Inkwell",
  wheel: "Wheel",
  sundial: "Sundial",
  gyroscope: "Gyroscope",
  balance: "Balance",
};

export const RESOURCE_LABEL: Record<Resource, string> = {
  wood: "wood",
  stone: "stone",
  clay: "clay",
  glass: "glass",
  papyrus: "papyrus",
};

/** The seven card colours, in the fixed order strips and score tables use. */
export const CARD_TYPES: CardType[] = [
  "raw_material",
  "manufactured_good",
  "civilian",
  "scientific",
  "commercial",
  "military",
  "guild",
];

/** The glyph id for each card type, so colour is never the only cue. */
export const TYPE_GLYPH: Record<CardType, string> = {
  raw_material: "g-raw",
  manufactured_good: "g-man",
  civilian: "g-civ",
  scientific: "g-sci",
  commercial: "g-com",
  military: "g-mil",
  guild: "g-gld",
};

/** A distinct hatch per type, layered under `[data-marks=on]` for players who
 * cannot rely on hue at all. */
export const TYPE_MARK: Record<CardType, string> = {
  raw_material: "repeating-linear-gradient(45deg,#0000 0 3px,#ffffff44 3px 5px)",
  manufactured_good: "repeating-linear-gradient(-45deg,#0000 0 3px,#ffffff44 3px 5px)",
  civilian: "repeating-linear-gradient(90deg,#0000 0 4px,#ffffff44 4px 6px)",
  scientific: "radial-gradient(#ffffff55 1.1px,#0000 1.2px) 0 0/5px 5px",
  commercial: "repeating-linear-gradient(0deg,#0000 0 4px,#00000033 4px 6px)",
  military: "repeating-linear-gradient(135deg,#0000 0 2px,#ffffff55 2px 4px)",
  guild: "radial-gradient(#ffffff55 1.6px,#0000 1.7px) 0 0/7px 7px",
};
