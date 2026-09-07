// Advanced (analysis) mode: reading `GET /rooms/:id/analysis` for display.
//
// Nothing here computes an evaluation, a legality or a cost - the server does
// all of that in `duels-eval`. This is lookup, formatting and ordering only,
// exactly like every other module in `lib/`.

import type { Action } from "../generated/Action";
import type { ActionAnalysis } from "../generated/ActionAnalysis";
import type { AnalysisPayload } from "../generated/AnalysisPayload";
import type { Catalog } from "../generated/Catalog";
import type { Observation } from "../generated/Observation";
import { cardName, tokenName, wonderName } from "./catalogHelpers";

/** A win probability as a percentage, for a badge that has to be readable at a
 * glance while scanning a dozen of them. One decimal, because two adjacent
 * options often differ by well under a point and rounding them to the same
 * integer would hide exactly the comparison this mode exists to make. */
export function winPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

/** The signed change in win probability an action would make, in percentage
 * points, against the position's current reading. */
export function deltaPct(action: number, current: number): string {
  const d = (action - current) * 100;
  const sign = d > 0 ? "+" : d < 0 ? "−" : "±";
  return `${sign}${Math.abs(d).toFixed(1)}`;
}

/** Colour band for a win probability: what "good" looks like without having to
 * read the number. */
export function tone(p: number): "good" | "even" | "bad" {
  if (p >= 0.6) return "good";
  if (p <= 0.4) return "bad";
  return "even";
}

/** A stable identity for an action, so an `ActionAnalysis` can be looked up by
 * the action a component already holds. `Action` is a tagged union of plain
 * scalars, so its JSON encoding is a serviceable key - and it is exactly the
 * encoding the two sides already agree on. */
export function actionKey(a: Action): string {
  return JSON.stringify(a);
}

/** Index an analysis by `actionKey`. */
export function byAction(analysis: AnalysisPayload | null): Map<string, ActionAnalysis> {
  const m = new Map<string, ActionAnalysis>();
  for (const a of analysis?.actions ?? []) m.set(actionKey(a.action), a);
  return m;
}

/** The best win probability reachable from one structure slot, over every
 * action that takes the card in it (build it, discard it, or spend it on any
 * of up to four wonders).
 *
 * The slot badge shows this rather than one chosen action's number: a slot is
 * what the player is scanning, and "the best this card can do for me" is the
 * comparison being made. The per-action split is one click away in the tray. */
export function bestForSlot(
  analysis: AnalysisPayload | null,
  slot: number,
): ActionAnalysis | null {
  let best: ActionAnalysis | null = null;
  for (const a of analysis?.actions ?? []) {
    if (!("slot" in a.action) || a.action.slot !== slot) continue;
    if (!best || a.win_probability > best.win_probability) best = a;
  }
  return best;
}

/** Every action, best first. The panel's list order - an analysis is read as a
 * ranking, not in `legal_actions` order. */
export function ranked(analysis: AnalysisPayload | null): ActionAnalysis[] {
  return [...(analysis?.actions ?? [])].sort((a, b) => b.win_probability - a.win_probability);
}

/** A short human label for an action, e.g. "Build Aqueduct (slot 7)". Uses the
 * catalog for names and the observation for what is in a slot, both of which
 * the client already has. */
export function describeAction(
  action: Action,
  catalog: Catalog | null,
  observation: Observation,
): string {
  const inSlot = (slot: number) => {
    const view = observation.slots[slot];
    return view && view.state === "face_up" ? cardName(catalog, view.card) : `slot ${slot}`;
  };
  switch (action.type) {
    case "Build":
      return `Build ${inSlot(action.slot)}`;
    case "Discard":
      return `Discard ${inSlot(action.slot)}`;
    case "BuildWonder":
      return `${wonderName(catalog, action.wonder)} ← ${inSlot(action.slot)}`;
    case "PickWonder":
      return `Draft ${wonderName(catalog, action.wonder)}`;
    case "ChooseProgressToken":
      return `Take ${tokenName(catalog, action.token)}`;
    case "ChooseGreatLibraryToken":
      return `Great Library: ${tokenName(catalog, action.token)}`;
    case "MausoleumBuild":
      return `Mausoleum: ${cardName(catalog, action.card)}`;
    case "DestroyOpponentCard":
      return `Destroy ${cardName(catalog, action.card)}`;
    case "ChooseFirstPlayer":
      return `${action.player === "one" ? "Player 1" : "Player 2"} starts the age`;
    default:
      return JSON.stringify(action);
  }
}
