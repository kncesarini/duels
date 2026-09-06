// Presentation of the server's `CostPlan`. Every number here already came
// from `duels_core::cost` (see `duels-server`'s `PlayerView`); these helpers
// only phrase it. No price, discount or affordability is computed client-side
// - if you find yourself adding arithmetic over rules concepts to this file,
// it belongs in `duels-core` and on the wire instead.

import type { CostPlan } from "../generated/CostPlan";
import type { PlayerView } from "../generated/PlayerView";
import type { Player } from "../generated/Player";
import type { Resource } from "../generated/Resource";

export const COIN = "¢";

export function coins(n: number): string {
  return `${n}${COIN}`;
}

export function seatIndex(p: Player): 0 | 1 {
  return p === "one" ? 0 : 1;
}

export function otherPlayer(p: Player): Player {
  return p === "one" ? "two" : "one";
}

export function slotPlan(view: PlayerView, slot: number): CostPlan | undefined {
  return view.slot_costs.find((s) => s.slot === slot)?.plan;
}

export function wonderPlan(view: PlayerView, wonder: string): CostPlan | undefined {
  return view.wonder_costs.find((w) => w.wonder === wonder)?.plan;
}

export function priceOf(view: PlayerView, resource: Resource): number {
  return view.trade_prices[resource];
}

export function productionOf(view: PlayerView, resource: Resource): number {
  return view.production[resource];
}

/** The badge that sits at the right of a cost row: text plus a kind that
 * decides its colour. Always carries a word or a number, never colour alone. */
export function costBadge(plan: CostPlan): { text: string; kind: "ok" | "bad" | "chain" } {
  if (plan.via_chain) return { text: "free", kind: "chain" };
  if (plan.coins === 0) return { text: `0${COIN} ✓`, kind: "ok" };
  return { text: coins(plan.coins), kind: plan.affordable ? "ok" : "bad" };
}

/** One short sentence per resource of a cost, for the tray's cost boxes. */
export function planLines(plan: CostPlan): Array<{ resource: Resource; count: number; note: string }> {
  return plan.lines.map((l) => ({
    resource: l.resource,
    count: l.required,
    note: l.bought > 0 ? `buy ${l.bought > 1 ? `${l.bought}×` : ""}${coins(l.unit_price)}` : coverNote(l),
  }));
}

function coverNote(l: CostPlan["lines"][number]): string {
  if (l.from_discount > 0 && l.from_choice > 0) return "rebate + choice ✓";
  if (l.from_discount > 0) return "rebate ✓";
  if (l.from_choice > 0) return "choice ✓";
  return "produced ✓";
}

/** The one-line "what was paid and why" used by the ticker and the log. */
export function paymentSummary(plan: CostPlan | undefined): string {
  if (!plan) return "";
  if (plan.via_chain) return "free via a chain symbol";
  if (plan.coins === 0) return "free (nothing to pay)";
  const bought = plan.lines.filter((l) => l.bought > 0);
  const parts: string[] = [];
  if (plan.coin_cost > 0) parts.push(`${coins(plan.coin_cost)} printed`);
  for (const l of bought) {
    parts.push(`${l.bought} ${l.resource} at ${coins(l.unit_price)}`);
  }
  return parts.length > 0 ? `paid ${coins(plan.coins)} (${parts.join(", ")})` : `paid ${coins(plan.coins)}`;
}

/** The long-form arithmetic shown when a log entry is expanded. */
export function paymentDetail(plan: CostPlan | undefined, before: number, after: number): string[] {
  if (!plan) return [];
  const out: string[] = [];
  if (plan.via_chain) {
    out.push("A chain symbol from an earlier building waived the whole cost.");
  } else {
    if (plan.coin_cost > 0) out.push(`Printed cost: ${coins(plan.coin_cost)} to the bank.`);
    for (const l of plan.lines) {
      const covered: string[] = [];
      if (l.produced > 0) covered.push(`${l.produced} produced`);
      if (l.from_choice > 0) covered.push(`${l.from_choice} from a flexible source`);
      if (l.from_discount > 0) covered.push(`${l.from_discount} from a cost rebate`);
      if (l.bought > 0) {
        covered.push(`${l.bought} bought at ${coins(l.unit_price)} each = ${coins(l.bought * l.unit_price)}`);
      }
      out.push(`${l.required} ${l.resource}: ${covered.join(", ")}.`);
    }
    if (plan.trade > 0) out.push(`Trade payments: ${coins(plan.trade)}.`);
  }
  out.push(`Coins ${before} → ${after}.`);
  return out;
}
