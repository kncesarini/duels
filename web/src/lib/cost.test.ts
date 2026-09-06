import { describe, expect, it } from "vitest";
import type { CostPlan } from "../generated/CostPlan";
import { costBadge, paymentDetail, paymentSummary, planLines } from "./cost";

function plan(over: Partial<CostPlan> = {}): CostPlan {
  return {
    lines: [],
    coin_cost: 0,
    coins: 0,
    trade: 0,
    via_chain: false,
    affordable: true,
    ...over,
  };
}

describe("costBadge", () => {
  it("marks a chained build as free, in its own colour", () => {
    expect(costBadge(plan({ via_chain: true }))).toEqual({ text: "free", kind: "chain" });
  });

  it("shows a tick when nothing has to be paid", () => {
    expect(costBadge(plan())).toEqual({ text: "0¢ ✓", kind: "ok" });
  });

  it("carries the number as well as the colour, both ways", () => {
    expect(costBadge(plan({ coins: 4 }))).toEqual({ text: "4¢", kind: "ok" });
    expect(costBadge(plan({ coins: 8, affordable: false }))).toEqual({ text: "8¢", kind: "bad" });
  });
});

describe("planLines", () => {
  it("says how each resource is covered", () => {
    const p = plan({
      coins: 3,
      trade: 3,
      lines: [
        { resource: "wood", required: 1, produced: 1, from_choice: 0, from_discount: 0, bought: 0, unit_price: 2 },
        { resource: "papyrus", required: 1, produced: 0, from_choice: 0, from_discount: 0, bought: 1, unit_price: 3 },
      ],
    });
    expect(planLines(p)).toEqual([
      { resource: "wood", count: 1, note: "produced ✓" },
      { resource: "papyrus", count: 1, note: "buy 3¢" },
    ]);
  });

  it("names the source when a flexible card or a rebate covered the unit", () => {
    const p = plan({
      lines: [
        { resource: "stone", required: 1, produced: 0, from_choice: 1, from_discount: 0, bought: 0, unit_price: 4 },
        { resource: "glass", required: 1, produced: 0, from_choice: 0, from_discount: 1, bought: 0, unit_price: 5 },
      ],
    });
    expect(planLines(p).map((l) => l.note)).toEqual(["choice ✓", "rebate ✓"]);
  });
});

describe("paymentSummary", () => {
  it("spells out the arithmetic behind the total", () => {
    const p = plan({
      coins: 5,
      trade: 3,
      coin_cost: 2,
      lines: [{ resource: "wood", required: 1, produced: 0, from_choice: 0, from_discount: 0, bought: 1, unit_price: 3 }],
    });
    expect(paymentSummary(p)).toBe("paid 5¢ (2¢ printed, 1 wood at 3¢)");
  });

  it("distinguishes a chained build from one that merely cost nothing", () => {
    expect(paymentSummary(plan({ via_chain: true }))).toBe("free via a chain symbol");
    expect(paymentSummary(plan())).toBe("free (nothing to pay)");
  });
});

describe("paymentDetail", () => {
  it("ends with the treasury before and after", () => {
    const lines = paymentDetail(plan({ coins: 4, trade: 4 }), 9, 5);
    expect(lines[lines.length - 1]).toBe("Coins 9 → 5.");
  });
});
