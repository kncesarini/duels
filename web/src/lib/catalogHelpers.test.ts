import { describe, expect, it } from "vitest";
import { resourceEntries } from "./catalogHelpers";

describe("resourceEntries", () => {
  it("drops resources with zero units", () => {
    const amounts = { wood: 2, clay: 0, stone: 0, glass: 1, papyrus: 0 };
    expect(resourceEntries(amounts)).toEqual([
      ["wood", 2],
      ["glass", 1],
    ]);
  });

  it("returns nothing for an all-zero cost", () => {
    expect(resourceEntries({ wood: 0, clay: 0, stone: 0, glass: 0, papyrus: 0 })).toEqual([]);
  });
});
