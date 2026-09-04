import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CardChip } from "./CardChip";
import type { CardCatalogEntry } from "../generated/CardCatalogEntry";

const sampleCard: CardCatalogEntry = {
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

describe("CardChip", () => {
  it("renders the card's name and production", () => {
    render(<CardChip card={sampleCard} />);
    expect(screen.getByText("Lumber Yard")).toBeInTheDocument();
    expect(screen.getByText("+1W")).toBeInTheDocument();
  });

  it("is only clickable when an onClick handler is supplied", () => {
    const onClick = vi.fn();
    const { rerender } = render(<CardChip card={sampleCard} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).not.toHaveBeenCalled();

    rerender(<CardChip card={sampleCard} onClick={onClick} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
