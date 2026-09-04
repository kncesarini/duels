import { useMemo } from "react";
import type { Observation } from "../generated/Observation";
import type { Catalog } from "../generated/Catalog";
import { CardChip, FaceDownChip } from "./CardChip";
import { cardById } from "../lib/catalogHelpers";

const CARD_W = 80;
const CARD_H = 112;
const ROW_GAP = 16;

interface StructureProps {
  observation: Observation;
  catalog: Catalog;
  actionableSlots: Set<number>;
  onSlotClick: (slot: number) => void;
}

export default function Structure({ observation, catalog, actionableSlots, onSlotClick }: StructureProps) {
  const layout = catalog.layouts[observation.age - 1];

  const { positions, width, height } = useMemo(() => {
    const positions = layout.positions;
    const rows = positions.map((p) => p[0]);
    const cols = positions.map((p) => p[1]);
    const minRow = Math.min(...rows);
    const minCol = Math.min(...cols);
    const maxCol = Math.max(...cols);
    const maxRow = Math.max(...rows);
    const pixelPositions = positions.map(([row, col]) => ({
      x: (col - minCol) * (CARD_W / 2),
      y: (row - minRow) * (CARD_H + ROW_GAP),
    }));
    return {
      positions: pixelPositions,
      width: (maxCol - minCol) * (CARD_W / 2) + CARD_W,
      height: (maxRow - minRow) * (CARD_H + ROW_GAP) + CARD_H,
    };
  }, [layout]);

  if (observation.slots.every((s) => s.state === "empty")) {
    return null;
  }

  return (
    <div className="relative mx-auto" style={{ width, height }}>
      {observation.slots.map((slot, i) => {
        const pos = positions[i];
        if (!pos) return null;
        if (slot.state === "empty") return null;
        return (
          <div
            key={i}
            className="absolute"
            style={{ left: pos.x, top: pos.y }}
            data-testid={`slot-${i}`}
          >
            {slot.state === "face_down" && <FaceDownChip />}
            {slot.state === "face_up" &&
              (() => {
                const card = cardById(catalog, slot.card);
                if (!card) return null;
                const clickable = actionableSlots.has(i);
                return (
                  <CardChip
                    card={card}
                    catalog={catalog}
                    onClick={clickable ? () => onSlotClick(i) : undefined}
                    highlight={clickable}
                  />
                );
              })()}
          </div>
        );
      })}
    </div>
  );
}
