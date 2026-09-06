// Phone tier only. The one tier where full state is not simultaneously
// visible, so the numbers a decision actually needs - both players' coins,
// running victory points, distinct science and the military position - stay
// on screen above the structure whatever the bottom sheet is showing.

import type { Catalog } from "../generated/Catalog";
import type { Observation } from "../generated/Observation";
import type { Player } from "../generated/Player";
import type { PlayerView } from "../generated/PlayerView";
import { seatIndex } from "../lib/cost";

interface Props {
  observation: Observation;
  views: [PlayerView, PlayerView];
  catalog: Catalog;
  seatNames: [string, string];
  bottom: Player;
}

export default function SummaryBar({ observation, views, catalog, seatNames, bottom }: Props) {
  const bottomIdx = seatIndex(bottom);
  const topIdx = bottomIdx === 0 ? 1 : 0;
  const cap = catalog.military.capital_distance;
  const towardsBottom = bottom === "one" ? -observation.conflict : observation.conflict;

  const side = (idx: number, right: boolean) => (
    <div className={`side ${right ? "r" : ""}`}>
      <span className="nm" style={{ color: idx === bottomIdx ? "var(--you)" : "var(--opp)" }}>
        {seatNames[idx]}
      </span>
      <div className="stats">
        <span>{observation.players[idx].coins}¢</span>
        <span>{views[idx].vp_now.total} VP</span>
        <span>{views[idx].distinct_science}/6</span>
      </div>
    </div>
  );

  return (
    <div className="summary" aria-label="Both players at a glance">
      {side(topIdx, false)}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
        <div className="miltrack">
          {Array.from({ length: cap * 2 + 1 }, (_, i) => (
            <i key={i} className={i - cap === towardsBottom ? "pw" : ""} />
          ))}
        </div>
        <span style={{ fontSize: 9, color: "var(--mute)" }}>
          military{" "}
          {observation.conflict === 0
            ? "level"
            : `${seatNames[observation.conflict > 0 ? 0 : 1]} +${Math.abs(observation.conflict)}`}
        </span>
      </div>
      {side(bottomIdx, true)}
    </div>
  );
}
