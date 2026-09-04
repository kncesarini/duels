import type { Observation } from "../generated/Observation";
import type { Catalog } from "../generated/Catalog";
import { TokenChip } from "./CardChip";

// The base game always builds exactly 7 of the 8 drafted wonders (the
// eighth drafted wonder can never be constructed) - a fixed, printed rule of
// the physical game, not something derived from any particular match, so
// it's fine as a display constant here (see duels_core::state::MAX_WONDERS_BUILT).
const MAX_WONDERS_BUILT = 7;

interface BoardProps {
  observation: Observation;
  catalog: Catalog;
  onOpenDiscard: () => void;
}

export default function Board({ observation, catalog, onOpenDiscard }: BoardProps) {
  const { military } = catalog;
  const cap = military.capital_distance;
  const conflict = observation.conflict;
  const pawnPercent = 50 + (conflict / cap) * 50;
  const faceDownCount = observation.slots.filter((s) => s.state === "face_down").length;
  const wondersBuilt = observation.players[0].wonders_built.length + observation.players[1].wonders_built.length;

  return (
    <div className="rounded-xl border border-stone-300 bg-white p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-sm">
        <div>
          <span className="font-semibold">Age {observation.age}</span> - turn {observation.turn}
          {observation.result === null && <span className="ml-2 text-stone-500">({phaseLabel(observation)})</span>}
        </div>
        <div className="flex items-center gap-3">
          <span>
            Wonders built: {wondersBuilt}/{MAX_WONDERS_BUILT}
          </span>
          <span>Face-down cards left: {faceDownCount}</span>
          <button
            type="button"
            onClick={onOpenDiscard}
            className="rounded bg-stone-200 px-2 py-1 hover:bg-stone-300"
          >
            Discard pile ({observation.discard.length})
          </button>
        </div>
      </div>

      <div className="mb-2 text-xs font-semibold text-stone-500">Military track</div>
      <div className="relative mb-1 h-8 rounded-full bg-gradient-to-r from-sky-200 via-stone-200 to-red-200">
        {/* loot tokens */}
        {military.loot.map(([distance], i) => (
          <div key={`loot-left-${i}`}>
            <div
              className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 text-[10px]"
              style={{ left: `${50 - (distance / cap) * 50}%` }}
              title={`Loot: ${military.loot[i][1]} coins forfeited`}
            >
              {observation.loot_taken[0][i] ? "·" : "$"}
            </div>
            <div
              className="absolute top-1/2 -translate-y-1/2 text-[10px]"
              style={{ left: `${50 + (distance / cap) * 50}%` }}
              title={`Loot: ${military.loot[i][1]} coins forfeited`}
            >
              {observation.loot_taken[1][i] ? "·" : "$"}
            </div>
          </div>
        ))}
        {/* pawn */}
        <div
          className="absolute top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-black bg-yellow-400 text-center text-xs font-bold leading-6"
          style={{ left: `${pawnPercent}%` }}
          title={`Conflict: ${conflict}`}
        >
          {conflict}
        </div>
      </div>
      <div className="mb-3 flex justify-between text-[10px] text-stone-500">
        <span>◀ Player Two capital</span>
        <span>centre</span>
        <span>Player One capital ▶</span>
      </div>

      <div className="text-xs font-semibold text-stone-500">Progress tokens available</div>
      <div className="mt-1 flex flex-wrap gap-2">
        {observation.board_tokens.map((id) => {
          const token = catalog.tokens.find((t) => t.id === id);
          return token ? <TokenChip key={id} token={token} disabled /> : null;
        })}
        {observation.board_tokens.length === 0 && <span className="text-xs text-stone-400">none left</span>}
      </div>
    </div>
  );
}

function phaseLabel(obs: Observation): string {
  switch (obs.phase) {
    case "wonder_draft":
      return "wonder draft";
    case "turn":
      return obs.pending ? `resolving: ${obs.pending.type.replace(/_/g, " ")}` : `${obs.current_player}'s turn`;
    case "choose_first_player":
      return "choosing first player";
    case "game_over":
      return "game over";
  }
}
