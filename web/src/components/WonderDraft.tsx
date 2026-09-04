import type { Observation } from "../generated/Observation";
import type { Catalog } from "../generated/Catalog";
import type { Action } from "../generated/Action";
import { WonderChip } from "./CardChip";
import { wonderById } from "../lib/catalogHelpers";

interface WonderDraftProps {
  observation: Observation;
  catalog: Catalog;
  legal: Action[];
  onSubmit: (action: Action) => void;
}

export default function WonderDraft({ observation, catalog, legal, onSubmit }: WonderDraftProps) {
  const pickable = new Set(
    legal.filter((a): a is Action & { type: "PickWonder" } => a.type === "PickWonder").map((a) => a.wonder),
  );

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4">
      <div className="text-center">
        <h2 className="text-2xl font-bold">Wonder draft</h2>
        <p className="text-sm text-stone-600">
          Pick {pickable.size > 0 ? "" : "-"} draft step {observation.draft_step + 1} of 8. {observation.undrafted_wonder_pool.length}{" "}
          wonder{observation.undrafted_wonder_pool.length === 1 ? "" : "s"} not yet revealed.
        </p>
      </div>

      <div className="flex flex-wrap justify-center gap-3">
        {observation.offered_wonders.map((id) => {
          const wonder = wonderById(catalog, id);
          if (!wonder) return null;
          const canPick = pickable.has(id);
          return (
            <WonderChip
              key={id}
              wonder={wonder}
              onClick={canPick ? () => onSubmit({ type: "PickWonder", wonder: id }) : undefined}
              disabled={!canPick}
            />
          );
        })}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {(["one", "two"] as const).map((p, i) => (
          <div key={p} className="rounded-lg border border-stone-300 bg-white p-3">
            <div className="mb-2 text-sm font-semibold">Player {i === 0 ? "One" : "Two"}'s wonders</div>
            <div className="flex flex-wrap gap-1.5">
              {observation.players[i].wonders.map((w) => {
                const wonder = wonderById(catalog, w);
                return wonder ? <WonderChip key={w} wonder={wonder} disabled /> : null;
              })}
              {observation.players[i].wonders.length === 0 && (
                <span className="text-xs text-stone-400">none yet</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
