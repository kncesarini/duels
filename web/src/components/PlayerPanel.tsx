import { useMemo } from "react";
import type { PublicPlayer } from "../generated/PublicPlayer";
import type { Catalog } from "../generated/Catalog";
import type { Science } from "../generated/Science";
import type { CardType } from "../generated/CardType";
import { CardChip, WonderChip } from "./CardChip";
import {
  CARD_TYPE_LABEL,
  RESOURCE_SYMBOL,
  SCIENCE_SYMBOL,
  cardById,
  wonderById,
} from "../lib/catalogHelpers";
import type { ResourceAmounts } from "../generated/ResourceAmounts";

const CARD_TYPE_ORDER: CardType[] = [
  "raw_material",
  "manufactured_good",
  "civilian",
  "scientific",
  "commercial",
  "military",
  "guild",
];

// Index order matches `duels_core::data::Science` (mortar, pendulum,
// inkwell, wheel, sundial, gyroscope, balance) - `PublicPlayer.science` is a
// plain `[u8; 7]` in that same order.
const SCIENCE_ORDER: Science[] = ["mortar", "pendulum", "inkwell", "wheel", "sundial", "gyroscope", "balance"];

interface PlayerPanelProps {
  label: string;
  player: PublicPlayer;
  catalog: Catalog;
  active: boolean;
  militaryLeader: boolean;
}

export default function PlayerPanel({ label, player, catalog, active, militaryLeader }: PlayerPanelProps) {
  const byType = useMemo(() => {
    const groups = new Map<CardType, string[]>();
    for (const id of player.built) {
      const card = cardById(catalog, id);
      if (!card) continue;
      const list = groups.get(card.kind) ?? [];
      list.push(id);
      groups.set(card.kind, list);
    }
    return groups;
  }, [player.built, catalog]);

  const production = useMemo(() => sumProduction(player.built, catalog), [player.built, catalog]);
  const unbuiltWonders = player.wonders.filter((w) => !player.wonders_built.includes(w));

  return (
    <div
      className={[
        "flex-1 rounded-xl border-2 p-4",
        active ? "border-amber-500 bg-amber-50" : "border-stone-300 bg-white",
      ].join(" ")}
    >
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-lg font-bold">
          {label} {active && <span className="text-amber-600">(on move)</span>}
        </h2>
        <div className="flex items-center gap-2 text-sm">
          {militaryLeader && <span title="Military leader">⚔ leading</span>}
          <span className="rounded bg-stone-800 px-2 py-0.5 font-semibold text-white">{player.coins}c</span>
        </div>
      </div>

      <div className="mb-3 flex flex-wrap gap-2 text-xs">
        <span className="font-semibold text-stone-500">Production:</span>
        {Object.entries(production)
          .filter(([, n]) => n > 0)
          .map(([r, n]) => (
            <span key={r} className="rounded bg-stone-200 px-1.5 py-0.5">
              {n}
              {RESOURCE_SYMBOL[r as keyof ResourceAmounts]}
            </span>
          ))}
        {Object.values(production).every((n) => n === 0) && <span className="text-stone-400">none</span>}
      </div>

      <div className="mb-3 flex flex-wrap gap-3 text-xs">
        <span className="font-semibold text-stone-500">Science:</span>
        {SCIENCE_ORDER.map((sym, i) => {
          const count = player.science[i] ?? 0;
          if (count === 0) return null;
          const awarded = player.pairs_awarded.includes(sym);
          return (
            <span
              key={sym}
              className={[
                "rounded px-1.5 py-0.5",
                count >= 2 ? (awarded ? "bg-emerald-200" : "bg-emerald-400") : "bg-stone-200",
              ].join(" ")}
              title={`${sym}: ${count}/2${awarded ? " (pair already rewarded)" : ""}`}
            >
              {SCIENCE_SYMBOL[sym]} {count}/2
            </span>
          );
        })}
      </div>

      {player.tokens.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2 text-xs">
          <span className="font-semibold text-stone-500">Tokens:</span>
          {player.tokens.map((t) => {
            const token = catalog.tokens.find((x) => x.id === t);
            return (
              <span key={t} className="rounded bg-teal-200 px-1.5 py-0.5">
                {token?.name ?? t}
              </span>
            );
          })}
        </div>
      )}

      <div className="mb-3 flex items-center gap-1 text-xs">
        <span className="font-semibold text-stone-500">Shields:</span> {player.shields}
      </div>

      <div className="space-y-2">
        {CARD_TYPE_ORDER.map((kind) => {
          const ids = byType.get(kind);
          if (!ids || ids.length === 0) return null;
          return (
            <div key={kind}>
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-stone-500">
                {CARD_TYPE_LABEL[kind]} ({ids.length})
              </div>
              <div className="flex flex-wrap gap-1.5">
                {ids.map((id) => {
                  const card = cardById(catalog, id);
                  return card ? <CardChip key={id} card={card} /> : null;
                })}
              </div>
            </div>
          );
        })}
      </div>

      {(player.wonders_built.length > 0 || unbuiltWonders.length > 0) && (
        <div className="mt-3">
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-stone-500">Wonders</div>
          <div className="flex flex-wrap gap-1.5">
            {player.wonders_built.map((w) => {
              const wonder = wonderById(catalog, w);
              return wonder ? <WonderChip key={w} wonder={wonder} built /> : null;
            })}
            {unbuiltWonders.map((w) => {
              const wonder = wonderById(catalog, w);
              return wonder ? <WonderChip key={w} wonder={wonder} disabled /> : null;
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function sumProduction(built: string[], catalog: Catalog): ResourceAmounts {
  const total: ResourceAmounts = { wood: 0, clay: 0, stone: 0, glass: 0, papyrus: 0 };
  for (const id of built) {
    const card = cardById(catalog, id);
    if (!card) continue;
    total.wood += card.produces.wood;
    total.clay += card.produces.clay;
    total.stone += card.produces.stone;
    total.glass += card.produces.glass;
    total.papyrus += card.produces.papyrus;
  }
  return total;
}
