// Winner and condition first, then the category-by-category breakdown as
// paired bars so the story of the game is visible, then the ways back into
// it: the turning points, and a scrubber over every position.

import type { Breakdown } from "../generated/Breakdown";
import type { Catalog } from "../generated/Catalog";
import type { GameResult } from "../generated/GameResult";
import type { Observation } from "../generated/Observation";
import { cardById } from "../lib/catalogHelpers";
import { exportText, type LogEntry } from "../lib/log";
import { seatIndex } from "../lib/cost";
import { verb } from "../lib/text";

interface Props {
  result: GameResult;
  breakdown: [Breakdown, Breakdown];
  observation: Observation;
  catalog: Catalog;
  entries: LogEntry[];
  seatNames: [string, string];
  onReview: (stepIndex: number) => void;
  onRematch: () => void;
  onLeave: () => void;
  historyLength: number;
}

const CATEGORIES: Array<{ key: keyof Breakdown; label: string; colour: string }> = [
  { key: "civilian", label: "Civilian buildings", colour: "var(--civilian)" },
  { key: "scientific", label: "Science buildings", colour: "var(--scientific)" },
  { key: "commercial", label: "Commercial buildings", colour: "var(--commercial)" },
  { key: "guilds", label: "Guilds", colour: "var(--guild)" },
  { key: "wonders", label: "Wonders", colour: "var(--you)" },
  { key: "progress_tokens", label: "Progress tokens", colour: "#7fcb95" },
  { key: "military", label: "Military track", colour: "var(--military)" },
  { key: "coins", label: "Coins ÷ 3", colour: "var(--coin)" },
];

const CONDITION: Record<string, string> = {
  military_supremacy: "military supremacy",
  scientific_supremacy: "scientific supremacy",
  civilian_victory: "victory points",
  civilian_tiebreak: "the civilian tie-break",
};

export default function EndScreen({
  result,
  breakdown,
  observation,
  catalog,
  entries,
  seatNames,
  onReview,
  onRematch,
  onLeave,
  historyLength,
}: Props) {
  const winner = result.type === "win" ? seatIndex(result.winner) : null;
  const max = Math.max(1, ...CATEGORIES.flatMap((c) => [breakdown[0][c.key], breakdown[1][c.key]]));

  const sources = (idx: number, key: keyof Breakdown): string => {
    const built = observation.players[idx].built
      .map((id) => cardById(catalog, id))
      .filter((c): c is NonNullable<typeof c> => Boolean(c));
    switch (key) {
      case "civilian":
        return list(built.filter((c) => c.kind === "civilian"));
      case "scientific":
        return list(built.filter((c) => c.kind === "scientific" && c.victory_points > 0));
      case "commercial":
        return list(built.filter((c) => c.kind === "commercial" && c.victory_points > 0));
      case "guilds":
        return built.filter((c) => c.is_guild).map((c) => c.name).join(", ") || "—";
      case "wonders":
        return (
          observation.players[idx].wonders_built
            .map((id) => catalog.wonders.find((w) => w.id === id))
            .filter(Boolean)
            .map((w) => `${w?.name} ${w?.victory_points}`)
            .join(", ") || "—"
        );
      case "progress_tokens":
        return (
          observation.players[idx].tokens
            .map((id) => catalog.tokens.find((t) => t.id === id)?.name)
            .filter(Boolean)
            .join(", ") || "—"
        );
      case "military":
        return `pawn at ${observation.conflict}`;
      case "coins":
        return `${observation.players[idx].coins}¢`;
      default:
        return "—";
    }
  };

  const turningPoints = entries.filter((e) => e.key).slice(-8);

  const download = () => {
    const blob = new Blob([exportText(entries, seatNames)], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "duel-log.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="endwrap" data-testid="end-screen">
      <div className="end">
        <div>
          <h1 style={{ color: winner === null ? "var(--fg)" : winner === 0 ? "var(--you)" : "var(--opp)" }}>
            {result.type === "draw"
              ? "It's a draw!"
              : `${seatNames[winner ?? 0]} ${verb(seatNames[winner ?? 0], "wins", "win")} on ${
                  CONDITION[result.kind] ?? result.kind
                } · ${breakdown[winner ?? 0].total} to ${breakdown[winner === 0 ? 1 : 0].total}`}
          </h1>
          <div className="sub">
            {result.type === "win" && result.kind === "military_supremacy"
              ? "The conflict pawn reached a capital, so the game ended immediately."
              : result.type === "win" && result.kind === "scientific_supremacy"
                ? "Six distinct scientific symbols ended the game immediately."
                : result.type === "win" && result.kind === "civilian_tiebreak"
                  ? "Totals were level, so the civilian-building tie-break decided it."
                  : "The Age III structure emptied and the points were counted."}
          </div>

          <table className="score">
            <thead>
              <tr>
                <th>Category</th>
                <th style={{ textAlign: "right", color: "var(--you)" }}>{seatNames[0]}</th>
                <th />
                <th />
                <th style={{ color: "var(--opp)" }}>{seatNames[1]}</th>
                <th>Where the points came from</th>
              </tr>
            </thead>
            <tbody>
              {CATEGORIES.map((c) => (
                <tr key={c.key}>
                  <td className="cat">
                    <i style={{ background: c.colour }} />
                    {c.label}
                  </td>
                  <td className="n" style={{ color: "var(--you)" }}>
                    {breakdown[0][c.key]}
                  </td>
                  <td className="bar">
                    <div className="b">
                      <i style={{ width: `${(breakdown[0][c.key] / max) * 100}%` }} />
                    </div>
                  </td>
                  <td className="bar">
                    <div className="b">
                      <i className="o" style={{ width: `${(breakdown[1][c.key] / max) * 100}%` }} />
                    </div>
                  </td>
                  <td className="n" style={{ color: "var(--opp)" }}>
                    {breakdown[1][c.key]}
                  </td>
                  <td className="src">
                    {sources(0, c.key)} <span style={{ color: "var(--line2)" }}>|</span> {sources(1, c.key)}
                  </td>
                </tr>
              ))}
              <tr className="tot">
                <td className="cat">Total</td>
                <td className="n" style={{ color: "var(--you)" }}>
                  {breakdown[0].total}
                </td>
                <td />
                <td />
                <td className="n" style={{ color: "var(--opp)" }}>
                  {breakdown[1].total}
                </td>
                <td />
              </tr>
            </tbody>
          </table>
        </div>

        <div className="endside">
          <div className="panel">
            <h4>Turning points</h4>
            {turningPoints.length === 0 && <div>Nothing stood out.</div>}
            {turningPoints.map((e) => (
              <button key={e.id} type="button" className="tp" onClick={() => onReview(e.stepIndex)}>
                <span className="mono" style={{ color: "var(--mute)" }}>
                  {e.turn === null ? "—" : `T${e.turn}`}
                </span>
                <span>
                  {e.text ??
                    `${e.actor ? seatNames[e.actor === "one" ? 0 : 1] : ""} ${e.verb} ${e.subject?.name ?? ""}${
                      e.chips.length ? ` — ${e.chips.map((c) => c.text).join(", ")}` : ""
                    }`}
                </span>
              </button>
            ))}
          </div>

          <div className="panel">
            <h4>Review the game</h4>
            <div>Step through every position; the table renders exactly as it looked at that move.</div>
            <input
              type="range"
              min={0}
              max={Math.max(0, historyLength - 1)}
              defaultValue={Math.max(0, historyLength - 1)}
              style={{ width: "100%", marginTop: 10 }}
              onChange={(e) => onReview(Number(e.target.value))}
              aria-label="Move scrubber"
            />
            <div style={{ display: "flex", gap: 6, marginTop: 10, flexWrap: "wrap" }}>
              <button type="button" className="btn primary" onClick={() => onReview(Math.max(0, historyLength - 1))}>
                Open review
              </button>
              <button type="button" className="btn" onClick={download}>
                Export log
              </button>
              <button type="button" className="btn" onClick={onRematch} data-testid="rematch">
                Rematch
              </button>
              <button type="button" className="btn ghost" onClick={onLeave} data-testid="leave-game">
                Leave game
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function list(cards: Array<{ name: string; victory_points: number }>): string {
  return cards.map((c) => `${c.name} ${c.victory_points}`).join(", ") || "—";
}
