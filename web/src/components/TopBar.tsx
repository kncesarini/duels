// Age progress, the turn prompt (the single place that says whose turn it is
// and what kind of decision is pending - it is never blank), the opponent's
// identity, and the cost lens.

import { useEffect, useState } from "react";
import type { Observation } from "../generated/Observation";
import type { Player } from "../generated/Player";
import type { SeatSpec } from "../generated/SeatSpec";
import { romanAge } from "../lib/log";
import { seatIndex } from "../lib/cost";
import { Ico } from "../lib/icons";

interface Props {
  observation: Observation;
  seats: [SeatSpec, SeatSpec];
  seatNames: [string, string];
  lens: Player;
  onLens: (p: Player) => void;
  /** True while the opponent's move is being waited for or paced. */
  thinking: boolean;
  thinkingSince: number | null;
  handover: boolean;
  cardsLeft: number;
  extraTurn: boolean;
  prompt: string;
  promptSeat: Player;
  onMenu: () => void;
  onToggleLog: () => void;
  logOpen: boolean;
}

export default function TopBar({
  observation,
  seats,
  seatNames,
  lens,
  onLens,
  thinking,
  thinkingSince,
  handover,
  cardsLeft,
  extraTurn,
  prompt,
  promptSeat,
  onMenu,
  onToggleLog,
  logOpen,
}: Props) {
  const [, force] = useState(0);
  useEffect(() => {
    if (!thinking) return;
    const t = setInterval(() => force((n) => n + 1), 250);
    return () => clearInterval(t);
  }, [thinking]);

  const elapsed = thinking && thinkingSince ? (Date.now() - thinkingSince) / 1000 : 0;
  const oppIdx = seatIndex(promptSeat) === 0 ? 1 : 0;
  void oppIdx;
  const agentSeat = seats.findIndex((s) => s.kind === "agent");
  const agentName = agentSeat >= 0 && seats[agentSeat].kind === "agent" ? (seats[agentSeat] as { name: string }).name : null;

  return (
    <div className="topbar">
      <span className="brand">DUEL</span>
      <div className="ages" aria-label={`Age ${romanAge(observation.age)}`}>
        {[1, 2, 3].map((a) => (
          <span key={a} className={observation.age === a ? "cur" : observation.age > a ? "done" : ""}>
            Age {romanAge(a)}
          </span>
        ))}
      </div>
      <span className="mono turncount" style={{ fontSize: 11, color: "var(--mute)", whiteSpace: "nowrap" }}>
        Turn {observation.turn} · {cardsLeft} cards left in age
      </span>

      <div className={`turnmsg ${promptSeat === "two" ? "opp" : ""}`} role="status" aria-live="polite">
        <span className="dot" />
        {handover ? `${seatNames[seatIndex(promptSeat)]} to move` : prompt}
        {extraTurn && (
          <span title="plays again">
            <Ico id="repeat" />
          </span>
        )}
        {thinking && elapsed > 1 && (
          <small className="mono">
            {elapsed.toFixed(1)}s{elapsed > 10 ? " · still thinking…" : ""}
          </small>
        )}
      </div>

      <div className="right">
        {agentName && (
          <span style={{ fontSize: 11, whiteSpace: "nowrap" }}>
            Opponent:{" "}
            <b className="mono" style={{ color: "var(--opp)", fontWeight: 500 }}>
              {agentName}
            </b>
          </span>
        )}
        <div className="lens" role="group" aria-label="Cost lens">
          <span>Costs for</span>
          <button type="button" className={lens === "one" ? "on" : ""} onClick={() => onLens("one")} data-testid="lens-one">
            {seatNames[0]}
          </button>
          <button
            type="button"
            className={lens === "two" ? "on opp" : ""}
            onClick={() => onLens("two")}
            data-testid="lens-two"
          >
            {seatNames[1]}
          </button>
        </div>
        <button type="button" className="btn ghost logtoggle" onClick={onToggleLog} aria-expanded={logOpen}>
          Log
        </button>
        <button type="button" className="btn ghost" onClick={onMenu} aria-label="Settings and menu" data-testid="menu">
          ☰
        </button>
      </div>
    </div>
  );
}
