import type { GameResult } from "../generated/GameResult";
import type { Breakdown } from "../generated/Breakdown";
import Modal from "./Modal";

interface GameOverProps {
  result: GameResult;
  breakdown: [Breakdown, Breakdown] | null;
  onLeave: () => void;
}

function victoryLabel(kind: string): string {
  switch (kind) {
    case "military_supremacy":
      return "Military supremacy - the conflict pawn reached the opponent's capital!";
    case "scientific_supremacy":
      return "Scientific supremacy - six distinct scientific symbols!";
    case "civilian_victory":
      return "Civilian victory on total points.";
    case "civilian_tiebreak":
      return "Tied on points - decided on civilian (blue) points.";
    default:
      return kind;
  }
}

const ROWS: Array<[keyof Breakdown, string]> = [
  ["civilian", "Civilian (blue)"],
  ["scientific", "Scientific (green)"],
  ["commercial", "Commercial (yellow)"],
  ["guilds", "Guilds (purple)"],
  ["wonders", "Wonders"],
  ["progress_tokens", "Progress tokens"],
  ["military", "Military"],
  ["coins", "Coins"],
];

export default function GameOver({ result, breakdown, onLeave }: GameOverProps) {
  const win = result.type === "win" ? result : null;
  const winnerLabel = win ? (win.winner === "one" ? "Player One" : "Player Two") : null;

  return (
    <Modal title="Game over">
      <div className="space-y-4 text-center">
        <p className="text-xl font-bold">{win ? `${winnerLabel} wins!` : "It's a draw!"}</p>
        {win && <p className="text-sm text-stone-600">{victoryLabel(win.kind)}</p>}

        {breakdown && (
          <div className="overflow-x-auto text-left">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-stone-300">
                  <th className="py-1 text-left">Category</th>
                  <th className="py-1 text-right">Player One</th>
                  <th className="py-1 text-right">Player Two</th>
                </tr>
              </thead>
              <tbody>
                {ROWS.map(([key, label]) => (
                  <tr key={key} className="border-b border-stone-100">
                    <td className="py-1">{label}</td>
                    <td className="py-1 text-right">{breakdown[0][key]}</td>
                    <td className="py-1 text-right">{breakdown[1][key]}</td>
                  </tr>
                ))}
                <tr className="font-bold">
                  <td className="py-1">Total</td>
                  <td className="py-1 text-right">{breakdown[0].total}</td>
                  <td className="py-1 text-right">{breakdown[1].total}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        <button
          type="button"
          onClick={onLeave}
          data-testid="leave-game"
          className="rounded-lg bg-amber-700 px-4 py-2 font-semibold text-white hover:bg-amber-800"
        >
          Back to lobby
        </button>
      </div>
    </Modal>
  );
}
