import { useState } from "react";
import { useGameStore } from "../store";

/** Short human-readable labels for the agent names `GET /agents` returns.
 * Purely cosmetic - any name without an entry here still renders (as
 * itself), so a new agent crate shows up in the picker the moment the server
 * knows about it, even before this map is updated. */
const AGENT_LABELS: Record<string, string> = {
  random: "Random",
  greedy: "Greedy",
  alphabeta: "Alpha-Beta Search",
  "mcts-uct": "MCTS (strongest)",
};

export default function Home() {
  const startVsBot = useGameStore((s) => s.startVsBot);
  const startHotSeat = useGameStore((s) => s.startHotSeat);
  const status = useGameStore((s) => s.status);
  const errorMessage = useGameStore((s) => s.errorMessage);
  const catalogError = useGameStore((s) => s.catalogError);
  const agents = useGameStore((s) => s.agents);
  const [seedInput, setSeedInput] = useState("");
  const [selectedAgent, setSelectedAgent] = useState("random");

  const seed = seedInput.trim() === "" ? undefined : Number(seedInput.trim());
  const busy = status === "connecting";
  // `agents` defaults to `["random"]` until `GET /agents` resolves, and a
  // previously selected agent could in principle disappear from a later
  // fetch; fall back to the first known agent rather than submitting a name
  // the server no longer lists.
  const agentToStart = agents.includes(selectedAgent) ? selectedAgent : agents[0];

  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-8 px-4">
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight">7 Wonders Duel</h1>
        <p className="mt-2 text-stone-600">A from-scratch implementation - milestone M2</p>
      </div>

      <div className="w-full max-w-sm space-y-4 rounded-xl border border-stone-300 bg-white p-6 shadow-sm">
        <div>
          <label htmlFor="seed" className="block text-sm font-medium text-stone-700">
            Seed (optional)
          </label>
          <input
            id="seed"
            type="number"
            inputMode="numeric"
            className="mt-1 w-full rounded border border-stone-300 px-3 py-2 text-sm"
            placeholder="random"
            value={seedInput}
            onChange={(e) => setSeedInput(e.target.value)}
          />
        </div>

        <div>
          <label htmlFor="opponent" className="block text-sm font-medium text-stone-700">
            Opponent
          </label>
          <select
            id="opponent"
            data-testid="opponent-select"
            className="mt-1 w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm"
            value={agentToStart}
            onChange={(e) => setSelectedAgent(e.target.value)}
          >
            {agents.map((name) => (
              <option key={name} value={name}>
                {AGENT_LABELS[name] ?? name}
              </option>
            ))}
          </select>
        </div>

        <button
          type="button"
          disabled={busy}
          data-testid="start-vs-bot"
          onClick={() => void startVsBot(seed, agentToStart)}
          className="w-full rounded-lg bg-amber-700 px-4 py-3 font-semibold text-white transition hover:bg-amber-800 disabled:opacity-50"
        >
          Play vs {AGENT_LABELS[agentToStart] ?? agentToStart} Bot
        </button>

        <button
          type="button"
          disabled={busy}
          data-testid="start-hotseat"
          onClick={() => void startHotSeat(seed)}
          className="w-full rounded-lg bg-stone-700 px-4 py-3 font-semibold text-white transition hover:bg-stone-800 disabled:opacity-50"
        >
          Hot-seat (two players, one browser)
        </button>

        {busy && <p className="text-center text-sm text-stone-500">Connecting...</p>}
        {errorMessage && <p className="text-center text-sm text-red-600">{errorMessage}</p>}
        {catalogError && (
          <p className="text-center text-sm text-red-600">
            Could not reach the server: {catalogError}
          </p>
        )}
      </div>
    </div>
  );
}
