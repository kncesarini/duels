import { useState } from "react";
import { useGameStore } from "../store";

/** Short human-readable labels for the agent names `GET /agents` returns.
 * Purely cosmetic - any name without an entry here still renders (as
 * itself), so a new agent crate shows up in the picker the moment the server
 * knows about it, even before this map is updated. */
const AGENT_LABELS: Record<string, string> = {
  phased: "Phased — shifts priorities by age",
  alphabeta: "Alpha-Beta — searches, thinks up to 1s",
  "mcts-uct": "MCTS — searches with playouts, thinks up to 1s",
  "mcts-eval": "MCTS+Eval — strongest, thinks up to 1s",
};

export default function Home() {
  const startVsBot = useGameStore((s) => s.startVsBot);
  const startHotSeat = useGameStore((s) => s.startHotSeat);
  const status = useGameStore((s) => s.status);
  const errorMessage = useGameStore((s) => s.errorMessage);
  const catalogError = useGameStore((s) => s.catalogError);
  const agents = useGameStore((s) => s.agents);
  const settings = useGameStore((s) => s.settings);
  const updateSettings = useGameStore((s) => s.updateSettings);
  const [seedInput, setSeedInput] = useState("");
  const [selectedAgent, setSelectedAgent] = useState("phased");

  const seed = seedInput.trim() === "" ? undefined : Number(seedInput.trim());
  const busy = status === "connecting";
  // `agents` defaults to `["phased"]` until `GET /agents` resolves, and a
  // previously selected agent could in principle disappear from a later
  // fetch; fall back to the first known agent rather than submitting a name
  // the server no longer lists.
  const agentToStart = agents.includes(selectedAgent) ? selectedAgent : agents[0];

  return (
    <div className="home">
      <div className="panel">
        <h1>7 Wonders Duel</h1>
        <p className="lede">
          Two cities, one table, nothing hidden. Both players see the same board at all times — the only unknown is
          which cards are still face down, and neither of you knows that.
        </p>

        <div className="field">
          <label htmlFor="opponent">Opponent</label>
          <select
            id="opponent"
            data-testid="opponent-select"
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

        <div className="field">
          <label htmlFor="seed">Seed (optional — the same seed deals the same game)</label>
          <input
            id="seed"
            type="number"
            inputMode="numeric"
            placeholder="random"
            value={seedInput}
            onChange={(e) => setSeedInput(e.target.value)}
          />
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button
            type="button"
            className="btn primary"
            style={{ flex: 1, justifyContent: "center", padding: "10px 16px" }}
            disabled={busy}
            data-testid="start-vs-bot"
            onClick={() => void startVsBot(seed, agentToStart)}
          >
            Play against the AI
          </button>
          <button
            type="button"
            className="btn"
            style={{ flex: 1, justifyContent: "center", padding: "10px 16px" }}
            disabled={busy}
            data-testid="start-hotseat"
            onClick={() => void startHotSeat(seed)}
          >
            Hot-seat, two players
          </button>
        </div>

        <div className="field" style={{ marginTop: 18, marginBottom: 0 }}>
          <label htmlFor="home-theme">Theme</label>
          <select
            id="home-theme"
            value={settings.theme}
            onChange={(e) => updateSettings({ theme: e.target.value as typeof settings.theme })}
            data-testid="home-theme"
          >
            <option value="system">Follow the system</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>

        {busy && <p style={{ color: "var(--mute)", marginTop: 12 }}>Connecting…</p>}
        {errorMessage && (
          <p style={{ color: "var(--bad)", marginTop: 12 }} role="alert">
            {errorMessage}
          </p>
        )}
        {catalogError && (
          <p style={{ color: "var(--bad)", marginTop: 12 }} role="alert">
            Could not reach the server: {catalogError}
          </p>
        )}
      </div>
    </div>
  );
}
