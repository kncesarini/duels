// Advanced (analysis) mode: what `duels-eval` makes of the live position, and
// the way to flag one as wrongly judged.
//
// A developer/analysis surface, not a player-facing feature - it is docked as
// an overlay rather than woven into the table layout, so turning it off leaves
// the game exactly as it was. Every number here comes from
// `GET /rooms/:id/analysis`; nothing is computed client-side.

import { useEffect, useState } from "react";
import type { AnalysisPayload } from "../generated/AnalysisPayload";
import type { Catalog } from "../generated/Catalog";
import type { Observation } from "../generated/Observation";
import { seatIndex } from "../lib/cost";
import { deltaPct, describeAction, ranked, tone, winPct } from "../lib/analysis";

interface Props {
  analysis: AnalysisPayload | null;
  analysisError: string | null;
  catalog: Catalog | null;
  /** The live position, which is what the analysis describes - not the frame
   * the table happens to be drawing during playback or review. */
  observation: Observation;
  seatNames: [string, string];
  /** True while an earlier position is on screen, in which case the numbers
   * here are about the live position rather than the drawn one and have to say
   * so. */
  reviewing: boolean;
  buildBundle: (notes: string) => Promise<string>;
}

type CopyState = "idle" | "copied" | "failed";

export default function AnalysisPanel({
  analysis,
  analysisError,
  catalog,
  observation,
  seatNames,
  reviewing,
  buildBundle,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [flagging, setFlagging] = useState(false);
  const [notes, setNotes] = useState("");
  const [bundle, setBundle] = useState<string | null>(null);
  const [bundleError, setBundleError] = useState<string | null>(null);
  const [copied, setCopied] = useState<CopyState>("idle");

  useEffect(() => {
    if (copied === "idle") return;
    const t = setTimeout(() => setCopied("idle"), 2500);
    return () => clearTimeout(t);
  }, [copied]);

  // A new position invalidates a bundle built for the previous one.
  useEffect(() => {
    setBundle(null);
    setBundleError(null);
  }, [analysis?.turn]);

  const make = async () => {
    setBundleError(null);
    try {
      const json = await buildBundle(notes);
      setBundle(json);
      return json;
    } catch (e) {
      setBundleError(e instanceof Error ? e.message : String(e));
      return null;
    }
  };

  const copy = async () => {
    const json = bundle ?? (await make());
    if (!json) return;
    try {
      await navigator.clipboard.writeText(json);
      setCopied("copied");
    } catch {
      // Clipboard access is refused in plenty of contexts (no user gesture,
      // an insecure origin, a headless browser). The textarea below is the
      // fallback: the JSON is on screen and selectable either way.
      setCopied("failed");
    }
  };

  const download = async () => {
    const json = bundle ?? (await make());
    if (!json) return;
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `duels-flag-${analysis?.room_id ?? "position"}-turn${analysis?.turn ?? 0}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (collapsed) {
    return (
      <div className="analysis collapsed" data-testid="analysis-panel">
        <button type="button" className="btn ghost" onClick={() => setCollapsed(false)}>
          AI analysis ▸
        </button>
      </div>
    );
  }

  const moverName = analysis ? seatNames[seatIndex(analysis.current_player)] : "";
  const rows = ranked(analysis);

  return (
    <div className="analysis" data-testid="analysis-panel">
      <div className="hd">
        <span className="ttl">AI analysis</span>
        <button type="button" className="btn ghost" onClick={() => setCollapsed(true)} aria-label="Collapse">
          ◂
        </button>
      </div>

      {!analysis ? (
        <div className="note">{analysisError ?? "Reading the position…"}</div>
      ) : (
        <>
          <div className={`now ${tone(analysis.win_probability)}`} data-testid="analysis-winprob">
            <span className="v">{winPct(analysis.win_probability)}</span>
            <span className="lbl">
              win for <b>{moverName}</b>, to move
              <br />
              eval {analysis.value.toFixed(2)} VP · age {analysis.age} · turn {analysis.turn}
            </span>
          </div>

          {reviewing && <div className="note warn">Reviewing an earlier position — these numbers are about the live one.</div>}
          {analysisError && <div className="note warn">Refresh failed ({analysisError}); showing the last reading.</div>}

          <div className="rows" data-testid="analysis-actions">
            {rows.length === 0 && <div className="note">{analysis.game_over ? "Game over — no actions to price." : "Nothing legal right now."}</div>}
            {rows.map((a, i) => (
              <div key={i} className={`arow ${tone(a.win_probability)}`}>
                <span className="bar" style={{ width: `${Math.round(a.win_probability * 100)}%` }} aria-hidden />
                <span className="lb">{describeAction(a.action, catalog, observation)}</span>
                <span className="pc mono">{winPct(a.win_probability)}</span>
                <span className="dl mono">{deltaPct(a.win_probability, analysis.win_probability)}</span>
              </div>
            ))}
          </div>

          <div className="gen" title={analysis.eval_generation}>
            eval generation: <span className="mono">{analysis.eval_generation}</span>
          </div>

          {!flagging ? (
            <button type="button" className="btn" onClick={() => setFlagging(true)} data-testid="flag-position">
              Flag this position
            </button>
          ) : (
            <div className="flag">
              <label htmlFor="flag-notes">What is wrong with this evaluation?</label>
              <textarea
                id="flag-notes"
                data-testid="flag-notes"
                rows={4}
                value={notes}
                placeholder="e.g. it rates discarding above building the second science pair, but the pair is the whole game here"
                onChange={(e) => setNotes(e.target.value)}
              />
              <div className="row">
                <button type="button" className="btn primary" onClick={() => void copy()} data-testid="flag-copy">
                  {copied === "copied" ? "Copied ✓" : copied === "failed" ? "Copy blocked — select below" : "Copy JSON"}
                </button>
                <button type="button" className="btn" onClick={() => void download()} data-testid="flag-download">
                  Download
                </button>
                <button type="button" className="btn ghost" onClick={() => setFlagging(false)}>
                  Close
                </button>
              </div>
              {bundleError && <div className="note warn">{bundleError}</div>}
              {bundle && (
                <textarea className="mono out" data-testid="flag-bundle" readOnly rows={8} value={bundle} />
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
