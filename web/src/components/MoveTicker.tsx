// The last completed action, narrated, until the next one happens - so a move
// an engine produced in five milliseconds is never "just gone". Clicking it
// scrolls the log to the same entry.

import { useEffect, useState } from "react";
import type { LogEntry } from "../lib/log";
import { typeColorVar } from "../lib/catalogHelpers";

interface Props {
  entry: LogEntry | null;
  /** How many consequence chips to show; the rest are still resolving. */
  effects: number;
  seatNames: [string, string];
  /** Wall-clock ms when the entry was first shown. */
  since: number | null;
  onClick: () => void;
  reviewNote: string | null;
}

export default function MoveTicker({ entry, effects, seatNames, since, onClick, reviewNote }: Props) {
  const [, force] = useState(0);
  useEffect(() => {
    const t = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, []);

  if (!entry) {
    return (
      <div className="ticker" aria-live="polite">
        <span style={{ color: "var(--mute)" }}>{reviewNote ?? "No moves yet."}</span>
      </div>
    );
  }

  const who = entry.actor === null ? null : seatNames[entry.actor === "one" ? 0 : 1];
  const seatClass = entry.actor === "one" ? "y" : "o";
  const age = since === null ? null : Math.round((Date.now() - since) / 1000);

  return (
    <button type="button" className="ticker" onClick={onClick} aria-live="polite" data-testid="ticker">
      {reviewNote && <span style={{ color: "var(--you)" }}>{reviewNote}</span>}
      {entry.text ? (
        <span style={{ fontStyle: "italic" }}>{entry.text}</span>
      ) : (
        <>
          <span className={`who ${seatClass}`}>{who}</span>
          <span>{entry.verb}</span>
          {entry.subject && (
            <span
              className="chipname"
              style={{ margin: 0, borderLeftColor: entry.subject.type ? typeColorVar(entry.subject.type) : "var(--you)" }}
            >
              {entry.subject.name}
            </span>
          )}
          {entry.payment && (
            <>
              <span className="sep">·</span>
              <span style={{ color: "var(--mute)" }}>{entry.payment}</span>
            </>
          )}
          {entry.chips.length > 0 && <span className="sep">→</span>}
          {entry.chips.slice(0, effects).map((c, i) => (
            <span key={i} className={`eff ${c.kind === "key" ? "key" : c.kind === "mil" ? "mil" : ""}`}>
              {c.text}
            </span>
          ))}
          {entry.reveals.length > 0 && effects >= entry.chips.length && (
            <span style={{ color: "var(--mute)" }}>revealed {entry.reveals.join(", ")}</span>
          )}
        </>
      )}
      <span className="age-t mono">
        {entry.turn !== null && `Turn ${entry.turn}`}
        {age !== null && ` · ${age < 2 ? "just now" : `${age}s ago`}`}
      </span>
    </button>
  );
}
