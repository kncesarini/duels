// The persistent record of everything the ticker narrates. Always on screen
// at desktop widths; a drawer below that. Scrolling back never costs you the
// live game - only the panel's scroll position is held.

import { useEffect, useMemo, useRef, useState } from "react";
import type { LogEntry } from "../lib/log";
import { exportText, romanAge } from "../lib/log";
import { typeColorVar } from "../lib/catalogHelpers";

type Filter = "all" | "you" | "opponent" | "key";

interface Props {
  entries: LogEntry[];
  seatNames: [string, string];
  /** The seat this browser calls "you" for the filter chips. */
  mySeat: "one" | "two";
  compact: boolean;
  reviewIndex: number | null;
  onReview: (stepIndex: number | null) => void;
  onHoverEntry: (cardId: string | null) => void;
  /** Set by the ticker to scroll the newest entry into view. */
  scrollToken: number;
  open: boolean;
}

export default function GameLog({
  entries,
  seatNames,
  mySeat,
  compact,
  reviewIndex,
  onReview,
  onHoverEntry,
  scrollToken,
  open,
}: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState<number | null>(null);
  const [collapsedAges, setCollapsedAges] = useState<Set<number>>(new Set());
  const [following, setFollowing] = useState(true);
  const [missed, setMissed] = useState(0);
  const body = useRef<HTMLDivElement>(null);
  const seen = useRef(entries.length);

  const shown = useMemo(
    () =>
      entries.filter((e) => {
        if (filter === "all") return true;
        if (filter === "key") return e.key;
        if (filter === "you") return e.actor === mySeat;
        return e.actor !== null && e.actor !== mySeat;
      }),
    [entries, filter, mySeat],
  );

  const currentAge = entries.length > 0 ? entries[entries.length - 1].age : 1;

  useEffect(() => {
    if (following && body.current) {
      body.current.scrollTop = body.current.scrollHeight;
      seen.current = entries.length;
      setMissed(0);
    } else {
      setMissed(entries.length - seen.current);
    }
  }, [entries.length, following, scrollToken, open]);

  const onScroll = () => {
    const el = body.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    setFollowing(atBottom);
    if (atBottom) {
      seen.current = entries.length;
      setMissed(0);
    }
  };

  const jumpToLive = () => {
    setFollowing(true);
    seen.current = entries.length;
    setMissed(0);
    if (body.current) body.current.scrollTop = body.current.scrollHeight;
  };

  const download = (kind: "txt" | "json") => {
    const text =
      kind === "txt"
        ? exportText(entries, seatNames)
        : JSON.stringify(entries, null, 2);
    const blob = new Blob([text], { type: kind === "txt" ? "text/plain" : "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `duel-log.${kind}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const rows: React.ReactNode[] = [];
  let lastAge = -1;
  for (const e of shown) {
    if (e.age !== lastAge) {
      lastAge = e.age;
      const age = e.age;
      const collapsed = collapsedAges.has(age);
      const count = shown.filter((x) => x.age === age).length;
      rows.push(
        <button
          key={`age-${age}`}
          type="button"
          className="agehd"
          onClick={() =>
            setCollapsedAges((s) => {
              const next = new Set(s);
              if (next.has(age)) next.delete(age);
              else next.add(age);
              return next;
            })
          }
        >
          <span>{age === 0 ? "Setup · wonder draft" : `Age ${romanAge(age)}`}</span>
          <small>
            {count} entr{count === 1 ? "y" : "ies"} {collapsed ? "▸" : "▾"}
          </small>
        </button>,
      );
    }
    if (collapsedAges.has(e.age)) continue;
    rows.push(<Entry key={e.id} entry={e} seatNames={seatNames} compact={compact} expanded={expanded === e.id} onToggle={() => setExpanded(expanded === e.id ? null : e.id)} onReview={onReview} onHoverEntry={onHoverEntry} reviewing={reviewIndex !== null && reviewIndex === e.stepIndex} />);
  }

  return (
    <div className={`log ${open ? "open" : ""}`} data-testid="log">
      <div className="hd">
        <div className="row">
          <h4 className="section" style={{ flex: 1 }}>
            Game log
          </h4>
          <span className="mono" style={{ fontSize: 10, color: "var(--mute)" }}>
            {entries.filter((e) => e.turn !== null).length} moves
          </span>
          <button type="button" className="btn ghost" style={{ padding: "2px 7px", fontSize: 10 }} onClick={() => download("txt")}>
            Export
          </button>
          <button type="button" className="btn ghost" style={{ padding: "2px 7px", fontSize: 10 }} onClick={() => download("json")}>
            JSON
          </button>
        </div>
        <div className="chips" role="group" aria-label="Log filter">
          {(["all", "you", "opponent", "key"] as Filter[]).map((f) => (
            <button key={f} type="button" className={filter === f ? "on" : ""} onClick={() => setFilter(f)}>
              {f === "all" ? "All" : f === "key" ? "Key events" : f === "you" ? seatNames[mySeat === "one" ? 0 : 1] : seatNames[mySeat === "one" ? 1 : 0]}
            </button>
          ))}
        </div>
      </div>
      <div className="body" ref={body} onScroll={onScroll}>
        {rows.length === 0 && <div style={{ padding: 12, color: "var(--mute)" }}>Nothing yet.</div>}
        {rows}
        {!following && (
          <button type="button" className="jump" onClick={jumpToLive}>
            ↓ Jump to live{missed > 0 ? ` · ${missed} new` : ""}
          </button>
        )}
      </div>
      <div className="ft">
        <span>
          {reviewIndex !== null ? (
            "Reviewing an earlier move"
          ) : following ? (
            <>
              <b>● Live</b> · following play
            </>
          ) : (
            `Reviewing Age ${romanAge(currentAge)}`
          )}
        </span>
        <span>Click a move to view that position</span>
      </div>
    </div>
  );
}

function Entry({
  entry,
  seatNames,
  compact,
  expanded,
  onToggle,
  onReview,
  onHoverEntry,
  reviewing,
}: {
  entry: LogEntry;
  seatNames: [string, string];
  compact: boolean;
  expanded: boolean;
  onToggle: () => void;
  onReview: (stepIndex: number | null) => void;
  onHoverEntry: (cardId: string | null) => void;
  reviewing: boolean;
}) {
  if (entry.text) {
    return (
      <div className="ent sys">
        <div>{entry.text}</div>
      </div>
    );
  }
  const seat = entry.actor === "one" ? "y" : "o";
  return (
    // A div rather than a button: the expanded body contains its own
    // "show this position" control, and nesting interactive elements is
    // invalid and breaks keyboard order.
    <div
      role="button"
      tabIndex={0}
      className={`ent ${seat} ${reviewing ? "reviewing" : ""}`}
      onClick={onToggle}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onToggle();
        }
      }}
      onMouseEnter={() => onHoverEntry(entry.subject?.id ?? null)}
      onMouseLeave={() => onHoverEntry(null)}
      data-testid="log-entry"
    >
      <span className="tn">{entry.turn === null ? "" : `T${entry.turn}`}</span>
      <div>
        <div className="h">
          <span className="p">{entry.actor ? seatNames[entry.actor === "one" ? 0 : 1] : ""}</span>
          <span className="verb">{entry.verb}</span>
          {entry.subject && (
            <span
              className="chipname"
              style={{ margin: 0, borderLeftColor: entry.subject.type ? typeColorVar(entry.subject.type) : "var(--you)" }}
            >
              {entry.subject.name}
            </span>
          )}
        </div>
        {!compact && entry.payment && <div className="sub">{entry.payment}</div>}
        {entry.chips.length > 0 && (
          <div className="fx">
            {entry.chips.map((c, i) => (
              <span key={i} className={c.kind === "key" ? "key" : c.kind === "mil" ? "mil" : ""}>
                {c.text}
              </span>
            ))}
          </div>
        )}
        {entry.reveals.length > 0 && <div className="sub">revealed {entry.reveals.join(", ")}</div>}
        {expanded && (
          <div className="expand">
            {entry.detail.map((d) => (
              <div key={d}>{d}</div>
            ))}
            {entry.notes.map((n) => (
              <div key={n}>{n}</div>
            ))}
            <span
              role="link"
              tabIndex={0}
              style={{ color: "var(--info)", cursor: "pointer" }}
              onClick={(e) => {
                e.stopPropagation();
                onReview(entry.stepIndex);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.stopPropagation();
                  onReview(entry.stepIndex);
                }
              }}
            >
              Show position after this move ↗
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
