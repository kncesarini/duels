// The age structure. Three visibility states, all of them readable:
// accessible (full brightness, interactive), covered but face-up (dimmed,
// cost row still rendered so you can plan ahead), and face-down (age glyph
// only - that really is all anybody knows, except that a guild's back is
// purple, which the server tells us in `hidden_guild_slots`).

import { Fragment, useLayoutEffect, useRef, useState } from "react";
import type { AnalysisPayload } from "../generated/AnalysisPayload";
import type { Catalog } from "../generated/Catalog";
import type { Observation } from "../generated/Observation";
import type { PlayerView } from "../generated/PlayerView";
import CardFace, { CardBack } from "./CardFace";
import { bestForSlot, tone, winPct } from "../lib/analysis";
import { cardById } from "../lib/catalogHelpers";
import { slotPlan } from "../lib/cost";
import { useHover } from "../lib/hover";
import { romanAge } from "../lib/log";
import { Ico } from "../lib/icons";

const SHAPE = ["", "pyramid", "inverted pyramid", "pinched"] as const;

interface Props {
  observation: Observation;
  catalog: Catalog;
  lensView: PlayerView;
  oppLens: boolean;
  accessible: Set<number>;
  /** Slots this browser may act on right now. */
  actionable: Set<number>;
  selectedSlot: number | null;
  onSelect: (slot: number) => void;
  /** Slots revealed by the move currently being played back. */
  revealed: Set<number>;
  /** The slot the card in flight came from. */
  takenSlot: number | null;
  dimmed: boolean;
  lensName: string;
  /** Advanced mode only, and only while the live position is on screen: the
   * server's read of every legal action, so each takeable slot can carry the
   * best win probability reachable from it. Null in ordinary play. */
  analysis: AnalysisPayload | null;
}

interface Metrics {
  w: number;
  h: number;
  gap: number;
  rowStep: number;
  offsetX: number;
  offsetY: number;
}

export default function Structure({
  observation,
  catalog,
  lensView,
  oppLens,
  accessible,
  actionable,
  selectedSlot,
  onSelect,
  revealed,
  takenSlot,
  dimmed,
  lensName,
  analysis,
}: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [metrics, setMetrics] = useState<Metrics>({ w: 92, h: 122, gap: 10, rowStep: 68, offsetX: 0, offsetY: 0 });
  const showHover = useHover((s) => s.show);

  const positions = catalog.layouts[observation.age - 1]?.positions ?? catalog.layouts[0].positions;
  const rows = positions.map(([r]) => r);
  const cols = positions.map(([, c]) => c);
  const minRow = Math.min(...rows);
  const maxRow = Math.max(...rows);
  const minCol = Math.min(...cols);
  const maxCol = Math.max(...cols);

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const measure = () => {
      const availW = el.clientWidth;
      const availH = el.clientHeight;
      if (availW === 0 || availH === 0) return;
      const colSpan = (maxCol - minCol) / 2;
      const rowSpan = maxRow - minRow;
      // width  = colSpan * (w + gap) + w, with gap = 0.11w
      // height = rowSpan * 0.55h + h, with h = 4w/3
      const byWidth = availW / (colSpan * 1.11 + 1);
      const byHeight = availH / ((rowSpan * 0.55 + 1) * (4 / 3));
      const w = Math.max(48, Math.min(120, Math.floor(Math.min(byWidth, byHeight))));
      const h = Math.round((w * 4) / 3);
      const gap = Math.round(w * 0.11);
      const rowStep = Math.round(h * 0.55);
      const totalW = colSpan * (w + gap) + w;
      const totalH = rowSpan * rowStep + h;
      setMetrics({
        w,
        h,
        gap,
        rowStep,
        offsetX: Math.max(0, (availW - totalW) / 2),
        offsetY: Math.max(0, (availH - totalH) / 2),
      });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [minCol, maxCol, minRow, maxRow]);

  const faceDown = observation.slots.filter((s) => s.state === "face_down").length;
  const guildBacks = observation.slots.filter((s, i) => s.state === "face_down" && isGuildBack(observation, i)).length;
  const availableCount = [...accessible].filter((s) => observation.slots[s].state === "face_up").length;
  const affordable = [...accessible].filter((s) => {
    const p = slotPlan(lensView, s);
    return p?.affordable ?? false;
  }).length;
  const chained = [...accessible]
    .map((s) => ({ s, p: slotPlan(lensView, s) }))
    .filter((x) => x.p?.via_chain)
    .map((x) => {
      const view = observation.slots[x.s];
      return view.state === "face_up" ? cardById(catalog, view.card)?.name : null;
    })
    .filter(Boolean) as string[];

  let revealIndex = 0;

  // The slot the evaluation likes most, so the badge on it can be marked as
  // the pick rather than leaving the reader to compare a dozen percentages.
  const bestSlot = analysis
    ? (analysis.actions.reduce<{ slot: number; p: number } | null>((acc, a) => {
        if (!("slot" in a.action)) return acc;
        return !acc || a.win_probability > acc.p ? { slot: a.action.slot, p: a.win_probability } : acc;
      }, null)?.slot ?? null)
    : null;

  return (
    <div className={`structure ${dimmed ? "dim" : ""}`}>
      <div className="shead">
      <div className="meta">
        <b>Age {romanAge(observation.age)}</b> · {SHAPE[observation.age]} · <b>{availableCount}</b> available ·{" "}
        <b>{affordable}</b> affordable for {lensName} · {faceDown} face-down
        {guildBacks > 0 && <span className="guild-hint"> ({guildBacks} guild)</span>}
        {chained.length > 0 && (
          <span className="chain-hint">
            {" · "}
            <Ico id="link" /> {chained.join(", ")} free for {lensName}
          </span>
        )}
      </div>
      <div className="legend" aria-hidden>
        <span>
          Available to take<i style={{ background: "var(--card-face)" }} />
        </span>
        <span>
          Covered, still readable<i style={{ background: "var(--card-face)", opacity: 0.55 }} />
        </span>
        <span>
          Face down<i style={{ background: "var(--back-a)" }} />
        </span>
        <span>
          Face down (guild)<i style={{ background: "var(--guild)" }} />
        </span>
        <span>
          Can&apos;t afford<i style={{ background: "var(--badge-bad)" }} />
        </span>
      </div>
      </div>
      <div className="pyr" ref={box} role="group" aria-label="Age structure">
        {positions.map(([row, col], slot) => {
          const view = observation.slots[slot];
          const style = {
            left: metrics.offsetX + ((col - minCol) / 2) * (metrics.w + metrics.gap),
            top: metrics.offsetY + (row - minRow) * metrics.rowStep,
            width: metrics.w,
            height: metrics.h,
            zIndex: row,
          };
          if (view.state === "empty") {
            return <div key={slot} className={`slot ${takenSlot === slot ? "pulse" : ""}`} style={style} />;
          }
          if (view.state === "face_down") {
            return <CardBack key={slot} age={observation.age} style={style} isGuild={isGuildBack(observation, slot)} />;
          }
          const card = cardById(catalog, view.card);
          if (!card) return null;
          const isAccessible = accessible.has(slot);
          const canAct = actionable.has(slot);
          const plan = slotPlan(lensView, slot);
          const isRevealing = revealed.has(slot);
          const delay = isRevealing ? revealIndex++ * 0.15 : 0;
          const best = analysis ? bestForSlot(analysis, slot) : null;
          return (
            <Fragment key={slot}>
              <CardFace
                card={card}
                plan={plan}
                oppLens={oppLens}
                accessible={isAccessible}
                covered={!isAccessible}
                unaffordable={isAccessible && plan ? !plan.affordable && !plan.via_chain : false}
                selected={selectedSlot === slot}
                revealing={isRevealing}
                style={{ ...style, animationDelay: delay ? `calc(${delay}s * var(--speed))` : undefined }}
                onClick={canAct ? () => onSelect(slot) : undefined}
                onHover={(el) => showHover("card", card.id, el, countUncovers(observation, catalog, slot))}
              />
              {best && (
                // Always drawn, never on hover: the point of advanced mode is
                // scanning every option at once.
                <span
                  className={`evalbadge ${tone(best.win_probability)} ${
                    bestSlot === slot ? "top" : ""
                  }`}
                  style={{ left: style.left + metrics.w - 4, top: style.top - 6, zIndex: row + 40 }}
                  title={`best from this slot: ${best.action.type} · ${best.value.toFixed(2)} VP`}
                >
                  {winPct(best.win_probability)}
                </span>
              )}
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}

/** Whether this face-down slot's back is purple, i.e. it holds one of the
 * three guilds. `hidden_guild_slots` is a slot bitmask the server derives in
 * `duels-core` (R-110); bit `i` corresponds to `observation.slots[i]`, the
 * same indexing this component already draws from. Public information - a
 * guild card back is a different colour in the physical game - and it says
 * only guild-or-not, never which guild. */
function isGuildBack(observation: Observation, slot: number): boolean {
  return (observation.hidden_guild_slots & (1 << slot)) !== 0;
}

/** How many face-down cards this slot is currently sitting on top of. Read
 * off the printed geometry the server sends in the catalog - the same
 * `(row, col)` table `duels-core` derives covering from - so it is a drawing
 * question, not a rules one: the authoritative "is it accessible" answer
 * still comes from the server's `accessible_slots`. */
function countUncovers(observation: Observation, catalog: Catalog, slot: number): number {
  const positions = catalog.layouts[observation.age - 1]?.positions;
  if (!positions) return 0;
  const [row, col] = positions[slot];
  let n = 0;
  positions.forEach(([r, c], i) => {
    if (r === row - 1 && Math.abs(c - col) === 1 && observation.slots[i].state === "face_down") n += 1;
  });
  return n;
}
