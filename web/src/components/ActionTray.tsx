// The action tray: the selected card, its effect written out in sentences,
// both players' costs side by side, and the three actions with their exact
// coin consequence printed in the button label.

import { useEffect, useState } from "react";
import type { Action } from "../generated/Action";
import type { ActionCost } from "../generated/ActionCost";
import type { AnalysisPayload } from "../generated/AnalysisPayload";
import type { Catalog } from "../generated/Catalog";
import type { CostPlan } from "../generated/CostPlan";
import type { Observation } from "../generated/Observation";
import type { Player } from "../generated/Player";
import type { PlayerView } from "../generated/PlayerView";
import CardFace from "./CardFace";
import { cardById, CARD_TYPE_LABEL, wonderById } from "../lib/catalogHelpers";
import { describeCardEffects } from "../lib/effectText";
import { actionKey, byAction, tone, winPct } from "../lib/analysis";
import { coins, planLines, seatIndex, slotPlan, wonderPlan } from "../lib/cost";
import { Ico } from "../lib/icons";
import { romanAge } from "../lib/log";

interface Props {
  catalog: Catalog;
  observation: Observation;
  views: [PlayerView, PlayerView];
  seatNames: [string, string];
  mover: Player;
  lens: Player;
  /** The seat drawn at the bottom of the table, so the cost boxes keep a
   * stable left-to-right order however the lens is pointed. */
  bottom: Player;
  selectedSlot: number | null;
  legal: Action[];
  actionCosts: ActionCost[];
  accessible: Set<number>;
  interactive: boolean;
  confirmFirst: boolean;
  onSubmit: (a: Action) => void;
  /** Shown instead of the actions while a move is playing back. */
  busyNote: string | null;
  reviewing: boolean;
  /** Advanced mode only: the server's per-action read, so each button can
   * carry the win probability that action leads to. Null in ordinary play. */
  analysis: AnalysisPayload | null;
}

/** The win-probability chip a single action button carries in advanced mode. */
function EvalChip({ p, testid }: { p: number | undefined; testid?: string }) {
  if (p === undefined) return null;
  return (
    <span className={`evalchip ${tone(p)}`} data-testid={testid}>
      {winPct(p)}
    </span>
  );
}

function CostBox({
  title,
  plan,
  purse,
  variant,
}: {
  title: string;
  plan: CostPlan | undefined;
  purse: number;
  variant: "you" | "opp";
}) {
  return (
    <div className={`costbox ${variant}`}>
      <h5>{title}</h5>
      {!plan ? (
        <div style={{ color: "var(--mute)" }}>not available</div>
      ) : plan.via_chain ? (
        <div className="line">
          <span>
            <Ico id="link" /> chain symbol
          </span>
          <span>free</span>
        </div>
      ) : (
        <>
          {plan.coin_cost > 0 && (
            <div className="line">
              <span>
                <Ico id="coin" className="coin" /> printed
              </span>
              <span>{coins(plan.coin_cost)}</span>
            </div>
          )}
          {planLines(plan).map((l) => (
            <div className="line" key={l.resource}>
              <span>
                <Ico id={l.resource} className={l.resource} /> {l.resource} ×{l.count}
              </span>
              <span>{l.note}</span>
            </div>
          ))}
          {plan.lines.length === 0 && plan.coin_cost === 0 && <div className="line">nothing to pay</div>}
        </>
      )}
      {plan && (
        <div className="tot">
          <span>Total</span>
          <span className={`v ${plan.affordable ? "ok" : "bad"}`}>
            {plan.via_chain ? "free" : coins(plan.coins)} · has {coins(purse)}
          </span>
        </div>
      )}
    </div>
  );
}

export default function ActionTray({
  catalog,
  observation,
  views,
  seatNames,
  mover,
  lens,
  bottom,
  selectedSlot,
  legal,
  actionCosts,
  accessible,
  interactive,
  confirmFirst,
  onSubmit,
  busyNote,
  reviewing,
  analysis,
}: Props) {
  const [wonderMenu, setWonderMenu] = useState(false);
  const [armed, setArmed] = useState<string | null>(null);

  useEffect(() => {
    setWonderMenu(false);
    setArmed(null);
  }, [selectedSlot, observation.turn]);

  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(null), 3000);
    return () => clearTimeout(t);
  }, [armed]);

  const moverIdx = seatIndex(mover);
  const lensIdx = seatIndex(lens);
  const bottomIdx = seatIndex(bottom);
  const topIdx = bottomIdx === 0 ? 1 : 0;

  const slotView = selectedSlot === null ? null : observation.slots[selectedSlot];
  const card = slotView && slotView.state === "face_up" ? cardById(catalog, slotView.card) : null;

  if (!card || selectedSlot === null) {
    const affordable = [...accessible].filter((s) => slotPlan(views[lensIdx], s)?.affordable).length;
    const chained = [...accessible].filter((s) => slotPlan(views[lensIdx], s)?.via_chain).length;
    return (
      <div className="tray empty" data-testid="tray">
        <div>
          <div style={{ fontSize: 15, fontWeight: 600, color: "var(--fg)" }}>
            {reviewing
              ? "Reviewing an earlier position — read-only."
              : busyNote ?? (interactive ? "Pick a card from the structure." : `${seatNames[moverIdx]} is deciding…`)}
          </div>
          <div style={{ color: "var(--fg2)", marginTop: 4 }}>
            {accessible.size} card{accessible.size === 1 ? "" : "s"} available · {affordable} affordable to build for{" "}
            {seatNames[lensIdx]}
            {chained > 0 && ` · ${chained} free via a chain`}
            {affordable === 0 && interactive && (
              <b style={{ color: "var(--bad)" }}> · only discards are possible this turn</b>
            )}
          </div>
        </div>
        <div style={{ display: "flex", gap: 18, fontSize: 11.5, color: "var(--fg2)" }}>
          {analysis && (
            <div data-testid="tray-winprob">
              <div style={{ color: "var(--mute)", textTransform: "uppercase", letterSpacing: ".1em", fontSize: 9.5 }}>
                AI win % · {seatNames[seatIndex(analysis.current_player)]}
              </div>
              <div className={`mono evalnum ${tone(analysis.win_probability)}`} style={{ fontSize: 16 }}>
                {winPct(analysis.win_probability)}
              </div>
              <div style={{ color: "var(--mute)" }}>{analysis.value.toFixed(2)} VP eval</div>
            </div>
          )}
          {[0, 1].map((i) => (
            <div key={i}>
              <div style={{ color: "var(--mute)", textTransform: "uppercase", letterSpacing: ".1em", fontSize: 9.5 }}>
                {seatNames[i]} VP now
              </div>
              <div className="mono" style={{ fontSize: 16, color: "var(--fg)" }}>
                {views[i].vp_now.total}
              </div>
              <div style={{ color: "var(--mute)" }}>
                civ {views[i].vp_now.civilian} · sci {views[i].vp_now.scientific} · wonders {views[i].vp_now.wonders} ·
                tokens {views[i].vp_now.progress_tokens} · mil {views[i].vp_now.military} · coins {views[i].vp_now.coins}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const buildAction = legal.find((a) => a.type === "Build" && a.slot === selectedSlot);
  const discardAction = legal.find((a) => a.type === "Discard" && a.slot === selectedSlot);
  const wonderActions = legal.filter((a) => a.type === "BuildWonder" && a.slot === selectedSlot) as Array<{
    type: "BuildWonder";
    slot: number;
    wonder: string;
  }>;
  const buildCost = actionCosts.find((c) => c.type === "Build" && c.slot === selectedSlot);
  const discardCost = actionCosts.find((c) => c.type === "Discard" && c.slot === selectedSlot);

  const scored = byAction(analysis);
  const scoreOf = (a: Action | undefined) =>
    a ? scored.get(actionKey(a))?.win_probability : undefined;

  const myPlan = slotPlan(views[moverIdx], selectedSlot);
  const lensPlan = slotPlan(views[lensIdx], selectedSlot);
  const bottomPlan = slotPlan(views[bottomIdx], selectedSlot);
  const topPlan = slotPlan(views[topIdx], selectedSlot);

  const submit = (a: Action, key: string) => {
    if (confirmFirst && armed !== key) {
      setArmed(key);
      return;
    }
    onSubmit(a);
  };

  return (
    <div className="tray" data-testid="tray">
      <div className="mini">
        <CardFace card={card} plan={lensPlan} oppLens={lens !== bottom} inline style={{ width: 76, height: 101 }} />
      </div>
      <div className="eff">
        <div className="t">
          {card.name}
          <span>
            {CARD_TYPE_LABEL[card.kind]} · Age {romanAge(card.age)}
          </span>
        </div>
        <ul>
          {describeCardEffects(card, catalog).map((line) => (
            <li key={line}>{line}</li>
          ))}
          {myPlan && myPlan.lines.some((l) => l.bought > 0) && (
            <li>
              {myPlan.lines
                .filter((l) => l.bought > 0)
                .map(
                  (l) =>
                    `${l.bought} ${l.resource} must be bought at ${l.unit_price}¢ each (${l.bought * l.unit_price}¢)`,
                )
                .join("; ")}
              .
            </li>
          )}
        </ul>
      </div>
      <div className="costcmp">
        <CostBox
          title={`Cost for ${seatNames[bottomIdx]}`}
          plan={bottomPlan}
          purse={observation.players[bottomIdx].coins}
          variant="you"
        />
        <CostBox
          title={`Cost for ${seatNames[topIdx]}`}
          plan={topPlan}
          purse={observation.players[topIdx].coins}
          variant="opp"
        />
      </div>
      <div className="acts">
        {busyNote ? (
          <div style={{ color: "var(--fg2)" }}>{busyNote}</div>
        ) : reviewing ? (
          <div style={{ color: "var(--fg2)" }}>Read-only while reviewing.</div>
        ) : (
          <>
            <button
              type="button"
              className="btn primary"
              disabled={!interactive || !buildAction}
              onClick={() => buildAction && submit(buildAction, "build")}
              data-testid="act-build"
            >
              {armed === "build" ? "Confirm build" : "Build"}
              <small>
                {buildAction
                  ? buildCost && buildCost.type === "Build"
                    ? buildCost.via_chain
                      ? "free"
                      : `−${coins(buildCost.coins)}`
                    : ""
                  : myPlan
                    ? `need ${coins(myPlan.coins)}, have ${coins(observation.players[moverIdx].coins)}`
                    : "not legal"}
              </small>
              <EvalChip p={scoreOf(buildAction)} testid="eval-build" />
              <span className="kbd">B</span>
            </button>
            <button
              type="button"
              className="btn"
              disabled={!interactive || !discardAction}
              onClick={() => discardAction && submit(discardAction, "discard")}
              data-testid="act-discard"
            >
              {armed === "discard" ? "Confirm discard" : "Discard for coins"}
              <small style={{ color: "var(--ok)" }}>
                {discardCost && discardCost.type === "Discard" ? `+${coins(discardCost.reward)}` : ""}
              </small>
              <EvalChip p={scoreOf(discardAction)} testid="eval-discard" />
              <span className="kbd">D</span>
            </button>
            <button
              type="button"
              className="btn"
              disabled={!interactive || wonderActions.length === 0}
              onClick={() => setWonderMenu((v) => !v)}
              data-testid="act-wonder"
              aria-expanded={wonderMenu}
            >
              Use for a wonder ▾
              <small style={{ color: "var(--mute)" }}>
                {wonderActions.length > 0 ? `${wonderActions.length} affordable` : "none affordable"}
              </small>
              <span className="kbd">1-4</span>
            </button>
            {wonderMenu && wonderActions.length > 0 && (
              <div className="wonder-pop" role="menu">
                {wonderActions.map((a, i) => {
                  const wonder = wonderById(catalog, a.wonder);
                  const plan = wonderPlan(views[moverIdx], a.wonder);
                  return (
                    <button key={a.wonder} type="button" onClick={() => onSubmit(a)} data-testid={`wonder-opt-${a.wonder}`}>
                      <span className="kbd">{i + 1}</span>
                      <span>
                        <b>{wonder?.name ?? a.wonder}</b>
                        <br />
                        {plan ? `${coins(plan.coins)} · ${plan.affordable ? "affordable" : "too expensive"}` : ""}
                      </span>
                      <EvalChip p={scoreOf(a)} />
                    </button>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
