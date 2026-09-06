// The table: a fixed, non-scrolling three-column layout. Everything either
// player is entitled to see is on screen at once - no tabs, no hover to
// reveal, no scrolling to find the opponent's coins.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Player } from "../generated/Player";
import { localSeat, seatNamesFor, useGameStore } from "../store";
import { otherPlayer, seatIndex } from "../lib/cost";
import { destinationRect, lastRect, recordRects } from "../lib/flight";
import { cardById } from "../lib/catalogHelpers";
import { motionOff } from "../lib/settings";
import { RESOURCES } from "../lib/iconData";
import TopBar from "./TopBar";
import BoardRail from "./BoardRail";
import CityStrip from "./CityStrip";
import Structure from "./Structure";
import MoveTicker from "./MoveTicker";
import ActionTray from "./ActionTray";
import ForcedChoice from "./ForcedChoice";
import GameLog from "./GameLog";
import WonderDraft from "./WonderDraft";
import EndScreen from "./EndScreen";
import SettingsMenu from "./SettingsMenu";
import CardFace from "./CardFace";
import CardHover from "./CardHover";
import SummaryBar from "./SummaryBar";

type SheetTab = "actions" | "you" | "opponent" | "board" | "log";

export default function Game() {
  const s = useGameStore();
  const {
    catalog,
    latest,
    history,
    entries,
    displayedIndex,
    playback,
    reviewIndex,
    handover,
    settings,
    lensOverride,
    peek,
    selectedSlot,
    mode,
    pending,
    pendingSince,
    status,
    errorMessage,
  } = s;

  const [menuOpen, setMenuOpen] = useState(false);
  const [logOpen, setLogOpen] = useState(false);
  const [ghosted, setGhosted] = useState<string | null>(null);
  const [scrollToken, setScrollToken] = useState(0);
  const [sheet, setSheet] = useState<SheetTab>("actions");
  const tickerSince = useRef<number>(Date.now());
  const lastEntryId = useRef<number>(-1);

  useLayoutEffect(() => {
    recordRects();
  });

  const submit = s.submitAction;

  // ---- keyboard ---------------------------------------------------------
  const accessibleList = useMemo(() => {
    if (!latest) return [] as number[];
    return latest.legal_actions
      .filter((a) => a.type === "Build" || a.type === "Discard" || a.type === "BuildWonder")
      .map((a) => (a as { slot: number }).slot)
      .filter((v, i, arr) => arr.indexOf(v) === i)
      .sort((a, b) => a - b);
  }, [latest]);

  const moveSelection = useCallback(
    (delta: number) => {
      if (accessibleList.length === 0) return;
      const at = selectedSlot === null ? -1 : accessibleList.indexOf(selectedSlot);
      const next = accessibleList[(at + delta + accessibleList.length * 2) % accessibleList.length];
      s.selectSlot(next);
    },
    [accessibleList, selectedSlot, s],
  );

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === "Alt") s.setPeek(true);
      if (e.key === "ArrowRight") {
        e.preventDefault();
        moveSelection(1);
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        moveSelection(-1);
      }
      if (e.key === "l" || e.key === "L") setLogOpen((v) => !v);
      if (e.key === "Escape") {
        s.review(null);
        setMenuOpen(false);
      }
      if (e.key === "[") s.stepReview(-1);
      if (e.key === "]") s.stepReview(1);
      if (!latest || selectedSlot === null) return;
      if (e.key === "b" || e.key === "B") {
        const a = latest.legal_actions.find((x) => x.type === "Build" && x.slot === selectedSlot);
        if (a) submit(a);
      }
      if (e.key === "d" || e.key === "D") {
        const a = latest.legal_actions.find((x) => x.type === "Discard" && x.slot === selectedSlot);
        if (a) submit(a);
      }
      if (/^[1-4]$/.test(e.key)) {
        const wonders = latest.legal_actions.filter((x) => x.type === "BuildWonder" && x.slot === selectedSlot);
        const a = wonders[Number(e.key) - 1];
        if (a) submit(a);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === "Alt") s.setPeek(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [latest, selectedSlot, submit, moveSelection, s]);

  if (!catalog || !latest) {
    return (
      <div className="home">
        <div className="panel" style={{ textAlign: "center" }}>
          {status === "error" ? (errorMessage ?? "Connection error") : "Connecting…"}
        </div>
      </div>
    );
  }

  const seatNames = seatNamesFor(latest, mode);
  const me = localSeat(latest);
  const bottom: Player = mode === "hotseat" ? "one" : me;
  const top = otherPlayer(bottom);

  const reviewing = reviewIndex !== null;
  const frameIndex = reviewing ? reviewIndex : displayedIndex;
  const frame =
    frameIndex >= 0 && history[frameIndex]
      ? history[frameIndex]
      : { observation: latest.observation, views: latest.views, accessible_slots: latest.accessible_slots, events: [] };
  const observation = frame.observation;
  const views = frame.views;
  const accessible = new Set(frame.accessible_slots);
  // Whose decision the *drawn* position is waiting on. During playback and in
  // review this is the actor of the frame on screen, not whoever is on move
  // in the live position - otherwise a replayed opponent choice would be
  // narrated as if it were yours.
  const mover = observation.current_player;

  const baseLens = lensOverride ?? mover;
  const lens = peek ? otherPlayer(baseLens) : baseLens;
  const lensIdx = seatIndex(lens);

  const live = !reviewing && displayedIndex === history.length - 1 && !playback;
  const interactive = live && !pending && latest.legal_actions.length > 0;

  // ---- what the move being played back is doing --------------------------
  const playedStep = playback ? history[playback.stepIndex] : null;
  const takeEvent = playedStep?.events.find((e) => e.type === "CardTaken");
  const takenCardId = takeEvent && takeEvent.type === "CardTaken" ? takeEvent.card : null;
  const takenSlot = takeEvent && takeEvent.type === "CardTaken" ? takeEvent.slot : null;
  const revealedSlots = new Set(
    playedStep && playback && playback.phase !== "thinking"
      ? playedStep.events.filter((e) => e.type === "SlotRevealed").map((e) => (e as { slot: number }).slot)
      : [],
  );
  const arrived = new Set<string>();
  if (playedStep) {
    for (const ev of playedStep.events) {
      if (ev.type === "CardBuilt") arrived.add(ev.card);
      if (ev.type === "WonderBuilt") arrived.add(ev.wonder);
      if (ev.type === "ProgressTokenTaken") arrived.add(ev.token);
      if (ev.type === "CardDiscarded") arrived.add(ev.card);
    }
  }
  const flashed = new Set<string>();
  if (playedStep && playback) {
    const before = history[playback.stepIndex - 1];
    if (before) {
      for (const r of RESOURCES) {
        for (const i of [0, 1] as const) {
          if (before.views[i].production[r] !== playedStep.views[i].production[r]) flashed.add(r);
        }
      }
    }
  }

  // ---- the ticker's entry ------------------------------------------------
  const shownStepIndex = playback ? playback.stepIndex : reviewing ? reviewIndex : history.length - 1;
  const tickerEntry =
    [...entries].reverse().find((e) => e.stepIndex === shownStepIndex && e.text === null) ?? null;
  if (tickerEntry && tickerEntry.id !== lastEntryId.current) {
    lastEntryId.current = tickerEntry.id;
    tickerSince.current = Date.now();
  }
  const effectsShown = playback
    ? playback.phase === "take" || playback.phase === "thinking"
      ? 0
      : playback.effects
    : (tickerEntry?.chips.length ?? 0);

  // ---- prompts -----------------------------------------------------------
  // "Thinking" covers both halves of the wait: the round trip while the agent
  // is actually searching, and the deliberate minimum hold afterwards.
  const thinking = (pending && mode === "bot") || playback?.phase === "thinking";
  const oppName = seatNames[seatIndex(otherPlayer(me))];
  const busyNote = playback
    ? playback.phase === "thinking"
      ? `${oppName} is thinking…`
      : "Playing the move…"
    : pending
      ? mode === "bot"
        ? `${oppName} is thinking…`
        : "Submitting…"
      : null;

  const pendingChoice = observation.pending !== null || observation.phase === "choose_first_player";
  const prompt = (() => {
    if (observation.result) return "Game over";
    if (thinking) return `${oppName} is thinking…`;
    if (pending) return "Submitting…";
    if (observation.phase === "wonder_draft") return `${seatNames[seatIndex(mover)]} — draft a wonder`;
    if (observation.phase === "choose_first_player") return `${seatNames[seatIndex(mover)]} — choose who starts the age`;
    if (observation.pending) {
      const kind =
        observation.pending.type === "progress_token"
          ? "choose a progress token"
          : observation.pending.type === "great_library_token"
            ? "choose a Great Library token"
            : observation.pending.type === "destroy"
              ? "choose a card to destroy"
              : "build a card from the discard";
      return `${seatNames[seatIndex(mover)]} — ${kind}`;
    }
    if (mover === me || mode === "hotseat") return `${seatNames[seatIndex(mover)]} — pick a card from the structure`;
    return `${seatNames[seatIndex(mover)]} is deciding…`;
  })();

  const cardsLeft = observation.slots.filter((x) => x.state !== "empty").length;

  const slotOfCard = new Map<string, number>();
  observation.slots.forEach((v, i) => {
    if (v.state === "face_up") slotOfCard.set(v.card, i);
  });

  const actionable = new Set(interactive ? accessibleList : []);

  // What is eligible for the pending choice, lit where it already lives.
  // Derived from the *drawn* position's public state (not from
  // `legal_actions`, which only exists when it is this browser's turn) so the
  // decision space is visible when the opponent is the one choosing too.
  const victimIdx = seatIndex(otherPlayer(mover));
  const destroyTargets = new Set(
    observation.pending?.type === "destroy"
      ? observation.players[victimIdx].built.filter(
          (id) => cardById(catalog, id)?.kind === (observation.pending as { card_type: string }).card_type,
        )
      : [],
  );
  const litTokens = new Set(observation.pending?.type === "progress_token" ? observation.board_tokens : []);
  const litDiscard = new Set(observation.pending?.type === "mausoleum_build" ? observation.discard : []);

  const stripFor = (seat: Player, position: "top" | "bottom") => {
    const idx = seatIndex(seat);
    const isAgent = latest.seats[idx].kind === "agent";
    const agentName = isAgent ? (latest.seats[idx] as { name: string }).name : null;
    return (
      <CityStrip
        key={seat}
        seat={seat}
        player={observation.players[idx]}
        view={views[idx]}
        catalog={catalog}
        name={seatNames[idx]}
        subtitle={`Seat ${idx + 1}${agentName ? ` · ${agentName}` : mover === seat ? " · to move" : ""}`}
        bright={mover === seat}
        position={position}
        conflict={observation.conflict}
        destroyTargets={seat !== mover ? destroyTargets : undefined}
        onDestroy={interactive ? (card) => submit({ type: "DestroyOpponentCard", card }) : undefined}
        arrived={arrived}
        ghosted={ghosted}
        flashed={flashed}
        className={
          sheet === (position === "bottom" ? "you" : "opponent") ? "" : "sheet-hidden"
        }
      />
    );
  };

  const showDraft = observation.phase === "wonder_draft" && !reviewing;

  return (
    <div className="app">
      <div className="table">
        <TopBar
          observation={observation}
          seats={latest.seats}
          seatNames={seatNames}
          lens={lens}
          onLens={(p) => s.setLens(p)}
          thinking={Boolean(thinking)}
          thinkingSince={pendingSince}
          handover={handover}
          cardsLeft={cardsLeft}
          extraTurn={observation.extra_turn}
          prompt={prompt}
          promptSeat={thinking ? otherPlayer(me) : mover}
          onMenu={() => setMenuOpen((v) => !v)}
          onToggleLog={() => setLogOpen((v) => !v)}
          logOpen={logOpen}
        />

        <BoardRail
          className={sheet === "board" ? "" : "sheet-hidden"}
          observation={observation}
          catalog={catalog}
          bottom={bottom}
          seatNames={seatNames}
          litTokens={litTokens}
          onToken={interactive ? (token) => submit({ type: "ChooseProgressToken", token }) : undefined}
          litDiscard={litDiscard}
          onDiscardCard={interactive ? (card) => submit({ type: "MausoleumBuild", card }) : undefined}
          ghosted={ghosted}
        />

        <div className={`centre ${showDraft ? "drafting" : ""} ${reviewing ? "reviewing" : ""}`}>
          {reviewing && (
            <div className="reviewband">
              <span>
                Reviewing move {reviewIndex + 1} of {history.length}
              </span>
              <button type="button" onClick={() => s.stepReview(-1)} aria-label="Previous move">
                ◂
              </button>
              <button type="button" onClick={() => s.stepReview(1)} aria-label="Next move">
                ▸
              </button>
              <button type="button" onClick={() => s.review(null)} data-testid="return-to-live">
                Return to live
              </button>
              {interactive && <span>· it is your turn</span>}
            </div>
          )}

          <SummaryBar
            observation={observation}
            views={views}
            catalog={catalog}
            seatNames={seatNames}
            bottom={bottom}
          />

          {stripFor(top, "top")}

          {showDraft ? (
            <WonderDraft
              observation={observation}
              catalog={catalog}
              legal={interactive ? latest.legal_actions : []}
              seatNames={seatNames}
              onSubmit={submit}
              busy={!interactive}
            />
          ) : (
            <Structure
              observation={observation}
              catalog={catalog}
              lensView={views[lensIdx]}
              oppLens={lens !== bottom}
              accessible={accessible}
              actionable={actionable}
              selectedSlot={selectedSlot}
              onSelect={(slot) => s.selectSlot(slot)}
              revealed={revealedSlots}
              takenSlot={takenSlot}
              dimmed={pendingChoice && !showDraft}
              lensName={seatNames[lensIdx]}
            />
          )}

          {!showDraft && (
          <MoveTicker
            entry={tickerEntry}
            effects={effectsShown}
            seatNames={seatNames}
            since={tickerSince.current}
            onClick={() => setScrollToken((n) => n + 1)}
            reviewNote={reviewing ? "Reviewing:" : handover ? `${seatNames[seatIndex(mover)]} to move ·` : null}
          />
          )}

          {showDraft ? null : pendingChoice ? (
            <ForcedChoice
              observation={observation}
              catalog={catalog}
              views={views}
              legal={latest.legal_actions}
              seatNames={seatNames}
              chooser={mover}
              mine={interactive}
              onSubmit={submit}
            />
          ) : (
            <ActionTray
              catalog={catalog}
              observation={observation}
              views={views}
              seatNames={seatNames}
              mover={mover}
              lens={lens}
              bottom={bottom}
              selectedSlot={selectedSlot}
              legal={latest.legal_actions}
              actionCosts={latest.action_costs}
              accessible={accessible}
              interactive={interactive}
              confirmFirst={settings.confirm}
              onSubmit={submit}
              busyNote={busyNote}
              reviewing={reviewing}
            />
          )}

          {stripFor(bottom, "bottom")}

          <div className="sheet-tabs" role="tablist">
            {(["actions", "you", "opponent", "board", "log"] as SheetTab[]).map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={sheet === t}
                className={sheet === t ? "on" : ""}
                onClick={() => {
                  setSheet(t);
                  if (t === "log") setLogOpen(true);
                }}
              >
                {t === "you" ? seatNames[seatIndex(bottom)] : t === "opponent" ? seatNames[seatIndex(top)] : t}
              </button>
            ))}
          </div>
        </div>

        <GameLog
          entries={entries}
          seatNames={seatNames}
          mySeat={me}
          compact={settings.compactLog}
          reviewIndex={reviewIndex}
          onReview={(i) => s.review(i)}
          onHoverEntry={setGhosted}
          scrollToken={scrollToken}
          open={logOpen}
        />
      </div>

      {menuOpen && (
        <SettingsMenu
          settings={settings}
          onChange={s.updateSettings}
          onLeave={s.leaveGame}
          onClose={() => setMenuOpen(false)}
        />
      )}

      {observation.result && latest.breakdown && !reviewing && (
        <EndScreen
          result={observation.result}
          breakdown={latest.breakdown}
          observation={latest.observation}
          catalog={catalog}
          entries={entries}
          seatNames={seatNames}
          onReview={(i) => s.review(i)}
          onRematch={() => {
            // Same opponent, fresh deal - the seats are fixed by the server's
            // room setup, so this is a new room rather than a reset.
            const agent = latest.seats.find((x) => x.kind === "agent");
            s.leaveGame();
            if (agent && agent.kind === "agent") void s.startVsBot(undefined, agent.name);
            else void s.startHotSeat();
          }}
          onLeave={s.leaveGame}
          historyLength={history.length}
        />
      )}

      {playback?.phase === "take" && takenCardId && !motionOff(settings) && (
        <Flier cardId={takenCardId} catalog={catalog} seat={playedStep?.actor ?? "one"} />
      )}

      <CardHover catalog={catalog} views={views} lens={lens} seatNames={seatNames} slotOfCard={slotOfCard} />

      {status === "reconnecting" && (
        <div style={{ position: "fixed", bottom: 12, left: 12, zIndex: 400 }} className="panel" role="status">
          Connection lost — reconnecting…
        </div>
      )}
      {errorMessage && (
        <div
          style={{ position: "fixed", bottom: 12, right: 12, zIndex: 400, borderColor: "var(--bad)" }}
          className="panel"
          role="alert"
        >
          {errorMessage}
        </div>
      )}
    </div>
  );
}

/** The card in flight: a copy of the taken card, pinned to the slot it left
 * and animated to wherever it has just landed. */
function Flier({
  cardId,
  catalog,
  seat,
}: {
  cardId: string;
  catalog: import("../generated/Catalog").Catalog;
  seat: Player;
}) {
  const [style, setStyle] = useState<React.CSSProperties | null>(null);
  const from = lastRect(cardId);

  useEffect(() => {
    if (!from) return;
    setStyle({ top: from.top, left: from.left, width: from.width, height: from.height, opacity: 1 });
    const id = requestAnimationFrame(() => {
      const to = destinationRect(cardId);
      if (!to) {
        setStyle((prev) => (prev ? { ...prev, opacity: 0 } : prev));
        return;
      }
      const dx = to.left + to.width / 2 - (from.left + from.width / 2);
      const dy = to.top + to.height / 2 - (from.top + from.height / 2);
      setStyle({
        top: from.top,
        left: from.left,
        width: from.width,
        height: from.height,
        transform: `translate(${dx}px, ${dy}px) scale(0.28)`,
        opacity: 0.15,
      });
    });
    return () => cancelAnimationFrame(id);
  }, [cardId, from]);

  const card = cardById(catalog, cardId);
  if (!card || !from || !style) return null;
  return (
    <div className="flier" style={style} aria-hidden>
      <CardFace
        card={card}
        inline
        style={{
          width: "100%",
          height: "100%",
          boxShadow: `0 0 0 2px ${seat === "one" ? "var(--you)" : "var(--opp)"}, 0 10px 24px var(--shadow)`,
        }}
      />
    </div>
  );
}
