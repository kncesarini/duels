// Turns the server's `StepPayload` stream into the game log's entries: one
// per applied action, with the actor, what was taken, what it cost (with the
// arithmetic), every consequence, and everything it revealed.
//
// Every fact here is read off an `Event`, an `Observation` or a server-computed
// `PlayerView`. Nothing is inferred by re-deriving a rule: the VP delta on an
// entry, for instance, is the difference between two `vp_now` totals the
// server scored, not a client-side count of blue cards.

import type { CardType } from "../generated/CardType";
import type { Catalog } from "../generated/Catalog";
import type { Event } from "../generated/Event";
import type { Player } from "../generated/Player";
import type { StepPayload } from "../generated/StepPayload";
import { cardById, cardName, tokenName, wonderName } from "./catalogHelpers";
import { COIN, coins, paymentDetail, paymentSummary, seatIndex, slotPlan, wonderPlan } from "./cost";
import { SCIENCE_LABEL } from "./iconData";
import { possessive, verb } from "./text";

export type ChipKind = "plain" | "key" | "mil" | "chain";

export interface Chip {
  text: string;
  kind: ChipKind;
}

export interface EntrySubject {
  id: string;
  name: string;
  /** `null` for a wonder, which has its own gold chip colour. */
  type: CardType | null;
}

export interface LogEntry {
  /** Position in the entry list. */
  id: number;
  /** Index into the step history, for review mode. `-1` for a derived
   * system entry that has no step of its own. */
  stepIndex: number;
  /** `null` for a system entry (age change, wonders revealed). */
  actor: Player | null;
  /** Which age group the entry belongs to; `0` means setup / the draft. */
  age: number;
  /** `duels_core`'s decision counter after this action. */
  turn: number | null;
  verb: string;
  subject: EntrySubject | null;
  /** e.g. `paid 4¢ (1 wood at 4¢)`. */
  payment: string;
  /** Long-form arithmetic, shown when the entry is expanded. */
  detail: string[];
  chips: Chip[];
  /** Names of cards this action turned face-up. */
  reveals: string[];
  /** Extra prose under the entry (the card spent on a wonder, tokens
   * returned to the box, and so on). */
  notes: string[];
  /** Whether the "Key events" filter keeps this entry. */
  key: boolean;
  /** For a system entry, the sentence to print. */
  text: string | null;
}

const KEY_EVENTS: ReadonlySet<Event["type"]> = new Set<Event["type"]>([
  "SciencePairCompleted",
  "ProgressTokenTaken",
  "WonderBuilt",
  "MilitaryLootTriggered",
  "CardDestroyed",
  "ExtraTurnGranted",
  "AgeStarted",
  "GameEnded",
  "FirstPlayerChosen",
]);

function playerLabel(p: Player, seatNames: [string, string]): string {
  return seatNames[seatIndex(p)];
}

/** Build every log entry for a whole step history. Cheap enough to redo on
 * each new step (a game is ~60 steps), which keeps the derivation in exactly
 * one place rather than split between an initial replay and a live append. */
export function buildEntries(
  steps: StepPayload[],
  catalog: Catalog | null,
  seatNames: [string, string],
): LogEntry[] {
  const entries: LogEntry[] = [];
  steps.forEach((step, i) => {
    const prev = i > 0 ? steps[i - 1] : null;
    for (const entry of entriesForStep(step, prev, i, catalog, seatNames)) {
      entries.push({ ...entry, id: entries.length });
    }
  });
  return entries;
}

function blank(): Omit<LogEntry, "id"> {
  return {
    stepIndex: -1,
    actor: null,
    age: 0,
    turn: null,
    verb: "",
    subject: null,
    payment: "",
    detail: [],
    chips: [],
    reveals: [],
    notes: [],
    key: false,
    text: null,
  };
}

function entriesForStep(
  step: StepPayload,
  prev: StepPayload | null,
  stepIndex: number,
  catalog: Catalog | null,
  seatNames: [string, string],
): Array<Omit<LogEntry, "id">> {
  const out: Array<Omit<LogEntry, "id">> = [];
  const actor = step.actor;
  const action = step.action;
  const drafting = action?.type === "PickWonder";
  // The age the action was taken *in*, which is the age before the step for
  // the move that empties a structure.
  const age = drafting ? 0 : (prev?.observation.age ?? step.observation.age);

  const entry: Omit<LogEntry, "id"> = {
    ...blank(),
    stepIndex,
    actor,
    age,
    turn: step.observation.turn,
  };

  const chips: Chip[] = [];
  const reveals: string[] = [];
  const notes: string[] = [];
  let key = false;
  const systemAfter: Array<Omit<LogEntry, "id">> = [];

  for (const ev of step.events) {
    if (KEY_EVENTS.has(ev.type)) key = true;
    switch (ev.type) {
      case "WonderPicked":
        entry.verb = "drafted";
        entry.subject = { id: ev.wonder, name: wonderName(catalog, ev.wonder), type: null };
        break;
      case "WonderGroupRevealed":
        systemAfter.push({
          ...blank(),
          stepIndex,
          age: 0,
          text: `Four wonders offered: ${ev.wonders.map((w) => wonderName(catalog, w)).join(", ")}.`,
        });
        break;
      case "CardBuilt": {
        const card = catalog ? cardById(catalog, ev.card) : undefined;
        // A Mausoleum build also emits `CardBuilt`, but its verb is set by
        // the action below; only claim the verb if nothing else has.
        if (!entry.subject) {
          entry.verb = "built";
          entry.subject = { id: ev.card, name: cardName(catalog, ev.card), type: card?.kind ?? null };
        }
        if (ev.via_chain) chips.push({ text: "free via chain", kind: "chain" });
        break;
      }
      case "CardDiscarded":
        if (!entry.subject) {
          const card = catalog ? cardById(catalog, ev.card) : undefined;
          entry.verb = "discarded";
          entry.subject = { id: ev.card, name: cardName(catalog, ev.card), type: card?.kind ?? null };
        }
        break;
      case "WonderBuilt":
        entry.verb = "built wonder";
        entry.subject = { id: ev.wonder, name: wonderName(catalog, ev.wonder), type: null };
        notes.push(`Card spent: ${cardName(catalog, ev.card)} (face-down under the wonder).`);
        notes.push(`${ev.total_built} of 7 wonders now built.`);
        break;
      case "SlotRevealed":
        reveals.push(cardName(catalog, ev.card));
        break;
      case "CoinsGained":
        if (ev.amount > 0) {
          chips.push({ text: `+${coins(ev.amount)}`, kind: ev.reason === "military_loot" ? "mil" : "plain" });
        }
        break;
      case "CoinsLost":
        // Construction costs and trade payments are the payment line, not a
        // consequence; everything else really is something that happened to
        // the player.
        if (ev.reason !== "construction_cost" && ev.reason !== "trade" && ev.amount > 0) {
          chips.push({
            text: `−${coins(ev.amount)} ${ev.reason === "military_loot" ? "loot" : "penalty"}`,
            kind: "mil",
          });
        }
        break;
      case "ConflictMoved":
        chips.push({
          text: `${ev.shields} shield${ev.shields === 1 ? "" : "s"} · pawn ${ev.from} → ${ev.to}`,
          kind: "mil",
        });
        break;
      case "MilitaryLootTriggered":
        chips.push({
          text: `loot at ${ev.distance}: ${playerLabel(ev.loser, seatNames)} ${verb(
            playerLabel(ev.loser, seatNames),
            "forfeits",
            "forfeit",
          )} ${coins(ev.coins)}`,
          kind: "mil",
        });
        break;
      case "ScienceGained":
        chips.push({ text: `${SCIENCE_LABEL[ev.symbol]} · ${ev.distinct} distinct`, kind: "plain" });
        break;
      case "SciencePairCompleted":
        chips.push({
          text: ev.token_available
            ? `${SCIENCE_LABEL[ev.symbol]} pair → a progress token`
            : `${SCIENCE_LABEL[ev.symbol]} pair (no tokens left)`,
          kind: "key",
        });
        break;
      case "ProgressTokenTaken":
        if (!entry.subject) {
          entry.verb = ev.from_great_library ? "took from the Great Library" : "claimed";
          entry.subject = { id: ev.token, name: tokenName(catalog, ev.token), type: null };
        } else {
          chips.push({ text: `claimed ${tokenName(catalog, ev.token)}`, kind: "key" });
        }
        break;
      case "GreatLibraryDraw":
        notes.push(`Drew ${ev.tokens.map((t) => tokenName(catalog, t)).join(", ")} from the set-aside tokens.`);
        break;
      case "DestroyPending":
        notes.push(`Must destroy one opponent ${ev.card_type.replace(/_/g, " ")} building.`);
        break;
      case "CardDestroyed":
        entry.verb = "destroyed";
        entry.subject = {
          id: ev.card,
          name: cardName(catalog, ev.card),
          type: (catalog ? cardById(catalog, ev.card)?.kind : null) ?? null,
        };
        notes.push(
          `Taken out of ${possessive(playerLabel(ev.victim, seatNames))} city and put in the discard pile.`,
        );
        break;
      case "ExtraTurnGranted":
        chips.push({ text: "plays again", kind: "key" });
        break;
      case "ExtraTurnLost":
        chips.push({ text: "extra turn lost", kind: "plain" });
        break;
      case "AgeEnded":
        break;
      case "AgeStarted":
        systemAfter.push({
          ...blank(),
          stepIndex,
          age: ev.age,
          text: `Age ${romanAge(ev.age)} begins.`,
          key: true,
        });
        break;
      case "FirstPlayerChosen":
        if (!entry.subject && action?.type === "ChooseFirstPlayer") {
          entry.verb = "chose";
          entry.subject = { id: ev.player, name: `${playerLabel(ev.player, seatNames)} to start`, type: null };
        } else {
          systemAfter.push({
            ...blank(),
            stepIndex,
            age: step.observation.age,
            text: `${playerLabel(ev.player, seatNames)} ${verb(
            playerLabel(ev.player, seatNames),
            "starts",
            "start",
          )} the age.`,
            key: true,
          });
        }
        break;
      case "GameEnded":
        systemAfter.push({
          ...blank(),
          stepIndex,
          age: step.observation.age,
          text:
            ev.result.type === "draw"
              ? "The game is a draw."
              : `${playerLabel(ev.result.winner, seatNames)} ${verb(
                  playerLabel(ev.result.winner, seatNames),
                  "wins",
                  "win",
                )} by ${ev.result.kind.replace(/_/g, " ")}.`,
          key: true,
        });
        break;
      case "CardTaken":
        break;
    }
  }

  // What the actor paid, from the cost plan for the position *before* this
  // action - the same plan the tray showed them when they chose it.
  if (actor && action && prev) {
    const view = prev.views[seatIndex(actor)];
    const plan =
      action.type === "Build"
        ? slotPlan(view, action.slot)
        : action.type === "BuildWonder"
          ? wonderPlan(view, action.wonder)
          : undefined;
    if (plan) {
      entry.payment = paymentSummary(plan);
      entry.detail = paymentDetail(
        plan,
        prev.observation.players[seatIndex(actor)].coins,
        step.observation.players[seatIndex(actor)].coins,
      );
    }
    if (action.type === "MausoleumBuild") {
      entry.verb = "built from the discard";
      entry.payment = "free (The Mausoleum)";
    }
    if (action.type === "Discard") {
      entry.payment = `+${coins(view.discard_reward)} (2 base + commercial buildings)`;
    }
  }

  // VP is holistic and end-game-heavy, so a running delta is the only way a
  // player sees a move's scoring effect at the time it happens.
  if (actor && prev) {
    const idx = seatIndex(actor);
    const delta = step.views[idx].vp_now.total - prev.views[idx].vp_now.total;
    if (delta !== 0) {
      chips.unshift({ text: `${delta > 0 ? "+" : ""}${delta} VP`, kind: "plain" });
    }
  }

  entry.chips = chips;
  entry.reveals = reveals;
  entry.notes = notes;
  entry.key = key;

  if (entry.subject || entry.verb) out.push(entry);
  out.push(...systemAfter);
  return out;
}

export function romanAge(age: number): string {
  return ["", "I", "II", "III"][age] ?? String(age);
}

/** The one-line narration the move ticker shows, as its parts. */
export function tickerParts(entry: LogEntry): { chips: Chip[]; payment: string } {
  return { chips: entry.chips, payment: entry.payment };
}

/** Plain-text export of the whole log, one line per entry. */
export function exportText(entries: LogEntry[], seatNames: [string, string]): string {
  return entries
    .map((e) => {
      if (e.text) return `           -- ${e.text}`;
      const who = e.actor ? playerLabel(e.actor, seatNames) : "";
      const turn = e.turn === null ? "" : `T${e.turn}`;
      const chips = e.chips.map((c) => c.text).join(" · ");
      const reveals = e.reveals.length > 0 ? ` [revealed ${e.reveals.join(", ")}]` : "";
      return `${turn.padEnd(5)} ${who} ${e.verb} ${e.subject?.name ?? ""}${
        e.payment ? ` — ${e.payment}` : ""
      }${chips ? ` — ${chips}` : ""}${reveals}`;
    })
    .join("\n");
}

export { COIN };
