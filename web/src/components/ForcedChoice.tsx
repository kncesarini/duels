// Every mid-turn choice uses one pattern: the tray turns gold, the banner
// says what the choice is, who is making it and which rule granted it, and
// the eligible objects light up *where they already live* on the table (the
// rail's tokens, the opponent's name chips, the discard list). No dialog ever
// covers the board.

import { useState } from "react";
import type { Action } from "../generated/Action";
import type { Catalog } from "../generated/Catalog";
import type { Observation } from "../generated/Observation";
import type { Player } from "../generated/Player";
import type { PlayerView } from "../generated/PlayerView";
import CardFace from "./CardFace";
import { cardById, tokenById } from "../lib/catalogHelpers";
import { describeTokenEffects } from "../lib/effectText";
import { coins, seatIndex } from "../lib/cost";
import { romanAge } from "../lib/log";
import { verb } from "../lib/text";
import { resourceEntries } from "../lib/catalogHelpers";
import type { Resource } from "../generated/Resource";

interface ChoiceModel {
  /** Gold banner headline. */
  title: string;
  /** The rule that grants the choice. */
  why: string;
  /** The rule text shown by "Why am I choosing this?". */
  rule: string;
  options: ChoiceOption[];
}

interface ChoiceOption {
  key: string;
  label: string;
  /** The button's own wording, when "Take <label>" would not read. */
  button?: string;
  /** One line of consequence, computed for this position. */
  detail: string;
  extra?: string;
  action: Action;
  node?: React.ReactNode;
}

interface Props {
  observation: Observation;
  catalog: Catalog;
  views: [PlayerView, PlayerView];
  legal: Action[];
  seatNames: [string, string];
  chooser: Player;
  mine: boolean;
  onSubmit: (a: Action) => void;
}

function buildChoice(
  observation: Observation,
  catalog: Catalog,
  views: [PlayerView, PlayerView],
  legal: Action[],
  seatNames: [string, string],
  chooser: Player,
): ChoiceModel | null {
  const chooserIdx = seatIndex(chooser);
  const otherIdx = chooserIdx === 0 ? 1 : 0;

  if (observation.phase === "choose_first_player") {
    const opts = legal.filter((a) => a.type === "ChooseFirstPlayer") as Array<{
      type: "ChooseFirstPlayer";
      player: Player;
    }>;
    return {
      title: `Age ${romanAge(observation.age)} begins · who goes first?`,
      why:
        observation.conflict === 0
          ? "the choice falls to whoever took the last card of the previous age"
          : "the choice falls to whoever is behind on the military track",
      rule: "At the start of Ages II and III the player behind on the military track decides who takes the first card; if the track is level, the player who took the last card of the previous age decides.",
      options: opts.map((a) => ({
        key: a.player,
        label: `${seatNames[seatIndex(a.player)]} ${verb(seatNames[seatIndex(a.player)], "goes", "go")} first`,
        button: `${seatNames[seatIndex(a.player)]} ${verb(seatNames[seatIndex(a.player)], "starts", "start")}`,
        detail: "They take the 1st, 3rd, 5th… card of the age.",
        action: a,
      })),
    };
  }

  const pending = observation.pending;
  if (!pending) return null;

  if (pending.type === "progress_token" || pending.type === "great_library_token") {
    const opts = legal.filter(
      (a) => a.type === "ChooseProgressToken" || a.type === "ChooseGreatLibraryToken",
    ) as Array<{ type: "ChooseProgressToken" | "ChooseGreatLibraryToken"; token: string }>;
    return {
      title:
        pending.type === "great_library_token"
          ? "The Great Library · choose one of three progress tokens"
          : "A science pair · choose a progress token",
      why: "The token is yours for the rest of the game",
      rule:
        pending.type === "great_library_token"
          ? "The Great Library draws three of the five progress tokens set aside during setup; its owner keeps one and the other two go back."
          : "Completing a pair of identical scientific symbols lets you take one progress token from the board immediately.",
      options: opts.map((a) => {
        const token = tokenById(catalog, a.token);
        return {
          key: a.token,
          label: token?.name ?? a.token,
          detail: token ? describeTokenEffects(token).join(" ") : "",
          extra: token ? whyThisToken(token.id, observation, views, chooserIdx, otherIdx, seatNames) : undefined,
          action: a,
        };
      }),
    };
  }

  if (pending.type === "destroy") {
    const opts = legal.filter((a) => a.type === "DestroyOpponentCard") as Array<{
      type: "DestroyOpponentCard";
      card: string;
    }>;
    return {
      title: `Destroy one of ${seatNames[otherIdx]}'s ${pending.card_type.replace(/_/g, " ")} buildings`,
      why: "The destroyed card goes to the discard pile",
      rule: "The Statue of Zeus destroys one opponent raw-material building; the Circus Maximus destroys one opponent manufactured-good building. The card goes to the discard pile.",
      options: opts.map((a) => {
        const card = cardById(catalog, a.card);
        const produced = card ? resourceEntries(card.produces) : [];
        const prices = produced
          .map(([r]) => `${r} currently costs ${seatNames[chooserIdx]} ${coins(views[chooserIdx].trade_prices[r as Resource])}`)
          .join("; ");
        return {
          key: a.card,
          label: card?.name ?? a.card,
          button: `Destroy ${card?.name ?? a.card}`,
          detail:
            produced.length > 0
              ? `${seatNames[otherIdx]} ${verb(seatNames[otherIdx], "loses", "lose")} ${produced.map(([r, n]) => `${n} ${r}`).join(", ")}.`
              : `${seatNames[otherIdx]} ${verb(seatNames[otherIdx], "loses", "lose")} this building.`,
          extra: prices || undefined,
          action: a,
        };
      }),
    };
  }

  // Mausoleum
  const opts = legal.filter((a) => a.type === "MausoleumBuild") as Array<{ type: "MausoleumBuild"; card: string }>;
  return {
    title: "The Mausoleum · build any discarded card for free",
    why: "The card joins your city at no cost",
    rule: "When The Mausoleum is constructed, its owner immediately builds one card from the discard pile for free.",
    options: opts.map((a) => {
      const card = cardById(catalog, a.card);
      return {
        key: a.card,
        label: card?.name ?? a.card,
        button: `Build ${card?.name ?? a.card}`,
        detail: card ? `Normally costs ${card.coin_cost > 0 ? coins(card.coin_cost) + " plus " : ""}its printed resources.` : "",
        action: a,
        node: card ? <CardFace card={card} freeOverride inline style={{ width: 76, height: 101 }} /> : undefined,
      };
    }),
  };
}

function whyThisToken(
  id: string,
  observation: Observation,
  views: [PlayerView, PlayerView],
  me: number,
  them: number,
  seatNames: [string, string],
): string {
  switch (id) {
    case "agriculture":
      return `You hold ${coins(observation.players[me].coins)} now.`;
    case "law":
      return `You have ${views[me].distinct_science} distinct symbols; Law counts as a seventh.`;
    case "mathematics":
      return `You hold ${observation.players[me].tokens.length} progress token${observation.players[me].tokens.length === 1 ? "" : "s"} so far.`;
    case "strategy":
      return `Every military building you build would push one extra step.`;
    case "economy":
      return `${seatNames[them]} pays ${Math.max(...Object.values(views[them].trade_prices))}¢ at most per bought resource, and that would come to you.`;
    case "urbanism":
      return `Chain builds would pay you coins each time.`;
    case "theology":
      return `You have ${observation.players[me].wonders.length - observation.players[me].wonders_built.length} unbuilt wonder(s), each of which would grant an extra turn.`;
    case "masonry":
      return `Civilian buildings would cost you two fewer resources.`;
    case "architecture":
      return `Wonders would cost you two fewer resources.`;
    case "philosophy":
      return `Flat victory points at game end.`;
    default:
      return "";
  }
}

export default function ForcedChoice({ observation, catalog, views, legal, seatNames, chooser, mine, onSubmit }: Props) {
  const [showRule, setShowRule] = useState(false);
  const model = buildChoice(observation, catalog, views, legal, seatNames, chooser);
  if (!model) return null;
  const chooserIdx = seatIndex(chooser);

  return (
    <div className={`choice ${mine ? "" : "opp"}`} data-testid="forced-choice">
      <div className="banner">
        <span>
          {mine ? model.title : `${seatNames[chooserIdx]} is choosing: ${model.title}`}{" "}
          <small>· {model.why}</small>
        </span>
        <button type="button" onClick={() => setShowRule((v) => !v)}>
          Why am I choosing this?
        </button>
      </div>
      {showRule && (
        <div style={{ padding: "6px 14px", fontSize: 11.5, color: "var(--fg2)", background: "var(--surf2)" }}>
          {model.rule}
        </div>
      )}
      <div className="opts">
        {model.options.map((o) => (
          <div className={`opt ${model.options.length === 1 ? "only" : ""}`} key={o.key}>
            {o.node}
            <div className="t">{o.label}</div>
            <div>{o.detail}</div>
            {o.extra && <div className="why">{o.extra}</div>}
            <button
              type="button"
              className={`btn ${model.options.length === 1 ? "primary" : ""}`}
              disabled={!mine}
              onClick={() => onSubmit(o.action)}
              data-testid={`choice-${o.key}`}
            >
              {o.button ?? (mine ? `Take ${o.label}` : o.label)}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
