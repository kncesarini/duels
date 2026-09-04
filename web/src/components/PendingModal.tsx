import type { Action } from "../generated/Action";
import type { Observation } from "../generated/Observation";
import type { Catalog } from "../generated/Catalog";
import { CardChip } from "./CardChip";
import { TokenChip } from "./CardChip";
import { cardById, tokenById } from "../lib/catalogHelpers";
import Modal from "./Modal";

interface PendingModalProps {
  observation: Observation;
  catalog: Catalog;
  legal: Action[];
  onSubmit: (action: Action) => void;
}

/** Renders whichever pending-choice modal applies right now, or `null` if
 * none does (a normal turn). Every option shown here comes straight from
 * `legal`; nothing is filtered or computed client-side. */
export default function PendingModal({ observation, catalog, legal, onSubmit }: PendingModalProps) {
  if (observation.phase === "choose_first_player") {
    const options = legal.filter((a): a is Action & { type: "ChooseFirstPlayer" } => a.type === "ChooseFirstPlayer");
    return (
      <Modal title={`Age ${observation.age}: who begins?`}>
        <p className="mb-3 text-sm text-stone-600">
          The conflict pawn favours{" "}
          {observation.conflict > 0 ? "Player One" : observation.conflict < 0 ? "Player Two" : "neither player"}, so
          the militarily weaker player chooses who takes the first turn.
        </p>
        <div className="flex gap-3">
          {options.map((a) => (
            <button
              key={a.player}
              type="button"
              className="flex-1 rounded-lg bg-amber-700 px-4 py-3 font-semibold text-white hover:bg-amber-800"
              onClick={() => onSubmit(a)}
            >
              {a.player === "one" ? "Player One" : "Player Two"}
            </button>
          ))}
        </div>
      </Modal>
    );
  }

  const pending = observation.pending;
  if (!pending) return null;

  if (pending.type === "progress_token") {
    const options = legal.filter(
      (a): a is Action & { type: "ChooseProgressToken" } => a.type === "ChooseProgressToken",
    );
    return (
      <Modal title="Choose a progress token">
        <div className="flex flex-wrap justify-center gap-3">
          {options.map((a) => {
            const token = tokenById(catalog, a.token);
            if (!token) return null;
            return <TokenChip key={a.token} token={token} onClick={() => onSubmit(a)} />;
          })}
        </div>
      </Modal>
    );
  }

  if (pending.type === "great_library_token") {
    const options = legal.filter(
      (a): a is Action & { type: "ChooseGreatLibraryToken" } => a.type === "ChooseGreatLibraryToken",
    );
    return (
      <Modal title="The Great Library: keep one token">
        <p className="mb-3 text-sm text-stone-600">
          Drawn from the tokens set aside at setup; the other two leave the game.
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          {options.map((a) => {
            const token = tokenById(catalog, a.token);
            if (!token) return null;
            return <TokenChip key={a.token} token={token} onClick={() => onSubmit(a)} />;
          })}
        </div>
      </Modal>
    );
  }

  if (pending.type === "destroy") {
    const options = legal.filter(
      (a): a is Action & { type: "DestroyOpponentCard" } => a.type === "DestroyOpponentCard",
    );
    return (
      <Modal title="Destroy an opponent building">
        <div className="flex flex-wrap justify-center gap-3">
          {options.map((a) => {
            const card = cardById(catalog, a.card);
            if (!card) return null;
            return <CardChip key={a.card} card={card} onClick={() => onSubmit(a)} />;
          })}
        </div>
      </Modal>
    );
  }

  if (pending.type === "mausoleum_build") {
    const options = legal.filter((a): a is Action & { type: "MausoleumBuild" } => a.type === "MausoleumBuild");
    return (
      <Modal title="The Mausoleum: build from the discard pile, for free">
        <div className="flex flex-wrap justify-center gap-3">
          {options.map((a) => {
            const card = cardById(catalog, a.card);
            if (!card) return null;
            return <CardChip key={a.card} card={card} onClick={() => onSubmit(a)} />;
          })}
        </div>
      </Modal>
    );
  }

  return null;
}
