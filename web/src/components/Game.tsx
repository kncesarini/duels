import { useState } from "react";
import { useGameStore } from "../store";
import Board from "./Board";
import Structure from "./Structure";
import { actionableSlotsFrom } from "../lib/actions";
import ActionMenu from "./ActionMenu";
import PlayerPanel from "./PlayerPanel";
import PendingModal from "./PendingModal";
import DiscardModal from "./DiscardModal";
import GameOver from "./GameOver";
import WonderDraft from "./WonderDraft";

// A stable empty-set reference (see the `actionableSlots` prop below) so
// re-renders while `pending` don't allocate a new `Set` every time.
const EMPTY_SLOTS = new Set<number>();

export default function Game() {
  const payload = useGameStore((s) => s.payload);
  const catalog = useGameStore((s) => s.catalog);
  const status = useGameStore((s) => s.status);
  const errorMessage = useGameStore((s) => s.errorMessage);
  const pending = useGameStore((s) => s.pending);
  const mode = useGameStore((s) => s.mode);
  const submitAction = useGameStore((s) => s.submitAction);
  const leaveGame = useGameStore((s) => s.leaveGame);

  const [selectedSlot, setSelectedSlot] = useState<number | null>(null);
  const [showDiscard, setShowDiscard] = useState(false);

  if (!catalog || !payload) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-stone-500">
          {status === "error" ? errorMessage ?? "Connection error" : "Connecting..."}
        </p>
      </div>
    );
  }

  const { observation, legal_actions: legal, action_costs: actionCosts, breakdown } = payload;
  const actionableSlots = actionableSlotsFrom(legal);

  return (
    <div className="mx-auto max-w-5xl space-y-4 p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">7 Wonders Duel</h1>
        <button
          type="button"
          onClick={leaveGame}
          data-testid="nav-leave-game"
          className="text-sm text-stone-500 underline"
        >
          Leave game
        </button>
      </div>

      {errorMessage && (
        <div className="rounded bg-red-100 px-3 py-2 text-sm text-red-700" role="alert">
          {errorMessage}
        </div>
      )}

      {status === "reconnecting" && (
        <div className="rounded bg-amber-100 px-3 py-2 text-sm text-amber-800" role="status">
          Connection lost - reconnecting...
        </div>
      )}

      {/* The server resolves any agent turns synchronously before it
          replies (see `room::drive_agents`), so while `pending` is true the
          opponent may be mid-search (`alphabeta`/`mcts-uct` can take up to
          ~1s under the server's interactive Budget) rather than the reply
          just being slow over the network. Without this, that stretch reads
          as a frozen board with no feedback - the same failure mode the
          wonder-draft "Submitting..." indicator exists to avoid, just for
          the main play phase instead of the draft. */}
      {pending && (
        <div className="rounded bg-stone-100 px-3 py-2 text-center text-sm text-stone-600" role="status">
          {mode === "bot" ? "Opponent is thinking..." : "Submitting..."}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <PlayerPanel
          label="Player One"
          player={observation.players[0]}
          catalog={catalog}
          active={observation.current_player === "one"}
          militaryLeader={observation.conflict > 0}
        />
        <PlayerPanel
          label="Player Two"
          player={observation.players[1]}
          catalog={catalog}
          active={observation.current_player === "two"}
          militaryLeader={observation.conflict < 0}
        />
      </div>

      <Board observation={observation} catalog={catalog} onOpenDiscard={() => setShowDiscard(true)} />

      {observation.phase === "wonder_draft" ? (
        <WonderDraft observation={observation} catalog={catalog} legal={legal} onSubmit={submitAction} pending={pending} />
      ) : (
        <Structure
          observation={observation}
          catalog={catalog}
          // `legal_actions` (and thus `actionableSlots`) still reflects the
          // state from before the in-flight action, since a fresh set only
          // arrives with the next `StatePayload` - hide affordances while
          // `pending` so a slot can't be clicked twice (once for real, once
          // against stale data) while the opponent is still resolving.
          actionableSlots={pending ? EMPTY_SLOTS : actionableSlots}
          onSlotClick={(slot) => setSelectedSlot(slot)}
        />
      )}

      {selectedSlot !== null &&
        (() => {
          const slotView = observation.slots[selectedSlot];
          const cardId = slotView.state === "face_up" ? slotView.card : null;
          if (!cardId) return null;
          return (
            <ActionMenu
              slot={selectedSlot}
              cardId={cardId}
              legal={legal}
              actionCosts={actionCosts}
              catalog={catalog}
              onSubmit={(action) => {
                submitAction(action);
                setSelectedSlot(null);
              }}
              onClose={() => setSelectedSlot(null)}
            />
          );
        })()}

      <PendingModal observation={observation} catalog={catalog} legal={legal} onSubmit={submitAction} />

      {showDiscard && (
        <DiscardModal discard={observation.discard} catalog={catalog} onClose={() => setShowDiscard(false)} />
      )}

      {observation.result && <GameOver result={observation.result} breakdown={breakdown} onLeave={leaveGame} />}
    </div>
  );
}
