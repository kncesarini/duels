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

export default function Game() {
  const payload = useGameStore((s) => s.payload);
  const catalog = useGameStore((s) => s.catalog);
  const status = useGameStore((s) => s.status);
  const errorMessage = useGameStore((s) => s.errorMessage);
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
        <WonderDraft observation={observation} catalog={catalog} legal={legal} onSubmit={submitAction} />
      ) : (
        <Structure
          observation={observation}
          catalog={catalog}
          actionableSlots={actionableSlots}
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
