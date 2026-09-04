import type { Action } from "../generated/Action";
import type { ActionCost } from "../generated/ActionCost";
import type { Catalog } from "../generated/Catalog";
import { cardById, wonderById } from "../lib/catalogHelpers";
import Modal from "./Modal";

interface ActionMenuProps {
  slot: number;
  cardId: string;
  legal: Action[];
  actionCosts: ActionCost[];
  catalog: Catalog;
  onSubmit: (action: Action) => void;
  onClose: () => void;
}

export default function ActionMenu({
  slot,
  cardId,
  legal,
  actionCosts,
  catalog,
  onSubmit,
  onClose,
}: ActionMenuProps) {
  const forSlot = legal.filter(
    (a) => (a.type === "Build" || a.type === "Discard" || a.type === "BuildWonder") && a.slot === slot,
  );
  const build = forSlot.find((a) => a.type === "Build");
  const discard = forSlot.find((a) => a.type === "Discard");
  const buildWonders = forSlot.filter((a) => a.type === "BuildWonder");

  const card = cardById(catalog, cardId);

  const buildCost = actionCosts.find((c) => c.type === "Build" && c.slot === slot);
  const discardCost = actionCosts.find((c) => c.type === "Discard" && c.slot === slot);

  return (
    <Modal onClose={onClose} title={card ? card.name : `Slot ${slot}`}>
      <div className="flex flex-col gap-2">
        {build && buildCost?.type === "Build" && (
          <button
            type="button"
            className="rounded-lg bg-emerald-700 px-4 py-3 text-left font-medium text-white hover:bg-emerald-800"
            onClick={() => onSubmit(build)}
          >
            Build
            {buildCost.via_chain ? (
              <span className="ml-2 text-emerald-200">free (chain)</span>
            ) : (
              <span className="ml-2 text-emerald-200">
                {buildCost.coins} coin{buildCost.coins === 1 ? "" : "s"}
                {buildCost.trade > 0 ? ` (${buildCost.trade} trade)` : ""}
              </span>
            )}
          </button>
        )}

        {discard && discardCost?.type === "Discard" && (
          <button
            type="button"
            className="rounded-lg bg-stone-600 px-4 py-3 text-left font-medium text-white hover:bg-stone-700"
            onClick={() => onSubmit(discard)}
          >
            Discard for coins
            <span className="ml-2 text-stone-200">+{discardCost.reward} coins</span>
          </button>
        )}

        {buildWonders.length > 0 && (
          <div className="mt-1">
            <div className="mb-1 text-sm font-semibold text-stone-600">Build a wonder with this card:</div>
            <div className="flex flex-col gap-2">
              {buildWonders.map((action) => {
                if (action.type !== "BuildWonder") return null;
                const wonder = wonderById(catalog, action.wonder);
                const cost = actionCosts.find(
                  (c) => c.type === "BuildWonder" && c.slot === slot && c.wonder === action.wonder,
                );
                return (
                  <button
                    key={action.wonder}
                    type="button"
                    className="rounded-lg bg-indigo-700 px-4 py-3 text-left font-medium text-white hover:bg-indigo-800"
                    onClick={() => onSubmit(action)}
                  >
                    {wonder?.name ?? action.wonder}
                    {cost?.type === "BuildWonder" && (
                      <span className="ml-2 text-indigo-200">
                        {cost.coins} coin{cost.coins === 1 ? "" : "s"}
                        {cost.trade > 0 ? ` (${cost.trade} trade)` : ""}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {forSlot.length === 0 && <p className="text-sm text-stone-500">No actions available here.</p>}
      </div>
    </Modal>
  );
}
