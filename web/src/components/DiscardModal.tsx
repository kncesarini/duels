import type { Catalog } from "../generated/Catalog";
import { CardChip } from "./CardChip";
import { cardById } from "../lib/catalogHelpers";
import Modal from "./Modal";

interface DiscardModalProps {
  discard: string[];
  catalog: Catalog;
  onClose: () => void;
}

export default function DiscardModal({ discard, catalog, onClose }: DiscardModalProps) {
  return (
    <Modal title={`Discard pile (${discard.length})`} onClose={onClose}>
      {discard.length === 0 ? (
        <p className="text-sm text-stone-500">Empty.</p>
      ) : (
        <div className="flex flex-wrap justify-center gap-3">
          {discard.map((id) => {
            const card = cardById(catalog, id);
            if (!card) return null;
            return <CardChip key={id} card={card} />;
          })}
        </div>
      )}
    </Modal>
  );
}
