// The hover panel: the full face at reading size, the effect in sentences,
// the chain both ways, and - the point of it - the cost for the lens player
// *and* the one-line cost for the other player, so "what would this cost
// them?" is never a calculation.

import type { Catalog } from "../generated/Catalog";
import type { Player } from "../generated/Player";
import type { PlayerView } from "../generated/PlayerView";
import CardFace from "./CardFace";
import { useHover } from "../lib/hover";
import { cardById, cardName, tokenById, wonderById } from "../lib/catalogHelpers";
import { describeCardEffects, describeTokenEffects, describeWonderEffects } from "../lib/effectText";
import { coins, otherPlayer, seatIndex, slotPlan, wonderPlan } from "../lib/cost";
import { Ico } from "../lib/icons";

interface Props {
  catalog: Catalog;
  views: [PlayerView, PlayerView];
  lens: Player;
  seatNames: [string, string];
  /** slot index by card id, for structure cards. */
  slotOfCard: Map<string, number>;
}

export default function CardHover({ catalog, views, lens, seatNames, slotOfCard }: Props) {
  const target = useHover((s) => s.target);
  if (!target) return null;

  const lensIdx = seatIndex(lens);
  const otherIdx = seatIndex(otherPlayer(lens));

  // Anchor beside the element, flipped to whichever side has room.
  const width = 430;
  const left = target.rect.right + width + 16 < window.innerWidth ? target.rect.right + 12 : Math.max(8, target.rect.left - width - 12);
  const top = Math.min(Math.max(8, target.rect.top - 40), Math.max(8, window.innerHeight - 330));

  let inner = null;

  if (target.kind === "card") {
    const card = cardById(catalog, target.id);
    if (!card) return null;
    const slot = slotOfCard.get(card.id);
    const mine = slot === undefined ? undefined : slotPlan(views[lensIdx], slot);
    const theirs = slot === undefined ? undefined : slotPlan(views[otherIdx], slot);
    inner = (
      <>
        <div style={{ width: 184, flex: "none" }}>
          <CardFace card={card} plan={mine} inline accessible style={{ width: 184, height: 244 }} />
        </div>
        <div style={{ minWidth: 0 }}>
          <h5>{card.name}</h5>
          <div style={{ color: "var(--mute)", fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".08em" }}>
            {card.kind.replace(/_/g, " ")} · age {card.age}
          </div>
          <ul>
            {describeCardEffects(card, catalog).map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
          {card.chain_to && (
            <div className="foot">
              <Ico id="link" /> Makes {cardName(catalog, card.chain_to)} free.
            </div>
          )}
          {theirs && (
            <div className="foot">
              {seatNames[otherIdx]}: {theirs.via_chain ? "free via a chain" : coins(theirs.coins)}
              {theirs.coins === 0 && !theirs.via_chain ? " (all produced)" : ""}
            </div>
          )}
          {target.uncovers !== undefined && target.uncovers > 0 && (
            <div className="foot">
              Taking this uncovers {target.uncovers} face-down card{target.uncovers === 1 ? "" : "s"}.
            </div>
          )}
        </div>
      </>
    );
  } else if (target.kind === "wonder") {
    const wonder = wonderById(catalog, target.id);
    if (!wonder) return null;
    const mine = wonderPlan(views[lensIdx], wonder.id);
    const theirs = wonderPlan(views[otherIdx], wonder.id);
    inner = (
      <div>
        <h5>{wonder.name}</h5>
        <ul>
          {describeWonderEffects(wonder).map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
        {mine && (
          <div className="foot">
            {seatNames[lensIdx]}: {coins(mine.coins)}
            {mine.affordable ? "" : " — not affordable"}
          </div>
        )}
        {theirs && (
          <div className="foot">
            {seatNames[otherIdx]}: {coins(theirs.coins)}
          </div>
        )}
      </div>
    );
  } else {
    const token = tokenById(catalog, target.id);
    if (!token) return null;
    inner = (
      <div>
        <h5>{token.name}</h5>
        <ul>
          {describeTokenEffects(token).map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </div>
    );
  }

  return (
    <div className="hovercard" style={{ left, top, width }} role="tooltip">
      {inner}
    </div>
  );
}
