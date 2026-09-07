// The card face. There is no artwork, so the face is an information panel:
// a type band (colour + glyph + printed name), one large effect glyph, and a
// cost row rendered for whichever player the cost lens is pointing at.
//
// The same component draws every size - structure, tray, draft, discard,
// hover - so nothing has to be re-learned when a card moves.

import type { CSSProperties } from "react";
import type { CardCatalogEntry } from "../generated/CardCatalogEntry";
import type { CostPlan } from "../generated/CostPlan";
import type { Resource } from "../generated/Resource";
import { compactCardEffect } from "../lib/effectText";
import { Ico, TypeGlyph } from "../lib/icons";
import { TYPE_MARK } from "../lib/iconData";
import { costBadge } from "../lib/cost";
import { CARD_TYPE_LABEL, resourceEntries, typeColorVar } from "../lib/catalogHelpers";

export interface CardFaceProps {
  card: CardCatalogEntry;
  /** The cost for the lens player, from the server. Omitted where a printed
   * cost is all that can be shown (the draft, a card nobody can build yet). */
  plan?: CostPlan;
  /** Renders the badge in the opponent's magenta so the two are never
   * confused with each other. */
  oppLens?: boolean;
  /** Show `FREE` instead of a cost (the Mausoleum's discard build). */
  freeOverride?: boolean;
  covered?: boolean;
  accessible?: boolean;
  selected?: boolean;
  unaffordable?: boolean;
  hinted?: boolean;
  revealing?: boolean;
  justTaken?: boolean;
  style?: CSSProperties;
  className?: string;
  onClick?: () => void;
  onHover?: (el: HTMLElement | null) => void;
  /** Suppress the absolute positioning the structure uses. */
  inline?: boolean;
  title?: string;
}

function CostRow({ card, plan, oppLens, freeOverride }: Pick<CardFaceProps, "card" | "plan" | "oppLens" | "freeOverride">) {
  if (freeOverride) {
    return (
      <div className="cf">
        <span className="bd">FREE</span>
      </div>
    );
  }
  if (plan?.via_chain) {
    return (
      <div className="cf">
        <span className="bd chain">
          <Ico id="link" /> free
        </span>
        <span className="via">via a chain</span>
      </div>
    );
  }

  const units: Array<{ resource: Resource; covered: boolean; price: number }> = [];
  if (plan) {
    for (const line of plan.lines) {
      const free = line.produced + line.from_choice + line.from_discount;
      for (let i = 0; i < line.required; i++) {
        units.push({ resource: line.resource, covered: i < free, price: line.unit_price });
      }
    }
  } else {
    for (const [r, n] of resourceEntries(card.resource_cost)) {
      for (let i = 0; i < n; i++) units.push({ resource: r as Resource, covered: false, price: 0 });
    }
  }

  const badge = plan ? costBadge(plan) : null;
  const printedFree = units.length === 0 && card.coin_cost === 0;

  return (
    <div className="cf">
      {units.map((u, i) => (
        <span key={i} className={`r ${u.covered ? "cov" : "miss"}`} data-p={u.covered ? undefined : u.price || undefined}>
          <Ico id={u.resource} className={u.resource} title={u.resource} />
        </span>
      ))}
      {card.coin_cost > 0 && (
        <span className="r">
          <Ico id="coin" className="coin" title={`${card.coin_cost} coins`} />
        </span>
      )}
      {badge ? (
        <span className={`bd ${badge.kind}${oppLens && badge.kind === "ok" ? " opp" : ""}`}>{badge.text}</span>
      ) : (
        <span className="bd">{printedFree ? "FREE" : `${card.coin_cost}¢`}</span>
      )}
    </div>
  );
}

function Body({ card }: { card: CardCatalogEntry }) {
  const produced = resourceEntries(card.produces);
  if (produced.length > 0) {
    return (
      <>
        {produced.map(([r, n]) => (
          <span key={r} style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
            <span className="big">+{n}</span>
            <Ico id={r} className={r} size={22} title={r} />
          </span>
        ))}
      </>
    );
  }
  if (card.science) {
    return (
      <>
        <Ico id={card.science} size={26} className="sci" title={card.science} />
        {card.victory_points > 0 && (
          <div className="vpb" style={{ width: 20, height: 20, fontSize: 11 }}>
            {card.victory_points}
          </div>
        )}
      </>
    );
  }
  if (card.shields > 0) {
    return (
      <>
        {Array.from({ length: card.shields }, (_, i) => (
          <Ico key={i} id="shield" className="shield" size={22} title="shield" />
        ))}
      </>
    );
  }
  if (card.kind === "civilian" && card.victory_points > 0) {
    return <div className="vpb">{card.victory_points}</div>;
  }
  const text = compactCardEffect(card);
  if (card.coins > 0) {
    return (
      <>
        <span className="big">+{card.coins}</span>
        <Ico id="coin" className="coin" size={22} title="coins" />
        {text && <div className="txt">{text}</div>}
      </>
    );
  }
  if (card.victory_points > 0) {
    return (
      <>
        <div className="vpb">{card.victory_points}</div>
        {text && <div className="txt">{text}</div>}
      </>
    );
  }
  return <div className="txt">{text || CARD_TYPE_LABEL[card.kind]}</div>;
}

export default function CardFace({
  card,
  plan,
  oppLens,
  freeOverride,
  covered,
  accessible,
  selected,
  unaffordable,
  hinted,
  revealing,
  justTaken,
  style,
  className = "",
  onClick,
  onHover,
  inline,
  title,
}: CardFaceProps) {
  const classes = [
    "card",
    inline ? "static" : "",
    covered ? "covered" : "",
    accessible ? "acc" : "",
    unaffordable ? "no" : "",
    selected ? "sel" : "",
    hinted ? "hint" : "",
    revealing ? "revealing" : "",
    justTaken ? "just" : "",
    onClick ? "clickable" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  const label = `${card.name}, ${CARD_TYPE_LABEL[card.kind]}, age ${card.age}${
    plan ? `, ${plan.via_chain ? "free via a chain" : `costs ${plan.coins} coins`}${plan.affordable ? "" : ", not affordable"}` : ""
  }`;

  const content = (
    <>
      <div
        className="type-band"
        style={{ background: typeColorVar(card.kind), color: card.kind === "commercial" ? "#2A2410" : "#fff", ["--mark" as string]: TYPE_MARK[card.kind] }}
      >
        <TypeGlyph type={card.kind} />
        <span className="nm">{card.name}</span>
        {card.chain_to && (
          <span className="ln" title="Owning this makes a later card free">
            <Ico id="link" />
          </span>
        )}
      </div>
      <div className="cb">
        <Body card={card} />
      </div>
      <CostRow card={card} plan={plan} oppLens={oppLens} freeOverride={freeOverride} />
    </>
  );

  if (onClick) {
    return (
      <button
        type="button"
        className={classes}
        style={style}
        onClick={onClick}
        onMouseEnter={(e) => onHover?.(e.currentTarget)}
        onMouseLeave={() => onHover?.(null)}
        onFocus={(e) => onHover?.(e.currentTarget)}
        onBlur={() => onHover?.(null)}
        aria-label={label}
        title={title}
        data-card={card.id}
      >
        {content}
      </button>
    );
  }
  return (
    <div
      className={classes}
      style={style}
      onMouseEnter={(e) => onHover?.(e.currentTarget)}
      onMouseLeave={() => onHover?.(null)}
      role="img"
      aria-label={label}
      title={title}
      data-card={card.id}
    >
      {content}
    </div>
  );
}

/** The face-down back: the age glyph, and - for a guild - the fact that this
 * back is purple. That really is all either player is entitled to know: a
 * guild card back is visually distinct in the physical game, so which
 * face-down Age III slots hold a guild is public information (R-110), while
 * *which* guild stays hidden until the slot is revealed. */
export function CardBack({
  age,
  style,
  inline,
  isGuild,
}: {
  age: number;
  style?: CSSProperties;
  inline?: boolean;
  /** This slot's back is purple: it holds one of the three guilds. */
  isGuild?: boolean;
}) {
  return (
    <div
      className={`card back ${isGuild ? "guild" : ""} ${inline ? "static" : ""}`}
      style={style}
      role="img"
      aria-label={isGuild ? `Face-down age ${age} guild card` : `Face-down age ${age} card`}
    >
      <div className="age">{["", "I", "II", "III"][age] ?? age}</div>
      <div className="q">{isGuild ? "guild" : "face down"}</div>
    </div>
  );
}
