import type { CardCatalogEntry } from "../generated/CardCatalogEntry";
import type { WonderCatalogEntry } from "../generated/WonderCatalogEntry";
import type { TokenCatalogEntry } from "../generated/TokenCatalogEntry";
import { CARD_TYPE_COLOR, RESOURCE_SYMBOL, SCIENCE_SYMBOL, resourceEntries } from "../lib/catalogHelpers";

interface CardChipProps {
  card: CardCatalogEntry;
  onClick?: () => void;
  disabled?: boolean;
  highlight?: boolean;
  title?: string;
}

export function CardChip({ card, onClick, disabled, highlight, title }: CardChipProps) {
  const clickable = !!onClick && !disabled;
  return (
    <button
      type="button"
      title={title ?? card.name}
      onClick={onClick}
      disabled={!clickable}
      className={[
        "relative flex h-28 w-20 flex-col justify-between rounded-md border-2 p-1 text-[10px] leading-tight shadow-sm transition",
        CARD_TYPE_COLOR[card.kind],
        clickable ? "cursor-pointer hover:scale-105 hover:shadow-md" : "cursor-default",
        highlight ? "ring-4 ring-yellow-300" : "",
        disabled ? "opacity-40" : "",
      ].join(" ")}
      data-testid={`card-${card.id}`}
    >
      <div className="font-semibold line-clamp-2">{card.name}</div>
      <div className="flex flex-wrap items-center gap-0.5">
        {card.coin_cost > 0 && (
          <span className="rounded bg-black/30 px-1">{card.coin_cost}c</span>
        )}
        {resourceEntries(card.resource_cost).map(([r, n]) => (
          <span key={r} className="rounded bg-black/30 px-1">
            {n}
            {RESOURCE_SYMBOL[r]}
          </span>
        ))}
      </div>
      <div className="flex items-center justify-between">
        <div className="flex flex-wrap gap-0.5">
          {resourceEntries(card.produces).map(([r, n]) => (
            <span key={r} className="rounded bg-white/30 px-1">
              +{n}
              {RESOURCE_SYMBOL[r]}
            </span>
          ))}
          {card.produces_choice && <span className="rounded bg-white/30 px-1">choice</span>}
          {card.shields > 0 && <span className="rounded bg-white/30 px-1">{card.shields}⚔</span>}
          {card.coins > 0 && <span className="rounded bg-white/30 px-1">+{card.coins}c</span>}
          {card.science && <span className="rounded bg-white/30 px-1">{SCIENCE_SYMBOL[card.science]}</span>}
        </div>
        {card.victory_points > 0 && (
          <span className="rounded-full bg-white/80 px-1 font-bold text-black">
            {card.victory_points}
          </span>
        )}
      </div>
    </button>
  );
}

export function FaceDownChip() {
  return (
    <div className="flex h-28 w-20 items-center justify-center rounded-md border-2 border-stone-700 bg-stone-500 text-2xl text-stone-300 shadow-sm">
      ?
    </div>
  );
}

export function WonderChip({
  wonder,
  onClick,
  disabled,
  built,
}: {
  wonder: WonderCatalogEntry;
  onClick?: () => void;
  disabled?: boolean;
  built?: boolean;
}) {
  const clickable = !!onClick && !disabled;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!clickable}
      title={wonder.name}
      data-testid={`wonder-${wonder.id}`}
      className={[
        "flex h-24 w-32 flex-col justify-between rounded-md border-2 border-indigo-950 bg-indigo-800 p-1.5 text-[10px] leading-tight text-indigo-50 shadow-sm transition",
        clickable ? "cursor-pointer hover:scale-105 hover:shadow-md" : "cursor-default",
        built ? "opacity-50 saturate-50" : "",
        disabled && !built ? "opacity-40" : "",
      ].join(" ")}
    >
      <div className="font-semibold line-clamp-2">{wonder.name}</div>
      <div className="flex flex-wrap gap-0.5">
        {wonder.coin_cost > 0 && <span className="rounded bg-black/30 px-1">{wonder.coin_cost}c</span>}
        {resourceEntries(wonder.resource_cost).map(([r, n]) => (
          <span key={r} className="rounded bg-black/30 px-1">
            {n}
            {RESOURCE_SYMBOL[r]}
          </span>
        ))}
      </div>
      <div className="flex items-center justify-between">
        <span className="text-[9px] opacity-80">
          {wonder.play_again && "play-again "}
          {wonder.destroy && "destroy "}
          {wonder.build_discarded_free && "mausoleum "}
          {wonder.choose_progress_token && "library "}
          {built && "built"}
        </span>
        {wonder.victory_points > 0 && (
          <span className="rounded-full bg-white/80 px-1 font-bold text-black">
            {wonder.victory_points}
          </span>
        )}
      </div>
    </button>
  );
}

export function TokenChip({
  token,
  onClick,
  disabled,
}: {
  token: TokenCatalogEntry;
  onClick?: () => void;
  disabled?: boolean;
}) {
  const clickable = !!onClick && !disabled;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!clickable}
      title={token.name}
      data-testid={`token-${token.id}`}
      className={[
        "flex h-16 w-16 flex-col items-center justify-center rounded-full border-2 border-teal-950 bg-teal-600 p-1 text-center text-[9px] leading-tight text-teal-50 shadow-sm transition",
        clickable ? "cursor-pointer hover:scale-105 hover:shadow-md" : "cursor-default",
        disabled ? "opacity-40" : "",
      ].join(" ")}
    >
      <div className="font-semibold">{token.name}</div>
      {token.victory_points > 0 && <div>{token.victory_points}vp</div>}
    </button>
  );
}
