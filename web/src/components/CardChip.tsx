import type { ReactNode } from "react";
import type { CardCatalogEntry } from "../generated/CardCatalogEntry";
import type { WonderCatalogEntry } from "../generated/WonderCatalogEntry";
import type { TokenCatalogEntry } from "../generated/TokenCatalogEntry";
import type { Catalog } from "../generated/Catalog";
import { CARD_TYPE_COLOR, CARD_TYPE_LABEL, RESOURCE_SYMBOL, SCIENCE_SYMBOL, resourceEntries } from "../lib/catalogHelpers";
import {
  describeCardEffects,
  describeCoinsByMajority,
  describeCoinsPerOwn,
  describePointsByMajority,
  describeTokenEffects,
  describeWonderEffects,
} from "../lib/effectText";

/**
 * Wraps a card/wonder/token chip so hovering it shows a small plain-English
 * panel of everything the game piece does. Chips only have room for a row
 * of terse icons (e.g. "1⚔", a chain arrow) - this is where a player who
 * doesn't already know the effect schema can see, in full sentences, what
 * "destroy" would destroy or how much military a shield icon is worth.
 */
function EffectTooltip({ lines, children }: { lines: string[]; children: ReactNode }) {
  if (lines.length === 0) return <>{children}</>;
  return (
    <div className="group/tip relative inline-block">
      {children}
      <div
        role="tooltip"
        className="pointer-events-none absolute left-1/2 top-full z-50 mt-1 w-56 -translate-x-1/2 rounded-md bg-stone-900 p-2 text-left text-[11px] font-normal leading-snug text-white opacity-0 shadow-lg transition-opacity duration-100 group-hover/tip:opacity-100"
      >
        <ul className="list-disc space-y-0.5 pl-3">
          {lines.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

interface CardChipProps {
  card: CardCatalogEntry;
  /** Used to resolve chain-link partner card names in the hover tooltip.
   * Every current call site has this in scope; omitting it just falls back
   * to showing the linked card's id instead of its printed name. */
  catalog?: Catalog;
  onClick?: () => void;
  disabled?: boolean;
  highlight?: boolean;
  title?: string;
}

export function CardChip({ card, catalog, onClick, disabled, highlight, title }: CardChipProps) {
  const clickable = !!onClick && !disabled;
  const effectLines = describeCardEffects(card, catalog);
  return (
    <EffectTooltip lines={effectLines}>
      <button
        type="button"
        title={title ?? card.name}
        onClick={onClick}
        disabled={!clickable}
        className={[
          "relative flex h-28 w-20 flex-col justify-between rounded-md border-2 p-1 text-[10px] leading-tight shadow-sm transition",
          CARD_TYPE_COLOR[card.kind],
          clickable ? "cursor-pointer hover:brightness-110 hover:shadow-md" : "cursor-default",
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
            {card.shields > 0 && (
              <span className="rounded bg-white/30 px-1" title={`+${card.shields} military shield${card.shields === 1 ? "" : "s"}`}>
                {card.shields}⚔
              </span>
            )}
            {card.coins > 0 && <span className="rounded bg-white/30 px-1">+{card.coins}c</span>}
            {card.science && <span className="rounded bg-white/30 px-1">{SCIENCE_SYMBOL[card.science]}</span>}
            {card.fixed_trade.length > 0 && (
              <span className="rounded bg-white/30 px-1" title={`Buy ${card.fixed_trade.join(", ")} for 1 coin each`}>
                🏪
              </span>
            )}
            {card.coins_per_own && (
              <span className="rounded bg-white/30 px-1" title={describeCoinsPerOwn(card.coins_per_own)}>
                +{card.coins_per_own[1]}c/bldg
              </span>
            )}
            {card.coins_by_majority && (
              <span className="rounded bg-white/30 px-1" title={describeCoinsByMajority(card.coins_by_majority)}>
                +{card.coins_by_majority[1]}c/maj
              </span>
            )}
            {card.points_by_majority && (
              <span
                className="rounded-full bg-white/80 px-1 font-bold text-black"
                title={describePointsByMajority(card.points_by_majority)}
              >
                {card.points_by_majority[1]}/maj
              </span>
            )}
          </div>
          <div className="flex items-center gap-0.5">
            {card.chain_to && (
              <span className="rounded bg-white/30 px-1" title="Unlocks a free build">
                ➜
              </span>
            )}
            {card.victory_points > 0 && (
              <span className="rounded-full bg-white/80 px-1 font-bold text-black">
                {card.victory_points}
              </span>
            )}
          </div>
        </div>
      </button>
    </EffectTooltip>
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
    <EffectTooltip lines={describeWonderEffects(wonder)}>
      <button
        type="button"
        onClick={onClick}
        disabled={!clickable}
        title={wonder.name}
        data-testid={`wonder-${wonder.id}`}
        className={[
          "flex min-h-[6rem] w-32 flex-col justify-between gap-1 rounded-md border-2 border-indigo-950 bg-indigo-800 p-1.5 text-[10px] leading-tight text-indigo-50 shadow-sm transition",
          clickable ? "cursor-pointer hover:brightness-110 hover:shadow-md" : "cursor-default",
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
        <div className="flex flex-wrap items-center gap-0.5">
          {wonder.shields > 0 && (
            <span
              className="rounded bg-white/30 px-1"
              title={`+${wonder.shields} military shield${wonder.shields === 1 ? "" : "s"}`}
            >
              {wonder.shields}⚔
            </span>
          )}
          {wonder.destroy && (
            <span className="rounded bg-white/30 px-1" title={`Destroy one of the opponent's ${CARD_TYPE_LABEL[wonder.destroy]} buildings`}>
              💥 {CARD_TYPE_LABEL[wonder.destroy]}
            </span>
          )}
          {wonder.coins > 0 && <span className="rounded bg-white/30 px-1">+{wonder.coins}c</span>}
          {wonder.opponent_loses_coins > 0 && (
            <span className="rounded bg-white/30 px-1" title="Opponent immediately loses coins">
              -{wonder.opponent_loses_coins}c opp
            </span>
          )}
          {wonder.produces_choice && <span className="rounded bg-white/30 px-1">choice</span>}
          {wonder.play_again && (
            <span className="rounded bg-white/30 px-1" title="Take an extra turn immediately">
              play again
            </span>
          )}
          {wonder.build_discarded_free && (
            <span className="rounded bg-white/30 px-1" title="Build a card from the discard pile for free">
              mausoleum
            </span>
          )}
          {wonder.choose_progress_token && (
            <span className="rounded bg-white/30 px-1" title="Draw 3 progress tokens set aside at setup and keep one">
              library
            </span>
          )}
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[9px] opacity-80">{built && "built"}</span>
          {wonder.victory_points > 0 && (
            <span className="rounded-full bg-white/80 px-1 font-bold text-black">
              {wonder.victory_points}
            </span>
          )}
        </div>
      </button>
    </EffectTooltip>
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
    <EffectTooltip lines={describeTokenEffects(token)}>
      <button
        type="button"
        onClick={onClick}
        disabled={!clickable}
        title={token.name}
        data-testid={`token-${token.id}`}
        className={[
          "flex h-16 w-16 flex-col items-center justify-center rounded-full border-2 border-teal-950 bg-teal-600 p-1 text-center text-[9px] leading-tight text-teal-50 shadow-sm transition",
          clickable ? "cursor-pointer hover:brightness-110 hover:shadow-md" : "cursor-default",
          disabled ? "opacity-40" : "",
        ].join(" ")}
      >
        <div className="line-clamp-2 font-semibold">{token.name}</div>
        {token.victory_points > 0 && <div>{token.victory_points}vp</div>}
      </button>
    </EffectTooltip>
  );
}
