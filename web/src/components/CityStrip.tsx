// One player's city. Both strips share the same four-column layout, so the
// same category sits at the same x-position for both players and can be
// compared by glancing vertically. The opponent's strip is quieter and a
// little smaller, but nothing is hidden or collapsed away.

import type { Catalog } from "../generated/Catalog";
import type { CardType } from "../generated/CardType";
import type { Player } from "../generated/Player";
import type { PublicPlayer } from "../generated/PublicPlayer";
import type { PlayerView } from "../generated/PlayerView";
import Counter from "./Counter";
import { cardById, CARD_TYPE_LABEL, typeColorVar } from "../lib/catalogHelpers";
import { compactWonderEffect } from "../lib/effectText";
import { coins } from "../lib/cost";
import { Ico, TypeGlyph } from "../lib/icons";
import { CARD_TYPES, RESOURCES, SCIENCES, SCIENCE_LABEL } from "../lib/iconData";
import { useHover } from "../lib/hover";
import { wonderById } from "../lib/catalogHelpers";
import type { WonderCostView } from "../generated/WonderCostView";

interface Props {
  seat: Player;
  player: PublicPlayer;
  view: PlayerView;
  catalog: Catalog;
  name: string;
  /** e.g. `Seat 2 · mcts-uct`. */
  subtitle: string;
  /** True for the brighter, taller strip (you, or the active hot-seat player). */
  bright: boolean;
  /** `you` puts the strip at the bottom of the table. */
  position: "top" | "bottom";
  conflict: number;
  /** Cards whose chip should pulse red as a destroy target. */
  destroyTargets?: Set<string>;
  onDestroy?: (cardId: string) => void;
  /** Card ids that arrived with the move being played back. */
  arrived: Set<string>;
  /** Card id the log entry under the cursor refers to. */
  ghosted: string | null;
  /** Resources whose chip should flash (production just changed). */
  flashed: Set<string>;
  className?: string;
}

export default function CityStrip({
  seat,
  player,
  view,
  catalog,
  name,
  subtitle,
  bright,
  position,
  conflict,
  destroyTargets,
  onDestroy,
  arrived,
  ghosted,
  flashed,
  className = "",
}: Props) {
  const showHover = useHover((s) => s.show);
  const seatClass = seat === "one" ? "seat-you" : "seat-opp";
  const military = seat === "one" ? conflict : -conflict;
  const distinct = view.distinct_science;

  const grouped = new Map<CardType, string[]>();
  for (const type of CARD_TYPES) grouped.set(type, []);
  for (const id of player.built) {
    const card = cardById(catalog, id);
    if (card) grouped.get(card.kind)?.push(id);
  }

  const unbuilt = player.wonders.filter((w) => !player.wonders_built.includes(w));
  const nextPair = SCIENCES.filter((_, i) => player.science[SCIENCE_INDEX[i]] === 1);

  return (
    <div
      className={`strip ${position === "bottom" ? "you" : ""} ${seatClass} ${bright ? "" : "quiet"} ${className}`}
      aria-label={`${name}'s city`}
    >
      <div className="who">
        <div className="name">
          {name}
          <span style={{ fontSize: 10, color: "var(--mute)", marginLeft: 6, textTransform: "uppercase", letterSpacing: ".06em" }}>
            {subtitle}
          </span>
        </div>
        <div className="bignum">
          <div>
            <span className="v" style={{ color: "var(--coin)" }}>
              <Counter value={player.coins} />
              {"¢"}
            </span>
            <span className="k">coins</span>
          </div>
          <div>
            <span className="v">
              <Counter value={view.vp_now.total} />
            </span>
            <span className="k">VP now</span>
          </div>
          <div>
            <span className="v" style={{ color: military > 0 ? "var(--military)" : "var(--mute)" }}>
              {military > 0 ? `+${military}` : military}
            </span>
            <span className="k">military</span>
          </div>
        </div>
      </div>

      <div>
        <h4 className="section">Production · price to buy</h4>
        <div className="res">
          {RESOURCES.map((r) => {
            const n = view.production[r];
            const price = view.trade_prices[r];
            return (
              <div key={r} className={`resc ${flashed.has(r) ? "flash" : ""}`} title={`${n} ${r} produced; buying one costs ${price} coins`}>
                <Ico id={r} className={r} title={r} />
                <span className={`p ${n ? "" : "z"}`}>{n}</span>
                <span className={`buy ${price <= 2 ? "cheap" : ""}`}>
                  buy <b>{coins(price)}</b>
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <h4 className="section">Science · {distinct}/6 distinct</h4>
        <div className="sci">
          {SCIENCES.map((sym, i) => {
            const n = player.science[SCIENCE_INDEX[i]];
            return (
              <div
                key={sym}
                className={`sym ${n ? "" : "none"} ${n >= 2 ? "pair" : ""}`}
                title={`${SCIENCE_LABEL[sym]}: ${n} held`}
              >
                <Ico id={sym} title={SCIENCE_LABEL[sym]} />
                <div className="pips2">
                  <i className={n >= 1 ? "f" : ""} />
                  <i className={n >= 2 ? "f" : ""} />
                </div>
              </div>
            );
          })}
        </div>
        <div className={`sciline ${distinct >= 4 ? "urgent" : ""}`}>
          {distinct >= 5 ? (
            <b>1 more distinct symbol wins the game</b>
          ) : distinct >= 4 ? (
            <b>2 more distinct symbols win the game</b>
          ) : nextPair.length > 0 ? (
            <>
              Pair <b>{nextPair.map((s) => SCIENCE_LABEL[s]).join(" or ")}</b> for a token
            </>
          ) : (
            <>No symbol is one card from a pair</>
          )}
        </div>
      </div>

      <div>
        <h4 className="section">
          Wonders · {player.wonders_built.length} of {player.wonders.length} built
        </h4>
        <div className="wonders">
          {player.wonders.map((id) => {
            const wonder = wonderById(catalog, id);
            if (!wonder) return null;
            const built = player.wonders_built.includes(id);
            const costView: WonderCostView | undefined = view.wonder_costs.find((w) => w.wonder === id);
            return (
              <div
                key={id}
                className={`wtile ${built ? "built" : ""} ${arrived.has(id) ? "just" : ""} ${!built && !costView ? "dead" : ""}`}
                data-dest={id}
                onMouseEnter={(e) => showHover("wonder", id, e.currentTarget)}
                onMouseLeave={() => showHover("wonder", id, null)}
              >
                <div className="wn">{wonder.name}</div>
                {built ? (
                  <>
                    <span className="check">✓ built</span>
                    <div className="weff">{compactWonderEffect(wonder)}</div>
                  </>
                ) : (
                  <div className="wc">
                    {costView?.plan.lines.flatMap((l) =>
                      Array.from({ length: l.required }, (_, i) => (
                        <Ico
                          key={`${l.resource}${i}`}
                          id={l.resource}
                          className={`${l.resource} ${i < l.produced + l.from_choice + l.from_discount ? "cov" : ""}`}
                          title={l.resource}
                        />
                      )),
                    )}
                    {costView && (
                      <span className={`wb ${costView.plan.affordable ? "ok" : "bad"}`}>{coins(costView.plan.coins)}</span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
          {unbuilt.length === 0 && player.wonders.length === 0 && <span className="railnote">not drafted yet</span>}
        </div>
      </div>

      <div className="built-row">
        {CARD_TYPES.map((type) => {
          const ids = grouped.get(type) ?? [];
          return (
            <div className="grp" key={type}>
              <span className="tl" style={{ color: typeColorVar(type) }}>
                <TypeGlyph type={type} /> {CARD_TYPE_LABEL[type]}
                {ids.length > 0 ? ` · ${ids.length}` : ""}
              </span>
              {ids.length === 0 ? (
                <span className="none">none</span>
              ) : (
                ids.map((id) => {
                  const card = cardById(catalog, id);
                  if (!card) return null;
                  const doomed = destroyTargets?.has(id) ?? false;
                  const cls = `chipname ${doomed ? "doomed interactive" : ""} ${arrived.has(id) ? "just" : ""} ${
                    ghosted === id ? "ghost" : ""
                  }`;
                  const style = { borderLeftColor: typeColorVar(type), ["--rim" as string]: seat === "one" ? "var(--you)" : "var(--opp)" };
                  return doomed && onDestroy ? (
                    <button key={id} type="button" className={cls} style={style} data-dest={id} onClick={() => onDestroy(id)}>
                      {card.name}
                    </button>
                  ) : (
                    <span
                      key={id}
                      className={cls}
                      style={style}
                      data-dest={id}
                      onMouseEnter={(e) => showHover("card", id, e.currentTarget)}
                      onMouseLeave={() => showHover("card", id, null)}
                    >
                      {card.name}
                    </span>
                  );
                })
              )}
            </div>
          );
        })}
        {player.tokens.length > 0 && (
          <div className="grp">
            <span className="tl">Progress tokens · {player.tokens.length}</span>
            {player.tokens.map((id) => (
              <span
                key={id}
                className={`chipname ${arrived.has(id) ? "just" : ""}`}
                style={{ borderLeftColor: "var(--scientific)" }}
                data-dest={id}
                onMouseEnter={(e) => showHover("token", id, e.currentTarget)}
                onMouseLeave={() => showHover("token", id, null)}
              >
                {catalog.tokens.find((t) => t.id === id)?.name ?? id}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** `PublicPlayer.science` is indexed by `duels_core::data::Science`'s own
 * order; `SCIENCES` is the display order. This maps one to the other. */
const SCIENCE_ORDER = ["mortar", "pendulum", "inkwell", "wheel", "sundial", "gyroscope", "balance"] as const;
const SCIENCE_INDEX = SCIENCES.map((s) => SCIENCE_ORDER.indexOf(s));
