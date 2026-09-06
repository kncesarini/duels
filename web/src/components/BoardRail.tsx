// The shared board: the military track drawn vertically so the opponent's
// capital is at the top and yours at the bottom (matching the strips), the
// five progress-token slots including the ones already claimed, the
// wonders-built counter, and the discard pile listed by name.

import type { Catalog } from "../generated/Catalog";
import type { Observation } from "../generated/Observation";
import type { Player } from "../generated/Player";
import { cardById, typeColorVar } from "../lib/catalogHelpers";
import { useHover } from "../lib/hover";
import { coins } from "../lib/cost";
import { possessive } from "../lib/text";

const MAX_WONDERS_BUILT = 7;

interface Props {
  observation: Observation;
  catalog: Catalog;
  /** The seat drawn at the bottom of the table. */
  bottom: Player;
  seatNames: [string, string];
  /** Tokens the current chooser may take. */
  litTokens?: Set<string>;
  onToken?: (id: string) => void;
  /** Discard cards the Mausoleum may build. */
  litDiscard?: Set<string>;
  onDiscardCard?: (id: string) => void;
  ghosted: string | null;
  className?: string;
}

export default function BoardRail({
  observation,
  catalog,
  bottom,
  seatNames,
  litTokens,
  onToken,
  litDiscard,
  onDiscardCard,
  ghosted,
  className = "",
}: Props) {
  const showHover = useHover((s) => s.show);
  const cap = catalog.military.capital_distance;
  // `conflict` is positive when player one is ahead. Redraw it as "distance
  // towards the bottom player's capital is negative".
  const towardsBottom = bottom === "one" ? -observation.conflict : observation.conflict;
  const cells = [];
  const bottomIdx = bottom === "one" ? 0 : 1;
  const topIdx = bottomIdx === 0 ? 1 : 0;

  for (let d = -cap; d <= cap; d++) {
    // d = -cap is the top player's capital.
    const isCap = Math.abs(d) === cap;
    const loot = catalog.military.loot.findIndex(([dist]) => dist === Math.abs(d));
    // A loot token at distance `dist` on the side towards the bottom player
    // is collected by the top player pushing there.
    const pusher = d > 0 ? bottomIdx : topIdx;
    const taken = loot >= 0 && d !== 0 ? observation.loot_taken[pusher][loot] : false;
    cells.push(
      <div
        key={d}
        className={[
          "cell",
          isCap ? `cap ${d < 0 ? "o" : "y"}` : "",
          d === 0 ? "mid" : "",
          loot >= 0 && d !== 0 && !isCap ? `loot ${taken ? "taken" : ""}` : "",
          towardsBottom === d ? "pawn" : "",
        ]
          .filter(Boolean)
          .join(" ")}
        style={towardsBottom === d ? { ["--rim" as string]: d > 0 ? "var(--opp)" : "var(--you)" } : undefined}
        title={
          isCap
            ? `${d < 0 ? seatNames[topIdx] : seatNames[bottomIdx]} capital — reaching it wins the game`
            : loot >= 0 && d !== 0
              ? `Loot: ${coins(catalog.military.loot[loot][1])} forfeited${taken ? " (already taken)" : ""}`
              : `distance ${Math.abs(d)}`
        }
      />,
    );
  }

  const leader =
    observation.conflict === 0 ? null : observation.conflict > 0 ? 0 : 1;
  const lead = Math.abs(observation.conflict);
  const nextLoot = catalog.military.loot
    .map(([dist, c]) => ({ dist, c }))
    .filter((l) => l.dist > lead)
    .sort((a, b) => a.dist - b.dist)[0];

  const wondersBuilt = observation.players[0].wonders_built.length + observation.players[1].wonders_built.length;
  // The observation carries who built what, not the interleaved build order,
  // so the pips group by owner and then show the remaining empty slots.
  const pips: Array<"y" | "o" | ""> = [
    ...observation.players[bottomIdx].wonders_built.map(() => "y" as const),
    ...observation.players[topIdx].wonders_built.map(() => "o" as const),
    ...Array.from({ length: Math.max(0, MAX_WONDERS_BUILT - wondersBuilt) }, () => "" as const),
  ];

  return (
    <div className={`rail ${className}`} aria-label="Shared board">
      <div>
        <h4 className="section">Military</h4>
        <div className="track">
          <div className="lbl">
            <span>{seatNames[topIdx]} capital</span>
            <span className="pen">−{catalog.military.loot[1]?.[1] ?? 5}¢</span>
            <span className="pen">−{catalog.military.loot[0]?.[1] ?? 2}¢</span>
            <span />
            <span className="pen">−{catalog.military.loot[0]?.[1] ?? 2}¢</span>
            <span className="pen">−{catalog.military.loot[1]?.[1] ?? 5}¢</span>
            <span>{seatNames[bottomIdx]} capital</span>
          </div>
          <div className="cells">{cells}</div>
          <div className="side">
            <span>WIN</span>
            <span>{catalog.military.loot[1]?.[0] ?? 6}</span>
            <span>{catalog.military.loot[0]?.[0] ?? 3}</span>
            <span>0</span>
            <span>{catalog.military.loot[0]?.[0] ?? 3}</span>
            <span>{catalog.military.loot[1]?.[0] ?? 6}</span>
            <span>WIN</span>
          </div>
        </div>
        <div className="mil-summary">
          {leader === null ? (
            <b>Level</b>
          ) : (
            <b style={{ color: leader === bottomIdx ? "var(--you)" : "var(--opp)" }}>
              {seatNames[leader]} +{lead}
            </b>
          )}
          {leader !== null && (
            <>
              {" "}
              toward {possessive(seatNames[leader === 0 ? 1 : 0])} capital
            </>
          )}
          <br />
          <span style={{ color: "var(--mute)" }}>
            {nextLoot
              ? `${nextLoot.dist - lead} more step${nextLoot.dist - lead === 1 ? "" : "s"}: the trailing player loses ${coins(nextLoot.c)}`
              : `${cap - lead} more step${cap - lead === 1 ? "" : "s"} wins outright`}
          </span>
        </div>
      </div>

      <div>
        <h4 className="section">Progress tokens · {observation.board_tokens.length} of 5 left</h4>
        <div className="tokens">
          {catalog.tokens
            .filter(
              (t) =>
                observation.board_tokens.includes(t.id) ||
                observation.players[0].tokens.includes(t.id) ||
                observation.players[1].tokens.includes(t.id),
            )
            .slice(0, 8)
            .map((t) => {
              const onBoard = observation.board_tokens.includes(t.id);
              const owner = observation.players[0].tokens.includes(t.id)
                ? 0
                : observation.players[1].tokens.includes(t.id)
                  ? 1
                  : null;
              const lit = litTokens?.has(t.id) ?? false;
              const cls = `tok ${onBoard ? "" : "empty"} ${lit ? "lit clickable" : ""}`;
              const label = onBoard ? t.name : `${t.name} → ${owner === null ? "?" : seatNames[owner]}`;
              return lit && onToken ? (
                <button key={t.id} type="button" className={cls} onClick={() => onToken(t.id)}>
                  {t.name}
                </button>
              ) : (
                <div
                  key={t.id}
                  className={cls}
                  onMouseEnter={(e) => showHover("token", t.id, e.currentTarget)}
                  onMouseLeave={() => showHover("token", t.id, null)}
                >
                  {label}
                </div>
              );
            })}
        </div>
        <div className="railnote">Claim one by pairing a science symbol.</div>
      </div>

      <div>
        <h4 className="section">
          Wonders built · {wondersBuilt} of {MAX_WONDERS_BUILT}
        </h4>
        <div className="pips">
          {pips.map((p, i) => (
            <i key={i} className={p} />
          ))}
        </div>
        <div className={`railnote ${wondersBuilt >= MAX_WONDERS_BUILT ? "warn" : ""}`}>
          {wondersBuilt >= MAX_WONDERS_BUILT
            ? "No more wonders can be built."
            : "The 7th wonder built ends the last player's 4th."}
        </div>
      </div>

      <div>
        <h4 className="section">
          Discard · <span className="mono">{observation.discard.length}</span> cards
        </h4>
        <div style={{ marginTop: 5 }}>
          {observation.discard.length === 0 && <span className="railnote">nothing discarded yet</span>}
          {observation.discard.map((id) => {
            const card = cardById(catalog, id);
            if (!card) return null;
            const lit = litDiscard?.has(id) ?? false;
            const cls = `chipname ${lit ? "interactive" : ""} ${ghosted === id ? "ghost" : ""}`;
            const style = { borderLeftColor: typeColorVar(card.kind), animation: lit ? "pulse-ok 1.2s ease-in-out infinite" : undefined };
            return lit && onDiscardCard ? (
              <button key={id} type="button" className={cls} style={style} data-dest={id} onClick={() => onDiscardCard(id)}>
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
          })}
        </div>
        <div className="railnote">The Mausoleum can build from here.</div>
      </div>
    </div>
  );
}
