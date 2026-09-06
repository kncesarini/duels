// The wonder draft: the one full-screen phase, because there is no table
// state yet. It keeps the card colours and type language so nothing has to be
// re-learned when the table appears.

import type { Action } from "../generated/Action";
import type { Catalog } from "../generated/Catalog";
import type { Observation } from "../generated/Observation";
import { resourceEntries, wonderById } from "../lib/catalogHelpers";
import { describeWonderEffects, wonderTags } from "../lib/effectText";
import { Ico } from "../lib/icons";
import { seatIndex } from "../lib/cost";
import type { Resource } from "../generated/Resource";

interface Props {
  observation: Observation;
  catalog: Catalog;
  legal: Action[];
  seatNames: [string, string];
  onSubmit: (a: Action) => void;
  busy: boolean;
}

/** The pick order of one draft round: first, second, second, first. Printed
 * on the rulebook's draft diagram; shown here only as a caption. */
const ROUND_ORDER = [0, 1, 1, 0];

export default function WonderDraft({ observation, catalog, legal, seatNames, onSubmit, busy }: Props) {
  const round = observation.draft_step < 4 ? 1 : 2;
  const stepInRound = observation.draft_step % 4;
  const firstIdx = seatIndex(observation.draft_first);
  const order = ROUND_ORDER.map((o) => (round === 1 ? o : 1 - o)).map((o) => (o === 0 ? firstIdx : 1 - firstIdx));
  const picker = seatIndex(observation.current_player);

  return (
    <div className="draft" data-testid="draft">
      <div className="round">
        <b className="cz" style={{ fontSize: 15 }}>
          Draft round {round} of 2
        </b>{" "}
        · pick order: {order.map((o) => seatNames[o]).join(", ")} · now:{" "}
        <b style={{ color: picker === 0 ? "var(--you)" : "var(--opp)" }}>{seatNames[picker]}</b>
        {" · "}
        pick {stepInRound + 1} of 4
      </div>

      <div className="offer">
        {observation.offered_wonders.map((id) => {
          const wonder = wonderById(catalog, id);
          if (!wonder) return null;
          const action = legal.find((a) => a.type === "PickWonder" && a.wonder === id);
          const cost = resourceEntries(wonder.resource_cost);
          const base = cost.reduce((n, [, k]) => n + k * 2, 0) + wonder.coin_cost;
          return (
            <button
              key={id}
              type="button"
              className="wcard"
              disabled={!action || busy}
              onClick={() => action && onSubmit(action)}
              data-testid={`wonder-${id}`}
            >
              <h3>{wonder.name}</h3>
              <div style={{ display: "flex", gap: 3, alignItems: "center", flexWrap: "wrap" }}>
                {cost.flatMap(([r, n]) =>
                  Array.from({ length: n }, (_, i) => <Ico key={`${r}${i}`} id={r as Resource} className={r} title={r} />),
                )}
                {wonder.coin_cost > 0 && <Ico id="coin" className="coin" title="coins" />}
                <span className="mono" style={{ marginLeft: 6, color: "var(--mute)", fontSize: 10 }}>
                  ≈ {base}¢ at base prices
                </span>
              </div>
              <div>
                {describeWonderEffects(wonder).map((line) => (
                  <div key={line}>{line}</div>
                ))}
              </div>
              <div className="tags">
                {wonderTags(wonder).map((t) => (
                  <span key={t}>{t}</span>
                ))}
              </div>
            </button>
          );
        })}
      </div>

      <div className="slots">
        {[0, 1].map((i) => (
          <ContentsRow key={i} name={seatNames[i]} colour={i === 0 ? "var(--you)" : "var(--opp)"} ids={observation.players[i].wonders} catalog={catalog} />
        ))}
      </div>
      <div style={{ color: "var(--mute)", fontSize: 11 }}>
        Each player ends with four wonders; only seven of the eight can ever be built.
      </div>
    </div>
  );
}

function ContentsRow({
  name,
  colour,
  ids,
  catalog,
}: {
  name: string;
  colour: string;
  ids: string[];
  catalog: Catalog;
}) {
  return (
    <>
      <div style={{ color: colour, fontWeight: 600 }}>{name}</div>
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className={`dslot ${ids[i] ? "filled" : ""}`}>
          {ids[i] ? (wonderById(catalog, ids[i])?.name ?? ids[i]) : "—"}
        </div>
      ))}
    </>
  );
}
