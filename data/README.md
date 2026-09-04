# Game data

This directory holds factual base-game data for 7 Wonders Duel (no Agora,
no Pantheon expansion content — see `docs/adr/0005-base-game-scope-no-expansions.md`):
names, costs, and structured effect descriptions for cards, wonders, and
tokens. This is factual/mechanical game data (not copyrighted artwork or
verbatim rulebook text), which is why it's fine to include directly in this
public repo.

**Provenance and confidence.** This data was cross-referenced against
[fromi/7-wonders-duel](https://github.com/fromi/7-wonders-duel), a
from-scratch open-source Kotlin implementation of the game whose card/
wonder/progress-token definitions (`Building.kt`, `Deck.kt`, `Wonder.kt`,
`ProgressToken.kt`) encode costs and effects directly in code, plus targeted
web research to cross-check the progress token and military token wording.
The resulting counts (23 Age I cards, 23 Age II cards, 20 Age III cards,
7 guild cards — 73 total; 12 wonders; 10 progress tokens) match the
official rulebook's stated component counts. That said: **this is still a
best-effort transcription, not a guarantee of correctness.** It has not
been checked card-by-card against a physical copy of the game or scans of
the actual cards. Before any M1 rules-engine work relies on this data for
legality checks or scoring, it should be spot-checked against the physical
rulebook/cards or BoardGameGeek, especially:

- Exact chain-symbol pairings (see "Chain symbols" below — the *pairing
  logic* is verified against working game code, but the *iconography names*
  are our own invented ids, not the game's actual printed symbol names).
- Scientific symbol naming (`mortar`, `pendulum`, `inkwell`, `wheel`,
  `sundial`, `gyroscope`, `balance`) — these are descriptive ids we chose,
  not official terminology, though the underlying 6-symbols-per-card-deck +
  1-from-the-Law-token structure is verified.
- Guild card majority-comparison wording (`majority_reward` effects) —
  verified against working game logic (it compares each guild's target
  building-type count across *both* players and rewards the guild's owner
  based on the higher of the two counts, not just their own count), but not
  against the exact printed card text.

## Files

- `cards.json` — all 73 age cards: the 23 Age I, 23 Age II, and 20 Age III
  non-guild cards, plus all 7 guild cards that exist in the base game (only
  3 of the 7 are randomly selected into the Age III structure per game,
  alongside 1 extra Age III slot removed with them — that per-game
  random-selection logic is engine/setup behavior for M1, not data).
- `wonders.json` — all 12 base-game wonders. 8 of the 12 are randomly
  drafted (4 offered to each player in turn) at the start of a game; that
  draft logic is also engine/setup behavior for M1, not data.
- `tokens.json` — all 10 progress tokens. Only 5 of the 10 are randomly
  made available in any single game (the other 5 are set aside entirely for
  that game) — again, setup logic for M1.
- `military.json` — the military conflict track and its 4 fixed-position
  military tokens, plus the end-of-game military scoring table used when
  neither player reaches instant military supremacy.

## Schema notes

### `cards.json`

Each entry:

```jsonc
{
  "id": "lumber-yard",          // stable slug, unique within cards.json
  "name": "Lumber Yard",
  "age": 1,                      // 1, 2, or 3 (guild cards are age 3)
  "type": "raw_material",        // raw_material | manufactured_good | civilian
                                  // | scientific | commercial | military | guild
  "cost": { "coins": 0, "resources": { "wood": 1 } },
  "chain_from": null,            // id of the card that, if already built by
                                  // this player, lets this card be built for
                                  // free — or null
  "chain_to": null,              // id of the (single, in base game) card
                                  // that this card unlocks a free build for
                                  // — or null. Inverse of that card's
                                  // chain_from.
  "effects": [ /* structured effect objects, see below */ ]
}
```

`chain_from`/`chain_to` model the game's chain-symbol mechanic (build a card
for free if you already own its prerequisite) as a direct card-to-card link
rather than via a separate named-symbol table, because in the base game
each chain relationship is a 1:1 pair. This is a modeling simplification
worth flagging: we do not know the *official* icon name for the printed
symbol on each pair (e.g. whether it's a "wheel", "sun", "target", etc.) —
only which card requires which. If that ever matters (e.g. for a UI that
renders the actual icon), it needs to be sourced from the physical cards.

### `wonders.json`

Same `cost`/`effects` shape as cards, minus `age`/`type`/`chain_*` (wonders
aren't part of the age-card chain-symbol or type system).

### `tokens.json` / `military.json`

Progress tokens and military tokens don't have a `cost` (they aren't built,
they're acquired via specific triggers described in `engine.rs`/rules, not
data) — each has an `id`, `name`, and a list of structured `effects`.

### Effect vocabulary

Effects are tagged objects (`{"type": "...", ...}`), used consistently
across all four files. As of this writing the types in use are:
`produce_resource`, `produce_resource_choice`, `victory_points`,
`victory_points_per_progress_token_owned`, `scientific_symbol`, `shield`,
`take_coins`, `opponent_loses_coins`, `fixed_trade_cost`,
`redirect_opponent_trade_payments`, `future_cost_rebate`,
`construction_triggered_bonus`, `chain_build_triggered_bonus`,
`take_coins_per_owned_building`, `take_coins_per_constructed_wonder`,
`majority_reward`, `counts_as_scientific_symbol`, `destroy_opponent_building`,
`build_discarded_card_free`, `choose_progress_token`, `play_again`. This
vocabulary is descriptive, not yet consumed by any code (`duels-core::data`
is a stub in M0) — M1's data-loading code is free to redesign this shape
entirely as long as it re-derives the same underlying facts; nothing in
`duels-core` parses these files yet.

## Resource/type vocabulary

Resources: `wood`, `clay`, `stone` (raw materials), `glass`, `papyrus`
(manufactured goods). Card types: `raw_material`, `manufactured_good`,
`civilian`, `scientific`, `commercial`, `military`, `guild`.
