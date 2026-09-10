# Round four: the turn the evaluator was scoring half of

Round four is one bug, one arithmetic fix, and two ideas that did not pay
for themselves. All of it is reproduced bit for bit by `Config::v3`
(`tests/v3_identity.rs`) except the arithmetic fix, which is landed
unconditionally because it is a fix.

**1. Four wonders were being scored before they had done anything.**
`engine::finish_turn` returns early while a
`duels_core::state::Pending` is outstanding, so the state `apply` returns
for Circus Maximus, the Statue of Zeus, the Mausoleum and the Great Library
is one in which the effect has **not yet happened**: the card the destroy
will take is still in the opponent's city, the retrieval and the token do
not exist, and the builder is still `current_player` even though the turn is
about to pass. (An ordinary `Build` of a green card that completes a science
pair leaves the same kind of state.) Scoring it directly credited none of
the effect — only `terms::wonder_power`'s flat "+3, this wonder does
something" — read the mover as moving again, which flips the sign
`menu::menu_term` puts on the position, and stood the rails down entirely,
since `rails::rail_owner` refuses to read a pending state.

`PendingModel::Completed` (**the new default**) finishes the mover's own
turn before judging it: it resolves the pending choice with the engine's own
`duels_core::engine::legal_actions` — which already enumerates the
concrete options, so nothing here re-implements a rule — takes the one the
*resolver* likes best, and scores what `finish_turn` then leaves. This is
not search: every action in the resolution belongs to the same player in the
same turn, and the engine already models that turn as those sequential
decisions.

The Great Library's three-token draw needs no special handling, which is
worth saying because it looks like it should: the draw arrives as
`duels_core::engine::chance_outcomes`' ten `C(5,3)` outcomes, which
`expected_value` already averages over, so each draw gets its own exact
max-over-three and the ten are weighted correctly for free.

**The engine chains exactly once**, and only there: `ChooseProgressToken`,
`ChooseGreatLibraryToken` and `DestroyOpponentCard` each clear the pending
flag without setting another, but `MausoleumBuild` runs the retrieved card
through `construct_card`, which sets `Pending::ProgressToken` if the card
off the discard pile completes a science pair.
`tests::a_mausoleum_retrieval_can_chain_into_a_progress_token_choice` builds
that position and pins it; `MAX_PENDING_DEPTH` is three, one more than the
chain the engine can actually produce.

**2. `wonder_potential` never checked the seven-wonder cap** — a bug, not a
model, so it is fixed unconditionally and `Config::v3()` does not restore
it. The base game builds seven wonders between the two players and no more,
and the term kept paying `0.5 x wonder_power` for every dead wonder in
either hand for the rest of the game, *asymmetrically*, since the two sides
rarely hold the same number of them. The audit below says this is not
hypothetical: 0.42 unbuildable wonders are left in a hand per game.

**3. `WonderModel::Budget` (default off — an honest negative).** A
per-effect price for an unbuilt wonder, scaled by the chance it is ever
built: `p_build = cap_share x turn_factor`, where `cap_share` rations the
remaining shared slots across both players' unbuilt wonders and
`turn_factor` rations the owner's remaining decisions. Every channel reuses
a pricer that already exists — `coin_marginal`,
`terms::military_shield_delta`, `menu::TakeValue::produced_value`,
`menu::TakeValue::free_value`, `duels_strategy::science::token_value` —
so the Great Library is priced from the tokens actually set aside and a
destroy from what the opponent actually owns, rather than at a flat `+3`.
It is a better model and it loses: **−11.0 / −5.6 / −3.9** Elo against
`phased:base=v3` over 3200 games on each of three disjoint seed ranges. (At
800 games per range it read `+2.6 / +1.7`, which is the whole argument for
this project's sample sizes.) Kept as `phased:wonder=budget`, with the
measurement written down.

**4. `Config::destroy_replace_discount` (default off — the second honest
negative).** A destroyed brown or grey card is only permanently gone if the
market cannot print another one, so the credit is discounted by
`min(1, sources_remaining(r) x dealt_frac x share_opp)`. Age III prints no
brown or grey card at all — counted off `data/cards.json` by
`terms::tests::no_production_source_survives_into_age_three`, not taken on
faith — so an Age III destroy is priced as the permanent loss it is and the
discount is an exact no-op there. **−0.9 / +5.8** Elo over 3200 games on two
disjoint ranges: the signs disagree, so it stays off.

## The audit, which is again the actual result

`duels-arena/examples/wonder_audit.rs` counts the behaviour directly rather
than inferring it from a win rate, for the same reason `rail_audit.rs` does:
both fixes are worth a couple of victory points in a game whose scores span
thirty, and both are about *which move gets played*. 200 self-play games at
`Nodes(1)`, on each of two disjoint seed ranges — `built% @ mean turn`:

```text
                             seed 1                    seed 5001
                       base=v3      this agent    base=v3      this agent
the four pending-       79% @ 39.0   86% @ 34.4   76% @ 39.7   81% @ 35.6
  effect wonders
  The Statue of Zeus    78% @ 41.3   93% @ 33.4   78% @ 41.1   87% @ 32.6
  The Mausoleum         75% @ 39.4   91% @ 36.0   76% @ 41.1   87% @ 37.3
  Circus Maximus        84% @ 38.1   77% @ 31.5   72% @ 37.0   78% @ 33.5
  The Great Library     78% @ 36.9   80% @ 37.1   80% @ 39.5   73% @ 39.5
every other wonder      88% @ 28.9   86% @ 31.3   90% @ 29.2   88% @ 30.4
drafted, never built       123          113          117          112
```

**The two wonders the flat bonus under-priced most move on both ranges, and
move a long way**: a destroy that takes a whole card out of the opponent's
city and a free build out of the discard pile go up nine to fifteen points
more often and up to eight and a half turns earlier. The two that do not
move consistently are the two whose *value* was already roughly right at a
flat `+3` and whose timing is the real question — Circus Maximus destroys a
grey card rather than a brown one, and the Great Library's token is worth
whatever the set-aside pile happens to hold. Both are now priced against
what is actually there, so both move in whichever direction that position
calls for; the mean build turn falls for Circus Maximus on both ranges,
which is the timing half of the same read.

## Elo

Against `phased:base=v3` at `Nodes(1)`, 3200 games per seed range on four
disjoint ranges:

```text
                       seed 1       seed 5001     seed 10001     seed 20001
pending resolution  +3.5 [-8.6,   +5.4 [-6.6,   +6.1 [-6.0,   +15.6 [+3.6,
                      +15.5]        +17.5]        +18.1]         +27.7]
wonder budget      -11.0 [-23.0,  -5.6 [-17.7,       —         -3.9 [-15.9,
                      +1.1]          +6.4]                         +8.1]
```

Only one of the four pending-resolution ranges clears zero on its own, but
all four agree in sign, and the audit is what the change was built for. The
wonder budget agrees in sign too — the other way — on all three of its.

Against the ladder, 400 games per seed range at seeds 1 and 5001
(`Nodes(1)`; `alphabeta` and `mcts-uct` at `Nodes(2000)`):

```text
                   this agent          phased:base=v3
vs random          400-0 / 399-1
vs greedy          399-1 / 398-2
vs greedy-ev       399-1 / 399-1
vs strategist      400-0 / 400-0
vs alphabeta       87/400 / 93/400     79/400 / 81/400
vs mcts-uct        44/400 / 30/400     40/400 / 34/400
```

`alphabeta` is again the only ladder opponent close enough to measure a
change against, and it moves the right way on both ranges: 87 and 93 wins in
400 against 79 and 81. Against `mcts-uct` the two are **level** — 74/800
pooled against 74/800, one range up and one down — which is the honest
negative of the round's headline: finishing a turn correctly does not close
the gap between a 1-ply evaluation and a real search. The win-condition
spread holds (military 0 and 2, science 19 and 13, civilian 25 and 13).

## Cost

`examples/decision_cost.rs`, every configuration timed on the same 5709
positions:

```text
default, pending effects unresolved     49.2 us/decision
default (pending effects completed)     56.6 us/decision   +15%
default + the wonder budget model       55.8 us/decision   (no measurable change)
default + the destroy discount          56.6 us/decision   (no measurable change)
```

+15%, and all of it in Ages II and III: the Age-I-only figure moves 47.7 to
49.4 us, because a pending effect comes from a wonder and wonders are not
built on turn three. The resolution is bounded by construction — at most
eight opponent cards for a destroy, the discard pile for the Mausoleum, five
board tokens, three Great Library tokens — and it runs only on the small
minority of candidates that create one.

The `TimeMs` half of this project's two-budget discipline is checked and
reported rather than assumed: `--budget time_ms:50` and `--budget nodes:1`
over the same 200 games and the same seed produce the identical
`91-108-1`, game for game, which is what "a 1-ply agent ignores its budget"
means when it is measured instead of asserted.
