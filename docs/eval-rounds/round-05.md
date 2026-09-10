# Round five: the cards nobody was pricing

Round five is one confirmed bug in `menu::TakeValue`, one term the
evaluation simply did not have, and four ideas that did not pay for
themselves. All of it is reproduced bit for bit by `Config::v4`
(`tests/v4_identity.rs`).

**1. Every guild card priced out negative.** `menu::TakeValue::free_value`
starts a card's value from `def.victory_points` and `def.coins`. Both are
**zero for all seven guilds** — a guild scores through
`points_by_majority` / `coins_by_majority`, which
`duels_core::scoring::breakdown` reads at scoring time and the menu's
pricer did not read at all — so a face-up guild was worth
`−cost × coin_marginal`: strictly negative, in every position, for both
players. This agent therefore never fought for a guild and never denied one
to an opponent who was collecting the colour it counts.
`terms::tests::every_guild_scores_through_a_majority_and_not_through_
printed_points` asserts the premise off the card data rather than arguing
it, and
`menu::tests::a_face_up_guild_used_to_price_out_negative_and_now_does_not`
pins the fix.

`GuildPricing::Projected` (**the new default**) prices it as

```text
v_guild(g)  = per_vp · Ĝ(t)  +  per_coin · live(t) · coin_marginal
Ĝ(t)        = max( c_1 + Δ_1(t),  c_2 + Δ_2(t) )
Δ_p(t)      = ρ_t · take_rate · decisions_left(p)   for a colour, or brown+grey
            = U_p · p_build(p)                       for wonders
            = 0                                      for coins / 3
```

`ρ_t` is `DevSupply::kind_fraction`, off the same pool walk every other
development price uses; `p_build` is `terms::wonder_p_build`, the
probability `WonderModel::Budget` rations wonders with, factored out and
called directly because it is a standalone estimate and has nothing to do
with how a wonder's *effects* happen to be priced. Root-fixed, like every
price in this crate. The points channel reads `Ĝ` and the coin channel reads
the live board, because the coins are paid on the spot and the points are
not. Which of the seven guilds keys off which category is asserted card by
card against `data/cards.json` in
`terms::tests::the_seven_guilds_key_off_the_categories_the_card_data_prints`
— including the two that need glass *and* papyrus, and the one that counts
coins rather than cards.

Both players read the same `Ĝ`, because the rule pays the guild's owner on
the higher of the two counts whether or not it is their own. So the race
dynamic and the denial both fall out of the menu differencing two per-player
values, with no special case anywhere.

**2. A yellow card's forward discard yield was entirely unpriced.**
`duels_core::cost::discard_reward` is `2 +` the player's own commercial
cards, so every yellow card in a city raises the payout of **every future
discard that city makes**. The coins from a discard already *made* flow
through `terms::coin_points` and `terms::coin_liquidity` exactly; the
forward half did not exist. `EvalWeights::yellow_equity` (**on by
default**) adds `coin_marginal · yellows(p) · rate · decisions_left(p)`, with
the matching per-card credit fed into `menu::TakeValue` so the menu and the
evaluation agree about what a yellow card is worth. `rate` is **measured,
not guessed** — `examples/discard_rate.rs` counts discards per decision over
whole self-play games and reads 0.2489 / 0.2499 / 0.2467 on three disjoint
seed ranges, hence `terms::DISCARD_RATE_PER_DECISION` = 0.249.

This is the largest gain of the round and the one to be most suspicious of;
see `EvalWeights::yellow_equity` for the fitted weight, the alternative
explanation that is ruled out, and the one that is not.

**3. `MenuFloor` (default off — an honest negative).** `menu::menu_term`
returns a hard `0` when nothing on the board is affordable, so an opponent
one coin short of everything reads identically to an opponent whose turn is
genuinely worthless — and, worse, taking their *last* affordable card is
under-rewarded, because the position after reads as a flat zero either way.
The real floor is the discard they can always take, and possibly a wonder
they can already pay for. Both entries are added to the softmax rather than
replacing it. **−4.8 / +3.5 / −2.9** (discard only) and **−12.9 / −1.2 /
+0.3** (discard and wonder) Elo against `phased:base=v4` over 3200 games on
each of three disjoint seed ranges: the signs disagree, so it stays off,
available as `phased:menufloor=discard`.

**4. `Config::menu_afford_soft` (default off — the second honest
negative).** `w_j = σ((coins − cost) / c_soft)` in place of the hard afford
cutoff, so a card the next mover is narrowly short on still carries partial
weight. **−15.3 / −10.3 / −11.1** at `c_soft = 1`, **−15.5 / −11.9 / −10.6**
at 2, **−17.6 / −11.5 / −8.3** at 3 and **−17.6 / −8.4 / −9.8** at 6, all
over 3200 games on each of three disjoint ranges. Negative on every range at
every width tested, which is at least an unambiguous answer.

**5. `EvalWeights::guild_projection` (default off — the third).** The same
`Ĝ` machinery applied to guilds a player has *already built*, as the forward
increment `per_vp · (Ĝ(t) − live(t))` on top of the snapshot `breakdown`
already credits. **−3.6 / +2.7 / −3.5** at 0.5, **−2.7 / +3.7 / −3.0** at
1.0 and **−2.3 / +2.5 / −0.9** at 2.0 against the same agent with guild
pricing on and this term off. Every interval crosses zero and the sign does
not agree across ranges.

**6. `SupplyModel::Dealt` (default off — the fourth).** `DevSupply`'s
pool adds every *whole undealt deck* at weight one, but setup deals 20 of
Ages I and II's 23 cards and — Age III being the only age with guilds — 17
of its 20 plain cards plus 3 of its 7 guilds, the rest going back in the box
unseen (`duels_core::engine::new_game`, `duels_core::state::GUILDS_IN_PLAY`).
Weighting each undealt entry by its own age's dealt fraction is the
straightforwardly more correct statistic, and it is worth **−8.7 / −1.2 /
−5.2** Elo over 3200 games on each of three disjoint ranges, and **−5.4 /
+7.3 / −3.3** measured on top of the yellow term instead. Negative or
neutral either way; kept as `phased:supply=dealt` with the measurement
written down. It is a small correction — Ages I and II are scaled uniformly,
so only Age III's 17/20-against-3/7 split moves anything relative.

## Elo, measured one change at a time

Against `phased:base=v4` at `Nodes(1)`, **3200 games per seed range on three
disjoint ranges**, each row a paired head-to-head of exactly that one change
against the round-four default:

```text
                             seed 1        seed 5001       seed 9001
yellow_equity = 4.0        +65.9 ± 12.3   +71.0 ± 12.3    +78.5 ± 12.4
guild pricing              +10.3 ± 12.0   +19.7 ± 12.1    +11.7 ± 12.1
the discard floor           -4.8 ± 12.1    +3.5 ± 12.1     -2.9 ± 12.1
the discard+wonder floor   -12.9 ± 12.1    -1.2 ± 12.0     +0.3 ± 12.1
the dealt supply weighting  -8.7 ± 12.1    -1.2 ± 12.0     -5.2 ± 12.0
soft affordability (3.0)   -17.6 ± 12.1   -11.5 ± 12.1     -8.3 ± 12.1
```

and the two accepts together, on **three further disjoint ranges** so the
default is not confirmed on the ranges it was chosen on:

```text
                         seed 13001     seed 17001      seed 21001
the new default          +83.7 ± 12.4   +94.4 ± 12.5    +95.5 ± 12.5
```

The two are additive: guild pricing is worth **+13.0 / +20.3 / +15.3**
measured on top of the yellow term rather than against the bare round-four
agent, which is the same number it reads on its own.

`yellow_equity`'s weight was swept over 0.5, 1, 1.5, 2, 3, 4, 6, 9 and 14 on
all three ranges. It rises to a broad plateau between 3 and 9 and falls again
by 14; 4.0 is the middle of the plateau, 6.0 is worth **+7.4 / +9.4** more
on two fresh ranges (both intervals crossing zero) and 3.0 **−9.2 / −9.4**
less. The weight is fitted, and `EvalWeights::yellow_equity` says so.

## Against the ladder, at both budget kinds and at ten times the budget

800 games per seed range at seeds 1 and 5001, `Nodes(1)` for the 1-ply
opponents:

```text
                   this agent          phased:base=v4
vs random          799-1 / 797-3       799-1 / 799-1
vs greedy          799-1 / 799-1       794-6 / 794-6
vs greedy-ev       799-1 / 799-1       799-1 / 798-2
vs strategist      800-0 / 795-5       800-0 / 797-3
```

Everything below `alphabeta` is at the ceiling and stays there. The two
search opponents are where the measurement is, and this project's
two-budget discipline matters for them even though it cannot matter for
`phased` itself (a 1-ply agent ignores its budget — `choose` takes
`_budget`), because it is what decides how strong the *opponent* is:

```text
                                 this agent            phased:base=v4
vs alphabeta  Nodes(2000)        260/800 / 250/800     176/800 / 169/800
              TimeMs(20)         254/800 / 257/800     197/800 / 169/800
              Nodes(20000)        88/400 /  79/400      56/400 /  51/400
vs mcts-uct   Nodes(2000)        123/800 / 125/800      84/800 /  72/800
              TimeMs(20)         131/800 / 137/800      95/800 /  86/800
```

Six paired comparisons against two unrelated searchers, and every one of
them moves the same way on both ranges: 22% to 32% against `alphabeta` at
`Nodes(2000)`, 10% to 16% against `mcts-uct`, and — the one worth having —
**14% to 22% against an `alphabeta` given ten times the nodes**, where it
concedes only 13-14% to the round-four agent. The gain is not an artefact of
a particular opponent, a particular budget kind, or a particular search
depth, and it is not self-play overfitting.

## The behaviour actually changed

`duels-arena/examples/matchup_profile.rs` now reports, per side, how many
guilds it builds and at what majority count they pay — because a win rate
cannot tell "it fights for guilds now" from "it got luckier". 800 games,
`Nodes(1)`, guild pricing alone against the round-four agent:

```text
                               phased:base=v4    + guild pricing
purple cards per game               0.7                1.3
games ending with a guild          57.5%              82.9%
mean majority count paid on         4.86               4.93
guild victory points per game       4.15               7.23
guilds (of 3 dealt) bought at all   2.02               2.02
```

Same three guilds on the table, nearly twice as many of them ending up on
*this* side, and at a marginally higher count — so it is not taking any
guild it sees, it is taking the ones that pay. The full default (yellow term
included) reads 1.2 purple and 7.62 guild VP per game against 0.8 and 4.29,
and shifts the city hard towards commercial cards: **5.5 yellow per game
against 2.6**, which is the yellow term made visible.

## Cost

`examples/decision_cost.rs`, every configuration timed on the same 2881
positions:

```text
v4 (the round-four agent)             51.0-51.3 us/decision
default (round five)                  50.8-51.1 us/decision   no measurable change
default, guilds unpriced              50.4-51.6 us/decision
default + the discard/wonder floor         55.0 us/decision    +7%
default + soft affordability               53.1 us/decision    +4%
```

**Round five is free**, within the run-to-run noise of the benchmark: the
guild table is ten majority counts and ten pool fractions once per decision,
and the yellow term is one `count` per player per evaluated state. Both
options that cost anything are off by default — the floor prices every
unbuilt wonder per chance outcome, and soft affordability stops the cutoff
from skipping the cards it used to skip. Round four's own `+15%` for the
pending-effect fix stands unchanged underneath.
