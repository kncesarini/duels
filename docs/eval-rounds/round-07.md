# Round seven: the forward-looking terms were collectively over-priced

Round seven is **one new gate, five re-weightings, four honest negatives
and two new measurement instruments**. All of it is reproduced bit for bit
by `Config::v6` (`tests/v6_identity.rs`).

## The brief, and the honest answer to it

The brief was to improve the leaf evaluation, touching nothing outside this
crate, and to raise `duels-agent-mcts-eval`'s Elo by **at least +200**
against a fixed `mcts-uct` anchor. **The achieved figure is +1.2 Elo, 95%
interval [-7.6, +10.0], over 12,800 games a side on four disjoint seed
ranges** — which is to say no measurable change at all, and nowhere near
the target.

That is the round's most useful output, because it is not a statement about
this change set. The same change set is worth **+149 / +166 Elo to
`phased`**, **+122 / +126 against `alphabeta`** and **+87 / +101 against
`mcts-uct`** — the largest movement any round of this crate has produced
against an unrelated opponent, and the first time `phased` has beaten
`alphabeta` at `Nodes(2000)`. **None of it reaches `mcts-eval`.** A leaf
that is half playout, blended at `weight = 0.5` with an exploration
constant tuned for that blend, extracts what it is going to extract from a
hand-crafted evaluation, and this crate is no longer the binding
constraint on it. The levers that could change that — the blend weight, the
exploration constant, a `Root` rebuilt deeper in the tree — all live in
`mcts-eval`, which this round was not allowed to touch and which is
where a +200 would have to come from.

## Two instruments this crate did not have

Every previous round could only sweep a knob that
`duels_arena::agent_spec::parse_phased_config` had a key for, which put the
instrument for measuring a `duels-eval` change in a crate a `duels-eval`
round is not supposed to touch. Round seven added two `examples/` binaries
instead, and they are why it could sweep as widely as it did.

**`examples/head_to_head.rs`** is the arena's `phased`-versus-`phased`
match reproduced inside this crate over two `Config` values given on the
command line: same paired seat-swap, same salts, same `PhasedAgent::choose`,
same Bradley-Terry fit. It agrees with the arena **to the game**:
`phased:science_ladder=0.2` against `phased` over 3200 games at seed 1
reads `+60.2 [+48.0, +72.4]` through `duels-arena` and `+60.2 [+48.0,
+72.4]` through this example, 1874-1325-1. It also runs a 3200-game match
in about three seconds, which is what made a five-round coordinate descent
affordable.

**`examples/leaf_probe.rs`** scores a *fixed* corpus of labelled positions
under many configurations at once — mean negative log-likelihood at each
configuration's own maximum-likelihood temperature, plus the
temperature-free sign accuracy. `examples/calibrate.rs` cannot do this: it
fits one configuration over that configuration's own self-play, so changing
the weights changes the questions. See "a better predictor is a worse leaf"
below for what the probe then measured, which is not what it was built to
find.

## 1. The science ladder was being paid for a race that was already over

`terms::science_ladder`'s rung is steeply convex — 6, 12 and 18 victory
points at three, four and five distinct symbols — and the convexity has
exactly one justification: six symbols win the game outright. The rung was
collected whether or not a sixth symbol was still *in* the game. A player
sitting on four symbols whose two missing ones were both buried in the
opponent's city collected twelve victory points for a race that could not
be run.

`ScienceWeights::dead_race_scale` (**`0.0` by default**) multiplies the
rung — and only the rung, never `pair_threat`, since a half-pair is still a
progress token — once `terms::supremacy_reachable` says the player can no
longer assemble `terms::SYMBOLS_TO_WIN` distinct symbols, counted off the
card data through the same public-information test
`terms::second_copy_obtainable` already applies to a second copy.

It is worth **+32.0 / +24.8** Elo on its own against `phased:base=v6` over
3200 games on each of two disjoint seed ranges, and — the reason it is a
gate and not a smaller number — **it keeps the scientific-supremacy route
intact**, 44 and 51 science wins in 3200 games against round six's 39 and
45. That is the whole argument for it, because the cheap alternative is
better on the aggregate and much worse on the route: driving
`EvalWeights::science_ladder` to `0.2` with nothing else changed is worth
**+61.6 / +58.3** and takes the science wins to **3**.

The shipped default is both: the gate, plus the ladder at `0.5` and
`ScienceWeights::pair_threat_weight` at `0.5`. Each field's own comment
carries its sweep.

## 2. Chain equity, and the fitted weight that was covering for it

With the ladder corrected, two weights that had been measured as
"indistinguishable from zero, kept on the pooled result" and "fitted, not
derived, and flagged as such" both turned out to be badly wrong, in the
same direction, for what is probably the same reason.

* `EvalWeights::chain_equity` `1.0` → **`0.25`**: the largest single
  re-weighting of the round, +71 / +82 Elo.
* `EvalWeights::resource_bill` `3.0` → **`1.0`**: +53 / +60 Elo, and a
  *reversal* — round two fitted `3.0` over the derived `1.0` and wrote down
  that it was "a measurement, not an argument". The derived rate now wins
  by a wide margin on a monotone curve. **The fitted weight was
  compensating for two other over-priced terms.**
* `EvalWeights::development` `1/3` → **`0.2`**: +4 / +14 Elo.

The pattern is the round's one-line summary and is worth keeping as a prior:
**this evaluation's forward-looking terms were collectively over-priced**,
each of them fitted against a baseline that contained the others, and
correcting the largest one moved the honest value of the rest a long way.

## Elo, measured one change at a time

Against `phased:base=v6` at `Nodes(1)`, **3200 games per seed range**, as a
leave-one-out against the round-seven default — so each row is a paired
head-to-head of exactly that one change. Every interval is ±12.2.

```text
                                    seed 1   seed 5001   the change is worth
the round-seven default             +149.2     +165.6
chain_equity back to 1.0             +77.8      +83.6      +71 / +82
resource_bill back to 3.0            +96.0     +106.0      +53 / +60
science_ladder back to 1.0          +124.0     +133.5      +25 / +32
pair_threat_weight back to 1.0      +138.6     +143.7      +11 / +22
the dead-race gate switched off     +142.5     +153.3       +7 / +12
development back to 1/3             +145.5     +151.8       +4 / +14

owned-token equity switched on      +147.0     +160.7       -2 /  -5
the count-priced menu switched on   +153.1     +165.8       +4 /  +0
to_move = 2                         +128.5     +149.3      -21 / -16
to_move = 6                         +106.7     +135.5      -43 / -30
to_move = 12                        +104.5     +130.0      -45 / -36
```

The six accepted rows sum to +171 and +217 while the whole default is worth
+149 and +166, and that is the point rather than an inconsistency: every one
of them was fitted, in an earlier round, against a baseline that contained
the others.

The default is then confirmed on **four further disjoint ranges it was not
tuned on** — **+150.4**, **+161.2**, **+157.6** and **+145.7** at seeds
9001, 13001, 17001 and 21001, 3200 games each — so the five rounds of
coordinate descent behind it are not two seed ranges' worth of noise.

## Against the ladder, including two unrelated searchers

800 games per seed range; the 1-ply opponents at `Nodes(1)` and the
searchers at `Nodes(2000)`.

```text
                                the new default        phased:base=v6
vs random        seed 1          798-2                  800-0
vs greedy        seed 1          800-0                  796-4
vs greedy-ev     seed 1          800-0                  797-3
vs strategist    seed 1          800-0                  794-6
vs alphabeta     seed 1        +54.2 [+29.9, +78.6]   -67.6 [-92.2, -43.1]
                 seed 5001     +35.7 [+11.5, +59.9]   -90.5 [-115.4, -65.6]
vs mcts-uct      seed 1       -134.3 [-160.1, -108.4] -221.4 [-250.5, -192.3]
                 seed 5001    -120.9 [-146.4, -95.4]  -222.1 [-251.2, -192.9]
```

**+122 / +126 Elo against `alphabeta` and +87 / +101 against `mcts-uct`**,
agreeing on both ranges against two unrelated searchers, so the gain is not
self-play overfitting. `phased` now **beats `alphabeta` at `Nodes(2000)`**,
which no generation of this crate has done before.

## The behaviour actually changed

`duels-arena/examples/matchup_profile.rs`, 400 games against `mcts-uct` at
`Nodes(2000)` and seed 1, is the check that a win rate cannot make: did the
agent start playing differently, or did it get luckier?

```text
                                  phased:base=v6    the new default
wins                                 91 / 400          128 / 400
  by military supremacy                 2                  2
  by scientific supremacy              30                  8
  civilian                             58                118
green cards per game                  3.9                1.6
yellow cards per game                 4.2                5.7
blue cards per game                   3.8                4.3
distinct symbols reached             3.54               1.25
guilds built (of 400 games)           321                393
guild victory points per game        4.82               6.71
victory-point margin in its losses   -7.22              -2.58
```

Two readings, and the second is the one that matters. The obvious one is
that the round traded the science lottery for civilian points: thirty
supremacy wins become eight, fifty-eight civilian wins become a hundred and
eighteen, and the city goes from four green cards to one and a half. The
useful one is the bottom row — **the losses got much closer, −2.58 points
against −7.22**. An agent that was entering a race it usually lost and then
losing the rest of the game by seven points is now losing by two and a half,
which is what a re-priced evaluation looks like from the inside and is not
something a win rate would have shown.

The military column is unchanged at 2 wins in 400, which is the one
dimension of the profile round seven neither improved nor damaged, and the
one `mcts-uct` still uses against it (72 military-supremacy wins before, 65
after).

## The finding that matters most: a better *predictor* is a worse *leaf*

`examples/leaf_probe.rs` was built on the reasoning that `phased` consumes
this crate as a *policy* (an argmax, scale-free) while `mcts-eval` consumes
it as a *value* (a fitted logistic averaged into a win rate), so the
objective that matters for a leaf is how well the number predicts the
winner. That reasoning is sound and the conclusion it leads to is **wrong**,
which is the most useful thing this round found.

A four-weight variant — `military_band = 3.0`, `vp_projection = 1.6`,
`development = 0.16`, `yellow_equity = 2.0` — is a *substantially* better
predictor than either round six or the round-seven default, reproduced on a
disjoint corpus:

```text
                   train (43k positions)      validate (disjoint, 46k)
                 T     NLL    sign  sgn-III     T     NLL    sign  sgn-III
round six      34.5  0.5982  0.6721  0.7247   25.1  0.5508  0.7036  0.7474
round seven    23.1  0.6136  0.6730  0.7408   16.6  0.5741  0.6985  0.7619
the predictor  14.0  0.5649  0.6981  0.7847   10.8  0.5210  0.7256  0.8197
```

The predictor is better on every column on both corpora — two to three
points of sign accuracy overall and four to seven in Age III — and as an
`mcts-eval` leaf value it is worth **+64.9** against the anchor, against
**+94.8** for the intermediate round-seven bundle it was built on top of
(the science gate and the ladder, without the chain-equity, bill and
development corrections) and **+89.3** for round six, all at 3200 games and
seed 1. The victory kinds say why: military wins go
236 → 494 and civilian collapses 1703 → 1368. Military standing predicts
the winner very well *and* steers a search into races it then loses, which
is this project's oldest finding — "win-condition awareness belongs in the
search policy, not the evaluation function" — arriving from a new direction.

The middle row makes the point sharper still, and it is the shipped
default: **round seven is a *worse* predictor than round six** — a tenth of
a nat of likelihood worse on both corpora, with sign accuracy flat — while
being +150 Elo stronger as a policy and, as a leaf, no worse. On this
corpus, over these two objectives, the correlation is not merely weak;
across the three rows it points the wrong way.

**So the probe is a screen for a hypothesis, not a proxy for leaf quality.**
The instrument that did predict `mcts-eval`'s direction was the boring one:
`phased` Elo, attenuated. Later rounds should treat it that way.

## `mcts-eval` against the fixed `mcts-uct` anchor

`mcts-eval` reads `Config::default` live and pins no generation (that is
deliberate; see its crate docs), so "old against new" is not a single-binary
match. The measurement is therefore indirect, against an anchor that does
not move: `mcts-uct` no longer depends on this crate at all, so the same
`mcts-uct` is on the other side of every row below and the **difference of
the two Elo-vs-anchor columns is the achieved gain**. Both columns were
measured with a binary built from the same tree, differing only in what
`Config::default()` returns.

```text
Nodes(2000), 3200 games per seed range, paired and seat-swapped
                  round six        round seven        the round is worth
seed 1          +89.3 +-12.4      +93.9 +-12.5              +4.7
seed 5001       +89.4 +-12.4     +104.4 +-12.6             +15.0
seed 9001       +94.6 +-12.5      +83.1 +-12.4             -11.6
seed 13001      +84.9 +-12.4      +82.0 +-12.4              -2.9
pooled (12800)  +89.6 +- 6.2      +90.8 +- 6.2       +1.2 [-7.6, +10.0]

TimeMs(20), 400 games per seed range, RAYON_NUM_THREADS=1, one at a time
seed 1          +54.2 +-34.4      +97.8 +-35.4             +43.6
seed 5001       +88.5 +-35.1      +55.9 +-34.4             -32.6
pooled (800)    +71.3 +-24.6      +76.7 +-24.6      +5.4 [-29.4, +40.2]
```

**+1.2 Elo, on an interval that comfortably contains zero, over twelve
thousand eight hundred games a side.** Two ranges up, two down, and the two
wall-clock ranges disagree with each other as well. The honest reading is
not "a small gain" but **"no measurable change"**: round seven is worth
about +150 Elo to `phased`, +122 against `alphabeta` and +95 against
`mcts-uct`, and none of it reaches `mcts-eval`.

Two things stop that being a statement about measurement noise. The
four-range protocol is what caught it — at the two ranges this round was
tuned on it reads +4.7 and +15.0, and a round that stopped there would have
reported a gain that the next two ranges erase. And `mcts-eval` is
demonstrably *not* insensitive to this crate in general: the predictor
variant above, a change of comparable size in the other direction, costs it
**−29.0 [±17.5]** at seed 1. The leaf can be broken from here. It cannot,
at `Nodes(2000)` and `weight = 0.5`, be much improved from here.

The win-condition breakdown says the same thing from the other side. Round
seven moves `mcts-eval`'s own profile a long way — military wins 236 → 324
and 259 → 314 on the first two ranges, scientific 43 → 25 and 46 → 28 —
while the totals stay put. It is playing differently and winning as often.

## Four honest negatives

**1. `EvalWeights::token_equity` (default `0.0`).** Six of the ten
progress tokens are rules changes that pay out over the remaining game —
Theology, Economy, Strategy, Architecture, Masonry, Urbanism — and this
evaluation priced **none** of them, which also meant that
`PendingModel::Completed`, the code that *chooses* a token when a science
pair completes, was choosing between them on printed victory points alone.
`terms::TokenTable` prices all six, each channel a quantity the evaluation
already computes (Economy's is literally `terms::resource_bill` read from
the other end). It measures at **+7.1 / +8.0** Elo at `0.5` and **+3.5 /
+15.0** at `1.0` against `phased:base=v6`, and then at **−2 / −5** as a
leave-one-out against the finished round-seven default, which is the
comparison that decides it. A real gap, correctly filled, worth nothing
once the terms it competes with are priced properly.

**2. `CountPricing::Counted` (default off).** Five Age III commercial
cards print no coins and instead pay a count of the builder's own city, so
`menu::TakeValue::free_value` — which starts from `def.coins` — could not
see nine coins on a Chamber of Commerce. Exactly the shape of the guild bug
round five fixed, and unlike that one it is a count already on the table
rather than a projection. **+5.9 / −2.3** Elo against `phased:base=v6`, and
**+3.9 / +0.2** as a leave-one-out against the round-seven default. Neutral
on both readings and with the signs disagreeing on one of them, so it stays
off, per this project's rule about not moving a default on a neutral
result. It is the more correct model and it is available as an option with
the measurement written down.

**3. `EvalWeights::to_move` (default `0.0`).** The right to move is worth
a great deal in this game (`../conventions.md` records ~67/33 between equal
`mcts-uct` configurations) and an extra turn is the only thing that
re-assigns the remaining slots. Nothing priced either. Two things came out
of trying: first, a term reading `GameState::extra_turn` is **exactly zero
at every position anything ever scores** — `engine::finish_turn` consumes
the flag the instant it would matter — measured over fifty thousand real
positions before the cause was found, and the reason
`terms::to_move` reads `current_player` instead. Second, it costs Elo:
`+2` is worth −21 / −16, `+6` −43 / −30 and `+12` −45 / −36 as a
leave-one-out against the round-seven default. The value objective *likes*
it, at one or two victory points; the policy objective does not, which is
the same divergence as the predictor above.

**4. `EvalWeights::value_scale` (default `1.0`).** The one knob a
`duels-eval` round has that a search can see and `phased` cannot: scaling
this crate's output by `k` divides `mcts-eval`'s fitted leaf temperature by
`k`. The maximum-likelihood calibration turns out to be about right —
measured on the same intermediate bundle as the predictor above, `k = 2.0`
reads **+84.7** and `k = 0.6` reads **+74.4** against the anchor where
`k = 1.0` reads **+94.8**, all at 3200 games and seed 1. Worth having
measured, because "the calibration `calibrate.rs` fits is also the
calibration the search wants" was an assumption and is now a measurement.
It is also the knob to re-check first if a future round moves the output
scale a long way: round seven took the maximum-likelihood temperature over
this corpus from 34.5 to 23.1 victory points while `mcts-eval`'s fitted
constants stayed where they were, and `k` is how a `duels-eval` round would
compensate for that without touching a search.

## The `yellow_equity` mystery is still open, and is now stranger

Round five's follow-up note flagged `EvalWeights::yellow_equity` as the
result to be most suspicious of: the term needed four times the weight its
own stated mechanism implies, and the working theory was that it was a proxy
for a different, unidentified mispricing of commercial cards. Round seven
corrected four genuinely mispriced terms and re-swept it, and `4.0` is
**still** on the plateau: 3.0 reads +139.1 / +154.3, 4.0 (the default)
+149.2 / +165.6, 4.5 +147.5 / +165.8, 5.0 +150.8 / +157.9, 5.5 +152.5 /
+155.2 and 6.0 +150.5 / +148.4 against `phased:base=v6` over 3200 games on
each of two disjoint seed ranges — flat from 4 to 5.5 and falling below 4,
which is where round five left it.

Two candidate mechanisms were ruled *out* along the way. It is not the
menu's blindness to the count-scaled Age III commercial cards — that is
`CountPricing` above, and pricing it is worth nothing. It is not the
coins-to-points rate, which round five had already ruled out. What remains
untested is the control round five named and did not build: a **flat**
per-yellow bonus, with `decisions_left` removed, which would say whether
the term is pricing discard yield at all or is pricing something that merely
correlates with holding commercial cards early. That needs a second knob and
is the obvious follow-up.

## Cost

`examples/eval_bench.rs`, every configuration timed on the same 2151
positions. The per-**leaf** number is the one that matters for this round,
because `evaluate` is what a search calls tens of thousands of times per
decision while `Root::new` is called once per tree node:

```text
                                    Root::new    evaluate       sum
v1 (the round-one evaluation)         1.819 us    0.220 us   2.039 us
v5 (the round-five evaluation)        3.467 us    0.455 us   3.922 us
v6 (the round-six evaluation)         3.462 us    0.457 us   3.919 us
default (round seven)                 3.488 us    0.470 us   3.958 us
default + owned-token equity          3.674 us    0.469 us   4.143 us
default + the count-priced menu       3.508 us    0.468 us   3.976 us
```

**Round seven is free**, at +0.013 us per leaf and +0.026 us per node,
which is inside the benchmark's run-to-run spread. It was not free when
first written: the dead-race gate's reachability walk read
**0.720 us** per `evaluate`, a **+47%** regression on the one number a leaf
value cannot afford to regress. Two guards fixed it and are in the code for
that reason — the walk is skipped entirely when the ladder rung is zero
(a player holding no symbols cannot care whether the race is alive), and
`terms::supremacy_live` stops at the *second* unreachable symbol, since
seven symbols exist and six win. The honest count,
`terms::supremacy_reachable`, is kept for the tests and the diagnostics
and is not on the hot path.
