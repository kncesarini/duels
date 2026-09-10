# Cross-round measurement tables

Measurements that span rounds rather than belonging to one of them, relocated
verbatim from `duels-eval`'s crate docs alongside the round-by-round history in
this directory. The five sections below are the round-one-to-three era tables
that later rounds kept citing. See [README.md](README.md) for the index.

## Measured

All paired and seat-swapped through `duels-arena`, at `Nodes(1)` unless
noted. A 1-ply agent ignores its budget entirely, so a `TimeMs` budget
changes nothing for it — `--budget time_ms:50` reproduces the `Nodes(1)`
result below to the game — and the only wall-clock figure worth reporting
is the per-decision cost further down.

Against the agent this crate shipped with (`phased:base=v1`), 600 games per
seed range, adding one change at a time:

```text
                                         seed 1              seed 5001
next_age_start halved (alone)            +9 [-25, +43]       (neutral)
MilitaryModel::Band (alone)              +14 [-20, +48]      (neutral)
CoinModel::Smooth (alone)                -17 [-51, +17]      (neutral)
the three together                       +38 [+10, +66]      +47 [+19, +75]
  + EconomyModel::Bill                   +167 [+136, +198]   +201 [+169, +234]
  + the opponent menu                    +208 [+175, +241]   +228 [+194, +262]
  + chain equity                         +233 [+199, +267]   +226 [+192, +260]
  + the two fitted weights = the default +329 [+288, +371]   +292 [+254, +330]
```

Every row but the last holds the two fitted weights at the value their own
units imply, so the table is a clean "what did each idea buy". They are
reproducible as
`phased:base=v1,start1=1.5,start2=1.0,mil=band,coin=smooth,band=1.0,bill=1.0,chaineq=0,lambda=0`
plus, in order, `econ=bill`, `lambda=0.6`, `chaineq=1.0`; the last row is
the bare `phased`.

and, at the default, the same thing as a leave-one-out:

```text
                           seed 1    seed 5001    the term is worth
default                    +329      +292
economy_model = legacy      +61       +58         +268 / +234
menu lambda = 0            +191      +216         +139 /  +76
military_model = legacy    +206      +201         +123 /  +91
military_band 2.0 -> 1.0   +265      +267          +64 /  +25
coin_model = legacy        +277      +287          +52 /   +5
next_age_start back to 4/3 +292      +277          +38 /  +15
chain_equity = 0           +278      +309          +51 /  -17
```

Two disjoint seed ranges agree in sign on everything except chain equity,
which is indistinguishable from zero and is kept on the strength of the
pooled result and of the fact that `menu` needs its table anyway. The
three Round-one fixes are individually noise and jointly worth about +40;
the resource bill is more than half the round on its own.

Against the rest of the ladder, 400 games per seed range at seeds 1 and
5001 (`Nodes(1)`; `alphabeta` at `Nodes(2000)` over 200 games each):

```text
                   new default          previous agent
vs random          400-0 / 398-2        291-9 over 300   (Elo +595)
vs greedy          399-1 / 397-3        297-3 over 300   (Elo +772)
vs greedy-ev       398-2 / 396-4        392-8 over 400   (Elo +666)
vs strategist      399-1 / 399-1        296-4 over 300   (Elo +728)
vs alphabeta       32-168 / 33-167      32-168 / 18-182
```

No regressions: `alphabeta` is the only ladder opponent close enough to
measure a change against, and 65 wins in 400 against the previous agent's
50 is an improvement.

## `mcts-uct`: the bar that was met, and the one that was not

Over 400 paired games at `Nodes(2000)`, `examples/matchup_profile.rs`:

```text
                       wins    military  science  civilian
round one       s1     32/400         0       32         0
round one       s5001  30/400         0       30         0
round two       s1     21/400         1        9        11
round two       s5001  24/400         3        6        15
round three     s1     44/400         1       19        24
round three     s5001  33/400         1       14        18
```

The last two rows are this round's, and they undo the paragraph below:
the aggregate rate is back up, to 77/800 pooled (9.6%) against round one's
62/800 and round two's 45/800, *and* the win-condition spread round two
bought is intact. What is left of the honest negative is that `mcts-uct`
still wins nine games in ten, for the reason this project has recorded
since its first agent: a 1-ply evaluation loses a long positional game to a
real search.

The **win-condition spread is fixed, on both seed ranges**: the agent now
wins by all three routes rather than only one. It also stops conceding the
military track — the pawn's mean final position moves from -5.9 in the
previous agent's games to -1.7, and `mcts-uct`'s own military-supremacy
wins drop from 101 in 400 to 40.

The **aggregate rate against `mcts-uct` got worse**, 62/800 to 45/800
pooled across the two seed ranges (7.8% to 5.6%),
and that is the honest negative of this round. It is the one opponent that
moved the wrong way while everything else moved a long way right, which is
exactly the failure mode you would expect from fitting two weights against
one baseline. Three things are worth saying about it. First, `mcts-uct`
plays a fast yellow/red tempo game (3.8 red and 4.6 yellow cards a game
against this agent's 2.6 and 2.5) and wins 334 of its 376 games on points,
not on a race: a 1-ply evaluation losing a long positional game to a real
search is this project's oldest finding, not a new one. Second, the
previous agent's 7.8% was *entirely* scientific supremacy on both seed
ranges — it entered one lottery every game and lost every other game it
played, 738-0 — so the two numbers do not measure the same kind of
competence. Third, at a *wall-clock* budget
(`time_ms:100`, 100 games, seed 1, single match on a quiet machine) the two
are level: both win 6, the old agent's six all by scientific supremacy and
the new agent's split 2 science / 4 civilian.

## Choosing `military_band`

Round two shipped `2.0` and flagged it as a judgement to revisit rather
than inherit. Round three revisited it and the answer changed, so both
tables are kept here: the argument is more useful than the number.

Round two's sweep, against `Config::v1`:

```text
band   Elo vs v1 (s1/s5001)   vs alphabeta   mil. wins vs mcts   Age I red keep
1.0      +265 / +267            72/400            0                20.9%
1.5      +322 / +261            68/400            -                29.1%
1.75     +324 / +290            56/400            -                38.1%
2.0      +329 / +292            65/400            1                41.0%   <- was default
2.5      +334 / +322            45/400            1 and 6          45.5%
```

At the time, `2.0` was the largest value that cleared every bar at once and
`1.0` was the only value measured that never beat `mcts-uct` militarily.
What changed is not the sweep but what else is in the evaluation. The
inflated slope was buying one thing — the occasional supremacy win — by
doubling the honest price of every shield in the game, all game, whether or
not anything was about to happen. `rails` buys the same thing by asking
whether a closing card *exists and is affordable*, which is both cheaper
and correct. With the rails in place, `1.0` is simply better:

```text
band   Elo vs the default (s1/s5001)   vs alphabeta      mil. wins vs mcts   Age I red keep
1.0     (the default)                   77/400, 76/400        1 and 1          26.3%
2.0      -56.0 / -31.8                  —                     —                41.0%
```

The Age I red keep rate lands at 26.3%, which is where this project's
calibration guidance expected an honest slope to put it (~20-25%), and the
military-supremacy column against `mcts-uct` is *not* zero any more — the
thing `1.0` was previously rejected for. `2.0` remains one spec string away
(`phased:band=2.0`), and the two tables above are the whole argument.

## What one decision costs

`examples/decision_cost.rs`, every configuration timed on the *same* 5715
positions (timing each one on its own self-play games measures the wrong
thing: a configuration that steers towards positions with fewer chance
outcomes looks faster while doing more work per decision, and written that
way this benchmark reported the full default as 15% *cheaper* than the same
agent with the menu term switched off).

```text
v1 (the round-one agent)                36.6 us/decision
default, menu and chain equity off      41.7 us/decision   +14%
default, menu off                       43.0 us/decision   +17%
default, rails off                      47.4 us/decision   +29%
default, one-sided menu shield price    47.4 us/decision   +30%
default (menu lambda = 0.6)             47.0 us/decision   +29%
v2 (the round-two agent)                47.6 us/decision   +30%
```

`menu::menu_term` is the first term in this crate whose cost scales with
the number of *chance outcomes* an action has — Age I's worst case is a
two-slot reveal from an eleven-card pool, over a hundred outcomes for one
candidate — so it is the one that was worth measuring. It costs about 5 us
per decision: the per-outcome work is bounded by the handful of accessible
slots, not by the outcome count alone.

**Round three costs nothing measurable.** The default, the same agent with
the rails switched off, and the round-two agent are 47.0, 47.4 and 47.6
us — a spread smaller than the run-to-run variation, with the full default
nominally the *fastest* of the three. That is not an accident of the
benchmark: `duels_strategy::closing_sources` takes a one-comparison early
exit unless somebody is within one action's shields of a capital or holds
five distinct symbols, which is a few hundred of every seven thousand
decisions, and Rail C's denial walk runs only inside that.

## Take profile

`examples/take_profile.rs`, Age I keep rates over 40 self-play games:

```text
            brown  grey   blue   green  yellow  red
phased      92.9   75.9   78.8   51.4   62.3    26.3
phased-v1   68.1   88.9   87.7   79.1   69.7     1.5
mcts-uct    80.2   76.4   73.6   25.9   78.9    50.7
```

Red moves from "never" (round one's 1.5%) to 26.3%, which is where this
project's calibration guidance expected an honest slope to put it. Round
two's `military_band = 2.0` overshot to 41%; see "Choosing
`military_band`" above.

The green column is the cost of round two, and it is a real one: the
resource bill makes production and denial compete with the science ladder.
Round three does not undo it — 51.4% against round one's 79.1% — but it no
longer costs games to `mcts-uct`, where the science-supremacy column is
back to 19 and 14 wins in 400.
