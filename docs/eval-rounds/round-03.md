# Round three: a rail instead of a gradient

Round two left one term doing a job it was never shaped for.
`terms::military_urgency` is a smooth quadratic in how far past the
second loot token the conflict pawn sits, and it was the only thing in the
evaluation claiming to notice an imminent loss. It is blind to whether a
closing card exists, blind to whether anybody can afford it, blind to
whether there are two of them, and it pays out at pawn positions where
nothing at all is about to happen. `military_band = 2.0` was carrying the
same load from the other direction: double the honest price of *every*
shield in the game, in the hope that the occasional supremacy win made it
back.

Round three replaces both with a question that has a yes/no answer,
adapted from the fix that was worth +26 Elo in `mcts-uct`'s rollout
policy — *take an available win, block an available one-move loss*. The
1-ply form asks it of the **post-action state** the evaluator is already
scoring, and answers it from the rules: `duels_strategy::closing_sources`
enumerates the actions that would end the game outright for either player,
cross-checked action-for-action against the engine by
`duels-strategy/tests/closing_sources_cross_check.rs`.

Five changes, all `Config` options, all reproduced exactly by
`Config::v2` (`tests/v2_identity.rs`):

1. **`rails` (default on).** Rails B, C and C′ over the post-action
   state. Elo-neutral, and adopted on the audit rather than on the Elo —
   see below.
2. **`military_band` 2.0 → 1.0 (the term's own units).** The single
   largest gain of the round, and the one the rails paid for: with a real
   imminence detector in place, the inflated slope has nothing left to buy.
3. **`MenuShieldPricing::Differenced` (default on).** `menu` priced a
   red card's shields at `k ×` a one-sided slope while the evaluation that
   scored the resulting position priced the same shields *differenced*
   across both players and under both players' weights — a real unit
   inconsistency, and a systematic under-valuation of red cards on the
   menu. Now an exact finite difference
   (`terms::military_shield_delta`), Strategy token included. Elo-neutral;
   adopted because it removes an inconsistency, not because it wins games.
4. **`Config::military_horizon` (default off).** A better-motivated
   smoothing width that makes no measurable difference. Honest negative.
5. **`EvalWeights::production_lock_in` (default off).** Age III really
   does print no brown or grey card, so an Age III resource bill is a fact
   rather than a projection — and amplifying the term by that consistently
   *costs* Elo. Honest negative.

## The audit, which is the actual result

A rail is a guarantee, and a guarantee is audited rather than sampled.
`duels-arena/examples/rail_audit.rs` replays real games and recomputes, at
every decision, **from the engine** rather than from the agent's own reads,
what was available. 200 self-play games per seed range at `Nodes(1)`:

```text
                                             this agent            phased:base=v2
                                           seed 1    seed 5001    seed 1    seed 5001
Rail A  an available win was taken          19/19       32/32      21/21       30/30
Rail B  an available block was taken        39/39       41/41      27/37       31/34
Rail C  an undeniable close was taken         2/2        5/6         2/2         5/7
        ...of those already on the table      0/0        1/1         0/0         2/2
B/C'    a rail firing on a closer was right  703/703    631/631    817/817     587/587
C       a rail calling it decisive was right   6/6        9/9        20/20         8/8
supremacy losses that were NOT blockable      9/9      12/12        5/9        9/12
```

**The bottom row is the round.** The round-two agent lost seven games
across the two seed ranges to a military or scientific supremacy that a
candidate on the table at its own last decision would have removed. This
agent loses none: **zero blockable supremacy losses**, on both seed ranges,
and Rail B at 100% against 73% and 91%.

The last two rows before the bottom are *precision*: when a rail fires,
does the engine agree? That matters more than recall, since a rail that
fires wrongly misvalues a move by five hundred points. It is **100% on
every run**, and on the `Nodes(2000)` runs against `alphabeta` and
`mcts-uct` too. (The detector is the same code in both columns, so the
`phased:base=v2` figures are the same read taken over the positions that
agent reached rather than a property of the round-two agent, which never
consults it.)

Rail C's recall is the honest negative of the round. It reads 5/6 on one
range, and the miss is not a fault so much as a boundary: of the eight
undeniable closes the audit found across the two ranges, only one was
*already on the table* when the candidate was played. The rest were closes
that every opposing reply happened to uncover — real, and a searching agent
would find them, but invisible to a rail that reads the post-action state
and does no search. Restricted to the closes it can see, Rail C is 1/1 and
0/0, and 1/1 again against `alphabeta`. It is a rare guard that is right
when it speaks, not a term that earns its keep every game, and both halves
are reported rather than only the flattering one.

The same audit against the two search agents, 200 games each at
`Nodes(2000)` and seed 1, which is where the losses actually are:

```text
                                       vs alphabeta     vs mcts-uct
Rail A  an available win was taken        14/14            12/12
Rail B  an available block was taken     328/328          138/138
Rail C  an undeniable close was taken       3/4              3/3
B/C'    firing on a closer was right    6521/6521        2337/2337
C       calling it decisive was right       7/7            19/19
supremacy losses that were NOT blockable  72/72            25/25
```

Ninety-seven military or scientific supremacy losses between them, and not
one of them had a candidate on the table that would have removed the
threat. Against a real search the rails have three hundred and twenty-eight
blocks to make and make all of them.

## Elo, measured one change at a time

Against `phased:base=v2`, 600 games per seed range at `Nodes(1)`:

```text
                                 seed 1                 seed 5001
the new default            +35.4 [+7.5, +63.3]     +37.7 [+9.8, +65.7]
```

and, as a leave-one-out against the new default itself (800 games per seed
range, so each row is a paired head-to-head of exactly that one change):

```text
                                  seed 1     seed 5001    the change is worth
military_band back to 2.0          -56.0        -31.8       +56 / +32
production lock-in switched on     -20.4        -13.0       (off is better)
rails switched off                  -9.5         -2.6       +10 /  +3
one-sided menu shield price         -3.9         +4.3        neutral
horizon = 2                         +5.2         -0.9        neutral
horizon = 3                         +2.6         +3.5        neutral
horizon = 5                         +1.7         +2.6        neutral
```

Only one row's confidence interval clears zero on both ranges, and it is
`military_band`. The rails are Elo-neutral in self-play and always were
going to be: they fire on a few hundred of seven thousand decisions, and
two agents that both hold the same rails cannot gain from them against each
other. What they buy is the bottom row of the audit table.

## Against the ladder

400 games per seed range at seeds 1 and 5001 (`Nodes(1)`; `alphabeta` and
`mcts-uct` at `Nodes(2000)`):

```text
                   this agent           round two (phased:base=v2)
vs random          400-0  / 399-1       398-2 / 396-4 (round two's own figures)
vs greedy          400-0  / 397-3
vs greedy-ev       398-2  / 395-5
vs strategist      398-2  / 399-1
vs alphabeta        77/400 / 76/400      66/400 / 80/400
vs mcts-uct         44/400 / —           21/400 / 24/400 (round two's own figures)
```

`alphabeta` is the only ladder opponent close enough to measure a change
against, and 77 and 76 wins in 400 clear the ~72 that `military_band = 1.0`
was worth in round two's own sweep. Against `mcts-uct`, 44 wins in 400
(11.0%) against round two's 21 and 24, with the win-condition spread
holding — military 1, science 19, civilian 24 — so the military-supremacy
column is nonzero at `band = 1.0`, which round two reported it never was.
