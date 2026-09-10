# Round six: the one wonder effect that is not like the others

Round four built `WonderModel::Budget`, a per-effect price for an unbuilt
wonder, and measured it **negative** (−11.0 / −5.6 / −3.9 Elo). Its
follow-up note blamed at least part of that on
`duels_strategy::science::token_value`'s flat, unmeasured constants
feeding the Great Library channel — which is to say, on a channel that has
nothing to do with the one thing a strong player will tell you first:
**an extra turn is the most valuable thing a wonder can print.**

`docs/strategy-backlog.md` §2.1 puts all five play-again wonders in its top
tier and nothing else, and §1.2 explains why in terms this evaluation has no
other way to see: with strict alternation the whole slot sequence is
pre-determined, and an extra turn is the one thing that re-assigns every
remaining slot. A 1-ply evaluation cannot search that out; a constant is the
only instrument it has.

`Budget` cannot answer that. It moves eight channels at once, several of
them known-weak, so its verdict on extra turns is confounded with its
verdict on everything else. Round six therefore leaves `Budget` exactly
where it is and adds **one number** to the default
`WonderModel::Flat` path instead:
`EvalWeights::wonder_extra_turn_premium`, added on top of
`terms::wonder_power`'s flat "+3, this wonder has an effect" for a wonder
that prints play again, and nothing else. At `0.0` — `Config::v5` — it is
round five bit for bit (`tests/v5_identity.rs`). **The new default is
`9.0`**, so a play-again wonder is worth `12` where every other effect is
worth `3`.

## Which wonders, and one mechanic this deliberately does not price

Exactly five wonders print play again — Piraeus, The Appian Way, The
Hanging Gardens, The Sphinx and The Temple of Artemis — read off
`data/wonders.json` through `WonderDef::play_again` rather than by slug, and
pinned at five by
`v5_identity::an_extra_turn_premium_moves_only_the_five_play_again_wonders`.
Each grants exactly one extra turn, on construction, unconditionally; there
is no per-wonder difference in when or how it triggers.

The **Theology** progress token grants play again for *every* wonder its
holder builds, and the premium does not price that. That is a real
omission, taken on purpose: folding it in would have every unbuilt wonder in
a Theology holder's hand collect the premium at once, and the sweep below
would then be measuring two things. It is the obvious follow-up.

## Elo by magnitude

`phased:wprem=P` against `phased:base=v5` at `Nodes(1)`, **3200 games per
seed range**, paired and seat-swapped. Every interval is ±12.1.

```text
premium   total    seed 1    seed 100001   seed 200001   seed 300001   seed 400001
   1.5      4.5    +18.8       +32.5         +27.1
   3.0      6.0    +30.1       +48.5         +43.1
   6.0      9.0    +39.2       +53.5         +46.7
   7.5     10.5                                            +39.2         +52.6
   9.0     12.0    +45.3       +51.6         +49.9         +38.5         +51.9
  10.5     13.5                                            +38.5         +51.1
  12.0     15.0    +46.4       +48.4         +48.4         +41.1         +51.2
  15.0     18.0    +35.5       +38.3         +33.9
```

Positive on **every range at every magnitude tested** — the sign of the
effect is not in doubt — on a curve that rises to a broad plateau between 6
and 12 and falls away by 15. `9.0` is the middle of that plateau and has the
best mean over five ranges (+47.4); it is not distinguishable from `12.0`,
which reads +47.1 and beats it head-to-head by −2.7 / +5.0 / +9.2 over 3200
games on three ranges. `9.0` is taken as the smaller change of two that
measure the same.

The shipped default is then confirmed on **two further disjoint ranges it
was not chosen on**: `phased` against `phased:base=v5` reads **+53.5**
[+41.3, +65.7] at seed 500001 and **+45.9** [+33.7, +58.0] at seed 600001,
3200 games each. As a null calibration, `phased` against `phased:wprem=9` —
the same configuration under two names, differing only in the tie-break RNG
seed the arena hands each side — reads **+1.7** [−22.3, +25.8].

## Two controls, because +47 Elo for one constant deserves them

A single number that buys forty-seven Elo is exactly the kind of result that
is usually measuring something other than what it claims. Two controls:

**1. Is it just "the wonder term wants a bigger weight"?** No.
`EvalWeights::wonder_potential` scales the *whole* flat wonder term;
raising it from its default 0.5 buys almost nothing:

```text
wonder_potential   seed 1    seed 100001
       0.60         -5.5       +5.9
       0.70         -1.4       +3.5
       0.85         +6.9       +7.8
       1.00        +12.6       +9.2
```

**2. Is it the premium, or is it *which wonders get it*?** The latter,
decisively. The identical `+9`, applied to the four **pending-effect**
wonders (Circus Maximus, the Statue of Zeus, the Mausoleum, the Great
Library) instead of the five play-again ones, is worth **−66.0 / −76.9 /
−66.9** Elo over 3200 games on each of three disjoint ranges — and `+3`
there is worth −26.9 / −25.2 / −24.4. A hundred and twenty Elo separates
the same constant on two comparable sets of wonders. The result is about
extra turns specifically, not about the shape or magnitude of the term.

## Against the ladder

600 games per seed range at `Nodes(2000)` for the searching opponent — the
budget kind still cannot matter for `phased`, which ignores it, but it is
what decides how strong the opponent is:

```text
                       this agent          phased:base=v5
vs alphabeta   seed 1   -83.7 [±28.6]      -122.8 [±29.5]
           seed 100001  -84.9 [±28.6]      -153.8 [±30.6]
vs mcts-uct    seed 1  -217.7 [±33.4]      -300.5 [±38.8]
           seed 100001 -214.4 [±33.2]      -310.9 [±39.7]
```

`matchup_profile`'s win-condition spread says it is not one route: against
`alphabeta` the wins go 165 → 185 civilian *and* 29 → 40 scientific, and
against `mcts-uct` 57 → 91 civilian and 32 → 41 scientific. Both routes
improve on both ranges.

## The behaviour actually changed — and not in the obvious place

`duels-arena/examples/wonder_audit.rs` now breaks out the five play-again
wonders and counts the decisions each side actually took, and how many of
them were extra turns. The first run of it was a **self-play** pair, which
showed the premium building play-again wonders *less* often (94% → 88%) and
*later* (turn 29.7 → 31.4) — the exact opposite of the intent, and worth
recording because it is what a self-play audit of a *draft* preference
always shows: which wonders are in play is fixed by the deal, and when both
sides share the preference the only thing it can do is cancel.

Run asymmetrically, 1000 games at `Nodes(1)`, the mechanism is unambiguous:

```text
                                   phased (premium 9)   phased:base=v5
play-again wonders drafted               2100                1250
play-again wonders built                 1882                1170
...as a share of the ones drafted         90%                 94%
decisions taken per game                34.87               34.06
...of which extra turns                  4.22                3.45
wonders left drafted and unbuilt          0.56                0.68
```

The premium is a **draft** signal, not a build signal. It wins the split of
the eight dealt wonders: 63% of the play-again wonders in play end up on
this side against 37%, and that converts into **22% more extra turns
actually taken** and eight tenths of a decision more per game. The build
*rate* falls slightly because the side now holding five play-again wonders
runs into the seven-slot cap that the side holding two never reaches — and
it still leaves fewer wonders unbuilt in absolute terms.

## Cost

One flag test and one addition on a constant, per unbuilt wonder, on a
branch that is not even taken at the default weight of every other wonder.
`examples/decision_cost.rs`, every configuration timed on the same 2157
positions:

```text
v5 (the round-five agent)             63.6 us/decision
default, extra-turn premium off       63.0 us/decision
default (round six)                   63.3 us/decision   +0.5%, and v5 reads higher
```

Three timings of what is arithmetically the same work plus one `f64`
addition, spread over 0.6 us — the premium is free, and the row ordering
(round five reading *slowest* of the three) is the benchmark's run-to-run
noise rather than a cost. Nothing is root-fixed for it because there is
nothing positional to fix: `play_again` is a property of the wonder, not of
the position, so there is no per-decision table for it to live in.
