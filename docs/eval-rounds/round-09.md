# Round nine: the wonder term was over-paying, and fixing it helps only the agent that does not matter

Round nine is **three new measurement instruments, one large calibration
defect found and quantified, one derived fix for it that is worth +30 to
+86 Elo to `phased` and −30 to `mcts-eval`, one exact refinement worth
nothing, and a diagnosis of a flagged position that turns out not to be an
evaluation error at all**. `Config::default` is **unchanged** — the first
round of this crate that measured its candidates and shipped none of them,
which is why there is no `v9()` and why `tests/round_nine_identity.rs` is
not a `vN_identity.rs`.

## The brief, and the honest answer to it

The brief was science, in detail: that science evaluation should not be
purely age-based, that it matters which turn of the age it is and how many
cards remain, that a missing symbol being *face up and known* is a different
claim from its being *face down but plausibly reachable*, that the relative
extra-turn-wonder count and who starts Age III should feed a science read.
Alongside it, flagged as separate, a cross-cutting note: **wonders seem
overrated, because the evaluation builds them too early while they are
still expensive.**

Every one of those was checked empirically rather than argued about, and the
answer is not the one the brief expected. **The science-specific factors are
mostly already priced or immaterial. The cross-cutting wonder note is
correct, large, and the two turn out to be the same finding** — a player
ahead on unbuilt play-again wonders is over-valued by up to twenty victory
points, which is simultaneously the parity factor the science brief named
and the over-rating the wonder note named.

And then the fix for it, derived rather than fitted, is worth **+30 Elo to
`phased` in self-play, +86 and +56 against `mcts-uct`, +3 against
`alphabeta` — and −24 and −35 to `mcts-eval`**. `mcts-eval` is the
consumer that decides, so it ships as `WonderModel::Rationed`, off by
default, with every column written down. That spread is the round's most
useful output and the reason the section below is long.

## Three instruments

**`examples/science_calibration.rs --factors`** (extended). Round eight's
instrument bucketed positions by the mover's distinct-symbol count. Round
nine adds the five factors the brief named — cards left in the structure,
how many symbols are still assemblable, how many missing symbols are face
up *right now*, the unbuilt-play-again-wonder differential, and who took
the first decision of the age — and splits every `(age, symbols)` cell by
each of them, plus a `δ_k` fit run *inside* each bin. The logic is what
makes the tables worth reading: a gap that is the same in every bin of a
factor is a factor the evaluation already prices; **a gap that moves across
the bins is a term the evaluation is missing**, and that is the only kind
of finding that names a new term.

**`examples/wonder_calibration.rs`** (new). The same method aimed at
unbuilt wonders: bucket by how many a player holds, by the play-again
differential, and by `terms::wonder_p_build` — the quantity the flat model
does not read at all.

**`examples/position_probe.rs`** (new). The offline half of advanced mode's
flag loop: replay an exported `{ seed, moves }`, print the ranked action
list the analysis endpoint would show, print the same list with one term
switched off, and print the per-term breakdown behind both.
`tests/flagged_positions.rs` is the committed form for the position this
round was handed.

## The flagged position: one real finding, one non-finding

Seed 1, thirteen moves, turn 13 of Age I, `value = -18.4436` and
`win_probability = 0.3321` — reconstructed exactly, and pinned in
`tests/flagged_positions.rs`.

**The "every legal action reads lower than standing still" anomaly is not
an evaluation error.** The five actions read 0.236-0.270 against a standing
0.332, and the whole of that is `menu::menu_term` changing sign with
whoever moves *next*. It is `+λ·menu(me)` on a pre-move state and
`−λ·menu(opp)` on a post-move one — here `+4.09` and about `−8.06` victory
points, a twelve-point step that has nothing to do with the moves being
bad. Switch the menu off on both sides and the ordering reverses exactly:
standing `-22.53`, best action `-18.20`.

```text
                             menu on            menu off
standing                     -18.44  p=0.332    -22.53  p=0.299
Build 16 (theater)           -26.25  p=0.270    -18.20  p=0.334
Discard 12 (clay-pit)        -26.89  p=0.265    -20.62  p=0.314
Build 15 (garrison)          -31.03  p=0.236    -22.97  p=0.295
```

`evaluate` is antisymmetric and internally consistent, and it is not
*supposed* to be comparable across a change of mover: a 1-ply agent takes
an argmax over candidates that all sit on the same side of the term. The
defect is in displaying a pre-action value beside post-action ones as if
they were the same quantity, and it is **general** — it will fire on every
position where the mover has a decent menu, which is most of them. The fix
is not in this crate. What `duels-server`'s analysis endpoint wants is
either the previous decision's post-action value as the baseline, or the
*pass* baseline (the best score available, so the deltas are between
candidates), and both are its own call to make. It is written down here
because the second-order effect is worse than the cosmetic one: an operator
reading that overlay concludes the evaluation is broken when it is not, and
flags positions accordingly.

**The wonder note is a real finding.** One move earlier, with seven coins
in hand, the evaluation ranks all three ways of building The Great
Lighthouse above every card build available, and reading the two on the same
side of the menu term it calls the build worth `+11.4` victory points for
seven coins. The breakdown says where that comes from, and it is not the
wonder term:

```text
                         before the build    after      the build is worth
terms::resource_bill          -35.06        -24.98            +10.08
terms::development_value       +3.94         +8.15             +4.21
printed victory points          0.00         +4.00             +4.00
coins (7 -> 0)                 +4.59          0.00             -4.59
wonder_potential              +16.80        +13.80             -3.00
```

A produce-a-raw-material wonder cuts the projected resource bill by ten
victory points, and seven coins are priced at four and a half. Both halves
are defensible on their own; what is missing is that **the same wonder built
later costs fewer coins, and no term in this evaluation can represent
"build it when it is cheap" as an alternative to "build it now".** A
`Build`-versus-`BuildWonder` comparison at 1 ply is a comparison against
*not building it at all*, and against that it genuinely is a good move.
Round nine did not build an option-value term; see "what is left".

## What each science factor actually said

One fixed corpus, 2,000 games, two science-tilted seats
(`phased:base=v7,sci=3.0`, exactly as round eight generated its), read by
`Config::v8()`. First-reach positions, the player to move.

**1. Round eight's own named follow-up is not supported.** Round eight left
"an age- or `decisions_left`-scaled rung is the obvious follow-up" on the
strength of Age III reading `δ4 = −9.6` and `δ5 = −14.7` where Age II was
flat. Split Age III by whether six symbols are still assemblable and that
disappears:

```text
 age   reachable          d1     d2     d3     d4     d5     (mover positions)
  2    6+ (race live)    +0.1   -5.4   -7.1   -1.6   -9.2     7341..187
  2    <=5 (dead)        +0.2   -4.5  -11.0   -2.0   -3.6     7768..0
  3    6+ (race live)   +11.3  +15.1   +6.2   +2.2   -3.3        0..1787
  3    <=5 (dead)        -1.7   -5.0   -8.8  -11.0  -15.5     6279..739
```

**Where the race is live the rung is right to within a couple of victory
points in both ages** (`+2.2` and `−3.3` at four and five symbols in Age
III). The whole of round eight's Age III over-credit is the *dead-race*
population — and `ScienceWeights::dead_race_scale` is already `0.0`
there, so the residual is not the rung at all. It is the rest of a science
city: four or five green cards' printed points, credited at face value by
`terms::card_and_token_vp`, for card picks that bought no production, no
shields and no coins. An age-scaled rung would have been fitted against the
wrong thing. **Not built, and the reason written down** — which is exactly
what round eight's instrument was for.

**2. The within-age gradient is real but small, and its obvious cause is
not the cause.** At three symbols in Age II the fitted correction runs
`−3.8` early in the age, `−9.0` mid and `−13.1` late — the project owner's
"it matters which turn of the age it is", worth about nine victory points.
The natural mechanism is that `terms::supremacy_live` grows *more*
optimistic as an age drains: three of every age's cards go back in the box
unseen, a boxed card is in none of the masks the walk consults, and by the
last turns of an age most of what is left in the deck list is exactly
those. `ReachModel::Structure` fixes that exactly (see below) and moves
the gradient by **0.7 of a victory point** (`−13.1` to `−12.4`). So the
gradient is real and this is not what causes it.

**3. Face-up against face-down missing symbols: no signal.** Age II at two
symbols reads `+0.038 / +0.015 / +0.125` across zero, one and two-or-more
missing symbols face up; Age III at three reads `−0.049 / −0.123 / −0.172`,
the other way. Signs disagree between ages and the thin bins carry ±0.17.
Nothing to price.

**4. Who started the age: a consistent signal, and confounded.** A player
who took the age's first decision beats the prediction and one who did not
falls short of it, on all four Age III symbol rows and both Age II rows
(`+0.121` against `−0.055` at two symbols, `−0.051` against `−0.167` at
three). It is worth 0.1-0.17 of win probability and it is exactly the
`age_start_lab` effect this project already measured — but *who* starts is
decided by the militarily weaker player, so the split is not a random
assignment and this corpus cannot separate "starting is good" from "the
kind of player who gets to start". Left as a follow-up with a design note:
it wants the `age_start_policy` harness, not this instrument.

**5. The extra-turn differential is a large, signed, mispriced factor — and
in the opposite direction to the read that prompted it.** This is the
finding.

```text
 age  extra-turn diff      n   predicted   actual     gap
  3   behind             114     0.558      0.675    +0.118
  3   level              424     0.523      0.512    -0.011
  3   ahead              126     0.493      0.294    -0.200
```

A player **ahead** on unbuilt play-again wonders wins far *less* than
`win_probability` claims, and one **behind** wins more. The brief's read
was that an extra-turn advantage is worth *more* in a science race; the data
says the evaluation is already paying too much for it.

## The wonder term, which is what that factor was actually measuring

`examples/wonder_calibration.rs` on an untilted corpus (4,000 games,
571,814 player-positions, read by `Config::v8()`) says the error is not
about play-again wonders specifically at all — it is about **unbuilt wonders
that are never going to be built**. Three independent slices, mover
positions:

```text
unbuilt wonders held             n     predicted   actual     gap    pot. vp
  age 3, none               38071       0.561      0.697    +0.136     0.00
  age 3, one                28469       0.477      0.381    -0.096     0.46
  age 3, two                15523       0.494      0.264    -0.229     8.62
  age 3, four                 162       0.380      0.099    -0.281    21.37

play-again differential          n     predicted   actual     gap    pot. vp
  age 3, -1                  7783       0.557      0.748    +0.191     0.50
  age 3, level              65985       0.517      0.501    -0.017     1.25
  age 3, +1                  8136       0.476      0.259    -0.217     8.32

p_build                          n     predicted   actual     gap    pot. vp
  age 3, < 0.40             29108       0.470      0.333    -0.138     1.31
  age 3, 0.40-0.75          15446       0.497      0.329    -0.169     8.06
  age 3, 0.75-0.99           1033       0.539      0.530    -0.010     9.21
```

and, in the actionable units, the fitted additive correction against the
`p_build ≥ 0.99` bin:

```text
 age      <0.40   0.40-0.75   0.75-0.99   >=0.99 (ref)
   1      -23.2      -30.2        -6.9         0.0
   2      -10.3      -19.3       -19.5         0.0
   3      -10.7      -19.7       -14.0         0.0
```

**Ten to thirty victory points, in every age, concentrated exactly where
`p_build` is low** — and `p_build` is the one thing
`WonderModel::Flat` does not read. The flat model pays
`wonder_potential × wonder_power` for a drafted-but-unbuilt wonder at full
weight until the seven-wonder cap closes, and `0.5` is a constant standing
in for "it will probably get built". That is not a constant. It is
`terms::wonder_p_build`, which has existed since round five as a
standalone probability estimate, factored out of
`WonderModel::Budget` precisely because it has nothing to do with how a
wonder's *effects* are priced.

`WonderModel::Rationed` reads it: `p_build × ` the flat model's own
per-effect power, extra-turn premium included, unchanged. It is the one
channel of round four's eight-channel `Budget` bundle that this measurement
says is right, taken on its own.

### Elo, and the control that says what the gain is about

`phased:wonder=rationed,wonder_potential=P` against `phased:base=v8` at
`Nodes(1)`, 3200 games per seed range, paired and seat-swapped. Every
interval is ±12.1.

```text
  P     seed 1   seed 5001   seed 9001   seed 13001   seed 17001
 0.35   -34.2
 0.50   -19.6
 0.75    +1.1
 1.00   +18.8      +22.7
 1.25   +29.7      +31.1       +30.5       +19.6        +17.4
 1.50   +24.4      +29.2
 2.00    -1.2
```

Unimodal with a plateau at 1.25-1.5, and `1.25` is positive on all five
ranges with every interval clearing zero. **The control is the result**:
the *same weight un-rationed* — `WonderModel::Flat` at
`wonder_potential = 1.25` — is worth **−73.0 / −75.5**, and `1.0` and
`0.75` there read −30.8 and −5.4. A hundred Elo separates one constant with
and without the `p_build` factor, so the gain is the rationing and not the
magnitude.

It also is not the extra-turn premium in disguise: re-swept on top of the
rationed model, `EvalWeights::wonder_extra_turn_premium` still wants `9.0`
(4.5 reads −16.0, 6.0 −10.1, 12.0 −2.1 and 0.0 **−80.0** against it), which
is where round six left it.

### It transfers to one unrelated searcher, not the other — and reverses inside `mcts-eval`

800 games per seed range, the searchers at `Nodes(2000)`:

```text
                            phased (default)          + the rationed model    worth
vs alphabeta    seed 1   +53.3 [+29.0, +77.7]      +56.4 [+32.1, +80.8]       +3.1
vs mcts-uct     seed 1  -133.3 [-159.1, -107.4]    -47.1 [-71.4, -22.8]      +86.2
                seed 5001 -118.0 [-143.4, -92.5]   -61.8 [-86.2, -37.4]      +56.2
```

**+86 and +56 Elo against `mcts-uct` and +3 against `alphabeta`.** So it is
not self-play overfitting — it is the largest movement against `mcts-uct`
since round seven — and it is also not uniform: `alphabeta` cannot tell the
difference. The victory kinds say the gain against `mcts-uct` is not one
route: on the second range this side's wins go 269 → 329, scientific
10 → 28 and civilian 252 → 288, while `mcts-uct`'s civilian wins fall
394 → 315. (The `+53.3` is also a useful null: it reproduces round eight's
own figure exactly, which is what "the default path is bit-identical" looks
like from the outside.)

And then the measurement that decides it. `mcts-eval` against
`mcts-eval:eval=v8` — the same binary either side, differing only in what
`Config::default` returns — at `Nodes(2000)`, 3200 games per range:

```text
                 seed 1                    seed 5001
  -24.4 [-36.4, -12.3]        -34.6 [-46.7, -22.5]
```

Negative on both ranges, both intervals clear of zero, and the victory kinds
say what happened: on the second range the rationed side wins **1112
civilian games against the anchor's 1451**, while its scientific wins go
40 → 59. That is the crate docs' own account of `mcts-eval`'s blend read
back at us — the evaluation half is there to supply *civilian-score
judgement*, and raising one term two and a half times spends that away.

**So this is the sharpest policy-versus-value split this crate has
measured, and it is the mirror image of round seven's.** Round seven found
a variant that predicted the winner two to three points better and was −29
Elo as a leaf. Round nine found one that is +30 Elo as a *policy* and −30 as
a leaf. The two findings are the same shape from opposite ends, and together
they say the thing worth carrying forward: **a 1-ply argmax is invariant to
a term's magnitude and a leaf value is not**, so a round that improves
`phased` by re-scaling a term has learned nothing about `mcts-eval` until it
runs the A/B. Round seven's advice — "the instrument that predicted
`mcts-eval`'s direction was the boring one: `phased` Elo, attenuated" —
needs amending: attenuated, and sometimes sign-flipped.

Round nine looked for a formulation that paid both and did not find one.
`EvalWeights::wonder_p_build_ref` exists because of that search: at
`terms::OPENING_P_BUILD` it is the same decay with the *opening scale
preserved* — `min(1, p_build / (7/8))`, so the term is worth exactly what
the flat model paid until wonders start going up. That is the shape without
the change of scale, and `phased` does not want it: **−11.8** at
`wonder_potential = 0.5` and **+5.5** at `0.65`. Nor does routing the draft
signal through the premium instead: rationed at `0.5` with the premium at
18 and 30 reads **+1.1** and **+6.3**. `phased`'s thirty Elo needs the
scale, and the scale is what `mcts-eval` cannot afford.

## `ReachModel::Structure`: exact, and worth nothing

The project owner's face-up-against-face-down distinction, built and
measured. `terms::supremacy_live`'s walk calls a symbol reachable if
*some* card printing it is not provably gone and belongs to the current age
or a later one — which counts the three cards every age returns to the box
unseen, for the whole of that age.
`ReachModel::Structure` reads the current age off the structure instead: a
current-age symbol counts if it is **face up in the structure**, or if the
structure still holds at least one face-down card. Later ages are read from
the deck list exactly as before, and the whole refinement stands down when
the structure is empty, because `state.age()` is then an age whose cards are
all still coming.

It is exact at the end of an age — with nothing face down left, the only
current-age symbols obtainable are the ones on the board — and it can only
ever call *fewer* symbols reachable, which
`round_nine_identity::the_structural_reach_model_never_calls_more_symbols_reachable`
asserts over real games rather than argues.

And it measures at nothing. **+1.1 / +2.6** Elo for `phased` against
`phased:base=v8` over 3200 games on each of two disjoint seed ranges, and
the calibration it was built for moves by under a victory point (Age II's
three-symbol late bin `−13.1 → −12.4`, Age III's five-symbol late bin
`−17.6 → −13.0`). Kept as an option with the measurement written down,
because it is the more correct model and because the next round should not
have to rebuild it to find that out. It applies to the **gate only**, not to
`pair_threat`'s `terms::second_copy_obtainable`, deliberately: they are
the same question, and moving both would have made this a measurement of two
things.

## What is left, in the order a tenth round should take it

**1. The dead-race science city is over-credited by ten to fifteen victory
points and the ladder is not where it lives.** The `δ` split above is
unambiguous: with the rung already gated to zero, four symbols and a dead
race still reads `−11.0` in Age III. What is over-priced is the *printed
points* of the green cards, or rather the opportunity cost of the picks that
bought them, and `EvalWeights::vp_projection` is a single scalar over all
card colours. A colour-aware or race-aware projection is a real term this
evaluation does not have.

**2. Wonder option value.** The flagged position's actual defect: nothing
represents "build this wonder later, when the production makes it cheap".
It is not a re-weighting — it needs a second candidate that does not exist
in the action list. The cheapest honest approximation is to charge a wonder
build the coins it pays *above* what the projected pool would let it pay
later, which is `terms::development_by_resource` read from the other end.

**3. The pre-action/post-action comparability defect** in
`duels-server`'s analysis overlay, described above. Out of this crate's
scope and worth fixing where it lives, because it makes the evaluation look
wrong to whoever is flagging positions.

**4. Who starts the age, measured properly.** The signal is consistent and
the corpus cannot de-confound it. `duels-arena/examples/age_start_lab.rs`
can.

**5. Denial asymmetry for science cards** was in the brief and is
untouched. The claim — a green card is worth more to the player who already
holds its symbol than to the one who does not, so denying it is
asymmetrically valuable — is real and is *partly* priced, through
`menu::MenuTables` differencing two per-player take values. Whether the
remaining asymmetry is worth a term is unmeasured, and the instrument for it
would be a `menu`-level bucket table rather than either of the two here.

**6. The one experiment that could flip the wonder verdict, and was not
run.** `EvalWeights::value_scale` is the knob round seven built for
exactly this shape of problem: it divides `mcts-eval`'s leaf temperature,
and round eight's closing note says a round that moves this crate's output
scale a long way should re-fit the calibration *and* re-run the `eval=vN`
A/B. The rationed model at `wonder_potential = 1.25` does move the output
scale. Round nine did not chase it, and the reason is an argument with a
measurement behind it rather than fatigue — but it is an argument, and a
tenth round should check it rather than believe it. Three things say a
global rescale cannot recover thirty Elo here. A *global* rescale cannot
change the leaf's **relative** term balance at all, and the victory kinds
say the harm is relative (civilian wins 1451 → 1112) rather than a loss of
resolution. The leaf is not saturating: the rationed wonder term
differenced reaches about ±20 victory points against an Age I temperature
of 26.4, which is `σ = 0.68` — nowhere near flat. And round seven's own
`value_scale` sweep **bounds** what the knob is worth: `k = 2.0` read
+84.7 and `k = 0.6` read +74.4 against an anchor where `k = 1.0` read
+94.8, so a global rescale in either direction moved that measurement by
ten to twenty Elo, not thirty-five.

Two further gaps in the round's own protocol, stated rather than glossed.
There is **no `TimeMs` column** for the wonder verdict: this project's
two-budget rule attaches to "whatever you recommend as a new default", and
round nine recommends none, so the reject rests on two disjoint 3200-game
`Nodes(2000)` ranges plus the mechanism. And the `alphabeta` transfer check
is **one seed range**, where the `mcts-uct` one is two; it was the run that
looked least likely to matter and it turned out to be the one that
disagrees with the other opponent, so a tenth round revisiting this should
start by giving it a second range.

## Cost

`examples/eval_bench.rs`, every configuration timed on the same 2146
positions. The **default path is bit-identical** to round eight's, so the
only cost worth reporting is what each new option costs when it is switched
on:

```text
                                        Root::new    evaluate       sum
v7 (the round-seven evaluation)          3.156 us    0.421 us   3.577 us
default (unchanged from round eight)     3.185 us    0.424 us   3.610 us
default + the rationed wonder model      3.176 us    0.422 us   3.598 us
default + the structural reach model     3.167 us    0.429 us   3.597 us
```

Both are free — the whole table spans 0.03 us, which is under this
benchmark's run-to-run noise, and the rationed row reads nominally *faster*
than the default it is a superset of. The structural reach model is the one
that had to be checked: round seven's dead-race walk was a **+47%**
regression on `evaluate` before its two guards went in, and this adds a
pass over the occupied slots *inside* that walk. It survives because both
guards still apply — the walk is skipped entirely when the ladder rung is
zero, and it stops at the second unreachable symbol — and because
`terms::faceup_symbols` is one pass for all seven symbols rather than one
per symbol.

## Correction: `p_build` was frozen at the root, and is not any more

`WonderModel::Rationed` shipped reading `p_build` **once from the root
position** and reusing it for every state scored against that
`Root`. The project owner ruled that a defect rather than a tradeoff, and
the rule it violates is worth stating in full because it constrains every
future term this crate adds:

> A leaf evaluation is always computed from the exact state being
> evaluated. A `Root` may cache a *price* — what a shield, a coin or a
> produced resource is worth in this game — because that is a property of
> the game and is what makes one shared `Root` cheap. It may not cache a
> *quantity about the position*, because its consumers hand
> `evaluate` states that are not the root.

`p_build` is the second kind: it is `cap_share × turn_factor`, built from
the wonder slots and the decisions left, both of which change on every
move. Freezing it bit hardest in `mcts-eval`, which builds one `Root` per
search tree — a leaf ten plies deep was priced by the root turn's slot and
decision counts. It also never held even at one ply:
`expected_value` scores *post-action* states against a `Root` read
*pre-action*, so `phased` was mis-scoring by one move throughout.

`terms::wonder_potential_rationed` now derives `p_build` from the state it
is scoring and no longer accepts it as an argument, so the mistake is
unrepresentable rather than merely fixed. `Root::wonder_p_build` is gone.

What this costs and what it invalidates:

* **Nothing on the default path.** `WonderModel::Flat` is the default and
  never read the cached value.
  `duels-agent-phased`'s `tests/p_build_identity.rs` is the proof, and it
  is a *measured* one rather than a reading of the `match` arms: twelve
  whole seeded self-play games, every decision hashed, against the digest
  the pre-fix tree produced. Its second half pins that the same harness
  under `WonderModel::Rationed` does **not** reproduce its pre-fix
  digest, so the fix is not vacuous.
  `round_nine_identity::the_rationed_term_reads_p_build_from_the_state_being_scored`
  is the term-level half: over real games it finds several hundred
  candidate moves that move `p_build`, and a worst-case stale read of
  several victory points.
* **Every `Rationed` Elo figure in this section predates the fix.** The
  `mcts-eval` verdict measurement was re-run at the same scale and is
  below; the `phased` sweep and the `mcts-uct` / `alphabeta` transfer
  checks were not, and their numbers describe the frozen-`p_build`
  function.

### The verdict measurement, re-run

`mcts-eval` (live, rationed at `wonder_potential = 1.25`) against
`mcts-eval:eval=v8` (pinned, flat at `0.5`), same binary either side,
`Nodes(2000)`, 3200 paired seat-swapped games per range — the identical
protocol, run twice per range: once against the pre-fix tree as a matched
control, once against the fix.

```text
             pre-fix (control)            fixed                  worth
 seed 1      -24.4 [-36.4, -12.3]   -19.7 [-31.7,  -7.6]         +4.7
 seed 5001   -34.6 [-46.7, -22.5]    -9.6 [-21.6,  +2.5]        +25.0
 pooled      -29.5 [-38.0, -21.0]   -14.6 [-23.1,  -6.1]   +14.9 +/-12.1
```

**The controls reproduce this section's published figures to the decimal**
(`-24.4` and `-34.6`), which is what makes the comparison a measurement of
the fix rather than of two differently-built harnesses.

So the fix is worth about **+15 Elo** to the rationed model as an
`mcts-eval` leaf, pooled over 6400 games, and the improvement itself clears
zero. **It does not overturn the round-nine verdict.** The fixed model is
still negative pooled, one range's interval still excludes zero, and the
other only reaches parity — so `Rationed` stays off by default, on the same
rule and now on better evidence. Two things worth carrying forward: the
effect is much larger on one range than the other, which is what a
single-range read of either number would have missed; and the victory kinds
move the way the mechanism predicts, with the rationed side's *military*
wins rising `239 -> 287` on the seed-5001 range as the leaf stops pricing
deep positions by the root turn's decision budget.

The `phased` numbers were **not** re-measured. `Rationed`'s value to
`phased` was `+30` and its cost to `mcts-eval` was `-30`; the fix moved the
second by `+15`, so the direction of that split is unlikely to have
reversed, but "unlikely" is not a measurement and the sweep is the obvious
thing for a tenth round to redo before quoting `+29.7 / +31.1 / ...` again.

### Cost: none

Deriving `p_build` per `evaluate` instead of once per `Root` is what this
fix trades, and at search volumes that is 2000 extra reads per tree.
`examples/eval_bench.rs` over 2146 positions puts `evaluate` at
**0.423 us** with the rationed model against **0.418 us** for the default —
about `+1.2%`, inside the benchmark's noise — and `Root::new` at
**3.095 us** against **3.136 us**, nominally *faster*, since the fix
removes two reads from it. Against one full `mcts-eval` simulation at
roughly 18.8 us the added work is under a tenth of a percent, and the four
matches above bear that out: `70.8-71.0` moves per game and wall times per
game (`1579 / 1704` fixed against `1628 / 1522` control) that overlap in
both directions. There is no throughput regression to report.
* The argument originally given for freezing — that a candidate building a
  wonder is otherwise credited twice, once for the wonder leaving the
  unbuilt set and again for `p_build` rising on what is left — is a real
  effect and is *not* what freezing fixed. Freezing removed the whole
  `p_build` response to a move, right and wrong parts together, and paid
  for it by mis-scoring every position that was not the root. A term that
  knows the difference is open work for a later round.
* **Two other root-fixed readers of `p_build` remain, and neither is on the
  default path.** `terms::WonderBudget::of` caches it for
  `WonderModel::Budget`, alongside per-effect prices that genuinely
  cannot be rebuilt per leaf at search volumes;
  `terms::GuildTable::of` caches it to project how many wonders the
  Builders Guild ends up counting. Both carry the same defect. Both are
  built only when something switches them on — `Budget` needs
  `wonder_model = Budget` or `menu_floor = DiscardAndWonder`, `GuildTable`
  needs `guild_pricing = Projected` or a non-zero
  `EvalWeights::guild_projection` — and the default sets none of those,
  so after this fix **no default-path code reads a stale `p_build` at
  all**. They were deliberately left alone: fixing either is a change to a
  different model, wants its own measurement, and belongs in its own PR.
