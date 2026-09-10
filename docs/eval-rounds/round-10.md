# Round ten: one weight, fitted, and the only one everything agreed about

Round ten changes **one number**: `MenuWeights::lambda`, the weight on
the opponent-menu term, from the `0.6` rounds one through nine all shipped
to `0.408`. No new option, no new model, no shape change anywhere —
`tests/v9_identity.rs` is the shortest identity file this crate has,
because `Config::v9` puts one scalar back and that is the whole delta.

**Read the confirmation sections before quoting this round's Elo — there
are two of them and they disagree by instrument.** On `phased`, where this
crate's weights decide the move outright, the change **reproduces**:
`+11.9` Elo, 95% CI `[+8.0, +15.9]`, over 30,000 games on three fresh
disjoint seed ranges, all three positive. On `mcts-eval`, where the same
evaluation is only half of a search leaf value, it does **not** reproduce
the magnitude that motivated it: the fitting thread measured `+18.1`, and
the confirmation A/B over 4,000 games reads `+5.0` with an interval
containing zero.

**There is no single Elo number for this change.** Quote
`+11.9 [+8.0, +15.9]` for `phased` and `+5.0 [-5.8, +15.7]` for
`mcts-eval`, and say which one is meant. That the two differ by roughly a
factor of two is this round's most interesting result rather than a
discrepancy to be resolved — see "Reading the two confirmations together"
below.

## Where the number came from

Not from a sweep. A separate research thread fitted this crate's eighteen
scalar weights **jointly**, by regression against a search-derived value
corpus: `duels-arena/examples/value_corpus.rs` records, per decision of
`mcts-eval` self-play, the win probability the search itself backed up at
its root, and a weight vector is then fitted to predict that. The
infrastructure for reaching such a vector from a spec string is
`duels-arena`'s eval-scalar keys — `phased:menu_lambda=...` and
`mcts-eval:menu_lambda=...` name the same field, which is what makes one
candidate vector measurable on both consumers.

The fit put `menu.lambda` at `0.407872`. `0.408` is that rounded to three
places, which is far inside the fit's own standard error and is what a
shipped weight should look like.

## Why this coefficient and not the other seventeen

**Because it is the only one where everything agreed.** The round had four
independent things to say about each coefficient, and they disagree freely:

* a **clean** fit target (a corpus held disjoint from anything used to
  check the result),
* a **contaminated** one (the same fit run where the check seeds overlap
  the training seeds — kept deliberately, as the thing a coefficient has to
  survive *both* of),
* `phased`, which is this crate's evaluation used as a **policy**
  (1-ply, the weights decide the move directly),
* `mcts-eval`, which is the same evaluation used as half a search **leaf
  value**.

Round seven already established that these last two do not have to agree —
"a better *predictor* is a worse *leaf*" is that round's headline — and
round nine found a fix worth `+30` Elo to the first and `−15` to the
second. So a coefficient the fit likes is not yet a coefficient to ship.
`menu.lambda` is the one where all four columns came out positive:

```text
                         seed range        Elo
 phased (policy)         range 1        +21.8
 phased (policy)         range 2        +15.3
 phased (policy)         range 3        +12.2
 mcts-eval (leaf)        two ranges     +18.1  [+2.9, +33.3]  (2000 games)
```

Three disjoint seed ranges positive on `phased`, and a pooled `mcts-eval`
interval that excludes zero. That is the bar this project's rules ask for,
and the project owner's decision to move the default rests on it.

Both columns have since been re-measured against `v9()` in a single
binary, and the two rows above turn out to have aged differently: the
`phased` figures hold up (a little lower, `+11.9` against a mean of
`+16.4`, which is what regression to a true effect looks like), the
`mcts-eval` figure does not. Both sections follow.

## The `mcts-eval` confirmation A/B, on current code — and it does not reproduce

**This is the round's honest negative, and it is about the round's own
headline change.** The figures above were taken in the fitting thread. The
`eval=vN` pattern exists so a default move can be checked once more in a
single binary against the exact generation it replaces, so that is what
shipped with it: `mcts-eval` (live, the new default) against
`mcts-eval:eval=v9` (pinned, reproducing the old default bit for bit), at
`Nodes(2000)`, on **four seed ranges disjoint from all three above** and
from each other. The two agents' recorded `params_string`s differ in
exactly one token (`menu=0.41@1.50` against `menu=0.60@1.50`), so this
measures this change and nothing else.

```text
                                     games      Elo        95% CI
 seed 50001                           1000     +8.3    [-13.2, +29.9]
 seed 60001                           1000     +4.2    [-17.4, +25.7]
 seed 70001                           1000     -3.1    [-24.6, +18.4]
 seed 80001                           1000    +10.4    [-11.1, +31.9]
 pooled                               4000     +5.0     [-5.8, +15.7]
```

Three of the four ranges are positive and the pooled point estimate is
positive, so nothing here contradicts the sign. But the interval **contains
zero**, and `duels-arena experiment`'s SPRT against `H1 = +20` returns
`AcceptH0` — a **Reject** verdict on the hypothesis that this is worth
twenty Elo. More pointedly, the pooled interval's upper bound is `+15.7`,
which sits *below* the `+18.1` the fitting thread measured: this run does
not reproduce that figure, it bounds it.

So the fair reading is **"positive, probably small, not established at four
thousand games"** rather than "confirmed". Two candidate explanations, both
untested: the fitting thread's `mcts-eval` measurement was two ranges where
this is four, and a two-range read of the four above could have landed
anywhere from `+6` to `+9`; or the fitted coefficient is worth more to
`phased`, where the weights decide the move outright, than to a search that
only half-listens to them, which is round seven's policy/leaf split showing
up again in miniature.

**The number to carry forward for `mcts-eval` is `+5.0 [-5.8, +15.7]`, not
`+18.1`.** A future round quoting this change at a *search* consumer should
quote that.

This arm has **no `TimeMs` column**, and it is a real gap rather than an
argument: a weight is a multiplication that was already being performed, so
there is no per-decision cost for a wall-clock budget to expose, but this
project's two-budget rule attaches to whatever a round recommends as a new
default and this round recommends one. The machine available was under
heavy concurrent load throughout, which is the one condition under which a
`TimeMs` row is worth less than not having it. (The `phased` arm below has
no such gap, for a structural reason given there.)

## The `phased` confirmation A/B — and it does reproduce

The section above left "the `phased` side, re-measured against `base=v9`"
as this round's second-most-important open item. It has since been run, and
it is the reason the default stays moved.

`phased` (live, the new default) against `phased:base=v9` (pinned,
reproducing round nine's value bit for bit) at `Nodes(1)`, on **three seed
ranges disjoint from the three fitting-thread ranges, from the four
`mcts-eval` confirmation ranges, and from each other**. The two agents'
recorded `params_string`s differ in exactly one of fifty-one tokens
(`menu=0.41@1.50` against `menu=0.60@1.50`) — the same one-token check the
`mcts-eval` arm rests on, so this measures this change and nothing else.

```text
                                     games      Elo        95% CI
 seeds 90001..95001                  10000    +12.8    [+6.0, +19.6]   AcceptH1
 seeds 100001..105001                10000    +14.6    [+7.8, +21.4]   AcceptH1
 seeds 110001..115001                10000     +8.4    [+1.6, +15.2]   Continue
 pooled                              30000    +11.9    [+8.0, +15.9]   AcceptH1
```

Every range is positive, every range's interval **excludes zero**, and
`duels-arena experiment`'s pooled SPRT against `H1 = +20` returns
`AcceptH1` — an **Accept** verdict. Thirty thousand games is affordable
here in a way it is not at `Nodes(2000)`: a 1-ply game costs about eight
milliseconds of one core, and the runner plays a cell's seeds in parallel,
so all three ranges together took twenty seconds of wall-clock against
four minutes of CPU. That is the whole reason this arm can be pinned to
`±4` Elo and the search arm cannot.

There is **no `TimeMs` gap on this arm, structurally**. `phased` takes
`_budget` in `Agent::choose` and never reads it, which
`duels_arena::leaderboard::tests::one_ply_agents_ignore_their_budget`
proves by playing games at two budgets and comparing every decision — so a
wall-clock cell here would replay the `Nodes(1)` games move for move and
report the same Elo. The two-budget rule is satisfied by construction
rather than by a missing row.

The victory kinds say something, unlike the `mcts-eval` arm's. Pooled over
the 30,000 games the candidate's wins break down `851` military / `169`
science / `14135` civilian / `349` tiebreak against the control's
`875 / 117 / 13171 / 310`. **The entire margin is civilian**: `+964`
civilian wins, against `-24` military. Repricing what the opponent can take
next buys civilian-score judgement and buys nothing in the military race —
which is exactly the division of labour the `mcts-eval` blend measurement
found between this evaluation and a playout, showing up here inside the
evaluation alone.

## Reading the two confirmations together

`+11.9 [+8.0, +15.9]` on the policy and `+5.0 [-5.8, +15.7]` on the leaf.
The intervals overlap, so this is not a contradiction and a single true
effect of about `+8` would be consistent with both. But the point
estimates differ by a factor of two in the direction round seven predicted,
and the `phased` interval is narrow enough to be worth taking literally.

The explanation the round offered for the shortfall before the `phased` arm
existed was a guess between two options: too few `mcts-eval` ranges, or the
policy/leaf split. The `phased` arm does not settle which, but it removes
the version of the story in which the coefficient is simply worth less than
the fit claimed — it is worth about what the fit claimed *to a policy*.
**The fair summary is that this is a genuine `phased` improvement which is
mildly positive-to-neutral on `mcts-eval`**, and a search that mixes this
evaluation half-and-half with a playout dilutes a change to it roughly as
much as the mixture weight suggests it should.

That makes the default move well-supported on the consumer the round was
tuning, without overclaiming on the consumer that ships as the strong
agent. It is also the third time this project has measured the same lesson
— round seven's "a better predictor is a worse leaf", round nine's `+30`
to one and `−15` to the other, and now a factor of two — so it is no
longer a surprise and should be the *expected* shape of any future
`duels-eval` result. **Measure both consumers, and report two numbers.**

## What it does to the flagged position, which is a partial answer to round nine's diagnosis

Round nine diagnosed a position the project owner flagged — every legal
move reading as worse than not moving — and found the opponent-menu term to
be the whole of it. `tests/flagged_positions.rs` pins that as a property
rather than as a set of numbers, and round ten moves it: the standing
position against the best action was `0.332` win probability against
`0.270`, a gap of `0.062`, and is now `0.321` against `0.290`, a gap of
`0.032`. **The anomaly is halved and not removed**, which is what a 32% cut
to the term's weight should do to an artefact that term is the whole of.

Worth being careful about what this is and is not. Round ten did not set
out to fix that position — it shipped a fitted coefficient, and the
mitigation is a side effect that the fit knew nothing about. Read the other
way round it is mild independent support for the number: two unrelated
instruments, a regression against search verdicts and a hand-flagged
position, both say the menu term was priced too high. It is *not* evidence
that `0.408` is the right value rather than merely a better one; the gap is
still there at the new weight, and the structural fix round nine named —
the term is read on the post-action state, so a move that hands the turn
over is charged for the opponent's whole menu — is untouched.

The other half of that flag, the wonder-build preference one move earlier,
is **unmoved**: both readings shift by exactly `+2.19` and the `6.70`-point
preference between them survives to the second decimal. The five candidates
there all hand the turn to the same opponent menu, so the term is common to
them and cancels out of the comparison entirely. That is a useful negative
— it says the wonder-timing question round nine left open is genuinely not
a menu-weight question.

## The `mcts-eval` victory kinds, which say nothing much

Pooled over the 4000 `mcts-eval` confirmation games, the candidate's wins
break down
`338` military / `43` science / `1616` civilian against the control's
`307 / 46 / 1586`. Both channels move slightly the candidate's way and
neither moves much — no repeat of the civilian-for-military trade the blend
weight produces.

The original reading of that was "the menu prices what the opponent can
take next, which is not a win condition, so no channel should move". **The
`phased` arm says otherwise** and is worth believing over this one: at
30,000 games the margin there is `+964` civilian and `−24` military, i.e.
entirely civilian. Four thousand games simply cannot resolve a channel
split inside a `+5` effect — with `1616` against `1586` civilian wins, the
difference here is thirty games. Read this paragraph as "no signal", not as
"no effect".

## Cost: none, and structurally so

There is nothing to benchmark. A weight is a multiplication that was
already being performed with a different operand, and the only branches
this field participates in are the three `menu.lambda == 0.0` gates that
skip building the menu tables — which take the same arm at `0.408` as at
`0.6`. That is asserted rather than assumed, by
`v9_identity::the_round_ten_default_does_not_cross_the_lambda_zero_gate`.
The confirmation match bears it out at the only scale where it would show:
`70.70` moves per game and a median `1660` ms per game, both inside round
nine's `1522`-`1704` band on the same hardware.

## What is left

Round nine's list under "What is left, in the order a tenth round should
take it" is **almost entirely untouched**, and this round does not pretend
otherwise: it took a fitted coefficient that had converged, not the next
item on that list. Everything there still stands, and four things join it.

**1. This coefficient on `mcts-eval`, measured properly.** Still the most
important open item, and now the *only* half of it left. `+5.0
[-5.8, +15.7]` over 4000 `Nodes(2000)` games neither establishes nor
refutes the change on a search consumer, and there is no `TimeMs` column on
that arm at all. An eleventh round should give it two more disjoint ranges
and a wall-clock cell on a quiet machine, and should be prepared for the
answer to be "this is worth about five Elo to a search", which the `phased`
arm now makes the *likely* answer rather than a disappointing one. Note the
asymmetry in what that costs: the `phased` arm reached `±4` Elo in twenty
seconds of wall-clock, and the same precision at `Nodes(2000)` is tens of
thousands of games — hours, on a machine that must stay quiet.

**2. Done — and it reproduced.** This item used to read "the `phased`
side, re-measured against `base=v9`", on the reasoning that the fitting
thread's three `phased` figures had never been taken in this binary and
that this was the column where the coefficient looked strongest. It was
run: `+11.9 [+8.0, +15.9]` over 30,000 games on three fresh ranges, all
three positive, SPRT `AcceptH1`. See "The `phased` confirmation A/B" above.
What replaces it as an open question is **why the two consumers differ by a
factor of two** — the policy/leaf split is a name for the observation, not
yet a mechanism, and three rounds have now seen it. A round that could
predict the leaf effect from the policy effect would change how every
future coefficient here gets measured.

**3. The other seventeen coefficients.** The fit produced a whole vector and
this round shipped one component of it. The rest were either
target-dependent (positive on the clean fit and negative on the
contaminated one, or the reverse) or consumer-dependent (`phased` and
`mcts-eval` disagreeing in sign), and each is its own measurement. The
vector itself is worth keeping: `agent_spec`'s
`mcts_eval_shares_phaseds_eval_key_names` records it verbatim as the
eighteen-key spec string, which is both a test of the key names and the
only copy of the fit's output in this repository.

**4. The whole vector, as one candidate.** Shipping coefficients one at a time
is what this project's rules ask for, and it is also the slowest possible
way to spend a joint fit — a joint fit's coefficients are fitted *together*
and are not independently meaningful. Measuring the full eighteen-key vector
as a single candidate against the default, at both budget kinds and on two
disjoint ranges, is a cheap experiment now that `duels-arena experiment`
exists, and it would say whether the fit is worth more than the sum of the
parts anyone dares ship from it.
