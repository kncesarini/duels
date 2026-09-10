# Round eleven: the science weight was frozen at the root, and unfreezing it is worth nothing

## The brief, and the honest answer to it

Round nine's closing correction laid down a rule — *a `Root` may cache a
price, never a quantity about the position* — and round eleven was asked to
apply it to the next place it is violated. The candidate was easy to name.
`TermWeights::science`, the multiplier on the science-ladder term, is

```text
w_sci   = 1 + (boost_sci − 1) · S(c_sci)
c_sci   = M_sci^alpha_m · (distinct / 6)^beta_prog
```

and `distinct` in it is the **root's** distinct-symbol count, baked in when
the `Root` was built. `terms::science_ladder` reads the leaf's count for the
*rung*, correctly; the multiplier on that rung still says what the race
mattered at the root. A leaf ten plies deeper, where a player has gone from
two symbols to five, is judged by the importance the science term had at
two. `mcts-eval` builds one `Root` per search tree, so this is its normal
operating condition.

**The answer is that this is not the same defect class as `p_build`, and
the difference decides the shape of the change.** `p_build` was cached by
accident, inside a term's *value*; nothing argued for it and nothing tested
it. The commitment weights are root-fixed **on purpose**, in the *weight*
layer: `blend`'s "Root-fixing" section argues for it and
`tests::a_committing_move_is_scored_under_the_root_weights_not_its_own`
pins it. At one ply, a weight that moved with the candidate action would
credit a committing move twice — once through the term's contents, which
should move, and again through the multiplier on them, which should not.
That argument is sound at one ply and stale at ten. Both readings are right
somewhere, so the change is an opt-in `ScienceProgress` and the default
did not move.

## Round ten landed first, and this section is measured against it

Round eleven's arena work was begun on a branch cut before round ten's
`MenuWeights::lambda` fit merged, and the numbers below were then re-taken
on the merged tree so that both arms of every A/B sit on the shipping
default. Neither arm was ever the old default: an A/B here pins one
configuration on both sides and moves exactly one field, so the question
round ten's landing raises is only "measured against which evaluation",
and the answer is round ten's.

## Why only half of `c_sci` can be re-read, and what that costs

Rebuilding a whole `Root` per leaf is what the cost table rules out — one
`Root::new` is about six and a half `evaluate`s. But `c_sci` factorises,
and only one factor is expensive:

```text
c_sci = M_sci^alpha_m · (distinct / 6)^beta_prog
        \___________/   \___________________/
         root-fixed        re-read per leaf
```

`Commitment::m_sci_alpha` keeps the first factor so
`TermWeights::science_at` can substitute the second for the price of one
`distinct_science()` and two `powf`s.

**This is an approximation and the direction of its error is known.**
`M_sci` is `duels_strategy::ScienceRead::magnitude` — the probability of
completing six symbols — which is itself a function of how many are held
(`missing = 6 − distinct` is one of its inputs). So it is *not* a price in
the sense the round-nine rule allows a `Root` to cache, and "the root's
magnitude with the leaf's progress" is not a clean factorisation. `M_sci`
rises with `distinct`, so a leaf that has advanced the race reads as
somewhat *less* committed than a freshly-built `Root` would say, and never
as more. The correction is real but partial, and conservative. Saying so is
part of the result: an architect's sketch of this fix described `M_sci` as
"achievability at the root", which the magnitude model does not support.

## Elo: neutral, on two disjoint ranges, for both consumers

`duels-arena experiment`, paired seeds and swapped seats, 1600 games per
cell over two disjoint ranges. Both arms of each A/B pin the same
`duels-eval` configuration and differ in exactly the one field (see
`duels_arena::agent_spec`'s `sciprog` key, and `mcts-eval`'s
`eval_override_none_is_bit_identical_to_pinning_todays_live_default` for
why pinning today's default is behaviourally free).

```text
                                      seeds 1..801   seeds 10000..10800   pooled (3200)
mcts-eval:sciprog=leaf  vs  =root       +3.0           −0.4               +1.3 [−10.7, +13.3]
phased:sciprog=leaf     vs  phased      −2.6           −2.2               −2.4 [−14.4,  +9.6]
```

`mcts-eval` at `Nodes(2000)`, `phased` at `Nodes(1)`. Both pooled rows are
SPRT `AcceptH0` against `H1 = 20` Elo, and `duels-arena`'s verdict for both
experiments is **Reject**. So `ScienceProgress::Root` stays the default.

The same two experiments were also run on the pre-round-ten tree, before
the merge described above, and read `+2.2 [−9.9, +14.2]` and
`−0.4 [−12.5, +11.6]`. Both agree with the merged-tree figures to well
inside one interval, which is worth recording: the verdict does not depend
on which of the two evaluations it was taken against.

No `TimeMs` column, deliberately. This round recommends no new default, and
the run was made on a machine at load average 50-110 from unrelated
concurrent work, where `duels_arena`'s own crate docs say a wall-clock cell
is not worth trusting. A `Nodes` cell is load-independent, which is why
these numbers are reported and a `TimeMs` cell was not attempted.

## The victory kinds, which is where the mechanism was supposed to show

Aggregate Elo can hide a mechanism when the mechanism only decides 2-4% of
games, so the science-race conversion rate is the number this round was
actually predicting. It moved in the predicted direction and not by enough
to matter:

```text
                                   military   science   civilian   tiebreak
mcts-eval:sciprog=leaf                 250        43       1297        16
mcts-eval:sciprog=root (control)       251        29       1285        29
   science race exposed in 151 of 3200 games (5%), military in 1534 (48%)

phased:sciprog=leaf                     97        19       1434        39
phased (control)                       105        19       1448        39
   science race exposed in 79 of 3200 games (2%)
```

For `mcts-eval` this is the cleanest reading of the mechanism the round
produced. Science wins go 29 → 43 out of the 72 the two arms share — about
1.6 standard deviations under a 50/50 null — with **military wins flat**
(251 → 250), so the extra science wins are not bought out of the other
race. What they are bought out of is the tiebreak column (29 → 16): games
that used to end level on points now end with the science player having
committed. That is exactly the behaviour change the option was designed to
produce, and it is still not worth measurable Elo, because a science
victory decides 2-5% of games and fourteen extra ones across 3,200 is
inside the noise of everything else.

The pre-round-ten run said the same thing less cleanly — science 31 → 40
there, with military moving 230 → 265 as well. The two runs agree on the
sign and the order of magnitude of the science column and disagree about
the military one, which is the expected behaviour of a ~1.5 sd effect
measured twice.

For `phased` there is no science effect at all (19 wins each way), a small
negative in Elo consistent on both ranges, and
`duels-agent-phased`'s `tests/science_progress_identity.rs` says why with a
stronger instrument than an arena run: the option **does** move candidate
scores at one ply (`expected_value` scores a post-action state against a
pre-action `Root`, so even here the root's count is a move stale), but
`S` is a quartic Hill curve with its midpoint near `0.32` and one symbol out
of six moves `c_sci` by a fraction of that. The measured worst case is a few
percent of one term, and across twelve whole self-play games it never once
flips a 1-ply argmax — the decision hash is bit-identical to the pre-fix
tree's. So the double-counting risk the root-fixing argument warns about is
real in principle and, at these weights, too small to observe cleanly: the
`−2.4` pooled figure has the sign the argument predicts and an interval
that contains zero, which is as much as 3,200 games can say about it.

## What this says, and what a later round could still try

The result is a documented negative, kept as an off-by-default option
rather than reverted, on the same terms as round nine's two candidates.
Three readings of it are worth carrying forward.

**1. Staleness at depth is not automatically worth Elo.** Round nine's
`p_build` fix recovered real strength, and it was tempting to read that as
"every root-frozen quantity is costing something". This one is frozen in
the same way and costs nothing measurable. The difference is plausibly
*what the frozen quantity multiplies*: `p_build` scaled a whole term's
value, while `w_sci` is a multiplier that the blend's own shape keeps
within `[1.0, 1.5]` and, in most real positions, within a few percent of
`1.0`. A quantity the shape flattens cannot be worth much however stale it
is.

**2. If the science weight is to matter, the shape is the thing to change,
not where it is read.** `beta_prog = 2` and `hill_n = 4` together mean four
of six symbols reads as barely committed. Whether that is right is a
separate, sweepable question (`Blend::beta_prog`, `Blend::c0`) and this
round deliberately did not confound it with the staleness question.

**3. The other root-fixed weights are untouched and probably not worth
chasing individually.** `TermWeights::military`, `vp`, `liquidity`,
`development`, `economy` and `race_liquidity` are all still read at the
root, and `c_mil` does not factorise the way `c_sci` does — `prog_mil`
needs `MilitaryRead::need`, not a count off the state. On this round's
evidence the honest next step for the blend is the sweep in (2), not six
more unfreezings.
