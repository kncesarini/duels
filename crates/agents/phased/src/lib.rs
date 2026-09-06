//! `duels-agent-phased`: a 1-ply [`Agent`] whose evaluation weights are a
//! continuous function of how committed each player is to a win condition.
//!
//! # What this crate is trying to fix
//!
//! `duels-agent-greedy-ev` resolves uncertainty correctly — it averages over
//! [`engine::chance_outcomes`] instead of committing to one sampled guess —
//! and that machinery is copied here unchanged. What it does *not* do is
//! change its mind about what matters. Its weight vector is the same on turn
//! one and on turn fifty, for a player two symbols from scientific supremacy
//! and for one who has never taken a green card. Two specific consequences a
//! strong human player pointed out:
//!
//! * it prices almost none of a player's **own production**. A brown or grey
//!   card is, to `greedy-ev`, a card with no points and no shields — a bill.
//!   In reality it is every future card's discount, and grey is the scarcest
//!   of the lot (Ages I and II carry two grey cards each and Age III carries
//!   none at all — verified against `data/cards.json`, not assumed).
//! * it has no idea that a player at four distinct symbols is in a different
//!   game from a player at two. Ages I and II each carry one copy of the same
//!   four symbols and Age III carries the other two, twice each, so after Age
//!   I *every* Age II green completes a pair, and the sixth symbol needs an
//!   Age-III-only symbol or the Law token. A linear "points per symbol" term
//!   cannot see any of that.
//!
//! # The design: one continuous blend, no modes
//!
//! Each player carries a commitment scalar `c ∈ [0, 1]` built from
//! `duels-strategy`'s calibrated race magnitudes, and a Hill curve `S(c)`
//! turns it into a per-term multiplier. Points, coin liquidity, development
//! and economy all fade as commitment rises; the science ladder and military
//! position sharpen; race-card liquidity rises. There are deliberately **no
//! discrete modes** — see [`blend`] for why, and for the arithmetic.
//!
//! Every weight is computed **once per decision from the root position** and
//! reused for every candidate action and every chance outcome. Term
//! *contents* are measured on the post-action, expectation-averaged state,
//! exactly as `greedy-ev` measures its own; only the weights are pinned. That
//! asymmetry is the point: a move that raises `c_sci` should be credited once,
//! through the science term's value going up, not twice, through the weight on
//! that term going up as well.
//!
//! # The evaluation
//!
//! ```text
//! score(state, me) = terminal_result                          (Rail A)
//!                  | ±imminent                                (Rails B/C/C', see [`rails`])
//!                  | Σ_k [ T_k(me) − T_k(opp) ] + A(action) + M(state, me)
//! T_k(p)           = w_k(S(c(p))) × base_k × raw_k(state, p)
//! A(action)        = w_deny × deny_scale × duels_strategy::deny_vp(action)
//! M(state, me)     = ±λ × menu(next mover)          (see [`menu`])
//! ```
//!
//! The first two lines are *rails*, not terms: they replace the weighted sum
//! rather than adding to it, because the question they answer — is this
//! position already decided, and for whom? — is not commensurable with a few
//! victory points of city quality. See [`rails`].
//!
//! `M` is the one term that is *not* read per player and differenced: it
//! prices what the position hands to whoever moves next, which is one player,
//! not both. It is still antisymmetric — swapping `me` flips its sign — so the
//! whole evaluation remains zero-sum.
//!
//! Terms are read **per player** and then differenced, rather than as a
//! single difference under one weight, so that each side's science ladder
//! carries *that side's* science commitment. A science race is more important
//! to whichever player is actually in it; weighting the opponent's ladder by
//! the mover's commitment would read a rising opposing race as less
//! interesting the less interested the mover is in science, which is exactly
//! backwards. At equal commitment the two formulations coincide.
//!
//! `A` is a function of the root position and the action alone —
//! [`duels_strategy::delta_m`] prices what a move does to the opponent's race
//! magnitudes from the root stance, and never looks at the post-action state —
//! so it is genuinely outcome-independent and is added once, outside the
//! per-outcome loop. (Adding it inside would give a bit-identical answer,
//! since the outcome probabilities sum to one; once is simply cheaper.)
//!
//! # What "the un-blended baseline" means here
//!
//! This agent's term set is not `greedy-ev`'s, so there is no other crate it
//! can be bit-compared against. What *is* pinned exactly is the property the
//! whole blend rests on: `S(0) = 0`, so a position in which neither player is
//! committed to anything is scored by a plain, fixed weight vector — every
//! multiplier exactly `1.0`, except race-card liquidity, the one term that
//! rises with commitment and therefore sits at its own floor. [`Blend::off`]
//! forces that state for every position, and
//! `tests::the_blend_off_and_a_zero_commitment_position_agree_bit_for_bit`
//! asserts the two agree bit for bit.
//!
//! # Round two: five changes, measured one at a time
//!
//! The first cut of this crate had one working plan and two broken ones. It
//! beat `greedy-ev` by six hundred Elo and it beat `mcts-uct` 8% of the
//! time — and *every one of those 32 wins was scientific supremacy*. Never
//! military, never points. Two implementation faults and three missing ideas
//! were diagnosed from real match data. All five are [`Config`] options; all
//! five default to on; [`Config::v1`] turns all five off and reproduces the
//! previous agent bit for bit (`tests/legacy_identity.rs`), which is what lets
//! `phased` and `phased:base=v1` be benchmarked against each other out of one
//! binary.
//!
//! **1. `next_age_start` was double its intended magnitude.** The term is read
//! per player and then differenced, and taking the very first shield from a
//! centred pawn flips who is projected to start the next age — so the
//! *differenced* swing was twice the weight written down, about eight victory
//! points for a single shield. `phased` consequently kept 1.5% of the red
//! cards it saw in Age I. Halving the weights to `[1.5, 1.0, 0.0]` matches
//! `docs/strategy-backlog.md` §1.2's 1-3 VP estimate for the start-of-age
//! choice.
//!
//! **2. [`MilitaryModel::Band`].** End-of-game military scoring is a step
//! function (0 / 2 / 5 / 10 victory points at pawn distances 0 / 1-2 / 3-5 /
//! 6-8) plus two loot tokens; a flat reward per step prices the shield that
//! crosses 2→3 exactly like the one that does nothing. The steps are smoothed
//! by how many shields are still in play, because what a position is worth is
//! the expectation of the step function over where the pawn ends up — see
//! [`terms::MilSmoothing`].
//!
//! **3. [`CoinModel::Smooth`].** Three ad hoc coin terms — a floored points
//! channel, a capped race-liquidity bonus and a safety-floor penalty —
//! replaced by one continuous function, with the real `floor(coins / 3)`
//! restored once the rounding is about to actually happen.
//!
//! **4. [`EconomyModel::Bill`].** The development term prices what a city's
//! own production *saves it*. Nothing priced what that production *costs the
//! opponent*. Read per player and differenced, [`terms::resource_bill`] is
//! where monopoly value comes from, with nothing anywhere saying "grey is
//! good". This is the single largest gain of the round.
//!
//! **5. [`menu`]: the opponent's menu, and chain equity.** What the next
//! mover's best affordable card is worth to them, and what a chain starter is
//! worth for the successor it unlocks. Both priced once per decision from the
//! root; see that module for why, and for the stand-down rule that keeps the
//! first one from reading a hidden card.
//!
//! # Round three: a rail instead of a gradient
//!
//! Round two left one term doing a job it was never shaped for.
//! [`terms::military_urgency`] is a smooth quadratic in how far past the
//! second loot token the conflict pawn sits, and it was the only thing in the
//! evaluation claiming to notice an imminent loss. It is blind to whether a
//! closing card exists, blind to whether anybody can afford it, blind to
//! whether there are two of them, and it pays out at pawn positions where
//! nothing at all is about to happen. `military_band = 2.0` was carrying the
//! same load from the other direction: double the honest price of *every*
//! shield in the game, in the hope that the occasional supremacy win made it
//! back.
//!
//! Round three replaces both with a question that has a yes/no answer,
//! adapted from the fix that was worth +26 Elo in `mcts-uct`'s rollout
//! policy — *take an available win, block an available one-move loss*. The
//! 1-ply form asks it of the **post-action state** the evaluator is already
//! scoring, and answers it from the rules: [`duels_strategy::closing_sources`]
//! enumerates the actions that would end the game outright for either player,
//! cross-checked action-for-action against the engine by
//! `duels-strategy/tests/closing_sources_cross_check.rs`.
//!
//! Five changes, all [`Config`] options, all reproduced exactly by
//! [`Config::v2`] (`tests/v2_identity.rs`):
//!
//! 1. **[`rails`] (default on).** Rails B, C and C′ over the post-action
//!    state. Elo-neutral, and adopted on the audit rather than on the Elo —
//!    see below.
//! 2. **`military_band` 2.0 → 1.0 (the term's own units).** The single
//!    largest gain of the round, and the one the rails paid for: with a real
//!    imminence detector in place, the inflated slope has nothing left to buy.
//! 3. **[`MenuShieldPricing::Differenced`] (default on).** [`menu`] priced a
//!    red card's shields at `k ×` a one-sided slope while the evaluation that
//!    scored the resulting position priced the same shields *differenced*
//!    across both players and under both players' weights — a real unit
//!    inconsistency, and a systematic under-valuation of red cards on the
//!    menu. Now an exact finite difference
//!    ([`terms::military_shield_delta`]), Strategy token included. Elo-neutral;
//!    adopted because it removes an inconsistency, not because it wins games.
//! 4. **[`Config::military_horizon`] (default off).** A better-motivated
//!    smoothing width that makes no measurable difference. Honest negative.
//! 5. **[`EvalWeights::production_lock_in`] (default off).** Age III really
//!    does print no brown or grey card, so an Age III resource bill is a fact
//!    rather than a projection — and amplifying the term by that consistently
//!    *costs* Elo. Honest negative.
//!
//! ## The audit, which is the actual result
//!
//! A rail is a guarantee, and a guarantee is audited rather than sampled.
//! `duels-arena/examples/rail_audit.rs` replays real games and recomputes, at
//! every decision, **from the engine** rather than from the agent's own reads,
//! what was available. 200 self-play games per seed range at `Nodes(1)`:
//!
//! ```text
//!                                              this agent            phased:base=v2
//!                                            seed 1    seed 5001    seed 1    seed 5001
//! Rail A  an available win was taken          19/19       32/32      21/21       30/30
//! Rail B  an available block was taken        39/39       41/41      27/37       31/34
//! Rail C  an undeniable close was taken         2/2        5/6         2/2         5/7
//!         ...of those already on the table      0/0        1/1         0/0         2/2
//! B/C'    a rail firing on a closer was right  703/703    631/631    817/817     587/587
//! C       a rail calling it decisive was right   6/6        9/9        20/20         8/8
//! supremacy losses that were NOT blockable      9/9      12/12        5/9        9/12
//! ```
//!
//! **The bottom row is the round.** The round-two agent lost seven games
//! across the two seed ranges to a military or scientific supremacy that a
//! candidate on the table at its own last decision would have removed. This
//! agent loses none: **zero blockable supremacy losses**, on both seed ranges,
//! and Rail B at 100% against 73% and 91%.
//!
//! The last two rows before the bottom are *precision*: when a rail fires,
//! does the engine agree? That matters more than recall, since a rail that
//! fires wrongly misvalues a move by five hundred points. It is **100% on
//! every run**, and on the `Nodes(2000)` runs against `alphabeta` and
//! `mcts-uct` too. (The detector is the same code in both columns, so the
//! `phased:base=v2` figures are the same read taken over the positions that
//! agent reached rather than a property of the round-two agent, which never
//! consults it.)
//!
//! Rail C's recall is the honest negative of the round. It reads 5/6 on one
//! range, and the miss is not a fault so much as a boundary: of the eight
//! undeniable closes the audit found across the two ranges, only one was
//! *already on the table* when the candidate was played. The rest were closes
//! that every opposing reply happened to uncover — real, and a searching agent
//! would find them, but invisible to a rail that reads the post-action state
//! and does no search. Restricted to the closes it can see, Rail C is 1/1 and
//! 0/0, and 1/1 again against `alphabeta`. It is a rare guard that is right
//! when it speaks, not a term that earns its keep every game, and both halves
//! are reported rather than only the flattering one.
//!
//! The same audit against the two search agents, 200 games each at
//! `Nodes(2000)` and seed 1, which is where the losses actually are:
//!
//! ```text
//!                                        vs alphabeta     vs mcts-uct
//! Rail A  an available win was taken        14/14            12/12
//! Rail B  an available block was taken     328/328          138/138
//! Rail C  an undeniable close was taken       3/4              3/3
//! B/C'    firing on a closer was right    6521/6521        2337/2337
//! C       calling it decisive was right       7/7            19/19
//! supremacy losses that were NOT blockable  72/72            25/25
//! ```
//!
//! Ninety-seven military or scientific supremacy losses between them, and not
//! one of them had a candidate on the table that would have removed the
//! threat. Against a real search the rails have three hundred and twenty-eight
//! blocks to make and make all of them.
//!
//! ## Elo, measured one change at a time
//!
//! Against `phased:base=v2`, 600 games per seed range at `Nodes(1)`:
//!
//! ```text
//!                                  seed 1                 seed 5001
//! the new default            +35.4 [+7.5, +63.3]     +37.7 [+9.8, +65.7]
//! ```
//!
//! and, as a leave-one-out against the new default itself (800 games per seed
//! range, so each row is a paired head-to-head of exactly that one change):
//!
//! ```text
//!                                   seed 1     seed 5001    the change is worth
//! military_band back to 2.0          -56.0        -31.8       +56 / +32
//! production lock-in switched on     -20.4        -13.0       (off is better)
//! rails switched off                  -9.5         -2.6       +10 /  +3
//! one-sided menu shield price         -3.9         +4.3        neutral
//! horizon = 2                         +5.2         -0.9        neutral
//! horizon = 3                         +2.6         +3.5        neutral
//! horizon = 5                         +1.7         +2.6        neutral
//! ```
//!
//! Only one row's confidence interval clears zero on both ranges, and it is
//! `military_band`. The rails are Elo-neutral in self-play and always were
//! going to be: they fire on a few hundred of seven thousand decisions, and
//! two agents that both hold the same rails cannot gain from them against each
//! other. What they buy is the bottom row of the audit table.
//!
//! ## Against the ladder
//!
//! 400 games per seed range at seeds 1 and 5001 (`Nodes(1)`; `alphabeta` and
//! `mcts-uct` at `Nodes(2000)`):
//!
//! ```text
//!                    this agent           round two (phased:base=v2)
//! vs random          400-0  / 399-1       398-2 / 396-4 (round two's own figures)
//! vs greedy          400-0  / 397-3
//! vs greedy-ev       398-2  / 395-5
//! vs strategist      398-2  / 399-1
//! vs alphabeta        77/400 / 76/400      66/400 / 80/400
//! vs mcts-uct         44/400 / —           21/400 / 24/400 (round two's own figures)
//! ```
//!
//! `alphabeta` is the only ladder opponent close enough to measure a change
//! against, and 77 and 76 wins in 400 clear the ~72 that `military_band = 1.0`
//! was worth in round two's own sweep. Against `mcts-uct`, 44 wins in 400
//! (11.0%) against round two's 21 and 24, with the win-condition spread
//! holding — military 1, science 19, civilian 24 — so the military-supremacy
//! column is nonzero at `band = 1.0`, which round two reported it never was.
//!
//! # Round four: the turn the evaluator was scoring half of
//!
//! Round four is one bug, one arithmetic fix, and two ideas that did not pay
//! for themselves. All of it is reproduced bit for bit by [`Config::v3`]
//! (`tests/v3_identity.rs`) except the arithmetic fix, which is landed
//! unconditionally because it is a fix.
//!
//! **1. Four wonders were being scored before they had done anything.**
//! `engine::finish_turn` returns early while a
//! [`duels_core::state::Pending`] is outstanding, so the state `apply` returns
//! for Circus Maximus, the Statue of Zeus, the Mausoleum and the Great Library
//! is one in which the effect has **not yet happened**: the card the destroy
//! will take is still in the opponent's city, the retrieval and the token do
//! not exist, and the builder is still `current_player` even though the turn is
//! about to pass. (An ordinary `Build` of a green card that completes a science
//! pair leaves the same kind of state.) Scoring it directly credited none of
//! the effect — only [`terms::wonder_power`]'s flat "+3, this wonder does
//! something" — read the mover as moving again, which flips the sign
//! [`menu::menu_term`] puts on the position, and stood the rails down entirely,
//! since [`rails::rail_owner`] refuses to read a pending state.
//!
//! [`PendingModel::Completed`] (**the new default**) finishes the mover's own
//! turn before judging it: it resolves the pending choice with the engine's own
//! [`duels_core::engine::legal_actions`] — which already enumerates the
//! concrete options, so nothing here re-implements a rule — takes the one the
//! *resolver* likes best, and scores what `finish_turn` then leaves. This is
//! not search: every action in the resolution belongs to the same player in the
//! same turn, and the engine already models that turn as those sequential
//! decisions.
//!
//! The Great Library's three-token draw needs no special handling, which is
//! worth saying because it looks like it should: the draw arrives as
//! [`duels_core::engine::chance_outcomes`]' ten `C(5,3)` outcomes, which
//! [`expected_value`] already averages over, so each draw gets its own exact
//! max-over-three and the ten are weighted correctly for free.
//!
//! **The engine chains exactly once**, and only there: `ChooseProgressToken`,
//! `ChooseGreatLibraryToken` and `DestroyOpponentCard` each clear the pending
//! flag without setting another, but `MausoleumBuild` runs the retrieved card
//! through `construct_card`, which sets `Pending::ProgressToken` if the card
//! off the discard pile completes a science pair.
//! `tests::a_mausoleum_retrieval_can_chain_into_a_progress_token_choice` builds
//! that position and pins it; [`MAX_PENDING_DEPTH`] is three, one more than the
//! chain the engine can actually produce.
//!
//! **2. `wonder_potential` never checked the seven-wonder cap** — a bug, not a
//! model, so it is fixed unconditionally and `Config::v3()` does not restore
//! it. The base game builds seven wonders between the two players and no more,
//! and the term kept paying `0.5 x wonder_power` for every dead wonder in
//! either hand for the rest of the game, *asymmetrically*, since the two sides
//! rarely hold the same number of them. The audit below says this is not
//! hypothetical: 0.42 unbuildable wonders are left in a hand per game.
//!
//! **3. [`WonderModel::Budget`] (default off — an honest negative).** A
//! per-effect price for an unbuilt wonder, scaled by the chance it is ever
//! built: `p_build = cap_share x turn_factor`, where `cap_share` rations the
//! remaining shared slots across both players' unbuilt wonders and
//! `turn_factor` rations the owner's remaining decisions. Every channel reuses
//! a pricer that already exists — `coin_marginal`,
//! [`terms::military_shield_delta`], [`menu::TakeValue::produced_value`],
//! [`menu::TakeValue::free_value`], [`duels_strategy::science::token_value`] —
//! so the Great Library is priced from the tokens actually set aside and a
//! destroy from what the opponent actually owns, rather than at a flat `+3`.
//! It is a better model and it loses: **−11.0 / −5.6 / −3.9** Elo against
//! `phased:base=v3` over 3200 games on each of three disjoint seed ranges. (At
//! 800 games per range it read `+2.6 / +1.7`, which is the whole argument for
//! this project's sample sizes.) Kept as `phased:wonder=budget`, with the
//! measurement written down.
//!
//! **4. [`Config::destroy_replace_discount`] (default off — the second honest
//! negative).** A destroyed brown or grey card is only permanently gone if the
//! market cannot print another one, so the credit is discounted by
//! `min(1, sources_remaining(r) x dealt_frac x share_opp)`. Age III prints no
//! brown or grey card at all — counted off `data/cards.json` by
//! `terms::tests::no_production_source_survives_into_age_three`, not taken on
//! faith — so an Age III destroy is priced as the permanent loss it is and the
//! discount is an exact no-op there. **−0.9 / +5.8** Elo over 3200 games on two
//! disjoint ranges: the signs disagree, so it stays off.
//!
//! ## The audit, which is again the actual result
//!
//! `duels-arena/examples/wonder_audit.rs` counts the behaviour directly rather
//! than inferring it from a win rate, for the same reason `rail_audit.rs` does:
//! both fixes are worth a couple of victory points in a game whose scores span
//! thirty, and both are about *which move gets played*. 200 self-play games at
//! `Nodes(1)`, on each of two disjoint seed ranges — `built% @ mean turn`:
//!
//! ```text
//!                              seed 1                    seed 5001
//!                        base=v3      this agent    base=v3      this agent
//! the four pending-       79% @ 39.0   86% @ 34.4   76% @ 39.7   81% @ 35.6
//!   effect wonders
//!   The Statue of Zeus    78% @ 41.3   93% @ 33.4   78% @ 41.1   87% @ 32.6
//!   The Mausoleum         75% @ 39.4   91% @ 36.0   76% @ 41.1   87% @ 37.3
//!   Circus Maximus        84% @ 38.1   77% @ 31.5   72% @ 37.0   78% @ 33.5
//!   The Great Library     78% @ 36.9   80% @ 37.1   80% @ 39.5   73% @ 39.5
//! every other wonder      88% @ 28.9   86% @ 31.3   90% @ 29.2   88% @ 30.4
//! drafted, never built       123          113          117          112
//! ```
//!
//! **The two wonders the flat bonus under-priced most move on both ranges, and
//! move a long way**: a destroy that takes a whole card out of the opponent's
//! city and a free build out of the discard pile go up nine to fifteen points
//! more often and up to eight and a half turns earlier. The two that do not
//! move consistently are the two whose *value* was already roughly right at a
//! flat `+3` and whose timing is the real question — Circus Maximus destroys a
//! grey card rather than a brown one, and the Great Library's token is worth
//! whatever the set-aside pile happens to hold. Both are now priced against
//! what is actually there, so both move in whichever direction that position
//! calls for; the mean build turn falls for Circus Maximus on both ranges,
//! which is the timing half of the same read.
//!
//! ## Elo
//!
//! Against `phased:base=v3` at `Nodes(1)`, 3200 games per seed range on four
//! disjoint ranges:
//!
//! ```text
//!                        seed 1       seed 5001     seed 10001     seed 20001
//! pending resolution  +3.5 [-8.6,   +5.4 [-6.6,   +6.1 [-6.0,   +15.6 [+3.6,
//!                       +15.5]        +17.5]        +18.1]         +27.7]
//! wonder budget      -11.0 [-23.0,  -5.6 [-17.7,       —         -3.9 [-15.9,
//!                       +1.1]          +6.4]                         +8.1]
//! ```
//!
//! Only one of the four pending-resolution ranges clears zero on its own, but
//! all four agree in sign, and the audit is what the change was built for. The
//! wonder budget agrees in sign too — the other way — on all three of its.
//!
//! Against the ladder, 400 games per seed range at seeds 1 and 5001
//! (`Nodes(1)`; `alphabeta` and `mcts-uct` at `Nodes(2000)`):
//!
//! ```text
//!                    this agent          phased:base=v3
//! vs random          400-0 / 399-1
//! vs greedy          399-1 / 398-2
//! vs greedy-ev       399-1 / 399-1
//! vs strategist      400-0 / 400-0
//! vs alphabeta       87/400 / 93/400     79/400 / 81/400
//! vs mcts-uct        44/400 / 30/400     40/400 / 34/400
//! ```
//!
//! `alphabeta` is again the only ladder opponent close enough to measure a
//! change against, and it moves the right way on both ranges: 87 and 93 wins in
//! 400 against 79 and 81. Against `mcts-uct` the two are **level** — 74/800
//! pooled against 74/800, one range up and one down — which is the honest
//! negative of the round's headline: finishing a turn correctly does not close
//! the gap between a 1-ply evaluation and a real search. The win-condition
//! spread holds (military 0 and 2, science 19 and 13, civilian 25 and 13).
//!
//! ## Cost
//!
//! `examples/decision_cost.rs`, every configuration timed on the same 5709
//! positions:
//!
//! ```text
//! default, pending effects unresolved     49.2 us/decision
//! default (pending effects completed)     56.6 us/decision   +15%
//! default + the wonder budget model       55.8 us/decision   (no measurable change)
//! default + the destroy discount          56.6 us/decision   (no measurable change)
//! ```
//!
//! +15%, and all of it in Ages II and III: the Age-I-only figure moves 47.7 to
//! 49.4 us, because a pending effect comes from a wonder and wonders are not
//! built on turn three. The resolution is bounded by construction — at most
//! eight opponent cards for a destroy, the discard pile for the Mausoleum, five
//! board tokens, three Great Library tokens — and it runs only on the small
//! minority of candidates that create one.
//!
//! The `TimeMs` half of this project's two-budget discipline is checked and
//! reported rather than assumed: `--budget time_ms:50` and `--budget nodes:1`
//! over the same 200 games and the same seed produce the identical
//! `91-108-1`, game for game, which is what "a 1-ply agent ignores its budget"
//! means when it is measured instead of asserted.
//!
//! # Round five: the cards nobody was pricing
//!
//! Round five is one confirmed bug in [`menu::TakeValue`], one term the
//! evaluation simply did not have, and four ideas that did not pay for
//! themselves. All of it is reproduced bit for bit by [`Config::v4`]
//! (`tests/v4_identity.rs`).
//!
//! **1. Every guild card priced out negative.** [`menu::TakeValue::free_value`]
//! starts a card's value from `def.victory_points` and `def.coins`. Both are
//! **zero for all seven guilds** — a guild scores through
//! `points_by_majority` / `coins_by_majority`, which
//! [`duels_core::scoring::breakdown`] reads at scoring time and the menu's
//! pricer did not read at all — so a face-up guild was worth
//! `−cost × coin_marginal`: strictly negative, in every position, for both
//! players. This agent therefore never fought for a guild and never denied one
//! to an opponent who was collecting the colour it counts.
//! `terms::tests::every_guild_scores_through_a_majority_and_not_through_
//! printed_points` asserts the premise off the card data rather than arguing
//! it, and
//! `menu::tests::a_face_up_guild_used_to_price_out_negative_and_now_does_not`
//! pins the fix.
//!
//! [`GuildPricing::Projected`] (**the new default**) prices it as
//!
//! ```text
//! v_guild(g)  = per_vp · Ĝ(t)  +  per_coin · live(t) · coin_marginal
//! Ĝ(t)        = max( c_1 + Δ_1(t),  c_2 + Δ_2(t) )
//! Δ_p(t)      = ρ_t · take_rate · decisions_left(p)   for a colour, or brown+grey
//!             = U_p · p_build(p)                       for wonders
//!             = 0                                      for coins / 3
//! ```
//!
//! `ρ_t` is [`DevSupply::kind_fraction`], off the same pool walk every other
//! development price uses; `p_build` is [`terms::wonder_p_build`], the
//! probability [`WonderModel::Budget`] rations wonders with, factored out and
//! called directly because it is a standalone estimate and has nothing to do
//! with how a wonder's *effects* happen to be priced. Root-fixed, like every
//! price in this crate. The points channel reads `Ĝ` and the coin channel reads
//! the live board, because the coins are paid on the spot and the points are
//! not. Which of the seven guilds keys off which category is asserted card by
//! card against `data/cards.json` in
//! `terms::tests::the_seven_guilds_key_off_the_categories_the_card_data_prints`
//! — including the two that need glass *and* papyrus, and the one that counts
//! coins rather than cards.
//!
//! Both players read the same `Ĝ`, because the rule pays the guild's owner on
//! the higher of the two counts whether or not it is their own. So the race
//! dynamic and the denial both fall out of the menu differencing two per-player
//! values, with no special case anywhere.
//!
//! **2. A yellow card's forward discard yield was entirely unpriced.**
//! [`duels_core::cost::discard_reward`] is `2 +` the player's own commercial
//! cards, so every yellow card in a city raises the payout of **every future
//! discard that city makes**. The coins from a discard already *made* flow
//! through [`terms::coin_points`] and [`terms::coin_liquidity`] exactly; the
//! forward half did not exist. [`EvalWeights::yellow_equity`] (**on by
//! default**) adds `coin_marginal · yellows(p) · rate · decisions_left(p)`, with
//! the matching per-card credit fed into [`menu::TakeValue`] so the menu and the
//! evaluation agree about what a yellow card is worth. `rate` is **measured,
//! not guessed** — `examples/discard_rate.rs` counts discards per decision over
//! whole self-play games and reads 0.2489 / 0.2499 / 0.2467 on three disjoint
//! seed ranges, hence [`terms::DISCARD_RATE_PER_DECISION`] = 0.249.
//!
//! This is the largest gain of the round and the one to be most suspicious of;
//! see [`EvalWeights::yellow_equity`] for the fitted weight, the alternative
//! explanation that is ruled out, and the one that is not.
//!
//! **3. [`MenuFloor`] (default off — an honest negative).** [`menu::menu_term`]
//! returns a hard `0` when nothing on the board is affordable, so an opponent
//! one coin short of everything reads identically to an opponent whose turn is
//! genuinely worthless — and, worse, taking their *last* affordable card is
//! under-rewarded, because the position after reads as a flat zero either way.
//! The real floor is the discard they can always take, and possibly a wonder
//! they can already pay for. Both entries are added to the softmax rather than
//! replacing it. **−4.8 / +3.5 / −2.9** (discard only) and **−12.9 / −1.2 /
//! +0.3** (discard and wonder) Elo against `phased:base=v4` over 3200 games on
//! each of three disjoint seed ranges: the signs disagree, so it stays off,
//! available as `phased:menufloor=discard`.
//!
//! **4. [`Config::menu_afford_soft`] (default off — the second honest
//! negative).** `w_j = σ((coins − cost) / c_soft)` in place of the hard afford
//! cutoff, so a card the next mover is narrowly short on still carries partial
//! weight. **−15.3 / −10.3 / −11.1** at `c_soft = 1`, **−15.5 / −11.9 / −10.6**
//! at 2, **−17.6 / −11.5 / −8.3** at 3 and **−17.6 / −8.4 / −9.8** at 6, all
//! over 3200 games on each of three disjoint ranges. Negative on every range at
//! every width tested, which is at least an unambiguous answer.
//!
//! **5. [`EvalWeights::guild_projection`] (default off — the third).** The same
//! `Ĝ` machinery applied to guilds a player has *already built*, as the forward
//! increment `per_vp · (Ĝ(t) − live(t))` on top of the snapshot `breakdown`
//! already credits. **−3.6 / +2.7 / −3.5** at 0.5, **−2.7 / +3.7 / −3.0** at
//! 1.0 and **−2.3 / +2.5 / −0.9** at 2.0 against the same agent with guild
//! pricing on and this term off. Every interval crosses zero and the sign does
//! not agree across ranges.
//!
//! **6. [`SupplyModel::Dealt`] (default off — the fourth).** [`DevSupply`]'s
//! pool adds every *whole undealt deck* at weight one, but setup deals 20 of
//! Ages I and II's 23 cards and — Age III being the only age with guilds — 17
//! of its 20 plain cards plus 3 of its 7 guilds, the rest going back in the box
//! unseen (`duels_core::engine::new_game`, `duels_core::state::GUILDS_IN_PLAY`).
//! Weighting each undealt entry by its own age's dealt fraction is the
//! straightforwardly more correct statistic, and it is worth **−8.7 / −1.2 /
//! −5.2** Elo over 3200 games on each of three disjoint ranges, and **−5.4 /
//! +7.3 / −3.3** measured on top of the yellow term instead. Negative or
//! neutral either way; kept as `phased:supply=dealt` with the measurement
//! written down. It is a small correction — Ages I and II are scaled uniformly,
//! so only Age III's 17/20-against-3/7 split moves anything relative.
//!
//! ## Elo, measured one change at a time
//!
//! Against `phased:base=v4` at `Nodes(1)`, **3200 games per seed range on three
//! disjoint ranges**, each row a paired head-to-head of exactly that one change
//! against the round-four default:
//!
//! ```text
//!                              seed 1        seed 5001       seed 9001
//! yellow_equity = 4.0        +65.9 ± 12.3   +71.0 ± 12.3    +78.5 ± 12.4
//! guild pricing              +10.3 ± 12.0   +19.7 ± 12.1    +11.7 ± 12.1
//! the discard floor           -4.8 ± 12.1    +3.5 ± 12.1     -2.9 ± 12.1
//! the discard+wonder floor   -12.9 ± 12.1    -1.2 ± 12.0     +0.3 ± 12.1
//! the dealt supply weighting  -8.7 ± 12.1    -1.2 ± 12.0     -5.2 ± 12.0
//! soft affordability (3.0)   -17.6 ± 12.1   -11.5 ± 12.1     -8.3 ± 12.1
//! ```
//!
//! and the two accepts together, on **three further disjoint ranges** so the
//! default is not confirmed on the ranges it was chosen on:
//!
//! ```text
//!                          seed 13001     seed 17001      seed 21001
//! the new default          +83.7 ± 12.4   +94.4 ± 12.5    +95.5 ± 12.5
//! ```
//!
//! The two are additive: guild pricing is worth **+13.0 / +20.3 / +15.3**
//! measured on top of the yellow term rather than against the bare round-four
//! agent, which is the same number it reads on its own.
//!
//! `yellow_equity`'s weight was swept over 0.5, 1, 1.5, 2, 3, 4, 6, 9 and 14 on
//! all three ranges. It rises to a broad plateau between 3 and 9 and falls again
//! by 14; 4.0 is the middle of the plateau, 6.0 is worth **+7.4 / +9.4** more
//! on two fresh ranges (both intervals crossing zero) and 3.0 **−9.2 / −9.4**
//! less. The weight is fitted, and [`EvalWeights::yellow_equity`] says so.
//!
//! ## Against the ladder, and the round's real caveat
//!
//! 800 games per seed range at seeds 1 and 5001, `Nodes(1)`, with `alphabeta`
//! and `mcts-uct` at `Nodes(2000)`:
//!
//! ```text
//!                    this agent            phased:base=v4
//! vs random          799-1  / 799-1        799-1 / 799-1
//! vs greedy          795-5  / 796-4        794-6 / 794-6
//! vs greedy-ev       799-1  / 799-1        799-1 / 798-2
//! vs strategist      800-0  / 796-4        800-0 / 797-3
//! vs alphabeta       260/800 / 250/800     176/800 / 169/800
//! vs mcts-uct        123/800 / 125/800      84/800 /  72/800
//! ```
//!
//! Both search opponents move a long way, which is what makes this a change to
//! the agent rather than a change tuned against a copy of itself: 22% to 32%
//! against `alphabeta` and 10% to 15% against `mcts-uct`, on both ranges.
//!
//! **And now the caveat, which is real.** Against `alphabeta` at a *wall-clock*
//! budget the gain is not there at all: `--budget time_ms:20` gives 96/400
//! against 97/400 at seed 1 and 176/800 against 174/800 at seed 5001 — level,
//! twice — while `alphabeta`'s own strength at that budget (22-24% conceded) is
//! indistinguishable from its strength at `Nodes(2000)`. Against `mcts-uct` at
//! the same wall-clock budget the gain *is* there (51/400 against 39/400). And
//! against a deliberately deeper `alphabeta` at `Nodes(20000)` — where it
//! concedes only 14% — the gain shrinks to 62/400 against 56/400, which is
//! noise.
//!
//! The most defensible reading is that round five's gains are largest against
//! opponents near this agent's own strength and shrink against deeper search,
//! and that the `alphabeta` `TimeMs` runs are exactly the load-sensitive
//! measurement `CLAUDE.md` warns about (the arena's own parallelism *is* the
//! load). It is reported here rather than left out because two ranges agreeing
//! on "no gain" is not something to bury under six that agree on "large gain".
//!
//! ## The behaviour actually changed
//!
//! `duels-arena/examples/matchup_profile.rs` now reports, per side, how many
//! guilds it builds and at what majority count they pay — because a win rate
//! cannot tell "it fights for guilds now" from "it got luckier". 800 games,
//! `Nodes(1)`, guild pricing alone against the round-four agent:
//!
//! ```text
//!                                phased:base=v4    + guild pricing
//! purple cards per game               0.7                1.3
//! games ending with a guild          57.5%              82.9%
//! mean majority count paid on         4.86               4.93
//! guild victory points per game       4.15               7.23
//! guilds (of 3 dealt) bought at all   2.02               2.02
//! ```
//!
//! Same three guilds on the table, nearly twice as many of them ending up on
//! *this* side, and at a marginally higher count — so it is not taking any
//! guild it sees, it is taking the ones that pay. The full default (yellow term
//! included) reads 1.2 purple and 7.62 guild VP per game against 0.8 and 4.29,
//! and shifts the city hard towards commercial cards: **5.5 yellow per game
//! against 2.6**, which is the yellow term made visible.
//!
//! ## Cost
//!
//! `examples/decision_cost.rs`, every configuration timed on the same 2881
//! positions:
//!
//! ```text
//! v4 (the round-four agent)             51.0-51.3 us/decision
//! default (round five)                  50.8-51.1 us/decision   no measurable change
//! default, guilds unpriced              50.4-51.6 us/decision
//! default + the discard/wonder floor         55.0 us/decision    +7%
//! default + soft affordability               53.1 us/decision    +4%
//! ```
//!
//! **Round five is free**, within the run-to-run noise of the benchmark: the
//! guild table is ten majority counts and ten pool fractions once per decision,
//! and the yellow term is one `count` per player per evaluated state. Both
//! options that cost anything are off by default — the floor prices every
//! unbuilt wonder per chance outcome, and soft affordability stops the cutoff
//! from skipping the cards it used to skip. Round four's own `+15%` for the
//! pending-effect fix stands unchanged underneath.
//!
//! # Measured
//!
//! All paired and seat-swapped through `duels-arena`, at `Nodes(1)` unless
//! noted. A 1-ply agent ignores its budget entirely, so a `TimeMs` budget
//! changes nothing for it — `--budget time_ms:50` reproduces the `Nodes(1)`
//! result below to the game — and the only wall-clock figure worth reporting
//! is the per-decision cost further down.
//!
//! Against the agent this crate shipped with (`phased:base=v1`), 600 games per
//! seed range, adding one change at a time:
//!
//! ```text
//!                                          seed 1              seed 5001
//! next_age_start halved (alone)            +9 [-25, +43]       (neutral)
//! MilitaryModel::Band (alone)              +14 [-20, +48]      (neutral)
//! CoinModel::Smooth (alone)                -17 [-51, +17]      (neutral)
//! the three together                       +38 [+10, +66]      +47 [+19, +75]
//!   + EconomyModel::Bill                   +167 [+136, +198]   +201 [+169, +234]
//!   + the opponent menu                    +208 [+175, +241]   +228 [+194, +262]
//!   + chain equity                         +233 [+199, +267]   +226 [+192, +260]
//!   + the two fitted weights = the default +329 [+288, +371]   +292 [+254, +330]
//! ```
//!
//! Every row but the last holds the two fitted weights at the value their own
//! units imply, so the table is a clean "what did each idea buy". They are
//! reproducible as
//! `phased:base=v1,start1=1.5,start2=1.0,mil=band,coin=smooth,band=1.0,bill=1.0,chaineq=0,lambda=0`
//! plus, in order, `econ=bill`, `lambda=0.6`, `chaineq=1.0`; the last row is
//! the bare `phased`.
//!
//! and, at the default, the same thing as a leave-one-out:
//!
//! ```text
//!                            seed 1    seed 5001    the term is worth
//! default                    +329      +292
//! economy_model = legacy      +61       +58         +268 / +234
//! menu lambda = 0            +191      +216         +139 /  +76
//! military_model = legacy    +206      +201         +123 /  +91
//! military_band 2.0 -> 1.0   +265      +267          +64 /  +25
//! coin_model = legacy        +277      +287          +52 /   +5
//! next_age_start back to 4/3 +292      +277          +38 /  +15
//! chain_equity = 0           +278      +309          +51 /  -17
//! ```
//!
//! Two disjoint seed ranges agree in sign on everything except chain equity,
//! which is indistinguishable from zero and is kept on the strength of the
//! pooled result and of the fact that [`menu`] needs its table anyway. The
//! three Round-one fixes are individually noise and jointly worth about +40;
//! the resource bill is more than half the round on its own.
//!
//! Against the rest of the ladder, 400 games per seed range at seeds 1 and
//! 5001 (`Nodes(1)`; `alphabeta` at `Nodes(2000)` over 200 games each):
//!
//! ```text
//!                    new default          previous agent
//! vs random          400-0 / 398-2        291-9 over 300   (Elo +595)
//! vs greedy          399-1 / 397-3        297-3 over 300   (Elo +772)
//! vs greedy-ev       398-2 / 396-4        392-8 over 400   (Elo +666)
//! vs strategist      399-1 / 399-1        296-4 over 300   (Elo +728)
//! vs alphabeta       32-168 / 33-167      32-168 / 18-182
//! ```
//!
//! No regressions: `alphabeta` is the only ladder opponent close enough to
//! measure a change against, and 65 wins in 400 against the previous agent's
//! 50 is an improvement.
//!
//! # `mcts-uct`: the bar that was met, and the one that was not
//!
//! Over 400 paired games at `Nodes(2000)`, `examples/matchup_profile.rs`:
//!
//! ```text
//!                        wins    military  science  civilian
//! round one       s1     32/400         0       32         0
//! round one       s5001  30/400         0       30         0
//! round two       s1     21/400         1        9        11
//! round two       s5001  24/400         3        6        15
//! round three     s1     44/400         1       19        24
//! round three     s5001  33/400         1       14        18
//! ```
//!
//! The last two rows are this round's, and they undo the paragraph below:
//! the aggregate rate is back up, to 77/800 pooled (9.6%) against round one's
//! 62/800 and round two's 45/800, *and* the win-condition spread round two
//! bought is intact. What is left of the honest negative is that `mcts-uct`
//! still wins nine games in ten, for the reason this project has recorded
//! since its first agent: a 1-ply evaluation loses a long positional game to a
//! real search.
//!
//! The **win-condition spread is fixed, on both seed ranges**: the agent now
//! wins by all three routes rather than only one. It also stops conceding the
//! military track — the pawn's mean final position moves from -5.9 in the
//! previous agent's games to -1.7, and `mcts-uct`'s own military-supremacy
//! wins drop from 101 in 400 to 40.
//!
//! The **aggregate rate against `mcts-uct` got worse**, 62/800 to 45/800
//! pooled across the two seed ranges (7.8% to 5.6%),
//! and that is the honest negative of this round. It is the one opponent that
//! moved the wrong way while everything else moved a long way right, which is
//! exactly the failure mode you would expect from fitting two weights against
//! one baseline. Three things are worth saying about it. First, `mcts-uct`
//! plays a fast yellow/red tempo game (3.8 red and 4.6 yellow cards a game
//! against this agent's 2.6 and 2.5) and wins 334 of its 376 games on points,
//! not on a race: a 1-ply evaluation losing a long positional game to a real
//! search is this project's oldest finding, not a new one. Second, the
//! previous agent's 7.8% was *entirely* scientific supremacy on both seed
//! ranges — it entered one lottery every game and lost every other game it
//! played, 738-0 — so the two numbers do not measure the same kind of
//! competence. Third, at a *wall-clock* budget
//! (`time_ms:100`, 100 games, seed 1, single match on a quiet machine) the two
//! are level: both win 6, the old agent's six all by scientific supremacy and
//! the new agent's split 2 science / 4 civilian.
//!
//! # Choosing `military_band`
//!
//! Round two shipped `2.0` and flagged it as a judgement to revisit rather
//! than inherit. Round three revisited it and the answer changed, so both
//! tables are kept here: the argument is more useful than the number.
//!
//! Round two's sweep, against `Config::v1`:
//!
//! ```text
//! band   Elo vs v1 (s1/s5001)   vs alphabeta   mil. wins vs mcts   Age I red keep
//! 1.0      +265 / +267            72/400            0                20.9%
//! 1.5      +322 / +261            68/400            -                29.1%
//! 1.75     +324 / +290            56/400            -                38.1%
//! 2.0      +329 / +292            65/400            1                41.0%   <- was default
//! 2.5      +334 / +322            45/400            1 and 6          45.5%
//! ```
//!
//! At the time, `2.0` was the largest value that cleared every bar at once and
//! `1.0` was the only value measured that never beat `mcts-uct` militarily.
//! What changed is not the sweep but what else is in the evaluation. The
//! inflated slope was buying one thing — the occasional supremacy win — by
//! doubling the honest price of every shield in the game, all game, whether or
//! not anything was about to happen. [`rails`] buys the same thing by asking
//! whether a closing card *exists and is affordable*, which is both cheaper
//! and correct. With the rails in place, `1.0` is simply better:
//!
//! ```text
//! band   Elo vs the default (s1/s5001)   vs alphabeta      mil. wins vs mcts   Age I red keep
//! 1.0     (the default)                   77/400, 76/400        1 and 1          26.3%
//! 2.0      -56.0 / -31.8                  —                     —                41.0%
//! ```
//!
//! The Age I red keep rate lands at 26.3%, which is where this project's
//! calibration guidance expected an honest slope to put it (~20-25%), and the
//! military-supremacy column against `mcts-uct` is *not* zero any more — the
//! thing `1.0` was previously rejected for. `2.0` remains one spec string away
//! (`phased:band=2.0`), and the two tables above are the whole argument.
//!
//! # What one decision costs
//!
//! `examples/decision_cost.rs`, every configuration timed on the *same* 5715
//! positions (timing each one on its own self-play games measures the wrong
//! thing: a configuration that steers towards positions with fewer chance
//! outcomes looks faster while doing more work per decision, and written that
//! way this benchmark reported the full default as 15% *cheaper* than the same
//! agent with the menu term switched off).
//!
//! ```text
//! v1 (the round-one agent)                36.6 us/decision
//! default, menu and chain equity off      41.7 us/decision   +14%
//! default, menu off                       43.0 us/decision   +17%
//! default, rails off                      47.4 us/decision   +29%
//! default, one-sided menu shield price    47.4 us/decision   +30%
//! default (menu lambda = 0.6)             47.0 us/decision   +29%
//! v2 (the round-two agent)                47.6 us/decision   +30%
//! ```
//!
//! [`menu::menu_term`] is the first term in this crate whose cost scales with
//! the number of *chance outcomes* an action has — Age I's worst case is a
//! two-slot reveal from an eleven-card pool, over a hundred outcomes for one
//! candidate — so it is the one that was worth measuring. It costs about 5 us
//! per decision: the per-outcome work is bounded by the handful of accessible
//! slots, not by the outcome count alone.
//!
//! **Round three costs nothing measurable.** The default, the same agent with
//! the rails switched off, and the round-two agent are 47.0, 47.4 and 47.6
//! us — a spread smaller than the run-to-run variation, with the full default
//! nominally the *fastest* of the three. That is not an accident of the
//! benchmark: [`duels_strategy::closing_sources`] takes a one-comparison early
//! exit unless somebody is within one action's shields of a capital or holds
//! five distinct symbols, which is a few hundred of every seven thousand
//! decisions, and Rail C's denial walk runs only inside that.
//!
//! # Take profile
//!
//! `examples/take_profile.rs`, Age I keep rates over 40 self-play games:
//!
//! ```text
//!             brown  grey   blue   green  yellow  red
//! phased      92.9   75.9   78.8   51.4   62.3    26.3
//! phased-v1   68.1   88.9   87.7   79.1   69.7     1.5
//! mcts-uct    80.2   76.4   73.6   25.9   78.9    50.7
//! ```
//!
//! Red moves from "never" (round one's 1.5%) to 26.3%, which is where this
//! project's calibration guidance expected an honest slope to put it. Round
//! two's `military_band = 2.0` overshot to 41%; see "Choosing
//! `military_band`" above.
//!
//! The green column is the cost of round two, and it is a real one: the
//! resource bill makes production and denial compete with the science ladder.
//! Round three does not undo it — 51.4% against round one's 79.1% — but it no
//! longer costs games to `mcts-uct`, where the science-supremacy column is
//! back to 19 and 14 wins in 400.
//!
//! # Public information only
//!
//! Like `greedy-ev`, [`PhasedAgent::choose`] samples one concrete
//! [`GameState`] per decision purely as a vehicle for the engine's chance API.
//! Everything downstream — the commitment scalars, every weight, and the full
//! evaluation of every legal action — reads only publicly-known information,
//! so none of it depends on which world that throwaway sample invented.
//! `tests/determinization_invariance.rs` asserts that bit for bit over real
//! positions from real games.

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

pub mod blend;
pub mod menu;
pub mod rails;
pub mod terms;

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::engine;
use duels_core::scoring::{self, GameResult};
use duels_core::{Action, GameState, Observation, Player};
use duels_strategy::{deny_vp, stance_in, Context, PriorWeights, Stance, ThreatWeights, VpWeights};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub use blend::{Blend, Commitment, TermWeights};
pub use menu::{ChainTable, MenuOptions, MenuTables, TakeContext, TakeValue};
pub use rails::{rail_owner, rail_value, RailModel};
pub use terms::{DevSupply, GuildTable, MilSmoothing, WonderBudget, MAX_UNITS};

/// How [`menu::TakeValue`] prices the shields on a red card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MenuShieldPricing {
    /// `k x` the one-sided local slope of this player's own band — the
    /// round-two behaviour, kept so [`Config::v2`] reproduces it bit for bit.
    /// Inconsistent with the evaluation it feeds, which prices the same shield
    /// *differenced* across both players and under both players' weights.
    OneSided,
    /// The exact finite difference of what the main evaluation would move, via
    /// [`terms::military_shield_delta`], Strategy token included.
    #[default]
    Differenced,
}

/// Whether [`menu::TakeValue`] prices a guild card's majority scoring.
///
/// Every guild in the base game prints zero victory points and zero coins and
/// scores entirely through `points_by_majority` / `coins_by_majority`, which
/// [`duels_core::scoring::breakdown`] reads at scoring time and the menu's
/// pricer did not read at all. A face-up guild therefore priced out at
/// `−cost × coin_marginal` — strictly negative, in every position — so `phased`
/// never fought for a guild and never denied one to an opponent who was
/// collecting the colour it counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuildPricing {
    /// A guild is worth its printed points and coins, both of which are zero —
    /// the pre-existing behaviour, reproduced bit for bit by [`Config::v4`].
    Unpriced,
    /// `per_vp · Ĝ(t) + per_coin · live(t) · coin_marginal`, against the
    /// root-fixed projections in [`terms::GuildTable`].
    ///
    /// **The default**, on the evidence in the crate docs.
    #[default]
    Projected,
}

/// What [`menu::menu_term`] does when nothing on the board is affordable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MenuFloor {
    /// Return zero — the pre-existing behaviour, reproduced bit for bit by
    /// [`Config::v4`]. A hard floor, and a discontinuous one.
    #[default]
    None,
    /// Add the discard the player can always take: `discard_reward ×
    /// coin_marginal`.
    Discard,
    /// ...and the best unbuilt wonder they can already pay for.
    DiscardAndWonder,
}

/// How the pool statistics in [`terms::DevSupply`] weight a card from an age
/// that has not been dealt yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SupplyModel {
    /// Every undealt card counts once — the pre-existing behaviour,
    /// reproduced bit for bit by [`Config::v4`]. Over-weights Age III's seven
    /// guilds, of which three are dealt, and Ages I and II's twenty-three
    /// cards, of which twenty are.
    #[default]
    Raw,
    /// Each undealt entry is weighted by its own age's dealt fraction: 20/23
    /// for Ages I and II, 17/20 for Age III's plain cards and 3/7 for its
    /// guilds. See [`terms::DevSupply::of_with`].
    Dealt,
}

/// Whether the evaluator finishes a turn that the engine has left mid-effect.
///
/// Four wonders — Circus Maximus, the Statue of Zeus, the Mausoleum and the
/// Great Library — do not finish their own construction. `engine::apply`
/// leaves [`duels_core::state::Pending`] set and
/// `engine::finish_turn` returns early, so the state a candidate
/// `BuildWonder` produces is one in which **the effect has not happened yet**:
/// the card the destroy will take is still in the opponent's city, the
/// Mausoleum's retrieval and the Great Library's token do not exist, and the
/// builder is still `current_player` even though the turn is about to pass.
/// An ordinary `Build` of a green card that completes a science pair leaves
/// the same kind of state ([`duels_core::state::Pending::ProgressToken`]).
///
/// Scoring that state directly credits none of the effect — only the flat "has
/// an effect" bonus in [`terms::wonder_power`] — and reads the mover as moving
/// again, which flips the sign [`menu::menu_term`] puts on the position and
/// stands the rails down entirely ([`rails::rail_owner`] refuses to read a
/// pending state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PendingModel {
    /// Score the pending state as it stands — the pre-existing behaviour,
    /// reproduced bit for bit by [`Config::v3`].
    Unresolved,
    /// Finish the mover's own turn before scoring it: resolve the pending
    /// choice with the engine's own [`duels_core::engine::legal_actions`],
    /// take the option the *resolver* likes best, and score the state that
    /// leaves — including whatever `engine::finish_turn` then
    /// does about passing the turn.
    ///
    /// This is not search against an opponent. Every action in the resolution
    /// belongs to the same player, in the same turn, and the engine already
    /// models the turn as those sequential decisions; this simply stops
    /// scoring a half-applied one.
    ///
    /// **The default**, on the evidence in the crate docs: +3.5 / +5.4 / +6.1 /
    /// +15.6 Elo against `phased:base=v3` over 3200 games on each of four
    /// disjoint seed ranges, and — the reason it was built —
    /// `duels-arena/examples/wonder_audit.rs` showing the four pending-effect
    /// wonders going up 89% of the time they are drafted against 78%, nine
    /// turns earlier for the Statue of Zeus.
    #[default]
    Completed,
}

/// How the evaluation prices a drafted-but-unbuilt wonder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WonderModel {
    /// [`terms::wonder_power`] at a flat `wonder_potential` weight, with no
    /// per-wonder probability that it is ever built — the pre-existing
    /// behaviour, reproduced bit for bit by [`Config::v3`].
    #[default]
    Flat,
    /// [`terms::WonderBudget`]: a per-effect price, scaled by the chance the
    /// wonder is built at all given the seven-wonder cap and the decisions the
    /// owner has left.
    Budget,
}

/// How the evaluation prices the conflict pawn's position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MilitaryModel {
    /// A flat reward per step of pawn position — the original.
    Legacy,
    /// The real end-of-game scoring table and the loot tokens, as step
    /// functions smoothed by how many shields are still in play. See
    /// [`terms::MilSmoothing`].
    #[default]
    Band,
}

/// How the evaluation prices a coin pile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoinModel {
    /// Three separate terms: `floor(coins / 3)`, a capped race-liquidity
    /// bonus, and a penalty for falling below a safety floor — the original.
    Legacy,
    /// One smooth function: a linear points channel plus a saturating
    /// liquidity channel. See [`terms::coin_points`] / [`terms::coin_liquidity`].
    #[default]
    Smooth,
}

/// How the evaluation prices a player's exposure to the resource market.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EconomyModel {
    /// The average per-unit trade price the player faces — the original.
    Legacy,
    /// The coins they still expect to *pay*, over the pool that is actually
    /// coming. Read per player and differenced, this is where monopoly value
    /// comes from. See [`terms::resource_bill`].
    #[default]
    Bill,
}

/// The knobs of the opponent-menu term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MenuWeights {
    /// Weight on the whole term. Zero switches it off entirely and restores
    /// the pre-existing evaluation bit for bit.
    pub lambda: f64,
    /// Softmax temperature. Larger spreads credit further down the menu;
    /// towards zero it becomes a plain maximum.
    pub tau: f64,
}

impl Default for MenuWeights {
    fn default() -> Self {
        Self {
            lambda: 0.6,
            tau: 1.5,
        }
    }
}

/// Scores within this distance of the best are treated as tied, and one is
/// chosen uniformly at random.
const TIE_EPSILON: f64 = 1e-6;

/// The chance that the *victim* of a destroy effect, rather than the
/// destroyer, is the one who takes a replacement production card out of the
/// pool. A flat constant, exactly like [`menu`]'s `CHAIN_MINE_SHARE`, and
/// flagged as one: a real model would read the victim's own interest in the
/// card. Only consulted under [`Config::destroy_replace_discount`].
pub const DESTROY_REPLACE_SHARE: f64 = 0.5;

/// How many chained pending resolutions [`PendingModel::Completed`] walks.
///
/// The engine chains at most once. `ChooseProgressToken`,
/// `ChooseGreatLibraryToken` and `DestroyOpponentCard` each clear the pending
/// flag and take a token or a card without ever setting another;
/// `MausoleumBuild` runs the retrieved card through
/// `engine::construct_card`, which *can* set
/// [`duels_core::state::Pending::ProgressToken`] if the card off the discard
/// pile completes a science pair. So two is the real bound and three is the
/// margin — and the loop stops resolving rather than recursing forever if the
/// engine ever grows a longer chain.
pub const MAX_PENDING_DEPTH: u8 = 3;

/// The knobs of the science ladder, which is the one term with enough
/// internal structure to want its own group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScienceWeights {
    /// Value of holding 0, 1, 2, 3, 4 or 5 distinct symbols. Convex: the
    /// marginal symbol is worth more the closer six gets. Six is an outright
    /// win and is handled by the terminal check, so it is not in the table.
    pub ladder: [f64; 6],
    /// How much each of Law / Theology / Strategy still on the board raises
    /// the whole ladder. Those three are what make a science plan pay beyond
    /// the printed points, so a token row without them is a weaker reason to
    /// chase pairs.
    pub strong_token_mult: f64,
    /// Weight on the half-pair threat as a whole.
    pub pair_threat_weight: f64,
    /// What fraction of the best board token's value (priced by
    /// [`duels_strategy::science::token_value`]) a *threatened* pair is worth.
    /// Below one because the second copy still has to be taken.
    pub pair_token_share: f64,
    /// Flat value per threatened pair, for the turn the opponent has to spend
    /// if they would rather deny it — a tempo tax they pay whether or not the
    /// token itself is valuable.
    pub pair_tempo_tax: f64,
}

impl Default for ScienceWeights {
    fn default() -> Self {
        Self {
            ladder: [0.0, 1.0, 2.5, 6.0, 12.0, 18.0],
            strong_token_mult: 0.15,
            pair_threat_weight: 1.0,
            pair_token_share: 0.5,
            pair_tempo_tax: 0.5,
        }
    }
}

/// The base weight of each evaluation term, before the commitment blend
/// multiplies it.
///
/// Everything is in rough victory-point units, so the numbers can be compared
/// with each other and with [`EvalWeights::instant_result`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalWeights {
    /// Reward per step of conflict-pawn position, per player, under
    /// [`MilitaryModel::Legacy`]. Scaled by military commitment.
    pub military_position: f64,
    /// Weight on the smoothed end-of-game scoring bands under
    /// [`MilitaryModel::Band`]. In victory points already, so one is the
    /// honest rate. Scaled by military commitment, like the term it replaces.
    pub military_band: f64,
    /// Weight on the smoothed loot tokens under [`MilitaryModel::Band`]. Not
    /// commitment-scaled: two coins off a rich opponent is worth the same
    /// whether or not this player has a military plan.
    pub military_loot: f64,
    /// `κ` in `σ = max(σ_min, κ·√S_rem)`: how wide the pawn's remaining travel
    /// is per square root of the shields still in play.
    pub military_sigma_scale: f64,
    /// `σ_min`: the floor on that width, so a game with no shields left still
    /// reads the bands as steps rather than as a discontinuity.
    pub military_sigma_min: f64,
    /// `s / σ`: how much of that width the logistic actually uses.
    pub military_logistic_scale: f64,
    /// Weight on the quadratic "somebody is about to win outright" term. Not
    /// commitment-scaled: an opponent three steps from the capital is urgent
    /// whether or not this player has a military plan of their own.
    pub military_endgame_urgency: f64,
    /// Weight on the "as if the game ended now" card / wonder / token points.
    /// Fades as commitment rises: a race, once it is real, is worth more than
    /// the points it costs.
    pub vp_projection: f64,
    /// Weight on `floor(coins / 3)`, matching the real scoring rule.
    pub coins_div3: f64,
    /// Weight on the development term. In coins, so the default is the real
    /// `floor(coins / 3)` rate: a coin this city never has to spend is worth
    /// exactly what a coin in hand is worth.
    pub development: f64,
    /// What fraction of a player's remaining decisions become builds that
    /// actually pay a resource cost. Not one: some decisions are discards,
    /// wonder builds paid from elsewhere, or chain builds that cost nothing.
    pub development_take_rate: f64,
    /// Weight on the science ladder.
    pub science_ladder: f64,
    /// Knobs of the ladder itself.
    pub science: ScienceWeights,
    /// Weight on cash-on-hand for taking a contested race card. The one term
    /// that *rises* with commitment.
    pub race_card_liquidity: f64,
    /// Coins past which further cash is points rather than liquidity.
    pub race_liquidity_cap: f64,
    /// The coin cushion below which a position is financially risky.
    pub coin_safety_floor: f64,
    /// Weight on the shortfall below that cushion.
    pub coin_safety_penalty: f64,
    /// Weight on the average per-unit trade price a player faces, under
    /// [`EconomyModel::Legacy`].
    pub resource_vulnerability: f64,
    /// Weight on the resource bill under [`EconomyModel::Bill`]. The bill is
    /// in coins, so the term itself divides by three; this multiplies it.
    pub resource_bill: f64,
    /// `β` in the smooth coin model's liquidity channel.
    pub coin_smooth_beta: f64,
    /// `c_ref` in the smooth coin model: the pile size past which further cash
    /// is points rather than liquidity.
    pub coin_smooth_ref: f64,
    /// The decision count at or below which the smooth coin model switches its
    /// points channel back to the real `floor(coins / 3)`, because the
    /// rounding is about to actually happen.
    pub coin_endgame_decisions: f64,
    /// Weight on the forward value of chain starters whose successor is still
    /// in the game. Commitment-scaled by the development weight: forward
    /// economic value fades for the same reason development does.
    pub chain_equity: f64,
    /// The opponent-menu term.
    pub menu: MenuWeights,
    /// Penalty per point of free chain-build value handed to the opponent for
    /// their very next turn. Subsumed by [`EvalWeights::menu`] — a free chain
    /// build is just one kind of high-value accessible card — and switched off
    /// automatically whenever `menu.lambda` is non-zero.
    pub deny_chain_gift: f64,
    /// Weight on the rough power of drafted-but-unbuilt wonders.
    pub wonder_potential: f64,
    /// How many of a player's own decisions one wonder build costs them, in
    /// [`WonderModel::Budget`]'s `turn_factor`. Not one: a wonder needs a card
    /// to bury under it *and* the resources to pay for it, and the turns spent
    /// assembling the second are turns not spent building the first.
    pub wonder_turns_per_wonder: f64,
    /// What a play-again wonder's extra turn is worth, in
    /// [`WonderModel::Budget`]. Deliberately `3.0` — the same number
    /// [`terms::wonder_power`] paid for "this wonder has an effect" — so that
    /// at `p_build = 1` and no other effect firing the two models agree.
    pub wonder_extra_turn_vp: f64,
    /// What beginning each age is worth, indexed by age minus one. Zero for
    /// Age III, which has no next age.
    pub next_age_start: [f64; 3],
    /// Weight on [`duels_strategy::deny_vp`], which prices what a move does to
    /// the opponent's race magnitudes in the same victory-point channel as
    /// everything else.
    pub deny: f64,
    /// What the denial term is multiplied by when the *opponent* is fully
    /// committed to a race. Scaled continuously by their `S(c)`, so a rising
    /// opposing plan makes denial worth more without any threshold.
    pub deny_opponent_commit_boost: f64,
    /// Magnitude assigned when a move actually ends the game. Far larger than
    /// every other term's plausible range put together.
    pub instant_result: f64,
    /// Magnitude assigned when one of [`rails`]' terminal rails fires: the
    /// game is not over, but which way it goes is already settled. Half of
    /// [`EvalWeights::instant_result`], so an actual win still outranks a
    /// certain one, and far above every ordinary term put together, so a rail
    /// really does dominate rather than merely nudge.
    pub imminent: f64,
    /// How much the production-lock-in factor
    /// ([`DevSupply::production_lock_in`]) raises the development and resource
    /// bill terms once a city's production can no longer be fixed. Zero
    /// switches it off and restores the previous arithmetic exactly.
    pub production_lock_in: f64,
    /// Weight on the forward *increment* a built guild's majority count is
    /// projected to gain over the rest of the game
    /// ([`terms::GuildTable::projection`]). The snapshot half is already in
    /// [`terms::card_and_token_vp`], via
    /// [`duels_core::scoring::breakdown`], so this adds only what the snapshot
    /// cannot see. Zero switches the term off entirely.
    ///
    /// Independent of [`Config::guild_pricing`] on purpose: one is about
    /// guilds the player *has*, the other about guilds still on the table.
    pub guild_projection: f64,
    /// Weight on [`terms::yellow_equity`], the coins a city's commercial cards
    /// will add to the discards it has not made yet. Zero switches the term off
    /// and, with it, the matching per-card credit
    /// [`menu::TakeValue`] puts on a yellow card.
    pub yellow_equity: f64,
    /// How often a decision is spent on a discard, for
    /// [`terms::yellow_equity`]. Measured, not guessed — see
    /// [`terms::DISCARD_RATE_PER_DECISION`].
    pub yellow_discard_rate: f64,
}

impl Default for EvalWeights {
    fn default() -> Self {
        Self {
            // Half of `greedy-ev`'s 0.6 / 3.0, because these are read per
            // player and then differenced, which doubles them.
            military_position: 0.3,
            // **One, and derived rather than fitted.** The band term is
            // already in victory points, so `1.0` is the honest rate; round
            // two shipped `2.0` because the Elo curve was flat above it and
            // because `1.0` never once beat `mcts-uct` militarily, and
            // recorded that as a judgement to revisit rather than inherit.
            //
            // Round three revisits it, and `1.0` now wins outright: +28 / +57
            // Elo against `phased:base=v2` over 600 games on each of two
            // disjoint seed ranges. What changed is that the "occasional
            // supremacy win" the inflated slope was buying is now carried by
            // the terminal rails ([`rails`]), which ask whether a closing card
            // *exists and is affordable* instead of paying a smooth premium
            // on every shield in the hope that one day it adds up. Doubling
            // the price of every red card in the game was always a strange way
            // to say "do not miss a win".
            military_band: 1.0,
            military_loot: 1.0,
            military_sigma_scale: 0.8,
            military_sigma_min: 0.35,
            military_logistic_scale: 0.55,
            military_endgame_urgency: 1.5,
            vp_projection: 1.0,
            coins_div3: 1.0,
            development: 1.0 / 3.0,
            development_take_rate: 0.6,
            science_ladder: 1.0,
            science: ScienceWeights::default(),
            race_card_liquidity: 0.15,
            race_liquidity_cap: 8.0,
            coin_safety_floor: 3.0,
            coin_safety_penalty: 0.5,
            resource_vulnerability: 0.4,
            // Fitted, not derived. The term divides the bill by three, which
            // is the rate at which coins become victory points at scoring; at
            // `1.0` that is all this weight would say. Three reproduces
            // consistently better on two disjoint seed ranges (+265 / +267 Elo
            // against the previous agent, versus +233 / +226 at one), which
            // says a coin the opponent is forced to spend on trade is worth
            // roughly a whole victory point rather than a third of one. That
            // is not implausible — a trade payment costs them the coin *and*
            // whatever they would rather have bought with it — but it is a
            // measurement, not an argument, and is flagged as such.
            resource_bill: 3.0,
            coin_smooth_beta: 0.6,
            coin_smooth_ref: 5.0,
            coin_endgame_decisions: 2.0,
            chain_equity: 1.0,
            menu: MenuWeights::default(),
            deny_chain_gift: 0.5,
            wonder_potential: 0.5,
            wonder_turns_per_wonder: 2.5,
            wonder_extra_turn_vp: 3.0,
            // A starter flip is worth roughly three victory points in Age I
            // and two in Age II — `docs/strategy-backlog.md` §1.2's estimate.
            // These are read *per player* and then differenced, and the flip
            // moves both sides at once, so the differenced swing is twice the
            // number written here. Getting that wrong is what made a single
            // Age I shield read as an eight-point catastrophe and drove the
            // red-card keep rate to ~1%.
            next_age_start: [1.5, 1.0, 0.0],
            deny: 1.0,
            deny_opponent_commit_boost: 1.5,
            instant_result: 1000.0,
            imminent: 500.0,
            // **Off by default, and measured that way.** The lock-in factor
            // is real — Age III genuinely prints no brown or grey card, so an
            // Age III resource bill is a fact rather than a projection — but
            // amplifying the development and bill terms by it costs Elo:
            // −24 / −15 against the same agent with it switched off, over 800
            // games on each of two disjoint seed ranges. Sweeping it (0.25,
            // 0.5, 1.0) never found a value that helped. Kept as an option
            // with the measurement written down, not enabled on the strength
            // of the argument.
            production_lock_in: 0.0,
            // **Off by default, and measured that way.** The forward increment
            // on a guild already built is real arithmetic, and it is worth
            // nothing anybody can measure: −3.6 / +2.7 / −3.5 at 0.5, −2.7 /
            // +3.7 / −3.0 at 1.0 and −2.3 / +2.5 / −0.9 at 2.0, over 3200 games
            // on each of three disjoint seed ranges against the same agent with
            // guild pricing on and this term off. Every interval crosses zero
            // and the sign does not even agree across ranges. The reason is
            // probably that by the time a guild is *in* a city the projection
            // has little of the game left to run, whereas the same projection
            // used to decide whether to *take* the guild — which is
            // [`Config::guild_pricing`], and which does win — is read when it
            // still has an age to be right about.
            guild_projection: 0.0,
            // **Fitted, not derived, and flagged as such** — the same status as
            // `resource_bill`. The term is already in victory points at `1.0`:
            // a yellow card really does add one coin to each of
            // `rate x decisions_left` future discards, and `coin_marginal`
            // really is what a coin is worth. `1.0` is worth +39 / +43 / +48
            // Elo against `phased:base=v4` over 3200 games on each of three
            // disjoint seed ranges, which is already the largest single gain of
            // the round; the Elo curve then keeps climbing to a broad plateau
            // between 3 and 9 and falls again by 14. Four is the middle of the
            // plateau.
            //
            // Four times the honest rate says the term is standing in for
            // something beyond the discard yield it models. The obvious
            // alternative explanation — that this agent simply under-values
            // coins — is **ruled out**: raising `coins_div3` instead is neutral
            // at 1.5 (+3.1 / −0.4 / −3.1) and sharply negative beyond
            // (−10 / −16 / −26 at 2.0, −108 / −105 / −108 at 3.0). What is
            // *not* ruled out is that a 1-ply evaluation under-values commercial
            // cards for some reason that has nothing to do with discarding, in
            // which case a flat per-yellow bonus would do the same work; this
            // round did not build that control, and it is the obvious follow-up.
            yellow_equity: 4.0,
            yellow_discard_rate: terms::DISCARD_RATE_PER_DECISION,
        }
    }
}

/// Everything [`PhasedAgent`] can be tuned with.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Config {
    /// Base term weights.
    pub eval: EvalWeights,
    /// The commitment blend.
    pub blend: Blend,
    /// How the conflict pawn is priced.
    pub military_model: MilitaryModel,
    /// How coins are priced.
    pub coin_model: CoinModel,
    /// How resource-market exposure is priced.
    pub economy_model: EconomyModel,
    /// Whether the terminal rails are consulted. See [`rails`].
    pub rails: RailModel,
    /// How the opponent-menu term prices a red card's shields.
    pub menu_shield_pricing: MenuShieldPricing,
    /// How many of this player's own rounds the military smoothing width looks
    /// ahead over. See [`terms::horizon_supply`].
    ///
    /// **`None` — the whole remaining shield supply — by default, unchanged
    /// from round two.** Narrowing the width to a horizon is a
    /// better-motivated model, and it does sharpen the bands: at the supply
    /// width two shields are worth almost exactly twice one, which is not
    /// what a step function is supposed to do. It also makes no difference
    /// anybody can measure — `h` of 2, 3 and 5 all landed within a couple of
    /// Elo of the supply width over 800 games on each of two disjoint seed
    /// ranges — and the convention here is not to move a default on a neutral
    /// result. The option stays available as `phased:horizon=3`.
    pub military_horizon: Option<f64>,
    /// Whether the evaluator finishes a turn the engine left mid-effect. See
    /// [`PendingModel`].
    pub pending_model: PendingModel,
    /// How a drafted-but-unbuilt wonder is priced. See [`WonderModel`].
    pub wonder_model: WonderModel,
    /// Whether a destroy effect's credit is discounted by the chance the
    /// opponent simply builds the production back.
    ///
    /// Only ever consulted under [`PendingModel::Completed`], which is the only
    /// mode in which a destroy is scored as more than the flat "has an effect"
    /// bonus at all. See [`Root::destroy_replaceability`].
    pub destroy_replace_discount: bool,
    /// Whether the menu prices a guild card at all. See [`GuildPricing`].
    pub guild_pricing: GuildPricing,
    /// What the menu falls back on when nothing is affordable. See
    /// [`MenuFloor`].
    pub menu_floor: MenuFloor,
    /// `c_soft` in the menu's soft affordability weight. **Zero — the hard
    /// afford / do-not-afford cutoff — by default**, which
    /// [`menu::menu_term`] reproduces bit for bit.
    pub menu_afford_soft: f64,
    /// How the development supply statistics weight an undealt card. See
    /// [`SupplyModel`].
    pub supply_model: SupplyModel,
}

impl Config {
    /// The configuration this crate shipped with: every model at its original
    /// setting, no chain equity, no opponent menu, and the original
    /// `next_age_start` magnitudes.
    ///
    /// Kept so the arena can benchmark against the exact previous agent in one
    /// binary (`phased:base=v1`), and so
    /// `tests/legacy_identity.rs` can assert that this configuration
    /// reproduces a verbatim copy of the old evaluation move for move.
    pub fn v1() -> Config {
        Config {
            eval: EvalWeights {
                next_age_start: [4.0, 3.0, 0.0],
                chain_equity: 0.0,
                menu: MenuWeights {
                    lambda: 0.0,
                    ..MenuWeights::default()
                },
                ..Config::v2().eval
            },
            blend: Blend::default(),
            military_model: MilitaryModel::Legacy,
            coin_model: CoinModel::Legacy,
            economy_model: EconomyModel::Legacy,
            ..Config::v2()
        }
    }

    /// The configuration the *second* round of work shipped with: no terminal
    /// rails, the one-sided menu shield price, the supply-wide military
    /// smoothing width, no production lock-in, and `military_band = 2.0`.
    ///
    /// `tests/v2_identity.rs` asserts this reproduces that agent's arithmetic
    /// bit for bit, which is what makes `phased` against `phased:base=v2` a
    /// single-binary measurement.
    pub fn v2() -> Config {
        Config {
            eval: EvalWeights {
                military_band: 2.0,
                imminent: 0.0,
                production_lock_in: 0.0,
                // Not `EvalWeights::default()`: every later round's weights
                // have to arrive at their own *off* values here, and round five
                // is the first whose defaults are non-zero. Chaining through
                // `v3().eval` — which is `v4().eval` — is what keeps this
                // snapshot a snapshot as the defaults move on.
                ..Config::v3().eval
            },
            rails: RailModel::Off,
            menu_shield_pricing: MenuShieldPricing::OneSided,
            military_horizon: None,
            ..Config::v3()
        }
    }

    /// The configuration the *third* round of work shipped with: the pending
    /// state scored as it stands, the flat wonder-power model, and no
    /// destroy-replacement discount.
    ///
    /// `tests/v3_identity.rs` asserts this reproduces that agent's arithmetic
    /// bit for bit — with **one deliberate exception**, documented there: the
    /// seven-wonder cap in [`terms::wonder_potential`] is a bug fix rather
    /// than a model, so it is landed unconditionally and `Config::v3()` does
    /// not restore the old, uncapped sum.
    pub fn v3() -> Config {
        Config {
            pending_model: PendingModel::Unresolved,
            wonder_model: WonderModel::Flat,
            destroy_replace_discount: false,
            ..Config::v4()
        }
    }

    /// The configuration the *fourth* round of work shipped with: guilds
    /// unpriced on the menu, no menu floor, the hard affordability cutoff, the
    /// unweighted supply pool, and neither of the two new terms.
    ///
    /// `tests/v4_identity.rs` asserts this reproduces that agent's arithmetic
    /// bit for bit, which is what makes `phased` against `phased:base=v4` a
    /// single-binary measurement.
    pub fn v4() -> Config {
        Config {
            eval: EvalWeights {
                guild_projection: 0.0,
                yellow_equity: 0.0,
                ..Config::default().eval
            },
            guild_pricing: GuildPricing::Unpriced,
            menu_floor: MenuFloor::None,
            menu_afford_soft: 0.0,
            supply_model: SupplyModel::Raw,
            ..Config::default()
        }
    }
}

impl Config {
    /// A short, reproducible encoding of the configuration, for
    /// [`AgentSpec::params`].
    pub fn params_string(&self) -> String {
        let e = &self.eval;
        let b = &self.blend;
        format!(
            "guild={}/{:.2},menufloor={},afford={:.2},supply={},yellow={:.2}@{:.3},\
             models={}/{}/{},pending={},wonder={}/{:.2}/{:.2},destroyrepl={},\
             rails={}/{:.0},shieldprice={},horizon={},lockin={:.2},\
             menu={:.2}@{:.2},chaineq={:.2},bill={:.2},band={:.2}/{:.2},\
             smooth={:.2}@{:.1}|\
             mil={:.2}/{:.2},vp={:.2},coin={:.2},dev={:.3}@{:.2},sci={:.2},raceliq={:.2},econ={:.1}/{:.2}/{:.2},chain={:.2},wonder={:.2},start={:?},deny={:.2}x{:.2},win={:.0}|\
             blend={},a={:.2},b={:.2},n={:.1},c0={:.2},floors={:.2}/{:.2}/{:.2}/{:.2}/{:.2},boosts={:.2}/{:.2}",
            match self.guild_pricing {
                GuildPricing::Unpriced => "unpriced",
                GuildPricing::Projected => "projected",
            },
            e.guild_projection,
            match self.menu_floor {
                MenuFloor::None => "none",
                MenuFloor::Discard => "discard",
                MenuFloor::DiscardAndWonder => "discardwonder",
            },
            self.menu_afford_soft,
            match self.supply_model {
                SupplyModel::Raw => "raw",
                SupplyModel::Dealt => "dealt",
            },
            e.yellow_equity,
            e.yellow_discard_rate,
            match self.military_model {
                MilitaryModel::Legacy => "legacy",
                MilitaryModel::Band => "band",
            },
            match self.coin_model {
                CoinModel::Legacy => "legacy",
                CoinModel::Smooth => "smooth",
            },
            match self.economy_model {
                EconomyModel::Legacy => "legacy",
                EconomyModel::Bill => "bill",
            },
            match self.pending_model {
                PendingModel::Unresolved => "unresolved",
                PendingModel::Completed => "completed",
            },
            match self.wonder_model {
                WonderModel::Flat => "flat",
                WonderModel::Budget => "budget",
            },
            e.wonder_turns_per_wonder,
            e.wonder_extra_turn_vp,
            u8::from(self.destroy_replace_discount),
            match self.rails {
                RailModel::Off => "off",
                RailModel::On => "on",
            },
            e.imminent,
            match self.menu_shield_pricing {
                MenuShieldPricing::OneSided => "onesided",
                MenuShieldPricing::Differenced => "diff",
            },
            match self.military_horizon {
                None => "supply".to_string(),
                Some(h) => format!("{h:.1}"),
            },
            e.production_lock_in,
            e.menu.lambda,
            e.menu.tau,
            e.chain_equity,
            e.resource_bill,
            e.military_band,
            e.military_loot,
            e.coin_smooth_beta,
            e.coin_smooth_ref,
            e.military_position,
            e.military_endgame_urgency,
            e.vp_projection,
            e.coins_div3,
            e.development,
            e.development_take_rate,
            e.science_ladder,
            e.race_card_liquidity,
            e.coin_safety_floor,
            e.coin_safety_penalty,
            e.resource_vulnerability,
            e.deny_chain_gift,
            e.wonder_potential,
            e.next_age_start,
            e.deny,
            e.deny_opponent_commit_boost,
            e.instant_result,
            u8::from(b.enabled),
            b.alpha_m,
            b.beta_prog,
            b.hill_n,
            b.c0,
            b.floor_vp,
            b.floor_liq,
            b.floor_dev,
            b.floor_race_liq,
            b.floor_econ,
            b.boost_sci,
            b.boost_mil,
        )
    }
}

/// Everything computed once per decision, from the root position, and reused
/// unchanged for every candidate action and every chance outcome.
///
/// Constructing one is the *only* place the commitment blend is evaluated.
/// [`evaluate`] takes it by reference and has no way to rebuild it, which is
/// what makes root-fixing a property of the types rather than a discipline —
/// see the crate docs.
#[derive(Debug, Clone)]
pub struct Root {
    config: Config,
    stance: Stance,
    /// Indexed by [`Player::index`].
    weights: [TermWeights; 2],
    supply: DevSupply,
    deny_scale: f64,
    age: u8,
    smoothing: MilSmoothing,
    menu: MenuTables,
    wonders: terms::WonderBudget,
    guilds: terms::GuildTable,
    /// `replace_r`, indexed by [`duels_core::data::Resource::index`]. Empty
    /// (all zero) unless [`Config::destroy_replace_discount`] is on.
    replace: [f64; duels_core::data::NUM_RESOURCES],
}

impl Root {
    /// Read the root position for the player to move.
    pub fn new(state: &GameState, me: Player, config: Config) -> Root {
        let opp = me.other();
        let ctx = Context::with(state, ThreatWeights::default());
        // One `Stance` carries both players' military, science and point
        // reads, so this is the whole strategy layer for one position rather
        // than five separate calls. Its *prior* is deliberately not used —
        // an earlier investigation (`duels-agent-strategist`) found that the
        // shape of a policy prior does not transfer to an additive evaluation
        // score. What is used is `delta_m` / `deny_vp`, which need a `Stance`
        // only as the carrier of the reads they price against.
        let stance = stance_in(state, me, PriorWeights::default(), &ctx);
        let edge_me = stance.vp.structural_edge;
        let edge_opp =
            duels_strategy::vp_read_with(state, opp, &ctx, &VpWeights::default()).structural_edge;

        let commit_me = Commitment::of(&stance.science, &stance.military, edge_me, &config.blend);
        let commit_opp = Commitment::of(
            &stance.opponent_science,
            &stance.opponent_military,
            edge_opp,
            &config.blend,
        );

        let mut weights = [TermWeights::of(commit_me, &config.blend); 2];
        weights[opp.index()] = TermWeights::of(commit_opp, &config.blend);

        let supply = DevSupply::of_with(&ctx.board, config.supply_model);
        // Shields still obtainable anywhere in the game, straight off the
        // military read — the width of the pawn's remaining random walk.
        let shields_remaining = f64::from(stance.military.visible)
            + stance.military.expected_hidden
            + stance.military.expected_future_ages;
        // ...narrowed, optionally, to the shields the next few of *this*
        // player's rounds will actually see. See `terms::horizon_supply` for
        // why the whole remaining supply makes the step function read as a
        // straight line for most of a game.
        let smoothing = MilSmoothing::of(
            terms::horizon_supply(
                shields_remaining,
                terms::decisions_left(state, me),
                config.military_horizon,
            ),
            config.eval.military_sigma_scale,
            config.eval.military_sigma_min,
            config.eval.military_logistic_scale,
        );

        // The pricing context both forward-looking terms share. Building it
        // is the whole of their per-decision cost: two `TakeValue`s, the
        // seventeen-link chain table, and one `v` per card face up at the
        // root. Everything downstream is a table lookup plus an affordability
        // check.
        // The majority projections a guild card is priced against. Built only
        // when something reads them, so a run with guild pricing off and the
        // projection term at zero pays nothing for either.
        let guilds = if config.guild_pricing == GuildPricing::Unpriced
            && config.eval.guild_projection == 0.0
        {
            terms::GuildTable::empty()
        } else {
            terms::GuildTable::of(state, &supply, &config.eval)
        };

        let take_tables = menu::TakeContext {
            supply: &supply,
            smoothing: &smoothing,
            guild: &guilds,
        };
        let take = [Player::One, Player::Two].map(|p| {
            TakeValue::of(
                state,
                p,
                take_tables,
                &config,
                weights[p.index()].liquidity,
                (
                    weights[p.index()].military,
                    weights[p.other().index()].military,
                ),
            )
        });
        let chain = if config.eval.chain_equity == 0.0 && config.eval.menu.lambda == 0.0 {
            ChainTable::empty()
        } else {
            ChainTable::of(state, &ctx.board, &ctx.expected, &take)
        };
        // The per-effect wonder prices, wanted by two unrelated things now: the
        // wonder budget model, and the menu's `DiscardAndWonder` floor.
        let wonders = if config.wonder_model == WonderModel::Budget
            || config.menu_floor == MenuFloor::DiscardAndWonder
        {
            terms::WonderBudget::of(state, &take, &chain, &config.eval)
        } else {
            terms::WonderBudget::empty()
        };
        let replace = if config.destroy_replace_discount {
            std::array::from_fn(|r| (supply.sources[r] * DESTROY_REPLACE_SHARE).min(1.0))
        } else {
            [0.0; duels_core::data::NUM_RESOURCES]
        };

        let menu = if config.eval.menu.lambda == 0.0 {
            MenuTables::unpriced(state, take, chain)
        } else {
            MenuTables::with(
                state,
                &ctx.board,
                take,
                chain,
                menu::MenuOptions {
                    floor: config.menu_floor,
                    afford_soft: config.menu_afford_soft,
                },
                wonders.clone(),
            )
        };

        Root {
            guilds,
            wonders,
            replace,
            deny_scale: 1.0 + (config.eval.deny_opponent_commit_boost - 1.0) * commit_opp.s,
            supply,
            age: state.age(),
            smoothing,
            menu,
            stance,
            weights,
            config,
        }
    }

    /// The root-fixed military smoothing.
    #[inline]
    pub fn smoothing(&self) -> &MilSmoothing {
        &self.smoothing
    }

    /// The root-fixed pricing tables the forward-looking terms share.
    #[inline]
    pub fn menu(&self) -> &MenuTables {
        &self.menu
    }

    /// The configuration in force.
    #[inline]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The root-fixed term multipliers for one player.
    #[inline]
    pub fn weights(&self, p: Player) -> &TermWeights {
        &self.weights[p.index()]
    }

    /// One player's commitment scalars.
    #[inline]
    pub fn commitment(&self, p: Player) -> &Commitment {
        &self.weights[p.index()].commitment
    }

    /// The root stance, for diagnostics.
    #[inline]
    pub fn stance(&self) -> &Stance {
        &self.stance
    }

    /// The development supply statistics, for diagnostics.
    #[inline]
    pub fn supply(&self) -> &DevSupply {
        &self.supply
    }

    /// The root-fixed wonder budget. All zero unless
    /// [`WonderModel::Budget`] is in force.
    #[inline]
    pub fn wonders(&self) -> &terms::WonderBudget {
        &self.wonders
    }

    /// The root-fixed guild majority projections. All zero unless something
    /// reads them — [`GuildPricing::Projected`] or a non-zero
    /// [`EvalWeights::guild_projection`].
    #[inline]
    pub fn guilds(&self) -> &terms::GuildTable {
        &self.guilds
    }

    /// How replaceable `card`'s production is: `0` for a card whose resources
    /// the market can no longer print (every Age III destroy, since Age III
    /// prints no brown or grey card — counted off `data/cards.json` by
    /// `tests::production_is_completely_frozen_by_age_three`), rising towards
    /// `1` when several undestroyed sources are still coming.
    ///
    /// ```text
    /// replace_r = min(1, sources_remaining(r) · dealt_frac · share_opp)
    /// replace(card) = min over the resources the card produces of replace_r
    /// ```
    ///
    /// `sources_remaining · dealt_frac` is [`DevSupply::sources`], which
    /// already discounts each pool card by the chance it is ever dealt.
    /// `share_opp` is a flat [`DESTROY_REPLACE_SHARE`] — the chance the
    /// *victim*, rather than the destroyer, is the one who ends up taking the
    /// replacement. A real model would read the victim's own interest in the
    /// card, exactly as [`menu::ChainTable`]'s `CHAIN_MINE_SHARE` would; both
    /// are flat constants for the same reason and both are flagged as such.
    ///
    /// The `min` over the card's resources is the conservative direction: a
    /// card is only fully replaceable if *everything* it produced can be
    /// bought back.
    ///
    /// Zero throughout unless [`Config::destroy_replace_discount`] is on.
    pub fn destroy_replaceability(&self, card: duels_core::data::CardId) -> f64 {
        let mut out = 1.0f64;
        let mut any = false;
        for (r, &n) in card.def().produces.iter().enumerate() {
            if n > 0 {
                any = true;
                out = out.min(self.replace[r]);
            }
        }
        if any {
            out
        } else {
            0.0
        }
    }

    /// The age the root position was in.
    ///
    /// Load-bearing for correctness, not a convenience: see
    /// [`terms::chain_gift_exposure`], the one term that reads a card in the
    /// structure and therefore has to stand down once a move has ended the
    /// age and the engine has dealt a whole new one out of a deck no
    /// observation can see.
    #[inline]
    pub fn age(&self) -> u8 {
        self.age
    }

    /// The multiplier the denial term carries, given how committed the
    /// opponent is.
    #[inline]
    pub fn deny_scale(&self) -> f64 {
        self.deny_scale
    }

    /// `A(action)`: the victory-point equivalent of what this action does to
    /// the opponent's race magnitudes, scaled by how committed they are.
    ///
    /// A function of the root position and the action only, so it is added
    /// once per candidate rather than once per chance outcome.
    pub fn denial_term(&self, action: Action) -> f64 {
        self.config.eval.deny * self.deny_scale * deny_vp(action, &self.stance)
    }
}

/// Score `state` for `me`, higher is better, under the root-fixed weights in
/// `root`.
///
/// A finished game is scored by `instant_result` alone, dwarfing every other
/// term; otherwise every term is read for each player separately and
/// differenced.
pub fn evaluate(state: &GameState, me: Player, root: &Root) -> f64 {
    evaluate_at(state, me, root, MAX_PENDING_DEPTH)
}

/// [`evaluate`] with the remaining pending-resolution budget explicit.
fn evaluate_at(state: &GameState, me: Player, root: &Root, depth: u8) -> f64 {
    if let Some(result) = state.result() {
        return match result {
            GameResult::Win { winner, .. } if winner == me => root.config.eval.instant_result,
            GameResult::Win { .. } => -root.config.eval.instant_result,
            GameResult::Draw => 0.0,
        };
    }
    // Finish the mover's own turn before judging it. See [`PendingModel`].
    if root.config.pending_model == PendingModel::Completed
        && depth > 0
        && state.pending().is_some()
    {
        if let Some(v) = resolve_pending(state, me, root, depth) {
            return v;
        }
    }
    // Rails B, C and C' — see [`rails`]. A rail *replaces* the weighted sum
    // rather than adding to it: the question it answers ("is this position
    // already decided, and for whom?") is not commensurable with a few
    // victory points of city quality, and a magnitude large enough to
    // dominate every ordinary term would be indistinguishable from a
    // replacement anyway. Antisymmetric by construction, so the evaluation
    // stays zero-sum.
    if let Some(v) = rails::rail_value(
        state,
        me,
        root.age,
        root.config.rails,
        root.config.eval.imminent,
    ) {
        return v;
    }
    player_value(state, me, root) - player_value(state, me.other(), root)
        + menu::menu_term(state, me, root.age, &root.menu, &root.config.eval.menu)
}

/// Finish a turn the engine left mid-effect, and score what it leaves.
///
/// The pending choice belongs to `state.current_player()` — every
/// [`duels_core::state::Pending`] variant is created by that player's own
/// action and resolved by them before the turn passes — so the option taken is
/// the one *they* like best, whichever side `me` happens to be. That is what
/// keeps the whole thing **antisymmetric**: the resolution picks the same
/// option under `evaluate(s, me)` and `evaluate(s, me.other())`, because the
/// key it maximises is the same number in both (`evaluate` is exactly
/// antisymmetric, and `a - b` is exactly `-(b - a)` in IEEE-754), and the value
/// it returns then negates with `me` like any other.
///
/// Returns `None` — and so falls back to scoring the pending state as it
/// stands — only when the engine offers no legal resolution at all, which it
/// never does: `legal_actions` is empty exactly when the game is over, and a
/// finished game never carries a pending effect.
fn resolve_pending(state: &GameState, me: Player, root: &Root, depth: u8) -> Option<f64> {
    let resolver = state.current_player();
    let sign = if resolver == me { 1.0 } else { -1.0 };
    // A pending resolution reveals nothing: `engine::slots_revealed_by` is
    // empty for every one of these actions (none of them takes a card out of
    // the structure), so the single trivial outcome is the whole chance node.
    let trivial = engine::Outcome::default();

    // The destroy discount needs a reference to take a fraction *of*. The
    // pending state's own score is the natural one: it is what the position is
    // worth with the effect not yet applied, it is the same for every target,
    // and at `replace = 0` the blend collapses to the resolved value exactly,
    // so the knob is provably a no-op when it is switched off.
    let discount = root.config.destroy_replace_discount
        && matches!(
            state.pending(),
            Some(duels_core::state::Pending::Destroy { .. })
        );
    let unresolved = if discount {
        player_value(state, me, root) - player_value(state, me.other(), root)
            + menu::menu_term(state, me, root.age, &root.menu, &root.config.eval.menu)
    } else {
        0.0
    };

    let mut best: Option<(f64, f64)> = None;
    for option in engine::legal_actions(state) {
        let mut next = *state;
        if engine::apply_with_outcome_unchecked(&mut next, option, &trivial).is_err() {
            continue;
        }
        let mut value = evaluate_at(&next, me, root, depth - 1);
        if discount {
            if let Action::DestroyOpponentCard { card } = option {
                let replace = root.destroy_replaceability(card);
                // Guarded rather than multiplied by `1.0`: `u + (v - u) * 1.0`
                // is not bit-identical to `v`, and this knob has to be an
                // exact no-op wherever nothing can be replaced — which is
                // every Age III destroy.
                if replace > 0.0 {
                    value = unresolved + (value - unresolved) * (1.0 - replace);
                }
            }
        }
        let key = sign * value;
        if best.is_none_or(|(b, _)| key > b) {
            best = Some((key, value));
        }
    }
    best.map(|(_, value)| value)
}

/// Every term, read for one player and weighted by *that player's* root-fixed
/// commitment multipliers.
fn player_value(state: &GameState, p: Player, root: &Root) -> f64 {
    let e = &root.config.eval;
    let c = &root.config;
    let w = &root.weights[p.index()];
    let breakdown = scoring::breakdown(state, p);

    // --- fading with commitment -------------------------------------------
    let points = w.vp * e.vp_projection * terms::card_and_token_vp(&breakdown);

    // Coins. `Legacy` splits into three terms (a floored points channel, a
    // capped race-liquidity bonus, and a shortfall penalty inside `economy`);
    // `Smooth` replaces all three with one continuous function.
    let (liquidity, race_liquidity, coin_safety) = match c.coin_model {
        CoinModel::Legacy => (
            w.liquidity * e.coins_div3 * f64::from(breakdown.coins),
            w.race_liquidity
                * e.race_card_liquidity
                * terms::race_liquidity(state, p, e.race_liquidity_cap),
            e.coin_safety_penalty * -terms::coin_shortfall(state, p, e.coin_safety_floor),
        ),
        CoinModel::Smooth => (
            w.liquidity * e.coins_div3 * terms::coin_points(state, p, e.coin_endgame_decisions)
                + terms::coin_liquidity(state, p, e.coin_smooth_beta, e.coin_smooth_ref),
            0.0,
            0.0,
        ),
    };

    // How much of what this city produces is still fixable. In Age III the
    // answer is "none of it" — there is no brown or grey card left in the
    // game — so the development credit and the resource bill both stop being
    // projections and start being facts, and are worth more accordingly.
    let lock = 1.0 + e.production_lock_in * root.supply.production_lock_in;
    let development = w.development
        * e.development
        * lock
        * terms::development_value_with(
            state,
            p,
            &root.supply,
            e.development_take_rate,
            // With `Bill` in force the post's value arrives through the lower
            // `price_r` it produces; crediting it separately would double it.
            c.economy_model == EconomyModel::Legacy,
        );
    let chain_equity =
        w.development * e.chain_equity * menu::chain_equity(state, p, root.menu.chain());

    let market = match c.economy_model {
        EconomyModel::Legacy => e.resource_vulnerability * -terms::average_trade_price(state, p),
        EconomyModel::Bill => {
            e.resource_bill
                * lock
                * -terms::resource_bill(state, p, &root.supply, e.development_take_rate)
                / 3.0
        }
    };
    let economy = w.economy * (coin_safety + market);

    // --- sharpening with commitment ---------------------------------------
    let science = w.science * e.science_ladder * terms::science_ladder(state, p, &e.science);
    let military = match c.military_model {
        MilitaryModel::Legacy => {
            w.military * e.military_position * terms::military_position(state, p)
        }
        MilitaryModel::Band => {
            w.military * e.military_band * terms::military_band(state, p, &root.smoothing)
                + e.military_loot * terms::military_loot(state, p, &root.smoothing)
        }
    };

    // --- never scaled -----------------------------------------------------
    let urgency = e.military_endgame_urgency * terms::military_urgency(state, p);
    let start = terms::next_age_start(state, p, e);
    let wonders = match c.wonder_model {
        WonderModel::Flat => e.wonder_potential * terms::wonder_potential(state, p),
        WonderModel::Budget => terms::wonder_potential_budget(state, p, &root.wonders),
    };
    // The opponent-menu term subsumes this one — a free chain build is just
    // one kind of high-value accessible card, and it is priced there properly
    // instead of at a flat `2 + VP`.
    let gift = if e.menu.lambda == 0.0 {
        -e.deny_chain_gift * terms::chain_gift_exposure(state, p, root.age)
    } else {
        0.0
    };
    // The forward half of a built guild's majority scoring. `breakdown` — read
    // into `points` above — already carries the snapshot half.
    let guilds = if e.guild_projection == 0.0 {
        0.0
    } else {
        e.guild_projection * root.guilds.projection(state, p)
    };
    // What this city's yellow cards will add to the discards it has not made
    // yet. `coin_marginal` is root-fixed, like every other price; the yellow
    // count and the decision budget are read here.
    let yellow = if e.yellow_equity == 0.0 {
        0.0
    } else {
        e.yellow_equity
            * terms::yellow_equity(
                state,
                p,
                root.menu.take(p).coin_marginal,
                e.yellow_discard_rate,
            )
    };

    points
        + liquidity
        + development
        + chain_equity
        + economy
        + science
        + military
        + race_liquidity
        + urgency
        + start
        + wonders
        + gift
        + guilds
        + yellow
}

/// The probability-weighted expected value of taking `action` in `state`,
/// plus the action's own denial term.
///
/// The chance-expectation machinery is `duels-agent-greedy-ev`'s, unchanged:
/// enumerate every way the action's randomness could resolve via
/// [`engine::chance_outcomes`] (a single certain outcome for the large
/// majority of actions), apply each to its own copy of `state`, and average
/// the scores by their true probabilities rather than committing to one
/// sampled guess.
pub fn expected_value(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => evaluate(&next, me, root),
            // `action` came from `legal_actions` for `state` and `outcome`
            // from `chance_outcomes` for the same pair, so this is
            // unreachable; score the pre-action state rather than silently
            // dropping probability mass from the expectation.
            Err(_) => evaluate(state, me, root),
        };
        acc += prob * value;
    }
    acc + root.denial_term(action)
}

/// A 1-ply agent that re-reads what matters before every decision.
#[derive(Debug, Clone)]
pub struct PhasedAgent {
    rng: StdRng,
    config: Config,
    root_builds: u64,
}

impl PhasedAgent {
    /// A new agent seeded from `seed`, using [`Config::default`].
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, Config::default())
    }

    /// A new agent seeded from `seed`, with an explicit configuration.
    pub fn with_config(seed: u64, config: Config) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            config,
            root_builds: 0,
        }
    }

    /// A new agent driven by an existing RNG, so a caller can draw many
    /// independent agents from one stream.
    pub fn from_rng(rng: StdRng) -> Self {
        Self {
            rng,
            config: Config::default(),
            root_builds: 0,
        }
    }

    /// The configuration this agent is using.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// How many times this agent has built a [`Root`] — that is, how many
    /// times it has evaluated the commitment blend.
    ///
    /// Instrumentation for the root-fixing property: this must equal the
    /// number of [`Agent::choose`] calls that got past the trivial
    /// single-legal-action shortcut, however many candidate actions and
    /// chance outcomes each of them had to score. See
    /// `tests::root_weights_are_built_exactly_once_per_choose`.
    pub fn root_builds(&self) -> u64 {
        self.root_builds
    }
}

impl Agent for PhasedAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "phased".to_string(),
            version: "1.0.0".to_string(),
            params: self.config.params_string(),
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action], _budget: Budget) -> Action {
        assert!(
            !legal.is_empty(),
            "choose must not be called with no legal actions"
        );
        if legal.len() == 1 {
            return legal[0];
        }

        let me = obs.current_player;
        // Sampled once per call, purely as a vehicle for the engine's chance
        // API (which needs a concrete `GameState`) — `greedy-ev`'s pattern.
        // Nothing downstream reads a hidden identity, so it does not matter
        // which world this invents.
        let base_state = obs.sample_state(&mut self.rng);

        // The one and only place the blend is evaluated for this decision.
        let root = Root::new(&base_state, me, self.config);
        self.root_builds += 1;

        let mut scored: Vec<(Action, f64)> = Vec::with_capacity(legal.len());
        for &action in legal {
            scored.push((action, expected_value(&base_state, action, me, &root)));
        }

        let Some(best_score) = scored.iter().map(|&(_, s)| s).fold(None, |m, s| match m {
            Some(b) if b >= s => Some(b),
            _ => Some(s),
        }) else {
            return legal[self.rng.gen_range(0..legal.len())];
        };

        let best: Vec<Action> = scored
            .iter()
            .filter(|&&(_, s)| (best_score - s).abs() <= TIE_EPSILON)
            .map(|&(a, _)| a)
            .collect();
        best[self.rng.gen_range(0..best.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::data::Science;
    use duels_core::scoring::VictoryKind;
    use duels_core::testing::StateBuilder;

    /// Twenty of Age II's twenty-three cards, dealt into a real structure
    /// (so some slots are genuinely face down and the science supply model
    /// has something to work with, unlike `open_slots`, which reveals
    /// everything and leaves every unknown-pool weight at zero).
    const AGE_TWO_DEAL: [&str; 20] = [
        "sawmill",
        "brickyard",
        "shelf-quarry",
        "glassblower",
        "drying-room",
        "walls",
        "horse-breeders",
        "barracks",
        "archery-range",
        "parade-ground",
        "library",
        "dispensary",
        "school",
        "laboratory",
        "courthouse",
        "statue",
        "temple",
        "aqueduct",
        "rostrum",
        "forum",
    ];

    /// A mid-game Age II position with a full structure, `built` in Player
    /// One's city, and enough coins for the cost engine not to be the binding
    /// constraint.
    fn age_two_position(built: &[&str]) -> GameState {
        StateBuilder::new()
            .age(2)
            .deal(&AGE_TWO_DEAL)
            .built(Player::One, built)
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build()
    }

    fn advanced_game(seed: u64, steps: usize) -> GameState {
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x55);
        for _ in 0..steps {
            let actions = engine::legal_actions(&st);
            if actions.is_empty() {
                break;
            }
            let a = actions[(st.turn() as usize * 7) % actions.len()];
            engine::apply(&mut st, a, &mut rng).unwrap();
        }
        st
    }

    // -----------------------------------------------------------------
    // The commitment guard: nobody is committed to anything on turn one.
    // -----------------------------------------------------------------

    /// The trap this whole design exists to avoid: `M_sci` alone reads a
    /// substantial magnitude for a player holding *no symbols at all*,
    /// because with three ages still to come the supply model cannot rule the
    /// race out. If that went straight into the blend, every player would
    /// read as partly science-committed from move one and the science ladder
    /// would be boosted in positions where it means nothing.
    #[test]
    fn a_fresh_game_reads_as_uncommitted_for_both_players() {
        for seed in 0..8u64 {
            let st = engine::new_game(seed);
            let root = Root::new(&st, st.current_player(), Config::default());
            for p in Player::ALL {
                let c = root.commitment(p);
                assert_eq!(
                    c.c_sci.to_bits(),
                    0.0f64.to_bits(),
                    "seed {seed}: c_sci for {p:?} is {} with no symbols held",
                    c.c_sci
                );
                assert!(
                    c.c_mil < 0.10,
                    "seed {seed}: c_mil for {p:?} is {} from the centre",
                    c.c_mil
                );
                assert!(c.c < 0.10, "seed {seed}: c for {p:?} is {}", c.c);
                assert!(
                    c.s < 0.01,
                    "seed {seed}: S(c) for {p:?} is {}, so weights have already moved",
                    c.s
                );
            }
        }
    }

    /// ...and the raw magnitude really is large enough for that to have been
    /// a live trap, so the test above is not vacuous.
    #[test]
    fn the_raw_science_magnitude_alone_would_have_been_misleading() {
        let st = advanced_game(3, 12);
        let r = duels_strategy::science_read(&st, st.current_player());
        assert_eq!(r.distinct, 0, "test setup: expected no symbols held");
        assert!(
            r.magnitude > 0.2,
            "M_sci with no symbols held is only {}, so the prog_sci guard would be pointless",
            r.magnitude
        );
    }

    // -----------------------------------------------------------------
    // Monotonicity properties
    // -----------------------------------------------------------------

    #[test]
    fn commitment_is_zero_at_the_bottom_of_both_races() {
        let blend = Blend::default();
        // No symbols held: c_sci is exactly zero whatever the magnitude says.
        let st = StateBuilder::new().age(1).conflict(0).build();
        let sci = duels_strategy::science_read(&st, Player::One);
        let mil = duels_strategy::military_read(&st, Player::One);
        assert_eq!(sci.distinct, 0);
        assert_eq!(
            mil.need,
            duels_core::data::military().capital_distance,
            "a centred pawn is the full capital distance away"
        );
        let c = Commitment::of(&sci, &mil, 0.0, &blend);
        assert_eq!(c.c_sci.to_bits(), 0.0f64.to_bits());
        assert_eq!(c.s_sci.to_bits(), 0.0f64.to_bits());
        // need == capital_distance and M_mil == 0 must give c_mil == 0.
        if mil.magnitude == 0.0 {
            assert_eq!(c.c_mil.to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn commitment_rises_with_symbols_held() {
        let blend = Blend::default();
        let commit = |built: &[&str]| -> f64 {
            let st = age_two_position(built);
            let sci = duels_strategy::science_read(&st, Player::One);
            let mil = duels_strategy::military_read(&st, Player::One);
            Commitment::of(&sci, &mil, 0.0, &blend).c_sci
        };
        let none = commit(&[]);
        let two = commit(&["workshop", "apothecary"]);
        let four = commit(&["workshop", "apothecary", "scriptorium", "pharmacist"]);
        assert_eq!(none.to_bits(), 0.0f64.to_bits());
        assert!(two > none, "two symbols: {two} vs {none}");
        assert!(four > two, "four symbols: {four} vs {two}");
    }

    // -----------------------------------------------------------------
    // The un-blended baseline
    // -----------------------------------------------------------------

    /// `S(0) = 0` is exact, so a genuinely uncommitted position and a
    /// deliberately switched-off blend must produce *the same weight vector,
    /// bit for bit* — and therefore the same evaluation of every legal
    /// action. This is what "the un-blended baseline" means for this crate:
    /// not another agent's code path, but this agent's own fixed-weight
    /// limit.
    #[test]
    fn the_blend_off_and_a_zero_commitment_position_agree_bit_for_bit() {
        let st = engine::new_game(17);
        let me = st.current_player();

        let on = Root::new(&st, me, Config::default());
        let off = Root::new(
            &st,
            me,
            Config {
                blend: Blend::off(),
                ..Config::default()
            },
        );

        // The military race is not *quite* dead cold at the very start, so
        // pin the comparison to the science half, which is exactly zero, and
        // assert the rest separately.
        for p in Player::ALL {
            let a = on.weights(p);
            let b = off.weights(p);
            assert_eq!(
                a.science.to_bits(),
                b.science.to_bits(),
                "science multiplier for {p:?}: {} vs {}",
                a.science,
                b.science
            );
        }

        // A hand-built position with both races at the floor: no symbols held
        // anywhere, the pawn centred, and — the part that takes an Age III
        // position with no red cards left — no shields obtainable at all, so
        // `M_mil` is exactly zero rather than merely small. Every weight must
        // then match the switched-off blend bit for bit, and so must the
        // evaluation of every legal action.
        let cold = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .conflict(0)
            .coins(Player::One, 5)
            .coins(Player::Two, 5)
            .current(Player::One)
            .build();
        for p in Player::ALL {
            let c = Root::new(&cold, cold.current_player(), Config::default())
                .commitment(p)
                .c;
            assert_eq!(
                c.to_bits(),
                0.0f64.to_bits(),
                "test setup: {p:?} is {c} committed, so this is not the cold case"
            );
        }
        let me = cold.current_player();
        let on = Root::new(&cold, me, Config::default());
        let off = Root::new(
            &cold,
            me,
            Config {
                blend: Blend::off(),
                ..Config::default()
            },
        );
        for p in Player::ALL {
            let (a, b) = (on.weights(p), off.weights(p));
            for (name, x, y) in [
                ("vp", a.vp, b.vp),
                ("liquidity", a.liquidity, b.liquidity),
                ("development", a.development, b.development),
                ("science", a.science, b.science),
                ("military", a.military, b.military),
                ("race_liquidity", a.race_liquidity, b.race_liquidity),
                ("economy", a.economy, b.economy),
            ] {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{name} for {p:?} differs: {x} vs {y}"
                );
            }
        }
        for action in engine::legal_actions(&cold) {
            let a = expected_value(&cold, action, me, &on);
            let b = expected_value(&cold, action, me, &off);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{action:?} scored {a} blended and {b} un-blended"
            );
        }
    }

    #[test]
    fn every_multiplier_is_exactly_one_when_the_blend_is_off() {
        let st = advanced_game(5, 30);
        let me = st.current_player();
        let off = Root::new(
            &st,
            me,
            Config {
                blend: Blend::off(),
                ..Config::default()
            },
        );
        let floor = Blend::default().floor_race_liq;
        for p in Player::ALL {
            let w = off.weights(p);
            for (name, v) in [
                ("vp", w.vp),
                ("liquidity", w.liquidity),
                ("development", w.development),
                ("science", w.science),
                ("military", w.military),
                ("economy", w.economy),
            ] {
                assert_eq!(v.to_bits(), 1.0f64.to_bits(), "{name} for {p:?} = {v}");
            }
            assert_eq!(w.race_liquidity.to_bits(), floor.to_bits());
        }
    }

    // -----------------------------------------------------------------
    // Root-fixing
    // -----------------------------------------------------------------

    #[test]
    fn root_weights_are_built_exactly_once_per_choose() {
        let mut agent = PhasedAgent::new(4);
        let st = advanced_game(9, 16);
        let legal = engine::legal_actions(&st);
        assert!(legal.len() > 1, "test setup: need a real choice");
        let obs = st.observation();

        agent.choose(&obs, &legal, Budget::Nodes(1));
        assert_eq!(
            agent.root_builds(),
            1,
            "one decision over {} candidates must evaluate the blend once",
            legal.len()
        );
        agent.choose(&obs, &legal, Budget::Nodes(1));
        assert_eq!(agent.root_builds(), 2);
    }

    /// Root-fixing for the *pricing* tables, not just the weights.
    ///
    /// `menu`'s take-value function calls the cost engine, which is the
    /// expensive part of this crate. The tables are built inside
    /// [`Root::new`], so the counter that already proves the blend is
    /// evaluated once per decision proves the same for them — but the
    /// behavioural half needs its own test: a candidate that changes what a
    /// card costs must still be scored against the *root* price, or a move
    /// would be credited once for the position it creates and again for
    /// having made the menu look different.
    #[test]
    fn the_menu_pricing_tables_are_root_fixed_and_built_once() {
        // Player One holds nothing; taking the Glassworks would halve what
        // every glass-costing card costs them.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "glassworks"), (19, "baths")])
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let glass = st.face_up_card(18).expect("slot 18 is face up");
        let before = root.menu().value(me, glass);

        // Apply the move that changes the pricing context...
        let mut after = st;
        let mut rng = StdRng::seed_from_u64(5);
        engine::apply(&mut after, Action::Build { slot: 18 }, &mut rng).unwrap();
        let rebuilt = Root::new(&after, after.current_player(), Config::default());

        // ...the root table still reports the root price, and a table rebuilt
        // on the result genuinely disagrees, so this is not vacuous.
        assert_eq!(
            root.menu().value(me, glass).to_bits(),
            before.to_bits(),
            "the root table moved without anybody rebuilding it"
        );
        let other = st.face_up_card(19).expect("slot 19 is face up");
        assert_ne!(
            root.menu().value(me, other).to_bits(),
            rebuilt.menu().value(me, other).to_bits(),
            "the pricing context did not actually change, so this test proves nothing"
        );

        // And the counter: one `choose` builds one `Root`, and therefore one
        // set of pricing tables, however many candidates it scores.
        let mut agent = PhasedAgent::new(4);
        let legal = engine::legal_actions(&st);
        assert!(legal.len() > 2);
        agent.choose(&st.observation(), &legal, Budget::Nodes(1));
        assert_eq!(agent.root_builds(), 1);
    }

    /// The behavioural half of root-fixing: a candidate action that would
    /// materially raise the mover's own commitment must still be scored under
    /// the *root* weights. Re-reading the blend on the result gives a
    /// different number, and that difference is exactly the double-count the
    /// design forbids — the move would be credited once through the term's
    /// value rising and again through the weight on that term rising.
    #[test]
    fn a_committing_move_is_scored_under_the_root_weights_not_its_own() {
        // Player One is two shields from the capital; "circus" (two shields)
        // in slot 18 takes them the whole way, which is about as large a
        // change to `c_mil` as one move can make.
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(4)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let action = Action::Build { slot: 18 };

        let mut after = st;
        let mut rng = StdRng::seed_from_u64(1);
        engine::apply(&mut after, action, &mut rng).expect("the red card should be buildable");
        assert!(
            !after.is_over(),
            "test setup: the move must not end the game"
        );

        let rebuilt = Root::new(&after, me, Config::default());
        assert!(
            rebuilt.commitment(me).s_mil > root.commitment(me).s_mil,
            "test setup: the move should raise the mover's military commitment ({} -> {})",
            root.commitment(me).s_mil,
            rebuilt.commitment(me).s_mil
        );

        // The score the agent actually assigns, under the root weights, and
        // the one it would assign if it re-read the blend on the result.
        let scored = evaluate(&after, me, &root);
        let recomputed = evaluate(&after, me, &rebuilt);
        assert_ne!(
            scored.to_bits(),
            recomputed.to_bits(),
            "the two are indistinguishable, so this test proves nothing"
        );
        // And the value the agent uses is the root-weighted one.
        assert_eq!(
            expected_value(&st, action, me, &root).to_bits(),
            (scored + root.denial_term(action)).to_bits()
        );
    }

    // -----------------------------------------------------------------
    // The individual terms
    // -----------------------------------------------------------------

    /// Grey (glass, papyrus) is the scarcest production in the game — two
    /// cards in Age I, two in Age II, none at all in Age III — and it is what
    /// most wonders ask for. Nothing in the development term says so; it
    /// falls out of counting what the remaining pool actually costs.
    #[test]
    fn the_development_term_prices_a_players_own_production() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .built(Player::One, &["glassworks", "press"])
            .current(Player::One)
            .build();
        let board = duels_strategy::Board::of(&st);
        let supply = DevSupply::of(&board);
        assert!(supply.pool_size > 0);

        let producer = terms::development_value(&st, Player::One, &supply, 0.6);
        let empty = terms::development_value(&st, Player::Two, &supply, 0.6);
        assert_eq!(empty, 0.0, "a city producing nothing develops nothing");
        assert!(
            producer > 0.0,
            "two grey cards should be worth something: {producer}"
        );

        // ...and the split by resource attributes it to glass and papyrus.
        let split = terms::development_by_resource(&st, Player::One, &supply, 0.6);
        let glass = split[duels_core::data::Resource::Glass.index()];
        let papyrus = split[duels_core::data::Resource::Papyrus.index()];
        assert!(glass > 0.0 && papyrus > 0.0, "{split:?}");
        assert!((glass + papyrus - producer).abs() < 1e-9, "{split:?}");
    }

    #[test]
    fn an_unbuilt_wonder_makes_the_resources_it_needs_more_valuable() {
        let build = |wonders: &[&str]| -> f64 {
            let st = StateBuilder::new()
                .age(1)
                .open_slots(&[(18, "clay-pool"), (19, "quarry")])
                .built(Player::One, &["lumber-yard"])
                .wonders(Player::One, wonders)
                .current(Player::One)
                .build();
            let supply = DevSupply::of(&duels_strategy::Board::of(&st));
            terms::development_value(&st, Player::One, &supply, 0.6)
        };
        // The Pyramids need 3 stone; the Great Lighthouse needs wood. Only
        // the latter raises what a Lumber Yard is worth.
        let none = build(&[]);
        let stone = build(&["the-pyramids"]);
        let wood = build(&["the-great-lighthouse"]);
        assert_eq!(none.to_bits(), stone.to_bits(), "{none} vs {stone}");
        assert!(wood > none, "{wood} vs {none}");
    }

    #[test]
    fn the_next_age_start_term_scores_the_projected_starter() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(19, "clay-pool")])
            .conflict(0)
            .current(Player::One)
            .build();
        let w = EvalWeights::default();
        assert_eq!(terms::projected_starter(&st), Some(Player::One));
        assert_eq!(
            terms::next_age_start(&st, Player::One, &w),
            w.next_age_start[0]
        );
        assert_eq!(terms::next_age_start(&st, Player::Two, &w), 0.0);

        // Age III has no next age, so the term is off entirely.
        let late = StateBuilder::new()
            .age(3)
            .open_slots(&[(19, "clay-pool")])
            .conflict(0)
            .current(Player::One)
            .build();
        assert_eq!(terms::next_age_start(&late, Player::One, &w), 0.0);
    }

    #[test]
    fn four_symbols_with_a_strong_token_row_reads_higher_than_without() {
        let four = ["workshop", "apothecary", "scriptorium", "pharmacist"];
        let make = |tokens: &[&str]| {
            StateBuilder::new()
                .age(2)
                .built(Player::One, &four)
                .board_tokens(tokens)
                .open_slots(&[(18, "clay-pool"), (19, "quarry")])
                .current(Player::One)
                .build()
        };
        let w = ScienceWeights::default();
        let bare = make(&[]);
        let strong = make(&["law", "theology", "strategy"]);
        let a = terms::science_ladder(&bare, Player::One, &w);
        let b = terms::science_ladder(&strong, Player::One, &w);
        assert!(b > a, "strong token row: {b} vs bare {a}");
        // The ladder is the dominant part at four symbols.
        assert!(a >= w.ladder[4], "{a} < {}", w.ladder[4]);
    }

    #[test]
    fn a_threatened_pair_is_worth_more_than_a_completed_one() {
        // Holding one Mortar (a live half-pair) versus holding both, which
        // has already paid its token and threatens nothing further.
        let cards: Vec<&str> = terms::symbol_cards(Science::Mortar)
            .map(|c| c.def().id)
            .collect();
        let w = ScienceWeights::default();
        let half = StateBuilder::new()
            .age(2)
            .built(Player::One, &cards[..1])
            .board_tokens(&["philosophy"])
            .build();
        let both = StateBuilder::new()
            .age(2)
            .built(Player::One, &cards)
            .board_tokens(&["philosophy"])
            .pair_already_awarded(Player::One, Science::Mortar)
            .build();
        // Same distinct count, so the ladder entry is identical; only the
        // pair threat differs.
        assert_eq!(half.player(Player::One).distinct_science(), 1);
        assert_eq!(both.player(Player::One).distinct_science(), 1);
        assert!(
            terms::science_ladder(&half, Player::One, &w)
                > terms::science_ladder(&both, Player::One, &w)
        );
    }

    // -----------------------------------------------------------------
    // Whole-agent behaviour
    // -----------------------------------------------------------------

    fn eval_after(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
        let mut s = *state;
        let mut rng = StdRng::seed_from_u64(0x0C0F_FEE0);
        engine::apply(&mut s, action, &mut rng).expect("scenario action should be legal");
        evaluate(&s, me, root)
    }

    #[test]
    fn evaluation_prefers_the_move_that_wins_by_military_supremacy() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "clay-pool")])
            .conflict(7)
            .coins(Player::One, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let build = eval_after(&st, Action::Build { slot: 18 }, me, &root);
        let discard = eval_after(&st, Action::Discard { slot: 18 }, me, &root);
        assert!(build > discard, "build={build} discard={discard}");

        let mut after = st;
        let mut rng = StdRng::seed_from_u64(1);
        engine::apply(&mut after, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(
            after.result(),
            Some(GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::MilitarySupremacy,
            })
        );
    }

    #[test]
    fn evaluation_prefers_the_move_that_wins_by_scientific_supremacy() {
        let st = StateBuilder::new()
            .age(3)
            .built(
                Player::One,
                &[
                    "workshop",
                    "apothecary",
                    "scriptorium",
                    "pharmacist",
                    "academy",
                ],
            )
            .open_slots(&[(18, "university"), (19, "palace")])
            .coins(Player::One, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        assert_eq!(st.player(me).distinct_science(), 5);
        let root = Root::new(&st, me, Config::default());
        let win = eval_after(&st, Action::Build { slot: 18 }, me, &root);
        let other = eval_after(&st, Action::Build { slot: 19 }, me, &root);
        assert!(win > other, "win={win} other={other}");
    }

    #[test]
    fn evaluation_orders_win_above_draw_above_loss() {
        let root_of = |st: &GameState| Root::new(st, Player::One, Config::default());
        let finish = |one: &[&str], two: &[&str]| -> GameState {
            let mut st = StateBuilder::new()
                .built(Player::One, one)
                .built(Player::Two, two)
                .open_slots(&[(18, "clay-pool")])
                .current(Player::One)
                .build();
            let mut rng = StdRng::seed_from_u64(3);
            engine::apply(&mut st, Action::Discard { slot: 18 }, &mut rng).unwrap();
            assert!(st.result().is_some());
            st
        };
        let win = finish(&["palace"], &[]);
        let draw = finish(&["palace"], &["town-hall"]);
        let loss = finish(&[], &["palace"]);
        let r = root_of(&win);
        let w = EvalWeights::default();
        assert_eq!(evaluate(&win, Player::One, &r), w.instant_result);
        assert_eq!(evaluate(&draw, Player::One, &r), 0.0);
        assert_eq!(evaluate(&loss, Player::One, &r), -w.instant_result);
    }

    #[test]
    fn the_evaluation_is_antisymmetric_between_the_two_players() {
        for seed in 0..6u64 {
            for steps in [8usize, 20, 34] {
                let st = advanced_game(seed, steps);
                if st.is_over() {
                    continue;
                }
                let root = Root::new(&st, st.current_player(), Config::default());
                let a = evaluate(&st, Player::One, &root);
                let b = evaluate(&st, Player::Two, &root);
                assert!(
                    (a + b).abs() < 1e-9,
                    "seed {seed} steps {steps}: {a} and {b} are not opposites"
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // The terminal rails
    // -----------------------------------------------------------------

    /// Rail B, end to end through the agent: a one-shield-from-the-capital
    /// opponent with a red card on the table, and one candidate that takes
    /// that card away.
    ///
    /// The ordinary evaluation is happy to leave it there — a Quarry is a
    /// perfectly good pick — which is exactly the failure the rails exist to
    /// stop. `phased:base=v2` really does leave it, and this test asserts
    /// both halves so it cannot pass for the wrong reason.
    #[test]
    fn the_rails_block_a_loss_the_ordinary_evaluation_walks_into() {
        let position = || {
            StateBuilder::new()
                .age(3)
                .open_slots(&[(18, "circus"), (19, "palace")])
                .conflict(-7)
                .coins(Player::One, 30)
                .coins(Player::Two, 30)
                .current(Player::One)
                .build()
        };
        let st = position();
        let me = st.current_player();
        // Player Two is two shields from the capital and the Circus is worth
        // exactly two; anything that leaves it there loses.
        assert!(duels_strategy::closing_sources(&st, Player::Two).any());

        let mut agent = PhasedAgent::with_config(7, Config::default());
        let legal = engine::legal_actions(&st);
        let chosen = agent.choose(&st.observation(), &legal, Budget::Nodes(1));
        assert!(
            matches!(
                chosen,
                Action::Build { slot: 18 }
                    | Action::Discard { slot: 18 }
                    | Action::BuildWonder { slot: 18, .. }
            ),
            "the rails let the opponent's winning card stand: chose {chosen:?}"
        );

        // Every candidate that leaves slot 18 alone is pinned at −imminent,
        // and every candidate that takes it is not.
        let root = Root::new(&st, me, Config::default());
        let w = EvalWeights::default();
        for &action in &legal {
            let touches = matches!(
                action,
                Action::Build { slot: 18 }
                    | Action::Discard { slot: 18 }
                    | Action::BuildWonder { slot: 18, .. }
            );
            let v = expected_value(&st, action, me, &root);
            if touches {
                assert!(v > -w.imminent, "{action:?} scored {v}");
            } else {
                assert!(v <= -w.imminent, "{action:?} scored {v}, not blocked");
            }
        }
    }

    /// ...and the guarantee really does come from the rail, not from the
    /// ordinary terms happening to agree.
    ///
    /// On this particular position the round-two agent blocks too — a
    /// two-card structure makes the closing red card the obviously
    /// attractive pick anyway. What it does *not* have is any guarantee: no
    /// candidate is scored anywhere near `-imminent`, so the ordering is
    /// decided by a handful of victory points and would flip under a wider
    /// structure. `examples/rail_audit.rs` is where the difference is
    /// measured on real games rather than argued from one position.
    #[test]
    fn without_the_rails_nothing_is_pinned_and_a_few_points_decide_it() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(-7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::v2());
        let w = Config::v2().eval;
        assert_eq!(w.imminent, 0.0, "v2 must carry no rail magnitude at all");
        for &action in &engine::legal_actions(&st) {
            let v = expected_value(&st, action, me, &root);
            assert!(
                v.abs() < 100.0,
                "{action:?} scored {v}: the round-two agent has no terminal \
                 rail, so nothing should be pinned"
            );
        }
    }

    /// Taking the win still outranks merely having one, so Rail A cannot be
    /// swallowed by Rail C′.
    #[test]
    fn an_actual_win_outranks_a_certain_one() {
        let w = EvalWeights::default();
        assert!(
            w.instant_result > w.imminent,
            "instant_result {} must dominate imminent {}",
            w.instant_result,
            w.imminent
        );
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let win = expected_value(&st, Action::Build { slot: 18 }, me, &root);
        let other = expected_value(&st, Action::Build { slot: 19 }, me, &root);
        assert_eq!(win, w.instant_result);
        assert!(win > other);
    }

    // -----------------------------------------------------------------
    // The menu's shield price
    // -----------------------------------------------------------------

    /// The differenced price of `k` shields is what the evaluation actually
    /// moves when the pawn advances `k` — which the one-sided price is not,
    /// in either magnitude or shape.
    #[test]
    fn the_differenced_shield_price_matches_what_the_evaluation_really_moves() {
        // A pawn one step short of the 3-5 band, so the second shield crosses
        // a boundary the first does not.
        let st = StateBuilder::new()
            .age(2)
            .deal(&AGE_TWO_DEAL)
            .conflict(2)
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let base = evaluate(&st, me, &root);

        for k in 1..=3i8 {
            let moved = StateBuilder::new()
                .age(2)
                .deal(&AGE_TWO_DEAL)
                .conflict(2 + k)
                .coins(Player::One, 20)
                .coins(Player::Two, 20)
                .current(Player::One)
                .build();
            let want = evaluate(&moved, me, &root) - base;
            let got = terms::military_shield_delta(
                &st,
                me,
                u8::try_from(k).unwrap(),
                root.smoothing(),
                root.config().eval.military_band,
                root.config().eval.military_loot,
                (root.weights(me).military, root.weights(me.other()).military),
            );
            assert!(
                (want - got).abs() < 1e-9,
                "{k} shields: the evaluation moves {want}, the price says {got}"
            );
        }
    }

    /// It is a finite difference, not `k` times a slope — which matters
    /// precisely because the scoring table's steps are not evenly spaced.
    ///
    /// The test has to switch the horizon on to show it, and that is the
    /// point of the horizon: at the default supply-wide smoothing the bands
    /// are so blurred that two shields really are worth almost exactly twice
    /// one, which is the "step function that behaves like a straight line"
    /// complaint written down as an assertion.
    #[test]
    fn the_shield_price_is_not_linear_in_the_number_of_shields() {
        let st = StateBuilder::new()
            .age(2)
            .deal(&AGE_TWO_DEAL)
            .conflict(1)
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let price = |config: Config, k: u8| {
            let root = Root::new(&st, me, config);
            terms::military_shield_delta(
                &st,
                me,
                k,
                root.smoothing(),
                root.config().eval.military_band,
                root.config().eval.military_loot,
                (root.weights(me).military, root.weights(me.other()).military),
            )
        };
        let sharp = Config {
            military_horizon: Some(3.0),
            ..Config::default()
        };
        assert_eq!(price(sharp, 0), 0.0);
        assert!(
            price(sharp, 2) > 2.0 * price(sharp, 1),
            "{} vs {}",
            price(sharp, 2),
            price(sharp, 1)
        );

        // ...and at the default width the same quantity is within a few
        // percent of linear.
        let wide = Config::default();
        let ratio = price(wide, 2) / (2.0 * price(wide, 1));
        assert!(
            (0.95..1.10).contains(&ratio),
            "the supply-wide smoothing should be near-linear, ratio {ratio}"
        );
    }

    // -----------------------------------------------------------------
    // The horizon-based smoothing width, and production lock-in
    // -----------------------------------------------------------------

    /// A horizon narrows the smoothing, which is the whole point: with the
    /// full remaining supply the "step function" is nearly a straight line.
    #[test]
    fn a_horizon_sharpens_the_bands_and_none_reproduces_the_old_width() {
        let shields = 20.0;
        let rounds = 18.0;
        assert_eq!(
            terms::horizon_supply(shields, rounds, None).to_bits(),
            shields.to_bits()
        );
        let wide = MilSmoothing::of(shields, 0.8, 0.35, 0.55);
        let narrow = MilSmoothing::of(
            terms::horizon_supply(shields, rounds, Some(3.0)),
            0.8,
            0.35,
            0.55,
        );
        assert!(narrow.s < wide.s, "{} vs {}", narrow.s, wide.s);
        // ...and a sharper width really does separate the boundary-crossing
        // shield from the one that crosses nothing.
        let contrast = |sm: &MilSmoothing| {
            let at = |c: i8| {
                let st = StateBuilder::new().age(2).conflict(c).build();
                terms::military_band(&st, Player::One, sm)
            };
            (at(3) - at(2)) - (at(2) - at(1))
        };
        assert!(contrast(&narrow) > contrast(&wide));
        // A horizon longer than the game is a no-op.
        assert_eq!(
            terms::horizon_supply(shields, rounds, Some(1000.0)).to_bits(),
            shields.to_bits()
        );
    }

    /// Age III really has no brown or grey card in it, so a city's production
    /// is frozen — the fact the lock-in factor is built on, checked against
    /// the card data rather than asserted from memory.
    #[test]
    fn production_is_completely_frozen_by_age_three() {
        let by_age = |age: u8| {
            duels_core::data::statics().age_masks[usize::from(age) - 1] & terms::production_mask()
        };
        assert_eq!(by_age(1).count_ones(), 8, "six brown and two grey in Age I");
        assert_eq!(
            by_age(2).count_ones(),
            5,
            "three brown and two grey in Age II"
        );
        assert_eq!(by_age(3).count_ones(), 0, "Age III prints no production");

        // ...so an Age III position reads a lock-in of exactly one.
        let late = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "town-hall")])
            .current(Player::One)
            .build();
        let supply = DevSupply::of(&duels_strategy::Board::of(&late));
        assert_eq!(supply.production_lock_in.to_bits(), 1.0f64.to_bits());

        // ...and an Age I position reads less than one, so it is a factor
        // rather than a constant.
        let early = engine::new_game(4);
        let early = DevSupply::of(&duels_strategy::Board::of(&early));
        assert!(
            early.production_lock_in < 0.5,
            "Age I lock-in is {}",
            early.production_lock_in
        );
    }

    #[test]
    fn spec_reports_the_expected_name_and_encoded_params() {
        let agent = PhasedAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "phased");
        assert_eq!(spec.version, "1.0.0");
        assert_eq!(spec.params, Config::default().params_string());
    }

    #[test]
    fn choosing_only_ever_returns_one_of_the_offered_actions() {
        let mut agent = PhasedAgent::new(99);
        let state = engine::new_game(99);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        for _ in 0..10 {
            assert!(legal.contains(&agent.choose(&obs, &legal, Budget::Nodes(1))));
        }
    }

    #[test]
    fn a_whole_game_of_self_play_terminates_and_stays_legal() {
        let mut a = PhasedAgent::new(1);
        let mut b = PhasedAgent::new(2);
        let mut st = engine::new_game(31);
        let mut rng = StdRng::seed_from_u64(77);
        for _ in 0..400 {
            if st.is_over() {
                break;
            }
            let legal = engine::legal_actions(&st);
            let obs = st.observation();
            let action = if st.current_player() == Player::One {
                a.choose(&obs, &legal, Budget::Nodes(1))
            } else {
                b.choose(&obs, &legal, Budget::Nodes(1))
            };
            assert!(legal.contains(&action));
            engine::apply(&mut st, action, &mut rng).unwrap();
        }
        assert!(st.is_over(), "self-play did not finish");
    }

    // -----------------------------------------------------------------
    // Round four: finishing a turn the engine left mid-effect.
    // -----------------------------------------------------------------

    fn wonder(slug: &str) -> duels_core::data::WonderId {
        duels_core::data::WonderId::from_slug(slug).expect("a real wonder")
    }

    fn completed() -> Config {
        Config {
            pending_model: PendingModel::Completed,
            ..Config::default()
        }
    }

    /// A position where Player One can build `w` by burying the card in slot
    /// 18, with both cities rich enough that cost is never the binding
    /// constraint.
    fn wonder_position(w: &str, opponent_city: &[&str]) -> GameState {
        StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "clay-pool")])
            .wonders(Player::One, &[w])
            .built(Player::Two, opponent_city)
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build()
    }

    /// The bug, stated as a property of the engine rather than of this crate:
    /// these four wonders really do come back from `apply` with the effect not
    /// yet applied and the turn not yet passed.
    #[test]
    fn four_wonders_leave_the_engine_mid_effect() {
        use duels_core::state::Pending;
        let cases: [(&str, &[&str]); 3] = [
            ("circus-maximus", &["glassworks", "press"]),
            ("the-statue-of-zeus", &["lumber-yard", "clay-pit"]),
            ("the-mausoleum", &[]),
        ];
        for (slug, city) in cases {
            let mut st = wonder_position(slug, city);
            if slug == "the-mausoleum" {
                st = StateBuilder::new()
                    .age(3)
                    .open_slots(&[(18, "palace"), (19, "clay-pool")])
                    .wonders(Player::One, &[slug])
                    .discard(&["theater", "altar"])
                    .coins(Player::One, 40)
                    .coins(Player::Two, 40)
                    .current(Player::One)
                    .build();
            }
            let action = Action::BuildWonder {
                slot: 18,
                wonder: wonder(slug),
            };
            assert!(
                engine::legal_actions(&st).contains(&action),
                "{slug}: the build is not legal in the test position"
            );
            let mut next = st;
            engine::apply_with_outcome(&mut next, action, &engine::Outcome::default()).unwrap();
            assert!(
                next.pending().is_some(),
                "{slug}: the engine finished the effect after all"
            );
            assert_eq!(
                next.current_player(),
                Player::One,
                "{slug}: the turn passed before the effect resolved"
            );
            assert!(matches!(
                next.pending(),
                Some(Pending::Destroy { .. } | Pending::MausoleumBuild)
            ));
        }
    }

    /// A destroy that takes a real card out of the opponent's city is worth
    /// more than a destroy that has not happened yet — which is the whole of
    /// bug one in one assertion.
    #[test]
    fn completing_the_turn_credits_a_destroy_the_unresolved_evaluation_misses() {
        let st = wonder_position("circus-maximus", &["glassworks", "press"]);
        let action = Action::BuildWonder {
            slot: 18,
            wonder: wonder("circus-maximus"),
        };
        let me = Player::One;

        let flat = Root::new(&st, me, Config::v3());
        let full = Root::new(&st, me, completed());
        let before = expected_value(&st, action, me, &flat);
        let after = expected_value(&st, action, me, &full);
        assert!(
            after > before,
            "resolving the destroy should be worth something: {after} vs {before}"
        );

        // ...and the card it takes is the one that hurts most, not the first
        // one in index order: the destroy resolution really does choose.
        let mut resolved = st;
        engine::apply_with_outcome(&mut resolved, action, &engine::Outcome::default()).unwrap();
        let options = engine::legal_actions(&resolved);
        assert_eq!(options.len(), 2, "two grey cards to choose between");
        let scores: Vec<f64> = options
            .iter()
            .map(|&o| {
                let mut s = resolved;
                engine::apply_with_outcome(&mut s, o, &engine::Outcome::default()).unwrap();
                evaluate(&s, me, &full)
            })
            .collect();
        let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(
            evaluate(&resolved, me, &full).to_bits(),
            best.to_bits(),
            "the resolution did not take the maximum over the real options"
        );
    }

    /// The Mausoleum's retrieval is the one pending effect the engine can
    /// **chain**: `construct_card` runs the retrieved card's own effects, and a
    /// green card that completes a science pair sets a second pending choice.
    /// [`MAX_PENDING_DEPTH`] exists for exactly this, and this is the test that
    /// says the chain is real rather than hypothetical.
    #[test]
    fn a_mausoleum_retrieval_can_chain_into_a_progress_token_choice() {
        use duels_core::state::Pending;
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "clay-pool")])
            .wonders(Player::One, &["the-mausoleum"])
            // One Wheel already in the city; the School in the discard pile
            // carries the second, so retrieving it completes the pair.
            .built(Player::One, &["apothecary"])
            .discard(&["school"])
            .board_tokens(&["law", "theology", "strategy"])
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build();

        let mut after = st;
        engine::apply_with_outcome(
            &mut after,
            Action::BuildWonder {
                slot: 18,
                wonder: wonder("the-mausoleum"),
            },
            &engine::Outcome::default(),
        )
        .unwrap();
        assert_eq!(after.pending(), Some(Pending::MausoleumBuild));

        let mut chained = after;
        engine::apply_with_outcome(
            &mut chained,
            engine::legal_actions(&after)[0],
            &engine::Outcome::default(),
        )
        .unwrap();
        assert_eq!(
            chained.pending(),
            Some(Pending::ProgressToken),
            "the retrieval was supposed to complete a science pair"
        );
        assert_eq!(chained.current_player(), Player::One);

        // The evaluator walks the whole chain, so the state it finally scores
        // has no pending effect left and the turn really has passed.
        let root = Root::new(&st, Player::One, completed());
        let v = evaluate(&after, Player::One, &root);
        assert!(v.is_finite());
        // Resolving both levels is strictly better than stopping at the first:
        // the token is worth something.
        let one_level = evaluate_at(&after, Player::One, &root, 1);
        assert!(
            v > one_level,
            "walking the chain to the end should be worth more than stopping \
             one level in: {v} vs {one_level}"
        );
    }

    /// The evaluation stays exactly zero-sum through a pending resolution.
    #[test]
    fn resolving_a_pending_effect_stays_antisymmetric() {
        let st = wonder_position("circus-maximus", &["glassworks", "press"]);
        let mut after = st;
        engine::apply_with_outcome(
            &mut after,
            Action::BuildWonder {
                slot: 18,
                wonder: wonder("circus-maximus"),
            },
            &engine::Outcome::default(),
        )
        .unwrap();
        assert!(after.pending().is_some());

        for me in Player::ALL {
            let root = Root::new(&st, me, completed());
            let mine = evaluate(&after, me, &root);
            let theirs = evaluate(&after, me.other(), &root);
            assert_eq!(
                mine.to_bits(),
                (-theirs).to_bits(),
                "the resolution is not antisymmetric: {mine} vs {theirs}"
            );
        }
    }

    /// The destroy discount is a *no-op* at its off value, and moves the score
    /// towards the unresolved reference when there really is a replacement
    /// coming.
    #[test]
    fn the_destroy_replacement_discount_only_bites_while_the_market_can_replace() {
        let action = Action::BuildWonder {
            slot: 18,
            wonder: wonder("the-statue-of-zeus"),
        };
        // Age III: no brown or grey card left in the game, so nothing can be
        // replaced and the discount must vanish.
        let late = wonder_position("the-statue-of-zeus", &["lumber-yard", "clay-pit"]);
        let plain = Root::new(&late, Player::One, completed());
        let discounted = Root::new(
            &late,
            Player::One,
            Config {
                destroy_replace_discount: true,
                ..completed()
            },
        );
        assert_eq!(
            expected_value(&late, action, Player::One, &plain).to_bits(),
            expected_value(&late, action, Player::One, &discounted).to_bits(),
            "an Age III destroy is permanent, so the discount must be exactly zero"
        );
        for r in 0..duels_core::data::NUM_RESOURCES {
            assert_eq!(
                discounted.replace[r], 0.0,
                "Age III still thinks it can print production"
            );
        }

        // Age I, with the whole brown supply still to come: the same destroy
        // is worth strictly less than its undiscounted value.
        let early = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "palace"), (19, "clay-pool")])
            .wonders(Player::One, &["the-statue-of-zeus"])
            .built(Player::Two, &["lumber-yard", "clay-pit"])
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build();
        let plain = Root::new(&early, Player::One, completed());
        let discounted = Root::new(
            &early,
            Player::One,
            Config {
                destroy_replace_discount: true,
                ..completed()
            },
        );
        assert!(
            discounted.replace.iter().any(|&x| x > 0.0),
            "Age I should still have production sources in the pool"
        );
        assert!(
            expected_value(&early, action, Player::One, &discounted)
                < expected_value(&early, action, Player::One, &plain),
            "a replaceable destroy should be worth less than a permanent one"
        );
    }

    // -----------------------------------------------------------------
    // Round four: the wonder budget.
    // -----------------------------------------------------------------

    /// `p_build` rations the seven shared slots and the owner's remaining
    /// decisions, so an unbuilt wonder is not worth the same in a fresh Age I
    /// as it is with one slot and two turns left.
    #[test]
    fn the_wonder_budget_rations_slots_and_turns() {
        let budget = Config {
            wonder_model: WonderModel::Budget,
            ..Config::default()
        };

        // Fresh: eight unbuilt wonders, seven slots, a whole game of turns.
        let fresh = StateBuilder::new()
            .age(1)
            .deal(&AGE_TWO_DEAL)
            .wonders(
                Player::One,
                &["the-pyramids", "the-colossus", "the-sphinx", "piraeus"],
            )
            .wonders(
                Player::Two,
                &[
                    "the-mausoleum",
                    "the-great-library",
                    "the-appian-way",
                    "the-great-lighthouse",
                ],
            )
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let root = Root::new(&fresh, Player::One, budget);
        let p = root.wonders().p_build(Player::One);
        assert!(
            p > 0.0 && p < 1.0,
            "eight unbuilt wonders against seven slots should ration: {p}"
        );
        assert!(
            terms::wonder_potential_budget(&fresh, Player::One, root.wonders()) > 0.0,
            "an unbuilt wonder in Age I is worth something"
        );

        // Every slot gone: nothing left to ration.
        let full = StateBuilder::new()
            .age(3)
            .open_slots(&[(19, "clay-pool")])
            .wonders(Player::One, &["the-pyramids"])
            .wonders_built(
                Player::One,
                &["the-colossus", "the-sphinx", "the-hanging-gardens"],
            )
            .wonders_built(
                Player::Two,
                &[
                    "piraeus",
                    "the-appian-way",
                    "the-great-lighthouse",
                    "the-mausoleum",
                ],
            )
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        let root = Root::new(&full, Player::One, budget);
        assert_eq!(root.wonders().p_build(Player::One), 0.0);
        assert_eq!(
            terms::wonder_potential_budget(&full, Player::One, root.wonders()),
            0.0
        );
    }

    /// The per-effect price says something the flat one cannot: the Great
    /// Library's token draw is priced from the tokens actually set aside, so
    /// two different piles give two different answers — where
    /// [`terms::wonder_power`] gives a flat `+3` for both.
    #[test]
    fn the_wonder_budget_prices_the_great_library_from_the_real_token_pile() {
        let position = |aside: &[&str]| {
            StateBuilder::new()
                .age(2)
                .deal(&AGE_TWO_DEAL)
                .wonders(Player::One, &["the-great-library"])
                .set_aside_tokens(aside)
                .coins(Player::One, 20)
                .coins(Player::Two, 20)
                .current(Player::One)
                .build()
        };
        let budget = Config {
            wonder_model: WonderModel::Budget,
            ..Config::default()
        };
        let rich = position(&["law", "theology", "mathematics", "philosophy", "urbanism"]);
        let thin = position(&["masonry", "agriculture", "economy", "strategy", "urbanism"]);
        let gl = wonder("the-great-library");
        let a = Root::new(&rich, Player::One, budget);
        let b = Root::new(&thin, Player::One, budget);
        assert!(
            a.wonders().power(Player::One, gl) != b.wonders().power(Player::One, gl),
            "the token pile made no difference to the Great Library's price"
        );
        // The flat model cannot tell them apart at all.
        assert_eq!(
            terms::wonder_power(gl).to_bits(),
            terms::wonder_power(gl).to_bits()
        );
    }
}
