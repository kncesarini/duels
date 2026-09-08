//! `duels-eval`: a hand-crafted position evaluation whose every weight is a
//! continuous function of how committed each player is to a win condition.
//!
//! # Where this sits
//!
//! This crate is a **library, not an agent**. It owns [`Config`], [`Root`],
//! [`evaluate`] and [`expected_value`] — everything needed to put a
//! victory-point-scale number on a position — and nothing that decides a move.
//! `duels-agent-phased` is the 1-ply agent built on it: it samples one concrete
//! state per decision, builds one [`Root`], scores every legal action with
//! [`expected_value`] and returns the best. It was this crate's only caller
//! when the evaluation was extracted out of it, and everything below was
//! written while the two were one crate — read "this agent" as "`phased`", and
//! every measurement as one taken with `phased` driving.
//!
//! It exists as its own crate because more than one agent is going to want the
//! evaluation, and this repository's rule is that **no agent crate depends on
//! another agent crate** (see `CLAUDE.md`). A shared evaluation therefore has
//! to live below the agents, next to [`duels_strategy`], rather than inside
//! whichever agent happened to build it first.
//!
//! Nothing here is random and nothing here reads a clock: [`Root::new`],
//! [`evaluate`] and [`expected_value`] are pure functions of the state handed
//! to them, which is why this crate depends on `duels-core` and
//! `duels-strategy` and on nothing else.
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
//! ## Against the ladder, at both budget kinds and at ten times the budget
//!
//! 800 games per seed range at seeds 1 and 5001, `Nodes(1)` for the 1-ply
//! opponents:
//!
//! ```text
//!                    this agent          phased:base=v4
//! vs random          799-1 / 797-3       799-1 / 799-1
//! vs greedy          799-1 / 799-1       794-6 / 794-6
//! vs greedy-ev       799-1 / 799-1       799-1 / 798-2
//! vs strategist      800-0 / 795-5       800-0 / 797-3
//! ```
//!
//! Everything below `alphabeta` is at the ceiling and stays there. The two
//! search opponents are where the measurement is, and this project's
//! two-budget discipline matters for them even though it cannot matter for
//! `phased` itself (a 1-ply agent ignores its budget — `choose` takes
//! `_budget`), because it is what decides how strong the *opponent* is:
//!
//! ```text
//!                                  this agent            phased:base=v4
//! vs alphabeta  Nodes(2000)        260/800 / 250/800     176/800 / 169/800
//!               TimeMs(20)         254/800 / 257/800     197/800 / 169/800
//!               Nodes(20000)        88/400 /  79/400      56/400 /  51/400
//! vs mcts-uct   Nodes(2000)        123/800 / 125/800      84/800 /  72/800
//!               TimeMs(20)         131/800 / 137/800      95/800 /  86/800
//! ```
//!
//! Six paired comparisons against two unrelated searchers, and every one of
//! them moves the same way on both ranges: 22% to 32% against `alphabeta` at
//! `Nodes(2000)`, 10% to 16% against `mcts-uct`, and — the one worth having —
//! **14% to 22% against an `alphabeta` given ten times the nodes**, where it
//! concedes only 13-14% to the round-four agent. The gain is not an artefact of
//! a particular opponent, a particular budget kind, or a particular search
//! depth, and it is not self-play overfitting.
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
//! # Round six: the one wonder effect that is not like the others
//!
//! Round four built [`WonderModel::Budget`], a per-effect price for an unbuilt
//! wonder, and measured it **negative** (−11.0 / −5.6 / −3.9 Elo). Its
//! follow-up note blamed at least part of that on
//! [`duels_strategy::science::token_value`]'s flat, unmeasured constants
//! feeding the Great Library channel — which is to say, on a channel that has
//! nothing to do with the one thing a strong player will tell you first:
//! **an extra turn is the most valuable thing a wonder can print.**
//!
//! `docs/strategy-backlog.md` §2.1 puts all five play-again wonders in its top
//! tier and nothing else, and §1.2 explains why in terms this evaluation has no
//! other way to see: with strict alternation the whole slot sequence is
//! pre-determined, and an extra turn is the one thing that re-assigns every
//! remaining slot. A 1-ply evaluation cannot search that out; a constant is the
//! only instrument it has.
//!
//! `Budget` cannot answer that. It moves eight channels at once, several of
//! them known-weak, so its verdict on extra turns is confounded with its
//! verdict on everything else. Round six therefore leaves `Budget` exactly
//! where it is and adds **one number** to the default
//! [`WonderModel::Flat`] path instead:
//! [`EvalWeights::wonder_extra_turn_premium`], added on top of
//! [`terms::wonder_power`]'s flat "+3, this wonder has an effect" for a wonder
//! that prints play again, and nothing else. At `0.0` — [`Config::v5`] — it is
//! round five bit for bit (`tests/v5_identity.rs`). **The new default is
//! `9.0`**, so a play-again wonder is worth `12` where every other effect is
//! worth `3`.
//!
//! ## Which wonders, and one mechanic this deliberately does not price
//!
//! Exactly five wonders print play again — Piraeus, The Appian Way, The
//! Hanging Gardens, The Sphinx and The Temple of Artemis — read off
//! `data/wonders.json` through `WonderDef::play_again` rather than by slug, and
//! pinned at five by
//! `v5_identity::an_extra_turn_premium_moves_only_the_five_play_again_wonders`.
//! Each grants exactly one extra turn, on construction, unconditionally; there
//! is no per-wonder difference in when or how it triggers.
//!
//! The **Theology** progress token grants play again for *every* wonder its
//! holder builds, and the premium does not price that. That is a real
//! omission, taken on purpose: folding it in would have every unbuilt wonder in
//! a Theology holder's hand collect the premium at once, and the sweep below
//! would then be measuring two things. It is the obvious follow-up.
//!
//! ## Elo by magnitude
//!
//! `phased:wprem=P` against `phased:base=v5` at `Nodes(1)`, **3200 games per
//! seed range**, paired and seat-swapped. Every interval is ±12.1.
//!
//! ```text
//! premium   total    seed 1    seed 100001   seed 200001   seed 300001   seed 400001
//!    1.5      4.5    +18.8       +32.5         +27.1
//!    3.0      6.0    +30.1       +48.5         +43.1
//!    6.0      9.0    +39.2       +53.5         +46.7
//!    7.5     10.5                                            +39.2         +52.6
//!    9.0     12.0    +45.3       +51.6         +49.9         +38.5         +51.9
//!   10.5     13.5                                            +38.5         +51.1
//!   12.0     15.0    +46.4       +48.4         +48.4         +41.1         +51.2
//!   15.0     18.0    +35.5       +38.3         +33.9
//! ```
//!
//! Positive on **every range at every magnitude tested** — the sign of the
//! effect is not in doubt — on a curve that rises to a broad plateau between 6
//! and 12 and falls away by 15. `9.0` is the middle of that plateau and has the
//! best mean over five ranges (+47.4); it is not distinguishable from `12.0`,
//! which reads +47.1 and beats it head-to-head by −2.7 / +5.0 / +9.2 over 3200
//! games on three ranges. `9.0` is taken as the smaller change of two that
//! measure the same.
//!
//! The shipped default is then confirmed on **two further disjoint ranges it
//! was not chosen on**: `phased` against `phased:base=v5` reads **+53.5**
//! [+41.3, +65.7] at seed 500001 and **+45.9** [+33.7, +58.0] at seed 600001,
//! 3200 games each. As a null calibration, `phased` against `phased:wprem=9` —
//! the same configuration under two names, differing only in the tie-break RNG
//! seed the arena hands each side — reads **+1.7** [−22.3, +25.8].
//!
//! ## Two controls, because +47 Elo for one constant deserves them
//!
//! A single number that buys forty-seven Elo is exactly the kind of result that
//! is usually measuring something other than what it claims. Two controls:
//!
//! **1. Is it just "the wonder term wants a bigger weight"?** No.
//! [`EvalWeights::wonder_potential`] scales the *whole* flat wonder term;
//! raising it from its default 0.5 buys almost nothing:
//!
//! ```text
//! wonder_potential   seed 1    seed 100001
//!        0.60         -5.5       +5.9
//!        0.70         -1.4       +3.5
//!        0.85         +6.9       +7.8
//!        1.00        +12.6       +9.2
//! ```
//!
//! **2. Is it the premium, or is it *which wonders get it*?** The latter,
//! decisively. The identical `+9`, applied to the four **pending-effect**
//! wonders (Circus Maximus, the Statue of Zeus, the Mausoleum, the Great
//! Library) instead of the five play-again ones, is worth **−66.0 / −76.9 /
//! −66.9** Elo over 3200 games on each of three disjoint ranges — and `+3`
//! there is worth −26.9 / −25.2 / −24.4. A hundred and twenty Elo separates
//! the same constant on two comparable sets of wonders. The result is about
//! extra turns specifically, not about the shape or magnitude of the term.
//!
//! ## Against the ladder
//!
//! 600 games per seed range at `Nodes(2000)` for the searching opponent — the
//! budget kind still cannot matter for `phased`, which ignores it, but it is
//! what decides how strong the opponent is:
//!
//! ```text
//!                        this agent          phased:base=v5
//! vs alphabeta   seed 1   -83.7 [±28.6]      -122.8 [±29.5]
//!            seed 100001  -84.9 [±28.6]      -153.8 [±30.6]
//! vs mcts-uct    seed 1  -217.7 [±33.4]      -300.5 [±38.8]
//!            seed 100001 -214.4 [±33.2]      -310.9 [±39.7]
//! ```
//!
//! `matchup_profile`'s win-condition spread says it is not one route: against
//! `alphabeta` the wins go 165 → 185 civilian *and* 29 → 40 scientific, and
//! against `mcts-uct` 57 → 91 civilian and 32 → 41 scientific. Both routes
//! improve on both ranges.
//!
//! ## The behaviour actually changed — and not in the obvious place
//!
//! `duels-arena/examples/wonder_audit.rs` now breaks out the five play-again
//! wonders and counts the decisions each side actually took, and how many of
//! them were extra turns. The first run of it was a **self-play** pair, which
//! showed the premium building play-again wonders *less* often (94% → 88%) and
//! *later* (turn 29.7 → 31.4) — the exact opposite of the intent, and worth
//! recording because it is what a self-play audit of a *draft* preference
//! always shows: which wonders are in play is fixed by the deal, and when both
//! sides share the preference the only thing it can do is cancel.
//!
//! Run asymmetrically, 1000 games at `Nodes(1)`, the mechanism is unambiguous:
//!
//! ```text
//!                                    phased (premium 9)   phased:base=v5
//! play-again wonders drafted               2100                1250
//! play-again wonders built                 1882                1170
//! ...as a share of the ones drafted         90%                 94%
//! decisions taken per game                34.87               34.06
//! ...of which extra turns                  4.22                3.45
//! wonders left drafted and unbuilt          0.56                0.68
//! ```
//!
//! The premium is a **draft** signal, not a build signal. It wins the split of
//! the eight dealt wonders: 63% of the play-again wonders in play end up on
//! this side against 37%, and that converts into **22% more extra turns
//! actually taken** and eight tenths of a decision more per game. The build
//! *rate* falls slightly because the side now holding five play-again wonders
//! runs into the seven-slot cap that the side holding two never reaches — and
//! it still leaves fewer wonders unbuilt in absolute terms.
//!
//! ## Cost
//!
//! One flag test and one addition on a constant, per unbuilt wonder, on a
//! branch that is not even taken at the default weight of every other wonder.
//! `examples/decision_cost.rs`, every configuration timed on the same 2157
//! positions:
//!
//! ```text
//! v5 (the round-five agent)             63.6 us/decision
//! default, extra-turn premium off       63.0 us/decision
//! default (round six)                   63.3 us/decision   +0.5%, and v5 reads higher
//! ```
//!
//! Three timings of what is arithmetically the same work plus one `f64`
//! addition, spread over 0.6 us — the premium is free, and the row ordering
//! (round five reading *slowest* of the three) is the benchmark's run-to-run
//! noise rather than a cost. Nothing is root-fixed for it because there is
//! nothing positional to fix: `play_again` is a property of the wonder, not of
//! the position, so there is no per-decision table for it to live in.
//!
//! # Round seven: the forward-looking terms were collectively over-priced
//!
//! Round seven is **one new gate, five re-weightings, four honest negatives
//! and two new measurement instruments**. All of it is reproduced bit for bit
//! by [`Config::v6`] (`tests/v6_identity.rs`).
//!
//! ## The brief, and the honest answer to it
//!
//! The brief was to improve the leaf evaluation, touching nothing outside this
//! crate, and to raise `duels-agent-mcts-eval`'s Elo by **at least +200**
//! against a fixed `mcts-uct` anchor. **The achieved figure is +1.2 Elo, 95%
//! interval [-7.6, +10.0], over 12,800 games a side on four disjoint seed
//! ranges** — which is to say no measurable change at all, and nowhere near
//! the target.
//!
//! That is the round's most useful output, because it is not a statement about
//! this change set. The same change set is worth **+149 / +166 Elo to
//! `phased`**, **+122 / +126 against `alphabeta`** and **+87 / +101 against
//! `mcts-uct`** — the largest movement any round of this crate has produced
//! against an unrelated opponent, and the first time `phased` has beaten
//! `alphabeta` at `Nodes(2000)`. **None of it reaches `mcts-eval`.** A leaf
//! that is half playout, blended at `weight = 0.5` with an exploration
//! constant tuned for that blend, extracts what it is going to extract from a
//! hand-crafted evaluation, and this crate is no longer the binding
//! constraint on it. The levers that could change that — the blend weight, the
//! exploration constant, a `Root` rebuilt deeper in the tree — all live in
//! `mcts-eval`, which this round was not allowed to touch and which is
//! where a +200 would have to come from.
//!
//! ## Two instruments this crate did not have
//!
//! Every previous round could only sweep a knob that
//! `duels_arena::agent_spec::parse_phased_config` had a key for, which put the
//! instrument for measuring a `duels-eval` change in a crate a `duels-eval`
//! round is not supposed to touch. Round seven added two `examples/` binaries
//! instead, and they are why it could sweep as widely as it did.
//!
//! **`examples/head_to_head.rs`** is the arena's `phased`-versus-`phased`
//! match reproduced inside this crate over two `Config` values given on the
//! command line: same paired seat-swap, same salts, same `PhasedAgent::choose`,
//! same Bradley-Terry fit. It agrees with the arena **to the game**:
//! `phased:science_ladder=0.2` against `phased` over 3200 games at seed 1
//! reads `+60.2 [+48.0, +72.4]` through `duels-arena` and `+60.2 [+48.0,
//! +72.4]` through this example, 1874-1325-1. It also runs a 3200-game match
//! in about three seconds, which is what made a five-round coordinate descent
//! affordable.
//!
//! **`examples/leaf_probe.rs`** scores a *fixed* corpus of labelled positions
//! under many configurations at once — mean negative log-likelihood at each
//! configuration's own maximum-likelihood temperature, plus the
//! temperature-free sign accuracy. `examples/calibrate.rs` cannot do this: it
//! fits one configuration over that configuration's own self-play, so changing
//! the weights changes the questions. See "a better predictor is a worse leaf"
//! below for what the probe then measured, which is not what it was built to
//! find.
//!
//! ## 1. The science ladder was being paid for a race that was already over
//!
//! [`terms::science_ladder`]'s rung is steeply convex — 6, 12 and 18 victory
//! points at three, four and five distinct symbols — and the convexity has
//! exactly one justification: six symbols win the game outright. The rung was
//! collected whether or not a sixth symbol was still *in* the game. A player
//! sitting on four symbols whose two missing ones were both buried in the
//! opponent's city collected twelve victory points for a race that could not
//! be run.
//!
//! [`ScienceWeights::dead_race_scale`] (**`0.0` by default**) multiplies the
//! rung — and only the rung, never `pair_threat`, since a half-pair is still a
//! progress token — once [`terms::supremacy_reachable`] says the player can no
//! longer assemble [`terms::SYMBOLS_TO_WIN`] distinct symbols, counted off the
//! card data through the same public-information test
//! [`terms::second_copy_obtainable`] already applies to a second copy.
//!
//! It is worth **+32.0 / +24.8** Elo on its own against `phased:base=v6` over
//! 3200 games on each of two disjoint seed ranges, and — the reason it is a
//! gate and not a smaller number — **it keeps the scientific-supremacy route
//! intact**, 44 and 51 science wins in 3200 games against round six's 39 and
//! 45. That is the whole argument for it, because the cheap alternative is
//! better on the aggregate and much worse on the route: driving
//! [`EvalWeights::science_ladder`] to `0.2` with nothing else changed is worth
//! **+61.6 / +58.3** and takes the science wins to **3**.
//!
//! The shipped default is both: the gate, plus the ladder at `0.5` and
//! [`ScienceWeights::pair_threat_weight`] at `0.5`. Each field's own comment
//! carries its sweep.
//!
//! ## 2. Chain equity, and the fitted weight that was covering for it
//!
//! With the ladder corrected, two weights that had been measured as
//! "indistinguishable from zero, kept on the pooled result" and "fitted, not
//! derived, and flagged as such" both turned out to be badly wrong, in the
//! same direction, for what is probably the same reason.
//!
//! * [`EvalWeights::chain_equity`] `1.0` → **`0.25`**: the largest single
//!   re-weighting of the round, +71 / +82 Elo.
//! * [`EvalWeights::resource_bill`] `3.0` → **`1.0`**: +53 / +60 Elo, and a
//!   *reversal* — round two fitted `3.0` over the derived `1.0` and wrote down
//!   that it was "a measurement, not an argument". The derived rate now wins
//!   by a wide margin on a monotone curve. **The fitted weight was
//!   compensating for two other over-priced terms.**
//! * [`EvalWeights::development`] `1/3` → **`0.2`**: +4 / +14 Elo.
//!
//! The pattern is the round's one-line summary and is worth keeping as a prior:
//! **this evaluation's forward-looking terms were collectively over-priced**,
//! each of them fitted against a baseline that contained the others, and
//! correcting the largest one moved the honest value of the rest a long way.
//!
//! ## Elo, measured one change at a time
//!
//! Against `phased:base=v6` at `Nodes(1)`, **3200 games per seed range**, as a
//! leave-one-out against the round-seven default — so each row is a paired
//! head-to-head of exactly that one change. Every interval is ±12.2.
//!
//! ```text
//!                                     seed 1   seed 5001   the change is worth
//! the round-seven default             +149.2     +165.6
//! chain_equity back to 1.0             +77.8      +83.6      +71 / +82
//! resource_bill back to 3.0            +96.0     +106.0      +53 / +60
//! science_ladder back to 1.0          +124.0     +133.5      +25 / +32
//! pair_threat_weight back to 1.0      +138.6     +143.7      +11 / +22
//! the dead-race gate switched off     +142.5     +153.3       +7 / +12
//! development back to 1/3             +145.5     +151.8       +4 / +14
//!
//! owned-token equity switched on      +147.0     +160.7       -2 /  -5
//! the count-priced menu switched on   +153.1     +165.8       +4 /  +0
//! to_move = 2                         +128.5     +149.3      -21 / -16
//! to_move = 6                         +106.7     +135.5      -43 / -30
//! to_move = 12                        +104.5     +130.0      -45 / -36
//! ```
//!
//! The six accepted rows sum to +171 and +217 while the whole default is worth
//! +149 and +166, and that is the point rather than an inconsistency: every one
//! of them was fitted, in an earlier round, against a baseline that contained
//! the others.
//!
//! The default is then confirmed on **four further disjoint ranges it was not
//! tuned on** — **+150.4**, **+161.2**, **+157.6** and **+145.7** at seeds
//! 9001, 13001, 17001 and 21001, 3200 games each — so the five rounds of
//! coordinate descent behind it are not two seed ranges' worth of noise.
//!
//! ## Against the ladder, including two unrelated searchers
//!
//! 800 games per seed range; the 1-ply opponents at `Nodes(1)` and the
//! searchers at `Nodes(2000)`.
//!
//! ```text
//!                                 the new default        phased:base=v6
//! vs random        seed 1          798-2                  800-0
//! vs greedy        seed 1          800-0                  796-4
//! vs greedy-ev     seed 1          800-0                  797-3
//! vs strategist    seed 1          800-0                  794-6
//! vs alphabeta     seed 1        +54.2 [+29.9, +78.6]   -67.6 [-92.2, -43.1]
//!                  seed 5001     +35.7 [+11.5, +59.9]   -90.5 [-115.4, -65.6]
//! vs mcts-uct      seed 1       -134.3 [-160.1, -108.4] -221.4 [-250.5, -192.3]
//!                  seed 5001    -120.9 [-146.4, -95.4]  -222.1 [-251.2, -192.9]
//! ```
//!
//! **+122 / +126 Elo against `alphabeta` and +87 / +101 against `mcts-uct`**,
//! agreeing on both ranges against two unrelated searchers, so the gain is not
//! self-play overfitting. `phased` now **beats `alphabeta` at `Nodes(2000)`**,
//! which no generation of this crate has done before.
//!
//! ## The behaviour actually changed
//!
//! `duels-arena/examples/matchup_profile.rs`, 400 games against `mcts-uct` at
//! `Nodes(2000)` and seed 1, is the check that a win rate cannot make: did the
//! agent start playing differently, or did it get luckier?
//!
//! ```text
//!                                   phased:base=v6    the new default
//! wins                                 91 / 400          128 / 400
//!   by military supremacy                 2                  2
//!   by scientific supremacy              30                  8
//!   civilian                             58                118
//! green cards per game                  3.9                1.6
//! yellow cards per game                 4.2                5.7
//! blue cards per game                   3.8                4.3
//! distinct symbols reached             3.54               1.25
//! guilds built (of 400 games)           321                393
//! guild victory points per game        4.82               6.71
//! victory-point margin in its losses   -7.22              -2.58
//! ```
//!
//! Two readings, and the second is the one that matters. The obvious one is
//! that the round traded the science lottery for civilian points: thirty
//! supremacy wins become eight, fifty-eight civilian wins become a hundred and
//! eighteen, and the city goes from four green cards to one and a half. The
//! useful one is the bottom row — **the losses got much closer, −2.58 points
//! against −7.22**. An agent that was entering a race it usually lost and then
//! losing the rest of the game by seven points is now losing by two and a half,
//! which is what a re-priced evaluation looks like from the inside and is not
//! something a win rate would have shown.
//!
//! The military column is unchanged at 2 wins in 400, which is the one
//! dimension of the profile round seven neither improved nor damaged, and the
//! one `mcts-uct` still uses against it (72 military-supremacy wins before, 65
//! after).
//!
//! ## The finding that matters most: a better *predictor* is a worse *leaf*
//!
//! `examples/leaf_probe.rs` was built on the reasoning that `phased` consumes
//! this crate as a *policy* (an argmax, scale-free) while `mcts-eval` consumes
//! it as a *value* (a fitted logistic averaged into a win rate), so the
//! objective that matters for a leaf is how well the number predicts the
//! winner. That reasoning is sound and the conclusion it leads to is **wrong**,
//! which is the most useful thing this round found.
//!
//! A four-weight variant — `military_band = 3.0`, `vp_projection = 1.6`,
//! `development = 0.16`, `yellow_equity = 2.0` — is a *substantially* better
//! predictor than either round six or the round-seven default, reproduced on a
//! disjoint corpus:
//!
//! ```text
//!                    train (43k positions)      validate (disjoint, 46k)
//!                  T     NLL    sign  sgn-III     T     NLL    sign  sgn-III
//! round six      34.5  0.5982  0.6721  0.7247   25.1  0.5508  0.7036  0.7474
//! round seven    23.1  0.6136  0.6730  0.7408   16.6  0.5741  0.6985  0.7619
//! the predictor  14.0  0.5649  0.6981  0.7847   10.8  0.5210  0.7256  0.8197
//! ```
//!
//! The predictor is better on every column on both corpora — two to three
//! points of sign accuracy overall and four to seven in Age III — and as an
//! `mcts-eval` leaf value it is worth **+64.9** against the anchor, against
//! **+94.8** for the intermediate round-seven bundle it was built on top of
//! (the science gate and the ladder, without the chain-equity, bill and
//! development corrections) and **+89.3** for round six, all at 3200 games and
//! seed 1. The victory kinds say why: military wins go
//! 236 → 494 and civilian collapses 1703 → 1368. Military standing predicts
//! the winner very well *and* steers a search into races it then loses, which
//! is this project's oldest finding — "win-condition awareness belongs in the
//! search policy, not the evaluation function" — arriving from a new direction.
//!
//! The middle row makes the point sharper still, and it is the shipped
//! default: **round seven is a *worse* predictor than round six** — a tenth of
//! a nat of likelihood worse on both corpora, with sign accuracy flat — while
//! being +150 Elo stronger as a policy and, as a leaf, no worse. On this
//! corpus, over these two objectives, the correlation is not merely weak;
//! across the three rows it points the wrong way.
//!
//! **So the probe is a screen for a hypothesis, not a proxy for leaf quality.**
//! The instrument that did predict `mcts-eval`'s direction was the boring one:
//! `phased` Elo, attenuated. Later rounds should treat it that way.
//!
//! ## `mcts-eval` against the fixed `mcts-uct` anchor
//!
//! `mcts-eval` reads [`Config::default`] live and pins no generation (that is
//! deliberate; see its crate docs), so "old against new" is not a single-binary
//! match. The measurement is therefore indirect, against an anchor that does
//! not move: `mcts-uct` no longer depends on this crate at all, so the same
//! `mcts-uct` is on the other side of every row below and the **difference of
//! the two Elo-vs-anchor columns is the achieved gain**. Both columns were
//! measured with a binary built from the same tree, differing only in what
//! `Config::default()` returns.
//!
//! ```text
//! Nodes(2000), 3200 games per seed range, paired and seat-swapped
//!                   round six        round seven        the round is worth
//! seed 1          +89.3 +-12.4      +93.9 +-12.5              +4.7
//! seed 5001       +89.4 +-12.4     +104.4 +-12.6             +15.0
//! seed 9001       +94.6 +-12.5      +83.1 +-12.4             -11.6
//! seed 13001      +84.9 +-12.4      +82.0 +-12.4              -2.9
//! pooled (12800)  +89.6 +- 6.2      +90.8 +- 6.2       +1.2 [-7.6, +10.0]
//!
//! TimeMs(20), 400 games per seed range, RAYON_NUM_THREADS=1, one at a time
//! seed 1          +54.2 +-34.4      +97.8 +-35.4             +43.6
//! seed 5001       +88.5 +-35.1      +55.9 +-34.4             -32.6
//! pooled (800)    +71.3 +-24.6      +76.7 +-24.6      +5.4 [-29.4, +40.2]
//! ```
//!
//! **+1.2 Elo, on an interval that comfortably contains zero, over twelve
//! thousand eight hundred games a side.** Two ranges up, two down, and the two
//! wall-clock ranges disagree with each other as well. The honest reading is
//! not "a small gain" but **"no measurable change"**: round seven is worth
//! about +150 Elo to `phased`, +122 against `alphabeta` and +95 against
//! `mcts-uct`, and none of it reaches `mcts-eval`.
//!
//! Two things stop that being a statement about measurement noise. The
//! four-range protocol is what caught it — at the two ranges this round was
//! tuned on it reads +4.7 and +15.0, and a round that stopped there would have
//! reported a gain that the next two ranges erase. And `mcts-eval` is
//! demonstrably *not* insensitive to this crate in general: the predictor
//! variant above, a change of comparable size in the other direction, costs it
//! **−29.0 [±17.5]** at seed 1. The leaf can be broken from here. It cannot,
//! at `Nodes(2000)` and `weight = 0.5`, be much improved from here.
//!
//! The win-condition breakdown says the same thing from the other side. Round
//! seven moves `mcts-eval`'s own profile a long way — military wins 236 → 324
//! and 259 → 314 on the first two ranges, scientific 43 → 25 and 46 → 28 —
//! while the totals stay put. It is playing differently and winning as often.
//!
//! ## Four honest negatives
//!
//! **1. [`EvalWeights::token_equity`] (default `0.0`).** Six of the ten
//! progress tokens are rules changes that pay out over the remaining game —
//! Theology, Economy, Strategy, Architecture, Masonry, Urbanism — and this
//! evaluation priced **none** of them, which also meant that
//! [`PendingModel::Completed`], the code that *chooses* a token when a science
//! pair completes, was choosing between them on printed victory points alone.
//! [`terms::TokenTable`] prices all six, each channel a quantity the evaluation
//! already computes (Economy's is literally [`terms::resource_bill`] read from
//! the other end). It measures at **+7.1 / +8.0** Elo at `0.5` and **+3.5 /
//! +15.0** at `1.0` against `phased:base=v6`, and then at **−2 / −5** as a
//! leave-one-out against the finished round-seven default, which is the
//! comparison that decides it. A real gap, correctly filled, worth nothing
//! once the terms it competes with are priced properly.
//!
//! **2. [`CountPricing::Counted`] (default off).** Five Age III commercial
//! cards print no coins and instead pay a count of the builder's own city, so
//! [`menu::TakeValue::free_value`] — which starts from `def.coins` — could not
//! see nine coins on a Chamber of Commerce. Exactly the shape of the guild bug
//! round five fixed, and unlike that one it is a count already on the table
//! rather than a projection. **+5.9 / −2.3** Elo against `phased:base=v6`, and
//! **+3.9 / +0.2** as a leave-one-out against the round-seven default. Neutral
//! on both readings and with the signs disagreeing on one of them, so it stays
//! off, per this project's rule about not moving a default on a neutral
//! result. It is the more correct model and it is available as an option with
//! the measurement written down.
//!
//! **3. [`EvalWeights::to_move`] (default `0.0`).** The right to move is worth
//! a great deal in this game (`CLAUDE.md` records ~67/33 between equal
//! `mcts-uct` configurations) and an extra turn is the only thing that
//! re-assigns the remaining slots. Nothing priced either. Two things came out
//! of trying: first, a term reading `GameState::extra_turn` is **exactly zero
//! at every position anything ever scores** — `engine::finish_turn` consumes
//! the flag the instant it would matter — measured over fifty thousand real
//! positions before the cause was found, and the reason
//! [`terms::to_move`] reads `current_player` instead. Second, it costs Elo:
//! `+2` is worth −21 / −16, `+6` −43 / −30 and `+12` −45 / −36 as a
//! leave-one-out against the round-seven default. The value objective *likes*
//! it, at one or two victory points; the policy objective does not, which is
//! the same divergence as the predictor above.
//!
//! **4. [`EvalWeights::value_scale`] (default `1.0`).** The one knob a
//! `duels-eval` round has that a search can see and `phased` cannot: scaling
//! this crate's output by `k` divides `mcts-eval`'s fitted leaf temperature by
//! `k`. The maximum-likelihood calibration turns out to be about right —
//! measured on the same intermediate bundle as the predictor above, `k = 2.0`
//! reads **+84.7** and `k = 0.6` reads **+74.4** against the anchor where
//! `k = 1.0` reads **+94.8**, all at 3200 games and seed 1. Worth having
//! measured, because "the calibration `calibrate.rs` fits is also the
//! calibration the search wants" was an assumption and is now a measurement.
//! It is also the knob to re-check first if a future round moves the output
//! scale a long way: round seven took the maximum-likelihood temperature over
//! this corpus from 34.5 to 23.1 victory points while `mcts-eval`'s fitted
//! constants stayed where they were, and `k` is how a `duels-eval` round would
//! compensate for that without touching a search.
//!
//! ## The `yellow_equity` mystery is still open, and is now stranger
//!
//! Round five's follow-up note flagged [`EvalWeights::yellow_equity`] as the
//! result to be most suspicious of: the term needed four times the weight its
//! own stated mechanism implies, and the working theory was that it was a proxy
//! for a different, unidentified mispricing of commercial cards. Round seven
//! corrected four genuinely mispriced terms and re-swept it, and `4.0` is
//! **still** on the plateau: 3.0 reads +139.1 / +154.3, 4.0 (the default)
//! +149.2 / +165.6, 4.5 +147.5 / +165.8, 5.0 +150.8 / +157.9, 5.5 +152.5 /
//! +155.2 and 6.0 +150.5 / +148.4 against `phased:base=v6` over 3200 games on
//! each of two disjoint seed ranges — flat from 4 to 5.5 and falling below 4,
//! which is where round five left it.
//!
//! Two candidate mechanisms were ruled *out* along the way. It is not the
//! menu's blindness to the count-scaled Age III commercial cards — that is
//! [`CountPricing`] above, and pricing it is worth nothing. It is not the
//! coins-to-points rate, which round five had already ruled out. What remains
//! untested is the control round five named and did not build: a **flat**
//! per-yellow bonus, with `decisions_left` removed, which would say whether
//! the term is pricing discard yield at all or is pricing something that merely
//! correlates with holding commercial cards early. That needs a second knob and
//! is the obvious follow-up.
//!
//! ## Cost
//!
//! `examples/eval_bench.rs`, every configuration timed on the same 2151
//! positions. The per-**leaf** number is the one that matters for this round,
//! because `evaluate` is what a search calls tens of thousands of times per
//! decision while `Root::new` is called once per tree node:
//!
//! ```text
//!                                     Root::new    evaluate       sum
//! v1 (the round-one evaluation)         1.819 us    0.220 us   2.039 us
//! v5 (the round-five evaluation)        3.467 us    0.455 us   3.922 us
//! v6 (the round-six evaluation)         3.462 us    0.457 us   3.919 us
//! default (round seven)                 3.488 us    0.470 us   3.958 us
//! default + owned-token equity          3.674 us    0.469 us   4.143 us
//! default + the count-priced menu       3.508 us    0.468 us   3.976 us
//! ```
//!
//! **Round seven is free**, at +0.013 us per leaf and +0.026 us per node,
//! which is inside the benchmark's run-to-run spread. It was not free when
//! first written: the dead-race gate's reachability walk read
//! **0.720 us** per `evaluate`, a **+47%** regression on the one number a leaf
//! value cannot afford to regress. Two guards fixed it and are in the code for
//! that reason — the walk is skipped entirely when the ladder rung is zero
//! (a player holding no symbols cannot care whether the race is alive), and
//! [`terms::supremacy_live`] stops at the *second* unreachable symbol, since
//! seven symbols exist and six win. The honest count,
//! [`terms::supremacy_reachable`], is kept for the tests and the diagnostics
//! and is not on the hot path.
//!
//! # Round eight: the calibration was stale, and the ladder stopped one rung
//! too early
//!
//! Round eight is **one refitted calibration, one re-shaped ladder, one new
//! instrument and one hypothesis that the instrument did not support**. All of
//! it is reproduced bit for bit by [`Config::v7`] (`tests/v7_identity.rs`).
//!
//! ## The brief, and the honest answer to it
//!
//! The brief was to improve the leaf evaluation with a focus on **science**,
//! against the project owner's read — offered explicitly as rough guidance
//! rather than a target — that three or four distinct symbols in Age I should
//! often be worth something like an 80-90% win rate, and four distinct symbols
//! in Age II almost always at least 70%.
//!
//! **Those magnitudes are not supported.** Measured over real games rather than
//! argued about, a player who first reaches three distinct symbols in Age I
//! goes on to win **45.9%** of the time (n = 270) and one who first reaches
//! four in Age II wins **55.7%** (n = 461). Four symbols in Age I reads 63.2%
//! on nineteen samples and five in Age II — one symbol from an outright win —
//! reads 78.3% on sixty-nine. A science position is a *good* position, not a
//! won one.
//!
//! **The direction of the evaluation's error at the top of the ladder is real,
//! though, and it is large.** Round seven's evaluation called that
//! four-symbol Age II position a 44.9% loss where it wins 55.7%, and the
//! five-symbol one 45.5% where it wins 78.3%. So the round's answer to the
//! brief is: the benchmark numbers are too optimistic, the intuition behind
//! them is not — and the fix is at four and five symbols, which is exactly
//! where `phased` never goes and a search does.
//!
//! ## The instrument: `examples/science_calibration.rs`
//!
//! `examples/calibrate.rs` fits **one** temperature over **all** positions, so
//! a term can be badly mispriced in one corner of the state space while the
//! aggregate stays honest, because the corner is a small share of the corpus.
//! Science is precisely such a corner.
//!
//! The new example replays self-play games the same way `calibrate.rs` does —
//! the same verbatim `PhasedAgent::choose`, the same `seed ^ 0xF00D` engine
//! stream, both seat orders — and at every decision records, **for each
//! player**, their distinct symbol count, [`win_probability`]'s claim about
//! them, and whether they went on to win. One [`Root`] serves both players,
//! which is exact rather than approximate: [`evaluate`] is antisymmetric, so
//! the two probabilities sum to one.
//!
//! Two things about it are worth copying into any future diagnostic here.
//!
//! **The configuration a position is *read* with is separate from the ones the
//! seats *play* with** (`--read`, against the two positional arguments). The
//! first draft let the reading config follow the seat, which compares each
//! evaluation against its own games and answers a different question; every
//! number below is one fixed evaluation read over one fixed corpus.
//!
//! **The default agent will not visit the interesting positions.** Round seven
//! traded the science lottery for civilian points and says so — `phased`
//! self-play reaches four distinct symbols in Age II about forty times in three
//! thousand games. So the corpus below is generated by two *science-tilted*
//! seats (`phased:base=v7,sci=3.0`, the round-seven evaluation with the ladder
//! weight at six times its default), which is off-policy and is the honest
//! trade: both seats carry the same tilt, so the corpus is symmetric and an
//! unconditional win rate is still 50%, and the buckets that matter carry
//! hundreds to thousands of samples instead of tens.
//!
//! ## 1. The win-probability calibration was stale by a factor of two
//!
//! [`win_probability_temperature`]'s constants were fitted against round
//! **six**'s evaluation and left untouched when round seven re-weighted five
//! terms — a staleness round seven documented as known and bounded. It is
//! neither: refitting `examples/calibrate.rs` against the round-seven default
//! moves every one of them by 40-55%.
//!
//! ```text
//!                  Age I    Age II   Age III   overall
//! shipped (round six)   47.57    43.75     25.18     38.61
//! refit, 3000 games     26.75    19.98     14.62     19.66   seeds 0..1500
//!                       26.33    18.16     16.04     19.44   seeds 5001..
//!                       26.38    20.43     16.86     20.65   seeds 9001..
//! refit, 8990 games     26.40    20.34     15.50     20.12   <- the new constants
//! ```
//!
//! Three disjoint 1500-seed ranges agree to within a victory point in Age I and
//! two elsewhere; the shipped numbers are the 8,990-game fit over 643,875
//! positions. `examples/calibrate.rs` grew a second argument — a first seed —
//! for exactly this: a fitted constant is measured, and this project reproduces
//! anything measured on a disjoint range before believing it.
//!
//! An evaluation whose temperature is nearly twice what it should be is one
//! whose every judgement is dragged towards `p = 0.5`. That is what made the
//! science buckets look mildly wrong rather than badly wrong, and it is why the
//! round's first change is not about science at all.
//!
//! ## 2. Two temperatures, because two consumers want different numbers
//!
//! [`EvalWeights::win_probability_temperature`] is a new field, and the reason
//! it is a field rather than a constant is that **the number a diagnostic wants
//! and the number a search wants are not obliged to be the same** — round
//! seven had already found that, from the other direction, when its
//! [`EvalWeights::value_scale`] sweep measured an exact halving of the
//! temperature as *worse* for `mcts-eval`.
//!
//! Making it a field is also what makes the question answerable. `mcts-eval`
//! pins a whole [`Config`] under the arena's `eval=vN` key, and a free constant
//! is invisible to that pin, so before round eight a change to the leaf
//! mapping could not have been A/B tested against the generation before it at
//! all.
//!
//! With that in place the answer turned out to be the opposite of round
//! seven's: **the honest refit is worth about +19.5 Elo to `mcts-eval`**, and
//! the module constants and the default field value now agree. They are kept
//! separable because the next round may find they should not.
//!
//! ## 3. The ladder's top two rungs
//!
//! With the temperature corrected, the bucket table is close to exact from zero
//! to three symbols and badly wrong above. One fixed corpus (2,000 games,
//! 283,572 player-positions), one row per `(age, symbols)` bucket, at the
//! position where a player **first** reaches that count:
//!
//! ```text
//!                              round seven's read      round eight's read
//! age  sym      n      actual   predicted    gap     predicted    gap
//!  1    2     1428      0.417     0.451    -0.034      0.418    -0.001
//!  1    3      270      0.459     0.436    +0.024      0.394    +0.065
//!  2    3     1290      0.443     0.478    -0.035      0.469    -0.026
//!  2    4      461      0.557     0.449    +0.108      0.554    +0.003
//!  2    5       69      0.783     0.455    +0.328      0.736    +0.047
//!  3    4     1250      0.470     0.451    +0.019      0.496    -0.026
//!  3    5      586      0.565     0.410    +0.154      0.653    -0.088
//! ```
//!
//! The same thing said in victory points, which is the actionable form: hold
//! the evaluation and the temperature fixed and fit **one additive correction
//! `δ_k` per symbol count** by maximum likelihood, on
//! `P(win) = σ((evaluate + δ_{sym(me)} − δ_{sym(opp)}) / T(age))` with
//! `δ_0 ≡ 0`. `δ_k` is then exactly the victory points that would have to be
//! *added* at `k` symbols for the evaluation to predict what actually happens.
//!
//! ```text
//!            round seven's read              round eight's read
//! age    d1    d2    d3    d4    d5      d1    d2    d3    d4    d5
//!  1   -0.3  -2.2  +6.0  +6.9    —     -0.3  -2.3  +5.7  -7.5    —
//!  2   +0.2  -5.0  -7.5 +13.4 +25.6    +0.2  -5.0  -8.9  -1.8  -6.2
//!  3   +0.2  -2.1  -4.5  -2.6  +8.0    +0.0  -2.7  -7.0  -9.6 -14.7
//! ```
//!
//! **`+13.4` and `+25.6` victory points** unclaimed at four and five symbols in
//! Age II, on 1,805 and 187 mover positions. The rung there was `12 × 0.5 = 6`
//! and `18 × 0.5 = 9` victory points, so the evaluation was crediting a third
//! of what the position is worth. [`ScienceWeights::ladder`]'s top two entries
//! become **`30` and `54`**; the first four are untouched, because the fit says
//! they are within a couple of points of right.
//!
//! ## Elo, measured one change at a time
//!
//! `mcts-eval` against `mcts-eval:eval=v7` — the same binary either side,
//! differing only in what [`Config::default`] returns — at `Nodes(2000)`,
//! 3200 games per seed range, paired and seat-swapped:
//!
//! ```text
//!                         seed 1  seed 5001  seed 9001  seed 13001   pooled (12800)
//! the temperature refit    +15.3     +19.0      +20.6      +22.9     +19.5 +- 6.0
//! ...and the top rungs     +20.3     +22.3      +20.0      +22.5     +20.9 +- 6.0
//! so the rungs are worth    +5.0      +3.3       -0.6       -0.4      +1.4
//! ```
//!
//! **The temperature refit is the round.** It is positive on all four ranges
//! with every interval clearing zero, and it is the largest single effect this
//! crate has produced for `mcts-eval` since the agent was created.
//!
//! **The ladder change is Elo-neutral and is adopted anyway**, which is a
//! deliberate call with a precedent in this crate: round three adopted
//! [`rails`] on its audit rather than on its Elo, for the same reason. Three
//! things say it belongs in:
//!
//! * it removes a **25-victory-point** error in the leaf value at exactly the
//!   positions the leaf is consulted about a science race, which is a defect
//!   whether or not two agents that share it can exploit each other over it;
//! * it costs nothing (+1.4 pooled, and `phased`, which is blind to the
//!   temperature and so measures the rungs alone, reads +0.7 / +2.1 / −2.4 /
//!   −1.3 over 3200 games on each of four ranges against a null calibration of
//!   +2.6);
//! * and it **restores the scientific-supremacy route**, consistently. Against
//!   the pinned round-seven anchor this side's science wins go 29 → 42,
//!   33 → 44, 26 → 39 and 31 → 49 on the four ranges; in `phased` self-play,
//!   8/17/16/15 against 4/11/8/4. Round seven kept `science_ladder = 0.5` over
//!   an Elo-better `0.2` for precisely this reason, and this is the same
//!   argument one rung further up.
//!
//! The rung magnitudes were swept, on top of the temperature refit, at seed 1:
//! `12/18` (round seven's) +15.3, `20/34` +18.8, `30/54` +20.3, `45/80` +17.9.
//! A shallow unimodal curve; `30/54` is its middle and is also roughly where
//! the fitted `δ` puts it.
//!
//! ## Against an unrelated opponent, and at a wall-clock budget
//!
//! A head-to-head gain between two configurations of the same agent is worth
//! little until it shows up against something that is not that agent.
//! `mcts-uct` does not consume this crate at all, so the same `mcts-uct` is on
//! the other side of both rows — `Nodes(2000)`, seed 1:
//!
//! ```text
//!                          1600 games              6400 games
//! round eight        +103.4 [+85.6, +121.2]   +98.7 [+89.8, +107.6]
//! round seven        +103.2 [+85.4, +121.0]   +84.2 [+75.5,  +93.0]
//! ```
//!
//! **The left column is a warning and the right column is the result.** At
//! 1,600 games a side the two are indistinguishable and the round looks like a
//! self-play artefact; at 6,400 the difference is **+14.5** on intervals that
//! barely touch. A 20-Elo effect simply cannot be resolved at 64% win rate over
//! 1,600 games, and a check that stopped at the left column would have reported
//! a false negative as confidently as round seven's four-range protocol caught
//! a false positive. The victory kinds say it is the same effect: against
//! `mcts-uct`, round eight wins 109 games by scientific supremacy against round
//! seven's 48, and 3,418 civilian against 3,238.
//!
//! At a wall-clock budget (`TimeMs(20)`, `RAYON_NUM_THREADS=1`, one match at a
//! time on a quiet machine, 400 games per range) `mcts-eval` against
//! `mcts-eval:eval=v7` reads **+12.1 [−21.9, +46.2]** at seed 1 and **+43.5
//! [+9.3, +77.8]** at seed 5001 — **+27.8 ± 24.1 pooled over 800 games**, which
//! agrees with the `Nodes` figure and is what this project's two-budget
//! discipline asks for.
//!
//! ## The ladder does not regress
//!
//! 800 games at seed 1; the 1-ply opponents at `Nodes(1)` and the searchers at
//! `Nodes(2000)`:
//!
//! ```text
//! phased      vs alphabeta    +53.3 [+29.0, +77.7]   (round seven: +54.2 [+29.9, +78.6])
//!             vs greedy-ev    800-0
//!             vs random       798-2
//! mcts-eval   vs alphabeta   +279.6 [+247.3, +311.9]
//!             vs greedy-ev    800-0
//! mcts-eval:eval=v7
//!             vs alphabeta   +282.7 [+250.3, +315.2]
//! ```
//!
//! `phased` is unchanged against `alphabeta` to well inside the interval, which
//! is the expected result: `phased` cannot see the temperature at all, and the
//! rungs it can see are at symbol counts a 1-ply agent almost never reaches.
//! Against `alphabeta`, `mcts-eval` is level either way — the +19.5 shows up
//! against `mcts-uct` and in self-play and not here, which is worth recording
//! rather than smoothing over.
//!
//! ## What is left, and what to do about it
//!
//! **The ladder is age-blind and the data says it should not be.** Look along
//! the round-eight `δ` rows above: Age II is now flat to within a couple of
//! victory points, and Age III reads `−9.6` and `−14.7` at four and five
//! symbols — the same rung that is right in Age II over-credits in Age III. It
//! is not hard to see why. The rung is convex because six symbols win the game,
//! and how much of the game is left to find a sixth symbol in is exactly what
//! an age is. Round eight ships one rung for all three ages because that is the
//! change the measurement supports; an age-scaled rung — or, better, a rung
//! scaled by `decisions_left` — is the obvious follow-up, and
//! `examples/science_calibration.rs` is the instrument that would judge it.
//!
//! **The two temperatures should be re-checked together.** They agree today by
//! measurement rather than by construction, and a round that moves this crate's
//! output scale a long way should re-run `examples/calibrate.rs` *and* re-run
//! the `eval=vN` A/B, rather than assuming the refit is automatically the right
//! leaf.
//!
//! ## Cost
//!
//! `examples/eval_bench.rs`, every configuration timed on the same 2146
//! positions:
//!
//! ```text
//!                                     Root::new    evaluate       sum
//! v6 (the round-six evaluation)         3.486 us    0.459 us   3.945 us
//! v7 (the round-seven evaluation)       3.484 us    0.471 us   3.955 us
//! default (round eight)                 3.515 us    0.473 us   3.988 us
//! ```
//!
//! **Round eight is free**, and could hardly not be: it changes six constants
//! in an array that was already being indexed, and turns one `match` on an age
//! into an index into a three-element array on the same `match`. The 0.03 us
//! spread is the benchmark's run-to-run noise.
//!
//! # Round nine: the wonder term was over-paying, and fixing it helps only
//! the agent that does not matter
//!
//! Round nine is **three new measurement instruments, one large calibration
//! defect found and quantified, one derived fix for it that is worth +30 to
//! +86 Elo to `phased` and −30 to `mcts-eval`, one exact refinement worth
//! nothing, and a diagnosis of a flagged position that turns out not to be an
//! evaluation error at all**. [`Config::default`] is **unchanged** — the first
//! round of this crate that measured its candidates and shipped none of them,
//! which is why there is no `v9()` and why `tests/round_nine_identity.rs` is
//! not a `vN_identity.rs`.
//!
//! ## The brief, and the honest answer to it
//!
//! The brief was science, in detail: that science evaluation should not be
//! purely age-based, that it matters which turn of the age it is and how many
//! cards remain, that a missing symbol being *face up and known* is a different
//! claim from its being *face down but plausibly reachable*, that the relative
//! extra-turn-wonder count and who starts Age III should feed a science read.
//! Alongside it, flagged as separate, a cross-cutting note: **wonders seem
//! overrated, because the evaluation builds them too early while they are
//! still expensive.**
//!
//! Every one of those was checked empirically rather than argued about, and the
//! answer is not the one the brief expected. **The science-specific factors are
//! mostly already priced or immaterial. The cross-cutting wonder note is
//! correct, large, and the two turn out to be the same finding** — a player
//! ahead on unbuilt play-again wonders is over-valued by up to twenty victory
//! points, which is simultaneously the parity factor the science brief named
//! and the over-rating the wonder note named.
//!
//! And then the fix for it, derived rather than fitted, is worth **+30 Elo to
//! `phased` in self-play, +86 and +56 against `mcts-uct`, +3 against
//! `alphabeta` — and −24 and −35 to `mcts-eval`**. `mcts-eval` is the
//! consumer that decides, so it ships as [`WonderModel::Rationed`], off by
//! default, with every column written down. That spread is the round's most
//! useful output and the reason the section below is long.
//!
//! ## Three instruments
//!
//! **`examples/science_calibration.rs --factors`** (extended). Round eight's
//! instrument bucketed positions by the mover's distinct-symbol count. Round
//! nine adds the five factors the brief named — cards left in the structure,
//! how many symbols are still assemblable, how many missing symbols are face
//! up *right now*, the unbuilt-play-again-wonder differential, and who took
//! the first decision of the age — and splits every `(age, symbols)` cell by
//! each of them, plus a `δ_k` fit run *inside* each bin. The logic is what
//! makes the tables worth reading: a gap that is the same in every bin of a
//! factor is a factor the evaluation already prices; **a gap that moves across
//! the bins is a term the evaluation is missing**, and that is the only kind
//! of finding that names a new term.
//!
//! **`examples/wonder_calibration.rs`** (new). The same method aimed at
//! unbuilt wonders: bucket by how many a player holds, by the play-again
//! differential, and by [`terms::wonder_p_build`] — the quantity the flat model
//! does not read at all.
//!
//! **`examples/position_probe.rs`** (new). The offline half of advanced mode's
//! flag loop: replay an exported `{ seed, moves }`, print the ranked action
//! list the analysis endpoint would show, print the same list with one term
//! switched off, and print the per-term breakdown behind both.
//! `tests/flagged_positions.rs` is the committed form for the position this
//! round was handed.
//!
//! ## The flagged position: one real finding, one non-finding
//!
//! Seed 1, thirteen moves, turn 13 of Age I, `value = -18.4436` and
//! `win_probability = 0.3321` — reconstructed exactly, and pinned in
//! `tests/flagged_positions.rs`.
//!
//! **The "every legal action reads lower than standing still" anomaly is not
//! an evaluation error.** The five actions read 0.236-0.270 against a standing
//! 0.332, and the whole of that is [`menu::menu_term`] changing sign with
//! whoever moves *next*. It is `+λ·menu(me)` on a pre-move state and
//! `−λ·menu(opp)` on a post-move one — here `+4.09` and about `−8.06` victory
//! points, a twelve-point step that has nothing to do with the moves being
//! bad. Switch the menu off on both sides and the ordering reverses exactly:
//! standing `-22.53`, best action `-18.20`.
//!
//! ```text
//!                              menu on            menu off
//! standing                     -18.44  p=0.332    -22.53  p=0.299
//! Build 16 (theater)           -26.25  p=0.270    -18.20  p=0.334
//! Discard 12 (clay-pit)        -26.89  p=0.265    -20.62  p=0.314
//! Build 15 (garrison)          -31.03  p=0.236    -22.97  p=0.295
//! ```
//!
//! `evaluate` is antisymmetric and internally consistent, and it is not
//! *supposed* to be comparable across a change of mover: a 1-ply agent takes
//! an argmax over candidates that all sit on the same side of the term. The
//! defect is in displaying a pre-action value beside post-action ones as if
//! they were the same quantity, and it is **general** — it will fire on every
//! position where the mover has a decent menu, which is most of them. The fix
//! is not in this crate. What `duels-server`'s analysis endpoint wants is
//! either the previous decision's post-action value as the baseline, or the
//! *pass* baseline (the best score available, so the deltas are between
//! candidates), and both are its own call to make. It is written down here
//! because the second-order effect is worse than the cosmetic one: an operator
//! reading that overlay concludes the evaluation is broken when it is not, and
//! flags positions accordingly.
//!
//! **The wonder note is a real finding.** One move earlier, with seven coins
//! in hand, the evaluation ranks all three ways of building The Great
//! Lighthouse above every card build available, and reading the two on the same
//! side of the menu term it calls the build worth `+11.4` victory points for
//! seven coins. The breakdown says where that comes from, and it is not the
//! wonder term:
//!
//! ```text
//!                          before the build    after      the build is worth
//! terms::resource_bill          -35.06        -24.98            +10.08
//! terms::development_value       +3.94         +8.15             +4.21
//! printed victory points          0.00         +4.00             +4.00
//! coins (7 -> 0)                 +4.59          0.00             -4.59
//! wonder_potential              +16.80        +13.80             -3.00
//! ```
//!
//! A produce-a-raw-material wonder cuts the projected resource bill by ten
//! victory points, and seven coins are priced at four and a half. Both halves
//! are defensible on their own; what is missing is that **the same wonder built
//! later costs fewer coins, and no term in this evaluation can represent
//! "build it when it is cheap" as an alternative to "build it now".** A
//! `Build`-versus-`BuildWonder` comparison at 1 ply is a comparison against
//! *not building it at all*, and against that it genuinely is a good move.
//! Round nine did not build an option-value term; see "what is left".
//!
//! ## What each science factor actually said
//!
//! One fixed corpus, 2,000 games, two science-tilted seats
//! (`phased:base=v7,sci=3.0`, exactly as round eight generated its), read by
//! `Config::v8()`. First-reach positions, the player to move.
//!
//! **1. Round eight's own named follow-up is not supported.** Round eight left
//! "an age- or `decisions_left`-scaled rung is the obvious follow-up" on the
//! strength of Age III reading `δ4 = −9.6` and `δ5 = −14.7` where Age II was
//! flat. Split Age III by whether six symbols are still assemblable and that
//! disappears:
//!
//! ```text
//!  age   reachable          d1     d2     d3     d4     d5     (mover positions)
//!   2    6+ (race live)    +0.1   -5.4   -7.1   -1.6   -9.2     7341..187
//!   2    <=5 (dead)        +0.2   -4.5  -11.0   -2.0   -3.6     7768..0
//!   3    6+ (race live)   +11.3  +15.1   +6.2   +2.2   -3.3        0..1787
//!   3    <=5 (dead)        -1.7   -5.0   -8.8  -11.0  -15.5     6279..739
//! ```
//!
//! **Where the race is live the rung is right to within a couple of victory
//! points in both ages** (`+2.2` and `−3.3` at four and five symbols in Age
//! III). The whole of round eight's Age III over-credit is the *dead-race*
//! population — and [`ScienceWeights::dead_race_scale`] is already `0.0`
//! there, so the residual is not the rung at all. It is the rest of a science
//! city: four or five green cards' printed points, credited at face value by
//! [`terms::card_and_token_vp`], for card picks that bought no production, no
//! shields and no coins. An age-scaled rung would have been fitted against the
//! wrong thing. **Not built, and the reason written down** — which is exactly
//! what round eight's instrument was for.
//!
//! **2. The within-age gradient is real but small, and its obvious cause is
//! not the cause.** At three symbols in Age II the fitted correction runs
//! `−3.8` early in the age, `−9.0` mid and `−13.1` late — the project owner's
//! "it matters which turn of the age it is", worth about nine victory points.
//! The natural mechanism is that [`terms::supremacy_live`] grows *more*
//! optimistic as an age drains: three of every age's cards go back in the box
//! unseen, a boxed card is in none of the masks the walk consults, and by the
//! last turns of an age most of what is left in the deck list is exactly
//! those. [`ReachModel::Structure`] fixes that exactly (see below) and moves
//! the gradient by **0.7 of a victory point** (`−13.1` to `−12.4`). So the
//! gradient is real and this is not what causes it.
//!
//! **3. Face-up against face-down missing symbols: no signal.** Age II at two
//! symbols reads `+0.038 / +0.015 / +0.125` across zero, one and two-or-more
//! missing symbols face up; Age III at three reads `−0.049 / −0.123 / −0.172`,
//! the other way. Signs disagree between ages and the thin bins carry ±0.17.
//! Nothing to price.
//!
//! **4. Who started the age: a consistent signal, and confounded.** A player
//! who took the age's first decision beats the prediction and one who did not
//! falls short of it, on all four Age III symbol rows and both Age II rows
//! (`+0.121` against `−0.055` at two symbols, `−0.051` against `−0.167` at
//! three). It is worth 0.1-0.17 of win probability and it is exactly the
//! `age_start_lab` effect this project already measured — but *who* starts is
//! decided by the militarily weaker player, so the split is not a random
//! assignment and this corpus cannot separate "starting is good" from "the
//! kind of player who gets to start". Left as a follow-up with a design note:
//! it wants the `age_start_policy` harness, not this instrument.
//!
//! **5. The extra-turn differential is a large, signed, mispriced factor — and
//! in the opposite direction to the read that prompted it.** This is the
//! finding.
//!
//! ```text
//!  age  extra-turn diff      n   predicted   actual     gap
//!   3   behind             114     0.558      0.675    +0.118
//!   3   level              424     0.523      0.512    -0.011
//!   3   ahead              126     0.493      0.294    -0.200
//! ```
//!
//! A player **ahead** on unbuilt play-again wonders wins far *less* than
//! [`win_probability`] claims, and one **behind** wins more. The brief's read
//! was that an extra-turn advantage is worth *more* in a science race; the data
//! says the evaluation is already paying too much for it.
//!
//! ## The wonder term, which is what that factor was actually measuring
//!
//! `examples/wonder_calibration.rs` on an untilted corpus (4,000 games,
//! 571,814 player-positions, read by `Config::v8()`) says the error is not
//! about play-again wonders specifically at all — it is about **unbuilt wonders
//! that are never going to be built**. Three independent slices, mover
//! positions:
//!
//! ```text
//! unbuilt wonders held             n     predicted   actual     gap    pot. vp
//!   age 3, none               38071       0.561      0.697    +0.136     0.00
//!   age 3, one                28469       0.477      0.381    -0.096     0.46
//!   age 3, two                15523       0.494      0.264    -0.229     8.62
//!   age 3, four                 162       0.380      0.099    -0.281    21.37
//!
//! play-again differential          n     predicted   actual     gap    pot. vp
//!   age 3, -1                  7783       0.557      0.748    +0.191     0.50
//!   age 3, level              65985       0.517      0.501    -0.017     1.25
//!   age 3, +1                  8136       0.476      0.259    -0.217     8.32
//!
//! p_build                          n     predicted   actual     gap    pot. vp
//!   age 3, < 0.40             29108       0.470      0.333    -0.138     1.31
//!   age 3, 0.40-0.75          15446       0.497      0.329    -0.169     8.06
//!   age 3, 0.75-0.99           1033       0.539      0.530    -0.010     9.21
//! ```
//!
//! and, in the actionable units, the fitted additive correction against the
//! `p_build ≥ 0.99` bin:
//!
//! ```text
//!  age      <0.40   0.40-0.75   0.75-0.99   >=0.99 (ref)
//!    1      -23.2      -30.2        -6.9         0.0
//!    2      -10.3      -19.3       -19.5         0.0
//!    3      -10.7      -19.7       -14.0         0.0
//! ```
//!
//! **Ten to thirty victory points, in every age, concentrated exactly where
//! `p_build` is low** — and `p_build` is the one thing
//! [`WonderModel::Flat`] does not read. The flat model pays
//! `wonder_potential × wonder_power` for a drafted-but-unbuilt wonder at full
//! weight until the seven-wonder cap closes, and `0.5` is a constant standing
//! in for "it will probably get built". That is not a constant. It is
//! [`terms::wonder_p_build`], which has existed since round five as a
//! standalone probability estimate, factored out of
//! [`WonderModel::Budget`] precisely because it has nothing to do with how a
//! wonder's *effects* are priced.
//!
//! [`WonderModel::Rationed`] reads it: `p_build × ` the flat model's own
//! per-effect power, extra-turn premium included, unchanged. It is the one
//! channel of round four's eight-channel `Budget` bundle that this measurement
//! says is right, taken on its own.
//!
//! ### Elo, and the control that says what the gain is about
//!
//! `phased:wonder=rationed,wonder_potential=P` against `phased:base=v8` at
//! `Nodes(1)`, 3200 games per seed range, paired and seat-swapped. Every
//! interval is ±12.1.
//!
//! ```text
//!   P     seed 1   seed 5001   seed 9001   seed 13001   seed 17001
//!  0.35   -34.2
//!  0.50   -19.6
//!  0.75    +1.1
//!  1.00   +18.8      +22.7
//!  1.25   +29.7      +31.1       +30.5       +19.6        +17.4
//!  1.50   +24.4      +29.2
//!  2.00    -1.2
//! ```
//!
//! Unimodal with a plateau at 1.25-1.5, and `1.25` is positive on all five
//! ranges with every interval clearing zero. **The control is the result**:
//! the *same weight un-rationed* — [`WonderModel::Flat`] at
//! `wonder_potential = 1.25` — is worth **−73.0 / −75.5**, and `1.0` and
//! `0.75` there read −30.8 and −5.4. A hundred Elo separates one constant with
//! and without the `p_build` factor, so the gain is the rationing and not the
//! magnitude.
//!
//! It also is not the extra-turn premium in disguise: re-swept on top of the
//! rationed model, [`EvalWeights::wonder_extra_turn_premium`] still wants `9.0`
//! (4.5 reads −16.0, 6.0 −10.1, 12.0 −2.1 and 0.0 **−80.0** against it), which
//! is where round six left it.
//!
//! ### It transfers to one unrelated searcher, not the other — and reverses
//! ### inside `mcts-eval`
//!
//! 800 games per seed range, the searchers at `Nodes(2000)`:
//!
//! ```text
//!                             phased (default)          + the rationed model    worth
//! vs alphabeta    seed 1   +53.3 [+29.0, +77.7]      +56.4 [+32.1, +80.8]       +3.1
//! vs mcts-uct     seed 1  -133.3 [-159.1, -107.4]    -47.1 [-71.4, -22.8]      +86.2
//!                 seed 5001 -118.0 [-143.4, -92.5]   -61.8 [-86.2, -37.4]      +56.2
//! ```
//!
//! **+86 and +56 Elo against `mcts-uct` and +3 against `alphabeta`.** So it is
//! not self-play overfitting — it is the largest movement against `mcts-uct`
//! since round seven — and it is also not uniform: `alphabeta` cannot tell the
//! difference. The victory kinds say the gain against `mcts-uct` is not one
//! route: on the second range this side's wins go 269 → 329, scientific
//! 10 → 28 and civilian 252 → 288, while `mcts-uct`'s civilian wins fall
//! 394 → 315. (The `+53.3` is also a useful null: it reproduces round eight's
//! own figure exactly, which is what "the default path is bit-identical" looks
//! like from the outside.)
//!
//! And then the measurement that decides it. `mcts-eval` against
//! `mcts-eval:eval=v8` — the same binary either side, differing only in what
//! [`Config::default`] returns — at `Nodes(2000)`, 3200 games per range:
//!
//! ```text
//!                  seed 1                    seed 5001
//!   -24.4 [-36.4, -12.3]        -34.6 [-46.7, -22.5]
//! ```
//!
//! Negative on both ranges, both intervals clear of zero, and the victory kinds
//! say what happened: on the second range the rationed side wins **1112
//! civilian games against the anchor's 1451**, while its scientific wins go
//! 40 → 59. That is the crate docs' own account of `mcts-eval`'s blend read
//! back at us — the evaluation half is there to supply *civilian-score
//! judgement*, and raising one term two and a half times spends that away.
//!
//! **So this is the sharpest policy-versus-value split this crate has
//! measured, and it is the mirror image of round seven's.** Round seven found
//! a variant that predicted the winner two to three points better and was −29
//! Elo as a leaf. Round nine found one that is +30 Elo as a *policy* and −30 as
//! a leaf. The two findings are the same shape from opposite ends, and together
//! they say the thing worth carrying forward: **a 1-ply argmax is invariant to
//! a term's magnitude and a leaf value is not**, so a round that improves
//! `phased` by re-scaling a term has learned nothing about `mcts-eval` until it
//! runs the A/B. Round seven's advice — "the instrument that predicted
//! `mcts-eval`'s direction was the boring one: `phased` Elo, attenuated" —
//! needs amending: attenuated, and sometimes sign-flipped.
//!
//! Round nine looked for a formulation that paid both and did not find one.
//! [`EvalWeights::wonder_p_build_ref`] exists because of that search: at
//! [`terms::OPENING_P_BUILD`] it is the same decay with the *opening scale
//! preserved* — `min(1, p_build / (7/8))`, so the term is worth exactly what
//! the flat model paid until wonders start going up. That is the shape without
//! the change of scale, and `phased` does not want it: **−11.8** at
//! `wonder_potential = 0.5` and **+5.5** at `0.65`. Nor does routing the draft
//! signal through the premium instead: rationed at `0.5` with the premium at
//! 18 and 30 reads **+1.1** and **+6.3**. `phased`'s thirty Elo needs the
//! scale, and the scale is what `mcts-eval` cannot afford.
//!
//! ## `ReachModel::Structure`: exact, and worth nothing
//!
//! The project owner's face-up-against-face-down distinction, built and
//! measured. [`terms::supremacy_live`]'s walk calls a symbol reachable if
//! *some* card printing it is not provably gone and belongs to the current age
//! or a later one — which counts the three cards every age returns to the box
//! unseen, for the whole of that age.
//! [`ReachModel::Structure`] reads the current age off the structure instead: a
//! current-age symbol counts if it is **face up in the structure**, or if the
//! structure still holds at least one face-down card. Later ages are read from
//! the deck list exactly as before, and the whole refinement stands down when
//! the structure is empty, because `state.age()` is then an age whose cards are
//! all still coming.
//!
//! It is exact at the end of an age — with nothing face down left, the only
//! current-age symbols obtainable are the ones on the board — and it can only
//! ever call *fewer* symbols reachable, which
//! `round_nine_identity::the_structural_reach_model_never_calls_more_symbols_reachable`
//! asserts over real games rather than argues.
//!
//! And it measures at nothing. **+1.1 / +2.6** Elo for `phased` against
//! `phased:base=v8` over 3200 games on each of two disjoint seed ranges, and
//! the calibration it was built for moves by under a victory point (Age II's
//! three-symbol late bin `−13.1 → −12.4`, Age III's five-symbol late bin
//! `−17.6 → −13.0`). Kept as an option with the measurement written down,
//! because it is the more correct model and because the next round should not
//! have to rebuild it to find that out. It applies to the **gate only**, not to
//! `pair_threat`'s [`terms::second_copy_obtainable`], deliberately: they are
//! the same question, and moving both would have made this a measurement of two
//! things.
//!
//! ## What is left, in the order a tenth round should take it
//!
//! **1. The dead-race science city is over-credited by ten to fifteen victory
//! points and the ladder is not where it lives.** The `δ` split above is
//! unambiguous: with the rung already gated to zero, four symbols and a dead
//! race still reads `−11.0` in Age III. What is over-priced is the *printed
//! points* of the green cards, or rather the opportunity cost of the picks that
//! bought them, and [`EvalWeights::vp_projection`] is a single scalar over all
//! card colours. A colour-aware or race-aware projection is a real term this
//! evaluation does not have.
//!
//! **2. Wonder option value.** The flagged position's actual defect: nothing
//! represents "build this wonder later, when the production makes it cheap".
//! It is not a re-weighting — it needs a second candidate that does not exist
//! in the action list. The cheapest honest approximation is to charge a wonder
//! build the coins it pays *above* what the projected pool would let it pay
//! later, which is [`terms::development_by_resource`] read from the other end.
//!
//! **3. The pre-action/post-action comparability defect** in
//! `duels-server`'s analysis overlay, described above. Out of this crate's
//! scope and worth fixing where it lives, because it makes the evaluation look
//! wrong to whoever is flagging positions.
//!
//! **4. Who starts the age, measured properly.** The signal is consistent and
//! the corpus cannot de-confound it. `duels-arena/examples/age_start_lab.rs`
//! can.
//!
//! **5. Denial asymmetry for science cards** was in the brief and is
//! untouched. The claim — a green card is worth more to the player who already
//! holds its symbol than to the one who does not, so denying it is
//! asymmetrically valuable — is real and is *partly* priced, through
//! [`menu::MenuTables`] differencing two per-player take values. Whether the
//! remaining asymmetry is worth a term is unmeasured, and the instrument for it
//! would be a `menu`-level bucket table rather than either of the two here.
//!
//! **6. The one experiment that could flip the wonder verdict, and was not
//! run.** [`EvalWeights::value_scale`] is the knob round seven built for
//! exactly this shape of problem: it divides `mcts-eval`'s leaf temperature,
//! and round eight's closing note says a round that moves this crate's output
//! scale a long way should re-fit the calibration *and* re-run the `eval=vN`
//! A/B. The rationed model at `wonder_potential = 1.25` does move the output
//! scale. Round nine did not chase it, and the reason is an argument with a
//! measurement behind it rather than fatigue — but it is an argument, and a
//! tenth round should check it rather than believe it. Three things say a
//! global rescale cannot recover thirty Elo here. A *global* rescale cannot
//! change the leaf's **relative** term balance at all, and the victory kinds
//! say the harm is relative (civilian wins 1451 → 1112) rather than a loss of
//! resolution. The leaf is not saturating: the rationed wonder term
//! differenced reaches about ±20 victory points against an Age I temperature
//! of 26.4, which is `σ = 0.68` — nowhere near flat. And round seven's own
//! `value_scale` sweep **bounds** what the knob is worth: `k = 2.0` read
//! +84.7 and `k = 0.6` read +74.4 against an anchor where `k = 1.0` read
//! +94.8, so a global rescale in either direction moved that measurement by
//! ten to twenty Elo, not thirty-five.
//!
//! Two further gaps in the round's own protocol, stated rather than glossed.
//! There is **no `TimeMs` column** for the wonder verdict: this project's
//! two-budget rule attaches to "whatever you recommend as a new default", and
//! round nine recommends none, so the reject rests on two disjoint 3200-game
//! `Nodes(2000)` ranges plus the mechanism. And the `alphabeta` transfer check
//! is **one seed range**, where the `mcts-uct` one is two; it was the run that
//! looked least likely to matter and it turned out to be the one that
//! disagrees with the other opponent, so a tenth round revisiting this should
//! start by giving it a second range.
//!
//! ## Cost
//!
//! `examples/eval_bench.rs`, every configuration timed on the same 2146
//! positions. The **default path is bit-identical** to round eight's, so the
//! only cost worth reporting is what each new option costs when it is switched
//! on:
//!
//! ```text
//!                                         Root::new    evaluate       sum
//! v7 (the round-seven evaluation)          3.156 us    0.421 us   3.577 us
//! default (unchanged from round eight)     3.185 us    0.424 us   3.610 us
//! default + the rationed wonder model      3.176 us    0.422 us   3.598 us
//! default + the structural reach model     3.167 us    0.429 us   3.597 us
//! ```
//!
//! Both are free — the whole table spans 0.03 us, which is under this
//! benchmark's run-to-run noise, and the rationed row reads nominally *faster*
//! than the default it is a superset of. The structural reach model is the one
//! that had to be checked: round seven's dead-race walk was a **+47%**
//! regression on `evaluate` before its two guards went in, and this adds a
//! pass over the occupied slots *inside* that walk. It survives because both
//! guards still apply — the walk is skipped entirely when the ladder rung is
//! zero, and it stops at the second unreachable symbol — and because
//! [`terms::faceup_symbols`] is one pass for all seven symbols rather than one
//! per symbol.
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
//! Like `greedy-ev`, `phased` samples one concrete
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

use duels_core::engine;
use duels_core::scoring::{self, GameResult};
use duels_core::{Action, GameState, Player};
use duels_strategy::{deny_vp, stance_in, Context, PriorWeights, Stance, ThreatWeights, VpWeights};

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

/// Whether [`menu::TakeValue`] prices the coins a card pays *per building its
/// taker already owns*.
///
/// Five commercial cards — all of them Age III — print no coins at all and
/// instead pay `amount_per_unit ×` a count of the builder's own city, straight
/// through [`duels_core::data::Card::coins_per_own`]: three coins per
/// manufactured good, two per raw material, one per military building, one per
/// commercial building, two per constructed wonder. This is exactly the shape
/// of the guild bug round five fixed — [`menu::TakeValue::free_value`] starts a
/// card's value from `def.victory_points` and `def.coins`, and `def.coins` is
/// zero for all five — except that here the payout is not a projection but a
/// count that is already on the table, so there is nothing to estimate.
///
/// The main evaluation was never wrong about these cards: the coins arrive in
/// the post-action state and [`terms::coin_points`] reads them. It was the
/// *menu* that could not see them, and so under-valued what the next mover's
/// turn was worth and how much taking one of these away from them was worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CountPricing {
    /// A card is worth its printed coins, which for these five is zero — the
    /// pre-existing behaviour, reproduced bit for bit by [`Config::v6`].
    #[default]
    Unpriced,
    /// `coins_per_own × count(taker's city) × coin_marginal`, read off the
    /// player the menu is pricing for.
    ///
    /// **Off by default — an honest negative.** It is the more correct model
    /// and it measures at nothing: −2.7 / −0.9 Elo as a leave-one-out against
    /// the round-seven default over 3200 games on each of two disjoint seed
    /// ranges. The reason is almost certainly that all five cards are Age III
    /// and the menu term is `λ = 0.6` of one softmax entry, so the blind spot
    /// was real and rarely load-bearing. Kept, with the measurement written
    /// down, rather than enabled on the strength of the argument.
    Counted,
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
    /// behaviour, reproduced bit for bit by [`Config::v3`], and **still the
    /// default** after round nine measured the alternative on both consumers.
    #[default]
    Flat,
    /// [`terms::WonderBudget`]: a per-effect price, scaled by the chance the
    /// wonder is built at all given the seven-wonder cap and the decisions the
    /// owner has left.
    Budget,
    /// [`Flat`](WonderModel::Flat)'s per-effect prices — the extra-turn
    /// premium included, unchanged — scaled by [`terms::wonder_p_build`], the
    /// standalone probability that any one of the owner's unbuilt wonders is
    /// ever built. See [`terms::wonder_potential_rationed`].
    ///
    /// This is the *one* channel [`Budget`](WonderModel::Budget) moved that
    /// round four's measurement could not isolate. `Budget` re-prices eight
    /// things at once — several of them known-weak, one of them
    /// `duels_strategy::science::token_value`'s unmeasured constants — and
    /// measured −11.0 / −5.6 / −3.9 Elo as a bundle. Its **rationing** is the
    /// part `examples/wonder_calibration.rs` says is right, so round nine
    /// takes only that and leaves `Flat`'s flat `+3, this wonder does
    /// something` exactly where it is.
    ///
    /// # Off by default — and the sharpest policy-versus-value split this
    /// crate has measured
    ///
    /// At [`EvalWeights::wonder_potential`] `= 1.25` this is worth **+29.7 /
    /// +31.1 / +30.5 / +19.6 / +17.4** Elo to `duels-agent-phased` over 3200
    /// games on each of five disjoint seed ranges, every interval clearing
    /// zero, and **+86 / +56** against `duels-agent-mcts-uct` — and
    /// **−24.4 / −34.6** Elo to `duels-agent-mcts-eval` over 3200 games at
    /// `Nodes(2000)` on two disjoint ranges. It is off because `mcts-eval` is
    /// the consumer that decides; it is *available*, and documented at this
    /// length, because those are not small numbers in either direction.
    ///
    /// Two controls say what the split is about. The **same weight
    /// un-rationed** is worth −73.0 / −75.5 to `phased`, so a hundred Elo
    /// separates one constant with and without the `p_build` factor: the
    /// rationing is what makes a large weight survivable at all. And the
    /// rationing **at the old magnitude** — [`EvalWeights::wonder_p_build_ref`]
    /// at [`terms::OPENING_P_BUILD`], which is the same decay with the opening
    /// scale preserved — is worth −11.8 at `0.5` and +5.5 at `0.65`, so
    /// `phased`'s gain needs the scale and not only the shape.
    ///
    /// That is the whole conflict. A 1-ply argmax is invariant to the term's
    /// magnitude and cares only that the *ordering* of draft and build
    /// candidates improves. A leaf value is not: raising one term two and a
    /// half times re-balances what the logistic can see, and the crate docs
    /// record that the evaluation half of `mcts-eval`'s blend is there to
    /// supply *civilian-score* judgement. Round nine found no formulation that
    /// paid both — see the round-nine section of the crate docs for the four
    /// that were tried.
    Rationed,
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

/// How [`terms::supremacy_live`] decides whether a symbol the player does not
/// hold can still be obtained.
///
/// The distinction is the project owner's, and it is the science half of round
/// nine's brief: a missing symbol may be **face up and known**, or **face down
/// but plausibly reachable**, and those are not the same claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReachModel {
    /// A symbol is reachable if *some* card printing it is not provably gone
    /// — not in a city, not under a wonder, not in the discard pile — and
    /// belongs to the current age or a later one.
    ///
    /// The pre-existing behaviour, reproduced bit for bit by [`Config::v8`],
    /// and **optimistic in a way that gets worse as an age drains**: three of
    /// every age's cards go back in the box unseen at setup
    /// ([`duels_core::engine::new_game`]), and a boxed card is in none of the
    /// three masks, so it reads as available for the whole of its own age. By
    /// the last turns of an age most of what is left in the deck list is
    /// exactly those boxed cards.
    #[default]
    Optimistic,
    /// A **current-age** card counts only if it can actually still be taken:
    /// its symbol is face up in the structure, or the structure still holds at
    /// least one face-down card. Later ages are read from the deck list
    /// exactly as before, since their cards genuinely have not been dealt.
    ///
    /// The refinement is exact at the end of an age — with no face-down slots
    /// left, the only current-age symbols obtainable are the ones visible on
    /// the board — and it stands down entirely when the structure is empty,
    /// because `state.age()` is then an age whose cards are all still coming.
    /// In between it is the same optimism as [`Optimistic`](ReachModel::Optimistic),
    /// bounded by whether *anything* is still hidden.
    ///
    /// `examples/science_calibration.rs --factors` is what motivated it: the
    /// fitted correction at three distinct symbols in Age II runs **−3.8
    /// victory points early in the age, −9.0 mid and −13.1 late**, a monotone
    /// nine-point gradient in exactly the direction a reachability test that
    /// grows more optimistic as the age drains would produce.
    Structure,
}

/// The science ladder rounds one through seven all shipped: the value of
/// holding 0-5 distinct symbols, before [`EvalWeights::science_ladder`]
/// multiplies it.
///
/// Kept as a named constant so [`Config::v7`] restores it exactly and
/// `tests/v7_identity.rs` can pin it, rather than the round-seven shape being
/// recoverable only from this file's history.
pub const SCIENCE_LADDER_V7: [f64; 6] = [0.0, 1.0, 2.5, 6.0, 12.0, 18.0];

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
    /// What the ladder rung is multiplied by once **scientific supremacy is no
    /// longer reachable** for this player — when the symbols they do not hold
    /// can no longer all be obtained, counted from the card data by
    /// [`terms::supremacy_reachable`].
    ///
    /// `1.0` is the pre-existing behaviour and is what [`Config::v6`] restores:
    /// a player at four distinct symbols collected the full rung whether or not
    /// a fifth and sixth were still physically in the game. That is the flaw
    /// this knob fixes, and it is why the whole ladder measured **over-priced**
    /// — see [`EvalWeights::science_ladder`].
    ///
    /// The rung is not taken to zero. Distinct symbols keep paying without
    /// supremacy: they are printed victory points (which
    /// [`terms::card_and_token_vp`] already counts) and they are pairs waiting
    /// to be completed for a progress token (which [`terms::science_ladder`]'s
    /// own `pair_threat` prices, and which this scale deliberately does **not**
    /// touch). What is worthless once the race is dead is the *convexity* — the
    /// rung's jump from 6 to 30 to 54 exists because six symbols win the game.
    ///
    /// Round eight roughly tripled the top two rungs, which makes this gate
    /// carry correspondingly more: it is the difference between "four symbols
    /// and a live race" and "four symbols and nowhere to go", and that is now
    /// worth fifteen victory points rather than six.
    pub dead_race_scale: f64,
    /// How [`dead_race_scale`](ScienceWeights::dead_race_scale)'s gate decides
    /// whether a missing symbol is still obtainable. See [`ReachModel`].
    ///
    /// Applies to the **gate only**, not to `pair_threat`'s
    /// [`terms::second_copy_obtainable`], deliberately: they are the same
    /// question and moving both at once would have made the round-nine
    /// measurement a bundle of two things, exactly as round six declined to
    /// fold Theology into the extra-turn premium for the same reason.
    pub reach_model: ReachModel,
}

impl Default for ScienceWeights {
    fn default() -> Self {
        Self {
            // **Round eight raises the top two rungs, `12` and `18` becoming
            // `30` and `54`, and leaves the first four exactly where round
            // seven left them.** `examples/science_calibration.rs` is the
            // argument, and it is an empirical one: the round-seven evaluation
            // predicts a player's win probability well at zero through three
            // distinct symbols and badly above that, calling a four-symbol
            // Age II position a 44.9% loss where the player actually wins
            // 55.7% of the time (n = 461) and a five-symbol one 45.5% where
            // they win 78.3% (n = 69). The maximum-likelihood correction is
            // +13.4 and +25.6 victory points against rungs worth 6 and 9.
            //
            // The Elo is **neutral** — +1.4 pooled over 12,800 games, signs
            // disagreeing across four seed ranges — and this is adopted anyway,
            // on the calibration and on the victory-kind breakdown, the way
            // round three adopted [`rails`] on its audit. Sweeping the two
            // rungs through 20/34, 30/54 and 45/80 reads +3.5 / +5.0 / +2.6 on
            // top of the temperature refit at seed 1, so 30/54 is the middle of
            // a shallow unimodal curve. See the round-eight crate docs.
            ladder: [0.0, 1.0, 2.5, 6.0, 30.0, 54.0],
            strong_token_mult: 0.15,
            pair_threat_weight: 0.5,
            pair_token_share: 0.5,
            pair_tempo_tax: 0.5,
            // **Measured.** See the round-seven section of the crate docs.
            dead_race_scale: 0.0,
            reach_model: ReachModel::Optimistic,
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
    /// The `p_build` at which [`WonderModel::Rationed`] pays the full
    /// [`EvalWeights::wonder_potential`] weight: the factor is
    /// `min(1, p_build / ref)`. See [`terms::wonder_potential_rationed`].
    ///
    /// `1.0` is plain rationing and is the value [`Config::v8`] carries (where
    /// the whole model is switched off anyway, so it is inert there).
    /// [`terms::OPENING_P_BUILD`] — seven eighths, the value
    /// [`terms::wonder_p_build`] returns for the whole of Age I — is the
    /// *shape* without the change of scale.
    pub wonder_p_build_ref: f64,
    /// What a play-again wonder's extra turn is worth, in
    /// [`WonderModel::Budget`]. Deliberately `3.0` — the same number
    /// [`terms::wonder_power`] paid for "this wonder has an effect" — so that
    /// at `p_build = 1` and no other effect firing the two models agree.
    pub wonder_extra_turn_vp: f64,
    /// What [`WonderModel::Flat`] pays for an unbuilt wonder that prints
    /// **play again**, *on top of* [`terms::wonder_power`]'s flat `+3, this
    /// wonder has an effect`. **`9.0` by default**, so a play-again wonder is
    /// worth `12` against every other effect's `3`; `0.0` ([`Config::v5`]) is
    /// the pre-existing uniform treatment, bit for bit
    /// (`tests/v5_identity.rs`).
    ///
    /// The project owner's read is that an extra turn is the most valuable thing a
    /// wonder can print, and the flat model prices it exactly like a destroy
    /// or a free discard build. This is the one knob that tests that read,
    /// isolated from [`WonderModel::Budget`]'s other channels — which were
    /// measured negative as a bundle and are confounded with
    /// `duels_strategy::science::token_value`'s flat constants.
    ///
    /// Only ever read under [`WonderModel::Flat`]; `Budget` has its own
    /// [`EvalWeights::wonder_extra_turn_vp`].
    pub wonder_extra_turn_premium: f64,
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
    /// Victory points for being the player **to move**
    /// ([`terms::to_move`]) — the value of the right to move, and, on a
    /// post-action state, the credit for a move that earned an extra turn.
    /// Zero switches the term off entirely and restores the previous
    /// arithmetic exactly.
    ///
    /// Round six priced a play-again wonder while it was still *unbuilt* and
    /// measured that at about +47 Elo. This is the other half of the same
    /// idea: the turn once it is actually in hand. See [`terms::to_move`] for
    /// why that has to be read off `current_player` rather than off
    /// `GameState::extra_turn`.
    ///
    /// **Zero by default — an honest negative, and an unambiguous one.** As a
    /// leave-one-out against the round-seven default over 3200 games on each of
    /// two disjoint seed ranges, `2` is worth −21 / −16, `6` is worth −43 / −30
    /// and `12` is worth −45 / −36. `examples/leaf_probe.rs` says the opposite
    /// — the right to move is worth one or two victory points as a *predictor*
    /// — which is the same policy-versus-value divergence the round-seven
    /// section of the crate docs is about.
    pub to_move: f64,
    /// Weight on [`terms::token_equity`], the forward value of the progress
    /// tokens a player already **owns** — what Theology, Economy, Strategy,
    /// Architecture, Masonry and Urbanism are worth for the rest of the game,
    /// as against the printed victory points
    /// [`duels_core::scoring::breakdown`] already counts. Zero switches the
    /// term off entirely and restores the previous arithmetic exactly.
    ///
    /// **Zero by default — an honest negative, and a genuine gap correctly
    /// filled.** Six of the ten tokens are rules changes this evaluation
    /// priced at nothing, which also meant [`PendingModel::Completed`] chose
    /// between them on printed victory points alone. It is worth +7.1 / +8.0
    /// Elo at `0.5` against `phased:base=v6` over 3200 games on each of two
    /// disjoint seed ranges, and −2 / −5 as a leave-one-out against the
    /// finished round-seven default, which is the comparison that decides it.
    /// Kept as an option with the measurement written down; see
    /// [`terms::TokenTable`] for what each channel is priced from.
    pub token_equity: f64,
    /// A single multiplier on the whole weighted sum
    /// ([`evaluate`]'s ordinary return, and [`Root::denial_term`] with it) —
    /// **not** on the rails or on a finished game's `instant_result`, which
    /// are magnitudes rather than judgements.
    ///
    /// # What this is for, and who it is invisible to
    ///
    /// It is invisible to `duels-agent-phased`, which takes an argmax: scaling
    /// every candidate's score by the same positive constant cannot reorder
    /// them (the tie window is `1e-6` against scores of order ten, and the
    /// rails it does not scale are five hundred). It is **not** invisible to
    /// `duels-agent-mcts-eval`, which maps this crate's output through a
    /// logistic of fitted temperature `T` and averages the result into a
    /// win-rate estimate: multiplying by `k` there is exactly dividing that
    /// temperature by `k`, so this knob is the one instrument a
    /// `duels-eval` round has for asking whether the leaf value a search wants
    /// is sharper or flatter than the maximum-likelihood calibration
    /// `examples/calibrate.rs` fits.
    ///
    /// `1.0` — the calibration as fitted — is [`Config::v6`]'s value and is
    /// what the shipped default keeps; see the round-seven section of the
    /// crate docs for the sweep that says so.
    pub value_scale: f64,
    /// The temperature [`win_probability`] divides this crate's victory-point
    /// score by, in victory points, indexed by **age minus one**.
    ///
    /// # Why this is a `Config` field and not just the module constants
    ///
    /// Two different consumers want two different numbers out of the same
    /// logistic, and until round eight they were forced to share one:
    ///
    /// * a **diagnostic** — `duels-server`'s advanced-mode read, and
    ///   `examples/science_calibration.rs`'s own predictions — wants the
    ///   maximum-likelihood calibration, because the only thing it is for is to
    ///   be right about how often this position wins. That is the module
    ///   constant, and round eight refit it.
    /// * a **search leaf** — `duels-agent-mcts-eval`, which calls
    ///   [`win_probability`] with a [`Root`] and averages the result into a win
    ///   rate — wants whatever temperature makes the *search* strongest, which
    ///   round seven's [`EvalWeights::value_scale`] sweep had reason to believe
    ///   is not the same thing (`k = 2.0`, an exact halving of the temperature,
    ///   measured worse than `k = 1.0` there — which round eight then
    ///   contradicted; see below).
    ///
    /// Putting it here is also what makes the difference measurable at all:
    /// `mcts-eval` pins a whole `Config` under the arena's `eval=vN` key, and a
    /// free constant is invisible to that pin, so before round eight a change
    /// to the leaf mapping could not have been A/B tested against the
    /// generation before it at all.
    ///
    /// # What the default is, and why it is not obviously right
    ///
    /// **The default is the module constants** — the maximum-likelihood refit —
    /// and that is a measurement rather than an assumption that the two
    /// consumers agree. Adopting it is worth **+15.3 / +19.0 / +20.6 / +22.9
    /// Elo** to `mcts-eval` against `mcts-eval:eval=v7` over 3200 games on
    /// each of four disjoint seed ranges (`+19.5 ± 6.0` pooled), and it
    /// reproduces at a wall-clock budget and against an unrelated `mcts-uct`
    /// anchor; the round-eight section of the crate docs has all of it.
    ///
    /// That is the *opposite* of what round seven's [`EvalWeights::value_scale`]
    /// sweep implied — there, `k = 2.0`, an exact halving of the temperature,
    /// measured 10 Elo worse than `k = 1.0`. The two are reconcilable (that
    /// sweep was run on an intermediate bundle, on one seed range, and scaled
    /// `evaluate`'s output rather than the temperature, so it also moved the
    /// rails' relative magnitude), but the disagreement is the reason this is a
    /// field: a future round that moves this crate's output scale a long way
    /// should re-fit the calibration *and* re-run the A/B rather than assume
    /// the fitted temperature is automatically the leaf a search wants.
    ///
    /// [`Config::v7`] carries [`WIN_PROBABILITY_TEMPERATURE_V7`], the stale
    /// pre-round-eight triple, which is what makes that A/B a single-binary
    /// measurement.
    pub win_probability_temperature: [f64; 3],
}

impl EvalWeights {
    /// The leaf temperature for a position in `age`.
    ///
    /// Ages outside `1..=3` cannot occur — [`duels_core::GameState::age`] only
    /// ever reports one of the three — and are read as Age III, the sharpest
    /// setting, exactly as the free [`win_probability_temperature`] does.
    #[inline]
    pub fn win_probability_temperature(&self, age: u8) -> f64 {
        match age {
            1 => self.win_probability_temperature[0],
            2 => self.win_probability_temperature[1],
            _ => self.win_probability_temperature[2],
        }
    }
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
            // **Was `1/3` — the rate at which a coin becomes a victory point
            // — on the argument that "a coin this city never has to spend is
            // worth exactly what a coin in hand is worth". Round seven cut it
            // to `0.2`.** The argument is sound about a coin and wrong about
            // this term: `development_value` counts a *want* the pool is
            // projected to have, over `take_rate x decisions_left` builds
            // that may never happen, and a projected saving is not a coin. It
            // is worth +4 / +14 Elo over 3200 games on each of two disjoint
            // seed ranges as the last step of the round-seven refit (+149.2 /
            // +165.6 against `phased:base=v6`, where `1/3` reads +145.5 /
            // +151.8), on a flat curve between 0.16 and 0.25.
            development: 0.2,
            development_take_rate: 0.6,
            // **Was `1.0`. Round seven halves it, and the same round adds
            // the reachability gate that explains why it was too big** — see
            // [`ScienceWeights::dead_race_scale`]. The two together are the
            // largest single effect of the round: `1.0` costs −24 / −27 Elo as
            // a leave-one-out against this default over 3200 games on each of
            // two disjoint seed ranges, and switching the gate off as well
            // costs another −10 / −7.
            //
            // The honest reading of the sweep is that the ladder was worth
            // *less* than nothing at `1.0`: with everything else at its
            // round-six value, driving this weight to `0.1` was worth +63 /
            // +57 Elo on its own. `0.5` with the gate in place is the setting
            // that keeps the scientific-supremacy route on the board — the
            // gate costs the route almost nothing, while a flat cut to `0.2`
            // takes this agent's science wins from ~40 in 3200 games to 3 —
            // and it is the setting the round-seven refit was tuned around.
            //
            // **Round eight left this weight alone and re-shaped
            // [`ScienceWeights::ladder`] instead**, which is the distinction
            // that matters: round seven's evidence was that the ladder's
            // *middle* was over-priced, and round eight's is that its *top* is
            // under-priced. A scalar cannot express both. See the round-eight
            // section of the crate docs for the empirical calibration that
            // separates them.
            science_ladder: 0.5,
            science: ScienceWeights::default(),
            race_card_liquidity: 0.15,
            race_liquidity_cap: 8.0,
            coin_safety_floor: 3.0,
            coin_safety_penalty: 0.5,
            resource_vulnerability: 0.4,
            // **`One`, and derived rather than fitted — which is a reversal.**
            // Round two fitted `3.0` and flagged it: the term already divides
            // the bill by three, which is the rate at which coins become
            // victory points, so `1.0` is all this weight has to say and
            // three said a trade coin was worth a whole victory point. Round
            // two's own comment called that "a measurement, not an argument".
            //
            // Round seven re-measured it with the science ladder and chain
            // equity corrected, and the derived value now wins by a wide
            // margin: sweeping 0.6 / 1.0 / 1.4 / 1.7 / 2.0 / 2.2 / 2.5 / 3.0
            // against `phased:base=v6` over 3200 games on each of two disjoint
            // seed ranges reads +135/+142, +145/+144, +148/+145, +130/+133,
            // +123/+127, +116/+122, +109/+117 and +98/+107 — monotone from
            // `3.0` down to a plateau at 1.0-1.4. **`3.0` was compensating for
            // two other over-priced terms**, which is exactly the failure mode
            // a fitted weight has and a derived one does not; `1.0` is taken
            // because it is the honest rate and is indistinguishable from the
            // top of the plateau.
            resource_bill: 1.0,
            coin_smooth_beta: 0.6,
            coin_smooth_ref: 5.0,
            coin_endgame_decisions: 2.0,
            // **Was `1.0`; round seven cuts it to a quarter, and this is the
            // single largest re-weighting of the round.** Round two shipped
            // `1.0` and recorded that switching it off was worth −51 / +17 —
            // "indistinguishable from zero, kept on the strength of the pooled
            // result and of the fact that [`menu`] needs its table anyway".
            // With the round-seven ladder in place the sign is no longer in
            // doubt: 0.9 / 0.6 / 0.5 / 0.25 / 0.1 / 0.0 read +66/+63,
            // +80/+79, +84/+90, +98/+107, +95/+112 and +96/+112 against
            // `phased:base=v6` over 3200 games on each of two disjoint seed
            // ranges. Flat below 0.25, so a quarter is taken rather than zero:
            // the forward value of a chain starter is real, it was simply
            // priced at four times what it is worth, and keeping the term
            // non-zero keeps `menu`'s table honest about what it feeds.
            chain_equity: 0.25,
            menu: MenuWeights::default(),
            deny_chain_gift: 0.5,
            // **Still `0.5`, and round nine is why that is now a measurement
            // rather than an inheritance.** Under [`WonderModel::Rationed`]
            // the honest weight is `1.25`, and it is worth **+29.7 / +31.1 /
            // +30.5 / +19.6 / +17.4** Elo to `phased` on five disjoint seed
            // ranges and **+86 / +56** against `mcts-uct` — and **-24.4 /
            // -34.6** to `mcts-eval`, which is the consumer that decides. The
            // pair is off by default together; the round-nine section of the
            // crate docs has every column and what separates them.
            wonder_potential: 0.5,
            wonder_turns_per_wonder: 2.5,
            wonder_p_build_ref: 1.0,
            wonder_extra_turn_vp: 3.0,
            // A play-again wonder is worth `3 + 9 = 12` to the flat model,
            // four times what every other effect gets. **Measured**, not
            // argued: +45.3 / +51.6 / +49.9 / +38.5 / +51.9 Elo against
            // `phased:base=v5` over 3200 games on each of five disjoint seed
            // ranges, on a curve that is unimodal in the premium and positive
            // on every range at every magnitude from 1.5 to 15. See the crate
            // docs for the two controls that say this is about *extra turns*
            // rather than about the wonder term wanting a bigger number.
            wonder_extra_turn_premium: 9.0,
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
            to_move: 0.0,
            token_equity: 0.0,
            // The calibration as `examples/calibrate.rs` fits it. See the
            // field docs, and the round-seven sweep that measured both
            // directions and found neither.
            value_scale: 1.0,
            // The refit calibration, which round eight measured as *also*
            // being the leaf a search wants: +19.5 +- 6.0 Elo to `mcts-eval`
            // against `mcts-eval:eval=v7` over 12,800 games, positive on all
            // four seed ranges. Written as the constants rather than as
            // literals so that the diagnostic mapping and the leaf mapping
            // cannot silently drift apart — a round that wants them to differ
            // should say so here, deliberately, with the measurement that
            // justifies it. See the field docs.
            win_probability_temperature: [
                WIN_PROBABILITY_TEMPERATURE_AGE_I,
                WIN_PROBABILITY_TEMPERATURE_AGE_II,
                WIN_PROBABILITY_TEMPERATURE_AGE_III,
            ],
        }
    }
}

/// Everything the evaluation can be tuned with.
///
/// Re-exported by `duels-agent-phased` as its own `Config`, so every
/// `phased:base=v1,...` spec string the arena already parses keeps working.
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
    ///
    /// **[`WonderModel::Rationed`] since round nine**; [`Config::v8`] restores
    /// [`WonderModel::Flat`].
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
    /// Whether the menu prices a commercial card's count-scaled coin payout.
    /// See [`CountPricing`].
    pub count_pricing: CountPricing,
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
                // have to arrive at their own *off* values here, and round
                // five is the first whose defaults are non-zero. Chaining
                // through `v3().eval` — which is `v4().eval`, which is
                // `v5().eval` — is what keeps this snapshot a snapshot as the
                // defaults move on.
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
                // Chained through `v5().eval`, not `default().eval`, for the
                // reason spelled out in `v2()`: every *later* round's weights
                // have to arrive at their own off values here, and round six's
                // extra-turn premium is the second whose default is non-zero.
                ..Config::v5().eval
            },
            guild_pricing: GuildPricing::Unpriced,
            menu_floor: MenuFloor::None,
            menu_afford_soft: 0.0,
            supply_model: SupplyModel::Raw,
            ..Config::v5()
        }
    }

    /// The configuration the *fifth* round of work shipped with: no extra-turn
    /// premium on [`WonderModel::Flat`], so a play-again wonder is worth the
    /// same flat `+3` as a destroy or a free discard build.
    ///
    /// `tests/v5_identity.rs` asserts this reproduces that agent's arithmetic
    /// bit for bit, which is what makes `phased` against `phased:base=v5` a
    /// single-binary measurement.
    pub fn v5() -> Config {
        Config {
            eval: EvalWeights {
                wonder_extra_turn_premium: 0.0,
                ..Config::v6().eval
            },
            ..Config::v6()
        }
    }

    /// The configuration the *sixth* round of work shipped with.
    ///
    /// Round seven changed [`Config::default`], so — following the contract
    /// spelled out under [`Config::v7`] — this function stops being an alias
    /// for the default and spells out round seven's *off* values instead:
    /// the science ladder back at its round-six weight with the dead-race gate
    /// disabled, no owned-token equity, the menu blind to a commercial card's
    /// count-scaled coins, and the value scale at one.
    ///
    /// `tests/v6_identity.rs` asserts this reproduces round six's arithmetic
    /// bit for bit, which is what makes `phased` against `phased:base=v6` a
    /// single-binary measurement.
    pub fn v6() -> Config {
        Config {
            eval: EvalWeights {
                development: 1.0 / 3.0,
                resource_bill: 3.0,
                chain_equity: 1.0,
                science_ladder: 1.0,
                science: ScienceWeights {
                    dead_race_scale: 1.0,
                    pair_threat_weight: 1.0,
                    ..Config::v7().eval.science
                },
                to_move: 0.0,
                token_equity: 0.0,
                value_scale: 1.0,
                ..Config::v7().eval
            },
            count_pricing: CountPricing::Unpriced,
            ..Config::v7()
        }
    }

    /// The configuration the *seventh* round of work shipped with.
    ///
    /// Round eight changed [`Config::default`], so — following the contract
    /// spelled out under [`Config::v8`], and exactly as round seven did to
    /// [`Config::v6`] — this function stops being an alias for the default and
    /// spells out round eight's *off* values instead: the leaf temperature at
    /// the pre-round-eight triple [`WIN_PROBABILITY_TEMPERATURE_V7`], and the
    /// science ladder's top two rungs at their round-seven values.
    ///
    /// `tests/v7_identity.rs` asserts this reproduces round seven's arithmetic
    /// bit for bit, which is what makes `mcts-eval` against
    /// `mcts-eval:eval=v7` a single-binary measurement.
    pub fn v7() -> Config {
        Config {
            eval: EvalWeights {
                win_probability_temperature: WIN_PROBABILITY_TEMPERATURE_V7,
                science: ScienceWeights {
                    ladder: SCIENCE_LADDER_V7,
                    ..Config::v8().eval.science
                },
                ..Config::v8().eval
            },
            ..Config::v8()
        }
    }

    /// The configuration the *eighth* round of work shipped with, which is
    /// also today's [`Config::default`].
    ///
    /// **Round nine deliberately did not move the default**, so this is still
    /// the newest link in the chain. It added two options — the
    /// [`WonderModel::Rationed`] wonder term and the
    /// [`ReachModel::Structure`] symbol-reachability test — and measured both,
    /// and each measurement said to leave the default alone: the first is
    /// worth +30 Elo to `phased` and **−24** to `mcts-eval`, the second is
    /// neutral to both. `tests/round_nine_identity.rs` is the guard that they
    /// really are off, and that the one *shape* round nine changed (the
    /// symbol walk grew a second branch) reproduces round eight bit for bit
    /// on the branch it kept.
    ///
    /// # Why a snapshot with no `#[cfg(test)]` copy behind it
    ///
    /// `v1()`-`v5()` each exist so a *later* round can be measured against an
    /// *earlier* one in a single binary, and each has a `tests/vN_identity.rs`
    /// holding a verbatim copy of the code it snapshots. There is nothing to
    /// copy here: this generation *is* the current arithmetic, so the identity
    /// that matters is a different one — that a **search agent pinning this
    /// generation keeps getting the same numbers**.
    ///
    /// This generation was `duels-agent-mcts-uct`'s pinned leaf-evaluation
    /// config while that agent's leaf value existed; that machinery has since
    /// moved to `duels-agent-mcts-eval`, which deliberately does **not** pin —
    /// it reads [`Config::default`] live at every tree construction, precisely
    /// so it keeps getting stronger as later `phased` rounds land, rather than
    /// needing a version bump to benefit from one (see that crate's docs for
    /// why). **No agent currently pins a generation of this crate**, so there
    /// is presently no downstream golden-values test that would catch a
    /// silent arithmetic change here — the closest thing on record was
    /// `mcts-uct`'s, and it no longer exists.
    ///
    /// # The contract for the next round, if a future consumer ever pins again
    ///
    /// The moment [`Config::default`] moves, this stops being a snapshot of
    /// anything. So a round that changes the default must, in the same PR:
    /// add `v9()` and re-point this function's `..` at it, spelling out round
    /// nine's *off* values here (exactly as [`Config::v2`]'s comment
    /// describes, and exactly as round eight did to [`Config::v7`]).
    /// **Round nine is the first round that did not have to**, having measured
    /// its two candidates and left the default alone; a round-ten change to a
    /// default still owes the chain a `v9()`, and the two round-nine options
    /// are already at their off values here, so that snapshot is one `..`
    /// away. If some future agent pins a generation the way `mcts-uct` once
    /// did, that agent's own golden-values test is what re-baselining means
    /// for it — this crate cannot enforce that on its behalf.
    ///
    /// Because the newest link in this chain is defined *as* the default
    /// (`v1`-`v7` are deltas from it, not literal field values), any
    /// same-crate check that `v8 == default` is a tautology — this is not a
    /// gap introduced by removing the downstream test, it was always true.
    /// See the note in
    /// `tests::the_generation_snapshots_are_a_chain_of_distinct_configurations`.
    pub fn v8() -> Config {
        Config::default()
    }
}

impl Config {
    /// A short, reproducible encoding of the configuration, for an agent's
    /// `duels_agents_api::AgentSpec::params` (this crate does not depend on
    /// that one; `duels-agent-phased` is the caller that fills it in).
    pub fn params_string(&self) -> String {
        let e = &self.eval;
        let b = &self.blend;
        format!(
            "sci={:.2}/dead={:.2}/pair={:.2}/reach={},ladder={:?},temp={:?},\
             tokeneq={:.2},tomove={:.2},scale={:.3},count={},\
             guild={}/{:.2},menufloor={},afford={:.2},supply={},yellow={:.2}@{:.3},\
             models={}/{}/{},pending={},wonder={}/{:.2}/{:.2}/{:.2},destroyrepl={},\
             rails={}/{:.0},shieldprice={},horizon={},lockin={:.2},\
             menu={:.2}@{:.2},chaineq={:.2},bill={:.2},band={:.2}/{:.2},\
             smooth={:.2}@{:.1}|\
             mil={:.2}/{:.2},vp={:.2},coin={:.2},dev={:.3}@{:.2},sci={:.2},raceliq={:.2},econ={:.1}/{:.2}/{:.2},chain={:.2},wonder={:.2},start={:?},deny={:.2}x{:.2},win={:.0}|\
             blend={},a={:.2},b={:.2},n={:.1},c0={:.2},floors={:.2}/{:.2}/{:.2}/{:.2}/{:.2},boosts={:.2}/{:.2}",
            e.science_ladder,
            e.science.dead_race_scale,
            e.science.pair_threat_weight,
            match e.science.reach_model {
                ReachModel::Optimistic => "optimistic",
                ReachModel::Structure => "structure",
            },
            e.science.ladder,
            e.win_probability_temperature,
            e.token_equity,
            e.to_move,
            e.value_scale,
            match self.count_pricing {
                CountPricing::Unpriced => "unpriced",
                CountPricing::Counted => "counted",
            },
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
                WonderModel::Rationed => "rationed",
            },
            e.wonder_turns_per_wonder,
            e.wonder_extra_turn_vp,
            e.wonder_extra_turn_premium,
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
    /// `p_build`, indexed by [`Player::index`], for
    /// [`WonderModel::Rationed`]. Root-fixed for the reason
    /// [`terms::wonder_potential_rationed`] spells out; all zero under every
    /// other model, so nothing reads it there and nothing pays for it.
    wonder_p_build: [f64; 2],
    guilds: terms::GuildTable,
    tokens: terms::TokenTable,
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
        // `p_build` for the rationed flat model, root-fixed exactly as
        // `WonderBudget` root-fixes the same number, and computed only when
        // something reads it — it is a handful of counts, but `Root::new` runs
        // once per search-tree node and this keeps every other model's
        // arithmetic bit-identical to before the variant existed.
        let wonder_p_build = if config.wonder_model == WonderModel::Rationed {
            [Player::One, Player::Two].map(|p| terms::wonder_p_build(state, p, &config.eval))
        } else {
            [0.0; 2]
        };
        let replace = if config.destroy_replace_discount {
            std::array::from_fn(|r| (supply.sources[r] * DESTROY_REPLACE_SHARE).min(1.0))
        } else {
            [0.0; duels_core::data::NUM_RESOURCES]
        };

        // The owned-token prices, read off the two `TakeValue`s the menu has
        // already built rather than recomputed, so the two cannot disagree
        // about what a shield or a coin is worth. Built only when something
        // reads it.
        let tokens = if config.eval.token_equity == 0.0 {
            terms::TokenTable::empty()
        } else {
            terms::TokenTable::of(
                state,
                &supply,
                chain.starters(),
                [take[0].shield_delta[1], take[1].shield_delta[1]],
                [take[0].coin_marginal, take[1].coin_marginal],
                &config.eval,
            )
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
            tokens,
            wonders,
            wonder_p_build,
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

    /// The root-fixed forward prices of the ten progress tokens, for
    /// diagnostics. All zero unless [`EvalWeights::token_equity`] is non-zero.
    #[inline]
    pub fn tokens(&self) -> &terms::TokenTable {
        &self.tokens
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
        let v = self.config.eval.deny * self.deny_scale * deny_vp(action, &self.stance);
        // Scaled with the position value it is added to, so a `value_scale`
        // that only a search can see cannot quietly reweight the one term a
        // 1-ply agent adds outside `evaluate`. Guarded, so `1.0` is exact.
        if self.config.eval.value_scale == 1.0 {
            v
        } else {
            v * self.config.eval.value_scale
        }
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

/// The maximum-likelihood temperature for an Age I position, in victory
/// points, fitted by `examples/calibrate.rs` over `phased` self-play (see that
/// example for how to reproduce it).
///
/// **Refitted in round eight, and the previous value was badly stale.** It read
/// `47.57` — fitted over 28,723 positions against round *six*'s evaluation, and
/// left untouched when round seven re-weighted five terms. The refit, over
/// 643,875 positions from 8,990 decided games of the round-seven default, is
/// `26.40`: the shipped constant was **1.8 times too flat**, and
/// [`win_probability`] was correspondingly pulled towards `0.5` everywhere.
/// See the round-eight section of the crate docs, and
/// [`EvalWeights::win_probability_temperature`] for why the number a *search*
/// wants and the number the *likelihood* wants are configured separately even
/// though round eight measured them to be the same.
pub const WIN_PROBABILITY_TEMPERATURE_AGE_I: f64 = 26.40;

/// The same fit restricted to Age II positions. Round eight: `43.75` → `20.34`.
pub const WIN_PROBABILITY_TEMPERATURE_AGE_II: f64 = 20.34;

/// The same fit restricted to Age III positions, where the evaluation is
/// sharpest. Round eight: `25.18` → `15.50`.
pub const WIN_PROBABILITY_TEMPERATURE_AGE_III: f64 = 15.50;

/// The same fit over every position at once, kept for reference: it is what a
/// single flat constant would have been, and the per-age spread above is why
/// [`win_probability`] does not use it. Round eight: `38.61` → `20.12`.
pub const WIN_PROBABILITY_TEMPERATURE_OVERALL: f64 = 20.12;

/// The pre-round-eight constants, kept so [`Config::v7`] can restore the leaf
/// mapping every earlier generation was measured under, bit for bit.
///
/// Indexed by age minus one, exactly like
/// [`EvalWeights::win_probability_temperature`].
pub const WIN_PROBABILITY_TEMPERATURE_V7: [f64; 3] = [47.57, 43.75, 25.18];

/// The calibrated temperature for a position in `age`.
///
/// Ages outside `1..=3` cannot occur — [`duels_core::GameState::age`] only
/// ever reports one of the three — and are read as Age III, the sharpest
/// setting, so a hypothetical fourth age could not accidentally get the
/// flattest curve.
#[inline]
pub fn win_probability_temperature(age: u8) -> f64 {
    match age {
        1 => WIN_PROBABILITY_TEMPERATURE_AGE_I,
        2 => WIN_PROBABILITY_TEMPERATURE_AGE_II,
        _ => WIN_PROBABILITY_TEMPERATURE_AGE_III,
    }
}

/// [`evaluate`]'s victory-point score for `state`, from `me`'s side, mapped
/// onto an estimated win probability in `[0, 1]` through the age-calibrated
/// logistic
///
/// ```text
/// P(me wins) = 1 / (1 + exp(-evaluate(state, me, root) / T(state.age())))
/// ```
///
/// This is the exact mapping `duels-agent-mcts-eval` uses to turn a leaf's
/// evaluation into a value its search can back up, moved here (not
/// duplicated) so any other consumer — a diagnostic tool, a server-side
/// analysis endpoint, a future agent — reads the identical calibration rather
/// than inventing a second one. `mcts-eval` reads it from here; nothing about
/// what it computes changed when it moved.
///
/// # `T` comes from the `Root`, and *may* differ from the module constant
///
/// Since round eight the temperature is
/// [`EvalWeights::win_probability_temperature`], read off the [`Root`]'s
/// configuration rather than from [`win_probability_temperature`] directly.
/// At [`Config::default`] the two agree — by measurement, not by
/// construction — and under an older generation snapshot they do not, which is
/// the whole point: it is what makes a change to the leaf mapping A/B testable
/// against the generation before it. Read that field's docs before assuming
/// the two must agree.
///
/// A caller that wants the *calibration* rather than a *leaf* — a diagnostic,
/// a display — can use [`win_probability_from_value`], which reads the
/// constants and needs no [`Root`]. `duels-server`'s advanced-mode read does.
///
/// Consumes no randomness and is invariant to which hidden-information sample
/// produced `state`, exactly like [`evaluate`] itself: `state`'s only two
/// uses are the score `evaluate` computes and the age `T` is picked from, and
/// both are pure functions of public information — checked alongside
/// `evaluate` itself, for every scenario in
/// `tests/determinization_invariance.rs`, not just once.
///
/// # Known limitation, inherited from where this used to live
///
/// The temperature was fitted on positions each scored against **their own**
/// [`Root`] (one fresh `Root` per decision, as `phased` and this function's
/// direct callers do). A caller that instead prices every position in a
/// search against one `Root` fixed at the tree's own root (as `mcts-eval`
/// does, deliberately, since rebuilding one per node is unaffordable) is
/// scoring a hybrid the fit never saw, and a deep position may be mapped
/// slightly off. This is a known, measured characteristic of that caller, not
/// a defect in this function — see `duels-agent-mcts-eval`'s crate docs for
/// the full account.
pub fn win_probability(state: &GameState, me: Player, root: &Root) -> f64 {
    let t = root
        .config
        .eval
        .win_probability_temperature(state.age().max(1));
    1.0 / (1.0 + (-evaluate(state, me, root) / t).exp())
}

/// The pure calibrated logistic underneath [`win_probability`], for a caller
/// that already has a victory-point-scale number and an age from somewhere
/// other than a single [`evaluate`] call — [`expected_value`]'s
/// chance-averaged result being the motivating case (there is no single
/// post-action `GameState` to hand [`win_probability`] when an action resolves
/// a chance node). Split out so its fixed points, monotonicity and range are
/// also unit-testable without a [`GameState`]/[`Root`] to drive `evaluate`
/// through.
pub fn win_probability_from_value(value: f64, age: u8) -> f64 {
    1.0 / (1.0 + (-value / win_probability_temperature(age)).exp())
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
    let sum = player_value(state, me, root) - player_value(state, me.other(), root)
        + menu::menu_term(state, me, root.age, &root.menu, &root.config.eval.menu);
    // The one place the output scale is applied. Guarded rather than
    // multiplied by `1.0`, so `value_scale = 1.0` is bit-identical to the
    // arithmetic before this knob existed. Deliberately below the rails and
    // the terminal result, which are magnitudes rather than judgements — see
    // [`EvalWeights::value_scale`].
    if root.config.eval.value_scale == 1.0 {
        sum
    } else {
        sum * root.config.eval.value_scale
    }
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
        let sum = player_value(state, me, root) - player_value(state, me.other(), root)
            + menu::menu_term(state, me, root.age, &root.menu, &root.config.eval.menu);
        if root.config.eval.value_scale == 1.0 {
            sum
        } else {
            sum * root.config.eval.value_scale
        }
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
        WonderModel::Flat => e.wonder_potential * terms::wonder_potential(state, p, e),
        WonderModel::Budget => terms::wonder_potential_budget(state, p, &root.wonders),
        WonderModel::Rationed => {
            e.wonder_potential
                * terms::wonder_potential_rationed(state, p, e, root.wonder_p_build[p.index()])
        }
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
    // What the progress tokens this city already holds are worth for the rest
    // of the game, beyond the printed points `breakdown` scores. Root-fixed
    // prices, post-action ownership — see [`terms::TokenTable`].
    let tokens = if e.token_equity == 0.0 {
        0.0
    } else {
        e.token_equity * terms::token_equity(state, p, &root.tokens)
    };
    // A turn in hand, as against round six's projection of one still under a
    // wonder. See [`terms::to_move`].
    let tempo = if e.to_move == 0.0 {
        0.0
    } else {
        e.to_move * terms::to_move(state, p)
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
        + tokens
        + tempo
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

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::data::Science;
    use duels_core::scoring::VictoryKind;
    use duels_core::testing::StateBuilder;
    // `rand` is a dev-dependency only: the engine takes an explicitly seeded
    // `StdRng`, and nothing this crate ships needs one.
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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

    // The counting half of root-fixing — one `Root` per decision, however
    // many candidates it scores — is an assertion about a *caller*, so it
    // lives with the caller:
    // `duels-agent-phased`'s `root_weights_are_built_exactly_once_per_choose`
    // reads `PhasedAgent::root_builds`. What this crate can and does pin is
    // the behavioural half, below: the tables do not move once built, and a
    // candidate is scored against the root's prices and the root's weights.

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
    fn the_menu_pricing_tables_are_root_fixed() {
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

    /// The move this evaluation would pick, ties broken by order rather than
    /// by an RNG draw.
    ///
    /// `duels-agent-phased` is the crate that turns scores into a decision,
    /// and it breaks ties uniformly at random from its own stream; this crate
    /// has no RNG and needs none. The positions below are built so that the
    /// intended move wins outright, so the two agree on all of them.
    fn best_action(state: &GameState, me: Player, root: &Root, legal: &[Action]) -> Action {
        let mut best = (f64::NEG_INFINITY, legal[0]);
        for &action in legal {
            let v = expected_value(state, action, me, root);
            if v > best.0 {
                best = (v, action);
            }
        }
        best.1
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

    /// Rail B, end to end through the evaluation: a one-shield-from-the-
    /// capital opponent with a red card on the table, and one candidate that
    /// takes that card away.
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

        let legal = engine::legal_actions(&st);
        let chosen = best_action(&st, me, &Root::new(&st, me, Config::default()), &legal);
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

    /// The version snapshots are a chain, and every link has to be a distinct
    /// configuration: a snapshot that equals its successor names a round that
    /// changed nothing, and one that equals [`Config::default`] while *not*
    /// being the newest link is a snapshot that has silently drifted.
    ///
    /// [`Config::v8`] is the newest link and is deliberately today's default;
    /// see its documentation for what the next round owes this chain.
    #[test]
    fn the_generation_snapshots_are_a_chain_of_distinct_configurations() {
        let chain = [
            ("v1", Config::v1()),
            ("v2", Config::v2()),
            ("v3", Config::v3()),
            ("v4", Config::v4()),
            ("v5", Config::v5()),
            ("v6", Config::v6()),
            ("v7", Config::v7()),
        ];
        for (i, (name, cfg)) in chain.iter().enumerate() {
            for (later, other) in chain.iter().skip(i + 1) {
                assert_ne!(cfg, other, "{name} and {later} are the same configuration");
            }
        }
        // Deliberately *not* `assert_eq!(Config::v8(), Config::default())`.
        // `v8` is defined as `Config::default()`, so that assertion is a
        // tautology: it cannot fail, and reading it as a guard against a
        // ninth round silently redefining this generation would be a
        // mistake. Nothing in this crate can catch that, because the newest
        // snapshot in this chain is *by construction* whatever the default
        // is (`v1`-`v7` are deltas from it, not literals).
        //
        // The guard that used to work lived with the consumer that had
        // something to lose: `duels-agent-mcts-uct`'s
        // `leaf::tests::the_pinned_generation_reproduces_its_golden_values`
        // held ~50 evaluations captured from `v6()` at the moment its search
        // was measured against it. That agent no longer consumes this crate at
        // all, and no current consumer pins a generation, so nothing
        // downstream enforces the chain today. See [`Config::v9`]'s "contract
        // for the next round".
    }

    // `spec_reports_the_expected_name_and_encoded_params`,
    // `choosing_only_ever_returns_one_of_the_offered_actions` and
    // `a_whole_game_of_self_play_terminates_and_stays_legal` are assertions
    // about an `Agent`, not about an evaluation, and live in
    // `duels-agent-phased` with the agent they are about. `params_string`
    // itself is still pinned from this side, by the identity tests in
    // `tests/`.

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
        // The discount *bites*: it blends the resolved value back towards the
        // pending state's own score by the modelled fraction, so the two must
        // differ. It is deliberately **not** asserted that the discounted
        // value is the *lower* of the two, which is what this test used to
        // claim. That only holds when the destroy's resolved score is above
        // the pending-state reference, and the reference is a distorted
        // quantity by construction — `menu_term` reads the mover as moving
        // again in a pending state, which is the whole reason
        // `PendingModel::Completed` exists. Round seven's re-weighting was
        // enough to flip the sign in this particular hand-built position
        // (plain -8.16, discounted -5.60), which is a fact about the reference
        // rather than about the discount, and is one more reason this knob is
        // off by default and measured at -0.9 / +5.8.
        let d = expected_value(&early, action, Player::One, &discounted);
        let pl = expected_value(&early, action, Player::One, &plain);
        assert_ne!(
            d.to_bits(),
            pl.to_bits(),
            "the discount did not bite on a replaceable destroy: \
             discounted {d}, plain {pl}"
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

    // `win_probability`'s calibration: moved here from `duels-agent-mcts-eval`
    // (which was the only caller until this function existed), not
    // duplicated — these two tests used to live there.

    #[test]
    fn the_temperature_lookup_is_the_calibrated_table() {
        assert_eq!(
            win_probability_temperature(1).to_bits(),
            WIN_PROBABILITY_TEMPERATURE_AGE_I.to_bits()
        );
        assert_eq!(
            win_probability_temperature(2).to_bits(),
            WIN_PROBABILITY_TEMPERATURE_AGE_II.to_bits()
        );
        assert_eq!(
            win_probability_temperature(3).to_bits(),
            WIN_PROBABILITY_TEMPERATURE_AGE_III.to_bits()
        );
        // Age III is the sharpest of the three, which is the finding the
        // per-age lookup exists for.
        assert!(win_probability_temperature(3) < win_probability_temperature(2));
        assert!(win_probability_temperature(2) < win_probability_temperature(1));
        // A flat constant would have been the overall fit, and it is bracketed
        // by the per-age ones.
        assert!(win_probability_temperature(3) < WIN_PROBABILITY_TEMPERATURE_OVERALL);
        assert!(WIN_PROBABILITY_TEMPERATURE_OVERALL < win_probability_temperature(1));
    }

    /// The mapping's three fixed points, plus its monotonicity and its range.
    #[test]
    fn the_sigmoid_maps_victory_points_onto_a_probability() {
        for age in 1..=3u8 {
            assert_eq!(win_probability_from_value(0.0, age), 0.5, "age {age}");
            let t = win_probability_temperature(age);
            // One temperature of advantage is the 73% point, by construction.
            let at_t = win_probability_from_value(t, age);
            assert!((at_t - 0.731_058_6).abs() < 1e-6, "age {age}: {at_t}");
            // Symmetric about a half, and monotone.
            assert!(
                (win_probability_from_value(t, age) + win_probability_from_value(-t, age) - 1.0)
                    .abs()
                    < 1e-12
            );
            let mut last = 0.0;
            for v in [-200.0, -50.0, -5.0, 0.0, 5.0, 50.0, 200.0] {
                let p = win_probability_from_value(v, age);
                assert!(p > last, "age {age}: not monotone at {v}");
                assert!((0.0..=1.0).contains(&p));
                last = p;
            }
        }
        // The same score is worth more in Age III, where the evaluation is
        // sharper.
        assert!(win_probability_from_value(10.0, 3) > win_probability_from_value(10.0, 1));
    }
}
