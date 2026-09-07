//! `duels-agent-mcts-uct`: Monte Carlo Tree Search with UCT selection and
//! **explicit chance nodes**, intended as the project's strongest non-learned
//! baseline and the yardstick for everything that comes after it.
//!
//! # Why chance nodes
//!
//! 7 Wonders Duel is a two-player zero-sum *stochastic* game: the cards
//! behind the face-down slots of the current age are unknown when a move is
//! chosen, and taking a card can uncover them. There is no player-private
//! information — both players always see the same public state — so the game
//! is much simpler than poker, but it is not Go: a plain alternating-move
//! tree would silently pretend the reveals were part of the mover's choice.
//!
//! This agent therefore builds a tree with three kinds of node:
//!
//! - **decision** nodes, one player to move, children = the legal actions,
//!   selected by UCB1;
//! - **chance** nodes, inserted between an action and the position it leads
//!   to whenever the engine says the action resolves randomness, children =
//!   possible reveals, selected **by their real probability**, never by UCB1;
//! - **terminal** nodes, where the [`duels_core::GameResult`] is settled.
//!
//! # How chance is handled, precisely
//!
//! Worth being explicit about, because it bounds how strong the agent can
//! get:
//!
//! 1. **The root is determinized.** `choose` only ever sees an
//!    [`Observation`], so it calls [`Observation::sample_state`] once per
//!    call to get one concrete world consistent with public knowledge.
//! 2. **Reveals inside the tree are *not* taken from that world.** Every
//!    chance node re-draws its outcome from the distribution the engine
//!    computes from public information alone
//!    (`engine::chance_outcomes`/`hidden_info`), and applies it with
//!    `engine::apply_with_outcome`, which rewrites the hidden layout to stay
//!    publicly consistent. So the tree integrates over reveals rather than
//!    committing to the root's guess, and the agent can never exploit
//!    knowledge of a card it should not know.
//! 3. **The draw is exact but not enumerated.** A two-slot reveal has
//!    hundreds of outcomes; the `chance` module draws from that distribution
//!    in O(1) and reports the drawn outcome's exact probability, which is
//!    verified statistically against `engine::chance_outcomes`.
//! 4. **Progressive widening is an approximation.** A chance node that
//!    created a fresh child on every visit would be a perfectly unbiased
//!    estimator of the expectation but would never let the tree grow past it.
//!    By default the number of distinct outcome children grows as
//!    `sqrt(visits)` and further visits re-select an existing child in
//!    proportion to its probability. That reweighting is the one place the
//!    search is not a faithful expectation; set
//!    [`Config::chance_widen_alpha`] to `1.0` with a large
//!    [`Config::chance_widen_c`] to recover the unbiased estimator.
//! 5. **Two sources of randomness are only root-determinized:** the
//!    composition and order of the *next* age's deck, and the four wonders
//!    not yet offered during the draft. Neither is exposed through the
//!    per-action chance API, so they stay fixed for the duration of one
//!    search.
//!
//! # Root ensembling (`Config::root_determinizations`)
//!
//! Point 5 is what [`Config::root_determinizations`] addresses. With `N > 1`,
//! `choose` samples `N` independent worlds, grows a separate tree in each
//! with `1/N` of the budget, and plays the move with the most root visits
//! summed across the trees (see `tree::best_of`). That is the standard
//! Perfect Information Monte Carlo ensemble at the root: no single guess at
//! the hidden information can decide the move on its own, at the price of `N`
//! times fewer samples per tree.
//!
//! ## What it measures
//!
//! Paired, seat-swapped matches against this same agent at `N = 1`, run
//! through `duels-arena`'s `ensemble_lab` example, `+/-` one binomial
//! standard error. The `N = 1` row is a **control**: the identical
//! configuration against itself, which should score 50% and is the honest
//! yardstick for how much of each column is noise.
//!
//! | N | `Nodes(2_000)`, 400 games | `TimeMs(20)`, 200 games |
//! |---|---|---|
//! | 1 (control) | 46.4% +/- 2.5 | 47.5% +/- 3.5 |
//! | 2 | 51.5% +/- 2.5 | 47.5% +/- 3.5 |
//! | 4 | 48.0% +/- 2.5 | 45.0% +/- 3.5 |
//! | 8 | 43.2% +/- 2.5 | 42.0% +/- 3.5 |
//!
//! (Games run in parallel across seeds with the pool capped at 4-6 threads,
//! on a 14-core machine that was also busy with other work. That lowers the
//! *operating point* of a `TimeMs` row — 20 ms buys fewer simulations on a
//! loaded box — but not its fairness: the two sides alternate inside one
//! thread, and the harness prints each side's wall clock per game as the
//! check that they were contended alike.)
//!
//! **No gain at any `N`, at either budget kind.** Every cell sits within
//! about two standard errors of the control, so the honest summary is "no
//! measurable effect at `N = 2`, and a mild loss by `N = 8`" — not a win, and
//! not the disaster a naive reading of "each tree only gets an eighth of the
//! budget" would predict either.
//!
//! The split itself is fair, which is what makes those numbers about the
//! technique rather than about lost work. Under `Budget::Nodes` every `N`
//! spends *exactly* the budget — `ensemble_lab --cost` reports 2000
//! simulations per decision at `N = 1` and at `N = 8` alike — and the
//! measured wall clock per game agrees to within about 1% across the whole
//! sweep. Under `Budget::TimeMs` the same tool reports 32-34k simulations per
//! decision at every `N`, against a repeated `N = 1` baseline that itself
//! wandered between 23k and 32k with the machine's load: the per-slice
//! overhead (one determinization and one tree allocation) is far below the
//! noise, which is what the shared, chained deadlines in `Slices` are for.
//!
//! Four times the budget does not change the answer either: at
//! `Nodes(8_000)` over 200 games, `N = 2` scores 50.0% +/- 3.5 against
//! `N = 1`, with a 49.5% +/- 3.5 control. Dead level, which is the same "no
//! effect" as the `Nodes(2_000)` column read against its own control.
//!
//! The reading: this tree already integrates over every reveal the chance API
//! exposes (point 2 above), so a fresh determinization re-rolls only what the
//! API does not cover — the next age's deal, the undrafted wonders, and the
//! identities a *playout* walks into past the tree's edge. A playout under a
//! kind-level random policy extracts very little from knowing those
//! identities, so there is little bias to average away; meanwhile halving the
//! samples that decide the move actually being made is an immediate,
//! certain cost. Pooling visit counts across trees is itself a small
//! variance reduction, which is presumably why the loss stays as mild as it
//! does.
//!
//! `N = 1` therefore stays the default, and this is an opt-in knob — kept
//! because a measured negative result is worth more than a remembered
//! intuition, and because the same question at a different budget (a
//! one-second move, say) is now one command away:
//!
//! ```text
//! cargo run --release -p duels-arena --example ensemble_lab -- \
//!     --a mcts-uct:dets=4 --b mcts-uct:dets=1 --games 400 --budget nodes:2000
//! ```
//!
//! # Race-aware rollouts (`Config::race`)
//!
//! [`RaceWeights`] biases the *playout* by how far along a win condition the
//! position already is, rather than by facts about the card alone. See its own
//! documentation for the mechanism, the affordability gate it had to clear,
//! and the two tiers it is built from; this section is the strength
//! measurement.
//!
//! It was built for a diagnostic finding. In self-play at `Nodes(2000)` over
//! 400 games, *either* player reached five distinct scientific symbols in only
//! **1.5%** of games — and 67% of those became a scientific-supremacy win. The
//! tree closes a science race fine once it is close; the rollouts almost never
//! build one up, so the tree is rarely shown a position where it is close.
//! (Both numbers reproduced exactly in the control run below: 6/400 exposed,
//! 4 of 6 converted. Military reproduced too, at 48% and 36%.)
//!
//! ## The tuning sweep chose `MEDIUM`, and `MEDIUM` was the wrong answer
//!
//! A sweep on a *separate* seed range (`20001..20050`, 100 games each vs
//! [`RaceWeights::NEUTRAL`], `Nodes(2000)`, `+/-` 5.0) read
//! `mild` 52.0%, `medium` 55.0%, `strong` 47.0% — noise, but unimodal, so
//! `MEDIUM` was committed to before the evaluation ranges were run. It did not
//! transfer:
//!
//! | candidate vs `NEUTRAL`, `Nodes(2000)` | `1..200` | `5001..5200` | `10001..10200` | pooled |
//! |---|---|---|---|---|
//! | [`RaceWeights::MEDIUM`] | 51.75% | 51.1% | — | **51.44% +/- 1.77**, Elo +10.0 [-14.1, +34.0] |
//! | [`RaceWeights::TIER1_ONLY`] | 52.25% | 56.25% | 52.75% | **53.75% +/- 1.44**, Elo +26.1 [+6.4, +45.8] |
//! | `NEUTRAL` vs `NEUTRAL` (control) | 46.25% | — | — | — |
//!
//! `MEDIUM` misses the bar this work was set (53.5% pooled, or SPRT
//! `elo0 = 0` vs `elo1 = 20` accepting H1). **`TIER1_ONLY` clears both**:
//! 53.75% over 1200 games across three disjoint seed ranges, positive on every
//! one of them, SPRT `llr = 3.194` against a `2.944` bound —
//! `AcceptH1`, the first decisive SPRT accept in this crate's history.
//!
//! The control matters here, and it was run on all three of the same ranges
//! rather than one: `NEUTRAL` against itself scores **48.67% +/- 1.44**
//! (46.25 / 50.13 / 49.50). `TIER1_ONLY`'s interval and the control's do not
//! overlap, so the honest reading is "clearly positive", not "marginally
//! positive".
//!
//! At `TimeMs(20)`, 200 games per range on a machine at load 18-48 (see
//! `duels-arena`'s "quiet machine" note — this is indicative, not
//! conclusive): 55.0% on `1..100`, 48.5% on `5001..5100`, pooled **51.75% +/-
//! 2.50**, Elo +12.1, against a 49.5% `NEUTRAL`-vs-`NEUTRAL` control run in the
//! same session. The 3% throughput cost does not eat the gain, which is the
//! whole reason the affordability gate came first.
//!
//! So: **the terminal rails are the entire effect, and the escalating Tier-2
//! tables give half of it back.** Telling a rollout "never walk past a move
//! that ends the game" is worth about +26 Elo; telling it "and lean into races
//! in proportion to how far along they are" costs about 15 of that. The
//! likeliest reading is that Tier 2 makes playouts commit to races the
//! position does not support, so a leaf's value becomes optimistic about race
//! lines *in general* rather than accurate about the ones that are real —
//! while a rail only ever fires on a move that is already decisive, where
//! there is nothing to be wrong about.
//!
//! ## The mechanism is military, and it is not the one this was built for
//!
//! `TIER1_ONLY` vs `NEUTRAL`, pooled over the 1200 games:
//!
//! | | `TIER1_ONLY` | `NEUTRAL` |
//! |---|---|---|
//! | wins by military supremacy | **138** | 68 |
//! | wins by scientific supremacy | 9 | 6 |
//! | wins by civilian score | 491 | 469 |
//! | wins by tiebreak | 7 | 12 |
//! | **total** | **645** | 555 |
//!
//! The number of games decided on the conflict track barely moves (206 here,
//! 207 in the control) — what moves is **who wins them**, from an even 99-108
//! split in the control to 138-68. That `+70` is most of the `+90` overall
//! margin. The rails are not creating military races; they are stopping the
//! playout from walking past the move that closes or breaks one.
//!
//! ## The science hypothesis did not survive contact
//!
//! Science was the strong prior for this work: a race the search finishes well
//! (67% conversion) but is almost never shown (1.5% exposure). If the rollout
//! policy were the bottleneck, rails should have raised exposure. Measured in
//! true self-play (both sides configured alike, 1200 games each over the same
//! three ranges), it did not:
//!
//! | self-play, `Nodes(2000)`, 1200 games | `NEUTRAL` vs `NEUTRAL` | `TIER1_ONLY` vs `TIER1_ONLY` |
//! |---|---|---|
//! | science exposure (either side reaches 5 symbols) | 2.50% (30) | 2.67% (32) |
//! | of those, converted to scientific supremacy | 46.7% (14) | 53.1% (17) |
//! | military exposure | 47.5% (570) | 44.0% (528) |
//! | of those, converted to military supremacy | 36.3% (207) | 34.5% (182) |
//!
//! Both science columns move by less than the control's own range-to-range
//! spread (its three ranges read 1.5%, 4.25%, 1.75% exposure — which is also
//! why the original 1.5% diagnostic, taken from a single range, overstated how
//! rare this is). On `n = 30`, a conversion rate has a standard error near 9
//! points. **Nothing here is a measurable science improvement.**
//!
//! What the self-play table does show is mutual denial: with rails on both
//! sides, military-decided games fall from 207 to 182 and exposure from 47.5%
//! to 44.0%. Each side is now taking the closing red card away from the other,
//! which is the "high exposure has value even without conversion" effect a
//! domain read predicted — visible here as races that get shut down rather
//! than won.
//!
//! ## Ladder sanity: nothing regressed
//!
//! 200 games each at `Nodes(2000)`, seeds `1..100`:
//!
//! | opponent | `TIER1_ONLY` | `NEUTRAL` |
//! |---|---|---|
//! | `random` | 200/200 | 200/200 |
//! | `greedy` | 200/200 | 200/200 |
//! | `greedy-ev` | 200/200 | 199/200 |
//! | `alphabeta` | 79.5% (+234 Elo) | 78.5% (+226 Elo) |
//!
//! ## The tree prior still buys nothing, even with a rollout that can value a
//! race
//!
//! [`PriorMode::ExpansionOrder`] reproducibly steers visits towards races and
//! reproducibly fails to convert that into Elo (the section below has the
//! numbers). One reading of that was that the *rollout* was the missing half:
//! a tree that looks at race lines learns nothing if the playout underneath
//! scores them at random. With the rails supplying that half, over 400 games
//! at `Nodes(2000)` on `1..200`:
//!
//! | | score |
//! |---|---|
//! | `race=tier1,prior=expansion_order` vs `race=tier1` | **50.0%** (200-200) |
//!
//! Dead level, against a control that reads 48.67%. The two ideas do not
//! compose; the prior's cost is still not repaid.
//!
//! ## Verdict: strong on strength, negative on the hypothesis — and so still
//! not the default
//!
//! [`RaceWeights::NEUTRAL`] stays [`Config::default`]. This work was
//! pre-registered against four accept criteria, and three of them pass
//! decisively for `TIER1_ONLY` — pooled `Nodes` score, `TimeMs` score, ladder
//! sanity. The fourth, "science-supremacy exposure or conversion measurably
//! improved in self-play", **fails**: the table above moves less than the
//! control's own variance. The criterion was not a formality; it was the
//! hypothesis. Promoting a default on the strength number alone, after the
//! stated reason for the change did not materialise, is how a codebase
//! accumulates changes nobody can later explain.
//!
//! That said, the strength evidence is the strongest any change in this crate
//! has produced — `+26.1` Elo with an interval clear of zero, an SPRT
//! `AcceptH1`, positive on three of three disjoint seed ranges, positive at a
//! wall-clock budget too, for 3% throughput and no regression anywhere on the
//! ladder. It is one line from being the default:
//!
//! ```text
//! race: RaceWeights::TIER1_ONLY,   // in Config::default()
//! ```
//!
//! and the case for doing so is a judgement about how much a mechanism
//! explanation is worth, not a gap in the measurement.
//!
//! ## Reproducing
//!
//! ```text
//! cargo run --release -p duels-arena -- match \
//!     --agent-a mcts-uct:race=tier1 --agent-b mcts-uct:race=neutral \
//!     --games 400 --budget nodes:2000 --seed 1 --sprt-elo0 0 --sprt-elo1 20
//! cargo run --release --example rollout_bench -p duels-agent-mcts-uct -- 0 100
//! ```
//!
//! # Strategy priors (`Config::prior`)
//!
//! [`Config::prior`] lets `duels-strategy`'s policy layer steer this tree.
//! On a decision node's **first expansion**, one
//! [`duels_strategy::Stance`] is computed for that node and its actions are
//! ranked by [`duels_strategy::action_prior`], highest first
//! ([`PriorMode::ExpansionOrder`]); [`PriorMode::ProgressiveBias`] also keeps
//! the normalised weights and adds `weight × prior / (visits + 1)` to each
//! child's UCB1 score. [`PriorMode::None`], the default, is bit-for-bit the
//! agent that existed before the option (see
//! `tests::prior_none_is_the_pre_prior_agent_move_for_move` and
//! `tree::tests::prior_none_grows_the_same_tree_as_the_pre_prior_search`,
//! which check the *move* and the *whole arena* against verbatim copies of the
//! pre-prior `expand`/`select_ucb1`/`simulate`).
//!
//! The design is shaped by one measured cost: a stance plus a full slate of
//! priors is roughly 29% of a rollout, which is unaffordable per *simulation*
//! and affordable per *node*. So it is paid once, on first expansion, under an
//! `expanded == 0` guard — never for a node a simulation only ever played out
//! from, and never for a node with a single legal action. The tree counts its
//! own consultations so that is checkable: on a real mid-game position a
//! 2000-simulation search allocates 2788 nodes and consults the strategy layer
//! 583 times, 0.29 per simulation, because only about a fifth of the nodes it
//! creates are ever revisited and expanded. At a fixed `Nodes`
//! budget that comes to **+6-8% wall clock**: 1265 vs 1195 ms/game over 40
//! paired games and 688 vs 636 ms/game over another 100, both at
//! `Nodes(2000)`, and the same ratio falls out of the budget-equivalence runs
//! below, where the half-budget prior side costs 0.537 of its opponent's time
//! against a no-prior control's 0.500. That overhead is the whole story of the
//! `TimeMs` row.
//!
//! ## What it measures: a gain too small to accept, and it does not reproduce
//!
//! `ExpansionOrder` against `None`, paired and seat-swapped through
//! `duels-arena match`, at `Nodes(2000)` over four **disjoint** seed ranges of
//! 400 games each. A node budget is not a wall-clock quantity, so none of this
//! is load-sensitive — which matters, because these ran on a machine whose
//! load average wandered between 6 and 150.
//!
//! | seed range | score for `ExpansionOrder` |
//! |---|---|
//! | `1..200` | 52.8% +/- 2.5 |
//! | `5001..5200` | 53.9% +/- 2.5 |
//! | `10001..10200` | 48.3% +/- 2.5 |
//! | `15001..15200` | 51.9% +/- 2.5 |
//! | **pooled, 1600 games** | **51.7% +/- 1.25**, Elo **+11.7** [-5.3, +28.8] |
//!
//! The bar this work was set — 55% at `Nodes(2000)`, reproduced on a second
//! seed range — is **not met**, and the reason is written across the table:
//! the first two ranges look like a 53% effect, the third is a *loss*. SPRT
//! (`elo0 = 0` vs `elo1 = 20`, `alpha = beta = 0.05`) still reads `Continue`
//! after 1600 games, and the Elo interval still contains zero. The honest
//! summary is "somewhere between nothing and +25 Elo, for +6-8% time".
//!
//! Hold *time* fixed instead and that trade goes underwater, which is what a
//! 6-8% throughput cost against a sub-noise gain has to do. `TimeMs(20)`, 200
//! games per seed range, run one match at a time on a four-thread pool with the
//! machine's load average between 8 and 12 (see `duels-arena`'s "Benchmarking
//! on a quiet machine"):
//!
//! | seed range | score for `ExpansionOrder` |
//! |---|---|
//! | `1..100` | 47.5% +/- 3.5 |
//! | `5001..5100` | 43.0% +/- 3.5 |
//! | **pooled, 400 games** | **45.3% +/- 2.5**, Elo **-33.0** [-67.2, +1.1] |
//! | `None` vs `None` (control, 200 games) | 51.3% +/- 3.5 |
//!
//! The control is the yardstick that says the 45.3% is about the technique and
//! not about the harness: the identical configuration against itself scores
//! 51.3%, a third of a standard error from even. The `TimeMs` bar for this work
//! was 53%; the measurement is not merely short of it but on the wrong side of
//! 50, consistently, across both ranges.
//!
//! ## The mechanism it was built for does fire, loudly
//!
//! What the win rate hides, the victory-kind breakdown shows. The strategy
//! layer exists to make a search see races that close three moves out, and it
//! demonstrably does — the prior converts wins that were being banked at
//! scoring into wins taken outright on the conflict track:
//!
//! | opponent (200 games, `Nodes(2000)`) | wins by military supremacy, `None` -> `ExpansionOrder` |
//! |---|---|
//! | `random` | 13 -> **52** |
//! | `greedy` | 72 -> **147** |
//! | `greedy-ev` | 58 -> **141** |
//! | itself (1600 games, pooled) | 120 -> **165** |
//!
//! Against `greedy` the prior-enabled agent takes 147 of its 200 wins by
//! military supremacy where the baseline took 72 — and both win 200 of 200, so
//! none of that reaches the scoreboard. In the self-play match the military
//! split is `165-120` over the 285 games decided that way (57.9% +/- 2.9), and
//! that +45 is essentially the *entire* +54 win margin: the prior is not
//! playing better points, it is closing more military races.
//!
//! That is the finding worth keeping. `duels-strategy` does what it was built
//! to do; what it does not do is convert into net wins, because the games it
//! now takes on the track were largely games it was already winning on points.
//!
//! ## Progressive bias is worse than plain ordering
//!
//! Measured, not assumed. A weight sweep on a *separate* tuning seed range
//! (`20001..20051`, 100 games each vs `None`) read 53.5 / 57.0 / 50.0 / 53.0 /
//! 40.0% at `weight` 2 / 5 / 10 / 20 / 40 — noise at +/- 5.0 apart from the
//! collapse at 40. The apparent best, `weight = 5`, then failed to transfer to
//! the evaluation ranges: 49.5% and 46.0% over 400 games each, pooled **47.75%
//! +/- 1.77**, Elo -15.6, and SPRT `AcceptH0` — the one decisive SPRT verdict
//! in this whole investigation, and it is against. Head to head with
//! `ExpansionOrder` over 400 games it scores 48.5%.
//!
//! Reading: at a UCB1 exploration constant of 1.0 the bias term is either too
//! small to matter or large enough to override the exploration bonus at
//! exactly the nodes that most need it, and there is no window in between.
//! Ordering, which never overrides anything and only decides which move gets
//! the first look, does not have that failure mode.
//!
//! ## Budget equivalence: the prior buys no search
//!
//! The cleanest way to size a search improvement is to ask how much budget it
//! is worth. Against `None` at `Nodes(2000)`, over 400 games each:
//!
//! | half-budget side | score vs `None` at `Nodes(2000)` |
//! |---|---|
//! | `ExpansionOrder` at `Nodes(1000)` | 38.8% +/- 2.4 |
//! | `None` at `Nodes(1000)` (control) | 40.0% +/- 2.4 |
//!
//! Halving the node budget costs about 10 points of score (roughly 70 Elo), and
//! the prior recovers **none** of it — 38.8% against a 40.0% control is, if
//! anything, slightly the wrong way. So the +11.7 Elo the pooled self-play
//! measurement suggests is worth well under a fifth of a budget doubling, and
//! at half budget it is worth nothing at all.
//!
//! ## Ladder sanity: nothing regressed, and the gap to `alphabeta` did not move
//!
//! 200 games each at `Nodes(2000)`, `ExpansionOrder` / `None`:
//!
//! | opponent | `ExpansionOrder` | `None` |
//! |---|---|---|
//! | `random` | 200/200 | 200/200 |
//! | `greedy` | 200/200 | 200/200 |
//! | `greedy-ev` | 200/200 | 199/200 |
//! | `alphabeta` | 79.5% (+234 Elo) | 78.5% (+226 Elo) |
//!
//! Nothing broke, and the one row with room to move — `alphabeta` — did not.
//!
//! ## Verdict
//!
//! [`PriorMode::None`] stays the default. The change ships as an opt-in
//! [`Config`] option because the evidence does not support making it standard:
//! a +11.7 Elo point estimate whose interval contains zero, which reversed sign
//! on one of four seed ranges, and which turns into a measured -33 Elo once the
//! 6-8% throughput cost has to be paid out of a wall clock, is not a champion.
//! It is kept, tested and documented because the victory-kind data is a real
//! result about the strategy layer rather than about this agent, and because
//! the next question — does the same prior pay at a budget where the search is
//! too shallow to find the race by itself? — is one command away:
//!
//! ```text
//! cargo run --release -p duels-arena -- match \
//!     --agent-a mcts-uct:prior=expansion_order --agent-b mcts-uct:prior=none \
//!     --games 400 --budget nodes:2000 --sprt-elo0 0 --sprt-elo1 20
//! ```
//!
//! # Leaf values (`Config::leaf`)
//!
//! [`LeafValue`] decides what a freshly added leaf is worth: the playout this
//! crate has always used, `duels-eval`'s hand-crafted evaluation through a
//! calibrated sigmoid, a truncated playout that ends in one, or a mixture.
//! See the `leaf` module for the mechanism — the per-age temperature
//! calibration, where the one `duels_eval::Root` is built, why the perspective
//! is always Player One, and the algebra relating [`LeafValue::Blend`]'s
//! weight to the exploration constant. This section is the measurement.
//!
//! `LeafValue::Rollout` is the default and is bit-for-bit the agent that
//! existed before the option, down to building no `duels_eval::Root` at all
//! (`tests::leaf_rollout_is_the_pre_leaf_agent_move_for_move`,
//! `tree::tests::leaf_rollout_grows_the_same_tree_as_the_pre_leaf_search`,
//! `tree::tests::the_rollout_leaf_builds_no_evaluation_root`; the move digests
//! of whole seeded self-play games at `Nodes(300)` and `Nodes(2000)` were also
//! checked against a build of the parent commit and agree).
//!
//! ## What each variant costs
//!
//! `examples/leaf_bench.rs`, 30 positions at `Nodes(2000)` — a node budget, so
//! the *work* is exactly fixed (52,000 simulations per column) and only the
//! elapsed time moves. Run on a machine that was not quiet, so read the ratio
//! column and not the absolute microseconds:
//!
//! | leaf | µs/simulation | throughput vs default |
//! |---|---|---|
//! | `Rollout` (default) | 18.84 | 1.00x |
//! | `Static` | 1.55 | **12.15x** |
//! | `Truncated { plies: 4 }` | 2.57 | 7.33x |
//! | `Truncated { plies: 8 }` | 3.83 | 4.92x |
//! | `Truncated { plies: 16 }` | 5.60 | 3.37x |
//! | `Blend { weight: 0.3 }` | 18.83 | 1.00x |
//! | `Blend { weight: 0.5 }` | 18.66 | 1.01x |
//!
//! Read as a decomposition: if a whole simulation is 18.84 µs and the same
//! simulation with the playout replaced by one cached-`Root` evaluation is
//! 1.55 µs, then **the playout is about 92% of what a simulation costs** and
//! everything else — descent, expansion, backpropagation, the evaluation
//! itself — is the remaining 1.5 µs. The truncated rows interpolate between
//! the two about as linearly as that implies.
//!
//! A blend measures at parity with a plain rollout rather than the few percent
//! *slower* it must strictly be — it does the playout and then a 1.5 µs
//! evaluation on top. That extra is about 8% of a simulation, which is inside
//! this bench's run-to-run spread on a machine that is not quiet, so read the
//! blend rows as "no measurable throughput cost" rather than as free. The
//! consequence for a wall-clock budget is the same either way, and the
//! `TimeMs` rows below are the actual test of it.
//!
//! ## The tuning sweep (a separate seed range, and **not** evidence)
//!
//! `20001..20151`, 300 games each against default `mcts-uct` at `Nodes(2000)`,
//! `+/-` about 2.9. Kept for the record and for what it says about the shape of
//! the family, not as a strength claim — the ranges below are the evidence:
//!
//! | candidate | score | Elo |
//! |---|---|---|
//! | `leaf=static` | 27.0% | -170.7 |
//! | `leaf=static,c=0.5` | 35.3% | -104.6 |
//! | `leaf=trunc:4` | 37.0% | -92.1 |
//! | `leaf=trunc:8` | 40.7% | -65.4 |
//! | `leaf=trunc:8,c=0.5` | 46.7% | -23.1 |
//! | `leaf=trunc:16` | 47.0% | -20.8 |
//! | `leaf=blend:0.9,c=0.1` | 47.3% | -18.5 |
//! | `leaf=blend:0.3` | 55.0% | +34.7 |
//! | `leaf=trunc:16,c=0.5` | 55.3% | +37.1 |
//! | `leaf=blend:0.5` | 56.3% | +46.4 |
//! | `leaf=blend:0.5,c=0.3` | 57.5% | +52.3 |
//! | `leaf=blend:0.3,c=0.7` | 58.0% | +55.9 |
//! | `leaf=blend:0.8,c=0.2` | 58.7% | +60.6 |
//! | `leaf=blend:0.3,c=0.5` | 60.3% | +72.6 |
//! | `leaf=blend:0.7,c=0.5` | 61.0% | +77.4 |
//! | `leaf=blend:0.7,c=0.2` | 62.3% | +87.2 |
//! | `leaf=blend:0.6,c=0.4` | 62.7% | +90.9 |
//! | **`leaf=blend:0.5,c=0.5`** | **63.7%** | **+97.1** |
//! | `leaf=blend:0.7,c=0.3` | 65.0% | +107.2 |
//! | `c=0.5` alone | 54.0% | +27.8 |
//! | control (default vs default) | 52.0% | +13.9 |
//!
//! Three things to read off it. **A pure static leaf is much weaker than the
//! playout it replaces** at a fixed node count — which is the project's
//! standing prior, holding up. The family is *ordered*: the more playout is
//! left in the leaf, the better, until the static term is gone entirely (at
//! `blend:0.9` the playout is too diluted and the gain is gone again). And
//! the good region is a broad **ridge running roughly along `c = 1 - weight`**
//! — every candidate from `blend:0.5,c=0.5` to `blend:0.7,c=0.3` scores
//! between 62% and 65%, which at `+/-` 2.9 is one indistinguishable plateau
//! rather than a peak. That is the direction [`LeafValue::Blend`]'s rescaling
//! algebra predicts, and it is as much as a 300-game sweep can confirm: at
//! `weight = 0.3` the "matching" `c = 0.7` (+55.9) actually scored *below* the
//! unmatched `c = 0.5` (+72.6), so the ridge's exact ridgeline is inside this
//! sweep's noise.
//!
//! ### What a *pure* static leaf actually gets wrong
//!
//! Worth recording, because it is the sharpest diagnostic in this whole
//! investigation. Victory kinds for `leaf=static` in that sweep (300 games):
//!
//! | | `leaf=static` | default |
//! |---|---|---|
//! | wins by military supremacy | **1** | 37 |
//! | wins by scientific supremacy | **28** | 2 |
//! | wins by civilian score | 51 | 177 |
//!
//! A static leaf wins by military supremacy **once in 81 wins** while
//! conceding 37, and wins by *scientific* supremacy fourteen times more often
//! than the agent it replaced. It is not uniformly blind: it over-values
//! science and under-sees military. The `duels-eval` terms give a position
//! credit for accumulated scientific symbols in a way the search can then go
//! and collect, whereas a military race is a *tempo* fact about the next few
//! moves that only a playout walking those moves discovers — and `Root`'s
//! military smoothing is fixed at the search root, so it cannot even move as
//! the leaf gets deeper (see `leaf`'s note on stale calibration).
//!
//! This is the concrete form of `CLAUDE.md`'s prior that win-condition
//! awareness belongs in the search policy rather than the evaluation, and it
//! is why the blend works: keeping half a playout keeps the military sight
//! that the evaluation has no way to supply.
//!
//! Two candidates were carried forward: the sweep's nominal maximum
//! (`blend:0.7,c=0.3`) and the middle of the plateau (`blend:0.5,c=0.5`).
//! **The maximum did not survive** — see the next section. Reporting the
//! sweep's argmax as the answer would have shipped the weaker of the two, and
//! this is the clearest illustration in this crate of why the sweep range is
//! quarantined from the evidence ranges.
//!
//! ## What it measures: `+89` Elo, on three disjoint ranges
//!
//! 1,200 games per range at `Nodes(2000)`, paired and seat-swapped, against
//! **today's actual default** (verified, not assumed: `c=1.000`,
//! `race=neutral`, `prior=none`, `dets=1`, `leaf=rollout`). `+/-` is one
//! binomial standard error:
//!
//! | arm | `1..600` | `5001..5600` | `10001..10600` | pooled (3,600 games) | Elo |
//! |---|---|---|---|---|---|
//! | **`leaf=blend:0.5,c=0.5`** | 64.42% | 62.04% | 61.21% | **62.56% +/- 0.81** | **+89.2 [+77.5, +101.0]** |
//! | `leaf=blend:0.7,c=0.3` (the sweep's argmax) | 61.92% | 61.88% | 58.92% | 60.90% +/- 0.81 | +77.0 [+65.4, +88.7] |
//! | `leaf=blend:0.5` (`c` unchanged) | 60.54% | 59.29% | 57.42% | 59.08% +/- 0.82 | +63.8 [+52.3, +75.4] |
//! | `c=0.5` alone (attribution control) | 49.12% | 51.46% | 49.71% | 50.10% +/- 0.83 | +0.7 [-10.7, +12.0] |
//! | `c=0.3` alone (attribution control) | 37.00% | 35.21% | 35.75% | 35.99% +/- 0.80 | -100.1 [-112.0, -88.3] |
//! | default vs default (noise floor) | 48.96% | 48.25% | 49.83% | 49.01% +/- 0.83 | -6.9 [-18.2, +4.5] |
//!
//! Every range is positive for all three blend arms, SPRT (`elo0 = 0` vs
//! `elo1 = 20`) reads `AcceptH1` on every one of their nine range-runs
//! (`llr` 10.3 to 17.9 against a 2.944 bound), and the intervals are nowhere
//! near the control's. This is by a wide margin the largest effect measured in
//! this crate: the previous best, the terminal rails, was `+26` Elo.
//!
//! **The exploration constant is not the effect.** `c = 0.5` on its own scores
//! 50.10% over the same 3,600 games — indistinguishable from the noise floor.
//! It is worth about `+25` Elo *in combination* with the blend (62.56% against
//! 59.08%), which is the direction [`LeafValue::Blend`]'s rescaling argument
//! predicts: at `weight = 0.5` the reward's spread is halved, so the
//! exploration bonus has to be halved with it to leave the balance where it
//! was tuned.
//!
//! `c = 0.3` makes that argument much more sharply, which is why it is in the
//! table. On its own it is a **disaster** — `-100` Elo, one of the largest
//! negatives ever measured here — and yet `blend:0.7,c=0.3`, which contains
//! it, is `+77`. A knob worth `-100` alone and `+77` in combination is not
//! plausibly an independent contribution; it is the rescaling the blended
//! reward requires.
//!
//! ## Why the sweep's argmax lost, and what it says about the mechanism
//!
//! `blend:0.7,c=0.3` won the 300-game sweep (+107 against +97) and then
//! finished 12 Elo *behind* `blend:0.5,c=0.5` over 3,600, losing on all three
//! ranges. Pooled victory kinds say why, and it is not noise:
//!
//! | pooled, 3,600 games | wins by military | wins by civilian score |
//! |---|---|---|
//! | `blend:0.5,c=0.5` vs default | **342** - 288 | 1,815 - 992 |
//! | `blend:0.7,c=0.3` vs default | 170 - **323** | 1,911 - 1,036 |
//!
//! At `weight = 0.7` the search gets *better* at city quality (1,911 civilian
//! wins, more than the 0.5 blend manages) and **loses the military race
//! outright** — 170 military wins against the default's 323, having been ahead
//! 342-288 at `weight = 0.5`. That is `CLAUDE.md`'s standing prior showing up
//! as a measurement: a static evaluation cannot see a race developing three
//! moves out, so diluting the playout past about half trades away exactly the
//! thing the playout was providing. The blend weight is not a free knob to
//! push towards the evaluation; it is the balance between two different kinds
//! of sight.
//!
//! ## Where the wins come from: points, not races
//!
//! Pooled victory kinds over the same 3,600 games, `leaf=blend:0.5,c=0.5`
//! against the default:
//!
//! | | blend | default |
//! |---|---|---|
//! | wins by civilian score | **1,815** | 992 |
//! | wins by military supremacy | 342 | 288 |
//! | wins by scientific supremacy | 62 | 34 |
//! | wins by tiebreak | 31 | 32 |
//!
//! `+823` of the `+904` win margin is **civilian score**. That matters because
//! the other mechanism this crate ships — [`RaceWeights::TIER1_ONLY`]'s
//! terminal rails — is *entirely* military (`138-68` there, with the number of
//! military-decided games unmoved). These are not the same effect wearing two
//! hats: the rails stop a playout walking past a decisive move, and the blend
//! makes the search better at the long game of city quality, which is where
//! `duels-eval`'s terms actually live.
//!
//! The composition test says the same thing, and more sharply. With
//! `race=tier1` on **both** sides, 1,200 games on each of two ranges:
//!
//! | | `1..600` | `5001..5600` | pooled (2,400) | Elo |
//! |---|---|---|---|---|
//! | `leaf=blend:0.5,c=0.5,race=tier1` vs `race=tier1` | 62.79% | 65.46% | **64.12% +/- 0.98** | **+100.9 [+86.6, +115.6]** |
//! | `race=tier1` vs `race=tier1` (control) | 51.12% | 50.75% | 50.94% +/- 1.02 | +6.5 [-7.4, +20.4] |
//!
//! `+100.9` with the rails on both sides, against `+89.2` with them nowhere:
//! the two mechanisms **add**, and if anything the blend is worth slightly
//! *more* once the rails are present.
//!
//! The victory kinds explain why they cannot be the same effect. With the
//! rails on both sides the blend's military edge disappears — 191 military
//! wins against 175, essentially level, where without rails it was 342-288 —
//! while its civilian margin is undiminished (1,293 against 652). The rails
//! were already supplying the military tempo sight, so the blend stops needing
//! to; what it adds on top is entirely city quality. Two mechanisms, two
//! win conditions, one addition.
//!
//! ## Budget equivalence: worth more than a doubling
//!
//! The cleanest way to size a search improvement, and the framing the terminal
//! rails were reported in. 400 games each, `1..201`, candidate at
//! `Nodes(1000)` against the default at `Nodes(2000)`:
//!
//! | half-budget side | score vs default at `Nodes(2000)` | ms/game, half-budget side vs full |
//! |---|---|---|
//! | `leaf=blend:0.5,c=0.5` at `Nodes(1000)` | **55.5% +/- 2.5** | 437 vs 782 (56%) |
//! | `leaf=rollout` at `Nodes(1000)` (control) | 40.0% +/- 2.4 | 408 vs 817 (50%) |
//!
//! Halving the node budget costs the default 10 points of score; the blend at
//! *half* the budget **beats** the full-budget default outright. So the leaf
//! value is worth more than a doubling of search.
//!
//! And it gets there on 56% of the opponent's wall clock, against the
//! control's 50% — i.e. the blend's own throughput cost shows up here as
//! about six points of extra wall clock for half the nodes, nothing like
//! enough to consume a 15-point score advantage. That is the first hint of
//! what the `TimeMs` rows below say, and this is the framing the terminal
//! rails were reported in too, so the two are directly comparable.
//!
//! ## Ladder: nothing regressed, and the gap to `phased` widened
//!
//! 400 games each at `Nodes(2000)`, seeds `1..200`:
//!
//! | opponent | `leaf=blend:0.5,c=0.5` | default |
//! |---|---|---|
//! | `greedy-ev` | **400/400** (+1161 Elo) | 399/400 (+970 Elo) |
//! | `phased` | **89.25%** (+365.9 Elo) | 80.13% (+241.4 Elo) |
//! | `alphabeta` | **84.50%** (+293.5 Elo) | 74.88% (+189.1 Elo) |
//!
//! Nothing regressed, and the margin widened on all three.
//!
//! The `phased` row was the pre-registered red flag, and it is the one to read
//! first: the blend scores its leaves with `phased`'s *own* evaluation, so if
//! it beat `phased` by **less** than the plain default does, that would point
//! at something wrong in the integration — an evaluation being read with the
//! wrong sign, a stale pricing context, a leaf value that is really just
//! noise — rather than at a mechanism that merely fails to help. It beats
//! `phased` by nine points more than the default does, which is the opposite
//! of that failure signature.
//!
//! ## At a wall-clock budget
//!
//! The test this crate has been burned by twice: a change that wins at a fixed
//! node count can lose at a fixed clock if it costs more per unit of work
//! (`Config::prior` is the cautionary tale — a `+11.7` point estimate at
//! `Nodes` measured `-33` at `TimeMs`). 400 games per range, **one match at a
//! time with `RAYON_NUM_THREADS=1`**, so each game gets a whole core and the
//! per-decision work is production-like; nothing else was running.
//!
//! | budget | `1..200` | `5001..5200` | pooled (800) | Elo | control (default vs default) |
//! |---|---|---|---|---|---|
//! | `TimeMs(20)` | 67.13% | 62.25% | **64.69% +/- 1.69** | **+105.2 [+80.5, +130.9]** | 46.25%, -26.0 |
//! | `TimeMs(100)` | 65.25% | 60.75% | **63.00% +/- 1.71** | **+92.5 [+67.9, +117.9]** | 48.50%, -10.4 |
//!
//! `AcceptH1` on all four range-runs. **The gain does not merely survive a
//! wall-clock budget, it grows**: `+105` at `TimeMs(20)` and `+93` at
//! `TimeMs(100)`, against `+89` at `Nodes(2000)`.
//!
//! That direction is the expected one rather than a surprise, and the cost
//! table is why. A blend has no measurable throughput cost — it does the same
//! playout and adds a ~1.5 µs evaluation — so a wall-clock budget buys it
//! essentially the same number of simulations it buys the default, and the
//! leaf-value advantage transfers intact. What is left is a budget effect:
//! `TimeMs(20)` buys roughly a thousand simulations, which is the
//! `Nodes(1000)` regime where the budget-equivalence table already showed the
//! blend at its most valuable. A better leaf value is worth more when there
//! are fewer leaves to average over — which is also why `TimeMs(100)`, at
//! roughly five thousand simulations, lands slightly *below* `TimeMs(20)` and
//! slightly above `Nodes(2000)`. The whole family of budgets is consistent:
//! the effect is large everywhere and largest where search is scarcest.
//!
//! This is the one section that has to be read with `CLAUDE.md`'s load
//! warning in mind, so to be explicit about the conditions: 2,400 games of
//! wall-clock measurement, each of the six matches run to completion on its
//! own with `RAYON_NUM_THREADS=1` and no other work on the machine,
//! `TimeMs(100)` averaging 6.8 seconds per game. These are not
//! small-sample indicative numbers.
//!
//! Note the controls: `-26.0` at `TimeMs(20)` and `-10.4` at `TimeMs(100)`,
//! against `-6.9` at `Nodes(2000)`. A wall-clock noise floor is genuinely
//! wider, and wider still at the shorter budget where a scheduling hiccup is
//! a larger fraction of a decision — which is the reason `CLAUDE.md` insists
//! on running these one at a time. Both are nowhere near the candidate's
//! interval: the closest approach is the `TimeMs(20)` control's upper bound
//! against that budget's lower bound, and they are 106 points apart.
//!
//! ## Verdict: it clears every criterion, and it is still not the default
//!
//! [`LeafValue::Rollout`] stays [`Config::default`], and this is a deliberate
//! call rather than an oversight — the same call, for the same reason, that
//! [`RaceWeights::TIER1_ONLY`] got.
//!
//! The strength evidence is not in question. It is the largest effect ever
//! measured in this crate by a factor of three: `+89.2` Elo pooled over 3,600
//! games, positive on three of three disjoint seed ranges, `AcceptH1` on every
//! range-run, an interval nowhere near the noise floor's, *larger* at both
//! wall-clock budgets than at the node budget (`+105` and `+93`), worth more
//! than a doubling of the node budget, no ladder regression anywhere, and
//! additive with the one other mechanism it could plausibly have
//! overlapped with. Every criterion this work was set is cleared, and the
//! `TimeMs` criterion — the one that has reversed this crate's conclusions
//! before — is cleared by the largest margin of any of them. The
//! hypothesis this work was built to test — *can `duels-eval`'s hand-crafted
//! evaluation serve as a leaf value in this search?* — is answered, and the
//! mechanism is identified rather than merely asserted: the gain is civilian
//! score, which is where `duels-eval`'s terms live, and pushing the weight
//! past a half trades away the playout's sight of military races.
//!
//! What stops it being a one-line default change here is that it is **not one
//! line**. The measured candidate changes [`Config::leaf`] *and*
//! [`Config::exploration`], because a blended reward has half the spread a
//! Bernoulli playout does and the exploration constant has to be rescaled to
//! match (see [`LeafValue::Blend`]). Flipping the default therefore moves this
//! crate's tuned `c`, which every other knob in `Config` was tuned against —
//! the rollout weights, the race tables, the widening constants, the prior
//! sweep — and it moves `leaderboard::CHAMPION`, the committed
//! `arena/leaderboard.*`, and what `duels-server` serves. That is a change
//! worth its own reviewable PR with the ladder refitted around the new `c`,
//! not a side effect of the PR that measured the leaf value.
//!
//! So the recommendation is explicit and on the record: **promote this**, as
//! its own change, as
//!
//! ```text
//! leaf: LeafValue::Blend { weight: 0.5 },   // in Config::default()
//! exploration: 0.5,                         // and its matching rescale
//! ```
//!
//! and re-run the ladder at the new default before the leaderboard is
//! believed. Everything needed to make that decision is in the tables above.
//!
//! ## Reproducing
//!
//! ```text
//! cargo run --release -p duels-arena -- match \
//!     --agent-a mcts-uct:leaf=blend:0.5,c=0.5 --agent-b mcts-uct \
//!     --games 1200 --budget nodes:2000 --seed 1 --sprt-elo0 0 --sprt-elo1 20
//! cargo run --release -p duels-agent-mcts-uct --example leaf_bench
//! cargo run --release -p duels-eval --example calibrate -- 200
//! ```
//!
//! The value convention (every node accumulates the result from
//! [`duels_core::Player::One`]'s perspective; the zero-sum flip happens once,
//! at selection) and the widening rule are documented in the `tree` module.
//!
//! # Example
//!
//! ```
//! use duels_agent_mcts_uct::MctsAgent;
//! use duels_agents_api::{Agent, Budget};
//! use duels_core::engine;
//!
//! let mut agent = MctsAgent::new(7);
//! let state = engine::new_game(7);
//! let legal = engine::legal_actions(&state);
//! let action = agent.choose(&state.observation(), &legal, Budget::Nodes(64));
//! assert!(legal.contains(&action));
//! ```

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

mod chance;
mod leaf;
mod rollout;
mod tree;

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::{engine, Action, Observation};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub use leaf::{
    generation_name, temperature, win_probability, LeafValue, TEMPERATURE_AGE_I,
    TEMPERATURE_AGE_II, TEMPERATURE_AGE_III, TEMPERATURE_OVERALL,
};
pub use rollout::{RaceWeights, RolloutWeights, RAIL};
pub use tree::{Config, PriorMode};

/// Monte Carlo Tree Search with UCT selection and explicit chance nodes.
#[derive(Debug)]
pub struct MctsAgent {
    cfg: Config,
    rng: StdRng,
    /// Simulations run over the agent's whole lifetime, for throughput
    /// reporting.
    total_simulations: u64,
    /// Nodes allocated during the most recent search.
    last_tree_size: usize,
}

impl MctsAgent {
    /// A new agent with the default configuration, seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, Config::default())
    }

    /// A new agent with an explicit configuration.
    pub fn with_config(seed: u64, cfg: Config) -> Self {
        Self {
            cfg,
            rng: StdRng::seed_from_u64(seed),
            total_simulations: 0,
            last_tree_size: 0,
        }
    }

    /// The configuration in force.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Total simulations this agent has run since it was created.
    pub fn total_simulations(&self) -> u64 {
        self.total_simulations
    }

    /// Nodes allocated by the most recent `choose` call.
    pub fn last_tree_size(&self) -> usize {
        self.last_tree_size
    }
}

impl Agent for MctsAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "mcts-uct".to_string(),
            version: "1.0.0".to_string(),
            params: self.cfg.describe(),
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action], budget: Budget) -> Action {
        assert!(
            !legal.is_empty(),
            "choose must not be called with no legal actions"
        );
        if legal.len() == 1 {
            return legal[0];
        }

        // `N` determinized worlds consistent with the observation, each with
        // its own tree and its own share of the budget. Hidden reveals
        // *inside* a search are re-drawn from public knowledge at each chance
        // node, so a world only fixes what the chance API does not cover
        // (future age decks, the undrafted wonder pool) — which is exactly
        // what a second determinization varies.
        let n = self.cfg.root_determinizations.max(1);
        let mut slices = Slices::new(budget, n);
        let mut trees: Vec<tree::Tree> = Vec::with_capacity(n);
        let mut offered: Option<Vec<Action>> = None;

        for i in 0..n {
            let root = obs.sample_state(&mut self.rng);

            // The offered actions and the determinized state must agree,
            // since legality is a function of public information only; filter
            // defensively so an unexpected mismatch can never return an
            // action the arena did not offer. Public legality does not vary
            // between determinizations, so this is settled once.
            let actions = match &offered {
                Some(actions) => actions.clone(),
                None => {
                    let mut actions: Vec<Action> = legal
                        .iter()
                        .copied()
                        .filter(|&a| engine::is_legal(&root, a))
                        .collect();
                    debug_assert_eq!(
                        actions.len(),
                        legal.len(),
                        "a determinized root disagreed with the offered legal actions"
                    );
                    if actions.is_empty() {
                        actions = legal.to_vec();
                    }
                    offered = Some(actions.clone());
                    actions
                }
            };

            let mut tree = tree::Tree::new(root, actions, self.cfg, &mut self.rng);
            slices.run(&mut tree, i, &mut self.rng);
            self.total_simulations += tree.simulations;
            trees.push(tree);
        }

        self.last_tree_size = trees.iter().map(|t| t.nodes.len()).sum();

        let chosen = tree::best_of(&trees).unwrap_or(legal[0]);
        if legal.contains(&chosen) {
            chosen
        } else {
            // Unreachable given the filter above; never hand back an action
            // the caller did not offer.
            legal[0]
        }
    }
}

/// One search budget, divided into `n` equal slices — one per root
/// determinization.
///
/// # How a slice is sized
///
/// A node budget is partitioned exactly: every slice gets `total / n`
/// simulations and the first `total % n` slices get one more, so the slices
/// sum to the whole budget however indivisible it is (a `Nodes(20)` budget
/// over 3 trees is `7 + 7 + 6`, not `6 + 6 + 6`).
///
/// A time budget is sliced by *absolute* deadlines measured from one shared
/// start — slice `i` ends at `start + total*(i+1)/n` — rather than by giving
/// each tree its own `total/n` milliseconds. That matters because a tree only
/// checks the clock every [`Config::time_check_interval`] simulations: with
/// per-tree stopwatches each overshoot would add to the total, while with
/// chained deadlines an overshooting slice eats into the next one instead and
/// only the last slice's overshoot escapes.
///
/// With `n == 1` both arms reduce to what this crate did before ensembling
/// existed: `total.max(1)` simulations, or a single deadline
/// `total` milliseconds after the first simulation.
#[derive(Debug)]
enum Slices {
    Nodes {
        total: u64,
        n: u64,
    },
    Time {
        total_ms: u64,
        n: u64,
        /// Captured on the first slice, so that the clock starts where the
        /// pre-ensemble code started it.
        start: Option<std::time::Instant>,
    },
}

impl Slices {
    fn new(budget: Budget, n: usize) -> Self {
        let n = n.max(1) as u64;
        match budget {
            Budget::Nodes(total) => Slices::Nodes { total, n },
            Budget::TimeMs(total_ms) => Slices::Time {
                total_ms,
                n,
                start: None,
            },
        }
    }

    /// Run slice `i` of the budget on `tree`.
    fn run(&mut self, tree: &mut tree::Tree, i: usize, rng: &mut StdRng) {
        let i = i as u64;
        match self {
            Slices::Nodes { total, n } => {
                // Written as a quotient plus a remainder rather than as
                // `total*(i+1)/n - total*i/n` so that a `Nodes(u64::MAX)`
                // budget cannot overflow the multiplication.
                let sims = *total / *n + u64::from(i < *total % *n);
                // A slice of zero still needs one simulation, otherwise there
                // are no visited children to choose between.
                for _ in 0..sims.max(1) {
                    tree.simulate(rng);
                }
            }
            Slices::Time { total_ms, n, start } => {
                // The workspace bans wall-clock reads so that the engine and
                // its agents stay reproducible from a seed; `Budget::TimeMs`
                // is the one place an agent is *asked* to read the clock, and
                // the read is confined to this function. `Budget::Nodes`
                // remains fully deterministic.
                #[allow(clippy::disallowed_methods)]
                let from = *start.get_or_insert_with(std::time::Instant::now);
                // In `u128` so that a `TimeMs(u64::MAX)` budget cannot
                // overflow the multiplication either.
                let elapsed_ms = u128::from(*total_ms) * u128::from(i + 1) / u128::from(*n);
                let deadline = from + std::time::Duration::from_millis(elapsed_ms as u64);
                let interval = tree.cfg.time_check_interval.max(1);
                loop {
                    for _ in 0..interval {
                        tree.simulate(rng);
                    }
                    #[allow(clippy::disallowed_methods)]
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_agent_random::RandomAgent;
    use duels_core::{GameResult, Player};

    /// A small budget: enough that the tree is exercised (root expansion,
    /// chance nodes, UCB1 re-selection) while keeping `cargo test` quick.
    const CI_BUDGET: Budget = Budget::Nodes(48);

    fn play(seed: u64, mcts_seat: Player, budget: Budget) -> (GameResult, u64) {
        let mut mcts = MctsAgent::new(seed ^ 0x0BAD_1DEA_0BAD_1DEA);
        let mut opponent = RandomAgent::new(seed ^ 0x5EED_5EED);
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);

        let mut plies = 0u32;
        loop {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = if state.current_player() == mcts_seat {
                mcts.choose(&obs, &legal, budget)
            } else {
                opponent.choose(&obs, &legal, budget)
            };
            assert!(
                legal.contains(&action),
                "agent returned an illegal action {action:?}"
            );
            engine::apply(&mut state, action, &mut rng).expect("a legal action");
            plies += 1;
            assert!(plies < 5_000, "game did not terminate after {plies} plies");
        }
        (
            state.result().expect("a finished game has a result"),
            mcts.total_simulations(),
        )
    }

    #[test]
    fn spec_reports_the_expected_name_version_and_params() {
        let agent = MctsAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "mcts-uct");
        assert_eq!(spec.version, "1.0.0");
        assert!(spec.params.contains("c=1.000"), "{}", spec.params);
        assert!(spec.params.contains("chance="), "{}", spec.params);
        assert!(spec.params.contains("rollout="), "{}", spec.params);
    }

    #[test]
    fn a_single_legal_action_is_returned_without_searching() {
        let mut agent = MctsAgent::new(3);
        let state = engine::new_game(3);
        let only = [engine::legal_actions(&state)[0]];
        let chosen = agent.choose(&state.observation(), &only, Budget::Nodes(10_000));
        assert_eq!(chosen, only[0]);
        assert_eq!(agent.total_simulations(), 0, "no search was needed");
    }

    #[test]
    fn every_returned_action_is_one_of_the_offered_ones() {
        let mut agent = MctsAgent::new(11);
        let state = engine::new_game(11);
        let legal = engine::legal_actions(&state);
        for _ in 0..5 {
            let a = agent.choose(&state.observation(), &legal, Budget::Nodes(20));
            assert!(legal.contains(&a));
        }
    }

    #[test]
    fn a_node_budget_runs_exactly_that_many_simulations() {
        let mut agent = MctsAgent::new(5);
        let state = engine::new_game(5);
        let legal = engine::legal_actions(&state);
        agent.choose(&state.observation(), &legal, Budget::Nodes(37));
        assert_eq!(agent.total_simulations(), 37);
        agent.choose(&state.observation(), &legal, Budget::Nodes(3));
        assert_eq!(agent.total_simulations(), 40);
    }

    #[test]
    fn a_time_budget_returns_promptly_and_does_some_work() {
        let mut agent = MctsAgent::new(9);
        let state = engine::new_game(9);
        let legal = engine::legal_actions(&state);
        let a = agent.choose(&state.observation(), &legal, Budget::TimeMs(20));
        assert!(legal.contains(&a));
        assert!(agent.total_simulations() > 0);
    }

    #[test]
    fn a_node_budget_is_reproducible_from_the_seed() {
        let state = engine::new_game(21);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        let pick = |seed: u64| {
            let mut agent = MctsAgent::new(seed);
            agent.choose(&obs, &legal, Budget::Nodes(200))
        };
        assert_eq!(pick(4), pick(4));
    }

    /// `choose` exactly as it read before root ensembling existed: one
    /// determinization, one tree, the whole node budget, the pre-ensemble
    /// move-selection rule — and, since it drives `tree::Tree::legacy_simulate`
    /// rather than the current `simulate`, the pre-*prior* search as well.
    ///
    /// This is the reference both the `root_determinizations = 1` path and the
    /// [`PriorMode::None`] path are checked against. It is a copy on purpose —
    /// a test that called the new code would prove nothing.
    fn legacy_choose(
        rng: &mut StdRng,
        cfg: Config,
        obs: &Observation,
        legal: &[Action],
        nodes: u64,
    ) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let root = obs.sample_state(rng);
        let mut actions: Vec<Action> = legal
            .iter()
            .copied()
            .filter(|&a| engine::is_legal(&root, a))
            .collect();
        if actions.is_empty() {
            actions = legal.to_vec();
        }
        let mut tree = tree::Tree::new(root, actions, cfg, rng);
        for _ in 0..nodes.max(1) {
            tree.legacy_simulate(rng);
        }
        let chosen = tree::legacy_best_action(&tree).unwrap_or(legal[0]);
        if legal.contains(&chosen) {
            chosen
        } else {
            legal[0]
        }
    }

    /// The equivalence that makes the option safe to add: with
    /// `root_determinizations = 1` the agent is bit-for-bit the agent this
    /// crate shipped before, under a node budget — same determinization drawn
    /// from the same RNG stream, same number of simulations, same
    /// move-selection rule, same move.
    ///
    /// Checked over whole games rather than only at the opening position, so
    /// that the RNG streams have to stay in step across dozens of `choose`
    /// calls, chance nodes, pending choices and all.
    #[test]
    fn one_determinization_is_the_pre_ensemble_agent_move_for_move() {
        for seed in 0..8u64 {
            let mut agent = MctsAgent::with_config(
                seed,
                Config {
                    root_determinizations: 1,
                    ..Config::default()
                },
            );
            // The same seed, so the same stream, driven by the copy above.
            let mut legacy_rng = StdRng::seed_from_u64(seed);
            let cfg = Config::default();

            let mut state = engine::new_game(seed ^ 0xC0FF_EE00);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);
            let mut decisions = 0u32;
            loop {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let obs = state.observation();
                let budget = 24 + u64::from(decisions % 7);
                let got = agent.choose(&obs, &legal, Budget::Nodes(budget));
                let want = legacy_choose(&mut legacy_rng, cfg, &obs, &legal, budget);
                assert_eq!(
                    got, want,
                    "seed {seed}, decision {decisions}: ensembling changed the N=1 move"
                );
                engine::apply(&mut state, got, &mut rng).expect("a legal action");
                decisions += 1;
                assert!(decisions < 5_000);
            }
            assert!(decisions > 20, "the game was too short to prove much");
        }
    }

    /// The equivalence that makes the prior option safe to add, and the twin
    /// of the test above: with [`PriorMode::None`] the agent is move-for-move
    /// the agent this crate shipped before priors existed, driven by the
    /// verbatim pre-prior `expand`/`select_ucb1`/`simulate` in `tree.rs`.
    ///
    /// Whole games again, so the two RNG streams have to stay in step across
    /// dozens of `choose` calls — the prior path consumes no randomness, and
    /// this is what says so over a long run rather than at one position.
    #[test]
    fn prior_none_is_the_pre_prior_agent_move_for_move() {
        for seed in 0..8u64 {
            let cfg = Config {
                prior: PriorMode::None,
                ..Config::default()
            };
            let mut agent = MctsAgent::with_config(seed, cfg);
            let mut legacy_rng = StdRng::seed_from_u64(seed);

            let mut state = engine::new_game(seed ^ 0xC0FF_EE00);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);
            let mut decisions = 0u32;
            loop {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let obs = state.observation();
                let budget = 24 + u64::from(decisions % 7);
                let got = agent.choose(&obs, &legal, Budget::Nodes(budget));
                let want = legacy_choose(&mut legacy_rng, cfg, &obs, &legal, budget);
                assert_eq!(
                    got, want,
                    "seed {seed}, decision {decisions}: PriorMode::None changed the move"
                );
                engine::apply(&mut state, got, &mut rng).expect("a legal action");
                decisions += 1;
                assert!(decisions < 5_000);
            }
            assert!(decisions > 20, "the game was too short to prove much");
        }
    }

    /// The equivalence that makes the race option safe to add, and the twin of
    /// the two above: with [`RaceWeights::NEUTRAL`] the agent is move-for-move
    /// the agent this crate shipped before race weights existed, driven by the
    /// verbatim pre-race `pick`/`play_out` in `rollout::legacy`.
    ///
    /// Whole seeded games again. The rollout policy is where nearly every RNG
    /// draw a search makes happens, so a single extra or reordered draw would
    /// desynchronise the two streams within one playout and change a move long
    /// before the game ended.
    #[test]
    fn race_neutral_is_biased_move_for_move() {
        for seed in 0..8u64 {
            let cfg = Config {
                race: RaceWeights::NEUTRAL,
                ..Config::default()
            };
            assert_eq!(cfg.rollout, RolloutWeights::BIASED);
            let mut agent = MctsAgent::with_config(seed, cfg);
            let mut legacy_rng = StdRng::seed_from_u64(seed);

            let mut state = engine::new_game(seed ^ 0xC0FF_EE00);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);
            let mut decisions = 0u32;
            loop {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let obs = state.observation();
                let budget = 24 + u64::from(decisions % 7);
                let got = agent.choose(&obs, &legal, Budget::Nodes(budget));
                let want = legacy_choose(&mut legacy_rng, cfg, &obs, &legal, budget);
                assert_eq!(
                    got, want,
                    "seed {seed}, decision {decisions}: RaceWeights::NEUTRAL changed the move"
                );
                engine::apply(&mut state, got, &mut rng).expect("a legal action");
                decisions += 1;
                assert!(decisions < 5_000);
            }
            assert!(decisions > 20, "the game was too short to prove much");
        }
    }

    /// The equivalence that makes the leaf-value option safe to add, and the
    /// twin of the three above: with [`LeafValue::Rollout`] the agent is
    /// move-for-move the agent this crate shipped before leaf values existed,
    /// driven by the verbatim pre-leaf `simulate` in `tree.rs` (which reaches
    /// the playout directly rather than through the new `leaf_value`).
    ///
    /// Whole seeded games, so the two RNG streams have to stay in step across
    /// dozens of `choose` calls. `tree::tests::leaf_rollout_grows_the_same_
    /// tree_as_the_pre_leaf_search` is the stronger, node-for-node form of the
    /// same claim, and `tree::tests::the_rollout_leaf_builds_no_evaluation_root`
    /// is the third part: the default path does not even build a
    /// `duels_eval::Root`.
    #[test]
    fn leaf_rollout_is_the_pre_leaf_agent_move_for_move() {
        for seed in 0..8u64 {
            let cfg = Config {
                leaf: LeafValue::Rollout,
                ..Config::default()
            };
            let mut agent = MctsAgent::with_config(seed, cfg);
            let mut legacy_rng = StdRng::seed_from_u64(seed);

            let mut state = engine::new_game(seed ^ 0xC0FF_EE00);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);
            let mut decisions = 0u32;
            loop {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let obs = state.observation();
                let budget = 24 + u64::from(decisions % 7);
                let got = agent.choose(&obs, &legal, Budget::Nodes(budget));
                let want = legacy_choose(&mut legacy_rng, cfg, &obs, &legal, budget);
                assert_eq!(
                    got, want,
                    "seed {seed}, decision {decisions}: LeafValue::Rollout changed the move"
                );
                engine::apply(&mut state, got, &mut rng).expect("a legal action");
                decisions += 1;
                assert!(decisions < 5_000);
            }
            assert!(decisions > 20, "the game was too short to prove much");
        }
    }

    /// A leaf variant must not change *what* the agent is allowed to do: full
    /// seeded games from both seats, every variant, no panic and no illegal
    /// move — and every variant has to actually search.
    #[test]
    fn every_leaf_value_plays_full_games_without_incident() {
        for (i, leaf) in [
            LeafValue::Rollout,
            LeafValue::Static,
            LeafValue::Truncated { plies: 8 },
            LeafValue::Blend { weight: 0.5 },
        ]
        .into_iter()
        .enumerate()
        {
            let mut wins = 0u32;
            for seed in 0..6u64 {
                let seat = if seed % 2 == 0 {
                    Player::One
                } else {
                    Player::Two
                };
                let mut mcts = MctsAgent::with_config(
                    seed ^ 0x0BAD_1DEA,
                    Config {
                        leaf,
                        ..Config::default()
                    },
                );
                let mut opponent = RandomAgent::new(seed ^ 0x5EED_5EED);
                let mut state = engine::new_game(seed + 900 * i as u64);
                let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);
                loop {
                    let legal = engine::legal_actions(&state);
                    if legal.is_empty() {
                        break;
                    }
                    let obs = state.observation();
                    let action = if state.current_player() == seat {
                        mcts.choose(&obs, &legal, CI_BUDGET)
                    } else {
                        opponent.choose(&obs, &legal, CI_BUDGET)
                    };
                    assert!(legal.contains(&action), "{leaf:?} returned {action:?}");
                    engine::apply(&mut state, action, &mut rng).expect("a legal action");
                }
                let result = state.result().expect("a finished game has a result");
                if result.winner() == Some(seat) {
                    wins += 1;
                }
                assert!(mcts.total_simulations() > 0);
            }
            println!("{leaf:?}: {wins}/6 against random at {CI_BUDGET:?}");
        }
    }

    /// The spec string a results file records has to name the leaf value *and*
    /// the evaluation generation it was scored against — a strength number
    /// measured against `duels-eval` v6 is not comparable with one measured
    /// against a later round, and a results file that does not say which is
    /// uninterpretable after the fact.
    #[test]
    fn the_spec_reports_the_leaf_value_and_its_evaluation_generation() {
        let describe = |leaf| {
            MctsAgent::with_config(
                1,
                Config {
                    leaf,
                    ..Config::default()
                },
            )
            .spec()
            .params
        };
        assert!(describe(LeafValue::Rollout).contains("leaf=rollout"));
        assert!(describe(LeafValue::Static).contains("leaf=static"));
        assert!(describe(LeafValue::Truncated { plies: 8 }).contains("leaf=truncated(8)"));
        assert!(describe(LeafValue::Blend { weight: 0.5 }).contains("leaf=blend(0.500)"));
        // The pin, spelled out where a results file will record it.
        assert!(describe(LeafValue::Static).contains("evalgen=v6"));
        let older = MctsAgent::with_config(
            1,
            Config {
                leaf: LeafValue::Static,
                eval_generation: duels_eval::Config::v1(),
                ..Config::default()
            },
        )
        .spec()
        .params;
        assert!(older.contains("evalgen=v1"), "{older}");
    }

    /// The pin is a *frozen snapshot*, not "whatever `duels-eval` defaults to
    /// today". The two are the same configuration as of this writing, which is
    /// exactly why this has to be asserted rather than eyeballed: the moment a
    /// seventh `duels-eval` round moves the default, this agent must keep
    /// scoring against the generation its strength was measured on until
    /// somebody re-measures it.
    #[test]
    fn the_default_leaf_generation_is_the_frozen_snapshot() {
        assert_eq!(
            Config::default().eval_generation,
            duels_eval::Config::v6(),
            "the pin must name a frozen generation"
        );
    }

    /// A race variant must not change *what* the agent is allowed to do: full
    /// seeded games from both seats, every variant, no panic and no illegal
    /// move.
    #[test]
    fn every_race_variant_plays_full_games_without_incident() {
        for (i, race) in [
            RaceWeights::NEUTRAL,
            RaceWeights::TIER1_ONLY,
            RaceWeights::mild(),
            RaceWeights::MEDIUM,
            RaceWeights::strong(),
        ]
        .into_iter()
        .enumerate()
        {
            let mut wins = 0u32;
            for seed in 0..6u64 {
                let seat = if seed % 2 == 0 {
                    Player::One
                } else {
                    Player::Two
                };
                let mut mcts = MctsAgent::with_config(
                    seed ^ 0x0BAD_1DEA,
                    Config {
                        race,
                        ..Config::default()
                    },
                );
                let mut opponent = RandomAgent::new(seed ^ 0x5EED_5EED);
                let mut state = engine::new_game(seed + 700 * i as u64);
                let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);
                loop {
                    let legal = engine::legal_actions(&state);
                    if legal.is_empty() {
                        break;
                    }
                    let obs = state.observation();
                    let action = if state.current_player() == seat {
                        mcts.choose(&obs, &legal, CI_BUDGET)
                    } else {
                        opponent.choose(&obs, &legal, CI_BUDGET)
                    };
                    assert!(
                        legal.contains(&action),
                        "{} returned {action:?}",
                        race.name()
                    );
                    engine::apply(&mut state, action, &mut rng).expect("a legal action");
                }
                let result = state.result().expect("a finished game has a result");
                if result.winner() == Some(seat) {
                    wins += 1;
                }
                assert!(mcts.total_simulations() > 0);
            }
            println!(
                "race={}: {wins}/6 against random at {CI_BUDGET:?}",
                race.name()
            );
        }
    }

    /// The spec string a results file records has to name the race variant, or
    /// an arena run cannot be told apart from the baseline after the fact.
    #[test]
    fn the_spec_reports_the_race_variant() {
        let describe = |race| {
            MctsAgent::with_config(
                1,
                Config {
                    race,
                    ..Config::default()
                },
            )
            .spec()
            .params
        };
        assert!(describe(RaceWeights::NEUTRAL).contains("race=neutral"));
        assert!(describe(RaceWeights::TIER1_ONLY).contains("race=tier1_only"));
        assert!(describe(RaceWeights::mild()).contains("race=mild"));
        assert!(describe(RaceWeights::MEDIUM).contains("race=medium"));
        assert!(describe(RaceWeights::strong()).contains("race=strong"));
    }

    /// A prior mode must not change *what* the agent is allowed to do: full
    /// seeded games from both seats, every mode, no panic and no illegal move.
    #[test]
    fn every_prior_mode_plays_full_games_without_incident() {
        for (i, prior) in [
            PriorMode::None,
            PriorMode::ExpansionOrder,
            PriorMode::ProgressiveBias { weight: 1.0 },
        ]
        .into_iter()
        .enumerate()
        {
            let mut wins = 0u32;
            for seed in 0..6u64 {
                let seat = if seed % 2 == 0 {
                    Player::One
                } else {
                    Player::Two
                };
                let mut mcts = MctsAgent::with_config(
                    seed ^ 0x0BAD_1DEA,
                    Config {
                        prior,
                        ..Config::default()
                    },
                );
                let mut opponent = RandomAgent::new(seed ^ 0x5EED_5EED);
                let mut state = engine::new_game(seed + 500 * i as u64);
                let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);
                loop {
                    let legal = engine::legal_actions(&state);
                    if legal.is_empty() {
                        break;
                    }
                    let obs = state.observation();
                    let action = if state.current_player() == seat {
                        mcts.choose(&obs, &legal, CI_BUDGET)
                    } else {
                        opponent.choose(&obs, &legal, CI_BUDGET)
                    };
                    assert!(legal.contains(&action), "{prior:?} returned {action:?}");
                    engine::apply(&mut state, action, &mut rng).expect("a legal action");
                }
                let result = state.result().expect("a finished game has a result");
                if result.winner() == Some(seat) {
                    wins += 1;
                }
                assert!(mcts.total_simulations() > 0);
            }
            println!("{prior:?}: {wins}/6 against random at {CI_BUDGET:?}");
        }
    }

    /// The spec string a results file records has to name the mode, or an
    /// arena run cannot be told apart from the baseline after the fact.
    #[test]
    fn the_spec_reports_the_prior_mode() {
        let describe = |prior| {
            MctsAgent::with_config(
                1,
                Config {
                    prior,
                    ..Config::default()
                },
            )
            .spec()
            .params
        };
        assert!(describe(PriorMode::None).contains("prior=none"));
        assert!(describe(PriorMode::ExpansionOrder).contains("prior=expansion_order"));
        assert!(describe(PriorMode::ProgressiveBias { weight: 1.5 })
            .contains("prior=progressive_bias(1.500)"));
    }

    /// A node budget is partitioned, not multiplied: `N` trees share the
    /// simulations one tree would have run.
    #[test]
    fn the_node_budget_is_split_across_determinizations() {
        let state = engine::new_game(31);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        for n in [1usize, 2, 4, 8] {
            let mut agent = MctsAgent::with_config(
                5,
                Config {
                    root_determinizations: n,
                    ..Config::default()
                },
            );
            let a = agent.choose(&obs, &legal, Budget::Nodes(400));
            assert!(legal.contains(&a));
            assert_eq!(
                agent.total_simulations(),
                400,
                "N={n} did not spend exactly the budget"
            );
        }
    }

    #[test]
    fn ensembling_still_returns_a_legal_move_at_a_time_budget() {
        let state = engine::new_game(17);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        let mut agent = MctsAgent::with_config(
            2,
            Config {
                root_determinizations: 4,
                ..Config::default()
            },
        );
        let a = agent.choose(&obs, &legal, Budget::TimeMs(20));
        assert!(legal.contains(&a));
        assert!(agent.total_simulations() > 0);
        assert!(agent.last_tree_size() > 0);
    }

    /// The headline robustness test: full seeded games against the random
    /// agent, from both seats, always reaching a `GameResult` without a
    /// panic, a hang, or an illegal action.
    #[test]
    fn plays_twenty_full_seeded_games_against_random_without_incident() {
        let mut sims = 0u64;
        for seed in 0..20u64 {
            let seat = if seed % 2 == 0 {
                Player::One
            } else {
                Player::Two
            };
            let (result, s) = play(seed, seat, CI_BUDGET);
            sims += s;
            println!("seed {seed} (mcts as {seat}): {result:?}");
        }
        assert!(sims > 0, "the agent never searched");
    }

    /// Even at a CI-sized budget the search should already be clearly better
    /// than uniform-random play. This is a loose smoke test, not the real
    /// strength measurement (see `examples/vs_random.rs`); it exists so that
    /// a sign error in backpropagation or a perspective flip cannot land
    /// silently.
    #[test]
    fn beats_random_at_a_small_budget() {
        let mut wins = 0u32;
        let games = 12u64;
        for seed in 0..games {
            let seat = if seed % 2 == 0 {
                Player::One
            } else {
                Player::Two
            };
            let (result, _) = play(100 + seed, seat, CI_BUDGET);
            if result.winner() == Some(seat) {
                wins += 1;
            }
        }
        assert!(
            wins * 2 > games as u32,
            "won only {wins}/{games} against random at {CI_BUDGET:?}; \
             suspect backpropagation sign, UCB1 perspective, or chance handling"
        );
    }
}
