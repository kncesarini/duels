//! `duels-agent-mcts-eval`: Monte Carlo Tree Search with UCT selection and
//! **explicit chance nodes**, whose leaf value is **half playout and half
//! [`duels_eval`]'s hand-crafted evaluation**.
//!
//! That leaf value is the whole point of this crate, and it is the largest
//! single strength effect this project has measured: `+89.2` Elo
//! `[+77.5, +101.0]` over 3,600 paired, seat-swapped games against the same
//! search with a pure playout leaf, positive on three of three disjoint seed
//! ranges, and *larger* at a wall-clock budget than at a node budget. The
//! measurement is reproduced in full below.
//!
//! # Why this is its own crate rather than a flag on `mcts-uct`
//!
//! The blend was first built and measured as an opt-in `Config::leaf` option
//! inside `mcts-uct`. Promoting it *there* would have meant moving that
//! crate's tuned `Config::exploration` — a blended reward has half the spread
//! a Bernoulli playout does, so `c` has to be rescaled with it (see
//! [`LeafValue::Blend`]) — and every other knob in that `Config` was tuned
//! against `c = 1.0`: the rollout weights, the race tables, the widening
//! constants, the prior sweep. Flipping it in place would have re-defined
//! `mcts-uct` and moved `leaderboard::CHAMPION` in the same breath as the PR
//! that merely *measured* the leaf value.
//!
//! So the two configurations are now two agents, each internally consistent,
//! each on the ladder in its own right, and the ablation between them is one
//! spec string away in one binary:
//!
//! ```text
//! cargo run --release -p duels-arena -- match \
//!     --agent-a mcts-eval --agent-b mcts-eval:base=rollout \
//!     --games 1200 --budget nodes:2000 --seed 1 --sprt-elo0 0 --sprt-elo1 20
//! ```
//!
//! [`Config::rollout_base`] is that control, and it is not a claim about
//! being `mcts-uct` — it is *checked* to be, node for node, against a
//! verbatim copy of that agent's `expand`/`select_ucb1`/`simulate`
//! (`tree::tests::the_rollout_base_grows_the_mcts_uct_tree_node_for_node`,
//! with `tree::tests::the_default_search_is_not_the_mcts_uct_search` as the
//! non-vacuity check).
//!
//! The search machinery itself — `tree`, `chance`, `rollout`, and the
//! [`Agent`] impl below — is a deliberate **copy** of `mcts-uct`'s, not a
//! dependency on it. `CLAUDE.md`'s "agent crates are self-contained"
//! invariant forbids one agent crate depending on another, and this is the
//! same accepted, intentional duplication `greedy-ev` carries against
//! `greedy`. `duels-eval` is a different matter: it is a shared library
//! *below* the agents, so both this crate and `phased` depend on it, which is
//! exactly what the layering is for.
//!
//! # Tracking `duels-eval` live, on purpose
//!
//! **This crate evaluates against [`duels_eval::Config::default`] — whatever
//! that is in the build you are running — and holds no version pin.** There
//! is no `Config::eval_generation` field, no `Config::vN()` snapshot, and no
//! golden-values test freezing what `evaluate` returns. The configuration is
//! read in `tree::Tree::new`, once per search tree, at the moment the tree is
//! built.
//!
//! ## This is the opposite of what `mcts-uct` did, deliberately
//!
//! `mcts-uct`'s leaf-value option pinned `duels_eval::Config::v6()` and held a
//! golden-values test against ~50 fixed positions, precisely so that a
//! seventh `phased`/`duels-eval` tuning round could not silently move a
//! measured `mcts-uct` strength number. `CLAUDE.md` still records that as a
//! standing prior — *"a search that consumes `duels-eval` must pin a
//! generation"* — and for an agent whose *identity* is its search, that is the
//! right call: the evaluation is an incidental input there, and an incidental
//! input moving under a measurement is a bug.
//!
//! Here the relationship is inverted. This agent's identity **is**
//! `duels-eval` inside a search. The project's reason for creating it, stated
//! when the decision was made, is that as `duels-eval` keeps improving
//! through future `phased`-style rounds this agent should get stronger right
//! along with it, automatically, with no manual version-bump step. A pin
//! would defeat that: it would freeze this crate at the sixth evaluation
//! round forever, and every future round's gain would need a deliberate,
//! easily-forgotten edit here to reach the ladder at all.
//!
//! **So: do not "fix" this into a pin by copying `mcts-uct`'s pattern.** It
//! looks like an oversight and it is not one. If you are reading this because
//! a `duels-eval` round moved this agent's rating, that is the design
//! working.
//!
//! ## What is given up, and what replaces it
//!
//! Being honest about the cost, because there is one. A pin buys
//! *comparability across time*: two results files from different months
//! measure the same agent. Live tracking gives that up — two builds of
//! `mcts-eval` either side of a `duels-eval` round are genuinely different
//! players, and pooling their games would be a mistake.
//!
//! Three things stand in for it:
//!
//! 1. **The spec string records the evaluation itself, not a label.**
//!    [`Config::describe`] ends with the whole
//!    [`duels_eval::Config::params_string`] this search will use. A results
//!    file therefore says exactly which evaluation produced it, which makes
//!    two files from different rounds *distinguishable* — strictly more
//!    information than a `v6` tag, and the thing a pin was protecting.
//! 2. **The nightly round robin re-measures everything anyway.** The ladder
//!    is refitted from scratch every night at whatever the current code is
//!    (see `duels_arena::leaderboard`), so this agent's rating is never a
//!    remembered number in the first place.
//! 3. **`duels-eval` owns its own identity tests.** Its `tests/vN_identity.rs`
//!    files and its `CODEOWNERS` mandatory-review rule are where a change to
//!    the evaluation gets noticed. That is the right layer for it; a
//!    downstream golden-values test in an agent crate was always a proxy.
//!
//! What live tracking does **not** relax is the per-decision invariant: within
//! one search, the leaf value must not depend on which hidden world the root
//! determinization drew. See
//! `tree::tests::a_static_leaf_value_is_determinization_invariant`, which
//! spells out the difference.
//!
//! One practical consequence worth naming: the configuration is read in
//! `Tree::new`, **not** in [`MctsEvalAgent::new`]. Reading it at agent
//! construction would capture a snapshot for that agent's whole lifetime — a
//! long-lived `duels-server` room, say — which is the same pin wearing
//! different clothes.
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
//! ## How chance is handled, precisely
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
//! # The leaf value (`Config::leaf`)
//!
//! [`LeafValue`] decides what a freshly added leaf is worth: the default
//! mixture, `duels-eval`'s evaluation alone through a calibrated sigmoid, a
//! truncated playout that ends in one, or the plain playout. See the `leaf`
//! module for the mechanism — the per-age temperature calibration, where the
//! one [`duels_eval::Root`] is built, why the perspective is always Player
//! One, and the algebra relating [`LeafValue::Blend`]'s weight to the
//! exploration constant. This section is the measurement, carried over from
//! the investigation that produced it (`mcts-uct` PR #38), where every number
//! below was taken with the candidate spelled `mcts-uct:leaf=blend:0.5,c=0.5`
//! — the configuration [`Config::default`] now *is*.
//!
//! ## What each variant costs
//!
//! `examples/leaf_bench.rs`, 30 positions at `Nodes(2000)` — a node budget, so
//! the *work* is exactly fixed (52,000 simulations per column) and only the
//! elapsed time moves. Run on a machine that was not quiet, so read the ratio
//! column and not the absolute microseconds:
//!
//! | leaf | µs/simulation | throughput vs plain playout |
//! |---|---|---|
//! | `Rollout` | 18.84 | 1.00x |
//! | `Static` | 1.55 | **12.15x** |
//! | `Truncated { plies: 4 }` | 2.57 | 7.33x |
//! | `Truncated { plies: 8 }` | 3.83 | 4.92x |
//! | `Truncated { plies: 16 }` | 5.60 | 3.37x |
//! | `Blend { weight: 0.3 }` | 18.83 | 1.00x |
//! | **`Blend { weight: 0.5 }`** (default) | 18.66 | 1.01x |
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
//! `20001..20151`, 300 games each against a pure-playout leaf at
//! `Nodes(2000)`, `+/-` about 2.9. Kept for the record and for what it says
//! about the shape of the family, not as a strength claim — the ranges below
//! are the evidence:
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
//! | **`leaf=blend:0.5,c=0.5`** (this crate's default) | **63.7%** | **+97.1** |
//! | `leaf=blend:0.7,c=0.3` | 65.0% | +107.2 |
//! | `c=0.5` alone | 54.0% | +27.8 |
//! | control (playout vs playout) | 52.0% | +13.9 |
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
//! investigation and it is the reason a *mixture* is the right shape. Victory
//! kinds for `leaf=static` in that sweep (300 games):
//!
//! | | `leaf=static` | plain playout |
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
//! **The maximum did not survive** — see below. Reporting the sweep's argmax
//! as the answer would have shipped the weaker of the two.
//!
//! ## What it measures: `+89` Elo, on three disjoint ranges
//!
//! 1,200 games per range at `Nodes(2000)`, paired and seat-swapped, against
//! the pure-playout leaf at `c = 1.0` (verified, not assumed: `c=1.000`,
//! `race=neutral`, `prior=none`, `dets=1`, `leaf=rollout` — which is exactly
//! [`Config::rollout_base`]). `+/-` is one binomial standard error:
//!
//! | arm | `1..600` | `5001..5600` | `10001..10600` | pooled (3,600 games) | Elo |
//! |---|---|---|---|---|---|
//! | **`blend:0.5,c=0.5`** (default) | 64.42% | 62.04% | 61.21% | **62.56% +/- 0.81** | **+89.2 [+77.5, +101.0]** |
//! | `blend:0.7,c=0.3` (the sweep's argmax) | 61.92% | 61.88% | 58.92% | 60.90% +/- 0.81 | +77.0 [+65.4, +88.7] |
//! | `blend:0.5` (`c` unchanged) | 60.54% | 59.29% | 57.42% | 59.08% +/- 0.82 | +63.8 [+52.3, +75.4] |
//! | `c=0.5` alone (attribution control) | 49.12% | 51.46% | 49.71% | 50.10% +/- 0.83 | +0.7 [-10.7, +12.0] |
//! | `c=0.3` alone (attribution control) | 37.00% | 35.21% | 35.75% | 35.99% +/- 0.80 | -100.1 [-112.0, -88.3] |
//! | playout vs playout (noise floor) | 48.96% | 48.25% | 49.83% | 49.01% +/- 0.83 | -6.9 [-18.2, +4.5] |
//!
//! Every range is positive for all three blend arms, SPRT (`elo0 = 0` vs
//! `elo1 = 20`) reads `AcceptH1` on every one of their nine range-runs
//! (`llr` 10.3 to 17.9 against a 2.944 bound), and the intervals are nowhere
//! near the control's. This is by a wide margin the largest effect measured
//! anywhere in this project: the previous best, `mcts-uct`'s terminal rails,
//! was `+26` Elo.
//!
//! **The exploration constant is not the effect.** `c = 0.5` on its own scores
//! 50.10% over the same 3,600 games — indistinguishable from the noise floor.
//! It is worth about `+25` Elo *in combination* with the blend (62.56% against
//! 59.08%), which is the direction [`LeafValue::Blend`]'s rescaling argument
//! predicts: at `weight = 0.5` the reward's spread is halved, so the
//! exploration bonus has to be halved with it to leave the balance where it
//! was tuned. This is why [`Config::default`] moves both fields together and
//! why they should not be thought of as two independent defaults.
//!
//! `c = 0.3` makes that argument much more sharply, which is why it is in the
//! table. On its own it is a **disaster** — `-100` Elo — and yet
//! `blend:0.7,c=0.3`, which contains it, is `+77`. A knob worth `-100` alone
//! and `+77` in combination is not plausibly an independent contribution; it
//! is the rescaling the blended reward requires.
//!
//! ## Why the sweep's argmax lost, and what it says about the mechanism
//!
//! `blend:0.7,c=0.3` won the 300-game sweep (+107 against +97) and then
//! finished 12 Elo *behind* `blend:0.5,c=0.5` over 3,600, losing on all three
//! ranges. Pooled victory kinds say why, and it is not noise:
//!
//! | pooled, 3,600 games | wins by military | wins by civilian score |
//! |---|---|---|
//! | `blend:0.5,c=0.5` vs playout | **342** - 288 | 1,815 - 992 |
//! | `blend:0.7,c=0.3` vs playout | 170 - **323** | 1,911 - 1,036 |
//!
//! At `weight = 0.7` the search gets *better* at city quality (1,911 civilian
//! wins, more than the 0.5 blend manages) and **loses the military race
//! outright** — 170 military wins against the playout's 323, having been
//! ahead 342-288 at `weight = 0.5`. The blend weight is not a free knob to
//! push towards the evaluation; it is the balance between two different kinds
//! of sight, and half is where it sits.
//!
//! ## Where the wins come from: points, not races
//!
//! Pooled victory kinds over the same 3,600 games, the default against the
//! pure-playout arm:
//!
//! | | default (blend) | plain playout |
//! |---|---|---|
//! | wins by civilian score | **1,815** | 992 |
//! | wins by military supremacy | 342 | 288 |
//! | wins by scientific supremacy | 62 | 34 |
//! | wins by tiebreak | 31 | 32 |
//!
//! `+823` of the `+904` win margin is **civilian score**. That matters because
//! the other mechanism available here — [`RaceWeights::TIER1_ONLY`]'s terminal
//! rails — is *entirely* military (`138-68` in its own measurement, with the
//! number of military-decided games unmoved). These are not the same effect
//! wearing two hats, and the composition test says so directly. With
//! `race=tier1` on **both** sides, 1,200 games on each of two ranges:
//!
//! | | `1..600` | `5001..5600` | pooled (2,400) | Elo |
//! |---|---|---|---|---|
//! | `blend:0.5,c=0.5,race=tier1` vs `race=tier1` | 62.79% | 65.46% | **64.12% +/- 0.98** | **+100.9 [+86.6, +115.6]** |
//! | `race=tier1` vs `race=tier1` (control) | 51.12% | 50.75% | 50.94% +/- 1.02 | +6.5 [-7.4, +20.4] |
//!
//! `+100.9` with the rails on both sides, against `+89.2` with them nowhere:
//! the two mechanisms **add**, and if anything the blend is worth slightly
//! *more* once the rails are present. With the rails on both sides the
//! blend's military edge disappears — 191 military wins against 175,
//! essentially level, where without rails it was 342-288 — while its civilian
//! margin is undiminished (1,293 against 652). The rails were already
//! supplying the military tempo sight, so the blend stops needing to; what it
//! adds on top is entirely city quality.
//!
//! (`RaceWeights::TIER1_ONLY` is nonetheless **not** [`Config::default`] here,
//! for the same reason it is not `mcts-uct`'s: its pre-registered mechanism
//! criterion — measurably better science exposure or conversion in self-play —
//! failed. Promoting it is a separate decision on its own evidence, and this
//! crate deliberately does not smuggle it in.)
//!
//! ## Budget equivalence: worth more than a doubling
//!
//! 400 games each, `1..201`, candidate at `Nodes(1000)` against the
//! pure-playout arm at `Nodes(2000)`:
//!
//! | half-budget side | score vs playout at `Nodes(2000)` | ms/game, half-budget side vs full |
//! |---|---|---|
//! | `blend:0.5,c=0.5` at `Nodes(1000)` | **55.5% +/- 2.5** | 437 vs 782 (56%) |
//! | `leaf=rollout` at `Nodes(1000)` (control) | 40.0% +/- 2.4 | 408 vs 817 (50%) |
//!
//! Halving the node budget costs the playout arm 10 points of score; the blend
//! at *half* the budget **beats** the full-budget playout outright. So the
//! leaf value is worth more than a doubling of search — and it gets there on
//! 56% of the opponent's wall clock against the control's 50%, i.e. its own
//! throughput cost is about six points of extra wall clock for half the
//! nodes, nothing like enough to consume a 15-point score advantage.
//!
//! ## Ladder: nothing regressed, and the gap to `phased` widened
//!
//! 400 games each at `Nodes(2000)`, seeds `1..200`:
//!
//! | opponent | default (blend) | plain playout |
//! |---|---|---|
//! | `greedy-ev` | **400/400** (+1161 Elo) | 399/400 (+970 Elo) |
//! | `phased` | **89.25%** (+365.9 Elo) | 80.13% (+241.4 Elo) |
//! | `alphabeta` | **84.50%** (+293.5 Elo) | 74.88% (+189.1 Elo) |
//!
//! The `phased` row was the pre-registered red flag, and it is the one to read
//! first: this agent scores its leaves with `phased`'s *own* evaluation, so if
//! it beat `phased` by **less** than a plain playout does, that would point at
//! something wrong in the integration — an evaluation read with the wrong
//! sign, a stale pricing context, a leaf value that is really just noise —
//! rather than at a mechanism that merely fails to help. It beats `phased` by
//! nine points more, which is the opposite of that failure signature.
//!
//! ## At a wall-clock budget
//!
//! The test this line of work has been burned by twice: a change that wins at
//! a fixed node count can lose at a fixed clock if it costs more per unit of
//! work (`mcts-uct`'s `Config::prior` is the cautionary tale — a `+11.7` point
//! estimate at `Nodes` measured `-33` at `TimeMs`). 400 games per range,
//! **one match at a time with `RAYON_NUM_THREADS=1`**, so each game gets a
//! whole core and the per-decision work is production-like; nothing else was
//! running.
//!
//! | budget | `1..200` | `5001..5200` | pooled (800) | Elo | control (playout vs playout) |
//! |---|---|---|---|---|---|
//! | `TimeMs(20)` | 67.13% | 62.25% | **64.69% +/- 1.69** | **+105.2 [+80.5, +130.9]** | 46.25%, -26.0 |
//! | `TimeMs(100)` | 65.25% | 60.75% | **63.00% +/- 1.71** | **+92.5 [+67.9, +117.9]** | 48.50%, -10.4 |
//!
//! `AcceptH1` on all four range-runs. **The gain does not merely survive a
//! wall-clock budget, it grows**: `+105` at `TimeMs(20)` and `+93` at
//! `TimeMs(100)`, against `+89` at `Nodes(2000)`.
//!
//! That direction is the expected one rather than a surprise, and the cost
//! table is why. A blend has no measurable throughput cost, so a wall-clock
//! budget buys it essentially the same number of simulations it buys a plain
//! playout, and the leaf-value advantage transfers intact. What is left is a
//! budget effect: `TimeMs(20)` buys roughly a thousand simulations, which is
//! the `Nodes(1000)` regime where the budget-equivalence table already showed
//! the blend at its most valuable. A better leaf value is worth more when
//! there are fewer leaves to average over — which is also why `TimeMs(100)`,
//! at roughly five thousand simulations, lands slightly *below* `TimeMs(20)`
//! and slightly above `Nodes(2000)`. The whole family of budgets is
//! consistent: the effect is large everywhere and largest where search is
//! scarcest. This is also why `duels-server` hands this agent a `TimeMs`
//! budget in a live room and expects it to be the strongest thing there.
//!
//! Note the controls: `-26.0` at `TimeMs(20)` and `-10.4` at `TimeMs(100)`,
//! against `-6.9` at `Nodes(2000)`. A wall-clock noise floor is genuinely
//! wider, and wider still at the shorter budget where a scheduling hiccup is
//! a larger fraction of a decision — which is the reason `CLAUDE.md` insists
//! on running these one at a time. Both are nowhere near the candidate's
//! interval: the closest approach is the `TimeMs(20)` control's upper bound
//! against that budget's lower bound, and they are 106 points apart.
//!
//! ## Reproducing
//!
//! ```text
//! cargo run --release -p duels-arena -- match \
//!     --agent-a mcts-eval --agent-b mcts-eval:base=rollout \
//!     --games 1200 --budget nodes:2000 --seed 1 --sprt-elo0 0 --sprt-elo1 20
//! cargo run --release -p duels-agent-mcts-eval --example leaf_bench
//! cargo run --release -p duels-eval --example calibrate -- 200
//! ```
//!
//! # The other knobs
//!
//! Every remaining [`Config`] field is `mcts-uct`'s, at `mcts-uct`'s tuned
//! value, and its measurement lives in that crate's documentation rather than
//! being restated here:
//!
//! - [`Config::race`] — [`RaceWeights::TIER1_ONLY`]'s terminal rails,
//!   `+26.1` Elo, additive with this crate's leaf value (see the composition
//!   table above), and non-default because the hypothesis it was built to test
//!   did not survive.
//! - [`Config::prior`] — [`PriorMode`], `duels-strategy` steering the tree.
//!   Reproducibly steers visits towards races; reproducibly fails to convert
//!   that into Elo (`+11.7` at `Nodes` with an interval containing zero,
//!   `-33` at `TimeMs` once its 6-8% throughput cost is paid).
//! - [`Config::root_determinizations`] — root ensembling. Measured at `N` of
//!   1, 2, 4 and 8, at two budget kinds, against a same-configuration control:
//!   no gain anywhere, a mild loss by `N = 8`.
//!
//! The value convention (every node accumulates the result from
//! [`duels_core::Player::One`]'s perspective; the zero-sum flip happens once,
//! at selection) and the widening rule are documented in the `tree` module.
//!
//! # Example
//!
//! ```
//! use duels_agent_mcts_eval::MctsEvalAgent;
//! use duels_agents_api::{Agent, Budget};
//! use duels_core::engine;
//!
//! let mut agent = MctsEvalAgent::new(7);
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

pub use leaf::LeafValue;
pub use rollout::{RaceWeights, RolloutWeights, RAIL};
pub use tree::{Config, PriorMode, RootStats};

/// Monte Carlo Tree Search with explicit chance nodes, scoring each leaf with
/// half a playout and half [`duels_eval`]'s evaluation.
#[derive(Debug)]
pub struct MctsEvalAgent {
    cfg: Config,
    rng: StdRng,
    /// Simulations run over the agent's whole lifetime, for throughput
    /// reporting.
    total_simulations: u64,
    /// Nodes allocated during the most recent search.
    last_tree_size: usize,
    /// What the most recent search concluded about its root, or `None` if the
    /// most recent decision was forced and no search happened.
    last_root: Option<RootStats>,
}

impl MctsEvalAgent {
    /// A new agent with the default configuration, seeded from `seed`.
    ///
    /// The [`duels_eval::Config`] the search will score against is **not**
    /// captured here — it is read in `tree::Tree::new`, per search. See the
    /// crate docs' "Tracking `duels-eval` live" section for why that
    /// distinction is load-bearing rather than incidental.
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
            last_root: None,
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

    /// What the most recent `choose` call's search concluded about its root
    /// position: the backed-up win probability, and the root visit
    /// distribution over the legal actions. See [`RootStats`] for what the
    /// value is and — importantly, for anyone fitting `duels-eval` against it
    /// — what it already contains.
    ///
    /// `None` when the most recent decision was **forced** (one legal action),
    /// because `choose` returns it without searching at all, and so there is
    /// no search verdict to report. A caller collecting a corpus should skip
    /// those plies rather than substitute anything for them.
    ///
    /// Reading this changes nothing: it is a snapshot the search already had.
    pub fn last_root(&self) -> Option<&RootStats> {
        self.last_root.as_ref()
    }
}

impl Agent for MctsEvalAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "mcts-eval".to_string(),
            version: "1.0.0".to_string(),
            params: self.cfg.describe(),
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action], budget: Budget) -> Action {
        assert!(
            !legal.is_empty(),
            "choose must not be called with no legal actions"
        );
        // Cleared first, so a forced move can never leave the *previous*
        // search's verdict readable as if it were this decision's.
        self.last_root = None;
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
        // Read-only, and read here rather than recomputed later because the
        // trees are dropped at the end of this call.
        self.last_root = tree::root_stats(&trees);

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
/// With `n == 1`, the default, both arms reduce to the plain thing:
/// `total.max(1)` simulations, or a single deadline `total` milliseconds after
/// the first simulation.
#[derive(Debug)]
enum Slices {
    Nodes {
        total: u64,
        n: u64,
    },
    Time {
        total_ms: u64,
        n: u64,
        /// Captured on the first slice.
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

    fn play(seed: u64, seat: Player, budget: Budget) -> (GameResult, u64) {
        let mut mcts = MctsEvalAgent::new(seed ^ 0x0BAD_1DEA_0BAD_1DEA);
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
            let action = if state.current_player() == seat {
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

    /// **The crate's identity, asserted rather than described.** The default
    /// configuration is exactly the one the `+89.2` Elo measurement was taken
    /// on: `leaf=blend:0.5` with `c` rescaled to `0.5`, and every other knob
    /// left at `mcts-uct`'s tuned value.
    ///
    /// If this test ever has to be *changed*, the crate documentation's
    /// measurement tables no longer describe the shipped agent, and the fix is
    /// to re-measure rather than to update the constants.
    #[test]
    fn the_default_configuration_is_the_one_that_was_measured() {
        let cfg = Config::default();
        assert_eq!(cfg.leaf, LeafValue::Blend { weight: 0.5 });
        assert_eq!(cfg.exploration.to_bits(), 0.5f64.to_bits());
        // The rescaling relation the blend's algebra derives, spelled out:
        // `c = c0 * (1 - weight)` against `mcts-uct`'s tuned `c0 = 1.0`.
        let LeafValue::Blend { weight } = cfg.leaf else {
            panic!("the default leaf is a blend")
        };
        assert_eq!(cfg.exploration.to_bits(), (1.0 * (1.0 - weight)).to_bits());
        // Everything else is untouched.
        assert_eq!(cfg.race, RaceWeights::NEUTRAL);
        assert_eq!(cfg.prior, PriorMode::None);
        assert_eq!(cfg.rollout, RolloutWeights::BIASED);
        assert_eq!(cfg.root_determinizations, 1);
        assert_eq!(cfg.chance_widen_c.to_bits(), 1.0f64.to_bits());
        assert_eq!(cfg.chance_widen_alpha.to_bits(), 0.5f64.to_bits());
    }

    /// **The live-tracking design, asserted rather than described.** There is
    /// no configuration field to pin, so the check has to be behavioural: the
    /// search's static leaf value must equal what a `duels_eval::Root` built
    /// from `duels_eval::Config::default()` produces — bit for bit, at
    /// whatever that default currently is.
    ///
    /// This is what a `Config::eval_generation` pin, or a golden-values table,
    /// would break. See the crate docs' "Tracking `duels-eval` live" section:
    /// the opposite choice from `mcts-uct`'s is deliberate.
    #[test]
    fn the_evaluation_configuration_is_duels_evals_live_default() {
        for seed in 0..8u64 {
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x7E57);
            for _ in 0..(6 + seed % 9) {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let a = legal[0];
                engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action");
            }
            if state.result().is_some() {
                continue;
            }

            // What the tree will actually score a leaf at this position with.
            let got = tree::static_leaf_value_for_test(&state);
            // What `duels-eval`'s current default says, computed here from
            // scratch through the public API.
            let root = duels_eval::Root::new(
                &state,
                state.current_player(),
                duels_eval::Config::default(),
            );
            let want = duels_eval::win_probability(&state, Player::One, &root);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "seed {seed}: the search is not scoring against duels-eval's live default"
            );
        }
    }

    #[test]
    fn spec_reports_the_expected_name_version_and_params() {
        let agent = MctsEvalAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "mcts-eval");
        assert_eq!(spec.version, "1.0.0");
        assert!(spec.params.contains("c=0.500"), "{}", spec.params);
        assert!(spec.params.contains("leaf=blend(0.500)"), "{}", spec.params);
        assert!(spec.params.contains("chance="), "{}", spec.params);
        assert!(spec.params.contains("rollout="), "{}", spec.params);
    }

    /// The spec string has to record the *evaluation itself*, not a
    /// generation label — that is what replaces the pin `mcts-uct` used, and
    /// it is what makes a results file from before a `duels-eval` round
    /// distinguishable from one after it. See the crate docs.
    #[test]
    fn the_spec_records_the_live_evaluation_configuration() {
        let params = MctsEvalAgent::new(1).spec().params;
        let live = duels_eval::Config::default().params_string();
        assert!(
            params.ends_with(&format!("eval={live}")),
            "the spec must carry the whole live duels-eval configuration: {params}"
        );
        // ...and no frozen generation label anywhere.
        assert!(!params.contains("evalgen="), "{params}");

        // The pure-playout ablation does not score anything, so it says so
        // rather than recording a configuration it never reads.
        let base = MctsEvalAgent::with_config(1, Config::rollout_base())
            .spec()
            .params;
        assert!(base.contains("eval=unused"), "{base}");
        assert!(base.contains("leaf=rollout"), "{base}");
        assert!(base.contains("c=1.000"), "{base}");
    }

    #[test]
    fn a_single_legal_action_is_returned_without_searching() {
        let mut agent = MctsEvalAgent::new(3);
        let state = engine::new_game(3);
        let only = [engine::legal_actions(&state)[0]];
        let chosen = agent.choose(&state.observation(), &only, Budget::Nodes(10_000));
        assert_eq!(chosen, only[0]);
        assert_eq!(agent.total_simulations(), 0, "no search was needed");
    }

    #[test]
    fn every_returned_action_is_one_of_the_offered_ones() {
        let mut agent = MctsEvalAgent::new(11);
        let state = engine::new_game(11);
        let legal = engine::legal_actions(&state);
        for _ in 0..5 {
            let a = agent.choose(&state.observation(), &legal, Budget::Nodes(20));
            assert!(legal.contains(&a));
        }
    }

    #[test]
    fn a_node_budget_runs_exactly_that_many_simulations() {
        let mut agent = MctsEvalAgent::new(5);
        let state = engine::new_game(5);
        let legal = engine::legal_actions(&state);
        agent.choose(&state.observation(), &legal, Budget::Nodes(37));
        assert_eq!(agent.total_simulations(), 37);
        agent.choose(&state.observation(), &legal, Budget::Nodes(3));
        assert_eq!(agent.total_simulations(), 40);
    }

    #[test]
    fn a_time_budget_returns_promptly_and_does_some_work() {
        let mut agent = MctsEvalAgent::new(9);
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
            let mut agent = MctsEvalAgent::new(seed);
            agent.choose(&obs, &legal, Budget::Nodes(200))
        };
        assert_eq!(pick(4), pick(4));
    }

    /// `choose` exactly as `mcts-uct`'s reads: one determinization, one tree,
    /// the whole node budget, that agent's move-selection rule — and, since it
    /// drives `tree::Tree::legacy_simulate` rather than `simulate`, that
    /// agent's search too.
    ///
    /// It is a copy on purpose: a test that called the live code would prove
    /// nothing.
    fn mcts_uct_choose(
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

    /// The ablation control is the real thing, at the whole-agent level:
    /// [`Config::rollout_base`] is `mcts-uct` move for move, over whole
    /// seeded games, against the verbatim copy above.
    ///
    /// Checked over games rather than only at the opening position, so that
    /// the RNG streams have to stay in step across dozens of `choose` calls,
    /// chance nodes, pending choices and all.
    /// `tree::tests::the_rollout_base_grows_the_mcts_uct_tree_node_for_node`
    /// is the stronger, arena-for-arena form of the same claim.
    #[test]
    fn the_rollout_base_is_the_mcts_uct_agent_move_for_move() {
        for seed in 0..8u64 {
            let cfg = Config::rollout_base();
            let mut agent = MctsEvalAgent::with_config(seed, cfg);
            // The same seed, so the same stream, driven by the copy above.
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
                let want = mcts_uct_choose(&mut legacy_rng, cfg, &obs, &legal, budget);
                assert_eq!(
                    got, want,
                    "seed {seed}, decision {decisions}: the rollout base is not mcts-uct"
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
            LeafValue::Blend { weight: 0.5 },
            LeafValue::Rollout,
            LeafValue::Static,
            LeafValue::Truncated { plies: 8 },
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
                let mut mcts = MctsEvalAgent::with_config(
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

    /// The spec string a results file records has to name the leaf value, or
    /// an arena run cannot be told apart from the default after the fact.
    #[test]
    fn the_spec_reports_the_leaf_value() {
        let describe = |leaf| {
            MctsEvalAgent::with_config(
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
                let mut mcts = MctsEvalAgent::with_config(
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
            MctsEvalAgent::with_config(
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
                let mut mcts = MctsEvalAgent::with_config(
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
            MctsEvalAgent::with_config(
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
            let mut agent = MctsEvalAgent::with_config(
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
        let mut agent = MctsEvalAgent::with_config(
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
            println!("seed {seed} (mcts-eval as {seat}): {result:?}");
        }
        assert!(sims > 0, "the agent never searched");
    }

    /// Even at a CI-sized budget the search should already be clearly better
    /// than uniform-random play. This is a loose smoke test, not the real
    /// strength measurement (the crate docs have that); it exists so that a
    /// sign error in backpropagation, a perspective flip, or an inverted
    /// evaluation cannot land silently.
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
             suspect backpropagation sign, UCB1 perspective, the evaluation's \
             sign, or chance handling"
        );
    }

    /// [`MctsEvalAgent::last_root`] must be **only** a readout: adding it may
    /// not change a single decision the agent makes.
    ///
    /// Driven the way the gold-standard identity tests in this repository are:
    /// whole seeded games, move for move. Here the "before" arm is the agent
    /// itself with `last_root` never read, which is the strongest statement
    /// available now that the field is not optional — so what this really
    /// pins is that reading it is side-effect free and that the RNG stream is
    /// untouched by populating it (equal total simulations, equal moves).
    #[test]
    fn reading_the_root_readout_changes_no_decision() {
        for seed in 0..6u64 {
            let mut quiet = MctsEvalAgent::new(seed);
            let mut watched = MctsEvalAgent::new(seed);
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FFEE);
            loop {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let obs = state.observation();
                let a = quiet.choose(&obs, &legal, CI_BUDGET);
                let b = watched.choose(&obs, &legal, CI_BUDGET);
                // The readout is consulted on every ply of the second arm and
                // on none of the first; the two must still agree.
                let readout = watched.last_root().cloned();
                assert_eq!(a, b, "seed {seed}: the readout moved a decision");
                assert_eq!(quiet.total_simulations(), watched.total_simulations());
                match readout {
                    None => assert_eq!(legal.len(), 1, "only a forced move has no verdict"),
                    Some(_) => assert!(legal.len() > 1, "a forced move was searched"),
                }
                engine::apply(&mut state, a, &mut rng).expect("a legal action");
            }
        }
    }

    /// What the readout says has to be internally consistent with the search
    /// that produced it, on real positions rather than a hand-built tree.
    #[test]
    fn the_root_readout_agrees_with_the_search_it_reports_on() {
        let budget = Budget::Nodes(200);
        let mut seen_searched_plies = 0u32;
        for seed in 0..4u64 {
            let mut agent = MctsEvalAgent::new(seed);
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xD15EA5E);
            loop {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let obs = state.observation();
                let mover = state.current_player();
                let chosen = agent.choose(&obs, &legal, budget);
                if let Some(r) = agent.last_root() {
                    seen_searched_plies += 1;
                    // A win probability, so it is a probability.
                    assert!(
                        (0.0..=1.0).contains(&r.value),
                        "seed {seed}: root value {} is not in [0, 1]",
                        r.value
                    );
                    assert_eq!(r.mover, mover, "the readout named the wrong mover");
                    assert_eq!(
                        r.value_for_mover(),
                        if mover == Player::One {
                            r.value
                        } else {
                            1.0 - r.value
                        }
                    );
                    // One entry per offered action, and nothing else.
                    assert_eq!(r.policy.len(), legal.len());
                    for (action, _) in &r.policy {
                        assert!(legal.contains(action), "policy named an unoffered action");
                    }
                    // Every simulation passes through the root and then through
                    // exactly one root child, so the child visits can only fall
                    // short of the root's by the simulations that ended at the
                    // root itself — of which there are none, since the root is
                    // never a leaf.
                    let policy_visits: u64 = r.policy.iter().map(|&(_, n)| u64::from(n)).sum();
                    assert!(
                        policy_visits <= r.visits,
                        "seed {seed}: children saw {policy_visits} of the root's {} visits",
                        r.visits
                    );
                    assert!(r.visits > 0, "a searched position ran no simulation");
                    // `best_of` picks on visits, ties broken by value, so the
                    // chosen action must be *a* visit-count maximum.
                    let top = r.policy.iter().map(|&(_, n)| n).max().unwrap_or(0);
                    let chosen_visits = r
                        .policy
                        .iter()
                        .find(|&&(a, _)| a == chosen)
                        .map(|&(_, n)| n)
                        .expect("the chosen action is in the policy");
                    assert_eq!(
                        chosen_visits, top,
                        "seed {seed}: played an action with {chosen_visits} visits \
                         while another had {top}"
                    );
                }
                engine::apply(&mut state, chosen, &mut rng).expect("a legal action");
            }
        }
        assert!(
            seen_searched_plies > 20,
            "only {seen_searched_plies} searched plies; the test proved little"
        );
    }
}
