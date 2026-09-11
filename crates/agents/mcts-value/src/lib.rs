//! `duels-agent-mcts-value`: Monte Carlo Tree Search with UCT selection and
//! **explicit chance nodes**, whose leaf value is [`duels_value`]'s **learned**
//! four-way outcome model — no playout at all — at the re-derived exploration
//! constant `c = 0.15`.
//!
//! # Read this before you read the Elo numbers
//!
//! This agent measures very strongly against [`mcts-eval`][mcts_eval], the
//! ladder's previous champion: `+140.1` Elo at `Nodes(32000)`, `+91.4` at the
//! ladder's production `Nodes(2000)`, `+140.6` at `TimeMs(1000)`, reproduced on
//! disjoint seed ranges at every one of those budgets. Those numbers are real,
//! reproducible, and recorded below with the directory each came from. The
//! production-budget figure was re-measured after `duels-core`'s chance model
//! was fixed to condition on the public guild mask (R-105, R-110, PR #61) and
//! holds at `+84.9` `[+60.1, +109.7]` over 800 games —
//! `arena/results/experiments/post-r105r110-confirm/`.
//!
//! **They are not a general strength improvement, and this crate is not
//! claiming to be one.** A round robin that measures the same margin *through a
//! third party* finds it almost entirely gone: 28% of it survives being
//! measured through `mcts-uct` and 12% through `alphabeta`, both intervals
//! containing zero. The victory-kind breakdown says why, and it is specific:
//! the learned leaf converts scientific supremacies `mcts-eval` never sees, but
//! it wins them out of its *own* civilian column rather than adding to its
//! total. What this agent is, on the evidence, is **a targeted counter to
//! `mcts-eval`'s already-documented science-value miscalibration** (see
//! `duels-arena`'s `science_residual` example, PR #57) rather than a better
//! player in general. The "What it does not measure" section below is the
//! load-bearing one; do not quote the headline numbers without it.
//!
//! This agent shipped *registered but deliberately unrated* — playable and
//! spec-addressable, off `duels_arena::leaderboard::LADDER` — precisely
//! because of the paragraph above, with promotion left as the project owner's
//! decision on its own evidence. **That decision has since been made: it is on
//! the ladder and it is `duels_arena::leaderboard::CHAMPION`.** Nothing in the
//! evidence changed when the status did, so the honest reading is unchanged
//! too: *strength against `mcts-eval` is thoroughly established; strength
//! against a third party has never been shown.* Two things follow, and both
//! matter more now that this is the bar `ai-candidate` measures against:
//!
//! * A candidate that beats this agent has cleared a real bar. A candidate
//!   that *loses* to it may only have failed to counter one specific leaf —
//!   check the victory-kind breakdown before reading such a result as weakness.
//! * Its published rating is a joint fit over records that are not transitive,
//!   so the gap between it and `mcts-eval` on the board reads smaller than the
//!   head-to-head number. Both are correct and they answer different questions.
//!
//! `duels_arena::leaderboard::CHAMPION`'s own docs carry the same caveat, so a
//! reader who arrives from the leaderboard rather than from here still meets
//! it.
//!
//! # Why this is its own crate rather than a flag on `mcts-eval`
//!
//! The same argument that made `mcts-eval` its own crate rather than a flag on
//! `mcts-uct`, one step further along. The learned leaf was built and measured
//! as `mcts-eval`'s opt-in [`LeafValue::Learned`], and promoting it *there*
//! would have moved that crate's tuned [`Config::exploration`] — a leaf that
//! **replaces** the playout rather than mixing with it needs a completely
//! different `c` (see "The exploration constant" below) — and re-defined the
//! ladder's champion in the same breath as the PR that merely measured a leaf
//! value. `mcts-eval` keeps its default; this is the other configuration, as
//! its own agent, with its own internally consistent `Config`.
//!
//! So the ablation is one spec string away in one binary:
//!
//! ```text
//! cargo run --release -p duels-arena -- experiment \
//!     --candidate mcts-value --control mcts-value:base=eval \
//!     --pairs 400 --budget nodes:2000 --seed 1 --label mcts-value-ablation
//! ```
//!
//! [`Config::eval_base`] is that control, and it is not a *claim* about being
//! `mcts-eval` — it is **checked** to be, node for node, against a verbatim
//! frozen copy of that agent's `expand`/`select_ucb1`/`leaf_value`/`simulate`
//! (`tree::tests::the_copied_search_is_the_mcts_eval_search_node_for_node`,
//! with `tree::tests::the_default_search_is_not_the_mcts_eval_search` as the
//! non-vacuity check and
//! `tests::the_eval_base_is_the_mcts_eval_agent_move_for_move` as the
//! whole-agent, whole-game form). [`Config::rollout_base`] is inherited from
//! `mcts-eval` and is checked the same way against `mcts-uct`.
//!
//! The search machinery itself — `tree`, `chance`, `rollout`, `leaf` and the
//! [`Agent`] impl below — is a deliberate **copy** of `mcts-eval`'s (which is
//! itself a copy of `mcts-uct`'s), not a dependency on it.
//! `docs/conventions.md`'s "agent crates are self-contained" invariant forbids
//! one agent crate depending on another, and this is the same accepted,
//! intentional duplication `mcts-eval` already carries. `duels-value` and
//! `duels-eval` are a different matter: they are shared libraries *below* the
//! agents, which is exactly what the layering is for.
//!
//! ## The copies drift, and nothing mechanical stops them
//!
//! Worth stating plainly, because this crate ran into it inside a day. `chance`
//! and `rollout` are **byte-identical** to `mcts-eval`'s and are meant to stay
//! that way; `tree` and `lib` diverge only in [`Config`] and documentation. But
//! there is no test, and no lint, that says so — a test in an agent crate
//! cannot read another agent crate, which is the same rule that forced the copy
//! in the first place.
//!
//! It is not a hypothetical. `duels-core` PR #61 changed the chance model to
//! condition on the public guild mask and had to hand-edit **three** copies of
//! `chance.rs` (`mcts-uct`'s, `mcts-eval`'s, and this crate's) to keep them the
//! same file. A missed one would not have failed anything here: the ablation
//! tests below compare the live search against a *frozen copy inside this
//! crate*, so they would have gone on passing while the crate's claim to be
//! `mcts-eval`'s search quietly stopped being true.
//!
//! So when changing anything in the shared search, `diff` the copies:
//!
//! ```text
//! for f in chance.rs rollout.rs; do
//!     diff crates/agents/mcts-eval/src/$f crates/agents/mcts-value/src/$f
//! done
//! ```
//!
//! And note what the frozen `eval_legacy` copy in `tree` *does* protect, since
//! it is a narrower thing than it first looks: it catches a change to the live
//! search made **inside this crate**, not a change made next door.
//!
//! # The weights are pinned, and that is the opposite of `mcts-eval`'s call
//!
//! `mcts-eval` reads `duels_eval::Config::default()` **live** and holds no
//! version pin, on the argument that its identity *is* "`duels-eval` inside a
//! search" and it should therefore get stronger automatically as future
//! evaluation rounds land. That argument does not transfer here, and this crate
//! makes the opposite choice on purpose:
//!
//! - **`duels-value`'s weights are a fitted artefact, not a tuned
//!   configuration.** A retrain is not an incremental improvement to a
//!   hand-written weight vector that a code owner reviewed; it is a different
//!   function, and the honest expectation is that it changes this agent's
//!   behaviour everywhere at once.
//! - **The measured effect here is narrow and mechanism-specific** (see below).
//!   An effect that runs through one opponent's calibration error is precisely
//!   the kind that a retrain can silently delete, leaving the crate docs'
//!   numbers describing an agent that no longer exists.
//!
//! So the `golden` module pins twenty fixed positions' predictions to
//! `weights/v1.bin` within a tight tolerance, and pins
//! [`duels_value::default_weights_id`]'s content hash outright. A retrain
//! **fails a test** rather than re-defining the agent, exactly as
//! `docs/conventions.md`'s standing prior asks for a search whose identity is
//! its search and whose value input is incidental. `Config::describe` records
//! the same weights identity in every [`AgentSpec`], so a results file names
//! the model that produced it.
//!
//! # What it measures
//!
//! Every number below is from a `duels-arena experiment` run committed to this
//! branch under `arena/results/experiments/`, named per row. Paired-seed and
//! seat-swapped throughout, against `mcts-eval` at its default unless stated.
//! The candidate was spelled `mcts-eval:leaf=learned,c=0.15` — the
//! configuration [`Config::default`] now *is*.
//!
//! | budget | games | Elo vs `mcts-eval` | per-range | SPRT | directory |
//! |---|---|---|---|---|---|
//! | `Nodes(32000)` | 600 (2 x 300) | **+140.1 [+110.0, +170.2]** | +163.4 / +117.4 | `AcceptH1` | `p0-learned-c0.15/` |
//! | `Nodes(2000)` | 800 (2 x 400) | **+91.4 [+66.5, +116.3]** | +128.6 / +55.9 | `AcceptH1` | `p0-learned-nodes2000/` |
//! | `TimeMs(1000)` | 400 (2 x 200) | **+140.6 [+103.8, +177.5]** | +118.5 / +163.1 | `AcceptH1` | `p1-timems-learned-c0.15/` |
//!
//! The `Nodes(2000)` row is the one that matters for the ladder, because it is
//! the budget the nightly round robin runs at, and it is the weakest of the
//! three — the effect grows with budget rather than being a low-budget
//! artefact. The `TimeMs(1000)` row was taken with each candidate run strictly
//! one after the other on a machine verified quiet by a process snapshot
//! before and after every cell, per `docs/conventions.md`'s warning about
//! load-sensitive wall-clock runs.
//!
//! ## The exploration constant, re-derived
//!
//! `docs/conventions.md`: *"any future change to what a leaf backs up should
//! re-derive `c` before measuring"*. This is that re-derivation, and it was
//! worth about 83 Elo. [`LeafValue::Blend`]'s `c = c₀·(1 - w)` rescaling gives
//! no guidance for a leaf that replaces the playout outright — there is no
//! Bernoulli spread left to shrink — so `c` was swept at `Nodes(32000)`, 600
//! games per row:
//!
//! | `c` | Elo vs `mcts-eval` | directory |
//! |---|---|---|
//! | `0.10` | +126.7 [+97.1, +156.4] | `p0-learned-c0.1/` |
//! | **`0.15`** (this crate's default) | **+140.1 [+110.0, +170.2]** | `p0-learned-c0.15/` |
//! | `0.25` | +101.0 [+72.1, +130.0] | `p0-learned-c0.25/` |
//! | `0.50` (inherited from `mcts-eval`) | +57.2 [+29.0, +85.3] | `spike-learned-pure/` |
//!
//! A bracketed interior optimum, so read it as *"somewhere in `[0.10, 0.15]`"*
//! rather than as a tuned peak; `0.15` is the argmax of four points at `+/-`
//! about 30 Elo each. The inherited `0.5` was leaving roughly 83 Elo on the
//! table, which is why a sweep here is not optional.
//!
//! ## Replacing the playout beats mixing with it, at this leaf
//!
//! The reverse of what the hand-crafted evaluation did, and the reason this
//! agent is a *pure* learned leaf rather than the blend `mcts-eval`'s own
//! history would predict. [`LeafValue::LearnedBlend`] at `weight = 0.5` stays
//! available and is the measured second-best option:
//!
//! | leaf | `Nodes(32000)` | `TimeMs(1000)` |
//! |---|---|---|
//! | **`learned`, `c = 0.15`** | **+140.1** | **+140.6** |
//! | `learned_blend:0.5`, `c = 0.5` | +106.1 (`spike-learned-blend/`) | +68.5 (`p1-timems-learnedblend-c0.5/`) |
//!
//! At the inherited `c = 0.5` the blend led at a fixed node count (`+106.1`
//! against `+57.2`); re-deriving `c` reverses that, and a wall-clock budget
//! widens the reversal to better than two to one. The wall-clock half is a cost
//! story rather than an accuracy one and `duels-value`'s `value_bench` called
//! it in advance: a leaf that **replaces** the playout runs at about `0.35x` a
//! playout's cost against the blend's `1.45x`, so a fixed clock buys it roughly
//! three times the simulations. A leaf **added** to a playout and one that
//! **replaces** it have opposite wall-clock economics; that generalises past
//! this crate.
//!
//! # What it does *not* measure — the part that decides what this agent is
//!
//! ## The margin does not survive a third party
//!
//! A mini round robin at `Nodes(2000)`, 400 games per pairing, asking whether
//! the direct margin shows up as a rating difference measured through an
//! opponent neither side was tuned against:
//!
//! | comparison | Elo | interval | share of direct | directory |
//! |---|---|---|---|---|
//! | direct: candidate - `mcts-eval` | +91.5 `+/- 12.7` | excludes zero | — | `p0-learned-nodes2000/` |
//! | indirect, via `mcts-uct` | +25.5 `+/- 27.8` | **[-28.9, +79.9]** | **28%** | `p2-cand-vs-mctsuct/`, `p2-ctrl-vs-mctsuct/` |
//! | indirect, via `alphabeta` | +11.2 `+/- 36.1` | **[-59.5, +82.0]** | **12%** | `p2-cand-vs-alphabeta/`, `p2-ctrl-vs-alphabeta/` |
//!
//! Both indirect intervals contain zero. A joint Bradley-Terry fit over all
//! five head-to-head records puts the candidate at 1213.2 and `mcts-eval` at
//! 1139.3 — `+74.0` where the direct match says `+91.5`, and that residual is
//! real intransitivity, not noise in one cell.
//!
//! ## The mechanism: route substitution, not extra wins
//!
//! Sharper than "it beats one opponent". Victory kinds from the two
//! `mcts-uct` pairings above, 400 games each:
//!
//! | vs `mcts-uct` at `Nodes(2000)` | this agent | `mcts-eval` |
//! |---|---|---|
//! | wins by scientific supremacy | **89** | 10 |
//! | wins by civilian score | 171 | **237** |
//! | wins by military supremacy | 38 | 34 |
//! | **total wins** | **298** | 287 |
//!
//! The learned leaf genuinely *sees the science race* — 89 scientific
//! supremacies against 10 is not a subtle difference — and it pays for them
//! almost exactly out of its own civilian column. Against `mcts-eval` the same
//! behaviour scores heavily only because `mcts-eval` concedes 129 science games
//! in 800 and wins one. `science_share` reads `Inconclusive` in all four
//! round-robin pairings and never `Pass`, always for the same reason: the
//! control wins too few science games to form a ratio against.
//!
//! Read plainly, and not as a gotcha: this agent has learned a real skill
//! `mcts-eval`'s own evaluation lacks — correctly recognizing and converting
//! scientific-supremacy chances. `mcts-eval`'s side of that gap is not a
//! surprise; `duels-arena`'s `science_residual` example (PR #57) measured and
//! documented it independently, before this line of work started. Against
//! `mcts-eval` that skill produces a very large margin because `mcts-eval`
//! concedes 129 of 800 science games and wins exactly one — a specific,
//! measured weakness in `mcts-eval`'s evaluation, not a flaw in how this agent
//! wins. Against `mcts-uct`, which does defend the race, the same skill still
//! shows up clearly (89 conversions against 10) even though the net win total
//! is close (298 vs 287) at this training stage — right now the extra science
//! wins are trading against this agent's own civilian wins more than they are
//! adding to the total. That is a measurement of where this agent's edge
//! currently sits, not a verdict on whether science strength is real or worth
//! having: a higher science-conversion rate is exactly what good play looks
//! like, and further training on a bigger, better corpus should be expected to
//! turn more of that skill into net wins against tougher opponents too.
//!
//! ## The model itself is known to be incoherent
//!
//! `duels-value`'s `tests/probability_coherence.rs` measures that the shipped
//! weights do not satisfy `P(win | One) + P(win | Two) ~= 1`: mean absolute gap
//! `0.0559`, p90 `0.1208`, max `0.4814`, with **43% of legal positions missing
//! a 0.05 bound** and the opening's two perspectives summing to `0.9396` on
//! average. That file asserts the failure deliberately, so it stays a visible
//! fact. Two things bound the damage — this search only ever asks for
//! `Player::One`, so one consistent scale is used throughout and a tree can
//! never disagree with itself about who is winning, and the aggregate
//! calibration is good (Brier `0.172`, monotone reliability) — but the
//! incoherence is **concentrated on science-lead positions**, which is the same
//! place this agent's Elo comes from. Read that as known headroom that is not
//! architectural: an antisymmetric head would make the property exact for free.
//!
//! ## An inference-time patch for it, checked offline before being wired in
//!
//! `docs/roadmap.md`'s Tier 0-B: averaging the net's two disagreeing
//! perspectives on the same position — `p_sym = (P(win|One) + (1 -
//! P(win|Two))) / 2` — is a free two-member ensemble over the incoherence
//! above, *if* it actually predicts better. Checked, not assumed, on the `v1`
//! corpus's held-out test split (`arena/corpus/mcts-eval-nodes2000.jsonl`,
//! `seed % 10 == 9`, the same split rule `tools/train_value.py` uses; 10,000
//! games, 1,343,298 (position, perspective) rows). The exact corpus `v2.bin`
//! itself trained on is not checked into the repository (it is `arena/corpus/`-
//! gitignored, like every corpus, and regenerating it — 55,000 games — was out
//! of this check's ~30-minute scope), but this corpus is a fully disjoint seed
//! range from either of `v2`'s two training corpora, so it is a valid held-out
//! set for it too:
//!
//! | predictor | Brier | log loss | ROC AUC |
//! | --- | --- | --- | --- |
//! | `win_probability()` (single perspective) | 0.17436 | 0.51480 | 0.81680 |
//! | **`p_sym` (averaged)** | **0.17266** | **0.51006** | **0.82017** |
//!
//! `p_sym` won on every metric measured, so it was implemented as a new,
//! opt-in leaf: [`LeafValue::LearnedSymmetric`]. It costs **one extra forward
//! pass per leaf** — both perspectives are evaluated instead of one — so it
//! trades throughput for accuracy, the same shape as
//! [`LeafValue::LearnedBlend`]'s trade against [`LeafValue::Learned`].
//!
//! **Arena result (exploratory, not a promotion decision):** 400 paired-seed,
//! seat-swapped games at `nodes:2000` against this crate's default
//! (`LeafValue::Learned`), one seed range — well under this project's
//! promotion convention of ~2,000 games at a tightened SPRT bound
//! (`docs/conventions.md`, `duels-value`'s crate docs' "Recalibrating the
//! test"), and explicitly not run to that standard here:
//!
//! | candidate | games | W-L-D | Elo | 95% CI | SPRT (`elo1=20`) | mechanism gate |
//! | --- | --: | --- | --: | --- | --- | --- |
//! | `leaf=learned_symmetric` vs `leaf=learned` | 400 | 222-178-0 | +38.3 | [+4.1, +72.5] | Continue (Inconclusive) | Pass |
//!
//! (`arena/results/experiments/sym-check-learned-symmetric-vs-learned/`.)
//! Victory kinds: the candidate won 31 military / 41 science / 150 civilian;
//! the control 32 military / 30 science / 112 civilian / 4 tiebreak — a
//! similar mix to the default's own, not a shift into one victory kind, which
//! the mechanism gate's `Pass` on all three shares confirms.
//!
//! **Read this plainly: a positive point estimate with a 95% CI that
//! excludes zero at the *cell* level, but the pooled SPRT itself reads
//! `Continue` (this project's `Inconclusive`, not `AcceptH1`) at the
//! `elo1 = 20` bound** — a single 400-game range is not enough evidence to
//! either accept or reject a `+20`-Elo-or-larger effect here, exactly the
//! shape `duels-value`'s crate docs describe for `v2` vs `v1`'s own first-pass
//! cell. This is a first read, not a confirmed effect: a second, disjoint
//! seed range and a larger sample at a tighter bound (`elo1 = 10`, ~2,000
//! games, this project's promotion convention) is what turning this into a
//! promotion decision would need, and that is future work, not something this
//! exploratory check did.
//!
//! Not the default. Opt-in, reachable as `mcts-value:leaf=learned_symmetric`
//! (or the `lsym` alias).
//!
//! # Why chance nodes
//!
//! 7 Wonders Duel is a two-player zero-sum *stochastic* game: the cards behind
//! the face-down slots of the current age are unknown when a move is chosen,
//! and taking a card can uncover them. There is no player-private information —
//! both players always see the same public state — so a plain alternating-move
//! tree would silently pretend the reveals were part of the mover's choice.
//!
//! This agent therefore builds a tree with three kinds of node:
//!
//! - **decision** nodes, one player to move, children = the legal actions,
//!   selected by UCB1;
//! - **chance** nodes, inserted between an action and the position it leads to
//!   whenever the engine says the action resolves randomness, children =
//!   possible reveals, selected **by their real probability**, never by UCB1;
//! - **terminal** nodes, where the [`duels_core::GameResult`] is settled.
//!
//! ## How chance is handled, precisely
//!
//! 1. **The root is determinized.** `choose` only ever sees an
//!    [`Observation`], so it calls [`Observation::sample_state`] once per call
//!    to get one concrete world consistent with public knowledge.
//! 2. **Reveals inside the tree are *not* taken from that world.** Every chance
//!    node re-draws its outcome from the distribution the engine computes from
//!    public information alone (`engine::chance_outcomes`/`hidden_info`), and
//!    applies it with `engine::apply_with_outcome`, which rewrites the hidden
//!    layout to stay publicly consistent. So the tree integrates over reveals
//!    rather than committing to the root's guess, and the agent can never
//!    exploit knowledge of a card it should not know.
//! 3. **The draw is exact but not enumerated.** A two-slot reveal has hundreds
//!    of outcomes; the `chance` module draws from that distribution in O(1) and
//!    reports the drawn outcome's exact probability.
//! 4. **Progressive widening is an approximation.** By default the number of
//!    distinct outcome children grows as `sqrt(visits)` and further visits
//!    re-select an existing child in proportion to its probability. Set
//!    [`Config::chance_widen_alpha`] to `1.0` with a large
//!    [`Config::chance_widen_c`] to recover the unbiased estimator.
//! 5. **Two sources of randomness are only root-determinized:** the composition
//!    and order of the *next* age's deck, and the four wonders not yet offered
//!    during the draft.
//!
//! # The leaf value (`Config::leaf`)
//!
//! [`LeafValue`] is inherited from `mcts-eval` whole, so this crate's default
//! and its control arms are all one enum and one dispatch point. What changed
//! is only which variant [`Config::default`] names. See the `leaf` module for
//! the mechanism: where the one [`duels_eval::Root`] is built (for the
//! inherited hand-crafted variants only), why the perspective is always Player
//! One, why the learned variants need no sigmoid calibration on the way out,
//! and the algebra relating [`LeafValue::Blend`]'s weight to the exploration
//! constant.
//!
//! The learned variants consume **no randomness**, exactly as the hand-crafted
//! static ones do, which is what keeps the tree's RNG stream a property of its
//! chance nodes alone — and is why the ablation tests below can compare arenas
//! node for node across the whole leaf family from one seeded stream.
//!
//! # The other knobs
//!
//! Every remaining [`Config`] field is `mcts-uct`'s, at `mcts-uct`'s tuned
//! value, inherited through `mcts-eval` unchanged, and its measurement lives in
//! those crates' documentation rather than being restated here:
//! [`Config::race`], [`Config::prior`], [`Config::root_determinizations`],
//! [`Config::rollout`], the two widening constants.
//!
//! Two are this crate's own concern:
//!
//! - [`Config::value_summation`] selects `duels_value::Summation`. The
//!   four-way accumulator unroll is the default; it is worth `1.41x` on the
//!   forward pass and measured Elo-neutral at `-1.7 [-25.8, +22.3]` over 798
//!   games (`p1-unroll-ab/`, `p1-unroll-ab-extended/`). It reassociates a
//!   floating-point sum, so `Summation::Serial` stays reachable
//!   (`mcts-value:value_sum=serial`) and the spec string records which order
//!   ran. `Summation::TransposedAxpy` (`mcts-value:value_sum=axpy`) is a third,
//!   opt-in option: it transposes the loop nest rather than the reduction,
//!   which `duels_value`'s own crate docs and `examples/value_bench.rs`
//!   measure at a further real speedup over the unroll — see those for the
//!   number and the Elo validation at production's `TimeMs` budget, and for
//!   why it is not (yet, at time of writing) the default.
//! - [`Config::eval_override`] is inherited and is only read by the inherited
//!   hand-crafted leaves. It does nothing on this crate's default path. Note
//!   that [`Config::eval_base`] deliberately leaves it `None`, so the ablation
//!   control is `mcts-eval` **as shipped** — tracking `duels-eval` live — and
//!   not a frozen snapshot of it.
//!
//! The value convention (every node accumulates the result from
//! [`duels_core::Player::One`]'s perspective; the zero-sum flip happens once,
//! at selection) and the widening rule are documented in the `tree` module.
//!
//! # Specialist objectives (`Config::objective`): how purely can the same
//! weights be searched toward one victory kind?
//!
//! `duels-value`'s crate docs end on a proposal: rather than another blind
//! corpus-size increase, try **specialist value functions/agents trained
//! under different reward shaping** — one rewarded only for scientific
//! supremacy, one only for military, one only civilian — as a deliberately
//! different idea from "add more games of the same kind". [`Objective`] is
//! that idea, built the cheapest way it can be: **no retrain.** `v2.bin`
//! already predicts a four-way softmax over [`duels_value::Outcome`], so a
//! specialist search does not need a new net — it needs to read a different
//! *component* of the existing one, and to score a finished game by a
//! different rule. [`Objective::WinProbability`] (the default, unchanged) is
//! [`duels_value::Dist::win_probability`] at the leaf and [`crate::tree::value_of`]
//! at a terminal, exactly as before, `me` unused. [`Objective::TargetKind`] is
//! [`duels_value::Dist::p`] of the one matching outcome at the leaf,
//! evaluated **for `me`** — the seat this specific tree is searching for, not
//! a game-fixed constant ([`leaf::learned_value`]'s two arms) — and
//! [`crate::tree::objective_value_of`] at a terminal: `1.0` for a win **by
//! `me`** of exactly that kind, `0.0` for everything else — the *other* seat
//! winning by any kind, a draw, or `me` winning by a *different* kind. Both
//! arms are bit-identical to the pre-existing path at the default
//! (`tests::objective_win_probability_is_bit_identical_to_value_of`,
//! `leaf::tests::learned_value_at_win_probability_is_bit_identical_to_win_probability`).
//!
//! **The point of this section is explicitly not strength.** Nothing below is
//! a candidate for `Config::default`, none of it is on the ladder, and a
//! specialist losing to the generalist is the expected, uninteresting result,
//! not a failure. The question this measures is **purity**: does a specialist
//! actually play differently, and does it convert and *provoke* its target
//! victory kind far more than the generalist does — not whether it wins more
//! often.
//!
//! ## A correctness bug this section's first measurement round had, and why
//! it invalidated that round rather than just limiting it
//!
//! An earlier version of [`Objective::TargetKind`]'s reward was anchored to a
//! **literal, game-fixed [`duels_core::Player::One`]** — "did `Player::One` win by kind
//! K" — regardless of which seat the tree calling it was actually searching
//! for, on the reasoning that this mirrored [`crate::tree::value_of`]'s own
//! `Player::One` anchoring. It does not, and the difference is not a matter
//! of degree: "Player One wins" and "Player Two wins" partition the outcome
//! space (up to a draw), so [`crate::tree::Tree::exploit`]'s `1.0 - mean` flip
//! at a Player Two node correctly reads as "Player Two's own win
//! probability". "Player One wins by kind K" and "Player Two wins by kind K"
//! do **not** partition the outcome space — most games, neither player wins
//! by a specific rare kind — so the identical flip at a Player Two node
//! reads as "how often does One *not* get kind K", a quantity dominated by
//! the ~90%+ mass in which *nobody* achieves K and therefore almost constant
//! regardless of Two's actual prospects. A tree built for `Player::Two` was,
//! in effect, searching to advance an all-but-uninformative signal.
//!
//! This was found, not merely suspected, from the first measurement round's
//! own numbers: split by seat, the science specialist showed real, sharp
//! target-seeking behaviour as `Player::One` (100% purity, well-elevated
//! science-race exposure) and **no** target-seeking behaviour at all as
//! `Player::Two` (0% science-race exposure, 0/150 wins in one matchup). That
//! is the exact signature the mechanism above predicts, and it was reported
//! in that round's writeup — but reported as a documented limitation and a
//! "what's next" item, when a defect this large (every pooled purity and
//! race-seeking number in that round mixed a fully-functional `Player::One`
//! search with a non-functional `Player::Two` one) makes every pooled number
//! in that round's tables unreliable, not merely caveated. **The fix
//! (anchor the reward to `Tree::me`, [`crate::tree::Tree`]'s own root seat —
//! instead of a fixed `Player::One`) is what this crate now does,
//! and it needed a full remeasurement, not an addendum, before any purity
//! number here could be trusted.** `tree::tests::exploit_anchors_target_kind_to_me_not_a_fixed_player`
//! and `tree::tests::target_kind_terminal_values_are_anchored_to_me_not_player_one`
//! are the regression tests for it.
//!
//! ## Measurement setup
//!
//! `duels-arena match`, `--budget nodes:2000` (the ladder's production
//! budget), `--games 300` (150 paired, seat-swapped seeds), `--seed 1001`,
//! one match per (specialist, opponent) cell: each of the three specialists
//! (`mcts-value:objective=science`, `objective=military`, `objective=civilian`)
//! against each of four opponents — the generalist itself (`mcts-value`,
//! self-play), `mcts-eval`, `mcts-uct`, and `alphabeta` — plus the generalist
//! against the same four, as the baseline every purity number below is read
//! against. Sixteen cells, 4,800 games total, run against the seat-anchoring
//! fix above (the numbers in the section immediately above this one were
//! withdrawn and re-run from scratch; nothing below reuses a pre-fix figure).
//! This is a descriptive/behavioural measurement, not a strength claim, so it
//! is not run at SPRT-rigor sample sizes; Elo is reported below purely as
//! context.
//!
//! **Sanity check.** Every one of the 4,800 games completed normally (a
//! legality violation panics rather than corrupting a result) with game
//! lengths in the same 36-77-ply range across every specialist and the
//! generalist alike — no stalling, no outlier caused by chasing an
//! unreachable target.
//!
//! ## The seat symmetry, confirmed
//!
//! Splitting every game by which literal seat the specialist occupied (600
//! games per seat per specialist, pooled across all four opponents) is the
//! direct check on the fix above:
//!
//! | specialist | as Player One: win rate, purity, target-race exposure | as Player Two: win rate, purity, target-race exposure |
//! |---|---|---|
//! | `objective=science` | 45.0%, 100.0%, 83.5% | 50.8%, 100.0%, 86.0% |
//! | `objective=military` | 45.7%, 99.3%, 80.3% | 48.2%, 99.7%, 82.0% |
//! | `objective=civilian` | 62.0%, 100.0%, (4.5% science / 33.5% military) | 58.3%, 100.0%, (7.0% science / 35.0% military) |
//!
//! Compare this to the first round's science specialist: 45.0% win rate /
//! 100.0% purity / 83.5% exposure as Player One, against **1.0%** win rate /
//! 3.3% exposure as Player Two — the exact collapse the correctness section
//! above describes. Every specialist is now within a few points of itself
//! across seats on every measure, which is what a genuinely seat-symmetric
//! specialist should look like, and is the property the fix was built to
//! restore.
//!
//! ## Conversion: when a specialist wins, how does it win?
//!
//! The generalist's own baseline mix, aggregated over all four opponents (its
//! self-play wins from both seats, plus its wins against `mcts-eval`,
//! `mcts-uct` and `alphabeta` — 980 wins):
//!
//! | kind | share of the generalist's own wins |
//! |---|---|
//! | military supremacy | 12.9% |
//! | scientific supremacy | 31.5% |
//! | civilian victory | 54.7% |
//! | civilian tiebreak | 0.9% |
//!
//! Each specialist's own win-kind mix, aggregated the same way (both seats,
//! all four opponents):
//!
//! | specialist | wins | target-kind wins | purity |
//! |---|---:|---:|---|
//! | `objective=science` | 575 | 575 | **100.0%** |
//! | `objective=military` | 563 | 560 (2 science, 1 civilian tiebreak) | **99.5%** |
//! | `objective=civilian` | 722 | 722 (708 civilian, 14 tiebreak) | **100.0%** |
//!
//! All three specialists now convert essentially every win into their target
//! kind — civilian included, which the first (buggy) round measured at only
//! 88.9% because half its Player Two games were not actually pursuing
//! civilian victory at all. With the anchor fixed, there is no meaningful
//! purity gap left between the three: **the "civilian is the least dramatic"
//! prediction was itself an artefact of the seat bug**, not a property of the
//! civilian objective — once both seats genuinely pursue their target, all
//! three specialists convert almost perfectly.
//!
//! ## Race-seeking: does a specialist provoke more races, not just convert
//! the ones that arise?
//!
//! Game-level race exposure (`crate::match_runner::race_exposure_at`: either
//! player within one step of an instant win, regardless of who eventually
//! wins), specialist vs. the generalist baseline in the *same* opponent slot:
//!
//! | opponent | baseline military / science | `objective=military` military | `objective=science` science |
//! |---|---|---|---|
//! | generalist (self-play) | 49.7% / 30.7% | 60.3% | 57.7% |
//! | `mcts-eval` | 42.0% / 44.0% | **85.7%** | **90.3%** |
//! | `mcts-uct` | 48.3% / 43.0% | **85.3%** | **97.0%** |
//! | `alphabeta` | 44.0% / 29.7% | **93.3%** | **94.0%** |
//!
//! Both specialists provoke dramatically more of their target race against
//! every opponent, third parties especially — science races nearly triple
//! their baseline rate against `mcts-uct` and `alphabeta`, and military races
//! roughly double against every third party. This is now an unambiguous
//! result, not the mixed one the seat-broken first round reported: fixing the
//! anchor did not just repair the losing seat, it more than doubled the
//! measured race-seeking effect in the *winning* seat too, since half of what
//! looked like "the specialist's Player One behaviour" in the first round's
//! pooled tables was being diluted by a non-functional Player Two half.
//!
//! The civilian specialist's signature is the mirror image, and sharper still
//! with the fix: it *suppresses both other races* far below baseline against
//! every opponent —
//!
//! | opponent | baseline military / science | `objective=civilian` military / science |
//! |---|---|---|
//! | generalist (self-play) | 49.7% / 30.7% | 39.0% / 22.0% |
//! | `mcts-eval` | 42.0% / 44.0% | 32.0% / **0.7%** |
//! | `mcts-uct` | 48.3% / 43.0% | 33.3% / **0.0%** |
//! | `alphabeta` | 44.0% / 29.7% | 32.7% / **0.3%** |
//!
//! Against every third party the science race is all but eliminated (0.0-0.7%
//! against a 29.7-44.0% baseline) and the military race is meaningfully
//! reduced too — a purely civilian strategy denying the opponent a shot at
//! *either* supremacy condition, not merely failing to pursue them itself.
//!
//! ## Overall win rate, as context — not a verdict
//!
//! Pooled over all four opponents, 1,200 games each: `objective=science`
//! 47.9%, `objective=military` 46.9%, `objective=civilian` 60.2% (the
//! generalist wins 51.7-87.0% across the same four opponents, for scale). All
//! three are markedly stronger than the seat-broken first round measured
//! (23.0% / 26.4% / 43.6%) simply because a specialist that actually plays
//! toward its goal in *both* seats is a better player than one that is
//! only functional in one — `objective=military` even shows a positive point
//! estimate against `mcts-uct` (+41.8 `[+2.2, +81.3]`) and `alphabeta`
//! (+135.8 `[+93.5, +178.1]`), and `objective=civilian` against `mcts-uct`
//! (+120.0) and `alphabeta` (+250.7). None of that promotes any of these to a
//! ladder candidate — the point of this section remains purity, not
//! strength — but it is worth stating plainly: a specialist that is genuinely
//! seat-symmetric is not obviously the weak, narrow thing "specialist" might
//! suggest, in this budget range and against this ladder.
//!
//! ## What's next
//!
//! 1. **A retrained, binary-target net**, per `duels-value`'s own suggestion:
//!    "did this player win by science" vs. everything else, as a dedicated
//!    fit rather than a repurposed component of the four-way head. Whether it
//!    sharpens a specialist beyond what reading `v2.bin`'s existing head
//!    already achieves is an open question — the conversion numbers above say
//!    the *leaf's* own ceiling is already close to 100% pure once the search
//!    around it is seat-correct, which narrows what a retrain would need to
//!    improve on.
//! 2. **Genuine sparring partners, not a mixture-of-experts blend.** These
//!    three specialists were built to be measurably different opponents for
//!    the arena, and the numbers above say they are: three agents that share
//!    every weight with the champion and one exploration constant, and still
//!    play recognisably different, seat-consistent games. Whether that
//!    diversity is useful training signal for a *future* generalist is a
//!    separate question this round deliberately did not ask.
//!
//! # Reproducing
//!
//! ```text
//! cargo run --release -p duels-arena -- experiment \
//!     --candidate mcts-value --control mcts-eval \
//!     --pairs 400 --budget nodes:2000 --seed 1 --label mcts-value-vs-champion
//! cargo run --release -p duels-value --example value_bench
//! ```
//!
//! # Example
//!
//! ```
//! use duels_agent_mcts_value::MctsValueAgent;
//! use duels_agents_api::{Agent, Budget};
//! use duels_core::engine;
//!
//! let mut agent = MctsValueAgent::new(7);
//! let state = engine::new_game(7);
//! let legal = engine::legal_actions(&state);
//! let action = agent.choose(&state.observation(), &legal, Budget::Nodes(64));
//! assert!(legal.contains(&action));
//! ```
//!
//! [mcts_eval]: https://github.com/kncesarini/duels/tree/main/crates/agents/mcts-eval

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

mod chance;
/// The learned weights, pinned to a table of twenty fixed positions. Tests
/// only — see the module's own docs for why this crate pins what `mcts-eval`
/// deliberately tracks live.
#[cfg(test)]
mod golden;
mod leaf;
mod rollout;
mod tree;

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::{engine, Action, Observation};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub use leaf::LeafValue;
pub use rollout::{RaceWeights, RolloutWeights, RAIL};
pub use tree::{Config, Objective, PriorMode, RootStats};

/// A frozen historical `duels-value` weights generation, embedded for A/B
/// measurement against the live default via
/// [`Config::value_weights_override`] — the identical convention
/// `duels-eval`'s `Config::v1()..v9()` establishes one layer down, applied to
/// a fitted artefact instead of a hand-written one. Read from the crate next
/// door rather than duplicated: `docs/conventions.md`'s "agent crates are
/// self-contained" rule is about not depending on another *agent* crate, and
/// `duels-value` is a library below the agents, exactly like `duels-eval`
/// (whose generations every agent crate that uses it already embeds this
/// same way via `include_bytes!` inside its own `Cargo.toml`-declared
/// dependency).
///
/// `v1` names the prior default — the generation every earlier number in
/// this crate's docs and `duels_arena::leaderboard::CHAMPION`'s original
/// promotion was measured against, superseded when `v2` (the corpus-
/// generalization retrain, see `duels_value`'s crate docs) was promoted in
/// its place. This constant keeps `v1` reachable and reproducibly comparable
/// (`mcts-value:weights=v1`) rather than only living in git history.
///
/// Exists purely so "the live default vs. the generation it replaced" is one
/// `duels-arena match --agent-a mcts-value --agent-b mcts-value:weights=v1`
/// away in one process, rather than requiring two separately-built binaries.
pub const WEIGHTS_V1: &[u8] = include_bytes!("../../../duels-value/weights/v1.bin");

/// `v2`, the generation `v3` replaced (`duels_value`'s crate docs, "Follow-up
/// round three"). Kept reachable and frozen the same way [`WEIGHTS_V1`] is
/// -- and additionally, per `docs/roadmap.md`'s Tier 1-G, as a member of the
/// **frozen reference panel** every future generation's promotion battery
/// measures against (at both `nodes:32000` and `nodes:2000`; see
/// `crates/duels-arena/examples/reference_panel.rs`), since it is the direct
/// ancestor at equal budget and a stable, non-moving yardstick across
/// however many further generations follow `v3`.
pub const WEIGHTS_V2: &[u8] = include_bytes!("../../../duels-value/weights/v2.bin");

/// `v3`, the generation `v4` (Generation 3 of the loop, `gen3-l05`) replaced
/// -- see `duels_value`'s crate docs, "Generation 3", and
/// `crates/duels-value/weights/generations.json`'s `tier1-arm-c-prime`
/// entry. Kept reachable and frozen the same way [`WEIGHTS_V1`]/
/// [`WEIGHTS_V2`] are, and additionally joins the frozen reference panel at
/// `nodes:2000` (the new "cumulative gain since the immediate predecessor"
/// cell, the role [`WEIGHTS_V2`]'s equal-budget cell already plays for the
/// generation before it) -- see
/// `crates/duels-arena/examples/reference_panel.rs`.
pub const WEIGHTS_V3: &[u8] = include_bytes!("../../../duels-value/weights/v3.bin");

/// Unpromoted candidate `duels-value` weights from the roadmap's Tier 1-D/E
/// experiment (`docs/roadmap.md`, "Tier 1 design"): three generations trained
/// from matched-conditions ~100k-game corpora to isolate the exploration/
/// specialist-mixing generator change (D) from the training-target blend (E).
///
/// **None of these is the default and none is promoted** — `duels-value`'s
/// `DEFAULT_WEIGHTS` is untouched, still `v2.bin`. These exist purely as the
/// same kind of A/B-testing device [`WEIGHTS_V1`] is: measuring an
/// unpromoted candidate against the live default in one binary, one process
/// (`mcts-value:weights=arm-a` etc.), rather than building three separate
/// binaries. The promotion-battery write-up (the PR that adds this) names
/// which arm, if any, is worth a human promoting; this crate takes no
/// position on that by merely making the candidates reachable.
///
/// * `arm-a` — control: corpus generated with plain argmax, no exploration,
///   no specialists (today's pre-Tier-1 generator behaviour), trained at
///   `--value-target-lambda 1.0` (today's exact training target).
/// * `arm-b` — corpus generated *with* exploration + specialist mixing
///   (format v2), trained at `--value-target-lambda 1.0`.
/// * `arm-c` — the *same* corpus as `arm-b`, retrained at
///   `--value-target-lambda 0.5`, to isolate E's effect from D's.
pub const WEIGHTS_ARM_A: &[u8] = include_bytes!("../../../duels-value/weights/arm-a-candidate.bin");
/// See [`WEIGHTS_ARM_A`]'s docs for what all three arms are.
pub const WEIGHTS_ARM_B: &[u8] = include_bytes!("../../../duels-value/weights/arm-b-candidate.bin");
/// See [`WEIGHTS_ARM_A`]'s docs for what all three arms are.
///
/// **Trained with a known-buggy gradient, see `tools/train_value.py`'s git
/// history.** `--value-target-lambda 0.5`'s two-term loss had a wrong
/// gradient for the `LOSS` logit (amplified without bound as the model got
/// confident, instead of the correct `s - t`); it only fires when
/// `lambda < 1.0`, so this is the one arm it corrupted (`arm-a`/`arm-b` use
/// `lambda = 1.0` and are unaffected). Kept reachable for the historical
/// record, not as a trustworthy reading on the value-target-blend idea --
/// see [`WEIGHTS_ARM_C_PRIME`] for the corrected retest from the same
/// corpus.
pub const WEIGHTS_ARM_C: &[u8] = include_bytes!("../../../duels-value/weights/arm-c-candidate.bin");

/// The corrected retest of `arm-c`'s idea, after `tools/train_value.py`'s
/// blended-loss gradient bug (see [`WEIGHTS_ARM_C`]'s docs) was found and
/// fixed: same `arm-b`/`arm-c` corpus, same `--value-target-lambda 0.5`,
/// trained with the fixed gradient. Not promoted merely by existing here;
/// see the promotion write-up (`crates/duels-value/weights/generations.json`,
/// id `tier1-arm-c-prime`) for the actual gating result.
pub const WEIGHTS_ARM_C_PRIME: &[u8] =
    include_bytes!("../../../duels-value/weights/arm-c-prime-candidate.bin");
/// The same corrected-gradient retest as [`WEIGHTS_ARM_C_PRIME`], at
/// `--value-target-lambda 0.75` instead of `0.5` -- a hedge in case the
/// blend's optimum isn't at the originally-proposed 0.5. See
/// `crates/duels-value/weights/generations.json`, id `tier1-arm-d-prime`.
pub const WEIGHTS_ARM_D_PRIME: &[u8] =
    include_bytes!("../../../duels-value/weights/arm-d-prime-candidate.bin");

/// Generation 3: docs/roadmap.md's "run it again from `v3`" follow-up to Tier
/// 1 -- a fresh ~100k-game corpus generated with the live champion (`v3`,
/// bare `mcts-value`) as the generator, same exploration+specialist-mixing
/// config that produced the winning `tier1-arm-bc-explore` corpus
/// (`--sample-plies 14 --tau 1.0 --specialist-frac 0.25`), on a disjoint seed
/// range (`2,400,001..=2,500,000`; see
/// `crates/duels-value/weights/generations.json`, id `tier1-gen3`).
/// Replay-verified cleanly and sealed to `/Volumes/storage/duels/` before
/// training, per this project's standing corpus-loss-prevention discipline.
///
/// Two siblings trained from the *same* corpus, window=1 (this generation's
/// corpus alone, not combined with `tier1-arm-bc-explore`): `gen3-l05` at
/// `--value-target-lambda 0.5` (this project's post-Tier-1 baseline, per
/// `v3`'s own promotion) and `gen3-l10` at `--value-target-lambda 1.0` (a
/// control sibling, to see whether `0.5`'s advantage holds on a fresh corpus
/// or was somewhat corpus-specific -- it does: `gen3-l10` did not clear the
/// gating bar against `v3`, `gen3-l05` did, and `gen3-l05` beats `gen3-l10`
/// directly by +35.4 Elo from the identical corpus).
///
/// **`gen3-l05` was held, not promoted.** It beats `v3` head-to-head, real
/// and reproduced (+17.2 Elo pooled, `AcceptH1`) -- but reads *weaker* than
/// `v3`'s own numbers against two of the three non-ancestor frozen panel
/// members, a statistically real regression an independent gate-design
/// review flagged as disqualifying on its own, regardless of the clean
/// head-to-head win. `v3.bin` stays `duels-value`'s `DEFAULT_WEIGHTS`.
/// `gen3-l05` and its `gen3-l10` (`--value-target-lambda 1.0`) control
/// sibling both stay reachable (`mcts-value:weights=gen3-l05`/`weights=gen3
/// -l10`) as a fully measured, archived, held result -- not a promotion,
/// but not a discarded one either. See `duels_value`'s crate docs,
/// "Generation 3", for the full write-up.
pub const WEIGHTS_GEN3_L05: &[u8] =
    include_bytes!("../../../duels-value/weights/gen3-l05-candidate.bin");
/// See [`WEIGHTS_GEN3_L05`]'s docs. The `--value-target-lambda 1.0` control
/// sibling, same corpus.
pub const WEIGHTS_GEN3_L10: &[u8] =
    include_bytes!("../../../duels-value/weights/gen3-l10-candidate.bin");

/// Monte Carlo Tree Search with explicit chance nodes, scoring each leaf with
/// [`duels_value`]'s learned outcome model and no playout at all.
///
/// Read the crate docs' "Read this before you read the Elo numbers" section
/// before treating this as the stronger agent: it is measured a long way ahead
/// of `mcts-eval` and essentially level with it through any third party.
#[derive(Debug)]
pub struct MctsValueAgent {
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

impl MctsValueAgent {
    /// A new agent with the default configuration, seeded from `seed`:
    /// [`LeafValue::Learned`] at `exploration = 0.15`.
    ///
    /// The [`duels_value::Net`] the search scores against is **not** parsed
    /// here — it is built in `tree::Tree::new`, per search, so that a
    /// long-lived `duels-server` room holds no hidden cached state. The
    /// weights it reads are pinned by the `golden` module, which is the
    /// opposite call from `mcts-eval`'s on `duels-eval`; the crate docs' "The
    /// weights are pinned" section is the argument for it.
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

impl Agent for MctsValueAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "mcts-value".to_string(),
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
        let mut mcts = MctsValueAgent::new(seed ^ 0x0BAD_1DEA_0BAD_1DEA);
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
    /// configuration is exactly the one the crate docs' Elo tables were taken
    /// on: `leaf=learned` at the swept `c = 0.15`, and every other knob left at
    /// `mcts-uct`'s tuned value, inherited through `mcts-eval`.
    ///
    /// If this test ever has to be *changed*, the crate documentation's
    /// measurement tables no longer describe the shipped agent, and the fix is
    /// to re-measure rather than to update the constants.
    #[test]
    fn the_default_configuration_is_the_one_that_was_measured() {
        let cfg = Config::default();
        assert_eq!(cfg.leaf, LeafValue::Learned);
        assert_eq!(cfg.exploration.to_bits(), 0.15f64.to_bits());
        // `c` here is *swept*, not derived. Spelled out as a negative, because
        // the one thing a reader is likely to assume is the relation that does
        // apply next door: `LeafValue::Blend`'s `c = c0 * (1 - weight)` against
        // `mcts-uct`'s `c0 = 1.0` would prescribe 1.0 for a leaf with no
        // playout weight at all, and 1.0 is nowhere near what measured best.
        assert_ne!(cfg.exploration.to_bits(), 1.0f64.to_bits());
        // Everything else is untouched.
        assert_eq!(cfg.race, RaceWeights::NEUTRAL);
        assert_eq!(cfg.prior, PriorMode::None);
        assert_eq!(cfg.rollout, RolloutWeights::BIASED);
        assert_eq!(cfg.root_determinizations, 1);
        assert_eq!(cfg.chance_widen_c.to_bits(), 1.0f64.to_bits());
        assert_eq!(cfg.chance_widen_alpha.to_bits(), 0.5f64.to_bits());
        assert_eq!(cfg.value_summation, duels_value::Summation::default());
        // Nothing on the default path reads the hand-crafted evaluation.
        assert_eq!(cfg.eval_override, None);
        assert!(!cfg.leaf.needs_eval_root());
        assert!(cfg.leaf.needs_learned_net());
    }

    /// **The ablation control is `mcts-eval` as shipped**, not a frozen copy of
    /// it: [`Config::eval_base`]'s static leaf value must equal what a
    /// `duels_eval::Root` built from `duels_eval::Config::default()` produces —
    /// bit for bit, at whatever that default currently is.
    ///
    /// Inherited from `mcts-eval`, where the same test pins that agent's
    /// deliberate live-tracking design, and it matters just as much here for a
    /// different reason: a control that froze the evaluation would stop being
    /// `mcts-eval` the moment a `duels-eval` round landed, and every Elo
    /// number in the crate docs is measured against that agent.
    ///
    /// Note what this does *not* say about the default path, which reads no
    /// evaluation at all — `tree::tests::the_evaluation_cannot_reach_the_default_leaf`
    /// is the complementary statement.
    #[test]
    fn the_eval_base_control_tracks_duels_evals_live_default() {
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
        let agent = MctsValueAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "mcts-value");
        assert_eq!(spec.version, "1.0.0");
        assert!(spec.params.contains("c=0.150"), "{}", spec.params);
        assert!(spec.params.contains("leaf=learned"), "{}", spec.params);
        assert!(spec.params.contains("chance="), "{}", spec.params);
        assert!(spec.params.contains("rollout="), "{}", spec.params);
    }

    /// **The spec string has to record which weights produced a results
    /// file.** The default path scores every leaf with a fitted artefact, so
    /// the params string carries [`duels_value::default_weights_id`] — the
    /// network's shape plus a content hash of the embedded bytes — and the
    /// summation order the forward pass used.
    ///
    /// This is the provenance half of the pinning design; the `golden`
    /// module's table is the other half. Between them a retrain cannot land
    /// silently *and* cannot leave an old results file uninterpretable.
    #[test]
    fn the_spec_records_which_learned_weights_it_used() {
        let params = MctsValueAgent::new(1).spec().params;
        assert!(
            params.ends_with(&format!(
                "value={}/{}",
                duels_value::default_weights_id(),
                duels_value::Summation::default().name()
            )),
            "the spec must name the weights and the summation order: {params}"
        );
        // The hash is a real hash of real bytes, not a placeholder.
        assert!(
            duels_value::default_weights_id().contains('/'),
            "{}",
            duels_value::default_weights_id()
        );
        // The default path reads no hand-crafted evaluation, and says so
        // rather than recording a configuration it never consults.
        assert!(params.contains("eval=unused"), "{params}");
        assert!(!params.contains("evalgen="), "{params}");

        // The `mcts-eval` control is the mirror image: it records the whole
        // live `duels-eval` configuration and no weights identity, because it
        // parses no network. That is what makes two results files from either
        // side of this ablation tell you which arm they came from.
        let base = MctsValueAgent::with_config(1, Config::eval_base())
            .spec()
            .params;
        let live = duels_eval::Config::default().params_string();
        assert!(base.ends_with(&format!("eval={live}")), "{base}");
        assert!(base.contains("leaf=blend(0.500)"), "{base}");
        assert!(base.contains("c=0.500"), "{base}");
        assert!(!base.contains("value="), "{base}");

        // ...and the pure-playout ablation at the bottom of the chain records
        // neither.
        let rollout = MctsValueAgent::with_config(1, Config::rollout_base())
            .spec()
            .params;
        assert!(rollout.contains("eval=unused"), "{rollout}");
        assert!(rollout.contains("leaf=rollout"), "{rollout}");
        assert!(rollout.contains("c=1.000"), "{rollout}");
        assert!(!rollout.contains("value="), "{rollout}");
    }

    #[test]
    fn a_single_legal_action_is_returned_without_searching() {
        let mut agent = MctsValueAgent::new(3);
        let state = engine::new_game(3);
        let only = [engine::legal_actions(&state)[0]];
        let chosen = agent.choose(&state.observation(), &only, Budget::Nodes(10_000));
        assert_eq!(chosen, only[0]);
        assert_eq!(agent.total_simulations(), 0, "no search was needed");
    }

    #[test]
    fn every_returned_action_is_one_of_the_offered_ones() {
        let mut agent = MctsValueAgent::new(11);
        let state = engine::new_game(11);
        let legal = engine::legal_actions(&state);
        for _ in 0..5 {
            let a = agent.choose(&state.observation(), &legal, Budget::Nodes(20));
            assert!(legal.contains(&a));
        }
    }

    #[test]
    fn a_node_budget_runs_exactly_that_many_simulations() {
        let mut agent = MctsValueAgent::new(5);
        let state = engine::new_game(5);
        let legal = engine::legal_actions(&state);
        agent.choose(&state.observation(), &legal, Budget::Nodes(37));
        assert_eq!(agent.total_simulations(), 37);
        agent.choose(&state.observation(), &legal, Budget::Nodes(3));
        assert_eq!(agent.total_simulations(), 40);
    }

    #[test]
    fn a_time_budget_returns_promptly_and_does_some_work() {
        let mut agent = MctsValueAgent::new(9);
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
            let mut agent = MctsValueAgent::new(seed);
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
            let mut agent = MctsValueAgent::with_config(seed, cfg);
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

    /// `choose` exactly as `mcts-eval`'s reads: one determinization, one tree,
    /// the whole node budget, that agent's ensemble move-selection rule — and,
    /// since it drives `tree::Tree::eval_legacy_simulate` rather than
    /// `simulate`, that agent's search and leaf dispatch too.
    ///
    /// It is a copy on purpose: a test that called the live code would prove
    /// nothing. The `Slices` budget arithmetic is not reproduced because this
    /// is only ever called at `root_determinizations == 1`, where that arm
    /// reduces to `nodes.max(1)` simulations on one tree — which is what the
    /// loop below does.
    fn mcts_eval_choose(
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
            tree.eval_legacy_simulate(rng);
        }
        let chosen = tree::eval_legacy_best_of(std::slice::from_ref(&tree)).unwrap_or(legal[0]);
        if legal.contains(&chosen) {
            chosen
        } else {
            legal[0]
        }
    }

    /// **The copied agent is `mcts-eval`**, at the whole-agent level and across
    /// the whole leaf family: this crate's `choose` plays move for move with
    /// the verbatim frozen copy above, over whole seeded games, at its own
    /// default and at both ablation controls.
    ///
    /// Checked over games rather than only at the opening position, so that the
    /// RNG streams have to stay in step across dozens of `choose` calls, chance
    /// nodes, pending choices and all.
    /// `tree::tests::the_copied_search_is_the_mcts_eval_search_node_for_node`
    /// is the stronger, arena-for-arena form of the same claim.
    ///
    /// Running it at [`Config::default`] as well as at [`Config::eval_base`] is
    /// the point: what is being pinned is not "the control arm is `mcts-eval`"
    /// alone but "*everything below the leaf value* is `mcts-eval`", which is
    /// what makes the crate docs' Elo numbers an ablation on one variable.
    #[test]
    fn the_eval_base_is_the_mcts_eval_agent_move_for_move() {
        for (name, cfg) in [
            ("eval_base", Config::eval_base()),
            ("default", Config::default()),
            ("rollout_base", Config::rollout_base()),
        ] {
            for seed in 0..4u64 {
                let mut agent = MctsValueAgent::with_config(seed, cfg);
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
                    let want = mcts_eval_choose(&mut legacy_rng, cfg, &obs, &legal, budget);
                    assert_eq!(
                        got, want,
                        "{name}, seed {seed}, decision {decisions}: not the mcts-eval agent"
                    );
                    engine::apply(&mut state, got, &mut rng).expect("a legal action");
                    decisions += 1;
                    assert!(decisions < 5_000);
                }
                assert!(decisions > 20, "the game was too short to prove much");
            }
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
            LeafValue::Learned,
            LeafValue::LearnedBlend { weight: 0.5 },
            LeafValue::LearnedSymmetric,
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
                let mut mcts = MctsValueAgent::with_config(
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
            MctsValueAgent::with_config(
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
        assert!(describe(LeafValue::Learned).contains("leaf=learned"));
        assert!(
            describe(LeafValue::LearnedBlend { weight: 0.5 }).contains("leaf=learned_blend(0.500)")
        );
        assert!(describe(LeafValue::LearnedSymmetric).contains("leaf=learned_symmetric"));

        // A learned leaf also has to record *which* weights it used: the
        // shape alone would make two results files from either side of a
        // retrain indistinguishable, which is the same failure the `eval=`
        // tail exists to prevent for the hand-crafted evaluation.
        let learned = describe(LeafValue::Learned);
        assert!(
            learned.contains(&format!("value={}", duels_value::default_weights_id())),
            "the spec does not name the value network: {learned}"
        );
        // The same argument applies to *how* the forward pass accumulates:
        // `Summation::Unrolled4` reassociates the hidden layer's sum, so two
        // results files taken either side of that change are not comparable
        // to the last few `f32` digits, and the spec has to say which one ran.
        assert!(
            learned.contains(&format!(
                "value={}/{}",
                duels_value::default_weights_id(),
                duels_value::Summation::default().name()
            )),
            "the spec does not name the summation order: {learned}"
        );
        // ...and the *default* configuration's spec string carries all of it,
        // because here a learned leaf is not an option but the product.
        let default = describe(Config::default().leaf);
        assert!(default.contains("leaf=learned;"), "{default}");
        assert!(default.contains("value="), "{default}");
        // The hand-crafted variants, which are the controls, carry none of it.
        for control in [LeafValue::Blend { weight: 0.5 }, LeafValue::Rollout] {
            assert!(!describe(control).contains("value="), "{control:?}");
        }
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
                let mut mcts = MctsValueAgent::with_config(
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
            MctsValueAgent::with_config(
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
                let mut mcts = MctsValueAgent::with_config(
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
            MctsValueAgent::with_config(
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
            let mut agent = MctsValueAgent::with_config(
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
        let mut agent = MctsValueAgent::with_config(
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

    /// [`MctsValueAgent::last_root`] must be **only** a readout: adding it may
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
            let mut quiet = MctsValueAgent::new(seed);
            let mut watched = MctsValueAgent::new(seed);
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
            let mut agent = MctsValueAgent::new(seed);
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
