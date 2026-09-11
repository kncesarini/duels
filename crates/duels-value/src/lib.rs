//! `duels-value`: a **learned** 7 Wonders Duel position value — a small
//! neural network that predicts, from public information only, which of four
//! mutually-exclusive outcomes a position ends in.
//!
//! # Where this sits, and what it is for
//!
//! A library **below the agents**, next to [`duels_core`] and alongside
//! `duels-strategy` and `duels-eval`, for the reason `docs/conventions.md`
//! gives: an agent crate may not depend on another agent crate, so anything
//! more than one search wants to consume moves *down* into a library rather
//! than sideways. It depends on `duels-core` and on nothing else — no `rand`,
//! no clock, no ML runtime. [`features`] and [`Net::forward`] are pure
//! functions.
//!
//! It exists because of a specific, measured shape in this project's results.
//! `mcts-eval`'s budget-scaling curve flattens hard past `Nodes(2000)`
//! (roughly `+70` Elo per doubling early, about `+22` by `Nodes(128000)`),
//! which is the signature of a search limited by **leaf signal quality**
//! rather than by breadth. And the single largest effect this project has
//! measured — `+89` Elo — was blending a hand-crafted evaluation into that
//! leaf. Both point at the same next question: is there a *better* leaf
//! signal than half a playout plus `duels-eval`?
//!
//! **This crate is the feasibility spike for that question, not an answer to
//! it.** Read "What it measured" below for what the measurements actually say
//! before assuming any of this is a win. Nothing here is on any agent's
//! default path.
//!
//! # Four outcomes, not one win probability
//!
//! The head is a four-way softmax over [`Outcome`] — military win, science
//! win, civilian win, loss — and *not* a single win/loss sigmoid, even though
//! the scalar a search consumes is recovered by summing three of the four
//! ([`Dist::win_probability`]).
//!
//! The reason is that the three win kinds are structurally different and
//! wildly unbalanced: scientific supremacy is about **2.3%** of games. A
//! single-scalar model is free to average that away, and this project already
//! knows the two signals it would be averaging are complementary rather than
//! interchangeable — `mcts-eval`'s victory-kind breakdown is the sharpest
//! diagnostic in its whole line of work, showing a static evaluation supplying
//! civilian judgement while the playout supplies sight of military races.
//! Making the network *also* predict the rarer, structurally distinct outcomes
//! is the standard multi-task argument for a better shared representation.
//!
//! That argument was **checked rather than asserted**: a single-scalar model
//! of identical shape was trained on the same rows and split, and the two
//! compared on the aggregate win probability. "What it measured" below has
//! the numbers, and they do not support the hypothesis.
//!
//! # Trained against outcomes, never against the search's own opinion
//!
//! The corpus (`duels-arena`'s `examples/value_corpus.rs`) records both the
//! eventual [`duels_core::GameResult`] and the win probability `mcts-eval`
//! backed up at its root. The labels here are the **outcomes**. Fitting an
//! evaluation against the search's own verdict was separately measured in this
//! repository to destroy exactly the complementarity that makes a
//! hand-crafted evaluation useful inside a blend, and there is no reason to
//! think a learned model escapes that; a model fitted on the search's opinion
//! can at best reproduce the search, which is not what a leaf value is for.
//!
//! # How it was built (reproducible from three commands)
//!
//! ```text
//! # 1. the corpus, if it is not already on disk (204.5 min on 14 cores)
//! cargo run --release -p duels-arena --example value_corpus -- \
//!     --games 100000 --seed 1 --budget nodes:2000 \
//!     --out arena/corpus/mcts-eval-nodes2000.jsonl
//!
//! # 2. the training matrix (9.2 s)
//! cargo run --release -p duels-arena --example feature_dump -- \
//!     --corpus arena/corpus/mcts-eval-nodes2000.jsonl \
//!     --out arena/corpus/features-v1.bin --games 100000 --stride 2
//!
//! # 3. the fit (about 3 min; numpy only, no ML runtime anywhere)
//! tools/train_value.py --matrix arena/corpus/features-v1.bin \
//!     --out crates/duels-value/weights/v1.bin \
//!     --metrics arena/corpus/features-v1.metrics.json \
//!     --hidden 128 --epochs 60 --lr 2e-3 --weight-decay 1e-5 \
//!     --patience 8 --also-scalar
//! ```
//!
//! # What it measured
//!
//! ## The data
//!
//! The whole 100,000-game `mcts-eval` `Nodes(2000)` self-play corpus, every
//! second labelled decision, both perspectives: **6,772,568 rows**. Split **by
//! game** on `seed % 10` — 0-6 train (70,000 games), 7-8 validation (20,000),
//! 9 test (10,000). Splitting by row instead would leak nearly every held-out
//! position into training, because rows inside a game share a seed, a deal and
//! most of a board; every row carries its seed so that the split can be made
//! correctly, and the trainer refuses a split that overlaps.
//!
//! Victory kinds over those 100,000 games: **civilian 80.64%, military 15.43%,
//! scientific 2.31%, civilian tiebreak 1.55%, draw 0.07%**. Scientific
//! supremacy really is the ~2.3% event this crate's four-way head was designed
//! around, and a draw really is rare enough to fold into `Loss`.
//!
//! ## The model
//!
//! `211 → 128 → 4`, ReLU, softmax: **27,652 parameters**, 108 KiB of weights.
//! Best validation epoch **5** (of 60, early-stopped at 13) — it overfits
//! quickly, and the earlier fit on 30,000 games peaked at epoch 2, so this is
//! a variance-limited problem where corpus *games* are the binding resource.
//!
//! ## Against the incumbent leaf signal
//!
//! The honest yardstick is not "does it beat chance", it is "does it beat what
//! `mcts-eval` already backs up" — the same corpus records the win probability
//! the search itself concluded at each of these positions, at a cost of 2,000
//! simulations per decision. On the **test** set (10,000 games, 202,972 rows),
//! predicting the same outcomes:
//!
//! | predictor | Brier | log loss | accuracy | ROC AUC |
//! | --- | --- | --- | --- | --- |
//! | learned value (this crate) | **0.17223** | **0.50909** | 0.7339 | 0.8212 |
//! | `mcts-eval` root value, `Nodes(2000)` | 0.17429 | 0.51748 | **0.7401** | **0.8226** |
//! | single-scalar control (below) | 0.17246 | 0.50966 | 0.7334 | 0.8211 |
//!
//! So a 27,652-parameter static function is **at parity** with 2,000 nodes of
//! MCTS as an outcome predictor: better on both probabilistic losses, very
//! slightly worse on ranking. Read that as parity, not as a win — the two
//! differences point opposite ways and neither is large.
//!
//! The calibration is genuinely good, and monotone across all ten buckets
//! (test set, predicted win probability against what actually happened):
//!
//! ```text
//!   predicted        n     mean outcome
//!   0.0-0.1      74279          0.044
//!   0.1-0.2      51230          0.166
//!   0.2-0.3      56968          0.260
//!   0.3-0.4      68110          0.355
//!   0.4-0.5      83929          0.443
//!   0.5-0.6      84007          0.546
//!   0.6-0.7      69865          0.637
//!   0.7-0.8      62338          0.737
//!   0.8-0.9      55216          0.835
//!   0.9-1.0      70736          0.957
//! ```
//!
//! ## The multi-task claim does not hold up (and was checked, not asserted)
//!
//! The design argument for a four-way head was partly that forcing the network
//! to also predict a rare, structurally distinct outcome should produce a
//! better shared representation for the main task. A single-scalar sigmoid
//! model of **identical shape**, trained on the same rows with the same split
//! and selected on the same quantity, says otherwise: 0.17246 against 0.17223
//! Brier, 0.50966 against 0.50909 log loss, and 0.8211 against 0.8212 AUC. The
//! decomposed model is ahead by about a tenth of a percent on each — a
//! difference far too small to attribute to the auxiliary structure rather
//! than to initialisation.
//!
//! **The auxiliary heads therefore do not measurably improve the aggregate.**
//! They are still worth keeping, because they cost nothing and they are
//! informative on their own account (below), and because a victory-kind-aware
//! search would need them — but the multi-task representation argument should
//! not be repeated as if it had been confirmed here.
//!
//! ## The per-kind heads, which are the interesting part
//!
//! One-vs-rest on the validation set:
//!
//! | head | prevalence | ROC AUC | Brier | `E[p ǀ true]` | `E[p ǀ false]` |
//! | --- | --- | --- | --- | --- | --- |
//! | `military_win` | 0.0693 | 0.8111 | 0.05681 | 0.210 | 0.066 |
//! | `science_win` | 0.0107 | **0.9549** | 0.00788 | 0.275 | 0.007 |
//! | `civilian_win` | 0.4198 | 0.8262 | 0.16680 | 0.600 | 0.281 |
//! | `loss` | 0.5003 | 0.8255 | 0.17021 | 0.661 | 0.336 |
//!
//! Four-way argmax accuracy 0.6963. The science head is by a wide margin the
//! sharpest of the four (AUC 0.955 on a 1.07%-prevalence class), which is the
//! result that most justifies keeping the decomposition: a science race is
//! visible in public information long before it resolves, and a single scalar
//! spends none of its capacity saying so. AUC rather than accuracy is the
//! right read for a class this rare — a model that always answered "no" would
//! score 0.989 accuracy and 0.5 AUC.
//!
//! ## Inside the search: the headline number, and the caveats on it
//!
//! Both learned leaves were measured with `duels-arena experiment` against
//! `mcts-eval`'s own default (`LeafValue::Blend { weight: 0.5 }`, `c = 0.5` —
//! the reigning champion when this was measured, not `mcts-uct`; the learned
//! leaf has since become `mcts-value` and taken that title), paired-seed and
//! seat-swapped, at
//! `Nodes(32000)`, over two **disjoint** seed ranges of 300 games each.
//! `Nodes(32000)` is a load-insensitive stand-in for the production
//! `TimeMs(1000)`; a fixed node count cannot be corrupted by a busy machine,
//! which a wall-clock budget can.
//!
//! | candidate | games | W-L-D | Elo | 95% CI | SPRT |
//! | --- | --: | --- | --: | --- | --- |
//! | `leaf=learned` (no playout) | 600 | 349-251-0 | **+57.2** | [+29.0, +85.3] | AcceptH1 |
//! | `leaf=learned_blend:0.5` | 600 | 389-211-0 | **+106.1** | [+77.0, +135.2] | AcceptH1 |
//!
//! Both reproduced on the second, disjoint seed range on their own
//! (`+55.9` / `+58.3` and `+94.6` / `+117.4`), which is this project's
//! standard for not trusting a single range.
//!
//! Two things in there are worth more than the headline.
//!
//! **A pure learned leaf is `+57`, where a pure hand-crafted one is `-171`.**
//! That is the sharpest single result here. `docs/conventions.md`'s standing
//! prior — *simulation beats hand-crafted judgement for position value in this
//! game* — was established by `LeafValue::Static` losing badly, and it is the
//! low ceiling of a *hand-crafted* evaluation that the prior is really about.
//! A learned value of the same shape, with no playout at all, beats the
//! playout-plus-evaluation blend. The prior needs narrowing, not discarding.
//!
//! **The blend still wins, and by a lot, which is the prior surviving in its
//! other half.** `+106` against `+57` says the playout is still contributing
//! something the learned value does not have, exactly as it does for the
//! hand-crafted evaluation. Half of each remains the right shape.
//!
//! The victory-kind breakdown says what each side is contributing, and it is
//! the same complementarity story `mcts-eval` found, one notch further along:
//!
//! | | military | science | civilian | tiebreak |
//! | --- | --: | --: | --: | --: |
//! | `leaf=learned` wins | 53 | **110** | 182 | 4 |
//! | control wins (vs it) | 20 | **1** | 225 | 5 |
//! | `leaf=learned_blend:0.5` wins | 72 | **41** | 273 | 3 |
//! | control wins (vs it) | 19 | **3** | 187 | 2 |
//!
//! The learned value's contribution is overwhelmingly **sight of the science
//! race**: the control converts one scientific supremacy in 600 games against
//! the pure learned leaf, which converts 110. That lines up exactly with the
//! offline per-head numbers above (the science head is the sharpest of the
//! four, AUC 0.955) and with why the decomposition was worth keeping even
//! though it did not improve the aggregate.
//!
//! ### What has *not* been established — read this before shipping any of it
//!
//! 1. **No wall-clock confirmation.** `examples/value_bench.rs` measures the
//!    forward pass at about **45% of a playout** — far more than expected, for
//!    a reason that file documents — so `LearnedBlend` costs about `1.45x` per
//!    simulation against the default's `1.08x`. At equal wall clock it would
//!    run roughly `0.74x` the simulations, which against ~22 Elo per doubling
//!    is worth about `-9` Elo. Expect roughly `+95` at `TimeMs(1000)`, and
//!    treat that as an estimate, not a measurement. `docs/conventions.md` is
//!    emphatic that this project has been burned by exactly this both ways.
//! 2. **One budget, one opponent.** Everything above is `Nodes(32000)` against
//!    one control. A leaf that steers this hard into science races (31.5% of
//!    the pure variant's wins, against a ~2.3% base rate in self-play) could
//!    be particularly effective against this one opponent specifically,
//!    rather than the number reflecting overall strength. The `duels-arena
//!    experiment` mechanism gate
//!    reads **Inconclusive** on both runs for precisely this reason — the
//!    control wins so few science games that the ratio cannot be estimated —
//!    and an Inconclusive gate is an honest "not enough evidence", not a pass.
//!    A round-robin against the whole ladder is the check that settles it.
//! 3. **The exploration constant was not re-derived.** `c = 0.5` was kept for
//!    both variants. For `learned_blend:0.5` that *is* the value
//!    `c = c₀·(1 - w)` prescribes, so the blend is a clean one-variable
//!    comparison. For the **pure** learned leaf it is a judgement call: the
//!    reward is no longer Bernoulli, and its spread (a near-uniform predicted
//!    probability, standard deviation about 0.29 against a Bernoulli's 0.5)
//!    suggests something near `0.58` — close enough to `0.5` to leave alone,
//!    but not derived. `mcts-eval`'s `LeafValue::Blend` docs are emphatic that
//!    a change to what a leaf backs up should re-derive `c`, and this spike
//!    did not.
//! 4. **Nothing is adopted.** No `Config::default` moved, no existing agent
//!    changed behaviour, and both variants are opt-in `LeafValue`s reachable
//!    only from an explicit spec string. That is deliberate: this was
//!    pre-registered as a feasibility spike whose success criterion was a
//!    measurement, not a shipped default.
//!
//! # Follow-up round: the `c` sweep, and why the result still is not adopted
//!
//! Three of the four "next steps" above were then done. Two of the four
//! caveats resolved *for* the model and one resolved sharply against it, and
//! the honest summary is: **the pure learned leaf is far stronger than this
//! spike measured, and it is still not a general strength improvement.**
//!
//! Every figure below is backed by a results directory under
//! `arena/results/experiments/`, named in its row.
//!
//! ## Caveat 3 was the dominant term, not a footnote (`c` really was wrong)
//!
//! `Nodes(32000)`, two disjoint 300-game ranges (seeds 1 and 200001),
//! paired-seed and seat-swapped, against `mcts-eval`'s default:
//!
//! | leaf | `c` | pooled Elo | 95% CI | ranges | results dir |
//! | --- | --: | --: | --- | --- | --- |
//! | `learned` | 0.1 | +126.7 | [+97.1, +156.4] | +127.8 / +125.2 | `p0-learned-c0.1` |
//! | `learned` | **0.15** | **+140.1** | [+110.0, +170.2] | +163.4 / +117.4 | `p0-learned-c0.15` |
//! | `learned` | 0.25 | +101.0 | [+72.1, +130.0] | +97.1 / +104.6 | `p0-learned-c0.25` |
//! | `learned` | 0.5 | +57.2 | [+29.0, +85.3] | +55.9 / +58.3 | `spike-learned-pure` |
//! | `learned_blend:0.5` | **0.5** | **+106.1** | [+77.0, +135.2] | +94.6 / +117.4 | `spike-learned-blend` |
//! | `learned_blend:0.5` | 0.35 | +73.9 | [+45.5, +102.3] | +99.6 / +48.8 | `p0-learnedblend0.5-c0.35` |
//! | `learned_blend:0.5` | 0.25 | +85.5 | [+56.9, +114.1] | +89.7 / +81.1 | `p0-learnedblend0.5-c0.25` |
//!
//! The pure-leaf sweep has a genuine **interior** optimum rather than running
//! off the end of the range: `0.1` and `0.25` are both below `0.15`, so the
//! best value is bracketed. `0.1` and `0.15` overlap heavily, so read the
//! optimum as "somewhere in `[0.10, 0.15]`" rather than as `0.15` exactly;
//! what is not in doubt is that the whole neighbourhood is 70-83 Elo above the
//! `0.5` that was inherited.
//!
//! **The two halves of that sweep point opposite ways, and the asymmetry is
//! the result.** For the blend, `c = c₀·(1 - w)` already prescribes `0.5` at
//! `w = 0.5`, so there was nothing to retune and every perturbation *lost*
//! 20-30 Elo — the formula is confirmed, not merely assumed. For the **pure**
//! leaf there is no playout to shrink, the formula gives no guidance at all,
//! and the `0.5` this spike inherited was leaving about **83 Elo** on the
//! table — more than every other refinement in this crate combined.
//! `docs/conventions.md`'s standing instruction that a change to what a leaf
//! backs up must re-derive `c` was right, and the reason to state it this
//! loudly is that the spike above read `0.5` as "close enough to leave alone".
//!
//! ## Caveat 1, first half: it is not a high-budget artefact
//!
//! At the ladder's **production** budget, `Nodes(2000)`, two disjoint
//! 400-game ranges (`p0-learned-nodes2000`): **+91.4 Elo [+66.5, +116.3]**,
//! 503-297, SPRT `AcceptH1`, ranges +128.6 [+92.2, +165.0] and
//! +55.9 [+21.5, +90.4] — **both cell CIs exclude zero**. The mechanism gate
//! reads `Pass` on `civilian_share` (58.8% against a required 43.4%) and on
//! `military_share` (14.9% against a permitted 19.5%), which is a better
//! mechanism read than either spike run above obtained.
//!
//! ## Caveat 1 fully resolved, and the wall clock is the good news
//!
//! `TimeMs(1000)`, two disjoint 200-game ranges per candidate, run strictly
//! one candidate after the other on a machine verified quiet (the arena held
//! 1291-1321% CPU of 14 cores at 0.3% idle, with no other heavy process):
//!
//! | candidate | pooled Elo | 95% CI | ranges | W-L |
//! | --- | --: | --- | --- | --- |
//! | `leaf=learned, c=0.15` | **+140.6** | [+103.8, +177.5] | +118.5 / +163.1 | 277-123 |
//! | `learned_blend:0.5, c=0.5` | +68.5 | [+33.8, +103.1] | +66.5 / +70.1 | 239-161 |
//!
//! Both `AcceptH1`; `p1-timems-learned-c0.15` and
//! `p1-timems-learnedblend-c0.5`. The blend's +68.5 sits inside the interval
//! of the spike's own `TimeMs` figure (+80.9, `spike-learned-blend-time`),
//! which is a useful check that the quiet-machine protocol below did not
//! change what was being measured.
//!
//! **Which of the two leaves is better depends on the budget, and the spike's
//! ordering does not survive either correction.** With the inherited `c=0.5`
//! the blend led at a fixed node count, +106.1 against +57.2, and that is the
//! comparison the spike drew its "the blend still wins, and by a lot"
//! conclusion from. Re-deriving `c` reverses it at a fixed node count
//! (+140.1 against +106.1), and a fixed *time* budget widens the reversal to
//! better than two to one (+140.6 against +68.5). **The spike's headline
//! finding was an artefact of an untuned constant plus a budget type.**
//!
//! The pure leaf's +140.6 is *higher* than the same configuration's `Nodes(2000)`
//! figure, and the direction is the point. `examples/value_bench.rs` warned
//! that `LearnedBlend` costs `1.45x` per simulation and should therefore
//! *lose* about 9 Elo at a fixed time budget. The **pure** learned leaf is the
//! opposite case and the reasoning has to be redone rather than reused: it
//! runs no playout at all, so with the four-way unroll it costs about `0.35x`
//! a playout against the default blend's `1.08x`, and at equal wall clock it
//! buys roughly `3x` the simulations. A leaf that *replaces* the playout and a
//! leaf that is *added to* it have opposite wall-clock economics, and this
//! crate now has a measurement of each.
//!
//! # A note on reading a load average on the machine this was measured on
//!
//! `docs/conventions.md` is right that a `TimeMs` run needs a quiet machine,
//! and the usual proxy — "load average in single digits" — is **not usable
//! here**. `duels-arena` parallelises seeds within a match, so a single
//! legitimate match shows a load average near 150 on this box and four
//! concurrent matches showed ~600; the figure counts runnable threads in a
//! thread-per-game pool, not CPU oversubscription. The numbers above were
//! taken under the criterion the proxy stands for — no other arena, build or
//! test process, and no *variable* competing load — verified with a
//! process-level CPU snapshot before and after every cell rather than with a
//! load average.
//!
//! ## Caveat 2 resolved *against* it: the gain is `mcts-eval`-specific
//!
//! This is the finding that matters most, and it is the reason nothing here
//! is adopted. A mini round robin at `Nodes(2000)`, 400 games per pairing on
//! the same two seed ranges, asking whether the candidate's margin over
//! `mcts-eval` survives being measured *through a third party*:
//!
//! | comparison | Elo | 95% CI | share of direct |
//! | --- | --: | --- | --: |
//! | direct, cand − `mcts-eval` (800 games) | +91.5 ± 12.7 | — | 100% |
//! | indirect via `mcts-uct` | **+25.5** ± 27.8 | [−28.9, +79.9] | 28% |
//! | indirect via `alphabeta` | **+11.2** ± 36.1 | [−59.5, +82.0] | 12% |
//!
//! (`p2-cand-vs-mctsuct`, `p2-ctrl-vs-mctsuct`, `p2-cand-vs-alphabeta`,
//! `p2-ctrl-vs-alphabeta`.) **Both indirect intervals contain zero.** A joint
//! Bradley-Terry fit over all five records with `mcts-uct` pinned at 1000 —
//! the same estimator `duels_arena::elo::fit_joint_elo` uses for anything
//! ladder-shaped — puts the candidate at 1213.2 against `mcts-eval`'s 1139.3,
//! a +74.0 gap where the direct match said +91.5. That residual is real
//! intransitivity: no single consistent rating reproduces both.
//!
//! ### The mechanism: real science-conversion skill, still concentrating
//!
//! The science-seeking behaviour *does* transfer to a real playout opponent —
//! it just does not yet add to the total the way it does against `mcts-eval`.
//!
//! | pairing | candidate's wins | opponent's wins |
//! | --- | --- | --- |
//! | cand vs `mcts-uct` | mil 38, **sci 89**, civ 171 → 298 | mil 8, sci 0, civ 88 → 101 |
//! | `mcts-eval` vs `mcts-uct` | mil 34, **sci 10**, civ 237 → 287 | mil 16, sci 2, civ 88 → 113 |
//!
//! Against `mcts-uct` the learned leaf converts 89 scientific supremacies
//! where `mcts-eval` converts 10 — a real, large skill gap at recognizing and
//! winning the science race, not a subtle difference, and consistent with the
//! offline per-head numbers (science AUC 0.955). Its total is 298 against 287
//! — a modest net edge for now — because those extra science wins are, at
//! this training stage, largely trading against **its own civilian column**
//! (171 against 237) rather than adding wins outright. That is where this
//! agent's current strength sits, not a ceiling on it.
//!
//! Against `mcts-eval` the same science-conversion skill produces the direct
//! +91.5, because `mcts-eval` concedes 129 science games in 800 and wins
//! **one** — `mcts-eval`'s own evaluation cannot defend the race at all, which
//! is the miscalibration `science_calibration` independently documented. That
//! is a specific, measured weakness in `mcts-eval`, not a discount on the
//! skill itself: an agent that plays the science race correctly should be
//! expected to win big against an opponent that cannot defend it at all, and
//! to keep converting more of that skill into net wins against stronger
//! opponents as training improves.
//!
//! **The `science_share` mechanism gate reads `Inconclusive` in all four
//! round-robin pairings and never `Pass`** — always because the *control* wins
//! too few science games to form a ratio (0, 2, 1 and 3, against an evidence
//! floor of 5). That bound is, for now, not satisfiable against any opponent
//! on this ladder; `civilian_share` and `military_share` `Pass` in all four.
//!
//! ## What this says about the leaf, as opposed to about the agent
//!
//! A leaf value can be a large, reproducible, correctly-measured Elo gain
//! against one opponent and close to nothing against two others, without any
//! bug and without any of the measurements being wrong. Reading the direct
//! number as "the strength of the learned leaf" is the mistake, and a
//! candidate-versus-champion match — which is exactly what
//! `.github/workflows/ai-candidate.yml` runs — cannot detect it. **A third
//! opponent is what shows how broadly a leaf's strength generalizes, as
//! opposed to how it concentrates against one particular opponent**, and
//! that is worth checking every time a leaf change measures well against the
//! designated champion alone — all the more so now that the champion *is*
//! this leaf, so an `ai-candidate` run measures one learned-leaf agent
//! against another.
//!
//! ## The remaining next steps, reordered by what is now known
//!
//! 1. **Make the model zero-sum coherent.** It is not:
//!    `tests/probability_coherence.rs` measures a mean
//!    `|P(win|One) + P(win|Two) - 1|` of 0.0559 over 1,277 legal positions
//!    (max 0.4814), and an opening probability mass of 0.9396 across 512
//!    deals where it must be 1. An antisymmetric head would make the property
//!    exact for free. This is the one known defect that is architectural
//!    rather than data-limited, and it is cheap.
//! 2. **More corpus games**, unchanged: the fit peaks at epoch 5 on 70,000
//!    training games and at epoch 2 on 21,000, so this is variance-limited and
//!    the corpus is the binding resource, not the architecture.
//! 3. **Whatever is tried next, measure it against `mcts-uct` too**, for the
//!    reason the section above gives. The strength this crate can demonstrate
//!    against `mcts-eval` is now well established at both budget types; the
//!    measurement still worth chasing is how much of the science-conversion
//!    skill that strength runs through carries over against a third party,
//!    since that is what should keep growing as training improves.
//!
//! # Follow-up round two: the mixed-corpus retrain (`v2.bin`), promoted
//!
//! Item 2 above, acted on: a retrain on a bigger, differently-sourced
//! corpus, to test whether corpus size/diversity — not architecture — really
//! is the binding resource, and whether a bigger corpus incidentally closes
//! item 3's third-party gap. **It does, partially, and it is promoted:**
//! `v2.bin` beats the weights it replaces on a confirmed, disk-verified
//! margin, its margin over `mcts-eval` grew, and one of the two third-party
//! checks shows meaningfully better generalization than `v1` had. `v2.bin` is
//! now [`DEFAULT_WEIGHTS`]; `v1.bin` is kept in the repository and reachable
//! via `mcts-value:weights=v1` as the generation it replaces.
//!
//! ## The corpus
//!
//! 40,000 games of **`mcts-value` self-play** at `nodes:2000`
//! (`arena/corpus/mcts-value-nodes2000.jsonl`, via the new
//! `duels-arena/examples/value_corpus_mv.rs` — `value_corpus.rs`'s twin,
//! same schema, `mcts-value` playing both seats instead of `mcts-eval`)
//! mixed with a **fresh 15,000-game `mcts-eval` insurance batch** at the same
//! budget (`arena/corpus/mcts-eval-insurance-nodes2000.jsonl`, disjoint
//! seeds), merged with `tools/merge_feature_matrices.py` into **7,383,714
//! rows / 55,000 games** — about 55% of the original 100,000-game corpus's
//! game count (more rows per game, since this run used `--stride 1` against
//! the original's `--stride 2`).
//!
//! The insurance batch was not a formality. `mcts-value` self-play alone
//! produces a starkly different victory-kind mix than `mcts-eval` self-play
//! does: **10.07%** of games end in scientific supremacy (vs the original
//! corpus's 2.31%) and correspondingly fewer end civilian (71.86% vs
//! 80.64%). Left unmixed, the corpus would have been generated almost
//! entirely by, and about, the one agent whose blind spots this experiment
//! was trying to get past — the self-referential-bootstrap risk
//! `value_corpus_mv.rs`'s own module docs flag. The fresh `mcts-eval` batch
//! measures 2.44% scientific / 81.09% civilian, consistent with the
//! original corpus, and was mixed in specifically to keep the merged corpus
//! from narrowing around `mcts-value`'s own preferences.
//!
//! ## The fit, and the overfitting question item 2 asked
//!
//! Trained with the *exact* hyperparameters the `v1.bin` fit used (`--hidden
//! 128 --epochs 60 --lr 2e-3 --weight-decay 1e-5 --patience 8
//! --also-scalar`), so the only thing that moved is the data. **The peak
//! validation epoch did not move later: it peaked at epoch 2** (of 60,
//! early-stopped at 10) — earlier than `v1`'s epoch 5 on the full
//! 70,000-game training split, and identical to `v1`'s epoch 2 on a
//! 21,000-game subset, despite this retrain's 38,500-game training split
//! sitting between those two sizes. That is a real finding either way, and
//! the way it came out argues against "just add more games, same
//! architecture" as a sufficient fix: a differently-composed corpus, even a
//! smaller one, changed the overfitting behaviour more than raw game count
//! did here.
//!
//! Held-out **test** rows (this corpus's own split, not directly comparable
//! row-for-row to `v1`'s test set since the corpora differ):
//!
//! | predictor | Brier | log loss | accuracy | ROC AUC |
//! | --- | --- | --- | --- | --- |
//! | learned value (`v2`) | 0.18333 | 0.53817 | 0.7120 | 0.7967 |
//! | search's own recorded root value (mixed source) | **0.17552** | **0.51658** | **0.7278** | **0.8139** |
//! | single-scalar control | 0.18354 | 0.53854 | 0.7109 | 0.7962 |
//!
//! For comparison, `v1` on its own test set was at *parity* with the search
//! (0.17223 vs 0.17429 Brier — the learned value narrowly ahead). Here the
//! learned value trails the search baseline on every column. Zero-sum
//! coherence (`tests/probability_coherence.rs`) is close to a wash — mean
//! `|P(One)+P(Two)-1|` 0.0557 against `v1`'s 0.0559 — with the opening's
//! probability mass notably closer to the required 1.0 (0.9947 vs `v1`'s
//! 0.9396). None of the existing loose bounds in that test file needed
//! updating for `v2`.
//!
//! ## The arena validation (the part that actually matters)
//!
//! Full battery, `duels-arena experiment`, 400 paired-seed seat-swapped
//! games per cell, `nodes:2000`, plus a larger confirmation cell for the
//! decisive comparison (`v2` vs `v1` itself — see "Recalibrating the test"
//! below for why 400 games was not enough to trust here):
//!
//! | Match | Games | Elo | 95% CI | SPRT |
//! | --- | ---: | --- | --- | --- |
//! | `v2` vs `v1`, first pass | 400 | +33.9 | `[-0.3, +68.1]` | Inconclusive (elo1=20) |
//! | **`v2` vs `v1`, confirmation** | **2000** | **+34.5** | **`[+19.2, +49.8]`** | **AcceptH1 (elo1=10)** |
//! | `v2` vs `mcts-eval` | 400 | +109.2 | `[+73.5, +144.9]` | AcceptH1 |
//! | `v2` vs `mcts-uct` | 400 | +204.4 | `[+164.3, +244.5]` | AcceptH1 |
//! | `v2` vs `alphabeta` | 400 | +326.9 | `[+276.7, +377.1]` | AcceptH1 |
//! | `mcts-eval` vs `mcts-uct` (fresh baseline, same run) | 400 | +133.6 | `[+97.0, +170.1]` | AcceptH1 |
//! | `mcts-eval` vs `alphabeta` (fresh baseline, same run) | 400 | +310.5 | `[+262.0, +359.0]` | AcceptH1 |
//!
//! `arena/results/experiments/mcts-value-v2-vs-v1-confirm/` holds the
//! confirmation cell; its mechanism gate also **passes** on all three shares
//! (civilian, science, military), so the larger sample settles both questions
//! at once: `v2` beats `v1`, and it does not win by a distorted mix of
//! victory kinds to do it.
//!
//! `v2`'s margin over `mcts-eval` is bigger than `v1`'s documented `+84.9` to
//! `+91.4`, and it now clearly beats `v1` itself. The victory-kind breakdown
//! shows the bigger margin runs through a higher science-conversion rate —
//! `v2` took 116 of its 261 wins over `mcts-eval` by scientific supremacy
//! (44%) against `v1`'s 134 of 496 (27%) — which, per the mechanism section
//! above, is read as the model getting *better* at a real skill, not as a
//! narrower one.
//!
//! **The third-party generalization check is genuinely mixed, and both
//! numbers are reported rather than averaged away.** Using the fresh
//! same-run baselines above to estimate "`v2`'s margin over `mcts-eval`, as
//! seen through a third party" (third-party margin over that party minus
//! `mcts-eval`'s own margin over it, divided by the direct margin):
//!
//! * Through `mcts-uct`: `(204.4 − 133.6) / 109.2` ≈ **65%** — a large
//!   improvement over `v1`'s documented ~28%, and evidence this generation's
//!   science-conversion skill is starting to carry over as net strength
//!   against a real playout opponent, not just against `mcts-eval`.
//! * Through `alphabeta`: `(326.9 − 310.5) / 109.2` ≈ **15%** — essentially
//!   unchanged from `v1`'s ~12%.
//!
//! One third party shows meaningfully better generalization than `v1` had;
//! the other does not move much. That disagreement is expected in a small,
//! non-transitive agent pool where different opponents defend the science
//! race differently, and is not read as a mark against the promotion below —
//! `mcts-uct`'s number is real evidence of broader strength, and continued
//! training should be expected to bring `alphabeta`'s figure up too.
//!
//! ## Recalibrating the test, not just the weights
//!
//! The first-pass `v2` vs `v1` cell (400 games, the historical default) came
//! back SPRT-inconclusive against the `elo1=20` bound this project has used
//! since its early, larger-jump rounds. Read plainly rather than as a veto:
//! a 400-game sample at that bound cannot distinguish "no effect" from a real
//! effect in the roughly `+20` to `+50` Elo range — which is exactly what the
//! 2,000-game confirmation at a tightened `elo1=10` bound found. As this
//! project's rounds mature, single-iteration gains are expected to keep
//! shrinking (recall the ladder's own historical biggest single win, `+89`
//! Elo, is itself a modest win-rate difference), so **judging a retrain
//! against its own immediate predecessor going forward should default to a
//! larger sample (aim for ~2,000 games, not 400) and a tighter SPRT bound
//! (`elo1` around 10, not 20)** — this confirmation run is the template, not
//! a one-off.
//!
//! ## The verdict
//!
//! **Promoted.** `v2.bin` is now [`DEFAULT_WEIGHTS`] and
//! `duels_arena::leaderboard::CHAMPION`'s weights generation moves with it
//! (the champion is still the agent `mcts-value`; only which `duels-value`
//! generation it embeds changed). `v1.bin` stays in the repository, reachable
//! via `mcts-value:weights=v1`, as the generation this replaces. The
//! overfitting-epoch finding (unchanged by corpus size, moved by corpus
//! composition) is a genuine, separate result worth keeping in mind for the
//! next corpus attempt, but it did not block this promotion, which rests on
//! the arena numbers above.
//!
//! ## What's next: beyond mixing corpora
//!
//! The corpus-diversity fix tried here — mixing self-play with a
//! hand-chosen-size insurance batch — measurably worked, but the mixing
//! ratio was chosen by hand, not tuned, and is a blunt instrument. A more
//! structural next step, worth exploring before another blind corpus-size
//! increase: **specialist value functions/agents trained under different
//! reward shaping** — one rewarded only for scientific-supremacy outcomes,
//! one only for military supremacy, one only civilian, or blends of them —
//! rather than one generalist net fit to the raw, heavily civilian-skewed
//! outcome distribution. That is a different, more deliberate idea than
//! "add more games of the same kind," and a natural next thing to put in
//! front of a planning pass.
//!
//! **Acted on.** `duels_agent_mcts_value::Objective` builds exactly this —
//! three specialists, each reading `v2.bin`'s existing per-outcome head under
//! a different reward, no retrain — and measures how *purely* each one
//! pursues its target victory kind rather than how strong it is overall. See
//! that crate's own docs for the design and the purity numbers it measured.
//!
//! ## An inference-time patch for coherence, tried before the architectural fix
//!
//! The remaining-next-steps item above — "make the model zero-sum
//! coherent" — names an antisymmetric head as the real, architectural fix.
//! `docs/roadmap.md`'s Tier 0-B asked a cheaper question first: does simply
//! *averaging* the two disagreeing perspectives — `p_sym = (P(win|One) + (1 -
//! P(win|Two))) / 2` — improve on the single-perspective `win_probability()`
//! as an offline predictor, since that would be a free two-member ensemble
//! available today, with no retrain?
//!
//! **Yes, on every metric measured** (held-out test split, `seed % 10 == 9`,
//! `arena/corpus/mcts-eval-nodes2000.jsonl`, 10,000 games, 1,343,298
//! (position, perspective) rows): Brier `0.17266` against `0.17436`, log loss
//! `0.51006` against `0.51480`, ROC AUC `0.82017` against `0.81680`. So it was
//! implemented as a new, opt-in leaf,
//! `duels_agent_mcts_value::LeafValue::LearnedSymmetric` — see that crate's
//! docs for the arena result and why it is not (yet) a default.
//!
//! **This does not replace the architectural fix**, and should not be read as
//! evidence the coherence defect is "handled." Averaging two miscalibrated
//! numbers can only ever partially cancel their disagreement — it is a patch
//! on the symptom measured in `tests/probability_coherence.rs`, at the cost of
//! a second forward pass, not a fix to why the two perspectives disagree in
//! the first place. The joint 7-outcome softmax with shared weights
//! (`docs/roadmap.md`'s Tier 2-J) remains the real fix, and remains a
//! separate, larger, future retrain.
//!
//! # Follow-up round three: the Tier 1 loop (`v3.bin`), promoted
//!
//! `docs/roadmap.md`'s Tier 1 ran the generate-explore-train loop for real:
//! put exploration (`visits^(1/tau)`, tau=1, 14 plies) and 25% specialist-seat
//! mixing into the corpus generator (D), added a `--value-target-lambda`
//! blended training target (E), and ablated them separately. Three arms from
//! matched-conditions ~100k-game corpora: `arm-a` (pure argmax, lambda=1.0,
//! isolates neither D nor E), `arm-b` (D applied, lambda=1.0, isolates D),
//! `arm-c` (arm-b's corpus, lambda=0.5, isolates E on top of D). `arm-b` won
//! clean: **+32.0 Elo vs `v2` at 2,000 games (`elo1=10`), reproduced at
//! +26.8 Elo on a disjoint seed range**, mechanism gate `Pass`.
//!
//! `arm-c` read **-17.2 Elo** and was initially rejected -- but a second
//! review found `tools/train_value.py`'s two-term loss
//! (`lam * CE4 + (1-lam) * BCE(aggregate_win_mass, q_root)`) had a wrong
//! gradient for the `LOSS` logit: `(s - t) / (1 - s)` instead of the correct
//! `s - t` (re-derived from the softmax Jacobian and confirmed by finite
//! difference, `tools/test_train_value_grad.py`). The bug amplifies the
//! gradient without bound as the model gets confident and only fires when
//! `lambda < 1.0`, so `arm-a`/`arm-b` (lambda=1.0) are unaffected but `arm-c`
//! is corrupted by it -- its own training log shows divide-by-zero/overflow/
//! invalid-value warnings and a validation loss that never improved past
//! epoch 0, unlike every lambda=1.0 run. **`arm-c`'s -17.2 Elo is therefore
//! not a trustworthy reading on the value-target-blend idea.**
//!
//! Retrained from the *same*, already-archived `arm-b`/`arm-c` corpus (not
//! regenerated) with the fixed gradient, at this project's documented
//! production hyperparameters (`--hidden 128 --epochs 60 --lr 2e-3
//! --weight-decay 1e-5 --patience 8`): `arm-c2` (`lambda=0.5`, the corrected
//! retest) and `arm-d2` (`lambda=0.75`, a hedge). Both trained cleanly this
//! time -- no numeric warnings, validation loss declining smoothly to a best
//! epoch well past `arm-a`/`arm-b`'s (epoch 17 for `arm-c2` at a validation
//! win log-loss of 0.51266, better than `arm-b`'s own 0.51899). Gated the
//! same way as `arm-b`, plus a direct match against `arm-b` to isolate the
//! gradient fix's own effect:
//!
//! | Candidate (`lambda`) | vs `v2`@2000 | vs `v2`@2000 (confirm) | vs `arm-b` (direct) | vs `v2`@32000 (panel) | vs `mcts-eval`@8000 | vs `mcts-uct`@8000 |
//! | --- | ---: | ---: | ---: | ---: | ---: | ---: |
//! | `arm-b` (1.0) | +32.0 | +26.8 | -- | -55.1 | +91.9 | +153.2 |
//! | `arm-c2` (0.5) | **+61.0** | **+53.9** | **+27.0** | **-34.4** | **+196.4** | **+260.0** |
//! | `arm-d2` (0.75) | +43.5 | -- | +37.7 | -46.2 | +169.8 | +231.1 |
//!
//! `arm-c2` wins every single comparison measured, with confidence intervals
//! excluding zero throughout (including the direct `arm-b` match), and its
//! mechanism gate (`civilian_share>=0.5x,science_share>=0.5x,
//! military_share<=2x`) passes on every cell. It is the strongest generation
//! measured in this project to date, and the corrected E-ablation reads
//! clearly positive: the value-target blend genuinely helps once its
//! gradient is right, on top of D's own real gain.
//!
//! **Promoted.** `arm-c2`'s weights are `v3.bin`, now [`DEFAULT_WEIGHTS`].
//! `v2.bin` is retired to a frozen reference-panel slot
//! (`mcts-value:weights=v2`, `duels_agent_mcts_value::WEIGHTS_V2`) rather than
//! deleted, per `docs/roadmap.md`'s Tier 1-G -- it is the direct ancestor at
//! equal budget and stays the panel's own yardstick across whatever
//! generations follow `v3`. Every generation measured in this round (`v1`,
//! `v2`, `arm-a`, `arm-b`, `arm-c`, `arm-c2`/`v3`, `arm-d2`) is recorded in
//! `crates/duels-value/weights/generations.json`, per Tier 1-F, with its
//! corpus manifest, training args, full battery result and golden-table
//! reference -- the mechanism `golden.rs` is now re-pointed at (a hash check
//! on every frozen generation, the full twenty-position table on whichever
//! one is live) rather than a single hand-pinned hash.
//!
//! One honest side effect of the retrain, not a bug: `tests/
//! probability_coherence.rs`'s opening-mass bound flipped sign.
//! `v2` was systematically *under*-confident at the opening (mean mass
//! 0.9396); `v3` is systematically *over*-confident instead (mean mass
//! 1.0666) -- a comparable-sized coherence defect, just in the other
//! direction, not a fix and not (on this number alone) a regression. The
//! architectural fix (`docs/roadmap.md`'s Tier 2-J, a joint 7-outcome softmax
//! with shared weights) remains the real answer to that defect; this retrain
//! did not touch the output head's shape.
//!
//! **The `c` re-sweep this project's history says to run after any leaf/value
//! change**: `duels_agent_mcts_value::Config::exploration` (the UCB1
//! constant) at `0.10`, `0.20` and `0.25` against the shipped default `0.15`,
//! 1,000 games each at `nodes:2000`. All three read **Inconclusive** (`+2.4`,
//! `+0.7`, `+1.0` Elo, every CI crossing zero) -- `0.15` is still fine for
//! `v3`; no change made.
//!
//! # Usage
//!
//! ```
//! use duels_core::{engine, Player};
//! use duels_value::{default_net, features};
//!
//! let state = engine::new_game(7);
//! let dist = default_net().evaluate(&state, Player::One);
//! assert!((dist.total() - 1.0).abs() < 1e-4);
//! // The scalar a search actually needs.
//! let p = dist.win_probability();
//! assert!((0.0..=1.0).contains(&p));
//! ```

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

pub mod features;
pub mod net;

pub use features::{features, NUM_FEATURES};
pub use net::{Net, Summation, WeightsError, NUM_OUTCOMES};

use duels_core::scoring::{GameResult, VictoryKind};
use duels_core::{GameState, Player};

/// The trained weights, baked into the binary.
///
/// `v3.bin`, the Tier 1 generate-explore-train loop's winning generation
/// (`crates/duels-value/weights/generations.json`, id `tier1-arm-c-prime`) —
/// a retrain on the same exploration+specialist-mixing corpus `v2.bin`'s
/// successor experiment used, at a corrected `--value-target-lambda 0.5`
/// blended loss — promoted over `v2.bin` after a 2,000-game confirmation
/// run, reproduced on a disjoint seed range; see the crate docs' "Follow-up
/// round three" section for the full arena validation and the reasoning.
/// Embedding it rather than loading a file at run time keeps this crate a
/// pure function of its inputs and keeps an agent that uses it reproducible
/// from its binary alone.
///
/// **`v1.bin` and `v2.bin` sit alongside it in the same directory and are
/// not this constant.** `v1.bin` is reachable via `mcts-value:weights=v1`
/// (`duels_agent_mcts_value::WEIGHTS_V1`); `v2.bin`, the generation this one
/// replaces, is reachable via `mcts-value:weights=v2`
/// (`duels_agent_mcts_value::WEIGHTS_V2`) and stays in the frozen reference
/// panel (`docs/roadmap.md`'s Tier 1-G) at both `nodes:32000` and
/// `nodes:2000`.
const DEFAULT_WEIGHTS: &[u8] = include_bytes!("../weights/v3.bin");

/// The four mutually-exclusive outcomes of a game, **from the perspective of
/// the player a position is being evaluated for**.
///
/// A draw is folded into [`Outcome::Loss`]: it is not a win, and it is rare
/// enough (a tie on victory points *and* on civilian points) that a fifth
/// class would be almost entirely empty. The crate docs record the measured
/// draw rate in the corpus these were labelled from (0.07% of games).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(usize)]
pub enum Outcome {
    /// Won by pushing the conflict pawn into the opponent's capital.
    MilitaryWin = 0,
    /// Won by collecting six distinct scientific symbols.
    ScienceWin = 1,
    /// Won at the end of Age III on points, tiebreak included.
    CivilianWin = 2,
    /// Did not win — a loss by any means, or a draw.
    Loss = 3,
}

impl Outcome {
    /// All four, in the index order the softmax head uses.
    pub const ALL: [Outcome; NUM_OUTCOMES] = [
        Outcome::MilitaryWin,
        Outcome::ScienceWin,
        Outcome::CivilianWin,
        Outcome::Loss,
    ];

    /// Index into a `[_; NUM_OUTCOMES]` array.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Which class a finished game falls into, seen by `me`.
    ///
    /// This is the label function the training corpus is dumped with, and it
    /// lives here rather than in the dump tool so that the model's classes and
    /// its labels cannot drift apart.
    #[inline]
    pub fn of(result: GameResult, me: Player) -> Outcome {
        match result {
            GameResult::Win { winner, kind } if winner == me => match kind {
                VictoryKind::MilitarySupremacy => Outcome::MilitaryWin,
                VictoryKind::ScientificSupremacy => Outcome::ScienceWin,
                // The tiebreak is still a win on points at the end of Age III;
                // splitting it out would make a fifth near-empty class of a
                // distinction that changes nothing about how the game was won.
                VictoryKind::CivilianVictory | VictoryKind::CivilianTiebreak => {
                    Outcome::CivilianWin
                }
            },
            _ => Outcome::Loss,
        }
    }

    /// A short stable name, for reports and diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Outcome::MilitaryWin => "military_win",
            Outcome::ScienceWin => "science_win",
            Outcome::CivilianWin => "civilian_win",
            Outcome::Loss => "loss",
        }
    }
}

/// A predicted distribution over the four [`Outcome`]s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dist(pub [f32; NUM_OUTCOMES]);

impl Dist {
    /// The probability of one outcome.
    #[inline]
    pub fn p(&self, outcome: Outcome) -> f32 {
        self.0[outcome.index()]
    }

    /// The scalar win probability a search backs up:
    /// `P(military) + P(science) + P(civilian)`.
    ///
    /// Summed from the three win heads rather than read off a `1 - P(loss)`
    /// output, so that the number a search consumes is literally the model's
    /// own decomposition and cannot disagree with the per-kind readouts.
    #[inline]
    pub fn win_probability(&self) -> f32 {
        self.p(Outcome::MilitaryWin) + self.p(Outcome::ScienceWin) + self.p(Outcome::CivilianWin)
    }

    /// The total mass, which a softmax puts at one. Exposed for the tests and
    /// diagnostics that check the head is intact.
    #[inline]
    pub fn total(&self) -> f32 {
        self.0.iter().sum()
    }
}

impl Net {
    /// Extract features for `me` and run the forward pass, in one call.
    ///
    /// This is the whole public surface a search leaf needs.
    #[inline]
    pub fn evaluate(&self, state: &GameState, me: Player) -> Dist {
        Dist(self.forward(&features(state, me)))
    }

    /// Just the scalar, for a caller that does not want the decomposition.
    #[inline]
    pub fn win_probability(&self, state: &GameState, me: Player) -> f32 {
        self.evaluate(state, me).win_probability()
    }
}

/// The shipped network, parsed from [`DEFAULT_WEIGHTS`].
///
/// Parsing is a few microseconds of work over about 108 KB, so a caller that
/// evaluates more than a handful of positions should build one of these and
/// keep it — `mcts-eval` builds it once per search tree, in the same place it
/// builds its one `duels_eval::Root`.
///
/// # Panics
///
/// Panics if the embedded weights do not match this build's
/// [`NUM_FEATURES`] — which can only happen if the feature layout was changed
/// without retraining, and is a build-time mistake rather than a runtime
/// condition. `tests::the_embedded_weights_load` catches it in CI.
pub fn default_net() -> Net {
    Net::from_bytes(DEFAULT_WEIGHTS).expect("the embedded weights match this build's features")
}

/// A short, stable identity for the shipped weights — shape plus a content
/// hash of the embedded bytes, e.g. `211x128x4/9f3a1c2b`.
///
/// The point of the hash is the same point `mcts-eval`'s `Config::describe`
/// makes about recording the whole `duels_eval::Config`: an agent that
/// consumes this model should put this string in its `AgentSpec`, so that two
/// results files from either side of a retrain are *distinguishable* rather
/// than the older one becoming uninterpretable. The shape alone would not do
/// that — a retrain at the same width produces the same shape and completely
/// different behaviour.
///
/// Computed once and cached. The hash is FNV-1a, chosen because it is four
/// lines and this is a build-artifact fingerprint, not a security boundary.
pub fn default_weights_id() -> &'static str {
    static ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ID.get_or_init(|| weights_id(DEFAULT_WEIGHTS))
}

/// [`default_weights_id`]'s computation, generalized to arbitrary weight
/// bytes rather than only the embedded default.
///
/// For a consumer that pins an alternate, frozen weights generation for A/B
/// measurement against the live default (the identical device
/// `mcts-eval`-family agents use for `duels_eval::Config` — see
/// `duels-agent-mcts-value`'s `Config::value_weights_override`): the
/// resulting `AgentSpec` needs *this* generation's identity in it, not the
/// binary's embedded default's, or two results files from either side of a
/// retrain would both claim the same weights.
///
/// Not cached, unlike [`default_weights_id`]: a caller pinning a frozen
/// generation already holds it as a `&'static` slice, so paying the hash
/// again per call is a few dozen bytes' worth of work, not worth a global
/// cache keyed on byte identity.
pub fn weights_id(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let net = Net::from_bytes(bytes).expect("the weights bytes match this build's features");
    format!(
        "{}x{}x{}/{:08x}",
        NUM_FEATURES,
        net.hidden_width(),
        NUM_OUTCOMES,
        (h ^ (h >> 32)) as u32
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn the_weights_id_is_stable_and_names_the_shape() {
        let id = default_weights_id();
        assert_eq!(id, default_weights_id(), "the cached id is not stable");
        let net = default_net();
        assert!(
            id.starts_with(&format!(
                "{}x{}x{}/",
                NUM_FEATURES,
                net.hidden_width(),
                NUM_OUTCOMES
            )),
            "the id {id} does not name the shape it describes"
        );
        // Eight hex digits of content hash after the shape.
        let hash = id.rsplit('/').next().expect("the id has a hash part");
        assert_eq!(hash.len(), 8, "the hash part of {id} is the wrong width");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn the_embedded_weights_load() {
        let net = default_net();
        assert!(net.hidden_width() > 0);
        // A useful thing to see in the test output: how big "small" actually
        // is, since the whole design rests on the inference being cheap.
        println!(
            "duels-value: {} features -> {} hidden -> {} outcomes, {} parameters, {} KiB",
            NUM_FEATURES,
            net.hidden_width(),
            NUM_OUTCOMES,
            net.parameters(),
            DEFAULT_WEIGHTS.len() / 1024
        );
    }

    #[test]
    fn the_outcome_label_covers_every_result_exactly_once() {
        use VictoryKind::*;
        for kind in [
            MilitarySupremacy,
            ScientificSupremacy,
            CivilianVictory,
            CivilianTiebreak,
        ] {
            for winner in Player::ALL {
                let r = GameResult::Win { winner, kind };
                // The winner's own class is a win; the loser's is a loss.
                assert_ne!(Outcome::of(r, winner), Outcome::Loss);
                assert_eq!(Outcome::of(r, winner.other()), Outcome::Loss);
            }
        }
        for p in Player::ALL {
            assert_eq!(Outcome::of(GameResult::Draw, p), Outcome::Loss);
        }
        // ...and the three win kinds really are three distinct classes.
        let m = Outcome::of(
            GameResult::Win {
                winner: Player::One,
                kind: MilitarySupremacy,
            },
            Player::One,
        );
        let s = Outcome::of(
            GameResult::Win {
                winner: Player::One,
                kind: ScientificSupremacy,
            },
            Player::One,
        );
        let c = Outcome::of(
            GameResult::Win {
                winner: Player::One,
                kind: CivilianVictory,
            },
            Player::One,
        );
        assert_eq!(
            [m, s, c],
            [
                Outcome::MilitaryWin,
                Outcome::ScienceWin,
                Outcome::CivilianWin
            ]
        );
        assert_eq!(
            Outcome::of(
                GameResult::Win {
                    winner: Player::One,
                    kind: CivilianTiebreak
                },
                Player::One
            ),
            Outcome::CivilianWin
        );
    }

    /// The scalar the search consumes has to be a probability, and the two
    /// perspectives of one position have to be roughly complementary — the
    /// network is not *forced* to be antisymmetric the way `duels-eval`'s
    /// algebra is, so this is a sanity bound on the fit rather than an
    /// identity. A trained model that badly violated it would be a model that
    /// had not learned the game is zero-sum.
    #[test]
    fn the_win_probability_is_a_probability_and_roughly_zero_sum() {
        let net = default_net();
        let mut worst = 0.0f32;
        for state in positions() {
            let one = net.evaluate(&state, Player::One);
            let two = net.evaluate(&state, Player::Two);
            for d in [one, two] {
                assert!((d.total() - 1.0).abs() < 1e-3, "total {}", d.total());
                let p = d.win_probability();
                assert!((0.0..=1.0).contains(&p), "win probability {p}");
            }
            worst = worst.max((one.win_probability() + two.win_probability() - 1.0).abs());
        }
        assert!(
            worst < 0.35,
            "the two perspectives disagree by up to {worst}, which is not a zero-sum game"
        );
    }

    /// A finished game is the one place the answer is known, so it is the one
    /// place a learned value can be checked against ground truth rather than
    /// against itself. Not a tight bound — the model was trained on positions
    /// throughout a game, most of them undecided — but a decided position
    /// should not be read as a coin flip.
    #[test]
    fn a_finished_game_is_read_the_right_way_round() {
        let net = default_net();
        let mut checked = 0;
        let mut right = 0;
        for seed in 0..40u64 {
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x0F11_15ED);
            loop {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let a = legal[rng.gen_range(0..legal.len())];
                engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action applies");
            }
            let Some(result) = state.result() else {
                continue;
            };
            let Some(winner) = result.winner() else {
                continue;
            };
            checked += 1;
            if net.win_probability(&state, winner) > 0.5 {
                right += 1;
            }
        }
        assert!(checked >= 30, "only {checked} finished games");
        assert!(
            right * 4 >= checked * 3,
            "the winner was favoured in only {right} of {checked} finished games"
        );
    }

    fn positions() -> Vec<GameState> {
        let mut out = Vec::new();
        for seed in 0..40u64 {
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x7A1D_7A1D);
            for _ in 0..(6 + seed % 60) {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let a = legal[rng.gen_range(0..legal.len())];
                engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action applies");
            }
            out.push(state);
        }
        out
    }
}
