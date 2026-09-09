//! `duels-value`: a **learned** 7 Wonders Duel position value — a small
//! neural network that predicts, from public information only, which of four
//! mutually-exclusive outcomes a position ends in.
//!
//! # Where this sits, and what it is for
//!
//! A library **below the agents**, next to [`duels_core`] and alongside
//! `duels-strategy` and `duels-eval`, for the reason `CLAUDE.md` gives: an
//! agent crate may not depend on another agent crate, so anything more than
//! one search wants to consume moves *down* into a library rather than
//! sideways. It depends on `duels-core` and on nothing else — no `rand`, no
//! clock, no ML runtime. [`features`] and [`Net::forward`] are pure functions.
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
//! the reigning champion, not `mcts-uct`), paired-seed and seat-swapped, at
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
//! That is the sharpest single result here. `CLAUDE.md`'s standing prior —
//! *simulation beats hand-crafted judgement for position value in this game* —
//! was established by `LeafValue::Static` losing badly, and it is the low
//! ceiling of a *hand-crafted* evaluation that the prior is really about. A
//! learned value of the same shape, with no playout at all, beats the
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
//!    treat that as an estimate, not a measurement. `CLAUDE.md` is emphatic
//!    that this project has been burned by exactly this both ways.
//! 2. **One budget, one opponent.** Everything above is `Nodes(32000)` against
//!    one control. A leaf that steers this hard into science races (31.5% of
//!    the pure variant's wins, against a ~2.3% base rate in self-play) could
//!    be exploiting something specific about *this* opponent rather than
//!    playing better in general. The `duels-arena experiment` mechanism gate
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
//! `CLAUDE.md`'s standing instruction that a change to what a leaf backs up
//! must re-derive `c` was right, and the reason to state it this loudly is
//! that the spike above read `0.5` as "close enough to leave alone".
//!
//! ## Caveat 1 partly resolved: it is not a high-budget artefact
//!
//! At the ladder's **production** budget, `Nodes(2000)`, two disjoint
//! 400-game ranges (`p0-learned-nodes2000`): **+91.4 Elo [+66.5, +116.3]**,
//! 503-297, SPRT `AcceptH1`, ranges +128.6 [+92.2, +165.0] and
//! +55.9 [+21.5, +90.4] — **both cell CIs exclude zero**. The mechanism gate
//! reads `Pass` on `civilian_share` (58.8% against a required 43.4%) and on
//! `military_share` (14.9% against a permitted 19.5%), which is a better
//! mechanism read than either spike run above obtained.
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
//! ### The mechanism, which is more specific than "it beats one opponent"
//!
//! The science-seeking behaviour *does* transfer. It just stops paying.
//!
//! | pairing | candidate's wins | opponent's wins |
//! | --- | --- | --- |
//! | cand vs `mcts-uct` | mil 38, **sci 89**, civ 171 → 298 | mil 8, sci 0, civ 88 → 101 |
//! | `mcts-eval` vs `mcts-uct` | mil 34, **sci 10**, civ 237 → 287 | mil 16, sci 2, civ 88 → 113 |
//!
//! Against `mcts-uct` the learned leaf still converts 89 scientific
//! supremacies where `mcts-eval` converts 10, so it genuinely *sees* the race
//! — the offline per-head numbers (science AUC 0.955) were not a fluke. But
//! its total is 298 against 287, because those extra science wins come almost
//! entirely out of **its own civilian column** (171 against 237). It is
//! **route substitution, not extra wins.**
//!
//! That reframes the direct +91.5 exactly. Against `mcts-eval` the same
//! behaviour scores, because `mcts-eval` concedes 129 science games in 800 and
//! wins **one** — it cannot defend the race at all, which is the
//! miscalibration `science_calibration` documented. Against an opponent whose
//! leaf is a real playout, and which therefore does see races, the learned
//! leaf converts games it would have won by other means.
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
//! opponent is what distinguishes a stronger agent from a counter to a
//! specific one**, and that is worth remembering the next time a leaf change
//! measures well against the champion alone.
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
//! 2. **A `TimeMs(1000)` confirmation**, still outstanding for the pure leaf.
//!    Note the cost profile now *favours* it: with the four-way unroll
//!    (`Summation::Unrolled4`, worth 1.41x) the learned leaf costs about
//!    `0.35x` a playout, against the default blend's `1.08x` — so at equal
//!    wall clock the pure learned leaf should run on the order of `3x` the
//!    simulations, the opposite sign from `LearnedBlend`'s `0.74x`.
//! 3. **More corpus games**, unchanged: the fit peaks at epoch 5 on 70,000
//!    training games and at epoch 2 on 21,000, so this is variance-limited and
//!    the corpus is the binding resource, not the architecture.
//! 4. **Whatever is tried next, measure it against `mcts-uct` too**, for the
//!    reason the section above gives.
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
/// Produced by `tools/train_value.py` from a feature dump of
/// `arena/corpus/mcts-eval-nodes2000.jsonl`; the crate docs record the exact
/// commands, the seed split and the held-out metrics. Embedding it
/// rather than loading a file at run time keeps this crate a pure function of
/// its inputs and keeps an agent that uses it reproducible from its binary
/// alone.
const DEFAULT_WEIGHTS: &[u8] = include_bytes!("../weights/v1.bin");

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
    ID.get_or_init(|| {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in DEFAULT_WEIGHTS {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        let net = default_net();
        format!(
            "{}x{}x{}/{:08x}",
            NUM_FEATURES,
            net.hidden_width(),
            NUM_OUTCOMES,
            (h ^ (h >> 32)) as u32
        )
    })
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
