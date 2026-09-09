//! A **learned** position value for 7 Wonders Duel: a small network trained
//! offline on actual game outcomes, with public-information features and
//! hand-rolled `f32` inference. A library below the agents, alongside
//! `duels-eval`, so any search may consume it without depending on another
//! agent (see "the layering" in `CLAUDE.md`).
//!
//! # Status: a feasibility spike, not a shipped default
//!
//! This crate exists to answer one question honestly — *does a learned value
//! beat the current default leaf at something close to production budget?* —
//! and nothing in it is a default anywhere. `mcts-eval` consumes it only
//! through opt-in `LeafValue` variants. Read the crate's measurement notes at
//! the bottom of this page before drawing a conclusion from a number.
//!
//! # Why a learned value, and why now
//!
//! `CLAUDE.md` records two findings that point the same way. A hand-crafted
//! static evaluation has a low ceiling in this game, yet blending one into an
//! MCTS leaf was the largest single gain the project has measured (`+89` Elo).
//! And the budget-scaling curve flattens hard past `Nodes(2000)` — the
//! signature of a search limited by *leaf signal quality* rather than breadth.
//! The next step of the same mechanism is a better leaf. This crate is the
//! cheapest serious test of whether a network trained on **what actually
//! happened** in 100,000 self-play games is that better leaf.
//!
//! Trained against actual outcomes, deliberately, and not against the search's
//! own recorded root value: an evaluation fitted to the search's opinion
//! reproduces its blind spots and, as a separate investigation in this
//! repository found, destroys the *complementarity* that makes a blended leaf
//! useful.
//!
//! # The target is victory-kind decomposed
//!
//! The network is a **4-way classifier** over the mutually exclusive ways a
//! game ends for the evaluated player — `military_win`, `science_win`,
//! `civilian_win` (tiebreak folded in), `loss` — trained with cross-entropy,
//! not a single sigmoid over win/loss. Scientific supremacy decides only about
//! 2.3% of games, and a single scalar target lets that rare, structurally
//! distinct outcome dissolve into the majority; forcing the network to also
//! predict it is the standard multi-task argument for a better shared
//! representation. The scalar a search wants is recovered by summing the three
//! winning heads ([`Distribution::win`]), so nothing about consuming this is
//! more complicated than a single number. Whether the decomposition actually
//! helps the aggregate is *measured* against a single-scalar baseline by
//! `tools/train.py`, not asserted — see the notes below.
//!
//! # Pieces
//!
//! - [`features`]: the public-information feature vector ([`NUM_FEATURES`]
//!   small integers), from one player's perspective. Determinization-invariant
//!   by construction and by test (`tests/determinization_invariance.rs`).
//! - [`Model`]: `NUM_FEATURES -> hidden -> 4`, ReLU then softmax, forward pass
//!   only, first layer walked sparsely. Weights arrive as a little file the
//!   trainer writes (`weights/value.bin`), embedded at build time.
//! - [`predict`] / [`win_probability`]: the two entry points a search uses.
//! - `crates/duels-arena/examples/feature_dump.rs`: replays the value corpus
//!   and writes `(features, outcome label)` rows plus comparison columns.
//! - `tools/train.py`: the offline trainer (Python, numpy/torch, **not** a
//!   workspace dependency). Splits train/validation **by game**, trains the
//!   decomposed model and a single-scalar control, reports held-out metrics,
//!   folds input normalisation into the first layer and writes the weights
//!   file.
//!
//! # Determinism and layering
//!
//! No randomness, no clock, no I/O at run time: [`predict`] is a pure function
//! of the state. The crate depends on `duels-core` and nothing else — not
//! `duels-eval`, so the learned signal can be compared to the hand-crafted one
//! without either containing the other.
//!
//! # What was measured
//!
//! Everything below is from one session on the 100,000-game
//! `mcts-eval-nodes2000` corpus (`arena/corpus/`, regenerable with
//! `value_corpus.rs`). Reproduce with the commands under "Reproducing".
//!
//! ## The model, on held-out games
//!
//! `feature_dump` produced 6,722,636 rows (one per searched decision, from
//! [`Player::One`]'s perspective). Trained on the 90,000 games with seeds
//! `1..=90000` (6.05 M rows), validated on the 10,000 games `90001..=100000`
//! (672 k rows) — **split by game**. `493-128-4`, ReLU, softmax, Adam,
//! 8 epochs, batch 4096, one-cycle learning rate `2e-3`, inputs RMS-scaled.
//! Every signal below is scored on the *same* validation rows against the
//! game's real outcome (`1` win, `0` loss, `0.5` draw):
//!
//! | signal | Brier | log-loss | sign accuracy |
//! |---|---|---|---|
//! | the search's own root value (`mcts-eval`, 2,000 simulations) | 0.1726 | 0.5136 | 0.740 |
//! | `duels_eval::win_probability` (the hand-crafted static value) | 0.2170 | 0.6272 | 0.658 |
//! | **this crate, decomposed 4-way, `P(win)` = sum of three heads** | **0.1829** | **0.5492** | **0.721** |
//! | single-scalar control, identical shape and schedule | 0.1872 | 0.5632 | 0.715 |
//!
//! Two readings. First, as a *static* value the network is a much better
//! predictor of what actually happens than `duels-eval` is — roughly
//! three-quarters of the way from the hand-crafted evaluation to the search's
//! own 2,000-simulation verdict, in 1.8 µs. Second, **the victory-kind
//! decomposition helps the aggregate**: the 4-way model's summed win
//! probability beats the single-sigmoid control by `0.0043` Brier and `0.014`
//! log-loss on 672 k held-out rows, the same direction and about the same size
//! as on a 400 k-row smoke run. That is the multi-task claim measured rather
//! than asserted, and it held.
//!
//! By age (held-out Brier): the search `0.224 / 0.176 / 0.101`, this crate
//! `0.226 / 0.184 / 0.126`, `duels-eval` `0.233 / 0.212 / 0.202`. The
//! learned value keeps pace with the search in Age I, where almost nothing is
//! decided, and falls behind in Age III, where the search's simulations are
//! reading a concrete endgame.
//!
//! Per head: the `science_win` head — 1.1% of rows — is well calibrated up to
//! `0.2` and over-confident above it (predicts `0.73`, sees `0.53`); the
//! aggregate is over-confident at both tails (predicts `0.965` where `0.906`
//! happens, `0.032` where `0.089` does). Some regularisation would sharpen
//! this; it was not pursued because of what the search measurement said.
//!
//! ## Inside the search
//!
//! `examples/value_bench.rs`: features `0.29 µs`, forward pass about `1.5 µs`
//! (112 non-zero inputs on average), `1.8 µs` for [`win_probability`] — well
//! inside the few-µs budget a leaf has, against an 18.8 µs playout.
//!
//! Wired into `mcts-eval` as opt-in `LeafValue::Learned` and
//! `LeafValue::LearnedBlend`, and measured with `duels-arena experiment` at
//! `Nodes(32000)` (a load-insensitive stand-in for the production wall-clock
//! budget), two disjoint 100-game seed ranges, paired and seat-swapped,
//! against `mcts-eval`'s default (`blend:0.5, c=0.5`):
//!
//! | candidate | W-L-D | Elo | 95% CI | per range | candidate wins mil/sci/civ/tie | control wins |
//! |---|---|---|---|---|---|---|
//! | `leaf=learned` — learned value alone | 54-146-0 | **-171.8** | [-225.8, -117.8] | -180, -162 | 8/20/25/1 | 11/0/133/2 |
//! | `leaf=learned-blend:0.5` — half learned, half playout; ranges `1`, `10001` | 104-96-0 | +13.8 | [-34.2, +61.9] | 0, +28 | 14/9/80/1 | 11/0/84/1 |
//! | `leaf=learned-blend:0.5`; ranges `20001`, `30001` | 108-91-1 | +29.5 | [-18.7, +77.7] | +3, +56 | 16/16/74/2 | 8/0/81/2 |
//! | `leaf=learned-blend:0.5`; **all four ranges pooled (400 games)** | 212-187-1 | **+21.7** | [-12.4, +55.9] | — | 30/25/154/3 | 19/0/165/3 |
//!
//! **Verdict: the pre-registered bar was not cleared.** The learned value
//! alone is a clear negative; the learned value blended with a playout is at
//! *parity* with the default's hand-crafted blend — `+21.7` Elo over 400
//! games, positive on every range but with an interval containing zero and
//! excluding the `+50` ceiling this spike was chartered against, and SPRT
//! `Continue` everywhere. Nothing here is adopted as a default, and nothing
//! should be on this evidence. The one clear, reproducible difference is the
//! *mechanism*: the learned blend wins by scientific supremacy 25 times to
//! the control's 0 across 400 games, on both pairs of ranges, so the
//! `science_win` head does inside a search exactly what the decomposition
//! was designed to make possible — and that sight is worth, at most, a couple
//! of dozen Elo against a control that already has half a playout.
//!
//! **The learned value alone is a clear negative**, and it is the *same*
//! negative the project already measured for a pure hand-crafted leaf
//! (`-170.7` Elo at `Nodes(2000)`; `mcts-eval`'s crate docs). Same size, same
//! victory-kind signature — the static leaf over-collects science (20 wins by
//! scientific supremacy against the control's 0) and loses the points game
//! (25 against 133). Being far better calibrated on the corpus distribution
//! did not move that at all, which sharpens `CLAUDE.md`'s prior: what a
//! static leaf lacks is not *accuracy on positions like the ones it was
//! trained on*, it is the playout's sight of how the next few moves resolve —
//! and a search hands its leaf value positions the corpus never contained
//! (deep in a tree, after the candidate's own hypothetical line), where a
//! function fitted to self-play trajectories is extrapolating.
//!
//! ## Reproducing
//!
//! ```text
//! cargo run --release -p duels-arena --example feature_dump -- \
//!     --corpus arena/corpus/mcts-eval-nodes2000.jsonl --out arena/corpus/features/full
//! cargo run --release -p duels-value --example train -- \
//!     --data arena/corpus/features/full --out crates/duels-value/weights/value.bin
//! cargo run --release -p duels-value --example value_bench
//! cargo run --release -p duels-arena -- experiment \
//!     --candidate mcts-eval:leaf=learned --control mcts-eval \
//!     --seeds 1,10001 --budgets nodes:32000 --games 100 --label learned-alone
//! cargo run --release -p duels-arena -- experiment \
//!     --candidate mcts-eval:leaf=learned-blend:0.5 --control mcts-eval \
//!     --seeds 1,10001 --budgets nodes:32000 --games 100 --label learned-blend
//! ```
//!
//! `weights/value.bin`'s embedded provenance line ([`describe`]) names the
//! training set and the held-out numbers of the exact weights in the build,
//! and `mcts-eval` records it in its `AgentSpec` params whenever a learned
//! leaf is in use, so a results file says which network it was measured with.

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

pub mod features;
pub mod model;

pub use features::{feature_names, features, NUM_FEATURES};
pub use model::{Distribution, Model, NUM_OUTCOMES};

use duels_core::{GameState, Player};

/// The embedded network's outcome distribution for `state`, from `me`'s side.
pub fn predict(state: &GameState, me: Player) -> Distribution {
    let x = features(state, me);
    model::embedded().predict(&x)
}

/// `P(me wins)` under the embedded network: the three winning heads summed,
/// on the `[0, 1]` scale a search backs up.
pub fn win_probability(state: &GameState, me: Player) -> f64 {
    f64::from(predict(state, me).win())
}

/// The embedded weights' provenance line, for an `AgentSpec` to record.
pub fn describe() -> &'static str {
    model::embedded().describe()
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn the_win_probability_is_a_probability_on_real_positions() {
        for seed in 0..30u64 {
            let mut st = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed);
            for _ in 0..(seed as usize * 3 % 60) {
                let legal = engine::legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                let a = legal[rng.gen_range(0..legal.len())];
                engine::apply_quiet(&mut st, a, &mut rng).unwrap();
            }
            for me in Player::ALL {
                let p = win_probability(&st, me);
                assert!(p.is_finite() && (0.0..=1.0).contains(&p), "{p}");
                let d = predict(&st, me);
                assert!((d.as_array().iter().sum::<f32>() - 1.0).abs() < 1e-5);
            }
        }
    }
}
