//! What victory-point score means what win probability?
//!
//! [`duels_eval::evaluate`] returns a number on a rough victory-point scale.
//! A search that wants to use it as a **leaf value** — `mcts-uct` backs up
//! `[0, 1]` win probabilities, not victory points — needs a mapping between
//! the two, and the honest way to get one is to measure it rather than to pick
//! a constant that looks about right.
//!
//! So: replay real games, record `(evaluate at this position, who eventually
//! won)` at every decision, and fit
//!
//! ```text
//! P(the player to move wins) = 1 / (1 + exp(-v / T))
//! ```
//!
//! by maximum likelihood in the one free parameter `T` — the **temperature**,
//! in victory points. `T` is the score at which the model says the mover wins
//! about 73% of the time; a small `T` means the evaluation is confident, a
//! large one that it is noisy.
//!
//! # How the games are replayed
//!
//! By playing them, which is what `duels-arena`'s own audit examples
//! (`rail_audit.rs`, `wonder_audit.rs`) do: `duels_arena::match_runner`'s
//! `GameRecord` stores a seed, the two agent specs and the result, not the
//! positions, so there is no position-level replay file in this project to
//! read. A seed and a policy *are* the replay. This example uses `phased`
//! self-play — the driver is `PhasedAgent::choose` line for line, since this
//! crate sits below every agent crate — over seeds `0..games` on each of the
//! two seat orders, and the same `seed ^ 0xF00D` engine stream as
//! `eval_bench.rs` and `decision_cost.rs`, so the three walk the same games.
//!
//! # What is deliberately *not* claimed
//!
//! The fit is over positions from one policy's self-play, so it calibrates the
//! evaluation *along the lines `phased` actually plays*. A search visiting
//! wilder positions would see a different spread. Drawn games are dropped
//! rather than counted as half a win — there are very few, and a Bernoulli
//! likelihood has nowhere to put them. Terminal positions are skipped: their
//! score is `±instant_result`, which is a rail, not a judgement.
//!
//! ```text
//! cargo run --release -p duels-eval --example calibrate
//! cargo run --release -p duels-eval --example calibrate -- 200
//! ```

use duels_core::scoring::GameResult;
use duels_core::{engine, Action, Observation, Player};
use duels_eval::{evaluate, expected_value, Config, Root};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `PhasedAgent`'s tie window, copied because the tie set feeds the RNG draw
/// and so decides which move comes out.
const TIE_EPSILON: f64 = 1e-6;

/// A verbatim copy of `PhasedAgent::choose`, RNG usage included.
struct Driver {
    rng: StdRng,
    config: Config,
}

impl Driver {
    fn new(seed: u64) -> Driver {
        Driver {
            rng: StdRng::seed_from_u64(seed),
            config: Config::default(),
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let me = obs.current_player;
        let base_state = obs.sample_state(&mut self.rng);
        let root = Root::new(&base_state, me, self.config);

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

/// One observation for the fit: what the evaluation said, and what happened.
#[derive(Clone, Copy)]
struct Sample {
    /// `evaluate(state, state.current_player(), root)`.
    value: f64,
    /// Did the player the score was read for go on to win?
    won: bool,
    /// The age the position was in (1, 2 or 3).
    age: u8,
}

/// Mean negative log-likelihood of `samples` under temperature `t`.
fn nll(samples: &[Sample], t: f64) -> f64 {
    let mut acc = 0.0;
    for s in samples {
        let z = s.value / t;
        // `ln(1 + e^-z)` computed in the stable direction on both tails.
        let ln_one_plus_exp = |x: f64| {
            if x > 0.0 {
                x + (-x).exp().ln_1p()
            } else {
                x.exp().ln_1p()
            }
        };
        acc += if s.won {
            ln_one_plus_exp(-z)
        } else {
            ln_one_plus_exp(z)
        };
    }
    acc / samples.len().max(1) as f64
}

/// The maximum-likelihood temperature, by golden-section search on `ln T`.
///
/// The likelihood is unimodal in `T` here and the search is one-dimensional
/// over a wide bracket, so nothing cleverer earns its keep.
fn fit_temperature(samples: &[Sample]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let (mut lo, mut hi) = (0.05f64.ln(), 200.0f64.ln());
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;
    let mut c = hi - phi * (hi - lo);
    let mut d = lo + phi * (hi - lo);
    let (mut fc, mut fd) = (nll(samples, c.exp()), nll(samples, d.exp()));
    for _ in 0..200 {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - phi * (hi - lo);
            fc = nll(samples, c.exp());
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + phi * (hi - lo);
            fd = nll(samples, d.exp());
        }
        if (hi - lo).abs() < 1e-9 {
            break;
        }
    }
    ((lo + hi) / 2.0).exp()
}

/// Fraction of samples the fitted model calls correctly at `p = 0.5`, which
/// for this model is just `sign(value)`. Reported because it does not depend
/// on `T` at all, and so says whether the *ordering* is any good separately
/// from whether the scale is.
fn sign_accuracy(samples: &[Sample]) -> f64 {
    let right = samples.iter().filter(|s| (s.value > 0.0) == s.won).count();
    right as f64 / samples.len().max(1) as f64
}

/// A reliability table: what the model predicted against what happened.
fn reliability(samples: &[Sample], t: f64) {
    let edges = [-30.0, -15.0, -7.0, -3.0, 0.0, 3.0, 7.0, 15.0, 30.0];
    println!(
        "\n  {:>16}  {:>8}  {:>10}  {:>10}",
        "score band", "n", "predicted", "actual"
    );
    for i in 0..=edges.len() {
        let lo = if i == 0 {
            f64::NEG_INFINITY
        } else {
            edges[i - 1]
        };
        let hi = if i == edges.len() {
            f64::INFINITY
        } else {
            edges[i]
        };
        let bucket: Vec<&Sample> = samples
            .iter()
            .filter(|s| s.value >= lo && s.value < hi)
            .collect();
        if bucket.is_empty() {
            continue;
        }
        let predicted: f64 = bucket
            .iter()
            .map(|s| 1.0 / (1.0 + (-s.value / t).exp()))
            .sum::<f64>()
            / bucket.len() as f64;
        let actual = bucket.iter().filter(|s| s.won).count() as f64 / bucket.len() as f64;
        println!(
            "  {:>7.1}..{:<7.1}  {:>8}  {:>9.3}  {:>9.3}",
            lo,
            hi,
            bucket.len(),
            predicted,
            actual
        );
    }
}

fn main() {
    // `calibrate <games> [first seed]`. The seed offset exists so a refit can
    // be reproduced on a **disjoint** range of games before it is believed,
    // which is this project's standing rule for anything measured (`CLAUDE.md`)
    // and applies to a fitted constant exactly as it applies to an Elo.
    let games: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let first: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let config = Config::default();
    let mut samples: Vec<Sample> = Vec::new();
    let mut decided = 0u32;
    let mut draws = 0u32;

    for seed in first..first + games {
        // Both seat orders, so a first-player advantage this large (see
        // `CLAUDE.md`) cannot bias the fit towards whoever moves first.
        for swap in [false, true] {
            let seeds = if swap {
                (seed * 1000 + 2, seed * 1000 + 1)
            } else {
                (seed * 1000 + 1, seed * 1000 + 2)
            };
            let mut driver = [Driver::new(seeds.0), Driver::new(seeds.1)];
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xF00D);
            let mut trace: Vec<(f64, Player, u8)> = Vec::new();

            while !state.is_over() {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let me = state.current_player();
                let root = Root::new(&state, me, config);
                trace.push((evaluate(&state, me, &root), me, state.age()));

                let obs: Observation = state.observation();
                let action = driver[me.index()].choose(&obs, &legal);
                engine::apply_quiet(&mut state, action, &mut rng)
                    .expect("the driver plays legally");
            }

            match state.result() {
                Some(GameResult::Win { winner, .. }) => {
                    decided += 1;
                    for (value, me, age) in trace {
                        samples.push(Sample {
                            value,
                            won: me == winner,
                            age,
                        });
                    }
                }
                _ => draws += 1,
            }
        }
    }

    println!(
        "{} positions from {decided} decided games ({draws} drawn, dropped)\n",
        samples.len()
    );

    let t = fit_temperature(&samples);
    println!(
        "  overall      T = {:>6.2} VP   mean NLL {:.4}   sign accuracy {:.3}   n = {}",
        t,
        nll(&samples, t),
        sign_accuracy(&samples),
        samples.len()
    );

    for age in 1..=3u8 {
        let by_age: Vec<Sample> = samples.iter().copied().filter(|s| s.age == age).collect();
        if by_age.is_empty() {
            continue;
        }
        let t_age = fit_temperature(&by_age);
        println!(
            "  age {age}        T = {:>6.2} VP   mean NLL {:.4}   sign accuracy {:.3}   n = {}",
            t_age,
            nll(&by_age, t_age),
            sign_accuracy(&by_age),
            by_age.len()
        );
    }

    reliability(&samples, t);

    println!(
        "\n  A leaf value for a [0, 1]-backing-up search is therefore\n  \
         1 / (1 + exp(-evaluate(state, me, root) / {t:.2}))."
    );
}
