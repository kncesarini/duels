//! What does the learned value cost, per call?
//!
//! A leaf value is paid once per simulation, so its absolute cost decides how
//! much of a wall-clock budget it eats. The numbers to compare against are the
//! ones `mcts-eval`'s `leaf_bench.rs` reports: a whole simulation with a
//! playout leaf is about 18.8 µs, and one cached-`Root` `duels_eval::evaluate`
//! is about 0.4 µs.
//!
//! ```text
//! cargo run --release -p duels-value --example value_bench
//! ```
//!
//! Measured over real mid-game positions (a short seeded random prefix per
//! seed), separately for the feature extraction and the forward pass, so the
//! two halves can be read against each other.

use duels_core::{engine, GameState, Player};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn positions(n: u64) -> Vec<GameState> {
    let mut out = Vec::with_capacity(n as usize);
    for seed in 0..n {
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xBE9C);
        for _ in 0..(8 + seed % 50) {
            let legal = engine::legal_actions(&st);
            if legal.is_empty() {
                break;
            }
            let a = legal[rng.gen_range(0..legal.len())];
            engine::apply_quiet(&mut st, a, &mut rng).unwrap();
        }
        if !st.is_over() {
            out.push(st);
        }
    }
    out
}

fn main() {
    let reps: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let pos = positions(200);
    let model = duels_value::model::embedded();
    println!(
        "duels-value: {} features, hidden {}, {} positions x {reps} reps",
        duels_value::NUM_FEATURES,
        model.hidden(),
        pos.len()
    );
    println!("weights: {}", model.describe());

    // Features only.
    #[allow(clippy::disallowed_methods)]
    let t0 = std::time::Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..reps {
        for st in &pos {
            let x = duels_value::features(st, Player::One);
            sink += x[9];
        }
    }
    let feat = t0.elapsed().as_secs_f64() / f64::from(reps) / pos.len() as f64;

    // Forward pass only, on precomputed features.
    let xs: Vec<[f32; duels_value::NUM_FEATURES]> = pos
        .iter()
        .map(|st| duels_value::features(st, Player::One))
        .collect();
    let nonzero: f64 = xs
        .iter()
        .map(|x| x.iter().filter(|&&v| v != 0.0).count() as f64)
        .sum::<f64>()
        / xs.len() as f64;
    #[allow(clippy::disallowed_methods)]
    let t1 = std::time::Instant::now();
    for _ in 0..reps {
        for x in &xs {
            sink += model.predict(x).win();
        }
    }
    let fwd = t1.elapsed().as_secs_f64() / f64::from(reps) / xs.len() as f64;

    // Both, as a search calls it.
    #[allow(clippy::disallowed_methods)]
    let t2 = std::time::Instant::now();
    for _ in 0..reps {
        for st in &pos {
            sink += duels_value::win_probability(st, Player::One) as f32;
        }
    }
    let both = t2.elapsed().as_secs_f64() / f64::from(reps) / pos.len() as f64;

    println!();
    println!("  features            {:8.3} us", feat * 1e6);
    println!(
        "  forward pass        {:8.3} us   ({nonzero:.0} non-zero inputs on average)",
        fwd * 1e6
    );
    println!("  win_probability     {:8.3} us", both * 1e6);
    println!();
    println!("(sink {sink:.3}, so the optimiser kept every call)");
}
