//! What the learned leaf actually costs, measured against the thing it sits
//! beside.
//!
//! ```text
//! cargo run --release -p duels-value --example value_bench
//! ```
//!
//! # Why the ratio and not the microseconds
//!
//! The absolute figures move with the machine and with whatever else is on it.
//! The **ratio** of one [`duels_value::Net::evaluate`] to one full random
//! playout does not, and it is the number the design rests on: a leaf value
//! that costs a small fraction of the playout it is blended with is
//! effectively free, and one that costs a multiple of it turns a fixed-node
//! measurement into a misleading one. `mcts-eval`'s `examples/leaf_bench.rs`
//! makes the same argument for the same reason, and both take their two
//! measurements in one run under one load so they are comparable to each
//! other.
//!
//! Measured on the machine this crate was developed on (Apple Silicon, 14
//! logical cores, one other agent's build sharing the box — so read the ratio,
//! not the microseconds):
//!
//! ```text
//! features            0.29 us
//! forward            15.79 us
//! evaluate (both)    15.72 us
//! full playout       35.08 us
//! evaluate / playout  44.8%   (features 2% of that, forward 100%)
//! ```
//!
//! **This came out the opposite way round from the guess it replaced, and the
//! difference matters.** The expectation was that feature extraction — two
//! `Breakdown`s, two trade-price tables and a `card_cost` per accessible slot
//! — would dominate a "mere" 27,520 multiply-adds. It does not: the features
//! are 0.29 µs and the forward pass is fifty times that.
//!
//! The reason is not the FLOP count, it is the **dependency chain**. Each
//! hidden unit is a serial `acc += w * x` reduction over 211 terms, and
//! floating-point addition is not associative, so LLVM may neither reorder nor
//! vectorise it; the loop runs at the latency of one `f32` add per element
//! rather than at throughput. `15.79 µs / 27,520` is about 0.57 ns per
//! multiply-add, which is roughly two cycles — exactly the latency-bound
//! figure, and about ten times off the throughput-bound one.
//!
//! **The consequence for how to read the crate docs' Elo numbers.** At 45% of
//! a playout, `LeafValue::LearnedBlend` costs about `1.45x` per simulation
//! where the default `LeafValue::Blend` costs about `1.08x`, so at equal wall
//! clock the learned blend runs roughly `0.74x` the simulations. Against the
//! budget-scaling curve at these budgets (~22 Elo per doubling) that is worth
//! about `-9` Elo — so the fixed-node `+106` should be read as an expectation
//! of roughly `+95` at a fixed time budget, not as a figure that transfers
//! unchanged. The crate docs say the same thing and flag the wall-clock
//! confirmation as outstanding.
//!
//! **The obvious repair, deliberately not applied.** Splitting the
//! accumulator into four independent partial sums breaks the dependency chain
//! and should recover most of the ten-fold gap. It is not done here because it
//! changes the summation order and therefore the network's output in the last
//! couple of `f32` digits, which would mean the shipped code no longer
//! computes what the measured Elo was measured with. Doing it is a small,
//! self-contained follow-up whose *own* before/after Elo run is cheap, and it
//! should be done that way rather than folded in silently.

use std::time::Instant;

use duels_core::{engine, GameState, Player};
use duels_value::{default_net, features};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Positions spread across the draft and all three ages, so the timing is not
/// taken entirely on cheap early boards.
fn positions() -> Vec<GameState> {
    let mut out = Vec::new();
    for seed in 0..64u64 {
        for plies in [4usize, 16, 30, 44, 58] {
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xBEA7_C051);
            let mut ok = true;
            for _ in 0..plies {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    ok = false;
                    break;
                }
                let a = legal[rng.gen_range(0..legal.len())];
                engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action applies");
            }
            if ok && state.result().is_none() {
                out.push(state);
            }
        }
    }
    out
}

fn main() {
    let states = positions();
    let net = default_net();
    println!(
        "{} positions, net {} params, {} features",
        states.len(),
        net.parameters(),
        duels_value::NUM_FEATURES
    );
    println!();

    // A warm-up pass, so the first measured loop is not paying for cold caches
    // and a cold branch predictor.
    let mut sink = 0.0f32;
    for s in &states {
        sink += net.win_probability(s, Player::One);
    }

    const REPS: usize = 200;

    // `Instant::now` is banned inside `duels-core` and the agent crates by
    // `clippy.toml`, and allowed here for the reason that file names: this is a
    // benchmark, nothing in the library reads a clock, and a benchmark that
    // cannot time anything is not a benchmark.
    #[allow(clippy::disallowed_methods)]
    let time = |name: &str, mut f: Box<dyn FnMut(&GameState) -> f32 + '_>| -> f64 {
        let start = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..REPS {
            for s in &states {
                acc += f(s);
            }
        }
        let per = start.elapsed().as_secs_f64() / (REPS * states.len()) as f64;
        // Consumed so the optimiser cannot delete the work being timed.
        println!("{name:<20} {:>7.2} us   (checksum {acc:.3e})", per * 1e6);
        per
    };

    let t_features = time(
        "features",
        Box::new(|s| features(s, Player::One).iter().sum::<f32>()),
    );
    let t_forward = {
        // Precomputed, so this line times the matrix arithmetic alone.
        let xs: Vec<_> = states.iter().map(|s| features(s, Player::One)).collect();
        #[allow(clippy::disallowed_methods)]
        let start = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..REPS {
            for x in &xs {
                acc += net.forward(x)[0];
            }
        }
        let per = start.elapsed().as_secs_f64() / (REPS * xs.len()) as f64;
        println!(
            "{:<20} {:>7.2} us   (checksum {acc:.3e})",
            "forward",
            per * 1e6
        );
        per
    };
    let t_evaluate = time(
        "evaluate (both)",
        Box::new(|s| net.win_probability(s, Player::One)),
    );

    // One full random playout to a real `GameResult` — the leaf value this
    // whole line of work is trying to improve on, and the denominator of the
    // only ratio here that travels between machines.
    #[allow(clippy::disallowed_methods)]
    let t_playout = {
        let mut rng = StdRng::seed_from_u64(0x9110_1234);
        let mut done = 0usize;
        let start = Instant::now();
        for _ in 0..40 {
            for s in &states {
                let mut state = *s;
                loop {
                    let legal = engine::legal_actions(&state);
                    if legal.is_empty() {
                        break;
                    }
                    let a = legal[rng.gen_range(0..legal.len())];
                    engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action applies");
                }
                sink += f32::from(u8::from(state.result().is_some()));
                done += 1;
            }
        }
        let per = start.elapsed().as_secs_f64() / done as f64;
        println!("{:<20} {:>7.2} us", "full playout", per * 1e6);
        per
    };

    println!();
    println!(
        "evaluate / playout  {:>6.1}%   (features {:.0}% of that, forward {:.0}%)",
        100.0 * t_evaluate / t_playout,
        100.0 * t_features / t_evaluate,
        100.0 * t_forward / t_evaluate,
    );
    println!("(checksum {sink:.3e})");
}
