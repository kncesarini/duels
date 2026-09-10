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
//! # The obvious repair, now applied — and it is worth `1.41x`, not ten
//!
//! Splitting the accumulator into four independent partial sums (now
//! [`duels_value::Summation::Unrolled4`], the default) was predicted here to
//! "recover most of the ten-fold gap". **It does not, and the prediction was
//! wrong.** Both orders, timed in one run under one load:
//!
//! ```text
//! features                0.29 us
//! forward (serial)       52.88 us
//! forward (unrolled4)    37.54 us
//! evaluate (both)        38.21 us
//! full playout          108.97 us
//!
//! evaluate / playout    35.1%   (features 1% of that, forward 98%)
//! unroll speedup        1.41x
//! serial   evaluate / playout would be  48.8%
//! unrolled evaluate / playout would be  34.7%
//! ```
//!
//! (The microseconds are about three times the figures above them because that
//! run shared the machine with four concurrent arena matches. The *ratios* are
//! the comparable part, which is this file's whole argument, and the two
//! summation rows were timed back to back inside one process.)
//!
//! So the dependency chain was **not** the whole story. Four accumulators
//! should give close to `4x` on a purely latency-bound reduction, and `1.41x`
//! says the loop is substantially **memory**-bound as well: `w1` is
//! `128 × 211 × 4` bytes = 108 KiB, which does not sit in L1, so every call
//! streams the whole weight matrix from L2. Widening the unroll further would
//! not help; what would is shrinking or quantising `w1`, or evaluating several
//! positions against one pass over the weights. Neither is done here, and the
//! second would need a batching interface a search leaf does not currently
//! have.
//!
//! The gain is still real and worth having: it takes the learned leaf from
//! roughly half a playout to roughly a third of one. For the **pure**
//! `LeafValue::Learned` — no playout at all — that means about `0.35x` a
//! playout against the default `LeafValue::Blend`'s `1.08x`, so at equal wall
//! clock the pure learned leaf runs on the order of `3x` the simulations. That
//! is the opposite sign from `LearnedBlend`'s `0.74x` above, and it is why the
//! wall-clock question has to be asked separately for the two variants rather
//! than answered once.
//!
//! Because the unroll reassociates a floating-point sum it is not
//! bit-identical to the order the crate docs' Elo was measured with, so
//! [`duels_value::Summation::Serial`] is kept reachable
//! (`mcts-eval:leaf=learned,value_sum=serial`) and the change was measured on
//! both axes rather than argued about:
//!
//! * **numerically**, `tests/summation_equivalence.rs` — worst difference
//!   `4.768e-7` over 2,000 (position, perspective) pairs, against a `1e-5`
//!   tolerance, i.e. passing by a factor of 21;
//! * **in Elo**, `arena/results/experiments/p1-unroll-ab{,-extended}` —
//!   serial against unrolled at `c = 0.15`, `Nodes(32000)`, three disjoint
//!   seed ranges, **800 games: `-1.7` Elo [`-25.8`, `+22.3`], 397-401-2.**
//!   Zero, as a reassociation at a fixed node count has to be.
//!
//! That Elo run is also a small lesson in sample size, and it is recorded
//! because it nearly produced a wrong conclusion. The first range alone said
//! `-41.7` [`-90.1`, `+6.7`], which looks alarming and is not significant;
//! the three cells were `-41.7`, `-18.5` and `+41.8`. There is no mechanism by
//! which a `5e-7` arithmetic difference can cost strength at a **fixed node
//! count** — a node is a node — so the scatter is divergent trajectories and
//! nothing else, and 200 games is simply not enough to see zero.

use std::time::Instant;

use duels_core::{engine, GameState, Player};
use duels_value::{default_net, features, NUM_FEATURES};

/// One forward-pass variant under the timer: a feature vector in, one output
/// out (so the optimiser cannot delete the work).
type ForwardUnderTest<'a> = Box<dyn FnMut(&[f32; NUM_FEATURES]) -> f32 + 'a>;
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
    // Precomputed, so these lines time the matrix arithmetic alone.
    let xs: Vec<_> = states.iter().map(|s| features(s, Player::One)).collect();
    // Both accumulation orders, in one run under one load, so the ratio
    // between them is meaningful even though the microseconds are not.
    #[allow(clippy::disallowed_methods)]
    let time_forward = |name: &str, mut f: ForwardUnderTest<'_>| {
        let start = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..REPS {
            for x in &xs {
                acc += f(x);
            }
        }
        let per = start.elapsed().as_secs_f64() / (REPS * xs.len()) as f64;
        println!("{name:<20} {:>7.2} us   (checksum {acc:.3e})", per * 1e6);
        per
    };
    let t_forward_serial = time_forward("forward (serial)", Box::new(|x| net.forward_serial(x)[0]));
    let t_forward = time_forward(
        "forward (unrolled4)",
        Box::new(|x| net.forward_unrolled4(x)[0]),
    );
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
    println!(
        "unroll speedup      {:>6.2}x   (serial {:.2} us -> unrolled4 {:.2} us)",
        t_forward_serial / t_forward,
        t_forward_serial * 1e6,
        t_forward * 1e6,
    );
    println!(
        "serial   evaluate / playout would be {:>5.1}%",
        100.0 * (t_features + t_forward_serial) / t_playout
    );
    println!(
        "unrolled evaluate / playout would be {:>5.1}%",
        100.0 * (t_features + t_forward) / t_playout
    );
    println!("(checksum {sink:.3e})");
}
