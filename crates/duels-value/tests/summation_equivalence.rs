//! The four-accumulator forward pass computes the same function as the
//! serial one — bounded numerically, before anything is believed about its
//! speed.
//!
//! # Why this test exists before the benchmark does
//!
//! `examples/value_bench.rs` explains why the original hidden layer cost 45%
//! of a full playout instead of the couple of microseconds 27,520
//! multiply-adds should take: each unit was one serial `f32` reduction over
//! 211 terms, and since `f32` addition is not associative LLVM could neither
//! reorder nor vectorise it, so the loop ran at add *latency*. It also
//! explains why the obvious repair was deliberately left undone —
//! "it changes the summation order and therefore the network's output in the
//! last couple of `f32` digits, which would mean the shipped code no longer
//! computes what the measured Elo was measured with" — and says the fix should
//! land with its own before/after measurement rather than folded in silently.
//!
//! This file is the first half of that: a **behaviour-preserving** claim,
//! bounded on real positions, so that any later speed claim is a claim about
//! the same function. The second half is an arena A/B whose result the crate
//! docs record.
//!
//! # What "the same function" can and cannot mean here
//!
//! It cannot mean bit-identical: reassociating a floating-point sum changes
//! the result, and that is the whole point of doing it. So the bound is
//! numerical — `1e-5` absolute on every one of the four softmax outputs and on
//! the scalar a search consumes — which is four orders of magnitude below the
//! differences that matter to a search (the model's own calibration error is
//! measured in hundredths; see `tests/probability_coherence.rs`).
//!
//! The bound is asserted on positions from real games rather than on random
//! vectors, because cancellation is data-dependent: a feature vector of mostly
//! small non-negative numbers with a few large ones is the case that actually
//! occurs, and a synthetic uniform vector would not exercise it.

use duels_core::{engine, GameState, Player};
use duels_value::{default_net, features, Summation, NUM_OUTCOMES};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The absolute tolerance on every output. See the module docs for why it is
/// a numerical bound and not bit equality.
const TOLERANCE: f32 = 1e-5;

/// A thousand positions from real games, spread over the draft and all three
/// ages so the comparison is not taken entirely on cheap early boards.
fn positions() -> Vec<GameState> {
    let mut out = Vec::new();
    'seeds: for seed in 0..256u64 {
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5044_0DDD);
        for ply in 0..70 {
            if ply % 8 == 0 {
                out.push(state);
                if out.len() >= 1_000 {
                    break 'seeds;
                }
            }
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                continue 'seeds;
            }
            let a = legal[rng.gen_range(0..legal.len())];
            if engine::apply_quiet(&mut state, a, &mut rng).is_err() {
                continue 'seeds;
            }
        }
    }
    out
}

#[test]
fn the_two_summation_orders_agree_to_within_the_tolerance() {
    let serial = default_net().with_summation(Summation::Serial);
    let unrolled = default_net().with_summation(Summation::Unrolled4);
    let all = positions();
    assert!(
        all.len() >= 1_000,
        "only {} positions; the bound was taken on 1,000",
        all.len()
    );

    let mut worst_output = 0.0f32;
    let mut worst_scalar = 0.0f32;
    let mut checked = 0usize;
    for state in &all {
        for me in [Player::One, Player::Two] {
            let x = features(state, me);
            let a = serial.forward(&x);
            let b = unrolled.forward(&x);
            for k in 0..NUM_OUTCOMES {
                let d = (a[k] - b[k]).abs();
                worst_output = worst_output.max(d);
                assert!(
                    d < TOLERANCE,
                    "output {k} differs by {d:.3e}, above the {TOLERANCE:.0e} tolerance"
                );
            }
            let sa = serial.win_probability(state, me);
            let sb = unrolled.win_probability(state, me);
            let d = (sa - sb).abs();
            worst_scalar = worst_scalar.max(d);
            assert!(
                d < TOLERANCE,
                "win probability differs by {d:.3e} ({sa} vs {sb})"
            );
            checked += 1;
        }
    }
    // Recorded rather than asserted: the margin the bound is passing by is
    // what a future change to `dot4` should be compared against.
    println!(
        "{checked} (position, perspective) pairs: worst output difference \
         {worst_output:.3e}, worst scalar difference {worst_scalar:.3e}"
    );
}

/// Anti-vacuity, in both directions.
///
/// The test above would pass trivially if `with_summation` did nothing and
/// both handles ran the same code, so this proves the two arms are genuinely
/// distinct — *and* that they are distinct in the last few digits rather than
/// in a way that would make the tolerance above a lie.
#[test]
fn the_two_orders_really_are_different_code_paths() {
    let serial = default_net().with_summation(Summation::Serial);
    let unrolled = default_net().with_summation(Summation::Unrolled4);
    assert_eq!(serial.summation(), Summation::Serial);
    assert_eq!(unrolled.summation(), Summation::Unrolled4);

    // `forward` must route to the arm the handle names.
    let x = features(&engine::new_game(1), Player::One);
    assert_eq!(serial.forward(&x), serial.forward_serial(&x));
    assert_eq!(unrolled.forward(&x), unrolled.forward_unrolled4(&x));

    // Somewhere in a thousand positions the two orders must disagree at all,
    // or reassociation was optimised away and the benchmark measures nothing.
    let differed = positions()
        .iter()
        .flat_map(|s| [Player::One, Player::Two].map(|me| features(s, me)))
        .any(|x| serial.forward_serial(&x) != serial.forward_unrolled4(&x));
    assert!(
        differed,
        "the two summation orders produced bit-identical results on every \
         position, which means they are not actually different reductions"
    );
}

/// The default is the unrolled order, so an agent that says nothing gets the
/// fast path. Pinned as a test because flipping it is a deliberate act with
/// an arena run behind it, not a default that should drift.
#[test]
fn the_default_summation_is_the_unrolled_one() {
    assert_eq!(default_net().summation(), Summation::Unrolled4);
    assert_eq!(Summation::default(), Summation::Unrolled4);
    assert_eq!(Summation::Serial.name(), "serial");
    assert_eq!(Summation::Unrolled4.name(), "unrolled4");
}
