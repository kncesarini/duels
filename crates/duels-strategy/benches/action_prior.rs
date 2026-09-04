//! What does this layer cost, and can a search afford to pay it per node?
//!
//! ```text
//! cargo bench -p duels-strategy
//! ```
//!
//! The number that matters is not the absolute time but the *ratio* to a
//! Monte-Carlo rollout, since a rollout is the unit of work an MCTS node buys.
//! `mcts_rollout_reference` therefore measures a uniform-random playout to a
//! terminal state from the same mid-game position, with the same
//! `apply_unchecked` + `legal_actions_into` loop `duels-agent-mcts-uct`'s own
//! rollout uses — measured here rather than quoted, so the comparison is
//! against a number from the same machine and the same run.
//!
//! Nothing is asserted: a threshold that passes on a laptop is flaky on a
//! shared CI runner. The observed figures are recorded in the crate's PR
//! description.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use duels_core::{engine, Action, GameState};
use duels_strategy::{action_prior, military_read, science_read, stance, vp_read, Board};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A real mid-game position with a full slate of legal moves, plus that list.
///
/// Walks a random game until it reaches an ordinary turn offering at least
/// `min_actions` moves, so the measurement reflects a branchy node rather than
/// a two-option effect choice.
fn mid_game(seed: u64, min_steps: usize, min_actions: usize) -> (GameState, Vec<Action>) {
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x4242);
    let mut steps = 0usize;
    loop {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            panic!("seed {seed} finished before reaching a branchy position");
        }
        if steps >= min_steps && legal.len() >= min_actions {
            return (state, legal);
        }
        let a = legal[rng.gen_range(0..legal.len())];
        engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action");
        steps += 1;
    }
}

/// One uniform-random playout to a terminal state — the unit of work an MCTS
/// simulation buys.
fn rollout(mut state: GameState, rng: &mut StdRng) -> u32 {
    let mut buf: Vec<Action> = Vec::with_capacity(64);
    loop {
        engine::legal_actions_into(&state, &mut buf);
        if buf.is_empty() {
            break;
        }
        let a = buf[rng.gen_range(0..buf.len())];
        engine::apply_unchecked(&mut state, a, rng);
    }
    state.turn()
}

fn bench(c: &mut Criterion) {
    // At least 30 decisions in and at least 18 legal moves: a mid-game node
    // with wonders still to build, which is where an MCTS search spends most
    // of its time and where the prior has the most work to do.
    let (state, legal) = mid_game(2024, 30, 18);
    let player = state.current_player();
    let precomputed = stance(&state, player);

    println!(
        "benchmark position: age {}, {} legal actions, {} cards left",
        state.age(),
        legal.len(),
        Board::of(&state).cards_left()
    );

    c.bench_function("stance_plus_all_priors", |b| {
        b.iter(|| {
            let s = stance(black_box(&state), player);
            let mut total = 0.0;
            for &a in &legal {
                total += action_prior(&state, a, &s);
            }
            black_box(total)
        })
    });

    c.bench_function("all_priors_precomputed_stance", |b| {
        b.iter(|| {
            let mut total = 0.0;
            for &a in &legal {
                total += action_prior(black_box(&state), a, &precomputed);
            }
            black_box(total)
        })
    });

    c.bench_function("action_prior_one", |b| {
        let a = legal[0];
        b.iter(|| black_box(action_prior(&state, a, &precomputed)))
    });

    c.bench_function("stance_only", |b| {
        b.iter(|| black_box(stance(black_box(&state), player)))
    });

    c.bench_function("board_of", |b| {
        b.iter(|| black_box(Board::of(black_box(&state))))
    });

    c.bench_function("military_read", |b| {
        b.iter(|| black_box(military_read(black_box(&state), player)))
    });

    c.bench_function("science_read", |b| {
        b.iter(|| black_box(science_read(black_box(&state), player)))
    });

    c.bench_function("vp_read", |b| {
        b.iter(|| black_box(vp_read(black_box(&state), player)))
    });

    c.bench_function("mcts_rollout_reference", |b| {
        let mut rng = StdRng::seed_from_u64(0x0110_7011);
        b.iter(|| black_box(rollout(state, &mut rng)))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
