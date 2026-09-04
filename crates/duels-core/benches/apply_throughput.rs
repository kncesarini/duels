//! Throughput micro-benchmarks for the hot engine paths.
//!
//! A search-based agent's inner loop is `legal_actions` + `apply`, so those
//! are what matter. Run with:
//!
//! ```text
//! cargo bench -p duels-core
//! ```
//!
//! The numbers observed on the M1 development machine (Apple Silicon,
//! `cargo bench`, release) are recorded in `docs/rules-spec.md`; treat them
//! as a reference point, not a CI gate — this benchmark deliberately asserts
//! nothing, because a threshold that passes on a laptop is flaky on shared CI
//! runners.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use duels_core::action::Action;
use duels_core::{engine, GameState};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// One full game played with a cheap deterministic policy.
fn playout(seed: u64) -> u32 {
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9e37);
    let mut buf: Vec<Action> = Vec::with_capacity(64);
    loop {
        engine::legal_actions_into(&state, &mut buf);
        if buf.is_empty() {
            break;
        }
        let a = buf[(state.turn() as usize) % buf.len()];
        engine::apply_unchecked(&mut state, a, &mut rng);
    }
    state.turn()
}

/// A mid-game position with plenty of legal actions, for per-apply timing.
fn mid_game() -> (GameState, StdRng) {
    let mut state = engine::new_game(4242);
    let mut rng = StdRng::seed_from_u64(4242);
    for _ in 0..20 {
        let actions = engine::legal_actions(&state);
        if actions.is_empty() {
            break;
        }
        engine::apply(&mut state, actions[actions.len() / 2], &mut rng).unwrap();
    }
    (state, rng)
}

fn bench(c: &mut Criterion) {
    c.bench_function("full_playout", |b| {
        let mut seed = 0u64;
        b.iter(|| {
            seed = seed.wrapping_add(1);
            black_box(playout(seed))
        })
    });

    let (state, mut rng) = mid_game();

    c.bench_function("legal_actions", |b| {
        let mut buf = Vec::with_capacity(64);
        b.iter(|| {
            engine::legal_actions_into(black_box(&state), &mut buf);
            black_box(buf.len())
        })
    });

    c.bench_function("copy_state", |b| {
        b.iter(|| {
            let copy: GameState = *black_box(&state);
            black_box(copy.turn())
        })
    });

    c.bench_function("apply_unchecked", |b| {
        let action = engine::legal_actions(&state)[0];
        b.iter(|| {
            let mut s = state;
            engine::apply_unchecked(&mut s, black_box(action), &mut rng);
            black_box(s.turn())
        })
    });

    c.bench_function("apply_validated", |b| {
        let action = engine::legal_actions(&state)[0];
        b.iter(|| {
            let mut s = state;
            engine::apply_quiet(&mut s, black_box(action), &mut rng).unwrap();
            black_box(s.turn())
        })
    });

    c.bench_function("observation", |b| {
        b.iter(|| black_box(black_box(&state).observation()))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
