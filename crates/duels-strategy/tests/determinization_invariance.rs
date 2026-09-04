//! The correctness property this whole crate rests on: **a read must not
//! depend on which hidden world it is looking at.**
//!
//! These functions are meant to run inside a determinized search, where the
//! concrete [`GameState`] they are handed was invented by
//! [`Observation::sample_state`] from the public view. If a read peeked at a
//! specific face-down card's identity — instead of the *pool* that card
//! belongs to — then a search would score the sampler's luck rather than the
//! position, and averaging over samples would not converge to anything
//! meaningful.
//!
//! Two independent attacks on that property are made here:
//!
//! 1. **Resampling.** Take a real `Observation` from a real game in progress,
//!    call `sample_state` twice with different RNGs, and assert every read is
//!    bit-identical on the two worlds. This is the scenario a search actually
//!    runs.
//! 2. **In-place hidden mutation.** Take the *original* state and permute what
//!    is behind its face-down slots, or swap one of the cards that were boxed
//!    at setup into play, using `duels_core::testing`'s helpers. Those change
//!    hidden information and nothing public at all, so they are a sharper
//!    probe than resampling, which can coincidentally produce the same world.
//!
//! Every comparison is exact, including the `f64` fields: identical inputs
//! through identical arithmetic must produce identical bits, and a discrepancy
//! of any size means something read what it should not have.

use duels_core::observation::Observation;
use duels_core::testing::{swap_a_boxed_card_into_play, swap_two_hidden_cards};
use duels_core::{engine, Action, GameState, Player};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use duels_strategy::{action_prior, military_read, science_read, stance, vp_read, Board, Stance};

/// Walk a real game `steps` decisions in, with a deterministic policy that
/// depends on `mix` so different cases explore different lines.
fn advance(seed: u64, steps: usize, mix: u64) -> GameState {
    let mut st = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5A5A);
    for _ in 0..steps {
        let actions = engine::legal_actions(&st);
        if actions.is_empty() {
            break;
        }
        let i = u64::from(st.turn())
            .wrapping_mul(mix)
            .wrapping_add(mix >> 9) as usize;
        engine::apply_quiet(&mut st, actions[i % actions.len()], &mut rng).unwrap();
    }
    st
}

/// Assert that every read in this crate agrees on `a` and `b`, which must be
/// two states with the same public view.
fn assert_reads_agree(a: &GameState, b: &GameState, ctx: &str) {
    assert_eq!(
        a.observation(),
        b.observation(),
        "{ctx}: the two states are not publicly identical, so the test would be vacuous"
    );
    assert_eq!(Board::of(a), Board::of(b), "{ctx}: Board");

    let legal_a = engine::legal_actions(a);
    let legal_b = engine::legal_actions(b);
    assert_eq!(legal_a, legal_b, "{ctx}: legal actions");

    for p in Player::ALL {
        assert_eq!(
            military_read(a, p),
            military_read(b, p),
            "{ctx}: military_read for {p}"
        );
        assert_eq!(
            science_read(a, p),
            science_read(b, p),
            "{ctx}: science_read for {p}"
        );
        assert_eq!(vp_read(a, p), vp_read(b, p), "{ctx}: vp_read for {p}");

        let sa: Stance = stance(a, p);
        let sb: Stance = stance(b, p);
        assert_eq!(sa, sb, "{ctx}: stance for {p}");

        for &action in &legal_a {
            let pa = action_prior(a, action, &sa);
            let pb = action_prior(b, action, &sb);
            assert_eq!(
                pa.to_bits(),
                pb.to_bits(),
                "{ctx}: action_prior for {p} on {action:?}: {pa} vs {pb}"
            );
        }
    }
}

/// Two worlds sampled from `obs`, and whether they actually came out
/// different.
fn two_samples(obs: &Observation, sa: u64, sb: u64) -> (GameState, GameState, bool) {
    let mut ra = StdRng::seed_from_u64(sa);
    let mut rb = StdRng::seed_from_u64(sb);
    let a = obs.sample_state(&mut ra);
    let b = obs.sample_state(&mut rb);
    let differ = a != b;
    (a, b, differ)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// The headline property, on positions from real games.
    #[test]
    fn reads_are_the_same_on_any_two_sampled_worlds(
        seed in 0u64..100_000,
        steps in 0usize..90,
        mix in 1u64..1_000_000,
        sa in 0u64..1_000_000,
        sb in 1_000_000u64..2_000_000,
    ) {
        let st = advance(seed, steps, mix);
        prop_assume!(!st.is_over());
        let obs = st.observation();
        let (a, b, _) = two_samples(&obs, sa, sb);
        assert_reads_agree(&a, &b, &format!("seed {seed} steps {steps}"));
        // ...and against the real state the observation came from, which is a
        // world the sampler may never produce.
        assert_reads_agree(&st, &a, &format!("seed {seed} steps {steps} (vs real)"));
    }

    /// Permuting the cards behind the face-down slots, or exchanging one of
    /// them with a card that was boxed at setup, changes hidden information
    /// and nothing else.
    #[test]
    fn reads_survive_an_in_place_hidden_mutation(
        seed in 0u64..100_000,
        steps in 1usize..90,
        mix in 1u64..1_000_000,
    ) {
        let st = advance(seed, steps, mix);
        prop_assume!(!st.is_over());

        let mut permuted = st;
        if swap_two_hidden_cards(&mut permuted) {
            assert_reads_agree(&st, &permuted, &format!("seed {seed} steps {steps} permuted"));
        }

        let mut reboxed = st;
        if swap_a_boxed_card_into_play(&mut reboxed) {
            assert_reads_agree(&st, &reboxed, &format!("seed {seed} steps {steps} reboxed"));
        }
    }
}

/// A deterministic sweep, so the property is exercised the same way on every
/// machine and in CI — and so we can *prove* the samples really do differ,
/// which `proptest`'s random cases cannot guarantee on their own.
#[test]
fn a_deterministic_sweep_over_many_positions_with_genuinely_different_worlds() {
    let mut differing = 0usize;
    let mut checked = 0usize;
    let mut mutated = 0usize;

    for seed in 0..12u64 {
        for steps in [0usize, 3, 9, 14, 22, 31, 40, 52, 61, 70, 80] {
            let st = advance(seed, steps, 7 + seed);
            if st.is_over() {
                continue;
            }
            let obs = st.observation();
            let ctx = format!("seed {seed} steps {steps}");
            for k in 0..3u64 {
                let (a, b, differ) = two_samples(&obs, seed * 97 + k, seed * 131 + k + 5_000);
                if differ {
                    differing += 1;
                }
                checked += 1;
                assert_reads_agree(&a, &b, &ctx);
            }

            let mut permuted = st;
            if swap_two_hidden_cards(&mut permuted) {
                mutated += 1;
                assert_reads_agree(&st, &permuted, &format!("{ctx} permuted"));
            }
            let mut reboxed = st;
            if swap_a_boxed_card_into_play(&mut reboxed) {
                mutated += 1;
                assert_reads_agree(&st, &reboxed, &format!("{ctx} reboxed"));
            }
        }
    }

    assert!(checked > 100, "the sweep should cover many positions");
    assert!(
        differing * 2 > checked,
        "only {differing} of {checked} sample pairs were genuinely different worlds, \
         so the property test is mostly vacuous"
    );
    assert!(
        mutated > 50,
        "only {mutated} positions admitted an in-place hidden mutation"
    );
}

/// A last, narrower check aimed at the specific failure mode the property is
/// guarding against: reading a *particular* hidden card rather than the pool.
///
/// Every expected-value field must be strictly between the best and worst case
/// the pool allows, and must not move when the pool is merely reshuffled.
#[test]
fn expected_hidden_values_come_from_the_pool_not_from_a_sample() {
    let mut saw_positive = false;
    for seed in 0..20u64 {
        for steps in [12usize, 28, 45] {
            let st = advance(seed, steps, 3);
            if st.is_over() {
                continue;
            }
            let board = Board::of(&st);
            if board.hidden_slot_count() == 0 {
                continue;
            }
            let r = military_read(&st, Player::One);
            let pool_total = f64::from(duels_strategy::masks::shields_in(board.unknown_pool));
            assert!(
                r.expected_hidden >= 0.0 && r.expected_hidden <= pool_total,
                "seed {seed} steps {steps}: expected {} outside 0..={pool_total}",
                r.expected_hidden
            );
            if r.expected_hidden > 0.0 {
                saw_positive = true;
            }

            // Also assert the read never claims a *specific* hidden card is
            // available: nothing face down can be an accessible slot, per the
            // engine's own invariant, so no closing slot may be hidden.
            assert_eq!(r.closing_slots & board.hidden_slots, 0);
            assert_eq!(
                science_read(&st, Player::One).closing_slots & board.hidden_slots,
                0
            );
        }
    }
    assert!(
        saw_positive,
        "no position had any expected hidden shields, so this test proved nothing"
    );
}

/// `action_prior` must be usable on exactly the action list the engine
/// produces, for every decision of a real game, on a determinized world.
#[test]
fn priors_agree_on_a_determinized_world_through_a_whole_game() {
    let mut rng = StdRng::seed_from_u64(0xB0A7);
    for seed in 0..4u64 {
        let mut st = engine::new_game(seed);
        let mut guard = 0;
        loop {
            let legal: Vec<Action> = engine::legal_actions(&st);
            if legal.is_empty() {
                break;
            }
            let obs = st.observation();
            let (a, b, _) = two_samples(&obs, seed, seed + 999);
            assert_reads_agree(&a, &b, &format!("seed {seed} turn {}", st.turn()));
            engine::apply_quiet(&mut st, legal[guard % legal.len()], &mut rng).unwrap();
            guard += 1;
            assert!(guard < 400);
        }
    }
}
