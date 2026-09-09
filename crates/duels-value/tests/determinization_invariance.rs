//! The mandatory correctness property for anything in this repository that
//! touches game state: **nothing may depend on which hidden world it is
//! looking at.**
//!
//! A search hands this crate a concrete `GameState` that `Observation::sample_state`
//! invented from public knowledge, and the learned value must score the
//! *position*, not the sampler's luck. Two independent attacks, matching
//! `duels-eval`'s and `duels-strategy`'s tests of the same name:
//!
//! 1. **Resampling.** Draw two unrelated concrete worlds from the same real
//!    observation and assert the feature vector and the prediction agree bit
//!    for bit.
//! 2. **In-place hidden mutation.** Permute what sits behind the face-down
//!    slots of the original state, or swap a boxed card into play, with
//!    `duels_core::testing`'s helpers — hidden information changes, public
//!    information does not.
//!
//! Every comparison is exact, `f32::to_bits` included: identical inputs
//! through identical arithmetic produce identical bits, and any difference
//! means a feature read something it should not have.

use duels_core::testing::{swap_a_boxed_card_into_play, swap_two_hidden_cards};
use duels_core::{engine, GameState, Player};
use duels_value::{features, predict, win_probability, NUM_FEATURES};
use rand::rngs::StdRng;
use rand::SeedableRng;

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

fn assert_everything_agrees(a: &GameState, b: &GameState, ctx: &str) {
    assert_eq!(
        a.observation(),
        b.observation(),
        "{ctx}: the two states are not publicly identical, so the test would be vacuous"
    );
    for me in Player::ALL {
        let fa = features(a, me);
        let fb = features(b, me);
        for i in 0..NUM_FEATURES {
            assert_eq!(
                fa[i].to_bits(),
                fb[i].to_bits(),
                "{ctx}: feature {i} ({}) for {me:?}: {} vs {}",
                duels_value::feature_names()[i],
                fa[i],
                fb[i]
            );
        }
        let (da, db) = (predict(a, me), predict(b, me));
        for (x, y) in da.as_array().iter().zip(db.as_array()) {
            assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: prediction for {me:?}");
        }
        assert_eq!(
            win_probability(a, me).to_bits(),
            win_probability(b, me).to_bits(),
            "{ctx}: win probability for {me:?}"
        );
    }
}

#[test]
fn resampling_the_same_observation_changes_nothing() {
    let mut cases = 0usize;
    let mut differing_worlds = 0usize;
    for seed in 0..16u64 {
        for &steps in &[0usize, 4, 11, 19, 27, 36, 45, 58] {
            let st = advance(seed, steps, 0x9E37_79B9_7F4A_7C15);
            if st.is_over() {
                continue;
            }
            let obs = st.observation();
            let mut rng_a = StdRng::seed_from_u64(seed * 7919 + steps as u64);
            let mut rng_b = StdRng::seed_from_u64(0xFFFF_0000_FFFF_0000 ^ seed ^ steps as u64);
            let a = obs.sample_state(&mut rng_a);
            let b = obs.sample_state(&mut rng_b);
            if a != b {
                differing_worlds += 1;
            }
            assert_everything_agrees(&a, &b, &format!("seed {seed} steps {steps}"));
            cases += 1;
        }
    }
    assert!(cases > 40, "only {cases} positions exercised");
    assert!(
        differing_worlds > 20,
        "the samples were too alike ({differing_worlds}) to prove anything"
    );
}

#[test]
fn permuting_the_hidden_cards_in_place_changes_nothing() {
    let mut permuted = 0usize;
    let mut boxed = 0usize;
    for seed in 0..16u64 {
        for &steps in &[6usize, 15, 24, 33, 42, 55] {
            let base = advance(seed, steps, 0xD1B5_4A32_D192_ED03);
            if base.is_over() {
                continue;
            }
            let mut shuffled = base;
            if swap_two_hidden_cards(&mut shuffled) {
                assert_everything_agrees(
                    &base,
                    &shuffled,
                    &format!("seed {seed} steps {steps} (hidden swap)"),
                );
                permuted += 1;
            }
            let mut smuggled = base;
            if swap_a_boxed_card_into_play(&mut smuggled) {
                assert_everything_agrees(
                    &base,
                    &smuggled,
                    &format!("seed {seed} steps {steps} (boxed card)"),
                );
                boxed += 1;
            }
        }
    }
    assert!(permuted > 0, "never managed to permute two hidden cards");
    assert!(boxed > 0, "never managed to swap a boxed card into play");
}

/// The sharpest case: a position whose *next age* is entirely the sampler's
/// invention. Two samples deal two different Age IIs; the features of the
/// position *before* the age turns must not see either.
#[test]
fn the_undealt_next_age_is_invisible() {
    let mut compared = 0usize;
    for seed in 0..24u64 {
        // Walk until one card is left in Age I, if this seed's line allows.
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed);
        while st.age() == 1 && st.occupied_slots().count_ones() != 1 && !st.is_over() {
            let legal = engine::legal_actions(&st);
            if legal.is_empty() {
                break;
            }
            let a = legal[st.turn() as usize % legal.len()];
            engine::apply_quiet(&mut st, a, &mut rng).unwrap();
        }
        if st.age() != 1 || st.occupied_slots().count_ones() != 1 {
            continue;
        }
        let obs = st.observation();
        let mut rng_a = StdRng::seed_from_u64(seed ^ 0x1111);
        let mut rng_b = StdRng::seed_from_u64(seed ^ 0x2222_2222);
        let a = obs.sample_state(&mut rng_a);
        let b = obs.sample_state(&mut rng_b);
        // Vacuity guard: the two worlds really do deal different Age IIs.
        // Taking the last card (any action on it ends the age) shows it.
        let action = engine::legal_actions(&a)[0];
        let outcomes = engine::chance_outcomes(&a, action);
        let (mut after_a, mut after_b) = (a, b);
        engine::apply_with_outcome(&mut after_a, action, &outcomes[0].0).unwrap();
        engine::apply_with_outcome(&mut after_b, action, &outcomes[0].0).unwrap();
        if after_a.age() != 2 {
            continue;
        }
        let structure = |s: &GameState| -> Vec<Option<duels_core::data::CardId>> {
            (0..20u8).map(|i| s.face_up_card(i)).collect()
        };
        if structure(&after_a) == structure(&after_b) {
            continue;
        }
        assert_everything_agrees(&a, &b, &format!("seed {seed} (last card of Age I)"));
        compared += 1;
    }
    assert!(compared > 3, "only {compared} last-card positions found");
}
