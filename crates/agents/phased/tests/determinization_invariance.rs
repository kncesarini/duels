//! The mandatory correctness property for anything in this repository that
//! touches game state: **nothing may depend on which hidden world it is
//! looking at.**
//!
//! [`PhasedAgent::choose`] is handed an [`Observation`], which carries no
//! hidden information, and immediately invents one concrete [`GameState`] from
//! it — the engine's chance API needs a real state to work on. That invented
//! world assigns a specific identity to every face-down card. If any part of
//! the commitment blend, any weight, or any term read one of those identities,
//! the agent would be scoring the sampler's luck rather than the position, and
//! two runs of the same decision could disagree for no reason a player could
//! see.
//!
//! Two independent attacks, matching `duels-strategy`'s own
//! `tests/determinization_invariance.rs`:
//!
//! 1. **Resampling.** Draw two unrelated concrete worlds from the same real
//!    observation and assert everything agrees bit for bit.
//! 2. **In-place hidden mutation.** Permute what sits behind the face-down
//!    slots of the *original* state, or swap one of the cards boxed at setup
//!    into play, with `duels_core::testing`'s helpers. Those change hidden
//!    information and nothing public at all, so they are a sharper probe than
//!    resampling, which can coincidentally draw the same world twice.
//!
//! Every comparison is exact, including on `f64`: identical inputs through
//! identical arithmetic produce identical bits, and a discrepancy of any size
//! means something read what it should not have.

use duels_agent_phased::{evaluate, expected_value, Blend, Config, PhasedAgent, Root};
use duels_agents_api::{Agent, Budget};
use duels_core::observation::Observation;
use duels_core::testing::{swap_a_boxed_card_into_play, swap_two_hidden_cards};
use duels_core::{engine, GameState, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Two `f64`s from identical inputs through identical arithmetic must be
/// identical *bit for bit*. Deliberately not an epsilon comparison.
#[track_caller]
fn same_bits(a: f64, b: f64, what: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{what}: {a} vs {b} differ in bits"
    );
}

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
        let action = actions[i % actions.len()];
        engine::apply_quiet(&mut st, action, &mut rng).unwrap();
    }
    st
}

/// Assert that every commitment scalar, every root-fixed weight, and the full
/// expected value of every legal action agree on `a` and `b`, which must be
/// two states with the same public view.
fn assert_everything_agrees(a: &GameState, b: &GameState, ctx: &str, config: Config) {
    assert_eq!(
        a.observation(),
        b.observation(),
        "{ctx}: the two states are not publicly identical, so the test would be vacuous"
    );

    let me = a.current_player();
    let root_a = Root::new(a, me, config);
    let root_b = Root::new(b, me, config);

    for p in Player::ALL {
        let (ca, cb) = (root_a.commitment(p), root_b.commitment(p));
        same_bits(ca.c_sci, cb.c_sci, &format!("{ctx}: c_sci for {p:?}"));
        same_bits(ca.c_mil, cb.c_mil, &format!("{ctx}: c_mil for {p:?}"));
        same_bits(ca.c, cb.c, &format!("{ctx}: c for {p:?}"));
        same_bits(ca.c0_eff, cb.c0_eff, &format!("{ctx}: c0_eff for {p:?}"));
        same_bits(ca.s, cb.s, &format!("{ctx}: S(c) for {p:?}"));
        same_bits(ca.s_sci, cb.s_sci, &format!("{ctx}: S(c_sci) for {p:?}"));
        same_bits(ca.s_mil, cb.s_mil, &format!("{ctx}: S(c_mil) for {p:?}"));

        let (wa, wb) = (root_a.weights(p), root_b.weights(p));
        for (name, x, y) in [
            ("vp", wa.vp, wb.vp),
            ("liquidity", wa.liquidity, wb.liquidity),
            ("development", wa.development, wb.development),
            ("science", wa.science, wb.science),
            ("military", wa.military, wb.military),
            ("race_liquidity", wa.race_liquidity, wb.race_liquidity),
            ("economy", wa.economy, wb.economy),
        ] {
            same_bits(x, y, &format!("{ctx}: {name} weight for {p:?}"));
        }
    }
    same_bits(
        root_a.deny_scale(),
        root_b.deny_scale(),
        &format!("{ctx}: denial scale"),
    );
    assert_eq!(
        root_a.supply(),
        root_b.supply(),
        "{ctx}: the development supply statistics disagree"
    );

    // The static evaluation of the two worlds themselves...
    for p in Player::ALL {
        same_bits(
            evaluate(a, p, &root_a),
            evaluate(b, p, &root_b),
            &format!("{ctx}: evaluate for {p:?}"),
        );
    }

    // ...and, the thing that actually decides a move, the expected value of
    // every legal action, chance outcomes and all.
    let legal_a = engine::legal_actions(a);
    let legal_b = engine::legal_actions(b);
    assert_eq!(legal_a, legal_b, "{ctx}: legal actions");
    for &action in &legal_a {
        same_bits(
            expected_value(a, action, me, &root_a),
            expected_value(b, action, me, &root_b),
            &format!("{ctx}: expected value of {action:?}"),
        );
        same_bits(
            root_a.denial_term(action),
            root_b.denial_term(action),
            &format!("{ctx}: denial term of {action:?}"),
        );
    }
}

#[test]
fn resampling_the_same_observation_changes_nothing() {
    let mut cases = 0usize;
    let mut multi_outcome_cases = 0usize;
    for seed in 0..12u64 {
        for &steps in &[4usize, 11, 19, 27, 36, 45] {
            let st = advance(seed, steps, 0x9E37_79B9_7F4A_7C15);
            if st.is_over() {
                continue;
            }
            let obs: Observation = st.observation();
            let mut rng_a = StdRng::seed_from_u64(seed * 7919 + steps as u64);
            let mut rng_b = StdRng::seed_from_u64(0xFFFF_0000_FFFF_0000 ^ seed ^ steps as u64);
            let a = obs.sample_state(&mut rng_a);
            let b = obs.sample_state(&mut rng_b);

            if engine::legal_actions(&a)
                .iter()
                .any(|&x| engine::chance_outcomes(&a, x).len() > 1)
            {
                multi_outcome_cases += 1;
            }
            assert_everything_agrees(
                &a,
                &b,
                &format!("seed {seed} steps {steps}"),
                Config::default(),
            );
            cases += 1;
        }
    }
    assert!(cases > 20, "only {cases} positions exercised");
    assert!(
        multi_outcome_cases > 0,
        "no position offered an action with a real chance node, so the \
         expectation half of the property was never tested"
    );
}

#[test]
fn permuting_the_hidden_cards_in_place_changes_nothing() {
    let mut permuted = 0usize;
    let mut boxed = 0usize;
    for seed in 0..12u64 {
        for &steps in &[6usize, 15, 24, 33, 42] {
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
                    Config::default(),
                );
                permuted += 1;
            }

            let mut smuggled = base;
            if swap_a_boxed_card_into_play(&mut smuggled) {
                assert_everything_agrees(
                    &base,
                    &smuggled,
                    &format!("seed {seed} steps {steps} (boxed card)"),
                    Config::default(),
                );
                boxed += 1;
            }
        }
    }
    assert!(permuted > 0, "never managed to permute two hidden cards");
    assert!(boxed > 0, "never managed to swap a boxed card into play");
}

/// The property has to hold with the blend switched off too — otherwise the
/// un-blended baseline the bit-identity test rests on would itself be
/// sample-dependent.
#[test]
fn the_property_holds_with_the_blend_switched_off() {
    let config = Config {
        blend: Blend::off(),
        ..Config::default()
    };
    for seed in 0..6u64 {
        for &steps in &[9usize, 21, 38] {
            let st = advance(seed, steps, 0x2545_F491_4F6C_DD1D);
            if st.is_over() {
                continue;
            }
            let obs = st.observation();
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut rng_b = StdRng::seed_from_u64(!seed);
            let a = obs.sample_state(&mut rng_a);
            let b = obs.sample_state(&mut rng_b);
            assert_everything_agrees(
                &a,
                &b,
                &format!("blend off, seed {seed} steps {steps}"),
                config,
            );
        }
    }
}

/// The whole-agent restatement: two agents whose internal `sample_state`
/// draws differ must score every candidate identically. The action each one
/// finally returns may still differ, because ties are broken from each
/// agent's own RNG — it is the scores feeding that choice which must not
/// depend on the sample.
#[test]
fn two_differently_seeded_agents_score_every_candidate_identically() {
    let st = advance(11, 18, 0x9E37_79B9_7F4A_7C15);
    assert!(!st.is_over());
    let obs = st.observation();
    let legal = engine::legal_actions(&st);
    assert!(legal.len() > 1, "test setup: need a real choice");
    let me = obs.current_player;

    let mut rng_a = StdRng::seed_from_u64(0x1111);
    let mut rng_b = StdRng::seed_from_u64(0x2222_2222_2222);
    let a = obs.sample_state(&mut rng_a);
    let b = obs.sample_state(&mut rng_b);
    let root_a = Root::new(&a, me, Config::default());
    let root_b = Root::new(&b, me, Config::default());

    for &action in &legal {
        same_bits(
            expected_value(&a, action, me, &root_a),
            expected_value(&b, action, me, &root_b),
            &format!("candidate {action:?}"),
        );
    }

    // ...and the agent itself, differently seeded, still returns something
    // legal every time.
    for seed in 0..8u64 {
        let mut agent = PhasedAgent::new(seed);
        assert!(legal.contains(&agent.choose(&obs, &legal, Budget::Nodes(1))));
    }
}
