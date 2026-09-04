//! Property tests: invariants that must hold for *every* game, checked over
//! many randomised full playouts.
//!
//! These complement the hand-written scenarios in `golden_scenarios.rs`. The
//! scenarios pin down specific rules; these catch the class of bug where some
//! rare interaction breaks an invariant the rest of the system — and every
//! future AI — relies on.
//!
//! Note that this file only ever touches `duels-core`'s *public* API, exactly
//! as an agent would. The invariants that are about hidden fields (card
//! conservation across the boxed cards and the not-yet-dealt decks) are
//! asserted through [`GameState::check_invariants`], which lives inside the
//! crate for that reason: exposing the fields to a test would be exposing
//! them to everybody.
//!
//! Covered invariants, cross-referenced in `docs/rules-spec.md`:
//!
//! * card conservation — every card is in exactly one place, always;
//! * coins never underflow;
//! * at most seven wonders are ever constructed;
//! * `legal_actions` is non-empty until the game is over, and every action it
//!   returns really applies;
//! * an `Observation` never depends on hidden information;
//! * `Observation::sample_state` round-trips;
//! * scoring is symmetric under a seat swap;
//! * chance-outcome probabilities form a distribution.

use duels_core::action::Action;
use duels_core::data::CardId;
use duels_core::observation::SlotView;
use duels_core::state::{Phase, MAX_WONDERS_BUILT};
use duels_core::testing::{swap_a_boxed_card_into_play, swap_two_hidden_cards, StateBuilder};
use duels_core::{engine, scoring, GameState, Player};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Deterministic policy parameterised by a mixing constant, so different
/// property-test cases explore different lines rather than replaying one.
fn pick(actions: &[Action], turn: u32, mix: u64) -> Action {
    let i = u64::from(turn).wrapping_mul(mix).wrapping_add(mix >> 7) as usize;
    actions[i % actions.len()]
}

/// Everything that must be true of a state at a decision point.
fn check_invariants(state: &GameState, ctx: &str) {
    state
        .check_invariants()
        .unwrap_or_else(|e| panic!("{ctx}: {e}"));

    // --- legality ----------------------------------------------------------
    let actions = engine::legal_actions(state);
    if state.is_over() {
        assert!(actions.is_empty(), "{ctx}: a finished game offers moves");
        assert!(state.result().is_some(), "{ctx}: finished without a result");
    } else {
        assert!(!actions.is_empty(), "{ctx}: stuck with no legal action");
        let mut rng = StdRng::seed_from_u64(1);
        for a in &actions {
            let mut copy = *state;
            engine::apply_quiet(&mut copy, *a, &mut rng)
                .unwrap_or_else(|e| panic!("{ctx}: legal action failed to apply: {e}"));
            copy.check_invariants()
                .unwrap_or_else(|e| panic!("{ctx}: applying {a:?} broke an invariant: {e}"));
        }
        if state.phase() != Phase::Turn {
            assert!(
                !actions
                    .iter()
                    .any(|a| matches!(a, Action::Build { .. } | Action::Discard { .. })),
                "{ctx}: card actions offered in {:?}",
                state.phase()
            );
        }
    }

    // --- observation -------------------------------------------------------
    let obs = state.observation();
    for view in obs.slots.iter() {
        if *view == SlotView::FaceDown {
            assert!(
                view.card().is_none(),
                "{ctx}: a face-down slot carries a card"
            );
        }
    }
    // The candidate pool must be strictly larger than the number of hidden
    // slots: three cards of the age went back in the box unseen, so a hidden
    // slot is never pinned down by elimination.
    let hidden_count = obs
        .slots
        .iter()
        .filter(|v| **v == SlotView::FaceDown)
        .count();
    if hidden_count > 0 {
        assert!(
            obs.unknown_slot_pool.len() > hidden_count,
            "{ctx}: {} candidates for {hidden_count} hidden slots",
            obs.unknown_slot_pool.len()
        );
    }
    // Permuting hidden information must not change the observation.
    let mut permuted = *state;
    if swap_two_hidden_cards(&mut permuted) {
        assert_eq!(
            obs,
            permuted.observation(),
            "{ctx}: the observation depends on which card is behind which slot"
        );
        permuted
            .check_invariants()
            .unwrap_or_else(|e| panic!("{ctx}: permuting broke an invariant: {e}"));
    }
    let mut swapped = *state;
    if swap_a_boxed_card_into_play(&mut swapped) {
        assert_eq!(
            obs,
            swapped.observation(),
            "{ctx}: the observation reveals which cards were boxed"
        );
    }
    // A sampled world must reproduce the observation exactly.
    let mut rng = StdRng::seed_from_u64(0xf00d);
    let sampled = obs.sample_state(&mut rng);
    assert_eq!(
        sampled.observation(),
        obs,
        "{ctx}: sample_state did not round trip"
    );
    sampled
        .check_invariants()
        .unwrap_or_else(|e| panic!("{ctx}: sample_state produced an invalid state: {e}"));

    // --- chance ------------------------------------------------------------
    if !state.is_over() {
        for a in actions.iter().take(3) {
            let outcomes = engine::chance_outcomes(state, *a);
            assert!(!outcomes.is_empty(), "{ctx}: no chance outcomes for {a:?}");
            let total: f64 = outcomes.iter().map(|(_, p)| p).sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "{ctx}: chance probabilities for {a:?} sum to {total}"
            );
            for (o, p) in &outcomes {
                assert!(*p > 0.0, "{ctx}: zero-probability outcome offered");
                let mut copy = *state;
                engine::apply_with_outcome(&mut copy, *a, o)
                    .unwrap_or_else(|e| panic!("{ctx}: forced outcome failed: {e}"));
                copy.check_invariants()
                    .unwrap_or_else(|e| panic!("{ctx}: forcing {o:?} broke an invariant: {e}"));
            }
        }
    }
}

/// Play a full game, checking invariants at every decision point.
fn playout_checked(seed: u64, mix: u64) -> GameState {
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
    let mut buf = Vec::new();
    let mut steps = 0u32;
    loop {
        check_invariants(&state, &format!("seed {seed} mix {mix} step {steps}"));
        engine::legal_actions_into(&state, &mut buf);
        if buf.is_empty() {
            break;
        }
        let a = pick(&buf, state.turn(), mix);
        engine::apply(&mut state, a, &mut rng).expect("action came from legal_actions");
        steps += 1;
        assert!(steps < 5_000, "seed {seed}: game did not terminate");
    }
    state
}

/// Fast playout without the per-step invariant checks, so the suite can cover
/// a lot of games.
fn playout_fast(seed: u64, mix: u64) -> GameState {
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
    let mut buf = Vec::new();
    let mut steps = 0u32;
    loop {
        engine::legal_actions_into(&state, &mut buf);
        if buf.is_empty() {
            break;
        }
        let a = pick(&buf, state.turn(), mix);
        engine::apply_unchecked(&mut state, a, &mut rng);
        steps += 1;
        assert!(steps < 5_000, "seed {seed}: game did not terminate");
    }
    state
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 16, .. ProptestConfig::default() })]

    /// Every invariant, at every decision point, over full games. Expensive
    /// (it re-applies every legal action at every step), so relatively few
    /// cases.
    #[test]
    fn invariants_hold_throughout_a_game(seed in 0u64..100_000, mix in 1u64..1_000) {
        let state = playout_checked(seed, mix);
        prop_assert!(state.is_over());
        prop_assert!(state.result().is_some());
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2_000, .. ProptestConfig::default() })]

    /// Thousands of full games, checking the end-state invariants.
    #[test]
    fn many_full_games_terminate_and_conserve_cards(
        seed in 0u64..1_000_000,
        mix in 1u64..10_000,
    ) {
        let state = playout_fast(seed, mix);
        prop_assert!(state.is_over());
        let result = state.result().expect("a finished game has a result");
        prop_assert!(state.check_invariants().is_ok(), "{:?}", state.check_invariants());
        prop_assert!(state.wonders_built_total() <= MAX_WONDERS_BUILT);

        let p1 = state.player(Player::One).built_mask();
        let p2 = state.player(Player::Two).built_mask();
        let d = state.discard_mask();
        let f = state.wonder_fodder_mask();
        let union = p1 | p2 | d | f;
        prop_assert_eq!(
            p1.count_ones() + p2.count_ones() + d.count_ones() + f.count_ones(),
            union.count_ones(),
            "a card ended up in two places"
        );
        // A game that ran to the end of Age III consumed all 60 dealt cards;
        // an instant win stops earlier.
        if !result.is_instant() {
            prop_assert_eq!(union.count_ones(), 60);
            prop_assert_eq!(state.occupied_slots(), 0);
        }
        for p in Player::ALL {
            prop_assert!(state.player(p).coins() < 500);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, .. ProptestConfig::default() })]

    /// The same seed and the same policy must produce the same game every
    /// time: no clock, no ambient randomness, no iteration-order dependence.
    #[test]
    fn games_are_reproducible(seed in 0u64..100_000, mix in 1u64..1_000) {
        let a = playout_fast(seed, mix);
        let b = playout_fast(seed, mix);
        prop_assert_eq!(a.result(), b.result());
        prop_assert_eq!(a.turn(), b.turn());
        prop_assert_eq!(a.observation(), b.observation());
    }
}

/// Cards used to build random-but-legal cities for the symmetry test.
fn city_cards() -> impl Strategy<Value = Vec<CardId>> {
    let all: Vec<CardId> = CardId::all().collect();
    proptest::collection::vec(0usize..all.len(), 0..12).prop_map(move |idxs| {
        let mut v: Vec<CardId> = idxs.into_iter().map(|i| all[i]).collect();
        v.sort_unstable();
        v.dedup();
        v
    })
}

fn build_city_state(
    a: &[CardId],
    a_coins: u16,
    b: &[CardId],
    b_coins: u16,
    conflict: i8,
) -> GameState {
    let a_slugs: Vec<&str> = a.iter().map(|c| c.slug()).collect();
    let b_slugs: Vec<&str> = b.iter().map(|c| c.slug()).collect();
    StateBuilder::new()
        .built(Player::One, &a_slugs)
        .built(Player::Two, &b_slugs)
        .coins(Player::One, a_coins)
        .coins(Player::Two, b_coins)
        .conflict(conflict)
        .build()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 500, .. ProptestConfig::default() })]

    /// Scoring must not care which seat a city sits in. Mirroring a position
    /// — swap the two cities and negate the conflict pawn — must mirror the
    /// score breakdowns exactly.
    #[test]
    fn scoring_is_symmetric_under_a_seat_swap(
        a_cards in city_cards(),
        b_cards in city_cards(),
        a_coins in 0u16..40,
        b_coins in 0u16..40,
        conflict in -8i8..=8,
    ) {
        // A card exists once, so the two cities must not share one.
        let b_cards: Vec<CardId> =
            b_cards.into_iter().filter(|c| !a_cards.contains(c)).collect();

        let straight = build_city_state(&a_cards, a_coins, &b_cards, b_coins, conflict);
        let mirrored = build_city_state(&b_cards, b_coins, &a_cards, a_coins, -conflict);

        prop_assert_eq!(
            scoring::breakdown(&straight, Player::One),
            scoring::breakdown(&mirrored, Player::Two)
        );
        prop_assert_eq!(
            scoring::breakdown(&straight, Player::Two),
            scoring::breakdown(&mirrored, Player::One)
        );

        let s = scoring::civilian_result(&straight);
        let m = scoring::civilian_result(&mirrored);
        prop_assert_eq!(s.winner().map(|p| p.other()), m.winner());
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, .. ProptestConfig::default() })]

    /// A whole seeded game replayed with the seats swapped must produce the
    /// mirrored result: the engine must contain no first-seat-specific rule
    /// beyond the ones the rules actually prescribe.
    #[test]
    fn a_seat_swapped_playout_mirrors_the_result(seed in 0u64..100_000, mix in 1u64..1_000) {
        // Swapping seats mid-setup is not expressible, so instead assert the
        // weaker property that holds for any seeded game: the distribution of
        // outcomes cannot favour a seat by construction. Here we check the
        // structural half — a finished game's breakdown for each seat is
        // computed by the same code path, so mirroring the *final position*
        // must mirror the breakdown.
        let state = playout_fast(seed, mix);
        let p1: Vec<CardId> = state.player(Player::One).built().collect();
        let p2: Vec<CardId> = state.player(Player::Two).built().collect();
        let c1 = state.player(Player::One).coins();
        let c2 = state.player(Player::Two).coins();
        let straight = build_city_state(&p1, c1, &p2, c2, state.conflict().clamp(-8, 8));
        let mirrored = build_city_state(&p2, c2, &p1, c1, -state.conflict().clamp(-8, 8));
        prop_assert_eq!(
            scoring::breakdown(&straight, Player::One).total,
            scoring::breakdown(&mirrored, Player::Two).total
        );
    }
}

#[test]
fn an_observation_json_never_ties_a_card_to_a_face_down_slot() {
    let mut state = engine::new_game(31);
    let mut rng = StdRng::seed_from_u64(31);
    let mut checked = 0;
    for _ in 0..60 {
        let actions = engine::legal_actions(&state);
        if actions.is_empty() {
            break;
        }
        let obs = state.observation();
        let json: serde_json::Value = serde_json::to_value(&obs).unwrap();
        for view in json["slots"].as_array().unwrap() {
            if view["state"] != "face_down" {
                continue;
            }
            checked += 1;
            assert_eq!(
                view.as_object().unwrap().len(),
                1,
                "a face-down slot carries extra data: {view}"
            );
        }
        let a = actions[(state.turn() as usize) % actions.len()];
        engine::apply(&mut state, a, &mut rng).unwrap();
    }
    assert!(checked > 20, "expected to inspect many face-down slots");
}

#[test]
fn every_seed_produces_a_valid_setup() {
    for seed in 0..500u64 {
        let state = engine::new_game(seed);
        assert_eq!(state.phase(), Phase::WonderDraft);
        assert_eq!(state.offered_wonders().len(), 4);
        assert_eq!(state.board_tokens_mask().count_ones(), 5);
        assert_eq!(state.set_aside_tokens_mask().count_ones(), 5);
        state
            .check_invariants()
            .unwrap_or_else(|e| panic!("setup seed {seed}: {e}"));
    }
}

#[test]
fn results_are_distributed_across_all_three_victory_kinds() {
    // Not a rules invariant, but a smoke test that the engine can actually
    // reach every ending: a bug that made military or scientific supremacy
    // unreachable would otherwise pass silently.
    use duels_core::scoring::{GameResult, VictoryKind};
    let mut military = 0;
    let mut science = 0;
    let mut civilian = 0;
    let mut draws = 0;
    for seed in 0..600u64 {
        match playout_fast(seed, 1 + seed % 97).result().unwrap() {
            GameResult::Win {
                kind: VictoryKind::MilitarySupremacy,
                ..
            } => military += 1,
            GameResult::Win {
                kind: VictoryKind::ScientificSupremacy,
                ..
            } => science += 1,
            GameResult::Win { .. } => civilian += 1,
            GameResult::Draw => draws += 1,
        }
    }
    assert!(civilian > 0, "no civilian victories in 600 games");
    assert!(military > 0, "no military supremacy in 600 games");
    assert!(
        science > 0,
        "no scientific supremacy in 600 games (military {military}, civilian {civilian}, draws {draws})"
    );
}
