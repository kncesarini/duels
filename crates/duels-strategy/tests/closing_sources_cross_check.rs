//! [`closing_sources`] against the engine itself.
//!
//! This crate is not the rules authority — `duels-core` is — so a read that
//! claims "this action ends the game outright" has to be checked against the
//! only thing that can actually answer that: applying the action and asking
//! [`duels_core::GameState::result`].
//!
//! The check runs in both directions and is exact, not statistical:
//!
//! * **no false negatives** — every action the engine turns into a military or
//!   scientific supremacy win must be flagged;
//! * **no false positives** — every action flagged must really end the game.
//!
//! It runs over real positions from replayed games (which is where the
//! false-positive direction gets its volume) plus hand-built positions from
//! [`StateBuilder`] (which is the only practical way to reach a five-symbol
//! science position or a two-shields-from-the-capital pawn on demand).

use duels_core::scoring::VictoryKind;
use duels_core::testing::StateBuilder;
use duels_core::{engine, Action, GameResult, GameState, Player};
use duels_strategy::{closing_sources, closing_sources_with};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Whether applying `action` ends the game by a *race*, over every way its
/// randomness can resolve. Asserts the outcomes agree with each other, since a
/// supremacy win is settled before any card is turned over.
fn ends_the_game(state: &GameState, action: Action) -> bool {
    let outcomes = engine::chance_outcomes(state, action);
    let mut seen: Option<bool> = None;
    for (outcome, _) in &outcomes {
        let mut next = *state;
        engine::apply_with_outcome(&mut next, action, outcome).expect("legal action");
        let won = matches!(
            next.result(),
            Some(GameResult::Win {
                kind: VictoryKind::MilitarySupremacy | VictoryKind::ScientificSupremacy,
                ..
            })
        );
        match seen {
            None => seen = Some(won),
            Some(before) => assert_eq!(
                before, won,
                "{action:?} ends the game in one chance outcome and not another"
            ),
        }
    }
    seen.unwrap_or(false)
}

/// The core assertion, for the player to move in `state`.
///
/// Returns `(military hits, science hits)` so callers can prove the positive
/// direction was actually exercised.
#[track_caller]
fn cross_check(state: &GameState, ctx: &str) -> (u32, u32) {
    let p = state.current_player();
    let sources = closing_sources(state, p);
    let mut military = 0;
    let mut science = 0;

    for action in engine::legal_actions(state) {
        let predicted = match action {
            Action::Build { slot } => {
                (sources.military_slots | sources.science_slots) & (1u32 << slot) != 0
            }
            Action::BuildWonder { wonder, .. } => {
                sources.military_wonders & (1u16 << wonder.index()) != 0
            }
            _ => false,
        };
        let actual = ends_the_game(state, action);
        assert_eq!(
            predicted, actual,
            "{ctx}: closing_sources says {predicted} for {action:?}, the engine says {actual}"
        );
        if actual {
            match action {
                Action::Build { slot } if sources.military_slots & (1u32 << slot) != 0 => {
                    military += 1;
                }
                Action::BuildWonder { .. } => military += 1,
                _ => science += 1,
            }
        }
    }
    (military, science)
}

/// A deterministic replay, so the sweep is reproducible.
///
/// `shield_hungry` makes Player One take the highest-shield card it can reach
/// on every turn. Without it a mechanical index policy essentially never
/// drives the pawn to a capital, and the sweep would only ever test the
/// no-false-positive half of the property.
fn advance(seed: u64, steps: usize, shield_hungry: bool) -> GameState {
    let mut st = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5A5A);
    for _ in 0..steps {
        if st.is_over() {
            break;
        }
        let actions = engine::legal_actions(&st);
        if actions.is_empty() {
            break;
        }
        let greedy = if shield_hungry && st.current_player() == Player::One {
            actions
                .iter()
                .copied()
                .filter_map(|a| match a {
                    Action::Build { slot } => st
                        .face_up_card(slot)
                        .filter(|c| c.def().shields > 0)
                        .map(|c| (c.def().shields, slot, a)),
                    _ => None,
                })
                .max_by_key(|&(shields, slot, _)| (shields, slot))
                .map(|(_, _, a)| a)
        } else {
            None
        };
        let a = greedy.unwrap_or(actions[(st.turn() as usize * 7 + seed as usize) % actions.len()]);
        engine::apply_quiet(&mut st, a, &mut rng).unwrap();
    }
    st
}

#[test]
fn no_replayed_position_ever_flags_an_action_the_engine_disagrees_with() {
    let mut positions = 0usize;
    let mut hits = 0u32;
    for seed in 0..24u64 {
        for (steps, hungry) in (0..60usize).flat_map(|s| [(s, false), (s, true)]) {
            let st = advance(seed, steps, hungry);
            if st.is_over()
                || st.phase() != duels_core::state::Phase::Turn
                || st.pending().is_some()
            {
                continue;
            }
            let (military, science) = cross_check(&st, &format!("seed {seed} steps {steps}"));
            hits += military + science;
            positions += 1;
        }
    }
    assert!(positions > 500, "only {positions} positions exercised");
    assert!(
        hits > 0,
        "the sweep never met a position with a real closing action, so it only \
         tested the no-false-positive half"
    );
}

#[test]
fn a_closing_red_card_and_a_closing_wonder_are_both_found() {
    // Two shields from the capital, with a two-shield red card face up and a
    // one-shield card that does not reach.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "circus"), (19, "guard-tower"), (17, "palace")])
        .conflict(7)
        .coins(Player::One, 30)
        .coins(Player::Two, 30)
        .current(Player::One)
        .build();
    let (military, science) = cross_check(&st, "closing red card");
    assert_eq!(military, 1, "exactly the Circus should close");
    assert_eq!(science, 0);
    let sources = closing_sources(&st, Player::One);
    assert_eq!(sources.need, 2);
    assert_eq!(sources.slot_shields[18], 2);

    // The same pawn, but the player cannot pay for the card.
    let broke = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "circus"), (19, "guard-tower")])
        .conflict(7)
        .coins(Player::One, 0)
        .coins(Player::Two, 30)
        .current(Player::One)
        .build();
    assert!(
        !closing_sources(&broke, Player::One).any(),
        "an unaffordable card is not a closing source"
    );
    cross_check(&broke, "unaffordable closing card");
}

#[test]
fn a_closing_source_is_found_for_the_player_who_is_not_to_move() {
    // The whole reason this function exists: `engine::legal_actions` can only
    // answer for the mover, and a 1-ply evaluator has to ask about the other
    // one.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "circus"), (19, "palace")])
        .conflict(-7)
        .coins(Player::One, 30)
        .coins(Player::Two, 30)
        .current(Player::One)
        .build();
    let theirs = closing_sources(&st, Player::Two);
    assert!(theirs.any(), "Player Two is two shields from the capital");
    assert_eq!(theirs.military_slots, 1u32 << 18);
    // ...and it really is their win: hand them the turn and the engine agrees.
    let handed = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "circus"), (19, "palace")])
        .conflict(-7)
        .coins(Player::One, 30)
        .coins(Player::Two, 30)
        .current(Player::Two)
        .build();
    cross_check(&handed, "the other player's closing card");
}

#[test]
fn the_science_closing_card_is_found_at_five_distinct_symbols() {
    let st = StateBuilder::new()
        .age(3)
        .built(
            Player::One,
            &[
                "workshop",
                "apothecary",
                "scriptorium",
                "pharmacist",
                "academy",
            ],
        )
        .open_slots(&[(18, "university"), (19, "palace")])
        .coins(Player::One, 30)
        .coins(Player::Two, 30)
        .current(Player::One)
        .build();
    assert_eq!(st.player(Player::One).distinct_science(), 5);
    let (military, science) = cross_check(&st, "closing green card");
    assert_eq!(science, 1, "the University completes the sixth symbol");
    assert_eq!(military, 0);
    assert_eq!(closing_sources(&st, Player::One).science_slots, 1u32 << 18);
}

#[test]
fn the_strategy_token_is_what_makes_a_one_shield_card_close() {
    let build = |tokens: &[&str]| {
        StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "guard-tower"), (19, "palace")])
            .tokens(Player::One, tokens)
            .conflict(8)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build()
    };
    let plain = build(&[]);
    let strategist = build(&["strategy"]);
    assert_eq!(closing_sources(&plain, Player::One).need, 1);
    assert!(closing_sources(&plain, Player::One).any());
    assert!(closing_sources(&strategist, Player::One).any());
    cross_check(&plain, "one shield, one needed");
    cross_check(&strategist, "one shield plus strategy");

    // Two shields needed: only the Strategy holder closes.
    let build2 = |tokens: &[&str]| {
        StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "guard-tower"), (19, "palace")])
            .tokens(Player::One, tokens)
            .conflict(7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build()
    };
    assert!(!closing_sources(&build2(&[]), Player::One).any());
    assert!(closing_sources(&build2(&["strategy"]), Player::One).any());
    cross_check(&build2(&[]), "one shield, two needed");
    cross_check(
        &build2(&["strategy"]),
        "one shield plus strategy, two needed",
    );
}

/// The stand-down variant reads no card in the structure at all, which is what
/// makes it safe across an age boundary.
#[test]
fn the_wonders_only_variant_drops_every_card_source_and_keeps_the_wonders() {
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "circus"), (19, "palace")])
        .conflict(7)
        .coins(Player::One, 30)
        .coins(Player::Two, 30)
        .current(Player::One)
        .build();
    let all = closing_sources_with(&st, Player::One, true);
    let lean = closing_sources_with(&st, Player::One, false);
    assert!(all.any());
    assert_eq!(lean.military_slots, 0);
    assert_eq!(lean.science_slots, 0);
    assert_eq!(lean.need, all.need);
    assert_eq!(lean.military_wonders, all.military_wonders);
}
