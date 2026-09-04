//! Hand-built positions that pin down when each classification fires.
//!
//! Every position here is constructed with [`StateBuilder`] rather than
//! reached by playing, so the test says exactly what it means — "the pawn is
//! two steps from the capital and there are two ways to push it" — and can
//! assert an exact status. The card and wonder slugs are the real ones from
//! `data/*.json`; `slugs_used_in_these_tests_exist` guards them against a
//! rename.

use duels_core::data::{CardId, Science, TokenId, WonderId};
use duels_core::state::{Pending, Phase};
use duels_core::testing::StateBuilder;
use duels_core::{engine, Action, Player};
use duels_strategy::{
    action_denies, action_prior, military_read, science_read, stance, MilitaryStatus,
    ScienceStatus, StanceMode,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The five green cards that give Player One mortar, pendulum, inkwell, wheel
/// and gyroscope — five distinct symbols, one short of scientific supremacy,
/// with sundial the only card symbol missing.
const FIVE_SYMBOLS: &[&str] = &[
    "pharmacist",  // mortar
    "workshop",    // pendulum
    "scriptorium", // inkwell
    "apothecary",  // wheel
    "university",  // gyroscope
];

/// A full, explicit Age II structure. Face-up rows put shields in the two
/// accessible slots (18, 19) and two more behind them (11, 12); everything
/// else is inert. The three age cards it leaves out (`horse-breeders`,
/// `school`, `laboratory`) are the ones setup returns to the box.
const AGE_TWO_DEAL: [&str; 20] = [
    "sawmill",
    "brickyard",
    "shelf-quarry",
    "glassblower",
    "drying-room",
    "courthouse",
    "statue",
    "temple",
    "aqueduct",
    "rostrum",
    "forum",
    "parade-ground", // 2 shields, face up but covered
    "barracks",      // 1 shield, face up but covered
    "caravansery",
    "customs-house",
    "brewery",
    "library",
    "dispensary",
    "walls",         // 2 shields, accessible
    "archery-range", // 2 shields, accessible
];

/// Two inert cards in the two accessible Age III slots: enough of a structure
/// that both players still have a decision to make, which a *race* needs.
const AGE_THREE_QUIET: [(u8, &str); 2] = [(18, "palace"), (19, "town-hall")];

// ---------------------------------------------------------------------------
// Military
// ---------------------------------------------------------------------------

#[test]
fn one_move_from_the_capital_reads_as_imminent() {
    // The pawn is at +7 of 9 and the Circus (two shields) is accessible and
    // affordable: Player One wins outright on their next move.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "circus"), (19, "palace")])
        .conflict(7)
        .coins(Player::One, 40)
        .current(Player::One)
        .build();

    let r = military_read(&st, Player::One);
    assert_eq!(r.need, 2);
    assert_eq!(r.best_single, 2);
    assert_eq!(r.status, MilitaryStatus::Imminent);
    assert_eq!(r.turns_to_close, Some(1));
    assert_eq!(r.closing_slots, 1 << 18, "the Circus is the closing card");
    assert_eq!(r.closing_fork, 1);
    assert!(
        !r.undeniable,
        "a single closing card can be taken away by the opponent"
    );

    // ...and building it really does end the game, so "Imminent" is not a
    // claim about a position the engine disagrees with.
    let mut after = st;
    let mut rng = StdRng::from_seed([7; 32]);
    engine::apply(&mut after, Action::Build { slot: 18 }, &mut rng).unwrap();
    assert_eq!(
        after.result().and_then(|r| r.winner()),
        Some(Player::One),
        "the scenario should genuinely reach military supremacy"
    );
}

#[test]
fn two_accessible_shield_cards_are_detected_as_a_fork() {
    // Same pawn position, but now *both* accessible cards close the race, so
    // one opposing turn cannot take them both.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "circus"), (19, "fortifications")])
        .conflict(7)
        .coins(Player::One, 40)
        .current(Player::One)
        .build();

    let r = military_read(&st, Player::One);
    assert_eq!(r.status, MilitaryStatus::Imminent);
    assert_eq!(r.fork, 2, "two independent ways to advance the pawn");
    assert_eq!(r.closing_fork, 2, "and both of them close the race");
    assert!(r.undeniable);
    assert_eq!(r.closing_slots, (1 << 18) | (1 << 19));

    // The stance layer promotes both closing moves and nothing else.
    let s = stance(&st, Player::One);
    assert_eq!(s.mode, StanceMode::PushImminentFork);
    let build18 = action_prior(&st, Action::Build { slot: 18 }, &s);
    let build19 = action_prior(&st, Action::Build { slot: 19 }, &s);
    let discard18 = action_prior(&st, Action::Discard { slot: 18 }, &s);
    assert!(build18 > discard18 * 10.0, "{build18} vs {discard18}");
    assert!(build19 > discard18 * 10.0, "{build19} vs {discard18}");
}

#[test]
fn an_affordable_shield_wonder_is_a_fork_the_opponent_cannot_deny() {
    // The pawn is one step from the capital and Player One holds an unbuilt
    // Colossus (two shields). A wonder needs *some* card to spend, not a
    // particular one, so there is nothing for the opponent to take away.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "palace"), (19, "town-hall")])
        .wonders(Player::One, &["the-colossus"])
        .conflict(8)
        .coins(Player::One, 40)
        .current(Player::One)
        .build();

    let r = military_read(&st, Player::One);
    assert_eq!(r.need, 1);
    assert_eq!(r.status, MilitaryStatus::Imminent);
    assert_ne!(r.closing_wonders, 0);
    assert!(
        r.undeniable,
        "a closing wonder cannot be denied even without a second source"
    );
}

#[test]
fn a_symmetric_contested_age_is_not_a_race_for_either_side() {
    // A full Age II structure, pawn centred: each side needs all nine shields
    // and there are only four on the visible table. The supply picture is
    // wide open — most of the game's shields are still to come — but it is
    // *shared*, and the magnitude model prices that honestly: two players
    // pushing equally hard from the centre means neither of them arrives.
    //
    // The first cut of this crate called this position `Live` for both, on an
    // optimistic "never outbid" estimate. That was the calibration error the
    // magnitude model exists to fix: a race nobody wins is not a race, and
    // pricing denial against it would waste turns.
    let st = StateBuilder::new()
        .age(2)
        .deal(&AGE_TWO_DEAL)
        .conflict(0)
        .coins(Player::One, 40)
        .coins(Player::Two, 40)
        .current(Player::One)
        .build();

    for p in Player::ALL {
        let r = military_read(&st, p);
        assert_eq!(r.need, 9, "{p}: the pawn is centred");
        assert_eq!(r.now, 4, "{p}: two accessible two-shield cards");
        assert!(
            r.best_single < r.need,
            "{p}: nobody should be able to force it in one move"
        );
        // The supply is real...
        assert_eq!(r.visible, 7);
        assert!(r.expected_hidden > 0.0 && r.expected_hidden < 1.0);
        assert!(
            r.expected_future_ages > 5.0,
            "Age III still holds most of the game's shields: {}",
            r.expected_future_ages
        );
        // ...but it is contested, so the simulation only ever limps to the
        // capital, and the tempo discount leaves almost nothing.
        assert!(
            r.magnitude < 0.05,
            "{p}: a symmetric centred pawn should not read as a threat: M = {}",
            r.magnitude
        );
        assert_eq!(r.status, MilitaryStatus::Closed, "{p}");
        let turns = r.turns_to_close.expect("the supply does eventually add up");
        assert!(
            turns > 10 && turns <= r.decisions_left,
            "{p}: implausible turns_to_close {turns} of {} decisions",
            r.decisions_left
        );
    }
}

#[test]
fn an_unanswerable_push_close_to_the_capital_is_a_real_race() {
    // The other half of the same coin. The pawn is five steps up, a
    // three-shield Arsenal is accessible, and Player One holds an unbuilt
    // Colossus for the last two shields.
    //
    // The Colossus is the load-bearing part: a *card* can always be denied,
    // because discarding it for coins costs the opponent nothing but a turn,
    // so two accessible red cards are not two routes. A wonder needs only
    // some card to spend, and there is nothing to take away.
    let st = STANCE_PUSH_POSITION();

    let r = military_read(&st, Player::One);
    assert_eq!(r.need, 4);
    assert_eq!(r.best_single, 3, "the Arsenal is three shields");
    assert_eq!(
        r.turns_to_close,
        Some(2),
        "the Arsenal, then the Colossus the opponent cannot touch"
    );
    assert_eq!(r.status, MilitaryStatus::Live);
    assert!((r.magnitude - 0.7).abs() < 1e-9, "M = {}", r.magnitude);

    // ...and the opponent's own race is nowhere.
    let theirs = military_read(&st, Player::Two);
    assert_eq!(theirs.need, 14);
    assert_eq!(theirs.status, MilitaryStatus::Closed);
    assert_eq!(theirs.magnitude, 0.0);
}

/// The position both the military-magnitude and the push-stance test use:
/// pawn at +5, a three-shield Arsenal on the table, and an unbuilt Colossus
/// in Player One's hand for the last two.
#[allow(non_snake_case)]
fn STANCE_PUSH_POSITION() -> duels_core::GameState {
    StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "arsenal"), (19, "palace"), (15, "town-hall")])
        .wonders(Player::One, &["the-colossus"])
        .conflict(5)
        .coins(Player::One, 40)
        .coins(Player::Two, 3)
        .current(Player::One)
        .build()
}

#[test]
fn a_hopeless_pawn_position_late_in_age_three_reads_as_closed() {
    // Player Two is nine steps behind with two inert cards on the table and
    // no wonders: there is no supply left anywhere.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "palace"), (19, "town-hall")])
        .conflict(8)
        .coins(Player::Two, 40)
        .current(Player::One)
        .build();

    let r = military_read(&st, Player::Two);
    assert_eq!(r.need, 17);
    assert_eq!(r.status, MilitaryStatus::Closed);
    assert_eq!(r.turns_to_close, None);
}

#[test]
fn loot_damage_is_capped_at_what_the_opponent_actually_holds() {
    // The first loot token sits at distance 3 and takes two coins; the second
    // at distance 6 and takes five. With the near token already collected and
    // the opponent down to one coin, the reported damage is one, not five.
    let st = StateBuilder::new()
        .age(3)
        .conflict(4)
        .loot_taken(Player::One, 0)
        .coins(Player::Two, 1)
        .build();
    let r = military_read(&st, Player::One);
    assert_eq!(r.loot_damage, 1);
    assert_eq!(r.loot_shields_needed, Some(2), "distance 4 -> 6");

    let rich = StateBuilder::new()
        .age(3)
        .conflict(4)
        .loot_taken(Player::One, 0)
        .coins(Player::Two, 30)
        .build();
    assert_eq!(military_read(&rich, Player::One).loot_damage, 5);
}

// ---------------------------------------------------------------------------
// Science
// ---------------------------------------------------------------------------

#[test]
fn missing_both_copies_of_an_age_three_symbol_kills_the_science_race() {
    // Player One holds five distinct symbols — as strong a science position as
    // exists short of winning — but both Sundial cards are in Player Two's
    // city, and no Law token is anywhere. The sixth symbol is physically
    // unreachable, so the race is dead however good the rest looks.
    let st = StateBuilder::new()
        .age(3)
        .built(Player::One, FIVE_SYMBOLS)
        .built(Player::Two, &["academy", "study"])
        .current(Player::One)
        .build();

    let r = science_read(&st, Player::One);
    assert_eq!(r.distinct, 5);
    assert_eq!(r.missing, 1);
    assert_eq!(r.availability[Science::Sundial.index()].gone, 2);
    assert!(!r.availability[Science::Sundial.index()].obtainable());
    assert_eq!(r.obtainable_missing, 0);
    assert!(r.dead, "no route to a sixth symbol");
    assert_eq!(r.status, ScienceStatus::Closed);
}

#[test]
fn the_law_token_on_the_board_revives_that_same_position() {
    // Identical to the position above, except the Law token — which supplies a
    // seventh symbol of its own — is still on the board. There are seven
    // distinct symbols in the game and only six are needed, so losing Sundial
    // is survivable exactly once.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&AGE_THREE_QUIET)
        .built(Player::One, FIVE_SYMBOLS)
        .built(Player::Two, &["academy", "study"])
        .board_tokens(&["law"])
        .current(Player::One)
        .build();

    let r = science_read(&st, Player::One);
    assert!(r.availability[Science::Balance.index()].via_law_board);
    assert_eq!(r.obtainable_missing, 1);
    assert!(!r.dead);
    assert_eq!(r.status, ScienceStatus::Live);
    assert!(r.magnitude > 0.8, "M = {}", r.magnitude);
}

#[test]
fn a_pending_token_choice_with_law_on_offer_is_an_imminent_science_win() {
    let st = StateBuilder::new()
        .age(3)
        .built(Player::One, FIVE_SYMBOLS)
        .built(Player::Two, &["academy", "study"])
        .board_tokens(&["law", "philosophy"])
        .pending(Pending::ProgressToken)
        .current(Player::One)
        .build();

    let r = science_read(&st, Player::One);
    let law = TokenId::from_slug("law").unwrap();
    assert_eq!(r.closing_via_token, Some(law));
    assert_eq!(r.status, ScienceStatus::Imminent);

    // The engine agrees: taking Law really does end the game.
    let mut after = st;
    let mut rng = StdRng::from_seed([3; 32]);
    engine::apply(
        &mut after,
        Action::ChooseProgressToken { token: law },
        &mut rng,
    )
    .unwrap();
    assert_eq!(after.result().and_then(|r| r.winner()), Some(Player::One));
}

#[test]
fn an_unbuilt_great_library_keeps_a_set_aside_law_token_in_reach() {
    let base = StateBuilder::new()
        .age(3)
        .open_slots(&AGE_THREE_QUIET)
        .built(Player::One, FIVE_SYMBOLS)
        .built(Player::Two, &["academy", "study"])
        .set_aside_tokens(&["law", "philosophy", "agriculture", "economy", "theology"])
        .current(Player::One);

    // Without the wonder, the set-aside pile is out of reach.
    let without = base.clone().build();
    let r = science_read(&without, Player::One);
    assert!(!r.availability[Science::Balance.index()].via_law_great_library);
    assert!(r.dead);

    // With an unbuilt Great Library it is reachable — a three-of-five draw,
    // which is why it is reported as its own route rather than folded in.
    let with = base.wonders(Player::One, &["the-great-library"]).build();
    let r = science_read(&with, Player::One);
    assert!(r.availability[Science::Balance.index()].via_law_great_library);
    assert!(!r.availability[Science::Balance.index()].via_law_board);
    assert!(!r.dead);
    assert_eq!(r.status, ScienceStatus::Live);
}

#[test]
fn a_built_great_library_no_longer_counts_the_set_aside_pile() {
    // The draw happens when the wonder is constructed; afterwards the pile is
    // no longer a route.
    let st = StateBuilder::new()
        .age(3)
        .built(Player::One, FIVE_SYMBOLS)
        .built(Player::Two, &["academy", "study"])
        .set_aside_tokens(&["law", "philosophy", "agriculture"])
        .wonders_built(Player::One, &["the-great-library"])
        .current(Player::One)
        .build();
    let r = science_read(&st, Player::One);
    assert!(!r.availability[Science::Balance.index()].via_law_great_library);
    assert!(r.dead);
}

#[test]
fn an_unbuilt_mausoleum_makes_a_discarded_symbol_obtainable_again() {
    // Player One holds four symbols. One Sundial card is in Player Two's city
    // and the other is in the discard pile; Gyroscope is untouched.
    let base = StateBuilder::new()
        .age(3)
        // Both Gyroscope cards are on the table but covered, so that symbol's
        // supply is certain without being takeable this turn, and the test is
        // only about Sundial's Mausoleum route.
        .open_slots(&[
            (11, "university"),
            (12, "observatory"),
            (15, "palace"),
            (16, "town-hall"),
            (17, "obelisk"),
            (18, "senate"),
            (19, "gardens"),
        ])
        .built(
            Player::One,
            &["pharmacist", "workshop", "scriptorium", "apothecary"],
        )
        .built(Player::Two, &["academy"])
        .discard(&["study"])
        .coins(Player::One, 40)
        .current(Player::One);

    let without = base.clone().build();
    let r = science_read(&without, Player::One);
    assert_eq!(r.missing, 2);
    assert_eq!(r.availability[Science::Sundial.index()].via_mausoleum, 0);
    assert!(!r.availability[Science::Sundial.index()].obtainable());
    assert_eq!(
        r.obtainable_missing, 1,
        "only Gyroscope is left, and two symbols are needed"
    );
    assert!(r.dead);

    let with = base.wonders(Player::One, &["the-mausoleum"]).build();
    let r = science_read(&with, Player::One);
    assert_eq!(r.availability[Science::Sundial.index()].via_mausoleum, 1);
    assert_eq!(r.obtainable_missing, 2);
    assert!(!r.dead);
    // The Mausoleum is affordable, so that copy counts at full weight.
    assert!((r.copies(Science::Sundial) - 1.0).abs() < 1e-9);
    assert!(
        r.kill_cost(Science::Sundial).is_infinite(),
        "a card in the discard pile behind a Mausoleum cannot be denied"
    );
    // Sundial is down to that single discarded copy: one point of failure.
    assert_eq!(r.fragility, 1);
    assert_eq!(r.status, ScienceStatus::Live);
}

#[test]
fn an_accessible_affordable_sixth_symbol_is_an_imminent_science_win() {
    let st = StateBuilder::new()
        .age(3)
        .built(
            Player::One,
            &[
                "pharmacist",
                "workshop",
                "scriptorium",
                "apothecary",
                "academy",
            ],
        )
        .open_slots(&[(18, "university"), (19, "palace")])
        .coins(Player::One, 40)
        .current(Player::One)
        .build();

    let r = science_read(&st, Player::One);
    assert_eq!(r.distinct, 5);
    assert_eq!(r.closing_slots, 1 << 18);
    assert_eq!(r.status, ScienceStatus::Imminent);

    // The same card with no coins to pay for it is not imminent: the read
    // uses the real cost engine, not just what is on the table.
    let broke = StateBuilder::new()
        .age(3)
        .built(
            Player::One,
            &[
                "pharmacist",
                "workshop",
                "scriptorium",
                "apothecary",
                "academy",
            ],
        )
        .open_slots(&[(18, "university"), (19, "palace")])
        .coins(Player::One, 1)
        .current(Player::One)
        .build();
    let r = science_read(&broke, Player::One);
    assert_eq!(r.closing_slots, 0);
    assert_ne!(r.status, ScienceStatus::Imminent);
}

#[test]
fn three_missing_symbols_is_pressure_rather_than_a_race() {
    // Mid Age II with three distinct symbols. Inkwell is down to the single
    // copy still in the age's unknown pool, Sundial and Gyroscope are both
    // waiting in the undealt Age III: three routes for three missing symbols,
    // which is alive but far too thin to call a race.
    let st = StateBuilder::new()
        .age(2)
        .deal(&AGE_TWO_DEAL)
        .built(Player::One, &["pharmacist", "workshop", "apothecary"])
        .coins(Player::One, 40)
        .current(Player::One)
        .build();
    let r = science_read(&st, Player::One);
    assert_eq!(r.distinct, 3);
    assert_eq!(r.missing, 3);
    assert_eq!(r.availability[Science::Inkwell.index()].in_unknown_pool, 1);
    assert_eq!(r.availability[Science::Sundial.index()].in_future_age, 2);
    assert_eq!(r.obtainable_missing, 3);
    assert!(!r.dead);
    assert_eq!(r.fragility, 1, "Inkwell is one card away from gone");
    assert_eq!(r.status, ScienceStatus::Pressure);
}

#[test]
fn a_symbol_whose_copies_are_all_in_finished_ages_is_simply_gone() {
    // By Age III, a symbol carried only by Age I and Age II cards is either in
    // a city already or was boxed at setup — either way there is no route to
    // it. That asymmetry (Sundial and Gyroscope exist only in Age III, the
    // other four only outside it in part) is what makes late science races so
    // brittle, and it falls straight out of the card data.
    let st = StateBuilder::new()
        .age(3)
        .built(Player::One, &["pharmacist", "workshop", "scriptorium"])
        .open_slots(&[(18, "university"), (19, "academy")])
        .coins(Player::One, 40)
        .current(Player::One)
        .build();
    let r = science_read(&st, Player::One);
    assert_eq!(r.missing, 3);
    // Wheel's two cards are Apothecary (Age I) and School (Age II): gone.
    assert_eq!(r.availability[Science::Wheel.index()].gone, 2);
    // Sundial and Gyroscope are right there on the table.
    assert!(r.availability[Science::Sundial.index()].obtainable());
    assert!(r.availability[Science::Gyroscope.index()].obtainable());
    // Three needed, only two reachable.
    assert_eq!(r.obtainable_missing, 2);
    assert!(r.dead);
    assert_eq!(r.status, ScienceStatus::Closed);
}

#[test]
fn pair_setup_reports_the_half_pairs_and_what_the_board_pays_for_them() {
    // Player One holds one Inkwell (Scriptorium) and the second one
    // (Library) is accessible and affordable: completing the pair claims a
    // progress token, and Philosophy is the best one on the board.
    let st = StateBuilder::new()
        .age(2)
        .built(Player::One, &["scriptorium"])
        .open_slots(&[(18, "library"), (19, "temple")])
        .board_tokens(&["philosophy", "economy"])
        .coins(Player::One, 40)
        .current(Player::One)
        .build();

    let r = science_read(&st, Player::One);
    assert!(r.pair_setup.candidates[Science::Inkwell.index()]);
    assert!(r.pair_setup.completable_now[Science::Inkwell.index()]);
    assert_eq!(r.pair_setup.completing_slots, 1 << 18);
    let (token, value) = r.pair_setup.best_board_token.expect("two tokens on offer");
    assert_eq!(
        token.slug(),
        "philosophy",
        "seven printed points beats Economy"
    );
    assert!(
        (value - 7.0).abs() < 1e-9,
        "value {value} should be Philosophy's printed points"
    );

    // A pair already paid for is not a candidate again.
    let paid = StateBuilder::new()
        .age(2)
        .built(Player::One, &["scriptorium"])
        .pair_already_awarded(Player::One, Science::Inkwell)
        .open_slots(&[(18, "library")])
        .board_tokens(&["philosophy"])
        .coins(Player::One, 40)
        .build();
    let r = science_read(&paid, Player::One);
    assert!(!r.pair_setup.candidates[Science::Inkwell.index()]);
    assert_eq!(r.pair_setup.completing_slots, 0);
}

// ---------------------------------------------------------------------------
// Stance
// ---------------------------------------------------------------------------

#[test]
fn an_imminent_opposing_military_win_puts_the_stance_into_deny_mode() {
    // Player Two is two steps from the capital with the Circus accessible and
    // affordable. Player One is to move and can take that card away.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "circus"), (19, "palace")])
        .conflict(-7)
        .coins(Player::One, 40)
        .coins(Player::Two, 40)
        .current(Player::One)
        .build();

    let s = stance(&st, Player::One);
    assert_eq!(s.opponent_military.status, MilitaryStatus::Imminent);
    assert_eq!(s.mode, StanceMode::DenyCertain);
    assert_ne!(s.deny_slots & (1 << 18), 0);
    assert!(action_denies(Action::Discard { slot: 18 }, &s));
    assert!(!action_denies(Action::Discard { slot: 19 }, &s));

    let deny = action_prior(&st, Action::Discard { slot: 18 }, &s);
    let ignore = action_prior(&st, Action::Discard { slot: 19 }, &s);
    assert!(deny > ignore * 10.0, "deny {deny} vs ignore {ignore}");
}

#[test]
fn an_imminent_opposing_science_win_is_denied_the_same_way() {
    let st = StateBuilder::new()
        .age(3)
        .built(
            Player::Two,
            &[
                "pharmacist",
                "workshop",
                "scriptorium",
                "apothecary",
                "academy",
            ],
        )
        .open_slots(&[(18, "university"), (19, "palace")])
        .coins(Player::One, 40)
        .coins(Player::Two, 40)
        .current(Player::One)
        .build();

    let s = stance(&st, Player::One);
    assert_eq!(s.opponent_science.status, ScienceStatus::Imminent);
    assert_eq!(s.mode, StanceMode::DenyCertain);
    assert_ne!(s.deny_slots & (1 << 18), 0);
}

#[test]
fn the_stance_leans_into_a_live_race_and_reports_which_one() {
    // Player One is four shields from the capital with a three-shield Arsenal
    // on the table and an unbuilt Colossus for the last two, and Player Two
    // has three coins and no shields anywhere. The push is unanswered, so the
    // stance takes the race.
    let st = STANCE_PUSH_POSITION();

    let s = stance(&st, Player::One);
    assert_eq!(s.opponent_military.best_single, 0, "Player Two is broke");
    assert_eq!(s.mode, StanceMode::PushLive);
    assert_eq!(s.race, Some(duels_strategy::Race::Military));
    assert!(s.tilt > 0.0);
    let push = action_prior(&st, Action::Build { slot: 18 }, &s);
    let inert = action_prior(&st, Action::Discard { slot: 18 }, &s);
    assert!(push > inert, "push {push} vs inert {inert}");
}

#[test]
fn a_race_the_opponent_can_answer_in_kind_is_not_pushed() {
    // Both players can put two shields on the board this turn, so leaning into
    // the pawn accomplishes nothing: the stance falls through to points.
    let st = StateBuilder::new()
        .age(2)
        .deal(&AGE_TWO_DEAL)
        .conflict(0)
        .coins(Player::One, 40)
        .coins(Player::Two, 40)
        .current(Player::One)
        .build();
    let s = stance(&st, Player::One);
    assert_eq!(s.mode, StanceMode::VpEfficient);
    assert_eq!(s.race, None);
}

#[test]
fn priors_are_always_positive_and_finite_over_a_whole_game() {
    // The prior is a weight a search will normalize; a zero or a NaN would
    // silently remove a legal move from consideration.
    let mut rng = StdRng::from_seed([42; 32]);
    for seed in 0..6u64 {
        let mut st = engine::new_game(seed);
        let mut guard = 0;
        loop {
            let legal = engine::legal_actions(&st);
            if legal.is_empty() {
                break;
            }
            for p in Player::ALL {
                let s = stance(&st, p);
                for &a in &legal {
                    let w = action_prior(&st, a, &s);
                    assert!(
                        w.is_finite() && w > 0.0,
                        "seed {seed} turn {} action {a:?}: prior {w}",
                        st.turn()
                    );
                }
            }
            let a = legal[(st.turn() as usize * 3) % legal.len()];
            engine::apply_quiet(&mut st, a, &mut rng).unwrap();
            guard += 1;
            assert!(guard < 500);
        }
    }
}

#[test]
fn uncovering_shields_for_a_near_opponent_is_priced_as_a_negative_delta_m() {
    // This is the test that used to assert a separate `exposure_risk` term.
    // It now asserts the thing that replaced it: handing the opponent a red
    // card they can reach *raises* their military magnitude, so `delta_m` is
    // negative and the move is priced down.
    //
    // Player Two needs three shields. Slot 18 covers slot 16, which holds a
    // face-up Arsenal (three shields); slot 19 covers 16 and 17, and 17 holds
    // an inert card. Taking 19 does not open the Arsenal on its own, because
    // 16 is still covered by 18 — so the two moves differ only in what they
    // expose.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[
            (16, "arsenal"),
            (17, "palace"),
            (18, "town-hall"),
            (19, "obelisk"),
        ])
        .conflict(-6)
        .coins(Player::One, 40)
        .coins(Player::Two, 40)
        .current(Player::One)
        .build();

    let s = stance(&st, Player::One);
    // Slot 18 alone does not uncover 16 either (19 still covers it), so
    // compare what each move exposes directly.
    let (known18, _) = s.board.newly_open_after(18);
    let (known19, _) = s.board.newly_open_after(19);
    assert_eq!(known18, 0, "slot 16 is still covered by slot 19");
    assert_eq!(
        known19,
        1u128 << CardId::from_slug("palace").unwrap().index()
    );

    // Now remove slot 18's cover so that taking 19 really does open the
    // Arsenal to a player three shields from supremacy.
    let sharp = StateBuilder::new()
        .age(3)
        .open_slots(&[(16, "arsenal"), (19, "obelisk")])
        .conflict(-6)
        .coins(Player::One, 40)
        .coins(Player::Two, 40)
        .current(Player::One)
        .build();
    let s2 = stance(&sharp, Player::One);
    let (known, _) = s2.board.newly_open_after(19);
    assert_eq!(
        known,
        1u128 << CardId::from_slug("arsenal").unwrap().index()
    );

    let d = duels_strategy::delta_m(Action::Discard { slot: 19 }, &s2);
    assert!(
        d.military < 0.0,
        "uncovering an Arsenal for a player who needs three shields should \
         raise their magnitude, not lower it: delta {d:?}"
    );
    assert!(
        duels_strategy::deny_vp(Action::Discard { slot: 19 }, &s2) < 0.0,
        "and that should be priced as a cost"
    );
}

#[test]
fn slugs_used_in_these_tests_exist() {
    for slug in FIVE_SYMBOLS.iter().chain(AGE_TWO_DEAL.iter()).chain(
        [
            "circus",
            "fortifications",
            "arsenal",
            "palace",
            "town-hall",
            "obelisk",
            "academy",
            "study",
            "university",
        ]
        .iter(),
    ) {
        assert!(CardId::from_slug(slug).is_some(), "missing card {slug}");
    }
    for slug in [
        "university",
        "observatory",
        "gardens",
        "pantheon",
        "senate",
        "pharmacist",
        "workshop",
        "scriptorium",
        "apothecary",
    ] {
        assert!(CardId::from_slug(slug).is_some(), "missing card {slug}");
    }
    for slug in ["the-colossus", "the-great-library", "the-mausoleum"] {
        assert!(WonderId::from_slug(slug).is_some(), "missing wonder {slug}");
    }
    for slug in ["law", "philosophy", "economy", "agriculture", "theology"] {
        assert!(TokenId::from_slug(slug).is_some(), "missing token {slug}");
    }
    // `StateBuilder::new` starts in Age III, mid-play.
    assert_eq!(StateBuilder::new().build().phase(), Phase::Turn);
}
