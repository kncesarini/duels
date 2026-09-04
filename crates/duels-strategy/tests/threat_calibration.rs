//! What the threat magnitudes actually say, on positions a strong player would
//! recognise.
//!
//! `scenarios.rs` pins down *which classification fires*. This file pins down
//! the numbers underneath it, because a magnitude is only useful if its scale
//! means something: "about even" has to come out near a half, "certain"
//! has to come out at one, and two positions a player would call equally
//! threatening have to come out equal. Every band here was predicted from the
//! model on paper before it was measured, and the comment on each test says
//! what the arithmetic is, so a number that drifts says *why* it drifted.
//!
//! Positions are built with [`StateBuilder`] rather than played into, in the
//! same style as `scenarios.rs`.
//!
//! # A note on `open_slots` and hidden supply
//!
//! [`StateBuilder::open_slots`] places cards face up and empties every other
//! slot, so such a position has **no face-down slots at all**. A card that is
//! neither placed nor accounted for is therefore in the unknown pool with an
//! expected zero copies on the table, which is arithmetically right (nothing
//! is hidden) but not what a mid-age position looks like. Tests that need a
//! symbol's supply to be *certain* put both its cards in the structure; tests
//! about symbols from an age that has not been dealt do not care, because
//! those are weighted by `p_dealt` instead.

use duels_core::data::{CardId, Science, TokenId, WonderId};
use duels_core::testing::StateBuilder;
use duels_core::{engine, Action, GameState, Player};
use duels_strategy::{
    action_prior, action_vp_value, delta_m, military_read, science_read, science_read_with, stance,
    Context, MilitaryStatus, PriorWeights, ScienceStatus, ThreatWeights,
};

/// Mortar, pendulum, inkwell and wheel: four symbols a player can hold before
/// Age III is dealt, which is the position the science calibration turns on.
const FOUR_EARLY: &[&str] = &["pharmacist", "workshop", "scriptorium", "apothecary"];

/// Mortar, pendulum, inkwell, wheel and gyroscope: five distinct symbols, one
/// short, with Sundial the only card symbol missing.
const FIVE_SYMBOLS: &[&str] = &[
    "pharmacist",
    "workshop",
    "scriptorium",
    "apothecary",
    "university",
];

/// The whole set-aside pile at setup: five of the ten progress tokens.
const FIVE_ASIDE: &[&str] = &["law", "philosophy", "agriculture", "economy", "theology"];

/// Four early symbols, two inert cards left in the current age's structure.
///
/// `age` is 1 or 2; the two cards belong to that age so the position is
/// self-consistent. Everything the race turns on — Sundial and Gyroscope —
/// lives in the undealt Age III either way, which is the point of the pair.
fn four_symbols_late_in(age: u8) -> StateBuilder {
    let slots: [(u8, &str); 2] = match age {
        1 => [(18, "lumber-yard"), (19, "clay-pool")],
        2 => [(18, "sawmill"), (19, "brickyard")],
        other => panic!("only Ages I and II have four-symbol positions, got {other}"),
    };
    StateBuilder::new()
        .age(age)
        .open_slots(&slots)
        .built(Player::One, FOUR_EARLY)
        .coins(Player::One, 20)
        .coins(Player::Two, 20)
        .current(Player::One)
}

/// Five symbols in Age III with both Sundial cards in the opponent's city, so
/// the Law token is the *only* sixth symbol left. Two inert cards on the
/// table, so there is still a decision to make.
fn balance_is_the_last_symbol() -> StateBuilder {
    StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "palace"), (19, "town-hall")])
        .built(Player::One, FIVE_SYMBOLS)
        .built(Player::Two, &["academy", "study"])
        .coins(Player::One, 40)
        .current(Player::One)
}

// ---------------------------------------------------------------------------
// 1-3: the scale of a science race, and the Age I / Age II equivalence
// ---------------------------------------------------------------------------

#[test]
fn four_early_symbols_with_nothing_in_play_is_about_an_even_race() {
    // Four symbols held, Sundial and Gyroscope missing, and nothing else in
    // play. Both those symbols exist only on Age III cards, so each has two
    // copies in the undealt deck at `p_dealt(3) = 17/20`:
    //
    //   c = 2 × 0.85 = 1.7, so supply is not the problem: surface = 1
    //   kill = 1.7 as well — that is what the defender would have to take
    //   P(defender kills one) = share^1.7 = 0.5^1.7 = 0.308
    //   two symbols needed of two available, so ONE kill is enough:
    //   P(stopped) = 1 - (1 - 0.308)^2 = 0.521
    //   M = 1 × (1 - 0.521) = 0.479
    //
    // An even race with a slight edge to the defender, which is what two
    // must-have symbols against a player who only has to spoil one should be.
    let st = four_symbols_late_in(2).build();
    let r = science_read(&st, Player::One);

    assert_eq!(r.distinct, 4);
    assert_eq!(r.missing, 2);
    assert!(!r.dead);
    assert_eq!(r.obtainable_missing, 2);
    assert!(
        (0.40..=0.55).contains(&r.magnitude),
        "M = {} is outside the calibrated band",
        r.magnitude
    );
    assert_eq!(r.status, ScienceStatus::Live);

    // ...and the pieces it is made of.
    for sym in [Science::Sundial, Science::Gyroscope] {
        assert_eq!(r.availability[sym.index()].in_future_age, 2);
        assert!((r.copies(sym) - 1.7).abs() < 1e-9, "c = {}", r.copies(sym));
        assert!((r.kill_cost(sym) - 1.7).abs() < 1e-9);
    }
    assert!((r.detail.surface - 1.0).abs() < 1e-9);
    assert_eq!(r.detail.slack, 0, "no spare route");
}

#[test]
fn the_law_token_on_the_board_plus_a_live_half_pair_lifts_the_same_race() {
    // The same position with Law on the board. Balance becomes a third route
    // for two needed symbols, so the defender now has to kill *two* of three
    // rather than one of two — and the Law route costs them three turns
    // because they have no pair of their own to claim it with.
    //
    //   P(kill Sundial) = P(kill Gyroscope) = 0.308, P(kill Balance) = 0.5^3 = 0.125
    //   P(stopped) = P(at least 2 of the three) = 0.148
    //   M = 0.852
    let with_law = four_symbols_late_in(2)
        .board_tokens(&["law", "philosophy"])
        .build();
    let r = science_read(&with_law, Player::One);

    assert!(r.pair_setup.has_live_half_pair(), "the p_law premise");
    assert!(r.availability[Science::Balance.index()].via_law_board);
    assert_eq!(r.obtainable_missing, 3);
    assert_eq!(r.detail.slack, 1, "one spare route now");
    assert!(r.magnitude >= 0.65, "M = {}", r.magnitude);

    let without = science_read(&four_symbols_late_in(2).build(), Player::One);
    assert!(
        r.magnitude > without.magnitude,
        "{} should beat {}",
        r.magnitude,
        without.magnitude
    );
}

#[test]
fn age_one_and_age_two_are_equally_threatening_at_the_same_symbol_count() {
    // Kristian's call, pinned as a regression: holding four symbols earlier is
    // *not* a bigger threat at the same count. Age I's extra practical value
    // is progress-token leverage — more turns in which to use the tokens a
    // pair claims — and that lives in the pair / token term, not in `M`.
    //
    // The equality is not a special case in the code; it falls out of the
    // model, because the two missing symbols are Age-III-only either way and
    // so carry the same `p_dealt`. The only thing that differs is the decision
    // budget, and two symbols fit comfortably inside both.
    let one = science_read(&four_symbols_late_in(1).build(), Player::One);
    let two = science_read(&four_symbols_late_in(2).build(), Player::One);

    assert_eq!(
        one.magnitude.to_bits(),
        two.magnitude.to_bits(),
        "Age I {} vs Age II {}",
        one.magnitude,
        two.magnitude
    );
    assert!((one.magnitude - two.magnitude).abs() < 1e-9);
    // The budgets really are different, so the equality is not vacuous.
    let ctx1 = Context::of(&four_symbols_late_in(1).build());
    let ctx2 = Context::of(&four_symbols_late_in(2).build());
    assert!(
        ctx1.tempo(Player::One).decisions_left_eff
            > ctx2.tempo(Player::One).decisions_left_eff + 5.0,
        "the two positions should differ in tempo: {} vs {}",
        ctx1.tempo(Player::One).decisions_left_eff,
        ctx2.tempo(Player::One).decisions_left_eff
    );
}

// ---------------------------------------------------------------------------
// 4: a dead race costs the defender nothing
// ---------------------------------------------------------------------------

#[test]
fn a_dead_science_race_is_worth_no_denial_at_all() {
    // Player One holds four symbols in Age III, but the two they lack —
    // Pendulum and Wheel — exist only on Age I and Age II cards, which are
    // all either in a city or boxed. Balance has no Law token anywhere. Two
    // needed, none obtainable.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "palace"), (19, "town-hall")])
        .built(
            Player::One,
            &["pharmacist", "scriptorium", "academy", "university"],
        )
        .conflict(-8)
        .coins(Player::One, 40)
        .coins(Player::Two, 40)
        .current(Player::Two)
        .build();

    let r = science_read(&st, Player::One);
    assert_eq!(r.distinct, 4);
    assert_eq!(r.missing, 2);
    assert_eq!(r.availability[Science::Pendulum.index()].gone, 2);
    assert_eq!(r.availability[Science::Wheel.index()].gone, 2);
    assert_eq!(r.obtainable_missing, 0);
    assert!(r.dead);
    assert_eq!(r.magnitude, 0.0);
    assert_eq!(r.status, ScienceStatus::Closed);

    // Player Two is on move. Nothing they can do changes a dead race, so the
    // denial channel is silent and every prior is exactly its
    // victory-point-only value.
    let s = stance(&st, Player::Two);
    let w = PriorWeights::default();
    for a in engine::legal_actions(&st) {
        let d = delta_m(a, &s);
        assert_eq!(d.science, 0.0, "{a:?} moved a dead science race");
        assert_eq!(
            d.military, 0.0,
            "{a:?} moved a military race that is nine steps out of reach"
        );
        let vp_only = w.base + w.vp * action_vp_value(&st, Player::Two, a);
        let got = action_prior(&st, a, &s);
        assert!(
            (got - vp_only).abs() < 1e-9 || got >= w.dominating,
            "{a:?}: prior {got} is neither the VP-only {vp_only} nor promoted"
        );
    }
}

// ---------------------------------------------------------------------------
// 5: the Mausoleum retrieves exactly one card
// ---------------------------------------------------------------------------

/// Player One holds four symbols; the two they lack that still exist anywhere
/// (Wheel and Sundial) are each down to a single copy, and **both** of those
/// copies are in the discard pile. One unbuilt, affordable Mausoleum can
/// retrieve one of them, not two.
fn two_mausoleum_only_symbols() -> StateBuilder {
    StateBuilder::new()
        .age(3)
        .open_slots(&[
            (15, "palace"),
            (16, "town-hall"),
            (17, "obelisk"),
            (18, "senate"),
            (19, "gardens"),
        ])
        .built(
            Player::One,
            &["pharmacist", "workshop", "scriptorium", "university"],
        )
        .built(Player::Two, &["study", "school"])
        .discard(&["apothecary", "academy"])
        .wonders(Player::One, &["the-mausoleum"])
        .coins(Player::One, 40)
        .current(Player::One)
}

#[test]
fn two_symbols_in_the_discard_pile_share_one_mausoleum() {
    // This is the bug the first cut of this crate had: counting each
    // discard-pile symbol as an independent route made a dead race look live.
    let st = two_mausoleum_only_symbols().build();
    let r = science_read(&st, Player::One);

    assert_eq!(r.distinct, 4);
    assert_eq!(r.missing, 2);
    for sym in [Science::Wheel, Science::Sundial] {
        let a = &r.availability[sym.index()];
        assert_eq!(a.via_mausoleum, 1, "{sym:?}");
        assert!(a.obtainable(), "{sym:?}");
        assert!(
            !a.obtainable_without_mausoleum(),
            "{sym:?} must have no other route for this test to mean anything"
        );
        assert!(
            r.undeniable_route[sym.index()],
            "{sym:?} sits behind a wonder the opponent cannot touch"
        );
        assert!(r.kill_cost(sym).is_infinite());
    }

    // The old, per-symbol formula would have said two.
    let naive = duels_strategy::masks::ALL_SCIENCE
        .into_iter()
        .filter(|s| r.availability[s.index()].held == 0 && r.availability[s.index()].obtainable())
        .count();
    assert_eq!(naive, 2, "the naive count is what this test is contrasting");
    assert_eq!(
        r.obtainable_missing, 1,
        "one Mausoleum, one card: the two discard-pile symbols are one route"
    );
    assert!(
        r.dead,
        "two symbols needed and only one retrievable is not a race"
    );
    assert_eq!(r.magnitude, 0.0);
    assert_eq!(r.status, ScienceStatus::Closed);
}

#[test]
fn the_law_token_gives_the_shared_mausoleum_a_second_route_to_pair_with() {
    // The same position plus Law on the board, which Player One can claim by
    // completing the Gyroscope pair they are half-way to. Now there really are
    // two routes for two symbols.
    let st = two_mausoleum_only_symbols().board_tokens(&["law"]).build();
    let r = science_read(&st, Player::One);

    assert!(
        r.pair_setup.has_live_half_pair(),
        "the Gyroscope pair is what claims the token"
    );
    assert!(r.second_copy_obtainable[Science::Gyroscope.index()]);
    assert!(r.availability[Science::Balance.index()].via_law_board);
    assert_eq!(
        r.obtainable_missing, 2,
        "Balance plus one Mausoleum retrieval"
    );
    assert!(!r.dead);
    assert!(r.magnitude > 0.0);
}

// ---------------------------------------------------------------------------
// The Law token: on the board, set aside, or set aside behind a wonder
// ---------------------------------------------------------------------------

#[test]
fn law_on_the_board_counts_at_full_weight_with_a_live_half_pair() {
    // Five of the ten tokens go on the board at setup and five are set aside.
    // Which pile Law lands in is one of the biggest single facts about how
    // threatening a science position is, and the three cases are genuinely
    // different quantities rather than three shades of the same one.
    //
    // Case 1: Law is on the board. Player One holds Gyroscope once with its
    // second copy still out there, so completing that pair claims the token
    // outright: `p_law = 1`, undiscounted.
    let st = balance_is_the_last_symbol().board_tokens(&["law"]).build();
    let r = science_read(&st, Player::One);

    assert_eq!(r.missing, 1, "Balance is the last symbol needed");
    let a = &r.availability[Science::Balance.index()];
    assert!(a.via_law_board);
    assert!(
        !a.via_law_great_library,
        "no Great-Library uncertainty should be mixed in"
    );
    assert!(r.pair_setup.has_live_half_pair());
    assert!(
        (r.copies(Science::Balance) - 1.0).abs() < 1e-12,
        "c(Balance) = {} should be exactly p_law = 1",
        r.copies(Science::Balance)
    );
    // The defender has no pair of their own to race for it, so taking it off
    // the board costs them three turns: P(stopped) = 0.5^3.
    assert!((r.kill_cost(Science::Balance) - 3.0).abs() < 1e-12);
    assert!((r.magnitude - 0.875).abs() < 1e-9, "M = {}", r.magnitude);
    assert_eq!(r.status, ScienceStatus::Live);
}

#[test]
fn law_among_the_set_aside_tokens_with_no_great_library_is_simply_gone() {
    // Case 2: Law is one of the five tokens set aside at setup, and Player One
    // holds no wonder that draws from that pile. There is no route at all —
    // not a discounted one, none — so the symbol is gone and the race with it.
    let st = balance_is_the_last_symbol()
        .set_aside_tokens(FIVE_ASIDE)
        .build();
    let r = science_read(&st, Player::One);

    let a = &r.availability[Science::Balance.index()];
    assert!(!a.via_law_board);
    assert!(!a.via_law_great_library);
    assert!(!a.obtainable());
    assert_eq!(a.gone, 1);
    assert_eq!(
        r.copies(Science::Balance),
        0.0,
        "an absent Law token contributes nothing to c_s"
    );
    assert_eq!(r.obtainable_missing, 0);
    assert!(r.dead);
    assert_eq!(r.magnitude, 0.0);
    assert_eq!(r.status, ScienceStatus::Closed);
}

#[test]
fn law_behind_an_unbuilt_great_library_counts_at_the_draw_odds_and_no_more() {
    // Case 3: Law is set aside, but Player One holds an unbuilt, affordable
    // Great Library, which draws three of the set-aside pile at random. That
    // is a real route and a genuinely smaller one than case 1: three of five.
    let st = balance_is_the_last_symbol()
        .set_aside_tokens(FIVE_ASIDE)
        .wonders(Player::One, &["the-great-library"])
        .build();
    let r = science_read(&st, Player::One);

    let a = &r.availability[Science::Balance.index()];
    assert!(!a.via_law_board);
    assert!(a.via_law_great_library);
    assert!(!r.dead);

    let w = ThreatWeights::default();
    let expected = w.great_library_draw / 5.0;
    assert!(
        (expected - 0.6).abs() < 1e-12,
        "three of the five set-aside tokens"
    );
    assert!(
        (r.copies(Science::Balance) - expected).abs() < 1e-12,
        "c(Balance) = {} should be exactly 3/|set aside| = {expected}",
        r.copies(Science::Balance)
    );
    // Nobody can take a card out of a random draw, so the *defender* has no
    // answer — but the draw itself may simply miss.
    assert!(r.kill_cost(Science::Balance).is_infinite());
    assert!(r.undeniable_route[Science::Balance.index()]);
    assert!(
        (r.magnitude - 0.6).abs() < 1e-12,
        "M = {} is exactly the draw odds",
        r.magnitude
    );
    assert_eq!(r.status, ScienceStatus::Live);

    // Strictly smaller than a Law token you can simply pick up.
    let on_board = science_read(
        &balance_is_the_last_symbol().board_tokens(&["law"]).build(),
        Player::One,
    );
    assert!(
        r.copies(Science::Balance) < on_board.copies(Science::Balance),
        "{} should be below {}",
        r.copies(Science::Balance),
        on_board.copies(Science::Balance)
    );
    assert!(
        r.magnitude < on_board.magnitude,
        "{} should be below {}",
        r.magnitude,
        on_board.magnitude
    );

    // And the pile size is not a constant: a smaller set-aside pile is better
    // odds, which is the ratio actually being read rather than a magic number.
    let three_aside = balance_is_the_last_symbol()
        .set_aside_tokens(&["law", "philosophy", "agriculture"])
        .wonders(Player::One, &["the-great-library"])
        .build();
    let smaller = science_read(&three_aside, Player::One);
    assert!(
        (smaller.copies(Science::Balance) - 1.0).abs() < 1e-12,
        "three of three is a certainty: c = {}",
        smaller.copies(Science::Balance)
    );
}

#[test]
fn the_three_law_cases_are_ordered_and_the_middle_one_is_not_a_race() {
    // The whole point, in one assertion: when Balance is the last symbol a
    // player needs, where the Law token sits decides whether they are winning,
    // gambling, or finished.
    let on_board = science_read(
        &balance_is_the_last_symbol().board_tokens(&["law"]).build(),
        Player::One,
    );
    let aside = science_read(
        &balance_is_the_last_symbol()
            .set_aside_tokens(FIVE_ASIDE)
            .build(),
        Player::One,
    );
    let aside_with_library = science_read(
        &balance_is_the_last_symbol()
            .set_aside_tokens(FIVE_ASIDE)
            .wonders(Player::One, &["the-great-library"])
            .build(),
        Player::One,
    );

    assert!(
        on_board.magnitude > aside_with_library.magnitude
            && aside_with_library.magnitude > aside.magnitude,
        "expected on-board {} > via-library {} > absent {}",
        on_board.magnitude,
        aside_with_library.magnitude,
        aside.magnitude
    );
    assert_eq!(on_board.status, ScienceStatus::Live);
    assert_eq!(aside_with_library.status, ScienceStatus::Live);
    assert_eq!(aside.status, ScienceStatus::Closed);
    assert!(aside.dead && !on_board.dead && !aside_with_library.dead);
}

// ---------------------------------------------------------------------------
// 6: extra turns reach behind a covering card
// ---------------------------------------------------------------------------

/// Age III slot 15 is covered by slot 18 and nothing else, and slot 18 is in
/// the bottom row, so it is accessible. Taking 18 uncovers 15 — and 15 is in a
/// face-up row of the real Age III pattern, so its identity is public.
///
/// That geometry is asserted against `layout::layout(3)` by
/// [`the_age_three_geometry_this_file_relies_on`].
fn sixth_symbol_one_card_deep(p1_wonders: &[&str], p2_wonders: &[&str]) -> GameState {
    StateBuilder::new()
        .age(3)
        .open_slots(&[(15, "academy"), (18, "palace"), (19, "town-hall")])
        .built(Player::One, FIVE_SYMBOLS)
        .wonders(Player::One, p1_wonders)
        .wonders(Player::Two, p2_wonders)
        .coins(Player::One, 40)
        .coins(Player::Two, 40)
        .current(Player::One)
        .build()
}

#[test]
fn the_age_three_geometry_this_file_relies_on() {
    let l = duels_core::layout::layout(3);
    assert_eq!(l.covered_by[15], 1 << 18, "slot 15 is covered only by 18");
    assert_eq!(l.covered_by[17], 1 << 19, "slot 17 is covered only by 19");
    assert_eq!(
        l.covered_by[16],
        (1 << 18) | (1 << 19),
        "slot 16 is covered by both"
    );
    assert_eq!(l.covered_by[18], 0);
    assert_eq!(l.covered_by[19], 0);
}

#[test]
fn a_play_again_wonder_reaches_the_sixth_symbol_behind_a_covering_card() {
    // Player One needs Sundial, and the only copy in play sits one card deep.
    // With The Sphinx affordable they get two actions in a row: spend the
    // covering card on the wonder, then take the Academy the extra turn
    // uncovers. Nothing the opponent does in between — because there is no in
    // between.
    let st = sixth_symbol_one_card_deep(&["the-sphinx"], &[]);
    let ctx = Context::of(&st);
    assert_eq!(ctx.tempo(Player::One).chain, 1, "one chained extra turn");

    let r = science_read_with(&st, Player::One, &ctx);
    assert_eq!(r.missing, 1);
    assert_ne!(
        r.reachable_slots & (1 << 15),
        0,
        "slot 15 should be reachable through the extra turn"
    );
    assert_eq!(r.detail.secured, Some(Science::Sundial));
    assert_eq!(r.magnitude, 1.0);
    assert_eq!(r.status, ScienceStatus::Imminent);

    // ...and the prior promotes the move that *starts* the close. There is no
    // closing slot here — the Academy is covered — so the winning move is the
    // wonder that buys the second action, and a rail that only looked at
    // closing slots would rank it below a plain build.
    let s = stance(&st, Player::One);
    assert!(s.can_close_now, "a certain close should read as closeable");
    let sphinx = WonderId::from_slug("the-sphinx").unwrap();
    assert_ne!(
        s.chain_close_wonders & (1u16 << sphinx.index()),
        0,
        "The Sphinx is the first half of the win"
    );
    let start = action_prior(
        &st,
        Action::BuildWonder {
            slot: 18,
            wonder: sphinx,
        },
        &s,
    );
    let plain = action_prior(&st, Action::Build { slot: 18 }, &s);
    assert!(
        start > plain * 10.0,
        "the chained close {start} should dominate a plain build {plain}"
    );

    // Without the wonder there is no chain, the card is out of reach this
    // turn, and the race is back to a contest.
    let no_wonder = sixth_symbol_one_card_deep(&[], &[]);
    let ctx2 = Context::of(&no_wonder);
    assert_eq!(ctx2.tempo(Player::One).chain, 0);
    let r2 = science_read_with(&no_wonder, Player::One, &ctx2);
    assert_eq!(r2.detail.secured, None);
    assert!(r2.magnitude < 1.0, "M = {}", r2.magnitude);
}

#[test]
fn a_defender_without_an_extra_turn_of_their_own_cannot_stop_it() {
    // Player Two is on move against that same chained close. Every move they
    // have either takes a card Player One did not need, or *uncovers* the
    // Academy for them. Nothing lowers the magnitude.
    let st = sixth_symbol_one_card_deep(&["the-sphinx"], &[]);
    let mut two = st;
    duels_core::testing::set_current_player(&mut two, Player::Two);

    let s = stance(&two, Player::Two);
    assert_eq!(s.opponent_science.magnitude, 1.0, "the threat is certain");
    let legal = engine::legal_actions(&two);
    assert!(legal.len() >= 4, "the position should offer real choices");
    for a in legal {
        let d = delta_m(a, &s);
        assert!(
            d.science <= 0.0,
            "{a:?} should not be able to deny a chained close: dM = {}",
            d.science
        );
        assert!(
            !d.breaks_certainty,
            "{a:?} should not read as breaking a certain win"
        );
    }
}

#[test]
fn a_defender_with_their_own_play_again_wonder_can_reach_it_and_gets_promoted() {
    // ...unless the defender has a chain of their own. Player Two spends the
    // covering card on Piraeus, and the extra turn takes the Academy that
    // uncovers before Player One ever sees it.
    let st = sixth_symbol_one_card_deep(&["the-sphinx"], &["piraeus"]);
    let mut two = st;
    duels_core::testing::set_current_player(&mut two, Player::Two);

    let s = stance(&two, Player::Two);
    assert_eq!(s.opponent_science.magnitude, 1.0);

    let piraeus = WonderId::from_slug("piraeus").unwrap();
    let chained = Action::BuildWonder {
        slot: 18,
        wonder: piraeus,
    };
    assert!(
        engine::legal_actions(&two).contains(&chained),
        "Piraeus should be affordable with the covering card"
    );

    let d = delta_m(chained, &s);
    assert!(
        d.science > 0.9,
        "spending slot 18 on a play-again wonder should take the Academy: dM = {}",
        d.science
    );
    assert!(d.breaks_certainty);

    // ...and the prior promotes it above every move that does not.
    let promoted = action_prior(&two, chained, &s);
    let ignore = action_prior(&two, Action::Discard { slot: 19 }, &s);
    assert!(
        promoted > ignore * 10.0,
        "denial {promoted} should dominate {ignore}"
    );
    assert!(promoted >= PriorWeights::default().dominating);
}

// ---------------------------------------------------------------------------
// 7: a fork is already secured, so denying half of it is worth nothing
// ---------------------------------------------------------------------------

#[test]
fn a_genuine_fork_cannot_be_denied_and_the_prior_does_not_try() {
    // Both Sundial cards are accessible and affordable to Player One, who
    // needs only that symbol. One opposing turn cannot take both, so taking
    // either one changes nothing — and the prior should not spend its
    // attention pretending otherwise.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "academy"), (19, "study"), (15, "palace")])
        .built(Player::One, FIVE_SYMBOLS)
        .coins(Player::One, 40)
        .coins(Player::Two, 40)
        .current(Player::Two)
        .build();

    let s = stance(&st, Player::Two);
    let r = &s.opponent_science;
    assert_eq!(r.missing, 1);
    assert_eq!(r.closing_slots.count_ones(), 2, "two closing cards");
    assert_eq!(r.magnitude, 1.0);

    for slot in [18u8, 19] {
        for a in [Action::Build { slot }, Action::Discard { slot }] {
            let d = delta_m(a, &s);
            assert!(
                d.science.abs() < 1e-12,
                "{a:?} on a fork should be worth nothing: dM = {}",
                d.science
            );
            assert!(!d.breaks_certainty);
            assert!(
                duels_strategy::deny_vp(a, &s).abs() < 1e-9,
                "{a:?} should carry no denial value"
            );
        }
    }

    // Taking both would work, and the model knows it: apply the two deltas in
    // sequence and the certainty is gone.
    let after_one = r.model.after_slot_taken(18, Some(Science::Sundial));
    assert_eq!(after_one.magnitude().value, 1.0, "one is not enough");
    let after_both = after_one.after_slot_taken(19, Some(Science::Sundial));
    assert!(
        after_both.magnitude().value < 1.0,
        "both copies gone should not still be certain"
    );
}

// ---------------------------------------------------------------------------
// 8: Theology turns a two-round military close into a one-round one
// ---------------------------------------------------------------------------

#[test]
fn theology_closes_a_four_shield_military_race_in_a_single_round() {
    // The pawn is five steps up, so four shields win. Player One holds an
    // unbuilt Colossus (two shields) and there is a two-shield Fortifications
    // accessible. Separately, neither closes.
    //
    // With Theology every wonder Player One builds grants another turn, so the
    // Colossus and the Fortifications land on the same visit: four shields,
    // one round, nothing in between for the opponent to answer.
    let position = |tokens: &[&str]| {
        StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "fortifications"), (19, "palace"), (15, "town-hall")])
            .wonders(Player::One, &["the-colossus"])
            .tokens(Player::One, tokens)
            .conflict(5)
            .coins(Player::One, 40)
            .coins(Player::Two, 3)
            .current(Player::One)
            .build()
    };

    let with_theology = position(&["theology"]);
    let ctx = Context::of(&with_theology);
    assert_eq!(
        ctx.tempo(Player::One).chain,
        1,
        "Theology makes the Colossus a play-again wonder"
    );
    let r = military_read(&with_theology, Player::One);
    assert_eq!(r.need, 4);
    assert_eq!(r.best_single, 2, "no single source is enough");
    assert_eq!(r.turns_to_close, Some(1));
    assert_eq!(r.magnitude, 1.0);
    assert_eq!(r.status, MilitaryStatus::Imminent);

    // The Colossus is two of the four shields, so it is not a *closing*
    // wonder on its own; what makes the round work is the extra turn Theology
    // attaches to it. The prior has to promote it anyway.
    let s = stance(&with_theology, Player::One);
    assert_eq!(r.closing_wonders, 0, "two shields do not close a four-gap");
    assert!(s.can_close_now);
    let colossus = WonderId::from_slug("the-colossus").unwrap();
    assert_ne!(s.chain_close_wonders & (1u16 << colossus.index()), 0);
    let start = action_prior(
        &with_theology,
        Action::BuildWonder {
            slot: 19,
            wonder: colossus,
        },
        &s,
    );
    let plain = action_prior(&with_theology, Action::Build { slot: 19 }, &s);
    assert!(
        start > plain * 10.0,
        "the chained close {start} should dominate a plain build {plain}"
    );

    // Without Theology the Colossus is just a wonder, and the second half of
    // the push has to wait a round the opponent gets to use.
    let without = position(&[]);
    let ctx2 = Context::of(&without);
    assert_eq!(ctx2.tempo(Player::One).chain, 0);
    let r2 = military_read(&without, Player::One);
    assert_eq!(r2.need, 4);
    assert_ne!(
        r2.turns_to_close,
        Some(1),
        "one round should not be enough without the extra turn"
    );
    assert!(r2.magnitude < 1.0, "M = {}", r2.magnitude);
    assert_ne!(r2.status, MilitaryStatus::Imminent);
    assert_eq!(
        stance(&without, Player::One).chain_close_wonders,
        0,
        "without the extra turn there is no chained close to promote"
    );
}

// ---------------------------------------------------------------------------
// 9: monotonicity
// ---------------------------------------------------------------------------

/// Walk a sweep in which the copies of the symbols `built` still needs are
/// taken out of reach one at a time, and return the magnitude at each step.
///
/// The greens start face up but *covered* — slots 11..14 sit under the row
/// 15..17, which sits under 18 and 19 — so their supply is certain without
/// being takeable this turn, which is what makes the sweep about supply rather
/// than about whether the very next card closes the race. Each step moves one
/// of them into the opponent's city, which is exactly where a denied card
/// goes.
fn supply_sweep(built: &[&str], greens: &[&str]) -> Vec<f64> {
    const COVERED: [u8; 4] = [11, 12, 13, 14];
    const FILLER: [(u8, &str); 5] = [
        (15, "palace"),
        (16, "town-hall"),
        (17, "obelisk"),
        (18, "senate"),
        (19, "gardens"),
    ];
    let mut out = Vec::new();
    for taken in 0..=greens.len() {
        let mut slots: Vec<(u8, &str)> = FILLER.to_vec();
        for (i, &g) in greens[taken..].iter().enumerate() {
            slots.push((COVERED[i], g));
        }
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&slots)
            .built(Player::One, built)
            .built(Player::Two, &greens[..taken])
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build();
        let r = science_read(&st, Player::One);
        assert!(
            (0.0..=1.0).contains(&r.magnitude),
            "M = {} out of range",
            r.magnitude
        );
        if r.dead {
            assert_eq!(r.magnitude, 0.0, "a dead race must be zero");
        }
        out.push(r.magnitude);
    }
    out
}

#[test]
fn losing_a_copy_of_a_needed_symbol_never_raises_the_magnitude() {
    // Not an intuition but a sweep: for a fixed holder, move the copies of the
    // symbols they need out of reach one at a time and assert the magnitude
    // never goes up.
    //
    // Two starting positions, because the interesting failure would be a term
    // that is non-monotone only where a route crosses from "several copies" to
    // "one", or from "one" to "none" — which is also where the one-Mausoleum
    // and slack-of-one rules interact.
    let cases: [(&str, Vec<&str>, Vec<&str>); 2] = [
        (
            "four early symbols, both Age III symbols to find",
            FOUR_EARLY.to_vec(),
            vec!["academy", "study", "university", "observatory"],
        ),
        (
            "five symbols, Sundial the last one",
            FIVE_SYMBOLS.to_vec(),
            vec!["academy", "study"],
        ),
    ];

    for (name, built, greens) in cases {
        let sweep = supply_sweep(&built, &greens);
        assert!(
            sweep[0] > 0.2,
            "{name}: the sweep should start from a real race, not {:?}",
            sweep
        );
        for w in sweep.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "{name}: the magnitude rose from {} to {} along {sweep:?}",
                w[0],
                w[1]
            );
        }
        assert_eq!(
            *sweep.last().expect("a non-empty sweep"),
            0.0,
            "{name}: taking every copy should end the race: {sweep:?}"
        );
        assert!(
            sweep.windows(2).any(|w| w[1] < w[0] - 1e-9),
            "{name}: nothing moved, so the test proved nothing: {sweep:?}"
        );
    }
}

#[test]
fn slugs_and_pieces_used_in_this_file_exist() {
    for slug in FOUR_EARLY.iter().chain(FIVE_SYMBOLS.iter()).chain(
        [
            "lumber-yard",
            "clay-pool",
            "sawmill",
            "brickyard",
            "palace",
            "town-hall",
            "obelisk",
            "senate",
            "gardens",
            "academy",
            "study",
            "university",
            "observatory",
            "library",
            "school",
            "fortifications",
        ]
        .iter(),
    ) {
        assert!(CardId::from_slug(slug).is_some(), "missing card {slug}");
    }
    for slug in [
        "the-sphinx",
        "piraeus",
        "the-colossus",
        "the-mausoleum",
        "the-great-library",
    ] {
        assert!(WonderId::from_slug(slug).is_some(), "missing wonder {slug}");
    }
    for slug in FIVE_ASIDE {
        assert!(TokenId::from_slug(slug).is_some(), "missing token {slug}");
    }
}
