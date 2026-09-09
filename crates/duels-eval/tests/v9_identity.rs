//! The bit-identity guard for this crate's **tenth** round of work.
//!
//! Round ten changes what [`Config::default`] returns in exactly one place,
//! and [`Config::v9`] sets it back:
//!
//! * [`duels_eval::MenuWeights::lambda`] — the weight on the opponent-menu
//!   term, `0.6` becoming the fitted `0.408`. `v9()` carries
//!   [`MENU_LAMBDA_V9`].
//!
//! # Why this file is shorter than `tests/v7_identity.rs`
//!
//! Because round ten changed no *shape*, and — unlike round eight — not even a
//! lookup. Round eight moved two sets of numbers *and* made
//! `win_probability` divide by a field where it used to divide by a constant,
//! so it had one three-line function to copy verbatim. Round ten moves a
//! single scalar that was already a field, read in already-existing
//! multiplications.
//!
//! It is worth being precise about the one way a scalar *could* have changed a
//! shape here, because [`duels_eval::MenuWeights::lambda`] is not only a
//! weight: three places in this crate branch on `menu.lambda == 0.0` to skip
//! building the menu tables at all. Both the old value and the new one are
//! non-zero, so every one of those gates takes the same arm it took in round
//! nine. [`the_round_ten_default_does_not_cross_the_lambda_zero_gate`] asserts
//! that rather than arguing it.
//!
//! So the identity proof is the same one round eight's ladder got: restore the
//! scalar and every arithmetic operation is the operation round nine
//! performed, checked over every candidate of every decision of whole seeded
//! games by [`every_candidate_of_every_decision_agrees_under_v9`].
//!
//! [`the_evaluation_is_affine_in_the_menu_weight`] is the positive form of the
//! same claim, and the one worth having for a round that moves only weights:
//! the evaluation responds to `menu.lambda` linearly, so `0.408` buys exactly
//! `0.408/0.6` of the menu term round nine paid for and changes nothing else.
//!
//! # The two halves of the claim
//!
//! The tests above are the identity.
//! [`the_round_ten_default_is_not_a_no_op`] is the other half — without it
//! this file would be asserting that a pile of dead code is dead.
//!
//! # `v8` and `v9` are the same configuration, on purpose
//!
//! Round nine built two options, measured both and adopted neither, so its
//! snapshot is round eight's. [`the_generation_chain_still_has_one_link_per_round`]
//! pins that equality rather than leaving it out of the distinctness loop, and
//! `tests/round_nine_identity.rs` holds the round-nine half of the story.

use duels_core::testing::StateBuilder;
use duels_core::{engine, GameState, Player};
use duels_eval::{
    evaluate, expected_value, Config, EvalWeights, MenuWeights, Root, MENU_LAMBDA_V9,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Whole seeded games, driven by a cheap deterministic policy, so the
/// comparison runs over real positions from all three ages rather than over
/// hand-built ones. The same walk `tests/v7_identity.rs` and
/// `tests/round_nine_identity.rs` use, so the three files sample the same
/// positions.
fn walk(seed: u64, mut visit: impl FnMut(&GameState)) {
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5EED);
    while !state.is_over() {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        visit(&state);
        let pick = (state.turn() as usize * 7 + seed as usize) % legal.len();
        if engine::apply(&mut state, legal[pick], &mut rng).is_err() {
            break;
        }
    }
}

/// Round nine's menu weight, written out as a literal so that a later edit to
/// [`MENU_LAMBDA_V9`] cannot make the tests below agree with themselves by
/// accident.
fn round_nine_lambda_written_out() -> Config {
    Config {
        eval: EvalWeights {
            menu: MenuWeights {
                lambda: 0.6,
                ..Config::default().eval.menu
            },
            ..Config::default().eval
        },
        ..Config::default()
    }
}

#[test]
fn v9_pins_the_round_nine_menu_weight() {
    assert_eq!(
        Config::v9().eval.menu.lambda.to_bits(),
        0.6f64.to_bits(),
        "v9 must carry the menu weight rounds one through nine all shipped"
    );
    assert_eq!(MENU_LAMBDA_V9.to_bits(), 0.6f64.to_bits());
    // ...and nothing *else* about round nine moved, so the snapshot has to be
    // the default with exactly that one field put back.
    assert_eq!(Config::v9(), round_nine_lambda_written_out());
}

/// The scalar restoration is what makes the evaluation itself identical, so it
/// is checked over every candidate of every decision rather than asserted
/// about the field alone.
#[test]
fn every_candidate_of_every_decision_agrees_under_v9() {
    let literal = round_nine_lambda_written_out();
    let mut checked = 0u32;
    let mut ages = [false; 3];
    for seed in 0..16u64 {
        walk(seed, |state| {
            ages[usize::from(state.age().max(1)) - 1] = true;
            let me = state.current_player();
            let a = Root::new(state, me, Config::v9());
            let b = Root::new(state, me, literal);
            for action in engine::legal_actions(state) {
                assert_eq!(
                    expected_value(state, action, me, &a).to_bits(),
                    expected_value(state, action, me, &b).to_bits(),
                    "v9 and the round-nine menu weight written out disagree on {action:?}"
                );
                checked += 1;
            }
        });
    }
    assert!(checked > 5000, "only {checked} candidates compared");
    assert!(ages.iter().all(|&seen| seen), "not every age was reached");
}

/// `evaluate` too, from both seats, since [`expected_value`] scores
/// post-action states and the menu term is read on every one of them.
#[test]
fn evaluate_under_v9_reproduces_round_nine_bit_for_bit() {
    let literal = round_nine_lambda_written_out();
    let mut checked = 0u32;
    for seed in 0..24u64 {
        walk(seed, |state| {
            for p in Player::ALL {
                let a = Root::new(state, p, Config::v9());
                let b = Root::new(state, p, literal);
                assert_eq!(
                    evaluate(state, p, &a).to_bits(),
                    evaluate(state, p, &b).to_bits(),
                    "evaluate under v9 diverged from round nine's"
                );
                checked += 1;
            }
        });
    }
    assert!(checked > 2000, "only {checked} positions compared");
}

/// The one way a scalar could have changed a *shape*: three places in this
/// crate skip the menu tables entirely when `menu.lambda` is exactly zero.
/// Round ten's value has to stay on the same side of that gate as round
/// nine's, or this would be a shape change wearing a weight's clothes.
#[test]
fn the_round_ten_default_does_not_cross_the_lambda_zero_gate() {
    assert_ne!(Config::default().eval.menu.lambda, 0.0);
    assert_ne!(MENU_LAMBDA_V9, 0.0);
    // And the gate really is a gate: switching the term off has to change the
    // evaluation, which is what makes "both values are on the same side of it"
    // a claim worth making.
    let off = Config {
        eval: EvalWeights {
            menu: MenuWeights {
                lambda: 0.0,
                ..Config::default().eval.menu
            },
            ..Config::default().eval
        },
        ..Config::default()
    };
    let mut moved = 0u32;
    for seed in 0..8u64 {
        walk(seed, |state| {
            let me = state.current_player();
            let base = evaluate(state, me, &Root::new(state, me, Config::default())).to_bits();
            if evaluate(state, me, &Root::new(state, me, off)).to_bits() != base {
                moved += 1;
            }
        });
    }
    assert!(
        moved > 100,
        "only {moved} positions moved with the menu off"
    );
}

/// Without this the file above would be asserting that a pile of dead code is
/// dead: round ten has to *change* something.
#[test]
fn the_round_ten_default_is_not_a_no_op() {
    let d = Config::default();
    assert_ne!(
        d.eval.menu.lambda.to_bits(),
        MENU_LAMBDA_V9.to_bits(),
        "round ten left the menu weight where round nine had it"
    );
    assert_eq!(
        d.eval.menu.lambda.to_bits(),
        0.408f64.to_bits(),
        "round ten's fitted menu weight is 0.408"
    );
    let v9 = Config::v9();
    let mut moved = 0u32;
    for seed in 0..16u64 {
        walk(seed, |state| {
            let me = state.current_player();
            if evaluate(state, me, &Root::new(state, me, d)).to_bits()
                != evaluate(state, me, &Root::new(state, me, v9)).to_bits()
            {
                moved += 1;
            }
        });
    }
    assert!(
        moved > 500,
        "only {moved} positions moved under the round-ten default"
    );
    // ...and the *shape* of what it moves, in two hand-built positions, since
    // this is the round's whole claim and it is worth stating from both sides.
    //
    // The opponent menu is a reading of the cards on the board, so a position
    // with no structure has no menu at all and round ten cannot reach it. The
    // four-symbol city `tests/v7_identity.rs` uses for the round-eight ladder
    // is exactly such a position — `built` places cities without laying out a
    // structure — so round ten leaves it precisely where round nine had it.
    let no_structure = StateBuilder::new()
        .age(2)
        .built(
            Player::One,
            &["workshop", "apothecary", "scriptorium", "pharmacist"],
        )
        .coins(Player::One, 10)
        .coins(Player::Two, 10)
        .current(Player::One)
        .build();
    assert_eq!(
        no_structure.occupied_slots(),
        0,
        "test setup: this position is meant to have no structure to read"
    );
    let me = no_structure.current_player();
    assert_eq!(
        evaluate(&no_structure, me, &Root::new(&no_structure, me, d)).to_bits(),
        evaluate(&no_structure, me, &Root::new(&no_structure, me, v9)).to_bits(),
        "round ten moved a position with no cards for the menu to price"
    );

    // Give the same city two cards to look at and the weight bites.
    let with_structure = StateBuilder::new()
        .age(2)
        .open_slots(&[(18, "palace"), (19, "clay-pool")])
        .built(
            Player::One,
            &["workshop", "apothecary", "scriptorium", "pharmacist"],
        )
        .coins(Player::One, 10)
        .coins(Player::Two, 10)
        .current(Player::One)
        .build();
    let me = with_structure.current_player();
    assert_ne!(
        evaluate(&with_structure, me, &Root::new(&with_structure, me, d)).to_bits(),
        evaluate(&with_structure, me, &Root::new(&with_structure, me, v9)).to_bits(),
        "the round-ten weight left a position with a live menu untouched"
    );
}

/// The evaluation under an arbitrary non-zero menu weight, for the affineness
/// check below.
fn at_lambda(state: &GameState, me: Player, lambda: f64) -> f64 {
    let cfg = Config {
        eval: EvalWeights {
            menu: MenuWeights {
                lambda,
                ..Config::default().eval.menu
            },
            ..Config::default().eval
        },
        ..Config::default()
    };
    evaluate(state, me, &Root::new(state, me, cfg))
}

/// Round ten moved a *weight*, and the substantive form of that claim is that
/// the evaluation is **affine** in `menu.lambda` — one linear term scaled and
/// nothing else touched, so `0.408` buys exactly `0.408/0.6` of what round
/// nine paid for the menu and leaves every other term alone. Three non-zero
/// probes settle it: the response to equal steps in `lambda` has to be equal.
///
/// Zero is deliberately not one of the probes. It is the one value that *does*
/// change the shape — see
/// [`the_round_ten_default_does_not_cross_the_lambda_zero_gate`] — so an
/// affineness check that included it would be measuring the gate instead of
/// the weight.
#[test]
fn the_evaluation_is_affine_in_the_menu_weight() {
    // Two steps of 0.192: 0.408 (the round-ten default), 0.6 (round nine's)
    // and 0.792.
    let (a, b, c) = (0.408, MENU_LAMBDA_V9, MENU_LAMBDA_V9 + 0.192);
    let mut checked = 0u32;
    let mut live = 0u32;
    for seed in 0..12u64 {
        walk(seed, |state| {
            for p in Player::ALL {
                let (ea, eb, ec) = (
                    at_lambda(state, p, a),
                    at_lambda(state, p, b),
                    at_lambda(state, p, c),
                );
                let first = eb - ea;
                let second = ec - eb;
                // A position whose menu is worthless responds to neither step;
                // every other one has to respond equally to both. Relative to
                // the step size, floored at one victory point so a tiny
                // response is not held to an unreachable absolute tolerance.
                let scale = first.abs().max(second.abs()).max(1.0);
                assert!(
                    (second - first).abs() / scale < 1e-9,
                    "the evaluation is not affine in menu.lambda: steps of \
                     {first} then {second}"
                );
                if first.abs() > 1e-9 {
                    live += 1;
                }
                checked += 1;
            }
        });
    }
    assert!(checked > 1500, "only {checked} positions compared");
    assert!(
        live > 500,
        "only {live} positions responded to the menu weight at all, so the \
         affineness above is mostly vacuous"
    );
}

#[test]
fn the_older_snapshots_still_switch_off_everything_newer() {
    // `v1` is the one exception, and it is the term's own origin: round one
    // predates the opponent menu entirely, so its weight is zero rather than
    // round nine's 0.6.
    assert_eq!(Config::v1().eval.menu.lambda.to_bits(), 0.0f64.to_bits());
    for (name, older) in [
        ("v2", Config::v2()),
        ("v3", Config::v3()),
        ("v4", Config::v4()),
        ("v5", Config::v5()),
        ("v6", Config::v6()),
        ("v7", Config::v7()),
        ("v8", Config::v8()),
    ] {
        assert_eq!(
            older.eval.menu.lambda.to_bits(),
            MENU_LAMBDA_V9.to_bits(),
            "{name} carries round ten's menu weight instead of round nine's"
        );
    }
}

#[test]
fn the_generation_chain_still_has_one_link_per_round() {
    let chain = [
        ("v1", Config::v1()),
        ("v2", Config::v2()),
        ("v3", Config::v3()),
        ("v4", Config::v4()),
        ("v5", Config::v5()),
        ("v6", Config::v6()),
        ("v7", Config::v7()),
        ("v8", Config::v8()),
    ];
    for i in 0..chain.len() {
        for j in i + 1..chain.len() {
            assert_ne!(
                chain[i].1, chain[j].1,
                "{} and {} are the same configuration",
                chain[i].0, chain[j].0
            );
        }
    }
    // `v9` is left out of the loop deliberately: round nine adopted neither of
    // the options it built, so its snapshot *is* round eight's. Pinned as an
    // equality so that a later edit which pulls them apart fails here rather
    // than quietly invalidating the round-nine narrative.
    assert_eq!(Config::v8(), Config::v9());
    // `v9` is *defined* as a delta from the default, so this is a tautology
    // and is here to fail loudly if a later round ever re-points it without
    // adding `v10`.
    assert_ne!(Config::v9(), Config::default());
    // ...and `params_string` has to be able to tell the two apart, or two
    // results files from either side of this round would be indistinguishable.
    assert_ne!(
        Config::v9().params_string(),
        Config::default().params_string()
    );
}
