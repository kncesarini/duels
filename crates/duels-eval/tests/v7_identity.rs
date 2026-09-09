//! The bit-identity guard for this crate's **eighth** round of work.
//!
//! Round eight changes what [`Config::default`] returns in exactly two places,
//! and [`Config::v7`] sets both of them back:
//!
//! * [`EvalWeights::win_probability_temperature`] — a new field, holding the
//!   temperature [`duels_eval::win_probability`] divides by. Round seven read
//!   that off a module constant, so `v7()` carries the constant's
//!   pre-round-eight value, [`WIN_PROBABILITY_TEMPERATURE_V7`].
//! * [`ScienceWeights::ladder`] — the top two rungs, `[.., 12, 18]` becoming
//!   `[.., 30, 54]`. `v7()` carries [`SCIENCE_LADDER_V7`].
//!
//! # Why this file is shorter than `tests/v6_identity.rs`
//!
//! Because round eight changed no *shape*. Every earlier round grew a branch,
//! a summand or a guard inside a function, so its identity test had to hold a
//! verbatim copy of that function as it stood before, and prove the new branch
//! was not taken. Round eight moved two sets of numbers and changed one
//! lookup — `win_probability` divides by a field where it used to divide by a
//! constant — so there is exactly one function to copy, and it is three lines.
//!
//! `evaluate` needs no copy at all: `terms::science_ladder` indexes
//! `w.ladder[distinct]` in round seven and in round eight alike, so restoring
//! the array restores the arithmetic operation for operation. That is asserted
//! rather than argued: `every_candidate_of_every_decision_agrees_under_v7`
//! drives whole seeded games and compares every candidate's
//! [`duels_eval::expected_value`] under `Config::v7()` against the same value
//! computed with the ladder written out as a literal, bit for bit.
//!
//! # The two halves of the claim
//!
//! `win_probability_under_v7_reproduces_round_seven_bit_for_bit` and its
//! neighbours are the identity. `the_round_eight_default_is_not_a_no_op` is the
//! other half — without it this file would be asserting that a pile of dead
//! code is dead.

use duels_core::testing::StateBuilder;
use duels_core::{engine, GameState, Player};
use duels_eval::{
    evaluate, expected_value, win_probability, Config, EvalWeights, Root, ScienceWeights,
    SCIENCE_LADDER_V7, WIN_PROBABILITY_TEMPERATURE_V7,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Round seven's `win_probability`, verbatim: the module constants of the day,
/// written out as literals so that a later edit to
/// [`duels_eval::WIN_PROBABILITY_TEMPERATURE_V7`] cannot make this test agree
/// with itself by accident.
fn v7_win_probability(state: &GameState, me: Player, root: &Root) -> f64 {
    let t = match state.age() {
        1 => 47.57,
        2 => 43.75,
        _ => 25.18,
    };
    1.0 / (1.0 + (-evaluate(state, me, root) / t).exp())
}

/// Whole seeded games, driven by a cheap deterministic policy, so the
/// comparison runs over real positions from all three ages rather than over
/// hand-built ones.
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

#[test]
fn v7_pins_the_round_seven_science_ladder() {
    assert_eq!(
        Config::v7().eval.science.ladder,
        [0.0, 1.0, 2.5, 6.0, 12.0, 18.0],
        "v7 must carry the ladder rounds one through seven all shipped"
    );
    assert_eq!(SCIENCE_LADDER_V7, [0.0, 1.0, 2.5, 6.0, 12.0, 18.0]);
}

#[test]
fn v7_pins_the_round_seven_leaf_temperature() {
    assert_eq!(
        Config::v7().eval.win_probability_temperature,
        [47.57, 43.75, 25.18],
        "v7 must carry the temperature round seven's win_probability divided by"
    );
    assert_eq!(WIN_PROBABILITY_TEMPERATURE_V7, [47.57, 43.75, 25.18]);
    // ...and the lookup has to agree with the free function it replaced,
    // including the out-of-range arm that reads an impossible age as Age III.
    let e = Config::v7().eval;
    assert_eq!(
        e.win_probability_temperature(1).to_bits(),
        47.57f64.to_bits()
    );
    assert_eq!(
        e.win_probability_temperature(2).to_bits(),
        43.75f64.to_bits()
    );
    assert_eq!(
        e.win_probability_temperature(3).to_bits(),
        25.18f64.to_bits()
    );
    assert_eq!(
        e.win_probability_temperature(9).to_bits(),
        25.18f64.to_bits()
    );
}

#[test]
fn win_probability_under_v7_reproduces_round_seven_bit_for_bit() {
    let cfg = Config::v7();
    let mut checked = 0u32;
    let mut ages = [false; 3];
    for seed in 0..24u64 {
        walk(seed, |state| {
            ages[usize::from(state.age().max(1)) - 1] = true;
            for p in Player::ALL {
                let root = Root::new(state, p, cfg);
                assert_eq!(
                    win_probability(state, p, &root).to_bits(),
                    v7_win_probability(state, p, &root).to_bits(),
                    "win_probability under v7 diverged from round seven's"
                );
                checked += 1;
            }
        });
    }
    assert!(checked > 2000, "only {checked} positions compared");
    assert!(ages.iter().all(|&seen| seen), "not every age was reached");
}

/// The ladder restoration is what makes `evaluate` itself identical, so it is
/// checked over every candidate of every decision rather than asserted about
/// the array alone.
#[test]
fn every_candidate_of_every_decision_agrees_under_v7() {
    let literal = Config {
        eval: EvalWeights {
            science: ScienceWeights {
                ladder: [0.0, 1.0, 2.5, 6.0, 12.0, 18.0],
                ..Config::v7().eval.science
            },
            ..Config::v7().eval
        },
        ..Config::v7()
    };
    let mut checked = 0u32;
    for seed in 0..16u64 {
        walk(seed, |state| {
            let me = state.current_player();
            let a = Root::new(state, me, Config::v7());
            let b = Root::new(state, me, literal);
            for action in engine::legal_actions(state) {
                assert_eq!(
                    expected_value(state, action, me, &a).to_bits(),
                    expected_value(state, action, me, &b).to_bits(),
                    "v7 and the round-seven ladder written out disagree on {action:?}"
                );
                checked += 1;
            }
        });
    }
    assert!(checked > 5000, "only {checked} candidates compared");
}

/// Without this the file above would be asserting that a pile of dead code is
/// dead: round eight has to *change* something, in both channels.
#[test]
fn the_round_eight_default_is_not_a_no_op() {
    let d = Config::default();
    assert_ne!(
        d.eval.science.ladder, SCIENCE_LADDER_V7,
        "round eight left the science ladder where round seven had it"
    );
    assert_ne!(
        d.eval.win_probability_temperature, WIN_PROBABILITY_TEMPERATURE_V7,
        "round eight left the leaf temperature where round seven had it"
    );
    // The temperature change has to move a real position's win probability,
    // and it moves every position that is not exactly balanced, so ordinary
    // self-play positions are the right sample for it.
    let mut moved_probability = false;
    for seed in 0..8u64 {
        walk(seed, |state| {
            let me = state.current_player();
            let v7 = Root::new(state, me, Config::v7());
            let now = Root::new(state, me, d);
            if win_probability(state, me, &v7).to_bits()
                != win_probability(state, me, &now).to_bits()
            {
                moved_probability = true;
            }
        });
    }
    assert!(
        moved_probability,
        "the round-eight temperature never changed a win probability"
    );

    // The **ladder** change needs a hand-built position, and that is the
    // round's whole point rather than a convenience: the two rungs it moves are
    // at four and five distinct symbols, and no `phased`-quality policy walks
    // into one of those by accident. `duels_core::testing::StateBuilder` is how
    // this repository builds a position it needs rather than one it can reach.
    let four = StateBuilder::new()
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
        four.player(Player::One).distinct_science(),
        4,
        "test setup: this city is meant to hold exactly four distinct symbols"
    );
    let me = four.current_player();
    let under_v7 = evaluate(&four, me, &Root::new(&four, me, Config::v7()));
    let under_now = evaluate(&four, me, &Root::new(&four, me, d));
    assert!(
        under_now > under_v7,
        "the round-eight ladder did not raise a four-symbol position: \
         {under_now} against {under_v7}"
    );
}

#[test]
fn the_older_snapshots_still_switch_off_everything_newer() {
    for older in [
        Config::v1(),
        Config::v2(),
        Config::v3(),
        Config::v4(),
        Config::v5(),
        Config::v6(),
    ] {
        assert_eq!(older.eval.science.ladder, SCIENCE_LADDER_V7);
        assert_eq!(
            older.eval.win_probability_temperature,
            WIN_PROBABILITY_TEMPERATURE_V7
        );
    }
}

#[test]
fn the_generation_chain_is_still_a_chain_of_distinct_configurations() {
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
    // `v9` is the newest link, defined as a delta from the default, so this is
    // a tautology and is here to fail loudly if a later round ever re-points it
    // without adding `v10`. It read `assert_eq!(Config::v8(), ...)` until round
    // ten moved the default and added `v9`, which is the contract under
    // `Config::v9` working as intended.
    assert_ne!(Config::v9(), Config::default());
    // `v9` is left out of the loop above because round nine adopted neither of
    // the two options it built, so its snapshot really is round eight's.
    assert_eq!(Config::v8(), Config::v9());
    // ...and `params_string` has to be able to tell the two apart, or two
    // results files from either side of this round would be indistinguishable.
    assert_ne!(
        Config::v7().params_string(),
        Config::default().params_string()
    );
}
