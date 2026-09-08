//! The bit-identity guard for this crate's **ninth** round of work.
//!
//! # Why this is not `tests/v8_identity.rs`
//!
//! Every previous round of this crate moved [`Config::default`], so each one
//! left behind a `tests/vN_identity.rs` proving that the snapshot before it
//! reproduced the arithmetic it had been measured under. **Round nine did not
//! move the default.** It added two options and measured both, and both
//! measurements said to leave the default where it was:
//!
//! * [`duels_eval::WonderModel::Rationed`] with
//!   [`duels_eval::EvalWeights::wonder_potential`] at `1.25` is worth about
//!   **+30 Elo to `phased`** on five disjoint seed ranges and **−24.4 to
//!   `mcts-eval`**. `mcts-eval` is the consumer that decides.
//! * [`duels_eval::ReachModel::Structure`] is neutral to `phased` (+1.1 /
//!   +2.6) and moves the calibration it was built for by under a victory
//!   point.
//!
//! So there is no new generation, `Config::v8()` is still the default, and what
//! this file has to prove is the *other* identity: that the two new options
//! really are off, and that the one place round nine changed a function's
//! **shape** is bit-identical on the branch it kept.
//!
//! # What needs a verbatim copy, and what does not
//!
//! [`duels_eval::WonderModel::Rationed`] is a new `match` arm and
//! [`duels_eval::EvalWeights::wonder_p_build_ref`] a new field read only
//! inside it, so with `Flat` in force nothing reaches either — that is the
//! kind of claim [`the_two_round_nine_options_are_off_and_change_nothing`]
//! settles by driving whole seeded games rather than by reading the code.
//!
//! The reachability walk **is** a shape change and does get a copy. Round
//! eight's `terms::supremacy_live` walked the symbol list with one expression;
//! round nine's [`duels_eval::terms::supremacy_live_with`] chooses between two,
//! and `Optimistic` has to be the old one exactly. [`v8_supremacy_live`] below
//! is that walk verbatim, and it is compared over every position of real games
//! — which is the whole identity proof for `terms::science_ladder` too, since
//! that function's only round-nine edit is which predicate it consults.
//!
//! [`the_round_nine_options_are_not_no_ops`] is the other half of the claim:
//! without it this file would be asserting that a pile of dead code is dead.

use duels_core::data;
use duels_core::testing::StateBuilder;
use duels_core::{engine, GameState, Player};
use duels_eval::terms::{self, SYMBOLS_TO_WIN};
use duels_eval::{
    evaluate, expected_value, Config, EvalWeights, ReachModel, Root, ScienceWeights, WonderModel,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Round eight's `terms::supremacy_live`, verbatim — the walk and its early
/// exit, with the reachability expression as it stood before round nine gave
/// it an alternative.
fn v8_supremacy_live(state: &GameState, p: Player) -> bool {
    let budget = duels_strategy::masks::ALL_SCIENCE.len() as u8 - SYMBOLS_TO_WIN;
    let mut missing = 0u8;
    let m = duels_strategy::masks::masks();
    let held = state.player(p).science();
    let gone = state.player(Player::One).built_mask()
        | state.player(Player::Two).built_mask()
        | state.wonder_fodder_mask()
        | state.discard_mask();
    let age = state.age().max(1);
    for sym in duels_strategy::masks::ALL_SCIENCE {
        let reachable = if held[sym.index()] > 0 {
            true
        } else if Some(sym) == m.law_symbol() {
            m.law_token()
                .is_some_and(|law| state.board_tokens().any(|t| t == law))
        } else {
            duels_strategy::masks::iter_cards(m.symbol_mask(sym))
                .any(|c| gone & (1u128 << c.index()) == 0 && c.def().age >= age)
        };
        if !reachable {
            missing += 1;
        }
        if missing > budget {
            return false;
        }
    }
    missing <= budget
}

/// Whole seeded games, driven by a cheap deterministic policy, so the
/// comparisons run over real positions from all three ages rather than over
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

/// The two round-nine options, and the field only one of them reads, at the
/// values that make them inert.
fn round_eight_values() -> Config {
    Config {
        eval: EvalWeights {
            wonder_potential: 0.5,
            wonder_p_build_ref: 1.0,
            science: ScienceWeights {
                reach_model: ReachModel::Optimistic,
                ..Config::default().eval.science
            },
            ..Config::default().eval
        },
        wonder_model: WonderModel::Flat,
        ..Config::default()
    }
}

/// The default carries the round-eight values, and `v8()` is still the
/// newest link in the generation chain.
#[test]
fn the_default_is_still_the_round_eight_configuration() {
    let d = Config::default();
    assert_eq!(d.wonder_model, WonderModel::Flat);
    assert_eq!(d.eval.wonder_potential.to_bits(), 0.5f64.to_bits());
    assert_eq!(d.eval.wonder_p_build_ref.to_bits(), 1.0f64.to_bits());
    assert_eq!(d.eval.science.reach_model, ReachModel::Optimistic);
    // ...and the round-eight numbers underneath are untouched, so "round nine
    // moved nothing" is a claim about the whole configuration and not only
    // about the two fields it added.
    assert_eq!(d.eval.science.ladder, [0.0, 1.0, 2.5, 6.0, 30.0, 54.0]);
    assert_eq!(
        d.eval.win_probability_temperature,
        [
            duels_eval::WIN_PROBABILITY_TEMPERATURE_AGE_I,
            duels_eval::WIN_PROBABILITY_TEMPERATURE_AGE_II,
            duels_eval::WIN_PROBABILITY_TEMPERATURE_AGE_III,
        ]
    );
    assert_eq!(d, round_eight_values());
    assert_eq!(Config::v8(), d);
}

/// The `Optimistic` reachability model has to be round eight's walk exactly,
/// because [`terms::science_ladder`]'s only round-nine edit is which predicate
/// it consults.
#[test]
fn the_optimistic_reach_model_is_round_eights_walk_bit_for_bit() {
    let mut checked = 0u32;
    let mut ages = [false; 3];
    for seed in 0..32u64 {
        walk(seed, |state| {
            ages[usize::from(state.age().max(1)) - 1] = true;
            for p in Player::ALL {
                assert_eq!(
                    terms::supremacy_live_with(state, p, ReachModel::Optimistic),
                    v8_supremacy_live(state, p),
                    "the optimistic walk diverged from round eight's"
                );
                // ...and the free function still means the optimistic model,
                // which is what every earlier snapshot's arithmetic rests on.
                assert_eq!(
                    terms::supremacy_live(state, p),
                    v8_supremacy_live(state, p),
                    "terms::supremacy_live is no longer the optimistic walk"
                );
                checked += 1;
            }
        });
    }
    assert!(checked > 2000, "only {checked} positions compared");
    assert!(ages.iter().all(|&seen| seen), "not every age was reached");
}

/// The two options are off, and being off they change nothing — over every
/// candidate of every decision of real games, which is the claim rather than a
/// reading of the `match` arms.
#[test]
fn the_two_round_nine_options_are_off_and_change_nothing() {
    let literal = round_eight_values();
    // `wonder_p_build_ref` is read only inside the rationed arm, so moving it
    // has to be inert under the default too — otherwise "off" is not off.
    let ref_moved = Config {
        eval: EvalWeights {
            wonder_p_build_ref: duels_eval::terms::OPENING_P_BUILD,
            ..Config::default().eval
        },
        ..Config::default()
    };
    let mut checked = 0u32;
    for seed in 0..16u64 {
        walk(seed, |state| {
            let me = state.current_player();
            let a = Root::new(state, me, Config::default());
            let b = Root::new(state, me, literal);
            let c = Root::new(state, me, ref_moved);
            for action in engine::legal_actions(state) {
                let want = expected_value(state, action, me, &a).to_bits();
                assert_eq!(
                    expected_value(state, action, me, &b).to_bits(),
                    want,
                    "the default and the round-eight values written out disagree on {action:?}"
                );
                assert_eq!(
                    expected_value(state, action, me, &c).to_bits(),
                    want,
                    "wonder_p_build_ref is read outside the rationed model, on {action:?}"
                );
                checked += 1;
            }
        });
    }
    assert!(checked > 5000, "only {checked} candidates compared");
}

/// `p_build` is a quantity about the position, so the rationed term has to read
/// it off **the state being scored** and not off the [`Root`] it is scored
/// against.
///
/// This is the guard for the frozen-`p_build` fix. The term used to take
/// `p_build` as an argument and `Root` used to cache it, so every state scored
/// against one `Root` was priced by the root turn's wonder slots and decisions
/// left. The signature no longer allows that; what this test adds is the
/// measurement of what it was worth, over real positions rather than by
/// argument. `duels-agent-phased`'s `tests/p_build_identity.rs` is the other
/// half — the same fix seen as a change in an agent's decisions.
#[test]
fn the_rationed_term_reads_p_build_from_the_state_being_scored() {
    let e = Config::default().eval;
    let mut moves_that_move_p_build = 0u32;
    let mut worst_stale_vp = 0.0f64;
    for seed in 0..16u64 {
        walk(seed, |state| {
            let me = state.current_player();
            for action in engine::legal_actions(state) {
                let mut next = *state;
                // One arbitrary chance outcome is enough: `p_build` is built
                // from counts no card reveal can touch.
                let outcomes = engine::chance_outcomes(state, action);
                let Some((outcome, _)) = outcomes.first() else {
                    continue;
                };
                if engine::apply_with_outcome(&mut next, action, outcome).is_err() {
                    continue;
                }
                let pre = terms::wonder_p_build(state, me, &e);
                let post = terms::wonder_p_build(&next, me, &e);
                if pre.to_bits() == post.to_bits() {
                    continue;
                }
                moves_that_move_p_build += 1;
                // The term the evaluation now pays, and the one it used to pay
                // for exactly this state — the whole delta is the staleness.
                let flat = terms::wonder_potential(&next, me, &e);
                let fresh = terms::wonder_potential_rationed(&next, me, &e);
                if post == 0.0 {
                    // The guarded early exit, so a dead hand is exactly `+0.0`
                    // with no signed zero to argue about.
                    assert_eq!(fresh.to_bits(), 0.0f64.to_bits());
                } else {
                    assert_eq!(
                        fresh.to_bits(),
                        (post * flat).to_bits(),
                        "the rationed term is not p_build(post) x flat(post) on {action:?}"
                    );
                }
                worst_stale_vp = worst_stale_vp.max((fresh - pre * flat).abs());
            }
        });
    }
    assert!(
        moves_that_move_p_build > 500,
        "only {moves_that_move_p_build} candidate moves moved p_build at all, \
         so the staleness this fixes is untested"
    );
    // Not a threshold to tune — a floor far below what was observed, so the
    // test says "this was worth real victory points" and not "this differs in
    // the last bit". The largest single-move staleness seen over these games
    // is several victory points.
    assert!(
        worst_stale_vp > 1.0,
        "the worst stale read was only {worst_stale_vp} victory points"
    );
}

/// ...and both options have to *do* something when they are switched on, or
/// this file would be asserting that a pile of dead code is dead.
#[test]
fn the_round_nine_options_are_not_no_ops() {
    let rationed = Config {
        eval: EvalWeights {
            wonder_potential: 1.25,
            ..Config::default().eval
        },
        wonder_model: WonderModel::Rationed,
        ..Config::default()
    };
    let structural = Config {
        eval: EvalWeights {
            science: ScienceWeights {
                reach_model: ReachModel::Structure,
                ..Config::default().eval.science
            },
            ..Config::default().eval
        },
        ..Config::default()
    };
    let mut moved_by_rationing = 0u32;
    let mut moved_by_reach = 0u32;
    for seed in 0..24u64 {
        walk(seed, |state| {
            let me = state.current_player();
            let base = evaluate(state, me, &Root::new(state, me, Config::default())).to_bits();
            if evaluate(state, me, &Root::new(state, me, rationed)).to_bits() != base {
                moved_by_rationing += 1;
            }
            if evaluate(state, me, &Root::new(state, me, structural)).to_bits() != base {
                moved_by_reach += 1;
            }
        });
    }
    assert!(
        moved_by_rationing > 100,
        "only {moved_by_rationing} positions moved under the rationed model"
    );
    assert!(
        moved_by_reach > 0,
        "the structural reach model never moved a position"
    );
    // `params_string` has to tell each option apart from the default, or two
    // results files from either side of it would be indistinguishable — which
    // is the whole price `mcts-eval` pays for pinning no generation.
    let d = Config::default().params_string();
    assert_ne!(rationed.params_string(), d);
    assert_ne!(structural.params_string(), d);
}

/// The rationing has to be a *rationing*: exactly the flat term where every
/// unbuilt wonder is certain to be built, and strictly less where it is not.
#[test]
fn the_rationed_model_is_the_flat_one_scaled_by_p_build() {
    let e = Config::default().eval;
    let mut rationed_below_flat = 0u32;
    let mut ever_certain = false;
    for seed in 0..24u64 {
        walk(seed, |state| {
            for p in Player::ALL {
                let p_build = terms::wonder_p_build(state, p, &e);
                let flat = terms::wonder_potential(state, p, &e);
                let rationed = terms::wonder_potential_rationed(state, p, &e);
                if p_build == 0.0 {
                    // The guarded early exit, so a dead hand is exactly zero
                    // with no signed-zero or rounding to argue about.
                    assert_eq!(rationed.to_bits(), 0.0f64.to_bits());
                    continue;
                }
                assert_eq!(
                    rationed.to_bits(),
                    (p_build * flat).to_bits(),
                    "the rationed term is not the flat one times p_build"
                );
                if p_build >= 1.0 {
                    ever_certain = true;
                    assert_eq!(
                        rationed.to_bits(),
                        flat.to_bits(),
                        "a certain build is not priced at the flat term"
                    );
                } else if flat > 0.0 {
                    assert!(rationed < flat);
                    rationed_below_flat += 1;
                }
            }
        });
    }
    assert!(
        rationed_below_flat > 100,
        "only {rationed_below_flat} positions were rationed at all"
    );
    assert!(
        ever_certain,
        "no position had p_build = 1, so the agreement half is untested"
    );

    // ...and the reference divisor is a clamp, not a rescale: at
    // `OPENING_P_BUILD` the factor is one for every position whose `p_build`
    // is at least that, and the plain `p_build` below it.
    let clamped = EvalWeights {
        wonder_p_build_ref: terms::OPENING_P_BUILD,
        ..e
    };
    let mut clamped_at_one = 0u32;
    for seed in 0..8u64 {
        walk(seed, |state| {
            for p in Player::ALL {
                let p_build = terms::wonder_p_build(state, p, &e);
                let flat = terms::wonder_potential(state, p, &e);
                let got = terms::wonder_potential_rationed(state, p, &clamped);
                if p_build >= terms::OPENING_P_BUILD {
                    assert_eq!(got.to_bits(), flat.to_bits());
                    clamped_at_one += 1;
                } else if p_build > 0.0 {
                    assert_eq!(
                        got.to_bits(),
                        ((p_build / terms::OPENING_P_BUILD) * flat).to_bits()
                    );
                }
            }
        });
    }
    assert!(
        clamped_at_one > 100,
        "only {clamped_at_one} positions reached the clamp"
    );
}

/// The rationed model has to *undercut* the flat one where the seven-wonder cap
/// and the decision budget say a hand will not all be built — which is round
/// nine's whole claim, in one hand-built position.
#[test]
fn a_hand_that_cannot_be_built_is_worth_less_than_one_that_can() {
    // Four unbuilt wonders and two decisions left: `turn_factor` is
    // `2 / (4 x 2.5)` = 0.2, so four fifths of the flat credit is withdrawn.
    let late = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "arena"), (19, "palace")])
        .wonders(Player::One, &["the-pyramids", "the-colossus"])
        .coins(Player::One, 10)
        .coins(Player::Two, 10)
        .current(Player::One)
        .build();
    let e = Config::default().eval;
    let flat = terms::wonder_potential(&late, Player::One, &e);
    assert!(flat > 0.0, "test setup: this hand holds unbuilt wonders");
    let p_build = terms::wonder_p_build(&late, Player::One, &e);
    assert!(
        p_build < 1.0,
        "test setup: p_build is {p_build}, so nothing is being rationed"
    );
    assert!(terms::wonder_potential_rationed(&late, Player::One, &e) < flat);
}

/// The structural reachability model has to be a *sharpening*: it can only ever
/// call fewer symbols reachable, never more.
#[test]
fn the_structural_reach_model_never_calls_more_symbols_reachable() {
    let mut differed = 0u32;
    for seed in 0..64u64 {
        walk(seed, |state| {
            for p in Player::ALL {
                let optimistic = terms::supremacy_reachable_with(state, p, ReachModel::Optimistic);
                let structural = terms::supremacy_reachable_with(state, p, ReachModel::Structure);
                assert!(
                    structural <= optimistic,
                    "the structural model called {structural} symbols reachable \
                     where the optimistic one called {optimistic}"
                );
                if structural != optimistic {
                    differed += 1;
                }
            }
        });
    }
    assert!(
        differed > 0,
        "the structural model never disagreed, so it is not a model"
    );
}

/// ...and it has to stand down entirely while the structure is empty, because
/// `state.age()` is then an age whose cards have not been dealt.
#[test]
fn the_structural_reach_model_stands_down_with_no_structure() {
    let drafting = engine::new_game(7);
    assert_eq!(
        drafting.occupied_slots(),
        0,
        "test setup: the wonder draft has no structure laid out"
    );
    for p in Player::ALL {
        assert_eq!(
            terms::supremacy_reachable_with(&drafting, p, ReachModel::Structure),
            terms::supremacy_reachable_with(&drafting, p, ReachModel::Optimistic),
            "the structural model read an undealt age off an empty structure"
        );
    }
}

/// The face-up symbol mask is public information and says what it claims: a
/// symbol appears in it exactly when some face-up card in the structure prints
/// it.
#[test]
fn the_faceup_symbol_mask_matches_the_face_up_cards() {
    for seed in 0..16u64 {
        walk(seed, |state| {
            let mask = terms::faceup_symbols(state);
            for sym in duels_strategy::masks::ALL_SCIENCE {
                let mut expected = false;
                let mut slots = state.occupied_slots();
                while slots != 0 {
                    let slot = slots.trailing_zeros() as u8;
                    slots &= slots - 1;
                    if let Some(card) = state.face_up_card(slot) {
                        if card.def().science == Some(sym) {
                            expected = true;
                        }
                    }
                }
                assert_eq!(
                    mask & (1u8 << sym.index()) != 0,
                    expected,
                    "faceup_symbols disagrees about {sym:?}"
                );
            }
        });
    }
}

/// Balance is the one symbol no card prints, so neither reachability model may
/// ever read it off the structure — it is the Law token or nothing.
#[test]
fn balance_is_still_reachable_only_through_the_law_token() {
    let m = duels_strategy::masks::masks();
    let balance = m.law_symbol().expect("the card data prints a Law symbol");
    assert_eq!(
        duels_strategy::masks::iter_cards(m.symbol_mask(balance)).count(),
        0,
        "some card prints Balance, which both reach models assume none does"
    );
    assert_eq!(balance, data::Science::Balance);
}

/// Every generation snapshot — the whole chain, the newest link included —
/// carries round nine's options at their off values, so a round-ten default
/// move that turned one of them on would have a `v9()` one `..` away.
#[test]
fn every_snapshot_still_switches_off_everything_round_nine_added() {
    for older in [
        Config::v1(),
        Config::v2(),
        Config::v3(),
        Config::v4(),
        Config::v5(),
        Config::v6(),
        Config::v7(),
        Config::v8(),
    ] {
        assert_eq!(older.wonder_model, WonderModel::Flat);
        assert_eq!(older.eval.wonder_potential.to_bits(), 0.5f64.to_bits());
        assert_eq!(older.eval.wonder_p_build_ref.to_bits(), 1.0f64.to_bits());
        assert_eq!(older.eval.science.reach_model, ReachModel::Optimistic);
    }
}
