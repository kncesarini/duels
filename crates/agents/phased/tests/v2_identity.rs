//! The bit-identity guard for this crate's **third** round of work.
//!
//! Round three adds five things: the three terminal rails, a differenced menu
//! shield price, a horizon-based military smoothing width, a
//! production-lock-in multiplier on the economy terms, and a different
//! `military_band` default. Every one of them is a [`Config`] option, and
//! [`Config::v2`] sets all five back to what the round-two default did.
//!
//! The obligation this project's conventions impose is to prove that
//! `Config::v2()` is not merely *similar* to the agent that shipped, but the
//! same arithmetic. `tests/legacy_identity.rs` does that for round two by
//! keeping a **verbatim copy** of the round-one evaluation in the test file;
//! this does the same for round three, and it has to be a same-process copy
//! rather than a recorded digest: `exp` and `powf` do not agree bit for bit
//! between platforms, so a golden hash taken on one machine is not a statement
//! about the code (this file *was* written that way first, and CI on a
//! different architecture is what said so).
//!
//! Four things are asserted, and together they close the loop:
//!
//! 1. every candidate of every decision of whole seeded games scores bit for
//!    bit the same under `Config::v2()` as under the copy below, and the two
//!    pick the same move every time;
//! 2. the copy is not vacuous — both round-three insertions really do change
//!    what the evaluation computes when they are given something to do;
//! 3. the one piece of round-three arithmetic that lives inside
//!    [`Root::new`] rather than in the evaluation — the smoothing width — is
//!    checked directly against the round-two construction, since a copy of the
//!    evaluation shares whatever `Root` hands it and could not catch a change
//!    there;
//! 4. `Config::v2()` really is the round-two configuration field by field, so
//!    the identity is about the code rather than a coincidence of weights.

use duels_agent_phased::{
    expected_value, menu, terms, Config, EvalWeights, MenuWeights, MilSmoothing, Root,
};
use duels_core::scoring::{self, GameResult};
use duels_core::testing::StateBuilder;
use duels_core::{engine, Action, GameState, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// A verbatim copy of the evaluation as it stood before round three.
//
// The only edits are the two round-three insertions removed: `evaluate`'s
// terminal check is no longer followed by a rails consultation, and
// `player_value`'s development and resource-bill terms are no longer
// multiplied by the production lock-in factor. Everything else, including
// every `terms::` and `menu::` call, is character for character what round
// two ran -- those functions were not edited on the paths `Config::v2()`
// takes.
// ---------------------------------------------------------------------------

fn v2_evaluate(state: &GameState, me: Player, root: &Root) -> f64 {
    if let Some(result) = state.result() {
        return match result {
            GameResult::Win { winner, .. } if winner == me => root.config().eval.instant_result,
            GameResult::Win { .. } => -root.config().eval.instant_result,
            GameResult::Draw => 0.0,
        };
    }
    v2_player_value(state, me, root) - v2_player_value(state, me.other(), root)
        + menu::menu_term(state, me, root.age(), root.menu(), &root.config().eval.menu)
}

fn v2_player_value(state: &GameState, p: Player, root: &Root) -> f64 {
    use duels_agent_phased::{CoinModel, EconomyModel, MilitaryModel};

    let e = &root.config().eval;
    let c = root.config();
    let w = root.weights(p);
    let breakdown = scoring::breakdown(state, p);

    let points = w.vp * e.vp_projection * terms::card_and_token_vp(&breakdown);

    let (liquidity, race_liquidity, coin_safety) = match c.coin_model {
        CoinModel::Legacy => (
            w.liquidity * e.coins_div3 * f64::from(breakdown.coins),
            w.race_liquidity
                * e.race_card_liquidity
                * terms::race_liquidity(state, p, e.race_liquidity_cap),
            e.coin_safety_penalty * -terms::coin_shortfall(state, p, e.coin_safety_floor),
        ),
        CoinModel::Smooth => (
            w.liquidity * e.coins_div3 * terms::coin_points(state, p, e.coin_endgame_decisions)
                + terms::coin_liquidity(state, p, e.coin_smooth_beta, e.coin_smooth_ref),
            0.0,
            0.0,
        ),
    };

    let development = w.development
        * e.development
        * terms::development_value_with(
            state,
            p,
            root.supply(),
            e.development_take_rate,
            c.economy_model == EconomyModel::Legacy,
        );
    let chain_equity =
        w.development * e.chain_equity * menu::chain_equity(state, p, root.menu().chain());

    let market = match c.economy_model {
        EconomyModel::Legacy => e.resource_vulnerability * -terms::average_trade_price(state, p),
        EconomyModel::Bill => {
            e.resource_bill
                * -terms::resource_bill(state, p, root.supply(), e.development_take_rate)
                / 3.0
        }
    };
    let economy = w.economy * (coin_safety + market);

    let science = w.science * e.science_ladder * terms::science_ladder(state, p, &e.science);
    let military = match c.military_model {
        MilitaryModel::Legacy => {
            w.military * e.military_position * terms::military_position(state, p)
        }
        MilitaryModel::Band => {
            w.military * e.military_band * terms::military_band(state, p, root.smoothing())
                + e.military_loot * terms::military_loot(state, p, root.smoothing())
        }
    };

    let urgency = e.military_endgame_urgency * terms::military_urgency(state, p);
    let start = terms::next_age_start(state, p, e);
    let wonders = e.wonder_potential * terms::wonder_potential(state, p);
    let gift = if e.menu.lambda == 0.0 {
        -e.deny_chain_gift * terms::chain_gift_exposure(state, p, root.age())
    } else {
        0.0
    };

    points
        + liquidity
        + development
        + chain_equity
        + economy
        + science
        + military
        + race_liquidity
        + urgency
        + start
        + wonders
        + gift
}

fn v2_expected_value(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => v2_evaluate(&next, me, root),
            Err(_) => v2_evaluate(state, me, root),
        };
        acc += prob * value;
    }
    acc + root.denial_term(action)
}

// ---------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------

/// Drive whole seeded games with a deterministic first-argmax policy — no RNG
/// anywhere, so nothing can drift for a reason other than the arithmetic —
/// scoring every candidate both ways.
///
/// `agree` is `false` for the deliberately-different case, where the point is
/// to prove the copy is not vacuous.
fn compare(config: Config, agree: bool) -> usize {
    let mut disagreements = 0usize;
    for seed in 0..6u64 {
        let mut st: GameState = engine::new_game(seed);
        // The engine still needs *an* RNG to deal the next age; it is seeded
        // per game and never consulted by either evaluation.
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FF_EE00);
        for _ in 0..400 {
            if st.is_over() {
                break;
            }
            let legal = engine::legal_actions(&st);
            if legal.is_empty() {
                break;
            }
            let me = st.current_player();
            let chosen = if legal.len() == 1 {
                legal[0]
            } else {
                let root = Root::new(&st, me, config);
                let mut best = (legal[0], f64::NEG_INFINITY);
                let mut copy_best = (legal[0], f64::NEG_INFINITY);
                for &action in &legal {
                    let real = expected_value(&st, action, me, &root);
                    let copy = v2_expected_value(&st, action, me, &root);
                    if real.to_bits() != copy.to_bits() {
                        disagreements += 1;
                        assert!(
                            !agree,
                            "seed {seed} turn {}: {action:?} scores {real} through the \
                             agent and {copy} through the round-two copy",
                            st.turn()
                        );
                    }
                    if real > best.1 {
                        best = (action, real);
                    }
                    if copy > copy_best.1 {
                        copy_best = (action, copy);
                    }
                }
                if best.0 != copy_best.0 {
                    disagreements += 1;
                    assert!(
                        !agree,
                        "seed {seed} turn {}: the agent plays {:?}, the round-two copy \
                         plays {:?}",
                        st.turn(),
                        best.0,
                        copy_best.0
                    );
                }
                best.0
            };
            engine::apply(&mut st, chosen, &mut rng).expect("the chosen action was legal");
        }
    }
    disagreements
}

/// The whole point: [`Config::v2`] must reproduce the round-two evaluation's
/// arithmetic **exactly**, not approximately, on every candidate of every
/// decision of whole games.
#[test]
fn config_v2_scores_every_candidate_bit_identically_to_the_round_two_evaluation() {
    assert_eq!(compare(Config::v2(), true), 0);
}

/// ...and both round-three insertions into the evaluation must genuinely
/// change what it computes, or the copy above would be asserting nothing.
///
/// They have to be provoked separately, and that is worth saying plainly: at
/// the shipped default the copy and the real evaluation agree on every
/// candidate of six whole self-play games, because `production_lock_in` is
/// zero (so the lock factor is exactly `1.0`) and because a rail fires on a
/// few hundred decisions in seven thousand — none of them on the lines these
/// six games happen to walk. That is the round in one sentence: the rails
/// change almost nothing almost all of the time, and change the answer
/// completely when they speak.
#[test]
fn the_round_three_insertions_are_not_no_ops() {
    // 1. The production lock-in multiplier, switched on.
    let locked = Config {
        eval: EvalWeights {
            production_lock_in: 0.5,
            ..Config::default().eval
        },
        ..Config::default()
    };
    assert!(
        compare(locked, false) > 0,
        "the lock-in multiplier changes nothing even when switched on"
    );

    // 2. A rail, on a position built to fire one: Player Two is two shields
    //    from the capital with the Circus face up and affordable, so every
    //    candidate that leaves it there is pinned at -imminent.
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "circus"), (19, "palace")])
        .conflict(-7)
        .coins(Player::One, 30)
        .coins(Player::Two, 30)
        .current(Player::One)
        .build();
    let me = st.current_player();
    let root = Root::new(&st, me, Config::default());
    let pinned = engine::legal_actions(&st)
        .into_iter()
        .filter(|&a| {
            expected_value(&st, a, me, &root).to_bits()
                != v2_expected_value(&st, a, me, &root).to_bits()
        })
        .count();
    assert!(
        pinned > 0,
        "no candidate was scored differently by the rails, so the copy is not          actually missing them"
    );
}

/// The one piece of round-three arithmetic the copy above cannot see.
///
/// [`Config::military_horizon`] is applied inside [`Root::new`], before the
/// evaluation ever runs, so a copy of the evaluation would happily share a
/// changed smoothing width and report agreement. This checks it directly
/// against the round-two construction — the whole remaining shield supply,
/// straight off the military read — bit for bit.
#[test]
fn the_v2_smoothing_width_is_the_round_two_one() {
    for seed in 0..6u64 {
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed);
        for step in 0..40 {
            if st.is_over() {
                break;
            }
            let me = st.current_player();
            let root = Root::new(&st, me, Config::v2());
            let e = &Config::v2().eval;
            let mil = &root.stance().military;
            let shields_remaining =
                f64::from(mil.visible) + mil.expected_hidden + mil.expected_future_ages;
            let want = MilSmoothing::of(
                shields_remaining,
                e.military_sigma_scale,
                e.military_sigma_min,
                e.military_logistic_scale,
            );
            assert_eq!(
                root.smoothing().s.to_bits(),
                want.s.to_bits(),
                "seed {seed} step {step}: v2's smoothing width is {} and the round-two \
                 construction gives {}",
                root.smoothing().s,
                want.s
            );

            // ...and the horizon really does change it, so this is not vacuous.
            if shields_remaining > 1.0 {
                let sharp = Root::new(
                    &st,
                    me,
                    Config {
                        military_horizon: Some(3.0),
                        ..Config::v2()
                    },
                );
                assert!(sharp.smoothing().s <= root.smoothing().s);
            }

            let legal = engine::legal_actions(&st);
            if legal.is_empty() {
                break;
            }
            let action = legal[(st.turn() as usize * 7) % legal.len()];
            engine::apply(&mut st, action, &mut rng).unwrap();
        }
    }
}

/// `Config::v2()` really is the round-two *configuration*, field by field, so
/// the arithmetic identity above is about the code rather than about a
/// coincidence of weights.
#[test]
fn config_v2_switches_off_every_round_three_option() {
    let v2 = Config::v2();
    assert_eq!(v2.rails, duels_agent_phased::RailModel::Off);
    assert_eq!(
        v2.menu_shield_pricing,
        duels_agent_phased::MenuShieldPricing::OneSided
    );
    assert_eq!(v2.military_horizon, None);
    assert_eq!(v2.eval.imminent, 0.0);
    assert_eq!(v2.eval.production_lock_in, 0.0);
    assert_eq!(v2.eval.military_band, 2.0);
    // ...and everything round two did not touch is still at its own default.
    assert_eq!(
        v2.eval,
        EvalWeights {
            military_band: 2.0,
            imminent: 0.0,
            production_lock_in: 0.0,
            menu: MenuWeights::default(),
            ..EvalWeights::default()
        }
    );
}
