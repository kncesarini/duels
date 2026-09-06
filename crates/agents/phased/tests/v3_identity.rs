//! The bit-identity guard for this crate's **fourth** round of work.
//!
//! Round four adds two [`Config`] options — [`PendingModel::Completed`], which
//! finishes a turn the engine left mid-effect, and [`WonderModel::Budget`],
//! which prices an unbuilt wonder effect by effect — plus an off-by-default
//! destroy-replacement discount. [`Config::v3`] sets all three back to what the
//! round-three default did, and this file asserts that it reproduces the
//! round-three *arithmetic*, not merely something similar, on every candidate
//! of every decision of whole seeded games. Same shape as
//! `tests/v2_identity.rs`, and for the same reason it has to be a same-process
//! copy rather than a recorded digest: `exp` and `powf` do not agree bit for
//! bit across platforms.
//!
//! # The one deliberate exception
//!
//! Round four also fixes a **bug**, and a bug fix is not a knob. Round three's
//! `terms::wonder_potential` filtered only on `!has_built_wonder`: it never
//! checked the seven-wonder cap, so once seven wonders were up between the two
//! players — after which no eighth is ever built — it kept paying
//! `0.5 x wonder_power` for every dead wonder still in either hand, for the
//! rest of the game, and asymmetrically, since the two sides rarely hold the
//! same number of them. That is landed unconditionally and `Config::v3()` does
//! not restore it, exactly as round two landed the halved `next_age_start`
//! magnitudes as a fix rather than an option.
//!
//! So the copy below is round three's evaluation **with that one fix applied**,
//! and `the_wonder_cap_fix_is_a_real_change` pins the fix itself against a
//! verbatim copy of the uncapped sum, on a position built to have no slots
//! left. Between them the two tests say: one thing changed at `Config::v3()`,
//! it is the thing that was meant to change, and nothing else did.

use duels_agent_phased::{
    expected_value, menu, terms, Config, EvalWeights, PendingModel, Root, WonderModel,
};
use duels_core::data::WonderId;
use duels_core::scoring::{self, GameResult};
use duels_core::testing::StateBuilder;
use duels_core::{engine, Action, GameState, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// A verbatim copy of the evaluation as it stood before round four, with the
// seven-wonder cap fix folded into `player_value`'s wonder term (see above).
//
// The only other edits are the two round-four insertions removed: `evaluate`
// no longer resolves a pending effect before consulting the rails, and
// `player_value`'s wonder term does not branch on `WonderModel`.
// ---------------------------------------------------------------------------

fn v3_evaluate(state: &GameState, me: Player, root: &Root) -> f64 {
    if let Some(result) = state.result() {
        return match result {
            GameResult::Win { winner, .. } if winner == me => root.config().eval.instant_result,
            GameResult::Win { .. } => -root.config().eval.instant_result,
            GameResult::Draw => 0.0,
        };
    }
    if let Some(v) = duels_agent_phased::rail_value(
        state,
        me,
        root.age(),
        root.config().rails,
        root.config().eval.imminent,
    ) {
        return v;
    }
    v3_player_value(state, me, root) - v3_player_value(state, me.other(), root)
        + menu::menu_term(state, me, root.age(), root.menu(), &root.config().eval.menu)
}

fn v3_player_value(state: &GameState, p: Player, root: &Root) -> f64 {
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

    let lock = 1.0 + e.production_lock_in * root.supply().production_lock_in;
    let development = w.development
        * e.development
        * lock
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
                * lock
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

fn v3_expected_value(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => v3_evaluate(&next, me, root),
            Err(_) => v3_evaluate(state, me, root),
        };
        acc += prob * value;
    }
    acc + root.denial_term(action)
}

// ---------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------

/// Drive whole seeded games with a deterministic first-argmax policy — no RNG
/// in either evaluation, so nothing can drift for a reason other than the
/// arithmetic — scoring every candidate both ways.
fn compare(config: Config, agree: bool) -> usize {
    let mut disagreements = 0usize;
    let mut pending_states = 0usize;
    for seed in 0..8u64 {
        let mut st: GameState = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FF_EE00);
        for _ in 0..400 {
            if st.is_over() {
                break;
            }
            if st.pending().is_some() {
                pending_states += 1;
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
                    let copy = v3_expected_value(&st, action, me, &root);
                    if real.to_bits() != copy.to_bits() {
                        disagreements += 1;
                        assert!(
                            !agree,
                            "seed {seed} turn {}: {action:?} scores {real} through the \
                             agent and {copy} through the round-three copy",
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
                        "seed {seed} turn {}: the agent plays {:?}, the round-three copy \
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
    if agree {
        assert!(
            pending_states > 0,
            "no game ever reached a pending effect, so the identity says nothing \
             about the part of round four that is about pending effects"
        );
    }
    disagreements
}

/// The whole point: [`Config::v3`] reproduces the round-three evaluation's
/// arithmetic exactly, on every candidate of every decision of whole games.
#[test]
fn config_v3_scores_every_candidate_bit_identically_to_the_round_three_evaluation() {
    assert_eq!(compare(Config::v3(), true), 0);
}

/// ...and both round-four insertions genuinely change what the evaluation
/// computes, or the copy above would be asserting nothing.
#[test]
fn the_round_four_insertions_are_not_no_ops() {
    let completed = Config {
        pending_model: PendingModel::Completed,
        ..Config::v3()
    };
    assert!(
        compare(completed, false) > 0,
        "resolving pending effects changes nothing even when switched on"
    );

    let budget = Config {
        wonder_model: WonderModel::Budget,
        ..Config::v3()
    };
    assert!(
        compare(budget, false) > 0,
        "the wonder budget model changes nothing even when switched on"
    );
}

/// `Config::v3()` really is the round-three *configuration*, field by field.
#[test]
fn config_v3_switches_off_every_round_four_option() {
    let v3 = Config::v3();
    assert_eq!(v3.pending_model, PendingModel::Unresolved);
    assert_eq!(v3.wonder_model, WonderModel::Flat);
    assert!(!v3.destroy_replace_discount);
    // ...and everything round three did not touch is still at its own default,
    // so this is a statement about the code and not about a coincidence of
    // weights.
    assert_eq!(v3.eval, EvalWeights::default());
    assert_eq!(v3.rails, Config::default().rails);
    assert_eq!(
        v3.menu_shield_pricing,
        Config::default().menu_shield_pricing
    );
    assert_eq!(v3.military_horizon, Config::default().military_horizon);
    assert_eq!(v3.blend, Config::default().blend);
}

/// `Config::v2()` still reproduces round two, now that it is written on top of
/// `Config::v3()`.
#[test]
fn the_older_snapshots_still_switch_off_everything_newer() {
    for older in [Config::v1(), Config::v2()] {
        assert_eq!(older.pending_model, PendingModel::Unresolved);
        assert_eq!(older.wonder_model, WonderModel::Flat);
        assert!(!older.destroy_replace_discount);
    }
}

// ---------------------------------------------------------------------------
// The one deliberate exception: the seven-wonder cap
// ---------------------------------------------------------------------------

/// A verbatim copy of round three's `terms::wonder_potential` — no cap.
fn uncapped_wonder_potential(state: &GameState, p: Player) -> f64 {
    let ps = state.player(p);
    ps.wonders()
        .filter(|&w| !ps.has_built_wonder(w))
        .map(terms::wonder_power)
        .sum()
}

/// The bug, and the fix, in one position.
///
/// Seven wonders are up between the two players, which is every slot the base
/// game has. Player One still holds an unbuilt Pyramids and Player Two holds
/// nothing unbuilt, so the old, uncapped term hands Player One nine points of
/// "potential" for a wonder that can never be built — and the difference is
/// one-sided, so it moves the whole differenced evaluation, not just a term.
#[test]
fn the_wonder_cap_fix_is_a_real_change() {
    let w = |slug: &str| WonderId::from_slug(slug).expect("a real wonder");
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "palace"), (19, "circus")])
        .wonders(Player::One, &["the-pyramids"])
        .wonders_built(
            Player::One,
            &["the-colossus", "the-sphinx", "the-hanging-gardens"],
        )
        .wonders_built(
            Player::Two,
            &["piraeus", "the-appian-way", "the-great-lighthouse"],
        )
        .coins(Player::One, 20)
        .coins(Player::Two, 20)
        .current(Player::One)
        .build();

    assert_eq!(
        st.wonders_built_total(),
        6,
        "test setup: six wonders built so far"
    );
    // With a slot still open the two agree exactly.
    assert_eq!(
        terms::wonder_potential(&st, Player::One).to_bits(),
        uncapped_wonder_potential(&st, Player::One).to_bits(),
        "with a slot left the cap must change nothing"
    );

    // Build the seventh, and the eighth becomes unbuildable.
    let full = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "palace"), (19, "circus")])
        .wonders(Player::One, &["the-pyramids"])
        .wonders_built(
            Player::One,
            &["the-colossus", "the-sphinx", "the-hanging-gardens"],
        )
        .wonders_built(
            Player::Two,
            &[
                "piraeus",
                "the-appian-way",
                "the-great-lighthouse",
                "the-mausoleum",
            ],
        )
        .coins(Player::One, 20)
        .coins(Player::Two, 20)
        .current(Player::One)
        .build();

    assert_eq!(full.wonders_built_total(), 7, "every slot is gone");
    assert!(!full.wonder_slots_left());
    assert_eq!(terms::wonder_potential(&full, Player::One), 0.0);
    assert_eq!(terms::wonder_potential(&full, Player::Two), 0.0);
    assert_eq!(
        uncapped_wonder_potential(&full, Player::One),
        terms::wonder_power(w("the-pyramids")),
        "the old term still paid for the unbuildable Pyramids"
    );
    assert_eq!(uncapped_wonder_potential(&full, Player::Two), 0.0);
    assert!(
        uncapped_wonder_potential(&full, Player::One) > 0.0,
        "the fix would be vacuous if the old term paid nothing here"
    );
}
