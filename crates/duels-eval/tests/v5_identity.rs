//! The bit-identity guard for this crate's **sixth** round of work.
//!
//! Round six adds one number: [`EvalWeights::wonder_extra_turn_premium`], what
//! [`WonderModel::Flat`] pays for an unbuilt wonder that prints *play again*,
//! on top of [`terms::wonder_power`]'s flat "+3, this wonder has an effect".
//! [`Config::v5`] sets it back to `0.0`, and this file asserts that doing so
//! reproduces the round-five *arithmetic* — not merely something similar — on
//! every candidate of every decision of whole seeded games.
//!
//! Same shape as `tests/v4_identity.rs`, and — as there — it has to be a
//! same-process copy rather than a recorded digest, because `exp` and `powf` do
//! not agree bit for bit across platforms.
//!
//! # What is copied, and what deliberately is not
//!
//! Round six's change is one line inside `player_value`: the flat wonder term
//! now reads the weights. So the copy below is the whole evaluation chain —
//! `evaluate`, the pending resolution, `player_value`, `expected_value` — with
//! **`v5_wonder_potential` and `v5_wonder_power` copied verbatim from round
//! five**, effect-blind `+3` and all, and everything else calling the real
//! functions.
//!
//! `menu::menu_term` is *not* copied, unlike in `v4_identity.rs`. Round six did
//! not touch the menu, nor anything that feeds it: the premium is added inside
//! `terms::wonder_power_flat`, which the menu never calls (its
//! `MenuFloor::DiscardAndWonder` prices a wonder off `terms::WonderBudget`,
//! which is the *other* model and is untouched). Copying it would be copying
//! code round six cannot have changed, and a stale copy of an unrelated
//! function is a liability rather than a guard.
//!
//! # The two halves of the claim
//!
//! `the_premium_at_zero_reproduces_round_five_bit_for_bit` is the identity.
//! `the_premium_is_not_a_no_op_when_it_is_switched_on` is the other half —
//! without it this file would be asserting that a piece of dead code is dead.
//! `an_extra_turn_premium_moves_only_the_five_play_again_wonders` pins the
//! blast radius: the seven wonders that do not print play again are worth
//! exactly what they were worth before, at any premium.

use duels_core::data::{WonderId, NUM_WONDERS};
use duels_core::scoring::{self, GameResult};
use duels_core::testing::StateBuilder;
use duels_core::{engine, Action, GameState, Player};
use duels_eval::{
    expected_value, menu, terms, CoinModel, Config, EconomyModel, EvalWeights, MilitaryModel, Root,
    WonderModel, DESTROY_REPLACE_SHARE, MAX_PENDING_DEPTH,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// A verbatim copy of the wonder term as it stood before round six.
// ---------------------------------------------------------------------------

/// Round five's `terms::wonder_power`, character for character: every effect
/// that is not points, coins or shields priced at a flat `+3`, play again
/// included.
fn v5_wonder_power(w: WonderId) -> f64 {
    let def = w.def();
    let mut v = f64::from(def.victory_points)
        + f64::from(def.coins) * 0.3
        + f64::from(def.shields)
        + f64::from(def.opponent_loses_coins) * 0.3;
    for flag in [
        def.play_again,
        def.destroy.is_some(),
        def.build_discarded_free,
        def.choose_progress_token,
    ] {
        if flag {
            v += 3.0;
        }
    }
    if def.produces_choice.is_some() {
        v += 2.0;
    }
    v
}

/// Round five's `terms::wonder_potential`: no weights, so no premium. The
/// seven-wonder cap is landed unconditionally (it is a bug fix, not a model —
/// see `tests/v3_identity.rs`), so it is present in this copy too.
fn v5_wonder_potential(state: &GameState, p: Player) -> f64 {
    if terms::wonder_slots_left(state) <= 0.0 {
        return 0.0;
    }
    let ps = state.player(p);
    ps.wonders()
        .filter(|&w| !ps.has_built_wonder(w))
        .map(v5_wonder_power)
        .sum()
}

// ---------------------------------------------------------------------------
// The evaluation chain around it, so the identity is on the whole score.
// ---------------------------------------------------------------------------

fn v5_evaluate(state: &GameState, me: Player, root: &Root) -> f64 {
    v5_evaluate_at(state, me, root, MAX_PENDING_DEPTH)
}

fn v5_evaluate_at(state: &GameState, me: Player, root: &Root, depth: u8) -> f64 {
    if let Some(result) = state.result() {
        return match result {
            GameResult::Win { winner, .. } if winner == me => root.config().eval.instant_result,
            GameResult::Win { .. } => -root.config().eval.instant_result,
            GameResult::Draw => 0.0,
        };
    }
    if root.config().pending_model == duels_eval::PendingModel::Completed
        && depth > 0
        && state.pending().is_some()
    {
        if let Some(v) = v5_resolve_pending(state, me, root, depth) {
            return v;
        }
    }
    if let Some(v) = duels_eval::rail_value(
        state,
        me,
        root.age(),
        root.config().rails,
        root.config().eval.imminent,
    ) {
        return v;
    }
    v5_player_value(state, me, root) - v5_player_value(state, me.other(), root)
        + menu::menu_term(state, me, root.age(), root.menu(), &root.config().eval.menu)
}

fn v5_resolve_pending(state: &GameState, me: Player, root: &Root, depth: u8) -> Option<f64> {
    let resolver = state.current_player();
    let sign = if resolver == me { 1.0 } else { -1.0 };
    let trivial = engine::Outcome::default();

    let discount = root.config().destroy_replace_discount
        && matches!(
            state.pending(),
            Some(duels_core::state::Pending::Destroy { .. })
        );
    let unresolved = if discount {
        v5_player_value(state, me, root) - v5_player_value(state, me.other(), root)
            + menu::menu_term(state, me, root.age(), root.menu(), &root.config().eval.menu)
    } else {
        0.0
    };

    let mut best: Option<(f64, f64)> = None;
    for option in engine::legal_actions(state) {
        let mut next = *state;
        if engine::apply_with_outcome_unchecked(&mut next, option, &trivial).is_err() {
            continue;
        }
        let mut value = v5_evaluate_at(&next, me, root, depth - 1);
        if discount {
            if let Action::DestroyOpponentCard { card } = option {
                let replace = root.destroy_replaceability(card);
                if replace > 0.0 {
                    value = unresolved + (value - unresolved) * (1.0 - replace);
                }
            }
        }
        let key = sign * value;
        if best.is_none_or(|(b, _)| key > b) {
            best = Some((key, value));
        }
    }
    best.map(|(_, value)| value)
}

fn v5_player_value(state: &GameState, p: Player, root: &Root) -> f64 {
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
    // The one line round six changed.
    let wonders = match c.wonder_model {
        WonderModel::Flat => e.wonder_potential * v5_wonder_potential(state, p),
        WonderModel::Budget => terms::wonder_potential_budget(state, p, root.wonders()),
        // Round nine's third model, which no snapshot this file drives can be
        // in: `Config::v8()` sets `WonderModel::Flat` and every older snapshot
        // chains through it. Spelled out rather than left to a wildcard so a
        // later round adding a fourth model has to think about this verbatim
        // copy instead of silently falling through it.
        WonderModel::Rationed => {
            unreachable!("this copy is only ever driven with WonderModel::Flat")
        }
    };
    let gift = if e.menu.lambda == 0.0 {
        -e.deny_chain_gift * terms::chain_gift_exposure(state, p, root.age())
    } else {
        0.0
    };
    let guilds = if e.guild_projection == 0.0 {
        0.0
    } else {
        e.guild_projection * root.guilds().projection(state, p)
    };
    let yellow = if e.yellow_equity == 0.0 {
        0.0
    } else {
        e.yellow_equity
            * terms::yellow_equity(
                state,
                p,
                root.menu().take(p).coin_marginal,
                e.yellow_discard_rate,
            )
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
        + guilds
        + yellow
}

fn v5_expected_value(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => v5_evaluate(&next, me, root),
            Err(_) => v5_evaluate(state, me, root),
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
    let mut unbuilt_seen = 0usize;
    let mut play_again_seen = 0usize;
    for seed in 0..8u64 {
        let mut st: GameState = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FF_EE00);
        for _ in 0..400 {
            if st.is_over() {
                break;
            }
            // The term is only ever non-zero when somebody holds an unbuilt
            // wonder, and the *premium* only when one of them prints play
            // again. Counted so a run through positions where neither is true
            // cannot pass for evidence.
            for p in Player::ALL {
                let ps = st.player(p);
                for w in ps.wonders() {
                    if ps.has_built_wonder(w) {
                        continue;
                    }
                    unbuilt_seen += 1;
                    if w.def().play_again {
                        play_again_seen += 1;
                    }
                }
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
                    let copy = v5_expected_value(&st, action, me, &root);
                    if real.to_bits() != copy.to_bits() {
                        disagreements += 1;
                        assert!(
                            !agree,
                            "seed {seed} turn {}: {action:?} scores {real} through the \
                             agent and {copy} through the round-five copy",
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
                        "seed {seed} turn {}: the agent plays {:?}, the round-five copy \
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
    assert!(
        unbuilt_seen > 500 && play_again_seen > 100,
        "only {unbuilt_seen} unbuilt wonders and {play_again_seen} unbuilt play-again \
         wonders were ever scored, so the comparison is close to vacuous"
    );
    disagreements
}

/// The identity: at `wonder_extra_turn_premium = 0.0`, every candidate of every
/// decision of eight whole games scores bit for bit what round five scored, and
/// the same move comes out.
#[test]
fn the_premium_at_zero_reproduces_round_five_bit_for_bit() {
    assert_eq!(compare(Config::v5(), true), 0);
}

/// ...and it is still an identity under the *other* wonder model, which the
/// premium must not touch at all.
#[test]
fn the_budget_model_is_untouched_by_round_six() {
    let budget = Config {
        wonder_model: WonderModel::Budget,
        ..Config::v5()
    };
    assert_eq!(compare(budget, true), 0);
    // ...including when the premium is set: `Budget` has its own
    // `wonder_extra_turn_vp` and must not read this one.
    let budget_with_premium = Config {
        wonder_model: WonderModel::Budget,
        eval: EvalWeights {
            wonder_extra_turn_premium: 30.0,
            ..Config::v5().eval
        },
        ..Config::v5()
    };
    assert_eq!(compare(budget_with_premium, true), 0);
}

/// The other half: without this, the identity above would be asserting that a
/// piece of dead code is dead.
#[test]
fn the_premium_is_not_a_no_op_when_it_is_switched_on() {
    let on = Config {
        eval: EvalWeights {
            wonder_extra_turn_premium: 9.0,
            ..Config::v5().eval
        },
        ..Config::v5()
    };
    assert!(
        compare(on, false) > 0,
        "the extra-turn premium changes nothing even when switched on"
    );
}

/// The blast radius, at the term itself: the seven wonders that do not print
/// play again are worth exactly what they were worth before, at any premium,
/// and the five that do are worth exactly `+premium` more.
#[test]
fn an_extra_turn_premium_moves_only_the_five_play_again_wonders() {
    let mut play_again = 0;
    for premium in [0.0, 1.5, 9.0, 30.0] {
        let e = EvalWeights {
            wonder_extra_turn_premium: premium,
            ..Config::v5().eval
        };
        play_again = 0;
        for i in 0..NUM_WONDERS {
            let w = WonderId::from_index(i);
            let want = if w.def().play_again {
                play_again += 1;
                v5_wonder_power(w) + premium
            } else {
                v5_wonder_power(w)
            };
            assert_eq!(
                terms::wonder_power_flat(w, &e).to_bits(),
                want.to_bits(),
                "{} at premium {premium}",
                w.def().name
            );
        }
    }
    // The five are Piraeus, The Appian Way, The Hanging Gardens, The Sphinx and
    // The Temple of Artemis — read off `data/wonders.json` through `def()`
    // rather than written down by slug, so a data change cannot silently
    // desynchronise the count this whole investigation is about.
    assert_eq!(
        play_again, 5,
        "the base game prints play again on exactly five wonders"
    );
}

/// At the term level, on a real hand: a player holding two unbuilt play-again
/// wonders and one that is not collects the premium exactly twice.
#[test]
fn the_premium_is_paid_once_per_unbuilt_play_again_wonder() {
    let st = StateBuilder::new()
        .age(2)
        .wonders(
            Player::One,
            &["the-sphinx", "the-hanging-gardens", "the-pyramids"],
        )
        .wonders(Player::Two, &["the-colossus"])
        .current(Player::One)
        .build();

    let off = terms::wonder_potential(&st, Player::One, &Config::v5().eval);
    let on = terms::wonder_potential(
        &st,
        Player::One,
        &EvalWeights {
            wonder_extra_turn_premium: 9.0,
            ..Config::v5().eval
        },
    );
    // Not a bit comparison: `on` and `off` are sums over the hand, so the
    // difference of the two totals is 18 to within a rounding step rather than
    // exactly. The bit-level claim is the one
    // `an_extra_turn_premium_moves_only_the_five_play_again_wonders` makes, on
    // each wonder's own price.
    assert!(
        (on - off - 18.0).abs() < 1e-9,
        "two unbuilt play-again wonders should collect the premium twice: {on} vs {off}"
    );

    // The opponent holds no play-again wonder, so nothing about their side
    // moves.
    let off_two = terms::wonder_potential(&st, Player::Two, &Config::v5().eval);
    let on_two = terms::wonder_potential(
        &st,
        Player::Two,
        &EvalWeights {
            wonder_extra_turn_premium: 9.0,
            ..Config::v5().eval
        },
    );
    assert_eq!(on_two.to_bits(), off_two.to_bits());
}

/// A wonder that can never be built is worth nothing, premium included — the
/// seven-wonder cap short-circuits before the premium is ever reached, so a
/// large premium cannot resurrect a dead wonder.
#[test]
fn the_premium_cannot_resurrect_a_wonder_past_the_seven_slot_cap() {
    let full = StateBuilder::new()
        .age(3)
        .wonders(Player::One, &["the-sphinx"])
        .wonders_built(
            Player::One,
            &["the-colossus", "the-pyramids", "the-hanging-gardens"],
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
        .current(Player::One)
        .build();

    for premium in [0.0, 9.0, 1000.0] {
        let e = EvalWeights {
            wonder_extra_turn_premium: premium,
            ..Config::v5().eval
        };
        assert_eq!(terms::wonder_potential(&full, Player::One, &e), 0.0);
    }
}

/// `Config::v5()` really is the round-five *configuration*, field by field.
#[test]
fn config_v5_switches_off_every_round_six_option() {
    let v5 = Config::v5();
    assert_eq!(v5.eval.wonder_extra_turn_premium, 0.0);

    // ...and everything round six did not touch is still at its own default,
    // so this is a statement about the code and not about a coincidence of
    // weights.
    let d = Config::default();
    assert_eq!(v5.pending_model, d.pending_model);
    // Against `v8` rather than the default: round nine moved `wonder_model`
    // off `Flat`, so "round six did not touch it" is now a statement about
    // the generation this snapshot chains through, not about today's default.
    assert_eq!(v5.wonder_model, Config::v8().wonder_model);
    assert_eq!(v5.guild_pricing, d.guild_pricing);
    assert_eq!(v5.menu_floor, d.menu_floor);
    assert_eq!(v5.supply_model, d.supply_model);
    assert_eq!(v5.destroy_replace_discount, d.destroy_replace_discount);
    assert_eq!(v5.rails, d.rails);
    assert_eq!(v5.blend, d.blend);
    // ...and the same for the weight, which round nine moved with the model.
    assert_eq!(v5.eval.wonder_potential, Config::v8().eval.wonder_potential);
    assert_eq!(v5.eval.wonder_extra_turn_vp, d.eval.wonder_extra_turn_vp);
    assert_eq!(v5.eval.guild_projection, d.eval.guild_projection);
    assert_eq!(v5.eval.yellow_equity, d.eval.yellow_equity);
}

/// Every older snapshot still switches off everything newer, now that they are
/// written on top of `Config::v5()`.
#[test]
fn the_older_snapshots_still_switch_off_everything_newer() {
    for older in [Config::v1(), Config::v2(), Config::v3(), Config::v4()] {
        assert_eq!(older.eval.wonder_extra_turn_premium, 0.0);
    }
}

/// The destroy-replacement share is still a flat constant and still in range —
/// a guard on the one number `v5_resolve_pending` above reproduces by hand.
#[test]
fn the_destroy_replacement_share_is_still_the_constant_the_copy_assumes() {
    assert!((0.0..=1.0).contains(&DESTROY_REPLACE_SHARE));
}
