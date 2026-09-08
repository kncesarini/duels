//! The bit-identity guard for this crate's **fifth** round of work.
//!
//! Round five adds guild pricing on the menu ([`GuildPricing`]), a forward
//! projection term for guilds already built, a menu floor ([`MenuFloor`]), a
//! soft affordability weight, a yellow-density term, and a dealt-fraction
//! weighting for the undealt half of the supply pool ([`SupplyModel`]).
//! [`Config::v4`] sets all six back to what the round-four default did, and this
//! file asserts that it reproduces the round-four *arithmetic*, not merely
//! something similar, on every candidate of every decision of whole seeded
//! games.
//!
//! Same shape as `tests/v3_identity.rs`, and — as there — it has to be a
//! same-process copy rather than a recorded digest, because `exp` and `powf` do
//! not agree bit for bit across platforms.
//!
//! # What is copied, and why the menu is copied too
//!
//! Round five's changes fall in two places. `player_value` gained two terms,
//! which the copy below simply does not have. `menu::menu_term` gained a floor
//! and a soft affordability weight *inside* itself, and the options that drive
//! them travel in [`MenuTables`] rather than in the signature — so a copy that
//! called the real `menu_term` would be testing nothing about them. The copy
//! below is therefore the round-four `menu_term` verbatim, hard cutoff and hard
//! zero included, and the identity says the real one reproduces it exactly when
//! the floor is off.
//!
//! Guild pricing needs no copy: it enters through `TakeValue::free_value`,
//! which every path shares, so an identity on the whole expected value covers
//! it. What it does need is `the_round_five_insertions_are_not_no_ops`, which
//! asserts each option genuinely changes the arithmetic when it is switched on
//! — without which this file would be asserting that six pieces of dead code
//! are dead.

use duels_core::scoring::{self, GameResult};
use duels_core::state::Phase;
use duels_core::{cost, engine, Action, GameState, Player};
use duels_eval::{
    expected_value, menu, terms, CoinModel, Config, EconomyModel, EvalWeights, GuildPricing,
    MenuFloor, MenuTables, MenuWeights, MilitaryModel, Root, SupplyModel, WonderModel,
    DESTROY_REPLACE_SHARE, MAX_PENDING_DEPTH,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// A verbatim copy of the evaluation as it stood before round five.
// ---------------------------------------------------------------------------

/// Round four's `menu::menu_term`, character for character: a hard afford /
/// do-not-afford cutoff, and a hard zero when nothing is affordable.
fn v4_menu_term(
    next: &GameState,
    me: Player,
    root_age: u8,
    tables: &MenuTables,
    w: &MenuWeights,
) -> f64 {
    if w.lambda == 0.0 || w.tau <= 0.0 {
        return 0.0;
    }
    if next.is_over() || next.age() != root_age || next.phase() != Phase::Turn {
        return 0.0;
    }
    let q = next.current_player();
    let coins = next.player(q).coins();

    let mut values: [f64; duels_core::layout::SLOTS] = [0.0; duels_core::layout::SLOTS];
    let mut n = 0usize;
    let mut best = f64::NEG_INFINITY;
    let mut mask = next.accessible_slots();
    while mask != 0 {
        let slot = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        let Some(card) = next.face_up_card(slot) else {
            continue;
        };
        if cost::card_cost(next, q, card).coins > coins {
            continue;
        }
        let v = tables.value(q, card);
        values[n] = v;
        n += 1;
        if v > best {
            best = v;
        }
    }
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = values[..n].iter().map(|v| ((v - best) / w.tau).exp()).sum();
    let menu = best + w.tau * sum.ln();
    if q == me {
        w.lambda * menu
    } else {
        -w.lambda * menu
    }
}

fn v4_evaluate(state: &GameState, me: Player, root: &Root) -> f64 {
    v4_evaluate_at(state, me, root, MAX_PENDING_DEPTH)
}

fn v4_evaluate_at(state: &GameState, me: Player, root: &Root, depth: u8) -> f64 {
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
        if let Some(v) = v4_resolve_pending(state, me, root, depth) {
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
    v4_player_value(state, me, root) - v4_player_value(state, me.other(), root)
        + v4_menu_term(state, me, root.age(), root.menu(), &root.config().eval.menu)
}

fn v4_resolve_pending(state: &GameState, me: Player, root: &Root, depth: u8) -> Option<f64> {
    let resolver = state.current_player();
    let sign = if resolver == me { 1.0 } else { -1.0 };
    let trivial = engine::Outcome::default();

    let discount = root.config().destroy_replace_discount
        && matches!(
            state.pending(),
            Some(duels_core::state::Pending::Destroy { .. })
        );
    let unresolved = if discount {
        v4_player_value(state, me, root) - v4_player_value(state, me.other(), root)
            + v4_menu_term(state, me, root.age(), root.menu(), &root.config().eval.menu)
    } else {
        0.0
    };

    let mut best: Option<(f64, f64)> = None;
    for option in engine::legal_actions(state) {
        let mut next = *state;
        if engine::apply_with_outcome_unchecked(&mut next, option, &trivial).is_err() {
            continue;
        }
        let mut value = v4_evaluate_at(&next, me, root, depth - 1);
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

fn v4_player_value(state: &GameState, p: Player, root: &Root) -> f64 {
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
    let wonders = match c.wonder_model {
        WonderModel::Flat => e.wonder_potential * terms::wonder_potential(state, p, e),
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

fn v4_expected_value(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => v4_evaluate(&next, me, root),
            Err(_) => v4_evaluate(state, me, root),
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
    let mut guilds_seen = 0usize;
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
            // Guilds actually reachable: face up in the structure, or already
            // in a city.
            let mut in_play =
                st.player(Player::One).built_mask() | st.player(Player::Two).built_mask();
            for slot in 0..duels_core::layout::SLOTS as u8 {
                if let Some(c) = st.face_up_card(slot) {
                    in_play |= 1u128 << c.index();
                }
            }
            guilds_seen += (in_play & duels_core::data::statics().guild_mask).count_ones() as usize;
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
                    let copy = v4_expected_value(&st, action, me, &root);
                    if real.to_bits() != copy.to_bits() {
                        disagreements += 1;
                        assert!(
                            !agree,
                            "seed {seed} turn {}: {action:?} scores {real} through the \
                             agent and {copy} through the round-four copy",
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
                        "seed {seed} turn {}: the agent plays {:?}, the round-four copy \
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
             about the round-four half of the evaluation it also has to reproduce"
        );
        assert!(
            guilds_seen > 0,
            "no game ever reached Age III, so the identity says nothing about \
             guilds at all"
        );
    }
    disagreements
}

/// The whole point: [`Config::v4`] reproduces the round-four evaluation's
/// arithmetic exactly, on every candidate of every decision of whole games.
#[test]
fn config_v4_scores_every_candidate_bit_identically_to_the_round_four_evaluation() {
    assert_eq!(compare(Config::v4(), true), 0);
}

/// ...and every round-five insertion genuinely changes what the evaluation
/// computes, or the copy above would be asserting nothing.
#[test]
fn the_round_five_insertions_are_not_no_ops() {
    let cases: [(&str, Config); 6] = [
        (
            "guild pricing",
            Config {
                guild_pricing: GuildPricing::Projected,
                ..Config::v4()
            },
        ),
        (
            "the guild projection term",
            Config {
                eval: EvalWeights {
                    guild_projection: 1.0,
                    ..Config::v4().eval
                },
                ..Config::v4()
            },
        ),
        (
            "the discard floor",
            Config {
                menu_floor: MenuFloor::Discard,
                ..Config::v4()
            },
        ),
        (
            "the discard-and-wonder floor",
            Config {
                menu_floor: MenuFloor::DiscardAndWonder,
                ..Config::v4()
            },
        ),
        (
            "soft affordability",
            Config {
                menu_afford_soft: 3.0,
                ..Config::v4()
            },
        ),
        (
            "the dealt-fraction supply weighting",
            Config {
                supply_model: SupplyModel::Dealt,
                ..Config::v4()
            },
        ),
    ];
    for (what, config) in cases {
        assert!(
            differs(Config::v4(), config) > 0,
            "{what} changes nothing even when switched on"
        );
    }

    // The yellow term's off-switch is a weight rather than an enum, so it is
    // named separately.
    let yellow = Config {
        eval: EvalWeights {
            yellow_equity: 1.0,
            ..Config::v4().eval
        },
        ..Config::v4()
    };
    assert!(
        differs(Config::v4(), yellow) > 0,
        "the yellow-density term changes nothing even when switched on"
    );
}

/// How many candidate scores two configurations disagree on, over whole seeded
/// games driven by the first of them.
///
/// # Why this is not `compare(config, false)`
///
/// Two of round five's six options — guild pricing and the supply weighting —
/// change what goes *into* [`Root`]'s shared tables rather than what the
/// evaluation does with them, and the copy above deliberately reads those
/// tables rather than rebuilding them. So the copy sees the new prices too, and
/// asking it whether the option is a no-op would always answer "yes", which is
/// a fact about the copy and not about the option. Comparing two real
/// configurations answers the question the copy cannot.
fn differs(base: Config, changed: Config) -> usize {
    let mut disagreements = 0usize;
    for seed in 0..8u64 {
        let mut st: GameState = engine::new_game(seed);
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
                let a = Root::new(&st, me, base);
                let b = Root::new(&st, me, changed);
                let mut best = (legal[0], f64::NEG_INFINITY);
                for &action in &legal {
                    let x = expected_value(&st, action, me, &a);
                    let y = expected_value(&st, action, me, &b);
                    if x.to_bits() != y.to_bits() {
                        disagreements += 1;
                    }
                    if x > best.1 {
                        best = (action, x);
                    }
                }
                best.0
            };
            engine::apply(&mut st, chosen, &mut rng).expect("the chosen action was legal");
        }
    }
    disagreements
}

/// `Config::v4()` really is the round-four *configuration*, field by field.
#[test]
fn config_v4_switches_off_every_round_five_option() {
    let v4 = Config::v4();
    assert_eq!(v4.guild_pricing, GuildPricing::Unpriced);
    assert_eq!(v4.menu_floor, MenuFloor::None);
    assert_eq!(v4.menu_afford_soft, 0.0);
    assert_eq!(v4.supply_model, SupplyModel::Raw);
    assert_eq!(v4.eval.guild_projection, 0.0);
    assert_eq!(v4.eval.yellow_equity, 0.0);

    // ...and everything round four did not touch is still at its own default,
    // so this is a statement about the code and not about a coincidence of
    // weights.
    let d = Config::default();
    assert_eq!(v4.pending_model, d.pending_model);
    // Against `v8` rather than the default: round nine moved `wonder_model`
    // off `Flat`, so "round four did not touch it" is now a statement about
    // the generation this snapshot chains through, not about today's default.
    assert_eq!(v4.wonder_model, Config::v8().wonder_model);
    assert_eq!(v4.destroy_replace_discount, d.destroy_replace_discount);
    assert_eq!(v4.rails, d.rails);
    assert_eq!(v4.menu_shield_pricing, d.menu_shield_pricing);
    assert_eq!(v4.military_horizon, d.military_horizon);
    assert_eq!(v4.blend, d.blend);
    assert_eq!(v4.eval.military_band, d.eval.military_band);
    assert_eq!(v4.eval.production_lock_in, d.eval.production_lock_in);
}

/// Every older snapshot still switches off everything newer, now that they are
/// written on top of `Config::v4()`.
#[test]
fn the_older_snapshots_still_switch_off_everything_newer() {
    for older in [Config::v1(), Config::v2(), Config::v3()] {
        assert_eq!(older.guild_pricing, GuildPricing::Unpriced);
        assert_eq!(older.menu_floor, MenuFloor::None);
        assert_eq!(older.menu_afford_soft, 0.0);
        assert_eq!(older.supply_model, SupplyModel::Raw);
        assert_eq!(older.eval.guild_projection, 0.0);
        assert_eq!(older.eval.yellow_equity, 0.0);
    }
}

/// The destroy-replacement share is still a flat constant and still in range —
/// a guard on the one number `v4_resolve_pending` above reproduces by hand.
#[test]
fn the_destroy_replacement_share_is_still_the_constant_the_copy_assumes() {
    assert!((0.0..=1.0).contains(&DESTROY_REPLACE_SHARE));
}
