//! The bit-identity guard for this crate's **seventh** round of work.
//!
//! Round seven changes what [`Config::default`] returns — five weights and one
//! new gate inside the science ladder — and adds three terms plus one menu
//! option that are off by default. [`Config::v6`] sets all of it back, and this file asserts that
//! doing so reproduces the round-six *arithmetic*, not merely something
//! similar, on every candidate of every decision of whole seeded games.
//!
//! Same shape as `tests/v5_identity.rs`, and — as there — it has to be a
//! same-process copy rather than a recorded digest, because `exp` and `powf`
//! do not agree bit for bit across platforms.
//!
//! # What is copied, and what deliberately is not
//!
//! Round seven touches the arithmetic in exactly three places, and each one is
//! copied below in its round-six form:
//!
//! * `terms::science_ladder` grew the [`ScienceWeights::dead_race_scale`] gate
//!   → `v6_science_ladder` (and, with it, `v6_pair_threat`, which it calls and
//!   which is private to the crate).
//! * `player_value` grew two summands, `token_equity` and `to_move`
//!   → `v6_player_value`.
//! * `evaluate` and `Root::denial_term` grew the
//!   [`EvalWeights::value_scale`] multiplication → `v6_evaluate`,
//!   `v6_evaluate_at`, `v6_resolve_pending`, `v6_expected_value`.
//!
//! `menu.rs` is **not** copied. Round seven's change there is one addition
//! ([`CountPricing`]) inside an `if self.count_pricing != Unpriced`, so under
//! `Unpriced` the sequence of floating-point operations `free_value` performs
//! is not merely equal to round six's, it is literally the same sequence. The
//! guard is deliberate and not a `x + 0.0`: adding a positive zero to a
//! negative zero is *not* bit-identical to leaving it alone, and this crate's
//! identity tests are the reason to care. `terms::science_ladder`'s gate and
//! `evaluate`'s scale are guarded for the same reason, which is also why
//! `v6_science_ladder` and `v6_evaluate` below can be short.
//!
//! # The two halves of the claim
//!
//! `config_v6_reproduces_round_six_bit_for_bit` is the identity.
//! `the_round_seven_default_is_not_a_no_op` is the other half — without it
//! this file would be asserting that a pile of dead code is dead.

use duels_core::data::{self, TokenId};
use duels_core::scoring::{self, GameResult};
use duels_core::testing::StateBuilder;
use duels_core::{engine, Action, GameState, Player};
use duels_eval::{
    evaluate, expected_value, menu, rails, terms, CoinModel, Config, CountPricing, EconomyModel,
    EvalWeights, MilitaryModel, PendingModel, Root, ScienceWeights, WonderModel, MAX_PENDING_DEPTH,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// A verbatim copy of the three changed functions as they stood before round
// seven.
// ---------------------------------------------------------------------------

/// Round six's `terms::pair_threat`, which is private to the crate and which
/// `science_ladder` calls, reproduced from its public parts.
fn v6_pair_threat(state: &GameState, p: Player, w: &ScienceWeights) -> f64 {
    let me = state.player(p);
    let held = me.science();
    let mut awarded = 0u8;
    for sym in me.pairs_awarded() {
        awarded |= 1u8 << sym.index();
    }
    let law_symbol = TokenId::all()
        .find(|t| t.def().science.is_some())
        .and_then(|t| t.def().science);

    let mut candidates = 0.0f64;
    for sym in data::Science::ALL {
        let i = sym.index();
        if held[i] != 1 || Some(sym) == law_symbol || awarded & (1u8 << i) != 0 {
            continue;
        }
        if terms::second_copy_obtainable(state, sym) {
            candidates += 1.0;
        }
    }
    if candidates == 0.0 {
        return 0.0;
    }
    let best_token = state
        .board_tokens()
        .map(|t| duels_strategy::science::token_value(state, p, t))
        .fold(0.0f64, f64::max);
    candidates * (w.pair_token_share * best_token + w.pair_tempo_tax)
}

/// Round six's `terms::science_ladder`: the rung, unconditionally, with no
/// reachability gate.
fn v6_science_ladder(state: &GameState, p: Player, w: &ScienceWeights) -> f64 {
    let distinct = usize::from(state.player(p).distinct_science()).min(w.ladder.len() - 1);
    let token_mult = 1.0 + w.strong_token_mult * terms::strong_board_tokens(state);
    w.ladder[distinct] * token_mult + w.pair_threat_weight * v6_pair_threat(state, p, w)
}

/// Round six's `player_value`: fourteen summands, without `token_equity` and
/// without `to_move`.
fn v6_player_value(state: &GameState, p: Player, root: &Root) -> f64 {
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

    let science = w.science * e.science_ladder * v6_science_ladder(state, p, &e.science);
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

fn v6_evaluate(state: &GameState, me: Player, root: &Root) -> f64 {
    v6_evaluate_at(state, me, root, MAX_PENDING_DEPTH)
}

/// Round six's `evaluate_at`: the weighted sum returned unscaled.
fn v6_evaluate_at(state: &GameState, me: Player, root: &Root, depth: u8) -> f64 {
    if let Some(result) = state.result() {
        return match result {
            GameResult::Win { winner, .. } if winner == me => root.config().eval.instant_result,
            GameResult::Win { .. } => -root.config().eval.instant_result,
            GameResult::Draw => 0.0,
        };
    }
    if root.config().pending_model == PendingModel::Completed
        && depth > 0
        && state.pending().is_some()
    {
        if let Some(v) = v6_resolve_pending(state, me, root, depth) {
            return v;
        }
    }
    if let Some(v) = rails::rail_value(
        state,
        me,
        root.age(),
        root.config().rails,
        root.config().eval.imminent,
    ) {
        return v;
    }
    v6_player_value(state, me, root) - v6_player_value(state, me.other(), root)
        + menu::menu_term(state, me, root.age(), root.menu(), &root.config().eval.menu)
}

fn v6_resolve_pending(state: &GameState, me: Player, root: &Root, depth: u8) -> Option<f64> {
    let resolver = state.current_player();
    let sign = if resolver == me { 1.0 } else { -1.0 };
    let trivial = engine::Outcome::default();

    let discount = root.config().destroy_replace_discount
        && matches!(
            state.pending(),
            Some(duels_core::state::Pending::Destroy { .. })
        );
    let unresolved = if discount {
        v6_player_value(state, me, root) - v6_player_value(state, me.other(), root)
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
        let mut value = v6_evaluate_at(&next, me, root, depth - 1);
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

/// Round six's `expected_value`: the denial term added unscaled.
fn v6_expected_value(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => v6_evaluate(&next, me, root),
            Err(_) => v6_evaluate(state, me, root),
        };
        acc += prob * value;
    }
    acc + root.config().eval.deny
        * root.deny_scale()
        * duels_strategy::deny_vp(action, root.stance())
}

// ---------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------

/// Drive whole seeded games under `config`, scoring every candidate of every
/// decision through both the shipped code and the round-six copy.
///
/// Returns how many candidates were compared, so a test cannot pass by
/// comparing nothing. When `agree` is false it instead asserts that at least
/// one candidate *disagrees*, which is the other half of the claim.
fn compare(config: Config, agree: bool) -> usize {
    let mut compared = 0usize;
    let mut disagreements = 0usize;
    for seed in 0..12u64 {
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xF00D);
        while !state.is_over() {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let me = state.current_player();
            let root = Root::new(&state, me, config);
            let mut best: Option<(Action, f64)> = None;
            for &action in &legal {
                let new = expected_value(&state, action, me, &root);
                let old = v6_expected_value(&state, action, me, &root);
                compared += 1;
                if agree {
                    assert_eq!(
                        new.to_bits(),
                        old.to_bits(),
                        "seed {seed} turn {} action {action:?}: {new} vs {old}",
                        state.turn()
                    );
                } else if new.to_bits() != old.to_bits() {
                    disagreements += 1;
                }
                if best.is_none_or(|(_, b)| new > b) {
                    best = Some((action, new));
                }
            }
            let (action, _) = best.expect("a non-empty legal set has a best action");
            engine::apply(&mut state, action, &mut rng).expect("the driver plays legally");
        }
    }
    assert!(compared > 500, "only {compared} candidates compared");
    if !agree {
        assert!(
            disagreements > 0,
            "{compared} candidates and not one disagreement: the round-seven \
             default is a no-op"
        );
    }
    compared
}

#[test]
fn config_v6_reproduces_round_six_bit_for_bit() {
    let compared = compare(Config::v6(), true);
    println!("{compared} candidates agreed bit for bit");
}

#[test]
fn the_round_seven_default_is_not_a_no_op() {
    // `Config::v7()`, not `Config::default()`: the claim this test makes is
    // about *round seven*, and driving it from today's default would fold in
    // every later round as well. Round nine is what made the distinction worth
    // spelling out — it added a third `WonderModel`, which the verbatim
    // round-six copy above cannot represent and does not have to, and which a
    // round-ten default move could put in this copy's path.
    compare(Config::v7(), false);
}

#[test]
fn every_round_seven_option_is_reproduced_by_v6_at_its_off_value() {
    let v6 = Config::v6();
    assert_eq!(v6.eval.science.dead_race_scale, 1.0);
    assert_eq!(v6.eval.science.pair_threat_weight, 1.0);
    assert_eq!(v6.eval.science_ladder, 1.0);
    assert_eq!(v6.eval.chain_equity, 1.0);
    assert_eq!(v6.eval.resource_bill, 3.0);
    assert_eq!(v6.eval.development, 1.0 / 3.0);
    assert_eq!(v6.eval.token_equity, 0.0);
    assert_eq!(v6.eval.to_move, 0.0);
    assert_eq!(v6.eval.value_scale, 1.0);
    assert_eq!(v6.count_pricing, CountPricing::Unpriced);
}

#[test]
fn the_older_snapshots_still_switch_off_everything_newer() {
    for older in [
        Config::v1(),
        Config::v2(),
        Config::v3(),
        Config::v4(),
        Config::v5(),
    ] {
        assert_eq!(older.eval.science.dead_race_scale, 1.0);
        assert_eq!(older.eval.science_ladder, 1.0);
        assert_eq!(older.eval.token_equity, 0.0);
        assert_eq!(older.eval.to_move, 0.0);
        assert_eq!(older.eval.value_scale, 1.0);
        assert_eq!(older.count_pricing, CountPricing::Unpriced);
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
    // a tautology and is here to fail loudly if a later round ever re-points
    // it without adding `v10`. Round ten is why this reads `v9` and not `v8`:
    // it moved the default, and the contract under `Config::v9` is that the
    // round doing so re-points the previous snapshot at literal values in the
    // same PR.
    assert_ne!(Config::v9(), Config::default());
    // `v9` is left out of the loop above because round nine adopted neither of
    // the options it built, so its snapshot really is round eight's.
    assert_eq!(Config::v8(), Config::v9());
}

/// The dead-race gate has to be a gate rather than a blanket reduction, so it
/// must be a no-op exactly while supremacy is still reachable. A fresh game is
/// the clearest case: nothing is gone, so all seven symbols are reachable.
#[test]
fn the_dead_race_gate_does_nothing_while_the_race_is_alive() {
    let st = engine::new_game(11);
    for p in Player::ALL {
        assert_eq!(
            terms::supremacy_reachable(&st, p),
            7,
            "a fresh game should leave every symbol reachable"
        );
        let w = ScienceWeights::default();
        let gated = terms::science_ladder(&st, p, &w);
        let ungated = terms::science_ladder(
            &st,
            p,
            &ScienceWeights {
                dead_race_scale: 1.0,
                ..w
            },
        );
        assert_eq!(gated.to_bits(), ungated.to_bits());
    }
}

/// ...and it must actually fire once the symbols are gone, or the whole term is
/// dead code. Both players hold three of the six card symbols between them and
/// the rest are in the discard pile, so neither can reach six.
#[test]
fn the_dead_race_gate_fires_once_supremacy_is_out_of_reach() {
    let mut builder = StateBuilder::new();
    builder = builder
        .age(3)
        .built(Player::One, &["workshop", "apothecary", "scriptorium"])
        .coins(Player::One, 10)
        .coins(Player::Two, 10)
        .current(Player::One);
    let st = builder.build();
    let reachable = terms::supremacy_reachable(&st, Player::One);
    assert!(
        reachable < terms::SYMBOLS_TO_WIN,
        "test setup: {reachable} symbols still reachable in Age III"
    );
    let w = ScienceWeights::default();
    let gated = terms::science_ladder(&st, Player::One, &w);
    let ungated = terms::science_ladder(
        &st,
        Player::One,
        &ScienceWeights {
            dead_race_scale: 1.0,
            ..w
        },
    );
    assert!(
        gated < ungated,
        "the gate did not reduce the rung: {gated} vs {ungated}"
    );
}

/// Six distinct symbols is still what wins the game, so
/// [`terms::SYMBOLS_TO_WIN`] is not a stale copy of a rule.
#[test]
fn six_distinct_symbols_is_still_what_wins_the_game() {
    let five = StateBuilder::new()
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
        .build();
    let held = five.player(Player::One).distinct_science();
    assert_eq!(
        u32::from(held) + 1,
        u32::from(terms::SYMBOLS_TO_WIN),
        "test setup: {held} distinct symbols, so this is not the boundary case"
    );
    assert!(
        five.result().is_none(),
        "{held} distinct symbols must not win the game"
    );
    assert!(terms::supremacy_reachable(&five, Player::One) >= terms::SYMBOLS_TO_WIN);
}

/// The rebate the token table prices is two units wide, and the **cost engine**
/// is the authority for that rather than this crate's constant.
///
/// The Aqueduct costs three stone. A city that produces nothing, facing an
/// opponent who produces nothing, pays the base trade price of two coins a
/// unit — six coins — and with Masonry pays for one unit only. The saving
/// divided by the unit price is therefore exactly how many units the rebate
/// covers, which is what [`terms::REBATE_UNITS`] claims.
#[test]
fn the_rebate_is_still_two_units_wide() {
    let position = |tokens: &[&str]| -> GameState {
        StateBuilder::new()
            .age(2)
            .open_slots(&[(18, "aqueduct")])
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .tokens(Player::One, tokens)
            .current(Player::One)
            .build()
    };
    let bare = position(&[]);
    let with_masonry = position(&["masonry"]);
    let card = bare.face_up_card(18).expect("slot 18 is face up");
    let before = duels_core::cost::card_cost(&bare, Player::One, card).coins;
    let after = duels_core::cost::card_cost(&with_masonry, Player::One, card).coins;
    let unit_price = u32::from(duels_core::cost::trade_prices(&bare, Player::One)[2]);
    assert_eq!(before, 6, "test setup: three stone at two coins a unit");
    assert_eq!(unit_price, 2, "test setup: the base trade price is two");
    let covered = f64::from(u32::from(before - after)) / f64::from(unit_price);
    assert_eq!(
        covered,
        terms::REBATE_UNITS,
        "the cost engine covers {covered} units, not {}",
        terms::REBATE_UNITS
    );
}

/// The three round-seven terms that are off by default really are the three
/// that are off, and switching each on really does move the score — otherwise
/// the honest negatives in the crate docs are negatives about nothing.
#[test]
fn each_off_by_default_option_moves_the_score_when_it_is_switched_on() {
    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "chamber-of-commerce"), (19, "arena")])
        .built(Player::One, &["glassblower", "drying-room", "press"])
        .tokens(Player::One, &["theology", "economy"])
        .coins(Player::One, 20)
        .coins(Player::Two, 20)
        .current(Player::One)
        .build();
    let me = st.current_player();
    let base = Config::default();
    let baseline = evaluate(&st, me, &Root::new(&st, me, base));

    for (name, cfg) in [
        (
            "token_equity",
            Config {
                eval: EvalWeights {
                    token_equity: 1.0,
                    ..base.eval
                },
                ..base
            },
        ),
        (
            "to_move",
            Config {
                eval: EvalWeights {
                    to_move: 5.0,
                    ..base.eval
                },
                ..base
            },
        ),
        (
            "value_scale",
            Config {
                eval: EvalWeights {
                    value_scale: 2.0,
                    ..base.eval
                },
                ..base
            },
        ),
        (
            "count_pricing",
            Config {
                count_pricing: CountPricing::Counted,
                ..base
            },
        ),
    ] {
        let moved = evaluate(&st, me, &Root::new(&st, me, cfg));
        assert_ne!(
            moved.to_bits(),
            baseline.to_bits(),
            "{name} switched on changed nothing: {moved} vs {baseline}"
        );
    }
}
