//! Positions the project owner flagged through `duels-server`'s advanced mode,
//! reconstructed and pinned.
//!
//! Advanced mode exports a flagged position as `{ seed, moves }`, and
//! `duels_server::room::replay` turns one back into a `GameState`. That is the
//! wrong side of the boundary for a `duels-eval` round to test from — this
//! crate depends on `duels-core` and `duels-strategy` and nothing else — so
//! the replay is nine lines here, seeded the same way, and
//! [`the_replay_matches_the_servers_own_stream`] pins the one thing that could
//! make the two diverge.
//!
//! `examples/position_probe.rs` is the interactive form of this file: the same
//! reconstruction, plus the per-term breakdown and the ranked action list.
//!
//! # The round-nine position, and what it is evidence of
//!
//! ```json
//! { "seed": 1, "moves": [ ...eight PickWonders..., 17, 14, 19,
//!                         BuildWonder(18, the-great-lighthouse), 13 ] }
//! ```
//!
//! Two things were flagged about it, and this file pins both because both are
//! **structural** claims that survive a re-weighting, where the numbers behind
//! them do not.
//!
//! **1. "It really overvalues building wonders here early in age 1."** At the
//! decision one move earlier — the one that actually built The Great Lighthouse
//! — the evaluation ranks all three ways of building it above every ordinary
//! card build. [`the_wonder_build_outranks_every_card_build_at_the_flagged_decision`]
//! pins that, so a future round that changes the ranking has to say so out
//! loud. What the evaluation is actually paying for is in the crate docs; the
//! short version is that a produce-a-raw-material wonder cuts
//! `terms::resource_bill` by ten victory points against a seven-coin bill worth
//! four and a half, and no term prices the option of building the same wonder
//! later for less.
//!
//! **2. "Every one of the five legal actions reads lower than standing
//! still."** `current_win_probability` is 0.332 and the five actions read
//! 0.236-0.270. That one is **not** an evaluation error at all, and
//! [`every_action_looks_worse_than_standing_still_and_the_menu_is_why`] is the
//! diagnosis: `menu::menu_term` changes sign with whoever moves *next*, so a
//! pre-move state (this player to move, `+λ·menu`) and a post-move state (the
//! opponent to move, `−λ·menu`) are on opposite sides of a tempo term worth
//! about twelve victory points here. Switching the menu off on both sides
//! reverses the ordering exactly. `evaluate` is antisymmetric and internally
//! consistent; what is not comparable is a *pre-action* value against a
//! *post-action* one, and the analysis endpoint displays them side by side.
//! Nothing in this crate is wrong, and nothing in this crate can fix it —
//! see the round-nine crate docs for what would.

use duels_core::data::WonderId;
use duels_core::{engine, Action, GameState, Player};
use duels_eval::{
    evaluate, expected_value, win_probability_from_value, Config, EvalWeights, MenuWeights, Root,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// `duels_server::room::game_rng`, verbatim: the stream a room applies its
/// moves against, so a position whose history includes The Great Library's
/// three-token draw replays identically rather than approximately.
fn game_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15)
}

fn replay(seed: u64, moves: &[Action]) -> GameState {
    let mut state = engine::new_game(seed);
    let mut rng = game_rng(seed);
    for (i, &a) in moves.iter().enumerate() {
        engine::apply(&mut state, a, &mut rng)
            .unwrap_or_else(|e| panic!("move {i} ({a:?}) does not replay: {e:?}"));
    }
    state
}

fn wonder(slug: &str) -> WonderId {
    WonderId::from_slug(slug).unwrap_or_else(|| panic!("no wonder {slug:?}"))
}

/// The flagged export's move list, in order.
fn flagged_moves() -> Vec<Action> {
    let mut moves: Vec<Action> = [
        "piraeus",
        "the-great-lighthouse",
        "the-statue-of-zeus",
        "circus-maximus",
        "the-temple-of-artemis",
        "the-pyramids",
        "the-hanging-gardens",
        "the-colossus",
    ]
    .iter()
    .map(|s| Action::PickWonder { wonder: wonder(s) })
    .collect();
    moves.extend([
        Action::Build { slot: 17 },
        Action::Build { slot: 14 },
        Action::Build { slot: 19 },
        Action::BuildWonder {
            slot: 18,
            wonder: wonder("the-great-lighthouse"),
        },
        Action::Build { slot: 13 },
    ]);
    moves
}

/// The RNG derivation has to be the server's, or a Great Library draw anywhere
/// in a flagged history would replay into a different world. Nothing in *this*
/// position consumes randomness, which is exactly why it has to be asserted
/// rather than inferred from the position replaying.
#[test]
fn the_replay_matches_the_servers_own_stream() {
    let moves = flagged_moves();
    let with_server_stream = replay(1, &moves);
    let mut other = engine::new_game(1);
    let mut rng = StdRng::seed_from_u64(1);
    for &a in &moves {
        engine::apply(&mut other, a, &mut rng).expect("replays under any stream");
    }
    assert_eq!(
        with_server_stream, other,
        "this history consumes randomness, so the stream salt is load-bearing \
         and the copy of duels_server::room::game_rng above must be kept in step"
    );
}

/// The position is the one the export describes.
///
/// As of round nine: `value = -18.4436` victory points and
/// `win_probability = 0.3321`, which is the `-18.44` / `0.332` the export
/// carried. As of round ten: `-19.7525` and `0.3212`, the menu weight having
/// come down. Those are *not* asserted — a `duels-eval` round is allowed to
/// move them, and pinning them would turn this file into a golden-values test
/// for the whole evaluation. What is asserted is the position.
#[test]
fn the_flagged_position_reconstructs() {
    let state = replay(1, &flagged_moves());
    assert_eq!(state.turn(), 13);
    assert_eq!(state.age(), 1);
    assert_eq!(state.current_player(), Player::One);
    assert_eq!(state.player(Player::One).coins(), 0);
    assert_eq!(state.player(Player::Two).coins(), 9);
    assert_eq!(
        engine::legal_actions(&state).len(),
        5,
        "the export listed five legal actions"
    );
    // The flagged move is in the history, and it really did build the wonder.
    assert!(state
        .player(Player::One)
        .has_built_wonder(wonder("the-great-lighthouse")));
}

/// The menu term is the whole of the "every move looks like a loss" anomaly.
#[test]
fn every_action_looks_worse_than_standing_still_and_the_menu_is_why() {
    let state = replay(1, &flagged_moves());
    let me = state.current_player();
    let legal = engine::legal_actions(&state);

    let with_menu = Config::default();
    let root = Root::new(&state, me, with_menu);
    let standing = evaluate(&state, me, &root);
    let best = legal
        .iter()
        .map(|&a| expected_value(&state, a, me, &root))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        best < standing,
        "the anomaly is gone: best action {best} against a standing {standing}"
    );
    // ...and it is not a rounding artefact. **Round ten shrank it by about
    // half**, which is what a 32% cut to the menu weight does to an anomaly
    // the menu is the whole of: the export's `0.332` against `0.270` — a gap
    // of `0.062`, which `Config::v9()` still reproduces to the fourth
    // decimal — is now `0.321` against `0.290`, a gap of `0.032`.
    //
    // The threshold is `0.02` rather than the `0.05` round nine could assert,
    // and this is deliberately *not* a quiet relaxation: the reason it moved
    // is round ten's own headline change, so the round-nine reading is pinned
    // alongside it. What that turns this test into is a measurement of the
    // mitigation — the anomaly is still there and still menu-shaped, and it is
    // now half the size.
    let age = state.age();
    let gap = win_probability_from_value(standing, age) - win_probability_from_value(best, age);
    assert!(
        gap > 0.02,
        "the gap is too small to be the thing that was flagged: {gap}"
    );
    let r9 = Root::new(&state, me, Config::v9());
    let standing_v9 = evaluate(&state, me, &r9);
    let best_v9 = legal
        .iter()
        .map(|&a| expected_value(&state, a, me, &r9))
        .fold(f64::NEG_INFINITY, f64::max);
    let gap_v9 =
        win_probability_from_value(standing_v9, age) - win_probability_from_value(best_v9, age);
    assert!(
        gap_v9 > 0.05,
        "round nine's menu weight no longer reproduces the flagged gap: {gap_v9}"
    );
    assert!(
        gap < gap_v9,
        "round ten was supposed to shrink this anomaly, not grow it: {gap} \
         against round nine's {gap_v9}"
    );

    // The same position with the opponent-menu term switched off on both
    // sides. `deny_chain_gift` switches itself *back on* when `menu.lambda` is
    // zero (it is the term the menu subsumed), so it has to be held off too or
    // this is a comparison of two changes.
    let bare = Config {
        eval: EvalWeights {
            menu: MenuWeights {
                lambda: 0.0,
                ..with_menu.eval.menu
            },
            deny_chain_gift: 0.0,
            ..with_menu.eval
        },
        ..with_menu
    };
    let bare_root = Root::new(&state, me, bare);
    let bare_standing = evaluate(&state, me, &bare_root);
    let bare_best = legal
        .iter()
        .map(|&a| expected_value(&state, a, me, &bare_root))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        bare_best > bare_standing,
        "with the menu off the ordering should reverse: best {bare_best} \
         against a standing {bare_standing}"
    );

    // And the mechanism, stated directly: the term is positive for the player
    // about to move and negative once the turn has passed.
    let mine =
        duels_eval::menu::menu_term(&state, me, root.age(), root.menu(), &with_menu.eval.menu);
    assert!(mine > 0.0, "the mover's own menu reads {mine}");
    for &action in &legal {
        let mut next = state;
        let outcomes = engine::chance_outcomes(&state, action);
        let (outcome, _) = outcomes[0];
        if engine::apply_with_outcome(&mut next, action, &outcome).is_err() || next.is_over() {
            continue;
        }
        if next.current_player() == me {
            // An extra turn would keep the sign; none of these five grant one.
            continue;
        }
        let theirs =
            duels_eval::menu::menu_term(&next, me, root.age(), root.menu(), &with_menu.eval.menu);
        assert!(
            theirs < 0.0,
            "after {action:?} the menu reads {theirs} for the same player"
        );
    }
}

/// The wonder-build half of the flag, at the decision that actually made it.
///
/// One move earlier, with seven coins in hand, the evaluation ranks all three
/// ways of burying a card under The Great Lighthouse above every ordinary card
/// build available. As of round nine the best `BuildWonder` reads `-14.71`
/// victory points against `-21.40` for the best `Build`, and switching the menu
/// off (so the two are read on the same side of the tempo term) puts the wonder
/// at `-7.85` against a standing `-19.21` — an eleven-point gain for seven
/// coins, of which ten is `terms::resource_bill` and four
/// `terms::development_value`.
///
/// **Round ten's lower menu weight moves both readings by the same `+2.19`**,
/// to `-12.51` against `-19.21`, leaving the `6.70`-point preference between
/// them untouched to the second decimal. That is worth recording rather than
/// glossing: unlike the anomaly in the test above, this half of the flag is
/// *not* a menu artefact — the five candidates here all hand the turn to the
/// same opponent menu, so the term is common to them and cancels out of the
/// comparison entirely.
#[test]
fn the_wonder_build_outranks_every_card_build_at_the_flagged_decision() {
    let mut moves = flagged_moves();
    moves.truncate(moves.len() - 2);
    let state = replay(1, &moves);
    assert_eq!(state.turn(), 11);
    assert_eq!(state.current_player(), Player::One);
    assert_eq!(
        state.player(Player::One).coins(),
        7,
        "test setup: this is the decision with seven coins in hand"
    );

    let me = state.current_player();
    let root = Root::new(&state, me, Config::default());
    let mut best_wonder = f64::NEG_INFINITY;
    let mut best_card = f64::NEG_INFINITY;
    for action in engine::legal_actions(&state) {
        let v = expected_value(&state, action, me, &root);
        match action {
            Action::BuildWonder { wonder: w, .. } if w == wonder("the-great-lighthouse") => {
                best_wonder = best_wonder.max(v)
            }
            Action::Build { .. } => best_card = best_card.max(v),
            _ => {}
        }
    }
    assert!(
        best_wonder.is_finite() && best_card.is_finite(),
        "test setup: both kinds of action have to be legal here"
    );
    assert!(
        best_wonder > best_card,
        "the flagged preference is gone: the wonder reads {best_wonder} \
         against a best card build of {best_card}. That may well be an \
         improvement -- but it is the thing the project owner flagged, so a \
         round that changes it should say so rather than let this test be \
         quietly relaxed."
    );
}
