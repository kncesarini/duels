//! The bit-identity guard for this crate's **eleventh** round of work.
//!
//! # What round eleven is, and what it deliberately is not
//!
//! [`duels_eval::TermWeights::science`] — the multiplier on the science-ladder
//! term — is built once in `Root::new` from the *root* position's
//! distinct-symbol count and reused for every state scored against that
//! `Root`. For `phased` that is a move stale; for `mcts-eval`, which builds one
//! `Root` per search tree, it is however many plies deep the leaf is.
//!
//! Round eleven adds [`ScienceProgress::Leaf`], which re-reads the progress half
//! of `c_sci` from the state being scored while leaving the magnitude half
//! (`M_sci^alpha_m`) root-fixed, and **leaves the default alone**. Like round
//! nine, this file therefore proves the *other* identity rather than
//! snapshotting a new generation: that the option is genuinely off, and that
//! where the root and the scored position coincide the new arithmetic is the
//! old arithmetic bit for bit.
//!
//! It is an option and not a plain fix on purpose, and the reason is worth
//! stating here because it is the one thing a reader is likely to want to
//! "clean up". Root-fixing the commitment weights is **not** an accident of
//! caching the way `terms::wonder_p_build` was: `blend`'s "Root-fixing"
//! section argues for it, and
//! `duels_eval`'s `a_committing_move_is_scored_under_the_root_weights_not_its_own`
//! pins it. That argument is sound at one ply and stale at ten, so the two
//! readings are both right somewhere and the choice belongs in `Config`. See
//! [`ScienceProgress`] and the `blend` module docs.
//!
//! # What is proved here
//!
//! * [`the_option_is_off_by_default_and_in_every_generation_snapshot`] — the
//!   default and `v1`-`v9` all read [`ScienceProgress::Root`], so no pinned
//!   generation's arithmetic moved.
//! * [`scoring_the_root_position_itself_is_bit_identical_under_both_readings`]
//!   — the load-bearing identity. Where the scored state *is* the root, the
//!   `Leaf` reading must reproduce the `Root` reading exactly, over every
//!   position of real games and both players. This is what says the
//!   reconstruction `M_sci^alpha_m · prog^beta_prog` is the same two factors
//!   in the same order rather than a rewrite that rounds differently.
//! * [`the_option_is_not_a_no_op_at_depth`] — the non-vacuity half. A state
//!   scored against an *earlier* root, where the science race has moved on,
//!   must score differently. Without this the file above would be asserting
//!   that a pile of dead code is dead.

use duels_core::{engine, GameState, Player};
use duels_eval::{
    evaluate, Blend, Config, EvalWeights, RailModel, Root, ScienceProgress, ScienceWeights,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The two readings, otherwise identical, with a ladder that pays from the
/// first symbol so the multiplier under test is never multiplied by a zero
/// rung.
fn configs() -> (Config, Config) {
    let base = Config {
        eval: EvalWeights {
            science: ScienceWeights {
                ladder: [0.0, 5.0, 10.0, 20.0, 40.0, 80.0],
                ..Config::default().eval.science
            },
            ..Config::default().eval
        },
        ..Config::default()
    };
    let leaf = Config {
        blend: Blend {
            science_progress: ScienceProgress::Leaf,
            ..base.blend
        },
        ..base
    };
    (base, leaf)
}

/// Whole seeded games, driven by a cheap deterministic policy, so the
/// comparisons run over real positions from all three ages rather than over
/// hand-built ones. The same walker `round_nine_identity.rs` uses.
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
fn the_option_is_off_by_default_and_in_every_generation_snapshot() {
    for (name, config) in [
        ("default", Config::default()),
        ("v1", Config::v1()),
        ("v2", Config::v2()),
        ("v3", Config::v3()),
        ("v4", Config::v4()),
        ("v5", Config::v5()),
        ("v6", Config::v6()),
        ("v7", Config::v7()),
        ("v8", Config::v8()),
        ("v9", Config::v9()),
    ] {
        assert_eq!(
            config.blend.science_progress,
            ScienceProgress::Root,
            "{name} does not read the root's science progress, so round eleven \
             silently redefined a pinned generation"
        );
    }
    // ...and the params string says which, so two results files from either
    // side of this round are distinguishable.
    assert!(Config::default().params_string().contains("sciprog=root"));
    let (_, leaf) = configs();
    assert!(leaf.params_string().contains("sciprog=leaf"));
}

/// **The load-bearing identity.** Root and scored position coincide, so the
/// two readings must agree bit for bit.
#[test]
fn scoring_the_root_position_itself_is_bit_identical_under_both_readings() {
    let (root_cfg, leaf_cfg) = configs();
    let mut positions = 0u32;
    for seed in 0..8u64 {
        walk(seed, |state| {
            for p in [Player::One, Player::Two] {
                let a = Root::new(state, p, root_cfg);
                let b = Root::new(state, p, leaf_cfg);
                // The stored weight and the re-read one, directly.
                let distinct = state.player(p).distinct_science();
                assert_eq!(
                    b.weights(p).science_at(distinct, &leaf_cfg.blend).to_bits(),
                    a.weights(p).science.to_bits(),
                    "the re-read multiplier disagreed at the root itself \
                     (seed {seed}, turn {}, {p:?})",
                    state.turn()
                );
                // ...and the whole evaluation built on top of it, wherever
                // `evaluate` really does score the state it was handed.
                //
                // It does not always: with `PendingModel::Completed` in force
                // (the default) a position with an unresolved pending is
                // scored by *finishing the turn first*, so the state that
                // reaches `player_value` is one or more actions deeper than
                // the one passed in — and resolving a pending is exactly how a
                // Mausoleum retrieval or a Law token adds a symbol. That is
                // the option working, not the identity breaking, so those
                // positions belong to the non-vacuity test rather than this
                // one.
                if state.pending().is_none() {
                    assert_eq!(
                        evaluate(state, p, &b).to_bits(),
                        evaluate(state, p, &a).to_bits(),
                        "evaluate disagreed at the root itself (seed {seed}, \
                         turn {}, {p:?})",
                        state.turn()
                    );
                }
                positions += 1;
            }
        });
    }
    assert!(positions > 500, "only {positions} positions compared");
}

/// **The non-vacuity half.** A position scored against a `Root` read earlier
/// in the same game, where the science race has since moved on, must score
/// differently — and must read as *more* committed, not less.
///
/// Real positions rather than a hand-built one, deliberately. The obvious
/// `StateBuilder` fixture does not work here and the reason is instructive:
/// a builder position has only the slots it was given, so
/// `decisions_left_eff` is tiny, `missing > decisions_left_eff`, and
/// `duels_strategy`'s magnitude model correctly reports the race as hopeless
/// (`M_sci = 0`). With `m_sci_alpha == 0` the multiplier is pinned at `1.0`
/// however many symbols the leaf holds — the dead-race behaviour
/// `blend`'s `re_reading_progress_moves_the_science_multiplier_but_not_a_dead_race`
/// pins on purpose. So the fixture has to be a position with a real game's
/// worth of future in it.
///
/// Rails are switched off in both arms so a decided position cannot
/// short-circuit `evaluate` and hide the term under test.
#[test]
fn the_option_is_not_a_no_op_at_depth() {
    let (mut root_cfg, mut leaf_cfg) = configs();
    root_cfg.rails = RailModel::Off;
    leaf_cfg.rails = RailModel::Off;

    let mut compared = 0u32;
    let mut differed = 0u32;
    for seed in 0..12u64 {
        // Every scoreable position of one game, in order.
        let mut line: Vec<GameState> = Vec::new();
        walk(seed, |state| {
            if state.pending().is_none() && !state.is_over() {
                line.push(*state);
            }
        });

        for (i, early) in line.iter().enumerate() {
            let root_fixed = Root::new(early, Player::One, root_cfg);
            let leaf_read = Root::new(early, Player::One, leaf_cfg);
            // A dead race pins the multiplier at 1.0 either way, so those
            // roots have nothing to say here.
            if root_fixed.commitment(Player::One).m_sci_alpha <= 0.0 {
                continue;
            }
            let d_early = early.player(Player::One).distinct_science();
            for later in &line[i + 1..] {
                // Only `Player::One` may have advanced, so the sign of the
                // difference is attributable to one player's weight.
                if later.player(Player::One).distinct_science() <= d_early
                    || later.player(Player::Two).distinct_science()
                        != early.player(Player::Two).distinct_science()
                {
                    continue;
                }
                let a = evaluate(later, Player::One, &root_fixed);
                let b = evaluate(later, Player::One, &leaf_read);
                compared += 1;
                if a.to_bits() != b.to_bits() {
                    differed += 1;
                    assert!(
                        b > a,
                        "more distinct symbols than the root knew about should \
                         read as more committed, not less (seed {seed}, {b} vs {a})"
                    );
                }
            }
        }
    }
    assert!(
        compared > 100,
        "only {compared} (root, leaf) pairs found — the fixture is too thin to \
         say anything"
    );
    assert!(
        differed > 0,
        "the leaf reading changed nothing over {compared} pairs where the \
         science race had advanced, so the option is inert"
    );
}
