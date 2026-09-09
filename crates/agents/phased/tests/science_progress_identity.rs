//! The bit-identity guard for `duels-eval`'s round-eleven
//! [`ScienceProgress`] option, from the caller's side.
//!
//! # What round eleven did, and why it is an option
//!
//! `duels_eval::TermWeights::science` — the multiplier on the science-ladder
//! term — is built once in `Root::new` from the root position's
//! distinct-symbol count. A search that prices leaves at depth therefore
//! judges a player who has gone from two distinct symbols to five by the
//! importance the science term had at two. `ScienceProgress::Leaf` re-reads
//! the progress half of `c_sci` off the state being scored;
//! `ScienceProgress::Root` — the default — keeps the shipped behaviour.
//!
//! **The interesting part is why it is not simply a fix.** The obvious reading
//! is that this is another frozen-`p_build` (see `p_build_identity.rs` next to
//! this file): a quantity about the position, cached from a different
//! position. It is not quite, and the difference decides the shape of the
//! change:
//!
//! * `p_build` was cached **by accident**, inside a term's *value*. Nothing
//!   argued for it and nothing tested it, so deriving it from the scored state
//!   was a plain correctness fix with no "off" state to gate.
//! * The commitment weights are root-fixed **on purpose**, in the *weight*
//!   layer. `duels_eval::blend`'s "Root-fixing" section argues for it and
//!   `duels_eval`'s `a_committing_move_is_scored_under_the_root_weights_not_its_own`
//!   pins it: at one ply, a weight that moved with the candidate action would
//!   credit a committing move twice — once through the term's contents, which
//!   should move, and again through the multiplier on them, which should not.
//!
//! That argument is sound at one ply and stale at ten, so both readings are
//! right somewhere and the choice belongs in `Config`. `phased` is the agent
//! that makes the "sound at one ply" half concrete, and what it turns out to
//! show is sharper than expected: the option moves candidate *scores* here —
//! `expected_value` scores a post-action state against a pre-action `Root`, so
//! even one ply is a move stale — but it moves them so little that it never
//! flips a decision across twelve whole games. The blend's shape is why (see
//! [`the_leaf_reading_changes_the_score_even_at_one_ply`]), and it is also why
//! the same option is a real change ten plies out.
//!
//! # What is pinned here
//!
//! * [`the_default_configuration_is_unchanged_by_the_round_eleven_option`] — the
//!   load-bearing no-regression claim. `ScienceProgress::Root` is the default,
//!   `player_value` still spells it as the same bare `w.science`, so every
//!   decision of every game must be move-for-move what it was before. `phased`
//!   and `mcts-eval` both ship on this path.
//! * [`the_leaf_reading_changes_the_score_even_at_one_ply`] — the
//!   non-vacuity claim, plus the measurement that explains the shape of this
//!   change. `duels_eval::expected_value` scores the *post-action* state
//!   against a `Root` read *pre-action*, so even at one ply the root's symbol
//!   count is a move stale (exactly as `p_build_identity.rs` observes for its
//!   own frozen read) and the option does move candidate scores. It moves
//!   them by only a few percent of one term, though, and never by enough to
//!   flip a one-ply argmax across the twelve games above — which is why the
//!   default stayed put and why this option is aimed at a search that scores
//!   leaves ten plies out, not at `phased`.
//!
//! # How the pinned constants were obtained
//!
//! From `p_build_identity.rs`, unchanged: the two files drive the same twelve
//! seeds through the same harness, so a change to either configuration's
//! decisions breaks both. **Two** pairs are pinned here for the same reason
//! that file pins two — `Config::v9()` is the configuration round nine's
//! recording was taken under, and `Config::default()` is round ten's fitted
//! `MenuWeights::lambda`, which plays 856 decisions rather than 865 for
//! reasons that have nothing to do with this option. Keeping the literals
//! here rather than importing them is deliberate: a shared constant that some
//! future round updates in one place would quietly re-baseline both guards at
//! once.

use duels_agent_phased::{expected_value, Blend, Config, PhasedAgent, Root, ScienceProgress};
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, Player};
use rand::{rngs::StdRng, SeedableRng};

/// Seeds driven end to end, matching `p_build_identity.rs` so the default
/// hash below is comparable with the one that file pins.
const SEEDS: u64 = 12;

/// FNV-1a over the `Debug` form of every action played, in order, with the
/// turn number interleaved so a transposition cannot cancel out. Hand-rolled
/// because a `DefaultHasher` is explicitly not stable across releases and this
/// constant has to mean the same thing next year. Byte-for-byte
/// `p_build_identity.rs`'s, on purpose: two hashes are only comparable if the
/// hashers are.
fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Play `SEEDS` whole self-play games with `config` on both seats and return
/// `(decisions, hash)` over the entire move sequence.
fn self_play_digest(config: Config) -> (u64, u64) {
    let mut decisions = 0u64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for seed in 0..SEEDS {
        let mut one = PhasedAgent::with_config(seed ^ 0xA1, config);
        let mut two = PhasedAgent::with_config(seed ^ 0xB2, config);
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5EED);
        for _ in 0..500 {
            if st.is_over() {
                break;
            }
            let legal = engine::legal_actions(&st);
            let obs = st.observation();
            let action = if st.current_player() == Player::One {
                one.choose(&obs, &legal, Budget::Nodes(1))
            } else {
                two.choose(&obs, &legal, Budget::Nodes(1))
            };
            assert!(legal.contains(&action), "illegal move on seed {seed}");
            h = fnv1a(&st.turn().to_le_bytes(), h);
            h = fnv1a(format!("{action:?}").as_bytes(), h);
            decisions += 1;
            engine::apply(&mut st, action, &mut rng).unwrap();
        }
        assert!(st.is_over(), "seed {seed} did not finish");
    }
    (decisions, h)
}

/// The round-eleven option switched on, with nothing else changed.
fn leaf_progress() -> Config {
    let d = Config::default();
    Config {
        blend: Blend {
            science_progress: ScienceProgress::Leaf,
            ..d.blend
        },
        ..d
    }
}

/// **The no-regression claim.** `ScienceProgress::Root` is the default and is
/// spelled as the same expression it always was, so adding the option cannot
/// have moved a single decision on the path both shipping consumers use.
///
/// Both of `p_build_identity.rs`'s pinned configurations are checked, for the
/// same reason it checks both: `v9()` is what round nine's digest was recorded
/// under, and `default()` is the shipping path round ten moved.
///
/// Both digests were re-recorded when `duels_core::engine`'s R-105/R-110
/// chance model was sharpened; `p_build_identity.rs`'s copy of this claim
/// carries the reasoning. In short: `expected_value` averages over
/// `engine::chance_outcomes`, so a rules-level change to the reveal
/// distribution moves `phased`'s decisions under every `Config` generation at
/// once, and neither claim this test makes is weakened by that.
#[test]
fn the_default_configuration_is_unchanged_by_the_round_eleven_option() {
    let (decisions, hash) = self_play_digest(Config::v9());
    assert_eq!(
        (decisions, hash),
        (867, 0x737d_0393_7152_9674),
        "round nine's evaluation moved: {decisions} decisions, hash {hash:#018x}"
    );
    let (decisions, hash) = self_play_digest(Config::default());
    assert_eq!(
        (decisions, hash),
        (857, 0xf022_060d_ad98_73ab),
        "the default evaluation moved: {decisions} decisions, hash {hash:#018x}"
    );
}

/// **The non-vacuity claim, and a measurement worth recording.**
///
/// The option really does reach `player_value` at one ply — `expected_value`
/// scores a post-action state against a pre-action `Root`, so a candidate that
/// adds a distinct symbol is scored under a weight the root did not have. This
/// test finds those candidates and asserts the two readings disagree on them.
///
/// What it also records is **how little that is worth at one ply**, which is
/// the reason the default stayed put and the reason this option exists for a
/// *search* rather than for `phased`. The blend's Hill curve is quartic
/// (`hill_n = 4`) and its midpoint is around `0.32`, while one extra symbol
/// out of six moves `c_sci` by a fraction of that — so a single move's worth
/// of progress moves the science multiplier by a few percent at most, and a
/// few percent of one term never flipped a one-ply argmax across all twelve
/// games of [`the_default_configuration_is_unchanged_by_the_round_eleven_option`]'s
/// harness. Ten plies of progress is a different quantity entirely, and that
/// is what `mcts-eval` scores.
#[test]
fn the_leaf_reading_changes_the_score_even_at_one_ply() {
    let root_cfg = Config::default();
    let leaf_cfg = leaf_progress();

    let mut differed = 0u32;
    let mut candidates = 0u32;
    let mut worst_relative = 0.0f64;
    for seed in 0..12u64 {
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5EED);
        while !st.is_over() {
            let legal = engine::legal_actions(&st);
            if legal.is_empty() {
                break;
            }
            let me = st.current_player();
            let a_root = Root::new(&st, me, root_cfg);
            let a_leaf = Root::new(&st, me, leaf_cfg);
            for &action in &legal {
                let x = expected_value(&st, action, me, &a_root);
                let y = expected_value(&st, action, me, &a_leaf);
                candidates += 1;
                if x.to_bits() != y.to_bits() {
                    differed += 1;
                    let scale = x.abs().max(1.0);
                    worst_relative = worst_relative.max((y - x).abs() / scale);
                }
            }
            // A deterministic policy, so the walk is a pure function of `seed`.
            let pick = (st.turn() as usize * 7 + seed as usize) % legal.len();
            if engine::apply(&mut st, legal[pick], &mut rng).is_err() {
                break;
            }
        }
    }
    assert!(
        candidates > 1000,
        "only {candidates} candidates scored — too thin to say anything"
    );
    assert!(
        differed > 0,
        "the leaf reading changed no score over {candidates} candidates, so \
         the option never reaches player_value"
    );
    // The recorded magnitude. A loose bound rather than a pinned number: the
    // claim is "this is small at one ply", and a tight literal here would
    // break on any unrelated re-weighting.
    assert!(
        worst_relative < 0.25,
        "one ply of science progress moved a candidate's score by {:.1}% -- \
         far more than the blend's shape allows, so something else changed",
        worst_relative * 100.0
    );
}
