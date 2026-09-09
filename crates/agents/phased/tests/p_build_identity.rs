//! The bit-identity guard for the frozen-`p_build` fix in `duels-eval`.
//!
//! # What was wrong and what changed
//!
//! `duels_eval::WonderModel::Rationed` shipped reading
//! `duels_eval::terms::wonder_p_build` **once from the root position**, caching
//! it on `duels_eval::Root`, and reusing that one number for every state scored
//! against that `Root`. `p_build` is a quantity about the *position* — the
//! wonder slots and the decisions left, both of which change on every move —
//! so caching it meant a leaf was priced by the root's turn number rather than
//! its own. It is now derived from the state being scored, and
//! `wonder_potential_rationed` no longer accepts it as an argument at all.
//!
//! # Why this file lives here and what it proves
//!
//! `duels-eval`'s own guards can say the *term* is a function of its state.
//! Only a caller can say what that did to an agent's *decisions*, which is the
//! claim that matters. `phased` is the right caller to ask: it builds one
//! `Root` per decision, so if anything was ever going to be unaffected by a
//! root-fixed read it would be this agent.
//!
//! Two things are pinned, and they say opposite things on purpose:
//!
//! * [`the_default_configuration_is_unchanged_by_the_p_build_fix`] — the
//!   default is `WonderModel::Flat`, which never read the cached value, so
//!   every decision of every game must be move-for-move what it was before.
//!   This is the load-bearing no-regression claim: `phased` and `mcts-eval`
//!   both ship on this path.
//! * [`the_rationed_configuration_is_deliberately_not_identical`] — under
//!   `WonderModel::Rationed` the fix *must* change decisions, or it fixed
//!   nothing. Without this half the file above would be asserting that a pile
//!   of dead code is dead.
//!
//! Note that "one `Root` per decision" was never the same thing as "the root
//! is the state being scored", which is the subtlety that makes the second
//! test come out the way it does: `duels_eval::expected_value` applies the
//! candidate action and scores the **post-action** state against a `Root` read
//! **pre-action**. So even at one ply the frozen read was a move stale. The
//! pinned pre-fix hash below is what that cost.
//!
//! # How the pinned constants were obtained
//!
//! By running this file against the pre-fix commit — the harness compiles
//! unchanged there, because it drives the agent through `Agent::choose` and
//! touches no signature the fix moved — and recording what it printed. The
//! test bodies then assert the post-fix run reproduces the flat hash and does
//! not reproduce the rationed one.
//!
//! Both are read against `Config::v9()`, which is the evaluation that commit
//! shipped. They were written as `Config::default()` when round nine's default
//! *was* that evaluation; round ten moved the default (one weight, the
//! opponent-menu `lambda`) and the constants moved with the name rather than
//! with the thing they describe. Round ten re-addressed them to `v9()` and
//! pinned the new default's own digest beside them, so this file now guards
//! two claims: that the `p_build` fix was inert where it had to be, and that
//! the shipping evaluation cannot move without an agent's decisions saying so.

use duels_agent_phased::{Config, EvalWeights, PhasedAgent, WonderModel};
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, Player};
use rand::{rngs::StdRng, SeedableRng};

/// Seeds driven end to end. Twelve whole games is about 865 decisions, each
/// scored over every legal candidate and every chance outcome, and still a
/// few seconds inside the default `cargo test` path.
const SEEDS: u64 = 12;

/// FNV-1a over the `Debug` form of every action played, in order, with the
/// turn number interleaved so a transposition cannot cancel out. Hand-rolled
/// because a `DefaultHasher` is explicitly not stable across releases and this
/// constant has to mean the same thing next year.
fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Play `SEEDS` whole self-play games with `config` on both seats and return
/// `(decisions, hash)` over the entire move sequence.
///
/// Everything that consumes randomness is seeded from the game seed, so this
/// is a pure function of `config` — which is what lets one number stand in for
/// "the agent decided the same way".
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

/// The rationed configuration the round-nine sweep measured: the model on, at
/// the `wonder_potential` weight that sweep landed on, over the evaluation
/// round nine shipped.
///
/// [`Config::v9`] rather than [`Config::default`] on purpose. The pre-fix
/// digest below was recorded against round nine's evaluation, so reading it
/// against a later default would make the comparison a measurement of two
/// changes at once instead of the one it is about.
fn rationed() -> Config {
    let d = Config::v9();
    Config {
        wonder_model: WonderModel::Rationed,
        eval: EvalWeights {
            wonder_potential: 1.25,
            ..d.eval
        },
        ..d
    }
}

/// **The no-regression claim.** `WonderModel::Flat` is the default and never
/// read the cached `p_build`, so removing the cache cannot have moved a single
/// decision on the path both shipping consumers use.
///
/// The pinned digest is read against [`Config::v9`] rather than
/// [`Config::default`], and that is a correction rather than a weakening.
/// Round nine's default is the configuration the pre-fix recording was taken
/// under, so `v9()` is the configuration this claim is *about*; addressing it
/// as "the default" only worked for as long as the default did not move.
/// **Round ten moved it** — one weight, `duels_eval::MenuWeights::lambda` —
/// and `phased` plays 856 decisions rather than 865 under the new one, which
/// has nothing to do with `p_build` and everything to do with a different
/// evaluation choosing different moves. Pinning the round-ten default
/// alongside it keeps the shipping path guarded going forward.
#[test]
fn the_default_configuration_is_unchanged_by_the_p_build_fix() {
    // Recorded from the pre-fix tree (`651d451`), over the same twelve seeds,
    // and reproduced exactly after the fix -- under the configuration that
    // tree shipped, which is now `Config::v9()`.
    let (decisions, hash) = self_play_digest(Config::v9());
    assert_eq!(
        (decisions, hash),
        (865, 0x58e8_6d3d_4266_1389),
        "round nine's evaluation moved: {decisions} decisions, hash {hash:#018x}"
    );
    // ...and the shipping default, pinned from round ten on. A later round
    // that moves `Config::default` is expected to fail here and re-record it,
    // exactly as round ten did: this is the guard that a *silent* change to
    // the evaluation cannot reach an agent's decisions unnoticed.
    let (decisions, hash) = self_play_digest(Config::default());
    assert_eq!(
        (decisions, hash),
        (856, 0x7d61_4d78_2e87_ef73),
        "the default evaluation moved: {decisions} decisions, hash {hash:#018x}"
    );
}

/// **The non-vacuity claim.** Under `WonderModel::Rationed` the fix has to
/// change what the agent plays. The pinned hash is the pre-fix behaviour, and
/// reproducing it would mean `p_build` is still being read from somewhere other
/// than the state being scored.
#[test]
fn the_rationed_configuration_is_deliberately_not_identical() {
    let (decisions, hash) = self_play_digest(rationed());
    // Recorded from the pre-fix tree (`651d451`), over the same twelve seeds.
    // Post-fix these twelve games run `(834, 0xfd46_4165_ca00_2500)` — 23
    // fewer decisions, because the games play out differently, not because
    // anything got cheaper.
    const PRE_FIX: (u64, u64) = (857, 0xc6bf_53c0_8742_62c2);
    assert_ne!(
        (decisions, hash),
        PRE_FIX,
        "the rationed model still plays its pre-fix game, so p_build is \
         still frozen somewhere"
    );
    // ...and it has not collapsed into the configuration it is an option on
    // top of either, which would be the other way to accidentally make the
    // option inert.
    let (_, base_hash) = self_play_digest(Config::v9());
    assert_ne!(hash, base_hash);
}
