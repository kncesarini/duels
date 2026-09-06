//! The bit-identity guard for this crate's second round of work.
//!
//! Five things changed at once: the military term became a smoothed step
//! function, the coin terms collapsed into one smooth function, the economy
//! term became a resource bill, chain equity and an opponent-menu term were
//! added, and the `next_age_start` magnitudes were halved. Every one of them
//! is a [`Config`] option, and [`Config::v1`] sets all five back — so the
//! obligation this project's conventions impose is to prove that
//! `Config::v1()` is not merely *similar* to the agent that shipped, but the
//! same arithmetic.
//!
//! The pattern is `mcts-uct`'s: a **verbatim copy** of the previous
//! evaluation lives in this file, whole seeded games are driven through both
//! it and the real agent under `Config::v1()`, and equality is asserted move
//! for move — plus, position by position, on the raw `f64` bits of every
//! candidate's score, which is the sharper of the two (a difference too small
//! to change a decision still fails).
//!
//! The copy calls the same public `terms::` functions the original did. Those
//! functions were not edited: `military_position`, `race_liquidity`,
//! `coin_shortfall`, `average_trade_price`, `chain_gift_exposure`,
//! `next_age_start`, `science_ladder`, `military_urgency`, `wonder_potential`
//! and `card_and_token_vp` are all byte-for-byte what they were, and
//! `development_value` is `development_value_with(.., true)`, its previous
//! and still-default behaviour.

use duels_agent_phased::{terms, Config, PhasedAgent, Root};
use duels_agents_api::{Agent, Budget};
use duels_core::engine;
use duels_core::scoring::{self, GameResult};
use duels_core::{Action, GameState, Player};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Scores within this distance of the best are treated as tied — the
/// original's constant, copied because the tie set feeds the RNG draw and so
/// changes which move comes out.
const TIE_EPSILON: f64 = 1e-6;

// ---------------------------------------------------------------------------
// A verbatim copy of the evaluation as it stood before this round of work.
// ---------------------------------------------------------------------------

fn legacy_player_value(state: &GameState, p: Player, root: &Root) -> f64 {
    let e = &root.config().eval;
    let w = root.weights(p);
    let breakdown = scoring::breakdown(state, p);

    let points = w.vp * e.vp_projection * terms::card_and_token_vp(&breakdown);
    let liquidity = w.liquidity * e.coins_div3 * f64::from(breakdown.coins);
    let development = w.development
        * e.development
        * terms::development_value(state, p, root.supply(), e.development_take_rate);
    let economy = w.economy
        * (e.coin_safety_penalty * -terms::coin_shortfall(state, p, e.coin_safety_floor)
            + e.resource_vulnerability * -terms::average_trade_price(state, p));

    let science = w.science * e.science_ladder * terms::science_ladder(state, p, &e.science);
    let military = w.military * e.military_position * terms::military_position(state, p);
    let race_liquidity = w.race_liquidity
        * e.race_card_liquidity
        * terms::race_liquidity(state, p, e.race_liquidity_cap);

    let urgency = e.military_endgame_urgency * terms::military_urgency(state, p);
    let start = terms::next_age_start(state, p, e);
    let wonders = e.wonder_potential * terms::wonder_potential(state, p);
    let gift = -e.deny_chain_gift * terms::chain_gift_exposure(state, p, root.age());

    points
        + liquidity
        + development
        + economy
        + science
        + military
        + race_liquidity
        + urgency
        + start
        + wonders
        + gift
}

fn legacy_evaluate(state: &GameState, me: Player, root: &Root) -> f64 {
    if let Some(result) = state.result() {
        return match result {
            GameResult::Win { winner, .. } if winner == me => root.config().eval.instant_result,
            GameResult::Win { .. } => -root.config().eval.instant_result,
            GameResult::Draw => 0.0,
        };
    }
    legacy_player_value(state, me, root) - legacy_player_value(state, me.other(), root)
}

fn legacy_expected_value(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => legacy_evaluate(&next, me, root),
            Err(_) => legacy_evaluate(state, me, root),
        };
        acc += prob * value;
    }
    acc + root.denial_term(action)
}

/// A verbatim copy of `PhasedAgent::choose`, including its RNG usage, so the
/// two agents draw from their streams in lockstep and a divergence can only
/// come from the evaluation.
struct LegacyAgent {
    rng: StdRng,
    config: Config,
}

impl LegacyAgent {
    fn new(seed: u64) -> LegacyAgent {
        LegacyAgent {
            rng: StdRng::seed_from_u64(seed),
            config: Config::v1(),
        }
    }
}

impl Agent for LegacyAgent {
    fn spec(&self) -> duels_agents_api::AgentSpec {
        duels_agents_api::AgentSpec {
            name: "phased-legacy".to_string(),
            version: "1.0.0".to_string(),
            params: self.config.params_string(),
        }
    }

    fn choose(
        &mut self,
        obs: &duels_core::Observation,
        legal: &[Action],
        _budget: Budget,
    ) -> Action {
        assert!(!legal.is_empty());
        if legal.len() == 1 {
            return legal[0];
        }
        let me = obs.current_player;
        let base_state = obs.sample_state(&mut self.rng);
        let root = Root::new(&base_state, me, self.config);

        let mut scored: Vec<(Action, f64)> = Vec::with_capacity(legal.len());
        for &action in legal {
            scored.push((
                action,
                legacy_expected_value(&base_state, action, me, &root),
            ));
        }
        let Some(best_score) = scored.iter().map(|&(_, s)| s).fold(None, |m, s| match m {
            Some(b) if b >= s => Some(b),
            _ => Some(s),
        }) else {
            return legal[self.rng.gen_range(0..legal.len())];
        };
        let best: Vec<Action> = scored
            .iter()
            .filter(|&&(_, s)| (best_score - s).abs() <= TIE_EPSILON)
            .map(|&(a, _)| a)
            .collect();
        best[self.rng.gen_range(0..best.len())]
    }
}

// ---------------------------------------------------------------------------
// The assertions
// ---------------------------------------------------------------------------

fn advance(seed: u64, steps: usize) -> GameState {
    let mut st = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5A5A);
    for _ in 0..steps {
        let actions = engine::legal_actions(&st);
        if actions.is_empty() {
            break;
        }
        let a = actions[(st.turn() as usize * 7 + seed as usize) % actions.len()];
        engine::apply_quiet(&mut st, a, &mut rng).unwrap();
    }
    st
}

/// Position by position: under `Config::v1()` the live evaluation and the
/// verbatim copy agree on the raw bits of every candidate's score.
#[test]
fn v1_scores_every_candidate_bit_identically_to_the_previous_evaluation() {
    let mut positions = 0usize;
    let mut candidates = 0usize;
    for seed in 0..14u64 {
        for &steps in &[0usize, 5, 12, 19, 27, 34, 41, 48, 55] {
            let st = advance(seed, steps);
            if st.is_over() {
                continue;
            }
            let me = st.current_player();
            let root = Root::new(&st, me, Config::v1());
            for action in engine::legal_actions(&st) {
                let a = duels_agent_phased::expected_value(&st, action, me, &root);
                let b = legacy_expected_value(&st, action, me, &root);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "seed {seed} steps {steps}: {action:?} scored {a} now and {b} before"
                );
                candidates += 1;
            }
            positions += 1;
        }
    }
    assert!(positions > 60, "only {positions} positions exercised");
    assert!(candidates > 500, "only {candidates} candidates exercised");
}

/// ...and whole games, move for move, driven by two agents that differ only
/// in which evaluation they call.
#[test]
fn v1_plays_whole_games_move_for_move_like_the_previous_agent() {
    for seed in 0..12u64 {
        let mut new_one = PhasedAgent::with_config(seed * 2 + 1, Config::v1());
        let mut new_two = PhasedAgent::with_config(seed * 2 + 2, Config::v1());
        let mut old_one = LegacyAgent::new(seed * 2 + 1);
        let mut old_two = LegacyAgent::new(seed * 2 + 2);

        let mut a = engine::new_game(seed);
        let mut b = engine::new_game(seed);
        let mut rng_a = StdRng::seed_from_u64(seed ^ 0xC0FF_EE00);
        let mut rng_b = StdRng::seed_from_u64(seed ^ 0xC0FF_EE00);

        let mut moves = 0u32;
        while !a.is_over() {
            let legal_a = engine::legal_actions(&a);
            let legal_b = engine::legal_actions(&b);
            assert_eq!(
                legal_a, legal_b,
                "seed {seed} move {moves}: states diverged"
            );
            if legal_a.is_empty() {
                break;
            }
            let obs_a = a.observation();
            let obs_b = b.observation();
            assert_eq!(
                obs_a, obs_b,
                "seed {seed} move {moves}: observations diverged"
            );

            let (x, y) = match a.current_player() {
                Player::One => (
                    new_one.choose(&obs_a, &legal_a, Budget::Nodes(1)),
                    old_one.choose(&obs_b, &legal_b, Budget::Nodes(1)),
                ),
                Player::Two => (
                    new_two.choose(&obs_a, &legal_a, Budget::Nodes(1)),
                    old_two.choose(&obs_b, &legal_b, Budget::Nodes(1)),
                ),
            };
            assert_eq!(x, y, "seed {seed} move {moves}: chose {x:?} vs {y:?}");
            engine::apply(&mut a, x, &mut rng_a).unwrap();
            engine::apply(&mut b, y, &mut rng_b).unwrap();
            moves += 1;
            assert!(moves < 400);
        }
        assert_eq!(a.result(), b.result(), "seed {seed}: results differ");
        assert!(moves > 40, "seed {seed}: only {moves} moves played");
    }
}

/// The test above would pass vacuously if `Config::default()` happened to be
/// `Config::v1()`. It is not, and this says so out loud: the shipped default
/// really does score positions differently.
#[test]
fn the_default_configuration_is_not_the_legacy_one() {
    assert_ne!(Config::default(), Config::v1());
    let mut differed = 0usize;
    for seed in 0..8u64 {
        let st = advance(seed, 24);
        if st.is_over() {
            continue;
        }
        let me = st.current_player();
        let new = Root::new(&st, me, Config::default());
        let old = Root::new(&st, me, Config::v1());
        for action in engine::legal_actions(&st) {
            let a = duels_agent_phased::expected_value(&st, action, me, &new);
            let b = legacy_expected_value(&st, action, me, &old);
            if a.to_bits() != b.to_bits() {
                differed += 1;
            }
        }
    }
    assert!(
        differed > 0,
        "the default and legacy evaluations never disagreed, so the identity \
         test above proves nothing"
    );
}
