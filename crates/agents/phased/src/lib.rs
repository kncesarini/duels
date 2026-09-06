//! `duels-agent-phased`: a 1-ply [`Agent`] over [`duels_eval`], whose
//! evaluation weights are a continuous function of how committed each player
//! is to a win condition.
//!
//! # What is where
//!
//! Everything this agent knows about *judging a position* lives in
//! [`duels_eval`] — [`Config`] and every model enum, the commitment blend, the
//! terms, the opponent menu, the terminal rails, [`Root`], [`evaluate`] and
//! [`expected_value`], and the research record that produced all of them. Read
//! that crate's documentation for the design; this file is only the part that
//! turns scores into a move.
//!
//! The evaluation was extracted into its own crate so that more than one agent
//! can use it. This repository's rule is that **no agent crate depends on
//! another agent crate** (`CLAUDE.md`), so a shared evaluation has to sit
//! below the agents next to `duels-strategy` rather than inside whichever
//! agent built it first. The extraction was a pure refactor: `phased` plays
//! move-for-move identically, under every configuration snapshot, to the agent
//! that had the evaluation inlined.
//!
//! Every public item of `duels-eval` is re-exported here under the name it had
//! before, so `phased:base=v1,rails=off,...` spec strings, and every caller in
//! `duels-arena` and `duels-server`, are unaffected.
//!
//! # What this crate does
//!
//! One decision:
//!
//! 1. sample one concrete [`duels_core::GameState`] from the [`Observation`],
//!    purely as a
//!    vehicle for the engine's chance API — `greedy-ev`'s pattern, and nothing
//!    downstream reads a hidden identity, which
//!    `duels-eval/tests/determinization_invariance.rs` asserts bit for bit;
//! 2. build **one** [`Root`] — the only place the commitment blend is
//!    evaluated for this decision, which [`PhasedAgent::root_builds`] counts
//!    so that `tests::root_weights_are_built_exactly_once_per_choose` can pin
//!    it;
//! 3. score every legal action with [`expected_value`];
//! 4. return the best, ties inside `TIE_EPSILON` (`1e-6`) broken uniformly at
//!    random from this agent's own seeded stream.
//!
//! The budget is taken and ignored: this is a fixed 1-ply agent, and
//! `duels_arena::leaderboard::tests::one_ply_agents_ignore_their_budget`
//! proves it plays the same game at every budget.

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::{Action, Observation};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub use duels_eval::{
    blend, evaluate, expected_value, menu, rail_owner, rail_value, rails, terms, Blend, ChainTable,
    CoinModel, Commitment, Config, DevSupply, EconomyModel, EvalWeights, GuildPricing, GuildTable,
    MenuFloor, MenuOptions, MenuShieldPricing, MenuTables, MenuWeights, MilSmoothing,
    MilitaryModel, PendingModel, RailModel, Root, ScienceWeights, SupplyModel, TakeContext,
    TakeValue, TermWeights, WonderBudget, WonderModel, DESTROY_REPLACE_SHARE, MAX_PENDING_DEPTH,
    MAX_UNITS,
};

/// Scores within this distance of the best are treated as tied, and one is
/// chosen uniformly at random.
const TIE_EPSILON: f64 = 1e-6;

/// A 1-ply agent that re-reads what matters before every decision.
#[derive(Debug, Clone)]
pub struct PhasedAgent {
    rng: StdRng,
    config: Config,
    root_builds: u64,
}

impl PhasedAgent {
    /// A new agent seeded from `seed`, using [`Config::default`].
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, Config::default())
    }

    /// A new agent seeded from `seed`, with an explicit configuration.
    pub fn with_config(seed: u64, config: Config) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            config,
            root_builds: 0,
        }
    }

    /// A new agent driven by an existing RNG, so a caller can draw many
    /// independent agents from one stream.
    pub fn from_rng(rng: StdRng) -> Self {
        Self {
            rng,
            config: Config::default(),
            root_builds: 0,
        }
    }

    /// The configuration this agent is using.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// How many times this agent has built a [`Root`] — that is, how many
    /// times it has evaluated the commitment blend.
    ///
    /// Instrumentation for the root-fixing property: this must equal the
    /// number of [`Agent::choose`] calls that got past the trivial
    /// single-legal-action shortcut, however many candidate actions and
    /// chance outcomes each of them had to score. See
    /// `tests::root_weights_are_built_exactly_once_per_choose`.
    pub fn root_builds(&self) -> u64 {
        self.root_builds
    }
}

impl Agent for PhasedAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "phased".to_string(),
            version: "1.0.0".to_string(),
            params: self.config.params_string(),
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action], _budget: Budget) -> Action {
        assert!(
            !legal.is_empty(),
            "choose must not be called with no legal actions"
        );
        if legal.len() == 1 {
            return legal[0];
        }

        let me = obs.current_player;
        // Sampled once per call, purely as a vehicle for the engine's chance
        // API (which needs a concrete `GameState`) — `greedy-ev`'s pattern.
        // Nothing downstream reads a hidden identity, so it does not matter
        // which world this invents.
        let base_state = obs.sample_state(&mut self.rng);

        // The one and only place the blend is evaluated for this decision.
        let root = Root::new(&base_state, me, self.config);
        self.root_builds += 1;

        let mut scored: Vec<(Action, f64)> = Vec::with_capacity(legal.len());
        for &action in legal {
            scored.push((action, expected_value(&base_state, action, me, &root)));
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

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::{engine, GameState, Player};

    /// Walk a real game a few decisions in, with a deterministic policy, so
    /// there is a position with a real choice in it.
    fn advanced_game(seed: u64, steps: usize) -> GameState {
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x55);
        for _ in 0..steps {
            let actions = engine::legal_actions(&st);
            if actions.is_empty() {
                break;
            }
            let a = actions[(st.turn() as usize * 7 + seed as usize) % actions.len()];
            engine::apply(&mut st, a, &mut rng).unwrap();
        }
        st
    }

    /// The counting half of root-fixing. The behavioural half — the tables and
    /// the weights not moving once built — is pinned in `duels-eval` itself;
    /// this is the half only a caller can assert.
    #[test]
    fn root_weights_are_built_exactly_once_per_choose() {
        let mut agent = PhasedAgent::new(4);
        let st = advanced_game(9, 16);
        let legal = engine::legal_actions(&st);
        assert!(legal.len() > 1, "test setup: need a real choice");
        let obs = st.observation();

        agent.choose(&obs, &legal, Budget::Nodes(1));
        assert_eq!(
            agent.root_builds(),
            1,
            "one decision over {} candidates must evaluate the blend once",
            legal.len()
        );
        agent.choose(&obs, &legal, Budget::Nodes(1));
        assert_eq!(agent.root_builds(), 2);
    }

    #[test]
    fn spec_reports_the_expected_name_and_encoded_params() {
        let agent = PhasedAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "phased");
        assert_eq!(spec.version, "1.0.0");
        assert_eq!(spec.params, Config::default().params_string());
    }

    /// The re-export really is transparent: `duels-arena`'s spec-string parser
    /// and `duels-server`'s registry both name these through
    /// `duels_agent_phased::`, and every one of them has to keep resolving to
    /// the same type `duels-eval` defines.
    #[test]
    fn the_configuration_types_are_re_exported_under_their_old_names() {
        let cfg: Config = Config::v1();
        assert_eq!(cfg, duels_eval::Config::v1());
        let _: MilitaryModel = MilitaryModel::Band;
        let _: CoinModel = CoinModel::Smooth;
        let _: EconomyModel = EconomyModel::Bill;
        let _: RailModel = RailModel::On;
        let _: MenuShieldPricing = MenuShieldPricing::Differenced;
        let _: PendingModel = PendingModel::Completed;
        let _: WonderModel = WonderModel::Budget;
        let _: GuildPricing = GuildPricing::Projected;
        let _: MenuFloor = MenuFloor::DiscardAndWonder;
        let _: SupplyModel = SupplyModel::Dealt;
        let _: Blend = Blend::off();
        for base in [
            Config::v1(),
            Config::v2(),
            Config::v3(),
            Config::v4(),
            Config::v5(),
            Config::default(),
        ] {
            assert!(!base.params_string().is_empty());
        }
    }

    #[test]
    fn choosing_only_ever_returns_one_of_the_offered_actions() {
        let mut agent = PhasedAgent::new(99);
        let state = engine::new_game(99);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        for _ in 0..10 {
            assert!(legal.contains(&agent.choose(&obs, &legal, Budget::Nodes(1))));
        }
    }

    #[test]
    fn a_whole_game_of_self_play_terminates_and_stays_legal() {
        let mut a = PhasedAgent::new(1);
        let mut b = PhasedAgent::new(2);
        let mut st = engine::new_game(31);
        let mut rng = StdRng::seed_from_u64(77);
        for _ in 0..400 {
            if st.is_over() {
                break;
            }
            let legal = engine::legal_actions(&st);
            let obs = st.observation();
            let action = if st.current_player() == Player::One {
                a.choose(&obs, &legal, Budget::Nodes(1))
            } else {
                b.choose(&obs, &legal, Budget::Nodes(1))
            };
            assert!(legal.contains(&action));
            engine::apply(&mut st, action, &mut rng).unwrap();
        }
        assert!(st.is_over(), "self-play did not finish");
    }
}
