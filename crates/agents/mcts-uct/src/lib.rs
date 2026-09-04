//! `duels-agent-mcts-uct`: Monte Carlo Tree Search with UCT selection and
//! **explicit chance nodes**, intended as the project's strongest non-learned
//! baseline and the yardstick for everything that comes after it.
//!
//! # Why chance nodes
//!
//! 7 Wonders Duel is a two-player zero-sum *stochastic* game: the cards
//! behind the face-down slots of the current age are unknown when a move is
//! chosen, and taking a card can uncover them. There is no player-private
//! information — both players always see the same public state — so the game
//! is much simpler than poker, but it is not Go: a plain alternating-move
//! tree would silently pretend the reveals were part of the mover's choice.
//!
//! This agent therefore builds a tree with three kinds of node:
//!
//! - **decision** nodes, one player to move, children = the legal actions,
//!   selected by UCB1;
//! - **chance** nodes, inserted between an action and the position it leads
//!   to whenever the engine says the action resolves randomness, children =
//!   possible reveals, selected **by their real probability**, never by UCB1;
//! - **terminal** nodes, where the [`duels_core::GameResult`] is settled.
//!
//! # How chance is handled, precisely
//!
//! Worth being explicit about, because it bounds how strong the agent can
//! get:
//!
//! 1. **The root is determinized.** `choose` only ever sees an
//!    [`Observation`], so it calls [`Observation::sample_state`] once per
//!    call to get one concrete world consistent with public knowledge.
//! 2. **Reveals inside the tree are *not* taken from that world.** Every
//!    chance node re-draws its outcome from the distribution the engine
//!    computes from public information alone
//!    (`engine::chance_outcomes`/`hidden_info`), and applies it with
//!    `engine::apply_with_outcome`, which rewrites the hidden layout to stay
//!    publicly consistent. So the tree integrates over reveals rather than
//!    committing to the root's guess, and the agent can never exploit
//!    knowledge of a card it should not know.
//! 3. **The draw is exact but not enumerated.** A two-slot reveal has
//!    hundreds of outcomes; the `chance` module draws from that distribution
//!    in O(1) and reports the drawn outcome's exact probability, which is
//!    verified statistically against `engine::chance_outcomes`.
//! 4. **Progressive widening is an approximation.** A chance node that
//!    created a fresh child on every visit would be a perfectly unbiased
//!    estimator of the expectation but would never let the tree grow past it.
//!    By default the number of distinct outcome children grows as
//!    `sqrt(visits)` and further visits re-select an existing child in
//!    proportion to its probability. That reweighting is the one place the
//!    search is not a faithful expectation; set
//!    [`Config::chance_widen_alpha`] to `1.0` with a large
//!    [`Config::chance_widen_c`] to recover the unbiased estimator.
//! 5. **Two sources of randomness are only root-determinized:** the
//!    composition and order of the *next* age's deck, and the four wonders
//!    not yet offered during the draft. Neither is exposed through the
//!    per-action chance API, so they stay fixed for the duration of one
//!    search. Re-determinizing per simulation (root ensembling) is the
//!    natural next step.
//!
//! The value convention (every node accumulates the result from
//! [`duels_core::Player::One`]'s perspective; the zero-sum flip happens once,
//! at selection) and the widening rule are documented in the `tree` module.
//!
//! # Example
//!
//! ```
//! use duels_agent_mcts_uct::MctsAgent;
//! use duels_agents_api::{Agent, Budget};
//! use duels_core::engine;
//!
//! let mut agent = MctsAgent::new(7);
//! let state = engine::new_game(7);
//! let legal = engine::legal_actions(&state);
//! let action = agent.choose(&state.observation(), &legal, Budget::Nodes(64));
//! assert!(legal.contains(&action));
//! ```

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

mod chance;
mod rollout;
mod tree;

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::{engine, Action, Observation};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub use rollout::RolloutWeights;
pub use tree::Config;

/// Monte Carlo Tree Search with UCT selection and explicit chance nodes.
#[derive(Debug)]
pub struct MctsAgent {
    cfg: Config,
    rng: StdRng,
    /// Simulations run over the agent's whole lifetime, for throughput
    /// reporting.
    total_simulations: u64,
    /// Nodes allocated during the most recent search.
    last_tree_size: usize,
}

impl MctsAgent {
    /// A new agent with the default configuration, seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, Config::default())
    }

    /// A new agent with an explicit configuration.
    pub fn with_config(seed: u64, cfg: Config) -> Self {
        Self {
            cfg,
            rng: StdRng::seed_from_u64(seed),
            total_simulations: 0,
            last_tree_size: 0,
        }
    }

    /// The configuration in force.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Total simulations this agent has run since it was created.
    pub fn total_simulations(&self) -> u64 {
        self.total_simulations
    }

    /// Nodes allocated by the most recent `choose` call.
    pub fn last_tree_size(&self) -> usize {
        self.last_tree_size
    }
}

impl Agent for MctsAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "mcts-uct".to_string(),
            version: "1.0.0".to_string(),
            params: self.cfg.describe(),
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action], budget: Budget) -> Action {
        assert!(
            !legal.is_empty(),
            "choose must not be called with no legal actions"
        );
        if legal.len() == 1 {
            return legal[0];
        }

        // One determinized world consistent with the observation. Hidden
        // reveals *inside* the search are re-drawn from public knowledge at
        // each chance node, so this world only fixes what the chance API does
        // not cover (future age decks, the undrafted wonder pool).
        let root = obs.sample_state(&mut self.rng);

        // The offered actions and the determinized state must agree, since
        // legality is a function of public information only; filter
        // defensively so an unexpected mismatch can never return an action
        // the arena did not offer.
        let mut actions: Vec<Action> = legal
            .iter()
            .copied()
            .filter(|&a| engine::is_legal(&root, a))
            .collect();
        debug_assert_eq!(
            actions.len(),
            legal.len(),
            "a determinized root disagreed with the offered legal actions"
        );
        if actions.is_empty() {
            actions = legal.to_vec();
        }

        let mut tree = tree::Tree::new(root, actions, self.cfg, &mut self.rng);
        run(&mut tree, budget, &mut self.rng);

        self.total_simulations += tree.simulations;
        self.last_tree_size = tree.nodes.len();

        let chosen = tree.best_action().unwrap_or(legal[0]);
        if legal.contains(&chosen) {
            chosen
        } else {
            // Unreachable given the filter above; never hand back an action
            // the caller did not offer.
            legal[0]
        }
    }
}

/// Run simulations until `budget` is spent.
fn run(tree: &mut tree::Tree, budget: Budget, rng: &mut StdRng) {
    match budget {
        Budget::Nodes(n) => {
            // A budget of zero still needs one simulation, otherwise there
            // are no visited children to choose between.
            for _ in 0..n.max(1) {
                tree.simulate(rng);
            }
        }
        Budget::TimeMs(ms) => {
            // The workspace bans wall-clock reads so that the engine and its
            // agents stay reproducible from a seed; `Budget::TimeMs` is the
            // one place an agent is *asked* to read the clock, and the read
            // is confined to this function. `Budget::Nodes` remains fully
            // deterministic.
            #[allow(clippy::disallowed_methods)]
            let start = std::time::Instant::now();
            let deadline = std::time::Duration::from_millis(ms);
            let interval = tree.cfg.time_check_interval.max(1);
            loop {
                for _ in 0..interval {
                    tree.simulate(rng);
                }
                if start.elapsed() >= deadline {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_agent_random::RandomAgent;
    use duels_core::{GameResult, Player};

    /// A small budget: enough that the tree is exercised (root expansion,
    /// chance nodes, UCB1 re-selection) while keeping `cargo test` quick.
    const CI_BUDGET: Budget = Budget::Nodes(48);

    fn play(seed: u64, mcts_seat: Player, budget: Budget) -> (GameResult, u64) {
        let mut mcts = MctsAgent::new(seed ^ 0x0BAD_1DEA_0BAD_1DEA);
        let mut opponent = RandomAgent::new(seed ^ 0x5EED_5EED);
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);

        let mut plies = 0u32;
        loop {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = if state.current_player() == mcts_seat {
                mcts.choose(&obs, &legal, budget)
            } else {
                opponent.choose(&obs, &legal, budget)
            };
            assert!(
                legal.contains(&action),
                "agent returned an illegal action {action:?}"
            );
            engine::apply(&mut state, action, &mut rng).expect("a legal action");
            plies += 1;
            assert!(plies < 5_000, "game did not terminate after {plies} plies");
        }
        (
            state.result().expect("a finished game has a result"),
            mcts.total_simulations(),
        )
    }

    #[test]
    fn spec_reports_the_expected_name_version_and_params() {
        let agent = MctsAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "mcts-uct");
        assert_eq!(spec.version, "1.0.0");
        assert!(spec.params.contains("c=1.000"), "{}", spec.params);
        assert!(spec.params.contains("chance="), "{}", spec.params);
        assert!(spec.params.contains("rollout="), "{}", spec.params);
    }

    #[test]
    fn a_single_legal_action_is_returned_without_searching() {
        let mut agent = MctsAgent::new(3);
        let state = engine::new_game(3);
        let only = [engine::legal_actions(&state)[0]];
        let chosen = agent.choose(&state.observation(), &only, Budget::Nodes(10_000));
        assert_eq!(chosen, only[0]);
        assert_eq!(agent.total_simulations(), 0, "no search was needed");
    }

    #[test]
    fn every_returned_action_is_one_of_the_offered_ones() {
        let mut agent = MctsAgent::new(11);
        let state = engine::new_game(11);
        let legal = engine::legal_actions(&state);
        for _ in 0..5 {
            let a = agent.choose(&state.observation(), &legal, Budget::Nodes(20));
            assert!(legal.contains(&a));
        }
    }

    #[test]
    fn a_node_budget_runs_exactly_that_many_simulations() {
        let mut agent = MctsAgent::new(5);
        let state = engine::new_game(5);
        let legal = engine::legal_actions(&state);
        agent.choose(&state.observation(), &legal, Budget::Nodes(37));
        assert_eq!(agent.total_simulations(), 37);
        agent.choose(&state.observation(), &legal, Budget::Nodes(3));
        assert_eq!(agent.total_simulations(), 40);
    }

    #[test]
    fn a_time_budget_returns_promptly_and_does_some_work() {
        let mut agent = MctsAgent::new(9);
        let state = engine::new_game(9);
        let legal = engine::legal_actions(&state);
        let a = agent.choose(&state.observation(), &legal, Budget::TimeMs(20));
        assert!(legal.contains(&a));
        assert!(agent.total_simulations() > 0);
    }

    #[test]
    fn a_node_budget_is_reproducible_from_the_seed() {
        let state = engine::new_game(21);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        let pick = |seed: u64| {
            let mut agent = MctsAgent::new(seed);
            agent.choose(&obs, &legal, Budget::Nodes(200))
        };
        assert_eq!(pick(4), pick(4));
    }

    /// The headline robustness test: full seeded games against the random
    /// agent, from both seats, always reaching a `GameResult` without a
    /// panic, a hang, or an illegal action.
    #[test]
    fn plays_twenty_full_seeded_games_against_random_without_incident() {
        let mut sims = 0u64;
        for seed in 0..20u64 {
            let seat = if seed % 2 == 0 {
                Player::One
            } else {
                Player::Two
            };
            let (result, s) = play(seed, seat, CI_BUDGET);
            sims += s;
            println!("seed {seed} (mcts as {seat}): {result:?}");
        }
        assert!(sims > 0, "the agent never searched");
    }

    /// Even at a CI-sized budget the search should already be clearly better
    /// than uniform-random play. This is a loose smoke test, not the real
    /// strength measurement (see `examples/vs_random.rs`); it exists so that
    /// a sign error in backpropagation or a perspective flip cannot land
    /// silently.
    #[test]
    fn beats_random_at_a_small_budget() {
        let mut wins = 0u32;
        let games = 12u64;
        for seed in 0..games {
            let seat = if seed % 2 == 0 {
                Player::One
            } else {
                Player::Two
            };
            let (result, _) = play(100 + seed, seat, CI_BUDGET);
            if result.winner() == Some(seat) {
                wins += 1;
            }
        }
        assert!(
            wins * 2 > games as u32,
            "won only {wins}/{games} against random at {CI_BUDGET:?}; \
             suspect backpropagation sign, UCB1 perspective, or chance handling"
        );
    }
}
