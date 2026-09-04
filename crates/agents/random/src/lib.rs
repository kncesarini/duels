//! `duels-agent-random`: the first concrete [`Agent`] implementation.
//!
//! [`RandomAgent`] does the simplest thing that conforms to the contract in
//! `duels-agents-api`: it uniformly picks one of the `legal` actions it is
//! handed, using its own seeded [`StdRng`]. It never reads the [`Observation`]
//! it is given beyond what `choose`'s signature requires, and it never
//! touches wall-clock time or ambient randomness (see `clippy.toml`, denied
//! crate-wide below).

#![deny(clippy::disallowed_methods)]

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::{Action, Observation};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Picks uniformly at random among the legal actions offered at each
/// decision point.
#[derive(Debug)]
pub struct RandomAgent {
    rng: StdRng,
}

impl RandomAgent {
    /// A new agent seeded from `seed`, so its play is reproducible.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// A new agent driven by an existing RNG, so a caller can draw many
    /// independent agents from one stream.
    pub fn from_rng(rng: StdRng) -> Self {
        Self { rng }
    }
}

impl Agent for RandomAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "random".to_string(),
            version: "0.1.0".to_string(),
            params: String::new(),
        }
    }

    fn choose(&mut self, _obs: &Observation, legal: &[Action], _budget: Budget) -> Action {
        assert!(
            !legal.is_empty(),
            "choose must not be called with no legal actions"
        );
        let i = self.rng.gen_range(0..legal.len());
        legal[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;

    /// The whole point of this crate: drive a full game to completion, both
    /// players controlled by the random agent, without ever panicking and
    /// without the agent ever seeing more than an `Observation`.
    fn play_full_game(seed: u64) -> duels_core::GameResult {
        let mut agent_one = RandomAgent::new(seed);
        let mut agent_two = RandomAgent::new(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x1234_5678);

        let mut guard = 0u32;
        loop {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = match state.current_player() {
                duels_core::Player::One => agent_one.choose(&obs, &legal, Budget::Nodes(1)),
                duels_core::Player::Two => agent_two.choose(&obs, &legal, Budget::Nodes(1)),
            };
            assert!(legal.contains(&action), "agent returned an illegal action");
            engine::apply(&mut state, action, &mut rng).expect("agent returned a legal action");

            guard += 1;
            assert!(
                guard < 10_000,
                "game did not terminate after {guard} decisions"
            );
        }
        state.result().expect("a finished game has a result")
    }

    #[test]
    fn random_vs_random_plays_a_full_game_to_completion_across_seeds() {
        for seed in 0..25u64 {
            let result = play_full_game(seed);
            // Just proving it terminates with *a* result is the point; print
            // it so a failure is easy to correlate with a seed.
            println!("seed {seed}: {result:?}");
        }
    }

    #[test]
    fn spec_reports_the_expected_name_and_version() {
        let agent = RandomAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "random");
        assert_eq!(spec.version, "0.1.0");
    }

    #[test]
    fn choosing_only_ever_returns_one_of_the_offered_actions() {
        let mut agent = RandomAgent::new(99);
        let state = engine::new_game(99);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        for _ in 0..50 {
            let a = agent.choose(&obs, &legal, Budget::Nodes(1));
            assert!(legal.contains(&a));
        }
    }
}
