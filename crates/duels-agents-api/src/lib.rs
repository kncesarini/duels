//! `duels-agents-api`: the contract every AI/bot player implements.
//!
//! This crate defines the [`Agent`] trait and its supporting types
//! ([`AgentSpec`], [`Budget`]). It intentionally depends only on
//! `duels-core`'s public types ([`duels_core::Observation`],
//! [`duels_core::Action`]) and never on `duels_core::GameState` — an agent
//! must be structurally incapable of seeing hidden information (deck order,
//! face-down cards), which in 7 Wonders Duel is the *only* kind of
//! information withheld from either player (there is no player-private
//! hidden information in this game).
//!
//! See `docs/agent-contract.md` for the versioned, human-readable contract
//! (`CONTRACT_VERSION`) that future breaking changes to these types must
//! bump.
//!
//! M1 scope: the trait and types only. Concrete agents (random, greedy,
//! minimax/MCTS, RL-trained via the PyO3 bindings) land in later
//! milestones, as does the tournament runner in `duels-arena` that drives
//! `Agent` instances against each other.
//!
//! An agent that wants to search rather than react can turn the
//! `Observation` it is handed into a concrete, playable world with
//! `Observation::sample_state(&mut rng)`, which samples the hidden
//! information uniformly from the pools the observation exposes. That is the
//! only bridge from the public view back to a full state, and it never
//! reveals what the *actual* game is hiding.

use duels_core::{Action, Observation};
use serde::{Deserialize, Serialize};

/// Identifying metadata for one `Agent` implementation/configuration.
///
/// Used for logging, tournament leaderboards (see `duels-arena`), and
/// reproducibility (pairing a recorded game with the exact agent version and
/// parameters that produced it).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSpec {
    /// Human-readable agent name, e.g. `"random"`, `"greedy-v1"`, `"mcts"`.
    pub name: String,
    /// Semver-ish version string for this agent's implementation.
    pub version: String,
    /// Free-form, agent-defined parameter description (e.g. serialized
    /// hyperparameters or a config summary) for reproducibility.
    pub params: String,
}

/// The computational budget an `Agent` is given to choose its next move.
///
/// The arena/server decides which variant to grant; an `Agent` should
/// respect whichever one it receives and return a legal action before the
/// budget is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Budget {
    /// Search/evaluate at most this many nodes (engine-agnostic unit of
    /// work, defined by the agent itself).
    Nodes(u64),
    /// Return a move within this many milliseconds (wall-clock).
    TimeMs(u64),
}

/// The contract every AI/bot player implements.
///
/// `choose` is only ever given the public [`Observation`], never a
/// `GameState`, and only ever needs to pick among the engine-provided
/// `legal` actions for the current turn — it does not validate legality
/// itself.
pub trait Agent {
    /// Static identifying metadata for this agent (name/version/params).
    fn spec(&self) -> AgentSpec;

    /// Choose one of the `legal` actions given the current public
    /// `Observation`, within `budget`.
    ///
    /// Implementations must return one of the elements of `legal` (by
    /// value/clone) — the caller is not required to re-validate the
    /// returned action, though a conforming engine-driven arena may do so
    /// defensively.
    fn choose(&mut self, obs: &Observation, legal: &[Action], budget: Budget) -> Action;
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;

    /// A trivial agent used only to prove the trait is object-safe /
    /// implementable and that the supporting types behave as expected.
    struct FirstLegalAgent;

    impl Agent for FirstLegalAgent {
        fn spec(&self) -> AgentSpec {
            AgentSpec {
                name: "first-legal".to_string(),
                version: "0.0.1".to_string(),
                params: String::new(),
            }
        }

        fn choose(&mut self, _obs: &Observation, legal: &[Action], _budget: Budget) -> Action {
            legal.first().copied().expect("no legal actions available")
        }
    }

    #[test]
    fn agent_trait_is_implementable_and_object_safe() {
        let mut agent: Box<dyn Agent> = Box::new(FirstLegalAgent);
        let state = engine::new_game(1);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);

        let chosen = agent.choose(&obs, &legal, Budget::Nodes(100));
        assert_eq!(chosen, legal[0]);
        assert_eq!(agent.spec().name, "first-legal");
    }

    /// The whole point of the contract: an agent can drive a game to
    /// completion while only ever seeing `Observation`s.
    #[test]
    fn an_agent_can_play_a_whole_game_from_observations_alone() {
        use rand::{rngs::StdRng, SeedableRng};

        let mut agent = FirstLegalAgent;
        let mut state = engine::new_game(7);
        let mut rng = StdRng::seed_from_u64(7);
        loop {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = agent.choose(&obs, &legal, Budget::Nodes(1));
            assert!(legal.contains(&action));
            engine::apply(&mut state, action, &mut rng).expect("agent returned a legal action");
        }
        assert!(state.result().is_some());
    }

    #[test]
    fn budget_variants_are_debug_clone_eq() {
        let a = Budget::Nodes(10);
        let b = Budget::TimeMs(50);
        assert_ne!(format!("{a:?}"), format!("{b:?}"));
        assert_eq!(a, a);
    }

    #[test]
    fn agent_spec_round_trips_through_json() {
        let spec = AgentSpec {
            name: "greedy".into(),
            version: "1.0.0".into(),
            params: "{}".into(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }
}
