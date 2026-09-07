//! Agent lookup by name.
//!
//! A tiny `match` on a string name, not a plugin system: the brief for this
//! crate explicitly asks for the cheapest thing that lets sibling agent
//! crates (`greedy`, `alphabeta`, `mcts-uct`, ...) be wired in with one new
//! arm each, once they exist. Mirrors `duels-server`'s `room::make_agent`.

use duels_agents_api::Agent;

/// Every agent name this build of `duels-arena` knows how to construct, for
/// `--help` text and error messages.
pub const KNOWN_AGENTS: &[&str] = &[
    "random",
    "greedy",
    "greedy-ev",
    "strategist",
    "phased",
    "alphabeta",
    "mcts-uct",
    "mcts-eval",
];

/// Construct the named `Agent`, seeded from `seed`.
///
/// Add one match arm per new agent crate as it lands; nothing else in this
/// crate needs to change.
pub fn make_agent(name: &str, seed: u64) -> Result<Box<dyn Agent + Send>, String> {
    match name {
        "random" => Ok(Box::new(duels_agent_random::RandomAgent::new(seed))),
        "greedy" => Ok(Box::new(duels_agent_greedy::GreedyAgent::new(seed))),
        "greedy-ev" => Ok(Box::new(duels_agent_greedy_ev::GreedyEvAgent::new(seed))),
        "strategist" => Ok(Box::new(duels_agent_strategist::StrategistAgent::new(seed))),
        "phased" => Ok(Box::new(duels_agent_phased::PhasedAgent::new(seed))),
        "alphabeta" => Ok(Box::new(duels_agent_alphabeta::AlphaBetaAgent::new(seed))),
        "mcts-uct" => Ok(Box::new(duels_agent_mcts_uct::MctsAgent::new(seed))),
        "mcts-eval" => Ok(Box::new(duels_agent_mcts_eval::MctsEvalAgent::new(seed))),
        other => Err(format!(
            "unknown agent \"{other}\" (known agents: {})",
            KNOWN_AGENTS.join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_is_registered() {
        let agent = make_agent("random", 1).expect("random should be a known agent");
        assert_eq!(agent.spec().name, "random");
    }

    #[test]
    fn greedy_ev_is_registered() {
        let agent = make_agent("greedy-ev", 1).expect("greedy-ev should be a known agent");
        assert_eq!(agent.spec().name, "greedy-ev");
    }

    #[test]
    fn strategist_is_registered() {
        let agent = make_agent("strategist", 1).expect("strategist should be a known agent");
        assert_eq!(agent.spec().name, "strategist");
    }

    #[test]
    fn phased_is_registered() {
        let agent = make_agent("phased", 1).expect("phased should be a known agent");
        assert_eq!(agent.spec().name, "phased");
    }

    #[test]
    fn mcts_eval_is_registered() {
        let agent = make_agent("mcts-eval", 1).expect("mcts-eval should be a known agent");
        assert_eq!(agent.spec().name, "mcts-eval");
        // The bare name has to build the *measured* configuration, since that
        // is what the leaderboard and `duels-server` construct.
        assert!(agent.spec().params.contains("leaf=blend(0.500)"));
        assert!(agent.spec().params.contains("c=0.500"));
    }

    #[test]
    fn unknown_name_is_rejected_with_a_helpful_message() {
        // `Box<dyn Agent>` doesn't implement `Debug`, so `unwrap_err` (which
        // requires `T: Debug` for its panic message) doesn't type-check here;
        // match it out by hand instead.
        let err = match make_agent("nonexistent", 1) {
            Ok(_) => panic!("expected an error for an unknown agent name"),
            Err(e) => e,
        };
        assert!(err.contains("nonexistent"));
        assert!(err.contains("random"));
    }
}
