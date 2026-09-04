//! Agent lookup by name.
//!
//! A tiny `match` on a string name, not a plugin system: the brief for this
//! crate explicitly asks for the cheapest thing that lets sibling agent
//! crates (`greedy`, `alphabeta`, `mcts-uct`, ...) be wired in with one new
//! arm each, once they exist. Mirrors `duels-server`'s `room::make_agent`.

use duels_agents_api::Agent;

/// Every agent name this build of `duels-arena` knows how to construct, for
/// `--help` text and error messages.
pub const KNOWN_AGENTS: &[&str] = &["random", "greedy", "alphabeta", "mcts-uct"];

/// Construct the named `Agent`, seeded from `seed`.
///
/// Add one match arm per new agent crate as it lands; nothing else in this
/// crate needs to change.
pub fn make_agent(name: &str, seed: u64) -> Result<Box<dyn Agent + Send>, String> {
    match name {
        "random" => Ok(Box::new(duels_agent_random::RandomAgent::new(seed))),
        "greedy" => Ok(Box::new(duels_agent_greedy::GreedyAgent::new(seed))),
        "alphabeta" => Ok(Box::new(duels_agent_alphabeta::AlphaBetaAgent::new(seed))),
        "mcts-uct" => Ok(Box::new(duels_agent_mcts_uct::MctsAgent::new(seed))),
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
