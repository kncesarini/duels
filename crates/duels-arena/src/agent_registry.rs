//! Agent lookup by name.
//!
//! A tiny `match` on a string name, not a plugin system: the brief for this
//! crate explicitly asks for the cheapest thing that lets sibling agent
//! crates (`phased`, `alphabeta`, `mcts-uct`, ...) be wired in with one new
//! arm each, once they exist. Mirrors `duels-server`'s `room::make_agent`.
//!
//! Registration here is what makes an agent **constructible**: it is what
//! `agent_spec` bare names resolve through. It is *not* by itself what puts an
//! agent on the leaderboard —
//! `leaderboard::tests::the_ladder_is_exactly_the_registered_agents` pins
//! [`KNOWN_AGENTS`] and `leaderboard::LADDER` to each other **up to
//! `leaderboard::REGISTERED_OFF_LADDER`**, the short list of agents that are
//! deliberately constructible without being rated. Read that constant's docs
//! before adding a name to it.
//!
//! `random`, `greedy`, `greedy-ev` and `strategist` were retired from the
//! roster and so are absent here; `duels-agent-random`'s crate survives as a
//! *dev-dependency* test fixture and deliberately cannot be named through
//! this function. See `docs/milestones.md`.

use duels_agents_api::Agent;

/// Every agent name this build of `duels-arena` knows how to construct, for
/// `--help` text and error messages.
///
/// Not every name here is on `leaderboard::LADDER`; see
/// `leaderboard::REGISTERED_OFF_LADDER`.
pub const KNOWN_AGENTS: &[&str] = &["phased", "alphabeta", "mcts-uct", "mcts-eval", "mcts-value"];

/// Construct the named `Agent`, seeded from `seed`.
///
/// Add one match arm per new agent crate as it lands; nothing else in this
/// crate needs to change.
pub fn make_agent(name: &str, seed: u64) -> Result<Box<dyn Agent + Send>, String> {
    match name {
        "phased" => Ok(Box::new(duels_agent_phased::PhasedAgent::new(seed))),
        "alphabeta" => Ok(Box::new(duels_agent_alphabeta::AlphaBetaAgent::new(seed))),
        "mcts-uct" => Ok(Box::new(duels_agent_mcts_uct::MctsAgent::new(seed))),
        "mcts-eval" => Ok(Box::new(duels_agent_mcts_eval::MctsEvalAgent::new(seed))),
        "mcts-value" => Ok(Box::new(duels_agent_mcts_value::MctsValueAgent::new(seed))),
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
    fn the_retired_agents_are_retired() {
        // `strategist`: its research question (whether `duels-strategy`'s
        // prior helps `greedy-ev`) was answered statistically
        // indistinguishable. `random`, `greedy` and `greedy-ev`: retired for
        // measured strength far below the rest of the roster, the same
        // reason. `greedy`, `greedy-ev` and `strategist` are gone from the
        // workspace entirely; `duels-agent-random` survives as a
        // *dev-dependency* test fixture, which is exactly why this asserts
        // that the name is still rejected here. See `docs/milestones.md`.
        for retired in ["random", "greedy", "greedy-ev", "strategist"] {
            assert!(
                make_agent(retired, 1).is_err(),
                "{retired} should not be constructible"
            );
            assert!(!KNOWN_AGENTS.contains(&retired), "{retired} still listed");
        }
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

    /// `mcts-value` is constructible from its bare name, and the bare name
    /// builds the *measured* configuration — `leaf=learned` at the swept
    /// `c = 0.15`, with the weights identity recorded so a results file says
    /// which network produced it.
    ///
    /// Being here and **not** on `leaderboard::LADDER` is the deliberate
    /// state: the agent is registered so its measurement is reproducible in
    /// one binary and so it is playable, and unrated because its margin over
    /// `mcts-eval` does not survive being measured through a third party. See
    /// its crate docs and `leaderboard::REGISTERED_OFF_LADDER`.
    #[test]
    fn mcts_value_is_registered_but_deliberately_unrated() {
        let agent = make_agent("mcts-value", 1).expect("mcts-value should be a known agent");
        assert_eq!(agent.spec().name, "mcts-value");
        assert!(agent.spec().params.contains("leaf=learned"));
        assert!(agent.spec().params.contains("c=0.150"));
        // The provenance record: which weights, and which summation order.
        assert!(agent.spec().params.contains("value="));
        assert!(!crate::leaderboard::LADDER
            .iter()
            .any(|e| e.agent == "mcts-value"));
        assert!(crate::leaderboard::REGISTERED_OFF_LADDER.contains(&"mcts-value"));
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
        assert!(err.contains("phased"));
    }
}
