//! `duels-agent-alphabeta`: a search agent for 7 Wonders Duel.
//!
//! [`AlphaBetaAgent`] picks its move by searching the game tree: expectimax
//! over the game's chance nodes, alpha-beta at the decision nodes, iterative
//! deepening within whatever [`Budget`] it is handed, a transposition table
//! and a move-ordering heuristic to make the pruning bite, and a cheap static
//! evaluation at the horizon.
//!
//! # Why expectimax and not just minimax
//!
//! 7 Wonders Duel is stochastic but has **no player-private information**:
//! both players see exactly the same board, and the only unknowns are what
//! the shuffle will turn up. That makes it a two-player zero-sum *stochastic*
//! game — chance nodes, no information sets — so none of the machinery for
//! imperfect-information games (ISMCTS, CFR) is called for, but pretending
//! the game is deterministic is not either. Two things are genuinely random
//! mid-game and both are handled as chance nodes, averaging the successor
//! values weighted by the probabilities
//! [`duels_core::engine::chance_outcomes`] computes from public knowledge:
//!
//! 1. taking a card can uncover one or two face-down cards in the structure;
//! 2. The Great Library draws three of the set-aside progress tokens.
//!
//! # The determinization approximation
//!
//! An [`Agent`] only ever sees an [`Observation`], which replaces every
//! hidden value with the pool it could resolve to. The search needs a
//! concrete world, so `choose` draws one with
//! [`Observation::sample_state`] and treats it as ground truth for the
//! duration of that one search. This is single-observer determinized search,
//! the standard approximation for this class of game, and it is worth being
//! precise about what it does and does not cost here:
//!
//! - It does **not** affect what is revealed as the current age's structure is
//!   dismantled. Those reveals go through the chance API, whose distribution
//!   is derived from public knowledge and whose outcomes are *forced* onto the
//!   state, so the sampled identity of a face-down card is never consulted.
//! - It **does** fix the deal of the *later* ages: the engine deals the Age II
//!   and Age III structures from the decks the state already holds, not
//!   through the chance API, so every line the search follows past an age
//!   boundary sees the same Age II/III layout — the one that got sampled.
//!   Searching deep enough for that to matter would need a fresh
//!   determinization per root and averaging over several, which is left for
//!   later; at the depths this agent actually reaches it is a horizon away.
//!
//! # Budgets
//!
//! Both [`Budget`] variants are honoured. `Budget::Nodes(n)` counts decision
//! nodes and makes the agent fully deterministic (given its seed), which is
//! what the tests use. `Budget::TimeMs(ms)` reads the wall clock — the only
//! place in the crate that does, see `search::clock` — and so may return
//! different moves on different machines.
//!
//! # What it measures
//!
//! Against `RandomAgent` over 50 seeded games with alternating seats
//! (`tests/vs_random.rs::benchmark_against_random`):
//!
//! | budget | win rate | mean completed depth |
//! |---|---|---|
//! | `Nodes(2_000)` | 90% | 4.6 |
//! | `Nodes(20_000)` | 96% | 5.8 |
//! | `TimeMs(200)` | 98% | 7.9 |
//!
//! Depth is uneven by design: early in an age most moves uncover face-down
//! cards, so the effective branching factor is `moves x chance_cap` and the
//! search reaches four or five plies; late in an age, and in the endgame,
//! there is nothing left to uncover and it often solves the rest of the game
//! outright.
//!
//! Of the three optimisations (see `examples/search_stats.rs`), Star1 at the
//! chance nodes is by far the most valuable — it roughly halves the tree in
//! chance-heavy positions. The move ordering is worth ~15% there, and the
//! transposition table hits ~25% of probes in real play but saves only a few
//! percent of nodes: genuine transpositions are rare in this game over a
//! short horizon, and the table mostly earns its keep as a best-move hint
//! between deepening iterations.
//!
//! # Example
//!
//! ```
//! use duels_agents_api::{Agent, Budget};
//! use duels_core::engine;
//! use duels_agent_alphabeta::AlphaBetaAgent;
//!
//! let state = engine::new_game(7);
//! let obs = state.observation();
//! let legal = engine::legal_actions(&state);
//!
//! let mut agent = AlphaBetaAgent::new(7);
//! let action = agent.choose(&obs, &legal, Budget::Nodes(2_000));
//! assert!(legal.contains(&action));
//! ```

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::{Action, Observation};
use rand::{rngs::StdRng, SeedableRng};

pub mod eval;
pub mod order;
pub mod search;
pub mod tt;

use search::{SearchResult, Searcher};
use tt::Table;

/// Search parameters.
///
/// The defaults are what [`AlphaBetaAgent::new`] uses and what the reported
/// win rate was measured with; the switches exist mostly so the tests can
/// turn each optimisation off and check it did not change the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Hard ceiling on iterative deepening, in decision plies. The budget
    /// normally binds long before this does.
    pub max_depth: u8,
    /// Most outcomes a chance node averages over; `0` means all of them. See
    /// [`search::reduced_outcomes`] for why a cap is needed and what it
    /// costs.
    pub chance_cap: usize,
    /// Transposition table size, as a power of two.
    pub tt_bits: u32,
    /// Use the transposition table at all.
    pub use_tt: bool,
    /// Prune at chance nodes (Star1) as well as at decision nodes.
    pub star1: bool,
    /// Sort moves before searching them, using static card data.
    pub order_moves: bool,
    /// Sort moves by a one-ply lookahead instead (overrides `order_moves`):
    /// better ordering, but an apply and an evaluation per move.
    pub order_lookahead: bool,
    /// Seed for the determinization sampling.
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_depth: 24,
            chance_cap: 3,
            tt_bits: 20,
            use_tt: true,
            star1: true,
            order_moves: true,
            order_lookahead: false,
            seed: 0,
        }
    }
}

impl Config {
    /// A compact description of the parameters, for [`AgentSpec::params`].
    pub fn describe(&self) -> String {
        format!(
            "max_depth={},chance_cap={},tt_bits={},tt={},star1={},order={}",
            self.max_depth,
            self.chance_cap,
            self.tt_bits,
            self.use_tt,
            self.star1,
            if self.order_lookahead {
                "lookahead"
            } else if self.order_moves {
                "static"
            } else {
                "none"
            },
        )
    }
}

/// Cumulative counters over an agent's lifetime, for reporting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Calls to [`Agent::choose`].
    pub decisions: u64,
    /// Decision nodes searched, in total.
    pub nodes: u64,
    /// Sum of the depths reached, so `depth_sum / decisions` is the mean.
    pub depth_sum: u64,
    /// Deepest completed iteration over the agent's life.
    pub max_depth_reached: u8,
}

impl Stats {
    /// Mean completed search depth per decision.
    pub fn mean_depth(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.depth_sum as f64 / self.decisions as f64
        }
    }

    /// Mean nodes searched per decision.
    pub fn mean_nodes(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.nodes as f64 / self.decisions as f64
        }
    }
}

/// An expectimax / alpha-beta search agent.
#[derive(Debug)]
pub struct AlphaBetaAgent {
    cfg: Config,
    tt: Table,
    rng: StdRng,
    stats: Stats,
    last: Option<SearchResult>,
}

impl AlphaBetaAgent {
    /// An agent with the default [`Config`], seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self::with_config(Config {
            seed,
            ..Config::default()
        })
    }

    /// An agent with explicit parameters.
    pub fn with_config(cfg: Config) -> Self {
        let rng = StdRng::seed_from_u64(cfg.seed);
        let tt = Table::with_bits(cfg.tt_bits);
        Self {
            cfg,
            tt,
            rng,
            stats: Stats::default(),
            last: None,
        }
    }

    /// The parameters this agent is running with.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Lifetime search counters.
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Transposition-table `(probes, hits)` over the agent's lifetime.
    pub fn tt_stats(&self) -> (u64, u64) {
        self.tt.stats()
    }

    /// What the most recent [`Agent::choose`] found, if any.
    pub fn last_search(&self) -> Option<SearchResult> {
        self.last
    }
}

impl Agent for AlphaBetaAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "alphabeta".to_string(),
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
            // Nothing to search; do not spend the budget or a determinization
            // on a forced move.
            return legal[0];
        }

        // One determinized world, used as ground truth for this search only.
        let root = obs.sample_state(&mut self.rng);
        let me = obs.current_player;

        let result = {
            let mut searcher = Searcher::new(me, &self.cfg, &mut self.tt, budget);
            searcher.think(&root, legal)
        };

        self.stats.decisions += 1;
        self.stats.nodes += result.nodes;
        self.stats.depth_sum += u64::from(result.depth);
        self.stats.max_depth_reached = self.stats.max_depth_reached.max(result.depth);
        self.last = Some(result);

        // The sampled state agrees with the observation on every public fact,
        // so the search only ever sees the moves it was given; the fallback is
        // belt and braces.
        if legal.contains(&result.best) {
            result.best
        } else {
            legal[0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;

    #[test]
    fn spec_reports_the_expected_name_version_and_parameters() {
        let agent = AlphaBetaAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "alphabeta");
        assert_eq!(spec.version, "1.0.0");
        assert!(spec.params.contains("max_depth="), "{}", spec.params);
        assert!(spec.params.contains("chance_cap="), "{}", spec.params);
    }

    #[test]
    fn choosing_only_ever_returns_one_of_the_offered_actions() {
        let state = engine::new_game(99);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        let mut agent = AlphaBetaAgent::with_config(Config {
            tt_bits: 14,
            ..Config::default()
        });
        for _ in 0..5 {
            let a = agent.choose(&obs, &legal, Budget::Nodes(500));
            assert!(legal.contains(&a));
        }
        assert_eq!(agent.stats().decisions, 5);
        assert!(agent.stats().nodes > 0);
        assert!(agent.stats().mean_depth() >= 1.0);
    }

    #[test]
    fn a_forced_move_is_returned_without_searching() {
        let state = engine::new_game(3);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        let mut agent = AlphaBetaAgent::with_config(Config {
            tt_bits: 12,
            ..Config::default()
        });
        let only = &legal[..1];
        assert_eq!(agent.choose(&obs, only, Budget::Nodes(10_000)), legal[0]);
        assert_eq!(agent.stats().nodes, 0);
    }

    #[test]
    fn the_same_seed_and_node_budget_give_the_same_move() {
        let state = engine::new_game(21);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        let mut a = AlphaBetaAgent::with_config(Config {
            seed: 5,
            tt_bits: 14,
            ..Config::default()
        });
        let mut b = AlphaBetaAgent::with_config(Config {
            seed: 5,
            tt_bits: 14,
            ..Config::default()
        });
        for _ in 0..3 {
            assert_eq!(
                a.choose(&obs, &legal, Budget::Nodes(1_500)),
                b.choose(&obs, &legal, Budget::Nodes(1_500))
            );
        }
    }

    #[test]
    fn a_time_budget_is_respected() {
        let state = engine::new_game(8);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        let mut agent = AlphaBetaAgent::with_config(Config {
            tt_bits: 14,
            ..Config::default()
        });
        // Deliberately reads the clock, which is what is being tested; the
        // margin is generous so a loaded CI box does not fail the build.
        #[allow(clippy::disallowed_methods)]
        let start = std::time::Instant::now();
        let a = agent.choose(&obs, &legal, Budget::TimeMs(50));
        #[allow(clippy::disallowed_methods)]
        let elapsed = start.elapsed();
        assert!(legal.contains(&a));
        assert!(
            elapsed < std::time::Duration::from_millis(2_000),
            "took {elapsed:?} for a 50 ms budget"
        );
    }
}
