//! `duels-agent-alphabeta`: a search agent for 7 Wonders Duel.
//!
//! [`AlphaBetaAgent`] picks its move by searching the game tree: expectimax
//! over the game's chance nodes, alpha-beta at the decision nodes, iterative
//! deepening within whatever [`Budget`] it is handed, a transposition table
//! and a move-ordering heuristic to make the pruning bite, and — at the
//! horizon — a short Monte Carlo playout to the end of the game rather than a
//! static score.
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
//!   boundary — a leaf playout included — sees the same Age II/III layout,
//!   the one that got sampled. Averaging over several determinizations per
//!   root is the natural next step and is not done here.
//!
//! # The horizon: a playout, not a score
//!
//! This is the part of the agent that decides how strong it is, and the part
//! that was originally wrong.
//!
//! 7 Wonders Duel scores holistically at the end. Guild majorities are
//! recomputed from the final board, `coins / 3` rounds once, and the whole
//! point of a resource base is builds twenty plies later. A static evaluation
//! four or five plies from the root is therefore judging a position on
//! evidence that is mostly not in yet — and extra depth does not rescue it,
//! because the horizon merely moves from ply five to ply eight in a
//! seventy-ply game. The first version of this crate did exactly that, and
//! the symptom was visible in its games: it discarded most of its Age II and
//! III turns, because banking two coins is a certain `2/3` of a point in the
//! *current* score while a brown card is worth nothing until it pays for a
//! build the search cannot see.
//!
//! So the leaves now play the position out under a cheap policy and read the
//! real scoring rules (see [`playout`]). The static form survives as a small
//! blended term, because a random playout almost never stumbles into the two
//! instant-win races and [`eval`] does know about them.
//!
//! Two consequences, both measured rather than assumed:
//!
//! - **Depth is worth much less than sampling.** One extra ply costs about
//!   thirtyfold here (a dozen legal moves, times the chance node after most of
//!   them), and thirty times the playouts at the horizon is worth more than
//!   the ply. Worse, a max node taking the largest of several *noisy* leaf
//!   estimates is biased upward by roughly the spread of that noise and a min
//!   node symmetrically downward, so deepening over under-sampled leaves
//!   actively degrades the root ordering. Spare budget therefore goes into
//!   doubling the playouts per leaf *before* another ply — see
//!   [`search::Searcher::think`]. At `TimeMs(20)` the agent settles around a
//!   mean completed depth of 1.4, going deeper only in the endgame where
//!   playouts are short.
//! - **`Budget::Nodes` now counts playouts too.** A leaf that runs `n`
//!   playouts charges `n` to the budget, which keeps a node budget a usable
//!   proxy for work and puts it on the same footing as `mcts-uct`, whose
//!   `Budget::Nodes(n)` is exactly `n` playouts.
//!
//! # What it measures
//!
//! Paired seat-swapped matches, 200 games each (100 seeds, both seats), run
//! through `duels-arena` and its `ab_lab` example. `v1` is [`Config::v1`], the
//! static-evaluation agent this replaced. The `+/-` figures are one binomial
//! standard error.
//!
//! | opponent | budget | v1 | this version |
//! |---|---|---|---|
//! | `random` | `Nodes(2_000)` | 92% | **100%** |
//! | `greedy` | `Nodes(2_000)` | 82% | **100%** |
//! | `mcts-uct` | `Nodes(2_000)` | 2.5% | **19.5%** +/- 2.8 |
//! | `mcts-uct` | `TimeMs(20)` | 3.0% | **27.5%** +/- 3.2 |
//! | `v1` | `TimeMs(20)` | — | **94%** +/- 1.7 |
//!
//! The wall-clock row is the one that counts. Read the node rows with care:
//! `v1` at `Nodes(2_000)` spent 25 ms per game against `mcts-uct`'s 674 ms, so
//! the original round-robin's "matched" node budget was giving `mcts-uct`
//! twentyfold the actual compute. Charging playouts to the node counter (see
//! above) is what brings the two back within 15% of each other on the clock at
//! the same nominal budget.
//!
//! Strength still rises monotonically with the budget — `Nodes(8_000)` beats
//! `Nodes(1_000)` 75% over 100 games — which is the useful signal there: a
//! search with a sign error or a broken expectation at the chance nodes tends
//! to get *worse* as it looks further, not better.
//!
//! # The honest ceiling
//!
//! `mcts-uct` is still ahead at a matched wall clock — 27.5%, or about 170 Elo
//! — and eight times the time only brings this agent to 39% (+/- 3.4), so the
//! remainder is not a tuning pass away.
//!
//! The reason is structural. Both agents now spend essentially their whole
//! budget on the same primitive, a determinized playout to the end of the
//! game, and UCT allocates those playouts adaptively down the lines that
//! actually matter while expectimax must evaluate every leaf of a fixed-depth
//! tree whether the line is plausible or not. At a 20 ms budget this agent
//! reaches a mean completed depth of 1.4: it is, in effect, a well-sampled
//! one-ply search, because the measurements above say a second ply is not
//! worth thirty times the sampling. Recovering the rest of the gap means
//! importing UCT's adaptive allocation, at which point the agent stops being
//! an alpha-beta search and becomes a second `mcts-uct`.
//!
//! So the honest summary is that a *static-evaluation* search agent has a very
//! low ceiling in this game — 3% against `mcts-uct` — and that a search agent
//! with a *simulation-based* leaf has a respectable but still clearly bounded
//! one. Two things might raise it without abandoning the search framing,
//! neither attempted here: averaging over several root determinizations, and a
//! leaf estimator accurate enough per unit of time to make a second ply pay
//! for itself.
//!
//! Ablations that did **not** help, so that nobody pays for them twice:
//! raising [`Config::chance_cap`] above 3 (a `cap` of 6 or 8 loses at a
//! matched wall clock — the thinning of the chance-outcome list, flagged as
//! the original build's other suspect, is not what was wrong); reporting a
//! playout's win / loss outcome instead of its margin; a playout policy that
//! follows the move-ordering heuristic; and the one-ply-lookahead move
//! ordering. Each is documented where it lives.
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
pub mod playout;
pub mod search;
pub mod tt;

use search::{SearchResult, Searcher};
use tt::Table;

/// Search parameters.
///
/// The defaults are what [`AlphaBetaAgent::new`] uses and what the reported
/// win rates were measured with. Every field either exists so a test can turn
/// an optimisation off and check it did not change the answer, or records a
/// measurement — the doc comment on each says which, and says what the
/// alternative scored, so an ablation is not re-run from scratch.
///
/// [`Config::v1`] is the whole pre-rework configuration in one call, which is
/// what the before/after numbers are taken against.
#[derive(Debug, Clone, PartialEq)]
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
    /// better ordering, but an apply and an evaluation per move. Measured
    /// neutral once the leaves run playouts (49% over 100 games), where the
    /// ordering's cost is negligible but so is the extra pruning it buys, the
    /// tree being only a ply or two deep.
    pub order_lookahead: bool,
    /// Coefficients for the static leaf evaluation.
    pub weights: eval::Weights,
    /// Playouts averaged at each horizon leaf; `0` means the leaf value is
    /// the static evaluation alone. See [`playout`] for why a search agent in
    /// this game wants any at all.
    pub rollouts: u32,
    /// How much of the leaf value comes from the playout average rather than
    /// the static evaluation, in `0.0..=1.0`. Ignored when `rollouts` is `0`.
    ///
    /// Not `1.0`: the small static remainder is what keeps the search aware of
    /// the two instant-win races, which a playout under a random policy
    /// essentially never walks into. Empirically flat between `0.75` and
    /// `1.0`, with `0.9` marginally ahead.
    pub rollout_blend: f64,
    /// The playout policy.
    pub rollout_policy: playout::PolicyWeights,
    /// What a playout reports back: the victory-point margin, or the win /
    /// draw / loss outcome. See [`playout::Metric`].
    pub rollout_metric: playout::Metric,
    /// Seed every leaf's playouts from a per-iteration constant instead of
    /// from the position's own hash: common random numbers, so that sibling
    /// leaves are compared under the same simulated luck.
    ///
    /// This is a variance-reduction device aimed squarely at the max-of-noise
    /// bias described on [`search::Searcher::think`], and it does what it
    /// says: with a two-ply ceiling it recovers the ground a second ply
    /// otherwise loses (43% -> 50% against the one-ply configuration over 100
    /// games each). It is off by default because at the one-ply depth the
    /// agent actually settles on there are no siblings to correlate, and it
    /// then measures as exactly neutral (51%/49%). Turn it on together with a
    /// depth ceiling above one.
    pub rollout_common_seed: bool,
    /// Ceiling on the sampling ramp: spare budget doubles the playouts per
    /// leaf, up to this many, before it buys another ply. See
    /// [`search::Searcher::think`].
    ///
    /// The cap is what stops the agent from being a one-ply agent forever: at
    /// a large enough budget the sampling saturates and the search deepens
    /// instead. `48` and `128` measure the same at `TimeMs(20)` and `512`
    /// measures worse (43%), so this is set at the top of the flat region,
    /// where it leaves the most room to deepen.
    pub rollout_cap: u32,
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
            weights: eval::Weights::DEFAULT,
            rollouts: 8,
            rollout_blend: 0.9,
            rollout_policy: playout::PolicyWeights::BIASED,
            rollout_metric: playout::Metric::MARGIN,
            rollout_common_seed: false,
            rollout_cap: 128,
            seed: 0,
        }
    }
}

impl Config {
    /// The parameters the crate shipped with before the leaf evaluation was
    /// reworked: a static evaluation at the horizon and nothing else.
    ///
    /// Kept so the tuning harness (`duels-arena`'s `ab_lab` example) and the
    /// tests can measure against the version whose win rates the crate docs
    /// used to quote, rather than against a remembered number.
    pub fn v1() -> Self {
        Self {
            weights: eval::Weights::V1,
            rollouts: 0,
            ..Self::default()
        }
    }

    /// A compact description of the parameters, for [`AgentSpec::params`].
    pub fn describe(&self) -> String {
        format!(
            "max_depth={},chance_cap={},tt_bits={},tt={},star1={},order={},leaf={}",
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
            self.describe_leaf(),
        )
    }

    /// How the horizon is evaluated, for [`Config::describe`].
    fn describe_leaf(&self) -> String {
        if self.rollouts == 0 {
            return "static".to_string();
        }
        let metric = match self.rollout_metric {
            playout::Metric::Margin { clamp } => format!("margin({clamp:.0})"),
            playout::Metric::Outcome { scale } => format!("outcome({scale:.0})"),
        };
        format!(
            "playout({}..{},blend={:.2},{metric})",
            self.rollouts, self.rollout_cap, self.rollout_blend
        )
    }
}

/// Cumulative counters over an agent's lifetime, for reporting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Calls to [`Agent::choose`].
    pub decisions: u64,
    /// Search work, in total, in units of "one decision node or one leaf
    /// playout" — see [`search::Searcher::think`] for why playouts are
    /// charged to the same counter.
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
            version: "2.0.0".to_string(),
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
        assert_eq!(spec.version, "2.0.0");
        assert!(spec.params.contains("max_depth="), "{}", spec.params);
        assert!(spec.params.contains("chance_cap="), "{}", spec.params);
        assert!(spec.params.contains("leaf=playout("), "{}", spec.params);
        assert!(
            Config::v1().describe().contains("leaf=static"),
            "{}",
            Config::v1().describe()
        );
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
