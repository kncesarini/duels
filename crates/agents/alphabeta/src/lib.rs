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
//!   the one that got sampled.
//!
//! [`Config::root_determinizations`] averages over several determinizations
//! per root instead of trusting one. It is implemented, measured, and **off
//! by default because it loses**; the numbers and the reason are below.
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
//! # Root ensembling, and why it is off
//!
//! [`Config::root_determinizations`] = `N` searches `N` independently sampled
//! worlds, each with `1/N` of the budget, and plays the move with the best
//! value *averaged over the searches* (see [`search::best_of`]). This is the
//! textbook fix for the determinization approximation above — Perfect
//! Information Monte Carlo ensembling — and on this agent it is a clear,
//! monotone **loss**.
//!
//! Paired, seat-swapped matches against this same agent at `N = 1`, through
//! `duels-arena`'s `ensemble_lab` example, `+/-` one binomial standard error.
//! The `N = 1` row is a control: the identical configuration against itself,
//! which says how much of each column is noise.
//!
//! | N | `Nodes(2_000)`, 400 games | `TimeMs(20)`, 200 games |
//! |---|---|---|
//! | 1 (control) | 52.2% +/- 2.5 | 49.5% +/- 3.5 |
//! | 2 | 46.9% +/- 2.5 | 44.2% +/- 3.5 |
//! | 4 | 35.5% +/- 2.4 | 32.0% +/- 3.3 |
//! | 8 | 32.2% +/- 2.3 | 33.0% +/- 3.3 |
//!
//! (Games run in parallel across seeds with the pool capped at 3-4 threads on
//! a 14-core machine that was also busy with other work, which lowers what a
//! `TimeMs` row's 20 ms buys without making it unfair — both sides alternate
//! inside one thread, and the harness prints each side's wall clock per game
//! as the check.)
//!
//! **The reason is the sampling ramp, not the ensembling.** This agent's
//! strength comes from spending spare budget on more playouts per leaf rather
//! than on another ply (see [`search::Searcher::think`]), and that ramp needs
//! budget to climb. `ensemble_lab --cost` at `Nodes(2_000)`:
//!
//! | N | nodes/decision | mean depth | playouts/leaf |
//! |---|---|---|---|
//! | 1 | 2000 | 1.28 | 116.5 |
//! | 2 | 2030 | 1.07 | 98.9 |
//! | 4 | 2092 | 1.00 | 66.2 |
//! | 8 | 2248 | 0.94 | 42.2 |
//!
//! Every configuration spends the budget it was given (the small overshoot is
//! one leaf's playouts per slice), but by `N = 8` a search is a single ply
//! evaluated with 42 playouts per leaf instead of 116 — and the depth column,
//! which reports the *deepest* of the `N` searches, has fallen below one:
//! even the best-off search in the ensemble sometimes returns a move without
//! having scored every root option. Averaging eight shallow, noisy opinions
//! is worse than holding one sharp one, and no combination rule at the root
//! can recover what the thinner sampling threw away.
//!
//! The budget is the whole story, and raising it says so: at `Nodes(8_000)`,
//! where even a halved slice saturates the ramp, `N = 2` scores 47.5% +/- 3.5
//! against a 46.5% +/- 3.5 control over 200 games — the five-point deficit it
//! showed at `Nodes(2_000)` is simply gone. Suggestive rather than conclusive
//! at that sample size, but it is the direction the explanation predicts, and
//! it is why this knob is worth keeping around rather than deleting: a future
//! configuration with budget to spare may want it.
//!
//! Two other costs are real but demonstrably secondary. An ensemble searches
//! every root move on a full window ([`Config::ensemble_exact_root`], so the
//! values being averaged are values and not fail-low bounds), which gives up
//! the root's alpha-beta cut-offs — but turning that off and averaging the
//! bounds instead scores 35.8% +/- 2.4 at `N = 4`, indistinguishable from the
//! 35.5% with it on, so the lost cut-offs are not what is doing the damage.
//! And `N` searches pay `N` determinizations and `N` iterative-deepening
//! restarts, which showed up as `N = 8` using 8% more wall clock than
//! `N = 1` for the same node budget — real, but small next to a 20-point
//! deficit.
//!
//! `N = 1` stays the default. The knob is kept, documented and tested,
//! because a measured negative result is worth more than a remembered
//! intuition — and because the same question at a budget where the ramp is
//! already saturated (a one-second move, say) is one command away:
//!
//! ```text
//! cargo run --release -p duels-arena --example ensemble_lab -- \
//!     --a alphabeta:dets=4 --b alphabeta:dets=1 --games 400 --budget nodes:2000
//! ```
//!
//! # Move-ordering priors, and why they are off too
//!
//! [`Config::order_priors`] adds `duels_strategy::action_prior` as a second
//! signal on top of `order_moves`'s static score (see
//! [`order::order_with_priors`]), gated to the top few plies so a
//! `duels_strategy::stance` — measured at about 29% of an MCTS rollout by
//! that crate's own benchmark — is not priced at every node of a search that
//! visits far more nodes per second than `mcts-uct` runs rollouts.
//!
//! Paired, seat-swapped matches through `ab_lab`, `+/-` one binomial standard
//! error, against the same identical-configuration noise floor the
//! ensembling section above uses:
//!
//! | budget | games | `order=priors` vs `order=static` | noise floor (identical config) |
//! |---|---|---|---|
//! | `Nodes(2_000)` | 400 | 52.0% +/- 2.5 | 52.2% +/- 2.5 |
//! | `TimeMs(20)` | 1,000 | 50.8% +/- 1.6 | 50.7% +/- 2.5 |
//! | `TimeMs(200)` | 400 | 50.8% +/- 2.5 | 51.5% +/- 3.5 |
//!
//! Every one of those falls right on top of its own noise floor. Individual
//! 200-game slices swung as high as 53% and as low as 48%, which is exactly
//! the spread the noise-floor column says to expect from this many games —
//! the effect, if there is one, is smaller than this harness can resolve. Nor
//! does it move the `mcts-uct` gap: 32.0% +/- 3.3 for `order=priors` against
//! 29.5% +/- 3.2 for `order=static`, on the same seeds at `TimeMs(20)`, a
//! two-and-a-half point gap well inside either side's own error bar.
//!
//! Two plausible reasons. First, at these budgets the search rarely completes
//! past the second ply anyway (mean depth 1.3-2.2 across the table above), so
//! the top-few-ply gate barely restricts *anything* — most nodes a real
//! search visits already qualify, which means the 29%-per-node cost is being
//! paid close to everywhere a deeper budget would have spared it, for
//! pruning gains that a tree this shallow has little room to cash in. Second,
//! `order_moves`'s existing static score already gets the wonders and the
//! highest-value cards to the front cheaply; the prior's main addition is
//! pricing *denial*, which matters most in exactly the imminent positions
//! `order_moves`'s own "closing/breaking a certainty" cases already exist to
//! catch by other means (see `score`'s wonder and card terms).
//!
//! `order_priors` stays off by default (`false`), for the same reason
//! `order_lookahead` and root ensembling do: a measured neutral result is
//! worth documenting and keeping around — the gate in [`order`] is there for
//! a future configuration with more budget to spare, where the tree is
//! actually deep enough for the top few plies to mean something — but it is
//! not a reason to spend the extra cycles by default.
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
//! one. Averaging over several root determinizations was the obvious
//! candidate for raising it and has since been tried: it loses, for the
//! reason given above — this agent has no spare budget to divide. What is
//! left is a leaf estimator accurate enough per unit of time to make a second
//! ply pay for itself.
//!
//! Ablations that did **not** help, so that nobody pays for them twice:
//! raising [`Config::chance_cap`] above 3 (a `cap` of 6 or 8 loses at a
//! matched wall clock — the thinning of the chance-outcome list, flagged as
//! the original build's other suspect, is not what was wrong); reporting a
//! playout's win / loss outcome instead of its margin; a playout policy that
//! follows the move-ordering heuristic; the one-ply-lookahead move ordering;
//! blending `duels_strategy::action_prior` into move ordering (above); and
//! root-determinization ensembling (above). Each is documented where it
//! lives.
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
    /// Blend `duels_strategy::action_prior` into `order_moves`'s static
    /// score instead (overridden by `order_lookahead`): see
    /// [`order::order_with_priors`] for the mechanism, and [`order::PRIOR_MAX_PLY`]
    /// / [`order::PRIOR_MIN_MOVES`] for the gate that keeps a `stance` from
    /// being priced at every node — `duels-strategy`'s own benchmark puts
    /// that at about 29% of an MCTS rollout, which this search would visit
    /// far more often per second than `mcts-uct` runs rollouts if it paid it
    /// unconditionally. Measured off by default; see the crate docs.
    pub order_priors: bool,
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
    /// How many independent root determinizations to search, each with its
    /// own `1/N` share of the budget, choosing the move with the best value
    /// averaged over the searches (see [`search::best_of`]).
    ///
    /// `1` is plain single-determinization search and is bit-for-bit the
    /// behaviour this crate had before the option existed. Larger values are
    /// Perfect Information Monte Carlo ensembling; see the crate docs for the
    /// measurement, which says they do not pay at these budgets.
    pub root_determinizations: usize,
    /// While ensembling, search every root move on a full window so the
    /// values being averaged are values rather than fail-low bounds. Costs
    /// the root's alpha-beta cut-offs; see the `root_full_window` field of
    /// [`search::Searcher`]. Ignored when `root_determinizations` is `1`, and
    /// exists so the ablation can be measured.
    pub ensemble_exact_root: bool,
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
            order_priors: false,
            weights: eval::Weights::DEFAULT,
            rollouts: 8,
            rollout_blend: 0.9,
            rollout_policy: playout::PolicyWeights::BIASED,
            rollout_metric: playout::Metric::MARGIN,
            rollout_common_seed: false,
            rollout_cap: 128,
            root_determinizations: 1,
            ensemble_exact_root: true,
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
            "max_depth={},chance_cap={},tt_bits={},tt={},star1={},order={},leaf={},dets={}",
            self.max_depth,
            self.chance_cap,
            self.tt_bits,
            self.use_tt,
            self.star1,
            if self.order_lookahead {
                "lookahead"
            } else if self.order_priors {
                "priors"
            } else if self.order_moves {
                "static"
            } else {
                "none"
            },
            self.describe_leaf(),
            self.root_determinizations.max(1),
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

        // `N` determinized worlds, each ground truth for one search only.
        let me = obs.current_player;
        let n = self.cfg.root_determinizations.max(1);
        let mut slices = Slices::new(budget, n);

        let mut runs: Vec<Vec<(Action, f64)>> = Vec::with_capacity(n);
        let mut first: Option<SearchResult> = None;
        let mut nodes = 0u64;
        let mut depth = 0u8;
        let mut samples = 0u32;
        for i in 0..n {
            let root = obs.sample_state(&mut self.rng);
            let mut searcher = Searcher::new(me, &self.cfg, &mut self.tt, slices.next(i));
            let result = searcher.think(&root, legal);
            runs.push(searcher.root_values().to_vec());
            first = first.or(Some(result));
            nodes += result.nodes;
            depth = depth.max(result.depth);
            samples = result.samples;
        }

        // With one determinization this is that search's own answer, by
        // construction: see `search::best_of`.
        let (best, value) = search::best_of(&runs).unwrap_or((legal[0], 0.0));

        self.stats.decisions += 1;
        self.stats.nodes += nodes;
        // The deepest of the ensemble's searches; with one search, its own.
        self.stats.depth_sum += u64::from(depth);
        self.stats.max_depth_reached = self.stats.max_depth_reached.max(depth);
        // One determinization reports its own search verbatim. An ensemble
        // has no single search to report, so it reports the pooled work and
        // the averaged value of the move it settled on.
        self.last = Some(match first {
            Some(result) if n == 1 => result,
            _ => SearchResult {
                best,
                value,
                depth,
                nodes,
                samples,
            },
        });

        // The sampled states agree with the observation on every public fact,
        // so a search only ever sees the moves it was given; the fallback is
        // belt and braces.
        if legal.contains(&best) {
            best
        } else {
            legal[0]
        }
    }
}

/// One search budget, divided into `n` slices — one per root determinization.
///
/// A node budget is partitioned exactly: every slice gets `total / n` nodes
/// and the first `total % n` slices get one more, so the slices sum to the
/// whole budget however indivisible it is.
///
/// A time budget is divided against a single shared deadline: slice `i` gets
/// whatever is left over the number of slices still to run, rounded up to the
/// millisecond [`Budget::TimeMs`] is expressed in. Sharing one deadline is
/// what keeps a slice that overran from stealing from the total rather than
/// adding to it, and the "remaining over remaining" form means the last slice
/// still gets the whole rest of the budget if the earlier ones came in under.
///
/// With `n == 1` both arms hand the original [`Budget`] straight through, so
/// a single determinization is not merely allocated the same budget, it is
/// given the identical value.
#[derive(Debug)]
enum Slices {
    Nodes { total: u64, n: u64 },
    Time { end: std::time::Instant, n: u64 },
    Whole(Budget),
}

impl Slices {
    fn new(budget: Budget, n: usize) -> Self {
        let n = n.max(1) as u64;
        match budget {
            _ if n == 1 => Slices::Whole(budget),
            Budget::Nodes(total) => Slices::Nodes { total, n },
            Budget::TimeMs(ms) => Slices::Time {
                // `Budget::TimeMs` is the one place an agent is asked to read
                // the clock; the workspace ban (see `clippy.toml`) is lifted
                // in `search::clock` and nowhere else.
                end: search::clock::now() + std::time::Duration::from_millis(ms),
                n,
            },
        }
    }

    /// The budget for slice `i`.
    fn next(&mut self, i: usize) -> Budget {
        let i = i as u64;
        match *self {
            Slices::Whole(budget) => budget,
            // A quotient plus a remainder, rather than
            // `total*(i+1)/n - total*i/n`, so a `Nodes(u64::MAX)` budget
            // cannot overflow the multiplication.
            Slices::Nodes { total, n } => Budget::Nodes(total / n + u64::from(i < total % n)),
            Slices::Time { end, n } => {
                let left = end.saturating_duration_since(search::clock::now());
                let per = left / u32::try_from(n - i).unwrap_or(1);
                // Rounded up to the millisecond: truncation would
                // systematically hand an ensemble less wall clock than a
                // single search gets.
                Budget::TimeMs(per.as_nanos().div_ceil(1_000_000) as u64)
            }
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

    /// `choose` exactly as it read before root ensembling existed: one
    /// determinization, one searcher, the whole budget, the searcher's own
    /// answer.
    ///
    /// The reference for the `root_determinizations = 1` path. A copy on
    /// purpose — a test that called the new code would prove nothing.
    fn legacy_choose(
        rng: &mut StdRng,
        cfg: &Config,
        tt: &mut Table,
        obs: &Observation,
        legal: &[Action],
        budget: Budget,
    ) -> (Action, SearchResult) {
        let root = obs.sample_state(rng);
        let me = obs.current_player;
        let result = {
            let mut searcher = Searcher::new(me, cfg, tt, budget);
            searcher.think(&root, legal)
        };
        let best = if legal.contains(&result.best) {
            result.best
        } else {
            legal[0]
        };
        (best, result)
    }

    /// The equivalence that makes the option safe to add: with
    /// `root_determinizations = 1` the agent is the agent this crate shipped
    /// before — same determinization from the same RNG stream, same budget
    /// handed to the same searcher, same transposition table, same move, and
    /// the same [`SearchResult`] reported back.
    ///
    /// Whole games, not just opening positions, so the two RNG streams and
    /// both tables have to stay in step across dozens of decisions.
    #[test]
    fn one_determinization_is_the_pre_ensemble_agent_move_for_move() {
        for seed in 0..3u64 {
            let cfg = Config {
                seed,
                tt_bits: 16,
                root_determinizations: 1,
                ..Config::default()
            };
            let mut agent = AlphaBetaAgent::with_config(cfg.clone());
            // The same seed and table size, driven by the copy above.
            let mut legacy_rng = StdRng::seed_from_u64(seed);
            let mut legacy_tt = Table::with_bits(cfg.tt_bits);

            let mut state = engine::new_game(seed ^ 0x0BAD_CAFE);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x99);
            let mut decisions = 0u32;
            loop {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let obs = state.observation();
                // Budgets small enough that rounds abort part-way, which is
                // where the reported root values and the chosen move could
                // most easily come apart.
                let budget = Budget::Nodes(60 + u64::from(decisions % 7) * 23);
                let got = agent.choose(&obs, &legal, budget);
                if legal.len() == 1 {
                    // A forced move is returned without touching the RNG or
                    // the table, in both versions.
                    assert_eq!(got, legal[0]);
                } else {
                    let (want, want_result) =
                        legacy_choose(&mut legacy_rng, &cfg, &mut legacy_tt, &obs, &legal, budget);
                    assert_eq!(
                        got, want,
                        "seed {seed}, decision {decisions}: ensembling changed the N=1 move"
                    );
                    let last = agent.last_search().expect("a search happened");
                    assert_eq!(last.best, want_result.best);
                    assert_eq!(last.value, want_result.value);
                    assert_eq!(last.depth, want_result.depth);
                    assert_eq!(last.nodes, want_result.nodes);
                    assert_eq!(last.samples, want_result.samples);
                }
                engine::apply(&mut state, got, &mut rng).expect("a legal action");
                decisions += 1;
                assert!(decisions < 5_000);
            }
            assert!(decisions > 20, "the game was too short to prove much");
        }
    }

    /// A node budget is partitioned, not multiplied: `N` searches share the
    /// nodes one search would have had.
    #[test]
    fn the_node_budget_is_split_across_determinizations() {
        let state = engine::new_game(23);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        let mut spent = Vec::new();
        for n in [1usize, 2, 4, 8] {
            let mut agent = AlphaBetaAgent::with_config(Config {
                seed: 4,
                tt_bits: 16,
                root_determinizations: n,
                ..Config::default()
            });
            let a = agent.choose(&obs, &legal, Budget::Nodes(4_000));
            assert!(legal.contains(&a));
            assert!(agent.config().describe().contains(&format!("dets={n}")));
            spent.push(agent.stats().nodes);
        }
        // Every configuration spends about one budget in total — the searcher
        // overshoots its slice by whatever the node it was in cost, so `N`
        // slices overshoot `N` times, which is the honest cost of the split
        // and is bounded by a few playouts per slice.
        for (i, nodes) in spent.iter().enumerate() {
            assert!(
                *nodes >= 4_000 && *nodes < 4_000 + 2_000 * (1 << i),
                "N={} spent {nodes} nodes on a 4000-node budget: {spent:?}",
                1 << i
            );
        }
    }

    #[test]
    fn ensembling_still_returns_a_legal_move_at_a_time_budget() {
        let state = engine::new_game(19);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        let mut agent = AlphaBetaAgent::with_config(Config {
            seed: 1,
            tt_bits: 16,
            root_determinizations: 4,
            ..Config::default()
        });
        #[allow(clippy::disallowed_methods)]
        let start = std::time::Instant::now();
        let a = agent.choose(&obs, &legal, Budget::TimeMs(40));
        #[allow(clippy::disallowed_methods)]
        let elapsed = start.elapsed();
        assert!(legal.contains(&a));
        assert!(agent.stats().nodes > 0);
        // Four slices of one budget, not four budgets. The bound is loose so a
        // loaded CI box cannot fail the build.
        assert!(
            elapsed < std::time::Duration::from_millis(2_000),
            "took {elapsed:?} for a 40 ms budget split four ways"
        );
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
