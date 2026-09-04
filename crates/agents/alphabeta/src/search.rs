//! The search itself: expectimax over the game's chance nodes, with
//! alpha-beta pruning at the decision nodes and Star1 pruning at the chance
//! nodes, driven by iterative deepening.
//!
//! # Node types
//!
//! - A **decision node** is a state where a player has a move. It is a *max*
//!   node when the player to move is the searching agent and a *min* node
//!   otherwise. That is not a strict alternation: `play_again` wonders, the
//!   banked extra turn, and the pending-choice actions (progress token,
//!   Mausoleum build, destroy, first player) all let the same player act
//!   twice in a row, which is why values are kept in one absolute frame
//!   ("victory points for `me`") instead of being negamax'd. Sign errors in
//!   a negamax formulation are exactly the kind of silent bug this game's
//!   irregular turn order invites.
//! - A **chance node** sits between a decision and its successor whenever the
//!   action resolves randomness: taking a card can uncover one or two
//!   face-down cards, and The Great Library draws three set-aside progress
//!   tokens. [`engine::chance_outcomes`] enumerates the distinct outcomes and
//!   their probabilities from public knowledge only; the node's value is the
//!   probability-weighted mean of its children's values. Nothing is resampled
//!   and no RNG is involved.
//!
//! - A **horizon leaf** is scored by [`leaf_value`]: a few playouts to the end
//!   of the game, blended with [`eval`]'s static form. That choice, and the
//!   fact that spare budget goes into more playouts rather than another ply,
//!   is the single biggest determinant of this agent's strength; see
//!   [`Searcher::think`] and the [`playout`] module.
//!
//! # Pruning
//!
//! Decision nodes are plain fail-soft alpha-beta. Chance nodes use Star1:
//! because every value lies in `[V_MIN, V_MAX]`, the partial sum after `k`
//! children bounds the node's final value, which both lets a chance node give
//! up early and lets it hand each child a narrowed window. Both are
//! value-preserving, which [`tests::pruning_agrees_with_unpruned_expectimax`]
//! checks against a deliberately naive reference implementation — with and
//! without the playout leaf, since the property only holds if a leaf value is
//! a pure function of its position.

use duels_core::engine::{self, Outcome};
use duels_core::{Action, GameResult, GameState, Player};

use crate::eval;
use crate::order;
use crate::playout;
use crate::tt::{Bound, Table};
use crate::Config;

/// Value of a won game, less one point per ply so the search prefers to win
/// sooner and to lose later.
pub const MATE: f64 = 100_000.0;
/// Anything at least this large is a proven win (or, negated, a proven loss)
/// rather than a heuristic evaluation.
pub const MATE_THRESHOLD: f64 = MATE - 10_000.0;
/// Upper bound on any value the search can produce; the "best conceivable"
/// bound Star1 uses at chance nodes.
pub const V_MAX: f64 = MATE + 1.0;
/// Lower bound on any value the search can produce.
pub const V_MIN: f64 = -V_MAX;

/// How often the budget is re-checked, in nodes (a `2^k - 1` mask). Small
/// enough that a node budget is not overshot by much, large enough that the
/// clock read a time budget needs is amortised away.
const CHECK_MASK: u64 = 0x3F;

/// The same, when the leaves run playouts: every node. A leaf then costs tens
/// of engine plies per playout, which dwarfs a clock read, and checking only
/// every 64th node overshoots a wall-clock budget by more than half again.
const CHECK_MASK_ROLLOUT: u64 = 0x0;

mod clock {
    //! The one place in this crate that reads a clock.
    //!
    //! The workspace `clippy.toml` bans wall-clock reads so that the rules
    //! engine and its agents stay reproducible, and asks the few callers that
    //! genuinely need time to allow the lint at the call site and say why.
    //! An agent honouring `Budget::TimeMs` is one: there is no way to obey a
    //! wall-clock budget without a wall clock. Under `Budget::Nodes` this is
    //! never called and the agent is bit-for-bit deterministic, which is what
    //! every test uses.

    /// Now, for deadline comparison only.
    #[allow(clippy::disallowed_methods)]
    pub fn now() -> std::time::Instant {
        std::time::Instant::now()
    }
}

/// What stops the search.
#[derive(Debug, Clone, Copy)]
enum Limit {
    Nodes(u64),
    Deadline(std::time::Instant),
}

/// An alpha-beta search window.
#[derive(Debug, Clone, Copy)]
struct Window {
    alpha: f64,
    beta: f64,
}

impl Window {
    const FULL: Window = Window {
        alpha: V_MIN,
        beta: V_MAX,
    };
}

/// What one root search found.
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    /// The move to play.
    pub best: Action,
    /// Its value, in victory points for the searching player.
    pub value: f64,
    /// The deepest iteration that ran to completion.
    pub depth: u8,
    /// Decision nodes visited.
    pub nodes: u64,
    /// Playouts per leaf in the last iteration that ran; `0` when the leaves
    /// were scored by the static evaluation alone.
    pub samples: u32,
}

/// One root search over one determinized state.
#[derive(Debug)]
pub struct Searcher<'a> {
    me: Player,
    cfg: &'a Config,
    tt: &'a mut Table,
    limit: Limit,
    nodes: u64,
    aborted: bool,
    /// One reusable move buffer per ply, so the hot loop does not allocate a
    /// `Vec<Action>` per node.
    bufs: Vec<Vec<Action>>,
    /// Scratch move list for playouts at the leaves.
    pbuf: Vec<Action>,
    /// How often to re-check the budget, in nodes; see [`CHECK_MASK`].
    check_mask: u64,
    /// Playouts per leaf for the iteration currently running. Starts at
    /// `cfg.rollouts` and doubles once the depth ceiling is reached.
    rollouts: u32,
    /// Playout seed for the round currently running, used only when
    /// [`Config::rollout_common_seed`] is set. Advanced once per round, so
    /// each round's estimate is an independent sample, but constant *within* a
    /// round, which is what makes the leaf value a pure function of its
    /// position (see [`playout`]).
    round_salt: u64,
}

impl<'a> Searcher<'a> {
    /// A searcher for `me`, bounded by `budget`, sharing `tt`.
    ///
    /// Bumps the table's generation: a new root means a new determinization,
    /// and values from the previous one are no longer sound.
    pub fn new(
        me: Player,
        cfg: &'a Config,
        tt: &'a mut Table,
        budget: duels_agents_api::Budget,
    ) -> Self {
        let limit = match budget {
            duels_agents_api::Budget::Nodes(n) => Limit::Nodes(n.max(1)),
            duels_agents_api::Budget::TimeMs(ms) => {
                Limit::Deadline(clock::now() + std::time::Duration::from_millis(ms.max(1)))
            }
        };
        tt.new_generation();
        Self {
            me,
            cfg,
            tt,
            limit,
            nodes: 0,
            aborted: false,
            bufs: Vec::new(),
            pbuf: Vec::with_capacity(32),
            check_mask: if cfg.rollouts > 0 {
                CHECK_MASK_ROLLOUT
            } else {
                CHECK_MASK
            },
            rollouts: cfg.rollouts,
            round_salt: 0x5EED,
        }
    }

    /// Search `root` by iterative deepening and return the best of `legal`.
    ///
    /// A move is available from the moment the depth-1 iteration finishes.
    ///
    /// # Partially completed iterations
    ///
    /// An iteration that runs out of budget part-way is *not* thrown away
    /// wholesale. The root moves are ordered best-first from the previous
    /// iteration, so the first of them is the move that iteration chose; any
    /// later move that outscored it here was judged with strictly more
    /// information — a ply more, or twice the playouts — and is adopted even
    /// though the rest of the list never got searched. `result.depth` still
    /// reports only the depth of the last iteration that *finished*, because
    /// that is what the value is worth.
    ///
    /// This matters much more than it looks. Iterative deepening spends most
    /// of its budget on its final, incomplete iteration; with a static leaf
    /// evaluation the wasted fraction is small because the next depth is only
    /// a few times more expensive, but once the leaves run playouts a node
    /// costs tens of engine plies and the aborted iteration can be most of
    /// the search.
    ///
    /// # Deepening the sampling before the horizon
    ///
    /// When the leaves run playouts ([`Config::rollouts`]), spare budget goes
    /// first into *doubling the playouts per leaf* and only then into another
    /// ply, up to [`Config::rollout_cap`]. That ordering is measured, not
    /// assumed, and it is the opposite of what a chess engine would do.
    ///
    /// Two things drive it. One ply here costs about thirtyfold — a dozen
    /// legal moves times the chance node that follows most of them — and at a
    /// realistic budget thirty times the sampling is worth more than the ply.
    /// And a max node taking the largest of several *noisy* leaf estimates is
    /// biased upward by roughly the spread of that noise, a min node
    /// symmetrically downward, so a deeper tree over under-sampled leaves
    /// does not merely fail to help, it actively degrades the root ordering.
    /// [`Config::rollout_common_seed`] attacks the same bias from the other
    /// side.
    ///
    /// # Panics
    ///
    /// Panics if `legal` is empty.
    pub fn think(&mut self, root: &GameState, legal: &[Action]) -> SearchResult {
        assert!(!legal.is_empty(), "think needs at least one legal move");
        let mut moves = legal.to_vec();
        self.order(root, &mut moves, None);

        let mut result = SearchResult {
            best: moves[0],
            value: 0.0,
            depth: 0,
            nodes: 0,
            samples: self.rollouts,
        };

        let mut depth = 1u8;
        loop {
            let mut window = Window::FULL;
            let mut iter_best = moves[0];
            let mut iter_value = V_MIN;
            let mut scored: Vec<(Action, f64)> = Vec::with_capacity(moves.len());

            for &action in &moves {
                let v = self.child_value(root, action, u32::from(depth) - 1, 1, window);
                if self.aborted {
                    break;
                }
                scored.push((action, v));
                if v > iter_value {
                    iter_value = v;
                    iter_best = action;
                }
                // The root is always a max node: `me` is the player to move.
                window.alpha = window.alpha.max(v);
            }

            if self.aborted {
                // Salvage a better-informed opinion if there is one:
                // `moves[0]` is the previous round's choice, so anything that
                // outscored it here beat the standing answer with more
                // information behind it.
                if !scored.is_empty() && iter_best != moves[0] {
                    result.best = iter_best;
                    result.value = iter_value;
                }
                break;
            }

            result.best = iter_best;
            result.value = iter_value;
            result.depth = depth;

            // Best-first for the next, deeper iteration.
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            moves.clear();
            moves.extend(scored.into_iter().map(|(a, _)| a));

            // A proven result will not change by looking further, and a
            // forced move needs no second opinion.
            if iter_value.abs() >= MATE_THRESHOLD || moves.len() == 1 || self.out_of_budget() {
                break;
            }

            // Spend the next slice of budget on sharper leaves before a
            // deeper horizon: see the note on the ramp above `think`.
            if self.rollouts > 0 && self.rollouts < self.cfg.rollout_cap {
                self.rollouts = (self.rollouts * 2).min(self.cfg.rollout_cap);
                // Leaf values just changed, so nothing stored under the old
                // sample count is comparable any more, and the next round
                // wants its own simulated luck.
                self.tt.new_generation();
                self.round_salt = self
                    .round_salt
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(1);
                result.samples = self.rollouts;
            } else if depth < self.cfg.max_depth {
                depth += 1;
            } else {
                break;
            }
        }

        result.nodes = self.nodes;
        result
    }

    /// Nodes visited so far.
    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    /// Value of taking `action` in `state`: the successor's value, or — when
    /// the action resolves randomness — the expectation over the resolutions.
    fn child_value(
        &mut self,
        state: &GameState,
        action: Action,
        depth: u32,
        ply: u32,
        window: Window,
    ) -> f64 {
        if !resolves_chance(state, action) {
            let mut next = *state;
            if !apply(&mut next, action, &Outcome::default()) {
                return self.static_eval(state);
            }
            return self.node(&next, depth, ply, window);
        }

        let outcomes = reduced_outcomes(state, action, self.cfg.chance_cap);
        if outcomes.len() == 1 {
            let mut next = *state;
            if !apply(&mut next, action, &outcomes[0].0) {
                return self.static_eval(state);
            }
            return self.node(&next, depth, ply, window);
        }
        self.chance_node(state, action, &outcomes, (depth, ply), window)
    }

    /// Probability-weighted mean of the outcomes' values, with Star1 pruning.
    ///
    /// `outcomes`' weights are normalised to sum to one by
    /// [`reduced_outcomes`], which the bounds below rely on.
    fn chance_node(
        &mut self,
        state: &GameState,
        action: Action,
        outcomes: &[(Outcome, f64)],
        (depth, ply): (u32, u32),
        window: Window,
    ) -> f64 {
        let Window { alpha, beta } = window;
        // `acc` is the weighted value of the children already searched;
        // `left` is the weight still outstanding, this one included.
        let mut acc = 0.0;
        let mut left = 1.0;

        for (outcome, weight) in outcomes {
            let w = *weight;
            let rest = left - w;
            if self.cfg.star1 {
                // What the node's value could still turn out to be.
                let upper = acc + left * V_MAX;
                let lower = acc + left * V_MIN;
                if lower >= beta {
                    return lower;
                }
                if upper <= alpha {
                    return upper;
                }
            }
            // The window this child has to land in for the node as a whole to
            // land inside `(alpha, beta)`, given that every child after it
            // could still be anything in `[V_MIN, V_MAX]`. `None` means the
            // child is searched on a full window, and so its result is a
            // value rather than a bound and cannot cut this node short.
            let narrowed = if self.cfg.star1 {
                let a = ((alpha - acc - rest * V_MAX) / w).max(V_MIN);
                let b = ((beta - acc - rest * V_MIN) / w).min(V_MAX);
                // `a < b` unless the bounds check above should already have
                // returned, so this is only `None` through floating-point
                // slop; falling back to a full window is always safe.
                (a < b).then_some(Window { alpha: a, beta: b })
            } else {
                None
            };

            let mut next = *state;
            if !apply(&mut next, action, outcome) {
                // The engine refused an outcome `chance_outcomes` offered.
                // Fall back to the static value rather than corrupting the
                // expectation by dropping a term.
                acc += w * self.static_eval(state);
                left = rest;
                continue;
            }
            let v = self.node(&next, depth, ply, narrowed.unwrap_or(Window::FULL));
            if self.aborted {
                return acc + w * v;
            }
            if let Some(child_window) = narrowed {
                if v <= child_window.alpha {
                    // The child failed low, so `v` bounds it from above and
                    // the node as a whole is worth at most `alpha`.
                    return acc + w * v + rest * V_MAX;
                }
                if v >= child_window.beta {
                    // The child failed high: the node is worth at least
                    // `beta`.
                    return acc + w * v + rest * V_MIN;
                }
            }
            acc += w * v;
            left = rest;
        }
        acc
    }

    /// A decision node: max if the searching player is to move, min otherwise.
    fn node(&mut self, state: &GameState, depth: u32, ply: u32, window: Window) -> f64 {
        self.nodes += 1;
        // A finished game is a hard terminal. Its heuristic value is
        // irrelevant and would often point the other way (losing on points
        // while holding a big military lead, say).
        if let Some(result) = state.result() {
            return terminal_value(result, self.me, ply);
        }
        // The budget is checked *before* the horizon test, not after: a leaf
        // that runs playouts is the most expensive node in the tree, so a
        // search that only checked the clock at interior nodes would sail
        // past a wall-clock deadline by a wide margin.
        if self.nodes & self.check_mask == 0 && self.out_of_budget() {
            self.aborted = true;
        }
        if self.aborted {
            return self.static_eval(state);
        }
        if depth == 0 {
            // A playout costs about what a decision node does, times the
            // length of the game it plays out, so charging the budget one
            // node per playout keeps `Budget::Nodes` a usable proxy for work
            // — and puts it on the same footing as `duels-agent-mcts-uct`,
            // whose `Budget::Nodes(n)` is exactly `n` playouts.
            self.nodes += u64::from(self.rollouts);
            return self.leaf(state);
        }

        let Window {
            mut alpha,
            mut beta,
        } = window;
        let key = if self.cfg.use_tt {
            Some(crate::tt::state_key(state))
        } else {
            None
        };
        let mut tt_move = None;
        if let Some(key) = key {
            if let Some(e) = self.tt.probe(key) {
                tt_move = e.best;
                if u32::from(e.depth) >= depth {
                    let v = from_tt(e.value, ply);
                    match e.bound {
                        Bound::Exact => return v,
                        Bound::Lower => alpha = alpha.max(v),
                        Bound::Upper => beta = beta.min(v),
                    }
                    if alpha >= beta {
                        return v;
                    }
                }
            }
        }
        // Captured after the table has narrowed the window, so the bound the
        // result is stored under matches the window it was produced with.
        let (alpha_orig, beta_orig) = (alpha, beta);

        let mut moves = self.take_buf(ply);
        engine::legal_actions_into(state, &mut moves);
        if moves.is_empty() {
            // Only a finished game has no legal move, and that was handled
            // above; be total anyway.
            self.put_buf(ply, moves);
            return self.static_eval(state);
        }
        self.order(state, &mut moves, tt_move);

        let maximizing = state.current_player() == self.me;
        let mut best_value = if maximizing { V_MIN } else { V_MAX };
        let mut best_move = moves[0];

        for &action in &moves {
            let v = self.child_value(state, action, depth - 1, ply + 1, Window { alpha, beta });
            if self.aborted {
                break;
            }
            if maximizing {
                if v > best_value {
                    best_value = v;
                    best_move = action;
                }
                alpha = alpha.max(best_value);
            } else {
                if v < best_value {
                    best_value = v;
                    best_move = action;
                }
                beta = beta.min(best_value);
            }
            if alpha >= beta {
                break;
            }
        }
        self.put_buf(ply, moves);

        if let Some(key) = key {
            if !self.aborted {
                let bound = if best_value <= alpha_orig {
                    Bound::Upper
                } else if best_value >= beta_orig {
                    Bound::Lower
                } else {
                    Bound::Exact
                };
                self.tt.store(
                    key,
                    to_tt(best_value, ply),
                    u8::try_from(depth).unwrap_or(u8::MAX),
                    bound,
                    Some(best_move),
                );
            }
        }
        best_value
    }

    /// The static evaluation, with this search's weights. Used for the
    /// degenerate paths — an aborted iteration, an outcome the engine refused
    /// — where paying for a playout would be waste.
    fn static_eval(&self, state: &GameState) -> f64 {
        static_eval(state, self.me, self.cfg)
    }

    /// The value of a horizon leaf: the static evaluation, blended with the
    /// mean of a few playouts when the configuration asks for them.
    fn leaf(&mut self, state: &GameState) -> f64 {
        let key = if self.cfg.rollout_common_seed {
            self.round_salt
        } else {
            crate::tt::state_key(state)
        };
        leaf_value(state, self.me, self.cfg, self.rollouts, key, &mut self.pbuf)
    }

    /// Apply whichever move ordering the configuration asks for.
    fn order(&self, state: &GameState, moves: &mut [Action], tt_move: Option<Action>) {
        if self.cfg.order_lookahead {
            order::order_by_lookahead(state, self.me, moves, tt_move);
        } else if self.cfg.order_moves {
            order::order(state, moves, tt_move);
        }
    }

    fn out_of_budget(&self) -> bool {
        match self.limit {
            Limit::Nodes(n) => self.nodes >= n,
            Limit::Deadline(t) => clock::now() >= t,
        }
    }

    fn take_buf(&mut self, ply: u32) -> Vec<Action> {
        let i = ply as usize;
        while self.bufs.len() <= i {
            self.bufs.push(Vec::with_capacity(32));
        }
        std::mem::take(&mut self.bufs[i])
    }

    fn put_buf(&mut self, ply: u32, buf: Vec<Action>) {
        let i = ply as usize;
        if i < self.bufs.len() {
            self.bufs[i] = buf;
        }
    }
}

/// The static evaluation of `state` for `me` under `cfg`'s weights.
fn static_eval(state: &GameState, me: Player, cfg: &Config) -> f64 {
    eval::evaluate_with(state, me, &cfg.weights)
}

/// The value of a horizon leaf.
///
/// With `cfg.rollouts == 0` this is just the static evaluation, which is what
/// the crate did originally. Otherwise the static term is blended with the
/// mean of `cfg.rollouts` playouts (see [`playout`]); the blend keeps the
/// static term's knowledge of the two instant-win races, which a playout under
/// a random policy almost never stumbles into, while letting the playout speak
/// for the end-of-game scoring the static term can only guess at.
///
/// A leaf value is a pure function of the position, playouts included — see
/// [`playout::estimate`] — so the transposition table stays sound and
/// `Budget::Nodes` stays reproducible.
fn leaf_value(
    state: &GameState,
    me: Player,
    cfg: &Config,
    rollouts: u32,
    key: u64,
    buf: &mut Vec<Action>,
) -> f64 {
    let stat = static_eval(state, me, cfg);
    if rollouts == 0 || cfg.rollout_blend <= 0.0 {
        return stat;
    }
    let mc = playout::estimate(
        state,
        me,
        key,
        rollouts,
        &cfg.rollout_policy,
        cfg.rollout_metric,
        buf,
    );
    let b = cfg.rollout_blend.clamp(0.0, 1.0);
    ((1.0 - b) * stat + b * mc).clamp(-eval::EVAL_CLAMP, eval::EVAL_CLAMP)
}

/// Apply `action` with `outcome` forced. `false` if the engine refused.
#[inline]
fn apply(state: &mut GameState, action: Action, outcome: &Outcome) -> bool {
    engine::apply_with_outcome_unchecked(state, action, outcome).is_ok()
}

/// Whether `action` resolves any randomness in `state`, answered without
/// building the outcome list (which allocates, and is quadratic in the pool
/// size when two cards are uncovered).
fn resolves_chance(state: &GameState, action: Action) -> bool {
    let slot = match action {
        Action::Build { slot } | Action::Discard { slot } => slot,
        Action::BuildWonder { slot, wonder } => {
            // The Great Library draws three of the set-aside tokens.
            if wonder.def().choose_progress_token && state.set_aside_tokens().count() >= 3 {
                return true;
            }
            slot
        }
        _ => return false,
    };
    let l = duels_core::layout::layout(state.age());
    let occupied = state.occupied_slots() & !(1u32 << slot);
    // Slots this card covers that are still face down; each is uncovered only
    // if nothing else still covers it.
    let mut candidates = l.covers[usize::from(slot)] & occupied & !state.revealed_slots();
    while candidates != 0 {
        let i = candidates.trailing_zeros();
        candidates &= candidates - 1;
        if l.covered_by[i as usize] & occupied == 0 {
            return true;
        }
    }
    false
}

/// The distinct chance outcomes of `action`, weighted, thinned to at most
/// `cap` entries (`0` means no cap) and renormalised to sum to one.
///
/// Uncovering two face-down cards from a pool of a dozen-plus unseen cards
/// gives well over a hundred distinct outcomes, which no search can afford to
/// average over at every node. Above `cap` the list is thinned by a
/// deterministic stride — evenly spaced through the engine's ordering, which
/// runs unseen guilds first and then card ids, so the survivors stay spread
/// across the pool — and their probabilities are renormalised. The stride
/// offset is a function of the position, so the choice is reproducible (two
/// iterative-deepening passes over the same node average over the same
/// outcomes, which is what makes the transposition table sound) without being
/// the same slice of the pool at every node.
///
/// This is the search's main approximation and it is a real one: the value of
/// a chance node is an expectation over a *sample* of the outcome space, not
/// all of it. Within one node the survivors are equally weighted and equally
/// likely a priori, so the estimate is unbiased, but a `cap` of three does
/// not make the variance of a 200-outcome distribution disappear.
///
/// It is nevertheless *not* what was holding the agent back, which is worth
/// recording because it was the obvious suspect. At a matched wall-clock
/// budget a `cap` of 6 or 8 loses to a `cap` of 3 — 45% and 37% over 100 games
/// with the static-evaluation leaf, 50% and 45% with the playout leaf — and a
/// `cap` of 1 or 2 loses too (44% and 48%). Three is a local optimum in both
/// regimes: the depth an honest expectation costs is worth more than the
/// variance it removes. A less lossy sampling scheme (weighted-without-
/// replacement top-K rather than a stride) would move numbers of that size at
/// best.
pub fn reduced_outcomes(state: &GameState, action: Action, cap: usize) -> Vec<(Outcome, f64)> {
    let mut all = engine::chance_outcomes(state, action);
    if cap > 0 && all.len() > cap {
        let n = all.len();
        // `stride >= 1`, and `(cap - 1) * stride < n`, so the kept indices are
        // distinct: no outcome is double-counted.
        let stride = n / cap;
        let offset = (chance_seed(state, action) % n as u64) as usize;
        let mut kept = Vec::with_capacity(cap);
        for i in 0..cap {
            kept.push(all[(offset + i * stride) % n]);
        }
        all = kept;
    }
    let total: f64 = all.iter().map(|(_, p)| *p).sum();
    if total > 0.0 {
        for e in all.iter_mut() {
            e.1 /= total;
        }
    } else if !all.is_empty() {
        let w = 1.0 / all.len() as f64;
        for e in all.iter_mut() {
            e.1 = w;
        }
    }
    all
}

/// A deterministic per-(position, action) value, used only to offset the
/// outcome stride.
fn chance_seed(state: &GameState, action: Action) -> u64 {
    let slot = match action {
        Action::Build { slot } | Action::Discard { slot } | Action::BuildWonder { slot, .. } => {
            slot
        }
        _ => 31,
    };
    u64::from(state.turn())
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(u64::from(state.occupied_slots()))
        .wrapping_add(u64::from(slot) << 32)
}

/// The value of a finished game, preferring to win sooner and to lose later.
pub fn terminal_value(result: GameResult, me: Player, ply: u32) -> f64 {
    match result.winner() {
        Some(w) if w == me => MATE - f64::from(ply),
        Some(_) => -(MATE - f64::from(ply)),
        None => 0.0,
    }
}

/// Normalise a mate score for storage: strip the distance to mate so the
/// entry can be reused at a different ply.
fn to_tt(v: f64, ply: u32) -> f64 {
    if v >= MATE_THRESHOLD {
        v + f64::from(ply)
    } else if v <= -MATE_THRESHOLD {
        v - f64::from(ply)
    } else {
        v
    }
}

/// Re-apply the distance to mate for the ply the entry is read at.
fn from_tt(v: f64, ply: u32) -> f64 {
    if v >= MATE_THRESHOLD {
        v - f64::from(ply)
    } else if v <= -MATE_THRESHOLD {
        v + f64::from(ply)
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// Reference implementation, for tests only
// ---------------------------------------------------------------------------

/// A deliberately naive expectimax: no pruning, no transposition table, no
/// move ordering, no budget, full recursion over the same reduced outcome
/// sets the real search uses.
///
/// The real search must agree with this exactly. Alpha-beta and Star1 bugs do
/// not crash — they silently return a wrong value and leave a weaker agent
/// behind — so this is the only thing standing between "pruning works" and
/// "pruning looks like it works".
#[cfg(test)]
fn reference_expectimax(state: &GameState, me: Player, depth: u32, ply: u32, cfg: &Config) -> f64 {
    if let Some(result) = state.result() {
        return terminal_value(result, me, ply);
    }
    if depth == 0 {
        let key = if cfg.rollout_common_seed {
            0x5EED
        } else {
            crate::tt::state_key(state)
        };
        return leaf_value(state, me, cfg, cfg.rollouts, key, &mut Vec::new());
    }
    let moves = engine::legal_actions(state);
    if moves.is_empty() {
        return static_eval(state, me, cfg);
    }
    let maximizing = state.current_player() == me;
    let mut best = if maximizing { V_MIN } else { V_MAX };
    for action in moves {
        let v = reference_child(state, me, action, depth - 1, ply + 1, cfg);
        best = if maximizing { best.max(v) } else { best.min(v) };
    }
    best
}

/// The reference's chance-node handling: a plain weighted mean, no pruning.
#[cfg(test)]
fn reference_child(
    state: &GameState,
    me: Player,
    action: Action,
    depth: u32,
    ply: u32,
    cfg: &Config,
) -> f64 {
    let outcomes = if resolves_chance(state, action) {
        reduced_outcomes(state, action, cfg.chance_cap)
    } else {
        vec![(Outcome::default(), 1.0)]
    };
    let mut acc = 0.0;
    for (outcome, w) in &outcomes {
        let mut next = *state;
        if !apply(&mut next, action, outcome) {
            acc += w * static_eval(state, me, cfg);
            continue;
        }
        acc += w * reference_expectimax(&next, me, depth, ply, cfg);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_agents_api::Budget;
    use duels_core::testing::StateBuilder;
    use rand::seq::SliceRandom;
    use rand::{rngs::StdRng, SeedableRng};

    fn cfg() -> Config {
        Config {
            max_depth: 3,
            chance_cap: 3,
            tt_bits: 14,
            rollouts: 0,
            ..Config::default()
        }
    }

    /// The same, with the playout leaf switched on. `rollout_cap` matches
    /// `rollouts` so the sampling ramp stays out of the way: the property
    /// under test is that pruning does not change a value, at a fixed leaf
    /// definition.
    fn rollout_cfg() -> Config {
        Config {
            rollouts: 2,
            rollout_cap: 2,
            rollout_blend: 0.75,
            ..cfg()
        }
    }

    /// Positions that between them cover the wonder draft, ordinary turns in
    /// a real structure (so with live chance nodes) and a hand-built
    /// late-Age-III position with none.
    fn positions() -> Vec<GameState> {
        let mut out = vec![engine::new_game(11)]; // the wonder draft
        for seed in [1u64, 4, 9] {
            let mut st = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed);
            // Play the draft out plus a few turns, to get into the structure.
            for _ in 0..14 {
                let legal = engine::legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                let a = legal[(seed as usize * 7 + st.turn() as usize) % legal.len()];
                engine::apply_quiet(&mut st, a, &mut rng).unwrap();
            }
            out.push(st);
        }
        // Mid-Age-I with most of the pyramid still face down: the one
        // position here with heavy chance branching.
        {
            let mut st = engine::new_game(4);
            let mut rng = StdRng::seed_from_u64(4);
            for _ in 0..10 {
                let legal = engine::legal_actions(&st);
                let a = legal[(28 + st.turn() as usize) % legal.len()];
                engine::apply_quiet(&mut st, a, &mut rng).unwrap();
            }
            out.push(st);
        }
        out.push(
            StateBuilder::new()
                .age(3)
                .built(Player::One, &["palace", "guard-tower"])
                .built(Player::Two, &["temple", "stable"])
                .coins(Player::One, 9)
                .coins(Player::Two, 6)
                .conflict(3)
                .open_slots(&[(13, "arsenal"), (14, "senate"), (18, "arena"), (19, "port")])
                .build(),
        );
        out
    }

    fn search_with(state: &GameState, cfg: &Config, depth: u8) -> SearchResult {
        let mut tt = Table::with_bits(cfg.tt_bits);
        let cfg = Config {
            max_depth: depth,
            ..cfg.clone()
        };
        let legal = engine::legal_actions(state);
        let mut s = Searcher::new(
            state.current_player(),
            &cfg,
            &mut tt,
            Budget::Nodes(u64::MAX),
        );
        let r = s.think(state, &legal);
        assert_eq!(r.depth, depth, "the search did not complete every ply");
        r
    }

    /// The headline correctness property: pruning is a performance device
    /// only. Every combination of the switches must return the same value as
    /// a naive full expectimax over the same tree.
    #[test]
    fn pruning_agrees_with_unpruned_expectimax() {
        let all = positions();
        // (position, depth). Depth 3 only where the tree is small enough for
        // the naive reference to be quick.
        let cases = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (5, 3),
        ];
        for (i, depth) in cases {
            let st = &all[i];
            // Both leaf policies: a static evaluation, and the playout blend,
            // whose value has to stay a pure function of the position for the
            // transposition table to be sound.
            for base in [cfg(), rollout_cfg()] {
                let want =
                    reference_expectimax(st, st.current_player(), u32::from(depth), 0, &base);
                for (star1, use_tt, order_moves, order_lookahead) in [
                    (false, false, false, false),
                    (true, false, false, false),
                    (false, true, false, false),
                    (false, false, true, false),
                    (false, false, false, true),
                    (true, true, true, false),
                    (true, true, false, true),
                ] {
                    let c = Config {
                        star1,
                        use_tt,
                        order_moves,
                        order_lookahead,
                        ..base.clone()
                    };
                    let got = search_with(st, &c, depth).value;
                    assert!(
                        (got - want).abs() < 1e-6,
                        "position {i} depth {depth} rollouts={} star1={star1} tt={use_tt} \
                         order={order_moves}/{order_lookahead}: got {got}, want {want}",
                        base.rollouts
                    );
                }
            }
        }
    }

    #[test]
    fn move_ordering_does_not_change_the_result() {
        // The root move list is handed to the searcher, so shuffling it and
        // toggling the ordering heuristic covers both halves of "does the
        // order the moves arrive in matter?".
        for (i, st) in positions().iter().enumerate() {
            let depth = 2u8;
            let base = search_with(st, &cfg(), depth);
            let base_legal = engine::legal_actions(st);

            for shuffle_seed in 0..4u64 {
                let mut legal = base_legal.clone();
                legal.shuffle(&mut StdRng::seed_from_u64(shuffle_seed));
                for (order_moves, order_lookahead) in [(true, false), (false, false), (false, true)]
                {
                    let c = Config {
                        max_depth: depth,
                        order_moves,
                        order_lookahead,
                        ..cfg()
                    };
                    let mut tt = Table::with_bits(c.tt_bits);
                    let mut s =
                        Searcher::new(st.current_player(), &c, &mut tt, Budget::Nodes(u64::MAX));
                    let got = s.think(st, &legal);
                    assert!(
                        (got.value - base.value).abs() < 1e-6,
                        "position {i} shuffle {shuffle_seed} ordering \
                         {order_moves}/{order_lookahead}: {} vs {}",
                        got.value,
                        base.value
                    );
                    // Which move is returned may differ between equally good
                    // moves, but it has to actually be worth the root value.
                    let v = reference_child(
                        st,
                        st.current_player(),
                        got.best,
                        u32::from(depth) - 1,
                        1,
                        &c,
                    );
                    assert!(
                        (v - base.value).abs() < 1e-6,
                        "position {i} shuffle {shuffle_seed}: chose {:?} worth {v}, \
                         root value is {}",
                        got.best,
                        base.value
                    );
                }
            }
        }
    }

    #[test]
    fn pruning_actually_prunes() {
        // Not a correctness property, but if this regresses the agent has
        // quietly lost most of its depth. Measured on the chance-heavy
        // position, which is the only kind where the node count matters.
        let st = &positions()[4];
        let mut nodes = [0u64; 2];
        for (i, (star1, order_moves)) in [(false, false), (true, true)].into_iter().enumerate() {
            let c = Config {
                star1,
                order_moves,
                use_tt: false,
                ..cfg()
            };
            nodes[i] = search_with(st, &c, 3).nodes;
        }
        assert!(
            nodes[0] > 5_000,
            "expected a big tree to prune, got {}",
            nodes[0]
        );
        assert!(
            nodes[1] * 4 < nodes[0] * 3,
            "pruned {} vs unpruned {} nodes",
            nodes[1],
            nodes[0]
        );
    }

    #[test]
    fn a_won_position_is_a_hard_terminal_not_an_evaluation() {
        // Player One is at +8 with a two-shield card on the board and money
        // to buy it: pushing to the capital is an outright win, and the
        // search has to see it as one rather than as a large heuristic bonus.
        let st = StateBuilder::new()
            .age(3)
            .conflict(8)
            .coins(Player::One, 20)
            .current(Player::One)
            .open_slots(&[(19, "circus"), (18, "senate")])
            .build();
        let r = search_with(&st, &cfg(), 1);
        assert!(
            r.value >= MATE_THRESHOLD,
            "expected a proven win, got {} for {:?}",
            r.value,
            r.best
        );
        assert_eq!(r.best, Action::Build { slot: 19 });
    }

    #[test]
    fn a_lost_position_is_seen_as_lost() {
        // Symmetrically: Player One to move, but Player Two is one shield
        // from the capital and every move leaves them there.
        let st = StateBuilder::new()
            .age(3)
            .conflict(-8)
            .coins(Player::Two, 20)
            .current(Player::One)
            .open_slots(&[(19, "circus"), (18, "senate")])
            .build();
        let r = search_with(&st, &cfg(), 2);
        assert!(
            r.value <= -MATE_THRESHOLD,
            "expected a proven loss, got {}",
            r.value
        );
    }

    #[test]
    fn a_deeper_search_visits_more_nodes_and_still_returns_a_legal_move() {
        let st = &positions()[2];
        let mut last = 0;
        for depth in 1..=3u8 {
            let legal = engine::legal_actions(st);
            let r = search_with(st, &cfg(), depth);
            assert!(legal.contains(&r.best));
            assert!(r.nodes > last, "depth {depth}: {} <= {last}", r.nodes);
            last = r.nodes;
        }
    }

    #[test]
    fn a_tiny_budget_still_returns_a_legal_move() {
        for st in positions() {
            let legal = engine::legal_actions(&st);
            for budget in [
                Budget::Nodes(1),
                Budget::Nodes(7),
                Budget::Nodes(200),
                Budget::TimeMs(1),
            ] {
                let c = Config {
                    max_depth: 12,
                    ..cfg()
                };
                let mut tt = Table::with_bits(c.tt_bits);
                let mut s = Searcher::new(st.current_player(), &c, &mut tt, budget);
                let r = s.think(&st, &legal);
                assert!(legal.contains(&r.best), "{:?} is not legal", r.best);
            }
        }
    }

    #[test]
    fn reduced_outcomes_are_a_normalised_distribution_without_duplicates() {
        let mut st = engine::new_game(5);
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..9 {
            let legal = engine::legal_actions(&st);
            engine::apply_quiet(&mut st, legal[0], &mut rng).unwrap();
        }
        let mut saw_a_reduction = false;
        for action in engine::legal_actions(&st) {
            if !resolves_chance(&st, action) {
                continue;
            }
            let full = engine::chance_outcomes(&st, action).len();
            for cap in [1usize, 3, 8, 0] {
                let got = reduced_outcomes(&st, action, cap);
                let want_len = if cap == 0 || full <= cap { full } else { cap };
                assert_eq!(got.len(), want_len, "cap {cap}");
                if cap > 0 && full > cap {
                    saw_a_reduction = true;
                }
                let total: f64 = got.iter().map(|(_, p)| p).sum();
                assert!((total - 1.0).abs() < 1e-9, "cap {cap}: total {total}");
                for i in 0..got.len() {
                    for j in i + 1..got.len() {
                        assert_ne!(got[i].0, got[j].0, "cap {cap} kept a duplicate");
                    }
                }
            }
        }
        assert!(saw_a_reduction, "expected at least one capped chance node");
    }

    /// The sampling ramp is the agent's main budget-allocation decision, so
    /// it needs to actually engage — and only upward.
    #[test]
    fn spare_budget_goes_into_more_playouts_per_leaf() {
        let st = &positions()[3];
        let legal = engine::legal_actions(st);
        let c = Config {
            max_depth: 2,
            rollouts: 2,
            rollout_cap: 64,
            ..cfg()
        };
        let mut samples = Vec::new();
        for budget in [50u64, 500, 5_000, 50_000] {
            let mut tt = Table::with_bits(c.tt_bits);
            let mut s = Searcher::new(st.current_player(), &c, &mut tt, Budget::Nodes(budget));
            let r = s.think(st, &legal);
            assert!(legal.contains(&r.best));
            assert!(
                r.samples >= c.rollouts && r.samples <= c.rollout_cap,
                "budget {budget}: {} samples is outside {}..={}",
                r.samples,
                c.rollouts,
                c.rollout_cap
            );
            samples.push(r.samples);
        }
        assert!(
            samples.windows(2).all(|w| w[1] >= w[0]),
            "sample counts did not rise with the budget: {samples:?}"
        );
        assert!(
            samples.last() > samples.first(),
            "the ramp never engaged: {samples:?}"
        );
    }

    /// With no playouts configured the ramp must not exist at all, so the
    /// original static-evaluation agent is still exactly reproducible.
    #[test]
    fn the_static_evaluation_configuration_never_ramps() {
        let st = &positions()[3];
        let legal = engine::legal_actions(st);
        let c = Config::v1();
        for budget in [100u64, 10_000] {
            let mut tt = Table::with_bits(14);
            let mut s = Searcher::new(st.current_player(), &c, &mut tt, Budget::Nodes(budget));
            let r = s.think(st, &legal);
            assert_eq!(r.samples, 0, "budget {budget}");
        }
    }

    /// The playout leaf must not disturb the two hard-terminal properties:
    /// a proven win is still `MATE`-scale, not a large simulated average.
    #[test]
    fn a_playout_leaf_still_reports_a_proven_win_as_a_win() {
        let st = StateBuilder::new()
            .age(3)
            .conflict(8)
            .coins(Player::One, 20)
            .current(Player::One)
            .open_slots(&[(19, "circus"), (18, "senate")])
            .build();
        let r = search_with(&st, &rollout_cfg(), 1);
        assert!(
            r.value >= MATE_THRESHOLD,
            "expected a proven win, got {} for {:?}",
            r.value,
            r.best
        );
        assert_eq!(r.best, Action::Build { slot: 19 });
    }

    #[test]
    fn mate_scores_survive_a_transposition_table_round_trip() {
        for ply in [0u32, 1, 7] {
            let v = MATE - f64::from(ply);
            assert_eq!(from_tt(to_tt(v, ply), ply), v);
            assert_eq!(from_tt(to_tt(-v, ply), ply), -v);
            // Read back at a deeper ply, the mate distance follows the ply.
            assert_eq!(from_tt(to_tt(v, ply), ply + 2), MATE - f64::from(ply + 2));
        }
        // Ordinary evaluations are stored verbatim.
        assert_eq!(from_tt(to_tt(-3.25, 4), 4), -3.25);
    }

    #[test]
    fn chance_nodes_are_detected_exactly_where_the_engine_says_they_are() {
        // `resolves_chance` is a fast path around `chance_outcomes`; if the
        // two ever disagree the search silently stops averaging over a real
        // chance node, or pays to average over a fake one.
        for seed in 0..12u64 {
            let mut st = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x5EED);
            loop {
                let legal = engine::legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                for &action in &legal {
                    let outcomes = engine::chance_outcomes(&st, action);
                    let engine_says = outcomes.len() > 1
                        || !outcomes
                            .first()
                            .map(|o| o.0)
                            .unwrap_or_default()
                            .is_trivial();
                    assert_eq!(
                        resolves_chance(&st, action),
                        engine_says,
                        "seed {seed} turn {} action {action:?}",
                        st.turn()
                    );
                }
                let a = legal[(st.turn() as usize + seed as usize) % legal.len()];
                engine::apply_quiet(&mut st, a, &mut rng).unwrap();
            }
        }
    }
}
