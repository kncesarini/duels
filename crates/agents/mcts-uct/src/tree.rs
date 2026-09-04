//! The search tree: an index-based arena of decision, chance and terminal
//! nodes, plus the four MCTS phases over it.
//!
//! # Value convention
//!
//! **Every node's `value_sum` is accumulated from [`Player::One`]'s point of
//! view**, on the scale `1.0` for a Player One win, `0.5` for a draw, `0.0`
//! for a Player Two win. Backpropagation therefore adds the *same* number to
//! every node on the path — no sign flipping on the way up, which is the
//! usual place a two-player MCTS goes subtly wrong.
//!
//! The perspective flip instead happens at selection time, in exactly one
//! place: [`Tree::exploit`] returns `mean` for a node whose parent's mover is
//! Player One and `1.0 - mean` for Player Two, so UCB1 at every decision node
//! maximises the win probability *of the player about to move there*. Because
//! the reward scale is symmetric about `0.5`, `1.0 - mean` is exactly the
//! zero-sum negation, and the tree stays a proper minimax-in-expectation.
//!
//! # Node kinds
//!
//! - **Decision**: one player to move; children are the legal actions,
//!   expanded one per visit, selected by UCB1 once all are expanded.
//! - **Chance**: sits between a decision node and the position its action
//!   leads to, whenever that action's outcome depends on a hidden reveal
//!   (a card uncovered face-down, or The Great Library's token draw).
//!   Children are outcomes, drawn *proportionally to their true probability*
//!   — never by UCB1: a chance node is not something the search is trying to
//!   win, it is real game randomness being integrated over.
//! - **Terminal**: the [`GameResult`] is already determined.
//!
//! # Chance-node widening
//!
//! A single reveal has hundreds of possible outcomes, so a chance node that
//! made a fresh child for every visit would never be visited twice at the
//! same child and the tree below it would never deepen — the search would
//! collapse to one-ply lookahead. [`Config::chance_widen_alpha`] applies
//! *progressive widening* (Couëtoux et al.): a chance node with `n` children
//! and `v` visits draws a fresh outcome only while
//! `n < c * (v + 1)^alpha`, and otherwise re-selects among the outcomes it
//! already has, proportionally to their probabilities (renormalised). With
//! `alpha = 1.0` and a large `c` this degenerates to the unbiased,
//! fully-faithful estimator; the default `alpha = 0.5` trades a little bias
//! for a much deeper tree.
//!
//! # Strategy priors ([`PriorMode`])
//!
//! Optionally, a decision node's actions are ranked by
//! [`duels_strategy::action_prior`] the first time the node is expanded, so
//! that UCT spends its first visits on the moves a win-condition read says
//! matter. See [`PriorMode`] for the cost model and where the ranking is
//! computed (exactly once per expanded node, never per simulation).

use duels_core::engine::{self, Outcome};
use duels_core::{Action, GameResult, GameState, Player};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::chance;
use crate::rollout::{self, RolloutWeights};

/// Index into [`Tree::nodes`].
pub(crate) type NodeId = u32;

/// "No child here yet."
pub(crate) const NO_NODE: NodeId = NodeId::MAX;

/// How, if at all, [`duels_strategy`]'s policy layer steers the tree.
///
/// # Where the cost goes
///
/// A [`duels_strategy::Stance`] plus a full slate of
/// [`duels_strategy::action_prior`] values costs a meaningful fraction of one
/// playout (about 29% of a rollout on the machine `duels-strategy`'s
/// `action_prior` bench was run on). That is far too expensive to recompute on
/// every simulation that passes through a node, and cheap enough to pay once
/// per node.
///
/// So it is paid **once per decision node, on that node's first expansion**,
/// which is a strictly smaller set than "once per node": a node created by a
/// simulation and never revisited is a playout target only, and never pays.
/// The tree's `expand` is the one call site, and the `expanded == 0` guard
/// is what makes it once. Nothing in the simulation loop recomputes anything.
///
/// The tree's own `rankings` counter records the consultations so the claim
/// can be checked
/// instead of believed. Measured on a real mid-game position, a 2000-simulation
/// search allocates 2788 nodes and consults the strategy layer **583 times** —
/// 0.29 per simulation, because only the fifth of nodes that get revisited ever
/// expand. At 29% of a rollout each that predicts about 8% overhead, which is
/// what the arena measures end to end.
///
/// The two non-trivial modes are nested, not alternatives: `ProgressiveBias`
/// is `ExpansionOrder` plus a decaying selection term, so an ablation between
/// them measures the selection term on its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PriorMode {
    /// Don't consult the strategy layer at all. Bit-for-bit the agent this
    /// crate shipped before priors existed — same RNG stream, same expansion
    /// order, same UCB1 scores; see
    /// `tests::prior_none_is_the_pre_prior_agent_move_for_move`.
    None,
    /// Rank a decision node's actions by [`duels_strategy::action_prior`] on
    /// first expansion, highest first, so the moves a win-condition read
    /// favours are expanded — and their subtrees grown — before the rest.
    /// UCB1 itself is untouched: once every child exists this is exactly the
    /// same search, only reached from a different order.
    ///
    /// Ties keep the shuffled order the node was built with, so a position
    /// where the prior says nothing is still unbiased.
    ExpansionOrder,
    /// [`PriorMode::ExpansionOrder`], plus a `weight * prior / (visits + 1)`
    /// term added to each child's UCB1 score, where `prior` is the node's
    /// prior slate normalised to sum to one. The term dominates at the first
    /// visit and decays as real statistics accumulate, so it biases *which*
    /// moves get the early samples without changing what the search converges
    /// to.
    ProgressiveBias {
        /// Multiplier on the decaying prior term.
        weight: f64,
    },
}

impl PriorMode {
    /// Whether this mode needs a node's priors kept after ordering.
    #[inline]
    fn keeps_priors(&self) -> bool {
        matches!(self, PriorMode::ProgressiveBias { .. })
    }

    /// A compact, stable description for [`Config::describe`].
    fn describe(&self) -> String {
        match self {
            PriorMode::None => "none".to_string(),
            PriorMode::ExpansionOrder => "expansion_order".to_string(),
            PriorMode::ProgressiveBias { weight } => {
                format!("progressive_bias({weight:.3})")
            }
        }
    }
}

/// Tuning knobs for the search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// UCB1 exploration constant `c`, against rewards in `[0, 1]`.
    pub exploration: f64,
    /// Playout policy weights.
    pub rollout: RolloutWeights,
    /// Progressive-widening coefficient at chance nodes.
    pub chance_widen_c: f64,
    /// Progressive-widening exponent at chance nodes. `1.0` (with a large
    /// coefficient) gives the unbiased "always draw a fresh outcome"
    /// estimator; smaller values deepen the tree.
    pub chance_widen_alpha: f64,
    /// Safety cap on playout length. The game is finite, so this only ever
    /// fires on a bug.
    pub max_rollout_plies: u32,
    /// How many simulations to run between wall-clock checks under
    /// [`duels_agents_api::Budget::TimeMs`].
    pub time_check_interval: u64,
    /// How many independent root determinizations to search, each with its
    /// own tree and its own `1/N` share of the budget, combining their root
    /// visit counts to pick the move. `1` is plain single-determinization
    /// search and is bit-for-bit the behaviour this crate had before the
    /// option existed; see the crate docs for what larger values measure.
    pub root_determinizations: usize,
    /// Whether, and how, [`duels_strategy`]'s policy layer steers the tree.
    /// [`PriorMode::None`] is bit-for-bit the behaviour this crate had before
    /// the option existed.
    pub prior: PriorMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            exploration: 1.0,
            rollout: RolloutWeights::BIASED,
            chance_widen_c: 1.0,
            chance_widen_alpha: 0.5,
            max_rollout_plies: 2_000,
            time_check_interval: 64,
            root_determinizations: 1,
            prior: PriorMode::None,
        }
    }
}

impl Config {
    /// A compact, stable description for [`duels_agents_api::AgentSpec`].
    pub fn describe(&self) -> String {
        let w = &self.rollout;
        format!(
            "c={:.3};rollout=weights(build={},wonder={},discard={},chain_free={},new_symbol={},pair_complete={});chance=progressive-widening(c={:.2},alpha={:.2});dets={};prior={}",
            self.exploration,
            w.build,
            w.wonder,
            w.discard,
            w.chain_free_mult,
            w.new_symbol_mult,
            w.pair_complete_mult,
            self.chance_widen_c,
            self.chance_widen_alpha,
            self.root_determinizations.max(1),
            self.prior.describe(),
        )
    }
}

/// One sampled outcome of a chance node.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChanceChild {
    /// The reveal this child resolves.
    pub outcome: Outcome,
    /// Its probability under the true chance distribution.
    pub prob: f64,
    /// The decision (or terminal) node it leads to.
    pub node: NodeId,
}

/// What kind of node this is, and its kind-specific payload.
#[derive(Debug)]
pub(crate) enum Kind {
    /// The game is over here; `value` is the result from Player One's view.
    Terminal { value: f64 },
    /// One player must choose among `actions`.
    Decision {
        mover: Player,
        /// Legal actions, shuffled once so expansion order carries no bias.
        actions: Vec<Action>,
        /// `children[i]` is the node for `actions[i]`, or [`NO_NODE`].
        children: Vec<NodeId>,
        /// How many of `children` have been created, always a prefix.
        expanded: usize,
        /// `priors[i]` is the normalised [`duels_strategy::action_prior`] of
        /// `actions[i]`, computed once on this node's first expansion.
        ///
        /// Empty unless [`Config::prior`] is a mode that reads priors during
        /// *selection* ([`PriorMode::ProgressiveBias`]) — the ordering modes
        /// consume the ranking as they sort and keep nothing, so a node costs
        /// one `Vec` header and no elements.
        priors: Vec<f32>,
    },
    /// `action` has been chosen and its randomness must be resolved.
    Chance {
        action: Action,
        children: Vec<ChanceChild>,
    },
}

/// One node of the arena.
#[derive(Debug)]
pub(crate) struct Node {
    /// For decision/terminal nodes, the position. For a chance node, the
    /// position *before* its action is applied.
    pub state: GameState,
    pub visits: u32,
    /// Sum of playout values, always from Player One's perspective.
    pub value_sum: f64,
    pub kind: Kind,
}

impl Node {
    #[inline]
    pub fn mean(&self) -> f64 {
        if self.visits == 0 {
            0.5
        } else {
            self.value_sum / f64::from(self.visits)
        }
    }
}

/// `1.0` if Player One won, `0.0` if Player Two won, `0.5` for a draw.
#[inline]
pub(crate) fn value_of(result: GameResult) -> f64 {
    match result.winner() {
        Some(Player::One) => 1.0,
        Some(Player::Two) => 0.0,
        None => 0.5,
    }
}

/// UCB1 for one child: exploitation from the mover's perspective plus the
/// exploration bonus.
#[inline]
pub(crate) fn ucb1(exploit: f64, child_visits: u32, parent_visits: u32, c: f64) -> f64 {
    if child_visits == 0 {
        return f64::INFINITY;
    }
    exploit + c * (f64::from(parent_visits).ln() / f64::from(child_visits)).sqrt()
}

/// The arena and the search over it.
pub(crate) struct Tree {
    pub nodes: Vec<Node>,
    pub cfg: Config,
    /// Scratch: the path of the current simulation, root first.
    path: Vec<NodeId>,
    /// Scratch: legal-action buffer, reused to keep playouts allocation-free.
    buf: Vec<Action>,
    /// Total playouts performed.
    pub simulations: u64,
    /// How many times the strategy layer was consulted — one
    /// [`duels_strategy::Stance`] plus one slate of priors per increment.
    ///
    /// Exists to make [`PriorMode`]'s central cost claim *checkable* rather
    /// than merely asserted in prose: it must equal the number of decision
    /// nodes that were actually expanded and had a choice to make, and it must
    /// stay far below [`Tree::simulations`]. See
    /// `tests::the_prior_is_computed_once_per_expanded_node_not_per_simulation`.
    pub rankings: u64,
}

impl Tree {
    /// A tree rooted at `state`, whose root actions are restricted to
    /// `actions` (the actions the arena actually offered).
    pub fn new(state: GameState, actions: Vec<Action>, cfg: Config, rng: &mut StdRng) -> Self {
        let mut tree = Self {
            nodes: Vec::with_capacity(1024),
            cfg,
            path: Vec::with_capacity(64),
            buf: Vec::with_capacity(32),
            simulations: 0,
            rankings: 0,
        };
        let root = decision_node(state, actions, rng);
        tree.nodes.push(root);
        tree
    }

    /// The root's statistics for `action`, or `None` if this tree never
    /// expanded that action.
    fn root_child(&self, action: Action) -> Option<&Node> {
        let Kind::Decision {
            actions, children, ..
        } = &self.nodes[0].kind
        else {
            return None;
        };
        let i = actions.iter().position(|&a| a == action)?;
        let child = children[i];
        (child != NO_NODE).then(|| &self.nodes[child as usize])
    }

    /// Exploitation term for `child` from `mover`'s perspective: the one and
    /// only place the two-player perspective flip happens.
    #[inline]
    pub fn exploit(&self, child: NodeId, mover: Player) -> f64 {
        let mean = self.nodes[child as usize].mean();
        match mover {
            Player::One => mean,
            Player::Two => 1.0 - mean,
        }
    }

    /// Build a node for `state`, classifying it as terminal or a decision.
    fn node_for(&mut self, state: GameState, rng: &mut StdRng) -> Node {
        if let Some(result) = state.result() {
            return Node {
                state,
                visits: 0,
                value_sum: 0.0,
                kind: Kind::Terminal {
                    value: value_of(result),
                },
            };
        }
        engine::legal_actions_into(&state, &mut self.buf);
        if self.buf.is_empty() {
            // `legal_actions` is empty exactly when the game is over, so this
            // is unreachable; score it rather than trusting the invariant.
            let value = value_of(duels_core::scoring::civilian_result(&state));
            return Node {
                state,
                visits: 0,
                value_sum: 0.0,
                kind: Kind::Terminal { value },
            };
        }
        let actions = self.buf.clone();
        decision_node(state, actions, rng)
    }

    fn push(&mut self, node: Node) -> NodeId {
        self.nodes.push(node);
        (self.nodes.len() - 1) as NodeId
    }

    /// The node reached from `state` by `action`: either a chance node (if the
    /// action's outcome depends on a hidden reveal) or the resulting position.
    fn child_after(&mut self, state: GameState, action: Action, rng: &mut StdRng) -> NodeId {
        if chance::resolves_randomness(&state, action) {
            let node = Node {
                state,
                visits: 0,
                value_sum: 0.0,
                kind: Kind::Chance {
                    action,
                    children: Vec::new(),
                },
            };
            return self.push(node);
        }
        let mut next = state;
        // No randomness to resolve, so the trivial outcome is exact.
        if engine::apply_with_outcome(&mut next, action, &Outcome::default()).is_err() {
            // Only reachable if the action was not legal in this state, which
            // the caller guarantees; treat it as a dead end scored as a draw.
            return self.push(Node {
                state,
                visits: 0,
                value_sum: 0.0,
                kind: Kind::Terminal { value: 0.5 },
            });
        }
        let node = self.node_for(next, rng);
        self.push(node)
    }

    /// Rank decision node `id`'s actions by [`duels_strategy::action_prior`],
    /// highest first, and (for [`PriorMode::ProgressiveBias`]) keep the
    /// normalised weights alongside them.
    ///
    /// **Called exactly once per node**, from [`Tree::expand`] under an
    /// `expanded == 0` guard — which is both what makes the reordering sound
    /// (no child exists yet, so nothing is invalidated by permuting `actions`)
    /// and what bounds the cost to one [`duels_strategy::Stance`] per expanded
    /// node rather than one per simulation.
    ///
    /// Consumes no randomness: the sort is stable, so the shuffle
    /// [`decision_node`] already applied survives as the tie-break among
    /// equally-rated moves.
    fn rank_by_prior(&mut self, id: NodeId) {
        let state = self.nodes[id as usize].state;
        let mut actions = match &mut self.nodes[id as usize].kind {
            Kind::Decision {
                actions, expanded, ..
            } => {
                debug_assert_eq!(*expanded, 0, "a node was ranked after it was expanded");
                std::mem::take(actions)
            }
            _ => return,
        };

        // One stance for the node, then one `action_prior` per legal move
        // against it — the split the strategy layer is designed around.
        self.rankings += 1;
        let stance = duels_strategy::stance(&state, state.current_player());
        let mut scored: Vec<(f64, Action)> = actions
            .iter()
            .map(|&a| (duels_strategy::action_prior(&state, a, &stance), a))
            .collect();
        // Descending, stably. `total_cmp` orders every f64 including the NaN
        // a weight should never be, so the sort can never panic.
        scored.sort_by(|x, y| y.0.total_cmp(&x.0));

        actions.clear();
        actions.extend(scored.iter().map(|&(_, a)| a));

        let priors = if self.cfg.prior.keeps_priors() {
            // Normalised so the selection term is on a scale that does not
            // move with the number of legal actions or the raw weights.
            let total: f64 = scored.iter().map(|&(w, _)| w).sum();
            let scale = if total > 0.0 { 1.0 / total } else { 0.0 };
            scored.iter().map(|&(w, _)| (w * scale) as f32).collect()
        } else {
            Vec::new()
        };

        match &mut self.nodes[id as usize].kind {
            Kind::Decision {
                actions: slot,
                priors: pslot,
                ..
            } => {
                *slot = actions;
                *pslot = priors;
            }
            _ => unreachable!("the kind was a decision a moment ago"),
        }
    }

    /// Create the next unexpanded child of decision node `id`, or `None` if
    /// they are all expanded.
    fn expand(&mut self, id: NodeId, rng: &mut StdRng) -> Option<NodeId> {
        // The node's first expansion is where the strategy layer is consulted
        // — once, for the whole node — and `PriorMode::None` never gets here.
        // A node with one legal action is skipped: ranking a single move can
        // change neither the expansion order nor a selection between children
        // there is only one of, so paying for a `Stance` would be pure loss,
        // and forced nodes (a pending effect choice with one answer) are
        // common enough to be worth the test.
        if self.cfg.prior != PriorMode::None
            && matches!(&self.nodes[id as usize].kind, Kind::Decision { expanded, actions, .. }
                if *expanded == 0 && actions.len() > 1)
        {
            self.rank_by_prior(id);
        }
        let (state, action, slot) = match &self.nodes[id as usize].kind {
            Kind::Decision {
                actions, expanded, ..
            } => {
                if *expanded >= actions.len() {
                    return None;
                }
                (self.nodes[id as usize].state, actions[*expanded], *expanded)
            }
            _ => return None,
        };
        let child = self.child_after(state, action, rng);
        match &mut self.nodes[id as usize].kind {
            Kind::Decision {
                children, expanded, ..
            } => {
                children[slot] = child;
                *expanded = slot + 1;
            }
            _ => unreachable!("expand called on a non-decision node"),
        }
        Some(child)
    }

    /// UCB1 selection among the expanded children of a fully expanded
    /// decision node, plus the [`PriorMode::ProgressiveBias`] term when that
    /// mode is on.
    fn select_ucb1(&self, id: NodeId) -> NodeId {
        let (mover, children, priors) = match &self.nodes[id as usize].kind {
            Kind::Decision {
                mover,
                children,
                priors,
                ..
            } => (*mover, children, priors),
            _ => unreachable!("select_ucb1 called on a non-decision node"),
        };
        let parent_visits = self.nodes[id as usize].visits.max(1);
        // Read once, outside the loop: `PriorMode::None` pays one branch per
        // selection and touches nothing else.
        let bias = match self.cfg.prior {
            PriorMode::ProgressiveBias { weight } => weight,
            _ => 0.0,
        };
        let mut best = NO_NODE;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &child) in children.iter().enumerate() {
            if child == NO_NODE {
                continue;
            }
            let child_visits = self.nodes[child as usize].visits;
            let mut score = ucb1(
                self.exploit(child, mover),
                child_visits,
                parent_visits,
                self.cfg.exploration,
            );
            if bias != 0.0 {
                if let Some(&p) = priors.get(i) {
                    score += bias * f64::from(p) / f64::from(child_visits + 1);
                }
            }
            if score > best_score {
                best_score = score;
                best = child;
            }
        }
        debug_assert_ne!(best, NO_NODE);
        best
    }

    /// Resolve chance node `id`: draw an outcome (or re-select an already
    /// expanded one under progressive widening) and return the child.
    fn resolve_chance(&mut self, id: NodeId, rng: &mut StdRng) -> NodeId {
        let (state, action) = match &self.nodes[id as usize].kind {
            Kind::Chance { action, .. } => (self.nodes[id as usize].state, *action),
            _ => unreachable!("resolve_chance called on a non-chance node"),
        };
        let visits = self.nodes[id as usize].visits;
        let width = match &self.nodes[id as usize].kind {
            Kind::Chance { children, .. } => children.len(),
            _ => unreachable!(),
        };
        let allowance =
            self.cfg.chance_widen_c * f64::from(visits + 1).powf(self.cfg.chance_widen_alpha);

        if width == 0 || (width as f64) < allowance {
            let (outcome, prob) = chance::sample(&state, action, rng);
            // A re-drawn outcome is not a new child; descend into the old one.
            if let Kind::Chance { children, .. } = &self.nodes[id as usize].kind {
                if let Some(existing) = children.iter().find(|c| c.outcome == outcome) {
                    return existing.node;
                }
            }
            let mut next = state;
            if engine::apply_with_outcome(&mut next, action, &outcome).is_err() {
                // The engine rejected a publicly consistent reveal, which
                // should not happen; fall back to the state's own layout.
                let mut fallback = state;
                if engine::apply_with_outcome(&mut fallback, action, &Outcome::default()).is_err() {
                    return self.push(Node {
                        state,
                        visits: 0,
                        value_sum: 0.0,
                        kind: Kind::Terminal { value: 0.5 },
                    });
                }
                next = fallback;
            }
            let node = self.node_for(next, rng);
            let child = self.push(node);
            if let Kind::Chance { children, .. } = &mut self.nodes[id as usize].kind {
                children.push(ChanceChild {
                    outcome,
                    prob,
                    node: child,
                });
            }
            return child;
        }

        // Widening exhausted: re-select among the outcomes already expanded,
        // proportionally to their true probabilities.
        let Kind::Chance { children, .. } = &self.nodes[id as usize].kind else {
            unreachable!()
        };
        let total: f64 = children.iter().map(|c| c.prob).sum();
        let mut r = rng.gen_range(0.0..total.max(f64::MIN_POSITIVE));
        for c in children {
            r -= c.prob;
            if r < 0.0 {
                return c.node;
            }
        }
        children[children.len() - 1].node
    }

    /// One selection / expansion / playout / backpropagation cycle.
    pub fn simulate(&mut self, rng: &mut StdRng) {
        self.path.clear();
        let mut node: NodeId = 0;
        self.path.push(node);
        // Set once the simulation has added a new node to the tree: the next
        // decision node reached is the playout's starting position.
        let mut fresh = false;
        let value;

        loop {
            // Read the kind out first so the arena is not borrowed while the
            // arm mutates it.
            let step = match &self.nodes[node as usize].kind {
                Kind::Terminal { value } => Step::Terminal(*value),
                Kind::Chance { .. } => Step::Chance,
                Kind::Decision { .. } => Step::Decision,
            };
            match step {
                Step::Terminal(v) => {
                    value = v;
                    break;
                }
                // Chance nodes are pass-throughs: they resolve real game
                // randomness and never end a simulation.
                Step::Chance => {
                    node = self.resolve_chance(node, rng);
                    self.path.push(node);
                }
                Step::Decision if fresh => {
                    let mut state = self.nodes[node as usize].state;
                    let result = rollout::play_out(
                        &mut state,
                        &self.cfg.rollout,
                        &mut self.buf,
                        rng,
                        self.cfg.max_rollout_plies,
                    );
                    value = value_of(result);
                    break;
                }
                Step::Decision => match self.expand(node, rng) {
                    Some(child) => {
                        node = child;
                        self.path.push(node);
                        fresh = true;
                    }
                    None => {
                        node = self.select_ucb1(node);
                        self.path.push(node);
                    }
                },
            }
        }

        self.simulations += 1;
        backpropagate(&mut self.nodes, &self.path, value);
    }
}

/// The move an ensemble of root determinizations agrees on: the action with
/// the most root visits *summed across the trees*, ties broken by the pooled
/// value from the mover's perspective.
///
/// # Why visit counts
///
/// Summing visits is the ensemble form of the rule a single tree already
/// uses. A tree's root visit count for an action is UCT's own verdict on it —
/// the search spends visits where it thinks the value is, and the count is
/// far less noisy than the mean — so pooling counts across `N` trees asks
/// "which move did the searches collectively spend their time on", which is
/// the same question one tree answers with `1/N` of the samples per tree.
/// Pooling the *means* instead would weight a tree that barely looked at a
/// move as heavily as one that concentrated on it.
///
/// # Single-tree equivalence
///
/// With one tree this reduces, term by term, to the pre-ensemble rule: the
/// iteration order is that tree's own (shuffled) root action order, an action
/// with no expanded child is skipped, `visits` is that child's visit count,
/// the tie-break score is [`Tree::exploit`]'s flip of its mean, the
/// comparison is "strictly more visits, or equal visits and a strictly better
/// score", and the fallback when nothing was expanded is the first root
/// action. That is why [`Tree::best_action`] is written in terms of this
/// function instead of alongside it.
///
/// All trees in the ensemble are rooted at the same public position, so they
/// share a mover and a root action set (in different orders); the first
/// tree's order decides ties, which keeps the result reproducible.
pub(crate) fn best_of(trees: &[Tree]) -> Option<Action> {
    let first = trees.first()?;
    let Kind::Decision { mover, actions, .. } = &first.nodes[0].kind else {
        return None;
    };
    let mut best: Option<Action> = None;
    let mut best_visits = 0u64;
    let mut best_score = f64::NEG_INFINITY;
    for &action in actions {
        let mut visits = 0u64;
        let mut value_sum = 0.0f64;
        let mut expanded = false;
        for tree in trees {
            if let Some(child) = tree.root_child(action) {
                expanded = true;
                visits += u64::from(child.visits);
                value_sum += child.value_sum;
            }
        }
        if !expanded {
            continue;
        }
        // Pooled mean, then the one perspective flip (see `Tree::exploit`).
        let mean = if visits == 0 {
            0.5
        } else {
            value_sum / visits as f64
        };
        let score = match mover {
            Player::One => mean,
            Player::Two => 1.0 - mean,
        };
        // Most visits wins; ties break on value, which matters at the tiny
        // budgets CI uses.
        let better =
            best.is_none() || visits > best_visits || (visits == best_visits && score > best_score);
        if better {
            best = Some(action);
            best_visits = visits;
            best_score = score;
        }
    }
    best.or_else(|| actions.first().copied())
}

/// The three search functions exactly as they read before [`PriorMode`]
/// existed, kept so a test can assert that [`PriorMode::None`] is not merely
/// *intended* to change nothing.
///
/// Copied verbatim from the pre-prior `Tree::expand`, `Tree::select_ucb1` and
/// `Tree::simulate`; do not "simplify" any of them to call the new code, since
/// that is the thing they exist to check. The only edit is the one the type
/// system forces: the new `priors` field is named in the pattern that
/// constructs a decision node's children, and never read.
#[cfg(test)]
impl Tree {
    fn legacy_expand(&mut self, id: NodeId, rng: &mut StdRng) -> Option<NodeId> {
        let (state, action, slot) = match &self.nodes[id as usize].kind {
            Kind::Decision {
                actions, expanded, ..
            } => {
                if *expanded >= actions.len() {
                    return None;
                }
                (self.nodes[id as usize].state, actions[*expanded], *expanded)
            }
            _ => return None,
        };
        let child = self.child_after(state, action, rng);
        match &mut self.nodes[id as usize].kind {
            Kind::Decision {
                children, expanded, ..
            } => {
                children[slot] = child;
                *expanded = slot + 1;
            }
            _ => unreachable!("expand called on a non-decision node"),
        }
        Some(child)
    }

    fn legacy_select_ucb1(&self, id: NodeId) -> NodeId {
        let (mover, children) = match &self.nodes[id as usize].kind {
            Kind::Decision {
                mover, children, ..
            } => (*mover, children),
            _ => unreachable!("select_ucb1 called on a non-decision node"),
        };
        let parent_visits = self.nodes[id as usize].visits.max(1);
        let mut best = NO_NODE;
        let mut best_score = f64::NEG_INFINITY;
        for &child in children {
            if child == NO_NODE {
                continue;
            }
            let score = ucb1(
                self.exploit(child, mover),
                self.nodes[child as usize].visits,
                parent_visits,
                self.cfg.exploration,
            );
            if score > best_score {
                best_score = score;
                best = child;
            }
        }
        debug_assert_ne!(best, NO_NODE);
        best
    }

    pub(crate) fn legacy_simulate(&mut self, rng: &mut StdRng) {
        self.path.clear();
        let mut node: NodeId = 0;
        self.path.push(node);
        let mut fresh = false;
        let value;

        loop {
            let step = match &self.nodes[node as usize].kind {
                Kind::Terminal { value } => Step::Terminal(*value),
                Kind::Chance { .. } => Step::Chance,
                Kind::Decision { .. } => Step::Decision,
            };
            match step {
                Step::Terminal(v) => {
                    value = v;
                    break;
                }
                Step::Chance => {
                    node = self.resolve_chance(node, rng);
                    self.path.push(node);
                }
                Step::Decision if fresh => {
                    let mut state = self.nodes[node as usize].state;
                    let result = rollout::play_out(
                        &mut state,
                        &self.cfg.rollout,
                        &mut self.buf,
                        rng,
                        self.cfg.max_rollout_plies,
                    );
                    value = value_of(result);
                    break;
                }
                Step::Decision => match self.legacy_expand(node, rng) {
                    Some(child) => {
                        node = child;
                        self.path.push(node);
                        fresh = true;
                    }
                    None => {
                        node = self.legacy_select_ucb1(node);
                        self.path.push(node);
                    }
                },
            }
        }

        self.simulations += 1;
        backpropagate(&mut self.nodes, &self.path, value);
    }
}

/// The move-selection rule exactly as it read before [`best_of`] existed, kept
/// so a test can assert the refactor is not merely equivalent in principle.
///
/// Copied verbatim from the pre-ensemble `Tree::best_action`; do not
/// "simplify" it to call the new code, since that is the thing it exists to
/// check.
#[cfg(test)]
pub(crate) fn legacy_best_action(tree: &Tree) -> Option<Action> {
    let Kind::Decision {
        mover,
        actions,
        children,
        ..
    } = &tree.nodes[0].kind
    else {
        return None;
    };
    let mut best: Option<Action> = None;
    let mut best_visits = 0u32;
    let mut best_score = f64::NEG_INFINITY;
    for (i, &child) in children.iter().enumerate() {
        if child == NO_NODE {
            continue;
        }
        let visits = tree.nodes[child as usize].visits;
        let score = tree.exploit(child, *mover);
        let better =
            best.is_none() || visits > best_visits || (visits == best_visits && score > best_score);
        if better {
            best = Some(actions[i]);
            best_visits = visits;
            best_score = score;
        }
    }
    best.or_else(|| actions.first().copied())
}

/// What the next step of a simulation should do at the current node.
enum Step {
    Terminal(f64),
    Chance,
    Decision,
}

/// A decision node for `state` over `actions`, shuffled once so that
/// expansion order carries no systematic bias.
///
/// The shuffle happens for every [`PriorMode`], and consumes the same
/// randomness in every one: under a prior mode it becomes the tie-break among
/// equally-rated moves (see [`Tree::rank_by_prior`]), which is what keeps a
/// prior that says nothing from silently reintroducing an ordering bias.
fn decision_node(state: GameState, mut actions: Vec<Action>, rng: &mut StdRng) -> Node {
    actions.shuffle(rng);
    let children = vec![NO_NODE; actions.len()];
    Node {
        state,
        visits: 0,
        value_sum: 0.0,
        kind: Kind::Decision {
            mover: state.current_player(),
            actions,
            children,
            expanded: 0,
            priors: Vec::new(),
        },
    }
}

/// Add one visit and `value` (Player One's perspective) to every node on
/// `path`, decision and chance alike.
///
/// Free function so a test can drive it against a hand-built arena.
pub(crate) fn backpropagate(nodes: &mut [Node], path: &[NodeId], value: f64) {
    for &id in path {
        let n = &mut nodes[id as usize];
        n.visits += 1;
        n.value_sum += value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn terminal(value: f64) -> Node {
        Node {
            state: engine::new_game(0),
            visits: 0,
            value_sum: 0.0,
            kind: Kind::Terminal { value },
        }
    }

    fn decision(mover: Player, n: usize) -> Node {
        let mut node = terminal(0.0);
        node.kind = Kind::Decision {
            mover,
            actions: vec![Action::Discard { slot: 0 }; n],
            children: vec![NO_NODE; n],
            expanded: 0,
            priors: Vec::new(),
        };
        node
    }

    /// Hand-computed UCB1: mean 0.5, 3 child visits, 10 parent visits, c = 1
    /// gives 0.5 + sqrt(ln 10 / 3) = 0.5 + sqrt(0.7675284) = 1.3760872.
    #[test]
    fn ucb1_matches_a_hand_computed_case() {
        let got = ucb1(0.5, 3, 10, 1.0);
        assert!(
            (got - 1.376_087_2).abs() < 1e-6,
            "ucb1 = {got}, expected 1.3760872"
        );
        // The exploration constant scales only the bonus.
        let doubled = ucb1(0.5, 3, 10, 2.0);
        assert!((doubled - (0.5 + 2.0 * (10f64.ln() / 3.0).sqrt())).abs() < 1e-12);
        // A never-visited child is always preferred.
        assert_eq!(ucb1(0.0, 0, 100, 0.0), f64::INFINITY);
        // Zero exploration reduces to greedy exploitation.
        assert_eq!(ucb1(0.7, 5, 50, 0.0), 0.7);
    }

    /// UCB1 must pick the child that maximises the formula, and it must do so
    /// from the perspective of the player to move at that node.
    #[test]
    fn ucb1_selection_picks_the_right_child_for_each_mover() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut tree = Tree::new(
            engine::new_game(0),
            vec![Action::Discard { slot: 0 }],
            Config {
                exploration: 1.0,
                ..Config::default()
            },
            &mut rng,
        );
        tree.nodes.clear();
        tree.nodes.push(decision(Player::One, 3));
        // Three children with means (from P1's view) 0.8, 0.5, 0.2.
        for (visits, sum) in [(10u32, 8.0f64), (10, 5.0), (10, 2.0)] {
            let mut n = terminal(0.0);
            n.visits = visits;
            n.value_sum = sum;
            let id = tree.push(n);
            if let Kind::Decision {
                children, expanded, ..
            } = &mut tree.nodes[0].kind
            {
                children[*expanded] = id;
                *expanded += 1;
            }
        }
        tree.nodes[0].visits = 30;

        // Equal visit counts, so the bonus is equal and exploitation decides.
        assert_eq!(tree.select_ucb1(0), 1, "Player One should prefer mean 0.8");

        // Same statistics, opposite mover: the flip must reverse the ranking.
        if let Kind::Decision { mover, .. } = &mut tree.nodes[0].kind {
            *mover = Player::Two;
        }
        assert_eq!(tree.select_ucb1(0), 3, "Player Two should prefer mean 0.2");

        // Now make the middle child rare: the exploration bonus must win.
        tree.nodes[2].visits = 1;
        tree.nodes[2].value_sum = 0.5;
        if let Kind::Decision { mover, .. } = &mut tree.nodes[0].kind {
            *mover = Player::One;
        }
        // child1: 0.8 + sqrt(ln30/10) = 0.8 + 0.5832 = 1.3832
        // child2: 0.5 + sqrt(ln30/1)  = 0.5 + 1.8443 = 2.3443  <- best
        // child3: 0.2 + sqrt(ln30/10) = 0.2 + 0.5832 = 0.7832
        assert_eq!(tree.select_ucb1(0), 2, "the rare child must be explored");
    }

    /// A tiny synthetic tree: root (P1 to move) -> chance -> decision (P2 to
    /// move) -> terminal. Backpropagation stores one global-perspective
    /// value on every node, and the flip shows up only in `exploit`.
    #[test]
    fn backpropagation_stores_exact_values_and_flips_only_at_selection() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut tree = Tree::new(
            engine::new_game(0),
            vec![Action::Discard { slot: 0 }],
            Config::default(),
            &mut rng,
        );
        tree.nodes.clear();
        tree.nodes.push(decision(Player::One, 1)); // 0: root, P1 to move
        let mut chance = terminal(0.0);
        chance.kind = Kind::Chance {
            action: Action::Discard { slot: 0 },
            children: Vec::new(),
        };
        tree.nodes.push(chance); // 1
        tree.nodes.push(decision(Player::Two, 1)); // 2, P2 to move
        tree.nodes.push(terminal(1.0)); // 3

        let path = [0u32, 1, 2, 3];
        // Three Player One wins and one Player Two win. 3/4 is exact in
        // binary, so every assertion below can be an equality.
        backpropagate(&mut tree.nodes, &path, 1.0);
        backpropagate(&mut tree.nodes, &path, 1.0);
        backpropagate(&mut tree.nodes, &path, 1.0);
        backpropagate(&mut tree.nodes, &path, 0.0);

        // Chance and decision nodes alike carry the same global-perspective
        // total: backpropagation does no flipping.
        for id in path {
            let n = &tree.nodes[id as usize];
            assert_eq!(n.visits, 4, "node {id}");
            assert_eq!(n.value_sum, 3.0, "node {id}");
            assert_eq!(n.mean(), 0.75, "node {id}");
        }

        // The root (P1 to move) sees its chance child as 0.75; the P2
        // decision node sees its own child as 0.25. Exactly.
        assert_eq!(tree.exploit(1, Player::One), 0.75);
        assert_eq!(tree.exploit(3, Player::Two), 0.25);
        assert_eq!(
            tree.exploit(3, Player::One) + tree.exploit(3, Player::Two),
            1.0,
            "the flip must be the exact zero-sum complement"
        );

        // A draw is worth exactly half to each side, and the complement
        // property survives an inexact mean.
        backpropagate(&mut tree.nodes, &[3], 0.5);
        assert_eq!(tree.nodes[3].value_sum, 3.5);
        assert_eq!(tree.nodes[3].visits, 5);
        let mean = 3.5 / 5.0;
        assert_eq!(tree.exploit(3, Player::One), mean);
        assert_eq!(tree.exploit(3, Player::Two), 1.0 - mean);

        // An all-draws node is 0.5 for whoever is to move.
        let d = tree.push(terminal(0.5));
        backpropagate(&mut tree.nodes, &[d], 0.5);
        backpropagate(&mut tree.nodes, &[d], 0.5);
        assert_eq!(tree.exploit(d, Player::One), 0.5);
        assert_eq!(tree.exploit(d, Player::Two), 0.5);
    }

    #[test]
    fn terminal_values_follow_the_player_one_convention() {
        use duels_core::scoring::VictoryKind;
        assert_eq!(
            value_of(GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::CivilianVictory
            }),
            1.0
        );
        assert_eq!(
            value_of(GameResult::Win {
                winner: Player::Two,
                kind: VictoryKind::MilitarySupremacy
            }),
            0.0
        );
        assert_eq!(value_of(GameResult::Draw), 0.5);
    }

    /// `best_of` over one tree must be the pre-ensemble rule, term for term,
    /// on real search trees rather than only in the argument for it.
    #[test]
    fn best_of_one_tree_is_the_pre_ensemble_rule() {
        for seed in 0..12u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let state = engine::new_game(seed);
            let actions = engine::legal_actions(&state);
            for sims in [1usize, 3, 17, 200] {
                let mut tree = Tree::new(state, actions.clone(), Config::default(), &mut rng);
                for _ in 0..sims {
                    tree.simulate(&mut rng);
                }
                assert_eq!(
                    best_of(std::slice::from_ref(&tree)),
                    legacy_best_action(&tree),
                    "seed {seed}, {sims} simulations"
                );
            }
        }
    }

    /// The ensemble rule pools visits: a move two trees each visited a little
    /// beats a move one tree visited more.
    #[test]
    fn best_of_sums_visits_across_trees() {
        let a = Action::Discard { slot: 0 };
        let b = Action::Discard { slot: 1 };
        // Two hand-built trees over the same two root actions, in *different*
        // orders, so the lookup is by action and not by index.
        let build = |order: [Action; 2], stats: [(u32, f64); 2]| {
            let mut rng = StdRng::seed_from_u64(1);
            let mut tree = Tree::new(
                engine::new_game(0),
                vec![Action::Discard { slot: 0 }],
                Config::default(),
                &mut rng,
            );
            tree.nodes.clear();
            let mut root = terminal(0.0);
            root.kind = Kind::Decision {
                mover: Player::One,
                actions: order.to_vec(),
                children: vec![NO_NODE; 2],
                expanded: 2,
                priors: Vec::new(),
            };
            tree.nodes.push(root);
            for (slot, (visits, value_sum)) in stats.into_iter().enumerate() {
                let mut n = terminal(0.0);
                n.visits = visits;
                n.value_sum = value_sum;
                let id = tree.push(n);
                if let Kind::Decision { children, .. } = &mut tree.nodes[0].kind {
                    children[slot] = id;
                }
            }
            tree
        };
        // Tree one visits `a` 10 times, `b` 4. Tree two (reversed order)
        // visits `b` 9 times, `a` 2. Pooled: a = 12, b = 13.
        let t1 = build([a, b], [(10, 5.0), (4, 2.0)]);
        let t2 = build([b, a], [(9, 4.0), (2, 1.0)]);
        assert_eq!(best_of(std::slice::from_ref(&t1)), Some(a));
        assert_eq!(best_of(std::slice::from_ref(&t2)), Some(b));
        assert_eq!(best_of(&[t1, t2]), Some(b), "pooled visits favour b");

        // An action no tree expanded is skipped, not returned with 0 visits.
        let mut lonely = build([a, b], [(3, 1.0), (5, 2.0)]);
        if let Kind::Decision { children, .. } = &mut lonely.nodes[0].kind {
            children[1] = NO_NODE;
        }
        assert_eq!(best_of(std::slice::from_ref(&lonely)), Some(a));
    }

    /// The strongest form of the [`PriorMode::None`] guarantee: not "the same
    /// move" but *the same tree*, node for node, against the verbatim
    /// pre-prior `simulate`.
    ///
    /// Both trees are grown from the same seed over the same position, so
    /// every node's kind, visit count, value sum, action order and child
    /// wiring has to agree exactly — which also proves the prior path
    /// consumes no randomness, since a single extra RNG draw would desynchronise
    /// every chance node below it.
    #[test]
    fn prior_none_grows_the_same_tree_as_the_pre_prior_search() {
        for seed in 0..10u64 {
            let state = engine::new_game(seed);
            let actions = engine::legal_actions(&state);
            let cfg = Config {
                prior: PriorMode::None,
                ..Config::default()
            };

            let mut rng_new = StdRng::seed_from_u64(seed ^ 0xABCD);
            let mut new = Tree::new(state, actions.clone(), cfg, &mut rng_new);
            let mut rng_old = StdRng::seed_from_u64(seed ^ 0xABCD);
            let mut old = Tree::new(state, actions.clone(), cfg, &mut rng_old);

            for _ in 0..500 {
                new.simulate(&mut rng_new);
                old.legacy_simulate(&mut rng_old);
            }

            assert_eq!(new.nodes.len(), old.nodes.len(), "seed {seed}: tree size");
            assert_eq!(new.simulations, old.simulations);
            for (i, (a, b)) in new.nodes.iter().zip(old.nodes.iter()).enumerate() {
                assert_eq!(a.visits, b.visits, "seed {seed}, node {i}: visits");
                assert_eq!(a.value_sum, b.value_sum, "seed {seed}, node {i}: value");
                match (&a.kind, &b.kind) {
                    (Kind::Terminal { value: x }, Kind::Terminal { value: y }) => {
                        assert_eq!(x, y, "seed {seed}, node {i}")
                    }
                    (
                        Kind::Decision {
                            mover: m1,
                            actions: a1,
                            children: c1,
                            expanded: e1,
                            ..
                        },
                        Kind::Decision {
                            mover: m2,
                            actions: a2,
                            children: c2,
                            expanded: e2,
                            ..
                        },
                    ) => {
                        assert_eq!(m1, m2, "seed {seed}, node {i}: mover");
                        assert_eq!(a1, a2, "seed {seed}, node {i}: action order");
                        assert_eq!(c1, c2, "seed {seed}, node {i}: children");
                        assert_eq!(e1, e2, "seed {seed}, node {i}: expanded");
                    }
                    (
                        Kind::Chance {
                            action: x1,
                            children: k1,
                        },
                        Kind::Chance {
                            action: x2,
                            children: k2,
                        },
                    ) => {
                        assert_eq!(x1, x2, "seed {seed}, node {i}: chance action");
                        assert_eq!(k1.len(), k2.len(), "seed {seed}, node {i}: outcomes");
                        for (u, v) in k1.iter().zip(k2.iter()) {
                            assert_eq!(u.outcome, v.outcome);
                            assert_eq!(u.prob, v.prob);
                            assert_eq!(u.node, v.node);
                        }
                    }
                    _ => panic!("seed {seed}, node {i}: different node kinds"),
                }
            }
        }
    }

    /// A real mid-game turn with a full slate of legal moves, so a ranking
    /// test is about a branchy position rather than the four-way wonder draft.
    fn mid_game(seed: u64) -> (GameState, Vec<Action>) {
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x4242);
        for step in 0.. {
            let legal = engine::legal_actions(&state);
            assert!(!legal.is_empty(), "seed {seed} ended before a branchy turn");
            if step >= 12 && legal.len() >= 6 {
                return (state, legal);
            }
            let a = legal[rng.gen_range(0..legal.len())];
            engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action");
        }
        unreachable!()
    }

    /// A prior mode reorders a node's actions into descending prior, and does
    /// it exactly once — on the first expansion, when no child exists yet.
    #[test]
    fn a_prior_mode_orders_a_node_by_descending_prior_once() {
        let mut reordered = 0u32;
        for seed in 0..8u64 {
            let (state, actions) = mid_game(seed);
            let mut rng = StdRng::seed_from_u64(seed);
            let mut tree = Tree::new(
                state,
                actions.clone(),
                Config {
                    prior: PriorMode::ExpansionOrder,
                    ..Config::default()
                },
                &mut rng,
            );

            // Before any expansion the order is the shuffle's.
            let shuffled = match &tree.nodes[0].kind {
                Kind::Decision { actions, .. } => actions.clone(),
                _ => panic!("the root is a decision"),
            };

            tree.simulate(&mut rng);
            let ranked = match &tree.nodes[0].kind {
                Kind::Decision {
                    actions, expanded, ..
                } => {
                    assert_eq!(*expanded, 1, "one child after one simulation");
                    actions.clone()
                }
                _ => panic!("the root is a decision"),
            };
            if ranked != shuffled {
                reordered += 1;
            }

            // The order is exactly `action_prior` descending, over exactly the
            // same set of moves.
            let s = duels_strategy::stance(&state, state.current_player());
            let weights: Vec<f64> = ranked
                .iter()
                .map(|&a| duels_strategy::action_prior(&state, a, &s))
                .collect();
            for w in weights.windows(2) {
                assert!(w[0] >= w[1], "seed {seed} not descending: {weights:?}");
            }
            assert_eq!(
                ranked.len(),
                shuffled.len(),
                "the ranking changed the count"
            );
            for a in &shuffled {
                assert!(ranked.contains(a), "the ranking lost {a:?}");
            }

            // And it is not redone: run the root out past fully expanded, then
            // check the order still matches the first ranking.
            for _ in 0..(ranked.len() as u32 + 20) {
                tree.simulate(&mut rng);
            }
            match &tree.nodes[0].kind {
                Kind::Decision {
                    actions, expanded, ..
                } => {
                    assert_eq!(*actions, ranked, "seed {seed}: the node was re-ranked");
                    assert_eq!(*expanded, ranked.len(), "seed {seed}: root never filled");
                }
                _ => panic!("the root is a decision"),
            }
        }
        assert!(
            reordered >= 4,
            "the ranking moved nothing in {} of 8 positions; it is not doing anything",
            8 - reordered
        );
    }

    /// The cost claim [`PriorMode`] is designed around, asserted rather than
    /// argued: the strategy layer is consulted **exactly once per expanded
    /// decision node that had a choice to make** — not once per simulation,
    /// not once per node created, and never twice for the same node.
    #[test]
    fn the_prior_is_computed_once_per_expanded_node_not_per_simulation() {
        for prior in [
            PriorMode::ExpansionOrder,
            PriorMode::ProgressiveBias { weight: 5.0 },
        ] {
            let (state, actions) = mid_game(3);
            let mut rng = StdRng::seed_from_u64(3);
            let mut tree = Tree::new(
                state,
                actions,
                Config {
                    prior,
                    ..Config::default()
                },
                &mut rng,
            );
            const SIMS: u64 = 2_000;
            for _ in 0..SIMS {
                tree.simulate(&mut rng);
            }

            // The set that should have paid, counted independently of the
            // counter by walking the finished arena.
            let expected = tree
                .nodes
                .iter()
                .filter(|n| {
                    matches!(&n.kind, Kind::Decision { expanded, actions, .. }
                        if *expanded > 0 && actions.len() > 1)
                })
                .count() as u64;
            assert_eq!(
                tree.rankings, expected,
                "{prior:?}: {} stance computations for {expected} expanded nodes",
                tree.rankings
            );
            assert_eq!(tree.simulations, SIMS);
            // The whole point of paying per node: it must be a small fraction
            // of the simulations, or the cost model in `PriorMode` is wrong.
            assert!(
                tree.rankings * 2 < SIMS,
                "{prior:?}: {} rankings against {SIMS} simulations is not \
                 'once per node'",
                tree.rankings
            );
            println!(
                "{prior:?}: {} rankings / {SIMS} simulations over {} nodes",
                tree.rankings,
                tree.nodes.len()
            );
        }
    }

    /// `ProgressiveBias` keeps a normalised prior per child; the ordering
    /// modes keep nothing, which is what makes them free per node after the
    /// first expansion.
    #[test]
    fn only_progressive_bias_retains_the_prior_slate() {
        let state = engine::new_game(6);
        let actions = engine::legal_actions(&state);
        for (prior, keeps) in [
            (PriorMode::None, false),
            (PriorMode::ExpansionOrder, false),
            (PriorMode::ProgressiveBias { weight: 2.0 }, true),
        ] {
            let mut rng = StdRng::seed_from_u64(6);
            let mut tree = Tree::new(
                state,
                actions.clone(),
                Config {
                    prior,
                    ..Config::default()
                },
                &mut rng,
            );
            tree.simulate(&mut rng);
            match &tree.nodes[0].kind {
                Kind::Decision {
                    priors, actions, ..
                } => {
                    if keeps {
                        assert_eq!(priors.len(), actions.len(), "{prior:?}");
                        let total: f32 = priors.iter().sum();
                        assert!((total - 1.0).abs() < 1e-4, "{prior:?} sum {total}");
                        assert!(priors.iter().all(|&p| p > 0.0), "{prior:?}");
                    } else {
                        assert!(priors.is_empty(), "{prior:?} kept {} priors", priors.len());
                    }
                }
                _ => panic!("the root is a decision"),
            }
        }
    }

    /// The progressive-bias term is exactly `weight * prior / (visits + 1)` on
    /// top of UCB1, and it decays: with equal statistics it picks the
    /// highest-prior child, and a large visit count washes it out.
    #[test]
    fn progressive_bias_adds_a_decaying_term_to_ucb1() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut tree = Tree::new(
            engine::new_game(0),
            vec![Action::Discard { slot: 0 }],
            Config {
                prior: PriorMode::ProgressiveBias { weight: 10.0 },
                ..Config::default()
            },
            &mut rng,
        );
        tree.nodes.clear();
        tree.nodes.push(decision(Player::One, 3));
        for _ in 0..3 {
            let mut n = terminal(0.0);
            n.visits = 9;
            n.value_sum = 4.5; // every child a dead-even 0.5
            let id = tree.push(n);
            if let Kind::Decision {
                children, expanded, ..
            } = &mut tree.nodes[0].kind
            {
                children[*expanded] = id;
                *expanded += 1;
            }
        }
        tree.nodes[0].visits = 27;
        if let Kind::Decision { priors, .. } = &mut tree.nodes[0].kind {
            *priors = vec![0.2, 0.5, 0.3];
        }
        // Identical exploitation and identical bonuses, so only the bias term
        // separates them: 10 * 0.5 / 10 for child 2 is the largest.
        assert_eq!(tree.select_ucb1(0), 2);

        // Same priors, but child 1 is now much better on the statistics that
        // matter: 0.9 vs 0.5 dwarfs a bias term divided by 10.
        tree.nodes[1].value_sum = 8.1;
        assert_eq!(tree.select_ucb1(0), 1);

        // With the weight at zero it is plain UCB1 again, and the shipped
        // `None` mode ignores the slate entirely even when one is present.
        tree.nodes[1].value_sum = 4.5;
        tree.cfg.prior = PriorMode::ProgressiveBias { weight: 0.0 };
        let unbiased = tree.select_ucb1(0);
        tree.cfg.prior = PriorMode::None;
        assert_eq!(tree.select_ucb1(0), unbiased);
        assert_eq!(unbiased, 1, "ties go to the first child scanned");
    }

    /// Chance nodes must be resolved by probability, not by UCB1: with
    /// widening switched off the children a chance node re-selects should
    /// follow the outcome probabilities it recorded.
    #[test]
    fn chance_reselection_follows_the_recorded_probabilities() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut tree = Tree::new(
            engine::new_game(0),
            vec![Action::Discard { slot: 0 }],
            Config {
                // No widening at all: always re-select an existing child.
                chance_widen_c: 0.0,
                chance_widen_alpha: 0.0,
                ..Config::default()
            },
            &mut rng,
        );
        tree.nodes.clear();
        let mut chance = terminal(0.0);
        chance.kind = Kind::Chance {
            action: Action::Discard { slot: 0 },
            children: Vec::new(),
        };
        tree.nodes.push(chance);
        let a = tree.push(terminal(1.0));
        let b = tree.push(terminal(0.0));
        if let Kind::Chance { children, .. } = &mut tree.nodes[0].kind {
            children.push(ChanceChild {
                outcome: Outcome::default(),
                prob: 0.25,
                node: a,
            });
            children.push(ChanceChild {
                outcome: Outcome {
                    library_tokens: None,
                    reveals: [Some((0, duels_core::data::CardId::from_index(0))), None],
                },
                prob: 0.75,
                node: b,
            });
        }

        let mut hits_a = 0u32;
        const N: u32 = 40_000;
        for _ in 0..N {
            if tree.resolve_chance(0, &mut rng) == a {
                hits_a += 1;
            }
        }
        let share = f64::from(hits_a) / f64::from(N);
        assert!(
            (share - 0.25).abs() < 0.01,
            "chance re-selection gave {share}, expected 0.25"
        );
    }
}
