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
        }
    }
}

impl Config {
    /// A compact, stable description for [`duels_agents_api::AgentSpec`].
    pub fn describe(&self) -> String {
        let w = &self.rollout;
        format!(
            "c={:.3};rollout=weights(build={},wonder={},discard={});chance=progressive-widening(c={:.2},alpha={:.2})",
            self.exploration, w.build, w.wonder, w.discard, self.chance_widen_c, self.chance_widen_alpha
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
        };
        let root = decision_node(state, actions, rng);
        tree.nodes.push(root);
        tree
    }

    /// The root's most-visited child: the standard robust move-selection
    /// rule, preferred over highest mean because a child with few visits has
    /// a noisy mean.
    pub fn best_action(&self) -> Option<Action> {
        let Kind::Decision {
            mover,
            actions,
            children,
            ..
        } = &self.nodes[0].kind
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
            let visits = self.nodes[child as usize].visits;
            let score = self.exploit(child, *mover);
            // Most visits wins; ties break on value, which matters at the
            // tiny budgets CI uses.
            let better = best.is_none()
                || visits > best_visits
                || (visits == best_visits && score > best_score);
            if better {
                best = Some(actions[i]);
                best_visits = visits;
                best_score = score;
            }
        }
        best.or_else(|| actions.first().copied())
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

    /// Create the next unexpanded child of decision node `id`, or `None` if
    /// they are all expanded.
    fn expand(&mut self, id: NodeId, rng: &mut StdRng) -> Option<NodeId> {
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
    /// decision node.
    fn select_ucb1(&self, id: NodeId) -> NodeId {
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

/// What the next step of a simulation should do at the current node.
enum Step {
    Terminal(f64),
    Chance,
    Decision,
}

/// A decision node for `state` over `actions`, shuffled once so that
/// expansion order carries no systematic bias.
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
