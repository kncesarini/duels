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
use duels_core::scoring::VictoryKind;
use duels_core::{Action, GameResult, GameState, Player};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::chance;
use crate::leaf::{self, LeafValue};
use crate::rollout::{self, RaceWeights, RolloutWeights};

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

/// What a search is rewarded for winning: any victory at all, or one
/// specific [`VictoryKind`].
///
/// # Why this exists
///
/// Every agent on this ladder, this crate's default included, is trained and
/// searched to maximise **any** win — the raw, heavily civilian-skewed outcome
/// distribution `duels-value`'s crate docs measure (civilian 80.64%, military
/// 15.43%, science 2.31% in the corpus that shaped `v1`). `Objective` is what
/// lets the *same* leaf model (`v2.bin`, retrained on nothing new) be searched
/// under a different reward instead: "did **this** specific victory kind
/// happen", so the resulting agent's play can be read as what maximally
/// specialised play toward one strategy looks like, and used as a genuinely
/// distinct sparring partner — not a stronger generalist. See
/// `duels_value`'s crate docs, "What's next: beyond mixing corpora", for the
/// research question this was built to answer, and this crate's own docs for
/// the purity measurements it produced.
///
/// **This is not a strength-seeking objective**, and a specialist built from
/// it is not expected to out-Elo [`Config::default`] — see this crate's docs
/// for why that is the wrong question to ask of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Reward `1.0` for a Player One win of any kind, `0.0` for a Player Two
    /// win, `0.5` for a draw -- [`value_of`], unchanged. [`Config::default`]'s
    /// objective, and the only one every other agent on this ladder has.
    WinProbability,
    /// Reward **only** a win of exactly this [`VictoryKind`] **by the seat
    /// this tree is actually searching for** ([`Tree::me`]): `1.0` if `me`
    /// won by `kind`, `0.0` for everything else -- the other seat winning
    /// (including a win of `kind` itself), a draw, or `me` winning by a
    /// *different* kind. See [`objective_value_of`]'s docs for the exact rule
    /// and [`Tree::exploit`]'s docs for why this is anchored to `me` rather
    /// than to a game-fixed `Player::One`: unlike ordinary win/loss, "One
    /// wins by kind K" and "Two wins by kind K" are not complementary, so a
    /// fixed anchor would reward the wrong player's achievement whenever this
    /// tree is searching for Player Two.
    ///
    /// [`duels_value::Outcome::of`] already folds
    /// [`VictoryKind::CivilianVictory`] and [`VictoryKind::CivilianTiebreak`]
    /// into one class, on the reasoning that a civilian win is a civilian win
    /// whether or not it needed the tiebreak; [`objective_value_of`] makes the
    /// identical call, so passing either civilian variant as `kind` matches
    /// both.
    TargetKind(VictoryKind),
}

impl Objective {
    /// A compact, stable description for [`Config::describe`]. `None` for the
    /// default, so [`Config::default`]'s params string -- and therefore every
    /// existing results file's -- stays byte-for-byte what it was before this
    /// field existed.
    fn describe(&self) -> Option<String> {
        match self {
            Objective::WinProbability => None,
            Objective::TargetKind(kind) => Some(format!("target({})", victory_kind_name(*kind))),
        }
    }
}

/// A short, stable name for a [`VictoryKind`], since the type itself has no
/// `Display`/name method and [`Objective::describe`] needs one that will not
/// silently change if `Debug`'s derive output ever does.
const fn victory_kind_name(kind: VictoryKind) -> &'static str {
    match kind {
        VictoryKind::MilitarySupremacy => "military",
        VictoryKind::ScientificSupremacy => "science",
        VictoryKind::CivilianVictory | VictoryKind::CivilianTiebreak => "civilian",
    }
}

/// Tuning knobs for the search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// UCB1 exploration constant `c`, against rewards in `[0, 1]`.
    pub exploration: f64,
    /// Playout policy weights.
    pub rollout: RolloutWeights,
    /// Race-progress multipliers layered on top of [`Config::rollout`].
    /// [`RaceWeights::NEUTRAL`] is the default and is bit-for-bit an unbiased
    /// (by race progress) playout policy; [`RaceWeights::TIER1_ONLY`]'s
    /// terminal rails measured `+26` Elo in `mcts-uct` and **add** to this
    /// crate's leaf value rather than overlapping with it (see the crate
    /// docs' composition table).
    pub race: RaceWeights,
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
    /// visit counts to pick the move. `1`, the default, is plain
    /// single-determinization search; see the crate docs for what larger
    /// values measure (nothing, at these budgets).
    pub root_determinizations: usize,
    /// Whether, and how, [`duels_strategy`]'s policy layer steers the tree.
    /// [`PriorMode::None`] is the default, and the alternatives measured
    /// neutral-to-negative in `mcts-uct`; kept for the same reason they are
    /// kept there.
    pub prior: PriorMode,
    /// What a freshly added leaf is worth.
    ///
    /// [`LeafValue::Learned`] is the default and is what this crate is *for*.
    /// The rest of the family is inherited from `mcts-eval` whole, because two
    /// of its members are this crate's ablation controls:
    /// [`LeafValue::Blend`] at `weight = 0.5` with `exploration: 0.5` is
    /// [`Config::eval_base`] (`mcts-eval`), and [`LeafValue::Rollout`] with
    /// `exploration: 1.0` is [`Config::rollout_base`] (`mcts-uct`).
    /// [`LeafValue::LearnedBlend`] is the measured second-best option here —
    /// `+106.1` Elo at a node budget against `mcts-eval` but only `+68.5` at a
    /// wall clock, where this crate's playout-free default takes `+140.6`.
    /// [`LeafValue::Static`] and [`LeafValue::Truncated`] are kept as measured,
    /// documented alternatives.
    pub leaf: LeafValue,
    /// Pin the *inherited hand-crafted* leaves to a specific frozen
    /// `duels-eval` generation instead of reading
    /// [`duels_eval::Config::default`] live.
    ///
    /// **Read by nothing on this crate's default path**, which scores its
    /// leaves with [`duels_value`] and builds no [`duels_eval::Root`] at all.
    /// It is inherited from `mcts-eval` along with those leaves, and it matters
    /// here for one reason: [`Config::eval_base`] leaves it `None`, so the
    /// ablation control really is `mcts-eval` as shipped — live-tracking — and
    /// not a snapshot a later `duels-eval` round would leave behind.
    ///
    /// As on `mcts-eval`, setting it is an A/B-testing device rather than a
    /// production configuration: it lets a `duels-eval` change be measured in
    /// one binary, one process, one `duels-arena match`, by freezing today's
    /// default as the next `Config::vN()` snapshot and matching the live arm
    /// against `mcts-value:base=eval,eval=vN`.
    pub eval_override: Option<duels_eval::Config>,
    /// Which accumulation order the learned leaves' forward pass uses — this
    /// crate's default leaf included, so unlike [`Config::eval_override`] this
    /// field is squarely on the default path.
    ///
    /// The four-way unroll is the default. It is worth `1.41x` on the forward
    /// pass and measured Elo-neutral (`-1.7 [-25.8, +22.3]` over 798 games),
    /// but it *reassociates a floating-point sum*, so it is not bit-identical
    /// to the arithmetic some of the crate docs' earlier rows were taken with.
    /// That is exactly why `duels_value::Summation::Serial` stays reachable
    /// rather than being deleted (`mcts-value:value_sum=serial`) and why
    /// [`Config::describe`] records which order ran.
    pub value_summation: duels_value::Summation,
    /// Pin the learned leaf to a specific frozen `duels-value` weights
    /// generation instead of `duels_value::default_net()`.
    ///
    /// `None` (the default) reads the live embedded weights, same as every
    /// other field on this struct that is not itself an ablation control.
    /// `Some(bytes)` is an A/B-testing device with the identical purpose as
    /// [`Config::eval_override`] one field up: it lets a `duels-value` retrain
    /// be measured against its predecessor in one binary, one process, one
    /// `duels-arena match`, rather than requiring two separately-built
    /// binaries. `crate::WEIGHTS_V1` is the frozen copy this exists for.
    pub value_weights_override: Option<&'static [u8]>,
    /// What the search is rewarded for winning: any victory
    /// ([`Objective::WinProbability`], the default) or one specific
    /// [`VictoryKind`] ([`Objective::TargetKind`]).
    ///
    /// This is the one field on this struct that changes *what the search is
    /// for* rather than *how well it plays toward the usual goal*: it is read
    /// by both [`Tree::leaf_value`] (the learned leaf reads
    /// [`duels_value::Dist::p`] of the target outcome instead of
    /// [`duels_value::Dist::win_probability`]) and by every terminal/rollout
    /// result this tree backs up (`Tree::node_for`, `Tree::rollout_from`, and
    /// the `Truncated` leaf's finished-early arm all go through
    /// [`objective_value_of`] rather than [`value_of`] directly). See
    /// [`Objective`]'s own docs for why this exists and
    /// `tests::objective_win_probability_is_bit_identical_to_value_of` for the
    /// proof that leaving it at its default changes nothing.
    pub objective: Objective,
}

impl Default for Config {
    /// **The configuration this crate exists to be**: [`duels_value`]'s
    /// learned outcome model as the whole leaf value — no playout — at the
    /// exploration constant that was re-derived for it.
    ///
    /// These two fields move *together* and are not independently tunable
    /// defaults, for the same reason `mcts-eval`'s pair is not: what a leaf
    /// backs up decides what `c` means. But the *relation* is different.
    /// [`LeafValue::Blend`]'s `c = c₀·(1 - weight)` prescribes a rescaling of
    /// the Bernoulli playout's spread, and there is no playout left here to
    /// shrink, so the formula gives no guidance at all and `c` had to be swept:
    /// `0.15` is the argmax of a bracketed four-point sweep at `Nodes(32000)`
    /// (`0.10` / `0.15` / `0.25` / `0.50` measuring `+126.7` / `+140.1` /
    /// `+101.0` / `+57.2`), so read it as "somewhere in `[0.10, 0.15]`" rather
    /// than as a tuned peak. The inherited `0.5` was worth about 83 Elo less.
    ///
    /// Every other field here is `mcts-uct`'s tuned value, inherited through
    /// `mcts-eval` unchanged.
    fn default() -> Self {
        Self {
            // Swept for `leaf` below, *not* derived from it; see this
            // function's doc comment and the crate docs' `c` table.
            exploration: 0.15,
            rollout: RolloutWeights::BIASED,
            race: RaceWeights::NEUTRAL,
            chance_widen_c: 1.0,
            chance_widen_alpha: 0.5,
            max_rollout_plies: 2_000,
            time_check_interval: 64,
            root_determinizations: 1,
            prior: PriorMode::None,
            // `+91.4` Elo at `Nodes(2000)` and `+140.1` at `Nodes(32000)`
            // against `mcts-eval`'s default — and essentially level with it
            // through any third party. See the crate docs, both halves.
            leaf: LeafValue::Learned,
            eval_override: None,
            value_summation: duels_value::Summation::default(),
            value_weights_override: None,
            objective: Objective::WinProbability,
        }
    }
}

impl Config {
    /// The configuration that turns this agent back into **`mcts-eval`**: the
    /// half-playout, half-[`duels_eval`] blend leaf at that crate's rescaled
    /// `c = 0.5`.
    ///
    /// This is the control arm every strength claim in the crate docs is
    /// measured against, so it is one spec string away
    /// (`mcts-value:base=eval`) and provable in one binary —
    /// `tests::the_copied_search_is_the_mcts_eval_search_node_for_node` and
    /// `crate::tests::the_eval_base_is_the_mcts_eval_agent_move_for_move` are
    /// that proof, against a verbatim frozen copy of that agent's search.
    ///
    /// [`Config::eval_override`] is deliberately left `None`, which means this
    /// control tracks `duels_eval::Config::default()` **live** exactly as
    /// `mcts-eval` itself does. That is the point: the control has to be the
    /// champion as shipped, not a snapshot of it that a later `duels-eval`
    /// round would leave behind.
    pub fn eval_base() -> Self {
        Self {
            exploration: 0.5,
            leaf: LeafValue::Blend { weight: 0.5 },
            ..Self::default()
        }
    }

    /// The configuration that turns this agent back into `mcts-uct`: a pure
    /// playout leaf at the unrescaled exploration constant.
    ///
    /// Inherited from `mcts-eval` along with the rest of the search, and kept
    /// for the same reason: it is the bottom of the ablation chain
    /// (`mcts-value:base=rollout`), and
    /// `tests::the_rollout_base_grows_the_mcts_uct_tree_node_for_node` proves
    /// it really is that agent, node for node.
    pub fn rollout_base() -> Self {
        Self {
            exploration: 1.0,
            leaf: LeafValue::Rollout,
            ..Self::default()
        }
    }

    /// The `duels_eval::Config` this search will actually score its leaves
    /// against: [`Config::eval_override`] if one is pinned, otherwise
    /// [`duels_eval::Config::default`], read live.
    pub fn eval_config(&self) -> duels_eval::Config {
        self.eval_override.unwrap_or_default()
    }

    /// A compact, stable description for [`duels_agents_api::AgentSpec`].
    ///
    /// The `value=` tail is this crate's **provenance record**, and it is the
    /// part that matters on the default path: it is
    /// [`duels_value::default_weights_id`] — the network's shape plus a
    /// content hash of the embedded weights — followed by the summation order
    /// the forward pass ran. A retrain at the same width is the same shape and
    /// completely different behaviour, so recording only `leaf=learned` would
    /// make two results files from either side of one indistinguishable. It is
    /// belt to the `golden` module's braces: the test stops a retrain
    /// landing silently, the hash makes every results file say which weights
    /// it was taken with.
    ///
    /// The `eval=` tail reads `unused` on the default path (nothing here
    /// consults the hand-crafted evaluation) and, for the inherited
    /// hand-crafted leaves, is the whole [`duels_eval::Config`] the search
    /// scored against rather than a generation label — so
    /// [`Config::eval_base`], the `mcts-eval` control, records the live
    /// evaluation it actually used.
    pub fn describe(&self) -> String {
        let w = &self.rollout;
        format!(
            "c={:.3};rollout=weights(build={},wonder={},discard={},chain_free={},new_symbol={},pair_complete={});race={};chance=progressive-widening(c={:.2},alpha={:.2});dets={};prior={};leaf={};eval={}",
            self.exploration,
            w.build,
            w.wonder,
            w.discard,
            w.chain_free_mult,
            w.new_symbol_mult,
            w.pair_complete_mult,
            self.race.name(),
            self.chance_widen_c,
            self.chance_widen_alpha,
            self.root_determinizations.max(1),
            self.prior.describe(),
            self.leaf.describe(),
            if self.leaf.needs_eval_root() {
                // Whatever `Tree::new` will actually build a `Root` with:
                // live, unless `eval_override` pins a frozen generation.
                self.eval_config().params_string()
            } else {
                "unused".to_string()
            },
        ) + &match self.leaf.needs_learned_net() {
            // Appended only for the opt-in learned leaves, so the default
            // configuration's params string — and therefore every existing
            // results file's — is byte-for-byte what it was.
            //
            // The identity is the shape plus a content hash of the embedded
            // weights, for exactly the reason the `eval=` tail above is the
            // whole `duels_eval::Config`: a retrain at the same width is the
            // same shape and completely different behaviour, so recording only
            // "learned" would make two results files indistinguishable.
            true => format!(
                ";value={}/{}",
                match self.value_weights_override {
                    Some(bytes) => duels_value::weights_id(bytes),
                    None => duels_value::default_weights_id().to_string(),
                },
                self.value_summation.name()
            ),
            false => String::new(),
        } + &match self.objective.describe() {
            // Appended only when set, for the same reason the `value=` tail
            // above is conditional: `Config::default`'s params string must
            // stay byte-for-byte what it was before this field existed.
            Some(desc) => format!(";objective={desc}"),
            None => String::new(),
        }
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
///
/// **Frozen.** This is the reward every agent on this ladder has always used,
/// and it is what the two verbatim historical copies further down this file
/// (`eval_legacy_*`, `legacy_*`) call directly rather than going through
/// [`objective_value_of`] -- they exist to reproduce `mcts-eval`/`mcts-uct`
/// node for node, and neither of those agents has an [`Objective`] to be
/// aware of. Production code calls [`objective_value_of`] instead, which
/// reproduces this function bit-for-bit at [`Objective::WinProbability`].
#[inline]
pub(crate) fn value_of(result: GameResult) -> f64 {
    match result.winner() {
        Some(Player::One) => 1.0,
        Some(Player::Two) => 0.0,
        None => 0.5,
    }
}

/// [`value_of`], generalized by [`Config::objective`]: the reward a search is
/// actually built to maximize.
///
/// At [`Objective::WinProbability`] this is [`value_of`], verbatim -- same
/// match, same three arms, same order, and `me` is ignored entirely -- which
/// is what makes
/// `tests::objective_win_probability_is_bit_identical_to_value_of` a real
/// check rather than a tautology the type system already guarantees.
///
/// At [`Objective::TargetKind`], only a win **by `me`** of exactly `kind`
/// scores `1.0`; everything else -- the *other* seat winning by any kind
/// (`kind` included), a draw, or `me` winning by a *different* kind -- scores
/// `0.0`. `me` is [`Tree::me`]: the seat this tree is actually searching for,
/// not a game-fixed constant -- see [`Tree::exploit`]'s docs for why
/// anchoring this to a fixed seat regardless of who is searching was a bug,
/// not a simplification, and this crate's docs for what fixing it measured.
/// That makes this a deliberately narrower reward than [`value_of`]'s: "only
/// rewarded from scientific-supremacy outcomes" (to take the sharpest
/// example) means a specialist gets nothing at all for winning by military
/// supremacy, civilian victory, or by science as the *other* player.
#[inline]
pub(crate) fn objective_value_of(result: GameResult, objective: Objective, me: Player) -> f64 {
    match objective {
        Objective::WinProbability => value_of(result),
        Objective::TargetKind(target) => match result {
            GameResult::Win { winner, kind }
                if winner == me && kind_matches_target(kind, target) =>
            {
                1.0
            }
            _ => 0.0,
        },
    }
}

/// Whether a finished game's actual [`VictoryKind`] counts as an achievement
/// of `target`, folding [`VictoryKind::CivilianVictory`] and
/// [`VictoryKind::CivilianTiebreak`] together the same way
/// [`duels_value::Outcome::of`] does -- see [`Objective::TargetKind`]'s docs.
#[inline]
pub(crate) fn kind_matches_target(actual: VictoryKind, target: VictoryKind) -> bool {
    use VictoryKind::{CivilianTiebreak, CivilianVictory};
    match (actual, target) {
        (CivilianVictory | CivilianTiebreak, CivilianVictory | CivilianTiebreak) => true,
        _ => actual == target,
    }
}

/// UCB1 for one child: exploitation from the mover's perspective plus the
/// exploration bonus.
///
/// `libm::log` rather than `f64::ln`: this is a search decision that feeds
/// straight into which move gets played, and the platform's own libm can
/// disagree with another architecture's in the last bit for a transcendental
/// function like this one. `libm` is a portable, software implementation, so
/// the same seed produces the same search on an ARM Raspberry Pi as on an
/// Apple Silicon workstation. `sqrt` is untouched: IEEE 754 requires it to be
/// correctly rounded, so unlike `ln`/`pow` it does not vary by platform.
#[inline]
pub(crate) fn ucb1(exploit: f64, child_visits: u32, parent_visits: u32, c: f64) -> f64 {
    if child_visits == 0 {
        return f64::INFINITY;
    }
    exploit + c * (libm::log(f64::from(parent_visits)) / f64::from(child_visits)).sqrt()
}

/// The arena and the search over it.
pub(crate) struct Tree {
    pub nodes: Vec<Node>,
    pub cfg: Config,
    /// Scratch: the path of the current simulation, root first.
    path: Vec<NodeId>,
    /// Scratch: legal-action buffer, reused to keep playouts allocation-free.
    buf: Vec<Action>,
    /// Scratch: one rollout weight per legal action, so the playout policy
    /// evaluates each candidate's weight exactly once per step instead of
    /// twice (once to total, once to draw). Owned here so it is allocated per
    /// *tree*, not per simulation.
    wbuf: Vec<f64>,
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
    /// The root-fixed pricing the *inherited hand-crafted* [`LeafValue`]
    /// variants score a leaf against — built **once per tree**, from the
    /// tree's own root position, and **only** when
    /// [`LeafValue::needs_eval_root`] says so.
    ///
    /// `None` on every default search here, which is the mirror image of
    /// `mcts-eval`, where it is `Some` on every default search and `None` only
    /// for the ablation. It is `Some` for this crate's [`Config::eval_base`]
    /// control, because that control *is* `mcts-eval`.
    ///
    /// One `duels_eval::Root::new` is about 3.5 µs against a ~8.6 µs
    /// simulation, so paying it per *tree* is free and paying it per node
    /// would not be (`crate::leaf` has the numbers).
    /// `tests::each_leaf_builds_only_the_fixtures_it_reads` pins which
    /// variants build one.
    pub eval_root: Option<duels_eval::Root>,
    /// The **learned** value network, parsed **once per tree** and only when
    /// [`Config::leaf`] is one of the learned variants — which is to say: on
    /// every default search in this crate, and on neither of its two ablation
    /// controls.
    ///
    /// Parsed per tree rather than cached in a process-wide global on purpose:
    /// a `duels-server` room lives for hours, and a lazily cached global would
    /// be one more piece of hidden state in something this crate works hard to
    /// keep a pure function of its inputs. About a hundred kilobytes of
    /// parsing against a whole tree's search is not a cost worth that.
    ///
    /// **Which** weights this is, is not left implicit anywhere: the content
    /// hash goes into [`Config::describe`], and the `golden` module pins
    /// twenty positions' predictions so a retrain fails a test.
    pub learned_net: Option<duels_value::Net>,
    /// The player to move at the **root** — i.e. whichever seat this tree is
    /// actually searching for, in the real game this tree was built to decide
    /// a move in. Captured once, here, because [`Objective::TargetKind`]'s
    /// reward has to be anchored to *this* seat rather than to a
    /// game-fixed one; see [`Tree::exploit`]'s docs for why.
    me: Player,
}

impl Tree {
    /// A tree rooted at `state`, whose root actions are restricted to
    /// `actions` (the actions the arena actually offered).
    ///
    /// # Where the leaf's value model comes from
    ///
    /// Both per-tree fixtures are built **here**, at tree-construction time,
    /// rather than in `MctsValueAgent::new` or in a process-wide cache: the
    /// [`duels_value::Net`] this crate's default leaf reads, and — for the
    /// inherited hand-crafted leaves only, [`Config::eval_base`] among them —
    /// one [`duels_eval::Root`] from [`Config::eval_config`].
    ///
    /// Building them per *search* rather than per *agent* is what keeps a
    /// long-lived `duels-server` room from carrying a snapshot of either for
    /// hours, and it is why the `eval_base` control tracks `duels-eval` live
    /// exactly as `mcts-eval` does.
    ///
    /// Note the asymmetry between the two, which is deliberate and is argued
    /// in the crate docs' "The weights are pinned" section: the *evaluation
    /// configuration* is read live, and the *learned weights* are pinned by
    /// the `golden` module. They are different kinds of thing — a tuned
    /// vector a code owner reviews, against a fitted artefact a retrain
    /// replaces wholesale.
    pub fn new(state: GameState, actions: Vec<Action>, cfg: Config, rng: &mut StdRng) -> Self {
        // Built for the player to move at the root, exactly as `phased` builds
        // it for a decision: the `Root`'s asymmetric parts (the stance it
        // carries and the denial scale) are read only by
        // `duels_eval::Root::denial_term`, which a leaf evaluation never
        // calls, and `duels_eval::evaluate` is exactly antisymmetric in its
        // `me` argument.
        let eval_root = cfg
            .leaf
            .needs_eval_root()
            .then(|| duels_eval::Root::new(&state, state.current_player(), cfg.eval_config()));
        let learned_net = cfg.leaf.needs_learned_net().then(|| {
            match cfg.value_weights_override {
                Some(bytes) => duels_value::Net::from_bytes(bytes)
                    .expect("a pinned weights override matches this build's features"),
                None => duels_value::default_net(),
            }
            .with_summation(cfg.value_summation)
        });
        let me = state.current_player();
        let mut tree = Self {
            nodes: Vec::with_capacity(1024),
            cfg,
            path: Vec::with_capacity(64),
            buf: Vec::with_capacity(32),
            wbuf: Vec::with_capacity(32),
            simulations: 0,
            rankings: 0,
            eval_root,
            learned_net,
            me,
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
    ///
    /// The anchor `mean` is backed up against depends on [`Config::objective`]:
    /// at [`Objective::WinProbability`] it is always [`Player::One`] (see
    /// [`value_of`] — this is what every other agent on this ladder does, and
    /// what this crate's default search does too, so this arm must stay
    /// exactly `Player::One`, unconditionally, for the bit-identical
    /// guarantee `tests::objective_win_probability_is_bit_identical_to_value_of`
    /// checks). `P(One wins) = 1 - P(Two wins)` always, which is what makes a
    /// single Player-One-anchored scalar plus this flip valid for *either*
    /// seat's tree.
    ///
    /// At [`Objective::TargetKind`] the anchor is [`Tree::me`] instead — the
    /// seat this *specific* tree is searching for, i.e. whichever seat had
    /// the move at the root. This is not the same fix as the
    /// `WinProbability` arm applied to a different constant: `P(One achieves
    /// kind K)` and `P(Two achieves kind K)` are **not** complementary (most
    /// games, neither player achieves a specific rare kind), so anchoring to
    /// a game-fixed `Player::One` regardless of which seat is searching is a
    /// bug, not a simplification — a tree built for `Player::Two` would then
    /// score every one of its own branches by "does the *other* seat get the
    /// kind", which is backwards. This was caught empirically: a science
    /// specialist showed 54.7% science-race exposure as `Player::One` but
    /// 0.0% as `Player::Two` in the same measurement run, which is exactly
    /// the signature of this bug (the flipped, `Player::One`-anchored
    /// quantity is dominated by the ~90%+ "science wasn't achieved by One"
    /// mass regardless of the true prospects for `Two`).
    #[inline]
    pub fn exploit(&self, child: NodeId, mover: Player) -> f64 {
        let mean = self.nodes[child as usize].mean();
        let anchor = match self.cfg.objective {
            Objective::WinProbability => Player::One,
            Objective::TargetKind(_) => self.me,
        };
        if mover == anchor {
            mean
        } else {
            1.0 - mean
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
                    value: objective_value_of(result, self.cfg.objective, self.me),
                },
            };
        }
        engine::legal_actions_into(&state, &mut self.buf);
        if self.buf.is_empty() {
            // `legal_actions` is empty exactly when the game is over, so this
            // is unreachable; score it rather than trusting the invariant.
            let value = objective_value_of(
                duels_core::scoring::civilian_result(&state),
                self.cfg.objective,
                self.me,
            );
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
        // `libm::pow`, not `f64::powf`, for the same cross-platform-
        // determinism reason `ucb1` uses `libm::log`: this decides how many
        // chance outcomes get expanded, which is itself a search decision.
        let allowance =
            self.cfg.chance_widen_c * libm::pow(f64::from(visits + 1), self.cfg.chance_widen_alpha);

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

    /// What a leaf the simulation has just added to the tree is worth, on the
    /// `[0, 1]` Player-One scale every node accumulates.
    ///
    /// The one dispatch point for [`Config::leaf`]. [`LeafValue::Rollout`]
    /// reaches `rollout_from` and nothing else, which is what makes
    /// [`Config::rollout_base`] reproduce `mcts-uct`'s search node for node.
    fn leaf_value(&mut self, node: NodeId, rng: &mut StdRng) -> f64 {
        match self.cfg.leaf {
            LeafValue::Rollout => self.rollout_from(node, rng),
            // `static_from` returns `None` only if the tree was built without
            // an evaluation root, which `Tree::new` cannot do for this
            // variant; falling back to the playout keeps a hypothetical
            // mis-construction searching properly rather than backing up a
            // constant.
            LeafValue::Static => self
                .static_from(node)
                .unwrap_or_else(|| self.rollout_from(node, rng)),
            LeafValue::Truncated { plies } => {
                let mut state = self.nodes[node as usize].state;
                let cap = plies.min(self.cfg.max_rollout_plies);
                let finished = rollout::play_out_capped(
                    &mut state,
                    &self.cfg.rollout,
                    &self.cfg.race,
                    &mut self.buf,
                    &mut self.wbuf,
                    rng,
                    cap,
                );
                match finished {
                    // The game ended inside the window, so there is a real
                    // result and no judgement to make.
                    Some(result) => objective_value_of(result, self.cfg.objective, self.me),
                    None => match self.eval_root.as_ref() {
                        Some(root) => leaf::static_value(&state, root),
                        None => objective_value_of(
                            rollout::play_out(
                                &mut state,
                                &self.cfg.rollout,
                                &self.cfg.race,
                                &mut self.buf,
                                &mut self.wbuf,
                                rng,
                                self.cfg.max_rollout_plies,
                            ),
                            self.cfg.objective,
                            self.me,
                        ),
                    },
                }
            }
            LeafValue::Blend { weight } => {
                // The static half first, though it consumes no randomness, so
                // that the playout's draws land in the same order as they
                // would for `Rollout`.
                let statically = self.static_from(node);
                let played = self.rollout_from(node, rng);
                match statically {
                    Some(s) => weight * s + (1.0 - weight) * played,
                    None => played,
                }
            }
            // The two learned variants mirror `Static` and `Blend` exactly,
            // including the fall-back-to-playout arm for a tree built without
            // the network (which `Tree::new` cannot do for these variants) and
            // the ordering that keeps the playout's draws where `Rollout`
            // would have them.
            LeafValue::Learned => self
                .learned_from(node)
                .unwrap_or_else(|| self.rollout_from(node, rng)),
            LeafValue::LearnedBlend { weight } => {
                let learned = self.learned_from(node);
                let played = self.rollout_from(node, rng);
                match learned {
                    Some(s) => weight * s + (1.0 - weight) * played,
                    None => played,
                }
            }
            // Mirrors `Learned` exactly, reading `learned_symmetric_from`
            // (one extra forward pass, both perspectives) instead of
            // `learned_from`.
            LeafValue::LearnedSymmetric => self
                .learned_symmetric_from(node)
                .unwrap_or_else(|| self.rollout_from(node, rng)),
        }
    }

    /// A full playout from `node`, valued as a real [`GameResult`]: the leaf
    /// value this crate has always used.
    fn rollout_from(&mut self, node: NodeId, rng: &mut StdRng) -> f64 {
        let mut state = self.nodes[node as usize].state;
        let result = rollout::play_out(
            &mut state,
            &self.cfg.rollout,
            &self.cfg.race,
            &mut self.buf,
            &mut self.wbuf,
            rng,
            self.cfg.max_rollout_plies,
        );
        objective_value_of(result, self.cfg.objective, self.me)
    }

    /// [`leaf::static_value`] of `node`, or `None` if this tree has no
    /// evaluation root.
    fn static_from(&self, node: NodeId) -> Option<f64> {
        let root = self.eval_root.as_ref()?;
        Some(leaf::static_value(&self.nodes[node as usize].state, root))
    }

    /// [`leaf::learned_value`] of `node`, or `None` if this tree has no value
    /// network.
    fn learned_from(&self, node: NodeId) -> Option<f64> {
        let net = self.learned_net.as_ref()?;
        Some(leaf::learned_value(
            &self.nodes[node as usize].state,
            net,
            self.cfg.objective,
            self.me,
        ))
    }

    /// [`leaf::learned_symmetric_value`] of `node`, or `None` if this tree has
    /// no value network. [`learned_from`](Tree::learned_from)'s twin: same
    /// fallback shape, one extra forward pass (both perspectives instead of
    /// one).
    fn learned_symmetric_from(&self, node: NodeId) -> Option<f64> {
        let net = self.learned_net.as_ref()?;
        Some(leaf::learned_symmetric_value(
            &self.nodes[node as usize].state,
            net,
            self.cfg.objective,
            self.me,
        ))
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
                    value = self.leaf_value(node, rng);
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

/// What the search concluded about its root position, read back out after the
/// budget is spent.
///
/// # Why this exists
///
/// The search already computes all of this on the way to picking a move; until
/// this type existed it computed it, used it inside [`best_of`], and dropped
/// it. Nothing in the agent needs it, so this is **read-only bookkeeping for
/// callers outside the agent** — a training-corpus generator, a diagnostic, a
/// UI that wants to show the search's own read of a position. Populating it
/// changes no search decision, consumes no randomness, and touches no node:
/// [`root_stats`] is a fold over statistics that are already there.
///
/// # What `value` is, exactly
///
/// The root's own backed-up mean, on the same `[0, 1]` **Player One** scale
/// every node in this tree accumulates (see the module docs' value
/// convention). So it is the average of `visits` leaf values, and under
/// [`Config::default`] a leaf value is *entirely* [`duels_value`]'s learned
/// win probability with no playout in it — which is to say: this field is a
/// search-*shaped* aggregate of the model's own opinions, and not a
/// search-derived quantity independent of the model.
///
/// **That makes it a poor training target for a `duels-value` retrain**, and
/// worse than `mcts-eval`'s version of the same field. There the target
/// contained the evaluation at the blend weight and a real playout for the
/// other half, so at least half the signal came from simulating the game.
/// Here every leaf is the model, so fitting the next generation of weights
/// against this would be self-distillation with no fresh outcome information
/// entering anywhere. Use `mcts-eval`'s `RootStats`, or
/// [`Config::rollout_base`]'s, where the same field is a pure playout win rate
/// with no learned value in it at all.
#[derive(Debug, Clone, PartialEq)]
pub struct RootStats {
    /// The root's backed-up mean value, from [`Player::One`]'s perspective.
    pub value: f64,
    /// The player who was to move at the root — whose perspective
    /// [`RootStats::value_for_mover`] reports.
    pub mover: Player,
    /// Root visits, summed across the ensemble's trees. Close to, but not
    /// exactly, the node budget: a slice of zero still runs one simulation
    /// (see `Slices::run`).
    pub visits: u64,
    /// Every root action and how many visits it received, summed across the
    /// ensemble's trees. An action the search never expanded is present with
    /// `0`. The order is the first tree's own (shuffled) root order, which is
    /// **not** `engine::legal_actions` order — match on the [`Action`], don't
    /// index.
    ///
    /// This is the raw material for a policy target: normalising these to sum
    /// to one gives the visit distribution AlphaZero-style training uses.
    pub policy: Vec<(Action, u32)>,
}

impl RootStats {
    /// [`RootStats::value`] from the point of view of the player who was
    /// actually to move — the one perspective flip, exactly as
    /// [`Tree::exploit`] does it.
    pub fn value_for_mover(&self) -> f64 {
        match self.mover {
            Player::One => self.value,
            Player::Two => 1.0 - self.value,
        }
    }
}

/// Pool [`RootStats`] over an ensemble of trees rooted at the same public
/// position.
///
/// Returns `None` if there are no trees, or if the first tree's root is not a
/// decision node (which cannot happen for a tree the agent built, since
/// `Tree::new` always makes one).
pub(crate) fn root_stats(trees: &[Tree]) -> Option<RootStats> {
    let first = trees.first()?;
    let Kind::Decision { mover, actions, .. } = &first.nodes[0].kind else {
        return None;
    };
    let mut visits = 0u64;
    let mut value_sum = 0.0f64;
    for tree in trees {
        visits += u64::from(tree.nodes[0].visits);
        value_sum += tree.nodes[0].value_sum;
    }
    let policy = actions
        .iter()
        .map(|&action| {
            let n: u32 = trees
                .iter()
                .filter_map(|t| t.root_child(action))
                .map(|c| c.visits)
                .sum();
            (action, n)
        })
        .collect();
    Some(RootStats {
        // The same `visits == 0` convention `Node::mean` uses, so a root that
        // somehow ran no simulation reads as "no information" rather than NaN.
        value: if visits == 0 {
            0.5
        } else {
            value_sum / visits as f64
        },
        mover: *mover,
        visits,
        policy,
    })
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

/// **`mcts-eval`'s search, verbatim** — its `expand`, `select_ucb1`,
/// `leaf_value` and `simulate`, frozen as they read in
/// `crates/agents/mcts-eval/src/tree.rs` at the commit this crate was copied
/// from.
///
/// This is **the load-bearing test asset of this crate**, and it is what makes
/// the "self-contained copy" invariant checkable rather than merely asserted.
/// `docs/conventions.md` forbids one agent crate depending on another, so the
/// search below the leaf value here is a copy of `mcts-eval`'s; every strength
/// number in the crate docs is an *ablation* against that agent, and the
/// ablation only means anything if the copy really is the same search. A
/// frozen second copy is how that gets said in one binary, node for node,
/// rather than by two humans reading two crates side by side — and if somebody
/// edits the live search, this is the test that notices.
///
/// It is checked at **every** configuration in the leaf family, not just at
/// [`Config::eval_base`], so the claim covers this crate's own default leaf as
/// well as the two controls — see
/// `tests::the_copied_search_is_the_mcts_eval_search_node_for_node`.
///
/// Do not "simplify" any of these to call the live code, since that is the
/// thing they exist to check. Four edits only, all forced: the `priors` field
/// the type system requires is named in the patterns that need it, the shared
/// `resolve_chance`/`child_after`/`rank_by_prior` helpers are called rather
/// than re-copied (they are reached identically from both arms, and copying
/// them would test nothing extra), `eval_legacy_leaf_value` takes its
/// `duels_eval::Root` and [`duels_value::Net`] from the live fields, since
/// `Tree::new` is what builds them in both arms, and `eval_legacy_leaf_value`'s
/// match is `unreachable!()` on [`LeafValue::LearnedSymmetric`] — that variant
/// is an `mcts-value`-only addition with nothing to copy from `mcts-eval`'s
/// tree.rs, so the arm exists only because [`LeafValue`] is one enum shared by
/// both the live and the frozen match, not because this frozen copy has ever
/// been asked to run it.
#[cfg(test)]
impl Tree {
    fn eval_legacy_expand(&mut self, id: NodeId, rng: &mut StdRng) -> Option<NodeId> {
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

    fn eval_legacy_select_ucb1(&self, id: NodeId) -> NodeId {
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

    fn eval_legacy_rollout_from(&mut self, node: NodeId, rng: &mut StdRng) -> f64 {
        let mut state = self.nodes[node as usize].state;
        let result = rollout::play_out(
            &mut state,
            &self.cfg.rollout,
            &self.cfg.race,
            &mut self.buf,
            &mut self.wbuf,
            rng,
            self.cfg.max_rollout_plies,
        );
        value_of(result)
    }

    fn eval_legacy_static_from(&self, node: NodeId) -> Option<f64> {
        let root = self.eval_root.as_ref()?;
        Some(leaf::static_value(&self.nodes[node as usize].state, root))
    }

    fn eval_legacy_learned_from(&self, node: NodeId) -> Option<f64> {
        let net = self.learned_net.as_ref()?;
        // Threaded through like the live `learned_from` above: the
        // equivalence tests that drive this frozen copy only ever build it
        // from `Config::default`/`eval_base`/`rollout_base`, all of which
        // leave `objective` at `Objective::WinProbability`, so this reads
        // identically to the pre-`Objective` call it replaces.
        Some(leaf::learned_value(
            &self.nodes[node as usize].state,
            net,
            self.cfg.objective,
            self.me,
        ))
    }

    fn eval_legacy_leaf_value(&mut self, node: NodeId, rng: &mut StdRng) -> f64 {
        match self.cfg.leaf {
            LeafValue::Rollout => self.eval_legacy_rollout_from(node, rng),
            LeafValue::Static => self
                .eval_legacy_static_from(node)
                .unwrap_or_else(|| self.eval_legacy_rollout_from(node, rng)),
            LeafValue::Truncated { plies } => {
                let mut state = self.nodes[node as usize].state;
                let cap = plies.min(self.cfg.max_rollout_plies);
                let finished = rollout::play_out_capped(
                    &mut state,
                    &self.cfg.rollout,
                    &self.cfg.race,
                    &mut self.buf,
                    &mut self.wbuf,
                    rng,
                    cap,
                );
                match finished {
                    Some(result) => value_of(result),
                    None => match self.eval_root.as_ref() {
                        Some(root) => leaf::static_value(&state, root),
                        None => value_of(rollout::play_out(
                            &mut state,
                            &self.cfg.rollout,
                            &self.cfg.race,
                            &mut self.buf,
                            &mut self.wbuf,
                            rng,
                            self.cfg.max_rollout_plies,
                        )),
                    },
                }
            }
            LeafValue::Blend { weight } => {
                let statically = self.eval_legacy_static_from(node);
                let played = self.eval_legacy_rollout_from(node, rng);
                match statically {
                    Some(s) => weight * s + (1.0 - weight) * played,
                    None => played,
                }
            }
            LeafValue::Learned => self
                .eval_legacy_learned_from(node)
                .unwrap_or_else(|| self.eval_legacy_rollout_from(node, rng)),
            LeafValue::LearnedBlend { weight } => {
                let learned = self.eval_legacy_learned_from(node);
                let played = self.eval_legacy_rollout_from(node, rng);
                match learned {
                    Some(s) => weight * s + (1.0 - weight) * played,
                    None => played,
                }
            }
            // `mcts-eval` has no analogue of this variant — see this impl
            // block's doc comment's "four edits" note. No test constructs an
            // `eval_legacy` tree with this leaf.
            LeafValue::LearnedSymmetric => {
                unreachable!("LeafValue::LearnedSymmetric has no mcts-eval original to copy")
            }
        }
    }

    pub(crate) fn eval_legacy_simulate(&mut self, rng: &mut StdRng) {
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
                    value = self.eval_legacy_leaf_value(node, rng);
                    break;
                }
                Step::Decision => match self.eval_legacy_expand(node, rng) {
                    Some(child) => {
                        node = child;
                        self.path.push(node);
                        fresh = true;
                    }
                    None => {
                        node = self.eval_legacy_select_ucb1(node);
                        self.path.push(node);
                    }
                },
            }
        }

        self.simulations += 1;
        backpropagate(&mut self.nodes, &self.path, value);
    }
}

/// **`mcts-eval`'s move-selection rule, verbatim** — its `best_of`, frozen the
/// same way and for the same reason as the search above.
///
/// Drives `crate::tests::the_eval_base_is_the_mcts_eval_agent_move_for_move`,
/// which is the whole-agent, whole-game form of the node-for-node claim: with
/// this and `Tree::eval_legacy_simulate`, a frozen copy of that agent's entire
/// decision procedure can be run against the live one over dozens of `choose`
/// calls from one seeded stream.
#[cfg(test)]
pub(crate) fn eval_legacy_best_of(trees: &[Tree]) -> Option<Action> {
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
        let mean = if visits == 0 {
            0.5
        } else {
            value_sum / visits as f64
        };
        let score = match mover {
            Player::One => mean,
            Player::Two => 1.0 - mean,
        };
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

/// **`mcts-uct`'s search, verbatim** — its `expand`, `select_ucb1` and
/// `simulate` as they read with `prior=none`, `race=neutral` and a pure
/// playout leaf, which is to say: at that agent's own default configuration.
///
/// Inherited from `mcts-eval` along with the search itself, and kept for the
/// reason that crate kept it: it is the bottom of the ablation chain, and it is
/// what makes [`Config::rollout_base`] a usable control rather than a claim.
///
/// Do not "simplify" any of them to call the live code, since that is the
/// thing they exist to check. Two edits only, both forced: the `priors` field
/// the type system requires is named in the pattern that constructs a decision
/// node's children and never read, and the playout goes through
/// `rollout::legacy::play_out`, itself a verbatim copy of the same policy.
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
                    let result = rollout::legacy::play_out(
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

/// The static leaf value this search assigns to `state`, reached the way
/// production reaches it — through a real [`Tree`] at [`Config::eval_base`],
/// so the [`duels_eval::Config`] involved really is the one [`Tree::new`]
/// picks rather than one the test chose.
///
/// Exists for
/// `crate::tests::the_eval_base_control_tracks_duels_evals_live_default`,
/// which is what pins the ablation control to the champion *as shipped*.
#[cfg(test)]
pub(crate) fn static_leaf_value_for_test(state: &GameState) -> f64 {
    let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(0);
    let actions = engine::legal_actions(state);
    let tree = Tree::new(*state, actions, Config::eval_base(), &mut rng);
    tree.static_from(0)
        .expect("the eval_base leaf value builds an evaluation root")
}

/// The **learned** leaf value this search assigns to `state`, reached the way
/// production reaches it — through a real [`Tree`] at [`Config::default`], so
/// the [`duels_value::Net`] involved really is the one [`Tree::new`] parses,
/// at the summation order the default configuration selects.
///
/// Exists for the `golden` module, which pins twenty of these outright.
/// Going through a `Tree` rather than calling `duels_value` directly is the
/// whole point: it is what makes that file a test of *this agent's leaf* and
/// not a second copy of `duels-value`'s own tests.
#[cfg(test)]
pub(crate) fn learned_leaf_value_for_test(state: &GameState) -> f64 {
    let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(0);
    let actions = engine::legal_actions(state);
    let tree = Tree::new(*state, actions, Config::default(), &mut rng);
    tree.learned_from(0)
        .expect("the default leaf value parses a value network")
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
    use duels_core::testing::StateBuilder;
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

    /// [`Config::objective`]'s off value must reproduce [`value_of`]
    /// bit-for-bit, over every kind of finished game, **regardless of `me`**
    /// -- this is the `docs/conventions.md` proof that a new opt-in `Config`
    /// field changes nothing when left at its default, and it holds by
    /// construction here: [`objective_value_of`] at
    /// [`Objective::WinProbability`] is a pure delegation to [`value_of`]
    /// with no arithmetic in between and `me` unused, so there is no
    /// floating-point reassociation for this test to miss, and no way for
    /// `me` to leak in.
    #[test]
    fn objective_win_probability_is_bit_identical_to_value_of() {
        use VictoryKind::{
            CivilianTiebreak, CivilianVictory, MilitarySupremacy, ScientificSupremacy,
        };
        let results = [
            GameResult::Draw,
            GameResult::Win {
                winner: Player::One,
                kind: MilitarySupremacy,
            },
            GameResult::Win {
                winner: Player::Two,
                kind: MilitarySupremacy,
            },
            GameResult::Win {
                winner: Player::One,
                kind: ScientificSupremacy,
            },
            GameResult::Win {
                winner: Player::Two,
                kind: ScientificSupremacy,
            },
            GameResult::Win {
                winner: Player::One,
                kind: CivilianVictory,
            },
            GameResult::Win {
                winner: Player::Two,
                kind: CivilianVictory,
            },
            GameResult::Win {
                winner: Player::One,
                kind: CivilianTiebreak,
            },
            GameResult::Win {
                winner: Player::Two,
                kind: CivilianTiebreak,
            },
        ];
        for result in results {
            for me in [Player::One, Player::Two] {
                assert_eq!(
                    objective_value_of(result, Objective::WinProbability, me),
                    value_of(result),
                    "objective_value_of disagreed with value_of for {result:?} (me={me:?})"
                );
            }
        }
        // And the field really is `Config::default`'s objective, not just an
        // option that happens to exist.
        assert_eq!(Config::default().objective, Objective::WinProbability);
    }

    /// The specialist reward is anchored to `me` — the seat the tree is
    /// actually searching for — not to a game-fixed `Player::One`.
    ///
    /// This replaces an earlier version of this test (and an earlier version
    /// of [`objective_value_of`]) that pinned the reward to a hardcoded
    /// `Player::One` regardless of `me`, on the mistaken assumption that this
    /// mirrored [`value_of`]'s own `Player::One` anchoring safely. It does
    /// not: `value_of`'s anchor is safe to fix at `Player::One` for *either*
    /// seat's tree only because `P(One wins) = 1 - P(Two wins)` always holds
    /// (win/loss is complementary), which [`Tree::exploit`]'s flip relies on.
    /// `P(One wins by kind K)` and `P(Two wins by kind K)` are **not**
    /// complementary — most games neither player wins by a specific kind —
    /// so a tree searching for `Player::Two` that scored its own reward by
    /// "did `Player::One` get the target kind" was rewarding the *wrong
    /// player's* achievement throughout. This was caught empirically: a
    /// science specialist showed real science-seeking behaviour as
    /// `Player::One` (54.7% science-race exposure, well above the ~31%
    /// generalist baseline) and none at all as `Player::Two` (0% exposure,
    /// 0/150 wins) in the same measurement run.
    #[test]
    fn target_kind_terminal_values_are_anchored_to_me_not_player_one() {
        let science = Objective::TargetKind(VictoryKind::ScientificSupremacy);

        // Each seat's own target-kind win scores 1.0 for `me` == that seat,
        // and 0.0 for `me` == the other seat: the reward genuinely swaps with
        // `me`, rather than staying pinned to one player regardless.
        for (winner, me, expected) in [
            (Player::One, Player::One, 1.0),
            (Player::One, Player::Two, 0.0),
            (Player::Two, Player::Two, 1.0),
            (Player::Two, Player::One, 0.0),
        ] {
            assert_eq!(
                objective_value_of(
                    GameResult::Win {
                        winner,
                        kind: VictoryKind::ScientificSupremacy,
                    },
                    science,
                    me,
                ),
                expected,
                "winner={winner:?}, me={me:?}: the reward must track whether \
                 `me` (not a fixed seat) won by the target kind"
            );
        }

        // Whichever seat is `me`, winning the *wrong* kind must not score as
        // a target win.
        for me in [Player::One, Player::Two] {
            assert_eq!(
                objective_value_of(
                    GameResult::Win {
                        winner: me,
                        kind: VictoryKind::MilitarySupremacy,
                    },
                    science,
                    me,
                ),
                0.0,
                "me={me:?} winning the *wrong* kind must not score as a target win"
            );
            assert_eq!(
                objective_value_of(GameResult::Draw, science, me),
                0.0,
                "a draw is not a target win (me={me:?})"
            );
        }

        // The civilian-tiebreak/civilian-victory distinction `Outcome::of`
        // already collapses: this objective makes the identical call, so a
        // civilian target matches either variant, for whichever seat is `me`.
        let civilian = Objective::TargetKind(VictoryKind::CivilianVictory);
        for me in [Player::One, Player::Two] {
            assert_eq!(
                objective_value_of(
                    GameResult::Win {
                        winner: me,
                        kind: VictoryKind::CivilianTiebreak,
                    },
                    civilian,
                    me,
                ),
                1.0,
                "a civilian target must match the tiebreak variant, not just \
                 the exact enum case (me={me:?})"
            );
        }
    }

    /// [`Tree::exploit`]'s anchor for [`Objective::TargetKind`] must be `me`,
    /// not a fixed `Player::One` -- the direct regression test for the bug
    /// [`target_kind_terminal_values_are_anchored_to_me_not_player_one`]'s
    /// docs describe. A hand-built two-node tree where the child is a
    /// terminal `Player::Two` science win: searched as `me = Player::Two`,
    /// `exploit` from `Two`'s own decision node must read this as a `1.0`
    /// (a real target win for the seat that's searching), not as `0.0` (what
    /// the old `Player::One`-anchored code, flipped for `Two`, produced).
    #[test]
    fn exploit_anchors_target_kind_to_me_not_a_fixed_player() {
        let science = Objective::TargetKind(VictoryKind::ScientificSupremacy);
        let cfg = Config {
            objective: science,
            ..Config::default()
        };
        let mut rng = StdRng::seed_from_u64(0);
        // A tree whose root move belongs to Player Two, so `Tree::me` is
        // `Player::Two` -- exactly the seat the reported bug affected.
        let root_state = StateBuilder::new().current(Player::Two).build();
        let mut tree = Tree::new(root_state, vec![Action::Discard { slot: 0 }], cfg, &mut rng);
        assert_eq!(tree.me, Player::Two, "the root's mover must set Tree::me");

        // Two won by science: a real target-win for the seat that's
        // searching, and the *only* fixture this test needs. `mean()` reads
        // `value_sum`/`visits`, not the `Kind::Terminal` label, so both are
        // set to one visit worth of this exact backed-up value.
        let value = objective_value_of(
            GameResult::Win {
                winner: Player::Two,
                kind: VictoryKind::ScientificSupremacy,
            },
            science,
            tree.me,
        );
        let mut node = terminal(value);
        node.visits = 1;
        node.value_sum = value;
        let child = tree.push(node);

        assert_eq!(
            tree.exploit(child, Player::Two),
            1.0,
            "Player Two's own science win must read as a full target-win, not \
             be flipped away by an anchor stuck on Player::One"
        );
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

    /// **The ablation arm is really `mcts-uct`**, node for node — not "the
    /// same move", the *same arena*: every node's kind, visit count, value
    /// sum, action order and child wiring, checked against the verbatim copy
    /// of that agent's `expand`/`select_ucb1`/`simulate` in the `legacy`
    /// block above.
    ///
    /// This is what makes [`Config::rollout_base`] a usable control rather
    /// than a claim. It also proves the whole `leaf_value` dispatch consumes
    /// no randomness on the playout path, since a single extra RNG draw would
    /// desynchronise every chance node below it.
    #[test]
    fn the_rollout_base_grows_the_mcts_uct_tree_node_for_node() {
        let cfg = Config::rollout_base();
        assert_eq!(cfg.exploration, 1.0);
        assert_eq!(cfg.leaf, LeafValue::Rollout);
        assert_eq!(cfg.prior, PriorMode::None);
        assert_eq!(cfg.race, RaceWeights::NEUTRAL);
        assert_eq!(cfg.rollout, RolloutWeights::BIASED);

        for seed in 0..10u64 {
            let state = engine::new_game(seed);
            let actions = engine::legal_actions(&state);

            let mut rng_new = StdRng::seed_from_u64(seed ^ 0x1EAF);
            let mut new = Tree::new(state, actions.clone(), cfg, &mut rng_new);
            let mut rng_old = StdRng::seed_from_u64(seed ^ 0x1EAF);
            let mut old = Tree::new(state, actions.clone(), cfg, &mut rng_old);

            for _ in 0..500 {
                new.simulate(&mut rng_new);
                old.legacy_simulate(&mut rng_old);
            }

            assert_same_arena(&new, &old, seed);
        }
    }

    /// **The copied machinery is `mcts-eval`'s search**, node for node — not
    /// "the same move", the *same arena*: every node's kind, visit count,
    /// value sum, action order and child wiring, checked against the verbatim
    /// frozen copy of that agent's
    /// `expand`/`select_ucb1`/`leaf_value`/`simulate` in the `eval_legacy`
    /// block above.
    ///
    /// This is the test that makes `docs/conventions.md`'s "agent crates are
    /// self-contained" duplication *checkable* here. Every strength number in
    /// the crate docs is an ablation against `mcts-eval`, so the copy below
    /// this crate's leaf value has to be that agent's search and not a
    /// lookalike.
    ///
    /// Run at **four** configurations, not just at the control, so the claim
    /// covers the whole leaf family and the whole dispatch — including this
    /// crate's own default. A single stray RNG draw in any arm would
    /// desynchronise every chance node below it and fail here.
    #[test]
    fn the_copied_search_is_the_mcts_eval_search_node_for_node() {
        for (name, cfg) in [
            ("default (learned, c=0.15)", Config::default()),
            ("eval_base (mcts-eval)", Config::eval_base()),
            ("rollout_base (mcts-uct)", Config::rollout_base()),
            (
                "learned_blend:0.5",
                Config {
                    leaf: LeafValue::LearnedBlend { weight: 0.5 },
                    ..Config::default()
                },
            ),
        ] {
            for seed in 0..6u64 {
                let state = engine::new_game(seed);
                let actions = engine::legal_actions(&state);

                let mut rng_new = StdRng::seed_from_u64(seed ^ 0x1EAF);
                let mut new = Tree::new(state, actions.clone(), cfg, &mut rng_new);
                let mut rng_old = StdRng::seed_from_u64(seed ^ 0x1EAF);
                let mut old = Tree::new(state, actions.clone(), cfg, &mut rng_old);

                for _ in 0..300 {
                    new.simulate(&mut rng_new);
                    old.eval_legacy_simulate(&mut rng_old);
                }

                assert_same_arena(&new, &old, seed);
                assert!(
                    new.nodes.len() > 30,
                    "{name}, seed {seed}: the tree was too small to prove much"
                );
            }
        }
    }

    /// ...and this crate's default is emphatically *not* `mcts-eval`'s search,
    /// which is what stops the test above from being a comparison between two
    /// spellings of one agent.
    ///
    /// The two differ in exactly two `Config` fields, and this asserts that
    /// those two fields really do produce a different search on every seed —
    /// so `Config::eval_base` is a *control*, in the sense a measurement needs,
    /// rather than a synonym for the default.
    #[test]
    fn the_default_search_is_not_the_mcts_eval_search() {
        assert_ne!(Config::default().leaf, Config::eval_base().leaf);
        assert_ne!(
            Config::default().exploration.to_bits(),
            Config::eval_base().exploration.to_bits()
        );
        let mut differed = 0u32;
        for seed in 0..10u64 {
            let state = engine::new_game(seed);
            let actions = engine::legal_actions(&state);

            let mut rng_new = StdRng::seed_from_u64(seed ^ 0x1EAF);
            let mut new = Tree::new(state, actions.clone(), Config::default(), &mut rng_new);
            let mut rng_old = StdRng::seed_from_u64(seed ^ 0x1EAF);
            let mut old = Tree::new(state, actions.clone(), Config::eval_base(), &mut rng_old);

            for _ in 0..300 {
                new.simulate(&mut rng_new);
                old.eval_legacy_simulate(&mut rng_old);
            }

            let same = new.nodes.len() == old.nodes.len()
                && new
                    .nodes
                    .iter()
                    .zip(old.nodes.iter())
                    .all(|(a, b)| a.visits == b.visits && a.value_sum == b.value_sum);
            if !same {
                differed += 1;
            }
        }
        assert_eq!(
            differed, 10,
            "the default configuration searched identically to mcts-eval somewhere"
        );
    }

    /// ...and nor is it `mcts-uct`'s, the bottom of the same ablation chain.
    #[test]
    fn the_default_search_is_not_the_mcts_uct_search() {
        let mut differed = 0u32;
        for seed in 0..10u64 {
            let state = engine::new_game(seed);
            let actions = engine::legal_actions(&state);

            let mut rng_new = StdRng::seed_from_u64(seed ^ 0x1EAF);
            let mut new = Tree::new(state, actions.clone(), Config::default(), &mut rng_new);
            let mut rng_old = StdRng::seed_from_u64(seed ^ 0x1EAF);
            let mut old = Tree::new(state, actions.clone(), Config::rollout_base(), &mut rng_old);

            for _ in 0..500 {
                new.simulate(&mut rng_new);
                old.legacy_simulate(&mut rng_old);
            }

            let same = new.nodes.len() == old.nodes.len()
                && new
                    .nodes
                    .iter()
                    .zip(old.nodes.iter())
                    .all(|(a, b)| a.visits == b.visits && a.value_sum == b.value_sum);
            if !same {
                differed += 1;
            }
        }
        assert_eq!(
            differed, 10,
            "the default configuration searched identically to mcts-uct somewhere"
        );
    }

    /// Which variants pay for which per-tree fixture: the three that read the
    /// hand-crafted evaluation build exactly one [`duels_eval::Root`], the two
    /// learned ones parse exactly one [`duels_value::Net`], and
    /// [`LeafValue::Rollout`] pays for neither. No variant pays for both.
    ///
    /// The mirror image of `mcts-eval`'s version of this test, which is the
    /// point: **this crate's default parses a network and builds no `Root`**,
    /// and its `eval_base` control does the opposite. So a `duels-eval` change
    /// cannot reach this agent's default path at all, and a `duels-value`
    /// retrain cannot reach its control arm.
    #[test]
    fn each_leaf_builds_only_the_fixtures_it_reads() {
        let (state, actions) = mid_game(2);
        // The default pays for a value network and for no `Root`; the two
        // ablation controls are the other way round (or pay for neither).
        assert!(Config::default().leaf.needs_learned_net());
        assert!(!Config::default().leaf.needs_eval_root());
        assert!(Config::eval_base().leaf.needs_eval_root());
        assert!(!Config::eval_base().leaf.needs_learned_net());
        assert!(!Config::rollout_base().leaf.needs_eval_root());
        assert!(!Config::rollout_base().leaf.needs_learned_net());
        for (leaf, wants_root, wants_net) in [
            (LeafValue::Rollout, false, false),
            (LeafValue::Static, true, false),
            (LeafValue::Truncated { plies: 8 }, true, false),
            (LeafValue::Blend { weight: 0.5 }, true, false),
            (LeafValue::Learned, false, true),
            (LeafValue::LearnedBlend { weight: 0.5 }, false, true),
            (LeafValue::LearnedSymmetric, false, true),
        ] {
            let mut rng = StdRng::seed_from_u64(9);
            let tree = Tree::new(
                state,
                actions.clone(),
                Config {
                    leaf,
                    ..Config::default()
                },
                &mut rng,
            );
            assert_eq!(
                tree.eval_root.is_some(),
                wants_root,
                "{leaf:?} built the wrong number of evaluation roots"
            );
            assert_eq!(
                tree.learned_net.is_some(),
                wants_net,
                "{leaf:?} parsed the wrong number of value networks"
            );
        }
    }

    /// The default leaf is [`LeafValue::Learned`] and nothing else: the whole
    /// default tree is grown twice — once through `Config::default().leaf`,
    /// once against an explicit spelling of it — and compared, and then
    /// against every other member of the family to show none of them is a
    /// synonym for it.
    ///
    /// The counterpart of `mcts-eval`'s
    /// `the_default_search_is_untouched_by_the_learned_variants`, inverted:
    /// there the learned leaves had to be provably *inert*, here they are the
    /// product and it is the hand-crafted ones that have to be provably
    /// *different*.
    #[test]
    fn the_default_leaf_is_the_learned_one_and_nothing_else() {
        let (state, actions) = mid_game(11);
        let grow = |leaf| {
            let mut rng = StdRng::seed_from_u64(0x1EA2_F1ED);
            let mut tree = Tree::new(
                state,
                actions.clone(),
                Config {
                    leaf,
                    ..Config::default()
                },
                &mut rng,
            );
            for _ in 0..400 {
                tree.simulate(&mut rng);
            }
            (
                tree.nodes.len(),
                tree.simulations,
                tree.nodes[0].visits,
                tree.nodes[0].value_sum.to_bits(),
            )
        };
        assert_eq!(
            grow(Config::default().leaf),
            grow(LeafValue::Learned),
            "the default leaf and its explicit spelling grew different trees"
        );
        // ...and every other leaf really is a different search, so the
        // equality above is not vacuous.
        for other in [
            LeafValue::Blend { weight: 0.5 },
            LeafValue::LearnedBlend { weight: 0.5 },
            LeafValue::LearnedSymmetric,
            LeafValue::Static,
            LeafValue::Rollout,
            LeafValue::Truncated { plies: 8 },
        ] {
            assert_ne!(
                grow(Config::default().leaf),
                grow(other),
                "{other:?} searched identically to the learned default"
            );
        }
    }

    /// A learned leaf value has to be a probability, whatever the network says
    /// about a position: that is what makes it commensurable with the playout
    /// values the same tree backs up, and it is the shape invariant that
    /// stands in for freezing the numbers (the same choice `mcts-eval` made
    /// for its `duels-eval` leaf).
    ///
    /// Covers both single-perspective and symmetrized leaves:
    /// [`LeafValue::LearnedSymmetric`] averages a `[0, 1]` value with a
    /// `1 -` complement of another, so this is the check that the averaging
    /// itself cannot produce something outside `[0, 1]`.
    #[test]
    fn every_learned_leaf_value_is_a_probability() {
        for leaf in [LeafValue::Learned, LeafValue::LearnedSymmetric] {
            for seed in 0..12u64 {
                let (state, actions) = mid_game(seed);
                let mut rng = StdRng::seed_from_u64(seed ^ 0x1A5E);
                let mut tree = Tree::new(
                    state,
                    actions,
                    Config {
                        leaf,
                        ..Config::default()
                    },
                    &mut rng,
                );
                for _ in 0..200 {
                    tree.simulate(&mut rng);
                }
                for (i, node) in tree.nodes.iter().enumerate() {
                    if node.visits > 0 {
                        let mean = node.mean();
                        assert!(
                            mean.is_finite() && (0.0..=1.0).contains(&mean),
                            "{leaf:?} seed {seed} node {i}: backed-up mean {mean}"
                        );
                    }
                }
            }
        }
    }

    /// **`duels-eval` cannot reach this crate's default path at all.** Pinning
    /// any generation whatsoever through [`Config::eval_override`] — including
    /// one deliberately chosen to be very different from today's — leaves the
    /// default search bit-identical, because [`LeafValue::Learned`] never
    /// builds a [`duels_eval::Root`] and never calls `duels_eval::evaluate`.
    ///
    /// This is the positive form of the crate docs' claim that a `duels-eval`
    /// round moves `mcts-eval` and does not move this agent. Note that it is
    /// the *opposite* shape of test from the two below, which assert the field
    /// is live on the `eval_base` control — that asymmetry is the design.
    #[test]
    fn the_evaluation_cannot_reach_the_default_leaf() {
        let (state, actions) = mid_game(3);
        let grow = |eval_override| {
            let mut rng = StdRng::seed_from_u64(0xE7A1);
            let mut tree = Tree::new(
                state,
                actions.clone(),
                Config {
                    eval_override,
                    ..Config::default()
                },
                &mut rng,
            );
            for _ in 0..300 {
                tree.simulate(&mut rng);
            }
            tree
        };
        let live = grow(None);
        assert!(live.eval_root.is_none(), "the default built a Root");
        for pinned in [duels_eval::Config::default(), duels_eval::Config::v1()] {
            let other = grow(Some(pinned));
            assert_eq!(live.nodes.len(), other.nodes.len());
            for (na, nb) in live.nodes.iter().zip(other.nodes.iter()) {
                assert_eq!(na.visits, nb.visits);
                assert_eq!(na.value_sum.to_bits(), nb.value_sum.to_bits());
            }
        }
    }

    /// [`Config::eval_override`]'s off value (`None`) must be bit-identical to
    /// not having the field at all *on the arm that reads it*: the same tree,
    /// simulation for simulation, as an explicit
    /// `Some(duels_eval::Config::default())`.
    ///
    /// Asserted on [`Config::eval_base`] rather than on the default, because
    /// that is the configuration this field is live for, and because it is the
    /// property the ablation control needs: `eval_base` has to be `mcts-eval`
    /// *tracking `duels-eval` live*, not a frozen snapshot of it.
    #[test]
    fn eval_override_none_is_bit_identical_to_pinning_todays_live_default() {
        let (state, actions) = mid_game(3);
        let grow = |eval_override| {
            let mut rng = StdRng::seed_from_u64(0xE7A1);
            let mut tree = Tree::new(
                state,
                actions.clone(),
                Config {
                    eval_override,
                    ..Config::eval_base()
                },
                &mut rng,
            );
            for _ in 0..300 {
                tree.simulate(&mut rng);
            }
            tree
        };
        let a = grow(None);
        assert!(a.eval_root.is_some(), "eval_base must build a Root");
        let b = grow(Some(duels_eval::Config::default()));
        assert_eq!(a.nodes.len(), b.nodes.len());
        for (na, nb) in a.nodes.iter().zip(b.nodes.iter()) {
            assert_eq!(na.visits, nb.visits);
            assert_eq!(na.value_sum.to_bits(), nb.value_sum.to_bits());
        }
    }

    /// A pinned generation must actually change what the `eval_base` control
    /// scores against — otherwise the identity test above would be passing for
    /// the trivial reason that nothing reads `eval_override` on that arm
    /// either.
    #[test]
    fn an_eval_override_grows_a_different_tree_from_the_live_default() {
        assert_ne!(
            duels_eval::Config::v1(),
            duels_eval::Config::default(),
            "v1 and today's default must differ or this test is vacuous"
        );
        let (state, actions) = mid_game(3);
        let grow = |eval_override| {
            let mut rng = StdRng::seed_from_u64(0xE7A1);
            let mut tree = Tree::new(
                state,
                actions.clone(),
                Config {
                    eval_override,
                    ..Config::eval_base()
                },
                &mut rng,
            );
            for _ in 0..300 {
                tree.simulate(&mut rng);
            }
            tree
        };
        let live = grow(None);
        let pinned = grow(Some(duels_eval::Config::v1()));
        let differed = live
            .nodes
            .iter()
            .zip(pinned.nodes.iter())
            .filter(|(a, b)| a.value_sum.to_bits() != b.value_sum.to_bits())
            .count();
        assert!(
            differed > 0,
            "pinning v1 scored every node identically to live"
        );
    }

    /// [`Config::describe`]'s `eval=` tail must name the pinned generation on
    /// the arm that reads one, and say `unused` on the arm that does not — a
    /// results file has to record which evaluation, if any, was in force.
    #[test]
    fn describe_reports_the_pinned_generation_not_the_live_one() {
        let live = Config::eval_base().describe();
        let pinned = Config {
            eval_override: Some(duels_eval::Config::v1()),
            ..Config::eval_base()
        }
        .describe();
        assert_ne!(live, pinned);
        assert!(pinned.contains(&duels_eval::Config::v1().params_string()));

        // ...and the default path names no evaluation at all, pinned or
        // otherwise, because it reads none.
        let default = Config::default().describe();
        assert!(default.contains("eval=unused"), "{default}");
        assert_eq!(
            default,
            Config {
                eval_override: Some(duels_eval::Config::v1()),
                ..Config::default()
            }
            .describe(),
            "an override changed the default configuration's spec string"
        );
    }

    /// A static leaf must actually *change* the search — otherwise the
    /// equivalence test above would be passing for the trivial reason.
    #[test]
    fn a_static_leaf_grows_a_different_tree() {
        let mut differed = 0u32;
        for seed in 0..8u64 {
            let (state, actions) = mid_game(seed);
            let grow = |leaf| {
                let mut rng = StdRng::seed_from_u64(seed ^ 0x7777);
                let mut tree = Tree::new(
                    state,
                    actions.clone(),
                    Config {
                        leaf,
                        ..Config::default()
                    },
                    &mut rng,
                );
                for _ in 0..400 {
                    tree.simulate(&mut rng);
                }
                (
                    tree.nodes.len(),
                    tree.nodes.iter().map(|n| n.value_sum).sum::<f64>(),
                )
            };
            if grow(LeafValue::Static) != grow(LeafValue::Rollout) {
                differed += 1;
            }
        }
        assert_eq!(
            differed, 8,
            "a static leaf changed nothing at some position"
        );
    }

    /// The two edges of the family: a zero-ply truncation is exactly a static
    /// evaluation, and a blend at weight zero is exactly the playout value.
    ///
    /// Both are asserted on the *leaf value itself* rather than on a whole
    /// search, because that is where the claim lives — and for the blend, on
    /// the same RNG stream, since `Blend` runs the same playout `Rollout`
    /// does.
    #[test]
    fn the_degenerate_leaf_settings_reduce_to_their_edges() {
        for seed in 0..6u64 {
            let (state, actions) = mid_game(seed);
            let leaf_value = |leaf, stream: u64| {
                let mut rng = StdRng::seed_from_u64(stream);
                let mut tree = Tree::new(
                    state,
                    actions.clone(),
                    Config {
                        leaf,
                        ..Config::default()
                    },
                    &mut rng,
                );
                // Expand one child so the leaf being valued is a real, fresh
                // decision node rather than the root itself.
                let child = tree.expand(0, &mut rng).expect("the root has children");
                tree.leaf_value(child, &mut rng)
            };
            assert_eq!(
                leaf_value(LeafValue::Truncated { plies: 0 }, seed).to_bits(),
                leaf_value(LeafValue::Static, seed).to_bits(),
                "seed {seed}: a zero-ply truncation is not a static evaluation"
            );
            assert_eq!(
                leaf_value(LeafValue::Blend { weight: 0.0 }, seed).to_bits(),
                leaf_value(LeafValue::Rollout, seed).to_bits(),
                "seed {seed}: a blend at weight zero is not the playout value"
            );
            // ...and a blend at weight one is the static value.
            assert_eq!(
                leaf_value(LeafValue::Blend { weight: 1.0 }, seed).to_bits(),
                leaf_value(LeafValue::Static, seed).to_bits(),
                "seed {seed}: a blend at weight one is not the static value"
            );
        }
    }

    /// A static leaf value is a `[0, 1]` probability, which is what makes it
    /// commensurable with the playout values the same tree backs up.
    #[test]
    fn every_static_leaf_value_is_a_probability() {
        for seed in 0..6u64 {
            let (state, actions) = mid_game(seed);
            let mut rng = StdRng::seed_from_u64(seed);
            let mut tree = Tree::new(
                state,
                actions,
                Config {
                    leaf: LeafValue::Static,
                    ..Config::default()
                },
                &mut rng,
            );
            for _ in 0..300 {
                tree.simulate(&mut rng);
            }
            for (i, n) in tree.nodes.iter().enumerate() {
                let mean = n.mean();
                assert!(
                    (0.0..=1.0).contains(&mean),
                    "seed {seed}, node {i}: mean {mean} is not a probability"
                );
            }
        }
    }

    /// The mandatory property for anything in this repository that touches
    /// game state: **the value may not depend on which hidden world produced
    /// the state it is looking at.**
    ///
    /// `duels-eval` asserts this of the evaluation itself
    /// (`duels-eval/tests/determinization_invariance.rs`); this asserts it of
    /// the way *this* crate calls it — two unrelated
    /// [`duels_core::Observation::sample_state`] draws from the same real
    /// observation, one tree each, and the static leaf value compared bit for
    /// bit. A discrepancy of any size would mean the leaf was scoring the
    /// sampler's luck.
    ///
    /// Only [`LeafValue::Static`] is checked, and deliberately so:
    /// [`LeafValue::Truncated`]'s and the default [`LeafValue::Blend`]'s
    /// playout walks the determinized layout exactly as
    /// [`LeafValue::Rollout`]'s does — that is perfect-information Monte
    /// Carlo, documented in the crate docs, and not a property this test could
    /// hold. `Blend`'s *static half* is the part this test covers, since it is
    /// `static_from` verbatim.
    ///
    /// # This invariant is *not* relaxed by tracking `duels-eval` live
    ///
    /// Worth being explicit, because the two ideas can be confused. The crate
    /// docs' live-tracking section says two *different builds* of this agent
    /// may score a position differently, and that this is intended. This test
    /// says something else entirely: inside **one** decision, one build, the
    /// value may not move with which hidden world the sampler happened to
    /// draw. That is a hidden-information leak whatever `duels-eval` currently
    /// says, and it stays mandatory.
    #[test]
    fn a_static_leaf_value_is_determinization_invariant() {
        for seed in 0..10u64 {
            let (real, _) = mid_game(seed);
            let obs = real.observation();
            let cfg = Config {
                leaf: LeafValue::Static,
                ..Config::default()
            };

            let sampled = |mix: u64| {
                let mut rng = StdRng::seed_from_u64(seed ^ mix);
                let state = obs.sample_state(&mut rng);
                let actions = engine::legal_actions(&state);
                let mut tree_rng = StdRng::seed_from_u64(7);
                let tree = Tree::new(state, actions, cfg, &mut tree_rng);
                (
                    state,
                    tree.static_from(0).expect("a static leaf has a root"),
                )
            };
            let (a, va) = sampled(0xAAAA_AAAA);
            let (b, vb) = sampled(0x5555_5555);
            assert_eq!(
                a.observation(),
                b.observation(),
                "seed {seed}: the two draws are not publicly identical, so the test is vacuous"
            );
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "seed {seed}: the static leaf value moved with the hidden world: {va} vs {vb}"
            );
        }
    }

    /// **The determinization-invariance property, for this crate's own leaf.**
    /// `docs/conventions.md` requires one for any new logic that touches game
    /// state, and this is the one that covers the default path: the learned
    /// leaf value of a position must not depend on *which* hidden world the
    /// root determinization drew, compared bit-for-bit via `to_bits`.
    ///
    /// `duels-value` holds the same property for the model itself
    /// (`tests/determinization_invariance.rs`); this is the statement one
    /// layer up, about the leaf as the search reaches it. Both are needed —
    /// that one could pass while this crate leaked hidden information in how
    /// it built or handed over the state.
    #[test]
    fn a_learned_leaf_value_is_determinization_invariant() {
        for seed in 0..10u64 {
            let (real, _) = mid_game(seed);
            let obs = real.observation();
            let cfg = Config::default();
            assert!(cfg.leaf.needs_learned_net(), "the default must be learned");

            let sampled = |mix: u64| {
                let mut rng = StdRng::seed_from_u64(seed ^ mix);
                let state = obs.sample_state(&mut rng);
                let actions = engine::legal_actions(&state);
                let mut tree_rng = StdRng::seed_from_u64(7);
                let tree = Tree::new(state, actions, cfg, &mut tree_rng);
                (
                    state,
                    tree.learned_from(0).expect("a learned leaf has a network"),
                )
            };
            let (a, va) = sampled(0xAAAA_AAAA);
            let (b, vb) = sampled(0x5555_5555);
            assert_eq!(
                a.observation(),
                b.observation(),
                "seed {seed}: the two draws are not publicly identical, so the test is vacuous"
            );
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "seed {seed}: the learned leaf value moved with the hidden world: {va} vs {vb}"
            );
        }
    }

    /// Two arenas must agree node for node: kind, statistics, action order and
    /// child wiring alike.
    fn assert_same_arena(new: &Tree, old: &Tree, seed: u64) {
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

    /// **The playout-policy knobs are inert on this crate's default path, and
    /// live on both of its controls.**
    ///
    /// This is not a weakening of `mcts-eval`'s version of this test
    /// (`a_race_variant_actually_grows_a_different_tree`, which asserted the
    /// variant *does* change the default search) — it is the same test
    /// pointing at a fact this crate's default makes true.
    /// [`Config::race`] and [`Config::rollout`] are multipliers on the
    /// **playout policy**, and [`LeafValue::Learned`] runs no playout at all,
    /// so on the default configuration neither knob has anything to act on.
    ///
    /// Worth pinning rather than leaving implicit, because it has a practical
    /// consequence: `mcts-uct`'s `+26` Elo terminal rails
    /// ([`RaceWeights::TIER1_ONLY`]), which `mcts-eval` measured as *additive*
    /// with its blend leaf, cannot compose with this agent's default at all.
    /// Anyone hoping to stack the two has to go through
    /// [`LeafValue::LearnedBlend`], where a playout still exists.
    #[test]
    fn the_playout_policy_knobs_are_inert_without_a_playout() {
        let grow = |cfg: Config, seed: u64| {
            let (state, actions) = mid_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x1234);
            let mut tree = Tree::new(state, actions, cfg, &mut rng);
            for _ in 0..400 {
                tree.simulate(&mut rng);
            }
            (
                tree.nodes.len(),
                tree.nodes
                    .iter()
                    .map(|n| n.value_sum.to_bits())
                    .collect::<Vec<_>>(),
            )
        };

        for seed in 0..8u64 {
            // Inert on the default: the leaf never plays a move out.
            for race in [
                RaceWeights::MEDIUM,
                RaceWeights::TIER1_ONLY,
                RaceWeights::strong(),
            ] {
                assert_eq!(
                    grow(
                        Config {
                            race,
                            ..Config::default()
                        },
                        seed
                    ),
                    grow(Config::default(), seed),
                    "seed {seed}: {} moved a search with no playout in it",
                    race.name()
                );
            }
            assert_eq!(
                grow(
                    Config {
                        rollout: RolloutWeights::UNIFORM,
                        ..Config::default()
                    },
                    seed
                ),
                grow(Config::default(), seed),
                "seed {seed}: the playout policy moved a search with no playout in it"
            );
        }

        // ...and live on both controls, which is what stops the equalities
        // above from being a statement about the knobs rather than about the
        // leaf. Counted rather than asserted per seed: a race multiplier can
        // legitimately fail to change a playout at some individual position.
        for base in [Config::eval_base(), Config::rollout_base()] {
            let differed = (0..8u64)
                .filter(|&seed| {
                    grow(
                        Config {
                            race: RaceWeights::MEDIUM,
                            ..base
                        },
                        seed,
                    ) != grow(base, seed)
                })
                .count();
            assert!(
                differed >= 6,
                "MEDIUM changed nothing in {} of 8 positions on a control arm \
                 that does run a playout",
                8 - differed
            );
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
