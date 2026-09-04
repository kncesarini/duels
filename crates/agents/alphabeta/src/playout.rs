//! A simulation-based leaf estimate: play the position out to a
//! [`GameResult`] under a fixed policy and score what happened.
//!
//! # Why a search agent in this game wants this
//!
//! 7 Wonders Duel scores holistically at the end. Guild majorities are
//! recomputed from the final board, `coins / 3` rounds once at the end, and
//! the whole point of a resource base is builds that happen twenty plies
//! later. A static evaluation five plies from the root therefore judges a
//! position on evidence that is mostly not in yet — and no amount of extra
//! depth fixes that, because the horizon just moves from ply five to ply eight
//! in a seventy-ply game.
//!
//! A playout reaches the *actual* scoring rules instead. It is a noisy
//! estimate — one sample of one policy's continuation — but it estimates the
//! right quantity, and averaging a few dozen of them at a leaf beats any
//! amount of the hand-tuned linear form it replaces. Head-to-head at a
//! matched wall-clock budget, the leaf rework is worth 94 games in 200 more
//! than the static evaluation alone (188W/12L, see the crate docs).
//!
//! # What the measurements said about the policy
//!
//! Two natural-looking refinements were tried and both *lost*, which is worth
//! recording so nobody pays for them twice:
//!
//! - **Reporting the outcome rather than the margin.** `mcts-uct` backs up win
//!   / draw / loss, which is the objective the game actually pays out on, so
//!   [`Metric::Outcome`] ought to be better. At the sample counts a leaf can
//!   afford it is worse — 43% over 100 games against the margin, and clamping
//!   the margin to approximate it is worse again at 39% (one standard error is
//!   about 5 points at that sample size, so neither is a rounding error).
//!   Three distinct values estimated from a few dozen samples simply does not
//!   separate close positions, where a margin does. Both remain available; the
//!   margin is the default.
//! - **A stronger playout policy.** Following [`crate::order::score`] a
//!   fraction of the time (see [`PolicyWeights::greedy`]) makes the simulated
//!   line more plausible and the estimate *worse* — 36% at `greedy = 0.5`, 46%
//!   at `0.2`, over 100 games each. This is the classic Monte-Carlo finding: a
//!   policy that plays well but narrowly stops being a sample of the
//!   position's possibilities. The kind weights are as far as it pays to go.
//!
//! # Determinism
//!
//! The RNG is seeded from a caller-supplied key, and the search passes the
//! position's own hash ([`crate::tt::state_key`]) rather than a stream carried
//! down the tree. That makes the leaf value a pure function of the position,
//! which the search relies on in three places: `Budget::Nodes` stays
//! bit-for-bit reproducible, the transposition table stays sound (a position
//! re-reached at another depth evaluates the same), and
//! `search::tests::pruning_agrees_with_unpruned_expectimax` can still compare
//! the pruned search against a naive reference.
//!
//! [`crate::Config::rollout_common_seed`] passes a per-iteration constant
//! instead, which keeps the purity but makes sibling leaves share their
//! simulated luck — see that field for what it is for.

use duels_core::{engine, scoring, Action, GameResult, GameState, Player};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Value of a playout that ended in an instant win, in victory points.
///
/// Military and scientific supremacy stop the game before anything is scored,
/// so there is no victory-point difference to report. The number only has to
/// be decisively larger than a realistic civilian margin without approaching
/// [`crate::eval::EVAL_CLAMP`], let alone `search::MATE`: a *simulated* win is
/// not a proven one.
pub const SUPREMACY_VALUE: f64 = 30.0;

/// What a finished playout reports.
///
/// [`Metric::Margin`] makes the search maximise the expected victory-point
/// difference; [`Metric::Outcome`] makes it maximise the probability of
/// winning. The second is the objective the game actually pays out on — a
/// four-point win and a thirty-point win are the same result — and it is what
/// `duels-agent-mcts-uct` backs up, so it is the one that ought to win.
/// It does not, at the sample counts a leaf can afford: see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    /// The victory-point difference the playout ended on, clamped.
    ///
    /// The clamp is the dial between the two objectives: a large one leaves a
    /// thirty-point win worth seven times a four-point win, a small one makes
    /// every comfortable win worth the same and turns the average into a
    /// win-rate estimate with the resolution of a margin. Measured, a
    /// generous clamp is better; see the module docs.
    Margin {
        /// Margins beyond this are all worth the same.
        clamp: f64,
    },
    /// Win, draw or loss, scaled to `+scale`, `0`, `-scale` so it composes
    /// with the static evaluation's victory-point units.
    Outcome {
        /// Victory-point-equivalent value of a won playout.
        scale: f64,
    },
}

impl Metric {
    /// The default: a margin, clamped generously enough that only a freak
    /// simulation is affected.
    pub const MARGIN: Metric = Metric::Margin { clamp: 40.0 };
}

/// Relative weights for the kinds of move a playout makes.
///
/// Uniform-random play discards roughly a third of the cards it touches, which
/// is far worse than any real line and makes a uniform playout a very weak
/// signal. These are the same defaults `duels-agent-mcts-uct` found worked for
/// its rollouts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyWeights {
    /// Weight for [`Action::Build`].
    pub build: f64,
    /// Weight for [`Action::BuildWonder`].
    pub wonder: f64,
    /// Weight for [`Action::Discard`].
    pub discard: f64,
    /// Probability of playing whichever move [`crate::order::score`] rates
    /// highest instead of drawing from the weights above. `0.0`, the default,
    /// is the plain kind-weighted policy, and is what measured best — see the
    /// module docs.
    pub greedy: f64,
}

impl PolicyWeights {
    /// These weights, but following [`crate::order::score`] a `greedy`
    /// fraction of the time. Measured worse than `0.0`; see the module docs.
    pub const fn with_greedy(mut self, greedy: f64) -> Self {
        self.greedy = greedy;
        self
    }

    /// Every kind equally likely: the plain uniform-random playout policy.
    pub const UNIFORM: PolicyWeights = PolicyWeights {
        build: 1.0,
        wonder: 1.0,
        discard: 1.0,
        greedy: 0.0,
    };

    /// The default bias: prefer putting cards into the city, take wonders
    /// readily, discard only when little else is on offer.
    pub const BIASED: PolicyWeights = PolicyWeights {
        build: 4.0,
        wonder: 2.0,
        discard: 1.0,
        greedy: 0.0,
    };

    #[inline]
    fn weight(&self, action: Action) -> f64 {
        match action {
            Action::Build { .. } => self.build,
            Action::BuildWonder { .. } => self.wonder,
            Action::Discard { .. } => self.discard,
            // Effect choices (progress tokens, Mausoleum, destroy, first
            // player) are picked uniformly: there is no cheap ordering over
            // them that is obviously right.
            _ => 1.0,
        }
    }

    /// Whether the kind weights are all equal, whatever `greedy` is.
    #[inline]
    fn kinds_are_flat(&self) -> bool {
        self.build == self.wonder && self.wonder == self.discard
    }
}

impl Default for PolicyWeights {
    fn default() -> Self {
        PolicyWeights::BIASED
    }
}

/// Safety net on playout length. The game is finite, so hitting this means a
/// bug elsewhere; stopping and scoring is the least misleading response.
const MAX_PLIES: u32 = 200;

/// Mean value of `n` playouts of `state`, in victory points for `me`.
///
/// `key` is the position's hash, used to seed the playouts (see the module
/// docs). `buf` is a scratch move list, reused across calls.
pub fn estimate(
    state: &GameState,
    me: Player,
    key: u64,
    n: u32,
    policy: &PolicyWeights,
    metric: Metric,
    buf: &mut Vec<Action>,
) -> f64 {
    debug_assert!(n > 0);
    let mut acc = 0.0;
    for i in 0..n {
        let mut rng =
            StdRng::seed_from_u64(key ^ (u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        let mut sim = *state;
        acc += one(&mut sim, me, policy, metric, buf, &mut rng);
    }
    acc / f64::from(n)
}

/// Play `state` out under `policy` and return the result in victory points
/// for `me`.
fn one(
    state: &mut GameState,
    me: Player,
    policy: &PolicyWeights,
    metric: Metric,
    buf: &mut Vec<Action>,
    rng: &mut StdRng,
) -> f64 {
    for _ in 0..MAX_PLIES {
        if state.is_over() {
            break;
        }
        engine::legal_actions_into(state, buf);
        if buf.is_empty() {
            break;
        }
        let action = pick(state, policy, buf, rng);
        // The state carries a determinized layout — sampled once at the root
        // and kept publicly consistent by every forced reveal the search made
        // on the way down — so `apply_unchecked` may read it for reveals. This
        // is one determinized continuation, exactly as in perfect-information
        // Monte Carlo search; no *decision* the agent actually makes is taken
        // on the strength of a card it should not know.
        engine::apply_unchecked(state, action, rng);
    }
    let result = state
        .result()
        .unwrap_or_else(|| scoring::civilian_result(state));
    value(state, me, result, metric)
}

/// Score a finished playout from `me`'s point of view.
fn value(state: &GameState, me: Player, result: GameResult, metric: Metric) -> f64 {
    match metric {
        Metric::Outcome { scale } => match result.winner() {
            None => 0.0,
            Some(w) if w == me => scale,
            Some(_) => -scale,
        },
        Metric::Margin { clamp } => {
            let v = if result.is_instant() {
                match result.winner() {
                    Some(w) if w == me => SUPREMACY_VALUE,
                    _ => -SUPREMACY_VALUE,
                }
            } else {
                let s = scoring::score(state);
                f64::from(s[me.index()].total) - f64::from(s[me.other().index()].total)
            };
            v.clamp(-clamp, clamp)
        }
    }
}

/// Pick one action according to `policy`.
fn pick(state: &GameState, policy: &PolicyWeights, legal: &[Action], rng: &mut StdRng) -> Action {
    debug_assert!(!legal.is_empty());
    if legal.len() == 1 {
        return legal[0];
    }
    if policy.greedy > 0.0 && rng.gen_bool(policy.greedy.min(1.0)) {
        let mut best = legal[0];
        let mut best_score = i32::MIN;
        for &a in legal {
            let s = crate::order::score(state, a);
            if s > best_score {
                best_score = s;
                best = a;
            }
        }
        return best;
    }
    if policy.kinds_are_flat() {
        return legal[rng.gen_range(0..legal.len())];
    }
    let total: f64 = legal.iter().map(|&a| policy.weight(a)).sum();
    let mut r = rng.gen_range(0.0..total);
    for &a in legal {
        r -= policy.weight(a);
        if r < 0.0 {
            return a;
        }
    }
    legal[legal.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::testing::StateBuilder;

    /// Shorthand, so the argument lists below stay on one line.
    const PW: PolicyWeights = PolicyWeights::BIASED;

    #[test]
    fn a_playout_always_reaches_a_finished_game() {
        let mut buf = Vec::new();
        for seed in 0..10u64 {
            let mut st = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed);
            let v = one(
                &mut st,
                Player::One,
                &PolicyWeights::BIASED,
                Metric::MARGIN,
                &mut buf,
                &mut rng,
            );
            assert!(st.is_over(), "seed {seed} did not finish");
            assert!(v.abs() <= 40.0);
        }
    }

    #[test]
    fn the_estimate_is_a_pure_function_of_the_position() {
        let mut buf = Vec::new();
        let st = engine::new_game(3);
        let key = crate::tt::state_key(&st);
        let a = estimate(&st, Player::One, key, 4, &PW, Metric::MARGIN, &mut buf);
        let b = estimate(&st, Player::One, key, 4, &PW, Metric::MARGIN, &mut buf);
        assert_eq!(a, b);
    }

    #[test]
    fn the_estimate_is_zero_sum() {
        let mut buf = Vec::new();
        for seed in 0..5u64 {
            let st = engine::new_game(seed);
            let key = crate::tt::state_key(&st);
            let a = estimate(&st, Player::One, key, 3, &PW, Metric::MARGIN, &mut buf);
            let b = estimate(&st, Player::Two, key, 3, &PW, Metric::MARGIN, &mut buf);
            assert!((a + b).abs() < 1e-9, "seed {seed}: {a} + {b}");
        }
    }

    #[test]
    fn a_hopeless_position_estimates_negative() {
        // Player Two is 40 victory points up with Age III nearly done; no
        // playout policy recovers that.
        let mut buf = Vec::new();
        let st = StateBuilder::new()
            .age(3)
            .built(
                Player::Two,
                &["palace", "town-hall", "pantheon", "senate", "obelisk"],
            )
            .open_slots(&[(18, "arena"), (19, "port")])
            .build();
        let key = crate::tt::state_key(&st);
        let v = estimate(&st, Player::One, key, 8, &PW, Metric::MARGIN, &mut buf);
        assert!(v < -5.0, "expected a clearly lost playout value, got {v}");
    }

    #[test]
    fn a_biased_policy_builds_more_than_it_discards() {
        // The premise of the default weights. Count what each policy actually
        // does over a few whole games.
        let mut buf = Vec::new();
        let mut counts = [[0u32; 2]; 2];
        for (i, policy) in [PolicyWeights::UNIFORM, PolicyWeights::BIASED]
            .iter()
            .enumerate()
        {
            for seed in 0..6u64 {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut st = engine::new_game(seed);
                while !st.is_over() {
                    engine::legal_actions_into(&st, &mut buf);
                    if buf.is_empty() {
                        break;
                    }
                    let a = pick(&st, policy, &buf, &mut rng);
                    match a {
                        Action::Build { .. } => counts[i][0] += 1,
                        Action::Discard { .. } => counts[i][1] += 1,
                        _ => {}
                    }
                    engine::apply_unchecked(&mut st, a, &mut rng);
                }
            }
        }
        let uniform_ratio = f64::from(counts[0][0]) / f64::from(counts[0][1].max(1));
        let biased_ratio = f64::from(counts[1][0]) / f64::from(counts[1][1].max(1));
        assert!(
            biased_ratio > uniform_ratio,
            "uniform built/discarded {uniform_ratio:.2}, biased {biased_ratio:.2}"
        );
    }
}
