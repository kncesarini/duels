//! The playout policy: a fast, self-contained default policy used from the
//! newly expanded leaf down to a `GameResult`.
//!
//! Deliberately cheap. Rollouts are where nearly all of the search's time
//! goes, so the policy is a single weighted draw over the legal actions with
//! no state evaluation at all. The only domain knowledge is a preference
//! ordering over *kinds* of move: uniform-random play discards roughly a
//! third of the cards it touches, which is far worse than any human line, and
//! that noise floor is what makes pure-random playouts a weak signal. Weights
//! of 1 across the board reproduce the uniform baseline exactly and are kept
//! as a configuration option so the effect can be measured.

use duels_core::engine;
use duels_core::{Action, GameResult, GameState};
use rand::rngs::StdRng;
use rand::Rng;

/// Relative weights for the kinds of move a playout can make.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RolloutWeights {
    /// Weight for [`Action::Build`].
    pub build: f64,
    /// Weight for [`Action::BuildWonder`].
    pub wonder: f64,
    /// Weight for [`Action::Discard`].
    pub discard: f64,
}

impl RolloutWeights {
    /// Every kind equally likely: the plain uniform-random playout policy.
    pub const UNIFORM: RolloutWeights = RolloutWeights {
        build: 1.0,
        wonder: 1.0,
        discard: 1.0,
    };

    /// The default bias: prefer putting cards into the city, take wonders
    /// readily, discard only when little else looks available.
    pub const BIASED: RolloutWeights = RolloutWeights {
        build: 4.0,
        wonder: 2.0,
        discard: 1.0,
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

    /// Whether this is the uniform policy.
    #[inline]
    pub fn is_uniform(&self) -> bool {
        *self == RolloutWeights::UNIFORM
    }
}

impl Default for RolloutWeights {
    fn default() -> Self {
        RolloutWeights::BIASED
    }
}

/// Pick one action according to `weights`.
pub(crate) fn pick(weights: &RolloutWeights, legal: &[Action], rng: &mut StdRng) -> Action {
    debug_assert!(!legal.is_empty());
    if weights.is_uniform() || legal.len() == 1 {
        return legal[rng.gen_range(0..legal.len())];
    }
    let total: f64 = legal.iter().map(|&a| weights.weight(a)).sum();
    let mut r = rng.gen_range(0.0..total);
    for &a in legal {
        r -= weights.weight(a);
        if r < 0.0 {
            return a;
        }
    }
    legal[legal.len() - 1]
}

/// Play `state` out to the end and return the result.
///
/// `max_plies` is a safety net, not a rule: the game is finite, so hitting it
/// means a bug elsewhere. Rather than spinning forever the playout stops and
/// scores the position as if Age III had just ended, which is the least
/// misleading thing available.
pub(crate) fn play_out(
    state: &mut GameState,
    weights: &RolloutWeights,
    buf: &mut Vec<Action>,
    rng: &mut StdRng,
    max_plies: u32,
) -> GameResult {
    for _ in 0..max_plies {
        engine::legal_actions_into(state, buf);
        if buf.is_empty() {
            break;
        }
        let action = pick(weights, buf, rng);
        // The state carries a determinized layout sampled at the root (and
        // kept publicly consistent by every forced reveal on the way down),
        // so `apply_unchecked` is free to read it for reveals: this is one
        // determinized playout, exactly as in perfect-information Monte Carlo
        // search, and it never leaks information into a *decision* — those
        // all happen at tree nodes reached through `apply_with_outcome`.
        engine::apply_unchecked(state, action, rng);
    }
    state
        .result()
        .unwrap_or_else(|| duels_core::scoring::civilian_result(state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::Player;
    use rand::SeedableRng;

    #[test]
    fn uniform_weights_are_reported_as_uniform() {
        assert!(RolloutWeights::UNIFORM.is_uniform());
        assert!(!RolloutWeights::BIASED.is_uniform());
    }

    #[test]
    fn weighted_picking_respects_the_weights() {
        let legal = [
            Action::Build { slot: 1 },
            Action::Discard { slot: 2 },
            Action::BuildWonder {
                slot: 3,
                wonder: duels_core::data::WonderId::from_index(0),
            },
        ];
        let w = RolloutWeights::BIASED;
        let mut rng = StdRng::seed_from_u64(1);
        let mut counts = [0u32; 3];
        const N: u32 = 70_000;
        for _ in 0..N {
            let a = pick(&w, &legal, &mut rng);
            let i = legal.iter().position(|&x| x == a).unwrap();
            counts[i] += 1;
        }
        let total = w.build + w.discard + w.wonder;
        for (i, expected) in [w.build, w.discard, w.wonder].iter().enumerate() {
            let got = f64::from(counts[i]) / f64::from(N);
            assert!(
                (got - expected / total).abs() < 0.01,
                "action {i}: got {got}, expected {}",
                expected / total
            );
        }
    }

    #[test]
    fn a_playout_always_reaches_a_result() {
        let mut buf = Vec::new();
        for seed in 0..20u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut state = engine::new_game(seed);
            let result = play_out(
                &mut state,
                &RolloutWeights::BIASED,
                &mut buf,
                &mut rng,
                1_000,
            );
            assert!(state.result().is_some(), "seed {seed} did not finish");
            assert_eq!(state.result().unwrap(), result);
            assert!(matches!(
                result.winner(),
                Some(Player::One) | Some(Player::Two) | None
            ));
        }
    }
}
