//! Logistic-Elo rating-difference estimate with a 95% confidence interval.
//!
//! # Method
//!
//! Treat each decisive game as a Bernoulli trial and each draw as half a win
//! for each side — the standard chess-rating convention for turning a
//! win/loss/draw record into an expected score. Under the logistic model,
//! `P(A beats B) = 1 / (1 + 10^(-d/400))` where `d = rating(A) - rating(B)`.
//! Writing `k = ln(10)/400` so that `P = sigmoid(k*d)`, the log-likelihood of
//! `d` given `W` effective wins and `L` effective losses (`W = wins +
//! draws/2`, `L = losses + draws/2`, `N = W + L`) is
//!
//! ```text
//! ℓ(d) = W·ln(sigmoid(k·d)) + L·ln(1 − sigmoid(k·d))
//! ```
//!
//! which is strictly concave in `d`, so it has a unique maximum. We find it
//! with Newton's method (`d ← d − ℓ'(d)/ℓ''(d)`), which for this
//! one-parameter model converges in a couple of steps to the same point a
//! closed form would give (`sigmoid(k·d̂) = W/N`) — Newton's method is used
//! anyway, per the brief, because it is the approach that generalizes if
//! this is ever extended to fit more than two agents at once, where no
//! closed form exists.
//!
//! To avoid an infinite/undefined estimate when one side sweeps every game
//! (`W = 0` or `W = N`), we add a weak symmetric prior: one pseudo-game
//! worth of a 50/50 split (`W' = W + 0.5`, `L' = L + 0.5`), a standard
//! continuity correction (equivalent to a `Beta(0.5, 0.5)`-flavored Bayesian
//! prior on the win probability, hence "Bayesian/logistic Elo" — this is the
//! Bayesian ingredient the brief asks for, kept intentionally simple rather
//! than a full posterior). It shrinks extreme empirical rates towards 0 Elo,
//! the conservative direction for a small, lopsided sample.
//!
//! # Confidence interval
//!
//! The 95% CI comes from the observed Fisher information at the MLE. For
//! this one-parameter exponential-family model the observed and expected
//! information coincide: `I(d) = k² · N' · p(d) · (1 − p(d))` (`N'` is the
//! prior-inflated total), so `Var(d̂) ≈ 1 / I(d̂)` and the interval is
//! `d̂ ± z · sqrt(Var(d̂))` with `z = 1.96` for 95%. This is the standard
//! asymptotic-normal MLE interval — a large-sample approximation. It is
//! known to under-cover when `N` is small or the true rate sits near 0 or 1,
//! which `tests::confidence_interval_has_roughly_nominal_coverage` checks
//! empirically rather than assuming.
//!
//! # Anchor
//!
//! Elo is only meaningful as a *difference* between two ratings, so one side
//! must be pinned. We anchor the reference/baseline agent (agent B in
//! [`fit_elo`]) at [`ANCHOR_ELO`] = 0. A caller building a leaderboard with a
//! friendlier baseline (e.g. "random = 1000") can just add a constant offset
//! to every reported number; the statistics themselves only ever depend on
//! the difference.

use serde::{Deserialize, Serialize};

/// `ln(10) / 400`: converts an Elo-point difference to a logit.
pub(crate) const ELO_TO_LOGIT: f64 = std::f64::consts::LN_10 / 400.0;

/// The rating pinned to the reference/anchor agent. See module docs.
pub const ANCHOR_ELO: f64 = 0.0;

/// The 97.5th percentile of the standard normal distribution, i.e. the `z`
/// for a two-sided 95% confidence interval.
const Z_95: f64 = 1.959_963_984_540_054;

pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A fitted logistic-Elo rating difference between a candidate agent and an
/// anchored reference agent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EloEstimate {
    /// Rating of the reference/anchor agent. Always [`ANCHOR_ELO`].
    pub anchor_elo: f64,
    /// Estimated rating of the candidate agent (`anchor_elo + rating_diff`).
    pub candidate_elo: f64,
    /// `candidate_elo - anchor_elo`, the quantity actually being estimated.
    pub rating_diff: f64,
    /// Lower bound of the 95% CI on `rating_diff`.
    pub diff_ci_low: f64,
    /// Upper bound of the 95% CI on `rating_diff`.
    pub diff_ci_high: f64,
    /// Games the candidate won, from the counts passed to [`fit_elo`].
    pub wins: u32,
    /// Games the candidate lost.
    pub losses: u32,
    /// Games that were drawn.
    pub draws: u32,
}

/// Fit a logistic-Elo rating difference for a candidate relative to the
/// anchored reference, from `(wins, losses, draws)` counts, all from the
/// candidate's perspective. See the module docs for the method.
pub fn fit_elo(wins: u32, losses: u32, draws: u32) -> EloEstimate {
    // One pseudo-game at 50/50 as a weak prior (see module docs) so the
    // estimate and its interval stay finite even after a perfect sweep.
    let w = wins as f64 + draws as f64 * 0.5 + 0.5;
    let l = losses as f64 + draws as f64 * 0.5 + 0.5;
    let n = w + l;

    // Newton's method on the root of ℓ'(d) = k·(W − N·sigmoid(k·d)).
    let mut d = 0.0f64;
    for _ in 0..50 {
        let p = sigmoid(ELO_TO_LOGIT * d);
        let grad = ELO_TO_LOGIT * (w - n * p);
        let hess = -ELO_TO_LOGIT * ELO_TO_LOGIT * n * p * (1.0 - p);
        if hess.abs() < 1e-12 {
            break;
        }
        let step = grad / hess;
        d -= step;
        if step.abs() < 1e-9 {
            break;
        }
    }

    let p = sigmoid(ELO_TO_LOGIT * d);
    let information = ELO_TO_LOGIT * ELO_TO_LOGIT * n * p * (1.0 - p);
    let se = if information > 0.0 {
        1.0 / information.sqrt()
    } else {
        f64::INFINITY
    };

    EloEstimate {
        anchor_elo: ANCHOR_ELO,
        candidate_elo: ANCHOR_ELO + d,
        rating_diff: d,
        diff_ci_low: d - Z_95 * se,
        diff_ci_high: d + Z_95 * se,
        wins,
        losses,
        draws,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The Elo-point difference that exactly produces win probability `p`
    /// under the same logistic model `fit_elo` assumes — i.e. the "true"
    /// rating difference a synthetic Bernoulli(`p`) generator corresponds to.
    fn true_diff_for(p: f64) -> f64 {
        (p / (1.0 - p)).ln() / ELO_TO_LOGIT
    }

    /// Simulate `n` decisive (no-draw) games at true win probability `p`.
    fn simulate(p: f64, n: u32, rng: &mut StdRng) -> (u32, u32) {
        let mut wins = 0;
        let mut losses = 0;
        for _ in 0..n {
            if rng.gen_bool(p) {
                wins += 1;
            } else {
                losses += 1;
            }
        }
        (wins, losses)
    }

    #[test]
    fn recovers_zero_diff_for_a_fair_coin() {
        let mut rng = StdRng::seed_from_u64(1);
        let (wins, losses) = simulate(0.5, 4000, &mut rng);
        let est = fit_elo(wins, losses, 0);
        assert!(
            est.rating_diff.abs() < 15.0,
            "expected ~0 elo from a fair coin, got {}",
            est.rating_diff
        );
        assert!(est.diff_ci_low < 0.0 && est.diff_ci_high > 0.0);
    }

    #[test]
    fn recovers_a_known_positive_rating_gap() {
        let true_elo = 100.0;
        let p = sigmoid(ELO_TO_LOGIT * true_elo);
        let mut rng = StdRng::seed_from_u64(2);
        let (wins, losses) = simulate(p, 4000, &mut rng);
        let est = fit_elo(wins, losses, 0);
        assert!(
            (est.rating_diff - true_elo).abs() < 15.0,
            "expected close to {true_elo} elo, got {}",
            est.rating_diff
        );
    }

    #[test]
    fn draws_count_as_half_a_win_each_side() {
        // All draws: no evidence either way, diff should stay ~0.
        let est = fit_elo(0, 0, 1000);
        assert!(est.rating_diff.abs() < 1.0);
    }

    #[test]
    fn a_perfect_sweep_stays_finite_thanks_to_the_prior() {
        let est = fit_elo(50, 0, 0);
        assert!(est.rating_diff.is_finite());
        assert!(est.diff_ci_low.is_finite());
        assert!(est.rating_diff > 0.0);
    }

    #[test]
    fn more_games_narrows_the_confidence_interval() {
        let mut rng = StdRng::seed_from_u64(3);
        let (w_small, l_small) = simulate(0.6, 100, &mut rng);
        let (w_large, l_large) = simulate(0.6, 100 + 4000, &mut rng);
        let small = fit_elo(w_small, l_small, 0);
        let large = fit_elo(w_large, l_large, 0);
        let width_small = small.diff_ci_high - small.diff_ci_low;
        let width_large = large.diff_ci_high - large.diff_ci_low;
        assert!(
            width_large < width_small,
            "{width_large} should be narrower than {width_small}"
        );
    }

    /// Repeatedly simulate games at a known true rating difference and check
    /// that the nominal-95% interval actually contains the true value close
    /// to 95% of the time — the point of the "expected coverage" check the
    /// brief asks for, not just "the interval prints a number".
    #[test]
    fn confidence_interval_has_roughly_nominal_coverage() {
        let true_elo = 40.0;
        let p = sigmoid(ELO_TO_LOGIT * true_elo);
        let trials = 400;
        let games_per_trial = 300;
        let mut covered = 0;
        for seed in 0..trials {
            let mut rng = StdRng::seed_from_u64(1_000_000 + seed);
            let (wins, losses) = simulate(p, games_per_trial, &mut rng);
            let est = fit_elo(wins, losses, 0);
            if est.diff_ci_low <= true_elo && true_elo <= est.diff_ci_high {
                covered += 1;
            }
        }
        let coverage = covered as f64 / trials as f64;
        // Nominal is 95%; the asymptotic-normal approximation is imperfect,
        // so allow a wide-ish band rather than demanding exactly 0.95 — this
        // is a calibration smoke test, not a proof.
        assert!(
            (0.88..=1.0).contains(&coverage),
            "expected roughly-95% coverage, got {coverage} ({covered}/{trials})"
        );
    }

    #[test]
    fn true_diff_for_is_the_inverse_of_sigmoid_at_elo_to_logit() {
        for elo in [-200.0, -50.0, 0.0, 50.0, 200.0_f64] {
            let p = sigmoid(ELO_TO_LOGIT * elo);
            assert!((true_diff_for(p) - elo).abs() < 1e-6);
        }
    }
}
