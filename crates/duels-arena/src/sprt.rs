//! Sequential Probability Ratio Test (SPRT) for candidate-vs-reference
//! testing, in the style of chess-engine testing frameworks (Fishtest et
//! al.).
//!
//! # Method
//!
//! Given two Elo hypotheses — `H0`: the candidate is `elo0` relative to the
//! reference, `H1`: the candidate is `elo1` — convert each to an expected
//! score via the same logistic model [`crate::elo`] uses: `p0 =
//! sigmoid(k·elo0)`, `p1 = sigmoid(k·elo1)` with `k = ln(10)/400`. Treat each
//! decisive game as a Bernoulli trial and each draw as half a win for each
//! side (same convention as [`crate::elo`]), giving effective win/loss
//! counts `W = wins + draws/2`, `L = losses + draws/2`. The running
//! log-likelihood ratio of `H1` over `H0` is then Wald's classic SPRT
//! statistic for a Bernoulli parameter:
//!
//! ```text
//! LLR = W·ln(p1/p0) + L·ln((1−p1)/(1−p0))
//! ```
//!
//! and the standard Wald decision boundaries for error rates `alpha`
//! (probability of accepting `H1` when `H0` is true) and `beta` (probability
//! of accepting `H0` when `H1` is true) are:
//!
//! ```text
//! lower = ln(beta / (1 − alpha))     — LLR ≤ lower ⇒ accept H0
//! upper = ln((1 − beta) / alpha)     — LLR ≥ upper ⇒ accept H1
//! ```
//!
//! with anything in between meaning "keep playing games".
//!
//! This is a deliberate simplification of what Fishtest actually runs (their
//! GSPRT models the draw rate as its own free parameter — "BayesElo" with a
//! `drawelo` — rather than splitting each draw into half a win and half a
//! loss). The brief for this crate calls for something "correct, [not]
//! fancy"; splitting draws is the same simplification [`crate::elo`] already
//! makes for the point estimate, so the two statistics stay consistent with
//! each other. A future revision that wants the exact Fishtest pentanomial
//! model would replace the body of [`sprt`] without changing its signature —
//! it already takes only accumulated counts, so it can be called
//! incrementally as games complete rather than only once at the end.

use serde::{Deserialize, Serialize};

use crate::elo::{sigmoid, ELO_TO_LOGIT};

/// The two Elo hypotheses and error-rate tolerances for one SPRT run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SprtParams {
    /// H0: the candidate is this many Elo relative to the reference (often
    /// 0, "no better than the reference", or a small negative "not worse
    /// than a regression margin").
    pub elo0: f64,
    /// H1: the candidate is this many Elo relative to the reference.
    pub elo1: f64,
    /// Probability of accepting H1 when H0 is actually true (false positive).
    pub alpha: f64,
    /// Probability of accepting H0 when H1 is actually true (false negative).
    pub beta: f64,
}

impl Default for SprtParams {
    /// `elo0 = 0`, `elo1 = 5`, `alpha = beta = 0.05` — a typical "is this
    /// candidate any better at all than the reference" Fishtest-style test.
    fn default() -> Self {
        Self {
            elo0: 0.0,
            elo1: 5.0,
            alpha: 0.05,
            beta: 0.05,
        }
    }
}

/// The three-way decision an SPRT run can reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SprtDecision {
    /// Stop: evidence favors H0 (candidate is not meaningfully better).
    AcceptH0,
    /// Stop: evidence favors H1 (candidate is meaningfully better).
    AcceptH1,
    /// Inconclusive — play more games.
    Continue,
}

/// The outcome of evaluating the SPRT at one point in an (ongoing) match.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SprtResult {
    /// The running log-likelihood ratio of H1 over H0.
    pub llr: f64,
    /// `LLR ≤ this ⇒ accept H0`.
    pub lower_bound: f64,
    /// `LLR ≥ this ⇒ accept H1`.
    pub upper_bound: f64,
    /// The resulting decision.
    pub decision: SprtDecision,
}

/// Evaluate the SPRT against accumulated `(wins, losses, draws)` counts
/// (candidate's perspective) under `params`. Pure function of the counts, so
/// it can be called after every game completes, not just once at the end.
pub fn sprt(wins: u32, losses: u32, draws: u32, params: &SprtParams) -> SprtResult {
    let w = wins as f64 + draws as f64 * 0.5;
    let l = losses as f64 + draws as f64 * 0.5;

    let p0 = sigmoid(ELO_TO_LOGIT * params.elo0);
    let p1 = sigmoid(ELO_TO_LOGIT * params.elo1);

    let llr = w * (p1.ln() - p0.ln()) + l * ((1.0 - p1).ln() - (1.0 - p0).ln());

    let lower_bound = (params.beta / (1.0 - params.alpha)).ln();
    let upper_bound = ((1.0 - params.beta) / params.alpha).ln();

    let decision = if llr >= upper_bound {
        SprtDecision::AcceptH1
    } else if llr <= lower_bound {
        SprtDecision::AcceptH0
    } else {
        SprtDecision::Continue
    };

    SprtResult {
        llr,
        lower_bound,
        upper_bound,
        decision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn elo_to_p(elo: f64) -> f64 {
        sigmoid(ELO_TO_LOGIT * elo)
    }

    /// Play decisive games one at a time at true win probability `p`,
    /// evaluating the SPRT after every game, and return the first decision
    /// that isn't `Continue` along with how many games it took — or `None`
    /// if it never resolves within `max_games`.
    fn run_until_decided(
        p: f64,
        params: &SprtParams,
        max_games: u32,
        seed: u64,
    ) -> Option<(SprtDecision, u32)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut wins = 0u32;
        let mut losses = 0u32;
        for n in 1..=max_games {
            if rng.gen_bool(p) {
                wins += 1;
            } else {
                losses += 1;
            }
            let result = sprt(wins, losses, 0, params);
            if result.decision != SprtDecision::Continue {
                return Some((result.decision, n));
            }
        }
        None
    }

    #[test]
    fn a_clearly_stronger_candidate_is_accepted_as_h1() {
        let params = SprtParams::default(); // elo0=0, elo1=5
                                            // True strength far above elo1: should resolve quickly and clearly.
        let p = elo_to_p(60.0);
        let (decision, games) =
            run_until_decided(p, &params, 5_000, 7).expect("should resolve within 5000 games");
        assert_eq!(decision, SprtDecision::AcceptH1);
        assert!(games < 5_000);
    }

    #[test]
    fn a_clearly_weaker_candidate_is_accepted_as_h0() {
        let params = SprtParams::default();
        // True strength well below elo0 (even negative): should resolve to H0.
        let p = elo_to_p(-60.0);
        let (decision, games) =
            run_until_decided(p, &params, 5_000, 11).expect("should resolve within 5000 games");
        assert_eq!(decision, SprtDecision::AcceptH0);
        assert!(games < 5_000);
    }

    #[test]
    fn an_ambiguous_middling_win_rate_stays_undecided_for_a_while() {
        let params = SprtParams::default(); // elo0=0, elo1=5
                                            // Right at the midpoint of the two hypotheses: with only a handful
                                            // of games, there should not yet be enough evidence for either side.
        let mid_elo = (params.elo0 + params.elo1) / 2.0;
        let p = elo_to_p(mid_elo);
        let mut rng = StdRng::seed_from_u64(21);
        let mut wins = 0u32;
        let mut losses = 0u32;
        for _ in 0..20 {
            if rng.gen_bool(p) {
                wins += 1;
            } else {
                losses += 1;
            }
            let result = sprt(wins, losses, 0, &params);
            assert_eq!(
                result.decision,
                SprtDecision::Continue,
                "expected still-undecided after {}+{} games at the midpoint elo",
                wins,
                losses
            );
        }
    }

    #[test]
    fn zero_games_is_always_continue() {
        let params = SprtParams::default();
        let result = sprt(0, 0, 0, &params);
        assert_eq!(result.llr, 0.0);
        assert_eq!(result.decision, SprtDecision::Continue);
    }

    #[test]
    fn draws_contribute_half_a_win_and_half_a_loss() {
        let params = SprtParams::default();
        let from_draws = sprt(0, 0, 100, &params);
        let from_split = sprt(50, 50, 0, &params);
        assert!((from_draws.llr - from_split.llr).abs() < 1e-9);
    }

    #[test]
    fn tighter_error_rates_widen_the_decision_boundaries() {
        let loose = SprtParams {
            alpha: 0.05,
            beta: 0.05,
            ..SprtParams::default()
        };
        let tight = SprtParams {
            alpha: 0.01,
            beta: 0.01,
            ..SprtParams::default()
        };
        let r_loose = sprt(10, 5, 0, &loose);
        let r_tight = sprt(10, 5, 0, &tight);
        assert!(r_tight.upper_bound > r_loose.upper_bound);
        assert!(r_tight.lower_bound < r_loose.lower_bound);
    }
}
