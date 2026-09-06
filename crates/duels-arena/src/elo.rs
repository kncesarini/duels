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
//!
//! # Fitting more than two agents at once
//!
//! [`fit_elo`] answers "how much stronger is A than B" from one head-to-head
//! record. A leaderboard over `n` agents has `C(n, 2)` such records and wants
//! *one* consistent rating per agent, which is a different (and strictly
//! better-conditioned) estimation problem: A's rating should be informed by
//! every game A played, and also — indirectly, through their shared
//! opponents — by games A never played at all. [`fit_joint_elo`] does that
//! properly rather than anchoring each agent independently against one chosen
//! reference and throwing the rest of the round-robin away. See its docs for
//! the method.

use std::collections::BTreeMap;

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

/// One head-to-head record between two named agents, as fed to
/// [`fit_joint_elo`]. `wins`/`losses`/`draws` are always from `agent_a`'s
/// perspective (the same convention [`fit_elo`] and
/// [`crate::match_runner::tally`] use).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairwiseRecord {
    pub agent_a: String,
    pub agent_b: String,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

/// One agent's row in a fitted [`JointEloTable`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointEloRating {
    /// The agent this rating is for.
    pub agent: String,
    /// Fitted rating, on the scale set by the anchor passed to
    /// [`fit_joint_elo`].
    pub elo: f64,
    /// Lower bound of the 95% CI on `elo − anchor_elo`, shifted onto the same
    /// scale as `elo`. Exactly zero-width for the anchor itself, which is
    /// pinned by definition rather than estimated.
    pub elo_ci_low: f64,
    /// Upper bound of the same interval.
    pub elo_ci_high: f64,
    /// Games this agent played across every pairing it appeared in.
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

/// A joint multi-agent Elo fit: one rating per agent, all estimated together
/// from every pairwise record at once. See [`fit_joint_elo`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointEloTable {
    /// The agent whose rating was pinned, and the value it was pinned to.
    pub anchor_agent: String,
    pub anchor_elo: f64,
    /// Every agent's rating, sorted strongest first (ties broken by name for
    /// a deterministic ordering).
    pub ratings: Vec<JointEloRating>,
    /// Iterations the MM fit took before converging (diagnostic; see the
    /// method notes on [`fit_joint_elo`]).
    pub iterations: u32,
    /// Whether the fit reached its convergence tolerance rather than running
    /// out of iterations.
    pub converged: bool,
}

/// Maximum MM iterations before [`fit_joint_elo`] gives up and reports
/// `converged: false`. The Bradley-Terry MM update converges monotonically
/// but only linearly, so this is generous; a 7-agent round robin converges in
/// well under a hundred.
const JOINT_MAX_ITERS: u32 = 10_000;

/// Convergence tolerance on the largest single-agent rating change (in Elo
/// points) between consecutive MM iterations.
const JOINT_TOLERANCE: f64 = 1e-9;

/// Fit one rating per agent from a whole set of pairwise records at once, with
/// `anchor_agent` pinned at `anchor_elo`.
///
/// # Why not just call [`fit_elo`] once per agent against a fixed reference
///
/// That would be a strictly weaker estimator, and this project's leaderboard
/// is a round robin precisely so it doesn't have to be one. Anchoring
/// independently uses only the games each agent played *against the
/// reference*: for a 7-agent round robin that discards 5/6ths of the evidence
/// about every agent, and — worse — produces a table that need not be
/// self-consistent (A can outrate B on the reference axis while losing their
/// head-to-head). The joint fit below uses every game.
///
/// # Method
///
/// The same logistic model [`fit_elo`] uses, extended to `n` agents:
/// `P(i beats j) = sigmoid(k·(r_i − r_j))` with `k = ln(10)/400`, draws
/// counted as half a win each side. In Bradley-Terry form, writing
/// `p_i = exp(k·r_i)` for agent `i`'s strength, `P(i beats j) = p_i/(p_i+p_j)`,
/// and the log-likelihood
///
/// ```text
/// ℓ(p) = Σ_{i<j} [ w_ij·ln(p_i/(p_i+p_j)) + w_ji·ln(p_j/(p_i+p_j)) ]
/// ```
///
/// is maximized by the classical MM (minorize-maximize) update
///
/// ```text
/// p_i ← W_i / Σ_{j≠i} n_ij / (p_i + p_j)
/// ```
///
/// where `W_i` is `i`'s total effective wins and `n_ij` the games `i` played
/// against `j`. This is Zermelo's algorithm: each step provably does not
/// decrease the likelihood, and it converges to the unique maximizer whenever
/// the comparison graph is *connected* and every agent has at least one
/// effective win and one effective loss. A full round robin is connected by
/// construction, and the same weak symmetric prior [`fit_elo`] uses (half a
/// pseudo-win to each side of every pairing) guarantees the win/loss condition
/// even if some agent swept or was swept — so the fit stays finite in exactly
/// the cases the pairwise version does.
///
/// Newton's method would converge faster, but MM cannot overshoot or diverge
/// and needs no line search; with `n = 7` and a table this small, robustness
/// is worth more than iteration count.
///
/// # Confidence intervals
///
/// From the joint observed Fisher information, which for this model is the
/// weighted graph Laplacian
///
/// ```text
/// I_ii = Σ_{j≠i} k²·n_ij·p_ij·(1−p_ij),   I_ij = −k²·n_ij·p_ij·(1−p_ij)
/// ```
///
/// (`p_ij` the fitted win probability). It is singular by construction — the
/// likelihood is invariant to shifting every rating by a constant — which is
/// exactly what pinning the anchor fixes: deleting the anchor's row and
/// column leaves a positive-definite matrix whose inverse is the covariance
/// of the *differences* `r_i − r_anchor`. Each agent's interval is then
/// `r̂_i ± 1.96·sqrt(Σ_ii)`, the same asymptotic-normal MLE interval
/// [`fit_elo`] reports, and the anchor's own interval is zero-width because
/// its rating is pinned rather than estimated.
///
/// # Errors
///
/// If `records` is empty, names an agent inconsistently, or does not mention
/// `anchor_agent`; or if the comparison graph is disconnected (some group of
/// agents never played anyone outside it), which would leave the relative
/// scale between the groups unidentified.
pub fn fit_joint_elo(
    records: &[PairwiseRecord],
    anchor_agent: &str,
    anchor_elo: f64,
) -> Result<JointEloTable, String> {
    if records.is_empty() {
        return Err("cannot fit a joint Elo table from zero pairwise records".to_string());
    }

    // Stable, name-sorted agent indexing so the fit is deterministic
    // regardless of the order results files happened to be read in.
    let mut names: Vec<&str> = Vec::new();
    for r in records {
        if r.agent_a == r.agent_b {
            return Err(format!(
                "pairwise record pits \"{}\" against itself; a leaderboard needs distinct agents",
                r.agent_a
            ));
        }
        names.push(&r.agent_a);
        names.push(&r.agent_b);
    }
    names.sort_unstable();
    names.dedup();
    let index: BTreeMap<&str, usize> = names.iter().enumerate().map(|(i, n)| (*n, i)).collect();
    let n = names.len();

    let anchor = *index.get(anchor_agent).ok_or_else(|| {
        format!(
            "anchor agent \"{anchor_agent}\" does not appear in any pairwise record (agents seen: {})",
            names.join(", ")
        )
    })?;

    // Effective wins per agent and effective games per unordered pair, with
    // the half-a-pseudo-win-each-side prior folded in once per pairing.
    let mut effective_wins = vec![0.0f64; n];
    let mut games = vec![0.0f64; n]; // n_ij summed over j, for the fit
    let mut pair_games = vec![0.0f64; n * n];
    // Raw (un-prior-inflated) counts, purely for reporting.
    let mut raw = vec![(0u32, 0u32, 0u32); n];

    for r in records {
        let (i, j) = (index[r.agent_a.as_str()], index[r.agent_b.as_str()]);
        let w = r.wins as f64 + r.draws as f64 * 0.5 + 0.5;
        let l = r.losses as f64 + r.draws as f64 * 0.5 + 0.5;
        effective_wins[i] += w;
        effective_wins[j] += l;
        pair_games[i * n + j] += w + l;
        pair_games[j * n + i] += w + l;
        games[i] += w + l;
        games[j] += w + l;

        raw[i].0 += r.wins;
        raw[i].1 += r.losses;
        raw[i].2 += r.draws;
        raw[j].0 += r.losses;
        raw[j].1 += r.wins;
        raw[j].2 += r.draws;
    }

    check_connected(n, &pair_games, &names)?;

    // MM iteration in strength space (p_i > 0), started from a flat table.
    let mut p = vec![1.0f64; n];
    let mut iterations = 0u32;
    let mut converged = false;
    while iterations < JOINT_MAX_ITERS {
        iterations += 1;
        let mut next = vec![0.0f64; n];
        for i in 0..n {
            let mut denom = 0.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let nij = pair_games[i * n + j];
                if nij > 0.0 {
                    denom += nij / (p[i] + p[j]);
                }
            }
            next[i] = if denom > 0.0 {
                effective_wins[i] / denom
            } else {
                p[i]
            };
        }
        // Re-normalize onto the anchor each step: the likelihood is scale
        // invariant, so without this the iterates can drift towards 0 or
        // infinity and lose precision even while the *differences* converge.
        let scale = next[anchor];
        for v in next.iter_mut() {
            *v /= scale;
        }
        let max_shift = (0..n)
            .map(|i| ((next[i].ln() - p[i].ln()) / ELO_TO_LOGIT).abs())
            .fold(0.0f64, f64::max);
        p = next;
        if max_shift < JOINT_TOLERANCE {
            converged = true;
            break;
        }
    }

    // p is already anchor-normalized (p[anchor] == 1), so ln(p_i)/k is
    // directly the Elo difference from the anchor.
    let diffs: Vec<f64> = p.iter().map(|v| v.ln() / ELO_TO_LOGIT).collect();
    let standard_errors = joint_standard_errors(n, anchor, &pair_games, &p);

    let mut ratings: Vec<JointEloRating> = (0..n)
        .map(|i| {
            let elo = anchor_elo + diffs[i];
            let se = standard_errors[i];
            JointEloRating {
                agent: names[i].to_string(),
                elo,
                elo_ci_low: elo - Z_95 * se,
                elo_ci_high: elo + Z_95 * se,
                games: raw[i].0 + raw[i].1 + raw[i].2,
                wins: raw[i].0,
                losses: raw[i].1,
                draws: raw[i].2,
            }
        })
        .collect();
    ratings.sort_by(|a, b| {
        b.elo
            .partial_cmp(&a.elo)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.agent.cmp(&b.agent))
    });

    Ok(JointEloTable {
        anchor_agent: anchor_agent.to_string(),
        anchor_elo,
        ratings,
        iterations,
        converged,
    })
}

/// Reject a comparison graph that falls into two or more groups with no games
/// between them: the relative scale of two such groups is not identified by
/// any amount of data, so the "leaderboard" would be silently meaningless
/// across the gap.
fn check_connected(n: usize, pair_games: &[f64], names: &[&str]) -> Result<(), String> {
    let mut seen = vec![false; n];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(i) = stack.pop() {
        for j in 0..n {
            if !seen[j] && pair_games[i * n + j] > 0.0 {
                seen[j] = true;
                stack.push(j);
            }
        }
    }
    if let Some(missing) = (0..n).find(|&i| !seen[i]) {
        return Err(format!(
            "the pairwise records do not connect every agent (\"{}\" shares no chain of opponents \
             with \"{}\"), so their ratings are not on a common scale",
            names[missing], names[0]
        ));
    }
    Ok(())
}

/// Per-agent standard errors of `r_i − r_anchor`, from the joint observed
/// information with the anchor's row/column deleted. See [`fit_joint_elo`].
/// The anchor's own entry is 0.
fn joint_standard_errors(n: usize, anchor: usize, pair_games: &[f64], p: &[f64]) -> Vec<f64> {
    // Free parameters: every agent except the pinned anchor.
    let free: Vec<usize> = (0..n).filter(|&i| i != anchor).collect();
    let m = free.len();
    if m == 0 {
        return vec![0.0; n];
    }

    // Information (= negative Hessian) over the free parameters.
    let mut info = vec![0.0f64; m * m];
    for (a, &i) in free.iter().enumerate() {
        for j in 0..n {
            if i == j {
                continue;
            }
            let nij = pair_games[i * n + j];
            if nij <= 0.0 {
                continue;
            }
            let pij = p[i] / (p[i] + p[j]);
            let weight = ELO_TO_LOGIT * ELO_TO_LOGIT * nij * pij * (1.0 - pij);
            info[a * m + a] += weight;
            if let Some(b) = free.iter().position(|&f| f == j) {
                info[a * m + b] -= weight;
            }
        }
    }

    let mut out = vec![0.0f64; n];
    match invert_symmetric(m, &info) {
        Some(cov) => {
            for (a, &i) in free.iter().enumerate() {
                let var = cov[a * m + a];
                out[i] = if var > 0.0 { var.sqrt() } else { f64::INFINITY };
            }
        }
        None => {
            for &i in &free {
                out[i] = f64::INFINITY;
            }
        }
    }
    out
}

/// Invert a small symmetric positive-definite matrix by Gauss-Jordan
/// elimination with partial pivoting. Returns `None` if it is singular to
/// working precision (which for the information matrix here would mean some
/// agent's rating is not identified). `n` is at most the agent count, so an
/// O(n³) dense inverse is free.
fn invert_symmetric(n: usize, matrix: &[f64]) -> Option<Vec<f64>> {
    let mut a = matrix.to_vec();
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }

    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&x, &y| {
                a[x * n + col]
                    .abs()
                    .partial_cmp(&a[y * n + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("col < n so the range is non-empty");
        if a[pivot_row * n + col].abs() < 1e-12 {
            return None;
        }
        if pivot_row != col {
            for k in 0..n {
                a.swap(col * n + k, pivot_row * n + k);
                inv.swap(col * n + k, pivot_row * n + k);
            }
        }
        let pivot = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= pivot;
            inv[col * n + k] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..n {
                a[row * n + k] -= factor * a[col * n + k];
                inv[row * n + k] -= factor * inv[col * n + k];
            }
        }
    }
    Some(inv)
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

    // --- joint multi-agent fit -------------------------------------------

    fn record(a: &str, b: &str, wins: u32, losses: u32, draws: u32) -> PairwiseRecord {
        PairwiseRecord {
            agent_a: a.to_string(),
            agent_b: b.to_string(),
            wins,
            losses,
            draws,
        }
    }

    fn rating_of<'a>(table: &'a JointEloTable, agent: &str) -> &'a JointEloRating {
        table
            .ratings
            .iter()
            .find(|r| r.agent == agent)
            .unwrap_or_else(|| panic!("{agent} should be in the fitted table"))
    }

    /// A synthetic round robin generated from *known* true ratings: three
    /// agents 200 Elo apart, every pairing played at the exact expected score
    /// the logistic model predicts (rounded to whole games). The joint fit
    /// should recover the gaps, not merely the ordering.
    #[test]
    fn joint_fit_recovers_a_hand_computed_three_agent_ladder() {
        // True ratings: weak 0, mid 200, strong 400.
        // Expected scores: sigmoid(k*200) = 0.7597..., sigmoid(k*400) = 0.9091.
        let n = 10_000.0;
        let p200 = sigmoid(ELO_TO_LOGIT * 200.0);
        let p400 = sigmoid(ELO_TO_LOGIT * 400.0);
        let mid_v_weak = (n * p200).round() as u32;
        let strong_v_mid = (n * p200).round() as u32;
        let strong_v_weak = (n * p400).round() as u32;

        let records = vec![
            record("mid", "weak", mid_v_weak, n as u32 - mid_v_weak, 0),
            record("strong", "mid", strong_v_mid, n as u32 - strong_v_mid, 0),
            record("strong", "weak", strong_v_weak, n as u32 - strong_v_weak, 0),
        ];

        let table = fit_joint_elo(&records, "weak", 1000.0).expect("a connected round robin fits");
        assert!(table.converged, "MM should converge on a clean dataset");

        assert_eq!(rating_of(&table, "weak").elo, 1000.0, "anchor is pinned");
        let mid = rating_of(&table, "mid").elo;
        let strong = rating_of(&table, "strong").elo;
        // Every pairing is consistent with the same true ratings, so the
        // joint MLE lands essentially exactly on them (the only perturbation
        // is the half-pseudo-game prior and the rounding to whole games).
        assert!((mid - 1200.0).abs() < 2.0, "expected mid ~1200, got {mid}");
        assert!(
            (strong - 1400.0).abs() < 2.0,
            "expected strong ~1400, got {strong}"
        );

        // Ranking order, strongest first.
        let order: Vec<&str> = table.ratings.iter().map(|r| r.agent.as_str()).collect();
        assert_eq!(order, vec!["strong", "mid", "weak"]);
    }

    /// The point of a *joint* fit: an agent's rating is informed by games it
    /// never played, through shared opponents. Here `c` never plays `a`, but
    /// beats `b` exactly as hard as `a` does, so the fit must place `c` at
    /// essentially the same rating as `a` — something independent pairwise
    /// anchoring against `a` could not produce at all (there are no `a`-vs-`c`
    /// games to anchor on).
    #[test]
    fn joint_fit_propagates_strength_through_a_shared_opponent() {
        let records = vec![record("a", "b", 750, 250, 0), record("c", "b", 750, 250, 0)];
        let table = fit_joint_elo(&records, "b", 1000.0).expect("connected through b");
        let a = rating_of(&table, "a").elo;
        let c = rating_of(&table, "c").elo;
        assert!(
            (a - c).abs() < 1e-6,
            "identical records against a shared opponent should give identical ratings, got {a} and {c}"
        );
        assert!(a > 1000.0, "both should outrate the anchor, got {a}");
        // ...but with no direct evidence, the a-vs-c comparison is the
        // *sum* of two noisy estimates, so both intervals are honestly wide.
        let a_row = rating_of(&table, "a");
        assert!(a_row.elo_ci_low < a && a < a_row.elo_ci_high);
    }

    /// With only two agents, the joint fit is the same estimation problem
    /// `fit_elo` solves, so the two must agree — both on the point estimate
    /// and on the interval width. This pins the joint code to the already
    /// tested pairwise path rather than letting the two drift apart.
    #[test]
    fn joint_fit_agrees_with_the_pairwise_fit_on_a_single_pairing() {
        for (w, l, d) in [(600u32, 400u32, 0u32), (55, 40, 5), (50, 0, 0)] {
            let pairwise = fit_elo(w, l, d);
            let table = fit_joint_elo(&[record("cand", "ref", w, l, d)], "ref", 0.0).unwrap();
            let cand = rating_of(&table, "cand");
            assert!(
                (cand.elo - pairwise.rating_diff).abs() < 1e-6,
                "joint {} vs pairwise {} for {w}/{l}/{d}",
                cand.elo,
                pairwise.rating_diff
            );
            assert!((cand.elo_ci_low - pairwise.diff_ci_low).abs() < 1e-6);
            assert!((cand.elo_ci_high - pairwise.diff_ci_high).abs() < 1e-6);
        }
    }

    #[test]
    fn joint_fit_counts_every_agents_games_from_its_own_perspective() {
        let records = vec![
            record("a", "b", 6, 3, 1),
            record("a", "c", 5, 5, 0),
            record("b", "c", 2, 8, 0),
        ];
        let table = fit_joint_elo(&records, "a", 0.0).unwrap();
        let a = rating_of(&table, "a");
        assert_eq!((a.wins, a.losses, a.draws), (11, 8, 1));
        assert_eq!(a.games, 20);
        let b = rating_of(&table, "b");
        assert_eq!((b.wins, b.losses, b.draws), (5, 14, 1));
        let c = rating_of(&table, "c");
        assert_eq!((c.wins, c.losses, c.draws), (13, 7, 0));
    }

    #[test]
    fn joint_fit_stays_finite_when_one_agent_sweeps_every_pairing() {
        let records = vec![
            record("god", "b", 100, 0, 0),
            record("god", "c", 100, 0, 0),
            record("b", "c", 50, 50, 0),
        ];
        let table = fit_joint_elo(&records, "b", 1000.0).unwrap();
        for r in &table.ratings {
            assert!(r.elo.is_finite(), "{} rating not finite", r.agent);
            assert!(r.elo_ci_low.is_finite() && r.elo_ci_high.is_finite());
        }
        assert!(rating_of(&table, "god").elo > 1000.0);
    }

    #[test]
    fn joint_fit_pins_the_anchor_with_a_zero_width_interval() {
        let records = vec![record("a", "b", 60, 40, 0), record("b", "c", 60, 40, 0)];
        let table = fit_joint_elo(&records, "b", 1000.0).unwrap();
        let anchor = rating_of(&table, "b");
        assert_eq!(anchor.elo, 1000.0);
        assert_eq!(anchor.elo_ci_low, 1000.0);
        assert_eq!(anchor.elo_ci_high, 1000.0);
    }

    #[test]
    fn joint_fit_narrows_intervals_as_games_accumulate() {
        let few = fit_joint_elo(
            &[record("a", "b", 60, 40, 0), record("b", "c", 60, 40, 0)],
            "b",
            0.0,
        )
        .unwrap();
        let many = fit_joint_elo(
            &[
                record("a", "b", 6000, 4000, 0),
                record("b", "c", 6000, 4000, 0),
            ],
            "b",
            0.0,
        )
        .unwrap();
        let width = |t: &JointEloTable, agent: &str| {
            let r = rating_of(t, agent);
            r.elo_ci_high - r.elo_ci_low
        };
        assert!(width(&many, "a") < width(&few, "a"));
        assert!(width(&many, "c") < width(&few, "c"));
    }

    #[test]
    fn joint_fit_is_invariant_to_the_order_records_are_supplied_in() {
        let mut records = vec![
            record("a", "b", 60, 40, 0),
            record("b", "c", 55, 45, 0),
            record("a", "c", 70, 30, 0),
        ];
        let first = fit_joint_elo(&records, "c", 1000.0).unwrap();
        records.reverse();
        let second = fit_joint_elo(&records, "c", 1000.0).unwrap();
        assert_eq!(first.ratings, second.ratings);
    }

    /// Reversing which side of a record is "agent A" flips wins and losses and
    /// must not change the fit at all.
    #[test]
    fn joint_fit_is_invariant_to_which_side_is_named_first() {
        let forward = fit_joint_elo(
            &[record("a", "b", 60, 40, 0), record("b", "c", 55, 45, 0)],
            "a",
            0.0,
        )
        .unwrap();
        let flipped = fit_joint_elo(
            &[record("b", "a", 40, 60, 0), record("c", "b", 45, 55, 0)],
            "a",
            0.0,
        )
        .unwrap();
        assert_eq!(forward.ratings, flipped.ratings);
    }

    #[test]
    fn joint_fit_rejects_a_disconnected_comparison_graph() {
        let records = vec![record("a", "b", 10, 5, 0), record("c", "d", 10, 5, 0)];
        let err = fit_joint_elo(&records, "a", 0.0).unwrap_err();
        assert!(err.contains("common scale"), "unexpected message: {err}");
    }

    #[test]
    fn joint_fit_rejects_an_unknown_anchor_and_an_empty_input() {
        let records = vec![record("a", "b", 1, 1, 0)];
        assert!(fit_joint_elo(&records, "nobody", 0.0)
            .unwrap_err()
            .contains("nobody"));
        assert!(fit_joint_elo(&[], "a", 0.0).is_err());
        assert!(fit_joint_elo(&[record("a", "a", 1, 1, 0)], "a", 0.0).is_err());
    }

    /// The Gauss-Jordan inverse used for the covariance, checked against a
    /// matrix whose inverse is known by hand.
    #[test]
    fn symmetric_inverse_matches_a_hand_computed_case() {
        // [[4, 1], [1, 3]] has determinant 11 and inverse [[3, -1], [-1, 4]]/11.
        let inv = invert_symmetric(2, &[4.0, 1.0, 1.0, 3.0]).unwrap();
        let expect = [3.0 / 11.0, -1.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0];
        for (got, want) in inv.iter().zip(expect.iter()) {
            assert!((got - want).abs() < 1e-12, "{got} != {want}");
        }
        assert!(invert_symmetric(2, &[1.0, 1.0, 1.0, 1.0]).is_none());
    }

    /// End-to-end sanity on noisy data with known truth: simulate a full
    /// 5-agent round robin from true ratings and check the fit both orders
    /// them correctly and covers each true rating with its interval.
    #[test]
    fn joint_fit_recovers_a_simulated_five_agent_round_robin() {
        let truth = [
            ("e", 0.0f64),
            ("d", 100.0),
            ("c", 200.0),
            ("b", 300.0),
            ("a", 400.0),
        ];
        let mut rng = StdRng::seed_from_u64(7);
        let mut records = Vec::new();
        for i in 0..truth.len() {
            for j in (i + 1)..truth.len() {
                let p = sigmoid(ELO_TO_LOGIT * (truth[i].1 - truth[j].1));
                let (w, l) = simulate(p, 2000, &mut rng);
                records.push(record(truth[i].0, truth[j].0, w, l, 0));
            }
        }
        let table = fit_joint_elo(&records, "e", 1000.0).unwrap();
        assert!(table.converged);

        let order: Vec<&str> = table.ratings.iter().map(|r| r.agent.as_str()).collect();
        assert_eq!(order, vec!["a", "b", "c", "d", "e"]);

        for (name, true_elo) in truth {
            let r = rating_of(&table, name);
            let expected = 1000.0 + true_elo;
            assert!(
                r.elo_ci_low <= expected && expected <= r.elo_ci_high,
                "{name}: true {expected} outside [{}, {}] (point {})",
                r.elo_ci_low,
                r.elo_ci_high,
                r.elo
            );
        }
    }
}
