//! Pre-registered bounds on *how* a candidate wins, not just how often.
//!
//! # Why this exists
//!
//! [`crate::experiment`] already measures a candidate against a control with
//! paired seeds, several budgets, Elo with a confidence interval and an SPRT
//! verdict — and it already *reports* the victory-kind breakdown (military /
//! scientific / civilian / tiebreak, per side) and the race-exposure flags.
//! But its verdict ([`crate::experiment::overall_verdict`]) is a pure SPRT
//! conjunction over aggregate Elo: nothing in the tool ever read the
//! victory-kind numbers.
//!
//! That gap has already cost this project a real investigation. A fitted
//! evaluation change measured as only "slightly negative" on aggregate Elo,
//! which reads as a boring null result — but its victory-kind breakdown showed
//! it had become a redundant military-race detector that had abandoned
//! civilian-score judgement altogether. A human caught that **by eye**, from a
//! table the tool printed but never judged. The whole point of this module is
//! that the next such candidate is caught by the tool.
//!
//! A [`MechanismGate`] is therefore a set of **pre-registered, directional
//! bounds** on the candidate's win-mechanism profile relative to the control's,
//! evaluated per budget (the same way Elo is pooled — never across budgets,
//! see [`crate::experiment`]'s module docs) and reported as its own verdict
//! **alongside** the Elo one, never merged into it. "Stronger on aggregate
//! Elo, but by the wrong mechanism" is a statement this project needs to be
//! able to make.
//!
//! # The metrics
//!
//! Every metric is a **share of one side's own wins**, so the two sides of a
//! head-to-head match stay comparable even though the candidate and the
//! control do not win the same number of games:
//!
//! | Metric | Events counted | Trials |
//! | ------ | -------------- | ------ |
//! | `military_share` | that side's military-supremacy wins | that side's wins |
//! | `science_share` | its scientific-supremacy wins | that side's wins |
//! | `civilian_share` | its civilian (VP) wins | that side's wins |
//! | `tiebreak_share` | its civilian-tiebreak wins | that side's wins |
//! | `military_exposure` | its wins in games where the military race was exposed | that side's wins |
//! | `science_exposure` | its wins in games where the science race was exposed | that side's wins |
//!
//! The two `*_exposure` metrics are **not** the match-level race-exposure rate
//! that [`crate::match_runner::RaceExposure`] reports and the experiment's
//! summary table prints. That rate is a property of the *game* — both arms of
//! a match play the same games, so it is identical for the candidate and the
//! control and has no control counterpart to compare against. What a gate
//! needs is a per-side quantity, so these two ask a per-side question instead:
//! *of the games this side won, how many came out of an exposed race?* See
//! [`crate::match_runner::win_race_exposure`].
//!
//! # The bounds
//!
//! A gate is written as a comma-separated list, e.g.
//!
//! ```text
//! --gate science_share>=0.8x,military_share<=1.5x,civilian_share>=0.8x
//! ```
//!
//! Each bound is `<metric><op><threshold>`, where `op` is `>=` (a floor) or
//! `<=` (a ceiling), and a trailing `x` makes the threshold **relative to the
//! control's** share of the same kind rather than an absolute share.
//!
//! Bounds are deliberately **directional and asymmetric**, not a symmetric
//! pass/fail band. `science_share>=0.8x` says "reject if the candidate's
//! science win share drops below 0.8x the control's" and says *nothing at all*
//! about a candidate that improved its science share by 1.5x — which is the
//! whole point: a gate should catch a candidate getting meaningfully *worse* on
//! a mechanism dimension, not demand that it look identical to the control. A
//! caller who genuinely wants a two-sided band writes both directions for that
//! metric.
//!
//! # Rare events, and why a raw ratio threshold would be noise
//!
//! Scientific supremacy is about 2.3% of games in this project's own self-play
//! data. A 1,000-game cell therefore holds on the order of 20-25 such games
//! *across both sides* — so a bare "is the observed ratio below 0.8"
//! comparison at that sample size mostly measures sampling noise, and would
//! turn a gate into a coin flip that occasionally vetoes a good change and
//! routinely blesses a bad one.
//!
//! Every check here is therefore three-valued ([`GateOutcome`]), and gets
//! there in two stages:
//!
//! 1. **An evidence floor.** The bound's own boundary share `q` (for a
//!    relative bound, `threshold * control_share`) has to imply at least
//!    [`MechanismGate::min_expected_events`] expected wins of that kind among
//!    the candidate's wins (`candidate_wins * q`), and a relative bound
//!    additionally needs at least that many such wins on the *control* side to
//!    have a reference worth taking a ratio against. Below either floor the
//!    check is [`GateOutcome::Inconclusive`] with a note saying so — never a
//!    silent pass and never a fail.
//! 2. **A noise test.** Above the floor, the observed share is compared to `q`
//!    with a one-sided z test using the variance implied by the *boundary*
//!    hypothesis (`q(1-q)/n`, plus the control's own binomial variance scaled
//!    by the threshold for a relative bound, since the reference is itself
//!    estimated). A check only **fails** when the bound is violated *and* the
//!    violation is beyond [`MechanismGate::z_critical`] standard errors; a
//!    violation inside the noise is `Inconclusive`, again with a note.
//!
//! Using the boundary variance rather than the observed one matters: an
//! observed share of exactly zero has zero observed variance, and a test built
//! on that would report infinite significance from a single missing win.
//!
//! One consequence is worth knowing before it surprises someone: a **relative**
//! bound on a kind the *control* barely ever wins by is `Inconclusive` however
//! extreme the candidate looks. `random` vs `greedy` measures 53.8% military
//! wins against the control's 0.1% — a 496x ratio — and
//! `military_share<=1.5x` still declines to rule, because a control estimate
//! built on a single game is no reference at all. That is the design working,
//! not a gap: a caller who wants to bound behaviour the control does not
//! exhibit should say so absolutely (`military_share<=0.2`), which needs no
//! reference and resolves cleanly.
//!
//! # What the verdict means
//!
//! Checks combine worst-first — one [`GateOutcome::Fail`] anywhere fails its
//! budget, and one failing budget fails the run — mirroring
//! [`crate::experiment::overall_verdict`]'s conjunction. As with that
//! conjunction, this is a reporting convenience rather than a statistical
//! statement with a controlled family-wise error rate: several bounds each
//! tested at `z_critical` are several chances to trip. Read
//! [`MechanismReport::checks`] for the actual numbers, and read the mechanism
//! verdict *and* the Elo verdict — neither is a summary of the other.

use serde::{Deserialize, Serialize};

use crate::match_runner::{MatchVictoryBreakdown, WinRaceExposure};

/// The gate applied when `duels-arena experiment` is run without an explicit
/// `--gate`, so forgetting the flag is not silently equivalent to no check at
/// all.
///
/// Every threshold here is a **ratio against the control**, deliberately: a
/// dimensionless "don't get much worse than the thing you are replacing" is a
/// defensible default, whereas an absolute share floor would bake this
/// project's current win-kind mix into the tool, where an explicit flag is
/// clearly the better place for it.
///
/// The three bounds are exactly the shape of the failure this module was built
/// for (see the module docs): a candidate that quietly stops winning on
/// civilian score, stops winning on science, or leans on military supremacy
/// far harder than the control does. They are loose on purpose — halving a win
/// share, or doubling the military one, is a gross change of playing style,
/// not a tuning wobble — because a default gate that fired on ordinary
/// variation would be worse than none.
pub const DEFAULT_GATE: &str = "civilian_share>=0.5x,science_share>=0.5x,military_share<=2x";

/// Default for [`MechanismGate::min_expected_events`]: the conventional
/// "expected count of at least 5" rule of thumb for trusting a normal
/// approximation to a binomial.
pub const DEFAULT_MIN_EXPECTED_EVENTS: f64 = 5.0;

/// Default for [`MechanismGate::z_critical`]: one-sided 95%.
pub const DEFAULT_Z_CRITICAL: f64 = 1.645;

/// Spellings of `--gate` that mean "no mechanism gate at all".
pub const GATE_DISABLED_SPELLINGS: [&str; 3] = ["none", "off", "disabled"];

/// One dimension of a candidate's win-mechanism profile. See the module docs
/// for what each one counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MechanismMetric {
    /// Military-supremacy wins as a share of the side's wins.
    MilitaryShare,
    /// Scientific-supremacy wins as a share of the side's wins.
    ScienceShare,
    /// Civilian (victory-point) wins as a share of the side's wins.
    CivilianShare,
    /// Civilian-tiebreak wins as a share of the side's wins.
    TiebreakShare,
    /// Wins out of a military-race-exposed game, as a share of the side's
    /// wins.
    MilitaryExposure,
    /// Wins out of a science-race-exposed game, as a share of the side's wins.
    ScienceExposure,
}

impl MechanismMetric {
    /// Every metric, in the order the help text lists them.
    pub const ALL: [MechanismMetric; 6] = [
        MechanismMetric::MilitaryShare,
        MechanismMetric::ScienceShare,
        MechanismMetric::CivilianShare,
        MechanismMetric::TiebreakShare,
        MechanismMetric::MilitaryExposure,
        MechanismMetric::ScienceExposure,
    ];

    /// The name this metric is written as on the command line.
    pub fn name(self) -> &'static str {
        match self {
            MechanismMetric::MilitaryShare => "military_share",
            MechanismMetric::ScienceShare => "science_share",
            MechanismMetric::CivilianShare => "civilian_share",
            MechanismMetric::TiebreakShare => "tiebreak_share",
            MechanismMetric::MilitaryExposure => "military_exposure",
            MechanismMetric::ScienceExposure => "science_exposure",
        }
    }

    /// Parse a metric name, listing the valid ones on failure.
    pub fn parse(s: &str) -> Result<Self, String> {
        MechanismMetric::ALL
            .into_iter()
            .find(|m| m.name() == s)
            .ok_or_else(|| {
                format!(
                    "unknown gate metric \"{s}\": expected one of {}",
                    MechanismMetric::ALL
                        .iter()
                        .map(|m| m.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
}

/// Which way a bound points. Bounds are one-sided on purpose — see the module
/// docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundDirection {
    /// `>=`: the candidate's share must not fall below the threshold.
    AtLeast,
    /// `<=`: the candidate's share must not rise above the threshold.
    AtMost,
}

impl BoundDirection {
    /// The operator this direction is written as.
    pub fn op(self) -> &'static str {
        match self {
            BoundDirection::AtLeast => ">=",
            BoundDirection::AtMost => "<=",
        }
    }
}

/// One pre-registered bound: `<metric><op><threshold>[x]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MechanismBound {
    /// Which dimension this bounds.
    pub metric: MechanismMetric,
    /// Which side of the threshold the candidate has to stay on.
    pub direction: BoundDirection,
    /// The threshold: a multiple of the control's share when `relative`, an
    /// absolute share in `0..=1` otherwise.
    pub threshold: f64,
    /// Whether `threshold` is relative to the control's share of the same
    /// metric (a trailing `x` on the command line).
    pub relative: bool,
}

impl MechanismBound {
    /// The bound written back out the way it was given, e.g.
    /// `"science_share>=0.8x"`.
    pub fn label(&self) -> String {
        format!(
            "{}{}{}{}",
            self.metric.name(),
            self.direction.op(),
            format_threshold(self.threshold),
            if self.relative { "x" } else { "" }
        )
    }

    /// Parse one bound. See [`MechanismGate::parse`] for the list form.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.is_empty() {
            return Err("empty gate bound".to_string());
        }
        let (metric_str, direction, rest) = if let Some((m, r)) = s.split_once(">=") {
            (m, BoundDirection::AtLeast, r)
        } else if let Some((m, r)) = s.split_once("<=") {
            (m, BoundDirection::AtMost, r)
        } else {
            return Err(format!(
                "gate bound \"{s}\" has no comparison: expected \"<metric>>=<n>\" or \
                 \"<metric><=<n>\" (a bare \">\" or \"<\" is not accepted, so the operator always \
                 reads the same in a shell and in a report)"
            ));
        };

        let metric = MechanismMetric::parse(metric_str.trim())?;
        let rest = rest.trim();
        let (number, relative) = match rest.strip_suffix(['x', 'X']) {
            Some(n) => (n.trim(), true),
            None => (rest, false),
        };
        let threshold: f64 = number
            .parse()
            .map_err(|_| format!("gate bound \"{s}\" has a non-numeric threshold \"{number}\""))?;
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(format!(
                "gate bound \"{s}\" needs a finite, non-negative threshold"
            ));
        }
        if relative && threshold == 0.0 {
            return Err(format!(
                "gate bound \"{s}\" is vacuous: a 0x floor or ceiling on a share relative to the \
                 control can never be informative"
            ));
        }
        if !relative && threshold > 1.0 {
            return Err(format!(
                "gate bound \"{s}\" has an absolute threshold above 1.0; shares are fractions of \
                 one side's wins - did you mean \"{}{}{}x\" (a multiple of the control's share)?",
                metric.name(),
                direction.op(),
                format_threshold(threshold)
            ));
        }
        Ok(Self {
            metric,
            direction,
            threshold,
            relative,
        })
    }
}

/// A whole pre-registered gate: the bounds, plus the two knobs that decide
/// when the evidence is thin enough that a check has to say "inconclusive"
/// rather than pass or fail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MechanismGate {
    /// The gate as the caller wrote it, kept verbatim for the report.
    pub spec: String,
    /// The parsed bounds, in the order given.
    pub bounds: Vec<MechanismBound>,
    /// Minimum expected count of the events a bound is about before that
    /// bound is evaluated at all — see the module docs on rare events.
    pub min_expected_events: f64,
    /// How many standard errors past the bound an observed violation has to
    /// be before it counts as a failure rather than noise.
    pub z_critical: f64,
}

impl MechanismGate {
    /// Parse a comma-separated gate spec. Returns `Ok(None)` for the explicit
    /// "no gate" spellings ([`GATE_DISABLED_SPELLINGS`]).
    ///
    /// Rejects the same metric bounded twice in the same direction: two such
    /// bounds are either redundant or contradictory, and neither is what the
    /// caller meant. The *opposite* directions on one metric are fine — that
    /// is how a caller asks for a two-sided band.
    pub fn parse(spec: &str) -> Result<Option<Self>, String> {
        let trimmed = spec.trim();
        if GATE_DISABLED_SPELLINGS.contains(&trimmed.to_ascii_lowercase().as_str()) {
            return Ok(None);
        }
        if trimmed.is_empty() {
            return Err(format!(
                "empty --gate: pass bounds like \"{DEFAULT_GATE}\", or \"none\" to disable the \
                 mechanism gate outright"
            ));
        }
        let mut bounds: Vec<MechanismBound> = Vec::new();
        for piece in trimmed.split(',') {
            let bound = MechanismBound::parse(piece)?;
            if bounds
                .iter()
                .any(|b| b.metric == bound.metric && b.direction == bound.direction)
            {
                return Err(format!(
                    "--gate bounds {} more than once in the same direction; use one bound per \
                     (metric, direction), or both directions to ask for a band",
                    bound.metric.name()
                ));
            }
            bounds.push(bound);
        }
        Ok(Some(Self {
            spec: trimmed.to_string(),
            bounds,
            min_expected_events: DEFAULT_MIN_EXPECTED_EVENTS,
            z_critical: DEFAULT_Z_CRITICAL,
        }))
    }

    /// Evaluate every bound against one budget's pooled counts.
    pub fn evaluate_budget(&self, budget: &str, counts: &MechanismCounts) -> Vec<MechanismCheck> {
        self.bounds
            .iter()
            .map(|b| self.check(budget, b, counts))
            .collect()
    }

    /// Evaluate the gate over every budget's pooled counts and combine the
    /// results into a report. `budgets` is `(budget label, counts)` in the
    /// order the report should list them.
    pub fn evaluate(&self, budgets: &[(String, MechanismCounts)]) -> MechanismReport {
        let mut checks: Vec<MechanismCheck> = Vec::new();
        let mut per_budget: Vec<BudgetGateOutcome> = Vec::new();
        for (budget, counts) in budgets {
            let mine = self.evaluate_budget(budget, counts);
            per_budget.push(BudgetGateOutcome {
                budget: budget.clone(),
                outcome: combine(mine.iter().map(|c| c.outcome)),
            });
            checks.extend(mine);
        }
        let verdict = if per_budget.is_empty() {
            GateOutcome::Inconclusive
        } else {
            combine(per_budget.iter().map(|b| b.outcome))
        };
        MechanismReport {
            gate: self.spec.clone(),
            min_expected_events: self.min_expected_events,
            z_critical: self.z_critical,
            checks,
            per_budget,
            verdict,
        }
    }

    /// Evaluate one bound against one budget's counts. This is the whole
    /// statistical core of the module; see the module docs' "rare events"
    /// section for the reasoning behind the two stages.
    fn check(
        &self,
        budget: &str,
        bound: &MechanismBound,
        counts: &MechanismCounts,
    ) -> MechanismCheck {
        let k_c = counts.candidate.events(bound.metric);
        let n_c = counts.candidate.wins;
        let k_t = counts.control.events(bound.metric);
        let n_t = counts.control.wins;
        let p_c = share(k_c, n_c);
        let p_t = share(k_t, n_t);
        let required = if bound.relative {
            bound.threshold * p_t
        } else {
            bound.threshold
        };

        let mut check = MechanismCheck {
            budget: budget.to_string(),
            metric: bound.metric.name().to_string(),
            bound: bound.label(),
            candidate_events: k_c,
            candidate_trials: n_c,
            candidate_value: p_c,
            control_events: k_t,
            control_trials: n_t,
            control_value: p_t,
            ratio: if n_t > 0 && p_t > 0.0 {
                Some(p_c / p_t)
            } else {
                None
            },
            required_value: required,
            z: None,
            outcome: GateOutcome::Inconclusive,
            note: String::new(),
        };

        // Stage 0: is there anything at all to compare?
        if n_c == 0 {
            check.note = "the candidate won no games at this budget, so it has no win-mechanism \
                          profile to bound"
                .to_string();
            return check;
        }
        if bound.relative && n_t == 0 {
            check.note = "the control won no games at this budget, so there is no share to take a \
                          ratio against"
                .to_string();
            return check;
        }

        // Stage 1: the evidence floor. Both the reference (for a relative
        // bound) and the count the bound implies have to be big enough for the
        // comparison to mean anything at this sample size.
        if bound.relative && (k_t as f64) < self.min_expected_events {
            check.note = format!(
                "the control won only {k_t} game(s) this way, below the {:.0} needed to estimate \
                 a share worth taking a ratio against; gate inconclusive rather than pass or fail",
                self.min_expected_events
            );
            return check;
        }
        let expected = n_c as f64 * required;
        if expected < self.min_expected_events {
            check.note = format!(
                "the bound implies only {expected:.1} expected win(s) of this kind out of the \
                 candidate's {n_c}, below the {:.0} this check needs; gate inconclusive rather \
                 than pass or fail",
                self.min_expected_events
            );
            return check;
        }

        // Stage 2: the noise test. Variance is taken under the *boundary*
        // hypothesis, not from the observed share, so an observed zero does
        // not produce an infinitely significant result.
        let q = required.clamp(0.0, 1.0);
        let mut var = q * (1.0 - q) / n_c as f64;
        if bound.relative && n_t > 0 {
            // The reference share is itself estimated, so its sampling error
            // belongs in the comparison.
            var += bound.threshold * bound.threshold * p_t * (1.0 - p_t) / n_t as f64;
        }
        if var <= 0.0 {
            check.note = "the bound sits exactly at 0% or 100% of the candidate's wins, where \
                          there is no sampling spread to test against; gate inconclusive rather \
                          than pass or fail"
                .to_string();
            return check;
        }
        let z = (p_c - required) / var.sqrt();
        check.z = Some(z);

        match bound.direction {
            BoundDirection::AtLeast => {
                if p_c >= required {
                    check.outcome = GateOutcome::Pass;
                    check.note = format!(
                        "{:.1}% is at or above the required {:.1}%",
                        100.0 * p_c,
                        100.0 * required
                    );
                } else if z <= -self.z_critical {
                    check.outcome = GateOutcome::Fail;
                    check.note = format!(
                        "{:.1}% is {:.2} standard errors below the required {:.1}% - past the \
                         {:.3} this gate treats as beyond sampling noise",
                        100.0 * p_c,
                        -z,
                        100.0 * required,
                        self.z_critical
                    );
                } else {
                    check.note = format!(
                        "{:.1}% is below the required {:.1}%, but only by {:.2} standard errors, \
                         which sampling noise alone produces at these counts; gate inconclusive \
                         rather than pass or fail",
                        100.0 * p_c,
                        100.0 * required,
                        -z
                    );
                }
            }
            BoundDirection::AtMost => {
                if p_c <= required {
                    check.outcome = GateOutcome::Pass;
                    check.note = format!(
                        "{:.1}% is at or below the permitted {:.1}%",
                        100.0 * p_c,
                        100.0 * required
                    );
                } else if z >= self.z_critical {
                    check.outcome = GateOutcome::Fail;
                    check.note = format!(
                        "{:.1}% is {z:.2} standard errors above the permitted {:.1}% - past the \
                         {:.3} this gate treats as beyond sampling noise",
                        100.0 * p_c,
                        100.0 * required,
                        self.z_critical
                    );
                } else {
                    check.note = format!(
                        "{:.1}% is above the permitted {:.1}%, but only by {z:.2} standard \
                         errors, which sampling noise alone produces at these counts; gate \
                         inconclusive rather than pass or fail",
                        100.0 * p_c,
                        100.0 * required
                    );
                }
            }
        }
        check
    }
}

/// One side's win-mechanism counts: the numerators every metric needs, plus
/// the shared denominator (that side's wins).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SideCounts {
    /// Games this side won — the denominator of every share.
    pub wins: u32,
    /// Wins by military supremacy.
    pub military_supremacy: u32,
    /// Wins by scientific supremacy.
    pub scientific_supremacy: u32,
    /// Wins on victory points at the end of Age III.
    pub civilian_victory: u32,
    /// Wins on the civilian-points tiebreak.
    pub civilian_tiebreak: u32,
    /// Wins in games where the military race was exposed at some point.
    pub military_exposed_wins: u32,
    /// Wins in games where the science race was exposed at some point.
    pub science_exposed_wins: u32,
}

impl SideCounts {
    /// The numerator this metric counts for this side.
    pub fn events(&self, metric: MechanismMetric) -> u32 {
        match metric {
            MechanismMetric::MilitaryShare => self.military_supremacy,
            MechanismMetric::ScienceShare => self.scientific_supremacy,
            MechanismMetric::CivilianShare => self.civilian_victory,
            MechanismMetric::TiebreakShare => self.civilian_tiebreak,
            MechanismMetric::MilitaryExposure => self.military_exposed_wins,
            MechanismMetric::ScienceExposure => self.science_exposed_wins,
        }
    }

    /// This side's share of its own wins for `metric` (`0.0` with no wins).
    pub fn share_of(&self, metric: MechanismMetric) -> f64 {
        share(self.events(metric), self.wins)
    }
}

/// Both sides' [`SideCounts`], candidate first — the only input a
/// [`MechanismGate`] reads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MechanismCounts {
    /// The candidate's counts ("role A" in a match, see
    /// [`crate::match_runner::GameRecord::agent_a_seat`]).
    pub candidate: SideCounts,
    /// The control's counts.
    pub control: SideCounts,
}

impl MechanismCounts {
    /// Assemble the counts from the two aggregates the match runner already
    /// derives from a set of games.
    pub fn from_parts(victories: &MatchVictoryBreakdown, exposure: &WinRaceExposure) -> Self {
        let side = |v: &crate::match_runner::VictoryBreakdown,
                    e: &crate::match_runner::SideWinRaceExposure| SideCounts {
            wins: v.total(),
            military_supremacy: v.military_supremacy,
            scientific_supremacy: v.scientific_supremacy,
            civilian_victory: v.civilian_victory,
            civilian_tiebreak: v.civilian_tiebreak,
            military_exposed_wins: e.military,
            science_exposed_wins: e.science,
        };
        Self {
            candidate: side(&victories.a, &exposure.a),
            control: side(&victories.b, &exposure.b),
        }
    }
}

/// The three-valued outcome of a mechanism check. Kept deliberately distinct
/// from [`crate::experiment::Verdict`] (the Elo SPRT's answer) so a summary
/// can never blur "stronger" with "stronger by the right mechanism".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateOutcome {
    /// The bound is satisfied, on enough events for that to mean something.
    Pass,
    /// The bound is violated by more than sampling noise at these counts.
    Fail,
    /// Not enough evidence to say either way — too few events for the bound to
    /// be evaluated, or a violation inside the noise. Explicitly *not* a pass.
    Inconclusive,
}

impl GateOutcome {
    /// Worst-first severity, so a set of outcomes can be combined by taking a
    /// maximum.
    fn severity(self) -> u8 {
        match self {
            GateOutcome::Pass => 0,
            GateOutcome::Inconclusive => 1,
            GateOutcome::Fail => 2,
        }
    }
}

/// Combine outcomes worst-first: any `Fail` wins, then any `Inconclusive`.
/// An empty set is `Inconclusive` — nothing was checked, which is not a pass.
fn combine(outcomes: impl IntoIterator<Item = GateOutcome>) -> GateOutcome {
    outcomes
        .into_iter()
        .max_by_key(|o| o.severity())
        .unwrap_or(GateOutcome::Inconclusive)
}

/// One bound evaluated against one budget's pooled counts, with every number
/// that produced the outcome so a reader never has to take the verdict on
/// faith.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MechanismCheck {
    /// The budget label these counts came from.
    pub budget: String,
    /// The metric's name.
    pub metric: String,
    /// The bound as written, e.g. `"science_share>=0.8x"`.
    pub bound: String,
    /// Candidate wins of this kind.
    pub candidate_events: u32,
    /// Candidate wins in total (the share's denominator).
    pub candidate_trials: u32,
    /// `candidate_events / candidate_trials`.
    pub candidate_value: f64,
    /// Control wins of this kind.
    pub control_events: u32,
    /// Control wins in total.
    pub control_trials: u32,
    /// `control_events / control_trials`.
    pub control_value: f64,
    /// `candidate_value / control_value`, or `None` when the control's share
    /// is zero and the ratio is undefined.
    pub ratio: Option<f64>,
    /// The share the bound actually required of the candidate (the threshold
    /// itself for an absolute bound, `threshold * control_value` for a
    /// relative one).
    pub required_value: f64,
    /// The one-sided z statistic of the observed share against
    /// `required_value`, or `None` when the check stopped at the evidence
    /// floor before computing one.
    pub z: Option<f64>,
    /// The outcome.
    pub outcome: GateOutcome,
    /// One sentence saying why, in the terms of the numbers above.
    pub note: String,
}

/// One budget's combined mechanism outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetGateOutcome {
    /// The budget label.
    pub budget: String,
    /// The worst outcome among that budget's checks.
    pub outcome: GateOutcome,
}

/// A whole gate evaluation: every check, each budget's combined outcome, and
/// the run's mechanism verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MechanismReport {
    /// The gate spec that was evaluated, as written.
    pub gate: String,
    /// The evidence floor that was in force.
    pub min_expected_events: f64,
    /// The significance threshold that was in force.
    pub z_critical: f64,
    /// Every check, budget-major.
    pub checks: Vec<MechanismCheck>,
    /// Per-budget combined outcomes, in the order the budgets were given.
    pub per_budget: Vec<BudgetGateOutcome>,
    /// The worst per-budget outcome — the run's mechanism verdict.
    pub verdict: GateOutcome,
}

/// `events / trials`, with no trials reading as `0.0`.
fn share(events: u32, trials: u32) -> f64 {
    if trials == 0 {
        0.0
    } else {
        events as f64 / trials as f64
    }
}

/// Render a threshold the way a human wrote it: `2` rather than `2.00`, but
/// `0.85` kept exact.
fn format_threshold(t: f64) -> String {
    if (t - t.round()).abs() < 1e-9 {
        format!("{}", t.round() as i64)
    } else {
        let s = format!("{t:.4}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts built by hand, so every branch of the statistics can be checked
    /// without playing a single game (the same trick
    /// `leaderboard::tests::synthetic_round_robin` and
    /// `experiment::tests::synthetic_cell` use). Each tuple is `(wins,
    /// military, science, civilian, tiebreak, military-exposed wins,
    /// science-exposed wins)`.
    fn counts(
        cand: (u32, u32, u32, u32, u32, u32, u32),
        ctrl: (u32, u32, u32, u32, u32, u32, u32),
    ) -> MechanismCounts {
        let side = |t: (u32, u32, u32, u32, u32, u32, u32)| SideCounts {
            wins: t.0,
            military_supremacy: t.1,
            scientific_supremacy: t.2,
            civilian_victory: t.3,
            civilian_tiebreak: t.4,
            military_exposed_wins: t.5,
            science_exposed_wins: t.6,
        };
        MechanismCounts {
            candidate: side(cand),
            control: side(ctrl),
        }
    }

    /// `(wins, events)` per side for one metric; every other field is zero,
    /// which is all a single-metric test reads.
    fn one_metric(metric: MechanismMetric, cand: (u32, u32), ctrl: (u32, u32)) -> MechanismCounts {
        let side = |(wins, events): (u32, u32)| {
            let mut s = SideCounts {
                wins,
                ..Default::default()
            };
            match metric {
                MechanismMetric::MilitaryShare => s.military_supremacy = events,
                MechanismMetric::ScienceShare => s.scientific_supremacy = events,
                MechanismMetric::CivilianShare => s.civilian_victory = events,
                MechanismMetric::TiebreakShare => s.civilian_tiebreak = events,
                MechanismMetric::MilitaryExposure => s.military_exposed_wins = events,
                MechanismMetric::ScienceExposure => s.science_exposed_wins = events,
            }
            s
        };
        MechanismCounts {
            candidate: side(cand),
            control: side(ctrl),
        }
    }

    fn gate(spec: &str) -> MechanismGate {
        MechanismGate::parse(spec).unwrap().unwrap()
    }

    fn only_check(spec: &str, counts: &MechanismCounts) -> MechanismCheck {
        let g = gate(spec);
        let mut checks = g.evaluate_budget("nodes:2000", counts);
        assert_eq!(checks.len(), 1, "this helper is for single-bound gates");
        checks.pop().unwrap()
    }

    // --- parsing ---------------------------------------------------------

    #[test]
    fn parses_a_directional_gate_list_in_order() {
        let g = gate("science_share>=0.8x,military_share<=1.5x,civilian_share>=0.25");
        assert_eq!(g.bounds.len(), 3);
        assert_eq!(g.bounds[0].metric, MechanismMetric::ScienceShare);
        assert_eq!(g.bounds[0].direction, BoundDirection::AtLeast);
        assert_eq!(g.bounds[0].threshold, 0.8);
        assert!(g.bounds[0].relative);
        assert_eq!(g.bounds[1].direction, BoundDirection::AtMost);
        assert!(g.bounds[1].relative);
        assert!(
            !g.bounds[2].relative,
            "no trailing x means an absolute share"
        );
        assert_eq!(g.min_expected_events, DEFAULT_MIN_EXPECTED_EVENTS);
        assert_eq!(g.z_critical, DEFAULT_Z_CRITICAL);
        // Round-trips through the label the report prints.
        assert_eq!(g.bounds[0].label(), "science_share>=0.8x");
        assert_eq!(g.bounds[1].label(), "military_share<=1.5x");
        assert_eq!(g.bounds[2].label(), "civilian_share>=0.25");
        // Whitespace around the pieces is tolerated.
        assert_eq!(gate(" science_share >= 0.8x ").bounds.len(), 1);
    }

    #[test]
    fn every_metric_name_parses_and_the_default_gate_is_valid() {
        for m in MechanismMetric::ALL {
            assert_eq!(MechanismMetric::parse(m.name()).unwrap(), m);
        }
        let g = gate(DEFAULT_GATE);
        assert_eq!(g.bounds.len(), 3);
        assert!(
            g.bounds.iter().all(|b| b.relative),
            "the default gate is ratios only, so it hardcodes no project-specific rate"
        );
        // The exact shape of the failure this module was built for.
        let metrics: Vec<&str> = g.bounds.iter().map(|b| b.metric.name()).collect();
        assert!(metrics.contains(&"civilian_share"));
        assert!(metrics.contains(&"military_share"));
    }

    #[test]
    fn none_off_and_disabled_mean_no_gate_at_all() {
        for spelling in GATE_DISABLED_SPELLINGS {
            assert_eq!(MechanismGate::parse(spelling).unwrap(), None);
            assert_eq!(
                MechanismGate::parse(&spelling.to_uppercase()).unwrap(),
                None
            );
        }
        assert!(
            MechanismGate::parse("").is_err(),
            "empty is a mistake, not a disable"
        );
    }

    #[test]
    fn rejects_malformed_duplicated_and_impossible_bounds() {
        let err = |s: &str| MechanismGate::parse(s).unwrap_err();
        assert!(err("science_share>0.8").contains("no comparison"));
        assert!(err("frobnicate>=0.8").contains("unknown gate metric"));
        assert!(err("science_share>=abc").contains("non-numeric"));
        assert!(err("science_share>=-1x").contains("non-negative"));
        assert!(err("science_share>=0x").contains("vacuous"));
        // An absolute share above 1 is almost always a forgotten `x`.
        let hint = err("military_share<=1.5");
        assert!(hint.contains("did you mean"), "unexpected: {hint}");
        assert!(hint.contains("military_share<=1.5x"));
        // Same metric, same direction, twice.
        assert!(err("science_share>=0.8x,science_share>=0.9x").contains("more than once"));
        // ...but both directions on one metric is a legitimate band.
        assert_eq!(
            gate("science_share>=0.8x,science_share<=1.5x").bounds.len(),
            2
        );
    }

    // --- the statistics: failing -----------------------------------------

    /// The motivating case, as a test: a candidate whose wins moved wholesale
    /// from civilian score to military supremacy. Aggregate Elo called this
    /// nearly a wash; the gate must not.
    #[test]
    fn a_candidate_that_abandoned_civilian_judgement_fails_the_default_gate() {
        // 300 wins each side. The control wins mostly on points; the candidate
        // has become a military-race detector.
        let counts = counts(
            (300, 240, 5, 40, 15, 260, 20),
            (300, 90, 6, 180, 24, 150, 22),
        );
        let report = gate(DEFAULT_GATE).evaluate(&[("nodes:2000".to_string(), counts)]);
        assert_eq!(report.verdict, GateOutcome::Fail);
        assert_eq!(report.per_budget.len(), 1);
        assert_eq!(report.per_budget[0].outcome, GateOutcome::Fail);

        let civilian = report
            .checks
            .iter()
            .find(|c| c.metric == "civilian_share")
            .unwrap();
        assert_eq!(civilian.outcome, GateOutcome::Fail);
        assert!((civilian.candidate_value - 40.0 / 300.0).abs() < 1e-12);
        assert!((civilian.control_value - 180.0 / 300.0).abs() < 1e-12);
        assert!(civilian.ratio.unwrap() < 0.5);
        assert!(civilian.z.unwrap() < -DEFAULT_Z_CRITICAL);
        assert!(civilian.note.contains("standard errors below"));

        let military = report
            .checks
            .iter()
            .find(|c| c.metric == "military_share")
            .unwrap();
        assert_eq!(
            military.outcome,
            GateOutcome::Fail,
            "2x the control's military share is a gross style change"
        );
        assert!(military.z.unwrap() > DEFAULT_Z_CRITICAL);

        // Science supremacy at ~2% of wins is exactly the rare event the gate
        // must refuse to rule on rather than guess about.
        let science = report
            .checks
            .iter()
            .find(|c| c.metric == "science_share")
            .unwrap();
        assert_eq!(science.outcome, GateOutcome::Inconclusive);
        assert!(science.note.contains("inconclusive"));
    }

    #[test]
    fn a_ceiling_fails_only_when_the_excess_is_beyond_noise() {
        // Control: 30% military of 400 wins, so the ceiling sits at 36%.
        let clear = one_metric(MechanismMetric::MilitaryShare, (400, 240), (400, 120));
        let c = only_check("military_share<=1.2x", &clear);
        assert_eq!(c.outcome, GateOutcome::Fail);
        assert!(c.z.unwrap() > DEFAULT_Z_CRITICAL);

        let under = one_metric(MechanismMetric::MilitaryShare, (400, 130), (400, 120));
        let c = only_check("military_share<=1.2x", &under);
        assert_eq!(
            c.outcome,
            GateOutcome::Pass,
            "32.5% is under the permitted 36%, so the bound is simply satisfied"
        );

        // Just over the line, but well inside the sampling spread.
        let noisy = one_metric(MechanismMetric::MilitaryShare, (400, 150), (400, 120));
        let c = only_check("military_share<=1.2x", &noisy);
        assert_eq!(c.outcome, GateOutcome::Inconclusive);
        assert!(c.z.unwrap() > 0.0 && c.z.unwrap() < DEFAULT_Z_CRITICAL);
        assert!(c.note.contains("sampling noise"));
    }

    #[test]
    fn a_floor_is_asymmetric_and_says_nothing_about_an_improvement() {
        // The candidate tripled its science share. A `>=` floor must pass it.
        let better = one_metric(MechanismMetric::ScienceShare, (400, 90), (400, 30));
        let c = only_check("science_share>=0.8x", &better);
        assert_eq!(c.outcome, GateOutcome::Pass);
        assert!(c.ratio.unwrap() > 2.5);

        // The same profile with a ceiling instead does fail - which is exactly
        // why bounds are one-sided and a caller states only what it cares
        // about.
        let c = only_check("science_share<=1.5x", &better);
        assert_eq!(c.outcome, GateOutcome::Fail);
    }

    #[test]
    fn absolute_bounds_ignore_the_control_entirely() {
        // The control's profile is irrelevant to an absolute bound...
        let counts = one_metric(MechanismMetric::CivilianShare, (400, 80), (400, 400));
        let c = only_check("civilian_share>=0.4", &counts);
        assert_eq!(c.outcome, GateOutcome::Fail);
        assert_eq!(c.required_value, 0.4);
        // ...and an absolute bound needs no control events at all, so it still
        // resolves where a relative one has nothing to reference.
        let lonely = one_metric(MechanismMetric::CivilianShare, (400, 80), (0, 0));
        assert_eq!(
            only_check("civilian_share>=0.4", &lonely).outcome,
            GateOutcome::Fail
        );
        assert_eq!(
            only_check("civilian_share>=0.9x", &lonely).outcome,
            GateOutcome::Inconclusive
        );
    }

    // --- the statistics: refusing to rule --------------------------------

    #[test]
    fn a_rare_event_at_a_realistic_sample_size_is_inconclusive_not_a_verdict() {
        // Scientific supremacy is ~2.3% of games in this project's self-play
        // data, so a 1,000-game cell holds ~20-25 of them across both sides.
        // A candidate at 4 and a control at 11 "looks like" a 0.36x collapse,
        // and a naive ratio gate would fail it. There is not remotely enough
        // evidence for that.
        let counts = one_metric(MechanismMetric::ScienceShare, (500, 4), (500, 11));
        let c = only_check("science_share>=0.8x", &counts);
        assert!(c.ratio.unwrap() < 0.4, "the raw ratio really does look bad");
        assert_eq!(c.outcome, GateOutcome::Inconclusive);
        assert!(
            c.note.contains("sampling noise"),
            "here the noise test is what refuses to rule: {}",
            c.note
        );

        // Fewer events still and the check never even reaches the noise test:
        // the evidence floor stops it, and says so.
        let thinner = one_metric(MechanismMetric::ScienceShare, (100, 0), (100, 5));
        let c = only_check("science_share>=0.8x", &thinner);
        assert_eq!(c.outcome, GateOutcome::Inconclusive);
        assert!(c.note.contains("expected win"), "unexpected: {}", c.note);
        assert_eq!(c.z, None);

        // The same *ratio* on ten times the events does resolve: the gate is
        // sample-size aware, not metric-blind.
        let plenty = one_metric(MechanismMetric::ScienceShare, (5000, 40), (5000, 110));
        let c = only_check("science_share>=0.8x", &plenty);
        assert_eq!(c.outcome, GateOutcome::Fail);
    }

    #[test]
    fn a_relative_bound_needs_control_events_to_reference() {
        // Two control events is no basis for a ratio, however many games the
        // candidate won.
        let counts = one_metric(MechanismMetric::ScienceShare, (400, 0), (400, 2));
        let c = only_check("science_share>=0.8x", &counts);
        assert_eq!(c.outcome, GateOutcome::Inconclusive);
        assert!(
            c.note.contains("control won only 2"),
            "unexpected: {}",
            c.note
        );
        assert_eq!(c.z, None, "the check stopped before computing a statistic");
    }

    #[test]
    fn zero_observed_events_does_not_produce_infinite_significance() {
        // Variance under the boundary hypothesis, not the observed one: a
        // candidate at exactly 0 is judged on how many events the bound
        // expected, not declared infinitely significant.
        let thin = one_metric(MechanismMetric::ScienceShare, (200, 0), (200, 8));
        let c = only_check("science_share>=0.8x", &thin);
        assert_eq!(c.candidate_value, 0.0);
        assert!(c.z.is_some_and(|z| z.is_finite()));

        // With enough events behind it, the same zero does fail.
        let solid = one_metric(MechanismMetric::ScienceShare, (2000, 0), (2000, 80));
        let c = only_check("science_share>=0.8x", &solid);
        assert_eq!(c.outcome, GateOutcome::Fail);
        assert!(c.z.unwrap().is_finite());
        assert!(c.z.unwrap() < -DEFAULT_Z_CRITICAL);
    }

    #[test]
    fn no_wins_on_either_side_is_inconclusive_rather_than_a_pass() {
        let none = one_metric(MechanismMetric::CivilianShare, (0, 0), (400, 200));
        let c = only_check("civilian_share>=0.8x", &none);
        assert_eq!(c.outcome, GateOutcome::Inconclusive);
        assert!(c.note.contains("candidate won no games"));

        let no_control = one_metric(MechanismMetric::CivilianShare, (400, 200), (0, 0));
        let c = only_check("civilian_share>=0.8x", &no_control);
        assert_eq!(c.outcome, GateOutcome::Inconclusive);
        assert!(c.note.contains("control won no games"));
        assert_eq!(c.ratio, None);
    }

    #[test]
    fn a_ceiling_the_data_cannot_violate_passes_rather_than_failing() {
        // 3x a 40% control share is 120% - above every possible candidate
        // share, so the ceiling is trivially satisfied.
        let counts = one_metric(MechanismMetric::MilitaryShare, (400, 400), (400, 160));
        let c = only_check("military_share<=3x", &counts);
        assert_eq!(c.outcome, GateOutcome::Pass);
        assert!(c.required_value > 1.0);
        // The mirror-image floor is unreachable, and fails.
        let c = only_check("military_share>=3x", &counts);
        assert_eq!(c.outcome, GateOutcome::Fail);
    }

    // --- combining -------------------------------------------------------

    #[test]
    fn budgets_combine_worst_first_and_are_never_pooled_together() {
        let clean = counts(
            (300, 60, 20, 180, 40, 150, 60),
            (300, 60, 20, 180, 40, 150, 60),
        );
        let broken = counts(
            (300, 240, 20, 20, 20, 260, 60),
            (300, 60, 20, 180, 40, 150, 60),
        );

        let g = gate(DEFAULT_GATE);
        let both_clean = g.evaluate(&[
            ("nodes:2000".to_string(), clean),
            ("time_ms:100".to_string(), clean),
        ]);
        assert_eq!(both_clean.verdict, GateOutcome::Pass);
        assert_eq!(both_clean.per_budget.len(), 2);
        assert_eq!(both_clean.checks.len(), 6, "3 bounds x 2 budgets");

        // Clean at one budget, broken at the other: the run fails. A change
        // this project ships has to hold at every budget tested.
        let mixed = g.evaluate(&[
            ("nodes:2000".to_string(), clean),
            ("time_ms:100".to_string(), broken),
        ]);
        assert_eq!(mixed.verdict, GateOutcome::Fail);
        assert_ne!(mixed.per_budget[0].outcome, GateOutcome::Fail);
        assert_eq!(mixed.per_budget[1].outcome, GateOutcome::Fail);
        // Every check names the budget it came from, so nothing is merged.
        assert!(mixed.checks.iter().any(|c| c.budget == "nodes:2000"));
        assert!(mixed.checks.iter().any(|c| c.budget == "time_ms:100"));
    }

    #[test]
    fn a_gate_with_nothing_to_evaluate_is_inconclusive() {
        let report = gate(DEFAULT_GATE).evaluate(&[]);
        assert_eq!(report.verdict, GateOutcome::Inconclusive);
        assert!(report.checks.is_empty());
        assert_eq!(combine([]), GateOutcome::Inconclusive);
        assert_eq!(
            combine([GateOutcome::Pass, GateOutcome::Inconclusive]),
            GateOutcome::Inconclusive
        );
        assert_eq!(
            combine([
                GateOutcome::Fail,
                GateOutcome::Inconclusive,
                GateOutcome::Pass
            ]),
            GateOutcome::Fail
        );
        assert_eq!(
            combine([GateOutcome::Pass, GateOutcome::Pass]),
            GateOutcome::Pass
        );
    }

    #[test]
    fn a_self_play_profile_is_never_a_failure() {
        // The identical profile on both sides is the sanity check: a candidate
        // that *is* the control must not be rejected by any bound, in either
        // direction, at any of the sample sizes this tool runs at.
        for wins in [20u32, 100, 400, 2000] {
            let side = SideCounts {
                wins,
                military_supremacy: wins * 2 / 10,
                scientific_supremacy: wins / 50,
                civilian_victory: wins * 6 / 10,
                civilian_tiebreak: wins / 10,
                military_exposed_wins: wins / 2,
                science_exposed_wins: wins / 5,
            };
            let counts = MechanismCounts {
                candidate: side,
                control: side,
            };
            let spec = "military_share<=1x,science_share>=1x,civilian_share>=1x,\
                        tiebreak_share>=1x,military_exposure>=1x,science_exposure>=1x";
            let report = gate(spec).evaluate(&[("nodes:200".to_string(), counts)]);
            assert_ne!(
                report.verdict,
                GateOutcome::Fail,
                "self-play at {wins} wins must never fail a mechanism gate: {:?}",
                report
                    .checks
                    .iter()
                    .filter(|c| c.outcome == GateOutcome::Fail)
                    .map(|c| c.note.clone())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn exposure_metrics_read_the_per_side_win_counts_not_the_match_rate() {
        use crate::match_runner::{SideWinRaceExposure, VictoryBreakdown};

        let victories = MatchVictoryBreakdown {
            a: VictoryBreakdown {
                military_supremacy: 10,
                scientific_supremacy: 2,
                civilian_victory: 30,
                civilian_tiebreak: 8,
            },
            b: VictoryBreakdown {
                military_supremacy: 5,
                scientific_supremacy: 1,
                civilian_victory: 40,
                civilian_tiebreak: 4,
            },
        };
        let exposure = WinRaceExposure {
            a: SideWinRaceExposure {
                wins: 50,
                military: 25,
                science: 10,
            },
            b: SideWinRaceExposure {
                wins: 50,
                military: 20,
                science: 5,
            },
        };
        let counts = MechanismCounts::from_parts(&victories, &exposure);
        assert_eq!(counts.candidate.wins, 50, "the victory breakdown's total");
        assert_eq!(counts.candidate.military_exposed_wins, 25);
        assert_eq!(counts.control.science_exposed_wins, 5);
        assert!((counts.candidate.share_of(MechanismMetric::MilitaryExposure) - 0.5).abs() < 1e-12);
        assert!(
            (counts.candidate.share_of(MechanismMetric::CivilianShare) - 0.6).abs() < 1e-12,
            "shares are of that side's own wins, not of all games"
        );
        // Every metric reads its own field, and none reads another's.
        for m in MechanismMetric::ALL {
            let v = counts.candidate.share_of(m);
            assert!((0.0..=1.0).contains(&v), "{} out of range: {v}", m.name());
        }
        assert_eq!(
            SideCounts::default().share_of(MechanismMetric::MilitaryShare),
            0.0
        );
    }

    #[test]
    fn thresholds_render_the_way_they_were_written() {
        assert_eq!(format_threshold(2.0), "2");
        assert_eq!(format_threshold(1.5), "1.5");
        assert_eq!(format_threshold(0.8), "0.8");
        assert_eq!(format_threshold(0.25), "0.25");
        assert_eq!(format_threshold(0.0), "0");
    }

    #[test]
    fn the_knobs_actually_move_the_outcome() {
        let counts = one_metric(MechanismMetric::ScienceShare, (500, 4), (500, 11));
        // Default: inconclusive at the evidence floor.
        let mut g = gate("science_share>=0.8x");
        assert_eq!(
            g.evaluate_budget("b", &counts)[0].outcome,
            GateOutcome::Inconclusive
        );
        // Drop the floor and the check runs; it is still noise-tested.
        g.min_expected_events = 1.0;
        let c = &g.evaluate_budget("b", &counts)[0];
        assert!(c.z.is_some(), "the floor no longer stops it");
        let with_default_z = c.outcome;
        // A much laxer significance threshold turns the same numbers into a
        // failure, which is the knob doing its job.
        g.z_critical = 0.5;
        let laxer = g.evaluate_budget("b", &counts)[0].outcome;
        assert!(
            with_default_z == GateOutcome::Inconclusive && laxer == GateOutcome::Fail,
            "{with_default_z:?} then {laxer:?}"
        );
    }

    #[test]
    fn a_report_round_trips_through_json() {
        let counts = counts(
            (300, 240, 5, 40, 15, 260, 20),
            (300, 90, 6, 180, 24, 150, 22),
        );
        let report = gate(DEFAULT_GATE).evaluate(&[("nodes:2000".to_string(), counts)]);
        let json = serde_json::to_string(&report).unwrap();
        let back: MechanismReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
        // The verdict is its own snake_case word in the JSON, so a consumer
        // can branch on it without parsing prose.
        assert!(json.contains("\"verdict\":\"fail\""));
    }
}
