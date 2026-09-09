//! The measurement protocol as a tool, rather than as a ritual.
//!
//! `CLAUDE.md` spells out how this project validates an agent change: paired
//! seeds and swapped seats, **two or more disjoint seed ranges**, **both a
//! `Nodes` and a `TimeMs` budget**, an explicit control arm, Elo with a
//! confidence interval rather than a bare win count, and an SPRT verdict. All
//! of that was prose, executed by hand as a series of `duels-arena match`
//! invocations whose numbers were then transcribed somewhere by eye — which
//! is how a session ends up hunting an earlier run's raw JSON out of
//! `arena/results/` to pool it with a later one.
//!
//! This module runs that protocol as one command and emits **one
//! machine-readable verdict** ([`ExperimentSummary`], written as JSON)
//! alongside a human-readable table ([`render_markdown`]).
//!
//! # The shape of a run
//!
//! A *cell* is one (seed range × budget) pair, played as an ordinary
//! paired-seed, seat-swapped match between the candidate and the control (see
//! [`crate::match_runner::play_paired_match`] — nothing about match playing is
//! reimplemented here, this is only orchestration). Cells are enumerated
//! budget-major ([`plan_cells`]) and played **one at a time**, so a `TimeMs`
//! cell never contends with another cell for CPU — see the "quiet machine"
//! note in the crate docs, which is exactly why this tool does not fan cells
//! out in parallel even though it easily could.
//!
//! Each cell reports its own Elo (95% CI) and SPRT decision, and every cell's
//! raw per-game records are written to their own results file in the same
//! format `duels-arena match` writes, so a later run can re-pool them.
//!
//! # Pooling, and what is deliberately *not* pooled
//!
//! Seed ranges within one budget are pooled: their win/loss/draw counts are
//! summed and [`crate::elo::fit_elo`] / [`crate::sprt::sprt`] are re-run on
//! the total ([`pool_by_budget`]). That is the "reproduce on a second,
//! disjoint seed range" step turned into a number.
//!
//! Different budgets are **never** pooled with each other. Pooling a `Nodes`
//! arm with a `TimeMs` arm would average two different experiments; the
//! project's convention is that a change has to hold at *both*, which is a
//! conjunction, not a mean. [`overall_verdict`] therefore takes the
//! conjunction over the per-budget pooled decisions rather than fitting
//! anything across them.
//!
//! # Early stopping
//!
//! `--early-stop` evaluates the SPRT after every chunk of games inside a cell
//! and abandons the rest of the cell once the decision is no longer
//! `Continue`. This needed no change to [`crate::match_runner`]: a cell's
//! seeds are independent (each `play_pair` depends only on its own seed), so
//! playing them in chunks yields exactly the games a single call would have
//! played, and `tests::chunked_play_is_identical_to_one_call` asserts that
//! move for move. It is opt-in because a stopped cell reports fewer games
//! than it was configured for, which is the right trade only when the caller
//! wants a verdict rather than a fixed-precision estimate.
//!
//! # Cost
//!
//! Total work is `sum over cells of games`, i.e. `ranges × budgets × games`.
//! At `nodes:2000` a search-agent game costs on the order of a second of one
//! core, and cells run sequentially while seeds within a cell run in
//! parallel — so a 2-range × 2-budget × 400-game experiment is 1,600 games and
//! is measured in tens of minutes, not seconds. The command prints its cell
//! plan before playing anything so the size of what was just asked for is
//! visible up front.

use std::path::{Path, PathBuf};

use duels_agents_api::Budget;
use serde::{Deserialize, Serialize};

use crate::elo::{fit_elo, EloEstimate};
use crate::match_runner::{
    parse_budget, play_paired_match, race_exposure, tally, GameRecord, MatchTally,
    MatchVictoryBreakdown, RaceExposure,
};
use crate::results_io::write_results;
use crate::sprt::{sprt, SprtDecision, SprtParams, SprtResult};

/// Schema version of [`ExperimentSummary`], so a future consumer can tell an
/// old summary file from a new one.
pub const SCHEMA_VERSION: u32 = 1;

/// H0 for an experiment's SPRT: "the candidate is no better than the
/// control". Matches how this project has used SPRT by hand.
pub const DEFAULT_ELO0: f64 = 0.0;

/// H1 for an experiment's SPRT: "the candidate is worth 20 Elo". Larger than
/// [`crate::sprt::SprtParams::default`]'s `elo1 = 5` on purpose: 5 Elo is
/// below the resolution this project's game counts can actually reach, and 20
/// is the threshold ad hoc runs here have used.
pub const DEFAULT_ELO1: f64 = 20.0;

/// Where an experiment's per-cell results files and summary go by default.
/// Under `arena/results/`, which `.gitignore` covers.
pub const DEFAULT_OUT_DIR: &str = "arena/results/experiments";

/// Games per SPRT check when `--early-stop` is on and no `--check-every` was
/// given. Half a chunk of parallel work at a time on a typical machine.
pub const DEFAULT_CHECK_EVERY_GAMES: u32 = 50;

/// A contiguous block of setup seeds, each played twice (both seat
/// assignments) — so `pairs` seeds are `2 * pairs` games.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedRange {
    /// First setup seed in the range.
    pub start: u64,
    /// How many consecutive seeds the range covers.
    pub pairs: u32,
}

impl SeedRange {
    /// One past the last seed in the range.
    pub fn end(&self) -> u64 {
        self.start + self.pairs as u64
    }

    /// Individual games this range represents (`2 * pairs`).
    pub fn games(&self) -> u32 {
        self.pairs * 2
    }

    /// Every setup seed in the range.
    pub fn seeds(&self) -> Vec<u64> {
        (self.start..self.end()).collect()
    }
}

/// One (seed range × budget) cell of an experiment: a single paired-seed,
/// seat-swapped match to play.
#[derive(Debug, Clone, PartialEq)]
pub struct CellPlan {
    /// The budget as the caller wrote it (`"nodes:2000"`), kept verbatim so
    /// the report and the pooling key read the same as the command line.
    pub budget_label: String,
    /// The parsed budget handed to every agent in this cell.
    pub budget: Budget,
    /// The seeds this cell plays.
    pub range: SeedRange,
}

/// Everything one `duels-arena experiment` run needs, after parsing.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentPlan {
    /// Candidate agent specification string (see [`crate::agent_spec`]).
    /// Always "role A", so a positive Elo means the candidate is ahead.
    pub candidate: String,
    /// Control/baseline agent specification string.
    pub control: String,
    /// Every cell to play, in the order they will be played.
    pub cells: Vec<CellPlan>,
    /// The hypotheses and error rates every SPRT in this run uses.
    pub sprt: SprtParams,
    /// Short name for this run; also the output subdirectory.
    pub label: String,
    /// Whether to abandon the rest of a cell once its running SPRT decides.
    pub early_stop: bool,
    /// Games between SPRT checks inside a cell. `None` plays each cell in one
    /// call (the only shape when `early_stop` is off).
    pub check_every_games: Option<u32>,
}

/// One cell's finished measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellSummary {
    /// Index of this cell in the run's plan, for cross-referencing.
    pub cell: u32,
    /// The budget label this cell was played at; also its pooling key.
    pub budget: String,
    /// First setup seed played.
    pub seed_start: u64,
    /// One past the last setup seed played (the range as configured, even if
    /// the cell stopped early).
    pub seed_end: u64,
    /// Games the cell was configured to play.
    pub games_planned: u32,
    /// Games actually played (less than planned only if it stopped early).
    pub games_played: u32,
    /// Whether the running SPRT ended this cell before its planned games.
    pub stopped_early: bool,
    /// Win/loss/draw from the **candidate's** perspective.
    pub tally: MatchTally,
    /// Candidate Elo relative to the control, with its 95% CI.
    pub elo: EloEstimate,
    /// SPRT over this cell alone.
    pub sprt: SprtResult,
    /// How each side's wins were achieved.
    pub victory_breakdown: MatchVictoryBreakdown,
    /// How many games came within reach of an instant win.
    pub race_exposure: RaceExposure,
    /// Mean actions applied per game.
    pub avg_moves: f64,
    /// Mean wall-clock milliseconds per game.
    pub avg_wall_ms: f64,
    /// Total wall-clock milliseconds summed over the cell's games. Sums
    /// per-game timings, so on a machine playing seeds in parallel this
    /// exceeds the elapsed time the cell actually took.
    pub wall_ms_total: u64,
    /// Path the cell's raw per-game records were written to, relative to the
    /// run's output directory.
    pub results_file: String,
}

/// Every seed range at one budget, pooled: the figure that says whether a
/// result reproduced across disjoint seeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PooledSummary {
    /// The budget these cells share.
    pub budget: String,
    /// How many cells (seed ranges) were pooled.
    pub cells: u32,
    /// Games pooled.
    pub games: u32,
    /// Summed win/loss/draw from the candidate's perspective.
    pub tally: MatchTally,
    /// Elo refitted on the pooled counts (not an average of the cells').
    pub elo: EloEstimate,
    /// SPRT on the pooled counts.
    pub sprt: SprtResult,
    /// Summed victory-kind breakdown.
    pub victory_breakdown: MatchVictoryBreakdown,
    /// Summed race exposure.
    pub race_exposure: RaceExposure,
}

/// The one-word answer a consumer of the JSON can branch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Every budget's pooled SPRT accepted H1.
    Accept,
    /// At least one budget's pooled SPRT accepted H0. A change this project
    /// would ship has to hold at *every* budget tested, so one H0 is a
    /// rejection of the whole experiment regardless of the others.
    Reject,
    /// No budget rejected, but at least one has not resolved either way.
    Inconclusive,
}

/// The single machine-readable artifact this whole subcommand exists to
/// produce. Serialized to `<out-dir>/<label>/summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentSummary {
    /// [`SCHEMA_VERSION`].
    pub schema: u32,
    /// The run's short name.
    pub label: String,
    /// UTC RFC-3339 stamp, supplied by the caller (this module never reads a
    /// clock — see `clippy.toml`).
    pub generated_at: String,
    /// Candidate specification string, as written on the command line.
    pub candidate: String,
    /// Control specification string.
    pub control: String,
    /// The candidate's own reported [`duels_agents_api::AgentSpec`] params,
    /// read back off the first game actually played, so a summary records the
    /// configuration that ran rather than only the string that requested it.
    pub candidate_spec_params: String,
    /// The control's reported spec params, likewise.
    pub control_spec_params: String,
    /// Budgets tested, in the order given.
    pub budgets: Vec<String>,
    /// Seed ranges tested.
    pub seed_ranges: Vec<SeedRange>,
    /// The hypotheses every SPRT here was run against.
    pub sprt_params: SprtParams,
    /// Whether early stopping was enabled.
    pub early_stop: bool,
    /// Per-cell measurements, in plan order.
    pub cells: Vec<CellSummary>,
    /// Per-budget pooled measurements, in the order the budgets were given.
    pub pooled: Vec<PooledSummary>,
    /// The conjunction over `pooled` — see [`overall_verdict`].
    pub verdict: Verdict,
    /// Games played across every cell.
    pub total_games: u32,
}

/// Parse `--budgets` (`"nodes:2000,time_ms:100"`) into labelled budgets,
/// preserving order and rejecting duplicates (two cells that shared a pooling
/// key would silently merge into one pooled row).
pub fn parse_budgets(s: &str) -> Result<Vec<(String, Budget)>, String> {
    let mut out: Vec<(String, Budget)> = Vec::new();
    for piece in s.split(',') {
        let label = piece.trim();
        if label.is_empty() {
            return Err(format!("empty budget in --budgets \"{s}\""));
        }
        let budget = parse_budget(label)?;
        if out.iter().any(|(_, b)| *b == budget) {
            return Err(format!(
                "--budgets lists {label} more than once; each budget is its own pooling key, so \
                 duplicates would silently merge"
            ));
        }
        out.push((label.to_string(), budget));
    }
    if out.is_empty() {
        return Err("--budgets named no budgets".to_string());
    }
    Ok(out)
}

/// Parse `--seeds` into seed ranges. Two forms, mixable in one list:
///
/// * `<start>` — a base seed; the range covers `ceil(default_games / 2)`
///   consecutive seeds from there, i.e. `--games` decides its length.
/// * `<start>..<end>` — an explicit half-open seed range, which names its own
///   length (`end - start` seeds, so `2 * (end - start)` games) and ignores
///   `--games`.
///
/// Ranges are returned in the order given and checked for overlap: the whole
/// point of a second range is that it is *disjoint* evidence, so silently
/// re-measuring the same seeds would be the one failure this tool exists to
/// prevent.
pub fn parse_seed_ranges(s: &str, default_games: u32) -> Result<Vec<SeedRange>, String> {
    let default_pairs = default_games.div_ceil(2).max(1);
    let mut out: Vec<SeedRange> = Vec::new();
    for piece in s.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            return Err(format!("empty seed range in --seeds \"{s}\""));
        }
        let range = match piece.split_once("..") {
            None => {
                let start: u64 = piece
                    .parse()
                    .map_err(|_| format!("invalid seed \"{piece}\" in --seeds \"{s}\""))?;
                SeedRange {
                    start,
                    pairs: default_pairs,
                }
            }
            Some((start, end)) => {
                let start: u64 = start
                    .trim()
                    .parse()
                    .map_err(|_| format!("invalid range start in \"{piece}\""))?;
                let end: u64 = end
                    .trim()
                    .parse()
                    .map_err(|_| format!("invalid range end in \"{piece}\""))?;
                if end <= start {
                    return Err(format!(
                        "seed range \"{piece}\" is empty: end must be greater than start"
                    ));
                }
                let pairs = u32::try_from(end - start)
                    .map_err(|_| format!("seed range \"{piece}\" is implausibly large"))?;
                SeedRange { start, pairs }
            }
        };
        out.push(range);
    }
    if out.is_empty() {
        return Err("--seeds named no seed ranges".to_string());
    }
    check_disjoint(&out)?;
    Ok(out)
}

/// Reject seed ranges that overlap. See [`parse_seed_ranges`] for why this is
/// an error rather than a warning.
pub fn check_disjoint(ranges: &[SeedRange]) -> Result<(), String> {
    for (i, a) in ranges.iter().enumerate() {
        for b in &ranges[i + 1..] {
            if a.start < b.end() && b.start < a.end() {
                return Err(format!(
                    "seed ranges {}..{} and {}..{} overlap; disjoint ranges are the whole point of \
                     running more than one (see duels_arena::experiment)",
                    a.start,
                    a.end(),
                    b.start,
                    b.end()
                ));
            }
        }
    }
    Ok(())
}

/// Enumerate the cells of an experiment, budget-major: every seed range at
/// the first budget, then every seed range at the second, and so on. Budget
/// -major so that all of one budget's evidence exists before the next budget
/// starts, which is the order a human reads the report in.
pub fn plan_cells(ranges: &[SeedRange], budgets: &[(String, Budget)]) -> Vec<CellPlan> {
    let mut out = Vec::with_capacity(ranges.len() * budgets.len());
    for (label, budget) in budgets {
        for range in ranges {
            out.push(CellPlan {
                budget_label: label.clone(),
                budget: *budget,
                range: *range,
            });
        }
    }
    out
}

/// Sum two [`MatchVictoryBreakdown`]s field by field.
fn add_victory_breakdown(
    a: MatchVictoryBreakdown,
    b: MatchVictoryBreakdown,
) -> MatchVictoryBreakdown {
    let side = |x: crate::match_runner::VictoryBreakdown,
                y: crate::match_runner::VictoryBreakdown| {
        crate::match_runner::VictoryBreakdown {
            military_supremacy: x.military_supremacy + y.military_supremacy,
            scientific_supremacy: x.scientific_supremacy + y.scientific_supremacy,
            civilian_victory: x.civilian_victory + y.civilian_victory,
            civilian_tiebreak: x.civilian_tiebreak + y.civilian_tiebreak,
        }
    };
    MatchVictoryBreakdown {
        a: side(a.a, b.a),
        b: side(a.b, b.b),
    }
}

/// Pool every cell of each budget into one figure per budget, refitting Elo
/// and the SPRT on the summed counts. Budgets appear in the order their first
/// cell does; cells of different budgets are never mixed (see the module
/// docs).
pub fn pool_by_budget(cells: &[CellSummary], params: &SprtParams) -> Vec<PooledSummary> {
    let mut order: Vec<&str> = Vec::new();
    for c in cells {
        if !order.contains(&c.budget.as_str()) {
            order.push(&c.budget);
        }
    }

    order
        .into_iter()
        .map(|budget| {
            let mine = cells.iter().filter(|c| c.budget == budget);
            let mut tally = MatchTally::default();
            let mut vb = MatchVictoryBreakdown::default();
            let mut re = RaceExposure::default();
            let mut count = 0u32;
            for c in mine {
                tally.a_wins += c.tally.a_wins;
                tally.b_wins += c.tally.b_wins;
                tally.draws += c.tally.draws;
                vb = add_victory_breakdown(vb, c.victory_breakdown);
                re.military_games += c.race_exposure.military_games;
                re.science_games += c.race_exposure.science_games;
                re.total_games += c.race_exposure.total_games;
                count += 1;
            }
            PooledSummary {
                budget: budget.to_string(),
                cells: count,
                games: tally.total(),
                elo: fit_elo(tally.a_wins, tally.b_wins, tally.draws),
                sprt: sprt(tally.a_wins, tally.b_wins, tally.draws, params),
                tally,
                victory_breakdown: vb,
                race_exposure: re,
            }
        })
        .collect()
}

/// The conjunction over the per-budget pooled decisions: [`Verdict::Reject`]
/// if any budget accepted H0, [`Verdict::Accept`] only if *every* budget
/// accepted H1, [`Verdict::Inconclusive`] otherwise (including when there is
/// nothing to decide).
///
/// This is a reporting convenience, not a statistical statement about the
/// conjunction's error rates: each budget's SPRT controls its own `alpha` and
/// `beta`, and requiring several of them to agree makes the combined test
/// more conservative in the accept direction and less so in the reject
/// direction. Read the per-budget rows for the actual evidence.
pub fn overall_verdict(pooled: &[PooledSummary]) -> Verdict {
    if pooled.is_empty() {
        return Verdict::Inconclusive;
    }
    if pooled
        .iter()
        .any(|p| p.sprt.decision == SprtDecision::AcceptH0)
    {
        return Verdict::Reject;
    }
    if pooled
        .iter()
        .all(|p| p.sprt.decision == SprtDecision::AcceptH1)
    {
        return Verdict::Accept;
    }
    Verdict::Inconclusive
}

/// Turn an arbitrary string (an agent spec, a label) into something safe to
/// put in a path: ASCII alphanumerics, `-` and `.` survive, every other run of
/// characters collapses to a single `_`.
pub fn slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '.' {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "unnamed".to_string()
    } else {
        trimmed
    }
}

/// The default `--label` for a candidate-vs-control run.
pub fn default_label(candidate: &str, control: &str) -> String {
    format!("{}-vs-{}", slug(candidate), slug(control))
}

/// The filename one cell's raw per-game records are written under, inside the
/// run's output directory. Carries the budget and the seed range so a
/// directory of these is self-describing when re-pooled later.
///
/// The `n` is the cell's *planned* game count, because the name is needed
/// before the games are played; a cell that stopped early holds fewer records
/// than its name suggests, and [`CellSummary::games_played`] is the
/// authoritative count.
pub fn cell_results_filename(cell: &CellPlan) -> String {
    format!(
        "cell-{}-seed{}-n{}.json",
        slug(&cell.budget_label),
        cell.range.start,
        cell.range.games()
    )
}

/// Everything [`play_cell`] derives from a cell's finished games, before the
/// cell's plan and output path are attached.
struct CellOutcome {
    records: Vec<GameRecord>,
    stopped_early: bool,
}

/// Play one cell, optionally checking the SPRT every `check_every_games`
/// games and abandoning the rest of the cell once it decides.
///
/// Chunking is safe because a cell's seeds are independent: `play_pair`
/// depends only on its own seed, so the games are the same games a single
/// call would have played (asserted by
/// `tests::chunked_play_is_identical_to_one_call`).
fn play_cell(
    candidate: &str,
    control: &str,
    cell: &CellPlan,
    params: &SprtParams,
    early_stop: bool,
    check_every_games: Option<u32>,
) -> Result<CellOutcome, String> {
    let seeds = cell.range.seeds();
    let chunk_pairs = match check_every_games {
        Some(games) => (games.div_ceil(2).max(1)) as usize,
        None => seeds.len().max(1),
    };

    let mut records: Vec<GameRecord> = Vec::with_capacity(seeds.len() * 2);
    let mut stopped_early = false;
    for chunk in seeds.chunks(chunk_pairs) {
        let mut part = play_paired_match(candidate, control, chunk, cell.budget)?;
        records.append(&mut part);
        if early_stop && records.len() < seeds.len() * 2 {
            let t = tally(&records);
            if sprt(t.a_wins, t.b_wins, t.draws, params).decision != SprtDecision::Continue {
                stopped_early = true;
                break;
            }
        }
    }
    // `play_paired_match` sorts within a chunk; re-sort so a chunked cell's
    // records file is byte-identical in ordering to an unchunked one.
    records.sort_by_key(|r| (r.seed, r.agent_a_seat));

    Ok(CellOutcome {
        records,
        stopped_early,
    })
}

/// Derive a cell's summary from its finished games.
fn summarize_cell(
    index: u32,
    cell: &CellPlan,
    outcome: &CellOutcome,
    params: &SprtParams,
) -> CellSummary {
    let records = &outcome.records;
    let t = tally(records);
    let games = records.len() as u32;
    let total_moves: u64 = records.iter().map(|r| r.moves as u64).sum();
    let wall_ms_total: u64 = records.iter().map(|r| r.wall_time_ms).sum();
    let per_game = |x: f64| if games == 0 { 0.0 } else { x / games as f64 };

    CellSummary {
        cell: index,
        budget: cell.budget_label.clone(),
        seed_start: cell.range.start,
        seed_end: cell.range.end(),
        games_planned: cell.range.games(),
        games_played: games,
        stopped_early: outcome.stopped_early,
        tally: t,
        elo: fit_elo(t.a_wins, t.b_wins, t.draws),
        sprt: sprt(t.a_wins, t.b_wins, t.draws, params),
        victory_breakdown: crate::match_runner::victory_breakdown(records),
        race_exposure: race_exposure(records),
        avg_moves: per_game(total_moves as f64),
        avg_wall_ms: per_game(wall_ms_total as f64),
        wall_ms_total,
        results_file: cell_results_filename(cell),
    }
}

/// Run a whole experiment: play every cell in [`ExperimentPlan::cells`] in
/// order, write each cell's raw records under `out_dir`, and return the
/// pooled summary. `generated_at` is supplied by the caller because this
/// module reads no clock.
///
/// `progress` is called with one human-readable line per milestone (cell
/// started, cell finished) so the CLI can print while a long run is in
/// flight; pass a no-op closure to run silently.
pub fn run(
    plan: &ExperimentPlan,
    out_dir: &Path,
    generated_at: &str,
    progress: &mut dyn FnMut(&str),
) -> Result<ExperimentSummary, String> {
    if plan.cells.is_empty() {
        return Err("an experiment needs at least one cell".to_string());
    }

    let mut cells: Vec<CellSummary> = Vec::with_capacity(plan.cells.len());
    // Read off the first game actually played, so an agent whose reported
    // params are legitimately empty (`random`) is recorded as empty rather
    // than re-read every cell.
    let mut specs: Option<(String, String)> = None;

    for (i, cell) in plan.cells.iter().enumerate() {
        progress(&format!(
            "cell {}/{}: budget {} seeds {}..{} ({} games)",
            i + 1,
            plan.cells.len(),
            cell.budget_label,
            cell.range.start,
            cell.range.end(),
            cell.range.games(),
        ));

        let outcome = play_cell(
            &plan.candidate,
            &plan.control,
            cell,
            &plan.sprt,
            plan.early_stop,
            plan.check_every_games,
        )?;

        if specs.is_none() {
            if let Some(first) = outcome.records.first() {
                // Role A occupied `agent_a_seat` in that game; the other
                // spec is the control's.
                let (cand, ctrl) = match first.agent_a_seat {
                    duels_core::Player::One => (&first.seat_one, &first.seat_two),
                    duels_core::Player::Two => (&first.seat_two, &first.seat_one),
                };
                specs = Some((cand.params.clone(), ctrl.params.clone()));
            }
        }

        let summary = summarize_cell(i as u32, cell, &outcome, &plan.sprt);
        let path = out_dir.join(&summary.results_file);
        write_results(&path, &outcome.records)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;

        progress(&format!(
            "  -> {} games: {}-{}-{} (W-L-D for the candidate), elo {:+.1} [{:+.1}, {:+.1}], sprt {:?}{}",
            summary.games_played,
            summary.tally.a_wins,
            summary.tally.b_wins,
            summary.tally.draws,
            summary.elo.rating_diff,
            summary.elo.diff_ci_low,
            summary.elo.diff_ci_high,
            summary.sprt.decision,
            if summary.stopped_early {
                " (stopped early)"
            } else {
                ""
            },
        ));

        cells.push(summary);
    }

    let pooled = pool_by_budget(&cells, &plan.sprt);
    let verdict = overall_verdict(&pooled);
    let total_games = cells.iter().map(|c| c.games_played).sum();

    let mut seed_ranges: Vec<SeedRange> = Vec::new();
    for cell in &plan.cells {
        if !seed_ranges.contains(&cell.range) {
            seed_ranges.push(cell.range);
        }
    }
    let mut budgets: Vec<String> = Vec::new();
    for cell in &plan.cells {
        if !budgets.contains(&cell.budget_label) {
            budgets.push(cell.budget_label.clone());
        }
    }
    let (candidate_spec_params, control_spec_params) = specs.unwrap_or_default();

    Ok(ExperimentSummary {
        schema: SCHEMA_VERSION,
        label: plan.label.clone(),
        generated_at: generated_at.to_string(),
        candidate: plan.candidate.clone(),
        control: plan.control.clone(),
        candidate_spec_params,
        control_spec_params,
        budgets,
        seed_ranges,
        sprt_params: plan.sprt,
        early_stop: plan.early_stop,
        cells,
        pooled,
        verdict,
        total_games,
    })
}

/// Render an [`ExperimentSummary`] as Markdown, in the style of
/// [`crate::leaderboard::render_markdown`].
pub fn render_markdown(summary: &ExperimentSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Experiment: {}\n\n", summary.label));
    out.push_str(
        "Generated by `duels-arena experiment`. **Do not edit by hand** - re-run the command \
         instead.\n\n",
    );
    out.push_str(&format!(
        "- Generated: `{}`\n- Candidate: `{}`\n- Control: `{}`\n- Budgets: {}\n- Seed ranges: \
         {}\n- SPRT: H0 elo={:.1} vs H1 elo={:.1} (alpha={}, beta={})\n- Early stopping: \
         {}\n- Total games: {}\n- **Verdict: {:?}**\n\n",
        summary.generated_at,
        summary.candidate,
        summary.control,
        summary
            .budgets
            .iter()
            .map(|b| format!("`{b}`"))
            .collect::<Vec<_>>()
            .join(", "),
        summary
            .seed_ranges
            .iter()
            .map(|r| format!("`{}..{}`", r.start, r.end()))
            .collect::<Vec<_>>()
            .join(", "),
        summary.sprt_params.elo0,
        summary.sprt_params.elo1,
        summary.sprt_params.alpha,
        summary.sprt_params.beta,
        if summary.early_stop { "on" } else { "off" },
        summary.total_games,
        summary.verdict,
    ));

    out.push_str("## Pooled across seed ranges, per budget\n\n");
    out.push_str("| Budget | Cells | Games | W-L-D | Elo | 95% CI | LLR | SPRT |\n");
    out.push_str("| ------ | ----: | ----: | ----- | --: | ------ | --: | ---- |\n");
    for p in &summary.pooled {
        out.push_str(&format!(
            "| `{}` | {} | {} | {}-{}-{} | {:+.1} | [{:+.1}, {:+.1}] | {:.2} | {:?} |\n",
            p.budget,
            p.cells,
            p.games,
            p.tally.a_wins,
            p.tally.b_wins,
            p.tally.draws,
            p.elo.rating_diff,
            p.elo.diff_ci_low,
            p.elo.diff_ci_high,
            p.sprt.llr,
            p.sprt.decision,
        ));
    }

    out.push_str("\n## Per cell (seed range x budget)\n\n");
    out.push_str("| # | Budget | Seeds | Games | W-L-D | Elo | 95% CI | LLR | SPRT |\n");
    out.push_str("| -: | ------ | ----- | ----: | ----- | --: | ------ | --: | ---- |\n");
    for c in &summary.cells {
        out.push_str(&format!(
            "| {} | `{}` | {}..{} | {}{} | {}-{}-{} | {:+.1} | [{:+.1}, {:+.1}] | {:.2} | {:?} |\n",
            c.cell + 1,
            c.budget,
            c.seed_start,
            c.seed_end,
            c.games_played,
            if c.stopped_early { " (early)" } else { "" },
            c.tally.a_wins,
            c.tally.b_wins,
            c.tally.draws,
            c.elo.rating_diff,
            c.elo.diff_ci_low,
            c.elo.diff_ci_high,
            c.sprt.llr,
            c.sprt.decision,
        ));
    }

    out.push_str("\n## Victory kinds and race exposure, pooled per budget\n\n");
    out.push_str(
        "| Budget | Candidate wins (mil/sci/civ/tie) | Control wins (mil/sci/civ/tie) | \
         Military race | Science race |\n",
    );
    out.push_str("| ------ | ------------------------------- | ----------------------------- | ------------- | ------------ |\n");
    for p in &summary.pooled {
        let pct = |n: u32| {
            if p.race_exposure.total_games == 0 {
                0.0
            } else {
                100.0 * n as f64 / p.race_exposure.total_games as f64
            }
        };
        out.push_str(&format!(
            "| `{}` | {}/{}/{}/{} | {}/{}/{}/{} | {} ({:.0}%) | {} ({:.0}%) |\n",
            p.budget,
            p.victory_breakdown.a.military_supremacy,
            p.victory_breakdown.a.scientific_supremacy,
            p.victory_breakdown.a.civilian_victory,
            p.victory_breakdown.a.civilian_tiebreak,
            p.victory_breakdown.b.military_supremacy,
            p.victory_breakdown.b.scientific_supremacy,
            p.victory_breakdown.b.civilian_victory,
            p.victory_breakdown.b.civilian_tiebreak,
            p.race_exposure.military_games,
            pct(p.race_exposure.military_games),
            p.race_exposure.science_games,
            pct(p.race_exposure.science_games),
        ));
    }

    out.push_str(
        "\n## How to read this\n\n\
         Elo is the candidate's rating minus the control's, fitted from the win/loss/draw counts \
         with draws as half a win each side (`duels_arena::elo::fit_elo`); the interval is the 95% \
         asymptotic MLE interval. Every cell is a paired-seed, seat-swapped match, because \
         first-player advantage in this game is large enough to swamp the effects being \
         measured.\n\n\
         The pooled row per budget sums that budget's cells and refits, so it is the \
         \"reproduced on a second disjoint seed range\" figure - not an average of the cells. \
         Budgets are never pooled with each other: a change this project ships has to hold at \
         both a `Nodes` and a `TimeMs` budget, which is a conjunction rather than a mean, and \
         the verdict above is that conjunction.\n\n\
         A `TimeMs` cell is wall-clock based and therefore load-sensitive - see the \"quiet \
         machine\" note in `duels_arena`'s crate docs before trusting a small-sample `TimeMs` \
         row.\n",
    );

    out
}

/// Write `summary` as JSON and Markdown into `out_dir`, returning the two
/// paths.
pub fn write_summary(
    summary: &ExperimentSummary,
    out_dir: &Path,
) -> Result<(PathBuf, PathBuf), String> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;
    let json_path = out_dir.join("summary.json");
    let md_path = out_dir.join("summary.md");
    let json = serde_json::to_string_pretty(summary)
        .map_err(|e| format!("failed to serialize the experiment summary: {e}"))?;
    std::fs::write(&json_path, format!("{json}\n"))
        .map_err(|e| format!("failed to write {}: {e}", json_path.display()))?;
    std::fs::write(&md_path, render_markdown(summary))
        .map_err(|e| format!("failed to write {}: {e}", md_path.display()))?;
    Ok((json_path, md_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SprtParams {
        SprtParams {
            elo0: DEFAULT_ELO0,
            elo1: DEFAULT_ELO1,
            alpha: 0.05,
            beta: 0.05,
        }
    }

    /// A synthetic cell summary with hand-chosen counts, so the pooling math
    /// can be checked without playing games (the same trick
    /// `leaderboard::tests::synthetic_round_robin` uses).
    fn synthetic_cell(index: u32, budget: &str, start: u64, w: u32, l: u32, d: u32) -> CellSummary {
        let t = MatchTally {
            a_wins: w,
            b_wins: l,
            draws: d,
        };
        let games = t.total();
        CellSummary {
            cell: index,
            budget: budget.to_string(),
            seed_start: start,
            seed_end: start + (games / 2) as u64,
            games_planned: games,
            games_played: games,
            stopped_early: false,
            tally: t,
            elo: fit_elo(w, l, d),
            sprt: sprt(w, l, d, &params()),
            victory_breakdown: MatchVictoryBreakdown {
                a: crate::match_runner::VictoryBreakdown {
                    civilian_victory: w,
                    ..Default::default()
                },
                b: crate::match_runner::VictoryBreakdown {
                    military_supremacy: l,
                    ..Default::default()
                },
            },
            race_exposure: RaceExposure {
                military_games: l,
                science_games: 0,
                total_games: games,
            },
            avg_moves: 40.0,
            avg_wall_ms: 1.0,
            wall_ms_total: games as u64,
            results_file: "cell.json".to_string(),
        }
    }

    // --- parsing ---------------------------------------------------------

    #[test]
    fn parses_a_budget_list_in_order() {
        let got = parse_budgets("nodes:2000,time_ms:100").unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], ("nodes:2000".to_string(), Budget::Nodes(2000)));
        assert_eq!(got[1], ("time_ms:100".to_string(), Budget::TimeMs(100)));
    }

    #[test]
    fn rejects_a_duplicated_or_malformed_budget() {
        assert!(parse_budgets("nodes:2000,nodes:2000")
            .unwrap_err()
            .contains("more than once"));
        // `time` and `time_ms` are the same budget spelled two ways.
        assert!(parse_budgets("time:100,time_ms:100").is_err());
        assert!(parse_budgets("nodes:2000,").is_err());
        assert!(parse_budgets("frobnicate:1").is_err());
    }

    #[test]
    fn a_bare_seed_takes_its_length_from_the_game_count() {
        let got = parse_seed_ranges("1,10000", 200).unwrap();
        assert_eq!(
            got,
            vec![
                SeedRange {
                    start: 1,
                    pairs: 100
                },
                SeedRange {
                    start: 10_000,
                    pairs: 100
                }
            ]
        );
        assert_eq!(got[0].games(), 200);
        assert_eq!(got[0].end(), 101);
        // An odd game count rounds up to a whole pair, as `match` does.
        assert_eq!(parse_seed_ranges("1", 5).unwrap()[0].pairs, 3);
        // ...and zero still plays one pair rather than nothing.
        assert_eq!(parse_seed_ranges("1", 0).unwrap()[0].pairs, 1);
    }

    #[test]
    fn an_explicit_range_names_its_own_length() {
        let got = parse_seed_ranges("1..51,10000..10025", 999).unwrap();
        assert_eq!(
            got[0],
            SeedRange {
                start: 1,
                pairs: 50
            }
        );
        assert_eq!(got[0].games(), 100);
        assert_eq!(
            got[1],
            SeedRange {
                start: 10_000,
                pairs: 25
            }
        );
        assert_eq!(got[1].games(), 50);
    }

    #[test]
    fn overlapping_seed_ranges_are_rejected() {
        // Two bare seeds too close together for the configured game count.
        let err = parse_seed_ranges("1,50", 200).unwrap_err();
        assert!(err.contains("overlap"), "unexpected: {err}");
        // Explicit overlap, and an identical repeat.
        assert!(parse_seed_ranges("1..100,50..150", 10).is_err());
        assert!(parse_seed_ranges("1..100,1..100", 10).is_err());
        // Exactly abutting ranges do not overlap.
        parse_seed_ranges("1..101,101..201", 10).unwrap();
        // Same seeds, but far enough apart for the game count: fine.
        parse_seed_ranges("1,10000", 200).unwrap();
    }

    #[test]
    fn rejects_empty_and_backwards_seed_ranges() {
        assert!(parse_seed_ranges("", 10).is_err());
        assert!(parse_seed_ranges("1,,2", 10).is_err());
        assert!(parse_seed_ranges("100..50", 10).is_err());
        assert!(parse_seed_ranges("50..50", 10).is_err());
        assert!(parse_seed_ranges("abc", 10).is_err());
    }

    // --- cell enumeration ------------------------------------------------

    #[test]
    fn cells_are_the_full_seed_range_by_budget_cross_product_budget_major() {
        let ranges = parse_seed_ranges("1,10000", 100).unwrap();
        let budgets = parse_budgets("nodes:2000,time_ms:50").unwrap();
        let cells = plan_cells(&ranges, &budgets);
        assert_eq!(cells.len(), 4);
        let labels: Vec<(&str, u64)> = cells
            .iter()
            .map(|c| (c.budget_label.as_str(), c.range.start))
            .collect();
        assert_eq!(
            labels,
            vec![
                ("nodes:2000", 1),
                ("nodes:2000", 10_000),
                ("time_ms:50", 1),
                ("time_ms:50", 10_000),
            ],
            "budget-major: every range at one budget before the next budget"
        );
        assert!(cells.iter().all(|c| c.range.games() == 100));
        // A single range at a single budget is one cell.
        assert_eq!(
            plan_cells(&ranges[..1], &budgets[..1]).len(),
            1,
            "the degenerate one-cell experiment is still a valid plan"
        );
    }

    #[test]
    fn one_cells_seeds_are_exactly_its_range() {
        let cell = CellPlan {
            budget_label: "nodes:1".to_string(),
            budget: Budget::Nodes(1),
            range: SeedRange { start: 7, pairs: 3 },
        };
        assert_eq!(cell.range.seeds(), vec![7, 8, 9]);
        assert_eq!(cell.range.games(), 6);
    }

    // --- pooling math ----------------------------------------------------

    #[test]
    fn pooling_sums_counts_within_a_budget_and_refits_rather_than_averaging() {
        let cells = vec![
            synthetic_cell(0, "nodes:2000", 1, 60, 40, 0),
            synthetic_cell(1, "nodes:2000", 10_000, 55, 45, 0),
            synthetic_cell(2, "time_ms:100", 1, 30, 70, 0),
        ];
        let pooled = pool_by_budget(&cells, &params());
        assert_eq!(pooled.len(), 2, "one row per budget, not per cell");

        let nodes = &pooled[0];
        assert_eq!(nodes.budget, "nodes:2000");
        assert_eq!(nodes.cells, 2);
        assert_eq!(
            (nodes.tally.a_wins, nodes.tally.b_wins, nodes.tally.draws),
            (115, 85, 0)
        );
        assert_eq!(nodes.games, 200);

        // The pooled Elo is a refit on the summed counts, which is *not* the
        // mean of the two cells' Elos - that is the whole reason pooling is a
        // tool rather than a spreadsheet column.
        let refit = fit_elo(115, 85, 0);
        assert_eq!(nodes.elo, refit);
        let mean_of_cells = (cells[0].elo.rating_diff + cells[1].elo.rating_diff) / 2.0;
        assert!(
            (nodes.elo.rating_diff - mean_of_cells).abs() > 1e-9,
            "a refit should differ from the average of the cells"
        );
        // ...and it is strictly better evidence: the pooled interval is
        // narrower than either cell's.
        let width = |e: &EloEstimate| e.diff_ci_high - e.diff_ci_low;
        assert!(width(&nodes.elo) < width(&cells[0].elo));
        assert!(width(&nodes.elo) < width(&cells[1].elo));

        // SPRT is likewise recomputed on the pooled counts.
        assert_eq!(nodes.sprt, sprt(115, 85, 0, &params()));

        // The other budget is kept entirely separate.
        assert_eq!(pooled[1].budget, "time_ms:100");
        assert_eq!(pooled[1].games, 100);
        assert_eq!(pooled[1].tally.a_wins, 30);
    }

    #[test]
    fn pooling_sums_victory_kinds_and_race_exposure_too() {
        let cells = vec![
            synthetic_cell(0, "nodes:1", 1, 6, 4, 0),
            synthetic_cell(1, "nodes:1", 100, 3, 5, 2),
        ];
        let pooled = pool_by_budget(&cells, &params());
        assert_eq!(pooled[0].victory_breakdown.a.civilian_victory, 9);
        assert_eq!(pooled[0].victory_breakdown.b.military_supremacy, 9);
        assert_eq!(pooled[0].race_exposure.military_games, 9);
        assert_eq!(pooled[0].race_exposure.total_games, 20);
        // Every game is accounted for: wins + losses + draws.
        assert_eq!(pooled[0].tally.total(), 20);
    }

    #[test]
    fn pooling_keeps_budgets_in_first_appearance_order() {
        let cells = vec![
            synthetic_cell(0, "time_ms:100", 1, 5, 5, 0),
            synthetic_cell(1, "nodes:2000", 1, 5, 5, 0),
            synthetic_cell(2, "time_ms:100", 100, 5, 5, 0),
        ];
        let pooled = pool_by_budget(&cells, &params());
        assert_eq!(
            pooled.iter().map(|p| p.budget.as_str()).collect::<Vec<_>>(),
            vec!["time_ms:100", "nodes:2000"]
        );
        assert_eq!(pooled[0].cells, 2);
    }

    #[test]
    fn pooling_nothing_is_no_rows_and_an_inconclusive_verdict() {
        let pooled = pool_by_budget(&[], &params());
        assert!(pooled.is_empty());
        assert_eq!(overall_verdict(&pooled), Verdict::Inconclusive);
    }

    // --- verdict ---------------------------------------------------------

    #[test]
    fn the_verdict_is_the_conjunction_over_budgets() {
        // A lopsided record at each budget resolves H1 at both.
        let strong = vec![
            synthetic_cell(0, "nodes:2000", 1, 400, 200, 0),
            synthetic_cell(1, "time_ms:100", 1, 400, 200, 0),
        ];
        assert_eq!(
            overall_verdict(&pool_by_budget(&strong, &params())),
            Verdict::Accept
        );

        // Strong at one budget, clearly worse at the other: rejected, not
        // averaged into an accept.
        let mixed = vec![
            synthetic_cell(0, "nodes:2000", 1, 400, 200, 0),
            synthetic_cell(1, "time_ms:100", 1, 200, 400, 0),
        ];
        assert_eq!(
            overall_verdict(&pool_by_budget(&mixed, &params())),
            Verdict::Reject
        );

        // Strong at one budget, not yet resolved at the other.
        let partial = vec![
            synthetic_cell(0, "nodes:2000", 1, 400, 200, 0),
            synthetic_cell(1, "time_ms:100", 1, 5, 5, 0),
        ];
        assert_eq!(
            overall_verdict(&pool_by_budget(&partial, &params())),
            Verdict::Inconclusive
        );
    }

    // --- paths -----------------------------------------------------------

    #[test]
    fn slugs_are_safe_for_a_filename() {
        assert_eq!(slug("mcts-eval"), "mcts-eval");
        assert_eq!(slug("mcts-eval:leaf=blend:0.7"), "mcts-eval_leaf_blend_0.7");
        assert_eq!(slug("nodes:2000"), "nodes_2000");
        assert_eq!(slug("a/../b"), "a_.._b");
        assert_eq!(slug("///"), "unnamed");
        assert!(!slug("mcts-eval:c=0.5,leaf=static").contains(['/', ':', '=', ',']));
    }

    #[test]
    fn default_labels_and_cell_filenames_describe_the_run() {
        assert_eq!(
            default_label("mcts-eval:c=0.4", "mcts-eval"),
            "mcts-eval_c_0.4-vs-mcts-eval"
        );
        let cell = CellPlan {
            budget_label: "time_ms:100".to_string(),
            budget: Budget::TimeMs(100),
            range: SeedRange {
                start: 10_000,
                pairs: 50,
            },
        };
        assert_eq!(
            cell_results_filename(&cell),
            "cell-time_ms_100-seed10000-n100.json"
        );
        // Distinct cells never collide on a filename.
        let ranges = parse_seed_ranges("1,10000", 100).unwrap();
        let budgets = parse_budgets("nodes:2000,time_ms:50").unwrap();
        let names: std::collections::BTreeSet<String> = plan_cells(&ranges, &budgets)
            .iter()
            .map(cell_results_filename)
            .collect();
        assert_eq!(names.len(), 4);
    }

    // --- chunking identity, and a real end-to-end run --------------------

    /// The load-bearing claim behind early stopping: playing a cell's seeds in
    /// chunks plays exactly the games one call would have. Compared field by
    /// field over everything deterministic — `wall_time_ms` is excluded
    /// because it is a measurement of the machine, not of the game.
    #[test]
    fn chunked_play_is_identical_to_one_call() {
        let cell = CellPlan {
            budget_label: "nodes:1".to_string(),
            budget: Budget::Nodes(1),
            range: SeedRange { start: 1, pairs: 7 },
        };
        let one = play_cell("random", "greedy", &cell, &params(), false, None).unwrap();
        let chunked = play_cell("random", "greedy", &cell, &params(), false, Some(4)).unwrap();
        assert_eq!(one.records.len(), 14);
        assert_eq!(chunked.records.len(), one.records.len());
        assert!(!one.stopped_early && !chunked.stopped_early);
        for (a, b) in one.records.iter().zip(chunked.records.iter()) {
            assert_eq!(a.seed, b.seed);
            assert_eq!(a.agent_a_seat, b.agent_a_seat);
            assert_eq!(a.seat_one.name, b.seat_one.name);
            assert_eq!(a.seat_two.name, b.seat_two.name);
            assert_eq!(a.result, b.result);
            assert_eq!(a.moves, b.moves);
            assert_eq!(a.military_race_exposed, b.military_race_exposed);
            assert_eq!(a.science_race_exposed, b.science_race_exposed);
        }
        assert_eq!(tally(&one.records), tally(&chunked.records));
    }

    /// Early stopping cuts a cell short once its SPRT decides, and reports
    /// that it did. `random` vs `greedy` at a wide H1 resolves fast.
    #[test]
    fn early_stopping_abandons_the_rest_of_a_decided_cell() {
        let cell = CellPlan {
            budget_label: "nodes:1".to_string(),
            budget: Budget::Nodes(1),
            range: SeedRange {
                start: 1,
                pairs: 200,
            },
        };
        let params = SprtParams {
            elo0: 0.0,
            elo1: 200.0,
            alpha: 0.05,
            beta: 0.05,
        };
        let outcome = play_cell("random", "greedy", &cell, &params, true, Some(20)).unwrap();
        assert!(outcome.stopped_early, "a decided cell should stop");
        assert!(
            outcome.records.len() < 400,
            "stopping early should mean fewer games than planned, got {}",
            outcome.records.len()
        );
        let t = tally(&outcome.records);
        assert_ne!(
            sprt(t.a_wins, t.b_wins, t.draws, &params).decision,
            SprtDecision::Continue,
            "it should only have stopped on a decision"
        );
        // With early stopping off, the same cell plays every game.
        let full = play_cell("random", "greedy", &cell, &params, false, Some(20)).unwrap();
        assert_eq!(full.records.len(), 400);
        assert!(!full.stopped_early);
    }

    /// A whole small experiment end to end: two disjoint seed ranges at one
    /// cheap budget, self-play so the expected Elo is ~0, written to a temp
    /// directory. Checks the artifacts, the cross-references between them, and
    /// that the pooled counts are exactly the cells' counts summed.
    #[test]
    fn a_small_experiment_runs_end_to_end_and_writes_its_artifacts() {
        let dir = std::env::temp_dir().join(format!(
            "duels-arena-experiment-{}-{}",
            std::process::id(),
            "end_to_end"
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let ranges = parse_seed_ranges("1,500", 4).unwrap();
        let budgets = parse_budgets("nodes:1").unwrap();
        let plan = ExperimentPlan {
            candidate: "random".to_string(),
            control: "random".to_string(),
            cells: plan_cells(&ranges, &budgets),
            sprt: params(),
            label: default_label("random", "random"),
            early_stop: false,
            check_every_games: None,
        };
        assert_eq!(plan.cells.len(), 2);

        let mut lines: Vec<String> = Vec::new();
        let summary = run(&plan, &dir, "2026-09-09T00:00:00Z", &mut |line| {
            lines.push(line.to_string())
        })
        .unwrap();

        assert_eq!(summary.schema, SCHEMA_VERSION);
        assert_eq!(summary.label, "random-vs-random");
        assert_eq!(summary.cells.len(), 2);
        assert_eq!(summary.pooled.len(), 1);
        assert_eq!(summary.total_games, 8);
        assert_eq!(summary.budgets, vec!["nodes:1".to_string()]);
        assert_eq!(summary.seed_ranges.len(), 2);
        // The recorded params are the agent's own reported ones, read off a
        // game that was actually played rather than re-derived from the spec
        // string (`random`'s happen to be empty, which is exactly why this
        // asserts equality and not non-emptiness).
        let own = crate::agent_spec::make_agent_from_spec("random", 1)
            .unwrap()
            .spec()
            .params;
        assert_eq!(summary.candidate_spec_params, own);
        assert_eq!(summary.control_spec_params, own);
        assert!(!lines.is_empty(), "progress should have been reported");

        // The pooled row is the cells summed, exactly.
        let pooled = &summary.pooled[0];
        assert_eq!(pooled.cells, 2);
        assert_eq!(
            pooled.tally.a_wins,
            summary.cells.iter().map(|c| c.tally.a_wins).sum::<u32>()
        );
        assert_eq!(pooled.games, 8);

        // Every cell wrote its raw records where the summary says it did, and
        // those records read back as the same games.
        for cell in &summary.cells {
            let path = dir.join(&cell.results_file);
            let back = crate::results_io::read_results(&path).unwrap();
            assert_eq!(back.records.len() as u32, cell.games_played);
            assert_eq!(back.tally, cell.tally);
            assert!(!cell.stopped_early);
            assert_eq!(cell.games_played, cell.games_planned);
        }

        let (json_path, md_path) = write_summary(&summary, &dir).unwrap();
        let back: ExperimentSummary =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(back.label, summary.label);
        assert_eq!(back.cells.len(), summary.cells.len());
        assert_eq!(back.verdict, summary.verdict);
        assert_eq!(back.total_games, summary.total_games);

        let md = std::fs::read_to_string(&md_path).unwrap();
        assert!(md.contains("# Experiment: random-vs-random"));
        assert!(md.contains("nodes:1"));
        assert!(md.contains("Verdict"));
        assert!(md.contains("2026-09-09T00:00:00Z"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_plan_is_an_error_not_an_empty_summary() {
        let plan = ExperimentPlan {
            candidate: "random".to_string(),
            control: "random".to_string(),
            cells: Vec::new(),
            sprt: params(),
            label: "empty".to_string(),
            early_stop: false,
            check_every_games: None,
        };
        assert!(run(
            &plan,
            std::env::temp_dir().as_path(),
            "2026-09-09T00:00:00Z",
            &mut |_| {}
        )
        .is_err());
    }

    #[test]
    fn markdown_renders_every_cell_and_pooled_row() {
        let cells = vec![
            synthetic_cell(0, "nodes:2000", 1, 60, 40, 0),
            synthetic_cell(1, "nodes:2000", 10_000, 55, 45, 0),
            synthetic_cell(2, "time_ms:100", 1, 30, 70, 0),
        ];
        let pooled = pool_by_budget(&cells, &params());
        let summary = ExperimentSummary {
            schema: SCHEMA_VERSION,
            label: "cand-vs-ctrl".to_string(),
            generated_at: "2026-09-09T00:00:00Z".to_string(),
            candidate: "cand".to_string(),
            control: "ctrl".to_string(),
            candidate_spec_params: "p=1".to_string(),
            control_spec_params: "p=0".to_string(),
            budgets: vec!["nodes:2000".to_string(), "time_ms:100".to_string()],
            seed_ranges: vec![
                SeedRange {
                    start: 1,
                    pairs: 50,
                },
                SeedRange {
                    start: 10_000,
                    pairs: 50,
                },
            ],
            sprt_params: params(),
            early_stop: false,
            verdict: overall_verdict(&pooled),
            total_games: cells.iter().map(|c| c.games_played).sum(),
            cells,
            pooled,
        };
        let md = render_markdown(&summary);
        assert!(md.contains("`nodes:2000`"));
        assert!(md.contains("`time_ms:100`"));
        assert!(md.contains("10000"));
        // One row per cell in the per-cell table.
        assert_eq!(md.matches("\n| 1 | `nodes:2000`").count(), 1);
        assert_eq!(md.matches("\n| 3 | `time_ms:100`").count(), 1);
        assert!(md.contains("Verdict"));
        assert!(md.contains("quiet"));
    }
}
