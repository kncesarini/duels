//! `duels-arena` CLI.
//!
//! ```text
//! duels-arena match --agent-a phased --agent-b mcts-uct --games 1000 \
//!     --budget nodes:2000 --seed 1 [--out arena/results/run.json] \
//!     [--sprt-elo0 0] [--sprt-elo1 5] [--alpha 0.05] [--beta 0.05]
//!
//! duels-arena experiment --candidate <SPEC> --control <SPEC> \
//!     --seeds 1,10000 --budgets nodes:2000,time_ms:100 --games 400 \
//!     [--label <NAME>] [--out-dir arena/results/experiments] \
//!     [--sprt-elo0 0] [--sprt-elo1 20] [--alpha 0.05] [--beta 0.05] \
//!     [--early-stop] [--check-every 50] [--dry-run]
//!
//! duels-arena pairings [--format matrix|lines]
//! duels-arena leaderboard --results-dir <DIR> \
//!     [--out-json arena/leaderboard.json] [--out-md arena/leaderboard.md] \
//!     [--commit <SHA>] [--generated-at <RFC3339>]
//! duels-arena champion [--field agent|budget|spec]
//! ```
//!
//! `experiment` is `match` run as this project's whole documented measurement
//! protocol instead of one match at a time: a candidate against a control
//! over every (seed range × budget) cell, each cell an ordinary paired-seed
//! match, with Elo + SPRT per cell *and* pooled across seed ranges within
//! each budget, and one machine-readable summary
//! (`<out-dir>/<label>/summary.json`) as the verdict. See
//! `duels_arena::experiment`.
//!
//! `pairings`, `leaderboard` and `champion` exist for the nightly round-robin
//! workflow (`.github/workflows/nightly-arena.yml`) and the `ai-candidate`
//! check (`.github/workflows/ai-candidate.yml`): `pairings --format matrix`
//! emits the GitHub Actions job matrix, `leaderboard` reduces the matrix's
//! per-pairing results artifacts to `arena/leaderboard.{json,md}`, and
//! `champion` prints the designated champion so a workflow never has to
//! hard-code it. See `duels_arena::leaderboard`.
//!
//! `--games N` is the total number of individual games to play; internally
//! this is `ceil(N/2)` paired seeds (see `match_runner::play_paired_match`),
//! so an odd `N` is rounded up to the next even number. Prints win/loss/draw
//! counts, a breakdown of *how* each side's wins were achieved (military /
//! scientific / civilian / tiebreak, see `match_runner::VictoryBreakdown`), a
//! "race exposure" count (how many games saw either player come within one
//! step of an instant win, regardless of who won — see
//! `match_runner::RaceExposure`), a logistic-Elo estimate (see `elo`), and an
//! SPRT verdict (see `sprt`) for agent A vs agent B. Writes every game's
//! record plus that same derived summary as JSON to `--out` (default:
//! `arena/results/<a>-vs-<b>-seed<seed>-n<games>.json`, see `results_io`).
//!
//! `--agent-a`/`--agent-b` accept a bare agent name (looked up in
//! `agent_registry`) or a `name:key=value,...` specification string that
//! builds one specific agent's own `Config`/`Weights` type (see
//! `agent_spec`), e.g. `--agent-a mcts-uct:exploration=1.2`.
//!
//! # A note on `time_ms` budgets
//!
//! See the crate-level docs (`lib.rs`) for why a `--budget time_ms:<n>` run
//! should be measured on an otherwise-quiet machine, one match at a time.

use std::path::PathBuf;
use std::process::ExitCode;

use duels_arena::agent_registry::KNOWN_AGENTS;
use duels_arena::elo::fit_elo;
use duels_arena::experiment;
use duels_arena::leaderboard;
use duels_arena::match_runner::{
    parse_budget, play_paired_match, race_exposure, tally, victory_breakdown, VictoryBreakdown,
};
use duels_arena::results_io::write_results;
use duels_arena::sprt::{sprt, SprtParams};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("match") => run_match(&args[1..]),
        Some("experiment") => run_experiment(&args[1..]),
        Some("pairings") => run_pairings(&args[1..]),
        Some("leaderboard") => run_leaderboard(&args[1..]),
        Some("champion") => run_champion(&args[1..]),
        Some("help") | Some("--help") | Some("-h") | None => {
            print_usage();
            Ok(())
        }
        Some(other) => Err(format!(
            "unknown subcommand \"{other}\" (try \"duels-arena match --help\")"
        )),
    }
}

fn print_usage() {
    println!(
        "duels-arena: tournament runner and statistical comparison for Agent implementations\n\n\
         USAGE:\n    duels-arena match --agent-a <SPEC> --agent-b <SPEC> --games <N> \\\n        \
         --budget <nodes:N|time_ms:N> --seed <N> [--out <PATH>]\n        \
         [--sprt-elo0 <F>] [--sprt-elo1 <F>] [--alpha <F>] [--beta <F>]\n\n    \
         duels-arena experiment --candidate <SPEC> --control <SPEC>\n        \
         [--seeds <RANGES>] [--budgets <B[,B...]>] [--games <N per cell>]\n        \
         [--label <NAME>] [--out-dir <DIR>] [--sprt-elo0 <F>] [--sprt-elo1 <F>]\n        \
         [--alpha <F>] [--beta <F>] [--early-stop] [--check-every <N>] [--dry-run]\n        \
         This project's whole measurement protocol in one command: every\n        \
         (seed range x budget) cell played as a paired-seed match, Elo and\n        \
         SPRT per cell and pooled across seed ranges within each budget, and\n        \
         one machine-readable verdict at <out-dir>/<label>/summary.json.\n        \
         --seeds takes base seeds (\"1,10000\", each covering --games worth of\n        \
         seeds) and/or explicit half-open ranges (\"1..201\", which name their\n        \
         own length); ranges must be disjoint. --games is per cell, so the\n        \
         total is ranges x budgets x games. --dry-run prints the cell plan\n        \
         and its cost without playing anything.\n\n    \
         duels-arena pairings [--format matrix|lines]\n        \
         Every round-robin pairing on the leaderboard ladder. \"matrix\" emits\n        \
         the GitHub Actions job matrix the nightly workflow fans out over.\n\n    \
         duels-arena leaderboard --results-dir <DIR> [--out-json <PATH>]\n        \
         [--out-md <PATH>] [--commit <SHA>] [--generated-at <RFC3339>]\n        \
         Fit a joint Elo table over a directory of per-pairing results files\n        \
         and write arena/leaderboard.json and arena/leaderboard.md.\n\n    \
         duels-arena champion [--field agent|budget|spec]\n        \
         The designated reigning champion an ai-candidate run measures\n        \
         against.\n\n\
         Known agents: {}\n\n\
         <SPEC> is a bare agent name, or \"name:key=value,...\" naming one\n\
         agent's own Config/Weights explicitly (e.g. \"mcts-uct:exploration=1.2\"\n\
         or \"alphabeta:max_depth=10\") -- see duels_arena::agent_spec.\n\n\
         --games N is the number of individual games; internally this is\n\
         ceil(N/2) paired seeds (agent A and agent B each play both seats\n\
         for every seed), so an odd N is rounded up.\n\n\
         A time_ms budget is wall-clock based and therefore sensitive to\n\
         load from other processes on the same machine -- see the\n\
         \"quiet machine\" note in duels_arena's crate docs before trusting a\n\
         small-sample time_ms comparison.\n",
        KNOWN_AGENTS.join(", ")
    );
}

/// Minimal hand-rolled `--flag value` parser: enough for this crate's one
/// subcommand without pulling in a CLI-parsing dependency.
struct Flags {
    values: std::collections::HashMap<String, String>,
    switches: std::collections::BTreeSet<String>,
}

impl Flags {
    fn parse(args: &[String]) -> Result<Self, String> {
        Self::parse_with_switches(args, &[])
    }

    /// Like [`Flags::parse`], but the names in `switches` are valueless
    /// boolean flags (`--early-stop`) rather than `--flag value` pairs.
    fn parse_with_switches(args: &[String], switches: &[&str]) -> Result<Self, String> {
        let mut values = std::collections::HashMap::new();
        let mut seen_switches = std::collections::BTreeSet::new();
        let mut i = 0;
        while i < args.len() {
            let flag = &args[i];
            let name = flag
                .strip_prefix("--")
                .ok_or_else(|| format!("expected a --flag, got \"{flag}\""))?;
            if switches.contains(&name) {
                seen_switches.insert(name.to_string());
                i += 1;
                continue;
            }
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("--{name} needs a value"))?;
            values.insert(name.to_string(), value.clone());
            i += 2;
        }
        Ok(Self {
            values,
            switches: seen_switches,
        })
    }

    /// Whether a valueless boolean flag was given.
    fn switch(&self, name: &str) -> bool {
        self.switches.contains(name)
    }

    fn required(&self, name: &str) -> Result<&str, String> {
        self.values
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("missing required --{name}"))
    }

    fn optional(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    fn parsed<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.values.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("invalid value for --{name}: \"{v}\"")),
        }
    }
}

/// Render one side's [`VictoryBreakdown`] as `"N wins (military M, scientific
/// M, civilian M, tiebreak M)"`, for the human-readable CLI summary.
fn describe_victory_breakdown(vb: &VictoryBreakdown) -> String {
    format!(
        "{} wins (military {}, scientific {}, civilian {}, tiebreak {})",
        vb.total(),
        vb.military_supremacy,
        vb.scientific_supremacy,
        vb.civilian_victory,
        vb.civilian_tiebreak,
    )
}

fn run_match(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args)?;

    let agent_a = flags.required("agent-a")?.to_string();
    let agent_b = flags.required("agent-b")?.to_string();
    let games: u32 = flags.parsed("games", 100)?;
    let budget = parse_budget(flags.optional("budget").unwrap_or("nodes:1000"))?;
    let seed: u64 = flags.parsed("seed", 1)?;

    let sprt_params = SprtParams {
        elo0: flags.parsed("sprt-elo0", SprtParams::default().elo0)?,
        elo1: flags.parsed("sprt-elo1", SprtParams::default().elo1)?,
        alpha: flags.parsed("alpha", SprtParams::default().alpha)?,
        beta: flags.parsed("beta", SprtParams::default().beta)?,
    };

    let num_pairs = games.div_ceil(2).max(1);
    let seeds: Vec<u64> = (0..num_pairs as u64).map(|i| seed + i).collect();

    let out_path: PathBuf = match flags.optional("out") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(format!(
            "arena/results/{agent_a}-vs-{agent_b}-seed{seed}-n{}.json",
            num_pairs * 2
        )),
    };

    println!(
        "duels-arena match: {agent_a} vs {agent_b}  ({} games = {num_pairs} paired seeds, budget {budget:?}, base seed {seed})",
        num_pairs * 2
    );

    let records = play_paired_match(&agent_a, &agent_b, &seeds, budget)?;
    let t = tally(&records);

    let total_moves: u64 = records.iter().map(|r| r.moves as u64).sum();
    let total_wall_ms: u64 = records.iter().map(|r| r.wall_time_ms).sum();

    println!(
        "results: {agent_a} {} wins, {agent_b} {} wins, {} draws  (out of {})",
        t.a_wins,
        t.b_wins,
        t.draws,
        t.total()
    );
    println!(
        "avg moves/game: {:.1}   avg wall time/game: {:.1} ms   total wall time: {} ms",
        total_moves as f64 / t.total() as f64,
        total_wall_ms as f64 / t.total() as f64,
        total_wall_ms
    );

    let vb = victory_breakdown(&records);
    println!(
        "victory kinds: {agent_a} {}   {agent_b} {}",
        describe_victory_breakdown(&vb.a),
        describe_victory_breakdown(&vb.b),
    );

    let re = race_exposure(&records);
    println!(
        "race exposure: military (pawn >= distance 6) in {}/{} games ({:.0}%)   \
         scientific (>= 5 distinct symbols) in {}/{} games ({:.0}%)",
        re.military_games,
        re.total_games,
        100.0 * re.military_games as f64 / re.total_games as f64,
        re.science_games,
        re.total_games,
        100.0 * re.science_games as f64 / re.total_games as f64,
    );

    let elo_estimate = fit_elo(t.a_wins, t.b_wins, t.draws);
    println!(
        "elo: {agent_a} = {:+.1} (anchor: {agent_b} = {:.1}), 95% CI [{:+.1}, {:+.1}]",
        elo_estimate.rating_diff,
        elo_estimate.anchor_elo,
        elo_estimate.diff_ci_low,
        elo_estimate.diff_ci_high
    );

    let sprt_result = sprt(t.a_wins, t.b_wins, t.draws, &sprt_params);
    println!(
        "sprt: H0 elo={:.1} vs H1 elo={:.1} (alpha={}, beta={}) -> llr={:.3} bounds=[{:.3}, {:.3}] -> {:?}",
        sprt_params.elo0,
        sprt_params.elo1,
        sprt_params.alpha,
        sprt_params.beta,
        sprt_result.llr,
        sprt_result.lower_bound,
        sprt_result.upper_bound,
        sprt_result.decision
    );

    write_results(&out_path, &records).map_err(|e| format!("failed to write {out_path:?}: {e}"))?;
    println!(
        "wrote {} game records to {}",
        records.len(),
        out_path.display()
    );

    Ok(())
}

/// Build an [`experiment::ExperimentPlan`] from parsed flags. Split out from
/// [`run_experiment`] so the plan a set of arguments produces can be tested
/// without playing any games.
fn experiment_plan(flags: &Flags) -> Result<experiment::ExperimentPlan, String> {
    let candidate = flags.required("candidate")?.to_string();
    let control = flags.required("control")?.to_string();
    let games: u32 = flags.parsed("games", 200)?;
    let ranges = experiment::parse_seed_ranges(flags.optional("seeds").unwrap_or("1"), games)?;
    let budgets = experiment::parse_budgets(flags.optional("budgets").unwrap_or("nodes:2000"))?;

    let sprt = SprtParams {
        elo0: flags.parsed("sprt-elo0", experiment::DEFAULT_ELO0)?,
        elo1: flags.parsed("sprt-elo1", experiment::DEFAULT_ELO1)?,
        alpha: flags.parsed("alpha", SprtParams::default().alpha)?,
        beta: flags.parsed("beta", SprtParams::default().beta)?,
    };

    let early_stop = flags.switch("early-stop");
    let check_every: Option<u32> = match flags.optional("check-every") {
        Some(_) => Some(flags.parsed("check-every", experiment::DEFAULT_CHECK_EVERY_GAMES)?),
        None if early_stop => Some(experiment::DEFAULT_CHECK_EVERY_GAMES),
        None => None,
    };

    let label = match flags.optional("label") {
        Some(l) => experiment::slug(l),
        None => experiment::default_label(&candidate, &control),
    };

    Ok(experiment::ExperimentPlan {
        cells: experiment::plan_cells(&ranges, &budgets),
        candidate,
        control,
        sprt,
        label,
        early_stop,
        check_every_games: check_every,
    })
}

/// `duels-arena experiment` — the measurement protocol as one command. See
/// `duels_arena::experiment` for what it computes and why budgets are pooled
/// separately from each other.
fn run_experiment(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse_with_switches(args, &["early-stop", "dry-run"])?;
    let plan = experiment_plan(&flags)?;
    let out_dir = PathBuf::from(
        flags
            .optional("out-dir")
            .unwrap_or(experiment::DEFAULT_OUT_DIR),
    )
    .join(&plan.label);

    let total_games: u32 = plan.cells.iter().map(|c| c.range.games()).sum();
    println!(
        "duels-arena experiment \"{}\": {} vs {} (control)",
        plan.label, plan.candidate, plan.control
    );
    println!(
        "plan: {} cells, {} games total{}",
        plan.cells.len(),
        total_games,
        if plan.early_stop {
            " at most (early stopping on)"
        } else {
            ""
        }
    );
    for (i, cell) in plan.cells.iter().enumerate() {
        println!(
            "  cell {}: budget {} seeds {}..{} ({} games)",
            i + 1,
            cell.budget_label,
            cell.range.start,
            cell.range.end(),
            cell.range.games()
        );
    }
    println!(
        "sprt: H0 elo={:.1} vs H1 elo={:.1} (alpha={}, beta={})",
        plan.sprt.elo0, plan.sprt.elo1, plan.sprt.alpha, plan.sprt.beta
    );
    if plan
        .cells
        .iter()
        .any(|c| matches!(c.budget, duels_agents_api::Budget::TimeMs(_)))
    {
        println!(
            "note: a time_ms cell is wall-clock based. Run this on an otherwise-quiet machine \
             (see duels_arena's crate docs); cells are played one at a time for exactly this \
             reason."
        );
    }
    println!("output: {}", out_dir.display());

    if flags.switch("dry-run") {
        println!("--dry-run: not playing anything");
        return Ok(());
    }

    let generated_at = leaderboard::format_rfc3339_utc(now_unix_seconds());
    let summary = experiment::run(&plan, &out_dir, &generated_at, &mut |line| {
        println!("{line}");
    })?;

    println!("\n{}", experiment::render_markdown(&summary));
    let (json_path, md_path) = experiment::write_summary(&summary, &out_dir)?;
    println!(
        "wrote {} and {} (plus {} per-cell results files)",
        json_path.display(),
        md_path.display(),
        summary.cells.len()
    );
    println!("verdict: {:?}", summary.verdict);
    Ok(())
}

/// `duels-arena pairings` — the round robin's schedule.
///
/// `--format matrix` (the default) prints a single line of JSON shaped for a
/// GitHub Actions `strategy.matrix`: an array of objects carrying both agent
/// names, the results filename to write, and a short id usable in an artifact
/// name. `--format lines` prints `a b` per line for shell consumption.
fn run_pairings(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args)?;
    let format = flags.optional("format").unwrap_or("matrix");
    let pairings = leaderboard::pairings();
    match format {
        "matrix" => {
            let entries: Vec<serde_json::Value> = pairings
                .iter()
                .map(|&(a, b)| {
                    serde_json::json!({
                        "agent_a": a,
                        "agent_b": b,
                        "id": format!("{a}--vs--{b}"),
                        "file": leaderboard::pairing_results_filename(a, b),
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string(&entries)
                    .map_err(|e| format!("failed to serialize the pairing matrix: {e}"))?
            );
        }
        "lines" => {
            for (a, b) in pairings {
                println!("{a} {b}");
            }
        }
        other => {
            return Err(format!(
                "invalid --format \"{other}\": expected \"matrix\" or \"lines\""
            ))
        }
    }
    Ok(())
}

/// `duels-arena champion` — the designated champion, for a workflow to read
/// rather than hard-code. `--field spec` (the default) prints
/// `"<agent> at <budget>"`; `agent` and `budget` print one piece each.
fn run_champion(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args)?;
    match flags.optional("field").unwrap_or("spec") {
        "agent" => println!("{}", leaderboard::CHAMPION.agent),
        "budget" => println!("{}", leaderboard::CHAMPION.budget),
        "spec" => println!(
            "{} at {}",
            leaderboard::CHAMPION.agent,
            leaderboard::CHAMPION.budget
        ),
        other => {
            return Err(format!(
                "invalid --field \"{other}\": expected \"agent\", \"budget\" or \"spec\""
            ))
        }
    }
    Ok(())
}

/// `duels-arena leaderboard` — reduce a directory of per-pairing results
/// files to the joint Elo table, and write both the JSON and the Markdown.
fn run_leaderboard(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args)?;
    let results_dir = PathBuf::from(flags.required("results-dir")?);
    let out_json = PathBuf::from(
        flags
            .optional("out-json")
            .unwrap_or("arena/leaderboard.json"),
    );
    let out_md = PathBuf::from(flags.optional("out-md").unwrap_or("arena/leaderboard.md"));
    let commit = flags.optional("commit").unwrap_or("unknown").to_string();
    let generated_at = match flags.optional("generated-at") {
        Some(t) => t.to_string(),
        None => leaderboard::format_rfc3339_utc(now_unix_seconds()),
    };

    let records = leaderboard::collect_pairwise_records(&results_dir)?;
    println!(
        "read {} pairing results from {}",
        records.len(),
        results_dir.display()
    );

    let board = leaderboard::build(&records, &generated_at, &commit)?;
    leaderboard::write(&board, &out_json, &out_md)?;

    println!("{}", leaderboard::render_markdown(&board));
    println!("wrote {} and {}", out_json.display(), out_md.display());
    Ok(())
}

/// Seconds since the Unix epoch, for stamping a leaderboard generated without
/// an explicit `--generated-at`.
///
/// `clippy.toml` bans wall-clock reads across the workspace so the rules
/// engine and the agents stay deterministic; `duels-arena` is one of the
/// crates explicitly carved out for reporting (see `match_runner`, which does
/// the same for `Instant::now`). Nothing about a match's *outcome* depends on
/// this — it only labels the report.
fn now_unix_seconds() -> i64 {
    #[allow(clippy::disallowed_methods)]
    let now = std::time::SystemTime::now();
    now.duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairings_matrix_is_valid_json_covering_every_pairing() {
        // The command prints to stdout, so rebuild the same value it does and
        // assert on that (the printing itself is one `println!`).
        let entries: Vec<serde_json::Value> = leaderboard::pairings()
            .iter()
            .map(|&(a, b)| {
                serde_json::json!({
                    "agent_a": a,
                    "agent_b": b,
                    "id": format!("{a}--vs--{b}"),
                    "file": leaderboard::pairing_results_filename(a, b),
                })
            })
            .collect();
        assert_eq!(
            entries.len(),
            leaderboard::LADDER.len() * (leaderboard::LADDER.len() - 1) / 2
        );
        let json = serde_json::to_string(&entries).unwrap();
        let back: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), entries.len());
        for e in &back {
            let a = e["agent_a"].as_str().unwrap();
            let b = e["agent_b"].as_str().unwrap();
            assert_ne!(a, b);
            assert_eq!(e["id"].as_str().unwrap(), format!("{a}--vs--{b}"));
            assert!(e["file"].as_str().unwrap().ends_with(".json"));
        }
    }

    #[test]
    fn pairings_and_champion_reject_an_unknown_format_or_field() {
        let args = |k: &str, v: &str| vec![format!("--{k}"), v.to_string()];
        assert!(run_pairings(&args("format", "yaml")).is_err());
        assert!(run_champion(&args("field", "nickname")).is_err());
        // The valid forms do not error.
        run_pairings(&args("format", "lines")).unwrap();
        run_champion(&args("field", "agent")).unwrap();
    }

    #[test]
    fn leaderboard_requires_a_results_dir_and_reports_a_missing_one() {
        assert!(run_leaderboard(&[]).is_err());
        let err = run_leaderboard(&[
            "--results-dir".to_string(),
            "/definitely/not/a/real/path/duels".to_string(),
        ])
        .unwrap_err();
        assert!(err.contains("failed to list"), "unexpected: {err}");
    }

    #[test]
    fn the_wall_clock_stamp_is_a_plausible_recent_timestamp() {
        let now = now_unix_seconds();
        // Any run of this test happens after the commit that introduced it.
        assert!(now > 1_750_000_000, "{now} looks wrong");
        let stamp = leaderboard::format_rfc3339_utc(now);
        assert!(stamp.ends_with('Z') && stamp.len() == 20, "{stamp}");
    }

    #[test]
    fn flags_parse_required_and_optional_and_typed_values() {
        let args: Vec<String> = [
            "--agent-a",
            "phased",
            "--agent-b",
            "phased",
            "--games",
            "10",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let flags = Flags::parse(&args).unwrap();
        assert_eq!(flags.required("agent-a").unwrap(), "phased");
        assert_eq!(flags.parsed::<u32>("games", 0).unwrap(), 10);
        assert_eq!(flags.parsed::<u32>("missing", 42).unwrap(), 42);
        assert!(flags.required("nope").is_err());
    }

    #[test]
    fn flags_reject_a_dangling_flag_without_a_value() {
        let args: Vec<String> = ["--agent-a"].into_iter().map(String::from).collect();
        assert!(Flags::parse(&args).is_err());
    }

    #[test]
    fn switches_are_valueless_and_do_not_swallow_the_next_flag() {
        let args: Vec<String> = ["--early-stop", "--games", "10"]
            .into_iter()
            .map(String::from)
            .collect();
        let flags = Flags::parse_with_switches(&args, &["early-stop"]).unwrap();
        assert!(flags.switch("early-stop"));
        assert!(!flags.switch("dry-run"));
        assert_eq!(flags.parsed::<u32>("games", 0).unwrap(), 10);
        // Without being declared a switch, it still wants a value.
        assert!(Flags::parse(&args).is_err());
    }

    fn experiment_args(pairs: &[&str]) -> Vec<String> {
        pairs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn experiment_defaults_to_one_seed_range_at_one_budget() {
        let plan = experiment_plan(
            &Flags::parse_with_switches(
                &experiment_args(&["--candidate", "mcts-eval", "--control", "mcts-uct"]),
                &["early-stop", "dry-run"],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(plan.cells.len(), 1);
        assert_eq!(plan.cells[0].budget_label, "nodes:2000");
        assert_eq!(plan.cells[0].range.start, 1);
        assert_eq!(plan.cells[0].range.games(), 200);
        assert_eq!(plan.label, "mcts-eval-vs-mcts-uct");
        // The project's own convention, not `SprtParams::default`'s elo1 = 5.
        assert_eq!(plan.sprt.elo0, 0.0);
        assert_eq!(plan.sprt.elo1, 20.0);
        assert!(!plan.early_stop);
        assert_eq!(plan.check_every_games, None);
    }

    #[test]
    fn experiment_enumerates_the_seed_range_by_budget_cross_product() {
        let plan = experiment_plan(
            &Flags::parse_with_switches(
                &experiment_args(&[
                    "--candidate",
                    "mcts-eval:c=0.4",
                    "--control",
                    "mcts-eval",
                    "--seeds",
                    "1,10000,20000..20050",
                    "--budgets",
                    "nodes:2000,time_ms:100",
                    "--games",
                    "100",
                    "--label",
                    "c 0.4 sweep",
                ]),
                &["early-stop", "dry-run"],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(plan.cells.len(), 6, "3 seed ranges x 2 budgets");
        assert_eq!(plan.label, "c_0.4_sweep", "a label is slugged for a path");
        // Budget-major, and the explicit range keeps its own length.
        let shape: Vec<(&str, u64, u32)> = plan
            .cells
            .iter()
            .map(|c| (c.budget_label.as_str(), c.range.start, c.range.games()))
            .collect();
        assert_eq!(
            shape,
            vec![
                ("nodes:2000", 1, 100),
                ("nodes:2000", 10_000, 100),
                ("nodes:2000", 20_000, 100),
                ("time_ms:100", 1, 100),
                ("time_ms:100", 10_000, 100),
                ("time_ms:100", 20_000, 100),
            ]
        );
    }

    #[test]
    fn experiment_turns_on_chunked_sprt_checks_only_when_asked() {
        let plan_of = |extra: &[&str]| {
            let mut args = experiment_args(&["--candidate", "phased", "--control", "mcts-uct"]);
            args.extend(extra.iter().map(|s| s.to_string()));
            experiment_plan(&Flags::parse_with_switches(&args, &["early-stop", "dry-run"]).unwrap())
                .unwrap()
        };
        assert_eq!(plan_of(&[]).check_every_games, None);
        let early = plan_of(&["--early-stop"]);
        assert!(early.early_stop);
        assert_eq!(
            early.check_every_games,
            Some(experiment::DEFAULT_CHECK_EVERY_GAMES)
        );
        let explicit = plan_of(&["--early-stop", "--check-every", "20"]);
        assert_eq!(explicit.check_every_games, Some(20));
    }

    #[test]
    fn experiment_rejects_missing_specs_and_overlapping_seed_ranges() {
        let plan_err = |args: Vec<String>| {
            experiment_plan(&Flags::parse_with_switches(&args, &["early-stop", "dry-run"]).unwrap())
                .unwrap_err()
        };
        assert!(plan_err(experiment_args(&["--control", "mcts-uct"])).contains("candidate"));
        assert!(plan_err(experiment_args(&["--candidate", "phased"])).contains("control"));
        let err = plan_err(experiment_args(&[
            "--candidate",
            "phased",
            "--control",
            "mcts-uct",
            "--seeds",
            "1,50",
            "--games",
            "200",
        ]));
        assert!(err.contains("overlap"), "unexpected: {err}");
    }

    /// `--dry-run` must not play a single game, so it is safe to ask a big
    /// experiment what it would cost.
    #[test]
    fn experiment_dry_run_plays_nothing() {
        run_experiment(&experiment_args(&[
            "--candidate",
            "mcts-eval",
            "--control",
            "mcts-uct",
            "--seeds",
            "1,10000",
            "--budgets",
            "nodes:2000,time_ms:100",
            "--games",
            "4000",
            "--dry-run",
        ]))
        .unwrap();
    }
}
