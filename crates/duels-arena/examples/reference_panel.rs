//! **"G-lite"**: a one-command driver for `docs/roadmap.md`'s Tier 1-G frozen
//! reference panel, built on top of `panel_battle.rs`'s existing
//! `play_paired_match_at_budgets` machinery rather than a new harness (this
//! project's convention: commit every measurement tool it builds, and prefer
//! composing an existing one over adding a parallel harness).
//!
//! # What the panel is, and why it is fixed
//!
//! Per `docs/roadmap.md`'s Tier 1-G ("resolved, with one adjustment"), the
//! panel exists to catch drift and route-substitution across generations —
//! "does the new generation still comfortably beat strong, stable,
//! untuned-against opponents", as opposed to the promotion battery's own
//! head-to-head against the immediately previous generation (still run
//! separately, via `duels-arena experiment`, with the mechanism gate — this
//! driver does not replace that, it is the "G" half of the battery, not the
//! "D vs the previous generation" half). `alphabeta` was dropped from the
//! panel (too wide a confidence interval at its ~87% win rate to detect a
//! ~20-Elo change); the frozen `v2` champion is tracked at *two* budgets
//! instead, since it is the direct ancestor at equal budget:
//!
//! | Member | Budget | Games |
//! | ------ | ------ | ----: |
//! | `mcts-value:weights=v2` (frozen) | `nodes:32000` | 800 |
//! | `mcts-value:weights=v2` (frozen) | `nodes:2000`  | 1,000 |
//! | `mcts-eval` (frozen at its current tuning) | `nodes:8000` | 800 |
//! | `mcts-uct` (the non-learned/library-free route-substitution detector) | `nodes:8000` | 800 |
//!
//! The candidate always plays at `nodes:2000` (production's own search
//! budget) against every member above.
//!
//! ```text
//! cargo run --release -p duels-arena --example reference_panel -- \
//!     --candidate mcts-value --seed 1 --label gen3-panel \
//!     [--out-dir arena/results/reference-panel] \
//!     [--sprt-elo0 0] [--sprt-elo1 20] [--alpha 0.05] [--beta 0.05]
//! ```
//!
//! `--candidate` takes any agent spec (`duels_arena::agent_spec`), so an
//! unpromoted candidate can be run through the exact same panel before a
//! promotion decision, not only the live default afterward.

use std::collections::HashMap;
use std::path::PathBuf;

use duels_agents_api::Budget;
use duels_arena::elo::fit_elo;
use duels_arena::match_runner::{
    parse_budget, play_paired_match_at_budgets, race_exposure, tally, victory_breakdown,
    MatchVictoryBreakdown, RaceExposure,
};
use duels_arena::results_io::write_results;
use duels_arena::sprt::{sprt, SprtDecision, SprtParams};
use serde::Serialize;

fn parse_flags(args: &[String]) -> HashMap<&str, &str> {
    let mut flags = HashMap::new();
    let mut i = 0;
    while i + 1 < args.len() {
        flags.insert(args[i].trim_start_matches("--"), args[i + 1].as_str());
        i += 2;
    }
    flags
}

/// One fixed panel member: `(label, agent spec, budget, games)`.
const PANEL: &[(&str, &str, Budget, u32)] = &[
    ("v2-nodes32000", "mcts-value:weights=v2", Budget::Nodes(32000), 800),
    ("v2-nodes2000", "mcts-value:weights=v2", Budget::Nodes(2000), 1000),
    ("mcts-eval-nodes8000", "mcts-eval", Budget::Nodes(8000), 800),
    ("mcts-uct-nodes8000", "mcts-uct", Budget::Nodes(8000), 800),
];

#[derive(Serialize)]
struct CellSummary {
    member: String,
    opponent: String,
    opponent_budget: String,
    games: u32,
    candidate_wins: u32,
    opponent_wins: u32,
    draws: u32,
    elo: duels_arena::elo::EloEstimate,
    sprt_decision: String,
    victory_breakdown: MatchVictoryBreakdown,
    race_exposure: RaceExposure,
}

#[derive(Serialize)]
struct PanelSummary {
    label: String,
    candidate: String,
    candidate_budget: String,
    generated_at: String,
    cells: Vec<CellSummary>,
    /// True only if every cell's SPRT decision is `AcceptH1` (candidate
    /// stronger) — a quick "did anything regress" read. This is a summary
    /// convenience, not a verdict this driver is entitled to make on its
    /// own: read each cell, exactly as `docs/conventions.md` asks.
    all_cells_accept_h1: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "reference_panel: run docs/roadmap.md's Tier 1-G frozen reference panel\n\n\
             cargo run --release -p duels-arena --example reference_panel -- \\\n    \
             --candidate <SPEC> [--candidate-budget nodes:2000] --seed <N> \\\n    \
             --label <NAME> [--out-dir <DIR>] \\\n    \
             [--sprt-elo0 <F>] [--sprt-elo1 <F>] [--alpha <F>] [--beta <F>]\n\n\
             Runs the candidate against all four fixed panel members\n\
             (frozen v2@nodes:32000/nodes:2000, mcts-eval@nodes:8000,\n\
             mcts-uct@nodes:8000) and writes one aggregate summary.\n\
             This is 'G' only -- the head-to-head against the immediately\n\
             previous generation, with the mechanism gate, is still run\n\
             separately via `duels-arena experiment`."
        );
        return;
    }
    let flags = parse_flags(&args);

    let candidate = *flags.get("candidate").unwrap_or(&"mcts-value");
    let candidate_budget = parse_budget(flags.get("candidate-budget").unwrap_or(&"nodes:2000"))
        .unwrap_or_else(|e| panic!("{e}"));
    let seed: u64 = flags.get("seed").unwrap_or(&"1").parse().unwrap();
    let label = flags
        .get("label")
        .unwrap_or_else(|| panic!("--label is required (used for the output directory)"));
    let out_dir =
        PathBuf::from(flags.get("out-dir").unwrap_or(&"arena/results/reference-panel"))
            .join(label);

    let sprt_params = SprtParams {
        elo0: flags.get("sprt-elo0").map_or(0.0, |v| v.parse().unwrap()),
        elo1: flags.get("sprt-elo1").map_or(20.0, |v| v.parse().unwrap()),
        alpha: flags.get("alpha").map_or(0.05, |v| v.parse().unwrap()),
        beta: flags.get("beta").map_or(0.05, |v| v.parse().unwrap()),
    };

    println!("reference_panel \"{label}\": {candidate} at {candidate_budget:?}");

    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", out_dir.display()));

    let mut cells = Vec::new();
    let mut all_accept = true;

    // Each panel member gets its own disjoint seed block so cells never
    // share a seed with each other -- `seed_base + member_index * 100_000`
    // leaves comfortable room under any per-member game count used here.
    for (i, &(member, opponent, opponent_budget, games)) in PANEL.iter().enumerate() {
        let base_seed = seed + (i as u64) * 100_000;
        let num_pairs = (games.div_ceil(2)).max(1);
        let seeds: Vec<u64> = (0..num_pairs as u64).map(|s| base_seed + s).collect();

        println!(
            "  cell {}/{}: {candidate}@{candidate_budget:?} vs {opponent}@{opponent_budget:?} \
             ({} games, seeds {base_seed}..{})",
            i + 1,
            PANEL.len(),
            num_pairs * 2,
            base_seed + num_pairs as u64
        );

        let records =
            play_paired_match_at_budgets(candidate, opponent, &seeds, candidate_budget, opponent_budget)
                .unwrap_or_else(|e| panic!("{member}: {e}"));
        let t = tally(&records);
        let elo = fit_elo(t.a_wins, t.b_wins, t.draws);
        let sprt_result = sprt(t.a_wins, t.b_wins, t.draws, &sprt_params);
        let vb = victory_breakdown(&records);
        let re = race_exposure(&records);

        println!(
            "    -> {} games: {}-{}-{} (W-L-D for the candidate), elo {:+.1} [{:+.1}, {:+.1}], sprt {:?}",
            t.total(),
            t.a_wins,
            t.b_wins,
            t.draws,
            elo.rating_diff,
            elo.diff_ci_low,
            elo.diff_ci_high,
            sprt_result.decision
        );

        let records_path = out_dir.join(format!("{member}-records.json"));
        write_results(&records_path, &records)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", records_path.display()));

        all_accept &= matches!(sprt_result.decision, SprtDecision::AcceptH1);
        cells.push(CellSummary {
            member: member.to_string(),
            opponent: opponent.to_string(),
            opponent_budget: format!("{opponent_budget:?}"),
            games: t.total(),
            candidate_wins: t.a_wins,
            opponent_wins: t.b_wins,
            draws: t.draws,
            elo,
            sprt_decision: format!("{:?}", sprt_result.decision),
            victory_breakdown: vb,
            race_exposure: re,
        });
    }

    let summary = PanelSummary {
        label: label.to_string(),
        candidate: candidate.to_string(),
        candidate_budget: format!("{candidate_budget:?}"),
        generated_at: humantime_now(),
        cells,
        all_cells_accept_h1: all_accept,
    };
    let summary_path = out_dir.join("summary.json");
    let json = serde_json::to_string_pretty(&summary).unwrap();
    std::fs::write(&summary_path, json)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", summary_path.display()));

    println!("wrote {}", summary_path.display());
    println!(
        "all cells AcceptH1: {} (a summary convenience -- read each cell; this is not a substitute \
         for the mechanism gate or the previous-generation head-to-head)",
        all_accept
    );
}

/// RFC3339 timestamp with no extra dependency -- matches the precision
/// `duels-arena experiment`'s own summaries use, without pulling in `chrono`
/// for one field.
#[allow(clippy::disallowed_methods, reason = "a report timestamp for a battery driver binary, never read by any game logic -- see clippy.toml")]
fn humantime_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Days since epoch -> proleptic Gregorian calendar (civil_from_days,
    // Howard Hinnant's algorithm), avoiding a chrono dependency for one
    // timestamp field.
    let days = (secs / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let rem = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}
