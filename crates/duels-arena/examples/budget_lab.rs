//! Self-play budget-scaling harness: the same agent spec against itself at
//! two different budgets, to price a budget doubling (or any other ratio)
//! directly in Elo.
//!
//! `duels-arena match`/`experiment` always play both sides of a cell at one
//! shared [`Budget`] (see `match_runner::play_one_game`), which is the right
//! design for "agent A vs agent B at equal search cost" but cannot express
//! "agent A at `Nodes(2000)` vs the same agent at `Nodes(4000)`". This
//! example is the asymmetric-budget primitive that measurement actually
//! needs — the same shape as the "half-budget side" comparisons already in
//! `duels-agent-mcts-uct`'s and `duels-agent-mcts-eval`'s crate docs (a
//! candidate at half budget against a fixed-budget control), generalized to
//! any two budgets and driven from the CLI rather than a one-off script. See
//! `duels_arena::match_runner::play_paired_match_at_budgets`, which does the
//! actual asymmetric-budget game-playing this wraps.
//!
//! ```text
//! cargo run --release -p duels-arena --example budget_lab -- \
//!     --agent mcts-value --budget-a nodes:2000 --budget-b nodes:4000 \
//!     --games 400 --seed 1 --label mcts-value-2000-vs-4000 \
//!     [--out-dir arena/results/experiments] \
//!     [--sprt-elo0 0] [--sprt-elo1 20] [--alpha 0.05] [--beta 0.05]
//! ```
//!
//! `--agent` is a bare agent name or a full `name:key=value,...` spec (see
//! `duels_arena::agent_spec`) — the *same* spec plays both budgets, since
//! this is a self-play measurement, not an A/B agent comparison. `agent-a`
//! always plays at `--budget-a`, `agent-b` at `--budget-b` — reported as `A`
//! throughout, matching `duels-arena match`'s convention of quoting Elo from
//! "agent A"'s perspective.
//!
//! Writes `<out-dir>/<label>/records.json` (every game, via the same
//! `results_io::write_results` a `match`/`experiment` run uses) and
//! `<out-dir>/<label>/summary.json` (tally, victory breakdown, race
//! exposure, Elo estimate, SPRT verdict, and both budgets — everything
//! this file prints to stdout, so a number quoted from a run is always
//! backed by a file on disk).

use std::collections::HashMap;
use std::path::PathBuf;

use duels_arena::elo::fit_elo;
use duels_arena::match_runner::{
    parse_budget, play_paired_match_at_budgets, race_exposure, tally, victory_breakdown,
};
use duels_arena::results_io::write_results;
use duels_arena::sprt::{sprt, SprtParams};
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

#[derive(Serialize)]
struct BudgetLabSummary {
    agent: String,
    budget_a: String,
    budget_b: String,
    games: u32,
    a_wins: u32,
    b_wins: u32,
    draws: u32,
    elo: duels_arena::elo::EloEstimate,
    sprt_params: SprtParams,
    sprt_llr: f64,
    sprt_decision: String,
    victory_breakdown: duels_arena::match_runner::MatchVictoryBreakdown,
    race_exposure: duels_arena::match_runner::RaceExposure,
    avg_moves_per_game: f64,
    avg_wall_ms_per_game: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "budget_lab: self-play budget-scaling harness for one agent spec\n\n\
             cargo run --release -p duels-arena --example budget_lab -- \\\n    \
             --agent <SPEC> --budget-a <B> --budget-b <B> --games <N> --seed <N> \\\n    \
             --label <NAME> [--out-dir <DIR>] \\\n    \
             [--sprt-elo0 <F>] [--sprt-elo1 <F>] [--alpha <F>] [--beta <F>]\n\n\
             <B> is \"nodes:<n>\" or \"time_ms:<n>\". Reports everything from\n\
             \"agent A\" (budget-a)'s perspective, matching `duels-arena match`."
        );
        return;
    }
    let flags = parse_flags(&args);

    let agent = *flags.get("agent").unwrap_or(&"mcts-value");
    let budget_a =
        parse_budget(flags.get("budget-a").unwrap_or_else(|| panic!("--budget-a is required")))
            .unwrap_or_else(|e| panic!("{e}"));
    let budget_b =
        parse_budget(flags.get("budget-b").unwrap_or_else(|| panic!("--budget-b is required")))
            .unwrap_or_else(|e| panic!("{e}"));
    let games: u32 = flags
        .get("games")
        .unwrap_or(&"400")
        .parse()
        .unwrap_or_else(|_| panic!("--games must be a non-negative integer"));
    let seed: u64 = flags.get("seed").unwrap_or(&"1").parse().unwrap();
    let label = flags
        .get("label")
        .unwrap_or_else(|| panic!("--label is required (used for the output directory)"));
    let out_dir = PathBuf::from(flags.get("out-dir").unwrap_or(&"arena/results/experiments"))
        .join(label);

    let sprt_params = SprtParams {
        elo0: flags
            .get("sprt-elo0")
            .map_or(0.0, |v| v.parse().unwrap()),
        elo1: flags
            .get("sprt-elo1")
            .map_or(20.0, |v| v.parse().unwrap()),
        alpha: flags.get("alpha").map_or(0.05, |v| v.parse().unwrap()),
        beta: flags.get("beta").map_or(0.05, |v| v.parse().unwrap()),
    };

    let num_pairs = (games.div_ceil(2)).max(1);
    let seeds: Vec<u64> = (0..num_pairs as u64).map(|i| seed + i).collect();

    println!(
        "budget_lab: {agent} at {budget_a:?} (A) vs {agent} at {budget_b:?} (B)  \
         ({} games = {num_pairs} paired seeds, base seed {seed})",
        num_pairs * 2
    );

    let records = play_paired_match_at_budgets(agent, agent, &seeds, budget_a, budget_b)
        .unwrap_or_else(|e| panic!("{e}"));
    let t = tally(&records);

    let total_moves: u64 = records.iter().map(|r| r.moves as u64).sum();
    let total_wall_ms: u64 = records.iter().map(|r| r.wall_time_ms).sum();
    let avg_moves = total_moves as f64 / t.total() as f64;
    let avg_wall_ms = total_wall_ms as f64 / t.total() as f64;

    println!(
        "results: A(={budget_a:?}) {} wins, B(={budget_b:?}) {} wins, {} draws  (out of {})",
        t.a_wins,
        t.b_wins,
        t.draws,
        t.total()
    );
    println!(
        "avg moves/game: {avg_moves:.1}   avg wall time/game: {avg_wall_ms:.1} ms   total wall time: {total_wall_ms} ms"
    );

    let vb = victory_breakdown(&records);
    println!(
        "victory kinds: A {} wins (military {}, scientific {}, civilian {}, tiebreak {})   \
         B {} wins (military {}, scientific {}, civilian {}, tiebreak {})",
        vb.a.total(),
        vb.a.military_supremacy,
        vb.a.scientific_supremacy,
        vb.a.civilian_victory,
        vb.a.civilian_tiebreak,
        vb.b.total(),
        vb.b.military_supremacy,
        vb.b.scientific_supremacy,
        vb.b.civilian_victory,
        vb.b.civilian_tiebreak,
    );

    let re = race_exposure(&records);
    println!(
        "race exposure: military in {}/{} games ({:.0}%)   scientific in {}/{} games ({:.0}%)",
        re.military_games,
        re.total_games,
        100.0 * re.military_games as f64 / re.total_games as f64,
        re.science_games,
        re.total_games,
        100.0 * re.science_games as f64 / re.total_games as f64,
    );

    let elo_estimate = fit_elo(t.a_wins, t.b_wins, t.draws);
    println!(
        "elo: A(={budget_a:?}) = {:+.1} relative to B(={budget_b:?}), 95% CI [{:+.1}, {:+.1}]",
        elo_estimate.rating_diff, elo_estimate.diff_ci_low, elo_estimate.diff_ci_high
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

    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", out_dir.display()));

    let records_path = out_dir.join("records.json");
    write_results(&records_path, &records)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", records_path.display()));

    let summary = BudgetLabSummary {
        agent: agent.to_string(),
        budget_a: format!("{budget_a:?}"),
        budget_b: format!("{budget_b:?}"),
        games: t.total(),
        a_wins: t.a_wins,
        b_wins: t.b_wins,
        draws: t.draws,
        elo: elo_estimate,
        sprt_params,
        sprt_llr: sprt_result.llr,
        sprt_decision: format!("{:?}", sprt_result.decision),
        victory_breakdown: vb,
        race_exposure: re,
        avg_moves_per_game: avg_moves,
        avg_wall_ms_per_game: avg_wall_ms,
    };
    let summary_path = out_dir.join("summary.json");
    let json = serde_json::to_string_pretty(&summary).unwrap();
    std::fs::write(&summary_path, json)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", summary_path.display()));

    println!(
        "wrote {} game records to {} and a summary to {}",
        records.len(),
        records_path.display(),
        summary_path.display()
    );
}
