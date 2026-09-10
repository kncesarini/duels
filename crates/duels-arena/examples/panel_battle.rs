//! Asymmetric-agent, asymmetric-budget battle harness — `budget_lab.rs`'s
//! twin, generalized from "one agent spec at two budgets" to "two
//! independent agent specs, each at its own budget".
//!
//! # Why this exists
//!
//! `duels-arena match`/`experiment` always play both sides of a cell at one
//! shared [`Budget`] (`match_runner::play_one_game`) — the right design for
//! "agent A vs agent B at equal search cost". `budget_lab.rs` covers "the
//! same agent spec at two different budgets" (a self-play scaling curve).
//! Neither expresses `docs/roadmap.md`'s Tier 1-G frozen reference panel,
//! which is deliberately **both** at once: a candidate at its own production
//! budget (`nodes:2000`) against e.g. the frozen champion buffed to
//! `nodes:32000`, or `mcts-eval`/`mcts-uct` at `nodes:8000` — a different
//! agent *and* a different budget on each side, on purpose (the panel members
//! are meant to be strong, stable, and untuned against, not equal-cost
//! opponents). `match_runner::play_paired_match_at_budgets` already supports
//! exactly this (two independent agent spec strings, two independent
//! budgets) — `budget_lab.rs`'s own CLI just doesn't expose the first half of
//! that generality, since it never needed to. This tool is that same
//! function with both halves exposed, and is otherwise identical to
//! `budget_lab.rs`: same summary shape, same paired-seed, seat-swapped
//! design, same output layout.
//!
//! ```text
//! cargo run --release -p duels-arena --example panel_battle -- \
//!     --agent-a "mcts-value:weights=arm-a" --budget-a nodes:2000 \
//!     --agent-b mcts-value --budget-b nodes:32000 \
//!     --games 800 --seed 1 --label arm-a-vs-v2-nodes32000 \
//!     [--out-dir arena/results/experiments] \
//!     [--sprt-elo0 0] [--sprt-elo1 20] [--alpha 0.05] [--beta 0.05]
//! ```

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
struct PanelBattleSummary {
    agent_a: String,
    budget_a: String,
    agent_b: String,
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
            "panel_battle: asymmetric-agent, asymmetric-budget battle harness\n\n\
             cargo run --release -p duels-arena --example panel_battle -- \\\n    \
             --agent-a <SPEC> --budget-a <B> --agent-b <SPEC> --budget-b <B> \\\n    \
             --games <N> --seed <N> --label <NAME> [--out-dir <DIR>] \\\n    \
             [--sprt-elo0 <F>] [--sprt-elo1 <F>] [--alpha <F>] [--beta <F>]\n\n\
             <SPEC> is a bare agent name or \"name:key=value,...\" (see\n\
             duels_arena::agent_spec). <B> is \"nodes:<n>\" or \"time_ms:<n>\".\n\
             Reports everything from agent A's perspective, matching\n\
             `duels-arena match`/`budget_lab`."
        );
        return;
    }
    let flags = parse_flags(&args);

    let agent_a = *flags
        .get("agent-a")
        .unwrap_or_else(|| panic!("--agent-a is required"));
    let agent_b = *flags
        .get("agent-b")
        .unwrap_or_else(|| panic!("--agent-b is required"));
    let budget_a = parse_budget(
        flags
            .get("budget-a")
            .unwrap_or_else(|| panic!("--budget-a is required")),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let budget_b = parse_budget(
        flags
            .get("budget-b")
            .unwrap_or_else(|| panic!("--budget-b is required")),
    )
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
    let out_dir =
        PathBuf::from(flags.get("out-dir").unwrap_or(&"arena/results/experiments")).join(label);

    let sprt_params = SprtParams {
        elo0: flags.get("sprt-elo0").map_or(0.0, |v| v.parse().unwrap()),
        elo1: flags.get("sprt-elo1").map_or(20.0, |v| v.parse().unwrap()),
        alpha: flags.get("alpha").map_or(0.05, |v| v.parse().unwrap()),
        beta: flags.get("beta").map_or(0.05, |v| v.parse().unwrap()),
    };

    let num_pairs = (games.div_ceil(2)).max(1);
    let seeds: Vec<u64> = (0..num_pairs as u64).map(|i| seed + i).collect();

    println!(
        "panel_battle: {agent_a} at {budget_a:?} (A) vs {agent_b} at {budget_b:?} (B)  \
         ({} games = {num_pairs} paired seeds, base seed {seed})",
        num_pairs * 2
    );

    let records = play_paired_match_at_budgets(agent_a, agent_b, &seeds, budget_a, budget_b)
        .unwrap_or_else(|e| panic!("{e}"));
    let t = tally(&records);

    let total_moves: u64 = records.iter().map(|r| r.moves as u64).sum();
    let total_wall_ms: u64 = records.iter().map(|r| r.wall_time_ms).sum();
    let avg_moves = total_moves as f64 / t.total() as f64;
    let avg_wall_ms = total_wall_ms as f64 / t.total() as f64;

    println!(
        "results: A({agent_a}@{budget_a:?}) {} wins, B({agent_b}@{budget_b:?}) {} wins, {} draws  (out of {})",
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
        "elo: A({agent_a}@{budget_a:?}) = {:+.1} relative to B({agent_b}@{budget_b:?}), 95% CI [{:+.1}, {:+.1}]",
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

    let summary = PanelBattleSummary {
        agent_a: agent_a.to_string(),
        budget_a: format!("{budget_a:?}"),
        agent_b: agent_b.to_string(),
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
