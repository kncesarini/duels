//! `duels-arena` CLI.
//!
//! ```text
//! duels-arena match --agent-a random --agent-b random --games 1000 \
//!     --budget nodes:2000 --seed 1 [--out arena/results/run.json] \
//!     [--sprt-elo0 0] [--sprt-elo1 5] [--alpha 0.05] [--beta 0.05]
//! ```
//!
//! `--games N` is the total number of individual games to play; internally
//! this is `ceil(N/2)` paired seeds (see `match_runner::play_paired_match`),
//! so an odd `N` is rounded up to the next even number. Prints win/loss/draw
//! counts, a logistic-Elo estimate (see `elo`), and an SPRT verdict (see
//! `sprt`) for agent A vs agent B, and writes every game's record as JSON to
//! `--out` (default: `arena/results/<a>-vs-<b>-seed<seed>-n<games>.json`).

use std::path::PathBuf;
use std::process::ExitCode;

use duels_arena::agent_registry::KNOWN_AGENTS;
use duels_arena::elo::fit_elo;
use duels_arena::match_runner::{parse_budget, play_paired_match, tally};
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
         USAGE:\n    duels-arena match --agent-a <NAME> --agent-b <NAME> --games <N> \\\n        \
         --budget <nodes:N|time_ms:N> --seed <N> [--out <PATH>]\n        \
         [--sprt-elo0 <F>] [--sprt-elo1 <F>] [--alpha <F>] [--beta <F>]\n\n\
         Known agents: {}\n\n\
         --games N is the number of individual games; internally this is\n\
         ceil(N/2) paired seeds (agent A and agent B each play both seats\n\
         for every seed), so an odd N is rounded up.\n",
        KNOWN_AGENTS.join(", ")
    );
}

/// Minimal hand-rolled `--flag value` parser: enough for this crate's one
/// subcommand without pulling in a CLI-parsing dependency.
struct Flags {
    values: std::collections::HashMap<String, String>,
}

impl Flags {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut values = std::collections::HashMap::new();
        let mut i = 0;
        while i < args.len() {
            let flag = &args[i];
            let name = flag
                .strip_prefix("--")
                .ok_or_else(|| format!("expected a --flag, got \"{flag}\""))?;
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("--{name} needs a value"))?;
            values.insert(name.to_string(), value.clone());
            i += 2;
        }
        Ok(Self { values })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_parse_required_and_optional_and_typed_values() {
        let args: Vec<String> = [
            "--agent-a",
            "random",
            "--agent-b",
            "random",
            "--games",
            "10",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let flags = Flags::parse(&args).unwrap();
        assert_eq!(flags.required("agent-a").unwrap(), "random");
        assert_eq!(flags.parsed::<u32>("games", 0).unwrap(), 10);
        assert_eq!(flags.parsed::<u32>("missing", 42).unwrap(), 42);
        assert!(flags.required("nope").is_err());
    }

    #[test]
    fn flags_reject_a_dangling_flag_without_a_value() {
        let args: Vec<String> = ["--agent-a"].into_iter().map(String::from).collect();
        assert!(Flags::parse(&args).is_err());
    }
}
