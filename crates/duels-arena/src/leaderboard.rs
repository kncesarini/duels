//! The arena leaderboard: who is actually strongest, over a full round robin.
//!
//! This is the data half of milestone M6b ("Arena live"). The workflow half
//! lives in `.github/workflows/nightly-arena.yml`, which runs every pairing
//! [`pairings`] lists as its own CI job and then calls
//! `duels-arena leaderboard` to turn the results into
//! `arena/leaderboard.json` (machine-readable, the input to any later
//! tooling) and `arena/leaderboard.md` (the human-readable table checked into
//! the repository).
//!
//! # The ladder, and why one budget runs all of it
//!
//! [`LADDER`] names the agents tracked and the budget each is *understood* to
//! play at: `Nodes(1)` for `phased`, the one remaining 1-ply agent, and
//! `Nodes(2000)` for the three search agents, matching how every ladder
//! comparison in this project's history has been run.
//!
//! The ladder is deliberately four agents. `strategist` retired first (its
//! research question, whether `duels-strategy`'s prior helps `greedy-ev`, was
//! answered statistically indistinguishable), and then `random`, `greedy` and
//! `greedy-ev` — the whole 1-ply floor tier below `phased` — went for measured
//! strength far below the rest of the roster: the last full refit had all
//! three inside a 200-Elo band scoring 0.0%-0.5% against every top-half agent,
//! so they cost the nightly fifteen of its twenty-one pairings and told it
//! nothing it did not already know. See `docs/milestones.md`; that retirement
//! is also what moved [`ANCHOR_AGENT`].
//!
//! `duels-arena match` grants both sides the same budget, so a mixed pairing
//! (`phased` vs `mcts-uct`, say) looks at first like it cannot honour both
//! numbers at once. It can: `phased` takes `_budget` in its `Agent::choose`
//! signature and never reads it, so a `Nodes(1)` and a `Nodes(2000)` `phased`
//! are *the same agent*. The whole round robin therefore runs at
//! [`ROUND_ROBIN_BUDGET`], and the per-agent budgets in [`LADDER`] are labels
//! on the report rather than a second thing to configure.
//! `tests::one_ply_agents_ignore_their_budget` checks this by playing games
//! rather than by trusting the signature.
//!
//! Each agent is tracked at its default configuration only. Notable config
//! variants (`phased:wonder=budget` and friends) belong in the per-agent
//! investigation docs where their comparison is controlled; a leaderboard
//! that mixed defaults and variants would invite reading a within-agent
//! ablation as a between-agent ranking.
//!
//! # Registered is not the same as rated
//!
//! Being constructible through `agent_registry` no longer implies being on
//! [`LADDER`]. [`REGISTERED_OFF_LADDER`] is the explicit list of agents that
//! are runnable, playable and spec-string addressable while carrying no
//! rating, and `tests::the_ladder_is_exactly_the_registered_agents` still pins
//! the two lists to each other up to it — so nothing falls off the board
//! silently, and every exception has to justify itself in that constant's
//! docs. `mcts-value` is the current entry, and the reason is worth reading
//! there: a large, reproducible margin over one specific opponent is not a
//! position in a transitive ranking.
//!
//! # The champion
//!
//! [`CHAMPION`] designates the agent an `ai-candidate` CI run measures a
//! changed agent against. It is deliberately a plain constant, not a value
//! read back out of the leaderboard: promoting a new champion automatically
//! is milestone M7, and until that exists a human changing one line here is
//! the honest mechanism.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::elo::{fit_joint_elo, JointEloTable, PairwiseRecord};
use crate::match_runner::{tally, GameRecord};
use crate::results_io::{read_results, ResultsFile};

/// One tracked agent and the budget it is understood to play at. See the
/// module docs for why the budget here is a label rather than a knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LadderEntry {
    /// The agent's `agent_registry` name, at its default configuration.
    pub agent: &'static str,
    /// The budget this agent's results should be read as, for the report.
    pub budget: &'static str,
}

/// Every agent on the leaderboard, with its production budget.
pub const LADDER: &[LadderEntry] = &[
    LadderEntry {
        agent: "phased",
        budget: "nodes:1",
    },
    LadderEntry {
        agent: "alphabeta",
        budget: "nodes:2000",
    },
    LadderEntry {
        agent: "mcts-uct",
        budget: "nodes:2000",
    },
    // A search agent, so it belongs in the `Nodes(2000)` tier alongside the
    // other two rather than with the 1-ply agents — its budget is read, and
    // the whole round robin is played at `ROUND_ROBIN_BUDGET` anyway.
    LadderEntry {
        agent: "mcts-eval",
        budget: "nodes:2000",
    },
];

/// Agents that `agent_registry` can construct but that are deliberately
/// **not** on [`LADDER`]: constructible, playable, spec-string addressable,
/// and unrated.
///
/// # Why this list exists at all
///
/// Registration and rating used to be the same act — `KNOWN_AGENTS` and
/// [`LADDER`] were pinned equal to each other, and a retired agent was deleted
/// from both. That equality was a good default and it is kept: this list is
/// the *explicit, documented* exception, and
/// `tests::the_ladder_is_exactly_the_registered_agents` still holds up to it,
/// so an agent cannot drift off the board by accident.
///
/// An entry belongs here when an agent should be **runnable but not
/// ranked** — typically because a measurement is real but does not support a
/// ranking claim. Putting an agent on [`LADDER`] asserts that its rating is a
/// meaningful position in a transitive ordering; an agent whose strength is
/// established against exactly one opponent has not earned that.
///
/// # The current entry
///
/// `mcts-value` measures `+91.4` Elo `[+66.5, +116.3]` over 800 games against
/// [`CHAMPION`] at [`ROUND_ROBIN_BUDGET`], and larger at higher budgets. It is
/// off the ladder anyway, because a mini round robin found that margin does
/// not survive a third party: 28% of it through `mcts-uct` and 12% through
/// `alphabeta`, both intervals containing zero, with a joint Bradley-Terry fit
/// over all five records putting the pair 74 points apart where the direct
/// match says 91.5. The mechanism is route substitution against one opponent's
/// documented science-value miscalibration rather than added strength — that
/// agent's crate docs have the whole measurement, including the victory-kind
/// table that says so.
///
/// Rating it would put a number on the board that means "beats `mcts-eval`"
/// while reading as "is the strongest agent", and the nightly refit would keep
/// republishing it. Whether to promote it is the project owner's decision on
/// its own evidence, exactly as moving [`CHAMPION`] is; this constant is where
/// that decision is *deferred*, visibly, rather than made by a side effect of
/// registering a crate.
pub const REGISTERED_OFF_LADDER: &[&str] = &["mcts-value"];

/// The budget every round-robin pairing is actually played at. Equivalent to
/// each agent's own [`LadderEntry::budget`] because the 1-ply agents ignore
/// theirs — see the module docs.
pub const ROUND_ROBIN_BUDGET: &str = "nodes:2000";

/// The agent whose rating pins the leaderboard's scale, and the value it is
/// pinned to.
///
/// # Why `mcts-uct`, and what changed
///
/// This was `greedy` at 1000, following the original architecture design's
/// "BayesElo anchored at greedy-v1 = 1000". The reasoning there is the part
/// worth keeping: **a never-changing baseline is the right thing to pin,
/// because every other agent's number then moves only when *that* agent's
/// strength moves.** `greedy` qualified because its evaluation was a frozen,
/// hand-written formula in its own crate that nothing else tuned.
///
/// `greedy` was retired from the roster, so the scale needed a new pin from
/// what is left: `phased`, `alphabeta`, `mcts-uct`, `mcts-eval`. The obvious
/// positional analogue is `phased` — the weakest survivor, 1-ply, and
/// budget-invariant. It is the wrong choice, and for exactly the reason above:
/// `phased`'s `Config` *is* [`duels_eval::Config`], and `PhasedAgent::new`
/// reads `duels_eval::Config::default()` live. `duels-eval` is re-tuned in
/// numbered rounds (ten of them so far, the most recent moving a default
/// weight), and each one silently redefines `phased`'s strength. Pinning the
/// scale there would shift *every* rating on the board on every tuning round,
/// which is precisely the failure the anchor exists to prevent.
///
/// `mcts-uct` is the defensible pin:
///
/// * It does not depend on `duels-eval` at all (check its `Cargo.toml`), and
///   its default `PriorMode::None` does not consult `duels-strategy` either,
///   so no library round can move it.
/// * Its `Config::default()` is frozen, and `mcts-eval` carries a verbatim
///   copy of its search as an ablation control asserted move-for-move — a
///   silent change to it fails a test in another crate.
/// * It is already this project's canonical yardstick: every knob in
///   `mcts-eval` was tuned against it, and it held [`CHAMPION`] until
///   `mcts-eval` measured past it.
/// * It sits second of four, so ratings still spread either side of 1000.
///
/// What it gives up against `greedy` is budget-independence: it is a search,
/// so its strength is a function of its budget. That is pinned too — the whole
/// round robin is played at [`ROUND_ROBIN_BUDGET`], which is `mcts-uct`'s own
/// ladder budget.
///
/// **Every Elo number generated before this change was measured against
/// `greedy` = 1000 and is not comparable to one measured after it.** The
/// ratings in `arena/leaderboard.{json,md}` are refitted from scratch by the
/// next nightly round robin; nothing rescales the old numbers, and they should
/// not be read alongside the new ones.
pub const ANCHOR_AGENT: &str = "mcts-uct";
/// See [`ANCHOR_AGENT`]. Unchanged at the conventional scale origin: only
/// *which* agent sits at 1000 moved, not the number it is pinned to.
pub const ANCHOR_ELO: f64 = 1000.0;

/// The reigning champion: the agent and budget a candidate agent is measured
/// against by the `ai-candidate` CI check. See the module docs on why this is
/// a hand-maintained constant.
pub const CHAMPION: LadderEntry = LadderEntry {
    agent: "mcts-eval",
    budget: "nodes:2000",
};

/// Every unordered pairing of [`LADDER`] agents, in a stable order — the
/// `C(n, 2)` = 6 matches one nightly round robin consists of.
pub fn pairings() -> Vec<(&'static str, &'static str)> {
    let mut out = Vec::new();
    for (i, a) in LADDER.iter().enumerate() {
        for b in &LADDER[i + 1..] {
            out.push((a.agent, b.agent));
        }
    }
    out
}

/// The filename `duels-arena match` results for one pairing are stored under,
/// so the nightly workflow's per-pairing artifacts and the aggregation step
/// agree without either having to parse the other's naming scheme.
pub fn pairing_results_filename(agent_a: &str, agent_b: &str) -> String {
    format!("{agent_a}--vs--{agent_b}.json")
}

/// One agent's row in the rendered leaderboard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeaderboardRow {
    /// 1 for the strongest agent.
    pub rank: u32,
    pub agent: String,
    /// The budget this agent is understood to play at ([`LadderEntry::budget`]).
    pub budget: String,
    pub elo: f64,
    pub elo_ci_low: f64,
    pub elo_ci_high: f64,
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    /// Whether this agent is the designated [`CHAMPION`].
    pub champion: bool,
}

/// One head-to-head cell of the round robin, kept alongside the ratings so a
/// reader can check a surprising rating against the games behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairingRow {
    pub agent_a: String,
    pub agent_b: String,
    pub a_wins: u32,
    pub b_wins: u32,
    pub draws: u32,
}

/// The complete leaderboard: what `arena/leaderboard.json` holds and
/// `arena/leaderboard.md` renders.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Leaderboard {
    /// Schema version, so a later reader can tell an old file from a new one.
    pub schema: u32,
    /// UTC RFC-3339 timestamp of the run that produced this table.
    pub generated_at: String,
    /// The commit the agents were built from.
    pub commit: String,
    /// The budget every pairing was played at ([`ROUND_ROBIN_BUDGET`]).
    pub budget: String,
    pub anchor_agent: String,
    pub anchor_elo: f64,
    /// [`CHAMPION`], recorded so a consumer of the JSON doesn't have to
    /// hard-code it a second time.
    pub champion_agent: String,
    pub champion_budget: String,
    /// Ratings, strongest first.
    pub rows: Vec<LeaderboardRow>,
    /// Every pairing's head-to-head record.
    pub pairings: Vec<PairingRow>,
    /// Total games across the whole round robin.
    pub total_games: u32,
    /// Whether the joint Elo fit converged (see [`fit_joint_elo`]).
    pub converged: bool,
}

/// The current schema version of [`Leaderboard`].
pub const SCHEMA_VERSION: u32 = 1;

/// Which agent played "role A" and which "role B" in a set of records, read
/// back off the records themselves rather than trusted from a filename.
///
/// # Errors
///
/// If `records` is empty, or if it does not describe a single consistent
/// pairing of two distinct agents.
pub fn pairing_of(records: &[GameRecord]) -> Result<(String, String), String> {
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    for r in records {
        let (a, b) = match r.agent_a_seat {
            duels_core::Player::One => (&r.seat_one.name, &r.seat_two.name),
            duels_core::Player::Two => (&r.seat_two.name, &r.seat_one.name),
        };
        seen.insert((a.clone(), b.clone()));
    }
    match seen.len() {
        0 => Err("no game records, so no pairing to identify".to_string()),
        1 => {
            let (a, b) = seen.into_iter().next().expect("len == 1");
            if a == b {
                return Err(format!(
                    "records are a self-play match ({a} vs {a}); a leaderboard needs distinct agents"
                ));
            }
            Ok((a, b))
        }
        _ => Err(format!(
            "records mix more than one pairing ({}), which a single results file should never do",
            seen.into_iter()
                .map(|(a, b)| format!("{a} vs {b}"))
                .collect::<Vec<_>>()
                .join("; ")
        )),
    }
}

/// Turn one match's results file into the head-to-head record the joint Elo
/// fit consumes.
pub fn pairwise_record_of(results: &ResultsFile) -> Result<PairwiseRecord, String> {
    let (agent_a, agent_b) = pairing_of(&results.records)?;
    // Recompute rather than trusting the file's stored `tally`: the two are
    // supposed to agree, and if a hand-edited file makes them disagree the
    // games are the ground truth.
    let t = tally(&results.records);
    Ok(PairwiseRecord {
        agent_a,
        agent_b,
        wins: t.a_wins,
        losses: t.b_wins,
        draws: t.draws,
    })
}

/// Read every `*.json` results file under `dir` (recursively, so the nightly
/// workflow can point at a directory of downloaded per-pairing artifacts
/// without flattening them first) and reduce each to one head-to-head record.
///
/// Returns the records sorted by `(agent_a, agent_b)` so the fit is
/// reproducible regardless of directory-iteration order.
pub fn collect_pairwise_records(dir: &Path) -> Result<Vec<PairwiseRecord>, String> {
    let mut files = Vec::new();
    collect_json_files(dir, &mut files)?;
    files.sort();

    let mut out = Vec::new();
    for path in files {
        let results =
            read_results(&path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        if results.records.is_empty() {
            return Err(format!("{} holds no game records", path.display()));
        }
        let record = pairwise_record_of(&results)
            .map_err(|e| format!("{} is not a usable pairing: {e}", path.display()))?;
        out.push(record);
    }
    if out.is_empty() {
        return Err(format!(
            "no *.json results files found under {}",
            dir.display()
        ));
    }
    out.sort_by(|x, y| {
        (&x.agent_a, &x.agent_b)
            .cmp(&(&y.agent_a, &y.agent_b))
            .then_with(|| x.wins.cmp(&y.wins))
    });
    Ok(out)
}

fn collect_json_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        fs::read_dir(dir).map_err(|e| format!("failed to list {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to list {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "json") {
            out.push(path);
        }
    }
    Ok(())
}

/// Fit the joint Elo table and assemble the leaderboard.
///
/// Every agent named in `records` must be on the [`LADDER`], and every
/// [`pairings`] pair must be present exactly once — a leaderboard silently
/// missing a matrix job's results would still fit, but its ratings would not
/// be the round robin they claim to be.
pub fn build(
    records: &[PairwiseRecord],
    generated_at: &str,
    commit: &str,
) -> Result<Leaderboard, String> {
    check_round_robin_is_complete(records)?;
    let table = fit_joint_elo(records, ANCHOR_AGENT, ANCHOR_ELO)?;
    Ok(assemble(&table, records, generated_at, commit))
}

/// Reject a set of records that isn't exactly the round robin [`pairings`]
/// describes: an unknown agent, a missing pairing, or a duplicated one.
fn check_round_robin_is_complete(records: &[PairwiseRecord]) -> Result<(), String> {
    let known: BTreeSet<&str> = LADDER.iter().map(|e| e.agent).collect();
    for r in records {
        for name in [&r.agent_a, &r.agent_b] {
            if !known.contains(name.as_str()) {
                return Err(format!(
                    "\"{name}\" is not on the leaderboard ladder ({}); either add it to \
                     `LADDER` or leave its results out of the round robin",
                    known.iter().copied().collect::<Vec<_>>().join(", ")
                ));
            }
        }
    }

    let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
    for r in records {
        let key = unordered(&r.agent_a, &r.agent_b);
        if !seen.insert(key) {
            return Err(format!(
                "pairing {} vs {} appears more than once; each matrix job should write exactly \
                 one results file",
                r.agent_a, r.agent_b
            ));
        }
    }

    let missing: Vec<String> = pairings()
        .into_iter()
        .filter(|&(a, b)| !seen.contains(&unordered(a, b)))
        .map(|(a, b)| format!("{a} vs {b}"))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "the round robin is incomplete - {} of {} pairings have no results ({}). \
             Did a nightly matrix job fail?",
            missing.len(),
            pairings().len(),
            missing.join("; ")
        ));
    }
    Ok(())
}

fn unordered<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn assemble(
    table: &JointEloTable,
    records: &[PairwiseRecord],
    generated_at: &str,
    commit: &str,
) -> Leaderboard {
    let budget_of = |agent: &str| {
        LADDER
            .iter()
            .find(|e| e.agent == agent)
            .map(|e| e.budget)
            .unwrap_or(ROUND_ROBIN_BUDGET)
            .to_string()
    };

    let rows: Vec<LeaderboardRow> = table
        .ratings
        .iter()
        .enumerate()
        .map(|(i, r)| LeaderboardRow {
            rank: i as u32 + 1,
            agent: r.agent.clone(),
            budget: budget_of(&r.agent),
            elo: r.elo,
            elo_ci_low: r.elo_ci_low,
            elo_ci_high: r.elo_ci_high,
            games: r.games,
            wins: r.wins,
            losses: r.losses,
            draws: r.draws,
            champion: r.agent == CHAMPION.agent,
        })
        .collect();

    let pairings: Vec<PairingRow> = records
        .iter()
        .map(|r| PairingRow {
            agent_a: r.agent_a.clone(),
            agent_b: r.agent_b.clone(),
            a_wins: r.wins,
            b_wins: r.losses,
            draws: r.draws,
        })
        .collect();
    let total_games = pairings.iter().map(|p| p.a_wins + p.b_wins + p.draws).sum();

    Leaderboard {
        schema: SCHEMA_VERSION,
        generated_at: generated_at.to_string(),
        commit: commit.to_string(),
        budget: ROUND_ROBIN_BUDGET.to_string(),
        anchor_agent: table.anchor_agent.clone(),
        anchor_elo: table.anchor_elo,
        champion_agent: CHAMPION.agent.to_string(),
        champion_budget: CHAMPION.budget.to_string(),
        rows,
        pairings,
        total_games,
        converged: table.converged,
    }
}

/// Render `board` as the Markdown checked in at `arena/leaderboard.md`.
pub fn render_markdown(board: &Leaderboard) -> String {
    let mut out = String::new();
    out.push_str("# Arena leaderboard\n\n");
    out.push_str(
        "Generated by `.github/workflows/nightly-arena.yml`. **Do not edit by hand** - the next \
         nightly round robin overwrites this file.\n\n",
    );
    out.push_str(&format!(
        "- Generated: `{}`\n- Commit: `{}`\n- Budget: `{}` for every pairing (the 1-ply \
         agents ignore their budget, so this is also `nodes:1` for them - see \
         `duels_arena::leaderboard`)\n- Anchor: `{}` pinned at {:.0} Elo (it replaced `greedy` \
         when the 1-ply floor tier was retired, so these numbers are on a different scale \
         from any generated before that - see `duels_arena::leaderboard::ANCHOR_AGENT`)\n\
         - Champion: `{}` at `{}`\n- Total games: {} across {} pairings, paired-seed and \
         seat-swapped\n\n",
        board.generated_at,
        board.commit,
        board.budget,
        board.anchor_agent,
        board.anchor_elo,
        board.champion_agent,
        board.champion_budget,
        board.total_games,
        board.pairings.len(),
    ));

    if !board.converged {
        out.push_str(
            "> **Warning:** the joint Elo fit did not converge. Treat these ratings as \
             provisional.\n\n",
        );
    }

    out.push_str("## Ratings\n\n");
    out.push_str("| Rank | Agent | Budget | Elo | 95% CI | Games | W-L-D |\n");
    out.push_str("| ---: | ----- | ------ | --: | ------ | ----: | ----- |\n");
    for row in &board.rows {
        let name = if row.champion {
            format!("**{}** (champion)", row.agent)
        } else {
            row.agent.clone()
        };
        let ci = if row.agent == board.anchor_agent {
            "anchor".to_string()
        } else {
            format!("[{:.0}, {:.0}]", row.elo_ci_low, row.elo_ci_high)
        };
        out.push_str(&format!(
            "| {} | {} | `{}` | {:.0} | {} | {} | {}-{}-{} |\n",
            row.rank, name, row.budget, row.elo, ci, row.games, row.wins, row.losses, row.draws,
        ));
    }

    out.push_str("\n## Head to head\n\n");
    out.push_str("| Pairing | A wins | B wins | Draws | A score |\n");
    out.push_str("| ------- | -----: | -----: | ----: | ------: |\n");
    for p in &board.pairings {
        let total = p.a_wins + p.b_wins + p.draws;
        let score = if total == 0 {
            0.0
        } else {
            (p.a_wins as f64 + p.draws as f64 * 0.5) / total as f64
        };
        out.push_str(&format!(
            "| {} vs {} | {} | {} | {} | {:.1}% |\n",
            p.agent_a,
            p.agent_b,
            p.a_wins,
            p.b_wins,
            p.draws,
            100.0 * score,
        ));
    }

    out.push_str(
        "\n## How to read this\n\n\
         Elo is fitted jointly over every pairing at once (a Bradley-Terry MLE, see \
         `duels_arena::elo::fit_joint_elo`), not by anchoring each agent independently against \
         one reference - so an agent's rating is informed by every game in the round robin, \
         including games it did not play, through its opponents. Intervals are 95% asymptotic \
         MLE intervals on the difference from the anchor, which is why the anchor's own row has \
         none: it is pinned, not estimated.\n\n\
         Every pairing is played with paired seeds and swapped seats (each seed is played twice, \
         once from each side), because first-player advantage in this game is large enough to \
         swamp the effects being measured.\n",
    );

    out
}

/// Write `board` to `json_path` and its rendered Markdown to `md_path`,
/// creating parent directories as needed.
pub fn write(board: &Leaderboard, json_path: &Path, md_path: &Path) -> Result<(), String> {
    for path in [json_path, md_path] {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
            }
        }
    }
    let json = serde_json::to_string_pretty(board)
        .map_err(|e| format!("failed to serialize the leaderboard: {e}"))?;
    fs::write(json_path, format!("{json}\n"))
        .map_err(|e| format!("failed to write {}: {e}", json_path.display()))?;
    fs::write(md_path, render_markdown(board))
        .map_err(|e| format!("failed to write {}: {e}", md_path.display()))?;
    Ok(())
}

/// Format `unix_seconds` as an RFC-3339 UTC timestamp (`2026-09-06T12:34:56Z`).
///
/// Hand-rolled rather than pulling in a date library for one string: the
/// civil-from-days conversion is Howard Hinnant's, the same algorithm every
/// such library uses, and `tests::formats_known_unix_timestamps` pins it
/// against dates computed independently.
pub fn format_rfc3339_utc(unix_seconds: i64) -> String {
    let days = unix_seconds.div_euclid(86_400);
    let secs_of_day = unix_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60,
    )
}

/// Days since the Unix epoch to a `(year, month, day)` civil date.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_spec::make_agent_from_spec;
    use crate::match_runner::play_paired_match;
    use duels_agents_api::Budget;
    use duels_core::engine;
    use rand::SeedableRng;

    fn record(a: &str, b: &str, wins: u32, losses: u32, draws: u32) -> PairwiseRecord {
        PairwiseRecord {
            agent_a: a.to_string(),
            agent_b: b.to_string(),
            wins,
            losses,
            draws,
        }
    }

    /// A complete synthetic round robin with a known strength order, so the
    /// leaderboard assembly can be tested without playing 60,000 games.
    fn synthetic_round_robin() -> Vec<PairwiseRecord> {
        // Strength order, weakest first; the win rate of the stronger side is
        // set by how far apart they are on this list.
        let order = ["phased", "alphabeta", "mcts-uct", "mcts-eval"];
        let mut out = Vec::new();
        for (i, a) in order.iter().enumerate() {
            for (j, b) in order.iter().enumerate().skip(i + 1) {
                let gap = (j - i) as u32;
                // b is stronger, so from a's perspective these are losses.
                let b_wins = 50 + 7 * gap;
                out.push(record(a, b, 100 - b_wins, b_wins, 0));
            }
        }
        out
    }

    #[test]
    /// Registration and rating stay pinned to each other — **up to
    /// [`REGISTERED_OFF_LADDER`]**, the documented list of agents that are
    /// deliberately runnable and unrated.
    ///
    /// So an agent still cannot fall off the board by accident: the only way
    /// to be registered and unrated is to be named in that constant, whose
    /// docs have to say why. And the reverse direction is unconditional — a
    /// ladder entry that nothing can construct is always a bug.
    fn the_ladder_is_exactly_the_registered_agents() {
        use crate::agent_registry::KNOWN_AGENTS;
        let ladder: BTreeSet<&str> = LADDER.iter().map(|e| e.agent).collect();
        let known: BTreeSet<&str> = KNOWN_AGENTS.iter().copied().collect();
        let off: BTreeSet<&str> = REGISTERED_OFF_LADDER.iter().copied().collect();

        // Nothing is both rated and declared unrated.
        assert!(
            ladder.is_disjoint(&off),
            "an agent is on the ladder and in REGISTERED_OFF_LADDER: {:?}",
            &ladder & &off
        );
        // Every deliberately-unrated agent is really registered, so the
        // exception list cannot accumulate dead names.
        assert!(
            off.is_subset(&known),
            "REGISTERED_OFF_LADDER names an unregistered agent: {:?}",
            &off - &known
        );
        // ...and the exception list is exactly the difference.
        assert_eq!(
            &known - &ladder,
            off,
            "a registered agent is neither on the leaderboard nor documented in \
             REGISTERED_OFF_LADDER — read that constant's docs before adding it there"
        );
        assert!(
            ladder.is_subset(&known),
            "a ladder agent is not registered: {:?}",
            &ladder - &known
        );
    }

    #[test]
    fn every_ladder_agent_can_actually_be_constructed() {
        for entry in LADDER {
            let agent = match make_agent_from_spec(entry.agent, 1) {
                Ok(a) => a,
                Err(e) => panic!(
                    "{} is on the ladder but not constructible: {e}",
                    entry.agent
                ),
            };
            assert_eq!(agent.spec().name, entry.agent);
        }
    }

    #[test]
    fn the_champion_is_on_the_ladder_at_its_ladder_budget() {
        let entry = LADDER
            .iter()
            .find(|e| e.agent == CHAMPION.agent)
            .expect("the champion must be a tracked agent");
        assert_eq!(entry.budget, CHAMPION.budget);
    }

    #[test]
    fn the_anchor_is_on_the_ladder() {
        assert!(LADDER.iter().any(|e| e.agent == ANCHOR_AGENT));
    }

    #[test]
    fn pairings_are_every_unordered_pair_exactly_once() {
        let p = pairings();
        let n = LADDER.len();
        assert_eq!(p.len(), n * (n - 1) / 2);
        assert_eq!(p.len(), 6);
        let unique: BTreeSet<(&str, &str)> = p.iter().map(|&(a, b)| unordered(a, b)).collect();
        assert_eq!(unique.len(), p.len(), "no pairing should repeat");
        assert!(p.iter().all(|&(a, b)| a != b), "no self-play pairings");
    }

    /// The load-bearing claim behind running the whole round robin at one
    /// budget: the 1-ply tier — `phased` alone, since the floor agents were
    /// retired — plays identically at `Nodes(1)` and `Nodes(2000)`. Checked by
    /// driving real games and comparing the chosen action at every decision,
    /// not by reading the `_budget` parameter name. Still written as a loop
    /// over the `nodes:1` entries so a future 1-ply agent is covered the
    /// moment it joins [`LADDER`].
    #[test]
    fn one_ply_agents_ignore_their_budget() {
        let one_ply: Vec<&LadderEntry> = LADDER.iter().filter(|e| e.budget == "nodes:1").collect();
        assert!(
            !one_ply.is_empty(),
            "no `nodes:1` ladder entry left for this test to cover — if the \
             1-ply tier is gone, the module docs' one-budget argument needs \
             rewriting, not this filter loosening"
        );
        for entry in one_ply {
            for seed in [1u64, 2, 3] {
                let mut cheap = make_agent_from_spec(entry.agent, seed).unwrap();
                let mut rich = make_agent_from_spec(entry.agent, seed).unwrap();
                let mut state = engine::new_game(seed);
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let mut decisions = 0;
                while !state.is_over() {
                    let legal = engine::legal_actions(&state);
                    if legal.is_empty() {
                        break;
                    }
                    let obs = state.observation();
                    let a = cheap.choose(&obs, &legal, Budget::Nodes(1));
                    let b = rich.choose(&obs, &legal, Budget::Nodes(2000));
                    assert_eq!(
                        a, b,
                        "{} chose differently at Nodes(1) vs Nodes(2000) on seed {seed}",
                        entry.agent
                    );
                    engine::apply(&mut state, a, &mut rng).unwrap();
                    decisions += 1;
                }
                assert!(decisions > 0, "{} made no decisions", entry.agent);
            }
        }
    }

    #[test]
    fn pairing_is_read_back_off_the_records() {
        let records = play_paired_match("phased", "alphabeta", &[1, 2], Budget::Nodes(1)).unwrap();
        let (a, b) = pairing_of(&records).unwrap();
        assert_eq!((a.as_str(), b.as_str()), ("phased", "alphabeta"));
    }

    #[test]
    fn pairing_of_rejects_self_play_and_empty_records() {
        let records = play_paired_match("phased", "phased", &[1], Budget::Nodes(1)).unwrap();
        assert!(pairing_of(&records).unwrap_err().contains("self-play"));
        assert!(pairing_of(&[]).is_err());
    }

    #[test]
    fn pairwise_record_matches_the_tally_of_the_games_behind_it() {
        let records =
            play_paired_match("phased", "alphabeta", &[1, 2, 3], Budget::Nodes(1)).unwrap();
        let file = ResultsFile::from_records(&records);
        let pr = pairwise_record_of(&file).unwrap();
        let t = tally(&records);
        assert_eq!(
            (pr.wins, pr.losses, pr.draws),
            (t.a_wins, t.b_wins, t.draws)
        );
        assert_eq!(pr.wins + pr.losses + pr.draws, 6);
    }

    #[test]
    fn a_complete_round_robin_builds_and_ranks_strongest_first() {
        let board = build(&synthetic_round_robin(), "2026-09-06T00:00:00Z", "abc1234").unwrap();
        assert_eq!(board.schema, SCHEMA_VERSION);
        assert_eq!(board.rows.len(), LADDER.len());
        assert_eq!(board.pairings.len(), 6);
        assert_eq!(board.total_games, 6 * 100);
        assert!(board.converged);

        let order: Vec<&str> = board.rows.iter().map(|r| r.agent.as_str()).collect();
        assert_eq!(
            order,
            vec!["mcts-eval", "mcts-uct", "alphabeta", "phased"],
            "the synthetic ladder's order should come straight back out"
        );
        assert_eq!(board.rows[0].rank, 1);
        // The champion is a hand-maintained constant, not "whoever is top of
        // this table" — see the module docs; the two coincide here because
        // `CHAMPION` was moved to `mcts-eval` once it measured strongest, not
        // because `champion` is derived from `rank`. `champion` is computed
        // by matching `CHAMPION.agent` against each row's own agent name
        // (see `champion: r.agent == CHAMPION.agent` above) — a future ladder
        // shuffle that outranked `mcts-eval` again would immediately show the
        // two diverge, without this test needing to change.
        assert!(board.rows[0].champion, "the top row should be the champion");
        assert!(
            board
                .rows
                .iter()
                .find(|r| r.agent == CHAMPION.agent)
                .unwrap()
                .champion
        );
        assert!(board.rows.iter().filter(|r| r.champion).count() == 1);

        // The anchor is pinned exactly where `ANCHOR_ELO` says. It is
        // `mcts-uct`, second of four here rather than near the bottom as
        // `greedy` was, so unlike before some ratings come out *below* 1000 —
        // see `ANCHOR_AGENT`'s docs for why the pin moved there.
        let anchor = board.rows.iter().find(|r| r.agent == ANCHOR_AGENT).unwrap();
        assert_eq!(anchor.elo, ANCHOR_ELO);
        assert!(
            board.rows.iter().any(|r| r.elo < ANCHOR_ELO),
            "an anchor above the bottom of the ladder should leave weaker \
             agents below it"
        );

        // Each agent plays 3 opponents x 100 games.
        for row in &board.rows {
            assert_eq!(row.games, 300, "{} played the wrong number", row.agent);
            assert_eq!(row.wins + row.losses + row.draws, row.games);
        }
        // The budget label follows the ladder, not the run.
        assert_eq!(
            board
                .rows
                .iter()
                .find(|r| r.agent == "mcts-uct")
                .unwrap()
                .budget,
            "nodes:2000"
        );
        assert_eq!(
            board
                .rows
                .iter()
                .find(|r| r.agent == "phased")
                .unwrap()
                .budget,
            "nodes:1"
        );
    }

    #[test]
    fn an_incomplete_round_robin_is_rejected_rather_than_silently_fitted() {
        let mut records = synthetic_round_robin();
        records.pop();
        let err = build(&records, "t", "c").unwrap_err();
        assert!(err.contains("incomplete"), "unexpected: {err}");
        assert!(err.contains("1 of 6"), "should say what is missing: {err}");
    }

    #[test]
    fn a_duplicated_pairing_is_rejected() {
        let mut records = synthetic_round_robin();
        let dup = records[0].clone();
        // Same pairing, sides swapped: still the same unordered pairing.
        records.push(record(&dup.agent_b, &dup.agent_a, 1, 1, 0));
        let err = build(&records, "t", "c").unwrap_err();
        assert!(err.contains("more than once"), "unexpected: {err}");
    }

    #[test]
    fn an_agent_not_on_the_ladder_is_rejected() {
        let mut records = synthetic_round_robin();
        records.push(record("phased", "some-experiment", 5, 5, 0));
        let err = build(&records, "t", "c").unwrap_err();
        assert!(err.contains("some-experiment"), "unexpected: {err}");
    }

    #[test]
    fn markdown_renders_every_row_and_pairing() {
        let board = build(&synthetic_round_robin(), "2026-09-06T00:00:00Z", "abc1234").unwrap();
        let md = render_markdown(&board);
        for row in &board.rows {
            assert!(
                md.contains(&row.agent),
                "{} missing from markdown",
                row.agent
            );
        }
        assert!(md.contains("(champion)"));
        assert!(md.contains("2026-09-06T00:00:00Z"));
        assert!(md.contains("abc1234"));
        assert!(md.contains("| anchor |"), "the anchor row should say so");
        // One header row + one row per agent, and the head-to-head table.
        assert_eq!(md.matches("\n| 1 |").count(), 1);
        for p in &board.pairings {
            assert!(md.contains(&format!("{} vs {}", p.agent_a, p.agent_b)));
        }
    }

    #[test]
    fn leaderboard_round_trips_through_json_and_writes_both_files() {
        let dir = std::env::temp_dir().join(format!(
            "duels-arena-leaderboard-{}-{}",
            std::process::id(),
            "round_trip"
        ));
        let board = build(&synthetic_round_robin(), "2026-09-06T00:00:00Z", "abc1234").unwrap();
        let json_path = dir.join("leaderboard.json");
        let md_path = dir.join("leaderboard.md");
        write(&board, &json_path, &md_path).unwrap();

        let back: Leaderboard =
            serde_json::from_str(&fs::read_to_string(&json_path).unwrap()).unwrap();

        // Everything discrete must come back untouched...
        assert_eq!(back.schema, board.schema);
        assert_eq!(back.generated_at, board.generated_at);
        assert_eq!(back.commit, board.commit);
        assert_eq!(back.budget, board.budget);
        assert_eq!(back.anchor_agent, board.anchor_agent);
        assert_eq!(back.champion_agent, board.champion_agent);
        assert_eq!(back.champion_budget, board.champion_budget);
        assert_eq!(back.pairings, board.pairings);
        assert_eq!(back.total_games, board.total_games);
        assert_eq!(back.converged, board.converged);
        assert_eq!(back.rows.len(), board.rows.len());

        // ...and the ratings to within the last ULP or so. Not asserted
        // bit-for-bit: JSON's decimal round trip is not guaranteed to preserve
        // the final bit of an f64, and a leaderboard reported to whole Elo
        // points has no use for it. (The determinization-invariance tests
        // elsewhere in this repo *do* compare `to_bits()`; those are checking
        // that two computations agree, which is a different question from
        // whether a decimal file preserves a float exactly.)
        for (got, want) in back.rows.iter().zip(board.rows.iter()) {
            assert_eq!(got.rank, want.rank);
            assert_eq!(got.agent, want.agent);
            assert_eq!(got.budget, want.budget);
            assert_eq!(got.champion, want.champion);
            assert_eq!(
                (got.games, got.wins, got.losses, got.draws),
                (want.games, want.wins, want.losses, want.draws)
            );
            for (a, b) in [
                (got.elo, want.elo),
                (got.elo_ci_low, want.elo_ci_low),
                (got.elo_ci_high, want.elo_ci_high),
            ] {
                assert!((a - b).abs() < 1e-9, "{a} != {b} for {}", got.agent);
            }
        }
        assert_eq!(
            fs::read_to_string(&md_path).unwrap(),
            render_markdown(&board)
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// The nightly workflow's real shape: a directory of per-pairing results
    /// files (nested, as `actions/download-artifact` leaves them) reduced to
    /// the head-to-head records the fit consumes.
    #[test]
    fn collects_pairwise_records_from_a_nested_results_directory() {
        use crate::results_io::write_results;

        let dir = std::env::temp_dir().join(format!(
            "duels-arena-leaderboard-{}-{}",
            std::process::id(),
            "collect"
        ));
        let _ = fs::remove_dir_all(&dir);

        let one = play_paired_match("phased", "alphabeta", &[1, 2], Budget::Nodes(1)).unwrap();
        let two = play_paired_match("phased", "mcts-uct", &[1, 2], Budget::Nodes(1)).unwrap();
        write_results(
            &dir.join("pairing-0")
                .join(pairing_results_filename("phased", "alphabeta")),
            &one,
        )
        .unwrap();
        write_results(
            &dir.join("pairing-1")
                .join(pairing_results_filename("phased", "mcts-uct")),
            &two,
        )
        .unwrap();

        let records = collect_pairwise_records(&dir).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].agent_a, "phased");
        assert_eq!(records[0].agent_b, "alphabeta");
        assert_eq!(records[1].agent_b, "mcts-uct");
        assert_eq!(records[0].wins + records[0].losses + records[0].draws, 4);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn collecting_from_an_empty_directory_is_an_error_not_an_empty_leaderboard() {
        let dir = std::env::temp_dir().join(format!(
            "duels-arena-leaderboard-{}-{}",
            std::process::id(),
            "empty"
        ));
        fs::create_dir_all(&dir).unwrap();
        assert!(collect_pairwise_records(&dir)
            .unwrap_err()
            .contains("no *.json"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pairing_filenames_are_distinct_for_every_pairing() {
        let names: BTreeSet<String> = pairings()
            .into_iter()
            .map(|(a, b)| pairing_results_filename(a, b))
            .collect();
        assert_eq!(names.len(), pairings().len());
    }

    #[test]
    fn formats_known_unix_timestamps() {
        assert_eq!(format_rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_rfc3339_utc(1), "1970-01-01T00:00:01Z");
        // 2000-03-01, just past a leap day in a leap century.
        assert_eq!(format_rfc3339_utc(951_868_800), "2000-03-01T00:00:00Z");
        // 2026-09-06T12:34:56Z.
        assert_eq!(format_rfc3339_utc(1_788_698_096), "2026-09-06T12:34:56Z");
        // 1900 was not a leap year: 1900-03-01.
        assert_eq!(format_rfc3339_utc(-2_203_891_200), "1900-03-01T00:00:00Z");
    }
}
