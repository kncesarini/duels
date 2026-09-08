//! Generates a **search-derived value corpus** from `mcts-eval` self-play:
//! per decision, the win probability the search itself backed up at its root,
//! together with enough information to reconstruct the exact position it was
//! looking at.
//!
//! This is training-data infrastructure, not an agent change. Nothing here
//! touches any agent's behaviour, any `Config::default`, or any existing test;
//! the one thing it needed from `mcts-eval` was a read-only accessor
//! (`MctsEvalAgent::last_root`) for a verdict the search already computed and
//! used to be dropped.
//!
//! ```text
//! # a small sample, with the built-in consistency check
//! cargo run --release -p duels-arena --example value_corpus -- \
//!     --games 200 --seed 1 --out arena/corpus/sample.jsonl
//! cargo run --release -p duels-arena --example value_corpus -- \
//!     --verify arena/corpus/sample.jsonl
//!
//! # the real run
//! cargo run --release -p duels-arena --example value_corpus -- \
//!     --games 100000 --seed 1 --budget nodes:2000 \
//!     --out arena/corpus/mcts-eval-nodes2000.jsonl
//! ```
//!
//! `arena/corpus/` is gitignored, following `arena/results/`: a corpus is a
//! regenerable data artifact, and only this generator belongs in the
//! repository.
//!
//! **Seed hygiene.** A game's seed is its identity here, so a training corpus
//! and anything used to check a model fitted on it must not share seeds. The
//! corpus generated on this machine used `--seed 1 --games 100000`
//! (seeds `1..=100000`); the two sanity-check samples deliberately sit far
//! outside that at `--seed 800001` and `--seed 900001`, which makes them a
//! ready-made disjoint holdout rather than only a smoke test.
//!
//! # Why the corpus is `(seed, actions)` and not a pile of positions
//!
//! `docs/rules-spec.md`'s **R-108** says the engine is deterministic given
//! `(seed, actions)`. A game therefore *is* its seed plus its action list, and
//! serializing a `GameState` per decision would store, at 256 bytes a copy,
//! something a replay reconstructs from about 40 small JSON objects. Replay is
//! also very cheap next to the thing that generated the corpus: `apply` plus
//! `legal_actions` is about 240 ns (the rules spec's Performance table), so
//! re-deriving a whole 40-ply game costs ~10 µs against the ~1.3 s of search
//! that produced its labels — five orders of magnitude.
//!
//! The one thing replay *cannot* recover is the search's own verdict, because
//! recovering it means re-running the search. So that — and only that — is
//! what is stored per decision.
//!
//! ## The exact replay recipe
//!
//! ```text
//! let mut state = engine::new_game(seed);
//! let mut rng   = StdRng::seed_from_u64(seed ^ ENGINE_RNG_SALT);
//! for action in actions { engine::apply(&mut state, action, &mut rng)?; }
//! ```
//!
//! `ENGINE_RNG_SALT` (`0x9E37_79B9_7F4A_7C15`) is the convention
//! `duels-server`'s `Room::new` and `duels-arena`'s `match_runner` both use;
//! the engine consumes that RNG only for The Great Library's token draw, but a
//! game containing one will not replay without it. The manifest records the
//! salt so a reader never has to know this from the source.
//!
//! # The format
//!
//! Two files per run: `<out>` (JSON Lines, one game per line) and
//! `<out>.manifest.json` (one object describing the run).
//!
//! A game line:
//!
//! ```json
//! {
//!   "seed": 1,
//!   "moves": 43,
//!   "result": { "Win": { "winner": "One", "kind": "CivilianVictory" } },
//!   "actions": [ { "type": "PickWonder", "wonder": "the_appian_way" }, ... ],
//!   "decisions": [
//!     { "ply": 0, "mover": "One", "value": 0.514, "visits": 2000,
//!       "chosen": 2, "policy": [131, 78, 1487, 304] },
//!     ...
//!   ]
//! }
//! ```
//!
//! - `actions` is the replay list, in play order; `actions.len() == moves`.
//! - `decisions` covers the plies that were actually **searched**, which is a
//!   subset of the plies: a forced move (one legal action) is returned by
//!   `Agent::choose` without a search, so there is no verdict to record and
//!   the ply is simply absent. `ply` indexes into `actions`, so the gaps are
//!   explicit rather than implied by position.
//! - `value` is the root's backed-up mean **from `Player::One`'s
//!   perspective**, matching the value convention `mcts-eval`'s tree module
//!   documents. `mover` says whose decision it was; flip with `1 - value` for
//!   that player's own win probability. Storing one convention rather than
//!   the mover's avoids a whole class of sign bug in the consumer.
//! - `chosen` and `policy` are indexed against **`engine::legal_actions` at
//!   that ply**, not against any order internal to the agent: `policy[i]` is
//!   the root visit count of `legal_actions(state)[i]`, and
//!   `legal_actions(state)[chosen] == actions[ply]`. That last identity is
//!   what `--verify` checks, and it is why the indices are safe to store: a
//!   corpus generated against a different `legal_actions` ordering fails the
//!   check loudly instead of silently mislabelling every policy target.
//! - `policy` is omitted (`null`) under `--no-policy`.
//!
//! # Is the label any good? (`--verify`)
//!
//! `--verify` replays every game from its `(seed, actions)` key, asserts every
//! claim the file makes about it (see [`verify`]), and then reports the
//! recorded values against what actually happened. Over the full 100,000-game
//! corpus generated on this machine:
//!
//! ```text
//! replayed 100000 games, 6722636 labelled decisions — every game reproduced
//! pearson r(root value, eventual outcome) = 0.5591   brier = 0.1731
//!
//!   predicted        n     mean outcome
//!   0.0-0.1     361271          0.022
//!   0.1-0.2     458101          0.096
//!   0.2-0.3     588864          0.184
//!   0.3-0.4     752357          0.301
//!   0.4-0.5    1123542          0.428
//!   0.5-0.6    1108210          0.579
//!   0.6-0.7     819122          0.705
//!   0.7-0.8     639241          0.821
//!   0.8-0.9     502281          0.903
//!   0.9-1.0     369647          0.979
//! ```
//!
//! That whole pass — 100,000 games re-derived from their seeds and every
//! assertion checked — takes **6.9 seconds**, against the 204.5 minutes of
//! search that generated the labels. The `(seed, actions)` format's central
//! bet, measured.
//!
//! Monotone across all ten buckets and close to the diagonal, which is a
//! stronger result than the roadmap's "correlate at least weakly" bar: the
//! search's root value is a genuinely *calibrated* win probability, not merely
//! a correlated score. The residual is a mild **under**confidence in the
//! middle of the range (the 0.6-0.7 bucket really wins 0.705, the 0.7-0.8
//! bucket 0.821), i.e. the empirical curve is slightly steeper than the
//! diagonal, and the effect shrinks towards the tails. A
//! consumer that wants a maximally sharp target may want to sharpen for that;
//! a consumer fitting `duels-eval` against the search should not, because the
//! search's own value is the thing being fitted.
//!
//! # What `value` actually contains — read this before fitting anything
//!
//! At `Config::default` an `mcts-eval` leaf is **half a playout and half
//! `duels_eval::win_probability`**. The root value is an average of those
//! leaves, so it is a search-derived target that already has `duels-eval`
//! inside it at roughly the blend weight. Fitting `duels-eval`'s own weights
//! against this corpus is therefore *partly* self-referential, and a fit that
//! reproduces today's weights well is not by itself evidence of anything.
//!
//! That is a property of the target the roadmap asked for, not a bug in the
//! collector, and there is a clean way to get an uncontaminated one from the
//! same tool: `--params base=rollout` puts the search on a pure-playout leaf
//! (`Config::rollout_base`, which is `mcts-uct`'s search node for node), whose
//! root value contains no evaluation at all. Generating both and comparing the
//! fits is the honest version of the experiment.
//!
//! # Cost, measured
//!
//! Apple Silicon development machine, 14 logical cores, `nodes:2000`, from the
//! actual 100,000-game run:
//!
//! | quantity | measured |
//! | --- | --- |
//! | searched decisions per game | 67.2 (plus 3.5 forced plies, unlabelled) |
//! | simulations per game | ~134,000 |
//! | throughput | 8.2 games/s on a quiet machine (≈11× effective parallelism) |
//! | corpus size | **9.9 KB per game** with `policy`, ~4 KB without |
//! | **100,000 games** | **204.5 min wall, ~28 core-hours, 985 MiB** |
//!
//! An order of magnitude more expensive than a reading of the rules spec's
//! `full_playout` row ("~47 k games/s") would suggest, and the difference is
//! the whole point: that row is *random* play with no search, while every
//! decision here pays for 2000 `mcts-eval` simulations. Budget a corpus in
//! core-hours, not core-seconds.
//!
//! Throughput is also sensitive to what else the machine is doing — the first
//! batches of that run measured 5.1-5.4 games/s while a `cargo clippy` and a
//! `cargo test` shared the box, and settled at 8.2 once they finished. Same
//! caveat as the crate docs' "Benchmarking on a quiet machine" note, for the
//! same reason.
//!
//! Worth knowing before choosing a size: rows within one game are strongly
//! correlated (they are the same game seen from successive plies), so the
//! effective sample size tracks the *game* count far more closely than the row
//! count. 100,000 games is generous provisioning for a neural value head and
//! extravagant for a ~25-parameter linear fit, which 512 games already gives
//! 34,000 rows for. Games are independent, so `--threads` scales; `--out` is
//! written in seed order regardless of scheduling.
//!
//! One consequence of that correlation is worth stating for whoever fits
//! against this: **split train/validation by game, never by row.** Two rows
//! from the same game share a seed, a deal and most of a board, so a row-wise
//! split leaks nearly every validation position into training. Splitting on
//! `seed` is the cheap, correct thing, and the seed is on every line.
//!
//! A run rewrites its manifest after every batch and flushes as it goes, so an
//! interrupted run leaves a shorter but entirely valid corpus rather than an
//! unlabelled one. That was checked live rather than assumed: the partial file
//! was `--verify`ed mid-run at 14,592 games and replayed clean.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use duels_agent_mcts_eval::{Config as MctsEvalConfig, MctsEvalAgent};
use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_arena::agent_spec::parse_mcts_eval_config;
use duels_arena::match_runner::parse_budget;
use duels_core::{engine, Action, GameResult, Player};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Salt for the engine's own mid-game RNG (The Great Library's token draw),
/// matching `duels-arena::match_runner` and `duels-server::room::Room::new`.
/// A replay must use it or a game containing a Great Library will diverge.
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Per-seat agent RNG salts, so the two sides of a self-play game make
/// independent random choices from the same game seed. Recorded in the
/// manifest because they are what makes a *regeneration* reproduce a corpus
/// (a *replay* needs only the seed and the actions).
const SEAT_ONE_SALT: u64 = 0xA011_7A9E_5B21_0001;
const SEAT_TWO_SALT: u64 = 0xB022_8C3F_6D42_0002;

/// How many seeds one rayon batch covers. Bounds peak memory and gives the
/// progress line something to report; has no effect on the output.
const BATCH: usize = 256;

/// One searched decision: what the search concluded, and where it was.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Decision {
    /// Index into the game's `actions`, so a skipped (forced) ply is visible
    /// as a gap rather than inferred.
    ply: u32,
    /// Whose decision it was.
    mover: Player,
    /// The root's backed-up win probability, from `Player::One`'s perspective.
    value: f64,
    /// Root visits the search actually spent (close to, not exactly, the node
    /// budget).
    visits: u64,
    /// Index into `engine::legal_actions` at this ply of the action played.
    chosen: usize,
    /// Root visit count per legal action, in `engine::legal_actions` order.
    #[serde(skip_serializing_if = "Option::is_none")]
    policy: Option<Vec<u32>>,
}

/// One self-play game: the replay key, the outcome, and the labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GameLine {
    seed: u64,
    moves: u32,
    result: GameResult,
    actions: Vec<Action>,
    decisions: Vec<Decision>,
}

/// The sidecar that makes a corpus file self-describing.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    kind: String,
    version: u32,
    generated_by: String,
    /// The agent that played both seats, including the whole `duels-eval`
    /// configuration its leaves were scored against — `mcts-eval` tracks
    /// `duels_eval::Config::default()` live, so this string is the only thing
    /// that makes two corpora from either side of a `duels-eval` round
    /// distinguishable.
    agent: AgentSpec,
    budget: String,
    seed_first: u64,
    seed_last: u64,
    games: u64,
    /// Searched plies, i.e. total `decisions` across every line — the corpus's
    /// real row count.
    decisions: u64,
    /// Plies that were forced and so carry no label.
    forced_plies: u64,
    policy_recorded: bool,
    /// Replay needs this; see the module docs' recipe.
    engine_rng_salt: String,
    /// Regeneration needs these.
    seat_salts: [String; 2],
    wall_time_s: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let has = |name: &str| args.iter().any(|a| a == name);

    if has("--help") || args.is_empty() {
        eprintln!(
            "value_corpus: generate an mcts-eval search-derived value corpus\n\
             \n\
             generate:  --out <path> [--games N] [--seed S] [--budget nodes:2000]\n\
             \x20          [--params <mcts-eval spec params>] [--threads N] [--no-policy]\n\
             verify:    --verify <path> [--sample N]\n"
        );
        return;
    }

    if let Some(path) = flag("--verify") {
        let sample: usize = flag("--sample")
            .map(|s| s.parse().expect("--sample must be a number"))
            .unwrap_or(usize::MAX);
        verify(Path::new(&path), sample);
        return;
    }

    let out = PathBuf::from(flag("--out").expect("--out <path> is required to generate"));
    let games: u64 = flag("--games")
        .map(|s| s.parse().expect("--games must be a number"))
        .unwrap_or(200);
    let seed0: u64 = flag("--seed")
        .map(|s| s.parse().expect("--seed must be a number"))
        .unwrap_or(1);
    let budget_str = flag("--budget").unwrap_or_else(|| "nodes:2000".to_string());
    let budget = parse_budget(&budget_str).expect("a valid --budget");
    let params = flag("--params").unwrap_or_default();
    let cfg = parse_mcts_eval_config(&params).expect("valid mcts-eval spec params");
    let policy = !has("--no-policy");
    if let Some(t) = flag("--threads") {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t.parse().expect("--threads must be a number"))
            .build_global()
            .expect("the rayon pool is configured once");
    }

    generate(&out, games, seed0, budget, &budget_str, cfg, policy);
}

/// Play one self-play game and return its line.
fn play(seed: u64, budget: Budget, cfg: MctsEvalConfig, want_policy: bool) -> GameLine {
    let mut one = MctsEvalAgent::with_config(seed ^ SEAT_ONE_SALT, cfg);
    let mut two = MctsEvalAgent::with_config(seed ^ SEAT_TWO_SALT, cfg);
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ ENGINE_RNG_SALT);

    let mut actions: Vec<Action> = Vec::with_capacity(64);
    let mut decisions: Vec<Decision> = Vec::with_capacity(48);

    loop {
        if state.is_over() {
            break;
        }
        // The same order a replay will reconstruct, and the order `chosen` and
        // `policy` are indexed against.
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let obs = state.observation();
        let mover = state.current_player();
        let agent: &mut MctsEvalAgent = match mover {
            Player::One => &mut one,
            Player::Two => &mut two,
        };
        let action = agent.choose(&obs, &legal, budget);
        assert!(
            legal.contains(&action),
            "seed {seed}: agent returned an action outside `legal`: {action:?}"
        );

        // `None` exactly when the move was forced; see `last_root`.
        if let Some(root) = agent.last_root() {
            let chosen = legal
                .iter()
                .position(|&a| a == action)
                .expect("the played action is one of the legal ones");
            let policy = want_policy.then(|| {
                // The readout is in the tree's own shuffled order, so it is
                // re-indexed into `legal_actions` order here — the one order a
                // replay can reproduce.
                legal
                    .iter()
                    .map(|&a| {
                        root.policy
                            .iter()
                            .find(|&&(pa, _)| pa == a)
                            .map(|&(_, n)| n)
                            .unwrap_or(0)
                    })
                    .collect()
            });
            decisions.push(Decision {
                ply: actions.len() as u32,
                mover,
                value: root.value,
                visits: root.visits,
                chosen,
                policy,
            });
        }

        engine::apply(&mut state, action, &mut rng).expect("a legal action applies");
        actions.push(action);
    }

    GameLine {
        seed,
        moves: actions.len() as u32,
        result: state.result().expect("a finished game has a result"),
        actions,
        decisions,
    }
}

fn generate(
    out: &Path,
    games: u64,
    seed0: u64,
    budget: Budget,
    budget_str: &str,
    cfg: MctsEvalConfig,
    policy: bool,
) {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).expect("the output directory is creatable");
    }
    let file = File::create(out).expect("the output file is creatable");
    let mut w = BufWriter::with_capacity(1 << 20, file);

    let spec = MctsEvalAgent::with_config(0, cfg).spec();
    println!("agent   {} {}", spec.name, spec.version);
    println!("params  {}", spec.params);
    println!("budget  {budget_str}");
    println!("seeds   {seed0}..{}", seed0 + games);
    println!("out     {}", out.display());
    println!();

    #[allow(clippy::disallowed_methods)]
    let start = std::time::Instant::now();
    let mut total_decisions = 0u64;
    let mut total_forced = 0u64;
    let mut done = 0u64;

    let mpath = manifest_path(out);
    // Rewritten after **every batch**, not once at the end. A full run is
    // hours long, so an interrupted one has to leave a corpus that is still
    // valid and still self-describing — the games already on disk, and a
    // manifest that counts exactly those. Flushing the writer in the same
    // place is what makes the two agree.
    let write_manifest = |games_done: u64, decisions: u64, forced: u64, wall: f64| {
        let manifest = Manifest {
            kind: "duels-value-corpus".to_string(),
            version: 1,
            generated_by: "duels-arena examples/value_corpus.rs".to_string(),
            agent: spec.clone(),
            budget: budget_str.to_string(),
            seed_first: seed0,
            seed_last: seed0 + games_done.max(1) - 1,
            games: games_done,
            decisions,
            forced_plies: forced,
            policy_recorded: policy,
            engine_rng_salt: format!("{ENGINE_RNG_SALT:#018x}"),
            seat_salts: [
                format!("{SEAT_ONE_SALT:#018x}"),
                format!("{SEAT_TWO_SALT:#018x}"),
            ],
            wall_time_s: wall,
        };
        fs::write(
            &mpath,
            serde_json::to_string_pretty(&manifest).expect("the manifest serializes"),
        )
        .expect("the manifest is writable");
    };

    let mut seed = seed0;
    while seed < seed0 + games {
        let end = (seed + BATCH as u64).min(seed0 + games);
        let batch: Vec<GameLine> = (seed..end)
            .into_par_iter()
            .map(|s| play(s, budget, cfg, policy))
            .collect();
        for line in &batch {
            total_decisions += line.decisions.len() as u64;
            total_forced += u64::from(line.moves) - line.decisions.len() as u64;
            serde_json::to_writer(&mut w, line).expect("a game line serializes");
            w.write_all(b"\n").expect("the corpus file is writable");
        }
        done += batch.len() as u64;
        w.flush().expect("the corpus file flushes");
        #[allow(clippy::disallowed_methods)]
        let elapsed = start.elapsed().as_secs_f64();
        write_manifest(done, total_decisions, total_forced, elapsed);
        let rate = done as f64 / elapsed.max(1e-9);
        let eta = (games - done) as f64 / rate.max(1e-9);
        println!(
            "  {done}/{games} games  {total_decisions} decisions  \
             {rate:.1} games/s  eta {:.0} min",
            eta / 60.0
        );
        seed = end;
    }

    #[allow(clippy::disallowed_methods)]
    let wall = start.elapsed().as_secs_f64();
    write_manifest(done, total_decisions, total_forced, wall);

    let bytes = fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    println!();
    println!("wrote   {games} games, {total_decisions} decisions");
    println!("        {} ({:.1} MiB)", out.display(), bytes as f64 / 1e6);
    println!("        {}", mpath.display());
    println!(
        "took    {:.1} min wall ({:.2} s/game/core-equivalent aggregate)",
        wall / 60.0,
        wall / games as f64
    );
}

fn manifest_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_owned();
    s.push(".manifest.json");
    PathBuf::from(s)
}

/// Replay the corpus against the engine and check every claim it makes, then
/// report whether the recorded values look like win probabilities of the games
/// that actually happened.
///
/// This is the gate the roadmap's "sanity-check before the full run" asks for,
/// and it is deliberately a *replay*: it re-derives every position from
/// `(seed, actions)` and would catch a corpus whose replay key does not
/// reconstruct the game it labelled.
fn verify(path: &Path, sample: usize) {
    let mpath = manifest_path(path);
    match fs::read_to_string(&mpath) {
        Ok(s) => {
            let m: Manifest = serde_json::from_str(&s).expect("the manifest parses");
            println!("manifest {} v{}", m.kind, m.version);
            println!("  agent   {} / {}", m.agent.name, m.agent.params);
            println!("  budget  {}", m.budget);
            println!(
                "  games   {} ({} decisions, {} forced plies), policy={}",
                m.games, m.decisions, m.forced_plies, m.policy_recorded
            );
        }
        Err(e) => println!("(no manifest at {}: {e})", mpath.display()),
    }

    let reader = BufReader::new(File::open(path).expect("the corpus file is readable"));
    let mut games = 0u64;
    let mut rows = 0u64;
    // Paired (recorded value from the mover's view, actual outcome for the
    // mover) for the correlation and the calibration table.
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if games as usize >= sample {
            break;
        }
        let line = line.expect("the corpus file reads");
        if line.trim().is_empty() {
            continue;
        }
        let g: GameLine = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("line {} does not parse: {e}", i + 1));

        assert_eq!(
            g.moves as usize,
            g.actions.len(),
            "seed {}: moves disagrees with the action list",
            g.seed
        );

        // The replay, exactly as the module docs describe it.
        let mut state = engine::new_game(g.seed);
        let mut rng = StdRng::seed_from_u64(g.seed ^ ENGINE_RNG_SALT);
        let mut next_decision = 0usize;
        for (ply, &action) in g.actions.iter().enumerate() {
            let legal = engine::legal_actions(&state);
            assert!(
                !legal.is_empty(),
                "seed {}: the replay ran out of legal actions at ply {ply}",
                g.seed
            );
            let labelled = g
                .decisions
                .get(next_decision)
                .is_some_and(|d| d.ply as usize == ply);
            if labelled {
                let d = &g.decisions[next_decision];
                next_decision += 1;
                assert!(
                    legal.len() > 1,
                    "seed {}: ply {ply} is labelled but forced",
                    g.seed
                );
                assert_eq!(
                    d.mover,
                    state.current_player(),
                    "seed {}: ply {ply} names the wrong mover",
                    g.seed
                );
                // The load-bearing check: the stored indices really do index
                // this replay's `legal_actions`.
                assert!(
                    d.chosen < legal.len(),
                    "seed {}: ply {ply} chose index {} of {} legal actions",
                    g.seed,
                    d.chosen,
                    legal.len()
                );
                assert_eq!(
                    legal[d.chosen], action,
                    "seed {}: ply {ply}'s chosen index does not name the played action",
                    g.seed
                );
                if let Some(p) = &d.policy {
                    assert_eq!(
                        p.len(),
                        legal.len(),
                        "seed {}: ply {ply}'s policy has the wrong width",
                        g.seed
                    );
                    let top = p.iter().max().copied().unwrap_or(0);
                    assert_eq!(
                        p[d.chosen], top,
                        "seed {}: ply {ply} played a non-argmax action",
                        g.seed
                    );
                }
                assert!(
                    (0.0..=1.0).contains(&d.value),
                    "seed {}: ply {ply}'s value {} is not a probability",
                    g.seed,
                    d.value
                );
                assert!(
                    d.visits > 0,
                    "seed {}: ply {ply} recorded no visits",
                    g.seed
                );
                rows += 1;
            } else {
                assert_eq!(
                    legal.len(),
                    1,
                    "seed {}: ply {ply} had {} legal actions but no label",
                    g.seed,
                    legal.len()
                );
            }
            assert!(
                legal.contains(&action),
                "seed {}: ply {ply}'s action is not legal in the replay",
                g.seed
            );
            engine::apply(&mut state, action, &mut rng)
                .unwrap_or_else(|e| panic!("seed {}: replay failed at ply {ply}: {e}", g.seed));
        }
        assert_eq!(
            next_decision,
            g.decisions.len(),
            "seed {}: {} decisions were never matched to a ply",
            g.seed,
            g.decisions.len() - next_decision
        );

        let replayed = state
            .result()
            .unwrap_or_else(|| panic!("seed {}: the replay did not finish the game", g.seed));
        assert_eq!(
            replayed, g.result,
            "seed {}: the replay reached a different result",
            g.seed
        );

        // Labels against what actually happened, from each mover's own view.
        for d in &g.decisions {
            let outcome = match g.result.winner() {
                None => 0.5,
                Some(w) if w == d.mover => 1.0,
                Some(_) => 0.0,
            };
            let value = match d.mover {
                Player::One => d.value,
                Player::Two => 1.0 - d.value,
            };
            xs.push(value);
            ys.push(outcome);
        }
        games += 1;
    }

    println!();
    println!("replayed {games} games, {rows} labelled decisions — every game reproduced");
    if xs.is_empty() {
        return;
    }
    println!(
        "pearson r(root value, eventual outcome) = {:.4}   brier = {:.4}",
        pearson(&xs, &ys),
        xs.iter()
            .zip(&ys)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            / xs.len() as f64
    );
    println!();
    println!("  predicted        n     mean outcome");
    for b in 0..10 {
        let lo = b as f64 / 10.0;
        let hi = lo + 0.1;
        let picked: Vec<f64> = xs
            .iter()
            .zip(&ys)
            .filter(|(x, _)| **x >= lo && (**x < hi || (b == 9 && **x <= 1.0)))
            .map(|(_, y)| *y)
            .collect();
        if picked.is_empty() {
            println!("  {lo:.1}-{hi:.1}          0            -");
            continue;
        }
        let mean = picked.iter().sum::<f64>() / picked.len() as f64;
        println!("  {lo:.1}-{hi:.1}   {:>8}          {mean:.3}", picked.len());
    }
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx * syy).sqrt()
}
