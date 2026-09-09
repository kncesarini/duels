//! Turns a value corpus (`examples/value_corpus.rs`) into a training set for
//! `duels-value`: per labelled decision, the public-information feature vector
//! and the **victory-kind-decomposed outcome** of the game it came from.
//!
//! ```text
//! cargo run --release -p duels-arena --example feature_dump -- \
//!     --corpus arena/corpus/mcts-eval-nodes2000.jsonl \
//!     --out arena/corpus/features
//! ```
//!
//! # What a row is
//!
//! One `(seed, ply)` labelled decision from the corpus, seen from `me`'s
//! perspective, where `me` is [`Player::One`] by default (`--perspective one`)
//! — exactly the perspective `mcts-eval`'s tree evaluates every leaf from, so
//! training and inference see the same distribution. `--perspective mover`
//! uses whoever was to move; `--perspective both` writes two rows per
//! decision, one per side, which doubles the corpus and imposes the
//! antisymmetry as data.
//!
//! The label is the game's **actual final result**, as one of four mutually
//! exclusive classes for `me`:
//!
//! | label | meaning |
//! |---|---|
//! | 0 | `me` won by military supremacy |
//! | 1 | `me` won by scientific supremacy |
//! | 2 | `me` won on points (civilian victory or civilian tiebreak) |
//! | 3 | `me` lost, by any of the three |
//! | 4 | draw (equal points and equal civilian points) — the trainer treats it as half a win |
//!
//! Deliberately **not** the search's recorded root value. A model fitted to
//! the search's opinion inherits its blind spots and — as `duels-eval`'s own
//! fitting round found — loses the complementarity that makes a blended leaf
//! useful. The root value is still written, as a *comparison column*, together
//! with `duels_eval::win_probability` of the same position: those two are the
//! yardsticks a learned value has to beat on held-out games, and having them
//! on every row means the trainer can report all three on identical rows.
//!
//! # Files
//!
//! `<out>.X.i8` — `rows x NUM_FEATURES` signed bytes, row-major. Every
//! feature is an integer in `[-127, 127]` by `duels_value::features`'
//! contract, so nothing is lost.
//!
//! `<out>.rows.bin` — 12 bytes per row, little-endian:
//! `seed: u32, ply: u16, mover_is_me: u8, label: u8, search_value: f32`
//! ... followed by a second 4-byte block per row in `<out>.aux.f32`:
//! `eval_wp: f32`. (Kept separate so a reader that does not want the
//! hand-crafted comparison never has to compute it.)
//!
//! `<out>.meta.json` — feature names, widths, counts, the perspective, and
//! the corpus manifest's agent string.
//!
//! **Split by game, never by row.** Rows within a game are the same game seen
//! from successive plies; `seed` is on every row so a reader can hold out
//! whole games. `tools/train.py` does exactly that.
//!
//! # Cost
//!
//! Replay is ~10 µs a game; the features and one `duels_eval::Root` per
//! decision are a few µs each; the whole 100,000-game corpus dumps in well
//! under a minute across `rayon`.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use duels_core::scoring::VictoryKind;
use duels_core::{engine, Action, GameResult, Player};
use duels_value::{feature_names, features, NUM_FEATURES};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use serde::Deserialize;

/// Matches `value_corpus.rs` and `duels-arena::match_runner`.
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const BATCH: usize = 512;

#[derive(Debug, Clone, Deserialize)]
struct Decision {
    ply: u32,
    mover: Player,
    value: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct GameLine {
    seed: u64,
    result: GameResult,
    actions: Vec<Action>,
    decisions: Vec<Decision>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Perspective {
    One,
    Mover,
    Both,
}

/// One dumped row, before serialisation.
struct Row {
    seed: u32,
    ply: u16,
    mover_is_me: u8,
    label: u8,
    search_value: f32,
    eval_wp: f32,
    x: [i8; NUM_FEATURES],
}

fn label_for(result: GameResult, me: Player) -> u8 {
    match result {
        GameResult::Draw => 4,
        GameResult::Win { winner, kind } if winner == me => match kind {
            VictoryKind::MilitarySupremacy => 0,
            VictoryKind::ScientificSupremacy => 1,
            VictoryKind::CivilianVictory | VictoryKind::CivilianTiebreak => 2,
        },
        GameResult::Win { .. } => 3,
    }
}

fn dump_game(g: &GameLine, perspective: Perspective, with_eval: bool) -> Vec<Row> {
    let mut state = engine::new_game(g.seed);
    let mut rng = StdRng::seed_from_u64(g.seed ^ ENGINE_RNG_SALT);
    let mut rows = Vec::with_capacity(g.decisions.len() * 2);
    let mut next = 0usize;
    for (ply, &action) in g.actions.iter().enumerate() {
        if g.decisions.get(next).is_some_and(|d| d.ply as usize == ply) {
            let d = &g.decisions[next];
            next += 1;
            debug_assert_eq!(d.mover, state.current_player());
            let sides: &[Player] = match perspective {
                Perspective::One => &[Player::One],
                Perspective::Mover => std::slice::from_ref(&d.mover),
                Perspective::Both => &Player::ALL,
            };
            // One `Root` per decision, built for the mover as `phased` and
            // `mcts-eval` both do; `win_probability` is antisymmetric in
            // `me`, so one root serves both perspectives.
            let root = with_eval.then(|| {
                duels_eval::Root::new(
                    &state,
                    state.current_player(),
                    duels_eval::Config::default(),
                )
            });
            for &me in sides {
                let f = features(&state, me);
                let mut x = [0i8; NUM_FEATURES];
                for (o, &v) in x.iter_mut().zip(f.iter()) {
                    debug_assert!(v.fract() == 0.0 && (-127.0..=127.0).contains(&v));
                    *o = v as i8;
                }
                let search_value = match me {
                    Player::One => d.value,
                    Player::Two => 1.0 - d.value,
                } as f32;
                let eval_wp = root
                    .as_ref()
                    .map(|r| duels_eval::win_probability(&state, me, r) as f32)
                    .unwrap_or(f32::NAN);
                rows.push(Row {
                    seed: g.seed as u32,
                    ply: ply as u16,
                    mover_is_me: u8::from(d.mover == me),
                    label: label_for(g.result, me),
                    search_value,
                    eval_wp,
                    x,
                });
            }
        }
        engine::apply(&mut state, action, &mut rng)
            .unwrap_or_else(|e| panic!("seed {}: replay failed at ply {ply}: {e}", g.seed));
    }
    assert_eq!(
        next,
        g.decisions.len(),
        "seed {}: unmatched decisions",
        g.seed
    );
    assert_eq!(
        state.result(),
        Some(g.result),
        "seed {}: the replay reached a different result",
        g.seed
    );
    rows
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
            "feature_dump: replay a value corpus into (features, outcome) rows\n\
             \n\
             --corpus <path.jsonl>  --out <prefix>  [--games N]  [--skip N]\n\
             [--perspective one|mover|both]  [--no-eval]  [--threads N]\n"
        );
        return;
    }
    let corpus = PathBuf::from(flag("--corpus").expect("--corpus is required"));
    let out = PathBuf::from(flag("--out").expect("--out <prefix> is required"));
    let max_games: usize = flag("--games")
        .map(|s| s.parse().expect("--games must be a number"))
        .unwrap_or(usize::MAX);
    let skip: usize = flag("--skip")
        .map(|s| s.parse().expect("--skip must be a number"))
        .unwrap_or(0);
    let perspective = match flag("--perspective").as_deref().unwrap_or("one") {
        "one" => Perspective::One,
        "mover" => Perspective::Mover,
        "both" => Perspective::Both,
        other => panic!("unknown --perspective {other}"),
    };
    let with_eval = !has("--no-eval");
    if let Some(t) = flag("--threads") {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t.parse().expect("--threads must be a number"))
            .build_global()
            .expect("the rayon pool is configured once");
    }

    let agent_params = fs::read_to_string(manifest_path(&corpus))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|m| m["agent"]["params"].as_str().map(str::to_string))
        .unwrap_or_default();

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("the output directory is creatable");
        }
    }
    let mut xw = BufWriter::with_capacity(1 << 22, File::create(with_ext(&out, "X.i8")).unwrap());
    let mut rw =
        BufWriter::with_capacity(1 << 20, File::create(with_ext(&out, "rows.bin")).unwrap());
    let mut aw =
        BufWriter::with_capacity(1 << 20, File::create(with_ext(&out, "aux.f32")).unwrap());

    let reader = BufReader::new(File::open(&corpus).expect("the corpus is readable"));
    let mut lines = reader.lines().skip(skip);
    let mut games = 0usize;
    let mut rows = 0u64;
    let mut label_counts = [0u64; 5];
    let mut first_seed = u64::MAX;
    let mut last_seed = 0u64;
    // Baselines on the same rows: Brier of the two comparison columns against
    // the win/loss outcome (`me` wins = 1, draw = 0.5).
    let (mut brier_search, mut brier_eval, mut n_eval) = (0.0f64, 0.0f64, 0u64);

    #[allow(clippy::disallowed_methods)]
    let start = std::time::Instant::now();
    loop {
        let mut batch: Vec<GameLine> = Vec::with_capacity(BATCH);
        while batch.len() < BATCH && games + batch.len() < max_games {
            match lines.next() {
                Some(line) => {
                    let line = line.expect("the corpus reads");
                    if line.trim().is_empty() {
                        continue;
                    }
                    batch.push(serde_json::from_str(&line).expect("a game line parses"));
                }
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        let dumped: Vec<Vec<Row>> = batch
            .par_iter()
            .map(|g| dump_game(g, perspective, with_eval))
            .collect();
        for (g, game_rows) in batch.iter().zip(&dumped) {
            first_seed = first_seed.min(g.seed);
            last_seed = last_seed.max(g.seed);
            for r in game_rows {
                // SAFETY-free reinterpretation: i8 -> u8 bytes.
                let bytes: Vec<u8> = r.x.iter().map(|&v| v as u8).collect();
                xw.write_all(&bytes).unwrap();
                rw.write_all(&r.seed.to_le_bytes()).unwrap();
                rw.write_all(&r.ply.to_le_bytes()).unwrap();
                rw.write_all(&[r.mover_is_me, r.label]).unwrap();
                rw.write_all(&r.search_value.to_le_bytes()).unwrap();
                aw.write_all(&r.eval_wp.to_le_bytes()).unwrap();
                label_counts[r.label as usize] += 1;
                let y = match r.label {
                    0..=2 => 1.0,
                    3 => 0.0,
                    _ => 0.5,
                };
                brier_search += (f64::from(r.search_value) - y).powi(2);
                if r.eval_wp.is_finite() {
                    brier_eval += (f64::from(r.eval_wp) - y).powi(2);
                    n_eval += 1;
                }
                rows += 1;
            }
        }
        games += batch.len();
        #[allow(clippy::disallowed_methods)]
        let el = start.elapsed().as_secs_f64();
        eprintln!("  {games} games, {rows} rows, {el:.1}s");
        if games >= max_games {
            break;
        }
    }
    xw.flush().unwrap();
    rw.flush().unwrap();
    aw.flush().unwrap();

    let meta = serde_json::json!({
        "kind": "duels-value-features",
        "version": 1,
        "corpus": corpus.display().to_string(),
        "corpus_agent_params": agent_params,
        "perspective": format!("{perspective:?}").to_lowercase(),
        "games": games,
        "rows": rows,
        "seed_first": first_seed,
        "seed_last": last_seed,
        "num_features": NUM_FEATURES,
        "feature_names": feature_names(),
        "row_record": "seed:u32 ply:u16 mover_is_me:u8 label:u8 search_value:f32 (12 bytes, LE)",
        "aux_record": "eval_wp:f32 (duels_eval::win_probability at Config::default, NaN if --no-eval)",
        "labels": ["military_win", "science_win", "civilian_win", "loss", "draw"],
        "label_counts": label_counts,
        "baseline_brier": {
            "search_root_value": brier_search / rows.max(1) as f64,
            "duels_eval_win_probability": if n_eval > 0 { brier_eval / n_eval as f64 } else { f64::NAN },
        },
    });
    fs::write(
        with_ext(&out, "meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();

    println!();
    println!("wrote {rows} rows from {games} games (seeds {first_seed}..={last_seed})");
    println!(
        "labels  military {}  science {}  civilian {}  loss {}  draw {}",
        label_counts[0], label_counts[1], label_counts[2], label_counts[3], label_counts[4]
    );
    println!(
        "brier on these rows: search root value {:.4}, duels-eval win_probability {:.4}",
        brier_search / rows.max(1) as f64,
        if n_eval > 0 {
            brier_eval / n_eval as f64
        } else {
            f64::NAN
        }
    );
    println!("{}", with_ext(&out, "meta.json").display());
}

fn with_ext(prefix: &Path, ext: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn manifest_path(corpus: &Path) -> PathBuf {
    let mut s = corpus.as_os_str().to_owned();
    s.push(".manifest.json");
    PathBuf::from(s)
}
