//! Turns the `mcts-eval` self-play corpus into a **training matrix** for
//! `duels-value`: one row per labelled decision per perspective, holding
//! [`duels_value::features`] and the four-way victory-kind-and-loss outcome
//! label read off the game's actual `GameResult`.
//!
//! This is training-data infrastructure, exactly like `value_corpus.rs` next
//! to it. It changes no agent's behaviour, no `Config::default`, and no
//! existing test.
//!
//! ```text
//! # a small sample first, with the built-in consistency report
//! cargo run --release -p duels-arena --example feature_dump -- \
//!     --corpus arena/corpus/mcts-eval-nodes2000.jsonl \
//!     --out arena/corpus/features-v1.bin --games 500
//!
//! # the run that produced the shipped weights: the whole 100,000-game corpus,
//! # every second labelled decision, both perspectives. 6,772,568 rows,
//! # 5.8 GiB, 9.2 s wall on 14 cores.
//! cargo run --release -p duels-arena --example feature_dump -- \
//!     --corpus arena/corpus/mcts-eval-nodes2000.jsonl \
//!     --out arena/corpus/features-v1.bin --games 100000 --stride 2
//! ```
//!
//! Replay is cheap next to the search that produced the corpus — `apply` plus
//! `legal_actions` is about 240 ns (`docs/rules-spec.md`'s Performance table),
//! so the whole pass runs at about **11,000 games/s** on 14 cores against the
//! 204.5 minutes of `mcts-eval` search that generated the labels. Feature
//! extraction, not replay, is the bulk of that.
//!
//! # Labels are outcomes, never the search's own value
//!
//! The corpus records both the eventual [`GameResult`] and the win probability
//! `mcts-eval` backed up at its root. The **label** written here is derived
//! from the result, via [`duels_value::Outcome::of`]. The search's value is
//! written alongside it as a *column*, never as the target: it is there so a
//! trained model can be scored against the incumbent leaf signal on identical
//! rows, which is the comparison that actually matters.
//!
//! A separate investigation in this repository measured that fitting an
//! evaluation against the search's own opinion destroys the complementarity
//! that makes a hand-crafted evaluation useful inside a blend. There is no
//! reason to assume a learned model escapes that failure mode, so this tool
//! does not offer the option.
//!
//! # Both perspectives, and why that is not just data doubling
//!
//! Every labelled decision is written **twice**: once from `Player::One`'s
//! point of view and once from `Player::Two`'s. Since
//! `duels_value::features` carries no seat bit and relativises every
//! player-specific quantity, the two rows are the same position read from
//! opposite sides.
//!
//! That would be pure redundancy for a single-scalar win/loss model. It is
//! *not* redundant for the four-way head, because the `Loss` class is not
//! decomposed: a decision in a game Player One won militarily contributes one
//! `military_win` row and one `loss` row. Writing both perspectives means
//! every game's decisions contribute a positive example to whichever win-kind
//! head applies, doubling the positive examples for the rare science head in
//! particular, and it makes the class distribution exactly symmetric instead
//! of carrying the first-player advantage as a label bias.
//!
//! # `--stride`
//!
//! Rows inside one game are strongly correlated — they are the same game seen
//! from successive plies — so the effective sample size tracks the *game*
//! count far more closely than the row count (`value_corpus.rs` says the same
//! thing about sizing a corpus). `--stride 2` keeps every second labelled
//! decision, halving the file for very little information, and `--games` is
//! the knob that actually decides how much data there is.
//!
//! # The output format
//!
//! Little-endian throughout. A 32-byte header:
//!
//! ```text
//! offset  size  field
//!      0     4  magic, b"DVFD"
//!      4     4  u32 version = 1 or 2
//!      8     4  u32 num_features
//!     12     4  u32 num_outcomes
//!     16     8  u64 rows
//!     24     8  u64 games
//! ```
//!
//! **Version 1** (still readable by `tools/train_value.py` for existing
//! corpora — this format never changes retroactively): `rows` records of
//! `12 + 4 * num_features` bytes:
//!
//! ```text
//!      0     4  u32 seed          — the game's seed, i.e. its identity
//!      4     4  u32 label         — the `duels_value::Outcome` index
//!      8     4  f32 search_value  — mcts-eval's root value, this row's
//!                                   perspective; a column, not a target
//!     12   4*N  f32 features
//! ```
//!
//! **Version 2** (this tool's current output): `rows` records of
//! `20 + 4 * num_features` bytes:
//!
//! ```text
//!      0     8  u64 seed          — widened from u32: `value_corpus_mv.rs`
//!                                   format v2 corpora can run seeds well
//!                                   past 2^32 (three generations' worth of
//!                                   disjoint seed ranges adds up)
//!      8     4  u32 label         — the `duels_value::Outcome` index
//!     12     4  f32 search_value  — the root value recorded for this
//!                                   decision, this row's perspective, or
//!                                   **NaN** if the decision was a
//!                                   specialist's (`role != 0` in the
//!                                   corpus) — see "Specialist rows" below.
//!                                   A column, not a target.
//!     16     4  u32 ply           — the decision's ply index in its game;
//!                                   cheap to keep for phase-based analysis,
//!                                   unused by `train_value.py` today
//!     20   4*N  f32 features
//! ```
//!
//! The seed is on every row because a consumer **must split train and
//! validation by game, not by row** — two rows from one game share a seed, a
//! deal and most of a board, so a row-wise split leaks nearly every
//! validation position into training. `tools/train_value.py` splits on
//! `seed % 10`.
//!
//! # Specialist rows: `search_value` is `NaN`, on purpose
//!
//! A `value_corpus_mv.rs` format-v2 corpus can mix in specialist-agent seats
//! (`mcts-value:objective=science/military/civilian`; see that file's module
//! docs, "What a specialist's recorded `value` means"). A specialist's
//! recorded value is `P(mover wins by its own target kind)`, not a win
//! probability, and is not complementary across seats the way a win
//! probability is. It must never be treated as the aggregate win-probability
//! yardstick `tools/train_value.py`'s `--value-target-lambda` blend and
//! offline reports both use `search_value` for, so this tool writes `NaN`
//! for every row whose decision has `role != 0`, regardless of which
//! perspective the row is written from. `train_value.py` skips the blend's
//! second loss term wherever `search_value` is `NaN` for exactly this reason.
//!
//! A sidecar `<out>.json` records the run and the label histogram, so a
//! matrix is self-describing the way the corpus it came from is.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use duels_core::scoring::VictoryKind;
use duels_core::{engine, Action, GameResult, Player};
use duels_value::{features, Outcome, NUM_FEATURES, NUM_OUTCOMES};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Salt for the engine's own mid-game RNG (The Great Library's token draw).
/// Must match `value_corpus.rs`, `duels-arena::match_runner` and
/// `duels-server::room::Room::new`, or a game containing a Great Library will
/// not replay. The corpus manifest records it for exactly this reason.
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// This tool's own output format version. Bump this, not the corpus's own
/// `version` field (which is a different, independent format this tool
/// reads) — see the module docs' "The output format".
const MATRIX_VERSION: u32 = 2;

/// Bytes per record in the current (v2) matrix format: `u64` seed, `u32`
/// label, `f32` search value, `u32` ply, then the features.
const RECORD_BYTES: usize = 20 + 4 * NUM_FEATURES;

/// How many corpus lines one rayon batch covers. Bounds peak memory; has no
/// effect on the output, which is written in corpus order regardless.
const BATCH: usize = 512;

// --------------------------------------------------------------------------
// The corpus format, as much of it as this tool reads.
//
// Deliberately a private re-declaration rather than a shared type: cargo
// examples cannot import each other, and the alternative — promoting the
// corpus schema into `duels-arena`'s library surface — would put a
// training-data format in the tournament runner's public API for the benefit
// of two diagnostics. `value_corpus.rs --verify` is the authority on whether a
// corpus file is well-formed; this reads the three fields it needs and would
// fail loudly on a schema change.
// --------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct Decision {
    ply: u32,
    mover: Player,
    /// The root's backed-up value. A win probability, from `Player::One`'s
    /// perspective, when `role == 0`; see `value_corpus_mv.rs`'s module docs
    /// ("What a specialist's recorded `value` means") for what it is
    /// otherwise. Read by `rows_for` only to decide the sign of the flip for
    /// `role == 0` rows — never emitted as `search_value` for any other role.
    value: f64,
    /// `0` = generalist; nonzero = one of the three specialists. Absent in a
    /// format-v1 corpus (written before `value_corpus_mv.rs`'s format-v2
    /// task), which was all-generalist — hence the default.
    #[serde(default)]
    role: u8,
}

#[derive(Debug, Clone, Deserialize)]
struct GameLine {
    seed: u64,
    moves: u32,
    result: GameResult,
    actions: Vec<Action>,
    decisions: Vec<Decision>,
}

/// The sidecar that makes a matrix self-describing.
#[derive(Debug, Clone, Serialize)]
struct Sidecar {
    kind: String,
    version: u32,
    generated_by: String,
    corpus: String,
    num_features: usize,
    num_outcomes: usize,
    record_bytes: usize,
    games: u64,
    rows: u64,
    stride: usize,
    /// Rows per `Outcome`, in index order — the class balance a trainer has to
    /// reckon with, and the measurement behind this crate's claim that the
    /// science head is the rare one.
    label_counts: [u64; NUM_OUTCOMES],
    label_names: [&'static str; NUM_OUTCOMES],
    /// Games per `VictoryKind`, plus draws: the same fact at the game level,
    /// unaffected by how many decisions each game contributed.
    games_by_kind: GamesByKind,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct GamesByKind {
    military_supremacy: u64,
    scientific_supremacy: u64,
    civilian_victory: u64,
    civilian_tiebreak: u64,
    draw: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    if args.is_empty() || args.iter().any(|a| a == "--help") {
        eprintln!(
            "feature_dump: turn an mcts-eval value corpus into a duels-value training matrix\n\
             \n\
             --corpus <path>   the .jsonl corpus (required)\n\
             --out <path>      the matrix to write (required)\n\
             --games N         how many corpus lines to read (default: all)\n\
             --stride N        keep every Nth labelled decision (default 1)\n\
             --threads N       rayon threads (default: all)\n"
        );
        return;
    }

    let corpus = PathBuf::from(flag("--corpus").expect("--corpus <path> is required"));
    let out = PathBuf::from(flag("--out").expect("--out <path> is required"));
    let games: usize = flag("--games")
        .map(|s| s.parse().expect("--games must be a number"))
        .unwrap_or(usize::MAX);
    let stride: usize = flag("--stride")
        .map(|s| s.parse().expect("--stride must be a number"))
        .unwrap_or(1);
    assert!(stride >= 1, "--stride must be at least 1");
    if let Some(t) = flag("--threads") {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t.parse().expect("--threads must be a number"))
            .build_global()
            .expect("the rayon pool is configured once");
    }

    dump(&corpus, &out, games, stride);
}

/// One game's rows, or a panic naming the seed that failed to replay.
///
/// Returns the encoded bytes rather than a `Vec<Row>` so the parallel section
/// allocates once per game and the serial section is a single `write_all`.
fn rows_for(g: &GameLine, stride: usize) -> (Vec<u8>, [u64; NUM_OUTCOMES]) {
    assert_eq!(
        g.moves as usize,
        g.actions.len(),
        "seed {}: moves disagrees with the action list",
        g.seed
    );
    let mut buf = Vec::with_capacity(g.decisions.len().div_ceil(stride) * 2 * RECORD_BYTES);
    let mut counts = [0u64; NUM_OUTCOMES];

    // The replay recipe from `value_corpus.rs`'s module docs, which
    // `docs/rules-spec.md` R-108 licenses: the engine is deterministic given
    // `(seed, actions)`.
    let mut state = engine::new_game(g.seed);
    let mut rng = StdRng::seed_from_u64(g.seed ^ ENGINE_RNG_SALT);
    let mut next = 0usize;
    let mut kept = 0usize;

    for (ply, &action) in g.actions.iter().enumerate() {
        let labelled = g.decisions.get(next).is_some_and(|d| d.ply as usize == ply);
        if labelled {
            let d = &g.decisions[next];
            next += 1;
            assert_eq!(
                d.mover,
                state.current_player(),
                "seed {}: ply {ply} names the wrong mover — the corpus does not replay",
                g.seed
            );
            if kept.is_multiple_of(stride) {
                for me in Player::ALL {
                    let label = Outcome::of(g.result, me);
                    counts[label.index()] += 1;
                    // `role != 0` means this decision's `value` is a
                    // specialist's `P(mover wins by its own target kind)`,
                    // not a win probability -- not complementary across
                    // seats, and not comparable to a generalist row. Write
                    // NaN rather than let it leak downstream looking like an
                    // aggregate win probability; see the module docs.
                    let value = if d.role != 0 {
                        f32::NAN
                    } else {
                        // The corpus stores the root value from Player One's
                        // view; this row's perspective may be the other one.
                        (match me {
                            Player::One => d.value,
                            Player::Two => 1.0 - d.value,
                        }) as f32
                    };
                    write_record(
                        &mut buf,
                        g.seed,
                        label,
                        value,
                        ply as u32,
                        &features(&state, me),
                    );
                }
            }
            kept += 1;
        }
        engine::apply(&mut state, action, &mut rng)
            .unwrap_or_else(|e| panic!("seed {}: replay failed at ply {ply}: {e}", g.seed));
    }

    assert_eq!(
        next,
        g.decisions.len(),
        "seed {}: {} decisions were never matched to a ply",
        g.seed,
        g.decisions.len() - next
    );
    let replayed = state
        .result()
        .unwrap_or_else(|| panic!("seed {}: the replay did not finish the game", g.seed));
    assert_eq!(
        replayed, g.result,
        "seed {}: the replay reached a different result",
        g.seed
    );
    (buf, counts)
}

fn write_record(
    buf: &mut Vec<u8>,
    seed: u64,
    label: Outcome,
    value: f32,
    ply: u32,
    f: &[f32; NUM_FEATURES],
) {
    buf.extend_from_slice(&seed.to_le_bytes());
    buf.extend_from_slice(&(label.index() as u32).to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
    buf.extend_from_slice(&ply.to_le_bytes());
    for x in f {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

fn dump(corpus: &Path, out: &Path, want_games: usize, stride: usize) {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).expect("the output directory is creatable");
    }
    let file = File::create(out).expect("the output file is creatable");
    let mut w = BufWriter::with_capacity(1 << 22, file);
    // A placeholder header, rewritten at the end once the row count is known.
    w.write_all(&[0u8; 32])
        .expect("the output file is writable");

    println!("corpus  {}", corpus.display());
    println!("out     {}", out.display());
    println!("shape   {NUM_FEATURES} features -> {NUM_OUTCOMES} outcomes, {RECORD_BYTES} B/row");
    println!("stride  {stride}");
    println!();

    let reader = BufReader::with_capacity(1 << 20, File::open(corpus).expect("the corpus opens"));
    let mut lines = reader.lines();
    let mut games = 0u64;
    let mut rows = 0u64;
    let mut counts = [0u64; NUM_OUTCOMES];
    let mut by_kind = GamesByKind::default();
    #[allow(clippy::disallowed_methods)]
    let start = std::time::Instant::now();

    loop {
        if games as usize >= want_games {
            break;
        }
        // Read a batch of raw lines serially — the JSON parse and the replay
        // are what parallelise, and the file is read once, in order.
        let mut batch: Vec<String> = Vec::with_capacity(BATCH);
        while batch.len() < BATCH && (games as usize + batch.len()) < want_games {
            match lines.next() {
                Some(Ok(line)) if line.trim().is_empty() => continue,
                Some(Ok(line)) => batch.push(line),
                Some(Err(e)) => panic!("the corpus file stopped reading: {e}"),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }

        let done: Vec<(Vec<u8>, [u64; NUM_OUTCOMES], GameResult)> = batch
            .par_iter()
            .map(|line| {
                let g: GameLine =
                    serde_json::from_str(line).unwrap_or_else(|e| panic!("a line failed: {e}"));
                let (buf, c) = rows_for(&g, stride);
                (buf, c, g.result)
            })
            .collect();

        for (buf, c, result) in &done {
            w.write_all(buf).expect("the output file is writable");
            rows += (buf.len() / RECORD_BYTES) as u64;
            for k in 0..NUM_OUTCOMES {
                counts[k] += c[k];
            }
            match result {
                GameResult::Draw => by_kind.draw += 1,
                GameResult::Win { kind, .. } => match kind {
                    VictoryKind::MilitarySupremacy => by_kind.military_supremacy += 1,
                    VictoryKind::ScientificSupremacy => by_kind.scientific_supremacy += 1,
                    VictoryKind::CivilianVictory => by_kind.civilian_victory += 1,
                    VictoryKind::CivilianTiebreak => by_kind.civilian_tiebreak += 1,
                },
            }
        }
        games += done.len() as u64;
        #[allow(clippy::disallowed_methods)]
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "  {games} games  {rows} rows  {:.0} games/s",
            games as f64 / elapsed.max(1e-9)
        );
        if done.len() < BATCH && (games as usize) < want_games {
            break; // the corpus ran out
        }
    }

    // Now that `rows` is known, go back and write the real header.
    w.flush().expect("the output file flushes");
    let mut file = w.into_inner().expect("the writer unwraps");
    file.seek(SeekFrom::Start(0)).expect("the file seeks");
    let mut header = Vec::with_capacity(32);
    header.extend_from_slice(b"DVFD");
    header.extend_from_slice(&MATRIX_VERSION.to_le_bytes());
    header.extend_from_slice(&(NUM_FEATURES as u32).to_le_bytes());
    header.extend_from_slice(&(NUM_OUTCOMES as u32).to_le_bytes());
    header.extend_from_slice(&rows.to_le_bytes());
    header.extend_from_slice(&games.to_le_bytes());
    assert_eq!(header.len(), 32);
    file.write_all(&header).expect("the header is writable");
    file.sync_all().expect("the file syncs");

    let sidecar = Sidecar {
        kind: "duels-value-features".to_string(),
        version: MATRIX_VERSION,
        generated_by: "duels-arena examples/feature_dump.rs".to_string(),
        corpus: corpus.display().to_string(),
        num_features: NUM_FEATURES,
        num_outcomes: NUM_OUTCOMES,
        record_bytes: RECORD_BYTES,
        games,
        rows,
        stride,
        label_counts: counts,
        label_names: [
            Outcome::MilitaryWin.name(),
            Outcome::ScienceWin.name(),
            Outcome::CivilianWin.name(),
            Outcome::Loss.name(),
        ],
        games_by_kind: by_kind,
    };
    let spath = {
        let mut s = out.as_os_str().to_owned();
        s.push(".json");
        PathBuf::from(s)
    };
    fs::write(
        &spath,
        serde_json::to_string_pretty(&sidecar).expect("the sidecar serializes"),
    )
    .expect("the sidecar is writable");

    #[allow(clippy::disallowed_methods)]
    let wall = start.elapsed().as_secs_f64();
    let bytes = fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    println!();
    println!(
        "wrote   {rows} rows from {games} games ({:.1} MiB)",
        bytes as f64 / 1e6
    );
    println!("        {}", out.display());
    println!("        {}", spath.display());
    println!("took    {wall:.1} s");
    println!();
    println!("label balance (rows, both perspectives — symmetric by construction)");
    for (outcome, n) in Outcome::ALL.iter().zip(&counts) {
        let name = outcome.name();
        let pct = 100.0 * *n as f64 / rows.max(1) as f64;
        println!("  {name:<14} {n:>10}  {pct:5.2}%");
    }
    println!();
    println!("games by victory kind");
    let g = games.max(1) as f64;
    for (name, n) in [
        ("military", by_kind.military_supremacy),
        ("science", by_kind.scientific_supremacy),
        ("civilian", by_kind.civilian_victory),
        ("tiebreak", by_kind.civilian_tiebreak),
        ("draw", by_kind.draw),
    ] {
        println!("  {name:<10} {n:>8}  {:5.2}%", 100.0 * n as f64 / g);
    }
}
