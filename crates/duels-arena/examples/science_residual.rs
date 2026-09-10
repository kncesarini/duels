//! **Is `mcts-eval`'s root value conditionally miscalibrated on the science
//! race?** A read-only diagnostic over an existing value corpus.
//!
//! ```text
//! cargo run --release -p duels-arena --example science_residual -- \
//!     arena/corpus/mcts-eval-nodes2000.jsonl
//! ```
//!
//! Not to be confused with `duels-eval`'s `examples/science_calibration.rs`,
//! which this file was called until the two collided on one output filename.
//! That one asks whether `duels_eval::win_probability` is well-calibrated per
//! `(age, distinct_science)` bucket over fresh self-play, and its answers are
//! written in victory points per science rung. This one holds a *search's*
//! backed-up root value fixed and asks whether `duels_strategy::science_read`
//! still predicts the outcome inside a value band — a residual test over an
//! existing corpus, which is what "residual" in the name means.
//!
//! # The question
//!
//! An earlier look at the corpus found that in the ~2.3% of games that end in
//! `VictoryKind::ScientificSupremacy`, the search's recorded root value — read
//! from the eventual *winner's* side — sits around 0.38-0.40 through the whole
//! middle of the game and only crosses 0.5 in the final fifth. Two readings fit
//! that equally well:
//!
//! 1. **A blind spot.** The search cannot see the science race coming, so it
//!    misprices positions where it is developing. Fixable, and cheaply: a
//!    `duels-strategy` science read is public information the search could
//!    already compute.
//! 2. **Correct pessimism.** A science win is a low-probability path, and a
//!    position on it genuinely is losing more often than not. Then 0.38 is not
//!    an error, it is the right answer, and there is nothing to fix.
//!
//! Looking at science-victory games alone cannot separate these, because
//! conditioning on the outcome guarantees a low-value-yet-winning trajectory
//! under *both* readings. The discriminating test has to hold the search's own
//! verdict fixed and ask whether a *second, freely available* signal still
//! predicts the outcome:
//!
//! > Among positions the search scored at (say) 0.35-0.45 for the mover, do
//! > those with a **high** [`duels_strategy::science_read`] magnitude go on to
//! > win more often than those with a **low** one?
//!
//! * **Yes** → the value is conditionally miscalibrated on information the
//!   search could have had for free. A proven, fixable blind spot.
//! * **Flat** → the value already contains whatever `science_read` knows. This
//!   particular mitigation is a dead end, and that is a decisive negative
//!   rather than a failure.
//!
//! This is [`duels_strategy`]'s own framing turned into a measurement: the
//! crate docs argue win-condition reads belong in the search *policy* because a
//! static evaluation cannot see a race three moves out. If that is true of the
//! evaluation inside an `mcts-eval` leaf, it should show up here as exactly
//! this kind of residual.
//!
//! # How it works
//!
//! Nothing here searches, plays or trains. It reuses `value_corpus.rs`'s replay
//! recipe (R-108: the engine is deterministic given `(seed, actions)`) to
//! re-derive every labelled position, and adds one
//! [`duels_strategy::Context`] per decision — from which both players' science
//! magnitudes come essentially free. The corpus's own `--verify` pass replays
//! 100,000 games in about 7 seconds single-threaded; this one does the same
//! work plus the reads, across `rayon`.
//!
//! # Reading the output
//!
//! Three things worth knowing before believing a number in it:
//!
//! * **Rows within a game are strongly correlated** — they are the same game
//!   from successive plies. So the plain binomial interval on a cell is
//!   optimistic, and the determining comparison is reported with a
//!   **game-clustered bootstrap** interval (resampling whole games, which is
//!   the unit that is actually independent) alongside it.
//! * **Magnitude and game phase are confounded.** A high science magnitude is
//!   mostly a late-game event, and late-game positions at a fixed value are not
//!   the same population as early-game ones. The key band is therefore also
//!   reported split by phase.
//! * **Some cells are thin.** Scientific supremacy is ~2.3% of games; a cell
//!   crossing a narrow value band with the top magnitude class and a single
//!   phase third can be down to a few hundred rows. Every cell prints its `n`.
//!
//! # The answer, measured
//!
//! Over the full `arena/corpus/mcts-eval-nodes2000.jsonl` (100,000 games,
//! 6,722,636 labelled decisions; the whole pass takes **3.7 seconds**):
//!
//! **Reading 1 wins. The value is conditionally miscalibrated on the science
//! read, and by a lot.** In the 0.35-0.45 band the effect is monotone across
//! all five magnitude classes and the extremes are 23 points apart:
//!
//! ```text
//!   magnitude class        n     realized     mean value
//!   [0.00,0.02)        314169      0.3254        0.3999
//!   [0.02,0.10)        126440      0.3682        0.4012
//!   [0.10,0.30)        209583      0.3763        0.4054
//!   [0.30,0.60)        210599      0.3952        0.4062
//!   [0.60,1.00]          9909      0.5581        0.3987
//!
//!   high minus low: +0.2327, game-clustered bootstrap 95% CI [+0.2143, +0.2512]
//! ```
//!
//! Note the `mean value` column: the classes are matched on the search's own
//! verdict to three decimals, so this is not a within-band gradient. Narrowing
//! the bands to 0.02 wide matches them to *four* decimals and the gap survives
//! intact in every one (+0.17 to +0.27, ten consecutive bands). It also
//! survives splitting the band by phase third (+0.21 early, +0.25 middle, +0.26
//! late), so it is not the "a live science race is a late-game event"
//! confound. The same comparison is positive in every value band: +0.31 below
//! 0.35, +0.16 in 0.45-0.55, +0.02 above 0.60.
//!
//! **The mechanism is specifically the science race, not general strength.**
//! Splitting the mover's realized wins in that band by victory kind: the top
//! magnitude class turns 35.5% of its rows into a *scientific supremacy* win
//! against the bottom class's 0.01%, while its civilian wins actually fall
//! (0.17 against 0.24) and its military wins fall too (0.03 against 0.07). The
//! extra win rate is not spread across the kinds — it is the science win, which
//! is the one the search was not pricing.
//!
//! **It is symmetric, and the opponent's side is the bigger error.** The
//! differential table (mover magnitude minus opponent's) shows the search
//! *over*-valuing positions where the opponent has the live race, more severely
//! than it under-values its own: at a recorded 0.7-0.8 the mover really wins
//! 0.659 when the opponent leads the science read by >0.30, against 0.845 when
//! neither side has a race. At a recorded 0.8-0.9 it is 0.742 against 0.923.
//!
//! **But the aggregate prize is modest, because the effect is concentrated.**
//! A holdout-validated recalibration (fitted on even-indexed games, scored on
//! odd — split by game, never by row) puts a number on what is available:
//!
//! ```text
//!   model                            brier      log-loss
//!   raw search value                0.17277    0.51416
//!   value only (recalibrated)       0.17045    0.50723
//!   value x science magnitude       0.16944    0.50375
//!   value x magnitude differential  0.16908    0.50291
//! ```
//!
//! Beyond what sharpening the value alone buys, the science read is worth
//! `0.0010` of Brier (`0.0014` for the differential) — about 0.6-0.8% relative.
//! That is small *because only ~1% of rows sit in the top magnitude class* and
//! 43.7% have a magnitude of exactly zero. Where it applies the per-position
//! error is 20+ points of win probability; averaged over every position in
//! every game it is a rounding error. Both halves of that sentence matter for
//! deciding what to do about it: this is a sharp, well-localized defect, not a
//! diffuse one, so the promising mitigations are the ones that cost nothing on
//! the 99% of positions where there is nothing to fix.

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use duels_core::scoring::VictoryKind;
use duels_core::{engine, Action, GameResult, Player};
use duels_strategy::{science_read_with, Context};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use serde::Deserialize;

/// Must match `value_corpus.rs`, or the replay diverges on any game containing
/// The Great Library. The corpus manifest records it too.
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Lines read into memory at once before being replayed in parallel. Bounds
/// peak memory against a ~1 GiB corpus; has no effect on the output.
const BATCH: usize = 4096;

/// Bootstrap resamples for the game-clustered intervals.
const BOOTSTRAP: usize = 2000;

/// Fixed seed for the bootstrap, so a rerun reproduces the intervals.
const BOOTSTRAP_SEED: u64 = 0x5C1E_5EED;

// --- the corpus format, read-only ------------------------------------------

/// One recorded decision. `value_corpus.rs` also writes `visits` and `chosen`
/// per decision; neither is read here, and serde ignores unknown fields by
/// default, so they are simply not declared.
#[derive(Debug, Clone, Deserialize)]
struct Decision {
    ply: u32,
    mover: Player,
    value: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct GameLine {
    seed: u64,
    moves: u32,
    result: GameResult,
    actions: Vec<Action>,
    decisions: Vec<Decision>,
}

// --- one labelled decision, as this diagnostic sees it ----------------------

/// Deliberately compact: the full corpus is 6.7M rows, and every field here is
/// paid for 6.7M times.
#[derive(Debug, Clone, Copy)]
struct Row {
    /// Cluster id for the bootstrap — the game's index in the file, not its
    /// seed, so it indexes a dense array.
    game: u32,
    /// The search's root value, re-expressed for the **mover** (the corpus
    /// stores `Player::One`'s view).
    value: f32,
    /// `science_read(state, mover).magnitude`.
    mag: f32,
    /// The same for the opponent, for the differential view.
    opp_mag: f32,
    /// What actually happened, for the mover: 1.0 win, 0.0 loss, 0.5 draw.
    outcome: f32,
    /// `ply / (moves - 1)`, quantized to 0..=255.
    phase: u8,
    /// Whether the game ended in `ScientificSupremacy`.
    sci_win: bool,
    /// How the game ended, as an index into [`KINDS`].
    kind: u8,
}

/// Victory kinds in the order [`Row::kind`] indexes them.
const KINDS: [&str; 5] = ["military", "science", "civilian", "tiebreak", "draw"];

fn kind_index(result: &GameResult) -> u8 {
    match result {
        GameResult::Win { kind, .. } => match kind {
            VictoryKind::MilitarySupremacy => 0,
            VictoryKind::ScientificSupremacy => 1,
            VictoryKind::CivilianVictory => 2,
            VictoryKind::CivilianTiebreak => 3,
        },
        GameResult::Draw => 4,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .or_else(|| flag("--corpus"))
        .unwrap_or_else(|| {
            eprintln!(
                "science_residual <corpus.jsonl> [--games N] \
                 [--mag-bins 0.02,0.10,0.30,0.60]\n\
                 \n\
                 Joint calibration of the corpus's recorded search value against\n\
                 duels-strategy's science magnitude, over the realized outcome."
            );
            std::process::exit(2);
        });
    let limit: usize = flag("--games")
        .map(|s| s.parse().expect("--games must be a number"))
        .unwrap_or(usize::MAX);
    let mag_cuts: Vec<f32> = flag("--mag-bins")
        .unwrap_or_else(|| "0.02,0.10,0.30,0.60".to_string())
        .split(',')
        .map(|s| s.trim().parse().expect("--mag-bins takes f32 thresholds"))
        .collect();

    let path = Path::new(&path);
    describe_manifest(path);

    #[allow(clippy::disallowed_methods)]
    let start = std::time::Instant::now();
    let (rows, games) = collect(path, limit);
    #[allow(clippy::disallowed_methods)]
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "replayed {games} games, {} labelled decisions in {elapsed:.1}s",
        rows.len()
    );
    if rows.is_empty() {
        return;
    }

    report(&rows, games, &mag_cuts);
}

fn describe_manifest(path: &Path) {
    let mut s = path.as_os_str().to_owned();
    s.push(".manifest.json");
    let mpath = PathBuf::from(s);
    match fs::read_to_string(&mpath) {
        Ok(text) => {
            let v: serde_json::Value = serde_json::from_str(&text).expect("the manifest parses");
            println!("corpus  {}", path.display());
            println!(
                "  agent  {} @ {}",
                v["agent"]["name"].as_str().unwrap_or("?"),
                v["budget"].as_str().unwrap_or("?")
            );
            println!(
                "  games  {} ({} decisions), seeds {}..={}",
                v["games"], v["decisions"], v["seed_first"], v["seed_last"]
            );
        }
        Err(e) => println!("(no manifest beside {}: {e})", path.display()),
    }
    println!();
}

/// Replay the corpus and build one [`Row`] per labelled decision.
fn collect(path: &Path, limit: usize) -> (Vec<Row>, usize) {
    let reader = BufReader::new(File::open(path).expect("the corpus file is readable"));
    let mut rows: Vec<Row> = Vec::new();
    let mut games = 0usize;
    let mut batch: Vec<String> = Vec::with_capacity(BATCH);

    let flush = |batch: &mut Vec<String>, games: &mut usize, rows: &mut Vec<Row>| {
        let base = *games as u32;
        let produced: Vec<Vec<Row>> = batch
            .par_iter()
            .enumerate()
            .map(|(i, line)| {
                let g: GameLine =
                    serde_json::from_str(line).expect("a corpus line parses as a game");
                replay(&g, base + i as u32)
            })
            .collect();
        *games += batch.len();
        for r in produced {
            rows.extend(r);
        }
        batch.clear();
    };

    for line in reader.lines() {
        if games + batch.len() >= limit {
            break;
        }
        let line = line.expect("the corpus file reads");
        if line.trim().is_empty() {
            continue;
        }
        batch.push(line);
        if batch.len() == BATCH {
            flush(&mut batch, &mut games, &mut rows);
        }
    }
    if !batch.is_empty() {
        flush(&mut batch, &mut games, &mut rows);
    }
    (rows, games)
}

/// One game: replay it, and read the science race at every labelled ply.
fn replay(g: &GameLine, game_idx: u32) -> Vec<Row> {
    let mut state = engine::new_game(g.seed);
    let mut rng = StdRng::seed_from_u64(g.seed ^ ENGINE_RNG_SALT);
    let mut out = Vec::with_capacity(g.decisions.len());
    let mut next = 0usize;
    let span = f32::from(g.moves.max(2) as u16 - 1);
    let sci_win = matches!(
        g.result,
        GameResult::Win {
            kind: VictoryKind::ScientificSupremacy,
            ..
        }
    );

    for (ply, &action) in g.actions.iter().enumerate() {
        if g.decisions.get(next).is_some_and(|d| d.ply as usize == ply) {
            let d = &g.decisions[next];
            next += 1;
            debug_assert_eq!(d.mover, state.current_player(), "seed {}", g.seed);
            // One `Context` serves both reads; the magnitudes are not symmetric
            // functions of one city, so both are computed from the shared one.
            let ctx = Context::of(&state);
            let mine = science_read_with(&state, d.mover, &ctx).magnitude;
            let theirs = science_read_with(&state, d.mover.other(), &ctx).magnitude;
            let value = match d.mover {
                Player::One => d.value,
                Player::Two => 1.0 - d.value,
            };
            let outcome = match g.result.winner() {
                None => 0.5,
                Some(w) if w == d.mover => 1.0,
                Some(_) => 0.0,
            };
            out.push(Row {
                game: game_idx,
                value: value as f32,
                mag: mine as f32,
                opp_mag: theirs as f32,
                outcome,
                phase: ((ply as f32 / span) * 255.0).clamp(0.0, 255.0) as u8,
                sci_win,
                kind: kind_index(&g.result),
            });
        }
        engine::apply(&mut state, action, &mut rng)
            .unwrap_or_else(|e| panic!("seed {}: replay failed at ply {ply}: {e}", g.seed));
    }
    assert_eq!(
        next,
        g.decisions.len(),
        "seed {}: a decision never matched a ply",
        g.seed
    );
    out
}

// --- bucketing --------------------------------------------------------------

/// Which magnitude class a value falls in, given ascending cut points.
fn class_of(mag: f32, cuts: &[f32]) -> usize {
    cuts.iter().take_while(|&&c| mag >= c).count()
}

fn class_label(i: usize, cuts: &[f32]) -> String {
    let lo = if i == 0 { 0.0 } else { cuts[i - 1] };
    match cuts.get(i) {
        Some(&hi) => format!("[{lo:.2},{hi:.2})"),
        None => format!("[{lo:.2},1.00]"),
    }
}

/// Mean and its plain binomial standard error, ignoring within-game clustering.
fn mean_se(xs: &[f32]) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = xs.len() as f64;
    let m = xs.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    (m, (m * (1.0 - m) / n).sqrt())
}

/// A cell of the joint table.
#[derive(Default, Clone)]
struct Cell {
    n: u64,
    wins: f64,
}

impl Cell {
    fn push(&mut self, outcome: f32) {
        self.n += 1;
        self.wins += f64::from(outcome);
    }
    fn rate(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.wins / self.n as f64
        }
    }
    fn se(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        let p = self.rate();
        (p * (1.0 - p) / self.n as f64).sqrt()
    }
}

// --- the report -------------------------------------------------------------

fn report(rows: &[Row], games: usize, cuts: &[f32]) {
    let classes = cuts.len() + 1;

    outcome_mix(rows);
    magnitude_distribution(rows);
    marginal_calibration(rows);
    science_trajectory(rows);

    println!();
    println!("=== joint: search value (rows) x mover science magnitude (cols) ===");
    println!("Each cell: realized mover win rate, then n. Value bands are deciles of");
    println!("the search's own root value, re-expressed for the mover.");
    println!();
    joint_table(rows, cuts, classes, |_| true);

    println!();
    println!("=== the determining comparison ===");
    println!("Within a fixed search-value band, does a high science magnitude predict a");
    println!("higher realized win rate than a low one? Intervals are game-clustered");
    println!("bootstrap ({BOOTSTRAP} resamples over {games} games), which is the honest");
    println!("unit of independence here.");
    for &(lo, hi, name) in &[
        (
            0.35_f32,
            0.45_f32,
            "the \"science winner reads as losing\" band",
        ),
        (0.45, 0.55, "the uncertain band"),
        (0.60, 1.01, "a clearly-winning band"),
        (0.00, 0.35, "a clearly-losing band"),
    ] {
        band_comparison(rows, cuts, classes, lo, hi, name, games);
    }

    println!();
    println!("=== the same band, split by game phase ===");
    println!("Magnitude and phase are confounded: a live science race is mostly a");
    println!("late-game event. If the magnitude effect is really a phase effect, it");
    println!("disappears inside a phase third.");
    for &(plo, phi, pname) in &[
        (0u8, 85u8, "early (first third of plies)"),
        (85, 170, "middle third"),
        (170, 255, "late third"),
    ] {
        println!();
        println!("-- value 0.35-0.45, {pname} --");
        let sel = |r: &Row| {
            r.value >= 0.35 && r.value < 0.45 && r.phase >= plo && (r.phase < phi || phi == 255)
        };
        class_row(rows, cuts, classes, &sel);
    }

    println!();
    println!("=== narrow value bands: the within-band gradient control ===");
    println!("A decile is wide enough that the magnitude classes could differ by their");
    println!("mean value inside it rather than by anything conditional. These bands are");
    println!("0.02 wide, so there is almost no room left for that.");
    narrow_bands(rows, cuts, classes);

    println!();
    println!("=== the mechanism check: where do the extra wins come from? ===");
    println!("If the residual really is the science race, the wins the search failed to");
    println!("see should arrive as scientific supremacy. If the extra wins were spread");
    println!("evenly over the victory kinds, `science_read` would just be proxying");
    println!("general strength and the label on the finding would be wrong.");
    kind_breakdown(rows, cuts, classes);

    println!();
    println!("=== differential view: mover magnitude minus opponent magnitude ===");
    println!("A position where both sides have a live science race is not the same as");
    println!("one where only the mover does.");
    differential_table(rows);

    println!();
    println!("=== how much calibration is actually on the table ===");
    println!("Three recalibrations of the corpus's own value, each fitted on even-indexed");
    println!("games and scored on odd-indexed ones — split by GAME, never by row, because");
    println!("rows within a game are near-duplicates. `value only` is the ceiling a");
    println!("consumer reaches by sharpening the value alone; anything the magnitude");
    println!("models add beyond it is information the search did not have.");
    recalibration(rows, cuts, classes);
}

/// The key comparison inside 0.02-wide value bands, which leaves essentially no
/// within-band value gradient for the effect to hide in.
fn narrow_bands(rows: &[Row], cuts: &[f32], classes: usize) {
    let top = classes - 1;
    println!(
        "  value band     low {:<13} high {:<13} difference",
        class_label(0, cuts),
        class_label(top, cuts)
    );
    println!("                 realized  n         realized  n         (mean value lo / hi)");
    for k in 15..25 {
        let lo = k as f32 / 50.0;
        let hi = lo + 0.02;
        let mut c_lo = Cell::default();
        let mut c_hi = Cell::default();
        let mut v_lo = (0.0f64, 0u64);
        let mut v_hi = (0.0f64, 0u64);
        for r in rows.iter().filter(|r| r.value >= lo && r.value < hi) {
            let c = class_of(r.mag, cuts);
            if c == 0 {
                c_lo.push(r.outcome);
                v_lo.0 += f64::from(r.value);
                v_lo.1 += 1;
            } else if c == top {
                c_hi.push(r.outcome);
                v_hi.0 += f64::from(r.value);
                v_hi.1 += 1;
            }
        }
        if c_lo.n == 0 || c_hi.n == 0 {
            println!("  {lo:.2}-{hi:.2}      (empty)");
            continue;
        }
        println!(
            "  {lo:.2}-{hi:.2}      {:.4} {:>8}    {:.4} {:>7}    {:+.4}   ({:.4} / {:.4})",
            c_lo.rate(),
            c_lo.n,
            c_hi.rate(),
            c_hi.n,
            c_hi.rate() - c_lo.rate(),
            v_lo.0 / v_lo.1 as f64,
            v_hi.0 / v_hi.1 as f64
        );
    }
}

/// Fit a lookup-table recalibration on half the games, score it on the other
/// half. The gap between the value-only table and the value-x-magnitude table
/// is the part of the improvement the science read is responsible for.
fn recalibration(rows: &[Row], cuts: &[f32], classes: usize) {
    /// Value bins for the recalibration tables.
    const VB: usize = 25;
    let vbin = |v: f32| ((v * VB as f32) as usize).min(VB - 1);
    let dcuts = [-0.30f32, -0.05, 0.05, 0.30];
    let dclasses = dcuts.len() + 1;
    let dbin = |r: &Row| {
        dcuts
            .iter()
            .take_while(|&&c| r.mag - r.opp_mag >= c)
            .count()
    };

    let mut t_v = vec![Cell::default(); VB];
    let mut t_vm = vec![Cell::default(); VB * classes];
    let mut t_vd = vec![Cell::default(); VB * dclasses];
    for r in rows.iter().filter(|r| r.game % 2 == 0) {
        let v = vbin(r.value);
        t_v[v].push(r.outcome);
        t_vm[v * classes + class_of(r.mag, cuts)].push(r.outcome);
        t_vd[v * dclasses + dbin(r)].push(r.outcome);
    }

    // A cell too thin to trust falls back to the value-only table, which falls
    // back to the raw value: a recalibration must never be worse for want of
    // data in one corner.
    let smooth = |cell: &Cell, fallback: f64| -> f64 {
        if cell.n < 200 {
            fallback
        } else {
            cell.rate()
        }
    };

    let mut n = 0u64;
    let (mut b_raw, mut b_v, mut b_vm, mut b_vd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut l_raw, mut l_v, mut l_vm, mut l_vd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let ll = |p: f64, y: f64| {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
    };
    for r in rows.iter().filter(|r| r.game % 2 == 1) {
        let y = f64::from(r.outcome);
        let raw = f64::from(r.value);
        let v = vbin(r.value);
        let p_v = smooth(&t_v[v], raw);
        let p_vm = smooth(&t_vm[v * classes + class_of(r.mag, cuts)], p_v);
        let p_vd = smooth(&t_vd[v * dclasses + dbin(r)], p_v);
        n += 1;
        b_raw += (raw - y) * (raw - y);
        b_v += (p_v - y) * (p_v - y);
        b_vm += (p_vm - y) * (p_vm - y);
        b_vd += (p_vd - y) * (p_vd - y);
        l_raw += ll(raw, y);
        l_v += ll(p_v, y);
        l_vm += ll(p_vm, y);
        l_vd += ll(p_vd, y);
    }
    let n = n as f64;
    println!("  model                          brier      log-loss   brier gain vs value-only");
    println!(
        "  raw search value              {:.5}    {:.5}          -",
        b_raw / n,
        l_raw / n
    );
    println!(
        "  value only (recalibrated)     {:.5}    {:.5}          -",
        b_v / n,
        l_v / n
    );
    println!(
        "  value x science magnitude     {:.5}    {:.5}     {:+.5}",
        b_vm / n,
        l_vm / n,
        (b_v - b_vm) / n
    );
    println!(
        "  value x magnitude differential{:.5}    {:.5}     {:+.5}",
        b_vd / n,
        l_vd / n,
        (b_v - b_vd) / n
    );
    println!("  ({} holdout rows from odd-indexed games)", n as u64);
}

fn outcome_mix(rows: &[Row]) {
    let sci_rows = rows.iter().filter(|r| r.sci_win).count();
    println!(
        "rows from games that ended in scientific supremacy: {sci_rows} \
         ({:.2}% of rows)",
        100.0 * sci_rows as f64 / rows.len() as f64
    );
}

fn magnitude_distribution(rows: &[Row]) {
    let mut mags: Vec<f32> = rows.iter().map(|r| r.mag).collect();
    mags.sort_by(|a, b| a.partial_cmp(b).expect("magnitudes are finite"));
    println!();
    println!("=== mover science magnitude, marginal distribution ===");
    print!(" quantiles:");
    for q in [0.5, 0.75, 0.90, 0.95, 0.975, 0.99, 0.999] {
        let i = ((mags.len() - 1) as f64 * q) as usize;
        print!("  p{:<5.1}={:.4}", q * 100.0, mags[i]);
    }
    println!();
    let zero = mags.iter().take_while(|&&m| m < 1e-6).count();
    println!(
        " exactly-zero magnitude: {zero} rows ({:.1}%) — the race is dead or not started",
        100.0 * zero as f64 / mags.len() as f64
    );
    println!(
        " data-driven terciles would cut at {:.4} and {:.4} (degenerate if both ~0)",
        mags[mags.len() / 3],
        mags[2 * mags.len() / 3]
    );
}

fn marginal_calibration(rows: &[Row]) {
    println!();
    println!("=== marginal calibration (the baseline this project already knows) ===");
    println!("  search value          n     realized     (binomial se)");
    for b in 0..10 {
        let lo = b as f32 / 10.0;
        let hi = lo + 0.1;
        let picked: Vec<f32> = rows
            .iter()
            .filter(|r| r.value >= lo && (r.value < hi || b == 9))
            .map(|r| r.outcome)
            .collect();
        let (m, se) = mean_se(&picked);
        println!(
            "  {lo:.1}-{hi:.1}     {:>9}      {m:.4}       +/-{:.4}",
            picked.len(),
            1.96 * se
        );
    }
}

/// Reproduce the finding that motivated all this, so the two live side by side.
fn science_trajectory(rows: &[Row]) {
    println!();
    println!("=== the motivating finding, reproduced ===");
    println!("Recorded search value, read from the side that eventually won, by tenth of");
    println!("the game. Science-victory games against every game, for contrast.");
    println!("  phase      sci-victory games        all games");
    println!("             mean value      n     mean value      n");
    for t in 0..10 {
        let plo = (t * 255 / 10) as u8;
        let phi = ((t + 1) * 255 / 10) as u8;
        let mut sci = (0.0f64, 0u64);
        let mut all = (0.0f64, 0u64);
        for r in rows {
            if r.phase < plo || (r.phase >= phi && t != 9) {
                continue;
            }
            // "from the winner's side": the mover's value if the mover won,
            // else its complement. Draws are dropped.
            let v = match r.outcome {
                o if o > 0.9 => Some(f64::from(r.value)),
                o if o < 0.1 => Some(1.0 - f64::from(r.value)),
                _ => None,
            };
            if let Some(v) = v {
                all.0 += v;
                all.1 += 1;
                if r.sci_win {
                    sci.0 += v;
                    sci.1 += 1;
                }
            }
        }
        let f = |(s, n): (f64, u64)| {
            if n == 0 {
                (f64::NAN, 0)
            } else {
                (s / n as f64, n)
            }
        };
        let (sm, sn) = f(sci);
        let (am, an) = f(all);
        println!(
            "  {:.1}-{:.1}      {sm:.4}   {sn:>7}       {am:.4}   {an:>8}",
            t as f64 / 10.0,
            (t + 1) as f64 / 10.0
        );
    }
}

fn joint_table(rows: &[Row], cuts: &[f32], classes: usize, keep: impl Fn(&Row) -> bool) {
    let mut grid = vec![Cell::default(); 10 * classes];
    for r in rows.iter().filter(|r| keep(r)) {
        let vb = ((r.value * 10.0) as usize).min(9);
        let cb = class_of(r.mag, cuts);
        grid[vb * classes + cb].push(r.outcome);
    }
    print!("  value    ");
    for c in 0..classes {
        print!("{:>18}", class_label(c, cuts));
    }
    println!();
    for b in 0..10 {
        print!("  {:.1}-{:.1}  ", b as f64 / 10.0, (b + 1) as f64 / 10.0);
        for c in 0..classes {
            let cell = &grid[b * classes + c];
            if cell.n == 0 {
                print!("{:>18}", "-");
            } else {
                print!("{:>18}", format!("{:.3} n={}", cell.rate(), cell.n));
            }
        }
        println!();
    }
}

/// One value band, broken out by magnitude class, with plain intervals.
fn class_row(rows: &[Row], cuts: &[f32], classes: usize, keep: &impl Fn(&Row) -> bool) {
    let mut cells = vec![Cell::default(); classes];
    for r in rows.iter().filter(|r| keep(r)) {
        cells[class_of(r.mag, cuts)].push(r.outcome);
    }
    println!("  magnitude class        n     realized     95% (binomial)     mean value");
    let mut means = vec![(0.0f64, 0u64); classes];
    for r in rows.iter().filter(|r| keep(r)) {
        let c = class_of(r.mag, cuts);
        means[c].0 += f64::from(r.value);
        means[c].1 += 1;
    }
    for c in 0..classes {
        let cell = &cells[c];
        if cell.n == 0 {
            println!(
                "  {:<16} {:>8}          -                  -",
                class_label(c, cuts),
                0
            );
            continue;
        }
        let mv = means[c].0 / means[c].1 as f64;
        println!(
            "  {:<16} {:>8}      {:.4}     +/-{:.4}            {mv:.4}",
            class_label(c, cuts),
            cell.n,
            cell.rate(),
            1.96 * cell.se()
        );
    }
}

/// The headline test for one value band: top magnitude class against the
/// bottom, with a game-clustered bootstrap on the difference.
#[allow(clippy::too_many_arguments)]
fn band_comparison(
    rows: &[Row],
    cuts: &[f32],
    classes: usize,
    lo: f32,
    hi: f32,
    name: &str,
    games: usize,
) {
    println!();
    println!("-- search value {lo:.2}-{hi:.2}: {name} --");
    let keep = |r: &Row| r.value >= lo && r.value < hi;
    class_row(rows, cuts, classes, &keep);

    // Per-game sums for the two extreme classes, so whole games can be
    // resampled.
    let top = classes - 1;
    let mut per_game: Vec<[f32; 4]> = vec![[0.0; 4]; games];
    for r in rows.iter().filter(|r| keep(r)) {
        let c = class_of(r.mag, cuts);
        let g = &mut per_game[r.game as usize];
        if c == 0 {
            g[0] += 1.0;
            g[1] += r.outcome;
        } else if c == top {
            g[2] += 1.0;
            g[3] += r.outcome;
        }
    }
    let point = |sel: &[usize]| -> Option<(f64, f64, f64)> {
        let mut n_lo = 0.0;
        let mut w_lo = 0.0;
        let mut n_hi = 0.0;
        let mut w_hi = 0.0;
        for &i in sel {
            let g = per_game[i];
            n_lo += f64::from(g[0]);
            w_lo += f64::from(g[1]);
            n_hi += f64::from(g[2]);
            w_hi += f64::from(g[3]);
        }
        if n_lo == 0.0 || n_hi == 0.0 {
            return None;
        }
        let a = w_lo / n_lo;
        let b = w_hi / n_hi;
        Some((a, b, b - a))
    };
    let all: Vec<usize> = (0..games).collect();
    let Some((r_lo, r_hi, diff)) = point(&all) else {
        println!("  (one of the extreme classes is empty in this band)");
        return;
    };

    // Each resample gets its own seeded RNG derived from the band, so the
    // interval is reproducible regardless of how rayon schedules the work.
    let base_seed = BOOTSTRAP_SEED ^ ((lo * 1000.0) as u64);
    let mut diffs: Vec<f64> = (0..BOOTSTRAP)
        .into_par_iter()
        .filter_map(|b| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(b as u64));
            let sel: Vec<usize> = (0..games).map(|_| rng.gen_range(0..games)).collect();
            point(&sel).map(|(_, _, d)| d)
        })
        .collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).expect("bootstrap diffs are finite"));
    let q = |p: f64| diffs[((diffs.len() - 1) as f64 * p) as usize];
    println!(
        "  high {} minus low {}: {:+.4}  (realized {:.4} vs {:.4})",
        class_label(top, cuts),
        class_label(0, cuts),
        diff,
        r_hi,
        r_lo
    );
    println!(
        "  game-clustered bootstrap 95% CI: [{:+.4}, {:+.4}]   {}",
        q(0.025),
        q(0.975),
        if q(0.025) > 0.0 {
            "-> high magnitude realizes MORE than the value says"
        } else if q(0.975) < 0.0 {
            "-> high magnitude realizes LESS than the value says"
        } else {
            "-> interval contains zero: no conditional signal"
        }
    );
}

/// Within the key value band, how the mover's realized *wins* split by victory
/// kind, per magnitude class. Rates are per row in the class, so the columns
/// sum to that class's realized win rate.
fn kind_breakdown(rows: &[Row], cuts: &[f32], classes: usize) {
    println!("  value 0.35-0.45. Each cell: mover wins of that kind, as a rate over the");
    println!("  class's rows. The last column is the class's total realized win rate.");
    print!("  magnitude class      n  ");
    for k in KINDS {
        print!("{k:>11}");
    }
    println!("        total");
    for c in 0..classes {
        let mut counts = [0u64; 5];
        let mut n = 0u64;
        let mut wins = 0.0f64;
        for r in rows
            .iter()
            .filter(|r| r.value >= 0.35 && r.value < 0.45 && class_of(r.mag, cuts) == c)
        {
            n += 1;
            wins += f64::from(r.outcome);
            // Only the mover's own wins are attributed to a kind; a loss says
            // nothing about how the mover would have won.
            if r.outcome > 0.9 {
                counts[r.kind as usize] += 1;
            } else if r.outcome > 0.4 {
                counts[4] += 1;
            }
        }
        if n == 0 {
            continue;
        }
        print!("  {:<16} {:>7}  ", class_label(c, cuts), n);
        for c in counts {
            print!("{:>11.4}", c as f64 / n as f64);
        }
        println!("      {:.4}", wins / n as f64);
    }
}

fn differential_table(rows: &[Row]) {
    let cuts = [-0.30f32, -0.05, 0.05, 0.30];
    let classes = cuts.len() + 1;
    let label = |i: usize| -> String {
        let lo = if i == 0 { -1.0 } else { f64::from(cuts[i - 1]) };
        match cuts.get(i) {
            Some(&hi) => format!("[{lo:+.2},{hi:+.2})"),
            None => format!("[{lo:+.2},+1.00]"),
        }
    };
    let mut grid = vec![Cell::default(); 10 * classes];
    for r in rows {
        let vb = ((r.value * 10.0) as usize).min(9);
        let d = r.mag - r.opp_mag;
        let cb = cuts.iter().take_while(|&&c| d >= c).count();
        grid[vb * classes + cb].push(r.outcome);
    }
    print!("  value    ");
    for c in 0..classes {
        print!("{:>18}", label(c));
    }
    println!();
    for b in 0..10 {
        print!("  {:.1}-{:.1}  ", b as f64 / 10.0, (b + 1) as f64 / 10.0);
        for c in 0..classes {
            let cell = &grid[b * classes + c];
            if cell.n == 0 {
                print!("{:>18}", "-");
            } else {
                print!("{:>18}", format!("{:.3} n={}", cell.rate(), cell.n));
            }
        }
        println!();
    }
}
