//! Run `tests/probability_coherence.rs`'s two measurements — zero-sum
//! coherence over reachable positions, and opening probability mass — against
//! an **arbitrary** weights file, not just the embedded [`default_net`].
//!
//! `tests/probability_coherence.rs` is deliberately scoped to the shipped
//! default: its assertions are pinned bounds on `default_net()`, and a
//! candidate weights file being evaluated for promotion (e.g. a
//! `tools/train_value.py --class-weight-mode ...` experiment, one still
//! living under `arena/corpus/`, never committed until adopted) has no
//! embedded constant to test against. This binary is the same measurement,
//! generalised to `--weights <path>`, so a candidate can be checked on this
//! axis before anyone decides whether to promote it — exactly the same
//! reason [`duels_value::weights_id`] was generalised from
//! `default_weights_id`.
//!
//! ```text
//! cargo run --release -p duels-value --example coherence_check -- \
//!     --weights arena/corpus/reweight-grid/inverse.bin
//! ```
//!
//! Omitting `--weights` measures the embedded default, reproducing the crate
//! docs' own numbers.

use duels_core::{engine, GameState, Player};
use duels_value::Net;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;

const PLIES: [usize; 10] = [0, 4, 8, 16, 24, 32, 40, 48, 56, 64];
const REACHABLE_SEEDS: u64 = 128;

fn reachable() -> Vec<GameState> {
    let mut out = Vec::new();
    let max = PLIES[PLIES.len() - 1];
    for seed in 0..REACHABLE_SEEDS {
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FF_EE11);
        if PLIES.contains(&0) {
            out.push(state.clone());
        }
        for ply in 1..=max {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let a = legal[rng.gen_range(0..legal.len())];
            if engine::apply_quiet(&mut state, a, &mut rng).is_err() {
                break;
            }
            if PLIES.contains(&ply) {
                out.push(state.clone());
            }
        }
    }
    out
}

fn quantile(sorted: &[f32], f: f32) -> f32 {
    sorted[((sorted.len() as f32 - 1.0) * f).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut weights_path = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                weights_path = Some(args[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(1);
            }
        }
    }

    let net = match &weights_path {
        Some(path) => {
            let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
            Net::from_bytes(&bytes).unwrap_or_else(|e| panic!("parsing {path}: {e}"))
        }
        None => duels_value::default_net(),
    };
    println!(
        "weights  {}",
        weights_path.as_deref().unwrap_or("(embedded default)")
    );

    // --- zero-sum coherence over reachable positions -----------------------
    let states = reachable();
    let mut gaps: Vec<f32> = states
        .iter()
        .map(|s| net.win_probability(s, Player::One) + net.win_probability(s, Player::Two) - 1.0)
        .collect();
    let signed_mean = gaps.iter().sum::<f32>() / gaps.len() as f32;
    let mut abs: Vec<f32> = gaps.drain(..).map(|g| g.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = abs.len() as f32;
    let mean = abs.iter().sum::<f32>() / n;
    let p90 = quantile(&abs, 0.90);
    let p99 = quantile(&abs, 0.99);
    let max = abs[abs.len() - 1];
    let within_05 = abs.iter().filter(|g| **g < 0.05).count() as f32 / n;
    let within_10 = abs.iter().filter(|g| **g < 0.10).count() as f32 / n;
    println!(
        "coherence over {} reachable positions: mean {mean:.4} p90 {p90:.4} p99 {p99:.4} \
         max {max:.4}  mean signed {signed_mean:.4}",
        abs.len()
    );
    println!("  fraction within 0.05: {within_05:.3}   within 0.10: {within_10:.3}");

    // --- opening probability mass ------------------------------------------
    let deals = 512u64;
    let mut sums = Vec::new();
    let mut means = [0.0f64; 2];
    let (mut min_p, mut max_p) = (f32::MAX, f32::MIN);
    for seed in 0..deals {
        let state = engine::new_game(seed);
        let one = net.win_probability(&state, Player::One);
        let two = net.win_probability(&state, Player::Two);
        for (i, p) in [one, two].into_iter().enumerate() {
            means[i] += f64::from(p) / deals as f64;
            min_p = min_p.min(p);
            max_p = max_p.max(p);
        }
        sums.push(one + two);
    }
    let mass = sums.iter().sum::<f32>() / deals as f32;
    println!(
        "opening over {deals} deals: mean P(One) {:.4}  mean P(Two) {:.4}  mean mass {mass:.4}  \
         per-seat range [{min_p:.4}, {max_p:.4}]",
        means[0], means[1]
    );
}
