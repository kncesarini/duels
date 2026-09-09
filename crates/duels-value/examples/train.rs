//! Train the `duels-value` network offline and write `weights/value.bin`.
//!
//! A self-contained trainer in plain Rust — minibatch Adam on a two-layer
//! network — so producing the weights needs nothing but `cargo`. It reads the
//! rows `crates/duels-arena/examples/feature_dump.rs` writes.
//! `tools/train.py` is the same procedure in numpy/torch for anyone who
//! prefers that toolchain; the two agree on the file format and the report.
//!
//! ```text
//! cargo run --release -p duels-arena --example feature_dump -- \
//!     --corpus arena/corpus/mcts-eval-nodes2000.jsonl --out arena/corpus/features/full
//! cargo run --release -p duels-value --example train -- \
//!     --data arena/corpus/features/full --out crates/duels-value/weights/value.bin \
//!     --hidden 128 --epochs 8
//! ```
//!
//! # What it does, in order
//!
//! 1. Splits train/validation **by game (seed), never by row**. Rows within
//!    a game are the same game seen from successive plies, and a row-wise
//!    split would leak nearly every validation position into training.
//! 2. Trains the **decomposed** model: a 4-way softmax over
//!    `{military_win, science_win, civilian_win, loss}` for the evaluated
//!    player, cross-entropy against the game's actual outcome (a draw is the
//!    soft target `[0, 0, 0.5, 0.5]`).
//! 3. Trains a **single-scalar control** of identical shape and schedule (one
//!    sigmoid, binary cross-entropy on win/loss), so "the decomposed target
//!    helps the aggregate win probability" is measured, not asserted.
//! 4. Reports held-out Brier / log-loss / accuracy of the aggregate win
//!    probability for both, per age, alongside the two comparison columns the
//!    dump carries on the *same rows* (the search's root value and
//!    `duels_eval::win_probability`), a calibration table, and the per-kind
//!    head quality of the decomposed model.
//! 5. Folds the input scaling into the first layer and writes the weights
//!    file `duels_value::Model::from_bytes` reads.
//!
//! # Scaling, not centring
//!
//! Inputs are divided by their training-set RMS and **not** mean-centred.
//! That keeps a zero feature zero, so both this trainer and the shipped
//! forward pass can skip the ~60% of inputs that are zero in any position;
//! the network's biases absorb what centring would have done.
//!
//! # Determinism
//!
//! Seeded throughout, but the minibatch gradient is a parallel reduction whose
//! summation order is not fixed, so two runs agree to float noise rather than
//! bit for bit. Good enough for a training tool; the weights file it writes is
//! what gets pinned.

use std::fs;
use std::io::Write;
use std::path::Path;

use duels_value::{feature_names, NUM_FEATURES, NUM_OUTCOMES};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const ROW_BYTES: usize = 12;

struct Data {
    n: usize,
    /// `n x NUM_FEATURES`, row-major, the bytes of an `i8` matrix.
    x: Vec<u8>,
    seed: Vec<u32>,
    label: Vec<u8>,
    search: Vec<f32>,
    eval: Vec<f32>,
    corpus: String,
    perspective: String,
    games: usize,
}

fn load(prefix: &str, max_rows: Option<usize>) -> Data {
    let meta = fs::read_to_string(format!("{prefix}.meta.json")).expect("meta.json");
    let field = |k: &str| -> String {
        let key = format!("\"{k}\": ");
        meta.find(&key)
            .map(|i| {
                let rest = &meta[i + key.len()..];
                let end = rest.find(",\n").unwrap_or(rest.len());
                rest[..end].trim().trim_matches('"').to_string()
            })
            .unwrap_or_default()
    };
    let nf: usize = field("num_features").parse().expect("num_features");
    assert_eq!(
        nf, NUM_FEATURES,
        "the dump was made with a different feature set"
    );
    let rows = fs::read(format!("{prefix}.rows.bin")).expect("rows.bin");
    let mut n = rows.len() / ROW_BYTES;
    if let Some(m) = max_rows {
        n = n.min(m);
    }
    let mut x = fs::read(format!("{prefix}.X.i8")).expect("X.i8");
    x.truncate(n * nf);
    let aux = fs::read(format!("{prefix}.aux.f32")).expect("aux.f32");
    let mut seed = Vec::with_capacity(n);
    let mut label = Vec::with_capacity(n);
    let mut search = Vec::with_capacity(n);
    let mut eval = Vec::with_capacity(n);
    for i in 0..n {
        let r = &rows[i * ROW_BYTES..(i + 1) * ROW_BYTES];
        seed.push(u32::from_le_bytes([r[0], r[1], r[2], r[3]]));
        label.push(r[7]);
        search.push(f32::from_le_bytes([r[8], r[9], r[10], r[11]]));
        let a = &aux[i * 4..(i + 1) * 4];
        eval.push(f32::from_le_bytes([a[0], a[1], a[2], a[3]]));
    }
    Data {
        n,
        x,
        seed,
        label,
        search,
        eval,
        corpus: Path::new(&field("corpus"))
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        perspective: field("perspective"),
        games: field("games").parse().unwrap_or(0),
    }
}

#[derive(Clone)]
struct Mlp {
    h: usize,
    o: usize,
    /// Input-major `[NUM_FEATURES][h]`.
    w1: Vec<f32>,
    b1: Vec<f32>,
    /// Hidden-major `[h][o]`.
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl Mlp {
    fn new(h: usize, o: usize, rng: &mut StdRng) -> Self {
        let lim1 = (6.0 / NUM_FEATURES as f32).sqrt();
        let lim2 = (6.0 / (h + o) as f32).sqrt();
        Mlp {
            h,
            o,
            w1: (0..NUM_FEATURES * h)
                .map(|_| rng.gen_range(-lim1..lim1))
                .collect(),
            b1: vec![0.0; h],
            w2: (0..h * o).map(|_| rng.gen_range(-lim2..lim2)).collect(),
            b2: vec![0.0; o],
        }
    }

    fn zeros_like(&self) -> Self {
        Mlp {
            h: self.h,
            o: self.o,
            w1: vec![0.0; self.w1.len()],
            b1: vec![0.0; self.b1.len()],
            w2: vec![0.0; self.w2.len()],
            b2: vec![0.0; self.b2.len()],
        }
    }

    fn params(&self) -> impl Iterator<Item = &f32> {
        self.w1
            .iter()
            .chain(&self.b1)
            .chain(&self.w2)
            .chain(&self.b2)
    }

    fn params_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.w1
            .iter_mut()
            .chain(self.b1.iter_mut())
            .chain(self.w2.iter_mut())
            .chain(self.b2.iter_mut())
    }

    fn add(&mut self, other: &Mlp) {
        for (a, b) in self.params_mut().zip(other.params()) {
            *a += *b;
        }
    }

    /// Forward pass on one scaled row; returns the output probabilities.
    fn forward(&self, x: &[u8], scale: &[f32], hidden: &mut [f32]) -> [f32; NUM_OUTCOMES] {
        hidden.copy_from_slice(&self.b1);
        for (i, &b) in x.iter().enumerate() {
            let xi = b as i8;
            if xi == 0 {
                continue;
            }
            let xs = f32::from(xi) * scale[i];
            let row = &self.w1[i * self.h..(i + 1) * self.h];
            for (acc, &w) in hidden.iter_mut().zip(row) {
                *acc += xs * w;
            }
        }
        let mut z = [0.0f32; NUM_OUTCOMES];
        z[..self.o].copy_from_slice(&self.b2);
        for (j, &a) in hidden.iter().enumerate() {
            if a <= 0.0 {
                continue;
            }
            let row = &self.w2[j * self.o..(j + 1) * self.o];
            for k in 0..self.o {
                z[k] += a * row[k];
            }
        }
        output(z, self.o)
    }

    /// Forward and backward on one row, accumulating into `g`; returns the loss.
    fn forward_backward(
        &self,
        x: &[u8],
        scale: &[f32],
        target: &[f32; NUM_OUTCOMES],
        g: &mut Mlp,
        hidden: &mut [f32],
    ) -> f32 {
        let p = self.forward(x, scale, hidden);
        let mut loss = 0.0f32;
        let mut dz = [0.0f32; NUM_OUTCOMES];
        for k in 0..self.o {
            let pk = p[k].clamp(1e-7, 1.0 - 1e-7);
            if self.o == 1 {
                loss -= target[0] * pk.ln() + (1.0 - target[0]) * (1.0 - pk).ln();
            } else {
                loss -= target[k] * pk.ln();
            }
            dz[k] = p[k] - target[k];
        }
        for (gb, &d) in g.b2.iter_mut().zip(&dz[..self.o]) {
            *gb += d;
        }
        // dh, then dW2 / dW1.
        for (j, hj) in hidden.iter_mut().enumerate() {
            let a = *hj;
            if a <= 0.0 {
                *hj = 0.0; // reuse as dh
                continue;
            }
            let row = &self.w2[j * self.o..(j + 1) * self.o];
            let grow = &mut g.w2[j * self.o..(j + 1) * self.o];
            let mut dh = 0.0f32;
            for ((gw, &w), &d) in grow.iter_mut().zip(row).zip(&dz[..self.o]) {
                *gw += a * d;
                dh += w * d;
            }
            *hj = dh;
        }
        for (gb, &dh) in g.b1.iter_mut().zip(hidden.iter()) {
            *gb += dh;
        }
        for (i, &b) in x.iter().enumerate() {
            let xi = b as i8;
            if xi == 0 {
                continue;
            }
            let xs = f32::from(xi) * scale[i];
            let grow = &mut g.w1[i * self.h..(i + 1) * self.h];
            for (gw, &dh) in grow.iter_mut().zip(hidden.iter()) {
                *gw += xs * dh;
            }
        }
        loss
    }
}

fn output(mut z: [f32; NUM_OUTCOMES], o: usize) -> [f32; NUM_OUTCOMES] {
    if o == 1 {
        let p = 1.0 / (1.0 + (-z[0]).exp());
        return [p, 0.0, 0.0, 0.0];
    }
    let m = z[..o].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in z[..o].iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    for v in z[..o].iter_mut() {
        *v /= sum;
    }
    z
}

fn target_of(label: u8, o: usize) -> [f32; NUM_OUTCOMES] {
    let mut t = [0.0f32; NUM_OUTCOMES];
    if o == 1 {
        t[0] = match label {
            0..=2 => 1.0,
            3 => 0.0,
            _ => 0.5,
        };
    } else {
        match label {
            0..=3 => t[label as usize] = 1.0,
            _ => {
                t[2] = 0.5;
                t[3] = 0.5;
            }
        }
    }
    t
}

fn win_of(label: u8) -> f32 {
    match label {
        0..=2 => 1.0,
        3 => 0.0,
        _ => 0.5,
    }
}

struct Args {
    data: String,
    out: Option<String>,
    hidden: usize,
    epochs: usize,
    batch: usize,
    lr: f32,
    val_frac: f64,
    seed: u64,
    max_rows: Option<usize>,
    control: bool,
}

fn train(data: &Data, train_idx: &[u32], scale: &[f32], o: usize, args: &Args, tag: &str) -> Mlp {
    let mut rng = StdRng::seed_from_u64(args.seed ^ (o as u64 * 0x9E37));
    let mut m = Mlp::new(args.hidden, o, &mut rng);
    let mut adam_m = m.zeros_like();
    let mut adam_v = m.zeros_like();
    let (beta1, beta2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let steps_per_epoch = train_idx.len().div_ceil(args.batch);
    let total_steps = steps_per_epoch * args.epochs;
    let warmup = (total_steps / 20).max(1);
    let mut order: Vec<u32> = train_idx.to_vec();
    let threads = rayon::current_num_threads().max(1);
    let mut step = 0usize;
    #[allow(clippy::disallowed_methods)]
    let t0 = std::time::Instant::now();
    for epoch in 0..args.epochs {
        order.shuffle(&mut rng);
        let mut loss_sum = 0.0f64;
        for batch in order.chunks(args.batch) {
            let chunk = batch.len().div_ceil(threads * 2).max(1);
            let (grad, loss) = batch
                .par_chunks(chunk)
                .map(|c| {
                    let mut g = m.zeros_like();
                    let mut hidden = vec![0.0f32; m.h];
                    let mut l = 0.0f32;
                    for &i in c {
                        let i = i as usize;
                        let x = &data.x[i * NUM_FEATURES..(i + 1) * NUM_FEATURES];
                        l += m.forward_backward(
                            x,
                            scale,
                            &target_of(data.label[i], o),
                            &mut g,
                            &mut hidden,
                        );
                    }
                    (g, l)
                })
                .reduce(
                    || (m.zeros_like(), 0.0f32),
                    |(mut a, la), (b, lb)| {
                        a.add(&b);
                        (a, la + lb)
                    },
                );
            let bn = batch.len() as f32;
            loss_sum += f64::from(loss / bn);
            step += 1;
            // Warmup then cosine decay.
            let lr = if step <= warmup {
                args.lr * step as f32 / warmup as f32
            } else {
                let t = (step - warmup) as f32 / (total_steps - warmup).max(1) as f32;
                args.lr * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
            };
            let bc1 = 1.0 - beta1.powi(step as i32);
            let bc2 = 1.0 - beta2.powi(step as i32);
            for (((p, g), am), av) in m
                .params_mut()
                .zip(grad.params())
                .zip(adam_m.params_mut())
                .zip(adam_v.params_mut())
            {
                let g = *g / bn;
                *am = beta1 * *am + (1.0 - beta1) * g;
                *av = beta2 * *av + (1.0 - beta2) * g * g;
                let mhat = *am / bc1;
                let vhat = *av / bc2;
                *p -= lr * mhat / (vhat.sqrt() + eps);
            }
        }
        println!(
            "  [{tag}] epoch {}/{}  train loss {:.4}  {:.0}s",
            epoch + 1,
            args.epochs,
            loss_sum / steps_per_epoch as f64,
            t0.elapsed().as_secs_f64()
        );
    }
    m
}

fn predict_all(m: &Mlp, data: &Data, idx: &[u32], scale: &[f32]) -> Vec<[f32; NUM_OUTCOMES]> {
    idx.par_chunks(4096)
        .flat_map_iter(|c| {
            let mut hidden = vec![0.0f32; m.h];
            c.iter()
                .map(|&i| {
                    let i = i as usize;
                    m.forward(
                        &data.x[i * NUM_FEATURES..(i + 1) * NUM_FEATURES],
                        scale,
                        &mut hidden,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Brier, log-loss, accuracy (draws excluded from accuracy) of `P(win)`.
fn binary_metrics(p: &[f32], y: &[f32]) -> (f64, f64, f64) {
    let mut brier = 0.0f64;
    let mut ll = 0.0f64;
    let (mut right, mut decided) = (0u64, 0u64);
    for (&p, &y) in p.iter().zip(y) {
        let p = f64::from(p).clamp(1e-6, 1.0 - 1e-6);
        let y = f64::from(y);
        brier += (p - y).powi(2);
        ll -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
        if y != 0.5 {
            decided += 1;
            if (p > 0.5) == (y > 0.5) {
                right += 1;
            }
        }
    }
    let n = p.len() as f64;
    (brier / n, ll / n, right as f64 / decided.max(1) as f64)
}

fn brier_where(p: &[f32], y: &[f32], mask: &[bool]) -> f64 {
    let mut s = 0.0f64;
    let mut n = 0usize;
    for i in 0..p.len() {
        if mask[i] {
            s += (f64::from(p[i]) - f64::from(y[i])).powi(2);
            n += 1;
        }
    }
    s / n.max(1) as f64
}

fn calibration_table(p: &[f32], y: &[f32], name: &str) {
    println!("  calibration of {name}:  bucket        n   mean pred   mean outcome");
    for b in 0..10 {
        let lo = b as f32 / 10.0;
        let hi = lo + 0.1;
        let (mut n, mut sp, mut sy) = (0usize, 0.0f64, 0.0f64);
        for (&pi, &yi) in p.iter().zip(y) {
            if pi >= lo && (pi < hi || (b == 9 && pi <= 1.0)) {
                n += 1;
                sp += f64::from(pi);
                sy += f64::from(yi);
            }
        }
        if n > 0 {
            println!(
                "    {lo:.1}-{hi:.1}  {n:9}     {:.3}       {:.3}",
                sp / n as f64,
                sy / n as f64
            );
        }
    }
}

fn write_weights(path: &str, m: &Mlp, scale: &[f32], desc: &str) {
    assert_eq!(m.o, NUM_OUTCOMES);
    let mut out = Vec::new();
    out.extend_from_slice(b"DVAL");
    for v in [
        1u32,
        NUM_FEATURES as u32,
        m.h as u32,
        NUM_OUTCOMES as u32,
        desc.len() as u32,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(desc.as_bytes());
    // Fold the scaling into the first layer: the shipped model reads raw
    // integers, so w1'[i] = w1[i] * scale[i].
    for (row, &s) in m.w1.chunks_exact(m.h).zip(scale) {
        for w in row {
            out.extend_from_slice(&(w * s).to_le_bytes());
        }
    }
    for v in m.b1.iter().chain(&m.w2).chain(&m.b2) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut f = fs::File::create(path).expect("the weights file is creatable");
    f.write_all(&out).unwrap();
    println!("wrote {path} ({} bytes)", out.len());
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == name)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let has = |name: &str| argv.iter().any(|a| a == name);
    if argv.is_empty() || has("--help") {
        eprintln!(
            "train: fit the duels-value network on a feature_dump\n\
             --data <prefix> [--out <weights.bin>] [--hidden 128] [--epochs 8] [--batch 4096]\n\
             [--lr 0.002] [--val-frac 0.1] [--seed 1] [--max-rows N] [--no-control] [--threads N]"
        );
        return;
    }
    let args = Args {
        data: flag("--data").expect("--data <prefix> is required"),
        out: flag("--out"),
        hidden: flag("--hidden").map(|s| s.parse().unwrap()).unwrap_or(128),
        epochs: flag("--epochs").map(|s| s.parse().unwrap()).unwrap_or(8),
        batch: flag("--batch").map(|s| s.parse().unwrap()).unwrap_or(4096),
        lr: flag("--lr").map(|s| s.parse().unwrap()).unwrap_or(2e-3),
        val_frac: flag("--val-frac")
            .map(|s| s.parse().unwrap())
            .unwrap_or(0.1),
        seed: flag("--seed").map(|s| s.parse().unwrap()).unwrap_or(1),
        max_rows: flag("--max-rows").map(|s| s.parse().unwrap()),
        control: !has("--no-control"),
    };
    if let Some(t) = flag("--threads") {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t.parse().unwrap())
            .build_global()
            .unwrap();
    }

    #[allow(clippy::disallowed_methods)]
    let t0 = std::time::Instant::now();
    let data = load(&args.data, args.max_rows);
    println!(
        "loaded {} rows x {NUM_FEATURES} features from {} games ({:.0}s)",
        data.n,
        data.games,
        t0.elapsed().as_secs_f64()
    );

    // Split by game: the last `val_frac` of the seeds are held out.
    let mut seeds: Vec<u32> = data.seed.clone();
    seeds.sort_unstable();
    seeds.dedup();
    let cut = seeds[((seeds.len() as f64) * (1.0 - args.val_frac)) as usize];
    let train_idx: Vec<u32> = (0..data.n as u32)
        .filter(|&i| data.seed[i as usize] < cut)
        .collect();
    let val_idx: Vec<u32> = (0..data.n as u32)
        .filter(|&i| data.seed[i as usize] >= cut)
        .collect();
    let n_tr_games = seeds.iter().filter(|&&s| s < cut).count();
    let n_va_games = seeds.len() - n_tr_games;
    println!(
        "split by game: train {} rows / {} games; validation {} rows / {} games (seeds >= {cut})",
        train_idx.len(),
        n_tr_games,
        val_idx.len(),
        n_va_games
    );
    let mut counts = [0usize; 5];
    for &i in &val_idx {
        counts[data.label[i as usize] as usize] += 1;
    }
    println!(
        "validation labels: military_win={} science_win={} civilian_win={} loss={} draw={}",
        counts[0], counts[1], counts[2], counts[3], counts[4]
    );

    // Scaling from a training subsample: 1 / RMS per feature.
    let mut rng = StdRng::seed_from_u64(0);
    let mut sq = vec![0.0f64; NUM_FEATURES];
    let sub = 1_000_000.min(train_idx.len());
    for _ in 0..sub {
        let i = train_idx[rng.gen_range(0..train_idx.len())] as usize;
        for (k, &b) in data.x[i * NUM_FEATURES..(i + 1) * NUM_FEATURES]
            .iter()
            .enumerate()
        {
            sq[k] += f64::from(b as i8).powi(2);
        }
    }
    let scale: Vec<f32> = sq
        .iter()
        .map(|&s| {
            let rms = (s / sub as f64).sqrt();
            if rms < 1e-3 {
                1.0
            } else {
                (1.0 / rms) as f32
            }
        })
        .collect();

    // Baselines on the validation rows.
    let yv: Vec<f32> = val_idx
        .iter()
        .map(|&i| win_of(data.label[i as usize]))
        .collect();
    let sv: Vec<f32> = val_idx.iter().map(|&i| data.search[i as usize]).collect();
    let ev: Vec<f32> = val_idx.iter().map(|&i| data.eval[i as usize]).collect();
    let names = feature_names();
    let age_col = |a: u8| {
        names
            .iter()
            .position(|n| n == &format!("g.age{a}"))
            .unwrap()
    };
    let ages: Vec<u8> = val_idx
        .iter()
        .map(|&i| {
            let r = &data.x[i as usize * NUM_FEATURES..(i as usize + 1) * NUM_FEATURES];
            (1..=3u8).find(|&a| r[age_col(a)] != 0).unwrap_or(0)
        })
        .collect();
    println!();
    println!("== held-out baselines (same rows) ==");
    let (b, ll, acc) = binary_metrics(&sv, &yv);
    println!("  search root value (mcts-eval nodes:2000)      brier {b:.4}  logloss {ll:.4}  acc {acc:.4}");
    let has_eval = ev.iter().all(|v| v.is_finite());
    if has_eval {
        let (b, ll, acc) = binary_metrics(&ev, &yv);
        println!("  duels_eval::win_probability                   brier {b:.4}  logloss {ll:.4}  acc {acc:.4}");
    }

    println!();
    println!(
        "== decomposed model: {NUM_FEATURES}-{}-4 softmax ==",
        args.hidden
    );
    let m4 = train(&data, &train_idx, &scale, NUM_OUTCOMES, &args, "4-way");
    let p4 = predict_all(&m4, &data, &val_idx, &scale);
    let win4: Vec<f32> = p4.iter().map(|p| p[0] + p[1] + p[2]).collect();
    let (b4, ll4, acc4) = binary_metrics(&win4, &yv);
    println!("  aggregate P(win) = sum of 3 heads:            brier {b4:.4}  logloss {ll4:.4}  acc {acc4:.4}");

    let mut control = None;
    if args.control {
        println!();
        println!(
            "== single-scalar control: {NUM_FEATURES}-{}-1 sigmoid ==",
            args.hidden
        );
        let m1 = train(&data, &train_idx, &scale, 1, &args, "scalar");
        let p1: Vec<f32> = predict_all(&m1, &data, &val_idx, &scale)
            .iter()
            .map(|p| p[0])
            .collect();
        let (b1, ll1, acc1) = binary_metrics(&p1, &yv);
        println!("  P(win):                                        brier {b1:.4}  logloss {ll1:.4}  acc {acc1:.4}");
        println!(
            "  decomposed - scalar: brier {:+.5}  logloss {:+.5}  acc {:+.5}  (negative brier/logloss = decomposed better)",
            b4 - b1,
            ll4 - ll1,
            acc4 - acc1
        );
        control = Some((p1, b1, ll1, acc1));
    }

    println!();
    println!("== held-out Brier by age ==");
    println!("  age        n    search    eval   4-way  scalar");
    for a in 1..=3u8 {
        let mask: Vec<bool> = ages.iter().map(|&x| x == a).collect();
        let n = mask.iter().filter(|&&m| m).count();
        if n == 0 {
            continue;
        }
        let ctl = control
            .as_ref()
            .map(|(p1, ..)| format!("{:.4}", brier_where(p1, &yv, &mask)))
            .unwrap_or_default();
        println!(
            "  {a}   {n:8}    {:.4}  {:.4}  {:.4}  {ctl}",
            brier_where(&sv, &yv, &mask),
            brier_where(&ev, &yv, &mask),
            brier_where(&win4, &yv, &mask)
        );
    }

    println!();
    println!("== decomposed heads on held-out rows ==");
    let lv: Vec<u8> = val_idx.iter().map(|&i| data.label[i as usize]).collect();
    let argmax: Vec<usize> = p4
        .iter()
        .map(|p| {
            (0..4)
                .max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap())
                .unwrap()
        })
        .collect();
    let (mut right, mut hard) = (0usize, 0usize);
    for (k, &l) in lv.iter().enumerate() {
        if l < 4 {
            hard += 1;
            if argmax[k] == l as usize {
                right += 1;
            }
        }
    }
    println!(
        "  4-way argmax accuracy (draws excluded): {:.4}",
        right as f64 / hard.max(1) as f64
    );
    println!(
        "  head            actual rate   mean pred   head brier   recall@argmax   precision@argmax"
    );
    for (k, name) in ["military_win", "science_win", "civilian_win", "loss"]
        .iter()
        .enumerate()
    {
        let n = lv.len() as f64;
        let actual = lv.iter().filter(|&&l| l as usize == k).count() as f64;
        let mean_pred: f64 = p4.iter().map(|p| f64::from(p[k])).sum::<f64>() / n;
        let hb: f64 = p4
            .iter()
            .zip(&lv)
            .map(|(p, &l)| (f64::from(p[k]) - f64::from(u8::from(l as usize == k))).powi(2))
            .sum::<f64>()
            / n;
        let tp = lv
            .iter()
            .zip(&argmax)
            .filter(|(&l, &a)| l as usize == k && a == k)
            .count() as f64;
        let predicted = argmax.iter().filter(|&&a| a == k).count() as f64;
        println!(
            "  {name:14}  {:.4}        {mean_pred:.4}      {hb:.4}       {:.3}           {:.3}",
            actual / n,
            tp / actual.max(1.0),
            tp / predicted.max(1.0)
        );
    }
    println!();
    println!("  science_win head, by predicted probability:");
    for (lo, hi) in [
        (0.0, 0.02),
        (0.02, 0.05),
        (0.05, 0.1),
        (0.1, 0.2),
        (0.2, 0.4),
        (0.4, 1.01),
    ] {
        let (mut n, mut sp, mut sy) = (0usize, 0.0f64, 0.0f64);
        for (p, &l) in p4.iter().zip(&lv) {
            if p[1] >= lo && p[1] < hi {
                n += 1;
                sp += f64::from(p[1]);
                sy += f64::from(u8::from(l == 1));
            }
        }
        if n > 0 {
            println!(
                "    [{lo:.2},{hi:.2})  n={n:8}  mean pred {:.3}  actual {:.3}",
                sp / n as f64,
                sy / n as f64
            );
        }
    }
    println!();
    calibration_table(&win4, &yv, "4-way aggregate P(win)");

    if let Some(out) = &args.out {
        let mut desc = format!(
            "duels-value mlp {NUM_FEATURES}-{}-4 softmax; trained on {n_tr_games} games (seeds < {cut}, {} rows, perspective={}) of {}; epochs={} lr={} batch={} seed={}; held-out ({n_va_games} games): brier {b4:.4} logloss {ll4:.4} acc {acc4:.4}",
            args.hidden,
            train_idx.len(),
            data.perspective,
            data.corpus,
            args.epochs,
            args.lr,
            args.batch,
            args.seed
        );
        if let Some((_, b1, ll1, acc1)) = &control {
            desc.push_str(&format!(
                "; single-scalar control brier {b1:.4} logloss {ll1:.4} acc {acc1:.4}"
            ));
        }
        write_weights(out, &m4, &scale, &desc);
        // Re-read through the library and check the folded weights reproduce
        // the trainer's own forward pass on raw integers.
        let bytes = fs::read(out).unwrap();
        let model = duels_value::Model::from_bytes(&bytes).expect("the written file parses");
        let mut worst = 0.0f32;
        let mut hidden = vec![0.0f32; m4.h];
        for &i in val_idx.iter().take(5000) {
            let i = i as usize;
            let row = &data.x[i * NUM_FEATURES..(i + 1) * NUM_FEATURES];
            let x: Vec<f32> = row.iter().map(|&b| f32::from(b as i8)).collect();
            let a = model.predict(&x).as_array();
            let b = m4.forward(row, &scale, &mut hidden);
            for k in 0..4 {
                worst = worst.max((a[k] - b[k]).abs());
            }
        }
        println!("  folded-weights check through duels_value::Model: max |diff| = {worst:.2e}");
        assert!(worst < 1e-4);
        println!("  desc: {desc}");
    }
}
