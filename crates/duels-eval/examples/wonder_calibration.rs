//! What is a drafted-but-unbuilt wonder actually worth, and what does this
//! crate say it is worth?
//!
//! `examples/science_calibration.rs` is this instrument's older sibling and the
//! method is identical: replay real self-play games, and at every decision
//! record, for each player, some public feature of the position, what
//! [`duels_eval::win_probability`] claims about them, and whether they went on
//! to win. Bucketing by the feature and comparing the mean prediction with the
//! realised rate says whether the evaluation is calibrated *in that corner of
//! the state space*, which an aggregate fit cannot.
//!
//! The corner here is **unbuilt wonders**, and specifically unbuilt *play-again*
//! wonders, for two reasons that arrived from opposite directions in round
//! nine. The project owner's read is that this crate over-values wonders,
//! because it builds them early while they are still expensive rather than
//! waiting for the production that makes them cheap. And
//! `science_calibration --factors` found, independently, that a player **ahead**
//! on unbuilt play-again wonders wins materially *less* than
//! [`duels_eval::win_probability`] predicts while a player **behind** wins
//! *more* — a signed, monotone error worth 0.15-0.35 of win probability in Age
//! III. This example is the direct test of that, off a corpus the science tilt
//! is not steering.
//!
//! # The three features, and the model each one tests
//!
//! * `unbuilt` — how many wonders the player still holds. The flat model pays
//!   [`duels_eval::EvalWeights::wonder_potential`] × `wonder_power` for every
//!   one of them, at full weight, until the seven-wonder cap closes.
//! * `play-again diff` — unbuilt play-again wonders, this player minus the
//!   opponent. This is the [`duels_eval::EvalWeights::wonder_extra_turn_premium`]
//!   channel isolated, since the premium is the whole of what separates a
//!   play-again wonder from any other in the flat model.
//! * `p_build` — [`duels_eval::terms::wonder_p_build`], the chance any one of
//!   the player's unbuilt wonders is ever built. The flat model **does not read
//!   this at all**; if the error is concentrated where `p_build` is low, the
//!   flat weight is a constant standing in for a quantity that is not
//!   constant, and rationing it is a derivation rather than a fit.
//!
//! ```text
//! cargo run --release -p duels-eval --example wonder_calibration -- --games 2000
//! cargo run --release -p duels-eval --example wonder_calibration -- \
//!     --games 2000 --seed 5001 --read "v8:"
//! ```

use duels_core::scoring::GameResult;
use duels_core::{engine, Action, GameState, Observation, Player};
use duels_eval::{expected_value, win_probability, Config, Root};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `PhasedAgent`'s tie window, copied because the tie set feeds the RNG draw.
const TIE_EPSILON: f64 = 1e-6;
/// `duels_arena::match_runner`'s per-role agent salts, verbatim.
const AGENT_A_SALT: u64 = 0xA011_7A9E_5B21_0001;
const AGENT_B_SALT: u64 = 0xB022_8C3F_6D42_0002;

/// A verbatim copy of `PhasedAgent::choose`, RNG usage included.
struct Driver {
    rng: StdRng,
    config: Config,
}

impl Driver {
    fn new(seed: u64, config: Config) -> Driver {
        Driver {
            rng: StdRng::seed_from_u64(seed),
            config,
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let me = obs.current_player;
        let base_state = obs.sample_state(&mut self.rng);
        let root = Root::new(&base_state, me, self.config);
        let mut scored: Vec<(Action, f64)> = Vec::with_capacity(legal.len());
        for &action in legal {
            scored.push((action, expected_value(&base_state, action, me, &root)));
        }
        let Some(best_score) = scored.iter().map(|&(_, s)| s).fold(None, |m, s| match m {
            Some(b) if b >= s => Some(b),
            _ => Some(s),
        }) else {
            return legal[self.rng.gen_range(0..legal.len())];
        };
        let best: Vec<Action> = scored
            .iter()
            .filter(|&&(_, s)| (best_score - s).abs() <= TIE_EPSILON)
            .map(|&(a, _)| a)
            .collect();
        best[self.rng.gen_range(0..best.len())]
    }
}

/// Play-again wonders `p` has drafted and not built.
fn play_again_unbuilt(state: &GameState, p: Player) -> u8 {
    let ps = state.player(p);
    ps.wonders()
        .filter(|&w| !ps.has_built_wonder(w) && w.def().play_again)
        .count() as u8
}

#[derive(Clone, Copy)]
struct Sample {
    age: u8,
    predicted: f64,
    won: bool,
    mover: bool,
    unbuilt: u8,
    play_again_diff: i8,
    p_build: f64,
    /// Total flat wonder-potential victory points this player is being
    /// credited, at the reading configuration's own weight — the size of the
    /// claim under test, in the units it is made in.
    potential_vp: f64,
}

fn play_one(
    seat_one: (Config, u64),
    seat_two: (Config, u64),
    read: Config,
    setup_seed: u64,
    out: &mut Vec<Sample>,
) {
    let mut driver = [
        Driver::new(seat_one.1, seat_one.0),
        Driver::new(seat_two.1, seat_two.0),
    ];
    let mut state = engine::new_game(setup_seed);
    let mut rng = StdRng::seed_from_u64(setup_seed ^ 0xF00D);
    let mut trace: Vec<(Player, Sample)> = Vec::new();

    while !state.is_over() {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let me = state.current_player();
        let root = Root::new(&state, me, read);
        let age = state.age();
        for p in [Player::One, Player::Two] {
            let ps = state.player(p);
            let unbuilt = ps.wonders().filter(|&w| !ps.has_built_wonder(w)).count() as u8;
            trace.push((
                p,
                Sample {
                    age,
                    predicted: win_probability(&state, p, &root),
                    won: false,
                    mover: p == me,
                    unbuilt,
                    play_again_diff: play_again_unbuilt(&state, p) as i8
                        - play_again_unbuilt(&state, p.other()) as i8,
                    p_build: duels_eval::terms::wonder_p_build(&state, p, &read.eval),
                    potential_vp: read.eval.wonder_potential
                        * duels_eval::terms::wonder_potential(&state, p, &read.eval),
                },
            ));
        }
        let obs: Observation = state.observation();
        let action = driver[me.index()].choose(&obs, &legal);
        engine::apply_quiet(&mut state, action, &mut rng).expect("the driver plays legally");
    }

    if let Some(GameResult::Win { winner, .. }) = state.result() {
        for (p, mut s) in trace {
            s.won = p == winner;
            out.push(s);
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Bucket {
    n: u32,
    predicted: f64,
    won: u32,
    potential: f64,
}

impl Bucket {
    fn add(&mut self, s: &Sample) {
        self.n += 1;
        self.predicted += s.predicted;
        self.won += u32::from(s.won);
        self.potential += s.potential_vp;
    }
}

fn table(
    title: &str,
    name: &str,
    samples: &[Sample],
    bin: impl Fn(&Sample) -> Option<(usize, &'static str)>,
    bins: usize,
) {
    println!("\n{title}  (factor: {name})");
    println!(
        "  {:>3}  {:>18}  {:>8}  {:>10}  {:>9}  {:>8}  {:>9}",
        "age", name, "n", "predicted", "actual", "gap", "pot. vp"
    );
    for age in 1..=3u8 {
        let mut rows: Vec<(String, Bucket)> = (0..bins)
            .map(|_| (String::new(), Bucket::default()))
            .collect();
        for s in samples.iter().filter(|s| s.age == age && s.mover) {
            if let Some((i, label)) = bin(s) {
                if i < bins {
                    rows[i].0 = label.to_string();
                    rows[i].1.add(s);
                }
            }
        }
        for (label, b) in &rows {
            if b.n < 50 {
                continue;
            }
            let predicted = b.predicted / f64::from(b.n);
            let actual = f64::from(b.won) / f64::from(b.n);
            let se = (actual * (1.0 - actual) / f64::from(b.n)).sqrt();
            println!(
                "  {age:>3}  {label:>18}  {:>8}  {predicted:>10.3}  {actual:>9.3}  \
                 {:>+8.3}  ±{:.3}  {:>9.2}",
                b.n,
                actual - predicted,
                1.96 * se,
                b.potential / f64::from(b.n)
            );
        }
    }
}

/// One additive correction in victory points per bin, by maximum likelihood on
/// `P(win) = σ((evaluate + δ_bin(me) − δ_bin(opp)) / T(age))`.
///
/// The same fit `science_calibration`'s `rung_correction` performs, over a
/// different feature, and reported for the same reason: a probability gap says
/// the evaluation is wrong, and `δ` says by how much in the units a weight is
/// written in. `evaluate` is recovered from the recorded prediction rather than
/// stored twice — the logistic is invertible.
fn correction(
    samples: &[Sample],
    age: u8,
    bin: impl Fn(&Sample) -> usize,
    bins: usize,
) -> Vec<f64> {
    let t = duels_eval::win_probability_temperature(age);
    // The mover's sample only — the two players' rows from one position are
    // exact complements, so counting both would halve every standard error
    // while adding no information — but the *opponent's* bin has to come from
    // that complementary row, so the two are paired up here. `play_one` pushes
    // both players of every position consecutively and only for decided games,
    // so consecutive pairs are exactly positions, thread concatenation
    // included. Rails are magnitudes rather than judgements, so the saturated
    // rows are dropped.
    let mut paired: Vec<(f64, usize, usize, bool)> = Vec::new();
    let mut i = 0;
    while i < samples.len() {
        // Positions are pushed as (Player::One, Player::Two) pairs.
        let (a, b) = (&samples[i], samples.get(i + 1));
        i += 2;
        let Some(b) = b else { break };
        if a.age != age || a.age != b.age {
            continue;
        }
        let (me, opp) = if a.mover { (a, b) } else { (b, a) };
        if !me.mover || me.predicted <= 1e-6 || me.predicted >= 1.0 - 1e-6 {
            continue;
        }
        let value = t * (me.predicted / (1.0 - me.predicted)).ln();
        if value.abs() >= 100.0 {
            continue;
        }
        paired.push((value, bin(me), bin(opp), me.won));
    }
    let mut delta = vec![0.0f64; bins];
    for _ in 0..60 {
        for k in 1..bins {
            let (mut grad, mut hess) = (0.0f64, 0.0f64);
            for &(value, a, b, won) in &paired {
                let d = f64::from(u8::from(a == k)) - f64::from(u8::from(b == k));
                if d == 0.0 {
                    continue;
                }
                let z = (value + delta[a] - delta[b]) / t;
                let p = 1.0 / (1.0 + (-z).exp());
                grad += (f64::from(u8::from(won)) - p) * d / t;
                hess += p * (1.0 - p) * d * d / (t * t);
            }
            if hess > 1e-12 {
                delta[k] += grad / hess;
            }
        }
    }
    delta
}

fn parse_variant(arg: &str) -> Result<(String, Config), String> {
    let (label, params) = arg.split_once(':').unwrap_or((arg, ""));
    let mut cfg = Config::default();
    for kv in params.split(',').filter(|s| !s.trim().is_empty()) {
        let (k, raw) = kv
            .split_once('=')
            .ok_or_else(|| format!("\"{kv}\" is not key=value"))?;
        let (k, raw) = (k.trim(), raw.trim());
        if k == "base" {
            cfg = match raw {
                "v5" => Config::v5(),
                "v6" => Config::v6(),
                "v7" => Config::v7(),
                "v8" => Config::v8(),
                "default" => Config::default(),
                other => return Err(format!("unknown base \"{other}\"")),
            };
            continue;
        }
        if k == "wonder" {
            cfg.wonder_model = match raw {
                "flat" => duels_eval::WonderModel::Flat,
                "budget" => duels_eval::WonderModel::Budget,
                "rationed" => duels_eval::WonderModel::Rationed,
                other => return Err(format!("unknown wonder model \"{other}\"")),
            };
            continue;
        }
        let v: f64 = raw
            .parse()
            .map_err(|_| format!("\"{raw}\" is not a number (key {k})"))?;
        match k {
            "wonder_potential" => cfg.eval.wonder_potential = v,
            "wprem" => cfg.eval.wonder_extra_turn_premium = v,
            "turns" => cfg.eval.wonder_turns_per_wonder = v,
            other => return Err(format!("unknown key \"{other}\"")),
        }
    }
    Ok((label.to_string(), cfg))
}

fn main() {
    let mut games: u64 = 1000;
    let mut seed: u64 = 0;
    let mut threads: usize = 8;
    let mut read_arg: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--games" => games = args.next().and_then(|s| s.parse().ok()).unwrap_or(games),
            "--seed" => seed = args.next().and_then(|s| s.parse().ok()).unwrap_or(seed),
            "--threads" => threads = args.next().and_then(|s| s.parse().ok()).unwrap_or(threads),
            "--read" => read_arg = args.next(),
            other => return eprintln!("wonder_calibration: unexpected argument {other:?}"),
        }
    }
    let (label, read) = match read_arg.as_deref() {
        None => ("default".to_string(), Config::default()),
        Some(s) => match parse_variant(s) {
            Ok(v) => v,
            Err(e) => return eprintln!("wonder_calibration: {e}"),
        },
    };

    let threads = threads.max(1);
    let mut samples: Vec<Sample> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let t = t as u64;
                let nthreads = threads as u64;
                scope.spawn(move || {
                    let mut out: Vec<Sample> = Vec::new();
                    let mut i = t;
                    while i < games {
                        let s = seed + i;
                        let a = (read, s ^ AGENT_A_SALT);
                        let b = (read, s ^ AGENT_B_SALT);
                        play_one(a, b, read, s, &mut out);
                        play_one(b, a, read, s, &mut out);
                        i += nthreads;
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            samples.extend(h.join().expect("a worker thread panicked"));
        }
    });

    println!(
        "wonder_calibration: {} games from base seed {seed}, both seat orders",
        games * 2
    );
    println!(
        "{} player-position samples from decided games",
        samples.len()
    );
    println!(
        "the evaluation being read is {label}: {}",
        read.params_string()
    );

    table(
        "unbuilt wonders held",
        "unbuilt",
        &samples,
        |s| {
            Some(match s.unbuilt {
                0 => (0, "0"),
                1 => (1, "1"),
                2 => (2, "2"),
                3 => (3, "3"),
                _ => (4, "4"),
            })
        },
        5,
    );
    table(
        "unbuilt play-again wonders, this player minus the opponent",
        "play-again diff",
        &samples,
        |s| {
            Some(match s.play_again_diff {
                i8::MIN..=-2 => (0, "-2 or worse"),
                -1 => (1, "-1"),
                0 => (2, "0"),
                1 => (3, "+1"),
                _ => (4, "+2 or better"),
            })
        },
        5,
    );
    table(
        "p_build: the chance an unbuilt wonder is ever built",
        "p_build",
        &samples,
        |s| {
            if s.unbuilt == 0 {
                return None;
            }
            Some(if s.p_build >= 0.99 {
                (3, ">=0.99")
            } else if s.p_build >= 0.75 {
                (2, "0.75-0.99")
            } else if s.p_build >= 0.4 {
                (1, "0.40-0.75")
            } else {
                (0, "<0.40")
            })
        },
        4,
    );

    println!(
        "\nfitted additive correction in victory points, per play-again diff bin\n  \
         (negative = the evaluation over-credits that bin; the level bin is the reference)"
    );
    println!(
        "  {:>3}  {:>12}  {:>12}  {:>12}  {:>12}",
        "age", "-1", "0 (ref)", "+1", "+2 or better"
    );
    for age in 1..=3u8 {
        let d = correction(
            &samples,
            age,
            |s| match s.play_again_diff {
                i8::MIN..=-1 => 1,
                0 => 0,
                1 => 2,
                _ => 3,
            },
            4,
        );
        println!(
            "  {age:>3}  {:>12}  {:>12}  {:>12}  {:>12}",
            format!("{:+.1}", d[1]),
            "0.0",
            format!("{:+.1}", d[2]),
            format!("{:+.1}", d[3]),
        );
    }

    println!(
        "\nfitted additive correction in victory points, per p_build bin\n  \
         (the >=0.99 bin is the reference)"
    );
    println!(
        "  {:>3}  {:>12}  {:>12}  {:>12}  {:>12}",
        "age", "<0.40", "0.40-0.75", "0.75-0.99", ">=0.99 (ref)"
    );
    for age in 1..=3u8 {
        let d = correction(
            &samples,
            age,
            |s| {
                if s.unbuilt == 0 || s.p_build >= 0.99 {
                    0
                } else if s.p_build >= 0.75 {
                    3
                } else if s.p_build >= 0.4 {
                    2
                } else {
                    1
                }
            },
            4,
        );
        println!(
            "  {age:>3}  {:>12}  {:>12}  {:>12}  {:>12}",
            format!("{:+.1}", d[1]),
            format!("{:+.1}", d[2]),
            format!("{:+.1}", d[3]),
            "0.0",
        );
    }
}
