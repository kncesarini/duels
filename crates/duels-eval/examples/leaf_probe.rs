//! Is this evaluation a *better value estimate* than that one?
//!
//! `examples/calibrate.rs` answers "what temperature turns
//! [`duels_eval::evaluate`] into a win probability", and it answers it for one
//! configuration at a time, over that configuration's own self-play games. That
//! makes it the wrong instrument for comparing two configurations: change the
//! weights and the position distribution changes with them, so the two fits are
//! not taken over the same questions.
//!
//! This example fixes the questions first. It plays a corpus of self-play games
//! under **one frozen reference policy** ([`Config::v6`], the generation the
//! sixth round shipped), records every non-terminal decision position and who
//! went on to win, and then scores that *same* corpus under every candidate
//! configuration asked for, reporting mean negative log-likelihood at each
//! candidate's own maximum-likelihood temperature, plus the temperature-free
//! sign accuracy.
//!
//! # Why this is the right screen for a *leaf* value
//!
//! `duels-agent-phased` consumes this crate as a **policy**: it takes the
//! argmax of [`duels_eval::expected_value`] and the scale of the numbers is
//! irrelevant. `duels-agent-mcts-eval` consumes it as a **value**: every leaf
//! is mapped through a fitted logistic and averaged into a win-rate estimate,
//! so how well the number *predicts the winner* is the whole of its
//! contribution. Those are different objectives and a term can move them in
//! different directions — a denial or menu term that steers a 1-ply agent well
//! says nothing about a position's value, and a mis-scaled term that ranks
//! moves correctly can still be a badly calibrated probability.
//!
//! An arena match measures the policy objective and costs minutes. This
//! measures the value objective and costs about a second per candidate over
//! fifty thousand positions, because a corpus is walked once and each candidate
//! is one `Root::new` plus one `evaluate` per position. It is a **screen**, not
//! a verdict: what it cannot see is anything about which move gets played, so
//! nothing here replaces `duels-arena`.
//!
//! # What is deliberately not claimed
//!
//! The labels are "who won this game, played out by the reference policy from
//! here", so the corpus inherits that policy's blind spots and a candidate that
//! agrees with the reference policy's *mistakes* scores well. `--epsilon`
//! widens the distribution by playing a fraction of moves at random, which is
//! closer to the wilder positions a search visits, at the cost of labels from a
//! weaker continuation. Both readings are worth taking.
//!
//! ```text
//! cargo run --release -p duels-eval --example leaf_probe -- 200
//! cargo run --release -p duels-eval --example leaf_probe -- 200 \
//!     "no yellow:yellow=0" "yellow 1:yellow=1" "yellow 8:yellow=8"
//! cargo run --release -p duels-eval --example leaf_probe -- 200 --epsilon 0.15
//! ```
//!
//! Each extra argument is `label:key=value,...`, applied on top of
//! [`Config::default`]; the bare default is always the first row and is what
//! every other row's `Δ` column is measured against.

use duels_core::scoring::GameResult;
use duels_core::{engine, Action, Observation, Player};
use duels_eval::{evaluate, expected_value, Config, Root};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `PhasedAgent`'s tie window, copied for the same reason `calibrate.rs`
/// copies it: the tie set feeds the RNG draw and so decides which move the
/// reference policy plays.
const TIE_EPSILON: f64 = 1e-6;

/// A verbatim copy of `PhasedAgent::choose`, with an optional exploration rate.
struct Driver {
    rng: StdRng,
    config: Config,
    epsilon: f64,
}

impl Driver {
    fn new(seed: u64, config: Config, epsilon: f64) -> Driver {
        Driver {
            rng: StdRng::seed_from_u64(seed),
            config,
            epsilon,
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        if self.epsilon > 0.0 && self.rng.gen::<f64>() < self.epsilon {
            return legal[self.rng.gen_range(0..legal.len())];
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

/// One corpus entry: a position, whose score is to be read, and the label.
struct Position {
    state: duels_core::GameState,
    /// The player the score is read for.
    me: Player,
    /// Did `me` go on to win?
    won: bool,
    age: u8,
}

/// Mean negative log-likelihood of `(value, won)` pairs under temperature `t`.
fn nll(scored: &[(f64, bool)], t: f64) -> f64 {
    let ln_one_plus_exp = |x: f64| {
        if x > 0.0 {
            x + (-x).exp().ln_1p()
        } else {
            x.exp().ln_1p()
        }
    };
    let mut acc = 0.0;
    for &(value, won) in scored {
        let z = value / t;
        acc += if won {
            ln_one_plus_exp(-z)
        } else {
            ln_one_plus_exp(z)
        };
    }
    acc / scored.len().max(1) as f64
}

/// The maximum-likelihood temperature, by golden-section search on `ln T` —
/// the same one-dimensional fit `calibrate.rs` performs.
fn fit_temperature(scored: &[(f64, bool)]) -> f64 {
    if scored.is_empty() {
        return f64::NAN;
    }
    let (mut lo, mut hi) = (0.05f64.ln(), 400.0f64.ln());
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;
    let mut c = hi - phi * (hi - lo);
    let mut d = lo + phi * (hi - lo);
    let (mut fc, mut fd) = (nll(scored, c.exp()), nll(scored, d.exp()));
    for _ in 0..200 {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - phi * (hi - lo);
            fc = nll(scored, c.exp());
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + phi * (hi - lo);
            fd = nll(scored, d.exp());
        }
        if (hi - lo).abs() < 1e-9 {
            break;
        }
    }
    ((lo + hi) / 2.0).exp()
}

fn sign_accuracy(scored: &[(f64, bool)]) -> f64 {
    let right = scored.iter().filter(|&&(v, w)| (v > 0.0) == w).count();
    right as f64 / scored.len().max(1) as f64
}

/// Apply one `key=value` override to a configuration.
///
/// Only the numeric weights this crate's rounds actually sweep are here; the
/// enum-valued models already have `phased:` spec keys in `duels-arena` and are
/// reachable from an arena match, which is the right instrument for them.
fn apply(cfg: &mut Config, key: &str, raw: &str) -> Result<(), String> {
    if key == "base" {
        *cfg = match raw {
            "v1" => Config::v1(),
            "v2" => Config::v2(),
            "v3" => Config::v3(),
            "v4" => Config::v4(),
            "v5" => Config::v5(),
            "v6" => Config::v6(),
            "v7" | "default" => Config::default(),
            other => return Err(format!("unknown base \"{other}\"")),
        };
        return Ok(());
    }
    if key == "count" {
        cfg.count_pricing = if raw == "0" || raw == "off" || raw == "unpriced" {
            duels_eval::CountPricing::Unpriced
        } else {
            duels_eval::CountPricing::Counted
        };
        return Ok(());
    }
    let v: f64 = raw
        .parse()
        .map_err(|_| format!("\"{raw}\" is not a number (key {key})"))?;
    let e = &mut cfg.eval;
    match key {
        "yellow" => e.yellow_equity = v,
        "discardrate" => e.yellow_discard_rate = v,
        "guildproj" => e.guild_projection = v,
        "band" => e.military_band = v,
        "loot" => e.military_loot = v,
        "urgency" => e.military_endgame_urgency = v,
        "lambda" => e.menu.lambda = v,
        "tau" => e.menu.tau = v,
        "coins_div3" => e.coins_div3 = v,
        "vp" => e.vp_projection = v,
        "dev" => e.development = v,
        "sci" => e.science_ladder = v,
        "bill" => e.resource_bill = v,
        "chaineq" => e.chain_equity = v,
        "wprem" => e.wonder_extra_turn_premium = v,
        "wonder_potential" => e.wonder_potential = v,
        "deny" => e.deny = v,
        "start1" => e.next_age_start[0] = v,
        "start2" => e.next_age_start[1] = v,
        "lockin" => e.production_lock_in = v,
        "beta" => e.coin_smooth_beta = v,
        "cref" => e.coin_smooth_ref = v,
        "raceliq" => e.race_card_liquidity = v,
        "denyboost" => e.deny_opponent_commit_boost = v,
        "kappa" => e.military_sigma_scale = v,
        "logistic" => e.military_logistic_scale = v,
        "start3" => e.next_age_start[2] = v,
        "takerate" => e.development_take_rate = v,
        "coinend" => e.coin_endgame_decisions = v,
        "strongtok" => e.science.strong_token_mult = v,
        "pairshare" => e.science.pair_token_share = v,
        "pairtax" => e.science.pair_tempo_tax = v,
        "dead" => e.science.dead_race_scale = v,
        "tokeneq" => e.token_equity = v,
        "tomove" => e.to_move = v,
        "scale" => e.value_scale = v,
        "pairthreat" => e.science.pair_threat_weight = v,
        "ladder3" => e.science.ladder[3] = v,
        "ladder4" => e.science.ladder[4] = v,
        "ladder5" => e.science.ladder[5] = v,
        other => return Err(format!("unknown key \"{other}\"")),
    }
    Ok(())
}

fn parse_variant(arg: &str) -> Result<(String, Config), String> {
    let (label, params) = arg.split_once(':').unwrap_or((arg, ""));
    let mut cfg = Config::default();
    for kv in params.split(',').filter(|s| !s.trim().is_empty()) {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| format!("\"{kv}\" is not key=value"))?;
        apply(&mut cfg, k.trim(), v.trim())?;
    }
    Ok((label.to_string(), cfg))
}

fn corpus(games: u64, offset: u64, epsilon: f64, policy: Config) -> Vec<Position> {
    let mut out: Vec<Position> = Vec::new();
    for seed in offset..offset + games {
        for swap in [false, true] {
            let seeds = if swap {
                (seed * 1000 + 2, seed * 1000 + 1)
            } else {
                (seed * 1000 + 1, seed * 1000 + 2)
            };
            let mut driver = [
                Driver::new(seeds.0, policy, epsilon),
                Driver::new(seeds.1, policy, epsilon),
            ];
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xF00D);
            let mut trace: Vec<(duels_core::GameState, Player, u8)> = Vec::new();

            while !state.is_over() {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let me = state.current_player();
                trace.push((state, me, state.age()));
                let obs: Observation = state.observation();
                let action = driver[me.index()].choose(&obs, &legal);
                engine::apply_quiet(&mut state, action, &mut rng)
                    .expect("the driver plays legally");
            }

            if let Some(GameResult::Win { winner, .. }) = state.result() {
                for (state, me, age) in trace {
                    out.push(Position {
                        state,
                        me,
                        won: me == winner,
                        age,
                    });
                }
            }
        }
    }
    out
}

fn main() {
    let mut games: u64 = 200;
    let mut offset: u64 = 0;
    let mut epsilon = 0.0f64;
    let mut variants: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1).peekable();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--epsilon" => {
                epsilon = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .expect("--epsilon takes a number");
            }
            "--offset" => {
                offset = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .expect("--offset takes a seed");
            }
            _ => match a.parse::<u64>() {
                Ok(n) if variants.is_empty() => games = n,
                _ => variants.push(a),
            },
        }
    }

    let positions = corpus(games, offset, epsilon, Config::v6());
    println!(
        "{} positions from {games} seeds at offset {offset} x 2 seat orders, \
         reference policy v6, epsilon {epsilon}\n",
        positions.len()
    );

    let mut rows: Vec<(String, Config)> = vec![("default".to_string(), Config::default())];
    for a in &variants {
        match parse_variant(a) {
            Ok(row) => rows.push(row),
            Err(e) => {
                eprintln!("leaf_probe: {e}");
                return;
            }
        }
    }

    println!(
        "  {:<28} {:>7} {:>8} {:>8} {:>7} {:>7} {:>7}",
        "configuration", "T", "NLL", "dNLL", "sign", "sgn-II", "sgn-III"
    );
    let mut base_nll = f64::NAN;
    for (label, cfg) in &rows {
        let scored: Vec<(f64, bool)> = positions
            .iter()
            .map(|p| {
                let root = Root::new(&p.state, p.me, *cfg);
                (evaluate(&p.state, p.me, &root), p.won)
            })
            .collect();
        let by_age = |age: u8| -> f64 {
            let s: Vec<(f64, bool)> = scored
                .iter()
                .zip(&positions)
                .filter(|(_, p)| p.age == age)
                .map(|(&s, _)| s)
                .collect();
            sign_accuracy(&s)
        };
        let t = fit_temperature(&scored);
        let n = nll(&scored, t);
        if base_nll.is_nan() {
            base_nll = n;
        }
        println!(
            "  {:<28} {:>7.2} {:>8.5} {:>+8.5} {:>7.4} {:>7.4} {:>7.4}",
            label,
            t,
            n,
            n - base_nll,
            sign_accuracy(&scored),
            by_age(2),
            by_age(3)
        );
    }
    println!(
        "\n  dNLL is against the first row; **lower NLL is better**. A change of\n  \
         0.001 over 50k positions is well outside the corpus noise; 0.0001 is not."
    );
}
