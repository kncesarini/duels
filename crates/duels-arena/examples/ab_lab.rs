//! Tuning harness for `duels-agent-alphabeta`.
//!
//! Plays paired, seat-swapped matches between two agent *specifications*,
//! where a specification is either a registered agent name or
//! `alphabeta:key=value,...` naming an explicit
//! [`duels_agent_alphabeta::Config`]. That is the thing the `duels-arena`
//! CLI cannot do — it only knows agents by name — and it is what an ablation
//! needs: every number quoted in the `duels-agent-alphabeta` docs was
//! produced by this example.
//!
//! ```text
//! cargo run --release -p duels-arena --example ab_lab -- \
//!     --a alphabeta:rollouts=8,blend=0.9 --b mcts-uct \
//!     --games 100 --budget time_ms:20 --seed 1
//! ```
//!
//! Flags: `--a`/`--b` (specifications), `--games`, `--seed`, `--budget`, and
//! `--budget-a`/`--budget-b` to give the two sides *different* budgets — how
//! much more time one agent needs to match the other is often the only honest
//! way to state a gap. `--stats 1` adds a short second pass reporting the
//! depth and work the alpha-beta side actually reached.
//!
//! Configuration keys, all optional: `base` (`v1` or `default`), `depth`,
//! `cap` (chance-outcome cap), `rollouts`, `cap-rollouts`, `blend`, `metric`
//! (`margin:<clamp>` or `outcome:<scale>`), `policy` (`uniform`/`biased`),
//! `greedy`, `crn`, `order` (`static`/`none`/`lookahead`), `tt`, `star1`,
//! `weights` (`v1`/`default`/`score-only`), and the individual evaluation
//! weights `card`, `coin`, `breadth`, `shield`, `threat`.
//!
//! Games run in parallel across seeds, so a wall-clock budget is contended
//! the same way for both sides but the absolute work per decision is lower
//! than it would be in a serial run. Comparisons stay fair; absolute node
//! counts do not transfer.

use std::collections::HashMap;

use duels_agent_alphabeta::{eval, playout, AlphaBetaAgent, Config};
use duels_agents_api::{Agent, Budget};
use duels_arena::match_runner::parse_budget;
use duels_core::{engine, Player};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

/// Build an agent from a specification string.
fn make(spec: &str, seed: u64) -> Box<dyn Agent + Send> {
    match spec.split_once(':') {
        Some(("alphabeta", params)) => Box::new(AlphaBetaAgent::with_config(Config {
            seed,
            ..parse_config(params)
        })),
        _ => duels_arena::agent_registry::make_agent(spec, seed).unwrap(),
    }
}

fn parse_config(params: &str) -> Config {
    let mut cfg = Config::default();
    let mut w = cfg.weights;
    for kv in params.split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').expect("key=value");
        match k {
            "base" => match v {
                "v1" => {
                    cfg = Config::v1();
                    w = cfg.weights;
                }
                "default" => {}
                other => panic!("unknown base {other}"),
            },
            "depth" => cfg.max_depth = v.parse().unwrap(),
            "cap" => cfg.chance_cap = v.parse().unwrap(),
            "rollouts" => cfg.rollouts = v.parse().unwrap(),
            "blend" => cfg.rollout_blend = v.parse().unwrap(),
            "policy" => {
                cfg.rollout_policy = match v {
                    "uniform" => playout::PolicyWeights::UNIFORM,
                    "biased" => playout::PolicyWeights::BIASED,
                    other => panic!("unknown policy {other}"),
                }
            }
            "order" => {
                cfg.order_moves = v != "none";
                cfg.order_lookahead = v == "lookahead";
            }
            "tt" => cfg.use_tt = v.parse().unwrap(),
            "star1" => cfg.star1 = v.parse().unwrap(),
            "weights" => {
                w = match v {
                    "score-only" => eval::Weights::SCORE_ONLY,
                    "v1" => eval::Weights::V1,
                    "default" => eval::Weights::DEFAULT,
                    other => panic!("unknown weights {other}"),
                }
            }
            "metric" => {
                cfg.rollout_metric = match v.split_once(':') {
                    Some(("margin", c)) => playout::Metric::Margin {
                        clamp: c.parse().unwrap(),
                    },
                    Some(("outcome", c)) => playout::Metric::Outcome {
                        scale: c.parse().unwrap(),
                    },
                    _ => panic!("metric=margin:<clamp> or metric=outcome:<scale>"),
                }
            }
            "cap-rollouts" => cfg.rollout_cap = v.parse().unwrap(),
            "crn" => cfg.rollout_common_seed = v.parse().unwrap(),
            "greedy" => cfg.rollout_policy.greedy = v.parse().unwrap(),
            "card" => w.card_in_city = v.parse().unwrap(),
            "coin" => w.coin = v.parse().unwrap(),
            "breadth" => w.resource_breadth = v.parse().unwrap(),
            "shield" => w.shield = v.parse().unwrap(),
            "threat" => w.capital_threat = v.parse().unwrap(),
            other => panic!("unknown key {other}"),
        }
    }
    cfg.weights = w;
    cfg
}

/// Play one game and return the winner, the move count and the per-agent
/// wall-clock time in microseconds.
fn play(
    seat_one: &str,
    seat_two: &str,
    one_seed: u64,
    two_seed: u64,
    setup: u64,
    budgets: [Budget; 2],
) -> (Option<Player>, u32, [u128; 2]) {
    let mut one = make(seat_one, one_seed);
    let mut two = make(seat_two, two_seed);
    let mut state = engine::new_game(setup);
    let mut rng = StdRng::seed_from_u64(setup ^ 0x9E37_79B9_7F4A_7C15);
    let mut micros = [0u128; 2];
    let mut moves = 0;
    while !state.is_over() {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let obs = state.observation();
        let p = state.current_player();
        #[allow(clippy::disallowed_methods)]
        let t = std::time::Instant::now();
        let action = match p {
            Player::One => one.choose(&obs, &legal, budgets[0]),
            Player::Two => two.choose(&obs, &legal, budgets[1]),
        };
        #[allow(clippy::disallowed_methods)]
        {
            micros[p.index()] += t.elapsed().as_micros();
        }
        engine::apply(&mut state, action, &mut rng).unwrap();
        moves += 1;
    }
    (state.result().and_then(|r| r.winner()), moves, micros)
}

/// Play a few games with a concrete [`AlphaBetaAgent`] and report what the
/// search actually reached.
fn stats(spec: &str, opponent: &str, budget: Budget, seeds: u64, seed0: u64) {
    let mut decisions = 0u64;
    let mut nodes = 0u64;
    let mut depth_sum = 0u64;
    let mut max_depth = 0u8;
    for k in 0..seeds {
        let s = seed0 + k;
        let cfg = Config {
            seed: s,
            ..parse_config(spec.split_once(':').map_or("", |(_, p)| p))
        };
        let mut ab = AlphaBetaAgent::with_config(cfg);
        let mut opp = duels_arena::agent_registry::make_agent(opponent, s ^ 0xBEEF).unwrap();
        let mut state = engine::new_game(s);
        let mut rng = StdRng::seed_from_u64(s ^ 0x9E37);
        while !state.is_over() {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = if state.current_player() == Player::One {
                ab.choose(&obs, &legal, budget)
            } else {
                opp.choose(&obs, &legal, budget)
            };
            engine::apply(&mut state, action, &mut rng).unwrap();
        }
        let st = ab.stats();
        decisions += st.decisions;
        nodes += st.nodes;
        depth_sum += st.depth_sum;
        max_depth = max_depth.max(st.max_depth_reached);
    }
    println!(
        "  stats over {seeds} games: {:.0} nodes/decision, mean completed depth {:.2}, max {max_depth}",
        nodes as f64 / decisions.max(1) as f64,
        depth_sum as f64 / decisions.max(1) as f64,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut flags: HashMap<&str, &str> = HashMap::new();
    let mut i = 0;
    while i + 1 < args.len() {
        flags.insert(args[i].trim_start_matches("--"), &args[i + 1]);
        i += 2;
    }
    let a = *flags.get("a").unwrap_or(&"alphabeta");
    let b = *flags.get("b").unwrap_or(&"mcts-uct");
    let games: u64 = flags.get("games").unwrap_or(&"100").parse().unwrap();
    let budget = parse_budget(flags.get("budget").unwrap_or(&"nodes:2000")).unwrap();
    let budget_a = flags
        .get("budget-a")
        .map_or(budget, |s| parse_budget(s).unwrap());
    let budget_b = flags
        .get("budget-b")
        .map_or(budget, |s| parse_budget(s).unwrap());
    let seed: u64 = flags.get("seed").unwrap_or(&"1").parse().unwrap();

    let pairs = (games / 2).max(1);
    let out: Vec<(u32, u32, u32, u32, u128, u128)> = (0..pairs)
        .into_par_iter()
        .map(|k| {
            let s = seed + k;
            let a_seed = s ^ 0xA011_7A9E_5B21_0001;
            let b_seed = s ^ 0xB022_8C3F_6D42_0002;
            let (mut aw, mut bw, mut dr, mut mv) = (0, 0, 0, 0);
            let (mut a_us, mut b_us) = (0u128, 0u128);
            // A as seat one, then A as seat two.
            let (w, m, us) = play(a, b, a_seed, b_seed, s, [budget_a, budget_b]);
            mv += m;
            a_us += us[0];
            b_us += us[1];
            match w {
                Some(Player::One) => aw += 1,
                Some(Player::Two) => bw += 1,
                None => dr += 1,
            }
            let (w, m, us) = play(b, a, b_seed, a_seed, s, [budget_b, budget_a]);
            mv += m;
            a_us += us[1];
            b_us += us[0];
            match w {
                Some(Player::Two) => aw += 1,
                Some(Player::One) => bw += 1,
                None => dr += 1,
            }
            (aw, bw, dr, mv, a_us, b_us)
        })
        .collect();

    let (mut aw, mut bw, mut dr, mut mv) = (0u32, 0u32, 0u32, 0u32);
    let (mut a_us, mut b_us) = (0u128, 0u128);
    for (x, y, z, m, au, bu) in out {
        aw += x;
        bw += y;
        dr += z;
        mv += m;
        a_us += au;
        b_us += bu;
    }
    let total = aw + bw + dr;
    println!(
        "{a} ({budget_a:?})  vs  {b} ({budget_b:?})   seeds {seed}..{}",
        seed + pairs
    );
    let n = f64::from(total.max(1));
    let rate = (f64::from(aw) + 0.5 * f64::from(dr)) / n;
    // Binomial standard error on the score rate, so a 3-point difference over
    // 100 games is not mistaken for a finding.
    let stderr = (rate * (1.0 - rate) / n).sqrt();
    println!(
        "  {aw}W / {bw}L / {dr}D over {total} games = {:.1}% +/- {:.1} for A   \
         (A {:.1} ms/game, B {:.1} ms/game, {:.1} moves/game)",
        100.0 * rate,
        100.0 * stderr,
        a_us as f64 / 1000.0 / n,
        b_us as f64 / 1000.0 / n,
        f64::from(mv) / n,
    );
    if flags.contains_key("stats") && a.starts_with("alphabeta") {
        stats(a, b, budget_a, 4, seed);
    }
}
