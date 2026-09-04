//! Measurement harness for root-determinization ensembling.
//!
//! Plays paired, seat-swapped matches between two *configurations* of the
//! same search agent, which is what the `duels-arena` CLI cannot do (it knows
//! agents only by name) and what this particular question needs: every cell of
//! the `N` sweep is one agent against itself at a different
//! `root_determinizations`, at a matched total budget.
//!
//! ```text
//! cargo run --release -p duels-arena --example ensemble_lab -- \
//!     --a mcts-uct:dets=4 --b mcts-uct:dets=1 \
//!     --games 200 --budget nodes:2000 --seed 1 [--threads 4]
//! ```
//!
//! Specifications:
//!
//! - `mcts-uct[:dets=N,alpha=F,c=F]`
//! - `alphabeta[:dets=N,exact=BOOL,rollouts=N,cap-rollouts=N,depth=N]`
//! - any registered agent name (`random`, `greedy`, ...)
//!
//! `--cost N` skips the match and reports, for each of the two
//! specifications, how much search a decision actually got — the check that a
//! wall-clock budget split `N` ways is not simply buying less search.
//!
//! `--threads` caps the rayon pool. **Use it for a `time_ms` budget**: games
//! run in parallel across seeds, so on a machine already busy with `P` cores'
//! worth of work a wall-clock budget buys less real search than it would in a
//! quiet serial run. Both sides of a game are contended identically — they
//! alternate inside one thread — so the comparison stays fair, but the
//! *operating point* moves, and a technique whose value depends on the budget
//! then gets measured at the wrong budget. Check `uptime` first, leave cores
//! spare, and report the achieved ms/game the harness prints.

use std::collections::HashMap;

use duels_agent_alphabeta::AlphaBetaAgent;
use duels_agent_mcts_uct::MctsAgent;
use duels_agents_api::{Agent, Budget};
use duels_arena::match_runner::parse_budget;
use duels_core::{engine, Player};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

/// Build an agent from a specification string.
fn make(spec: &str, seed: u64) -> Box<dyn Agent + Send> {
    let (name, params) = match spec.split_once(':') {
        Some((name, params)) => (name, params),
        None => (spec, ""),
    };
    match name {
        "mcts-uct" => Box::new(MctsAgent::with_config(seed, mcts_config(params))),
        "alphabeta" => Box::new(AlphaBetaAgent::with_config(duels_agent_alphabeta::Config {
            seed,
            ..alphabeta_config(params)
        })),
        _ => duels_arena::agent_registry::make_agent(spec, seed).unwrap(),
    }
}

fn mcts_config(params: &str) -> duels_agent_mcts_uct::Config {
    let mut cfg = duels_agent_mcts_uct::Config::default();
    for kv in params.split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').expect("key=value");
        match k {
            "dets" => cfg.root_determinizations = v.parse().unwrap(),
            "alpha" => cfg.chance_widen_alpha = v.parse().unwrap(),
            "c" => cfg.exploration = v.parse().unwrap(),
            other => panic!("unknown mcts-uct key {other}"),
        }
    }
    cfg
}

fn alphabeta_config(params: &str) -> duels_agent_alphabeta::Config {
    let mut cfg = duels_agent_alphabeta::Config::default();
    for kv in params.split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').expect("key=value");
        match k {
            "dets" => cfg.root_determinizations = v.parse().unwrap(),
            "exact" => cfg.ensemble_exact_root = v.parse().unwrap(),
            "rollouts" => cfg.rollouts = v.parse().unwrap(),
            "cap-rollouts" => cfg.rollout_cap = v.parse().unwrap(),
            "depth" => cfg.max_depth = v.parse().unwrap(),
            other => panic!("unknown alphabeta key {other}"),
        }
    }
    cfg
}

/// Play one game and return the winner, the move count and the per-seat
/// wall-clock time in microseconds.
fn play(
    seat_one: &str,
    seat_two: &str,
    one_seed: u64,
    two_seed: u64,
    setup: u64,
    budget: Budget,
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
            Player::One => one.choose(&obs, &legal, budget),
            Player::Two => two.choose(&obs, &legal, budget),
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

/// How much *search* one configuration actually gets per decision, playing
/// `seeds` serial games against `random` from seat one.
///
/// The point of the exercise: a wall-clock budget split `N` ways should buy
/// about as many simulations in total as one undivided search, or the win
/// rates below are measuring lost throughput rather than the technique. For
/// `mcts-uct` the unit is one playout; for `alphabeta` it is the crate's own
/// node counter (a decision node or a leaf playout), which is not comparable
/// across the two agents but is comparable across `N`.
///
/// For `alphabeta` it also reports the mean completed depth and the mean
/// playouts per leaf the sampling ramp reached, since a search given `1/N` of
/// the budget can lose its strength to a collapsed ramp rather than to the
/// ensembling itself.
fn cost(spec: &str, budget: Budget, seeds: u64, seed0: u64) {
    let (name, params) = spec.split_once(':').unwrap_or((spec, ""));
    if name != "mcts-uct" && name != "alphabeta" {
        println!("  {spec}: not a search agent, nothing to count");
        return;
    }
    let mut work = 0u64;
    let mut decisions = 0u64;
    let mut depth_sum = 0f64;
    let mut samples = 0u64;
    let mut searches = 0u64;
    for k in 0..seeds {
        let s = seed0 + k;
        let mut opponent = duels_arena::agent_registry::make_agent("random", s ^ 0xBEEF).unwrap();
        let mut mcts = (name == "mcts-uct").then(|| MctsAgent::with_config(s, mcts_config(params)));
        let mut ab = (name == "alphabeta").then(|| {
            AlphaBetaAgent::with_config(duels_agent_alphabeta::Config {
                seed: s,
                ..alphabeta_config(params)
            })
        });
        let mut state = engine::new_game(s);
        let mut rng = StdRng::seed_from_u64(s ^ 0x9E37);
        while !state.is_over() {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = if state.current_player() == Player::One {
                if legal.len() > 1 {
                    decisions += 1;
                }
                match (&mut mcts, &mut ab) {
                    (Some(m), _) => m.choose(&obs, &legal, budget),
                    (_, Some(x)) => {
                        let a = x.choose(&obs, &legal, budget);
                        if let Some(r) = x.last_search() {
                            samples += u64::from(r.samples);
                            searches += 1;
                        }
                        a
                    }
                    _ => unreachable!("the specification was checked above"),
                }
            } else {
                opponent.choose(&obs, &legal, budget)
            };
            engine::apply(&mut state, action, &mut rng).unwrap();
        }
        work += match (&mcts, &ab) {
            (Some(m), _) => m.total_simulations(),
            (_, Some(x)) => x.stats().nodes,
            _ => 0,
        };
        if let Some(x) = &ab {
            depth_sum += x.stats().mean_depth() * x.stats().decisions as f64;
        }
    }
    print!(
        "  {spec} at {budget:?}: {:.0} units of search per decision over {decisions} decisions",
        work as f64 / decisions.max(1) as f64,
    );
    if searches > 0 {
        print!(
            ", mean depth {:.2}, mean playouts/leaf {:.1}",
            depth_sum / decisions.max(1) as f64,
            samples as f64 / searches as f64,
        );
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut flags: HashMap<&str, &str> = HashMap::new();
    let mut i = 0;
    while i + 1 < args.len() {
        flags.insert(args[i].trim_start_matches("--"), &args[i + 1]);
        i += 2;
    }
    let a = *flags.get("a").unwrap_or(&"mcts-uct:dets=2");
    let b = *flags.get("b").unwrap_or(&"mcts-uct:dets=1");
    let games: u64 = flags.get("games").unwrap_or(&"200").parse().unwrap();
    let budget = parse_budget(flags.get("budget").unwrap_or(&"nodes:2000")).unwrap();
    let seed: u64 = flags.get("seed").unwrap_or(&"1").parse().unwrap();
    if let Some(threads) = flags.get("threads") {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.parse().unwrap())
            .build_global()
            .expect("the pool is configured once");
    }

    // `--cost N` measures throughput instead of playing a match: N serial
    // games per specification, reporting the search each one actually got.
    if let Some(seeds) = flags.get("cost") {
        let seeds: u64 = seeds.parse().unwrap();
        println!("search per decision at {budget:?} (serial, {seeds} games each)");
        cost(a, budget, seeds, seed);
        cost(b, budget, seeds, seed);
        return;
    }

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
            let (w, m, us) = play(a, b, a_seed, b_seed, s, budget);
            mv += m;
            a_us += us[0];
            b_us += us[1];
            match w {
                Some(Player::One) => aw += 1,
                Some(Player::Two) => bw += 1,
                None => dr += 1,
            }
            let (w, m, us) = play(b, a, b_seed, a_seed, s, budget);
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
    let n = f64::from(total.max(1));
    let rate = (f64::from(aw) + 0.5 * f64::from(dr)) / n;
    // Binomial standard error on the score rate, so a 3-point difference over
    // 100 games is not mistaken for a finding.
    let stderr = (rate * (1.0 - rate) / n).sqrt();
    println!("{a}  vs  {b}   {budget:?}   seeds {seed}..{}", seed + pairs);
    println!(
        "  {aw}W / {bw}L / {dr}D over {total} games = {:.1}% +/- {:.1} for A   \
         (A {:.1} ms/game, B {:.1} ms/game, {:.1} moves/game)",
        100.0 * rate,
        100.0 * stderr,
        a_us as f64 / 1000.0 / n,
        b_us as f64 / 1000.0 / n,
        f64::from(mv) / n,
    );
}
