//! What does each [`LeafValue`] cost, in simulations per second?
//!
//! A leaf value is paid once per simulation, so its cost lands directly on
//! how much search a wall-clock budget buys. This is the number that decides
//! whether a `Nodes` gain survives at `TimeMs`, and this crate has been
//! burned by that trade more than once (see the `Config::prior` section of the
//! crate docs, where a 6-8% throughput cost turned a `+11.7` Elo point
//! estimate into a measured `-33`).
//!
//! Measured under `Budget::Nodes`, where the *work* is exactly fixed — the
//! same number of simulations on every candidate — so only the elapsed time
//! varies. That is the cleaner instrument on a machine that is not perfectly
//! quiet; `rollout_bench.rs` explains the choice at length.
//!
//! The positions are `rollout_bench.rs`'s: a short deterministic random prefix
//! per seed, so the mix of ages and legal-move counts is realistic and does
//! not depend on the candidate under test.
//!
//! ```text
//! cargo run --release -p duels-agent-mcts-uct --example leaf_bench
//! cargo run --release -p duels-agent-mcts-uct --example leaf_bench -- 60 2000
//! ```

use duels_agent_mcts_uct::{Config, LeafValue, MctsAgent};
use duels_agents_api::{Agent, Budget};
use duels_core::engine;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The candidates, in the order the table prints them.
fn candidates() -> Vec<(&'static str, LeafValue)> {
    vec![
        ("rollout (default)", LeafValue::Rollout),
        ("static", LeafValue::Static),
        ("truncated:4", LeafValue::Truncated { plies: 4 }),
        ("truncated:8", LeafValue::Truncated { plies: 8 }),
        ("truncated:16", LeafValue::Truncated { plies: 16 }),
        ("blend:0.3", LeafValue::Blend { weight: 0.3 }),
        ("blend:0.5", LeafValue::Blend { weight: 0.5 }),
    ]
}

/// One position per `p`: a fresh game walked a short deterministic prefix.
fn position(p: u32) -> Option<(duels_core::GameState, Vec<duels_core::Action>)> {
    let seed = 1000 + u64::from(p);
    let mut state = engine::new_game(seed);
    let mut walk = StdRng::seed_from_u64(seed ^ 0xC0FFEE);
    for _ in 0..(p % 15) {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let a = legal[walk.gen_range(0..legal.len())];
        engine::apply_unchecked(&mut state, a, &mut walk);
    }
    let legal = engine::legal_actions(&state);
    (!legal.is_empty()).then_some((state, legal))
}

/// Simulations per second for `leaf` at `Budget::Nodes(nodes)`, over
/// `positions` distinct positions. Also returns the total simulations, as the
/// check that every candidate really did the same amount of work.
fn throughput(leaf: LeafValue, positions: u32, nodes: u64) -> (f64, u64) {
    let cfg = Config {
        leaf,
        ..Config::default()
    };
    let mut total = 0u64;
    #[allow(clippy::disallowed_methods)]
    let start = std::time::Instant::now();
    for p in 0..positions {
        let Some((state, legal)) = position(p) else {
            continue;
        };
        let mut agent = MctsAgent::with_config(u64::from(p) ^ 0xA6E17, cfg);
        agent.choose(&state.observation(), &legal, Budget::Nodes(nodes));
        total += agent.total_simulations();
    }
    let elapsed = start.elapsed().as_secs_f64();
    (total as f64 / elapsed, total)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let positions: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let nodes: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2_000);

    println!("leaf-value throughput: {positions} positions, Budget::Nodes({nodes}) per decision\n");
    println!(
        "  {:<20}  {:>12}  {:>10}  {:>9}  {:>10}",
        "leaf", "sims/s", "us/sim", "vs default", "sims"
    );

    let mut baseline = None::<f64>;
    for (name, leaf) in candidates() {
        let (rate, sims) = throughput(leaf, positions, nodes);
        let base = *baseline.get_or_insert(rate);
        println!(
            "  {:<20}  {:>12.0}  {:>10.3}  {:>8.2}x  {:>10}",
            name,
            rate,
            1e6 / rate,
            rate / base,
            sims
        );
    }
    println!(
        "\n(A `Nodes` budget fixes the work, so the sims column must be identical\n\
         across candidates; the us/sim column is the cost of the leaf value plus\n\
         the descent it shares with every other candidate.)"
    );
}
