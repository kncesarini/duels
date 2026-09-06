//! What does one decision cost, and how much of that is the opponent-menu
//! term?
//!
//! [`duels_agent_phased::menu::menu_term`] is the first term in this crate
//! whose cost scales with the number of *chance outcomes* an action has, not
//! merely with the number of candidate actions. Every other term reads the
//! post-outcome state with a handful of table lookups; this one re-prices the
//! accessible slots against the next mover's purse, which means a call into
//! [`duels_core::cost::card_cost`] per accessible slot per outcome. Age I's
//! worst case is a two-slot reveal from an eleven-card unknown pool, which is
//! over a hundred outcomes for a single candidate action.
//!
//! So the cost is worth measuring rather than assuming.
//!
//! # Why every configuration is timed on the *same* positions
//!
//! The obvious version of this benchmark — play a self-play game under each
//! configuration and divide total time by decisions — measures the wrong
//! thing, and misleadingly: different configurations play different games, and
//! a configuration that steers towards positions with fewer chance outcomes
//! comes out "faster" while doing more work per decision. (Written that way,
//! this benchmark reported the full default as 15% *cheaper* than the same
//! agent with the menu term switched off.) So one policy drives the game and
//! every configuration is timed on the identical position.
//!
//! ```text
//! cargo run --release -p duels-agent-phased --example decision_cost
//! cargo run --release -p duels-agent-phased --example decision_cost -- 40
//! ```

use duels_agent_phased::{Config, EvalWeights, MenuWeights, PhasedAgent};
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, Observation};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Duration;

/// One configuration's running cost.
struct Row {
    name: &'static str,
    config: Config,
    total: Duration,
    age_one: Duration,
    decisions: u64,
    age_one_decisions: u64,
}

fn main() {
    let games: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let menu_off = Config {
        eval: EvalWeights {
            menu: MenuWeights {
                lambda: 0.0,
                ..Config::default().eval.menu
            },
            ..Config::default().eval
        },
        ..Config::default()
    };
    let no_chain = Config {
        eval: EvalWeights {
            chain_equity: 0.0,
            ..menu_off.eval
        },
        ..menu_off
    };

    let mut rows: Vec<Row> = [
        ("v1 (the previous agent)", Config::v1()),
        ("default, menu and chain equity off", no_chain),
        ("default, menu off", menu_off),
        ("default (menu lambda = 0.6)", Config::default()),
    ]
    .into_iter()
    .map(|(name, config)| Row {
        name,
        config,
        total: Duration::ZERO,
        age_one: Duration::ZERO,
        decisions: 0,
        age_one_decisions: 0,
    })
    .collect();

    for seed in 0..games {
        // One policy drives the game, so every configuration below is timed
        // on exactly the same sequence of positions.
        let mut driver = [
            PhasedAgent::new(seed * 1000 + 1),
            PhasedAgent::new(seed * 1000 + 2),
        ];
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xF00D);
        while !state.is_over() {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs: Observation = state.observation();
            let me = state.current_player();
            let in_age_one = state.age() <= 1;

            for row in rows.iter_mut() {
                let mut agent = PhasedAgent::with_config(seed * 97 + 13, row.config);
                // The arena and the benchmarks are the crates allowed to read
                // the wall clock; this is an `examples/` diagnostic doing the
                // same, and nothing it measures feeds a rules decision.
                #[allow(clippy::disallowed_methods)]
                let start = std::time::Instant::now();
                let chosen = agent.choose(&obs, &legal, Budget::Nodes(1));
                #[allow(clippy::disallowed_methods)]
                let elapsed = start.elapsed();
                std::hint::black_box(chosen);
                row.total += elapsed;
                row.decisions += 1;
                if in_age_one {
                    row.age_one += elapsed;
                    row.age_one_decisions += 1;
                }
            }

            let action = driver[me.index()].choose(&obs, &legal, Budget::Nodes(1));
            engine::apply_quiet(&mut state, action, &mut rng).expect("agents play legal moves");
        }
    }

    println!(
        "per-decision cost, every configuration timed on the same {} positions \
         from {games} games\n",
        rows[0].decisions
    );
    let us = |d: Duration, n: u64| d.as_secs_f64() * 1e6 / n.max(1) as f64;
    let baseline = us(rows[0].total, rows[0].decisions);
    for row in &rows {
        println!(
            "  {:<36} {:7.1} us/decision ({:+.0}% vs v1)   age I only: {:7.1} us",
            row.name,
            us(row.total, row.decisions),
            100.0 * (us(row.total, row.decisions) / baseline - 1.0),
            us(row.age_one, row.age_one_decisions),
        );
    }
}
