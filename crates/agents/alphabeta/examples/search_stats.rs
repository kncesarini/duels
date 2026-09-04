//! Node counts, reached depth and transposition-table hit rate for each
//! combination of the search optimisations, on a handful of positions from a
//! real game.
//!
//! This is the measurement behind the claims in the crate docs about which
//! optimisation is worth what. Run it with:
//!
//! ```text
//! cargo run --release -p duels-agent-alphabeta --example search_stats
//! ```

use duels_agent_alphabeta::{search::Searcher, tt::Table, Config};
use duels_agents_api::Budget;
use duels_core::{engine, GameState};
use rand::{rngs::StdRng, SeedableRng};

/// A position `plies` quasi-random decisions into the game from `seed`.
fn position(seed: u64, plies: u32) -> GameState {
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..plies {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let action = legal[(seed as usize * 7 + state.turn() as usize) % legal.len()];
        engine::apply_quiet(&mut state, action, &mut rng).expect("legal");
    }
    state
}

fn main() {
    let positions = [
        ("draft", engine::new_game(11)),
        ("age1-early", position(4, 10)),
        ("age1-late", position(4, 18)),
        ("age2", position(4, 34)),
        ("age3", position(4, 54)),
    ];

    let variants: [(&str, Config); 5] = [
        (
            "none",
            Config {
                star1: false,
                order_moves: false,
                use_tt: false,
                ..Config::default()
            },
        ),
        (
            "order",
            Config {
                star1: false,
                use_tt: false,
                ..Config::default()
            },
        ),
        (
            "order+star1",
            Config {
                use_tt: false,
                ..Config::default()
            },
        ),
        ("order+star1+tt", Config::default()),
        (
            "lookahead+star1+tt",
            Config {
                order_lookahead: true,
                ..Config::default()
            },
        ),
    ];

    for (label, state) in positions {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            continue;
        }
        println!(
            "== {label}: age {}, turn {}, {} legal moves",
            state.age(),
            state.turn(),
            legal.len()
        );
        for depth in [3u8, 4] {
            for (name, base) in &variants {
                let cfg = Config {
                    max_depth: depth,
                    tt_bits: 18,
                    ..base.clone()
                };
                let mut tt = Table::with_bits(cfg.tt_bits);
                let mut searcher = Searcher::new(
                    state.current_player(),
                    &cfg,
                    &mut tt,
                    Budget::Nodes(u64::MAX),
                );
                let result = searcher.think(&state, &legal);
                let (probes, hits) = tt.stats();
                println!(
                    "   depth {depth}  {name:<19} nodes {:>9}  value {:>7.2}  \
                     tt {:>4.1}% of {probes}",
                    result.nodes,
                    result.value,
                    100.0 * hits as f64 / probes.max(1) as f64,
                );
            }
        }
    }
}
