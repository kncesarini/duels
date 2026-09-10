//! What do [`duels_eval::Root::new`] and [`duels_eval::evaluate`] cost, **on
//! their own**?
//!
//! `duels-agent-phased`'s `examples/decision_cost.rs` measures the whole
//! per-decision cost of a 1-ply agent: one `Root`, then one `expected_value`
//! per candidate action, each of which enumerates that action's chance
//! outcomes, applies each to a copy of the state and calls `evaluate` on the
//! result. That number is the right one for "how fast is `phased`". It is the
//! wrong one for "what would it cost a search to call this evaluation at a
//! leaf", which is a single `evaluate` on a state the search already holds,
//! against a `Root` the search would build once per tree node and cache
//! (`docs/conventions.md`: `duels-strategy`'s reads are cheap enough per node,
//! too expensive per simulation — and `Root::new` *is* a slate of those
//! reads).
//!
//! So this benchmark separates the two:
//!
//! * **`Root::new`** — the per-node cost. One `Stance`, two commitment
//!   scalars, the supply statistics, the military smoothing, two `TakeValue`s,
//!   the chain table, the guild table and the menu tables.
//! * **`evaluate`** — the per-leaf cost. Every term read per player and
//!   differenced, plus the menu term, against tables that already exist.
//!
//! # The positions
//!
//! Deliberately **the same position set `decision_cost.rs` uses**, so the two
//! benchmarks' numbers can be put next to each other: seeds `0..games` (30 by
//! default), `engine::new_game(seed)`, one `phased` self-play game per seed
//! driven by two policies seeded `seed * 1000 + 1` and `seed * 1000 + 2`, with
//! the engine's own RNG seeded `seed ^ 0xF00D`. The driver is
//! `PhasedAgent::choose` line for line — this crate sits below every agent
//! crate and cannot call the agent itself.
//!
//! ```text
//! cargo run --release -p duels-eval --example eval_bench
//! cargo run --release -p duels-eval --example eval_bench -- 40
//! ```

use duels_core::{engine, Action, GameState, Observation, Player};
use duels_eval::{evaluate, expected_value, Config, Root};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

/// `PhasedAgent`'s tie window, copied because the tie set feeds the RNG draw
/// and so decides which move comes out — and therefore which positions this
/// benchmark visits.
const TIE_EPSILON: f64 = 1e-6;

/// How many times each timed call is repeated per position. Both calls are
/// well under a microsecond, which is the same order as the clock read around
/// them, so they are timed in batches.
const REPEATS: u32 = 64;

/// A verbatim copy of `PhasedAgent::choose`, RNG usage included, so this
/// benchmark walks exactly the games `decision_cost.rs` walks.
struct Driver {
    rng: StdRng,
    config: Config,
}

impl Driver {
    fn new(seed: u64) -> Driver {
        Driver {
            rng: StdRng::seed_from_u64(seed),
            config: Config::default(),
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

/// One configuration's running totals.
struct Row {
    name: &'static str,
    config: Config,
    root: Duration,
    eval: Duration,
    root_age_one: Duration,
    eval_age_one: Duration,
    positions: u64,
    age_one_positions: u64,
}

impl Row {
    fn new(name: &'static str, config: Config) -> Row {
        Row {
            name,
            config,
            root: Duration::ZERO,
            eval: Duration::ZERO,
            root_age_one: Duration::ZERO,
            eval_age_one: Duration::ZERO,
            positions: 0,
            age_one_positions: 0,
        }
    }

    /// Time `Root::new` and `evaluate` on one position.
    fn sample(&mut self, state: &GameState, me: Player) {
        // The arena and the benchmarks are the crates allowed to read the wall
        // clock; this is an `examples/` diagnostic doing the same, and nothing
        // it measures feeds a rules decision.
        #[allow(clippy::disallowed_methods)]
        let start = std::time::Instant::now();
        for _ in 0..REPEATS {
            std::hint::black_box(Root::new(
                std::hint::black_box(state),
                me,
                std::hint::black_box(self.config),
            ));
        }
        #[allow(clippy::disallowed_methods)]
        let root_time = start.elapsed();

        let root = Root::new(state, me, self.config);
        #[allow(clippy::disallowed_methods)]
        let start = std::time::Instant::now();
        for _ in 0..REPEATS {
            std::hint::black_box(evaluate(
                std::hint::black_box(state),
                me,
                std::hint::black_box(&root),
            ));
        }
        #[allow(clippy::disallowed_methods)]
        let eval_time = start.elapsed();

        self.root += root_time;
        self.eval += eval_time;
        self.positions += 1;
        if state.age() <= 1 {
            self.root_age_one += root_time;
            self.eval_age_one += eval_time;
            self.age_one_positions += 1;
        }
    }
}

fn main() {
    let games: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let mut rows = vec![
        Row::new("default", Config::default()),
        Row::new("v1 (the round-one evaluation)", Config::v1()),
        Row::new("v2 (the round-two evaluation)", Config::v2()),
        Row::new("v3 (the round-three evaluation)", Config::v3()),
        Row::new("v4 (the round-four evaluation)", Config::v4()),
        Row::new("v5 (the round-five evaluation)", Config::v5()),
        Row::new("v6 (the round-six evaluation)", Config::v6()),
        Row::new("v7 (the round-seven evaluation)", Config::v7()),
        // Round nine's two options. The rationed wonder model adds one
        // `wonder_p_build` read per `evaluate` — it used to add two to
        // `Root::new` and cache them, which was the frozen-`p_build` defect;
        // this row is the measurement of what un-freezing it costs, and it is
        // the one place that cost shows up, since `evaluate` runs many times
        // per `Root`. The structural reach model adds a pass over the occupied
        // slots inside the dead-race gate's walk, which is the one that has to
        // be watched — round seven's version of that walk was a +47%
        // regression on `evaluate` before its guards went in.
        Row::new(
            "default + the rationed wonder model",
            Config {
                eval: duels_eval::EvalWeights {
                    wonder_potential: 1.25,
                    ..Config::default().eval
                },
                wonder_model: duels_eval::WonderModel::Rationed,
                ..Config::default()
            },
        ),
        Row::new(
            "default + the structural reach model",
            Config {
                eval: duels_eval::EvalWeights {
                    science: duels_eval::ScienceWeights {
                        reach_model: duels_eval::ReachModel::Structure,
                        ..Config::default().eval.science
                    },
                    ..Config::default().eval
                },
                ..Config::default()
            },
        ),
        Row::new(
            "default + owned-token equity",
            Config {
                eval: duels_eval::EvalWeights {
                    token_equity: 1.0,
                    ..Config::default().eval
                },
                ..Config::default()
            },
        ),
        Row::new(
            "default + count-priced menu",
            Config {
                count_pricing: duels_eval::CountPricing::Counted,
                ..Config::default()
            },
        ),
    ];

    for seed in 0..games {
        // One policy drives the game, exactly as in `decision_cost.rs`, so
        // every configuration below is measured on the identical positions.
        let mut driver = [Driver::new(seed * 1000 + 1), Driver::new(seed * 1000 + 2)];
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xF00D);
        while !state.is_over() {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs: Observation = state.observation();
            let me = state.current_player();

            for row in rows.iter_mut() {
                row.sample(&state, me);
            }

            let action = driver[me.index()].choose(&obs, &legal);
            engine::apply_quiet(&mut state, action, &mut rng).expect("the driver plays legally");
        }
    }

    let us = |d: Duration, n: u64| d.as_secs_f64() * 1e6 / (n.max(1) * u64::from(REPEATS)) as f64;
    println!(
        "Root::new and evaluate() in isolation, {} positions from {games} games, \
         {REPEATS} repeats each\n",
        rows[0].positions
    );
    println!(
        "  {:<32} {:>12} {:>12} {:>12}   {:>12} {:>12}",
        "", "Root::new", "evaluate", "sum", "root (age I)", "eval (age I)"
    );
    for row in &rows {
        println!(
            "  {:<32} {:>9.3} us {:>9.3} us {:>9.3} us   {:>9.3} us {:>9.3} us",
            row.name,
            us(row.root, row.positions),
            us(row.eval, row.positions),
            us(row.root, row.positions) + us(row.eval, row.positions),
            us(row.root_age_one, row.age_one_positions),
            us(row.eval_age_one, row.age_one_positions),
        );
    }
    println!(
        "\n  evaluate() is {:.1}% of one Root::new under the default configuration.",
        100.0 * us(rows[0].eval, rows[0].positions) / us(rows[0].root, rows[0].positions)
    );
}
