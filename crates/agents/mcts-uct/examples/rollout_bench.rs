//! Measures whether a rollout policy is actually worth its cost.
//!
//! Rollout quality only matters if it survives being measured at a **fixed
//! time budget**: a smarter-but-slower policy runs fewer simulations per
//! second, so a head-to-head at equal *node* count would flatter it. This
//! tool always compares at equal `Budget::TimeMs`, which is what
//! `docs`/the task asked for two things:
//!
//! 1. **Throughput**: simulations/second for each candidate policy, measured
//!    in isolation (one `MctsAgent` searching a handful of real positions,
//!    no opponent).
//! 2. **Head-to-head win rate**: candidate vs. [`RolloutWeights::UNIFORM`]
//!    (the baseline), both sides given the same `Budget::TimeMs`, alternating
//!    seats, over many seeded games.
//!
//! ```text
//! cargo run --release --example rollout_bench -- [games] [time_ms] [seed]
//! ```
//!
//! Defaults to 100 games and a 200ms/move budget, which is the scale the
//! task asked this to be measured at.
//!
//! # A caveat learned the hard way
//!
//! `Budget::TimeMs` reads the wall clock, so on a machine under variable
//! load (other concurrent builds/tests, background agents, ...) the *number
//! of simulations run per move* varies from run to run even with identical
//! seeds — unlike `Budget::Nodes`, which is exactly reproducible. Two
//! back-to-back 40-game runs of `biased (kind-only)` vs. `uniform` here
//! landed at 60% and then 45%, a bigger swing than any rollout-quality
//! difference this tool is trying to detect. Treat any single run's win
//! rate as noisy at this sample size; only take a candidate seriously if it
//! separates from 50% by a wide, repeatable margin (see `RolloutWeights::SMART`'s
//! doc comment in `rollout.rs` for a worked example of a candidate that did
//! *not* clear that bar).

use duels_agent_mcts_uct::{Config, MctsAgent, RolloutWeights};
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// One named candidate to compare.
struct Candidate {
    name: &'static str,
    weights: RolloutWeights,
}

const CANDIDATES: &[Candidate] = &[
    Candidate {
        name: "uniform",
        weights: RolloutWeights::UNIFORM,
    },
    Candidate {
        name: "biased (kind-only)",
        weights: RolloutWeights::BIASED,
    },
    Candidate {
        name: "smart (per-card)",
        weights: RolloutWeights::SMART,
    },
    // A deliberately milder version of `SMART`, to check whether the
    // aggressiveness of the multipliers (rather than the idea itself) is
    // what made `SMART` lose to uniform in the first measurement. Not part
    // of the public API — constructed here ad hoc for this one comparison.
    Candidate {
        name: "smart-mild",
        weights: RolloutWeights {
            build: 4.0,
            wonder: 2.0,
            discard: 1.0,
            chain_free_mult: 2.0,
            new_symbol_mult: 1.2,
            pair_complete_mult: 1.75,
        },
    },
];

/// Simulations/second for `weights`, searching from a handful of distinct,
/// non-trivial positions (not just the fixed opening) so the measurement
/// reflects a realistic mix of tree shapes and legal-move counts.
fn throughput(weights: RolloutWeights, time_ms: u64, positions: u32) -> (f64, u64) {
    let cfg = Config {
        rollout: weights,
        ..Config::default()
    };
    let mut total_sims = 0u64;
    #[allow(clippy::disallowed_methods)]
    let start = std::time::Instant::now();

    for p in 0..positions {
        // Walk a short, deterministic random prefix so each position differs
        // (different legal-move counts, different ages) without depending on
        // the policy under test.
        let seed = 1000 + u64::from(p);
        let mut state = engine::new_game(seed);
        let mut walk_rng = StdRng::seed_from_u64(seed ^ 0xC0FFEE);
        for _ in 0..(p % 15) {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            use rand::Rng;
            let a = legal[walk_rng.gen_range(0..legal.len())];
            engine::apply_unchecked(&mut state, a, &mut walk_rng);
        }
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            continue;
        }
        let mut agent = MctsAgent::with_config(seed ^ 0xA6E17, cfg);
        let obs = state.observation();
        agent.choose(&obs, &legal, Budget::TimeMs(time_ms));
        total_sims += agent.total_simulations();
    }

    let elapsed = start.elapsed().as_secs_f64();
    (total_sims as f64 / elapsed, total_sims)
}

/// Head-to-head: `candidate` (as `MctsAgent`) vs. `RolloutWeights::UNIFORM`
/// (as `MctsAgent`), both given `Budget::TimeMs(time_ms)`, alternating seats
/// over `games` seeded games. Returns (candidate wins, draws, candidate
/// losses).
fn head_to_head(
    candidate: RolloutWeights,
    games: u64,
    time_ms: u64,
    base_seed: u64,
) -> (u32, u32, u32) {
    let cand_cfg = Config {
        rollout: candidate,
        ..Config::default()
    };
    let base_cfg = Config {
        rollout: RolloutWeights::UNIFORM,
        ..Config::default()
    };
    let budget = Budget::TimeMs(time_ms);

    let mut wins = 0u32;
    let mut draws = 0u32;
    let mut losses = 0u32;

    for game in 0..games {
        let seed = base_seed.wrapping_add(game);
        let cand_seat = if game % 2 == 0 {
            Player::One
        } else {
            Player::Two
        };
        let mut cand = MctsAgent::with_config(seed ^ 0x0BAD_1DEA_0BAD_1DEA, cand_cfg);
        let mut base = MctsAgent::with_config(seed ^ 0x1234_5678_9ABC_DEF0, base_cfg);
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);

        loop {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = if state.current_player() == cand_seat {
                cand.choose(&obs, &legal, budget)
            } else {
                base.choose(&obs, &legal, budget)
            };
            engine::apply_quiet(&mut state, action, &mut rng).expect("a legal action");
        }

        let result = state.result().expect("a finished game has a result");
        match result.winner() {
            Some(w) if w == cand_seat => wins += 1,
            Some(_) => losses += 1,
            None => draws += 1,
        }
    }

    (wins, draws, losses)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(100);
    let time_ms: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(200);
    let base_seed: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);

    println!("=== throughput at {time_ms}ms/move (mean over 24 positions) ===");
    for c in CANDIDATES {
        let (sims_per_sec, total_sims) = throughput(c.weights, time_ms, 24);
        println!(
            "  {:<20} {sims_per_sec:>10.0} sims/s   ({total_sims} sims total)",
            c.name
        );
    }

    println!();
    println!("=== head-to-head vs. uniform baseline, {games} games @ {time_ms}ms/move ===");
    for c in CANDIDATES {
        if c.weights == RolloutWeights::UNIFORM {
            continue;
        }
        let (wins, draws, losses) = head_to_head(c.weights, games, time_ms, base_seed);
        let played = games as f64;
        println!(
            "  {:<20} {:.1}% win rate  ({wins}W {draws}D {losses}L of {games})",
            c.name,
            100.0 * f64::from(wins) / played
        );
    }
}
