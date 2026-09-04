//! Sanity benchmark: `mcts-uct` against `random` over seeded games.
//!
//! Reports the win rate, the seat split (so a systematic first-player edge is
//! visible), and the achieved throughput in simulations per second and
//! engine plies per second.
//!
//! ```text
//! cargo run --release --example vs_random -- [games] [simulations-per-move] [seed]
//! ```
//!
//! Two environment variables switch the experiment:
//!
//! - `UNIFORM_ROLLOUT=1` replaces the biased playout policy with the plain
//!   uniform-random one, to measure what the bias is worth.
//! - `OPPONENT_SIMS=n` replaces the random opponent with a second `mcts-uct`
//!   at `n` simulations per move. This is the important diagnostic: beating
//!   uniform-random play proves little on its own, whereas a high-budget
//!   search beating a low-budget one shows the tree statistics genuinely
//!   improve with more samples (and an equal-budget match should land near
//!   50%).
//! - `OPPONENT_UNIFORM_ROLLOUT=1` and `CHANCE_ALPHA` /
//!   `OPPONENT_CHANCE_ALPHA` vary the playout policy and the chance-node
//!   widening exponent per side, so an `OPPONENT_SIMS` match at equal budget
//!   becomes a controlled A/B of one design choice. `CHANCE_ALPHA=1` selects
//!   the unbiased, never-re-selecting chance estimator.
//!
//! This is a measurement tool, not a CI gate: `cargo test` keeps a small,
//! fast version of the same match (`beats_random_at_a_small_budget`).

use duels_agent_mcts_uct::{Config, MctsAgent, RolloutWeights};
use duels_agent_random::RandomAgent;
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut args = std::env::args().skip(1);
    let games: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(40);
    let sims: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(2_000);
    let base_seed: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    let uniform = std::env::var("UNIFORM_ROLLOUT").is_ok();
    let opponent_sims: Option<u64> = std::env::var("OPPONENT_SIMS")
        .ok()
        .and_then(|v| v.parse().ok());

    let weights = |uniform: bool| {
        if uniform {
            RolloutWeights::UNIFORM
        } else {
            RolloutWeights::BIASED
        }
    };
    // Chance-node widening: `alpha = 1.0` with a large coefficient is the
    // unbiased "fresh outcome every visit" estimator, so the pair of
    // variables below A/Bs faithfulness against tree depth.
    let widening = |var: &str| -> (f64, f64) {
        match std::env::var(var).ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(alpha) if alpha >= 1.0 => (1e9, 1.0),
            Some(alpha) => (Config::default().chance_widen_c, alpha),
            None => (
                Config::default().chance_widen_c,
                Config::default().chance_widen_alpha,
            ),
        }
    };
    let (c, alpha) = widening("CHANCE_ALPHA");
    let cfg = Config {
        rollout: weights(uniform),
        chance_widen_c: c,
        chance_widen_alpha: alpha,
        ..Config::default()
    };
    // Lets one `OPPONENT_SIMS` match A/B the playout policy or the widening
    // rule at equal budget.
    let (oc, oalpha) = widening("OPPONENT_CHANCE_ALPHA");
    let opponent_cfg = Config {
        rollout: weights(std::env::var("OPPONENT_UNIFORM_ROLLOUT").is_ok()),
        chance_widen_c: oc,
        chance_widen_alpha: oalpha,
        ..Config::default()
    };
    let budget = Budget::Nodes(sims);

    match opponent_sims {
        None => println!("mcts-uct vs random: {games} games, {sims} simulations/move"),
        Some(n) => println!("mcts-uct({sims}) vs mcts-uct({n}): {games} games"),
    }
    println!("config: {}", cfg.describe());
    if opponent_sims.is_some() {
        println!("opponent config: {}", opponent_cfg.describe());
    }

    let mut wins = 0u32;
    let mut draws = 0u32;
    let mut wins_by_seat = [0u32; 2];
    let mut games_by_seat = [0u32; 2];
    let mut total_sims = 0u64;
    let mut total_plies = 0u64;
    let mut total_nodes = 0u64;
    let mut searches = 0u64;

    // The benchmark is allowed to read the clock; the agent itself only does
    // so under `Budget::TimeMs`.
    #[allow(clippy::disallowed_methods)]
    let start = std::time::Instant::now();

    for game in 0..games {
        let seed = base_seed.wrapping_add(game);
        // Alternate seats so a first-player advantage cannot flatter either
        // agent.
        let seat = if game % 2 == 0 {
            Player::One
        } else {
            Player::Two
        };
        let mut mcts = MctsAgent::with_config(seed ^ 0x0BAD_1DEA_0BAD_1DEA, cfg);
        let mut opponent: Box<dyn Agent> = match opponent_sims {
            None => Box::new(RandomAgent::new(seed ^ 0x5EED_5EED)),
            Some(_) => Box::new(MctsAgent::with_config(
                seed ^ 0x1234_5678_9ABC_DEF0,
                opponent_cfg,
            )),
        };
        let opponent_budget = Budget::Nodes(opponent_sims.unwrap_or(1));
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);

        loop {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = if state.current_player() == seat {
                let before = mcts.total_simulations();
                let a = mcts.choose(&obs, &legal, budget);
                if mcts.total_simulations() > before {
                    searches += 1;
                    total_nodes += mcts.last_tree_size() as u64;
                }
                a
            } else {
                opponent.choose(&obs, &legal, opponent_budget)
            };
            engine::apply_quiet(&mut state, action, &mut rng).expect("a legal action");
            total_plies += 1;
        }

        let result = state.result().expect("a finished game has a result");
        total_sims += mcts.total_simulations();
        games_by_seat[seat.index()] += 1;
        match result.winner() {
            Some(w) if w == seat => {
                wins += 1;
                wins_by_seat[seat.index()] += 1;
            }
            Some(_) => {}
            None => draws += 1,
        }
        println!("  seed {seed:>4} mcts={seat} -> {result:?}");
    }

    let elapsed = start.elapsed().as_secs_f64();
    let played = games as f64;
    println!();
    println!(
        "win rate: {:.1}%  ({wins} wins, {draws} draws, {} losses of {games})",
        100.0 * f64::from(wins) / played,
        games as u32 - wins - draws
    );
    for p in Player::ALL {
        let n = games_by_seat[p.index()];
        if n > 0 {
            println!(
                "  as {p}: {:.1}% ({}/{n})",
                100.0 * f64::from(wins_by_seat[p.index()]) / f64::from(n),
                wins_by_seat[p.index()]
            );
        }
    }
    println!("searches: {searches}, total simulations: {total_sims}");
    if searches > 0 {
        println!(
            "mean tree size: {:.0} nodes/search",
            total_nodes as f64 / searches as f64
        );
    }
    println!(
        "throughput: {:.0} simulations/s, {:.0} game plies/s, wall clock {elapsed:.1}s",
        total_sims as f64 / elapsed,
        total_plies as f64 / elapsed
    );
}
