//! Integration tests: the search agent playing whole games against
//! [`RandomAgent`].
//!
//! The cheap ones run in CI and only assert that nothing panics, that every
//! game reaches a [`GameResult`], and that the agent beats a random player by
//! a wide margin at a CI-sized budget. The expensive one — a 50-game match at
//! a production-sized budget — is `#[ignore]`d; run it with:
//!
//! ```text
//! cargo test -p duels-agent-alphabeta --release -- --ignored --nocapture
//! ```

use duels_agent_alphabeta::{AlphaBetaAgent, Config};
use duels_agent_random::RandomAgent;
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, GameResult, Player};
use rand::{rngs::StdRng, SeedableRng};

/// Outcome of one game from the search agent's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Win,
    Loss,
    Draw,
}

struct GameStats {
    outcome: Outcome,
    decisions: u64,
    nodes: u64,
    depth_sum: u64,
    max_depth: u8,
    tt_probes: u64,
    tt_hits: u64,
}

/// Play one full game, `me` controlled by the search agent and the other seat
/// by [`RandomAgent`].
fn play(seed: u64, me: Player, budget: Budget, cfg: Config) -> GameStats {
    let mut search = AlphaBetaAgent::with_config(Config { seed, ..cfg });
    let mut random = RandomAgent::new(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x1234_5678);

    let mut guard = 0u32;
    loop {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let obs = state.observation();
        let action = if state.current_player() == me {
            search.choose(&obs, &legal, budget)
        } else {
            random.choose(&obs, &legal, budget)
        };
        assert!(
            legal.contains(&action),
            "seed {seed}: agent returned an illegal action {action:?}"
        );
        engine::apply(&mut state, action, &mut rng).expect("the action came from legal_actions");

        guard += 1;
        assert!(guard < 10_000, "seed {seed}: game did not terminate");
    }

    let result = state.result().expect("a finished game has a result");
    let outcome = match result {
        GameResult::Draw => Outcome::Draw,
        GameResult::Win { winner, .. } if winner == me => Outcome::Win,
        GameResult::Win { .. } => Outcome::Loss,
    };
    let stats = search.stats();
    let (tt_probes, tt_hits) = search.tt_stats();
    GameStats {
        outcome,
        decisions: stats.decisions,
        nodes: stats.nodes,
        depth_sum: stats.depth_sum,
        max_depth: stats.max_depth_reached,
        tt_probes,
        tt_hits,
    }
}

/// Run a match and print a summary. Returns `(wins, losses, draws)`.
fn run_match(seeds: std::ops::Range<u64>, budget: Budget, cfg: Config) -> (u32, u32, u32) {
    let (mut wins, mut losses, mut draws) = (0, 0, 0);
    let mut decisions = 0u64;
    let mut nodes = 0u64;
    let mut depth_sum = 0u64;
    let mut max_depth = 0u8;
    let mut probes = 0u64;
    let mut hits = 0u64;

    for seed in seeds.clone() {
        // Alternate seats, so a first-player advantage cannot flatter or
        // punish the agent.
        let me = if seed % 2 == 0 {
            Player::One
        } else {
            Player::Two
        };
        let g = play(seed, me, budget, cfg.clone());
        match g.outcome {
            Outcome::Win => wins += 1,
            Outcome::Loss => losses += 1,
            Outcome::Draw => draws += 1,
        }
        decisions += g.decisions;
        nodes += g.nodes;
        depth_sum += g.depth_sum;
        max_depth = max_depth.max(g.max_depth);
        probes += g.tt_probes;
        hits += g.tt_hits;
    }

    let games = seeds.end - seeds.start;
    println!(
        "budget {budget:?}: {wins}W/{losses}L/{draws}D over {games} games \
         ({:.0}% win rate)",
        100.0 * f64::from(wins) / games as f64
    );
    println!(
        "  {decisions} decisions, {nodes} nodes ({:.0}/decision), \
         mean depth {:.2}, max depth {max_depth}, tt {:.1}% of {probes} probes",
        nodes as f64 / decisions.max(1) as f64,
        depth_sum as f64 / decisions.max(1) as f64,
        100.0 * hits as f64 / probes.max(1) as f64,
    );
    (wins, losses, draws)
}

/// The CI test: 24 complete games at a small node budget.
#[test]
fn plays_complete_games_against_random_across_seeds() {
    let (wins, losses, draws) = run_match(0..24, Budget::Nodes(400), Config::default());
    assert_eq!(wins + losses + draws, 24);
    // A search agent that does not comfortably beat a random player has a
    // bug, not bad luck. This is deterministic (node budget, seeded), so the
    // bar can be tight without being flaky.
    assert!(
        wins >= 20,
        "only won {wins}/24 against a random player: expect a correctness bug"
    );
}

/// The same, with a wall-clock budget, which exercises the time-limited
/// iterative-deepening path (the results are not reproducible, so nothing is
/// asserted about them beyond termination).
#[test]
fn plays_complete_games_under_a_time_budget() {
    let (wins, losses, draws) = run_match(100..104, Budget::TimeMs(5), Config::default());
    assert_eq!(wins + losses + draws, 4);
}

/// Every optimisation switched off must still produce a working agent — if
/// this fails but the default configuration passes, the bug is in the
/// pruning or the table rather than in the search.
#[test]
fn plays_complete_games_with_every_optimisation_disabled() {
    let cfg = Config {
        use_tt: false,
        star1: false,
        order_moves: false,
        ..Config::default()
    };
    let (wins, losses, draws) = run_match(200..208, Budget::Nodes(400), cfg);
    assert_eq!(wins + losses + draws, 8);
    assert!(wins >= 6, "only won {wins}/8 with pruning disabled");
}

/// The reported sanity benchmark. Not a CI gate: it takes tens of seconds.
#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture"]
fn benchmark_against_random() {
    for budget in [
        Budget::Nodes(2_000),
        Budget::Nodes(20_000),
        Budget::TimeMs(200),
    ] {
        let (wins, _, _) = run_match(1_000..1_050, budget, Config::default());
        assert!(wins >= 36, "only won {wins}/50 at {budget:?}");
    }
}
