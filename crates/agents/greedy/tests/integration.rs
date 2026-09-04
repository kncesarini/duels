//! Integration tests: `GreedyAgent` driving a whole game to completion, both
//! against itself and against `RandomAgent`, plus a sanity benchmark of the
//! resulting win rate.
//!
//! These only ever see [`duels_core::Observation`]s and the `legal` actions
//! handed to `choose`, exactly like a real arena run would drive them — see
//! `duels-agent-random`'s own tests for the pattern this follows.

use duels_agent_greedy::GreedyAgent;
use duels_agent_random::RandomAgent;
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, GameResult, Player};
use rand::{rngs::StdRng, SeedableRng};

/// Drive one full game between two agents, asserting every move they return
/// is legal and that the game terminates with a result.
fn play_full_game<A: Agent, B: Agent>(mut one: A, mut two: B, seed: u64) -> GameResult {
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x1234_5678);

    let mut guard = 0u32;
    loop {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let obs = state.observation();
        let action = match state.current_player() {
            Player::One => one.choose(&obs, &legal, Budget::Nodes(64)),
            Player::Two => two.choose(&obs, &legal, Budget::Nodes(64)),
        };
        assert!(legal.contains(&action), "agent returned an illegal action");
        engine::apply(&mut state, action, &mut rng).expect("agent returned a legal action");

        guard += 1;
        assert!(
            guard < 10_000,
            "game did not terminate after {guard} decisions"
        );
    }
    state.result().expect("a finished game has a result")
}

#[test]
fn greedy_vs_greedy_plays_full_games_to_completion_across_seeds() {
    for seed in 0..25u64 {
        let one = GreedyAgent::new(seed);
        let two = GreedyAgent::new(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
        let result = play_full_game(one, two, seed);
        println!("seed {seed}: {result:?}");
    }
}

#[test]
fn greedy_vs_random_plays_full_games_to_completion_across_seeds_and_seats() {
    for seed in 0..25u64 {
        let greedy_as_one = play_full_game(
            GreedyAgent::new(seed),
            RandomAgent::new(seed ^ 0xDEAD_BEEF),
            seed,
        );
        println!("seed {seed} (greedy=P1): {greedy_as_one:?}");

        let greedy_as_two = play_full_game(
            RandomAgent::new(seed ^ 0xDEAD_BEEF),
            GreedyAgent::new(seed),
            seed,
        );
        println!("seed {seed} (greedy=P2): {greedy_as_two:?}");
    }
}

/// A sanity floor, not a tuned target: `GreedyAgent` is a 1-ply heuristic and
/// should convincingly beat an agent that picks uniformly at random. A
/// heuristic agent failing to clear a comfortable margin here almost
/// certainly means a bug in the evaluation function or the move-simulation
/// plumbing, not bad luck — see the module docs on `duels_agent_greedy` for
/// what `evaluate` is supposed to reward.
///
/// The measured win rate (typically ~90%+) is reported via `println!` for a
/// human to read from `cargo test -- --nocapture`; the hard-coded threshold
/// below is deliberately loose so this does not flake on ordinary seed
/// variance.
#[test]
fn greedy_convincingly_beats_random_over_many_seeded_games() {
    const SEEDS: u64 = 60;
    let mut greedy_wins = 0u32;
    let mut random_wins = 0u32;
    let mut draws = 0u32;
    let mut total = 0u32;

    for seed in 0..SEEDS {
        for greedy_is_one in [true, false] {
            let result = if greedy_is_one {
                play_full_game(
                    GreedyAgent::new(seed),
                    RandomAgent::new(seed ^ 0x5EED_5EED),
                    seed,
                )
            } else {
                play_full_game(
                    RandomAgent::new(seed ^ 0x5EED_5EED),
                    GreedyAgent::new(seed),
                    seed,
                )
            };
            total += 1;
            match result.winner() {
                Some(Player::One) if greedy_is_one => greedy_wins += 1,
                Some(Player::Two) if !greedy_is_one => greedy_wins += 1,
                Some(_) => random_wins += 1,
                None => draws += 1,
            }
        }
    }

    let win_rate = f64::from(greedy_wins) / f64::from(total);
    println!(
        "GreedyAgent vs RandomAgent over {total} games ({SEEDS} seeds, both seats): \
         greedy {greedy_wins} wins, random {random_wins} wins, {draws} draws \
         (greedy win rate = {win_rate:.1}%)",
        win_rate = win_rate * 100.0
    );

    assert!(
        win_rate > 0.6,
        "expected GreedyAgent to convincingly beat RandomAgent, got a {win_rate:.1}% win rate \
         over {total} games -- treat this as a bug in the evaluation function or the move \
         simulation, not bad luck",
    );
}
