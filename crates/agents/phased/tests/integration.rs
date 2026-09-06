//! Integration tests: `PhasedAgent` driving whole games to completion against
//! `RandomAgent` (a sanity floor) and head-to-head against
//! `duels-agent-greedy-ev` (the reference this crate exists to beat).
//!
//! These only ever see [`duels_core::Observation`]s and the `legal` actions
//! handed to `choose`, exactly as a real arena run would drive them.
//!
//! The real, large-N win-rate measurement with victory-kind and race-exposure
//! breakdowns belongs to `duels-arena`
//! (`cargo run --release -p duels-arena -- match --agent-a phased --agent-b
//! greedy-ev --games 400 --budget nodes:1 --seed 1`), whose numbers this
//! crate's PR description reports. What is here is small enough to stay in the
//! default `cargo test` path, and is a floor rather than the headline: it
//! asserts a margin so wide that only a real regression could break it.

use duels_agent_greedy_ev::GreedyEvAgent;
use duels_agent_phased::PhasedAgent;
use duels_agent_random::RandomAgent;
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, GameResult, Player};
use rand::{rngs::StdRng, SeedableRng};

/// Drive one full game, asserting every move returned is legal and that the
/// game terminates with a result.
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
            Player::One => one.choose(&obs, &legal, Budget::Nodes(1)),
            Player::Two => two.choose(&obs, &legal, Budget::Nodes(1)),
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

/// Paired, seat-swapped win rate for `phased` against one opponent. Seat
/// swapping is not optional in this game: first-player advantage is large
/// even between equally strong agents.
fn paired_win_rate(opponent: &str, seeds: u64) -> f64 {
    let mut wins = 0u32;
    let mut games = 0u32;
    for seed in 0..seeds {
        for phased_is_one in [true, false] {
            let phased = PhasedAgent::new(seed * 4 + u64::from(phased_is_one));
            let other_seed = seed * 4 + 2;
            let result = match (phased_is_one, opponent) {
                (true, "random") => play_full_game(phased, RandomAgent::new(other_seed), seed),
                (false, "random") => play_full_game(RandomAgent::new(other_seed), phased, seed),
                (true, _) => play_full_game(phased, GreedyEvAgent::new(other_seed), seed),
                (false, _) => play_full_game(GreedyEvAgent::new(other_seed), phased, seed),
            };
            games += 1;
            match result {
                GameResult::Win { winner, .. } => {
                    if (winner == Player::One) == phased_is_one {
                        wins += 1;
                    }
                }
                GameResult::Draw => {}
            }
        }
    }
    f64::from(wins) / f64::from(games)
}

#[test]
fn phased_vs_phased_plays_full_games_to_completion_across_seeds() {
    for seed in 0..20u64 {
        let one = PhasedAgent::new(seed);
        let two = PhasedAgent::new(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
        let result = play_full_game(one, two, seed);
        println!("seed {seed}: {result:?}");
    }
}

#[test]
fn phased_convincingly_beats_random() {
    let rate = paired_win_rate("random", 25);
    println!(
        "phased vs random over 50 paired games: {:.1}%",
        rate * 100.0
    );
    assert!(
        rate > 0.85,
        "phased only won {:.1}% against random",
        rate * 100.0
    );
}

/// The head-to-head this crate exists for. `duels-arena` measures this at
/// around 98% over 400 paired games on three disjoint seed ranges; the
/// threshold here is set far below that so it flags a regression rather than
/// ordinary run-to-run noise on a small sample.
#[test]
fn phased_convincingly_beats_greedy_ev() {
    let rate = paired_win_rate("greedy-ev", 25);
    println!(
        "phased vs greedy-ev over 50 paired games: {:.1}%",
        rate * 100.0
    );
    assert!(
        rate > 0.80,
        "phased only won {:.1}% against greedy-ev",
        rate * 100.0
    );
}
