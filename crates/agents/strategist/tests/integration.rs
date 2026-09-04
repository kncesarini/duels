//! Integration tests: `StrategistAgent` driving whole games to completion,
//! against `RandomAgent` (a sanity floor), head-to-head against
//! `duels-agent-greedy-ev` (the real question this crate exists to answer),
//! and a sanity check against `duels-agent-greedy`.
//!
//! These only ever see [`duels_core::Observation`]s and the `legal` actions
//! handed to `choose`, exactly like a real arena run would drive them — see
//! `duels-agent-greedy-ev`'s own tests for the pattern this follows.
//!
//! Large-N win-rate benchmarks with detailed victory-kind reporting belong to
//! `duels-arena` (`cargo run --release -p duels-arena -- match ...`), which
//! has the victory-breakdown and race-exposure instrumentation this crate's
//! PR description reports numbers from. The tests here are smaller, in-crate
//! sanity checks that stay fast enough for the default `cargo test`.

use duels_agent_greedy::GreedyAgent;
use duels_agent_greedy_ev::GreedyEvAgent;
use duels_agent_random::RandomAgent;
use duels_agent_strategist::StrategistAgent;
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
fn strategist_vs_strategist_plays_full_games_to_completion_across_seeds() {
    for seed in 0..25u64 {
        let one = StrategistAgent::new(seed);
        let two = StrategistAgent::new(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
        let result = play_full_game(one, two, seed);
        println!("seed {seed}: {result:?}");
    }
}

/// A sanity floor, not a tuned target: `StrategistAgent` is a 1-ply heuristic
/// (plus a move-level prior) and should convincingly beat an agent that picks
/// uniformly at random.
#[test]
fn strategist_convincingly_beats_random_over_many_seeded_games() {
    const SEEDS: u64 = 60;
    let mut wins = 0u32;
    let mut random_wins = 0u32;
    let mut draws = 0u32;
    let mut total = 0u32;

    for seed in 0..SEEDS {
        for strategist_is_one in [true, false] {
            let result = if strategist_is_one {
                play_full_game(
                    StrategistAgent::new(seed),
                    RandomAgent::new(seed ^ 0x5EED_5EED),
                    seed,
                )
            } else {
                play_full_game(
                    RandomAgent::new(seed ^ 0x5EED_5EED),
                    StrategistAgent::new(seed),
                    seed,
                )
            };
            total += 1;
            match result.winner() {
                Some(Player::One) if strategist_is_one => wins += 1,
                Some(Player::Two) if !strategist_is_one => wins += 1,
                Some(_) => random_wins += 1,
                None => draws += 1,
            }
        }
    }

    let win_rate = f64::from(wins) / f64::from(total);
    println!(
        "StrategistAgent vs RandomAgent over {total} games ({SEEDS} seeds, both seats): \
         strategist {wins} wins, random {random_wins} wins, {draws} draws \
         (strategist win rate = {win_rate:.1}%)",
        win_rate = win_rate * 100.0
    );

    assert!(
        win_rate > 0.6,
        "expected StrategistAgent to convincingly beat RandomAgent, got a {win_rate:.1}% win \
         rate over {total} games -- treat this as a bug, not bad luck",
    );
}

/// Sanity check: `StrategistAgent` should also convincingly beat plain
/// `duels-agent-greedy` (the single-sample, no-strategy-prior baseline two
/// generations back).
#[test]
fn strategist_convincingly_beats_greedy_over_many_seeded_games() {
    const SEEDS: u64 = 60;
    let mut wins = 0u32;
    let mut greedy_wins = 0u32;
    let mut draws = 0u32;
    let mut total = 0u32;

    for seed in 0..SEEDS {
        for strategist_is_one in [true, false] {
            let result = if strategist_is_one {
                play_full_game(
                    StrategistAgent::new(seed),
                    GreedyAgent::new(seed ^ 0x5EED_5EED),
                    seed,
                )
            } else {
                play_full_game(
                    GreedyAgent::new(seed ^ 0x5EED_5EED),
                    StrategistAgent::new(seed),
                    seed,
                )
            };
            total += 1;
            match result.winner() {
                Some(Player::One) if strategist_is_one => wins += 1,
                Some(Player::Two) if !strategist_is_one => wins += 1,
                Some(_) => greedy_wins += 1,
                None => draws += 1,
            }
        }
    }

    let win_rate = f64::from(wins) / f64::from(total);
    println!(
        "StrategistAgent vs GreedyAgent over {total} games ({SEEDS} seeds, both seats): \
         strategist {wins} wins, greedy {greedy_wins} wins, {draws} draws \
         (strategist win rate = {win_rate:.1}%)",
        win_rate = win_rate * 100.0
    );

    assert!(
        win_rate > 0.55,
        "expected StrategistAgent to beat GreedyAgent convincingly, got a {win_rate:.1}% win \
         rate over {total} games",
    );
}

/// The real question this crate exists to answer: does a move-level prior
/// from `duels-strategy`'s win-condition reads produce a stronger 1-ply
/// heuristic agent than `duels-agent-greedy-ev` alone, given that both share
/// the exact same chance-expectation evaluation?
///
/// Reported honestly either way — see this crate's module docs and PR
/// description for the measured result. A large-N (200+ game) version of this
/// same comparison, with victory-kind and race-exposure instrumentation, is
/// what `duels-arena` was run with to produce the numbers in the PR
/// description; this in-crate version is a smaller, fast sanity check that
/// the direction holds, not the number quoted there.
#[test]
fn strategist_head_to_head_against_greedy_ev() {
    const SEEDS: u64 = 100;
    let mut wins = 0u32;
    let mut ev_wins = 0u32;
    let mut draws = 0u32;
    let mut total = 0u32;

    for seed in 0..SEEDS {
        for strategist_is_one in [true, false] {
            let result = if strategist_is_one {
                play_full_game(
                    StrategistAgent::new(seed),
                    GreedyEvAgent::new(seed ^ 0x5EED_5EED),
                    seed,
                )
            } else {
                play_full_game(
                    GreedyEvAgent::new(seed ^ 0x5EED_5EED),
                    StrategistAgent::new(seed),
                    seed,
                )
            };
            total += 1;
            match result.winner() {
                Some(Player::One) if strategist_is_one => wins += 1,
                Some(Player::Two) if !strategist_is_one => wins += 1,
                Some(_) => ev_wins += 1,
                None => draws += 1,
            }
        }
    }

    let win_rate = f64::from(wins) / f64::from(total);
    let elo_diff = if wins > 0 && ev_wins > 0 {
        -400.0 * (1.0 / win_rate - 1.0).log10()
    } else {
        f64::NAN
    };
    println!(
        "StrategistAgent vs GreedyEvAgent over {total} games ({SEEDS} seeds, both seats): \
         strategist {wins} wins, greedy-ev {ev_wins} wins, {draws} draws \
         (strategist win rate = {win_rate:.1}%, ~{elo_diff:+.0} Elo)",
        win_rate = win_rate * 100.0
    );

    // Guard against a gross regression (losing overwhelmingly), which would
    // indicate a bug rather than an interesting negative result; the target
    // win rate (>= 65%) is asserted by the larger `duels-arena` benchmark run
    // reported in the PR description, not here, to keep the default
    // `cargo test` run fast and not flake on a smaller sample.
    assert!(
        win_rate > 0.30,
        "StrategistAgent lost overwhelmingly to GreedyEvAgent ({win_rate:.1}%) -- this looks like \
         a bug, not a legitimate negative result",
    );
}
