//! Integration tests: `GreedyEvAgent` driving whole games to completion,
//! against `RandomAgent` (a sanity floor) and head-to-head against
//! `duels-agent-greedy` (the real question this crate exists to answer),
//! plus a measurement of the wall-clock cost of the exact chance-outcome
//! enumeration this agent does every decision.
//!
//! These only ever see [`duels_core::Observation`]s and the `legal` actions
//! handed to `choose`, exactly like a real arena run would drive them — see
//! `duels-agent-greedy`'s own tests for the pattern this follows.

use duels_agent_greedy::GreedyAgent;
use duels_agent_greedy_ev::GreedyEvAgent;
use duels_agent_random::RandomAgent;
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, GameResult, Player};
use rand::{rngs::StdRng, SeedableRng};

/// The workspace `clippy.toml` bans wall-clock reads workspace-wide so the
/// rules engine and its agents stay reproducible; this benchmark test is a
/// legitimate exception (there is no way to measure wall-clock cost without
/// reading the wall clock), mirroring `duels-agent-alphabeta`'s and
/// `duels-agent-mcts-uct`'s own `#[allow(...)]`'d clock reads.
#[allow(clippy::disallowed_methods)]
fn now() -> std::time::Instant {
    std::time::Instant::now()
}

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
fn greedy_ev_vs_greedy_ev_plays_full_games_to_completion_across_seeds() {
    for seed in 0..25u64 {
        let one = GreedyEvAgent::new(seed);
        let two = GreedyEvAgent::new(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
        let result = play_full_game(one, two, seed);
        println!("seed {seed}: {result:?}");
    }
}

/// A sanity floor, not a tuned target: `GreedyEvAgent` is a 1-ply heuristic
/// and should convincingly beat an agent that picks uniformly at random. A
/// heuristic agent failing to clear a comfortable margin here almost
/// certainly means a bug in the evaluation function or the chance-expectation
/// plumbing, not bad luck.
///
/// The measured win rate is reported via `println!` for a human to read from
/// `cargo test -- --nocapture`; the hard-coded threshold below is
/// deliberately loose so this does not flake on ordinary seed variance.
#[test]
fn greedy_ev_convincingly_beats_random_over_many_seeded_games() {
    const SEEDS: u64 = 60;
    let mut ev_wins = 0u32;
    let mut random_wins = 0u32;
    let mut draws = 0u32;
    let mut total = 0u32;

    for seed in 0..SEEDS {
        for ev_is_one in [true, false] {
            let result = if ev_is_one {
                play_full_game(
                    GreedyEvAgent::new(seed),
                    RandomAgent::new(seed ^ 0x5EED_5EED),
                    seed,
                )
            } else {
                play_full_game(
                    RandomAgent::new(seed ^ 0x5EED_5EED),
                    GreedyEvAgent::new(seed),
                    seed,
                )
            };
            total += 1;
            match result.winner() {
                Some(Player::One) if ev_is_one => ev_wins += 1,
                Some(Player::Two) if !ev_is_one => ev_wins += 1,
                Some(_) => random_wins += 1,
                None => draws += 1,
            }
        }
    }

    let win_rate = f64::from(ev_wins) / f64::from(total);
    println!(
        "GreedyEvAgent vs RandomAgent over {total} games ({SEEDS} seeds, both seats): \
         greedy-ev {ev_wins} wins, random {random_wins} wins, {draws} draws \
         (greedy-ev win rate = {win_rate:.1}%)",
        win_rate = win_rate * 100.0
    );

    assert!(
        win_rate > 0.6,
        "expected GreedyEvAgent to convincingly beat RandomAgent, got a {win_rate:.1}% win rate \
         over {total} games -- treat this as a bug in the evaluation function or the \
         chance-expectation plumbing, not bad luck",
    );
}

/// The real question this crate exists to answer: does properly respecting
/// uncertainty (exact expectation over chance outcomes) produce a stronger
/// 1-ply heuristic agent than `duels-agent-greedy`'s single-sample approach,
/// given that both use the *same* evaluation weights and terms?
///
/// Seeded and seat-swapped across many games, reported honestly either way —
/// see the crate's module docs and this crate's PR description for the
/// measured result and an explanation of why it came out the way it did.
#[test]
fn greedy_ev_head_to_head_against_greedy() {
    const SEEDS: u64 = 100;
    let mut ev_wins = 0u32;
    let mut greedy_wins = 0u32;
    let mut draws = 0u32;
    let mut total = 0u32;

    for seed in 0..SEEDS {
        for ev_is_one in [true, false] {
            let result = if ev_is_one {
                play_full_game(
                    GreedyEvAgent::new(seed),
                    GreedyAgent::new(seed ^ 0x5EED_5EED),
                    seed,
                )
            } else {
                play_full_game(
                    GreedyAgent::new(seed ^ 0x5EED_5EED),
                    GreedyEvAgent::new(seed),
                    seed,
                )
            };
            total += 1;
            match result.winner() {
                Some(Player::One) if ev_is_one => ev_wins += 1,
                Some(Player::Two) if !ev_is_one => ev_wins += 1,
                Some(_) => greedy_wins += 1,
                None => draws += 1,
            }
        }
    }

    let win_rate = f64::from(ev_wins) / f64::from(total);
    // A rough logistic-Elo read on the same win rate, purely for a human
    // skimming `--nocapture` output; not asserted on.
    let elo_diff = if ev_wins > 0 && greedy_wins > 0 {
        -400.0 * (1.0 / win_rate - 1.0).log10()
    } else {
        f64::NAN
    };
    println!(
        "GreedyEvAgent vs GreedyAgent over {total} games ({SEEDS} seeds, both seats): \
         greedy-ev {ev_wins} wins, greedy {greedy_wins} wins, {draws} draws \
         (greedy-ev win rate = {win_rate:.1}%, ~{elo_diff:+.0} Elo)",
        win_rate = win_rate * 100.0
    );

    // Deliberately not asserting `win_rate > 0.5`: the whole point of this
    // test, per the crate's design brief, is to report the head-to-head
    // result honestly, even if fixing the sampling flaw turns out not to
    // matter much in practice for a 1-ply heuristic. Only guard against a
    // gross regression (the fixed agent losing overwhelmingly), which would
    // indicate an actual bug rather than an interesting negative result.
    assert!(
        win_rate > 0.30,
        "GreedyEvAgent lost overwhelmingly to GreedyAgent ({win_rate:.1}%) -- this looks like a \
         bug in greedy-ev, not a legitimate negative result",
    );
}

/// Measures the wall-clock cost of the exact chance-outcome enumeration this
/// agent does every decision, on real games rather than a hand-built worst
/// case. Reported via `println!`; the loose assertion just guards against a
/// gross performance regression (an accidental exponential blow-up), not a
/// tuned target.
#[test]
fn exact_chance_enumeration_is_fast_enough_for_interactive_use() {
    const GAMES: u64 = 20;
    let mut agent_a = GreedyEvAgent::new(1);
    let mut agent_b = GreedyEvAgent::new(2);
    let mut rng = StdRng::seed_from_u64(0x00C0_FFEE);

    let mut decisions = 0u64;
    let mut total_elapsed = std::time::Duration::ZERO;
    let mut worst_single_decision = std::time::Duration::ZERO;

    for seed in 0..GAMES {
        let mut state = engine::new_game(seed);
        let mut guard = 0u32;
        loop {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let start = now();
            let action = match state.current_player() {
                Player::One => agent_a.choose(&obs, &legal, Budget::Nodes(64)),
                Player::Two => agent_b.choose(&obs, &legal, Budget::Nodes(64)),
            };
            let elapsed = start.elapsed();
            total_elapsed += elapsed;
            worst_single_decision = worst_single_decision.max(elapsed);
            decisions += 1;

            engine::apply(&mut state, action, &mut rng).expect("agent returned a legal action");
            guard += 1;
            assert!(guard < 10_000, "game did not terminate");
        }
    }

    let avg_micros = total_elapsed.as_secs_f64() * 1_000_000.0 / decisions as f64;
    println!(
        "exact chance enumeration over {decisions} decisions across {GAMES} games: \
         avg {avg_micros:.1} us/decision, worst single decision {worst:.2} ms",
        worst = worst_single_decision.as_secs_f64() * 1000.0
    );

    assert!(
        worst_single_decision.as_millis() < 500,
        "a single decision took {worst_single_decision:?}, which is too slow for interactive/ \
         tournament use -- exact enumeration may need the capped-and-renormalised approximation \
         `alphabeta::search::reduced_outcomes` uses",
    );
}
