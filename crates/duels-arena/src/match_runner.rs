//! The core match runner: plays one complete game between two named agents,
//! and pairs up seat-swapped games per seed so first-player advantage and
//! setup randomness roughly cancel across the pair.
//!
//! Mirrors the loop in `duels-server`'s `room::drive_agents`: agents are fed
//! only `Observation`s and `legal_actions`, never a `GameState`, exactly as
//! the `Agent` contract requires.

use std::time::Instant;

use duels_agents_api::{AgentSpec, Budget};
use duels_core::{engine, GameResult, Player};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::agent_registry::make_agent;

/// Salts distinguishing "the agent playing role A" and "the agent playing
/// role B" from the game-setup seed and from each other, so each has its own
/// reproducible RNG stream regardless of which physical seat it ends up in.
const AGENT_A_SALT: u64 = 0xA011_7A9E_5B21_0001;
const AGENT_B_SALT: u64 = 0xB022_8C3F_6D42_0002;

/// Salt for the engine's own mid-game RNG (used only for The Great Library),
/// matching the convention `duels-server::room::Room::new` uses.
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// One played game, self-contained enough to re-derive win/loss for either
/// named agent and to serialize into a results file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameRecord {
    /// The seed this game's setup (`engine::new_game`) was built from.
    pub seed: u64,
    /// Which seat "role A" (the first agent named to [`play_paired_match`])
    /// occupied in this particular game. Comparing this to `result`'s
    /// winner, rather than comparing agent names, is what keeps the
    /// bookkeeping correct in a self-play match (`agent_a == agent_b`).
    pub agent_a_seat: Player,
    /// Spec of the agent that played [`Player::One`].
    pub seat_one: AgentSpec,
    /// Spec of the agent that played [`Player::Two`].
    pub seat_two: AgentSpec,
    /// How the game ended.
    pub result: GameResult,
    /// Number of actions applied over the course of the game.
    pub moves: u32,
    /// Wall-clock time to play the whole game, for basic performance
    /// reporting (not used by any statistic).
    pub wall_time_ms: u64,
}

/// Aggregate win/loss/draw counts for "role A" vs "role B" over a set of
/// [`GameRecord`]s (see `agent_a_seat` for why this isn't just a name
/// comparison).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchTally {
    pub a_wins: u32,
    pub b_wins: u32,
    pub draws: u32,
}

impl MatchTally {
    pub fn total(&self) -> u32 {
        self.a_wins + self.b_wins + self.draws
    }
}

/// Tally [`GameRecord`]s from the perspective of "role A".
pub fn tally(records: &[GameRecord]) -> MatchTally {
    let mut t = MatchTally::default();
    for r in records {
        match r.result.winner() {
            None => t.draws += 1,
            Some(winner) => {
                if winner == r.agent_a_seat {
                    t.a_wins += 1;
                } else {
                    t.b_wins += 1;
                }
            }
        }
    }
    t
}

/// Play one complete game: `seat_one_name`/`seat_two_name` occupy
/// [`Player::One`]/[`Player::Two`], each agent's own decisions seeded from
/// `seat_one_seed`/`seat_two_seed`, and the game setup (deal, shuffle, first
/// player) from `setup_seed`.
///
/// Returns the two agents' specs, the finished game's result, the move
/// count, and wall-clock time. Feeds each agent only `Observation`s and
/// `legal_actions`, never `GameState`, per the `Agent` contract.
#[allow(clippy::too_many_arguments)]
fn play_one_game(
    seat_one_name: &str,
    seat_two_name: &str,
    seat_one_seed: u64,
    seat_two_seed: u64,
    setup_seed: u64,
    budget: Budget,
) -> Result<(AgentSpec, AgentSpec, GameResult, u32, u64), String> {
    let mut agent_one = make_agent(seat_one_name, seat_one_seed)?;
    let mut agent_two = make_agent(seat_two_name, seat_two_seed)?;
    let spec_one = agent_one.spec();
    let spec_two = agent_two.spec();

    let mut state = engine::new_game(setup_seed);
    let mut rng = StdRng::seed_from_u64(setup_seed ^ ENGINE_RNG_SALT);

    // The rules engine bans wall-clock reads (see `clippy.toml`), but the
    // arena is one of the crates explicitly carved out to time work for
    // reporting, not for any rules decision.
    #[allow(clippy::disallowed_methods)]
    let start = Instant::now();

    let mut moves = 0u32;
    loop {
        if state.is_over() {
            break;
        }
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let obs = state.observation();
        let action = match state.current_player() {
            Player::One => agent_one.choose(&obs, &legal, budget),
            Player::Two => agent_two.choose(&obs, &legal, budget),
        };
        if !legal.contains(&action) {
            return Err(format!(
                "agent returned an illegal action outside `legal`: {action:?}"
            ));
        }
        engine::apply(&mut state, action, &mut rng).map_err(|e| e.to_string())?;
        moves += 1;
    }

    #[allow(clippy::disallowed_methods)]
    let wall_time_ms = start.elapsed().as_millis() as u64;

    let result = state
        .result()
        .ok_or_else(|| "game loop ended without a legal action but no result".to_string())?;
    Ok((spec_one, spec_two, result, moves, wall_time_ms))
}

/// Play one paired seed: `agent_a` as [`Player::One`] / `agent_b` as
/// [`Player::Two`], then the same `seed`'s setup again with seats swapped.
/// Each agent keeps the same per-role RNG stream across both halves of the
/// pair (see `AGENT_A_SALT`/`AGENT_B_SALT`), independent of which seat it
/// sits in, so the pair isolates the effect of seat/setup asymmetry from the
/// agents' own random choices.
fn play_pair(
    agent_a: &str,
    agent_b: &str,
    seed: u64,
    budget: Budget,
) -> Result<[GameRecord; 2], String> {
    let a_seed = seed ^ AGENT_A_SALT;
    let b_seed = seed ^ AGENT_B_SALT;

    let (seat_one, seat_two, result, moves, wall_time_ms) =
        play_one_game(agent_a, agent_b, a_seed, b_seed, seed, budget)?;
    let first = GameRecord {
        seed,
        agent_a_seat: Player::One,
        seat_one,
        seat_two,
        result,
        moves,
        wall_time_ms,
    };

    let (seat_one, seat_two, result, moves, wall_time_ms) =
        play_one_game(agent_b, agent_a, b_seed, a_seed, seed, budget)?;
    let second = GameRecord {
        seed,
        agent_a_seat: Player::Two,
        seat_one,
        seat_two,
        result,
        moves,
        wall_time_ms,
    };

    Ok([first, second])
}

/// Play a full paired-seed match: for every seed in `seeds`, play it twice
/// (agent A as seat One / seat Two) via [`play_pair`], in parallel across
/// seeds with `rayon` — each game is single-threaded with its own seeded RNG
/// and no shared mutable state, so seeds are entirely independent work.
///
/// Returns `2 * seeds.len()` records, sorted by `(seed, agent_a_seat)` for a
/// deterministic, reviewable results file regardless of thread scheduling.
pub fn play_paired_match(
    agent_a: &str,
    agent_b: &str,
    seeds: &[u64],
    budget: Budget,
) -> Result<Vec<GameRecord>, String> {
    let mut records: Vec<GameRecord> = seeds
        .par_iter()
        .map(|&seed| play_pair(agent_a, agent_b, seed, budget))
        .collect::<Result<Vec<[GameRecord; 2]>, String>>()?
        .into_iter()
        .flatten()
        .collect();

    records.sort_by_key(|r| (r.seed, r.agent_a_seat));
    Ok(records)
}

/// Parse a `--budget` CLI value such as `"nodes:2000"` or `"time_ms:100"`.
pub fn parse_budget(s: &str) -> Result<Budget, String> {
    let (kind, value) = s.split_once(':').ok_or_else(|| {
        format!("invalid --budget \"{s}\": expected \"nodes:<n>\" or \"time_ms:<n>\"")
    })?;
    let value: u64 = value
        .parse()
        .map_err(|_| format!("invalid --budget value in \"{s}\": not a non-negative integer"))?;
    match kind {
        "nodes" => Ok(Budget::Nodes(value)),
        "time_ms" | "time" => Ok(Budget::TimeMs(value)),
        other => Err(format!(
            "invalid --budget kind \"{other}\": expected \"nodes\" or \"time_ms\""
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nodes_and_time_budgets() {
        assert_eq!(parse_budget("nodes:2000").unwrap(), Budget::Nodes(2000));
        assert_eq!(parse_budget("time_ms:100").unwrap(), Budget::TimeMs(100));
        assert_eq!(parse_budget("time:50").unwrap(), Budget::TimeMs(50));
    }

    #[test]
    fn rejects_malformed_budgets() {
        assert!(parse_budget("2000").is_err());
        assert!(parse_budget("nodes:abc").is_err());
        assert!(parse_budget("frobnicate:1").is_err());
    }

    #[test]
    fn random_vs_random_reaches_a_game_result() {
        let records = play_paired_match("random", "random", &[1], Budget::Nodes(1)).unwrap();
        assert_eq!(records.len(), 2);
        for r in &records {
            // Just proving a `GameResult` was reached is the point here;
            // `GameResult` has no "unset" variant, so this always holds — the
            // real assertion is that `play_paired_match` returned `Ok` at
            // all, i.e. the game ran to completion without an engine or
            // agent-contract error.
            let _ = r.result;
        }
    }

    #[test]
    fn seat_swap_pairing_uses_the_same_setup_with_seats_swapped() {
        let records = play_paired_match("random", "random", &[42], Budget::Nodes(1)).unwrap();
        assert_eq!(records.len(), 2);
        let (first, second) = (&records[0], &records[1]);
        assert_eq!(first.seed, second.seed);
        assert_eq!(first.agent_a_seat, Player::One);
        assert_eq!(second.agent_a_seat, Player::Two);

        // Both games share the deterministic setup seed, so `engine::new_game`
        // itself produced the identical initial state both times...
        let setup_one = engine::new_game(first.seed);
        let setup_two = engine::new_game(second.seed);
        assert_eq!(setup_one.current_player(), setup_two.current_player());
        assert_eq!(
            engine::legal_actions(&setup_one),
            engine::legal_actions(&setup_two)
        );

        // ...but the two full games are not required to (and with a
        // seed-dependent random agent, generally won't) end identically,
        // since each half of the pair draws its in-game decisions from an
        // independent RNG stream (`AGENT_A_SALT`/`AGENT_B_SALT`).
        // We don't assert inequality here (a random agent *could*
        // legitimately produce the same result by chance for some seeds);
        // the point is only that the pairing does not force the two games to
        // be trivial re-runs of one another, which the differing move counts
        // for most seeds already demonstrates in the larger end-to-end run.
        assert_eq!(first.seat_one.name, "random");
        assert_eq!(second.seat_one.name, "random");
    }

    #[test]
    fn multiple_seeds_run_in_parallel_and_all_complete() {
        let seeds: Vec<u64> = (0..20).collect();
        let records = play_paired_match("random", "random", &seeds, Budget::Nodes(1)).unwrap();
        assert_eq!(records.len(), 40);
        let t = tally(&records);
        assert_eq!(t.total(), 40);
    }

    #[test]
    fn unknown_agent_name_surfaces_as_an_error_not_a_panic() {
        let err = play_paired_match("nope", "random", &[1], Budget::Nodes(1)).unwrap_err();
        assert!(err.contains("nope"));
    }
}
