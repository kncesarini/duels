//! The core match runner: plays one complete game between two agents named
//! by a [`crate::agent_spec`] specification string (a bare registered name,
//! or `name:key=value,...`), and pairs up seat-swapped games per seed so
//! first-player advantage and setup randomness roughly cancel across the
//! pair.
//!
//! Mirrors the loop in `duels-server`'s `room::drive_agents`: agents are fed
//! only `Observation`s and `legal_actions`, never a `GameState`, exactly as
//! the `Agent` contract requires.
//!
//! Each game is also watched, move by move, for "race exposure" — whether
//! either player's conflict pawn or scientific-symbol count came within one
//! step of an instant win (see [`race_exposure_at`]) — without any extra
//! replay pass over the finished game: the flag is folded in as the game is
//! played, the same way `moves`/`wall_time_ms` already are.

use std::time::Instant;

use duels_agents_api::{AgentSpec, Budget};
use duels_core::scoring::VictoryKind;
use duels_core::{engine, GameResult, GameState, Player};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::agent_spec::make_agent_from_spec;

/// Conflict-pawn distance from centre (`0`) at which a player is one loot
/// token away from the capital (distance 9, an instant military win) — see
/// `duels_core::data::MilitaryTrack`. Reaching this distance, regardless of
/// who eventually wins the game or how, is what [`GameRecord::military_race_exposed`]
/// flags: it is the threshold at which a human reviewing a match would call
/// the military track "in play" for that game.
pub const MILITARY_RACE_DISTANCE: u8 = 6;

/// Distinct scientific symbols at which a player is one build away from
/// scientific supremacy (6 distinct symbols wins outright). Reaching this
/// count, regardless of who eventually wins, is what
/// [`GameRecord::science_race_exposed`] flags.
pub const SCIENCE_RACE_SYMBOLS: u8 = 5;

/// Whether `state` currently shows either side within the "race exposure"
/// zone of the military track / the scientific-symbol count, as `(military,
/// science)`. Called after every move (see `play_one_game`) to build up a
/// game's race-exposure flags without re-simulating anything.
fn race_exposure_at(state: &GameState) -> (bool, bool) {
    let military = state.conflict().unsigned_abs() >= MILITARY_RACE_DISTANCE;
    let science = state.player(Player::One).distinct_science() >= SCIENCE_RACE_SYMBOLS
        || state.player(Player::Two).distinct_science() >= SCIENCE_RACE_SYMBOLS;
    (military, science)
}

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
    /// Whether either player's conflict pawn reached [`MILITARY_RACE_DISTANCE`]
    /// or beyond at any point in this game, regardless of who eventually won
    /// or how. Lets a human sanity-check *why* one agent beats another (e.g.
    /// "it wins by consistently getting the military race into contention"),
    /// not just that it does.
    pub military_race_exposed: bool,
    /// Whether either player reached [`SCIENCE_RACE_SYMBOLS`] distinct
    /// scientific symbols at any point in this game, regardless of who
    /// eventually won or how.
    pub science_race_exposed: bool,
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

/// One side's wins, broken down by *how* they were won (see [`VictoryKind`]).
/// [`Self::total`] always equals that side's win count from [`MatchTally`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VictoryBreakdown {
    /// Wins where the conflict pawn reached the loser's capital.
    pub military_supremacy: u32,
    /// Wins on six distinct scientific symbols.
    pub scientific_supremacy: u32,
    /// Wins on victory points at the end of Age III.
    pub civilian_victory: u32,
    /// Wins on the civilian (blue) points tiebreak after equal totals.
    pub civilian_tiebreak: u32,
}

impl VictoryBreakdown {
    /// Every win this breakdown accounts for, regardless of kind.
    pub fn total(&self) -> u32 {
        self.military_supremacy
            + self.scientific_supremacy
            + self.civilian_victory
            + self.civilian_tiebreak
    }

    fn record(&mut self, kind: VictoryKind) {
        match kind {
            VictoryKind::MilitarySupremacy => self.military_supremacy += 1,
            VictoryKind::ScientificSupremacy => self.scientific_supremacy += 1,
            VictoryKind::CivilianVictory => self.civilian_victory += 1,
            VictoryKind::CivilianTiebreak => self.civilian_tiebreak += 1,
        }
    }
}

/// Both sides' [`VictoryBreakdown`]s over a match, from the "role A" / "role
/// B" perspective (see [`GameRecord::agent_a_seat`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchVictoryBreakdown {
    /// How role A's wins were achieved.
    pub a: VictoryBreakdown,
    /// How role B's wins were achieved.
    pub b: VictoryBreakdown,
}

/// Break down how each side's wins in `records` were achieved. Draws
/// contribute to neither side's breakdown; see [`tally`] for the draw count.
/// `victory_breakdown(records).a.total() == tally(records).a_wins` always
/// holds (and likewise for `b`) — see the round-trip test in this module.
pub fn victory_breakdown(records: &[GameRecord]) -> MatchVictoryBreakdown {
    let mut out = MatchVictoryBreakdown::default();
    for r in records {
        if let GameResult::Win { winner, kind } = r.result {
            if winner == r.agent_a_seat {
                out.a.record(kind);
            } else {
                out.b.record(kind);
            }
        }
    }
    out
}

/// How many games in a match came within reach of an instant win, regardless
/// of who eventually won or how — see [`GameRecord::military_race_exposed`] /
/// [`GameRecord::science_race_exposed`]. This is what lets a human
/// sanity-check *why* one agent beats another (e.g. "it wins mostly by
/// keeping the game out of either race, not by winning races itself") rather
/// than only observing that it does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RaceExposure {
    /// Games where either player's conflict pawn reached
    /// [`MILITARY_RACE_DISTANCE`] or beyond.
    pub military_games: u32,
    /// Games where either player reached [`SCIENCE_RACE_SYMBOLS`] distinct
    /// scientific symbols.
    pub science_games: u32,
    /// Total games the breakdown was computed over, for turning the above
    /// into a rate.
    pub total_games: u32,
}

/// Compute [`RaceExposure`] over `records`.
pub fn race_exposure(records: &[GameRecord]) -> RaceExposure {
    let mut out = RaceExposure {
        total_games: records.len() as u32,
        ..RaceExposure::default()
    };
    for r in records {
        if r.military_race_exposed {
            out.military_games += 1;
        }
        if r.science_race_exposed {
            out.science_games += 1;
        }
    }
    out
}

/// Everything [`play_one_game`] learns about one finished game, before it is
/// attached to a `seed`/`agent_a_seat` and turned into a [`GameRecord`].
struct OneGameOutcome {
    spec_one: AgentSpec,
    spec_two: AgentSpec,
    result: GameResult,
    moves: u32,
    wall_time_ms: u64,
    military_race_exposed: bool,
    science_race_exposed: bool,
}

/// Play one complete game: `seat_one_name`/`seat_two_name` occupy
/// [`Player::One`]/[`Player::Two`], each agent's own decisions seeded from
/// `seat_one_seed`/`seat_two_seed`, and the game setup (deal, shuffle, first
/// player) from `setup_seed`.
///
/// Returns the two agents' specs, the finished game's result, the move
/// count, wall-clock time, and the two race-exposure flags (see
/// [`race_exposure_at`]). Feeds each agent only `Observation`s and
/// `legal_actions`, never `GameState`, per the `Agent` contract.
#[allow(clippy::too_many_arguments)]
fn play_one_game(
    seat_one_name: &str,
    seat_two_name: &str,
    seat_one_seed: u64,
    seat_two_seed: u64,
    setup_seed: u64,
    budget: Budget,
) -> Result<OneGameOutcome, String> {
    let mut agent_one = make_agent_from_spec(seat_one_name, seat_one_seed)?;
    let mut agent_two = make_agent_from_spec(seat_two_name, seat_two_seed)?;
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
    let mut military_race_exposed = false;
    let mut science_race_exposed = false;
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
        let (military, science) = race_exposure_at(&state);
        military_race_exposed |= military;
        science_race_exposed |= science;
    }

    #[allow(clippy::disallowed_methods)]
    let wall_time_ms = start.elapsed().as_millis() as u64;

    let result = state
        .result()
        .ok_or_else(|| "game loop ended without a legal action but no result".to_string())?;
    Ok(OneGameOutcome {
        spec_one,
        spec_two,
        result,
        moves,
        wall_time_ms,
        military_race_exposed,
        science_race_exposed,
    })
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

    let outcome = play_one_game(agent_a, agent_b, a_seed, b_seed, seed, budget)?;
    let first = GameRecord {
        seed,
        agent_a_seat: Player::One,
        seat_one: outcome.spec_one,
        seat_two: outcome.spec_two,
        result: outcome.result,
        moves: outcome.moves,
        wall_time_ms: outcome.wall_time_ms,
        military_race_exposed: outcome.military_race_exposed,
        science_race_exposed: outcome.science_race_exposed,
    };

    let outcome = play_one_game(agent_b, agent_a, b_seed, a_seed, seed, budget)?;
    let second = GameRecord {
        seed,
        agent_a_seat: Player::Two,
        seat_one: outcome.spec_one,
        seat_two: outcome.spec_two,
        result: outcome.result,
        moves: outcome.moves,
        wall_time_ms: outcome.wall_time_ms,
        military_race_exposed: outcome.military_race_exposed,
        science_race_exposed: outcome.science_race_exposed,
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

    #[test]
    fn race_exposure_thresholds_match_the_documented_distances() {
        use duels_core::testing::StateBuilder;

        // Fresh game: neither race is in reach.
        let fresh = StateBuilder::new().build();
        assert_eq!(race_exposure_at(&fresh), (false, false));

        // Just below the military threshold: not yet exposed.
        let st = StateBuilder::new().conflict(5).build();
        assert_eq!(race_exposure_at(&st), (false, false));
        // At and beyond the threshold (either direction): exposed.
        let st = StateBuilder::new().conflict(6).build();
        assert_eq!(race_exposure_at(&st), (true, false));
        let st = StateBuilder::new().conflict(-8).build();
        assert_eq!(race_exposure_at(&st), (true, false));

        // Four distinct symbols: not yet exposed. Five (one build away from
        // the 6-symbol instant win): exposed. Matches
        // `duels_core::engine::tests::six_distinct_symbols_wins_instantly`,
        // which builds the same five cards to reach exactly 5 symbols.
        let four = StateBuilder::new()
            .built(
                Player::One,
                &["workshop", "apothecary", "scriptorium", "pharmacist"],
            )
            .build();
        assert_eq!(race_exposure_at(&four), (false, false));
        let five = StateBuilder::new()
            .built(
                Player::One,
                &[
                    "workshop",
                    "apothecary",
                    "scriptorium",
                    "pharmacist",
                    "academy",
                ],
            )
            .build();
        assert_eq!(race_exposure_at(&five), (false, true));
    }

    #[test]
    fn victory_breakdown_sums_to_the_win_count_for_each_side() {
        // A real, sizeable match end to end: random vs random over enough
        // seeds to see a mix of civilian/tiebreak/draw outcomes (military
        // and scientific supremacy are rare with random play but the sum
        // property must hold regardless of which kinds actually occur).
        let seeds: Vec<u64> = (0..60).collect();
        let records = play_paired_match("random", "random", &seeds, Budget::Nodes(1)).unwrap();
        let t = tally(&records);
        let vb = victory_breakdown(&records);
        assert_eq!(vb.a.total(), t.a_wins);
        assert_eq!(vb.b.total(), t.b_wins);
        // Every record is accounted for by exactly one of: A's breakdown, B's
        // breakdown, or a draw.
        assert_eq!(vb.a.total() + vb.b.total() + t.draws, t.total());
    }

    #[test]
    fn race_exposure_counts_never_exceed_the_game_count() {
        let seeds: Vec<u64> = (0..30).collect();
        let records = play_paired_match("random", "random", &seeds, Budget::Nodes(1)).unwrap();
        let re = race_exposure(&records);
        assert_eq!(re.total_games, records.len() as u32);
        assert!(re.military_games <= re.total_games);
        assert!(re.science_games <= re.total_games);
    }
}
