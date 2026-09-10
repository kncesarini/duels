//! Does it actually matter who chooses to go first at an age boundary?
//!
//! `phased` (`crates/agents/phased`), this project's strongest hand-crafted
//! 1-ply evaluator, is asked exactly once per age boundary where the pawn
//! sits off-centre: `Phase::ChooseFirstPlayer`, resolved by the militarily
//! weaker player choosing who opens the next age. Reading
//! `crates/agents/phased/src/terms.rs`'s `next_age_start` and
//! `crates/agents/phased/src/menu.rs` together shows that `phased` *always*
//! answers "myself" — not because anyone measured that this is correct, but
//! because the only term that differs between the two candidate actions
//! (`λ × (menu_me + menu_opp)`) is positive whenever both menus are, which
//! makes "choose first" win by construction rather than by evidence.
//!
//! This tool turns that observation into an ordinary `duels-arena` matchup
//! instead of a change to `phased`'s evaluation weights: `age_start_policy`'s
//! `AgeStartPolicyAgent` wraps *any* agent, forcing its
//! `Phase::ChooseFirstPlayer` decisions to a fixed policy while leaving every
//! other decision completely untouched. Comparing two wrapped copies of the
//! same underlying agent isolates the age-start question from everything
//! else the agent does.
//!
//! # Modes
//!
//! * `first-vs-second` — the headline comparison: the wrapped agent always
//!   chooses to start the new age itself, vs. always handing it to the
//!   opponent. If `phased`'s "always choose first" policy is actually
//!   correct, `first` should beat `second` here.
//! * `agent-vs-first` — a sanity check on the wrapper itself, *not* a
//!   question about the game: the agent's own unwrapped behaviour vs. the
//!   `AlwaysFirst` wrapper around the same agent. If the diagnosis that
//!   `phased` already always chooses first is correct, this should read
//!   ~0 Elo — any large gap here means the wrapper isn't a faithful
//!   pass-through (or the diagnosis was wrong), not that "going first" is
//!   good or bad.
//! * `split` — Age III opens with two immediately accessible cards, so
//!   whoever starts it reveals for the second draw right away; that could
//!   easily cut the other way from Age II. This forces "start Age II myself,
//!   hand Age III to the opponent" against the exact reverse, to check
//!   whether one age-boundary answer generalizes to the other.
//!
//! # Usage
//!
//! ```text
//! cargo run --release -p duels-arena --example age_start_lab -- \
//!     first-vs-second phased 800 --seed 1 --budget nodes:1
//! cargo run --release -p duels-arena --example age_start_lab -- \
//!     agent-vs-first phased 800 --seed 5001 --budget nodes:1
//! cargo run --release -p duels-arena --example age_start_lab -- \
//!     first-vs-second mcts-uct 800 --seed 1 --budget nodes:2000
//! cargo run --release -p duels-arena --example age_start_lab -- \
//!     split phased 800 --seed 1 --budget nodes:1
//! ```
//!
//! Arguments: the mode (`first-vs-second` / `agent-vs-first` / `split`), then
//! the base agent's spec string (anything [`duels_arena::agent_spec`] accepts,
//! so `phased:base=v1` works — default `phased`), then optionally the number
//! of games (default 800, rounded down to an even number of paired seeds),
//! `--seed <N>` (default 1) and `--budget <spec>` (default `nodes:2000`; use
//! `nodes:1` for `phased`, which is a 1-ply agent, per the production budget
//! `duels_arena::leaderboard::LADDER` records for it).
//!
//! `--seed` is not a convenience here: this project's standing requirement
//! is that no single-seed-range result is trusted, so every experiment this
//! tool runs is expected to be reproduced at a second, disjoint base seed
//! (e.g. `1` and `5001`) before it's believed.

use duels_agents_api::{Agent, Budget};
use duels_arena::age_start_policy::{AgeStartChoice, AgeStartPolicyAgent};
use duels_arena::agent_spec::make_agent_from_spec;
use duels_arena::elo::fit_elo;
use duels_arena::match_runner::parse_budget;
use duels_core::{engine, GameState, Player};
use rand::{rngs::StdRng, SeedableRng};

/// Salts matching `match_runner`'s (and `matchup_profile`'s), so a lab run
/// and a `duels-arena match` run of the same seeds see the same games.
const AGENT_A_SALT: u64 = 0xA011_7A9E_5B21_0001;
const AGENT_B_SALT: u64 = 0xB022_8C3F_6D42_0002;
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    FirstVsSecond,
    AgentVsFirst,
    Split,
}

impl Mode {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "first-vs-second" => Ok(Mode::FirstVsSecond),
            "agent-vs-first" => Ok(Mode::AgentVsFirst),
            "split" => Ok(Mode::Split),
            other => Err(format!(
                "unknown mode \"{other}\": expected \"first-vs-second\", \"agent-vs-first\" or \"split\""
            )),
        }
    }

    /// Human-readable name for each side, for the report header.
    fn side_labels(self) -> (&'static str, &'static str) {
        match self {
            Mode::FirstVsSecond => ("always-first", "always-second"),
            Mode::AgentVsFirst => ("unwrapped", "always-first"),
            Mode::Split => ("split(II=self,III=opp)", "split(II=opp,III=self)"),
        }
    }

    /// The policy override for "side A"; `None` means the agent's own
    /// unwrapped behaviour.
    fn choice_a(self) -> Option<AgeStartChoice> {
        match self {
            Mode::FirstVsSecond => Some(AgeStartChoice::AlwaysFirst),
            Mode::AgentVsFirst => None,
            Mode::Split => Some(AgeStartChoice::Split {
                age2_self_first: true,
                age3_self_first: false,
            }),
        }
    }

    /// The policy override for "side B".
    fn choice_b(self) -> AgeStartChoice {
        match self {
            Mode::FirstVsSecond => AgeStartChoice::AlwaysSecond,
            Mode::AgentVsFirst => AgeStartChoice::AlwaysFirst,
            Mode::Split => AgeStartChoice::Split {
                age2_self_first: false,
                age3_self_first: true,
            },
        }
    }
}

/// Build "side A" of the comparison: `base` (freshly constructed, seeded
/// from `seed`), optionally wrapped in [`AgeStartPolicyAgent`] per `mode`.
fn make_side_a(mode: Mode, base: &str, seed: u64) -> Result<Box<dyn Agent + Send>, String> {
    let inner = make_agent_from_spec(base, seed)?;
    Ok(match mode.choice_a() {
        Some(choice) => Box::new(AgeStartPolicyAgent::new(inner, choice)),
        None => inner,
    })
}

/// Build "side B" of the comparison.
fn make_side_b(mode: Mode, base: &str, seed: u64) -> Result<Box<dyn Agent + Send>, String> {
    let inner = make_agent_from_spec(base, seed)?;
    Ok(Box::new(AgeStartPolicyAgent::new(inner, mode.choice_b())))
}

/// Play one complete game, `seat_one`/`seat_two` occupying [`Player::One`]/
/// [`Player::Two`], mirroring `match_runner::play_one_game` and
/// `matchup_profile`'s `play` — agents are fed only `Observation`s and
/// `legal_actions`, never `GameState`.
fn play(
    mut seat_one: Box<dyn Agent + Send>,
    mut seat_two: Box<dyn Agent + Send>,
    setup_seed: u64,
    budget: Budget,
) -> Result<GameState, String> {
    let mut state = engine::new_game(setup_seed);
    let mut rng = StdRng::seed_from_u64(setup_seed ^ ENGINE_RNG_SALT);
    let mut moves = 0u32;
    while !state.is_over() {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let obs = state.observation();
        let action = match state.current_player() {
            Player::One => seat_one.choose(&obs, &legal, budget),
            Player::Two => seat_two.choose(&obs, &legal, budget),
        };
        engine::apply(&mut state, action, &mut rng).map_err(|e| e.to_string())?;
        moves += 1;
        if moves > 400 {
            return Err("game did not terminate".to_string());
        }
    }
    Ok(state)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = args
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .collect::<Vec<_>>();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let mode = match positional
        .first()
        .map(|s| Mode::parse(s))
        .unwrap_or_else(|| Ok(Mode::FirstVsSecond))
    {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let base_agent = positional
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("phased")
        .to_string();
    let games: u64 = positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(800);
    let seed: u64 = flag("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let budget = match parse_budget(&flag("--budget").unwrap_or_else(|| "nodes:2000".to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let (label_a, label_b) = mode.side_labels();
    let pairs = (games / 2).max(1);
    println!(
        "age_start_lab: {base_agent}[{label_a}] vs {base_agent}[{label_b}]  (mode {mode:?})\n  \
         {} games = {pairs} paired seat-swapped seeds from base seed {seed}, budget {budget:?}",
        pairs * 2
    );

    let mut a_wins = 0u32;
    let mut b_wins = 0u32;
    let mut draws = 0u32;
    let mut played = 0u32;

    for i in 0..pairs {
        let setup = seed.wrapping_add(i);
        let seed_a = setup ^ AGENT_A_SALT;
        let seed_b = setup ^ AGENT_B_SALT;

        for a_is_one in [true, false] {
            let (one, two): (Box<dyn Agent + Send>, Box<dyn Agent + Send>) = if a_is_one {
                let one = make_side_a(mode, &base_agent, seed_a).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                });
                let two = make_side_b(mode, &base_agent, seed_b).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                });
                (one, two)
            } else {
                let one = make_side_b(mode, &base_agent, seed_b).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                });
                let two = make_side_a(mode, &base_agent, seed_a).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                });
                (one, two)
            };

            let state = match play(one, two, setup, budget) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("seed {setup}: {e}");
                    std::process::exit(1);
                }
            };
            let Some(result) = state.result() else {
                continue;
            };
            let a_seat = if a_is_one { Player::One } else { Player::Two };
            match result.winner() {
                None => draws += 1,
                Some(winner) if winner == a_seat => a_wins += 1,
                Some(_) => b_wins += 1,
            }
            played += 1;
        }
        if (i + 1) % 25 == 0 {
            eprintln!("  ... {} games", (i + 1) * 2);
        }
    }

    println!(
        "results: {label_a} {a_wins} wins, {label_b} {b_wins} wins, {draws} draws  (out of {played})"
    );

    let est = fit_elo(a_wins, b_wins, draws);
    println!(
        "elo: {label_a} = {:+.1} (anchor: {label_b} = {:.1}), 95% CI [{:+.1}, {:+.1}]",
        est.rating_diff, est.anchor_elo, est.diff_ci_low, est.diff_ci_high
    );
}
