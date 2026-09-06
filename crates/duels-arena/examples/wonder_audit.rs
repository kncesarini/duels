//! **Does the agent actually build the wonders it was under-valuing?**
//!
//! `duels-agent-phased`'s round four fixes two things about wonders, and
//! neither of them is a thing an Elo number can see clearly:
//!
//! 1. Four wonders — Circus Maximus, the Statue of Zeus, the Mausoleum and the
//!    Great Library — leave `engine::apply` with their effect **not yet
//!    applied** (`finish_turn` returns early on a pending effect). Scoring that
//!    state directly credits none of the destroy, none of the retrieval and
//!    none of the token: only the flat "this wonder has an effect" bonus. So
//!    they were systematically under-built.
//! 2. `terms::wonder_potential` never checked the seven-wonder cap, so an
//!    unbuildable wonder kept scoring for the rest of the game.
//!
//! A win rate is a weak instrument for either: both are worth a couple of
//! victory points in a game whose scores routinely span thirty, and both are
//! about *which move gets played* rather than about how often the agent wins.
//! So this counts the behaviour directly, exactly as `rail_audit.rs` counts
//! rails rather than sampling them:
//!
//! ```text
//! per side, over the whole match:
//!   how many drafted wonders were never built
//!   the mean turn at which each wonder was built, and how often
//!   the four effect wonders, called out on their own line
//!   the five play-again wonders, called out on their own line
//!   the decisions each side actually took, and how many were extra turns
//!   how many games ended with a wonder that could never have been built
//! ```
//!
//! Run the same match twice — once with the round-four options off, once with
//! them on — and the difference is the behaviour change, independent of Elo:
//!
//! ```text
//! cargo run --release -p duels-arena --example wonder_audit -- \
//!     phased:base=v3 phased:base=v3 200 --budget nodes:1
//! cargo run --release -p duels-arena --example wonder_audit -- \
//!     phased:pending=completed phased:pending=completed 200 --budget nodes:1
//! ```
//!
//! Round six asks the same question of the extra-turn premium, and the
//! decisions/extra-turns lines are what answer it: a premium that makes the
//! agent build play-again wonders more often and earlier should also show up
//! as it *taking more turns*, which is the only reason to want them.
//!
//! ```text
//! cargo run --release -p duels-arena --example wonder_audit -- \
//!     phased:base=v5 phased:base=v5 400 --budget nodes:1
//! cargo run --release -p duels-arena --example wonder_audit -- \
//!     phased:wprem=9 phased:wprem=9 400 --budget nodes:1
//! ```
//!
//! Arguments: the two agent specification strings, then optionally the number
//! of games (default 200, rounded down to an even number of paired seeds),
//! `--seed <N>` (default 1) and `--budget <spec>` (default `nodes:1`, which is
//! what a 1-ply agent needs; use `nodes:2000` against a searching opponent).

use duels_agents_api::Budget;
use duels_arena::agent_spec::make_agent_from_spec;
use duels_arena::match_runner::parse_budget;
use duels_core::data::{WonderId, NUM_WONDERS};
use duels_core::state::MAX_WONDERS_BUILT;
use duels_core::{engine, Action, GameState, Player};
use rand::{rngs::StdRng, SeedableRng};

/// Salts matching `match_runner`'s, so an audit run and a `duels-arena match`
/// run of the same seeds see the same games.
const AGENT_A_SALT: u64 = 0xA011_7A9E_5B21_0001;
const AGENT_B_SALT: u64 = 0xB022_8C3F_6D42_0002;
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// The four wonders whose construction the engine leaves half-finished: they
/// set a [`duels_core::state::Pending`] and `finish_turn` returns early. Read
/// off the wonder data rather than written down by slug, so a change to
/// `data/wonders.json` cannot silently desynchronise this list.
fn leaves_a_pending_effect(w: WonderId) -> bool {
    let def = w.def();
    def.destroy.is_some() || def.build_discarded_free || def.choose_progress_token
}

/// The five wonders that print **play again** -- Piraeus, The Appian Way, The
/// Hanging Gardens, The Sphinx and The Temple of Artemis. Read off the wonder
/// data for the same reason as above, and because the whole point of the
/// round-six premium is that this set is priced differently from the rest.
fn grants_an_extra_turn(w: WonderId) -> bool {
    w.def().play_again
}

#[derive(Clone)]
struct Side {
    games: u32,
    /// Times each wonder was drafted, built, and the sum of the turn numbers
    /// it was built on, indexed by [`WonderId::index`].
    drafted: [u32; NUM_WONDERS],
    built: [u32; NUM_WONDERS],
    build_turn: [f64; NUM_WONDERS],
    /// Drafted-but-never-built wonders at game end, summed over games.
    unbuilt_at_end: u32,
    /// ...of those, the ones that were *unbuildable* — the seven shared slots
    /// were already gone. This is what bug two was paying for.
    dead_at_end: u32,
    /// Decisions this side actually took, summed over games.
    decisions: u32,
    /// ...of those, the ones taken immediately after its own previous
    /// decision: an extra turn, whether granted by a play-again wonder or by
    /// Theology. This is the behaviour the round-six premium is *for*, as
    /// opposed to the build counts, which are only the means.
    extra_turns: u32,
}

impl Default for Side {
    fn default() -> Self {
        Side {
            games: 0,
            drafted: [0; NUM_WONDERS],
            built: [0; NUM_WONDERS],
            build_turn: [0.0; NUM_WONDERS],
            unbuilt_at_end: 0,
            dead_at_end: 0,
            decisions: 0,
            extra_turns: 0,
        }
    }
}

impl Side {
    fn merge(&mut self, other: &Side) {
        self.games += other.games;
        for i in 0..NUM_WONDERS {
            self.drafted[i] += other.drafted[i];
            self.built[i] += other.built[i];
            self.build_turn[i] += other.build_turn[i];
        }
        self.unbuilt_at_end += other.unbuilt_at_end;
        self.dead_at_end += other.dead_at_end;
        self.decisions += other.decisions;
        self.extra_turns += other.extra_turns;
    }

    fn record_end(&mut self, state: &GameState, seat: Player) {
        self.games += 1;
        let me = state.player(seat);
        let no_slots = state.wonders_built_total() >= MAX_WONDERS_BUILT;
        for w in me.wonders() {
            self.drafted[w.index()] += 1;
            if !me.has_built_wonder(w) {
                self.unbuilt_at_end += 1;
                if no_slots {
                    self.dead_at_end += 1;
                }
            }
        }
    }

    /// How often a wonder of the given class was built, per game, and the mean
    /// turn it went up on.
    fn class(&self, pick: impl Fn(WonderId) -> bool) -> (u32, u32, f64) {
        let mut drafted = 0;
        let mut built = 0;
        let mut turns = 0.0;
        for i in 0..NUM_WONDERS {
            let w = WonderId::from_index(i);
            if !pick(w) {
                continue;
            }
            drafted += self.drafted[i];
            built += self.built[i];
            turns += self.build_turn[i];
        }
        let mean = if built == 0 {
            f64::NAN
        } else {
            turns / f64::from(built)
        };
        (built, drafted, mean)
    }
}

/// Play one game, recording every wonder each seat builds and when.
fn play(
    one: &str,
    two: &str,
    seed_one: u64,
    seed_two: u64,
    setup_seed: u64,
    budget: Budget,
    sides: &mut [Side; 2],
) -> Result<GameState, String> {
    let mut agents = [
        make_agent_from_spec(one, seed_one)?,
        make_agent_from_spec(two, seed_two)?,
    ];
    let mut state = engine::new_game(setup_seed);
    let mut rng = StdRng::seed_from_u64(setup_seed ^ ENGINE_RNG_SALT);
    let mut moves = 0u32;
    // Who moved last. A decision taken by the same player twice running is an
    // extra turn -- read off the sequence of movers rather than from a flag, so
    // it counts what actually happened at the table.
    let mut previous: Option<Player> = None;

    while !state.is_over() {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let obs = state.observation();
        let me = state.current_player();
        let action = agents[me.index()].choose(&obs, &legal, budget);
        // A pending resolution (a destroy target, a progress token) is part of
        // the same turn, not a new one, so it is not a decision for this count.
        if state.pending().is_none() {
            sides[me.index()].decisions += 1;
            if previous == Some(me) {
                sides[me.index()].extra_turns += 1;
            }
            previous = Some(me);
        }
        if let Action::BuildWonder { wonder, .. } = action {
            let side = &mut sides[me.index()];
            side.built[wonder.index()] += 1;
            side.build_turn[wonder.index()] += f64::from(state.turn());
        }
        engine::apply(&mut state, action, &mut rng).map_err(|e| e.to_string())?;
        moves += 1;
        if moves > 400 {
            return Err("game did not terminate".to_string());
        }
    }
    Ok(state)
}

fn report(name: &str, s: &Side) {
    println!("{name}  ({} games)", s.games);
    let per = |n: u32| f64::from(n) / f64::from(s.games.max(1));
    let (built, drafted, turn) = s.class(|_| true);
    println!(
        "  wonders built                {built:>5} / {drafted:<5} drafted  \
         ({:.2} per game, mean turn {turn:.1})",
        per(built)
    );
    println!(
        "  drafted and never built      {:>5}          ({:.2} per game)",
        s.unbuilt_at_end,
        per(s.unbuilt_at_end)
    );
    println!(
        "  ...of those, unbuildable     {:>5}          ({:.2} per game) -- what the \
         uncapped wonder term used to keep paying for",
        s.dead_at_end,
        per(s.dead_at_end)
    );

    let (built, drafted, turn) = s.class(leaves_a_pending_effect);
    println!(
        "  the four pending-effect wonders (Circus Maximus / Statue of Zeus / \
         Mausoleum / Great Library):"
    );
    println!(
        "      built                    {built:>5} / {drafted:<5} drafted  \
         ({:.0}% of the ones drafted, mean turn {turn:.1})",
        100.0 * f64::from(built) / f64::from(drafted.max(1))
    );
    let (built, drafted, turn) = s.class(|w| !leaves_a_pending_effect(w));
    println!(
        "      every other wonder       {built:>5} / {drafted:<5} drafted  \
         ({:.0}%, mean turn {turn:.1})",
        100.0 * f64::from(built) / f64::from(drafted.max(1))
    );

    println!(
        "  the five play-again wonders (Piraeus / Appian Way / Hanging Gardens / \
         Sphinx / Temple of Artemis):"
    );
    let (built, drafted, turn) = s.class(grants_an_extra_turn);
    println!(
        "      built                    {built:>5} / {drafted:<5} drafted  \
         ({:.0}% of the ones drafted, mean turn {turn:.1})",
        100.0 * f64::from(built) / f64::from(drafted.max(1))
    );
    let (built, drafted, turn) = s.class(|w| !grants_an_extra_turn(w));
    println!(
        "      every other wonder       {built:>5} / {drafted:<5} drafted  \
         ({:.0}%, mean turn {turn:.1})",
        100.0 * f64::from(built) / f64::from(drafted.max(1))
    );
    println!(
        "      decisions taken          {:>5}          ({:.2} per game), of which \
         {} were extra turns ({:.2} per game)",
        s.decisions,
        per(s.decisions),
        s.extra_turns,
        per(s.extra_turns)
    );

    println!("  per wonder:");
    let mut rows: Vec<usize> = (0..NUM_WONDERS).collect();
    rows.sort_by_key(|&i| std::cmp::Reverse(s.built[i]));
    for i in rows {
        let w = WonderId::from_index(i);
        if s.drafted[i] == 0 {
            continue;
        }
        let mean = if s.built[i] == 0 {
            f64::NAN
        } else {
            s.build_turn[i] / f64::from(s.built[i])
        };
        println!(
            "      {:<22} {:>4} / {:<4}  ({:>3.0}%)  mean turn {mean:>5.1}{}",
            w.def().name,
            s.built[i],
            s.drafted[i],
            100.0 * f64::from(s.built[i]) / f64::from(s.drafted[i]),
            match (leaves_a_pending_effect(w), grants_an_extra_turn(w)) {
                (true, _) => "   <- pending effect",
                (_, true) => "   <- play again",
                _ => "",
            }
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = args.iter().take_while(|a| !a.starts_with("--")).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let agent_a = positional
        .first()
        .map(|s| s.as_str())
        .unwrap_or("phased")
        .to_string();
    let agent_b = positional
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("phased")
        .to_string();
    let games: u64 = positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let seed: u64 = flag("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let budget = match parse_budget(&flag("--budget").unwrap_or_else(|| "nodes:1".to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let pairs = (games / 2).max(1);
    println!(
        "wonder audit: {agent_a} vs {agent_b}\n  \
         {} games = {pairs} paired seat-swapped seeds from base seed {seed}, budget {budget:?}\n",
        pairs * 2
    );

    let mut a = Side::default();
    let mut b = Side::default();
    for i in 0..pairs {
        let setup = seed.wrapping_add(i);
        let (seed_a, seed_b) = (setup ^ AGENT_A_SALT, setup ^ AGENT_B_SALT);
        for a_is_one in [true, false] {
            let (one, two, s_one, s_two) = if a_is_one {
                (&agent_a, &agent_b, seed_a, seed_b)
            } else {
                (&agent_b, &agent_a, seed_b, seed_a)
            };
            // Indexed by seat; folded into the per-agent totals below.
            let mut seats = [Side::default(), Side::default()];
            let state = match play(one, two, s_one, s_two, setup, budget, &mut seats) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("seed {setup}: {e}");
                    std::process::exit(1);
                }
            };
            let a_seat = if a_is_one { Player::One } else { Player::Two };
            seats[a_seat.index()].record_end(&state, a_seat);
            seats[a_seat.other().index()].record_end(&state, a_seat.other());
            a.merge(&seats[a_seat.index()]);
            b.merge(&seats[a_seat.other().index()]);
        }
        if (i + 1) % 25 == 0 {
            eprintln!("  ... {} games", (i + 1) * 2);
        }
    }

    report(&agent_a, &a);
    println!();
    report(&agent_b, &b);
}
