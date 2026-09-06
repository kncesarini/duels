//! *How* does one agent beat another, and how does it lose?
//!
//! `duels-arena match` answers "how often", which is the number that decides
//! whether a change ships. It is not the number that tells you what to build
//! next. An agent that wins 13% of its games and wins **every one of them by
//! scientific supremacy** is not 13% of the way to being good — it has one
//! working plan and two broken ones, and the aggregate rate hides that
//! completely.
//!
//! So this example plays the same paired, seat-swapped match the real runner
//! does and reports, per side:
//!
//! * the win-condition breakdown (military supremacy / scientific supremacy /
//!   civilian victory / civilian tiebreak);
//! * the composition of the cities it builds, by card colour;
//! * **in its losses specifically**: the final pawn distance, the military
//!   victory points it conceded, the final victory-point margin, and how many
//!   distinct scientific symbols it reached. Losses are where the diagnosis
//!   is: a loss by 30 points with the pawn at 7 against you and two symbols on
//!   the table is a different disease from a loss by 2 points on the tiebreak.
//! * the mean game length in decisions, which is how you see an agent that is
//!   winning by ending the game early.
//!
//! ```text
//! cargo run --release -p duels-arena --example matchup_profile -- phased mcts-uct
//! cargo run --release -p duels-arena --example matchup_profile -- \
//!     phased:base=v1 phased 200 --seed 5001 --budget nodes:1
//! ```
//!
//! Arguments: the two agent specification strings (anything
//! [`duels_arena::agent_spec`] accepts, so `phased:base=v1` works), then
//! optionally the number of games (default 200, rounded down to an even number
//! of paired seeds), `--seed <N>` (default 1) and `--budget <spec>` (default
//! `nodes:2000`, which is what `mcts-uct` needs to be itself; use `nodes:1`
//! for a match between two 1-ply agents).
//!
//! `--seed` is not a convenience: reproducing an accept on a second, disjoint
//! seed range is this project's standing requirement, and without it this
//! diagnostic could only ever look at one.

use duels_agents_api::Budget;
use duels_arena::agent_spec::make_agent_from_spec;
use duels_arena::match_runner::parse_budget;
use duels_core::data::CardType;
use duels_core::scoring::{self, VictoryKind};
use duels_core::{engine, GameResult, GameState, Player};
use rand::{rngs::StdRng, SeedableRng};

/// Salts matching `match_runner`'s, so a profile run and a `duels-arena match`
/// run of the same seeds see the same games.
const AGENT_A_SALT: u64 = 0xA011_7A9E_5B21_0001;
const AGENT_B_SALT: u64 = 0xB022_8C3F_6D42_0002;
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

const COLOURS: [(CardType, &str); 7] = [
    (CardType::RawMaterial, "brown"),
    (CardType::ManufacturedGood, "grey"),
    (CardType::Civilian, "blue"),
    (CardType::Scientific, "green"),
    (CardType::Commercial, "yellow"),
    (CardType::Military, "red"),
    (CardType::Guild, "purple"),
];

/// What one side did across a match.
#[derive(Default, Clone)]
struct SideProfile {
    games: u32,
    wins: u32,
    /// Wins by kind, indexed as in [`kind_index`].
    by_kind: [u32; 4],
    /// Cards built, by colour.
    cards: [u32; 7],
    /// Losses only.
    losses: u32,
    loss_pawn_distance: f64,
    loss_military_vp_conceded: f64,
    loss_vp_margin: f64,
    loss_science_symbols: f64,
    /// Every game, not just losses.
    science_symbols: f64,
    pawn_distance: f64,
}

fn kind_index(kind: VictoryKind) -> usize {
    match kind {
        VictoryKind::MilitarySupremacy => 0,
        VictoryKind::ScientificSupremacy => 1,
        VictoryKind::CivilianVictory => 2,
        VictoryKind::CivilianTiebreak => 3,
    }
}

const KIND_LABELS: [&str; 4] = ["military", "science", "civilian", "tiebreak"];

impl SideProfile {
    fn record(&mut self, state: &GameState, seat: Player, result: GameResult) {
        self.games += 1;
        let me = state.player(seat);
        let mine = scoring::breakdown(state, seat);
        let theirs = scoring::breakdown(state, seat.other());

        for (i, (kind, _)) in COLOURS.iter().enumerate() {
            self.cards[i] += count_colour(state, seat, *kind);
        }
        let symbols = f64::from(me.distinct_science());
        self.science_symbols += symbols;
        // Signed so positive means the pawn favours this side.
        let signed = match seat {
            Player::One => f64::from(state.conflict()),
            Player::Two => -f64::from(state.conflict()),
        };
        self.pawn_distance += signed;

        match result {
            GameResult::Win { winner, kind } if winner == seat => {
                self.wins += 1;
                self.by_kind[kind_index(kind)] += 1;
            }
            _ => {
                self.losses += 1;
                self.loss_pawn_distance += signed;
                self.loss_military_vp_conceded += f64::from(theirs.military);
                self.loss_vp_margin += f64::from(mine.total) - f64::from(theirs.total);
                self.loss_science_symbols += symbols;
            }
        }
    }
}

fn count_colour(state: &GameState, p: Player, kind: CardType) -> u32 {
    let s = duels_core::data::statics();
    (state.player(p).built_mask() & s.card_masks[kind.index()]).count_ones()
}

fn mean(total: f64, n: u32) -> f64 {
    if n == 0 {
        0.0
    } else {
        total / f64::from(n)
    }
}

fn pct(a: u32, b: u32) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * f64::from(a) / f64::from(b)
    }
}

/// Play one game and hand back the finished state plus which seat "role A"
/// occupied.
fn play(
    seat_one: &str,
    seat_two: &str,
    seat_one_seed: u64,
    seat_two_seed: u64,
    setup_seed: u64,
    budget: Budget,
) -> Result<(GameState, u32), String> {
    let mut one = make_agent_from_spec(seat_one, seat_one_seed)?;
    let mut two = make_agent_from_spec(seat_two, seat_two_seed)?;
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
            Player::One => one.choose(&obs, &legal, budget),
            Player::Two => two.choose(&obs, &legal, budget),
        };
        engine::apply(&mut state, action, &mut rng).map_err(|e| e.to_string())?;
        moves += 1;
        if moves > 400 {
            return Err("game did not terminate".to_string());
        }
    }
    Ok((state, moves))
}

fn report(name: &str, p: &SideProfile) {
    println!("=== {name} ===");
    println!(
        "  {} / {} wins ({:.1}%)",
        p.wins,
        p.games,
        pct(p.wins, p.games)
    );
    print!("  by win condition   ");
    for (i, label) in KIND_LABELS.iter().enumerate() {
        print!("{label} {:>4}  ", p.by_kind[i]);
    }
    println!();
    print!("  city composition   ");
    let total_cards: u32 = p.cards.iter().sum();
    for (i, (_, label)) in COLOURS.iter().enumerate() {
        print!("{label} {:>4.1}  ", mean(f64::from(p.cards[i]), p.games));
    }
    println!(
        "(mean cards/game, {:.1} total)",
        mean(f64::from(total_cards), p.games)
    );
    println!(
        "  overall            {:.2} distinct symbols, pawn {:+.2} from centre",
        mean(p.science_symbols, p.games),
        mean(p.pawn_distance, p.games)
    );
    println!(
        "  in its {} losses  pawn {:+.2}, military VP conceded {:.2}, VP margin {:+.2}, symbols {:.2}",
        p.losses,
        mean(p.loss_pawn_distance, p.losses),
        mean(p.loss_military_vp_conceded, p.losses),
        mean(p.loss_vp_margin, p.losses),
        mean(p.loss_science_symbols, p.losses),
    );
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

    let agent_a = positional
        .first()
        .map(|s| s.as_str())
        .unwrap_or("phased")
        .to_string();
    let agent_b = positional
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("mcts-uct")
        .to_string();
    let games: u64 = positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let seed: u64 = flag("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let budget = match parse_budget(&flag("--budget").unwrap_or_else(|| "nodes:2000".to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let pairs = (games / 2).max(1);
    println!(
        "matchup profile: {agent_a} vs {agent_b}\n  \
         {} games = {pairs} paired seat-swapped seeds from base seed {seed}, budget {budget:?}\n",
        pairs * 2
    );

    let mut a = SideProfile::default();
    let mut b = SideProfile::default();
    let mut total_moves = 0u64;
    let mut played = 0u32;

    for i in 0..pairs {
        let setup = seed.wrapping_add(i);
        let seed_a = setup ^ AGENT_A_SALT;
        let seed_b = setup ^ AGENT_B_SALT;
        for a_is_one in [true, false] {
            let (one, two, seed_one, seed_two) = if a_is_one {
                (&agent_a, &agent_b, seed_a, seed_b)
            } else {
                (&agent_b, &agent_a, seed_b, seed_a)
            };
            let (state, moves) = match play(one, two, seed_one, seed_two, setup, budget) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("seed {setup}: {e}");
                    std::process::exit(1);
                }
            };
            let Some(result) = state.result() else {
                continue;
            };
            let a_seat = if a_is_one { Player::One } else { Player::Two };
            a.record(&state, a_seat, result);
            b.record(&state, a_seat.other(), result);
            total_moves += u64::from(moves);
            played += 1;
        }
        if (i + 1) % 25 == 0 {
            eprintln!("  ... {} games", (i + 1) * 2);
        }
    }

    let draws = played - a.wins - b.wins;
    println!(
        "{}-{}-{} (A-B-draws) over {played} games, mean length {:.1} decisions\n",
        a.wins,
        b.wins,
        draws,
        total_moves as f64 / f64::from(played.max(1))
    );
    report(&agent_a, &a);
    println!();
    report(&agent_b, &b);
}
