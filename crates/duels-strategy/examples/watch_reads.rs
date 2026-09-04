//! Play a self-play game and narrate what the race reads say after every move.
//!
//! This is the human checkpoint for this crate: the point is to read it and
//! judge whether "Live" / "Imminent" / "Closed" match how *you* read those
//! positions, and whether the stance the mover ends up with is the one you
//! would want a search to spend its nodes on.
//!
//! ```text
//! cargo run --release -p duels-strategy --example watch_reads
//! cargo run --release -p duels-strategy --example watch_reads -- 7 greedy random
//! cargo run --release -p duels-strategy --example watch_reads -- 7 greedy random --quiet
//! ```
//!
//! Arguments, all optional: `seed`, `player-one agent`, `player-two agent`
//! (`greedy` or `random`), and `--quiet` to print only the turns where a
//! classification changed.
//!
//! The default pairing is `greedy` vs `random`, which is the matchup that
//! motivated this crate: `greedy` carries explicit military-race terms in its
//! evaluation and still loses to `random` by military supremacy in roughly one
//! game in ten, because a one-ply evaluation cannot see a race two moves out.

use duels_agent_greedy::GreedyAgent;
use duels_agent_random::RandomAgent;
use duels_agents_api::{Agent, Budget};
use duels_core::data::Science;
use duels_core::state::Phase;
use duels_core::{engine, Action, GameState, Player};
use duels_strategy::masks::ALL_SCIENCE;
use duels_strategy::{
    action_prior, military_read, science_read, stance, vp_read, Board, MilitaryRead,
    MilitaryStatus, ScienceRead, ScienceStatus, VpRead,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_agent(name: &str, seed: u64) -> Box<dyn Agent> {
    match name {
        "random" => Box::new(RandomAgent::new(seed)),
        "greedy" => Box::new(GreedyAgent::new(seed)),
        other => {
            eprintln!("unknown agent {other:?}; using greedy");
            Box::new(GreedyAgent::new(seed))
        }
    }
}

fn military_label(s: MilitaryStatus) -> &'static str {
    match s {
        MilitaryStatus::Imminent => "IMMINENT",
        MilitaryStatus::Live => "live",
        MilitaryStatus::Closed => "closed",
    }
}

fn science_label(s: ScienceStatus) -> &'static str {
    match s {
        ScienceStatus::Imminent => "IMMINENT",
        ScienceStatus::Live => "live",
        ScienceStatus::Pressure => "pressure",
        ScienceStatus::Closed => "closed",
    }
}

fn symbol_char(s: Science) -> char {
    match s {
        Science::Mortar => 'M',
        Science::Pendulum => 'P',
        Science::Inkwell => 'I',
        Science::Wheel => 'W',
        Science::Sundial => 'S',
        Science::Gyroscope => 'G',
        Science::Balance => 'L',
    }
}

/// The pawn track as a picture, with the capitals at either end.
fn pawn_bar(conflict: i8) -> String {
    let mut bar = String::from("P1 |");
    for d in -9i8..=9 {
        bar.push(if d == conflict {
            '#'
        } else if d == 0 {
            '+'
        } else {
            '-'
        });
    }
    bar.push_str("| P2");
    bar
}

fn describe_military(r: &MilitaryRead) -> String {
    let mut s = format!(
        "military {:<8} need {:>2}   now {:>2} (best {:>1}, fork {})   table {:>2} + {:.1} hidden + {:.1} to come",
        military_label(r.status),
        r.need,
        r.now,
        r.best_single,
        r.fork,
        r.visible,
        r.expected_hidden,
        r.expected_future_ages,
    );
    match r.turns_to_close {
        Some(t) => s.push_str(&format!(
            "\n      closes in ~{t} of its own turns (of {} left)",
            r.decisions_left
        )),
        None => s.push_str("\n      no route to the capital left"),
    }
    if r.undeniable {
        s.push_str("   [UNDENIABLE]");
    }
    if r.loot_damage > 0 {
        if let Some(n) = r.loot_shields_needed {
            s.push_str(&format!(
                "\n      next loot token: {n} more shields, costs the opponent {} coins",
                r.loot_damage
            ));
        }
    }
    let next_band = r
        .bands
        .iter()
        .find(|b| b.shields_needed > 0 && b.vp_gain > 0);
    if let Some(b) = next_band {
        s.push_str(&format!(
            "\n      next scoring band: +{} VP for {} more shields",
            b.vp_gain, b.shields_needed
        ));
    }
    s
}

fn describe_science(r: &ScienceRead) -> String {
    let held: String = ALL_SCIENCE
        .iter()
        .map(|&sym| {
            let a = &r.availability[sym.index()];
            if a.held > 0 {
                symbol_char(sym)
            } else {
                '.'
            }
        })
        .collect();
    let mut s = format!(
        "science  {:<8} {} of 6 held [{held}]   missing {}   reachable {}   fragility {}",
        science_label(r.status),
        r.distinct,
        r.missing,
        r.obtainable_missing,
        r.fragility,
    );
    if r.dead {
        s.push_str("   [DEAD]");
    }
    let mut routes: Vec<String> = Vec::new();
    for sym in r.missing_symbols() {
        let a = &r.availability[sym.index()];
        let mut how: Vec<String> = Vec::new();
        if a.face_up > 0 {
            how.push(format!("{} on the table", a.face_up));
        }
        if a.in_unknown_pool > 0 {
            how.push(format!("{} face down", a.in_unknown_pool));
        }
        if a.in_future_age > 0 {
            how.push(format!("{} in a later age", a.in_future_age));
        }
        if a.via_mausoleum > 0 {
            how.push(format!("{} via the Mausoleum", a.via_mausoleum));
        }
        if a.via_law_board {
            how.push("the Law token on the board".to_string());
        }
        if a.via_law_great_library {
            how.push("the Law token via the Great Library".to_string());
        }
        if how.is_empty() {
            how.push("gone".to_string());
        }
        routes.push(format!("{}: {}", symbol_char(sym), how.join(", ")));
    }
    if !routes.is_empty() {
        s.push_str(&format!("\n      {}", routes.join("  |  ")));
    }
    let pairs: Vec<char> = ALL_SCIENCE
        .iter()
        .filter(|&&sym| r.pair_setup.candidates[sym.index()])
        .map(|&sym| symbol_char(sym))
        .collect();
    if !pairs.is_empty() {
        let token = r
            .pair_setup
            .best_board_token
            .map(|(t, v)| format!("{} (worth ~{v:.1} VP)", t.def().name))
            .unwrap_or_else(|| "nothing on the board".to_string());
        s.push_str(&format!("\n      half-pairs {pairs:?} would claim {token}"));
    }
    s
}

fn describe_vp(r: &VpRead) -> String {
    format!(
        "points   {:>3} vs {:<3} (gap {:+})   unbuilt wonders {} vs {}   guild lean {:+.1}   structural edge {:+.1}\n      \
         blue still out there: {:.0} on the table, {:.1} face down, {:.1} in later ages",
        r.my_total,
        r.their_total,
        r.gap,
        r.my_unbuilt_wonder_vp,
        r.their_unbuilt_wonder_vp,
        r.guild_lean,
        r.structural_edge,
        r.civilian_vp_face_up,
        r.civilian_vp_hidden,
        r.civilian_vp_future_ages,
    )
}

fn describe_action(state: &GameState, action: Action) -> String {
    match action {
        Action::Build { slot } => match state.face_up_card(slot) {
            Some(c) => format!("builds {} (slot {slot})", c.def().name),
            None => format!("builds slot {slot}"),
        },
        Action::Discard { slot } => match state.face_up_card(slot) {
            Some(c) => format!("discards {} (slot {slot}) for coins", c.def().name),
            None => format!("discards slot {slot}"),
        },
        Action::BuildWonder { slot, wonder } => match state.face_up_card(slot) {
            Some(c) => format!("spends {} to build {}", c.def().name, wonder.def().name),
            None => format!("builds {} with slot {slot}", wonder.def().name),
        },
        Action::PickWonder { wonder } => format!("drafts {}", wonder.def().name),
        Action::ChooseProgressToken { token } => format!("takes the {} token", token.def().name),
        Action::ChooseGreatLibraryToken { token } => {
            format!(
                "keeps the {} token from the Great Library",
                token.def().name
            )
        }
        Action::MausoleumBuild { card } => {
            format!("rebuilds {} from the discard pile", card.def().name)
        }
        Action::DestroyOpponentCard { card } => format!("destroys {}", card.def().name),
        Action::ChooseFirstPlayer { player } => format!("gives the first turn to {player}"),
    }
}

/// A short signature of every classification, so `--quiet` can show only the
/// turns where one of them moved.
fn classification_signature(state: &GameState) -> String {
    let board = Board::of(state);
    Player::ALL
        .iter()
        .map(|&p| {
            format!(
                "{p}:{}/{}",
                military_label(duels_strategy::military_read_with(state, p, &board).status),
                science_label(duels_strategy::science_read_with(state, p, &board).status),
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn print_position(state: &GameState, mover: Player) {
    let board = Board::of(state);
    println!(
        "  {}      age {}   {} cards left   coins {} / {}",
        pawn_bar(state.conflict()),
        state.age(),
        board.cards_left(),
        state.player(Player::One).coins(),
        state.player(Player::Two).coins(),
    );
    for p in Player::ALL {
        let tag = if p == mover { "*" } else { " " };
        println!(
            "  {tag}{p}  {}",
            describe_military(&military_read(state, p))
        );
        println!("      {}", describe_science(&science_read(state, p)));
    }
    println!("      {}", describe_vp(&vp_read(state, mover)));

    if state.phase() == Phase::Turn || state.phase() == Phase::WonderDraft {
        let s = stance(state, mover);
        println!("      stance for {mover}: {}", s.headline());
        let legal = engine::legal_actions(state);
        let mut scored: Vec<(Action, f64)> = legal
            .iter()
            .map(|&a| (a, action_prior(state, a, &s)))
            .collect();
        let total: f64 = scored.iter().map(|&(_, w)| w).sum();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let shown: Vec<String> = scored
            .iter()
            .take(4)
            .map(|&(a, w)| {
                format!(
                    "{} {:.0}%",
                    describe_action(state, a),
                    100.0 * w / total.max(f64::MIN_POSITIVE)
                )
            })
            .collect();
        println!(
            "      prior wants ({} legal): {}",
            legal.len(),
            shown.join("  |  ")
        );
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let seed: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(11);
    let one = args.next().unwrap_or_else(|| "greedy".to_string());
    let two = args.next().unwrap_or_else(|| "random".to_string());
    let quiet = std::env::args().any(|a| a == "--quiet");

    let mut agents: [Box<dyn Agent>; 2] =
        [make_agent(&one, seed ^ 0xA1), make_agent(&two, seed ^ 0xB2)];
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);

    println!("=======================================================================");
    println!(" watch_reads: seed {seed}   P1 = {one}   P2 = {two}");
    if quiet {
        println!(" --quiet: printing only the turns where a classification changed");
    }
    println!(" IMMINENT = wins next move if unopposed   |   live = reachable");
    println!(" pressure = worth forcing denial, not worth winning   |   closed / DEAD");
    println!("=======================================================================");

    let mut last_signature = String::new();
    let mut guard = 0u32;
    loop {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let mover = state.current_player();
        let signature = classification_signature(&state);
        let changed = signature != last_signature;
        last_signature = signature;

        if !quiet || changed {
            println!();
            println!(
                "--- turn {} ({:?}) --- {mover} to move{}",
                state.turn(),
                state.phase(),
                if changed && quiet {
                    "   [a classification changed]"
                } else {
                    ""
                }
            );
            print_position(&state, mover);
        }

        let obs = state.observation();
        let action = agents[mover.index()].choose(&obs, &legal, Budget::Nodes(1));
        if !quiet || changed {
            println!("  -> {mover} {}", describe_action(&state, action));
        }
        engine::apply_quiet(&mut state, action, &mut rng).expect("agents return legal actions");

        guard += 1;
        if guard > 400 {
            eprintln!("bailing out: the game did not terminate");
            break;
        }
    }

    println!();
    println!("=======================================================================");
    match state.result() {
        Some(r) => println!(" result: {r:?}"),
        None => println!(" the game did not finish"),
    }
    let [a, b] = duels_core::scoring::score(&state);
    println!(" final points: P1 {} vs P2 {}", a.total, b.total);
    println!("=======================================================================");
}
