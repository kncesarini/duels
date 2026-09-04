//! Play a self-play game and narrate what the race reads say after every move.
//!
//! This is the human checkpoint for this crate. The point is to read it and
//! judge whether the *numbers* match how you read those positions: is a race
//! at `M = 0.48` really about even? Is the symbol the model says the threat
//! holder "secures" the one you would take? Is the action the denial channel
//! prices highest the one you would play?
//!
//! ```text
//! cargo run --release -p duels-strategy --example watch_reads
//! cargo run --release -p duels-strategy --example watch_reads -- 7 greedy random
//! cargo run --release -p duels-strategy --example watch_reads -- 7 greedy random --quiet
//! cargo run --release -p duels-strategy --example watch_reads -- --calibration
//! ```
//!
//! Arguments, all optional: `seed`, `player-one agent`, `player-two agent`
//! (`greedy` or `random`), and `--quiet` to print only the turns where a
//! classification changed.
//!
//! `--calibration` skips the game entirely and renders the hand-built
//! positions from `tests/threat_calibration.rs` instead — the four-symbol
//! race, the three places the Law token can be, and the chained extra turn —
//! so the numbers those tests assert can be read in context rather than as
//! bare floats in an assertion message.
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
    action_prior, delta_m, deny_vp, military_read_with, science_read_with, stance_in, vp_read_with,
    Context, MilitaryRead, MilitaryStatus, ScienceRead, ScienceStatus, Tempo, VpRead, VpWeights,
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

/// A magnitude as a bar, so a column of them is scannable.
fn bar(m: f64) -> String {
    let filled = (m * 10.0).round().clamp(0.0, 10.0) as usize;
    format!("[{}{}]", "#".repeat(filled), ".".repeat(10 - filled))
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

fn describe_tempo(t: &Tempo) -> String {
    format!(
        "tempo    {:>2} decisions left ({:.1} effective)   share {:.0}%   chain {} now / {:.1} expected{}",
        t.decisions_left,
        t.decisions_left_eff,
        100.0 * t.share,
        t.chain,
        t.extra_expected,
        if t.banked { "   [extra turn banked]" } else { "" },
    )
}

fn describe_military(r: &MilitaryRead) -> String {
    let mut s = format!(
        "military {:<8} M {:.2} {}   need {:>2}   now {:>2} (best {:>1}, fork {})   table {:>2} + {:.1} hidden + {:.1} to come",
        military_label(r.status),
        r.magnitude,
        bar(r.magnitude),
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
            "\n      closes on round {t} of the simulation (chain {}, {:.2} shields/card of stream, defender answers with {})",
            r.model.chain, r.model.avg_stream, r.model.defender_best_single,
        )),
        None => s.push_str(&format!(
            "\n      never closes within {:.0} rounds (defender answers with {})",
            r.model.horizon, r.model.defender_best_single
        )),
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
        "science  {:<8} M {:.2} {}   {} of 6 held [{held}]   missing {}   routes {}   fragility {}",
        science_label(r.status),
        r.magnitude,
        bar(r.magnitude),
        r.distinct,
        r.missing,
        r.obtainable_missing,
        r.fragility,
    );
    if r.dead {
        s.push_str("   [DEAD]");
    }
    s.push_str(&format!(
        "\n      surface {:.2} x (1 - P(stopped) {:.2}), slack {}{}",
        r.detail.surface,
        r.detail.p_stop,
        r.detail.slack,
        match r.detail.secured {
            Some(sym) => format!(
                "   secures {} next turn (reachable slots {:?})",
                symbol_char(sym),
                slot_list(r.model.symbols[sym.index()].reachable_slots)
            ),
            None => String::new(),
        }
    ));
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
        let kill = r.kill_cost(sym);
        let kill_text = if kill.is_finite() {
            format!(
                "kill {kill:.2} turns -> P {:.2}",
                r.model.share_defender.powf(kill)
            )
        } else {
            "UNDENIABLE".to_string()
        };
        routes.push(format!(
            "{}: c {:.2} ({}) {kill_text}",
            symbol_char(sym),
            r.copies(sym),
            how.join(", "),
        ));
    }
    for line in routes {
        s.push_str(&format!("\n        {line}"));
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
        s.push_str(&format!(
            "\n      half-pairs {pairs:?} ({}) would claim {token}",
            if r.pair_setup.has_live_half_pair() {
                "completable"
            } else {
                "second copy gone"
            }
        ));
    }
    s
}

fn slot_list(mask: u32) -> Vec<u8> {
    duels_strategy::board::iter_slots(mask).collect()
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
            Some(c) => format!("build {} (slot {slot})", c.def().name),
            None => format!("build slot {slot}"),
        },
        Action::Discard { slot } => match state.face_up_card(slot) {
            Some(c) => format!("discard {} (slot {slot})", c.def().name),
            None => format!("discard slot {slot}"),
        },
        Action::BuildWonder { slot, wonder } => match state.face_up_card(slot) {
            Some(c) => format!("spend {} on {}", c.def().name, wonder.def().name),
            None => format!("build {} with slot {slot}", wonder.def().name),
        },
        Action::PickWonder { wonder } => format!("draft {}", wonder.def().name),
        Action::ChooseProgressToken { token } => format!("take the {} token", token.def().name),
        Action::ChooseGreatLibraryToken { token } => {
            format!("keep the {} token from the Great Library", token.def().name)
        }
        Action::MausoleumBuild { card } => {
            format!("rebuild {} from the discard pile", card.def().name)
        }
        Action::DestroyOpponentCard { card } => format!("destroy {}", card.def().name),
        Action::ChooseFirstPlayer { player } => format!("give the first turn to {player}"),
    }
}

/// A short signature of every classification, so `--quiet` can show only the
/// turns where one of them moved.
fn classification_signature(state: &GameState) -> String {
    let ctx = Context::of(state);
    Player::ALL
        .iter()
        .map(|&p| {
            format!(
                "{p}:{}/{}",
                military_label(military_read_with(state, p, &ctx).status),
                science_label(science_read_with(state, p, &ctx).status),
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn print_position(state: &GameState, mover: Player) {
    let ctx = Context::of(state);
    println!(
        "  {}      age {}   {} cards left   coins {} / {}",
        pawn_bar(state.conflict()),
        state.age(),
        ctx.board.cards_left(),
        state.player(Player::One).coins(),
        state.player(Player::Two).coins(),
    );
    for p in Player::ALL {
        let tag = if p == mover { "*" } else { " " };
        println!("  {tag}{p}  {}", describe_tempo(ctx.tempo(p)));
        println!(
            "      {}",
            describe_military(&military_read_with(state, p, &ctx))
        );
        println!(
            "      {}",
            describe_science(&science_read_with(state, p, &ctx))
        );
    }
    println!(
        "      {}",
        describe_vp(&vp_read_with(state, mover, &ctx, &VpWeights::default()))
    );

    if state.phase() == Phase::Turn || state.phase() == Phase::WonderDraft {
        let s = stance_in(state, mover, Default::default(), &ctx);
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

        // The denial channel, on its own: what each move does to the
        // *opponent's* two magnitudes, and what that is worth in points.
        let mut denial: Vec<(Action, f64, duels_strategy::DeltaM)> = legal
            .iter()
            .map(|&a| (a, deny_vp(a, &s), delta_m(a, &s)))
            .collect();
        denial.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if denial.iter().any(|&(_, v, _)| v.abs() > 0.005) {
            println!("      denial value (dM_sci / dM_mil -> VP):");
            for &(a, v, d) in denial.iter().take(4) {
                println!(
                    "        {:<44} {:+.3} / {:+.3} -> {v:+.2} VP{}",
                    describe_action(state, a),
                    d.science,
                    d.military,
                    if d.breaks_certainty {
                        "   [BREAKS A CERTAIN WIN]"
                    } else {
                        ""
                    }
                );
            }
            if let Some(&(a, v, d)) = denial.last() {
                if denial.len() > 4 {
                    println!(
                        "        ...worst: {:<34} {:+.3} / {:+.3} -> {v:+.2} VP",
                        describe_action(state, a),
                        d.science,
                        d.military
                    );
                }
            }
        }
    }
}

/// The hand-built positions the calibration tests assert on, rendered with
/// the same narration as a real game.
fn print_calibration() {
    use duels_core::testing::StateBuilder;

    const FOUR_EARLY: &[&str] = &["pharmacist", "workshop", "scriptorium", "apothecary"];
    const FIVE_SYMBOLS: &[&str] = &[
        "pharmacist",
        "workshop",
        "scriptorium",
        "apothecary",
        "university",
    ];
    const FIVE_ASIDE: &[&str] = &["law", "philosophy", "agriculture", "economy", "theology"];

    let four_late = |age: u8| {
        let slots: [(u8, &str); 2] = if age == 1 {
            [(18, "lumber-yard"), (19, "clay-pool")]
        } else {
            [(18, "sawmill"), (19, "brickyard")]
        };
        StateBuilder::new()
            .age(age)
            .open_slots(&slots)
            .built(Player::One, FOUR_EARLY)
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
    };
    let balance_last = || {
        StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "town-hall")])
            .built(Player::One, FIVE_SYMBOLS)
            .built(Player::Two, &["academy", "study"])
            .coins(Player::One, 40)
            .current(Player::One)
    };

    let cases: Vec<(&str, GameState)> = vec![
        (
            "four early symbols, end of Age II, nothing else in play",
            four_late(2).build(),
        ),
        (
            "the same at the end of Age I (must read identically)",
            four_late(1).build(),
        ),
        (
            "the same plus the Law token on the board",
            four_late(2).board_tokens(&["law", "philosophy"]).build(),
        ),
        (
            "five symbols, Balance the last one, Law ON THE BOARD",
            balance_last().board_tokens(&["law"]).build(),
        ),
        (
            "five symbols, Balance the last one, Law SET ASIDE, no wonder",
            balance_last().set_aside_tokens(FIVE_ASIDE).build(),
        ),
        (
            "five symbols, Balance the last one, Law SET ASIDE + Great Library",
            balance_last()
                .set_aside_tokens(FIVE_ASIDE)
                .wonders(Player::One, &["the-great-library"])
                .build(),
        ),
        (
            "five symbols, the sixth one card deep, an affordable Sphinx",
            StateBuilder::new()
                .age(3)
                .open_slots(&[(15, "academy"), (18, "palace"), (19, "town-hall")])
                .built(Player::One, FIVE_SYMBOLS)
                .wonders(Player::One, &["the-sphinx"])
                .coins(Player::One, 40)
                .coins(Player::Two, 40)
                .current(Player::One)
                .build(),
        ),
        (
            "two missing symbols in the discard pile, one Mausoleum",
            StateBuilder::new()
                .age(3)
                .open_slots(&[
                    (15, "palace"),
                    (16, "town-hall"),
                    (17, "obelisk"),
                    (18, "senate"),
                    (19, "gardens"),
                ])
                .built(
                    Player::One,
                    &["pharmacist", "workshop", "scriptorium", "university"],
                )
                .built(Player::Two, &["study", "school"])
                .discard(&["apothecary", "academy"])
                .wonders(Player::One, &["the-mausoleum"])
                .coins(Player::One, 40)
                .current(Player::One)
                .build(),
        ),
        (
            "Theology plus an affordable Colossus, four shields from the capital",
            StateBuilder::new()
                .age(3)
                .open_slots(&[(18, "fortifications"), (19, "palace"), (15, "town-hall")])
                .wonders(Player::One, &["the-colossus"])
                .tokens(Player::One, &["theology"])
                .conflict(5)
                .coins(Player::One, 40)
                .coins(Player::Two, 3)
                .current(Player::One)
                .build(),
        ),
    ];

    for (name, st) in cases {
        println!();
        println!("--- {name} ---");
        print_position(&st, st.current_player());
    }
}

fn main() {
    if std::env::args().any(|a| a == "--calibration") {
        println!("=======================================================================");
        println!(" watch_reads --calibration: the positions tests/threat_calibration.rs");
        println!(" asserts on, rendered rather than asserted.");
        println!("=======================================================================");
        print_calibration();
        return;
    }
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
    println!(" M is the win probability of that race from public information alone.");
    println!(" IMMINENT = M 1.00 (certain if unopposed)   |   live = M >= 0.25");
    println!(" pressure = M >= 0.05, worth forcing denial  |   closed / DEAD");
    println!(" dM is what a move does to the OPPONENT's M: positive denies, negative gifts.");
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
