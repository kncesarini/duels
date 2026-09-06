//! What colours does an agent actually take, and when does it discard?
//!
//! The single most direct check on whether this crate's design landed. A
//! strong human player, in Age I, mostly avoids red and blue (both are slow:
//! they develop nothing, and taking one can cost the "start the next age"
//! claim) and prefers grey, green and yellow. `greedy-ev` cannot express that
//! preference at all — it has no term for what a card *produces* — so the
//! question this example answers is whether `phased`'s development term and
//! next-age-start tempo term move the profile in the direction a human would
//! recognise.
//!
//! ```text
//! cargo run --release -p duels-agent-phased --example take_profile
//! cargo run --release -p duels-agent-phased --example take_profile -- 40 phased greedy-ev mcts-uct
//! ```
//!
//! Arguments: the number of self-play games per agent (default 30), then the
//! agents to profile (default `phased greedy-ev mcts-uct`). Each agent plays
//! itself, so the profile is that agent's own taste rather than a reaction to
//! somebody else's; `mcts-uct` is included as the empirical reference point —
//! it is the strongest agent in the repository, so whatever *it* does in Age I
//! is the closest thing available to ground truth.
//!
//! `mcts-uct` is a real search and is given `Nodes(2000)`, so it is far
//! slower than the two 1-ply agents; 30 games of it takes a couple of minutes.

use duels_agent_greedy_ev::GreedyEvAgent;
use duels_agent_mcts_uct::MctsAgent;
use duels_agent_phased::PhasedAgent;
use duels_agent_random::RandomAgent;
use duels_agents_api::{Agent, Budget};
use duels_core::data::CardType;
use duels_core::{engine, Action, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

const COLOURS: [(CardType, &str); 7] = [
    (CardType::RawMaterial, "brown"),
    (CardType::ManufacturedGood, "grey"),
    (CardType::Civilian, "blue"),
    (CardType::Scientific, "green"),
    (CardType::Commercial, "yellow"),
    (CardType::Military, "red"),
    (CardType::Guild, "purple"),
];

fn make_agent(name: &str, seed: u64) -> Box<dyn Agent> {
    match name {
        "random" => Box::new(RandomAgent::new(seed)),
        "greedy-ev" => Box::new(GreedyEvAgent::new(seed)),
        "mcts-uct" => Box::new(MctsAgent::new(seed)),
        "phased" => Box::new(PhasedAgent::new(seed)),
        other => {
            eprintln!("unknown agent {other:?}; using phased");
            Box::new(PhasedAgent::new(seed))
        }
    }
}

fn budget(name: &str) -> Budget {
    match name {
        "mcts-uct" | "alphabeta" => Budget::Nodes(2_000),
        _ => Budget::Nodes(1),
    }
}

/// Counts of what happened to each colour, per age.
#[derive(Default, Clone, Copy)]
struct Counts {
    built: [u32; 7],
    discarded: [u32; 7],
    wonder_fodder: [u32; 7],
}

impl Counts {
    fn total(&self, i: usize) -> u32 {
        self.built[i] + self.discarded[i] + self.wonder_fodder[i]
    }

    fn builds(&self) -> u32 {
        self.built.iter().sum()
    }
}

/// `100 × a / b`, or zero when `b` is zero.
fn pct(a: u32, b: u32) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * f64::from(a) / f64::from(b)
    }
}

fn profile(name: &str, games: u64) -> [Counts; 3] {
    let mut per_age = [Counts::default(); 3];
    for seed in 0..games {
        let mut agents = [
            make_agent(name, seed * 1000 + 1),
            make_agent(name, seed * 1000 + 2),
        ];
        let b = budget(name);
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xF00D);
        for _ in 0..600 {
            if state.is_over() {
                break;
            }
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let me = state.current_player();
            let age = usize::from(state.age().max(1)) - 1;
            let obs = state.observation();
            let action = agents[me.index()].choose(&obs, &legal, b);

            let slot_card = |slot: u8| state.face_up_card(slot);
            match action {
                Action::Build { slot } => {
                    if let Some(card) = slot_card(slot) {
                        per_age[age].built[card.def().kind.index()] += 1;
                    }
                }
                Action::Discard { slot } => {
                    if let Some(card) = slot_card(slot) {
                        per_age[age].discarded[card.def().kind.index()] += 1;
                    }
                }
                Action::BuildWonder { slot, .. } => {
                    if let Some(card) = slot_card(slot) {
                        per_age[age].wonder_fodder[card.def().kind.index()] += 1;
                    }
                }
                _ => {}
            }
            engine::apply_quiet(&mut state, action, &mut rng)
                .expect("agents must play legal moves");
        }
        let _ = Player::ALL;
    }
    per_age
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let games: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(30);
    let names: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        vec![
            "phased".to_string(),
            "greedy-ev".to_string(),
            "mcts-uct".to_string(),
        ]
    };

    println!("self-play take profile over {games} games per agent\n");
    println!(
        "  keep%  = of the cards of that colour that appeared, how many were built\n\
         \x20        (the rest were discarded for coins or spent under a wonder)\n\
         \x20 mix%  = of everything this agent built in that age, what share was that colour\n\
         \x20\n\
         \x20Note there is no \"which colours did it take\" figure: in self-play the two\n\
         \x20seats between them take every card of every age, so the colours *seen* are\n\
         \x20just the deal. What an agent chooses is whether to build a card or cash it in,\n\
         \x20which is what keep% and mix% measure.\n"
    );

    for name in &names {
        let per_age = profile(name, games);
        println!("=== {name} ===");
        for (age, counts) in per_age.iter().enumerate() {
            if counts.builds() == 0 {
                continue;
            }
            let builds = counts.builds();
            print!("  age {} keep%  ", age + 1);
            for (i, (_, label)) in COLOURS.iter().enumerate() {
                print!("{label} {:>5.1}  ", pct(counts.built[i], counts.total(i)));
            }
            println!();
            print!("  age {} mix%   ", age + 1);
            for (i, (_, label)) in COLOURS.iter().enumerate() {
                print!("{label} {:>5.1}  ", pct(counts.built[i], builds));
            }
            println!("\n");
        }
    }
}
