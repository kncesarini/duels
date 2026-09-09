//! What colours does an agent actually take, and when does it discard?
//!
//! The single most direct check on whether this crate's design landed. A
//! strong human player, in Age I, mostly avoids red and blue (both are slow:
//! they develop nothing, and taking one can cost the "start the next age"
//! claim) and prefers grey, green and yellow. `greedy-ev`, the 1-ply
//! reference this example was written against, could not express that
//! preference at all — it had no term for what a card *produces* — so the
//! question this example answers is whether `phased`'s development term and
//! next-age-start tempo term move the profile in the direction a human would
//! recognise. `greedy-ev` has since been retired (see `docs/milestones.md`)
//! and is gone from the agent list below; `random` is the remaining
//! no-preference baseline to read the profile against.
//!
//! ```text
//! cargo run --release -p duels-agent-phased --example take_profile
//! cargo run --release -p duels-agent-phased --example take_profile -- 40 phased random mcts-uct
//! ```
//!
//! Arguments: the number of self-play games per agent (default 30), then the
//! agents to profile (default `phased random mcts-uct`). Each agent plays
//! itself, so the profile is that agent's own taste rather than a reaction to
//! somebody else's. `phased-v1` is the configuration this crate first
//! shipped with, and `phased:band=<x>` overrides the military band weight, so
//! a round of work can be read colour by colour against what it replaced.
//!
//! `mcts-uct` is a real search and is given `Nodes(2000)`, so it is far
//! slower than the 1-ply agents; 30 games of it takes a couple of minutes. It
//! is included as an empirical reference point rather than as a target: it is
//! the strongest agent in the repository, but its Age I appetite for red cards
//! is a known consequence of a rollout policy that does not understand the
//! game deeply, not something to imitate.
//!
//! The round-two numbers this produces, Age I keep rates over 40 games:
//!
//! ```text
//!             brown  grey   blue   green  yellow  red
//! phased      90.8   79.2   82.1   51.8   59.9    41.0
//! phased-v1   68.1   88.9   87.7   79.1   69.7     1.5
//! mcts-uct    80.2   76.4   73.6   25.9   78.9    50.7
//! ```
//!
//! Red at 1.5% was the symptom of the `next_age_start` magnitude bug; the
//! whole ladder from `phased:band=1.0` (20.9%) to `phased:band=2.5` (45.5%)
//! is tabulated in the crate docs, along with why the default sits where it
//! does.

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
        "mcts-uct" => Box::new(MctsAgent::new(seed)),
        "phased" => Box::new(PhasedAgent::new(seed)),
        // The configuration this crate shipped with, so the two rounds of
        // work can be compared colour by colour in one run.
        "phased-v1" => Box::new(PhasedAgent::with_config(
            seed,
            duels_agent_phased::Config::v1(),
        )),
        // `phased:band=<x>` overrides the one weight that visibly moves this
        // table — the military band model's — because "how much military does
        // this weight actually buy" is a question about colours, and this is
        // the diagnostic that answers it. Deliberately not a general spec
        // parser: `duels-arena`'s `agent_spec` is that, and this crate cannot
        // depend on it.
        _ if name.starts_with("phased:band=") => {
            let band: f64 = name["phased:band=".len()..].parse().unwrap_or(1.0);
            let mut config = duels_agent_phased::Config::default();
            config.eval.military_band = band;
            Box::new(PhasedAgent::with_config(seed, config))
        }
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
            "random".to_string(),
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
