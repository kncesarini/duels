//! How often is a decision actually spent on a discard?
//!
//! [`duels_agent_phased::terms::yellow_equity`] prices a commercial card by the
//! coins it will add to the discards its owner has *not made yet*, which needs
//! a rate: discards per decision, over the rest of the game. That number is a
//! property of how the agents in this repository actually play, not of the
//! rules, so it is measured here rather than guessed — the constant
//! [`duels_agent_phased::terms::DISCARD_RATE_PER_DECISION`] is whatever this
//! prints.
//!
//! ```text
//! cargo run --release -p duels-agent-phased --example discard_rate
//! cargo run --release -p duels-agent-phased --example discard_rate -- 200 5001
//! ```
//!
//! Arguments: the number of self-play games (default 200) and the base seed
//! (default 1). Run it on two disjoint seed ranges; a rate that moves between
//! them is a rate not worth hard-coding.
//!
//! Only *card* decisions count towards the denominator. The wonder draft, a
//! start-of-age first-player choice and the pending resolutions the four
//! mid-effect wonders create are turns at which a discard is not on the menu at
//! all, and folding them in would deflate the rate for a reason that has
//! nothing to do with how willing a player is to discard.

use duels_agent_phased::PhasedAgent;
use duels_agents_api::{Agent, Budget};
use duels_core::{engine, Action, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// One side's tally.
#[derive(Default, Clone, Copy)]
struct Tally {
    /// Decisions at which a `Discard` was legal.
    decisions: u64,
    /// ...of which a `Discard` was taken.
    discards: u64,
    /// Commercial cards in this player's city, summed over every decision, so
    /// the mean says whether a rate measured over all games is being pulled by
    /// yellow-heavy cities.
    yellow_at_decision: u64,
}

impl Tally {
    fn rate(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.discards as f64 / self.decisions as f64
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let base: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut by_age = [Tally::default(); 4];
    let mut total = Tally::default();

    for i in 0..games {
        let seed = base.wrapping_add(i);
        let mut agents = [
            PhasedAgent::new(seed * 1000 + 1),
            PhasedAgent::new(seed * 1000 + 2),
        ];
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xD15C_A2D0);
        while !state.is_over() {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let me = state.current_player();
            let obs = state.observation();
            let action = agents[me.index()].choose(&obs, &legal, Budget::Nodes(1));

            // Only count turns at which discarding was an option at all.
            if legal.iter().any(|a| matches!(a, Action::Discard { .. })) {
                let age = usize::from(state.age().min(3));
                let yellow = u64::from(state.player(me).count(
                    duels_core::data::CountTarget::Cards(duels_core::data::CardType::Commercial),
                ));
                for t in [&mut by_age[age], &mut total] {
                    t.decisions += 1;
                    t.yellow_at_decision += yellow;
                    if matches!(action, Action::Discard { .. }) {
                        t.discards += 1;
                    }
                }
            }
            engine::apply(&mut state, action, &mut rng).expect("agents play legal moves");
        }
    }

    println!(
        "discard rate over {games} phased self-play games from base seed {base}\n\
         (denominator: decisions at which a Discard was legal)\n"
    );
    for (age, t) in by_age.iter().enumerate().skip(1) {
        println!(
            "  age {age}      {:>7} / {:<7} = {:.4}   (mean yellow in city {:.2})",
            t.discards,
            t.decisions,
            t.rate(),
            t.yellow_at_decision as f64 / t.decisions.max(1) as f64,
        );
    }
    println!(
        "  overall    {:>7} / {:<7} = {:.4}   (mean yellow in city {:.2})",
        total.discards,
        total.decisions,
        total.rate(),
        total.yellow_at_decision as f64 / total.decisions.max(1) as f64,
    );
    println!(
        "\n  duels_agent_phased::terms::DISCARD_RATE_PER_DECISION is {:.3}",
        duels_agent_phased::terms::DISCARD_RATE_PER_DECISION
    );
    let _ = Player::ALL;
}
