//! `duels-agent-mcts-uct`: Monte Carlo Tree Search with UCT selection and
//! **explicit chance nodes**, intended as the project's strongest non-learned
//! baseline and the yardstick for everything that comes after it.
//!
//! # Why chance nodes
//!
//! 7 Wonders Duel is a two-player zero-sum *stochastic* game: the cards
//! behind the face-down slots of the current age are unknown when a move is
//! chosen, and taking a card can uncover them. There is no player-private
//! information — both players always see the same public state — so the game
//! is much simpler than poker, but it is not Go: a plain alternating-move
//! tree would silently pretend the reveals were part of the mover's choice.
//!
//! This agent therefore builds a tree with three kinds of node:
//!
//! - **decision** nodes, one player to move, children = the legal actions,
//!   selected by UCB1;
//! - **chance** nodes, inserted between an action and the position it leads
//!   to whenever the engine says the action resolves randomness, children =
//!   possible reveals, selected **by their real probability**, never by UCB1;
//! - **terminal** nodes, where the [`duels_core::GameResult`] is settled.
//!
//! # How chance is handled, precisely
//!
//! Worth being explicit about, because it bounds how strong the agent can
//! get:
//!
//! 1. **The root is determinized.** `choose` only ever sees an
//!    [`Observation`], so it calls [`Observation::sample_state`] once per
//!    call to get one concrete world consistent with public knowledge.
//! 2. **Reveals inside the tree are *not* taken from that world.** Every
//!    chance node re-draws its outcome from the distribution the engine
//!    computes from public information alone
//!    (`engine::chance_outcomes`/`hidden_info`), and applies it with
//!    `engine::apply_with_outcome`, which rewrites the hidden layout to stay
//!    publicly consistent. So the tree integrates over reveals rather than
//!    committing to the root's guess, and the agent can never exploit
//!    knowledge of a card it should not know.
//! 3. **The draw is exact but not enumerated.** A two-slot reveal has
//!    hundreds of outcomes; the `chance` module draws from that distribution
//!    in O(1) and reports the drawn outcome's exact probability, which is
//!    verified statistically against `engine::chance_outcomes`.
//! 4. **Progressive widening is an approximation.** A chance node that
//!    created a fresh child on every visit would be a perfectly unbiased
//!    estimator of the expectation but would never let the tree grow past it.
//!    By default the number of distinct outcome children grows as
//!    `sqrt(visits)` and further visits re-select an existing child in
//!    proportion to its probability. That reweighting is the one place the
//!    search is not a faithful expectation; set
//!    [`Config::chance_widen_alpha`] to `1.0` with a large
//!    [`Config::chance_widen_c`] to recover the unbiased estimator.
//! 5. **Two sources of randomness are only root-determinized:** the
//!    composition and order of the *next* age's deck, and the four wonders
//!    not yet offered during the draft. Neither is exposed through the
//!    per-action chance API, so they stay fixed for the duration of one
//!    search.
//!
//! # Root ensembling (`Config::root_determinizations`)
//!
//! Point 5 is what [`Config::root_determinizations`] addresses. With `N > 1`,
//! `choose` samples `N` independent worlds, grows a separate tree in each
//! with `1/N` of the budget, and plays the move with the most root visits
//! summed across the trees (see `tree::best_of`). That is the standard
//! Perfect Information Monte Carlo ensemble at the root: no single guess at
//! the hidden information can decide the move on its own, at the price of `N`
//! times fewer samples per tree.
//!
//! ## What it measures
//!
//! Paired, seat-swapped matches against this same agent at `N = 1`, run
//! through `duels-arena`'s `ensemble_lab` example, `+/-` one binomial
//! standard error. The `N = 1` row is a **control**: the identical
//! configuration against itself, which should score 50% and is the honest
//! yardstick for how much of each column is noise.
//!
//! | N | `Nodes(2_000)`, 400 games | `TimeMs(20)`, 200 games |
//! |---|---|---|
//! | 1 (control) | 46.4% +/- 2.5 | 47.5% +/- 3.5 |
//! | 2 | 51.5% +/- 2.5 | 47.5% +/- 3.5 |
//! | 4 | 48.0% +/- 2.5 | 45.0% +/- 3.5 |
//! | 8 | 43.2% +/- 2.5 | 42.0% +/- 3.5 |
//!
//! (Games run in parallel across seeds with the pool capped at 4-6 threads,
//! on a 14-core machine that was also busy with other work. That lowers the
//! *operating point* of a `TimeMs` row — 20 ms buys fewer simulations on a
//! loaded box — but not its fairness: the two sides alternate inside one
//! thread, and the harness prints each side's wall clock per game as the
//! check that they were contended alike.)
//!
//! **No gain at any `N`, at either budget kind.** Every cell sits within
//! about two standard errors of the control, so the honest summary is "no
//! measurable effect at `N = 2`, and a mild loss by `N = 8`" — not a win, and
//! not the disaster a naive reading of "each tree only gets an eighth of the
//! budget" would predict either.
//!
//! The split itself is fair, which is what makes those numbers about the
//! technique rather than about lost work. Under `Budget::Nodes` every `N`
//! spends *exactly* the budget — `ensemble_lab --cost` reports 2000
//! simulations per decision at `N = 1` and at `N = 8` alike — and the
//! measured wall clock per game agrees to within about 1% across the whole
//! sweep. Under `Budget::TimeMs` the same tool reports 32-34k simulations per
//! decision at every `N`, against a repeated `N = 1` baseline that itself
//! wandered between 23k and 32k with the machine's load: the per-slice
//! overhead (one determinization and one tree allocation) is far below the
//! noise, which is what the shared, chained deadlines in `Slices` are for.
//!
//! Four times the budget does not change the answer either: at
//! `Nodes(8_000)` over 200 games, `N = 2` scores 50.0% +/- 3.5 against
//! `N = 1`, with a 49.5% +/- 3.5 control. Dead level, which is the same "no
//! effect" as the `Nodes(2_000)` column read against its own control.
//!
//! The reading: this tree already integrates over every reveal the chance API
//! exposes (point 2 above), so a fresh determinization re-rolls only what the
//! API does not cover — the next age's deal, the undrafted wonders, and the
//! identities a *playout* walks into past the tree's edge. A playout under a
//! kind-level random policy extracts very little from knowing those
//! identities, so there is little bias to average away; meanwhile halving the
//! samples that decide the move actually being made is an immediate,
//! certain cost. Pooling visit counts across trees is itself a small
//! variance reduction, which is presumably why the loss stays as mild as it
//! does.
//!
//! `N = 1` therefore stays the default, and this is an opt-in knob — kept
//! because a measured negative result is worth more than a remembered
//! intuition, and because the same question at a different budget (a
//! one-second move, say) is now one command away:
//!
//! ```text
//! cargo run --release -p duels-arena --example ensemble_lab -- \
//!     --a mcts-uct:dets=4 --b mcts-uct:dets=1 --games 400 --budget nodes:2000
//! ```
//!
//! The value convention (every node accumulates the result from
//! [`duels_core::Player::One`]'s perspective; the zero-sum flip happens once,
//! at selection) and the widening rule are documented in the `tree` module.
//!
//! # Example
//!
//! ```
//! use duels_agent_mcts_uct::MctsAgent;
//! use duels_agents_api::{Agent, Budget};
//! use duels_core::engine;
//!
//! let mut agent = MctsAgent::new(7);
//! let state = engine::new_game(7);
//! let legal = engine::legal_actions(&state);
//! let action = agent.choose(&state.observation(), &legal, Budget::Nodes(64));
//! assert!(legal.contains(&action));
//! ```

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

mod chance;
mod rollout;
mod tree;

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::{engine, Action, Observation};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub use rollout::RolloutWeights;
pub use tree::Config;

/// Monte Carlo Tree Search with UCT selection and explicit chance nodes.
#[derive(Debug)]
pub struct MctsAgent {
    cfg: Config,
    rng: StdRng,
    /// Simulations run over the agent's whole lifetime, for throughput
    /// reporting.
    total_simulations: u64,
    /// Nodes allocated during the most recent search.
    last_tree_size: usize,
}

impl MctsAgent {
    /// A new agent with the default configuration, seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, Config::default())
    }

    /// A new agent with an explicit configuration.
    pub fn with_config(seed: u64, cfg: Config) -> Self {
        Self {
            cfg,
            rng: StdRng::seed_from_u64(seed),
            total_simulations: 0,
            last_tree_size: 0,
        }
    }

    /// The configuration in force.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Total simulations this agent has run since it was created.
    pub fn total_simulations(&self) -> u64 {
        self.total_simulations
    }

    /// Nodes allocated by the most recent `choose` call.
    pub fn last_tree_size(&self) -> usize {
        self.last_tree_size
    }
}

impl Agent for MctsAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "mcts-uct".to_string(),
            version: "1.0.0".to_string(),
            params: self.cfg.describe(),
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action], budget: Budget) -> Action {
        assert!(
            !legal.is_empty(),
            "choose must not be called with no legal actions"
        );
        if legal.len() == 1 {
            return legal[0];
        }

        // `N` determinized worlds consistent with the observation, each with
        // its own tree and its own share of the budget. Hidden reveals
        // *inside* a search are re-drawn from public knowledge at each chance
        // node, so a world only fixes what the chance API does not cover
        // (future age decks, the undrafted wonder pool) — which is exactly
        // what a second determinization varies.
        let n = self.cfg.root_determinizations.max(1);
        let mut slices = Slices::new(budget, n);
        let mut trees: Vec<tree::Tree> = Vec::with_capacity(n);
        let mut offered: Option<Vec<Action>> = None;

        for i in 0..n {
            let root = obs.sample_state(&mut self.rng);

            // The offered actions and the determinized state must agree,
            // since legality is a function of public information only; filter
            // defensively so an unexpected mismatch can never return an
            // action the arena did not offer. Public legality does not vary
            // between determinizations, so this is settled once.
            let actions = match &offered {
                Some(actions) => actions.clone(),
                None => {
                    let mut actions: Vec<Action> = legal
                        .iter()
                        .copied()
                        .filter(|&a| engine::is_legal(&root, a))
                        .collect();
                    debug_assert_eq!(
                        actions.len(),
                        legal.len(),
                        "a determinized root disagreed with the offered legal actions"
                    );
                    if actions.is_empty() {
                        actions = legal.to_vec();
                    }
                    offered = Some(actions.clone());
                    actions
                }
            };

            let mut tree = tree::Tree::new(root, actions, self.cfg, &mut self.rng);
            slices.run(&mut tree, i, &mut self.rng);
            self.total_simulations += tree.simulations;
            trees.push(tree);
        }

        self.last_tree_size = trees.iter().map(|t| t.nodes.len()).sum();

        let chosen = tree::best_of(&trees).unwrap_or(legal[0]);
        if legal.contains(&chosen) {
            chosen
        } else {
            // Unreachable given the filter above; never hand back an action
            // the caller did not offer.
            legal[0]
        }
    }
}

/// One search budget, divided into `n` equal slices — one per root
/// determinization.
///
/// # How a slice is sized
///
/// A node budget is partitioned exactly: every slice gets `total / n`
/// simulations and the first `total % n` slices get one more, so the slices
/// sum to the whole budget however indivisible it is (a `Nodes(20)` budget
/// over 3 trees is `7 + 7 + 6`, not `6 + 6 + 6`).
///
/// A time budget is sliced by *absolute* deadlines measured from one shared
/// start — slice `i` ends at `start + total*(i+1)/n` — rather than by giving
/// each tree its own `total/n` milliseconds. That matters because a tree only
/// checks the clock every [`Config::time_check_interval`] simulations: with
/// per-tree stopwatches each overshoot would add to the total, while with
/// chained deadlines an overshooting slice eats into the next one instead and
/// only the last slice's overshoot escapes.
///
/// With `n == 1` both arms reduce to what this crate did before ensembling
/// existed: `total.max(1)` simulations, or a single deadline
/// `total` milliseconds after the first simulation.
#[derive(Debug)]
enum Slices {
    Nodes {
        total: u64,
        n: u64,
    },
    Time {
        total_ms: u64,
        n: u64,
        /// Captured on the first slice, so that the clock starts where the
        /// pre-ensemble code started it.
        start: Option<std::time::Instant>,
    },
}

impl Slices {
    fn new(budget: Budget, n: usize) -> Self {
        let n = n.max(1) as u64;
        match budget {
            Budget::Nodes(total) => Slices::Nodes { total, n },
            Budget::TimeMs(total_ms) => Slices::Time {
                total_ms,
                n,
                start: None,
            },
        }
    }

    /// Run slice `i` of the budget on `tree`.
    fn run(&mut self, tree: &mut tree::Tree, i: usize, rng: &mut StdRng) {
        let i = i as u64;
        match self {
            Slices::Nodes { total, n } => {
                // Written as a quotient plus a remainder rather than as
                // `total*(i+1)/n - total*i/n` so that a `Nodes(u64::MAX)`
                // budget cannot overflow the multiplication.
                let sims = *total / *n + u64::from(i < *total % *n);
                // A slice of zero still needs one simulation, otherwise there
                // are no visited children to choose between.
                for _ in 0..sims.max(1) {
                    tree.simulate(rng);
                }
            }
            Slices::Time { total_ms, n, start } => {
                // The workspace bans wall-clock reads so that the engine and
                // its agents stay reproducible from a seed; `Budget::TimeMs`
                // is the one place an agent is *asked* to read the clock, and
                // the read is confined to this function. `Budget::Nodes`
                // remains fully deterministic.
                #[allow(clippy::disallowed_methods)]
                let from = *start.get_or_insert_with(std::time::Instant::now);
                // In `u128` so that a `TimeMs(u64::MAX)` budget cannot
                // overflow the multiplication either.
                let elapsed_ms = u128::from(*total_ms) * u128::from(i + 1) / u128::from(*n);
                let deadline = from + std::time::Duration::from_millis(elapsed_ms as u64);
                let interval = tree.cfg.time_check_interval.max(1);
                loop {
                    for _ in 0..interval {
                        tree.simulate(rng);
                    }
                    #[allow(clippy::disallowed_methods)]
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_agent_random::RandomAgent;
    use duels_core::{GameResult, Player};

    /// A small budget: enough that the tree is exercised (root expansion,
    /// chance nodes, UCB1 re-selection) while keeping `cargo test` quick.
    const CI_BUDGET: Budget = Budget::Nodes(48);

    fn play(seed: u64, mcts_seat: Player, budget: Budget) -> (GameResult, u64) {
        let mut mcts = MctsAgent::new(seed ^ 0x0BAD_1DEA_0BAD_1DEA);
        let mut opponent = RandomAgent::new(seed ^ 0x5EED_5EED);
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);

        let mut plies = 0u32;
        loop {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = if state.current_player() == mcts_seat {
                mcts.choose(&obs, &legal, budget)
            } else {
                opponent.choose(&obs, &legal, budget)
            };
            assert!(
                legal.contains(&action),
                "agent returned an illegal action {action:?}"
            );
            engine::apply(&mut state, action, &mut rng).expect("a legal action");
            plies += 1;
            assert!(plies < 5_000, "game did not terminate after {plies} plies");
        }
        (
            state.result().expect("a finished game has a result"),
            mcts.total_simulations(),
        )
    }

    #[test]
    fn spec_reports_the_expected_name_version_and_params() {
        let agent = MctsAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "mcts-uct");
        assert_eq!(spec.version, "1.0.0");
        assert!(spec.params.contains("c=1.000"), "{}", spec.params);
        assert!(spec.params.contains("chance="), "{}", spec.params);
        assert!(spec.params.contains("rollout="), "{}", spec.params);
    }

    #[test]
    fn a_single_legal_action_is_returned_without_searching() {
        let mut agent = MctsAgent::new(3);
        let state = engine::new_game(3);
        let only = [engine::legal_actions(&state)[0]];
        let chosen = agent.choose(&state.observation(), &only, Budget::Nodes(10_000));
        assert_eq!(chosen, only[0]);
        assert_eq!(agent.total_simulations(), 0, "no search was needed");
    }

    #[test]
    fn every_returned_action_is_one_of_the_offered_ones() {
        let mut agent = MctsAgent::new(11);
        let state = engine::new_game(11);
        let legal = engine::legal_actions(&state);
        for _ in 0..5 {
            let a = agent.choose(&state.observation(), &legal, Budget::Nodes(20));
            assert!(legal.contains(&a));
        }
    }

    #[test]
    fn a_node_budget_runs_exactly_that_many_simulations() {
        let mut agent = MctsAgent::new(5);
        let state = engine::new_game(5);
        let legal = engine::legal_actions(&state);
        agent.choose(&state.observation(), &legal, Budget::Nodes(37));
        assert_eq!(agent.total_simulations(), 37);
        agent.choose(&state.observation(), &legal, Budget::Nodes(3));
        assert_eq!(agent.total_simulations(), 40);
    }

    #[test]
    fn a_time_budget_returns_promptly_and_does_some_work() {
        let mut agent = MctsAgent::new(9);
        let state = engine::new_game(9);
        let legal = engine::legal_actions(&state);
        let a = agent.choose(&state.observation(), &legal, Budget::TimeMs(20));
        assert!(legal.contains(&a));
        assert!(agent.total_simulations() > 0);
    }

    #[test]
    fn a_node_budget_is_reproducible_from_the_seed() {
        let state = engine::new_game(21);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        let pick = |seed: u64| {
            let mut agent = MctsAgent::new(seed);
            agent.choose(&obs, &legal, Budget::Nodes(200))
        };
        assert_eq!(pick(4), pick(4));
    }

    /// `choose` exactly as it read before root ensembling existed: one
    /// determinization, one tree, the whole node budget, the pre-ensemble
    /// move-selection rule.
    ///
    /// This is the reference the `root_determinizations = 1` path is checked
    /// against. It is a copy on purpose — a test that called the new code
    /// would prove nothing.
    fn legacy_choose(
        rng: &mut StdRng,
        cfg: Config,
        obs: &Observation,
        legal: &[Action],
        nodes: u64,
    ) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let root = obs.sample_state(rng);
        let mut actions: Vec<Action> = legal
            .iter()
            .copied()
            .filter(|&a| engine::is_legal(&root, a))
            .collect();
        if actions.is_empty() {
            actions = legal.to_vec();
        }
        let mut tree = tree::Tree::new(root, actions, cfg, rng);
        for _ in 0..nodes.max(1) {
            tree.simulate(rng);
        }
        let chosen = tree::legacy_best_action(&tree).unwrap_or(legal[0]);
        if legal.contains(&chosen) {
            chosen
        } else {
            legal[0]
        }
    }

    /// The equivalence that makes the option safe to add: with
    /// `root_determinizations = 1` the agent is bit-for-bit the agent this
    /// crate shipped before, under a node budget — same determinization drawn
    /// from the same RNG stream, same number of simulations, same
    /// move-selection rule, same move.
    ///
    /// Checked over whole games rather than only at the opening position, so
    /// that the RNG streams have to stay in step across dozens of `choose`
    /// calls, chance nodes, pending choices and all.
    #[test]
    fn one_determinization_is_the_pre_ensemble_agent_move_for_move() {
        for seed in 0..8u64 {
            let mut agent = MctsAgent::with_config(
                seed,
                Config {
                    root_determinizations: 1,
                    ..Config::default()
                },
            );
            // The same seed, so the same stream, driven by the copy above.
            let mut legacy_rng = StdRng::seed_from_u64(seed);
            let cfg = Config::default();

            let mut state = engine::new_game(seed ^ 0xC0FF_EE00);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xFEED);
            let mut decisions = 0u32;
            loop {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let obs = state.observation();
                let budget = 24 + u64::from(decisions % 7);
                let got = agent.choose(&obs, &legal, Budget::Nodes(budget));
                let want = legacy_choose(&mut legacy_rng, cfg, &obs, &legal, budget);
                assert_eq!(
                    got, want,
                    "seed {seed}, decision {decisions}: ensembling changed the N=1 move"
                );
                engine::apply(&mut state, got, &mut rng).expect("a legal action");
                decisions += 1;
                assert!(decisions < 5_000);
            }
            assert!(decisions > 20, "the game was too short to prove much");
        }
    }

    /// A node budget is partitioned, not multiplied: `N` trees share the
    /// simulations one tree would have run.
    #[test]
    fn the_node_budget_is_split_across_determinizations() {
        let state = engine::new_game(31);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        for n in [1usize, 2, 4, 8] {
            let mut agent = MctsAgent::with_config(
                5,
                Config {
                    root_determinizations: n,
                    ..Config::default()
                },
            );
            let a = agent.choose(&obs, &legal, Budget::Nodes(400));
            assert!(legal.contains(&a));
            assert_eq!(
                agent.total_simulations(),
                400,
                "N={n} did not spend exactly the budget"
            );
        }
    }

    #[test]
    fn ensembling_still_returns_a_legal_move_at_a_time_budget() {
        let state = engine::new_game(17);
        let obs = state.observation();
        let legal = engine::legal_actions(&state);
        let mut agent = MctsAgent::with_config(
            2,
            Config {
                root_determinizations: 4,
                ..Config::default()
            },
        );
        let a = agent.choose(&obs, &legal, Budget::TimeMs(20));
        assert!(legal.contains(&a));
        assert!(agent.total_simulations() > 0);
        assert!(agent.last_tree_size() > 0);
    }

    /// The headline robustness test: full seeded games against the random
    /// agent, from both seats, always reaching a `GameResult` without a
    /// panic, a hang, or an illegal action.
    #[test]
    fn plays_twenty_full_seeded_games_against_random_without_incident() {
        let mut sims = 0u64;
        for seed in 0..20u64 {
            let seat = if seed % 2 == 0 {
                Player::One
            } else {
                Player::Two
            };
            let (result, s) = play(seed, seat, CI_BUDGET);
            sims += s;
            println!("seed {seed} (mcts as {seat}): {result:?}");
        }
        assert!(sims > 0, "the agent never searched");
    }

    /// Even at a CI-sized budget the search should already be clearly better
    /// than uniform-random play. This is a loose smoke test, not the real
    /// strength measurement (see `examples/vs_random.rs`); it exists so that
    /// a sign error in backpropagation or a perspective flip cannot land
    /// silently.
    #[test]
    fn beats_random_at_a_small_budget() {
        let mut wins = 0u32;
        let games = 12u64;
        for seed in 0..games {
            let seat = if seed % 2 == 0 {
                Player::One
            } else {
                Player::Two
            };
            let (result, _) = play(100 + seed, seat, CI_BUDGET);
            if result.winner() == Some(seat) {
                wins += 1;
            }
        }
        assert!(
            wins * 2 > games as u32,
            "won only {wins}/{games} against random at {CI_BUDGET:?}; \
             suspect backpropagation sign, UCB1 perspective, or chance handling"
        );
    }
}
