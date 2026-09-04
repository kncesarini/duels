//! `duels-arena`: the tournament runner and statistical comparison framework
//! for `Agent` implementations.
//!
//! - [`agent_registry`] looks up a boxed `Agent` by bare name (`"random"`
//!   today; add one match arm per new agent crate as it lands).
//! - [`agent_spec`] generalizes that into a *specification string* — a bare
//!   name, or a name plus `key=value` parameters
//!   (`"mcts-uct:exploration=1.2"`) that build one specific agent crate's own
//!   `Config`/`Weights` type — so a parameter sweep needs no new registry
//!   code.
//! - [`match_runner`] plays one game or a whole paired-seed match between two
//!   named agents, driving `duels-core::engine` exactly as `duels-server`
//!   does — agents only ever see `Observation`s and `legal_actions`. It also
//!   flags each game for "race exposure" (either player coming within one
//!   step of an instant win) and tallies *how* each side's wins were
//!   achieved, not just the win/loss/draw count.
//! - [`elo`] fits a logistic-Elo rating difference (with a 95% CI) from a
//!   set of game results.
//! - [`sprt`] runs a Sequential Probability Ratio Test, in the style of
//!   chess-engine testing frameworks, over accumulated win/loss/draw counts.
//! - [`results_io`] serializes a match's [`match_runner::GameRecord`]s, plus
//!   the derived tally/victory-breakdown/race-exposure summary, to a JSON
//!   results file.
//!
//! Two examples go beyond what the CLI reports:
//!
//! - `examples/ab_lab.rs`, the `duels-agent-alphabeta` tuning harness, which
//!   also gives the two sides *different* budgets (`--budget-a`/`--budget-b`);
//! - `examples/ensemble_lab.rs`, the root-determinization sweep behind both
//!   search agents' ensembling docs, which reports each side's wall clock per
//!   game and, with `--cost`, how much search a decision actually got.
//!
//! # Benchmarking on a quiet machine
//!
//! `Budget::TimeMs` runs are wall-clock based, so the number of simulations
//! (and hence the measured strength) an agent gets through in a fixed budget
//! depends on how much CPU it actually receives — not just on its own code.
//! This was observed directly during the `mcts-uct` rollout-policy
//! investigation (see `duels-agent-mcts-uct`'s `rollout` module docs): the
//! *same* `BIASED`-vs-`UNIFORM` comparison at `n=40` games scored 60% in one
//! run and 45% in another, run back to back on the same machine under
//! different concurrent load. That is a bigger swing than most of the
//! effects this crate is used to measure, so it can silently invalidate a
//! conclusion drawn from a single `TimeMs` run.
//!
//! `Budget::Nodes` runs do not have this problem (a node count is not a
//! wall-clock quantity), so prefer `Nodes` whenever a comparison does not
//! specifically need to hold *time* fixed. When a `TimeMs` comparison is
//! unavoidable:
//!
//! - Run one match at a time. `duels-arena` already parallelizes *within* a
//!   match across seeds (see [`match_runner::play_paired_match`]); running a
//!   second match concurrently contends every game in both for CPU and biases
//!   both towards fewer simulations per decision, in a way that need not
//!   cancel out between them.
//! - Don't run a `TimeMs` match alongside another `cargo build`, `cargo
//!   test`, or anything else CPU-heavy on the same machine.
//! - Treat a small-sample `TimeMs` result (dozens of games) as indicative,
//!   not conclusive, even on an otherwise-quiet machine: game-outcome
//!   variance and load-dependent simulation-count variance stack, and only
//!   the first of those shrinks with more games under a *fixed* budget. The
//!   SPRT ([`sprt`]) and the Elo confidence interval ([`elo`]) both already
//!   report how much evidence a given run actually represents — read those
//!   rather than a bare win percentage.

pub mod agent_registry;
pub mod agent_spec;
pub mod elo;
pub mod match_runner;
pub mod results_io;
pub mod sprt;
