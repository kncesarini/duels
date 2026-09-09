//! `duels-arena`: the tournament runner and statistical comparison framework
//! for `Agent` implementations.
//!
//! - [`agent_registry`] looks up a boxed `Agent` by bare name (`"phased"`,
//!   `"mcts-uct"`, ...; add one match arm per new agent crate as it lands).
//!   Being registered there *is* what puts an agent on the roster — the
//!   retired ones are absent, deliberately.
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
//!   set of game results — pairwise ([`elo::fit_elo`]) for one head-to-head
//!   match, or jointly over a whole round robin ([`elo::fit_joint_elo`]) for a
//!   leaderboard.
//! - [`leaderboard`] turns a directory of per-pairing results files into the
//!   `arena/leaderboard.json` / `arena/leaderboard.md` pair the nightly
//!   round-robin workflow commits, and holds the ladder, the rating anchor,
//!   and the designated champion the `ai-candidate` check measures against.
//! - [`sprt`] runs a Sequential Probability Ratio Test, in the style of
//!   chess-engine testing frameworks, over accumulated win/loss/draw counts.
//! - [`experiment`] runs this project's whole documented measurement protocol
//!   — candidate vs control, over several disjoint seed ranges and several
//!   budgets, each cell a paired-seed match, Elo and SPRT per cell *and*
//!   pooled across seed ranges within a budget — as one command producing one
//!   machine-readable verdict. It is orchestration only: every game it plays
//!   goes through [`match_runner`] and every statistic through [`elo`] /
//!   [`sprt`] / [`mechanism`].
//! - [`mechanism`] holds the *other* verdict an experiment reports: bounds,
//!   pre-registered on the command line, on **how** the candidate wins
//!   (victory kinds and race exposure, per side) relative to the control —
//!   because "stronger on aggregate Elo, but by the wrong mechanism" has
//!   already happened here and a pure Elo SPRT cannot say it. Deliberately a
//!   separate verdict from the Elo one, and deliberately three-valued, so a
//!   rare victory kind reads as "not enough evidence" rather than as a pass.
//! - [`results_io`] serializes a match's [`match_runner::GameRecord`]s, plus
//!   the derived tally/victory-breakdown/race-exposure summary, to a JSON
//!   results file.
//! - [`age_start_policy`] wraps any `Agent`, forcing its
//!   `Phase::ChooseFirstPlayer` decisions to a fixed policy while leaving
//!   every other decision untouched, so "does it matter who chooses to go
//!   first at an age boundary" becomes an ordinary matchup (see
//!   `examples/age_start_lab.rs`) instead of a change to any agent's own
//!   evaluation code.
//!
//! Two examples go beyond what the CLI reports:
//!
//! - `examples/ab_lab.rs`, the `duels-agent-alphabeta` tuning harness, which
//!   also gives the two sides *different* budgets (`--budget-a`/`--budget-b`);
//! - `examples/ensemble_lab.rs`, the root-determinization sweep behind both
//!   search agents' ensembling docs, which reports each side's wall clock per
//!   game and, with `--cost`, how much search a decision actually got.
//! - `examples/age_start_lab.rs`, the age-start-policy measurement harness
//!   described above.
//! - `examples/value_corpus.rs`, which is not a measurement harness at all:
//!   it generates a **training corpus** from `mcts-eval` self-play, recording
//!   per decision the win probability the search itself backed up at its root,
//!   keyed by `(seed, actions)` so every position replays from the engine
//!   (rules-spec R-108) instead of being serialized. `--verify` replays a
//!   corpus and reports the labels' calibration against the games that
//!   actually happened. Output goes to the gitignored `arena/corpus/`; read
//!   that example's module docs before fitting anything against it, in
//!   particular the section on what the recorded value already contains.
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

pub mod age_start_policy;
pub mod agent_registry;
pub mod agent_spec;
pub mod elo;
pub mod experiment;
pub mod leaderboard;
pub mod match_runner;
pub mod mechanism;
pub mod results_io;
pub mod sprt;
