//! `duels-arena`: the tournament runner and statistical comparison framework
//! for `Agent` implementations.
//!
//! - [`agent_registry`] looks up a boxed `Agent` by name (`"random"` today;
//!   add one match arm per new agent crate as it lands).
//! - [`match_runner`] plays one game or a whole paired-seed match between two
//!   named agents, driving `duels-core::engine` exactly as `duels-server`
//!   does — agents only ever see `Observation`s and `legal_actions`.
//! - [`elo`] fits a logistic-Elo rating difference (with a 95% CI) from a
//!   set of game results.
//! - [`sprt`] runs a Sequential Probability Ratio Test, in the style of
//!   chess-engine testing frameworks, over accumulated win/loss/draw counts.
//! - [`results_io`] serializes a match's [`match_runner::GameRecord`]s to a
//!   JSON results file.

pub mod agent_registry;
pub mod elo;
pub mod match_runner;
pub mod results_io;
pub mod sprt;
