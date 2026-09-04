//! `duels-core`: the 7 Wonders Duel rules engine.
//!
//! This crate is the single source of truth for game rules. It has two
//! halves that must stay strictly separated:
//!
//! - [`state::GameState`]: the full, authoritative state, including
//!   information that is not yet public (deck order, face-down slots).
//!   Only the engine and server-side simulation code ever touch this.
//! - [`observation::Observation`]: the public view of the game, derived
//!   from a `GameState`, with all not-yet-revealed information replaced by
//!   the pool of possibilities it could still resolve to. This is the only
//!   type an `Agent` (see the sibling `duels-agents-api` crate) is allowed
//!   to see.
//!
//! 7 Wonders Duel has no player-private hidden information — both players
//! always observe identical public state — so there is exactly one
//! `Observation` per `GameState`, not one per player. The split exists
//! purely to keep future randomness (deck order, discard-pile order) out of
//! reach of AI agents, enforced by the type system rather than convention:
//! an `Agent` implementation is only ever handed `&Observation`.
//!
//! M0 scope: module stubs and a first-pass [`action::Action`] enum only.
//! The actual rules (legal move generation, effects, scoring) land in M1.

pub mod action;
pub mod data;
pub mod engine;
pub mod observation;
pub mod scoring;
pub mod state;

pub use action::Action;
pub use observation::Observation;
pub use state::GameState;
