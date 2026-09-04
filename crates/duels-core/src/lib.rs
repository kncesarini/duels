//! `duels-core`: the 7 Wonders Duel rules engine.
//!
//! This crate is the single source of truth for game rules. Everything else —
//! the server, the arena, AI agents, the web client — reads state from here
//! and submits [`Action`]s back; nothing else implements rules logic.
//!
//! # The two state types
//!
//! - [`GameState`] is the full, authoritative state, including information
//!   that is not public: the identity of face-down cards in the current age's
//!   structure, which cards were returned to the box unseen, the composition
//!   of the not-yet-dealt age decks, and the wonders not yet offered in the
//!   draft. Only the engine and server-side simulation code hold one.
//! - [`Observation`] is the public view, produced by
//!   [`GameState::observation`], with every not-yet-public value replaced by
//!   the pool of values it could still resolve to.
//!
//! 7 Wonders Duel is stochastic but has **no player-private hidden
//! information** — both players always observe identical public state — so
//! there is exactly one `Observation` per `GameState`, not one per player.
//! The split exists to keep future randomness out of reach of AI agents, and
//! it is enforced by the type system rather than by convention: every field of
//! `GameState` is private, the hidden ones are reachable only via
//! `pub(crate)` accessors, and `duels-agents-api` depends on `Observation`
//! and never on `GameState`.
//!
//! # Determinism and performance
//!
//! The engine does no I/O, reads no clock, and keeps no global mutable state.
//! Randomness enters only through an explicitly passed
//! `rand::rngs::StdRng` (see [`engine::new_game`]); a workspace
//! `clippy.toml` bans `Instant::now`, `SystemTime::now` and `thread_rng` to
//! keep it that way. `GameState` is [`Copy`] and allocation-free so that a
//! search agent can clone it cheaply: built cards are a `u128` bitset over
//! the 73 card ids, the board is a pair of `u32` slot masks.
//!
//! # Rule traceability
//!
//! Every non-trivial rule this crate implements has a numbered `R-xxx` entry
//! in `docs/rules-spec.md` naming the test that covers it.
//!
//! # Example
//!
//! ```
//! use duels_core::{engine, Player};
//! use rand::{rngs::StdRng, SeedableRng};
//!
//! let mut state = engine::new_game(42);
//! let mut rng = StdRng::seed_from_u64(42);
//!
//! // Play a whole game with a "first legal action" policy.
//! while let Some(&action) = engine::legal_actions(&state).first() {
//!     engine::apply(&mut state, action, &mut rng).expect("action came from legal_actions");
//! }
//!
//! let result = state.result().expect("a finished game has a result");
//! println!("{result:?}");
//! ```

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};

pub mod action;
pub mod cost;
pub mod data;
pub mod engine;
pub mod event;
pub mod layout;
pub mod observation;
pub mod scoring;
pub mod state;
pub mod testing;

pub use action::Action;
pub use event::Event;
pub use observation::Observation;
pub use scoring::{Breakdown, GameResult};
pub use state::GameState;

/// One of the two players.
///
/// The conflict pawn convention follows this ordering: a positive
/// [`GameState::conflict`] means the pawn has been pushed towards
/// [`Player::Two`]'s capital, i.e. [`Player::One`] is militarily ahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum Player {
    /// The first seat.
    One,
    /// The second seat.
    Two,
}

impl Player {
    /// Both players, in seat order.
    pub const ALL: [Player; 2] = [Player::One, Player::Two];

    /// Index into a two-element array.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The other player.
    #[inline]
    pub const fn other(self) -> Player {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }

    /// Seat `0` or `1`.
    ///
    /// # Panics
    ///
    /// Panics if `i > 1`.
    #[inline]
    pub fn from_index(i: usize) -> Player {
        match i {
            0 => Player::One,
            1 => Player::Two,
            other => panic!("player index must be 0 or 1, got {other}"),
        }
    }
}

impl std::fmt::Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Player::One => f.write_str("P1"),
            Player::Two => f.write_str("P2"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_index_and_other_are_consistent() {
        for p in Player::ALL {
            assert_eq!(Player::from_index(p.index()), p);
            assert_eq!(p.other().other(), p);
            assert_ne!(p.index(), p.other().index());
        }
    }

    /// The determinism guarantee rests on a workspace clippy config, which is
    /// easy to lose in a refactor. Embed it and assert the bans are still
    /// there.
    #[test]
    fn the_clippy_config_still_bans_nondeterminism() {
        const CLIPPY_TOML: &str = include_str!("../../../clippy.toml");
        for banned in [
            "std::time::Instant::now",
            "std::time::SystemTime::now",
            "rand::thread_rng",
            "rand::random",
        ] {
            assert!(
                CLIPPY_TOML.contains(banned),
                "clippy.toml no longer bans {banned}"
            );
        }
    }

    #[test]
    fn player_round_trips_through_json() {
        let json = serde_json::to_string(&Player::Two).unwrap();
        assert_eq!(json, "\"two\"");
        assert_eq!(serde_json::from_str::<Player>(&json).unwrap(), Player::Two);
    }
}
