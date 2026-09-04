//! Public observation of the game state — the only view an [`Agent`] is
//! allowed to see.
//!
//! [`Agent`]: the `duels-agents-api` crate's `Agent` trait.
//!
//! An `Observation` is derived from a [`crate::state::GameState`] but must
//! never expose information that is not yet public per the rules — most
//! importantly the concrete identity/order of unseen cards. Because 7 Wonders
//! Duel has no player-private information, everything that *is* public is
//! identical for both players; the only thing an `Observation` hides is
//! future randomness, and it does so by replacing "the hidden card at this
//! position" with "the pool of cards it could still resolve to" (e.g. the
//! remaining composition of the current age's deck), never a concrete card
//! identity.
//!
//! This separation is enforced by type, not convention: an `Agent`
//! implementation only ever receives `&Observation` from the engine/arena
//! driving it, so it is structurally incapable of inspecting deck order or
//! other hidden state even by accident.
//!
//! M0 note: placeholder fields only; the full shape (visible board state,
//! each player's built structures/coins/science, remaining token pools,
//! and the "possible cards" pool for hidden positions) lands in M1 next to
//! the engine that produces it from a `GameState`.

use serde::{Deserialize, Serialize};

/// Placeholder for the public observation of a [`crate::state::GameState`].
///
/// TODO(M1): replace with the real observation shape described above.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Placeholder turn counter, mirrors `GameState::turn` for now.
    pub turn: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_default_is_turn_zero() {
        assert_eq!(Observation::default().turn, 0);
    }
}
