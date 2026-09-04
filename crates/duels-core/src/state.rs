//! Full game state (server-side / simulation-side only).
//!
//! `GameState` holds everything needed to advance the game deterministically,
//! including information that is not yet public by the rules of the game
//! itself — e.g. the face-down order of cards still in the current age's
//! deck, and which age-III guild cards were set aside during setup and never
//! entered the game.
//!
//! 7 Wonders Duel has no *player-private* hidden information: both players
//! always see identical public state. The only thing that is ever unknown
//! is future randomness (deck order, discard order). That distinction still
//! must never leak to an AI agent, so it is modeled as a separate type
//! ([`crate::observation::Observation`]) rather than left to convention —
//! `Agent::choose` (in `duels-agents-api`) is only ever given a reference to
//! an `Observation`, never a `GameState`.
//!
//! M0 note: this is an intentionally minimal placeholder so the crate
//! compiles and the workspace shape is in place. The real fields and
//! invariants (age-card board layout per age, deck/discard order, wonder
//! selection & construction, coins, science symbols owned, military track
//! position, remaining progress/military tokens, current age, whose turn it
//! is, and RNG state for shuffling) land in M1 together with the engine
//! that mutates them.

use serde::{Deserialize, Serialize};

/// Placeholder for the full, authoritative game state.
///
/// TODO(M1): replace with the real state shape described above.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GameState {
    /// Placeholder turn counter so the struct is non-trivial; will be
    /// replaced by real turn/age/player-to-move tracking in M1.
    pub turn: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_state_default_is_turn_zero() {
        assert_eq!(GameState::default().turn, 0);
    }
}
