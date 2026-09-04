//! The rules engine: legal-move generation, action application, and turn
//! sequencing.
//!
//! This is where [`crate::state::GameState`] transitions will live: given a
//! `GameState` and an [`crate::action::Action`], produce the next
//! `GameState` or reject the action as illegal. It will also own:
//!
//! - `legal_actions(&GameState) -> Vec<Action>`, the canonical source of
//!   truth for what moves are available (used both to drive real play and
//!   to validate an `Agent`'s chosen action against).
//! - Setup: shuffling and laying out each age's card pyramid, selecting the
//!   3 (of 7) guild cards used for Age III plus the extra guild slot,
//!   choosing the wonder draft, and initializing the military track and
//!   token pools.
//! - Deriving a [`crate::observation::Observation`] from a `GameState`.
//!
//! M0 stub only — no logic yet, lands in M1.
