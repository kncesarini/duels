//! `duels-strategy`: pure win-condition reads over a 7 Wonders Duel position.
//!
//! This crate answers one family of questions about a position and nothing
//! else: **which win conditions are still live, for whom, and how urgently?**
//! It plays no moves, holds no state, and never decides anything. Its output
//! is meant to become a *policy / prior* layer inside a search — which moves
//! deserve attention — rather than a static value estimate.
//!
//! # Why a prior and not an evaluation
//!
//! `duels-agent-greedy` already carries explicit military-race terms in its
//! evaluation function, and it still loses to `duels-agent-random` by military
//! supremacy in about one game in ten. A one-ply evaluation cannot see a race
//! that closes three moves out: by the time the pawn is close enough for a
//! positional term to notice, the shields that would have denied it are gone.
//! That is a search problem, not a scoring problem, which is why everything
//! here reports *reachability and tempo* — `need`, `fork`, `turns_to_close`,
//! `fragility` — rather than trying to fold the race into one number.
//!
//! # The four reads and the stance
//!
//! * [`military_read`] — shields still needed, shields reachable now / visible
//!   / expected, how many independent ways there are to close, and a
//!   [`MilitaryStatus`] of `Imminent` / `Live` / `Closed`.
//! * [`science_read`] — distinct symbols, which of the missing ones are still
//!   physically obtainable and by what route, how fragile that is, and a
//!   [`ScienceStatus`] that includes a `Pressure` band for races worth forcing
//!   denial on but not worth winning.
//! * [`vp_read`] — who is ahead if the game stopped now (straight from
//!   [`duels_core::scoring::breakdown`], not reimplemented), plus the swing
//!   still available and a signed [`VpRead::structural_edge`].
//! * [`stance`] and [`action_prior`] — the priority-ordered decision rule over
//!   those reads, and a normalizable per-action weight.
//!
//! # Public information only
//!
//! Every function here reads only what both players can see. That is not a
//! convention but a hard requirement: this layer is meant to run inside a
//! determinized search, where the concrete [`duels_core::GameState`] it is
//! handed was *invented* by [`duels_core::Observation::sample_state`] from the
//! public view. Two samples of the same observation must produce identical
//! reads, or the search would be scoring the sampler's luck instead of the
//! position. `tests/determinization_invariance.rs` asserts exactly that,
//! field by field, over real positions from real games.
//!
//! In practice the reads take a `&GameState` rather than an `&Observation`,
//! because the cost engine ([`duels_core::cost`]) and the scoring functions
//! ([`duels_core::scoring`]) are defined on it — but they touch only
//! `GameState`'s public accessors, which by construction expose no hidden
//! information, plus the public unknown-card *pools*
//! ([`duels_core::engine::hidden_info`], mirrored allocation-free by
//! [`Board`]).
//!
//! # Not wired into anything
//!
//! Deliberately. No agent crate depends on this yet, and `duels-core` never
//! will — the rules engine stays free of strategy. Turning these reads into
//! tree priors and a rollout policy is separate, later work.
//!
//! # Example
//!
//! ```
//! use duels_core::{engine, Player};
//! use duels_strategy::{action_prior, military_read, stance, MilitaryStatus};
//!
//! let state = engine::new_game(7);
//! let me = state.current_player();
//!
//! let mil = military_read(&state, me);
//! assert_eq!(mil.need, 9, "the pawn starts centred");
//! assert_eq!(mil.now, 0, "and nothing is on the table during the draft");
//! // Three whole ages of shields are still to come, so the race is open.
//! assert_eq!(mil.status, MilitaryStatus::Live);
//!
//! let s = stance(&state, me);
//! let legal = engine::legal_actions(&state);
//! let priors: Vec<f64> = legal.iter().map(|&a| action_prior(&state, a, &s)).collect();
//! assert!(priors.iter().all(|&p| p > 0.0));
//! let _ = Player::One;
//! ```

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

pub mod board;
pub mod masks;
pub mod military;
pub mod science;
pub mod stance;
pub mod vp;

pub use board::Board;
pub use masks::{masks, AgeSupply, Masks};
pub use military::{
    military_read, military_read_with, MilitaryBand, MilitaryRead, MilitaryStatus, ShieldSource,
};
pub use science::{
    science_read, science_read_with, PairSetup, ScienceRead, ScienceStatus, SymbolAvailability,
    TokenValueWeights,
};
pub use stance::{
    action_advances, action_closes, action_denies, action_prior, action_priors, action_vp_value,
    stance, stance_with, PriorWeights, Race, Stance, StanceMode,
};
pub use vp::{vp_read, vp_read_with, VpRead, VpWeights};
