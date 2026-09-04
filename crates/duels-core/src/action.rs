//! Player actions: the complete set of moves a player (human or [`Agent`]) can
//! submit on their turn.
//!
//! [`Agent`]: https://docs.rs/duels-agents-api (see `duels-agents-api` crate)
//!
//! M0 implements a first-pass `Action` enum covering the core decision
//! points of the game: playing a card to build it, discarding a card for
//! coins, using a card to build a wonder stage, and choosing a progress
//! token when an effect grants one. Finer-grained variants (e.g. choosing
//! which card to take from the discard pile via the Economy token effect,
//! or resolving a Diplomacy-token skip) will be added in M1 alongside the
//! rules engine logic that needs them.
//!
//! Adding a variant is expected to be backwards compatible for agents that
//! match exhaustively with a wildcard arm; removing or renaming one is a
//! breaking change and must bump `CONTRACT_VERSION` (see
//! `docs/agent-contract.md`).

use serde::{Deserialize, Serialize};

/// Identifier for an age card, matching the `id` field in `data/cards.json`.
pub type CardId = String;

/// Identifier for a wonder, matching the `id` field in `data/wonders.json`.
pub type WonderId = String;

/// Identifier for a progress token, matching the `id` field in
/// `data/tokens.json`.
pub type ProgressTokenId = String;

/// A single legal move a player can make on their turn.
///
/// This is intentionally minimal for M0. The rules engine landing in M1
/// will grow `legal_actions(&GameState) -> Vec<Action>` to enumerate exactly
/// which of these are available at any point, including the wonder-building
/// cost check and chain-symbol free-build rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Action {
    /// Construct the named card from the currently available age-card
    /// structure, paying its cost (or building it for free via a chain
    /// symbol).
    PlayCard { card: CardId },
    /// Discard the named card for coins instead of building it.
    DiscardCardForCoins { card: CardId },
    /// Use the named card to build one stage of the named wonder.
    BuildWonder { wonder: WonderId, card: CardId },
    /// Choose a progress token, e.g. after completing a pair of distinct
    /// science symbols, or from a pool of tokens revealed by a card effect.
    ChooseProgressToken { token: ProgressTokenId },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_is_debug_clone_eq() {
        let a = Action::PlayCard {
            card: "card-001".to_string(),
        };
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    #[test]
    fn action_serializes_and_round_trips() {
        let a = Action::PlayCard {
            card: "card-001".to_string(),
        };
        let json = serde_json::to_string(&a).expect("Action should serialize");
        assert!(json.contains("card-001"));

        let round_tripped: Action = serde_json::from_str(&json).expect("Action should deserialize");
        assert_eq!(a, round_tripped);
    }

    #[test]
    fn all_variants_round_trip_through_json() {
        let variants = vec![
            Action::PlayCard { card: "c1".into() },
            Action::DiscardCardForCoins { card: "c2".into() },
            Action::BuildWonder {
                wonder: "w1".into(),
                card: "c3".into(),
            },
            Action::ChooseProgressToken { token: "t1".into() },
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: Action = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }
}
