//! Player actions: the complete set of moves a player (human or agent) can
//! submit at a decision point.
//!
//! [`crate::engine::legal_actions`] is the canonical source of truth for which
//! of these are available; an agent picks one of the actions it is handed and
//! never has to validate legality itself.
//!
//! Adding a variant is expected to be backwards compatible for agents that
//! match exhaustively with a wildcard arm; removing or renaming one is a
//! breaking change and must bump `CONTRACT_VERSION` (see
//! `docs/agent-contract.md`).

use serde::{Deserialize, Serialize};

use crate::data::{CardId, TokenId, WonderId};
use crate::Player;

/// Index of a card slot in the current age structure, `0..20`.
///
/// See [`crate::layout`] for the geometry the index refers to.
pub type Slot = u8;

/// A single legal move at the current decision point.
///
/// Most of the game consists of [`Action::Build`], [`Action::Discard`] and
/// [`Action::BuildWonder`]; the remaining variants resolve a *pending choice*
/// created by an effect (a progress token from a science pair, the Great
/// Library's token draw, the Mausoleum's free build, a destroy effect) or the
/// start-of-age first-player decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
#[serde(tag = "type")]
pub enum Action {
    /// Take one of the four currently offered wonders during the initial
    /// draft.
    PickWonder {
        /// The wonder to take.
        wonder: WonderId,
    },
    /// Take the card in `slot` and construct it, paying its cost (or building
    /// it for free via a chain symbol).
    Build {
        /// An accessible slot in the current age structure.
        slot: Slot,
    },
    /// Take the card in `slot` and discard it for coins instead of building
    /// it.
    Discard {
        /// An accessible slot in the current age structure.
        slot: Slot,
    },
    /// Take the card in `slot` and spend it to construct one of your own
    /// unbuilt wonders, paying the wonder's cost.
    BuildWonder {
        /// An accessible slot in the current age structure; the card itself is
        /// consumed and its own effects never apply.
        slot: Slot,
        /// One of the acting player's four drafted, not-yet-built wonders.
        wonder: WonderId,
    },
    /// Take one of the progress tokens still available on the board, after
    /// completing a pair of identical scientific symbols.
    ChooseProgressToken {
        /// One of the tokens currently on the board.
        token: TokenId,
    },
    /// Keep one of the three progress tokens drawn from the out-of-play pile
    /// by The Great Library.
    ChooseGreatLibraryToken {
        /// One of the three drawn tokens.
        token: TokenId,
    },
    /// Construct a card from the discard pile for free (The Mausoleum).
    MausoleumBuild {
        /// One of the cards currently in the discard pile.
        card: CardId,
    },
    /// Discard one of the opponent's constructed buildings (Circus Maximus,
    /// The Statue of Zeus).
    DestroyOpponentCard {
        /// One of the opponent's built cards of the required colour.
        card: CardId,
    },
    /// Decide who takes the first turn of the new age. Only the militarily
    /// weaker player is ever asked.
    ChooseFirstPlayer {
        /// The player who will act first.
        player: Player,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_actions() -> Vec<Action> {
        let card = CardId::from_slug("lumber-yard").unwrap();
        let wonder = WonderId::from_slug("the-pyramids").unwrap();
        let token = TokenId::from_slug("law").unwrap();
        vec![
            Action::PickWonder { wonder },
            Action::Build { slot: 19 },
            Action::Discard { slot: 0 },
            Action::BuildWonder { slot: 3, wonder },
            Action::ChooseProgressToken { token },
            Action::ChooseGreatLibraryToken { token },
            Action::MausoleumBuild { card },
            Action::DestroyOpponentCard { card },
            Action::ChooseFirstPlayer {
                player: Player::Two,
            },
        ]
    }

    #[test]
    fn action_is_debug_clone_eq() {
        for a in sample_actions() {
            let b = a;
            assert_eq!(a, b);
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
    }

    #[test]
    fn all_variants_round_trip_through_json() {
        for v in sample_actions() {
            let json = serde_json::to_string(&v).unwrap();
            let back: Action = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back, "round trip failed for {json}");
        }
    }

    #[test]
    fn ids_appear_as_readable_slugs_in_json() {
        let json = serde_json::to_string(&Action::PickWonder {
            wonder: WonderId::from_slug("the-pyramids").unwrap(),
        })
        .unwrap();
        assert!(json.contains("the-pyramids"), "{json}");
    }
}
