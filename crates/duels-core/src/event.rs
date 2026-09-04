//! Events emitted while applying an action.
//!
//! Every state change the engine makes is described by an [`Event`], in the
//! order it happened. Events exist so that a UI can animate a turn, a replay
//! can be reconstructed from `(seed, actions)`, and tests can assert on
//! effect *ordering* rather than only on the resulting state.
//!
//! Events are purely descriptive: replaying them does not reproduce the state
//! (replay `(seed, actions)` for that). They never contain information that is
//! hidden from a player at the time they are emitted — a card is only named in
//! a [`Event::SlotRevealed`] once it is actually face up.

use serde::{Deserialize, Serialize};

use crate::data::{CardId, CardType, Science, TokenId, WonderId};
use crate::scoring::GameResult;
use crate::Player;

/// Why a player's coin total changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoinReason {
    /// Paid the printed coin cost of a card or wonder to the bank.
    ConstructionCost,
    /// Paid the bank for resources not produced by the player's own city.
    Trade,
    /// Received the opponent's trade payment (Economy).
    EconomyToken,
    /// Discarded a card for coins.
    DiscardedCard,
    /// A one-off "take N coins" effect on a card, wonder or token.
    CardEffect,
    /// A yellow Age III card's "coins per building you own" effect.
    PerOwnBuilding,
    /// A guild's immediate "coins per building, whoever has more" effect.
    GuildMajority,
    /// The Urbanism token's chain-build bonus.
    UrbanismChainBonus,
    /// A wonder's "opponent loses N coins" effect.
    WonderPenalty,
    /// A military loot token was passed.
    MilitaryLoot,
}

/// One observable step of applying an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum Event {
    /// A wonder was taken during the initial draft.
    WonderPicked {
        /// Who took it.
        player: Player,
        /// Which wonder.
        wonder: WonderId,
    },
    /// The four wonders of a draft group became visible.
    WonderGroupRevealed {
        /// The four wonders now on offer.
        wonders: [WonderId; 4],
    },
    /// A card left the age structure.
    CardTaken {
        /// Who took it.
        player: Player,
        /// The slot it came from.
        slot: u8,
        /// The card.
        card: CardId,
    },
    /// A previously face-down slot became uncovered and was turned face up.
    SlotRevealed {
        /// The slot.
        slot: u8,
        /// The card now visible there.
        card: CardId,
    },
    /// A card was constructed into a player's city.
    CardBuilt {
        /// Who built it.
        player: Player,
        /// The card.
        card: CardId,
        /// Whether it was free via a chain symbol.
        via_chain: bool,
    },
    /// A card was discarded for coins.
    CardDiscarded {
        /// Who discarded it.
        player: Player,
        /// The card.
        card: CardId,
    },
    /// A wonder was constructed.
    WonderBuilt {
        /// Who built it.
        player: Player,
        /// The wonder.
        wonder: WonderId,
        /// The card spent to build it.
        card: CardId,
        /// How many wonders have now been built in total by both players.
        total_built: u8,
    },
    /// A player gained coins.
    CoinsGained {
        /// Who gained them.
        player: Player,
        /// How many.
        amount: u16,
        /// Why.
        reason: CoinReason,
    },
    /// A player lost coins. Coin totals floor at zero, so `amount` is what
    /// was actually paid, which may be less than the nominal penalty.
    CoinsLost {
        /// Who lost them.
        player: Player,
        /// How many were actually deducted.
        amount: u16,
        /// Why.
        reason: CoinReason,
    },
    /// The conflict pawn moved.
    ConflictMoved {
        /// The player whose shields pushed it.
        player: Player,
        /// Shields applied, including any Strategy bonus.
        shields: u8,
        /// Pawn position before, positive meaning Player One is ahead.
        from: i8,
        /// Pawn position after.
        to: i8,
    },
    /// A military loot token was passed and removed from the board.
    MilitaryLootTriggered {
        /// The player who forfeits coins.
        loser: Player,
        /// Distance from centre of the token that triggered.
        distance: u8,
        /// The nominal coin penalty (see [`Event::CoinsLost`] for the amount
        /// actually paid).
        coins: u8,
    },
    /// A scientific symbol was acquired.
    ScienceGained {
        /// Who acquired it.
        player: Player,
        /// The symbol.
        symbol: Science,
        /// How many distinct symbols the player now has.
        distinct: u8,
    },
    /// A pair of identical scientific symbols completed, entitling the player
    /// to a progress token if any remain on the board.
    SciencePairCompleted {
        /// Who completed it.
        player: Player,
        /// The doubled symbol.
        symbol: Science,
        /// False if the board had no tokens left, so nothing was granted.
        token_available: bool,
    },
    /// A progress token was taken.
    ProgressTokenTaken {
        /// Who took it.
        player: Player,
        /// The token.
        token: TokenId,
        /// True if it came from The Great Library's draw rather than the
        /// board.
        from_great_library: bool,
    },
    /// The Great Library drew three tokens from the out-of-play pile.
    GreatLibraryDraw {
        /// Who drew.
        player: Player,
        /// The three tokens on offer.
        tokens: [TokenId; 3],
    },
    /// A destroy effect is waiting on a choice of target.
    DestroyPending {
        /// Who chooses.
        player: Player,
        /// The colour that may be destroyed.
        card_type: CardType,
    },
    /// An opponent's building was destroyed and put in the discard pile.
    CardDestroyed {
        /// Who destroyed it.
        player: Player,
        /// Whose city lost it.
        victim: Player,
        /// The card.
        card: CardId,
    },
    /// A player gained an extra turn.
    ExtraTurnGranted {
        /// Who gained it.
        player: Player,
    },
    /// A pending extra turn was forfeited because the age ended first.
    ExtraTurnLost {
        /// Who lost it.
        player: Player,
    },
    /// The current age's structure was emptied.
    AgeEnded {
        /// The age that just finished.
        age: u8,
    },
    /// A new age's structure was dealt.
    AgeStarted {
        /// The new age.
        age: u8,
    },
    /// The militarily weaker player chose who begins the new age.
    FirstPlayerChosen {
        /// Who acts first.
        player: Player,
    },
    /// The game ended.
    GameEnded {
        /// The outcome.
        result: GameResult,
    },
}

/// A sink for [`Event`]s that can be turned off entirely.
///
/// Recording events allocates; a search-based agent applying millions of
/// actions per second does not want that, so the engine's internals push
/// through this type and construct the [`Event`] value lazily.
#[derive(Debug, Default)]
pub struct EventLog {
    events: Vec<Event>,
    recording: bool,
}

impl EventLog {
    /// A log that records events.
    pub fn recording() -> Self {
        Self {
            events: Vec::new(),
            recording: true,
        }
    }

    /// A log that discards everything pushed to it without allocating.
    pub fn discarding() -> Self {
        Self {
            events: Vec::new(),
            recording: false,
        }
    }

    /// Push an event. `f` is only called when the log is recording.
    #[inline]
    pub fn push(&mut self, f: impl FnOnce() -> Event) {
        if self.recording {
            self.events.push(f());
        }
    }

    /// The recorded events, in order.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Consume the log and return the recorded events.
    pub fn into_events(self) -> Vec<Event> {
        self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discarding_log_records_nothing() {
        let mut log = EventLog::discarding();
        log.push(|| panic!("must not be constructed"));
        assert!(log.events().is_empty());
    }

    #[test]
    fn recording_log_keeps_order() {
        let mut log = EventLog::recording();
        log.push(|| Event::AgeEnded { age: 1 });
        log.push(|| Event::AgeStarted { age: 2 });
        assert_eq!(
            log.into_events(),
            vec![Event::AgeEnded { age: 1 }, Event::AgeStarted { age: 2 }]
        );
    }

    #[test]
    fn events_round_trip_through_json() {
        let e = Event::CoinsGained {
            player: Player::One,
            amount: 4,
            reason: CoinReason::CardEffect,
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }
}
