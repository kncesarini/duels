//! What everything on the table costs each player, priced once.
//!
//! [`duels_core::cost::card_cost`] is the single most expensive thing this
//! crate calls: it walks the resource cost, prices every missing unit against
//! the opponent's production, and applies whatever discount token the player
//! holds. Both races, the tempo model, the reachability walk and the exposure
//! half of the denial channel all want the same answers — "can this player
//! afford the card in slot 7", "what would a chained sequence of two wonder
//! builds and a card cost" — and the first cut of this crate asked the cost
//! engine those questions four or five times over per position.
//!
//! [`Prices`] asks once. One pass per player over the revealed slots and the
//! drafted-but-unbuilt wonders, and everything downstream is a table lookup.
//! Nothing here is a heuristic; it is the real cost engine's answers, cached.

use duels_core::data::{WonderId, NUM_WONDERS};
use duels_core::layout::SLOTS;
use duels_core::{cost, GameState, Player};

use crate::board::{iter_slots, Board};

/// The coin price of everything one player could pay for, and which of it they
/// can actually afford right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prices {
    /// Coins in the treasury.
    pub coins: u16,
    /// Coin price of the card in each slot, or [`u16::MAX`] where the slot has
    /// not been priced.
    pub slot_cost: [u16; SLOTS],
    /// Which slots have been priced at all.
    ///
    /// Not every revealed slot is: the cost engine is the expensive part of
    /// this crate, and the only slots any read asks about are the accessible
    /// ones and the frontier a chained extra turn or an opponent's next move
    /// could reach ([`Board::one_step_reveals`], extended by
    /// [`Prices::price_also`] when somebody can chain deeper than one).
    pub priced: u32,
    /// Priced slots this player can pay for — whether or not they are
    /// accessible yet, because a chained extra turn can reach past the front
    /// row.
    pub affordable_slots: u32,
    /// Of those, the ones that are accessible right now.
    pub affordable_now: u32,
    /// Coin price of each drafted, unbuilt wonder, or [`u16::MAX`] for a
    /// wonder this player does not own or has already built.
    pub wonder_cost: [u16; NUM_WONDERS],
    /// Bitmask over [`WonderId::index`] of the drafted, unbuilt wonders this
    /// player can pay for. Empty when there is no card left to spend on one,
    /// or no wonder slot left in the game.
    pub affordable_wonders: u16,
    /// Whether a wonder could be constructed at all: some card in the
    /// structure to spend, and fewer than the shared maximum already built.
    pub can_build_wonder: bool,
}

impl Prices {
    /// Price the position for `player`: every drafted, unbuilt wonder, and the
    /// slots one move can reach.
    pub fn of(state: &GameState, player: Player, board: &Board) -> Prices {
        let me = state.player(player);
        let coins = me.coins();
        let mut slot_cost = [u16::MAX; SLOTS];
        let mut affordable_slots = 0u32;
        let priced = (board.accessible | board.one_step_reveals) & board.revealed;
        for slot in iter_slots(priced) {
            let Some(card) = board.slot_card[slot as usize] else {
                continue;
            };
            let c = cost::card_cost(state, player, card).coins;
            slot_cost[slot as usize] = c;
            if c <= coins {
                affordable_slots |= 1u32 << slot;
            }
        }

        let can_build_wonder = board.accessible != 0 && state.wonder_slots_left();
        let mut wonder_cost = [u16::MAX; NUM_WONDERS];
        let mut affordable_wonders = 0u16;
        for wonder in me.wonders() {
            if me.has_built_wonder(wonder) {
                continue;
            }
            let c = cost::wonder_cost(state, player, wonder).coins;
            wonder_cost[wonder.index()] = c;
            if can_build_wonder && c <= coins {
                affordable_wonders |= 1u16 << wonder.index();
            }
        }

        Prices {
            coins,
            slot_cost,
            priced,
            affordable_slots,
            affordable_now: affordable_slots & board.accessible,
            wonder_cost,
            affordable_wonders,
            can_build_wonder,
        }
    }

    /// Price `extra` slots too, for the rare position where somebody can chain
    /// two or more extra turns and so reach past the one-move frontier.
    pub fn price_also(&mut self, state: &GameState, player: Player, board: &Board, extra: u32) {
        for slot in iter_slots(extra & board.revealed & !self.priced) {
            let Some(card) = board.slot_card[slot as usize] else {
                continue;
            };
            let c = cost::card_cost(state, player, card).coins;
            self.slot_cost[slot as usize] = c;
            self.priced |= 1u32 << slot;
            if c <= self.coins {
                self.affordable_slots |= 1u32 << slot;
            }
        }
    }

    /// Whether the card in `slot` is affordable *and* accessible right now.
    #[inline]
    pub fn can_take_slot(&self, slot: u8) -> bool {
        self.affordable_now & (1u32 << slot) != 0
    }

    /// Whether `wonder` is drafted, unbuilt and affordable right now.
    #[inline]
    pub fn can_afford_wonder(&self, wonder: WonderId) -> bool {
        self.affordable_wonders & (1u16 << wonder.index()) != 0
    }

    /// Whether the card in `slot` is affordable after `prefix` coins have
    /// already been committed — a chained sequence of wonder builds ahead of
    /// it.
    ///
    /// Deliberately a plain coin check against the state as it stands rather
    /// than a resource-trade replay: the wonders in the chain may well change
    /// what a later card costs, and modelling that exactly would mean running
    /// the cost engine against hypothetical cities.
    #[inline]
    pub fn can_afford_slot_after(&self, slot: u8, prefix: u16) -> bool {
        match self.slot_cost.get(slot as usize) {
            Some(&c) if c != u16::MAX => prefix.saturating_add(c) <= self.coins,
            _ => false,
        }
    }
}
