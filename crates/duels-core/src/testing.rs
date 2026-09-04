//! Test helpers for constructing specific game states.
//!
//! Rules engines are hard to test end-to-end: reaching an interesting
//! position by playing a real game takes dozens of moves and depends on the
//! shuffle. [`StateBuilder`] lets a test say exactly what it means — "Player
//! One owns these five cards, the pawn is at +4, and these three slots are
//! open" — and then assert an exact cost, legal-move set, or score.
//!
//! This module is compiled into the library (not gated behind `cfg(test)`) so
//! that integration tests, the arena, and future agent crates can use it too.
//! It is a *testing* API: it can build states that a real game would never
//! reach, and it performs no rules validation.

use crate::data::{CardId, TokenId, WonderId};
use crate::layout;
use crate::state::{GameState, Pending, Phase};
use crate::Player;

/// Builds a [`GameState`] from an explicit description.
///
/// The default is a mid-Age-III position with an empty structure, both
/// players on zero coins (rather than the game's starting seven, so that
/// coin arithmetic in tests is unambiguous), and the conflict pawn centred.
///
/// # Examples
///
/// ```
/// use duels_core::testing::StateBuilder;
/// use duels_core::{scoring, Player};
///
/// let state = StateBuilder::new()
///     .built(Player::One, &["palace"]) // 7 victory points
///     .coins(Player::One, 6)           // 2 more
///     .build();
/// assert_eq!(scoring::breakdown(&state, Player::One).total, 9);
/// ```
#[derive(Debug, Clone)]
pub struct StateBuilder {
    state: GameState,
}

impl Default for StateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn card(slug: &str) -> CardId {
    CardId::from_slug(slug).unwrap_or_else(|| panic!("unknown card slug {slug:?}"))
}

fn wonder(slug: &str) -> WonderId {
    WonderId::from_slug(slug).unwrap_or_else(|| panic!("unknown wonder slug {slug:?}"))
}

fn token(slug: &str) -> TokenId {
    TokenId::from_slug(slug).unwrap_or_else(|| panic!("unknown token slug {slug:?}"))
}

impl StateBuilder {
    /// An empty Age III position with both players at zero coins.
    pub fn new() -> Self {
        let mut state = GameState::empty();
        state.set_age(3);
        state.set_phase(Phase::Turn);
        state.set_board(0, 0);
        *state.player_mut(Player::One).coins_mut() = 0;
        *state.player_mut(Player::Two).coins_mut() = 0;
        Self { state }
    }

    /// Set a player's treasury.
    pub fn coins(mut self, p: Player, coins: u16) -> Self {
        *self.state.player_mut(p).coins_mut() = coins;
        self
    }

    /// Add constructed cards to a player's city.
    pub fn built(mut self, p: Player, slugs: &[&str]) -> Self {
        for s in slugs {
            self.state.player_mut(p).add_built_card(card(s));
        }
        self
    }

    /// Give a player drafted (but not constructed) wonders.
    pub fn wonders(mut self, p: Player, slugs: &[&str]) -> Self {
        for s in slugs {
            self.state.player_mut(p).draft_wonder(wonder(s));
        }
        self
    }

    /// Give a player constructed wonders. Implies they were drafted.
    pub fn wonders_built(mut self, p: Player, slugs: &[&str]) -> Self {
        for s in slugs {
            let w = wonder(s);
            self.state.player_mut(p).draft_wonder(w);
            self.state.player_mut(p).mark_wonder_built(w);
        }
        self
    }

    /// Give a player progress tokens.
    pub fn tokens(mut self, p: Player, slugs: &[&str]) -> Self {
        for s in slugs {
            self.state.player_mut(p).add_token(token(s));
        }
        self
    }

    /// Record that a player has already been awarded the token for pairing
    /// `symbol`, so re-gathering the pair will not award another.
    pub fn pair_already_awarded(mut self, p: Player, symbol: crate::data::Science) -> Self {
        self.state.player_mut(p).mark_pair_awarded(symbol);
        self
    }

    /// Put progress tokens on the board (available to a science pair).
    pub fn board_tokens(mut self, slugs: &[&str]) -> Self {
        let mut mask = 0u16;
        for s in slugs {
            mask |= 1u16 << token(s).index();
        }
        let aside = self.state.set_aside_tokens_mask();
        self.state.set_tokens(mask, aside);
        self
    }

    /// Put progress tokens in the out-of-play pile (available to The Great
    /// Library).
    pub fn set_aside_tokens(mut self, slugs: &[&str]) -> Self {
        let mut mask = 0u16;
        for s in slugs {
            mask |= 1u16 << token(s).index();
        }
        let board = self.state.board_tokens_mask();
        self.state.set_tokens(board, mask);
        self
    }

    /// Put cards in the shared discard pile.
    pub fn discard(mut self, slugs: &[&str]) -> Self {
        for s in slugs {
            self.state.add_to_discard(card(s));
        }
        self
    }

    /// Set the conflict pawn, positive favouring [`Player::One`].
    pub fn conflict(mut self, conflict: i8) -> Self {
        self.state.set_conflict(conflict);
        self
    }

    /// Mark a loot token as already collected.
    pub fn loot_taken(mut self, pusher: Player, index: usize) -> Self {
        self.state.take_loot(pusher, index);
        self
    }

    /// Set the current age (this does *not* deal a structure).
    pub fn age(mut self, age: u8) -> Self {
        self.state.set_age(age);
        self
    }

    /// Set the player to move.
    pub fn current(mut self, p: Player) -> Self {
        self.state.set_current(p);
        self
    }

    /// Set the phase.
    pub fn phase(mut self, phase: Phase) -> Self {
        self.state.set_phase(phase);
        self
    }

    /// Set an outstanding effect choice.
    pub fn pending(mut self, pending: Pending) -> Self {
        self.state.set_pending(Some(pending));
        self
    }

    /// Give the current player a banked extra turn.
    pub fn extra_turn(mut self, v: bool) -> Self {
        self.state.set_extra_turn(v);
        self
    }

    /// Record who took the last card of the age.
    pub fn last_card_taker(mut self, p: Player) -> Self {
        self.state.set_last_card_taker(p);
        self
    }

    /// Place specific cards in specific slots of the current age's structure,
    /// face up, and mark every other slot empty.
    ///
    /// Whether a placed slot is *accessible* still follows the printed
    /// geometry: a slot is accessible when none of the slots covering it is
    /// occupied. Placing only bottom-row slots therefore makes them all
    /// immediately playable.
    pub fn open_slots(mut self, slots: &[(u8, &str)]) -> Self {
        let age = self.state.age();
        let mut deck = *self.state.age_deck(age);
        let mut occupied = 0u32;
        for &(slot, slug) in slots {
            assert!(
                (slot as usize) < layout::SLOTS,
                "slot {slot} out of range 0..{}",
                layout::SLOTS
            );
            deck[slot as usize] = card(slug);
            occupied |= 1u32 << slot;
        }
        self.state.set_age_deck(age, deck);
        self.state.set_board(occupied, occupied);
        self
    }

    /// Deal an explicit 20-card deck into the current age's structure, using
    /// the real face-up/face-down pattern for that age.
    ///
    /// # Panics
    ///
    /// Panics if `slugs` is not exactly 20 cards.
    pub fn deal(mut self, slugs: &[&str]) -> Self {
        assert_eq!(
            slugs.len(),
            layout::SLOTS,
            "an age structure holds 20 cards"
        );
        let age = self.state.age();
        let mut deck = *self.state.age_deck(age);
        for (i, s) in slugs.iter().enumerate() {
            deck[i] = card(s);
        }
        self.state.set_age_deck(age, deck);
        let l = layout::layout(age);
        self.state.set_board(layout::ALL_SLOTS, l.face_up);
        self
    }

    /// Finish and return the state.
    pub fn build(self) -> GameState {
        self.state
    }
}

/// Set the player to move on an existing state.
///
/// A testing-only mutation: real play advances the turn through
/// [`crate::engine::apply`]. Handy for asserting what a position offers the
/// *other* player without replaying the whole turn.
pub fn set_current_player(state: &mut GameState, player: Player) {
    state.set_current(player);
}

/// Swap the cards behind two face-down slots, changing hidden information
/// without changing anything public.
///
/// Returns `false` if there were fewer than two face-down slots to swap.
/// Used to assert that [`crate::Observation`] does not depend on hidden
/// state: the observation before and after must be equal. This mutates but
/// never reveals, so it is safe to expose outside the crate.
pub fn swap_two_hidden_cards(state: &mut GameState) -> bool {
    let hidden: Vec<u8> =
        crate::state::iter_slots(state.occupied_slots() & !state.revealed_slots()).collect();
    if hidden.len() < 2 {
        return false;
    }
    state.swap_slot_cards(hidden[0], hidden[hidden.len() - 1]);
    true
}

/// Move a card that was returned to the box during setup into a face-down
/// slot, swapping out whatever was there.
///
/// Like [`swap_two_hidden_cards`] this changes only hidden information: which
/// three cards of the age never entered play is not public. Returns `false`
/// if there is no face-down slot or no boxed card of the current age with the
/// same colour class (a guild card can only trade places with a guild card,
/// because the number of guilds in the Age III structure is public).
pub fn swap_a_boxed_card_into_play(state: &mut GameState) -> bool {
    let s = crate::data::statics();
    let age_mask = s.age_masks[(state.age().max(1) - 1) as usize];
    let hidden: Vec<u8> =
        crate::state::iter_slots(state.occupied_slots() & !state.revealed_slots()).collect();
    let Some(&slot) = hidden.first() else {
        return false;
    };
    let displaced = state.slot_card_hidden(slot);
    let want_guild = displaced.def().is_guild();
    let candidates = state.out_of_game_mask() & age_mask;
    let Some(replacement) =
        crate::state::iter_mask_u128(candidates).find(|c| c.def().is_guild() == want_guild)
    else {
        return false;
    };
    let mut out = state.out_of_game_mask();
    out &= !(1u128 << replacement.index());
    out |= 1u128 << displaced.index();
    state.set_out_of_game(out);
    let mut deck = *state.age_deck(state.age());
    deck[slot as usize] = replacement;
    state.set_age_deck(state.age(), deck);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine;

    #[test]
    fn open_slots_controls_accessibility() {
        // Age I slots 14..20 are the base row.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "lumber-yard"), (15, "quarry")])
            .build();
        assert_eq!(st.accessible_slots(), (1 << 14) | (1 << 15));
        assert_eq!(st.face_up_card(14).unwrap().slug(), "lumber-yard");

        // A slot whose coverers are occupied is not accessible.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(9, "lumber-yard"), (14, "quarry")])
            .build();
        // Slot 9 = (5, 2), covered by slots 14 = (6, 1) and 15 = (6, 3).
        assert_eq!(st.accessible_slots(), 1 << 14);
    }

    #[test]
    fn default_state_has_no_legal_actions_because_the_structure_is_empty() {
        let st = StateBuilder::new().build();
        assert!(engine::legal_actions(&st).is_empty());
    }
}
