//! A once-per-position digest of everything the race reads need to know about
//! *where the cards are*.
//!
//! Each read in this crate asks the same structural questions — which cards
//! are face up in the structure, which are already in a city, which of the
//! current age is still unaccounted for — and [`Board::of`] answers them once,
//! as bitmasks, so a [`crate::stance`] call does not repeat the work five
//! times.
//!
//! # Public information only
//!
//! Everything here comes from [`duels_core::GameState`]'s *public* accessors.
//! In particular [`Board::unknown_pool`] is the same set as
//! [`duels_core::engine::hidden_info`]'s pool: every card of the current age
//! whose whereabouts are not public, which is deliberately larger than the
//! number of face-down slots because three cards of the age were returned to
//! the box unseen and remain candidates. Nothing in this module can tell one
//! determinization of a position from another; that is asserted by
//! `tests/determinization_invariance.rs`.

use duels_core::data::{self, CardId, NUM_CARDS};
use duels_core::layout::{self, SLOTS};
use duels_core::state::{Phase, GUILDS_IN_PLAY};
use duels_core::{GameState, Player};

use crate::masks::{iter_cards, masks};

/// Where every card publicly is, for the position as it stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Board {
    /// The current age, 1 to 3.
    pub age: u8,
    /// Slots that still hold a card.
    pub occupied: u32,
    /// Slots whose card is face up.
    pub revealed: u32,
    /// Slots whose card may be taken right now.
    pub accessible: u32,
    /// Slots that still hold a face-down card.
    pub hidden_slots: u32,
    /// The face-up card in each slot, indexed by slot.
    pub slot_card: [Option<CardId>; SLOTS],
    /// Cards face up in the structure and not yet taken.
    pub face_up: u128,
    /// Cards face up *and* currently accessible.
    pub accessible_cards: u128,
    /// Cards in either player's city.
    pub in_city: u128,
    /// Cards in the shared discard pile.
    pub discard: u128,
    /// Cards spent under a wonder.
    pub fodder: u128,
    /// Cards of the current age whose whereabouts are not public: the
    /// candidates for every face-down slot, plus the three that were boxed
    /// unseen.
    pub unknown_pool: u128,
    /// The guild cards within [`Board::unknown_pool`].
    pub unknown_guilds: u128,
    /// The non-guild cards within [`Board::unknown_pool`].
    pub unknown_plain: u128,
    /// How many face-down slots hold a guild card. Public knowledge: exactly
    /// [`GUILDS_IN_PLAY`] guilds are dealt into Age III.
    pub hidden_guild_count: u8,
    /// The first age whose deck has not been dealt into a structure yet, so
    /// `first_undealt_age..=3` are the ages whose contents are still a
    /// statistical question rather than a board position.
    ///
    /// Normally `age + 1`, but during the wonder draft the Age I structure has
    /// not been laid out yet (the physical setup order), so the current age is
    /// itself still undealt.
    pub first_undealt_age: u8,
}

impl Board {
    /// Digest `state`.
    pub fn of(state: &GameState) -> Board {
        let s = data::statics();
        let m = masks();
        let age = state.age().max(1);
        let age_mask = s.age_masks[(age - 1) as usize];

        let occupied = state.occupied_slots();
        let revealed = state.revealed_slots();
        let accessible = layout::accessible(age, occupied);

        let slot_card: [Option<CardId>; SLOTS] =
            std::array::from_fn(|i| state.face_up_card(i as u8));
        let mut face_up = 0u128;
        let mut accessible_cards = 0u128;
        for (i, entry) in slot_card.iter().enumerate() {
            if let Some(card) = entry {
                let bit = 1u128 << card.index();
                face_up |= bit;
                if accessible & (1u32 << i) != 0 {
                    accessible_cards |= bit;
                }
            }
        }

        let in_city =
            state.player(Player::One).built_mask() | state.player(Player::Two).built_mask();
        let discard = state.discard_mask();
        let fodder = state.wonder_fodder_mask();

        let seen = (in_city | discard | fodder | face_up) & age_mask;
        let unknown_pool = age_mask & !seen & ((1u128 << NUM_CARDS) - 1);
        let unknown_guilds = unknown_pool & m.guild_mask();
        let unknown_plain = unknown_pool & !m.guild_mask();

        // Guilds only ever enter the Age III structure, and exactly
        // `GUILDS_IN_PLAY` of them do, so whatever has not been seen must be
        // behind a face-down slot.
        let hidden_guild_count = if age == 3 {
            let seen_guilds = (seen & m.guild_mask()).count_ones();
            u8::try_from(GUILDS_IN_PLAY)
                .unwrap_or(0)
                .saturating_sub(u8::try_from(seen_guilds).unwrap_or(u8::MAX))
        } else {
            0
        };

        Board {
            age,
            occupied,
            revealed,
            accessible,
            hidden_slots: occupied & !revealed,
            slot_card,
            face_up,
            accessible_cards,
            in_city,
            discard,
            fodder,
            unknown_pool,
            unknown_guilds,
            unknown_plain,
            hidden_guild_count,
            first_undealt_age: if state.phase() == Phase::WonderDraft {
                age
            } else {
                age + 1
            },
        }
    }

    /// The ages whose decks have not been dealt yet.
    #[inline]
    pub fn undealt_ages(&self) -> std::ops::RangeInclusive<u8> {
        self.first_undealt_age..=3
    }

    /// How many ages have not been dealt yet.
    #[inline]
    pub fn undealt_age_count(&self) -> u8 {
        u8::try_from(self.undealt_ages().count()).unwrap_or(0)
    }

    /// How many cards are still in the structure.
    #[inline]
    pub fn cards_left(&self) -> u8 {
        u8::try_from(self.occupied.count_ones()).unwrap_or(u8::MAX)
    }

    /// How many face-down slots there are.
    #[inline]
    pub fn hidden_slot_count(&self) -> u8 {
        u8::try_from(self.hidden_slots.count_ones()).unwrap_or(u8::MAX)
    }

    /// How many face-down slots hold a non-guild card.
    #[inline]
    pub fn hidden_plain_count(&self) -> u8 {
        self.hidden_slot_count()
            .saturating_sub(self.hidden_guild_count)
    }

    /// Slot indices that are accessible right now.
    pub fn accessible_slots(&self) -> impl Iterator<Item = u8> {
        iter_slots(self.accessible)
    }

    /// The expected value of `per_card`, summed over the cards still hidden in
    /// the current age's structure.
    ///
    /// The face-down slots are filled by drawing `hidden_guild_count` cards
    /// from [`Board::unknown_guilds`] and the rest from
    /// [`Board::unknown_plain`], uniformly — exactly the sampling
    /// [`duels_core::Observation::sample_state`] performs — so the expectation
    /// is each pool's total scaled by `slots / pool size`.
    pub fn expected_hidden(&self, per_card: impl Fn(CardId) -> f64) -> f64 {
        let scaled = |pool: u128, slots: u8| -> f64 {
            let n = pool.count_ones();
            if n == 0 || slots == 0 {
                return 0.0;
            }
            let total: f64 = iter_cards(pool).map(&per_card).sum();
            total * f64::from(slots) / f64::from(n)
        };
        scaled(self.unknown_guilds, self.hidden_guild_count)
            + scaled(self.unknown_plain, self.hidden_plain_count())
    }

    /// What taking the card in `slot` would open up: the cards that become
    /// accessible and are already face up (so their identity is public), and a
    /// count of the face-down slots that would be turned over.
    ///
    /// This mirrors the engine's own reveal rule: taking a card uncovers
    /// exactly the slots it covers that nothing else still covers, and turns
    /// face up any of those that were face down. Note that a slot can be face
    /// up *and* covered (the face-up rows are dealt face up regardless of the
    /// geometry), which is why the known half is worth reporting separately
    /// from the unknown half.
    pub fn newly_open_after(&self, slot: u8) -> (u128, u8) {
        let l = layout::layout(self.age);
        let occ = self.occupied & !(1u32 << slot);
        let mut known = 0u128;
        let mut face_down = 0u8;
        let mut rest = l.covers[slot as usize] & occ;
        while rest != 0 {
            let i = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            if l.covered_by[i] & occ != 0 {
                continue;
            }
            match self.slot_card[i] {
                Some(card) => known |= 1u128 << card.index(),
                None => face_down += 1,
            }
        }
        (known, face_down)
    }
}

/// Iterate the set bits of a slot bitmask as slot indices.
#[inline]
pub fn iter_slots(mut mask: u32) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if mask == 0 {
            None
        } else {
            let i = mask.trailing_zeros() as u8;
            mask &= mask - 1;
            Some(i)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn advanced(seed: u64, steps: usize) -> GameState {
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x55);
        for _ in 0..steps {
            let actions = engine::legal_actions(&st);
            if actions.is_empty() {
                break;
            }
            let a = actions[(st.turn() as usize * 7) % actions.len()];
            engine::apply(&mut st, a, &mut rng).unwrap();
        }
        st
    }

    #[test]
    fn the_unknown_pool_matches_the_engines_own_hidden_info() {
        for seed in 0..8u64 {
            for steps in [10usize, 25, 40, 55] {
                let st = advanced(seed, steps);
                if st.is_over() {
                    continue;
                }
                let b = Board::of(&st);
                let info = engine::hidden_info(&st);
                let mut want = 0u128;
                for c in info.pool() {
                    want |= 1u128 << c.index();
                }
                assert_eq!(
                    b.unknown_pool, want,
                    "seed {seed} steps {steps}: pool disagrees with the engine"
                );
                assert_eq!(
                    u32::from(b.hidden_guild_count),
                    info.hidden_guild_count,
                    "seed {seed} steps {steps}: hidden guild count disagrees"
                );
                assert_eq!(b.hidden_slots, info.hidden_slots);
            }
        }
    }

    #[test]
    fn newly_open_after_agrees_with_actually_taking_the_card() {
        let mut rng = StdRng::seed_from_u64(9);
        for seed in 0..6u64 {
            for steps in [8usize, 20, 35] {
                let st = advanced(seed, steps);
                if st.is_over() || st.phase() != duels_core::state::Phase::Turn {
                    continue;
                }
                let b = Board::of(&st);
                if b.cards_left() <= 1 {
                    continue;
                }
                for slot in b.accessible_slots() {
                    let (known, face_down) = b.newly_open_after(slot);
                    let mut after = st;
                    if engine::apply_quiet(
                        &mut after,
                        duels_core::Action::Discard { slot },
                        &mut rng,
                    )
                    .is_err()
                    {
                        continue;
                    }
                    if after.is_over() || after.age() != st.age() {
                        continue;
                    }
                    let b2 = Board::of(&after);
                    let opened = b2.accessible & !b.accessible & !(1u32 << slot);
                    assert_eq!(
                        u32::from(face_down) + known.count_ones(),
                        opened.count_ones(),
                        "seed {seed} steps {steps} slot {slot}: wrong number of newly open slots"
                    );
                    // Every card predicted as "already face up" really is in
                    // the structure and now reachable.
                    assert_eq!(known & b2.accessible_cards, known);
                }
            }
        }
    }
}
