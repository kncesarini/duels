//! Public observation of the game state — the only view an agent ever sees.
//!
//! An [`Observation`] is derived from a [`GameState`] by
//! [`GameState::observation`]. It contains *everything* a player is allowed to
//! know and *nothing* they are not:
//!
//! * a face-down slot is [`SlotView::FaceDown`], which carries no card id;
//! * the cards that could be behind those slots appear only as an unordered
//!   pool ([`Observation::unknown_slot_pool`]), together with how many of the
//!   face-down slots hold a guild card — public because exactly three guilds
//!   are dealt into Age III;
//! * the composition of the not-yet-dealt age decks is absent entirely (it is
//!   derivable from the static card list, so there is nothing to carry);
//! * the wonders not yet offered in the draft appear only as a pool.
//!
//! The five progress tokens set aside during setup *are* exposed: they are the
//! complement of the five placed on the board, so both players can deduce
//! them. The only randomness they carry is which three The Great Library will
//! draw, and that is resolved when the wonder is built.
//!
//! Because 7 Wonders Duel has no player-private information, one
//! `Observation` serves both players and any spectator. The correctness
//! property that matters is:
//!
//! > two `GameState`s that differ only in hidden information must produce
//! > equal `Observation`s.
//!
//! That is asserted directly in this module's tests and in
//! `tests/properties.rs`.
//!
//! [`Observation::sample_state`] goes the other way: it invents a `GameState`
//! consistent with the observation by sampling the hidden parts uniformly.
//! That is what a search-based agent needs in order to run a determinized
//! search from a public position.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::data::{self, CardId, Science, TokenId, WonderId, NUM_SCIENCE};
use crate::engine;
use crate::layout::{self, SLOTS};
use crate::scoring::GameResult;
use crate::state::{
    iter_mask_u128, iter_slots, GameState, Pending, Phase, CARDS_REMOVED_PER_AGE, GUILDS_IN_PLAY,
};
use crate::Player;

/// What is publicly known about one slot of the age structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SlotView {
    /// The card has been taken.
    Empty,
    /// The card is face up and everyone can see it.
    FaceUp {
        /// The visible card.
        card: CardId,
    },
    /// The card is face down. Deliberately carries no id: the candidates are
    /// in [`Observation::unknown_slot_pool`].
    FaceDown,
}

impl SlotView {
    /// The visible card, if any.
    #[inline]
    pub fn card(&self) -> Option<CardId> {
        match self {
            SlotView::FaceUp { card } => Some(*card),
            _ => None,
        }
    }
}

/// One player's city, as everyone sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
pub struct PublicPlayer {
    /// Coins in the treasury.
    pub coins: u16,
    /// Constructed cards, ascending by id.
    pub built: Vec<CardId>,
    /// Drafted wonders, ascending by id.
    pub wonders: Vec<WonderId>,
    /// Constructed wonders, ascending by id.
    pub wonders_built: Vec<WonderId>,
    /// Progress tokens owned, ascending by id.
    pub tokens: Vec<TokenId>,
    /// Total shields ever generated.
    pub shields: u8,
    /// Count of each scientific symbol held.
    pub science: [u8; NUM_SCIENCE],
    /// Symbols for which a pair has already been rewarded, so re-completing
    /// the pair after a destroy effect grants nothing.
    pub pairs_awarded: Vec<Science>,
}

/// The complete public view of a game.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
pub struct Observation {
    /// What kind of decision is pending.
    pub phase: Phase,
    /// The current age.
    pub age: u8,
    /// Whose decision it is.
    pub current_player: Player,
    /// How many decisions have been resolved.
    pub turn: u32,
    /// Conflict pawn, positive favouring [`Player::One`].
    pub conflict: i8,
    /// Which loot tokens have been collected, indexed by pushing player then
    /// by loot index.
    pub loot_taken: [[bool; 2]; 2],
    /// Whether the player to move has a banked extra turn.
    pub extra_turn: bool,
    /// The outstanding effect choice.
    pub pending: Option<Pending>,
    /// Who took the last card of the age.
    pub last_card_taker: Player,
    /// Both players, indexed by [`Player::index`].
    pub players: [PublicPlayer; 2],
    /// The age structure.
    pub slots: [SlotView; SLOTS],
    /// The shared discard pile, ascending by id.
    pub discard: Vec<CardId>,
    /// Cards spent under wonders, ascending by id.
    pub wonder_fodder: Vec<CardId>,
    /// Progress tokens on the board, ascending by id.
    pub board_tokens: Vec<TokenId>,
    /// Progress tokens set aside at setup and still available to The Great
    /// Library, ascending by id.
    pub set_aside_tokens: Vec<TokenId>,
    /// Wonders currently on offer in the draft.
    pub offered_wonders: Vec<WonderId>,
    /// Wonders neither drafted nor on offer: during the first half of the
    /// draft this is a genuine unknown pool of eight, four of which will be
    /// offered next.
    pub undrafted_wonder_pool: Vec<WonderId>,
    /// Which of the eight draft picks is next.
    pub draft_step: u8,
    /// Who picks first in the draft.
    pub draft_first: Player,
    /// Every card that could be behind a face-down slot of the current age,
    /// ascending by id. Larger than the number of face-down slots, because
    /// three cards of the age were returned to the box unseen and remain
    /// candidates.
    pub unknown_slot_pool: Vec<CardId>,
    /// How many face-down slots hold a guild card.
    pub hidden_guild_count: u8,
    /// The outcome, once the game is over.
    pub result: Option<GameResult>,
}

impl GameState {
    /// The public view of this state.
    ///
    /// This is the *only* way to get from a `GameState` to something an agent
    /// may see, and it is deliberately lossy.
    pub fn observation(&self) -> Observation {
        let hidden = engine::hidden_info(self);
        let mut unknown_slot_pool = hidden.pool();
        unknown_slot_pool.sort_unstable();

        let offered_wonders = self.offered_wonders();
        let owned_or_offered = self
            .player(Player::One)
            .wonders()
            .chain(self.player(Player::Two).wonders())
            .chain(offered_wonders.iter().copied())
            .fold(0u16, |m, w| m | (1u16 << w.index()));
        let mut undrafted_wonder_pool: Vec<WonderId> = WonderId::all()
            .filter(|w| owned_or_offered & (1u16 << w.index()) == 0)
            .collect();
        undrafted_wonder_pool.sort_unstable();

        let slots: [SlotView; SLOTS] = std::array::from_fn(|i| {
            let bit = 1u32 << i;
            if self.occupied_slots() & bit == 0 {
                SlotView::Empty
            } else if self.revealed_slots() & bit != 0 {
                SlotView::FaceUp {
                    card: self.slot_card_hidden(i as u8),
                }
            } else {
                SlotView::FaceDown
            }
        });

        Observation {
            phase: self.phase(),
            age: self.age(),
            current_player: self.current_player(),
            turn: self.turn(),
            conflict: self.conflict(),
            loot_taken: [
                [
                    !self.loot_available(Player::One, 0),
                    !self.loot_available(Player::One, 1),
                ],
                [
                    !self.loot_available(Player::Two, 0),
                    !self.loot_available(Player::Two, 1),
                ],
            ],
            extra_turn: self.extra_turn(),
            pending: self.pending(),
            last_card_taker: self.last_card_taker(),
            players: [
                public_player(self, Player::One),
                public_player(self, Player::Two),
            ],
            slots,
            discard: self.discard_pile().collect(),
            wonder_fodder: iter_mask_u128(self.wonder_fodder_mask()).collect(),
            board_tokens: self.board_tokens().collect(),
            set_aside_tokens: self.set_aside_tokens().collect(),
            offered_wonders,
            undrafted_wonder_pool,
            draft_step: self.draft_step(),
            draft_first: self.draft_first(),
            unknown_slot_pool,
            hidden_guild_count: hidden.hidden_guild_count as u8,
            result: self.result(),
        }
    }
}

fn public_player(state: &GameState, p: Player) -> PublicPlayer {
    let ps = state.player(p);
    PublicPlayer {
        coins: ps.coins(),
        built: ps.built().collect(),
        wonders: ps.wonders().collect(),
        wonders_built: ps.wonders_built().collect(),
        tokens: ps.tokens().collect(),
        shields: ps.shields(),
        science: ps.science(),
        pairs_awarded: ps.pairs_awarded().collect(),
    }
}

impl Observation {
    /// Invent a [`GameState`] consistent with this observation, sampling the
    /// hidden information uniformly.
    ///
    /// Used by search-based agents: play out a determinized world, repeat with
    /// fresh samples, average. The sample respects every public constraint,
    /// including that exactly three guild cards are in the Age III structure.
    ///
    /// `sample_state(rng).observation() == *self` for every sample, which is
    /// asserted as a property test.
    pub fn sample_state(&self, rng: &mut StdRng) -> GameState {
        let s = data::statics();
        let mut st = GameState::empty();

        st.set_age(self.age.max(1));
        st.set_phase(self.phase);
        st.set_pending(self.pending);
        st.set_extra_turn(self.extra_turn);
        st.set_conflict(self.conflict);
        st.set_last_card_taker(self.last_card_taker);
        st.set_current(self.current_player);
        st.set_turn(self.turn);
        if let Some(r) = self.result {
            st.set_result(r);
            st.set_phase(self.phase);
        }
        for p in Player::ALL {
            for i in 0..2 {
                if self.loot_taken[p.index()][i] {
                    st.take_loot(p, i);
                }
            }
        }

        for p in Player::ALL {
            let src = &self.players[p.index()];
            let dst = st.player_mut(p);
            *dst.coins_mut() = src.coins;
            for &c in &src.built {
                dst.add_built_card(c);
            }
            for &w in &src.wonders {
                dst.draft_wonder(w);
            }
            for &w in &src.wonders_built {
                dst.mark_wonder_built(w);
            }
            for &t in &src.tokens {
                dst.add_token(t);
            }
            dst.add_shields(src.shields);
            for &sym in &src.pairs_awarded {
                dst.mark_pair_awarded(sym);
            }
        }

        for &c in &self.discard {
            st.add_to_discard(c);
        }
        for &c in &self.wonder_fodder {
            st.add_wonder_fodder(c);
        }

        let mask16 = |ids: &[TokenId]| ids.iter().fold(0u16, |m, t| m | (1u16 << t.index()));
        st.set_tokens(mask16(&self.board_tokens), mask16(&self.set_aside_tokens));

        let mut occupied = 0u32;
        let mut revealed = 0u32;
        for (i, view) in self.slots.iter().enumerate() {
            match view {
                SlotView::Empty => {}
                SlotView::FaceUp { .. } => {
                    occupied |= 1u32 << i;
                    revealed |= 1u32 << i;
                }
                SlotView::FaceDown => occupied |= 1u32 << i,
            }
        }
        st.set_board(occupied, revealed);

        // Cards whose whereabouts are already public.
        let taken = self.players[0]
            .built
            .iter()
            .chain(self.players[1].built.iter())
            .chain(self.discard.iter())
            .chain(self.wonder_fodder.iter())
            .fold(0u128, |m, c| m | (1u128 << c.index()));

        let mut dealt = 0u128;
        for age in 1..=3u8 {
            let age_mask = s.age_masks[(age - 1) as usize];
            let deck: [CardId; SLOTS] = if age == self.age {
                current_age_deck(self, rng)
            } else if age < self.age {
                // A finished age: all 20 of its cards are in a city, the
                // discard pile, or under a wonder, so nothing is hidden. The
                // order no longer matters.
                let mut cards: Vec<CardId> = iter_mask_u128(taken & age_mask).collect();
                cards.shuffle(rng);
                pad_deck(cards, age_mask)
            } else {
                sample_future_deck(age, rng)
            };
            for &c in deck.iter() {
                dealt |= 1u128 << c.index();
            }
            st.set_age_deck(age, deck);
        }
        st.set_out_of_game(!dealt & ((1u128 << data::NUM_CARDS) - 1));

        // The wonder pile: the currently offered group has to sit in the
        // group's slots so that `offered_wonders` reproduces, the drafted ones
        // go where they were taken from, and the rest is sampled.
        st.set_draft_first(self.draft_first);
        st.set_current(self.current_player);
        st.set_draft_step(self.draft_step);
        st.set_draft_pile(sample_wonder_pile(self, rng));

        st
    }
}

/// Fill a deck array, padding with arbitrary cards of the age if `cards` is
/// short (which can only happen for a state built by test helpers).
fn pad_deck(mut cards: Vec<CardId>, age_mask: u128) -> [CardId; SLOTS] {
    if cards.len() < SLOTS {
        for c in iter_mask_u128(age_mask) {
            if cards.len() == SLOTS {
                break;
            }
            if !cards.contains(&c) {
                cards.push(c);
            }
        }
    }
    cards.truncate(SLOTS);
    std::array::from_fn(|i| cards[i])
}

/// Sample a layout for the age currently on the table: face-up slots keep
/// their card, face-down slots get a uniformly random consistent assignment,
/// and emptied slots are backfilled with cards already taken from this age.
fn current_age_deck(obs: &Observation, rng: &mut StdRng) -> [CardId; SLOTS] {
    let s = data::statics();
    let age_mask = s.age_masks[(obs.age.max(1) - 1) as usize];

    let mut guilds: Vec<CardId> = obs
        .unknown_slot_pool
        .iter()
        .copied()
        .filter(|c| c.def().is_guild())
        .collect();
    let mut plain: Vec<CardId> = obs
        .unknown_slot_pool
        .iter()
        .copied()
        .filter(|c| !c.def().is_guild())
        .collect();
    guilds.shuffle(rng);
    plain.shuffle(rng);

    let mut hidden_slots: Vec<usize> = (0..SLOTS)
        .filter(|&i| obs.slots[i] == SlotView::FaceDown)
        .collect();
    hidden_slots.shuffle(rng);

    // Anything of this age already taken can go in the emptied slots.
    let taken_here: u128 = obs
        .players
        .iter()
        .flat_map(|p| p.built.iter())
        .chain(obs.discard.iter())
        .chain(obs.wonder_fodder.iter())
        .fold(0u128, |m, c| m | (1u128 << c.index()))
        & age_mask;
    let mut spare: Vec<CardId> = iter_mask_u128(taken_here).collect();
    spare.shuffle(rng);

    let mut assigned: Vec<Option<CardId>> = obs.slots.iter().map(SlotView::card).collect();
    for (n, &slot) in hidden_slots.iter().enumerate() {
        let card = if n < obs.hidden_guild_count as usize {
            guilds.pop()
        } else {
            plain.pop()
        }
        .or_else(|| plain.pop())
        .or_else(|| guilds.pop());
        assigned[slot] = card;
    }

    // Backfill the rest with cards already taken, then with anything of the
    // age that is still unused, so the array stays a permutation.
    let mut used = assigned
        .iter()
        .flatten()
        .fold(0u128, |m, c| m | (1u128 << c.index()));
    let mut filler = |used: &mut u128| -> CardId {
        while let Some(c) = spare.pop() {
            if *used & (1u128 << c.index()) == 0 {
                *used |= 1u128 << c.index();
                return c;
            }
        }
        // Unreachable for any state a real game can reach; every age has at
        // least 20 cards, so `spare` plus the assignment always covers them.
        match iter_mask_u128(age_mask & !*used).next() {
            Some(c) => {
                *used |= 1u128 << c.index();
                c
            }
            None => CardId::from_index(0),
        }
    };
    std::array::from_fn(|i| assigned[i].unwrap_or_else(|| filler(&mut used)))
}

/// Sample the deck for an age that has not been dealt yet.
fn sample_future_deck(age: u8, rng: &mut StdRng) -> [CardId; SLOTS] {
    let s = data::statics();
    if age < 3 {
        let mut d: Vec<CardId> = iter_mask_u128(s.age_masks[(age - 1) as usize]).collect();
        d.shuffle(rng);
        d.truncate(SLOTS);
        std::array::from_fn(|i| d[i])
    } else {
        let mut plain: Vec<CardId> = iter_mask_u128(s.age_masks[2] & !s.guild_mask).collect();
        plain.shuffle(rng);
        plain.truncate(SLOTS - CARDS_REMOVED_PER_AGE);
        let mut guilds: Vec<CardId> = iter_mask_u128(s.guild_mask).collect();
        guilds.shuffle(rng);
        guilds.truncate(GUILDS_IN_PLAY);
        plain.extend(guilds);
        plain.shuffle(rng);
        std::array::from_fn(|i| plain[i])
    }
}

/// Sample a wonder pile consistent with what has been drafted and offered.
fn sample_wonder_pile(obs: &Observation, rng: &mut StdRng) -> [WonderId; data::NUM_WONDERS] {
    let group = (obs.draft_step / 4) as usize;
    let mut pile: Vec<Option<WonderId>> = vec![None; data::NUM_WONDERS];

    // The offered group sits in its own four positions.
    for (i, &w) in obs.offered_wonders.iter().enumerate() {
        pile[group * 4 + i] = Some(w);
    }
    // Everything already drafted goes in the earliest free position, which
    // reproduces `offered_wonders` for the current step.
    let owned: Vec<WonderId> = obs.players[0]
        .wonders
        .iter()
        .chain(obs.players[1].wonders.iter())
        .copied()
        .collect();
    let mut pool = obs.undrafted_wonder_pool.clone();
    pool.shuffle(rng);

    let mut fill = owned;
    fill.extend(pool);
    let mut it = fill.into_iter();
    for slot in pile.iter_mut() {
        if slot.is_none() {
            *slot = it.next();
        }
    }
    std::array::from_fn(|i| pile[i].unwrap_or_else(|| WonderId::from_index(i)))
}

/// The slots that may be taken right now, for convenience.
impl Observation {
    /// Slot indices whose card may be taken, derived from the same geometry
    /// the engine uses.
    pub fn accessible_slots(&self) -> Vec<u8> {
        let occupied = (0..SLOTS)
            .filter(|&i| self.slots[i] != SlotView::Empty)
            .fold(0u32, |m, i| m | (1u32 << i));
        iter_slots(layout::accessible(self.age.max(1), occupied)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;
    use crate::engine::{apply, legal_actions, new_game};
    use rand::SeedableRng;

    fn advanced_game(seed: u64, steps: usize) -> GameState {
        let mut st = new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x55);
        for _ in 0..steps {
            let actions = legal_actions(&st);
            if actions.is_empty() {
                break;
            }
            let a = actions[(st.turn() as usize * 7) % actions.len()];
            apply(&mut st, a, &mut rng).unwrap();
        }
        st
    }

    #[test]
    fn face_down_slots_carry_no_card_id() {
        let st = advanced_game(4, 12);
        let obs = st.observation();
        let json = serde_json::to_string(&obs).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let slots = value["slots"].as_array().unwrap();
        assert_eq!(slots.len(), SLOTS);
        let mut face_down = 0;
        for slot in slots {
            if slot["state"] == "face_down" {
                face_down += 1;
                assert!(
                    slot.get("card").is_none(),
                    "a face-down slot leaked a card id: {slot}"
                );
            }
        }
        assert!(face_down > 0, "the test position should have hidden cards");
    }

    #[test]
    fn permuting_hidden_cards_does_not_change_the_observation() {
        let st = advanced_game(9, 14);
        let before = st.observation();
        let hidden: Vec<u8> = iter_slots(st.occupied_slots() & !st.revealed_slots()).collect();
        assert!(hidden.len() >= 2);
        let mut permuted = st;
        permuted.swap_slot_cards(hidden[0], hidden[1]);
        assert_ne!(
            st.slot_card_hidden(hidden[0]),
            permuted.slot_card_hidden(hidden[0]),
            "the swap should actually change the hidden layout"
        );
        assert_eq!(before, permuted.observation());
    }

    #[test]
    fn changing_which_cards_were_boxed_does_not_change_the_observation() {
        let st = advanced_game(21, 16);
        let before = st.observation();
        // Swap a face-down card with one of the cards returned to the box.
        let hidden: Vec<u8> = iter_slots(st.occupied_slots() & !st.revealed_slots()).collect();
        let s = data::statics();
        let age_mask = s.age_masks[(st.age() - 1) as usize];
        let boxed = iter_mask_u128(st.out_of_game_mask() & age_mask)
            .next()
            .expect("three cards of the age are boxed");
        let mut swapped = st;
        let displaced = swapped.slot_card_hidden(hidden[0]);
        let mut out = swapped.out_of_game_mask();
        out &= !(1u128 << boxed.index());
        out |= 1u128 << displaced.index();
        swapped.set_out_of_game(out);
        let mut deck = *swapped.age_deck(swapped.age());
        deck[hidden[0] as usize] = boxed;
        swapped.set_age_deck(swapped.age(), deck);

        assert_eq!(before, swapped.observation());
    }

    #[test]
    fn the_undrafted_wonder_pool_hides_the_second_group() {
        let st = new_game(2);
        let obs = st.observation();
        assert_eq!(obs.offered_wonders.len(), 4);
        assert_eq!(obs.undrafted_wonder_pool.len(), 8);
        // Permuting the not-yet-offered part of the pile must not show.
        let mut permuted = st;
        let mut pile = *permuted.draft_pile();
        pile.swap(4, 11);
        permuted.set_draft_pile(pile);
        assert_eq!(obs, permuted.observation());
    }

    #[test]
    fn sampled_states_reproduce_the_observation() {
        for seed in 0..12u64 {
            for steps in [0usize, 9, 20, 40, 70] {
                let st = advanced_game(seed, steps);
                let obs = st.observation();
                let mut rng = StdRng::seed_from_u64(seed * 31 + steps as u64);
                for _ in 0..3 {
                    let sampled = obs.sample_state(&mut rng);
                    assert_eq!(
                        sampled.observation(),
                        obs,
                        "seed {seed} steps {steps}: sample_state did not round trip"
                    );
                }
            }
        }
    }

    #[test]
    fn sampled_states_are_playable() {
        let st = advanced_game(6, 25);
        let obs = st.observation();
        let mut rng = StdRng::seed_from_u64(77);
        let mut sampled = obs.sample_state(&mut rng);
        assert_eq!(legal_actions(&sampled), legal_actions(&st));
        // And a full playout from the sample terminates.
        let mut guard = 0;
        loop {
            let actions = legal_actions(&sampled);
            if actions.is_empty() {
                break;
            }
            apply(&mut sampled, actions[0], &mut rng).unwrap();
            guard += 1;
            assert!(guard < 10_000);
        }
        assert!(sampled.is_over());
    }

    #[test]
    fn accessible_slots_agree_with_the_engine() {
        let st = advanced_game(13, 22);
        let obs = st.observation();
        let engine_slots: Vec<u8> = iter_slots(st.accessible_slots()).collect();
        assert_eq!(obs.accessible_slots(), engine_slots);
    }

    #[test]
    fn the_observation_round_trips_through_json() {
        let st = advanced_game(3, 30);
        let obs = st.observation();
        let json = serde_json::to_string(&obs).unwrap();
        let back: Observation = serde_json::from_str(&json).unwrap();
        assert_eq!(obs, back);
    }

    #[test]
    fn a_pending_choice_is_visible_but_the_pool_behind_it_is_not() {
        // The Great Library's three drawn tokens are public once drawn.
        let mut st = crate::testing::StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "obelisk")])
            .wonders(Player::One, &["the-great-library"])
            .built(
                Player::One,
                &["press", "glassworks", "sawmill", "lumber-yard"],
            )
            .set_aside_tokens(&["philosophy", "agriculture", "mathematics", "law", "economy"])
            .coins(Player::One, 20)
            .build();
        let mut rng = StdRng::seed_from_u64(1);
        apply(
            &mut st,
            Action::BuildWonder {
                slot: 18,
                wonder: WonderId::from_slug("the-great-library").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        let obs = st.observation();
        assert!(matches!(
            obs.pending,
            Some(Pending::GreatLibraryToken { .. })
        ));
        assert_eq!(obs.set_aside_tokens.len(), 5);
    }
}
