//! Full game state (server-side / simulation-side only).
//!
//! [`GameState`] holds everything needed to advance the game
//! deterministically, **including information that is not public by the rules
//! of the game**: the identity of the face-down cards in the current age's
//! structure, which cards were returned to the box from each age deck, the
//! composition of the not-yet-dealt age decks, and the wonders that have not
//! yet been offered in the draft.
//!
//! 7 Wonders Duel has no *player-private* hidden information: both players
//! always see identical public state. The only thing ever unknown is future
//! randomness, and it is unknown equally to both players and to a spectator.
//! That distinction still must never leak to an AI agent, so it is enforced by
//! the type system rather than by convention:
//!
//! - every field of `GameState` is private, and the only public accessors are
//!   ones that return public information;
//! - the hidden parts are reachable only through `pub(crate)` accessors, so
//!   nothing outside `duels-core` can read them even by accident;
//! - [`crate::Observation`] is a separate type produced by
//!   [`GameState::observation`], and `duels-agents-api` depends on
//!   `Observation` and never on `GameState`.
//!
//! `GameState` is deliberately [`Copy`] and free of heap allocation: a
//! search-based agent will clone it millions of times per second. Built cards
//! are a `u128` bitset over the 73 card ids, science symbols a `[u8; 7]`, and
//! the board a pair of `u32` bitmasks over the 20 slots.
//!
//! # Serialisation warning
//!
//! `GameState` implements [`Serialize`]/[`Deserialize`] so the authoritative
//! server can persist and restore a game. **A serialised `GameState` contains
//! hidden information and must never be sent to a client or an agent** — send
//! [`crate::Observation`] instead.

use serde::{Deserialize, Serialize};

use crate::data::{
    self, CardId, CardType, CountTarget, Resource, Science, TokenId, WonderId, NUM_RESOURCES,
    NUM_SCIENCE, NUM_WONDERS,
};
use crate::layout::{self, SLOTS};
use crate::scoring::GameResult;
use crate::Player;

/// Total number of wonders that may be constructed across both players in one
/// game. The eighth drafted wonder can never be built.
pub const MAX_WONDERS_BUILT: u8 = 7;

/// Coins each player starts with.
pub const STARTING_COINS: u16 = 7;

/// Progress tokens made available on the board at setup.
pub const TOKENS_ON_BOARD: usize = 5;

/// Wonders drafted in total (4 each).
pub const WONDERS_DRAFTED: usize = 8;

/// Number of cards returned to the box, unseen, from each age deck.
pub const CARDS_REMOVED_PER_AGE: usize = 3;

/// Guild cards shuffled into the Age III deck.
pub const GUILDS_IN_PLAY: usize = 3;

/// What kind of decision the game is waiting for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// The initial wonder draft: players alternate 1-2-1, then 1-2-1 with the
    /// roles reversed.
    WonderDraft,
    /// Normal play: take a card from the structure.
    Turn,
    /// A new age has been dealt and the militarily weaker player must say who
    /// begins it.
    ChooseFirstPlayer,
    /// The game is over; see [`GameState::result`].
    GameOver,
}

/// A choice an effect has created that must be resolved before the turn can
/// pass. At most one is ever outstanding at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Pending {
    /// A pair of identical scientific symbols completed; take a progress
    /// token from the board.
    ProgressToken,
    /// The Great Library drew these three tokens from the out-of-play pile;
    /// keep one.
    GreatLibraryToken {
        /// The three tokens on offer.
        tokens: [TokenId; 3],
    },
    /// A wonder's destroy effect; discard one opponent building of this
    /// colour.
    Destroy {
        /// The colour that may be destroyed.
        card_type: CardType,
    },
    /// The Mausoleum; construct one card from the discard pile for free.
    MausoleumBuild,
}

/// One player's city: what they own and what it produces.
///
/// The `production` / `choice_*` / `fixed_trade` / `science` fields are caches
/// derived from `built`, `wonders_built` and `tokens`; they are maintained
/// incrementally so the cost engine never has to walk the built-cards bitset.
/// [`PlayerState::recompute_derived`] rebuilds them from scratch, which is
/// what a destroy effect needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerState {
    coins: u16,
    built: u128,
    wonders: u16,
    wonders_built: u16,
    tokens: u16,
    production: [u8; NUM_RESOURCES],
    choice_raw: u8,
    choice_manufactured: u8,
    fixed_trade: [bool; NUM_RESOURCES],
    shields: u8,
    science: [u8; NUM_SCIENCE],
    pairs_awarded: u8,
}

impl PlayerState {
    fn new() -> Self {
        Self {
            coins: STARTING_COINS,
            built: 0,
            wonders: 0,
            wonders_built: 0,
            tokens: 0,
            production: [0; NUM_RESOURCES],
            choice_raw: 0,
            choice_manufactured: 0,
            fixed_trade: [false; NUM_RESOURCES],
            shields: 0,
            science: [0; NUM_SCIENCE],
            pairs_awarded: 0,
        }
    }

    /// Coins in this player's treasury.
    #[inline]
    pub fn coins(&self) -> u16 {
        self.coins
    }

    /// Bitset of the cards this player has constructed, indexed by
    /// [`CardId::index`].
    #[inline]
    pub fn built_mask(&self) -> u128 {
        self.built
    }

    /// Whether this player has constructed `card`.
    #[inline]
    pub fn has_built(&self, card: CardId) -> bool {
        self.built & (1u128 << card.index()) != 0
    }

    /// The cards this player has constructed.
    pub fn built(&self) -> impl Iterator<Item = CardId> + '_ {
        iter_mask_u128(self.built)
    }

    /// The four wonders this player drafted.
    pub fn wonders(&self) -> impl Iterator<Item = WonderId> + '_ {
        iter_mask_u16(self.wonders).map(WonderId::from_index)
    }

    /// The wonders this player has constructed.
    pub fn wonders_built(&self) -> impl Iterator<Item = WonderId> + '_ {
        iter_mask_u16(self.wonders_built).map(WonderId::from_index)
    }

    /// Whether this player drafted `wonder`.
    #[inline]
    pub fn owns_wonder(&self, wonder: WonderId) -> bool {
        self.wonders & (1u16 << wonder.index()) != 0
    }

    /// Whether this player has constructed `wonder`.
    #[inline]
    pub fn has_built_wonder(&self, wonder: WonderId) -> bool {
        self.wonders_built & (1u16 << wonder.index()) != 0
    }

    /// How many wonders this player has constructed.
    #[inline]
    pub fn wonder_count(&self) -> u8 {
        self.wonders_built.count_ones() as u8
    }

    /// The progress tokens this player owns.
    pub fn tokens(&self) -> impl Iterator<Item = TokenId> + '_ {
        iter_mask_u16(self.tokens).map(TokenId::from_index)
    }

    /// Whether this player owns `token`.
    #[inline]
    pub fn has_token(&self, token: TokenId) -> bool {
        self.tokens & (1u16 << token.index()) != 0
    }

    /// How many progress tokens this player owns.
    #[inline]
    pub fn token_count(&self) -> u8 {
        self.tokens.count_ones() as u8
    }

    /// Resources produced by this player's brown, grey and green cards plus
    /// fixed-production wonders. Choice-production sources are counted
    /// separately (see [`PlayerState::choice_sources`]).
    #[inline]
    pub fn production(&self) -> [u8; NUM_RESOURCES] {
        self.production
    }

    /// `(raw, manufactured)` counts of "produce one of your choice" sources.
    #[inline]
    pub fn choice_sources(&self) -> (u8, u8) {
        (self.choice_raw, self.choice_manufactured)
    }

    /// Resources this player may buy for a flat 1 coin per unit.
    #[inline]
    pub fn fixed_trade(&self) -> [bool; NUM_RESOURCES] {
        self.fixed_trade
    }

    /// Total shields this player has ever generated. Destroying a red card
    /// does not undo the conflict-pawn movement it caused, so this is an
    /// accumulator, not a count of surviving cards.
    #[inline]
    pub fn shields(&self) -> u8 {
        self.shields
    }

    /// Count of each scientific symbol this player holds (0, 1 or 2).
    #[inline]
    pub fn science(&self) -> [u8; NUM_SCIENCE] {
        self.science
    }

    /// How many *distinct* scientific symbols this player holds. Six wins the
    /// game outright.
    #[inline]
    pub fn distinct_science(&self) -> u8 {
        self.science.iter().filter(|&&n| n > 0).count() as u8
    }

    /// The brown + grey production the *opponent* uses to price this player's
    /// trades. Yellow cards and wonders are excluded per the rules, which is
    /// why this is not simply [`PlayerState::production`].
    pub fn trade_relevant_production(&self, resource: Resource) -> u8 {
        let s = data::statics();
        let mask = self.built & s.raw_and_manufactured_mask;
        iter_mask_u128(mask)
            .map(|c| c.def().produces[resource.index()])
            .sum()
    }

    /// How many units of `target` this player holds.
    pub fn count(&self, target: CountTarget) -> u16 {
        let s = data::statics();
        match target {
            CountTarget::Cards(t) => (self.built & s.card_masks[t.index()]).count_ones() as u16,
            CountTarget::RawAndManufactured => {
                (self.built & s.raw_and_manufactured_mask).count_ones() as u16
            }
            CountTarget::Wonders => self.wonders_built.count_ones() as u16,
            CountTarget::CoinsDiv3 => self.coins / 3,
        }
    }

    /// Rebuild the derived production / science caches from `built`,
    /// `wonders_built` and `tokens`.
    fn recompute_derived(&mut self) {
        self.production = [0; NUM_RESOURCES];
        self.choice_raw = 0;
        self.choice_manufactured = 0;
        self.fixed_trade = [false; NUM_RESOURCES];
        self.science = [0; NUM_SCIENCE];
        for card in iter_mask_u128(self.built) {
            let d = card.def();
            for r in 0..NUM_RESOURCES {
                self.production[r] += d.produces[r];
                self.fixed_trade[r] |= d.fixed_trade[r];
            }
            match d.produces_choice {
                Some(data::ResourceGroup::RawMaterial) => self.choice_raw += 1,
                Some(data::ResourceGroup::ManufacturedGood) => self.choice_manufactured += 1,
                None => {}
            }
            if let Some(sym) = d.science {
                self.science[sym.index()] += 1;
            }
        }
        for w in iter_mask_u16(self.wonders_built).map(WonderId::from_index) {
            match w.def().produces_choice {
                Some(data::ResourceGroup::RawMaterial) => self.choice_raw += 1,
                Some(data::ResourceGroup::ManufacturedGood) => self.choice_manufactured += 1,
                None => {}
            }
        }
        for t in iter_mask_u16(self.tokens).map(TokenId::from_index) {
            if let Some(sym) = t.def().science {
                self.science[sym.index()] += 1;
            }
        }
    }

    // -- crate-internal mutation ------------------------------------------

    pub(crate) fn coins_mut(&mut self) -> &mut u16 {
        &mut self.coins
    }

    /// Subtract coins, flooring at zero. Returns how many were actually paid.
    pub(crate) fn pay_up_to(&mut self, amount: u16) -> u16 {
        let paid = amount.min(self.coins);
        self.coins -= paid;
        paid
    }

    pub(crate) fn add_built_card(&mut self, card: CardId) {
        self.built |= 1u128 << card.index();
        let d = card.def();
        for r in 0..NUM_RESOURCES {
            self.production[r] += d.produces[r];
            self.fixed_trade[r] |= d.fixed_trade[r];
        }
        match d.produces_choice {
            Some(data::ResourceGroup::RawMaterial) => self.choice_raw += 1,
            Some(data::ResourceGroup::ManufacturedGood) => self.choice_manufactured += 1,
            None => {}
        }
        if let Some(sym) = d.science {
            self.science[sym.index()] += 1;
        }
    }

    pub(crate) fn remove_built_card(&mut self, card: CardId) {
        self.built &= !(1u128 << card.index());
        self.recompute_derived();
    }

    pub(crate) fn draft_wonder(&mut self, wonder: WonderId) {
        self.wonders |= 1u16 << wonder.index();
    }

    pub(crate) fn mark_wonder_built(&mut self, wonder: WonderId) {
        self.wonders_built |= 1u16 << wonder.index();
        match wonder.def().produces_choice {
            Some(data::ResourceGroup::RawMaterial) => self.choice_raw += 1,
            Some(data::ResourceGroup::ManufacturedGood) => self.choice_manufactured += 1,
            None => {}
        }
    }

    pub(crate) fn add_token(&mut self, token: TokenId) {
        self.tokens |= 1u16 << token.index();
        if let Some(sym) = token.def().science {
            self.science[sym.index()] += 1;
        }
    }

    pub(crate) fn add_shields(&mut self, n: u8) {
        self.shields = self.shields.saturating_add(n);
    }

    pub(crate) fn mark_pair_awarded(&mut self, sym: Science) {
        self.pairs_awarded |= 1u8 << sym.index();
    }

    pub(crate) fn pair_already_awarded(&self, sym: Science) -> bool {
        self.pairs_awarded & (1u8 << sym.index()) != 0
    }

    /// Symbols for which this player has already been awarded a progress
    /// token. Public: everyone saw it happen.
    pub fn pairs_awarded(&self) -> impl Iterator<Item = Science> + '_ {
        const ALL: [Science; NUM_SCIENCE] = [
            Science::Mortar,
            Science::Pendulum,
            Science::Inkwell,
            Science::Wheel,
            Science::Sundial,
            Science::Gyroscope,
            Science::Balance,
        ];
        let mask = self.pairs_awarded;
        ALL.into_iter()
            .filter(move |s| mask & (1u8 << s.index()) != 0)
    }

    /// The progress token the player owns that satisfies `f`, if any.
    pub(crate) fn token_with(&self, f: impl Fn(&data::ProgressToken) -> bool) -> Option<TokenId> {
        iter_mask_u16(self.tokens)
            .map(TokenId::from_index)
            .find(|t| f(t.def()))
    }

    /// Whether the player owns a token satisfying `f`.
    #[inline]
    pub(crate) fn has_token_with(&self, f: impl Fn(&data::ProgressToken) -> bool) -> bool {
        self.token_with(f).is_some()
    }
}

/// The full, authoritative game state.
///
/// See the module docs: this contains hidden information and must never be
/// handed to an agent. Use [`GameState::observation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameState {
    players: [PlayerState; 2],
    current: Player,
    age: u8,
    phase: Phase,
    pending: Option<Pending>,
    extra_turn: bool,
    conflict: i8,
    /// `loot_taken[p][i]` is true once the loot token at
    /// `data::military().loot[i]` on the side that player `p` pushes towards
    /// has been collected.
    loot_taken: [[bool; 2]; 2],
    occupied: u32,
    revealed: u32,
    discard: u128,
    wonder_fodder: u128,
    /// HIDDEN: the card dealt into each slot of each age, in slot order.
    age_decks: [[CardId; SLOTS]; 3],
    /// HIDDEN: the cards returned to the box during setup.
    out_of_game_cards: u128,
    /// HIDDEN beyond the currently offered group: the shuffled wonder pile.
    /// Positions `0..4` are the first draft group, `4..8` the second, and
    /// `8..12` were returned to the box.
    draft_wonders: [WonderId; NUM_WONDERS],
    draft_step: u8,
    draft_first: Player,
    tokens_board: u16,
    /// Set aside at setup. Publicly deducible (the complement of the board
    /// five), so this is *not* hidden information; The Great Library draws
    /// three of these at random when it is built.
    tokens_aside: u16,
    last_card_taker: Player,
    turn: u32,
    result: Option<GameResult>,
}

impl GameState {
    pub(crate) fn empty() -> Self {
        let placeholder = CardId::from_index(0);
        Self {
            players: [PlayerState::new(), PlayerState::new()],
            current: Player::One,
            age: 1,
            phase: Phase::WonderDraft,
            pending: None,
            extra_turn: false,
            conflict: 0,
            loot_taken: [[false; 2]; 2],
            occupied: 0,
            revealed: 0,
            discard: 0,
            wonder_fodder: 0,
            age_decks: [[placeholder; SLOTS]; 3],
            out_of_game_cards: 0,
            draft_wonders: [WonderId::from_index(0); NUM_WONDERS],
            draft_step: 0,
            draft_first: Player::One,
            tokens_board: 0,
            tokens_aside: 0,
            last_card_taker: Player::One,
            turn: 0,
            result: None,
        }
    }

    // -- public, public-information accessors ------------------------------

    /// The player whose decision the engine is waiting for.
    #[inline]
    pub fn current_player(&self) -> Player {
        self.current
    }

    /// One player's public city state.
    #[inline]
    pub fn player(&self, p: Player) -> &PlayerState {
        &self.players[p.index()]
    }

    /// The current age, 1 to 3.
    #[inline]
    pub fn age(&self) -> u8 {
        self.age
    }

    /// What kind of decision is pending.
    #[inline]
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// The outstanding effect choice, if any.
    #[inline]
    pub fn pending(&self) -> Option<Pending> {
        self.pending
    }

    /// Whether the current player has an extra turn banked.
    #[inline]
    pub fn extra_turn(&self) -> bool {
        self.extra_turn
    }

    /// Conflict pawn position: positive means [`Player::One`] is ahead
    /// (the pawn has been pushed towards Player Two's capital).
    #[inline]
    pub fn conflict(&self) -> i8 {
        self.conflict
    }

    /// The player the conflict pawn currently favours, or `None` at centre.
    #[inline]
    pub fn military_leader(&self) -> Option<Player> {
        match self.conflict.cmp(&0) {
            std::cmp::Ordering::Greater => Some(Player::One),
            std::cmp::Ordering::Less => Some(Player::Two),
            std::cmp::Ordering::Equal => None,
        }
    }

    /// Whether the loot token at `data::military().loot[index]` on the side
    /// `pusher` pushes towards is still on the board.
    #[inline]
    pub fn loot_available(&self, pusher: Player, index: usize) -> bool {
        !self.loot_taken[pusher.index()][index]
    }

    /// Bitmask of slots that still hold a card.
    #[inline]
    pub fn occupied_slots(&self) -> u32 {
        self.occupied
    }

    /// Bitmask of slots whose card is face up.
    #[inline]
    pub fn revealed_slots(&self) -> u32 {
        self.revealed
    }

    /// Bitmask of slots whose card may be taken right now.
    #[inline]
    pub fn accessible_slots(&self) -> u32 {
        layout::accessible(self.age, self.occupied)
    }

    /// The face-up card in `slot`, or `None` if the slot is empty or still
    /// face down.
    #[inline]
    pub fn face_up_card(&self, slot: u8) -> Option<CardId> {
        let bit = 1u32 << slot;
        if self.occupied & bit != 0 && self.revealed & bit != 0 {
            Some(self.age_decks[(self.age - 1) as usize][slot as usize])
        } else {
            None
        }
    }

    /// The cards in the shared discard pile, available to The Mausoleum.
    pub fn discard_pile(&self) -> impl Iterator<Item = CardId> + '_ {
        iter_mask_u128(self.discard)
    }

    /// Bitset of the discard pile.
    #[inline]
    pub fn discard_mask(&self) -> u128 {
        self.discard
    }

    /// Bitset of cards spent to construct wonders. Those cards are out of
    /// play and are *not* in the discard pile.
    #[inline]
    pub fn wonder_fodder_mask(&self) -> u128 {
        self.wonder_fodder
    }

    /// Progress tokens still available on the board.
    pub fn board_tokens(&self) -> impl Iterator<Item = TokenId> + '_ {
        iter_mask_u16(self.tokens_board).map(TokenId::from_index)
    }

    /// Bitset of the progress tokens on the board.
    #[inline]
    pub fn board_tokens_mask(&self) -> u16 {
        self.tokens_board
    }

    /// Progress tokens set aside at setup and still eligible for The Great
    /// Library's draw. Public information: it is the complement of the five
    /// placed on the board.
    pub fn set_aside_tokens(&self) -> impl Iterator<Item = TokenId> + '_ {
        iter_mask_u16(self.tokens_aside).map(TokenId::from_index)
    }

    /// Bitset of the set-aside progress tokens.
    #[inline]
    pub fn set_aside_tokens_mask(&self) -> u16 {
        self.tokens_aside
    }

    /// Total wonders constructed by both players.
    #[inline]
    pub fn wonders_built_total(&self) -> u8 {
        self.players[0].wonder_count() + self.players[1].wonder_count()
    }

    /// Whether another wonder may still be constructed.
    #[inline]
    pub fn wonder_slots_left(&self) -> bool {
        self.wonders_built_total() < MAX_WONDERS_BUILT
    }

    /// How many decisions have been resolved so far. Useful for logging and
    /// for detecting non-progressing loops in tests.
    #[inline]
    pub fn turn(&self) -> u32 {
        self.turn
    }

    /// The player who took the last card of the age. Decides who begins the
    /// next age when the conflict pawn is centred.
    #[inline]
    pub fn last_card_taker(&self) -> Player {
        self.last_card_taker
    }

    /// The outcome, once [`GameState::phase`] is [`Phase::GameOver`].
    #[inline]
    pub fn result(&self) -> Option<GameResult> {
        self.result
    }

    /// Whether the game has finished.
    #[inline]
    pub fn is_over(&self) -> bool {
        self.phase == Phase::GameOver
    }

    /// Which player is choosing during the wonder draft, and which step of
    /// the eight-pick sequence it is.
    #[inline]
    pub fn draft_step(&self) -> u8 {
        self.draft_step
    }

    /// The wonders currently on offer in the draft. Empty outside
    /// [`Phase::WonderDraft`].
    pub fn offered_wonders(&self) -> Vec<WonderId> {
        if self.phase != Phase::WonderDraft {
            return Vec::new();
        }
        let group = (self.draft_step / 4) as usize;
        let taken = self.players[0].wonders | self.players[1].wonders;
        self.draft_wonders[group * 4..group * 4 + 4]
            .iter()
            .copied()
            .filter(|w| taken & (1u16 << w.index()) == 0)
            .collect()
    }

    /// Who picks first in the wonder draft. Public information.
    #[inline]
    pub fn draft_first(&self) -> Player {
        self.draft_first
    }

    /// Whose turn it is to pick in the draft, per the 1-2-1 / 2-1-2 order.
    pub(crate) fn draft_picker(&self) -> Player {
        draft_order(self.draft_first)[self.draft_step as usize]
    }

    /// Check every structural invariant this type is supposed to guarantee.
    ///
    /// This lives here, rather than in the test suite, precisely *because*
    /// most of these invariants are about hidden fields: an external test
    /// cannot read them without breaking the `GameState` / `Observation`
    /// separation, so the check is exposed as a method instead of the fields
    /// being exposed as data.
    ///
    /// Returns `Err` with a human-readable description of the first violation.
    /// Cheap enough to call from a property test at every decision point;
    /// too expensive for a hot search loop.
    pub fn check_invariants(&self) -> Result<(), String> {
        let s = data::statics();
        let all_cards = (1u128 << data::NUM_CARDS) - 1;

        // Card conservation: 60 dealt across the three ages, the rest boxed,
        // together exactly the 73 cards, with no overlap.
        let mut dealt = 0u128;
        let mut dealt_count = 0u32;
        for age in 1..=3usize {
            for &c in self.age_decks[age - 1].iter() {
                let b = 1u128 << c.index();
                if dealt & b != 0 {
                    return Err(format!("card {c} is dealt into more than one slot"));
                }
                if s.age_masks[age - 1] & b == 0 {
                    return Err(format!("card {c} is dealt into age {age}"));
                }
                dealt |= b;
                dealt_count += 1;
            }
        }
        if dealt_count != 3 * SLOTS as u32 {
            return Err(format!("{dealt_count} cards dealt, expected 60"));
        }
        if dealt & self.out_of_game_cards != 0 {
            return Err("a dealt card is also out of the game".into());
        }
        if (dealt | self.out_of_game_cards) != all_cards {
            return Err(format!(
                "{} cards accounted for, expected {}",
                (dealt | self.out_of_game_cards).count_ones(),
                data::NUM_CARDS
            ));
        }
        // Exactly three guilds in the Age III structure.
        let guilds_dealt = (self.age_decks[2]
            .iter()
            .fold(0u128, |m, c| m | (1u128 << c.index()))
            & s.guild_mask)
            .count_ones();
        if guilds_dealt != GUILDS_IN_PLAY as u32 {
            return Err(format!("{guilds_dealt} guilds dealt, expected 3"));
        }

        // Each card that has left a structure is in exactly one place.
        let places = [
            self.players[0].built,
            self.players[1].built,
            self.discard,
            self.wonder_fodder,
        ];
        let mut union = 0u128;
        let mut sum = 0u32;
        for m in places {
            if m & self.out_of_game_cards != 0 {
                return Err("a boxed card entered play".into());
            }
            union |= m;
            sum += m.count_ones();
        }
        if sum != union.count_ones() {
            return Err("a card is in two of {city, city, discard, wonder}".into());
        }
        // ...and is not simultaneously still sitting in the structure.
        for slot in iter_slots(self.occupied) {
            let card = self.slot_card_hidden(slot);
            if union & (1u128 << card.index()) != 0 {
                return Err(format!("{card} is both in the structure and in play"));
            }
        }
        // Revealed slots must be occupied.
        if self.revealed & !self.occupied != 0 {
            return Err("an empty slot is marked revealed".into());
        }
        // An accessible slot must be face up: uncovering turns cards over.
        let accessible = layout::accessible(self.age, self.occupied);
        if accessible & !self.revealed != 0 {
            return Err("an accessible slot is still face down".into());
        }

        // Coins are unsigned; the meaningful check is that nothing wrapped.
        for p in Player::ALL {
            if self.players[p.index()].coins > 1000 {
                return Err(format!(
                    "{p} holds {} coins, which suggests a wrapped subtraction",
                    self.players[p.index()].coins
                ));
            }
        }

        // Wonders.
        if self.wonders_built_total() > MAX_WONDERS_BUILT {
            return Err(format!("{} wonders built", self.wonders_built_total()));
        }
        if self.players[0].wonders & self.players[1].wonders != 0 {
            return Err("both players drafted the same wonder".into());
        }
        for p in Player::ALL {
            let ps = &self.players[p.index()];
            if ps.wonders.count_ones() > 4 {
                return Err(format!("{p} drafted {} wonders", ps.wonders.count_ones()));
            }
            if ps.wonders_built & !ps.wonders != 0 {
                return Err(format!("{p} built an undrafted wonder"));
            }
        }

        // Science: two copies of a symbol at most, and six distinct symbols
        // ends the game at once.
        for p in Player::ALL {
            let ps = &self.players[p.index()];
            if ps.science.iter().any(|&n| n > 2) {
                return Err(format!("{p} holds more than two of a symbol"));
            }
            if ps.distinct_science() >= 6 && self.phase != Phase::GameOver {
                return Err(format!("{p} has six symbols but play continues"));
            }
        }

        // Military.
        let cap = i8::try_from(data::military().capital_distance).unwrap_or(9);
        if self.conflict.abs() > cap {
            return Err(format!("conflict pawn at {}", self.conflict));
        }
        if self.conflict.abs() == cap && self.phase != Phase::GameOver {
            return Err("the pawn reached a capital but play continues".into());
        }

        // Progress tokens.
        if self.tokens_board & self.tokens_aside != 0 {
            return Err("a token is both on the board and set aside".into());
        }
        let owned = self.players[0].tokens | self.players[1].tokens;
        if self.players[0].tokens & self.players[1].tokens != 0 {
            return Err("both players own the same progress token".into());
        }
        if owned & (self.tokens_board | self.tokens_aside) != 0 {
            return Err("an owned token is still available".into());
        }

        // Phase consistency.
        match self.phase {
            Phase::GameOver => {
                if self.result.is_none() {
                    return Err("the game is over with no result".into());
                }
            }
            Phase::WonderDraft => {
                if self.occupied != 0 {
                    return Err("a structure was dealt during the draft".into());
                }
            }
            Phase::Turn | Phase::ChooseFirstPlayer => {
                if self.result.is_some() {
                    return Err("play continues after a result was recorded".into());
                }
                if self.phase == Phase::ChooseFirstPlayer && self.pending.is_some() {
                    return Err("a pending effect survived into a new age".into());
                }
            }
        }

        Ok(())
    }

    // -- crate-internal accessors (hidden information) ---------------------

    pub(crate) fn player_mut(&mut self, p: Player) -> &mut PlayerState {
        &mut self.players[p.index()]
    }

    /// HIDDEN: the true card in `slot` of the current age, face up or not.
    pub(crate) fn slot_card_hidden(&self, slot: u8) -> CardId {
        self.age_decks[(self.age - 1) as usize][slot as usize]
    }

    pub(crate) fn age_deck(&self, age: u8) -> &[CardId; SLOTS] {
        &self.age_decks[(age - 1) as usize]
    }

    pub(crate) fn set_age_deck(&mut self, age: u8, deck: [CardId; SLOTS]) {
        self.age_decks[(age - 1) as usize] = deck;
    }

    pub(crate) fn swap_slot_cards(&mut self, a: u8, b: u8) {
        let deck = &mut self.age_decks[(self.age - 1) as usize];
        deck.swap(a as usize, b as usize);
    }

    pub(crate) fn out_of_game_mask(&self) -> u128 {
        self.out_of_game_cards
    }

    pub(crate) fn set_out_of_game(&mut self, mask: u128) {
        self.out_of_game_cards = mask;
    }

    pub(crate) fn draft_pile(&self) -> &[WonderId; NUM_WONDERS] {
        &self.draft_wonders
    }

    pub(crate) fn set_draft_pile(&mut self, pile: [WonderId; NUM_WONDERS]) {
        self.draft_wonders = pile;
    }

    pub(crate) fn set_draft_first(&mut self, p: Player) {
        self.draft_first = p;
        self.current = p;
    }

    pub(crate) fn set_draft_step(&mut self, step: u8) {
        self.draft_step = step;
    }

    pub(crate) fn set_turn(&mut self, turn: u32) {
        self.turn = turn;
    }

    pub(crate) fn advance_draft(&mut self) {
        self.draft_step += 1;
        if self.draft_step as usize >= WONDERS_DRAFTED {
            self.phase = Phase::Turn;
            self.current = self.draft_first;
        } else {
            self.current = self.draft_picker();
        }
    }

    pub(crate) fn set_phase(&mut self, phase: Phase) {
        self.phase = phase;
    }

    pub(crate) fn set_current(&mut self, p: Player) {
        self.current = p;
    }

    pub(crate) fn set_age(&mut self, age: u8) {
        self.age = age;
    }

    pub(crate) fn set_pending(&mut self, pending: Option<Pending>) {
        self.pending = pending;
    }

    pub(crate) fn set_extra_turn(&mut self, v: bool) {
        self.extra_turn = v;
    }

    pub(crate) fn set_last_card_taker(&mut self, p: Player) {
        self.last_card_taker = p;
    }

    pub(crate) fn bump_turn(&mut self) {
        self.turn += 1;
    }

    pub(crate) fn set_result(&mut self, r: GameResult) {
        self.result = Some(r);
        self.phase = Phase::GameOver;
    }

    pub(crate) fn set_board(&mut self, occupied: u32, revealed: u32) {
        self.occupied = occupied;
        self.revealed = revealed;
    }

    pub(crate) fn clear_slot(&mut self, slot: u8) {
        self.occupied &= !(1u32 << slot);
        self.revealed &= !(1u32 << slot);
    }

    pub(crate) fn reveal_slot(&mut self, slot: u8) {
        self.revealed |= 1u32 << slot;
    }

    pub(crate) fn add_to_discard(&mut self, card: CardId) {
        self.discard |= 1u128 << card.index();
    }

    pub(crate) fn remove_from_discard(&mut self, card: CardId) {
        self.discard &= !(1u128 << card.index());
    }

    pub(crate) fn add_wonder_fodder(&mut self, card: CardId) {
        self.wonder_fodder |= 1u128 << card.index();
    }

    pub(crate) fn set_conflict(&mut self, c: i8) {
        self.conflict = c;
    }

    pub(crate) fn take_loot(&mut self, pusher: Player, index: usize) {
        self.loot_taken[pusher.index()][index] = true;
    }

    pub(crate) fn set_tokens(&mut self, board: u16, aside: u16) {
        self.tokens_board = board;
        self.tokens_aside = aside;
    }

    pub(crate) fn remove_board_token(&mut self, token: TokenId) {
        self.tokens_board &= !(1u16 << token.index());
    }

    pub(crate) fn remove_aside_tokens(&mut self, mask: u16) {
        self.tokens_aside &= !mask;
    }
}

/// The order players pick in during the wonder draft: the first player takes
/// one, the second takes two, the first takes the last; then the same with
/// the roles reversed for the second group of four.
pub(crate) fn draft_order(first: Player) -> [Player; WONDERS_DRAFTED] {
    let a = first;
    let b = first.other();
    [a, b, b, a, b, a, a, b]
}

/// Iterate the set bits of a 128-bit card bitset as [`CardId`]s.
#[inline]
pub(crate) fn iter_mask_u128(mut mask: u128) -> impl Iterator<Item = CardId> {
    std::iter::from_fn(move || {
        if mask == 0 {
            None
        } else {
            let i = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            Some(CardId::from_index(i))
        }
    })
}

/// Iterate the set bits of a 16-bit bitset as indices.
#[inline]
pub(crate) fn iter_mask_u16(mut mask: u16) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if mask == 0 {
            None
        } else {
            let i = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            Some(i)
        }
    })
}

/// Iterate the set bits of a 32-bit slot bitmask as slot indices.
#[inline]
pub(crate) fn iter_slots(mut mask: u32) -> impl Iterator<Item = u8> {
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

    #[test]
    fn game_state_is_copy_and_small() {
        // A search agent clones this constantly; keep an eye on its size.
        fn assert_copy<T: Copy>() {}
        assert_copy::<GameState>();
        let size = std::mem::size_of::<GameState>();
        assert!(size <= 320, "GameState grew to {size} bytes");
        // Recorded in docs/rules-spec.md; update both if this changes.
        println!("size_of::<GameState>() = {size}");
    }

    #[test]
    fn draft_order_is_one_two_one_then_reversed() {
        assert_eq!(
            draft_order(Player::One),
            [
                Player::One,
                Player::Two,
                Player::Two,
                Player::One,
                Player::Two,
                Player::One,
                Player::One,
                Player::Two,
            ]
        );
        // Each player picks four.
        let order = draft_order(Player::Two);
        assert_eq!(order.iter().filter(|p| **p == Player::One).count(), 4);
        assert_eq!(order.iter().filter(|p| **p == Player::Two).count(), 4);
    }

    #[test]
    fn bitset_iteration_round_trips() {
        let a = CardId::from_slug("tavern").unwrap();
        let b = CardId::from_slug("palace").unwrap();
        let mask = (1u128 << a.index()) | (1u128 << b.index());
        let mut got: Vec<_> = iter_mask_u128(mask).collect();
        got.sort();
        let mut want = vec![a, b];
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn pay_up_to_floors_at_zero() {
        let mut p = PlayerState::new();
        *p.coins_mut() = 3;
        assert_eq!(p.pay_up_to(5), 3);
        assert_eq!(p.coins(), 0);
        assert_eq!(p.pay_up_to(2), 0);
        assert_eq!(p.coins(), 0);
    }

    #[test]
    fn derived_caches_match_a_full_recompute() {
        let mut p = PlayerState::new();
        for slug in [
            "sawmill",
            "forum",
            "stone-reserve",
            "workshop",
            "laboratory",
            "customs-house",
        ] {
            p.add_built_card(CardId::from_slug(slug).unwrap());
        }
        p.mark_wonder_built(WonderId::from_slug("the-great-lighthouse").unwrap());
        p.add_token(TokenId::from_slug("law").unwrap());
        let incremental = p;
        p.recompute_derived();
        assert_eq!(incremental.production, p.production);
        assert_eq!(incremental.choice_raw, p.choice_raw);
        assert_eq!(incremental.choice_manufactured, p.choice_manufactured);
        assert_eq!(incremental.fixed_trade, p.fixed_trade);
        assert_eq!(incremental.science, p.science);
        // Two pendulums -> that symbol is doubled, plus Law's balance.
        assert_eq!(p.science[Science::Pendulum.index()], 2);
        assert_eq!(p.science[Science::Balance.index()], 1);
        assert_eq!(p.distinct_science(), 2);
    }

    #[test]
    fn trade_relevant_production_excludes_yellow_and_wonders() {
        let mut p = PlayerState::new();
        p.add_built_card(CardId::from_slug("sawmill").unwrap()); // brown, wood x2
        p.add_built_card(CardId::from_slug("forum").unwrap()); // yellow choice
        p.mark_wonder_built(WonderId::from_slug("the-great-lighthouse").unwrap());
        assert_eq!(p.trade_relevant_production(Resource::Wood), 2);
        // Neither the Forum nor The Great Lighthouse raises what the opponent
        // pays.
        assert_eq!(p.trade_relevant_production(Resource::Glass), 0);
        assert_eq!(p.trade_relevant_production(Resource::Stone), 0);
    }
}
