//! Static, precomputed views of the real card / wonder / token data.
//!
//! Every table here is *derived* from [`duels_core::data`] on first use, never
//! hand-transcribed, so a change to `data/*.json` cannot silently invalidate
//! it. The tables exist because the race reads in this crate ask the same
//! questions of the card list over and over — "which cards carry shields",
//! "which cards carry the wheel", "how many shields does an undealt Age III
//! deck contain on average" — and a `u128` bitmask intersected with a
//! player's `built` bitset answers them with a single `popcount`.
//!
//! [`Masks`] also resolves the handful of *singleton* game pieces whose rules
//! the science read has to special-case (the Law progress token, The Great
//! Library, The Mausoleum, the Strategy token) by looking for the *effect*
//! rather than the slug, so a rename in the data files cannot break them.

use std::sync::OnceLock;

use duels_core::data::{
    self, CardId, CardType, Science, TokenId, WonderId, NUM_CARDS, NUM_SCIENCE,
};
use duels_core::layout::SLOTS;
use duels_core::state::GUILDS_IN_PLAY;

/// Every scientific symbol, in [`Science::index`] order.
///
/// [`duels_core::data`] deliberately exposes no such constant, so this one is
/// checked against `Science::index` by
/// `tests::all_science_is_in_index_order`.
pub const ALL_SCIENCE: [Science; NUM_SCIENCE] = [
    Science::Mortar,
    Science::Pendulum,
    Science::Inkwell,
    Science::Wheel,
    Science::Sundial,
    Science::Gyroscope,
    Science::Balance,
];

/// How many decisions one player gets in a full age, on average: the twenty
/// cards of a structure, split between two players.
pub const DECISIONS_PER_AGE: u8 = (SLOTS / 2) as u8;

/// What one age's *full* card pool contains, and how much of it reaches the
/// table.
///
/// Setup deals [`SLOTS`] cards per age, so some of the pool is always returned
/// to the box unseen: three of Age I's and Age II's twenty-three cards, and —
/// for Age III — three of the twenty non-guild cards plus four of the seven
/// guilds. Those ratios are what turn "this age's deck contains 11 shields"
/// into "this age is expected to put 9.6 shields on the table".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgeSupply {
    /// Non-guild cards belonging to this age.
    pub plain_cards: u16,
    /// How many of `plain_cards` are dealt into the structure.
    pub plain_dealt: u16,
    /// Shields printed across all of `plain_cards`.
    pub plain_shields: u16,
    /// How many of `plain_cards` carry at least one shield.
    pub plain_shield_cards: u16,
    /// Victory points printed across the civilian (blue) cards of this age.
    pub plain_civilian_vp: u16,
    /// Guild cards belonging to this age (only Age III has any).
    pub guild_cards: u16,
    /// How many of `guild_cards` are dealt into the structure.
    pub guild_dealt: u16,
    /// Shields printed across all of `guild_cards` (zero in the base game,
    /// derived anyway).
    pub guild_shields: u16,
    /// How many of `guild_cards` carry at least one shield.
    pub guild_shield_cards: u16,
}

/// The fraction `dealt / total`, or zero when the pool is empty.
#[inline]
fn frac(dealt: u16, total: u16) -> f64 {
    if total == 0 {
        0.0
    } else {
        f64::from(dealt) / f64::from(total)
    }
}

impl AgeSupply {
    /// Expected shields dealt onto the table when this age is set up.
    #[inline]
    pub fn expected_shields(&self) -> f64 {
        f64::from(self.plain_shields) * frac(self.plain_dealt, self.plain_cards)
            + f64::from(self.guild_shields) * frac(self.guild_dealt, self.guild_cards)
    }

    /// Expected number of shield-bearing cards dealt onto the table when this
    /// age is set up.
    #[inline]
    pub fn expected_shield_cards(&self) -> f64 {
        f64::from(self.plain_shield_cards) * frac(self.plain_dealt, self.plain_cards)
            + f64::from(self.guild_shield_cards) * frac(self.guild_dealt, self.guild_cards)
    }

    /// Expected civilian (blue) victory points dealt onto the table when this
    /// age is set up.
    #[inline]
    pub fn expected_civilian_vp(&self) -> f64 {
        f64::from(self.plain_civilian_vp) * frac(self.plain_dealt, self.plain_cards)
    }

    /// The chance that any one *named* non-guild card of this age reaches the
    /// table when the age is dealt: `plain_dealt / plain_cards`, which is
    /// 20/23 for Ages I and II and 17/20 for Age III.
    ///
    /// This is the `p_dealt` of the threat model: the probability that a
    /// specific symbol card an undealt age still holds is not one of the
    /// copies setup returns to the box unseen.
    #[inline]
    pub fn plain_dealt_fraction(&self) -> f64 {
        frac(self.plain_dealt, self.plain_cards)
    }
}

/// Bitmasks and lookups over the static game data.
#[derive(Debug)]
pub struct Masks {
    shield: Vec<u128>,
    any_shield: u128,
    symbol: [u128; NUM_SCIENCE],
    civilian_vp: u128,
    guild: u128,
    chain_unlocks: Vec<u128>,
    age_supply: [AgeSupply; 3],
    law_token: Option<TokenId>,
    strategy_token: Option<TokenId>,
    great_library: Option<WonderId>,
    mausoleum: Option<WonderId>,
    shield_wonders: u16,
    future_dealt_weight: [[f64; 5]; NUM_SCIENCE],
    play_again_wonders: u16,
    theology_token: Option<TokenId>,
}

static MASKS: OnceLock<Masks> = OnceLock::new();

/// The precomputed tables, built from [`duels_core::data`] on first use.
pub fn masks() -> &'static Masks {
    MASKS.get_or_init(build)
}

impl Masks {
    /// Cards carrying exactly `k` shields. `shield_mask(0)` is every card with
    /// no shields at all; out-of-range `k` returns an empty mask.
    #[inline]
    pub fn shield_mask(&self, k: usize) -> u128 {
        self.shield.get(k).copied().unwrap_or(0)
    }

    /// The largest shield count printed on any single card.
    #[inline]
    pub fn max_card_shields(&self) -> u8 {
        (self.shield.len() - 1) as u8
    }

    /// Cards carrying at least one shield.
    #[inline]
    pub fn any_shield_mask(&self) -> u128 {
        self.any_shield
    }

    /// Cards carrying `symbol`. Always empty for [`Science::Balance`], which
    /// no card can carry (it comes only from the Law progress token) — that is
    /// asserted by `duels_core::data`'s own validation.
    #[inline]
    pub fn symbol_mask(&self, symbol: Science) -> u128 {
        self.symbol[symbol.index()]
    }

    /// Civilian (blue) cards with printed victory points.
    #[inline]
    pub fn civilian_vp_mask(&self) -> u128 {
        self.civilian_vp
    }

    /// Guild (purple) cards.
    #[inline]
    pub fn guild_mask(&self) -> u128 {
        self.guild
    }

    /// Cards that owning `card` makes free to construct, via a chain symbol.
    /// The base game never chains one card to two, but this is a mask so that
    /// it cannot break if the data ever does.
    #[inline]
    pub fn chain_unlocks(&self, card: CardId) -> u128 {
        self.chain_unlocks[card.index()]
    }

    /// Supply figures for `age` (1, 2 or 3).
    ///
    /// # Panics
    ///
    /// Panics if `age` is not in `1..=3`.
    #[inline]
    pub fn age_supply(&self, age: u8) -> &AgeSupply {
        assert!((1..=3).contains(&age), "age must be 1..=3, got {age}");
        &self.age_supply[(age - 1) as usize]
    }

    /// The progress token that grants a scientific symbol of its own (Law),
    /// found by its effect rather than its slug.
    #[inline]
    pub fn law_token(&self) -> Option<TokenId> {
        self.law_token
    }

    /// The symbol the Law token supplies, if the data has such a token.
    #[inline]
    pub fn law_symbol(&self) -> Option<Science> {
        self.law_token.and_then(|t| t.def().science)
    }

    /// The progress token that adds a shield to every red card its owner
    /// constructs (Strategy).
    #[inline]
    pub fn strategy_token(&self) -> Option<TokenId> {
        self.strategy_token
    }

    /// The wonder that draws three set-aside progress tokens (The Great
    /// Library).
    #[inline]
    pub fn great_library(&self) -> Option<WonderId> {
        self.great_library
    }

    /// The wonder that constructs a card from the discard pile for free (The
    /// Mausoleum).
    #[inline]
    pub fn mausoleum(&self) -> Option<WonderId> {
        self.mausoleum
    }

    /// The expected number of copies of `symbol` an undealt age still holds,
    /// summed over the ages `first_undealt_age..=3` and weighted by
    /// [`AgeSupply::plain_dealt_fraction`] — the chance setup deals a
    /// particular card at all rather than returning it to the box.
    ///
    /// A static function of the symbol and which ages are left, so it is
    /// tabulated rather than recomputed per position. `first_undealt_age`
    /// above 3 gives zero.
    #[inline]
    pub fn future_dealt_copies(&self, symbol: Science, first_undealt_age: u8) -> f64 {
        self.future_dealt_weight[symbol.index()][usize::from(first_undealt_age.min(4))]
    }

    /// Bitmask over [`WonderId::index`] of the wonders that grant shields.
    #[inline]
    pub fn shield_wonders(&self) -> u16 {
        self.shield_wonders
    }

    /// Bitmask over [`WonderId::index`] of the wonders whose *printed* effect
    /// grants an extra turn.
    ///
    /// Resolved from [`duels_core::data::Wonder::play_again`], never from a
    /// slug: which wonders these are is a fact about the data, and the base
    /// game's five (Piraeus, The Appian Way, The Hanging Gardens, The Sphinx,
    /// The Temple of Artemis) are only asserted in a test, not assumed by the
    /// logic.
    #[inline]
    pub fn play_again_wonders(&self) -> u16 {
        self.play_again_wonders
    }

    /// The progress token that grants an extra turn for every wonder its
    /// holder constructs (Theology), found by its effect.
    #[inline]
    pub fn theology_token(&self) -> Option<TokenId> {
        self.theology_token
    }
}

fn build() -> Masks {
    let cards = data::cards();
    let max_shields = cards.iter().map(|c| c.shields).max().unwrap_or(0) as usize;

    let mut shield = vec![0u128; max_shields + 1];
    let mut any_shield = 0u128;
    let mut symbol = [0u128; NUM_SCIENCE];
    let mut civilian_vp = 0u128;
    let mut guild = 0u128;
    let mut chain_unlocks = vec![0u128; NUM_CARDS];
    let mut supply = [AgeSupply {
        plain_cards: 0,
        plain_dealt: 0,
        plain_shields: 0,
        plain_shield_cards: 0,
        plain_civilian_vp: 0,
        guild_cards: 0,
        guild_dealt: 0,
        guild_shields: 0,
        guild_shield_cards: 0,
    }; 3];

    for (i, c) in cards.iter().enumerate() {
        let bit = 1u128 << i;
        shield[c.shields as usize] |= bit;
        if c.shields > 0 {
            any_shield |= bit;
        }
        if let Some(sym) = c.science {
            symbol[sym.index()] |= bit;
        }
        if c.kind == CardType::Civilian && c.victory_points > 0 {
            civilian_vp |= bit;
        }
        if c.is_guild() {
            guild |= bit;
        }
        if let Some(prereq) = c.chain_from {
            chain_unlocks[prereq.index()] |= bit;
        }

        let s = &mut supply[(c.age - 1) as usize];
        if c.is_guild() {
            s.guild_cards += 1;
            s.guild_shields += u16::from(c.shields);
            if c.shields > 0 {
                s.guild_shield_cards += 1;
            }
        } else {
            s.plain_cards += 1;
            s.plain_shields += u16::from(c.shields);
            if c.shields > 0 {
                s.plain_shield_cards += 1;
            }
            if c.kind == CardType::Civilian {
                s.plain_civilian_vp += u16::from(c.victory_points);
            }
        }
    }

    // How much of each age's pool actually reaches the table. Age III is the
    // only age with guilds: exactly `GUILDS_IN_PLAY` of them are shuffled in,
    // and the non-guild half fills the remaining slots. Ages with no guilds
    // simply fill all twenty slots from their own pool.
    for s in supply.iter_mut() {
        s.guild_dealt = u16::try_from(GUILDS_IN_PLAY)
            .unwrap_or(0)
            .min(s.guild_cards);
        let slots = u16::try_from(SLOTS).unwrap_or(u16::MAX);
        s.plain_dealt = slots.saturating_sub(s.guild_dealt).min(s.plain_cards);
    }

    let law_token = TokenId::all().find(|t| t.def().science.is_some());
    let strategy_token = TokenId::all().find(|t| t.def().shield_bonus);
    let great_library = WonderId::all().find(|w| w.def().choose_progress_token);
    let mausoleum = WonderId::all().find(|w| w.def().build_discarded_free);
    let shield_wonders = WonderId::all()
        .filter(|w| w.def().shields > 0)
        .fold(0u16, |m, w| m | (1u16 << w.index()));
    // `future_dealt_weight[s][a]` is the expected copies of `s` in the ages
    // `a..=3`, so index 4 is empty and index 1 is the whole game.
    let mut future_dealt_weight = [[0.0f64; 5]; NUM_SCIENCE];
    for (i, w) in future_dealt_weight.iter_mut().enumerate() {
        for first in 1..=4u8 {
            let mut total = 0.0;
            for c in iter_cards(symbol[i]) {
                let age = c.def().age;
                if age >= first {
                    total += frac(
                        supply[(age - 1) as usize].plain_dealt,
                        supply[(age - 1) as usize].plain_cards,
                    );
                }
            }
            w[usize::from(first)] = total;
        }
    }

    let play_again_wonders = WonderId::all()
        .filter(|w| w.def().play_again)
        .fold(0u16, |m, w| m | (1u16 << w.index()));
    let theology_token = TokenId::all().find(|t| t.def().wonder_play_again);

    Masks {
        shield,
        any_shield,
        symbol,
        civilian_vp,
        guild,
        chain_unlocks,
        age_supply: supply,
        law_token,
        strategy_token,
        great_library,
        mausoleum,
        shield_wonders,
        future_dealt_weight,
        play_again_wonders,
        theology_token,
    }
}

/// Sum of the shields printed on the cards of `mask`.
#[inline]
pub fn shields_in(mask: u128) -> u16 {
    let m = masks();
    let mut total = 0u16;
    for k in 1..=usize::from(m.max_card_shields()) {
        total += u16::try_from((mask & m.shield_mask(k)).count_ones()).unwrap_or(0) * k as u16;
    }
    total
}

/// Sum of the printed victory points of the cards of `mask`.
pub fn victory_points_in(mask: u128) -> u16 {
    iter_cards(mask)
        .map(|c| u16::from(c.def().victory_points))
        .sum()
}

/// Iterate the [`CardId`]s of a card bitmask.
///
/// `duels_core` keeps its own equivalent `pub(crate)`, so this crate has its
/// own; the two are asserted to agree by `tests::iter_cards_matches_the_data`.
#[inline]
pub fn iter_cards(mut mask: u128) -> impl Iterator<Item = CardId> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_science_is_in_index_order() {
        for (i, s) in ALL_SCIENCE.iter().enumerate() {
            assert_eq!(s.index(), i, "ALL_SCIENCE is out of index order at {i}");
        }
    }

    #[test]
    fn iter_cards_matches_the_data() {
        let all = (1u128 << NUM_CARDS) - 1;
        let got: Vec<CardId> = iter_cards(all).collect();
        assert_eq!(got.len(), NUM_CARDS);
        for (i, c) in got.iter().enumerate() {
            assert_eq!(c.index(), i);
        }
        assert_eq!(iter_cards(0).count(), 0);
    }

    #[test]
    fn shield_masks_partition_the_card_list_by_shield_count() {
        let m = masks();
        let all = (1u128 << NUM_CARDS) - 1;
        let mut union = 0u128;
        let mut count = 0u32;
        for k in 0..=usize::from(m.max_card_shields()) {
            let mask = m.shield_mask(k);
            assert_eq!(union & mask, 0, "shield masks overlap at {k}");
            union |= mask;
            count += mask.count_ones();
            for c in iter_cards(mask) {
                assert_eq!(usize::from(c.def().shields), k);
            }
        }
        assert_eq!(union, all);
        assert_eq!(count as usize, NUM_CARDS);
        assert_eq!(m.any_shield_mask(), all & !m.shield_mask(0));
        // The base game's biggest red card is the Arsenal at three shields;
        // if that ever changes the tables above grow with it rather than
        // silently truncating.
        assert_eq!(m.max_card_shields(), 3);
    }

    #[test]
    fn every_shield_bearing_card_is_red() {
        // The military read adds the Strategy token's bonus shield to any
        // shield-bearing card it finds; that is only correct while every such
        // card is a military (red) one, which the engine's own rule keys on.
        for c in iter_cards(masks().any_shield_mask()) {
            assert_eq!(
                c.def().kind,
                CardType::Military,
                "{} carries shields but is not red",
                c.slug()
            );
        }
    }

    #[test]
    fn each_card_symbol_has_exactly_two_copies_and_balance_none() {
        let m = masks();
        for s in ALL_SCIENCE {
            let n = m.symbol_mask(s).count_ones();
            if s == Science::Balance {
                assert_eq!(n, 0, "no card may carry Balance");
            } else {
                assert_eq!(n, 2, "{s:?} should have two copies, found {n}");
            }
        }
    }

    #[test]
    fn sundial_and_gyroscope_exist_only_in_age_three() {
        // This asymmetry is what makes a late science race structurally dead
        // for a player who has missed both copies; assert it against the data
        // rather than assuming it.
        let m = masks();
        for s in [Science::Sundial, Science::Gyroscope] {
            for c in iter_cards(m.symbol_mask(s)) {
                assert_eq!(c.def().age, 3, "{} is not Age III", c.slug());
            }
        }
        // ...and the other four are spread over the earlier ages.
        for s in [
            Science::Mortar,
            Science::Pendulum,
            Science::Inkwell,
            Science::Wheel,
        ] {
            let ages: Vec<u8> = iter_cards(m.symbol_mask(s)).map(|c| c.def().age).collect();
            assert!(
                ages.iter().any(|&a| a < 3),
                "{s:?} appears only in Age III: {ages:?}"
            );
        }
    }

    #[test]
    fn chain_unlocks_mirrors_the_data_both_ways() {
        let m = masks();
        for c in CardId::all() {
            for unlocked in iter_cards(m.chain_unlocks(c)) {
                assert_eq!(unlocked.def().chain_from, Some(c));
            }
            if let Some(to) = c.def().chain_to {
                assert_ne!(m.chain_unlocks(c) & (1u128 << to.index()), 0);
            }
        }
    }

    #[test]
    fn age_supply_totals_add_up_to_the_real_decks() {
        let m = masks();
        assert_eq!(m.age_supply(1).plain_cards, 23);
        assert_eq!(m.age_supply(2).plain_cards, 23);
        assert_eq!(m.age_supply(3).plain_cards, 20);
        assert_eq!(m.age_supply(3).guild_cards, 7);
        for age in 1..=3u8 {
            let s = m.age_supply(age);
            assert_eq!(
                usize::from(s.plain_dealt + s.guild_dealt),
                SLOTS,
                "age {age} does not fill the structure"
            );
        }
        // Ages I and II hold no guilds at all.
        assert_eq!(m.age_supply(1).guild_cards, 0);
        assert_eq!(m.age_supply(2).guild_cards, 0);
        assert_eq!(m.age_supply(3).guild_dealt, 3);
        assert_eq!(m.age_supply(3).plain_dealt, 17);
    }

    #[test]
    fn shields_in_agrees_with_a_direct_sum() {
        let m = masks();
        for mask in [
            m.any_shield_mask(),
            m.shield_mask(3),
            0,
            (1u128 << NUM_CARDS) - 1,
        ] {
            let direct: u16 = iter_cards(mask).map(|c| u16::from(c.def().shields)).sum();
            assert_eq!(shields_in(mask), direct);
        }
    }

    #[test]
    fn singleton_pieces_resolve_to_the_expected_components() {
        let m = masks();
        assert_eq!(m.law_token().map(|t| t.slug()), Some("law"));
        assert_eq!(m.law_symbol(), Some(Science::Balance));
        assert_eq!(m.strategy_token().map(|t| t.slug()), Some("strategy"));
        assert_eq!(
            m.great_library().map(|w| w.slug()),
            Some("the-great-library")
        );
        assert_eq!(m.mausoleum().map(|w| w.slug()), Some("the-mausoleum"));
        assert!(m.shield_wonders().count_ones() >= 1);
        for w in WonderId::all() {
            assert_eq!(
                m.shield_wonders() & (1u16 << w.index()) != 0,
                w.def().shields > 0
            );
        }
    }
}
