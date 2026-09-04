//! The science race read: how close is this player to six distinct symbols,
//! and is the sixth one still physically obtainable?
//!
//! # The slack of one
//!
//! Six *distinct* symbols wins the game outright, and there are **seven**
//! distinct symbols in the data: the six that appear on green cards (twice
//! each) plus [`Science::Balance`], which only the Law progress token
//! supplies. A player holding `k` distinct symbols therefore needs `6 - k`
//! more out of the `7 - k` they do not hold: exactly one spare. That is why
//! [`ScienceRead::dead`] is *not* "some needed symbol is gone" but "fewer than
//! `missing` of the symbols I lack are still obtainable" — losing one symbol
//! is survivable, losing two is not.
//!
//! # What "obtainable" means
//!
//! A green card carrying a symbol is obtainable if it is
//!
//! * face up in the structure and not yet taken, or
//! * in the current age's unknown pool (a face-down slot — or one of the three
//!   cards boxed unseen, which is why this is a *possibility*, not a
//!   certainty), or
//! * in an age that has not been dealt yet, or
//! * in the discard pile *and* this player still holds an unbuilt Mausoleum.
//!
//! It is **not** obtainable once it is in either city, spent under a wonder,
//! or — for a finished age — publicly unaccounted for, which is exactly the
//! set of cards that were returned to the box. Since Sundial and Gyroscope
//! exist only on Age III cards (asserted against the data in
//! `masks::tests::sundial_and_gyroscope_exist_only_in_age_three`), a player who
//! watches both copies of one of them disappear has a structurally dead
//! science race no matter how many other symbols they hold.
//!
//! [`Science::Balance`] follows the token rules instead: the Law token is
//! obtainable if it is on the board (claimable by completing any symbol pair),
//! or if it was set aside at setup and this player still holds an unbuilt
//! Great Library. The Great Library route is genuinely uncertain — it draws
//! three of the set-aside tokens at random — and is reported through
//! [`SymbolAvailability::via_law_great_library`] so a caller can discount it.

use duels_core::data::{self, CardId, Science, TokenId, NUM_SCIENCE};
use duels_core::state::Pending;
use duels_core::{cost, GameState, Player};

use crate::board::{iter_slots, Board};
use crate::masks::{iter_cards, masks, ALL_SCIENCE};

/// Distinct symbols needed to win outright.
pub const SYMBOLS_TO_WIN: u8 = 6;

/// How reachable scientific supremacy is for one player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScienceStatus {
    /// One legal, affordable move completes the sixth distinct symbol.
    Imminent,
    /// Two or fewer symbols missing, at most one of them down to a single
    /// obtainable copy, and none structurally gone.
    Live,
    /// Three or fewer missing and none structurally gone: not realistically
    /// winnable against a denying opponent, but worth real value for the
    /// denial it forces.
    Pressure,
    /// Not a race.
    Closed,
}

/// Where the copies of one symbol still are, from one player's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SymbolAvailability {
    /// How many copies of this symbol the player already holds (0, 1 or 2).
    pub held: u8,
    /// Copies face up in the structure and not yet taken.
    pub face_up: u8,
    /// Copies in the current age's unknown pool.
    pub in_unknown_pool: u8,
    /// Copies in an age that has not been dealt yet.
    pub in_future_age: u8,
    /// Copies in the discard pile, reachable only because the player holds an
    /// unbuilt Mausoleum.
    pub via_mausoleum: u8,
    /// Copies that are gone: in a city, under a wonder, or boxed.
    pub gone: u8,
    /// The Law token is on the board, so this symbol can be claimed by
    /// completing any pair.
    pub via_law_board: bool,
    /// The Law token was set aside and the player still holds an unbuilt Great
    /// Library — a three-of-five draw, not a certainty.
    pub via_law_great_library: bool,
}

impl SymbolAvailability {
    /// How many copies of this symbol the player could still acquire.
    #[inline]
    pub fn obtainable_copies(&self) -> u8 {
        self.face_up
            .saturating_add(self.in_unknown_pool)
            .saturating_add(self.in_future_age)
            .saturating_add(self.via_mausoleum)
            .saturating_add(u8::from(self.via_law_board))
            .saturating_add(u8::from(self.via_law_great_library))
    }

    /// Whether the player could still acquire this symbol at all.
    #[inline]
    pub fn obtainable(&self) -> bool {
        self.obtainable_copies() > 0
    }
}

/// Which of a player's half-pairs could still be completed, and what the board
/// currently pays for completing one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairSetup {
    /// Symbols the player holds exactly once and has not already been paid a
    /// progress token for. Completing any of these claims a token.
    pub candidates: [bool; NUM_SCIENCE],
    /// Of those, the ones an accessible, affordable card would complete right
    /// now.
    pub completable_now: [bool; NUM_SCIENCE],
    /// Slots whose card would complete one of `candidates`.
    pub completing_slots: u32,
    /// The most valuable progress token currently on the board, and its value
    /// to this player in victory points.
    pub best_board_token: Option<(TokenId, f64)>,
    /// Sum of every board token's value to this player: a rough measure of how
    /// much the token row is worth fighting over at all.
    pub board_token_total: f64,
}

impl PairSetup {
    /// How many half-pairs the player is sitting on.
    #[inline]
    pub fn candidate_count(&self) -> u8 {
        u8::try_from(self.candidates.iter().filter(|&&c| c).count()).unwrap_or(u8::MAX)
    }

    /// The value of setting up (and then completing) a pair right now: the
    /// best token on the board, but only if the player actually has a
    /// half-pair to complete.
    #[inline]
    pub fn value(&self) -> f64 {
        if self.candidate_count() == 0 {
            0.0
        } else {
            self.best_board_token.map_or(0.0, |(_, v)| v)
        }
    }
}

/// The science race, read for one player.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScienceRead {
    /// The player this read is about.
    pub player: Player,
    /// Distinct symbols held, counting Balance.
    pub distinct: u8,
    /// Distinct symbols still needed to reach [`SYMBOLS_TO_WIN`].
    pub missing: u8,
    /// Per-symbol availability, indexed by [`Science::index`].
    pub availability: [SymbolAvailability; NUM_SCIENCE],
    /// How many of the symbols the player lacks are still obtainable.
    pub obtainable_missing: u8,
    /// How many of the *card* symbols the player lacks are down to a single
    /// obtainable copy — one denial away from gone. Balance is excluded: it
    /// only ever has one source (the Law token) and is a spare path rather
    /// than a required one, so counting it would swamp the measure.
    pub fragility: u8,
    /// True when fewer than `missing` of the lacked symbols are still
    /// obtainable, i.e. six distinct symbols is now physically impossible.
    pub dead: bool,
    /// The classification.
    pub status: ScienceStatus,
    /// Accessible slots whose card would complete the sixth distinct symbol.
    pub closing_slots: u32,
    /// A pending progress-token choice would complete it (the Law token is on
    /// offer and this player is the one choosing).
    pub closing_via_token: Option<TokenId>,
    /// Accessible slots whose card would add a symbol the player does not hold
    /// at all.
    pub new_symbol_slots: u32,
    /// The pair / progress-token picture.
    pub pair_setup: PairSetup,
}

impl ScienceRead {
    /// Symbols the player does not hold at all.
    pub fn missing_symbols(&self) -> impl Iterator<Item = Science> + '_ {
        ALL_SCIENCE
            .into_iter()
            .filter(move |s| self.availability[s.index()].held == 0)
    }
}

/// Read the science race for `player`.
///
/// Reads only public information, so two determinizations of the same position
/// produce identical results.
pub fn science_read(state: &GameState, player: Player) -> ScienceRead {
    science_read_with(state, player, &Board::of(state))
}

/// [`science_read`] against a [`Board`] the caller already built.
pub fn science_read_with(state: &GameState, player: Player, board: &Board) -> ScienceRead {
    let m = masks();
    let me = state.player(player);
    let held = me.science();
    let distinct = me.distinct_science();
    let missing = SYMBOLS_TO_WIN.saturating_sub(distinct);

    let wonder_slots = state.wonder_slots_left();
    let has_unbuilt = |find: Option<duels_core::data::WonderId>| {
        find.is_some_and(|w| me.owns_wonder(w) && !me.has_built_wonder(w)) && wonder_slots
    };
    let mausoleum_live = (has_unbuilt(m.mausoleum()) && board.discard != 0)
        || (state.pending() == Some(Pending::MausoleumBuild) && state.current_player() == player);
    let great_library_live = has_unbuilt(m.great_library());

    let law = m.law_token();
    let law_on_board = law.is_some_and(|t| state.board_tokens().any(|b| b == t));
    let law_set_aside = law.is_some_and(|t| state.set_aside_tokens().any(|a| a == t));
    let law_in_pending_draw = match (law, state.pending()) {
        (Some(t), Some(Pending::GreatLibraryToken { tokens })) => {
            state.current_player() == player && tokens.contains(&t)
        }
        _ => false,
    };

    // --- per-symbol availability ------------------------------------------
    let mut availability = [SymbolAvailability::default(); NUM_SCIENCE];
    for sym in ALL_SCIENCE {
        let a = &mut availability[sym.index()];
        a.held = held[sym.index()];
        if a.held > 0 {
            // Nothing else about a symbol already held matters to the race.
            continue;
        }
        if m.law_symbol() == Some(sym) {
            a.via_law_board = law_on_board;
            a.via_law_great_library = (law_set_aside && great_library_live) || law_in_pending_draw;
            if !a.obtainable() {
                a.gone = 1;
            }
            continue;
        }
        for card in iter_cards(m.symbol_mask(sym)) {
            let bit = 1u128 << card.index();
            let age = card.def().age;
            if board.face_up & bit != 0 {
                a.face_up += 1;
            } else if age >= board.first_undealt_age {
                // An age whose structure has not been laid out yet. Note that
                // during the wonder draft this includes the *current* age,
                // whose deck is shuffled but not yet dealt.
                a.in_future_age += 1;
            } else if age == board.age && board.unknown_pool & bit != 0 {
                a.in_unknown_pool += 1;
            } else if board.discard & bit != 0 && mausoleum_live {
                a.via_mausoleum += 1;
            } else {
                a.gone += 1;
            }
        }
    }

    let obtainable_missing = u8::try_from(
        ALL_SCIENCE
            .into_iter()
            .filter(|s| availability[s.index()].held == 0 && availability[s.index()].obtainable())
            .count(),
    )
    .unwrap_or(u8::MAX);
    let fragility = u8::try_from(
        ALL_SCIENCE
            .into_iter()
            .filter(|&s| {
                Some(s) != m.law_symbol()
                    && availability[s.index()].held == 0
                    && availability[s.index()].obtainable_copies() == 1
            })
            .count(),
    )
    .unwrap_or(u8::MAX);
    let dead = missing > 0 && obtainable_missing < missing;

    // --- what is takeable right now ---------------------------------------
    let mut closing_slots = 0u32;
    let mut new_symbol_slots = 0u32;
    let mut completing_slots = 0u32;
    let mut completable_now = [false; NUM_SCIENCE];

    let mut candidates = [false; NUM_SCIENCE];
    for sym in ALL_SCIENCE {
        candidates[sym.index()] = held[sym.index()] == 1
            && Some(sym) != m.law_symbol()
            && !me.pairs_awarded().any(|s| s == sym);
    }

    for slot in iter_slots(board.accessible) {
        let Some(card) = board.slot_card[slot as usize] else {
            continue;
        };
        let Some(sym) = card.def().science else {
            continue;
        };
        if !cost::card_cost(state, player, card).affordable_by(state, player) {
            continue;
        }
        if held[sym.index()] == 0 {
            new_symbol_slots |= 1u32 << slot;
            if missing == 1 {
                closing_slots |= 1u32 << slot;
            }
        }
        if candidates[sym.index()] {
            completing_slots |= 1u32 << slot;
            completable_now[sym.index()] = true;
        }
    }

    // A pending token choice can complete the set too: the Law token supplies
    // a symbol of its own.
    let closing_via_token = if missing == 1 {
        match (law, state.pending()) {
            (Some(t), Some(Pending::ProgressToken))
                if state.current_player() == player && law_on_board =>
            {
                Some(t)
            }
            (Some(t), Some(Pending::GreatLibraryToken { tokens }))
                if state.current_player() == player && tokens.contains(&t) =>
            {
                Some(t)
            }
            _ => None,
        }
    } else {
        None
    };

    // --- board token values ------------------------------------------------
    let mut best_board_token: Option<(TokenId, f64)> = None;
    let mut board_token_total = 0.0;
    for token in state.board_tokens() {
        let v = token_value(state, player, token);
        board_token_total += v;
        if best_board_token.is_none_or(|(_, best)| v > best) {
            best_board_token = Some((token, v));
        }
    }

    let pair_setup = PairSetup {
        candidates,
        completable_now,
        completing_slots,
        best_board_token,
        board_token_total,
    };

    let status = if missing == 0 || closing_slots != 0 || closing_via_token.is_some() {
        ScienceStatus::Imminent
    } else if !dead && missing <= 2 && fragility <= 1 {
        ScienceStatus::Live
    } else if !dead && missing <= 3 {
        ScienceStatus::Pressure
    } else {
        ScienceStatus::Closed
    };

    ScienceRead {
        player,
        distinct,
        missing,
        availability,
        obtainable_missing,
        fragility,
        dead,
        status,
        closing_slots,
        closing_via_token,
        new_symbol_slots,
        pair_setup,
    }
}

/// Named weights for the parts of a progress token's worth that are *not*
/// printed as victory points.
///
/// The victory-point and coin parts of [`token_value`] are read straight out
/// of [`duels_core::data`] and need no tuning. The rules effects — a cost
/// rebate, redirected trade payments, a bonus shield per red card, an extra
/// turn per wonder, coins on a chain build — have no printed point value at
/// all, so a number has to be invented for them. Keeping those numbers here,
/// named, means the judgement calls are visible and tunable rather than
/// scattered through the code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenValueWeights {
    /// Points per coin, matching the real `floor(coins / 3)` scoring rule.
    pub coins_to_vp: f64,
    /// Worth of the extra distinct scientific symbol the Law token supplies.
    /// Large, because for a player at five symbols it *is* the game.
    pub science_symbol: f64,
    /// Worth of a two-unit cost rebate on wonders or blue cards
    /// (Architecture, Masonry).
    pub discount: f64,
    /// Worth of receiving the opponent's trade payments (Economy).
    pub trade_redirect: f64,
    /// Worth of a bonus shield on every red card built afterwards
    /// (Strategy).
    pub shield_bonus: f64,
    /// Worth of an extra turn per wonder constructed (Theology).
    pub wonder_play_again: f64,
    /// Worth per coin of a chain-build bonus (Urbanism), discounted because it
    /// only pays when a chain build actually happens.
    pub chain_build_coin: f64,
}

impl Default for TokenValueWeights {
    fn default() -> Self {
        Self {
            coins_to_vp: 1.0 / 3.0,
            science_symbol: 4.0,
            discount: 3.0,
            trade_redirect: 2.0,
            shield_bonus: 3.0,
            wonder_play_again: 3.0,
            chain_build_coin: 0.25,
        }
    }
}

/// What a progress token is worth to `player`, in rough victory points.
///
/// The printed parts (flat victory points, Mathematics' per-token bonus, the
/// one-off coins) come from [`duels_core::data`]; the rules effects are priced
/// by [`TokenValueWeights`].
pub fn token_value(state: &GameState, player: Player, token: TokenId) -> f64 {
    token_value_with(state, player, token, &TokenValueWeights::default())
}

/// [`token_value`] with explicit weights.
pub fn token_value_with(
    state: &GameState,
    player: Player,
    token: TokenId,
    w: &TokenValueWeights,
) -> f64 {
    let def = token.def();
    let me = state.player(player);
    // Mathematics scores per token *including itself*, and it re-prices every
    // token the player already holds, so taking it is worth its own rate times
    // the resulting count.
    let after = f64::from(me.token_count()) + 1.0;
    let mut v = f64::from(def.victory_points) + f64::from(def.vp_per_token) * after;
    // ...and if the player already owns Mathematics, any further token is
    // worth its rate again.
    for owned in me.tokens() {
        v += f64::from(owned.def().vp_per_token);
    }
    v += f64::from(def.coins) * w.coins_to_vp;
    if def.science.is_some() {
        v += w.science_symbol;
    }
    if def.discount.is_some() {
        v += w.discount;
    }
    if def.gain_trade_costs {
        v += w.trade_redirect;
    }
    if def.shield_bonus {
        v += w.shield_bonus;
    }
    if def.wonder_play_again {
        v += w.wonder_play_again;
    }
    v += f64::from(def.chain_build_coins) * w.chain_build_coin;
    v
}

/// Whether `player` holds a progress token satisfying `f`.
pub fn holds_token_with(
    state: &GameState,
    player: Player,
    f: impl Fn(&data::ProgressToken) -> bool,
) -> bool {
    state.player(player).tokens().any(|t| f(t.def()))
}

/// Every card carrying `symbol`, for callers that want to explain a read.
pub fn symbol_cards(symbol: Science) -> impl Iterator<Item = CardId> {
    iter_cards(masks().symbol_mask(symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::testing::StateBuilder;

    #[test]
    fn a_fresh_position_needs_all_six_symbols_and_none_are_gone() {
        let st = StateBuilder::new().age(1).build();
        let r = science_read(&st, Player::One);
        assert_eq!(r.distinct, 0);
        assert_eq!(r.missing, 6);
        assert!(!r.dead);
        // Seven symbols are lacked and all seven have a route... except
        // Balance, which needs the Law token to be somewhere reachable; the
        // default builder puts no tokens anywhere.
        assert_eq!(r.obtainable_missing, 6);
        assert_eq!(r.status, ScienceStatus::Closed, "six missing is not a race");
    }

    #[test]
    fn token_value_reads_the_printed_points_from_the_data() {
        let st = StateBuilder::new().build();
        let philosophy = TokenId::from_slug("philosophy").unwrap();
        assert_eq!(philosophy.def().victory_points, 7);
        assert!((token_value(&st, Player::One, philosophy) - 7.0).abs() < 1e-9);

        // Mathematics scores three per token including itself.
        let maths = TokenId::from_slug("mathematics").unwrap();
        assert!((token_value(&st, Player::One, maths) - 3.0).abs() < 1e-9);
        let with_two = StateBuilder::new()
            .tokens(Player::One, &["philosophy", "agriculture"])
            .build();
        assert!((token_value(&with_two, Player::One, maths) - 9.0).abs() < 1e-9);
    }
}
