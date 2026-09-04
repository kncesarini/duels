//! The science race read: how close is this player to six distinct symbols,
//! how likely are they to get there, and what would it cost to stop them?
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
//! # What "obtainable" means, and the one Mausoleum
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
//! That last route is the one place the obvious per-symbol arithmetic is
//! wrong, and it used to be wrong here. **The Mausoleum retrieves exactly one
//! card.** Two different missing symbols both sitting in the discard pile with
//! one unbuilt Mausoleum are not two routes but one, so
//! [`ScienceRead::obtainable_missing`] counts the symbols with a route that
//! does *not* need the Mausoleum, and then adds at most one for all the
//! symbols that do:
//!
//! ```text
//! O = { missing s : obtainable without the Mausoleum }
//! X = { missing s : obtainable only via the Mausoleum }
//! obtainable_missing = |O| + min(1, |X|)
//! dead = missing > 0 && obtainable_missing < missing
//! ```
//!
//! [`Science::Balance`] follows the token rules instead: the Law token is
//! obtainable if it is on the board (claimable by completing any symbol pair),
//! or if it was set aside at setup and this player still holds an unbuilt
//! Great Library. Which of those two it is matters enormously and is *not*
//! interchangeable — a Law token on the board is a symbol you take by
//! completing a pair you may already be half-way to, while a set-aside Law
//! token is a three-of-five draw you only get to make if you build a
//! particular wonder. If it is on neither, Balance is simply not in the game
//! for you.
//!
//! # The magnitude
//!
//! [`ScienceRead::magnitude`] is the one number the policy layer actually
//! uses: the probability, from public information only, that this player
//! completes six symbols if both sides play on. It is built from three
//! ingredients per missing symbol — how many copies are still out there
//! ([`SymbolModel::copies`], discounted for the ones that might have been
//! boxed), what it would cost the *defender* to personally take every copy
//! away ([`SymbolModel::kill_cost`]), and whether the threat-holder can secure
//! one of them on their very next turn, extra turns included. The discrete
//! [`ScienceStatus`] is then just a label read off the magnitude, not a
//! separate judgement.

use duels_core::data::{self, CardId, Science, TokenId, NUM_SCIENCE};
use duels_core::state::Pending;
use duels_core::{GameState, Player};

use crate::board::{iter_slots, Board};
use crate::context::Context;
use crate::masks::{iter_cards, masks, ALL_SCIENCE};
use crate::prices::Prices;
use crate::tempo::{Tempo, ThreatWeights, MAX_CHAIN};

/// Distinct symbols needed to win outright.
pub const SYMBOLS_TO_WIN: u8 = 6;

/// How reachable scientific supremacy is for one player.
///
/// Purely a label on [`ScienceRead::magnitude`], for logs and test
/// assertions. Nothing in the weighting math branches on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScienceStatus {
    /// Certain if unopposed: the sixth symbol is secured on this player's very
    /// next turn (`magnitude == 1`).
    Imminent,
    /// A real race: `magnitude >= ThreatWeights::live_threshold`.
    Live,
    /// Not realistically winnable against a denying opponent, but worth real
    /// value for the denial it forces:
    /// `pressure_threshold <= magnitude < live_threshold`.
    Pressure,
    /// Not a race, or physically dead.
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

    /// Whether this symbol has a route that does **not** go through the
    /// Mausoleum — the `NonMaus(s)` of the module docs.
    #[inline]
    pub fn obtainable_without_mausoleum(&self) -> bool {
        self.face_up > 0
            || self.in_unknown_pool > 0
            || self.in_future_age > 0
            || self.via_law_board
            || self.via_law_great_library
    }

    /// Whether the *only* route left is one the opponent cannot interfere
    /// with: a Mausoleum retrieval from the discard pile, or the Great
    /// Library's draw from the set-aside tokens.
    ///
    /// Nothing an opponent does on their turn can take either away, which is
    /// why [`SymbolModel::kill_cost`] reports these as infinite.
    #[inline]
    pub fn undeniable_route_only(&self) -> bool {
        self.obtainable()
            && self.face_up == 0
            && self.in_unknown_pool == 0
            && self.in_future_age == 0
            && !self.via_law_board
    }
}

/// Which of a player's half-pairs could still be completed, and what the board
/// currently pays for completing one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairSetup {
    /// Symbols the player holds exactly once and has not already been paid a
    /// progress token for. Completing any of these claims a token.
    pub candidates: [bool; NUM_SCIENCE],
    /// Of those, the ones whose second copy is still physically obtainable
    /// somewhere — the pair is not merely unclaimed but actually completable.
    pub completable_ever: [bool; NUM_SCIENCE],
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

    /// Whether the player has a half-pair they could actually complete — the
    /// `p_law` condition of the threat model, and also what makes the Law
    /// token on the board cheap for a *defender* to take first.
    #[inline]
    pub fn has_live_half_pair(&self) -> bool {
        self.completable_ever.iter().any(|&c| c)
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

/// Everything the magnitude model knows about one symbol, split by route so
/// that a single action's effect on it can be applied without rebuilding the
/// read.
///
/// The weights (`p_hidden`, `p_dealt`, `p_maus`, `p_law`, `p_gl`) are already
/// folded in: `hidden_cards` is a *weighted* copy count, not a card count.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SymbolModel {
    /// Untaken face-up copies in the structure. Certain, and deniable.
    pub face_up: f64,
    /// Copies in the current age's unknown pool weighted by the chance a
    /// face-down slot holds one rather than the box, plus copies in undealt
    /// ages weighted by the chance setup deals them at all.
    pub hidden_cards: f64,
    /// Whether any unknown-pool or undealt-age copy exists at all. Kept
    /// separately because `hidden_cards` can be a small fraction and "a
    /// fractional route" is still a route.
    pub hidden_cards_any: bool,
    /// Copies in the discard pile, weighted by whether the Mausoleum is
    /// affordable right now. Zero unless the player owns an unbuilt Mausoleum.
    pub mausoleum: f64,
    /// The Law token is on the board (only ever set for [`Science::Balance`]).
    pub law_board: bool,
    /// The Law token is set aside and only an unbuilt Great Library can reach
    /// it.
    pub law_great_library: bool,
    /// The unknown-pool and undealt-age part of what a defender would have to
    /// take to deny this symbol.
    pub kill_hidden: f64,
    /// Slots in `R_T` — the ones the threat-holder can reach on their very
    /// next turn, extra turns included — whose card carries this symbol.
    pub reachable_slots: u32,
}

impl SymbolModel {
    /// `c_s`: how many copies of this symbol the player can expect to still be
    /// able to get, across every route, each discounted by its uncertainty.
    ///
    /// The two token routes are weighted by figures that are the same for
    /// every symbol, so they live on [`SciModel`] and are passed in;
    /// [`SciModel::copies`] is the convenient form.
    #[inline]
    pub fn copies_with(&self, p_law: f64, p_gl: f64) -> f64 {
        self.face_up
            + self.hidden_cards
            + self.mausoleum
            + if self.law_board { p_law } else { 0.0 }
            + if self.law_great_library { p_gl } else { 0.0 }
    }

    /// Whether a route exists that does not need the Mausoleum.
    #[inline]
    pub fn non_mausoleum(&self) -> bool {
        self.face_up > 0.0 || self.hidden_cards_any || self.law_board || self.law_great_library
    }

    /// Whether the only route is a Mausoleum retrieval.
    #[inline]
    pub fn mausoleum_only(&self) -> bool {
        !self.non_mausoleum() && self.mausoleum > 0.0
    }

    /// Whether any route at all remains.
    #[inline]
    pub fn obtainable(&self) -> bool {
        self.non_mausoleum() || self.mausoleum > 0.0
    }

    /// `kill_s`: how many of the defender's own turns it would take to
    /// personally remove this symbol from the threat-holder's options.
    ///
    /// Infinite when the defender simply cannot: a card in the discard pile
    /// behind a Mausoleum, or a token in the set-aside pile behind a Great
    /// Library, is out of their reach entirely. The Law token on the board is
    /// one turn if the defender has a pair of their own to complete and three
    /// if they would have to build one first.
    #[inline]
    pub fn kill_cost(&self, defender_has_half_pair: bool) -> f64 {
        if self.law_board {
            return if defender_has_half_pair { 1.0 } else { 3.0 };
        }
        if self.face_up <= 0.0 && !self.hidden_cards_any {
            return f64::INFINITY;
        }
        self.face_up + self.kill_hidden
    }

    /// Take one face-up copy off the table: the defender built or discarded
    /// it.
    #[inline]
    fn take_face_up(&mut self) {
        self.face_up = (self.face_up - 1.0).max(0.0);
        self.kill_hidden = self.kill_hidden.max(0.0);
    }
}

/// The inputs [`SciModel::magnitude`] consumes, and the only thing an
/// incremental "what if the defender played this?" needs to touch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SciModel {
    /// Per-symbol routes, indexed by [`Science::index`].
    pub symbols: [SymbolModel; NUM_SCIENCE],
    /// Copies of each symbol the player holds.
    pub held: [u8; NUM_SCIENCE],
    /// Distinct symbols still needed.
    pub missing: u8,
    /// Whether six distinct symbols is already physically impossible.
    pub dead: bool,
    /// The threat-holder's remaining decision budget.
    pub decisions_left_eff: f64,
    /// The defender's share of the remaining decisions — the probability they
    /// win any one contested card.
    pub share_defender: f64,
    /// Whether the defender could complete a symbol pair of their own, and so
    /// claim the Law token off the board in a single turn.
    pub defender_has_half_pair: bool,
    /// Weight on a Law-token-on-the-board route: 1 for a player with a live
    /// half-pair to claim it with, [`ThreatWeights::p_law_from_scratch`]
    /// otherwise.
    pub p_law: f64,
    /// Weight on a Law-via-Great-Library route: the three-of-`n` draw, times a
    /// discount when the wonder is not affordable yet.
    pub p_gl: f64,
}

/// What [`SciModel::magnitude`] worked out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SciMagnitude {
    /// `M_sci`: the probability of completing six symbols, in `0.0..=1.0`.
    pub value: f64,
    /// The symbol the player secures on their very next turn, if any. At most
    /// one: one turn, one card.
    pub secured: Option<Science>,
    /// The product of `min(1, c_s)` over the symbols still needed after
    /// securing: how much of the race is a supply problem rather than a
    /// contest.
    pub surface: f64,
    /// The probability the defender takes away enough symbols to kill it.
    pub p_stop: f64,
    /// Spare routes after securing: `obtainable_missing - missing`, so zero
    /// means every remaining route is load-bearing.
    pub slack: i8,
}

impl SciModel {
    /// `c_s` for one symbol.
    #[inline]
    pub fn copies(&self, symbol: Science) -> f64 {
        self.symbols[symbol.index()].copies_with(self.p_law, self.p_gl)
    }

    /// `M_sci`: how likely this player is to reach six distinct symbols.
    pub fn magnitude(&self) -> SciMagnitude {
        let certain = |secured| SciMagnitude {
            value: 1.0,
            secured,
            surface: 1.0,
            p_stop: 0.0,
            slack: 0,
        };
        let hopeless = |secured| SciMagnitude {
            value: 0.0,
            secured,
            surface: 0.0,
            p_stop: 1.0,
            slack: -1,
        };

        if self.missing == 0 {
            return certain(None);
        }
        if self.dead || f64::from(self.missing) > self.decisions_left_eff {
            return hopeless(None);
        }

        // One turn secures at most one symbol, and the one worth securing is
        // the one that would otherwise be hardest to get back.
        let mut secured: Option<Science> = None;
        let mut secured_copies = f64::INFINITY;
        for sym in ALL_SCIENCE {
            let i = sym.index();
            if self.held[i] > 0 || self.symbols[i].reachable_slots == 0 {
                continue;
            }
            let c = self.symbols[i].copies_with(self.p_law, self.p_gl);
            if c < secured_copies {
                secured_copies = c;
                secured = Some(sym);
            }
        }

        let remaining = self.missing - u8::from(secured.is_some());
        if remaining == 0 {
            return certain(secured);
        }

        // Recount the routes over what is left, respecting the one-Mausoleum
        // rule: every symbol that needs the Mausoleum is competing for the
        // same single retrieval.
        let mut non_maus = 0u8;
        let mut maus_only = 0u8;
        let mut kills = [0.0f64; NUM_SCIENCE];
        let mut kill_n = 0usize;
        let mut supply = [0.0f64; NUM_SCIENCE];
        let mut supply_n = 0usize;
        for sym in ALL_SCIENCE {
            let i = sym.index();
            if self.held[i] > 0 || secured == Some(sym) {
                continue;
            }
            let s = &self.symbols[i];
            if s.non_mausoleum() {
                non_maus += 1;
            } else if s.mausoleum_only() {
                maus_only += 1;
            }
            supply[supply_n] = s.copies_with(self.p_law, self.p_gl).min(1.0);
            supply_n += 1;
            if s.obtainable() {
                let k = s.kill_cost(self.defender_has_half_pair);
                kills[kill_n] = if k.is_finite() {
                    self.share_defender.powf(k)
                } else {
                    0.0
                };
                kill_n += 1;
            }
        }
        let obtainable = non_maus + maus_only.min(1);
        let slack = i16::from(obtainable) - i16::from(remaining);
        if slack < 0 {
            return hopeless(secured);
        }

        // The player will chase the `remaining` best-supplied symbols, and a
        // symbol with an expected two thirds of a copy left is two thirds of a
        // race.
        supply[..supply_n]
            .sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let surface: f64 = supply[..supply_n.min(usize::from(remaining))]
            .iter()
            .product();

        // The defender has to take away enough symbols to push the player
        // below `remaining` routes, which means `slack + 1` of them.
        let k = usize::try_from(slack + 1).unwrap_or(usize::MAX);
        let p_stop = p_at_least_k(&kills[..kill_n], k);

        SciMagnitude {
            value: (surface * (1.0 - p_stop)).clamp(0.0, 1.0),
            secured,
            surface,
            p_stop,
            slack: i8::try_from(slack).unwrap_or(i8::MAX),
        }
    }

    /// The same model after the card in `slot` leaves the structure, whatever
    /// symbol (if any) it carried.
    ///
    /// The slot stops being reachable either way; a symbol the player still
    /// lacks also loses one face-up copy, which is the whole point of taking
    /// it.
    pub fn after_slot_taken(&self, slot: u8, symbol: Option<Science>) -> SciModel {
        let mut out = *self;
        for s in out.symbols.iter_mut() {
            s.reachable_slots &= !(1u32 << slot);
        }
        if let Some(sym) = symbol {
            if out.held[sym.index()] == 0 {
                out.symbols[sym.index()].take_face_up();
                out.recount();
            }
        }
        out
    }

    /// The same model after a card is retrieved from the discard pile, so the
    /// one Mausoleum retrieval can no longer be spent on it.
    pub fn after_discard_taken(&self, symbol: Science) -> SciModel {
        let mut out = *self;
        let s = &mut out.symbols[symbol.index()];
        if s.mausoleum > 0.0 {
            // One copy's worth of the weighted route, floored at nothing.
            let per = if s.mausoleum > 1.0 {
                s.mausoleum / (s.mausoleum.ceil()).max(1.0)
            } else {
                s.mausoleum
            };
            s.mausoleum = (s.mausoleum - per).max(0.0);
        }
        out.recount();
        out
    }

    /// The same model after the defender claims the Law token off the board.
    pub fn after_law_taken(&self) -> SciModel {
        let mut out = *self;
        for s in out.symbols.iter_mut() {
            s.law_board = false;
        }
        out.recount();
        out
    }

    /// The same model with `slot` newly reachable, because the defender's move
    /// uncovered it.
    pub fn with_reachable(&self, symbol: Science, slot: u8) -> SciModel {
        let mut out = *self;
        if out.held[symbol.index()] == 0 {
            out.symbols[symbol.index()].reachable_slots |= 1u32 << slot;
        }
        out
    }

    /// Re-derive [`SciModel::dead`] after a route was removed.
    fn recount(&mut self) {
        if self.missing == 0 {
            self.dead = false;
            return;
        }
        let mut non_maus = 0u8;
        let mut maus_only = 0u8;
        for sym in ALL_SCIENCE {
            let i = sym.index();
            if self.held[i] > 0 {
                continue;
            }
            let s = &self.symbols[i];
            if s.non_mausoleum() {
                non_maus += 1;
            } else if s.mausoleum_only() {
                maus_only += 1;
            }
        }
        self.dead = non_maus + maus_only.min(1) < self.missing;
    }
}

/// The probability that at least `k` of a set of independent events happen.
///
/// A textbook Poisson-binomial DP. There are never more than
/// [`NUM_SCIENCE`] events, so this is a handful of multiplications and is not
/// worth approximating.
fn p_at_least_k(ps: &[f64], k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > ps.len() {
        return 0.0;
    }
    let mut dp = [0.0f64; NUM_SCIENCE + 1];
    dp[0] = 1.0;
    let mut n = 0usize;
    for &p in ps {
        n += 1;
        for j in (1..=n).rev() {
            dp[j] = dp[j] * (1.0 - p) + dp[j - 1] * p;
        }
        dp[0] *= 1.0 - p;
    }
    dp[k..=n].iter().sum::<f64>().clamp(0.0, 1.0)
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
    /// How many of the symbols the player lacks are still obtainable, counting
    /// every symbol that needs the Mausoleum as **one** between them: the
    /// `|O| + min(1, |X|)` of the module docs.
    pub obtainable_missing: u8,
    /// For each symbol the player holds exactly once, whether the second copy
    /// is still obtainable — so whether the pair (and the progress token it
    /// claims, and with it the Law token) is still live.
    pub second_copy_obtainable: [bool; NUM_SCIENCE],
    /// For each missing symbol, whether its only remaining route is one the
    /// opponent cannot take away: a Mausoleum retrieval or a Great Library
    /// draw.
    pub undeniable_route: [bool; NUM_SCIENCE],
    /// How many of the *card* symbols the player lacks are down to a single
    /// obtainable copy — one denial away from gone. Balance is excluded: it
    /// only ever has one source (the Law token) and is a spare path rather
    /// than a required one, so counting it would swamp the measure.
    pub fragility: u8,
    /// True when fewer than `missing` of the lacked symbols are still
    /// obtainable, i.e. six distinct symbols is now physically impossible.
    pub dead: bool,
    /// `M_sci`: the probability this player completes six symbols.
    pub magnitude: f64,
    /// The rest of what the magnitude model worked out, for diagnostics.
    pub detail: SciMagnitude,
    /// The magnitude model itself, so a caller can price one action against it
    /// without rebuilding the read.
    pub model: SciModel,
    /// The label on `magnitude`.
    pub status: ScienceStatus,
    /// Slots the player can take on their very next turn, extra turns
    /// included: `R_T`.
    pub reachable_slots: u32,
    /// Revealed slots — accessible or not — whose card carries a symbol this
    /// player does not hold at all.
    ///
    /// The denial channel's fast path: a move that touches none of these and
    /// none of `reachable_missing_slots` cannot change this player's science
    /// magnitude, and does not have to be priced.
    pub missing_symbol_slots: u32,
    /// The `reachable_slots` that carry a symbol this player still needs.
    pub reachable_missing_slots: u32,
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

    /// `c_s` for one symbol: expected obtainable copies.
    #[inline]
    pub fn copies(&self, symbol: Science) -> f64 {
        self.model.copies(symbol)
    }

    /// `kill_s` for one symbol: what it would cost the opponent to deny it.
    #[inline]
    pub fn kill_cost(&self, symbol: Science) -> f64 {
        self.model.symbols[symbol.index()].kill_cost(self.model.defender_has_half_pair)
    }
}

/// The half of a science read that does not depend on the *other* player's
/// position, computed once per player by [`Context`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SciBase {
    pub held: [u8; NUM_SCIENCE],
    pub distinct: u8,
    pub missing: u8,
    pub availability: [SymbolAvailability; NUM_SCIENCE],
    pub candidates: [bool; NUM_SCIENCE],
    pub completable_ever: [bool; NUM_SCIENCE],
    pub has_live_half_pair: bool,
    pub obtainable_missing: u8,
    pub dead: bool,
    pub mausoleum_affordable: bool,
    pub great_library_affordable: bool,
    pub law_on_board: bool,
}

impl SciBase {
    /// Read the position-only half of the science race for `player`.
    pub(crate) fn of(state: &GameState, player: Player, board: &Board, prices: &Prices) -> SciBase {
        let m = masks();
        let me = state.player(player);
        let held = me.science();
        let distinct = me.distinct_science();
        let missing = SYMBOLS_TO_WIN.saturating_sub(distinct);

        let wonder_slots = state.wonder_slots_left();
        let unbuilt = |find: Option<duels_core::data::WonderId>| -> Option<data::WonderId> {
            find.filter(|&w| me.owns_wonder(w) && !me.has_built_wonder(w) && wonder_slots)
        };
        let mausoleum = unbuilt(m.mausoleum());
        let great_library = unbuilt(m.great_library());
        let mausoleum_live = (mausoleum.is_some() && board.discard != 0)
            || (state.pending() == Some(Pending::MausoleumBuild)
                && state.current_player() == player);
        let great_library_live = great_library.is_some();
        let affordable =
            |w: Option<data::WonderId>| -> bool { w.is_some_and(|w| prices.can_afford_wonder(w)) };

        let law = m.law_token();
        let law_on_board = law.is_some_and(|t| state.board_tokens().any(|b| b == t));
        let law_set_aside = law.is_some_and(|t| state.set_aside_tokens().any(|a| a == t));
        let law_in_pending_draw = match (law, state.pending()) {
            (Some(t), Some(Pending::GreatLibraryToken { tokens })) => {
                state.current_player() == player && tokens.contains(&t)
            }
            _ => false,
        };

        let mut availability = [SymbolAvailability::default(); NUM_SCIENCE];
        for sym in ALL_SCIENCE {
            let a = &mut availability[sym.index()];
            a.held = held[sym.index()];
            if m.law_symbol() == Some(sym) {
                if a.held == 0 {
                    a.via_law_board = law_on_board;
                    a.via_law_great_library =
                        (law_set_aside && great_library_live) || law_in_pending_draw;
                    if !a.obtainable() {
                        a.gone = 1;
                    }
                }
                continue;
            }
            // Note this runs for symbols the player already holds too: the
            // magnitude model needs to know whether the *second* copy of a
            // half-pair is still out there, which is what makes the Law token
            // on the board claimable.
            for card in iter_cards(m.symbol_mask(sym)) {
                let bit = 1u128 << card.index();
                let age = card.def().age;
                // Publicly placed cards are resolved first, so that a card
                // whose whereabouts are known can never also be counted as
                // "still to be dealt". Legal play cannot produce a card of an
                // undealt age in a city, but the read should not depend on
                // that.
                if board.face_up & bit != 0 {
                    a.face_up += 1;
                } else if board.discard & bit != 0 {
                    // The Mausoleum takes any one card from the discard pile,
                    // whatever age it belongs to.
                    if mausoleum_live {
                        a.via_mausoleum += 1;
                    } else {
                        a.gone += 1;
                    }
                } else if (board.in_city | board.fodder) & bit != 0 {
                    a.gone += 1;
                } else if age >= board.first_undealt_age {
                    a.in_future_age += 1;
                } else if age == board.age && board.unknown_pool & bit != 0 {
                    a.in_unknown_pool += 1;
                } else {
                    a.gone += 1;
                }
            }
        }

        // `|O| + min(1, |X|)`: the one Mausoleum retrieval is shared.
        let mut non_maus = 0u8;
        let mut maus_only = 0u8;
        for sym in ALL_SCIENCE {
            let a = &availability[sym.index()];
            if a.held > 0 {
                continue;
            }
            if a.obtainable_without_mausoleum() {
                non_maus += 1;
            } else if a.via_mausoleum > 0 {
                maus_only += 1;
            }
        }
        let obtainable_missing = non_maus + maus_only.min(1);
        let dead = missing > 0 && obtainable_missing < missing;

        // One pass over the awarded pairs rather than one per symbol.
        let mut awarded = 0u8;
        for sym in me.pairs_awarded() {
            awarded |= 1u8 << sym.index();
        }
        let mut candidates = [false; NUM_SCIENCE];
        let mut completable_ever = [false; NUM_SCIENCE];
        for sym in ALL_SCIENCE {
            let i = sym.index();
            candidates[i] =
                held[i] == 1 && Some(sym) != m.law_symbol() && awarded & (1u8 << i) == 0;
            // The second copy of a held symbol: `availability` counted both
            // copies' whereabouts, and one of them is in this player's city,
            // so anything it found is the other one.
            completable_ever[i] = candidates[i] && availability[i].obtainable();
        }

        SciBase {
            held,
            distinct,
            missing,
            availability,
            candidates,
            completable_ever,
            has_live_half_pair: completable_ever.iter().any(|&c| c),
            obtainable_missing,
            dead,
            mausoleum_affordable: affordable(mausoleum)
                || (state.pending() == Some(Pending::MausoleumBuild)
                    && state.current_player() == player),
            great_library_affordable: affordable(great_library) || law_in_pending_draw,
            law_on_board,
        }
    }
}

/// Read the science race for `player`.
///
/// Reads only public information, so two determinizations of the same position
/// produce identical results.
pub fn science_read(state: &GameState, player: Player) -> ScienceRead {
    science_read_with(state, player, &Context::of(state))
}

/// [`science_read`] against a [`Context`] the caller already built.
pub fn science_read_with(state: &GameState, player: Player, ctx: &Context) -> ScienceRead {
    let m = masks();
    let board = &ctx.board;
    let w = &ctx.weights;
    let base = ctx.science_base(player);
    let opponent = ctx.science_base(player.other());
    let tempo = ctx.tempo(player);
    let prices = ctx.prices(player);
    let held = base.held;

    // --- the uncertainty weights -------------------------------------------
    // A named card of this age that is not publicly placed sits behind one of
    // the face-down slots with this probability; the rest of the pool was
    // boxed at setup.
    let p_hidden = ctx.expected.p_hidden;
    let p_maus = if base.mausoleum_affordable {
        1.0
    } else {
        w.maus_unaffordable
    };
    let set_aside = u32::try_from(state.set_aside_tokens().count()).unwrap_or(0);
    let p_gl = if set_aside == 0 {
        0.0
    } else {
        (w.great_library_draw / f64::from(set_aside)).min(1.0)
            * if base.great_library_affordable {
                1.0
            } else {
                w.great_library_unaffordable
            }
    };
    let p_law = if base.has_live_half_pair {
        1.0
    } else {
        w.p_law_from_scratch
    };

    // --- R_T: what this player can reach on their very next turn -----------
    let (reachable_slots, reachable_by_symbol) = reachable(board, prices, tempo);

    // --- the per-symbol model ----------------------------------------------
    let mut symbols = [SymbolModel::default(); NUM_SCIENCE];
    for sym in ALL_SCIENCE {
        let i = sym.index();
        let a = &base.availability[i];
        let s = &mut symbols[i];
        s.reachable_slots = reachable_by_symbol[i];
        if held[i] > 0 && Some(sym) == m.law_symbol() {
            continue;
        }
        s.face_up = f64::from(a.face_up);
        s.law_board = a.via_law_board;
        s.law_great_library = a.via_law_great_library;
        s.mausoleum = f64::from(a.via_mausoleum) * p_maus;
        s.hidden_cards_any = a.in_unknown_pool > 0 || a.in_future_age > 0;

        // A card in this age's unknown pool is behind a face-down slot with
        // probability `p_hidden`. A card in an age not yet dealt reaches the
        // table with the probability setup gives it — 20 of 23 for Ages I and
        // II, 17 of 20 for Age III — which is a fact about the decks rather
        // than about the position, and so is tabulated per symbol.
        let mut hidden = f64::from(a.in_unknown_pool) * p_hidden;
        if a.in_future_age > 0 {
            hidden += m.future_dealt_copies(sym, board.first_undealt_age);
        }
        s.hidden_cards = hidden;
        // Denying a symbol means taking every copy the defender can reach: the
        // face-up ones for certain, the hidden ones in expectation.
        s.kill_hidden = hidden;
    }

    // --- what is takeable right now ----------------------------------------
    let mut closing_slots = 0u32;
    let mut new_symbol_slots = 0u32;
    let mut completing_slots = 0u32;
    let mut completable_now = [false; NUM_SCIENCE];
    // Every revealed card carrying a symbol this player lacks, accessible or
    // not: what a denial has to touch to matter. Straight off the board's
    // per-symbol slot masks rather than a scan.
    let mut missing_symbol_slots = 0u32;
    let mut any_symbol_slots = 0u32;
    for sym in ALL_SCIENCE {
        let slots = board.symbol_slots[sym.index()];
        any_symbol_slots |= slots;
        if held[sym.index()] == 0 {
            missing_symbol_slots |= slots;
        }
    }
    let reachable_missing_slots = reachable_slots & missing_symbol_slots;
    for slot in iter_slots(board.accessible & any_symbol_slots & prices.affordable_now) {
        let Some(card) = board.slot_card[slot as usize] else {
            continue;
        };
        let Some(sym) = card.def().science else {
            continue;
        };
        if held[sym.index()] == 0 {
            new_symbol_slots |= 1u32 << slot;
            if base.missing == 1 {
                closing_slots |= 1u32 << slot;
            }
        }
        if base.candidates[sym.index()] {
            completing_slots |= 1u32 << slot;
            completable_now[sym.index()] = true;
        }
    }

    let law = m.law_token();
    let closing_via_token = if base.missing == 1 {
        match (law, state.pending()) {
            (Some(t), Some(Pending::ProgressToken))
                if state.current_player() == player && base.law_on_board =>
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
        candidates: base.candidates,
        completable_ever: base.completable_ever,
        completable_now,
        completing_slots,
        best_board_token,
        board_token_total,
    };

    let model = SciModel {
        symbols,
        held,
        missing: base.missing,
        dead: base.dead,
        decisions_left_eff: tempo.decisions_left_eff,
        share_defender: ctx.tempo(player.other()).share,
        defender_has_half_pair: opponent.has_live_half_pair,
        p_law,
        p_gl,
    };
    let detail = model.magnitude();

    // A pending token choice that hands this player the Law token is a closed
    // race the structural model cannot see: it is not a card in a slot.
    let magnitude = if closing_via_token.is_some() {
        1.0
    } else {
        detail.value
    };

    let fragility = u8::try_from(
        ALL_SCIENCE
            .into_iter()
            .filter(|&s| {
                Some(s) != m.law_symbol()
                    && base.availability[s.index()].held == 0
                    && base.availability[s.index()].obtainable_copies() == 1
            })
            .count(),
    )
    .unwrap_or(u8::MAX);

    let status = status_for(magnitude, base.dead, w);

    ScienceRead {
        player,
        distinct: base.distinct,
        missing: base.missing,
        availability: base.availability,
        obtainable_missing: base.obtainable_missing,
        second_copy_obtainable: base.completable_ever,
        undeniable_route: std::array::from_fn(|i| {
            base.availability[i].held == 0 && base.availability[i].undeniable_route_only()
        }),
        fragility,
        dead: base.dead,
        magnitude,
        detail,
        model,
        status,
        reachable_slots,
        missing_symbol_slots,
        reachable_missing_slots,
        closing_slots,
        closing_via_token,
        new_symbol_slots,
        pair_setup,
    }
}

/// The label for a science magnitude.
pub fn status_for(magnitude: f64, dead: bool, w: &ThreatWeights) -> ScienceStatus {
    if dead {
        ScienceStatus::Closed
    } else if magnitude >= 1.0 {
        ScienceStatus::Imminent
    } else if magnitude >= w.live_threshold {
        ScienceStatus::Live
    } else if magnitude >= w.pressure_threshold {
        ScienceStatus::Pressure
    } else {
        ScienceStatus::Closed
    }
}

/// `R_T`: the slots `player` could take on their very next turn, and which
/// symbol each of them carries.
///
/// Depth zero is the accessible slots. Each further step models one extra turn
/// from a play-again wonder: the wonder eats one reachable card, whatever that
/// uncovers and was already face up becomes reachable too. A card at depth `d`
/// counts only if the player can pay for the `d` wonder builds *and* the card,
/// which is the design's deliberately simple combined coin check rather than a
/// full resource-trade replay.
///
/// Beyond depth one this is optimistic: it lets every slot reachable so far be
/// the one that got eaten. Depth two needs two play-again wonders affordable
/// at once, which is rare enough not to be worth an exact search.
fn reachable(board: &Board, prices: &Prices, tempo: &Tempo) -> (u32, [u32; NUM_SCIENCE]) {
    let mut out = 0u32;

    let mut layer = board.accessible;
    let mut seen = board.accessible;
    let mut occupancy = board.occupied;
    let chain = usize::from(tempo.chain).min(MAX_CHAIN);
    for depth in 0..=chain {
        let prefix = tempo.chain_cost[depth];
        for slot in iter_slots(layer) {
            if !prices.can_afford_slot_after(slot, prefix) {
                continue;
            }
            out |= 1u32 << slot;
        }
        if depth == chain {
            break;
        }
        occupancy &= !layer;
        let mut next = 0u32;
        for slot in iter_slots(layer) {
            let (known, _) = board.newly_open_slots_after(slot, occupancy | (1u32 << slot));
            next |= known;
        }
        next &= !seen;
        if next == 0 {
            break;
        }
        seen |= next;
        layer = next;
    }
    (out, std::array::from_fn(|i| out & board.symbol_slots[i]))
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
        assert_eq!(
            r.status,
            ScienceStatus::Closed,
            "six missing with no structure dealt is not a race"
        );
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

    #[test]
    fn p_at_least_k_agrees_with_the_binomial_it_generalises() {
        // Five independent coin flips: P(at least 3 heads) = 16/32.
        let ps = [0.5f64; 5];
        assert!((p_at_least_k(&ps, 3) - 0.5).abs() < 1e-12);
        assert!((p_at_least_k(&ps, 0) - 1.0).abs() < 1e-12);
        assert!((p_at_least_k(&ps, 6) - 0.0).abs() < 1e-12);
        // A certain event and an impossible one.
        assert!((p_at_least_k(&[1.0, 0.0], 1) - 1.0).abs() < 1e-12);
        assert!((p_at_least_k(&[1.0, 0.0], 2) - 0.0).abs() < 1e-12);
        assert!((p_at_least_k(&[], 1) - 0.0).abs() < 1e-12);
    }
}
