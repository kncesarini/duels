//! The rules engine: setup, legal-move generation, action application, and
//! the chance API.
//!
//! # Determinism
//!
//! The engine performs no I/O, reads no clock, and holds no global mutable
//! state. Every random decision enters through an explicitly passed
//! [`StdRng`]: [`new_game`] seeds one from a `u64`, and the only in-game
//! random event — The Great Library drawing three of the out-of-play progress
//! tokens — takes the same `&mut StdRng`. A game is therefore fully
//! reproducible from `(seed, actions)`.
//!
//! # Two ways to apply an action
//!
//! [`apply`] is the ordinary path: it uses the state's own (already shuffled)
//! card layout for any reveal, and the `rng` for The Great Library.
//!
//! [`chance_outcomes`] + [`apply_with_outcome`] is the search path. A
//! search-based agent that wants to expand a *chance node* instead of
//! determinizing asks the engine what could be revealed and with what
//! probability — computed from public knowledge only, ignoring what this
//! particular state happens to be hiding — and then applies the action with a
//! chosen outcome forced. Applying with a forced outcome rewrites the state's
//! hidden layout to stay consistent, so the resulting state is a legal state
//! that could have arisen from that reveal.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::action::Action;
use crate::cost::{self, Cost};
use crate::data::{self, CardId, CardType, Science, TokenId, WonderId};
use crate::event::{CoinReason, Event, EventLog};
use crate::layout;
use crate::scoring::{self, GameResult, VictoryKind};
use crate::state::{
    iter_mask_u128, iter_slots, GameState, Pending, Phase, CARDS_REMOVED_PER_AGE, GUILDS_IN_PLAY,
    TOKENS_ON_BOARD,
};
use crate::Player;

/// An action that is not legal in the given state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IllegalAction {
    /// The action that was rejected.
    pub action: Action,
    /// Why it was rejected.
    pub reason: &'static str,
}

impl std::fmt::Display for IllegalAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "illegal action {:?}: {}", self.action, self.reason)
    }
}

impl std::error::Error for IllegalAction {}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/// Set up a new game from `seed`.
///
/// Everything random happens here: which player drafts first, the shuffle of
/// each age deck, which three cards are returned to the box from each age,
/// which three of the seven guilds join the Age III deck, which five progress
/// tokens go on the board, and the order of the wonder pile.
///
/// The returned state is in [`Phase::WonderDraft`]; the Age I structure is
/// not dealt until the draft finishes, matching the physical setup order.
pub fn new_game(seed: u64) -> GameState {
    let mut rng = StdRng::seed_from_u64(seed);
    new_game_with_rng(&mut rng)
}

/// Set up a new game from an existing RNG, so a caller can drive many games
/// from one stream.
pub fn new_game_with_rng(rng: &mut StdRng) -> GameState {
    let s = data::statics();
    let mut st = GameState::empty();

    // The rulebook says only "choose a first player"; we choose one at
    // random so that a seeded game is fully specified.
    let first = if rng.gen_bool(0.5) {
        Player::One
    } else {
        Player::Two
    };
    st.set_draft_first(first);

    let mut pile: Vec<WonderId> = WonderId::all().collect();
    pile.shuffle(rng);
    st.set_draft_pile(
        pile.try_into()
            .unwrap_or_else(|_| unreachable!("12 wonders")),
    );

    let mut toks: Vec<TokenId> = TokenId::all().collect();
    toks.shuffle(rng);
    let mut board = 0u16;
    let mut aside = 0u16;
    for (i, t) in toks.iter().enumerate() {
        if i < TOKENS_ON_BOARD {
            board |= 1u16 << t.index();
        } else {
            aside |= 1u16 << t.index();
        }
    }
    st.set_tokens(board, aside);

    let mut out_of_game = 0u128;
    let bit = |c: CardId| 1u128 << c.index();

    for age in 1..=3u8 {
        let deck: Vec<CardId> = if age < 3 {
            // 23 cards; three go back in the box unseen.
            let mut d: Vec<CardId> = iter_mask_u128(s.age_masks[(age - 1) as usize]).collect();
            d.shuffle(rng);
            for c in d.drain(layout::SLOTS..) {
                out_of_game |= bit(c);
            }
            d
        } else {
            // Three of the 20 Age III cards go back in the box, then three
            // of the seven guilds are shuffled in, giving exactly 20.
            let mut plain: Vec<CardId> = iter_mask_u128(s.age_masks[2] & !s.guild_mask).collect();
            plain.shuffle(rng);
            for c in plain.drain(layout::SLOTS - CARDS_REMOVED_PER_AGE..) {
                out_of_game |= bit(c);
            }
            let mut guilds: Vec<CardId> = iter_mask_u128(s.guild_mask).collect();
            guilds.shuffle(rng);
            for c in guilds.drain(GUILDS_IN_PLAY..) {
                out_of_game |= bit(c);
            }
            plain.extend(guilds);
            plain.shuffle(rng);
            plain
        };
        debug_assert_eq!(deck.len(), layout::SLOTS);
        st.set_age_deck(
            age,
            deck.try_into()
                .unwrap_or_else(|_| unreachable!("20 cards per age structure")),
        );
    }
    st.set_out_of_game(out_of_game);
    st.set_phase(Phase::WonderDraft);
    st
}

fn start_age(st: &mut GameState, age: u8, log: &mut EventLog) {
    st.set_age(age);
    let l = layout::layout(age);
    st.set_board(layout::ALL_SLOTS, l.face_up);
    log.push(|| Event::AgeStarted { age });
}

// ---------------------------------------------------------------------------
// Legal actions
// ---------------------------------------------------------------------------

/// Every action legal in `state`, in a deterministic order.
///
/// Empty exactly when the game is over: while play continues, discarding an
/// accessible card for coins is always available.
pub fn legal_actions(state: &GameState) -> Vec<Action> {
    let mut out = Vec::new();
    legal_actions_into(state, &mut out);
    out
}

/// [`legal_actions`] into a caller-owned buffer, for hot loops that want to
/// avoid re-allocating.
pub fn legal_actions_into(state: &GameState, out: &mut Vec<Action>) {
    out.clear();
    match state.phase() {
        Phase::GameOver => {}
        Phase::WonderDraft => {
            for wonder in state.offered_wonders() {
                out.push(Action::PickWonder { wonder });
            }
        }
        Phase::ChooseFirstPlayer => {
            out.push(Action::ChooseFirstPlayer {
                player: Player::One,
            });
            out.push(Action::ChooseFirstPlayer {
                player: Player::Two,
            });
        }
        Phase::Turn => match state.pending() {
            Some(Pending::ProgressToken) => {
                for token in state.board_tokens() {
                    out.push(Action::ChooseProgressToken { token });
                }
            }
            Some(Pending::GreatLibraryToken { tokens }) => {
                for token in tokens {
                    out.push(Action::ChooseGreatLibraryToken { token });
                }
            }
            Some(Pending::Destroy { card_type }) => {
                let s = data::statics();
                let mask = state.player(state.current_player().other()).built_mask()
                    & s.card_masks[card_type.index()];
                for card in iter_mask_u128(mask) {
                    out.push(Action::DestroyOpponentCard { card });
                }
            }
            Some(Pending::MausoleumBuild) => {
                for card in state.discard_pile() {
                    out.push(Action::MausoleumBuild { card });
                }
            }
            None => {
                let p = state.current_player();
                let me = state.player(p);
                let mut buildable_wonders: Vec<WonderId> = Vec::new();
                if state.wonder_slots_left() {
                    for w in me.wonders() {
                        if !me.has_built_wonder(w)
                            && cost::wonder_cost(state, p, w).affordable_by(state, p)
                        {
                            buildable_wonders.push(w);
                        }
                    }
                }
                for slot in iter_slots(state.accessible_slots()) {
                    let Some(card) = state.face_up_card(slot) else {
                        continue;
                    };
                    if cost::card_cost(state, p, card).affordable_by(state, p) {
                        out.push(Action::Build { slot });
                    }
                    out.push(Action::Discard { slot });
                    for &wonder in &buildable_wonders {
                        out.push(Action::BuildWonder { slot, wonder });
                    }
                }
            }
        },
    }
}

/// Whether `action` is legal in `state`.
pub fn is_legal(state: &GameState, action: Action) -> bool {
    legal_actions(state).contains(&action)
}

// ---------------------------------------------------------------------------
// Chance
// ---------------------------------------------------------------------------

/// The `(slot, card)` pairs an action turns face up: at most two, because a
/// card covers at most two others.
pub type RevealSlots = [Option<(u8, CardId)>; 2];

/// A resolution of the randomness an action triggers.
///
/// At most two slots can be uncovered by removing one card (a card covers at
/// most two others), and the only other random event in the game is The
/// Great Library's three-token draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Outcome {
    /// `(slot, card)` pairs for the slots turned face up by this action.
    pub reveals: RevealSlots,
    /// The three tokens The Great Library drew, if it was built.
    pub library_tokens: Option<[TokenId; 3]>,
}

impl Outcome {
    /// Whether this outcome resolves no randomness at all.
    pub fn is_trivial(&self) -> bool {
        self.reveals.iter().all(Option::is_none) && self.library_tokens.is_none()
    }
}

/// Public knowledge about what the current age is still hiding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiddenInfo {
    /// Slots that still hold a face-down card.
    pub hidden_slots: u32,
    /// Guild cards of the current age nobody has seen yet.
    pub unseen_guilds: Vec<CardId>,
    /// Non-guild cards of the current age nobody has seen yet.
    pub unseen_plain: Vec<CardId>,
    /// How many of the face-down slots hold a guild card. Publicly known
    /// because exactly three guilds are dealt into Age III.
    pub hidden_guild_count: u32,
    /// *Which* face-down slots hold a guild card, as a slot bitmask (bit `i`
    /// for slot `i`, the same indexing as [`GameState::occupied_slots`]).
    ///
    /// Publicly known because guild cards have a distinguishable purple card
    /// back: both players can see which face-down Age III slots are guilds
    /// from the backs alone, without seeing *which* guild. Always a subset of
    /// [`Self::hidden_slots`], and always zero in Ages I and II, which
    /// contain no guilds (R-022).
    pub hidden_guild_slots: u32,
}

impl HiddenInfo {
    /// Every card that could be behind a face-down slot, guilds first.
    pub fn pool(&self) -> Vec<CardId> {
        let mut v = self.unseen_guilds.clone();
        v.extend(self.unseen_plain.iter().copied());
        v
    }
}

/// What the current age is still hiding, derived from public information
/// only.
///
/// The pool is deliberately larger than the number of face-down slots: three
/// cards were returned to the box unseen, so they remain candidates for every
/// hidden slot until the age ends.
///
/// Two of the fields read a face-down slot's actual card, which is hidden
/// information — but only to project it onto something that is public.
/// [`HiddenInfo::hidden_guild_slots`] keeps a single bit per slot, guild or
/// not, which is exactly what the purple card back shows (R-110); the
/// `revealed_slots` loop below reads only slots that are already face up.
/// Nothing here narrows *which* card sits in a face-down slot, and the
/// determinization-invariance tests in `observation.rs` and
/// `tests/properties.rs` are what pin that down.
pub fn hidden_info(state: &GameState) -> HiddenInfo {
    let s = data::statics();
    let age = state.age();
    let age_mask = s.age_masks[(age.max(1) - 1) as usize];

    // Everything of this age whose identity is public: face-up in the
    // structure, or already taken into a city, the discard pile, or under a
    // wonder.
    let mut seen = (state.player(Player::One).built_mask()
        | state.player(Player::Two).built_mask()
        | state.discard_mask()
        | state.wonder_fodder_mask())
        & age_mask;
    let hidden_slots = state.occupied_slots() & !state.revealed_slots();
    for slot in iter_slots(state.revealed_slots() & state.occupied_slots()) {
        seen |= 1u128 << state.slot_card_hidden(slot).index();
    }

    let unseen = age_mask & !seen;
    let unseen_guilds: Vec<CardId> = iter_mask_u128(unseen & s.guild_mask).collect();
    let unseen_plain: Vec<CardId> = iter_mask_u128(unseen & !s.guild_mask).collect();

    // Exactly three guilds are in the Age III structure, so however many have
    // been seen, the rest are behind face-down slots. Earlier ages contain no
    // guilds at all.
    let hidden_guild_count = if age == 3 {
        let guilds_seen = (seen & s.guild_mask).count_ones();
        (GUILDS_IN_PLAY as u32).saturating_sub(guilds_seen)
    } else {
        0
    };

    // Which of those face-down slots is a guild: public from the card back.
    let hidden_guild_slots = iter_slots(hidden_slots)
        .filter(|&slot| s.guild_mask & (1u128 << state.slot_card_hidden(slot).index()) != 0)
        .fold(0u32, |m, slot| m | (1u32 << slot));

    HiddenInfo {
        hidden_slots,
        unseen_guilds,
        unseen_plain,
        hidden_guild_count,
        hidden_guild_slots,
    }
}

/// The slots `action` would turn face up, if any.
fn slots_revealed_by(state: &GameState, action: Action) -> Vec<u8> {
    let slot = match action {
        Action::Build { slot } | Action::Discard { slot } | Action::BuildWonder { slot, .. } => {
            slot
        }
        _ => return Vec::new(),
    };
    let l = layout::layout(state.age());
    let occ = state.occupied_slots() & !(1u32 << slot);
    iter_slots(l.covers[slot as usize] & occ)
        .filter(|&i| {
            l.covered_by[i as usize] & occ == 0 && state.revealed_slots() & (1u32 << i) == 0
        })
        .collect()
}

/// Every way the randomness triggered by `action` could resolve, with
/// probabilities computed from public knowledge only.
///
/// Returns a single trivial outcome with probability 1 when the action
/// resolves no randomness. Probabilities sum to 1 (up to floating-point
/// rounding).
///
/// # What the distribution conditions on (R-105)
///
/// Everything public about the face-down slots, which since R-110 includes
/// *which* of them hold a guild: a guild's purple card back is visible to
/// both players. So a slot in [`HiddenInfo::hidden_guild_slots`] can only
/// reveal a guild, a slot outside it can only reveal an ordinary card, and
/// the identity within that class is a uniform draw from the matching unseen
/// pool — the same conditioning [`crate::Observation::sample_state`] already
/// applies (R-111). Because the classes are fixed per slot rather than
/// distributed over the slots by a count, the enumeration needs no
/// combinatorics: one slot is `1/|pool|` per card, and two slots are
/// `1/(n·(n-1))` when they share a class and `1/(n₀·n₁)` when they do not.
///
/// This conditioning is what makes the distribution *accurate*; it does not
/// make it leak. Every input is read off [`hidden_info`], whose fields are all
/// public, so two determinizations of the same
/// [`Observation`](crate::Observation) produce the same distribution — which
/// `engine::tests::chance_outcomes_are_invariant_across_determinizations`
/// asserts directly.
pub fn chance_outcomes(state: &GameState, action: Action) -> Vec<(Outcome, f64)> {
    let reveal_slots = slots_revealed_by(state, action);
    let info = hidden_info(state);
    let is_guild_slot = |slot: u8| info.hidden_guild_slots & (1u32 << slot) != 0;
    let pool_for = |slot: u8| -> &[CardId] {
        if is_guild_slot(slot) {
            &info.unseen_guilds
        } else {
            &info.unseen_plain
        }
    };

    // The reveal half.
    let mut reveal_options: Vec<(RevealSlots, f64)> = Vec::new();
    match reveal_slots.len() {
        0 => reveal_options.push(([None, None], 1.0)),
        1 => {
            let slot = reveal_slots[0];
            let pool = pool_for(slot);
            if pool.is_empty() {
                // Unreachable under the real rules: a face-down slot's own
                // card is by definition unseen, so its class pool holds at
                // least it. Falling back to the trivial outcome keeps the
                // probabilities summing to 1 rather than returning nothing.
                reveal_options.push(([None, None], 1.0));
            } else {
                let p = 1.0 / pool.len() as f64;
                for &card in pool {
                    reveal_options.push(([Some((slot, card)), None], p));
                }
            }
        }
        _ => {
            let (s0, s1) = (reveal_slots[0], reveal_slots[1]);
            let (p0, p1) = (pool_for(s0), pool_for(s1));
            // Same class means one pool drawn from twice without replacement;
            // different classes means two disjoint pools, so `c0 != c1` below
            // never fires and the count is the plain product.
            let support = if is_guild_slot(s0) == is_guild_slot(s1) {
                p0.len() * p0.len().saturating_sub(1)
            } else {
                p0.len() * p1.len()
            };
            if support == 0 {
                reveal_options.push(([None, None], 1.0));
            } else {
                let p = 1.0 / support as f64;
                for &c0 in p0 {
                    for &c1 in p1 {
                        if c0 != c1 {
                            reveal_options.push(([Some((s0, c0)), Some((s1, c1))], p));
                        }
                    }
                }
            }
        }
    }

    // The Great Library half.
    let library_options: Vec<(Option<[TokenId; 3]>, f64)> = match action {
        Action::BuildWonder { wonder, .. } if wonder.def().choose_progress_token => {
            let aside: Vec<TokenId> = state.set_aside_tokens().collect();
            if aside.len() < 3 {
                vec![(None, 1.0)]
            } else {
                let mut v = Vec::new();
                let n = aside.len();
                for i in 0..n {
                    for j in i + 1..n {
                        for k in j + 1..n {
                            v.push((Some([aside[i], aside[j], aside[k]]), 0.0));
                        }
                    }
                }
                let p = 1.0 / v.len() as f64;
                for e in &mut v {
                    e.1 = p;
                }
                v
            }
        }
        _ => vec![(None, 1.0)],
    };

    let mut out = Vec::with_capacity(reveal_options.len() * library_options.len());
    for (reveals, pr) in &reveal_options {
        for (library_tokens, pl) in &library_options {
            out.push((
                Outcome {
                    reveals: *reveals,
                    library_tokens: *library_tokens,
                },
                pr * pl,
            ));
        }
    }
    out
}

/// Rewrite the state's hidden card layout so that `outcome`'s reveals will
/// happen.
///
/// The layout is re-derived rather than patched, because a naive swap can
/// violate a public constraint: forcing a guild card out of the box and into
/// a slot, displacing a non-guild card, would leave four guilds in the Age
/// III structure when exactly three are dealt. The re-derivation keeps each
/// still-hidden slot's current card wherever that is consistent, so the
/// perturbation stays as small as the constraints allow, and reassigns the
/// rest deterministically.
///
/// # Preserving the public guild mask (R-105, R-110)
///
/// The re-derivation must not move a guild between two slots that both stay
/// face down: which face-down slots are guilds is *public* (R-110), so doing
/// so would silently rewrite public information. Every slot that stays hidden
/// therefore keeps the class its own card back already shows, and only the
/// identity within that class can move.
///
/// That is achievable exactly when the forced reveals agree with the mask,
/// which is all [`chance_outcomes`] can produce. A hand-built outcome that
/// puts a guild behind a plain back cannot preserve both the mask and the
/// three-guild deal at once, so it falls back to the pre-R-110 rule —
/// honour the guild *count* only — rather than failing. That keeps this
/// function total for callers that construct outcomes themselves, and is the
/// only path on which the mask can still shift.
fn force_outcome(state: &mut GameState, outcome: &Outcome) -> Result<(), &'static str> {
    let forced: Vec<(u8, CardId)> = outcome.reveals.iter().flatten().copied().collect();
    if forced.is_empty() {
        return Ok(());
    }
    let info = hidden_info(state);
    let age = state.age();
    let age_mask = data::statics().age_masks[(age.max(1) - 1) as usize];

    let mut guild_pool = info.unseen_guilds.clone();
    let mut plain_pool = info.unseen_plain.clone();
    let mut guilds_needed = info.hidden_guild_count as usize;

    // Take the forced cards out of the pool.
    for &(slot, card) in &forced {
        if state.occupied_slots() & (1u32 << slot) == 0
            || state.revealed_slots() & (1u32 << slot) != 0
        {
            return Err("a forced reveal names a slot that is not face down");
        }
        let pool = if card.def().is_guild() {
            &mut guild_pool
        } else {
            &mut plain_pool
        };
        let Some(at) = pool.iter().position(|&c| c == card) else {
            return Err("a forced reveal names a card that is not a candidate");
        };
        pool.swap_remove(at);
        if card.def().is_guild() {
            if guilds_needed == 0 {
                return Err("a forced reveal exceeds the number of hidden guilds");
            }
            guilds_needed -= 1;
        }
    }

    let remaining: Vec<u8> = iter_slots(state.occupied_slots() & !state.revealed_slots())
        .filter(|s| !forced.iter().any(|(f, _)| f == s))
        .collect();
    if guilds_needed > remaining.len() {
        return Err("not enough hidden slots left for the remaining guilds");
    }
    let mut plains_needed = remaining.len() - guilds_needed;

    let deck = *state.age_deck(age);
    let mut assignment: Vec<Option<CardId>> = vec![None; remaining.len()];

    // Do the forced reveals agree with the public per-slot guild mask? They
    // always do when the outcome came from `chance_outcomes`.
    let mask_consistent = forced.iter().all(|&(slot, card)| {
        (info.hidden_guild_slots & (1u32 << slot) != 0) == card.def().is_guild()
    });

    if mask_consistent {
        // Every slot that stays hidden keeps its own class, so the public mask
        // is untouched on exactly the slots that remain public knowledge.
        let wants_guild = |slot: u8| info.hidden_guild_slots & (1u32 << slot) != 0;
        // Keep each slot's current card unless a forced reveal claimed it.
        for (i, &slot) in remaining.iter().enumerate() {
            let current = deck[slot as usize];
            let pool = if wants_guild(slot) {
                &mut guild_pool
            } else {
                &mut plain_pool
            };
            if let Some(at) = pool.iter().position(|&c| c == current) {
                pool.swap_remove(at);
                assignment[i] = Some(current);
            }
        }
        // Backfill the rest from that slot's own class pool.
        for (i, &slot) in remaining.iter().enumerate() {
            if assignment[i].is_some() {
                continue;
            }
            let pool = if wants_guild(slot) {
                &mut guild_pool
            } else {
                &mut plain_pool
            };
            assignment[i] = Some(
                pool.pop()
                    .ok_or("ran out of candidate cards while forcing an outcome")?,
            );
        }
    } else {
        // Pre-R-110 fallback: only the guild count is honoured, so a guild can
        // move between two slots that both stay face down.
        //
        // Keep each slot's current card where the quota and the pool allow it.
        for (i, &slot) in remaining.iter().enumerate() {
            let current = deck[slot as usize];
            let is_guild = current.def().is_guild();
            let (pool, quota) = if is_guild {
                (&mut guild_pool, &mut guilds_needed)
            } else {
                (&mut plain_pool, &mut plains_needed)
            };
            if *quota == 0 {
                continue;
            }
            if let Some(at) = pool.iter().position(|&c| c == current) {
                pool.swap_remove(at);
                *quota -= 1;
                assignment[i] = Some(current);
            }
        }
        // Fill whatever is left, guilds first.
        for slot in assignment.iter_mut() {
            if slot.is_some() {
                continue;
            }
            let card = if guilds_needed > 0 {
                guilds_needed -= 1;
                guild_pool.pop()
            } else {
                plains_needed = plains_needed.saturating_sub(1);
                plain_pool.pop()
            };
            *slot = Some(card.ok_or("ran out of candidate cards while forcing an outcome")?);
        }
    }

    let mut new_deck = deck;
    for &(slot, card) in &forced {
        new_deck[slot as usize] = card;
    }
    for (i, &slot) in remaining.iter().enumerate() {
        new_deck[slot as usize] = assignment[i].expect("every slot was assigned");
    }
    state.set_age_deck(age, new_deck);

    // Whatever is left in the pools is what went back in the box.
    let mut out = state.out_of_game_mask() & !age_mask;
    for c in guild_pool.iter().chain(plain_pool.iter()) {
        out |= 1u128 << c.index();
    }
    state.set_out_of_game(out);
    Ok(())
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

enum Chance<'a> {
    Rng(&'a mut StdRng),
    Forced(&'a Outcome),
}

/// Apply `action`, sampling any randomness from `rng`.
///
/// Reveals come from the layout the state already holds (the cards were
/// shuffled at setup), so `rng` is consumed only by The Great Library.
pub fn apply(
    state: &mut GameState,
    action: Action,
    rng: &mut StdRng,
) -> Result<Vec<Event>, IllegalAction> {
    let mut log = EventLog::recording();
    apply_inner(state, action, &mut Chance::Rng(rng), &mut log, true)?;
    Ok(log.into_events())
}

/// [`apply`] without recording events, for simulation throughput.
pub fn apply_quiet(
    state: &mut GameState,
    action: Action,
    rng: &mut StdRng,
) -> Result<(), IllegalAction> {
    let mut log = EventLog::discarding();
    apply_inner(state, action, &mut Chance::Rng(rng), &mut log, true)
}

/// [`apply_quiet`] that trusts the caller to have taken `action` from
/// [`legal_actions`].
///
/// Skipping the legality check roughly halves the cost of a simulated turn.
/// Applying an illegal action leaves the state in an unspecified — though
/// still memory-safe — condition.
pub fn apply_unchecked(state: &mut GameState, action: Action, rng: &mut StdRng) {
    let mut log = EventLog::discarding();
    let _ = apply_inner(state, action, &mut Chance::Rng(rng), &mut log, false);
}

/// Apply `action` with its randomness forced to `outcome`.
///
/// Use with [`chance_outcomes`] to expand a chance node during search without
/// needing an RNG. The state's hidden layout is rewritten to match, so the
/// result is a state that genuinely could have arisen from that reveal.
pub fn apply_with_outcome(
    state: &mut GameState,
    action: Action,
    outcome: &Outcome,
) -> Result<Vec<Event>, IllegalAction> {
    let mut log = EventLog::recording();
    apply_inner(state, action, &mut Chance::Forced(outcome), &mut log, true)?;
    Ok(log.into_events())
}

/// [`apply_with_outcome`] without recording events and without re-checking
/// legality, for search hot loops.
///
/// A search agent expanding a chance node already has `action` from
/// [`legal_actions`] and `outcome` from [`chance_outcomes`]; re-deriving the
/// legal move list to validate the action and allocating an event log for a
/// node it is about to throw away is pure overhead (together they dominate
/// the cost of a search node). Errors are still reported, because forcing an
/// outcome can fail for reasons the caller cannot check cheaply.
///
/// Applying an illegal action leaves the state in an unspecified — though
/// still memory-safe — condition, exactly as with [`apply_unchecked`].
pub fn apply_with_outcome_unchecked(
    state: &mut GameState,
    action: Action,
    outcome: &Outcome,
) -> Result<(), IllegalAction> {
    let mut log = EventLog::discarding();
    apply_inner(state, action, &mut Chance::Forced(outcome), &mut log, false)
}

fn illegal(action: Action, reason: &'static str) -> IllegalAction {
    IllegalAction { action, reason }
}

fn apply_inner(
    state: &mut GameState,
    action: Action,
    chance: &mut Chance<'_>,
    log: &mut EventLog,
    validate: bool,
) -> Result<(), IllegalAction> {
    if validate && !is_legal(state, action) {
        return Err(illegal(action, "not in legal_actions for this state"));
    }
    if let Chance::Forced(outcome) = chance {
        force_outcome(state, outcome).map_err(|r| illegal(action, r))?;
    }

    match action {
        Action::PickWonder { wonder } => {
            let p = state.current_player();
            state.player_mut(p).draft_wonder(wonder);
            log.push(|| Event::WonderPicked { player: p, wonder });
            state.advance_draft();
            if state.phase() == Phase::Turn {
                start_age(state, 1, log);
            } else if state.draft_step() == 4 {
                let pile = *state.draft_pile();
                log.push(|| Event::WonderGroupRevealed {
                    wonders: [pile[4], pile[5], pile[6], pile[7]],
                });
            }
        }

        Action::ChooseFirstPlayer { player } => {
            state.set_current(player);
            state.set_phase(Phase::Turn);
            log.push(|| Event::FirstPlayerChosen { player });
        }

        Action::Build { slot } => {
            let p = state.current_player();
            let card = state.slot_card_hidden(slot);
            let cost = cost::card_cost(state, p, card);
            take_card(state, slot, p, log);
            pay(state, p, cost, log);
            construct_card(state, p, card, cost.via_chain, log);
            finish_turn(state, log);
        }

        Action::Discard { slot } => {
            let p = state.current_player();
            let card = take_card(state, slot, p, log);
            state.add_to_discard(card);
            log.push(|| Event::CardDiscarded { player: p, card });
            let gained = cost::discard_reward(state, p);
            gain(state, p, gained, CoinReason::DiscardedCard, log);
            finish_turn(state, log);
        }

        Action::BuildWonder { slot, wonder } => {
            let p = state.current_player();
            let cost = cost::wonder_cost(state, p, wonder);
            let card = take_card(state, slot, p, log);
            state.add_wonder_fodder(card);
            pay(state, p, cost, log);
            construct_wonder(state, p, wonder, card, chance, log);
            finish_turn(state, log);
        }

        Action::ChooseProgressToken { token } => {
            let p = state.current_player();
            state.set_pending(None);
            state.remove_board_token(token);
            take_token(state, p, token, false, log);
            finish_turn(state, log);
        }

        Action::ChooseGreatLibraryToken { token } => {
            let p = state.current_player();
            let Some(Pending::GreatLibraryToken { tokens }) = state.pending() else {
                return Err(illegal(action, "no Great Library draw is pending"));
            };
            state.set_pending(None);
            let mut mask = 0u16;
            for t in tokens {
                mask |= 1u16 << t.index();
            }
            // The chosen token is kept; the other two go back in the box.
            state.remove_aside_tokens(mask);
            take_token(state, p, token, true, log);
            finish_turn(state, log);
        }

        Action::MausoleumBuild { card } => {
            let p = state.current_player();
            state.set_pending(None);
            state.remove_from_discard(card);
            construct_card(state, p, card, false, log);
            finish_turn(state, log);
        }

        Action::DestroyOpponentCard { card } => {
            let p = state.current_player();
            let victim = p.other();
            state.set_pending(None);
            state.player_mut(victim).remove_built_card(card);
            state.add_to_discard(card);
            log.push(|| Event::CardDestroyed {
                player: p,
                victim,
                card,
            });
            finish_turn(state, log);
        }
    }

    state.bump_turn();
    Ok(())
}

/// Remove the card in `slot`, turning face up anything it was covering.
fn take_card(state: &mut GameState, slot: u8, player: Player, log: &mut EventLog) -> CardId {
    let card = state.slot_card_hidden(slot);
    state.clear_slot(slot);
    state.set_last_card_taker(player);
    log.push(|| Event::CardTaken { player, slot, card });

    let l = layout::layout(state.age());
    let occ = state.occupied_slots();
    let candidates = l.covers[slot as usize] & occ & !state.revealed_slots();
    for i in iter_slots(candidates) {
        if l.covered_by[i as usize] & occ == 0 {
            state.reveal_slot(i);
            let revealed = state.slot_card_hidden(i);
            log.push(|| Event::SlotRevealed {
                slot: i,
                card: revealed,
            });
        }
    }
    card
}

fn gain(
    state: &mut GameState,
    player: Player,
    amount: u16,
    reason: CoinReason,
    log: &mut EventLog,
) {
    if amount == 0 {
        return;
    }
    *state.player_mut(player).coins_mut() += amount;
    log.push(|| Event::CoinsGained {
        player,
        amount,
        reason,
    });
}

fn lose(
    state: &mut GameState,
    player: Player,
    amount: u16,
    reason: CoinReason,
    log: &mut EventLog,
) {
    if amount == 0 {
        return;
    }
    let paid = state.player_mut(player).pay_up_to(amount);
    if paid > 0 {
        log.push(|| Event::CoinsLost {
            player,
            amount: paid,
            reason,
        });
    }
}

fn pay(state: &mut GameState, player: Player, cost: Cost, log: &mut EventLog) {
    if cost.coins == 0 {
        return;
    }
    let bank_part = cost.coins - cost.trade;
    state.player_mut(player).pay_up_to(cost.coins);
    if bank_part > 0 {
        log.push(|| Event::CoinsLost {
            player,
            amount: bank_part,
            reason: CoinReason::ConstructionCost,
        });
    }
    if cost.trade > 0 {
        log.push(|| Event::CoinsLost {
            player,
            amount: cost.trade,
            reason: CoinReason::Trade,
        });
        if cost::opponent_has_economy(state, player) {
            gain(
                state,
                player.other(),
                cost.trade,
                CoinReason::EconomyToken,
                log,
            );
        }
    }
}

/// Put `card` into `player`'s city and resolve its immediate effects.
///
/// Ordering (see `docs/rules-spec.md` R-050): the card enters the city first,
/// so that "per building you own" and guild majority effects count it; then
/// coins, then shields (which may end the game), then science (which may end
/// the game or create a pending token choice).
fn construct_card(
    state: &mut GameState,
    player: Player,
    card: CardId,
    via_chain: bool,
    log: &mut EventLog,
) {
    let def = card.def();
    state.player_mut(player).add_built_card(card);
    log.push(|| Event::CardBuilt {
        player,
        card,
        via_chain,
    });

    if via_chain {
        if let Some(t) = state
            .player(player)
            .token_with(|t| t.chain_build_coins > 0)
            .map(|t| t.def().chain_build_coins)
        {
            gain(
                state,
                player,
                u16::from(t),
                CoinReason::UrbanismChainBonus,
                log,
            );
        }
    }

    gain(
        state,
        player,
        u16::from(def.coins),
        CoinReason::CardEffect,
        log,
    );

    if let Some((target, per)) = def.coins_per_own {
        let n = state.player(player).count(target);
        gain(
            state,
            player,
            n * u16::from(per),
            CoinReason::PerOwnBuilding,
            log,
        );
    }

    if let Some((target, per)) = def.coins_by_majority {
        let n = scoring::majority_count(state, target);
        gain(
            state,
            player,
            n * u16::from(per),
            CoinReason::GuildMajority,
            log,
        );
    }

    if def.shields > 0 {
        let mut shields = def.shields;
        if def.kind == CardType::Military && state.player(player).has_token_with(|t| t.shield_bonus)
        {
            shields += 1;
        }
        resolve_shields(state, player, shields, log);
        if state.is_over() {
            return;
        }
    }

    if let Some(symbol) = def.science {
        resolve_science(state, player, symbol, log);
    }
}

fn construct_wonder(
    state: &mut GameState,
    player: Player,
    wonder: WonderId,
    card: CardId,
    chance: &mut Chance<'_>,
    log: &mut EventLog,
) {
    let def = wonder.def();
    state.player_mut(player).mark_wonder_built(wonder);
    let total_built = state.wonders_built_total();
    log.push(|| Event::WonderBuilt {
        player,
        wonder,
        card,
        total_built,
    });

    gain(
        state,
        player,
        u16::from(def.coins),
        CoinReason::CardEffect,
        log,
    );
    lose(
        state,
        player.other(),
        u16::from(def.opponent_loses_coins),
        CoinReason::WonderPenalty,
        log,
    );

    if def.shields > 0 {
        resolve_shields(state, player, def.shields, log);
        if state.is_over() {
            return;
        }
    }

    if def.play_again || state.player(player).has_token_with(|t| t.wonder_play_again) {
        state.set_extra_turn(true);
        log.push(|| Event::ExtraTurnGranted { player });
    }

    // At most one pending choice; no base-game wonder creates two.
    if let Some(card_type) = def.destroy {
        let s = data::statics();
        let mask = state.player(player.other()).built_mask() & s.card_masks[card_type.index()];
        if mask != 0 {
            state.set_pending(Some(Pending::Destroy { card_type }));
            log.push(|| Event::DestroyPending { player, card_type });
        }
    } else if def.choose_progress_token {
        let aside: Vec<TokenId> = state.set_aside_tokens().collect();
        if aside.len() >= 3 {
            let tokens: [TokenId; 3] = match chance {
                Chance::Forced(o) => o
                    .library_tokens
                    .unwrap_or_else(|| [aside[0], aside[1], aside[2]]),
                Chance::Rng(rng) => {
                    let mut pool = aside.clone();
                    pool.shuffle(*rng);
                    [pool[0], pool[1], pool[2]]
                }
            };
            state.set_pending(Some(Pending::GreatLibraryToken { tokens }));
            log.push(|| Event::GreatLibraryDraw { player, tokens });
        }
    } else if def.build_discarded_free && state.discard_mask() != 0 {
        state.set_pending(Some(Pending::MausoleumBuild));
    }
}

/// Move the conflict pawn, collect any loot token passed, and check for
/// military supremacy.
fn resolve_shields(state: &mut GameState, player: Player, shields: u8, log: &mut EventLog) {
    if shields == 0 {
        return;
    }
    let m = data::military();
    let cap = i8::try_from(m.capital_distance).unwrap_or(9);
    let dir: i8 = if player == Player::One { 1 } else { -1 };
    let from = state.conflict();
    let to = (from + dir * i8::try_from(shields).unwrap_or(i8::MAX)).clamp(-cap, cap);
    state.set_conflict(to);
    state.player_mut(player).add_shields(shields);
    log.push(|| Event::ConflictMoved {
        player,
        shields,
        from,
        to,
    });

    let reached = to * dir;
    let victim = player.other();
    for (i, &(distance, coins)) in m.loot.iter().enumerate() {
        if state.loot_available(player, i) && reached >= i8::try_from(distance).unwrap_or(i8::MAX) {
            state.take_loot(player, i);
            log.push(|| Event::MilitaryLootTriggered {
                loser: victim,
                distance,
                coins,
            });
            lose(
                state,
                victim,
                u16::from(coins),
                CoinReason::MilitaryLoot,
                log,
            );
        }
    }

    if reached >= cap {
        finish(
            state,
            GameResult::Win {
                winner: player,
                kind: VictoryKind::MilitarySupremacy,
            },
            log,
        );
    }
}

/// Note a newly acquired scientific symbol: check for scientific supremacy,
/// then for a completed pair.
fn resolve_science(state: &mut GameState, player: Player, symbol: Science, log: &mut EventLog) {
    let me = state.player(player);
    let distinct = me.distinct_science();
    log.push(|| Event::ScienceGained {
        player,
        symbol,
        distinct,
    });
    if distinct >= 6 {
        finish(
            state,
            GameResult::Win {
                winner: player,
                kind: VictoryKind::ScientificSupremacy,
            },
            log,
        );
        return;
    }
    let me = state.player(player);
    if me.science()[symbol.index()] >= 2 && !me.pair_already_awarded(symbol) {
        state.player_mut(player).mark_pair_awarded(symbol);
        let token_available = state.board_tokens_mask() != 0;
        log.push(|| Event::SciencePairCompleted {
            player,
            symbol,
            token_available,
        });
        if token_available {
            state.set_pending(Some(Pending::ProgressToken));
        }
    }
}

fn take_token(
    state: &mut GameState,
    player: Player,
    token: TokenId,
    from_great_library: bool,
    log: &mut EventLog,
) {
    let def = token.def();
    state.player_mut(player).add_token(token);
    log.push(|| Event::ProgressTokenTaken {
        player,
        token,
        from_great_library,
    });
    gain(
        state,
        player,
        u16::from(def.coins),
        CoinReason::CardEffect,
        log,
    );
    // The Law token supplies a seventh symbol and can therefore complete the
    // six-distinct set.
    if let Some(symbol) = def.science {
        let distinct = state.player(player).distinct_science();
        log.push(|| Event::ScienceGained {
            player,
            symbol,
            distinct,
        });
        if distinct >= 6 {
            finish(
                state,
                GameResult::Win {
                    winner: player,
                    kind: VictoryKind::ScientificSupremacy,
                },
                log,
            );
        }
    }
}

fn finish(state: &mut GameState, result: GameResult, log: &mut EventLog) {
    state.set_pending(None);
    state.set_extra_turn(false);
    state.set_result(result);
    log.push(|| Event::GameEnded { result });
}

/// Decide who acts next, ending the age or the game if the structure is
/// empty.
fn finish_turn(state: &mut GameState, log: &mut EventLog) {
    if state.is_over() || state.pending().is_some() {
        return;
    }
    if state.occupied_slots() == 0 {
        end_age(state, log);
        return;
    }
    if state.extra_turn() {
        state.set_extra_turn(false);
    } else {
        state.set_current(state.current_player().other());
    }
}

fn end_age(state: &mut GameState, log: &mut EventLog) {
    let age = state.age();
    log.push(|| Event::AgeEnded { age });

    // An extra turn that had nowhere to go is simply lost.
    if state.extra_turn() {
        let player = state.current_player();
        state.set_extra_turn(false);
        log.push(|| Event::ExtraTurnLost { player });
    }

    if age >= 3 {
        finish(state, scoring::civilian_result(state), log);
        return;
    }

    start_age(state, age + 1, log);
    match state.military_leader() {
        // Pawn centred: the player who took the last card of the age simply
        // begins the next one.
        None => {
            let player = state.last_card_taker();
            state.set_current(player);
            state.set_phase(Phase::Turn);
            log.push(|| Event::FirstPlayerChosen { player });
        }
        // Otherwise the militarily weaker player chooses who begins.
        Some(leader) => {
            state.set_current(leader.other());
            state.set_phase(Phase::ChooseFirstPlayer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::MAX_WONDERS_BUILT;
    use crate::testing::StateBuilder;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(12345)
    }

    #[test]
    fn setup_partitions_every_card_exactly_once() {
        for seed in 0..50u64 {
            let st = new_game(seed);
            let mut seen = 0u128;
            for age in 1..=3u8 {
                for &c in st.age_deck(age).iter() {
                    let bit = 1u128 << c.index();
                    assert_eq!(seen & bit, 0, "card {c} dealt twice (seed {seed})");
                    seen |= bit;
                }
            }
            let out = st.out_of_game_mask();
            assert_eq!(seen & out, 0, "a dealt card is also out of the game");
            assert_eq!(
                (seen | out).count_ones(),
                data::NUM_CARDS as u32,
                "seed {seed}: {} cards accounted for",
                (seen | out).count_ones()
            );
            // 3 removed from each of Age I and II, 3 Age III cards and 4
            // guilds returned to the box.
            assert_eq!(out.count_ones(), 3 + 3 + 3 + 4);
        }
    }

    #[test]
    fn setup_deals_exactly_three_guilds_into_age_three() {
        let s = data::statics();
        for seed in 0..50u64 {
            let st = new_game(seed);
            let guilds = st
                .age_deck(3)
                .iter()
                .filter(|c| s.guild_mask & (1u128 << c.index()) != 0)
                .count();
            assert_eq!(guilds, 3, "seed {seed}");
            // ...and no guild is dealt into an earlier age.
            for age in [1u8, 2] {
                assert!(st
                    .age_deck(age)
                    .iter()
                    .all(|c| s.guild_mask & (1u128 << c.index()) == 0));
            }
        }
    }

    #[test]
    fn setup_puts_five_tokens_on_the_board_and_five_aside() {
        for seed in 0..20u64 {
            let st = new_game(seed);
            assert_eq!(st.board_tokens_mask().count_ones(), 5, "seed {seed}");
            assert_eq!(st.set_aside_tokens_mask().count_ones(), 5, "seed {seed}");
            assert_eq!(st.board_tokens_mask() & st.set_aside_tokens_mask(), 0);
        }
    }

    #[test]
    fn setup_starts_in_the_wonder_draft_with_an_undealt_structure() {
        let st = new_game(7);
        assert_eq!(st.phase(), Phase::WonderDraft);
        assert_eq!(st.occupied_slots(), 0);
        assert_eq!(st.offered_wonders().len(), 4);
        assert_eq!(st.player(Player::One).coins(), 7);
        assert_eq!(st.player(Player::Two).coins(), 7);
    }

    #[test]
    fn the_draft_gives_each_player_four_wonders_in_one_two_one_order() {
        let mut st = new_game(3);
        let mut rng = rng();
        let mut picks = Vec::new();
        for _ in 0..8 {
            assert_eq!(st.phase(), Phase::WonderDraft);
            let p = st.current_player();
            let a = legal_actions(&st)[0];
            picks.push(p);
            apply(&mut st, a, &mut rng).unwrap();
        }
        let first = st.draft_first();
        assert_eq!(
            picks,
            crate::state::draft_order(first).to_vec(),
            "draft order should be 1-2-2-1 then 2-1-1-2"
        );
        assert_eq!(st.player(Player::One).wonders().count(), 4);
        assert_eq!(st.player(Player::Two).wonders().count(), 4);
        // Age I is dealt only once the draft is over.
        assert_eq!(st.phase(), Phase::Turn);
        assert_eq!(st.age(), 1);
        assert_eq!(st.occupied_slots(), layout::ALL_SLOTS);
        assert_eq!(st.current_player(), first);
    }

    /// Play a whole game with a deterministic "first legal action" policy.
    fn play_out(seed: u64) -> GameState {
        let mut st = new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xabcd);
        let mut buf = Vec::new();
        let mut guard = 0;
        loop {
            legal_actions_into(&st, &mut buf);
            if buf.is_empty() {
                break;
            }
            let a = buf[(st.turn() as usize) % buf.len()];
            apply(&mut st, a, &mut rng).unwrap();
            guard += 1;
            assert!(guard < 10_000, "game did not terminate");
        }
        st
    }

    #[test]
    fn games_terminate_with_a_result() {
        for seed in 0..40u64 {
            let st = play_out(seed);
            assert!(st.is_over(), "seed {seed}");
            assert!(st.result().is_some(), "seed {seed}");
            assert!(legal_actions(&st).is_empty());
        }
    }

    #[test]
    fn a_played_out_game_conserves_every_card() {
        for seed in 0..25u64 {
            let st = play_out(seed);
            let p1 = st.player(Player::One).built_mask();
            let p2 = st.player(Player::Two).built_mask();
            assert_eq!(p1 & p2, 0, "a card cannot be built by both players");
            let accounted = p1 | p2 | st.discard_mask() | st.wonder_fodder_mask();
            // Cards may only leave the structure into one of those four
            // places, and never into two.
            let n = p1.count_ones()
                + p2.count_ones()
                + st.discard_mask().count_ones()
                + st.wonder_fodder_mask().count_ones();
            assert_eq!(
                n,
                accounted.count_ones(),
                "seed {seed}: double-counted card"
            );
            // Everything accounted for came from an age structure.
            let dealt: u128 = (1..=3u8)
                .flat_map(|a| st.age_deck(a).to_vec())
                .fold(0u128, |m, c| m | (1u128 << c.index()));
            assert_eq!(accounted & !dealt, 0, "seed {seed}: card from nowhere");
        }
    }

    #[test]
    fn at_most_seven_wonders_are_ever_built() {
        for seed in 0..40u64 {
            let st = play_out(seed);
            assert!(
                st.wonders_built_total() <= MAX_WONDERS_BUILT,
                "seed {seed}: {} wonders",
                st.wonders_built_total()
            );
        }
    }

    #[test]
    fn the_seventh_wonder_closes_the_wonder_option() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "lumber-yard")])
            .wonders_built(
                Player::One,
                &["the-pyramids", "the-colossus", "the-sphinx", "piraeus"],
            )
            .wonders_built(
                Player::Two,
                &["the-great-library", "the-mausoleum", "the-hanging-gardens"],
            )
            .wonders(Player::One, &["the-appian-way"])
            .coins(Player::One, 50)
            .build();
        assert_eq!(st.wonders_built_total(), MAX_WONDERS_BUILT);
        let actions = legal_actions(&st);
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::BuildWonder { .. })),
            "no eighth wonder may be built: {actions:?}"
        );
    }

    #[test]
    fn discarding_pays_two_plus_your_yellow_cards() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "quarry")])
            .built(Player::One, &["tavern", "brewery", "forum"])
            .coins(Player::One, 0)
            .build();
        let mut st = st;
        let mut rng = rng();
        apply(&mut st, Action::Discard { slot: 14 }, &mut rng).unwrap();
        assert_eq!(st.player(Player::One).coins(), 5);
        assert!(st.discard_pile().any(|c| c.slug() == "quarry"));
    }

    #[test]
    fn a_chain_symbol_makes_a_build_free_and_pays_urbanism() {
        // Statue chains from Theater. With Urbanism the free build also pays
        // four coins.
        let mut st = StateBuilder::new()
            .age(2)
            .open_slots(&[(18, "statue")])
            .built(Player::One, &["theater"])
            .tokens(Player::One, &["urbanism"])
            .coins(Player::One, 0)
            .build();
        // Urbanism's own six-coin payout is granted when the token is taken,
        // not by the builder helper, so start from zero.
        let mut rng = rng();
        let events = apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(st.player(Player::One).coins(), 4);
        assert!(events.iter().any(|e| matches!(
            e,
            Event::CardBuilt {
                via_chain: true,
                ..
            }
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            Event::CoinsGained {
                reason: CoinReason::UrbanismChainBonus,
                ..
            }
        )));
    }

    #[test]
    fn coins_never_go_negative() {
        // The Appian Way makes the opponent lose three coins; they only have
        // one.
        let mut st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "quarry")])
            .wonders(Player::One, &["the-appian-way"])
            .built(
                Player::One,
                &["clay-pool", "clay-pit", "quarry", "stone-pit", "press"],
            )
            .coins(Player::One, 0)
            .coins(Player::Two, 1)
            .build();
        let mut rng = rng();
        apply(
            &mut st,
            Action::BuildWonder {
                slot: 14,
                wonder: WonderId::from_slug("the-appian-way").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(st.player(Player::Two).coins(), 0);
    }

    #[test]
    fn military_loot_triggers_once_per_zone_entry_not_per_turn() {
        // Slot 18/19 are Age II's accessible pair. Player One builds Walls
        // (2 shields, needs 2 stone) twice, crossing +3 on the second build.
        let mut st = StateBuilder::new()
            .age(2)
            .open_slots(&[(18, "walls"), (19, "walls")])
            .built(Player::One, &["shelf-quarry"])
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .build();
        let mut rng = rng();
        // First Walls: 0 -> +2, no loot.
        let ev = apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(st.conflict(), 2);
        assert!(!ev
            .iter()
            .any(|e| matches!(e, Event::MilitaryLootTriggered { .. })));
        assert_eq!(st.player(Player::Two).coins(), 20);

        // Give the turn back to Player One and build the second Walls: +2 ->
        // +4, crossing the token at distance 3.
        st.set_current(Player::One);
        let ev = apply(&mut st, Action::Build { slot: 19 }, &mut rng).unwrap();
        assert_eq!(st.conflict(), 4);
        assert_eq!(
            ev.iter()
                .filter(|e| matches!(e, Event::MilitaryLootTriggered { .. }))
                .count(),
            1
        );
        assert_eq!(st.player(Player::Two).coins(), 18);
        assert!(!st.loot_available(Player::One, 0));
        // The far token at distance 6 is still there.
        assert!(st.loot_available(Player::One, 1));
    }

    #[test]
    fn one_big_push_collects_both_loot_tokens_at_once() {
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "arsenal")])
            .built(Player::One, &["sawmill", "brickyard", "clay-pool"])
            .conflict(4)
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .build();
        let mut rng = rng();
        // Arsenal is 3 shields: +4 -> +7, passing the distance-6 token. The
        // distance-3 token was never collected, so it triggers too.
        let ev = apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(st.conflict(), 7);
        assert_eq!(
            ev.iter()
                .filter(|e| matches!(e, Event::MilitaryLootTriggered { .. }))
                .count(),
            2
        );
        assert_eq!(st.player(Player::Two).coins(), 20 - 2 - 5);
    }

    #[test]
    fn strategy_adds_a_shield_to_red_cards_but_not_to_wonders() {
        let mut st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "guard-tower")])
            .tokens(Player::One, &["strategy"])
            .coins(Player::One, 20)
            .build();
        let mut rng = rng();
        apply(&mut st, Action::Build { slot: 14 }, &mut rng).unwrap();
        assert_eq!(st.conflict(), 2, "Guard Tower's 1 shield plus Strategy");

        let mut st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "guard-tower")])
            .tokens(Player::One, &["strategy"])
            .wonders(Player::One, &["the-colossus"])
            .built(
                Player::One,
                &["clay-pool", "clay-pit", "brickyard", "glassworks"],
            )
            .coins(Player::One, 20)
            .build();
        apply(
            &mut st,
            Action::BuildWonder {
                slot: 14,
                wonder: WonderId::from_slug("the-colossus").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(
            st.conflict(),
            2,
            "The Colossus' 2 shields, no Strategy bonus"
        );
    }

    #[test]
    fn reaching_the_opposing_capital_wins_instantly() {
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "pretorium"), (19, "arsenal")])
            .conflict(7)
            .coins(Player::One, 20)
            .build();
        let mut rng = rng();
        let ev = apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(st.conflict(), 9);
        assert_eq!(
            st.result(),
            Some(GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::MilitarySupremacy
            })
        );
        assert!(ev.iter().any(|e| matches!(e, Event::GameEnded { .. })));
        assert!(legal_actions(&st).is_empty());
    }

    #[test]
    fn six_distinct_symbols_wins_instantly() {
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "university")])
            .built(
                Player::One,
                &[
                    "workshop",
                    "apothecary",
                    "scriptorium",
                    "pharmacist",
                    "academy",
                ],
            )
            .board_tokens(&["philosophy"])
            .coins(Player::One, 20)
            .build();
        assert_eq!(st.player(Player::One).distinct_science(), 5);
        let mut rng = rng();
        apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(
            st.result(),
            Some(GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::ScientificSupremacy
            })
        );
        // The instant win short-circuits the pending token choice.
        assert_eq!(st.pending(), None);
    }

    #[test]
    fn the_law_token_can_complete_the_sixth_symbol() {
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "study")])
            .built(
                Player::One,
                &[
                    "workshop",
                    "apothecary",
                    "scriptorium",
                    "pharmacist",
                    "academy",
                ],
            )
            .board_tokens(&["law"])
            .coins(Player::One, 20)
            .build();
        // Academy and Study share the sundial symbol, so building Study
        // completes a pair (5 distinct symbols, not 6) and offers a token.
        let mut rng = rng();
        apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(st.pending(), Some(Pending::ProgressToken));
        apply(
            &mut st,
            Action::ChooseProgressToken {
                token: TokenId::from_slug("law").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(
            st.result(),
            Some(GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::ScientificSupremacy
            })
        );
    }

    #[test]
    fn a_science_pair_with_no_tokens_left_grants_nothing() {
        let mut st = StateBuilder::new()
            .age(2)
            .open_slots(&[(18, "library"), (19, "brewery")])
            .built(Player::One, &["scriptorium"])
            .board_tokens(&[])
            .coins(Player::One, 20)
            .build();
        let mut rng = rng();
        let ev = apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(st.pending(), None);
        assert!(ev.iter().any(|e| matches!(
            e,
            Event::SciencePairCompleted {
                token_available: false,
                ..
            }
        )));
        // The turn passed normally.
        assert_eq!(st.current_player(), Player::Two);
    }

    #[test]
    fn an_extra_turn_keeps_the_same_player_on_move() {
        let mut st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "quarry"), (15, "clay-pool")])
            .wonders(Player::One, &["the-temple-of-artemis"])
            .built(
                Player::One,
                &["lumber-yard", "quarry", "glassworks", "press"],
            )
            .coins(Player::One, 0)
            .build();
        let mut rng = rng();
        let ev = apply(
            &mut st,
            Action::BuildWonder {
                slot: 14,
                wonder: WonderId::from_slug("the-temple-of-artemis").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        assert!(ev
            .iter()
            .any(|e| matches!(e, Event::ExtraTurnGranted { .. })));
        assert_eq!(st.current_player(), Player::One);
        assert!(!st.extra_turn(), "the extra turn is consumed immediately");
        assert_eq!(st.player(Player::One).coins(), 12);
    }

    #[test]
    fn an_extra_turn_is_lost_when_the_age_ends() {
        // The last card of Age I is spent on a play-again wonder.
        let mut st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "quarry")])
            .wonders(Player::One, &["the-sphinx"])
            .built(
                Player::One,
                &["clay-pool", "quarry", "glassworks", "glassblower"],
            )
            .coins(Player::One, 0)
            .build();
        let mut rng = rng();
        let ev = apply(
            &mut st,
            Action::BuildWonder {
                slot: 14,
                wonder: WonderId::from_slug("the-sphinx").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        assert!(ev
            .iter()
            .any(|e| matches!(e, Event::ExtraTurnGranted { .. })));
        assert!(ev.iter().any(|e| matches!(e, Event::ExtraTurnLost { .. })));
        assert!(!st.extra_turn());
        assert_eq!(st.age(), 2);
    }

    #[test]
    fn theology_grants_an_extra_turn_on_any_wonder() {
        let mut st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "quarry"), (15, "clay-pool")])
            .wonders(Player::One, &["the-pyramids"])
            .tokens(Player::One, &["theology"])
            .built(Player::One, &["quarry", "shelf-quarry", "press"])
            .coins(Player::One, 0)
            .build();
        let mut rng = rng();
        let ev = apply(
            &mut st,
            Action::BuildWonder {
                slot: 14,
                wonder: WonderId::from_slug("the-pyramids").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        assert!(ev
            .iter()
            .any(|e| matches!(e, Event::ExtraTurnGranted { .. })));
        assert_eq!(st.current_player(), Player::One);
    }

    #[test]
    fn the_weaker_player_chooses_who_begins_the_next_age() {
        let mut st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "quarry")])
            .conflict(3) // Player One is ahead, so Player Two is weaker
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        let mut rng = rng();
        apply(&mut st, Action::Build { slot: 14 }, &mut rng).unwrap();
        assert_eq!(st.age(), 2);
        assert_eq!(st.phase(), Phase::ChooseFirstPlayer);
        assert_eq!(st.current_player(), Player::Two);
        apply(
            &mut st,
            Action::ChooseFirstPlayer {
                player: Player::One,
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(st.phase(), Phase::Turn);
        assert_eq!(st.current_player(), Player::One);
    }

    #[test]
    fn a_centred_pawn_hands_the_next_age_to_the_last_card_taker() {
        let mut st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "quarry")])
            .conflict(0)
            .coins(Player::Two, 20)
            .current(Player::Two)
            .build();
        let mut rng = rng();
        apply(&mut st, Action::Build { slot: 14 }, &mut rng).unwrap();
        assert_eq!(st.age(), 2);
        assert_eq!(st.phase(), Phase::Turn);
        assert_eq!(st.current_player(), Player::Two);
    }

    #[test]
    fn destroying_a_building_puts_it_in_the_discard_pile() {
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "obelisk")])
            .wonders(Player::One, &["the-statue-of-zeus"])
            .built(
                Player::One,
                &["press", "drying-room", "clay-pool", "quarry", "lumber-yard"],
            )
            .built(Player::Two, &["lumber-yard", "clay-pool"])
            .coins(Player::One, 20)
            .build();
        let mut rng = rng();
        apply(
            &mut st,
            Action::BuildWonder {
                slot: 18,
                wonder: WonderId::from_slug("the-statue-of-zeus").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(
            st.pending(),
            Some(Pending::Destroy {
                card_type: CardType::RawMaterial
            })
        );
        let target = CardId::from_slug("lumber-yard").unwrap();
        apply(
            &mut st,
            Action::DestroyOpponentCard { card: target },
            &mut rng,
        )
        .unwrap();
        assert!(!st.player(Player::Two).has_built(target));
        assert!(st.discard_pile().any(|c| c == target));
        // Player Two loses that wood production.
        assert_eq!(st.player(Player::Two).production()[0], 0);
    }

    #[test]
    fn a_destroy_effect_with_no_target_is_skipped() {
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "obelisk")])
            .wonders(Player::One, &["circus-maximus"])
            .built(Player::One, &["shelf-quarry", "glassworks", "lumber-yard"])
            .built(Player::Two, &["lumber-yard"]) // brown, not grey
            .coins(Player::One, 20)
            .build();
        let mut rng = rng();
        apply(
            &mut st,
            Action::BuildWonder {
                slot: 18,
                wonder: WonderId::from_slug("circus-maximus").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(st.pending(), None);
    }

    #[test]
    fn the_mausoleum_builds_from_the_discard_pile_for_free() {
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "obelisk")])
            .wonders(Player::One, &["the-mausoleum"])
            .built(
                Player::One,
                &["press", "glassworks", "glassblower", "brickyard"],
            )
            .discard(&["palace", "tavern"])
            .coins(Player::One, 0)
            .build();
        let mut rng = rng();
        apply(
            &mut st,
            Action::BuildWonder {
                slot: 18,
                wonder: WonderId::from_slug("the-mausoleum").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(st.pending(), Some(Pending::MausoleumBuild));
        let palace = CardId::from_slug("palace").unwrap();
        apply(&mut st, Action::MausoleumBuild { card: palace }, &mut rng).unwrap();
        assert!(st.player(Player::One).has_built(palace));
        assert!(!st.discard_pile().any(|c| c == palace));
        assert_eq!(st.player(Player::One).coins(), 0, "free of charge");
    }

    #[test]
    fn the_great_library_draws_three_of_the_set_aside_tokens() {
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "obelisk")])
            .wonders(Player::One, &["the-great-library"])
            .built(
                Player::One,
                &["press", "glassworks", "sawmill", "lumber-yard"],
            )
            .set_aside_tokens(&["philosophy", "agriculture", "mathematics", "law", "economy"])
            .board_tokens(&["strategy"])
            .coins(Player::One, 20)
            .build();
        let mut rng = rng();
        apply(
            &mut st,
            Action::BuildWonder {
                slot: 18,
                wonder: WonderId::from_slug("the-great-library").unwrap(),
            },
            &mut rng,
        )
        .unwrap();
        let Some(Pending::GreatLibraryToken { tokens }) = st.pending() else {
            panic!("expected a Great Library draw, got {:?}", st.pending());
        };
        // All three came from the out-of-play pile, not the board.
        for t in tokens {
            assert!(st.set_aside_tokens().any(|x| x == t));
            assert!(!st.board_tokens().any(|x| x == t));
        }
        let chosen = tokens[0];
        apply(
            &mut st,
            Action::ChooseGreatLibraryToken { token: chosen },
            &mut rng,
        )
        .unwrap();
        assert!(st.player(Player::One).has_token(chosen));
        // The other two are gone for good.
        assert_eq!(st.set_aside_tokens_mask().count_ones(), 2);
        assert!(!st.set_aside_tokens().any(|x| x == chosen));
    }

    #[test]
    fn chance_outcome_probabilities_sum_to_one() {
        let mut st = new_game(11);
        let mut rng = rng();
        for _ in 0..8 {
            let a = legal_actions(&st)[0];
            apply(&mut st, a, &mut rng).unwrap();
        }
        for _ in 0..30 {
            let actions = legal_actions(&st);
            if actions.is_empty() {
                break;
            }
            let a = actions[0];
            let outcomes = chance_outcomes(&st, a);
            let total: f64 = outcomes.iter().map(|(_, p)| p).sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "probabilities summed to {total} for {a:?}"
            );
            assert!(outcomes.iter().all(|(_, p)| *p > 0.0));
            apply(&mut st, a, &mut rng).unwrap();
        }
    }

    /// Walk seeded games and hand every chance action of every position to
    /// `check`, together with the position it came from. Asserts the walk
    /// actually reached Age III chance nodes with a purple back among the
    /// slots being uncovered, since that is the only place R-110 has teeth.
    fn for_each_chance_action(
        seeds: std::ops::Range<u64>,
        mut check: impl FnMut(&GameState, Action),
    ) {
        let mut rng = rng();
        let mut purple_backed_reveals = 0;
        for seed in seeds {
            let mut st = new_game(seed);
            for _ in 0..400 {
                let legal = legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                let mask = hidden_info(&st).hidden_guild_slots;
                for &a in &legal {
                    let slots = slots_revealed_by(&st, a);
                    if slots.is_empty() {
                        continue;
                    }
                    purple_backed_reveals +=
                        slots.iter().filter(|&&s| mask & (1u32 << s) != 0).count();
                    check(&st, a);
                }
                let a = legal[(seed as usize + 7) % legal.len()];
                apply(&mut st, a, &mut rng).unwrap();
            }
        }
        assert!(
            purple_backed_reveals > 0,
            "the walk never uncovered a purple-backed slot, so it proves nothing"
        );
    }

    /// R-105 / R-110. A guild's purple card back is public, so a guild-backed
    /// slot can only reveal a guild and a plain-backed slot can only reveal an
    /// ordinary card. Before this was fixed, `chance_outcomes` reasoned from
    /// the guild *count* and offered an ordinary card for a purple-backed slot
    /// at nonzero probability.
    #[test]
    fn chance_outcomes_respect_the_public_guild_mask() {
        for_each_chance_action(0..12, |st, a| {
            let mask = hidden_info(st).hidden_guild_slots;
            let outcomes = chance_outcomes(st, a);
            let total: f64 = outcomes.iter().map(|(_, p)| p).sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "probabilities summed to {total} for {a:?}"
            );
            for (o, p) in &outcomes {
                assert!(*p > 0.0, "a zero-probability outcome: {o:?}");
                for (slot, card) in o.reveals.iter().flatten().copied() {
                    assert_eq!(
                        card.def().is_guild(),
                        mask & (1u32 << slot) != 0,
                        "slot {slot} offers {card:?} against a mask of {mask:#b}"
                    );
                }
            }
        });
    }

    /// R-105. The distribution is computed from public information only, so
    /// two determinizations of the same `Observation` must produce the same
    /// one — bit-for-bit on the probabilities, not merely the same support.
    /// This is what makes conditioning on the guild mask a sharpening rather
    /// than a leak.
    #[test]
    fn chance_outcomes_are_invariant_across_determinizations() {
        for_each_chance_action(20..28, |st, a| {
            let obs = st.observation();
            let mut ra = StdRng::seed_from_u64(0xA11CE);
            let mut rb = StdRng::seed_from_u64(0xB0B);
            let (da, db) = (obs.sample_state(&mut ra), obs.sample_state(&mut rb));
            let (oa, ob) = (chance_outcomes(&da, a), chance_outcomes(&db, a));
            assert_eq!(oa.len(), ob.len(), "different support sizes for {a:?}");
            for ((ea, pa), (eb, pb)) in oa.iter().zip(ob.iter()) {
                assert_eq!(ea, eb, "different outcome enumerated for {a:?}");
                assert_eq!(
                    pa.to_bits(),
                    pb.to_bits(),
                    "different probability for {ea:?}"
                );
            }
        });
    }

    /// R-105 / R-110. Forcing an outcome re-derives the hidden layout, and
    /// that re-derivation must not move a guild between two slots that both
    /// stay face down: the mask is public, so shifting it would silently
    /// rewrite public information.
    #[test]
    fn forcing_an_outcome_preserves_the_public_guild_mask() {
        for_each_chance_action(30..38, |st, a| {
            let before = hidden_info(st).hidden_guild_slots;
            let age = st.age();
            // Every 13th outcome: the support runs to hundreds for a two-slot
            // reveal, and forcing each one is not free.
            for (o, _) in chance_outcomes(st, a).into_iter().step_by(13) {
                let mut next = *st;
                apply_with_outcome(&mut next, a, &o).expect("an enumerated outcome is forceable");
                if next.age() != age {
                    // The age ended and the whole structure was re-dealt;
                    // there is no old mask to compare against.
                    continue;
                }
                let after = hidden_info(&next);
                assert_eq!(
                    after.hidden_guild_slots,
                    before & after.hidden_slots,
                    "the guild mask moved on slots that stayed face down, forcing {o:?}"
                );
                assert_eq!(
                    after.hidden_guild_slots.count_ones(),
                    after.hidden_guild_count,
                    "the mask and the guild count disagree after forcing {o:?}"
                );
            }
        });
    }

    #[test]
    fn apply_with_outcome_forces_the_reveal() {
        let mut st = new_game(5);
        let mut rng = rng();
        // Play forward until some legal action would actually uncover a
        // face-down card (in the Age I pyramid both coverers must be gone).
        let mut action = None;
        for _ in 0..40 {
            let actions = legal_actions(&st);
            assert!(!actions.is_empty());
            if let Some(a) = actions
                .iter()
                .copied()
                .find(|a| !slots_revealed_by(&st, *a).is_empty())
            {
                action = Some(a);
                break;
            }
            apply(&mut st, actions[0], &mut rng).unwrap();
        }
        let action = action.expect("some move must uncover a face-down card");
        let outcomes = chance_outcomes(&st, action);
        // Pick an outcome that is not what the state currently hides.
        let (outcome, _) = outcomes
            .iter()
            .find(|(o, _)| {
                o.reveals
                    .iter()
                    .flatten()
                    .any(|&(slot, card)| st.slot_card_hidden(slot) != card)
            })
            .expect("more than one card could be behind the slot");
        let mut forced = st;
        apply_with_outcome(&mut forced, action, outcome).unwrap();
        for &(slot, card) in outcome.reveals.iter().flatten() {
            assert_eq!(forced.face_up_card(slot), Some(card));
        }
        // The forced layout is still a permutation of the age's cards.
        let mut seen = 0u128;
        for &c in forced.age_deck(forced.age()).iter() {
            seen |= 1u128 << c.index();
        }
        assert_eq!(seen.count_ones(), 20);
        assert_eq!(seen & forced.out_of_game_mask(), 0);
    }

    /// The search-path fast variant must be indistinguishable from
    /// [`apply_with_outcome`] in its effect on the state; it only skips the
    /// event log and the legality re-check.
    #[test]
    fn apply_with_outcome_unchecked_matches_apply_with_outcome() {
        let mut rng = rng();
        for seed in 0..6u64 {
            let mut st = new_game(seed);
            for _ in 0..60 {
                let actions = legal_actions(&st);
                if actions.is_empty() {
                    break;
                }
                for &a in actions.iter().take(3) {
                    for (outcome, _) in chance_outcomes(&st, a).into_iter().take(4) {
                        let mut slow = st;
                        let mut fast = st;
                        let slow_res = apply_with_outcome(&mut slow, a, &outcome);
                        let fast_res = apply_with_outcome_unchecked(&mut fast, a, &outcome);
                        assert_eq!(slow_res.is_ok(), fast_res.is_ok());
                        assert_eq!(slow, fast, "seed {seed}, action {a:?}");
                    }
                }
                let a = actions[(seed as usize + st.turn() as usize) % actions.len()];
                apply(&mut st, a, &mut rng).unwrap();
            }
        }
    }

    #[test]
    fn illegal_actions_are_rejected_without_changing_the_state() {
        let st = new_game(1);
        let mut copy = st;
        let mut rng = rng();
        let err = apply(&mut copy, Action::Build { slot: 0 }, &mut rng).unwrap_err();
        assert_eq!(err.action, Action::Build { slot: 0 });
        assert_eq!(copy, st);
    }

    #[test]
    fn unaffordable_builds_are_not_offered() {
        // Pretorium costs 8 coins flat.
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "pretorium")])
            .coins(Player::One, 7)
            .build();
        assert_eq!(legal_actions(&st), vec![Action::Discard { slot: 18 }]);
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "pretorium")])
            .coins(Player::One, 8)
            .build();
        assert!(legal_actions(&st).contains(&Action::Build { slot: 18 }));
    }

    #[test]
    fn covered_slots_are_never_offered() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(9, "quarry"), (14, "clay-pool"), (15, "lumber-yard")])
            .coins(Player::One, 20)
            .build();
        let slots: Vec<u8> = legal_actions(&st)
            .into_iter()
            .filter_map(|a| match a {
                Action::Build { slot } => Some(slot),
                _ => None,
            })
            .collect();
        assert_eq!(slots, vec![14, 15]);
    }

    #[test]
    fn economy_redirects_the_opponents_trade_payments() {
        // Player One must buy 2 stone at 2 coins each for the Baths-like
        // cost of Aqueduct (3 stone); they produce one.
        let mut st = StateBuilder::new()
            .age(2)
            .open_slots(&[(18, "aqueduct")])
            .built(Player::One, &["quarry"])
            .tokens(Player::Two, &["economy"])
            .coins(Player::One, 10)
            .coins(Player::Two, 0)
            .build();
        let mut rng = rng();
        apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(st.player(Player::One).coins(), 6);
        assert_eq!(st.player(Player::Two).coins(), 4, "trade goes to Economy");
    }

    #[test]
    fn a_cards_printed_coin_cost_is_not_redirected_by_economy() {
        let mut st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "scriptorium")]) // 2 coins, no resources
            .tokens(Player::Two, &["economy"])
            .coins(Player::One, 10)
            .coins(Player::Two, 0)
            .build();
        let mut rng = rng();
        apply(&mut st, Action::Build { slot: 14 }, &mut rng).unwrap();
        assert_eq!(st.player(Player::One).coins(), 8);
        assert_eq!(st.player(Player::Two).coins(), 0);
    }

    /// A trading post's coins go to the bank, not the opponent — this is the
    /// bug a playtester reported when the UI text for Stone/Clay/Wood Reserve
    /// wrongly implied "buy from your opponent". Player Two produces stone
    /// (so an open-market price would be inflated by their production) but
    /// holds no Economy token, so none of Player One's trade payment should
    /// ever reach them: the coins are simply spent, same as any other cost.
    #[test]
    fn a_trading_post_pays_the_bank_not_the_opponent() {
        let mut st = StateBuilder::new()
            .age(2)
            .open_slots(&[(18, "aqueduct")]) // 3 stone
            .built(Player::One, &["stone-reserve"])
            .built(Player::Two, &["shelf-quarry"]) // produces 2 stone
            .coins(Player::One, 10)
            .coins(Player::Two, 5)
            .build();
        let mut rng = rng();
        apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        // Stone Reserve fixes the price at 1/unit regardless of Player Two's
        // production: 3 coins total, all to the bank.
        assert_eq!(st.player(Player::One).coins(), 7);
        assert_eq!(
            st.player(Player::Two).coins(),
            5,
            "a trading post's payment never reaches the opponent without Economy"
        );
    }

    #[test]
    fn yellow_age_three_cards_count_their_own_colour_including_themselves() {
        // Lighthouse pays 1 coin per yellow card owned, and it is yellow.
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "lighthouse")])
            .built(Player::One, &["tavern", "brewery"])
            .coins(Player::One, 20)
            .build();
        let mut rng = rng();
        apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        // Tavern chains into Lighthouse, so the build is free; 3 yellow cards
        // afterwards.
        assert_eq!(st.player(Player::One).coins(), 23);
    }

    #[test]
    fn a_guild_pays_coins_on_the_count_at_the_moment_it_is_built() {
        // Player Two has 3 yellow cards; Player One has 1. The Merchants
        // Guild pays One 3 coins now. Building more yellow later changes the
        // final point total but not the coins already banked.
        let mut st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "merchants-guild")])
            .built(Player::One, &["tavern"])
            .built(Player::Two, &["brewery", "forum", "caravansery"])
            .coins(Player::One, 20)
            .build();
        let mut rng = rng();
        let ev = apply(&mut st, Action::Build { slot: 18 }, &mut rng).unwrap();
        let gained: u16 = ev
            .iter()
            .filter_map(|e| match e {
                Event::CoinsGained {
                    amount,
                    reason: CoinReason::GuildMajority,
                    ..
                } => Some(*amount),
                _ => None,
            })
            .sum();
        assert_eq!(gained, 3);
    }
}
