//! The military race read: how far is this player from military supremacy,
//! how likely are they to get there, and how soon?
//!
//! Military supremacy is the pawn reaching the opponent's capital at
//! `data::military().capital_distance` (9 in the base game). What makes it a
//! *race* rather than a score is that the shields which push the pawn are a
//! finite, contested supply: they sit on a handful of red cards, and taking a
//! red card is the only way to deny it. [`military_read`] measures both sides
//! of that: how many shields the player still needs, and how many are
//! reachable — now, on the visible table, and in expectation from what has
//! not been dealt.
//!
//! # The magnitude
//!
//! [`MilitaryRead::magnitude`] plays the race out. Each round the
//! threat-holder takes the best shield source they can reach — starting with
//! however many play-again wonders they can chain, whose shields no opponent
//! can take away — and the defender answers with whichever single reply hurts
//! more: taking the threat-holder's best remaining red card, or pushing the
//! pawn back the other way. Whatever red cards are still to be uncovered or
//! dealt arrive as a per-round stream, split by each side's share of the
//! remaining decisions. If the pawn reaches the capital on round `t`, the
//! magnitude is `turn_discount^(t - 1)`; if it never does, it is zero.
//!
//! That makes [`MilitaryStatus`] a label rather than a judgement:
//! `Imminent` is exactly `magnitude == 1`, which is exactly "closes this
//! round" — one card, or a chained sequence the defender never gets to
//! interrupt.

use duels_core::data::{self, CardId, WonderId};
use duels_core::{GameState, Player};

use crate::board::{iter_slots, Board};
use crate::context::Context;
use crate::masks::masks;
use crate::prices::Prices;
use crate::tempo::{grants_extra_turn, holds_theology, ThreatWeights};

/// How reachable military supremacy is for one player.
///
/// Purely a label on [`MilitaryRead::magnitude`], for logs and test
/// assertions. Nothing in the weighting math branches on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilitaryStatus {
    /// Certain if unopposed: the capital is reached on this player's very next
    /// turn (`magnitude == 1`). Check [`MilitaryRead::undeniable`] to see
    /// whether the opponent could take that action's card away first.
    Imminent,
    /// A real race: `magnitude >= ThreatWeights::live_threshold`.
    Live,
    /// Not reachable soon enough, or at all, to be worth calling a race.
    Closed,
}

/// Where a shield the player could take *right now* comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShieldSource {
    /// An accessible, affordable red card in the structure.
    Card {
        /// Its slot.
        slot: u8,
        /// The card itself.
        card: CardId,
        /// Shields it would add, including the Strategy token's bonus.
        shields: u8,
    },
    /// A drafted, unbuilt, affordable wonder that grants shields. The opponent
    /// cannot deny this one at all: it needs only some card to spend, not a
    /// specific one.
    Wonder {
        /// The wonder.
        wonder: WonderId,
        /// Shields it would add.
        shields: u8,
        /// Whether building it also grants an extra turn, so its shields and
        /// the next take both land before the opponent replies.
        play_again: bool,
    },
}

impl ShieldSource {
    /// Shields this source would add.
    #[inline]
    pub fn shields(&self) -> u8 {
        match self {
            ShieldSource::Card { shields, .. } | ShieldSource::Wonder { shields, .. } => *shields,
        }
    }
}

/// The largest number of distinct shield sources a position can offer: six
/// accessible slots in Age I plus four drafted wonders, with headroom.
pub const MAX_SHIELD_SOURCES: usize = 12;

/// One band of the end-of-game military scoring table, seen from one player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MilitaryBand {
    /// The smallest pawn distance that scores this band.
    pub distance: u8,
    /// Victory points the band awards.
    pub victory_points: u8,
    /// Shields the player must still generate to reach it. Zero if the pawn is
    /// already at or past `distance` in their favour.
    pub shields_needed: u8,
    /// Victory points this band would add over what the pawn currently scores
    /// for this player.
    pub vp_gain: u8,
}

/// A small descending shield stack: the simulation's working set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShieldStack {
    shields: [u8; MAX_SHIELD_SOURCES],
    len: usize,
}

impl ShieldStack {
    /// An empty stack.
    pub const EMPTY: ShieldStack = ShieldStack {
        shields: [0; MAX_SHIELD_SOURCES],
        len: 0,
    };

    /// Add one source. Silently ignores anything past the capacity, which the
    /// game's own card counts never reach.
    pub fn push(&mut self, shields: u8) {
        if self.len < MAX_SHIELD_SOURCES {
            self.shields[self.len] = shields;
            self.len += 1;
        }
    }

    /// Sort descending, so `pop_largest` is a cheap decrement.
    fn sort(&mut self) {
        self.shields[..self.len].sort_unstable_by(|a, b| b.cmp(a));
    }

    /// The largest remaining source, without removing it.
    #[inline]
    pub fn largest(&self) -> u8 {
        if self.len == 0 {
            0
        } else {
            self.shields[0]
        }
    }

    /// Remove and return the largest remaining source.
    #[inline]
    fn pop_largest(&mut self) -> Option<u8> {
        if self.len == 0 {
            return None;
        }
        let v = self.shields[0];
        self.shields.copy_within(1..self.len, 0);
        self.len -= 1;
        Some(v)
    }

    /// Remove one source of exactly `shields`, if there is one. Used to apply
    /// "the defender took that card" without rebuilding the read.
    fn remove_one(&mut self, shields: u8) -> bool {
        if let Some(i) = self.shields[..self.len].iter().position(|&s| s == shields) {
            self.shields.copy_within(i + 1..self.len, i);
            self.len -= 1;
            true
        } else {
            false
        }
    }

    /// How many sources there are.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// The `i`th source, or zero past the end.
    #[inline]
    pub fn at(&self, i: usize) -> u8 {
        if i < self.len {
            self.shields[i]
        } else {
            0
        }
    }

    /// Whether there are none.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The inputs [`MilModel::magnitude`] consumes, and the only thing an
/// incremental "what if the defender played this?" needs to touch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MilModel {
    /// Shields still needed to reach the capital.
    pub need: f64,
    /// How many extra turns the threat-holder can chain on their next turn.
    pub chain: u8,
    /// Whether one of those extra turns is already banked, and so needs no
    /// wonder built to pay for it.
    pub banked: bool,
    /// Shields on the play-again wonders they can afford, which land before
    /// the defender gets to reply.
    pub play_again: ShieldStack,
    /// Shields on the red cards they can reach — the only sources a defender
    /// can take away.
    pub cards: ShieldStack,
    /// Shields on the affordable wonders that do *not* grant an extra turn.
    /// Undeniable, but one per turn.
    pub wonders: ShieldStack,
    /// Average shields per card in the stream still to be uncovered or dealt.
    pub avg_stream: f64,
    /// Cards left in the current age's structure, which bounds a chained
    /// sequence.
    pub cards_left: u8,
    /// The threat-holder's remaining decision budget: the number of rounds to
    /// simulate.
    pub horizon: f64,
    /// The threat-holder's share of the remaining decisions.
    pub share: f64,
    /// The defender's share.
    pub share_defender: f64,
    /// The largest single push the defender can make the other way.
    pub defender_best_single: u8,
    /// Every push the defender can currently enumerate, largest first. The
    /// defender's counter-push is bounded by their *own* supply: past it, the
    /// share of the shared stream they take out of the threat-holder's hands
    /// is already netted off `avg_stream`, and charging both would let `need`
    /// grow without limit.
    pub defender_sources: ShieldStack,
    /// How much of a counter-push actually converts.
    pub counter_efficiency: f64,
    /// Per-round discount on the close.
    pub turn_discount: f64,
}

impl MilModel {
    /// Which of the threat-holder's own rounds the capital is reached on, if
    /// it is reached at all.
    pub fn turns_to_close(&self) -> Option<u8> {
        if self.need <= 0.0 {
            return Some(0);
        }
        let horizon = if self.horizon <= 0.0 {
            0u32
        } else {
            self.horizon.floor().min(64.0) as u32
        };
        let mut need = self.need;
        let mut acc = 0.0f64;
        let mut play_again = self.play_again;
        let mut cards = self.cards;
        let mut wonders = self.wonders;
        let mut defender = self.defender_sources;
        let mut cards_left = f64::from(self.cards_left);
        // The stream is what the shared supply is worth to this player per
        // round after the defender takes their cut. Never negative: shields
        // already banked are never lost.
        let stream = (self.avg_stream
            * (self.share - self.counter_efficiency * self.share_defender))
            .max(0.0);

        for round in 1..=horizon {
            let mut takes = 0.0f64;
            // Extra turns are a *right now* quantity: the wonders granting
            // them are built on this very next turn or not at all. So round
            // one is `1 + chain` consecutive actions, and every later round is
            // a single action the defender gets to answer.
            let (actions, mut wonder_steps) = if round == 1 {
                (
                    1 + u32::from(self.chain),
                    self.chain.saturating_sub(u8::from(self.banked)),
                )
            } else {
                (1, 0)
            };
            for _ in 0..actions {
                if cards_left - takes < 1.0 {
                    break;
                }
                // Each chained turn past a banked one has to be paid for by
                // building a play-again wonder, so those go first; their
                // shields cannot be denied.
                if wonder_steps > 0 {
                    match play_again.pop_largest() {
                        Some(s) => {
                            acc += f64::from(s);
                            takes += 1.0;
                            wonder_steps -= 1;
                            continue;
                        }
                        None => wonder_steps = 0,
                    }
                }
                if !wonders.is_empty() && wonders.largest() >= cards.largest() {
                    acc += f64::from(wonders.pop_largest().unwrap_or(0));
                    takes += 1.0;
                } else if let Some(s) = cards.pop_largest() {
                    acc += f64::from(s);
                    takes += 1.0;
                } else {
                    break;
                }
            }
            if acc >= need {
                return Some(u8::try_from(round).unwrap_or(u8::MAX));
            }

            // The defender's single reply: take the best remaining red card,
            // or push the pawn the other way, whichever costs more progress.
            let counter = f64::from(defender.largest()) * self.counter_efficiency;
            if f64::from(cards.largest()) >= counter {
                cards.pop_largest();
            } else if let Some(s) = defender.pop_largest() {
                need += f64::from(s) * self.counter_efficiency;
            }

            cards_left = (cards_left - takes - 1.0).max(0.0);
            acc += stream;
            if acc >= need {
                return Some(u8::try_from(round).unwrap_or(u8::MAX));
            }

            // Once neither side has an enumerated source left, every further
            // round is identical: the stream arrives and `need` no longer
            // moves. Close the tail in one division rather than looping to the
            // horizon, which is what keeps this cheap enough to re-run per
            // legal action.
            if cards.is_empty() && wonders.is_empty() && play_again.is_empty() {
                if stream <= 0.0 {
                    return None;
                }
                let more = ((need - acc) / stream).ceil();
                if !more.is_finite() || more < 0.0 {
                    return None;
                }
                let total = f64::from(round) + more;
                return if total <= self.horizon {
                    Some(u8::try_from(total as u32).unwrap_or(u8::MAX))
                } else {
                    None
                };
            }
        }
        None
    }

    /// `M_mil`: how likely this player is to reach the capital, discounted for
    /// how long it takes.
    pub fn magnitude(&self) -> f64 {
        match self.turns_to_close() {
            None => 0.0,
            Some(0) => 1.0,
            Some(t) => self.turn_discount.powi(i32::from(t) - 1).clamp(0.0, 1.0),
        }
    }

    /// The same model after the defender takes a red card worth `shields` from
    /// the threat-holder's reach.
    pub fn after_card_denied(&self, shields: u8) -> MilModel {
        let mut out = *self;
        out.cards.remove_one(shields);
        out
    }

    /// The same model after the defender gains `shields` of their own, which
    /// pushes the pawn back and raises the bar.
    pub fn after_counter_push(&self, shields: u8) -> MilModel {
        let mut out = *self;
        out.need += f64::from(shields);
        out
    }

    /// The same model after `shields` worth of red card is uncovered and
    /// handed to the threat-holder.
    pub fn after_card_exposed(&self, shields: u8) -> MilModel {
        let mut out = *self;
        out.cards.push(shields);
        out.cards.sort();
        out
    }
}

/// The military race, read for one player.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MilitaryRead {
    /// The player this read is about.
    pub player: Player,
    /// The pawn's distance from centre, signed so that positive favours
    /// `player`.
    pub distance: i8,
    /// Shields `player` must still generate to reach the opponent's capital.
    pub need: u8,
    /// Shields available to `player` right now: the sum over every source in
    /// [`MilitaryRead::sources`]. An upper bound on immediate supply — the
    /// sources are not all takeable on one turn.
    pub now: u8,
    /// The largest single-action shield gain available right now.
    pub best_single: u8,
    /// Shields printed on every face-up card still in the structure,
    /// accessible or not, without the Strategy bonus.
    pub visible: u8,
    /// Expected shields still to come from the current age's face-down slots.
    pub expected_hidden: f64,
    /// Expected shields from the ages that have not been dealt yet.
    pub expected_future_ages: f64,
    /// How many independent ways `player` has, right now, to advance the pawn
    /// at all. A card counts once and a shield-granting wonder counts once,
    /// because those are the units the opponent would have to deny — and a
    /// wonder cannot be denied at all.
    pub fork: u8,
    /// How many of those would, alone, reach the capital. This is the count
    /// that decides [`MilitaryRead::undeniable`]: two closing actions cannot
    /// both be taken away by one opposing turn.
    pub closing_fork: u8,
    /// Roughly how many more decisions `player` gets in the current age.
    pub tempo: u8,
    /// Roughly how many more decisions `player` gets in the whole game.
    pub decisions_left: u8,
    /// `M_mil`: the probability, discounted for tempo, that `player` reaches
    /// the capital.
    pub magnitude: f64,
    /// The magnitude model itself, so a caller can price one action against it
    /// without rebuilding the read.
    pub model: MilModel,
    /// The label on `magnitude`.
    pub status: MilitaryStatus,
    /// Whether a close on this player's very next turn cannot be prevented:
    /// two closing cards, or a closing wonder, which the opponent has no way
    /// to take away.
    pub undeniable: bool,
    /// Which of `player`'s own rounds the capital is reached on, per the
    /// simulation. `None` when it never is.
    pub turns_to_close: Option<u8>,
    /// Coins the opponent would forfeit when the pawn next crosses an
    /// uncollected loot token on `player`'s side, capped at the coins the
    /// opponent actually holds.
    pub loot_damage: u16,
    /// Shields needed to reach that loot token, if there is one left.
    pub loot_shields_needed: Option<u8>,
    /// The scoring bands, ascending.
    pub bands: [MilitaryBand; 4],
    /// Accessible slots whose card alone would reach the capital.
    pub closing_slots: u32,
    /// Bitmask over [`WonderId::index`] of wonders that alone would reach the
    /// capital.
    pub closing_wonders: u16,
    /// Accessible slots whose card would advance the pawn at all.
    pub advancing_slots: u32,
    /// Bitmask over [`WonderId::index`] of affordable unbuilt wonders that
    /// would advance the pawn.
    pub advancing_wonders: u16,
    sources: [Option<ShieldSource>; MAX_SHIELD_SOURCES],
}

impl MilitaryRead {
    /// The shield sources available right now.
    pub fn sources(&self) -> impl Iterator<Item = ShieldSource> + '_ {
        self.sources.iter().flatten().copied()
    }

    /// Total shields still obtainable this age, visible or expected.
    #[inline]
    pub fn age_supply(&self) -> f64 {
        f64::from(self.visible) + self.expected_hidden
    }

    /// The shields of the accessible red card in `slot`, if that card is one
    /// of this player's reachable sources.
    pub fn card_source_shields(&self, slot: u8) -> Option<u8> {
        self.sources().find_map(|s| match s {
            ShieldSource::Card {
                slot: at, shields, ..
            } if at == slot => Some(shields),
            _ => None,
        })
    }
}

/// The pawn's distance from centre, signed so that positive favours `player`.
#[inline]
pub fn signed_distance(state: &GameState, player: Player) -> i8 {
    match player {
        Player::One => state.conflict(),
        Player::Two => -state.conflict(),
    }
}

/// The half of a military read that does not depend on the *other* player's
/// position, computed once per player by [`Context`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MilBase {
    pub sources: [Option<ShieldSource>; MAX_SHIELD_SOURCES],
    pub count: usize,
    pub now: u16,
    pub best_single: u8,
    pub closing_slots: u32,
    pub closing_wonders: u16,
    pub advancing_slots: u32,
    pub advancing_wonders: u16,
    pub need: u8,
    pub distance: i8,
    pub strategy: bool,
    pub cards: ShieldStack,
    pub wonders: ShieldStack,
    pub play_again: ShieldStack,
}

impl MilBase {
    /// Gather every shield source `player` can act on right now.
    pub(crate) fn of(state: &GameState, player: Player, board: &Board, prices: &Prices) -> MilBase {
        let m = masks();
        let track = data::military();
        let me = state.player(player);

        let distance = signed_distance(state, player);
        let cap = i8::try_from(track.capital_distance).unwrap_or(9);
        let need = u8::try_from((i16::from(cap) - i16::from(distance)).max(0)).unwrap_or(u8::MAX);

        let strategy = m
            .strategy_token()
            .is_some_and(|t| me.tokens().any(|held| held == t));
        let theology = holds_theology(state, player);

        let mut out = MilBase {
            sources: [None; MAX_SHIELD_SOURCES],
            count: 0,
            now: 0,
            best_single: 0,
            closing_slots: 0,
            closing_wonders: 0,
            advancing_slots: 0,
            advancing_wonders: 0,
            need,
            distance,
            strategy,
            cards: ShieldStack::EMPTY,
            wonders: ShieldStack::EMPTY,
            play_again: ShieldStack::EMPTY,
        };

        // Only the accessible red cards can be a shield source, and the board
        // digest already knows which slots those are.
        for slot in iter_slots(board.accessible & board.shield_slots & prices.affordable_now) {
            let Some(card) = board.slot_card[slot as usize] else {
                continue;
            };
            let def = card.def();
            // The engine grants Strategy's extra shield to red cards only, and
            // every shield-bearing card in the data is red (asserted in
            // `masks::tests::every_shield_bearing_card_is_red`).
            let shields = def.shields + u8::from(strategy);
            out.advancing_slots |= 1u32 << slot;
            if u16::from(shields) >= u16::from(need) && need > 0 {
                out.closing_slots |= 1u32 << slot;
            }
            out.now += u16::from(shields);
            out.best_single = out.best_single.max(shields);
            out.cards.push(shields);
            out.push(ShieldSource::Card {
                slot,
                card,
                shields,
            });
        }

        // A wonder needs some card from the structure to spend, and a free
        // wonder slot; which card does not matter, so the opponent cannot deny
        // it.
        if prices.can_build_wonder {
            for wonder in me.wonders() {
                let def = wonder.def();
                if me.has_built_wonder(wonder) || !prices.can_afford_wonder(wonder) {
                    continue;
                }
                let play_again = grants_extra_turn(wonder, theology);
                if def.shields == 0 {
                    // A shieldless play-again wonder is still a tempo source:
                    // it buys the turn on which a red card gets taken.
                    if play_again {
                        out.play_again.push(0);
                    }
                    continue;
                }
                out.advancing_wonders |= 1u16 << wonder.index();
                if u16::from(def.shields) >= u16::from(need) && need > 0 {
                    out.closing_wonders |= 1u16 << wonder.index();
                }
                out.now += u16::from(def.shields);
                out.best_single = out.best_single.max(def.shields);
                if play_again {
                    out.play_again.push(def.shields);
                } else {
                    out.wonders.push(def.shields);
                }
                out.push(ShieldSource::Wonder {
                    wonder,
                    shields: def.shields,
                    play_again,
                });
            }
        }
        out.cards.sort();
        out.wonders.sort();
        out.play_again.sort();
        out
    }

    /// Every push this player can enumerate right now, largest first: cards,
    /// wonders and play-again wonders together.
    pub(crate) fn all_sources(&self) -> ShieldStack {
        let mut out = self.cards;
        for stack in [&self.wonders, &self.play_again] {
            for i in 0..stack.len() {
                out.push(stack.at(i));
            }
        }
        out.sort();
        out
    }

    fn push(&mut self, src: ShieldSource) {
        if self.count < MAX_SHIELD_SOURCES {
            self.sources[self.count] = Some(src);
            self.count += 1;
        }
    }
}

/// Read the military race for `player`.
///
/// Reads only public information, so two determinizations of the same position
/// produce identical results.
pub fn military_read(state: &GameState, player: Player) -> MilitaryRead {
    military_read_with(state, player, &Context::of(state))
}

/// [`military_read`] against a [`Context`] the caller already built.
pub fn military_read_with(state: &GameState, player: Player, ctx: &Context) -> MilitaryRead {
    let m = masks();
    let track = data::military();
    let opp = state.player(player.other());
    let board = &ctx.board;
    let w = &ctx.weights;
    let base = ctx.military_base(player);
    let defender = ctx.military_base(player.other());
    let tempo = ctx.tempo(player);
    let defender_tempo = ctx.tempo(player.other());

    let need = base.need;
    let distance = base.distance;

    // --- what is still out there ------------------------------------------
    let visible = u8::try_from(crate::masks::shields_in(board.face_up).min(u16::from(u8::MAX)))
        .unwrap_or(u8::MAX);
    let expected_hidden = ctx.expected.hidden_shields;
    let expected_hidden_cards = ctx.expected.hidden_shield_cards;
    let expected_future_ages = ctx.expected.future_shields;
    let expected_future_cards = ctx.expected.future_shield_cards;

    // Shield cards that are visible but not takeable this turn (covered, or
    // unaffordable right now), plus the expected hidden and undealt ones.
    let immediate_cards: u128 = base
        .sources
        .iter()
        .flatten()
        .filter_map(|s| match s {
            ShieldSource::Card { card, .. } => Some(1u128 << card.index()),
            ShieldSource::Wonder { .. } => None,
        })
        .fold(0u128, |a, b| a | b);
    let later_visible = board.face_up & m.any_shield_mask() & !immediate_cards;
    let stream_cards =
        f64::from(later_visible.count_ones()) + expected_hidden_cards + expected_future_cards;
    let stream_shields =
        f64::from(crate::masks::shields_in(later_visible)) + expected_hidden + expected_future_ages;

    let model = MilModel {
        need: f64::from(need),
        chain: tempo.chain,
        banked: tempo.banked,
        play_again: base.play_again,
        cards: base.cards,
        wonders: base.wonders,
        avg_stream: if stream_cards > 0.0 {
            stream_shields / stream_cards
        } else {
            0.0
        },
        cards_left: board.cards_left(),
        horizon: tempo.decisions_left_eff,
        share: tempo.share,
        share_defender: defender_tempo.share,
        defender_best_single: defender.best_single,
        defender_sources: defender.all_sources(),
        counter_efficiency: w.counter_efficiency,
        turn_discount: w.turn_discount,
    };
    let turns_to_close = model.turns_to_close();
    let magnitude = model.magnitude();
    let status = status_for(magnitude, w);

    let fork = u8::try_from(base.count).unwrap_or(u8::MAX);
    let closing_fork =
        u8::try_from(base.closing_slots.count_ones() + base.closing_wonders.count_ones())
            .unwrap_or(u8::MAX);
    // A shield-granting wonder cannot be taken away, so one alone is already
    // undeniable.
    let undeniable =
        status == MilitaryStatus::Imminent && (closing_fork >= 2 || base.closing_wonders != 0);

    // --- loot and scoring bands -------------------------------------------
    let mut loot_damage = 0u16;
    let mut loot_shields_needed = None;
    for (i, &(d, coins)) in track.loot.iter().enumerate() {
        if state.loot_available(player, i) {
            loot_damage = u16::from(coins).min(opp.coins());
            loot_shields_needed =
                Some(u8::try_from((i16::from(d) - i16::from(distance)).max(0)).unwrap_or(u8::MAX));
            break;
        }
    }

    let current_vp = if distance > 0 {
        track.vp_for_distance(distance.unsigned_abs())
    } else {
        0
    };
    let bands: [MilitaryBand; 4] = std::array::from_fn(|i| {
        let (_max, vp) = track.victory_points[i];
        let entry = if i == 0 {
            0
        } else {
            track.victory_points[i - 1].0.saturating_add(1)
        };
        MilitaryBand {
            distance: entry,
            victory_points: vp,
            shields_needed: u8::try_from((i16::from(entry) - i16::from(distance)).max(0))
                .unwrap_or(u8::MAX),
            vp_gain: vp.saturating_sub(current_vp),
        }
    });

    MilitaryRead {
        player,
        distance,
        need,
        now: u8::try_from(base.now.min(u16::from(u8::MAX))).unwrap_or(u8::MAX),
        best_single: base.best_single,
        visible,
        expected_hidden,
        expected_future_ages,
        fork,
        closing_fork,
        tempo: tempo.picks_in_age,
        decisions_left: tempo.decisions_left,
        magnitude,
        model,
        status,
        undeniable,
        turns_to_close,
        loot_damage,
        loot_shields_needed,
        bands,
        closing_slots: base.closing_slots,
        closing_wonders: base.closing_wonders,
        advancing_slots: base.advancing_slots,
        advancing_wonders: base.advancing_wonders,
        sources: base.sources,
    }
}

/// The label for a military magnitude, for callers that compute one directly.
pub fn status_for(magnitude: f64, w: &ThreatWeights) -> MilitaryStatus {
    if magnitude >= 1.0 {
        MilitaryStatus::Imminent
    } else if magnitude >= w.live_threshold {
        MilitaryStatus::Live
    } else {
        MilitaryStatus::Closed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::testing::StateBuilder;

    #[test]
    fn need_is_read_off_the_real_track_from_both_sides() {
        let cap = i8::try_from(data::military().capital_distance).unwrap();
        for conflict in [-9i8, -3, 0, 4, 8] {
            let st = StateBuilder::new().conflict(conflict).build();
            let one = military_read(&st, Player::One);
            let two = military_read(&st, Player::Two);
            assert_eq!(one.distance, conflict);
            assert_eq!(two.distance, -conflict);
            assert_eq!(i16::from(one.need), i16::from(cap) - i16::from(conflict));
            assert_eq!(i16::from(two.need), i16::from(cap) + i16::from(conflict));
        }
    }

    #[test]
    fn the_scoring_bands_come_from_the_data() {
        let st = StateBuilder::new().conflict(0).build();
        let r = military_read(&st, Player::One);
        let want: Vec<(u8, u8)> = r
            .bands
            .iter()
            .map(|b| (b.distance, b.victory_points))
            .collect();
        assert_eq!(want, vec![(0, 0), (1, 2), (3, 5), (6, 10)]);
        // From the centre, reaching the 10-point band needs six shields.
        assert_eq!(r.bands[3].shields_needed, 6);
        assert_eq!(r.bands[3].vp_gain, 10);
    }

    #[test]
    fn a_shield_stack_keeps_its_order_and_removes_by_value() {
        let mut s = ShieldStack::EMPTY;
        for v in [1u8, 3, 2] {
            s.push(v);
        }
        s.sort();
        assert_eq!(s.largest(), 3);
        assert!(s.remove_one(2));
        assert!(!s.remove_one(2));
        assert_eq!(s.len(), 2);
        assert_eq!(s.pop_largest(), Some(3));
        assert_eq!(s.pop_largest(), Some(1));
        assert_eq!(s.pop_largest(), None);
        assert!(s.is_empty());
    }
}
