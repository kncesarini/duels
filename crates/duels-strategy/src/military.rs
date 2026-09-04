//! The military race read: how far is this player from military supremacy,
//! and can they actually get there?
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
//! # What "unopposed" means here
//!
//! [`MilitaryStatus::Imminent`] is an *if-unopposed* judgement: the player has
//! one legal, affordable action that alone reaches the capital. It says
//! nothing about whether the opponent gets to move first — that is precisely
//! why [`MilitaryRead::fork`] is reported alongside it. A fork of two or more
//! means there are two independent ways to close, and the opponent cannot take
//! both cards with one turn.
//!
//! [`MilitaryRead::turns_to_close`] is likewise deliberately optimistic: it
//! assumes the player takes the best shield source available on each of their
//! own decisions and is never outbid. A pessimistic estimate would collapse
//! almost every race to `Closed` and defeat the purpose of the read, which is
//! to tell a search *where to look*.

use duels_core::data::{self, CardId, WonderId};
use duels_core::{cost, GameState, Player};

use crate::board::{iter_slots, Board};
use crate::masks::{masks, AgeSupply, DECISIONS_PER_AGE};

/// How reachable military supremacy is for one player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilitaryStatus {
    /// One legal, affordable action reaches the capital on this player's very
    /// next move. Check [`MilitaryRead::undeniable`] to see whether the
    /// opponent could take that action's card away first.
    Imminent,
    /// Not this move, but reachable: [`MilitaryRead::turns_to_close`] holds
    /// the estimate, in this player's own decisions.
    Live,
    /// Not reachable within the decisions this player has left, given every
    /// shield still on the table and every shield still expected to be dealt.
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
    /// The largest single-action shield gain available right now. This, not
    /// `now`, is what decides [`MilitaryStatus::Imminent`].
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
    /// The classification.
    pub status: MilitaryStatus,
    /// Whether an [`MilitaryStatus::Imminent`] close cannot be prevented:
    /// `closing_fork >= 2`, or a closing wonder, which the opponent has no way
    /// to take away.
    pub undeniable: bool,
    /// Estimated number of `player`'s own decisions to reach the capital, if
    /// they are never outbid. `None` when the status is
    /// [`MilitaryStatus::Closed`].
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
}

/// The pawn's distance from centre, signed so that positive favours `player`.
#[inline]
pub fn signed_distance(state: &GameState, player: Player) -> i8 {
    match player {
        Player::One => state.conflict(),
        Player::Two => -state.conflict(),
    }
}

/// Read the military race for `player`.
///
/// Reads only public information, so two determinizations of the same position
/// produce identical results.
pub fn military_read(state: &GameState, player: Player) -> MilitaryRead {
    military_read_with(state, player, &Board::of(state))
}

/// [`military_read`] against a [`Board`] the caller already built.
pub fn military_read_with(state: &GameState, player: Player, board: &Board) -> MilitaryRead {
    let m = masks();
    let track = data::military();
    let me = state.player(player);
    let opp = state.player(player.other());

    let distance = signed_distance(state, player);
    let cap = i8::try_from(track.capital_distance).unwrap_or(9);
    let need = u8::try_from((i16::from(cap) - i16::from(distance)).max(0)).unwrap_or(u8::MAX);

    let strategy = m
        .strategy_token()
        .is_some_and(|t| me.tokens().any(|held| held == t));

    // --- what is takeable right now ---------------------------------------
    let mut sources = [None; MAX_SHIELD_SOURCES];
    let mut count = 0usize;
    let mut closing_slots = 0u32;
    let mut advancing_slots = 0u32;
    let mut now = 0u16;
    let mut best_single = 0u8;

    let push = |src: ShieldSource, sources: &mut [Option<ShieldSource>], count: &mut usize| {
        if *count < MAX_SHIELD_SOURCES {
            sources[*count] = Some(src);
            *count += 1;
        }
    };

    for slot in iter_slots(board.accessible) {
        let Some(card) = board.slot_card[slot as usize] else {
            continue;
        };
        let def = card.def();
        if def.shields == 0 {
            continue;
        }
        // The engine grants Strategy's extra shield to red cards only, and
        // every shield-bearing card in the data is red (asserted in
        // `masks::tests::every_shield_bearing_card_is_red`).
        let shields = def.shields + u8::from(strategy);
        if !cost::card_cost(state, player, card).affordable_by(state, player) {
            continue;
        }
        advancing_slots |= 1u32 << slot;
        if u16::from(shields) >= u16::from(need) && need > 0 {
            closing_slots |= 1u32 << slot;
        }
        now += u16::from(shields);
        best_single = best_single.max(shields);
        push(
            ShieldSource::Card {
                slot,
                card,
                shields,
            },
            &mut sources,
            &mut count,
        );
    }

    // A wonder needs some card from the structure to spend, and a free wonder
    // slot; which card does not matter, so the opponent cannot deny it.
    let wonders_possible = board.accessible != 0 && state.wonder_slots_left();
    let mut closing_wonders = 0u16;
    let mut advancing_wonders = 0u16;
    if wonders_possible {
        for wonder in me.wonders() {
            let def = wonder.def();
            if def.shields == 0 || me.has_built_wonder(wonder) {
                continue;
            }
            if !cost::wonder_cost(state, player, wonder).affordable_by(state, player) {
                continue;
            }
            advancing_wonders |= 1u16 << wonder.index();
            if u16::from(def.shields) >= u16::from(need) && need > 0 {
                closing_wonders |= 1u16 << wonder.index();
            }
            now += u16::from(def.shields);
            best_single = best_single.max(def.shields);
            push(
                ShieldSource::Wonder {
                    wonder,
                    shields: def.shields,
                },
                &mut sources,
                &mut count,
            );
        }
    }

    // --- what is still out there ------------------------------------------
    let visible = u8::try_from(crate::masks::shields_in(board.face_up).min(u16::from(u8::MAX)))
        .unwrap_or(u8::MAX);
    let expected_hidden = board.expected_hidden(|c| f64::from(c.def().shields));
    let expected_hidden_cards =
        board.expected_hidden(|c| if c.def().shields > 0 { 1.0 } else { 0.0 });

    let mut expected_future_ages = 0.0;
    let mut expected_future_cards = 0.0;
    for age in board.undealt_ages() {
        let s: &AgeSupply = m.age_supply(age);
        expected_future_ages += s.expected_shields();
        expected_future_cards += s.expected_shield_cards();
    }

    // --- tempo -------------------------------------------------------------
    let cards_left = board.cards_left();
    let tempo = if state.current_player() == player {
        cards_left.div_ceil(2)
    } else {
        cards_left / 2
    };
    let decisions_left =
        tempo.saturating_add(board.undealt_age_count().saturating_mul(DECISIONS_PER_AGE));

    // --- the future supply stream -----------------------------------------
    // Shield cards that are visible but not takeable this turn (covered, or
    // unaffordable right now), plus the expected hidden and undealt ones.
    let immediate_cards: u128 = sources
        .iter()
        .flatten()
        .filter_map(|s| match s {
            ShieldSource::Card { card, .. } => Some(1u128 << card.index()),
            ShieldSource::Wonder { .. } => None,
        })
        .fold(0u128, |a, b| a | b);
    let later_visible = board.face_up & m.any_shield_mask() & !immediate_cards;
    let later_visible_shields = f64::from(crate::masks::shields_in(later_visible));
    let later_visible_cards = f64::from(later_visible.count_ones());

    let stream_cards = later_visible_cards + expected_hidden_cards + expected_future_cards;
    let stream_shields = later_visible_shields + expected_hidden + expected_future_ages;

    let mut turns_to_close = estimate_turns(
        need,
        &sources[..count],
        stream_cards,
        stream_shields,
        decisions_left,
    );

    // --- classification ----------------------------------------------------
    // A player with no decisions left cannot reach anything, however good the
    // card in front of them looks — which happens exactly once per game, on
    // the last card of Age III, where whatever the mover does ends it.
    let status = if decisions_left == 0 {
        MilitaryStatus::Closed
    } else if need == 0 || best_single >= need {
        MilitaryStatus::Imminent
    } else if turns_to_close.is_some() {
        MilitaryStatus::Live
    } else {
        MilitaryStatus::Closed
    };
    // Keep the two consistent: an imminent close takes one turn by
    // definition, and a closed race has no estimate.
    match status {
        MilitaryStatus::Imminent => {
            turns_to_close = Some(u8::from(need > 0));
        }
        MilitaryStatus::Closed => turns_to_close = None,
        MilitaryStatus::Live => {}
    }

    let fork = u8::try_from(count).unwrap_or(u8::MAX);
    let closing_fork =
        u8::try_from(closing_slots.count_ones() + closing_wonders.count_ones()).unwrap_or(u8::MAX);
    // A shield-granting wonder cannot be taken away, so one alone is already
    // undeniable.
    let undeniable =
        status == MilitaryStatus::Imminent && (closing_fork >= 2 || closing_wonders != 0);

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
        now: u8::try_from(now.min(u16::from(u8::MAX))).unwrap_or(u8::MAX),
        best_single,
        visible,
        expected_hidden,
        expected_future_ages,
        fork,
        closing_fork,
        tempo,
        decisions_left,
        status,
        undeniable,
        turns_to_close,
        loot_damage,
        loot_shields_needed,
        bands,
        closing_slots,
        closing_wonders,
        advancing_slots,
        advancing_wonders,
        sources,
    }
}

/// Estimate how many of the player's own decisions it takes to accumulate
/// `need` shields.
///
/// The model: spend the first turns on the immediate sources, largest first,
/// then on a stream of `stream_cards` further shield cards averaging
/// `stream_shields / stream_cards` each, one per decision. Give up — and
/// report the race as closed — once `horizon` decisions are used up.
fn estimate_turns(
    need: u8,
    immediate: &[Option<ShieldSource>],
    stream_cards: f64,
    stream_shields: f64,
    horizon: u8,
) -> Option<u8> {
    if need == 0 {
        return Some(0);
    }
    if horizon == 0 {
        return None;
    }
    let target = f64::from(need);

    let mut gains: [u8; MAX_SHIELD_SOURCES] = [0; MAX_SHIELD_SOURCES];
    let mut n = 0usize;
    for src in immediate.iter().flatten() {
        gains[n] = src.shields();
        n += 1;
    }
    gains[..n].sort_unstable_by(|a, b| b.cmp(a));

    let mut acc = 0.0f64;
    let mut turns = 0u8;
    for &g in &gains[..n] {
        if acc >= target {
            break;
        }
        if turns >= horizon {
            return None;
        }
        acc += f64::from(g);
        turns += 1;
    }
    if acc >= target {
        return Some(turns);
    }

    let avg = if stream_cards > 0.0 {
        stream_shields / stream_cards
    } else {
        0.0
    };
    if avg <= 0.0 {
        return None;
    }
    let mut left = stream_cards;
    while acc < target && turns < horizon && left > 1e-9 {
        let take = left.min(1.0);
        acc += take * avg;
        left -= take;
        turns += 1;
    }
    if acc >= target {
        Some(turns)
    } else {
        None
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
}
