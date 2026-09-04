//! The victory-point race read: who is ahead if the game stopped now, and how
//! durable is that lead?
//!
//! The "ahead now" half is not re-derived here. [`duels_core::scoring::breakdown`]
//! already computes the real end-of-game score from the real card data,
//! including the guild majority recount and `floor(coins / 3)`, and
//! `duels-agent-greedy` already leans on it for exactly this purpose; this
//! module calls it and subtracts.
//!
//! What it adds is the *swing still available*: unbuilt wonder points (already
//! drafted, so they are a private reserve nobody can take away), the civilian
//! points still sitting in the structure or still to be dealt, the way the
//! in-play guilds' majority targets currently lean, and each side's coin
//! trajectory. [`VpRead::structural_edge`] folds the durable parts of that
//! into one signed number, which the stance layer uses to decide how hard a
//! trailing player should tilt into a race.

use duels_core::data::CardType;
use duels_core::scoring;
use duels_core::{GameState, Player};

use crate::board::Board;
use crate::masks::{iter_cards, masks};

/// Named weights for [`VpRead::structural_edge`].
///
/// `gap` is the only term with a natural unit (it is already victory points);
/// the other two are guesses about how much of a *potential* point converts
/// into a real one, and are grouped here so they can be tuned rather than
/// buried.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VpWeights {
    /// Weight on the "as if the game ended now" point difference.
    pub gap: f64,
    /// Weight on the difference in drafted-but-unbuilt wonder points. Below
    /// one because a drafted wonder still has to be paid for, and the eighth
    /// wonder can never be built at all.
    pub unbuilt_wonder: f64,
    /// Weight on the in-play guilds' majority lean.
    pub guild_lean: f64,
}

impl Default for VpWeights {
    fn default() -> Self {
        Self {
            gap: 1.0,
            unbuilt_wonder: 0.5,
            guild_lean: 0.5,
        }
    }
}

/// The victory-point race, read for one player.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VpRead {
    /// The player this read is about.
    pub player: Player,
    /// This player's total if the game ended now.
    pub my_total: u16,
    /// The opponent's total if the game ended now.
    pub their_total: u16,
    /// `my_total - their_total`.
    pub gap: i32,
    /// Points printed on this player's drafted-but-unbuilt wonders.
    pub my_unbuilt_wonder_vp: u16,
    /// Points printed on the opponent's drafted-but-unbuilt wonders.
    pub their_unbuilt_wonder_vp: u16,
    /// Civilian (blue) points on face-up cards still in the structure.
    pub civilian_vp_face_up: u16,
    /// Expected civilian points behind the current age's face-down slots.
    pub civilian_vp_hidden: f64,
    /// Expected civilian points from the ages not yet dealt.
    pub civilian_vp_future_ages: f64,
    /// How the in-play guilds' majority targets lean, signed positive when
    /// this player holds the majority the guilds pay on. See
    /// [`guild_lean`].
    pub guild_lean: f64,
    /// This player's `floor(coins / 3)`.
    pub my_coin_vp: u16,
    /// The opponent's `floor(coins / 3)`.
    pub their_coin_vp: u16,
    /// `gap` plus the wonder and guild lean terms: who is ahead, and by how
    /// structurally durable a margin.
    pub structural_edge: f64,
}

impl VpRead {
    /// Total civilian points still obtainable from the structure, visible or
    /// expected, plus the undealt ages.
    #[inline]
    pub fn civilian_swing(&self) -> f64 {
        f64::from(self.civilian_vp_face_up) + self.civilian_vp_hidden + self.civilian_vp_future_ages
    }
}

/// Read the victory-point race for `player`.
pub fn vp_read(state: &GameState, player: Player) -> VpRead {
    vp_read_with(state, player, &Board::of(state), &VpWeights::default())
}

/// [`vp_read`] against a [`Board`] the caller already built, with explicit
/// weights.
pub fn vp_read_with(state: &GameState, player: Player, board: &Board, w: &VpWeights) -> VpRead {
    let m = masks();
    let opp = player.other();
    let mine = scoring::breakdown(state, player);
    let theirs = scoring::breakdown(state, opp);

    let unbuilt_wonder_vp = |p: Player| -> u16 {
        let ps = state.player(p);
        ps.wonders()
            .filter(|&wid| !ps.has_built_wonder(wid))
            .map(|wid| u16::from(wid.def().victory_points))
            .sum()
    };

    let civilian_face_up = crate::masks::victory_points_in(board.face_up & m.civilian_vp_mask());
    let civ_mask = m.civilian_vp_mask();
    let civilian_vp_hidden = board.expected_hidden(|c| {
        if civ_mask & (1u128 << c.index()) != 0 {
            f64::from(c.def().victory_points)
        } else {
            0.0
        }
    });
    let mut civilian_vp_future_ages = 0.0;
    for age in board.undealt_ages() {
        civilian_vp_future_ages += m.age_supply(age).expected_civilian_vp();
    }

    let lean = guild_lean(state, player, board);
    let gap = i32::from(mine.total) - i32::from(theirs.total);
    let my_unbuilt_wonder_vp = unbuilt_wonder_vp(player);
    let their_unbuilt_wonder_vp = unbuilt_wonder_vp(opp);

    let structural_edge = f64::from(gap) * w.gap
        + (f64::from(my_unbuilt_wonder_vp) - f64::from(their_unbuilt_wonder_vp)) * w.unbuilt_wonder
        + lean * w.guild_lean;

    VpRead {
        player,
        my_total: mine.total,
        their_total: theirs.total,
        gap,
        my_unbuilt_wonder_vp,
        their_unbuilt_wonder_vp,
        civilian_vp_face_up: civilian_face_up,
        civilian_vp_hidden,
        civilian_vp_future_ages,
        guild_lean: lean,
        my_coin_vp: mine.coins,
        their_coin_vp: theirs.coins,
        structural_edge,
    }
}

/// How the guilds that are publicly in play lean, from `player`'s side.
///
/// A guild pays its owner `points_per_unit × max(my count, their count)` on
/// some category, so what matters strategically is *who holds the majority in
/// the categories the guilds actually key on* — a guild keyed to a category
/// the opponent dominates pays well whoever ends up owning it, which makes it
/// a liability to leave lying around. This sums `per × (my count − their
/// count)` over every guild whose identity is public: face up in the
/// structure, in either city, in the discard pile, or spent under a wonder.
///
/// Positive means the guilds in play are keyed to categories `player` leads —
/// a rough read, not a projection: it deliberately does not try to guess who
/// will end up owning them.
pub fn guild_lean(state: &GameState, player: Player, board: &Board) -> f64 {
    let m = masks();
    let known_guilds =
        (board.face_up | board.in_city | board.discard | board.fodder) & m.guild_mask();
    let mut lean = 0.0;
    for card in iter_cards(known_guilds) {
        let Some((target, per)) = card.def().points_by_majority else {
            continue;
        };
        let mine = f64::from(state.player(player).count(target));
        let theirs = f64::from(state.player(player.other()).count(target));
        lean += f64::from(per) * (mine - theirs);
    }
    lean
}

/// Printed victory points on the civilian cards of one player's city, for
/// callers explaining a read.
pub fn civilian_vp_built(state: &GameState, player: Player) -> u16 {
    let s = duels_core::data::statics();
    crate::masks::victory_points_in(
        state.player(player).built_mask() & s.card_masks[CardType::Civilian.index()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::testing::StateBuilder;

    #[test]
    fn the_gap_is_exactly_the_scoring_breakdown_difference() {
        let st = StateBuilder::new()
            .built(Player::One, &["palace", "theater"])
            .built(Player::Two, &["altar"])
            .coins(Player::One, 9)
            .conflict(4)
            .build();
        let r = vp_read(&st, Player::One);
        let a = scoring::breakdown(&st, Player::One);
        let b = scoring::breakdown(&st, Player::Two);
        assert_eq!(r.my_total, a.total);
        assert_eq!(r.their_total, b.total);
        assert_eq!(r.gap, i32::from(a.total) - i32::from(b.total));
        // ...and it is antisymmetric.
        assert_eq!(vp_read(&st, Player::Two).gap, -r.gap);
    }

    #[test]
    fn unbuilt_wonder_points_are_counted_separately_from_the_gap() {
        let st = StateBuilder::new()
            .wonders(Player::One, &["the-pyramids"])
            .build();
        let r = vp_read(&st, Player::One);
        assert_eq!(r.my_unbuilt_wonder_vp, 9);
        assert_eq!(r.gap, 0, "an unbuilt wonder scores nothing yet");
        assert!(r.structural_edge > 0.0);
    }

    #[test]
    fn guild_lean_follows_the_majority_target_not_the_owner() {
        // Player Two owns the Merchants Guild (keyed to yellow cards), but
        // Player One holds the yellow majority, so the lean is positive for
        // Player One even though the guild is in the opponent's city.
        let st = StateBuilder::new()
            .built(Player::Two, &["merchants-guild"])
            .built(Player::One, &["tavern", "brewery", "forum"])
            .build();
        let b = Board::of(&st);
        assert!(guild_lean(&st, Player::One, &b) > 0.0);
        assert!(guild_lean(&st, Player::Two, &b) < 0.0);
    }
}
