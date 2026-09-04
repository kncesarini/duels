//! Everything worth computing exactly once per position, for both players.
//!
//! The threat magnitudes are not symmetric functions of one player's own
//! position: how likely Player One is to win the science race depends on how
//! many decisions Player Two has left, on whether Player Two could snatch the
//! Law token off the board with a pair of their own, and on how hard Player
//! Two can shove the conflict pawn back. So the reads cannot be computed
//! independently, and computing each one twice — once for its own sake and
//! once as the other's input — would double the expensive part.
//!
//! That expensive part is the cost engine. [`Context`] therefore gathers, in
//! one pass:
//!
//! * where the cards publicly are ([`Board`]);
//! * what each of them costs each player ([`Prices`]);
//! * the expectations over the *unknown* cards, which are the same for both
//!   players and used by both races and the point read ([`Expectations`]);
//! * how many decisions each side has left, and how many come in a row
//!   ([`Tempo`]);
//! * the position-only half of both races.
//!
//! [`crate::military_read_with`] / [`crate::science_read_with`] then finish the
//! job with both halves in hand, and touch the cost engine not at all.
//!
//! Like everything else in this crate it reads only public information, which
//! `tests/determinization_invariance.rs` asserts field by field and bit by
//! bit.

use duels_core::{GameState, Player};

use crate::board::Board;
use crate::masks::masks;
use crate::military::MilBase;
use crate::prices::Prices;
use crate::science::SciBase;
use crate::tempo::{tempo_pair, Tempo, ThreatWeights};

/// The expected contents of everything not yet revealed, which is the same
/// question for both players.
///
/// Each figure is the sum over the current age's unknown pool scaled by how
/// many face-down slots there are to fill (exactly the sampling
/// [`duels_core::Observation::sample_state`] performs), plus, for the undealt
/// ages, the deck total scaled by how much of the deck setup actually deals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Expectations {
    /// Shields still to come from the current age's face-down slots.
    pub hidden_shields: f64,
    /// Shield-bearing *cards* still to come from those slots.
    pub hidden_shield_cards: f64,
    /// Civilian victory points still to come from those slots.
    pub hidden_civilian_vp: f64,
    /// Shields from the ages that have not been dealt yet.
    pub future_shields: f64,
    /// Shield-bearing cards from those ages.
    pub future_shield_cards: f64,
    /// Civilian victory points from those ages.
    pub future_civilian_vp: f64,
    /// The chance that a named card of the current age which is not publicly
    /// placed is behind a face-down slot rather than in the box: the
    /// `p_hidden` of the science model.
    pub p_hidden: f64,
}

impl Expectations {
    /// Compute the expectations for `board`.
    pub fn of(board: &Board) -> Expectations {
        let m = masks();
        let civ = m.civilian_vp_mask();
        let shield = m.any_shield_mask();
        // One pass over the pool, three accumulators: the pool is walked once
        // instead of once per figure per player.
        let mut hidden_shields = 0.0;
        let mut hidden_shield_cards = 0.0;
        let mut hidden_civilian_vp = 0.0;
        if board.hidden_slot_count() > 0 {
            hidden_shields = board.expected_hidden(|c| f64::from(c.def().shields));
            hidden_shield_cards =
                board.expected_hidden(|c| f64::from(u8::from(shield & (1u128 << c.index()) != 0)));
            hidden_civilian_vp = board.expected_hidden(|c| {
                if civ & (1u128 << c.index()) != 0 {
                    f64::from(c.def().victory_points)
                } else {
                    0.0
                }
            });
        }

        let mut future_shields = 0.0;
        let mut future_shield_cards = 0.0;
        let mut future_civilian_vp = 0.0;
        for age in board.undealt_ages() {
            let s = m.age_supply(age);
            future_shields += s.expected_shields();
            future_shield_cards += s.expected_shield_cards();
            future_civilian_vp += s.expected_civilian_vp();
        }

        let pool = board.unknown_plain.count_ones();
        Expectations {
            hidden_shields,
            hidden_shield_cards,
            hidden_civilian_vp,
            future_shields,
            future_shield_cards,
            future_civilian_vp,
            p_hidden: if pool == 0 {
                0.0
            } else {
                f64::from(board.hidden_plain_count()) / f64::from(pool)
            },
        }
    }
}

/// The once-per-position digest every read in this crate is built on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Context {
    /// Where the cards publicly are.
    pub board: Board,
    /// What is expected from the cards that are not.
    pub expected: Expectations,
    /// The weights in force.
    pub weights: ThreatWeights,
    prices: [Prices; 2],
    tempo: [Tempo; 2],
    mil: [MilBase; 2],
    sci: [SciBase; 2],
}

impl Context {
    /// Digest `state` with [`ThreatWeights::default`].
    pub fn of(state: &GameState) -> Context {
        Context::with(state, ThreatWeights::default())
    }

    /// Digest `state` with explicit weights.
    pub fn with(state: &GameState, weights: ThreatWeights) -> Context {
        let board = Board::of(state);
        let expected = Expectations::of(&board);
        let mut prices = [
            Prices::of(state, Player::One, &board),
            Prices::of(state, Player::Two, &board),
        ];
        let tempo = tempo_pair(state, &board, &prices, &weights);
        // `Prices` covers the accessible slots and the frontier one move out,
        // which is everything a read asks about unless somebody can chain two
        // or more extra turns and reach deeper.
        let deepest = tempo[0].chain.max(tempo[1].chain);
        if deepest >= 2 {
            let extra = board.reveals_within(deepest);
            for (i, p) in prices.iter_mut().enumerate() {
                p.price_also(state, Player::ALL[i], &board, extra);
            }
        }
        Context {
            mil: [
                MilBase::of(state, Player::One, &board, &prices[0]),
                MilBase::of(state, Player::Two, &board, &prices[1]),
            ],
            sci: [
                SciBase::of(state, Player::One, &board, &prices[0]),
                SciBase::of(state, Player::Two, &board, &prices[1]),
            ],
            board,
            expected,
            weights,
            prices,
            tempo,
        }
    }

    /// One player's decision budget.
    #[inline]
    pub fn tempo(&self, player: Player) -> &Tempo {
        &self.tempo[player.index()]
    }

    /// What everything costs one player.
    #[inline]
    pub fn prices(&self, player: Player) -> &Prices {
        &self.prices[player.index()]
    }

    #[inline]
    pub(crate) fn military_base(&self, player: Player) -> &MilBase {
        &self.mil[player.index()]
    }

    #[inline]
    pub(crate) fn science_base(&self, player: Player) -> &SciBase {
        &self.sci[player.index()]
    }
}
