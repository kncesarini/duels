//! The leaf evaluation: a static estimate of how good a non-terminal
//! position is for one player.
//!
//! This runs at every cut-off leaf, so it stays cheap. It is deliberately
//! thin: for a search agent, another ply of lookahead buys more than another
//! hand-tuned term, and the terms that *are* here are the ones a two- or
//! three-ply search cannot see for itself — the two instant-win races
//! (military supremacy and six science symbols) and the latent value of a
//! resource base.
//!
//! The unit is "victory points from `me`'s point of view": the exact
//! end-of-game victory-point difference the position would score right now
//! (from [`duels_core::scoring::score`], which uses the real per-card values,
//! guild majorities, the military track and `coins / 3`), plus positional
//! terms expressed on the same scale.
//!
//! # Why the weights are a struct
//!
//! Every coefficient lives in [`Weights`] rather than in a `const`, so a
//! tuning harness can sweep them and an ablation can zero one out. The
//! defaults are what [`crate::Config::default`] uses.

use duels_core::state::PlayerState;
use duels_core::{scoring, GameState, Player};

/// Evaluation is clamped to this magnitude so it can never be confused with a
/// proven win or loss (see `search::MATE`).
pub const EVAL_CLAMP: f64 = 20_000.0;

/// Coefficients for [`evaluate_with`], all in victory points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    /// Value of holding `n` distinct scientific symbols, indexed by `n`. Six
    /// is an instant win, so it never appears at a non-terminal leaf, but the
    /// curve has to be convex enough that the search wants the fifth symbol
    /// far more than the second.
    pub science_ladder: [f64; 6],
    /// Per shield held, on top of the military victory points the score
    /// already counts. Shields keep paying: they push the pawn further next
    /// time.
    pub shield: f64,
    /// Per step of conflict-pawn advantage, on top of its victory points.
    pub conflict: f64,
    /// Extra weight per step, squared, once the pawn is within striking
    /// distance of the opponent's capital, where the threat is an outright win
    /// rather than points.
    pub capital_threat: f64,
    /// Distance from the centre at which [`Weights::capital_threat`] starts to
    /// count.
    pub capital_threat_from: u8,
    /// Per coin. Coins are already worth `1/3` VP each; the surplus is
    /// optionality.
    pub coin: f64,
    /// Per distinct resource the player produces at all (cheaper future
    /// builds).
    pub resource_breadth: f64,
    /// Per unpaired scientific symbol: half of a future progress token.
    pub science_single: f64,
    /// Per drafted-but-unbuilt wonder: a standing option rather than an asset.
    pub wonder_option: f64,
    /// Per card in the player's city.
    ///
    /// An evaluation that counts only victory points *already earned*
    /// systematically prefers banking coins to taking cards: a discard is
    /// worth a certain `2/3` of a point through the `coins / 3` term, while a
    /// brown or grey card is worth nothing at all until it pays for a build
    /// beyond the horizon. That bias was plainly visible in the original
    /// agent's games — it discarded most of its Age II and Age III turns — and
    /// a flat per-card credit is the cheapest correction available. It is not
    /// a claim that all cards are equally good, only that holding one beats
    /// holding two coins.
    ///
    /// Worth 58% over 100 games against `1.0` on its own, and still 55% on top
    /// of the playout leaf, which is the more interesting result: the playouts
    /// see the same thing from the other direction and the two do not fully
    /// overlap.
    pub card_in_city: f64,
}

impl Weights {
    /// The tuned defaults.
    pub const DEFAULT: Weights = Weights {
        science_ladder: [0.0, 1.0, 3.0, 6.5, 12.0, 24.0],
        shield: 0.7,
        conflict: 0.6,
        capital_threat: 3.0,
        capital_threat_from: 5,
        coin: 0.10,
        resource_breadth: 0.55,
        science_single: 1.1,
        wonder_option: 0.4,
        card_in_city: 1.0,
    };

    /// What the crate shipped with before the leaf evaluation was reworked:
    /// no per-card credit. Kept for before/after measurement.
    pub const V1: Weights = Weights {
        science_ladder: [0.0, 1.0, 3.0, 6.5, 12.0, 24.0],
        shield: 0.7,
        conflict: 0.6,
        capital_threat: 3.0,
        capital_threat_from: 5,
        coin: 0.10,
        resource_breadth: 0.55,
        science_single: 1.1,
        wonder_option: 0.4,
        card_in_city: 0.0,
    };

    /// Every positional term zeroed: the evaluation degenerates to the exact
    /// victory-point difference the position scores right now. For ablations
    /// through `duels-arena`'s `ab_lab` example (`weights=score-only`).
    pub const SCORE_ONLY: Weights = Weights {
        science_ladder: [0.0; 6],
        shield: 0.0,
        conflict: 0.0,
        capital_threat: 0.0,
        capital_threat_from: 9,
        coin: 0.0,
        resource_breadth: 0.0,
        science_single: 0.0,
        wonder_option: 0.0,
        card_in_city: 0.0,
    };
}

impl Default for Weights {
    fn default() -> Self {
        Weights::DEFAULT
    }
}

/// Static evaluation of `state` from `me`'s point of view, in victory points,
/// with the default [`Weights`].
///
/// Only meaningful for a position that is not over; a finished game has a
/// [`duels_core::GameResult`] and is scored as a hard terminal by the search.
pub fn evaluate(state: &GameState, me: Player) -> f64 {
    evaluate_with(state, me, &Weights::DEFAULT)
}

/// [`evaluate`] with explicit coefficients.
pub fn evaluate_with(state: &GameState, me: Player, w: &Weights) -> f64 {
    let opp = me.other();
    let scores = scoring::score(state);
    let vp = f64::from(scores[me.index()].total) - f64::from(scores[opp.index()].total);

    let v = vp + player_term(state, me, w) - player_term(state, opp, w);
    v.clamp(-EVAL_CLAMP, EVAL_CLAMP)
}

/// Everything positional about one player, in victory points.
fn player_term(state: &GameState, p: Player, w: &Weights) -> f64 {
    let ps = state.player(p);
    military(state, p, ps, w) + science(ps, w) + economy(ps, w) + wonder_options(ps, w)
}

fn military(state: &GameState, p: Player, ps: &PlayerState, w: &Weights) -> f64 {
    let mut v = w.shield * f64::from(ps.shields());
    if state.military_leader() == Some(p) {
        let distance = state.conflict().unsigned_abs();
        v += w.conflict * f64::from(distance);
        if distance >= w.capital_threat_from {
            let over = distance - w.capital_threat_from;
            v += w.capital_threat * f64::from(over) * f64::from(over);
        }
    }
    v
}

fn science(ps: &PlayerState, w: &Weights) -> f64 {
    let distinct = usize::from(ps.distinct_science());
    let mut v = w.science_ladder[distinct.min(w.science_ladder.len() - 1)];
    // A symbol held exactly once is half a progress token away.
    let singles = ps.science().iter().filter(|&&n| n == 1).count();
    v += w.science_single * singles as f64;
    v
}

fn economy(ps: &PlayerState, w: &Weights) -> f64 {
    let mut breadth = ps.production().iter().filter(|&&n| n > 0).count();
    let (raw_choice, manufactured_choice) = ps.choice_sources();
    breadth += usize::from(raw_choice > 0) + usize::from(manufactured_choice > 0);
    let cards = ps.built_mask().count_ones();
    w.coin * f64::from(ps.coins())
        + w.resource_breadth * breadth as f64
        + w.card_in_city * f64::from(cards)
}

fn wonder_options(ps: &PlayerState, w: &Weights) -> f64 {
    let unbuilt = ps.wonders().filter(|&w| !ps.has_built_wonder(w)).count();
    w.wonder_option * unbuilt as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use duels_core::testing::StateBuilder;

    #[test]
    fn a_pile_of_victory_points_evaluates_positive() {
        let st = StateBuilder::new()
            .built(Player::One, &["palace", "town-hall", "pantheon"])
            .build();
        assert!(evaluate(&st, Player::One) > 10.0);
        // And symmetrically negative for the other seat.
        assert!(evaluate(&st, Player::Two) < -10.0);
    }

    #[test]
    fn the_evaluation_is_zero_sum() {
        for seed in 0..5u64 {
            let st = engine::new_game(seed);
            let a = evaluate(&st, Player::One);
            let b = evaluate(&st, Player::Two);
            assert!((a + b).abs() < 1e-9, "seed {seed}: {a} + {b}");
        }
    }

    #[test]
    fn being_one_step_from_the_capital_beats_being_five_steps_away() {
        let near = StateBuilder::new().conflict(8).build();
        let far = StateBuilder::new().conflict(4).build();
        assert!(
            evaluate(&near, Player::One) > evaluate(&far, Player::One) + 5.0,
            "near {} vs far {}",
            evaluate(&near, Player::One),
            evaluate(&far, Player::One)
        );
    }

    #[test]
    fn the_fifth_science_symbol_is_worth_more_than_the_second() {
        let two = StateBuilder::new()
            .built(Player::One, &["apothecary", "workshop"])
            .build();
        let five = StateBuilder::new()
            .built(
                Player::One,
                &[
                    "apothecary",
                    "workshop",
                    "scriptorium",
                    "pharmacist",
                    "academy",
                ],
            )
            .build();
        let d2 = evaluate(&two, Player::One);
        let d5 = evaluate(&five, Player::One);
        assert!(d5 > d2 * 2.0, "two = {d2}, five = {d5}");
    }

    #[test]
    fn evaluation_stays_well_inside_the_mate_range() {
        for seed in 0..20u64 {
            let st = engine::new_game(seed);
            let v = evaluate(&st, Player::One);
            assert!(v.abs() <= EVAL_CLAMP);
        }
    }

    #[test]
    fn a_card_in_the_city_beats_the_coins_a_discard_would_have_paid() {
        // The bias `card_in_city` exists to correct. Three coins are a
        // certain point through `coins / 3`, so a card with no printed victory
        // points has to be worth at least that much or the search prefers to
        // throw it away — which is what the first version of this crate did,
        // all game long.
        let discarded = StateBuilder::new().coins(Player::One, 3).build();
        let built = StateBuilder::new()
            .built(Player::One, &["lumber-yard"])
            .coins(Player::One, 0)
            .build();
        let (d, b) = (
            evaluate(&discarded, Player::One),
            evaluate(&built, Player::One),
        );
        assert!(b > d, "built {b} vs discarded {d}");
        // And with the pre-rework weights it was the other way round, which
        // is the point.
        let d1 = evaluate_with(&discarded, Player::One, &Weights::V1);
        let b1 = evaluate_with(&built, Player::One, &Weights::V1);
        assert!(b1 < d1, "v1: built {b1} vs discarded {d1}");
    }

    #[test]
    fn the_score_only_ablation_is_exactly_the_victory_point_difference() {
        for seed in 0..5u64 {
            let st = engine::new_game(seed);
            let s = scoring::score(&st);
            let want = f64::from(s[0].total) - f64::from(s[1].total);
            let got = evaluate_with(&st, Player::One, &Weights::SCORE_ONLY);
            assert!((got - want).abs() < 1e-9, "seed {seed}: {got} vs {want}");
        }
    }
}
