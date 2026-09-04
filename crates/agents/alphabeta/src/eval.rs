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

use duels_core::data;
use duels_core::state::PlayerState;
use duels_core::{scoring, GameState, Player};

/// Evaluation is clamped to this magnitude so it can never be confused with a
/// proven win or loss (see `search::MATE`).
pub const EVAL_CLAMP: f64 = 20_000.0;

/// Value of holding `n` distinct scientific symbols. Six is an instant win,
/// so it never appears at a non-terminal leaf, but the curve has to be convex
/// enough that the search wants the fifth symbol far more than the second.
const SCIENCE_LADDER: [f64; 6] = [0.0, 1.0, 3.0, 6.5, 12.0, 24.0];

/// Per shield held, on top of the military victory points the score already
/// counts. Shields keep paying: they push the pawn further next time.
const W_SHIELD: f64 = 0.7;
/// Per step of conflict-pawn advantage, on top of its victory points.
const W_CONFLICT: f64 = 0.6;
/// Extra weight per step once the pawn is within striking distance of the
/// opponent's capital, where the threat is an outright win rather than points.
const W_CAPITAL_THREAT: f64 = 3.0;
/// Distance from the centre at which the capital threat starts to count.
const CAPITAL_THREAT_FROM: u8 = 5;
/// Per coin. Coins are already worth `1/3` VP each; the surplus is optionality.
const W_COIN: f64 = 0.10;
/// Per distinct resource the player produces at all (cheaper future builds).
const W_RESOURCE_BREADTH: f64 = 0.55;
/// Per unpaired scientific symbol: half of a future progress token.
const W_SCIENCE_SINGLE: f64 = 1.1;
/// Per drafted-but-unbuilt wonder, which is a standing option, not a asset.
const W_WONDER_OPTION: f64 = 0.4;

/// Static evaluation of `state` from `me`'s point of view, in victory points.
///
/// Only meaningful for a position that is not over; a finished game has a
/// [`duels_core::GameResult`] and is scored as a hard terminal by the search.
pub fn evaluate(state: &GameState, me: Player) -> f64 {
    let opp = me.other();
    let scores = scoring::score(state);
    let vp = f64::from(scores[me.index()].total) - f64::from(scores[opp.index()].total);

    let v = vp + player_term(state, me) - player_term(state, opp);
    v.clamp(-EVAL_CLAMP, EVAL_CLAMP)
}

/// Everything positional about one player, in victory points.
fn player_term(state: &GameState, p: Player) -> f64 {
    let ps = state.player(p);
    military(state, p, ps) + science(ps) + economy(ps) + wonder_options(ps)
}

fn military(state: &GameState, p: Player, ps: &PlayerState) -> f64 {
    let mut v = W_SHIELD * f64::from(ps.shields());
    if state.military_leader() == Some(p) {
        let distance = state.conflict().unsigned_abs();
        v += W_CONFLICT * f64::from(distance);
        if distance >= CAPITAL_THREAT_FROM {
            let over = distance - CAPITAL_THREAT_FROM;
            v += W_CAPITAL_THREAT * f64::from(over) * f64::from(over);
        }
    }
    v
}

fn science(ps: &PlayerState) -> f64 {
    let distinct = usize::from(ps.distinct_science());
    let mut v = SCIENCE_LADDER[distinct.min(SCIENCE_LADDER.len() - 1)];
    // A symbol held exactly once is half a progress token away.
    let singles = ps.science().iter().filter(|&&n| n == 1).count();
    v += W_SCIENCE_SINGLE * singles as f64;
    v
}

fn economy(ps: &PlayerState) -> f64 {
    let mut breadth = ps.production().iter().filter(|&&n| n > 0).count();
    let (raw_choice, manufactured_choice) = ps.choice_sources();
    breadth += usize::from(raw_choice > 0) + usize::from(manufactured_choice > 0);
    W_COIN * f64::from(ps.coins()) + W_RESOURCE_BREADTH * breadth as f64
}

fn wonder_options(ps: &PlayerState) -> f64 {
    let unbuilt = ps.wonders().filter(|&w| !ps.has_built_wonder(w)).count();
    W_WONDER_OPTION * unbuilt as f64
}

/// Distance the conflict pawn still has to travel for an outright win, for
/// the player it currently favours. Exposed for tests and for the move
/// ordering, which likes to know when a build is a killer blow.
pub fn steps_to_capital(state: &GameState) -> u8 {
    data::military()
        .capital_distance
        .saturating_sub(state.conflict().unsigned_abs())
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
    fn steps_to_capital_counts_down() {
        assert_eq!(
            steps_to_capital(&StateBuilder::new().conflict(0).build()),
            data::military().capital_distance
        );
        assert_eq!(
            steps_to_capital(&StateBuilder::new().conflict(-8).build()),
            1
        );
    }
}
