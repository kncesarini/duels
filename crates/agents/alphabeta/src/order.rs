//! Move ordering.
//!
//! Alpha-beta only prunes if good moves come first, and a chance node cannot
//! prune at all until its parent has a decent alpha, so ordering matters more
//! here than in a deterministic game of the same branching factor.
//!
//! Two orderings live here. [`order`] scores moves from static card and
//! wonder data — cheap integers, no apply, no evaluation call — preferring
//! the transposition table's move, then wonders (the highest-leverage single
//! action in the game), then cards by what they contribute, with chain-free
//! builds boosted and plain discards last unless the player is broke.
//! [`order_by_lookahead`] instead applies each move and evaluates the
//! result, which correlates far better and costs far more.
//!
//! Neither is ever compared against a search value; they exist only to sort.

use duels_core::data::CardId;
use duels_core::engine::{self, Outcome};
use duels_core::{cost, Action, GameState, Player};

/// Longest move list `order` sorts. A turn offers at most `accessible slots x
/// (build + discard + up to four wonders)`, which cannot reach this.
const MAX_SORTED: usize = 128;

/// Sort `moves` best-first, in place.
///
/// Purely a performance device: the search visits exactly the same set of
/// moves in any order and returns the same value, which
/// `search::tests::move_ordering_does_not_change_the_result` asserts.
pub fn order(state: &GameState, moves: &mut [Action], tt_move: Option<Action>) {
    let n = moves.len().min(MAX_SORTED);
    debug_assert_eq!(n, moves.len(), "move list longer than MAX_SORTED");
    let mut scores = [0i32; MAX_SORTED];
    for i in 0..n {
        scores[i] = if Some(moves[i]) == tt_move {
            i32::MAX
        } else {
            score(state, moves[i])
        };
    }
    // Insertion sort, descending: n is small and this is stable, so a
    // shuffled input of equal-scoring moves keeps its relative order.
    for i in 1..n {
        let (s, m) = (scores[i], moves[i]);
        let mut j = i;
        while j > 0 && scores[j - 1] < s {
            scores[j] = scores[j - 1];
            moves[j] = moves[j - 1];
            j -= 1;
        }
        scores[j] = s;
        moves[j] = m;
    }
}

/// Order `moves` by a one-ply lookahead instead: the static value of the
/// position each move leads to, best-first for the player to move.
///
/// Much better correlated with the search value than [`score`] is, and much
/// more expensive — an apply and an evaluation per move. Randomness is not
/// expanded here: a move that would uncover a face-down card is applied with
/// the trivial outcome, which lets the engine reveal whatever the
/// determinized state happens to hide. That makes this a heuristic reading of
/// one possible future, which is all an ordering needs to be.
pub fn order_by_lookahead(
    state: &GameState,
    me: Player,
    moves: &mut [Action],
    tt_move: Option<Action>,
) {
    let n = moves.len().min(MAX_SORTED);
    debug_assert_eq!(n, moves.len(), "move list longer than MAX_SORTED");
    let sign = if state.current_player() == me {
        1.0
    } else {
        -1.0
    };
    let mut scores = [0.0f64; MAX_SORTED];
    for i in 0..n {
        scores[i] = if Some(moves[i]) == tt_move {
            f64::INFINITY
        } else {
            let mut next = *state;
            if engine::apply_with_outcome_unchecked(&mut next, moves[i], &Outcome::default())
                .is_err()
            {
                f64::NEG_INFINITY
            } else {
                sign * crate::eval::evaluate(&next, me)
            }
        };
    }
    for i in 1..n {
        let (s, m) = (scores[i], moves[i]);
        let mut j = i;
        while j > 0 && scores[j - 1] < s {
            scores[j] = scores[j - 1];
            moves[j] = moves[j - 1];
            j -= 1;
        }
        scores[j] = s;
        moves[j] = m;
    }
}

/// Heuristic desirability of `action` for the player to move, in arbitrary
/// integer units.
pub fn score(state: &GameState, action: Action) -> i32 {
    let p = state.current_player();
    match action {
        Action::PickWonder { wonder } => {
            let d = wonder.def();
            10 * i32::from(d.victory_points)
                + 16 * i32::from(d.shields)
                + 2 * i32::from(d.coins)
                + if d.play_again { 25 } else { 0 }
                + if d.destroy.is_some() { 15 } else { 0 }
                + if d.choose_progress_token { 20 } else { 0 }
                + if d.build_discarded_free { 10 } else { 0 }
        }
        Action::BuildWonder { slot, wonder } => {
            let d = wonder.def();
            let mut s = 70
                + 10 * i32::from(d.victory_points)
                + 16 * i32::from(d.shields)
                + 3 * i32::from(d.coins)
                + 2 * i32::from(d.opponent_loses_coins)
                + if d.play_again { 25 } else { 0 }
                + if d.destroy.is_some() { 18 } else { 0 }
                + if d.choose_progress_token { 20 } else { 0 }
                + if d.build_discarded_free { 12 } else { 0 };
            // Spending a card the opponent wanted is a bonus; spending one we
            // wanted ourselves is not.
            if let Some(card) = state.face_up_card(slot) {
                s -= card_value(card) / 4;
            }
            s
        }
        Action::Build { slot } => {
            let Some(card) = state.face_up_card(slot) else {
                return 0;
            };
            let mut s = card_value(card);
            let c = cost::card_cost(state, p, card);
            if c.via_chain {
                // Free by chain symbol: strictly better than paying for it.
                s += 30;
            } else {
                s -= 2 * i32::from(c.coins);
                s -= i32::from(c.trade);
            }
            s
        }
        Action::Discard { slot } => {
            let broke = state.player(p).coins() < 3;
            let mut s = -25 + if broke { 45 } else { 0 };
            // Taking a card away from the opponent is worth something even
            // when we only get coins for it.
            if let Some(card) = state.face_up_card(slot) {
                s += card_value(card) / 5;
            }
            s
        }
        Action::ChooseProgressToken { token } | Action::ChooseGreatLibraryToken { token } => {
            let d = token.def();
            30 + 10 * i32::from(d.victory_points)
                + 8 * i32::from(d.vp_per_token)
                + 3 * i32::from(d.coins)
                + if d.science.is_some() { 25 } else { 0 }
                + if d.discount.is_some() { 12 } else { 0 }
                + if d.shield_bonus { 14 } else { 0 }
                + if d.wonder_play_again { 12 } else { 0 }
                + 4 * i32::from(d.chain_build_coins)
                + if d.gain_trade_costs { 8 } else { 0 }
        }
        Action::MausoleumBuild { card } => card_value(card),
        Action::DestroyOpponentCard { card } => card_value(card),
        Action::ChooseFirstPlayer { player } => {
            // Going first in a new age is usually right, but not always, so
            // this is only a tie-break nudge.
            if player == p {
                5
            } else {
                0
            }
        }
    }
}

/// What owning `card` is worth, before its cost.
fn card_value(card: CardId) -> i32 {
    let d = card.def();
    let mut s = 10 * i32::from(d.victory_points)
        + 16 * i32::from(d.shields)
        + 3 * i32::from(d.coins)
        + if d.science.is_some() { 22 } else { 0 };
    // A resource producer or a chain head pays off later.
    s += 2 * i32::from(d.produces.iter().sum::<u8>());
    if d.produces_choice.is_some() {
        s += 6;
    }
    if d.chain_to.is_some() {
        s += 10;
    }
    if d.is_guild() {
        s += 14;
    }
    if d.coins_per_own.is_some() || d.coins_by_majority.is_some() {
        s += 6;
    }
    if d.points_by_majority.is_some() {
        s += 12;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::testing::StateBuilder;

    #[test]
    fn ordering_is_a_permutation_of_the_input() {
        for seed in 0..10u64 {
            let st = engine::new_game(seed);
            let mut moves = engine::legal_actions(&st);
            let before = moves.clone();
            order(&st, &mut moves, None);
            assert_eq!(moves.len(), before.len());
            for m in &before {
                assert!(moves.contains(m), "{m:?} was dropped");
            }
        }
    }

    #[test]
    fn the_transposition_table_move_is_tried_first() {
        let st = engine::new_game(3);
        let mut moves = engine::legal_actions(&st);
        assert!(moves.len() > 1);
        let favourite = moves[moves.len() - 1];
        order(&st, &mut moves, Some(favourite));
        assert_eq!(moves[0], favourite);
    }

    #[test]
    fn a_chain_free_build_outranks_the_same_card_paid_for() {
        // Horse Breeders chains from Stable; with the Stable already built the
        // build is free, which must score higher than paying for it.
        let free = StateBuilder::new()
            .age(2)
            .built(Player::One, &["stable"])
            .open_slots(&[(19, "horse-breeders")])
            .coins(Player::One, 20)
            .build();
        let paid = StateBuilder::new()
            .age(2)
            .open_slots(&[(19, "horse-breeders")])
            .coins(Player::One, 20)
            .build();
        let a = Action::Build { slot: 19 };
        assert!(
            score(&free, a) > score(&paid, a),
            "free {} vs paid {}",
            score(&free, a),
            score(&paid, a)
        );
    }

    #[test]
    fn discarding_is_a_last_resort_unless_broke() {
        let rich = StateBuilder::new()
            .open_slots(&[(19, "palace")])
            .coins(Player::One, 20)
            .build();
        let broke = StateBuilder::new()
            .open_slots(&[(19, "palace")])
            .coins(Player::One, 0)
            .build();
        let d = Action::Discard { slot: 19 };
        assert!(score(&rich, d) < 0);
        assert!(score(&broke, d) > score(&rich, d));
    }

    #[test]
    fn building_a_wonder_outranks_an_ordinary_card() {
        let st = StateBuilder::new()
            .age(2)
            .wonders(Player::One, &["the-colossus"])
            .open_slots(&[(19, "tavern")])
            .coins(Player::One, 30)
            .build();
        assert!(
            score(
                &st,
                Action::BuildWonder {
                    slot: 19,
                    wonder: duels_core::data::WonderId::from_slug("the-colossus").unwrap(),
                }
            ) > score(&st, Action::Build { slot: 19 })
        );
    }
}
