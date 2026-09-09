//! The feature vector: everything the learned value is allowed to see.
//!
//! # The one invariant that matters
//!
//! [`features`] takes a [`GameState`] — because that is what an MCTS leaf
//! holds, and because `duels-core`'s scoring and cost accessors are written
//! against it — but it reads **only public information**. Concretely, it never
//! touches a face-down slot's identity, the composition of an undealt age
//! deck, or which cards were returned to the box. [`GameState::face_up_card`]
//! is the only slot accessor used, and it answers `None` for a face-down slot.
//!
//! That is not a convention here, it is a test:
//! `tests/determinization_invariance.rs` samples two different concrete
//! `GameState`s from the same [`duels_core::Observation`] and asserts the two
//! feature vectors are equal **bit for bit** (`to_bits()` on every `f32`),
//! which is this repository's established shape for the check (`CLAUDE.md`,
//! "Non-negotiable invariants"). If a future feature reaches for a hidden
//! identity, that test fails rather than a search quietly getting stronger by
//! cheating.
//!
//! # The perspective convention
//!
//! Every feature is written from `me`'s point of view, and the vector carries
//! no "which seat am I" bit at all: a quantity that belongs to one player
//! appears as a `me` feature and an `opp` feature, and a signed quantity (the
//! conflict pawn) is relativised so that positive always favours `me`. So
//! `features(state, One)` and `features(state, Two)` are the same function of
//! the same position seen from the two sides, and a model trained on one
//! perspective is automatically valid for the other. `me_to_move` carries the
//! tempo fact the seat bit would otherwise have to.
//!
//! # Scaling
//!
//! Raw counts are divided by a plausible maximum so that essentially every
//! feature lands in roughly `[-1, 1]`, and none is standardised against corpus
//! statistics. That is deliberate: a mean/variance normalisation fitted on one
//! corpus is a second, invisible set of parameters that has to travel with the
//! weights and be reproduced exactly at inference time. Fixed divisors written
//! down in the source cannot drift, and a one-hidden-layer network absorbs the
//! remaining scale differences in its first weight matrix.
//!
//! # What is deliberately *not* in here
//!
//! No per-card indicator (73 of them per player would be a 146-feature bag
//! that dwarfs everything else and mostly restates the colour, point and
//! production aggregates that are here), and nothing from `duels-strategy` or
//! `duels-eval`. The second omission is the more interesting one: the point of
//! a learned value is to be an *independent* signal from the hand-crafted
//! evaluation, and feeding the hand-crafted evaluation in as a feature would
//! both couple this crate to a mandatory-review path and make "is the learned
//! value adding anything?" unanswerable.

use duels_core::cost;
use duels_core::data::{
    self, CardType, Resource, TokenId, WonderId, NUM_RESOURCES, NUM_SCIENCE, NUM_TOKENS,
    NUM_WONDERS,
};
use duels_core::layout::SLOTS;
use duels_core::scoring::{self, Breakdown};
use duels_core::state::{Pending, Phase};
use duels_core::{GameState, Player};

/// How many `f32` inputs [`features`] produces.
///
/// Pinned by `tests::the_layout_fills_the_vector_exactly`, which is what
/// catches a section that grew without this constant following it. It is also
/// written into the trained weights' header, so [`crate::Net::from_bytes`]
/// refuses a weights file built against a different value rather than
/// silently mis-reading a shifted vector.
pub const NUM_FEATURES: usize = 211;

/// The public-information feature vector for `state`, from `me`'s point of
/// view.
///
/// Pure, reads no randomness, and allocates nothing beyond the returned
/// array. See the module docs for the perspective convention and for the
/// hidden-information invariant this function is required to satisfy.
pub fn features(state: &GameState, me: Player) -> [f32; NUM_FEATURES] {
    let mut out = [0.0f32; NUM_FEATURES];
    let mut w = Writer {
        out: &mut out,
        n: 0,
    };
    let opp = me.other();

    clock(&mut w, state, me);
    military(&mut w, state, me, opp);
    science(&mut w, state, me, opp);
    economy(&mut w, state, me, opp);
    city(&mut w, state, me, opp);
    victory_points(&mut w, state, me, opp);
    structure(&mut w, state, me, opp);
    wonders(&mut w, state, me, opp);
    tokens(&mut w, state, me, opp);

    debug_assert_eq!(
        w.n, NUM_FEATURES,
        "the feature layout wrote {} of {NUM_FEATURES} slots",
        w.n
    );
    out
}

/// A cursor over the output array, so each section reads as a straight-line
/// list of pushes without every index being restated.
struct Writer<'a> {
    out: &'a mut [f32; NUM_FEATURES],
    n: usize,
}

impl Writer<'_> {
    #[inline]
    fn push(&mut self, v: f32) {
        self.out[self.n] = v;
        self.n += 1;
    }

    #[inline]
    fn flag(&mut self, b: bool) {
        self.push(if b { 1.0 } else { 0.0 });
    }

    /// A count divided by a plausible maximum. Not clamped: a value slightly
    /// over one is information, and the network is free to use it.
    #[inline]
    fn ratio(&mut self, v: f64, max: f64) {
        self.push((v / max) as f32);
    }

    /// A one-hot over `n` positions, with nothing set if `i` is out of range.
    #[inline]
    fn one_hot(&mut self, i: usize, n: usize) {
        for k in 0..n {
            self.flag(k == i);
        }
    }
}

// ---------------------------------------------------------------------------
// The sections. Each doc comment's width is checked against `NUM_FEATURES` by
// the `const` assertion at the bottom of this file.
// ---------------------------------------------------------------------------

/// Where in the game we are, and whose tempo it is. 13 features.
fn clock(w: &mut Writer, state: &GameState, me: Player) {
    let age = state.age().clamp(1, 3);
    w.one_hot(usize::from(age - 1), 3);
    w.ratio(f64::from(state.turn()), 70.0);
    w.ratio(f64::from(state.occupied_slots().count_ones()), SLOTS as f64);
    let phase = match state.phase() {
        Phase::WonderDraft => 0,
        Phase::Turn => 1,
        Phase::ChooseFirstPlayer => 2,
        Phase::GameOver => 3,
    };
    w.one_hot(phase, 4);
    w.ratio(f64::from(state.draft_step()), 8.0);
    w.flag(state.current_player() == me);
    // `extra_turn` is a fact about the player to move, so it is only `me`'s
    // when `me` is to move — hence the conjunction rather than a bare flag.
    w.flag(state.extra_turn() && state.current_player() == me);
    w.flag(state.last_card_taker() == me);
}

/// The conflict track and the shields that move it. 21 features.
fn military(w: &mut Writer, state: &GameState, me: Player, opp: Player) {
    let mil = data::military();
    // Positive always favours `me`, whichever seat that is.
    let pawn = match me {
        Player::One => i32::from(state.conflict()),
        Player::Two => -i32::from(state.conflict()),
    };
    let dist = pawn.unsigned_abs();
    w.ratio(f64::from(pawn), f64::from(mil.capital_distance));
    w.ratio(f64::from(dist), f64::from(mil.capital_distance));
    // Which victory-point band the pawn sits in, signed: end-of-game military
    // scoring is a step function, so the band is worth stating categorically
    // rather than leaving a linear layer to rediscover it from the distance.
    // The bands are 0 / 1-2 / 3-5 / 6-8 either way, so seven signed cells.
    let band = mil
        .victory_points
        .iter()
        .position(|&(max, _)| dist <= u32::from(max))
        .unwrap_or(mil.victory_points.len() - 1);
    let signed_band = if pawn == 0 {
        3
    } else if pawn > 0 {
        3 + band
    } else {
        3 - band
    };
    w.one_hot(signed_band, 7);
    let vp = f64::from(mil.vp_for_distance(dist.min(u32::from(mil.capital_distance)) as u8));
    w.ratio(if pawn >= 0 { vp } else { -vp }, 10.0);

    let my_shields = state.player(me).shields();
    let op_shields = state.player(opp).shields();
    w.ratio(f64::from(my_shields), 12.0);
    w.ratio(f64::from(op_shields), 12.0);
    w.ratio(
        f64::from(i32::from(my_shields) - i32::from(op_shields)),
        12.0,
    );

    // Loot tokens: whether each of the two on `me`'s side of the track, and
    // each of the two on the opponent's, has already been collected.
    for p in [me, opp] {
        for i in 0..2 {
            w.flag(!state.loot_available(p, i));
        }
    }
    w.flag(state.military_leader() == Some(me));
    w.flag(state.military_leader() == Some(opp));
    // The Strategy token turns every red card into two shields, which changes
    // what a race is worth to whoever holds it.
    let strategy = TokenId::from_slug("strategy");
    w.flag(strategy.is_some_and(|t| state.player(me).has_token(t)));
    w.flag(strategy.is_some_and(|t| state.player(opp).has_token(t)));
}

/// The science race — the rarest of the three win conditions (~2.3% of games)
/// and the reason this model has a per-kind head at all. 26 features.
fn science(w: &mut Writer, state: &GameState, me: Player, opp: Player) {
    for p in [me, opp] {
        let sci = state.player(p).science();
        // Per symbol, capped at two: a third copy of a symbol is worth nothing
        // at all towards either a pair or the sixth distinct symbol.
        for count in sci {
            w.ratio(f64::from(count.min(2)), 2.0);
        }
    }
    for p in [me, opp] {
        let d = state.player(p).distinct_science();
        w.ratio(f64::from(d), 6.0);
        // One symbol from scientific supremacy is a categorically different
        // position from two, and the difference is not linear in the count.
        w.flag(d >= 5);
        w.flag(d >= 4);
        w.ratio(state.player(p).pairs_awarded().count() as f64, 6.0);
    }
    // What is still available: how many progress tokens are on the board at
    // all, and where Law is — the only source of a seventh symbol, and so a
    // shortcut to supremacy that does not need a green card.
    w.ratio(f64::from(state.board_tokens_mask().count_ones()), 5.0);
    let law = TokenId::from_slug("law");
    w.flag(law.is_some_and(|t| state.board_tokens_mask() & (1u16 << t.index()) != 0));
    w.flag(law.is_some_and(|t| state.player(me).has_token(t)));
    w.flag(law.is_some_and(|t| state.player(opp).has_token(t)));
}

/// Coins, production, and what a missing resource costs. 29 features.
fn economy(w: &mut Writer, state: &GameState, me: Player, opp: Player) {
    let my_coins = state.player(me).coins();
    let op_coins = state.player(opp).coins();
    w.ratio(f64::from(my_coins), 20.0);
    w.ratio(f64::from(op_coins), 20.0);
    w.ratio(f64::from(i32::from(my_coins) - i32::from(op_coins)), 20.0);

    for p in [me, opp] {
        let prod = state.player(p).production();
        for r in Resource::ALL {
            w.ratio(f64::from(prod[r.index()]), 4.0);
        }
        let (raw, manufactured) = state.player(p).choice_sources();
        w.ratio(f64::from(raw), 3.0);
        w.ratio(f64::from(manufactured), 3.0);
    }
    // Trade prices are the compact statement of "how expensive is the rest of
    // the game for this player": two coins a unit is a healthy city, six is a
    // player about to be priced out of Age III.
    for p in [me, opp] {
        let prices = cost::trade_prices(state, p);
        for r in Resource::ALL {
            w.ratio(f64::from(prices[r.index()]), 6.0);
        }
    }
    // The Economy token redirects the opponent's trade payments, which turns
    // their weak production into the holder's income.
    let economy = TokenId::from_slug("economy");
    w.flag(economy.is_some_and(|t| state.player(me).has_token(t)));
    w.flag(economy.is_some_and(|t| state.player(opp).has_token(t)));
}

/// What each city is made of. 22 features.
fn city(w: &mut Writer, state: &GameState, me: Player, opp: Player) {
    for p in [me, opp] {
        let mut by_kind = [0u8; 7];
        for card in state.player(p).built() {
            by_kind[card.def().kind.index()] += 1;
        }
        for k in CardType::ALL {
            w.ratio(f64::from(by_kind[k.index()]), 8.0);
        }
    }
    for p in [me, opp] {
        let ps = state.player(p);
        let built_wonders = ps.wonders_built().count();
        // A drafted-but-unbuilt wonder is a different asset from a built one:
        // one is a plan, the other is banked.
        w.ratio(f64::from(ps.wonder_count()) - built_wonders as f64, 4.0);
        w.ratio(built_wonders as f64, 4.0);
        w.ratio(f64::from(ps.token_count()), 5.0);
        w.ratio(ps.built().count() as f64, 20.0);
    }
}

/// The victory-point breakdown as `duels-core` computes it. 20 features.
///
/// This is the most direct signal in the vector — for the ~97% of games that
/// end on points, the winner is whoever is ahead here at the end — and it
/// comes from the rules authority rather than being re-derived, which is why
/// `duels-core` is this crate's only dependency.
fn victory_points(w: &mut Writer, state: &GameState, me: Player, opp: Player) {
    let mine = scoring::breakdown(state, me);
    let theirs = scoring::breakdown(state, opp);
    for b in [&mine, &theirs] {
        push_breakdown(w, b);
    }
    w.ratio(
        f64::from(i32::from(mine.total) - i32::from(theirs.total)),
        20.0,
    );
    w.ratio(
        f64::from(i32::from(mine.civilian) - i32::from(theirs.civilian)),
        15.0,
    );
}

/// One player's nine `Breakdown` categories.
fn push_breakdown(w: &mut Writer, b: &Breakdown) {
    w.ratio(f64::from(b.civilian), 20.0);
    w.ratio(f64::from(b.scientific), 12.0);
    w.ratio(f64::from(b.commercial), 12.0);
    w.ratio(f64::from(b.guilds), 15.0);
    w.ratio(f64::from(b.wonders), 15.0);
    w.ratio(f64::from(b.progress_tokens), 12.0);
    w.ratio(f64::from(b.military), 10.0);
    w.ratio(f64::from(b.coins), 8.0);
    w.ratio(f64::from(b.total), 60.0);
}

/// What is on the table to be taken, and how much game is left. 26 features.
fn structure(w: &mut Writer, state: &GameState, me: Player, opp: Player) {
    let accessible = state.accessible_slots();
    w.ratio(f64::from(accessible.count_ones()), 6.0);
    w.ratio(
        f64::from((state.occupied_slots() & !state.revealed_slots()).count_ones()),
        12.0,
    );
    w.ratio(f64::from(state.discard_mask().count_ones()), 10.0);

    // Aggregates over the cards that can actually be taken this turn. A
    // face-down accessible slot contributes nothing but the counts above,
    // because its identity is not public.
    let mut by_kind = [0u8; 7];
    let mut vp = 0u16;
    let mut shields = 0u8;
    let mut symbols = 0u8;
    let mut affordable_me = 0u8;
    let mut affordable_opp = 0u8;
    let mut cheapest_me: Option<u16> = None;
    for slot in 0..SLOTS {
        if accessible & (1u32 << slot) == 0 {
            continue;
        }
        let Some(card) = state.face_up_card(slot as u8) else {
            continue;
        };
        let def = card.def();
        by_kind[def.kind.index()] += 1;
        vp += u16::from(def.victory_points);
        shields += def.shields;
        symbols += u8::from(def.science.is_some());
        let mine = cost::card_cost(state, me, card);
        let theirs = cost::card_cost(state, opp, card);
        affordable_me += u8::from(mine.affordable_by(state, me));
        affordable_opp += u8::from(theirs.affordable_by(state, opp));
        cheapest_me = Some(cheapest_me.map_or(mine.coins, |c| c.min(mine.coins)));
    }
    for k in CardType::ALL {
        w.ratio(f64::from(by_kind[k.index()]), 4.0);
    }
    w.ratio(f64::from(vp), 12.0);
    w.ratio(f64::from(shields), 4.0);
    w.ratio(f64::from(symbols), 3.0);
    w.ratio(f64::from(affordable_me), 6.0);
    w.ratio(f64::from(affordable_opp), 6.0);
    w.ratio(f64::from(cheapest_me.unwrap_or(0)), 8.0);

    // An outstanding effect choice changes what the position even means, so it
    // is stated categorically. A destroy and the Mausoleum are the two that
    // can move a lot of value in a single step.
    let pending = match state.pending() {
        None => 0,
        Some(Pending::ProgressToken) => 1,
        Some(Pending::GreatLibraryToken { .. }) => 2,
        Some(Pending::Destroy { .. }) => 3,
        Some(Pending::MausoleumBuild) => 4,
    };
    w.one_hot(pending, 5);
    // How much of the game is left to play, which is what makes a lead safe or
    // not: cards still in this age's structure plus whole ages still to come.
    w.ratio(
        f64::from(u32::from(3 - state.age().clamp(1, 3)) * SLOTS as u32)
            + f64::from(state.occupied_slots().count_ones()),
        60.0,
    );
    w.flag(state.wonder_slots_left());
    w.ratio(f64::from(state.wonders_built_total()), 8.0);
    // What a discard is worth right now, which is the floor under every other
    // option and rises with the number of yellow cards a player holds.
    w.ratio(f64::from(cost::discard_reward(state, me)), 8.0);
    w.ratio(f64::from(cost::discard_reward(state, opp)), 8.0);
}

/// Which of the twelve wonders each side holds, and how far along. 24 features.
///
/// One feature per wonder per player, three-valued: `0` not owned, `0.5`
/// drafted but unbuilt, `1` built. Wonders are individually very unequal — the
/// extra-turn ones are the strongest class (see `docs/strategy-backlog.md`) —
/// so identity is worth carrying and not only a count.
fn wonders(w: &mut Writer, state: &GameState, me: Player, opp: Player) {
    for p in [me, opp] {
        let ps = state.player(p);
        for i in 0..NUM_WONDERS {
            let id = WonderId::from_index(i);
            w.push(if ps.has_built_wonder(id) {
                1.0
            } else if ps.owns_wonder(id) {
                0.5
            } else {
                0.0
            });
        }
    }
}

/// Which of the ten progress tokens are where. 30 features.
///
/// Owned by `me`, owned by the opponent, and still on the board — three
/// disjoint indicators per token, so "nobody has Theology and it is gone"
/// (set aside at setup) is distinguishable from "it is still there to take".
fn tokens(w: &mut Writer, state: &GameState, me: Player, opp: Player) {
    let board = state.board_tokens_mask();
    for p in [me, opp] {
        for i in 0..NUM_TOKENS {
            w.flag(state.player(p).has_token(TokenId::from_index(i)));
        }
    }
    for i in 0..NUM_TOKENS {
        w.flag(board & (1u16 << i) != 0);
    }
}

// The section widths in the doc comments above have to add up, and the game
// data this layout is shaped around has to stay the shape it was written for.
const _: () = assert!(NUM_FEATURES == 13 + 21 + 26 + 29 + 22 + 20 + 26 + 24 + 30);
const _: () = assert!(NUM_RESOURCES == 5 && NUM_SCIENCE == 7);
const _: () = assert!(NUM_TOKENS == 10 && NUM_WONDERS == 12);

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Reproducible mid-game positions across all three ages and both movers,
    /// named by seed rather than checked in — the same fixture shape
    /// `mcts-eval`'s leaf tests use.
    pub(super) fn positions() -> Vec<GameState> {
        let mut out = Vec::new();
        for seed in 0..40u64 {
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x5A17_5A17);
            for _ in 0..(6 + seed % 60) {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let a = legal[rng.gen_range(0..legal.len())];
                engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action applies");
            }
            out.push(state);
        }
        out
    }

    /// `features` debug-asserts this, but stating it as its own test means a
    /// failure names the invariant rather than an internal assert, and it
    /// holds in release builds too.
    #[test]
    fn the_layout_fills_the_vector_exactly() {
        for state in positions() {
            let mut out = [0.0f32; NUM_FEATURES];
            let mut w = Writer {
                out: &mut out,
                n: 0,
            };
            clock(&mut w, &state, Player::One);
            military(&mut w, &state, Player::One, Player::Two);
            science(&mut w, &state, Player::One, Player::Two);
            economy(&mut w, &state, Player::One, Player::Two);
            city(&mut w, &state, Player::One, Player::Two);
            victory_points(&mut w, &state, Player::One, Player::Two);
            structure(&mut w, &state, Player::One, Player::Two);
            wonders(&mut w, &state, Player::One, Player::Two);
            tokens(&mut w, &state, Player::One, Player::Two);
            assert_eq!(w.n, NUM_FEATURES);
        }
    }

    #[test]
    fn every_feature_is_finite_and_roughly_scaled() {
        for (i, state) in positions().into_iter().enumerate() {
            for me in Player::ALL {
                let f = features(&state, me);
                for (k, v) in f.iter().enumerate() {
                    assert!(v.is_finite(), "position {i} feature {k} is {v}");
                    assert!(
                        v.abs() <= 4.0,
                        "position {i} feature {k} is {v}, far outside the intended scale"
                    );
                }
            }
        }
    }

    /// The perspective convention: the vector carries no seat bit, so the two
    /// readings of one position have to differ. Checked in the weak form that
    /// they *do* differ, which is what catches a section that forgot to
    /// relativise a player-specific quantity.
    #[test]
    fn the_two_perspectives_are_different_readings_of_the_same_position() {
        let mut differing = 0;
        for state in positions() {
            let a = features(&state, Player::One);
            let b = features(&state, Player::Two);
            if a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()) {
                differing += 1;
            }
        }
        assert!(
            differing >= 35,
            "only {differing} positions read differently from the two seats"
        );
    }

    /// The sharp end of the relativisation: the signed pawn flips with the
    /// perspective and the unsigned distance does not.
    #[test]
    fn the_conflict_band_is_relativised() {
        use duels_core::testing::StateBuilder;
        let state = StateBuilder::new().age(2).conflict(7).build();
        let one = features(&state, Player::One);
        let two = features(&state, Player::Two);
        // The signed pawn is the first military feature, immediately after the
        // thirteen clock features.
        let pawn = 13;
        assert!(one[pawn] > 0.0, "Player One is ahead on a +7 pawn");
        assert!(
            (one[pawn] + two[pawn]).abs() < 1e-6,
            "the pawn is not antisymmetric: {} vs {}",
            one[pawn],
            two[pawn]
        );
        assert!(
            (one[pawn + 1] - two[pawn + 1]).abs() < 1e-6,
            "the distance should not depend on the perspective"
        );
    }
}
