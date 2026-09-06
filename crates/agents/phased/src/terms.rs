//! The individual evaluation terms, each read for **one** player.
//!
//! Every function here takes a player and returns that player's own value for
//! one idea; [`crate::evaluate`] differences the two players and applies the
//! root-fixed weights from [`crate::blend`]. Writing them per player rather
//! than as differences is what lets each side's term carry *its own* owner's
//! commitment multiplier — a science race is more important to the player who
//! is in it, whichever of the two that is.
//!
//! Everything here reads only public information and only `duels-core`'s own
//! accessors and cost/scoring engines. Nothing re-implements a rule.

use duels_core::cost;
use duels_core::data::{self, CardId, Resource, TokenId, WonderId, NUM_RESOURCES};
use duels_core::scoring::Breakdown;
use duels_core::state::Phase;
use duels_core::{GameState, Player};
use duels_strategy::board::Board;
use duels_strategy::masks::{iter_cards, masks, DECISIONS_PER_AGE};
use duels_strategy::military::signed_distance;
use duels_strategy::science::token_value;

use crate::{EvalWeights, ScienceWeights};

/// The largest number of units of one resource the development term prices.
///
/// No card or wonder in the base game asks for more than three of anything,
/// so a fourth unit of the same resource is already worth nothing; the table
/// is sized one past that so the zero is visible rather than assumed.
pub const MAX_UNITS: usize = 4;

// ---------------------------------------------------------------------------
// Decision budget
// ---------------------------------------------------------------------------

/// Roughly how many more decisions `p` gets this game.
///
/// A deliberately cheap restatement of [`duels_strategy::Tempo::decisions_left`]
/// — the mover gets the odd card of the current structure, plus
/// [`DECISIONS_PER_AGE`] for every age not yet dealt. `Tempo` itself is
/// computed from a whole [`duels_strategy::Context`], which is far too
/// expensive to rebuild for every candidate action against every chance
/// outcome; this is a handful of integer operations and agrees with it
/// exactly on the parts it models (it omits `Tempo`'s extra-turn adjustment,
/// which is a fractional correction).
pub fn decisions_left(state: &GameState, p: Player) -> f64 {
    let cards_left = state.occupied_slots().count_ones();
    let picks = if state.current_player() == p {
        cards_left.div_ceil(2)
    } else {
        cards_left / 2
    };
    let age = u32::from(state.age().max(1));
    // During the wonder draft the current age's structure has not been laid
    // out yet, so the current age is itself still undealt.
    let first_undealt = if state.phase() == Phase::WonderDraft {
        age
    } else {
        age + 1
    };
    let undealt = 4u32.saturating_sub(first_undealt).min(3);
    f64::from(picks + undealt * u32::from(DECISIONS_PER_AGE))
}

// ---------------------------------------------------------------------------
// Development
// ---------------------------------------------------------------------------

/// How much of the card pool still to come wants each unit of each resource.
///
/// `f[k - 1][r]` is the fraction of the pool whose printed cost demands a
/// `k`-th unit of resource `r`. The pool is the current age's unknown cards
/// plus every age not yet dealt — all of it public information (it is
/// [`Board::unknown_pool`], which
/// `duels-strategy`'s own test pins to [`duels_core::engine::hidden_info`]'s
/// pool, plus whole undealt decks).
///
/// This is a *supply statistic*, not a position: taking one card out of a
/// sixty-card pool moves it by less than the rounding on any weight it feeds.
/// It is therefore computed once per decision at the root, like every weight,
/// while the parts a move genuinely changes — what the player produces, what
/// they would have to pay, which wonders they still owe resources for — are
/// measured on the post-action state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DevSupply {
    /// `f[k - 1][r]`: fraction of the pool needing a `k`-th unit of `r`.
    pub f: [[f64; NUM_RESOURCES]; MAX_UNITS],
    /// How many cards the pool holds. Zero means every fraction is zero.
    pub pool_size: u32,
}

impl DevSupply {
    /// Read the supply statistics off a root position's board.
    pub fn of(board: &Board) -> DevSupply {
        let s = data::statics();
        let mut pool = board.unknown_pool;
        for age in board.undealt_ages() {
            if age >= 1 && usize::from(age) <= s.age_masks.len() {
                pool |= s.age_masks[usize::from(age) - 1];
            }
        }

        let pool_size = pool.count_ones();
        let mut counts = [[0u32; NUM_RESOURCES]; MAX_UNITS];
        for card in iter_cards(pool) {
            let cost = card.def().resource_cost;
            for (r, &need) in cost.iter().enumerate() {
                for (k, row) in counts.iter_mut().enumerate() {
                    if u32::from(need) >= (k + 1) as u32 {
                        row[r] += 1;
                    }
                }
            }
        }

        let scale = if pool_size == 0 {
            0.0
        } else {
            1.0 / f64::from(pool_size)
        };
        DevSupply {
            f: std::array::from_fn(|k| std::array::from_fn(|r| f64::from(counts[k][r]) * scale)),
            pool_size,
        }
    }
}

/// What `p`'s own production is worth, in coins they will not have to spend.
///
/// For each unit of each resource the player actually produces, the term asks
/// how often that unit will be *wanted* — by the cards still to come
/// (`f_k(r)` of the pool, over the `take_rate × decisions_left` further cards
/// this player expects to build) and by their own drafted, unbuilt wonders
/// (a certain want, weight one) — and prices each want at the trade price
/// they would otherwise pay ([`duels_core::cost::trade_prices`]: `2 +` the
/// opponent's production, or a flat 1 behind a trading post).
///
/// A "produce one of your choice" source (Forum, Caravansery, Piraeus, The
/// Great Lighthouse) is valued as one more unit of whichever resource of its
/// group is currently worth the most, greedily and one at a time.
///
/// A trading post is valued separately, by the discount it gives on the units
/// the player does *not* produce — because for the units they do produce, the
/// post has already reduced `price_r` to 1 and so quietly deflated the main
/// term.
///
/// Nothing here says "grey is good": the value of a glass card falls out of
/// how many pool cards and wonders ask for glass, against what glass costs.
/// `examples/watch_blend.rs` prints the per-resource split so that claim can
/// be checked rather than believed.
pub fn development_value(state: &GameState, p: Player, supply: &DevSupply, take_rate: f64) -> f64 {
    development_by_resource(state, p, supply, take_rate)
        .iter()
        .sum()
}

/// [`development_value`] split by resource, for the diagnostics.
pub fn development_by_resource(
    state: &GameState,
    p: Player,
    supply: &DevSupply,
    take_rate: f64,
) -> [f64; NUM_RESOURCES] {
    let me = state.player(p);
    let prices = cost::trade_prices(state, p);
    let n_take = (take_rate * decisions_left(state, p)).max(0.0);

    // Wonders this player has drafted and not yet built are a *certain* want:
    // they will be paid for out of this city or not at all.
    let mut wonder = [[0.0f64; NUM_RESOURCES]; MAX_UNITS];
    for w in me.wonders() {
        if me.has_built_wonder(w) {
            continue;
        }
        let cost = w.def().resource_cost;
        for (r, &need) in cost.iter().enumerate() {
            for (k, row) in wonder.iter_mut().enumerate() {
                if u32::from(need) >= (k + 1) as u32 {
                    row[r] += 1.0;
                }
            }
        }
    }

    // The marginal worth of the `k`-th unit of `r`, zero past the table.
    let marginal = |k: usize, r: usize| -> f64 {
        if k == 0 || k > MAX_UNITS {
            return 0.0;
        }
        (supply.f[k - 1][r] * n_take + wonder[k - 1][r]) * f64::from(prices[r])
    };

    let production = me.production();
    let mut have: [usize; NUM_RESOURCES] = std::array::from_fn(|r| usize::from(production[r]));
    let mut out = [0.0f64; NUM_RESOURCES];
    for (r, &n) in have.iter().enumerate() {
        for k in 1..=n.min(MAX_UNITS) {
            out[r] += marginal(k, r);
        }
    }

    // Choice sources stand in for one more unit of the best-valued resource
    // in their group, taken greedily so two of them do not both claim the
    // same slot.
    let (choice_raw, choice_manufactured) = me.choice_sources();
    for (count, raw) in [(choice_raw, true), (choice_manufactured, false)] {
        for _ in 0..count {
            let mut best: Option<(usize, f64)> = None;
            for (r, resource) in Resource::ALL.iter().enumerate() {
                if resource.is_raw() != raw {
                    continue;
                }
                let v = marginal(have[r] + 1, r);
                if best.is_none_or(|(_, b)| v > b) {
                    best = Some((r, v));
                }
            }
            if let Some((r, v)) = best {
                out[r] += v;
                have[r] += 1;
            }
        }
    }

    // A trading post is worth the gouging it prevents on the units this city
    // still cannot make for itself.
    let fixed = me.fixed_trade();
    let opponent = state.player(p.other());
    for (r, &has_post) in fixed.iter().enumerate() {
        if !has_post {
            continue;
        }
        let raw_price = 2.0 + f64::from(opponent.trade_relevant_production(Resource::ALL[r]));
        let discount = (raw_price - 1.0).max(0.0);
        let mut deficit = 0.0;
        for k in (have[r] + 1)..=MAX_UNITS {
            deficit += supply.f[k - 1][r] * n_take + wonder[k - 1][r];
        }
        out[r] += discount * deficit;
    }

    out
}

// ---------------------------------------------------------------------------
// Next-age-start tempo
// ---------------------------------------------------------------------------

/// Who is projected to begin the next age.
///
/// Verified against `duels_core::engine::end_age`:
///
/// * if the conflict pawn is **not** centred, the militarily *weaker* player
///   chooses who starts — and will choose themselves;
/// * if it **is** centred, the player who took the age's last card simply
///   begins, with no choice at all;
/// * an extra turn still pending when the age ends is lost.
///
/// So while the age is running, only the player currently projected to take
/// its *last* card is the one whose "start the next age" claim a slow, low-
/// development card would actually forfeit. That projection is a parity
/// count on the cards left, flipped by a banked extra turn (the mover then
/// takes two in a row). Returns `None` when the question is not yet meaningful
/// — during the wonder draft, or once the game is over.
///
/// **Known approximation**: a play-again wonder built later in the age flips
/// the parity again and is not modelled, and the pawn can move before the age
/// ends, changing who the weaker player is. Both are accepted: the term is a
/// tie-breaker worth a few points, and modelling them exactly would need the
/// search this agent does not have.
pub fn projected_starter(state: &GameState) -> Option<Player> {
    if state.is_over() {
        return None;
    }
    match state.phase() {
        // The weaker player has been asked and is about to answer; they will
        // say themselves.
        Phase::ChooseFirstPlayer => Some(state.current_player()),
        Phase::WonderDraft | Phase::GameOver => None,
        Phase::Turn => match state.military_leader() {
            Some(leader) => Some(leader.other()),
            None => {
                let cards_left = state.occupied_slots().count_ones();
                if cards_left == 0 {
                    return Some(state.last_card_taker());
                }
                let n = cards_left + u32::from(state.extra_turn());
                Some(if n % 2 == 1 {
                    state.current_player()
                } else {
                    state.current_player().other()
                })
            }
        },
    }
}

/// What beginning the next age is worth to `p` in this position.
///
/// Not commitment-scaled: the value of moving first into a fresh structure is
/// about the cards, not about anybody's plan. Zero in Age III by default,
/// where there is no next age to begin.
pub fn next_age_start(state: &GameState, p: Player, w: &EvalWeights) -> f64 {
    if projected_starter(state) != Some(p) {
        return 0.0;
    }
    let age = usize::from(state.age().max(1));
    let i = age.saturating_sub(1).min(w.next_age_start.len() - 1);
    w.next_age_start[i]
}

// ---------------------------------------------------------------------------
// Science ladder
// ---------------------------------------------------------------------------

/// Whether the second copy of `symbol` is still physically obtainable.
///
/// A cheap, public-information restatement of the part of
/// [`duels_strategy::ScienceRead::second_copy_obtainable`] this term needs: a
/// copy is gone once it is in a city, spent under a wonder, or in the discard
/// pile, and a copy printed on a card of an *earlier* age than the one being
/// played was returned to the box if it never appeared. The full read is far
/// too expensive to rebuild per candidate action per chance outcome.
///
/// Slightly optimistic: a current-age copy may be one of the three cards
/// boxed unseen at setup.
pub fn second_copy_obtainable(state: &GameState, symbol: data::Science) -> bool {
    let gone = state.player(Player::One).built_mask()
        | state.player(Player::Two).built_mask()
        | state.wonder_fodder_mask()
        | state.discard_mask();
    let age = state.age().max(1);
    iter_cards(masks().symbol_mask(symbol))
        .any(|c| gone & (1u128 << c.index()) == 0 && c.def().age >= age)
}

/// How many of the three progress tokens that a science player most wants —
/// Law (a seventh symbol), Theology (an extra turn per wonder) and Strategy
/// (a shield per red card) — are still on the board to be claimed.
///
/// Resolved by *effect* via [`duels_strategy::masks`], not by slug, so a
/// rename in `data/tokens.json` cannot silently break it.
pub fn strong_board_tokens(state: &GameState) -> f64 {
    let m = masks();
    let strong: [Option<TokenId>; 3] = [m.law_token(), m.theology_token(), m.strategy_token()];
    state
        .board_tokens()
        .filter(|t| strong.contains(&Some(*t)))
        .count() as f64
}

/// The value of the half-pairs `p` is sitting on: each one is a progress
/// token the moment its twin is built, and a turn the opponent has to spend
/// if they would rather it were not.
fn pair_threat(state: &GameState, p: Player, w: &ScienceWeights) -> f64 {
    let m = masks();
    let me = state.player(p);
    let held = me.science();
    let mut awarded = 0u8;
    for sym in me.pairs_awarded() {
        awarded |= 1u8 << sym.index();
    }

    let mut candidates = 0.0f64;
    for sym in duels_strategy::masks::ALL_SCIENCE {
        let i = sym.index();
        // Balance only ever comes from the Law token, so it can never pair.
        if held[i] != 1 || Some(sym) == m.law_symbol() || awarded & (1u8 << i) != 0 {
            continue;
        }
        if second_copy_obtainable(state, sym) {
            candidates += 1.0;
        }
    }
    if candidates == 0.0 {
        return 0.0;
    }

    // What completing a pair would actually pay: the best token currently on
    // the board, priced by `duels-strategy`'s own calibration rather than by
    // a number invented here.
    let best_token = state
        .board_tokens()
        .map(|t| token_value(state, p, t))
        .fold(0.0f64, f64::max);
    candidates * (w.pair_token_share * best_token + w.pair_tempo_tax)
}

/// How much `p`'s scientific position is worth: a convex ladder in distinct
/// symbols, scaled by how good the tokens on offer are, plus the half-pairs
/// they are threatening to complete.
///
/// Convex on purpose. The fourth symbol is worth far more than the second:
/// after Age I's four symbols are dealt and taken, *every* Age II green card
/// completes a pair (Ages I and II each carry one copy of the same four
/// symbols), and the sixth symbol needs one of Age III's two exclusive
/// symbols or the Law token. A linear "points per symbol" term cannot see any
/// of that.
///
/// Six distinct symbols wins outright and is handled by the terminal check,
/// so the ladder's last entry is five.
pub fn science_ladder(state: &GameState, p: Player, w: &ScienceWeights) -> f64 {
    let distinct = usize::from(state.player(p).distinct_science()).min(w.ladder.len() - 1);
    let token_mult = 1.0 + w.strong_token_mult * strong_board_tokens(state);
    w.ladder[distinct] * token_mult + w.pair_threat_weight * pair_threat(state, p, w)
}

// ---------------------------------------------------------------------------
// Military, points, economy, tactics
// ---------------------------------------------------------------------------

/// The conflict pawn's position, signed so positive favours `p`.
pub fn military_position(state: &GameState, p: Player) -> f64 {
    f64::from(signed_distance(state, p))
}

/// The escalating "somebody is about to win outright" term: quadratic in how
/// far past the second loot token the pawn is, signed towards whoever it
/// favours.
///
/// Deliberately **not** commitment-scaled. It is the one military term that
/// is as much about the opponent's race as about this player's own plan, and
/// a player who is not committed to military still has to notice a pawn three
/// steps from their capital.
pub fn military_urgency(state: &GameState, p: Player) -> f64 {
    let signed = military_position(state, p);
    let second_loot = f64::from(data::military().loot[1].0);
    if signed.abs() > second_loot {
        let excess = signed.abs() - second_loot;
        signed.signum() * excess * excess
    } else {
        0.0
    }
}

/// The "as if the game ended now" points from cards, wonders and tokens.
///
/// Straight out of [`duels_core::scoring::breakdown`], excluding military
/// (which [`military_position`] reads as board position, not points) and
/// coins (which get their own liquidity terms).
pub fn card_and_token_vp(b: &Breakdown) -> f64 {
    f64::from(b.civilian + b.scientific + b.commercial + b.guilds + b.wonders + b.progress_tokens)
}

/// A rough, hand-tuned "how strong is this wonder" score, summed over `p`'s
/// drafted-but-unbuilt wonders. Without it the evaluation cannot tell two
/// wonder-draft picks apart, since the point projection only credits *built*
/// wonders.
pub fn wonder_potential(state: &GameState, p: Player) -> f64 {
    let ps = state.player(p);
    ps.wonders()
        .filter(|&w| !ps.has_built_wonder(w))
        .map(wonder_power)
        .sum()
}

fn wonder_power(w: WonderId) -> f64 {
    let def = w.def();
    let mut v = f64::from(def.victory_points)
        + f64::from(def.coins) * 0.3
        + f64::from(def.shields)
        + f64::from(def.opponent_loses_coins) * 0.3;
    for flag in [
        def.play_again,
        def.destroy.is_some(),
        def.build_discarded_free,
        def.choose_progress_token,
    ] {
        if flag {
            v += 3.0;
        }
    }
    if def.produces_choice.is_some() {
        v += 2.0;
    }
    v
}

/// Cash on hand, capped: what it takes to actually *pay* for the race card
/// the moment it turns up, rather than watch it go.
///
/// Capped because past the price of one contested card, further coins are
/// points rather than liquidity, and the points are already counted.
pub fn race_liquidity(state: &GameState, p: Player, cap: f64) -> f64 {
    f64::from(state.player(p).coins()).min(cap)
}

/// How far below a safe coin cushion `p` is.
pub fn coin_shortfall(state: &GameState, p: Player, floor: f64) -> f64 {
    (floor - f64::from(state.player(p).coins())).max(0.0)
}

/// `p`'s average per-unit trade price: how exposed they are to being gouged.
pub fn average_trade_price(state: &GameState, p: Player) -> f64 {
    let prices = cost::trade_prices(state, p);
    prices.iter().map(|&c| f64::from(c)).sum::<f64>() / prices.len() as f64
}

/// The total value of the free chain builds `p` is about to hand their
/// opponent, if it is genuinely the opponent's turn next.
///
/// Deliberately shallow — no search of the opponent's actual best reply, just
/// a refusal to give away an obviously free, valuable card when an
/// alternative move avoids it.
///
/// # Why `root_age` is a parameter
///
/// This is the only term that reads a *card in the structure*, and that makes
/// it the only one that can leak hidden information. Within one age it is
/// safe: a move that turns a face-down slot over is a real chance event, the
/// engine enumerates every way it can resolve
/// ([`duels_core::engine::chance_outcomes`]), and averaging over them is
/// exactly what this agent does. But a move that empties the structure ends
/// the age, and the engine then deals the *whole next age* from a deck the
/// observation knows nothing about — randomness that is not enumerated as
/// chance outcomes and is instead invented wholesale by
/// [`duels_core::Observation::sample_state`]. Reading those cards would make
/// an age-ending discard score differently depending on which world the
/// throwaway sample happened to invent.
///
/// So the term is switched off the moment the age has moved on from the root
/// position's. `tests/determinization_invariance.rs::
/// resampling_the_same_observation_changes_nothing` is the test that found
/// this, and is the test that keeps it fixed.
pub fn chain_gift_exposure(state: &GameState, p: Player, root_age: u8) -> f64 {
    let taker = p.other();
    if state.age() != root_age || state.phase() != Phase::Turn || state.current_player() != taker {
        return 0.0;
    }
    let mut value = 0.0;
    let mut mask = state.accessible_slots();
    while mask != 0 {
        let slot = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        if let Some(card) = state.face_up_card(slot) {
            let def = card.def();
            if let Some(prereq) = def.chain_from {
                if state.player(taker).has_built(prereq) {
                    value += 2.0 + f64::from(def.victory_points);
                }
            }
        }
    }
    value
}

/// Every card carrying `symbol`, for the diagnostics.
pub fn symbol_cards(symbol: data::Science) -> impl Iterator<Item = CardId> {
    iter_cards(masks().symbol_mask(symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use duels_core::testing::StateBuilder;

    #[test]
    fn the_decision_budget_agrees_with_duels_strategys_tempo() {
        for seed in 0..6u64 {
            let mut st = engine::new_game(seed);
            let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
            for _ in 0..40 {
                if st.is_over() {
                    break;
                }
                let ctx = duels_strategy::Context::of(&st);
                for p in Player::ALL {
                    let want = f64::from(ctx.tempo(p).decisions_left);
                    let got = decisions_left(&st, p);
                    assert_eq!(want, got, "seed {seed}: decisions_left for {p:?}");
                }
                let legal = engine::legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                let action = legal[st.turn() as usize % legal.len()];
                engine::apply_quiet(&mut st, action, &mut rng).unwrap();
            }
        }
    }

    #[test]
    fn the_projected_starter_matches_who_really_begins_the_next_age() {
        // Pawn centred, one card left: whoever takes it begins Age II.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(19, "clay-pool")])
            .conflict(0)
            .current(Player::One)
            .build();
        assert_eq!(projected_starter(&st), Some(Player::One));
        let mut after = st;
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(1);
        engine::apply_quiet(
            &mut after,
            duels_core::Action::Discard { slot: 19 },
            &mut rng,
        )
        .unwrap();
        assert_eq!(after.age(), 2);
        assert_eq!(after.current_player(), Player::One);

        // Two cards left: the mover takes the first, the opponent the last.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .conflict(0)
            .current(Player::One)
            .build();
        assert_eq!(projected_starter(&st), Some(Player::Two));

        // ...unless the mover has a banked extra turn and takes both.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .conflict(0)
            .extra_turn(true)
            .current(Player::One)
            .build();
        assert_eq!(projected_starter(&st), Some(Player::One));
    }

    #[test]
    fn an_off_centre_pawn_hands_the_choice_to_the_weaker_player() {
        // Player One leads militarily, so Player Two chooses and picks
        // themselves — regardless of who would take the last card.
        for cards in 1..=3u8 {
            let slots: Vec<(u8, &str)> = (0..cards).map(|i| (19 - i, "clay-pool")).collect();
            let st = StateBuilder::new()
                .age(1)
                .open_slots(&slots)
                .conflict(3)
                .current(Player::One)
                .build();
            assert_eq!(
                projected_starter(&st),
                Some(Player::Two),
                "{cards} cards left"
            );
        }
    }

    #[test]
    fn the_science_ladder_is_convex_and_rewards_a_strong_token_row() {
        let w = ScienceWeights::default();
        let mut last_step = 0.0;
        for i in 1..w.ladder.len() {
            let step = w.ladder[i] - w.ladder[i - 1];
            assert!(
                step >= last_step,
                "the ladder is not convex at {i}: {step} < {last_step}"
            );
            last_step = step;
        }

        // Two token rows with the *same* best token, so the only thing that
        // differs is how many of Law / Theology / Strategy are on offer.
        let four = ["workshop", "apothecary", "scriptorium", "pharmacist"];
        let row = |tokens: &[&str]| {
            StateBuilder::new()
                .age(2)
                .built(Player::One, &four)
                .board_tokens(tokens)
                .build()
        };
        let plain = row(&["philosophy"]);
        let strong = row(&["philosophy", "law", "theology"]);
        assert_eq!(plain.player(Player::One).distinct_science(), 4);
        assert_eq!(strong_board_tokens(&plain), 0.0);
        assert_eq!(strong_board_tokens(&strong), 2.0);
        let plain_v = science_ladder(&plain, Player::One, &w);
        let strong_v = science_ladder(&strong, Player::One, &w);
        assert!(
            strong_v > plain_v,
            "law/theology/strategy on the board should read higher: {strong_v} vs {plain_v}"
        );
    }

    #[test]
    fn a_second_copy_is_unobtainable_once_both_copies_are_spoken_for() {
        let mortar = data::Science::Mortar;
        let cards: Vec<CardId> = symbol_cards(mortar).collect();
        assert_eq!(cards.len(), 2, "every card symbol prints exactly twice");

        let fresh = StateBuilder::new().age(1).build();
        assert!(second_copy_obtainable(&fresh, mortar));

        let slugs: Vec<&str> = cards.iter().map(|c| c.def().id).collect();
        let both_built = StateBuilder::new()
            .age(1)
            .built(Player::Two, &slugs)
            .build();
        assert!(!second_copy_obtainable(&both_built, mortar));
    }
}
