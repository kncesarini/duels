//! `duels-agent-greedy`: a 1-ply heuristic [`Agent`].
//!
//! [`GreedyAgent`] samples one concrete world consistent with the
//! [`Observation`] it is handed (see [`Observation::sample_state`]), then, for
//! every legal action, applies that single action to a copy of the sampled
//! state and scores the result with [`evaluate`] — a hand-crafted position
//! evaluation, not [`duels_core::scoring::score`] (which only produces a real
//! number at game end). It never looks more than one ply deep and never
//! simulates the opponent's reply; that is `alphabeta`'s and `mcts-uct`'s job.
//!
//! Sampling once per [`Agent::choose`] call and reusing that single sampled
//! state for every candidate keeps the comparison apples-to-apples: every
//! action is judged against the same determinized world, so differences in
//! score reflect the action, not the luck of a fresh sample.
//!
//! # The evaluation function
//!
//! [`evaluate`] adds up several independently-weighted terms, each described
//! on the corresponding [`EvalWeights`] field:
//!
//! * a terminal-result term, dwarfing everything else, so a real win is
//!   always preferred and a real loss always avoided if any alternative
//!   exists;
//! * military track position, plus an escalating bonus for approaching an
//!   instant win (or letting the opponent approach one);
//! * scientific symbols: distinct-symbol count (progress toward the 6-symbol
//!   instant win), an escalating bonus near that threshold, and a reward for
//!   holding a symbol once (halfway to the progress-token pair bonus);
//! * an "as if the game ended now" victory-point projection built from real
//!   per-card/-wonder/-token data (see `duels_core::data`), plus the standard
//!   `floor(coins / 3)` term;
//! * economic health: a penalty for dropping below a safe coin cushion, and a
//!   penalty for facing worse average trade prices than the opponent;
//! * a shallow tactical check that refuses to hand the opponent an
//!   obviously free, valuable chain-build on their very next turn when an
//!   alternative move avoids it.
//!
//! Every term is a *difference* between the acting player and their
//! opponent, so the sign of the total score is "how much better is this
//! resulting position for me than for them", and the weights in
//! [`EvalWeights`] are the only place the relative importance of each idea
//! lives — no magic numbers are scattered through the term functions
//! themselves.

#![deny(clippy::disallowed_methods)]

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::cost;
use duels_core::data::{self, WonderId, NUM_SCIENCE};
use duels_core::engine;
use duels_core::scoring::{self, Breakdown, GameResult};
use duels_core::state::Phase;
use duels_core::{Action, GameState, Observation, Player};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Scores within this distance of the best score are treated as tied, and one
/// is chosen uniformly at random. Guards against picking a fixed action out
/// of several that are equivalent up to floating-point noise.
const TIE_EPSILON: f64 = 1e-6;

/// Named, independently-tunable weights for [`evaluate`].
///
/// Grouping every coefficient here (rather than scattering literals through
/// the evaluation code) is what lets a tournament runner such as
/// `duels-arena` sweep or fit these later without touching the algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalWeights {
    /// Linear reward for the conflict pawn's position in the acting player's
    /// favour, per step out of `-9..=9`. Keeps the agent generally leaning
    /// towards pushing the military track, since board position there is
    /// compounding pressure (loot tokens, proximity to an instant win) even
    /// before it pays off in points.
    pub military_position: f64,
    /// Extra, quadratically-growing reward for pushing the pawn past the
    /// second loot token (distance 6) towards the capital (distance 9, an
    /// instant win). Quadratic growth makes the agent visibly prioritise
    /// finishing a near-certain military win over marginal economic gains,
    /// and treats the opponent approaching that same threshold as similarly
    /// urgent to prevent (the term is symmetric).
    pub military_endgame_urgency: f64,
    /// Reward per distinct scientific symbol held beyond the opponent's
    /// count. Each additional distinct symbol is a step toward the 6-symbol
    /// instant win and, incidentally, usually a card that was worth
    /// building anyway.
    ///
    /// Kept deliberately small: an early sanity benchmark against
    /// `RandomAgent` (see the `duels-agent-greedy` integration tests) showed
    /// that a heavier weight here (2.5, matching `vp_projection`-scale
    /// terms) made a 1-ply agent chase the *first* copy of a symbol even
    /// when a clearly better-value card was on offer, without any way to
    /// tell whether the second copy would ever be reachable. A 1-ply
    /// evaluation cannot judge that trade-off well, so most of the "value
    /// symbols" pressure is left to `science_near_supremacy`, which only
    /// fires once the payoff (or threat) is concrete.
    pub science_distinct_symbol: f64,
    /// Extra reward for holding 4 or 5 distinct symbols (a smooth ramp: 5 is
    /// one card away from scientific supremacy). Kept as a bonus rather than
    /// folded into `science_distinct_symbol` so the marginal value of a
    /// symbol increases sharply near the win threshold instead of scaling
    /// linearly all game — by construction this only engages late enough
    /// that the 1-ply blind spot above does not apply.
    pub science_near_supremacy: f64,
    /// Reward per scientific symbol currently held exactly once (Balance —
    /// the Law token's symbol — is excluded, since no card ever carries it
    /// and it can therefore never form a pair). Holding one of a pair sets
    /// up a progress-token reward the moment its twin is built, so it is
    /// worth valuing before the pair actually completes — kept small for
    /// the same reason as `science_distinct_symbol`.
    pub science_pair_setup: f64,
    /// Weight on the difference in "as if the game ended now" victory
    /// points from civilian, commercial, guild, wonder and progress-token
    /// cards (each pulled from the real per-card data via
    /// [`duels_core::scoring::breakdown`]). This is the closest single
    /// number to "who is winning on points", hence the largest weight.
    pub vp_projection: f64,
    /// Weight on the difference in `floor(coins / 3)`, matching the real
    /// end-of-game scoring rule exactly. Kept separate from `vp_projection`
    /// (rather than folded into the same term) so it can be tuned
    /// independently of the economic-health terms below, which also look at
    /// coins.
    pub coins_div3: f64,
    /// The coin cushion below which a position is treated as financially
    /// risky: an opponent's trade-cost gouging, or simply being unable to
    /// afford the next useful card, becomes a real danger. Not itself a
    /// weight — see `coin_safety_penalty`.
    pub coin_safety_floor: f64,
    /// Weight on (the opponent's shortfall below `coin_safety_floor`) minus
    /// (mine). Being flush relative to the floor is good; being caught
    /// under it while the opponent isn't is bad.
    pub coin_safety_penalty: f64,
    /// Weight on the difference between the opponent's and my own average
    /// per-unit trade price (see [`duels_core::cost::trade_prices`]). A high
    /// average price means a player is exposed to being cost-gouged (the
    /// opponent produces resources they lack and they hold no trading
    /// post), so facing cheaper trade than the opponent is rewarded.
    pub resource_vulnerability: f64,
    /// Penalty per victory-point-ish unit of value in a card that would
    /// become accessible to the opponent on their very next turn via one of
    /// their existing chain symbols (a free build). Only counted when it
    /// will genuinely be the opponent's turn next — an extra turn or a
    /// pending effect choice does not trigger it. Deliberately shallow (no
    /// search of the opponent's actual best reply, unlike `alphabeta` or
    /// `mcts-uct`): it just refuses to hand over an obviously free, valuable
    /// card when an alternative move avoids it.
    pub deny_chain_gift: f64,
    /// Weight on the difference in a rough, hand-tuned "how strong is this
    /// wonder" score (victory points, one-off coins, shields, and a flat
    /// bonus per notable rules effect such as play-again, destroy, or the
    /// Great Library's token draw) summed over each player's drafted-but-
    /// not-yet-built wonders. Without this term the evaluation function
    /// cannot tell two wonder-draft picks apart, since `vp_projection` only
    /// credits *built* wonders.
    pub wonder_potential: f64,
    /// Magnitude of the score assigned when a candidate move actually ends
    /// the game (see [`GameResult`]). Set far larger than the sum of every
    /// other term's plausible range so a genuine win is always preferred
    /// over any positional improvement, and a genuine loss is always
    /// avoided while any legal alternative exists.
    pub instant_result: f64,
}

impl Default for EvalWeights {
    fn default() -> Self {
        Self {
            military_position: 0.6,
            military_endgame_urgency: 3.0,
            science_distinct_symbol: 0.3,
            science_near_supremacy: 2.0,
            science_pair_setup: 0.15,
            vp_projection: 1.0,
            coins_div3: 1.0,
            coin_safety_floor: 3.0,
            coin_safety_penalty: 0.5,
            resource_vulnerability: 0.4,
            deny_chain_gift: 0.5,
            wonder_potential: 0.5,
            instant_result: 1000.0,
        }
    }
}

impl EvalWeights {
    /// A short, reproducible encoding of every weight, suitable for
    /// [`AgentSpec::params`].
    pub fn params_string(&self) -> String {
        format!(
            "mil={:.2}/{:.2},sci={:.2}/{:.2}/{:.2},vp={:.2},coin={:.2}floor{:.1}/{:.2},res={:.2},chain={:.2},wonder={:.2},win={:.0}",
            self.military_position,
            self.military_endgame_urgency,
            self.science_distinct_symbol,
            self.science_near_supremacy,
            self.science_pair_setup,
            self.vp_projection,
            self.coins_div3,
            self.coin_safety_floor,
            self.coin_safety_penalty,
            self.resource_vulnerability,
            self.deny_chain_gift,
            self.wonder_potential,
            self.instant_result,
        )
    }
}

/// Picks the legal action that scores best after one ply of simulation
/// against a single sampled world, under a hand-crafted [`EvalWeights`]
/// evaluation. Ties are broken uniformly at random from its own seeded
/// [`StdRng`].
#[derive(Debug, Clone)]
pub struct GreedyAgent {
    rng: StdRng,
    weights: EvalWeights,
}

impl GreedyAgent {
    /// A new agent seeded from `seed`, using [`EvalWeights::default`].
    pub fn new(seed: u64) -> Self {
        Self::with_weights(seed, EvalWeights::default())
    }

    /// A new agent seeded from `seed`, with custom evaluation weights (for
    /// tournament-based tuning, e.g. by `duels-arena`).
    pub fn with_weights(seed: u64, weights: EvalWeights) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            weights,
        }
    }

    /// A new agent driven by an existing RNG, so a caller can draw many
    /// independent agents from one stream.
    pub fn from_rng(rng: StdRng) -> Self {
        Self {
            rng,
            weights: EvalWeights::default(),
        }
    }

    /// The evaluation weights this agent is using.
    pub fn weights(&self) -> &EvalWeights {
        &self.weights
    }
}

impl Agent for GreedyAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "greedy".to_string(),
            version: "1.0.0".to_string(),
            params: self.weights.params_string(),
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action], _budget: Budget) -> Action {
        assert!(
            !legal.is_empty(),
            "choose must not be called with no legal actions"
        );
        if legal.len() == 1 {
            return legal[0];
        }

        let me = obs.current_player;
        // Determinize once and reuse the same sampled world for every
        // candidate, so the comparison is apples-to-apples.
        let base_state = obs.sample_state(&mut self.rng);
        // Each candidate gets its own derived scratch RNG (for the rare
        // Great Library draw) rather than sharing `self.rng`, so evaluating
        // N candidates never depends on N's order or count.
        let choose_seed: u64 = self.rng.gen();

        let mut scored: Vec<(Action, f64)> = Vec::with_capacity(legal.len());
        for (i, &action) in legal.iter().enumerate() {
            let mut state = base_state;
            let mut scratch =
                StdRng::seed_from_u64(choose_seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            if engine::apply(&mut state, action, &mut scratch).is_err() {
                // `action` came from `legal` for `obs`, and `sample_state`
                // guarantees a sampled world has the same legal actions, so
                // this should be unreachable; skip rather than panic.
                continue;
            }
            scored.push((action, evaluate(&state, me, &self.weights)));
        }

        let Some(best_score) = scored.iter().map(|&(_, s)| s).fold(None, |m, s| match m {
            Some(b) if b >= s => Some(b),
            _ => Some(s),
        }) else {
            // Simulation failed for every candidate; fall back to a uniform
            // pick so the agent still returns a legal move.
            return legal[self.rng.gen_range(0..legal.len())];
        };

        let best: Vec<Action> = scored
            .iter()
            .filter(|&&(_, s)| (best_score - s).abs() <= TIE_EPSILON)
            .map(|&(a, _)| a)
            .collect();
        best[self.rng.gen_range(0..best.len())]
    }
}

/// Score `state` for `me`, higher is better, using a hand-crafted position
/// evaluation (see the module docs and [`EvalWeights`] for the terms).
///
/// A finished game (win/loss/draw) is scored by `weights.instant_result`
/// alone, dwarfing every other term; otherwise every term is a difference
/// between `me` and their opponent.
pub fn evaluate(state: &GameState, me: Player, weights: &EvalWeights) -> f64 {
    if let Some(result) = state.result() {
        return match result {
            GameResult::Win { winner, .. } if winner == me => weights.instant_result,
            GameResult::Win { .. } => -weights.instant_result,
            GameResult::Draw => 0.0,
        };
    }

    let opp = me.other();
    military_term(state, me, weights)
        + science_term(state, me, opp, weights)
        + vp_term(state, me, opp, weights)
        + economy_term(state, me, opp, weights)
        + tactical_term(state, me, weights)
}

/// Military track position plus an escalating push-towards-the-capital
/// bonus. `data::military()` is consulted rather than hard-coding the
/// distances, so this stays correct if the data ever changes.
fn military_term(state: &GameState, me: Player, w: &EvalWeights) -> f64 {
    let signed = match me {
        Player::One => f64::from(state.conflict()),
        Player::Two => -f64::from(state.conflict()),
    };

    let second_loot = f64::from(data::military().loot[1].0);
    let urgency = if signed.abs() > second_loot {
        let excess = signed.abs() - second_loot;
        signed.signum() * excess * excess
    } else {
        0.0
    };

    signed * w.military_position + urgency * w.military_endgame_urgency
}

/// A smooth ramp rewarding a player who is close to the 6-distinct-symbol
/// instant win.
fn near_supremacy(distinct: u8) -> f64 {
    match distinct {
        5 => 1.0,
        4 => 0.4,
        _ => 0.0,
    }
}

/// How many scientific symbols this science array holds exactly once,
/// excluding Balance (index `NUM_SCIENCE - 1`, the Law token's symbol, which
/// no card ever carries and so can never form a pair).
fn pair_setup_count(science: [u8; NUM_SCIENCE]) -> f64 {
    science[..NUM_SCIENCE - 1]
        .iter()
        .filter(|&&n| n == 1)
        .count() as f64
}

fn science_term(state: &GameState, me: Player, opp: Player, w: &EvalWeights) -> f64 {
    let mine = state.player(me);
    let theirs = state.player(opp);

    let distinct_diff = f64::from(mine.distinct_science()) - f64::from(theirs.distinct_science());
    let near_diff =
        near_supremacy(mine.distinct_science()) - near_supremacy(theirs.distinct_science());
    let pair_diff = pair_setup_count(mine.science()) - pair_setup_count(theirs.science());

    distinct_diff * w.science_distinct_symbol
        + near_diff * w.science_near_supremacy
        + pair_diff * w.science_pair_setup
}

/// A rough, hand-tuned "how strong is this wonder" score: printed victory
/// points and one-off resources, plus a flat bonus per notable rules effect.
/// Not derived from any principled source (unlike the VP projection below,
/// which uses real scoring data) — a reasonable target for tournament-based
/// re-tuning.
fn wonder_power(w: WonderId) -> f64 {
    let def = w.def();
    let mut v = f64::from(def.victory_points)
        + f64::from(def.coins) * 0.3
        + f64::from(def.shields)
        + f64::from(def.opponent_loses_coins) * 0.3;
    if def.play_again {
        v += 3.0;
    }
    if def.destroy.is_some() {
        v += 3.0;
    }
    if def.build_discarded_free {
        v += 3.0;
    }
    if def.choose_progress_token {
        v += 3.0;
    }
    if def.produces_choice.is_some() {
        v += 2.0;
    }
    v
}

/// Sum of `wonder_power` over `p`'s drafted-but-not-yet-built wonders.
fn wonder_potential(state: &GameState, p: Player) -> f64 {
    let ps = state.player(p);
    ps.wonders()
        .filter(|&w| !ps.has_built_wonder(w))
        .map(wonder_power)
        .sum()
}

/// Sum of one category of [`Breakdown`], excluding military (handled by
/// `military_term`, which cares about board position, not just the resulting
/// points) and coins (handled separately by `coins_div3`, alongside the
/// economic-health terms which also look at coins).
fn card_and_token_vp(b: &Breakdown) -> f64 {
    f64::from(b.civilian + b.scientific + b.commercial + b.guilds + b.wonders + b.progress_tokens)
}

fn vp_term(state: &GameState, me: Player, opp: Player, w: &EvalWeights) -> f64 {
    let mine = scoring::breakdown(state, me);
    let theirs = scoring::breakdown(state, opp);

    let projection_diff = card_and_token_vp(&mine) - card_and_token_vp(&theirs);
    let coins_diff = f64::from(mine.coins) - f64::from(theirs.coins);
    let wonder_diff = wonder_potential(state, me) - wonder_potential(state, opp);

    projection_diff * w.vp_projection + coins_diff * w.coins_div3 + wonder_diff * w.wonder_potential
}

fn coin_shortfall(state: &GameState, p: Player, floor: f64) -> f64 {
    (floor - f64::from(state.player(p).coins())).max(0.0)
}

fn average_trade_price(state: &GameState, p: Player) -> f64 {
    let prices = cost::trade_prices(state, p);
    prices.iter().map(|&c| f64::from(c)).sum::<f64>() / prices.len() as f64
}

fn economy_term(state: &GameState, me: Player, opp: Player, w: &EvalWeights) -> f64 {
    let safety_diff = coin_shortfall(state, opp, w.coin_safety_floor)
        - coin_shortfall(state, me, w.coin_safety_floor);
    let vulnerability_diff = average_trade_price(state, opp) - average_trade_price(state, me);

    safety_diff * w.coin_safety_penalty + vulnerability_diff * w.resource_vulnerability
}

/// Total "value" of every accessible, face-up card the opponent could build
/// for free on their very next turn via a chain symbol they already own.
/// Zero unless it will genuinely be the opponent's turn next.
fn opponent_chain_gift_value(state: &GameState, me: Player) -> f64 {
    if state.phase() != Phase::Turn || state.current_player() != me.other() {
        return 0.0;
    }
    let opp = me.other();
    let mut value = 0.0;
    let mut mask = state.accessible_slots();
    while mask != 0 {
        let slot = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        if let Some(card) = state.face_up_card(slot) {
            let def = card.def();
            if let Some(prereq) = def.chain_from {
                if state.player(opp).has_built(prereq) {
                    value += 2.0 + f64::from(def.victory_points);
                }
            }
        }
    }
    value
}

fn tactical_term(state: &GameState, me: Player, w: &EvalWeights) -> f64 {
    -opponent_chain_gift_value(state, me) * w.deny_chain_gift
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::scoring::VictoryKind;
    use duels_core::testing::StateBuilder;

    /// Apply `action` to a copy of `state` and evaluate the result for
    /// `me`, the player who was to move in `state`.
    fn eval_after(state: &GameState, action: Action, me: Player, weights: &EvalWeights) -> f64 {
        let mut s = *state;
        let mut rng = StdRng::seed_from_u64(0x0C0F_FEE0);
        engine::apply(&mut s, action, &mut rng).expect("scenario action should be legal");
        evaluate(&s, me, weights)
    }

    #[test]
    fn evaluation_prefers_the_move_that_wins_by_military_supremacy() {
        // Player One is 2 shields from the capital (distance 9); "circus"
        // (age 3, 2 shields) accessible in slot 18 wins outright, while
        // discarding it does not. A second, harmless card (slot 19) keeps
        // the age structure non-empty either way, so the only route to a
        // finished game here is the military track, not an incidental
        // civilian-victory-by-emptying-the-table artifact of the test setup.
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "clay-pool")])
            .conflict(7)
            .coins(Player::One, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let weights = EvalWeights::default();

        let legal = engine::legal_actions(&st);
        assert!(legal.contains(&Action::Build { slot: 18 }));
        assert!(legal.contains(&Action::Discard { slot: 18 }));

        let build_score = eval_after(&st, Action::Build { slot: 18 }, me, &weights);
        let discard_score = eval_after(&st, Action::Discard { slot: 18 }, me, &weights);
        assert!(
            build_score > discard_score,
            "build={build_score} discard={discard_score}"
        );

        let mut after = st;
        let mut rng = StdRng::seed_from_u64(1);
        engine::apply(&mut after, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(
            after.result(),
            Some(GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::MilitarySupremacy,
            }),
            "the scenario should actually reach an instant military win"
        );
    }

    #[test]
    fn evaluation_prefers_the_move_that_wins_by_scientific_supremacy() {
        // Player One already holds 5 distinct scientific symbols;
        // "university" (gyroscope) in slot 18 completes the sixth and wins
        // outright, while building "palace" (a fine but unrelated card) in
        // slot 19 does not.
        let st = StateBuilder::new()
            .age(3)
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
            .open_slots(&[(18, "university"), (19, "palace")])
            .coins(Player::One, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        assert_eq!(st.player(me).distinct_science(), 5);
        let weights = EvalWeights::default();

        let win_score = eval_after(&st, Action::Build { slot: 18 }, me, &weights);
        let other_score = eval_after(&st, Action::Build { slot: 19 }, me, &weights);
        assert!(
            win_score > other_score,
            "win={win_score} other={other_score}"
        );

        let mut after = st;
        let mut rng = StdRng::seed_from_u64(1);
        engine::apply(&mut after, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(
            after.result(),
            Some(GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::ScientificSupremacy,
            }),
            "the scenario should actually reach an instant scientific win"
        );
    }

    #[test]
    fn evaluation_avoids_gifting_the_opponent_a_free_chain_build() {
        // Player Two already owns "tavern", whose chain unlocks "lighthouse"
        // for free. Slot 18 ("clay-pool") covers slot 15 ("lighthouse"):
        // taking slot 18 exposes lighthouse to Player Two next turn. Slot 19
        // ("quarry") instead uncovers a harmless "lumber-yard" in slot 17.
        // The two accessible cards are both free, single-resource producers,
        // so nothing but the chain-gift term should separate them.
        let st = StateBuilder::new()
            .age(3)
            .built(Player::Two, &["tavern"])
            .open_slots(&[
                (15, "lighthouse"),
                (17, "lumber-yard"),
                (18, "clay-pool"),
                (19, "quarry"),
            ])
            .current(Player::One)
            .build();
        let me = st.current_player();
        assert_eq!(st.accessible_slots(), (1 << 18) | (1 << 19));
        let weights = EvalWeights::default();

        let exposes_gift = eval_after(&st, Action::Build { slot: 18 }, me, &weights);
        let avoids_gift = eval_after(&st, Action::Build { slot: 19 }, me, &weights);
        assert!(
            avoids_gift > exposes_gift,
            "avoids={avoids_gift} exposes={exposes_gift}"
        );

        // And the agent itself, offered only those two actions, must agree.
        let mut agent = GreedyAgent::new(7);
        let obs = st.observation();
        let legal = [Action::Build { slot: 18 }, Action::Build { slot: 19 }];
        let chosen = agent.choose(&obs, &legal, Budget::Nodes(1));
        assert_eq!(chosen, Action::Build { slot: 19 });
    }

    #[test]
    fn evaluation_orders_win_above_draw_above_loss() {
        // Reach a genuine terminal `GameState` by actually emptying the Age
        // III structure (one throwaway card, discarded), rather than trying
        // to poke `result` directly — `StateBuilder` deliberately exposes no
        // way to do that, since reaching game-over is a rules outcome, not a
        // testing primitive.
        let weights = EvalWeights::default();
        let finish = |one: &[&str], two: &[&str]| -> GameState {
            let mut st = StateBuilder::new()
                .built(Player::One, one)
                .built(Player::Two, two)
                .open_slots(&[(18, "clay-pool")])
                .current(Player::One)
                .build();
            let mut rng = StdRng::seed_from_u64(3);
            engine::apply(&mut st, Action::Discard { slot: 18 }, &mut rng).unwrap();
            assert!(
                st.result().is_some(),
                "discarding the only card should empty the age and end the game"
            );
            st
        };

        // Same civilian scenario as `duels_core::scoring`'s own
        // `civilian_victory_is_decided_on_totals_then_blue_then_draw` test:
        // "palace" (7 blue VP) beats an empty city, and "palace" vs.
        // "town-hall" (also 7 blue VP) is a true draw.
        let win = finish(&["palace"], &[]);
        let draw = finish(&["palace"], &["town-hall"]);
        let loss = finish(&[], &["palace"]);

        let win_score = evaluate(&win, Player::One, &weights);
        let draw_score = evaluate(&draw, Player::One, &weights);
        let loss_score = evaluate(&loss, Player::One, &weights);

        assert_eq!(win_score, weights.instant_result);
        assert_eq!(draw_score, 0.0);
        assert_eq!(loss_score, -weights.instant_result);
        assert!(win_score > draw_score && draw_score > loss_score);
    }

    #[test]
    fn spec_reports_the_expected_name_and_weight_encoded_params() {
        let agent = GreedyAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "greedy");
        assert_eq!(spec.version, "1.0.0");
        assert!(!spec.params.is_empty());
        assert_eq!(spec.params, EvalWeights::default().params_string());
    }

    #[test]
    fn choosing_only_ever_returns_one_of_the_offered_actions() {
        let mut agent = GreedyAgent::new(99);
        let state = engine::new_game(99);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        for _ in 0..10 {
            let a = agent.choose(&obs, &legal, Budget::Nodes(1));
            assert!(legal.contains(&a));
        }
    }
}
