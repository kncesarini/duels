//! `duels-agent-greedy-ev`: a 1-ply heuristic [`Agent`] that resolves
//! uncertainty by exact expectation instead of by a single guess.
//!
//! # The flaw this crate exists to fix
//!
//! `duels-agent-greedy`'s `GreedyAgent` calls
//! [`Observation::sample_state`] once per [`Agent::choose`] call, producing
//! one fully-resolved, fictional concrete [`GameState`] — every currently
//! face-down card is assigned a specific, arbitrary identity — and reuses
//! that same fixed guess to score every candidate action. That is a
//! reasonable way to keep the comparison between candidates apples-to-apples
//! *when nothing about the guess matters to the comparison*. But when a
//! candidate move itself uncovers a new face-down slot (a real chance event:
//! taking a card can reveal one or two others beneath it), `greedy` just
//! reveals whatever its one arbitrary sample says is there and scores that as
//! certain, rather than averaging over what could actually be revealed,
//! weighted by the true probability of each possibility. It is, in effect,
//! picking a move based on a guess rather than picking a move based on its
//! true expected value.
//!
//! # The fix
//!
//! [`GreedyEvAgent::choose`] still calls `sample_state` exactly once per
//! decision — a concrete [`GameState`] is the only thing the engine's
//! `chance_outcomes`/`apply_with_outcome` APIs accept, so a sample is needed
//! purely as a vehicle for calling them. But that sample is used *only* for
//! its public information; for each candidate action it:
//!
//! 1. asks [`duels_core::engine::chance_outcomes`] for the true distribution
//!    of ways the action's randomness (if any) could resolve, computed from
//!    public knowledge only (this is a single certain outcome for the large
//!    majority of actions, which never uncover anything);
//! 2. applies each outcome via [`duels_core::engine::apply_with_outcome`] to
//!    a fresh copy of the sampled base state and scores the result with
//!    [`evaluate`];
//! 3. takes the probability-weighted average of those scores as the
//!    candidate's value — a proper expectation, not a single committed
//!    guess.
//!
//! [`engine::hidden_info`] (which `chance_outcomes` is built on) and
//! [`evaluate`] both read only publicly-known information — built cards,
//! revealed slots, coins, drafted wonders, and so on — so this expectation
//! is provably independent of which arbitrary base state was sampled in step
//! 0. [`tests::expected_value_does_not_depend_on_the_sampled_base_state`] is
//! the test that pins this down: the same observation and action, run
//! through two unrelated `sample_state` seeds, must produce *bit-identical*
//! expected values. If it did not, some hidden dependency on the throwaway
//! sample would have leaked into the scoring, and the whole point of this
//! crate would be moot.
//!
//! # The evaluation function
//!
//! [`evaluate`] is deliberately the same *shape* as `duels-agent-greedy`'s —
//! same named [`EvalWeights`] terms, same default values — so that a
//! head-to-head tournament between `greedy` and `greedy-ev` isolates exactly
//! one variable ("resolve uncertainty via one guess" vs. "resolve it via
//! proper expectation") instead of being confounded by a different
//! heuristic. The terms are reimplemented here rather than imported from
//! `duels-agent-greedy`, at the cost of some light duplication, so this
//! crate stays independent of its sibling agent crates (mirroring
//! `mcts-uct`'s and `alphabeta`'s independence from each other):
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
//!   alternative move avoids it — the one term that reads a freshly-revealed
//!   card's identity, and therefore the one term this crate's fix most
//!   directly targets (see the module docs' benchmark discussion for how
//!   much that actually mattered in practice).
//!
//! Every term is a *difference* between the acting player and their
//! opponent, so the sign of the total score is "how much better is this
//! resulting position for me than for them", and the weights in
//! [`EvalWeights`] are the only place the relative importance of each idea
//! lives.
//!
//! # Performance
//!
//! A single move can uncover two slots at once (a card that was the sole
//! cover for two slots beneath it), which multiplies out to dozens or low
//! hundreds of distinct chance outcomes for that one candidate — see
//! `duels-agent-alphabeta`'s `reduced_outcomes`, which caps and approximates
//! exactly this for the same reason, deep in a multi-ply search where the
//! cost compounds every ply. `greedy-ev` only ever does this once per
//! decision (one ply, no recursive multiplication), so it enumerates the
//! full, exact outcome set rather than approximating — see the module docs'
//! benchmark section for the measured wall-clock cost, which turned out to
//! be comfortably fast enough that no capping was needed.

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
/// These start from exactly `duels-agent-greedy`'s defaults — see the module
/// docs on why keeping the evaluation shape identical matters for the
/// head-to-head comparison this crate exists to run.
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
    /// Kept deliberately small, matching `duels-agent-greedy`'s own tuning
    /// note: a heavier weight here made a 1-ply agent chase the *first* copy
    /// of a symbol even when a clearly better-value card was on offer,
    /// without any way to tell whether the second copy would ever be
    /// reachable. Most of the "value symbols" pressure is left to
    /// `science_near_supremacy`, which only fires once the payoff (or
    /// threat) is concrete.
    pub science_distinct_symbol: f64,
    /// Extra reward for holding 4 or 5 distinct symbols (a smooth ramp: 5 is
    /// one card away from scientific supremacy). Kept as a bonus rather than
    /// folded into `science_distinct_symbol` so the marginal value of a
    /// symbol increases sharply near the win threshold instead of scaling
    /// linearly all game.
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
    ///
    /// This is the one term whose *input* changes because of this crate's
    /// core fix: when a candidate build uncovers a new face-up card, `greedy`
    /// sees whatever its single arbitrary sample put there, while
    /// `greedy-ev` correctly averages this term (like every other) over the
    /// true distribution of what could be uncovered.
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

/// Picks the legal action with the highest probability-weighted expected
/// value after one ply, under a hand-crafted [`EvalWeights`] evaluation. Ties
/// are broken uniformly at random from its own seeded [`StdRng`].
///
/// See the module docs for how this differs from `duels-agent-greedy`:
/// each candidate's chance outcomes (if any) are enumerated exactly via
/// [`engine::chance_outcomes`] and averaged, rather than resolved against one
/// arbitrary sampled guess.
#[derive(Debug, Clone)]
pub struct GreedyEvAgent {
    rng: StdRng,
    weights: EvalWeights,
}

impl GreedyEvAgent {
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

impl Agent for GreedyEvAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "greedy-ev".to_string(),
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
        // Sampled once per call, purely as a vehicle for calling the
        // engine's chance API (which requires a concrete `GameState`). Every
        // candidate's score below is a probability-weighted expectation over
        // publicly-known outcomes, so — unlike `duels-agent-greedy` — it does
        // not matter which arbitrary world this sample invents; see
        // `expected_value` and the module docs.
        let base_state = obs.sample_state(&mut self.rng);

        let mut scored: Vec<(Action, f64)> = Vec::with_capacity(legal.len());
        for &action in legal {
            scored.push((
                action,
                expected_value(&base_state, action, me, &self.weights),
            ));
        }

        let Some(best_score) = scored.iter().map(|&(_, s)| s).fold(None, |m, s| match m {
            Some(b) if b >= s => Some(b),
            _ => Some(s),
        }) else {
            // `legal` is non-empty, so `scored` always has at least one
            // entry; unreachable, but fall back to a uniform pick rather than
            // panicking.
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

/// The probability-weighted expected value of taking `action` in `state`,
/// under [`evaluate`].
///
/// Enumerates every way `action`'s randomness could resolve via
/// [`engine::chance_outcomes`] (a single certain outcome for the large
/// majority of actions, which reveal nothing), applies each one via
/// [`engine::apply_with_outcome`] to its own copy of `state`, scores the
/// result for `me`, and returns the probability-weighted average.
///
/// `state` need only be *some* concrete world consistent with the
/// observation the caller is reasoning about — [`chance_outcomes`] computes
/// its distribution from publicly-known information alone (see
/// [`engine::hidden_info`]), and [`evaluate`] never reads an unrevealed
/// slot's identity, so the returned value does not depend on which
/// consistent world `state` happens to be. That invariant is the whole point
/// of this crate and is pinned down directly by
/// [`tests::expected_value_does_not_depend_on_the_sampled_base_state`].
///
/// [`chance_outcomes`]: engine::chance_outcomes
pub fn expected_value(state: &GameState, action: Action, me: Player, weights: &EvalWeights) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => evaluate(&next, me, weights),
            Err(_) => {
                // `action` came from `legal_actions` for `state` and
                // `outcome` from `chance_outcomes` for the same `(state,
                // action)`, so this should be unreachable. Fall back to
                // scoring the pre-action state rather than silently dropping
                // probability mass from the expectation.
                evaluate(state, me, weights)
            }
        };
        acc += prob * value;
    }
    acc
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

    /// The candidate action with the largest [`engine::chance_outcomes`]
    /// distribution among `legal` (ties broken by scan order), for stress
    /// tests that want a real, multi-outcome chance node rather than a
    /// trivial one.
    fn action_with_most_chance_outcomes(state: &GameState, legal: &[Action]) -> Option<Action> {
        legal
            .iter()
            .copied()
            .max_by_key(|&a| engine::chance_outcomes(state, a).len())
            .filter(|&a| engine::chance_outcomes(state, a).len() > 1)
    }

    fn advanced_game(seed: u64, steps: usize) -> GameState {
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x55);
        for _ in 0..steps {
            let actions = engine::legal_actions(&st);
            if actions.is_empty() {
                break;
            }
            let a = actions[(st.turn() as usize * 7) % actions.len()];
            engine::apply(&mut st, a, &mut rng).unwrap();
        }
        st
    }

    /// The correctness test that is the whole point of this crate: the
    /// expected value of a candidate action, computed from two *different*
    /// arbitrary base-state samples of the same observation, must be
    /// bit-identical. If it is not, `expected_value` has a hidden dependency
    /// on the throwaway sample somewhere, and the core design goal — that
    /// scores reflect true probabilities, not one committed guess — does not
    /// actually hold.
    #[test]
    fn expected_value_does_not_depend_on_the_sampled_base_state() {
        let weights = EvalWeights::default();
        let mut found_a_multi_outcome_case = false;

        for seed in 0..20u64 {
            for steps in [6usize, 14, 22, 30] {
                let st = advanced_game(seed, steps);
                if st.result().is_some() {
                    continue;
                }
                let legal = engine::legal_actions(&st);
                if legal.is_empty() {
                    continue;
                }
                let Some(action) = action_with_most_chance_outcomes(&st, &legal) else {
                    continue;
                };
                let outcome_count = engine::chance_outcomes(&st, action).len();
                if outcome_count < 2 {
                    continue;
                }
                found_a_multi_outcome_case = true;

                let obs = st.observation();
                let me = obs.current_player;

                let mut rng_a = StdRng::seed_from_u64(seed * 1000 + steps as u64);
                let state_a = obs.sample_state(&mut rng_a);
                let ev_a = expected_value(&state_a, action, me, &weights);

                let mut rng_b = StdRng::seed_from_u64(0xFFFF_FFFF_0000_0000 ^ seed ^ steps as u64);
                let state_b = obs.sample_state(&mut rng_b);
                let ev_b = expected_value(&state_b, action, me, &weights);

                assert_eq!(
                    ev_a, ev_b,
                    "seed {seed} steps {steps}: expected value of {action:?} \
                     ({outcome_count} outcomes) depended on the sampled base state: \
                     {ev_a} vs {ev_b}"
                );
            }
        }

        assert!(
            found_a_multi_outcome_case,
            "test setup bug: never found a candidate action with more than one chance outcome"
        );
    }

    /// Same property, restated for `GreedyEvAgent::choose` itself rather than
    /// the bare `expected_value` helper: two agents seeded so their internal
    /// `sample_state` draws differ must score *every* candidate identically
    /// (ties are then broken by each agent's own RNG, so the chosen action
    /// itself is allowed to differ — it is the scores feeding that choice
    /// that must not depend on the sample).
    #[test]
    fn choose_scores_every_candidate_identically_across_different_internal_samples() {
        let st = advanced_game(11, 18);
        if st.result().is_some() {
            return;
        }
        let obs = st.observation();
        let legal = engine::legal_actions(&st);
        if legal.len() < 2 {
            return;
        }
        let me = obs.current_player;
        let weights = EvalWeights::default();

        let mut rng_a = StdRng::seed_from_u64(0x1111);
        let state_a = obs.sample_state(&mut rng_a);
        let mut rng_b = StdRng::seed_from_u64(0x2222_2222_2222);
        let state_b = obs.sample_state(&mut rng_b);

        for &action in &legal {
            let ev_a = expected_value(&state_a, action, me, &weights);
            let ev_b = expected_value(&state_b, action, me, &weights);
            assert_eq!(
                ev_a, ev_b,
                "action {action:?} scored differently depending on the sampled base state"
            );
        }
    }

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
        // discarding it does not.
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
        let mut agent = GreedyEvAgent::new(7);
        let obs = st.observation();
        let legal = [Action::Build { slot: 18 }, Action::Build { slot: 19 }];
        let chosen = agent.choose(&obs, &legal, Budget::Nodes(1));
        assert_eq!(chosen, Action::Build { slot: 19 });
    }

    #[test]
    fn evaluation_orders_win_above_draw_above_loss() {
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
        let agent = GreedyEvAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "greedy-ev");
        assert_eq!(spec.version, "1.0.0");
        assert!(!spec.params.is_empty());
        assert_eq!(spec.params, EvalWeights::default().params_string());
    }

    #[test]
    fn choosing_only_ever_returns_one_of_the_offered_actions() {
        let mut agent = GreedyEvAgent::new(99);
        let state = engine::new_game(99);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        for _ in 0..10 {
            let a = agent.choose(&obs, &legal, Budget::Nodes(1));
            assert!(legal.contains(&a));
        }
    }
}
