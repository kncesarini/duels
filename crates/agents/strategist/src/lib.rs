//! `duels-agent-strategist`: `duels-agent-greedy-ev`'s exact chance-expectation
//! 1-ply evaluation, plus a move-level prior from `duels-strategy`'s
//! win-condition reads.
//!
//! # Why this crate exists
//!
//! `duels-strategy`'s own module docs put it plainly: `greedy` carries
//! explicit military-race terms in its evaluation function and still loses to
//! `random` by military supremacy in roughly one game in ten, because a
//! one-ply evaluation cannot see a race that closes three moves out — by the
//! time the pawn is close enough for a positional term to notice, the shields
//! that would have denied it are gone. That is a search problem, not a
//! scoring problem, and `duels-strategy` was built as the fix: [`stance()`]
//! reads the whole board once per decision and prices every legal move
//! against the *actual* race dynamics (how many shields are reachable, how
//! many turns a race takes to close, who can answer it), not just this
//! position's static snapshot.
//!
//! This crate is the smallest possible proof that the signal is real: bolt
//! [`duels_strategy::action_prior`] onto an otherwise-unchanged one-ply
//! heuristic and see whether the military-supremacy-loss problem actually
//! goes away. It is deliberately *not* the project's primary investment in
//! that layer (wiring it into `mcts-uct`'s tree search, happening separately,
//! is) — this is a validation rung and an extra opponent for the web UI.
//!
//! # The evaluation function
//!
//! [`evaluate`] is exactly `duels-agent-greedy-ev`'s evaluation, reimplemented
//! here rather than imported (at the cost of some light duplication, the same
//! trade `greedy-ev` itself makes against `greedy`) so this crate stays
//! independent of its sibling agent crates. See that crate's module docs for
//! the terms themselves: terminal result, military position, scientific
//! symbols, victory-point projection, economic health, and the shallow
//! chain-gift check.
//!
//! On top of that, `GreedyEvAgent::choose`'s own pattern — sample one base
//! state per decision purely as a vehicle for the engine's chance API, and
//! reuse it across every candidate — is extended one step further:
//! [`stance()`] is *also* computed exactly once per `choose()` call, on that
//! same sampled base state, and reused across every candidate action
//! alongside it. Both
//! `evaluate` and `stance`/`action_prior` read only publicly-known
//! information (see `duels-strategy`'s crate docs on why that must hold), so
//! reusing one arbitrary sample for both is exactly as sound as `greedy-ev`
//! reusing it for `evaluate` alone.
//!
//! [`action_prior`] is a *policy* weight, not a value estimate: it is always
//! strictly positive, scales multiplicatively (a move that denies a certain
//! opposing win or closes this player's own race is promoted by
//! [`duels_strategy::PriorWeights::dominating`], roughly 50x, so a search
//! always looks there first), and is meant to be normalized into a
//! probability distribution over legal moves rather than summed with a value.
//! Folding it into a scalar per-candidate *score* — the shape this crate's
//! one-ply argmax needs — means converting that multiplicative signal into an
//! additive one: [`strategy_term`] adds `strategy_weight * action_prior(...)`
//! directly to the chance-weighted expected value.
//!
//! An earlier version of this crate tried gating that weight by
//! [`duels_strategy::StanceMode`] (large in `DenyCertain`/`PushImminentFork`,
//! tiny in `VpEfficient`), on the theory that `action_prior`'s own
//! `action_vp_value` component is a cruder restatement of what `evaluate`
//! already computes precisely, and should therefore be suppressed outside an
//! actual race. That tuning attempt made head-to-head play against
//! `duels-agent-greedy-ev` measurably *worse* without improving the
//! military-supremacy-loss rate at all, for a specific reason a diagnostic
//! run turned up: `StanceMode` classifies whether *this player's own* race is
//! worth pushing, not whether the *opponent's* race is dangerous, so a rising
//! (but not yet certain) opposing threat very often falls under
//! `VpEfficient` by elimination — precisely where that scheme suppressed the
//! signal hardest. A single scalar, applied uniformly, turned out to be both
//! simpler and better: see [`EvalWeights::strategy_weight`] and this crate's
//! PR description for the measured numbers behind the chosen value, kept well
//! below the terminal-result term's magnitude so a real, current-ply win or
//! loss is never overridden by a move that only looks good for a race a few
//! turns out.
//!
//! # Measured results (report honestly, not hoped-for)
//!
//! Against `duels-agent-random`, this crate roughly halves the
//! military-supremacy-loss rate relative to `greedy-ev` (measured ~12/200
//! here vs. ~24/200 for `greedy-ev` over the same seeds) — a real, repeatable
//! improvement, but short of the "no worse than a handful of unavoidable
//! losses" bar this crate set out to clear. A diagnostic run isolating each
//! loss found why: in the large majority of them, `stance` never classified
//! the opponent's race as [`duels_strategy::StanceMode::DenyCertain`] (nor
//! even as `under_imminent_threat`) in any of this player's last few
//! decisions before the loss — the race closed on the strength of `random`'s
//! actual dice, faster than the model's *expected* trajectory predicted, with
//! no single move available at any of those decision points that would have
//! priced as a real denial. That is a detection-horizon limit of reacting
//! from one position at a time, not a scoring or tuning bug: it is exactly
//! the gap `duels-strategy`'s own docs say a *search* (evaluating the
//! consequence of a plan across several plies, not just one) is needed to
//! close, which is what the parallel `mcts-uct` integration effort is for.
//!
//! Against `duels-agent-greedy-ev` head-to-head, this crate wins consistently
//! but modestly (see the PR description for the exact measured rate) —
//! positive and repeatable across seeds, but well under an ambitious target.
//! The instructive negative finding along the way was that a *stronger*
//! version of the prior (a larger scalar, or one gated to spike higher in an
//! actual race) reliably made head-to-head play *worse*, not better — see
//! [`strategy_term`]'s docs for why.
//!
//! # Performance
//!
//! `stance` costs one pass over both players' military and science reads,
//! paid once per `choose()` call (not per candidate, and not per chance
//! outcome) — see `duels-strategy`'s own benchmark discussion for its
//! measured cost relative to a rollout. `action_prior` itself is cheap (a
//! handful of bitmask tests and one `delta_m` call) and is paid once per
//! candidate action, on the pre-action base state, never per chance outcome —
//! unlike `evaluate`, it needs no lookahead into what a candidate build might
//! uncover, since `duels-strategy`'s own denial pricing already accounts for
//! that inside `delta_m`.

#![deny(clippy::disallowed_methods)]

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::cost;
use duels_core::data::{self, WonderId, NUM_SCIENCE};
use duels_core::engine;
use duels_core::scoring::{self, Breakdown, GameResult};
use duels_core::state::Phase;
use duels_core::{Action, GameState, Observation, Player};
use duels_strategy::{action_prior, stance, Stance};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Scores within this distance of the best score are treated as tied, and one
/// is chosen uniformly at random. Guards against picking a fixed action out
/// of several that are equivalent up to floating-point noise.
const TIE_EPSILON: f64 = 1e-6;

/// Named, independently-tunable weights for [`evaluate`] and [`strategy_term`].
///
/// The first thirteen fields are exactly `duels-agent-greedy-ev`'s
/// `EvalWeights`, at the same defaults, so a head-to-head tournament between
/// `greedy-ev` and `strategist` isolates one variable — "add a move-level
/// prior from the race reads" — rather than being confounded by a different
/// base heuristic. [`Self::strategy_weight`] is the one new knob.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalWeights {
    /// Linear reward for the conflict pawn's position in the acting player's
    /// favour, per step out of `-9..=9`.
    pub military_position: f64,
    /// Extra, quadratically-growing reward for pushing the pawn past the
    /// second loot token (distance 6) towards the capital (distance 9, an
    /// instant win).
    pub military_endgame_urgency: f64,
    /// Reward per distinct scientific symbol held beyond the opponent's
    /// count.
    pub science_distinct_symbol: f64,
    /// Extra reward for holding 4 or 5 distinct symbols.
    pub science_near_supremacy: f64,
    /// Reward per scientific symbol currently held exactly once (excluding
    /// Balance).
    pub science_pair_setup: f64,
    /// Weight on the difference in "as if the game ended now" victory points
    /// from civilian, commercial, guild, wonder and progress-token cards.
    pub vp_projection: f64,
    /// Weight on the difference in `floor(coins / 3)`.
    pub coins_div3: f64,
    /// The coin cushion below which a position is treated as financially
    /// risky.
    pub coin_safety_floor: f64,
    /// Weight on (the opponent's shortfall below `coin_safety_floor`) minus
    /// (mine).
    pub coin_safety_penalty: f64,
    /// Weight on the difference between the opponent's and my own average
    /// per-unit trade price.
    pub resource_vulnerability: f64,
    /// Penalty per victory-point-ish unit of value in a card that would
    /// become accessible to the opponent on their very next turn via one of
    /// their existing chain symbols.
    pub deny_chain_gift: f64,
    /// Weight on the difference in a rough "how strong is this wonder" score
    /// summed over each player's drafted-but-not-yet-built wonders.
    pub wonder_potential: f64,
    /// Magnitude of the score assigned when a candidate move actually ends
    /// the game.
    pub instant_result: f64,
    /// Weight on [`duels_strategy::action_prior`], added directly to the
    /// chance-weighted expected value of `evaluate` (see [`strategy_term`]).
    ///
    /// `action_prior` is a multiplicative policy weight (base around `1.0`,
    /// floored at `0.05`, promoted by roughly 50x for a move that denies a
    /// certain opposing win or closes this player's own race), not a
    /// VP-scaled value — so this weight is deliberately kept well under `1.0`
    /// rather than matching `vp_projection`. See [`strategy_term`]'s docs for
    /// why this is one scalar applied uniformly rather than gated by
    /// position, and this crate's PR description for the sweep behind the
    /// chosen default: small enough that `action_prior`'s own
    /// `action_vp_value` component (a cruder, single-player restatement of
    /// what `evaluate`'s VP terms already compute precisely) does not
    /// routinely override `evaluate`'s fine-grained ranking in an ordinary
    /// turn, while still large enough that the `dominating` spike reliably
    /// wins out over an ordinary VP difference when a race actually is
    /// certain or one move from closing.
    pub strategy_weight: f64,
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
            strategy_weight: 0.05,
        }
    }
}

impl EvalWeights {
    /// A short, reproducible encoding of every weight, suitable for
    /// [`AgentSpec::params`].
    pub fn params_string(&self) -> String {
        format!(
            "mil={:.2}/{:.2},sci={:.2}/{:.2}/{:.2},vp={:.2},coin={:.2}floor{:.1}/{:.2},res={:.2},chain={:.2},wonder={:.2},win={:.0},strat={:.2}",
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
            self.strategy_weight,
        )
    }
}

/// Picks the legal action with the highest probability-weighted expected
/// value after one ply, under [`EvalWeights`], where the expected value is
/// `greedy-ev`'s exact chance-expectation evaluation plus a move-level prior
/// from [`duels_strategy::action_prior`] (see [`strategy_term`] and the
/// module docs). Ties are broken uniformly at random from its own seeded
/// [`StdRng`].
#[derive(Debug, Clone)]
pub struct StrategistAgent {
    rng: StdRng,
    weights: EvalWeights,
}

impl StrategistAgent {
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

impl Agent for StrategistAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "strategist".to_string(),
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
        // Sampled once per call, purely as a vehicle for calling the engine's
        // chance API (which requires a concrete `GameState`) — exactly
        // `greedy-ev`'s pattern. `evaluate` and `stance`/`action_prior` both
        // read only publicly-known information, so it does not matter which
        // arbitrary world this sample invents; see `expected_value` and the
        // module docs.
        let base_state = obs.sample_state(&mut self.rng);
        // Computed once per decision, on the same base state, and reused
        // across every candidate action below — mirroring how `evaluate`
        // itself is reused across candidates via `expected_value`.
        let s = stance(&base_state, me);

        let mut scored: Vec<(Action, f64)> = Vec::with_capacity(legal.len());
        for &action in legal {
            let ev = expected_value(&base_state, action, me, &self.weights);
            let prior = strategy_term(&base_state, action, &s, &self.weights);
            scored.push((action, ev + prior));
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

/// `weights.strategy_weight * duels_strategy::action_prior(state, action, s)`
/// — the additive move-level term this crate adds on top of `greedy-ev`'s
/// chance-expectation evaluation.
///
/// A single scalar applied uniformly, rather than one gated by
/// [`duels_strategy::StanceMode`] or the race magnitudes inside `s` — see the
/// module docs for why an earlier attempt at gating by `StanceMode`
/// specifically made things worse (`StanceMode` is about whether *this
/// player's own* race is worth pushing, not whether the opponent's is
/// dangerous, so a real rising threat very often falls under
/// `StanceMode::VpEfficient` by elimination — exactly where that scheme
/// suppressed the signal hardest).
///
/// Takes the *pre-action* `state` (not a state `action` has been applied to):
/// `action_prior` already prices what a move does to the opponent's races via
/// `duels_strategy::delta_m`, so unlike `evaluate` it needs no lookahead of
/// its own into what the move might uncover, and is therefore computed once
/// per candidate rather than averaged over chance outcomes.
pub fn strategy_term(state: &GameState, action: Action, s: &Stance, weights: &EvalWeights) -> f64 {
    weights.strategy_weight * action_prior(state, action, s)
}

/// The probability-weighted expected value of taking `action` in `state`,
/// under [`evaluate`] — identical in shape to `duels-agent-greedy-ev`'s own
/// `expected_value`.
///
/// Enumerates every way `action`'s randomness could resolve via
/// [`engine::chance_outcomes`] (a single certain outcome for the large
/// majority of actions, which reveal nothing), applies each one via
/// [`engine::apply_with_outcome`] to its own copy of `state`, scores the
/// result for `me`, and returns the probability-weighted average.
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

/// Score `state` for `me`, higher is better — identical to
/// `duels-agent-greedy-ev`'s `evaluate` (see that crate's module docs and
/// [`EvalWeights`] for the terms). The strategy-prior term is *not* part of
/// this function: it is added separately, per candidate action, by
/// [`strategy_term`], since it needs the pre-action state and a precomputed
/// [`Stance`] rather than a resulting state.
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
/// `military_term`) and coins (handled by `coins_div3` and the economic-health
/// terms).
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

    /// The full per-candidate score `choose` uses: `evaluate`'s chance
    /// expectation plus the strategy prior, exactly as `choose` computes it.
    fn full_score(state: &GameState, action: Action, me: Player, weights: &EvalWeights) -> f64 {
        let s = stance(state, me);
        expected_value(state, action, me, weights) + strategy_term(state, action, &s, weights)
    }

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
                let Some(action) = legal
                    .iter()
                    .copied()
                    .max_by_key(|&a| engine::chance_outcomes(&st, a).len())
                    .filter(|&a| engine::chance_outcomes(&st, a).len() > 1)
                else {
                    continue;
                };
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
                    "seed {seed} steps {steps}: expected value of {action:?} depended on the \
                     sampled base state: {ev_a} vs {ev_b}"
                );

                // The strategy term reads only public information too, so it
                // must be equally independent of the sampled base state.
                let s_a = stance(&state_a, me);
                let s_b = stance(&state_b, me);
                let prior_a = strategy_term(&state_a, action, &s_a, &weights);
                let prior_b = strategy_term(&state_b, action, &s_b, &weights);
                assert_eq!(
                    prior_a, prior_b,
                    "seed {seed} steps {steps}: strategy term for {action:?} depended on the \
                     sampled base state: {prior_a} vs {prior_b}"
                );
            }
        }

        assert!(
            found_a_multi_outcome_case,
            "test setup bug: never found a candidate action with more than one chance outcome"
        );
    }

    #[test]
    fn evaluation_prefers_the_move_that_wins_by_military_supremacy() {
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

        let build_score = full_score(&st, Action::Build { slot: 18 }, me, &weights);
        let discard_score = full_score(&st, Action::Discard { slot: 18 }, me, &weights);
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

        let win_score = full_score(&st, Action::Build { slot: 18 }, me, &weights);
        let other_score = full_score(&st, Action::Build { slot: 19 }, me, &weights);
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

    /// The scenario this whole crate exists to fix, one ply *earlier* than
    /// the instant-win case above: the opponent is one build away from an
    /// undeniable military-supremacy close (needs one more shield, has two
    /// independent ways to get it), and this player has exactly one move that
    /// denies both routes at once versus a move that ignores the threat
    /// entirely for a slightly better card. A static VP-projection evaluation
    /// alone cannot tell these two moves apart yet (the denial move's payoff
    /// is a shield the opponent doesn't get *next* turn, not this one) — the
    /// `action_prior` term must be what breaks the tie towards denial.
    #[test]
    fn strategy_term_promotes_denying_an_imminent_military_close_over_a_richer_card() {
        // Player Two's conflict pawn is at -7 (7 shields from Player One's
        // capital at -9): two shields anywhere closes it. "circus" (age 3, 2
        // shields) in slot 18 is exactly the deny: taking it for Player One
        // removes Player Two's two-shield route and also pushes the pawn back
        // towards Player Two, since the shields are counted for whoever
        // builds. "palace" in slot 19 is a strictly richer card in raw
        // `action_vp_value` terms (more points, no shields) but does nothing
        // about the threat.
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(-7)
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build();
        let me = st.current_player();
        assert_eq!(me, Player::One);
        let s = stance(&st, me);

        // Sanity: the position is read as an imminent opposing threat this
        // player can do something about.
        assert!(
            s.under_imminent_threat(),
            "scenario setup bug: Player Two should be one shield from closing"
        );

        let weights = EvalWeights::default();
        let deny_prior = action_prior(&st, Action::Build { slot: 18 }, &s);
        let richer_prior = action_prior(&st, Action::Build { slot: 19 }, &s);
        assert!(
            deny_prior > richer_prior,
            "deny={deny_prior} richer={richer_prior}: action_prior itself should already favour \
             the denying move"
        );

        let deny_score = full_score(&st, Action::Build { slot: 18 }, me, &weights);
        let richer_score = full_score(&st, Action::Build { slot: 19 }, me, &weights);
        assert!(
            deny_score > richer_score,
            "deny={deny_score} richer={richer_score}: the combined score should prefer denying \
             the imminent military close"
        );

        // And the agent itself, offered only those two actions, must agree.
        let mut agent = StrategistAgent::new(7);
        let obs = st.observation();
        let legal = [Action::Build { slot: 18 }, Action::Build { slot: 19 }];
        let chosen = agent.choose(&obs, &legal, Budget::Nodes(1));
        assert_eq!(chosen, Action::Build { slot: 18 });
    }

    #[test]
    fn evaluation_avoids_gifting_the_opponent_a_free_chain_build() {
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

        let exposes_gift = full_score(&st, Action::Build { slot: 18 }, me, &weights);
        let avoids_gift = full_score(&st, Action::Build { slot: 19 }, me, &weights);
        assert!(
            avoids_gift > exposes_gift,
            "avoids={avoids_gift} exposes={exposes_gift}"
        );

        let mut agent = StrategistAgent::new(7);
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
        let agent = StrategistAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "strategist");
        assert_eq!(spec.version, "1.0.0");
        assert!(!spec.params.is_empty());
        assert_eq!(spec.params, EvalWeights::default().params_string());
    }

    #[test]
    fn choosing_only_ever_returns_one_of_the_offered_actions() {
        let mut agent = StrategistAgent::new(99);
        let state = engine::new_game(99);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        for _ in 0..10 {
            let a = agent.choose(&obs, &legal, Budget::Nodes(1));
            assert!(legal.contains(&a));
        }
    }
}
