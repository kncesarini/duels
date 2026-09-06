//! `duels-agent-phased`: a 1-ply [`Agent`] whose evaluation weights are a
//! continuous function of how committed each player is to a win condition.
//!
//! # What this crate is trying to fix
//!
//! `duels-agent-greedy-ev` resolves uncertainty correctly — it averages over
//! [`engine::chance_outcomes`] instead of committing to one sampled guess —
//! and that machinery is copied here unchanged. What it does *not* do is
//! change its mind about what matters. Its weight vector is the same on turn
//! one and on turn fifty, for a player two symbols from scientific supremacy
//! and for one who has never taken a green card. Two specific consequences a
//! strong human player pointed out:
//!
//! * it prices almost none of a player's **own production**. A brown or grey
//!   card is, to `greedy-ev`, a card with no points and no shields — a bill.
//!   In reality it is every future card's discount, and grey is the scarcest
//!   of the lot (Ages I and II carry two grey cards each and Age III carries
//!   none at all — verified against `data/cards.json`, not assumed).
//! * it has no idea that a player at four distinct symbols is in a different
//!   game from a player at two. Ages I and II each carry one copy of the same
//!   four symbols and Age III carries the other two, twice each, so after Age
//!   I *every* Age II green completes a pair, and the sixth symbol needs an
//!   Age-III-only symbol or the Law token. A linear "points per symbol" term
//!   cannot see any of that.
//!
//! # The design: one continuous blend, no modes
//!
//! Each player carries a commitment scalar `c ∈ [0, 1]` built from
//! `duels-strategy`'s calibrated race magnitudes, and a Hill curve `S(c)`
//! turns it into a per-term multiplier. Points, coin liquidity, development
//! and economy all fade as commitment rises; the science ladder and military
//! position sharpen; race-card liquidity rises. There are deliberately **no
//! discrete modes** — see [`blend`] for why, and for the arithmetic.
//!
//! Every weight is computed **once per decision from the root position** and
//! reused for every candidate action and every chance outcome. Term
//! *contents* are measured on the post-action, expectation-averaged state,
//! exactly as `greedy-ev` measures its own; only the weights are pinned. That
//! asymmetry is the point: a move that raises `c_sci` should be credited once,
//! through the science term's value going up, not twice, through the weight on
//! that term going up as well.
//!
//! # The evaluation
//!
//! ```text
//! score(state, me) = terminal_result
//!                  | Σ_k [ T_k(me) − T_k(opp) ] + A(action)
//! T_k(p)           = w_k(S(c(p))) × base_k × raw_k(state, p)
//! A(action)        = w_deny × deny_scale × duels_strategy::deny_vp(action)
//! ```
//!
//! Terms are read **per player** and then differenced, rather than as a
//! single difference under one weight, so that each side's science ladder
//! carries *that side's* science commitment. A science race is more important
//! to whichever player is actually in it; weighting the opponent's ladder by
//! the mover's commitment would read a rising opposing race as less
//! interesting the less interested the mover is in science, which is exactly
//! backwards. At equal commitment the two formulations coincide.
//!
//! `A` is a function of the root position and the action alone —
//! [`duels_strategy::delta_m`] prices what a move does to the opponent's race
//! magnitudes from the root stance, and never looks at the post-action state —
//! so it is genuinely outcome-independent and is added once, outside the
//! per-outcome loop. (Adding it inside would give a bit-identical answer,
//! since the outcome probabilities sum to one; once is simply cheaper.)
//!
//! # What "the un-blended baseline" means here
//!
//! This agent's term set is not `greedy-ev`'s, so there is no other crate it
//! can be bit-compared against. What *is* pinned exactly is the property the
//! whole blend rests on: `S(0) = 0`, so a position in which neither player is
//! committed to anything is scored by a plain, fixed weight vector — every
//! multiplier exactly `1.0`, except race-card liquidity, the one term that
//! rises with commitment and therefore sits at its own floor. [`Blend::off`]
//! forces that state for every position, and
//! `tests::the_blend_off_and_a_zero_commitment_position_agree_bit_for_bit`
//! asserts the two agree bit for bit.
//!
//! # Measured
//!
//! Preliminary, at default `Config` — a sanity check, not the full acceptance
//! campaign (no parameter sweep, no held-out test set). All paired and
//! seat-swapped through `duels-arena` at `Nodes(1)`, which is the whole budget
//! a 1-ply agent uses:
//!
//! ```text
//! phased vs greedy-ev   400 games  seed 1     392-8    Elo +666  [+548, +784]
//! phased vs greedy-ev   400 games  seed 5001  391-9    Elo +646  [+534, +758]
//! phased vs greedy-ev   400 games  seed 9001  394-6    Elo +713  [+579, +848]
//! phased vs greedy       300 games seed 1     297-3    Elo +772
//! phased vs strategist   300 games seed 1     296-4    Elo +728
//! phased vs random       300 games seed 1     291-9    Elo +595
//!   (for scale: greedy-ev vs random is 221-79, Elo +178)
//! phased vs mcts-uct     200 games  Nodes(2000)  16-184  Elo -419
//! ```
//!
//! Three disjoint seed ranges agree, so the margin over `greedy-ev` is real
//! rather than a seed artefact. `mcts-uct` remains far ahead — this is a 1-ply
//! evaluation against a real search, and the project's standing finding that
//! simulation beats hand-crafted judgement for *position value* in this game
//! is not contradicted by any of the above.
//!
//! Where the wins come from is as informative as the count: against
//! `greedy-ev` roughly 45% of them are scientific supremacy and only ~9%
//! military, and 77% of games reach five distinct symbols for somebody. The
//! science ladder is doing most of the work.
//!
//! `examples/take_profile.rs` shows the mechanism directly. In Age I,
//! `greedy-ev` builds 99% of the blue cards it sees and *discards* 77% of
//! brown, 100% of grey and 65% of green — it has no term for what a card
//! produces, so anything without printed points is a bill worth cashing in.
//! `phased` keeps 69% of brown, 87% of grey and 83% of green, and discards
//! 99% of red. See that example's own docs for the full table and for what
//! `mcts-uct` does, which is different again.
//!
//! # Public information only
//!
//! Like `greedy-ev`, [`PhasedAgent::choose`] samples one concrete
//! [`GameState`] per decision purely as a vehicle for the engine's chance API.
//! Everything downstream — the commitment scalars, every weight, and the full
//! evaluation of every legal action — reads only publicly-known information,
//! so none of it depends on which world that throwaway sample invented.
//! `tests/determinization_invariance.rs` asserts that bit for bit over real
//! positions from real games.

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

pub mod blend;
pub mod terms;

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::engine;
use duels_core::scoring::{self, GameResult};
use duels_core::{Action, GameState, Observation, Player};
use duels_strategy::{deny_vp, stance_in, Context, PriorWeights, Stance, ThreatWeights, VpWeights};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub use blend::{Blend, Commitment, TermWeights};
pub use terms::{DevSupply, MAX_UNITS};

/// Scores within this distance of the best are treated as tied, and one is
/// chosen uniformly at random.
const TIE_EPSILON: f64 = 1e-6;

/// The knobs of the science ladder, which is the one term with enough
/// internal structure to want its own group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScienceWeights {
    /// Value of holding 0, 1, 2, 3, 4 or 5 distinct symbols. Convex: the
    /// marginal symbol is worth more the closer six gets. Six is an outright
    /// win and is handled by the terminal check, so it is not in the table.
    pub ladder: [f64; 6],
    /// How much each of Law / Theology / Strategy still on the board raises
    /// the whole ladder. Those three are what make a science plan pay beyond
    /// the printed points, so a token row without them is a weaker reason to
    /// chase pairs.
    pub strong_token_mult: f64,
    /// Weight on the half-pair threat as a whole.
    pub pair_threat_weight: f64,
    /// What fraction of the best board token's value (priced by
    /// [`duels_strategy::science::token_value`]) a *threatened* pair is worth.
    /// Below one because the second copy still has to be taken.
    pub pair_token_share: f64,
    /// Flat value per threatened pair, for the turn the opponent has to spend
    /// if they would rather deny it — a tempo tax they pay whether or not the
    /// token itself is valuable.
    pub pair_tempo_tax: f64,
}

impl Default for ScienceWeights {
    fn default() -> Self {
        Self {
            ladder: [0.0, 1.0, 2.5, 6.0, 12.0, 18.0],
            strong_token_mult: 0.15,
            pair_threat_weight: 1.0,
            pair_token_share: 0.5,
            pair_tempo_tax: 0.5,
        }
    }
}

/// The base weight of each evaluation term, before the commitment blend
/// multiplies it.
///
/// Everything is in rough victory-point units, so the numbers can be compared
/// with each other and with [`EvalWeights::instant_result`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalWeights {
    /// Reward per step of conflict-pawn position, per player. Board position
    /// is compounding pressure — loot tokens, proximity to an instant win —
    /// before it pays off in points. Scaled by military commitment.
    pub military_position: f64,
    /// Weight on the quadratic "somebody is about to win outright" term. Not
    /// commitment-scaled: an opponent three steps from the capital is urgent
    /// whether or not this player has a military plan of their own.
    pub military_endgame_urgency: f64,
    /// Weight on the "as if the game ended now" card / wonder / token points.
    /// Fades as commitment rises: a race, once it is real, is worth more than
    /// the points it costs.
    pub vp_projection: f64,
    /// Weight on `floor(coins / 3)`, matching the real scoring rule.
    pub coins_div3: f64,
    /// Weight on the development term. In coins, so the default is the real
    /// `floor(coins / 3)` rate: a coin this city never has to spend is worth
    /// exactly what a coin in hand is worth.
    pub development: f64,
    /// What fraction of a player's remaining decisions become builds that
    /// actually pay a resource cost. Not one: some decisions are discards,
    /// wonder builds paid from elsewhere, or chain builds that cost nothing.
    pub development_take_rate: f64,
    /// Weight on the science ladder.
    pub science_ladder: f64,
    /// Knobs of the ladder itself.
    pub science: ScienceWeights,
    /// Weight on cash-on-hand for taking a contested race card. The one term
    /// that *rises* with commitment.
    pub race_card_liquidity: f64,
    /// Coins past which further cash is points rather than liquidity.
    pub race_liquidity_cap: f64,
    /// The coin cushion below which a position is financially risky.
    pub coin_safety_floor: f64,
    /// Weight on the shortfall below that cushion.
    pub coin_safety_penalty: f64,
    /// Weight on the average per-unit trade price a player faces.
    pub resource_vulnerability: f64,
    /// Penalty per point of free chain-build value handed to the opponent for
    /// their very next turn.
    pub deny_chain_gift: f64,
    /// Weight on the rough power of drafted-but-unbuilt wonders.
    pub wonder_potential: f64,
    /// What beginning each age is worth, indexed by age minus one. Zero for
    /// Age III, which has no next age.
    pub next_age_start: [f64; 3],
    /// Weight on [`duels_strategy::deny_vp`], which prices what a move does to
    /// the opponent's race magnitudes in the same victory-point channel as
    /// everything else.
    pub deny: f64,
    /// What the denial term is multiplied by when the *opponent* is fully
    /// committed to a race. Scaled continuously by their `S(c)`, so a rising
    /// opposing plan makes denial worth more without any threshold.
    pub deny_opponent_commit_boost: f64,
    /// Magnitude assigned when a move actually ends the game. Far larger than
    /// every other term's plausible range put together.
    pub instant_result: f64,
}

impl Default for EvalWeights {
    fn default() -> Self {
        Self {
            // Half of `greedy-ev`'s 0.6 / 3.0, because these are read per
            // player and then differenced, which doubles them.
            military_position: 0.3,
            military_endgame_urgency: 1.5,
            vp_projection: 1.0,
            coins_div3: 1.0,
            development: 1.0 / 3.0,
            development_take_rate: 0.6,
            science_ladder: 1.0,
            science: ScienceWeights::default(),
            race_card_liquidity: 0.15,
            race_liquidity_cap: 8.0,
            coin_safety_floor: 3.0,
            coin_safety_penalty: 0.5,
            resource_vulnerability: 0.4,
            deny_chain_gift: 0.5,
            wonder_potential: 0.5,
            next_age_start: [4.0, 3.0, 0.0],
            deny: 1.0,
            deny_opponent_commit_boost: 1.5,
            instant_result: 1000.0,
        }
    }
}

/// Everything [`PhasedAgent`] can be tuned with.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Config {
    /// Base term weights.
    pub eval: EvalWeights,
    /// The commitment blend.
    pub blend: Blend,
}

impl Config {
    /// A short, reproducible encoding of the configuration, for
    /// [`AgentSpec::params`].
    pub fn params_string(&self) -> String {
        let e = &self.eval;
        let b = &self.blend;
        format!(
            "mil={:.2}/{:.2},vp={:.2},coin={:.2},dev={:.3}@{:.2},sci={:.2},raceliq={:.2},econ={:.1}/{:.2}/{:.2},chain={:.2},wonder={:.2},start={:?},deny={:.2}x{:.2},win={:.0}|\
             blend={},a={:.2},b={:.2},n={:.1},c0={:.2},floors={:.2}/{:.2}/{:.2}/{:.2}/{:.2},boosts={:.2}/{:.2}",
            e.military_position,
            e.military_endgame_urgency,
            e.vp_projection,
            e.coins_div3,
            e.development,
            e.development_take_rate,
            e.science_ladder,
            e.race_card_liquidity,
            e.coin_safety_floor,
            e.coin_safety_penalty,
            e.resource_vulnerability,
            e.deny_chain_gift,
            e.wonder_potential,
            e.next_age_start,
            e.deny,
            e.deny_opponent_commit_boost,
            e.instant_result,
            u8::from(b.enabled),
            b.alpha_m,
            b.beta_prog,
            b.hill_n,
            b.c0,
            b.floor_vp,
            b.floor_liq,
            b.floor_dev,
            b.floor_race_liq,
            b.floor_econ,
            b.boost_sci,
            b.boost_mil,
        )
    }
}

/// Everything computed once per decision, from the root position, and reused
/// unchanged for every candidate action and every chance outcome.
///
/// Constructing one is the *only* place the commitment blend is evaluated.
/// [`evaluate`] takes it by reference and has no way to rebuild it, which is
/// what makes root-fixing a property of the types rather than a discipline —
/// see the crate docs.
#[derive(Debug, Clone)]
pub struct Root {
    config: Config,
    stance: Stance,
    /// Indexed by [`Player::index`].
    weights: [TermWeights; 2],
    supply: DevSupply,
    deny_scale: f64,
    age: u8,
}

impl Root {
    /// Read the root position for the player to move.
    pub fn new(state: &GameState, me: Player, config: Config) -> Root {
        let opp = me.other();
        let ctx = Context::with(state, ThreatWeights::default());
        // One `Stance` carries both players' military, science and point
        // reads, so this is the whole strategy layer for one position rather
        // than five separate calls. Its *prior* is deliberately not used —
        // an earlier investigation (`duels-agent-strategist`) found that the
        // shape of a policy prior does not transfer to an additive evaluation
        // score. What is used is `delta_m` / `deny_vp`, which need a `Stance`
        // only as the carrier of the reads they price against.
        let stance = stance_in(state, me, PriorWeights::default(), &ctx);
        let edge_me = stance.vp.structural_edge;
        let edge_opp =
            duels_strategy::vp_read_with(state, opp, &ctx, &VpWeights::default()).structural_edge;

        let commit_me = Commitment::of(&stance.science, &stance.military, edge_me, &config.blend);
        let commit_opp = Commitment::of(
            &stance.opponent_science,
            &stance.opponent_military,
            edge_opp,
            &config.blend,
        );

        let mut weights = [TermWeights::of(commit_me, &config.blend); 2];
        weights[opp.index()] = TermWeights::of(commit_opp, &config.blend);

        Root {
            deny_scale: 1.0 + (config.eval.deny_opponent_commit_boost - 1.0) * commit_opp.s,
            supply: DevSupply::of(&ctx.board),
            age: state.age(),
            stance,
            weights,
            config,
        }
    }

    /// The configuration in force.
    #[inline]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The root-fixed term multipliers for one player.
    #[inline]
    pub fn weights(&self, p: Player) -> &TermWeights {
        &self.weights[p.index()]
    }

    /// One player's commitment scalars.
    #[inline]
    pub fn commitment(&self, p: Player) -> &Commitment {
        &self.weights[p.index()].commitment
    }

    /// The root stance, for diagnostics.
    #[inline]
    pub fn stance(&self) -> &Stance {
        &self.stance
    }

    /// The development supply statistics, for diagnostics.
    #[inline]
    pub fn supply(&self) -> &DevSupply {
        &self.supply
    }

    /// The age the root position was in.
    ///
    /// Load-bearing for correctness, not a convenience: see
    /// [`terms::chain_gift_exposure`], the one term that reads a card in the
    /// structure and therefore has to stand down once a move has ended the
    /// age and the engine has dealt a whole new one out of a deck no
    /// observation can see.
    #[inline]
    pub fn age(&self) -> u8 {
        self.age
    }

    /// The multiplier the denial term carries, given how committed the
    /// opponent is.
    #[inline]
    pub fn deny_scale(&self) -> f64 {
        self.deny_scale
    }

    /// `A(action)`: the victory-point equivalent of what this action does to
    /// the opponent's race magnitudes, scaled by how committed they are.
    ///
    /// A function of the root position and the action only, so it is added
    /// once per candidate rather than once per chance outcome.
    pub fn denial_term(&self, action: Action) -> f64 {
        self.config.eval.deny * self.deny_scale * deny_vp(action, &self.stance)
    }
}

/// Score `state` for `me`, higher is better, under the root-fixed weights in
/// `root`.
///
/// A finished game is scored by `instant_result` alone, dwarfing every other
/// term; otherwise every term is read for each player separately and
/// differenced.
pub fn evaluate(state: &GameState, me: Player, root: &Root) -> f64 {
    if let Some(result) = state.result() {
        return match result {
            GameResult::Win { winner, .. } if winner == me => root.config.eval.instant_result,
            GameResult::Win { .. } => -root.config.eval.instant_result,
            GameResult::Draw => 0.0,
        };
    }
    player_value(state, me, root) - player_value(state, me.other(), root)
}

/// Every term, read for one player and weighted by *that player's* root-fixed
/// commitment multipliers.
fn player_value(state: &GameState, p: Player, root: &Root) -> f64 {
    let e = &root.config.eval;
    let w = &root.weights[p.index()];
    let breakdown = scoring::breakdown(state, p);

    // --- fading with commitment -------------------------------------------
    let points = w.vp * e.vp_projection * terms::card_and_token_vp(&breakdown);
    let liquidity = w.liquidity * e.coins_div3 * f64::from(breakdown.coins);
    let development = w.development
        * e.development
        * terms::development_value(state, p, &root.supply, e.development_take_rate);
    let economy = w.economy
        * (e.coin_safety_penalty * -terms::coin_shortfall(state, p, e.coin_safety_floor)
            + e.resource_vulnerability * -terms::average_trade_price(state, p));

    // --- sharpening with commitment ---------------------------------------
    let science = w.science * e.science_ladder * terms::science_ladder(state, p, &e.science);
    let military = w.military * e.military_position * terms::military_position(state, p);
    let race_liquidity = w.race_liquidity
        * e.race_card_liquidity
        * terms::race_liquidity(state, p, e.race_liquidity_cap);

    // --- never scaled -----------------------------------------------------
    let urgency = e.military_endgame_urgency * terms::military_urgency(state, p);
    let start = terms::next_age_start(state, p, e);
    let wonders = e.wonder_potential * terms::wonder_potential(state, p);
    let gift = -e.deny_chain_gift * terms::chain_gift_exposure(state, p, root.age);

    points
        + liquidity
        + development
        + economy
        + science
        + military
        + race_liquidity
        + urgency
        + start
        + wonders
        + gift
}

/// The probability-weighted expected value of taking `action` in `state`,
/// plus the action's own denial term.
///
/// The chance-expectation machinery is `duels-agent-greedy-ev`'s, unchanged:
/// enumerate every way the action's randomness could resolve via
/// [`engine::chance_outcomes`] (a single certain outcome for the large
/// majority of actions), apply each to its own copy of `state`, and average
/// the scores by their true probabilities rather than committing to one
/// sampled guess.
pub fn expected_value(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => evaluate(&next, me, root),
            // `action` came from `legal_actions` for `state` and `outcome`
            // from `chance_outcomes` for the same pair, so this is
            // unreachable; score the pre-action state rather than silently
            // dropping probability mass from the expectation.
            Err(_) => evaluate(state, me, root),
        };
        acc += prob * value;
    }
    acc + root.denial_term(action)
}

/// A 1-ply agent that re-reads what matters before every decision.
#[derive(Debug, Clone)]
pub struct PhasedAgent {
    rng: StdRng,
    config: Config,
    root_builds: u64,
}

impl PhasedAgent {
    /// A new agent seeded from `seed`, using [`Config::default`].
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, Config::default())
    }

    /// A new agent seeded from `seed`, with an explicit configuration.
    pub fn with_config(seed: u64, config: Config) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            config,
            root_builds: 0,
        }
    }

    /// A new agent driven by an existing RNG, so a caller can draw many
    /// independent agents from one stream.
    pub fn from_rng(rng: StdRng) -> Self {
        Self {
            rng,
            config: Config::default(),
            root_builds: 0,
        }
    }

    /// The configuration this agent is using.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// How many times this agent has built a [`Root`] — that is, how many
    /// times it has evaluated the commitment blend.
    ///
    /// Instrumentation for the root-fixing property: this must equal the
    /// number of [`Agent::choose`] calls that got past the trivial
    /// single-legal-action shortcut, however many candidate actions and
    /// chance outcomes each of them had to score. See
    /// `tests::root_weights_are_built_exactly_once_per_choose`.
    pub fn root_builds(&self) -> u64 {
        self.root_builds
    }
}

impl Agent for PhasedAgent {
    fn spec(&self) -> AgentSpec {
        AgentSpec {
            name: "phased".to_string(),
            version: "1.0.0".to_string(),
            params: self.config.params_string(),
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
        // Sampled once per call, purely as a vehicle for the engine's chance
        // API (which needs a concrete `GameState`) — `greedy-ev`'s pattern.
        // Nothing downstream reads a hidden identity, so it does not matter
        // which world this invents.
        let base_state = obs.sample_state(&mut self.rng);

        // The one and only place the blend is evaluated for this decision.
        let root = Root::new(&base_state, me, self.config);
        self.root_builds += 1;

        let mut scored: Vec<(Action, f64)> = Vec::with_capacity(legal.len());
        for &action in legal {
            scored.push((action, expected_value(&base_state, action, me, &root)));
        }

        let Some(best_score) = scored.iter().map(|&(_, s)| s).fold(None, |m, s| match m {
            Some(b) if b >= s => Some(b),
            _ => Some(s),
        }) else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::data::Science;
    use duels_core::scoring::VictoryKind;
    use duels_core::testing::StateBuilder;

    /// Twenty of Age II's twenty-three cards, dealt into a real structure
    /// (so some slots are genuinely face down and the science supply model
    /// has something to work with, unlike `open_slots`, which reveals
    /// everything and leaves every unknown-pool weight at zero).
    const AGE_TWO_DEAL: [&str; 20] = [
        "sawmill",
        "brickyard",
        "shelf-quarry",
        "glassblower",
        "drying-room",
        "walls",
        "horse-breeders",
        "barracks",
        "archery-range",
        "parade-ground",
        "library",
        "dispensary",
        "school",
        "laboratory",
        "courthouse",
        "statue",
        "temple",
        "aqueduct",
        "rostrum",
        "forum",
    ];

    /// A mid-game Age II position with a full structure, `built` in Player
    /// One's city, and enough coins for the cost engine not to be the binding
    /// constraint.
    fn age_two_position(built: &[&str]) -> GameState {
        StateBuilder::new()
            .age(2)
            .deal(&AGE_TWO_DEAL)
            .built(Player::One, built)
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build()
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

    // -----------------------------------------------------------------
    // The commitment guard: nobody is committed to anything on turn one.
    // -----------------------------------------------------------------

    /// The trap this whole design exists to avoid: `M_sci` alone reads a
    /// substantial magnitude for a player holding *no symbols at all*,
    /// because with three ages still to come the supply model cannot rule the
    /// race out. If that went straight into the blend, every player would
    /// read as partly science-committed from move one and the science ladder
    /// would be boosted in positions where it means nothing.
    #[test]
    fn a_fresh_game_reads_as_uncommitted_for_both_players() {
        for seed in 0..8u64 {
            let st = engine::new_game(seed);
            let root = Root::new(&st, st.current_player(), Config::default());
            for p in Player::ALL {
                let c = root.commitment(p);
                assert_eq!(
                    c.c_sci.to_bits(),
                    0.0f64.to_bits(),
                    "seed {seed}: c_sci for {p:?} is {} with no symbols held",
                    c.c_sci
                );
                assert!(
                    c.c_mil < 0.10,
                    "seed {seed}: c_mil for {p:?} is {} from the centre",
                    c.c_mil
                );
                assert!(c.c < 0.10, "seed {seed}: c for {p:?} is {}", c.c);
                assert!(
                    c.s < 0.01,
                    "seed {seed}: S(c) for {p:?} is {}, so weights have already moved",
                    c.s
                );
            }
        }
    }

    /// ...and the raw magnitude really is large enough for that to have been
    /// a live trap, so the test above is not vacuous.
    #[test]
    fn the_raw_science_magnitude_alone_would_have_been_misleading() {
        let st = advanced_game(3, 12);
        let r = duels_strategy::science_read(&st, st.current_player());
        assert_eq!(r.distinct, 0, "test setup: expected no symbols held");
        assert!(
            r.magnitude > 0.2,
            "M_sci with no symbols held is only {}, so the prog_sci guard would be pointless",
            r.magnitude
        );
    }

    // -----------------------------------------------------------------
    // Monotonicity properties
    // -----------------------------------------------------------------

    #[test]
    fn commitment_is_zero_at_the_bottom_of_both_races() {
        let blend = Blend::default();
        // No symbols held: c_sci is exactly zero whatever the magnitude says.
        let st = StateBuilder::new().age(1).conflict(0).build();
        let sci = duels_strategy::science_read(&st, Player::One);
        let mil = duels_strategy::military_read(&st, Player::One);
        assert_eq!(sci.distinct, 0);
        assert_eq!(
            mil.need,
            duels_core::data::military().capital_distance,
            "a centred pawn is the full capital distance away"
        );
        let c = Commitment::of(&sci, &mil, 0.0, &blend);
        assert_eq!(c.c_sci.to_bits(), 0.0f64.to_bits());
        assert_eq!(c.s_sci.to_bits(), 0.0f64.to_bits());
        // need == capital_distance and M_mil == 0 must give c_mil == 0.
        if mil.magnitude == 0.0 {
            assert_eq!(c.c_mil.to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn commitment_rises_with_symbols_held() {
        let blend = Blend::default();
        let commit = |built: &[&str]| -> f64 {
            let st = age_two_position(built);
            let sci = duels_strategy::science_read(&st, Player::One);
            let mil = duels_strategy::military_read(&st, Player::One);
            Commitment::of(&sci, &mil, 0.0, &blend).c_sci
        };
        let none = commit(&[]);
        let two = commit(&["workshop", "apothecary"]);
        let four = commit(&["workshop", "apothecary", "scriptorium", "pharmacist"]);
        assert_eq!(none.to_bits(), 0.0f64.to_bits());
        assert!(two > none, "two symbols: {two} vs {none}");
        assert!(four > two, "four symbols: {four} vs {two}");
    }

    // -----------------------------------------------------------------
    // The un-blended baseline
    // -----------------------------------------------------------------

    /// `S(0) = 0` is exact, so a genuinely uncommitted position and a
    /// deliberately switched-off blend must produce *the same weight vector,
    /// bit for bit* — and therefore the same evaluation of every legal
    /// action. This is what "the un-blended baseline" means for this crate:
    /// not another agent's code path, but this agent's own fixed-weight
    /// limit.
    #[test]
    fn the_blend_off_and_a_zero_commitment_position_agree_bit_for_bit() {
        let st = engine::new_game(17);
        let me = st.current_player();

        let on = Root::new(&st, me, Config::default());
        let off = Root::new(
            &st,
            me,
            Config {
                blend: Blend::off(),
                ..Config::default()
            },
        );

        // The military race is not *quite* dead cold at the very start, so
        // pin the comparison to the science half, which is exactly zero, and
        // assert the rest separately.
        for p in Player::ALL {
            let a = on.weights(p);
            let b = off.weights(p);
            assert_eq!(
                a.science.to_bits(),
                b.science.to_bits(),
                "science multiplier for {p:?}: {} vs {}",
                a.science,
                b.science
            );
        }

        // A hand-built position with both races at the floor: no symbols held
        // anywhere, the pawn centred, and — the part that takes an Age III
        // position with no red cards left — no shields obtainable at all, so
        // `M_mil` is exactly zero rather than merely small. Every weight must
        // then match the switched-off blend bit for bit, and so must the
        // evaluation of every legal action.
        let cold = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .conflict(0)
            .coins(Player::One, 5)
            .coins(Player::Two, 5)
            .current(Player::One)
            .build();
        for p in Player::ALL {
            let c = Root::new(&cold, cold.current_player(), Config::default())
                .commitment(p)
                .c;
            assert_eq!(
                c.to_bits(),
                0.0f64.to_bits(),
                "test setup: {p:?} is {c} committed, so this is not the cold case"
            );
        }
        let me = cold.current_player();
        let on = Root::new(&cold, me, Config::default());
        let off = Root::new(
            &cold,
            me,
            Config {
                blend: Blend::off(),
                ..Config::default()
            },
        );
        for p in Player::ALL {
            let (a, b) = (on.weights(p), off.weights(p));
            for (name, x, y) in [
                ("vp", a.vp, b.vp),
                ("liquidity", a.liquidity, b.liquidity),
                ("development", a.development, b.development),
                ("science", a.science, b.science),
                ("military", a.military, b.military),
                ("race_liquidity", a.race_liquidity, b.race_liquidity),
                ("economy", a.economy, b.economy),
            ] {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{name} for {p:?} differs: {x} vs {y}"
                );
            }
        }
        for action in engine::legal_actions(&cold) {
            let a = expected_value(&cold, action, me, &on);
            let b = expected_value(&cold, action, me, &off);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{action:?} scored {a} blended and {b} un-blended"
            );
        }
    }

    #[test]
    fn every_multiplier_is_exactly_one_when_the_blend_is_off() {
        let st = advanced_game(5, 30);
        let me = st.current_player();
        let off = Root::new(
            &st,
            me,
            Config {
                blend: Blend::off(),
                ..Config::default()
            },
        );
        let floor = Blend::default().floor_race_liq;
        for p in Player::ALL {
            let w = off.weights(p);
            for (name, v) in [
                ("vp", w.vp),
                ("liquidity", w.liquidity),
                ("development", w.development),
                ("science", w.science),
                ("military", w.military),
                ("economy", w.economy),
            ] {
                assert_eq!(v.to_bits(), 1.0f64.to_bits(), "{name} for {p:?} = {v}");
            }
            assert_eq!(w.race_liquidity.to_bits(), floor.to_bits());
        }
    }

    // -----------------------------------------------------------------
    // Root-fixing
    // -----------------------------------------------------------------

    #[test]
    fn root_weights_are_built_exactly_once_per_choose() {
        let mut agent = PhasedAgent::new(4);
        let st = advanced_game(9, 16);
        let legal = engine::legal_actions(&st);
        assert!(legal.len() > 1, "test setup: need a real choice");
        let obs = st.observation();

        agent.choose(&obs, &legal, Budget::Nodes(1));
        assert_eq!(
            agent.root_builds(),
            1,
            "one decision over {} candidates must evaluate the blend once",
            legal.len()
        );
        agent.choose(&obs, &legal, Budget::Nodes(1));
        assert_eq!(agent.root_builds(), 2);
    }

    /// The behavioural half of root-fixing: a candidate action that would
    /// materially raise the mover's own commitment must still be scored under
    /// the *root* weights. Re-reading the blend on the result gives a
    /// different number, and that difference is exactly the double-count the
    /// design forbids — the move would be credited once through the term's
    /// value rising and again through the weight on that term rising.
    #[test]
    fn a_committing_move_is_scored_under_the_root_weights_not_its_own() {
        // Player One is two shields from the capital; "circus" (two shields)
        // in slot 18 takes them the whole way, which is about as large a
        // change to `c_mil` as one move can make.
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(4)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let action = Action::Build { slot: 18 };

        let mut after = st;
        let mut rng = StdRng::seed_from_u64(1);
        engine::apply(&mut after, action, &mut rng).expect("the red card should be buildable");
        assert!(
            !after.is_over(),
            "test setup: the move must not end the game"
        );

        let rebuilt = Root::new(&after, me, Config::default());
        assert!(
            rebuilt.commitment(me).s_mil > root.commitment(me).s_mil,
            "test setup: the move should raise the mover's military commitment ({} -> {})",
            root.commitment(me).s_mil,
            rebuilt.commitment(me).s_mil
        );

        // The score the agent actually assigns, under the root weights, and
        // the one it would assign if it re-read the blend on the result.
        let scored = evaluate(&after, me, &root);
        let recomputed = evaluate(&after, me, &rebuilt);
        assert_ne!(
            scored.to_bits(),
            recomputed.to_bits(),
            "the two are indistinguishable, so this test proves nothing"
        );
        // And the value the agent uses is the root-weighted one.
        assert_eq!(
            expected_value(&st, action, me, &root).to_bits(),
            (scored + root.denial_term(action)).to_bits()
        );
    }

    // -----------------------------------------------------------------
    // The individual terms
    // -----------------------------------------------------------------

    /// Grey (glass, papyrus) is the scarcest production in the game — two
    /// cards in Age I, two in Age II, none at all in Age III — and it is what
    /// most wonders ask for. Nothing in the development term says so; it
    /// falls out of counting what the remaining pool actually costs.
    #[test]
    fn the_development_term_prices_a_players_own_production() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .built(Player::One, &["glassworks", "press"])
            .current(Player::One)
            .build();
        let board = duels_strategy::Board::of(&st);
        let supply = DevSupply::of(&board);
        assert!(supply.pool_size > 0);

        let producer = terms::development_value(&st, Player::One, &supply, 0.6);
        let empty = terms::development_value(&st, Player::Two, &supply, 0.6);
        assert_eq!(empty, 0.0, "a city producing nothing develops nothing");
        assert!(
            producer > 0.0,
            "two grey cards should be worth something: {producer}"
        );

        // ...and the split by resource attributes it to glass and papyrus.
        let split = terms::development_by_resource(&st, Player::One, &supply, 0.6);
        let glass = split[duels_core::data::Resource::Glass.index()];
        let papyrus = split[duels_core::data::Resource::Papyrus.index()];
        assert!(glass > 0.0 && papyrus > 0.0, "{split:?}");
        assert!((glass + papyrus - producer).abs() < 1e-9, "{split:?}");
    }

    #[test]
    fn an_unbuilt_wonder_makes_the_resources_it_needs_more_valuable() {
        let build = |wonders: &[&str]| -> f64 {
            let st = StateBuilder::new()
                .age(1)
                .open_slots(&[(18, "clay-pool"), (19, "quarry")])
                .built(Player::One, &["lumber-yard"])
                .wonders(Player::One, wonders)
                .current(Player::One)
                .build();
            let supply = DevSupply::of(&duels_strategy::Board::of(&st));
            terms::development_value(&st, Player::One, &supply, 0.6)
        };
        // The Pyramids need 3 stone; the Great Lighthouse needs wood. Only
        // the latter raises what a Lumber Yard is worth.
        let none = build(&[]);
        let stone = build(&["the-pyramids"]);
        let wood = build(&["the-great-lighthouse"]);
        assert_eq!(none.to_bits(), stone.to_bits(), "{none} vs {stone}");
        assert!(wood > none, "{wood} vs {none}");
    }

    #[test]
    fn the_next_age_start_term_scores_the_projected_starter() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(19, "clay-pool")])
            .conflict(0)
            .current(Player::One)
            .build();
        let w = EvalWeights::default();
        assert_eq!(terms::projected_starter(&st), Some(Player::One));
        assert_eq!(
            terms::next_age_start(&st, Player::One, &w),
            w.next_age_start[0]
        );
        assert_eq!(terms::next_age_start(&st, Player::Two, &w), 0.0);

        // Age III has no next age, so the term is off entirely.
        let late = StateBuilder::new()
            .age(3)
            .open_slots(&[(19, "clay-pool")])
            .conflict(0)
            .current(Player::One)
            .build();
        assert_eq!(terms::next_age_start(&late, Player::One, &w), 0.0);
    }

    #[test]
    fn four_symbols_with_a_strong_token_row_reads_higher_than_without() {
        let four = ["workshop", "apothecary", "scriptorium", "pharmacist"];
        let make = |tokens: &[&str]| {
            StateBuilder::new()
                .age(2)
                .built(Player::One, &four)
                .board_tokens(tokens)
                .open_slots(&[(18, "clay-pool"), (19, "quarry")])
                .current(Player::One)
                .build()
        };
        let w = ScienceWeights::default();
        let bare = make(&[]);
        let strong = make(&["law", "theology", "strategy"]);
        let a = terms::science_ladder(&bare, Player::One, &w);
        let b = terms::science_ladder(&strong, Player::One, &w);
        assert!(b > a, "strong token row: {b} vs bare {a}");
        // The ladder is the dominant part at four symbols.
        assert!(a >= w.ladder[4], "{a} < {}", w.ladder[4]);
    }

    #[test]
    fn a_threatened_pair_is_worth_more_than_a_completed_one() {
        // Holding one Mortar (a live half-pair) versus holding both, which
        // has already paid its token and threatens nothing further.
        let cards: Vec<&str> = terms::symbol_cards(Science::Mortar)
            .map(|c| c.def().id)
            .collect();
        let w = ScienceWeights::default();
        let half = StateBuilder::new()
            .age(2)
            .built(Player::One, &cards[..1])
            .board_tokens(&["philosophy"])
            .build();
        let both = StateBuilder::new()
            .age(2)
            .built(Player::One, &cards)
            .board_tokens(&["philosophy"])
            .pair_already_awarded(Player::One, Science::Mortar)
            .build();
        // Same distinct count, so the ladder entry is identical; only the
        // pair threat differs.
        assert_eq!(half.player(Player::One).distinct_science(), 1);
        assert_eq!(both.player(Player::One).distinct_science(), 1);
        assert!(
            terms::science_ladder(&half, Player::One, &w)
                > terms::science_ladder(&both, Player::One, &w)
        );
    }

    // -----------------------------------------------------------------
    // Whole-agent behaviour
    // -----------------------------------------------------------------

    fn eval_after(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
        let mut s = *state;
        let mut rng = StdRng::seed_from_u64(0x0C0F_FEE0);
        engine::apply(&mut s, action, &mut rng).expect("scenario action should be legal");
        evaluate(&s, me, root)
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
        let root = Root::new(&st, me, Config::default());
        let build = eval_after(&st, Action::Build { slot: 18 }, me, &root);
        let discard = eval_after(&st, Action::Discard { slot: 18 }, me, &root);
        assert!(build > discard, "build={build} discard={discard}");

        let mut after = st;
        let mut rng = StdRng::seed_from_u64(1);
        engine::apply(&mut after, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(
            after.result(),
            Some(GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::MilitarySupremacy,
            })
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
        let root = Root::new(&st, me, Config::default());
        let win = eval_after(&st, Action::Build { slot: 18 }, me, &root);
        let other = eval_after(&st, Action::Build { slot: 19 }, me, &root);
        assert!(win > other, "win={win} other={other}");
    }

    #[test]
    fn evaluation_orders_win_above_draw_above_loss() {
        let root_of = |st: &GameState| Root::new(st, Player::One, Config::default());
        let finish = |one: &[&str], two: &[&str]| -> GameState {
            let mut st = StateBuilder::new()
                .built(Player::One, one)
                .built(Player::Two, two)
                .open_slots(&[(18, "clay-pool")])
                .current(Player::One)
                .build();
            let mut rng = StdRng::seed_from_u64(3);
            engine::apply(&mut st, Action::Discard { slot: 18 }, &mut rng).unwrap();
            assert!(st.result().is_some());
            st
        };
        let win = finish(&["palace"], &[]);
        let draw = finish(&["palace"], &["town-hall"]);
        let loss = finish(&[], &["palace"]);
        let r = root_of(&win);
        let w = EvalWeights::default();
        assert_eq!(evaluate(&win, Player::One, &r), w.instant_result);
        assert_eq!(evaluate(&draw, Player::One, &r), 0.0);
        assert_eq!(evaluate(&loss, Player::One, &r), -w.instant_result);
    }

    #[test]
    fn the_evaluation_is_antisymmetric_between_the_two_players() {
        for seed in 0..6u64 {
            for steps in [8usize, 20, 34] {
                let st = advanced_game(seed, steps);
                if st.is_over() {
                    continue;
                }
                let root = Root::new(&st, st.current_player(), Config::default());
                let a = evaluate(&st, Player::One, &root);
                let b = evaluate(&st, Player::Two, &root);
                assert!(
                    (a + b).abs() < 1e-9,
                    "seed {seed} steps {steps}: {a} and {b} are not opposites"
                );
            }
        }
    }

    #[test]
    fn spec_reports_the_expected_name_and_encoded_params() {
        let agent = PhasedAgent::new(1);
        let spec = agent.spec();
        assert_eq!(spec.name, "phased");
        assert_eq!(spec.version, "1.0.0");
        assert_eq!(spec.params, Config::default().params_string());
    }

    #[test]
    fn choosing_only_ever_returns_one_of_the_offered_actions() {
        let mut agent = PhasedAgent::new(99);
        let state = engine::new_game(99);
        let legal = engine::legal_actions(&state);
        let obs = state.observation();
        for _ in 0..10 {
            assert!(legal.contains(&agent.choose(&obs, &legal, Budget::Nodes(1))));
        }
    }

    #[test]
    fn a_whole_game_of_self_play_terminates_and_stays_legal() {
        let mut a = PhasedAgent::new(1);
        let mut b = PhasedAgent::new(2);
        let mut st = engine::new_game(31);
        let mut rng = StdRng::seed_from_u64(77);
        for _ in 0..400 {
            if st.is_over() {
                break;
            }
            let legal = engine::legal_actions(&st);
            let obs = st.observation();
            let action = if st.current_player() == Player::One {
                a.choose(&obs, &legal, Budget::Nodes(1))
            } else {
                b.choose(&obs, &legal, Budget::Nodes(1))
            };
            assert!(legal.contains(&action));
            engine::apply(&mut st, action, &mut rng).unwrap();
        }
        assert!(st.is_over(), "self-play did not finish");
    }
}
