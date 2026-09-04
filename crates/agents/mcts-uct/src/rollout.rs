//! The playout policy: a fast, self-contained default policy used from the
//! newly expanded leaf down to a `GameResult`.
//!
//! Deliberately cheap. Rollouts are where nearly all of the search's time
//! goes, so the policy is a single weighted draw over the legal actions with
//! no lookahead and no opponent model. Two kinds of bias were tried here:
//!
//! - a preference ordering over *kinds* of move (weights of 1 across the
//!   board reproduce the uniform baseline exactly, kept as a configuration
//!   option so the effect can be measured — see [`RolloutWeights::UNIFORM`]
//!   and [`RolloutWeights::BIASED`], the shipped default);
//! - for [`Action::Build`] specifically, a handful of *per-card* multipliers
//!   ([`RolloutWeights::SMART`]) that look up statically-known facts about
//!   the card in the slot — whether it is free via a chain the mover already
//!   owns, and whether it would grant a brand-new scientific symbol or
//!   complete a pair. Each lookup is an array index or a bitmask test, so it
//!   stays cheap in isolation, but it is not free: `examples/rollout_bench.rs`
//!   measured it at 10-25% fewer simulations/second than `UNIFORM` at a fixed
//!   time budget (the exact number moves with system load, since a lookup
//!   this small is easily dwarfed by scheduling noise on a busy machine).
//!
//! **`SMART` did not earn that cost.** Two independent 40-game head-to-head
//! runs against `UNIFORM` at equal `Budget::TimeMs` put it at 47.5% and then
//! 50.0% — statistically indistinguishable from a coin flip (n=40 has a
//! ~7.9-point standard error), and never once ahead of `UNIFORM` by a margin
//! that survived the slower throughput. A milder version of the same idea
//! (weaker multipliers) scored 55% in one run, but with only one run and the
//! same-size sample that is not evidence either way. For comparison, `BIASED`
//! against `UNIFORM` scored 60% and then 45% across those same two runs — a
//! swing bigger than any effect being measured, which is itself the
//! headline finding: at n=40 games on a machine shared with other concurrent
//! work, `Budget::TimeMs` head-to-heads are noisy enough (through both game
//! randomness and load-dependent simulation counts) to swallow a real but
//! modest rollout-quality difference. `SMART` is kept here, tested and
//! documented, as a measured negative result rather than shipped as the
//! default — see the module's git history / PR description for the full
//! numbers and how they were obtained.
//!
//! Uniform-random play discards roughly a third of the cards it touches and
//! is indifferent between a free chain-build and a bad trade, which is far
//! worse than any human line; that noise floor is what makes pure-random
//! playouts a weak signal of a position's true value, and is presumably why
//! `BIASED`'s kind-level bias was adopted as the default in the first place.

use duels_core::engine;
use duels_core::{Action, GameResult, GameState};
use rand::rngs::StdRng;
use rand::Rng;

/// Relative weights for the kinds of move a playout can make, plus a few
/// cheap per-card multipliers applied to [`Action::Build`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RolloutWeights {
    /// Weight for [`Action::Build`].
    pub build: f64,
    /// Weight for [`Action::BuildWonder`].
    pub wonder: f64,
    /// Weight for [`Action::Discard`].
    pub discard: f64,
    /// Multiplier on an [`Action::Build`] whose card is free because the
    /// mover already owns its chain prerequisite. A free build is never
    /// worse than the same card bought outright, so this only ever biases
    /// the playout towards a strictly-at-least-as-good line.
    pub chain_free_mult: f64,
    /// Multiplier on an [`Action::Build`] whose card carries a scientific
    /// symbol the mover does not hold at all yet (progress towards the
    /// 6-symbol instant win).
    pub new_symbol_mult: f64,
    /// Multiplier on an [`Action::Build`] whose card carries a scientific
    /// symbol the mover already holds exactly once, i.e. completing it wins
    /// a progress token immediately. Kept larger than `new_symbol_mult`
    /// since the payoff is immediate rather than merely a step closer.
    pub pair_complete_mult: f64,
}

impl RolloutWeights {
    /// Every kind equally likely and no per-card bias: the plain
    /// uniform-random playout policy.
    pub const UNIFORM: RolloutWeights = RolloutWeights {
        build: 1.0,
        wonder: 1.0,
        discard: 1.0,
        chain_free_mult: 1.0,
        new_symbol_mult: 1.0,
        pair_complete_mult: 1.0,
    };

    /// The original bias: prefer putting cards into the city, take wonders
    /// readily, discard only when little else looks available. No per-card
    /// distinctions among builds.
    pub const BIASED: RolloutWeights = RolloutWeights {
        build: 4.0,
        wonder: 2.0,
        discard: 1.0,
        chain_free_mult: 1.0,
        new_symbol_mult: 1.0,
        pair_complete_mult: 1.0,
    };

    /// `BIASED` plus the cheap per-card multipliers: a free chain build is
    /// strongly preferred, and builds that grant or complete a scientific
    /// symbol are moderately preferred. **Measured and rejected as the
    /// default** — see the module docs: it costs 10-25% throughput and did
    /// not show a reproducible win rate over [`RolloutWeights::UNIFORM`].
    /// Kept for future experimentation and so the negative result is
    /// reproducible.
    pub const SMART: RolloutWeights = RolloutWeights {
        build: 4.0,
        wonder: 2.0,
        discard: 1.0,
        chain_free_mult: 4.0,
        new_symbol_mult: 1.5,
        pair_complete_mult: 3.0,
    };

    /// Whether this is the uniform policy (no kind bias and no per-card
    /// bias).
    #[inline]
    pub fn is_uniform(&self) -> bool {
        *self == RolloutWeights::UNIFORM
    }

    /// Whether any per-card [`Action::Build`] multiplier is active, i.e.
    /// whether `weight` needs to look at the card in the slot at all. Lets
    /// [`RolloutWeights::weight`] skip the `face_up_card`/`science` lookups
    /// entirely for policies (like [`RolloutWeights::BIASED`]) that never
    /// use them.
    #[inline]
    fn needs_card_lookup(&self) -> bool {
        self.chain_free_mult != 1.0 || self.new_symbol_mult != 1.0 || self.pair_complete_mult != 1.0
    }

    /// The weight of `action` in `state`, where `state.current_player()` is
    /// the mover about to take it.
    #[inline]
    fn weight(&self, state: &GameState, action: Action) -> f64 {
        match action {
            Action::Build { slot } => {
                let mut w = self.build;
                if self.needs_card_lookup() {
                    if let Some(card) = state.face_up_card(slot) {
                        let def = card.def();
                        let mover = state.player(state.current_player());
                        if let Some(prereq) = def.chain_from {
                            if mover.has_built(prereq) {
                                w *= self.chain_free_mult;
                            }
                        }
                        if let Some(sym) = def.science {
                            match mover.science()[sym.index()] {
                                0 => w *= self.new_symbol_mult,
                                1 => w *= self.pair_complete_mult,
                                _ => {}
                            }
                        }
                    }
                }
                w
            }
            Action::BuildWonder { .. } => self.wonder,
            Action::Discard { .. } => self.discard,
            // Effect choices (progress tokens, Mausoleum, destroy, first
            // player) are picked uniformly: there is no cheap ordering over
            // them that is obviously right.
            _ => 1.0,
        }
    }
}

impl Default for RolloutWeights {
    fn default() -> Self {
        RolloutWeights::BIASED
    }
}

/// Pick one action according to `weights`, given the position it would be
/// taken in. `state` is needed only for [`RolloutWeights::SMART`]'s per-card
/// lookups; policies for which [`RolloutWeights::needs_card_lookup`] is false
/// never read it beyond the cheap `Action::Build` match arm.
pub(crate) fn pick(
    state: &GameState,
    weights: &RolloutWeights,
    legal: &[Action],
    rng: &mut StdRng,
) -> Action {
    debug_assert!(!legal.is_empty());
    if weights.is_uniform() || legal.len() == 1 {
        return legal[rng.gen_range(0..legal.len())];
    }
    let total: f64 = legal.iter().map(|&a| weights.weight(state, a)).sum();
    let mut r = rng.gen_range(0.0..total);
    for &a in legal {
        r -= weights.weight(state, a);
        if r < 0.0 {
            return a;
        }
    }
    legal[legal.len() - 1]
}

/// Play `state` out to the end and return the result.
///
/// `max_plies` is a safety net, not a rule: the game is finite, so hitting it
/// means a bug elsewhere. Rather than spinning forever the playout stops and
/// scores the position as if Age III had just ended, which is the least
/// misleading thing available.
pub(crate) fn play_out(
    state: &mut GameState,
    weights: &RolloutWeights,
    buf: &mut Vec<Action>,
    rng: &mut StdRng,
    max_plies: u32,
) -> GameResult {
    for _ in 0..max_plies {
        engine::legal_actions_into(state, buf);
        if buf.is_empty() {
            break;
        }
        let action = pick(state, weights, buf, rng);
        // The state carries a determinized layout sampled at the root (and
        // kept publicly consistent by every forced reveal on the way down),
        // so `apply_unchecked` is free to read it for reveals: this is one
        // determinized playout, exactly as in perfect-information Monte Carlo
        // search, and it never leaks information into a *decision* — those
        // all happen at tree nodes reached through `apply_with_outcome`.
        engine::apply_unchecked(state, action, rng);
    }
    state
        .result()
        .unwrap_or_else(|| duels_core::scoring::civilian_result(state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::data::WonderId;
    use duels_core::testing::StateBuilder;
    use duels_core::Player;
    use rand::SeedableRng;

    #[test]
    fn uniform_weights_are_reported_as_uniform() {
        assert!(RolloutWeights::UNIFORM.is_uniform());
        assert!(!RolloutWeights::BIASED.is_uniform());
        assert!(!RolloutWeights::SMART.is_uniform());
    }

    #[test]
    fn biased_and_smart_agree_when_no_card_carries_a_bonus() {
        // At the very first decision of a fresh game nothing is buildable
        // for free and nobody holds a scientific symbol yet, so SMART's
        // per-card multipliers are all inert and it must reduce to BIASED
        // exactly.
        let state = engine::new_game(0);
        let legal = [
            Action::Build { slot: 1 },
            Action::Discard { slot: 2 },
            Action::BuildWonder {
                slot: 3,
                wonder: WonderId::from_index(0),
            },
        ];
        for &a in &legal {
            assert_eq!(
                RolloutWeights::BIASED.weight(&state, a),
                RolloutWeights::SMART.weight(&state, a),
                "{a:?}"
            );
        }
    }

    #[test]
    fn weighted_picking_respects_the_weights() {
        let state = engine::new_game(0);
        let legal = [
            Action::Build { slot: 1 },
            Action::Discard { slot: 2 },
            Action::BuildWonder {
                slot: 3,
                wonder: WonderId::from_index(0),
            },
        ];
        let w = RolloutWeights::BIASED;
        let mut rng = StdRng::seed_from_u64(1);
        let mut counts = [0u32; 3];
        const N: u32 = 70_000;
        for _ in 0..N {
            let a = pick(&state, &w, &legal, &mut rng);
            let i = legal.iter().position(|&x| x == a).unwrap();
            counts[i] += 1;
        }
        let total = w.build + w.discard + w.wonder;
        for (i, expected) in [w.build, w.discard, w.wonder].iter().enumerate() {
            let got = f64::from(counts[i]) / f64::from(N);
            assert!(
                (got - expected / total).abs() < 0.01,
                "action {i}: got {got}, expected {}",
                expected / total
            );
        }
    }

    /// The crux of `SMART`: a chain-free build must be picked far more often
    /// than an equal-kind build that is not free, at a ratio matching
    /// `chain_free_mult`.
    #[test]
    fn smart_prefers_a_free_chain_build_over_an_equal_kind_alternative() {
        // "fortifications" chains from "palisade"; give Player One
        // "palisade" so "fortifications" is free, alongside an unrelated
        // buildable card ("clay-pool") that carries neither a chain nor a
        // science bonus.
        let st = StateBuilder::new()
            .built(Player::One, &["palisade"])
            .open_slots(&[(18, "fortifications"), (19, "clay-pool")])
            .coins(Player::One, 10)
            .current(Player::One)
            .build();
        assert_eq!(
            duels_core::data::CardId::from_slug("fortifications")
                .unwrap()
                .def()
                .chain_from,
            duels_core::data::CardId::from_slug("palisade")
        );

        let legal = [Action::Build { slot: 18 }, Action::Build { slot: 19 }];
        let w = RolloutWeights::SMART;
        let free_w = w.weight(&st, legal[0]);
        let priced_w = w.weight(&st, legal[1]);
        assert!(
            (free_w / priced_w - w.chain_free_mult).abs() < 1e-9,
            "free={free_w} priced={priced_w} mult={}",
            w.chain_free_mult
        );

        let mut rng = StdRng::seed_from_u64(9);
        let mut hits_free = 0u32;
        const N: u32 = 20_000;
        for _ in 0..N {
            if pick(&st, &w, &legal, &mut rng) == legal[0] {
                hits_free += 1;
            }
        }
        let share = f64::from(hits_free) / f64::from(N);
        let expected = w.chain_free_mult / (w.chain_free_mult + 1.0);
        assert!(
            (share - expected).abs() < 0.01,
            "got {share}, expected {expected}"
        );
    }

    #[test]
    fn a_playout_always_reaches_a_result() {
        let mut buf = Vec::new();
        for seed in 0..20u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut state = engine::new_game(seed);
            let result = play_out(
                &mut state,
                &RolloutWeights::SMART,
                &mut buf,
                &mut rng,
                1_000,
            );
            assert!(state.result().is_some(), "seed {seed} did not finish");
            assert_eq!(state.result().unwrap(), result);
            assert!(matches!(
                result.winner(),
                Some(Player::One) | Some(Player::Two) | None
            ));
        }
    }
}
