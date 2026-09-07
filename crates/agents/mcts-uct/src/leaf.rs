//! What a leaf is worth: the playout this crate has always used, or
//! [`duels_eval`]'s hand-crafted evaluation mapped through a calibrated
//! sigmoid, or a mixture of the two.
//!
//! # Why this is even a question
//!
//! `CLAUDE.md` records a hard-won prior: *simulation beats hand-crafted
//! judgement for position value in this game* — `alphabeta` with a static
//! leaf won 2.5% of its games against this agent, and swapping the static
//! leaf for a real playout raised that to 19.5%. That prior was measured
//! against `alphabeta`'s own small evaluation, before `duels-eval` existed.
//! `duels-eval` is a much larger instrument: a commitment blend over every
//! weight, an opponent-menu term, terminal rails, forward-looking supply and
//! chain equity. Whether *it* can stand in for a playout is a different
//! question, and this module is the apparatus for asking it.
//!
//! # The cost structure this design is built on
//!
//! Measured, not assumed. The load-bearing numbers are the **ratios** in this
//! crate's `examples/leaf_bench.rs`, because they are taken against each other
//! in one run under one machine load; treat any absolute microsecond figure
//! below as scenery.
//!
//! At `Nodes(2000)`, one full simulation — descent, expansion, playout,
//! backpropagation — costs about **18.8 µs**, and swapping the playout for one
//! [`duels_eval::evaluate`] against a **cached** [`duels_eval::Root`] brings
//! that to about **1.55 µs**: a **12x** speed-up per simulation. The playout is
//! therefore nearly the whole cost of a simulation, and the evaluation is
//! nearly free by comparison.
//!
//! The `Root` is the part that is *not* free. `duels-eval`'s
//! `examples/eval_bench.rs` reports one `evaluate` at **15.1% of one
//! [`duels_eval::Root::new`]** under the pinned configuration — so a `Root`
//! costs about six and a half evaluations, which is affordable once per tree
//! and ruinous once per node — precisely `CLAUDE.md`'s standing note that `duels-strategy`'s
//! reads are cheap per node and unaffordable per simulation, and `Root::new`
//! is a slate of exactly those reads. [`crate::tree::Tree`] therefore builds
//! **one** `Root`, in `Tree::new`, from the tree's own root position, and only
//! when [`LeafValue`] actually needs one: the default [`LeafValue::Rollout`]
//! allocates nothing and calls nothing here.
//!
//! # Victory points to win probability
//!
//! `duels_eval::evaluate` returns a number on a rough victory-point scale;
//! this tree backs up win probabilities in `[0, 1]` (see [`crate::tree`]'s
//! value convention). The mapping between them is a logistic
//!
//! ```text
//! P(Player One wins) = 1 / (1 + exp(-v / T))
//! ```
//!
//! whose one parameter `T` — the *temperature*, in victory points — was fitted
//! by maximum likelihood over 28,723 self-play positions by
//! `duels-eval`'s `examples/calibrate.rs`. That fit is the source of the
//! constants below, and its headline finding is that **one constant is the
//! wrong model**: the evaluation is nearly twice as sharp in Age III as in
//! Age I, with sign accuracy climbing 0.60 → 0.70 in step.
//!
//! | age | fitted `T` (VP) | sign accuracy | positions |
//! |---|---|---|---|
//! | I | 47.57 | 0.602 | 11,452 |
//! | II | 43.75 | 0.679 | 8,877 |
//! | III | 25.18 | 0.701 | 8,394 |
//! | all positions | 38.61 | 0.655 | 28,723 |
//!
//! (Reproduce with `cargo run --release -p duels-eval --example calibrate --
//! 200`; the table above is that command's output, re-run against this pin.)
//!
//! So [`temperature`] is a per-age lookup. It reads the **leaf's** age, not
//! the root's, because that is what the fit is conditioned on.
//!
//! # The calibration is stale at depth, and neither age choice fixes that
//!
//! Worth stating plainly, because it is the most likely explanation for why
//! [`LeafValue::Static`] is so much weaker than the playout it replaces.
//!
//! `calibrate.rs` fits `T` over positions each scored against **its own**
//! `duels_eval::Root` — that is how `phased` uses the evaluation, one fresh
//! `Root` per decision — so in the fit, the root-fixed pricing context and the
//! position being scored are the *same* position. In this tree they are not:
//! one `Root` is built at the search root and every leaf, however deep and
//! however many ages later, is priced against it. `duels_eval::evaluate` reads
//! `Root`'s cached age for its rails and its menu term while [`temperature`]
//! reads the leaf's own, so a deep leaf is scored by a hybrid the calibration
//! never saw.
//!
//! Reading the root's age instead would not repair this — it would only make
//! the staleness uniform. The real fix is a `Root` rebuilt deeper in the tree,
//! which the cost numbers above rule out at these budgets (3.5 µs against a
//! ~0.5 µs evaluation: affordable per tree, not per node). So this is a known,
//! measured limitation of the cheap integration rather than a defect, and it
//! is consistent with what the measurements show — a static leaf alone loses
//! badly, while a *blend* that keeps a real playout alongside it wins clearly.
//!
//! # Where the perspective flip is (and is not)
//!
//! [`static_value`] always evaluates for [`Player::One`], because that is the
//! perspective every node in this tree accumulates; the zero-sum flip happens
//! once, in `Tree::exploit`, exactly as it does for a playout's value. Nothing
//! here evaluates both sides — `tests::the_two_perspectives_are_complementary`
//! is the one place that does, to check that `v(P1) + v(P2) = 1`.
//!
//! # Rails-owned positions
//!
//! `duels-eval`'s terminal rails (`duels_eval::rails`) *replace* the weighted
//! sum with `±imminent` — `500` victory points by default — for a position
//! that is already decided. Through this sigmoid that saturates on its own:
//! `500 / 25.18` is about 19.9, and `1/(1 + e^-19.9)` is within `3e-9` of one.
//! No special case is needed, and
//! `tests::a_rails_owned_position_saturates_the_sigmoid` pins it.

use duels_core::{GameState, Player};

/// The maximum-likelihood temperature for an Age I position, in victory
/// points, from `duels-eval`'s `examples/calibrate.rs` over 28,723 `phased`
/// self-play positions.
pub const TEMPERATURE_AGE_I: f64 = 47.57;

/// The same fit restricted to Age II positions.
pub const TEMPERATURE_AGE_II: f64 = 43.75;

/// The same fit restricted to Age III positions, where the evaluation is
/// nearly twice as sharp as in Age I.
pub const TEMPERATURE_AGE_III: f64 = 25.18;

/// The same fit over every position at once, kept for reference: it is what a
/// single flat constant would have been, and the per-age spread above is why
/// this crate does not use it.
pub const TEMPERATURE_OVERALL: f64 = 38.61;

/// The calibrated temperature for a position in `age`.
///
/// Ages outside `1..=3` cannot occur — [`duels_core::GameState::age`] only
/// ever reports one of the three — and are read as Age III, the sharpest
/// setting, so a hypothetical fourth age could not accidentally get the
/// flattest curve.
#[inline]
pub fn temperature(age: u8) -> f64 {
    match age {
        1 => TEMPERATURE_AGE_I,
        2 => TEMPERATURE_AGE_II,
        _ => TEMPERATURE_AGE_III,
    }
}

/// Map a victory-point score for a position in `age` onto a win probability
/// in `[0, 1]`, through the calibrated logistic.
#[inline]
pub fn win_probability(value: f64, age: u8) -> f64 {
    1.0 / (1.0 + (-value / temperature(age)).exp())
}

/// The static value of `state` on this tree's `[0, 1]` scale, **always from
/// [`Player::One`]'s perspective**, under the root-fixed pricing in `root`.
///
/// Consumes no randomness at all, which is what makes
/// [`LeafValue::Static`]'s RNG stream a property of the tree's chance nodes
/// alone.
#[inline]
pub(crate) fn static_value(state: &GameState, root: &duels_eval::Root) -> f64 {
    win_probability(duels_eval::evaluate(state, Player::One, root), state.age())
}

/// What the search backs up from a leaf it has just added to the tree.
///
/// [`LeafValue::Rollout`] is the default and is bit-for-bit the agent this
/// crate shipped before the option existed — same RNG stream, same tree, same
/// move, and no [`duels_eval::Root`] built at all (see
/// `crate::tests::leaf_rollout_is_the_pre_leaf_agent_move_for_move` and
/// `crate::tree::tests::leaf_rollout_grows_the_same_tree_as_the_pre_leaf_search`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeafValue {
    /// Play the position out to a real [`duels_core::GameResult`] under
    /// [`crate::RolloutWeights`] and [`crate::RaceWeights`], and back up
    /// `1.0` / `0.5` / `0.0`. The default.
    Rollout,
    /// Score the leaf with [`duels_eval::evaluate`] and map it through
    /// [`win_probability`]. No playout, and no randomness consumed.
    Static,
    /// Play `plies` steps of the ordinary playout policy and then score what
    /// is left with [`Static`](LeafValue::Static) — unless the game ends
    /// first, in which case the real result is used, exactly as
    /// [`Rollout`](LeafValue::Rollout) would.
    ///
    /// The classic truncated-playout compromise: some of the simulation's
    /// ability to discover a tactic the evaluation cannot see, for a fraction
    /// of its cost and variance.
    Truncated {
        /// How many plies to play before evaluating. Capped by
        /// [`crate::Config::max_rollout_plies`].
        plies: u32,
    },
    /// `weight * Static + (1 - weight) * Rollout`, both computed.
    ///
    /// Strictly *more* work than [`Rollout`](LeafValue::Rollout) — the
    /// playout still happens — so this is an accuracy experiment, never a
    /// throughput one.
    ///
    /// # It changes the *scale* of the reward, so it changes what `c` means
    ///
    /// Worth stating as algebra rather than discovering as a tuning curiosity.
    /// A playout's value is a Bernoulli `0`/`1`; blending it with a static
    /// value at `weight = w` shrinks its spread by `1 - w` and offsets it by
    /// the static term. If the static term were *constant*, the blended reward
    /// would be an exact affine map `a + (1-w)·v` of the old one, and UCB1's
    /// argmax would be unchanged — but only if the exploration constant were
    /// scaled to match, since the bonus is *not* multiplied by `1 - w`:
    ///
    /// ```text
    /// a + (1-w)·exploit + c'·bonus   ranks the same as   exploit + (c'/(1-w))·bonus
    /// ```
    ///
    /// So `c' = c·(1 - w)` is the setting that leaves the exploration /
    /// exploitation balance where [`crate::Config::exploration`] was tuned,
    /// and any *other* `c'` is a second, confounded change. At `w = 0.5` that
    /// is `c = 0.5`.
    ///
    /// # What the sweep says about that prediction
    ///
    /// It confirms the *direction* and not the exact line. Rescaling `c`
    /// downwards with `w` is worth a lot — `c = 0.5` beat the unchanged
    /// `c = 1.0` at `w = 0.5` by about 50 Elo — and the whole high-scoring
    /// region of the sweep lies near `c = 1 - w`. But the region is a broad
    /// plateau, not a ridgeline: at `w = 0.3` the matching `c = 0.7` scored
    /// *below* the unmatched `c = 0.5`, and everything from `w = 0.5, c = 0.5`
    /// to `w = 0.7, c = 0.3` is one statistical tie. Treat `c = c₀(1 - w)` as
    /// the right *starting point* for a new weight, not as a tuned optimum.
    ///
    /// # The corollary that actually matters
    ///
    /// This is documented here rather than in a tuning note because of what
    /// the algebra rules *out*. Since a constant static term would make the
    /// blend an exact affine no-op under that rescaling, none of the measured
    /// gain can be an artefact of the rescaling itself. What is left is the
    /// information in the static term — and the rescaling on its own, with no
    /// blend, measures as **nothing**: `c = 0.5` alone scores 50.10% over
    /// 3,600 games, against a 49.01% noise floor. And `c = 0.3` alone scores
    /// 35.99% — `-100` Elo — while `blend:0.7,c=0.3` scores `+77`. A setting
    /// that is catastrophic alone and strongly positive inside the blend is
    /// the rescaling this section describes, not a contribution of its own.
    /// See the crate docs for the full attribution table.
    Blend {
        /// How much of the static value to mix in, on `[0, 1]`.
        weight: f64,
    },
}

impl LeafValue {
    /// Whether this variant needs a [`duels_eval::Root`] built for the tree.
    ///
    /// The default answers `false`, which is what keeps the pre-existing code
    /// path free of any new allocation or call.
    #[inline]
    pub fn needs_eval_root(&self) -> bool {
        !matches!(self, LeafValue::Rollout)
    }

    /// A compact, stable description for [`crate::Config::describe`].
    pub fn describe(&self) -> String {
        match self {
            LeafValue::Rollout => "rollout".to_string(),
            LeafValue::Static => "static".to_string(),
            LeafValue::Truncated { plies } => format!("truncated({plies})"),
            LeafValue::Blend { weight } => format!("blend({weight:.3})"),
        }
    }
}

/// The name of the [`duels_eval::Config`] generation `cfg` is, or `"custom"`.
///
/// A results file has to record *which* evaluation generation a leaf value was
/// measured against, or a later `duels-eval` round makes the number
/// uninterpretable. See [`crate::Config::eval_generation`].
pub fn generation_name(cfg: &duels_eval::Config) -> &'static str {
    // Newest first, so today's default reports as `v6` rather than matching
    // some older snapshot that happens to be equal (none is — see
    // `duels_eval::tests::the_generation_snapshots_are_a_chain_of_distinct_configurations`).
    for (name, snapshot) in [
        ("v6", duels_eval::Config::v6()),
        ("v5", duels_eval::Config::v5()),
        ("v4", duels_eval::Config::v4()),
        ("v3", duels_eval::Config::v3()),
        ("v2", duels_eval::Config::v2()),
        ("v1", duels_eval::Config::v1()),
    ] {
        if *cfg == snapshot {
            return name;
        }
    }
    "custom"
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use duels_core::testing::StateBuilder;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The pinned generation, spelled the way the agent spells it.
    fn pinned() -> duels_eval::Config {
        crate::Config::default().eval_generation
    }

    #[test]
    fn the_temperature_lookup_is_the_calibrated_table() {
        assert_eq!(temperature(1).to_bits(), TEMPERATURE_AGE_I.to_bits());
        assert_eq!(temperature(2).to_bits(), TEMPERATURE_AGE_II.to_bits());
        assert_eq!(temperature(3).to_bits(), TEMPERATURE_AGE_III.to_bits());
        // Age III is the sharpest of the three, which is the finding the
        // per-age lookup exists for.
        assert!(temperature(3) < temperature(2));
        assert!(temperature(2) < temperature(1));
        // A flat constant would have been the overall fit, and it is bracketed
        // by the per-age ones.
        assert!(temperature(3) < TEMPERATURE_OVERALL);
        assert!(TEMPERATURE_OVERALL < temperature(1));
    }

    /// The mapping's three fixed points, plus its monotonicity and its range.
    #[test]
    fn the_sigmoid_maps_victory_points_onto_a_probability() {
        for age in 1..=3u8 {
            assert_eq!(win_probability(0.0, age), 0.5, "age {age}");
            let t = temperature(age);
            // One temperature of advantage is the 73% point, by construction.
            let at_t = win_probability(t, age);
            assert!((at_t - 0.731_058_6).abs() < 1e-6, "age {age}: {at_t}");
            // Symmetric about a half, and monotone.
            assert!((win_probability(t, age) + win_probability(-t, age) - 1.0).abs() < 1e-12);
            let mut last = 0.0;
            for v in [-200.0, -50.0, -5.0, 0.0, 5.0, 50.0, 200.0] {
                let p = win_probability(v, age);
                assert!(p > last, "age {age}: not monotone at {v}");
                assert!((0.0..=1.0).contains(&p));
                last = p;
            }
        }
        // The same score is worth more in Age III, where the evaluation is
        // sharper.
        assert!(win_probability(10.0, 3) > win_probability(10.0, 1));
    }

    /// A position the terminal rails own is scored `±imminent` — 500 victory
    /// points — and this sigmoid has to turn that into (very nearly) a
    /// certainty on its own, without a special case.
    #[test]
    fn a_rails_owned_position_saturates_the_sigmoid() {
        // Player Two moves and can close on the conflict track; the rails call
        // the position theirs (this is `rails`' own Rail B fixture).
        let state = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(-7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::Two)
            .build();
        assert_eq!(
            duels_eval::rail_owner(&state, 3, duels_eval::RailModel::On),
            Some(Player::Two),
            "the fixture must be a rails-owned position or the test is vacuous"
        );

        let root = duels_eval::Root::new(&state, state.current_player(), pinned());
        let p = static_value(&state, &root);
        assert!(
            p < 1e-8,
            "a position the rails give to Player Two scored {p} for Player One"
        );

        // ...and the complement, from the other side of the same rail.
        let v = duels_eval::evaluate(&state, Player::Two, &root);
        assert!(win_probability(v, state.age()) > 1.0 - 1e-8, "{v}");
    }

    /// The evaluation is antisymmetric, so the two perspectives' probabilities
    /// must sum to one. Production code only ever asks for [`Player::One`];
    /// this is the one place both are read.
    #[test]
    fn the_two_perspectives_are_complementary() {
        for (i, state) in fixed_positions().into_iter().enumerate() {
            let root = duels_eval::Root::new(&state, state.current_player(), pinned());
            let one = static_value(&state, &root);
            let two = win_probability(
                duels_eval::evaluate(&state, Player::Two, &root),
                state.age(),
            );
            assert!(
                (one + two - 1.0).abs() < 1e-12,
                "position {i}: {one} + {two} != 1"
            );
        }
    }

    /// Fifty reproducible mid-game positions, named by seed rather than
    /// checked in as states: `engine::new_game(seed)` driven `8 + seed % 24`
    /// plies by a uniform policy from a seeded stream, which reaches all three
    /// ages and both movers.
    ///
    /// The generator is deliberately dull and self-contained — a golden-value
    /// test whose positions came from an agent would re-baseline itself every
    /// time that agent changed.
    fn fixed_positions() -> Vec<GameState> {
        let mut out = Vec::with_capacity(50);
        for seed in 0..50u64 {
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x1EAF_1EAF);
            for _ in 0..(8 + seed % 24) {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let a = legal[rng.gen_range(0..legal.len())];
                engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action");
            }
            if state.result().is_none() {
                out.push(state);
            }
        }
        assert!(out.len() >= 45, "only {} usable positions", out.len());
        out
    }

    /// The pin's whole purpose, asserted rather than trusted: the numbers
    /// [`crate::Config::eval_generation`] produces are the numbers this work
    /// was measured against.
    ///
    /// The constants below were captured from `duels_eval::Config::v6()` —
    /// today's `duels_eval::Config::default()` — when the pin was made, by the
    /// `#[ignore]`d `print_the_golden_values` below (which shares this file's
    /// position generator, so the two can never drift apart). If this test
    /// fails, the pinned generation's
    /// arithmetic has moved, and every strength number in this crate's
    /// `LeafValue` documentation was measured against a different evaluator.
    /// The fix is not to update the constants quietly: it is to re-baseline
    /// *and re-measure*, or to pin the generation this crate was measured
    /// against (see `duels_eval::Config::v6`'s contract for the next round).
    ///
    /// # Why a tolerance rather than `to_bits`
    ///
    /// Everywhere else in this repository a golden comparison on `f64` is
    /// exact. It cannot be here: `duels_eval`'s terms call `exp` and `powf`,
    /// and those do not agree bit for bit across platforms — which is exactly
    /// why `duels-eval`'s own `tests/vN_identity.rs` files are same-process
    /// copies rather than recorded digests. The tolerance is `1e-9` relative,
    /// some nine orders of magnitude tighter than any change to a weight or a
    /// model could hide in.
    #[test]
    fn the_pinned_generation_reproduces_its_golden_values() {
        let positions = fixed_positions();
        assert_eq!(
            positions.len(),
            GOLDEN_VALUES.len(),
            "the position generator changed; re-capture with print_the_golden_values"
        );
        for (i, (state, &want)) in positions.iter().zip(GOLDEN_VALUES.iter()).enumerate() {
            let root = duels_eval::Root::new(state, state.current_player(), pinned());
            let got = duels_eval::evaluate(state, Player::One, &root);
            let tol = 1e-9 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "position {i} (age {}): evaluate = {got}, golden {want}",
                state.age()
            );
        }
    }

    /// Re-capture [`GOLDEN_VALUES`] after a deliberate, measured re-baseline:
    ///
    /// ```text
    /// cargo test -p duels-agent-mcts-uct --release \
    ///     leaf::tests::print_the_golden_values -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "a capture tool, not a check; prints the constants the golden test asserts"]
    fn print_the_golden_values() {
        println!("    const GOLDEN_VALUES: &[f64] = &[");
        for state in fixed_positions() {
            let root = duels_eval::Root::new(&state, state.current_player(), pinned());
            let v = duels_eval::evaluate(&state, Player::One, &root);
            println!("        {v:?},");
        }
        println!("    ];");
    }

    /// `evaluate`'s golden values over [`fixed_positions`], in order, under
    /// the pinned `duels_eval::Config::v6()`.
    const GOLDEN_VALUES: &[f64] = &[
        -8.770570092256175,
        10.499330850326587,
        -36.646297264103985,
        0.6136215253130448,
        -57.14781893387184,
        -6.649101258951161,
        29.621452662351835,
        17.894577730322624,
        -50.14222011004918,
        30.191616852130366,
        10.509440332134165,
        48.754871571498384,
        -44.15345179339767,
        1.2772073235824397,
        13.558590879480082,
        30.663912263351662,
        19.09711399964494,
        15.70903338351609,
        -57.54406720695947,
        -23.952103967784968,
        -9.47086305923548,
        -77.0495149344806,
        64.55104041293096,
        6.815531420174832,
        29.278599571145627,
        -1.881176741914036,
        3.3854981235944432,
        -22.451298494175614,
        -12.867793479774619,
        41.630025886263354,
        -2.6087073493535264,
        20.516623195241714,
        1.8485750695290957,
        -9.921729659293213,
        4.682569243365474,
        46.09190584530731,
        -55.41015770742843,
        -9.979526531202236,
        80.40053371565254,
        -23.041443077124963,
        6.357883627259987,
        -34.276214101261544,
        -19.312805246526896,
        -20.928052810000466,
        -82.06730587538483,
        73.8678176292537,
        43.4420916800415,
        -5.277513730163665,
        12.838223120699444,
        3.9732885315116437,
    ];
}
