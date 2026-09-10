//! What a leaf is worth: [`duels_eval`]'s hand-crafted evaluation mapped
//! through a calibrated sigmoid, a playout, or — the default, and this crate's
//! whole reason to exist — an even mixture of the two.
//!
//! # Why a mixture
//!
//! `docs/conventions.md` records a hard-won prior: *simulation beats
//! hand-crafted judgement for position value in this game*. That prior is
//! correct as far as it goes, and [`LeafValue::Static`] is the measurement
//! that confirms it — a pure `duels-eval` leaf is about `-171` Elo against the
//! playout it replaces at a fixed node count. What the same investigation
//! found is that the two signals are *complementary* rather than competing:
//! the evaluation supplies civilian-score judgement, where its terms live, and
//! the playout supplies sight of military races, which are a tempo fact only a
//! simulation walking the next few moves discovers. Half of each beats either
//! alone by a wide margin. See the crate docs for the full measurement.
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
//! nearly free by comparison — which is why the default
//! [`LeafValue::Blend`], which does *both*, measures at throughput parity with
//! a plain playout.
//!
//! The `Root` is the part that is *not* free. `duels-eval`'s
//! `examples/eval_bench.rs` reports one `evaluate` at **15.1% of one
//! [`duels_eval::Root::new`]** — so a `Root` costs about six and a half
//! evaluations, which is affordable once per tree and ruinous once per node,
//! precisely `docs/conventions.md`'s standing note that `duels-strategy`'s
//! reads are cheap per node and unaffordable per simulation (`Root::new` is a
//! slate of exactly those reads). [`crate::tree::Tree`] therefore builds
//! **one** `Root`, in `Tree::new`, from the tree's own root position, and only
//! when [`LeafValue`] actually needs one: the non-default
//! [`LeafValue::Rollout`] allocates nothing and calls nothing here.
//!
//! # Victory points to win probability
//!
//! `duels_eval::evaluate` returns a number on a rough victory-point scale;
//! this tree backs up win probabilities in `[0, 1]` (see [`crate::tree`]'s
//! value convention). The mapping is `duels_eval::win_probability` — moved
//! there (not duplicated) so every consumer of the evaluation, not just this
//! search, reads one calibration. It is a logistic
//!
//! ```text
//! P(me wins) = 1 / (1 + exp(-evaluate(state, me, root) / T(state.age())))
//! ```
//!
//! whose one parameter `T` — the *temperature*, in victory points — was fitted
//! by maximum likelihood over 28,723 self-play positions by
//! `duels-eval`'s `examples/calibrate.rs`. Its headline finding is that **one
//! constant is the wrong model**: the evaluation is nearly twice as sharp in
//! Age III as in Age I, with sign accuracy climbing 0.60 → 0.70 in step — see
//! `duels_eval::win_probability_temperature`'s docs for the full per-age
//! table. So the temperature is a per-age lookup, and it reads the **leaf's**
//! age, not the root's, because that is what the fit is conditioned on.
//!
//! ## The temperature is a calibration constant, and it is not re-fitted here
//!
//! Worth saying out loud, because it is the one place this crate's
//! live-tracking design (see the crate docs) has a seam. The constants in
//! `duels_eval::win_probability_temperature` were fitted against the
//! `duels-eval` generation that was current when this leaf value was
//! measured. A later `duels-eval` round that changes the
//! *scale* of `evaluate`'s output — as opposed to its ranking of positions —
//! would leave them mildly mis-calibrated until somebody re-runs
//! `calibrate.rs` and updates them.
//!
//! That is a deliberately accepted, bounded cost rather than a reason to pin
//! the evaluation. A sigmoid temperature is a *monotone* reparameterisation of
//! the same ordering: getting it somewhat wrong flattens or sharpens how
//! confidently a leaf is scored, it does not make the leaf score the wrong
//! position better. The fitted spread across the three ages is only about
//! 1.9x, and the search was measured to be strong across the whole
//! `blend`/`c` plateau, so it is not a knife edge. Re-running `calibrate.rs`
//! after a `duels-eval` round is a worthwhile tune-up; it is not a
//! correctness gate, and nothing here should be turned into one.
//!
//! # The calibration is stale at depth, and neither age choice fixes that
//!
//! `calibrate.rs` fits `T` over positions each scored against **its own**
//! `duels_eval::Root` — that is how `phased` uses the evaluation, one fresh
//! `Root` per decision — so in the fit, the root-fixed pricing context and the
//! position being scored are the *same* position. In this tree they are not:
//! one `Root` is built at the search root and every leaf, however deep and
//! however many ages later, is priced against it. `duels_eval::evaluate` reads
//! `Root`'s cached age for its rails and its menu term while `duels_eval::win_probability_temperature`
//! reads the leaf's own, so a deep leaf is scored by a hybrid the calibration
//! never saw.
//!
//! Reading the root's age instead would not repair this — it would only make
//! the staleness uniform. The real fix is a `Root` rebuilt deeper in the tree,
//! which the cost numbers above rule out at these budgets. So this is a known,
//! measured limitation of the cheap integration rather than a defect, and it
//! is consistent with what the measurements show — a static leaf alone loses
//! badly, while the default blend, which keeps a real playout alongside it,
//! wins clearly.
//!
//! **What a shared `Root` may and may not carry.** The line between the two is
//! `duels-eval`'s to draw, and it drew it: a `Root` holds *prices* — what a
//! shield or a coin or a produced resource is worth in this game — which are
//! properly read once and are what makes the shared `Root` cheap. It must not
//! hold a *quantity about the position*, because this tree will hand it leaves
//! that are different positions. `duels_eval::terms::wonder_p_build` was
//! cached in `Root` when `WonderModel::Rationed` landed, which made it the
//! second kind, and every leaf in this tree was scored by the root turn's
//! wonder-slot and decision counts. It is now derived from the state being
//! scored. Nothing here changed and nothing on the default path moved —
//! `Rationed` is not the default — but the rule is worth stating where it gets
//! violated, which is here rather than in `duels-eval`.
//!
//! **The rule's hardest case, and why it is a `Config` option rather than a
//! fix.** `duels_eval::TermWeights` — the commitment blend's per-term
//! multipliers — is a quantity about the position by the rule above, and the
//! `Root` this tree shares does hold it. `duels-eval` root-fixes it
//! *deliberately*, though, and has a test pinning that: at one ply a weight
//! that moved with the candidate action would credit a committing move twice,
//! once through the term's contents and again through the multiplier on them.
//! So this is not a `p_build`-shaped defect with an obvious repair — the same
//! reading is right for `phased` and stale for this tree.
//!
//! `duels_eval::ScienceProgress` is round ten's answer for the one factor of
//! it that can be re-read cheaply (the science weight's progress half, at the
//! cost of one `distinct_science()` per `player_value` — a whole `Root`
//! rebuild is six and a half evaluations and is what the cost table above
//! rules out). It is **off by default**; see that enum's docs and this crate's
//! `Config` for whether measurement moved it. The rest of the blend's weights
//! remain root-fixed here, and are the known residue of the cheap
//! integration, alongside the calibration staleness above.
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

/// The static value of `state` on this tree's `[0, 1]` scale, **always from
/// [`Player::One`]'s perspective**, under the root-fixed pricing in `root`.
///
/// Consumes no randomness at all, which is what makes
/// [`LeafValue::Static`]'s RNG stream a property of the tree's chance nodes
/// alone. A thin wrapper over `duels_eval::win_probability` — see the module
/// docs above for why the calibration itself lives there, not here.
#[inline]
pub(crate) fn static_value(state: &GameState, root: &duels_eval::Root) -> f64 {
    duels_eval::win_probability(state, Player::One, root)
}

/// The **learned** static value of `state` on this tree's `[0, 1]` scale, also
/// always from [`Player::One`]'s perspective.
///
/// At [`crate::tree::Objective::WinProbability`] (the default), the scalar is
/// `duels_value`'s four-way head collapsed to `P(military) + P(science) +
/// P(civilian)` — see [`duels_value::Dist::win_probability`]. No sigmoid
/// calibration is applied on the way out, and none is needed: unlike
/// [`static_value`], whose input is a victory-point-scale number that has to
/// be squashed by a fitted temperature, this model was trained as a
/// classifier on real outcomes and so emits a probability directly.
///
/// At [`crate::tree::Objective::TargetKind`], the scalar is instead
/// [`duels_value::Dist::p`] of just the one matching
/// [`duels_value::Outcome`] — the same four-way head, read for a single class
/// rather than summed over the three win classes — **evaluated for `me`, the
/// seat this tree is actually searching for, not always [`Player::One`]**.
/// This is what lets a specialist search reuse the exact same trained weights
/// as the generalist: nothing here is retrained, only which component of the
/// model's own four-way softmax the search consumes, and from which player's
/// point of view. Reading it from a game-fixed `Player::One` regardless of
/// `me` would be wrong the same way [`crate::tree::objective_value_of`]'s
/// equivalent bug was: `P(One wins by kind K)` is not a stand-in for
/// `P(Two wins by kind K)`, so the model must be asked about the player who
/// actually needs the answer.
///
/// Consumes no randomness, exactly like [`static_value`], which is what keeps
/// [`LeafValue::Learned`]'s RNG stream a property of the tree's chance nodes
/// alone.
#[inline]
pub(crate) fn learned_value(
    state: &GameState,
    net: &duels_value::Net,
    objective: crate::tree::Objective,
    me: Player,
) -> f64 {
    match objective {
        crate::tree::Objective::WinProbability => {
            f64::from(net.win_probability(state, Player::One))
        }
        crate::tree::Objective::TargetKind(kind) => {
            f64::from(net.evaluate(state, me).p(target_outcome(kind)))
        }
    }
}

/// Which [`duels_value::Outcome`] class a [`crate::tree::Objective::TargetKind`]
/// reads off the model's four-way head — the learned-leaf mirror of
/// [`crate::tree::kind_matches_target`], which makes the identical call for
/// the *terminal* reward: [`VictoryKind::CivilianVictory`] and
/// [`VictoryKind::CivilianTiebreak`] both read
/// [`duels_value::Outcome::CivilianWin`], the same class
/// [`duels_value::Outcome::of`] labels either of them with.
#[inline]
fn target_outcome(kind: duels_core::scoring::VictoryKind) -> duels_value::Outcome {
    use duels_core::scoring::VictoryKind;
    match kind {
        VictoryKind::MilitarySupremacy => duels_value::Outcome::MilitaryWin,
        VictoryKind::ScientificSupremacy => duels_value::Outcome::ScienceWin,
        VictoryKind::CivilianVictory | VictoryKind::CivilianTiebreak => {
            duels_value::Outcome::CivilianWin
        }
    }
}

/// What the search backs up from a leaf it has just added to the tree.
///
/// [`LeafValue::Blend`] at `weight = 0.5` is [`crate::Config::default`] and is
/// what this crate exists to be; the other three variants are kept as
/// measured, documented alternatives (see the crate docs for what each one
/// scores).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeafValue {
    /// Play the position out to a real [`duels_core::GameResult`] under
    /// [`crate::RolloutWeights`] and [`crate::RaceWeights`], and back up
    /// `1.0` / `0.5` / `0.0`.
    ///
    /// This is what `mcts-uct` does, and setting it here (together with
    /// `exploration: 1.0`) turns this agent back into that one. It is not the
    /// default here — the whole point of this crate is that it is not.
    Rollout,
    /// Score the leaf with [`duels_eval::evaluate`] and map it through
    /// `duels_eval::win_probability`. No playout, and no randomness consumed.
    ///
    /// Kept available, and measurably *weaker* than the playout it replaces
    /// (`-171` Elo at `Nodes(2000)`): a static evaluation cannot see a
    /// military race developing three moves out. See the crate docs' victory-
    /// kind breakdown, which is the sharpest diagnostic in this whole line of
    /// work.
    Static,
    /// Play `plies` steps of the ordinary playout policy and then score what
    /// is left with [`Static`](LeafValue::Static) — unless the game ends
    /// first, in which case the real result is used, exactly as
    /// [`Rollout`](LeafValue::Rollout) would.
    ///
    /// The classic truncated-playout compromise: some of the simulation's
    /// ability to discover a tactic the evaluation cannot see, for a fraction
    /// of its cost and variance. Measured as a monotone family — the more
    /// playout is left in, the stronger — which is what pointed at the blend.
    Truncated {
        /// How many plies to play before evaluating. Capped by
        /// [`crate::Config::max_rollout_plies`].
        plies: u32,
    },
    /// `weight * Static + (1 - weight) * Rollout`, both computed. **The
    /// default, at `weight = 0.5`.**
    ///
    /// Strictly *more* work than [`Rollout`](LeafValue::Rollout) — the
    /// playout still happens — so this is an accuracy win, never a throughput
    /// one. It measures at throughput parity all the same, because the
    /// evaluation is about 8% of a simulation.
    ///
    /// # It changes the *scale* of the reward, so it changes what `c` means
    ///
    /// Worth stating as algebra rather than discovering as a tuning curiosity,
    /// because it is why [`crate::Config::default`] moves *two* fields and not
    /// one. A playout's value is a Bernoulli `0`/`1`; blending it with a
    /// static value at `weight = w` shrinks its spread by `1 - w` and offsets
    /// it by the static term. If the static term were *constant*, the blended
    /// reward would be an exact affine map `a + (1-w)·v` of the old one, and
    /// UCB1's argmax would be unchanged — but only if the exploration constant
    /// were scaled to match, since the bonus is *not* multiplied by `1 - w`:
    ///
    /// ```text
    /// a + (1-w)·exploit + c'·bonus   ranks the same as   exploit + (c'/(1-w))·bonus
    /// ```
    ///
    /// So `c' = c·(1 - w)` is the setting that leaves the exploration /
    /// exploitation balance where [`crate::Config::exploration`] was tuned,
    /// and any *other* `c'` is a second, confounded change. At `w = 0.5` that
    /// is `c = 0.5`, which is exactly what this crate defaults to.
    ///
    /// # What the sweep says about that prediction
    ///
    /// It confirms the *direction* and not the exact line. Rescaling `c`
    /// downwards with `w` is worth a lot — `c = 0.5` beat the unchanged
    /// `c = 1.0` at `w = 0.5` by about 25 Elo over 3,600 games — and the whole
    /// high-scoring region lies near `c = 1 - w`. But that region is a broad
    /// plateau, not a ridgeline: at `w = 0.3` the matching `c = 0.7` scored
    /// *below* the unmatched `c = 0.5`, and everything from `w = 0.5, c = 0.5`
    /// to `w = 0.7, c = 0.3` was one statistical tie in the sweep. Treat
    /// `c = c₀(1 - w)` as the right *starting point* for a new weight, not as
    /// a tuned optimum.
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
    /// Score the leaf with [`duels_value`]'s **learned** network — no playout,
    /// no `duels_eval::Root`, and no randomness consumed.
    ///
    /// The learned analogue of [`Static`](LeafValue::Static), and it exists to
    /// answer the same question that variant answered for the hand-crafted
    /// evaluation: *is this a good value signal at all, before the
    /// complementarity question a blend introduces?* Opt-in, never the
    /// default, and `duels_value`'s crate docs record what it measured.
    Learned,
    /// `weight * Learned + (1 - weight) * Rollout`, both computed — the
    /// learned analogue of [`Blend`](LeafValue::Blend).
    ///
    /// The blend exists because this project has already established, for the
    /// hand-crafted evaluation, that a pure static leaf loses badly to a
    /// blended one for a reason specific to this game: a playout discovers
    /// military tempo that a position-shaped judgement cannot. Nothing about
    /// a *learned* static value escapes that limitation — it is still a
    /// function of the position alone — so the same mixture is the obvious
    /// second thing to try, and the same `c = c₀·(1 - weight)` rescaling
    /// argument from [`Blend`](LeafValue::Blend) applies unchanged.
    LearnedBlend {
        /// How much of the learned value to mix in, on `[0, 1]`.
        weight: f64,
    },
}

impl LeafValue {
    /// Whether this variant needs a [`duels_eval::Root`] built for the tree.
    ///
    /// Answered by naming the variants that read the hand-crafted evaluation
    /// rather than by excluding the ones that do not: the learned variants
    /// need no `Root` either, and a negation would have silently started
    /// building one for them.
    #[inline]
    pub fn needs_eval_root(&self) -> bool {
        matches!(
            self,
            LeafValue::Static | LeafValue::Truncated { .. } | LeafValue::Blend { .. }
        )
    }

    /// Whether this variant needs a [`duels_value::Net`] built for the tree.
    ///
    /// Parsing the embedded weights is about a hundred kilobytes of work, so
    /// it is paid once per tree next to the `Root`, and only when a learned
    /// leaf is actually configured.
    #[inline]
    pub fn needs_learned_net(&self) -> bool {
        matches!(self, LeafValue::Learned | LeafValue::LearnedBlend { .. })
    }

    /// A compact, stable description for [`crate::Config::describe`].
    pub fn describe(&self) -> String {
        match self {
            LeafValue::Rollout => "rollout".to_string(),
            LeafValue::Static => "static".to_string(),
            LeafValue::Truncated { plies } => format!("truncated({plies})"),
            LeafValue::Blend { weight } => format!("blend({weight:.3})"),
            LeafValue::Learned => "learned".to_string(),
            LeafValue::LearnedBlend { weight } => format!("learned_blend({weight:.3})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use duels_core::scoring::VictoryKind;
    use duels_core::testing::StateBuilder;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The evaluation configuration this crate scores against: whatever
    /// `duels-eval` currently defaults to, spelled the way
    /// [`crate::tree::Tree::new`] spells it. **Not** a frozen generation — see
    /// the crate docs' "Tracking `duels-eval` live" section for why that is
    /// deliberate, and why a golden-values test belongs in `duels-eval` rather
    /// than here.
    fn tracked() -> duels_eval::Config {
        duels_eval::Config::default()
    }

    // The calibration itself (the temperature table, the sigmoid's fixed
    // points and monotonicity) is tested in `duels-eval`, where it now lives —
    // see `duels_eval::tests::the_temperature_lookup_is_the_calibrated_table`
    // and `duels_eval::tests::the_sigmoid_maps_victory_points_onto_a_probability`.
    // What stays here is specific to this tree's own usage of it.

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

        let root = duels_eval::Root::new(&state, state.current_player(), tracked());
        let p = static_value(&state, &root);
        assert!(
            p < 1e-8,
            "a position the rails give to Player Two scored {p} for Player One"
        );

        // ...and the complement, from the other side of the same rail.
        let two = duels_eval::win_probability(&state, Player::Two, &root);
        assert!(two > 1.0 - 1e-8, "{two}");
    }

    /// The evaluation is antisymmetric, so the two perspectives' probabilities
    /// must sum to one. Production code only ever asks for [`Player::One`];
    /// this is the one place both are read.
    #[test]
    fn the_two_perspectives_are_complementary() {
        for (i, state) in fixed_positions().into_iter().enumerate() {
            let root = duels_eval::Root::new(&state, state.current_player(), tracked());
            let one = static_value(&state, &root);
            let two = duels_eval::win_probability(&state, Player::Two, &root);
            assert!(
                (one + two - 1.0).abs() < 1e-12,
                "position {i}: {one} + {two} != 1"
            );
        }
    }

    /// Every static leaf value must land inside `[0, 1]`, whatever
    /// `duels-eval` currently says about a position — that is what makes it
    /// commensurable with the playout values the same tree backs up, and it
    /// has to keep holding as the evaluation is re-tuned underneath.
    ///
    /// This is the *shape* invariant that replaces `mcts-uct`'s golden-values
    /// test: it constrains the integration rather than freezing the numbers.
    #[test]
    fn every_static_leaf_value_is_a_probability_under_the_tracked_config() {
        for (i, state) in fixed_positions().into_iter().enumerate() {
            let root = duels_eval::Root::new(&state, state.current_player(), tracked());
            let p = static_value(&state, &root);
            assert!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "position {i} (age {}): static leaf value {p}",
                state.age()
            );
        }
    }

    /// [`crate::tree::Objective::WinProbability`] must reproduce the
    /// pre-`Objective` call bit-for-bit: same expression, no arithmetic
    /// inserted in between, **regardless of `me`** (this arm ignores it).
    /// This is the `docs/conventions.md` proof that the new option changes
    /// nothing at its default.
    #[test]
    fn learned_value_at_win_probability_is_bit_identical_to_win_probability() {
        use crate::tree::Objective;
        let net = duels_value::default_net();
        for state in fixed_positions() {
            for me in [Player::One, Player::Two] {
                assert_eq!(
                    learned_value(&state, &net, Objective::WinProbability, me),
                    f64::from(net.win_probability(&state, Player::One)),
                    "me={me:?} must not change the WinProbability arm"
                );
            }
        }
    }

    /// [`crate::tree::Objective::TargetKind`] must read exactly the matching
    /// [`duels_value::Outcome`] component of the model's own four-way head —
    /// not a rescaled or renormalized version of it — for all three win
    /// kinds, the civilian variant must read the same component whichever
    /// civilian [`VictoryKind`] names the target, and — the part a prior
    /// version of this function got wrong — it must read that component
    /// **for `me`**, not always for [`Player::One`]: `P(One wins by kind K)`
    /// is not a stand-in for `P(Two wins by kind K)`, so a tree searching for
    /// `Player::Two` must have this function actually consult `Two`'s own
    /// distribution.
    #[test]
    fn learned_value_at_target_kind_reads_the_matching_outcome_probability_for_me() {
        use crate::tree::Objective;
        use duels_value::Outcome;
        let net = duels_value::default_net();
        let cases = [
            (VictoryKind::MilitarySupremacy, Outcome::MilitaryWin),
            (VictoryKind::ScientificSupremacy, Outcome::ScienceWin),
            (VictoryKind::CivilianVictory, Outcome::CivilianWin),
            (VictoryKind::CivilianTiebreak, Outcome::CivilianWin),
        ];
        for state in fixed_positions() {
            for me in [Player::One, Player::Two] {
                let dist = net.evaluate(&state, me);
                for (kind, outcome) in cases {
                    assert_eq!(
                        learned_value(&state, &net, Objective::TargetKind(kind), me),
                        f64::from(dist.p(outcome)),
                        "me={me:?}: objective TargetKind({kind:?}) did not read \
                         Outcome::{outcome:?} from me's own distribution"
                    );
                }
            }
        }
    }

    /// Fifty reproducible mid-game positions, named by seed rather than
    /// checked in as states: `engine::new_game(seed)` driven `8 + seed % 24`
    /// plies by a uniform policy from a seeded stream, which reaches all three
    /// ages and both movers.
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
}
