//! **The learned weights, pinned.** Twenty fixed positions' leaf values,
//! checked against a table in this file, plus the weights' content hash
//! checked outright.
//!
//! # Why this exists, and why `mcts-eval` deliberately has no equivalent
//!
//! `docs/conventions.md` records the rule and both of its sides: *whether a
//! search that consumes a value library should pin a generation depends on
//! whether the value is incidental to the agent or **is** the agent.*
//! `mcts-eval` is the second case for `duels-eval` — its whole reason to exist
//! is "`duels-eval` inside a search", it is meant to strengthen automatically
//! as evaluation rounds land, and so it pins nothing and holds no
//! golden-values test.
//!
//! This crate is the **first** case for `duels-value`, and the argument is not
//! symmetric with that one:
//!
//! - `duels-eval`'s generations are hand-written weight vectors under
//!   mandatory code-owner review, changed a term at a time with a measurement
//!   attached. `duels-value`'s weights are a *fitted artefact*: a retrain is a
//!   different function, changing this agent's behaviour at every position at
//!   once, and nothing about the training pipeline is reviewed the way a
//!   `Config::vN()` diff is.
//! - The effect this agent measures is **narrow and mechanism-specific** — it
//!   runs through `mcts-eval`'s science-value miscalibration (see the crate
//!   docs' "What it does *not* measure" section). An effect of that shape is
//!   exactly the kind a retrain can silently delete while every test still
//!   passes, which would leave the crate docs' Elo tables describing an agent
//!   that no longer exists.
//!
//! So: **a retrain fails this test.** That is the intended behaviour and not a
//! nuisance. The fix when it fires is to re-run the measurements in the crate
//! docs against the new weights and update both this table and those tables
//! together — never to update this table alone, which would silently re-point
//! the documentation at an unmeasured agent.
//!
//! # What is pinned, and what is not
//!
//! The **leaf value the search actually consumes**, reached the way production
//! reaches it: through a real [`crate::tree::Tree`] at
//! [`crate::Config::default`], so the [`duels_value::Net`] involved is the one
//! `Tree::new` parses at the summation order the default configuration selects.
//! That is what makes this a test of *this agent's leaf* rather than a second
//! copy of `duels-value`'s own tests.
//!
//! Alongside it, for five of the twenty, the whole four-way outcome
//! distribution — because the crate docs' mechanism claim is specifically
//! about *which* victory kind the model sees. A retrain that preserved every
//! total win probability while redistributing science against civilian would
//! pass a scalar-only table and would have changed the one thing this agent's
//! measured Elo runs through.
//!
//! Positions are named by seed and depth rather than checked in as states, the
//! same convention `crate::leaf`'s tests use, and are reached by **legal play
//! from `engine::new_game`** so that every one of them is a board this search
//! could really be handed.
//!
//! # The tolerance
//!
//! `TOLERANCE` is `1e-6` on an `f32`-valued quantity in `[0, 1]`, which is
//! about eight times that type's epsilon near one. Tight enough that no
//! retrain survives it and no weight edit does either; loose enough that
//! `f32` reassociation across target architectures does not turn a green CI
//! red on a different machine — see `duels_value::Summation` for the default
//! order (`TransposedAxpy`, since promoted over `Unrolled4`) and why it is
//! *not* itself expected to need this slack (it is checked bit-for-bit
//! against `Serial`, not merely to within tolerance). Bit-for-bit equality
//! across every order was the alternative and was rejected for the remaining
//! `Unrolled4` case, on portability grounds; `duels_value`'s own
//! `tests/summation_equivalence.rs` measures `Serial` and `Unrolled4` at worst
//! `4.768e-7` apart over 2,000 pairs, which is the scale this bound is set
//! against.

#[cfg(test)]
mod tests {
    use crate::tree;
    use duels_core::{engine, GameState, Player};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// See the module docs' "The tolerance" section before changing this.
    const TOLERANCE: f64 = 1e-6;

    /// The shape and content hash of the weights every number in this file was
    /// taken against — `duels-value`'s `weights/v3.bin`.
    ///
    /// Checked on its own as well as through the values, so that a retrain
    /// fails with "the weights changed" rather than with twenty confusing
    /// numeric mismatches.
    const WEIGHTS_ID: &str = "211x128x4/3e1dd480";

    /// The summation order the table was generated at. Recorded because a
    /// summation order change can move the hidden layer's sum (a
    /// reassociation, for [`duels_value::Summation::Unrolled4`]; a reordered
    /// loop nest that happens to be bit-identical to `Serial`, for the
    /// current default [`duels_value::Summation::TransposedAxpy`]), so the
    /// table is only reproducible to `TOLERANCE` and not to the last bit
    /// across orders in general.
    const SUMMATION: &str = "axpy";

    /// One reproducible position: a game seed and how many plies of uniform
    /// legal play to apply to `engine::new_game(seed)`.
    ///
    /// The policy is a seeded uniform draw, so this is deterministic; the
    /// depths are spread across all three ages so the table is not twenty
    /// samples of one game phase.
    fn position(seed: u64, plies: u32) -> Option<GameState> {
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x601D_0000_601D);
        for _ in 0..plies {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                return None;
            }
            let a = legal[rng.gen_range(0..legal.len())];
            engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action");
        }
        // A finished game is not a position this leaf is ever asked about.
        state.result().is_none().then_some(state)
    }

    /// `(seed, plies, expected leaf value)` — the leaf value
    /// [`crate::Config::default`]'s search assigns from `Player::One`'s
    /// perspective.
    ///
    /// **Generated, not hand-written**: run `regenerate_the_golden_table`
    /// below (it is `#[ignore]`d) and paste. Do not edit an entry by hand to
    /// make a failing test pass; see the module docs.
    const GOLDEN: &[(u64, u32, f64)] = &[
        (0, 6, 0.207779959),
        (1, 12, 0.164361119),
        (2, 18, 0.261967212),
        (3, 24, 0.323361874),
        (4, 30, 0.462161571),
        (5, 36, 0.540856838),
        (6, 42, 0.106912859),
        (7, 48, 0.686690211),
        (8, 52, 0.037132513),
        (9, 56, 0.567905903),
        (10, 8, 0.638949752),
        (11, 14, 0.517001033),
        (12, 20, 0.643608391),
        (13, 26, 0.254723907),
        (14, 32, 0.888707876),
        (15, 38, 0.415695608),
        (16, 44, 0.019701943),
        (17, 50, 0.041186132),
        (18, 54, 0.961764872),
        (19, 58, 0.906231046),
    ];

    /// `(seed, plies, [P(military), P(science), P(civilian), P(loss)])` for the
    /// first five entries of [`GOLDEN`], in `duels_value::Outcome::ALL` order.
    ///
    /// Held as `f64` although the network emits `f32`, so the literals can
    /// carry every digit the generator printed: an `f32` literal with more
    /// precision than the type holds is a clippy error, and rounding the table
    /// to fit would make it less faithful than the comparison needs.
    ///
    /// Pins the *decomposition*, not just its sum — see the module docs for
    /// why a scalar-only table would miss the change that matters most here.
    const GOLDEN_DIST: &[(u64, u32, [f64; 4])] = &[
        (0, 6, [0.036992673, 0.111281894, 0.059505392, 0.792220056]),
        (1, 12, [0.085754678, 0.021291843, 0.057314601, 0.835638940]),
        (2, 18, [0.015223198, 0.020112470, 0.226631537, 0.738032818]),
        (3, 24, [0.014608924, 0.027844837, 0.280908108, 0.676638126]),
        (4, 30, [0.057523295, 0.002094244, 0.402544022, 0.537838459]),
    ];

    /// The weights identity is pinned on its own, so a retrain says so in one
    /// line instead of failing twenty numeric assertions.
    #[test]
    fn the_weights_are_the_ones_the_table_was_taken_against() {
        assert_eq!(
            duels_value::default_weights_id(),
            WEIGHTS_ID,
            "the embedded weights changed. This agent's whole measured strength \
             was taken against {WEIGHTS_ID}; see this module's docs before \
             updating anything here"
        );
        assert_eq!(duels_value::Summation::default().name(), SUMMATION);
    }

    /// **Every frozen generation gets a golden check too, not just the live
    /// default** — `docs/roadmap.md`'s Tier 1-F design, resolved: a
    /// generations registry (`crates/duels-value/weights/generations.json`)
    /// replaces a single hand-pinned hash specifically so this stops being a
    /// one-generation check. This is deliberately the *hash* only, not the
    /// full twenty-position table above: those frozen generations are
    /// reachable for comparison and (`v2`'s case) sit in the frozen
    /// reference panel, but their own behaviour was already fully validated
    /// when *they* were the live default (or, for the arms, in the
    /// promotion write-up that measured them) — what would silently break
    /// here is the constant pointing at the wrong bytes (a copy-paste or a
    /// rebuild-with-different-file mistake), which the hash alone catches.
    #[test]
    fn every_frozen_generation_still_matches_its_recorded_hash() {
        let frozen: &[(&str, &[u8], &str)] = &[
            ("v1", crate::WEIGHTS_V1, "211x128x4/036d2b5e"),
            ("v2", crate::WEIGHTS_V2, "211x128x4/17fee9ab"),
            ("arm-a", crate::WEIGHTS_ARM_A, "211x128x4/21061eaa"),
            ("arm-b", crate::WEIGHTS_ARM_B, "211x128x4/81d06b58"),
            ("arm-c", crate::WEIGHTS_ARM_C, "211x128x4/3b584273"),
            (
                "arm-c2 (tier1-arm-c-prime; identical to the live v3 default)",
                crate::WEIGHTS_ARM_C_PRIME,
                "211x128x4/3e1dd480",
            ),
            (
                "arm-d2 (tier1-arm-d-prime)",
                crate::WEIGHTS_ARM_D_PRIME,
                "211x128x4/6ec85ab3",
            ),
            ("gen3-l05", crate::WEIGHTS_GEN3_L05, "211x128x4/7b93dbf0"),
            ("gen3-l10", crate::WEIGHTS_GEN3_L10, "211x128x4/5afaaed3"),
            (
                "gen3-l05-fixedrecipe",
                crate::WEIGHTS_GEN3_L05_FIXEDRECIPE,
                "211x128x4/d1627cfa",
            ),
            ("nb2000", crate::WEIGHTS_NB2000, "211x128x4/ae81ec4e"),
            ("nb8000", crate::WEIGHTS_NB8000, "211x128x4/4e324c56"),
        ];
        for (name, bytes, want) in frozen {
            let got = duels_value::weights_id(bytes);
            assert_eq!(
                &got, want,
                "mcts-value:weights={name} no longer matches the hash recorded in \
                 crates/duels-value/weights/generations.json -- if this is an \
                 intentional retrain of that generation, update both the constant's \
                 doc comment and generations.json's entry for it together"
            );
        }
    }

    /// Every position in the table must still be a legal, reachable,
    /// unfinished position — otherwise the table below could quietly become a
    /// list of skipped entries.
    #[test]
    fn every_golden_position_is_still_reachable() {
        assert_eq!(
            GOLDEN.len(),
            20,
            "the table is meant to hold twenty entries"
        );
        let mut ages = [0u32; 4];
        for &(seed, plies, _) in GOLDEN {
            let state = position(seed, plies)
                .unwrap_or_else(|| panic!("seed {seed} at {plies} plies is no longer a position"));
            ages[usize::from(state.age()) % 4] += 1;
        }
        // Spread across the game rather than twenty openings.
        for age in 1..=3usize {
            assert!(
                ages[age] > 0,
                "no golden position is in age {age}: {ages:?}"
            );
        }
    }

    /// **The pin.** Each position's leaf value, as this agent's default search
    /// computes it, against the table.
    #[test]
    fn the_learned_leaf_values_match_the_golden_table() {
        for &(seed, plies, want) in GOLDEN {
            let state = position(seed, plies).expect("a reachable position");
            let got = tree::learned_leaf_value_for_test(&state);
            assert!(
                (got - want).abs() <= TOLERANCE,
                "seed {seed} at {plies} plies (age {}): leaf value {got:.9}, \
                 table says {want:.9} (delta {:.3e}, tolerance {TOLERANCE:.0e}). \
                 If duels-value was retrained, read this module's docs: the fix \
                 is to re-measure, not to update this number",
                state.age(),
                (got - want).abs()
            );
            // A leaf value is a probability, whatever the table says.
            assert!((0.0..=1.0).contains(&got), "seed {seed}: {got}");
        }
    }

    /// **The decomposition pin.** The four-way head itself, for five of the
    /// twenty.
    #[test]
    fn the_learned_outcome_distributions_match_the_golden_table() {
        let net = duels_value::default_net();
        for &(seed, plies, want) in GOLDEN_DIST {
            let state = position(seed, plies).expect("a reachable position");
            let dist = net.evaluate(&state, Player::One);
            for (i, outcome) in duels_value::Outcome::ALL.into_iter().enumerate() {
                let got = dist.p(outcome);
                assert!(
                    (f64::from(got) - want[i]).abs() <= TOLERANCE,
                    "seed {seed} at {plies} plies, P({}): {got:.9}, table says {:.9}",
                    outcome.name(),
                    want[i]
                );
            }
            // ...and the scalar the search consumes really is this
            // distribution's win mass, not a separately-computed number.
            let scalar = tree::learned_leaf_value_for_test(&state);
            assert!(
                (scalar - f64::from(dist.win_probability())).abs() <= TOLERANCE,
                "seed {seed}: the leaf value {scalar} is not the head's win mass {}",
                dist.win_probability()
            );
        }
    }

    /// Prints the two tables above, ready to paste. `#[ignore]`d so it never
    /// runs in CI: it is a generator, not a check.
    ///
    /// ```text
    /// cargo test -p duels-agent-mcts-value --lib -- --ignored --nocapture \
    ///     golden::tests::regenerate_the_golden_table
    /// ```
    #[test]
    #[ignore = "a generator for the tables above, not a test"]
    fn regenerate_the_golden_table() {
        println!(
            "    const WEIGHTS_ID: &str = {:?};",
            duels_value::default_weights_id()
        );
        println!(
            "    const SUMMATION: &str = {:?};",
            duels_value::Summation::default().name()
        );
        println!("    const GOLDEN: &[(u64, u32, f64)] = &[");
        for &(seed, plies, _) in GOLDEN {
            let state = position(seed, plies).expect("a reachable position");
            let v = tree::learned_leaf_value_for_test(&state);
            println!("        ({seed}, {plies}, {v:.9}),");
        }
        println!("    ];");
        let net = duels_value::default_net();
        println!("    const GOLDEN_DIST: &[(u64, u32, [f64; 4])] = &[");
        for &(seed, plies, _) in GOLDEN_DIST {
            let state = position(seed, plies).expect("a reachable position");
            let d = net.evaluate(&state, Player::One);
            let p: Vec<String> = duels_value::Outcome::ALL
                .into_iter()
                .map(|o| format!("{:.9}", f64::from(d.p(o))))
                .collect();
            println!("        ({seed}, {plies}, [{}]),", p.join(", "));
        }
        println!("    ];");
    }
}
