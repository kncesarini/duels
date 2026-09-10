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
//! `f32` reassociation across target architectures (the hidden layer's sum is
//! four-way unrolled by default — see `duels_value::Summation`) does not turn
//! a green CI red on a different machine. Bit-for-bit equality was the
//! alternative and was rejected for exactly that portability reason;
//! `duels_value`'s own `tests/summation_equivalence.rs` measures the two
//! orders at worst `4.768e-7` apart over 2,000 pairs, which is the scale this
//! bound is set against.

#[cfg(test)]
mod tests {
    use crate::tree;
    use duels_core::{engine, GameState, Player};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// See the module docs' "The tolerance" section before changing this.
    const TOLERANCE: f64 = 1e-6;

    /// The shape and content hash of the weights every number in this file was
    /// taken against — `duels-value`'s `weights/v2.bin` (the mixed-corpus
    /// retrain; see that crate's docs for provenance).
    ///
    /// Checked on its own as well as through the values, so that a retrain
    /// fails with "the weights changed" rather than with twenty confusing
    /// numeric mismatches.
    const WEIGHTS_ID: &str = "211x128x4/17fee9ab";

    /// The summation order the table was generated at. Recorded because the
    /// unroll reassociates the hidden layer's sum, so the table is only
    /// reproducible to `TOLERANCE` and not to the last bit across orders.
    const SUMMATION: &str = "unrolled4";

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
        (0, 6, 0.249034613),
        (1, 12, 0.147285119),
        (2, 18, 0.254448652),
        (3, 24, 0.377593189),
        (4, 30, 0.274122626),
        (5, 36, 0.340896249),
        (6, 42, 0.131832182),
        (7, 48, 0.626218438),
        (8, 52, 0.028342869),
        (9, 56, 0.734272659),
        (10, 8, 0.702561021),
        (11, 14, 0.543785334),
        (12, 20, 0.705659032),
        (13, 26, 0.276564270),
        (14, 32, 0.896245122),
        (15, 38, 0.458042026),
        (16, 44, 0.028793165),
        (17, 50, 0.006973101),
        (18, 54, 0.881050408),
        (19, 58, 0.948036909),
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
        (0, 6, [0.045614015, 0.107367657, 0.096052952, 0.750965416]),
        (1, 12, [0.057360068, 0.007702519, 0.082222529, 0.852714837]),
        (2, 18, [0.012840232, 0.004448325, 0.237160087, 0.745551348]),
        (3, 24, [0.037631761, 0.013930339, 0.326031089, 0.622406840]),
        (4, 30, [0.032998152, 0.000616496, 0.240507990, 0.725877345]),
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
