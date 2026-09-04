//! Property tests over the threat magnitudes: invariants that must hold in
//! *every* position, checked over many randomised playouts.
//!
//! `threat_calibration.rs` says what the numbers are on positions worth
//! naming. This file says what they can never be anywhere, which is the half
//! that catches the rare interaction a hand-written scenario would miss.
//!
//! Covered invariants:
//!
//! * `0 <= M <= 1` for both races, both players, always;
//! * `ΔM(a) <= M_before` for every legal action — you cannot deny more of a
//!   race than there was;
//! * `dead` implies `M_science == 0`, and `M_science == 1` implies not `dead`;
//! * the one-Mausoleum `obtainable_missing` is never *greater* than the naive
//!   per-symbol count it replaced, and differs from it **only** where two or
//!   more symbols are competing for the same single retrieval;
//! * `Tempo::share` is a probability and the two players' shares sum to one;
//! * every prior is finite and at least the floor.
//!
//! Only this crate's public API is touched, exactly as a search would.

use duels_core::{engine, Action, GameState, Player};
use duels_strategy::masks::ALL_SCIENCE;
use duels_strategy::{
    action_prior, delta_m, military_read_with, science_read_with, stance_in, Context, PriorWeights,
    ScienceRead, ScienceStatus,
};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Deterministic policy parameterised by a mixing constant, so different cases
/// explore different lines rather than replaying one.
fn pick(actions: &[Action], turn: u32, mix: u64) -> Action {
    let i = u64::from(turn).wrapping_mul(mix).wrapping_add(mix >> 7) as usize;
    actions[i % actions.len()]
}

/// The count `obtainable_missing` used to be: every missing symbol with *any*
/// route, each treated as independent.
///
/// This is the bug. Two symbols sitting in the discard pile behind one unbuilt
/// Mausoleum are one route, not two, and this function is kept only so the
/// property test can assert exactly where the two formulas part company.
fn naive_obtainable_missing(r: &ScienceRead) -> u8 {
    u8::try_from(
        ALL_SCIENCE
            .into_iter()
            .filter(|s| {
                r.availability[s.index()].held == 0 && r.availability[s.index()].obtainable()
            })
            .count(),
    )
    .unwrap_or(u8::MAX)
}

/// How many missing symbols have *only* the shared Mausoleum retrieval left.
fn mausoleum_only_count(r: &ScienceRead) -> u8 {
    u8::try_from(
        ALL_SCIENCE
            .into_iter()
            .filter(|s| {
                let a = &r.availability[s.index()];
                a.held == 0 && a.via_mausoleum > 0 && !a.obtainable_without_mausoleum()
            })
            .count(),
    )
    .unwrap_or(u8::MAX)
}

/// Everything that must be true of the magnitudes at one decision point.
fn check(state: &GameState, ctx: &str) -> Stats {
    let mut stats = Stats::default();
    let w = PriorWeights::default();
    let cx = Context::of(state);
    let legal = engine::legal_actions(state);

    // The two shares are a probability split of the same pool.
    let (s1, s2) = (cx.tempo(Player::One).share, cx.tempo(Player::Two).share);
    assert!(
        (0.0..=1.0).contains(&s1) && (0.0..=1.0).contains(&s2),
        "{ctx}: shares {s1} / {s2} are not probabilities"
    );
    assert!(
        (s1 + s2 - 1.0).abs() < 1e-12,
        "{ctx}: shares {s1} + {s2} do not sum to one"
    );

    for p in Player::ALL {
        let mil = military_read_with(state, p, &cx);
        let sci = science_read_with(state, p, &cx);

        for (race, m) in [("military", mil.magnitude), ("science", sci.magnitude)] {
            assert!(
                m.is_finite() && (0.0..=1.0).contains(&m),
                "{ctx}: M_{race} for {p} is {m}"
            );
        }
        if mil.magnitude >= 1.0 {
            stats.certain_military += 1;
        }
        if sci.magnitude >= 1.0 {
            stats.certain_science += 1;
        }

        // A physically impossible race is worth exactly nothing, and a
        // certain one is not impossible.
        if sci.dead {
            assert_eq!(sci.magnitude, 0.0, "{ctx}: a dead race for {p} is not zero");
            assert_eq!(sci.status, ScienceStatus::Closed);
            stats.dead += 1;
        } else if sci.magnitude >= 1.0 {
            assert!(!sci.dead);
        }

        // The one-Mausoleum rule: never more generous than the naive count,
        // and different from it only where the retrieval is actually shared.
        let naive = naive_obtainable_missing(&sci);
        let shared = mausoleum_only_count(&sci);
        assert!(
            sci.obtainable_missing <= naive,
            "{ctx}: {p} obtainable_missing {} exceeds the naive {naive}",
            sci.obtainable_missing
        );
        if sci.obtainable_missing != naive {
            assert!(
                shared >= 2,
                "{ctx}: {p} obtainable_missing {} differs from the naive {naive} \
                 with only {shared} Mausoleum-only symbol(s)",
                sci.obtainable_missing
            );
            assert_eq!(
                naive - sci.obtainable_missing,
                shared - 1,
                "{ctx}: {p} the difference should be exactly the symbols beyond the first"
            );
            stats.mausoleum_contended += 1;
        }
        if shared >= 2 {
            stats.two_mausoleum_only += 1;
        }

        // Denial cannot take away more of a race than there was, in either
        // race, for any legal action.
        let s = stance_in(state, p, w, &cx);
        for &a in &legal {
            let d = delta_m(a, &s);
            assert!(
                d.science.is_finite() && d.military.is_finite(),
                "{ctx}: {p} on {a:?}: non-finite delta {d:?}"
            );
            assert!(
                d.science <= s.opponent_science.magnitude + 1e-12,
                "{ctx}: {p} on {a:?}: dM_science {} exceeds M_before {}",
                d.science,
                s.opponent_science.magnitude
            );
            assert!(
                d.military <= s.opponent_military.magnitude + 1e-12,
                "{ctx}: {p} on {a:?}: dM_military {} exceeds M_before {}",
                d.military,
                s.opponent_military.magnitude
            );
            // ...and it cannot push a magnitude out of range the other way.
            assert!(
                d.science >= s.opponent_science.magnitude - 1.0 - 1e-12,
                "{ctx}: {p} on {a:?}: dM_science {} implies an M above one",
                d.science
            );
            if d.breaks_certainty {
                assert!(
                    s.opponent_science.magnitude >= 1.0 || s.opponent_military.magnitude >= 1.0,
                    "{ctx}: {p} on {a:?}: broke a certainty that was not there"
                );
                stats.broke_certainty += 1;
            }

            let prior = action_prior(state, a, &s);
            assert!(
                prior.is_finite() && prior >= w.floor,
                "{ctx}: {p} on {a:?}: prior {prior}"
            );
        }
        stats.positions += 1;
    }
    stats
}

/// What a sweep actually exercised, so a test can prove it was not vacuous.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Stats {
    positions: usize,
    dead: usize,
    certain_science: usize,
    certain_military: usize,
    two_mausoleum_only: usize,
    mausoleum_contended: usize,
    broke_certainty: usize,
}

impl Stats {
    fn merge(&mut self, other: Stats) {
        self.positions += other.positions;
        self.dead += other.dead;
        self.certain_science += other.certain_science;
        self.certain_military += other.certain_military;
        self.two_mausoleum_only += other.two_mausoleum_only;
        self.mausoleum_contended += other.mausoleum_contended;
        self.broke_certainty += other.broke_certainty;
    }
}

/// Play one game out, checking every decision point.
fn playout_checked(seed: u64, mix: u64) -> Stats {
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37);
    let mut stats = Stats::default();
    let mut steps = 0u32;
    loop {
        let actions = engine::legal_actions(&state);
        if actions.is_empty() {
            break;
        }
        stats.merge(check(
            &state,
            &format!("seed {seed} mix {mix} turn {}", state.turn()),
        ));
        let a = pick(&actions, state.turn(), mix);
        engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action");
        steps += 1;
        assert!(steps < 500, "seed {seed}: game did not terminate");
    }
    stats
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, .. ProptestConfig::default() })]

    /// Every magnitude invariant, at every decision point, over full games.
    /// Expensive — it prices every legal action for both players at every
    /// step — so relatively few cases.
    #[test]
    fn magnitude_invariants_hold_throughout_a_game(
        seed in 0u64..1_000_000,
        mix in 1u64..10_000,
    ) {
        let stats = playout_checked(seed, mix);
        prop_assert!(stats.positions > 40, "{stats:?}");
    }
}

/// A deterministic sweep, so the properties are exercised the same way on
/// every machine — and so the interesting cases can be *proved* to have come
/// up rather than hoped for.
#[test]
fn a_deterministic_sweep_covers_the_cases_the_properties_are_about() {
    let mut stats = Stats::default();
    for seed in 0..24u64 {
        for mix in [3u64, 17, 101, 997] {
            stats.merge(playout_checked(seed, mix));
        }
    }
    println!("magnitude property sweep: {stats:?}");
    assert!(stats.positions > 5_000, "{stats:?}");
    assert!(
        stats.dead > 50,
        "the sweep saw only {} dead science races, so the `dead` implication \
         proved little",
        stats.dead
    );
    assert!(
        stats.certain_military + stats.certain_science > 10,
        "the sweep saw only {} certain races, so the certainty rail proved \
         little",
        stats.certain_military + stats.certain_science
    );
}

/// The Mausoleum contention the new formula exists for does not come up often
/// in random play — a player has to own an unbuilt Mausoleum *and* be short of
/// two symbols that are both in the discard pile — so it gets a built position
/// rather than a hoped-for one.
#[test]
fn the_mausoleum_contention_case_is_reachable_and_the_two_formulas_differ_there() {
    use duels_core::testing::StateBuilder;

    let st = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "palace"), (19, "town-hall")])
        .built(
            Player::One,
            &["pharmacist", "workshop", "scriptorium", "university"],
        )
        .built(Player::Two, &["study", "school"])
        .discard(&["apothecary", "academy"])
        .wonders(Player::One, &["the-mausoleum"])
        .coins(Player::One, 40)
        .current(Player::One)
        .build();

    let cx = Context::of(&st);
    let r = science_read_with(&st, Player::One, &cx);
    assert_eq!(mausoleum_only_count(&r), 2);
    assert_eq!(naive_obtainable_missing(&r), 2);
    assert_eq!(r.obtainable_missing, 1);
    assert!(r.dead);
    // ...and the invariants still hold on it.
    check(&st, "the hand-built Mausoleum contention");
}
