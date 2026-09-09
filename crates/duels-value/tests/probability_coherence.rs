//! Does the model's win probability *cohere* — as a probability, and as a
//! probability in a **zero-sum** game?
//!
//! `tests/determinization_invariance.rs` next door checks the other
//! non-negotiable, that this crate never sees hidden information. That test
//! would pass just as happily for a network that answered `0.9` to both
//! players at once, because it only ever compares one perspective against
//! itself. This file is the complementary check, and unlike that one it is a
//! check on the **weights** rather than on the code around them.
//!
//! # The property, and why nothing enforces it
//!
//! 7 Wonders Duel is two-player zero-sum with no private information
//! (`CLAUDE.md`), and a draw is a 0.07% event in the corpus these weights were
//! fitted on. So for any position,
//!
//! ```text
//! P(win | Player::One) + P(win | Player::Two) ~= 1
//! ```
//!
//! [`duels_value::features`] is *written* to be perspective-symmetric — every
//! quantity appears as a `me` feature and an `opp` feature, the conflict pawn
//! is relativised so positive always favours `me`, and whose turn it is
//! arrives as a `me_to_move` flag — so `features(state, One)` and
//! `features(state, Two)` really are mirror images of one another. But the
//! network on top of them is a nonlinear function `f`, and **nothing makes
//! `f(x) + f(mirror(x))` equal one.** There is no antisymmetric head, no
//! shared trunk with a negated output, no constraint in the loss. Coherence is
//! something the fit has to *learn*, and `duels-arena`'s `feature_dump` gave
//! it every chance to: it writes **both perspectives of every position** with
//! complementary labels, so the paired rows are right there in the training
//! set.
//!
//! # What it actually measured — read this before trusting the model
//!
//! **The fit did not learn it to better than a few percent, and in places it
//! is off by a lot.** Measured with the diagnostics below on the shipped
//! `weights/v1.bin`:
//!
//! Over **1,277 legal, reachable positions** (played out from `new_game` with
//! legal moves, sampled at ten ply depths):
//!
//! ```text
//!   mean |gap|  0.0559     median  0.0413     p90  0.1208     p99  0.2692
//!   max  0.4814          mean signed (P1 + P2 - 1)  -0.0194
//!   fraction within 0.05:  0.567      within 0.10:  0.833
//! ```
//!
//! So **43% of legal positions miss a 0.05 zero-sum bound**, and the tail
//! reaches 0.48 — a position both perspectives call a near-certain win.
//!
//! At the **opening**, which is not an off-distribution curiosity but the
//! exact position every corpus game starts from, over 512 deals:
//!
//! ```text
//!   P(win | One)   min 0.3886   median 0.4457   max 0.5700   mean 0.4670
//!   P(win | Two)   min 0.3806   median 0.4969   max 0.5823   mean 0.4725
//!   sum            min 0.8610   median 0.9384   max 1.0329   mean 0.9396
//! ```
//!
//! The opening is where the answer is known a priori — the draft has not yet
//! assigned who begins Age I, so the two seats are genuinely symmetric and a
//! coherent head has to answer one half. The model answers about
//! **0.47 to each side**, mass 0.94, consistently across all 512 deals. That
//! is a systematic under-confidence, not sampling noise, and only **45.5%** of
//! the 1,024 perspectives land inside `[0.45, 0.55]`.
//!
//! ## Why this is a defect worth a permanent test but *not* a refutation of
//! the crate's Elo numbers
//!
//! Two things keep it from being fatal, and both should be stated plainly
//! rather than used to wave the finding away.
//!
//! 1. **The search never mixes the two perspectives.** `mcts-eval`'s
//!    `leaf::learned_value` takes `net.win_probability(state, Player::One)`
//!    and nothing else — the whole tree works on one consistent
//!    `Player::One` scale. So an incoherence between the two readouts cannot
//!    make a tree disagree with itself about who is winning; it shows up as
//!    *calibration error on the one scale the search uses*, which MCTS is
//!    considerably more tolerant of than of an inconsistency.
//! 2. **The aggregate calibration is genuinely good** — Brier 0.172 and a
//!    monotone ten-bucket reliability curve on held-out games, at parity with
//!    2,000 nodes of MCTS (see the crate docs). The incoherence is
//!    concentrated, not uniform.
//!
//! What it does mean is that there is **known headroom in the leaf signal that
//! is not architectural** — an antisymmetric head (evaluate once, emit
//! `p` and `1 - p`, or average `f(x)` against `1 - f(mirror(x))`) would make
//! the property exact for free, and this measurement is the argument for
//! trying it. It also means a future retrain must not be judged on Brier
//! alone: these bounds are the second axis.
//!
//! # Why hand-built positions are *also* here, and their sharp caveat
//!
//! [`duels_core::testing::StateBuilder`] is this repository's established way
//! to say exactly what a position is (`CLAUDE.md`, "Testing conventions").
//! Its own docs are explicit that it "performs no rules validation" and "can
//! build states that a real game would never reach", and that bites
//! specifically here: **a truly symmetric board is illegal in this game**,
//! because every card exists exactly once, so two identical cities cannot
//! coexist. Positions built that way are off-manifold and a network's
//! behaviour off-manifold says nothing about its fit. They are therefore kept
//! as a labelled diagnostic only, and every *assertion* in this file is made
//! on positions reached by legal play from [`duels_core::engine::new_game`].
//!
//! The hand-built diagnostic is still worth its keep, because it localises
//! the defect in a way the aggregate cannot. The worst legal-shaped boards are
//! the science-lead ones (`+0.42`, `+0.38`), which is the same place
//! `mcts-eval`'s own measured miscalibration lives and the same mechanism the
//! learned leaf's Elo gain runs through.
//!
//! ```text
//! cargo test -p duels-value --test probability_coherence -- --ignored --nocapture
//! ```

use duels_core::testing::StateBuilder;
use duels_core::{engine, GameState, Player};
use duels_value::{default_net, Outcome};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The ply depths the reachable suite samples, spanning the wonder draft
/// (`0`), the opening moves, and all three ages out to a finished game.
const PLIES: [usize; 10] = [0, 4, 8, 16, 24, 32, 40, 48, 56, 64];

/// How many `new_game` seeds the reachable suite walks. 128 seeds across ten
/// depths is about 1,277 positions — comfortably more than the "~50 hand-built
/// positions" this check was scoped at, and every one of them legal.
const REACHABLE_SEEDS: u64 = 128;

// ===================== the positions =====================================

/// Positions reached by playing *legal* moves from a real
/// [`engine::new_game`], which is the only way this crate can build boards it
/// is sure are reachable.
///
/// Random legal play is the same generator
/// `tests/determinization_invariance.rs` uses, and it is off the training
/// distribution (the corpus is `mcts-eval` self-play), so the numbers here
/// should be read as coherence *over legal positions* rather than over the
/// positions a search actually visits. The `ply == 0` row is the exception and
/// the important one: it is exactly where every corpus game starts.
fn reachable() -> Vec<(u64, usize, GameState)> {
    let mut out = Vec::new();
    let max = PLIES[PLIES.len() - 1];
    for seed in 0..REACHABLE_SEEDS {
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FF_EE11);
        if PLIES.contains(&0) {
            out.push((seed, 0, state.clone()));
        }
        for ply in 1..=max {
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let a = legal[rng.gen_range(0..legal.len())];
            if engine::apply_quiet(&mut state, a, &mut rng).is_err() {
                break;
            }
            if PLIES.contains(&ply) {
                out.push((seed, ply, state.clone()));
            }
        }
    }
    out
}

/// `P(win | One) + P(win | Two) - 1` for every reachable position.
fn signed_gaps() -> Vec<f32> {
    let net = default_net();
    reachable()
        .iter()
        .map(|(_, _, s)| {
            net.win_probability(s, Player::One) + net.win_probability(s, Player::Two) - 1.0
        })
        .collect()
}

fn quantile(sorted: &[f32], f: f32) -> f32 {
    sorted[((sorted.len() as f32 - 1.0) * f).round() as usize]
}

fn sorted_abs(v: &[f32]) -> Vec<f32> {
    let mut w: Vec<f32> = v.iter().map(|x| x.abs()).collect();
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    w
}

// ===================== the assertions ====================================

/// The suite is the size and shape it claims, so nothing below can quietly
/// become a check on three boards.
#[test]
fn the_reachable_suite_is_large_and_spans_the_game() {
    let all = reachable();
    assert!(
        all.len() >= 1_000,
        "only {} reachable positions; this file's numbers were taken on ~1,277",
        all.len()
    );
    for ply in PLIES {
        let n = all.iter().filter(|(_, p, _)| *p == ply).count();
        assert!(
            n >= REACHABLE_SEEDS as usize - 8,
            "ply {ply} has only {n} positions, so games are terminating before it"
        );
    }
    // Anti-vacuity: these are genuinely different boards, not one board
    // sampled a thousand times. Measured by the model distinguishing them.
    let net = default_net();
    let distinct: std::collections::BTreeSet<u32> = all
        .iter()
        .map(|(_, _, s)| net.win_probability(s, Player::One).to_bits())
        .collect();
    assert!(
        distinct.len() > all.len() * 9 / 10,
        "only {} distinct win probabilities over {} positions",
        distinct.len(),
        all.len()
    );
}

/// Zero-sum coherence, at the bounds the shipped weights actually meet.
///
/// **These bounds are loose on purpose and the looseness is the finding**, not
/// an accommodation — see this file's docs for the full measurement and for
/// why a correct antisymmetric head would make the property exact. They are
/// set about 40% above the measured values so ordinary retraining jitter does
/// not trip them, and no looser than that: the point of the test is that a
/// retrain which makes coherence *worse* fails CI.
///
/// Measured on `weights/v1.bin`: mean 0.0559, p90 0.1208, p99 0.2692.
#[test]
fn the_two_perspectives_roughly_sum_to_one() {
    let gaps = signed_gaps();
    let abs = sorted_abs(&gaps);
    let n = abs.len() as f32;
    let mean = abs.iter().sum::<f32>() / n;
    let p90 = quantile(&abs, 0.90);
    let p99 = quantile(&abs, 0.99);
    let worst = abs[abs.len() - 1];
    println!("coherence over {n} reachable positions: mean {mean:.4} p90 {p90:.4} p99 {p99:.4} max {worst:.4}");

    assert!(
        mean < 0.08,
        "mean |P(One) + P(Two) - 1| = {mean:.4}, was 0.0559 when this bound was set"
    );
    assert!(
        p90 < 0.17,
        "p90 |P(One) + P(Two) - 1| = {p90:.4}, was 0.1208 when this bound was set"
    );
    assert!(
        p99 < 0.38,
        "p99 |P(One) + P(Two) - 1| = {p99:.4}, was 0.2692 when this bound was set"
    );
    let within_10 = abs.iter().filter(|g| **g < 0.10).count() as f32 / n;
    assert!(
        within_10 > 0.75,
        "only {:.3} of positions are within 0.10 of coherent; was 0.833",
        within_10
    );
}

/// The tight bound the property *deserves* — asserted to **fail**, so the
/// defect is a permanent, visible fact about this model rather than a note in
/// a doc comment someone stops reading.
///
/// A well-fitted zero-sum value head would put nearly every legal position
/// inside `|P(One) + P(Two) - 1| < 0.05`. This one puts 56.7% of them there.
///
/// **If this test starts failing, the model got better.** That is the good
/// outcome: delete this test and tighten
/// [`the_two_perspectives_roughly_sum_to_one`]'s bounds to match, rather than
/// "fixing" it.
#[test]
fn the_tight_zero_sum_bound_still_does_not_hold() {
    let abs = sorted_abs(&signed_gaps());
    let within_05 = abs.iter().filter(|g| **g < 0.05).count() as f32 / abs.len() as f32;
    println!("fraction of reachable positions within 0.05 of coherent: {within_05:.3}");
    assert!(
        within_05 < 0.95,
        "the model now satisfies a 0.05 zero-sum bound on {within_05:.3} of legal \
         positions (it satisfied 0.567 when this characterisation test was \
         written). This is an IMPROVEMENT: delete this test and tighten \
         the_two_perspectives_roughly_sum_to_one instead of loosening anything."
    );
}

/// The opening is near even **on average**, and the whole opening band is
/// recorded.
///
/// The one position in this file whose answer is known a priori: at
/// `Phase::WonderDraft` the draft has not yet assigned who begins Age I, so
/// the two seats are symmetric and a coherent head must answer one half.
///
/// The averaged form is what passes. The **per-deal** `[0.45, 0.55]` band does
/// *not* — 45.5% of the 1,024 perspectives land inside it, and the extremes
/// are 0.3886 and 0.5823 — so this test asserts the mean inside `[0.45, 0.55]`
/// and every individual deal inside a recorded, much wider `[0.35, 0.65]`.
/// Both numbers matter and the docs above give the distribution.
#[test]
fn the_opening_is_near_even_on_average() {
    let net = default_net();
    let deals = 512u64;
    let mut sums = Vec::new();
    let mut means = [0.0f64; 2];
    for seed in 0..deals {
        let state = engine::new_game(seed);
        let one = net.win_probability(&state, Player::One);
        let two = net.win_probability(&state, Player::Two);
        for (i, p) in [one, two].into_iter().enumerate() {
            means[i] += f64::from(p) / deals as f64;
            assert!(
                (0.35..=0.65).contains(&p),
                "opening (seed {seed}, seat {i}): {p:.4} outside the recorded [0.35, 0.65]"
            );
        }
        sums.push(one + two);
    }
    let mass = sums.iter().sum::<f32>() / deals as f32;
    println!(
        "opening over {deals} deals: mean P(One) {:.4}, mean P(Two) {:.4}, mean mass {mass:.4}",
        means[0], means[1]
    );
    for (i, m) in means.iter().enumerate() {
        assert!(
            (0.45..=0.55).contains(m),
            "mean opening win probability for seat {i} is {m:.4}, outside [0.45, 0.55] \
             (was 0.4670 / 0.4725 when this bound was set)"
        );
    }
    // The systematic under-confidence itself, pinned so a retrain that fixes
    // it is visible and one that worsens it fails.
    assert!(
        (0.90..=1.02).contains(&mass),
        "opening probability mass is {mass:.4}; was 0.9396 when this bound was set"
    );
}

/// Every distribution is a distribution: four finite non-negative numbers
/// summing to one. Cheap, and it is what would break first if `softmax` or the
/// weight loader were damaged.
#[test]
fn every_distribution_is_normalised() {
    let net = default_net();
    for (seed, ply, state) in reachable() {
        for me in [Player::One, Player::Two] {
            let d = net.evaluate(&state, me);
            assert!(
                (d.total() - 1.0).abs() < 1e-4,
                "seed {seed} ply {ply} {me:?}: total mass {:.6}",
                d.total()
            );
            for o in Outcome::ALL {
                let q = d.p(o);
                assert!(
                    q.is_finite() && (0.0..=1.0).contains(&q),
                    "seed {seed} ply {ply} {me:?}: P({}) = {q}",
                    o.name()
                );
            }
        }
    }
}

// ===================== the diagnostics ===================================

/// The per-ply coherence table this file's docs quote.
#[test]
#[ignore = "diagnostic output, not an assertion"]
fn report_coherence_by_ply() {
    let net = default_net();
    let all = reachable();
    println!("reachable positions: {}", all.len());
    println!(
        "{:>5} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "ply", "n", "mean|g|", "p50", "p90", "p99", "max"
    );
    let mut every = Vec::new();
    for ply in PLIES {
        let gaps: Vec<f32> = all
            .iter()
            .filter(|(_, p, _)| *p == ply)
            .map(|(_, _, s)| {
                net.win_probability(s, Player::One) + net.win_probability(s, Player::Two) - 1.0
            })
            .collect();
        if gaps.is_empty() {
            continue;
        }
        every.extend(gaps.iter().copied());
        let abs = sorted_abs(&gaps);
        println!(
            "{ply:>5} {:>6} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
            abs.len(),
            abs.iter().sum::<f32>() / abs.len() as f32,
            quantile(&abs, 0.5),
            quantile(&abs, 0.9),
            quantile(&abs, 0.99),
            abs[abs.len() - 1]
        );
    }
    let abs = sorted_abs(&every);
    println!(
        "  ALL {:>6} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
        abs.len(),
        abs.iter().sum::<f32>() / abs.len() as f32,
        quantile(&abs, 0.5),
        quantile(&abs, 0.9),
        quantile(&abs, 0.99),
        abs[abs.len() - 1]
    );
    println!(
        "mean SIGNED (P1 + P2 - 1) = {:+.4}",
        every.iter().sum::<f32>() / every.len() as f32
    );
    for tol in [0.02f32, 0.05, 0.10, 0.15, 0.20, 0.30] {
        println!(
            "  fraction with |gap| < {tol:.2}: {:.3}",
            abs.iter().filter(|g| **g < tol).count() as f32 / abs.len() as f32
        );
    }
}

/// The opening distribution this file's docs quote.
#[test]
#[ignore = "diagnostic output, not an assertion"]
fn report_the_opening() {
    let net = default_net();
    let mut cols: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for seed in 0..512u64 {
        let s = engine::new_game(seed);
        let a = net.win_probability(&s, Player::One);
        let b = net.win_probability(&s, Player::Two);
        cols[0].push(a);
        cols[1].push(b);
        cols[2].push(a + b);
    }
    println!("opening over 512 deals (min / median / max / mean)");
    for (name, v) in ["P(win|One)", "P(win|Two)", "sum"].iter().zip(cols.iter()) {
        let mut w = v.clone();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {name:<12} {:.4} / {:.4} / {:.4} / mean {:.4}",
            w[0],
            w[w.len() / 2],
            w[w.len() - 1],
            v.iter().sum::<f32>() / v.len() as f32
        );
    }
    for band in [0.02f32, 0.05, 0.10] {
        let inside = cols[0]
            .iter()
            .chain(cols[1].iter())
            .filter(|p| (**p - 0.5).abs() <= band)
            .count();
        println!(
            "  fraction of the 1024 perspectives within +-{band:.2} of 0.5: {:.3}",
            inside as f32 / 1024.0
        );
    }
}

/// Hand-built boards, which localise the defect. **Not assertions**: the
/// "symmetric" ones are illegal (two cities cannot share a card) and the rest
/// are legal-*shaped* rather than proven reachable. See this file's docs.
#[test]
#[ignore = "diagnostic output, not an assertion"]
fn report_hand_built() {
    const SMALL: &[&str] = &["lumber-yard", "clay-pool", "theater", "altar"];
    const MID: &[&str] = &[
        "lumber-yard",
        "clay-pool",
        "quarry",
        "theater",
        "altar",
        "baths",
        "statue",
        "temple",
    ];
    const BIG: &[&str] = &[
        "lumber-yard",
        "clay-pool",
        "quarry",
        "press",
        "glassworks",
        "theater",
        "altar",
        "baths",
        "statue",
        "temple",
        "aqueduct",
        "palace",
        "town-hall",
    ];
    const ROW: &[(u8, &str)] = &[
        (14, "guard-tower"),
        (15, "workshop"),
        (16, "theater"),
        (17, "lumber-yard"),
        (18, "clay-pool"),
        (19, "tavern"),
    ];

    let net = default_net();
    println!(
        "{:<38} {:>8} {:>8} {:>8}",
        "position", "P(One)", "P(Two)", "gap"
    );
    let show = |name: &str, state: &GameState| {
        let one = net.win_probability(state, Player::One);
        let two = net.win_probability(state, Player::Two);
        println!("{name:<38} {one:>8.4} {two:>8.4} {:>+8.4}", one + two - 1.0);
    };

    // Illegal by construction: identical cities cannot coexist.
    for (age, city) in [(1u8, SMALL), (2, MID), (3, BIG)] {
        for mover in [Player::One, Player::Two] {
            show(
                "symmetric (ILLEGAL: shared cards)",
                &StateBuilder::new()
                    .age(age)
                    .built(Player::One, city)
                    .built(Player::Two, city)
                    .coins(Player::One, 7)
                    .coins(Player::Two, 7)
                    .conflict(0)
                    .current(mover)
                    .open_slots(ROW)
                    .build(),
            );
        }
    }
    // Legal-shaped: disjoint cities, one lead per board.
    let boards: [(&str, &[&str], &[&str], i8, u8); 8] = [
        ("legal-early-thin", &["lumber-yard"], &["quarry"], 0, 1),
        (
            "legal-early-mil-vs-sci",
            &["guard-tower", "stable"],
            &["workshop", "apothecary"],
            2,
            1,
        ),
        (
            "legal-mid-military-lead",
            &["guard-tower", "stable", "garrison", "walls", "barracks"],
            &["theater", "altar", "baths", "statue"],
            5,
            2,
        ),
        (
            "legal-mid-science-lead",
            &["workshop", "apothecary", "scriptorium", "library", "school"],
            &["lumber-yard", "clay-pool", "quarry", "theater"],
            0,
            2,
        ),
        (
            "legal-mid-points-lead",
            &["theater", "altar", "baths", "statue", "temple", "aqueduct"],
            &["lumber-yard", "clay-pool"],
            -2,
            2,
        ),
        (
            "legal-mid-economy-lead",
            &["tavern", "brewery", "forum", "caravansery"],
            &["theater", "altar"],
            0,
            2,
        ),
        ("legal-late-lopsided", BIG, &["tavern", "sawmill"], 0, 3),
        ("legal-late-lopsided-rev", &["tavern", "sawmill"], BIG, 0, 3),
    ];
    for (name, one, two, conflict, age) in boards {
        for mover in [Player::One, Player::Two] {
            show(
                name,
                &StateBuilder::new()
                    .age(age)
                    .built(Player::One, one)
                    .built(Player::Two, two)
                    .coins(Player::One, 6)
                    .coins(Player::Two, 6)
                    .conflict(conflict)
                    .current(mover)
                    .open_slots(ROW)
                    .build(),
            );
        }
    }
}
