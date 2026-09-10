//! The non-negotiable invariant: `duels-value` never sees hidden information.
//!
//! `CLAUDE.md`'s form of this check is "compare two different
//! `Observation::sample_state` draws bit-for-bit (`to_bits()` on floats)", and
//! that is what this file does — for the feature vector, for the forward pass
//! over it, and for the scalar a search would consume.
//!
//! Why it has teeth even though [`duels_value::features`] takes a
//! [`GameState`]: two samples of the same [`Observation`] genuinely *disagree*
//! about which card sits behind which face-down back and about the whole
//! composition of the undealt age decks. A feature that consulted any of that
//! would differ between the two — and
//! [`the_samples_really_do_disagree_about_the_hidden_layout`] proves the
//! samples are distinguishable, so the equality tests cannot pass vacuously.

use duels_core::{engine, Action, GameState, Observation, Player};
use duels_value::{default_net, features, NUM_FEATURES};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Reproducible positions across the draft, all three ages and both movers.
/// Named by `(seed, plies)` rather than checked in, the same way
/// `mcts-eval`'s and `duels-eval`'s fixtures are.
fn observations() -> Vec<(u64, usize, Observation)> {
    let mut out = Vec::new();
    for seed in 0..16u64 {
        for plies in [3usize, 12, 25, 40, 55, 68] {
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xD37E_2A11);
            let mut reached = 0;
            for _ in 0..plies {
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                let a = legal[rng.gen_range(0..legal.len())];
                engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action applies");
                reached += 1;
            }
            if reached == plies {
                out.push((seed, plies, state.observation()));
            }
        }
    }
    assert!(out.len() >= 80, "only {} usable positions", out.len());
    out
}

/// Two samples of one observation, drawn from the same stream so they are
/// independent draws rather than the same draw twice.
fn two_samples(obs: &Observation, salt: u64) -> (GameState, GameState) {
    let mut rng = StdRng::seed_from_u64(salt);
    let a = obs.sample_state(&mut rng);
    let b = obs.sample_state(&mut rng);
    (a, b)
}

#[test]
fn the_feature_vector_is_identical_across_determinizations() {
    for (seed, plies, obs) in observations() {
        let (a, b) = two_samples(&obs, seed * 7919 + plies as u64);
        for me in Player::ALL {
            let fa = features(&a, me);
            let fb = features(&b, me);
            for k in 0..NUM_FEATURES {
                assert_eq!(
                    fa[k].to_bits(),
                    fb[k].to_bits(),
                    "seed {seed} plies {plies} {me:?}: feature {k} leaked hidden information \
                     ({} vs {})",
                    fa[k],
                    fb[k]
                );
            }
        }
    }
}

#[test]
fn the_predicted_distribution_is_identical_across_determinizations() {
    let net = default_net();
    for (seed, plies, obs) in observations() {
        let (a, b) = two_samples(&obs, seed * 104_729 + plies as u64);
        for me in Player::ALL {
            let da = net.evaluate(&a, me);
            let db = net.evaluate(&b, me);
            for k in 0..duels_value::NUM_OUTCOMES {
                assert_eq!(
                    da.0[k].to_bits(),
                    db.0[k].to_bits(),
                    "seed {seed} plies {plies} {me:?}: outcome {k} differs across samples"
                );
            }
            assert_eq!(
                da.win_probability().to_bits(),
                db.win_probability().to_bits(),
                "seed {seed} plies {plies} {me:?}: the scalar differs across samples"
            );
        }
    }
}

/// Many samples, not two, and one long run of them: a feature that read a
/// hidden identity only occasionally — a guild behind one particular back, say
/// — would slip past a two-sample check often enough to matter.
#[test]
fn a_long_run_of_samples_all_agree() {
    for (seed, plies, obs) in observations().into_iter().take(24) {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xBEEF_0000 ^ plies as u64);
        let first = features(&obs.sample_state(&mut rng), Player::One);
        for draw in 1..32 {
            let f = features(&obs.sample_state(&mut rng), Player::One);
            for k in 0..NUM_FEATURES {
                assert_eq!(
                    first[k].to_bits(),
                    f[k].to_bits(),
                    "seed {seed} plies {plies}: draw {draw} disagrees on feature {k}"
                );
            }
        }
    }
}

/// The anti-vacuity check. The samples above have to be genuinely different
/// worlds, or the equality tests prove nothing.
///
/// `GameState`'s hidden fields are private and reachable only inside
/// `duels-core`, so the difference is demonstrated the way an opponent would
/// discover it: play the same forced-choice sequence forward from each sample
/// and watch the public observations diverge as face-down cards are revealed.
#[test]
fn the_samples_really_do_disagree_about_the_hidden_layout() {
    let mut diverged = 0;
    let mut compared = 0;
    for (seed, plies, obs) in observations() {
        // The draft has nothing face-down in the structure to disagree about,
        // and a position at the very end of Age III may have nothing left.
        if obs
            .slots
            .iter()
            .all(|s| *s != duels_core::observation::SlotView::FaceDown)
        {
            continue;
        }
        compared += 1;
        let (mut a, mut b) = two_samples(&obs, seed * 31 + plies as u64 + 1);
        // A policy that depends only on public information, so both worlds are
        // driven identically until the hidden cards themselves show.
        let mut ra = StdRng::seed_from_u64(11);
        let mut rb = StdRng::seed_from_u64(11);
        for _ in 0..8 {
            let la = engine::legal_actions(&a);
            let lb = engine::legal_actions(&b);
            if la.is_empty() || lb.is_empty() {
                break;
            }
            let pick = |l: &[Action]| l[0];
            engine::apply_quiet(&mut a, pick(&la), &mut ra).expect("legal");
            engine::apply_quiet(&mut b, pick(&lb), &mut rb).expect("legal");
        }
        if a.observation() != b.observation() {
            diverged += 1;
        }
    }
    assert!(
        compared >= 40,
        "only {compared} positions had a face-down slot"
    );
    assert!(
        diverged * 2 >= compared,
        "only {diverged} of {compared} sample pairs were distinguishable worlds — the \
         invariance tests above would be vacuous"
    );
}
