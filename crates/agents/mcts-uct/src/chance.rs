//! Sampling the engine's chance distribution, cheaply.
//!
//! [`duels_core::engine::chance_outcomes`] is the authoritative distribution
//! over what an action can reveal, but it *enumerates*: a build that uncovers
//! two face-down slots has `|pool| * (|pool| - 1)` outcomes (several hundred
//! in Age III), and a Great Library build multiplies that by `C(aside, 3)`.
//! Enumerating — let alone storing one such vector per chance node —
//! would dominate the search cost and blow up memory.
//!
//! MCTS never needs the whole distribution: at a chance node it only needs to
//! *draw* from it. [`sample`] does that in O(1), returning the drawn
//! [`Outcome`] together with its exact probability (the probability is what
//! progressive widening needs in order to re-select among already-expanded
//! outcomes without bias).
//!
//! The distribution is re-derived here rather than reused, so it is pinned
//! down by statistical tests against `engine::chance_outcomes` below. The
//! derivation: the `slots` face-down slots of the current age hold `ng` guild
//! cards and `nn = slots - ng` non-guild cards (`ng` is public — exactly
//! three guilds are dealt into Age III). Which slots are the guild slots is a
//! uniform subset, and the cards are a uniform injection from
//! `unseen_guilds` (size `ug`) and `unseen_plain` (size `un`) — both pools
//! larger than their slot counts, because three cards of the age went back in
//! the box unseen and stay candidates all age. So for one slot the card is a
//! guild with probability `ng / slots` and then uniform over the `ug`
//! candidates; for two slots the types are drawn without replacement and the
//! cards are distinct uniform draws from their pools.

use duels_core::data::{CardId, TokenId};
use duels_core::engine::{self, Outcome, RevealSlots};
use duels_core::layout;
use duels_core::{Action, GameState};
use rand::rngs::StdRng;
use rand::Rng;

/// The face-down slots `action` would turn face up, ascending, at most two
/// (a card covers at most two others).
///
/// Mirrors the engine's private `slots_revealed_by` using public accessors.
pub(crate) fn revealed_slots(state: &GameState, action: Action) -> [Option<u8>; 2] {
    let slot = match action {
        Action::Build { slot } | Action::Discard { slot } | Action::BuildWonder { slot, .. } => {
            slot
        }
        _ => return [None, None],
    };
    let l = layout::layout(state.age());
    let occ = state.occupied_slots() & !(1u32 << slot);
    let revealed = state.revealed_slots();
    let mut out = [None, None];
    let mut n = 0;
    let mut rest = l.covers[slot as usize] & occ;
    while rest != 0 {
        let i = rest.trailing_zeros() as u8;
        rest &= rest - 1;
        if l.covered_by[i as usize] & occ == 0 && revealed & (1u32 << i) == 0 {
            out[n] = Some(i);
            n += 1;
            if n == 2 {
                break;
            }
        }
    }
    out
}

/// The set-aside progress tokens The Great Library would draw from, if
/// `action` builds it and there are at least three left.
fn library_pool(state: &GameState, action: Action) -> Option<Vec<TokenId>> {
    match action {
        Action::BuildWonder { wonder, .. } if wonder.def().choose_progress_token => {
            let aside: Vec<_> = state.set_aside_tokens().collect();
            (aside.len() >= 3).then_some(aside)
        }
        _ => None,
    }
}

/// Whether `action` in `state` resolves any randomness, i.e. whether the
/// search must insert a chance node between this action and the decision node
/// it leads to.
///
/// True whenever the action uncovers a face-down slot or builds The Great
/// Library with at least three tokens set aside. `false` means
/// `engine::chance_outcomes` reports only the single trivial outcome, so the
/// action can be applied straight through.
pub(crate) fn resolves_randomness(state: &GameState, action: Action) -> bool {
    revealed_slots(state, action)[0].is_some() || library_pool(state, action).is_some()
}

/// Draw one `(outcome, probability)` pair from the true chance distribution of
/// `action` in `state`, using public information only.
///
/// `probability` is the probability of drawing exactly this outcome, so the
/// probabilities over the (never enumerated) support sum to 1.
pub(crate) fn sample(state: &GameState, action: Action, rng: &mut StdRng) -> (Outcome, f64) {
    let (reveals, p_reveal) = sample_reveals(state, action, rng);
    let (library_tokens, p_library) = sample_library(state, action, rng);
    (
        Outcome {
            reveals,
            library_tokens,
        },
        p_reveal * p_library,
    )
}

fn sample_library(
    state: &GameState,
    action: Action,
    rng: &mut StdRng,
) -> (Option<[TokenId; 3]>, f64) {
    let Some(aside) = library_pool(state, action) else {
        return (None, 1.0);
    };
    let n = aside.len();
    // Draw three distinct indices, then keep them ascending so the outcome is
    // byte-identical to the one `chance_outcomes` enumerates for this triple.
    let mut idx = [0usize; 3];
    let mut picked = 0;
    while picked < 3 {
        let c = rng.gen_range(0..n);
        if idx[..picked].contains(&c) {
            continue;
        }
        idx[picked] = c;
        picked += 1;
    }
    idx.sort_unstable();
    let triples = (n * (n - 1) * (n - 2) / 6) as f64;
    (
        Some([aside[idx[0]], aside[idx[1]], aside[idx[2]]]),
        1.0 / triples,
    )
}

/// The candidate cards behind the current age's face-down slots, with the
/// per-kind quotas that make an assignment publicly consistent.
struct Pools {
    guild: Vec<CardId>,
    plain: Vec<CardId>,
    guilds_left: u32,
    plains_left: u32,
}

impl Pools {
    /// Draw the card behind one specific face-down slot, given that `left`
    /// face-down slots (including this one) are still unassigned. Returns the
    /// card and the probability of having drawn it.
    fn draw(&mut self, left: u32, rng: &mut StdRng) -> Option<(CardId, f64)> {
        if left == 0 {
            return None;
        }
        let take_guild = if self.guilds_left == 0 {
            false
        } else if self.plains_left == 0 {
            true
        } else {
            rng.gen_bool(f64::from(self.guilds_left) / f64::from(left))
        };
        let (pool, quota) = if take_guild {
            (&mut self.guild, &mut self.guilds_left)
        } else {
            (&mut self.plain, &mut self.plains_left)
        };
        if pool.is_empty() {
            return None;
        }
        let p = f64::from(*quota) / f64::from(left) / pool.len() as f64;
        *quota -= 1;
        let card = pool.swap_remove(rng.gen_range(0..pool.len()));
        Some((card, p))
    }
}

fn sample_reveals(state: &GameState, action: Action, rng: &mut StdRng) -> (RevealSlots, f64) {
    let slots = revealed_slots(state, action);
    let Some(s0) = slots[0] else {
        return ([None, None], 1.0);
    };
    let info = engine::hidden_info(state);
    let total = info.hidden_slots.count_ones();
    let guilds_left = info.hidden_guild_count;
    let mut pools = Pools {
        guild: info.unseen_guilds,
        plain: info.unseen_plain,
        guilds_left,
        plains_left: total.saturating_sub(guilds_left),
    };

    // Unreachable under the real rules (a face-down slot always has
    // candidates); falling back to the trivial outcome keeps the search
    // correct-by-the-engine rather than panicking, because a trivial forced
    // outcome makes `apply_with_outcome` reveal whatever the state's own
    // determinized layout holds.
    let Some((c0, p0)) = pools.draw(total, rng) else {
        return ([None, None], 1.0);
    };
    let mut out = [Some((s0, c0)), None];
    let mut p = p0;
    if let Some(s1) = slots[1] {
        match pools.draw(total - 1, rng) {
            Some((c1, p1)) => {
                out[1] = Some((s1, c1));
                p *= p1;
            }
            None => return ([None, None], 1.0),
        }
    }
    (out, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine::legal_actions;
    use rand::SeedableRng;
    use std::collections::HashMap;

    /// `Outcome` is not `Hash`, so key it by its parts for the tally.
    type Key = (u32, u32, u32);

    fn key(o: &Outcome) -> Key {
        let one = |r: Option<(u8, CardId)>| match r {
            None => 0,
            Some((s, c)) => 1 + u32::from(s) * 128 + c.index() as u32,
        };
        let lib = match o.library_tokens {
            None => 0,
            Some([a, b, c]) => 1 + (a.index() * 400 + b.index() * 20 + c.index()) as u32,
        };
        (one(o.reveals[0]), one(o.reveals[1]), lib)
    }

    /// Walk a random game until a state is reached where some action uncovers
    /// exactly `want` face-down slots, so the sampler can be exercised on a
    /// real position rather than a synthetic one.
    fn find_reveal_position(seed: u64, want: usize) -> Option<(GameState, Action)> {
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xBEEF);
        for _ in 0..400 {
            let legal = legal_actions(&st);
            if legal.is_empty() {
                return None;
            }
            for &a in &legal {
                if revealed_slots(&st, a).iter().flatten().count() == want {
                    return Some((st, a));
                }
            }
            let a = legal[rng.gen_range(0..legal.len())];
            engine::apply_unchecked(&mut st, a, &mut rng);
        }
        None
    }

    /// The crux of the whole agent: the sampler must reproduce the engine's
    /// distribution, not merely something plausible.
    fn assert_matches_engine(state: &GameState, action: Action, draws: usize, tol: f64) {
        let truth: HashMap<Key, f64> = engine::chance_outcomes(state, action)
            .into_iter()
            .map(|(o, p)| (key(&o), p))
            .collect();
        assert!(truth.len() > 1, "expected a real chance node");
        let sum: f64 = truth.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "engine probabilities sum to {sum}"
        );

        let mut rng = StdRng::seed_from_u64(0x5EED);
        let mut counts: HashMap<Key, u32> = HashMap::new();
        for _ in 0..draws {
            let (o, p) = sample(state, action, &mut rng);
            let k = key(&o);
            let expected = *truth
                .get(&k)
                .unwrap_or_else(|| panic!("sampled an impossible outcome: {o:?}"));
            assert!(
                (p - expected).abs() < 1e-9,
                "reported probability {p} != engine's {expected} for {o:?}"
            );
            *counts.entry(k).or_default() += 1;
        }
        for (k, p) in &truth {
            let got = f64::from(counts.get(k).copied().unwrap_or(0)) / draws as f64;
            assert!(
                (got - p).abs() < tol,
                "outcome {k:?}: sampled {got:.4}, engine says {p:.4}"
            );
        }
    }

    #[test]
    fn sampled_reveals_match_the_engine_distribution_for_one_slot() {
        let (st, a) = find_reveal_position(3, 1).expect("a one-slot reveal exists");
        assert_matches_engine(&st, a, 60_000, 0.01);
    }

    #[test]
    fn sampled_reveals_match_the_engine_distribution_for_two_slots() {
        let (st, a) = find_reveal_position(11, 2).expect("a two-slot reveal exists");
        // Hundreds of outcomes, each individually rare, so the tolerance is
        // tight in absolute terms while still being many standard errors.
        assert_matches_engine(&st, a, 60_000, 0.004);
    }

    #[test]
    fn a_sampled_outcome_can_always_be_forced_onto_the_state() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut chance_nodes = 0;
        for seed in 0..8u64 {
            let mut st = engine::new_game(seed);
            for _ in 0..400 {
                let legal = legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                let a = legal[rng.gen_range(0..legal.len())];
                if resolves_randomness(&st, a) {
                    let (o, p) = sample(&st, a, &mut rng);
                    assert!(p > 0.0 && p <= 1.0, "probability out of range: {p}");
                    let mut next = st;
                    engine::apply_with_outcome(&mut next, a, &o)
                        .expect("a sampled outcome is always forceable");
                    chance_nodes += 1;
                }
                engine::apply_unchecked(&mut st, a, &mut rng);
            }
        }
        assert!(chance_nodes > 100, "expected plenty of chance actions");
    }

    #[test]
    fn actions_that_resolve_nothing_are_not_chance_actions() {
        let st = engine::new_game(5);
        // The wonder draft reveals nothing: the second group of four is fixed
        // by the root determinization, not by a per-action reveal.
        for a in legal_actions(&st) {
            assert!(!resolves_randomness(&st, a), "{a:?}");
            assert_eq!(engine::chance_outcomes(&st, a).len(), 1);
        }
    }

    #[test]
    fn resolves_randomness_agrees_with_the_engine_on_a_random_walk() {
        let mut rng = StdRng::seed_from_u64(42);
        for seed in 0..5u64 {
            let mut st = engine::new_game(seed);
            for _ in 0..400 {
                let legal = legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                for &a in &legal {
                    let outcomes = engine::chance_outcomes(&st, a);
                    let nontrivial = outcomes.len() > 1 || !outcomes[0].0.is_trivial();
                    assert_eq!(
                        resolves_randomness(&st, a),
                        nontrivial,
                        "disagreed about {a:?}"
                    );
                }
                let a = legal[rng.gen_range(0..legal.len())];
                engine::apply_unchecked(&mut st, a, &mut rng);
            }
        }
    }
}
