//! A power-of-two indexed transposition table, plus the state hash it is
//! keyed on.
//!
//! # Why hashing the *public* state is enough
//!
//! [`GameState`] does not derive `Hash`, and the fields that make two
//! otherwise-identical states differ — the identity of the face-down cards in
//! the current age's structure, the composition of future age decks, what went
//! back in the box — are private to `duels-core` and unreachable from an agent
//! crate. That turns out not to matter, because the search never *looks* at
//! the hidden layout of the current age: chance nodes are expanded from
//! [`duels_core::engine::chance_outcomes`], whose probabilities are computed
//! from public knowledge only, and applied with
//! [`duels_core::engine::apply_with_outcome`], which rewrites the hidden
//! layout to match the outcome the search forced. Two states that agree on
//! everything public therefore have the same search value.
//!
//! The one exception is the layout of the *future* age decks, which the engine
//! deals from the state itself rather than through the chance API. Those are
//! fixed for the whole search (they come from the single determinized root, see
//! the crate docs), so within one `choose` call they are a constant and need
//! not enter the key. Across `choose` calls they change, which is exactly why
//! every entry carries a generation stamp that is bumped per root search.

use duels_core::data::{TokenId, WonderId};
use duels_core::state::{Pending, Phase};
use duels_core::{Action, GameState, Player};

/// What a stored value says about the true value of a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// The stored value is the position's value.
    Exact,
    /// The true value is at least the stored value (a beta cutoff).
    Lower,
    /// The true value is at most the stored value (every move failed low).
    Upper,
}

/// One transposition-table slot.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    /// Full 64-bit key, checked on probe so an index collision is not a hit.
    pub key: u64,
    /// The value, normalised for mate distance (see `search::to_tt`).
    pub value: f64,
    /// The move that produced `value`, for move ordering.
    pub best: Option<Action>,
    /// Remaining search depth the value was produced with.
    pub depth: u8,
    /// How `value` relates to the true value.
    pub bound: Bound,
    /// Which root search wrote this entry.
    pub generation: u32,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            key: 0,
            value: 0.0,
            best: None,
            depth: 0,
            bound: Bound::Exact,
            // Generation 0 is never used by a live search, so a freshly
            // allocated table reads as entirely empty.
            generation: 0,
        }
    }
}

/// A fixed-size, power-of-two indexed transposition table.
#[derive(Debug)]
pub struct Table {
    entries: Vec<Entry>,
    mask: usize,
    generation: u32,
    probes: u64,
    hits: u64,
}

impl Table {
    /// A table with `2^bits` slots.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is 0 or above 26 (a 26-bit table is already ~2 GiB).
    pub fn with_bits(bits: u32) -> Self {
        assert!((1..=26).contains(&bits), "tt bits must be in 1..=26");
        let len = 1usize << bits;
        Self {
            entries: vec![Entry::default(); len],
            mask: len - 1,
            generation: 1,
            probes: 0,
            hits: 0,
        }
    }

    /// Number of slots.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Always false; a table always has at least one slot.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Invalidate every entry by moving to a new generation.
    ///
    /// Called once per root search: a new root means a new determinization,
    /// and values from the previous one are no longer sound.
    pub fn new_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped past the "never written" sentinel; clear for real.
            self.entries.fill(Entry::default());
            self.generation = 1;
        }
    }

    /// The current generation stamp.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Probe/hit counters since construction, for reporting.
    pub fn stats(&self) -> (u64, u64) {
        (self.probes, self.hits)
    }

    /// Look up `key`, if a live entry for it is stored.
    pub fn probe(&mut self, key: u64) -> Option<Entry> {
        self.probes += 1;
        let e = self.entries[(key as usize) & self.mask];
        if e.key == key && e.generation == self.generation {
            self.hits += 1;
            Some(e)
        } else {
            None
        }
    }

    /// Store an entry, preferring to keep deeper searches from this
    /// generation.
    pub fn store(&mut self, key: u64, value: f64, depth: u8, bound: Bound, best: Option<Action>) {
        let slot = (key as usize) & self.mask;
        let cur = self.entries[slot];
        let stale = cur.generation != self.generation;
        if stale || cur.key == key || cur.depth <= depth {
            self.entries[slot] = Entry {
                key,
                value,
                best,
                depth,
                bound,
                generation: self.generation,
            };
        }
    }
}

#[inline]
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Accumulator for the state hash: one multiply-based mix per field.
#[derive(Debug, Clone, Copy)]
struct Hasher(u64);

impl Hasher {
    #[inline]
    fn new() -> Self {
        Hasher(0x243F_6A88_85A3_08D3)
    }

    #[inline]
    fn add(&mut self, v: u64) {
        self.0 = self.0.rotate_left(7) ^ splitmix(v);
    }

    #[inline]
    fn add_u128(&mut self, v: u128) {
        self.add(v as u64);
        self.add((v >> 64) as u64);
    }

    #[inline]
    fn finish(self) -> u64 {
        splitmix(self.0)
    }
}

fn phase_tag(p: Phase) -> u64 {
    match p {
        Phase::WonderDraft => 1,
        Phase::ChooseFirstPlayer => 2,
        Phase::Turn => 3,
        Phase::GameOver => 4,
    }
}

fn pending_tag(p: Option<Pending>) -> u64 {
    match p {
        None => 0,
        Some(Pending::ProgressToken) => 1,
        Some(Pending::GreatLibraryToken { tokens }) => {
            let mut m = 0u64;
            for t in tokens {
                m |= 1u64 << t.index();
            }
            2 | (m << 8)
        }
        Some(Pending::Destroy { card_type }) => 3 | ((card_type.index() as u64) << 8),
        Some(Pending::MausoleumBuild) => 4,
    }
}

fn wonder_mask(it: impl Iterator<Item = WonderId>) -> u64 {
    it.fold(0u64, |m, w| m | (1u64 << w.index()))
}

fn token_mask(it: impl Iterator<Item = TokenId>) -> u64 {
    it.fold(0u64, |m, t| m | (1u64 << t.index()))
}

/// A 64-bit key over everything publicly observable about `state`.
///
/// Deliberately *not* a function of the hidden card layout — see the module
/// docs for why that is both unavoidable from an agent crate and sound for
/// this search.
pub fn state_key(state: &GameState) -> u64 {
    let mut h = Hasher::new();
    h.add(phase_tag(state.phase()));
    h.add(pending_tag(state.pending()));
    h.add(u64::from(state.age()));
    h.add(state.current_player().index() as u64);
    h.add(state.last_card_taker().index() as u64);
    h.add(u64::from(state.extra_turn()));
    h.add(u64::from(state.turn()));
    h.add(state.conflict() as i64 as u64);
    h.add(u64::from(state.occupied_slots()));
    h.add(u64::from(state.revealed_slots()));
    h.add_u128(state.discard_mask());
    h.add_u128(state.wonder_fodder_mask());
    h.add(u64::from(state.board_tokens_mask()));
    h.add(u64::from(state.set_aside_tokens_mask()));

    // The identity of the face-up cards in the structure. These are public,
    // and the search rewrites them at chance nodes, so they must be keyed.
    let mut faces = 0u64;
    for slot in 0..20u8 {
        if let Some(card) = state.face_up_card(slot) {
            faces = faces
                .rotate_left(5)
                .wrapping_add(u64::from(slot) << 8 | card.index() as u64);
        }
    }
    h.add(faces);

    let mut loot = 0u64;
    for p in Player::ALL {
        for i in 0..2 {
            loot = (loot << 1) | u64::from(state.loot_available(p, i));
        }
    }
    h.add(loot);

    for p in Player::ALL {
        let ps = state.player(p);
        h.add_u128(ps.built_mask());
        h.add(u64::from(ps.coins()));
        h.add(u64::from(ps.shields()));
        h.add(wonder_mask(ps.wonders()));
        h.add(wonder_mask(ps.wonders_built()));
        h.add(token_mask(ps.tokens()));
        h.add(
            ps.pairs_awarded()
                .fold(0u64, |m, s| m | (1u64 << s.index())),
        );
    }

    // The wonder draft is over after eight actions and its offered group is
    // not derivable from anything above, so key it only while it lasts (it
    // allocates).
    if state.phase() == Phase::WonderDraft {
        h.add(u64::from(state.draft_step()));
        h.add(state.draft_first().index() as u64);
        h.add(wonder_mask(state.offered_wonders().into_iter()));
    }

    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use duels_core::testing::StateBuilder;

    fn some_action() -> Action {
        Action::Build { slot: 7 }
    }

    #[test]
    fn an_entry_round_trips_through_the_table() {
        let mut tt = Table::with_bits(10);
        let key = 0xDEAD_BEEF_CAFE_F00D;
        tt.store(key, -12.5, 4, Bound::Lower, Some(some_action()));
        let got = tt.probe(key).expect("just stored");
        assert_eq!(got.key, key);
        assert_eq!(got.value, -12.5);
        assert_eq!(got.depth, 4);
        assert_eq!(got.bound, Bound::Lower);
        assert_eq!(got.best, Some(some_action()));
        assert_eq!(got.generation, tt.generation());
    }

    #[test]
    fn probing_a_key_that_was_never_stored_misses() {
        let mut tt = Table::with_bits(10);
        assert!(tt.probe(0).is_none(), "a fresh table has no live entries");
        assert!(tt.probe(12345).is_none());
        tt.store(1, 1.0, 1, Bound::Exact, None);
        // Same index (the table has 1024 slots), different key.
        assert!(tt.probe(1 + 1024).is_none());
        assert!(tt.probe(1).is_some());
    }

    #[test]
    fn a_new_generation_invalidates_every_entry() {
        let mut tt = Table::with_bits(8);
        tt.store(42, 3.0, 9, Bound::Exact, None);
        assert!(tt.probe(42).is_some());
        tt.new_generation();
        assert!(tt.probe(42).is_none());
        // ... and the slot is reusable.
        tt.store(42, 4.0, 1, Bound::Upper, None);
        assert_eq!(tt.probe(42).unwrap().value, 4.0);
    }

    #[test]
    fn a_shallower_entry_does_not_evict_a_deeper_one_from_the_same_search() {
        let mut tt = Table::with_bits(8);
        tt.store(7, 1.0, 8, Bound::Exact, None);
        tt.store(7 + 256, 2.0, 2, Bound::Exact, None); // same slot, shallower
        assert_eq!(tt.probe(7).unwrap().value, 1.0);
        // A deeper one does evict.
        tt.store(7 + 256, 2.0, 9, Bound::Exact, None);
        assert!(tt.probe(7).is_none());
        assert_eq!(tt.probe(7 + 256).unwrap().value, 2.0);
    }

    #[test]
    fn probe_and_hit_counters_track_lookups() {
        let mut tt = Table::with_bits(8);
        tt.store(5, 1.0, 1, Bound::Exact, None);
        tt.probe(5);
        tt.probe(6);
        let (probes, hits) = tt.stats();
        assert_eq!((probes, hits), (2, 1));
    }

    #[test]
    fn the_key_is_stable_and_sensitive_to_public_fields() {
        let base = StateBuilder::new()
            .built(Player::One, &["palace"])
            .coins(Player::One, 5)
            .conflict(2)
            .build();
        assert_eq!(state_key(&base), state_key(&base));

        let variants = [
            StateBuilder::new()
                .built(Player::One, &["palace"])
                .coins(Player::One, 6)
                .conflict(2)
                .build(),
            StateBuilder::new()
                .built(Player::One, &["palace"])
                .coins(Player::One, 5)
                .conflict(3)
                .build(),
            StateBuilder::new()
                .built(Player::Two, &["palace"])
                .coins(Player::One, 5)
                .conflict(2)
                .build(),
            StateBuilder::new()
                .built(Player::One, &["palace"])
                .coins(Player::One, 5)
                .conflict(2)
                .current(Player::Two)
                .build(),
        ];
        for (i, v) in variants.iter().enumerate() {
            assert_ne!(state_key(&base), state_key(v), "variant {i} collides");
        }
    }

    #[test]
    fn distinct_positions_from_real_games_get_distinct_keys() {
        // Not a proof of collision-freedom, but a 64-bit key over a few
        // thousand real positions should not collide at all. Two states that
        // differ *only* in hidden information share a key by design, so the
        // comparison is between public views.
        use rand::{rngs::StdRng, SeedableRng};
        let mut keys: std::collections::HashMap<u64, duels_core::Observation> =
            std::collections::HashMap::new();
        let mut collisions = 0;
        for seed in 0..40u64 {
            let mut st = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed);
            loop {
                let legal = engine::legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                let key = state_key(&st);
                let obs = st.observation();
                if let Some(prev) = keys.insert(key, obs.clone()) {
                    if prev != obs {
                        collisions += 1;
                    }
                }
                let a = legal[(seed as usize + st.turn() as usize) % legal.len()];
                engine::apply_quiet(&mut st, a, &mut rng).unwrap();
            }
        }
        assert!(
            keys.len() > 1000,
            "expected a decent sample, got {}",
            keys.len()
        );
        assert_eq!(collisions, 0, "state_key collided on real positions");
    }
}
