//! The three age card structures: slot geometry, covering adjacency, and
//! which slots start face up.
//!
//! Each age lays out exactly [`SLOTS`] = 20 cards. Slots are numbered
//! `0..20` reading the structure top-to-bottom, left-to-right; that numbering
//! is stable and is what [`crate::Action::Build`] and friends refer to.
//!
//! A slot is *accessible* (its card may be taken) only once every slot that
//! covers it is empty. Covering is derived purely from the printed geometry:
//! a card at `(row, col)` is covered by the cards at `(row + 1, col - 1)` and
//! `(row + 1, col + 1)`, which is the standard overlapping-rows layout of the
//! physical game (lower rows lie on top of the row above them, so the bottom
//! row is what you can reach first).
//!
//! # Provenance of the geometry
//!
//! The rulebook prints these as diagrams, not as text, so the `(row, col)`
//! tables below were taken from an independent open-source implementation
//! (`boardzilla/7-wonders-duel`) and cross-checked for the invariants the
//! physical structures are known to satisfy: 20 slots per age, row sizes
//! 2-3-4-5-6 for Age I, 6-5-4-3-2 for Age II, and the "pinched" Age III
//! structure 2-3-4-2-4-3-2 whose two halves are joined only through the
//! 2-slot middle row; exactly 8 face-down slots in every age; and exactly one
//! accessible row at the start of each age. See `docs/rules-spec.md` R-010
//! and the caveats in the M1 PR description.

use std::sync::OnceLock;

/// Number of card slots in every age structure.
pub const SLOTS: usize = 20;

/// `(row, column)` of each slot, per age. Columns are odd/even interleaved so
/// that the two slots covering `(r, c)` are exactly `(r + 1, c - 1)` and
/// `(r + 1, c + 1)`.
const POSITIONS: [[(u8, u8); SLOTS]; 3] = [
    // Age I: a 2-3-4-5-6 pyramid, apex at the top, base (row 6) accessible.
    [
        (2, 5),
        (2, 7),
        (3, 4),
        (3, 6),
        (3, 8),
        (4, 3),
        (4, 5),
        (4, 7),
        (4, 9),
        (5, 2),
        (5, 4),
        (5, 6),
        (5, 8),
        (5, 10),
        (6, 1),
        (6, 3),
        (6, 5),
        (6, 7),
        (6, 9),
        (6, 11),
    ],
    // Age II: the same pyramid inverted, 6-5-4-3-2, the 2-slot row (row 6)
    // accessible.
    [
        (2, 1),
        (2, 3),
        (2, 5),
        (2, 7),
        (2, 9),
        (2, 11),
        (3, 2),
        (3, 4),
        (3, 6),
        (3, 8),
        (3, 10),
        (4, 3),
        (4, 5),
        (4, 7),
        (4, 9),
        (5, 4),
        (5, 6),
        (5, 8),
        (6, 5),
        (6, 7),
    ],
    // Age III: 2-3-4-2-4-3-2. The 2-slot row 4 is the bottleneck: the whole
    // upper half (rows 1-3) is unreachable until both of its cards are gone.
    [
        (1, 5),
        (1, 7),
        (2, 4),
        (2, 6),
        (2, 8),
        (3, 3),
        (3, 5),
        (3, 7),
        (3, 9),
        (4, 4),
        (4, 8),
        (5, 3),
        (5, 5),
        (5, 7),
        (5, 9),
        (6, 4),
        (6, 6),
        (6, 8),
        (7, 5),
        (7, 7),
    ],
];

/// Rows whose cards are dealt face up, per age. Every other row is dealt face
/// down and is turned face up when it becomes uncovered.
const FACE_UP_ROWS: [&[u8]; 3] = [&[2, 4, 6], &[2, 4, 6], &[1, 3, 5, 7]];

/// Precomputed geometry for one age structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgeLayout {
    /// `(row, column)` of each slot, for renderers.
    pub positions: [(u8, u8); SLOTS],
    /// `covered_by[i]` is a bitmask of the slots that cover slot `i`. Slot
    /// `i` is accessible exactly when none of those slots still holds a card.
    pub covered_by: [u32; SLOTS],
    /// `covers[j]` is a bitmask of the slots that slot `j` covers — the
    /// inverse of `covered_by`, so that emptying slot `j` only has to check
    /// those slots for becoming accessible.
    pub covers: [u32; SLOTS],
    /// Bitmask of the slots dealt face up.
    pub face_up: u32,
}

static LAYOUTS: OnceLock<[AgeLayout; 3]> = OnceLock::new();

/// Geometry for `age` (1, 2 or 3).
///
/// # Panics
///
/// Panics if `age` is not in `1..=3`.
pub fn layout(age: u8) -> &'static AgeLayout {
    assert!((1..=3).contains(&age), "age must be 1..=3, got {age}");
    &layouts()[(age - 1) as usize]
}

fn layouts() -> &'static [AgeLayout; 3] {
    LAYOUTS.get_or_init(|| {
        std::array::from_fn(|a| {
            let positions = POSITIONS[a];
            let mut covered_by = [0u32; SLOTS];
            for (i, &(ri, ci)) in positions.iter().enumerate() {
                for (j, &(rj, cj)) in positions.iter().enumerate() {
                    if rj == ri + 1 && (cj + 1 == ci || cj == ci + 1) {
                        covered_by[i] |= 1 << j;
                    }
                }
            }
            let mut covers = [0u32; SLOTS];
            for (i, &mask) in covered_by.iter().enumerate() {
                for (j, c) in covers.iter_mut().enumerate() {
                    if mask & (1 << j) != 0 {
                        *c |= 1 << i;
                    }
                }
            }
            let mut face_up = 0u32;
            for (i, &(r, _)) in positions.iter().enumerate() {
                if FACE_UP_ROWS[a].contains(&r) {
                    face_up |= 1 << i;
                }
            }
            AgeLayout {
                positions,
                covered_by,
                covers,
                face_up,
            }
        })
    })
}

/// Bitmask of every slot: `0b1111_1111_1111_1111_1111`.
pub const ALL_SLOTS: u32 = (1u32 << SLOTS) - 1;

/// The slots that are accessible given which slots still hold a card.
///
/// `occupied` is a bitmask over slot indices.
#[inline]
pub fn accessible(age: u8, occupied: u32) -> u32 {
    let l = layout(age);
    let mut out = 0u32;
    let mut rest = occupied;
    while rest != 0 {
        let i = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        if l.covered_by[i] & occupied == 0 {
            out |= 1 << i;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn row_sizes(age: u8) -> Vec<usize> {
        let mut by_row: BTreeMap<u8, usize> = BTreeMap::new();
        for &(r, _) in &layout(age).positions {
            *by_row.entry(r).or_default() += 1;
        }
        by_row.into_values().collect()
    }

    #[test]
    fn row_sizes_match_the_printed_structures() {
        assert_eq!(row_sizes(1), vec![2, 3, 4, 5, 6]);
        assert_eq!(row_sizes(2), vec![6, 5, 4, 3, 2]);
        assert_eq!(row_sizes(3), vec![2, 3, 4, 2, 4, 3, 2]);
    }

    #[test]
    fn every_age_has_twenty_slots_and_eight_face_down() {
        for age in 1..=3 {
            let l = layout(age);
            assert_eq!(l.positions.len(), SLOTS);
            assert_eq!(
                (ALL_SLOTS & !l.face_up).count_ones(),
                8,
                "age {age} should deal 8 cards face down"
            );
        }
    }

    #[test]
    fn slot_positions_are_unique_per_age() {
        for age in 1..=3 {
            let mut seen = std::collections::BTreeSet::new();
            for &p in &layout(age).positions {
                assert!(seen.insert(p), "duplicate position {p:?} in age {age}");
            }
        }
    }

    #[test]
    fn initially_accessible_slots_are_the_bottom_row_and_are_face_up() {
        // Age I: the 6-card base row (slots 14..20).
        assert_eq!(accessible(1, ALL_SLOTS), 0b1111_1100_0000_0000_0000);
        // Age II: the 2-card bottom row (slots 18, 19).
        assert_eq!(accessible(2, ALL_SLOTS), 0b1100_0000_0000_0000_0000);
        // Age III: the 2-card bottom row (slots 18, 19).
        assert_eq!(accessible(3, ALL_SLOTS), 0b1100_0000_0000_0000_0000);

        for age in 1..=3 {
            let l = layout(age);
            let acc = accessible(age, ALL_SLOTS);
            assert_eq!(
                acc & l.face_up,
                acc,
                "age {age}: accessible slots must be face up"
            );
        }
    }

    #[test]
    fn a_face_down_slot_is_never_accessible_before_it_is_uncovered() {
        // Every face-down slot is covered by at least one other slot, so it
        // cannot be taken while still face down.
        for age in 1..=3 {
            let l = layout(age);
            for i in 0..SLOTS {
                if l.face_up & (1 << i) == 0 {
                    assert_ne!(
                        l.covered_by[i], 0,
                        "age {age} slot {i} is face down but uncovered"
                    );
                }
            }
        }
    }

    #[test]
    fn emptying_the_structure_makes_every_slot_accessible_exactly_once() {
        // Greedily take accessible slots until the structure is empty; every
        // slot must be reachable, i.e. the covering graph is acyclic and
        // rooted at the accessible row.
        for age in 1..=3 {
            let mut occupied = ALL_SLOTS;
            let mut taken = 0;
            while occupied != 0 {
                let acc = accessible(age, occupied);
                assert_ne!(acc, 0, "age {age} deadlocked with {occupied:#b} left");
                let i = acc.trailing_zeros();
                occupied &= !(1 << i);
                taken += 1;
            }
            assert_eq!(taken, SLOTS, "age {age}");
        }
    }

    #[test]
    fn covers_is_the_inverse_of_covered_by() {
        for age in 1..=3 {
            let l = layout(age);
            for i in 0..SLOTS {
                for j in 0..SLOTS {
                    assert_eq!(
                        l.covered_by[i] & (1 << j) != 0,
                        l.covers[j] & (1 << i) != 0,
                        "age {age}: covered_by[{i}] and covers[{j}] disagree"
                    );
                }
            }
        }
    }

    #[test]
    fn emptying_one_slot_can_uncover_at_most_two() {
        // The chance API models the reveal from a single card removal as at
        // most two cards; that must hold for every slot of every age.
        for age in 1..=3 {
            let l = layout(age);
            for j in 0..SLOTS {
                assert!(
                    l.covers[j].count_ones() <= 2,
                    "age {age} slot {j} covers {} slots",
                    l.covers[j].count_ones()
                );
            }
        }
    }

    #[test]
    fn age_three_upper_half_is_gated_by_the_middle_row() {
        // Slots 9 and 10 are the two middle-row cards. While either is still
        // present, at least one of the four row-3 slots (5..9) stays covered.
        let l = layout(3);
        assert_eq!(l.positions[9], (4, 4));
        assert_eq!(l.positions[10], (4, 8));
        // Row-3 slots 5 and 6 are covered only by slot 9; slots 7 and 8 only
        // by slot 10.
        assert_eq!(l.covered_by[5], 1 << 9);
        assert_eq!(l.covered_by[6], 1 << 9);
        assert_eq!(l.covered_by[7], 1 << 10);
        assert_eq!(l.covered_by[8], 1 << 10);
        // ...and the middle row is itself covered by two row-5 slots each.
        assert_eq!(l.covered_by[9], (1 << 11) | (1 << 12));
        assert_eq!(l.covered_by[10], (1 << 13) | (1 << 14));
    }
}
