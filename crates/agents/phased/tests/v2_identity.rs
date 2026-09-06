//! The bit-identity guard for this crate's **third** round of work.
//!
//! Round three adds five things: the three terminal rails, a differenced menu
//! shield price, a horizon-based military smoothing width, a
//! production-lock-in multiplier on the economy terms, and a different
//! `military_band` default. Every one of them is a [`Config`] option, and
//! [`Config::v2`] sets all five back to what the round-two default did.
//!
//! `tests/legacy_identity.rs` proves round two's `Config::v1()` against a
//! verbatim copy of the round-one evaluation. Copying the round-two evaluation
//! the same way would mean copying `menu::TakeValue` — whose fields are
//! private, and whose pricing context is half the round — into a test file, so
//! this round uses the other sharp instrument the project accepts: a **golden
//! digest**, recorded by running this exact harness against the round-two code
//! before a line of it was touched.
//!
//! The harness drives whole seeded games with a deterministic
//! first-argmax policy (no RNG anywhere, so nothing can drift for a reason
//! other than the arithmetic), and folds into one 64-bit FNV-1a digest, for
//! every decision of every game: the raw `f64` bits of *every* candidate's
//! [`expected_value`], and the chosen action. A difference too small to change
//! a single move still fails.

use duels_agent_phased::{expected_value, Config, Root};
use duels_core::{engine, Action, GameState};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The digest recorded from the round-two code, before this round's changes.
///
/// If this constant ever has to move, the change is not an "off by default"
/// one and the round's whole premise needs re-examining.
const V2_DIGEST: u64 = 0x8e3c_e2d5_484d_b5c1;

/// How many seeded self-play games the digest covers.
const GAMES: u64 = 6;

struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    fn action(&mut self, a: Action) {
        self.write(format!("{a:?}").as_bytes());
    }
}

/// Drive `GAMES` whole games under `config` with a deterministic first-argmax
/// policy, folding every candidate score and every chosen move into a digest.
fn digest(config: Config) -> u64 {
    let mut h = Fnv::new();
    for seed in 0..GAMES {
        let mut st: GameState = engine::new_game(seed);
        // The engine still needs *an* RNG to deal the next age; it is seeded
        // per game and never consulted by the evaluation.
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FF_EE00);
        for _ in 0..400 {
            if st.is_over() {
                break;
            }
            let legal = engine::legal_actions(&st);
            if legal.is_empty() {
                break;
            }
            let me = st.current_player();
            h.u64(u64::from(st.turn()));
            h.u64(legal.len() as u64);
            let chosen = if legal.len() == 1 {
                // The agent's own shortcut: no `Root`, no scoring.
                legal[0]
            } else {
                let root = Root::new(&st, me, config);
                let mut best = (legal[0], f64::NEG_INFINITY);
                for &action in &legal {
                    let v = expected_value(&st, action, me, &root);
                    h.f64(v);
                    if v > best.1 {
                        best = (action, v);
                    }
                }
                best.0
            };
            h.action(chosen);
            engine::apply(&mut st, chosen, &mut rng).expect("the chosen action was legal");
        }
        match st.result() {
            None => h.write(b"unfinished"),
            Some(duels_core::GameResult::Draw) => h.write(b"draw"),
            Some(duels_core::GameResult::Win { winner, kind }) => {
                h.write(format!("{winner:?}/{kind:?}").as_bytes());
            }
        }
    }
    h.0
}

/// The whole point: [`Config::v2`] must reproduce the round-two default's
/// arithmetic **exactly**, not approximately.
#[test]
fn config_v2_reproduces_the_round_two_default_bit_for_bit() {
    let got = digest(Config::v2());
    assert_eq!(
        got, V2_DIGEST,
        "Config::v2() no longer reproduces the round-two agent: got {got:#018x}, \
         expected {V2_DIGEST:#018x}. Some round-three change is not actually \
         switched off by v2()."
    );
}

/// ...and the new default must genuinely differ, or the guard above would be
/// asserting nothing about the round.
#[test]
fn the_new_default_is_not_the_round_two_default() {
    assert_ne!(
        digest(Config::default()),
        V2_DIGEST,
        "the new default plays exactly the round-two agent, so nothing shipped"
    );
}
