//! The public-information feature vector the learned value reads.
//!
//! Every entry is a small **integer** stored as an `f32` — a count, a coin
//! total, a bit — clamped to `[-127, 127]`. That is a deliberate storage
//! decision rather than an accident of the game: the training corpus dumps
//! these as `i8`, so 6.7 million rows fit in a few gigabytes, and the network
//! learns its own scaling (which the trainer then folds back into the first
//! layer so inference reads the raw integers). `tests::every_feature_is_a_small_integer`
//! pins the contract.
//!
//! # What is and is not read
//!
//! Only accessors that [`duels_core::Observation`] also carries: built masks,
//! coins, shields, science counts, the conflict pawn, the loot tokens, the
//! wonders drafted and built, the progress tokens on the board and in each
//! city, the discard pile, and — for the structure — which slots are occupied,
//! which are face up, and the identity of a **face-up** card. A face-down
//! slot contributes its occupancy bit and nothing else. Nothing here calls
//! `engine::hidden_info` or looks at a deck. `tests/determinization_invariance.rs`
//! is the proof: two unrelated hidden worlds sampled from the same observation
//! produce the same bits.
//!
//! # Perspective
//!
//! Everything is expressed from `me`'s side: `me` first, the opponent second,
//! the conflict pawn signed positive when it moves towards the opponent's
//! capital. So a single network serves either player, and a caller that
//! always evaluates for [`Player::One`] (as `mcts-eval`'s tree does) reads
//! `features(state, Player::One)`.
//!
//! # Layout
//!
//! Fixed offsets, so the same numbers index the trainer's columns, the dumped
//! corpus, and the first layer of the network. [`feature_names`] spells every
//! column out for a human and is checked against [`NUM_FEATURES`] in a test.

use duels_core::data::{CardId, CardType, TokenId, WonderId, NUM_CARDS, NUM_TOKENS, NUM_WONDERS};
use duels_core::state::{Pending, Phase};
use duels_core::{cost, scoring, GameState, Player};

// -- global block -----------------------------------------------------------

const G_AGE: usize = 0; // 3: age one-hot
const G_PHASE: usize = 3; // 4: phase one-hot
const G_TO_MOVE: usize = 7; // me to move
const G_EXTRA_TURN: usize = 8; // the mover has a banked extra turn
const G_CONFLICT: usize = 9; // pawn, +ve towards the opponent's capital
const G_LOOT: usize = 10; // 4: [me near, me far, opp near, opp far] taken
const G_TURN: usize = 14;
const G_OCCUPIED: usize = 15;
const G_REVEALED: usize = 16;
const G_ACCESSIBLE: usize = 17;
const G_FACE_DOWN: usize = 18;
const G_WONDERS_BUILT_TOTAL: usize = 19;
const G_WONDER_SLOTS_LEFT: usize = 20;
const G_BOARD_TOKENS: usize = 21; // 10
const G_ASIDE_TOKENS: usize = 31; // 10
const G_PENDING: usize = 41; // 5: none, token, great library, destroy, mausoleum
const G_DRAFT_STEP: usize = 46;
const G_LAST_TAKER_ME: usize = 47;

// -- discard block ----------------------------------------------------------

const D_TYPE: usize = 48; // 7: discarded cards by colour
const D_SCIENCE: usize = 55; // 6: discarded science symbols (no Balance)
const D_FODDER: usize = 61; // cards buried under wonders

// -- structure block --------------------------------------------------------

const T_OCCUPIED: usize = 62; // 20
const T_REVEALED: usize = 82; // 20
const T_FACE_UP: usize = 102; // 73: identity of every face-up card

// -- per-player block (me, then opponent) -----------------------------------

const P_BASE: usize = 175;
const P_COINS: usize = 0;
const P_SHIELDS: usize = 1;
const P_SCIENCE: usize = 2; // 7
const P_DISTINCT: usize = 9;
const P_PAIRS: usize = 10; // 6
const P_PRODUCTION: usize = 16; // 5
const P_CHOICE: usize = 21; // 2: raw, manufactured
const P_FIXED: usize = 23; // 5
const P_TYPES: usize = 28; // 7: built cards by colour
const P_WONDERS_UNBUILT: usize = 35;
const P_WONDERS_BUILT_N: usize = 36;
const P_TOKENS_N: usize = 37;
const P_BREAKDOWN: usize = 38; // 9: scoring::Breakdown, in field order
const P_TRADE: usize = 47; // 5: per-unit trade price
const P_BUILT: usize = 52; // 73
const P_WONDERS_OWNED: usize = 125; // 12
const P_WONDERS_BUILT: usize = 137; // 12
const P_TOKENS: usize = 149; // 10
const P_LEN: usize = 159;

/// The width of the feature vector.
pub const NUM_FEATURES: usize = P_BASE + 2 * P_LEN;

const _: () = assert!(P_TOKENS + NUM_TOKENS == P_LEN);
const _: () = assert!(P_WONDERS_BUILT + NUM_WONDERS == P_TOKENS);
const _: () = assert!(P_BUILT + NUM_CARDS == P_WONDERS_OWNED);
const _: () = assert!(T_FACE_UP + NUM_CARDS == P_BASE);

/// Clamp a count into the `i8` range the corpus stores.
#[inline]
fn q<T: Into<i64>>(v: T) -> f32 {
    v.into().clamp(-127, 127) as f32
}

#[inline]
fn bit(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// The feature vector of `state` from `me`'s perspective.
///
/// Reads public information only — see the module docs — and allocates
/// nothing.
pub fn features(state: &GameState, me: Player) -> [f32; NUM_FEATURES] {
    let mut x = [0.0f32; NUM_FEATURES];
    let opp = me.other();

    // Global.
    let age = state.age().clamp(1, 3) as usize;
    x[G_AGE + age - 1] = 1.0;
    x[G_PHASE
        + match state.phase() {
            Phase::WonderDraft => 0,
            Phase::Turn => 1,
            Phase::ChooseFirstPlayer => 2,
            Phase::GameOver => 3,
        }] = 1.0;
    x[G_TO_MOVE] = bit(state.current_player() == me);
    x[G_EXTRA_TURN] = bit(state.extra_turn());
    x[G_CONFLICT] = match me {
        Player::One => q(state.conflict()),
        Player::Two => q(-i16::from(state.conflict())),
    };
    x[G_LOOT] = bit(!state.loot_available(me, 0));
    x[G_LOOT + 1] = bit(!state.loot_available(me, 1));
    x[G_LOOT + 2] = bit(!state.loot_available(opp, 0));
    x[G_LOOT + 3] = bit(!state.loot_available(opp, 1));
    x[G_TURN] = q(state.turn().min(127) as u8);
    let occupied = state.occupied_slots();
    let revealed = state.revealed_slots() & occupied;
    x[G_OCCUPIED] = q(occupied.count_ones() as u8);
    x[G_REVEALED] = q(revealed.count_ones() as u8);
    x[G_ACCESSIBLE] = q(state.accessible_slots().count_ones() as u8);
    x[G_FACE_DOWN] = q((occupied & !revealed).count_ones() as u8);
    x[G_WONDERS_BUILT_TOTAL] = q(state.wonders_built_total());
    x[G_WONDER_SLOTS_LEFT] = bit(state.wonder_slots_left());
    for t in state.board_tokens() {
        x[G_BOARD_TOKENS + t.index()] = 1.0;
    }
    for t in state.set_aside_tokens() {
        x[G_ASIDE_TOKENS + t.index()] = 1.0;
    }
    x[G_PENDING
        + match state.pending() {
            None => 0,
            Some(Pending::ProgressToken) => 1,
            Some(Pending::GreatLibraryToken { .. }) => 2,
            Some(Pending::Destroy { .. }) => 3,
            Some(Pending::MausoleumBuild) => 4,
        }] = 1.0;
    x[G_DRAFT_STEP] = q(state.draft_step());
    x[G_LAST_TAKER_ME] = bit(state.last_card_taker() == me);

    // Discard pile and wonder fodder.
    for c in state.discard_pile() {
        let def = c.def();
        x[D_TYPE + def.kind.index()] += 1.0;
        if let Some(sym) = def.science {
            if sym.index() < 6 {
                x[D_SCIENCE + sym.index()] += 1.0;
            }
        }
    }
    x[D_FODDER] = q(state.wonder_fodder_mask().count_ones() as u8);

    // The structure: occupancy, which slots are face up, and the identity of
    // the face-up cards. `face_up_card` is `None` for a face-down slot.
    for slot in 0..20u8 {
        let b = 1u32 << slot;
        if occupied & b != 0 {
            x[T_OCCUPIED + slot as usize] = 1.0;
        }
        if revealed & b != 0 {
            x[T_REVEALED + slot as usize] = 1.0;
            if let Some(c) = state.face_up_card(slot) {
                x[T_FACE_UP + c.index()] = 1.0;
            }
        }
    }

    // Both cities, me first.
    for (i, p) in [me, opp].into_iter().enumerate() {
        let o = P_BASE + i * P_LEN;
        let ps = state.player(p);
        x[o + P_COINS] = q(ps.coins().min(127));
        x[o + P_SHIELDS] = q(ps.shields());
        let sci = ps.science();
        for (k, &n) in sci.iter().enumerate() {
            x[o + P_SCIENCE + k] = q(n);
        }
        x[o + P_DISTINCT] = q(ps.distinct_science());
        for sym in ps.pairs_awarded() {
            if sym.index() < 6 {
                x[o + P_PAIRS + sym.index()] = 1.0;
            }
        }
        let prod = ps.production();
        for (k, &n) in prod.iter().enumerate() {
            x[o + P_PRODUCTION + k] = q(n);
        }
        let (raw, man) = ps.choice_sources();
        x[o + P_CHOICE] = q(raw);
        x[o + P_CHOICE + 1] = q(man);
        for (k, &f) in ps.fixed_trade().iter().enumerate() {
            x[o + P_FIXED + k] = bit(f);
        }
        for c in ps.built() {
            x[o + P_BUILT + c.index()] = 1.0;
            x[o + P_TYPES + c.def().kind.index()] += 1.0;
        }
        let mut owned = 0u8;
        for w in ps.wonders() {
            x[o + P_WONDERS_OWNED + w.index()] = 1.0;
            owned += 1;
        }
        for w in ps.wonders_built() {
            x[o + P_WONDERS_BUILT + w.index()] = 1.0;
        }
        x[o + P_WONDERS_BUILT_N] = q(ps.wonder_count());
        x[o + P_WONDERS_UNBUILT] = q(owned.saturating_sub(ps.wonder_count()));
        for t in ps.tokens() {
            x[o + P_TOKENS + t.index()] = 1.0;
        }
        x[o + P_TOKENS_N] = q(ps.token_count());
        let b = scoring::breakdown(state, p);
        for (k, v) in [
            b.civilian,
            b.scientific,
            b.commercial,
            b.guilds,
            b.wonders,
            b.progress_tokens,
            b.military,
            b.coins,
            b.total,
        ]
        .into_iter()
        .enumerate()
        {
            x[o + P_BREAKDOWN + k] = q(v.min(127));
        }
        for (k, &price) in cost::trade_prices(state, p).iter().enumerate() {
            x[o + P_TRADE + k] = q(price.min(127));
        }
    }

    x
}

/// A human-readable name for every column, in feature order.
///
/// Used by the corpus dump's metadata and by anyone reading a trained first
/// layer; `tests::the_names_cover_every_column` keeps it the same width as
/// [`NUM_FEATURES`].
pub fn feature_names() -> Vec<String> {
    let mut n: Vec<String> = Vec::with_capacity(NUM_FEATURES);
    let mut push = |s: String| n.push(s);
    for a in 1..=3 {
        push(format!("g.age{a}"));
    }
    for p in ["draft", "turn", "choose_first", "over"] {
        push(format!("g.phase.{p}"));
    }
    push("g.to_move".into());
    push("g.extra_turn".into());
    push("g.conflict".into());
    for l in ["me_near", "me_far", "opp_near", "opp_far"] {
        push(format!("g.loot.{l}"));
    }
    push("g.turn".into());
    push("g.occupied".into());
    push("g.revealed".into());
    push("g.accessible".into());
    push("g.face_down".into());
    push("g.wonders_built_total".into());
    push("g.wonder_slots_left".into());
    for t in TokenId::all() {
        push(format!("g.board_token.{}", t.slug()));
    }
    for t in TokenId::all() {
        push(format!("g.aside_token.{}", t.slug()));
    }
    for p in ["none", "token", "great_library", "destroy", "mausoleum"] {
        push(format!("g.pending.{p}"));
    }
    push("g.draft_step".into());
    push("g.last_taker_me".into());
    for k in CardType::ALL {
        push(format!("d.type.{k:?}"));
    }
    for s in &duels_core::data::Science::ALL[..6] {
        push(format!("d.science.{s:?}"));
    }
    push("d.fodder".into());
    for s in 0..20 {
        push(format!("t.occupied.{s}"));
    }
    for s in 0..20 {
        push(format!("t.revealed.{s}"));
    }
    for c in CardId::all() {
        push(format!("t.face_up.{}", c.slug()));
    }
    for side in ["me", "opp"] {
        push(format!("{side}.coins"));
        push(format!("{side}.shields"));
        for s in duels_core::data::Science::ALL {
            push(format!("{side}.science.{s:?}"));
        }
        push(format!("{side}.distinct_science"));
        for s in &duels_core::data::Science::ALL[..6] {
            push(format!("{side}.pair.{s:?}"));
        }
        for r in duels_core::data::Resource::ALL {
            push(format!("{side}.production.{r:?}"));
        }
        push(format!("{side}.choice.raw"));
        push(format!("{side}.choice.manufactured"));
        for r in duels_core::data::Resource::ALL {
            push(format!("{side}.fixed_trade.{r:?}"));
        }
        for k in CardType::ALL {
            push(format!("{side}.type.{k:?}"));
        }
        push(format!("{side}.wonders_unbuilt"));
        push(format!("{side}.wonders_built_n"));
        push(format!("{side}.tokens_n"));
        for f in [
            "civilian",
            "scientific",
            "commercial",
            "guilds",
            "wonders",
            "progress_tokens",
            "military",
            "coins",
            "total",
        ] {
            push(format!("{side}.vp.{f}"));
        }
        for r in duels_core::data::Resource::ALL {
            push(format!("{side}.trade_price.{r:?}"));
        }
        for c in CardId::all() {
            push(format!("{side}.built.{}", c.slug()));
        }
        for w in WonderId::all() {
            push(format!("{side}.wonder_owned.{}", w.slug()));
        }
        for w in WonderId::all() {
            push(format!("{side}.wonder_built.{}", w.slug()));
        }
        for t in TokenId::all() {
            push(format!("{side}.token.{}", t.slug()));
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn positions() -> Vec<GameState> {
        let mut out = Vec::new();
        for seed in 0..40u64 {
            let mut st = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xF00D);
            let steps = 1 + (seed as usize * 7) % 70;
            for _ in 0..steps {
                let legal = engine::legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                let a = legal[rng.gen_range(0..legal.len())];
                engine::apply_quiet(&mut st, a, &mut rng).unwrap();
            }
            out.push(st);
        }
        out
    }

    #[test]
    fn the_names_cover_every_column() {
        let names = feature_names();
        assert_eq!(names.len(), NUM_FEATURES);
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate feature names");
        // Spot-check that the fixed offsets and the names agree.
        assert_eq!(names[G_CONFLICT], "g.conflict");
        assert_eq!(names[D_FODDER], "d.fodder");
        assert_eq!(
            names[T_FACE_UP],
            format!("t.face_up.{}", CardId::from_index(0).slug())
        );
        assert_eq!(names[P_BASE + P_COINS], "me.coins");
        assert_eq!(names[P_BASE + P_LEN + P_COINS], "opp.coins");
        assert_eq!(
            names[P_BASE + P_LEN + P_TOKENS + NUM_TOKENS - 1],
            format!("opp.token.{}", TokenId::from_index(NUM_TOKENS - 1).slug())
        );
    }

    /// The storage contract: every feature is an integer in `[-127, 127]`.
    #[test]
    fn every_feature_is_a_small_integer() {
        for st in positions() {
            for me in Player::ALL {
                for (i, &v) in features(&st, me).iter().enumerate() {
                    assert!(
                        v.fract() == 0.0 && (-127.0..=127.0).contains(&v),
                        "feature {i} ({}) = {v}",
                        feature_names()[i]
                    );
                }
            }
        }
    }

    /// The two perspectives see the same board with the sides swapped: the
    /// per-player blocks exchange places, the conflict flips sign, and the
    /// `to_move` bit is complementary on any position with a mover.
    #[test]
    fn the_two_perspectives_are_mirror_images() {
        for st in positions() {
            let a = features(&st, Player::One);
            let b = features(&st, Player::Two);
            assert_eq!(a[G_CONFLICT], -b[G_CONFLICT]);
            assert_eq!(a[G_TO_MOVE] + b[G_TO_MOVE], 1.0);
            assert_eq!(a[G_LAST_TAKER_ME] + b[G_LAST_TAKER_ME], 1.0);
            assert_eq!(&a[G_LOOT..G_LOOT + 2], &b[G_LOOT + 2..G_LOOT + 4]);
            assert_eq!(
                &a[P_BASE..P_BASE + P_LEN],
                &b[P_BASE + P_LEN..P_BASE + 2 * P_LEN]
            );
            assert_eq!(
                &a[P_BASE + P_LEN..P_BASE + 2 * P_LEN],
                &b[P_BASE..P_BASE + P_LEN]
            );
            // Everything that is not perspective-dependent agrees.
            assert_eq!(&a[D_TYPE..P_BASE], &b[D_TYPE..P_BASE]);
        }
    }

    #[test]
    fn a_fresh_game_looks_like_one() {
        let st = engine::new_game(3);
        let x = features(&st, Player::One);
        assert_eq!(x[G_AGE], 1.0);
        assert_eq!(x[G_PHASE], 1.0, "the draft is on");
        assert_eq!(x[G_CONFLICT], 0.0);
        assert_eq!(x[P_BASE + P_COINS], 7.0);
        assert_eq!(x[P_BASE + P_LEN + P_COINS], 7.0);
        assert_eq!(x[P_BASE + P_BREAKDOWN + 8], 2.0, "7 coins score 2");
    }
}
