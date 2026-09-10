//! The playout policy: a fast, self-contained default policy used from the
//! newly expanded leaf down to a `GameResult`.
//!
//! Deliberately cheap. Rollouts are where nearly all of the search's time
//! goes, so the policy is a single weighted draw over the legal actions with
//! no lookahead and no opponent model. Two kinds of bias were tried here:
//!
//! - a preference ordering over *kinds* of move (weights of 1 across the
//!   board reproduce the uniform baseline exactly, kept as a configuration
//!   option so the effect can be measured — see [`RolloutWeights::UNIFORM`]
//!   and [`RolloutWeights::BIASED`], the shipped default);
//! - for [`Action::Build`] specifically, a handful of *per-card* multipliers
//!   ([`RolloutWeights::SMART`]) that look up statically-known facts about
//!   the card in the slot — whether it is free via a chain the mover already
//!   owns, and whether it would grant a brand-new scientific symbol or
//!   complete a pair. Each lookup is an array index or a bitmask test, so it
//!   stays cheap in isolation, but it is not free: `examples/rollout_bench.rs`
//!   measured it at 10-25% fewer simulations/second than `UNIFORM` at a fixed
//!   time budget (the exact number moves with system load, since a lookup
//!   this small is easily dwarfed by scheduling noise on a busy machine).
//!
//! **`SMART` did not earn that cost.** Two independent 40-game head-to-head
//! runs against `UNIFORM` at equal `Budget::TimeMs` put it at 47.5% and then
//! 50.0% — statistically indistinguishable from a coin flip (n=40 has a
//! ~7.9-point standard error), and never once ahead of `UNIFORM` by a margin
//! that survived the slower throughput. A milder version of the same idea
//! (weaker multipliers) scored 55% in one run, but with only one run and the
//! same-size sample that is not evidence either way. For comparison, `BIASED`
//! against `UNIFORM` scored 60% and then 45% across those same two runs — a
//! swing bigger than any effect being measured, which is itself the
//! headline finding: at n=40 games on a machine shared with other concurrent
//! work, `Budget::TimeMs` head-to-heads are noisy enough (through both game
//! randomness and load-dependent simulation counts) to swallow a real but
//! modest rollout-quality difference. `SMART` is kept here, tested and
//! documented, as a measured negative result rather than shipped as the
//! default — see the module's git history / PR description for the full
//! numbers and how they were obtained.
//!
//! Uniform-random play discards roughly a third of the cards it touches and
//! is indifferent between a free chain-build and a bad trade, which is far
//! worse than any human line; that noise floor is what makes pure-random
//! playouts a weak signal of a position's true value, and is presumably why
//! `BIASED`'s kind-level bias was adopted as the default in the first place.
//!
//! # Race awareness ([`RaceWeights`])
//!
//! A third, independent layer sits on top of the two above: multipliers keyed
//! not to the card alone but to *how far along a win condition the position
//! already is*. See [`RaceWeights`] for the mechanism, the tuning variants,
//! and why it is gated on a cheap "is anything happening" test so that quiet
//! Age I positions pay almost nothing.

use duels_core::data::{CardType, NUM_SCIENCE};
use duels_core::engine;
use duels_core::{Action, GameResult, GameState, Player};
use rand::rngs::StdRng;
use rand::Rng;

/// Relative weights for the kinds of move a playout can make, plus a few
/// cheap per-card multipliers applied to [`Action::Build`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RolloutWeights {
    /// Weight for [`Action::Build`].
    pub build: f64,
    /// Weight for [`Action::BuildWonder`].
    pub wonder: f64,
    /// Weight for [`Action::Discard`].
    pub discard: f64,
    /// Multiplier on an [`Action::Build`] whose card is free because the
    /// mover already owns its chain prerequisite. A free build is never
    /// worse than the same card bought outright, so this only ever biases
    /// the playout towards a strictly-at-least-as-good line.
    pub chain_free_mult: f64,
    /// Multiplier on an [`Action::Build`] whose card carries a scientific
    /// symbol the mover does not hold at all yet (progress towards the
    /// 6-symbol instant win).
    pub new_symbol_mult: f64,
    /// Multiplier on an [`Action::Build`] whose card carries a scientific
    /// symbol the mover already holds exactly once, i.e. completing it wins
    /// a progress token immediately. Kept larger than `new_symbol_mult`
    /// since the payoff is immediate rather than merely a step closer.
    pub pair_complete_mult: f64,
}

impl RolloutWeights {
    /// Every kind equally likely and no per-card bias: the plain
    /// uniform-random playout policy.
    pub const UNIFORM: RolloutWeights = RolloutWeights {
        build: 1.0,
        wonder: 1.0,
        discard: 1.0,
        chain_free_mult: 1.0,
        new_symbol_mult: 1.0,
        pair_complete_mult: 1.0,
    };

    /// The original bias: prefer putting cards into the city, take wonders
    /// readily, discard only when little else looks available. No per-card
    /// distinctions among builds.
    pub const BIASED: RolloutWeights = RolloutWeights {
        build: 4.0,
        wonder: 2.0,
        discard: 1.0,
        chain_free_mult: 1.0,
        new_symbol_mult: 1.0,
        pair_complete_mult: 1.0,
    };

    /// `BIASED` plus the cheap per-card multipliers: a free chain build is
    /// strongly preferred, and builds that grant or complete a scientific
    /// symbol are moderately preferred. **Measured and rejected as the
    /// default** — see the module docs: it costs 10-25% throughput and did
    /// not show a reproducible win rate over [`RolloutWeights::UNIFORM`].
    /// Kept for future experimentation and so the negative result is
    /// reproducible.
    pub const SMART: RolloutWeights = RolloutWeights {
        build: 4.0,
        wonder: 2.0,
        discard: 1.0,
        chain_free_mult: 4.0,
        new_symbol_mult: 1.5,
        pair_complete_mult: 3.0,
    };

    /// Whether this is the uniform policy (no kind bias and no per-card
    /// bias).
    #[inline]
    pub fn is_uniform(&self) -> bool {
        *self == RolloutWeights::UNIFORM
    }

    /// Whether any per-card [`Action::Build`] multiplier is active, i.e.
    /// whether `weight` needs to look at the card in the slot at all. Lets
    /// [`RolloutWeights::weight`] skip the `face_up_card`/`science` lookups
    /// entirely for policies (like [`RolloutWeights::BIASED`]) that never
    /// use them.
    #[inline]
    fn needs_card_lookup(&self) -> bool {
        self.chain_free_mult != 1.0 || self.new_symbol_mult != 1.0 || self.pair_complete_mult != 1.0
    }

    /// The weight of `action` in `state`, where `state.current_player()` is
    /// the mover about to take it.
    #[inline]
    fn weight(&self, state: &GameState, action: Action) -> f64 {
        match action {
            Action::Build { slot } => {
                let mut w = self.build;
                if self.needs_card_lookup() {
                    if let Some(card) = state.face_up_card(slot) {
                        let def = card.def();
                        let mover = state.player(state.current_player());
                        if let Some(prereq) = def.chain_from {
                            if mover.has_built(prereq) {
                                w *= self.chain_free_mult;
                            }
                        }
                        if let Some(sym) = def.science {
                            match mover.science()[sym.index()] {
                                0 => w *= self.new_symbol_mult,
                                1 => w *= self.pair_complete_mult,
                                _ => {}
                            }
                        }
                    }
                }
                w
            }
            Action::BuildWonder { .. } => self.wonder,
            Action::Discard { .. } => self.discard,
            // Effect choices (progress tokens, Mausoleum, destroy, first
            // player) are picked uniformly: there is no cheap ordering over
            // them that is obviously right.
            _ => 1.0,
        }
    }
}

impl Default for RolloutWeights {
    fn default() -> Self {
        RolloutWeights::BIASED
    }
}

/// The multiplier a Tier-1 *terminal rail* applies (see [`RaceWeights::rail`]).
///
/// Deliberately the same magnitude as `duels_strategy`'s
/// `PriorWeights::dominating`, which promotes the same class of move
/// (close a race, or break a certain opposing one) for the same reason, one
/// layer up in the search. Keeping the two numbers equal is a readability
/// choice, not a coupling: nothing here calls the strategy layer.
pub const RAIL: f64 = 50.0;

/// Race-progress multipliers layered on top of [`RolloutWeights`].
///
/// # What this is for
///
/// [`RolloutWeights`] judges a move by facts about the *card*: is it free via
/// a chain, does it carry a symbol. That is blind to the thing that actually
/// decides a race — how far along the race the position already is. A second
/// scientific symbol is close to worthless; a *sixth* ends the game on the
/// spot, and a rollout that takes it with the same probability as any other
/// build reports a won position as a coin flip.
///
/// A self-play diagnostic at `Nodes(2000)` measured exactly that asymmetry.
/// Over 400 games, *either* player reached 5 distinct symbols in only **1.5%**
/// of games (6/400) — but 67% of those (4/6) then converted into an actual
/// scientific-supremacy win. The tree finishes a science race fine once it is
/// close; the rollouts almost never build one up in the first place, so the
/// tree is rarely shown a position where it is close. Military told a milder
/// version of the same story: 48% exposure, 36% conversion.
///
/// **That hypothesis did not survive being tested** — see the crate docs. The
/// rails below are worth real strength, but by not missing military closes,
/// not by making the search race for science. Two caveats worth carrying
/// forward: the 1.5% came from one seed range and the same measurement over
/// three ranges reads 2.5%; and 5-symbol positions are rare enough
/// (`n ~= 30` per 1200 games) that a conversion rate over them has a standard
/// error near 9 points, so that channel is very hard to measure at all.
///
/// # The mechanism, in two tiers
///
/// Both are computed from raw public state — `GameState`'s accessors and
/// `card.def()`, each an array index or a bitmask test. **Nothing here calls
/// `duels_strategy`**: one full slate of its reads costs about 29% of a whole
/// rollout, which is affordable once per *tree node* and not once per *ply*,
/// and its slot-level outputs go stale within a couple of plies anyway.
///
/// - **Tier 1, terminal rails.** An action that takes an available win (the
///   mover's 6th distinct symbol, or enough shields to reach the opponent's
///   capital) or that removes from the structure the card the *opponent*
///   needs for theirs, gets its weight floored at [`RaceWeights::rail`]. A
///   floor rather than a factor, so a rail is a rail regardless of what Tier 2
///   said about the same move.
/// - **Tier 2, escalating commitment.** Four small tables, indexed by race
///   progress *before* the move, multiply a push (advance my race) or a deny
///   (take the card that advances theirs). They ramp: the tables are ~1.0 in
///   the flat early range and only bite once a race is real.
///
/// A move that both pushes and denies multiplies both factors.
///
/// # Deliberately out of scope
///
/// Pair completion and chain-free builds already live in [`RolloutWeights`]
/// and are untouched here. Progress tokens (including Law's free symbol),
/// The Mausoleum, and age-tempo gating are all left out of this pass on
/// purpose: each is a separate question, and bundling them would make the
/// measurement uninterpretable.
///
/// # Cost
///
/// [`RaceWeights::NEUTRAL`] is bit-for-bit the pre-race policy — the whole
/// layer is skipped, not multiplied through by ones. Even when it is enabled,
/// a position where neither player holds 3 distinct symbols and the pawn is
/// within 2 of centre is *inactive*: no per-card lookup happens at all, which
/// is most of Age I.
///
/// The affordability gate this was built against was "within 5% of plain
/// [`RolloutWeights::BIASED`]'s throughput, or do not bother measuring
/// strength". `examples/rollout_bench.rs`, best of five interleaved rounds
/// over 24 real positions:
///
/// | policy | `Nodes(2000)` | `TimeMs` |
/// |---|---|---|
/// | `BIASED` (baseline) | 100.0% | 100.0% |
/// | `SMART` | 97.2% | 96.1% |
/// | `BIASED` + `TIER1_ONLY` | 96.5% | 96.9% |
/// | `BIASED` + `MEDIUM` | **97.4%** | **97.2%** |
/// | `BIASED` + `strong()` | 96.8% | 96.9% |
///
/// A **2.6-3.5% cost**, and the gate passes. Two things are worth recording
/// about how it got there, because both were surprising:
///
/// - **Every variant costs the same**, including [`RaceWeights::TIER1_ONLY`],
///   whose tables are all `1.0`. The cost is the *lookup* — reaching the card
///   in the slot at all — not the arithmetic the tables drive. Tuning the
///   numbers is therefore free; only the decision to look is not.
/// - The first draft measured **93.5%**, outside the gate, and the fix was
///   not algorithmic. Every `duels_core::data` accessor (`CardId::def`,
///   `TokenId::def`, `military()`) goes through a `OnceLock`, so testing "does
///   this player hold Strategy" by iterating their tokens cost one atomic
///   acquire load *per token per ply*. Resolving those once into
///   [`race_statics`] — a bit test against the token bitmask, an array index
///   for wonder shields — was worth 3.5 points on its own, far more than the
///   [`SlotMemo`] added at the same time, which measured as nothing when it
///   was tried on its own beforehand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaceWeights {
    /// Floor applied to any action that takes an available win or removes an
    /// available opposing one. `1.0` switches the rails off (they can never
    /// lower a weight, since every table entry is at least `1.0`).
    pub rail: f64,
    /// Indexed by the mover's distinct-symbol count *before* the move:
    /// multiplies an [`Action::Build`] of a card carrying a symbol they do
    /// not hold.
    pub sci_push: [f64; 6],
    /// Indexed by the opponent's distinct-symbol count: multiplies any action
    /// that removes from the structure a card carrying a symbol the opponent
    /// does not hold.
    pub sci_deny: [f64; 6],
    /// Indexed by `|d|`, the conflict pawn's signed distance in the mover's
    /// favour: multiplies a shield-carrying [`Action::Build`] or
    /// [`Action::BuildWonder`].
    pub mil_push: [f64; 9],
    /// Indexed by `max(-d, 0)`, i.e. how far the *opponent* is ahead:
    /// multiplies an [`Action::Discard`] or [`Action::BuildWonder`] that
    /// consumes a red card.
    pub mil_deny: [f64; 9],
}

impl RaceWeights {
    /// No race awareness at all: every table entry `1.0` and the rails off.
    ///
    /// **Bit-for-bit the policy this module shipped before `RaceWeights`
    /// existed** — the code path is skipped rather than multiplied by ones.
    /// See `tests::race_neutral_is_the_pre_race_policy_move_for_move` and
    /// `tree::tests::race_neutral_grows_the_same_tree_as_the_pre_race_search`.
    pub const NEUTRAL: RaceWeights = RaceWeights {
        rail: 1.0,
        sci_push: [1.0; 6],
        sci_deny: [1.0; 6],
        mil_push: [1.0; 9],
        mil_deny: [1.0; 9],
    };

    /// Tier 1 only: the terminal rails, with every Tier-2 table left at `1.0`.
    ///
    /// Added so the two tiers' contributions could be measured apart — "does a
    /// rollout that merely never *misses* a win already explain the effect, or
    /// does the escalating commitment matter too?" — and it turned out to be
    /// **the strongest setting in this struct**: `+26.1` Elo against
    /// [`RaceWeights::NEUTRAL`] over 1200 games, where
    /// [`RaceWeights::MEDIUM`] managed `+10.0`. The crate docs carry the full
    /// tables and why it is nonetheless not the default.
    pub const TIER1_ONLY: RaceWeights = RaceWeights {
        rail: RAIL,
        sci_push: [1.0; 6],
        sci_deny: [1.0; 6],
        mil_push: [1.0; 9],
        mil_deny: [1.0; 9],
    };

    /// The tabulated middle setting; [`RaceWeights::mild`] and
    /// [`RaceWeights::strong`] are this one's entries raised to the power
    /// `0.5` and `1.5`.
    pub const MEDIUM: RaceWeights = RaceWeights {
        rail: RAIL,
        sci_push: [1.0, 1.0, 1.25, 1.75, 3.0, RAIL],
        sci_deny: [1.0, 1.0, 1.0, 1.25, 2.0, RAIL],
        mil_push: [1.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.0, 8.0],
        mil_deny: [1.0, 1.0, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0],
    };

    /// Half the exponent of [`RaceWeights::MEDIUM`]: the same shape, half as
    /// committed on a log scale.
    pub fn mild() -> RaceWeights {
        RaceWeights::MEDIUM.powf(0.5)
    }

    /// Half again as much exponent as [`RaceWeights::MEDIUM`].
    pub fn strong() -> RaceWeights {
        RaceWeights::MEDIUM.powf(1.5)
    }

    /// Every Tier-2 entry raised to `p`. [`RaceWeights::rail`] is **not**
    /// scaled: a rail states a fact about the position ("this move ends the
    /// game") rather than a tuning opinion, so diluting it in the mild variant
    /// would be measuring something else.
    ///
    /// `libm::pow` rather than `f64::powf`: these weights feed the playout
    /// policy's action scores during self-play, so they are as much a search
    /// decision as `tree.rs`'s UCB1/progressive-widening math, and the same
    /// cross-platform-libm-disagreement risk applies. See this crate's
    /// `Cargo.toml` for why `libm` is a dependency at all.
    fn powf(self, p: f64) -> RaceWeights {
        let map6 = |a: [f64; 6]| a.map(|x| libm::pow(x, p));
        let map9 = |a: [f64; 9]| a.map(|x| libm::pow(x, p));
        RaceWeights {
            rail: self.rail,
            sci_push: map6(self.sci_push),
            sci_deny: map6(self.sci_deny),
            mil_push: map9(self.mil_push),
            mil_deny: map9(self.mil_deny),
        }
    }

    /// Whether this is [`RaceWeights::NEUTRAL`], i.e. whether the whole layer
    /// can be skipped.
    #[inline]
    pub fn is_neutral(&self) -> bool {
        *self == RaceWeights::NEUTRAL
    }

    /// A compact, stable name for [`crate::Config::describe`], matching the
    /// `race=` spec-string key. `custom` for anything hand-built.
    pub fn name(&self) -> &'static str {
        if self.is_neutral() {
            "neutral"
        } else if *self == RaceWeights::TIER1_ONLY {
            "tier1_only"
        } else if *self == RaceWeights::MEDIUM {
            "medium"
        } else if *self == RaceWeights::mild() {
            "mild"
        } else if *self == RaceWeights::strong() {
            "strong"
        } else {
            "custom"
        }
    }
}

impl Default for RaceWeights {
    fn default() -> Self {
        RaceWeights::NEUTRAL
    }
}

/// The handful of facts about the *rules* (as opposed to the position) that
/// the race layer needs, resolved once for the whole process.
///
/// Every `duels_core::data` accessor — `CardId::def`, `TokenId::def`,
/// `military()` — goes through one `OnceLock`, so reading three of them per
/// rollout *ply* is three atomic acquire loads plus three bounds-checked
/// indexes for data that cannot change. Resolving them here turns the
/// Strategy-token test into a single bit test against the player's token
/// bitmask and the wonder-shield lookup into one array index.
struct RaceStatics {
    /// `duels_core::data::military().capital_distance`.
    cap: i32,
    /// The Strategy progress token (`shield_bonus`), if the data set has one.
    strategy: Option<duels_core::data::TokenId>,
    /// Shields per wonder, indexed by `WonderId::index`.
    wonder_shields: [u8; duels_core::data::NUM_WONDERS],
}

fn race_statics() -> &'static RaceStatics {
    static CACHE: std::sync::OnceLock<RaceStatics> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let mut wonder_shields = [0u8; duels_core::data::NUM_WONDERS];
        for w in duels_core::data::WonderId::all() {
            wonder_shields[w.index()] = w.def().shields;
        }
        RaceStatics {
            cap: i32::from(duels_core::data::military().capital_distance),
            strategy: duels_core::data::TokenId::all().find(|t| t.def().shield_bonus),
            wonder_shields,
        }
    })
}

/// Everything [`RaceWeights`] needs to know about a position, read once per
/// rollout step rather than once per legal action.
///
/// Reads **only public information** — symbol counts, the conflict pawn,
/// progress tokens, the military track's own capital distance — so the
/// multipliers it produces are provably invariant to which hidden-info sample
/// the determinized rollout state came from. See
/// `tests::weights_are_determinization_invariant`.
#[derive(Debug, Clone, Copy)]
struct StepCtx {
    /// The mover's symbol counts, and the opponent's.
    my_sci: [u8; NUM_SCIENCE],
    opp_sci: [u8; NUM_SCIENCE],
    /// Distinct-symbol counts, clamped into the tables' `0..=5` range.
    my_k: usize,
    opp_k: usize,
    /// Signed conflict-pawn distance *towards the opponent's capital*, from
    /// the mover's point of view: positive means the mover is winning the
    /// military race.
    d: i32,
    /// Whether the mover / the opponent holds the Strategy progress token,
    /// which adds a shield to every red building its owner constructs.
    my_strat: bool,
    opp_strat: bool,
    /// The real capital distance from `duels_core::data::military()`.
    cap: i32,
    /// Whether either race is far enough along to be worth looking at. When
    /// false, no per-card lookup happens at all.
    active: bool,
}

impl StepCtx {
    const INACTIVE: StepCtx = StepCtx {
        my_sci: [0; NUM_SCIENCE],
        opp_sci: [0; NUM_SCIENCE],
        my_k: 0,
        opp_k: 0,
        d: 0,
        my_strat: false,
        opp_strat: false,
        cap: 0,
        active: false,
    };

    /// Read `state` from `state.current_player()`'s point of view.
    fn new(state: &GameState, race: &RaceWeights) -> StepCtx {
        if race.is_neutral() {
            return StepCtx::INACTIVE;
        }
        let mover = state.current_player();
        let me = state.player(mover);
        let opp = state.player(mover.other());
        let my_k = usize::from(me.distinct_science());
        let opp_k = usize::from(opp.distinct_science());
        let conflict = i32::from(state.conflict());
        let d = match mover {
            Player::One => conflict,
            Player::Two => -conflict,
        };
        // The cheap gate: in a quiet position nothing below can fire, so pay
        // for none of it.
        if my_k < 3 && opp_k < 3 && d.abs() < 3 {
            return StepCtx::INACTIVE;
        }
        let statics = race_statics();
        StepCtx {
            my_sci: me.science(),
            opp_sci: opp.science(),
            my_k: my_k.min(5),
            opp_k: opp_k.min(5),
            d,
            my_strat: statics.strategy.is_some_and(|t| me.has_token(t)),
            opp_strat: statics.strategy.is_some_and(|t| opp.has_token(t)),
            cap: statics.cap,
            active: true,
        }
    }

    /// The race multiplier for `action`. `1.0` for anything that neither
    /// touches a card in the structure nor moves a race.
    #[inline]
    fn multiplier(
        &self,
        state: &GameState,
        race: &RaceWeights,
        action: Action,
        memo: &mut SlotMemo,
    ) -> f64 {
        if !self.active {
            return 1.0;
        }
        // The three actions that take a card out of the structure. Everything
        // else (effect choices, the wonder draft) has no card to read.
        let (slot, wonder, is_build) = match action {
            Action::Build { slot } => (slot, None, true),
            Action::Discard { slot } => (slot, None, false),
            Action::BuildWonder { slot, wonder } => (slot, Some(wonder), false),
            _ => return 1.0,
        };
        let Some(def) = memo.get(state, slot) else {
            return 1.0;
        };

        let mut m = 1.0f64;
        let mut rail = false;

        if let Some(sym) = def.science {
            let i = sym.index();
            // Push: only a real `Build` puts the symbol in the mover's city;
            // a wonder consumes the card without its effects.
            if is_build && self.my_sci[i] == 0 {
                m *= race.sci_push[self.my_k];
                rail |= self.my_k == 5;
            }
            // Deny: all three actions take the card off the board.
            if self.opp_sci[i] == 0 {
                m *= race.sci_deny[self.opp_k];
                rail |= self.opp_k == 5;
            }
        }

        // Shields the *mover* would gain. Strategy's bonus is written on red
        // buildings only, so a wonder never gets it.
        let my_shields = match wonder {
            Some(w) => i32::from(race_statics().wonder_shields[w.index()]),
            None if is_build => {
                i32::from(def.shields) + i32::from(self.my_strat && def.kind == CardType::Military)
            }
            None => 0,
        };
        if my_shields > 0 {
            m *= race.mil_push[self.d.unsigned_abs().min(8) as usize];
            rail |= self.d + my_shields >= self.cap;
        }

        if def.shields > 0 {
            // What the same card would be worth to the opponent if it were
            // left on the board for them. Taking it away is a rail exactly
            // when leaving it would hand them the capital.
            let opp_shields = i32::from(def.shields)
                + i32::from(self.opp_strat && def.kind == CardType::Military);
            rail |= -self.d + opp_shields >= self.cap;
            // The `Build` case is the push above; this arm is the pure denial
            // of a red card the mover does not want for itself.
            if !is_build {
                m *= race.mil_deny[(-self.d).clamp(0, 8) as usize];
            }
        }

        // A floor, not a factor: whatever Tier 2 made of the same move, a move
        // that ends the game (or stops the opponent ending it) sits at the top
        // of the draw. Every table entry is >= 1.0, so this never lowers `m`.
        if rail {
            m.max(race.rail)
        } else {
            m
        }
    }
}

/// A one-entry memo over "which card is in this slot".
///
/// [`duels_core::engine::legal_actions_into`] emits every action for a slot
/// consecutively — a `Build` (when affordable), a `Discard`, then one
/// `BuildWonder` per buildable wonder — so remembering just the last slot
/// collapses the six-or-so lookups a single slot would otherwise cause into
/// one. `CardId::def` in particular is a `OnceLock` read, which
/// [`RaceWeights`]'s cost notes record as the expensive part of this whole
/// layer.
///
/// Measured on its own, before [`race_statics`] existed, this was worth
/// nothing at all; it is kept because it is strictly less work for one branch,
/// not because a benchmark demanded it.
#[derive(Debug, Clone, Copy)]
struct SlotMemo {
    slot: u8,
    def: Option<&'static duels_core::data::Card>,
}

impl SlotMemo {
    const EMPTY: SlotMemo = SlotMemo {
        slot: u8::MAX,
        def: None,
    };

    #[inline]
    fn get(&mut self, state: &GameState, slot: u8) -> Option<&'static duels_core::data::Card> {
        if self.slot != slot {
            self.slot = slot;
            self.def = state.face_up_card(slot).map(|c| c.def());
        }
        self.def
    }
}

/// Pick one action according to `weights` and `race`, given the position it
/// would be taken in.
///
/// `wbuf` is a caller-owned scratch buffer, reused across every step of every
/// playout so the weighted draw stays allocation-free. Each candidate's weight
/// is computed **once** into it and read back from it, rather than being
/// recomputed inside the draw loop as this function used to do: the total is
/// still summed left to right over the same values, so the RNG draw and the
/// action it selects are unchanged.
pub(crate) fn pick(
    state: &GameState,
    weights: &RolloutWeights,
    race: &RaceWeights,
    legal: &[Action],
    wbuf: &mut Vec<f64>,
    rng: &mut StdRng,
) -> Action {
    debug_assert!(!legal.is_empty());
    if (weights.is_uniform() && race.is_neutral()) || legal.len() == 1 {
        return legal[rng.gen_range(0..legal.len())];
    }
    let ctx = StepCtx::new(state, race);
    wbuf.clear();
    let mut total = 0.0f64;
    if ctx.active {
        let mut memo = SlotMemo::EMPTY;
        for &a in legal {
            let w = weights.weight(state, a) * ctx.multiplier(state, race, a, &mut memo);
            total += w;
            wbuf.push(w);
        }
    } else {
        for &a in legal {
            let w = weights.weight(state, a);
            total += w;
            wbuf.push(w);
        }
    }
    let mut r = rng.gen_range(0.0..total);
    for (i, &a) in legal.iter().enumerate() {
        r -= wbuf[i];
        if r < 0.0 {
            return a;
        }
    }
    legal[legal.len() - 1]
}

/// Play `state` out to the end and return the result.
///
/// `max_plies` is a safety net, not a rule: the game is finite, so hitting it
/// means a bug elsewhere. Rather than spinning forever the playout stops and
/// scores the position as if Age III had just ended, which is the least
/// misleading thing available.
pub(crate) fn play_out(
    state: &mut GameState,
    weights: &RolloutWeights,
    race: &RaceWeights,
    buf: &mut Vec<Action>,
    wbuf: &mut Vec<f64>,
    rng: &mut StdRng,
    max_plies: u32,
) -> GameResult {
    for _ in 0..max_plies {
        engine::legal_actions_into(state, buf);
        if buf.is_empty() {
            break;
        }
        let action = pick(state, weights, race, buf, wbuf, rng);
        // The state carries a determinized layout sampled at the root (and
        // kept publicly consistent by every forced reveal on the way down),
        // so `apply_unchecked` is free to read it for reveals: this is one
        // determinized playout, exactly as in perfect-information Monte Carlo
        // search, and it never leaks information into a *decision* — those
        // all happen at tree nodes reached through `apply_with_outcome`.
        engine::apply_unchecked(state, action, rng);
    }
    state
        .result()
        .unwrap_or_else(|| duels_core::scoring::civilian_result(state))
}

/// Play at most `plies` steps of the same policy [`play_out`] uses, and report
/// the [`GameResult`] **only if the game actually ended** inside that window.
///
/// `None` means the window ran out with the game still going, and `state` has
/// been advanced by exactly `plies` legal moves — which is what
/// [`crate::LeafValue::Truncated`] then scores statically.
///
/// A `plies` of zero returns `None` without touching `state` or the RNG, so a
/// zero-ply truncation is exactly a static evaluation of the leaf.
///
/// Deliberately *not* written as `play_out` with a smaller `max_plies`: that
/// function's cap is a safety net that scores the position as if Age III had
/// just ended, which is the wrong answer for a truncation — the whole point
/// here is to tell "the game finished" apart from "the window closed".
pub(crate) fn play_out_capped(
    state: &mut GameState,
    weights: &RolloutWeights,
    race: &RaceWeights,
    buf: &mut Vec<Action>,
    wbuf: &mut Vec<f64>,
    rng: &mut StdRng,
    plies: u32,
) -> Option<GameResult> {
    for _ in 0..plies {
        engine::legal_actions_into(state, buf);
        if buf.is_empty() {
            break;
        }
        let action = pick(state, weights, race, buf, wbuf, rng);
        engine::apply_unchecked(state, action, rng);
    }
    // `legal_actions` is empty exactly when the game is over, so an empty
    // action list above is the same "the game ended" case as a settled
    // result; both come back through `state.result()`.
    state.result()
}

/// `pick` and `play_out` exactly as they read before [`RaceWeights`] existed —
/// including the double `weight()` evaluation the single-pass rewrite removed.
///
/// Copied verbatim; do not "simplify" either of them to call the new code,
/// since that is the thing they exist to check. `tree::Tree::legacy_simulate`
/// drives these, so every pre-change equivalence test in this crate reaches
/// them.
#[cfg(test)]
pub(crate) mod legacy {
    use super::*;

    pub(crate) fn pick(
        state: &GameState,
        weights: &RolloutWeights,
        legal: &[Action],
        rng: &mut StdRng,
    ) -> Action {
        debug_assert!(!legal.is_empty());
        if weights.is_uniform() || legal.len() == 1 {
            return legal[rng.gen_range(0..legal.len())];
        }
        let total: f64 = legal.iter().map(|&a| weights.weight(state, a)).sum();
        let mut r = rng.gen_range(0.0..total);
        for &a in legal {
            r -= weights.weight(state, a);
            if r < 0.0 {
                return a;
            }
        }
        legal[legal.len() - 1]
    }

    pub(crate) fn play_out(
        state: &mut GameState,
        weights: &RolloutWeights,
        buf: &mut Vec<Action>,
        rng: &mut StdRng,
        max_plies: u32,
    ) -> GameResult {
        for _ in 0..max_plies {
            engine::legal_actions_into(state, buf);
            if buf.is_empty() {
                break;
            }
            let action = pick(state, weights, buf, rng);
            engine::apply_unchecked(state, action, rng);
        }
        state
            .result()
            .unwrap_or_else(|| duels_core::scoring::civilian_result(state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::data::WonderId;
    use duels_core::testing::StateBuilder;
    use duels_core::Player;
    use rand::SeedableRng;

    #[test]
    fn uniform_weights_are_reported_as_uniform() {
        assert!(RolloutWeights::UNIFORM.is_uniform());
        assert!(!RolloutWeights::BIASED.is_uniform());
        assert!(!RolloutWeights::SMART.is_uniform());
    }

    #[test]
    fn biased_and_smart_agree_when_no_card_carries_a_bonus() {
        // At the very first decision of a fresh game nothing is buildable
        // for free and nobody holds a scientific symbol yet, so SMART's
        // per-card multipliers are all inert and it must reduce to BIASED
        // exactly.
        let state = engine::new_game(0);
        let legal = [
            Action::Build { slot: 1 },
            Action::Discard { slot: 2 },
            Action::BuildWonder {
                slot: 3,
                wonder: WonderId::from_index(0),
            },
        ];
        for &a in &legal {
            assert_eq!(
                RolloutWeights::BIASED.weight(&state, a),
                RolloutWeights::SMART.weight(&state, a),
                "{a:?}"
            );
        }
    }

    #[test]
    fn weighted_picking_respects_the_weights() {
        let state = engine::new_game(0);
        let legal = [
            Action::Build { slot: 1 },
            Action::Discard { slot: 2 },
            Action::BuildWonder {
                slot: 3,
                wonder: WonderId::from_index(0),
            },
        ];
        let w = RolloutWeights::BIASED;
        let mut rng = StdRng::seed_from_u64(1);
        let mut counts = [0u32; 3];
        const N: u32 = 70_000;
        let mut wbuf = Vec::new();
        for _ in 0..N {
            let a = pick(
                &state,
                &w,
                &RaceWeights::NEUTRAL,
                &legal,
                &mut wbuf,
                &mut rng,
            );
            let i = legal.iter().position(|&x| x == a).unwrap();
            counts[i] += 1;
        }
        let total = w.build + w.discard + w.wonder;
        for (i, expected) in [w.build, w.discard, w.wonder].iter().enumerate() {
            let got = f64::from(counts[i]) / f64::from(N);
            assert!(
                (got - expected / total).abs() < 0.01,
                "action {i}: got {got}, expected {}",
                expected / total
            );
        }
    }

    /// A truncated playout must be **the prefix of the playout it truncates**:
    /// the same states, and — the load-bearing half —
    /// *exactly the same randomness consumed*, so that
    /// [`crate::LeafValue::Truncated`]'s evaluation step is provably free of
    /// RNG draws of its own.
    ///
    /// Checked by running both against streams seeded alike and then drawing
    /// from each: a single extra or missing draw inside either function makes
    /// the two follow-up draws differ.
    #[test]
    fn a_truncated_playout_is_the_prefix_of_the_full_one() {
        for seed in 0..8u64 {
            for plies in [0u32, 1, 4, 8, 16] {
                let state = engine::new_game(seed);
                let w = RolloutWeights::BIASED;
                let race = RaceWeights::NEUTRAL;

                let mut capped_state = state;
                let mut rng_a = StdRng::seed_from_u64(seed ^ 0xC0DE);
                let (mut buf, mut wbuf) = (Vec::new(), Vec::new());
                let finished = play_out_capped(
                    &mut capped_state,
                    &w,
                    &race,
                    &mut buf,
                    &mut wbuf,
                    &mut rng_a,
                    plies,
                );

                let mut full_state = state;
                let mut rng_b = StdRng::seed_from_u64(seed ^ 0xC0DE);
                let (mut buf, mut wbuf) = (Vec::new(), Vec::new());
                // The same policy, stopped by the same number of plies. This
                // *scores* an unfinished game rather than reporting it as
                // unfinished, which is the one thing the capped version is
                // written not to do — but it walks the identical prefix.
                play_out(
                    &mut full_state,
                    &w,
                    &race,
                    &mut buf,
                    &mut wbuf,
                    &mut rng_b,
                    plies,
                );

                assert_eq!(
                    capped_state, full_state,
                    "seed {seed}, {plies} plies: the prefixes diverged"
                );
                assert_eq!(
                    rng_a.gen::<u64>(),
                    rng_b.gen::<u64>(),
                    "seed {seed}, {plies} plies: the two playouts consumed different randomness"
                );
                // A real game is far longer than 16 plies, so none of these
                // may claim to have finished; the "finished" arm is exercised
                // by the whole-search tests.
                assert!(
                    finished.is_none(),
                    "seed {seed}: a game ended within {plies} plies"
                );
            }
        }
    }

    /// A zero-ply truncation is a pure no-op: no state change, and not one
    /// random number drawn — which is what makes
    /// `tree::tests::the_degenerate_leaf_settings_reduce_to_their_edges`
    /// able to compare it against a static evaluation bit for bit.
    #[test]
    fn a_zero_ply_truncation_touches_nothing() {
        let state = engine::new_game(4);
        let mut advanced = state;
        let mut rng = StdRng::seed_from_u64(11);
        let mut reference = StdRng::seed_from_u64(11);
        let (mut buf, mut wbuf) = (Vec::new(), Vec::new());
        let finished = play_out_capped(
            &mut advanced,
            &RolloutWeights::BIASED,
            &RaceWeights::NEUTRAL,
            &mut buf,
            &mut wbuf,
            &mut rng,
            0,
        );
        assert!(finished.is_none());
        assert_eq!(advanced, state);
        assert_eq!(rng.gen::<u64>(), reference.gen::<u64>());
    }

    /// The crux of `SMART`: a chain-free build must be picked far more often
    /// than an equal-kind build that is not free, at a ratio matching
    /// `chain_free_mult`.
    #[test]
    fn smart_prefers_a_free_chain_build_over_an_equal_kind_alternative() {
        // "fortifications" chains from "palisade"; give Player One
        // "palisade" so "fortifications" is free, alongside an unrelated
        // buildable card ("clay-pool") that carries neither a chain nor a
        // science bonus.
        let st = StateBuilder::new()
            .built(Player::One, &["palisade"])
            .open_slots(&[(18, "fortifications"), (19, "clay-pool")])
            .coins(Player::One, 10)
            .current(Player::One)
            .build();
        assert_eq!(
            duels_core::data::CardId::from_slug("fortifications")
                .unwrap()
                .def()
                .chain_from,
            duels_core::data::CardId::from_slug("palisade")
        );

        let legal = [Action::Build { slot: 18 }, Action::Build { slot: 19 }];
        let w = RolloutWeights::SMART;
        let free_w = w.weight(&st, legal[0]);
        let priced_w = w.weight(&st, legal[1]);
        assert!(
            (free_w / priced_w - w.chain_free_mult).abs() < 1e-9,
            "free={free_w} priced={priced_w} mult={}",
            w.chain_free_mult
        );

        let mut rng = StdRng::seed_from_u64(9);
        let mut wbuf = Vec::new();
        let mut hits_free = 0u32;
        const N: u32 = 20_000;
        for _ in 0..N {
            if pick(&st, &w, &RaceWeights::NEUTRAL, &legal, &mut wbuf, &mut rng) == legal[0] {
                hits_free += 1;
            }
        }
        let share = f64::from(hits_free) / f64::from(N);
        let expected = w.chain_free_mult / (w.chain_free_mult + 1.0);
        assert!(
            (share - expected).abs() < 0.01,
            "got {share}, expected {expected}"
        );
    }

    /// The full weight one rollout step gives `action`: the kind/card weight
    /// times the race multiplier, exactly as `pick` composes them.
    fn full_weight(
        state: &GameState,
        weights: &RolloutWeights,
        race: &RaceWeights,
        action: Action,
    ) -> f64 {
        let ctx = StepCtx::new(state, race);
        let mut memo = SlotMemo::EMPTY;
        weights.weight(state, action) * ctx.multiplier(state, race, action, &mut memo)
    }

    /// A mid-game position from a seeded random walk, plus its legal actions.
    fn mid_game(seed: u64) -> (GameState, Vec<Action>) {
        let mut state = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x4242);
        for step in 0.. {
            let legal = engine::legal_actions(&state);
            assert!(!legal.is_empty(), "seed {seed} ended before a branchy turn");
            if step >= 16 && legal.len() >= 6 {
                return (state, legal);
            }
            let a = legal[rng.gen_range(0..legal.len())];
            engine::apply_quiet(&mut state, a, &mut rng).expect("a legal action");
        }
        unreachable!()
    }

    #[test]
    fn the_tuning_variants_are_powers_of_medium_with_the_rail_held_fixed() {
        for (v, p) in [(RaceWeights::mild(), 0.5), (RaceWeights::strong(), 1.5)] {
            assert_eq!(v.rail, RaceWeights::MEDIUM.rail, "the rail must not scale");
            for (a, b) in v.sci_push.iter().zip(RaceWeights::MEDIUM.sci_push.iter()) {
                assert!((a - b.powf(p)).abs() < 1e-12);
            }
            for (a, b) in v.mil_push.iter().zip(RaceWeights::MEDIUM.mil_push.iter()) {
                assert!((a - b.powf(p)).abs() < 1e-12);
            }
        }
        // Mild is uniformly gentler than medium, strong uniformly sharper —
        // every entry is >= 1, so raising the exponent can only raise it.
        for i in 0..6 {
            assert!(RaceWeights::mild().sci_push[i] <= RaceWeights::MEDIUM.sci_push[i]);
            assert!(RaceWeights::strong().sci_push[i] >= RaceWeights::MEDIUM.sci_push[i]);
        }
        // Every entry of every table is at least 1, which is what makes the
        // Tier-1 rail a floor rather than something Tier 2 can undercut.
        for v in [
            RaceWeights::mild(),
            RaceWeights::MEDIUM,
            RaceWeights::strong(),
        ] {
            let all = v
                .sci_push
                .iter()
                .chain(v.sci_deny.iter())
                .chain(v.mil_push.iter())
                .chain(v.mil_deny.iter());
            assert!(all.copied().all(|x| x >= 1.0), "{v:?}");
        }
    }

    #[test]
    fn each_variant_names_itself_for_the_spec_string() {
        assert_eq!(RaceWeights::NEUTRAL.name(), "neutral");
        assert_eq!(RaceWeights::TIER1_ONLY.name(), "tier1_only");
        assert_eq!(RaceWeights::mild().name(), "mild");
        assert_eq!(RaceWeights::MEDIUM.name(), "medium");
        assert_eq!(RaceWeights::strong().name(), "strong");
        assert_eq!(RaceWeights::default(), RaceWeights::NEUTRAL);
        let hand_built = RaceWeights {
            rail: 3.0,
            ..RaceWeights::MEDIUM
        };
        assert_eq!(hand_built.name(), "custom");
    }

    /// The correctness requirement the whole option rests on: with
    /// [`RaceWeights::NEUTRAL`] the policy is the pre-race policy *bit for
    /// bit*, checked against the verbatim copy in [`legacy`] over whole
    /// playouts (so the two RNG streams have to stay in step for hundreds of
    /// draws, not just agree once).
    #[test]
    fn race_neutral_is_the_pre_race_policy_move_for_move() {
        for seed in 0..12u64 {
            for w in [
                RolloutWeights::UNIFORM,
                RolloutWeights::BIASED,
                RolloutWeights::SMART,
            ] {
                let mut new_rng = StdRng::seed_from_u64(seed ^ 0xF00D);
                let mut old_rng = StdRng::seed_from_u64(seed ^ 0xF00D);
                let mut new_state = engine::new_game(seed);
                let mut old_state = engine::new_game(seed);
                let (mut buf, mut wbuf, mut old_buf) = (Vec::new(), Vec::new(), Vec::new());
                let mut steps = 0u32;
                loop {
                    engine::legal_actions_into(&new_state, &mut buf);
                    engine::legal_actions_into(&old_state, &mut old_buf);
                    assert_eq!(buf, old_buf, "seed {seed}: the states diverged");
                    if buf.is_empty() {
                        break;
                    }
                    let got = pick(
                        &new_state,
                        &w,
                        &RaceWeights::NEUTRAL,
                        &buf,
                        &mut wbuf,
                        &mut new_rng,
                    );
                    let want = legacy::pick(&old_state, &w, &old_buf, &mut old_rng);
                    assert_eq!(got, want, "seed {seed}, step {steps}: {w:?}");
                    engine::apply_unchecked(&mut new_state, got, &mut new_rng);
                    engine::apply_unchecked(&mut old_state, want, &mut old_rng);
                    steps += 1;
                }
                assert!(steps > 20, "the playout was too short to prove much");
            }
        }
    }

    /// The mandatory property for anything in this repo that reads game state:
    /// two different hidden-information samples of the *same* observation must
    /// produce identical weights, to the bit. The race policy reads only
    /// public information, so this has to hold — and if it ever stops holding,
    /// the rollout is leaking something an agent must not see.
    #[test]
    fn weights_are_determinization_invariant() {
        let mut checked = 0u32;
        for seed in 0..12u64 {
            let (state, _) = mid_game(seed);
            let obs = state.observation();
            let mut rng_a = StdRng::seed_from_u64(seed ^ 0xA1);
            let mut rng_b = StdRng::seed_from_u64(seed ^ 0xB2);
            let a = obs.sample_state(&mut rng_a);
            let b = obs.sample_state(&mut rng_b);

            let legal_a = engine::legal_actions(&a);
            let legal_b = engine::legal_actions(&b);
            assert_eq!(
                legal_a, legal_b,
                "seed {seed}: legality is public, so it cannot differ"
            );
            for &action in &legal_a {
                for race in [
                    RaceWeights::MEDIUM,
                    RaceWeights::mild(),
                    RaceWeights::strong(),
                    RaceWeights::TIER1_ONLY,
                ] {
                    let wa = full_weight(&a, &RolloutWeights::BIASED, &race, action);
                    let wb = full_weight(&b, &RolloutWeights::BIASED, &race, action);
                    assert_eq!(
                        wa.to_bits(),
                        wb.to_bits(),
                        "seed {seed}, {action:?}, {}: {wa} vs {wb}",
                        race.name()
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 100, "only {checked} weights were compared");
    }

    /// Five symbols in, with the sixth on the board: taking it ends the game,
    /// so it must be railed.
    #[test]
    fn the_sixth_symbol_is_railed() {
        // mortar, pendulum, inkwell, wheel, sundial — five distinct.
        let st = StateBuilder::new()
            .built(
                Player::One,
                &[
                    "pharmacist",
                    "workshop",
                    "scriptorium",
                    "apothecary",
                    "academy",
                ],
            )
            // "university" carries the gyroscope, the sixth; "palace" is an
            // ordinary card with no race relevance at all.
            .open_slots(&[(18, "university"), (19, "palace")])
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        assert_eq!(st.player(Player::One).distinct_science(), 5);

        let w = RolloutWeights::BIASED;
        let race = RaceWeights::MEDIUM;
        let win = full_weight(&st, &w, &race, Action::Build { slot: 18 });
        let other = full_weight(&st, &w, &race, Action::Build { slot: 19 });
        assert_eq!(win, w.build * RAIL, "the sixth symbol must be railed");
        assert_eq!(other, w.build, "an unrelated build must be untouched");
        assert!(win / other >= RAIL);

        // TIER1_ONLY reaches the same conclusion through the rail alone.
        let t1 = full_weight(
            &st,
            &w,
            &RaceWeights::TIER1_ONLY,
            Action::Build { slot: 18 },
        );
        assert_eq!(t1, w.build * RAIL);
        // ... and NEUTRAL sees nothing at all.
        let n = full_weight(&st, &w, &RaceWeights::NEUTRAL, Action::Build { slot: 18 });
        assert_eq!(n, w.weight(&st, Action::Build { slot: 18 }));
    }

    /// The opponent is one symbol from an instant win and the card is on the
    /// board: every way of taking it off the board is railed, including the
    /// discard the mover gains nothing else from.
    #[test]
    fn taking_the_opponents_sixth_symbol_away_is_railed() {
        let st = StateBuilder::new()
            .built(
                Player::Two,
                &[
                    "pharmacist",
                    "workshop",
                    "scriptorium",
                    "apothecary",
                    "academy",
                ],
            )
            .wonders(Player::One, &["the-pyramids"])
            .open_slots(&[(18, "university"), (19, "palace")])
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        assert_eq!(st.player(Player::Two).distinct_science(), 5);
        assert_eq!(st.player(Player::One).distinct_science(), 0);

        let w = RolloutWeights::BIASED;
        let race = RaceWeights::MEDIUM;
        let wonder = duels_core::data::WonderId::from_slug("the-pyramids").unwrap();
        for (action, kind_weight) in [
            (Action::Build { slot: 18 }, w.build),
            (Action::Discard { slot: 18 }, w.discard),
            (Action::BuildWonder { slot: 18, wonder }, w.wonder),
        ] {
            assert_eq!(
                full_weight(&st, &w, &race, action),
                kind_weight * RAIL,
                "{action:?} does not take the card away"
            );
        }
        // The card that denies nothing is untouched, whatever is done with it.
        assert_eq!(
            full_weight(&st, &w, &race, Action::Discard { slot: 19 }),
            w.discard
        );
    }

    /// One shield from the capital, with a shield card available: the rail
    /// fires on the build that ends the game, and on taking away the red card
    /// that would end it the other way.
    #[test]
    fn a_closing_shield_card_is_railed_from_both_sides() {
        let cap = i32::from(duels_core::data::military().capital_distance);
        let w = RolloutWeights::BIASED;
        let race = RaceWeights::MEDIUM;

        // Player One is one step from Player Two's capital, and "guard-tower"
        // (1 shield) is on the board.
        let st = StateBuilder::new()
            .conflict((cap - 1) as i8)
            .open_slots(&[(18, "guard-tower"), (19, "palace")])
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        assert_eq!(
            full_weight(&st, &w, &race, Action::Build { slot: 18 }),
            w.build * RAIL
        );
        assert_eq!(
            full_weight(&st, &w, &race, Action::Build { slot: 19 }),
            w.build
        );

        // Mirror image: Player Two is one step from Player One's capital, so
        // Player One must take the red card off the board. A `Discard` gains
        // no shields at all and is still railed, because it denies.
        let st = StateBuilder::new()
            .conflict(-((cap - 1) as i8))
            .open_slots(&[(18, "guard-tower"), (19, "palace")])
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        assert_eq!(
            full_weight(&st, &w, &race, Action::Discard { slot: 18 }),
            w.discard * RAIL
        );
        assert_eq!(
            full_weight(&st, &w, &race, Action::Discard { slot: 19 }),
            w.discard
        );
    }

    /// Strategy adds a shield to every red building its owner constructs, so
    /// it can turn a card that falls one short into a closing one — from
    /// either seat.
    #[test]
    fn the_strategy_token_is_counted_on_both_sides_of_the_rail() {
        let cap = i32::from(duels_core::data::military().capital_distance);
        let w = RolloutWeights::BIASED;
        let race = RaceWeights::MEDIUM;
        let build18 = Action::Build { slot: 18 };

        // Two short of the capital with a 1-shield card: no rail without the
        // token, a rail with it.
        let base = StateBuilder::new()
            .conflict((cap - 2) as i8)
            .open_slots(&[(18, "guard-tower")])
            .coins(Player::One, 20)
            .current(Player::One);
        assert!(full_weight(&base.clone().build(), &w, &race, build18) < w.build * RAIL);
        let with_token = base.clone().tokens(Player::One, &["strategy"]).build();
        assert_eq!(full_weight(&with_token, &w, &race, build18), w.build * RAIL);

        // And symmetrically: the *opponent* holding Strategy makes the same
        // card a closing one for them, so taking it away is a rail.
        let st = StateBuilder::new()
            .conflict(-((cap - 2) as i8))
            .tokens(Player::Two, &["strategy"])
            .open_slots(&[(18, "guard-tower")])
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        assert_eq!(
            full_weight(&st, &w, &race, Action::Discard { slot: 18 }),
            w.discard * RAIL
        );
    }

    /// The escalation the Tier-2 tables exist for: the same card is worth more
    /// the further along the race the position already is, and the tables are
    /// flat (inert) at the bottom.
    #[test]
    fn tier_two_escalates_with_race_progress_and_is_flat_early() {
        let w = RolloutWeights::BIASED;
        let race = RaceWeights::MEDIUM;
        let build = Action::Build { slot: 18 };
        // A fresh symbol for a mover who holds `k` others already.
        let symbols = [
            "pharmacist",
            "workshop",
            "scriptorium",
            "apothecary",
            "academy",
        ];
        let mut last = 0.0;
        for k in 0..=5usize {
            let st = StateBuilder::new()
                .built(Player::One, &symbols[..k])
                .open_slots(&[(18, "university")])
                .coins(Player::One, 20)
                .current(Player::One)
                .build();
            assert_eq!(usize::from(st.player(Player::One).distinct_science()), k);
            let got = full_weight(&st, &w, &race, build);
            // Below the activity gate the position is not looked at at all,
            // which is why `sci_push[0..=2]` are all 1.0 in the first place:
            // they can only ever be reached when the *other* race (or the
            // opponent's) has already made the position active.
            let expected = if k >= 3 {
                w.build * race.sci_push[k]
            } else {
                w.build
            };
            assert_eq!(got, expected, "k={k}");
            assert!(got >= last, "k={k} did not escalate");
            last = got;
        }
        // ... and with the position made active by the military race instead,
        // the low end of the science table is reachable and still flat.
        let quiet_science = StateBuilder::new()
            .built(Player::One, &symbols[..2])
            .conflict(4)
            .open_slots(&[(18, "university")])
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        assert!(StepCtx::new(&quiet_science, &race).active);
        assert_eq!(
            full_weight(&quiet_science, &w, &race, build),
            w.build * race.sci_push[2]
        );
        // Flat at the bottom: nothing happens at all below the gate.
        assert_eq!(race.sci_push[0], 1.0);
        assert_eq!(race.sci_push[1], 1.0);
        assert_eq!(race.mil_push[0], 1.0);
        assert_eq!(race.mil_push[1], 1.0);
    }

    /// A quiet early position — neither player near a race — must produce
    /// exactly the weights plain `BIASED` would, with no per-card race lookup
    /// happening at all.
    #[test]
    fn a_quiet_position_is_untouched_by_any_variant() {
        let wonder = duels_core::data::WonderId::from_slug("the-pyramids").unwrap();
        let st = StateBuilder::new()
            .age(1)
            .built(Player::One, &["workshop"])
            .built(Player::Two, &["guard-tower"])
            .conflict(1)
            .wonders(Player::One, &["the-pyramids"])
            .open_slots(&[(18, "university"), (19, "palisade"), (17, "palace")])
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        assert!(!StepCtx::new(&st, &RaceWeights::MEDIUM).active);

        let w = RolloutWeights::BIASED;
        for race in [
            RaceWeights::mild(),
            RaceWeights::MEDIUM,
            RaceWeights::strong(),
            RaceWeights::TIER1_ONLY,
        ] {
            for action in [
                Action::Build { slot: 18 },
                Action::Discard { slot: 19 },
                Action::Build { slot: 17 },
                Action::BuildWonder { slot: 19, wonder },
            ] {
                assert_eq!(
                    full_weight(&st, &w, &race, action).to_bits(),
                    w.weight(&st, action).to_bits(),
                    "{}: {action:?}",
                    race.name()
                );
            }
        }
    }

    /// Effect choices (progress tokens, Mausoleum, destroy, first player) have
    /// no card in a slot to read, so the race layer leaves them alone.
    #[test]
    fn non_card_actions_are_never_multiplied() {
        let st = StateBuilder::new()
            .built(
                Player::One,
                &[
                    "pharmacist",
                    "workshop",
                    "scriptorium",
                    "apothecary",
                    "academy",
                ],
            )
            .conflict(8)
            .current(Player::One)
            .build();
        let ctx = StepCtx::new(&st, &RaceWeights::MEDIUM);
        assert!(ctx.active, "this position is very much active");
        let mut memo = SlotMemo::EMPTY;
        for action in [
            Action::ChooseFirstPlayer {
                player: Player::One,
            },
            Action::ChooseProgressToken {
                token: duels_core::data::TokenId::from_slug("law").unwrap(),
            },
            Action::PickWonder {
                wonder: duels_core::data::WonderId::from_index(0),
            },
            // An empty slot has no card to look up either.
            Action::Build { slot: 0 },
        ] {
            assert_eq!(
                ctx.multiplier(&st, &RaceWeights::MEDIUM, action, &mut memo),
                1.0,
                "{action:?}"
            );
        }
    }

    #[test]
    fn a_playout_always_reaches_a_result() {
        let mut buf = Vec::new();
        let mut wbuf = Vec::new();
        for seed in 0..20u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut state = engine::new_game(seed);
            let result = play_out(
                &mut state,
                &RolloutWeights::SMART,
                &RaceWeights::MEDIUM,
                &mut buf,
                &mut wbuf,
                &mut rng,
                1_000,
            );
            assert!(state.result().is_some(), "seed {seed} did not finish");
            assert_eq!(state.result().unwrap(), result);
            assert!(matches!(
                result.winner(),
                Some(Player::One) | Some(Player::Two) | None
            ));
        }
    }
}
