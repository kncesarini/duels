//! The continuous commitment blend: how committed is each player to a win
//! condition, and what does that do to every evaluation weight?
//!
//! # Why continuous, and not a mode switch
//!
//! An earlier cut of this design had discrete phases ("economy phase",
//! "science phase", ...) with a different weight vector per phase. That is
//! wrong for the same reason `duels-strategy`'s own discrete denial gate was
//! wrong (see its `stance` module docs): the interesting part of this game is
//! the middle, where a player is *partly* committed, and a switch makes the
//! evaluation discontinuous exactly where positions are closest together.
//! Two positions one card apart should not be judged by two different
//! functions.
//!
//! So there are no modes. Each player carries one scalar
//! [`Commitment::c`] in `0..=1` — "how much of my game is now riding on a
//! race" — and every previously-static weight is a smooth function of it.
//!
//! # The scalars
//!
//! ```text
//! prog_sci(p) = distinct(p) / 6
//! c_sci(p)    = M_sci(p)^alpha_m · prog_sci(p)^beta_prog
//!
//! prog_mil(p) = clamp((D_cap − need(p)) / D_cap, 0, 1)
//! c_mil(p)    = 1 − (1 − M_mil(p)) · (1 − prog_mil(p)^mil_prog_exp)
//!
//! c(p)        = 1 − (1 − c_sci(p)) · (1 − c_mil(p))
//! ```
//!
//! `M_sci` and `M_mil` are `duels-strategy`'s calibrated race magnitudes. The
//! `prog_sci^beta_prog` factor is not decoration: `M_sci` alone reads around
//! 0.4-0.5 for a player holding *no symbols at all*, because with three whole
//! ages still to come the supply model genuinely cannot rule the race out.
//! Weighting a fresh position as half-committed to science would be a serious
//! bug, and `prog_sci = 0` kills it exactly. `tests::a_fresh_game_reads_as_
//! uncommitted_for_both_players` is the guard.
//!
//! The two races combine as a probabilistic OR rather than a max or a sum, so
//! a player who is halfway into both is more committed than one halfway into
//! either, without ever exceeding one.
//!
//! # The shape
//!
//! ```text
//! S(c)   = c^n / (c^n + c0_eff^n)                 (n = hill_n, default 4)
//! c0_eff = c0 · clamp(1 + edge_scale · edge, lo, hi)
//! ```
//!
//! A Hill curve: flat near zero, steep through `c0_eff`, saturating after. It
//! behaves like a mode switch where a mode switch is right (deep in a race)
//! and like a blend where a blend is right (the middle), and it is continuous
//! and monotone everywhere. `S(0) = 0` *exactly*, which is what makes the
//! whole apparatus collapse to a plain fixed-weight evaluation in a fresh
//! position — see [`Blend::off`] and the crate docs' "un-blended baseline".
//!
//! `edge` is [`duels_strategy::VpRead::structural_edge`], and the
//! `edge_scale` / clamp band are deliberately the same numbers
//! `duels-strategy` already uses for its own `stakes` multiplier
//! ([`duels_strategy::ThreatWeights::stakes_scale`], `stakes_min`,
//! `stakes_max`), so the two layers agree on what "ahead" is worth. A trailing
//! player gets a *lower* midpoint and so commits to a race more readily; a
//! leading player gets a higher one and banks the lead instead.
//!
//! # Root-fixing
//!
//! Every weight here is computed **once per decision, from the root position**
//! and reused unchanged for every candidate action and every chance outcome.
//! Recomputing them on the post-action state would credit a move twice: once
//! through the term's own contents (which do move with the action, and
//! should) and again through the weight that term is multiplied by. A move
//! that raises `c_sci` would then earn a bonus simply for having raised the
//! importance of the science term, on top of the legitimate gain in the
//! term's value. See [`crate::Root`],
//! `crate::tests::a_committing_move_is_scored_under_the_root_weights_not_its_own`
//! and, for the counting half, `duels-agent-phased`'s
//! `tests::root_weights_are_built_exactly_once_per_choose`.
//!
//! ## ...and where root-fixing stops being right
//!
//! The paragraph above is an argument about **one ply**. It is a good one
//! there: `phased` builds a `Root` from the pre-action position and scores the
//! post-action position against it, so the "credited twice" failure is real
//! and one move away. It is *not* an argument about a search that scores a
//! leaf ten plies deeper, where the root weights are simply stale — a player
//! who has gone from two distinct symbols to five is still being judged by the
//! importance the science term had when they held two. `mcts-eval` builds one
//! `Root` per tree, so that is exactly what happens there.
//!
//! The two readings genuinely conflict, and the shape of `c_sci` says which
//! half of it can be repaired cheaply:
//!
//! ```text
//! c_sci = M_sci^alpha_m · (distinct / 6)^beta_prog
//!         \___________/   \___________________/
//!          root-fixed        can be re-read
//! ```
//!
//! [`Commitment::m_sci_alpha`] keeps the first factor, so
//! [`TermWeights::science_at`] can re-read the second from the position being
//! scored for the price of one `distinct_science()` and two `powf`s — against
//! rebuilding a whole `Root`, which costs about six and a half `evaluate`s and
//! is unaffordable per leaf.
//!
//! **This is an approximation, and knowing where it is loose matters.**
//! `M_sci` is [`duels_strategy::ScienceRead::magnitude`] — the probability
//! this player completes six symbols — which is itself a function of how many
//! they hold (`missing = 6 − distinct` is one of its inputs). It is not a
//! price in the sense a `Root` is allowed to cache prices. So "the root's
//! magnitude with the leaf's progress" is not a clean factorisation; it is a
//! *conservative* one. `M_sci` rises with `distinct`, so a leaf that has
//! advanced the race reads as somewhat less committed than a freshly-built
//! `Root` would say, and never as more. That is the right direction for an
//! error to point in, but it is an error.
//!
//! Because of that — and because at one ply the root-fixed reading is the
//! deliberate, tested behaviour rather than a defect —
//! [`ScienceProgress::Leaf`] is **opt-in** and [`ScienceProgress::Root`]
//! remains the default. See [`ScienceProgress`].

use duels_core::data;
use duels_strategy::{MilitaryRead, ScienceRead, ThreatWeights};

/// Distinct scientific symbols that win the game outright.
const SYMBOLS_TO_WIN: f64 = 6.0;

/// Where the science-ladder weight's *progress* factor is read from.
///
/// The magnitude factor `M_sci^alpha_m` is root-fixed either way — see the
/// module docs' "where root-fixing stops being right" for why only half of
/// `c_sci` can be re-read cheaply, and for the sense in which doing so is an
/// approximation rather than a strictly better answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScienceProgress {
    /// The root position's distinct-symbol count, baked into
    /// [`Commitment::c_sci`] when the [`crate::Root`] was built and reused for
    /// every state scored against it.
    ///
    /// The default, and correct at one ply: `crate::evaluate` scores a
    /// *post-action* state against a `Root` read *pre-action*, so a weight
    /// that moved with the action would credit a committing move twice.
    Root,
    /// The distinct-symbol count of the state being scored, against the
    /// root's magnitude.
    ///
    /// For a search that prices leaves at depth, where the root's count is
    /// not stale by one move but by ten. Costs one `distinct_science()` and
    /// two `powf`s per `player_value`.
    Leaf,
}

/// Tunables for the commitment blend.
///
/// Every judgement call the blend makes lives here rather than as a literal in
/// the code, so an arena sweep can fit it without touching any logic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blend {
    /// When false, `S(c)` is zero for every input and every weight sits at
    /// its un-blended value. See [`Blend::off`].
    pub enabled: bool,
    /// Exponent on the science race magnitude in `c_sci`. Below one, so a
    /// merely plausible race already counts for something.
    pub alpha_m: f64,
    /// Exponent on `distinct / 6` in `c_sci`. Above one, so holding one or
    /// two symbols is not yet a commitment — this is the factor that stops a
    /// fresh position reading as science-committed.
    pub beta_prog: f64,
    /// Exponent on `(D_cap − need) / D_cap` in `c_mil`.
    pub mil_prog_exp: f64,
    /// The Hill exponent `n`. Larger is steeper; 1 would be a plain
    /// saturating curve and the design specifically wants something that
    /// behaves like a switch deep in a race.
    pub hill_n: f64,
    /// The nominal midpoint of the Hill curve: the commitment at which a
    /// player with no structural edge is exactly half committed.
    pub c0: f64,
    /// How fast the effective midpoint moves with
    /// [`duels_strategy::VpRead::structural_edge`]. Defaults to
    /// [`ThreatWeights::stakes_scale`].
    pub edge_scale: f64,
    /// Lower and upper bound on the midpoint multiplier. Defaults to
    /// [`ThreatWeights::stakes_min`] / [`ThreatWeights::stakes_max`].
    pub edge_clamp: (f64, f64),
    /// What fraction of the victory-point projection's weight survives at
    /// full commitment.
    pub floor_vp: f64,
    /// What fraction of the general coin-liquidity weight survives at full
    /// commitment.
    pub floor_liq: f64,
    /// What fraction of the development weight survives at full commitment.
    /// The lowest floor of the set: a player one card from scientific
    /// supremacy should not care what their city produces.
    pub floor_dev: f64,
    /// What the science-ladder weight is multiplied by at full *science*
    /// commitment.
    pub boost_sci: f64,
    /// What the military position weight is multiplied by at full *military*
    /// commitment.
    pub boost_mil: f64,
    /// What fraction of the race-card-liquidity weight applies at zero
    /// commitment. This is the one term that *rises* with commitment, so this
    /// is a floor at the bottom of the curve rather than at the top.
    pub floor_race_liq: f64,
    /// What fraction of the economy weights survive at full commitment.
    pub floor_econ: f64,
    /// Whether the science-ladder weight's progress factor is read from the
    /// root or from the state being scored. See [`ScienceProgress`].
    pub science_progress: ScienceProgress,
}

impl Default for Blend {
    fn default() -> Self {
        let t = ThreatWeights::default();
        Self {
            enabled: true,
            alpha_m: 0.5,
            beta_prog: 2.0,
            mil_prog_exp: 2.0,
            hill_n: 4.0,
            c0: 0.30,
            edge_scale: t.stakes_scale,
            edge_clamp: (t.stakes_min, t.stakes_max),
            floor_vp: 0.30,
            floor_liq: 0.20,
            floor_dev: 0.15,
            boost_sci: 1.5,
            boost_mil: 1.5,
            floor_race_liq: 0.30,
            floor_econ: 0.25,
            science_progress: ScienceProgress::Root,
        }
    }
}

impl Blend {
    /// The blend switched off: `S(c) == 0.0` for every input, so every weight
    /// sits at its un-blended value and the agent is a plain fixed-weight
    /// 1-ply evaluator.
    ///
    /// This is the escape hatch the crate's bit-identity test uses: with it,
    /// the weight vector must be bit-identical to the one a genuinely
    /// uncommitted position (no symbols held, pawn centred) produces with the
    /// blend switched *on*. See
    /// `crate::tests::the_blend_off_and_a_zero_commitment_position_agree_bit_for_bit`.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// The Hill curve `S(c)`, given an already-adjusted midpoint.
    ///
    /// Exactly zero at `c == 0` — not approximately — which is the property
    /// the un-blended baseline rests on. Non-decreasing in `c` for `c >= 0`.
    #[inline]
    pub fn shape(&self, c: f64, c0_eff: f64) -> f64 {
        if !self.enabled || c <= 0.0 {
            return 0.0;
        }
        if c0_eff <= 0.0 {
            return 1.0;
        }
        let a = c.powf(self.hill_n);
        let b = c0_eff.powf(self.hill_n);
        let denom = a + b;
        if denom <= 0.0 || !denom.is_finite() {
            return 0.0;
        }
        (a / denom).clamp(0.0, 1.0)
    }

    /// The effective Hill midpoint for a player with this structural edge.
    ///
    /// The multiplier is the same `clamp(1 + scale × edge, lo, hi)` shape
    /// `duels-strategy` uses for its `stakes`, with the same defaults.
    #[inline]
    pub fn midpoint(&self, structural_edge: f64) -> f64 {
        let mult =
            (1.0 + self.edge_scale * structural_edge).clamp(self.edge_clamp.0, self.edge_clamp.1);
        self.c0 * mult
    }
}

/// How committed one player is to each win condition, and to any of them.
///
/// Computed once per decision, from the root position, for both players.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Commitment {
    /// `M_sci^alpha_m × (distinct / 6)^beta_prog`.
    pub c_sci: f64,
    /// The magnitude half of [`Commitment::c_sci`] on its own:
    /// `M_sci^alpha_m`, or exactly `0.0` when the race is dead
    /// (`M_sci <= 0`).
    ///
    /// Kept so [`Commitment::c_sci_at`] can substitute a different progress
    /// factor without rebuilding the science read. `c_sci_at(root distinct)`
    /// reproduces `c_sci` bit for bit — the two are the same two factors
    /// multiplied in the same order, which is what
    /// `tests::the_magnitude_half_reconstructs_c_sci_bit_for_bit` pins.
    pub m_sci_alpha: f64,
    /// `1 − (1 − M_mil) × (1 − prog_mil^mil_prog_exp)`.
    pub c_mil: f64,
    /// The two combined as a probabilistic OR.
    pub c: f64,
    /// The Hill midpoint in force for this player, after the structural-edge
    /// adjustment.
    pub c0_eff: f64,
    /// `S(c)`.
    pub s: f64,
    /// `S(c_sci)`.
    pub s_sci: f64,
    /// `S(c_mil)`.
    pub s_mil: f64,
}

impl Commitment {
    /// Read one player's commitment off their race reads.
    ///
    /// `structural_edge` is that player's own
    /// [`duels_strategy::VpRead::structural_edge`] (it is antisymmetric, so
    /// the opponent's is its negation).
    pub fn of(
        science: &ScienceRead,
        military: &MilitaryRead,
        structural_edge: f64,
        blend: &Blend,
    ) -> Commitment {
        let prog_sci = (f64::from(science.distinct) / SYMBOLS_TO_WIN).clamp(0.0, 1.0);
        // The magnitude half, kept on its own so `c_sci_at` can re-read the
        // progress half from a deeper position. Zero for a dead race, which is
        // what makes `m_sci_alpha <= 0.0` an exact stand-in for
        // `science.magnitude <= 0.0` in the guard below.
        let m_sci_alpha = if science.magnitude <= 0.0 {
            0.0
        } else {
            science.magnitude.clamp(0.0, 1.0).powf(blend.alpha_m)
        };
        let c_sci = if prog_sci <= 0.0 || science.magnitude <= 0.0 {
            // Written out rather than left to `powf`, so the monotonicity
            // property "no symbols held, or a dead race, means no science
            // commitment" holds exactly rather than to within a rounding.
            0.0
        } else {
            m_sci_alpha * prog_sci.powf(blend.beta_prog)
        };

        let d_cap = f64::from(data::military().capital_distance);
        let prog_mil = if d_cap <= 0.0 {
            0.0
        } else {
            ((d_cap - f64::from(military.need)) / d_cap).clamp(0.0, 1.0)
        };
        let m_mil = military.magnitude.clamp(0.0, 1.0);
        let c_mil = 1.0 - (1.0 - m_mil) * (1.0 - prog_mil.powf(blend.mil_prog_exp));

        let c = 1.0 - (1.0 - c_sci) * (1.0 - c_mil);
        let c0_eff = blend.midpoint(structural_edge);

        Commitment {
            c_sci,
            m_sci_alpha,
            c_mil,
            c,
            c0_eff,
            s: blend.shape(c, c0_eff),
            s_sci: blend.shape(c_sci, c0_eff),
            s_mil: blend.shape(c_mil, c0_eff),
        }
    }

    /// `c_sci` re-read for a different distinct-symbol count, against this
    /// commitment's root-fixed magnitude.
    ///
    /// Bit-identical to [`Commitment::c_sci`] when `distinct` is the count the
    /// root held. See the module docs for what the root-fixed magnitude
    /// factor costs in accuracy.
    #[inline]
    pub fn c_sci_at(&self, distinct: u8, blend: &Blend) -> f64 {
        let prog = (f64::from(distinct) / SYMBOLS_TO_WIN).clamp(0.0, 1.0);
        if prog <= 0.0 || self.m_sci_alpha <= 0.0 {
            0.0
        } else {
            self.m_sci_alpha * prog.powf(blend.beta_prog)
        }
    }
}

/// The per-term multipliers one player's commitment produces.
///
/// Every field is a pure multiplier on that term's base weight from
/// [`crate::EvalWeights`]; at zero commitment every one of them is exactly
/// `1.0` except [`TermWeights::race_liquidity`], the one term that rises with
/// commitment and therefore starts at its own floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TermWeights {
    /// The commitment this was derived from, kept for diagnostics
    /// (`examples/watch_blend.rs`) and for the tests.
    pub commitment: Commitment,
    /// Multiplier on the victory-point projection.
    pub vp: f64,
    /// Multiplier on general coin liquidity (`floor(coins / 3)`).
    pub liquidity: f64,
    /// Multiplier on the development term.
    pub development: f64,
    /// Multiplier on the science ladder. Driven by `S(c_sci)` specifically,
    /// not by the combined commitment: it is the *science* race that makes
    /// symbols matter more.
    pub science: f64,
    /// Multiplier on military board position. Driven by `S(c_mil)`. The
    /// escalating military *urgency* term is deliberately not scaled at all —
    /// see [`crate::EvalWeights::military_endgame_urgency`].
    pub military: f64,
    /// Multiplier on race-card liquidity. Rises with commitment.
    pub race_liquidity: f64,
    /// Multiplier on the economy terms (coin safety floor, trade-price
    /// vulnerability).
    pub economy: f64,
}

impl TermWeights {
    /// Derive the multipliers from one player's commitment.
    pub fn of(commitment: Commitment, blend: &Blend) -> TermWeights {
        // Falling with commitment: `1` at S = 0, `floor` at S = 1.
        let fall = |floor: f64, s: f64| 1.0 - (1.0 - floor) * s;
        // Rising with commitment: `1` at S = 0, `boost` at S = 1.
        let rise = |boost: f64, s: f64| 1.0 + (boost - 1.0) * s;

        TermWeights {
            vp: fall(blend.floor_vp, commitment.s),
            liquidity: fall(blend.floor_liq, commitment.s),
            development: fall(blend.floor_dev, commitment.s),
            science: rise(blend.boost_sci, commitment.s_sci),
            military: rise(blend.boost_mil, commitment.s_mil),
            // The inverse shape: `floor` at S = 0, `1` at S = 1.
            race_liquidity: blend.floor_race_liq + (1.0 - blend.floor_race_liq) * commitment.s,
            economy: fall(blend.floor_econ, commitment.s),
            commitment,
        }
    }

    /// The science multiplier for a state holding `distinct` symbols, with the
    /// magnitude half of the commitment still read from the root.
    ///
    /// Bit-identical to [`TermWeights::science`] when `distinct` is the root's
    /// own count — the same two factors, the same `shape`, the same `rise`.
    /// Read by `crate::player_value` only under
    /// [`ScienceProgress::Leaf`].
    #[inline]
    pub fn science_at(&self, distinct: u8, blend: &Blend) -> f64 {
        let c_sci = self.commitment.c_sci_at(distinct, blend);
        1.0 + (blend.boost_sci - 1.0) * blend.shape(c_sci, self.commitment.c0_eff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shape_is_exactly_zero_at_zero_and_monotone_after() {
        let b = Blend::default();
        let c0 = b.midpoint(0.0);
        assert_eq!(b.shape(0.0, c0).to_bits(), 0.0f64.to_bits());

        let mut last = 0.0;
        for i in 0..=200 {
            let c = f64::from(i) / 200.0;
            let s = b.shape(c, c0);
            assert!((0.0..=1.0).contains(&s), "S({c}) = {s} out of range");
            assert!(s >= last, "S is not monotone at c = {c}: {s} < {last}");
            last = s;
        }
        assert!(
            last > 0.99,
            "S(1) should be close to saturation, got {last}"
        );
        // The midpoint really is the midpoint.
        assert!((b.shape(c0, c0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_trailing_player_commits_more_easily_than_a_leading_one() {
        let b = Blend::default();
        let behind = b.midpoint(-10.0);
        let level = b.midpoint(0.0);
        let ahead = b.midpoint(10.0);
        assert!(behind < level && level < ahead);
        // ...which means the same commitment reads as more committed.
        let c = 0.3;
        assert!(b.shape(c, behind) > b.shape(c, level));
        assert!(b.shape(c, level) > b.shape(c, ahead));
        // The clamp band matches duels-strategy's own stakes band.
        let t = ThreatWeights::default();
        assert!((b.midpoint(-1000.0) - b.c0 * t.stakes_min).abs() < 1e-12);
        assert!((b.midpoint(1000.0) - b.c0 * t.stakes_max).abs() < 1e-12);
    }

    #[test]
    fn switching_the_blend_off_zeroes_the_shape_everywhere() {
        let off = Blend::off();
        for i in 0..=20 {
            let c = f64::from(i) / 20.0;
            assert_eq!(off.shape(c, off.midpoint(0.0)).to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn every_multiplier_is_one_at_zero_commitment_except_race_liquidity() {
        let b = Blend::default();
        let zero = Commitment {
            c_sci: 0.0,
            m_sci_alpha: 0.0,
            c_mil: 0.0,
            c: 0.0,
            c0_eff: b.midpoint(0.0),
            s: 0.0,
            s_sci: 0.0,
            s_mil: 0.0,
        };
        let w = TermWeights::of(zero, &b);
        for (name, v) in [
            ("vp", w.vp),
            ("liquidity", w.liquidity),
            ("development", w.development),
            ("science", w.science),
            ("military", w.military),
            ("economy", w.economy),
        ] {
            assert_eq!(v.to_bits(), 1.0f64.to_bits(), "{name} = {v}, expected 1.0");
        }
        assert_eq!(w.race_liquidity.to_bits(), b.floor_race_liq.to_bits());
    }

    #[test]
    fn full_commitment_reaches_every_floor_and_boost() {
        let b = Blend::default();
        let full = Commitment {
            c_sci: 1.0,
            m_sci_alpha: 1.0,
            c_mil: 1.0,
            c: 1.0,
            c0_eff: b.midpoint(0.0),
            s: 1.0,
            s_sci: 1.0,
            s_mil: 1.0,
        };
        let w = TermWeights::of(full, &b);
        assert!((w.vp - b.floor_vp).abs() < 1e-12);
        assert!((w.liquidity - b.floor_liq).abs() < 1e-12);
        assert!((w.development - b.floor_dev).abs() < 1e-12);
        assert!((w.science - b.boost_sci).abs() < 1e-12);
        assert!((w.military - b.boost_mil).abs() < 1e-12);
        assert!((w.race_liquidity - 1.0).abs() < 1e-12);
        assert!((w.economy - b.floor_econ).abs() < 1e-12);
    }

    /// A hand-built [`Commitment`] is not enough for this one: the claim is
    /// about what [`Commitment::of`] stores, so it has to go through that
    /// function. `duels-strategy`'s reads are only constructible from a real
    /// position, so `crate::tests` owns the whole-game version
    /// (`the_leaf_progress_option_is_off_by_default_and_identical_at_the_root`);
    /// this one covers the arithmetic over the parameter space directly.
    #[test]
    fn the_magnitude_half_reconstructs_c_sci_bit_for_bit() {
        let b = Blend::default();
        // Every `(magnitude, distinct)` pair the game can present, including
        // the dead race and the no-symbols position that `c_sci`'s guard
        // singles out.
        for mag_i in 0..=20 {
            let magnitude = f64::from(mag_i) / 20.0;
            for distinct in 0u8..=7 {
                let prog = (f64::from(distinct) / SYMBOLS_TO_WIN).clamp(0.0, 1.0);
                let m_sci_alpha = if magnitude <= 0.0 {
                    0.0
                } else {
                    magnitude.clamp(0.0, 1.0).powf(b.alpha_m)
                };
                let expected = if prog <= 0.0 || magnitude <= 0.0 {
                    0.0
                } else {
                    m_sci_alpha * prog.powf(b.beta_prog)
                };
                let c = Commitment {
                    c_sci: expected,
                    m_sci_alpha,
                    c_mil: 0.0,
                    c: expected,
                    c0_eff: b.midpoint(0.0),
                    s: 0.0,
                    s_sci: b.shape(expected, b.midpoint(0.0)),
                    s_mil: 0.0,
                };
                assert_eq!(
                    c.c_sci_at(distinct, &b).to_bits(),
                    expected.to_bits(),
                    "c_sci_at disagreed at magnitude {magnitude}, distinct {distinct}"
                );
                // ...and so does the multiplier built on top of it.
                let w = TermWeights::of(c, &b);
                assert_eq!(
                    w.science_at(distinct, &b).to_bits(),
                    w.science.to_bits(),
                    "science_at disagreed at magnitude {magnitude}, distinct {distinct}"
                );
            }
        }
    }

    /// The non-vacuity half: re-reading progress upward really does raise the
    /// multiplier, and a dead race stays dead however many symbols the leaf
    /// holds.
    #[test]
    fn re_reading_progress_moves_the_science_multiplier_but_not_a_dead_race() {
        let b = Blend::default();
        let c0 = b.midpoint(0.0);
        // A live race read at two distinct symbols.
        let m_sci_alpha = 0.6f64.powf(b.alpha_m);
        let c_sci = m_sci_alpha * (2.0 / SYMBOLS_TO_WIN).powf(b.beta_prog);
        let live = Commitment {
            c_sci,
            m_sci_alpha,
            c_mil: 0.0,
            c: c_sci,
            c0_eff: c0,
            s: b.shape(c_sci, c0),
            s_sci: b.shape(c_sci, c0),
            s_mil: 0.0,
        };
        let w = TermWeights::of(live, &b);
        let mut last = w.science_at(0, &b);
        for distinct in 1u8..=6 {
            let next = w.science_at(distinct, &b);
            assert!(
                next >= last,
                "the multiplier fell going from {} to {distinct} symbols",
                distinct - 1
            );
            last = next;
        }
        assert!(
            w.science_at(5, &b) > w.science * 1.05,
            "five symbols should read as materially more committed than the \
             root's two ({} vs {})",
            w.science_at(5, &b),
            w.science
        );
        assert!(last <= b.boost_sci + 1e-12);

        // A dead race: `m_sci_alpha == 0`, so no amount of leaf progress
        // resurrects it.
        let dead = Commitment {
            c_sci: 0.0,
            m_sci_alpha: 0.0,
            ..live
        };
        let dw = TermWeights::of(dead, &b);
        for distinct in 0u8..=7 {
            assert_eq!(dw.science_at(distinct, &b).to_bits(), 1.0f64.to_bits());
        }
    }
}
