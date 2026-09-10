//! `duels-eval`: a hand-crafted position evaluation whose every weight is a
//! continuous function of how committed each player is to a win condition.
//!
//! # Where this sits
//!
//! This crate is a **library, not an agent**. It owns [`Config`], [`Root`],
//! [`evaluate`] and [`expected_value`] — everything needed to put a
//! victory-point-scale number on a position — and nothing that decides a move.
//! `duels-agent-phased` is the 1-ply agent built on it: it samples one concrete
//! state per decision, builds one [`Root`], scores every legal action with
//! [`expected_value`] and returns the best. It was this crate's only caller
//! when the evaluation was extracted out of it, and everything below was
//! written while the two were one crate — read "this agent" as "`phased`", and
//! every measurement as one taken with `phased` driving.
//!
//! It exists as its own crate because more than one agent is going to want the
//! evaluation, and this repository's rule is that **no agent crate depends on
//! another agent crate** (see `CLAUDE.md`). A shared evaluation therefore has
//! to live below the agents, next to [`duels_strategy`], rather than inside
//! whichever agent happened to build it first.
//!
//! Nothing here is random and nothing here reads a clock: [`Root::new`],
//! [`evaluate`] and [`expected_value`] are pure functions of the state handed
//! to them, which is why this crate depends on `duels-core` and
//! `duels-strategy` and on nothing else.
//!
//! # What this crate is trying to fix
//!
//! `duels-agent-greedy-ev` was the agent this crate was written to replace. It
//! has since been retired from the roster (#60), so the tense below is
//! historical, but the two defects it names are what every weight here is
//! shaped around and are worth keeping written down.
//!
//! `greedy-ev` resolved uncertainty correctly — it averaged over
//! [`engine::chance_outcomes`] instead of committing to one sampled guess —
//! and that machinery is copied here unchanged. What it did *not* do is
//! change its mind about what matters. Its weight vector is the same on turn
//! one and on turn fifty, for a player two symbols from scientific supremacy
//! and for one who has never taken a green card. Two specific consequences a
//! strong human player pointed out:
//!
//! * it prices almost none of a player's **own production**. A brown or grey
//!   card is, to `greedy-ev`, a card with no points and no shields — a bill.
//!   In reality it is every future card's discount, and grey is the scarcest
//!   of the lot (Ages I and II carry two grey cards each and Age III carries
//!   none at all — verified against `data/cards.json`, not assumed).
//! * it has no idea that a player at four distinct symbols is in a different
//!   game from a player at two. Ages I and II each carry one copy of the same
//!   four symbols and Age III carries the other two, twice each, so after Age
//!   I *every* Age II green completes a pair, and the sixth symbol needs an
//!   Age-III-only symbol or the Law token. A linear "points per symbol" term
//!   cannot see any of that.
//!
//! # The design: one continuous blend, no modes
//!
//! Each player carries a commitment scalar `c ∈ [0, 1]` built from
//! `duels-strategy`'s calibrated race magnitudes, and a Hill curve `S(c)`
//! turns it into a per-term multiplier. Points, coin liquidity, development
//! and economy all fade as commitment rises; the science ladder and military
//! position sharpen; race-card liquidity rises. There are deliberately **no
//! discrete modes** — see [`blend`] for why, and for the arithmetic.
//!
//! Every weight is computed **once per decision from the root position** and
//! reused for every candidate action and every chance outcome. Term
//! *contents* are measured on the post-action, expectation-averaged state,
//! exactly as `greedy-ev` measures its own; only the weights are pinned. That
//! asymmetry is the point: a move that raises `c_sci` should be credited once,
//! through the science term's value going up, not twice, through the weight on
//! that term going up as well.
//!
//! # The evaluation
//!
//! ```text
//! score(state, me) = terminal_result                          (Rail A)
//!                  | ±imminent                                (Rails B/C/C', see [`rails`])
//!                  | Σ_k [ T_k(me) − T_k(opp) ] + A(action) + M(state, me)
//! T_k(p)           = w_k(S(c(p))) × base_k × raw_k(state, p)
//! A(action)        = w_deny × deny_scale × duels_strategy::deny_vp(action)
//! M(state, me)     = ±λ × menu(next mover)          (see [`menu`])
//! ```
//!
//! The first two lines are *rails*, not terms: they replace the weighted sum
//! rather than adding to it, because the question they answer — is this
//! position already decided, and for whom? — is not commensurable with a few
//! victory points of city quality. See [`rails`].
//!
//! `M` is the one term that is *not* read per player and differenced: it
//! prices what the position hands to whoever moves next, which is one player,
//! not both. It is still antisymmetric — swapping `me` flips its sign — so the
//! whole evaluation remains zero-sum.
//!
//! Terms are read **per player** and then differenced, rather than as a
//! single difference under one weight, so that each side's science ladder
//! carries *that side's* science commitment. A science race is more important
//! to whichever player is actually in it; weighting the opponent's ladder by
//! the mover's commitment would read a rising opposing race as less
//! interesting the less interested the mover is in science, which is exactly
//! backwards. At equal commitment the two formulations coincide.
//!
//! `A` is a function of the root position and the action alone —
//! [`duels_strategy::delta_m`] prices what a move does to the opponent's race
//! magnitudes from the root stance, and never looks at the post-action state —
//! so it is genuinely outcome-independent and is added once, outside the
//! per-outcome loop. (Adding it inside would give a bit-identical answer,
//! since the outcome probabilities sum to one; once is simply cheaper.)
//!
//! # What "the un-blended baseline" means here
//!
//! This agent's term set is not `greedy-ev`'s, so there is no other crate it
//! can be bit-compared against. What *is* pinned exactly is the property the
//! whole blend rests on: `S(0) = 0`, so a position in which neither player is
//! committed to anything is scored by a plain, fixed weight vector — every
//! multiplier exactly `1.0`, except race-card liquidity, the one term that
//! rises with commitment and therefore sits at its own floor. [`Blend::off`]
//! forces that state for every position, and
//! `tests::the_blend_off_and_a_zero_commitment_position_agree_bit_for_bit`
//! asserts the two agree bit for bit.
//!
//! # The measurement history
//!
//! Every weight in this crate was arrived at by a numbered **round**: one PR
//! that states a hypothesis, builds it behind a [`Config`] option, measures it
//! against the previous generation over paired, seat-swapped games, and then
//! keeps it or writes down why it was not kept. Rounds two through eleven —
//! what each tried, what it measured, what was kept and what was reverted,
//! with the Elo figures and the PR each landed under — live in
//! **`docs/eval-rounds/`**, one file per round, plus
//! `cross-round-measurements.md` for the tables that span rounds (the early
//! Elo and leave-one-out tables, the `mcts-uct` bar, the `military_band`
//! sweep, the per-decision cost breakdown and the Age I take profile).
//! `docs/eval-rounds/README.md` is the index.
//!
//! That history used to sit here, and it was about 2,700 lines of doc comment
//! above the code it explains. It was relocated rather than summarised:
//! nothing was dropped, and the files are the doc comments verbatim. What
//! stays here is the crate's *current* design — the sections above, and the
//! invariant below.
//!
//! Two conventions the round files assume, worth knowing before reading one:
//!
//! * **[`Config::v1`] through [`Config::v9`] are generations, not version
//!   numbers.** Each round that changes a default adds a `vN` that reproduces
//!   the *previous* generation bit for bit, with a `tests/vN_identity.rs` that
//!   asserts it. That is what lets one binary benchmark a round against the
//!   round before it (`phased:base=v3`, `mcts-eval:eval=v9`).
//! * **Every Elo figure is paired and seat-swapped** on at least two disjoint
//!   seed ranges, and a change whose ranges disagree in sign is reported as
//!   neutral rather than as the flattering half.
//!
//! # Public information only
//!
//! `phased` samples one concrete
//! [`GameState`] per decision purely as a vehicle for the engine's chance API.
//! Everything downstream — the commitment scalars, every weight, and the full
//! evaluation of every legal action — reads only publicly-known information,
//! so none of it depends on which world that throwaway sample invented.
//! `tests/determinization_invariance.rs` asserts that bit for bit over real
//! positions from real games.

#![deny(clippy::disallowed_methods)]
#![warn(missing_docs)]

pub mod blend;
pub mod menu;
pub mod rails;
pub mod terms;

use duels_core::engine;
use duels_core::scoring::{self, GameResult};
use duels_core::{Action, GameState, Player};
use duels_strategy::{deny_vp, stance_in, Context, PriorWeights, Stance, ThreatWeights, VpWeights};

pub use blend::{Blend, Commitment, ScienceProgress, TermWeights};
pub use menu::{ChainTable, MenuOptions, MenuTables, TakeContext, TakeValue};
pub use rails::{rail_owner, rail_value, RailModel};
pub use terms::{DevSupply, GuildTable, MilSmoothing, WonderBudget, MAX_UNITS};

/// How [`menu::TakeValue`] prices the shields on a red card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MenuShieldPricing {
    /// `k x` the one-sided local slope of this player's own band — the
    /// round-two behaviour, kept so [`Config::v2`] reproduces it bit for bit.
    /// Inconsistent with the evaluation it feeds, which prices the same shield
    /// *differenced* across both players and under both players' weights.
    OneSided,
    /// The exact finite difference of what the main evaluation would move, via
    /// [`terms::military_shield_delta`], Strategy token included.
    #[default]
    Differenced,
}

/// Whether [`menu::TakeValue`] prices a guild card's majority scoring.
///
/// Every guild in the base game prints zero victory points and zero coins and
/// scores entirely through `points_by_majority` / `coins_by_majority`, which
/// [`duels_core::scoring::breakdown`] reads at scoring time and the menu's
/// pricer did not read at all. A face-up guild therefore priced out at
/// `−cost × coin_marginal` — strictly negative, in every position — so `phased`
/// never fought for a guild and never denied one to an opponent who was
/// collecting the colour it counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuildPricing {
    /// A guild is worth its printed points and coins, both of which are zero —
    /// the pre-existing behaviour, reproduced bit for bit by [`Config::v4`].
    Unpriced,
    /// `per_vp · Ĝ(t) + per_coin · live(t) · coin_marginal`, against the
    /// root-fixed projections in [`terms::GuildTable`].
    ///
    /// **The default**, on the evidence in the crate docs.
    #[default]
    Projected,
}

/// Whether [`menu::TakeValue`] prices the coins a card pays *per building its
/// taker already owns*.
///
/// Five commercial cards — all of them Age III — print no coins at all and
/// instead pay `amount_per_unit ×` a count of the builder's own city, straight
/// through [`duels_core::data::Card::coins_per_own`]: three coins per
/// manufactured good, two per raw material, one per military building, one per
/// commercial building, two per constructed wonder. This is exactly the shape
/// of the guild bug round five fixed — [`menu::TakeValue::free_value`] starts a
/// card's value from `def.victory_points` and `def.coins`, and `def.coins` is
/// zero for all five — except that here the payout is not a projection but a
/// count that is already on the table, so there is nothing to estimate.
///
/// The main evaluation was never wrong about these cards: the coins arrive in
/// the post-action state and [`terms::coin_points`] reads them. It was the
/// *menu* that could not see them, and so under-valued what the next mover's
/// turn was worth and how much taking one of these away from them was worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CountPricing {
    /// A card is worth its printed coins, which for these five is zero — the
    /// pre-existing behaviour, reproduced bit for bit by [`Config::v6`].
    #[default]
    Unpriced,
    /// `coins_per_own × count(taker's city) × coin_marginal`, read off the
    /// player the menu is pricing for.
    ///
    /// **Off by default — an honest negative.** It is the more correct model
    /// and it measures at nothing: −2.7 / −0.9 Elo as a leave-one-out against
    /// the round-seven default over 3200 games on each of two disjoint seed
    /// ranges. The reason is almost certainly that all five cards are Age III
    /// and the menu term is `λ = 0.6` of one softmax entry, so the blind spot
    /// was real and rarely load-bearing. Kept, with the measurement written
    /// down, rather than enabled on the strength of the argument.
    Counted,
}

/// What [`menu::menu_term`] does when nothing on the board is affordable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MenuFloor {
    /// Return zero — the pre-existing behaviour, reproduced bit for bit by
    /// [`Config::v4`]. A hard floor, and a discontinuous one.
    #[default]
    None,
    /// Add the discard the player can always take: `discard_reward ×
    /// coin_marginal`.
    Discard,
    /// ...and the best unbuilt wonder they can already pay for.
    DiscardAndWonder,
}

/// How the pool statistics in [`terms::DevSupply`] weight a card from an age
/// that has not been dealt yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SupplyModel {
    /// Every undealt card counts once — the pre-existing behaviour,
    /// reproduced bit for bit by [`Config::v4`]. Over-weights Age III's seven
    /// guilds, of which three are dealt, and Ages I and II's twenty-three
    /// cards, of which twenty are.
    #[default]
    Raw,
    /// Each undealt entry is weighted by its own age's dealt fraction: 20/23
    /// for Ages I and II, 17/20 for Age III's plain cards and 3/7 for its
    /// guilds. See [`terms::DevSupply::of_with`].
    Dealt,
}

/// Whether the evaluator finishes a turn that the engine has left mid-effect.
///
/// Four wonders — Circus Maximus, the Statue of Zeus, the Mausoleum and the
/// Great Library — do not finish their own construction. `engine::apply`
/// leaves [`duels_core::state::Pending`] set and
/// `engine::finish_turn` returns early, so the state a candidate
/// `BuildWonder` produces is one in which **the effect has not happened yet**:
/// the card the destroy will take is still in the opponent's city, the
/// Mausoleum's retrieval and the Great Library's token do not exist, and the
/// builder is still `current_player` even though the turn is about to pass.
/// An ordinary `Build` of a green card that completes a science pair leaves
/// the same kind of state ([`duels_core::state::Pending::ProgressToken`]).
///
/// Scoring that state directly credits none of the effect — only the flat "has
/// an effect" bonus in [`terms::wonder_power`] — and reads the mover as moving
/// again, which flips the sign [`menu::menu_term`] puts on the position and
/// stands the rails down entirely ([`rails::rail_owner`] refuses to read a
/// pending state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PendingModel {
    /// Score the pending state as it stands — the pre-existing behaviour,
    /// reproduced bit for bit by [`Config::v3`].
    Unresolved,
    /// Finish the mover's own turn before scoring it: resolve the pending
    /// choice with the engine's own [`duels_core::engine::legal_actions`],
    /// take the option the *resolver* likes best, and score the state that
    /// leaves — including whatever `engine::finish_turn` then
    /// does about passing the turn.
    ///
    /// This is not search against an opponent. Every action in the resolution
    /// belongs to the same player, in the same turn, and the engine already
    /// models the turn as those sequential decisions; this simply stops
    /// scoring a half-applied one.
    ///
    /// **The default**, on the evidence in the crate docs: +3.5 / +5.4 / +6.1 /
    /// +15.6 Elo against `phased:base=v3` over 3200 games on each of four
    /// disjoint seed ranges, and — the reason it was built —
    /// `duels-arena/examples/wonder_audit.rs` showing the four pending-effect
    /// wonders going up 89% of the time they are drafted against 78%, nine
    /// turns earlier for the Statue of Zeus.
    #[default]
    Completed,
}

/// How the evaluation prices a drafted-but-unbuilt wonder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WonderModel {
    /// [`terms::wonder_power`] at a flat `wonder_potential` weight, with no
    /// per-wonder probability that it is ever built — the pre-existing
    /// behaviour, reproduced bit for bit by [`Config::v3`], and **still the
    /// default** after round nine measured the alternative on both consumers.
    #[default]
    Flat,
    /// [`terms::WonderBudget`]: a per-effect price, scaled by the chance the
    /// wonder is built at all given the seven-wonder cap and the decisions the
    /// owner has left.
    Budget,
    /// [`Flat`](WonderModel::Flat)'s per-effect prices — the extra-turn
    /// premium included, unchanged — scaled by [`terms::wonder_p_build`], the
    /// standalone probability that any one of the owner's unbuilt wonders is
    /// ever built. See [`terms::wonder_potential_rationed`].
    ///
    /// This is the *one* channel [`Budget`](WonderModel::Budget) moved that
    /// round four's measurement could not isolate. `Budget` re-prices eight
    /// things at once — several of them known-weak, one of them
    /// `duels_strategy::science::token_value`'s unmeasured constants — and
    /// measured −11.0 / −5.6 / −3.9 Elo as a bundle. Its **rationing** is the
    /// part `examples/wonder_calibration.rs` says is right, so round nine
    /// takes only that and leaves `Flat`'s flat `+3, this wonder does
    /// something` exactly where it is.
    ///
    /// # Off by default — and the sharpest policy-versus-value split this
    /// crate has measured
    ///
    /// At [`EvalWeights::wonder_potential`] `= 1.25` this is worth **+29.7 /
    /// +31.1 / +30.5 / +19.6 / +17.4** Elo to `duels-agent-phased` over 3200
    /// games on each of five disjoint seed ranges, every interval clearing
    /// zero, and **+86 / +56** against `duels-agent-mcts-uct` — and
    /// **−24.4 / −34.6** Elo to `duels-agent-mcts-eval` over 3200 games at
    /// `Nodes(2000)` on two disjoint ranges. It is off because `mcts-eval` is
    /// the consumer that decides; it is *available*, and documented at this
    /// length, because those are not small numbers in either direction.
    ///
    /// Two controls say what the split is about. The **same weight
    /// un-rationed** is worth −73.0 / −75.5 to `phased`, so a hundred Elo
    /// separates one constant with and without the `p_build` factor: the
    /// rationing is what makes a large weight survivable at all. And the
    /// rationing **at the old magnitude** — [`EvalWeights::wonder_p_build_ref`]
    /// at [`terms::OPENING_P_BUILD`], which is the same decay with the opening
    /// scale preserved — is worth −11.8 at `0.5` and +5.5 at `0.65`, so
    /// `phased`'s gain needs the scale and not only the shape.
    ///
    /// That is the whole conflict. A 1-ply argmax is invariant to the term's
    /// magnitude and cares only that the *ordering* of draft and build
    /// candidates improves. A leaf value is not: raising one term two and a
    /// half times re-balances what the logistic can see, and the crate docs
    /// record that the evaluation half of `mcts-eval`'s blend is there to
    /// supply *civilian-score* judgement. Round nine found no formulation that
    /// paid both — see the round-nine section of the crate docs for the four
    /// that were tried.
    ///
    /// # Every Elo figure above predates the frozen-`p_build` fix
    ///
    /// All of them were measured while `p_build` was read once from the root
    /// position and reused for every state scored against it — a defect, now
    /// fixed (see [`terms::wonder_potential_rationed`]). `Flat` is the default
    /// and is untouched, so nothing on the default path moved and nothing else
    /// in this crate's measured history is affected; but this model is a
    /// different function than it was when those numbers were taken.
    ///
    /// The `mcts-eval` figure **was** re-measured, against a matched pre-fix
    /// control that reproduced the old numbers to the decimal: `-19.7` and
    /// `-9.6` in place of `-24.4` and `-34.6`, so the fix is worth about
    /// `+15` Elo here and the verdict — off by default — is unchanged. The
    /// `phased` sweep and the `mcts-uct` / `alphabeta` transfer checks were
    /// **not** re-measured; treat those as the shape of the policy/value
    /// split rather than as this option's current strength, and re-run the
    /// sweep before quoting them again. The full table is in the round-nine
    /// section of the crate docs.
    Rationed,
}

/// How the evaluation prices the conflict pawn's position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MilitaryModel {
    /// A flat reward per step of pawn position — the original.
    Legacy,
    /// The real end-of-game scoring table and the loot tokens, as step
    /// functions smoothed by how many shields are still in play. See
    /// [`terms::MilSmoothing`].
    #[default]
    Band,
}

/// How the evaluation prices a coin pile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoinModel {
    /// Three separate terms: `floor(coins / 3)`, a capped race-liquidity
    /// bonus, and a penalty for falling below a safety floor — the original.
    Legacy,
    /// One smooth function: a linear points channel plus a saturating
    /// liquidity channel. See [`terms::coin_points`] / [`terms::coin_liquidity`].
    #[default]
    Smooth,
}

/// How the evaluation prices a player's exposure to the resource market.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EconomyModel {
    /// The average per-unit trade price the player faces — the original.
    Legacy,
    /// The coins they still expect to *pay*, over the pool that is actually
    /// coming. Read per player and differenced, this is where monopoly value
    /// comes from. See [`terms::resource_bill`].
    #[default]
    Bill,
}

/// The opponent-menu weight rounds one through nine all shipped, before round
/// ten's joint regression fit cut it to [`MenuWeights::default`]'s `0.408`.
///
/// Kept as a named constant so [`Config::v9`] restores it exactly and
/// `tests/v9_identity.rs` can pin it, rather than the round-nine value being
/// recoverable only from this file's history — exactly as
/// [`SCIENCE_LADDER_V7`] and [`WIN_PROBABILITY_TEMPERATURE_V7`] do for round
/// eight's two changes.
pub const MENU_LAMBDA_V9: f64 = 0.6;

/// The knobs of the opponent-menu term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MenuWeights {
    /// Weight on the whole term. Zero switches it off entirely and restores
    /// the pre-existing evaluation bit for bit.
    ///
    /// **`0.408` since round ten; [`MENU_LAMBDA_V9`] is the `0.6` rounds one
    /// through nine shipped.** `0.6` was never fitted — it was the value the
    /// term was introduced at, chosen because the menu's units are victory
    /// points and a little over half of one seemed the honest share of a
    /// take the opponent has not made yet. Round ten's joint regression over
    /// this crate's eighteen scalar weights put it at `0.407872`, and this is
    /// the one coefficient where every instrument the round had agreed: the
    /// clean fit target, the contaminated one, `phased` on three disjoint seed
    /// ranges (+21.8 / +15.3 / +12.2 Elo) and `mcts-eval` pooled over two
    /// (+18.1, 95% CI [+2.9, +33.3]).
    ///
    /// **The two confirmation A/Bs disagree by instrument, and there is no
    /// single Elo number for this change.** `phased` against
    /// `phased:base=v9` over 30,000 `Nodes(1)` games on three further
    /// disjoint ranges reads `+11.9`, 95% CI `[+8.0, +15.9]`, all three ranges
    /// positive and each excluding zero — the fitting thread's `phased`
    /// figures reproduce. `mcts-eval` against `mcts-eval:eval=v9` over 4000
    /// `Nodes(2000)` games on four more reads only `+5.0`, 95% CI
    /// `[-5.8, +15.7]` — positive, three of four ranges positive, and not
    /// distinguishable from zero; the `+18.1` does *not* reproduce. So quote
    /// `+11.9` as the policy effect and `+5.0` as the leaf effect, never
    /// `+18.1`, and read the round-ten section of the crate docs before moving
    /// it again.
    pub lambda: f64,
    /// Softmax temperature. Larger spreads credit further down the menu;
    /// towards zero it becomes a plain maximum.
    pub tau: f64,
}

impl Default for MenuWeights {
    fn default() -> Self {
        Self {
            // Round ten: `MENU_LAMBDA_V9` (0.6) -> the fitted 0.408, rounded
            // to three places from the regression's `0.407872` because the
            // fourth is far inside the fit's own standard error and a weight
            // this crate ships should be readable.
            lambda: 0.408,
            tau: 1.5,
        }
    }
}

/// The chance that the *victim* of a destroy effect, rather than the
/// destroyer, is the one who takes a replacement production card out of the
/// pool. A flat constant, exactly like [`menu`]'s `CHAIN_MINE_SHARE`, and
/// flagged as one: a real model would read the victim's own interest in the
/// card. Only consulted under [`Config::destroy_replace_discount`].
pub const DESTROY_REPLACE_SHARE: f64 = 0.5;

/// How many chained pending resolutions [`PendingModel::Completed`] walks.
///
/// The engine chains at most once. `ChooseProgressToken`,
/// `ChooseGreatLibraryToken` and `DestroyOpponentCard` each clear the pending
/// flag and take a token or a card without ever setting another;
/// `MausoleumBuild` runs the retrieved card through
/// `engine::construct_card`, which *can* set
/// [`duels_core::state::Pending::ProgressToken`] if the card off the discard
/// pile completes a science pair. So two is the real bound and three is the
/// margin — and the loop stops resolving rather than recursing forever if the
/// engine ever grows a longer chain.
pub const MAX_PENDING_DEPTH: u8 = 3;

/// How [`terms::supremacy_live`] decides whether a symbol the player does not
/// hold can still be obtained.
///
/// The distinction is the project owner's, and it is the science half of round
/// nine's brief: a missing symbol may be **face up and known**, or **face down
/// but plausibly reachable**, and those are not the same claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReachModel {
    /// A symbol is reachable if *some* card printing it is not provably gone
    /// — not in a city, not under a wonder, not in the discard pile — and
    /// belongs to the current age or a later one.
    ///
    /// The pre-existing behaviour, reproduced bit for bit by [`Config::v8`],
    /// and **optimistic in a way that gets worse as an age drains**: three of
    /// every age's cards go back in the box unseen at setup
    /// ([`duels_core::engine::new_game`]), and a boxed card is in none of the
    /// three masks, so it reads as available for the whole of its own age. By
    /// the last turns of an age most of what is left in the deck list is
    /// exactly those boxed cards.
    #[default]
    Optimistic,
    /// A **current-age** card counts only if it can actually still be taken:
    /// its symbol is face up in the structure, or the structure still holds at
    /// least one face-down card. Later ages are read from the deck list
    /// exactly as before, since their cards genuinely have not been dealt.
    ///
    /// The refinement is exact at the end of an age — with no face-down slots
    /// left, the only current-age symbols obtainable are the ones visible on
    /// the board — and it stands down entirely when the structure is empty,
    /// because `state.age()` is then an age whose cards are all still coming.
    /// In between it is the same optimism as [`Optimistic`](ReachModel::Optimistic),
    /// bounded by whether *anything* is still hidden.
    ///
    /// `examples/science_calibration.rs --factors` is what motivated it: the
    /// fitted correction at three distinct symbols in Age II runs **−3.8
    /// victory points early in the age, −9.0 mid and −13.1 late**, a monotone
    /// nine-point gradient in exactly the direction a reachability test that
    /// grows more optimistic as the age drains would produce.
    Structure,
}

/// The science ladder rounds one through seven all shipped: the value of
/// holding 0-5 distinct symbols, before [`EvalWeights::science_ladder`]
/// multiplies it.
///
/// Kept as a named constant so [`Config::v7`] restores it exactly and
/// `tests/v7_identity.rs` can pin it, rather than the round-seven shape being
/// recoverable only from this file's history.
pub const SCIENCE_LADDER_V7: [f64; 6] = [0.0, 1.0, 2.5, 6.0, 12.0, 18.0];

/// The knobs of the science ladder, which is the one term with enough
/// internal structure to want its own group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScienceWeights {
    /// Value of holding 0, 1, 2, 3, 4 or 5 distinct symbols. Convex: the
    /// marginal symbol is worth more the closer six gets. Six is an outright
    /// win and is handled by the terminal check, so it is not in the table.
    pub ladder: [f64; 6],
    /// How much each of Law / Theology / Strategy still on the board raises
    /// the whole ladder. Those three are what make a science plan pay beyond
    /// the printed points, so a token row without them is a weaker reason to
    /// chase pairs.
    pub strong_token_mult: f64,
    /// Weight on the half-pair threat as a whole.
    pub pair_threat_weight: f64,
    /// What fraction of the best board token's value (priced by
    /// [`duels_strategy::science::token_value`]) a *threatened* pair is worth.
    /// Below one because the second copy still has to be taken.
    pub pair_token_share: f64,
    /// Flat value per threatened pair, for the turn the opponent has to spend
    /// if they would rather deny it — a tempo tax they pay whether or not the
    /// token itself is valuable.
    pub pair_tempo_tax: f64,
    /// What the ladder rung is multiplied by once **scientific supremacy is no
    /// longer reachable** for this player — when the symbols they do not hold
    /// can no longer all be obtained, counted from the card data by
    /// [`terms::supremacy_reachable`].
    ///
    /// `1.0` is the pre-existing behaviour and is what [`Config::v6`] restores:
    /// a player at four distinct symbols collected the full rung whether or not
    /// a fifth and sixth were still physically in the game. That is the flaw
    /// this knob fixes, and it is why the whole ladder measured **over-priced**
    /// — see [`EvalWeights::science_ladder`].
    ///
    /// The rung is not taken to zero. Distinct symbols keep paying without
    /// supremacy: they are printed victory points (which
    /// [`terms::card_and_token_vp`] already counts) and they are pairs waiting
    /// to be completed for a progress token (which [`terms::science_ladder`]'s
    /// own `pair_threat` prices, and which this scale deliberately does **not**
    /// touch). What is worthless once the race is dead is the *convexity* — the
    /// rung's jump from 6 to 30 to 54 exists because six symbols win the game.
    ///
    /// Round eight roughly tripled the top two rungs, which makes this gate
    /// carry correspondingly more: it is the difference between "four symbols
    /// and a live race" and "four symbols and nowhere to go", and that is now
    /// worth fifteen victory points rather than six.
    pub dead_race_scale: f64,
    /// How [`dead_race_scale`](ScienceWeights::dead_race_scale)'s gate decides
    /// whether a missing symbol is still obtainable. See [`ReachModel`].
    ///
    /// Applies to the **gate only**, not to `pair_threat`'s
    /// [`terms::second_copy_obtainable`], deliberately: they are the same
    /// question and moving both at once would have made the round-nine
    /// measurement a bundle of two things, exactly as round six declined to
    /// fold Theology into the extra-turn premium for the same reason.
    pub reach_model: ReachModel,
}

impl Default for ScienceWeights {
    fn default() -> Self {
        Self {
            // **Round eight raises the top two rungs, `12` and `18` becoming
            // `30` and `54`, and leaves the first four exactly where round
            // seven left them.** `examples/science_calibration.rs` is the
            // argument, and it is an empirical one: the round-seven evaluation
            // predicts a player's win probability well at zero through three
            // distinct symbols and badly above that, calling a four-symbol
            // Age II position a 44.9% loss where the player actually wins
            // 55.7% of the time (n = 461) and a five-symbol one 45.5% where
            // they win 78.3% (n = 69). The maximum-likelihood correction is
            // +13.4 and +25.6 victory points against rungs worth 6 and 9.
            //
            // The Elo is **neutral** — +1.4 pooled over 12,800 games, signs
            // disagreeing across four seed ranges — and this is adopted anyway,
            // on the calibration and on the victory-kind breakdown, the way
            // round three adopted [`rails`] on its audit. Sweeping the two
            // rungs through 20/34, 30/54 and 45/80 reads +3.5 / +5.0 / +2.6 on
            // top of the temperature refit at seed 1, so 30/54 is the middle of
            // a shallow unimodal curve. See the round-eight crate docs.
            ladder: [0.0, 1.0, 2.5, 6.0, 30.0, 54.0],
            strong_token_mult: 0.15,
            pair_threat_weight: 0.5,
            pair_token_share: 0.5,
            pair_tempo_tax: 0.5,
            // **Measured.** See the round-seven section of the crate docs.
            dead_race_scale: 0.0,
            reach_model: ReachModel::Optimistic,
        }
    }
}

/// The base weight of each evaluation term, before the commitment blend
/// multiplies it.
///
/// Everything is in rough victory-point units, so the numbers can be compared
/// with each other and with [`EvalWeights::instant_result`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalWeights {
    /// Reward per step of conflict-pawn position, per player, under
    /// [`MilitaryModel::Legacy`]. Scaled by military commitment.
    pub military_position: f64,
    /// Weight on the smoothed end-of-game scoring bands under
    /// [`MilitaryModel::Band`]. In victory points already, so one is the
    /// honest rate. Scaled by military commitment, like the term it replaces.
    pub military_band: f64,
    /// Weight on the smoothed loot tokens under [`MilitaryModel::Band`]. Not
    /// commitment-scaled: two coins off a rich opponent is worth the same
    /// whether or not this player has a military plan.
    pub military_loot: f64,
    /// `κ` in `σ = max(σ_min, κ·√S_rem)`: how wide the pawn's remaining travel
    /// is per square root of the shields still in play.
    pub military_sigma_scale: f64,
    /// `σ_min`: the floor on that width, so a game with no shields left still
    /// reads the bands as steps rather than as a discontinuity.
    pub military_sigma_min: f64,
    /// `s / σ`: how much of that width the logistic actually uses.
    pub military_logistic_scale: f64,
    /// Weight on the quadratic "somebody is about to win outright" term. Not
    /// commitment-scaled: an opponent three steps from the capital is urgent
    /// whether or not this player has a military plan of their own.
    pub military_endgame_urgency: f64,
    /// Weight on the "as if the game ended now" card / wonder / token points.
    /// Fades as commitment rises: a race, once it is real, is worth more than
    /// the points it costs.
    pub vp_projection: f64,
    /// Weight on `floor(coins / 3)`, matching the real scoring rule.
    pub coins_div3: f64,
    /// Weight on the development term. In coins, so the default is the real
    /// `floor(coins / 3)` rate: a coin this city never has to spend is worth
    /// exactly what a coin in hand is worth.
    pub development: f64,
    /// What fraction of a player's remaining decisions become builds that
    /// actually pay a resource cost. Not one: some decisions are discards,
    /// wonder builds paid from elsewhere, or chain builds that cost nothing.
    pub development_take_rate: f64,
    /// Weight on the science ladder.
    pub science_ladder: f64,
    /// Knobs of the ladder itself.
    pub science: ScienceWeights,
    /// Weight on cash-on-hand for taking a contested race card. The one term
    /// that *rises* with commitment.
    pub race_card_liquidity: f64,
    /// Coins past which further cash is points rather than liquidity.
    pub race_liquidity_cap: f64,
    /// The coin cushion below which a position is financially risky.
    pub coin_safety_floor: f64,
    /// Weight on the shortfall below that cushion.
    pub coin_safety_penalty: f64,
    /// Weight on the average per-unit trade price a player faces, under
    /// [`EconomyModel::Legacy`].
    pub resource_vulnerability: f64,
    /// Weight on the resource bill under [`EconomyModel::Bill`]. The bill is
    /// in coins, so the term itself divides by three; this multiplies it.
    pub resource_bill: f64,
    /// `β` in the smooth coin model's liquidity channel.
    pub coin_smooth_beta: f64,
    /// `c_ref` in the smooth coin model: the pile size past which further cash
    /// is points rather than liquidity.
    pub coin_smooth_ref: f64,
    /// The decision count at or below which the smooth coin model switches its
    /// points channel back to the real `floor(coins / 3)`, because the
    /// rounding is about to actually happen.
    pub coin_endgame_decisions: f64,
    /// Weight on the forward value of chain starters whose successor is still
    /// in the game. Commitment-scaled by the development weight: forward
    /// economic value fades for the same reason development does.
    pub chain_equity: f64,
    /// The opponent-menu term.
    pub menu: MenuWeights,
    /// Penalty per point of free chain-build value handed to the opponent for
    /// their very next turn. Subsumed by [`EvalWeights::menu`] — a free chain
    /// build is just one kind of high-value accessible card — and switched off
    /// automatically whenever `menu.lambda` is non-zero.
    pub deny_chain_gift: f64,
    /// Weight on the rough power of drafted-but-unbuilt wonders.
    pub wonder_potential: f64,
    /// How many of a player's own decisions one wonder build costs them, in
    /// [`WonderModel::Budget`]'s `turn_factor`. Not one: a wonder needs a card
    /// to bury under it *and* the resources to pay for it, and the turns spent
    /// assembling the second are turns not spent building the first.
    pub wonder_turns_per_wonder: f64,
    /// The `p_build` at which [`WonderModel::Rationed`] pays the full
    /// [`EvalWeights::wonder_potential`] weight: the factor is
    /// `min(1, p_build / ref)`. See [`terms::wonder_potential_rationed`].
    ///
    /// `1.0` is plain rationing and is the value [`Config::v8`] carries (where
    /// the whole model is switched off anyway, so it is inert there).
    /// [`terms::OPENING_P_BUILD`] — seven eighths, the value
    /// [`terms::wonder_p_build`] returns for the whole of Age I — is the
    /// *shape* without the change of scale.
    pub wonder_p_build_ref: f64,
    /// What a play-again wonder's extra turn is worth, in
    /// [`WonderModel::Budget`]. Deliberately `3.0` — the same number
    /// [`terms::wonder_power`] paid for "this wonder has an effect" — so that
    /// at `p_build = 1` and no other effect firing the two models agree.
    pub wonder_extra_turn_vp: f64,
    /// What [`WonderModel::Flat`] pays for an unbuilt wonder that prints
    /// **play again**, *on top of* [`terms::wonder_power`]'s flat `+3, this
    /// wonder has an effect`. **`9.0` by default**, so a play-again wonder is
    /// worth `12` against every other effect's `3`; `0.0` ([`Config::v5`]) is
    /// the pre-existing uniform treatment, bit for bit
    /// (`tests/v5_identity.rs`).
    ///
    /// The project owner's read is that an extra turn is the most valuable thing a
    /// wonder can print, and the flat model prices it exactly like a destroy
    /// or a free discard build. This is the one knob that tests that read,
    /// isolated from [`WonderModel::Budget`]'s other channels — which were
    /// measured negative as a bundle and are confounded with
    /// `duels_strategy::science::token_value`'s flat constants.
    ///
    /// Only ever read under [`WonderModel::Flat`]; `Budget` has its own
    /// [`EvalWeights::wonder_extra_turn_vp`].
    pub wonder_extra_turn_premium: f64,
    /// What beginning each age is worth, indexed by age minus one. Zero for
    /// Age III, which has no next age.
    pub next_age_start: [f64; 3],
    /// Weight on [`duels_strategy::deny_vp`], which prices what a move does to
    /// the opponent's race magnitudes in the same victory-point channel as
    /// everything else.
    pub deny: f64,
    /// What the denial term is multiplied by when the *opponent* is fully
    /// committed to a race. Scaled continuously by their `S(c)`, so a rising
    /// opposing plan makes denial worth more without any threshold.
    pub deny_opponent_commit_boost: f64,
    /// Magnitude assigned when a move actually ends the game. Far larger than
    /// every other term's plausible range put together.
    pub instant_result: f64,
    /// Magnitude assigned when one of [`rails`]' terminal rails fires: the
    /// game is not over, but which way it goes is already settled. Half of
    /// [`EvalWeights::instant_result`], so an actual win still outranks a
    /// certain one, and far above every ordinary term put together, so a rail
    /// really does dominate rather than merely nudge.
    pub imminent: f64,
    /// How much the production-lock-in factor
    /// ([`DevSupply::production_lock_in`]) raises the development and resource
    /// bill terms once a city's production can no longer be fixed. Zero
    /// switches it off and restores the previous arithmetic exactly.
    pub production_lock_in: f64,
    /// Weight on the forward *increment* a built guild's majority count is
    /// projected to gain over the rest of the game
    /// ([`terms::GuildTable::projection`]). The snapshot half is already in
    /// [`terms::card_and_token_vp`], via
    /// [`duels_core::scoring::breakdown`], so this adds only what the snapshot
    /// cannot see. Zero switches the term off entirely.
    ///
    /// Independent of [`Config::guild_pricing`] on purpose: one is about
    /// guilds the player *has*, the other about guilds still on the table.
    pub guild_projection: f64,
    /// Weight on [`terms::yellow_equity`], the coins a city's commercial cards
    /// will add to the discards it has not made yet. Zero switches the term off
    /// and, with it, the matching per-card credit
    /// [`menu::TakeValue`] puts on a yellow card.
    pub yellow_equity: f64,
    /// How often a decision is spent on a discard, for
    /// [`terms::yellow_equity`]. Measured, not guessed — see
    /// [`terms::DISCARD_RATE_PER_DECISION`].
    pub yellow_discard_rate: f64,
    /// Victory points for being the player **to move**
    /// ([`terms::to_move`]) — the value of the right to move, and, on a
    /// post-action state, the credit for a move that earned an extra turn.
    /// Zero switches the term off entirely and restores the previous
    /// arithmetic exactly.
    ///
    /// Round six priced a play-again wonder while it was still *unbuilt* and
    /// measured that at about +47 Elo. This is the other half of the same
    /// idea: the turn once it is actually in hand. See [`terms::to_move`] for
    /// why that has to be read off `current_player` rather than off
    /// `GameState::extra_turn`.
    ///
    /// **Zero by default — an honest negative, and an unambiguous one.** As a
    /// leave-one-out against the round-seven default over 3200 games on each of
    /// two disjoint seed ranges, `2` is worth −21 / −16, `6` is worth −43 / −30
    /// and `12` is worth −45 / −36. `examples/leaf_probe.rs` says the opposite
    /// — the right to move is worth one or two victory points as a *predictor*
    /// — which is the same policy-versus-value divergence the round-seven
    /// section of the crate docs is about.
    pub to_move: f64,
    /// Weight on [`terms::token_equity`], the forward value of the progress
    /// tokens a player already **owns** — what Theology, Economy, Strategy,
    /// Architecture, Masonry and Urbanism are worth for the rest of the game,
    /// as against the printed victory points
    /// [`duels_core::scoring::breakdown`] already counts. Zero switches the
    /// term off entirely and restores the previous arithmetic exactly.
    ///
    /// **Zero by default — an honest negative, and a genuine gap correctly
    /// filled.** Six of the ten tokens are rules changes this evaluation
    /// priced at nothing, which also meant [`PendingModel::Completed`] chose
    /// between them on printed victory points alone. It is worth +7.1 / +8.0
    /// Elo at `0.5` against `phased:base=v6` over 3200 games on each of two
    /// disjoint seed ranges, and −2 / −5 as a leave-one-out against the
    /// finished round-seven default, which is the comparison that decides it.
    /// Kept as an option with the measurement written down; see
    /// [`terms::TokenTable`] for what each channel is priced from.
    pub token_equity: f64,
    /// A single multiplier on the whole weighted sum
    /// ([`evaluate`]'s ordinary return, and [`Root::denial_term`] with it) —
    /// **not** on the rails or on a finished game's `instant_result`, which
    /// are magnitudes rather than judgements.
    ///
    /// # What this is for, and who it is invisible to
    ///
    /// It is invisible to `duels-agent-phased`, which takes an argmax: scaling
    /// every candidate's score by the same positive constant cannot reorder
    /// them (the tie window is `1e-6` against scores of order ten, and the
    /// rails it does not scale are five hundred). It is **not** invisible to
    /// `duels-agent-mcts-eval`, which maps this crate's output through a
    /// logistic of fitted temperature `T` and averages the result into a
    /// win-rate estimate: multiplying by `k` there is exactly dividing that
    /// temperature by `k`, so this knob is the one instrument a
    /// `duels-eval` round has for asking whether the leaf value a search wants
    /// is sharper or flatter than the maximum-likelihood calibration
    /// `examples/calibrate.rs` fits.
    ///
    /// `1.0` — the calibration as fitted — is [`Config::v6`]'s value and is
    /// what the shipped default keeps; see the round-seven section of the
    /// crate docs for the sweep that says so.
    pub value_scale: f64,
    /// The temperature [`win_probability`] divides this crate's victory-point
    /// score by, in victory points, indexed by **age minus one**.
    ///
    /// # Why this is a `Config` field and not just the module constants
    ///
    /// Two different consumers want two different numbers out of the same
    /// logistic, and until round eight they were forced to share one:
    ///
    /// * a **diagnostic** — `duels-server`'s advanced-mode read, and
    ///   `examples/science_calibration.rs`'s own predictions — wants the
    ///   maximum-likelihood calibration, because the only thing it is for is to
    ///   be right about how often this position wins. That is the module
    ///   constant, and round eight refit it.
    /// * a **search leaf** — `duels-agent-mcts-eval`, which calls
    ///   [`win_probability`] with a [`Root`] and averages the result into a win
    ///   rate — wants whatever temperature makes the *search* strongest, which
    ///   round seven's [`EvalWeights::value_scale`] sweep had reason to believe
    ///   is not the same thing (`k = 2.0`, an exact halving of the temperature,
    ///   measured worse than `k = 1.0` there — which round eight then
    ///   contradicted; see below).
    ///
    /// Putting it here is also what makes the difference measurable at all:
    /// `mcts-eval` pins a whole `Config` under the arena's `eval=vN` key, and a
    /// free constant is invisible to that pin, so before round eight a change
    /// to the leaf mapping could not have been A/B tested against the
    /// generation before it at all.
    ///
    /// # What the default is, and why it is not obviously right
    ///
    /// **The default is the module constants** — the maximum-likelihood refit —
    /// and that is a measurement rather than an assumption that the two
    /// consumers agree. Adopting it is worth **+15.3 / +19.0 / +20.6 / +22.9
    /// Elo** to `mcts-eval` against `mcts-eval:eval=v7` over 3200 games on
    /// each of four disjoint seed ranges (`+19.5 ± 6.0` pooled), and it
    /// reproduces at a wall-clock budget and against an unrelated `mcts-uct`
    /// anchor; the round-eight section of the crate docs has all of it.
    ///
    /// That is the *opposite* of what round seven's [`EvalWeights::value_scale`]
    /// sweep implied — there, `k = 2.0`, an exact halving of the temperature,
    /// measured 10 Elo worse than `k = 1.0`. The two are reconcilable (that
    /// sweep was run on an intermediate bundle, on one seed range, and scaled
    /// `evaluate`'s output rather than the temperature, so it also moved the
    /// rails' relative magnitude), but the disagreement is the reason this is a
    /// field: a future round that moves this crate's output scale a long way
    /// should re-fit the calibration *and* re-run the A/B rather than assume
    /// the fitted temperature is automatically the leaf a search wants.
    ///
    /// [`Config::v7`] carries [`WIN_PROBABILITY_TEMPERATURE_V7`], the stale
    /// pre-round-eight triple, which is what makes that A/B a single-binary
    /// measurement.
    pub win_probability_temperature: [f64; 3],
}

impl EvalWeights {
    /// The leaf temperature for a position in `age`.
    ///
    /// Ages outside `1..=3` cannot occur — [`duels_core::GameState::age`] only
    /// ever reports one of the three — and are read as Age III, the sharpest
    /// setting, exactly as the free [`win_probability_temperature`] does.
    #[inline]
    pub fn win_probability_temperature(&self, age: u8) -> f64 {
        match age {
            1 => self.win_probability_temperature[0],
            2 => self.win_probability_temperature[1],
            _ => self.win_probability_temperature[2],
        }
    }
}

impl Default for EvalWeights {
    fn default() -> Self {
        Self {
            // Half of `greedy-ev`'s 0.6 / 3.0, because these are read per
            // player and then differenced, which doubles them.
            military_position: 0.3,
            // **One, and derived rather than fitted.** The band term is
            // already in victory points, so `1.0` is the honest rate; round
            // two shipped `2.0` because the Elo curve was flat above it and
            // because `1.0` never once beat `mcts-uct` militarily, and
            // recorded that as a judgement to revisit rather than inherit.
            //
            // Round three revisits it, and `1.0` now wins outright: +28 / +57
            // Elo against `phased:base=v2` over 600 games on each of two
            // disjoint seed ranges. What changed is that the "occasional
            // supremacy win" the inflated slope was buying is now carried by
            // the terminal rails ([`rails`]), which ask whether a closing card
            // *exists and is affordable* instead of paying a smooth premium
            // on every shield in the hope that one day it adds up. Doubling
            // the price of every red card in the game was always a strange way
            // to say "do not miss a win".
            military_band: 1.0,
            military_loot: 1.0,
            military_sigma_scale: 0.8,
            military_sigma_min: 0.35,
            military_logistic_scale: 0.55,
            military_endgame_urgency: 1.5,
            vp_projection: 1.0,
            coins_div3: 1.0,
            // **Was `1/3` — the rate at which a coin becomes a victory point
            // — on the argument that "a coin this city never has to spend is
            // worth exactly what a coin in hand is worth". Round seven cut it
            // to `0.2`.** The argument is sound about a coin and wrong about
            // this term: `development_value` counts a *want* the pool is
            // projected to have, over `take_rate x decisions_left` builds
            // that may never happen, and a projected saving is not a coin. It
            // is worth +4 / +14 Elo over 3200 games on each of two disjoint
            // seed ranges as the last step of the round-seven refit (+149.2 /
            // +165.6 against `phased:base=v6`, where `1/3` reads +145.5 /
            // +151.8), on a flat curve between 0.16 and 0.25.
            development: 0.2,
            development_take_rate: 0.6,
            // **Was `1.0`. Round seven halves it, and the same round adds
            // the reachability gate that explains why it was too big** — see
            // [`ScienceWeights::dead_race_scale`]. The two together are the
            // largest single effect of the round: `1.0` costs −24 / −27 Elo as
            // a leave-one-out against this default over 3200 games on each of
            // two disjoint seed ranges, and switching the gate off as well
            // costs another −10 / −7.
            //
            // The honest reading of the sweep is that the ladder was worth
            // *less* than nothing at `1.0`: with everything else at its
            // round-six value, driving this weight to `0.1` was worth +63 /
            // +57 Elo on its own. `0.5` with the gate in place is the setting
            // that keeps the scientific-supremacy route on the board — the
            // gate costs the route almost nothing, while a flat cut to `0.2`
            // takes this agent's science wins from ~40 in 3200 games to 3 —
            // and it is the setting the round-seven refit was tuned around.
            //
            // **Round eight left this weight alone and re-shaped
            // [`ScienceWeights::ladder`] instead**, which is the distinction
            // that matters: round seven's evidence was that the ladder's
            // *middle* was over-priced, and round eight's is that its *top* is
            // under-priced. A scalar cannot express both. See the round-eight
            // section of the crate docs for the empirical calibration that
            // separates them.
            science_ladder: 0.5,
            science: ScienceWeights::default(),
            race_card_liquidity: 0.15,
            race_liquidity_cap: 8.0,
            coin_safety_floor: 3.0,
            coin_safety_penalty: 0.5,
            resource_vulnerability: 0.4,
            // **`One`, and derived rather than fitted — which is a reversal.**
            // Round two fitted `3.0` and flagged it: the term already divides
            // the bill by three, which is the rate at which coins become
            // victory points, so `1.0` is all this weight has to say and
            // three said a trade coin was worth a whole victory point. Round
            // two's own comment called that "a measurement, not an argument".
            //
            // Round seven re-measured it with the science ladder and chain
            // equity corrected, and the derived value now wins by a wide
            // margin: sweeping 0.6 / 1.0 / 1.4 / 1.7 / 2.0 / 2.2 / 2.5 / 3.0
            // against `phased:base=v6` over 3200 games on each of two disjoint
            // seed ranges reads +135/+142, +145/+144, +148/+145, +130/+133,
            // +123/+127, +116/+122, +109/+117 and +98/+107 — monotone from
            // `3.0` down to a plateau at 1.0-1.4. **`3.0` was compensating for
            // two other over-priced terms**, which is exactly the failure mode
            // a fitted weight has and a derived one does not; `1.0` is taken
            // because it is the honest rate and is indistinguishable from the
            // top of the plateau.
            resource_bill: 1.0,
            coin_smooth_beta: 0.6,
            coin_smooth_ref: 5.0,
            coin_endgame_decisions: 2.0,
            // **Was `1.0`; round seven cuts it to a quarter, and this is the
            // single largest re-weighting of the round.** Round two shipped
            // `1.0` and recorded that switching it off was worth −51 / +17 —
            // "indistinguishable from zero, kept on the strength of the pooled
            // result and of the fact that [`menu`] needs its table anyway".
            // With the round-seven ladder in place the sign is no longer in
            // doubt: 0.9 / 0.6 / 0.5 / 0.25 / 0.1 / 0.0 read +66/+63,
            // +80/+79, +84/+90, +98/+107, +95/+112 and +96/+112 against
            // `phased:base=v6` over 3200 games on each of two disjoint seed
            // ranges. Flat below 0.25, so a quarter is taken rather than zero:
            // the forward value of a chain starter is real, it was simply
            // priced at four times what it is worth, and keeping the term
            // non-zero keeps `menu`'s table honest about what it feeds.
            chain_equity: 0.25,
            menu: MenuWeights::default(),
            deny_chain_gift: 0.5,
            // **Still `0.5`, and round nine is why that is now a measurement
            // rather than an inheritance.** Under [`WonderModel::Rationed`]
            // the honest weight is `1.25`, and it is worth **+29.7 / +31.1 /
            // +30.5 / +19.6 / +17.4** Elo to `phased` on five disjoint seed
            // ranges and **+86 / +56** against `mcts-uct` — and **-24.4 /
            // -34.6** to `mcts-eval`, which is the consumer that decides. The
            // pair is off by default together; the round-nine section of the
            // crate docs has every column and what separates them.
            wonder_potential: 0.5,
            wonder_turns_per_wonder: 2.5,
            wonder_p_build_ref: 1.0,
            wonder_extra_turn_vp: 3.0,
            // A play-again wonder is worth `3 + 9 = 12` to the flat model,
            // four times what every other effect gets. **Measured**, not
            // argued: +45.3 / +51.6 / +49.9 / +38.5 / +51.9 Elo against
            // `phased:base=v5` over 3200 games on each of five disjoint seed
            // ranges, on a curve that is unimodal in the premium and positive
            // on every range at every magnitude from 1.5 to 15. See the crate
            // docs for the two controls that say this is about *extra turns*
            // rather than about the wonder term wanting a bigger number.
            wonder_extra_turn_premium: 9.0,
            // A starter flip is worth roughly three victory points in Age I
            // and two in Age II — `docs/strategy-backlog.md` §1.2's estimate.
            // These are read *per player* and then differenced, and the flip
            // moves both sides at once, so the differenced swing is twice the
            // number written here. Getting that wrong is what made a single
            // Age I shield read as an eight-point catastrophe and drove the
            // red-card keep rate to ~1%.
            next_age_start: [1.5, 1.0, 0.0],
            deny: 1.0,
            deny_opponent_commit_boost: 1.5,
            instant_result: 1000.0,
            imminent: 500.0,
            // **Off by default, and measured that way.** The lock-in factor
            // is real — Age III genuinely prints no brown or grey card, so an
            // Age III resource bill is a fact rather than a projection — but
            // amplifying the development and bill terms by it costs Elo:
            // −24 / −15 against the same agent with it switched off, over 800
            // games on each of two disjoint seed ranges. Sweeping it (0.25,
            // 0.5, 1.0) never found a value that helped. Kept as an option
            // with the measurement written down, not enabled on the strength
            // of the argument.
            production_lock_in: 0.0,
            // **Off by default, and measured that way.** The forward increment
            // on a guild already built is real arithmetic, and it is worth
            // nothing anybody can measure: −3.6 / +2.7 / −3.5 at 0.5, −2.7 /
            // +3.7 / −3.0 at 1.0 and −2.3 / +2.5 / −0.9 at 2.0, over 3200 games
            // on each of three disjoint seed ranges against the same agent with
            // guild pricing on and this term off. Every interval crosses zero
            // and the sign does not even agree across ranges. The reason is
            // probably that by the time a guild is *in* a city the projection
            // has little of the game left to run, whereas the same projection
            // used to decide whether to *take* the guild — which is
            // [`Config::guild_pricing`], and which does win — is read when it
            // still has an age to be right about.
            guild_projection: 0.0,
            // **Fitted, not derived, and flagged as such** — the same status as
            // `resource_bill`. The term is already in victory points at `1.0`:
            // a yellow card really does add one coin to each of
            // `rate x decisions_left` future discards, and `coin_marginal`
            // really is what a coin is worth. `1.0` is worth +39 / +43 / +48
            // Elo against `phased:base=v4` over 3200 games on each of three
            // disjoint seed ranges, which is already the largest single gain of
            // the round; the Elo curve then keeps climbing to a broad plateau
            // between 3 and 9 and falls again by 14. Four is the middle of the
            // plateau.
            //
            // Four times the honest rate says the term is standing in for
            // something beyond the discard yield it models. The obvious
            // alternative explanation — that this agent simply under-values
            // coins — is **ruled out**: raising `coins_div3` instead is neutral
            // at 1.5 (+3.1 / −0.4 / −3.1) and sharply negative beyond
            // (−10 / −16 / −26 at 2.0, −108 / −105 / −108 at 3.0). What is
            // *not* ruled out is that a 1-ply evaluation under-values commercial
            // cards for some reason that has nothing to do with discarding, in
            // which case a flat per-yellow bonus would do the same work; this
            // round did not build that control, and it is the obvious follow-up.
            yellow_equity: 4.0,
            yellow_discard_rate: terms::DISCARD_RATE_PER_DECISION,
            to_move: 0.0,
            token_equity: 0.0,
            // The calibration as `examples/calibrate.rs` fits it. See the
            // field docs, and the round-seven sweep that measured both
            // directions and found neither.
            value_scale: 1.0,
            // The refit calibration, which round eight measured as *also*
            // being the leaf a search wants: +19.5 +- 6.0 Elo to `mcts-eval`
            // against `mcts-eval:eval=v7` over 12,800 games, positive on all
            // four seed ranges. Written as the constants rather than as
            // literals so that the diagnostic mapping and the leaf mapping
            // cannot silently drift apart — a round that wants them to differ
            // should say so here, deliberately, with the measurement that
            // justifies it. See the field docs.
            win_probability_temperature: [
                WIN_PROBABILITY_TEMPERATURE_AGE_I,
                WIN_PROBABILITY_TEMPERATURE_AGE_II,
                WIN_PROBABILITY_TEMPERATURE_AGE_III,
            ],
        }
    }
}

/// Everything the evaluation can be tuned with.
///
/// Re-exported by `duels-agent-phased` as its own `Config`, so every
/// `phased:base=v1,...` spec string the arena already parses keeps working.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Config {
    /// Base term weights.
    pub eval: EvalWeights,
    /// The commitment blend.
    pub blend: Blend,
    /// How the conflict pawn is priced.
    pub military_model: MilitaryModel,
    /// How coins are priced.
    pub coin_model: CoinModel,
    /// How resource-market exposure is priced.
    pub economy_model: EconomyModel,
    /// Whether the terminal rails are consulted. See [`rails`].
    pub rails: RailModel,
    /// How the opponent-menu term prices a red card's shields.
    pub menu_shield_pricing: MenuShieldPricing,
    /// How many of this player's own rounds the military smoothing width looks
    /// ahead over. See [`terms::horizon_supply`].
    ///
    /// **`None` — the whole remaining shield supply — by default, unchanged
    /// from round two.** Narrowing the width to a horizon is a
    /// better-motivated model, and it does sharpen the bands: at the supply
    /// width two shields are worth almost exactly twice one, which is not
    /// what a step function is supposed to do. It also makes no difference
    /// anybody can measure — `h` of 2, 3 and 5 all landed within a couple of
    /// Elo of the supply width over 800 games on each of two disjoint seed
    /// ranges — and the convention here is not to move a default on a neutral
    /// result. The option stays available as `phased:horizon=3`.
    pub military_horizon: Option<f64>,
    /// Whether the evaluator finishes a turn the engine left mid-effect. See
    /// [`PendingModel`].
    pub pending_model: PendingModel,
    /// How a drafted-but-unbuilt wonder is priced. See [`WonderModel`].
    ///
    /// **Still [`WonderModel::Flat`].** Round nine built
    /// [`WonderModel::Rationed`], measured it on both consumers and did not
    /// adopt it — it is worth about +30 Elo to `phased` and −15 to
    /// `mcts-eval`, and `mcts-eval` is the consumer that decides. Reachable as
    /// `phased:wonder=rationed`.
    pub wonder_model: WonderModel,
    /// Whether a destroy effect's credit is discounted by the chance the
    /// opponent simply builds the production back.
    ///
    /// Only ever consulted under [`PendingModel::Completed`], which is the only
    /// mode in which a destroy is scored as more than the flat "has an effect"
    /// bonus at all. See [`Root::destroy_replaceability`].
    pub destroy_replace_discount: bool,
    /// Whether the menu prices a guild card at all. See [`GuildPricing`].
    pub guild_pricing: GuildPricing,
    /// Whether the menu prices a commercial card's count-scaled coin payout.
    /// See [`CountPricing`].
    pub count_pricing: CountPricing,
    /// What the menu falls back on when nothing is affordable. See
    /// [`MenuFloor`].
    pub menu_floor: MenuFloor,
    /// `c_soft` in the menu's soft affordability weight. **Zero — the hard
    /// afford / do-not-afford cutoff — by default**, which
    /// [`menu::menu_term`] reproduces bit for bit.
    pub menu_afford_soft: f64,
    /// How the development supply statistics weight an undealt card. See
    /// [`SupplyModel`].
    pub supply_model: SupplyModel,
}

impl Config {
    /// The configuration this crate shipped with: every model at its original
    /// setting, no chain equity, no opponent menu, and the original
    /// `next_age_start` magnitudes.
    ///
    /// Kept so the arena can benchmark against the exact previous agent in one
    /// binary (`phased:base=v1`), and so
    /// `tests/legacy_identity.rs` can assert that this configuration
    /// reproduces a verbatim copy of the old evaluation move for move.
    pub fn v1() -> Config {
        Config {
            eval: EvalWeights {
                next_age_start: [4.0, 3.0, 0.0],
                chain_equity: 0.0,
                menu: MenuWeights {
                    lambda: 0.0,
                    ..MenuWeights::default()
                },
                ..Config::v2().eval
            },
            blend: Blend::default(),
            military_model: MilitaryModel::Legacy,
            coin_model: CoinModel::Legacy,
            economy_model: EconomyModel::Legacy,
            ..Config::v2()
        }
    }

    /// The configuration the *second* round of work shipped with: no terminal
    /// rails, the one-sided menu shield price, the supply-wide military
    /// smoothing width, no production lock-in, and `military_band = 2.0`.
    ///
    /// `tests/v2_identity.rs` asserts this reproduces that agent's arithmetic
    /// bit for bit, which is what makes `phased` against `phased:base=v2` a
    /// single-binary measurement.
    pub fn v2() -> Config {
        Config {
            eval: EvalWeights {
                military_band: 2.0,
                imminent: 0.0,
                production_lock_in: 0.0,
                // Not `EvalWeights::default()`: every later round's weights
                // have to arrive at their own *off* values here, and round
                // five is the first whose defaults are non-zero. Chaining
                // through `v3().eval` — which is `v4().eval`, which is
                // `v5().eval` — is what keeps this snapshot a snapshot as the
                // defaults move on.
                ..Config::v3().eval
            },
            rails: RailModel::Off,
            menu_shield_pricing: MenuShieldPricing::OneSided,
            military_horizon: None,
            ..Config::v3()
        }
    }

    /// The configuration the *third* round of work shipped with: the pending
    /// state scored as it stands, the flat wonder-power model, and no
    /// destroy-replacement discount.
    ///
    /// `tests/v3_identity.rs` asserts this reproduces that agent's arithmetic
    /// bit for bit — with **one deliberate exception**, documented there: the
    /// seven-wonder cap in [`terms::wonder_potential`] is a bug fix rather
    /// than a model, so it is landed unconditionally and `Config::v3()` does
    /// not restore the old, uncapped sum.
    pub fn v3() -> Config {
        Config {
            pending_model: PendingModel::Unresolved,
            wonder_model: WonderModel::Flat,
            destroy_replace_discount: false,
            ..Config::v4()
        }
    }

    /// The configuration the *fourth* round of work shipped with: guilds
    /// unpriced on the menu, no menu floor, the hard affordability cutoff, the
    /// unweighted supply pool, and neither of the two new terms.
    ///
    /// `tests/v4_identity.rs` asserts this reproduces that agent's arithmetic
    /// bit for bit, which is what makes `phased` against `phased:base=v4` a
    /// single-binary measurement.
    pub fn v4() -> Config {
        Config {
            eval: EvalWeights {
                guild_projection: 0.0,
                yellow_equity: 0.0,
                // Chained through `v5().eval`, not `default().eval`, for the
                // reason spelled out in `v2()`: every *later* round's weights
                // have to arrive at their own off values here, and round six's
                // extra-turn premium is the second whose default is non-zero.
                ..Config::v5().eval
            },
            guild_pricing: GuildPricing::Unpriced,
            menu_floor: MenuFloor::None,
            menu_afford_soft: 0.0,
            supply_model: SupplyModel::Raw,
            ..Config::v5()
        }
    }

    /// The configuration the *fifth* round of work shipped with: no extra-turn
    /// premium on [`WonderModel::Flat`], so a play-again wonder is worth the
    /// same flat `+3` as a destroy or a free discard build.
    ///
    /// `tests/v5_identity.rs` asserts this reproduces that agent's arithmetic
    /// bit for bit, which is what makes `phased` against `phased:base=v5` a
    /// single-binary measurement.
    pub fn v5() -> Config {
        Config {
            eval: EvalWeights {
                wonder_extra_turn_premium: 0.0,
                ..Config::v6().eval
            },
            ..Config::v6()
        }
    }

    /// The configuration the *sixth* round of work shipped with.
    ///
    /// Round seven changed [`Config::default`], so — following the contract
    /// spelled out under [`Config::v7`] — this function stops being an alias
    /// for the default and spells out round seven's *off* values instead:
    /// the science ladder back at its round-six weight with the dead-race gate
    /// disabled, no owned-token equity, the menu blind to a commercial card's
    /// count-scaled coins, and the value scale at one.
    ///
    /// `tests/v6_identity.rs` asserts this reproduces round six's arithmetic
    /// bit for bit, which is what makes `phased` against `phased:base=v6` a
    /// single-binary measurement.
    pub fn v6() -> Config {
        Config {
            eval: EvalWeights {
                development: 1.0 / 3.0,
                resource_bill: 3.0,
                chain_equity: 1.0,
                science_ladder: 1.0,
                science: ScienceWeights {
                    dead_race_scale: 1.0,
                    pair_threat_weight: 1.0,
                    ..Config::v7().eval.science
                },
                to_move: 0.0,
                token_equity: 0.0,
                value_scale: 1.0,
                ..Config::v7().eval
            },
            count_pricing: CountPricing::Unpriced,
            ..Config::v7()
        }
    }

    /// The configuration the *seventh* round of work shipped with.
    ///
    /// Round eight changed [`Config::default`], so — following the contract
    /// spelled out under [`Config::v8`], and exactly as round seven did to
    /// [`Config::v6`] — this function stops being an alias for the default and
    /// spells out round eight's *off* values instead: the leaf temperature at
    /// the pre-round-eight triple [`WIN_PROBABILITY_TEMPERATURE_V7`], and the
    /// science ladder's top two rungs at their round-seven values.
    ///
    /// `tests/v7_identity.rs` asserts this reproduces round seven's arithmetic
    /// bit for bit, which is what makes `mcts-eval` against
    /// `mcts-eval:eval=v7` a single-binary measurement.
    pub fn v7() -> Config {
        Config {
            eval: EvalWeights {
                win_probability_temperature: WIN_PROBABILITY_TEMPERATURE_V7,
                science: ScienceWeights {
                    ladder: SCIENCE_LADDER_V7,
                    ..Config::v8().eval.science
                },
                ..Config::v8().eval
            },
            ..Config::v8()
        }
    }

    /// The configuration the *eighth* round of work shipped with.
    ///
    /// **Round nine deliberately did not move the default**, so for the whole
    /// of round nine this function was an alias for it. It added two
    /// options — the [`WonderModel::Rationed`] wonder term and the
    /// [`ReachModel::Structure`] symbol-reachability test — and measured both,
    /// and each measurement said to leave the default alone: the first is
    /// worth +30 Elo to `phased` and **−24** to `mcts-eval`, the second is
    /// neutral to both. `tests/round_nine_identity.rs` is the guard that they
    /// really are off, and that the one *shape* round nine changed (the
    /// symbol walk grew a second branch) reproduces round eight bit for bit
    /// on the branch it kept.
    ///
    /// Round ten moved [`Config::default`], so — following the contract this
    /// function's own documentation spelled out, and exactly as round eight
    /// did to [`Config::v7`] — it now reads through [`Config::v9`] and spells
    /// out round nine's two *off* values as literals. **They are the values
    /// `v9()` carries anyway**, because round nine left the default alone;
    /// they are written out here for the explicitness the rest of this chain
    /// uses, so that a future round which does turn one of them on cannot
    /// silently redefine round eight.
    ///
    /// This makes `v8()` and [`Config::v9`] the same configuration, and that
    /// is the honest reading of the history rather than a defect: round nine
    /// changed nothing about the default. It is the one place in this chain
    /// where two adjacent links are equal, and the two chain tests carve it
    /// out by name.
    ///
    /// `tests/v9_identity.rs` covers this generation's arithmetic through
    /// `v9()`.
    pub fn v8() -> Config {
        Config {
            eval: EvalWeights {
                science: ScienceWeights {
                    reach_model: ReachModel::Optimistic,
                    ..Config::v9().eval.science
                },
                ..Config::v9().eval
            },
            wonder_model: WonderModel::Flat,
            ..Config::v9()
        }
    }

    /// The configuration the *ninth* round of work shipped with, which is
    /// numerically identical to [`Config::v8`] — round nine measured two
    /// options and adopted neither, so the only thing separating this
    /// generation from round eight's is the calendar.
    ///
    /// Round ten changed [`Config::default`] in exactly one place, and this
    /// function sets it back: [`MenuWeights::lambda`], the weight on the
    /// opponent-menu term, from [`MENU_LAMBDA_V9`] (`0.6`) to the fitted
    /// `0.408`. That is the whole delta — round ten moved no *shape* at all,
    /// so `tests/v9_identity.rs` needs no verbatim copy of a function, only a
    /// proof that restoring the scalar restores the arithmetic over real
    /// games.
    ///
    /// `tests/v9_identity.rs` asserts this reproduces round nine's arithmetic
    /// bit for bit, which is what makes both of round ten's confirmation A/Bs
    /// — `mcts-eval` against `mcts-eval:eval=v9`, and `phased` against
    /// `phased:base=v9` — single-binary measurements of one scalar.
    ///
    /// # Why a snapshot with no `#[cfg(test)]` copy behind it
    ///
    /// `v1()`-`v5()` each exist so a *later* round can be measured against an
    /// *earlier* one in a single binary, and each has a `tests/vN_identity.rs`
    /// holding a verbatim copy of the code it snapshots. There is nothing to
    /// copy here: this generation *is* the current arithmetic apart from one
    /// scalar, so the identity that matters is a different one — that a
    /// **search agent pinning this generation keeps getting the same
    /// numbers**.
    ///
    /// The one agent that ever pinned a generation of this crate was
    /// `duels-agent-mcts-uct`, and it pinned [`Config::v6`] — the generation
    /// current when its leaf value was measured — behind a golden-values test
    /// over about fifty fixed positions. That machinery has since moved to
    /// `duels-agent-mcts-eval`, which deliberately does **not** pin: it reads
    /// [`Config::default`] live at every tree construction, precisely so it
    /// keeps getting stronger as later `phased` rounds land, rather than
    /// needing a version bump to benefit from one (see that crate's docs for
    /// why). **No agent currently pins a generation of this crate**, so there
    /// is presently no downstream golden-values test that would catch a silent
    /// arithmetic change here.
    ///
    /// The nearest thing that survives is `duels-agent-phased`'s
    /// `tests/p_build_identity.rs`, which hashes twelve whole self-play games
    /// per configuration. It is not a generation pin — it guards a different
    /// claim — but it does pin the shipping default's *decisions*, and round
    /// ten found out the useful way: it failed, and re-recording it was part of
    /// this round. An evaluation change that reaches an agent cannot land
    /// silently while that file exists.
    ///
    /// # The contract for the next round, if a future consumer ever pins again
    ///
    /// The moment [`Config::default`] moves, this stops being a snapshot of
    /// anything. So a round that changes the default must, in the same PR:
    /// add `v10()` and re-point this function's `..` at it, spelling out round
    /// ten's *off* values here (exactly as [`Config::v2`]'s comment
    /// describes, and exactly as rounds eight and ten did to [`Config::v7`]
    /// and [`Config::v8`]). Round ten added no option, so there is nothing
    /// for a `v10()` to switch *off* beyond the scalar itself — that snapshot
    /// is one `..` away. If some future agent pins a generation the way
    /// `mcts-uct` once did, that agent's own golden-values test is what
    /// re-baselining means for it — this crate cannot enforce that on its
    /// behalf.
    ///
    /// Because the newest link in this chain is defined *as* the default
    /// (`v1`-`v8` are deltas from it, not literal field values), any
    /// same-crate check that the newest snapshot equals the default is a
    /// tautology — this is not a gap introduced by removing the downstream
    /// test, it was always true. See the note in
    /// `tests::the_generation_snapshots_are_a_chain_of_distinct_configurations`.
    pub fn v9() -> Config {
        Config {
            eval: EvalWeights {
                menu: MenuWeights {
                    lambda: MENU_LAMBDA_V9,
                    ..Config::default().eval.menu
                },
                ..Config::default().eval
            },
            ..Config::default()
        }
    }
}

impl Config {
    /// A short, reproducible encoding of the configuration, for an agent's
    /// `duels_agents_api::AgentSpec::params` (this crate does not depend on
    /// that one; `duels-agent-phased` is the caller that fills it in).
    pub fn params_string(&self) -> String {
        let e = &self.eval;
        let b = &self.blend;
        format!(
            "sci={:.2}/dead={:.2}/pair={:.2}/reach={},ladder={:?},temp={:?},\
             tokeneq={:.2},tomove={:.2},scale={:.3},count={},\
             guild={}/{:.2},menufloor={},afford={:.2},supply={},yellow={:.2}@{:.3},\
             models={}/{}/{},pending={},wonder={}/{:.2}/{:.2}/{:.2},destroyrepl={},\
             rails={}/{:.0},shieldprice={},horizon={},lockin={:.2},\
             menu={:.2}@{:.2},chaineq={:.2},bill={:.2},band={:.2}/{:.2},\
             smooth={:.2}@{:.1}|\
             mil={:.2}/{:.2},vp={:.2},coin={:.2},dev={:.3}@{:.2},sci={:.2},raceliq={:.2},econ={:.1}/{:.2}/{:.2},chain={:.2},wonder={:.2},start={:?},deny={:.2}x{:.2},win={:.0}|\
             blend={},a={:.2},b={:.2},n={:.1},c0={:.2},floors={:.2}/{:.2}/{:.2}/{:.2}/{:.2},boosts={:.2}/{:.2},sciprog={}",
            e.science_ladder,
            e.science.dead_race_scale,
            e.science.pair_threat_weight,
            match e.science.reach_model {
                ReachModel::Optimistic => "optimistic",
                ReachModel::Structure => "structure",
            },
            e.science.ladder,
            e.win_probability_temperature,
            e.token_equity,
            e.to_move,
            e.value_scale,
            match self.count_pricing {
                CountPricing::Unpriced => "unpriced",
                CountPricing::Counted => "counted",
            },
            match self.guild_pricing {
                GuildPricing::Unpriced => "unpriced",
                GuildPricing::Projected => "projected",
            },
            e.guild_projection,
            match self.menu_floor {
                MenuFloor::None => "none",
                MenuFloor::Discard => "discard",
                MenuFloor::DiscardAndWonder => "discardwonder",
            },
            self.menu_afford_soft,
            match self.supply_model {
                SupplyModel::Raw => "raw",
                SupplyModel::Dealt => "dealt",
            },
            e.yellow_equity,
            e.yellow_discard_rate,
            match self.military_model {
                MilitaryModel::Legacy => "legacy",
                MilitaryModel::Band => "band",
            },
            match self.coin_model {
                CoinModel::Legacy => "legacy",
                CoinModel::Smooth => "smooth",
            },
            match self.economy_model {
                EconomyModel::Legacy => "legacy",
                EconomyModel::Bill => "bill",
            },
            match self.pending_model {
                PendingModel::Unresolved => "unresolved",
                PendingModel::Completed => "completed",
            },
            match self.wonder_model {
                WonderModel::Flat => "flat",
                WonderModel::Budget => "budget",
                WonderModel::Rationed => "rationed",
            },
            e.wonder_turns_per_wonder,
            e.wonder_extra_turn_vp,
            e.wonder_extra_turn_premium,
            u8::from(self.destroy_replace_discount),
            match self.rails {
                RailModel::Off => "off",
                RailModel::On => "on",
            },
            e.imminent,
            match self.menu_shield_pricing {
                MenuShieldPricing::OneSided => "onesided",
                MenuShieldPricing::Differenced => "diff",
            },
            match self.military_horizon {
                None => "supply".to_string(),
                Some(h) => format!("{h:.1}"),
            },
            e.production_lock_in,
            e.menu.lambda,
            e.menu.tau,
            e.chain_equity,
            e.resource_bill,
            e.military_band,
            e.military_loot,
            e.coin_smooth_beta,
            e.coin_smooth_ref,
            e.military_position,
            e.military_endgame_urgency,
            e.vp_projection,
            e.coins_div3,
            e.development,
            e.development_take_rate,
            e.science_ladder,
            e.race_card_liquidity,
            e.coin_safety_floor,
            e.coin_safety_penalty,
            e.resource_vulnerability,
            e.deny_chain_gift,
            e.wonder_potential,
            e.next_age_start,
            e.deny,
            e.deny_opponent_commit_boost,
            e.instant_result,
            u8::from(b.enabled),
            b.alpha_m,
            b.beta_prog,
            b.hill_n,
            b.c0,
            b.floor_vp,
            b.floor_liq,
            b.floor_dev,
            b.floor_race_liq,
            b.floor_econ,
            b.boost_sci,
            b.boost_mil,
            match b.science_progress {
                ScienceProgress::Root => "root",
                ScienceProgress::Leaf => "leaf",
            },
        )
    }
}

/// Everything computed once per decision, from the root position, and reused
/// unchanged for every candidate action and every chance outcome.
///
/// Constructing one is the *only* place the commitment blend is evaluated.
/// [`evaluate`] takes it by reference and has no way to rebuild it, which is
/// what makes root-fixing a property of the types rather than a discipline —
/// see the crate docs.
#[derive(Debug, Clone)]
pub struct Root {
    config: Config,
    stance: Stance,
    /// Indexed by [`Player::index`].
    weights: [TermWeights; 2],
    supply: DevSupply,
    deny_scale: f64,
    age: u8,
    smoothing: MilSmoothing,
    menu: MenuTables,
    wonders: terms::WonderBudget,
    guilds: terms::GuildTable,
    tokens: terms::TokenTable,
    /// `replace_r`, indexed by [`duels_core::data::Resource::index`]. Empty
    /// (all zero) unless [`Config::destroy_replace_discount`] is on.
    replace: [f64; duels_core::data::NUM_RESOURCES],
}

impl Root {
    /// Read the root position for the player to move.
    pub fn new(state: &GameState, me: Player, config: Config) -> Root {
        let opp = me.other();
        let ctx = Context::with(state, ThreatWeights::default());
        // One `Stance` carries both players' military, science and point
        // reads, so this is the whole strategy layer for one position rather
        // than five separate calls. Its *prior* is deliberately not used —
        // an earlier investigation (`duels-agent-strategist`) found that the
        // shape of a policy prior does not transfer to an additive evaluation
        // score. What is used is `delta_m` / `deny_vp`, which need a `Stance`
        // only as the carrier of the reads they price against.
        let stance = stance_in(state, me, PriorWeights::default(), &ctx);
        let edge_me = stance.vp.structural_edge;
        let edge_opp =
            duels_strategy::vp_read_with(state, opp, &ctx, &VpWeights::default()).structural_edge;

        let commit_me = Commitment::of(&stance.science, &stance.military, edge_me, &config.blend);
        let commit_opp = Commitment::of(
            &stance.opponent_science,
            &stance.opponent_military,
            edge_opp,
            &config.blend,
        );

        let mut weights = [TermWeights::of(commit_me, &config.blend); 2];
        weights[opp.index()] = TermWeights::of(commit_opp, &config.blend);

        let supply = DevSupply::of_with(&ctx.board, config.supply_model);
        // Shields still obtainable anywhere in the game, straight off the
        // military read — the width of the pawn's remaining random walk.
        let shields_remaining = f64::from(stance.military.visible)
            + stance.military.expected_hidden
            + stance.military.expected_future_ages;
        // ...narrowed, optionally, to the shields the next few of *this*
        // player's rounds will actually see. See `terms::horizon_supply` for
        // why the whole remaining supply makes the step function read as a
        // straight line for most of a game.
        let smoothing = MilSmoothing::of(
            terms::horizon_supply(
                shields_remaining,
                terms::decisions_left(state, me),
                config.military_horizon,
            ),
            config.eval.military_sigma_scale,
            config.eval.military_sigma_min,
            config.eval.military_logistic_scale,
        );

        // The pricing context both forward-looking terms share. Building it
        // is the whole of their per-decision cost: two `TakeValue`s, the
        // seventeen-link chain table, and one `v` per card face up at the
        // root. Everything downstream is a table lookup plus an affordability
        // check.
        // The majority projections a guild card is priced against. Built only
        // when something reads them, so a run with guild pricing off and the
        // projection term at zero pays nothing for either.
        let guilds = if config.guild_pricing == GuildPricing::Unpriced
            && config.eval.guild_projection == 0.0
        {
            terms::GuildTable::empty()
        } else {
            terms::GuildTable::of(state, &supply, &config.eval)
        };

        let take_tables = menu::TakeContext {
            supply: &supply,
            smoothing: &smoothing,
            guild: &guilds,
        };
        let take = [Player::One, Player::Two].map(|p| {
            TakeValue::of(
                state,
                p,
                take_tables,
                &config,
                weights[p.index()].liquidity,
                (
                    weights[p.index()].military,
                    weights[p.other().index()].military,
                ),
            )
        });
        let chain = if config.eval.chain_equity == 0.0 && config.eval.menu.lambda == 0.0 {
            ChainTable::empty()
        } else {
            ChainTable::of(state, &ctx.board, &ctx.expected, &take)
        };
        // The per-effect wonder prices, wanted by two unrelated things now: the
        // wonder budget model, and the menu's `DiscardAndWonder` floor.
        let wonders = if config.wonder_model == WonderModel::Budget
            || config.menu_floor == MenuFloor::DiscardAndWonder
        {
            terms::WonderBudget::of(state, &take, &chain, &config.eval)
        } else {
            terms::WonderBudget::empty()
        };
        // `p_build` for `WonderModel::Rationed` is deliberately *not* cached
        // here. It used to be, and that was the defect: a `Root` is built once
        // per decision (`phased`) or once per search tree (`mcts-eval`) and
        // every leaf is priced against it, so a cached `p_build` scored leaves
        // by the root's wonder-slot and decision counts rather than their own.
        // `terms::wonder_potential_rationed` reads it off the state it is
        // scoring instead — see its docs and the round-nine crate docs.
        let replace = if config.destroy_replace_discount {
            std::array::from_fn(|r| (supply.sources[r] * DESTROY_REPLACE_SHARE).min(1.0))
        } else {
            [0.0; duels_core::data::NUM_RESOURCES]
        };

        // The owned-token prices, read off the two `TakeValue`s the menu has
        // already built rather than recomputed, so the two cannot disagree
        // about what a shield or a coin is worth. Built only when something
        // reads it.
        let tokens = if config.eval.token_equity == 0.0 {
            terms::TokenTable::empty()
        } else {
            terms::TokenTable::of(
                state,
                &supply,
                chain.starters(),
                [take[0].shield_delta[1], take[1].shield_delta[1]],
                [take[0].coin_marginal, take[1].coin_marginal],
                &config.eval,
            )
        };

        let menu = if config.eval.menu.lambda == 0.0 {
            MenuTables::unpriced(state, take, chain)
        } else {
            MenuTables::with(
                state,
                &ctx.board,
                take,
                chain,
                menu::MenuOptions {
                    floor: config.menu_floor,
                    afford_soft: config.menu_afford_soft,
                },
                wonders.clone(),
            )
        };

        Root {
            guilds,
            tokens,
            wonders,
            replace,
            deny_scale: 1.0 + (config.eval.deny_opponent_commit_boost - 1.0) * commit_opp.s,
            supply,
            age: state.age(),
            smoothing,
            menu,
            stance,
            weights,
            config,
        }
    }

    /// The root-fixed military smoothing.
    #[inline]
    pub fn smoothing(&self) -> &MilSmoothing {
        &self.smoothing
    }

    /// The root-fixed pricing tables the forward-looking terms share.
    #[inline]
    pub fn menu(&self) -> &MenuTables {
        &self.menu
    }

    /// The configuration in force.
    #[inline]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The root-fixed term multipliers for one player.
    #[inline]
    pub fn weights(&self, p: Player) -> &TermWeights {
        &self.weights[p.index()]
    }

    /// One player's commitment scalars.
    #[inline]
    pub fn commitment(&self, p: Player) -> &Commitment {
        &self.weights[p.index()].commitment
    }

    /// The root stance, for diagnostics.
    #[inline]
    pub fn stance(&self) -> &Stance {
        &self.stance
    }

    /// The development supply statistics, for diagnostics.
    #[inline]
    pub fn supply(&self) -> &DevSupply {
        &self.supply
    }

    /// The root-fixed wonder budget. All zero unless
    /// [`WonderModel::Budget`] is in force.
    #[inline]
    pub fn wonders(&self) -> &terms::WonderBudget {
        &self.wonders
    }

    /// The root-fixed guild majority projections. All zero unless something
    /// reads them — [`GuildPricing::Projected`] or a non-zero
    /// [`EvalWeights::guild_projection`].
    #[inline]
    pub fn guilds(&self) -> &terms::GuildTable {
        &self.guilds
    }

    /// The root-fixed forward prices of the ten progress tokens, for
    /// diagnostics. All zero unless [`EvalWeights::token_equity`] is non-zero.
    #[inline]
    pub fn tokens(&self) -> &terms::TokenTable {
        &self.tokens
    }

    /// How replaceable `card`'s production is: `0` for a card whose resources
    /// the market can no longer print (every Age III destroy, since Age III
    /// prints no brown or grey card — counted off `data/cards.json` by
    /// `tests::production_is_completely_frozen_by_age_three`), rising towards
    /// `1` when several undestroyed sources are still coming.
    ///
    /// ```text
    /// replace_r = min(1, sources_remaining(r) · dealt_frac · share_opp)
    /// replace(card) = min over the resources the card produces of replace_r
    /// ```
    ///
    /// `sources_remaining · dealt_frac` is [`DevSupply::sources`], which
    /// already discounts each pool card by the chance it is ever dealt.
    /// `share_opp` is a flat [`DESTROY_REPLACE_SHARE`] — the chance the
    /// *victim*, rather than the destroyer, is the one who ends up taking the
    /// replacement. A real model would read the victim's own interest in the
    /// card, exactly as [`menu::ChainTable`]'s `CHAIN_MINE_SHARE` would; both
    /// are flat constants for the same reason and both are flagged as such.
    ///
    /// The `min` over the card's resources is the conservative direction: a
    /// card is only fully replaceable if *everything* it produced can be
    /// bought back.
    ///
    /// Zero throughout unless [`Config::destroy_replace_discount`] is on.
    pub fn destroy_replaceability(&self, card: duels_core::data::CardId) -> f64 {
        let mut out = 1.0f64;
        let mut any = false;
        for (r, &n) in card.def().produces.iter().enumerate() {
            if n > 0 {
                any = true;
                out = out.min(self.replace[r]);
            }
        }
        if any {
            out
        } else {
            0.0
        }
    }

    /// The age the root position was in.
    ///
    /// Load-bearing for correctness, not a convenience: see
    /// [`terms::chain_gift_exposure`], the one term that reads a card in the
    /// structure and therefore has to stand down once a move has ended the
    /// age and the engine has dealt a whole new one out of a deck no
    /// observation can see.
    #[inline]
    pub fn age(&self) -> u8 {
        self.age
    }

    /// The multiplier the denial term carries, given how committed the
    /// opponent is.
    #[inline]
    pub fn deny_scale(&self) -> f64 {
        self.deny_scale
    }

    /// `A(action)`: the victory-point equivalent of what this action does to
    /// the opponent's race magnitudes, scaled by how committed they are.
    ///
    /// A function of the root position and the action only, so it is added
    /// once per candidate rather than once per chance outcome.
    pub fn denial_term(&self, action: Action) -> f64 {
        let v = self.config.eval.deny * self.deny_scale * deny_vp(action, &self.stance);
        // Scaled with the position value it is added to, so a `value_scale`
        // that only a search can see cannot quietly reweight the one term a
        // 1-ply agent adds outside `evaluate`. Guarded, so `1.0` is exact.
        if self.config.eval.value_scale == 1.0 {
            v
        } else {
            v * self.config.eval.value_scale
        }
    }
}

/// Score `state` for `me`, higher is better, under the root-fixed weights in
/// `root`.
///
/// A finished game is scored by `instant_result` alone, dwarfing every other
/// term; otherwise every term is read for each player separately and
/// differenced.
pub fn evaluate(state: &GameState, me: Player, root: &Root) -> f64 {
    evaluate_at(state, me, root, MAX_PENDING_DEPTH)
}

/// The maximum-likelihood temperature for an Age I position, in victory
/// points, fitted by `examples/calibrate.rs` over `phased` self-play (see that
/// example for how to reproduce it).
///
/// **Refitted in round eight, and the previous value was badly stale.** It read
/// `47.57` — fitted over 28,723 positions against round *six*'s evaluation, and
/// left untouched when round seven re-weighted five terms. The refit, over
/// 643,875 positions from 8,990 decided games of the round-seven default, is
/// `26.40`: the shipped constant was **1.8 times too flat**, and
/// [`win_probability`] was correspondingly pulled towards `0.5` everywhere.
/// See the round-eight section of the crate docs, and
/// [`EvalWeights::win_probability_temperature`] for why the number a *search*
/// wants and the number the *likelihood* wants are configured separately even
/// though round eight measured them to be the same.
pub const WIN_PROBABILITY_TEMPERATURE_AGE_I: f64 = 26.40;

/// The same fit restricted to Age II positions. Round eight: `43.75` → `20.34`.
pub const WIN_PROBABILITY_TEMPERATURE_AGE_II: f64 = 20.34;

/// The same fit restricted to Age III positions, where the evaluation is
/// sharpest. Round eight: `25.18` → `15.50`.
pub const WIN_PROBABILITY_TEMPERATURE_AGE_III: f64 = 15.50;

/// The same fit over every position at once, kept for reference: it is what a
/// single flat constant would have been, and the per-age spread above is why
/// [`win_probability`] does not use it. Round eight: `38.61` → `20.12`.
pub const WIN_PROBABILITY_TEMPERATURE_OVERALL: f64 = 20.12;

/// The pre-round-eight constants, kept so [`Config::v7`] can restore the leaf
/// mapping every earlier generation was measured under, bit for bit.
///
/// Indexed by age minus one, exactly like
/// [`EvalWeights::win_probability_temperature`].
pub const WIN_PROBABILITY_TEMPERATURE_V7: [f64; 3] = [47.57, 43.75, 25.18];

/// The calibrated temperature for a position in `age`.
///
/// Ages outside `1..=3` cannot occur — [`duels_core::GameState::age`] only
/// ever reports one of the three — and are read as Age III, the sharpest
/// setting, so a hypothetical fourth age could not accidentally get the
/// flattest curve.
#[inline]
pub fn win_probability_temperature(age: u8) -> f64 {
    match age {
        1 => WIN_PROBABILITY_TEMPERATURE_AGE_I,
        2 => WIN_PROBABILITY_TEMPERATURE_AGE_II,
        _ => WIN_PROBABILITY_TEMPERATURE_AGE_III,
    }
}

/// [`evaluate`]'s victory-point score for `state`, from `me`'s side, mapped
/// onto an estimated win probability in `[0, 1]` through the age-calibrated
/// logistic
///
/// ```text
/// P(me wins) = 1 / (1 + exp(-evaluate(state, me, root) / T(state.age())))
/// ```
///
/// This is the exact mapping `duels-agent-mcts-eval` uses to turn a leaf's
/// evaluation into a value its search can back up, moved here (not
/// duplicated) so any other consumer — a diagnostic tool, a server-side
/// analysis endpoint, a future agent — reads the identical calibration rather
/// than inventing a second one. `mcts-eval` reads it from here; nothing about
/// what it computes changed when it moved.
///
/// # `T` comes from the `Root`, and *may* differ from the module constant
///
/// Since round eight the temperature is
/// [`EvalWeights::win_probability_temperature`], read off the [`Root`]'s
/// configuration rather than from [`win_probability_temperature`] directly.
/// At [`Config::default`] the two agree — by measurement, not by
/// construction — and under an older generation snapshot they do not, which is
/// the whole point: it is what makes a change to the leaf mapping A/B testable
/// against the generation before it. Read that field's docs before assuming
/// the two must agree.
///
/// A caller that wants the *calibration* rather than a *leaf* — a diagnostic,
/// a display — can use [`win_probability_from_value`], which reads the
/// constants and needs no [`Root`]. `duels-server`'s advanced-mode read does.
///
/// Consumes no randomness and is invariant to which hidden-information sample
/// produced `state`, exactly like [`evaluate`] itself: `state`'s only two
/// uses are the score `evaluate` computes and the age `T` is picked from, and
/// both are pure functions of public information — checked alongside
/// `evaluate` itself, for every scenario in
/// `tests/determinization_invariance.rs`, not just once.
///
/// # Known limitation, inherited from where this used to live
///
/// The temperature was fitted on positions each scored against **their own**
/// [`Root`] (one fresh `Root` per decision, as `phased` and this function's
/// direct callers do). A caller that instead prices every position in a
/// search against one `Root` fixed at the tree's own root (as `mcts-eval`
/// does, deliberately, since rebuilding one per node is unaffordable) is
/// scoring a hybrid the fit never saw, and a deep position may be mapped
/// slightly off. This is a known, measured characteristic of that caller, not
/// a defect in this function — see `duels-agent-mcts-eval`'s crate docs for
/// the full account.
pub fn win_probability(state: &GameState, me: Player, root: &Root) -> f64 {
    let t = root
        .config
        .eval
        .win_probability_temperature(state.age().max(1));
    1.0 / (1.0 + (-evaluate(state, me, root) / t).exp())
}

/// The pure calibrated logistic underneath [`win_probability`], for a caller
/// that already has a victory-point-scale number and an age from somewhere
/// other than a single [`evaluate`] call — [`expected_value`]'s
/// chance-averaged result being the motivating case (there is no single
/// post-action `GameState` to hand [`win_probability`] when an action resolves
/// a chance node). Split out so its fixed points, monotonicity and range are
/// also unit-testable without a [`GameState`]/[`Root`] to drive `evaluate`
/// through.
pub fn win_probability_from_value(value: f64, age: u8) -> f64 {
    1.0 / (1.0 + (-value / win_probability_temperature(age)).exp())
}

/// [`evaluate`] with the remaining pending-resolution budget explicit.
fn evaluate_at(state: &GameState, me: Player, root: &Root, depth: u8) -> f64 {
    if let Some(result) = state.result() {
        return match result {
            GameResult::Win { winner, .. } if winner == me => root.config.eval.instant_result,
            GameResult::Win { .. } => -root.config.eval.instant_result,
            GameResult::Draw => 0.0,
        };
    }
    // Finish the mover's own turn before judging it. See [`PendingModel`].
    if root.config.pending_model == PendingModel::Completed
        && depth > 0
        && state.pending().is_some()
    {
        if let Some(v) = resolve_pending(state, me, root, depth) {
            return v;
        }
    }
    // Rails B, C and C' — see [`rails`]. A rail *replaces* the weighted sum
    // rather than adding to it: the question it answers ("is this position
    // already decided, and for whom?") is not commensurable with a few
    // victory points of city quality, and a magnitude large enough to
    // dominate every ordinary term would be indistinguishable from a
    // replacement anyway. Antisymmetric by construction, so the evaluation
    // stays zero-sum.
    if let Some(v) = rails::rail_value(
        state,
        me,
        root.age,
        root.config.rails,
        root.config.eval.imminent,
    ) {
        return v;
    }
    let sum = player_value(state, me, root) - player_value(state, me.other(), root)
        + menu::menu_term(state, me, root.age, &root.menu, &root.config.eval.menu);
    // The one place the output scale is applied. Guarded rather than
    // multiplied by `1.0`, so `value_scale = 1.0` is bit-identical to the
    // arithmetic before this knob existed. Deliberately below the rails and
    // the terminal result, which are magnitudes rather than judgements — see
    // [`EvalWeights::value_scale`].
    if root.config.eval.value_scale == 1.0 {
        sum
    } else {
        sum * root.config.eval.value_scale
    }
}

/// Finish a turn the engine left mid-effect, and score what it leaves.
///
/// The pending choice belongs to `state.current_player()` — every
/// [`duels_core::state::Pending`] variant is created by that player's own
/// action and resolved by them before the turn passes — so the option taken is
/// the one *they* like best, whichever side `me` happens to be. That is what
/// keeps the whole thing **antisymmetric**: the resolution picks the same
/// option under `evaluate(s, me)` and `evaluate(s, me.other())`, because the
/// key it maximises is the same number in both (`evaluate` is exactly
/// antisymmetric, and `a - b` is exactly `-(b - a)` in IEEE-754), and the value
/// it returns then negates with `me` like any other.
///
/// Returns `None` — and so falls back to scoring the pending state as it
/// stands — only when the engine offers no legal resolution at all, which it
/// never does: `legal_actions` is empty exactly when the game is over, and a
/// finished game never carries a pending effect.
fn resolve_pending(state: &GameState, me: Player, root: &Root, depth: u8) -> Option<f64> {
    let resolver = state.current_player();
    let sign = if resolver == me { 1.0 } else { -1.0 };
    // A pending resolution reveals nothing: `engine::slots_revealed_by` is
    // empty for every one of these actions (none of them takes a card out of
    // the structure), so the single trivial outcome is the whole chance node.
    let trivial = engine::Outcome::default();

    // The destroy discount needs a reference to take a fraction *of*. The
    // pending state's own score is the natural one: it is what the position is
    // worth with the effect not yet applied, it is the same for every target,
    // and at `replace = 0` the blend collapses to the resolved value exactly,
    // so the knob is provably a no-op when it is switched off.
    let discount = root.config.destroy_replace_discount
        && matches!(
            state.pending(),
            Some(duels_core::state::Pending::Destroy { .. })
        );
    let unresolved = if discount {
        let sum = player_value(state, me, root) - player_value(state, me.other(), root)
            + menu::menu_term(state, me, root.age, &root.menu, &root.config.eval.menu);
        if root.config.eval.value_scale == 1.0 {
            sum
        } else {
            sum * root.config.eval.value_scale
        }
    } else {
        0.0
    };

    let mut best: Option<(f64, f64)> = None;
    for option in engine::legal_actions(state) {
        let mut next = *state;
        if engine::apply_with_outcome_unchecked(&mut next, option, &trivial).is_err() {
            continue;
        }
        let mut value = evaluate_at(&next, me, root, depth - 1);
        if discount {
            if let Action::DestroyOpponentCard { card } = option {
                let replace = root.destroy_replaceability(card);
                // Guarded rather than multiplied by `1.0`: `u + (v - u) * 1.0`
                // is not bit-identical to `v`, and this knob has to be an
                // exact no-op wherever nothing can be replaced — which is
                // every Age III destroy.
                if replace > 0.0 {
                    value = unresolved + (value - unresolved) * (1.0 - replace);
                }
            }
        }
        let key = sign * value;
        if best.is_none_or(|(b, _)| key > b) {
            best = Some((key, value));
        }
    }
    best.map(|(_, value)| value)
}

/// Every term, read for one player and weighted by *that player's* root-fixed
/// commitment multipliers.
fn player_value(state: &GameState, p: Player, root: &Root) -> f64 {
    let e = &root.config.eval;
    let c = &root.config;
    let w = &root.weights[p.index()];
    let breakdown = scoring::breakdown(state, p);

    // --- fading with commitment -------------------------------------------
    let points = w.vp * e.vp_projection * terms::card_and_token_vp(&breakdown);

    // Coins. `Legacy` splits into three terms (a floored points channel, a
    // capped race-liquidity bonus, and a shortfall penalty inside `economy`);
    // `Smooth` replaces all three with one continuous function.
    let (liquidity, race_liquidity, coin_safety) = match c.coin_model {
        CoinModel::Legacy => (
            w.liquidity * e.coins_div3 * f64::from(breakdown.coins),
            w.race_liquidity
                * e.race_card_liquidity
                * terms::race_liquidity(state, p, e.race_liquidity_cap),
            e.coin_safety_penalty * -terms::coin_shortfall(state, p, e.coin_safety_floor),
        ),
        CoinModel::Smooth => (
            w.liquidity * e.coins_div3 * terms::coin_points(state, p, e.coin_endgame_decisions)
                + terms::coin_liquidity(state, p, e.coin_smooth_beta, e.coin_smooth_ref),
            0.0,
            0.0,
        ),
    };

    // How much of what this city produces is still fixable. In Age III the
    // answer is "none of it" — there is no brown or grey card left in the
    // game — so the development credit and the resource bill both stop being
    // projections and start being facts, and are worth more accordingly.
    let lock = 1.0 + e.production_lock_in * root.supply.production_lock_in;
    let development = w.development
        * e.development
        * lock
        * terms::development_value_with(
            state,
            p,
            &root.supply,
            e.development_take_rate,
            // With `Bill` in force the post's value arrives through the lower
            // `price_r` it produces; crediting it separately would double it.
            c.economy_model == EconomyModel::Legacy,
        );
    let chain_equity =
        w.development * e.chain_equity * menu::chain_equity(state, p, root.menu.chain());

    let market = match c.economy_model {
        EconomyModel::Legacy => e.resource_vulnerability * -terms::average_trade_price(state, p),
        EconomyModel::Bill => {
            e.resource_bill
                * lock
                * -terms::resource_bill(state, p, &root.supply, e.development_take_rate)
                / 3.0
        }
    };
    let economy = w.economy * (coin_safety + market);

    // --- sharpening with commitment ---------------------------------------
    // `ScienceProgress::Root` — the default — is spelled as the bare
    // `w.science` it always was, so the option's off value is the same
    // expression rather than a reconstruction of it. See
    // [`ScienceProgress`] for why the leaf reading is opt-in and not simply
    // the fix.
    let science_weight = match root.config.blend.science_progress {
        ScienceProgress::Root => w.science,
        ScienceProgress::Leaf => {
            w.science_at(state.player(p).distinct_science(), &root.config.blend)
        }
    };
    let science = science_weight * e.science_ladder * terms::science_ladder(state, p, &e.science);
    let military = match c.military_model {
        MilitaryModel::Legacy => {
            w.military * e.military_position * terms::military_position(state, p)
        }
        MilitaryModel::Band => {
            w.military * e.military_band * terms::military_band(state, p, &root.smoothing)
                + e.military_loot * terms::military_loot(state, p, &root.smoothing)
        }
    };

    // --- never scaled -----------------------------------------------------
    let urgency = e.military_endgame_urgency * terms::military_urgency(state, p);
    let start = terms::next_age_start(state, p, e);
    let wonders = match c.wonder_model {
        WonderModel::Flat => e.wonder_potential * terms::wonder_potential(state, p, e),
        WonderModel::Budget => terms::wonder_potential_budget(state, p, &root.wonders),
        // `p_build` is read off `state` — the position being scored — and not
        // off `root`. See `terms::wonder_potential_rationed`.
        WonderModel::Rationed => e.wonder_potential * terms::wonder_potential_rationed(state, p, e),
    };
    // The opponent-menu term subsumes this one — a free chain build is just
    // one kind of high-value accessible card, and it is priced there properly
    // instead of at a flat `2 + VP`.
    let gift = if e.menu.lambda == 0.0 {
        -e.deny_chain_gift * terms::chain_gift_exposure(state, p, root.age)
    } else {
        0.0
    };
    // The forward half of a built guild's majority scoring. `breakdown` — read
    // into `points` above — already carries the snapshot half.
    let guilds = if e.guild_projection == 0.0 {
        0.0
    } else {
        e.guild_projection * root.guilds.projection(state, p)
    };
    // What this city's yellow cards will add to the discards it has not made
    // yet. `coin_marginal` is root-fixed, like every other price; the yellow
    // count and the decision budget are read here.
    let yellow = if e.yellow_equity == 0.0 {
        0.0
    } else {
        e.yellow_equity
            * terms::yellow_equity(
                state,
                p,
                root.menu.take(p).coin_marginal,
                e.yellow_discard_rate,
            )
    };
    // What the progress tokens this city already holds are worth for the rest
    // of the game, beyond the printed points `breakdown` scores. Root-fixed
    // prices, post-action ownership — see [`terms::TokenTable`].
    let tokens = if e.token_equity == 0.0 {
        0.0
    } else {
        e.token_equity * terms::token_equity(state, p, &root.tokens)
    };
    // A turn in hand, as against round six's projection of one still under a
    // wonder. See [`terms::to_move`].
    let tempo = if e.to_move == 0.0 {
        0.0
    } else {
        e.to_move * terms::to_move(state, p)
    };

    points
        + liquidity
        + development
        + chain_equity
        + economy
        + science
        + military
        + race_liquidity
        + urgency
        + start
        + wonders
        + gift
        + guilds
        + yellow
        + tokens
        + tempo
}

/// The probability-weighted expected value of taking `action` in `state`,
/// plus the action's own denial term.
///
/// The chance-expectation machinery is `duels-agent-greedy-ev`'s, unchanged:
/// enumerate every way the action's randomness could resolve via
/// [`engine::chance_outcomes`] (a single certain outcome for the large
/// majority of actions), apply each to its own copy of `state`, and average
/// the scores by their true probabilities rather than committing to one
/// sampled guess.
pub fn expected_value(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
    let outcomes = engine::chance_outcomes(state, action);
    let mut acc = 0.0;
    for (outcome, prob) in &outcomes {
        let mut next = *state;
        let value = match engine::apply_with_outcome(&mut next, action, outcome) {
            Ok(_) => evaluate(&next, me, root),
            // `action` came from `legal_actions` for `state` and `outcome`
            // from `chance_outcomes` for the same pair, so this is
            // unreachable; score the pre-action state rather than silently
            // dropping probability mass from the expectation.
            Err(_) => evaluate(state, me, root),
        };
        acc += prob * value;
    }
    acc + root.denial_term(action)
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::data::Science;
    use duels_core::scoring::VictoryKind;
    use duels_core::testing::StateBuilder;
    // `rand` is a dev-dependency only: the engine takes an explicitly seeded
    // `StdRng`, and nothing this crate ships needs one.
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Twenty of Age II's twenty-three cards, dealt into a real structure
    /// (so some slots are genuinely face down and the science supply model
    /// has something to work with, unlike `open_slots`, which reveals
    /// everything and leaves every unknown-pool weight at zero).
    const AGE_TWO_DEAL: [&str; 20] = [
        "sawmill",
        "brickyard",
        "shelf-quarry",
        "glassblower",
        "drying-room",
        "walls",
        "horse-breeders",
        "barracks",
        "archery-range",
        "parade-ground",
        "library",
        "dispensary",
        "school",
        "laboratory",
        "courthouse",
        "statue",
        "temple",
        "aqueduct",
        "rostrum",
        "forum",
    ];

    /// A mid-game Age II position with a full structure, `built` in Player
    /// One's city, and enough coins for the cost engine not to be the binding
    /// constraint.
    fn age_two_position(built: &[&str]) -> GameState {
        StateBuilder::new()
            .age(2)
            .deal(&AGE_TWO_DEAL)
            .built(Player::One, built)
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build()
    }

    fn advanced_game(seed: u64, steps: usize) -> GameState {
        let mut st = engine::new_game(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x55);
        for _ in 0..steps {
            let actions = engine::legal_actions(&st);
            if actions.is_empty() {
                break;
            }
            let a = actions[(st.turn() as usize * 7) % actions.len()];
            engine::apply(&mut st, a, &mut rng).unwrap();
        }
        st
    }

    // -----------------------------------------------------------------
    // The commitment guard: nobody is committed to anything on turn one.
    // -----------------------------------------------------------------

    /// The trap this whole design exists to avoid: `M_sci` alone reads a
    /// substantial magnitude for a player holding *no symbols at all*,
    /// because with three ages still to come the supply model cannot rule the
    /// race out. If that went straight into the blend, every player would
    /// read as partly science-committed from move one and the science ladder
    /// would be boosted in positions where it means nothing.
    #[test]
    fn a_fresh_game_reads_as_uncommitted_for_both_players() {
        for seed in 0..8u64 {
            let st = engine::new_game(seed);
            let root = Root::new(&st, st.current_player(), Config::default());
            for p in Player::ALL {
                let c = root.commitment(p);
                assert_eq!(
                    c.c_sci.to_bits(),
                    0.0f64.to_bits(),
                    "seed {seed}: c_sci for {p:?} is {} with no symbols held",
                    c.c_sci
                );
                assert!(
                    c.c_mil < 0.10,
                    "seed {seed}: c_mil for {p:?} is {} from the centre",
                    c.c_mil
                );
                assert!(c.c < 0.10, "seed {seed}: c for {p:?} is {}", c.c);
                assert!(
                    c.s < 0.01,
                    "seed {seed}: S(c) for {p:?} is {}, so weights have already moved",
                    c.s
                );
            }
        }
    }

    /// ...and the raw magnitude really is large enough for that to have been
    /// a live trap, so the test above is not vacuous.
    #[test]
    fn the_raw_science_magnitude_alone_would_have_been_misleading() {
        let st = advanced_game(3, 12);
        let r = duels_strategy::science_read(&st, st.current_player());
        assert_eq!(r.distinct, 0, "test setup: expected no symbols held");
        assert!(
            r.magnitude > 0.2,
            "M_sci with no symbols held is only {}, so the prog_sci guard would be pointless",
            r.magnitude
        );
    }

    // -----------------------------------------------------------------
    // Monotonicity properties
    // -----------------------------------------------------------------

    #[test]
    fn commitment_is_zero_at_the_bottom_of_both_races() {
        let blend = Blend::default();
        // No symbols held: c_sci is exactly zero whatever the magnitude says.
        let st = StateBuilder::new().age(1).conflict(0).build();
        let sci = duels_strategy::science_read(&st, Player::One);
        let mil = duels_strategy::military_read(&st, Player::One);
        assert_eq!(sci.distinct, 0);
        assert_eq!(
            mil.need,
            duels_core::data::military().capital_distance,
            "a centred pawn is the full capital distance away"
        );
        let c = Commitment::of(&sci, &mil, 0.0, &blend);
        assert_eq!(c.c_sci.to_bits(), 0.0f64.to_bits());
        assert_eq!(c.s_sci.to_bits(), 0.0f64.to_bits());
        // need == capital_distance and M_mil == 0 must give c_mil == 0.
        if mil.magnitude == 0.0 {
            assert_eq!(c.c_mil.to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn commitment_rises_with_symbols_held() {
        let blend = Blend::default();
        let commit = |built: &[&str]| -> f64 {
            let st = age_two_position(built);
            let sci = duels_strategy::science_read(&st, Player::One);
            let mil = duels_strategy::military_read(&st, Player::One);
            Commitment::of(&sci, &mil, 0.0, &blend).c_sci
        };
        let none = commit(&[]);
        let two = commit(&["workshop", "apothecary"]);
        let four = commit(&["workshop", "apothecary", "scriptorium", "pharmacist"]);
        assert_eq!(none.to_bits(), 0.0f64.to_bits());
        assert!(two > none, "two symbols: {two} vs {none}");
        assert!(four > two, "four symbols: {four} vs {two}");
    }

    // -----------------------------------------------------------------
    // The un-blended baseline
    // -----------------------------------------------------------------

    /// `S(0) = 0` is exact, so a genuinely uncommitted position and a
    /// deliberately switched-off blend must produce *the same weight vector,
    /// bit for bit* — and therefore the same evaluation of every legal
    /// action. This is what "the un-blended baseline" means for this crate:
    /// not another agent's code path, but this agent's own fixed-weight
    /// limit.
    #[test]
    fn the_blend_off_and_a_zero_commitment_position_agree_bit_for_bit() {
        let st = engine::new_game(17);
        let me = st.current_player();

        let on = Root::new(&st, me, Config::default());
        let off = Root::new(
            &st,
            me,
            Config {
                blend: Blend::off(),
                ..Config::default()
            },
        );

        // The military race is not *quite* dead cold at the very start, so
        // pin the comparison to the science half, which is exactly zero, and
        // assert the rest separately.
        for p in Player::ALL {
            let a = on.weights(p);
            let b = off.weights(p);
            assert_eq!(
                a.science.to_bits(),
                b.science.to_bits(),
                "science multiplier for {p:?}: {} vs {}",
                a.science,
                b.science
            );
        }

        // A hand-built position with both races at the floor: no symbols held
        // anywhere, the pawn centred, and — the part that takes an Age III
        // position with no red cards left — no shields obtainable at all, so
        // `M_mil` is exactly zero rather than merely small. Every weight must
        // then match the switched-off blend bit for bit, and so must the
        // evaluation of every legal action.
        let cold = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .conflict(0)
            .coins(Player::One, 5)
            .coins(Player::Two, 5)
            .current(Player::One)
            .build();
        for p in Player::ALL {
            let c = Root::new(&cold, cold.current_player(), Config::default())
                .commitment(p)
                .c;
            assert_eq!(
                c.to_bits(),
                0.0f64.to_bits(),
                "test setup: {p:?} is {c} committed, so this is not the cold case"
            );
        }
        let me = cold.current_player();
        let on = Root::new(&cold, me, Config::default());
        let off = Root::new(
            &cold,
            me,
            Config {
                blend: Blend::off(),
                ..Config::default()
            },
        );
        for p in Player::ALL {
            let (a, b) = (on.weights(p), off.weights(p));
            for (name, x, y) in [
                ("vp", a.vp, b.vp),
                ("liquidity", a.liquidity, b.liquidity),
                ("development", a.development, b.development),
                ("science", a.science, b.science),
                ("military", a.military, b.military),
                ("race_liquidity", a.race_liquidity, b.race_liquidity),
                ("economy", a.economy, b.economy),
            ] {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{name} for {p:?} differs: {x} vs {y}"
                );
            }
        }
        for action in engine::legal_actions(&cold) {
            let a = expected_value(&cold, action, me, &on);
            let b = expected_value(&cold, action, me, &off);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{action:?} scored {a} blended and {b} un-blended"
            );
        }
    }

    #[test]
    fn every_multiplier_is_exactly_one_when_the_blend_is_off() {
        let st = advanced_game(5, 30);
        let me = st.current_player();
        let off = Root::new(
            &st,
            me,
            Config {
                blend: Blend::off(),
                ..Config::default()
            },
        );
        let floor = Blend::default().floor_race_liq;
        for p in Player::ALL {
            let w = off.weights(p);
            for (name, v) in [
                ("vp", w.vp),
                ("liquidity", w.liquidity),
                ("development", w.development),
                ("science", w.science),
                ("military", w.military),
                ("economy", w.economy),
            ] {
                assert_eq!(v.to_bits(), 1.0f64.to_bits(), "{name} for {p:?} = {v}");
            }
            assert_eq!(w.race_liquidity.to_bits(), floor.to_bits());
        }
    }

    // -----------------------------------------------------------------
    // Root-fixing
    // -----------------------------------------------------------------

    // The counting half of root-fixing — one `Root` per decision, however
    // many candidates it scores — is an assertion about a *caller*, so it
    // lives with the caller:
    // `duels-agent-phased`'s `root_weights_are_built_exactly_once_per_choose`
    // reads `PhasedAgent::root_builds`. What this crate can and does pin is
    // the behavioural half, below: the tables do not move once built, and a
    // candidate is scored against the root's prices and the root's weights.

    /// Root-fixing for the *pricing* tables, not just the weights.
    ///
    /// `menu`'s take-value function calls the cost engine, which is the
    /// expensive part of this crate. The tables are built inside
    /// [`Root::new`], so the counter that already proves the blend is
    /// evaluated once per decision proves the same for them — but the
    /// behavioural half needs its own test: a candidate that changes what a
    /// card costs must still be scored against the *root* price, or a move
    /// would be credited once for the position it creates and again for
    /// having made the menu look different.
    #[test]
    fn the_menu_pricing_tables_are_root_fixed() {
        // Player One holds nothing; taking the Glassworks would halve what
        // every glass-costing card costs them.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "glassworks"), (19, "baths")])
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let glass = st.face_up_card(18).expect("slot 18 is face up");
        let before = root.menu().value(me, glass);

        // Apply the move that changes the pricing context...
        let mut after = st;
        let mut rng = StdRng::seed_from_u64(5);
        engine::apply(&mut after, Action::Build { slot: 18 }, &mut rng).unwrap();
        let rebuilt = Root::new(&after, after.current_player(), Config::default());

        // ...the root table still reports the root price, and a table rebuilt
        // on the result genuinely disagrees, so this is not vacuous.
        assert_eq!(
            root.menu().value(me, glass).to_bits(),
            before.to_bits(),
            "the root table moved without anybody rebuilding it"
        );
        let other = st.face_up_card(19).expect("slot 19 is face up");
        assert_ne!(
            root.menu().value(me, other).to_bits(),
            rebuilt.menu().value(me, other).to_bits(),
            "the pricing context did not actually change, so this test proves nothing"
        );
    }

    /// The behavioural half of root-fixing: a candidate action that would
    /// materially raise the mover's own commitment must still be scored under
    /// the *root* weights. Re-reading the blend on the result gives a
    /// different number, and that difference is exactly the double-count the
    /// design forbids — the move would be credited once through the term's
    /// value rising and again through the weight on that term rising.
    #[test]
    fn a_committing_move_is_scored_under_the_root_weights_not_its_own() {
        // Player One is two shields from the capital; "circus" (two shields)
        // in slot 18 takes them the whole way, which is about as large a
        // change to `c_mil` as one move can make.
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(4)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let action = Action::Build { slot: 18 };

        let mut after = st;
        let mut rng = StdRng::seed_from_u64(1);
        engine::apply(&mut after, action, &mut rng).expect("the red card should be buildable");
        assert!(
            !after.is_over(),
            "test setup: the move must not end the game"
        );

        let rebuilt = Root::new(&after, me, Config::default());
        assert!(
            rebuilt.commitment(me).s_mil > root.commitment(me).s_mil,
            "test setup: the move should raise the mover's military commitment ({} -> {})",
            root.commitment(me).s_mil,
            rebuilt.commitment(me).s_mil
        );

        // The score the agent actually assigns, under the root weights, and
        // the one it would assign if it re-read the blend on the result.
        let scored = evaluate(&after, me, &root);
        let recomputed = evaluate(&after, me, &rebuilt);
        assert_ne!(
            scored.to_bits(),
            recomputed.to_bits(),
            "the two are indistinguishable, so this test proves nothing"
        );
        // And the value the agent uses is the root-weighted one.
        assert_eq!(
            expected_value(&st, action, me, &root).to_bits(),
            (scored + root.denial_term(action)).to_bits()
        );
    }

    // -----------------------------------------------------------------
    // The individual terms
    // -----------------------------------------------------------------

    /// Grey (glass, papyrus) is the scarcest production in the game — two
    /// cards in Age I, two in Age II, none at all in Age III — and it is what
    /// most wonders ask for. Nothing in the development term says so; it
    /// falls out of counting what the remaining pool actually costs.
    #[test]
    fn the_development_term_prices_a_players_own_production() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .built(Player::One, &["glassworks", "press"])
            .current(Player::One)
            .build();
        let board = duels_strategy::Board::of(&st);
        let supply = DevSupply::of(&board);
        assert!(supply.pool_size > 0);

        let producer = terms::development_value(&st, Player::One, &supply, 0.6);
        let empty = terms::development_value(&st, Player::Two, &supply, 0.6);
        assert_eq!(empty, 0.0, "a city producing nothing develops nothing");
        assert!(
            producer > 0.0,
            "two grey cards should be worth something: {producer}"
        );

        // ...and the split by resource attributes it to glass and papyrus.
        let split = terms::development_by_resource(&st, Player::One, &supply, 0.6);
        let glass = split[duels_core::data::Resource::Glass.index()];
        let papyrus = split[duels_core::data::Resource::Papyrus.index()];
        assert!(glass > 0.0 && papyrus > 0.0, "{split:?}");
        assert!((glass + papyrus - producer).abs() < 1e-9, "{split:?}");
    }

    #[test]
    fn an_unbuilt_wonder_makes_the_resources_it_needs_more_valuable() {
        let build = |wonders: &[&str]| -> f64 {
            let st = StateBuilder::new()
                .age(1)
                .open_slots(&[(18, "clay-pool"), (19, "quarry")])
                .built(Player::One, &["lumber-yard"])
                .wonders(Player::One, wonders)
                .current(Player::One)
                .build();
            let supply = DevSupply::of(&duels_strategy::Board::of(&st));
            terms::development_value(&st, Player::One, &supply, 0.6)
        };
        // The Pyramids need 3 stone; the Great Lighthouse needs wood. Only
        // the latter raises what a Lumber Yard is worth.
        let none = build(&[]);
        let stone = build(&["the-pyramids"]);
        let wood = build(&["the-great-lighthouse"]);
        assert_eq!(none.to_bits(), stone.to_bits(), "{none} vs {stone}");
        assert!(wood > none, "{wood} vs {none}");
    }

    #[test]
    fn the_next_age_start_term_scores_the_projected_starter() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(19, "clay-pool")])
            .conflict(0)
            .current(Player::One)
            .build();
        let w = EvalWeights::default();
        assert_eq!(terms::projected_starter(&st), Some(Player::One));
        assert_eq!(
            terms::next_age_start(&st, Player::One, &w),
            w.next_age_start[0]
        );
        assert_eq!(terms::next_age_start(&st, Player::Two, &w), 0.0);

        // Age III has no next age, so the term is off entirely.
        let late = StateBuilder::new()
            .age(3)
            .open_slots(&[(19, "clay-pool")])
            .conflict(0)
            .current(Player::One)
            .build();
        assert_eq!(terms::next_age_start(&late, Player::One, &w), 0.0);
    }

    #[test]
    fn four_symbols_with_a_strong_token_row_reads_higher_than_without() {
        let four = ["workshop", "apothecary", "scriptorium", "pharmacist"];
        let make = |tokens: &[&str]| {
            StateBuilder::new()
                .age(2)
                .built(Player::One, &four)
                .board_tokens(tokens)
                .open_slots(&[(18, "clay-pool"), (19, "quarry")])
                .current(Player::One)
                .build()
        };
        let w = ScienceWeights::default();
        let bare = make(&[]);
        let strong = make(&["law", "theology", "strategy"]);
        let a = terms::science_ladder(&bare, Player::One, &w);
        let b = terms::science_ladder(&strong, Player::One, &w);
        assert!(b > a, "strong token row: {b} vs bare {a}");
        // The ladder is the dominant part at four symbols.
        assert!(a >= w.ladder[4], "{a} < {}", w.ladder[4]);
    }

    #[test]
    fn a_threatened_pair_is_worth_more_than_a_completed_one() {
        // Holding one Mortar (a live half-pair) versus holding both, which
        // has already paid its token and threatens nothing further.
        let cards: Vec<&str> = terms::symbol_cards(Science::Mortar)
            .map(|c| c.def().id)
            .collect();
        let w = ScienceWeights::default();
        let half = StateBuilder::new()
            .age(2)
            .built(Player::One, &cards[..1])
            .board_tokens(&["philosophy"])
            .build();
        let both = StateBuilder::new()
            .age(2)
            .built(Player::One, &cards)
            .board_tokens(&["philosophy"])
            .pair_already_awarded(Player::One, Science::Mortar)
            .build();
        // Same distinct count, so the ladder entry is identical; only the
        // pair threat differs.
        assert_eq!(half.player(Player::One).distinct_science(), 1);
        assert_eq!(both.player(Player::One).distinct_science(), 1);
        assert!(
            terms::science_ladder(&half, Player::One, &w)
                > terms::science_ladder(&both, Player::One, &w)
        );
    }

    // -----------------------------------------------------------------
    // Whole-agent behaviour
    // -----------------------------------------------------------------

    fn eval_after(state: &GameState, action: Action, me: Player, root: &Root) -> f64 {
        let mut s = *state;
        let mut rng = StdRng::seed_from_u64(0x0C0F_FEE0);
        engine::apply(&mut s, action, &mut rng).expect("scenario action should be legal");
        evaluate(&s, me, root)
    }

    /// The move this evaluation would pick, ties broken by order rather than
    /// by an RNG draw.
    ///
    /// `duels-agent-phased` is the crate that turns scores into a decision,
    /// and it breaks ties uniformly at random from its own stream; this crate
    /// has no RNG and needs none. The positions below are built so that the
    /// intended move wins outright, so the two agree on all of them.
    fn best_action(state: &GameState, me: Player, root: &Root, legal: &[Action]) -> Action {
        let mut best = (f64::NEG_INFINITY, legal[0]);
        for &action in legal {
            let v = expected_value(state, action, me, root);
            if v > best.0 {
                best = (v, action);
            }
        }
        best.1
    }

    #[test]
    fn evaluation_prefers_the_move_that_wins_by_military_supremacy() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "clay-pool")])
            .conflict(7)
            .coins(Player::One, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let build = eval_after(&st, Action::Build { slot: 18 }, me, &root);
        let discard = eval_after(&st, Action::Discard { slot: 18 }, me, &root);
        assert!(build > discard, "build={build} discard={discard}");

        let mut after = st;
        let mut rng = StdRng::seed_from_u64(1);
        engine::apply(&mut after, Action::Build { slot: 18 }, &mut rng).unwrap();
        assert_eq!(
            after.result(),
            Some(GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::MilitarySupremacy,
            })
        );
    }

    #[test]
    fn evaluation_prefers_the_move_that_wins_by_scientific_supremacy() {
        let st = StateBuilder::new()
            .age(3)
            .built(
                Player::One,
                &[
                    "workshop",
                    "apothecary",
                    "scriptorium",
                    "pharmacist",
                    "academy",
                ],
            )
            .open_slots(&[(18, "university"), (19, "palace")])
            .coins(Player::One, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        assert_eq!(st.player(me).distinct_science(), 5);
        let root = Root::new(&st, me, Config::default());
        let win = eval_after(&st, Action::Build { slot: 18 }, me, &root);
        let other = eval_after(&st, Action::Build { slot: 19 }, me, &root);
        assert!(win > other, "win={win} other={other}");
    }

    #[test]
    fn evaluation_orders_win_above_draw_above_loss() {
        let root_of = |st: &GameState| Root::new(st, Player::One, Config::default());
        let finish = |one: &[&str], two: &[&str]| -> GameState {
            let mut st = StateBuilder::new()
                .built(Player::One, one)
                .built(Player::Two, two)
                .open_slots(&[(18, "clay-pool")])
                .current(Player::One)
                .build();
            let mut rng = StdRng::seed_from_u64(3);
            engine::apply(&mut st, Action::Discard { slot: 18 }, &mut rng).unwrap();
            assert!(st.result().is_some());
            st
        };
        let win = finish(&["palace"], &[]);
        let draw = finish(&["palace"], &["town-hall"]);
        let loss = finish(&[], &["palace"]);
        let r = root_of(&win);
        let w = EvalWeights::default();
        assert_eq!(evaluate(&win, Player::One, &r), w.instant_result);
        assert_eq!(evaluate(&draw, Player::One, &r), 0.0);
        assert_eq!(evaluate(&loss, Player::One, &r), -w.instant_result);
    }

    #[test]
    fn the_evaluation_is_antisymmetric_between_the_two_players() {
        for seed in 0..6u64 {
            for steps in [8usize, 20, 34] {
                let st = advanced_game(seed, steps);
                if st.is_over() {
                    continue;
                }
                let root = Root::new(&st, st.current_player(), Config::default());
                let a = evaluate(&st, Player::One, &root);
                let b = evaluate(&st, Player::Two, &root);
                assert!(
                    (a + b).abs() < 1e-9,
                    "seed {seed} steps {steps}: {a} and {b} are not opposites"
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // The terminal rails
    // -----------------------------------------------------------------

    /// Rail B, end to end through the evaluation: a one-shield-from-the-
    /// capital opponent with a red card on the table, and one candidate that
    /// takes that card away.
    ///
    /// The ordinary evaluation is happy to leave it there — a Quarry is a
    /// perfectly good pick — which is exactly the failure the rails exist to
    /// stop. `phased:base=v2` really does leave it, and this test asserts
    /// both halves so it cannot pass for the wrong reason.
    #[test]
    fn the_rails_block_a_loss_the_ordinary_evaluation_walks_into() {
        let position = || {
            StateBuilder::new()
                .age(3)
                .open_slots(&[(18, "circus"), (19, "palace")])
                .conflict(-7)
                .coins(Player::One, 30)
                .coins(Player::Two, 30)
                .current(Player::One)
                .build()
        };
        let st = position();
        let me = st.current_player();
        // Player Two is two shields from the capital and the Circus is worth
        // exactly two; anything that leaves it there loses.
        assert!(duels_strategy::closing_sources(&st, Player::Two).any());

        let legal = engine::legal_actions(&st);
        let chosen = best_action(&st, me, &Root::new(&st, me, Config::default()), &legal);
        assert!(
            matches!(
                chosen,
                Action::Build { slot: 18 }
                    | Action::Discard { slot: 18 }
                    | Action::BuildWonder { slot: 18, .. }
            ),
            "the rails let the opponent's winning card stand: chose {chosen:?}"
        );

        // Every candidate that leaves slot 18 alone is pinned at −imminent,
        // and every candidate that takes it is not.
        let root = Root::new(&st, me, Config::default());
        let w = EvalWeights::default();
        for &action in &legal {
            let touches = matches!(
                action,
                Action::Build { slot: 18 }
                    | Action::Discard { slot: 18 }
                    | Action::BuildWonder { slot: 18, .. }
            );
            let v = expected_value(&st, action, me, &root);
            if touches {
                assert!(v > -w.imminent, "{action:?} scored {v}");
            } else {
                assert!(v <= -w.imminent, "{action:?} scored {v}, not blocked");
            }
        }
    }

    /// ...and the guarantee really does come from the rail, not from the
    /// ordinary terms happening to agree.
    ///
    /// On this particular position the round-two agent blocks too — a
    /// two-card structure makes the closing red card the obviously
    /// attractive pick anyway. What it does *not* have is any guarantee: no
    /// candidate is scored anywhere near `-imminent`, so the ordering is
    /// decided by a handful of victory points and would flip under a wider
    /// structure. `examples/rail_audit.rs` is where the difference is
    /// measured on real games rather than argued from one position.
    #[test]
    fn without_the_rails_nothing_is_pinned_and_a_few_points_decide_it() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(-7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::v2());
        let w = Config::v2().eval;
        assert_eq!(w.imminent, 0.0, "v2 must carry no rail magnitude at all");
        for &action in &engine::legal_actions(&st) {
            let v = expected_value(&st, action, me, &root);
            assert!(
                v.abs() < 100.0,
                "{action:?} scored {v}: the round-two agent has no terminal \
                 rail, so nothing should be pinned"
            );
        }
    }

    /// Taking the win still outranks merely having one, so Rail A cannot be
    /// swallowed by Rail C′.
    #[test]
    fn an_actual_win_outranks_a_certain_one() {
        let w = EvalWeights::default();
        assert!(
            w.instant_result > w.imminent,
            "instant_result {} must dominate imminent {}",
            w.instant_result,
            w.imminent
        );
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let win = expected_value(&st, Action::Build { slot: 18 }, me, &root);
        let other = expected_value(&st, Action::Build { slot: 19 }, me, &root);
        assert_eq!(win, w.instant_result);
        assert!(win > other);
    }

    // -----------------------------------------------------------------
    // The menu's shield price
    // -----------------------------------------------------------------

    /// The differenced price of `k` shields is what the evaluation actually
    /// moves when the pawn advances `k` — which the one-sided price is not,
    /// in either magnitude or shape.
    #[test]
    fn the_differenced_shield_price_matches_what_the_evaluation_really_moves() {
        // A pawn one step short of the 3-5 band, so the second shield crosses
        // a boundary the first does not.
        let st = StateBuilder::new()
            .age(2)
            .deal(&AGE_TWO_DEAL)
            .conflict(2)
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let root = Root::new(&st, me, Config::default());
        let base = evaluate(&st, me, &root);

        for k in 1..=3i8 {
            let moved = StateBuilder::new()
                .age(2)
                .deal(&AGE_TWO_DEAL)
                .conflict(2 + k)
                .coins(Player::One, 20)
                .coins(Player::Two, 20)
                .current(Player::One)
                .build();
            let want = evaluate(&moved, me, &root) - base;
            let got = terms::military_shield_delta(
                &st,
                me,
                u8::try_from(k).unwrap(),
                root.smoothing(),
                root.config().eval.military_band,
                root.config().eval.military_loot,
                (root.weights(me).military, root.weights(me.other()).military),
            );
            assert!(
                (want - got).abs() < 1e-9,
                "{k} shields: the evaluation moves {want}, the price says {got}"
            );
        }
    }

    /// It is a finite difference, not `k` times a slope — which matters
    /// precisely because the scoring table's steps are not evenly spaced.
    ///
    /// The test has to switch the horizon on to show it, and that is the
    /// point of the horizon: at the default supply-wide smoothing the bands
    /// are so blurred that two shields really are worth almost exactly twice
    /// one, which is the "step function that behaves like a straight line"
    /// complaint written down as an assertion.
    #[test]
    fn the_shield_price_is_not_linear_in_the_number_of_shields() {
        let st = StateBuilder::new()
            .age(2)
            .deal(&AGE_TWO_DEAL)
            .conflict(1)
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let me = st.current_player();
        let price = |config: Config, k: u8| {
            let root = Root::new(&st, me, config);
            terms::military_shield_delta(
                &st,
                me,
                k,
                root.smoothing(),
                root.config().eval.military_band,
                root.config().eval.military_loot,
                (root.weights(me).military, root.weights(me.other()).military),
            )
        };
        let sharp = Config {
            military_horizon: Some(3.0),
            ..Config::default()
        };
        assert_eq!(price(sharp, 0), 0.0);
        assert!(
            price(sharp, 2) > 2.0 * price(sharp, 1),
            "{} vs {}",
            price(sharp, 2),
            price(sharp, 1)
        );

        // ...and at the default width the same quantity is within a few
        // percent of linear.
        let wide = Config::default();
        let ratio = price(wide, 2) / (2.0 * price(wide, 1));
        assert!(
            (0.95..1.10).contains(&ratio),
            "the supply-wide smoothing should be near-linear, ratio {ratio}"
        );
    }

    // -----------------------------------------------------------------
    // The horizon-based smoothing width, and production lock-in
    // -----------------------------------------------------------------

    /// A horizon narrows the smoothing, which is the whole point: with the
    /// full remaining supply the "step function" is nearly a straight line.
    #[test]
    fn a_horizon_sharpens_the_bands_and_none_reproduces_the_old_width() {
        let shields = 20.0;
        let rounds = 18.0;
        assert_eq!(
            terms::horizon_supply(shields, rounds, None).to_bits(),
            shields.to_bits()
        );
        let wide = MilSmoothing::of(shields, 0.8, 0.35, 0.55);
        let narrow = MilSmoothing::of(
            terms::horizon_supply(shields, rounds, Some(3.0)),
            0.8,
            0.35,
            0.55,
        );
        assert!(narrow.s < wide.s, "{} vs {}", narrow.s, wide.s);
        // ...and a sharper width really does separate the boundary-crossing
        // shield from the one that crosses nothing.
        let contrast = |sm: &MilSmoothing| {
            let at = |c: i8| {
                let st = StateBuilder::new().age(2).conflict(c).build();
                terms::military_band(&st, Player::One, sm)
            };
            (at(3) - at(2)) - (at(2) - at(1))
        };
        assert!(contrast(&narrow) > contrast(&wide));
        // A horizon longer than the game is a no-op.
        assert_eq!(
            terms::horizon_supply(shields, rounds, Some(1000.0)).to_bits(),
            shields.to_bits()
        );
    }

    /// Age III really has no brown or grey card in it, so a city's production
    /// is frozen — the fact the lock-in factor is built on, checked against
    /// the card data rather than asserted from memory.
    #[test]
    fn production_is_completely_frozen_by_age_three() {
        let by_age = |age: u8| {
            duels_core::data::statics().age_masks[usize::from(age) - 1] & terms::production_mask()
        };
        assert_eq!(by_age(1).count_ones(), 8, "six brown and two grey in Age I");
        assert_eq!(
            by_age(2).count_ones(),
            5,
            "three brown and two grey in Age II"
        );
        assert_eq!(by_age(3).count_ones(), 0, "Age III prints no production");

        // ...so an Age III position reads a lock-in of exactly one.
        let late = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "town-hall")])
            .current(Player::One)
            .build();
        let supply = DevSupply::of(&duels_strategy::Board::of(&late));
        assert_eq!(supply.production_lock_in.to_bits(), 1.0f64.to_bits());

        // ...and an Age I position reads less than one, so it is a factor
        // rather than a constant.
        let early = engine::new_game(4);
        let early = DevSupply::of(&duels_strategy::Board::of(&early));
        assert!(
            early.production_lock_in < 0.5,
            "Age I lock-in is {}",
            early.production_lock_in
        );
    }

    /// The version snapshots are a chain, and every link has to be a distinct
    /// configuration: a snapshot that equals its successor names a round that
    /// changed nothing, and one that equals [`Config::default`] while *not*
    /// being the newest link is a snapshot that has silently drifted.
    ///
    /// [`Config::v9`] is the newest link and is deliberately today's default;
    /// see its documentation for what the next round owes this chain.
    ///
    /// **`v8` and `v9` are the one deliberate exception**, and it is exactly
    /// the case the paragraph above names: round nine really did change
    /// nothing about the default. It built two options, measured both and
    /// adopted neither, so its snapshot and round eight's are the same
    /// configuration. That is asserted rather than tolerated, so a later edit
    /// which pulls them apart has to say so here.
    #[test]
    fn the_generation_snapshots_are_a_chain_of_distinct_configurations() {
        let chain = [
            ("v1", Config::v1()),
            ("v2", Config::v2()),
            ("v3", Config::v3()),
            ("v4", Config::v4()),
            ("v5", Config::v5()),
            ("v6", Config::v6()),
            ("v7", Config::v7()),
            ("v8", Config::v8()),
        ];
        for (i, (name, cfg)) in chain.iter().enumerate() {
            for (later, other) in chain.iter().skip(i + 1) {
                assert_ne!(cfg, other, "{name} and {later} are the same configuration");
            }
        }
        // Round nine adopted neither of the options it built, so its snapshot
        // is round eight's. Pinned as an equality rather than left out of the
        // loop above: if some later edit makes them differ, the round-nine
        // narrative in this file and in `tests/round_nine_identity.rs` has
        // become wrong and should fail rather than quietly pass.
        assert_eq!(
            Config::v8(),
            Config::v9(),
            "round nine moved the default after all, so it owes this chain a \
             snapshot of its own"
        );
        // Round ten moved the default, so `v9` is a real delta from it and
        // `v8` is no longer an alias for it either.
        assert_ne!(Config::v9(), Config::default());
        assert_ne!(Config::v8(), Config::default());

        // Deliberately *not* `assert_eq!(Config::v9(), Config::default())`.
        // The newest snapshot in this chain is *by construction* a delta from
        // whatever the default is (`v1`-`v8` are deltas, not literals), so a
        // check that the newest link tracks the default cannot catch an
        // eleventh round silently redefining this generation.
        //
        // The guard that used to work lived with the consumer that had
        // something to lose: `duels-agent-mcts-uct`'s
        // `leaf::tests::the_pinned_generation_reproduces_its_golden_values`
        // held ~50 evaluations captured from `v6()` at the moment its search
        // was measured against it. That agent no longer consumes this crate at
        // all, and no current consumer pins a generation, so nothing
        // downstream enforces the *chain* today. See [`Config::v9`]'s
        // "contract for the next round".
        //
        // What does still exist downstream is `duels-agent-phased`'s
        // `tests/p_build_identity.rs`, which hashes twelve whole self-play
        // games under the default. It guards a different claim and is not a
        // generation pin, but it does mean a change to the shipping default
        // cannot reach an agent's decisions unnoticed — round ten confirmed
        // that by failing it.
    }

    // `spec_reports_the_expected_name_and_encoded_params`,
    // `choosing_only_ever_returns_one_of_the_offered_actions` and
    // `a_whole_game_of_self_play_terminates_and_stays_legal` are assertions
    // about an `Agent`, not about an evaluation, and live in
    // `duels-agent-phased` with the agent they are about. `params_string`
    // itself is still pinned from this side, by the identity tests in
    // `tests/`.

    // -----------------------------------------------------------------
    // Round four: finishing a turn the engine left mid-effect.
    // -----------------------------------------------------------------

    fn wonder(slug: &str) -> duels_core::data::WonderId {
        duels_core::data::WonderId::from_slug(slug).expect("a real wonder")
    }

    fn completed() -> Config {
        Config {
            pending_model: PendingModel::Completed,
            ..Config::default()
        }
    }

    /// A position where Player One can build `w` by burying the card in slot
    /// 18, with both cities rich enough that cost is never the binding
    /// constraint.
    fn wonder_position(w: &str, opponent_city: &[&str]) -> GameState {
        StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "clay-pool")])
            .wonders(Player::One, &[w])
            .built(Player::Two, opponent_city)
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build()
    }

    /// The bug, stated as a property of the engine rather than of this crate:
    /// these four wonders really do come back from `apply` with the effect not
    /// yet applied and the turn not yet passed.
    #[test]
    fn four_wonders_leave_the_engine_mid_effect() {
        use duels_core::state::Pending;
        let cases: [(&str, &[&str]); 3] = [
            ("circus-maximus", &["glassworks", "press"]),
            ("the-statue-of-zeus", &["lumber-yard", "clay-pit"]),
            ("the-mausoleum", &[]),
        ];
        for (slug, city) in cases {
            let mut st = wonder_position(slug, city);
            if slug == "the-mausoleum" {
                st = StateBuilder::new()
                    .age(3)
                    .open_slots(&[(18, "palace"), (19, "clay-pool")])
                    .wonders(Player::One, &[slug])
                    .discard(&["theater", "altar"])
                    .coins(Player::One, 40)
                    .coins(Player::Two, 40)
                    .current(Player::One)
                    .build();
            }
            let action = Action::BuildWonder {
                slot: 18,
                wonder: wonder(slug),
            };
            assert!(
                engine::legal_actions(&st).contains(&action),
                "{slug}: the build is not legal in the test position"
            );
            let mut next = st;
            engine::apply_with_outcome(&mut next, action, &engine::Outcome::default()).unwrap();
            assert!(
                next.pending().is_some(),
                "{slug}: the engine finished the effect after all"
            );
            assert_eq!(
                next.current_player(),
                Player::One,
                "{slug}: the turn passed before the effect resolved"
            );
            assert!(matches!(
                next.pending(),
                Some(Pending::Destroy { .. } | Pending::MausoleumBuild)
            ));
        }
    }

    /// A destroy that takes a real card out of the opponent's city is worth
    /// more than a destroy that has not happened yet — which is the whole of
    /// bug one in one assertion.
    #[test]
    fn completing_the_turn_credits_a_destroy_the_unresolved_evaluation_misses() {
        let st = wonder_position("circus-maximus", &["glassworks", "press"]);
        let action = Action::BuildWonder {
            slot: 18,
            wonder: wonder("circus-maximus"),
        };
        let me = Player::One;

        let flat = Root::new(&st, me, Config::v3());
        let full = Root::new(&st, me, completed());
        let before = expected_value(&st, action, me, &flat);
        let after = expected_value(&st, action, me, &full);
        assert!(
            after > before,
            "resolving the destroy should be worth something: {after} vs {before}"
        );

        // ...and the card it takes is the one that hurts most, not the first
        // one in index order: the destroy resolution really does choose.
        let mut resolved = st;
        engine::apply_with_outcome(&mut resolved, action, &engine::Outcome::default()).unwrap();
        let options = engine::legal_actions(&resolved);
        assert_eq!(options.len(), 2, "two grey cards to choose between");
        let scores: Vec<f64> = options
            .iter()
            .map(|&o| {
                let mut s = resolved;
                engine::apply_with_outcome(&mut s, o, &engine::Outcome::default()).unwrap();
                evaluate(&s, me, &full)
            })
            .collect();
        let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(
            evaluate(&resolved, me, &full).to_bits(),
            best.to_bits(),
            "the resolution did not take the maximum over the real options"
        );
    }

    /// The Mausoleum's retrieval is the one pending effect the engine can
    /// **chain**: `construct_card` runs the retrieved card's own effects, and a
    /// green card that completes a science pair sets a second pending choice.
    /// [`MAX_PENDING_DEPTH`] exists for exactly this, and this is the test that
    /// says the chain is real rather than hypothetical.
    #[test]
    fn a_mausoleum_retrieval_can_chain_into_a_progress_token_choice() {
        use duels_core::state::Pending;
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "clay-pool")])
            .wonders(Player::One, &["the-mausoleum"])
            // One Wheel already in the city; the School in the discard pile
            // carries the second, so retrieving it completes the pair.
            .built(Player::One, &["apothecary"])
            .discard(&["school"])
            .board_tokens(&["law", "theology", "strategy"])
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build();

        let mut after = st;
        engine::apply_with_outcome(
            &mut after,
            Action::BuildWonder {
                slot: 18,
                wonder: wonder("the-mausoleum"),
            },
            &engine::Outcome::default(),
        )
        .unwrap();
        assert_eq!(after.pending(), Some(Pending::MausoleumBuild));

        let mut chained = after;
        engine::apply_with_outcome(
            &mut chained,
            engine::legal_actions(&after)[0],
            &engine::Outcome::default(),
        )
        .unwrap();
        assert_eq!(
            chained.pending(),
            Some(Pending::ProgressToken),
            "the retrieval was supposed to complete a science pair"
        );
        assert_eq!(chained.current_player(), Player::One);

        // The evaluator walks the whole chain, so the state it finally scores
        // has no pending effect left and the turn really has passed.
        let root = Root::new(&st, Player::One, completed());
        let v = evaluate(&after, Player::One, &root);
        assert!(v.is_finite());
        // Resolving both levels is strictly better than stopping at the first:
        // the token is worth something.
        let one_level = evaluate_at(&after, Player::One, &root, 1);
        assert!(
            v > one_level,
            "walking the chain to the end should be worth more than stopping \
             one level in: {v} vs {one_level}"
        );
    }

    /// The evaluation stays exactly zero-sum through a pending resolution.
    #[test]
    fn resolving_a_pending_effect_stays_antisymmetric() {
        let st = wonder_position("circus-maximus", &["glassworks", "press"]);
        let mut after = st;
        engine::apply_with_outcome(
            &mut after,
            Action::BuildWonder {
                slot: 18,
                wonder: wonder("circus-maximus"),
            },
            &engine::Outcome::default(),
        )
        .unwrap();
        assert!(after.pending().is_some());

        for me in Player::ALL {
            let root = Root::new(&st, me, completed());
            let mine = evaluate(&after, me, &root);
            let theirs = evaluate(&after, me.other(), &root);
            assert_eq!(
                mine.to_bits(),
                (-theirs).to_bits(),
                "the resolution is not antisymmetric: {mine} vs {theirs}"
            );
        }
    }

    /// The destroy discount is a *no-op* at its off value, and moves the score
    /// towards the unresolved reference when there really is a replacement
    /// coming.
    #[test]
    fn the_destroy_replacement_discount_only_bites_while_the_market_can_replace() {
        let action = Action::BuildWonder {
            slot: 18,
            wonder: wonder("the-statue-of-zeus"),
        };
        // Age III: no brown or grey card left in the game, so nothing can be
        // replaced and the discount must vanish.
        let late = wonder_position("the-statue-of-zeus", &["lumber-yard", "clay-pit"]);
        let plain = Root::new(&late, Player::One, completed());
        let discounted = Root::new(
            &late,
            Player::One,
            Config {
                destroy_replace_discount: true,
                ..completed()
            },
        );
        assert_eq!(
            expected_value(&late, action, Player::One, &plain).to_bits(),
            expected_value(&late, action, Player::One, &discounted).to_bits(),
            "an Age III destroy is permanent, so the discount must be exactly zero"
        );
        for r in 0..duels_core::data::NUM_RESOURCES {
            assert_eq!(
                discounted.replace[r], 0.0,
                "Age III still thinks it can print production"
            );
        }

        // Age I, with the whole brown supply still to come: the same destroy
        // is worth strictly less than its undiscounted value.
        let early = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "palace"), (19, "clay-pool")])
            .wonders(Player::One, &["the-statue-of-zeus"])
            .built(Player::Two, &["lumber-yard", "clay-pit"])
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build();
        let plain = Root::new(&early, Player::One, completed());
        let discounted = Root::new(
            &early,
            Player::One,
            Config {
                destroy_replace_discount: true,
                ..completed()
            },
        );
        assert!(
            discounted.replace.iter().any(|&x| x > 0.0),
            "Age I should still have production sources in the pool"
        );
        // The discount *bites*: it blends the resolved value back towards the
        // pending state's own score by the modelled fraction, so the two must
        // differ. It is deliberately **not** asserted that the discounted
        // value is the *lower* of the two, which is what this test used to
        // claim. That only holds when the destroy's resolved score is above
        // the pending-state reference, and the reference is a distorted
        // quantity by construction — `menu_term` reads the mover as moving
        // again in a pending state, which is the whole reason
        // `PendingModel::Completed` exists. Round seven's re-weighting was
        // enough to flip the sign in this particular hand-built position
        // (plain -8.16, discounted -5.60), which is a fact about the reference
        // rather than about the discount, and is one more reason this knob is
        // off by default and measured at -0.9 / +5.8.
        let d = expected_value(&early, action, Player::One, &discounted);
        let pl = expected_value(&early, action, Player::One, &plain);
        assert_ne!(
            d.to_bits(),
            pl.to_bits(),
            "the discount did not bite on a replaceable destroy: \
             discounted {d}, plain {pl}"
        );
    }

    // -----------------------------------------------------------------
    // Round four: the wonder budget.
    // -----------------------------------------------------------------

    /// `p_build` rations the seven shared slots and the owner's remaining
    /// decisions, so an unbuilt wonder is not worth the same in a fresh Age I
    /// as it is with one slot and two turns left.
    #[test]
    fn the_wonder_budget_rations_slots_and_turns() {
        let budget = Config {
            wonder_model: WonderModel::Budget,
            ..Config::default()
        };

        // Fresh: eight unbuilt wonders, seven slots, a whole game of turns.
        let fresh = StateBuilder::new()
            .age(1)
            .deal(&AGE_TWO_DEAL)
            .wonders(
                Player::One,
                &["the-pyramids", "the-colossus", "the-sphinx", "piraeus"],
            )
            .wonders(
                Player::Two,
                &[
                    "the-mausoleum",
                    "the-great-library",
                    "the-appian-way",
                    "the-great-lighthouse",
                ],
            )
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let root = Root::new(&fresh, Player::One, budget);
        let p = root.wonders().p_build(Player::One);
        assert!(
            p > 0.0 && p < 1.0,
            "eight unbuilt wonders against seven slots should ration: {p}"
        );
        assert!(
            terms::wonder_potential_budget(&fresh, Player::One, root.wonders()) > 0.0,
            "an unbuilt wonder in Age I is worth something"
        );

        // Every slot gone: nothing left to ration.
        let full = StateBuilder::new()
            .age(3)
            .open_slots(&[(19, "clay-pool")])
            .wonders(Player::One, &["the-pyramids"])
            .wonders_built(
                Player::One,
                &["the-colossus", "the-sphinx", "the-hanging-gardens"],
            )
            .wonders_built(
                Player::Two,
                &[
                    "piraeus",
                    "the-appian-way",
                    "the-great-lighthouse",
                    "the-mausoleum",
                ],
            )
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        let root = Root::new(&full, Player::One, budget);
        assert_eq!(root.wonders().p_build(Player::One), 0.0);
        assert_eq!(
            terms::wonder_potential_budget(&full, Player::One, root.wonders()),
            0.0
        );
    }

    /// The per-effect price says something the flat one cannot: the Great
    /// Library's token draw is priced from the tokens actually set aside, so
    /// two different piles give two different answers — where
    /// [`terms::wonder_power`] gives a flat `+3` for both.
    #[test]
    fn the_wonder_budget_prices_the_great_library_from_the_real_token_pile() {
        let position = |aside: &[&str]| {
            StateBuilder::new()
                .age(2)
                .deal(&AGE_TWO_DEAL)
                .wonders(Player::One, &["the-great-library"])
                .set_aside_tokens(aside)
                .coins(Player::One, 20)
                .coins(Player::Two, 20)
                .current(Player::One)
                .build()
        };
        let budget = Config {
            wonder_model: WonderModel::Budget,
            ..Config::default()
        };
        let rich = position(&["law", "theology", "mathematics", "philosophy", "urbanism"]);
        let thin = position(&["masonry", "agriculture", "economy", "strategy", "urbanism"]);
        let gl = wonder("the-great-library");
        let a = Root::new(&rich, Player::One, budget);
        let b = Root::new(&thin, Player::One, budget);
        assert!(
            a.wonders().power(Player::One, gl) != b.wonders().power(Player::One, gl),
            "the token pile made no difference to the Great Library's price"
        );
        // The flat model cannot tell them apart at all.
        assert_eq!(
            terms::wonder_power(gl).to_bits(),
            terms::wonder_power(gl).to_bits()
        );
    }

    // `win_probability`'s calibration: moved here from `duels-agent-mcts-eval`
    // (which was the only caller until this function existed), not
    // duplicated — these two tests used to live there.

    #[test]
    fn the_temperature_lookup_is_the_calibrated_table() {
        assert_eq!(
            win_probability_temperature(1).to_bits(),
            WIN_PROBABILITY_TEMPERATURE_AGE_I.to_bits()
        );
        assert_eq!(
            win_probability_temperature(2).to_bits(),
            WIN_PROBABILITY_TEMPERATURE_AGE_II.to_bits()
        );
        assert_eq!(
            win_probability_temperature(3).to_bits(),
            WIN_PROBABILITY_TEMPERATURE_AGE_III.to_bits()
        );
        // Age III is the sharpest of the three, which is the finding the
        // per-age lookup exists for.
        assert!(win_probability_temperature(3) < win_probability_temperature(2));
        assert!(win_probability_temperature(2) < win_probability_temperature(1));
        // A flat constant would have been the overall fit, and it is bracketed
        // by the per-age ones.
        assert!(win_probability_temperature(3) < WIN_PROBABILITY_TEMPERATURE_OVERALL);
        assert!(WIN_PROBABILITY_TEMPERATURE_OVERALL < win_probability_temperature(1));
    }

    /// The mapping's three fixed points, plus its monotonicity and its range.
    #[test]
    fn the_sigmoid_maps_victory_points_onto_a_probability() {
        for age in 1..=3u8 {
            assert_eq!(win_probability_from_value(0.0, age), 0.5, "age {age}");
            let t = win_probability_temperature(age);
            // One temperature of advantage is the 73% point, by construction.
            let at_t = win_probability_from_value(t, age);
            assert!((at_t - 0.731_058_6).abs() < 1e-6, "age {age}: {at_t}");
            // Symmetric about a half, and monotone.
            assert!(
                (win_probability_from_value(t, age) + win_probability_from_value(-t, age) - 1.0)
                    .abs()
                    < 1e-12
            );
            let mut last = 0.0;
            for v in [-200.0, -50.0, -5.0, 0.0, 5.0, 50.0, 200.0] {
                let p = win_probability_from_value(v, age);
                assert!(p > last, "age {age}: not monotone at {v}");
                assert!((0.0..=1.0).contains(&p));
                last = p;
            }
        }
        // The same score is worth more in Age III, where the evaluation is
        // sharper.
        assert!(win_probability_from_value(10.0, 3) > win_probability_from_value(10.0, 1));
    }
}
