//! The policy layer: what is this position *about*, and which moves deserve
//! search attention?
//!
//! # Denial is a price, not a mode
//!
//! The first cut of this layer had a discrete gate: if the opponent could win
//! outright next turn, promote every move that took the card away, and
//! otherwise ignore denial entirely. That is exactly wrong in the middle,
//! which is where most of the game is. A player who is two symbols short with
//! four copies still on the table is not "imminent", but letting them take one
//! is worth several points of eventual swing, and a player who is one symbol
//! short with the only copy behind an unbuilt Mausoleum is "imminent" and
//! cannot be denied at all.
//!
//! So denial is priced continuously. Each race carries a magnitude
//! `M ∈ [0, 1]` per player — [`crate::ScienceRead::magnitude`],
//! [`crate::MilitaryRead::magnitude`] — and an action is worth
//!
//! ```text
//! deny_vp(a) = game_swing_vp × stakes × (ΔM_science + ΔM_military)
//! ΔM_race(a) = M_race(opponent) − M_race(opponent | a)
//! ```
//!
//! which drops straight into the same linear victory-point channel as
//! [`action_vp_value`]. A negative `ΔM` — an action that uncovers the shields
//! the opponent needed, or hands them the slot they could not reach — is a
//! *cost*, which is why this subsumes the old separate "exposure risk" term
//! rather than sitting beside it.
//!
//! `stakes` scales that by who can afford a race: a player behind on points
//! should be gambling on a race of their own rather than spending turns
//! denying one (0.6×), a player ahead has more to lose (1.4×).
//!
//! # What is left of the discrete rules
//!
//! One rail: an action that takes a *certain* opposing win (`M == 1`) and
//! makes it uncertain, or that closes this player's own race, is promoted by
//! [`PriorWeights::dominating`] so a search always looks there first. The
//! [`StanceMode`] enum survives as a label — computed *from* the magnitudes,
//! not consulted by them — plus the push-side tilts, which are about this
//! player's own plan rather than about the opponent's.
//!
//! # This is a prior, not a decision
//!
//! Nothing here plays a move. A weight of `dominating` says "look here
//! first", and it is deliberately possible for two different actions to both
//! get it — a player who can *also* win right now has both the deny and the
//! win promoted, and the search decides.

use duels_core::state::Phase;
use duels_core::{cost, engine, scoring, Action, GameState, Player};

use crate::board::{iter_slots, Board};
use crate::context::Context;
use crate::military::{military_read_with, MilModel, MilitaryRead, MilitaryStatus, ShieldSource};
use crate::science::{science_read_with, token_value, SciModel, ScienceRead, ScienceStatus};
use crate::tempo::{grants_extra_turn, holds_theology, ThreatWeights};
use crate::vp::{vp_read_with, VpRead, VpWeights};

/// Points per coin, matching the real `floor(coins / 3)` scoring rule.
const COIN_VP: f64 = 1.0 / 3.0;

/// Which win condition a stance is pushing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Race {
    /// The conflict track.
    Military,
    /// Six distinct scientific symbols.
    Science,
}

/// What a position is about. A label over the magnitudes, and the carrier of
/// the push-side tilts; the denial half of the prior does not consult it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StanceMode {
    /// The opponent has a race at `M == 1` — certain if unopposed — and this
    /// player has a way to make it uncertain.
    DenyCertain,
    /// This player closes next turn and cannot be stopped.
    PushImminentFork,
    /// A race worth leaning into.
    PushLive,
    /// A science race worth forcing denial on, but not worth winning.
    Pressure,
    /// Play for points.
    VpEfficient,
}

/// Named weights for [`action_prior`].
///
/// Every judgement call the prior makes lives here or in [`ThreatWeights`]
/// rather than as a literal in the code, so it can be swept or fitted later by
/// a tournament runner without touching the rules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriorWeights {
    /// Weight every action starts from.
    pub base: f64,
    /// Multiplier applied to a move that closes a race outright, or that
    /// breaks a certain opposing one. Set far above the range the other terms
    /// can reach, so such a move is always at the top of the list.
    pub dominating: f64,
    /// Tilt applied to a race-advancing move in [`StanceMode::PushLive`],
    /// before the tempo and edge adjustments.
    pub push_tilt: f64,
    /// How much harder a trailing player leans into a live race, per point of
    /// negative [`VpRead::structural_edge`].
    pub trail_scale: f64,
    /// How much a leading player's tilt is damped, per point of positive
    /// [`VpRead::structural_edge`].
    pub lead_scale: f64,
    /// Tilt applied to a symbol-gaining move in [`StanceMode::Pressure`].
    pub pressure_tilt: f64,
    /// How far below the best available move a symbol-gaining move may be
    /// before the pressure tilt is withheld entirely, in victory points.
    pub pressure_margin: f64,
    /// Weight on the linear victory-point channel: [`action_vp_value`] plus
    /// [`deny_vp`].
    pub vp: f64,
    /// Tilt for a move that keeps one of this player's races alive.
    pub optionality: f64,
    /// Lower bound on a returned weight, so a prior is never zero or negative
    /// and every legal move keeps some probability.
    pub floor: f64,
}

impl Default for PriorWeights {
    fn default() -> Self {
        Self {
            base: 1.0,
            dominating: 50.0,
            push_tilt: 1.5,
            trail_scale: 0.08,
            lead_scale: 0.08,
            pressure_tilt: 0.35,
            pressure_margin: 3.0,
            vp: 0.35,
            optionality: 0.15,
            floor: 0.05,
        }
    }
}

/// Everything [`action_prior`] needs, computed once per position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stance {
    /// The player this stance is for.
    pub player: Player,
    /// Where the cards publicly are.
    pub board: Board,
    /// The threat-model weights in force.
    pub threat: ThreatWeights,
    /// This player's military race.
    pub military: MilitaryRead,
    /// The opponent's military race — this player's military threat level.
    pub opponent_military: MilitaryRead,
    /// This player's science race.
    pub science: ScienceRead,
    /// The opponent's science race.
    pub opponent_science: ScienceRead,
    /// The victory-point race.
    pub vp: VpRead,
    /// The classification.
    pub mode: StanceMode,
    /// Which race the stance is pushing, if any.
    pub race: Option<Race>,
    /// The multiplier a race-advancing move receives in
    /// [`StanceMode::PushLive`] or [`StanceMode::Pressure`], as `1 + tilt`.
    pub tilt: f64,
    /// How much a point of the opponent's race magnitude is worth to this
    /// player: `clamp(1 + stakes_scale × structural_edge, 0.6, 1.4)`.
    pub stakes: f64,
    /// Slots whose card the opponent needs to close a race, and which this
    /// player can therefore take away.
    pub deny_slots: u32,
    /// Slots whose card would push the conflict pawn far enough back to break
    /// an imminent opposing military close.
    pub counter_slots: u32,
    /// Wonders that would do the same. Bitmask over
    /// [`duels_core::data::WonderId::index`].
    pub counter_wonders: u16,
    /// Play-again wonders whose construction *begins* a certain close that
    /// needs the extra turn to finish. Bitmask over
    /// [`duels_core::data::WonderId::index`].
    ///
    /// A race at `M == 1` is not always closed by one card: the sixth symbol
    /// may sit behind a covering card, or the last shields may be split
    /// between a wonder and a card. In those positions the winning move is the
    /// wonder that buys the second action, and there is no closing *slot* at
    /// all — so the promotion rail would miss it without this.
    pub chain_close_wonders: u16,
    /// Slots that advance the race in [`Stance::race`].
    pub push_slots: u32,
    /// Wonders that advance it. Bitmask over
    /// [`duels_core::data::WonderId::index`].
    pub push_wonders: u16,
    /// Slots whose card keeps one of this player's races alive: an affordable
    /// shield card, or an affordable card carrying a symbol they want.
    pub optionality_slots: u32,
    /// Revealed slots whose card the *opponent* could afford right now. What
    /// makes an uncovered card a real gift rather than a theoretical one.
    pub opponent_affordable_slots: u32,
    /// Whether the *mover* holds Theology, so every wonder they build hands
    /// them another turn — and with it the card that build uncovers. This is
    /// what lets a defender reach behind a covering card.
    pub mover_holds_theology: bool,
    /// Whether the opponent holds Strategy, so every red card they build is
    /// worth one shield more than it prints.
    pub opponent_strategy: bool,
    /// Revealed slots holding a red card the opponent could pay for. What
    /// makes uncovering one a gift rather than a shrug, and the fast path the
    /// denial channel tests before touching a model.
    pub exposed_shield_slots: u32,
    /// The best [`action_vp_value`] among the legal actions. Used to withhold
    /// the pressure tilt when a clearly stronger card is on the table, and
    /// therefore computed only in [`StanceMode::Pressure`] — it is zero in
    /// every other mode, which is the one part of a stance that costs a pass
    /// over the whole legal-action list.
    pub best_action_vp: f64,
    /// Whether this player has a move that closes a race right now.
    pub can_close_now: bool,
    /// The weights in force.
    pub weights: PriorWeights,
}

impl Stance {
    /// Whether the opponent is one move from winning either race.
    #[inline]
    pub fn under_imminent_threat(&self) -> bool {
        self.opponent_military.status == MilitaryStatus::Imminent
            || self.opponent_science.status == ScienceStatus::Imminent
    }

    /// A one-line summary, for logs and the `watch_reads` example.
    pub fn headline(&self) -> String {
        let race = match self.race {
            Some(Race::Military) => "military",
            Some(Race::Science) => "science",
            None => "-",
        };
        format!(
            "{:?} (race: {race}, tilt: {:+.2}, stakes: {:.2}x)",
            self.mode, self.tilt, self.stakes
        )
    }
}

/// Classify the position for `player`, using [`PriorWeights::default`].
pub fn stance(state: &GameState, player: Player) -> Stance {
    stance_with(state, player, PriorWeights::default())
}

/// [`stance`] with explicit prior weights and [`ThreatWeights::default`].
pub fn stance_with(state: &GameState, player: Player, weights: PriorWeights) -> Stance {
    stance_in(
        state,
        player,
        weights,
        &Context::with(state, ThreatWeights::default()),
    )
}

/// [`stance`] against a [`Context`] the caller already built, which is where
/// the threat weights come from.
pub fn stance_in(
    state: &GameState,
    player: Player,
    weights: PriorWeights,
    ctx: &Context,
) -> Stance {
    let opp = player.other();
    let military = military_read_with(state, player, ctx);
    let opponent_military = military_read_with(state, opp, ctx);
    let science = science_read_with(state, player, ctx);
    let opponent_science = science_read_with(state, opp, ctx);
    let vp = vp_read_with(state, player, ctx, &VpWeights::default());
    let board = &ctx.board;
    let tw = &ctx.weights;

    // --- what this player could do about the opponent's race ---------------
    let deny_slots = opponent_military.closing_slots | opponent_science.closing_slots;
    // Pushing the pawn the other way is also a defence: it raises the
    // opponent's `need` above what one of their moves can cover. Computed
    // against the opponent's current best single push, which is an
    // approximation — taking a card changes what they can reach — but a cheap
    // and directionally right one.
    let mut counter_slots = 0u32;
    let mut counter_wonders = 0u16;
    if opponent_military.status == MilitaryStatus::Imminent {
        for src in military.sources() {
            let after = u16::from(opponent_military.need) + u16::from(src.shields());
            if after <= u16::from(opponent_military.best_single) {
                continue;
            }
            match src {
                ShieldSource::Card { slot, .. } => counter_slots |= 1u32 << slot,
                ShieldSource::Wonder { wonder, .. } => counter_wonders |= 1u16 << wonder.index(),
            }
        }
    }

    // A close that needs the extra turn a play-again wonder buys has no
    // closing slot and no closing wonder of its own: the magnitude says the
    // race is won, and the wonder is the first half of winning it.
    let needs_a_chain = (science.magnitude >= 1.0
        && science.closing_slots == 0
        && science.closing_via_token.is_none())
        || (military.magnitude >= 1.0
            && military.closing_slots == 0
            && military.closing_wonders == 0
            && military.need > 0);
    let chain_close_wonders = if needs_a_chain && ctx.tempo(player).chain > 0 {
        let affordable = ctx.prices(player).affordable_wonders;
        if holds_theology(state, player) {
            affordable
        } else {
            affordable & crate::masks::masks().play_again_wonders()
        }
    } else {
        0
    };

    let can_close_now = military.closing_slots != 0
        || military.closing_wonders != 0
        || science.closing_slots != 0
        || science.closing_via_token.is_some()
        || chain_close_wonders != 0;

    // --- which of this player's races is worth pushing ---------------------
    let military_live = military.magnitude >= tw.live_threshold;
    let science_live = science.magnitude >= tw.live_threshold;
    let military_turns = military.turns_to_close.unwrap_or(u8::MAX);
    let opponent_military_turns = opponent_military.turns_to_close.unwrap_or(u8::MAX);
    let science_turns = if science.magnitude >= 1.0 {
        1
    } else {
        science.missing.max(1)
    };

    // Don't lean into a race the opponent can simply answer. On the conflict
    // track that takes *both* halves: they must be able to shove the pawn back
    // at least as hard as this player can push it, **and** be no further from
    // the capital themselves. Supply parity alone is not enough — with the
    // pawn at -3 the two players' identical two-shield cards are worth very
    // different things, and treating that as a stalemate would suppress the
    // push in exactly the positions where it decides the game.
    let military_answerable = opponent_military.best_single >= military.best_single
        && opponent_military_turns <= military_turns;
    // For science, a symbol that is down to a single obtainable copy is a
    // symbol the opponent can simply take.
    let science_answerable = science.fragility > 0;

    let military_pushable = military_live && !military_answerable;
    let science_pushable = science_live && !science_answerable;
    let race = match (military_pushable, science_pushable) {
        (true, true) if science_turns < military_turns => Some(Race::Science),
        (true, _) => Some(Race::Military),
        (false, true) => Some(Race::Science),
        (false, false) => None,
    };

    let (push_slots, push_wonders, race_turns) = match race {
        Some(Race::Military) => (
            military.advancing_slots,
            military.advancing_wonders,
            military_turns,
        ),
        Some(Race::Science) => (
            science.new_symbol_slots | science.closing_slots,
            0,
            science_turns,
        ),
        None => (0, 0, u8::MAX),
    };

    // --- optionality: moves that keep a race alive -------------------------
    let mut optionality_slots = military.advancing_slots | science.new_symbol_slots;
    optionality_slots |= science.pair_setup.completing_slots;
    optionality_slots &= board.accessible;

    // --- the priority order -----------------------------------------------
    let certain_threat = opponent_military.magnitude >= 1.0 || opponent_science.magnitude >= 1.0;
    let can_deny = deny_slots != 0 || counter_slots != 0 || counter_wonders != 0;

    let military_unstoppable = military.status == MilitaryStatus::Imminent && military.undeniable;
    // The science analogue of a fork: two accessible cards either of which
    // completes the sixth symbol, so one opposing turn cannot take both.
    let science_unstoppable = science.status == ScienceStatus::Imminent
        && (science.closing_slots.count_ones() >= 2
            // A chained close happens inside one visit to the table, so the
            // opponent only gets to interfere if they can chain too.
            || (chain_close_wonders != 0 && ctx.tempo(opp).chain == 0));

    let mut tilt = 0.0;
    let mut final_race = race;
    let mode = if certain_threat && can_deny {
        StanceMode::DenyCertain
    } else if military_unstoppable || science_unstoppable {
        final_race = Some(if military_unstoppable {
            Race::Military
        } else {
            Race::Science
        });
        StanceMode::PushImminentFork
    } else if race.is_some() && (push_slots != 0 || push_wonders != 0) {
        // Tempo: closing next turn is worth far more attention than closing in
        // five. Edge: a player who is behind on points has to take the race,
        // a player who is ahead should mostly bank the lead.
        let tempo = 1.0 / f64::from(race_turns.max(1));
        let edge = vp.structural_edge;
        let trailing = 1.0 + (-edge).max(0.0) * weights.trail_scale;
        let leading = 1.0 + edge.max(0.0) * weights.lead_scale;
        tilt = weights.push_tilt * tempo * trailing / leading;
        StanceMode::PushLive
    } else if science.status == ScienceStatus::Pressure && science.new_symbol_slots != 0 {
        final_race = Some(Race::Science);
        tilt = weights.pressure_tilt;
        StanceMode::Pressure
    } else {
        final_race = None;
        StanceMode::VpEfficient
    };

    // The best move on the table, which the pressure tilt needs so it can
    // stand down when a clearly stronger card is available. Computed *after*
    // the mode is known, because it is the one part of a stance that costs a
    // pass over the whole legal-action list — and only one of the five modes
    // reads it.
    let best_action_vp = if mode == StanceMode::Pressure && !state.is_over() {
        let mut buf: Vec<Action> = Vec::with_capacity(32);
        engine::legal_actions_into(state, &mut buf);
        buf.iter()
            .map(|&a| action_vp_value(state, player, a))
            .fold(0.0f64, f64::max)
    } else {
        0.0
    };

    Stance {
        player,
        board: *board,
        threat: *tw,
        military,
        opponent_military,
        science,
        opponent_science,
        vp,
        mode,
        race: final_race,
        tilt,
        stakes: (1.0 + tw.stakes_scale * vp.structural_edge).clamp(tw.stakes_min, tw.stakes_max),
        deny_slots,
        counter_slots,
        counter_wonders,
        chain_close_wonders,
        push_slots,
        push_wonders,
        optionality_slots,
        // What the opponent could pay for: the delta model asks this of every
        // card a move would uncover, and `Prices` already knows.
        opponent_affordable_slots: ctx.prices(opp).affordable_slots,
        exposed_shield_slots: ctx.prices(opp).affordable_slots & board.shield_slots,
        mover_holds_theology: holds_theology(state, player),
        opponent_strategy: crate::masks::masks()
            .strategy_token()
            .is_some_and(|t| state.player(opp).tokens().any(|held| held == t)),
        best_action_vp,
        can_close_now,
        weights,
    }
}

/// The slot an action takes a card from, if it takes one.
#[inline]
pub fn action_slot(action: Action) -> Option<u8> {
    match action {
        Action::Build { slot } | Action::Discard { slot } | Action::BuildWonder { slot, .. } => {
            Some(slot)
        }
        _ => None,
    }
}

/// Whether `action` would close one of this player's races outright.
pub fn action_closes(action: Action, stance: &Stance) -> bool {
    match action {
        Action::Build { slot } => {
            let bit = 1u32 << slot;
            stance.military.closing_slots & bit != 0 || stance.science.closing_slots & bit != 0
        }
        Action::BuildWonder { wonder, .. } => {
            (stance.military.closing_wonders | stance.chain_close_wonders)
                & (1u16 << wonder.index())
                != 0
        }
        Action::ChooseProgressToken { token } | Action::ChooseGreatLibraryToken { token } => {
            stance.science.closing_via_token == Some(token)
        }
        _ => false,
    }
}

/// Whether `action` would take away the card the opponent needs to close, or
/// shove the conflict pawn back out of their reach.
///
/// A diagnostic: [`action_prior`] prices denial through [`delta_m`] instead, so
/// that a move which only *partly* spoils an opposing race is worth a
/// proportionate amount rather than nothing.
pub fn action_denies(action: Action, stance: &Stance) -> bool {
    match action {
        Action::Build { slot } | Action::Discard { slot } => {
            let bit = 1u32 << slot;
            stance.deny_slots & bit != 0 || stance.counter_slots & bit != 0
        }
        Action::BuildWonder { slot, wonder } => {
            let bit = 1u32 << slot;
            stance.deny_slots & bit != 0 || stance.counter_wonders & (1u16 << wonder.index()) != 0
        }
        _ => false,
    }
}

/// Whether `action` advances the race this stance is pushing.
pub fn action_advances(action: Action, stance: &Stance) -> bool {
    match action {
        Action::Build { slot } => stance.push_slots & (1u32 << slot) != 0,
        Action::BuildWonder { wonder, .. } => stance.push_wonders & (1u16 << wonder.index()) != 0,
        Action::ChooseProgressToken { token } | Action::ChooseGreatLibraryToken { token } => {
            stance.science.closing_via_token == Some(token)
        }
        _ => false,
    }
}

/// How much one action moves the *opponent's* race magnitudes.
///
/// Positive is denial: the opponent is less likely to win that race after this
/// move. Negative means the move helps them — uncovering the red card they
/// needed, or clearing the cover off the symbol they could not reach.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DeltaM {
    /// `M_science(opponent) − M_science(opponent | action)`.
    pub science: f64,
    /// `M_military(opponent) − M_military(opponent | action)`.
    pub military: f64,
    /// Whether the action turns a *certain* opposing win into an uncertain
    /// one: what the [`PriorWeights::dominating`] rail fires on.
    pub breaks_certainty: bool,
}

impl DeltaM {
    /// The two races together.
    #[inline]
    pub fn total(&self) -> f64 {
        self.science + self.military
    }
}

/// [`DeltaM`] for one action.
///
/// Applies the action to the *extracted inputs* of the opponent's reads
/// ([`SciModel`], [`MilModel`]) rather than replaying it through
/// [`duels_core::engine::apply`]: a prior that cost a real state transition per
/// legal move would cost more than the search it is meant to guide.
pub fn delta_m(action: Action, s: &Stance) -> DeltaM {
    let board = &s.board;
    let sci_before = s.opponent_science.magnitude;
    let mil_before = s.opponent_military.magnitude;

    // A race that is physically dead, or that needs more symbols than the
    // opponent has decisions left, is already at zero and nothing below can
    // add a route back — so it never has to be recomputed. Likewise, no move
    // can take shields off a military race that already reads zero; only
    // uncovering a red card can move it, and that is checked for separately.
    let sci_frozen = s.opponent_science.dead
        || f64::from(s.opponent_science.missing) > s.opponent_science.model.decisions_left_eff;

    // The models are several hundred bytes each and almost every legal move in
    // almost every position touches neither, so they are copied lazily: a
    // bitmask test says whether this action can move a magnitude at all, and
    // only then is there a model to modify.
    let mut sci: Option<SciModel> = None;
    let mut mil: Option<MilModel> = None;

    match action {
        Action::Build { slot } | Action::Discard { slot } | Action::BuildWonder { slot, .. } => {
            let bit = 1u32 << slot;
            // The card leaves the structure, so the opponent cannot have it.
            if !sci_frozen
                && (s.opponent_science.missing_symbol_slots
                    | s.opponent_science.reachable_missing_slots)
                    & bit
                    != 0
            {
                let symbol = board.slot_card[slot as usize].and_then(|c| c.def().science);
                sci = Some(
                    sci.unwrap_or(s.opponent_science.model)
                        .after_slot_taken(slot, symbol),
                );
            }
            if mil_before > 0.0 {
                if let Some(shields) = s.opponent_military.card_source_shields(slot) {
                    mil = Some(
                        mil.unwrap_or(s.opponent_military.model)
                            .after_card_denied(shields),
                    );
                }
            }

            // ...and whatever it was sitting on comes into view. A play-again
            // wonder buys the turn on which the mover takes the best of those
            // for themselves, so it is a denial rather than a gift.
            let (uncovered, _) = board.newly_open_slots_after(slot, board.occupied);
            let sci_uncovered = if sci_frozen {
                0
            } else {
                uncovered & s.opponent_science.missing_symbol_slots
            };
            let mil_uncovered = uncovered & s.exposed_shield_slots;
            if sci_uncovered | mil_uncovered != 0 {
                let chained = matches!(action, Action::BuildWonder { wonder, .. }
                    if grants_extra_turn(wonder, s.mover_holds_theology));
                let claimed = if chained {
                    most_damaging(
                        sci_uncovered,
                        board,
                        sci.as_ref().unwrap_or(&s.opponent_science.model),
                    )
                } else {
                    None
                };
                for uslot in iter_slots(sci_uncovered | mil_uncovered) {
                    let ubit = 1u32 << uslot;
                    let Some(card) = board.slot_card[uslot as usize] else {
                        continue;
                    };
                    if Some(uslot) == claimed {
                        if let Some(sym) = card.def().science {
                            sci = Some(
                                sci.unwrap_or(s.opponent_science.model)
                                    .after_slot_taken(uslot, Some(sym)),
                            );
                        }
                        continue;
                    }
                    if s.opponent_affordable_slots & ubit == 0 {
                        continue;
                    }
                    if sci_uncovered & ubit != 0 {
                        if let Some(sym) = card.def().science {
                            sci = Some(
                                sci.unwrap_or(s.opponent_science.model)
                                    .with_reachable(sym, uslot),
                            );
                        }
                    }
                    if mil_uncovered & ubit != 0 {
                        mil = Some(mil.unwrap_or(s.opponent_military.model).after_card_exposed(
                            card.def().shields + u8::from(s.opponent_strategy),
                        ));
                    }
                }
            }

            // Shields the mover gains push the pawn back the other way, so the
            // opponent needs more of them.
            if mil_before > 0.0 {
                let gained = match action {
                    Action::Build { slot } => board.slot_card[slot as usize]
                        .map(|c| c.def().shields)
                        .unwrap_or(0),
                    Action::BuildWonder { wonder, .. } => wonder.def().shields,
                    _ => 0,
                };
                if gained > 0 {
                    mil = Some(
                        mil.unwrap_or(s.opponent_military.model)
                            .after_counter_push(gained),
                    );
                }
            }
        }
        Action::ChooseProgressToken { token } | Action::ChooseGreatLibraryToken { token } => {
            if !sci_frozen && crate::masks::masks().law_token() == Some(token) {
                sci = Some(s.opponent_science.model.after_law_taken());
            }
        }
        Action::MausoleumBuild { card } => {
            if !sci_frozen {
                if let Some(sym) = card.def().science {
                    sci = Some(s.opponent_science.model.after_discard_taken(sym));
                }
            }
            if mil_before > 0.0 && card.def().shields > 0 {
                mil = Some(
                    s.opponent_military
                        .model
                        .after_counter_push(card.def().shields),
                );
            }
        }
        // Destroy effects only ever target brown and grey buildings, which
        // carry neither symbols nor shields, so they move no magnitude.
        Action::DestroyOpponentCard { .. }
        | Action::PickWonder { .. }
        | Action::ChooseFirstPlayer { .. } => {}
    }

    let sci_after = sci.map_or(sci_before, |m| m.magnitude().value);
    let mil_after = mil.map_or(mil_before, |m| m.magnitude());

    DeltaM {
        science: sci_before - sci_after,
        military: mil_before - mil_after,
        breaks_certainty: (sci_before >= 1.0 && sci_after < 1.0)
            || (mil_before >= 1.0 && mil_after < 1.0),
    }
}

/// The uncovered slot the mover would most want for themselves on a chained
/// extra turn: a symbol the threat-holder still needs, preferring the scarcest.
fn most_damaging(uncovered: u32, board: &Board, sci: &SciModel) -> Option<u8> {
    let mut best: Option<(u8, f64)> = None;
    for slot in iter_slots(uncovered) {
        let Some(card) = board.slot_card[slot as usize] else {
            continue;
        };
        let Some(sym) = card.def().science else {
            continue;
        };
        if sci.held[sym.index()] > 0 {
            continue;
        }
        let c = sci.copies(sym);
        if best.is_none_or(|(_, b)| c < b) {
            best = Some((slot, c));
        }
    }
    best.map(|(slot, _)| slot)
}

/// The victory-point equivalent of what `action` does to the opponent's races.
///
/// `game_swing_vp × stakes × ΔM`, so it drops into the same linear channel as
/// [`action_vp_value`] and needs no separate weight of its own.
pub fn deny_vp(action: Action, stance: &Stance) -> f64 {
    let d = delta_m(action, stance);
    stance.threat.game_swing_vp * stance.stakes * d.total()
}

/// A normalizable prior weight for one action, given a precomputed
/// [`Stance`].
///
/// Always strictly positive (at least [`PriorWeights::floor`]), so a caller
/// can normalize over any legal-action list without ever zeroing a move out of
/// the search.
pub fn action_prior(state: &GameState, action: Action, stance: &Stance) -> f64 {
    let w = &stance.weights;
    let d = delta_m(action, stance);
    let deny = stance.threat.game_swing_vp * stance.stakes * d.total();
    let vp = action_vp_value(state, stance.player, action);
    let mut weight = w.base + w.vp * (vp + deny);

    // The push side of the prior: this player's own plan, which the denial
    // channel says nothing about.
    match stance.mode {
        StanceMode::PushLive => {
            if action_advances(action, stance) {
                weight *= 1.0 + stance.tilt;
            }
        }
        StanceMode::Pressure => {
            let symbol_move = matches!(action, Action::Build { slot }
                if stance.science.new_symbol_slots & (1u32 << slot) != 0);
            // No tilt at all when a clearly stronger card is on the table.
            if symbol_move && vp + w.pressure_margin >= stance.best_action_vp {
                weight *= 1.0 + stance.tilt;
            }
        }
        StanceMode::VpEfficient => {
            if let Some(slot) = action_slot(action) {
                if stance.optionality_slots & (1u32 << slot) != 0 {
                    weight *= 1.0 + w.optionality;
                }
            }
        }
        StanceMode::DenyCertain | StanceMode::PushImminentFork => {}
    }

    // The one surviving discrete rail: make a search look at the moves that
    // end the game, in either direction. The promoted weight is floored at
    // `base` *before* the multiplier, because a game-ending move is very
    // often an expensive one — the card that reaches the capital can easily be
    // a net loss in coins, and `base + vp × (a big negative)` would otherwise
    // leave it below a free discard however hard it is multiplied.
    if d.breaks_certainty || (stance.can_close_now && action_closes(action, stance)) {
        weight = weight.max(w.base) * w.dominating;
    }

    weight.max(w.floor)
}

/// A cheap "how many victory points is this move worth to me right now"
/// proxy.
///
/// Everything comes out of the real data: printed points and coins from
/// [`duels_core::data`], guild majorities from
/// [`duels_core::scoring::majority_count`], the military band swing from
/// [`duels_core::data::MilitaryTrack::vp_for_distance`], and every coin —
/// gained, spent, or looted — converted at the real `floor(coins / 3)` rate.
///
/// It is deliberately a one-move estimate with no lookahead: the whole point
/// of the surrounding prior layer is that a static value estimate has a low
/// ceiling in this game. What it must *not* be is systematically blind to a
/// whole category of value, which is why the shield and progress-token terms
/// are here — without them a red card would look like nothing but a bill, and
/// the layer would quietly bias away from exactly the moves it exists to
/// highlight.
pub fn action_vp_value(state: &GameState, player: Player, action: Action) -> f64 {
    match action {
        Action::Build { slot } => {
            let Some(card) = state.face_up_card(slot) else {
                return 0.0;
            };
            let def = card.def();
            let mut v = f64::from(def.victory_points);
            if let Some((target, per)) = def.points_by_majority {
                v += f64::from(per) * f64::from(scoring::majority_count(state, target));
            }
            let mut coins = f64::from(def.coins);
            if let Some((target, per)) = def.coins_per_own {
                coins += f64::from(per) * f64::from(state.player(player).count(target));
            }
            if let Some((target, per)) = def.coins_by_majority {
                coins += f64::from(per) * f64::from(scoring::majority_count(state, target));
            }
            coins -= f64::from(cost::card_cost(state, player, card).coins);

            let shields = if def.shields > 0 {
                def.shields + u8::from(has_strategy(state, player))
            } else {
                0
            };
            v += military_vp_swing(state, player, shields);
            v += loot_vp_swing(state, player, shields);
            v += pair_token_value(state, player, card);
            v + coins * COIN_VP
        }
        Action::BuildWonder { wonder, .. } => {
            let def = wonder.def();
            let coins = f64::from(def.coins) + f64::from(def.opponent_loses_coins)
                - f64::from(cost::wonder_cost(state, player, wonder).coins);
            f64::from(def.victory_points)
                + military_vp_swing(state, player, def.shields)
                + loot_vp_swing(state, player, def.shields)
                + coins * COIN_VP
        }
        Action::Discard { .. } => f64::from(cost::discard_reward(state, player)) * COIN_VP,
        Action::ChooseProgressToken { token } | Action::ChooseGreatLibraryToken { token } => {
            token_value(state, player, token)
        }
        Action::MausoleumBuild { card } => {
            let def = card.def();
            f64::from(def.victory_points) + f64::from(def.coins) * COIN_VP
        }
        // Destroying an opponent's card removes their points, which is worth
        // as much to the gap as scoring them.
        Action::DestroyOpponentCard { card } => f64::from(card.def().victory_points),
        Action::PickWonder { wonder } => f64::from(wonder.def().victory_points) * 0.5,
        Action::ChooseFirstPlayer { .. } => 0.0,
    }
}

/// Whether `player` holds the Strategy progress token, which adds a shield to
/// every red card they construct.
#[inline]
fn has_strategy(state: &GameState, player: Player) -> bool {
    crate::masks::masks()
        .strategy_token()
        .is_some_and(|t| state.player(player).tokens().any(|held| held == t))
}

/// The end-of-game military points `shields` would swing: what this player's
/// band gains, plus what the opponent's band loses. Both come from the real
/// scoring table.
fn military_vp_swing(state: &GameState, player: Player, shields: u8) -> f64 {
    if shields == 0 {
        return 0.0;
    }
    let track = duels_core::data::military();
    let cap = i16::from(track.capital_distance);
    let before = i16::from(crate::military::signed_distance(state, player));
    let after = (before + i16::from(shields)).min(cap);
    let vp = |d: i16| -> f64 {
        if d <= 0 {
            0.0
        } else {
            f64::from(track.vp_for_distance(u8::try_from(d).unwrap_or(0)))
        }
    };
    (vp(after) - vp(before)) + (vp(-before) - vp(-after))
}

/// The coins `shields` would loot off the opponent, in victory points.
///
/// Matches the engine's rule: every loot token on this player's side that has
/// not been collected triggers once the pawn is at or past its distance, and
/// the opponent cannot pay more than they hold.
fn loot_vp_swing(state: &GameState, player: Player, shields: u8) -> f64 {
    if shields == 0 {
        return 0.0;
    }
    let track = duels_core::data::military();
    let cap = i16::from(track.capital_distance);
    let after =
        (i16::from(crate::military::signed_distance(state, player)) + i16::from(shields)).min(cap);
    let mut coins = 0u16;
    for (i, &(distance, amount)) in track.loot.iter().enumerate() {
        if state.loot_available(player, i) && after >= i16::from(distance) {
            coins += u16::from(amount);
        }
    }
    f64::from(coins.min(state.player(player.other()).coins())) * COIN_VP
}

/// The progress token this card would win, if it completes an unrewarded pair
/// of scientific symbols and there is still a token on the board to take.
fn pair_token_value(state: &GameState, player: Player, card: duels_core::data::CardId) -> f64 {
    let Some(sym) = card.def().science else {
        return 0.0;
    };
    let me = state.player(player);
    if me.science()[sym.index()] != 1 || me.pairs_awarded().any(|s| s == sym) {
        return 0.0;
    }
    state
        .board_tokens()
        .map(|t| token_value(state, player, t))
        .fold(0.0, f64::max)
}

/// Prior weights for every action in `legal`, in the same order.
///
/// A convenience for callers that want the whole vector at once; the per-action
/// [`action_prior`] is the primitive.
pub fn action_priors(state: &GameState, legal: &[Action], stance: &Stance) -> Vec<f64> {
    legal
        .iter()
        .map(|&a| action_prior(state, a, stance))
        .collect()
}

/// Slots whose card is currently accessible and affordable to `player`, for
/// callers explaining a stance.
pub fn affordable_slots(state: &GameState, player: Player, board: &Board) -> u32 {
    if state.phase() != Phase::Turn {
        return 0;
    }
    let mut out = 0u32;
    for slot in iter_slots(board.accessible) {
        if let Some(card) = board.slot_card[slot as usize] {
            if cost::card_cost(state, player, card).affordable_by(state, player) {
                out |= 1u32 << slot;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::testing::StateBuilder;

    #[test]
    fn every_prior_is_strictly_positive() {
        let st = engine::new_game(11);
        let s = stance(&st, st.current_player());
        for a in engine::legal_actions(&st) {
            let p = action_prior(&st, a, &s);
            assert!(p >= s.weights.floor, "prior {p} for {a:?}");
            assert!(p.is_finite());
        }
    }

    /// A `Stance` is returned by value and held for the life of a search
    /// node, so keep an eye on how big it is.
    #[test]
    fn a_stance_is_copy_and_of_a_reasonable_size() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Stance>();
        let size = std::mem::size_of::<Stance>();
        assert!(size <= 4096, "Stance grew to {size} bytes");
        println!("size_of::<Stance>() = {size}");
        println!("size_of::<Context>() = {}", std::mem::size_of::<Context>());
        println!("size_of::<Board>() = {}", std::mem::size_of::<Board>());
        println!(
            "size_of::<MilitaryRead>() = {}",
            std::mem::size_of::<MilitaryRead>()
        );
        println!(
            "size_of::<ScienceRead>() = {}",
            std::mem::size_of::<ScienceRead>()
        );
    }

    #[test]
    fn a_quiet_position_plays_for_points() {
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(14, "lumber-yard"), (15, "clay-pool")])
            .coins(Player::One, 5)
            .current(Player::One)
            .build();
        let s = stance(&st, Player::One);
        assert_eq!(s.mode, StanceMode::VpEfficient);
    }
}
