//! The policy layer: what is this position *about*, and which moves deserve
//! search attention?
//!
//! [`stance`] reads both races for both players once and classifies the
//! position into one of five [`StanceMode`]s, following a strict priority
//! order. [`action_prior`] then prices a single action against that
//! classification, returning a normalizable weight — the caller normalizes
//! over its own legal-action list.
//!
//! # The priority order
//!
//! 1. **Deny-Imminent** — the opponent has a move that wins outright next
//!    turn, and this player can take the card it needs away (or, for a
//!    military close, shove the pawn back far enough that it no longer
//!    reaches). Nothing else matters.
//! 2. **Push-Imminent-with-fork** — this player wins outright next turn and
//!    the close cannot be prevented, because there are two independent
//!    closing moves or because the closing move is a wonder, which needs no
//!    particular card. Take it.
//! 3. **Push-Live** — a race this player can realistically close, tilted by
//!    `1 / turns_to_close` and by [`VpRead::structural_edge`]: a trailing
//!    player leans in hard, a leading player barely leans at all. Suppressed
//!    when the opponent's own supply could simply answer the push, because
//!    tilting into a race the opponent trivially reverses is worse than
//!    playing for points.
//! 4. **Pressure** (science only) — the race is not really winnable against a
//!    denying opponent, but forcing the denial has real value. A small tilt
//!    toward a symbol, and none at all when a clearly stronger card is on the
//!    table.
//! 5. **VP-efficient** — otherwise play for points, with an *optionality*
//!    tilt: a mild preference for moves that keep a race alive, and a mild
//!    aversion to moves that hand the opponent a race they do not currently
//!    have.
//!
//! # This is a prior, not a decision
//!
//! Nothing here plays a move. A weight of `dominating` says "look here first",
//! and it is deliberately possible for two different actions to both get it —
//! in Deny-Imminent mode, a player who can *also* win right now has both the
//! deny and the win promoted, and the search decides.

use duels_core::state::Phase;
use duels_core::{cost, engine, scoring, Action, GameState, Player};

use crate::board::{iter_slots, Board};
use crate::military::{military_read_with, MilitaryRead, MilitaryStatus};
use crate::science::{science_read_with, token_value, ScienceRead, ScienceStatus};
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

/// What a position is about, per the priority order in the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StanceMode {
    /// The opponent closes next turn and this player can stop it.
    DenyImminent,
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
/// Every judgement call the prior makes lives here rather than as a literal in
/// the code, so it can be swept or fitted later by a tournament runner without
/// touching the rules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriorWeights {
    /// Weight every action starts from.
    pub base: f64,
    /// Multiplier applied to a move that closes or denies a race outright.
    /// Set far above the range the other terms can reach, so a closing move is
    /// always at the top of the list.
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
    /// Weight on [`action_vp_value`] in [`StanceMode::VpEfficient`].
    pub vp: f64,
    /// Tilt for a move that keeps one of this player's races alive.
    pub optionality: f64,
    /// Aversion to a move that opens up shields for an opponent who is close
    /// to military supremacy, at full exposure.
    pub exposure: f64,
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
            exposure: 0.4,
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
    /// Slots whose card the opponent needs to close a race, and which this
    /// player can therefore take away.
    pub deny_slots: u32,
    /// Slots whose card would push the conflict pawn far enough back to break
    /// an imminent opposing military close.
    pub counter_slots: u32,
    /// Wonders that would do the same. Bitmask over
    /// [`duels_core::data::WonderId::index`].
    pub counter_wonders: u16,
    /// Slots that advance the race in [`Stance::race`].
    pub push_slots: u32,
    /// Wonders that advance it. Bitmask over
    /// [`duels_core::data::WonderId::index`].
    pub push_wonders: u16,
    /// Slots whose card keeps one of this player's races alive: an affordable
    /// shield card, or an affordable card carrying a symbol they want.
    pub optionality_slots: u32,
    /// The best [`action_vp_value`] among the legal actions. Used to withhold
    /// the pressure tilt when a clearly stronger card is on the table, and
    /// therefore computed only in [`StanceMode::Pressure`] — it is zero in
    /// every other mode, which is the one part of a stance that costs a pass
    /// over the whole legal-action list.
    pub best_action_vp: f64,
    /// Whether this player has a move that closes a race right now.
    pub can_close_now: bool,
    /// Expected shields behind one face-down slot of the current age.
    pub expected_shields_per_hidden: f64,
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
        format!("{:?} (race: {race}, tilt: {:+.2})", self.mode, self.tilt)
    }
}

/// Classify the position for `player`, using [`PriorWeights::default`].
pub fn stance(state: &GameState, player: Player) -> Stance {
    stance_with(state, player, PriorWeights::default())
}

/// [`stance`] with explicit prior weights.
pub fn stance_with(state: &GameState, player: Player, weights: PriorWeights) -> Stance {
    let opp = player.other();
    let board = Board::of(state);
    let military = military_read_with(state, player, &board);
    let opponent_military = military_read_with(state, opp, &board);
    let science = science_read_with(state, player, &board);
    let opponent_science = science_read_with(state, opp, &board);
    let vp = vp_read_with(state, player, &board, &VpWeights::default());

    let hidden = board.hidden_slot_count();
    let expected_shields_per_hidden = if hidden == 0 {
        0.0
    } else {
        board.expected_hidden(|c| f64::from(c.def().shields)) / f64::from(hidden)
    };

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
                crate::military::ShieldSource::Card { slot, .. } => {
                    counter_slots |= 1u32 << slot;
                }
                crate::military::ShieldSource::Wonder { wonder, .. } => {
                    counter_wonders |= 1u16 << wonder.index();
                }
            }
        }
    }

    let can_close_now = military.closing_slots != 0
        || military.closing_wonders != 0
        || science.closing_slots != 0
        || science.closing_via_token.is_some();

    // --- which of this player's races is worth pushing ---------------------
    let military_live = matches!(
        military.status,
        MilitaryStatus::Imminent | MilitaryStatus::Live
    );
    let science_live = matches!(
        science.status,
        ScienceStatus::Imminent | ScienceStatus::Live
    );
    let military_turns = military.turns_to_close.unwrap_or(u8::MAX);
    let opponent_military_turns = opponent_military.turns_to_close.unwrap_or(u8::MAX);
    let science_turns = if science.status == ScienceStatus::Imminent {
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
    let threatened = opponent_military.status == MilitaryStatus::Imminent
        || opponent_science.status == ScienceStatus::Imminent;
    let can_deny = deny_slots != 0 || counter_slots != 0 || counter_wonders != 0;

    let military_unstoppable = military.status == MilitaryStatus::Imminent && military.undeniable;
    // The science analogue of a fork: two accessible cards either of which
    // completes the sixth symbol, so one opposing turn cannot take both.
    let science_unstoppable =
        science.status == ScienceStatus::Imminent && science.closing_slots.count_ones() >= 2;

    let mut tilt = 0.0;
    let mut final_race = race;
    let mode = if threatened && can_deny {
        StanceMode::DenyImminent
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
        board,
        military,
        opponent_military,
        science,
        opponent_science,
        vp,
        mode,
        race: final_race,
        tilt,
        deny_slots,
        counter_slots,
        counter_wonders,
        push_slots,
        push_wonders,
        optionality_slots,
        best_action_vp,
        can_close_now,
        expected_shields_per_hidden,
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
            stance.military.closing_wonders & (1u16 << wonder.index()) != 0
        }
        Action::ChooseProgressToken { token } | Action::ChooseGreatLibraryToken { token } => {
            stance.science.closing_via_token == Some(token)
        }
        _ => false,
    }
}

/// Whether `action` would take away the card the opponent needs to close, or
/// shove the conflict pawn back out of their reach.
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

/// A normalizable prior weight for one action, given a precomputed
/// [`Stance`].
///
/// Always strictly positive (at least [`PriorWeights::floor`]), so a caller
/// can normalize over any legal-action list without ever zeroing a move out of
/// the search.
pub fn action_prior(state: &GameState, action: Action, stance: &Stance) -> f64 {
    let w = &stance.weights;
    let mut weight = w.base;

    match stance.mode {
        StanceMode::DenyImminent => {
            if action_denies(action, stance) {
                weight *= w.dominating;
            }
            // A player who can win right now should not be steered into
            // defending instead; both get promoted and the search picks.
            if stance.can_close_now && action_closes(action, stance) {
                weight *= w.dominating;
            }
        }
        StanceMode::PushImminentFork => {
            if action_closes(action, stance) {
                weight *= w.dominating;
            }
        }
        StanceMode::PushLive => {
            weight += w.vp * action_vp_value(state, stance.player, action);
            if action_advances(action, stance) {
                weight *= 1.0 + stance.tilt;
            }
        }
        StanceMode::Pressure => {
            let vp = action_vp_value(state, stance.player, action);
            let symbol_move = matches!(action, Action::Build { slot }
                if stance.science.new_symbol_slots & (1u32 << slot) != 0);
            weight += w.vp * vp;
            // No tilt at all when a clearly stronger card is on the table.
            if symbol_move && vp + w.pressure_margin >= stance.best_action_vp {
                weight *= 1.0 + stance.tilt;
            }
        }
        StanceMode::VpEfficient => {
            weight += w.vp * action_vp_value(state, stance.player, action);
            if let Some(slot) = action_slot(action) {
                if stance.optionality_slots & (1u32 << slot) != 0 {
                    weight *= 1.0 + w.optionality;
                }
            }
            weight *= 1.0 - w.exposure * exposure_risk(action, stance);
        }
    }

    weight.max(w.floor)
}

/// How much this action would open up the opponent's military race, in
/// `0.0..=1.0`.
///
/// Taking a card uncovers what it was sitting on. Some of that is face up
/// already, so its shields are known exactly; the rest is turned over, and is
/// priced at the expected shields behind one face-down slot of this age. The
/// result is scaled by how few shields the opponent still needs, so exposing a
/// three-shield card matters enormously at `need == 3` and hardly at all at
/// `need == 9`.
pub fn exposure_risk(action: Action, stance: &Stance) -> f64 {
    let Some(slot) = action_slot(action) else {
        return 0.0;
    };
    let (known, face_down) = stance.board.newly_open_after(slot);
    let exposed = f64::from(crate::masks::shields_in(known))
        + f64::from(face_down) * stance.expected_shields_per_hidden;
    if exposed <= 0.0 {
        return 0.0;
    }
    let need = f64::from(stance.opponent_military.need.max(1));
    (exposed / need).clamp(0.0, 1.0)
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
        assert!(size <= 1024, "Stance grew to {size} bytes");
        println!("size_of::<Stance>() = {size}");
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
