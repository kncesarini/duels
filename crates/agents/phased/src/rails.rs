//! The terminal rails: three yes/no questions that override the weighted sum.
//!
//! # What this fixes
//!
//! [`crate::terms::military_urgency`] was the only thing in this evaluation
//! that claimed to notice an imminent loss, and it does not: it is a smooth
//! quadratic in how far past the second loot token the pawn sits. It is blind
//! to whether a closing card actually exists, blind to whether anybody can
//! afford it, blind to whether there are two of them, and it pays out at pawn
//! positions where nothing at all is about to happen. A gradient is the wrong
//! instrument for a question with a yes/no answer.
//!
//! The fix is adapted from one that already paid for itself in `mcts-uct`'s
//! rollout policy — *always take an available win, always block an available
//! one-move loss*, worth +26 Elo there. A 1-ply evaluator cannot bias a
//! rollout, but it can ask the same question of the *post-action state* it is
//! already scoring, and answer it from the rules rather than from a
//! projection.
//!
//! # The three rails
//!
//! Everything below is asked of `s`, the state a candidate action produces.
//! `q = s.current_player()` is whoever moves next in it.
//!
//! * **Rail A — take the win.** Already in [`crate::evaluate`]: a terminal `s`
//!   scores `±instant_result`. Nothing here duplicates it.
//! * **Rail B — do not hand one over.** If `q` has a closing action available
//!   ([`duels_strategy::closing_sources`]), the candidate that produced `s`
//!   hands `q` the game. Every alternative that removes every such action
//!   scores strictly higher, because this one is pinned at `−imminent`.
//! * **Rail C — an undeniable close is as good as a win.** If `q` has no
//!   closing action but the *other* player does, and no single reply of `q`'s
//!   takes all of them away, the game is already decided: `+imminent` for the
//!   player who holds it.
//! * **Rail C′ — the extra turn.** A banked extra turn makes the holder the
//!   mover in `s`, and then their own closing action needs no defending at
//!   all. This is not a special case: it is the `q` branch of Rail B read from
//!   the other side, which is why [`rail_owner`] is written as one question
//!   about the position rather than three about the mover.
//!
//! Writing it as "which player, if either, does this position belong to"
//! keeps the whole thing **antisymmetric** — `rail_value(s, me)` is exactly
//! `−rail_value(s, me.other())` — which the rest of this zero-sum evaluation
//! depends on and `crate::tests::the_evaluation_is_antisymmetric_between_the_
//! two_players` asserts.
//!
//! # The stand-down rule
//!
//! Like [`crate::menu::menu_term`] and [`crate::terms::chain_gift_exposure`],
//! the card half of the detection reads cards in the structure and therefore
//! has to stand down once a candidate has ended the age: the engine then deals
//! a whole new age out of a deck no [`duels_core::Observation`] can see, and
//! those identities are the throwaway sample's invention. The *wonder* half
//! survives the boundary untouched — a wonder's cost, its shields, its owner's
//! purse and [`duels_core::GameState::wonder_slots_left`] are public whatever
//! age it is — so it is kept rather than thrown away with the rest.
//!
//! # What the denial check does and does not model
//!
//! Rail C's check is integer arithmetic over
//! [`duels_core::engine::legal_actions`] with no state applied: for each
//! single reply, does at least one of my closing sources still close? A reply
//! resolves the threat if it consumes the only closing card, if the shields it
//! gains push `need` past every surviving source, if the coins it takes off me
//! — loot the pawn collects, or a wonder's own "opponent loses coins" — leave
//! me unable to pay, if it empties the structure and ends the age with every
//! accessible card in it, or if it grants the opponent a further action (a
//! play-again wonder, Theology, or a Mausoleum rebuild — the last
//! conservatively, since what it retrieves from the discard pile could itself
//! be a red card).
//!
//! The coin-theft and age-ending clauses were added while chasing what looked
//! like false positives in `duels-arena`'s `examples/rail_audit.rs` (the
//! disagreement turned out to be the audit's own bookkeeping, not the rail's);
//! they are kept because both are real ways to take a close away, an Appian
//! Way build stripping three coins being the obvious one.
//!
//! Two things it deliberately does not model, both of them in the direction of
//! *not* firing:
//!
//! * a reply that builds a **production** card raises the trade prices I face,
//!   which could put a closing card out of my reach without taking it. Pricing
//!   that would mean running the cost engine against a hypothetical city.
//! * a close that does not exist yet — one the opponent's own reply would have
//!   to uncover — is invisible to a rail that reads the post-action state.
//!   `duels-arena`'s `examples/rail_audit.rs` measures exactly this: over 200
//!   self-play games it found six positions where a candidate created an
//!   undeniable close, and **all six** were of that second kind, so Rail C
//!   never fired at all. It is a rare guard that is right when it speaks
//!   rather than a term that earns its keep every game, and the audit reports
//!   both halves rather than only the flattering one.

use duels_core::data;
use duels_core::state::{Phase, MAX_WONDERS_BUILT};
use duels_core::{cost, engine, Action, GameState, Player};
use duels_strategy::board::iter_slots;
use duels_strategy::tempo::{grants_extra_turn, holds_theology};
use duels_strategy::{closing_sources_with, loot_loss_from_push, ClosingSources};

/// Whether the terminal rails are consulted at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RailModel {
    /// No rail ever fires; the weighted sum decides every non-terminal
    /// position. The round-two behaviour, kept so [`crate::Config::v2`]
    /// reproduces it bit for bit.
    Off,
    /// Rails B, C and C′ as described in the module docs.
    #[default]
    On,
}

/// Which player, if either, this position already belongs to.
///
/// `None` means no rail fires and the ordinary weighted sum decides.
pub fn rail_owner(state: &GameState, root_age: u8, model: RailModel) -> Option<Player> {
    if model == RailModel::Off || state.is_over() {
        return None;
    }
    // A pending effect (a token pick, a destroy, a Mausoleum rebuild) is a
    // decision of its own with its own action space; the closing-source read
    // is written for an ordinary turn and stands down rather than guessing.
    if state.phase() != Phase::Turn || state.pending().is_some() {
        return None;
    }
    // The stand-down rule: see the module docs.
    let cards = state.age() == root_age;

    let mover = state.current_player();
    // Rail B, and Rail C′ from the other side: whoever is to move needs no
    // defence at all, so one closing action settles it.
    if closing_sources_with(state, mover, cards).any() {
        return Some(mover);
    }

    // Rail C: the waiting player's close survives every single reply.
    let waiter = mover.other();
    let mine = closing_sources_with(state, waiter, cards);
    if mine.any() && survives_every_reply(state, waiter, &mine) {
        return Some(waiter);
    }
    None
}

/// `±imminent`, signed towards `me`, or `None` when no rail fires.
#[inline]
pub fn rail_value(
    state: &GameState,
    me: Player,
    root_age: u8,
    model: RailModel,
    imminent: f64,
) -> Option<f64> {
    if imminent == 0.0 {
        return None;
    }
    rail_owner(state, root_age, model).map(|owner| if owner == me { imminent } else { -imminent })
}

/// Whether at least one of `holder`'s closing sources survives **every** single
/// reply the player to move can make.
fn survives_every_reply(state: &GameState, holder: Player, mine: &ClosingSources) -> bool {
    first_resolving_reply(state, holder, mine).is_none()
}

/// The first reply this check believes takes `holder`'s close away, or `None`
/// when it believes none does.
///
/// Exposed for `duels-arena`'s `examples/rail_audit.rs`, which compares the
/// answer against the engine: when the audit finds a close the rail failed to
/// call decisive, this says which reply it was worried about, and so whether
/// the miss is one of the deliberate conservatisms or a real fault.
pub fn first_resolving_reply(
    state: &GameState,
    holder: Player,
    mine: &ClosingSources,
) -> Option<Action> {
    engine::legal_actions(state)
        .into_iter()
        .find(|&reply| !survives(state, holder, mine, reply))
}

/// What one reply does, as the handful of numbers the arithmetic below
/// needs.
struct Reply {
    /// The structure slot it consumes, if any.
    slot: Option<u8>,
    /// Shields it gains the replying player.
    shields: u8,
    /// Whether it grants them a further action before the holder moves.
    grants_action: bool,
    /// Whether it consumes one of the game's shared wonder slots.
    builds_wonder: bool,
    /// Coins it takes off the holder — a wonder's "opponent loses coins"
    /// effect, on top of whatever loot the pawn collects. The Appian Way
    /// takes three, which is enough to price a closing card out of reach.
    steals_coins: u16,
}

fn read_reply(state: &GameState, replier: Player, action: Action) -> Reply {
    let me = state.player(replier);
    match action {
        Action::Discard { slot } => Reply {
            slot: Some(slot),
            shields: 0,
            grants_action: false,
            builds_wonder: false,
            steals_coins: 0,
        },
        Action::Build { slot } => {
            let card = state.face_up_card(slot);
            let strategy = duels_strategy::masks()
                .strategy_token()
                .is_some_and(|t| me.tokens().any(|held| held == t));
            let def = card.map(|c| c.def());
            let shields = def.map_or(0, |d| {
                d.shields
                    .saturating_add(u8::from(strategy && d.shields > 0))
            });
            // A second copy of a symbol they already hold completes a pair and
            // hands them a progress token to choose — a further decision, and
            // one that can itself be a Law token. Counted as resolving, which
            // is conservative: it is not a further card take.
            let completes_pair = def.and_then(|d| d.science).is_some_and(|s| {
                me.science()[s.index()] == 1 && !me.pairs_awarded().any(|p| p == s)
            });
            Reply {
                slot: Some(slot),
                shields,
                grants_action: completes_pair,
                builds_wonder: false,
                steals_coins: 0,
            }
        }
        Action::BuildWonder { slot, wonder } => {
            let def = wonder.def();
            let theology = holds_theology(state, replier);
            Reply {
                slot: Some(slot),
                shields: def.shields,
                grants_action: grants_extra_turn(wonder, theology) || def.build_discarded_free,
                builds_wonder: true,
                steals_coins: u16::from(def.opponent_loses_coins),
            }
        }
        // Nothing else can be legal in `Phase::Turn` with no pending effect.
        _ => Reply {
            slot: None,
            shields: 0,
            grants_action: true,
            builds_wonder: false,
            steals_coins: 0,
        },
    }
}

/// Whether at least one closing source survives this one reply.
fn survives(state: &GameState, holder: Player, mine: &ClosingSources, action: Action) -> bool {
    let replier = holder.other();
    let r = read_reply(state, replier, action);
    if r.grants_action {
        // They get to answer twice; one closing source cannot be called
        // undeniable against that.
        return false;
    }

    // A reply that empties the structure ends the age, and every accessible
    // card -- closing or not -- goes back in the box with it. Nothing in the
    // structure can be relied on past that point.
    let cards_after = state
        .occupied_slots()
        .count_ones()
        .saturating_sub(u32::from(r.slot.is_some()));
    if cards_after == 0 {
        return false;
    }

    // Their shields push the pawn back, so every military source now has to
    // cover that much more ground; the coins they take -- loot the pawn
    // collects, or a wonder's own theft -- leave the holder with less to pay
    // with.
    let need = u16::from(mine.need) + u16::from(r.shields);
    let coins = state
        .player(holder)
        .coins()
        .saturating_sub(loot_loss_from_push(state, replier, r.shields))
        .saturating_sub(r.steals_coins);
    let taken = r.slot.map_or(0u32, |s| 1u32 << s);

    for slot in iter_slots(mine.military_slots & !taken) {
        if u16::from(mine.slot_shields[slot as usize]) < need {
            continue;
        }
        if let Some(card) = state.face_up_card(slot) {
            if cost::card_cost(state, holder, card).coins <= coins {
                return true;
            }
        }
    }
    // A science close is untouched by anything the pawn does.
    for slot in iter_slots(mine.science_slots & !taken) {
        if let Some(card) = state.face_up_card(slot) {
            if cost::card_cost(state, holder, card).coins <= coins {
                return true;
            }
        }
    }

    if mine.military_wonders != 0 {
        let built_after = state.wonders_built_total() + u8::from(r.builds_wonder);
        if built_after < MAX_WONDERS_BUILT {
            let mut mask = mine.military_wonders;
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                if u16::from(mine.wonder_shields[i]) < need {
                    continue;
                }
                let wonder = data::WonderId::from_index(i);
                if cost::wonder_cost(state, holder, wonder).coins <= coins {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::testing::StateBuilder;

    /// Rail B: the opponent moves next and can close, so the position belongs
    /// to them however pretty the rest of the evaluation looks.
    #[test]
    fn a_position_where_the_mover_can_close_belongs_to_the_mover() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(-7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::Two)
            .build();
        assert_eq!(rail_owner(&st, 3, RailModel::On), Some(Player::Two));
        let w = 500.0;
        assert_eq!(
            rail_value(&st, Player::One, 3, RailModel::On, w),
            Some(-500.0)
        );
        assert_eq!(
            rail_value(&st, Player::Two, 3, RailModel::On, w),
            Some(500.0)
        );
    }

    /// Rail C: one closing card, and an opponent who can take it away, is not
    /// undeniable.
    #[test]
    fn a_single_closing_card_the_opponent_can_take_is_deniable() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::Two)
            .build();
        // Player One holds the close, Player Two moves and can simply discard
        // slot 18.
        assert!(duels_strategy::closing_sources(&st, Player::One).any());
        assert_eq!(rail_owner(&st, 3, RailModel::On), None);
    }

    /// Two closing cards against an opponent who cannot pay for either: one
    /// reply cannot remove both, so it is decided.
    #[test]
    fn a_two_card_fork_against_a_broke_opponent_is_undeniable() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "fortifications")])
            .conflict(7)
            .coins(Player::One, 40)
            .coins(Player::Two, 7)
            .current(Player::Two)
            .build();
        let mine = duels_strategy::closing_sources(&st, Player::One);
        assert_eq!(mine.military_slots.count_ones(), 2);
        assert_eq!(rail_owner(&st, 3, RailModel::On), Some(Player::One));

        // ...and once the opponent can pay for one of them, building it pushes
        // the pawn back and the other no longer reaches.
        let rich = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "fortifications")])
            .conflict(7)
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::Two)
            .build();
        assert_eq!(rail_owner(&rich, 3, RailModel::On), None);
    }

    /// The stand-down rule: with the root in a different age, no card in the
    /// structure may be read, and a card-only close disappears.
    #[test]
    fn the_card_half_stands_down_across_an_age_boundary() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(-7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::Two)
            .build();
        assert_eq!(rail_owner(&st, 3, RailModel::On), Some(Player::Two));
        assert_eq!(rail_owner(&st, 2, RailModel::On), None);
    }

    #[test]
    fn switching_the_rails_off_silences_every_one_of_them() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(-7)
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::Two)
            .build();
        assert_eq!(rail_owner(&st, 3, RailModel::Off), None);
        assert_eq!(rail_value(&st, Player::One, 3, RailModel::Off, 500.0), None);
    }
}
