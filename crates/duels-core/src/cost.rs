//! The cost engine: what does it actually cost, in coins, to construct a
//! given card or wonder right now?
//!
//! The rules (see `docs/rules-spec.md` R-030..R-036):
//!
//! 1. A card whose chain prerequisite the player already owns is free — no
//!    coins, no resources, cost ignored entirely.
//! 2. Otherwise the player owes the printed coin cost to the bank, plus one
//!    *trade payment* per unit of resource the printed cost demands and the
//!    player's own city does not produce.
//! 3. A trade payment for resource `r` is `2 + (units of r produced by the
//!    opponent's brown and grey cards)`. Yellow cards and wonders never raise
//!    the price. A player owning a trading post for `r` (Stone/Clay/Wood
//!    Reserve, Customs House) pays a flat 1 instead, whatever the opponent
//!    produces.
//! 4. "Produce one of your choice" sources (Forum, Caravansery, Piraeus, The
//!    Great Lighthouse) are decided at payment time and may be decided
//!    differently for each payment.
//! 5. The Architecture and Masonry tokens reduce a wonder's / a blue card's
//!    resource cost by 2 units of the owner's choice. Coin costs are
//!    untouched, and the reduction cannot go below zero.
//!
//! Points 3-5 interact: which resource a choice source should produce depends
//! on the trade prices *and* on where the rebate is best spent. The engine
//! resolves them together, always in the player's favour — a player is never
//! charged more than the cheapest legal way to pay.

use crate::data::{self, CardId, CardType, DiscountTarget, Resource, WonderId, NUM_RESOURCES};
use crate::state::GameState;
use crate::Player;

/// What it costs to construct something right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cost {
    /// Total coins the player must have.
    pub coins: u16,
    /// The part of `coins` that is a trade payment for missing resources.
    /// This is the part that goes to the opponent instead of the bank when
    /// the opponent owns the Economy token.
    pub trade: u16,
    /// True if a chain symbol made this free, in which case `coins` and
    /// `trade` are both zero.
    pub via_chain: bool,
}

impl Cost {
    /// A free build.
    pub const FREE: Cost = Cost {
        coins: 0,
        trade: 0,
        via_chain: false,
    };

    /// Whether `player` can afford this.
    #[inline]
    pub fn affordable_by(&self, state: &GameState, player: Player) -> bool {
        state.player(player).coins() >= self.coins
    }
}

/// The per-unit price `player` pays to trade for each resource.
pub fn trade_prices(state: &GameState, player: Player) -> [u16; NUM_RESOURCES] {
    let me = state.player(player);
    let opp = state.player(player.other());
    let fixed = me.fixed_trade();
    std::array::from_fn(|i| {
        if fixed[i] {
            1
        } else {
            2 + u16::from(opp.trade_relevant_production(Resource::ALL[i]))
        }
    })
}

/// What it costs `player` to build `card` from the structure.
pub fn card_cost(state: &GameState, player: Player, card: CardId) -> Cost {
    let def = card.def();
    let me = state.player(player);

    if let Some(prereq) = def.chain_from {
        if me.has_built(prereq) {
            return Cost {
                coins: 0,
                trade: 0,
                via_chain: true,
            };
        }
    }

    let discount = if def.kind == CardType::Civilian
        && me.has_token_with(|t| t.discount == Some(DiscountTarget::CivilianBuildings))
    {
        2
    } else {
        0
    };

    let trade = resource_payment(state, player, def.resource_cost, discount);
    Cost {
        coins: u16::from(def.coin_cost) + trade,
        trade,
        via_chain: false,
    }
}

/// What it costs `player` to construct `wonder`.
pub fn wonder_cost(state: &GameState, player: Player, wonder: WonderId) -> Cost {
    let def = wonder.def();
    let me = state.player(player);
    let discount = if me.has_token_with(|t| t.discount == Some(DiscountTarget::Wonders)) {
        2
    } else {
        0
    };
    let trade = resource_payment(state, player, def.resource_cost, discount);
    Cost {
        coins: u16::from(def.coin_cost) + trade,
        trade,
        via_chain: false,
    }
}

/// The minimum coins `player` must pay to cover `resource_cost`, given their
/// own production, their choice-production sources, `discount` free units, and
/// the trade prices they face.
fn resource_payment(
    state: &GameState,
    player: Player,
    resource_cost: [u8; NUM_RESOURCES],
    discount: u8,
) -> u16 {
    let me = state.player(player);
    let production = me.production();
    let mut need = [0u8; NUM_RESOURCES];
    let mut any = false;
    for i in 0..NUM_RESOURCES {
        need[i] = resource_cost[i].saturating_sub(production[i]);
        any |= need[i] > 0;
    }
    if !any {
        return 0;
    }
    let (choice_raw, choice_manufactured) = me.choice_sources();
    min_trade_cost(
        need,
        trade_prices(state, player),
        choice_raw,
        choice_manufactured,
        discount,
    )
}

/// Cheapest way to cover `need` units of resources.
///
/// Each still-needed unit can be covered for free by
/// * a raw choice source, if the unit is `wood`/`clay`/`stone`,
/// * a manufactured choice source, if the unit is `glass`/`papyrus`,
/// * a rebate unit (Architecture / Masonry), which covers any resource,
///
/// and otherwise costs `prices[resource]` coins.
///
/// Covering the most expensive units first, preferring the *more constrained*
/// coverer where both apply, is optimal: the coverers form a transversal
/// matroid over the units (raw and manufactured sources cover disjoint
/// subsets, rebate units cover everything), and greedy by decreasing weight
/// is optimal on a matroid. `min_trade_cost_exhaustive` in this module's tests
/// brute-forces every assignment and a property test asserts the two agree.
pub(crate) fn min_trade_cost(
    need: [u8; NUM_RESOURCES],
    prices: [u16; NUM_RESOURCES],
    mut choice_raw: u8,
    mut choice_manufactured: u8,
    mut discount: u8,
) -> u16 {
    // Expand into individual units, most expensive first. A cost never asks
    // for more than 3 of one resource, so this is at most a handful.
    let mut order: [usize; NUM_RESOURCES] = std::array::from_fn(|i| i);
    order.sort_unstable_by(|&a, &b| prices[b].cmp(&prices[a]));

    let mut total = 0u16;
    for &r in &order {
        let resource = Resource::ALL[r];
        for _ in 0..need[r] {
            if resource.is_raw() && choice_raw > 0 {
                choice_raw -= 1;
            } else if !resource.is_raw() && choice_manufactured > 0 {
                choice_manufactured -= 1;
            } else if discount > 0 {
                discount -= 1;
            } else {
                total += prices[r];
            }
        }
    }
    total
}

/// A convenience view of what a player can and cannot afford, for a UI or a
/// heuristic agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotCost {
    /// The slot in the age structure.
    pub slot: u8,
    /// The card there.
    pub card: CardId,
    /// Cost to build it.
    pub cost: Cost,
    /// Whether the acting player can pay.
    pub affordable: bool,
}

/// Costs for every accessible slot, for the player to move.
pub fn accessible_slot_costs(state: &GameState) -> Vec<SlotCost> {
    let player = state.current_player();
    crate::state::iter_slots(state.accessible_slots())
        .filter_map(|slot| {
            let card = state.face_up_card(slot)?;
            let cost = card_cost(state, player, card);
            Some(SlotCost {
                slot,
                card,
                cost,
                affordable: cost.affordable_by(state, player),
            })
        })
        .collect()
}

/// Whether `player`'s trade payments are redirected to their opponent.
pub(crate) fn opponent_has_economy(state: &GameState, player: Player) -> bool {
    state
        .player(player.other())
        .has_token_with(|t| t.gain_trade_costs)
}

/// All resource kinds a choice source of each group can stand in for. Exposed
/// for tests and for UI explanations of a computed cost.
pub fn choice_group_members(group: data::ResourceGroup) -> &'static [Resource] {
    group.members()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Brute-force reference implementation: try every way to point the
    /// choice sources and spend the rebate.
    fn min_trade_cost_exhaustive(
        need: [u8; NUM_RESOURCES],
        prices: [u16; NUM_RESOURCES],
        choice_raw: u8,
        choice_manufactured: u8,
        discount: u8,
    ) -> u16 {
        // Flatten the needed units into a vector of resource indices.
        let mut units = Vec::new();
        for (r, &n) in need.iter().enumerate() {
            for _ in 0..n {
                units.push(r);
            }
        }
        let n = units.len();
        // Coverer kinds: 0 = raw source, 1 = manufactured source, 2 = rebate.
        let coverers: Vec<u8> = std::iter::repeat_n(0u8, choice_raw as usize)
            .chain(std::iter::repeat_n(1u8, choice_manufactured as usize))
            .chain(std::iter::repeat_n(2u8, discount as usize))
            .collect();

        // Assign each coverer to a unit or to nothing; n and coverers.len()
        // are both tiny, so a plain recursive search is fine.
        fn search(
            units: &[usize],
            used: &mut Vec<bool>,
            coverers: &[u8],
            ci: usize,
            prices: &[u16; NUM_RESOURCES],
        ) -> u16 {
            if ci == coverers.len() {
                let mut total = 0;
                for (i, &r) in units.iter().enumerate() {
                    if !used[i] {
                        total += prices[r];
                    }
                }
                return total;
            }
            // Option: leave this coverer unused.
            let mut best = search(units, used, coverers, ci + 1, prices);
            for i in 0..units.len() {
                if used[i] {
                    continue;
                }
                let is_raw = Resource::ALL[units[i]].is_raw();
                let ok = match coverers[ci] {
                    0 => is_raw,
                    1 => !is_raw,
                    _ => true,
                };
                if !ok {
                    continue;
                }
                used[i] = true;
                best = best.min(search(units, used, coverers, ci + 1, prices));
                used[i] = false;
            }
            best
        }

        let mut used = vec![false; n];
        search(&units, &mut used, &coverers, 0, &prices)
    }

    #[test]
    fn no_need_costs_nothing() {
        assert_eq!(min_trade_cost([0; 5], [2; 5], 0, 0, 0), 0);
    }

    #[test]
    fn hand_computed_multi_resource_scenarios() {
        // Palace: 1 clay, 1 stone, 1 wood, 2 glass. Player produces nothing.
        // Opponent produces 2 wood and 1 glass. No trading posts, no rebate,
        // no choice sources.
        //   wood  : 2 + 2 = 4
        //   clay  : 2 + 0 = 2
        //   stone : 2 + 0 = 2
        //   glass : 2 + 1 = 3, twice
        // total = 4 + 2 + 2 + 3 + 3 = 14
        let need = [1, 1, 1, 2, 0];
        let prices = [4, 2, 2, 3, 2];
        assert_eq!(min_trade_cost(need, prices, 0, 0, 0), 14);

        // Same, but the player owns the Caravansery (one raw of choice). It
        // should stand in for the *wood*, the priciest raw unit: 14 - 4 = 10.
        assert_eq!(min_trade_cost(need, prices, 1, 0, 0), 10);

        // Plus the Forum (one manufactured of choice) removes one glass: -3.
        assert_eq!(min_trade_cost(need, prices, 1, 1, 0), 7);

        // Plus Masonry (2 free units of any resource) removes the remaining
        // glass (3) and one of the two-cost raws (2): 7 - 5 = 2.
        assert_eq!(min_trade_cost(need, prices, 1, 1, 2), 2);
    }

    #[test]
    fn rebate_is_not_wasted_on_a_unit_a_choice_source_could_cover() {
        // One wood (price 6) and one glass (price 2) needed; one raw choice
        // source and one rebate unit. The raw source must take the wood and
        // the rebate the glass, for a total of 0 -- a naive implementation
        // that spends the rebate on the most expensive unit first and then
        // finds the raw source has nothing left to do would charge 2.
        let need = [1, 0, 0, 1, 0];
        let prices = [6, 2, 2, 2, 2];
        assert_eq!(min_trade_cost(need, prices, 1, 0, 1), 0);
    }

    #[test]
    fn rebate_covers_a_group_with_no_matching_choice_source() {
        // Two papyrus needed, no manufactured choice source, 2 rebate units.
        let need = [0, 0, 0, 0, 2];
        let prices = [2, 2, 2, 2, 5];
        assert_eq!(min_trade_cost(need, prices, 3, 0, 2), 0);
        // Raw choice sources cannot help with papyrus at all.
        assert_eq!(min_trade_cost(need, prices, 3, 0, 0), 10);
    }

    #[test]
    fn surplus_coverers_never_reduce_cost_below_zero() {
        assert_eq!(min_trade_cost([1, 0, 0, 0, 0], [7, 2, 2, 2, 2], 5, 5, 5), 0);
    }

    #[test]
    fn trading_post_beats_a_heavily_producing_opponent() {
        // Stone Reserve fixes stone at 1 even if the opponent produces 4.
        let need = [0, 0, 3, 0, 0];
        let prices = [2, 2, 1, 2, 2];
        assert_eq!(min_trade_cost(need, prices, 0, 0, 0), 3);
    }

    proptest! {
        /// The fast greedy assignment must always agree with brute force.
        #[test]
        fn greedy_matches_exhaustive(
            need in proptest::array::uniform5(0u8..4),
            prices in proptest::array::uniform5(1u16..8),
            choice_raw in 0u8..3,
            choice_manufactured in 0u8..3,
            discount in 0u8..3,
        ) {
            let fast = min_trade_cost(need, prices, choice_raw, choice_manufactured, discount);
            let slow = min_trade_cost_exhaustive(
                need, prices, choice_raw, choice_manufactured, discount);
            prop_assert_eq!(fast, slow,
                "need={:?} prices={:?} raw={} mfg={} disc={}",
                need, prices, choice_raw, choice_manufactured, discount);
        }
    }
}
