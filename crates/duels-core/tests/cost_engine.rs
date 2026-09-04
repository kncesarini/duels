//! Hand-computed cost scenarios.
//!
//! Every expected number below is worked out in the comment above it from the
//! rules: `2 + opponent's brown/grey production` per missing unit, 1 flat with
//! a trading post, choice-production sources decided in the payer's favour,
//! and the Architecture / Masonry rebate spent where it saves the most.
//!
//! See `docs/rules-spec.md` R-030..R-036.

use duels_core::cost;
use duels_core::data::{CardId, WonderId};
use duels_core::testing::StateBuilder;
use duels_core::{GameState, Player};

fn card(slug: &str) -> CardId {
    CardId::from_slug(slug).unwrap()
}

fn wonder(slug: &str) -> WonderId {
    WonderId::from_slug(slug).unwrap()
}

fn cost_of(state: &GameState, slug: &str) -> u16 {
    cost::card_cost(state, Player::One, card(slug)).coins
}

fn wonder_cost_of(state: &GameState, slug: &str) -> u16 {
    cost::wonder_cost(state, Player::One, wonder(slug)).coins
}

#[test]
fn a_card_you_can_fully_produce_is_free() {
    // Aqueduct needs 3 stone; Shelf Quarry gives 2 and Quarry 1.
    let state = StateBuilder::new()
        .built(Player::One, &["shelf-quarry", "quarry"])
        .build();
    assert_eq!(cost_of(&state, "aqueduct"), 0);
}

#[test]
fn missing_resources_cost_two_plus_the_opponents_production() {
    // Palace: 1 clay, 1 stone, 1 wood, 2 glass.
    // Player One produces nothing.
    // Player Two has Sawmill (2 wood) and Glassworks (1 glass).
    //   wood  : 2 + 2 = 4
    //   clay  : 2 + 0 = 2
    //   stone : 2 + 0 = 2
    //   glass : 2 + 1 = 3, twice
    //   total = 4 + 2 + 2 + 6 = 14
    let state = StateBuilder::new()
        .built(Player::Two, &["sawmill", "glassworks"])
        .build();
    assert_eq!(cost_of(&state, "palace"), 14);
}

#[test]
fn a_choice_source_stands_in_for_the_priciest_unit_of_its_group() {
    let base = StateBuilder::new().built(Player::Two, &["sawmill", "glassworks"]);

    // The Caravansery produces one raw material of choice. It should cover
    // the wood (4) rather than the clay or stone (2): 14 - 4 = 10.
    let state = base.clone().built(Player::One, &["caravansery"]).build();
    assert_eq!(cost_of(&state, "palace"), 10);

    // Adding the Forum (one manufactured good of choice) covers one glass
    // (3): 10 - 3 = 7.
    let state = base
        .clone()
        .built(Player::One, &["caravansery", "forum"])
        .build();
    assert_eq!(cost_of(&state, "palace"), 7);
}

#[test]
fn masonry_rebates_two_resources_of_the_blue_cards_cost() {
    // Continuing the Palace example: with the Caravansery and Forum covering
    // the wood and one glass, the remaining bill is glass 3 + clay 2 + stone
    // 2 = 7. Masonry removes the two priciest remaining units, the glass (3)
    // and one of the twos: 7 - 5 = 2.
    let state = StateBuilder::new()
        .built(Player::Two, &["sawmill", "glassworks"])
        .built(Player::One, &["caravansery", "forum"])
        .tokens(Player::One, &["masonry"])
        .build();
    assert_eq!(cost_of(&state, "palace"), 2);
}

#[test]
fn masonry_does_not_apply_to_other_colours() {
    // Town Hall is blue, Arsenal is red; both cost 5 resources.
    // Town Hall: 3 stone, 2 wood -> 10 at 2 each, minus 2 units = 6.
    // Arsenal  : 3 clay, 2 wood  -> 10, no rebate.
    let state = StateBuilder::new()
        .tokens(Player::One, &["masonry"])
        .build();
    assert_eq!(cost_of(&state, "town-hall"), 6);
    assert_eq!(cost_of(&state, "arsenal"), 10);
}

#[test]
fn architecture_rebates_two_resources_of_a_wonders_cost() {
    // The Pyramids: 3 stone + 1 papyrus, nothing produced on either side, so
    // 4 units at 2 = 8. Architecture removes two units: 4.
    let plain = StateBuilder::new().build();
    assert_eq!(wonder_cost_of(&plain, "the-pyramids"), 8);

    let with_token = StateBuilder::new()
        .tokens(Player::One, &["architecture"])
        .build();
    assert_eq!(wonder_cost_of(&with_token, "the-pyramids"), 4);
}

#[test]
fn architecture_does_not_touch_a_wonders_coin_cost() {
    // No base-game wonder has a coin cost, so assert the analogous rule for
    // cards: Masonry leaves a blue card's coin cost alone. No blue card has
    // one either, so use Pretorium (red, 8 coins) to prove the coin cost is
    // never rebated by a resource discount.
    let state = StateBuilder::new()
        .tokens(Player::One, &["masonry", "architecture"])
        .build();
    assert_eq!(cost_of(&state, "pretorium"), 8);
}

#[test]
fn a_trading_post_fixes_the_price_at_one_however_much_the_opponent_produces() {
    // Aqueduct needs 3 stone. Player Two has Shelf Quarry (2 stone), so the
    // open-market price is 2 + 2 = 4 per unit: 12 coins.
    let expensive = StateBuilder::new()
        .built(Player::Two, &["shelf-quarry"])
        .build();
    assert_eq!(cost_of(&expensive, "aqueduct"), 12);

    // With the Stone Reserve it is 1 per unit: 3 coins.
    let cheap = StateBuilder::new()
        .built(Player::Two, &["shelf-quarry"])
        .built(Player::One, &["stone-reserve"])
        .build();
    assert_eq!(cost_of(&cheap, "aqueduct"), 3);
}

#[test]
fn the_customs_house_fixes_both_manufactured_goods() {
    // Observatory: 1 stone, 2 papyrus. Player Two has Drying Room (1 papyrus)
    // and Shelf Quarry (2 stone).
    //   stone   : 2 + 2 = 4
    //   papyrus : 2 + 1 = 3, twice
    //   total   = 4 + 6 = 10
    let plain = StateBuilder::new()
        .built(Player::Two, &["drying-room", "shelf-quarry"])
        .build();
    assert_eq!(cost_of(&plain, "observatory"), 10);

    // The Customs House drops both papyrus to 1: 4 + 1 + 1 = 6.
    let state = StateBuilder::new()
        .built(Player::Two, &["drying-room", "shelf-quarry"])
        .built(Player::One, &["customs-house"])
        .build();
    assert_eq!(cost_of(&state, "observatory"), 6);
}

#[test]
fn yellow_cards_and_wonders_never_raise_the_opponents_prices() {
    // Player Two's Caravansery and The Great Lighthouse both produce raw
    // materials, but neither counts towards what Player One pays.
    let state = StateBuilder::new()
        .built(Player::Two, &["caravansery", "forum"])
        .wonders_built(Player::Two, &["the-great-lighthouse"])
        .build();
    // Town Hall: 3 stone + 2 wood at the base price of 2 = 10.
    assert_eq!(cost_of(&state, "town-hall"), 10);
    assert_eq!(cost::trade_prices(&state, Player::One), [2, 2, 2, 2, 2]);
}

#[test]
fn a_chain_symbol_ignores_the_cost_entirely() {
    // Pantheon costs 1 clay, 1 wood, 2 papyrus, but chains from the Temple.
    let without = StateBuilder::new().build();
    assert_eq!(cost_of(&without, "pantheon"), 8);

    let with_chain = StateBuilder::new().built(Player::One, &["temple"]).build();
    let c = cost::card_cost(&with_chain, Player::One, card("pantheon"));
    assert_eq!(c.coins, 0);
    assert_eq!(c.trade, 0);
    assert!(c.via_chain);
}

#[test]
fn a_chain_symbol_also_waives_a_coin_cost() {
    // Barracks costs 3 coins but chains from the Garrison.
    let without = StateBuilder::new().build();
    assert_eq!(cost_of(&without, "barracks"), 3);
    let with_chain = StateBuilder::new()
        .built(Player::One, &["garrison"])
        .build();
    assert_eq!(cost_of(&with_chain, "barracks"), 0);
}

#[test]
fn a_wonders_choice_production_helps_pay_for_the_next_one() {
    // The Great Lighthouse produces one raw material of choice once built.
    // Town Hall then needs 3 stone + 2 wood; the wonder covers one unit at
    // the base price of 2, so 10 - 2 = 8.
    let state = StateBuilder::new()
        .wonders_built(Player::One, &["the-great-lighthouse"])
        .build();
    assert_eq!(cost_of(&state, "town-hall"), 8);
}

#[test]
fn the_trade_portion_is_reported_separately_from_the_coin_cost() {
    // Barracks: 3 coins, no resources -> no trade component.
    let state = StateBuilder::new().build();
    let c = cost::card_cost(&state, Player::One, card("barracks"));
    assert_eq!((c.coins, c.trade), (3, 0));

    // Forum: 3 coins + 1 clay -> 3 to the bank, 2 in trade.
    let c = cost::card_cost(&state, Player::One, card("forum"));
    assert_eq!((c.coins, c.trade), (5, 2));

    // Only the trade half is what the Economy token redirects.
    let with_economy = StateBuilder::new()
        .tokens(Player::Two, &["economy"])
        .build();
    let c = cost::card_cost(&with_economy, Player::One, card("forum"));
    assert_eq!((c.coins, c.trade), (5, 2));
}

#[test]
fn accessible_slot_costs_reports_affordability_per_slot() {
    let state = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "obelisk"), (19, "pretorium")])
        .coins(Player::One, 6)
        .build();
    let costs = cost::accessible_slot_costs(&state);
    assert_eq!(costs.len(), 2);
    // Obelisk: 2 stone + 1 glass = 6.
    assert_eq!(costs[0].card.slug(), "obelisk");
    assert_eq!(costs[0].cost.coins, 6);
    assert!(costs[0].affordable);
    // Pretorium: 8 coins flat.
    assert_eq!(costs[1].card.slug(), "pretorium");
    assert_eq!(costs[1].cost.coins, 8);
    assert!(!costs[1].affordable);
}
