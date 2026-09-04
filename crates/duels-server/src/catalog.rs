//! Builds the static `GET /catalog` payload from `duels_core::data`.
//!
//! This is pure translation: every field comes straight from the embedded
//! `data/*.json` via `duels_core::data`'s public accessors, reshaped into
//! the wire DTOs in `crate::protocol` (named resource amounts instead of a
//! positional array, human-readable majority-target labels instead of the
//! internal `CountTarget` enum). No rule is computed here, `duels-core`'s
//! own `MilitaryTrack::vp_for_distance` is used for the military table so it
//! can never drift from what the engine actually pays out.

use duels_core::data::{self, CardId, CountTarget, TokenId, WonderId};

use crate::protocol::{
    AgeStructureLayout, CardCatalogEntry, Catalog, MilitaryCatalog, TokenCatalogEntry,
    WonderCatalogEntry,
};

fn count_target_label(t: CountTarget) -> String {
    match t {
        CountTarget::Cards(kind) => format!("{} cards", card_type_label(kind)),
        CountTarget::RawAndManufactured => "raw material + manufactured good cards".to_string(),
        CountTarget::Wonders => "constructed wonders".to_string(),
        CountTarget::CoinsDiv3 => "coins (rounded down to the nearest 3)".to_string(),
    }
}

/// A human-readable, space-separated name for a card colour, for use inside
/// a sentence (e.g. "5 coins per *raw material* card"). `CardType`'s
/// `Debug` output (`RawMaterial`, `ManufacturedGood`) is fine as an
/// internal identifier but reads as a run-together enum name to a player,
/// which is exactly the kind of unclear wording this catalog is meant to
/// avoid.
fn card_type_label(kind: data::CardType) -> &'static str {
    match kind {
        data::CardType::RawMaterial => "raw material",
        data::CardType::ManufacturedGood => "manufactured good",
        data::CardType::Civilian => "civilian",
        data::CardType::Scientific => "scientific",
        data::CardType::Commercial => "commercial",
        data::CardType::Military => "military",
        data::CardType::Guild => "guild",
    }
}

/// Build the full, static catalog. Cheap enough to call per-request; nothing
/// here depends on any particular game's state.
pub fn build() -> Catalog {
    let cards = CardId::all()
        .map(|id| {
            let d = id.def();
            CardCatalogEntry {
                id,
                name: d.name.to_string(),
                age: d.age,
                kind: d.kind,
                coin_cost: d.coin_cost,
                resource_cost: d.resource_cost.into(),
                chain_from: d.chain_from,
                chain_to: d.chain_to,
                produces: d.produces.into(),
                produces_choice: d.produces_choice.map(Into::into),
                victory_points: d.victory_points,
                science: d.science,
                shields: d.shields,
                coins: d.coins,
                fixed_trade: data::Resource::ALL
                    .into_iter()
                    .filter(|r| d.fixed_trade[r.index()])
                    .collect(),
                coins_per_own: d.coins_per_own.map(|(t, n)| (count_target_label(t), n)),
                coins_by_majority: d.coins_by_majority.map(|(t, n)| (count_target_label(t), n)),
                points_by_majority: d
                    .points_by_majority
                    .map(|(t, n)| (count_target_label(t), n)),
                is_guild: d.is_guild(),
            }
        })
        .collect();

    let wonders = WonderId::all()
        .map(|id| {
            let d = id.def();
            WonderCatalogEntry {
                id,
                name: d.name.to_string(),
                coin_cost: d.coin_cost,
                resource_cost: d.resource_cost.into(),
                victory_points: d.victory_points,
                shields: d.shields,
                coins: d.coins,
                opponent_loses_coins: d.opponent_loses_coins,
                produces_choice: d.produces_choice.map(Into::into),
                play_again: d.play_again,
                destroy: d.destroy,
                build_discarded_free: d.build_discarded_free,
                choose_progress_token: d.choose_progress_token,
            }
        })
        .collect();

    let tokens = TokenId::all()
        .map(|id| {
            let d = id.def();
            TokenCatalogEntry {
                id,
                name: d.name.to_string(),
                coins: d.coins,
                victory_points: d.victory_points,
                vp_per_token: d.vp_per_token,
                science: d.science,
                discount: d.discount.map(Into::into),
                gain_trade_costs: d.gain_trade_costs,
                shield_bonus: d.shield_bonus,
                wonder_play_again: d.wonder_play_again,
                chain_build_coins: d.chain_build_coins,
            }
        })
        .collect();

    let mt = data::military();
    let military = MilitaryCatalog {
        capital_distance: mt.capital_distance,
        loot: mt.loot,
        victory_points_by_distance: (0..mt.capital_distance)
            .map(|d| mt.vp_for_distance(d))
            .collect(),
    };

    let layouts = std::array::from_fn(|i| AgeStructureLayout {
        positions: duels_core::layout::layout((i + 1) as u8).positions,
    });

    Catalog {
        cards,
        wonders,
        tokens,
        military,
        layouts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_covers_every_card_wonder_and_token() {
        let c = build();
        assert_eq!(c.cards.len(), data::NUM_CARDS);
        assert_eq!(c.wonders.len(), data::NUM_WONDERS);
        assert_eq!(c.tokens.len(), data::NUM_TOKENS);
        assert_eq!(c.military.victory_points_by_distance.len(), 9);
        for l in &c.layouts {
            assert_eq!(l.positions.len(), duels_core::layout::SLOTS);
        }
    }
}
