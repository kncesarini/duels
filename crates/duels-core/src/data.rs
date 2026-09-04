//! Static game data: cards, wonders, progress tokens, and the military track.
//!
//! The repo-root `data/` directory holds the factual base-game data
//! (`cards.json`, `wonders.json`, `tokens.json`, `military.json`) as the
//! source of truth, documented in `data/README.md`. Those files are embedded
//! into the binary with [`include_str!`] and parsed once, on first access,
//! into dense tables indexed by small integer id ([`CardId`], [`WonderId`],
//! [`TokenId`]).
//!
//! The JSON shape is a loose, human-editable "effect vocabulary"; this module
//! *normalises* it into flat, fixed-size structs so the engine never has to
//! walk a list of tagged effect objects on a hot path. Every effect type that
//! appears in `data/*.json` is handled here, and an unknown effect type is a
//! hard parse error (see [`load`]) rather than being silently ignored — that
//! way adding an effect to the data without teaching the engine about it fails
//! loudly instead of quietly changing the rules.
//!
//! # Ids
//!
//! Ids are opaque newtypes over `u8`. They serialise as their stable JSON slug
//! (`"lumber-yard"`), so replays and wire messages stay readable, but in memory
//! they are one byte, which is what lets [`crate::GameState`] be a small,
//! `Copy`-cheap value (see `docs/rules-spec.md`, R-001).

use std::fmt;
use std::sync::OnceLock;

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer};

/// Number of age cards in the base game: 23 Age I + 23 Age II + 20 Age III
/// non-guild + 7 guilds.
pub const NUM_CARDS: usize = 73;
/// Number of wonders in the base game.
pub const NUM_WONDERS: usize = 12;
/// Number of progress tokens in the base game.
pub const NUM_TOKENS: usize = 10;
/// Number of resource kinds (`wood`, `clay`, `stone`, `glass`, `papyrus`).
pub const NUM_RESOURCES: usize = 5;
/// Number of distinct scientific symbols, including the Law token's symbol.
pub const NUM_SCIENCE: usize = 7;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// A tradeable resource. The first three are raw materials (brown), the last
/// two are manufactured goods (grey).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    /// Wood (raw material).
    Wood,
    /// Clay (raw material).
    Clay,
    /// Stone (raw material).
    Stone,
    /// Glass (manufactured good).
    Glass,
    /// Papyrus (manufactured good).
    Papyrus,
}

impl Resource {
    /// All five resources, in index order.
    pub const ALL: [Resource; NUM_RESOURCES] = [
        Resource::Wood,
        Resource::Clay,
        Resource::Stone,
        Resource::Glass,
        Resource::Papyrus,
    ];

    /// Index into a `[_; NUM_RESOURCES]` array.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Whether this resource is a raw material (brown) rather than a
    /// manufactured good (grey).
    #[inline]
    pub const fn is_raw(self) -> bool {
        matches!(self, Resource::Wood | Resource::Clay | Resource::Stone)
    }
}

/// The two groups a "produce one resource of your choice" effect can offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceGroup {
    /// `wood` / `clay` / `stone`.
    RawMaterial,
    /// `glass` / `papyrus`.
    ManufacturedGood,
}

impl ResourceGroup {
    /// The resources this group can be spent as.
    #[inline]
    pub fn members(self) -> &'static [Resource] {
        match self {
            ResourceGroup::RawMaterial => &[Resource::Wood, Resource::Clay, Resource::Stone],
            ResourceGroup::ManufacturedGood => &[Resource::Glass, Resource::Papyrus],
        }
    }
}

/// The seven card colours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum CardType {
    /// Brown.
    RawMaterial,
    /// Grey.
    ManufacturedGood,
    /// Blue.
    Civilian,
    /// Green.
    Scientific,
    /// Yellow.
    Commercial,
    /// Red.
    Military,
    /// Purple.
    Guild,
}

impl CardType {
    /// All seven card types, in index order.
    pub const ALL: [CardType; 7] = [
        CardType::RawMaterial,
        CardType::ManufacturedGood,
        CardType::Civilian,
        CardType::Scientific,
        CardType::Commercial,
        CardType::Military,
        CardType::Guild,
    ];

    /// Index into a `[_; 7]` array.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// A scientific symbol. Six appear on green cards (twice each); `Balance` is
/// the seventh, granted only by the Law progress token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum Science {
    /// Mortar and pestle.
    Mortar,
    /// Pendulum.
    Pendulum,
    /// Inkwell and quill.
    Inkwell,
    /// Wheel.
    Wheel,
    /// Sundial.
    Sundial,
    /// Gyroscope / armillary sphere.
    Gyroscope,
    /// Granted by the Law progress token only; no card carries it, so it can
    /// never form a pair.
    Balance,
}

impl Science {
    /// Index into a `[_; NUM_SCIENCE]` array.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// What a "per X" effect counts. Used both for a player's own holdings
/// (yellow Age III cards) and for the higher of the two players' holdings
/// (guild cards).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CountTarget {
    /// Cards of one colour.
    Cards(CardType),
    /// Brown + grey cards together (Shipowners Guild).
    RawAndManufactured,
    /// Wonders constructed.
    Wonders,
    /// `floor(coins / 3)` (Moneylenders Guild).
    CoinsDiv3,
}

/// What a progress token's cost rebate applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscountTarget {
    /// Architecture.
    Wonders,
    /// Masonry.
    CivilianBuildings,
}

// ---------------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------------

macro_rules! id_newtype {
    ($name:ident, $count:expr, $table:ident, $what:literal) => {
        #[doc = concat!("Opaque id for a ", $what, ". Serialises as its stable JSON slug.")]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u8);

        impl $name {
            /// Wrap a raw index. Panics if out of range; prefer
            #[doc = concat!("[`", stringify!($name), "::from_slug`] for untrusted input.")]
            #[inline]
            pub fn from_index(i: usize) -> Self {
                assert!(
                    i < $count,
                    concat!(stringify!($name), " index out of range")
                );
                Self(i as u8)
            }

            /// The raw index, usable to index the static tables.
            #[inline]
            pub const fn index(self) -> usize {
                self.0 as usize
            }

            /// The stable JSON slug for this id.
            #[inline]
            pub fn slug(self) -> &'static str {
                $table()[self.index()].id
            }

            /// Look up an id by its stable JSON slug.
            pub fn from_slug(slug: &str) -> Option<Self> {
                $table()
                    .iter()
                    .position(|e| e.id == slug)
                    .map(|i| Self(i as u8))
            }

            /// Every id of this kind, in index order.
            pub fn all() -> impl Iterator<Item = Self> {
                (0..$count).map(|i| Self(i as u8))
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.slug())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.slug())
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(self.slug())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                struct V;
                impl Visitor<'_> for V {
                    type Value = $name;
                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        write!(f, concat!("a ", $what, " slug"))
                    }
                    fn visit_str<E: de::Error>(self, v: &str) -> Result<$name, E> {
                        $name::from_slug(v).ok_or_else(|| {
                            E::custom(format!(concat!("unknown ", $what, ": {}"), v))
                        })
                    }
                }
                d.deserialize_str(V)
            }
        }

        // Wire format is the slug string (see `Serialize`/`Deserialize`
        // above), not the `u8` this type is backed by, so `TS` cannot be
        // derived: it would describe the in-memory repr, not the JSON shape.
        // This mirrors how ts-rs itself hand-implements `TS` for opaque
        // string-like types such as `uuid::Uuid` (`impl_primitives!`).
        #[cfg(feature = "ts")]
        impl ts_rs::TS for $name {
            type WithoutGenerics = Self;
            type OptionInnerType = Self;

            fn name(_: &ts_rs::Config) -> String {
                "string".to_owned()
            }

            fn inline(cfg: &ts_rs::Config) -> String {
                <Self as ts_rs::TS>::name(cfg)
            }
        }
    };
}

id_newtype!(CardId, NUM_CARDS, cards, "age card");
id_newtype!(WonderId, NUM_WONDERS, wonders, "wonder");
id_newtype!(TokenId, NUM_TOKENS, tokens, "progress token");

impl CardId {
    /// The static definition of this card.
    #[inline]
    pub fn def(self) -> &'static Card {
        &cards()[self.index()]
    }
}

impl WonderId {
    /// The static definition of this wonder.
    #[inline]
    pub fn def(self) -> &'static Wonder {
        &wonders()[self.index()]
    }
}

impl TokenId {
    /// The static definition of this progress token.
    #[inline]
    pub fn def(self) -> &'static ProgressToken {
        &tokens()[self.index()]
    }
}

// ---------------------------------------------------------------------------
// Normalised definitions
// ---------------------------------------------------------------------------

/// A fully normalised age card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Card {
    /// Stable slug, matching `data/cards.json`.
    pub id: &'static str,
    /// Printed name.
    pub name: &'static str,
    /// 1, 2, or 3. Guild cards are age 3.
    pub age: u8,
    /// Card colour.
    pub kind: CardType,
    /// Coins that must be paid to the bank to construct this card.
    pub coin_cost: u8,
    /// Resources that must be produced or bought to construct this card.
    pub resource_cost: [u8; NUM_RESOURCES],
    /// If the player already owns this card, this one is built for free.
    pub chain_from: Option<CardId>,
    /// The card this one unlocks a free build for.
    pub chain_to: Option<CardId>,
    /// Resources this card produces every time a cost is paid.
    pub produces: [u8; NUM_RESOURCES],
    /// A "produce one of these, your choice, each time you pay" source.
    pub produces_choice: Option<ResourceGroup>,
    /// Victory points printed on the card, scored at the end of the game.
    pub victory_points: u8,
    /// Scientific symbol, if any.
    pub science: Option<Science>,
    /// Shields, applied once when the card is constructed.
    pub shields: u8,
    /// Coins granted once, when the card is constructed.
    pub coins: u8,
    /// Resources whose trade cost this card fixes at 1 coin per unit.
    pub fixed_trade: [bool; NUM_RESOURCES],
    /// Coins granted once on construction, per unit of the owner's own
    /// holdings (yellow Age III cards).
    pub coins_per_own: Option<(CountTarget, u8)>,
    /// Coins granted once on construction, per unit of *whichever player has
    /// more* (guild cards). Uses the count at the moment the guild is built.
    pub coins_by_majority: Option<(CountTarget, u8)>,
    /// Victory points per unit of whichever player has more, computed only at
    /// final scoring (guild cards). May diverge from `coins_by_majority`.
    pub points_by_majority: Option<(CountTarget, u8)>,
}

impl Card {
    /// Whether this card is a guild (purple) card.
    #[inline]
    pub fn is_guild(&self) -> bool {
        self.kind == CardType::Guild
    }
}

/// A fully normalised wonder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wonder {
    /// Stable slug, matching `data/wonders.json`.
    pub id: &'static str,
    /// Printed name.
    pub name: &'static str,
    /// Coins that must be paid to the bank to construct this wonder.
    pub coin_cost: u8,
    /// Resources that must be produced or bought.
    pub resource_cost: [u8; NUM_RESOURCES],
    /// Victory points, scored at the end of the game.
    pub victory_points: u8,
    /// Shields, applied once when the wonder is constructed.
    pub shields: u8,
    /// Coins granted once, when the wonder is constructed.
    pub coins: u8,
    /// Coins the opponent loses once, when this wonder is constructed.
    pub opponent_loses_coins: u8,
    /// A "produce one of these, your choice" source.
    pub produces_choice: Option<ResourceGroup>,
    /// Grants an extra turn when constructed.
    pub play_again: bool,
    /// Lets the owner discard one opponent building of this colour.
    pub destroy: Option<CardType>,
    /// The Mausoleum: build one card from the discard pile for free.
    pub build_discarded_free: bool,
    /// The Great Library: draw 3 of the progress tokens set aside during
    /// setup and keep one.
    pub choose_progress_token: bool,
}

/// A fully normalised progress token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressToken {
    /// Stable slug, matching `data/tokens.json`.
    pub id: &'static str,
    /// Printed name.
    pub name: &'static str,
    /// Coins granted once, when the token is taken.
    pub coins: u8,
    /// Flat victory points.
    pub victory_points: u8,
    /// Victory points per progress token the owner holds at game end
    /// (Mathematics), including this one.
    pub vp_per_token: u8,
    /// A scientific symbol this token itself provides (Law).
    pub science: Option<Science>,
    /// Reduces the resource cost of future builds by 2, owner's choice of
    /// which resources (Architecture, Masonry).
    pub discount: Option<DiscountTarget>,
    /// Economy: the opponent's trade payments go to this token's owner.
    pub gain_trade_costs: bool,
    /// Strategy: every red building the owner constructs grants +1 shield.
    pub shield_bonus: bool,
    /// Theology: every wonder the owner constructs grants an extra turn.
    pub wonder_play_again: bool,
    /// Urbanism: coins gained each time the owner builds via a chain symbol.
    pub chain_build_coins: u8,
}

/// The military track and its scoring, from `data/military.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MilitaryTrack {
    /// Distance from centre at which a player wins outright.
    pub capital_distance: u8,
    /// Distances from centre at which a loot token sits, in ascending order,
    /// paired with the coins the *losing* player forfeits.
    pub loot: [(u8, u8); 2],
    /// `(inclusive max distance from centre, victory points)`, ascending.
    pub victory_points: [(u8, u8); 4],
}

impl MilitaryTrack {
    /// Victory points awarded to the player the conflict pawn favours, given
    /// the pawn's distance from the centre.
    pub fn vp_for_distance(&self, distance: u8) -> u8 {
        for &(max, vp) in &self.victory_points {
            if distance <= max {
                return vp;
            }
        }
        // Distance 9 is an instant win and never scored this way, but be
        // total anyway.
        self.victory_points[self.victory_points.len() - 1].1
    }
}

// ---------------------------------------------------------------------------
// Static access
// ---------------------------------------------------------------------------

const CARDS_JSON: &str = include_str!("../../../data/cards.json");
const WONDERS_JSON: &str = include_str!("../../../data/wonders.json");
const TOKENS_JSON: &str = include_str!("../../../data/tokens.json");
const MILITARY_JSON: &str = include_str!("../../../data/military.json");

/// All static game data, parsed once.
#[derive(Debug)]
pub struct Statics {
    /// All 73 age cards, indexed by [`CardId`].
    pub cards: Vec<Card>,
    /// All 12 wonders, indexed by [`WonderId`].
    pub wonders: Vec<Wonder>,
    /// All 10 progress tokens, indexed by [`TokenId`].
    pub tokens: Vec<ProgressToken>,
    /// The military track.
    pub military: MilitaryTrack,
    /// `card_masks[t]` has a bit set for every [`CardId`] of card type `t`,
    /// so a player's count of a colour is a single population count over
    /// their `built` bitset.
    pub card_masks: [u128; 7],
    /// Bit set for every card that is a raw material *or* manufactured good.
    pub raw_and_manufactured_mask: u128,
    /// `age_masks[age - 1]` has a bit set for every card of that age,
    /// including guilds for age 3.
    pub age_masks: [u128; 3],
    /// Bit set for every guild card.
    pub guild_mask: u128,
}

static STATICS: OnceLock<Statics> = OnceLock::new();

/// All static game data, parsed on first use.
///
/// # Panics
///
/// Panics if the embedded `data/*.json` files are malformed or fail
/// validation. That is a build-time authoring error, not a runtime condition:
/// the data is embedded in the binary, so if this succeeds once it succeeds
/// always. [`try_load`] exposes the fallible form for tests.
pub fn statics() -> &'static Statics {
    STATICS.get_or_init(|| try_load().expect("embedded data/*.json must parse and validate"))
}

/// All 73 age cards, indexed by [`CardId`].
#[inline]
pub fn cards() -> &'static [Card] {
    &statics().cards
}

/// All 12 wonders, indexed by [`WonderId`].
#[inline]
pub fn wonders() -> &'static [Wonder] {
    &statics().wonders
}

/// All 10 progress tokens, indexed by [`TokenId`].
#[inline]
pub fn tokens() -> &'static [ProgressToken] {
    &statics().tokens
}

/// The military track definition.
#[inline]
pub fn military() -> &'static MilitaryTrack {
    &statics().military
}

/// Parse and validate the embedded data files, returning an error instead of
/// panicking. Used by [`statics`] and by the data-validation tests.
pub fn try_load() -> Result<Statics, String> {
    load(CARDS_JSON, WONDERS_JSON, TOKENS_JSON, MILITARY_JSON)
}

// ---------------------------------------------------------------------------
// Raw JSON shapes (private) and normalisation
// ---------------------------------------------------------------------------

mod raw {
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    pub struct CardsFile {
        pub cards: Vec<Card>,
    }

    #[derive(Deserialize)]
    pub struct Cost {
        #[serde(default)]
        pub coins: u8,
        #[serde(default)]
        pub resources: BTreeMap<String, u8>,
    }

    #[derive(Deserialize)]
    pub struct Card {
        pub id: String,
        pub name: String,
        pub age: u8,
        #[serde(rename = "type")]
        pub kind: String,
        pub cost: Cost,
        pub chain_from: Option<String>,
        pub chain_to: Option<String>,
        pub effects: Vec<Effect>,
    }

    #[derive(Deserialize)]
    pub struct WondersFile {
        pub wonders: Vec<Wonder>,
    }

    #[derive(Deserialize)]
    pub struct Wonder {
        pub id: String,
        pub name: String,
        pub cost: Cost,
        pub effects: Vec<Effect>,
    }

    #[derive(Deserialize)]
    pub struct TokensFile {
        pub progress_tokens: Vec<Token>,
    }

    #[derive(Deserialize)]
    pub struct Token {
        pub id: String,
        pub name: String,
        pub effects: Vec<Effect>,
    }

    /// The union of every effect object shape used across the four data files.
    /// `deny_unknown_fields` is deliberately *not* set (the data carries
    /// human-readable `note` fields), but the `type` tag is exhaustive: an
    /// unrecognised tag fails to deserialise.
    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Effect {
        ProduceResource {
            resource: String,
            quantity: u8,
        },
        ProduceResourceChoice {
            resource_type: String,
        },
        VictoryPoints {
            amount: u8,
        },
        VictoryPointsPerProgressTokenOwned {
            amount_per_token: u8,
        },
        ScientificSymbol {
            symbol: String,
        },
        Shield {
            quantity: u8,
        },
        TakeCoins {
            amount: u8,
        },
        OpponentLosesCoins {
            amount: u8,
        },
        FixedTradeCost {
            resource: String,
            cost_per_unit: u8,
        },
        RedirectOpponentTradePayments {},
        FutureCostRebate {
            applies_to: String,
            resource_discount: u8,
        },
        ConstructionTriggeredBonus {
            applies_to: String,
            triggered_effect: Box<Effect>,
        },
        ChainBuildTriggeredBonus {
            triggered_effect: Box<Effect>,
        },
        TakeCoinsPerOwnedBuilding {
            amount_per_unit: u8,
            building_type: String,
        },
        TakeCoinsPerConstructedWonder {
            amount_per_unit: u8,
        },
        CoinsByMajority {
            compare: String,
            coins_per_unit: u8,
        },
        PointsByMajority {
            compare: String,
            points_per_unit: u8,
        },
        CountsAsScientificSymbol {
            symbol: String,
        },
        DestroyOpponentBuilding {
            building_type: String,
        },
        BuildDiscardedCardFree {},
        ChooseProgressToken {
            from_pool_size: u8,
        },
        PlayAgain {},
    }

    #[derive(Deserialize)]
    pub struct MilitaryFile {
        pub conflict_track: ConflictTrack,
        pub military_tokens: Vec<MilitaryToken>,
        pub end_of_game_military_scoring: Scoring,
    }

    #[derive(Deserialize)]
    pub struct ConflictTrack {
        pub capital_position: u8,
    }

    #[derive(Deserialize)]
    pub struct MilitaryToken {
        pub distance_from_center: u8,
        pub effect: Effect,
    }

    #[derive(Deserialize)]
    pub struct Scoring {
        pub table: Vec<ScoringRow>,
    }

    #[derive(Deserialize)]
    pub struct ScoringRow {
        pub pawn_distance_from_center: Option<u8>,
        pub pawn_distance_from_center_range: Option<[u8; 2]>,
        pub victory_points: u8,
    }
}

fn parse_resource(s: &str) -> Result<Resource, String> {
    Ok(match s {
        "wood" => Resource::Wood,
        "clay" => Resource::Clay,
        "stone" => Resource::Stone,
        "glass" => Resource::Glass,
        "papyrus" => Resource::Papyrus,
        other => return Err(format!("unknown resource {other:?}")),
    })
}

fn parse_card_type(s: &str) -> Result<CardType, String> {
    Ok(match s {
        "raw_material" => CardType::RawMaterial,
        "manufactured_good" => CardType::ManufacturedGood,
        "civilian" => CardType::Civilian,
        "scientific" => CardType::Scientific,
        "commercial" => CardType::Commercial,
        "military" => CardType::Military,
        "guild" => CardType::Guild,
        other => return Err(format!("unknown card type {other:?}")),
    })
}

fn parse_resource_group(s: &str) -> Result<ResourceGroup, String> {
    Ok(match s {
        "raw_material" => ResourceGroup::RawMaterial,
        "manufactured_good" => ResourceGroup::ManufacturedGood,
        other => return Err(format!("unknown resource group {other:?}")),
    })
}

fn parse_science(s: &str) -> Result<Science, String> {
    Ok(match s {
        "mortar" => Science::Mortar,
        "pendulum" => Science::Pendulum,
        "inkwell" => Science::Inkwell,
        "wheel" => Science::Wheel,
        "sundial" => Science::Sundial,
        "gyroscope" => Science::Gyroscope,
        "balance" => Science::Balance,
        other => return Err(format!("unknown scientific symbol {other:?}")),
    })
}

/// Parse a `compare` / `building_type` string into a [`CountTarget`].
fn parse_count_target(s: &str) -> Result<CountTarget, String> {
    Ok(match s {
        "commercial_buildings" | "commercial" => CountTarget::Cards(CardType::Commercial),
        "civilian_buildings" | "civilian" => CountTarget::Cards(CardType::Civilian),
        "scientific_buildings" | "scientific" => CountTarget::Cards(CardType::Scientific),
        "military_buildings" | "military" => CountTarget::Cards(CardType::Military),
        "raw_material" => CountTarget::Cards(CardType::RawMaterial),
        "manufactured_good" => CountTarget::Cards(CardType::ManufacturedGood),
        "raw_and_manufactured_buildings" => CountTarget::RawAndManufactured,
        "constructed_wonders" => CountTarget::Wonders,
        "coins_div_3" => CountTarget::CoinsDiv3,
        other => return Err(format!("unknown count target {other:?}")),
    })
}

fn cost_resources(cost: &raw::Cost) -> Result<[u8; NUM_RESOURCES], String> {
    let mut out = [0u8; NUM_RESOURCES];
    for (k, v) in &cost.resources {
        out[parse_resource(k)?.index()] += v;
    }
    Ok(out)
}

/// Parse and validate all four data files.
///
/// Returns `Err` with a human-readable message on any malformed entry,
/// unknown effect type, unresolvable chain reference, or failed count
/// expectation.
pub fn load(
    cards_json: &'static str,
    wonders_json: &'static str,
    tokens_json: &'static str,
    military_json: &'static str,
) -> Result<Statics, String> {
    let raw_cards: raw::CardsFile =
        serde_json::from_str(cards_json).map_err(|e| format!("cards.json: {e}"))?;
    let raw_wonders: raw::WondersFile =
        serde_json::from_str(wonders_json).map_err(|e| format!("wonders.json: {e}"))?;
    let raw_tokens: raw::TokensFile =
        serde_json::from_str(tokens_json).map_err(|e| format!("tokens.json: {e}"))?;
    let raw_military: raw::MilitaryFile =
        serde_json::from_str(military_json).map_err(|e| format!("military.json: {e}"))?;

    // Slug -> index, so chain references can be resolved.
    let mut index_of = std::collections::HashMap::new();
    for (i, c) in raw_cards.cards.iter().enumerate() {
        if index_of.insert(c.id.clone(), i).is_some() {
            return Err(format!("duplicate card id {:?}", c.id));
        }
    }
    let card_id = |slug: &str| -> Result<CardId, String> {
        index_of
            .get(slug)
            .map(|&i| CardId(i as u8))
            .ok_or_else(|| format!("unknown card reference {slug:?}"))
    };

    let mut cards = Vec::with_capacity(raw_cards.cards.len());
    for rc in &raw_cards.cards {
        let mut c = Card {
            id: leak(&rc.id),
            name: leak(&rc.name),
            age: rc.age,
            kind: parse_card_type(&rc.kind).map_err(|e| format!("card {}: {e}", rc.id))?,
            coin_cost: rc.cost.coins,
            resource_cost: cost_resources(&rc.cost).map_err(|e| format!("card {}: {e}", rc.id))?,
            chain_from: rc.chain_from.as_deref().map(&card_id).transpose()?,
            chain_to: rc.chain_to.as_deref().map(&card_id).transpose()?,
            produces: [0; NUM_RESOURCES],
            produces_choice: None,
            victory_points: 0,
            science: None,
            shields: 0,
            coins: 0,
            fixed_trade: [false; NUM_RESOURCES],
            coins_per_own: None,
            coins_by_majority: None,
            points_by_majority: None,
        };
        for e in &rc.effects {
            let ctx = |e: String| format!("card {}: {e}", rc.id);
            match e {
                raw::Effect::ProduceResource { resource, quantity } => {
                    c.produces[parse_resource(resource).map_err(ctx)?.index()] += quantity;
                }
                raw::Effect::ProduceResourceChoice { resource_type } => {
                    c.produces_choice = Some(parse_resource_group(resource_type).map_err(ctx)?);
                }
                raw::Effect::VictoryPoints { amount } => c.victory_points += amount,
                raw::Effect::ScientificSymbol { symbol } => {
                    c.science = Some(parse_science(symbol).map_err(ctx)?);
                }
                raw::Effect::Shield { quantity } => c.shields += quantity,
                raw::Effect::TakeCoins { amount } => c.coins += amount,
                raw::Effect::FixedTradeCost {
                    resource,
                    cost_per_unit,
                } => {
                    if *cost_per_unit != 1 {
                        return Err(ctx(format!(
                            "fixed_trade_cost per-unit is {cost_per_unit}, engine only models 1"
                        )));
                    }
                    c.fixed_trade[parse_resource(resource).map_err(ctx)?.index()] = true;
                }
                raw::Effect::TakeCoinsPerOwnedBuilding {
                    amount_per_unit,
                    building_type,
                } => {
                    c.coins_per_own = Some((
                        parse_count_target(building_type).map_err(ctx)?,
                        *amount_per_unit,
                    ));
                }
                raw::Effect::TakeCoinsPerConstructedWonder { amount_per_unit } => {
                    c.coins_per_own = Some((CountTarget::Wonders, *amount_per_unit));
                }
                raw::Effect::CoinsByMajority {
                    compare,
                    coins_per_unit,
                } => {
                    c.coins_by_majority =
                        Some((parse_count_target(compare).map_err(ctx)?, *coins_per_unit));
                }
                raw::Effect::PointsByMajority {
                    compare,
                    points_per_unit,
                } => {
                    c.points_by_majority =
                        Some((parse_count_target(compare).map_err(ctx)?, *points_per_unit));
                }
                other => {
                    return Err(ctx(format!(
                        "effect {:?} is not valid on an age card",
                        effect_name(other)
                    )))
                }
            }
        }
        cards.push(c);
    }

    let mut wonders = Vec::with_capacity(raw_wonders.wonders.len());
    for rw in &raw_wonders.wonders {
        let mut w = Wonder {
            id: leak(&rw.id),
            name: leak(&rw.name),
            coin_cost: rw.cost.coins,
            resource_cost: cost_resources(&rw.cost)
                .map_err(|e| format!("wonder {}: {e}", rw.id))?,
            victory_points: 0,
            shields: 0,
            coins: 0,
            opponent_loses_coins: 0,
            produces_choice: None,
            play_again: false,
            destroy: None,
            build_discarded_free: false,
            choose_progress_token: false,
        };
        for e in &rw.effects {
            let ctx = |e: String| format!("wonder {}: {e}", rw.id);
            match e {
                raw::Effect::VictoryPoints { amount } => w.victory_points += amount,
                raw::Effect::Shield { quantity } => w.shields += quantity,
                raw::Effect::TakeCoins { amount } => w.coins += amount,
                raw::Effect::OpponentLosesCoins { amount } => w.opponent_loses_coins += amount,
                raw::Effect::ProduceResourceChoice { resource_type } => {
                    w.produces_choice = Some(parse_resource_group(resource_type).map_err(ctx)?);
                }
                raw::Effect::PlayAgain {} => w.play_again = true,
                raw::Effect::DestroyOpponentBuilding { building_type } => {
                    w.destroy = Some(parse_card_type(building_type).map_err(ctx)?);
                }
                raw::Effect::BuildDiscardedCardFree {} => w.build_discarded_free = true,
                raw::Effect::ChooseProgressToken { from_pool_size } => {
                    if *from_pool_size != 3 {
                        return Err(ctx(format!(
                            "choose_progress_token pool size is {from_pool_size}, engine models 3"
                        )));
                    }
                    w.choose_progress_token = true;
                }
                other => {
                    return Err(ctx(format!(
                        "effect {:?} is not valid on a wonder",
                        effect_name(other)
                    )))
                }
            }
        }
        wonders.push(w);
    }

    let mut tokens = Vec::with_capacity(raw_tokens.progress_tokens.len());
    for rt in &raw_tokens.progress_tokens {
        let mut t = ProgressToken {
            id: leak(&rt.id),
            name: leak(&rt.name),
            coins: 0,
            victory_points: 0,
            vp_per_token: 0,
            science: None,
            discount: None,
            gain_trade_costs: false,
            shield_bonus: false,
            wonder_play_again: false,
            chain_build_coins: 0,
        };
        for e in &rt.effects {
            let ctx = |e: String| format!("token {}: {e}", rt.id);
            match e {
                raw::Effect::TakeCoins { amount } => t.coins += amount,
                raw::Effect::VictoryPoints { amount } => t.victory_points += amount,
                raw::Effect::VictoryPointsPerProgressTokenOwned { amount_per_token } => {
                    t.vp_per_token = *amount_per_token;
                }
                raw::Effect::CountsAsScientificSymbol { symbol } => {
                    t.science = Some(parse_science(symbol).map_err(ctx)?);
                }
                raw::Effect::FutureCostRebate {
                    applies_to,
                    resource_discount,
                } => {
                    if *resource_discount != 2 {
                        return Err(ctx(format!(
                            "future_cost_rebate discount is {resource_discount}, engine models 2"
                        )));
                    }
                    t.discount = Some(match applies_to.as_str() {
                        "wonders" => DiscountTarget::Wonders,
                        "civilian_buildings" => DiscountTarget::CivilianBuildings,
                        other => return Err(ctx(format!("unknown rebate target {other:?}"))),
                    });
                }
                raw::Effect::RedirectOpponentTradePayments {} => t.gain_trade_costs = true,
                raw::Effect::ConstructionTriggeredBonus {
                    applies_to,
                    triggered_effect,
                } => match (applies_to.as_str(), &**triggered_effect) {
                    ("military_buildings", raw::Effect::Shield { quantity: 1 }) => {
                        t.shield_bonus = true;
                    }
                    ("wonders", raw::Effect::PlayAgain {}) => t.wonder_play_again = true,
                    (a, e) => {
                        return Err(ctx(format!(
                            "unsupported construction_triggered_bonus {a:?} -> {:?}",
                            effect_name(e)
                        )))
                    }
                },
                raw::Effect::ChainBuildTriggeredBonus { triggered_effect } => {
                    match &**triggered_effect {
                        raw::Effect::TakeCoins { amount } => t.chain_build_coins = *amount,
                        e => {
                            return Err(ctx(format!(
                                "unsupported chain_build_triggered_bonus -> {:?}",
                                effect_name(e)
                            )))
                        }
                    }
                }
                other => {
                    return Err(ctx(format!(
                        "effect {:?} is not valid on a progress token",
                        effect_name(other)
                    )))
                }
            }
        }
        tokens.push(t);
    }

    // Military track.
    let mut loot_vec: Vec<(u8, u8)> = Vec::new();
    for mt in &raw_military.military_tokens {
        match &mt.effect {
            raw::Effect::OpponentLosesCoins { amount } => {
                loot_vec.push((mt.distance_from_center, *amount));
            }
            e => {
                return Err(format!(
                    "military.json: unsupported military token effect {:?}",
                    effect_name(e)
                ))
            }
        }
    }
    loot_vec.sort_unstable();
    let loot: [(u8, u8); 2] = loot_vec
        .try_into()
        .map_err(|v: Vec<_>| format!("military.json: expected 2 loot tokens, got {}", v.len()))?;

    let mut vp_vec: Vec<(u8, u8)> = Vec::new();
    for row in &raw_military.end_of_game_military_scoring.table {
        let max = match (
            row.pawn_distance_from_center,
            row.pawn_distance_from_center_range,
        ) {
            (Some(d), None) => d,
            (None, Some([_, hi])) => hi,
            _ => {
                return Err(
                    "military.json: scoring row needs exactly one of distance / range".into(),
                )
            }
        };
        vp_vec.push((max, row.victory_points));
    }
    vp_vec.sort_unstable();
    let victory_points: [(u8, u8); 4] = vp_vec
        .try_into()
        .map_err(|v: Vec<_>| format!("military.json: expected 4 scoring rows, got {}", v.len()))?;

    let military = MilitaryTrack {
        capital_distance: raw_military.conflict_track.capital_position,
        loot,
        victory_points,
    };

    // Derived bitmasks.
    let mut card_masks = [0u128; 7];
    let mut age_masks = [0u128; 3];
    let mut guild_mask = 0u128;
    for (i, c) in cards.iter().enumerate() {
        let bit = 1u128 << i;
        card_masks[c.kind.index()] |= bit;
        if c.age < 1 || c.age > 3 {
            return Err(format!("card {}: age {} out of range", c.id, c.age));
        }
        age_masks[(c.age - 1) as usize] |= bit;
        if c.is_guild() {
            guild_mask |= bit;
        }
    }
    let raw_and_manufactured_mask =
        card_masks[CardType::RawMaterial.index()] | card_masks[CardType::ManufacturedGood.index()];

    let s = Statics {
        cards,
        wonders,
        tokens,
        military,
        card_masks,
        raw_and_manufactured_mask,
        age_masks,
        guild_mask,
    };
    validate(&s)?;
    Ok(s)
}

/// Cross-check the parsed data against the base game's known component counts
/// and internal consistency rules.
fn validate(s: &Statics) -> Result<(), String> {
    if s.cards.len() != NUM_CARDS {
        return Err(format!("expected {NUM_CARDS} cards, got {}", s.cards.len()));
    }
    if s.wonders.len() != NUM_WONDERS {
        return Err(format!(
            "expected {NUM_WONDERS} wonders, got {}",
            s.wonders.len()
        ));
    }
    if s.tokens.len() != NUM_TOKENS {
        return Err(format!(
            "expected {NUM_TOKENS} progress tokens, got {}",
            s.tokens.len()
        ));
    }

    let age1 = s.cards.iter().filter(|c| c.age == 1).count();
    let age2 = s.cards.iter().filter(|c| c.age == 2).count();
    let age3_non_guild = s
        .cards
        .iter()
        .filter(|c| c.age == 3 && !c.is_guild())
        .count();
    let guilds = s.cards.iter().filter(|c| c.is_guild()).count();
    if (age1, age2, age3_non_guild, guilds) != (23, 23, 20, 7) {
        return Err(format!(
            "expected 23/23/20 age cards + 7 guilds, got {age1}/{age2}/{age3_non_guild} + {guilds}"
        ));
    }
    if s.cards.iter().any(|c| c.is_guild() && c.age != 3) {
        return Err("guild cards must be age 3".into());
    }

    // Chain links must be symmetric.
    for (i, c) in s.cards.iter().enumerate() {
        if let Some(to) = c.chain_to {
            let other = &s.cards[to.index()];
            if other.chain_from.map(|f| f.index()) != Some(i) {
                return Err(format!(
                    "chain_to {} on {} is not mirrored by chain_from",
                    other.id, c.id
                ));
            }
        }
        if let Some(from) = c.chain_from {
            let other = &s.cards[from.index()];
            if other.chain_to.map(|t| t.index()) != Some(i) {
                return Err(format!(
                    "chain_from {} on {} is not mirrored by chain_to",
                    other.id, c.id
                ));
            }
            if other.age >= c.age {
                return Err(format!(
                    "chain prerequisite {} (age {}) is not from an earlier age than {} (age {})",
                    other.id, other.age, c.id, c.age
                ));
            }
        }
    }

    // Every card-borne scientific symbol must appear exactly twice, so a pair
    // is always achievable and never achievable more than once.
    let mut science_counts = [0usize; NUM_SCIENCE];
    for c in &s.cards {
        if let Some(sym) = c.science {
            science_counts[sym.index()] += 1;
        }
    }
    if science_counts[Science::Balance.index()] != 0 {
        return Err("no age card may carry the Balance symbol (Law token only)".into());
    }
    for sym in [
        Science::Mortar,
        Science::Pendulum,
        Science::Inkwell,
        Science::Wheel,
        Science::Sundial,
        Science::Gyroscope,
    ] {
        if science_counts[sym.index()] != 2 {
            return Err(format!(
                "scientific symbol {sym:?} appears {} times, expected 2",
                science_counts[sym.index()]
            ));
        }
    }

    // Guild cards are the only ones with majority-based effects.
    for c in &s.cards {
        let has_majority = c.coins_by_majority.is_some() || c.points_by_majority.is_some();
        if has_majority != c.is_guild() {
            return Err(format!(
                "{} has majority effects = {has_majority} but is_guild = {}",
                c.id,
                c.is_guild()
            ));
        }
    }

    // Exactly one Law-style token, one Economy, one Strategy, one Theology,
    // one Urbanism, one Mathematics, and two rebate tokens.
    let count = |f: fn(&ProgressToken) -> bool| s.tokens.iter().filter(|t| f(t)).count();
    for (what, got, want) in [
        ("science tokens", count(|t| t.science.is_some()), 1),
        ("rebate tokens", count(|t| t.discount.is_some()), 2),
        ("economy tokens", count(|t| t.gain_trade_costs), 1),
        ("strategy tokens", count(|t| t.shield_bonus), 1),
        ("theology tokens", count(|t| t.wonder_play_again), 1),
        ("urbanism tokens", count(|t| t.chain_build_coins > 0), 1),
        ("mathematics tokens", count(|t| t.vp_per_token > 0), 1),
    ] {
        if got != want {
            return Err(format!("expected {want} {what}, got {got}"));
        }
    }

    // Wonders: exactly one Great Library, one Mausoleum, two destroyers.
    let wcount = |f: fn(&Wonder) -> bool| s.wonders.iter().filter(|w| f(w)).count();
    for (what, got, want) in [
        ("great libraries", wcount(|w| w.choose_progress_token), 1),
        ("mausoleums", wcount(|w| w.build_discarded_free), 1),
        ("destroyer wonders", wcount(|w| w.destroy.is_some()), 2),
    ] {
        if got != want {
            return Err(format!("expected {want} {what}, got {got}"));
        }
    }

    if s.military.capital_distance != 9 {
        return Err(format!(
            "expected capital at distance 9, got {}",
            s.military.capital_distance
        ));
    }
    if s.military.loot != [(3, 2), (6, 5)] {
        return Err(format!(
            "expected loot tokens at 3/-2 and 6/-5, got {:?}",
            s.military.loot
        ));
    }
    Ok(())
}

fn effect_name(e: &raw::Effect) -> &'static str {
    use raw::Effect as E;
    match e {
        E::ProduceResource { .. } => "produce_resource",
        E::ProduceResourceChoice { .. } => "produce_resource_choice",
        E::VictoryPoints { .. } => "victory_points",
        E::VictoryPointsPerProgressTokenOwned { .. } => "victory_points_per_progress_token_owned",
        E::ScientificSymbol { .. } => "scientific_symbol",
        E::Shield { .. } => "shield",
        E::TakeCoins { .. } => "take_coins",
        E::OpponentLosesCoins { .. } => "opponent_loses_coins",
        E::FixedTradeCost { .. } => "fixed_trade_cost",
        E::RedirectOpponentTradePayments {} => "redirect_opponent_trade_payments",
        E::FutureCostRebate { .. } => "future_cost_rebate",
        E::ConstructionTriggeredBonus { .. } => "construction_triggered_bonus",
        E::ChainBuildTriggeredBonus { .. } => "chain_build_triggered_bonus",
        E::TakeCoinsPerOwnedBuilding { .. } => "take_coins_per_owned_building",
        E::TakeCoinsPerConstructedWonder { .. } => "take_coins_per_constructed_wonder",
        E::CoinsByMajority { .. } => "coins_by_majority",
        E::PointsByMajority { .. } => "points_by_majority",
        E::CountsAsScientificSymbol { .. } => "counts_as_scientific_symbol",
        E::DestroyOpponentBuilding { .. } => "destroy_opponent_building",
        E::BuildDiscardedCardFree {} => "build_discarded_card_free",
        E::ChooseProgressToken { .. } => "choose_progress_token",
        E::PlayAgain {} => "play_again",
    }
}

/// Promote an owned string from the parsed data to `'static`.
///
/// The static tables live for the whole process (they are behind a
/// [`OnceLock`]), so this leak is bounded by the size of `data/*.json` and
/// happens at most once per process. Keeping the names as `&'static str`
/// rather than `String` is what lets [`Card`] be cheaply copied around and
/// referenced from `Debug` output without lifetimes leaking into
/// [`crate::GameState`].
fn leak(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_data_parses_and_validates() {
        try_load().expect("data/*.json should parse and validate");
    }

    #[test]
    fn component_counts_match_the_base_game() {
        let s = statics();
        assert_eq!(s.cards.len(), 73);
        assert_eq!(s.wonders.len(), 12);
        assert_eq!(s.tokens.len(), 10);
        assert_eq!(s.cards.iter().filter(|c| c.age == 1).count(), 23);
        assert_eq!(s.cards.iter().filter(|c| c.age == 2).count(), 23);
        assert_eq!(
            s.cards
                .iter()
                .filter(|c| c.age == 3 && !c.is_guild())
                .count(),
            20
        );
        assert_eq!(s.cards.iter().filter(|c| c.is_guild()).count(), 7);
    }

    #[test]
    fn ids_round_trip_through_slugs() {
        for id in CardId::all() {
            assert_eq!(CardId::from_slug(id.slug()), Some(id));
        }
        for id in WonderId::all() {
            assert_eq!(WonderId::from_slug(id.slug()), Some(id));
        }
        for id in TokenId::all() {
            assert_eq!(TokenId::from_slug(id.slug()), Some(id));
        }
        assert_eq!(CardId::from_slug("not-a-card"), None);
    }

    #[test]
    fn ids_serialise_as_slugs() {
        let id = CardId::from_slug("lumber-yard").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"lumber-yard\"");
        let back: CardId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn spot_check_normalised_cards() {
        let tavern = CardId::from_slug("tavern").unwrap().def();
        assert_eq!(tavern.coins, 4);
        assert_eq!(tavern.kind, CardType::Commercial);
        assert_eq!(tavern.chain_to.unwrap().slug(), "lighthouse");

        let reserve = CardId::from_slug("stone-reserve").unwrap().def();
        assert_eq!(reserve.coin_cost, 3);
        assert!(reserve.fixed_trade[Resource::Stone.index()]);
        assert!(!reserve.fixed_trade[Resource::Wood.index()]);

        let customs = CardId::from_slug("customs-house").unwrap().def();
        assert!(customs.fixed_trade[Resource::Glass.index()]);
        assert!(customs.fixed_trade[Resource::Papyrus.index()]);

        let sawmill = CardId::from_slug("sawmill").unwrap().def();
        assert_eq!(sawmill.produces[Resource::Wood.index()], 2);
        assert_eq!(sawmill.coin_cost, 2);

        let forum = CardId::from_slug("forum").unwrap().def();
        assert_eq!(forum.produces_choice, Some(ResourceGroup::ManufacturedGood));

        let builders = CardId::from_slug("builders-guild").unwrap().def();
        assert_eq!(builders.coins_by_majority, None);
        assert_eq!(builders.points_by_majority, Some((CountTarget::Wonders, 2)));

        let merchants = CardId::from_slug("merchants-guild").unwrap().def();
        assert_eq!(
            merchants.coins_by_majority,
            Some((CountTarget::Cards(CardType::Commercial), 1))
        );
        assert_eq!(
            merchants.points_by_majority,
            Some((CountTarget::Cards(CardType::Commercial), 1))
        );
    }

    #[test]
    fn spot_check_normalised_wonders_and_tokens() {
        let library = WonderId::from_slug("the-great-library").unwrap().def();
        assert!(library.choose_progress_token);
        assert_eq!(library.victory_points, 4);

        let zeus = WonderId::from_slug("the-statue-of-zeus").unwrap().def();
        assert_eq!(zeus.destroy, Some(CardType::RawMaterial));
        assert_eq!(zeus.shields, 1);

        let appian = WonderId::from_slug("the-appian-way").unwrap().def();
        assert_eq!(appian.coins, 3);
        assert_eq!(appian.opponent_loses_coins, 3);
        assert!(appian.play_again);

        let law = TokenId::from_slug("law").unwrap().def();
        assert_eq!(law.science, Some(Science::Balance));

        let arch = TokenId::from_slug("architecture").unwrap().def();
        assert_eq!(arch.discount, Some(DiscountTarget::Wonders));

        let urbanism = TokenId::from_slug("urbanism").unwrap().def();
        assert_eq!(urbanism.coins, 6);
        assert_eq!(urbanism.chain_build_coins, 4);

        let maths = TokenId::from_slug("mathematics").unwrap().def();
        assert_eq!(maths.vp_per_token, 3);
    }

    #[test]
    fn military_track_scoring_table() {
        let m = military();
        assert_eq!(m.vp_for_distance(0), 0);
        assert_eq!(m.vp_for_distance(1), 2);
        assert_eq!(m.vp_for_distance(2), 2);
        assert_eq!(m.vp_for_distance(3), 5);
        assert_eq!(m.vp_for_distance(5), 5);
        assert_eq!(m.vp_for_distance(6), 10);
        assert_eq!(m.vp_for_distance(8), 10);
        assert_eq!(m.loot, [(3, 2), (6, 5)]);
        assert_eq!(m.capital_distance, 9);
    }

    #[test]
    fn unknown_effect_type_is_a_hard_error() {
        let bad = r#"{"$schema_version":1,"cards":[{"id":"x","name":"X","age":1,
            "type":"civilian","cost":{"coins":0,"resources":{}},"chain_from":null,
            "chain_to":null,"effects":[{"type":"teleport","amount":1}]}]}"#;
        let err = load(
            Box::leak(bad.to_owned().into_boxed_str()),
            WONDERS_JSON,
            TOKENS_JSON,
            MILITARY_JSON,
        )
        .expect_err("unknown effect type must not be silently ignored");
        assert!(err.contains("cards.json"), "unexpected error: {err}");
    }

    #[test]
    fn effect_valid_on_the_wrong_entity_is_a_hard_error() {
        let bad = r#"{"$schema_version":1,"cards":[{"id":"x","name":"X","age":1,
            "type":"civilian","cost":{"coins":0,"resources":{}},"chain_from":null,
            "chain_to":null,"effects":[{"type":"play_again"}]}]}"#;
        let err = load(
            Box::leak(bad.to_owned().into_boxed_str()),
            WONDERS_JSON,
            TOKENS_JSON,
            MILITARY_JSON,
        )
        .expect_err("play_again is not an age-card effect");
        assert!(err.contains("play_again"), "unexpected error: {err}");
    }
}
