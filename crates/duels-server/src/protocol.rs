//! The wire protocol between `duels-server` and any client (the `web`
//! React app today; a spectator or a second remote human later).
//!
//! Every type here is `#[derive(TS)]`'d so `cargo test` regenerates
//! `web/src/generated/*.ts` from these exact definitions — the web client
//! never hand-writes a parallel schema. See `docs/agent-contract.md` for the
//! equivalent contract between `duels-core` and an `Agent`; this module is
//! the analogous contract between the server and a browser.
//!
//! # Room / seat model
//!
//! A [`Room`] has exactly two seats, each independently a [`SeatSpec::Human`]
//! or a [`SeatSpec::Agent`]. M2 only ever creates `Human vs Agent` (vs the
//! `random` bot) or `Human vs Human` (hot-seat, one browser tab controlling
//! both) rooms, and a single WebSocket connection per room controls whichever
//! seat is currently on move — there is no per-connection identity yet.
//! Nothing here forecloses adding one later: a second human connection, a
//! spectator connection, or a different agent name are all just more
//! [`SeatSpec`] values and more subscribers to the same broadcast stream.
//!
//! [`Room`]: crate::room::Room

use duels_core::{Action, Breakdown, Event, Observation};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

use duels_core::data::{CardId, TokenId, WonderId};

/// One seat of a room: a human at a browser, or a named `Agent`
/// implementation. See `room::KNOWN_AGENTS` (served over `GET /agents`) for
/// the names this build of the server accepts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SeatSpec {
    /// Controlled by whichever browser is connected to the room's WebSocket.
    Human,
    /// Controlled by a `duels-agents-api::Agent`, driven server-side.
    Agent {
        /// The agent's name, as reported by `Agent::spec().name`. Must be
        /// one of `room::KNOWN_AGENTS`.
        name: String,
    },
}

/// `POST /rooms` request body: what each seat should be, and an optional
/// seed for reproducibility (a fresh one is minted if omitted).
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct CreateRoomRequest {
    /// The two seats, indexed like [`duels_core::Player::index`].
    pub seats: [SeatSpec; 2],
    /// The RNG seed to build the game from. Random if omitted.
    pub seed: Option<u64>,
}

/// `POST /rooms` response: the id of the room just created.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct CreateRoomResponse {
    /// The room's id, used in `/rooms/:id` and `/rooms/:id/ws`.
    pub room_id: String,
}

/// Coarse room lifecycle state, for `GET /rooms/:id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum RoomStatus {
    /// The game is still in progress (including the wonder draft).
    Playing,
    /// [`duels_core::GameState::result`] is `Some`.
    GameOver,
}

/// `GET /rooms/:id` response: room metadata without the (potentially large)
/// game state.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct RoomInfo {
    /// The room's id.
    pub room_id: String,
    /// The two seats.
    pub seats: [SeatSpec; 2],
    /// Whether the game is still going.
    pub status: RoomStatus,
    /// How many decisions have been resolved so far.
    pub turn: u32,
}

/// The coin cost or reward of one of the current legal actions, computed
/// server-side from `duels_core::cost` (and
/// [`duels_core::cost::discard_reward`] for `Discard`) so the client never
/// has to reimplement the cost engine. Only the three actions that move
/// coins are represented; the rest (picking a wonder, resolving a pending
/// choice) have no cost to display.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(tag = "type")]
pub enum ActionCost {
    /// Cost to construct the card in `slot`.
    Build {
        /// The slot this cost applies to.
        slot: u8,
        /// Total coins owed.
        coins: u16,
        /// The portion of `coins` that is a trade payment (see
        /// [`duels_core::cost::Cost::trade`]).
        trade: u16,
        /// Whether a chain symbol makes this free.
        via_chain: bool,
    },
    /// Coins gained for discarding the card in `slot`.
    Discard {
        /// The slot this reward applies to.
        slot: u8,
        /// Coins gained.
        reward: u16,
    },
    /// Cost to spend the card in `slot` on constructing `wonder`.
    BuildWonder {
        /// The slot this cost applies to.
        slot: u8,
        /// The wonder being constructed.
        wonder: WonderId,
        /// Total coins owed.
        coins: u16,
        /// The portion of `coins` that is a trade payment.
        trade: u16,
    },
}

/// A message the client sends over the room's WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(tag = "type")]
pub enum ClientMessage {
    /// Submit an action for whichever seat is currently on move. The server
    /// rejects it (with [`ServerMessage::Error`]) unless it is exactly one of
    /// the actions most recently sent in [`StatePayload::legal_actions`].
    Action {
        /// The action to apply.
        action: Action,
    },
}

/// The full state snapshot broadcast to every connection on a room: sent
/// once on connect, and again after every action (human- or agent-chosen)
/// is applied.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct StatePayload {
    /// The public view of the game, straight from
    /// [`duels_core::GameState::observation`].
    pub observation: Observation,
    /// The current seat assignment, so the client knows whether the seat on
    /// move is a human (and should show controls) or an agent (already
    /// resolved server-side by the time this message arrives).
    pub seats: [SeatSpec; 2],
    /// Every action legal right now. Empty iff the game is over. Since agent
    /// turns are resolved synchronously before broadcasting, whenever this is
    /// non-empty the seat on move is a [`SeatSpec::Human`].
    pub legal_actions: Vec<Action>,
    /// Coin cost/reward for the `Build`/`Discard`/`BuildWonder` entries of
    /// `legal_actions`, computed server-side.
    pub action_costs: Vec<ActionCost>,
    /// What happened since the previous [`StatePayload`] (empty for the very
    /// first one sent on connect). A UI can animate these; this milestone's
    /// client just uses them for a lightweight log.
    pub events: Vec<Event>,
    /// The full victory-point breakdown, present once
    /// `observation.result` is `Some`.
    pub breakdown: Option<[Breakdown; 2]>,
}

/// A message the server sends over the room's WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(tag = "type")]
pub enum ServerMessage {
    /// A new state snapshot.
    State(Box<StatePayload>),
    /// The submitted `ClientMessage` was rejected (most commonly: the action
    /// was not in the `legal_actions` most recently sent). The room's state
    /// did not change.
    Error {
        /// A human-readable explanation.
        message: String,
    },
}

/// One card's static, game-independent facts, for `GET /catalog`.
///
/// Deliberately a server-side DTO rather than exposing
/// `duels_core::data::Card` directly: that type isn't `Serialize` (it holds
/// `&'static str` and is an internal, load-time representation), and this
/// shape lets the catalog stay a stable wire contract independent of how
/// `duels-core` normalises `data/cards.json` internally.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct CardCatalogEntry {
    /// This card's id.
    pub id: CardId,
    /// Printed name.
    pub name: String,
    /// 1, 2 or 3.
    pub age: u8,
    /// The card's colour.
    pub kind: duels_core::data::CardType,
    /// Printed coin cost.
    pub coin_cost: u8,
    /// Printed resource cost.
    pub resource_cost: ResourceAmounts,
    /// The earlier-age card that makes this one free, if any.
    pub chain_from: Option<CardId>,
    /// The later-age card this one makes free, if any.
    pub chain_to: Option<CardId>,
    /// Resources this card produces unconditionally.
    pub produces: ResourceAmounts,
    /// The resource group this card lets its owner produce one of, per
    /// payment, if any.
    pub produces_choice: Option<ResourceGroupLabel>,
    /// Printed victory points.
    pub victory_points: u8,
    /// The scientific symbol this card carries, if any.
    pub science: Option<duels_core::data::Science>,
    /// Shields this card grants when built.
    pub shields: u8,
    /// Coins gained immediately when built.
    pub coins: u8,
    /// Resources this card fixes at 1 coin per unit for its owner
    /// (a trading post), regardless of the opponent's production.
    pub fixed_trade: Vec<duels_core::data::Resource>,
    /// A yellow Age III "coins per building you own" effect, as
    /// `(what is counted, coins per unit)`.
    pub coins_per_own: Option<(String, u8)>,
    /// A guild's immediate "coins per building, whoever has more" effect.
    pub coins_by_majority: Option<(String, u8)>,
    /// A guild's "victory points per building, whoever has more at game end"
    /// effect.
    pub points_by_majority: Option<(String, u8)>,
    /// Whether this is one of the seven guild cards.
    pub is_guild: bool,
}

/// One wonder's static facts, for `GET /catalog`.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct WonderCatalogEntry {
    /// This wonder's id.
    pub id: WonderId,
    /// Printed name.
    pub name: String,
    /// Printed coin cost.
    pub coin_cost: u8,
    /// Printed resource cost.
    pub resource_cost: ResourceAmounts,
    /// Printed victory points.
    pub victory_points: u8,
    /// Shields granted when built.
    pub shields: u8,
    /// Coins gained immediately when built.
    pub coins: u8,
    /// Coins the opponent loses immediately when built.
    pub opponent_loses_coins: u8,
    /// The resource group this wonder lets its owner produce one of.
    pub produces_choice: Option<ResourceGroupLabel>,
    /// Grants an immediate extra turn when built.
    pub play_again: bool,
    /// Lets the owner discard one opponent building of this colour when
    /// built.
    pub destroy: Option<duels_core::data::CardType>,
    /// Lets the owner build a card from the discard pile for free when
    /// built (The Mausoleum).
    pub build_discarded_free: bool,
    /// Lets the owner immediately take a progress token when built (The
    /// Great Library).
    pub choose_progress_token: bool,
}

/// One progress token's static facts, for `GET /catalog`.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct TokenCatalogEntry {
    /// This token's id.
    pub id: TokenId,
    /// Printed name.
    pub name: String,
    /// Coins gained once, when taken.
    pub coins: u8,
    /// Flat victory points.
    pub victory_points: u8,
    /// Victory points per progress token the owner holds at game end
    /// (Mathematics).
    pub vp_per_token: u8,
    /// A scientific symbol this token itself provides (Law).
    pub science: Option<duels_core::data::Science>,
    /// What this token discounts by 2 resources, if anything (Architecture,
    /// Masonry).
    pub discount: Option<DiscountLabel>,
    /// Economy: the opponent's trade payments come to this token's owner
    /// instead of the bank.
    pub gain_trade_costs: bool,
    /// Strategy: +1 shield on every red building the owner constructs.
    pub shield_bonus: bool,
    /// Theology: an extra turn on every wonder the owner constructs.
    pub wonder_play_again: bool,
    /// Urbanism: coins gained each time the owner builds via a chain symbol.
    pub chain_build_coins: u8,
}

/// The `(row, column)` position of every slot of one age's structure, so the
/// client can render the pyramid shape without hard-coding it. Straight from
/// `duels_core::layout::layout(age).positions` — already public, static
/// geometry, so no `duels-core` change was needed for this one.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct AgeStructureLayout {
    /// `positions[slot] = (row, column)`, 1-indexed, matching
    /// `duels_core::layout::AgeLayout::positions`.
    pub positions: [(u8, u8); duels_core::layout::SLOTS],
}

/// The military track's static facts, for `GET /catalog`.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct MilitaryCatalog {
    /// Distance from centre at which a player wins outright.
    pub capital_distance: u8,
    /// `(distance, coins forfeited by the losing player)` for the two loot
    /// tokens on one side of the track.
    pub loot: [(u8, u8); 2],
    /// Victory points for the leading player at every distance
    /// `0..capital_distance`, computed via
    /// [`duels_core::data::MilitaryTrack::vp_for_distance`] so this can never
    /// drift from what the engine actually pays out.
    pub victory_points_by_distance: Vec<u8>,
}

/// `GET /catalog` response: everything the UI needs to render a card,
/// wonder or token it isn't currently looking at the full definition of
/// (printed costs, chain links, effect text), without ever computing rules
/// itself.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct Catalog {
    /// Every age card, in id order.
    pub cards: Vec<CardCatalogEntry>,
    /// Every wonder, in id order.
    pub wonders: Vec<WonderCatalogEntry>,
    /// Every progress token, in id order.
    pub tokens: Vec<TokenCatalogEntry>,
    /// The military track.
    pub military: MilitaryCatalog,
    /// Slot geometry for ages I, II and III, indexed by `age - 1`.
    pub layouts: [AgeStructureLayout; 3],
}

/// `{ wood, clay, stone, glass, papyrus }`, named rather than a positional
/// array so the client never has to know `duels_core::data::Resource`'s
/// index order.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct ResourceAmounts {
    /// Units of wood.
    pub wood: u8,
    /// Units of clay.
    pub clay: u8,
    /// Units of stone.
    pub stone: u8,
    /// Units of glass.
    pub glass: u8,
    /// Units of papyrus.
    pub papyrus: u8,
}

impl From<[u8; duels_core::data::NUM_RESOURCES]> for ResourceAmounts {
    fn from(a: [u8; duels_core::data::NUM_RESOURCES]) -> Self {
        Self {
            wood: a[0],
            clay: a[1],
            stone: a[2],
            glass: a[3],
            papyrus: a[4],
        }
    }
}

/// Which group of resources a "produce one of your choice" source stands in
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum ResourceGroupLabel {
    /// Wood, clay or stone.
    RawMaterial,
    /// Glass or papyrus.
    ManufacturedGood,
}

impl From<duels_core::data::ResourceGroup> for ResourceGroupLabel {
    fn from(g: duels_core::data::ResourceGroup) -> Self {
        match g {
            duels_core::data::ResourceGroup::RawMaterial => ResourceGroupLabel::RawMaterial,
            duels_core::data::ResourceGroup::ManufacturedGood => {
                ResourceGroupLabel::ManufacturedGood
            }
        }
    }
}

/// What a progress token's cost rebate applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum DiscountLabel {
    /// Architecture: wonders.
    Wonders,
    /// Masonry: civilian (blue) buildings.
    CivilianBuildings,
}

impl From<duels_core::data::DiscountTarget> for DiscountLabel {
    fn from(t: duels_core::data::DiscountTarget) -> Self {
        match t {
            duels_core::data::DiscountTarget::Wonders => DiscountLabel::Wonders,
            duels_core::data::DiscountTarget::CivilianBuildings => DiscountLabel::CivilianBuildings,
        }
    }
}
