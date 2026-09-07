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

/// Per-unit trade prices, `{ wood, clay, stone, glass, papyrus }`, straight
/// from [`duels_core::cost::trade_prices`]: what this player pays the bank
/// (or, under the Economy token, the opponent) for one unit of each resource
/// right now.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct ResourcePrices {
    /// Coins for one wood.
    pub wood: u16,
    /// Coins for one clay.
    pub clay: u16,
    /// Coins for one stone.
    pub stone: u16,
    /// Coins for one glass.
    pub glass: u16,
    /// Coins for one papyrus.
    pub papyrus: u16,
}

impl From<[u16; duels_core::data::NUM_RESOURCES]> for ResourcePrices {
    fn from(a: [u16; duels_core::data::NUM_RESOURCES]) -> Self {
        Self {
            wood: a[0],
            clay: a[1],
            stone: a[2],
            glass: a[3],
            papyrus: a[4],
        }
    }
}

/// How one resource of a printed cost gets paid for, from
/// [`duels_core::cost::ResourceLine`]. Only resources the cost actually
/// demands appear in a [`CostPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct CostLine {
    /// Which resource.
    pub resource: duels_core::data::Resource,
    /// Units the printed cost demands.
    pub required: u8,
    /// Units the player's own production covers.
    pub produced: u8,
    /// Units a "produce one of your choice" source covers.
    pub from_choice: u8,
    /// Units an Architecture / Masonry rebate covers.
    pub from_discount: u8,
    /// Units that must be bought.
    pub bought: u8,
    /// Coins per bought unit.
    pub unit_price: u16,
}

/// What building one specific thing would cost one specific player, itemised.
///
/// Every field comes from [`duels_core::cost::PaymentPlan`], so the client can
/// render "1 papyrus, bought at 3¢" and the net total without ever computing
/// a price itself — and, crucially, can render the same card's cost from
/// *either* player's point of view (the "cost lens" of the UI spec), which
/// [`ActionCost`] cannot express because it only covers the player on move.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct CostPlan {
    /// One entry per resource the printed cost demands, in
    /// `duels_core::data::Resource::ALL` order.
    pub lines: Vec<CostLine>,
    /// The printed coin cost, owed on top of any trade.
    pub coin_cost: u16,
    /// Total coins owed.
    pub coins: u16,
    /// The portion of `coins` that is a trade payment.
    pub trade: u16,
    /// Whether a chain symbol makes this free.
    pub via_chain: bool,
    /// Whether this player's treasury covers `coins` right now. (Affordability
    /// alone does not make an action legal — only `legal_actions` does; this
    /// exists so an unaffordable card can be shown as such to *either* player.)
    pub affordable: bool,
}

/// What one face-up structure slot would cost one player.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct SlotCostView {
    /// The slot.
    pub slot: u8,
    /// The card visible there.
    pub card: CardId,
    /// The itemised cost for this view's player.
    pub plan: CostPlan,
}

/// What one of a player's drafted, unbuilt wonders would cost them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct WonderCostView {
    /// The wonder.
    pub wonder: WonderId,
    /// The itemised cost for this view's player.
    pub plan: CostPlan,
}

/// Everything derived from the rules that a UI wants to show *per player*,
/// computed server-side for both seats.
///
/// The [`Observation`] carries the raw public state (coins, built cards,
/// science counts); this carries the numbers you would otherwise have to
/// re-derive with rules knowledge — current production including wonder and
/// "choice" sources, the trade prices each player faces, the running victory
/// point total, and the itemised cost of every buildable thing from *this*
/// player's point of view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct PlayerView {
    /// Unconditional production, per resource.
    pub production: ResourceAmounts,
    /// What one unit of each resource costs this player to buy right now.
    pub trade_prices: ResourcePrices,
    /// How many distinct scientific symbols this player holds.
    pub distinct_science: u8,
    /// The victory points this player would score if the game ended now,
    /// from `duels_core::scoring::breakdown`.
    pub vp_now: Breakdown,
    /// Coins this player would get for discarding a card right now.
    pub discard_reward: u16,
    /// The itemised cost of every face-up slot, for this player.
    pub slot_costs: Vec<SlotCostView>,
    /// The itemised cost of every drafted, unbuilt wonder, for this player.
    pub wonder_costs: Vec<WonderCostView>,
}

/// One applied action, with everything it changed and the position it left
/// behind.
///
/// A single [`StatePayload`] can carry several of these (a human move
/// followed by however many agent moves the server resolved before handing
/// control back), and a client that wants to *play the moves back* rather
/// than snap to the final state needs them separated: which events belong to
/// which action, who acted, and what the board looked like after each one.
/// Keeping the per-step [`Observation`] also lets the client render an
/// earlier position for review without asking the server for it again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct StepPayload {
    /// Who acted, or `None` for a step the engine took on its own.
    pub actor: Option<duels_core::Player>,
    /// The action applied.
    pub action: Option<Action>,
    /// Everything that happened as a result, in rules order.
    pub events: Vec<Event>,
    /// The public state immediately after this step.
    pub observation: Observation,
    /// Both players' derived views immediately after this step.
    pub views: [PlayerView; 2],
    /// The slots that could be taken in this position (see
    /// [`StatePayload::accessible_slots`]).
    pub accessible_slots: Vec<u8>,
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
    /// Both players' derived views of the current position (production, trade
    /// prices, running VP, itemised costs), so the client can render either
    /// player's costs without owning a cost engine.
    pub views: [PlayerView; 2],
    /// Which structure slots are uncovered, from
    /// [`duels_core::observation::Observation::accessible_slots`]. A face-up
    /// slot that is not in this list is covered: still readable, not takeable.
    /// Sent rather than derived, because "what covers what" is a rule
    /// (`docs/rules-spec.md` R-010), not a drawing detail — and it has to be
    /// right even when it is not this browser's turn and `legal_actions` is
    /// therefore empty.
    pub accessible_slots: Vec<u8>,
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
    /// What happened since the previous [`StatePayload`], one entry per
    /// applied action, in order. Empty only when nothing has happened yet.
    /// The client animates these in sequence and turns them into log entries.
    pub steps: Vec<StepPayload>,
    /// True when `steps` is the room's *entire* history rather than what just
    /// happened — which is what a freshly connected (or reconnected, or
    /// reloaded) client is sent, so it can populate its log and its
    /// position-review history without replaying any animation.
    pub replay: bool,
    /// The full victory-point breakdown, present once
    /// `observation.result` is `Some`.
    pub breakdown: Option<[Breakdown; 2]>,
}

/// One legal action, priced by [`duels_eval`], for `GET /rooms/:id/analysis`.
///
/// `value` is `duels_eval::expected_value` — a victory-point-scale number,
/// chance-averaged over every way the action's randomness could resolve —
/// and `win_probability` is that same number put through
/// `duels_eval::win_probability_from_value` at the current age's calibrated
/// temperature. The pair is deliberately both: the probability is what a
/// human reads at a glance, the raw value is what a later analysis of an
/// exported position wants to reason about, since the probability mapping is
/// monotone and therefore throws away scale near the tails.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct ActionAnalysis {
    /// The action, exactly as it appears in [`StatePayload::legal_actions`].
    pub action: Action,
    /// `duels_eval::expected_value` for the player on move.
    pub value: f64,
    /// `value` mapped onto `[0, 1]` through the age-calibrated logistic.
    pub win_probability: f64,
}

/// `GET /rooms/:id/analysis`: what `duels-eval` thinks of this room's current
/// position, and of every action available in it.
///
/// Computed from the room's **real** `GameState` rather than a sampled
/// determinization of its `Observation`. That is safe, and it is why this
/// endpoint can exist at all: `duels_eval::evaluate` and
/// `duels_eval::expected_value` are provably invariant to which
/// hidden-information sample produced the state they are handed (a
/// non-negotiable invariant of this workspace, asserted bit-for-bit in
/// `duels-eval/tests/determinization_invariance.rs`), so nothing computed
/// here can depend on a face-down identity, and nothing it returns can tell a
/// player anything that public information does not already imply.
///
/// Everything is reported from the point of view of
/// [`AnalysisPayload::current_player`], the seat on move — the same
/// convention `duels-agent-phased` uses when it scores a decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct AnalysisPayload {
    /// The room this is an analysis of.
    pub room_id: String,
    /// `GameState::turn` when it was computed, so a client can tell a stale
    /// analysis from a current one.
    pub turn: u32,
    /// The age the position is in (1, 2 or 3), which selects the calibrated
    /// temperature the win probabilities were mapped through.
    pub age: u8,
    /// The seat on move, whose side every number here is from.
    pub current_player: duels_core::Player,
    /// True once the game has a result, in which case `actions` is empty and
    /// `win_probability` reflects the terminal rail rather than a judgement.
    pub game_over: bool,
    /// `duels_eval::evaluate` for the current position, in victory points.
    pub value: f64,
    /// `duels_eval::win_probability` for the current position.
    pub win_probability: f64,
    /// Every legal action, in [`StatePayload::legal_actions`] order, priced.
    pub actions: Vec<ActionAnalysis>,
    /// `duels_eval::Config::default().params_string()`: the full parameter
    /// encoding of the evaluation generation these numbers came from, so an
    /// exported position is self-describing and a later round of tuning can
    /// never be confused for the one that was actually flagged.
    pub eval_generation: String,
}

/// `GET /rooms/:id/export`: everything needed to reconstruct this room's exact
/// current position later, with no server, no room and no WebSocket.
///
/// [`crate::room::replay`] is the canonical reconstruction:
/// `duels_core::engine::new_game(seed)` followed by every entry of `moves` in
/// order, against an RNG derived from `seed` the same way the room derived
/// its own. `room::tests::an_export_replays_back_to_the_rooms_exact_state`
/// asserts that this reproduces the room's whole `GameState`, hidden layout
/// included — not merely its public `Observation`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct ExportPayload {
    /// The room this came from.
    pub room_id: String,
    /// The seed the room's game was dealt from.
    pub seed: u64,
    /// Every action applied to the room so far, in order — both seats' moves,
    /// including any the server's agent seats chose.
    pub moves: Vec<Action>,
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
    /// The scientific symbols in the order `PublicPlayer::science` counts
    /// them, from `duels_core::data::Science::ALL`. Sent rather than
    /// restated client-side, so a display can never line its symbols up
    /// against the wrong counts.
    pub science_order: Vec<duels_core::data::Science>,
}

/// `{ wood, clay, stone, glass, papyrus }`, named rather than a positional
/// array so the client never has to know `duels_core::data::Resource`'s
/// index order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, TS)]
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
