//! Room storage and the game loop: applying a human's action, then driving
//! any agent seats to their next human-or-game-over decision point before
//! broadcasting the new state.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use duels_agents_api::{Agent, Budget};
use duels_core::{engine, scoring, Event, GameState};
use rand::{rngs::StdRng, SeedableRng};
use tokio::sync::{broadcast, Mutex as AsyncMutex};

use crate::protocol::{ActionCost, RoomInfo, RoomStatus, SeatSpec, ServerMessage, StatePayload};

/// Monotonic counter backing both room ids and (when the client doesn't
/// supply one) game seeds. An `AtomicU64` rather than `rand::thread_rng` or
/// a wall-clock read, because both are banned workspace-wide (see
/// `clippy.toml`) - a predictable seed is a fine default for a casual game
/// against the random bot, and a caller who cares can always pass one
/// explicitly in `CreateRoomRequest::seed`.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Every agent name this server knows how to construct, in the order the
/// web client's opponent picker should offer them (weakest/cheapest first).
/// `GET /agents` serves this list so the UI never hand-maintains its own
/// copy. Mirrors `duels-arena`'s `agent_registry::KNOWN_AGENTS`.
pub const KNOWN_AGENTS: &[&str] = &["random", "greedy", "greedy-ev", "alphabeta", "mcts-uct"];

/// Construct the `Agent` for an agent seat. Unknown names are rejected when
/// the room is created rather than silently falling back to something.
///
/// Mirrors `duels-arena`'s `agent_registry::make_agent`: add one match arm
/// (and a `KNOWN_AGENTS` entry) per new agent crate as it lands.
pub fn make_agent(name: &str, seed: u64) -> Result<Box<dyn Agent + Send>, String> {
    match name {
        "random" => Ok(Box::new(duels_agent_random::RandomAgent::new(seed))),
        "greedy" => Ok(Box::new(duels_agent_greedy::GreedyAgent::new(seed))),
        "greedy-ev" => Ok(Box::new(duels_agent_greedy_ev::GreedyEvAgent::new(seed))),
        "alphabeta" => Ok(Box::new(duels_agent_alphabeta::AlphaBetaAgent::new(seed))),
        "mcts-uct" => Ok(Box::new(duels_agent_mcts_uct::MctsAgent::new(seed))),
        other => Err(format!(
            "unknown agent \"{other}\" (known agents: {})",
            KNOWN_AGENTS.join(", ")
        )),
    }
}

/// The [`Budget`] an agent seat gets per move in a live, human-facing room.
///
/// `random` and `greedy` ignore whatever `Budget` they are handed (random
/// picks uniformly, greedy is a fixed 1-ply heuristic), so `Nodes(1)` is a
/// fine, instant default for both. `alphabeta` and `mcts-uct` are real
/// anytime searches that get meaningfully stronger with more time (see their
/// crate-level docs: e.g. alphabeta measures 82%/96%/96% win rate against
/// `random` at `Nodes(2_000)`/`Nodes(20_000)`/`TimeMs(200)` respectively) -
/// `TimeMs(1_000)` is chosen here as a "feels responsive but plays well"
/// budget for an interactive game against a human, not the (often larger)
/// budgets `duels-arena` uses to benchmark agents against each other.
fn interactive_budget(name: &str) -> Budget {
    match name {
        "alphabeta" | "mcts-uct" => Budget::TimeMs(1_000),
        _ => Budget::Nodes(1),
    }
}

/// The mutable parts of a room: the authoritative state, its RNG, and the
/// `Agent` instance for each agent seat (kept, not rebuilt, so a stateful
/// future agent could hold onto e.g. a search tree between calls).
struct RoomInner {
    state: GameState,
    rng: StdRng,
    agents: [Option<Box<dyn Agent + Send>>; 2],
    /// Per-seat `Budget` for agent seats, chosen once at room creation by
    /// [`interactive_budget`] (irrelevant, but harmless, for human seats).
    budgets: [Budget; 2],
}

/// One room: two seats playing a single game, plus a broadcast channel every
/// connected WebSocket subscribes to.
pub struct Room {
    pub id: String,
    pub seats: [SeatSpec; 2],
    inner: AsyncMutex<RoomInner>,
    tx: broadcast::Sender<ServerMessage>,
}

impl Room {
    fn new(id: String, seats: [SeatSpec; 2], seed: u64) -> Result<Arc<Self>, String> {
        let state = engine::new_game(seed);
        let rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut agents: [Option<Box<dyn Agent + Send>>; 2] = [None, None];
        let mut budgets = [Budget::Nodes(1), Budget::Nodes(1)];
        for (i, seat) in seats.iter().enumerate() {
            if let SeatSpec::Agent { name } = seat {
                // Give each agent seat its own stream, derived from the game
                // seed, so two agent seats in one room don't play identically.
                let agent_seed = seed ^ (0xD1B5_4A32_D192_ED03u64.wrapping_mul(i as u64 + 1));
                agents[i] = Some(make_agent(name, agent_seed)?);
                budgets[i] = interactive_budget(name);
            }
        }
        let (tx, _rx) = broadcast::channel(64);
        Ok(Arc::new(Self {
            id,
            seats,
            inner: AsyncMutex::new(RoomInner {
                state,
                rng,
                agents,
                budgets,
            }),
            tx,
        }))
    }

    /// Subscribe to this room's broadcast stream (for a new WebSocket
    /// connection).
    pub fn subscribe(&self) -> broadcast::Receiver<ServerMessage> {
        self.tx.subscribe()
    }

    /// The current state, packaged exactly as it would be broadcast, for a
    /// freshly connected client or `GET /rooms/:id`.
    pub async fn snapshot(&self) -> StatePayload {
        let inner = self.inner.lock().await;
        build_payload(&inner.state, &self.seats, Vec::new())
    }

    /// Basic metadata, for `GET /rooms/:id`.
    pub async fn info(&self) -> RoomInfo {
        let inner = self.inner.lock().await;
        RoomInfo {
            room_id: self.id.clone(),
            seats: self.seats.clone(),
            status: if inner.state.is_over() {
                RoomStatus::GameOver
            } else {
                RoomStatus::Playing
            },
            turn: inner.state.turn(),
        }
    }

    /// Apply a client-submitted action for whichever seat is currently on
    /// move, then drive any agent seats that follow, and broadcast the
    /// result. Returns an error message (not applied, nothing broadcast) if
    /// `action` is not currently legal.
    pub async fn apply_client_action(
        self: &Arc<Self>,
        action: duels_core::Action,
    ) -> Result<(), String> {
        let mut inner = self.inner.lock().await;
        let legal = engine::legal_actions(&inner.state);
        if !legal.contains(&action) {
            return Err("that action is not currently legal".to_string());
        }
        let RoomInner { state, rng, .. } = &mut *inner;
        let mut events = engine::apply(state, action, rng).map_err(|e| e.to_string())?;
        events.extend(drive_agents(&mut inner).await);
        let payload = build_payload(&inner.state, &self.seats, events);
        drop(inner);
        let _ = self.tx.send(ServerMessage::State(Box::new(payload)));
        Ok(())
    }

    /// Drive any agent seats up front (e.g. if the wonder draft's first
    /// picker is an agent seat), broadcasting the result. Called once right
    /// after the room is created.
    pub async fn kick_off(self: &Arc<Self>) {
        let mut inner = self.inner.lock().await;
        let events = drive_agents(&mut inner).await;
        let payload = build_payload(&inner.state, &self.seats, events);
        drop(inner);
        let _ = self.tx.send(ServerMessage::State(Box::new(payload)));
    }
}

/// While the game isn't over and the seat on move is an `Agent`, ask it to
/// choose (on a blocking task, per the M2 spec) and apply the result,
/// repeating until either a human seat is on move or the game ends.
async fn drive_agents(inner: &mut RoomInner) -> Vec<Event> {
    let mut events = Vec::new();
    loop {
        if inner.state.is_over() {
            break;
        }
        let legal = engine::legal_actions(&inner.state);
        if legal.is_empty() {
            break;
        }
        let idx = inner.state.current_player().index();
        let Some(mut agent) = inner.agents[idx].take() else {
            // A human seat is on move: give it back (there was nothing to
            // take) and stop, so the client gets a turn.
            break;
        };
        let obs = inner.state.observation();
        let budget = inner.budgets[idx];
        let (agent, action) = tokio::task::spawn_blocking(move || {
            let action = agent.choose(&obs, &legal, budget);
            (agent, action)
        })
        .await
        .expect("agent task panicked");
        inner.agents[idx] = Some(agent);

        match engine::apply(&mut inner.state, action, &mut inner.rng) {
            Ok(ev) => events.extend(ev),
            Err(e) => {
                // The `Agent` contract guarantees a legal return value; this
                // would indicate a bug in the agent, not a client mistake.
                // Stop driving rather than looping forever.
                tracing::error!("agent returned an illegal action: {e}");
                break;
            }
        }
    }
    events
}

/// Costs for the `Build`/`Discard`/`BuildWonder` entries of `legal`, computed
/// from the authoritative `state` so the client never has to.
fn action_costs(state: &GameState, legal: &[duels_core::Action]) -> Vec<ActionCost> {
    use duels_core::{cost, Action};
    let player = state.current_player();
    legal
        .iter()
        .filter_map(|a| match *a {
            Action::Build { slot } => {
                let card = state.face_up_card(slot)?;
                let c = cost::card_cost(state, player, card);
                Some(ActionCost::Build {
                    slot,
                    coins: c.coins,
                    trade: c.trade,
                    via_chain: c.via_chain,
                })
            }
            Action::Discard { slot } => Some(ActionCost::Discard {
                slot,
                reward: cost::discard_reward(state, player),
            }),
            Action::BuildWonder { slot, wonder } => {
                let c = cost::wonder_cost(state, player, wonder);
                Some(ActionCost::BuildWonder {
                    slot,
                    wonder,
                    coins: c.coins,
                    trade: c.trade,
                })
            }
            _ => None,
        })
        .collect()
}

fn build_payload(state: &GameState, seats: &[SeatSpec; 2], events: Vec<Event>) -> StatePayload {
    let legal = engine::legal_actions(state);
    let action_costs = action_costs(state, &legal);
    let breakdown = state.result().map(|_| scoring::score(state));
    StatePayload {
        observation: state.observation(),
        seats: seats.clone(),
        legal_actions: legal,
        action_costs,
        events,
        breakdown,
    }
}

/// In-memory room storage. A `std::sync::Mutex` around the map itself is
/// fine: every operation on it (insert/get/clone an `Arc`) is O(1) and never
/// awaits.
#[derive(Default)]
pub struct Rooms(StdMutex<HashMap<String, Arc<Room>>>);

impl Rooms {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a room with the given seats, returning it already inserted
    /// into the table. Does *not* drive agent seats yet; call
    /// [`Room::kick_off`] once the caller is ready to broadcast.
    pub fn create(&self, seats: [SeatSpec; 2], seed: Option<u64>) -> Result<Arc<Room>, String> {
        let seed = seed.unwrap_or_else(next_id);
        let id = format!("room-{}", next_id());
        let room = Room::new(id.clone(), seats, seed)?;
        self.0.lock().unwrap().insert(id, room.clone());
        Ok(room)
    }

    pub fn get(&self, id: &str) -> Option<Arc<Room>> {
        self.0.lock().unwrap().get(id).cloned()
    }
}
