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

use crate::protocol::{
    ActionAnalysis, ActionCost, AnalysisPayload, CostLine, CostPlan, ExportPayload, PlayerView,
    RoomInfo, RoomStatus, SeatSpec, ServerMessage, SlotCostView, StatePayload, StepPayload,
    WonderCostView,
};

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
///
/// `random`, `greedy` and `greedy-ev` were retired from the roster (see
/// `docs/milestones.md`), which is why the easy end of the picker now starts
/// at `phased`. `duels-agent-random` still exists as a test fixture but is
/// deliberately not linked by this crate at all.
///
/// `mcts-value` is last because this list is ordered weakest-first for the
/// picker and it is the strongest thing here: it is on
/// `duels_arena::leaderboard::LADDER` and is the current
/// `duels_arena::leaderboard::CHAMPION`. Its margin comes largely from
/// countering `mcts-eval`'s science-value miscalibration rather than from
/// uniformly better play, so it is a harder opponent than `mcts-eval` without
/// being a strictly better one — read that constant's docs before treating
/// this order as a clean difficulty ramp at the top end.
pub const KNOWN_AGENTS: &[&str] = &["phased", "alphabeta", "mcts-uct", "mcts-eval", "mcts-value"];

/// Construct the `Agent` for an agent seat. Unknown names are rejected when
/// the room is created rather than silently falling back to something.
///
/// Mirrors `duels-arena`'s `agent_registry::make_agent`: add one match arm
/// (and a `KNOWN_AGENTS` entry) per new agent crate as it lands.
pub fn make_agent(name: &str, seed: u64) -> Result<Box<dyn Agent + Send>, String> {
    match name {
        "phased" => Ok(Box::new(duels_agent_phased::PhasedAgent::new(seed))),
        "alphabeta" => Ok(Box::new(duels_agent_alphabeta::AlphaBetaAgent::new(seed))),
        "mcts-uct" => Ok(Box::new(duels_agent_mcts_uct::MctsAgent::new(seed))),
        "mcts-eval" => Ok(Box::new(duels_agent_mcts_eval::MctsEvalAgent::new(seed))),
        "mcts-value" => Ok(Box::new(duels_agent_mcts_value::MctsValueAgent::new(seed))),
        other => Err(format!(
            "unknown agent \"{other}\" (known agents: {})",
            KNOWN_AGENTS.join(", ")
        )),
    }
}

/// The [`Budget`] an agent seat gets per move in a live, human-facing room.
///
/// `phased` ignores whatever `Budget` it is handed — it is a fixed 1-ply
/// evaluation — so `Nodes(1)` is a fine, instant default for it.
/// `alphabeta`, `mcts-uct`, `mcts-eval` and `mcts-value` are real anytime
/// searches that get meaningfully stronger with more time (see their
/// crate-level docs: e.g. alphabeta measures 82%/96%/96% win rate against the
/// uniform-random floor at `Nodes(2_000)`/`Nodes(20_000)`/`TimeMs(200)`
/// respectively) - `TimeMs(1_000)` is chosen here as a "feels responsive but
/// plays well" budget for an interactive game against a human, not the (often
/// larger) budgets `duels-arena` uses to benchmark agents against each other.
///
/// `mcts-eval` is the one to hand a wall-clock budget to with most
/// confidence: its leaf value has no measurable throughput cost and its
/// advantage over a plain playout is *larger* at `TimeMs(20)` and
/// `TimeMs(100)` than at `Nodes(2000)`, because a better leaf value is worth
/// more when there are fewer leaves to average over.
/// `mcts-value` gets the same wall-clock budget for the same reason and then
/// some: its leaf value replaces the playout rather than adding to it, which
/// costs about a third of a playout per simulation, so a fixed clock buys it
/// roughly three times the simulations. That is also the budget kind its
/// largest measured margin over `mcts-eval` was taken at.
fn interactive_budget(name: &str) -> Budget {
    match name {
        "alphabeta" | "mcts-uct" | "mcts-eval" | "mcts-value" => Budget::TimeMs(1_000),
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
    /// Every action applied to this room so far, in order. Replayed in full
    /// to a freshly connected client so a reload restores the game log and
    /// the position history rather than starting them empty. Bounded by the
    /// length of a game (a little over sixty actions), so it never grows
    /// without limit.
    history: Vec<StepPayload>,
}

/// The RNG a room applies its moves against, derived from the game seed.
///
/// Kept as a named function rather than an inline expression in [`Room::new`]
/// because [`replay`] has to derive the *identical* stream to reconstruct a
/// room's position from an [`ExportPayload`]: `engine::apply` consumes
/// randomness (only The Great Library does, per its docs, but "only rarely"
/// is not "never"), so a replay that seeded a different stream would diverge
/// on exactly the games that are most interesting to analyse.
fn game_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15)
}

/// Reconstruct the `GameState` an [`ExportPayload`] describes: deal `seed`'s
/// game and apply `moves` in order.
///
/// This is the whole point of the export format — a flagged position has to be
/// reproducible later, offline, with no room and no server still alive. The
/// result is the room's complete state, hidden layout included, not just its
/// public `Observation`;
/// `tests::an_export_replays_back_to_the_rooms_exact_state` asserts that
/// equality directly.
///
/// Fails if a move is not legal in the position it is replayed into, which
/// would mean the export did not come from this build of the engine.
pub fn replay(seed: u64, moves: &[duels_core::Action]) -> Result<GameState, String> {
    let mut state = engine::new_game(seed);
    let mut rng = game_rng(seed);
    for (i, &action) in moves.iter().enumerate() {
        engine::apply(&mut state, action, &mut rng)
            .map_err(|e| format!("move {i} ({action:?}) does not replay: {e}"))?;
    }
    Ok(state)
}

/// Price `state` and every action legal in it with [`duels_eval`], from the
/// point of view of the seat on move.
///
/// A free function taking the state, rather than a method on [`Room`], so a
/// test can drive it with a `StateBuilder` position — the endpoint's job is to
/// describe a *position*, and being able to hand it a hand-built decisive one
/// is the only way to check its numbers say what they should.
fn analyse(room_id: &str, state: &GameState) -> AnalysisPayload {
    let me = state.current_player();
    // One `Root` for the whole position, exactly as `PhasedAgent::choose`
    // builds one per decision: the commitment blend and the opponent-menu
    // tables are root-fixed by design, and rebuilding them per action would
    // both cost more and price the actions against different weights.
    let root = duels_eval::Root::new(state, me, duels_eval::Config::default());
    let value = duels_eval::evaluate(state, me, &root);
    let actions = engine::legal_actions(state)
        .into_iter()
        .map(|action| {
            let value = duels_eval::expected_value(state, action, me, &root);
            ActionAnalysis {
                action,
                value,
                // `win_probability_from_value`, not `win_probability`: an
                // action that resolves a chance node has many possible
                // resulting states and `expected_value` has already averaged
                // over them, so there is no single post-action `GameState` to
                // evaluate. The age is the one the decision is *made* in.
                win_probability: duels_eval::win_probability_from_value(value, state.age()),
            }
        })
        .collect();
    AnalysisPayload {
        room_id: room_id.to_string(),
        turn: state.turn(),
        age: state.age(),
        current_player: me,
        game_over: state.is_over(),
        value,
        win_probability: duels_eval::win_probability_from_value(value, state.age()),
        actions,
        eval_generation: duels_eval::Config::default().params_string(),
    }
}

/// One room: two seats playing a single game, plus a broadcast channel every
/// connected WebSocket subscribes to.
pub struct Room {
    pub id: String,
    pub seats: [SeatSpec; 2],
    /// The seed this room's game was dealt from. Kept so
    /// [`Room::export`] can hand out a `{ seed, moves }` bundle that
    /// reconstructs the position exactly.
    pub seed: u64,
    inner: AsyncMutex<RoomInner>,
    tx: broadcast::Sender<ServerMessage>,
}

impl Room {
    fn new(id: String, seats: [SeatSpec; 2], seed: u64) -> Result<Arc<Self>, String> {
        let state = engine::new_game(seed);
        let rng = game_rng(seed);
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
            seed,
            inner: AsyncMutex::new(RoomInner {
                state,
                rng,
                agents,
                budgets,
                history: Vec::new(),
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
        let mut payload = build_payload(&inner.state, &self.seats, inner.history.clone());
        payload.replay = true;
        payload
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

    /// This room's authoritative `GameState`, copied out.
    ///
    /// `GameState` is `Copy` and holds hidden information, so this is
    /// deliberately not part of any wire type — it exists for server-side
    /// analysis (see [`Room::analysis`]) and for tests that need to compare a
    /// replayed position against the real one.
    pub async fn state(&self) -> GameState {
        self.inner.lock().await.state
    }

    /// What `duels-eval` makes of this room's current position, and of every
    /// action available in it.
    ///
    /// Structurally the same work `duels-agent-phased` does to pick a move
    /// (`PhasedAgent::choose`): build **one** `Root` for the position, then
    /// score every legal action against it. Two differences, both because
    /// this is for display rather than for playing:
    ///
    /// - the real `GameState` is evaluated instead of a sampled
    ///   determinization of the `Observation`, which the server has and an
    ///   agent does not, and which changes nothing (see
    ///   [`AnalysisPayload`]'s note on why that is safe);
    /// - the scores are also mapped onto win probabilities, and every one of
    ///   them is returned rather than just the argmax.
    pub async fn analysis(&self) -> AnalysisPayload {
        analyse(&self.id, &self.state().await)
    }

    /// The seed and full move list this room's current position is built from.
    /// See [`replay`], which turns one back into a `GameState`.
    pub async fn export(&self) -> ExportPayload {
        let inner = self.inner.lock().await;
        ExportPayload {
            room_id: self.id.clone(),
            seed: self.seed,
            // Every `StepPayload` this server records carries `Some(action)`
            // (`step` is only ever called with one); `flatten` rather than
            // `expect` so a future engine-initiated step with no action of its
            // own could be added without this silently panicking on it.
            moves: inner.history.iter().filter_map(|s| s.action).collect(),
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
        let actor = state.current_player();
        let events = engine::apply(state, action, rng).map_err(|e| e.to_string())?;
        let mut steps = vec![step(&inner.state, actor, action, events)];
        steps.extend(drive_agents(&mut inner).await);
        inner.history.extend(steps.iter().cloned());
        let payload = build_payload(&inner.state, &self.seats, steps);
        drop(inner);
        let _ = self.tx.send(ServerMessage::State(Box::new(payload)));
        Ok(())
    }

    /// Drive any agent seats up front (e.g. if the wonder draft's first
    /// picker is an agent seat), broadcasting the result. Called once right
    /// after the room is created.
    pub async fn kick_off(self: &Arc<Self>) {
        let mut inner = self.inner.lock().await;
        let steps = drive_agents(&mut inner).await;
        inner.history.extend(steps.iter().cloned());
        let payload = build_payload(&inner.state, &self.seats, steps);
        drop(inner);
        let _ = self.tx.send(ServerMessage::State(Box::new(payload)));
    }
}

/// While the game isn't over and the seat on move is an `Agent`, ask it to
/// choose (on a blocking task, per the M2 spec) and apply the result,
/// repeating until either a human seat is on move or the game ends.
async fn drive_agents(inner: &mut RoomInner) -> Vec<StepPayload> {
    let mut steps = Vec::new();
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

        let actor = inner.state.current_player();
        match engine::apply(&mut inner.state, action, &mut inner.rng) {
            Ok(ev) => steps.push(step(&inner.state, actor, action, ev)),
            Err(e) => {
                // The `Agent` contract guarantees a legal return value; this
                // would indicate a bug in the agent, not a client mistake.
                // Stop driving rather than looping forever.
                tracing::error!("agent returned an illegal action: {e}");
                break;
            }
        }
    }
    steps
}

/// Package one applied action, the events it produced and the position it
/// left behind.
fn step(
    state: &GameState,
    actor: duels_core::Player,
    action: duels_core::Action,
    events: Vec<Event>,
) -> StepPayload {
    StepPayload {
        actor: Some(actor),
        action: Some(action),
        events,
        observation: state.observation(),
        views: player_views(state),
        accessible_slots: state.observation().accessible_slots(),
    }
}

/// Both players' derived views of `state`. See [`PlayerView`].
fn player_views(state: &GameState) -> [PlayerView; 2] {
    [
        player_view(state, duels_core::Player::One),
        player_view(state, duels_core::Player::Two),
    ]
}

fn player_view(state: &GameState, player: duels_core::Player) -> PlayerView {
    use duels_core::cost;
    let me = state.player(player);
    let slot_costs = (0..duels_core::layout::SLOTS as u8)
        .filter_map(|slot| {
            let card = state.face_up_card(slot)?;
            Some(SlotCostView {
                slot,
                card,
                plan: cost_plan(cost::card_payment_plan(state, player, card), me.coins()),
            })
        })
        .collect();
    let wonder_costs = me
        .wonders()
        .filter(|w| !me.has_built_wonder(*w))
        .map(|wonder| WonderCostView {
            wonder,
            plan: cost_plan(cost::wonder_payment_plan(state, player, wonder), me.coins()),
        })
        .collect();
    PlayerView {
        production: me.production().into(),
        trade_prices: cost::trade_prices(state, player).into(),
        distinct_science: me.distinct_science(),
        vp_now: scoring::breakdown(state, player),
        discard_reward: cost::discard_reward(state, player),
        slot_costs,
        wonder_costs,
    }
}

/// Flatten a `duels_core::cost::PaymentPlan` onto the wire, dropping the
/// resources the printed cost does not ask for.
fn cost_plan(plan: duels_core::cost::PaymentPlan, coins: u16) -> CostPlan {
    use duels_core::data::Resource;
    CostPlan {
        lines: plan
            .lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.required > 0)
            .map(|(i, l)| CostLine {
                resource: Resource::ALL[i],
                required: l.required,
                produced: l.produced,
                from_choice: l.from_choice,
                from_discount: l.from_discount,
                bought: l.bought,
                unit_price: l.unit_price,
            })
            .collect(),
        coin_cost: plan.coin_cost,
        coins: plan.cost.coins,
        trade: plan.cost.trade,
        via_chain: plan.cost.via_chain,
        affordable: coins >= plan.cost.coins,
    }
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

fn build_payload(
    state: &GameState,
    seats: &[SeatSpec; 2],
    steps: Vec<StepPayload>,
) -> StatePayload {
    let legal = engine::legal_actions(state);
    let action_costs = action_costs(state, &legal);
    let breakdown = state.result().map(|_| scoring::score(state));
    StatePayload {
        observation: state.observation(),
        views: player_views(state),
        accessible_slots: state.observation().accessible_slots(),
        seats: seats.clone(),
        legal_actions: legal,
        action_costs,
        steps,
        replay: false,
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

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::Action;

    /// Drive a real room several moves in, playing the human seat with a
    /// deterministic "take the last legal action" policy (last, not first, so
    /// the playout is not the same one `tests/integration.rs` drives and the
    /// two do not share a blind spot). Returns the room.
    async fn played_room(seed: u64, human_decisions: usize) -> Arc<Room> {
        let rooms = Rooms::new();
        let room = rooms
            .create(
                [
                    SeatSpec::Human,
                    SeatSpec::Agent {
                        name: "phased".to_string(),
                    },
                ],
                Some(seed),
            )
            .expect("create room");
        room.kick_off().await;
        for _ in 0..human_decisions {
            let legal = engine::legal_actions(&room.state().await);
            let Some(&action) = legal.last() else { break };
            room.apply_client_action(action)
                .await
                .expect("scripted action was legal");
        }
        room
    }

    /// The load-bearing property of the export format: what it hands out has
    /// to reconstruct the position it came from *exactly*.
    ///
    /// `GameState`, not `Observation`: the whole point of exporting a flagged
    /// position is that a later analysis can evaluate it, and every consumer
    /// of `duels-eval` starts from a concrete state. An export that agreed on
    /// public information but dealt a different hidden layout would replay
    /// into a position with different chance outcomes ahead of it and would be
    /// worthless for the intended workflow.
    #[tokio::test]
    async fn an_export_replays_back_to_the_rooms_exact_state() {
        for seed in [1_u64, 7, 20260907, u64::MAX / 3] {
            let room = played_room(seed, 12).await;
            let export = room.export().await;
            assert_eq!(export.seed, seed);
            assert!(
                export.moves.len() >= 12,
                "seed {seed}: expected at least the human's moves, got {}",
                export.moves.len()
            );

            let replayed = replay(export.seed, &export.moves).expect("export replays");
            assert_eq!(
                replayed,
                room.state().await,
                "seed {seed}: replaying the export did not reproduce the room's state"
            );
        }
    }

    /// The export is a *complete* history, not just the moves this browser
    /// made: an agent seat's choices are in it too, in order, interleaved
    /// where they actually happened.
    #[tokio::test]
    async fn the_export_carries_both_seats_moves_in_order() {
        let room = played_room(31, 10).await;
        let export = room.export().await;
        let inner = room.inner.lock().await;
        let actors: Vec<_> = inner.history.iter().filter_map(|s| s.actor).collect();
        assert_eq!(actors.len(), export.moves.len());
        assert!(
            actors.contains(&duels_core::Player::One) && actors.contains(&duels_core::Player::Two),
            "test setup: expected moves from both seats, got {actors:?}"
        );
    }

    /// A replay that is handed a move the position does not allow says so
    /// rather than quietly producing some other position.
    #[tokio::test]
    async fn replaying_an_impossible_move_is_an_error() {
        let room = played_room(5, 4).await;
        let mut export = room.export().await;
        let legal = engine::legal_actions(&room.state().await);
        let impossible = (0..duels_core::layout::SLOTS as u8)
            .map(|slot| Action::Build { slot })
            .find(|a| !legal.contains(a))
            .expect("some slot is not buildable right now");
        export.moves.push(impossible);
        let err = replay(export.seed, &export.moves).expect_err("should not replay");
        assert!(err.contains("does not replay"), "unhelpful error: {err}");
    }

    /// The analysis is a well-formed read of the position: one entry per legal
    /// action, in the same order the client is offered them, every probability
    /// in range, and the evaluation generation recorded.
    #[tokio::test]
    async fn the_analysis_prices_every_legal_action_in_order() {
        let room = played_room(11, 14).await;
        let state = room.state().await;
        let legal = engine::legal_actions(&state);
        assert!(legal.len() > 1, "test setup: need a real choice");

        let analysis = room.analysis().await;
        assert_eq!(analysis.room_id, room.id);
        assert_eq!(analysis.turn, state.turn());
        assert_eq!(analysis.age, state.age());
        assert_eq!(analysis.current_player, state.current_player());
        assert!(!analysis.game_over);
        assert_eq!(
            analysis
                .actions
                .iter()
                .map(|a| a.action)
                .collect::<Vec<_>>(),
            legal,
            "the analysis must line up with `legal_actions` position by position"
        );
        assert!((0.0..=1.0).contains(&analysis.win_probability));
        for a in &analysis.actions {
            assert!(
                (0.0..=1.0).contains(&a.win_probability),
                "{:?} mapped outside [0, 1]: {}",
                a.action,
                a.win_probability
            );
        }
        assert_eq!(
            analysis.eval_generation,
            duels_eval::Config::default().params_string()
        );
    }

    /// The numbers have to mean something, not merely be in range: on a
    /// position where one action wins the game outright, that action must read
    /// as a certainty and the alternatives must not.
    ///
    /// Player One is two spaces from Player Two's capital with `circus` (two
    /// shields) sitting in an open slot and the coins to build it, so `Build`
    /// on that slot ends the game by military supremacy immediately. This is
    /// the sanity check that the endpoint is showing the *mover's* side of the
    /// evaluation and not, say, a sign-flipped one — a bug that no in-range
    /// assertion would catch.
    #[test]
    fn an_action_that_wins_outright_reads_as_a_certainty() {
        use duels_core::testing::StateBuilder;
        use duels_core::Player;

        let state = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "palace")])
            .conflict(7)
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build();

        let analysis = analyse("room-test", &state);
        assert_eq!(analysis.current_player, Player::One);
        let winning = analysis
            .actions
            .iter()
            .find(|a| a.action == Action::Build { slot: 18 })
            .expect("building the closing card is legal");
        assert!(
            winning.win_probability > 0.999,
            "an outright win should read as one, not {:.4}",
            winning.win_probability
        );

        // And discarding it instead — handing the same card to the opponent's
        // next turn — must not read the same way.
        let discard = analysis
            .actions
            .iter()
            .find(|a| a.action == Action::Discard { slot: 18 })
            .expect("discarding it is also legal");
        assert!(
            discard.win_probability < winning.win_probability,
            "throwing the win away scored {:.4}, taking it scored {:.4}",
            discard.win_probability,
            winning.win_probability
        );
    }

    /// The mirror image of the test above, on the identical position with the
    /// other seat to move: the same evaluation, read from the side that is
    /// *about to lose*, must be a near-certain loss. Together the two pin the
    /// orientation of every number this endpoint reports.
    #[test]
    fn the_same_position_read_from_the_losing_side_is_a_near_certain_loss() {
        use duels_core::testing::StateBuilder;
        use duels_core::Player;

        let state = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "circus"), (19, "fortifications")])
            .conflict(7)
            .coins(Player::One, 40)
            .coins(Player::Two, 7)
            .current(Player::Two)
            .build();

        let analysis = analyse("room-test", &state);
        assert_eq!(analysis.current_player, Player::Two);
        assert!(
            analysis.win_probability < 0.001,
            "a rail-owned loss should read as one, not {:.4}",
            analysis.win_probability
        );
    }

    /// The win probability is genuinely the calibrated mapping of the value,
    /// not a second, independently-invented curve — the reason
    /// `duels-eval` owns `win_probability_from_value` at all.
    #[tokio::test]
    async fn the_reported_probabilities_are_the_calibrated_mapping_of_the_values() {
        let room = played_room(23, 9).await;
        let analysis = room.analysis().await;
        let age = analysis.age;
        assert_eq!(
            analysis.win_probability.to_bits(),
            duels_eval::win_probability_from_value(analysis.value, age).to_bits()
        );
        for a in &analysis.actions {
            assert_eq!(
                a.win_probability.to_bits(),
                duels_eval::win_probability_from_value(a.value, age).to_bits(),
                "{:?}",
                a.action
            );
        }
    }

    /// Analysing a finished game reports the result rather than failing: the
    /// advanced-mode client keeps polling after the last move.
    #[tokio::test]
    async fn a_finished_game_still_analyses() {
        let room = played_room(20260907, 200).await;
        let state = room.state().await;
        assert!(state.is_over(), "test setup: game should have finished");

        let analysis = room.analysis().await;
        assert!(analysis.game_over);
        assert!(analysis.actions.is_empty());
        // A decided game is scored by the terminal rail, which is far outside
        // the range any ordinary position reaches, so the logistic saturates.
        assert!(
            analysis.win_probability > 0.99 || analysis.win_probability < 0.01,
            "a finished game should not read as a close position: {}",
            analysis.win_probability
        );

        // And the export of a finished game still replays: this is exactly the
        // case the project owner will flag most often ("that line lost, why
        // did the evaluation like it?").
        let export = room.export().await;
        assert_eq!(replay(export.seed, &export.moves).expect("replays"), state);
    }

    /// How long one analysis takes on a real mid-game position. Ignored by
    /// default (a timing measurement is not a correctness assertion, and
    /// `docs/conventions.md` keeps benchmark-shaped runs out of the default
    /// `cargo test` path); run with `cargo test -p duels-server --release --
    /// --ignored --nocapture analysis_cost`.
    #[tokio::test]
    #[ignore = "timing measurement, not an assertion"]
    async fn analysis_cost_on_a_real_position() {
        // `duels-server` is one of the crates `clippy.toml` explicitly exempts
        // from the wall-clock ban, and a "how long does this take" measurement
        // is the reason that exemption exists.
        #[allow(clippy::disallowed_methods)]
        let now = std::time::Instant::now;

        for decisions in [6usize, 20, 30] {
            let room = played_room(77, decisions).await;
            let state = room.state().await;
            if state.is_over() {
                continue;
            }
            let legal = engine::legal_actions(&state).len();
            // One warm-up, then a timed batch, so the first call's lazy
            // static-data initialisation is not attributed to the endpoint.
            let _ = room.analysis().await;
            let start = now();
            const REPS: u32 = 100;
            for _ in 0..REPS {
                std::hint::black_box(room.analysis().await);
            }
            let each = start.elapsed() / REPS;
            println!(
                "age {} turn {} ({legal} legal actions): {each:?} per analysis",
                state.age(),
                state.turn()
            );
        }
    }
}
