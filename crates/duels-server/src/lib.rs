//! `duels-server`: the server-authoritative game server.
//!
//! A room-based REST + WebSocket server that drives live 7 Wonders Duel
//! games using `duels-core`'s engine as the single source of truth for
//! legality and scoring. Clients — the TypeScript/React web client in
//! `web/`, or a raw WebSocket test harness — never run rules logic
//! themselves; they only render the `Observation`s this server pushes and
//! submit `Action`s from the `legal_actions` list it sends alongside them.
//!
//! See [`protocol`] for the wire contract and [`room`] for the room/seat
//! model and the game loop (apply a human's action, then drive any agent
//! seats to their next human-or-game-over decision point, then broadcast).
//! [`app`] builds the router; `src/main.rs` just binds it to a socket, so
//! integration tests can build the exact same app against an in-process
//! listener.

pub mod catalog;
pub mod protocol;
mod rest;
pub mod room;
mod ws;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;

use room::Rooms;

/// Build the axum [`Router`] for the whole server, backed by a fresh,
/// empty [`Rooms`] table.
pub fn app() -> Router {
    app_with_rooms(Arc::new(Rooms::new()))
}

/// Build the router against an existing [`Rooms`] table, so a test can hold
/// onto it (or a caller can otherwise control room creation ahead of time).
pub fn app_with_rooms(rooms: Arc<Rooms>) -> Router {
    Router::new()
        .route("/catalog", get(rest::get_catalog))
        .route("/agents", get(rest::get_agents))
        .route("/rooms", post(rest::create_room))
        .route("/rooms/:id", get(rest::get_room))
        .route("/rooms/:id/ws", get(ws::room_ws))
        .layer(CorsLayer::permissive())
        .with_state(rooms)
}
