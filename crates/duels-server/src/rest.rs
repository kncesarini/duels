//! `POST /rooms`, `GET /rooms/:id` and `GET /catalog`.

use std::sync::{Arc, OnceLock};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::protocol::{Catalog, CreateRoomRequest, CreateRoomResponse, RoomInfo};
use crate::room::Rooms;

/// A REST error: just a status code and a message.
pub struct AppError(StatusCode, String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

fn bad_request(msg: impl Into<String>) -> AppError {
    AppError(StatusCode::BAD_REQUEST, msg.into())
}

fn not_found(msg: impl Into<String>) -> AppError {
    AppError(StatusCode::NOT_FOUND, msg.into())
}

/// `POST /rooms`: create a room with the requested seats, kick off any
/// leading agent turns (e.g. an agent seat drafting first), and return its
/// id.
pub async fn create_room(
    State(rooms): State<Arc<Rooms>>,
    Json(req): Json<CreateRoomRequest>,
) -> Result<Json<CreateRoomResponse>, AppError> {
    let room = rooms.create(req.seats, req.seed).map_err(bad_request)?;
    room.kick_off().await;
    Ok(Json(CreateRoomResponse {
        room_id: room.id.clone(),
    }))
}

/// `GET /rooms/:id`.
pub async fn get_room(
    State(rooms): State<Arc<Rooms>>,
    Path(id): Path<String>,
) -> Result<Json<RoomInfo>, AppError> {
    let room = rooms
        .get(&id)
        .ok_or_else(|| not_found(format!("no room {id}")))?;
    Ok(Json(room.info().await))
}

static CATALOG: OnceLock<Catalog> = OnceLock::new();

/// `GET /catalog`: static card/wonder/token/military reference data. Built
/// once and cached; nothing in it depends on any room's state.
pub async fn get_catalog() -> Json<&'static Catalog> {
    Json(CATALOG.get_or_init(crate::catalog::build))
}
