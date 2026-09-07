//! `POST /rooms`, `GET /rooms/:id`, `GET /rooms/:id/analysis`,
//! `GET /rooms/:id/export`, `GET /catalog` and `GET /agents`.

use std::sync::{Arc, OnceLock};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::protocol::{
    AnalysisPayload, Catalog, CreateRoomRequest, CreateRoomResponse, ExportPayload, RoomInfo,
};
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

/// `GET /rooms/:id/analysis`: `duels-eval`'s read of the room's current
/// position and of every action legal in it, for the web client's advanced
/// mode.
///
/// Not part of the WebSocket protocol on purpose. It is an opt-in developer /
/// analysis tool that a client fetches when it wants one, so putting it in
/// every broadcast would make every ordinary player pay for it — and it is
/// derived, cacheable-by-position data that no client needs in order to play.
pub async fn get_analysis(
    State(rooms): State<Arc<Rooms>>,
    Path(id): Path<String>,
) -> Result<Json<AnalysisPayload>, AppError> {
    let room = rooms
        .get(&id)
        .ok_or_else(|| not_found(format!("no room {id}")))?;
    Ok(Json(room.analysis().await))
}

/// `GET /rooms/:id/export`: the seed and move list that reconstruct this
/// room's exact position, for a position the project owner wants to flag and
/// come back to. See [`crate::room::replay`].
pub async fn get_export(
    State(rooms): State<Arc<Rooms>>,
    Path(id): Path<String>,
) -> Result<Json<ExportPayload>, AppError> {
    let room = rooms
        .get(&id)
        .ok_or_else(|| not_found(format!("no room {id}")))?;
    Ok(Json(room.export().await))
}

static CATALOG: OnceLock<Catalog> = OnceLock::new();

/// `GET /catalog`: static card/wonder/token/military reference data. Built
/// once and cached; nothing in it depends on any room's state.
pub async fn get_catalog() -> Json<&'static Catalog> {
    Json(CATALOG.get_or_init(crate::catalog::build))
}

/// `GET /agents`: every agent name `POST /rooms` will accept for a
/// `SeatSpec::Agent`, in the order the web client's opponent picker should
/// offer them. Backed by `room::KNOWN_AGENTS` so the UI never hand-maintains
/// its own copy that could drift from `room::make_agent`.
pub async fn get_agents() -> Json<&'static [&'static str]> {
    Json(crate::room::KNOWN_AGENTS)
}
