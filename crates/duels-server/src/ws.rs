//! `GET /rooms/:id/ws`: the live connection for a room.
//!
//! On connect, the server immediately sends the current [`StatePayload`].
//! From then on the connection is simultaneously: a receiver of every
//! [`ServerMessage`] broadcast to the room (from this connection's own
//! actions, another connection's, or an agent seat resolving its turn), and
//! a sender of [`ClientMessage::Action`]s, each validated against the room's
//! current legal actions before being applied.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::response::IntoResponse;

use crate::protocol::{ClientMessage, ServerMessage};
use crate::room::Rooms;

pub async fn room_ws(
    State(rooms): State<Arc<Rooms>>,
    Path(id): Path<String>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, rooms, id))
}

async fn handle_socket(mut socket: WebSocket, rooms: Arc<Rooms>, id: String) {
    let Some(room) = rooms.get(&id) else {
        let msg = ServerMessage::Error {
            message: format!("no room {id}"),
        };
        if let Ok(text) = serde_json::to_string(&msg) {
            let _ = socket.send(Message::Text(text)).await;
        }
        return;
    };

    let mut rx = room.subscribe();

    // Send the current snapshot immediately, per the M2 spec, rather than
    // waiting for the next broadcast.
    let initial = ServerMessage::State(Box::new(room.snapshot().await));
    if send(&mut socket, &initial).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            broadcast = rx.recv() => {
                match broadcast {
                    Ok(msg) => {
                        if send(&mut socket, &msg).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        // This connection missed some messages (a very slow
                        // client); catch it up with a fresh snapshot rather
                        // than replaying a partial history.
                        let snap = ServerMessage::State(Box::new(room.snapshot().await));
                        if send(&mut socket, &snap).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Text(text))) => {
                        if handle_client_text(&room, &mut socket, &text).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {} // ignore binary/ping/pong frames
                    Some(Err(_)) => break,
                }
            }
        }
    }
}

async fn handle_client_text(
    room: &Arc<crate::room::Room>,
    socket: &mut WebSocket,
    text: &str,
) -> Result<(), ()> {
    let parsed: Result<ClientMessage, _> = serde_json::from_str(text);
    match parsed {
        Ok(ClientMessage::Action { action }) => {
            if let Err(message) = room.apply_client_action(action).await {
                send(socket, &ServerMessage::Error { message }).await?;
            }
            // On success, the new state is delivered via the broadcast
            // subscription this same loop iteration will pick up next.
            Ok(())
        }
        Err(e) => {
            send(
                socket,
                &ServerMessage::Error {
                    message: format!("could not parse message: {e}"),
                },
            )
            .await
        }
    }
}

async fn send(socket: &mut WebSocket, msg: &ServerMessage) -> Result<(), ()> {
    let text = serde_json::to_string(msg).map_err(|_| ())?;
    socket.send(Message::Text(text)).await.map_err(|_| ())
}
