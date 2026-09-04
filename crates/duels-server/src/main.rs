//! `duels-server`: the server-authoritative game server.
//!
//! A later milestone will drive live 7 Wonders Duel games over WebSocket
//! using `duels-core`'s engine as the single source of truth for legality
//! and scoring (clients, including the TypeScript/React web client, never
//! run rules logic themselves — they only render `Observation`s pushed by
//! this server and submit `Action`s). M0 ships only a minimal HTTP "hello"
//! endpoint to prove the crate compiles and the `axum` dependency is wired
//! up correctly.

use axum::{routing::get, Router};

async fn hello() -> &'static str {
    "duels-server: not yet implemented"
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(hello));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("failed to bind 127.0.0.1:3000");
    println!("duels-server listening on http://127.0.0.1:3000");
    axum::serve(listener, app).await.expect("server error");
}
