//! Binds `duels_server::app()` to a socket. See `src/lib.rs` for everything
//! that matters.

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = duels_server::app();

    let addr = std::env::var("DUELS_SERVER_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {addr}: {e}"));
    tracing::info!("duels-server listening on http://{addr}");
    axum::serve(listener, app).await.expect("server error");
}
