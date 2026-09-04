//! End-to-end test: spin up the real server on an ephemeral port, create a
//! human-vs-`random` room over REST, then drive the human seat over a raw
//! WebSocket connection with a scripted "always pick `legal[0]`" policy
//! until the game reaches a [`duels_core::GameResult`].
//!
//! This is the server-side analogue of the browser e2e test in `web/`: it
//! proves the REST + WebSocket + room/agent-driving plumbing actually works
//! end to end, independent of any UI.

use std::net::SocketAddr;

use duels_server::protocol::{
    ClientMessage, CreateRoomRequest, CreateRoomResponse, SeatSpec, ServerMessage,
};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message as WsMessage;

async fn spawn_server() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let app = duels_server::app();
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("server error");
    });
    addr
}

#[tokio::test]
async fn a_full_game_against_the_random_bot_reaches_a_result() {
    let addr = spawn_server().await;
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let create: CreateRoomResponse = client
        .post(format!("{base}/rooms"))
        .json(&CreateRoomRequest {
            seats: [
                SeatSpec::Human,
                SeatSpec::Agent {
                    name: "random".to_string(),
                },
            ],
            seed: Some(20260904),
        })
        .send()
        .await
        .expect("POST /rooms")
        .json()
        .await
        .expect("decode CreateRoomResponse");

    let ws_url = format!("ws://{addr}/rooms/{}/ws", create.room_id);
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&ws_url)
        .await
        .unwrap_or_else(|e| panic!("connect to {ws_url}: {e}"));

    let mut decisions = 0u32;
    let mut saw_a_build = false;
    let mut saw_a_wonder_pick = false;

    loop {
        let msg = ws
            .next()
            .await
            .expect("websocket closed before a result arrived")
            .expect("websocket error");
        let WsMessage::Text(text) = msg else {
            continue;
        };
        let server_msg: ServerMessage = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("bad ServerMessage json: {e}\n{text}"));

        let ServerMessage::State(state) = server_msg else {
            panic!("unexpected Error message: {text}");
        };

        if let Some(result) = state.observation.result {
            println!("game over after {decisions} human decisions: {result:?}");
            assert!(saw_a_build, "the scripted playout never built anything");
            assert!(
                saw_a_wonder_pick,
                "the scripted playout never drafted a wonder"
            );
            return;
        }

        // Whenever legal_actions is non-empty, it is the human seat's turn
        // (agent seats are driven synchronously server-side before a
        // broadcast is ever sent) — apply the "always pick legal[0]" policy.
        let Some(action) = state.legal_actions.first().copied() else {
            // No result yet but nothing legal either would be a server bug;
            // keep waiting for the next message just in case this is a
            // transient state during a reconnect-style resend.
            continue;
        };

        use duels_core::Action;
        match action {
            Action::Build { .. } | Action::BuildWonder { .. } => saw_a_build = true,
            Action::PickWonder { .. } => saw_a_wonder_pick = true,
            _ => {}
        }

        let request = ClientMessage::Action { action };
        ws.send(WsMessage::Text(
            serde_json::to_string(&request).expect("serialize ClientMessage"),
        ))
        .await
        .expect("send action");

        decisions += 1;
        assert!(decisions < 500, "game did not terminate in time");
    }
}

#[tokio::test]
async fn creating_a_room_with_an_unknown_agent_is_rejected() {
    let addr = spawn_server().await;
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/rooms"))
        .json(&CreateRoomRequest {
            seats: [
                SeatSpec::Human,
                SeatSpec::Agent {
                    name: "not-a-real-agent".to_string(),
                },
            ],
            seed: Some(1),
        })
        .send()
        .await
        .expect("POST /rooms");

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

/// Smoke-tests every non-`random` agent `room::KNOWN_AGENTS` lists: creates a
/// human-vs-agent room and drives a handful of human decisions ("always pick
/// `legal[0]`"), checking that the agent seat actually replies with a fresh
/// `State` broadcast each time rather than erroring or hanging. Doesn't play
/// to completion (unlike the `random` test above) since `alphabeta` and
/// `mcts-uct` run a real, if bounded, search per move under the server's
/// interactive `Budget` and this only needs to prove the wiring works.
#[tokio::test]
async fn every_known_agent_besides_random_can_play_a_few_turns() {
    for name in duels_server::room::KNOWN_AGENTS
        .iter()
        .filter(|n| **n != "random")
    {
        let addr = spawn_server().await;
        let base = format!("http://{addr}");
        let client = reqwest::Client::new();

        let create: CreateRoomResponse = client
            .post(format!("{base}/rooms"))
            .json(&CreateRoomRequest {
                seats: [
                    SeatSpec::Human,
                    SeatSpec::Agent {
                        name: name.to_string(),
                    },
                ],
                seed: Some(7),
            })
            .send()
            .await
            .unwrap_or_else(|e| panic!("POST /rooms for agent {name}: {e}"))
            .json()
            .await
            .expect("decode CreateRoomResponse");

        let ws_url = format!("ws://{addr}/rooms/{}/ws", create.room_id);
        let (mut ws, _resp) = tokio_tungstenite::connect_async(&ws_url)
            .await
            .unwrap_or_else(|e| panic!("connect to {ws_url}: {e}"));

        for decision in 0..6u32 {
            let Some(msg) = ws.next().await else {
                panic!("agent {name}: websocket closed after {decision} decisions");
            };
            let msg = msg.unwrap_or_else(|e| panic!("agent {name}: websocket error: {e}"));
            let WsMessage::Text(text) = msg else {
                continue;
            };
            let server_msg: ServerMessage = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("agent {name}: bad ServerMessage json: {e}\n{text}"));
            let ServerMessage::State(state) = server_msg else {
                panic!("agent {name}: unexpected Error message: {text}");
            };

            if state.observation.result.is_some() {
                break; // a short game (unlikely within 6 decisions) is fine too
            }
            let Some(action) = state.legal_actions.first().copied() else {
                continue;
            };
            let request = ClientMessage::Action { action };
            ws.send(WsMessage::Text(
                serde_json::to_string(&request).expect("serialize ClientMessage"),
            ))
            .await
            .unwrap_or_else(|e| panic!("agent {name}: send action: {e}"));
        }
    }
}

#[tokio::test]
async fn get_agents_lists_random_first_and_every_known_agent() {
    let addr = spawn_server().await;
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let agents: Vec<String> = client
        .get(format!("{base}/agents"))
        .send()
        .await
        .expect("GET /agents")
        .json()
        .await
        .expect("decode agent list");

    assert_eq!(agents.first().map(String::as_str), Some("random"));
    for name in duels_server::room::KNOWN_AGENTS {
        assert!(
            agents.iter().any(|a| a == name),
            "GET /agents missing {name}"
        );
    }
}

#[tokio::test]
async fn get_catalog_covers_every_card_wonder_and_token() {
    let addr = spawn_server().await;
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let catalog: duels_server::protocol::Catalog = client
        .get(format!("{base}/catalog"))
        .send()
        .await
        .expect("GET /catalog")
        .json()
        .await
        .expect("decode Catalog");

    assert_eq!(catalog.cards.len(), duels_core::data::NUM_CARDS);
    assert_eq!(catalog.wonders.len(), duels_core::data::NUM_WONDERS);
    assert_eq!(catalog.tokens.len(), duels_core::data::NUM_TOKENS);
}
