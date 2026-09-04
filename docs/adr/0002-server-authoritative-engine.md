# 0002. Server-authoritative engine

## Context

7 Wonders Duel is fully public-information-stochastic: there is no
player-private hidden information, only future randomness (deck order,
which cards are face-down) that is equally unknown to both players. A client
could in principle run the rules engine locally and only sync moves, but
that would let a modified/compromised client claim illegal moves are legal,
desync from an opponent's or spectator's view, or (for the AI-training use
case) let an agent peek at data a client-side implementation happens to
compute even if it isn't supposed to be shown.

## Decision

Game state lives on a Rust server (`duels-server`) built on `duels-core`.
Clients (the web UI, and later any bot-vs-bot arena spectating) only ever
receive `Observation`s pushed by the server and submit `Action`s back over
WebSocket; the server is the sole authority on legality, effect resolution,
and scoring. No rules logic is duplicated client-side beyond optional
UI-level affordances (e.g. greying out obviously-illegal buttons), which
must always defer to the server's actual response.

## Consequences

- Single source of truth for game legality/scoring — eliminates a whole
  class of client/server desync bugs.
- Requires a persistent connection (WebSocket) and a server deployment,
  rather than a fully static client; acceptable given multiplayer and
  AI-vs-human play are core goals.
- `duels-server` in M0 is a placeholder binary only (a minimal HTTP "hello"
  endpoint) to reserve the crate and prove the `axum` dependency wiring; the
  actual WebSocket game-driving protocol is designed and built in a later
  milestone once `duels-core`'s engine exists.
