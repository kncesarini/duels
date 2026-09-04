# duels

A from-scratch implementation of **7 Wonders Duel** (base game only — no
Agora, no Pantheon), built as: a Rust rules engine, an `Agent` trait for
AI/bot players (hand-written or RL-trained via future PyO3 bindings), a
server-authoritative game server, and a TypeScript/React web client.

## Status: M0 (scaffold)

This milestone lays out the project shape so later milestones can build the
real rules engine, AI agents, and UI without restructuring:

- **Cargo workspace** (`cargo build` / `cargo test` from repo root):
  - `crates/duels-core` — the rules engine. M0 ships module stubs
    (`state`, `observation`, `action`, `data`, `engine`, `scoring`) and a
    first-pass `Action` enum; the actual rules land in M1.
  - `crates/duels-agents-api` — the `Agent` trait contract that AI/bot
    players implement against.
  - `crates/duels-arena` — placeholder binary; the future tournament/Elo
    runner for pitting agents against each other.
  - `crates/duels-server` — placeholder binary; the future
    server-authoritative WebSocket game server.
- **CI** (`.github/workflows/ci.yml`): `fmt` + `clippy` + `test`, gated
  behind a single required `gate` check on PRs to `main`.
- **Game data** (`data/`): all 73 base-game age cards, 12 wonders, 10
  progress tokens, and the military track/tokens, as structured JSON. This
  is a best-effort transcription — see `data/README.md` for provenance and
  what still needs spot-checking before M1 relies on it.
- **Docs**:
  - `docs/agent-contract.md` — the versioned contract (`CONTRACT_VERSION`)
    between the engine and any `Agent` implementation, and why `GameState`
    (full, hidden-info-included) and `Observation` (public-only) are
    separate types enforced by the type system, not convention.
  - `docs/adr/` — architecture decision records for the choices this
    scaffold is built on (Rust core + TS web, server-authoritative engine,
    public repo, CI-only PR gating, base-game-only scope).

## Why `GameState` vs. `Observation`?

7 Wonders Duel is stochastic but has **no player-private hidden
information** — both players always see the same public state. The only
unknowns are future card reveals (face-down slots, deck order, remaining
tokens), which are unknown to both players equally. `duels-core` models the
full state and the public observation as distinct types so that an AI
`Agent` can never be handed hidden information by accident — see
`docs/agent-contract.md` for details.

## Repo layout

```
Cargo.toml                 workspace root
crates/
  duels-core/               rules engine (state, action, observation, data, engine, scoring)
  duels-agents-api/         Agent trait, AgentSpec, Budget
  duels-arena/              tournament/Elo runner (placeholder)
  duels-server/             WebSocket game server (placeholder)
data/                       cards.json, wonders.json, tokens.json, military.json + README
docs/
  agent-contract.md         Agent/Observation/Action contract + versioning
  adr/                      architecture decision records
.github/workflows/ci.yml    fmt + clippy + test, gated behind `gate`
CODEOWNERS                  mandatory review on docs/, .github/, data/
```

## Getting started

```
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```
