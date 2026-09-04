# duels

A from-scratch implementation of **7 Wonders Duel** (base game only — no
Agora, no Pantheon), built as: a Rust rules engine, an `Agent` trait for
AI/bot players (hand-written or RL-trained via future PyO3 bindings), a
server-authoritative game server, and a TypeScript/React web client.

## Status: M2 (playable slice)

A person can now play a full game of 7 Wonders Duel end to end in a browser,
human vs. the random bot or hot-seat (two humans, one tab). `duels-core`
(M1) is a complete, tested implementation of the base game; everything
downstream — the server, agents, the web client — reads state from it and
submits `Action`s back, never implementing rules logic itself.

- **Cargo workspace** (`cargo build` / `cargo test` from repo root):
  - `crates/duels-core` — **the rules engine**: static data loading, setup,
    legal-move generation, the cost engine, effects, the chance API,
    observations, and scoring. Zero I/O, zero async, no global mutable
    state, and no randomness except through an explicitly passed
    `rand::rngs::StdRng`.
  - `crates/duels-agents-api` — the `Agent` trait contract that AI/bot
    players implement against (`CONTRACT_VERSION = 2`).
  - `crates/agents/random` — `RandomAgent`, the first concrete `Agent`:
    uniformly picks among the actions it's offered.
  - `crates/duels-arena` — placeholder binary; the future tournament/Elo
    runner for pitting agents against each other.
  - `crates/duels-server` — the server-authoritative game server: a
    room-based REST + WebSocket API (`axum`) that drives games with
    `duels-core`'s engine as the sole source of truth for legality and
    scoring, and exports its wire types to TypeScript (`ts-rs`) for `web/`.
- **`web/`** — the React/Vite/Zustand/Tailwind client. Talks to
  `duels-server` only over the generated wire types; renders every rule the
  engine exposes (structure, both cities, the military track, the wonder
  draft, every pending-choice modal, final scoring) without reimplementing
  any of it.
- **CI** (`.github/workflows/ci.yml`): `fmt` + `clippy` + `test` (Rust) and
  path-filtered `web` (typecheck/lint/vitest/build + a TypeScript-bindings
  drift check) and `e2e` (Playwright, plays a full game through the
  rendered UI) jobs, gated behind a single required `gate` check on PRs to
  `main`. A workspace `clippy.toml` bans `Instant::now`, `SystemTime::now`,
  `rand::thread_rng` and `rand::random` so the engine cannot quietly become
  non-deterministic.
- **Game data** (`data/`): all 73 base-game age cards, 12 wonders, 10
  progress tokens, and the military track/tokens, as structured JSON,
  embedded with `include_str!` and validated on load. See `data/README.md`
  for provenance.
- **Docs**:
  - `docs/rules-spec.md` — **every rule the engine implements, numbered
    `R-xxx`, with the test that covers it**, plus measured performance
    numbers and the list of rules that were inferred rather than read off
    the rulebook.
  - `docs/agent-contract.md` — the versioned contract between the engine and
    any `Agent` implementation.
  - `docs/adr/` — architecture decision records.

### What the engine covers

Setup (age structures including Age III's pinched layout, the three-cards-
returned-to-the-box rule, the three-of-seven guild selection, the 1-2-1
wonder draft, five-of-ten progress tokens), the full cost engine (trade
prices, trading posts, choice production, the Architecture and Masonry
rebates, chain symbols), every card, wonder and progress-token effect in the
base game, the military track with its loot tokens, both instant wins, the
end-of-age first-player rule, and final scoring with tie-breaks.

Notably included, since they are the fiddly ones: The Great Library (draws
three of the progress tokens set aside at setup), The Mausoleum (rebuilds
from the discard pile, with full effects), Circus Maximus and The Statue of
Zeus (destroyed buildings go to the discard pile), Theology and Play Again
(with the extra turn lost if the age ends first), Urbanism, Economy,
Strategy, Law and Mathematics, and the guild coins-versus-points timing
split.

### Quick example

```rust
use duels_core::engine;
use rand::{rngs::StdRng, SeedableRng};

let mut state = engine::new_game(42);
let mut rng = StdRng::seed_from_u64(42);

while let Some(&action) = engine::legal_actions(&state).first() {
    engine::apply(&mut state, action, &mut rng).unwrap();
}
println!("{:?}", state.result().unwrap());
```

An agent is handed `state.observation()` and `engine::legal_actions(&state)`,
and never a `GameState`.

## Why `GameState` vs. `Observation`?

7 Wonders Duel is stochastic but has **no player-private hidden
information** — both players always see the same public state. The only
unknowns are future card reveals (which card is behind a face-down slot,
which three cards of each age went back in the box, the composition of the
not-yet-dealt age decks), which are unknown to both players equally. `duels-core` models the
full state and the public observation as distinct types so that an AI
`Agent` can never be handed hidden information by accident — see
`docs/agent-contract.md` for details.

## Repo layout

```
Cargo.toml                 workspace root
clippy.toml                workspace lint config: bans clock reads and ambient RNG
crates/
  duels-core/               rules engine
    src/data.rs             static card/wonder/token data, normalised and validated
    src/layout.rs           the three age structures: geometry and covering
    src/state.rs            GameState + PlayerState (all fields private)
    src/observation.rs      Observation, and sampling a state back from one
    src/action.rs           the Action enum
    src/cost.rs             the cost engine
    src/engine.rs           setup, legal_actions, apply, the chance API
    src/scoring.rs          victory conditions and final scoring
    src/event.rs            Events emitted while applying an action
    src/testing.rs          StateBuilder and friends, for hand-built positions
    tests/                  cost_engine, golden_scenarios, properties
    benches/                apply/legal_actions throughput
  duels-agents-api/         Agent trait, AgentSpec, Budget
  duels-arena/              tournament/Elo runner (placeholder)
  duels-server/             room-based REST + WebSocket game server
    src/protocol.rs          the wire contract (ts-rs-derived TypeScript bindings)
    src/room.rs               room/seat model and the apply-then-drive-agents game loop
    src/catalog.rs            static card/wonder/token/military reference data
  agents/
    random/                 RandomAgent: uniformly picks among legal actions
web/                        Vite + React 18 + TypeScript + Zustand + Tailwind client
  src/generated/            TypeScript bindings generated from duels-core/duels-server
  e2e/                      Playwright spec: full game through the rendered UI
data/                       cards.json, wonders.json, tokens.json, military.json + README
docs/
  rules-spec.md             numbered rules (R-xxx) -> covering tests, perf, open questions
  agent-contract.md         Agent/Observation/Action contract + versioning
  adr/                      architecture decision records
.github/workflows/ci.yml    fmt + clippy + test + web + e2e, gated behind `gate`
docker-compose.yml          `docker compose up` runs the server and the web client
CODEOWNERS                  mandatory review on docs/, .github/, data/
```

## Getting started

```
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p duels-core      # engine throughput
```

Measured on an Apple Silicon development machine: one `apply` is ~16 ns, a
`GameState` copy ~0.9 ns, and a complete game (setup plus ~70 decisions)
~21 µs. See `docs/rules-spec.md` for the full table.

## Playing it

```
docker compose up --build
# open http://localhost:5173
```

or without Docker, in two terminals:

```
cargo run -p duels-server            # http://localhost:8080
npm --prefix web install && npm --prefix web run dev   # http://localhost:5173
```

`web/`'s TypeScript types are generated, not hand-written — regenerate them
after changing a wire type in `duels-core` or `duels-server` and commit the
result:

```
cargo test -p duels-server           # also exercises duels-core's `ts` feature
git status web/src/generated         # should be clean if nothing drifted
```
