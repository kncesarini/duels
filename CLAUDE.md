# CLAUDE.md

Guidance for Claude Code working in this repository. This is a from-scratch implementation of *7 Wonders Duel* (Rust rules engine + React/TS web client) with a research pipeline for building and benchmarking multiple AI opponents. Read this before making changes — it encodes conventions that took real trial and error to establish.

## Architecture at a glance

```
crates/
  duels-core/          the ONLY place rules logic lives. Single source of truth.
  duels-agents-api/     the Agent trait every AI implements: choose(&Observation, legal, Budget) -> Action
  duels-strategy/       pure, public-information-only "win condition" reads (military/science/VP race
                        magnitudes) — a policy/prior signal for search, NOT a value estimator
  duels-arena/          tournament runner: paired-seed matches, Bayesian Elo, SPRT, spec-string agent configs
  duels-server/         axum WebSocket/REST game server, server-authoritative
  agents/
    random/             floor baseline
    greedy/             1-ply heuristic, samples one hidden-info guess and commits to it
    greedy-ev/          same evaluation as greedy, but properly averages over chance_outcomes
                        instead of guessing — see "AI agent conventions" below
    alphabeta/          expectimax + alpha-beta + Star1 pruning; simulation-based leaves (NOT static
                        eval — see "what we learned" below)
    mcts-uct/           chance-node MCTS — the current champion
web/                    React + TypeScript + Vite + Zustand + Tailwind, generated types from Rust (ts-rs)
data/                   card/wonder/token/military JSON (factual game data; see data/README.md for
                        provenance caveats — best-effort, spot-checked, not verbatim from a physical copy)
docs/
  rules-spec.md         numbered R-xxx rule statements, each naming the test that covers it
  agent-contract.md     the versioned Agent/Observation/Action contract (CONTRACT_VERSION)
  adr/                  architecture decision records
arena/                  arena/results/ (gitignored) holds tournament output JSON
```

## Non-negotiable invariants

These are load-bearing. Breaking them silently is the single most likely way to introduce a bug that looks fine until an AI agent starts exploiting it.

- **`duels-core` is the only rules authority.** No other crate — not an agent, not the server, not the web client — reimplements legality, cost, effects, or scoring. If you need a rule, add an accessor to `duels-core`; don't approximate it elsewhere.
- **`GameState` vs `Observation` is enforced by the type system, not convention.** `GameState` holds hidden information (deck order, face-down identities); `Observation` never does. Every `Agent` implementation — and every function in `duels-strategy` — must be provably invariant to *which* hidden-info sample produced the concrete state it's handed. This project writes a **determinization-invariance property test** for any new logic that touches game state, comparing two different `Observation::sample_state` draws bit-for-bit (`to_bits()` on floats). If you can't write that test, the logic is leaking hidden information somewhere.
- **Determinism is enforced by lint, not discipline.** `clippy.toml` bans `Instant::now`, `SystemTime::now`, `rand::thread_rng`, `rand::random` inside `duels-core` and every agent crate. Randomness only ever enters through an explicitly-passed, seeded `StdRng`.
- **Agent crates are self-contained.** No agent crate depends on another agent crate, even when it would save duplicating an evaluation function. This is deliberate — it lets multiple agents be built in parallel by independent agents without cross-crate coordination, and it means benchmarking one never risks silently coupling to another's internals. Some duplication (e.g. `greedy-ev` reimplementing `greedy`'s evaluation terms) is an accepted, intentional cost.

## AI agent development: the established pattern

This project has now built and refined five agents plus a strategy layer. A consistent discipline emerged; follow it for any new agent work:

1. **New capability = opt-in `Config`, old behavior stays the default (or an explicit, proven-identical option).** Never silently change what an agent does. When adding a mode/parameter, write a test proving the new option, set to its "off" value, is **bit-identical** to the pre-existing code path (see `mcts-uct`'s root-determinization-ensembling PR for the gold-standard version of this test: a verbatim copy of the old function, whole seeded games driven through both, move-for-move equality asserted).
2. **Validate empirically via `duels-arena`, always.** Build it in release mode (`cargo build --release -p duels-arena`). Use paired-seed, seat-swapped matches. Report Elo with a confidence interval, not just a win count.
3. **Test at both `Nodes` and `TimeMs` budgets.** A change that helps at a fixed node count can lose at a fixed wall-clock budget if it costs more per unit of work, and vice versa — this project has been burned by exactly that more than once. Report both.
4. **`TimeMs` runs are load-sensitive.** A benchmark run on a machine with other concurrent work (including other Claude Code agents) can swing 15+ points between runs. Run one match at a time on a quiet machine for anything you intend to trust; treat small-sample `TimeMs` results as indicative, not conclusive, until reproduced.
5. **Reproduce on a second, disjoint seed range before trusting an accept.** One seed range is not evidence.
6. **Report honest negatives.** Several real investigations in this codebase concluded "this doesn't help" (a smarter MCTS rollout policy, root-determinization ensembling at practical budgets) and shipped the attempt as a documented, non-default option rather than hiding it or forcing a marginal win into the story. Do the same. A well-documented negative result is a valid, valued deliverable here.
7. **Only change `Config::default()` when the evidence clearly supports it.**

### What we've learned about this game specifically (useful priors for future work)

- **7 Wonders Duel is a two-player zero-sum *stochastic* game with *no private information*** — both players always see the same public state; only future card reveals are unknown to both equally. One `Observation` serves both players and any spectator. This is why chance-node search (expectimax, MCTS with explicit chance nodes) applies directly — no need for anything from the imperfect-information literature (ISMCTS, CFR).
- **A static, hand-crafted position evaluation has a low ceiling in this game.** Scoring is holistic and end-game-heavy (most VP resolves only in aggregate at game end), so a few-plies-deep static eval judges positions badly — `alphabeta` with a static leaf only won ~2.5% of the time against `mcts-uct` even with 25x the search budget. Blending in an actual random playout to a real `GameResult` (instead of a static score) raised that to ~19.5%. **Simulation beats hand-crafted judgment for *position value* in this game.**
- **Win-condition awareness belongs in the search policy, not the evaluation function.** `greedy` has explicit military-race terms in its static evaluation and *still* loses to `random` by military supremacy ~10% of the time, because a 1-ply view can't see a race developing three moves out. `duels-strategy` exists specifically to bias *where search looks* (tree priors, rollout policy) rather than to replace simulation as the value signal — see its crate-level doc comment for the full reasoning.
- **`duels-strategy`'s reads are genuinely not free** (~17-29% of one MCTS rollout for a full slate of action priors on a real position) — cheap enough to compute once per search-tree node, too expensive to recompute per simulation. Cache it.
- **First-player advantage is real and large** even between equally-strong agents (~67/33 observed at equal MCTS budget) — never compare agents without paired, seat-swapped matches.

## Testing conventions

- **`duels_core::testing::StateBuilder`** constructs hand-built positions for unit tests across every crate in this repo. Use it rather than driving a game from scratch when you need a specific scenario (e.g. "one move from military supremacy").
- **`proptest`** for randomized invariant checking across many played-out games (card conservation, coins never negative, `Observation` never leaks a hidden identity, etc.) — see `duels-core/tests/properties.rs` for the established style.
- Keep large-N benchmark-style runs (hundreds of games, release-mode timing) **out of the default `cargo test` path** — use `#[ignore]` or a separate `examples/` binary, so CI stays fast. `duels-strategy`'s `examples/watch_reads.rs` is the pattern for a human-readable diagnostic tool.

## Rules traceability

Every non-trivial rule `duels-core` implements has a numbered `R-xxx` entry in `docs/rules-spec.md` naming the test that covers it. If you touch rules logic, update this file. `data/README.md` documents the same discipline for the factual game data (card/wonder/token definitions) and flags what's been spot-checked vs. best-effort.

## Git / PR workflow

- **All changes to `main` go through a PR.** Branch protection requires the `gate` CI check (fmt, clippy `-D warnings`, test, web typecheck/lint/build, e2e) to pass, plus code-owner review on `docs/**`, `.github/**`, `data/**`, and `CODEOWNERS` itself.
- Squash merge, linear history. Delete the branch after merging.
- **Branch protection uses a "strict" status policy** — a PR's branch must be up to date with `main` before merging, even if the diff doesn't textually conflict. If `gh pr merge` refuses with "not up to date," merge `origin/main` into the PR branch, re-verify build/tests, push, wait for the new CI run, then merge.
- Expect trivial `Cargo.toml`/`Cargo.lock` conflicts when multiple PRs each add a new workspace member (a new agent crate) in parallel — resolve by keeping all the added member lines and regenerating the lockfile with a build, not by picking one side.
- No CI-required LLM review gate at present (declined as a deliberate cost/complexity tradeoff early on) — revisit if that changes.

## Orchestrating multiple Claude Code agents in this repo

If you are an orchestrator dispatching multiple background agents that will write to this repository concurrently: **always pass `isolation: "worktree"`**, even for agents that are each told to use a different branch. Without it, concurrent agents share one working directory, and one agent's `git checkout` can silently clobber another's in-progress, uncommitted edits — this happened in this repo's history and cost real cleanup effort. A single sequential agent doesn't need this.

When several PRs from parallel agents each add a new crate to the workspace, merge them one at a time, updating each PR's branch against the latest `main` (and resolving the trivial `Cargo.toml`/lock conflict) before each merge — don't try to land them all at once.

## Local development

```bash
docker compose up -d --build     # server on :8080, web on :4173 (port-mapped 1:1, not proxied)
curl http://localhost:8080/agents   # list registered AI agents
cargo build --release -p duels-arena
./target/release/duels-arena match --agent-a mcts-uct --agent-b alphabeta \
    --games 200 --budget nodes:2000 --seed 1
```

The web client never implements rules/legality/cost logic — it only renders what the server sends (an `Observation` plus legal actions) and submits `Action`s back. Card/wonder/token effect descriptions in the UI are generated from structured data (`web/src/lib/effectText.ts`), not hand-written per card.

## Current state (living reference — verify against `duels-arena` for ground truth, this will drift)

Milestones complete: rules engine, playable web UI + server, five AI agents (`random`/`greedy`/`greedy-ev`/`alphabeta`/`mcts-uct`), tournament infrastructure, and `duels-strategy` (a win-condition-aware policy layer, built but — as of this writing — still being wired into search). `mcts-uct` is the strongest agent. Not started: self-play RL, promotion automation, hosting/polish (see `docs/adr/` for the original architecture decisions and their rationale).
