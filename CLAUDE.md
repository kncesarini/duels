# CLAUDE.md

Guidance for Claude Code working in this repository. This is a from-scratch implementation of *7 Wonders Duel* (Rust rules engine + React/TS web client) with a research pipeline for building and benchmarking multiple AI opponents. Read this before making changes — it encodes conventions that took real trial and error to establish.

## Architecture at a glance

```
crates/
  duels-core/          the ONLY place rules logic lives. Single source of truth.
  duels-agents-api/     the Agent trait every AI implements: choose(&Observation, legal, Budget) -> Action
  duels-strategy/       pure, public-information-only "win condition" reads (military/science/VP race
                        magnitudes) — a policy/prior signal for search, NOT a value estimator
  duels-eval/           the hand-crafted position evaluation: Config, Root, evaluate, expected_value,
                        and the commitment blend / terms / menu / rails behind them. A library BELOW
                        the agents (see "the layering" in the invariants), shared so more than one
                        agent can use it. Mandatory code-owner review — see CODEOWNERS.
  duels-arena/          tournament runner: paired-seed matches, Bayesian Elo, SPRT, spec-string agent configs
  duels-server/         axum WebSocket/REST game server, server-authoritative
  agents/
    random/             floor baseline
    greedy/             1-ply heuristic, samples one hidden-info guess and commits to it
    greedy-ev/          same evaluation as greedy, but properly averages over chance_outcomes
                        instead of guessing — see "AI agent conventions" below
    strategist/         greedy-ev plus a duels-strategy move-level prior
    phased/             a thin 1-ply Agent over duels-eval: sample a state, build one Root, score
                        every legal action, play the best. Holds no evaluation logic of its own.
    alphabeta/          expectimax + alpha-beta + Star1 pruning; simulation-based leaves (NOT static
                        eval — see "what we learned" below)
    mcts-uct/           chance-node MCTS, playout leaf value, `c = 1.0`. The designated champion
                        (`leaderboard::CHAMPION`) and the yardstick every knob here was tuned against
    mcts-eval/          the same search, with a leaf value that is half playout and half duels-eval
                        (`LeafValue::Blend { weight: 0.5 }`, `c = 0.5`). +89 Elo over mcts-uct, the
                        largest effect this project has measured. Deliberately tracks
                        `duels_eval::Config::default()` LIVE — no generation pin; read its crate
                        docs before "fixing" that. Carries a verbatim copy of mcts-uct's search as
                        its ablation control (`Config::rollout_base`), asserted move-for-move
web/                    React + TypeScript + Vite + Zustand + Tailwind, generated types from Rust (ts-rs)
data/                   card/wonder/token/military JSON (factual game data; see data/README.md for
                        provenance caveats — best-effort, spot-checked, not verbatim from a physical copy)
docs/
  rules-spec.md         numbered R-xxx rule statements, each naming the test that covers it
  agent-contract.md     the versioned Agent/Observation/Action contract (CONTRACT_VERSION)
  adr/                  architecture decision records
arena/                  arena/results/ (gitignored) holds tournament output JSON;
                        arena/leaderboard.md + .json ARE committed — they're the
                        leaderboard, refreshed nightly by CI (see below)
.github/workflows/
  ci.yml                the required `gate` check (fmt, clippy, test, web, e2e)
  nightly-arena.yml     nightly full round robin: 21 pairings as a job matrix,
                        joint Elo refit, leaderboard update proposed as a PR
  ai-candidate.yml      informational (NOT gating) candidate-vs-champion match on any PR
                        touching crates/agents/**, duels-agents-api/** or duels-eval/**
```

## Non-negotiable invariants

These are load-bearing. Breaking them silently is the single most likely way to introduce a bug that looks fine until an AI agent starts exploiting it.

- **`duels-core` is the only rules authority.** No other crate — not an agent, not the server, not the web client — reimplements legality, cost, effects, or scoring. If you need a rule, add an accessor to `duels-core`; don't approximate it elsewhere.
- **`GameState` vs `Observation` is enforced by the type system, not convention.** `GameState` holds hidden information (deck order, face-down identities); `Observation` never does. Every `Agent` implementation — and every function in `duels-strategy` — must be provably invariant to *which* hidden-info sample produced the concrete state it's handed. This project writes a **determinization-invariance property test** for any new logic that touches game state, comparing two different `Observation::sample_state` draws bit-for-bit (`to_bits()` on floats). If you can't write that test, the logic is leaking hidden information somewhere.
- **Determinism is enforced by lint, not discipline.** `clippy.toml` bans `Instant::now`, `SystemTime::now`, `rand::thread_rng`, `rand::random` inside `duels-core` and every agent crate. Randomness only ever enters through an explicitly-passed, seeded `StdRng`.
- **Agent crates are self-contained.** No agent crate depends on another agent crate, even when it would save duplicating an evaluation function. This is deliberate — it lets multiple agents be built in parallel by independent agents without cross-crate coordination, and it means benchmarking one never risks silently coupling to another's internals. Some duplication (e.g. `greedy-ev` reimplementing `greedy`'s evaluation terms) is an accepted, intentional cost.
- **The layering is `duels-core` → `duels-strategy` → `duels-eval` → agents**, each layer depending only on the ones above it. So when two agents genuinely should share code, it moves *down* into a library rather than sideways between agents: `duels-eval` exists because `phased`'s evaluation was wanted as a search leaf value too, and the rule above forbids one agent depending on another (`mcts-eval` therefore carries a *copy* of `mcts-uct`'s tree/chance/rollout machinery, not a dependency on it — the same accepted duplication as `greedy-ev` against `greedy`). A shared library at this level carries obligations the agent above it does not: `duels-eval` depends on `duels-core` and `duels-strategy` and on **nothing else** (no `rand`, no `duels-agents-api` — `Root::new`, `evaluate` and `expected_value` are pure functions), it holds the determinization-invariance and version-snapshot identity tests for everything it owns, and it is a **mandatory-review path in `CODEOWNERS`** — its behaviour is not to be changed in passing inside a PR about something else, because more than one agent's measured strength moves when it does.

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
- **...but a hand-crafted evaluation *blended with* a playout, as an MCTS leaf value, is the biggest win this project has measured** (`+89` Elo pooled over 3,600 games; it is now the `mcts-eval` agent, whose crate docs hold the full measurement). This refines the two priors above rather than contradicting them. A **pure** `duels-eval` leaf is far *worse* than the playout it replaces (`-171` Elo at a fixed node count), exactly as the low-ceiling prior says. Half playout and half evaluation beats both. The victory-kind breakdown says why they are complementary: the evaluation supplies civilian-score judgement (where its terms live), the playout supplies sight of military races — and pushing the blend weight past ~0.5 visibly trades the second away for the first. **When a hand-crafted signal doesn't work as a replacement for simulation, try it as a mixture before concluding it doesn't work.**
- **A blended reward changes what the exploration constant means.** Mixing a static value into a Bernoulli playout at weight `w` shrinks the reward's spread by `1 - w`, so UCB1's `c` has to be scaled by `1 - w` to leave the tuned exploration/exploitation balance alone. This is not a subtlety to discover by sweeping: `c = 0.3` alone measures at `-100` Elo and yet is strongly *positive* inside a `weight = 0.7` blend. Any future change to what a leaf backs up should re-derive `c` before measuring.
- **Whether a search that consumes `duels-eval` should *pin* a generation depends on whether the evaluation is incidental to the agent or *is* the agent.** `duels-eval` is tuned by `phased` rounds, so an agent whose measured strength depends on it and whose identity is its *search* should pin a frozen snapshot (`Config::vN()`) plus a golden-values test — a later round then fails a test rather than silently re-defining what was measured. `mcts-eval` is the deliberate exception and the opposite call: its whole reason to exist is "`duels-eval` inside a search", it is meant to get stronger automatically as future rounds land, so it reads `duels_eval::Config::default()` live in `Tree::new`, pins nothing, and holds no golden-values test. It pays for that by recording the **whole** `duels_eval::Config::params_string()` in its `AgentSpec` params, which makes two results files from either side of a round distinguishable rather than making the older one uninterpretable. Read that crate's "Tracking `duels-eval` live" section before adding a pin to it; the absence of one is not an oversight. **Corollary worth knowing:** `mcts-uct` no longer consumes `duels-eval` at all, so `duels_eval::Config::v6()`'s doc-comment claim that a downstream `mcts-uct` golden-values test enforces the generation chain is now stale — nothing downstream enforces it. Re-establishing that guard (in `duels-eval` itself, where it belongs) needs a code-owner-reviewed PR.
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

## The leaderboard and the nightly round robin (M6b)

`arena/leaderboard.md` (rendered from `arena/leaderboard.json`) is the standing ranking of every registered agent. Both files are **generated** — `.github/workflows/nightly-arena.yml` rebuilds them every night and opens a PR; don't hand-edit them.

- **The ladder is defined in code**, in `duels_arena::leaderboard::LADDER`: each agent at its production budget (`Nodes(1)` for the five 1-ply agents, `Nodes(2000)` for `alphabeta`/`mcts-uct`/`mcts-eval`), default config only. Adding an agent there extends the nightly matrix automatically — the workflow never lists agent names, so `LADDER` plus `agent_registry` is the whole registration for the nightly job. (Eight agents means `C(8,2)` = 28 pairings, up from 21; `nightly-arena.yml`'s prose still says 21 in a comment, harmless but stale — `.github/**` is a code-owner path so it was left alone.)
- **The whole round robin runs at one budget (`nodes:2000`)** and that is not a compromise: the five 1-ply agents take `_budget` in `Agent::choose` and never read it, which `leaderboard::tests::one_ply_agents_ignore_their_budget` proves by playing games at both budgets and comparing every decision.
- **Ratings are fitted jointly**, not pairwise — `elo::fit_joint_elo` is a Bradley-Terry MLE over all 28 head-to-head records at once (MM iteration, CIs from the joint Fisher information with the anchor's row/column deleted). `greedy` is pinned at 1000. Use `fit_elo` for a single head-to-head comparison; use `fit_joint_elo` for anything ladder-shaped.
- **`main` cannot be pushed to directly** — the `main-protection` ruleset has an empty `bypass_actors` list — so the nightly proposes a PR. GitHub does not start workflows for `GITHUB_TOKEN`-authored PRs, so that PR's required `gate` check needs a close/reopen (or a `NIGHTLY_ARENA_TOKEN` PAT secret) before it can merge. The workflow says so in the PR body.
- **`ai-candidate` is informational and must stay that way for now** (an explicit decision). It is a separate workflow file precisely so it cannot drift into `ci.yml`'s `gate` job. Promoting it to blocking means two deliberate edits: add it to `gate`'s `needs:` *and* to the ruleset's required-status-check list.
- **The champion is a plain constant** (`leaderboard::CHAMPION`, currently `mcts-uct` at `Nodes(2000)`), not something read back out of the leaderboard. Automated promotion is M7 and does not exist yet; until it does, a human changing one line is the honest mechanism. **It is now deliberately not the top of the ladder**: `mcts-eval` measures about `+100` Elo above `mcts-uct` and should be expected to rank first once the nightly refits, and moving `CHAMPION` is a separate one-line decision on its own evidence rather than a side effect of adding an agent. `leaderboard::tests::a_complete_round_robin_builds_and_ranks_strongest_first` asserts that separation directly.

## Current state

See `docs/milestones.md` for the milestone table, the four human checkpoints, and what's
actively being worked on. Update that file, not this section, as things change — this
avoids keeping two summaries in sync. `docs/adr/` has the original architecture decisions
and their rationale.
