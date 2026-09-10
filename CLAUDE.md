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
  duels-value/          the LEARNED position value, next to duels-eval and below the agents: 211
                        public-information-only features plus a small hand-rolled MLP predicting the
                        four-way victory-kind-and-loss distribution. Depends on duels-core and
                        NOTHING else — no rand, no clock, no ML runtime; the forward pass is an f32
                        matrix-vector product over weights/v1.bin baked in at compile time. Holds
                        its own determinization-invariance test AND a zero-sum coherence test that
                        asserts a REAL, MEASURED DEFECT in the shipped weights (43% of legal
                        positions miss a 0.05 bound; worst cases are science-lead boards). Read that
                        test file before trusting the model.
  duels-arena/          tournament runner: paired-seed matches, Bayesian Elo, SPRT, spec-string agent configs
  duels-server/         axum WebSocket/REST game server, server-authoritative
  agents/
    random/             RETIRED from the roster, kept as a TEST FIXTURE. Uniform-random play;
                        not registered, not selectable, no rating. It is the correctness floor
                        four crates measure themselves against ("a search agent that doesn't
                        beat a random player has a bug"), so it survives as a dev-dependency
                        everywhere. Read its crate docs before deleting or re-registering it.
    phased/             a thin 1-ply Agent over duels-eval: sample a state, build one Root, score
                        every legal action, play the best. Holds no evaluation logic of its own.
                        The only 1-ply agent left on the ladder.
    alphabeta/          expectimax + alpha-beta + Star1 pruning; simulation-based leaves (NOT static
                        eval — see "what we learned" below)
    mcts-uct/           chance-node MCTS, playout leaf value, `c = 1.0`. The yardstick every knob
                        in mcts-eval below was tuned against; no longer the ladder's champion
    mcts-eval/          the same search, with a leaf value that is half playout and half duels-eval
                        (`LeafValue::Blend { weight: 0.5 }`, `c = 0.5`). +89 Elo over mcts-uct, the
                        largest effect this project has measured, and now the designated champion
                        (`leaderboard::CHAMPION`). Deliberately tracks
                        `duels_eval::Config::default()` LIVE — no generation pin; read its crate
                        docs before "fixing" that. Carries a verbatim copy of mcts-uct's search as
                        its ablation control (`Config::rollout_base`), asserted move-for-move
    mcts-value/         the same search again, with duels-value's LEARNED leaf replacing the
                        playout outright (`LeafValue::Learned`) at a re-derived `c = 0.15`.
                        REGISTERED AND PLAYABLE BUT NOT ON THE LADDER AND NOT THE CHAMPION — see
                        `leaderboard::REGISTERED_OFF_LADDER`, and read the honest reading below
                        before quoting its Elo. +91.4 Elo over mcts-eval at the ladder's
                        nodes:2000, +140 at nodes:32000 and at TimeMs(1000) — and that margin is
                        a TARGETED COUNTER TO mcts-eval's KNOWN SCIENCE-VALUE MISCALIBRATION
                        (#57), NOT A GENERAL STRENGTH IMPROVEMENT: only 28% of it survives being
                        measured through mcts-uct and 12% through alphabeta, both CIs crossing
                        zero. Pins duels-value's weights with a golden-values test (the `golden`
                        module) — the OPPOSITE call from mcts-eval's live tracking, argued there.
                        Carries verbatim frozen copies of BOTH mcts-eval's and mcts-uct's searches
                        as its ablation controls (`Config::eval_base`, `Config::rollout_base`),
                        each asserted move-for-move
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
  nightly-arena.yml     nightly full round robin: every LADDER pairing as a job matrix
                        (6 today; the workflow's "21" comment is stale prose),
                        joint Elo refit, leaderboard update proposed as a PR
  ai-candidate.yml      informational (NOT gating) candidate-vs-champion match on any PR
                        touching crates/agents/**, duels-agents-api/** or duels-eval/**
```

## Non-negotiable invariants

These are load-bearing. Breaking them silently is the single most likely way to introduce a bug that looks fine until an AI agent starts exploiting it.

- **`duels-core` is the only rules authority.** No other crate — not an agent, not the server, not the web client — reimplements legality, cost, effects, or scoring. If you need a rule, add an accessor to `duels-core`; don't approximate it elsewhere.
- **`GameState` vs `Observation` is enforced by the type system, not convention.** `GameState` holds hidden information (deck order, face-down identities); `Observation` never does. Every `Agent` implementation — and every function in `duels-strategy` — must be provably invariant to *which* hidden-info sample produced the concrete state it's handed. This project writes a **determinization-invariance property test** for any new logic that touches game state, comparing two different `Observation::sample_state` draws bit-for-bit (`to_bits()` on floats). If you can't write that test, the logic is leaking hidden information somewhere.
- **Determinism is enforced by lint, not discipline.** `clippy.toml` bans `Instant::now`, `SystemTime::now`, `rand::thread_rng`, `rand::random` inside `duels-core` and every agent crate. Randomness only ever enters through an explicitly-passed, seeded `StdRng`.
- **Agent crates are self-contained.** No agent crate depends on another agent crate, even when it would save duplicating an evaluation function. This is deliberate — it lets multiple agents be built in parallel by independent agents without cross-crate coordination, and it means benchmarking one never risks silently coupling to another's internals. Some duplication (e.g. `mcts-eval` and `mcts-value` each carrying a verbatim copy of `mcts-uct`'s tree/chance/rollout machinery; historically, `greedy-ev` reimplementing `greedy`'s evaluation terms) is an accepted, intentional cost. **What that cost actually is, now that there are three copies: nothing mechanical keeps them in sync.** A test in an agent crate cannot read another agent crate — the same rule that forces the copy — so the frozen-copy ablation tests each crate holds catch a change made *inside* that crate and not one made next door. `duels-core` PR #61 had to hand-edit three `chance.rs` files to keep them one file, and a missed one would have kept every test green while an agent silently stopped being the search it documents itself as. When touching the shared search, `diff` the copies (`crates/agents/*/src/chance.rs`, `rollout.rs`) as part of the change; `mcts-value`'s crate docs have the one-liner. The one exception is `duels-agent-random` as a **dev**-dependency: it is a retired agent kept as a shared test fixture, never linked by a shipping binary, and several agent crates' correctness floors need it.
- **The layering is `duels-core` → `duels-strategy` → `duels-eval` / `duels-value` → agents**, each layer depending only on the ones above it. (`duels-value` sits *beside* `duels-eval`, not below it: it depends on `duels-core` alone and the two never reference each other, which is what lets one agent read either, both, or neither.) So when two agents genuinely should share code, it moves *down* into a library rather than sideways between agents: `duels-eval` exists because `phased`'s evaluation was wanted as a search leaf value too, and the rule above forbids one agent depending on another (`mcts-eval` therefore carries a *copy* of `mcts-uct`'s tree/chance/rollout machinery, not a dependency on it — the same accepted duplication `greedy-ev` used to carry against `greedy`). This layering is also why `duels-strategy`'s `watch_reads` example drives its games with `duels-agent-random` and not `phased`: sitting below `duels-eval`, it cannot dev-depend on anything above it without closing a cycle. A shared library at this level carries obligations the agent above it does not: `duels-eval` depends on `duels-core` and `duels-strategy` and on **nothing else** (no `rand`, no `duels-agents-api` — `Root::new`, `evaluate` and `expected_value` are pure functions), it holds the determinization-invariance and version-snapshot identity tests for everything it owns, and it is a **mandatory-review path in `CODEOWNERS`** — its behaviour is not to be changed in passing inside a PR about something else, because more than one agent's measured strength moves when it does.

## AI agent development: the established pattern

This project has built and refined nine agents plus a strategy layer and two value libraries; four of the agents have since been retired, and one more is registered but deliberately unrated (see `leaderboard::REGISTERED_OFF_LADDER`). A consistent discipline emerged; follow it for any new agent work:

1. **New capability = opt-in `Config`, old behavior stays the default (or an explicit, proven-identical option).** Never silently change what an agent does. When adding a mode/parameter, write a test proving the new option, set to its "off" value, is **bit-identical** to the pre-existing code path (see `mcts-uct`'s root-determinization-ensembling PR for the gold-standard version of this test: a verbatim copy of the old function, whole seeded games driven through both, move-for-move equality asserted).
2. **Validate empirically via `duels-arena`, always.** Build it in release mode (`cargo build --release -p duels-arena`). Use paired-seed, seat-swapped matches. Report Elo with a confidence interval, not just a win count. **`duels-arena experiment` runs points 2-5 of this list as one command** — candidate vs control over every (seed range × budget) cell, per-cell *and* pooled-per-budget Elo/SPRT, one machine-readable `summary.json` verdict plus each cell's raw records for later re-pooling. Prefer it over a hand-driven series of `duels-arena match` invocations; see `duels_arena::experiment` for the cost model and `--dry-run` to price a run before starting it.
3. **Test at both `Nodes` and `TimeMs` budgets.** A change that helps at a fixed node count can lose at a fixed wall-clock budget if it costs more per unit of work, and vice versa — this project has been burned by exactly that more than once. Report both.
4. **`TimeMs` runs are load-sensitive.** A benchmark run on a machine with other concurrent work (including other Claude Code agents) can swing 15+ points between runs. Run one match at a time on a quiet machine for anything you intend to trust; treat small-sample `TimeMs` results as indicative, not conclusive, until reproduced.
5. **Reproduce on a second, disjoint seed range before trusting an accept.** One seed range is not evidence.
6. **Report honest negatives.** Several real investigations in this codebase concluded "this doesn't help" (a smarter MCTS rollout policy, root-determinization ensembling at practical budgets) and shipped the attempt as a documented, non-default option rather than hiding it or forcing a marginal win into the story. Do the same. A well-documented negative result is a valid, valued deliverable here.
7. **Only change `Config::default()` when the evidence clearly supports it.**

### What we've learned about this game specifically (useful priors for future work)

- **7 Wonders Duel is a two-player zero-sum *stochastic* game with *no private information*** — both players always see the same public state; only future card reveals are unknown to both equally. One `Observation` serves both players and any spectator. This is why chance-node search (expectimax, MCTS with explicit chance nodes) applies directly — no need for anything from the imperfect-information literature (ISMCTS, CFR).
- **A static, hand-crafted position evaluation has a low ceiling in this game.** Scoring is holistic and end-game-heavy (most VP resolves only in aggregate at game end), so a few-plies-deep static eval judges positions badly — `alphabeta` with a static leaf only won ~2.5% of the time against `mcts-uct` even with 25x the search budget. Blending in an actual random playout to a real `GameResult` (instead of a static score) raised that to ~19.5%. **Simulation beats hand-crafted judgment for *position value* in this game.**
- **Win-condition awareness belongs in the search policy, not the evaluation function.** `greedy` had explicit military-race terms in its static evaluation and *still* lost to `random` by military supremacy ~10% of the time, because a 1-ply view can't see a race developing three moves out. (Both agents are retired; the measurement is a historical record and the conclusion it supports is unchanged.) `duels-strategy` exists specifically to bias *where search looks* (tree priors, rollout policy) rather than to replace simulation as the value signal — see its crate-level doc comment for the full reasoning.
- **...but a hand-crafted evaluation *blended with* a playout, as an MCTS leaf value, is the biggest win this project has measured** (`+89` Elo pooled over 3,600 games; it is now the `mcts-eval` agent, whose crate docs hold the full measurement). This refines the two priors above rather than contradicting them. A **pure** `duels-eval` leaf is far *worse* than the playout it replaces (`-171` Elo at a fixed node count), exactly as the low-ceiling prior says. Half playout and half evaluation beats both. The victory-kind breakdown says why they are complementary: the evaluation supplies civilian-score judgement (where its terms live), the playout supplies sight of military races — and pushing the blend weight past ~0.5 visibly trades the second away for the first. **When a hand-crafted signal doesn't work as a replacement for simulation, try it as a mixture before concluding it doesn't work.**
- **A large, reproducible margin over one opponent is not a strength improvement — measure it through a third party before believing it.** This is the newest and most expensive lesson here, and it is the reason `mcts-value` is registered but unrated. A *learned* leaf (`duels-value`, replacing the playout entirely, at a re-derived `c = 0.15`) measures **+91.4 Elo [+66.5, +116.3] over 800 games** against `mcts-eval` at the ladder's `nodes:2000`, and about **+140** at both `Nodes(32000)` and `TimeMs(1000)`, on disjoint seed ranges, `AcceptH1` everywhere. It looked like the largest single effect this project had ever measured. It is not: a mini round robin at `nodes:2000` found only **28% of that margin survives being measured through `mcts-uct` (+25.5 ± 27.8, [-28.9, +79.9]) and 12% through `alphabeta` (+11.2 ± 36.1, [-59.5, +82.0])**, both intervals containing zero, with a joint Bradley-Terry fit over all five records putting the pair 74 points apart where the direct match said 91.5 — real intransitivity. The victory-kind breakdown says exactly what is happening: against `mcts-uct` the candidate wins **89 science games to `mcts-eval`'s 10** while its civilian column drops correspondingly (**171 against 237**), for a total of 298 against 287. **Route substitution, not extra wins** — and the route it substitutes into is precisely the science-value miscalibration `duels-eval`'s `science_calibration` (#57) had already measured in `mcts-eval`. `duels-value`'s own `tests/probability_coherence.rs` closes the loop: the model is measurably incoherent (`P(win|One) + P(win|Two) != 1`, 43% of legal positions missing a 0.05 bound) and *worst on science-lead boards*. **A direct head-to-head against the champion answers "does this beat the champion", never "is this stronger".** The whole record is in `crates/agents/mcts-value`'s crate docs and `arena/results/experiments/p0-*`, `p1-*`, `p2-*`.
- **A blended reward changes what the exploration constant means.** Mixing a static value into a Bernoulli playout at weight `w` shrinks the reward's spread by `1 - w`, so UCB1's `c` has to be scaled by `1 - w` to leave the tuned exploration/exploitation balance alone. This is not a subtlety to discover by sweeping: `c = 0.3` alone measures at `-100` Elo and yet is strongly *positive* inside a `weight = 0.7` blend. Any future change to what a leaf backs up should re-derive `c` before measuring. **And "re-derive" sometimes means "sweep, because the formula does not apply":** a leaf that *replaces* the playout has no Bernoulli spread left to shrink, so `c = c₀(1 - w)` gives no guidance whatsoever. `mcts-value` inherited `c = 0.5` from the blend and was leaving about **83 Elo** on the table; a four-point sweep at `Nodes(32000)` (`0.10` / `0.15` / `0.25` / `0.50` measuring `+126.7` / `+140.1` / `+101.0` / `+57.2`) found a bracketed interior optimum near `0.15`. The same distinction shows up in cost: a replacing leaf runs at about `0.35x` a playout per simulation against a blend's `1.45x`, so a fixed wall clock buys it roughly three times the simulations — and the *ranking of the two leaves reverses* between a node budget and a wall-clock one.
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

**Registering an agent in `duels-server`'s `room::KNOWN_AGENTS` puts it in the web opponent picker**, with no web change required: `GET /agents` serves that list and `Home.tsx` renders whatever comes back, falling back to the raw name when its cosmetic `AGENT_LABELS` map has no entry. That is deliberate (a new agent is playable the moment the server knows about it) but it means server registration is a **user-visible** change. `mcts-value` is in the picker unlabelled for exactly this reason; giving it a label is a separate, deliberate edit, and any label should not imply it is the strongest option — it is unrated, and `mcts-eval` is still the champion.

## The leaderboard and the nightly round robin (M6b)

`arena/leaderboard.md` (rendered from `arena/leaderboard.json`) is the standing ranking of every registered agent. Both files are **generated** — `.github/workflows/nightly-arena.yml` rebuilds them every night and opens a PR; don't hand-edit them.

- **The ladder is defined in code**, in `duels_arena::leaderboard::LADDER`: each agent at its production budget (`Nodes(1)` for `phased`, the only 1-ply agent left, `Nodes(2000)` for `alphabeta`/`mcts-uct`/`mcts-eval`), default config only. Adding an agent there extends the nightly matrix automatically — the workflow never lists agent names, so `LADDER` plus `agent_registry` is the whole registration for the nightly job. **Four agents means `C(4,2)` = 6 pairings**, down from 21: `strategist` was retired first, then `random`, `greedy` and `greedy-ev` (the whole 1-ply floor tier) — see "Current state" below. `nightly-arena.yml`'s comment saying 21 is stale again; `.github/**` is a code-owner path and was left alone, as it was in the `strategist` retirement.
- **Registered is no longer the same as rated.** `agent_registry::KNOWN_AGENTS` and `LADDER` used to be pinned equal to each other; they are now pinned equal **up to `leaderboard::REGISTERED_OFF_LADDER`**, the explicit list of agents that are constructible, spec-string addressable and playable while carrying no rating. `mcts-value` is the only entry and the reason is in that constant's docs: a `+91.4` Elo margin over `mcts-eval` that does not survive a third party is not a position in a transitive ordering, and rating it would publish a number reading "strongest agent" every night. The test still fails if an agent is registered and neither rated nor named there, so nothing drifts off the board by accident — adding an entry means writing down why.
- **The whole round robin runs at one budget (`nodes:2000`)** and that is not a compromise: `phased` takes `_budget` in `Agent::choose` and never reads it, which `leaderboard::tests::one_ply_agents_ignore_their_budget` proves by playing games at both budgets and comparing every decision. That test now also asserts the `nodes:1` tier is non-empty, so it can't quietly become vacuous.
- **Ratings are fitted jointly**, not pairwise — `elo::fit_joint_elo` is a Bradley-Terry MLE over all six head-to-head records at once (MM iteration, CIs from the joint Fisher information with the anchor's row/column deleted). **`mcts-uct` is pinned at 1000**, replacing `greedy`, which was deleted with the floor tier. The anchor has to be an agent nothing else re-tunes, or every rating on the board moves when a library does — which rules out the positionally obvious `phased`, whose `Config` *is* `duels_eval::Config::default()` read live. `leaderboard::ANCHOR_AGENT`'s doc comment has the full argument; read it before moving the pin again. **Elo numbers from before that change are not comparable to ones after it**, and nothing rescales the old ones. Use `fit_elo` for a single head-to-head comparison; use `fit_joint_elo` for anything ladder-shaped.
- **`main` cannot be pushed to directly** — the `main-protection` ruleset has an empty `bypass_actors` list — so the nightly proposes a PR. GitHub does not start workflows for `GITHUB_TOKEN`-authored PRs, so that PR's required `gate` check needs a close/reopen (or a `NIGHTLY_ARENA_TOKEN` PAT secret) before it can merge. The workflow says so in the PR body.
- **`ai-candidate` is informational and must stay that way for now** (an explicit decision). It is a separate workflow file precisely so it cannot drift into `ci.yml`'s `gate` job. Promoting it to blocking means two deliberate edits: add it to `gate`'s `needs:` *and* to the ruleset's required-status-check list.
- **The champion is a plain constant** (`leaderboard::CHAMPION`, currently `mcts-eval` at `Nodes(2000)`), not something read back out of the leaderboard. Automated promotion is M7 and does not exist yet; until it does, a human changing one line is the honest mechanism. `CHAMPION` was moved from `mcts-uct` to `mcts-eval` once the latter measured ~+100 Elo stronger and confirmed as the top of the ladder — a separate one-line decision on its own evidence, not a side effect of adding an agent. **`mcts-value` did not move it, and the contrast is the point:** it measures further ahead of `mcts-eval` than `mcts-eval` ever measured ahead of `mcts-uct`, and it is still not the champion, because "beats the champion by a lot" and "is the strongest agent" turned out to be different claims here (see the prior above). `leaderboard::tests::a_complete_round_robin_builds_and_ranks_strongest_first` asserts the champion is computed by matching `CHAMPION.agent` against the ladder, not derived from rank — a property that held when the two intentionally differed and still holds now that they coincide.

## Current state

See `docs/milestones.md` for the milestone table, the four human checkpoints, and what's
actively being worked on. Update that file, not this section, as things change — this
avoids keeping two summaries in sync. `docs/adr/` has the original architecture decisions
and their rationale.
