# Milestones

Living reference. Update this file whenever a milestone or checkpoint changes status —
at a merge that finishes one, when scope on an open one shifts, or when a new one is
added. This is the source of truth for "where are we"; `CLAUDE.md`'s architecture
section should stay pointed at this file rather than duplicating a summary.

The table and the four checkpoints below are the original delivery plan from this
project's kickoff Architect design pass (see `git log` around the M0-M3 PRs for the
original context). "Current work" tracks the granular threads underneath M3/M4 that the
milestone table is too coarse to show.

## Milestone table

| Milestone | Content | Status |
|---|---|---|
| **M0** Scaffold | workspace, CI, contracts, ADRs | ✅ Done |
| **M1** Rules engine | `duels-core`: legality, cost, effects, scoring, property tests | ✅ Done |
| **M2a** UI shell | React client, all screens | ✅ Done (rebuilt to the table design, PR #28) |
| **M2b** Server + playable | axum rooms, WebSocket, `random` agent, e2e | ✅ Done |
| **M3** Classical AIs | `greedy`, `alphabeta` | ✅ Done — plus `greedy-ev`, `strategist`, `phased` beyond original scope |
| **M4** MCTS | `mcts-uct` with chance nodes | ✅ Done — current search champion |
| **M6a** Arena skeleton | runner, paired seeds, Elo/SPRT | ✅ Done |
| **M6b** Arena live | real agents, leaderboard, nightly workflow, `ai-candidate` gate | ✅ Done (PR #35) — nightly opens a PR that needs a manual close/reopen to trigger `gate` (deliberate: avoids adding a PAT secret, keeping ADR 0004's CI-stays-secret-free stance) |
| **M5** RL pipeline | PyO3 bindings, self-play, ONNX, training loop, `mcts-valuenet`/`mcts-nn` | ❌ Not started — deferred; see Current work below for the bridge step happening first |
| **M7** Promotion automation | `promote.yml` bot PR, champion epoching | ❌ Not started |
| **M8** Polish | replay scrubber, AI-eval panel, AI-vs-AI spectating, remote human-vs-human, hosting | ❌ Not started |

## Human checkpoints

- **CP1** (after M2b): the project owner plays several full games and signs off on rules fidelity, before anything in the AI track is trusted. **Still open** — the rebuilt web UI (PR #28) has not yet been hands-on tested. The stack is kept current at `http://localhost:4173/` for this.
- **CP2** (after M6b): review the leaderboard, decide RL investment/timing. Resolved informally rather than as a scheduled checkpoint — the decision was to build a strong hand-crafted evaluation first (`phased`), reuse it inside `mcts-uct`, and defer self-play RL until after that.
- **CP3** (every champion promotion): no formal promotion mechanism exists (that's M7); every merge has so far been an ad hoc CP3.
- **CP4** (hosting decision, before M8): not reached.

## Current work (the granular thread under M3/M4)

**`phased` evaluation strength** — the dominant thread since M4. Rounds shipped: continuous
commitment blend (#25), two eval fixes + three forward terms (#27), terminal rails + honest
military model (#29), wonder pending-effect fix + 7-wonder cap bug (#31), guild pricing +
yellow-card density (#32), extra-turn wonder premium (#33). `docs/strategy-backlog.md`
tracks the remaining unimplemented items (token-specific valuations, draft-phase menu
coherence, and others still open).

**Reusing `phased`'s evaluation inside `mcts-uct`** — the current active effort, per
the original stated plan ("strong hand-crafted eval, then reuse it as a leaf/value
function in MCTS, then take a stab at ML"). Sequenced as:
- PR 0 — extract `phased`'s evaluation into a shared `duels-eval` crate (pure refactor). ✅ Done (#36).
- PR 1 — `LeafValue` config in `mcts-uct` (static/truncated/blend leaf evaluation using
  `duels-eval`, per-age temperature calibration). 🔄 In progress.
- PR 2 (optional) — a `duels-eval`-priced rollout policy, only if PR 1 leaves appetite.

**Agent roster** — retiring `strategist` (its research question, whether `duels-strategy`'s
prior helps `greedy-ev`, was answered: statistically indistinguishable). Approved, not yet
executed. `random` and `greedy` are staying — `greedy` is the Elo leaderboard's anchor, and
both serve as the easy end of the web UI's opponent picker.
