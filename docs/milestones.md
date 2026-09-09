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
| **M2b** Server + playable | axum rooms, WebSocket, `random` agent, e2e | ✅ Done — `random` has since been retired from the roster (see Current work); the e2e plays against `phased` |
| **M3** Classical AIs | `greedy`, `alphabeta` | ✅ Done — plus `greedy-ev`, `strategist`, `phased` beyond original scope; `greedy`, `greedy-ev` and `strategist` have since been retired (see Current work) |
| **M4** MCTS | `mcts-uct` with chance nodes | ✅ Done — `mcts-eval` (below) has since surpassed it as the strongest agent measured |
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

**Reusing `phased`'s evaluation inside search** — per the original stated plan ("strong
hand-crafted eval, then reuse it as a leaf/value function in MCTS, then take a stab on
ML"). Done, then promoted further than originally scoped:
- PR 0 — extract `phased`'s evaluation into a shared `duels-eval` crate (pure refactor). ✅ Done (#36).
- PR 1 — `LeafValue` config in `mcts-uct` (static/truncated/blend leaf evaluation using
  `duels-eval`, per-age temperature calibration), off by default. ✅ Done (#38) — the largest
  single Elo gain measured in this project (+89 Elo pooled, 3,600 games), but shipped
  experimental because the winning config also rescales the search's exploration constant.
- Promoted to its own standing agent, **`mcts-eval`** (#40), rather than left as an opt-in
  flag on `mcts-uct` — it's now the strongest agent on the leaderboard. Unlike `mcts-uct`'s
  old pin, it tracks `duels-eval`'s live default, so it gets stronger automatically as future
  `phased` rounds land, with no manual version bump. `mcts-uct` itself is back to exactly its
  pre-leaf-value behavior (bit-identical, proven).
- PR 2 (optional) — a `duels-eval`-priced rollout policy for `mcts-eval`, only if there's
  appetite; not started.

**Agent roster** — the whole 1-ply floor tier is retired. `strategist` went first (its
research question, whether `duels-strategy`'s prior helps `greedy-ev`, was answered:
statistically indistinguishable), and then `random`, `greedy` and `greedy-ev`, on the
project owner's explicit decision, for measured strength far below the rest of the roster:
the last full refit had all three within a 200-Elo band, scoring 0.0%–0.5% against every
top-half agent, so they cost the nightly fifteen of its twenty-one pairings and told it
nothing. **This supersedes the earlier note here that `random` and `greedy` were staying
as the anchor and the easy end of the opponent picker** — that policy is withdrawn, not
misread. Each was removed from `LADDER`, `agent_registry`/`duels-server::room`'s
`KNOWN_AGENTS`, the web UI's opponent picker, and `agent_spec`'s parameter parsers.
`greedy` and `greedy-ev`'s crates are deleted outright; **`crates/agents/random` survives
as a test-only fixture** (see "the one crate that stayed" below). The ladder is now four
agents and six pairings: `phased`, `alphabeta`, `mcts-uct`, `mcts-eval`
(`leaderboard::CHAMPION`).

**The Elo anchor moved from `greedy` to `mcts-uct`, and the scale changed with it.**
Deleting `greedy` removed `ANCHOR_AGENT` entirely, so the joint Bradley-Terry fit needed a
new pin. The rationale in `leaderboard::ANCHOR_AGENT`'s own docs is what decided it: the
anchor must be a *never-changing* baseline, because every other agent's rating then moves
only when that agent's strength moves. The positionally obvious replacement is `phased`
(weakest survivor, 1-ply, budget-invariant), and it is the wrong one — `phased`'s `Config`
*is* `duels_eval::Config`, read live from `Config::default()`, and `duels-eval` is re-tuned
in numbered rounds (ten so far, the tenth landing in #56). Anchoring there would shift every
rating on the board on every tuning round. `mcts-uct` does not depend on `duels-eval` at
all, its default `PriorMode::None` does not consult `duels-strategy` either, its
`Config::default()` is frozen and guarded by `mcts-eval`'s move-for-move ablation control,
and it is already this project's canonical yardstick. `ANCHOR_ELO` stays at 1000; only
*which* agent sits there changed. Consequence, stated plainly: **every Elo number in
`arena/leaderboard.md`/`.json` predating this was measured against `greedy` = 1000 and is
not comparable to anything measured after it.** The next nightly round robin refits from
scratch against the new anchor; nothing rescales the old numbers, and with the anchor now
second of four rather than second-from-bottom of seven, ratings below 1000 are expected.
Those two files were left as the nightly last generated them (they are generated artifacts,
and #55 set the same precedent) — they will be stale, listing retired agents, until that
run lands.

**The one crate that stayed, and why.** `crates/agents/random` is retired from the roster
but not deleted: a uniform-random opponent is the yardstick four surviving crates measure a
correctness floor against ("a search agent that does not comfortably beat a random player
has a bug, not bad luck") — `alphabeta`'s `tests/vs_random.rs`, `mcts-uct` and `mcts-eval`'s
in-crate `beats_a_random_opponent` tests and `vs_random` example, `phased`'s
`phased_convincingly_beats_random`, `duels-arena`'s `age_start_policy` wrapper tests (which
need a cheap *stateful, RNG-consuming* inner agent and scan up to 200 whole games), and
`duels-strategy`'s `watch_reads` example, which sits below `duels-eval` in the layering and
so cannot dev-depend on any surviving agent without closing a cycle. Those assertions are
*about* a random baseline; substituting a stronger agent would not make them stricter, it
would make them mean something else. It is now a **dev-dependency everywhere** — no shipped
binary links it, `make_agent("random")` and the spec string `random` are both errors, and
`agent_registry::tests::the_retired_agents_are_retired` pins that. Treat it as a fixture in
the same class as `duels_core::testing::StateBuilder`. Two knock-on losses worth knowing:
the web UI's easy end is now `phased`, and `duels-strategy`'s `watch_reads` diagnostic
drives both seats randomly instead of `greedy` vs `random`.
