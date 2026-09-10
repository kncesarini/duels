# 0006. Arena live: a jointly-fitted leaderboard, a nightly round robin, and a non-blocking `ai-candidate` check

## Status

Accepted at M6b (#35). **All five decisions still stand.** The roster figures
the text below quotes do not: the ladder was seven agents when this was
written and is five now, so the numbers in the body are a record of the
decision's context, not the current state. What moved, and where the live
answer is:

- **The anchor is `mcts-uct` pinned at 1000, not `greedy`** (§1). `greedy` was
  deleted with the whole 1-ply floor tier (`random`, `greedy`, `greedy-ev`) in
  #60, which also moved the pin. The anchor has to be an agent nothing else
  re-tunes — read `leaderboard::ANCHOR_AGENT`'s doc comment before moving it
  again. **Elo numbers from before that change are not comparable to ones
  after it**, and nothing rescales the old ones.
- **There is one 1-ply agent, `phased`, not five** (§2). The `Nodes(2000)`
  tier is `alphabeta`, `mcts-uct`, `mcts-eval` and `mcts-value`. The decision
  — one budget for the whole round robin, proved rather than asserted by
  `leaderboard::tests::one_ply_agents_ignore_their_budget` — is unaffected,
  and that test now also asserts the `nodes:1` tier is non-empty so it cannot
  quietly become vacuous.
- **The champion is `mcts-value`, not `mcts-uct`** (§5). It moved to
  `mcts-eval` in #51 and to `mcts-value` in #63, each a separate one-line
  decision on its own evidence. `leaderboard::CHAMPION` is the live answer.
- **The round robin is `C(5,2)` = 10 pairings, not 21** (Consequences); a
  sixth agent would take it to 15. `duels_arena::leaderboard::pairings()` is
  the live answer, and no workflow file hard-codes the count.
- `ai-candidate` also runs on `crates/duels-eval/**` now (§4), because every
  agent built on that library moves when it does.

`duels_arena::leaderboard::LADDER` is the source of truth for the roster in
all cases. This section is the drift record; the body is left as it was
accepted.

## Context

Milestone M6b of the original architecture pass was specified as "real agents,
leaderboard, nightly workflow, `ai-candidate` gate", with the executive summary
calling for an "Arena with paired-seed games, Bayesian-Elo leaderboard, and SPRT
gating; nightly round-robin in CI; promotion of a new champion happens via an
automated PR that the project owner approves."

By the time it was built, the arena skeleton (`duels-arena`: paired-seed
matches, pairwise logistic Elo, SPRT, spec-string agent configs) already
existed, and seven agents were registered. What was missing was the standing
ranking, the schedule that keeps it current, and the PR-time signal.

Four choices in building it were non-obvious enough to record.

## Decision

### 1. Ratings are fitted jointly, not by anchoring each agent independently

`elo::fit_elo` estimates one head-to-head rating difference. A leaderboard over
`n` agents could be built by running each agent against a single fixed
reference and reporting those differences — which is what "BayesElo anchored at
greedy-v1 = 1000" could be read to mean, and is a legitimate simplification.

We rejected it. For a 7-agent round robin it uses only the games each agent
played against the reference, discarding five sixths of the evidence about
every agent, and it can produce a table that is not self-consistent (A outrates
B on the reference axis while losing their head-to-head). Since the nightly
plays the *whole* round robin anyway, the extra information is already paid
for.

`elo::fit_joint_elo` therefore fits all ratings simultaneously: a
Bradley-Terry MLE by the classical MM (Zermelo) iteration, with the same weak
symmetric prior `fit_elo` uses so a clean sweep still yields a finite estimate,
and confidence intervals from the joint observed Fisher information with the
anchor's row and column deleted. `greedy` is pinned at 1000, keeping the
original design's scale.

### 2. The round robin runs at one budget, and that is not a compromise

The ladder records each agent at its production budget — `Nodes(1)` for the
five 1-ply agents, `Nodes(2000)` for `alphabeta` and `mcts-uct` — but every
pairing is actually played at `nodes:2000`. The five 1-ply agents take
`_budget` in `Agent::choose` and never read it, so the two are the same agent.
Rather than assert that, `leaderboard::tests::one_ply_agents_ignore_their_budget`
plays whole games at both budgets and compares every decision.

This avoids adding per-side budgets to the `match` CLI for a distinction that
does not exist.

### 3. The nightly proposes a pull request because it cannot do anything else

The `main-protection` ruleset requires a pull request, requires the `gate`
status check, and has an empty `bypass_actors` list — so no actor, the
workflow's own token included, can push to `main`. A PR is the only available
mechanism, not a stylistic preference.

One consequence is worth stating plainly: GitHub deliberately does not trigger
workflows from events authored by `GITHUB_TOKEN`, so a nightly PR opened with
the default token will not have `gate` running on it and cannot be merged until
someone closes and reopens it. The workflow says so in the PR body, and reads
an optional `NIGHTLY_ARENA_TOKEN` secret (a PAT) to avoid the problem where one
is configured. We did not create such a secret as part of this work; adding a
long-lived credential to CI is the maintainer's call, and ADR 0004's reasoning
about keeping CI free of secrets applies.

### 4. `ai-candidate` lands informational, not blocking

The original spec called it a "gate". It ships as an informational check: it
runs on every PR touching `crates/agents/**` or `crates/duels-agents-api/**`,
measures the changed agent against the designated champion, and reports the
Elo delta as a PR comment and a job summary — but its CI status is wired into
nothing.

Two things enforce that structurally rather than by convention: it lives in its
own workflow file (so it cannot accumulate into `ci.yml`'s `gate` job by
accident), and `gate` remains the only required status check on `main`.

The reason is calibration, not caution about automation in general. A CI-budget
match is a few hundred games; this project's own standard for an accept/reject
decision is ~3,000 games reproduced on a second disjoint seed range. Gating on
the former would either block correct changes on noise or, if the threshold
were loosened enough not to, gate on nothing. Making it blocking later is a
deliberate two-part change: add it to `gate`'s `needs:` *and* to the ruleset's
required-check list.

### 5. The champion is a constant, not a computed value

`leaderboard::CHAMPION` is a plain designation (`mcts-uct` at `Nodes(2000)`),
not something read back out of the leaderboard. Automatic champion promotion is
M7 and does not exist; until it does, a human changing one line is the honest
mechanism, and a constant makes "what are we measuring against, and who decided"
answerable from the source.

## Consequences

- `arena/leaderboard.md` and `arena/leaderboard.json` are generated artifacts
  that are nonetheless committed — a leaderboard nobody can see is not one.
  `.gitignore` ignores `/arena/results/` (the raw per-run records) and
  deliberately not these.
- Adding an agent means adding it to `leaderboard::LADDER`; the nightly matrix,
  the `C(n,2)` pairing list, and the leaderboard all follow. No workflow file
  names an agent.
- The nightly's cost scales as `C(n,2)`: an eighth agent takes the round robin
  from 21 pairings to 28. That is the accepted price of the full-rigor-nightly
  decision; the matrix means it costs wall-clock parallelism rather than a
  longer job.
- A failed matrix job produces an incomplete round robin, which the aggregation
  step rejects rather than fitting — no leaderboard is published that night,
  and the failure is visible.
- The joint fit requires a connected comparison graph. A full round robin is
  connected by construction, but a future partial schedule (skipping pairings
  to save time) would have to preserve that; `fit_joint_elo` errors rather than
  silently producing ratings on incomparable scales.
