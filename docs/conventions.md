# Project conventions, invariants and priors

The standing rules this codebase is built around, and the things it has learned
about 7 Wonders Duel by measuring them. Doc comments across the workspace cite
this file by section; it exists so those citations have somewhere to point.

This is **not** an architecture summary — `README.md` has the crate map,
`docs/milestones.md` is the source of truth for "where are we",
`docs/agent-contract.md` is the `Agent`/`Observation` API contract,
`docs/adr/` holds the original architecture decisions, and
`docs/rules-spec.md` is the rules traceability index. What is here is only the
material that other files' comments needed a stable home for.

Historical measurements are kept in the past tense and are labelled where the
agent that produced them has since been retired. A retired agent's number is
still evidence for the conclusion it supports.

## The layering, and why agent crates are self-contained

The dependency order is **`duels-core` → `duels-strategy` → `duels-eval` /
`duels-value` → agents**, each layer depending only on the ones above it.

- **`duels-core` is the only rules authority.** No other crate — not an agent,
  not the server, not the web client — reimplements legality, cost, effects or
  scoring. If you need a rule, add an accessor to `duels-core`; do not
  approximate it elsewhere.
- **Agent crates are self-contained.** No agent crate depends on another agent
  crate, even when it would save duplicating a search. This is deliberate: it
  lets agents be built in parallel without cross-crate coordination, and it
  means benchmarking one never risks silently coupling to another's internals.
  The duplication that follows is an accepted, intentional cost —
  `crates/agents/mcts-eval` carries a verbatim copy of `mcts-uct`'s
  tree/chance/rollout machinery, and `mcts-value` a copy of `mcts-eval`'s.
  Each copy is *checkable* rather than merely asserted: the copying crate keeps
  a frozen second copy of the copied agent's search as an ablation control and
  asserts move-for-move equality against it, so "the new leaf beats the old
  one" is a claim about two agents and not about two similar searches.
- **Code two agents should share moves *down* into a library, never sideways.**
  `duels-eval` exists because `phased`'s evaluation was wanted as a search leaf
  value too; `duels-value` exists for the same reason one layer over. A library
  at that level carries obligations the agent above it does not: it depends
  only on the layers above it (no `rand`, no clock, no `duels-agents-api` — its
  entry points are pure functions), it holds the determinization-invariance and
  version-snapshot identity tests for everything it owns, and `duels-eval` is a
  **mandatory-review path in `CODEOWNERS`**, because more than one agent's
  measured strength moves when it does.
- **The one exception is a dev-dependency.** `crates/agents/random` is a
  retired agent kept as a shared test fixture — the uniform-random correctness
  floor several agent crates' test suites measure against. It is never linked
  by a shipping binary and is registered in neither `duels-arena`'s
  `agent_registry` nor `duels-server`'s `room`. Its unreachability from the
  real registries is the point, not an oversight.
- **Determinism is enforced by lint, not discipline.** `clippy.toml` bans
  `Instant::now`, `SystemTime::now`, `rand::thread_rng` and `rand::random`
  inside `duels-core` and every agent crate. Randomness enters only through an
  explicitly passed, seeded `StdRng`.

## Hidden information, and the determinization-invariance test

`GameState` holds hidden information (deck order, face-down identities);
`Observation` never does. The separation is enforced by the type system rather
than by convention — see `docs/agent-contract.md` for the types themselves.

Every `Agent` implementation, and every function in `duels-strategy`,
`duels-eval` and `duels-value`, must be provably invariant to *which* hidden
information sample produced the concrete state it was handed. The established
shape of that proof is a **determinization-invariance property test**: sample
two different concrete `GameState`s from the same `Observation` via
`Observation::sample_state` and assert the results are equal **bit for bit**
(`to_bits()` on every float). This project writes one for any new logic that
touches game state. If you cannot write that test, the logic is leaking hidden
information somewhere.

## Validating an agent change

1. **A new capability is an opt-in `Config`; the old behaviour stays the
   default**, or stays reachable as an explicit, proven-identical option. Never
   silently change what an agent does. When adding a mode or parameter, write a
   test proving the new option at its "off" value is **bit-identical** to the
   pre-existing path — a verbatim copy of the old function, whole seeded games
   driven through both, move-for-move equality asserted.
2. **Validate empirically through `duels-arena`, always**, built in release
   mode (`cargo build --release -p duels-arena`). Paired seeds, swapped seats.
   Report Elo with a confidence interval, not a bare win count.
   **`duels-arena experiment` runs points 2-5 of this list as one command** —
   candidate against control over every (seed range × budget) cell, per-cell
   *and* pooled-per-budget Elo and SPRT, one machine-readable `summary.json`
   verdict plus each cell's raw records for later re-pooling. Prefer it to a
   hand-driven series of `duels-arena match` invocations; see
   `duels_arena::experiment` for the cost model, and `--dry-run` to price a run
   before starting it.
3. **Test at both `Nodes` and `TimeMs` budgets.** A change that helps at a
   fixed node count can lose at a fixed wall-clock budget if it costs more per
   unit of work, and vice versa. This project has been burned by exactly that
   in both directions. Report both.
4. **`TimeMs` runs are load-sensitive.** A run on a machine doing other
   concurrent work can swing 15+ Elo between repeats. Run one match at a time
   on a quiet machine for anything you intend to trust, and treat small-sample
   `TimeMs` results as indicative until reproduced. The usual proxy — "load
   average in single digits" — does not work here: `duels-arena` parallelises
   seeds within a match, so one legitimate match can show a load average in the
   hundreds. Snapshot the actual process list instead.
5. **Reproduce on a second, disjoint seed range before trusting an accept.**
   One seed range is not evidence. This applies to a fitted constant exactly as
   it applies to an Elo.
6. **Report honest negatives.** Several real investigations here concluded
   "this does not help" and shipped the attempt as a documented, non-default
   option rather than forcing a marginal win into the story. A well-documented
   negative result is a valid, valued deliverable.
7. **Only move `Config::default()` when the evidence clearly supports it.**
8. **Size the sample and the SPRT bound to the effect you actually expect, not
   to habit.** This project's `elo1 = 20`, 400-games-per-cell default was
   calibrated for its early, larger-jump rounds. As the ladder strengthens,
   single-iteration gains are expected to shrink, and a sample sized for a
   20-point jump reads a genuine ~30-point one as "inconclusive" — which is a
   test underpowered for the effect, not evidence the effect isn't there. When
   a first-pass result comes back inconclusive at the historical default and
   the point estimate still looks meaningful, re-run at a larger sample (as a
   rough default, ~2,000 games) and a tighter bound (`elo1` around 10) before
   concluding "no effect" either way. `duels-value`'s `v2` retrain
   (`crates/duels-value/src/lib.rs`, "Recalibrating the test, not just the
   weights") is the worked example: a 400-game cell read Inconclusive, and a
   2,000-game re-run at `elo1 = 10` confirmed a real `+34.5` Elo effect that
   was there all along.

`duels-eval`'s round history in `docs/eval-rounds/` is ten worked examples of
this protocol, honest negatives included.

## Priors about this game

Each of these was measured, and several are cited from the code that acts on
them.

- **7 Wonders Duel is a two-player zero-sum *stochastic* game with *no private
  information*.** Both players always see the same public state; only future
  card reveals are unknown, and unknown to both equally. One `Observation`
  serves both players and any spectator. This is why chance-node search
  (expectimax, MCTS with explicit chance nodes) applies directly, with no need
  for anything from the imperfect-information literature (ISMCTS, CFR).
- **First-player advantage is real and large** — about **67/33** between
  equally strong `mcts-uct` configurations at equal budget. Never compare
  agents without paired, seat-swapped matches.
- **A static, hand-crafted position evaluation has a low ceiling in this
  game.** Scoring is holistic and end-game-heavy, so a few-plies-deep static
  evaluation judges positions badly: `alphabeta` with a static leaf won ~2.5%
  against `mcts-uct` even with 25x the search budget, and blending in an actual
  random playout to a real `GameResult` raised that to ~19.5%. **Simulation
  beats hand-crafted judgement for position value in this game.**
- **Win-condition awareness belongs in the search policy, not the evaluation
  function.** The retired `greedy` had explicit military-race terms in its
  static evaluation and *still* lost to `random` by military supremacy ~10% of
  the time, because a 1-ply view cannot see a race developing three moves out.
  Both agents are retired; the measurement is a historical record and the
  conclusion it supports is unchanged. `duels-strategy` exists specifically to
  bias *where search looks* — tree priors, rollout policy — rather than to
  replace simulation as the value signal.
- **...but a hand-crafted evaluation *blended with* a playout, as an MCTS leaf
  value, was the biggest single win this project had measured** at the time
  (`+89` Elo pooled over 3,600 games; it became the `mcts-eval` agent, whose
  crate docs hold the measurement). This refines the two priors above rather
  than contradicting them: a **pure** `duels-eval` leaf is far *worse* than the
  playout it replaces (`-171` Elo at a fixed node count), exactly as the
  low-ceiling prior says, and half playout plus half evaluation beats both. The
  victory-kind breakdown says why they are complementary — the evaluation
  supplies civilian-score judgement, the playout supplies sight of military
  races. **When a hand-crafted signal does not work as a replacement for
  simulation, try it as a mixture before concluding it does not work.**
- **A *learned* value of the same shape is a different matter, and the sharpest
  result on this list.** A pure learned leaf measures `+57` where the pure
  hand-crafted one measures `-171`: the low ceiling is a ceiling on
  *hand-crafted* judgement, not on a static leaf as such. That leaf is
  `duels-value`, read by the `mcts-value` agent; both crates' docs hold the
  measurement record.
- **A blended reward changes what the exploration constant means.** Mixing a
  static value into a Bernoulli playout at weight `w` shrinks the reward's
  spread by `1 - w`, so UCB1's `c` has to be scaled by `1 - w` to leave the
  tuned exploration/exploitation balance alone. This is not a subtlety to
  discover by sweeping: `c = 0.3` alone measures at `-100` Elo and is yet
  strongly *positive* inside a `weight = 0.7` blend. **Any change to what a
  leaf backs up should re-derive `c` before measuring** — `mcts-value` skipped
  that at first and left about 83 Elo on the table, more than every other
  refinement in that crate combined.
- **`duels-strategy`'s reads are genuinely not free** — about 17-29% of one
  MCTS rollout for a full slate of action priors on a real position. Cheap
  enough to compute once per search-tree node, too expensive to recompute per
  simulation. Cache it. `duels_eval::Root::new` is a slate of exactly those
  reads and inherits the same rule.
- **Whether a search that consumes a value library should *pin* a generation
  depends on whether the value is incidental to the agent or *is* the agent.**
  An agent whose identity is its *search*, and whose measured strength merely
  depends on a value input, should pin a frozen snapshot plus a golden-values
  test — a later retune then fails a test rather than silently re-defining what
  was measured. `mcts-value` is that case for `duels-value`. `mcts-eval` is the
  deliberate opposite call for `duels-eval`: its whole reason to exist is
  "`duels-eval` inside a search", it is meant to strengthen automatically as
  future rounds land, so it reads `duels_eval::Config::default()` live and pins
  nothing. It pays for that by recording the whole
  `duels_eval::Config::params_string()` in its `AgentSpec`, which makes two
  results files from either side of a round distinguishable rather than making
  the older one uninterpretable. Read that crate's "Tracking `duels-eval` live"
  section before adding a pin to it; the absence of one is not an oversight.

## Testing conventions

- **`duels_core::testing::StateBuilder`** is this repository's established way
  to say exactly what a position is. Use it to construct hand-built positions
  rather than driving a game from scratch when a test needs a specific scenario
  ("one move from military supremacy"). Its own docs are explicit that it
  performs no rules validation and can build states a real game would never
  reach — which is a feature for a unit test and a trap for anything asserting
  a property of *reachable* positions.
- **`proptest`** for randomized invariant checking across many played-out games
  (card conservation, coins never negative, `Observation` never leaking a
  hidden identity). `duels-core/tests/properties.rs` is the established style.
- **Keep large-N benchmark-shaped runs out of the default `cargo test` path** —
  hundreds of games, release-mode timing — so CI stays fast. Use `#[ignore]`
  with a reason, or a separate `examples/` binary.
  `duels-strategy/examples/watch_reads.rs` is the pattern for a human-readable
  diagnostic tool.
- **Every non-trivial rule `duels-core` implements has a numbered `R-xxx` entry
  in `docs/rules-spec.md`** naming the test that covers it. If you touch rules
  logic, update that file. `data/README.md` documents the same discipline for
  the factual game data and flags what has been spot-checked against the
  physical game versus taken best-effort.
