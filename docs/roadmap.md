# AI roadmap

Living reference for the next phase of AI work toward this project's stated goal —
build the best possible AI opponent. Written 2026-09-10 from an explicitly
unconstrained architect planning pass (the project owner's framing: think freely,
don't be limited by what this repo's code or docs currently say or by decisions
already made). Update this file as tiers land or the plan changes; do not let a
plan like this live only in chat history where it can be forgotten.

Cross-referenced from `docs/milestones.md`'s "Current work" section (M5, the RL
pipeline milestone).

## The direct question, and the direct answer

The project owner asked: is it time for iterative (Leela-Zero-style) self-play
training, feature/architecture refinement, a learned policy output, or something
else?

**Yes, it is time to run the loop — but three prerequisites come first, and one
piece of the framing needs correcting before any of it makes sense.**

This project has already run exactly one iteration of the AlphaZero-style loop
without quite naming it that: `v1.bin` → `v2.bin` was generate-with-the-current-net,
fit-to-outcomes, promote-on-a-real-battery, and it worked (+34.5 Elo, confirmed at
2,000 games). What it has not done yet is fix the three things that make a second
and third iteration risky rather than just repeating the first: the generator has
no exploration, the training target is needlessly high-variance, and the promotion
machinery (a hash-pinned golden test) is built to resist retrains rather than
absorb a series of them.

## Two corrections to the framing

1. **7 Wonders Duel is not imperfect-information in the ISMCTS/CFR sense.**
   Nothing is hidden from one player but not the other — every unknown (face-down
   card identities, future deck composition) is unknown to *both* players equally,
   which makes it a chance event, not an information set. The existing chance-node
   MCTS (`crates/agents/mcts-value/src/tree.rs`) is already the right shape for
   this — the backgammon/Stochastic-MuZero setting, not the poker one. A PUCT policy
   prior at decision nodes and root visit counts as a policy training target are
   **directly valid** here; "a naive policy-network port assumes perfect
   information" is not a real objection in this codebase. The one real residue of
   determinization: the next age's deck and the undrafted wonder pool are fixed per
   search (`Observation::sample_state`, `crates/duels-core/src/observation.rs`),
   which biases play near age boundaries but does not corrupt outcome labels — a
   `duels-core` fix (expose the age deal as its own chance event) is real but
   lower-priority (Tier 4).
2. **"Never train the value net against the search's own opinion" was a finding
   about a different mixture, and it does not transfer as-is.** The original
   finding (an older investigation, a hand-crafted evaluator blended into a
   search) was that fitting the evaluator to a search whose leaf was half that same
   evaluator collapsed the complementarity that made the blend work. That doesn't
   describe this situation: 2,000 nodes of rule-following lookahead over real
   chance draws *is* fresh information the net didn't have — the standard
   policy-improvement operator every expert-iteration system runs on. The real risk
   is feedback drift, and the standard control is anchoring the target on the real
   outcome (`z`), i.e. a blend `λ·z + (1−λ)·q_root`, not avoiding search-derived
   targets altogether.

## Tier 0 — this week, no ML risk, direct Elo

**A. Fix the value net's forward-pass layout.** Measured in the planning pass
(scratch benchmark, not yet landed): transposing `net.rs`'s `w1` and accumulating
axpy-style instead of as 211 row-major dot products took the forward pass from
**7.45 µs to 1.29 µs — a 5.8x speedup** — same output to within summation-order
tolerance. This is the dominant per-simulation cost for `mcts-value`'s leaf, so at a
fixed wall-clock (`TimeMs`) budget this is 3-5x more simulations for free. By the
budget-scaling curve already measured (PR #68, ~+43 Elo per doubling in the
`Nodes(2000)`-`Nodes(8000)` range), this is plausibly **+60 to +100 Elo at
production's actual `TimeMs(1000)` budget**, and it makes every future corpus
generation and arena run 3-5x cheaper too — everything downstream of this gets
easier the moment it lands. Land it as a new `Summation` variant (the convention
already exists for "not bit-identical, keep the old order reachable"). **This is
the very next concrete action** — implement, benchmark with
`examples/value_bench.rs`, then a `TimeMs(1000)` cell against the current default
on a quiet machine.
**B. Offline symmetrization check (~30 minutes).** `tests/probability_coherence.rs`
already shows `P(win|One)` and `1 − P(win|Two)` disagree by ~0.056 on average.
Averaging two disagreeing views of the same position is a free two-member
ensemble. Score `(p_One + 1 − p_Two)/2` on the held-out test rows with `v2.bin`; if
Brier improves, ship it as a new leaf variant (2 forward passes — 2.6 µs after A)
and arena-test it. If it doesn't help offline, the coherence defect needs a
training-time fix instead (Tier 2-J), not an inference-time patch.
**C. Cross-platform determinism.** UCB1/progressive-widening call `f64::ln`/`powf`
from the platform libm; ARM glibc and Apple's libm can differ in the last ulp, so
the same seed could in principle diverge between a Mac and a Raspberry Pi.
Corpus *replay* (`seed, actions`) is unaffected; corpus *regeneration* is not.
Switch those calls to the `libm` crate. This is also a prerequisite for mixing
Pi-generated and Mac-generated corpora with a straight face (see the fleet section
below).

## Tier 1 — make the loop real

Three prerequisites, then run three generations and see whether the gain per
generation holds or decays.

**D. Put exploration into the generator.**
`crates/duels-arena/examples/value_corpus_mv.rs` currently plays the argmax-visit
move from both seats with the same weights (its own `--verify` flag asserts
"non-argmax action" as an *error*). All diversity currently comes from the deal
alone. This — not corpus size — is why unconstrained `mcts-value` self-play
collapsed to ~10% scientific-supremacy games and why the `v2` round needed a
hand-sized "insurance batch" mixed in by hand. Standard fix: sample the root move
from `visits^(1/τ)` for the first ~10-15 plies (τ = 1), argmax after; add a
sparring mix using the three specialist agents
(`mcts-value:objective=science/military/civilian`, already built and validated) as
a fraction of one seat, since they already play recognizably different games.
Record the sampled action; training targets stay valid either way.
**E. Change the training target.** `tools/train_value.py` currently fits to the
one-hot outcome only. Add a `--value-target-lambda` option: value target
`λ·z + (1−λ)·q_root` for the aggregate win probability (the corpus already records
`value` per decision node), keeping the four-way victory-kind decomposition on the
real outcome `z`. Ablate `λ ∈ {1.0, 0.5}` **by arena result, not offline Brier** —
offline Brier against `z` is structurally biased toward `λ = 1`. Prediction (not
yet measured): `λ ≈ 0.5` wins and pushes the training peak epoch later than the
current epoch 2-5 — both fits so far peaked implausibly early for 7M rows and ~27k
parameters, which reads as label noise (a single Bernoulli outcome shared by ~60
correlated rows per game), exactly what a lower-variance target should fix.
**F. Replace the retrain brake with a generations registry.**
`crates/agents/mcts-value/src/golden.rs` currently pins the embedded weights'
content hash so any retrain fails a test outright — the right instinct when the
leaf was a one-off artifact, actively hostile to a real generate-train loop. Make
generations first-class data (`v1, v2, v3, …`, each with its corpus manifest,
training args, and full promotion-battery result recorded), regenerate the golden
table with a tool, and pin "the default is whichever generation the registry
designates champion." Keep the spirit (a retrain is a materially different agent
and its results must stay distinguishable and attributable) without the friction
of a hard-coded hash blocking every iteration.
**G. Fix the measuring stick before iterating on top of it.** The ladder's current
reference agents are weak relative to the champion (e.g. +327 Elo over
`alphabeta` is an 87% win rate, with widening confidence intervals as the gap
grows), and "measure the new generation only against the previous generation"
cannot by itself catch the route-substitution trap this project has already hit
twice (v1 vs. mcts-eval, and again with v2). Define a **frozen reference panel**:
`mcts-value` with `v2.bin` frozen at `nodes:32000`, `mcts-eval` at `nodes:8000`,
`mcts-uct` at `nodes:8000`, `alphabeta` at `nodes:2000` — high-budget versions of
agents that already exist, free to build, strong, stable, and untuned against.
Promotion battery per generation: 2,000 games vs. the immediately previous
generation at `elo1 = 10` (per `docs/conventions.md`'s sample-size rule) plus 800
games vs. each panel member, with the existing mechanism gate. Every generation's
numbers are then comparable to every other generation's.

**Then run it**: generate ~100k games with generation *k* (today: ~2.2 core-hours
at ~1.1 core-seconds/game measured; ~35 minutes once Tier 0-A lands), train
(~3 minutes), run the battery, promote or stop. Three generations is enough to
see whether the gain per generation is holding (~+30 Elo, matching the one
generation already measured) or decaying toward zero. Stop rule: two consecutive
generations landing within ±10 Elo of the previous one against the frozen panel.

## Tier 2 — the value net itself (concrete, not "try a bigger net")

**H. Per-card inputs.** `crates/duels-value/src/features.rs` deliberately excludes
card identity today — a city is represented as 7 color counts plus a VP
breakdown, accessible slots as aggregates. That makes chain equity (a per-card
fact), guild targets, and "which specific card is on offer right now" invisible
to the leaf. Add per-card ownership indicators (mine/theirs/gone, ~219 features)
and the identity of the up-to-6 currently accessible face-up cards. The standing
worry that this "would dwarf everything else" is a data question, not an
architecture one, and it's exactly what Tier 1's larger, lower-variance corpus is
for — a ten-minute offline experiment with the existing pipeline once that corpus
exists.
**I. A score-margin auxiliary head.** Add a regression head predicting final VP
margin (recoverable by replay from any existing corpus, no new data collection).
~80% of games end on points; a dense continuous target is one of the
best-precedented representation improvements in this family (KataGo's score
head), and it gives the search a tiebreak signal a bare win probability can't.
**J. Fix zero-sum coherence architecturally, not by inference-time averaging.**
Share weights across both player perspectives and use one joint softmax over 7
outcomes — {One wins by military/science/civilian, Two wins by
military/science/civilian, draw} — so `P(One) + P(Two) + P(draw) = 1` exactly and
seat symmetry holds by construction, rather than approximately. (PR #70's loss-
reweighting experiment showed the *opposite* direction breaks coherence badly —
opening-position probability mass went to 1.49; this is the fix in the direction
that actually makes the property exact.) Do this after H and I, since it changes
the output head's shape.
**K. Only then widen or deepen the net** (e.g. 256 hidden units, or two hidden
layers) — affordable to explore once Tier 0-A has made a forward pass cheap.

## Tier 3 — a learned policy head (yes, but not first)

Measured branching factor under random play: mean 5.8, max 34
(`engine::legal_actions` sampled over 300 games); likely 8-15 under strong play.
UCB1 with a good value estimate already visits every child several times at 2,000
nodes, so a policy prior in a branching-~6 game buys an estimated one to two
"doublings" worth of effective search — roughly **+40 to +80 Elo** — real, but
comparably sized to Tier 0-A's free win for meaningfully more engineering effort.
Worth doing once the loop is actually running, in part because **the training
data is already free**: the corpora already record root visit distributions
(`policy` fields in `value_corpus*.rs`).

Design sketch, fitted to this game rather than ported from a fixed-board game:
an **action-conditioned scorer**, not a slot-indexed output — what matters is
*which card*, not *which slot index*, and the action space is both large and
variable in shape (`PickWonder`, `Build`, `Discard`, `BuildWonder{slot, wonder}`,
token choices, `MausoleumBuild`, `DestroyOpponentCard`, `ChooseFirstPlayer`; a flat
one-hot over all of it is ~460 wide). Score each legal action from a shared trunk
state embedding plus (card one-hot, action type, wonder id, my cost, their cost,
what slots it reveals) → one logit; softmax over just the legal set; cross-entropy
to the recorded visit counts. Integration point in `tree.rs` is the path
`PriorMode` already carved: compute once per decision node on first expansion (the
already-validated ~8% overhead cost model), expand children in prior order
(`ExpansionOrder` already exists), replace `ucb1` with PUCT-style
`Q + c·P·sqrt(N)/(1+n)`. Chance nodes are untouched by any of this. Re-sweep the
exploration constant `c` afterward — this project has already paid real Elo once
(documented in the eval-rounds history) to learn that a leaf/value change without
a `c` re-derivation leaves gains on the table.

## Tier 4 — search correctness, cheap sweeps

Expose the age deal as its own chance event in `duels-core` (removes the last
determinization residue, see the framing correction above); re-sweep
`chance_widen_alpha`/`chance_widen_c` (originally tuned around a ~20 µs playout
leaf, now inherited by a leaf that's ~1.3 µs post-Tier-0-A and therefore searches
much deeper trees at the same budget); re-derive the exploration constant `c` after
every generation, not just once.

## What this plan reconsiders from prior decisions — on purpose

- The hash-pinned golden test as the mechanism that gates every promotion (F) —
  right instinct for a one-off, wrong shape for a loop.
- Hand-sized "insurance batches" as the corpus-diversity mechanism (D replaces it
  with real exploration in the generator).
- Measuring a new generation only against the immediately previous one and the
  `Nodes(2000)` ladder (G).
- The existing claim in `tree.rs`'s docs that root-derived values are unusable
  training targets (E argues this doesn't transfer from the finding it's based on).
- `features.rs`'s decision to exclude card identity (H).
- Reading PR #70's loss-reweighting negative as evidence against emphasizing rare
  win classes *in general* — it's evidence against doing that *without* a
  coherence-preserving output head, specifically. The specialist agents already
  supply the behavioral diversity that experiment was chasing, by a different and
  now-validated route.

## The Raspberry Pi fleet — honest sizing, not hype

Measured today: `mcts-value` self-play at `nodes:2000` costs ~1.1 core-seconds per
game on the project's own workstation. Scaling by typical Cortex-A72 (Pi 4) /
Cortex-A53 (Pi 3) vs. Apple P-core throughput on this kind of scalar-heavy code:
Pi 4 ≈ 35-45k games/day, each Pi 3 ≈ 15-20k games/day, **fleet ≈ 70k games/day ≈
one `v2`-sized corpus every ~18 hours ≈ roughly one workstation core-equivalent,
about 7-10% of the workstation's own throughput.** Tier 0-A speeds both the
workstation and the fleet up by the same factor, so this ratio holds regardless.

**Conclusion: don't design a distributed training system around this hardware.**
Its value is that it's always on and always quiet, not that it's fast.
Concretely, in priority order:

1. **Continuous background self-play generation** — cross-compile from the Mac
   (`cargo zigbuild` or `cross`, `aarch64-unknown-linux-gnu`; the Pi 3s need a
   64-bit OS); do not attempt to build the workspace *on* a Pi 3 (1 GB RAM won't
   compile it). `GameState` is 256 bytes and a 2,000-node tree is well under a
   megabyte, so several game threads per Pi fit comfortably even on a Pi 3; use
   dumb seed partitioning by hostname and rsync corpora back nightly. Tier 0-C
   (the libm fix) is a real prerequisite here, not optional polish.
2. **The permanent, quiet home for the nightly regression and promotion battery**
   (Tier 1-G's ~4,400-game battery per generation is ~3-5 hours on the fleet
   today, ~1.5 hours post-Tier-0-A) — this frees the workstation and the project
   owner's attention from a task that doesn't need either.
3. **A realistic place to measure the actual production time budget**, if the
   server is ever hosted on hardware like the Pi 4 (milestone M8) — `TimeMs(1000)`
   *on a Pi 4* is the number that would actually matter then, and the fleet is the
   only place to measure it honestly rather than extrapolating from the
   workstation.

**Do not put training on the fleet.** The feature matrix for a `v2`-sized corpus is
~5.4 GB (856 bytes/row) — it doesn't fit a Pi 3 and barely fits a Pi 4 — and
training itself is ~3 minutes on the workstation already, so there's no wait to
save.

## Evaluation discipline for everything above

Expected effects from here on are +20 to +50 Elo per step, not the larger jumps
this project's early rounds saw — every accept in this plan uses 2,000-game cells
at `elo1 = 10`, two disjoint seed ranges, both a `Nodes` and a `TimeMs` budget,
measured against the frozen reference panel (never against the previous
generation alone), with the mechanism gate applied. Tier 0-A is the one
deliberate exception — a 400-game `TimeMs` cell may be enough there, because the
predicted effect size is unusually large for a pure performance change with no
behavioral difference expected.

## What's measured vs. speculative in this plan

**Measured, not estimated:** the 5.8x forward-pass speedup (A), the generator's
current lack of exploration (D), the branching factor (Tier 3), the ~1.1
core-second per-game cost (Pi sizing), and the coherence gap size (B, J).
**Genuinely speculative, flagged as predictions to verify, not facts:** the λ-mix
target's effect on the training peak epoch (E), the per-card feature gain's actual
size (H), and the policy head's Elo estimate (Tier 3). The Raspberry Pi throughput
figures are estimated from published core-architecture ratios, not yet measured on
real hardware — measuring them with one cross-compiled binary is the first thing
to do once the Pis are available.

## The very next concrete action

Transpose `w1` and rewrite `forward` axpy-style in `crates/duels-value/src/net.rs`
as a new `Summation` variant, verify against `examples/value_bench.rs`, then run a
`TimeMs(1000)` cell against the current default on a quiet machine. Highest
expected Elo-per-hour of anything in this plan, and it makes every later tier
cheaper the moment it lands.
