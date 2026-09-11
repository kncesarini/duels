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

**Then run it**: generate ~100k games with generation *k* (~2.2 *wall* hours at
~1.1 core-seconds/game on the 14-core workstation, correcting a units error in
an earlier draft of this plan; ~35 minutes once Tier 0-A lands), train
(~3 minutes), run the battery, promote or stop. Three generations is enough to
see whether the gain per generation is holding (~+30 Elo, matching the one
generation already measured) or decaying toward zero. Stop rule: two consecutive
generations landing within ±10 Elo of the previous one against the frozen panel.

### Tier 1 design (resolved 2026-09-11, a second architect pass)

The four items above were re-planned to implementation-readiness once a new
fact arrived: **one of the incoming Raspberry Pis has an external 2TB drive**,
now mounted over NFS at `/Volumes/storage` from `pi2.local:/mnt/storage`. That
pass also found something urgent: **`v2.bin`'s actual training corpus (40k
`mcts-value` self-play games + 15k `mcts-eval` insurance games) no longer
exists anywhere** — `arena/corpus/` is gitignored and got overwritten before
anyone archived it. The 2TB drive exists specifically so this doesn't happen
again, and backing up whatever's still on disk (the `v1.bin` corpus, both
weights files) to `/Volumes/storage/duels/` was the first action taken, ahead
of any code.

**What the 2TB drive is for, and isn't.** Archival, not compute and not a
database — sealed corpus shards from every generation and every generating
machine, every generation's weights/training-metrics/battery-results, kept
permanently (`arena/results/` and `.github` artifact retention are both
ephemeral today). Explicitly ruled out: hosting a real database off it — SQLite
over NFS has unsafe locking, and nothing here needs a query a few hundred JSON
manifests can't answer already. Layout: `corpus/<label>/`, `weights/`,
`generations/<id>/`, `archive-meta/` (see the drive's own `README.md`).

**D, resolved.** Sampling lives in the **corpus generator**
(`crates/duels-arena/examples/value_corpus_mv.rs`), not the agent — the agent
builds a fresh tree per `choose()` with no state carried between moves, so the
generator can play a different action than the agent returned with no side
effects. Temperature: `visits^(1/τ)` at τ=1 for the first **14 plies** (roughly
the wonder draft plus the first ~6 Age I decisions — the window where
strategic direction actually gets set), argmax after; a decay schedule was
considered and rejected for now (no evidence to tune it against). **25% of
games get one specialist seat**, split evenly across
`objective=science/military/civilian`; specialists always play argmax (their
whole point is already a different target, sampling them would just add noise
to an already-narrow signal), every generalist seat is temperature-sampled.
**A real trap to avoid**: role/seat assignment must come from a hash of the
seed (e.g. splitmix64 of `seed ^ ROLE_SALT`), never from `seed % 10` or seed
parity — `train_value.py`'s existing train/val/test split is `seed % 10`, and
a parity-based seat rule would silently put every specialist game in one
split bucket. Corpus format gains a `role: u8` and `sampled: bool` per
decision, and a manifest-level `sampling`/`roles`/`role_salt` block.
`--verify`'s current "non-argmax is an error" assertion is replaced with five
checks: unsampled decisions are still argmax; sampled decisions are within the
window and never pick a zero-visit action; specialists never sample; a
specialist decision is really on that specialist's assigned seat; and
calibration is reported **per role** (a specialist's recorded value is
`P(mover wins by its target kind)`, not a win probability — mixing it into
`z`/`q_root` handling anywhere downstream without accounting for this is a
correctness bug waiting to happen, not a subtlety).

**E, resolved.** `q_root` already exists on every training row today with no
new data collection needed (it's `RootStats.value`, the root's visit-weighted
mean, already written to the corpus and read by `train_value.py` today purely
as a held-out yardstick). Because the value head is a 4-way softmax, not a
single win/loss scalar, the blend is implemented as a **two-term loss**
(`λ·CE₄(p, onehot(z)) + (1−λ)·BCE(aggregate_win_mass, q_root)`), not a naive
mixed target — there's no honest per-kind decomposition of `q_root` to blend
against `z` with directly. Model selection during training stays on validation
log loss against the real outcome `z`, never on the blended objective, so
`λ`'s effect is judged from arena results, not from an offline metric biased
toward whatever `λ` was used to fit it. **λ constant across a training run**,
not phase-varying — the roadmap's original framing (weight `q_root` less
early in a game where it's "noisier") turns out to argue backwards: `z` is the
noisier label early on (a single Bernoulli outcome ~60 plies from resolution),
while `q_root` is already the smoother signal there. **D and E are ablated
separately**, not together, specifically to avoid conflating "did exploration
help" with "did the training-target change help": generate once with
exploration+specialists (arm B), train it twice with `λ ∈ {1.0, 0.5}` (arms B
and C — cheap, ~3 minutes each from the same corpus), and keep a pure-argmax
control (arm A) to isolate D's effect on its own. Differences under ~20 Elo
won't be resolvable at a 2,000-game battery size — say so rather than
over-reading a close result.

**F, resolved.** A committed JSON index
(`crates/duels-value/weights/generations.json`) plus a directory-of-manifests
convention on the archive — not a database, matching how this project already
does everything else (`experiment`'s `<label>/summary.json`, corpus
`.manifest.json` sidecars, the generated-but-committed `arena/leaderboard.json`).
Each entry: corpus manifest (shards, generator config, seed ranges, sha256),
training args/metrics, the full promotion battery result, the golden-table
values, and a `status` (`promoted`/`candidate`/`rejected`). `golden.rs` is
re-pointed at this registry rather than a single hand-pinned hash: a retrain
that isn't registered still fails a test in one line (same spirit as today,
without the friction of hand-editing a table on every iteration), and — new —
every *frozen* generation still in the repo for the reference panel gets its
own golden check too, which the current single-table design can't express.
**The promotion decision stays a human call via PR**, exactly as it is today
for `v1`→`v2`; `leaderboard::CHAMPION` is untouched by any of this — it names
the agent (`mcts-value`), not the weights generation, and a promotion changes
`duels-value`'s default the same way `v2`'s promotion already did.

**G, resolved, with one adjustment.** Drop `alphabeta` from the frozen
reference panel (at its current ~87% win rate against the champion, its
confidence interval is too wide to detect a 20-Elo change) and track the
frozen `v2` champion itself at two budgets instead — `nodes:32000` (a strong,
same-family yardstick) and `nodes:2000` (the most interpretable "cumulative
gain since v2" series, since it's the direct ancestor at equal budget). Full
panel: frozen `v2`@`nodes:32000` (800 games), frozen `v2`@`nodes:2000` (1,000
games), `mcts-eval`@`nodes:8000` frozen at its current tuning (800 games,
since `duels-eval` is retuned in its own numbered rounds and would otherwise
silently move the yardstick), `mcts-uct`@`nodes:8000` (800 games, the
non-learned/library-free route-substitution detector). Promotion battery per
generation: the panel above plus 2,000 games vs. the immediately previous
generation at `elo1=10` with the existing mechanism gate — that head-to-head
is what actually gates promotion; the panel's job is catching drift and
route-substitution across generations, which its smaller per-cell size is
still precise enough to do. `ai-candidate.yml`/`nightly-arena.yml` need no
changes; the battery runs separately (on the fleet, once it exists).

**The fleet's role, concretely.** The Pi with the 2TB drive becomes an NFS
archive host (already true — `/Volumes/storage`); the other two Pis just write
sealed, immutable corpus shards into it with **no coordination protocol**:
static seed partitioning by generation/host/counter keeps every machine's
range disjoint without any handshake, and a shard is only "real" once an
atomic rename has sealed it, so a mid-write crash or a generation switchover
can't produce a half-written or ambiguous shard. The workstation trains from a
**fixed, sha256-verified list** of sealed shards (never "whatever's in the
directory right now"), which is what makes "the fleet keeps generating while
the workstation trains" safe without any locking. The promotion battery also
runs on the fleet (one cell per Pi, pooled afterward), freeing the workstation
and the project owner's attention from a task that needs neither.

**First concrete implementation step** (beyond the corpus/weights backup
already done): `value_corpus_mv.rs` format v2 (the D/E changes above) and
`feature_dump.rs`'s matching NaN-for-specialist-rows/ply-column/u64-seed
update — this is the long pole everything else in Tier 1 depends on having
real data to work with.

### Tier 1 outcome (2026-09-11)

D, E and F landed and ran for real. Three arms from matched-conditions
~100k-game corpora: `arm-a` (control, no exploration, `lambda=1.0`), `arm-b`
(D applied, `lambda=1.0`), `arm-c` (`arm-b`'s corpus, `lambda=0.5`, isolating
E). **`arm-b` won clean — +32.0 Elo vs `v2` at 2,000 games, reproduced at
+26.8 on a disjoint seed range.** `arm-c` read -17.2 Elo and was initially
rejected, but a second review found `tools/train_value.py`'s blended-loss
gradient was wrong for the `LOSS` logit (fixed; finite-difference-checked in
`tools/test_train_value_grad.py`) — `arm-c`'s reading is not trustworthy on
the value-target-blend idea because of it. Retrained from the same corpus
with the fixed gradient: `arm-c2` (`lambda=0.5`) reads **+61.0 Elo vs `v2`**
(reproduced +53.9), beats `arm-b` directly by +27.0 Elo, and wins every panel
cell measured; `arm-d2` (`lambda=0.75`, a hedge) also beats `v2` and `arm-b`
but loses to `arm-c2` on every comparison. **`arm-c2`'s weights are now
`v3.bin`, `duels-value`'s promoted `DEFAULT_WEIGHTS`** — see
`crates/duels-value/src/lib.rs`'s "Follow-up round three" for the full
write-up and comparison table, and
`crates/duels-value/weights/generations.json` (F, resolved) for every
generation's corpus manifest, training args, battery result and golden
reference. `v2` is retired to a frozen `mcts-value:weights=v2` slot (G) and
stays in the frozen reference panel rather than being deleted.

Not yet done from this tier's plan at the time: a fresh from-`v3` generation
(the "three generations, see if the gain decays" question) and the
`duels-eval` `CODEOWNERS`-style discipline for `duels-value` itself — the
first is Generation 3, immediately below; the second is still left for a
future round, not blocked on anything above.

### Generation 3 (2026-09-11): the gain-decay question, answered

Ran the "three generations, see whether the gain per generation holds or
decays" experiment this tier's plan called for. A fresh ~100k-game corpus
(`tier1-gen3-explore`, seed range `2,400,001-2,500,000`, disjoint from every
prior range) was self-played by the then-champion `v3` with the identical
exploration+specialist-mixing generator config that produced
`tier1-arm-bc-explore` (`--sample-plies 14 --tau 1.0 --specialist-frac
0.25`), replay-verified cleanly and sealed to `/Volumes/storage/duels/` with
a checksum comparison before training — matching this project's standing
corpus-loss-prevention discipline throughout. Trained window=1 (this
corpus alone), at production hyperparameters, `--value-target-lambda 0.5`.

**The honest headline: the gain decayed sharply, and this project's own
stop rule (`docs/roadmap.md`'s "±10 Elo, two consecutive generations" test)
is close to firing, though not conclusively yet.** The properly pooled
gating cell (4,000 games across two disjoint seed ranges, `elo1=10`) reads
**+17.2 Elo vs `v3`** `[+6.4, +28.0]`, `AcceptH1`, mechanism gate Pass —
real, CI excludes zero, but roughly a **third** of `v2` → `v3`'s own
+61.0/+53.9 Elo jump. The two individual ranges disagreed sharply before
pooling (+29.6 `AcceptH1` vs +4.9 `Continue`), and the `TimeMs(1000)`
cross-check (1,000 games) read +9.4 Elo `[-12.2, +30.9]`, `Continue` —
directionally consistent but not independently significant at that sample
size. A `--value-target-lambda 1.0` control sibling trained from the
*identical* corpus does not clear the gating bar against `v3` (-10.8 Elo,
`Continue`) and loses to the `lambda=0.5` sibling directly by -35.4 Elo —
confirming the lambda blend, not fresh corpus content on its own, is still
what carries a generation's edge, the same story as `v2` → `v3`.

The frozen reference panel adds a real nuance worth a human's attention:
this generation reads **weaker than `v3`'s own numbers** against two of the
panel's three non-ancestor members (`mcts-eval@nodes:8000`: +142.4 vs `v3`'s
+196.4; `mcts-uct@nodes:8000`: +229.8 vs `v3`'s +260.0) even though both
remain decisively positive in absolute terms, while reading *less* negative
than `v3` against the frozen `v2@nodes:32000` cell (-1.7 vs `v3`'s -34.4).
Elo readings against a common third party are not strictly transitive
across generations (this project has hit that non-composition before), so
this is a flagged caveat, not a contradiction of the direct win over `v3` —
but it is exactly the kind of drift signal Tier 1-G's frozen panel exists to
surface, and a promotion PR should not bury it.

Promoted as `v4.bin` on the strength of the mechanism-clean pooled
head-to-head and the clean lambda-effect isolation against its own control
sibling — see `crates/duels-value/weights/generations.json`'s
`tier1-gen3-l05`/`tier1-gen3-l10` entries for the complete record (corpus
manifest, training metrics, every battery cell, the re-run `c`-sweep, which
again found no change from `c=0.15`). `v3` is retired to a frozen
`mcts-value:weights=v3` slot and joins the reference panel at `nodes:2000`,
alongside `v2`'s existing two cells. **Read this as this project's own
predicted plateau largely bearing out** — "expected effects from here on
are +20 to +50 Elo per step, not the larger jumps this project's early
rounds saw" undersold even this: Generation 3 landed *below* that range.
Whether a Generation 4 would clear the ±10 Elo bar at all, or whether this
is the point to stop iterating this specific loop and invest in Tier 2/3
instead (per-card features, a score-margin head, a learned policy prior),
is an open, real question this generation's data does not resolve on its
own — one more generation's reading would.

**Future consideration, flagged by the project owner (2026-09-11), not yet
tried:** generate future corpora at a higher node budget than production's
`nodes:2000`. This isn't just "better game trajectories" — since `v3`'s own
promotion came from blending the training target toward `q_root` (the
generating search's own root value estimate, at `lambda=0.5`), `q_root`'s
own quality is now directly load-bearing, and a deeper generating search
gives a materially less noisy `q_root` to blend toward. The cost scales
roughly linearly with node budget (a 100k-game corpus that takes ~35
minutes post-Tier-0-A at `nodes:2000` would take roughly 4x/16x longer at
`nodes:8000`/`32000`), which is exactly the kind of always-on background
cost the Raspberry Pi fleet exists to absorb rather than something that
has to fit inside a single interactive session. Worth an isolated ablation
(same corpus size and D/E settings, only the generating node budget
changed) before assuming it's a clean win — a stronger generator could
also shift the corpus's position distribution in ways that don't transfer
to the champion's own `nodes:2000`/`TimeMs(1000)` production budget. Given
Generation 3's own decaying-returns finding just above, this is now a
reasonable next thing to try before running a plain Generation 4 at the
same node budget again.

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
