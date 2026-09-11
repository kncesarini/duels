#!/usr/bin/env python3
"""The autonomous self-play/train/gate/promote loop.

Implements `docs/roadmap.md`'s "Autonomous self-play loop design" as a
runnable state machine, instead of a human re-running each step by hand for
every generation. One generation is:

    generate -> verify -> seal -> build training matrix -> train
    -> gate (4 stages) -> record -> promote-or-hold -> stop-check

# Archive-authoritative, not main-authoritative

This script never touches git. The champion pointer, every generation's
weights, corpus and metrics, and the full decision history live under
`--archive-root` (default `/Volumes/storage/duels/autoloop/`, the NAS-backed
storage this project already archives every generation to). `main` is
untouched for the whole run; turning a run's results into a change this
project ships is a separate, human-reviewed step (folding the run's
`state.json` history into `crates/duels-value/weights/generations.json` and,
for a promoted champion, `duels-value/src/lib.rs`'s `DEFAULT_WEIGHTS` -- the
same PR-and-review path every promotion in this project's history has gone
through by hand). See `docs/roadmap.md`'s "Autonomous self-play loop design"
section for the full reasoning behind every choice below.

# Gating a candidate without touching source

Every generation's candidate is gated via `mcts-value:weights=file:<path>`
(`crates/duels-arena/src/agent_spec.rs`), which loads a weights file from disk
at runtime rather than requiring a new compiled-in `WEIGHTS_*` constant and
`agent_spec` match arm per candidate. Promoting a generation to the live
default still goes through that reviewed path; this script only avoids
needing it during gating itself.

# Usage

    tools/autoloop.py run --generations 1 --smoke
    tools/autoloop.py run --generations 20
    tools/autoloop.py status

`--smoke` scales every game/epoch count down to a few minutes' worth of work,
for validating the state machine itself before trusting it with a real,
many-hour generation. Never run `--smoke` results through the promotion path
for real -- they exist to prove the pipeline is wired correctly, not to
produce a usable champion.

Every step checks for its own already-produced output before redoing work, so
an interrupted run resumes cleanly with `tools/autoloop.py run` again: nothing
here needs `--resume` as a separate mode.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Fixed design parameters (docs/roadmap.md's "Autonomous self-play loop
# design"). Not CLI flags: these stay fixed for a whole run so generations
# stay comparable to each other, which is the entire point of a fixed
# protocol -- see the roadmap's "Exploration parameters ... stay fixed for an
# entire run so generations stay comparable; do not auto-tune them".
# ---------------------------------------------------------------------------
SAMPLE_PLIES = 14
TAU = 1.0
SPECIALIST_FRAC = 0.25
NODE_BUDGET = "nodes:2000"

# Fixed training recipe (the recipe-calibration-day fix, docs/roadmap.md).
RECIPE = dict(
    hidden=128,
    epochs=30,
    warmup_epochs=2,
    lr_floor=2e-5,
    weight_decay=1e-5,
    swa_tail_frac=0.333,
    value_target_lambda=0.5,
)

# Gate stage 1 (head-to-head vs parent): elo1=10, two disjoint 2,000-game
# seed ranges pooled by `duels-arena experiment` itself -- the exact protocol
# already used (by hand) for Generation 3 and its recipe-fix retest.
STAGE1_GAMES_PER_RANGE = 2000
STAGE1_ELO1 = 10.0

# Gate stage 3 (TimeMs cross-check, veto-only): one 1,000-game range.
STAGE3_GAMES = 1000

# Gate stage 2 (frozen-panel non-regression): the two non-ancestor panel
# members the second architect review's worked example (Generation 3) used --
# see docs/roadmap.md's "the gain-decay question, answered" section. v2's two
# cells are still recorded (informational) but do not feed this z-test.
NON_REGRESSION_MEMBERS = ["mcts-eval-nodes8000", "mcts-uct-nodes8000"]
# One-sided 95% critical value -- the same z this project's own mechanism
# gate already uses by default (`duels_arena::mechanism::DEFAULT_Z_CRITICAL`),
# so the new gate stage is consistent with an existing, already-reviewed
# convention rather than an invented threshold.
NON_REGRESSION_Z_CRITICAL = 1.645

# Replay window: "at least 2 generations, not window=1" (docs/roadmap.md), but
# explicitly bounded -- the same document measures a 100k-game corpus's
# feature matrix at ~11.5 GB and the workstation's 48 GiB RAM as bounding a
# same-machine window to "about 2 corpora at full row density" (training
# briefly holds both the full matrix and its train/val/test split at once,
# ~2x peak). Growing past 2 without down-sampling older corpora would risk an
# OOM crash mid-run -- a real failure this script would rather not court
# unattended. --replay-window-corpora lets a future run on more RAM (or with
# striding added later) widen this.
DEFAULT_REPLAY_WINDOW_CORPORA = 2

# Periodic audit cadence (docs/roadmap.md): whichever fires first.
AUDIT_EVERY_PROMOTIONS = 3
AUDIT_EVERY_GENERATIONS = 5

# Seed spaces, chosen to never collide with any seed range this project's
# manual experiments have used so far (which have reached ~3.12M) or with
# each other. Corpus generation and gating each get their own monotonic
# counter in state.json so no two calls across the whole run's lifetime ever
# share a seed.
CORPUS_SEED_BASE = 10_000_000
GATE_SEED_BASE = 50_000_000
GATE_SEED_STRIDE = 1_000_000  # >> reference_panel's own internal 5*100,000 span

MIN_FREE_DISK_GB = 5.0


# ---------------------------------------------------------------------------
# Small statistics helpers (no scipy dependency, matching this project's
# existing tools/*.py convention of numpy-only, or no dependency at all).
# ---------------------------------------------------------------------------


def norm_sf(z):
    """Upper-tail standard normal survival function, via erfc (exact, no
    scipy needed)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def elo_se_from_ci95(ci_low, ci_high):
    """Standard error implied by a reported 95% CI, assuming the normal
    approximation `duels_arena::elo::fit_elo` itself uses."""
    return (ci_high - ci_low) / (2 * 1.959964)


def two_elo_z(candidate_elo, candidate_ci, parent_elo, parent_ci):
    """z-score for "candidate reads weaker than parent against the same fixed
    opponent", from each side's own Elo point estimate and 95% CI. Negative
    means the candidate reads worse than the parent did.
    """
    se_c = elo_se_from_ci95(*candidate_ci)
    se_p = elo_se_from_ci95(*parent_ci)
    se = math.sqrt(se_c**2 + se_p**2)
    if se == 0:
        return 0.0
    return (candidate_elo - parent_elo) / se


def stouffer(zs):
    """Combine independent z-scores (Stouffer's method, equal weight)."""
    if not zs:
        return 0.0
    return sum(zs) / math.sqrt(len(zs))


# ---------------------------------------------------------------------------
# Small process/filesystem helpers.
# ---------------------------------------------------------------------------


def log(msg):
    print(f"[autoloop {time_tag()}] {msg}", flush=True)


def time_tag():
    # A log timestamp is reporting, not game logic -- fine to read the clock
    # here (this script lives outside the Rust workspace's determinism rule).
    return time.strftime("%H:%M:%S")


def run(cmd, cwd=None, check=True):
    """Run a subprocess, streaming its output, and return (returncode,
    stdout_text). Raises RuntimeError with the tail of the output on a
    nonzero exit when `check`."""
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True
    )
    out = proc.stdout + proc.stderr
    sys.stdout.write(out)
    if check and proc.returncode != 0:
        tail = "\n".join(out.splitlines()[-40:])
        raise RuntimeError(
            f"command failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"--- last 40 lines of output ---\n{tail}"
        )
    return proc.returncode, out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def check_free_disk_gb(path):
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


# ---------------------------------------------------------------------------
# Binaries. Built once at startup, then referenced directly (not via
# `cargo run`) for the rest of a run -- see the module docs.
# ---------------------------------------------------------------------------


class Bins:
    def __init__(self, release_dir):
        self.arena = release_dir / "duels-arena"
        self.value_corpus_mv = release_dir / "examples" / "value_corpus_mv"
        self.reference_panel = release_dir / "examples" / "reference_panel"
        self.feature_dump = release_dir / "examples" / "feature_dump"

    def check(self):
        for p in [self.arena, self.value_corpus_mv, self.reference_panel, self.feature_dump]:
            if not p.exists():
                raise RuntimeError(f"expected binary missing after build: {p}")


def build_binaries():
    log("building release binaries (duels-arena, and its examples this loop needs)...")
    run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "duels-arena",
            "--bins",
            "--examples",
        ]
    )
    return Bins(REPO_ROOT / "target" / "release")


def check_python_deps():
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "tools/train_value.py needs numpy, which this interpreter does not have.\n"
            "  python3 -m venv .venv && .venv/bin/pip install numpy\n"
            "  tools/autoloop.py run --python .venv/bin/python ..."
        )


# ---------------------------------------------------------------------------
# State.
# ---------------------------------------------------------------------------

STATE_SCHEMA = 1


def default_state():
    return {
        "schema": STATE_SCHEMA,
        "champion": {
            # v3 is the live default (crates/duels-value/weights/v3.bin,
            # DEFAULT_WEIGHTS as of this script's writing) -- reachable
            # without an override.
            "generation_id": "v3",
            "spec": "mcts-value:weights=v3",
            "weights_path": None,
            "weights_sha256": None,
        },
        # Seeded from crates/duels-value/weights/generations.json's
        # tier1-arm-c-prime entry (v3's own recorded panel numbers) the first
        # time state.json is created -- see seed_initial_panel_history().
        "panel_history": {},
        "replay_window": [],  # list of {"generation_id", "corpus_dir", "manifest"}
        "next_corpus_seed": CORPUS_SEED_BASE,
        "next_gate_seed": GATE_SEED_BASE,
        "generations": [],  # full per-generation record, promote-or-hold and why
        "promotions": [],  # generation_ids that were promoted, in order
        "consecutive_holds": 0,
        "generations_since_last_audit": 0,
        "promotions_since_last_audit": 0,
        "audits": [],
        "in_progress": None,  # {"generation_id", "phase"} while a generation runs
        "stop": None,  # {"reason", "at_generation"} once a stop condition fires
    }


def seed_initial_panel_history(state):
    if state["panel_history"]:
        return
    gpath = REPO_ROOT / "crates" / "duels-value" / "weights" / "generations.json"
    data = json.loads(gpath.read_text())
    gens = data["generations"] if isinstance(data, dict) else data
    v3 = next(g for g in gens if g["id"] == "tier1-arm-c-prime")
    cells = {c.get("opponent", ""): c for c in v3["battery"]["cells"]}
    entry = {}
    for opponent_substr, member in [
        ("mcts-eval", "mcts-eval-nodes8000"),
        ("mcts-uct", "mcts-uct-nodes8000"),
    ]:
        cell = next(c for k, c in cells.items() if opponent_substr in k and "8000" in c["budget"])
        # Normalize generations.json's flat {"elo": float, "elo_ci": [lo, hi]}
        # into the same {"elo": {"rating_diff", "diff_ci_low", "diff_ci_high"}}
        # shape `reference_panel`'s own summary.json cells use, so
        # step_gate_stage2_panel can read either a seeded historical baseline
        # or a fresh panel run identically.
        entry[member] = {
            "elo": {
                "rating_diff": cell["elo"],
                "diff_ci_low": cell["elo_ci"][0],
                "diff_ci_high": cell["elo_ci"][1],
            },
            "games": cell["games"],
        }
    state["panel_history"]["v3"] = entry
    log("seeded panel_history[v3] from generations.json's tier1-arm-c-prime entry")


def load_state(state_path):
    if state_path.exists():
        state = json.loads(state_path.read_text())
        return state
    state = default_state()
    seed_initial_panel_history(state)
    atomic_write_json(state_path, state)
    log(f"initialized new state at {state_path}")
    return state


def save_state(state_path, state):
    atomic_write_json(state_path, state)


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


class Cfg:
    def __init__(self, args):
        self.archive_root = Path(args.archive_root)
        self.work_dir = Path(args.work_dir)
        self.python = args.python
        self.smoke = args.smoke
        self.replay_window_corpora = args.replay_window_corpora
        if args.smoke:
            self.games_per_gen = 200
            self.epochs = 2
            self.stage1_games_per_range = 40
            self.stage3_games = 20
            self.panel_games_scale = 0.02  # thins reference_panel's fixed counts
        else:
            self.games_per_gen = args.games_per_gen
            self.epochs = RECIPE["epochs"]
            self.stage1_games_per_range = STAGE1_GAMES_PER_RANGE
            self.stage3_games = STAGE3_GAMES
            self.panel_games_scale = 1.0


# ---------------------------------------------------------------------------
# Pipeline steps.
# ---------------------------------------------------------------------------


def gen_dirs(cfg, gen_id):
    archive_gen = cfg.archive_root / "generations" / gen_id
    work_gen = cfg.work_dir / gen_id
    archive_gen.mkdir(parents=True, exist_ok=True)
    work_gen.mkdir(parents=True, exist_ok=True)
    return archive_gen, work_gen


def step_generate_and_seal_corpus(cfg, bins, state, gen_id):
    """Self-play the current champion into a fresh corpus, verify the replay,
    then seal it (sha256 + copy) to the archive. Skips work already done."""
    archive_gen, work_gen = gen_dirs(cfg, gen_id)
    corpus_dir = cfg.archive_root / "corpus" / gen_id
    sealed_jsonl = corpus_dir / f"{gen_id}.jsonl"
    sealed_manifest = corpus_dir / f"{gen_id}.jsonl.manifest.json"
    seal_record = archive_gen / "corpus_seal.json"

    if seal_record.exists() and sealed_jsonl.exists():
        log(f"[{gen_id}] corpus already sealed at {sealed_jsonl}, skipping generation")
        return json.loads(seal_record.read_text())

    local_jsonl = work_gen / f"{gen_id}.jsonl"
    seed0 = state["next_corpus_seed"]

    if not local_jsonl.exists():
        champion_spec = state["champion"]["spec"]
        params = "" if champion_spec == "mcts-value:weights=v3" else champion_spec.split(":", 1)[1]
        cmd = [
            str(bins.value_corpus_mv),
            "--games",
            str(cfg.games_per_gen),
            "--seed",
            str(seed0),
            "--budget",
            NODE_BUDGET,
            "--sample-plies",
            str(SAMPLE_PLIES),
            "--tau",
            str(TAU),
            "--specialist-frac",
            str(SPECIALIST_FRAC),
            "--out",
            str(local_jsonl),
        ]
        if params:
            cmd += ["--params", params]
        log(f"[{gen_id}] generating {cfg.games_per_gen} games from seed {seed0} "
            f"(champion: {champion_spec})")
        run(cmd)
    else:
        log(f"[{gen_id}] local corpus already exists at {local_jsonl}, skipping generation")

    log(f"[{gen_id}] verifying replay...")
    run([str(bins.value_corpus_mv), "--verify", str(local_jsonl)])

    corpus_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = Path(str(local_jsonl) + ".manifest.json")
    shutil.copy2(local_jsonl, sealed_jsonl)
    shutil.copy2(local_manifest, sealed_manifest)
    checksum = sha256_file(sealed_jsonl)
    manifest = json.loads(sealed_manifest.read_text())

    record = {
        "generation_id": gen_id,
        "corpus_dir": str(corpus_dir),
        "jsonl": str(sealed_jsonl),
        "manifest_path": str(sealed_manifest),
        "sha256": checksum,
        "seed_first": manifest["seed_first"],
        "seed_last": manifest["seed_last"],
        "games": manifest["games"],
        "decisions": manifest["decisions"],
        "generating_agent": manifest["agent"],
    }
    atomic_write_json(seal_record, record)

    # This generation's own corpus seeds are consumed; bump the counter with
    # generous headroom so a re-run at a larger --games-per-gen never
    # collides with this generation's already-sealed range.
    state["next_corpus_seed"] = max(
        state["next_corpus_seed"] + cfg.games_per_gen + 1,
        manifest["seed_last"] + 1,
    )
    log(f"[{gen_id}] sealed corpus: {manifest['games']} games, {manifest['decisions']} decisions, "
        f"sha256 {checksum[:12]}...")
    return record


def step_build_training_matrix(cfg, bins, state, gen_id, corpus_record):
    """Feature-dump this generation's corpus (cached), slide the replay
    window, and merge the window's matrices into this generation's training
    input."""
    archive_gen, work_gen = gen_dirs(cfg, gen_id)
    feature_cache = cfg.archive_root / "feature_cache"
    feature_cache.mkdir(parents=True, exist_ok=True)

    this_matrix = feature_cache / f"{gen_id}.bin"
    if not this_matrix.exists():
        log(f"[{gen_id}] dumping features from this generation's corpus...")
        run(
            [
                str(bins.feature_dump),
                "--corpus",
                corpus_record["jsonl"],
                "--out",
                str(this_matrix),
            ]
        )
    else:
        log(f"[{gen_id}] feature matrix already cached at {this_matrix}")

    window = state["replay_window"] + [
        {"generation_id": gen_id, "matrix": str(this_matrix)}
    ]
    window = window[-cfg.replay_window_corpora :]
    state["replay_window"] = window

    merged = work_gen / "train_matrix.bin"
    if not merged.exists():
        inputs = [w["matrix"] for w in window]
        if len(inputs) == 1:
            shutil.copy2(inputs[0], merged)
            shutil.copy2(str(inputs[0]) + ".json", str(merged) + ".json")
        else:
            log(f"[{gen_id}] merging {len(inputs)}-generation replay window into training matrix...")
            run([cfg.python, str(REPO_ROOT / "tools" / "merge_feature_matrices.py"), "--out", str(merged)] + inputs)
    else:
        log(f"[{gen_id}] training matrix already merged at {merged}")

    # Feature-cache entries that fell out of the sliding window are no
    # longer referenced by any future generation (window only ever looks
    # back cfg.replay_window_corpora generations) -- delete them to bound
    # disk usage. The raw corpus itself is never deleted: it stays archived
    # permanently, per this project's "held generations keep their corpus"
    # design and its general archive-everything discipline.
    kept = {w["generation_id"] for w in window}
    for f in feature_cache.glob("*.bin"):
        if f.stem not in kept:
            f.unlink(missing_ok=True)
            sidecar = Path(str(f) + ".json")
            sidecar.unlink(missing_ok=True)

    return merged, [w["generation_id"] for w in window]


def step_train(cfg, gen_id, matrix_path):
    archive_gen, work_gen = gen_dirs(cfg, gen_id)
    candidate = archive_gen / "candidate.bin"
    metrics_path = archive_gen / "metrics.json"
    if candidate.exists() and metrics_path.exists():
        log(f"[{gen_id}] already trained, reusing {candidate}")
        return candidate, json.loads(metrics_path.read_text())

    cmd = [
        cfg.python,
        str(REPO_ROOT / "tools" / "train_value.py"),
        "--matrix",
        str(matrix_path),
        "--out",
        str(candidate),
        "--metrics",
        str(metrics_path),
        "--hidden",
        str(RECIPE["hidden"]),
        "--epochs",
        str(cfg.epochs),
        "--warmup-epochs",
        str(RECIPE["warmup_epochs"]),
        "--lr-floor",
        str(RECIPE["lr_floor"]),
        "--no-patience",
        "--weight-decay",
        str(RECIPE["weight_decay"]),
        "--swa-tail-frac",
        str(RECIPE["swa_tail_frac"]),
        "--value-target-lambda",
        str(RECIPE["value_target_lambda"]),
    ]
    log(f"[{gen_id}] training candidate ({cfg.epochs} epochs)...")
    run(cmd)
    return candidate, json.loads(metrics_path.read_text())


def gate_stage0_offline_sanity(metrics):
    ll = metrics["val"]["decomposed"]["log_loss"]
    chance = math.log(4)
    if not math.isfinite(ll):
        return False, f"validation log-loss is not finite ({ll})"
    if ll >= chance:
        return False, f"validation log-loss {ll:.5f} is no better than chance ({chance:.5f})"
    return True, f"validation log-loss {ll:.5f} (chance {chance:.5f})"


def next_gate_seeds(state, n):
    seeds = []
    for _ in range(n):
        seeds.append(state["next_gate_seed"])
        state["next_gate_seed"] += GATE_SEED_STRIDE
    return seeds


def step_gate_stage1(cfg, bins, state, gen_id, candidate_path):
    archive_gen, _ = gen_dirs(cfg, gen_id)
    out_dir = archive_gen / "stage1_head_to_head"
    label = f"{gen_id}-vs-parent"
    summary_path = out_dir / label / "summary.json"
    if summary_path.exists():
        log(f"[{gen_id}] stage 1 already gated, reusing {summary_path}")
        return json.loads(summary_path.read_text())

    seed_a, seed_b = next_gate_seeds(state, 2)
    candidate_spec = f"mcts-value:weights=file:{candidate_path}"
    parent_spec = state["champion"]["spec"]
    run(
        [
            str(bins.arena),
            "experiment",
            "--candidate",
            candidate_spec,
            "--control",
            parent_spec,
            "--seeds",
            f"{seed_a},{seed_b}",
            "--budgets",
            NODE_BUDGET,
            "--games",
            str(cfg.stage1_games_per_range),
            "--label",
            label,
            "--out-dir",
            str(out_dir),
            "--sprt-elo0",
            "0",
            "--sprt-elo1",
            str(STAGE1_ELO1),
        ]
    )
    return json.loads(summary_path.read_text())


def step_gate_stage2_panel(cfg, bins, state, gen_id, candidate_path):
    archive_gen, _ = gen_dirs(cfg, gen_id)
    out_dir = archive_gen / "stage2_panel"
    label = f"{gen_id}-panel"
    summary_path = out_dir / label / "summary.json"
    if not summary_path.exists():
        (seed,) = next_gate_seeds(state, 1)
        candidate_spec = f"mcts-value:weights=file:{candidate_path}"
        run(
            [
                str(bins.reference_panel),
                "--candidate",
                candidate_spec,
                "--seed",
                str(seed),
                "--label",
                label,
                "--out-dir",
                str(out_dir),
                "--games-scale",
                str(cfg.panel_games_scale),
            ]
        )
    panel = json.loads(summary_path.read_text())
    cells = {c["member"]: c for c in panel["cells"]}

    parent_id = state["champion"]["generation_id"]
    parent_panel = state["panel_history"].get(parent_id)
    if parent_panel is None:
        # No recorded baseline for this parent (should only happen for a
        # very first champion with no panel history at all) -- cannot
        # compute the non-regression z-test; treat as a pass-through with a
        # clear note rather than blocking forever.
        return panel, None, "no panel_history recorded for parent %r -- non-regression check skipped" % parent_id

    zs = []
    per_member = {}
    for member in NON_REGRESSION_MEMBERS:
        cand_cell = cells[member]
        parent_cell = parent_panel[member]
        z = two_elo_z(
            cand_cell["elo"]["rating_diff"],
            (cand_cell["elo"]["diff_ci_low"], cand_cell["elo"]["diff_ci_high"]),
            parent_cell["elo"]["rating_diff"],
            (parent_cell["elo"]["diff_ci_low"], parent_cell["elo"]["diff_ci_high"]),
        )
        per_member[member] = {
            "candidate_elo": cand_cell["elo"]["rating_diff"],
            "parent_elo": parent_cell["elo"]["rating_diff"],
            "z": z,
        }
        zs.append(z)
    z_pooled = stouffer(zs)
    regressed = z_pooled <= -NON_REGRESSION_Z_CRITICAL
    note = (
        f"pooled z={z_pooled:.2f} across {NON_REGRESSION_MEMBERS} vs parent {parent_id!r} "
        f"({'REGRESSED' if regressed else 'no statistically real regression'}, "
        f"threshold z<=-{NON_REGRESSION_Z_CRITICAL})"
    )
    non_regression = {"per_member": per_member, "z_pooled": z_pooled, "regressed": regressed, "note": note}
    return panel, non_regression, note


def step_gate_stage3_timems(cfg, bins, state, gen_id, candidate_path):
    archive_gen, _ = gen_dirs(cfg, gen_id)
    out_dir = archive_gen / "stage3_timems"
    label = f"{gen_id}-vs-parent-timems"
    summary_path = out_dir / label / "summary.json"
    if summary_path.exists():
        log(f"[{gen_id}] stage 3 already gated, reusing {summary_path}")
        return json.loads(summary_path.read_text())

    (seed,) = next_gate_seeds(state, 1)
    candidate_spec = f"mcts-value:weights=file:{candidate_path}"
    parent_spec = state["champion"]["spec"]
    run(
        [
            str(bins.arena),
            "experiment",
            "--candidate",
            candidate_spec,
            "--control",
            parent_spec,
            "--seeds",
            str(seed),
            "--budgets",
            "time_ms:1000",
            "--games",
            str(cfg.stage3_games),
            "--label",
            label,
            "--out-dir",
            str(out_dir),
            "--sprt-elo0",
            "0",
            "--sprt-elo1",
            str(STAGE1_ELO1),
        ]
    )
    return json.loads(summary_path.read_text())


def decide_promotion(stage0, stage1_summary, non_regression, stage3_summary):
    reasons = []
    ok0, note0 = stage0
    reasons.append(f"stage 0 (offline sanity): {'PASS' if ok0 else 'FAIL'} -- {note0}")
    if not ok0:
        return False, reasons

    verdict1 = stage1_summary["verdict"]
    reasons.append(f"stage 1 (head-to-head vs parent): {verdict1}")
    stage1_pass = verdict1 == "accept"

    if non_regression is None:
        reasons.append("stage 2 (frozen-panel non-regression): skipped (no baseline)")
        stage2_pass = True
    else:
        reasons.append(f"stage 2 (frozen-panel non-regression): {non_regression['note']}")
        stage2_pass = not non_regression["regressed"]

    verdict3 = stage3_summary["verdict"]
    # Veto-only: a clean AcceptH0-equivalent ("reject") is disqualifying; a
    # Continue-equivalent ("inconclusive") or an accept never blocks on its
    # own -- see docs/roadmap.md, gate stage 3 is "veto-only, not a full
    # gate", exactly because TimeMs is wall-clock sensitive and noisier at
    # this sample size than the nodes-budget stages.
    stage3_veto = verdict3 == "reject"
    reasons.append(
        f"stage 3 (TimeMs cross-check, veto-only): {verdict3}"
        + (" -- VETOES promotion" if stage3_veto else "")
    )

    promote = stage1_pass and stage2_pass and not stage3_veto
    return promote, reasons


def maybe_run_periodic_audit(cfg, bins, state):
    trigger = (
        state["promotions_since_last_audit"] >= AUDIT_EVERY_PROMOTIONS
        or state["generations_since_last_audit"] >= AUDIT_EVERY_GENERATIONS
    )
    if not trigger or len(state["promotions"]) < 3:
        return None

    current = state["champion"]
    three_back_id = state["promotions"][-3]
    three_back = next(g for g in state["generations"] if g["generation_id"] == three_back_id)
    three_back_spec = three_back["promoted_to_spec"]

    label = f"audit-{current['generation_id']}-vs-{three_back_id}"
    out_dir = cfg.archive_root / "audits" / label
    (seed_a, seed_b) = next_gate_seeds(state, 2)
    run(
        [
            str(bins.arena),
            "experiment",
            "--candidate",
            current["spec"],
            "--control",
            three_back_spec,
            "--seeds",
            f"{seed_a},{seed_b}",
            "--budgets",
            NODE_BUDGET,
            "--games",
            str(STAGE1_GAMES_PER_RANGE),
            "--label",
            label,
            "--out-dir",
            str(out_dir),
            "--sprt-elo0",
            "0",
            "--sprt-elo1",
            str(STAGE1_ELO1),
        ]
    )
    summary = json.loads((out_dir / label / "summary.json").read_text())
    (panel_seed,) = next_gate_seeds(state, 1)
    panel_label = f"{label}-panel"
    run(
        [
            str(bins.reference_panel),
            "--candidate",
            current["spec"],
            "--seed",
            str(panel_seed),
            "--label",
            panel_label,
            "--out-dir",
            str(cfg.archive_root / "audits"),
            "--games-scale",
            str(cfg.panel_games_scale),
        ]
    )
    audit = {
        "label": label,
        "current": current["generation_id"],
        "three_back": three_back_id,
        "head_to_head_verdict": summary["verdict"],
        "head_to_head_pooled_elo": summary["pooled"][0]["elo"]["rating_diff"] if summary["pooled"] else None,
    }
    state["audits"].append(audit)
    state["promotions_since_last_audit"] = 0
    state["generations_since_last_audit"] = 0
    log(f"periodic audit: {audit}")
    return audit


def check_stop_conditions(state, this_gen_entry):
    if state["consecutive_holds"] >= 2:
        return "two consecutive holds"
    stage1 = this_gen_entry["stage1_summary"]
    if stage1["verdict"] == "reject" and stage1["pooled"] and stage1["pooled"][0]["elo"]["rating_diff"] < 0:
        return "candidate measurably lost to its own parent"
    return None


# ---------------------------------------------------------------------------
# One generation, end to end.
# ---------------------------------------------------------------------------


def run_one_generation(cfg, bins, state, state_path):
    gen_index = len(state["generations"]) + 1
    gen_id = f"auto-gen{gen_index:04d}"
    state["in_progress"] = {"generation_id": gen_id, "phase": "generate"}
    save_state(state_path, state)

    for path in [cfg.archive_root, cfg.work_dir]:
        free_gb = check_free_disk_gb(path)
        if free_gb < MIN_FREE_DISK_GB:
            raise RuntimeError(f"low disk space at {path}: {free_gb:.1f} GB free (floor {MIN_FREE_DISK_GB} GB)")

    log(f"=== generation {gen_id} (champion: {state['champion']['generation_id']}) ===")

    corpus_record = step_generate_and_seal_corpus(cfg, bins, state, gen_id)
    save_state(state_path, state)
    state["in_progress"]["phase"] = "train"
    save_state(state_path, state)

    matrix_path, window_ids = step_build_training_matrix(cfg, bins, state, gen_id, corpus_record)
    candidate_path, metrics = step_train(cfg, gen_id, matrix_path)
    save_state(state_path, state)
    state["in_progress"]["phase"] = "gate"
    save_state(state_path, state)

    stage0 = gate_stage0_offline_sanity(metrics)
    stage1_summary = step_gate_stage1(cfg, bins, state, gen_id, candidate_path)
    save_state(state_path, state)
    panel_summary, non_regression, stage2_note = step_gate_stage2_panel(cfg, bins, state, gen_id, candidate_path)
    save_state(state_path, state)
    stage3_summary = step_gate_stage3_timems(cfg, bins, state, gen_id, candidate_path)
    save_state(state_path, state)

    promote, reasons = decide_promotion(stage0, stage1_summary, non_regression, stage3_summary)

    entry = {
        "generation_id": gen_id,
        "replay_window": window_ids,
        "corpus": corpus_record,
        "training": {"metrics_path": str((cfg.archive_root / "generations" / gen_id / "metrics.json"))},
        "stage0_offline_sanity": {"passed": stage0[0], "note": stage0[1]},
        "stage1_summary": stage1_summary,
        "stage2_panel": panel_summary,
        "stage2_non_regression": non_regression,
        "stage3_summary": stage3_summary,
        "decision": "promote" if promote else "hold",
        "decision_reasons": reasons,
    }

    if promote:
        champion_bin = cfg.archive_root / "champion.bin"
        shutil.copy2(candidate_path, champion_bin)
        weights_sha256 = sha256_file(champion_bin)
        entry["promoted_to_spec"] = f"mcts-value:weights=file:{champion_bin}"
        state["champion"] = {
            "generation_id": gen_id,
            "spec": entry["promoted_to_spec"],
            "weights_path": str(champion_bin),
            "weights_sha256": weights_sha256,
        }
        cells = {c["member"]: c for c in panel_summary["cells"]}
        state["panel_history"][gen_id] = {
            m: {"elo": cells[m]["elo"], "games": cells[m]["games"]} for m in NON_REGRESSION_MEMBERS
        }
        state["promotions"].append(gen_id)
        state["consecutive_holds"] = 0
        state["promotions_since_last_audit"] += 1
        log(f"[{gen_id}] PROMOTED -- new champion {champion_bin} (sha256 {weights_sha256[:12]}...)")
    else:
        state["consecutive_holds"] += 1
        log(f"[{gen_id}] HELD -- champion unchanged ({state['champion']['generation_id']})")

    for reason in reasons:
        log(f"  {reason}")

    state["generations"].append(entry)
    state["generations_since_last_audit"] += 1
    state["in_progress"] = None

    stop_reason = check_stop_conditions(state, entry)
    if stop_reason:
        state["stop"] = {"reason": stop_reason, "at_generation": gen_id}
        log(f"STOP CONDITION: {stop_reason}")

    save_state(state_path, state)
    maybe_run_periodic_audit(cfg, bins, state)
    save_state(state_path, state)
    return entry


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def cmd_run(args):
    cfg = Cfg(args)
    check_python_deps()
    cfg.archive_root.mkdir(parents=True, exist_ok=True)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.archive_root / "state.json"
    state = load_state(state_path)

    if state["stop"] and not args.force_after_stop:
        log(f"a previous stop condition is recorded: {state['stop']} -- pass --force-after-stop "
            f"to run more generations anyway (a human should look at why it stopped first)")
        return

    bins = build_binaries()
    bins.check()

    for i in range(args.generations):
        if state["stop"] and not args.force_after_stop:
            break
        run_one_generation(cfg, bins, state, state_path)

    log(f"run finished: {len(state['generations'])} generation(s) total, "
        f"champion is now {state['champion']['generation_id']}")
    if state["stop"]:
        log(f"stopped: {state['stop']}")


def cmd_status(args):
    state_path = Path(args.archive_root) / "state.json"
    if not state_path.exists():
        print("no state.json yet -- nothing has run")
        return
    state = json.loads(state_path.read_text())
    print(f"champion: {state['champion']['generation_id']} ({state['champion']['spec']})")
    print(f"generations run: {len(state['generations'])}")
    print(f"promotions: {state['promotions']}")
    print(f"consecutive holds: {state['consecutive_holds']}")
    print(f"replay window: {[w['generation_id'] for w in state['replay_window']]}")
    if state["in_progress"]:
        print(f"in progress: {state['in_progress']}")
    if state["stop"]:
        print(f"STOPPED: {state['stop']}")
    for g in state["generations"][-5:]:
        print(f"  {g['generation_id']}: {g['decision']} (window {g['replay_window']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--archive-root", default="/Volumes/storage/duels/autoloop")

    run_p = sub.add_parser("run", parents=[common], help="run one or more generations")
    run_p.add_argument("--generations", type=int, default=1)
    run_p.add_argument("--games-per-gen", type=int, default=100_000)
    run_p.add_argument("--work-dir", default="/tmp/duels-autoloop")
    run_p.add_argument("--python", default=sys.executable)
    run_p.add_argument("--replay-window-corpora", type=int, default=DEFAULT_REPLAY_WINDOW_CORPORA)
    run_p.add_argument(
        "--smoke",
        action="store_true",
        help="scale every game/epoch count down to a few minutes' worth of work, to validate "
        "the pipeline itself -- never treat a --smoke champion as real",
    )
    run_p.add_argument(
        "--force-after-stop",
        action="store_true",
        help="run more generations even though a previous run recorded a stop condition",
    )
    run_p.set_defaults(func=cmd_run)

    status_p = sub.add_parser("status", parents=[common], help="print the current state")
    status_p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
