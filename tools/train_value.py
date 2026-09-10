#!/usr/bin/env python3
"""Fit `duels-value`'s network offline and write a weights file the Rust crate
embeds.

This is a **build-time tool, not a workspace dependency**. Nothing in the Rust
workspace imports it, nothing links an ML runtime, and CI never runs it: its
whole output is `crates/duels-value/weights/v1.bin`, which is checked in. The
inference side is eighty lines of hand-written `f32` arithmetic in
`crates/duels-value/src/net.rs`.

Input is the matrix written by `duels-arena`'s `examples/feature_dump.rs`; see
that file's module docs for the record layout. Usage:

    tools/train_value.py --matrix arena/corpus/features-v1.bin \\
        --out crates/duels-value/weights/v1.bin \\
        --metrics arena/corpus/features-v1.metrics.json

Requires only numpy. On a machine without it:

    python3 -m venv .venv && .venv/bin/pip install numpy
    .venv/bin/python tools/train_value.py ...

Three things in here are load-bearing and should not be "simplified" away.

**The split is by game, never by row.** Rows inside one game are the same game
seen from successive plies — they share a seed, a deal and most of a board — so
a row-wise split leaks nearly every validation position into training and every
held-out number it produces is a lie. Every row carries its game's seed for
exactly this reason, and the split is `seed % 10`: 0-6 train, 7-8 validation
(model selection), 9 test (reported once, never selected on).

**The target is the four-way outcome, and it is compared against a
single-scalar model of identical shape.** The claim that making the network
also predict the rare science outcome improves the shared representation for
the main task is a hypothesis, not a fact, so `--also-scalar` trains the
one-sigmoid version on the same rows and the same split and reports both.

**The search's own recorded value is scored on the same held-out rows.** That
is the honest yardstick: the incumbent leaf signal already produces a
calibrated win probability, at the cost of 2000 nodes of search. A learned
value that does not beat it as a *predictor* has no business being tried as a
leaf.
"""

import argparse
import json
import struct
import sys
import time

import numpy as np

MAGIC_MATRIX = b"DVFD"
MAGIC_WEIGHTS = b"DVW1"
HEADER_BYTES = 32

# Must match `duels_value::Outcome::ALL`.
OUTCOME_NAMES = ["military_win", "science_win", "civilian_win", "loss"]
LOSS = 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_matrix(path):
    """Memory-map a feature matrix, returning (seed, label, search_value, x).

    Reads both matrix format versions `examples/feature_dump.rs` has ever
    written: **version 1** (`u32` seed; no `ply` column — every matrix that
    predates this task) and **version 2** (`u64` seed, widened for
    `value_corpus_mv.rs` format-v2 corpora whose seed ranges run well past
    2**32; a `ply` column; `sv` may be `NaN` for a specialist-agent row — see
    that file's module docs, "Specialist rows: `search_value` is `NaN`").
    Existing v1 corpora must keep training exactly as before, which is why
    this stays a version dispatch rather than a single reshaped dtype.
    """
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
    if len(head) < HEADER_BYTES or head[:4] != MAGIC_MATRIX:
        raise SystemExit(f"{path} is not a feature_dump matrix")
    version, n_in, n_out = struct.unpack("<III", head[4:16])
    rows, games = struct.unpack("<QQ", head[16:32])
    if version == 1:
        dt = np.dtype(
            [("seed", "<u4"), ("label", "<u4"), ("sv", "<f4"), ("x", "<f4", (n_in,))]
        )
    elif version == 2:
        dt = np.dtype(
            [
                ("seed", "<u8"),
                ("label", "<u4"),
                ("sv", "<f4"),
                ("ply", "<u4"),
                ("x", "<f4", (n_in,)),
            ]
        )
    else:
        raise SystemExit(f"{path} is version {version}, this tool reads 1 or 2")
    m = np.memmap(path, dtype=dt, mode="r", offset=HEADER_BYTES)
    if len(m) != rows:
        raise SystemExit(f"{path} header claims {rows} rows, file holds {len(m)}")
    print(f"matrix   {path}")
    print(f"         {rows:,} rows from {games:,} games, {n_in} features, {n_out} outcomes")
    print(f"         matrix format v{version}")
    return m, n_in, n_out, games


def split_by_game(seeds):
    """Train / validation / test masks, split on the *game*, never the row."""
    bucket = seeds % 10
    return bucket <= 6, (bucket == 7) | (bucket == 8), bucket == 9


# ---------------------------------------------------------------------------
# The network. Plain numpy: one hidden ReLU layer, then either a four-way
# softmax or a single sigmoid, trained with Adam on minibatches.
# ---------------------------------------------------------------------------


class Mlp:
    def __init__(self, n_in, hidden, n_out, seed, softmax):
        rng = np.random.default_rng(seed)
        # He initialisation for the ReLU layer; a small last layer so the
        # initial predictions start near the class prior rather than saturated.
        self.w1 = (rng.standard_normal((n_in, hidden)) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.w2 = (rng.standard_normal((hidden, n_out)) * 0.01).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)
        self.softmax = softmax
        self.params = [self.w1, self.b1, self.w2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x):
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        z = h @ self.w2 + self.b2
        if self.softmax:
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
        else:
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return h, p

    def predict(self, x, batch=65536):
        out = []
        for i in range(0, len(x), batch):
            out.append(self.forward(x[i : i + batch])[1])
        return np.concatenate(out) if out else np.zeros((0, self.w2.shape[1]), np.float32)

    def step(self, x, target, lr, weight_decay, q=None, lam=1.0):
        """One Adam step on a minibatch. `target` is one-hot (softmax) or a
        column of 0/1 (sigmoid); the gradient of cross-entropy through either
        output layer is the same `p - target`.

        `q` and `lam` implement `--value-target-lambda`'s two-term loss
        (decomposed/softmax model only; `q is None` is the plain
        single-target path and is **exactly** the original code, unchanged,
        so `lam = 1.0` / no `q` is bit-identical to training before this
        option existed):

            lam * CE4(p, onehot(z)) + (1 - lam) * BCE(1 - p_loss, q)

        `1 - p_loss` is `duels_value::Dist::win_probability`'s aggregate win
        mass (`p_military + p_science + p_civilian`), and `q` is the corpus's
        recorded `search_value` (`NaN` for a specialist row, in which case
        this row's second term is skipped entirely, per
        `examples/feature_dump.rs`'s module docs -- the row still trains on
        `lam * CE4` alone, not on a locally-renormalised `lam = 1.0`).
        Model selection never uses this blended loss -- see `train`'s docs.
        """
        n = len(x)
        h, p = self.forward(x)
        if q is None or lam >= 1.0:
            # The exact original path: no blend to compute, nothing to mask.
            dz = (p - target) / n
        else:
            ce = p - target
            valid = ~np.isnan(q)
            # `BCE(1 - p_loss, q)` is algebraically `BCE(p_loss, 1 - q)` (the
            # standard symmetric identity `BCE(1-x, q) = BCE(x, 1-q)`), which
            # is why this only ever needs the softmax's own `LOSS` column:
            # `q_safe`'s value is thrown away by the `valid` mask below for
            # every row it would otherwise touch.
            q_safe = np.where(valid, q, 0.5)
            t = 1.0 - q_safe
            s = np.clip(p[:, LOSS], 1e-7, 1 - 1e-7)
            # d/dz of BCE(s, t) composed with the softmax that produced `s`:
            # `g = (s - t) / (1 - s)` on the `LOSS` logit, `-g * p_k` on every
            # other logit (derived from the softmax Jacobian; see the PR
            # description for the algebra).
            g = (s - t) / (1.0 - s)
            bce = np.empty_like(p)
            bce[:, LOSS] = g
            bce[:, :LOSS] = -g[:, None] * p[:, :LOSS]
            bce *= valid[:, None]
            dz = (lam * ce + (1.0 - lam) * bce) / n
        gw2 = h.T @ dz
        gb2 = dz.sum(axis=0)
        dh = dz @ self.w2.T
        dh[h <= 0.0] = 0.0
        gw1 = x.T @ dh
        gb1 = dh.sum(axis=0)
        grads = [gw1, gb1, gw2, gb2]
        # Decay only the weight matrices; biases are not regularised.
        if weight_decay:
            grads[0] = grads[0] + weight_decay * self.w1
            grads[2] = grads[2] + weight_decay * self.w2

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - b1**self.t
        bc2 = 1.0 - b2**self.t
        for i, (param, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g * g)
            param -= (lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + eps)).astype(
                np.float32
            )


def write_weights(path, net):
    """The DVW1 format `duels_value::Net::from_bytes` reads: row-major
    `(hidden, n_in)` then `(n_out, hidden)`, so both matrices are transposed
    out of the column-major-ish layout the training loop uses."""
    n_in, hidden = net.w1.shape
    n_out = net.w2.shape[1]
    with open(path, "wb") as f:
        f.write(MAGIC_WEIGHTS)
        f.write(struct.pack("<III", n_in, hidden, n_out))
        for a in (net.w1.T, net.b1, net.w2.T, net.b2):
            f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
    return n_in * hidden + hidden + hidden * n_out + n_out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def log_loss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p, y):
    return float(np.mean((p > 0.5) == (y > 0.5)))


def auc(p, y):
    """One-vs-rest ROC AUC by rank, which needs no thresholds and is the right
    read for a class as rare as scientific supremacy: accuracy at 0.5 on a
    1%-prevalence class is 0.99 for a model that always says no."""
    y = y.astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average ranks within ties, so a constant predictor scores exactly 0.5.
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def calibration(p, y, bins=10):
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = (p >= lo) & ((p < hi) | (b == bins - 1))
        n = int(sel.sum())
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": n,
                "predicted": float(np.mean(p[sel])) if n else None,
                "actual": float(np.mean(y[sel])) if n else None,
            }
        )
    return out


def scalar_report(name, p, y):
    r = {
        "name": name,
        "n": int(len(y)),
        "brier": brier(p, y),
        "log_loss": log_loss(p, y),
        "accuracy": accuracy(p, y),
        "auc": auc(p, y),
        "calibration": calibration(p, y),
    }
    print(
        f"  {name:<28} brier {r['brier']:.5f}  logloss {r['log_loss']:.5f}  "
        f"acc {r['accuracy']:.4f}  auc {r['auc']:.4f}"
    )
    return r


def print_calibration(rows):
    print("      predicted        n     mean outcome")
    for r in rows:
        if not r["n"]:
            print(f"      {r['lo']:.1f}-{r['hi']:.1f}          0            -")
        else:
            print(f"      {r['lo']:.1f}-{r['hi']:.1f}   {r['n']:>8}          {r['actual']:.3f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(net, xtr, ttr, xva, yva_win, args, tag, qtr=None, lam=1.0):
    """Minibatch Adam with validation-log-loss early stopping.

    Selection is on the *aggregate win probability's log loss against the
    real outcome* (`yva_win`, derived from `z`) for both model kinds and
    **regardless of `lam`** -- never on the blended training loss. This is
    deliberate and load-bearing, not just "the only quantity the two model
    kinds share" (though it is that too): it is what keeps different `lam`
    values comparable to each other and to `lam = 1.0` on one common,
    training-loss-independent yardstick, exactly as `docs/roadmap.md`'s Tier
    1-E calls for ("ablate lam by arena result / a shared offline yardstick,
    not by each arm's own training objective")."""
    n = len(xtr)
    rng = np.random.default_rng(args.seed + 1)
    best = (float("inf"), None, -1)
    history = []
    for epoch in range(args.epochs):
        # Cosine decay from `lr` to a tenth of it, which needs no schedule
        # tuning and behaves well for a fit this small.
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * epoch / max(1, args.epochs - 1))))
        order = rng.permutation(n)
        t0 = time.time()
        for i in range(0, n, args.batch):
            idx = order[i : i + args.batch]
            q_batch = qtr[idx] if qtr is not None else None
            net.step(xtr[idx], ttr[idx], lr, args.weight_decay, q=q_batch, lam=lam)
        p = net.predict(xva)
        pw = win_prob(p)
        ll = log_loss(pw, yva_win)
        history.append({"epoch": epoch, "lr": float(lr), "val_win_log_loss": ll})
        flag = ""
        if ll < best[0]:
            best = (ll, [p.copy() for p in net.params], epoch)
            flag = " *"
        print(
            f"  [{tag}] epoch {epoch:>3}  lr {lr:.5f}  val win logloss {ll:.5f}"
            f"  ({time.time() - t0:.1f}s){flag}"
        )
        if epoch - best[2] >= args.patience:
            print(f"  [{tag}] no improvement for {args.patience} epochs, stopping")
            break
    if best[1] is not None:
        for p, b in zip(net.params, best[1]):
            p[...] = b
    print(f"  [{tag}] best epoch {best[2]}, val win logloss {best[0]:.5f}")
    return history, best[2]


def win_prob(p):
    """The scalar a search consumes, for either model kind."""
    if p.shape[1] == 1:
        return p[:, 0]
    # Summed from the three win heads, exactly as `Dist::win_probability` does,
    # rather than as `1 - P(loss)`: the two agree to float error under a
    # softmax, and using the same formula as the Rust side removes a whole
    # class of "the report and the search disagree" bug.
    return p[:, 0] + p[:, 1] + p[:, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True, help="the DVW1 weights file to write")
    ap.add_argument("--metrics", default=None, help="where to write the metrics JSON")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument(
        "--also-scalar",
        action="store_true",
        help="train the single-sigmoid baseline of identical shape and compare",
    )
    ap.add_argument(
        "--value-target-lambda",
        type=float,
        default=1.0,
        help=(
            "blend weight for the decomposed model's training target: "
            "lam * CE4(p, onehot(z)) + (1 - lam) * BCE(aggregate_win_mass, search_value). "
            "1.0 (the default) is today's exact behaviour, bit-identical -- see "
            "docs/roadmap.md Tier 1-E. Rows with a NaN search_value (specialist-agent "
            "rows; see examples/feature_dump.rs) always skip the second term, "
            "regardless of lam. Model selection is unaffected by this option: it is "
            "always validation log loss against the real outcome z."
        ),
    )
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all; for a quick smoke run")
    args = ap.parse_args()
    if not (0.0 <= args.value_target_lambda <= 1.0):
        raise SystemExit("--value-target-lambda must be in [0, 1]")

    m, n_in, n_out, games = load_matrix(args.matrix)
    if args.max_rows:
        m = m[: args.max_rows]

    seeds = np.asarray(m["seed"])
    labels = np.asarray(m["label"]).astype(np.int64)
    tr, va, te = split_by_game(seeds)
    print(
        f"split    train {int(tr.sum()):,} rows / {len(np.unique(seeds[tr])):,} games   "
        f"val {int(va.sum()):,} / {len(np.unique(seeds[va])):,}   "
        f"test {int(te.sum()):,} / {len(np.unique(seeds[te])):,}"
    )
    overlap = set(np.unique(seeds[tr])) & set(np.unique(seeds[va]))
    if overlap:
        raise SystemExit(f"the split leaks {len(overlap)} games between train and validation")

    print("labels   " + "  ".join(
        f"{OUTCOME_NAMES[k]} {int((labels == k).sum()):,}" for k in range(n_out)
    ))

    # Materialise the feature blocks. Reading a memmap through a boolean mask
    # copies, which is what we want: the training loop then indexes RAM.
    t0 = time.time()
    x = np.asarray(m["x"])
    xtr, xva, xte = x[tr], x[va], x[te]
    ytr, yva, yte = labels[tr], labels[va], labels[te]
    sv_tr = np.asarray(m["sv"])[tr]
    sv_va, sv_te = np.asarray(m["sv"])[va], np.asarray(m["sv"])[te]
    del x, m
    print(f"loaded   {xtr.nbytes / 1e9:.2f} GB train in {time.time() - t0:.1f}s")

    yva_win = (yva != LOSS).astype(np.float64)
    yte_win = (yte != LOSS).astype(np.float64)

    lam = args.value_target_lambda
    n_nan_tr = int(np.isnan(sv_tr).sum())
    if n_nan_tr:
        print(
            f"note     {n_nan_tr:,} / {len(sv_tr):,} training rows have no search_value "
            f"(specialist rows) -- their second loss term is always skipped, "
            f"regardless of --value-target-lambda"
        )

    results = {
        "matrix": args.matrix,
        "games": int(games),
        "hidden": args.hidden,
        "features": int(n_in),
        "value_target_lambda": lam,
        "split": "seed % 10: 0-6 train, 7-8 validation, 9 test",
        "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "games_in_split": {
            "train": int(len(np.unique(seeds[tr]))),
            "val": int(len(np.unique(seeds[va]))),
            "test": int(len(np.unique(seeds[te]))),
        },
        "label_counts": [int((labels == k).sum()) for k in range(n_out)],
    }

    # --- the decomposed model, the one that ships -------------------------
    print()
    print(f"training the four-way decomposed model (value-target-lambda={lam})")
    onehot = np.zeros((len(ytr), n_out), np.float32)
    onehot[np.arange(len(ytr)), ytr] = 1.0
    net = Mlp(n_in, args.hidden, n_out, args.seed, softmax=True)
    results["decomposed_history"], results["decomposed_best_epoch"] = train(
        net,
        xtr,
        onehot,
        xva,
        yva_win,
        args,
        "4way",
        qtr=sv_tr.astype(np.float64),
        lam=lam,
    )

    print()
    print("held-out (validation) aggregate win probability")
    pva = net.predict(xva)
    va_has_sv = ~np.isnan(sv_va)
    results["val"] = {
        "decomposed": scalar_report("decomposed (sum of 3 heads)", win_prob(pva), yva_win),
        "search_root_value": scalar_report(
            f"search root value ({int(va_has_sv.sum()):,}/{len(sv_va):,} rows)",
            sv_va[va_has_sv].astype(np.float64),
            yva_win[va_has_sv],
        ),
    }

    # --- the single-scalar control ---------------------------------------
    if args.also_scalar:
        print()
        print("training the single-scalar control of identical shape")
        scalar_target = (ytr != LOSS).astype(np.float32).reshape(-1, 1)
        snet = Mlp(n_in, args.hidden, 1, args.seed, softmax=False)
        results["scalar_history"], results["scalar_best_epoch"] = train(
            snet, xtr, scalar_target, xva, yva_win, args, "1way"
        )
        print()
        print("held-out (validation) aggregate win probability, both models")
        results["val"]["scalar"] = scalar_report(
            "single-scalar control", win_prob(snet.predict(xva)), yva_win
        )
    else:
        snet = None

    # --- per-kind heads ---------------------------------------------------
    print()
    print("per-kind heads on validation (one-vs-rest)")
    per_kind = []
    for k in range(n_out):
        yk = (yva == k).astype(np.float64)
        r = {
            "outcome": OUTCOME_NAMES[k],
            "prevalence": float(yk.mean()),
            "auc": auc(pva[:, k].astype(np.float64), yk),
            "brier": brier(pva[:, k].astype(np.float64), yk),
            "mean_p_when_true": float(pva[yva == k, k].mean()) if yk.sum() else None,
            "mean_p_when_false": float(pva[yva != k, k].mean()),
        }
        per_kind.append(r)
        print(
            f"  {r['outcome']:<14} prevalence {r['prevalence']:.4f}  auc {r['auc']:.4f}  "
            f"brier {r['brier']:.5f}  E[p|true] {r['mean_p_when_true']:.3f}  "
            f"E[p|false] {r['mean_p_when_false']:.3f}"
        )
    results["val"]["per_kind"] = per_kind
    argmax_acc = float(np.mean(pva.argmax(axis=1) == yva))
    results["val"]["four_way_argmax_accuracy"] = argmax_acc
    print(f"  four-way argmax accuracy {argmax_acc:.4f}")

    # --- the test set: reported once, never selected on -------------------
    print()
    print("test set (never used for selection)")
    pte = net.predict(xte)
    te_has_sv = ~np.isnan(sv_te)
    results["test"] = {
        "decomposed": scalar_report("decomposed (sum of 3 heads)", win_prob(pte), yte_win),
        "search_root_value": scalar_report(
            f"search root value ({int(te_has_sv.sum()):,}/{len(sv_te):,} rows)",
            sv_te[te_has_sv].astype(np.float64),
            yte_win[te_has_sv],
        ),
    }
    if snet is not None:
        results["test"]["scalar"] = scalar_report(
            "single-scalar control", win_prob(snet.predict(xte)), yte_win
        )
    print()
    print("  calibration of the decomposed model's win probability, test set")
    print_calibration(results["test"]["decomposed"]["calibration"])

    params = write_weights(args.out, net)
    results["parameters"] = int(params)
    print()
    print(f"wrote    {args.out}  ({params:,} parameters)")
    if args.metrics:
        with open(args.metrics, "w") as f:
            json.dump(results, f, indent=2)
        print(f"         {args.metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
