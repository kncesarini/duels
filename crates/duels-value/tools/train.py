#!/usr/bin/env python3
"""Train the `duels-value` network offline and write `weights/value.bin`.

A build-time tool, not a workspace dependency: it needs numpy and torch in
whatever Python you point it at, and nothing in the Rust workspace imports it.

    # produce the rows (see crates/duels-arena/examples/feature_dump.rs)
    cargo run --release -p duels-arena --example feature_dump -- \
        --corpus arena/corpus/mcts-eval-nodes2000.jsonl --out arena/corpus/features/full

    # train, evaluate on held-out GAMES, write the weights
    python3 crates/duels-value/tools/train.py --data arena/corpus/features/full \
        --out crates/duels-value/weights/value.bin --hidden 128 --epochs 8

What it does, in order:

1. Loads the int8 feature matrix and the per-row records, and splits
   train/validation **by game (seed), never by row** -- rows within a game are
   the same game seen from successive plies, and a row-wise split would leak
   nearly every validation position into training.
2. Trains the **decomposed** model: a 4-way softmax over
   {military_win, science_win, civilian_win, loss} for the evaluated player,
   with cross-entropy against the game's actual outcome (a draw is the soft
   target [0, 0, 0.5, 0.5]).
3. Trains a **single-scalar control** of identical shape (one sigmoid output,
   BCE on win/loss) so the claim "the decomposed target helps the aggregate
   win probability" is measured rather than asserted.
4. Reports held-out Brier / log-loss / accuracy of the aggregate win
   probability for both, per-age, alongside the two comparison columns the
   dump carries on the same rows (the search's root value and
   `duels_eval::win_probability`), a calibration table, and the per-kind head
   quality of the decomposed model.
5. Folds the input normalisation into the first layer and writes the weights
   file `duels_value::Model::from_bytes` reads.
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROW_DTYPE = np.dtype(
    [
        ("seed", "<u4"),
        ("ply", "<u2"),
        ("mover_is_me", "u1"),
        ("label", "u1"),
        ("search_value", "<f4"),
    ]
)
LABELS = ["military_win", "science_win", "civilian_win", "loss"]


def load(prefix, max_rows=None):
    meta = json.load(open(prefix + ".meta.json"))
    nf = meta["num_features"]
    rows = np.fromfile(prefix + ".rows.bin", dtype=ROW_DTYPE)
    n = len(rows)
    if max_rows is not None and max_rows < n:
        n = max_rows
        rows = rows[:n]
    X = np.fromfile(prefix + ".X.i8", dtype=np.int8, count=n * nf).reshape(n, nf)
    aux = np.fromfile(prefix + ".aux.f32", dtype="<f4", count=n)
    assert X.shape[0] == n == len(aux), (X.shape, n, len(aux))
    return meta, X, rows, aux


def split_by_game(rows, val_frac, holdout_seed_from=None):
    seeds = rows["seed"]
    if holdout_seed_from is None:
        uniq = np.unique(seeds)
        cut = uniq[int(len(uniq) * (1.0 - val_frac))]
    else:
        cut = holdout_seed_from
    val = seeds >= cut
    return ~val, val, int(cut)


def soft_targets(labels):
    """label 0..3 -> one-hot; 4 (draw) -> half civilian_win, half loss."""
    t = np.zeros((len(labels), 4), dtype=np.float32)
    ok = labels < 4
    t[np.arange(len(labels))[ok], labels[ok]] = 1.0
    t[~ok, 2] = 0.5
    t[~ok, 3] = 0.5
    return t


def win_outcome(labels):
    y = np.where(labels < 3, 1.0, 0.0).astype(np.float32)
    y[labels == 4] = 0.5
    return y


class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        self.hidden = hidden
        if hidden > 0:
            self.l1 = nn.Linear(n_in, hidden)
            self.l2 = nn.Linear(hidden, n_out)
        else:
            self.l1 = nn.Linear(n_in, n_out)

    def forward(self, x):
        if self.hidden > 0:
            return self.l2(F.relu(self.l1(x)))
        return self.l1(x)


def batches(n, batch, rng):
    idx = rng.permutation(n)
    for i in range(0, n, batch):
        yield idx[i : i + batch]


def train_model(X, T, train_idx, mean, std, args, n_out, seed, tag):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(X.shape[1], args.hidden, n_out)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.1, anneal_strategy="cos"
    )
    mean_t = torch.from_numpy(mean)
    std_t = torch.from_numpy(std)
    Xtr = X[train_idx]
    Ttr = T[train_idx]
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        run, nb = 0.0, 0
        for b in batches(len(Xtr), args.batch, rng):
            xb = (torch.from_numpy(Xtr[b].astype(np.float32)) - mean_t) / std_t
            tb = torch.from_numpy(Ttr[b])
            logits = model(xb)
            if n_out == 4:
                loss = -(tb * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(logits[:, 0], tb[:, 0])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss)
            nb += 1
            step += 1
        print(f"  [{tag}] epoch {epoch + 1}/{args.epochs}  train loss {run / nb:.4f}  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model, X, idx, mean, std, n_out, batch=65536):
    out = np.zeros((len(idx), n_out), dtype=np.float32)
    mean_t = torch.from_numpy(mean)
    std_t = torch.from_numpy(std)
    for i in range(0, len(idx), batch):
        b = idx[i : i + batch]
        xb = (torch.from_numpy(X[b].astype(np.float32)) - mean_t) / std_t
        logits = model(xb)
        if n_out == 4:
            out[i : i + batch] = F.softmax(logits, dim=1).numpy()
        else:
            out[i : i + batch, 0] = torch.sigmoid(logits[:, 0]).numpy()
    return out


def binary_metrics(p, y):
    """Brier, log-loss, accuracy (draws excluded from accuracy) of P(win)."""
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    y = y.astype(np.float64)
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    dec = y != 0.5
    acc = float(np.mean((p[dec] > 0.5) == (y[dec] > 0.5)))
    return brier, ll, acc


def calibration_table(p, y, name):
    print(f"  calibration of {name}:  bucket        n   mean pred   mean outcome")
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (p >= lo) & ((p < hi) | ((b == 9) & (p <= 1.0)))
        if m.sum() == 0:
            continue
        print(f"    {lo:.1f}-{hi:.1f}  {m.sum():9d}     {p[m].mean():.3f}       {y[m].mean():.3f}")


def age_of(X, meta):
    names = meta["feature_names"]
    a1, a2, a3 = names.index("g.age1"), names.index("g.age2"), names.index("g.age3")
    return 1 * (X[:, a1] > 0) + 2 * (X[:, a2] > 0) + 3 * (X[:, a3] > 0)


def write_weights(path, model, mean, std, desc):
    n_in = model.l1.in_features
    hidden = model.hidden
    assert hidden > 0
    w1 = model.l1.weight.detach().numpy().astype(np.float64)  # (H, F)
    b1 = model.l1.bias.detach().numpy().astype(np.float64)
    w2 = model.l2.weight.detach().numpy().astype(np.float64)  # (4, H)
    b2 = model.l2.bias.detach().numpy().astype(np.float64)
    # Fold (x - mean) / std into the first layer: W' = W / std, b' = b - W' . mean
    w1f = w1 / std[None, :]
    b1f = b1 - w1f @ mean
    desc_b = desc.encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"DVAL")
        f.write(struct.pack("<IIIII", 1, n_in, hidden, w2.shape[0], len(desc_b)))
        f.write(desc_b)
        f.write(w1f.T.astype("<f4").tobytes())  # input-major: [F][H]
        f.write(b1f.astype("<f4").tobytes())
        f.write(w2.T.astype("<f4").tobytes())  # hidden-major: [H][4]
        f.write(b2.astype("<f4").tobytes())
    return w1f, b1f, w2, b2


def check_fold(model, w1f, b1f, w2, b2, X, idx, mean, std):
    """The folded weights on raw integers must reproduce the normalised model."""
    x = X[idx[:2000]].astype(np.float64)
    h = np.maximum(x @ w1f.T + b1f, 0.0)
    z = h @ w2.T + b2
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    q = predict(model, X, idx[:2000], mean, std, 4)
    err = float(np.abs(p - q).max())
    print(f"  folded-weights check: max |diff| = {err:.2e}")
    assert err < 1e-4, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="prefix written by feature_dump")
    ap.add_argument("--out", default=None, help="weights file to write")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of GAMES held out (by seed)")
    ap.add_argument("--holdout-seed-from", type=int, default=None, help="explicit seed cut instead of --val-frac")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-control", action="store_true", help="skip the single-scalar control")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.time()
    meta, X, rows, aux = load(args.data, args.max_rows)
    print(f"loaded {len(rows):,} rows x {X.shape[1]} features from {meta['games']:,} games ({time.time() - t0:.0f}s)")
    labels = rows["label"].astype(np.int64)
    T = soft_targets(labels)
    y = win_outcome(labels)
    tr, va, cut = split_by_game(rows, args.val_frac, args.holdout_seed_from)
    tr_idx = np.nonzero(tr)[0]
    va_idx = np.nonzero(va)[0]
    n_tr_games = len(np.unique(rows["seed"][tr]))
    n_va_games = len(np.unique(rows["seed"][va]))
    print(f"split by game: train {len(tr_idx):,} rows / {n_tr_games:,} games; validation {len(va_idx):,} rows / {n_va_games:,} games (seeds >= {cut})")
    counts = np.bincount(labels[va], minlength=5)
    print("validation label counts: " + ", ".join(f"{n}={c}" for n, c in zip(LABELS + ["draw"], counts)))

    # Normalisation statistics from a training subsample.
    sub = tr_idx[np.random.default_rng(0).permutation(len(tr_idx))[:1_000_000]]
    xs = X[sub].astype(np.float32)
    mean = xs.mean(axis=0)
    std = xs.std(axis=0)
    std[std < 1e-3] = 1.0  # constant columns pass through unscaled
    del xs

    # Baselines on the validation rows.
    yv = y[va_idx]
    sv = rows["search_value"][va_idx].astype(np.float64)
    ev = aux[va_idx].astype(np.float64)
    ages = age_of(X, meta)[va_idx]
    print()
    print("== held-out baselines (same rows) ==")
    for name, p in [("search root value (mcts-eval nodes:2000)", sv), ("duels_eval::win_probability", ev)]:
        if np.all(np.isfinite(p)):
            b, ll, acc = binary_metrics(p, yv)
            print(f"  {name:44s} brier {b:.4f}  logloss {ll:.4f}  acc {acc:.4f}")

    # The decomposed model.
    print()
    print(f"== decomposed model: {X.shape[1]}-{args.hidden}-4 softmax ==")
    model4 = train_model(X, T, tr_idx, mean, std, args, 4, args.seed, "4-way")
    p4 = predict(model4, X, va_idx, mean, std, 4)
    win4 = p4[:, :3].sum(axis=1)
    b4, ll4, acc4 = binary_metrics(win4, yv)
    print(f"  aggregate P(win) = sum of 3 heads:            brier {b4:.4f}  logloss {ll4:.4f}  acc {acc4:.4f}")

    # The single-scalar control.
    results = {"decomposed": {"brier": b4, "logloss": ll4, "acc": acc4}}
    if not args.no_control:
        print()
        print(f"== single-scalar control: {X.shape[1]}-{args.hidden}-1 sigmoid ==")
        model1 = train_model(X, y[:, None], tr_idx, mean, std, args, 1, args.seed, "scalar")
        p1 = predict(model1, X, va_idx, mean, std, 1)[:, 0]
        b1, ll1, acc1 = binary_metrics(p1, yv)
        print(f"  P(win):                                        brier {b1:.4f}  logloss {ll1:.4f}  acc {acc1:.4f}")
        results["scalar"] = {"brier": b1, "logloss": ll1, "acc": acc1}
        print(f"  decomposed - scalar: brier {b4 - b1:+.5f}  logloss {ll4 - ll1:+.5f}  acc {acc4 - acc1:+.5f}  (negative brier/logloss = decomposed better)")

    # Per-age breakdown, all four signals.
    print()
    print("== held-out Brier by age ==")
    print("  age      n    search   eval    4-way" + ("   scalar" if not args.no_control else ""))
    for a in (1, 2, 3):
        m = ages == a
        if m.sum() == 0:
            continue
        line = f"  {a}   {m.sum():8d}   {np.mean((sv[m] - yv[m]) ** 2):.4f}  {np.mean((ev[m] - yv[m]) ** 2):.4f}  {np.mean((win4[m] - yv[m]) ** 2):.4f}"
        if not args.no_control:
            line += f"   {np.mean((p1[m] - yv[m]) ** 2):.4f}"
        print(line)

    # Per-kind heads.
    print()
    print("== decomposed heads on held-out rows ==")
    lv = labels[va_idx]
    hard = lv < 4
    pred_cls = p4.argmax(axis=1)
    print(f"  4-way argmax accuracy (draws excluded): {np.mean(pred_cls[hard] == lv[hard]):.4f}")
    print("  head            actual rate   mean pred   head brier   recall@argmax   precision@argmax")
    for k, name in enumerate(LABELS):
        actual = (lv == k).astype(np.float64)
        ph = p4[:, k].astype(np.float64)
        rec = np.mean(pred_cls[lv == k] == k) if (lv == k).any() else float("nan")
        prec = np.mean(lv[pred_cls == k] == k) if (pred_cls == k).any() else float("nan")
        print(f"  {name:14s}  {actual.mean():.4f}        {ph.mean():.4f}      {np.mean((ph - actual) ** 2):.4f}       {rec:.3f}           {prec:.3f}")
    # Science-head calibration specifically: the rare kind this design exists for.
    print()
    ps = p4[:, 1]
    print("  science_win head, by predicted probability:")
    for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]:
        m = (ps >= lo) & (ps < hi)
        if m.sum():
            print(f"    [{lo:.2f},{hi:.2f})  n={m.sum():8d}  mean pred {ps[m].mean():.3f}  actual {np.mean(lv[m] == 1):.3f}")
    print()
    calibration_table(win4, yv, "4-way aggregate P(win)")

    if args.out:
        desc = (
            f"duels-value mlp {X.shape[1]}-{args.hidden}-4 softmax; trained on {n_tr_games} games "
            f"(seeds < {cut}, {len(tr_idx)} rows, perspective={meta.get('perspective')}) of "
            f"{os.path.basename(meta.get('corpus', '?'))}; epochs={args.epochs} lr={args.lr} batch={args.batch} seed={args.seed}; "
            f"held-out ({n_va_games} games): brier {b4:.4f} logloss {ll4:.4f} acc {acc4:.4f}"
        )
        if not args.no_control:
            desc += f"; single-scalar control brier {b1:.4f} logloss {ll1:.4f} acc {acc1:.4f}"
        w1f, b1f, w2, b2 = write_weights(args.out, model4, mean.astype(np.float64), std.astype(np.float64), desc)
        check_fold(model4, w1f, b1f, w2, b2, X, va_idx, mean, std)
        print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")
        print(f"  desc: {desc}")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
