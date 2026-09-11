#!/usr/bin/env python3
"""Finite-difference check for `train_value.py`'s blended-loss gradient.

This is the test that should have caught the original `--value-target-lambda`
gradient bug (the `LOSS` logit's contribution was computed as
`(s - t) / (1 - s)` instead of `s - t`, which agrees with the correct gradient
only in the small-`s` limit and diverges as a confident prediction's `s -> 1`
-- see `blended_dz`'s docstring in `train_value.py` for the from-scratch
derivation). It exists so that bug shape can't silently recur: any future
change to the two-term loss's gradient must keep agreeing with a numerical
derivative of the loss it claims to implement.

Independent by construction: the loss below is written directly from the
`--value-target-lambda` docstring's formula
(`lam * CE4(p, onehot(z)) + (1 - lam) * BCE(1 - p_loss, q)`), not copied from
`blended_dz`'s implementation, so this cannot pass merely by re-deriving the
same bug twice.

Plain `python3` + `numpy`, no pytest dependency (matching this tool's own "a
build-time tool, nothing in the workspace imports it" framing). Run directly:

    tools/test_train_value_grad.py
    # or, from a venv without a system numpy:
    .venv/bin/python tools/test_train_value_grad.py
"""

import sys

import numpy as np

from train_value import LOSS, blended_dz

RNG = np.random.default_rng(20260911)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reference_loss(z, target, q, lam):
    """`lam * CE4(p, onehot(z)) + (1 - lam) * BCE(1 - p_loss, q)`, per row.

    Written independently from `train_value.py`'s implementation, straight
    from the docstring's formula, specifically so this test cannot pass by
    duplicating whatever `blended_dz` computes.
    """
    p = softmax(z)
    p_safe = np.clip(p, 1e-12, 1.0)
    ce = -np.sum(target * np.log(p_safe), axis=1)
    if q is None or lam >= 1.0:
        return ce
    valid = ~np.isnan(q)
    q_safe = np.where(valid, q, 0.5)
    # Same [1e-7, 1-1e-7] clip `blended_dz` uses for its `s`: the analytic
    # gradient is exactly the gradient of this *clipped* loss, not of the
    # unclipped one, so comparing against an unclipped reference would flag
    # a spurious mismatch in the extreme tail where the clip is active.
    s = np.clip(p[:, LOSS], 1e-7, 1 - 1e-7)
    # BCE(1 - p_loss, q) = -[q*log(1-s) + (1-q)*log(s)]
    bce = -(q_safe * np.log(1.0 - s) + (1.0 - q_safe) * np.log(s))
    bce = np.where(valid, bce, 0.0)
    return lam * ce + (1.0 - lam) * bce


def numeric_dz(z, target, q, lam, eps=1e-4):
    """Central-difference d(loss_i)/dz_i, per row, one logit column at a time.

    Perturbing column `j` for every row simultaneously and differencing the
    *per-row* loss vector (not a batch mean) is exact here because each row's
    loss depends only on its own four logits -- row `i`'s finite difference
    is unaffected by every other row sharing the same `eps` perturbation, so
    this vectorizes over rows for free without averaging distinct rows'
    gradients together.
    """
    n, k = z.shape
    grad = np.zeros_like(z)
    for j in range(k):
        zp = z.copy()
        zp[:, j] += eps
        zm = z.copy()
        zm[:, j] -= eps
        lp = reference_loss(zp, target, q, lam)
        lm = reference_loss(zm, target, q, lam)
        grad[:, j] = (lp - lm) / (2 * eps)
    return grad


def check(lam, with_nan_rows, tol=2e-4):
    n, k = 64, 4
    z = RNG.normal(scale=1.5, size=(n, k)).astype(np.float64)
    labels = RNG.integers(0, k, size=n)
    target = np.zeros((n, k))
    target[np.arange(n), labels] = 1.0
    q = RNG.uniform(0.02, 0.98, size=n)
    if with_nan_rows:
        nan_mask = RNG.random(n) < 0.3
        q = np.where(nan_mask, np.nan, q)

    p = softmax(z)
    analytic = blended_dz(p, target, q, lam)
    numeric = numeric_dz(z, target, q, lam)

    err = np.abs(analytic - numeric)
    max_err = err.max()
    status = "OK" if max_err < tol else "FAIL"
    print(
        f"lam={lam:<4} nan_rows={with_nan_rows!s:<5} max|analytic-numeric|={max_err:.3e}  "
        f"[{status}]"
    )
    return max_err < tol


def check_confident_row(lam=0.5, tol=2e-4):
    """A row where the network is already confident of LOSS (s close to 1,
    but comfortably above the implementation's own 1e-7 stability clip) is
    exactly where the original bug's `/(1-s)` blowup showed up as `inf`/`nan`
    in real training runs (see the Arm C training log). `1 - s` here is
    ~3.7e-4, so the buggy formula would amplify the gradient by a factor of
    roughly `1/(1-s)` ~= 2,700x relative to the correct `s - t` -- comfortably
    inside float64 precision, so this is a real check, not a clipping
    artifact.
    """
    z = np.array([[-3.0, -3.0, -3.0, 6.0]], dtype=np.float64)
    target = np.array([[0.0, 0.0, 0.0, 1.0]])
    q = np.array([0.9])  # t = 1 - q = 0.1, far from s -> strong signal
    p = softmax(z)
    analytic = blended_dz(p, target, q, lam)
    numeric = numeric_dz(z, target, q, lam)
    max_err = np.abs(analytic - numeric).max()
    status = "OK" if max_err < tol else "FAIL"
    print(f"confident-row lam={lam}  s(LOSS)={p[0, LOSS]:.6f}  max|analytic-numeric|={max_err:.3e}  [{status}]")
    return max_err < tol


def main():
    ok = True
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        for with_nan in (False, True):
            ok &= check(lam, with_nan)
    ok &= check_confident_row(0.5)
    ok &= check_confident_row(0.1)
    if not ok:
        print("FAIL: analytic gradient disagrees with finite difference")
        return 1
    print("all gradient checks passed")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    sys.exit(main())
