#!/usr/bin/env python3
"""Finite-n hard-margin SVMs on the imbalanced 2-GMM, against the limits in
asymptotics.py.

The SVM (Eq 2b) is solved exactly (to tolerance) through its dual with a
maximal-violating-pair SMO, the method inside LIBSVM, with C = infinity:

    min_a  1/2 a^T Q a - 1^T a,   Q_ij = y_i y_j <x_i, x_j>,   a >= 0,  y^T a = 0.

Then w = sum_i a_i y_i x_i, beta = w/||w||, and the intercept is set by the
margin-balancing condition of Lemma C.3 (equal minority and majority margins).
Because ||beta|| = 1 and the test point is Gaussian, the test errors of a
trained (beta, beta0) are exact: Err+ = Phi(-rho ||mu|| - beta0),
Err- = Phi(-rho ||mu|| + beta0).

Checked here:

  1. Figure 1's setting scaled down (n = 2000, d = 800, ||mu|| = 1.75,
     pi = 0.15): rho, beta0, kappa and both test errors against Theorem 2.1.
  2. The ELD: quantiles of the training margins y_i f(x_i), per class,
     against max{kappa*, N(rho* ||mu|| + y beta0*, 1)}; the atom at the margin
     is the support vectors.
  3. Proposition C.1: re-solving with a different tau changes only the
     intercept.  (Solved directly: the rebalanced SVM is the same dual with
     the linear term 1^T a replaced by s^T a, s_i = tau for the minority.)
  4. Figure 6's setting (||mu|| = 1, n/d = 2, pi = 0.1): the finite-n SVM also
     has beta0 + kappa < 0, so no tau > 0 puts the boundary at beta0 = 0.

Standard library and numpy only. Takes about a minute.

Run:  python3 svm_simulation.py
"""
from __future__ import annotations

import math

import numpy as np

from asymptotics import Phi, solve


def smo_hard_margin(X: np.ndarray, y: np.ndarray, s: np.ndarray | None = None,
                    tol: float = 1e-7, max_iter: int = 2_000_000) -> np.ndarray:
    """Dual of  min ||w||^2/2  s.t.  y_i(<x_i,w> + b) >= s_i  (s = 1: plain SVM)."""
    n = len(y)
    if s is None:
        s = np.ones(n)
    K = X @ X.T
    diag = np.diag(K).copy()
    a = np.zeros(n)
    grad = -s.copy()                                  # Q a - s at a = 0
    pos, neg = y > 0, y < 0
    for it in range(max_iter):
        v = -y * grad
        up = pos | (neg & (a > 0))                    # indices that may move "up"
        low = neg | (pos & (a > 0))
        i = np.flatnonzero(up)[np.argmax(v[up])]
        j = np.flatnonzero(low)[np.argmin(v[low])]
        gap = v[i] - v[j]
        if gap < tol * max(1.0, abs(v[i])):
            break
        curv = diag[i] + diag[j] - 2 * K[i, j]
        lam = gap / max(curv, 1e-12)
        # a_i += y_i lam, a_j -= y_j lam; keep both >= 0
        if y[i] < 0:
            lam = min(lam, a[i])
        if y[j] > 0:
            lam = min(lam, a[j])
        di, dj = y[i] * lam, -y[j] * lam
        a[i] += di
        a[j] += dj
        grad += y * (K[:, i] * (y[i] * di) + K[:, j] * (y[j] * dj))
    return a


def fit(X, y, tau=1.0):
    s = np.where(y > 0, tau, 1.0)
    a = smo_hard_margin(X, y, s)
    w = (a * y) @ X
    beta = w / np.linalg.norm(w)
    proj = X @ beta
    # margin balancing (Lemma C.3): tau^-1 (min_+ proj + b0) = -(max_- proj + b0)
    pmin, mmax = proj[y > 0].min(), proj[y < 0].max()
    b0 = -(tau * mmax + pmin) / (tau + 1)
    kappa = min(((proj + b0) / s * y)[y > 0].min(), (-(proj + b0))[y < 0].min())
    return beta, b0, kappa, a, proj


def sample(n, d, m, pi, rng):
    y = np.where(rng.random(n) < pi, 1.0, -1.0)
    X = rng.standard_normal((n, d))
    X[:, 0] += y * m                                  # mu = (m, 0, ..., 0) by rotation invariance
    return X, y


def main() -> None:
    rng = np.random.default_rng(7)
    m, pi, n, d = 1.75, 0.15, 2000, 800
    th = solve(m, pi, n / d)
    print(f"1. Figure 1 scaled down: n={n}, d={d} (delta={n / d}), ||mu||={m}, pi={pi}")
    rows = []
    for rep in range(3):
        X, y = sample(n, d, m, pi, rng)
        beta, b0, kappa, a, proj = fit(X, y)
        rho = beta[0]
        rows.append((rho, b0, kappa, Phi(-rho * m - b0), Phi(-rho * m + b0)))
        if rep == 0:
            keep = (X, y, beta, b0, kappa, a, proj)
    rows = np.array(rows)
    names = ("rho", "beta0", "kappa", "Err+", "Err-")
    theory = (th['rho'], th['b0'], th['kappa'], th['err_p'], th['err_m'])
    for k, nm in enumerate(names):
        print(f"   {nm:6s} simulated {rows[:, k].mean():+.4f} (+/- {rows[:, k].std():.4f} over 3 draws)"
              f"   theory {theory[k]:+.4f}")

    X, y, beta, b0, kappa, a, proj = keep
    marg = y * (proj + b0)
    print("\n2. ELD quantiles of y_i f(x_i), per class (first draw), against max{kappa*, N(rho m + y b0, 1)}")
    zs = np.random.default_rng(3).standard_normal(400_000)
    for cls, lab in ((1.0, "minority"), (-1.0, "majority")):
        emp = np.sort(marg[y == cls])
        thr = np.maximum(th['kappa'], zs + th['rho'] * m + cls * th['b0'])
        qs = (0.1, 0.3, 0.5, 0.7, 0.9)
        e = np.quantile(emp, qs); t = np.quantile(thr, qs)
        print(f"   {lab}: quantiles {qs}")
        print(f"      simulated {np.round(e, 3)}")
        print(f"      theory    {np.round(t, 3)}")
        sv = np.mean(a[y == cls] > 1e-6 * a.max())
        th_sv = Phi(th['kappa'] - th['rho'] * m - cls * th['b0'])
        print(f"      fraction on the margin (support vectors): simulated {sv:.3f}, theory {th_sv:.3f}")
    nsv = int(np.sum(a > 1e-6 * a.max()))
    print(f"   total support vectors {nsv} <= d + 1 = {d + 1}  (theory predicts {n * (pi * Phi(th['kappa'] - th['rho'] * m - th['b0']) + (1 - pi) * Phi(th['kappa'] - th['rho'] * m + th['b0'])):.0f})")

    print("\n3. Proposition C.1: tau moves only the intercept (same data, re-solved)")
    for tau in (2.0, 4.0):
        beta_t, b0_t, kappa_t, _, _ = fit(X, y, tau)
        print(f"   tau={tau}: |beta(tau) - beta(1)| = {np.linalg.norm(beta_t - beta):.2e};"
              f" beta0 {b0_t:+.4f} vs Eq 17 {b0 + (tau - 1) / (tau + 1) * kappa:+.4f};"
              f" kappa {kappa_t:.4f} vs Eq 17 {2 / (tau + 1) * kappa:.4f}")

    print("\n4. Figure 6's setting: ||mu|| = 1, n = 1000, d = 500, pi = 0.1")
    th6 = solve(1.0, 0.1, 2.0)
    for rep in range(2):
        X6, y6 = sample(1000, 500, 1.0, 0.1, rng)
        _, b6, k6, _, _ = fit(X6, y6)
        print(f"   draw {rep}: beta0 + kappa = {b6 + k6:+.4f}  (theory {th6['b0'] + th6['kappa']:+.4f});"
              f" the tau>0 range of beta0(tau) is ({b6 - k6:+.3f}, {b6 + k6:+.3f}), which excludes 0")


if __name__ == "__main__":
    main()
