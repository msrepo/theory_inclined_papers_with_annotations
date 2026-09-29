#!/usr/bin/env python3
"""A toy law of robustness: how far is the decision boundary of an interpolating
random-features classifier, as the number of parameters p grows?

Setup (all small enough to run in a second or two):

  * inputs  x uniform on the unit sphere in R^d, d = 20 (an isoperimetric
    distribution with c = O(1), the paper's assumption H1);
  * labels  y = sign(x_1), each flipped with probability 0.15, so the Bayes
    risk is R* = 0.15 > 0 (Remark 7: the law needs label noise);
  * model   g(x) = a . relu(W x) / sqrt(p), with W ~ N(0, I) frozen and only the
    p output weights a trained; f = sign(g);
  * fit     for p >= n, the minimum-norm a with g(x_i) = y_i exactly, so f
    interpolates the noisy labels (training error 0 < R* - eps);
            for p < n, least squares (it cannot interpolate).

For each p we estimate on fresh test points:

  * the class stability S(f) = E[h_f(x)], the mean distance to the boundary.
    h_f is found by a DeepFool-style walk to the boundary (linearise g, step to
    its zero, repeat) followed by bisection on the segment, so it is an upper
    bound on the true distance, like the paper's adversarial estimate;
  * the local co-margin |g(x)| / ||grad g(x)||, the first DeepFool step;
  * the normalized co-stability S*(g)/L(g) with the crude global Lipschitz
    bound L(g) <= ||a|| ||W||_op / sqrt(p), which by Eq (7) lower-bounds S(f).

What to expect from the paper: at p ~ n the interpolant is forced (double
descent: ||a|| blows up), and Corollary 6 says it must be unstable. More
parameters buy room for the boundary to sit away from the data.

Run:  python3 rf_margin_toy.py
"""
from __future__ import annotations

import numpy as np

D, N, FLIP = 20, 100, 0.15
N_TEST = 150
P_GRID = (20, 50, 80, 95, 100, 105, 120, 200, 400, 1000, 3000)
SEEDS = (0, 1, 2)


def sphere(rng, m, d):
    x = rng.standard_normal((m, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def labels(rng, x):
    y = np.sign(x[:, 0])
    flip = rng.random(len(x)) < FLIP
    return np.where(flip, -y, y)


class RF:
    def __init__(self, W, a):
        self.W, self.a, self.p = W, a, W.shape[0]

    def g(self, x):
        return np.maximum(x @ self.W.T, 0) @ self.a / np.sqrt(self.p)

    def grad(self, x):
        act = (x @ self.W.T > 0).astype(float)          # (m, p)
        return (act * self.a) @ self.W / np.sqrt(self.p)  # (m, d)

    def lip_upper(self):
        return np.linalg.norm(self.a) * np.linalg.norm(self.W, 2) / np.sqrt(self.p)


def fit(rng, x, y, p):
    W = rng.standard_normal((p, x.shape[1]))
    Phi = np.maximum(x @ W.T, 0) / np.sqrt(p)
    a = np.linalg.pinv(Phi) @ y       # min-norm interpolant if p >= n, else LS
    return RF(W, a)


def boundary_distance(model, x, steps=30, overshoot=1.02):
    """Upper bound on the distance from each row of x to {g = 0}."""
    s0 = np.sign(model.g(x))
    z = x.copy()
    done = np.zeros(len(x), bool)
    for _ in range(steps):
        gz = model.g(z)
        done |= np.sign(gz) != s0
        if done.all():
            break
        gr = model.grad(z)
        nrm2 = np.maximum((gr ** 2).sum(1), 1e-18)
        step = (gz / nrm2)[:, None] * gr * overshoot
        z = np.where(done[:, None], z, z - step)
    # bisection on the segment x -> z: the first sign change along it
    lo, hi = np.zeros(len(x)), np.ones(len(x))
    for _ in range(40):
        mid = (lo + hi) / 2
        flipped = np.sign(model.g(x + mid[:, None] * (z - x))) != s0
        hi = np.where(flipped, mid, hi)
        lo = np.where(flipped, lo, mid)
    dist = hi * np.linalg.norm(z - x, axis=1)
    return np.where(done, dist, np.nan)


def run(p, seed, d=D):
    rng = np.random.default_rng(seed)
    x = sphere(rng, N, d)
    y = labels(rng, x)
    m = fit(rng, x, y, p)
    xt = sphere(rng, N_TEST, d)
    yt = labels(rng, xt)
    train_err = np.mean(np.sign(m.g(x)) != y)
    test_err = np.mean(np.sign(m.g(xt)) != yt)
    h = boundary_distance(m, xt)
    gt = m.g(xt)
    local = np.abs(gt) / np.linalg.norm(m.grad(xt), axis=1)
    norm_co = np.mean(np.abs(gt)) / m.lip_upper()
    return dict(train=train_err, test=test_err, S=np.nanmean(h),
                found=np.mean(~np.isnan(h)), local=np.mean(local),
                norm_co=norm_co, anorm=np.linalg.norm(m.a))


def main():
    print(f"d = {D}, n = {N}, label flips {FLIP:.0%}, {N_TEST} test points, "
          f"mean over seeds {SEEDS}")
    print(f"{'p':>6} {'p/n':>5} {'train err':>9} {'test err':>8} {'S(f)':>7} "
          f"{'|g|/|grad g|':>12} {'S*/L bound':>10} {'||a||':>9}")
    rows = {}
    for p in P_GRID:
        rs = [run(p, s) for s in SEEDS]
        r = {k: float(np.mean([q[k] for q in rs])) for k in rs[0]}
        rows[p] = r
        print(f"{p:>6} {p / N:>5.2f} {r['train']:>9.3f} {r['test']:>8.3f} "
              f"{r['S']:>7.3f} {r['local']:>12.3f} {r['norm_co']:>10.4f} "
              f"{r['anorm']:>9.1f}")
    # the numbers quoted in the notes
    at_n = rows[100]
    big = rows[P_GRID[-1]]
    print()
    print(f"S(f) at p = n: {at_n['S']:.3f};  at p = {P_GRID[-1]}: {big['S']:.3f}  "
          f"(ratio {big['S'] / at_n['S']:.1f}x)")
    print(f"||a|| at p = n: {at_n['anorm']:.0f};  at p = {P_GRID[-1]}: {big['anorm']:.1f}")
    ok = all(r['S'] >= r['norm_co'] - 1e-9 for r in rows.values())
    print(f"Eq (7) S(f) >= S*(g)/L(g) holds at every p: {ok}")

    # the same comparison across input dimension: the dip at p = n, and the
    # overall shrinking of margins as d grows (the sqrt(c/d) scale of Cor 6)
    print()
    print(f"{'d':>4} {'S(f), p = n/5':>14} {'p = n':>7} {'p = 30n':>8} {'ratio 30n / n':>13}")
    for d in (5, 20, 50):
        s = {p: float(np.mean([run(p, sd, d)['S'] for sd in SEEDS])) for p in (N // 5, N, 30 * N)}
        print(f"{d:>4} {s[N // 5]:>14.3f} {s[N]:>7.3f} {s[30 * N]:>8.3f} {s[30 * N] / s[N]:>13.2f}")


if __name__ == "__main__":
    main()
