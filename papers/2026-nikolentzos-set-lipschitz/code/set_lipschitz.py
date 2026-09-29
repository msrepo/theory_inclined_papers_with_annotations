#!/usr/bin/env python3
"""Nikolentzos & Skianis, ICLR 2026: Lipschitz constants of set aggregators, checked by hand.

Every claim the notes make about the paper's three multiset distances and its
aggregators (sum, mean, max, attention) is computed here on small multisets,
exactly, without training anything.

Checked here, in the order the notes use them:

  1. the worked example X = {0, 2}, Y = {0, 2, 2} on the real line: EMD 1/3,
     Hausdorff 0, matching 2; mean, sum and max each sit exactly on "their"
     distance;
  2. the matching distance equals a min-cost perfect matching after padding the
     smaller multiset with zero vectors (the "phantom origin" reading), checked
     against the literal definition; its triangle inequality; and it is blind
     to adding zero vectors;
  3. Proposition 2.3: d_M = M * d_EMD when |X| = |Y| = M;
  4. Theorem 3.1, the positive half: random search never beats the constants
     1 (mean/EMD), 1 (sum/matching), sqrt(d) (max/Hausdorff), and each is
     attained;
  5. the negative half: five of the six failures are pairs at distance zero
     whose aggregates differ (each distance's "blind spot");
  6. the sixth failure, max/EMD, is different: for sizes at most M the max IS
     Lipschitz w.r.t. EMD, with constant between M and sqrt(d) M. The paper's
     witness needs m = floor(L + 1) elements, so it only rules out a constant
     that is uniform in M;
  7. Lemma 3.2 (equal sizes): 1/M, M, M and 1, plus d_H <= d_M on S_M;
  8. attention (Prop 3.3): the paper's witness ratio is exactly
     [c(e-1) + eps(1+3e)] / (2(1+e) eps) with e = exp(d eps), which tends to
     c d / 4 + 1 as eps -> 0. The blow-up needs c -> infinity (unbounded inputs).
     Attention is duplication-invariant, and on a bounded domain the search finds
     a finite EMD ratio that grows with the radius; against Hausdorff and
     matching it fails at distance zero, exactly like the mean;
  9. Appendix B.6 (l2 attention): the printed q = (-eps, ..., -eps) does not give
     the printed attention weights; q = (+eps, ..., +eps) does;
 10. Theorem 3.4.2: NN_sum with a bias is not Lipschitz w.r.t. matching (the
     bias turns a free zero element into a cost-free shift of b1); with
     f1(0) = 0 the bound Lip(f1) Lip(f2) holds;
 11. Theorem 3.4.3: the sqrt(d) in the NN_max bound must be the width of the
     layer the max is taken over, not the input dimension; a 1-D input and a
     16-wide tent-function layer beat the bound as printed by a factor 4;
 12. Proposition 3.6: the EMD after adding one element is not just bounded by
     (1 / (n(n+1))) sum_i ||v_i - v_new||, it always equals it; the Hausdorff
     distance is the nearest-neighbour distance of the new element.

With --figures it also regenerates the SVGs in ../figures/ from these same
computations.

Standard library and numpy only.

Run:  python3 set_lipschitz.py            (checks)
      python3 set_lipschitz.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ assignment

def hungarian(a: np.ndarray) -> float:
    """Minimum-cost perfect matching on a square cost matrix (Kuhn-Munkres, O(n^3))."""
    n = a.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)     # p[j] = row matched to column j (1-based, 0 = none)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = a[i0 - 1, :] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            masked = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return float(sum(a[p[j] - 1, j - 1] for j in range(1, n + 1)))


def pdist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


# ------------------------------------------------------------------ the three distances

def d_emd(X: np.ndarray, Y: np.ndarray) -> float:
    """EMD with total weights 1 on both sides (W1 between the empirical distributions).

    Replicating every point of X L/m times and every point of Y L/n times, with
    L = lcm(m, n), leaves both distributions unchanged and makes all masses 1/L,
    so by Birkhoff-von Neumann the transport problem is an L x L assignment.
    """
    m, n = len(X), len(Y)
    L = m * n // math.gcd(m, n)
    Xr = np.repeat(X, L // m, axis=0)
    Yr = np.repeat(Y, L // n, axis=0)
    return hungarian(pdist(Xr, Yr)) / L


def d_haus(X: np.ndarray, Y: np.ndarray) -> float:
    D = pdist(X, Y)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def d_match(X: np.ndarray, Y: np.ndarray) -> float:
    """Matching distance, computed by padding the smaller multiset with zero vectors."""
    m, n = len(X), len(Y)
    k = max(m, n)
    Xp = np.vstack([X, np.zeros((k - m, X.shape[1]))])
    Yp = np.vstack([Y, np.zeros((k - n, Y.shape[1]))])
    return hungarian(pdist(Xp, Yp))


def d_match_literal(X: np.ndarray, Y: np.ndarray) -> float:
    """The paper's definition, by brute force over permutations of the larger multiset."""
    if len(X) < len(Y):
        X, Y = Y, X
    m, n = len(X), len(Y)
    best = np.inf
    for pi in itertools.permutations(range(m)):
        c = sum(np.linalg.norm(X[pi[i]] - Y[i]) for i in range(n))
        c += sum(np.linalg.norm(X[pi[i]]) for i in range(n, m))
        best = min(best, c)
    return float(best)


DIST = {"EMD": d_emd, "Hausdorff": d_haus, "matching": d_match}


# ------------------------------------------------------------------ aggregators

def f_sum(X):
    return X.sum(axis=0)


def f_mean(X):
    return X.mean(axis=0)


def f_max(X):
    return X.max(axis=0)


def f_att(X, W, q, g=lambda z: np.maximum(z, 0.0)):
    s = g(X @ W.T) @ q
    a = np.exp(s - s.max())
    a /= a.sum()
    return a @ X


def f_att_l2(X, W, q, g=lambda z: np.maximum(z, 0.0)):
    s = -np.linalg.norm(q[None, :] - g(X @ W.T), axis=1)
    a = np.exp(s - s.max())
    a /= a.sum()
    return a @ X, a


AGG = {"mean": f_mean, "sum": f_sum, "max": f_max}


def ratio(f, dist, X, Y):
    dd = dist(X, Y)
    num = np.linalg.norm(f(X) - f(Y))
    if dd < 1e-12:
        return np.inf if num > 1e-12 else 0.0
    return num / dd


def worst_ratio(f, dist, sizes, d, trials=200, steps=60, scale=1.0, radius=None, rng=RNG):
    """Random restarts + greedy hill climbing on the ratio ||f(X) - f(Y)|| / dist(X, Y)."""
    def clip(Z):
        if radius is None:
            return Z
        nrm = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z * np.minimum(1.0, radius / np.maximum(nrm, 1e-12))

    best = 0.0
    for _ in range(trials):
        m, n = sizes[rng.integers(len(sizes))]
        X = clip(rng.normal(size=(m, d)) * scale)
        Y = clip(rng.normal(size=(n, d)) * scale)
        r = ratio(f, dist, X, Y)
        step = 0.3 * scale
        for _ in range(steps):
            X2 = clip(X + step * rng.normal(size=X.shape))
            Y2 = clip(Y + step * rng.normal(size=Y.shape))
            r2 = ratio(f, dist, X2, Y2)
            if np.isfinite(r2) and r2 > r:
                X, Y, r = X2, Y2, r2
            else:
                step *= 0.97
        if np.isfinite(r):
            best = max(best, r)
    return best


# ------------------------------------------------------------------ checks

def check_worked_example():
    print("1. Worked example on the real line: X = {0, 2}, Y = {0, 2, 2}")
    X = np.array([[0.0], [2.0]])
    Y = np.array([[0.0], [2.0], [2.0]])
    e, h, mt = d_emd(X, Y), d_haus(X, Y), d_match(X, Y)
    print(f"   EMD = {e:.4f} (1/3),  Hausdorff = {h:.4f},  matching = {mt:.4f}")
    for name, f in AGG.items():
        print(f"   {name:5s}: f(X) = {f(X)[0]:.4f}, f(Y) = {f(Y)[0]:.4f}, |difference| = {abs(f(X)[0] - f(Y)[0]):.4f}")
    print("   mean moves by exactly the EMD, sum by exactly the matching distance, max by exactly the Hausdorff distance (0)")


def check_matching():
    print("\n2. Matching distance = zero-padded min-cost perfect matching")
    worst = 0.0
    for _ in range(300):
        m, n = RNG.integers(1, 6, size=2)
        X, Y = RNG.normal(size=(m, 2)), RNG.normal(size=(n, 2))
        worst = max(worst, abs(d_match(X, Y) - d_match_literal(X, Y)))
    print(f"   300 random pairs, sizes 1..5: max |padded - literal| = {worst:.2e}")
    viol = 0.0
    for _ in range(2000):
        s = RNG.integers(1, 6, size=3)
        X, Y, Z = (RNG.normal(size=(k, 2)) for k in s)
        viol = max(viol, d_match(X, Y) - d_match(X, Z) - d_match(Z, Y))
    print(f"   triangle inequality, 2000 random triples: max violation = {viol:.2e} (<= 0 means none)")
    v = np.array([[1.0, -2.0]])
    print(f"   blind spot: d_M({{v}}, {{v, 0}}) = {d_match(v, np.vstack([v, [[0, 0]]])):.1f}")


def check_prop_2_3():
    print("\n3. Proposition 2.3: d_M = M * d_EMD for equal sizes")
    worst = 0.0
    for _ in range(200):
        M = RNG.integers(1, 7)
        X, Y = RNG.normal(size=(M, 3)), RNG.normal(size=(M, 3))
        worst = max(worst, abs(d_match(X, Y) - M * d_emd(X, Y)))
    print(f"   200 random pairs, M = 1..6: max |d_M - M d_EMD| = {worst:.2e}")


def check_theorem_3_1_positive():
    print("\n4. Theorem 3.1, positive half (sizes 1..4 mixed, d = 3)")
    d = 3
    sizes = [(m, n) for m in range(1, 5) for n in range(1, 5)]
    for (agg, dist, L) in [("mean", "EMD", 1.0), ("sum", "matching", 1.0), ("max", "Hausdorff", math.sqrt(d))]:
        r = worst_ratio(AGG[agg], DIST[dist], sizes, d, trials=80, steps=60)
        print(f"   {agg:4s} / {dist:9s}: worst ratio found {r:.4f}  <=  claimed constant {L:.4f}")
    eps = 0.1
    X = eps * np.eye(d)
    Y = np.zeros((1, d))
    print(f"   max/Hausdorff attained: X = {{eps e_1, ..., eps e_d}}, Y = {{0}}: ratio = {ratio(f_max, d_haus, X, Y):.4f} = sqrt(3)")
    v, u = np.array([[1.0, 2.0, 0.5]]), np.array([[0.0, 0.0, 0.0]])
    print(f"   mean/EMD and sum/matching attained by singletons: {ratio(f_mean, d_emd, v, u):.4f}, {ratio(f_sum, d_match, v, u):.4f}")


def check_blind_spots():
    print("\n5. The failures at distance zero (each distance's blind spot)")
    a, b = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    v = np.array([-1.0, -2.0])
    cases = [
        ("mean", "Hausdorff", np.array([a, b]), np.array([a, b, b])),
        ("mean", "matching", np.array([v]), np.array([v, [0, 0]])),
        ("sum", "EMD", np.array([v]), np.array([v, v])),
        ("sum", "Hausdorff", np.array([v]), np.array([v, v])),
        ("max", "matching", np.array([v]), np.array([v, [0, 0]])),
    ]
    for agg, dist, X, Y in cases:
        dd = DIST[dist](X, Y)
        diff = np.linalg.norm(AGG[agg](X) - AGG[agg](Y))
        print(f"   {agg:4s} / {dist:9s}: distance = {dd:.1f}, ||f(X) - f(Y)|| = {diff:.4f}")
    print("   max / EMD has no such pair: EMD = 0 means one multiset is the other with every")
    print("   multiplicity scaled, and duplicating elements never changes a maximum")


def paper_max_emd_witness(m, d=2):
    """Appendix B.3.7: m - 1 shared points summing to 0, and one top point per set, 1 apart."""
    base = RNG.normal(size=(m - 1, d)) if m > 1 else np.zeros((0, d))
    if m > 1:
        base -= base.mean(axis=0)
    top = np.full(d, 10.0)
    X = np.vstack([base, top])
    Y = np.vstack([base, top + np.ones(d) / math.sqrt(d)])
    return X, Y


def max_emd_curve(Ms=(1, 2, 3, 4, 5, 6), d=2, trials=40, steps=60):
    rows = []
    for M in Ms:
        sizes = [(m, n) for m in range(1, M + 1) for n in range(1, M + 1)]
        found = worst_ratio(f_max, d_emd, sizes, d, trials=trials, steps=steps)
        X, Y = paper_max_emd_witness(M, d)
        rows.append((M, found, ratio(f_max, d_emd, X, Y)))
    return rows


def check_max_emd(rows):
    print("\n6. max / EMD with sizes at most M (d = 2): Lipschitz, constant in [M, sqrt(d) M]")
    for M, found, wit in rows:
        print(f"   M = {M}: paper's witness (m = M) ratio {wit:.4f};  worst found by search {found:.4f};  sqrt(d) M = {math.sqrt(2) * M:.4f}")


def check_lemma_3_2():
    print("\n7. Lemma 3.2 (all sizes equal to M = 4, d = 3)")
    M, d = 4, 3
    sizes = [(M, M)]
    for agg, dist, L in [("mean", "matching", 1 / M), ("sum", "EMD", M), ("max", "EMD", M), ("max", "matching", 1.0)]:
        r = worst_ratio(AGG[agg], DIST[dist], sizes, d, trials=40, steps=60)
        print(f"   {agg:4s} / {dist:9s}: worst ratio found {r:.4f}  <=  claimed {L:.4f}")
    worst = 0.0
    for _ in range(300):
        X, Y = RNG.normal(size=(M, d)), RNG.normal(size=(M, d))
        worst = max(worst, d_haus(X, Y) - d_match(X, Y))
    print(f"   d_H - d_M over 300 equal-size pairs: max = {worst:.4f} (<= 0: Hausdorff is dominated by matching)")


def att_witness_ratio(c, eps, d):
    e = math.exp(d * eps)
    return (c * (e - 1) + eps * (1 + 3 * e)) / (2 * (1 + e) * eps)


def check_attention():
    print("\n8. Attention (Prop 3.3, Appendix B.5), W = -I, q = 1, g = ReLU")
    d = 3
    W, q = -np.eye(d), np.ones(d)
    for c, eps in [(1.0, 0.1), (1.0, 1e-3), (1.0, 1e-6), (10.0, 1e-3), (100.0, 1e-3), (1000.0, 1e-3)]:
        X = np.array([np.full(d, c), np.full(d, eps)])
        Y = np.array([np.full(d, c), np.full(d, -eps)])
        r = np.linalg.norm(f_att(X, W, q) - f_att(Y, W, q)) / d_emd(X, Y)
        print(f"   c = {c:7.1f}, eps = {eps:.0e}: ratio/EMD = {r:10.4f}   formula {att_witness_ratio(c, eps, d):10.4f}   c d/4 + 1 = {c * d / 4 + 1:8.2f}")
    X = np.array([[1.0, 0.5, -1.0], [0.2, -0.3, 2.0]])
    print(f"   duplication invariance: ||att(X) - att(X doubled)|| = {np.linalg.norm(f_att(X, W, q) - f_att(np.vstack([X, X]), W, q)):.1e}")
    a, b = X[0], X[1]
    print(f"   Hausdorff blind spot: d_H({{a,b}}, {{a,b,b}}) = 0, attention moves by {np.linalg.norm(f_att(np.array([a, b]), W, q) - f_att(np.array([a, b, b]), W, q)):.4f}")
    v = np.array([[-1.0, -2.0, 0.5]])
    print(f"   matching blind spot: d_M({{v}}, {{v,0}}) = 0, attention moves by {np.linalg.norm(f_att(v, W, q) - f_att(np.vstack([v, np.zeros((1, d))]), W, q)):.4f}")
    rng = np.random.default_rng(1)
    Wr, qr = rng.normal(size=(d, d)) / math.sqrt(d), rng.normal(size=d) / math.sqrt(d)
    sizes = [(m, n) for m in range(1, 4) for n in range(1, 4)]
    for R in (1.0, 4.0, 16.0):
        r = worst_ratio(lambda Z: f_att(Z, Wr, qr), d_emd, sizes, d, trials=40, steps=60, scale=R / 2, radius=R, rng=rng)
        print(f"   bounded inputs ||v|| <= {R:4.0f}, fixed random W, q: worst ratio/EMD found {r:.3f}")


def check_attention_l2():
    print("\n9. l2 attention (Appendix B.6.1): which q gives the printed weights?")
    d, eps, c = 3, 0.2, 5.0
    W = -np.eye(d)
    X = np.array([np.full(d, c), np.full(d, eps)])
    Y = np.array([np.full(d, c), np.full(d, -eps)])
    s = math.sqrt(d) * eps
    printed = (math.exp(-s) / (1 + math.exp(-s)), 1 / (1 + math.exp(-s)))
    for sign in (-1, +1):
        q = sign * eps * np.ones(d)
        _, aX = f_att_l2(X, W, q)
        _, aY = f_att_l2(Y, W, q)
        print(f"   q = {'+' if sign > 0 else '-'}eps 1: alpha^X = {np.round(aX, 4).tolist()}, alpha^Y = {np.round(aY, 4).tolist()}")
    print(f"   printed: alpha^X = [0.5, 0.5], alpha^Y = {np.round(printed, 4).tolist()}")


def check_nn_sum_bias():
    print("\n10. NN_sum and the bias (Theorem 3.4.2, Appendix B.7.2)")
    a1, b1, a2, b2 = 1.0, 0.5, 1.0, 0.0
    f1 = lambda x: np.maximum(a1 * x + b1, 0.0)
    for c in (0.1, 0.01, 0.001):
        X, Y = np.array([[c], [c]]), np.array([[c]])
        num = abs(a2 * f1(X).sum() + b2 - (a2 * f1(Y).sum() + b2))
        print(f"   X = {{c, c}}, Y = {{c}}, c = {c:5.3f}: d_M = {d_match(X, Y):.3f}, output change = {num:.4f}, ratio = {num / d_match(X, Y):8.1f}")
    rng = np.random.default_rng(2)
    W1, W2 = rng.normal(size=(8, 3)), rng.normal(size=(2, 8))
    lip = np.linalg.norm(W1, 2) * np.linalg.norm(W2, 2)
    net = lambda Z: W2 @ np.maximum(Z @ W1.T, 0.0).sum(axis=0)
    sizes = [(m, n) for m in range(1, 5) for n in range(1, 5)]
    r = worst_ratio(net, d_match, sizes, 3, trials=60, steps=60, rng=rng)
    print(f"   no bias (f1(0) = 0): worst ratio found {r:.3f} <= Lip(f1) Lip(f2) = {lip:.3f}")


def check_nn_max_width(k=16):
    print(f"\n11. NN_max: the sqrt(d) must be the hidden width (1-D input, {k}-wide tent layer)")
    centres = 3.0 * np.arange(k)

    def f1(x):  # x: (n, 1) -> (n, k); tent of height 1 and slope 1 around each centre
        return np.maximum(1.0 - np.abs(x - centres[None, :]), 0.0)

    X = centres[:, None]
    Y = centres[:, None] + 0.25
    dH = d_haus(X, Y)
    change = np.linalg.norm(f1(X).max(axis=0) - f1(Y).max(axis=0))
    lip_f1 = 1.0  # the tents have disjoint supports, so at any point at most one coordinate moves, at slope 1
    print(f"   d_H(X, Y) = {dH:.3f}; ||max f1(X) - max f1(Y)|| = {change:.3f}; Lip(f1) = {lip_f1:.0f}, Lip(f2) = 1 (identity)")
    print(f"   bound with d = input dim 1: {1 * lip_f1 * dH:.3f};  with d = width {k}: {math.sqrt(k) * lip_f1 * dH:.3f};  ratio/printed bound = {change / (lip_f1 * dH):.1f}")
    xs = np.linspace(-1, 3 * k, 20001)[:, None]
    F = f1(xs)
    print(f"   numerical Lip(f1) on a fine grid: {np.linalg.norm(np.diff(F, axis=0), axis=1).max() / (xs[1, 0] - xs[0, 0]):.4f}")


def check_prop_3_6():
    print("\n12. Proposition 3.6: adding one element")
    worst_gap, worst_h = 0.0, 0.0
    for _ in range(200):
        n = RNG.integers(1, 7)
        X = RNG.normal(size=(n, 3))
        w = RNG.normal(size=(1, 3)) * 3
        Xp = np.vstack([X, w])
        bound = np.linalg.norm(X - w, axis=1).sum() / (n * (n + 1))
        worst_gap = max(worst_gap, abs(d_emd(X, Xp) - bound))
        worst_h = max(worst_h, abs(d_haus(X, Xp) - np.linalg.norm(X - w, axis=1).min()))
    print(f"   200 random (X, new point), n = 1..6: max |EMD - bound| = {worst_gap:.2e}  (the bound is an equality)")
    print(f"   max |d_H - nearest-neighbour distance of the new point| = {worst_h:.2e}")
    n = 100
    X = RNG.normal(size=(n, 3))
    w = X[np.argmax(np.linalg.norm(X, axis=1))] * 1.5
    print(f"   n = 100 Gaussian points, new point 1.5x the farthest: EMD change = {np.linalg.norm(X - w, axis=1).sum() / (n * (n + 1)):.4f}, Hausdorff change = {np.linalg.norm(X - w, axis=1).min():.4f}")


# ------------------------------------------------------------------ figures

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:10.5px;fill:#57564f} .v{font-size:11px;fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .box{fill:none;stroke:#c9c7bf;stroke-width:1}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s4{stroke:#eda100}
  .f1{fill:#2a78d6} .f2{fill:#eb6834} .f3{fill:#1baf7a} .f4{fill:#eda100}
  .ln{stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
  .dash{stroke-width:1.3;fill:none;stroke-dasharray:5 3}
  .arr{stroke-width:1.6;fill:none}
  .ring{stroke:#fdfdfc;stroke-width:1.5}
  .ph{fill:none;stroke:#8a8880;stroke-width:1.2;stroke-dasharray:2 2}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e} .box{stroke:#4a4844}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500}
    .ring{stroke:#161615} .ph{stroke:#85837b}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


def fig_three_distances(out: Path):
    """Three panels on a number line: EMD plan, Hausdorff nearest neighbours, matching with a phantom zero."""
    W, H = 920, 330
    body = []
    panels = [("EMD = 1/3", "move 1/6 of mass from 0 to 2, over distance 2"),
              ("Hausdorff = 0", "every point has a twin at distance 0"),
              ("matching = 2", "pad X with a phantom 0; the spare 2 pays |2 − 0|")]
    pw = W / 3
    for k, (hd, sub) in enumerate(panels):
        x0 = k * pw + 20
        sx = lambda t, x0=x0: x0 + 40 + t * 100
        yX, yY = 120, 205
        body.append(f'<text class="hd" x="{x0}" y="26">{hd}</text>')
        body.append(f'<text class="sm" x="{x0}" y="44">{sub}</text>')
        body.append(f'<line class="ax" x1="{sx(-0.25)}" y1="{yX}" x2="{sx(2.35)}" y2="{yX}"/>')
        body.append(f'<line class="ax" x1="{sx(-0.25)}" y1="{yY}" x2="{sx(2.35)}" y2="{yY}"/>')
        body.append(f'<text class="lab" x="{x0 - 4}" y="{yX + 4}">X</text>')
        body.append(f'<text class="lab" x="{x0 - 4}" y="{yY + 4}">Y</text>')
        for t in (0, 1, 2):
            body.append(f'<text class="sm" x="{sx(t) - 3}" y="{yY + 38}">{t}</text>')
            body.append(f'<line class="gd" x1="{sx(t)}" y1="{yX - 30}" x2="{sx(t)}" y2="{yY + 24}"/>')
        # X = {0, 2}: masses 1/2 each; Y = {0, 2, 2}: masses 1/3 at 0, 2/3 at 2
        if k == 0:
            body.append(f'<rect class="f1" x="{sx(0) - 9}" y="{yX - 30}" width="18" height="30" opacity=".85"/>')
            body.append(f'<rect class="f1" x="{sx(2) - 9}" y="{yX - 30}" width="18" height="30" opacity=".85"/>')
            body.append(f'<rect class="f2" x="{sx(0) - 9}" y="{yY - 20}" width="18" height="20" opacity=".85"/>')
            body.append(f'<rect class="f2" x="{sx(2) - 9}" y="{yY - 40}" width="18" height="40" opacity=".85"/>')
            body.append(f'<text class="sm" x="{sx(0) + 12}" y="{yX - 18}">1/2</text>')
            body.append(f'<text class="sm" x="{sx(2) + 12}" y="{yX - 18}">1/2</text>')
            body.append(f'<text class="sm" x="{sx(0) + 12}" y="{yY - 6}">1/3</text>')
            body.append(f'<text class="sm" x="{sx(2) + 12}" y="{yY - 6}">2/3</text>')
            body.append(f'<line class="arr s3" x1="{sx(0)}" y1="{yX + 4}" x2="{sx(0)}" y2="{yY - 24}"/>')
            body.append(f'<text class="sm" x="{sx(0) - 34}" y="{(yX + yY) / 2 - 6}">1/3</text>')
            body.append(f'<line class="arr s3" x1="{sx(2)}" y1="{yX + 4}" x2="{sx(2)}" y2="{yY - 44}"/>')
            body.append(f'<text class="sm" x="{sx(2) + 6}" y="{(yX + yY) / 2 - 6}">1/2</text>')
            body.append(f'<path class="arr s4" d="M{sx(0) + 6},{yX + 6} C{sx(0.8)},{yX + 40} {sx(1.6)},{yY - 70} {sx(2) - 6},{yY - 44}"/>')
            body.append(f'<text class="sm" x="{sx(0.9)}" y="{yX + 36}">1/6 × 2</text>')
        else:
            for t in (0, 2):
                body.append(f'<circle class="f1 ring" cx="{sx(t)}" cy="{yX - 8}" r="6"/>')
            body.append(f'<circle class="f2 ring" cx="{sx(0)}" cy="{yY - 8}" r="6"/>')
            body.append(f'<circle class="f2 ring" cx="{sx(2) - 7}" cy="{yY - 8}" r="6"/>')
            body.append(f'<circle class="f2 ring" cx="{sx(2) + 7}" cy="{yY - 8}" r="6"/>')
            if k == 1:
                body.append(f'<line class="arr s3" x1="{sx(0)}" y1="{yX}" x2="{sx(0)}" y2="{yY - 16}"/>')
                body.append(f'<line class="arr s3" x1="{sx(2)}" y1="{yX}" x2="{sx(2) - 7}" y2="{yY - 16}"/>')
                body.append(f'<line class="arr s3" x1="{sx(2)}" y1="{yX}" x2="{sx(2) + 7}" y2="{yY - 16}"/>')
            else:
                body.append(f'<circle class="ph" cx="{sx(0) + 14}" cy="{yX - 8}" r="6"/>')
                body.append(f'<text class="sm" x="{sx(0) + 8}" y="{yX - 22}">phantom 0</text>')
                body.append(f'<line class="arr s3" x1="{sx(0)}" y1="{yX}" x2="{sx(0)}" y2="{yY - 16}"/>')
                body.append(f'<line class="arr s3" x1="{sx(2)}" y1="{yX}" x2="{sx(2) - 7}" y2="{yY - 16}"/>')
                body.append(f'<line class="arr s4" x1="{sx(0) + 14}" y1="{yX}" x2="{sx(2) + 7}" y2="{yY - 16}"/>')
                body.append(f'<text class="sm" x="{sx(1.1)}" y="{(yX + yY) / 2 + 4}">cost 2</text>')
    reads = [("mean", "1 → 4/3", "moves 1/3 = EMD"), ("sum", "2 → 4", "moves 2 = matching"), ("max", "2 → 2", "moves 0 = Hausdorff")]
    for k, (a, b, c) in enumerate(reads):
        x0 = k * pw + 20
        body.append(f'<text class="v" x="{x0}" y="{H - 30}"><tspan font-weight="500">{a}</tspan>: {b}</text>')
        body.append(f'<text class="sm" x="{x0}" y="{H - 14}">{c}</text>')
    desc = ("Three panels comparing X = {0, 2} (blue, top line) with Y = {0, 2, 2} (orange, bottom line). "
            "EMD: X has mass 1/2 at 0 and 2, Y has mass 1/3 at 0 and 2/3 at 2; the cheapest plan moves 1/6 of mass "
            "from 0 to 2 at distance 2, costing 1/3. Hausdorff: every point in either set has a copy in the other, so it is 0. "
            "Matching: X is padded with a phantom zero; two pairs match at cost 0 and the spare 2 in Y is matched to the "
            "phantom zero at cost 2. Below: the mean moves from 1 to 4/3, by exactly the EMD; the sum from 2 to 4, by "
            "exactly the matching distance; the max stays at 2, matching the Hausdorff distance of 0.")
    out.write_text(svg(W, H, "One pair of multisets, three distances", desc, body))


def fig_max_emd(out: Path, rows):
    W, H = 620, 330
    x0, y0, w, h = 70, 30, 480, 240
    Ms = [r[0] for r in rows]
    ymax = math.sqrt(2) * max(Ms) * 1.05
    sx = lambda M: x0 + (M - 0.5) / (max(Ms) + 0.5) * w
    sy = lambda r: y0 + h - r / ymax * h
    body = [f'<line class="ax" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>',
            f'<line class="ax" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}"/>']
    for t in range(0, int(ymax) + 1, 2):
        body.append(f'<line class="gd" x1="{x0}" y1="{sy(t)}" x2="{x0 + w}" y2="{sy(t)}"/>')
        body.append(f'<text class="sm" x="{x0 - 20}" y="{sy(t) + 4}">{t}</text>')
    for M in Ms:
        body.append(f'<text class="sm" x="{sx(M) - 3}" y="{y0 + h + 16}">{M}</text>')
    body.append(f'<text class="lab" x="{x0 + w / 2 - 110}" y="{y0 + h + 36}">M, the largest multiset size allowed</text>')
    body.append(f'<text class="lab" transform="translate({x0 - 44},{y0 + h / 2 + 90}) rotate(-90)">worst ‖Δ max‖ / EMD</text>')
    line = lambda pts, cls: '<polyline class="' + cls + '" points="' + " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in pts) + '"/>'
    body.append(line([(M, math.sqrt(2) * M) for M in Ms], "dash s2"))
    body.append(line([(M, M) for M in Ms], "dash s1"))
    for M, found, wit in rows:
        body.append(f'<circle class="f3 ring" cx="{sx(M):.1f}" cy="{sy(found):.1f}" r="5"/>')
        body.append(f'<rect class="f4" x="{sx(M) - 3.5:.1f}" y="{sy(wit) - 3.5:.1f}" width="7" height="7"/>')
    lx, ly = x0 + 16, y0 + 12
    body += [f'<line class="dash s2" x1="{lx}" y1="{ly}" x2="{lx + 24}" y2="{ly}"/>',
             f'<text class="sm" x="{lx + 30}" y="{ly + 4}">√d·M, the upper bound proved in the notes (d = 2)</text>',
             f'<line class="dash s1" x1="{lx}" y1="{ly + 18}" x2="{lx + 24}" y2="{ly + 18}"/>',
             f'<text class="sm" x="{lx + 30}" y="{ly + 22}">M, the equal-size constant of Lemma 3.2</text>',
             f'<circle class="f3 ring" cx="{lx + 12}" cy="{ly + 36}" r="5"/>',
             f'<text class="sm" x="{lx + 30}" y="{ly + 40}">worst ratio found by search, sizes 1..M</text>',
             f'<rect class="f4" x="{lx + 8.5}" y="{ly + 50.5}" width="7" height="7"/>',
             f'<text class="sm" x="{lx + 30}" y="{ly + 58}">the paper’s witness (Appendix B.3.7) with m = M</text>']
    desc = ("Chart of the worst ratio of the change in the max to the EMD, for multisets of size at most M, M from 1 to "
            f"{max(Ms)}, in d = 2. Search results (green dots) and the paper's witness (yellow squares) both sit on the "
            "line ratio = M, below the proved upper bound sqrt(d) times M. The ratio is finite for every M and grows "
            "linearly with M, so the max is Lipschitz with respect to EMD for bounded sizes, but not uniformly in M.")
    out.write_text(svg(W, H, "Max versus EMD: finite for every size cap, growing with it", desc, body))


def fig_blowups(out: Path):
    """Left: ratio vs 1/eps for mean/matching and attention/EMD at fixed c. Right: attention ratio vs c."""
    W, H = 920, 320
    d = 3
    body = []
    # left panel, log-log
    x0, y0, w, h = 70, 40, 360, 210
    epss = np.logspace(-4, -0.5, 60)
    mean_r = [(1 - e) / (2 * e) + 0.0 for e in epss]  # X = {c 1, eps 1}, Y = {c 1}, c = 1: ||mean diff|| / d_M
    att_r = [att_witness_ratio(1.0, e, d) for e in epss]
    lx = lambda e: x0 + (math.log10(1 / e)) / 4 * w
    ly = lambda r: y0 + h - (math.log10(max(r, 0.1)) + 1) / 5 * h
    body += [f'<text class="hd" x="{x0}" y="22">Shrink ε at fixed input scale c = 1</text>',
             f'<line class="ax" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>',
             f'<line class="ax" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}"/>']
    for k in range(-1, 5):
        body.append(f'<line class="gd" x1="{x0}" y1="{ly(10 ** k)}" x2="{x0 + w}" y2="{ly(10 ** k)}"/>')
        body.append(f'<text class="sm" x="{x0 - 36}" y="{ly(10 ** k) + 4}">10^{k}</text>')
    for k in range(0, 5):
        body.append(f'<text class="sm" x="{x0 + k / 4 * w - 10}" y="{y0 + h + 16}">10^{k}</text>')
    body.append(f'<text class="lab" x="{x0 + w / 2 - 20}" y="{y0 + h + 34}">1/ε</text>')
    body.append(f'<text class="lab" transform="translate({x0 - 50},{y0 + h / 2 + 60}) rotate(-90)">‖Δf‖ / distance</text>')
    body.append('<polyline class="ln s1" points="' + " ".join(f"{lx(e):.1f},{ly(r):.1f}" for e, r in zip(epss, mean_r)) + '"/>')
    body.append('<polyline class="ln s2" points="' + " ".join(f"{lx(e):.1f},{ly(r):.1f}" for e, r in zip(epss, att_r)) + '"/>')
    body.append(f'<text class="v" x="{x0 + w - 186}" y="{ly(2e3)}">mean vs matching → ∞</text>')
    body.append(f'<text class="v" x="{x0 + w - 186}" y="{ly(att_r[0]) - 8}">attention vs EMD → cd/4 + 1 = {d / 4 + 1:.2f}</text>')
    # right panel, attention vs c
    x1 = 520
    cs = np.logspace(-1, 3, 60)
    r_c = [att_witness_ratio(c, 1e-4, d) for c in cs]
    lx2 = lambda c: x1 + (math.log10(c) + 1) / 4 * w
    body += [f'<text class="hd" x="{x1}" y="22">Grow the input scale c (ε = 10⁻⁴)</text>',
             f'<line class="ax" x1="{x1}" y1="{y0 + h}" x2="{x1 + w}" y2="{y0 + h}"/>',
             f'<line class="ax" x1="{x1}" y1="{y0}" x2="{x1}" y2="{y0 + h}"/>']
    for k in range(-1, 5):
        body.append(f'<line class="gd" x1="{x1}" y1="{ly(10 ** k)}" x2="{x1 + w}" y2="{ly(10 ** k)}"/>')
        body.append(f'<text class="sm" x="{x1 - 36}" y="{ly(10 ** k) + 4}">10^{k}</text>')
    for k in range(-1, 4):
        body.append(f'<text class="sm" x="{lx2(10 ** k) - 10}" y="{y0 + h + 16}">10^{k}</text>')
    body.append(f'<text class="lab" x="{x1 + w / 2 - 50}" y="{y0 + h + 34}">input scale c</text>')
    body.append('<polyline class="ln s2" points="' + " ".join(f"{lx2(c):.1f},{ly(r):.1f}" for c, r in zip(cs, r_c)) + '"/>')
    body.append(f'<text class="v" x="{x1 + 20}" y="{ly(300)}">attention vs EMD ≈ c·d/4: unbounded</text>')
    body.append(f'<text class="v" x="{x1 + 20}" y="{ly(300) + 16}">only because inputs are unbounded</text>')
    desc = ("Two log-log charts of the ratio of output change to input distance for the paper's witnesses, d = 3. "
            "Left, at input scale c = 1 with epsilon shrinking: the mean against the matching distance grows like "
            "1/(2 epsilon) without bound, while attention against EMD levels off at c d / 4 + 1 = 1.75. Right, with "
            "epsilon fixed at 1e-4 and c growing from 0.1 to 1000: the attention ratio grows linearly, about c d / 4. "
            "So the mean fails because of a blind spot of the distance, attention against EMD only because the "
            "inputs are unbounded.")
    out.write_text(svg(W, H, "Two kinds of blow-up", desc, body))


def figures(rows):
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    fig_three_distances(d / "three-distances.svg")
    fig_max_emd(d / "max-emd-growth.svg", rows)
    fig_blowups(d / "two-blowups.svg")
    print("\nfigures: wrote", ", ".join(sorted(p.name for p in d.glob("*.svg"))))


def main():
    # sanity: Hungarian against brute force
    for _ in range(50):
        n = RNG.integers(1, 6)
        C = RNG.random((n, n))
        brute = min(sum(C[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
        assert abs(hungarian(C) - brute) < 1e-9
    check_worked_example()
    check_matching()
    check_prop_2_3()
    check_theorem_3_1_positive()
    check_blind_spots()
    rows = max_emd_curve()
    check_max_emd(rows)
    check_lemma_3_2()
    check_attention()
    check_attention_l2()
    check_nn_sum_bias()
    check_nn_max_width()
    check_prop_3_6()
    if "--figures" in sys.argv:
        figures(rows)


if __name__ == "__main__":
    main()
