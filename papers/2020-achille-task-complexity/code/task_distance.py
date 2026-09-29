#!/usr/bin/env python3
"""Sections 4, 6 and Figure 1 of Achille, Paolini, Mbeng & Soatto: the task distance.

The identity behind everything here. When the beta-minimal sufficient
statistics are unique, Definition 6.1 (and Corollary 4.5 for Definition 4.1) read

    d(1 -> 2) = I_12 - I_1,        d(2 -> 1) = I_12 - I_2,

with I = information in the weights of the optimal Q. Subtracting,

    d(1 -> 2) - d(2 -> 1) = I_2 - I_1.

The asymmetry of the distance is exactly a difference of task complexities.

Checked here:

  1. Figure 1's matrix (transcribed from the paper). Its antisymmetric part is
     c_j - c_i for a single vector c to within the two-decimal rounding: the
     paper's estimate satisfies the identity, and c ranks the eight tasks by
     complexity. The symmetric part, the count of negative entries and the
     triangle-inequality violations are printed too.
  2. An exactly solvable instance of Definition 6.1: linear-Gaussian
     regression with a shared block and one block per task (the index i of
     D1 u D2 selects the block), P = N(0, lam^2 I), Gibbs = Gaussian Q.
       - d grows with the angle between the two tasks' true weight vectors;
       - a task's distance to an exact copy of itself is not O(1), and a copy
         with more samples is further away still;
       - the label-independent log-det term dominates the information: a
         pure-noise task carries almost as much as a structured one.

Standard library and numpy only.

Run:  python3 task_distance.py             (prints every number quoted in the notes)
      python3 task_distance.py --figures   (also rewrites ../figures/figure1-antisymmetry.svg)
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np

NAMES = ["cifar10", "mnist", "fashion", "ifashion", "letters", "cifar100", "natural", "artificial"]
# Figure 1 (left), row = target task D2, column = source task D1, entry d(D1 -> D2).
M = np.array([
    [0.00, 0.29, 0.31, 0.28, 0.10, 0.01, -0.08, -0.19],
    [0.15, 0.00, 0.20, 0.15, 0.04, -0.01, -0.06, -0.01],
    [0.21, 0.25, 0.00, 0.07, 0.12, 0.06, -0.03, -0.04],
    [0.24, 0.25, 0.12, 0.00, 0.10, 0.07, 0.02, 0.01],
    [0.61, 0.68, 0.72, 0.64, 0.00, 0.41, 0.32, 0.33],
    [0.52, 0.64, 0.67, 0.62, 0.42, 0.00, 0.12, 0.07],
    [0.41, 0.57, 0.55, 0.54, 0.30, 0.10, 0.00, -0.14],
    [0.25, 0.58, 0.49, 0.50, 0.27, 0.00, -0.18, 0.00],
])


def fit_potential(d):
    """Least-squares c with d[i,j] - d[j,i] ~ c[j] - c[i], sum(c) = 0."""
    n = len(d)
    A = d - d.T
    rows, rhs = [], []
    for i, j in itertools.combinations(range(n), 2):
        r = np.zeros(n); r[j], r[i] = 1, -1
        rows.append(r); rhs.append(A[i, j])
    rows.append(np.ones(n)); rhs.append(0.0)
    c = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return A, c


def check_figure1():
    print("1. Figure 1: d(1->2) - d(2->1) should be I_2 - I_1")
    d = M.T                                   # d[i, j] = d(task i -> task j)
    n = len(d)
    A, c = fit_potential(d)
    iu = np.triu_indices(n, 1)
    res = (A - (c[None, :] - c[:, None]))[iu]
    r2 = 1 - (res**2).sum() / (A[iu] ** 2).sum()
    print(f"   antisymmetric part: rms {np.sqrt(np.mean(A[iu] ** 2)):.3f}; after fitting c_j - c_i: "
          f"rms residual {np.sqrt(np.mean(res**2)):.4f}, max {np.abs(res).max():.4f}, R^2 = {r2:.5f}")
    print("   (each entry is rounded to 0.01, so a difference of two entries carries up to 0.01 of rounding)")
    cyc = [abs(A[i, j] + A[j, k] + A[k, i]) for i, j, k in itertools.combinations(range(n), 3)]
    print(f"   cycle sums A_ij + A_jk + A_ki over all 56 triangles: max {max(cyc):.3f}, mean {np.mean(cyc):.4f}")
    order = np.argsort(c)
    print("   recovered complexity c (up to a constant), simplest first:")
    print("     " + ", ".join(f"{NAMES[i]} {c[i]:+.3f}" for i in order))
    neg = [(NAMES[i], NAMES[j], d[i, j]) for i in range(n) for j in range(n) if d[i, j] < 0]
    print(f"   negative entries: {len(neg)} of 56 off-diagonal; sources involved: "
          f"{sorted(set(s for s, _, _ in neg))}")
    S = d + d.T
    i, j = np.unravel_index(np.argmin(S + np.eye(n) * 9), S.shape)
    print(f"   symmetric part d(i->j) + d(j->i) = 2 I_12 - I_1 - I_2: smallest {S[i, j]:+.2f} "
          f"for ({NAMES[i]}, {NAMES[j]}): the union needs less information than either part")
    viol, worst = 0, (0, None)
    for a, b, e in itertools.permutations(range(n), 3):
        v = d[a, e] - d[a, b] - d[b, e]
        if v > 1e-9:
            viol += 1
            if v > worst[0]:
                worst = (v, (a, b, e))
    a, b, e = worst[1]
    print(f"   triangle inequality d(a->e) <= d(a->b) + d(b->e): {viol} of 336 ordered triples violate it; worst by "
          f"{worst[0]:.2f}: {NAMES[a]} -> {NAMES[e]} = {d[a, e]:.2f} vs via {NAMES[b]}: {d[a, b]:.2f} + {d[b, e]:.2f}")
    return A, c


# ----------------------------------------------------------------------- 2
B1 = np.array([1.0, 1.0, 0.0])      # task 1 predicts x . (w_s + w_1)
B2 = np.array([1.0, 0.0, 1.0])      # task 2 predicts x . (w_s + w_2)


def g(s):
    return np.log1p(s) - s / (1 + s)


def info(tasks, k, lam=3.0, beta=1.0, sig=1.0):
    """Information KL(Q*||P) of the optimal Q for a set of tasks [(B, N, u)].

    Population design: X'X = N I and X'y = N u for each task. The Hessian is
    (sum_i N_i B_i B_i' / sig^2) (x) I_k, so every quantity splits into 3 x 3
    blocks, one per coordinate of the k-dimensional input."""
    Mh = sum(N * np.outer(B, B) for B, N, _ in tasks) / sig**2
    A = Mh + beta / lam**2 * np.eye(3)
    mu2 = 0.0
    for c in range(k):
        b = sum(N * u[c] * B for B, N, u in tasks) / sig**2
        m = np.linalg.solve(A, b)
        mu2 += m @ m
    h = np.linalg.eigvalsh(Mh)
    return mu2 / (2 * lam**2) + 0.5 * k * g(np.clip(h, 0, None) * lam**2 / beta).sum()


def dist(t1, t2, k):
    """d(1 -> 2) = I(D1 u D2) - I(D1); task 1 in block 1, task 2 in block 2."""
    (N1, u1), (N2, u2) = t1, t2
    return info([(B1, N1, u1), (B2, N2, u2)], k) - info([(B1, N1, u1)], k)


def check_linear_gaussian():
    k, N = 10, 100
    print(f"\n2. Definition 6.1, exactly: linear-Gaussian, k = {k} inputs, 3 blocks (shared, task 1, task 2),")
    print(f"   N = {N} per task, sigma = 1, lam = 3, beta = 1, population design (X'X = N I)")
    e1 = np.eye(k)[0]; e2 = np.eye(k)[1]
    u = e1
    print("   task 2 = task 1 rotated by theta (|u| = 1):")
    for th in (0, 30, 60, 90, 180):
        t = math.radians(th)
        v = math.cos(t) * e1 + math.sin(t) * e2
        d12 = dist((N, u), (N, v), k)
        print(f"     theta = {th:3d} deg: d(1->2) = {d12:.4f}, d(2->1) = {dist((N, v), (N, u), k):.4f}")
    big = 3 * u
    d12, d21 = dist((N, u), (N, big), k), dist((N, big), (N, u), k)
    I1, I2 = info([(B1, N, u)], k), info([(B1, N, big)], k)
    print(f"   |u2| = 3 |u1|, same direction: d(1->2) = {d12:.4f}, d(2->1) = {d21:.4f}; "
          f"difference {d12 - d21:.4f} = I_2 - I_1 = {I2 - I1:.4f}")
    same = dist((N, u), (N, u), k)
    print(f"   a task to an exact copy of itself: d = {same:.3f} nats (Lemma 4.2 wants O(1); here it grows with k and N:")
    for kk, NN in ((10, 100), (10, 10_000), (100, 100)):
        print(f"       k = {kk:3d}, N = {NN:6d}: d(D -> copy of D) = {dist((NN, np.eye(kk)[0]), (NN, np.eye(kk)[0]), kk):8.3f}")
    print("   the same task, target with more samples: d(D with N -> D with m N)")
    for m in (1, 4, 16):
        print(f"       m = {m:2d}: {dist((N, u), (m * N, u), k):.3f} nats")
    Isig = info([(B1, N, u)], k)
    Inoise = info([(B1, N, np.zeros(k))], k)
    print(f"   information of a structured task (|u| = 1): {Isig:.3f} nats; of a pure-noise task (u = 0): {Inoise:.3f} nats")
    print(f"   the labels enter only through |mu|^2 / (2 lam^2) = {Isig - Inoise:.3f}; the log-det term "
          f"{Inoise:.3f} depends on X alone")


# ----------------------------------------------------------------- figures
def figures(A, c):
    from svgkit import Axes, svg, write
    n = len(c)
    body = ['<text class="hd" x="20" y="22">Figure 1 obeys d(1→2) − d(2→1) = I₂ − I₁ to rounding</text>',
            '<text class="sub" x="20" y="38">Left: each pair of tasks, measured asymmetry against the fitted c_j − c_i. Right: the fitted complexities c.</text>']
    ax = Axes(62, 70, 250, 230, (-0.75, 0.75), (-0.75, 0.75))
    body.append(ax.frame([-0.6, -0.3, 0, 0.3, 0.6], [-0.6, -0.3, 0, 0.3, 0.6],
                         "fitted c_j − c_i", "d(i→j) − d(j→i), from the figure", xfmt="{:+.1f}", yfmt="{:+.1f}"))
    body.append(ax.path([-0.75, 0.75], [-0.75, 0.75], "k"))
    for i, j in itertools.combinations(range(n), 2):
        body.append(ax.dot(c[j] - c[i], A[i, j], "bf", 3))
    order = np.argsort(c)
    bx = Axes(470, 70, 200, 230, (-0.62, 0.4), (0, n))
    body.append(f'<line class="ax" x1="{bx.X(0):.1f}" x2="{bx.X(0):.1f}" y1="{bx.y0}" y2="{bx.y0 + bx.h}"/>')
    for r, i in enumerate(order):
        y0, y1 = bx.Y(r + 0.8), bx.Y(r + 0.2)
        x0, x1 = sorted((bx.X(0), bx.X(c[i])))
        body.append(f'<rect class="{"of" if c[i] > 0 else "bf"}" x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="2"/>')
        body.append(f'<text class="lab" x="{bx.x0 - 4}" y="{(y0 + y1) / 2 + 4:.1f}" text-anchor="end">{NAMES[i]}</text>')
        body.append(f'<text class="tiny" x="{(x1 + 4) if c[i] > 0 else (x0 - 4):.1f}" y="{(y0 + y1) / 2 + 3:.1f}" '
                    f'text-anchor="{"start" if c[i] > 0 else "end"}">{c[i]:+.2f}</text>')
    body.append(f'<text class="lab" x="{bx.X(0):.1f}" y="{bx.y0 + bx.h + 28}" text-anchor="middle">c (complexity, up to a constant)</text>')
    body.append('<text class="lab" x="20" y="340">All 28 pairs lie on the diagonal: once c is fixed, the asymmetric half of the 8 × 8 matrix carries no other information.</text>')
    write("figure1-antisymmetry.svg", svg(700, 352, "Figure 1 antisymmetry",
                                          "Measured asymmetry of each task pair in the paper's Figure 1 against the difference of "
                                          "fitted complexities, and the fitted complexities themselves.", body))


if __name__ == "__main__":
    A, c = check_figure1()
    check_linear_gaussian()
    if "--figures" in sys.argv:
        figures(A, c)
