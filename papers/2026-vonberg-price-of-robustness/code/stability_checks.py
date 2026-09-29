#!/usr/bin/env python3
"""The Price of Robustness: numerical checks of the proof steps, and of the
places where the statements need care.

Every number quoted in ../notes.md is printed by this script or by
rf_margin_toy.py.

Checked here:

  1. Margins are 1-Lipschitz on any X; signed distances need path-connectedness
     (Lemma 17). On a finite point cloud, the most disconnected X there is,
     |d_f| can change twice as fast as |x - y|.
  2. The surrogate F = f * min(1, h_f / gamma) is (2/gamma)-Lipschitz with no
     connectedness at all, and |f - F| = max(0, 1 - h_f/gamma) is
     (1/gamma)-Lipschitz. This is the step that would let Theorem 4.2 drop
     its extra conditions.
  3. f itself is 2/m-Lipschitz on the set {h_f >= m} (proof of Thm 4.1).
  4. |sgn - sgn_gamma| is a continuous tent even though sgn jumps (Eq 21).
  5. The worst term of the union bound (App C, Eq 17): AM-GM gives the paper's
     bound for every r in one line, no case split needed.
  6. From a tail bound 2N exp(-lambda r) to an expectation (log 2N + 1)/lambda,
     and the n log 2 that the 2^n subsets leave behind.
  7. Isoperimetry on N(0, I/d): 1-Lipschitz functions concentrate at scale
     1/sqrt(d); Hoeffding does not see d.
  8. The 0-1 loss contraction (Eq 2) with the absolute value in the Rademacher
     complexity: exact enumeration shows the extra 1/(2 sqrt n).
  9. Scales: Theorem 4.1 at p = nd, the C.1 threshold sqrt(c/d), Corollary 6's
     threshold, and where the paper's MLPs sit relative to p = nd.
 10. Corollary 15 as stated: letting the covering scale grow removes p from
     the threshold.

Standard library and numpy only.

Run:  python3 stability_checks.py
"""
from __future__ import annotations

import itertools
import math

import numpy as np

rng = np.random.default_rng(0)


def pairwise(x):
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def margins(x, f):
    """h_f and d_f on a finite X: distance to the nearest point of the other class."""
    D = pairwise(x)
    other = f[:, None] != f[None, :]
    h = np.where(other, D, np.inf).min(1)
    return h, f * h


def lip_ratio(x, v):
    D = pairwise(x)
    dv = np.abs(v[:, None] - v[None, :])
    mask = D > 0
    return (dv[mask] / D[mask]).max()


# --------------------------------------------------------------------- 1-3
def check_lipschitz():
    print("1-3. Lipschitz facts on a finite point cloud (X = 300 points in R^2, random labels)")
    x = rng.standard_normal((300, 2))
    f = np.where(rng.random(300) < 0.5, 1.0, -1.0)
    h, d = margins(x, f)
    print(f"   Lip(h_f) = {lip_ratio(x, h):.3f}   (<= 1 on any X)")
    print(f"   Lip(d_f) = {lip_ratio(x, d):.3f}   (up to 2 without path-connectedness)")
    # the one-line counterexample: X = [0,1] u [2,3], f = +1 then -1
    print("   X = [0,1] u [2,3], f = +1 on the first piece: d_f(1) = +1, d_f(2) = -1, "
          "|d_f(1) - d_f(2)| = 2 = 2 |1 - 2|")
    gamma = np.median(h)
    F = f * np.minimum(1.0, h / gamma)
    R = np.abs(f - F)
    print(f"   gamma = median margin = {gamma:.3f}")
    print(f"   Lip(f * min(1, h/gamma)) * gamma = {lip_ratio(x, F) * gamma:.3f}   (<= 2)")
    print(f"   Lip(|f - F|) * gamma           = {lip_ratio(x, R) * gamma:.3f}   (<= 1)")
    m = np.quantile(h, 0.5)
    keep = h >= m
    print(f"   f restricted to {{h_f >= m}}, m = {m:.3f}: Lip * m = "
          f"{lip_ratio(x[keep], f[keep]) * m:.3f}   (<= 2, the proof of Thm 4.1)")
    print()


# ----------------------------------------------------------------------- 4
def check_tent():
    print("4. |sgn - sgn_gamma| is continuous (Eq 21)")
    gamma = 0.25
    t = np.linspace(-1, 1, 200001)
    sgn = np.where(t >= 0, 1.0, -1.0)
    sgn_g = np.clip(t / gamma, -1, 1)
    tent = np.abs(sgn - sgn_g)
    dt = t[1] - t[0]
    print(f"   largest jump of sgn over one grid step:          {np.abs(np.diff(sgn)).max():.3f}")
    print(f"   largest slope of |sgn - sgn_gamma|, times gamma: {np.abs(np.diff(tent)).max() / dt * gamma:.3f}")
    print(f"   tent equals max(0, 1 - |t|/gamma): "
          f"{np.allclose(tent, np.maximum(0, 1 - np.abs(t) / gamma))}")
    print()


# ----------------------------------------------------------------------- 5
def check_union_worst_term():
    print("5. Worst term of the union bound, App C Eq 17")
    n, c, d, S, c1, c2 = 50, 1.0, 100.0, 0.3, 1 / 8, 1 / 8
    b = d * S ** 2 * c2 / c
    worst = 0.0
    for r in np.linspace(0.01, 40, 400):
        k = np.arange(1, n + 1)                    # k = n - |I|, the points off the good set
        exact = (r ** 2 * c1 / k + k * b).min()
        amgm = 2 * r * S * math.sqrt(d * c1 * c2 / c)
        worst = max(worst, amgm - exact)
    print("   min_k [r^2 c1/k + k d S^2 c2/c]  >=  2 r S sqrt(d c1 c2 / c)  for every r: "
          f"{worst <= 1e-12}  (largest violation {worst:.1e})")
    print("   the bound is a/k + b k >= 2 sqrt(ab), true for all k > 0: Cases I and II are not needed")
    print()


# ----------------------------------------------------------------------- 6
def check_tail_to_mean():
    print("6. Tail bound to expectation")
    for N, lam in ((10.0, 2.0), (1e6, 5.0), (2.0 ** 50 * 1e3, 30.0)):
        r = np.linspace(0, 60 * (math.log(2 * N) + 1) / lam, 400001)
        integral = np.trapezoid(np.minimum(1.0, 2 * N * np.exp(-lam * r)), r)
        print(f"   N = {N:9.3g}, lambda = {lam:4.1f}: int min(1, 2N e^(-lambda r)) dr = "
              f"{integral:.4f};  (log 2N + 1)/lambda = {(math.log(2 * N) + 1) / lam:.4f}")
    n, logF = 1000, 5000.0
    print(f"   with N = |F| 2^n the mean carries (log|F| + n log 2)/n: n = {n}, log|F| = {logF:.0f} "
          f"gives {logF / n:.2f} + {math.log(2):.3f}; the log 2 does not shrink with n")
    print()


# ----------------------------------------------------------------------- 7
def check_isoperimetry():
    print("7. Isoperimetry for N(0, I/d), where c = 1: P(|f - Ef| >= t) <= 2 exp(-d t^2 / 2)")
    m = 20000
    for d in (2, 20, 200):
        x = rng.standard_normal((m, d)) / math.sqrt(d)
        fs = {"x_1": x[:, 0], "||x||": np.linalg.norm(x, axis=1),
              "dist to 5 fixed points": pairwise_to(x, rng.standard_normal((5, d)) / math.sqrt(d))}
        t = 2 / math.sqrt(d)
        bound = 2 * math.exp(-d * t ** 2 / 2)
        parts = ", ".join(f"{k}: sd {v.std():.3f}, P(dev >= t) {np.mean(np.abs(v - v.mean()) >= t):.4f}"
                          for k, v in fs.items())
        print(f"   d = {d:3d}, t = 2/sqrt(d) = {t:.3f}, bound {bound:.4f} | {parts}")
    print("   Hoeffding for a [-1,1]-valued function: 2 exp(-t^2/2), no d. Isoperimetry is sharper"
          " exactly when L < sqrt(d/c) (App A.1).")
    # Fact 2 of App C: the margin itself concentrates, so few points sit near the boundary
    d = 200
    x = rng.standard_normal((200000, d)) / math.sqrt(d)
    b = 0.0
    h = np.abs(x[:, 0] - b)       # f = sgn(x_1 - b); X = R^d, so h_f is the hyperplane distance
    S = h.mean()
    print(f"   halfspace f = sgn(x_1) in d = {d}: S(f) = {S:.4f} ~ sqrt(2/(pi d)) = {math.sqrt(2 / math.pi / d):.4f}")
    print("   this is the paper's 'natural scale' sqrt(c/d): a margin of that size is what a random"
          " hyperplane gets for free")
    print()


def pairwise_to(x, pts):
    return np.linalg.norm(x[:, None, :] - pts[None, :, :], axis=-1).min(1)


# ----------------------------------------------------------------------- 8
def check_contraction():
    print("8. 0-1 loss contraction, Eq 2, with |.| inside the Rademacher complexity")
    n, k = 10, 6
    F = np.where(rng.random((k, n)) < 0.5, 1.0, -1.0)      # 6 classifiers on 10 fixed points
    y = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    loss = (1 - y * F) / 2                                 # 1{f(x) != y}
    sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    R_F = np.mean(np.abs(sig @ F.T).max(1)) / n
    R_L = np.mean(np.abs(sig @ loss.T).max(1)) / n
    R_F_noabs = np.mean((sig @ F.T).max(1)) / n
    R_L_noabs = np.mean((sig @ loss.T).max(1)) / n
    print(f"   without |.|: R(loss o F) = {R_L_noabs:.4f}, R(F)/2 = {R_F_noabs / 2:.4f}  (equal)")
    print(f"   with |.|:    R(loss o F) = {R_L:.4f}, R(F)/2 = {R_F / 2:.4f}, "
          f"R(F)/2 + 1/(2 sqrt n) = {R_F / 2 + 0.5 / math.sqrt(n):.4f}")
    print()


# ----------------------------------------------------------------------- 9
def mlp_params(d_in, width, hidden, classes=10):
    """Fully connected, `hidden` hidden layers of `width`, biases and batch-norm (2 per unit)."""
    p = d_in * width + width
    p += (hidden - 1) * (width * width + width)
    p += width * classes + classes
    p += hidden * 2 * width
    return p


def check_scales():
    print("9. Scales")
    c, d = 1.0, 784
    print(f"   Thm 4.1 at log|F| = nd: sqrt(c) log|F| / (S n sqrt d) = sqrt(c d)/S = {math.sqrt(c * d):.0f}/S "
          f"(d = {d}); vacuous unless S >> {math.sqrt(c * d):.0f}")
    print(f"   Thm 4.2 at log|F| = nd: sqrt(c/S^2 * log|F|/(nd)) = sqrt(c)/S; O(1) at S = O(1)")
    print(f"   C.1: the stability bound beats sqrt(log|F|/n) iff S >= sqrt(c/d) = {math.sqrt(c / d):.4f}")
    K, eps = 1.0, 0.1
    for name, n, d in (("MNIST", 60000, 784), ("CIFAR-10", 50000, 3072)):
        for p in (n, 10 * n, n * d):
            s1 = 3 * K / eps * math.sqrt(c * p / (n * d))
            s2 = math.sqrt(8 * c / d * math.log(6 * K / eps))
            print(f"   Cor 6, {name}, K = 1, eps = 0.1, c = 1, p = {p / n:6.0f} n: "
                  f"S* = max({s1:.3f}, {s2:.3f}) = {max(s1, s2):.3f}")
    print("   Parameter counts of the paper's MLPs against nd:")
    for name, n, d, depth, widths in (("CIFAR-10", 50000, 3072, 8, (256, 2048)),
                                      ("CIFAR-10", 50000, 3072, 4, (128, 2048)),
                                      ("MNIST", 60000, 784, 4, (128, 2048)),
                                      ("MNIST", 60000, 784, 8, (128, 2048, 16384))):
        parts = ", ".join(f"w={w}: {mlp_params(d, w, depth):.2e} = {mlp_params(d, w, depth) / (n * d):.3f} nd"
                          for w in widths)
        print(f"   {name:8s} depth {depth}: nd = {n * d:.2e}; {parts}")
    print()


# ---------------------------------------------------------------------- 10
def check_cor15():
    print("10. Corollary 15 as stated, K = 1, eps = 0.1, c = 1, d = 784, n = 60000, W = J = 1")
    K, eps, c, d, n, W, J = 1.0, 0.1, 1.0, 784, 60000, 1.0, 1.0
    floor = math.sqrt(8 * c / d * math.log(6 * K / eps))
    for p in (1e5, 1e9):
        vals = []
        for et in (1e-3, 1.0, 1e3, 1e9):
            first = 3 * K / eps * math.sqrt(p / (n * d)) * math.sqrt(c * math.log(1 + 60 * W * J / et))
            vals.append(f"eps~={et:g}: {max(first, floor):.3f}")
        print(f"   p = {p:.0e}: threshold on S*/L  " + ";  ".join(vals))
    print(f"   as eps~ -> infinity the threshold falls to {floor:.3f} for every p: a p-free law.")
    print("   Theorem 13 has the term J eps~ / S* that prevents this; the corollary dropped it,"
          " so eps~ must be tied to eps S*/(3KJ).")
    print()


if __name__ == "__main__":
    check_lipschitz()
    check_tent()
    check_union_worst_term()
    check_tail_to_mean()
    check_isoperimetry()
    check_contraction()
    check_scales()
    check_cor15()
