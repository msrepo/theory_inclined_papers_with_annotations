#!/usr/bin/env python3
"""Integral probability metrics: every number quoted in notes.md.

Checked here, in the order the notes use them:

  1. total variation: a four-bin example. The sup over judges 0 <= h <= 1 equals the
     sup over sets, which is half the L1 distance; judges in [-1, 1] give twice that;
  2. two point masses at 0 and D: TV = 1, W1 = D, Dudley = 2D/(D+2), KL = inf,
     and a Gaussian-kernel MMD that saturates at sqrt(2);
  3. a ramp judge h(x) = clip(L (x - c), -a, a) has gap L * (signed area
     between the two CDFs over its slope window) -- checked three ways;
  4. the Gaussian shift N(0,1) vs N(m,1): KL, W1, TV, Kolmogorov, Dudley
     (ramp family, exact for a symmetric single hump), and Dudley / W1 as m -> 0;
  5. narrowing bumps: KL blows up, TV saturates at 1, Dudley -> 2D/(D+2);
  6. W1 in 1-D from samples: sorted-sample formula = area between empirical
     CDFs = Hungarian assignment, and the dual judge read off the potentials;
  7. Dudley on samples, two independent ways: ramp search vs the truncated-
     cost optimal-transport route max_a (1-a) W^(tau), tau = 2a/(1-a);
  7b. the best judge of every shape on a grid (dynamic programming), against
     the best ramp, for the shift pair and for a two-humped Q;
  8. MMD: closed form for two Gaussians, sample estimates, the witness
     function, the unit-ball claim, MMD <= 2 TV and MMD <= (1+1/sigma) Dudley,
     and what the bandwidth does (huge: means only, tiny: blind); 8b: the
     three sample pairs of the interactive page across bandwidths;
  9. panels: a bigger panel gives a bigger sup; lower bounds by one judge
     (cos(y)/2) and the ceiling that step (29) of Tahir et al. ignores, the
     panel a downstream task sees, and what a bounded panel cannot see;
 10. the summary table and the test vectors the interactive page checks
     itself against (it recomputes 45 numbers on load).

With --figures it also regenerates the SVGs in ../figures/ from these same
computations, so every number drawn is a real one.

Standard library and numpy only.

Run:  python3 ipm.py            (checks)
      python3 ipm.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np

np.set_printoptions(precision=4, suppress=True)

SQ2 = math.sqrt(2.0)
SQ2PI = math.sqrt(2.0 * math.pi)
verf = np.vectorize(math.erf)


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ------------------------------------------------------------------ Gaussian toolkit

def Phi(z):
    """Standard normal CDF (scalar or array)."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + verf(z / SQ2))


def pdf(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / SQ2PI


def Iphi(z):
    """Antiderivative of Phi: int Phi = z Phi(z) + pdf(z)."""
    z = np.asarray(z, dtype=float)
    return z * Phi(z) + pdf(z)


def clip_mean(mu, s, lo, hi):
    """E clip(X, lo, hi) for X ~ N(mu, s^2), in closed form."""
    al, be = (lo - mu) / s, (hi - mu) / s
    return (lo * Phi(al) + hi * (1.0 - Phi(be))
            + mu * (Phi(be) - Phi(al)) - s * (pdf(be) - pdf(al)))


def ramp_gap_gauss(mp, sp, mq, sq, c, a, L):
    """E_P h - E_Q h for h(x) = clip(L (x - c), -a, a), P = N(mp, sp^2), Q = N(mq, sq^2).
    h = L * (clip(x, c - a/L, c + a/L) - c), so the closed form above applies."""
    lo, hi = c - a / L, c + a / L
    return L * (clip_mean(mp, sp, lo, hi) - clip_mean(mq, sq, lo, hi))


def kl_gauss(m1, s1, m2, s2):
    return math.log(s2 / s1) + (s1 * s1 + (m1 - m2) ** 2) / (2 * s2 * s2) - 0.5


def tv_shift(m, s=1.0):
    return float(2 * Phi(abs(m) / (2 * s)) - 1)


def window_area(m, s, w):
    """int over the window [m/2 - w/2, m/2 + w/2] of (F - G), P = N(0,s^2), Q = N(m,s^2), m > 0.
    F - G is symmetric about m/2 and unimodal, so this centred window is the best one."""
    lo, hi = m / 2 - w / 2, m / 2 + w / 2
    return float(s * (Iphi(hi / s) - Iphi(lo / s)) - s * (Iphi((hi - m) / s) - Iphi((lo - m) / s)))


A_GRID = np.linspace(0.0005, 0.9995, 1999)


def dudley_shift(m, s=1.0, return_a=False):
    """Dudley distance (sum-norm ball ||h||_L + ||h||_inf <= 1) for N(0,s^2) vs N(m,s^2):
    max over a of L * area(window of length 2a/L), with L = 1 - a."""
    m = abs(m)
    if m == 0:
        return (0.0, 0.0) if return_a else 0.0
    L = 1.0 - A_GRID
    w = 2 * A_GRID / L
    vals = np.array([Lk * window_area(m, s, wk) for Lk, wk in zip(L, w)])
    k = int(np.argmax(vals))
    return (float(vals[k]), float(A_GRID[k])) if return_a else float(vals[k])


def dudley_points(D):
    return 2.0 * D / (D + 2.0)


def w1_shift(m):
    return abs(m)


def mmd2_gauss(m, s, sig):
    """MMD^2 with k(x,y) = exp(-(x-y)^2 / (2 sig^2)) between N(0,s^2) and N(m,s^2).
    E k(x,y) for x - y ~ N(mu, v) is sig/sqrt(sig^2+v) * exp(-mu^2 / (2 (sig^2+v)))."""
    v = 2 * s * s
    kself = sig / math.sqrt(sig * sig + v)
    kcross = kself * math.exp(-m * m / (2 * (sig * sig + v)))
    return 2 * kself - 2 * kcross


def mmd2_pop_mix(P, Q, sig):
    """Population MMD^2 (Gaussian kernel, bandwidth sig) between Gaussian mixtures [(weight, mean, sd), ...]."""
    def E(A, B):
        t = 0.0
        for w1, m1, s1 in A:
            for w2, m2, s2 in B:
                v = s1 * s1 + s2 * s2
                t += w1 * w2 * sig / math.sqrt(sig * sig + v) * math.exp(-(m1 - m2) ** 2 / (2 * (sig * sig + v)))
        return t
    return E(P, P) + E(Q, Q) - 2 * E(P, Q)


PRESETS = {  # the three pairs of the interactive page, all with mean 0 in the last two
    "means": ([(1, 0, 1)], [(1, 1.5, 1)]),
    "width": ([(1, 0, 1)], [(1, 0, 2)]),
    "humps": ([(1, 0, 1.34164)], [(0.5, -1.2, 0.6), (0.5, 1.2, 0.6)]),
}


def mmd_shift(m, s=1.0, sig=1.0):
    return math.sqrt(max(mmd2_gauss(m, s, sig), 0.0))


# ------------------------------------------------------------------ 1. total variation

P4 = np.array([0.5, 0.3, 0.2, 0.0])
Q4 = np.array([0.2, 0.3, 0.1, 0.4])


def check_tv():
    section("1. Total variation on four bins")
    print("P =", P4, " Q =", Q4)
    tv = 0.5 * np.abs(P4 - Q4).sum()
    print(f"half the L1 distance, (1/2) sum|p_i - q_i| = {tv:.4f}")
    best, winners = 0.0, []
    for r in range(1, 5):
        for A in itertools.combinations(range(4), r):
            g = abs(P4[list(A)].sum() - Q4[list(A)].sum())
            if g > best + 1e-12:
                best, winners = g, [A]
            elif abs(g - best) <= 1e-12:
                winners.append(A)
    A_star = tuple(int(i) for i in np.where(P4 > Q4)[0])
    print(f"sup over the 15 non-empty bin sets |P(A) - Q(A)| = {best:.4f}; the sets that attain it: "
          + ", ".join("{" + ",".join(str(i + 1) for i in A) + "}" for A in winners))
    print(f"the set {{p > q}} = bins {[i + 1 for i in A_star]}: P(A) = {P4[list(A_star)].sum():.2f}, Q(A) = {Q4[list(A_star)].sum():.2f}, gap {P4[list(A_star)].sum() - Q4[list(A_star)].sum():.2f}")
    # h in [0,1]^4: the gap is linear in h, so the sup sits at a vertex of the cube (0/1 vectors)
    rng = np.random.default_rng(0)
    H = rng.random((200000, 4))
    gaps = np.abs(H @ (P4 - Q4))
    print(f"200000 random judges 0 <= h <= 1: largest gap {gaps.max():.4f} (never above {tv:.4f})")
    sgn = np.sign(P4 - Q4)
    print(f"judge h = sign(p - q) with |h| <= 1: gap {abs(sgn @ (P4 - Q4)):.4f} = 2 TV = {2 * tv:.4f}")
    F, G = np.cumsum(P4), np.cumsum(Q4)
    print(f"Kolmogorov distance (half-line indicators) = max|F - G| = {np.abs(F - G).max():.4f}; "
          f"W1 with bins at 0,1,2,3 = sum|F - G| = {np.abs(F - G)[:-1].sum():.4f}")
    return tv


# ------------------------------------------------------------------ 2. two point masses

def check_points():
    section("2. Two point masses at 0 and D")
    print("  D      TV    W1    Dudley 2D/(D+2)   grid-max_a min(2a,(1-a)D)   argmax a = D/(D+2)   L = 2/(D+2)   MMD(sig=1)")
    a = np.linspace(0, 1, 200001)
    for D in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0):
        best = np.minimum(2 * a, (1 - a) * D)
        k = int(np.argmax(best))
        mm = math.sqrt(2 - 2 * math.exp(-D * D / 2))
        print(f"{D:6g}  1.00  {D:6g}  {dudley_points(D):.6f}          {best[k]:.6f}                "
              f"{a[k]:.5f} vs {D / (D + 2):.5f}   {2 / (D + 2):.5f}    {mm:.4f}")
    print("KL(P||Q) = +inf for every D > 0 (P puts mass where Q has none).")
    print(f"MMD (Gaussian kernel, any bandwidth) saturates at sqrt(2) = {SQ2:.4f} when the masses are far apart.")
    D = 1.0
    print(f"D = 1: Dudley = {dudley_points(D):.4f} against W1 = 1 and TV = 1; ratio Dudley/W1 = 2/(D+2), which is "
          f"{dudley_points(0.01) / 0.01:.4f} at D = 0.01 and {dudley_points(100.0) / 100.0:.4f} at D = 100")
    # the closed form again from the window picture: gap = L * min(window, D), window = 2a/L
    for D in (1.0, 5.0):
        a_star = D / (D + 2)
        L = 1 - a_star
        w = 2 * a_star / L
        print(f"D = {D:g}: best a = {a_star:.4f}, L = {L:.4f}, window length 2a/L = {w:.4f} = D: "
              f"gap = L * min(w, D) = {L * min(w, D):.4f}")


# ------------------------------------------------------------------ 3. ramp gap = L * area

def quad_gap(mp, sp, mq, sq, c, a, L, n=400001, span=14.0):
    """Brute-force trapezoid of int h (p - q) dx, independent of the closed forms."""
    lo = min(mp - span * sp, mq - span * sq)
    hi = max(mp + span * sp, mq + span * sq)
    x = np.linspace(lo, hi, n)
    h = np.clip(L * (x - c), -a, a)
    p = pdf((x - mp) / sp) / sp
    q = pdf((x - mq) / sq) / sq
    f = h * (p - q)
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x)))


def check_ramp_identity():
    section("3. A ramp judge sees L x (signed area between the CDFs over its slope window)")
    print("P = N(0,1), Q = N(1,1), ramp h(x) = clip(L (x - c), -a, a) with a = 0.6, L = 0.4, c = 0.5")
    m, a, L, c = 1.0, 0.6, 0.4, 0.5
    lo, hi = c - a / L, c + a / L
    g_closed = ramp_gap_gauss(0, 1, m, 1, c, a, L)
    g_quad = quad_gap(0, 1, m, 1, c, a, L)
    x = np.linspace(lo, hi, 200001)
    diff = Phi(x) - Phi(x - m)  # F - G
    area = float(np.sum(0.5 * (diff[1:] + diff[:-1]) * np.diff(x)))
    print(f"E_P h - E_Q h: closed form {g_closed:+.6f}, trapezoid of int h (p - q) {g_quad:+.6f}")
    print(f"slope window [{lo:.2f}, {hi:.2f}], area of F - G over it = {area:.6f}; L * area = {L * area:.6f}"
          f"  (gap = -L * int (F - G) = {-L * area:+.6f} for an increasing ramp)")
    print("window length 2a/L =", 2 * a / L, "; whole-line limit with L = 1 is W1 = int |F - G| =",
          f"{w1_quadrature(0, 1, m, 1):.6f}")


def w1_quadrature(mp, sp, mq, sq, n=400001, span=14.0):
    lo = min(mp - span * sp, mq - span * sq)
    hi = max(mp + span * sp, mq + span * sq)
    x = np.linspace(lo, hi, n)
    d = np.abs(Phi((x - mp) / sp) - Phi((x - mq) / sq))
    return float(np.sum(0.5 * (d[1:] + d[:-1]) * np.diff(x)))


# ------------------------------------------------------------------ 4. Gaussian shift

def dudley_ramp_search(m, s=1.0):
    """Search the ramp family directly (centre c, a + L = 1) with the closed-form gap."""
    best = (0.0, None, None)
    for a in np.linspace(0.01, 0.99, 99):
        L = 1 - a
        cs = np.linspace(-3 * s, m + 3 * s, 601)
        gaps = np.abs(ramp_gap_gauss(0, s, m, s, cs, a, L))
        k = int(np.argmax(gaps))
        if gaps[k] > best[0]:
            best = (float(gaps[k]), float(a), float(cs[k]))
    return best


def check_gauss_shift():
    section("4. Gaussian shift, N(0,1) vs N(m,1)")
    print("  m     KL=m^2/2    W1=|m|    W1 (quadrature)   TV=2Phi(m/2)-1   Kolmogorov   Dudley(window)  ramp search: value  a    c")
    rows = {}
    for m in (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        dud, a_w = dudley_shift(m, 1.0, return_a=True)
        gs, a_s, c_s = dudley_ramp_search(m)
        xs = np.linspace(-12, 12 + m, 300001)
        kol = float(np.max(np.abs(Phi(xs) - Phi(xs - m))))
        w1q = w1_quadrature(0, 1, m, 1)
        rows[m] = (dud, a_w)
        print(f"{m:5g}  {m * m / 2:9.4f}  {m:7.4f}   {w1q:9.4f}        {tv_shift(m):.4f}         {kol:.4f}     "
              f"{dud:.4f}         {gs:.4f}  {a_s:.2f}  {c_s:.2f}   (best a, window: {a_w:.3f})")
    print("KL(N(0,1)||N(m,1)) computed from the general formula at m=1:", f"{kl_gauss(0, 1, 1, 1):.4f}")
    d1, a1 = dudley_shift(1.0, 1.0, return_a=True)
    print(f"m = 1: Dudley = {d1:.4f} at a = {a1:.3f} (L = {1 - a1:.3f}); TV = {tv_shift(1.0):.4f}, 2TV = {2 * tv_shift(1.0):.4f}, W1 = 1")
    print("centre of the optimal ramp is m/2 by symmetry (the search above lands on c = 0.5 for m = 1)")
    # Dudley / W1 as m -> 0, for different widths of the two distributions
    print("\nDudley / W1 as m -> 0 (equal widths s):")
    for s in (2.0, 1.0, 0.3, 0.1, 0.03, 0.01):
        r = dudley_shift(1e-4 * s, s) / (1e-4 * s)
        print(f"  s = {s:5g}: ratio {r:.4f}")
    # max of (1-a)(2 Phi(a/(1-a)) - 1) for s = 1 (the limit constant)
    a = np.linspace(0.001, 0.999, 9999)
    lim = (1 - a) * (2 * Phi(a / (1 - a)) - 1)
    k = int(np.argmax(lim))
    print(f"limit constant for s = 1: max_a (1-a)(2 Phi(a/(1-a)) - 1) = {lim[k]:.4f} at a = {a[k]:.3f}")
    # Pinsker
    print(f"Pinsker at m = 1: TV = {tv_shift(1.0):.4f} <= sqrt(KL/2) = {math.sqrt(0.5 / 2):.4f}")
    return rows


# ------------------------------------------------------------------ 5. narrowing bumps

def check_bumps():
    section("5. Narrowing bumps: N(0, s^2) vs N(D, s^2), D = 1")
    print("  s      KL = D^2/(2 s^2)    TV        W1       Dudley     MMD(sig=1)    (Dudley limit 2D/(D+2) = 0.6667)")
    for s in (1.0, 0.5, 0.3, 0.1, 0.03):
        kl = 1.0 / (2 * s * s)
        print(f"{s:5g}  {kl:12.2f}      {tv_shift(1.0, s):.4f}   1.0000   {dudley_shift(1.0, s):.4f}     "
              f"{mmd_shift(1.0, s, 1.0):.4f}")


# ------------------------------------------------------------------ 6. W1 from samples, Hungarian

def hungarian(cost):
    """Minimum-cost perfect matching, O(n^3) potentials method.
    Returns (total, assignment, (u, v)) with dual potentials u_i + v_j <= cost_ij, tight on the matching."""
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1, :] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        assign[p[j] - 1] = j - 1
    return float(cost[np.arange(n), assign].sum()), assign, (u[1:].copy(), v[1:].copy())


def ot_truncated(x, y, tau):
    """Equal-size empirical OT with ground cost min(|x - y|, tau): an assignment problem."""
    C = np.minimum(np.abs(x[:, None] - y[None, :]), tau)
    return hungarian(C)[0] / len(x)


def w1_area(x, y):
    """int |F_x - F_y| dt for two equal-size empirical samples (exact staircase integral)."""
    pts = np.sort(np.concatenate([x, y]))
    Fx = np.searchsorted(np.sort(x), pts[:-1], side="right") / len(x)
    Fy = np.searchsorted(np.sort(y), pts[:-1], side="right") / len(y)
    return float(np.sum(np.abs(Fx - Fy) * np.diff(pts)))


def make_samples(n=60, m=1.0, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n), rng.normal(m, 1, n)


def check_w1_samples():
    section("6. W1 in one dimension from samples")
    # Hungarian against brute force on n = 6
    rng = np.random.default_rng(11)
    C = rng.random((6, 6))
    brute = min(sum(C[i, s[i]] for i in range(6)) for s in itertools.permutations(range(6)))
    print(f"Hungarian vs all 720 permutations, random 6x6 cost: {hungarian(C)[0]:.10f} vs {brute:.10f}")
    x, y = make_samples()
    sm = float(np.mean(np.abs(np.sort(x) - np.sort(y))))
    ar = w1_area(x, y)
    hg = ot_truncated(x, y, 1e6)
    print(f"n = 60 samples of N(0,1) and N(1,1):")
    print(f"  sorted-sample formula mean|x_(i) - y_(i)| = {sm:.6f}")
    print(f"  area between the empirical CDFs          = {ar:.6f}")
    print(f"  Hungarian assignment, cost |x - y|       = {hg:.6f}")
    print(f"  (population W1 = 1. The sample means differ by {abs(y.mean() - x.mean()):.6f}; W1 >= |difference of means| always, "
          f"with equality here because y_(i) > x_(i) for every i: {bool(np.all(np.sort(y) > np.sort(x)))})")
    # duality: the Hungarian potentials give a judge h(t) = min_j (|t - y_j| - v_j); it is 1-Lipschitz and earns exactly W1
    tot, _, (u, v) = hungarian(np.abs(x[:, None] - y[None, :]))
    hj = lambda t: np.min(np.abs(t[:, None] - y[None, :]) - v[None, :], axis=1)
    pts = np.concatenate([x, y])
    hp = hj(pts)
    dpt = np.abs(pts[:, None] - pts[None, :]) + np.eye(len(pts))
    slope = float(np.max(np.abs(hp[:, None] - hp[None, :]) / dpt))
    print(f"  dual judge from the Hungarian potentials: largest slope between the 120 sample points {slope:.4f}, "
          f"gap E_P h - E_Q h = {hj(x).mean() - hj(y).mean():.6f} (= W1 = {tot / 60:.6f})")
    x2, y2 = make_samples(seed=5)
    print(f"  second seed: sorted {np.mean(np.abs(np.sort(x2) - np.sort(y2))):.6f}, Hungarian {ot_truncated(x2, y2, 1e6):.6f}")
    print(f"  random pairing (not sorted) costs {np.mean(np.abs(x - y[np.random.default_rng(1).permutation(60)])):.4f}, more than {sm:.4f}")


# ------------------------------------------------------------------ 7. Dudley on samples, two ways

def ramp_search_samples(x, y, a, ngrid=1601):
    """max over c of |mean h(x) - mean h(y)| for h = clip(L (t - c), -a, a), L = 1 - a."""
    L = 1.0 - a
    lo = min(x.min(), y.min()) - 2.0
    hi = max(x.max(), y.max()) + 2.0
    cs = np.linspace(lo, hi, ngrid)[:, None]
    hx = np.clip(L * (x[None, :] - cs), -a, a).mean(1)
    hy = np.clip(L * (y[None, :] - cs), -a, a).mean(1)
    g = np.abs(hx - hy)
    k = int(np.argmax(g))
    return float(g[k]), float(cs[k, 0])


def check_dudley_samples():
    section("7. Dudley on samples: ramp search versus optimal transport with the truncated cost")
    out = {}
    for label, (x, y) in (("random Gaussian samples, n = 60, seed 3", make_samples(60, 1.0, 3)),
                          ("quantile grids of N(0,1), N(1,1), n = 60", quantile_pair(60, 1.0))):
        avals = np.round(np.arange(0.02, 0.9801, 0.02), 4)
        ramp, ot = [], []
        for a in avals:
            r, _ = ramp_search_samples(x, y, a)
            tau = 2 * a / (1 - a)
            ot.append((1 - a) * ot_truncated(x, y, tau))
            ramp.append(r)
        ramp, ot = np.array(ramp), np.array(ot)
        diff = ot - ramp
        kr, ko = int(np.argmax(ramp)), int(np.argmax(ot))
        print(f"{label}:")
        print(f"  best ramp:  {ramp[kr]:.4f} at a = {avals[kr]:.2f}    best OT route: {ot[ko]:.4f} at a = {avals[ko]:.2f}")
        print(f"  per-a difference (OT - ramp): min {diff.min():+.5f}, max {diff.max():.5f}, at the best a: {ot[ko] - ramp[ko]:.5f}")
        print(f"  OT route >= ramp for every a: {bool(np.all(diff >= -1e-9))}")
        out[label] = (avals, ramp, ot)
    # refine both around the optimum on the first data set
    x, y = make_samples(60, 1.0, 3)
    fine = np.arange(0.36, 0.5601, 0.005)
    r_f = max(ramp_search_samples(x, y, a, 4001)[0] for a in fine)
    o_f = max((1 - a) * ot_truncated(x, y, 2 * a / (1 - a)) for a in fine)
    print(f"refined maximum over a in [0.36, 0.56] (step 0.005), random samples: ramp {r_f:.4f}, OT route {o_f:.4f}")
    a_pop = dudley_shift(1.0)
    print(f"population value for N(0,1) vs N(1,1): {a_pop:.4f}")
    a = 0.57
    print(f"\nHow far the ramp falls short of the exact optimum at a = {a} (population ramp value {dudley_shift(1.0):.4f} at its best a):")
    print("  quantile grids  n:  ramp     OT      OT - ramp")
    for n in (30, 60, 120, 200):
        x, y = quantile_pair(n, 1.0)
        r, _ = ramp_search_samples(x, y, a, 3001)
        o = (1 - a) * ot_truncated(x, y, 2 * a / (1 - a))
        print(f"  {n:15d}    {r:.4f}   {o:.4f}   {o - r:.5f}")
    print("  random samples (seeds 0-3), n = 60 and n = 200: OT - ramp")
    for n in (60, 200):
        dd = []
        for seed in range(4):
            rng = np.random.default_rng(seed)
            x, y = rng.normal(0, 1, n), rng.normal(1, 1, n)
            r, _ = ramp_search_samples(x, y, a, 3001)
            o = (1 - a) * ot_truncated(x, y, 2 * a / (1 - a))
            dd.append(o - r)
        print(f"    n = {n:3d}: " + "  ".join(f"{v:.4f}" for v in dd))
    return out


def quantile_pair(n, m):
    """Deterministic samples: the n mid-quantiles of N(0,1) and the same shifted by m."""
    u = (np.arange(n) + 0.5) / n
    z = np.array([norm_ppf(v) for v in u])
    return z, z + m


def norm_ppf(u):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ 7b. every judge shape, by dynamic programming

GX = np.linspace(-12.0, 12.0, 961)
GDX = float(GX[1] - GX[0])


def best_judge_dp(d, dx, a, L, path=False):
    """max over ALL judges h on the grid with |h_i| <= a and |h_(i+1) - h_i| <= L dx of sum_i h_i d_i,
    where d_i = (p_i - q_i) dx. The class is symmetric (h -> -h), so this is also the largest |gap|.
    Chain dynamic programming over h-levels that are multiples of delta = L dx / j, so that the steepest
    allowed step is exactly j levels. j is capped so that the level array stays small."""
    j = int(max(1, min(6, math.floor(1250 * L * dx / a))))
    delta = L * dx / j
    K = int(a / delta)
    levels = np.arange(-K, K + 1) * delta
    f = d[0] * levels
    back = np.zeros((len(d), len(levels)), dtype=np.int16) if path else None
    for i in range(1, len(d)):
        g = f.copy()
        arg = np.zeros(len(levels), dtype=np.int16) if path else None
        for sft in range(1, j + 1):
            m1 = f[:-sft] > g[sft:]
            g[sft:][m1] = f[:-sft][m1]
            if path:
                arg[sft:][m1] = -sft
            m2 = f[sft:] > g[:-sft]
            g[:-sft][m2] = f[sft:][m2]
            if path:
                arg[:-sft][m2] = sft
        f = g + d[i] * levels
        if path:
            back[i] = arg
    k = int(np.argmax(f))
    val = float(f[k])
    if not path:
        return val
    traj = [k]
    for i in range(len(d) - 1, 0, -1):
        k += int(back[i][k])
        traj.append(k)
    return val, levels[np.array(traj[::-1])]


def ramp_best_grid(d, x, a, nc=1601):
    """Best |sum_i h_i d_i| over ramps h = clip(L (x - c), -a, a), L = 1 - a, on the same grid as the DP."""
    L = 1 - a
    cs = np.linspace(-5, 5, nc)[:, None]
    g = np.abs(np.clip(L * (x[None, :] - cs), -a, a) @ d)
    k = int(np.argmax(g))
    return float(g[k]), float(cs[k, 0])


def bimodal_q(x):
    return 0.5 * pdf((x + 1.5) / 0.5) / 0.5 + 0.5 * pdf((x - 1.5) / 0.5) / 0.5


def check_all_shapes():
    section("7b. Every judge shape on a grid (dynamic programming), against the ramp")
    x, dx = GX, GDX
    p = pdf(x)
    print("N(0,1) vs N(1,1): best judge of ANY shape versus the best ramp, at three budgets a (L = 1 - a)")
    for a in (0.3, 0.57, 0.8):
        dp = best_judge_dp((p - pdf(x - 1.0)) * dx, dx, a, 1 - a)
        rp = max(abs(ramp_gap_gauss(0, 1, 1, 1, c, a, 1 - a)) for c in np.linspace(0, 1, 101))
        print(f"  a = {a:.2f}: any shape {dp:.4f}, ramp {rp:.4f}")
    print("Bimodal Q = 0.5 N(-1.5, 0.5^2) + 0.5 N(1.5, 0.5^2) against P = N(0,1) (equal means, a very different shape):")
    d2 = (p - bimodal_q(x)) * dx
    best_dp, best_rp = (0.0, None), (0.0, None, None)
    for a in np.arange(0.1, 0.9001, 0.05):
        L = 1 - a
        v = best_judge_dp(d2, dx, a, L)
        if v > best_dp[0]:
            best_dp = (v, a)
        r, c = ramp_best_grid(d2, x, a)
        if r > best_rp[0]:
            best_rp = (r, a, c)
    print(f"  best any-shape judge: {best_dp[0]:.4f} at a = {best_dp[1]:.2f};  best ramp: {best_rp[0]:.4f} at a = {best_rp[1]:.2f}, c = {best_rp[2]:.2f}")
    print(f"  ratio any-shape / ramp = {best_dp[0] / best_rp[0]:.3f}")
    return best_dp, best_rp


# ------------------------------------------------------------------ 8. MMD

def gk(a, b, sig):
    return np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * sig * sig))


def mmd2_biased(x, y, sig):
    return float(gk(x, x, sig).mean() + gk(y, y, sig).mean() - 2 * gk(x, y, sig).mean())


def mmd2_unbiased(x, y, sig):
    n, m = len(x), len(y)
    Kxx, Kyy, Kxy = gk(x, x, sig), gk(y, y, sig), gk(x, y, sig)
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean())


def witness(x, y, sig, t):
    """w(t) = mean_i k(x_i, t) - mean_j k(y_j, t); the best RKHS judge is w / ||w||."""
    return gk(t, x, sig).mean(1) - gk(t, y, sig).mean(1)


TX = np.array([-1.0, 0.0, 0.5, 2.0])
TY = np.array([0.5, 1.5, 2.0, 3.5])


def check_mmd():
    section("8. MMD: the RKHS unit ball as a panel of judges")
    # closed form vs quadrature vs samples
    m, s, sig = 1.0, 1.0, 1.0
    cf = mmd2_gauss(m, s, sig)
    x = np.linspace(-12, 13, 1501)
    p, q = pdf(x) , pdf(x - m)
    dx = x[1] - x[0]
    K = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * sig * sig))
    d = (p - q) * dx
    quad = float(d @ K @ d)
    ub, bi = [], []
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        xs, ys = rng.normal(0, 1, 500), rng.normal(m, 1, 500)
        ub.append(mmd2_unbiased(xs, ys, sig))
        bi.append(mmd2_biased(xs, ys, sig))
    ub, bi = np.array(ub), np.array(bi)
    print(f"N(0,1) vs N(1,1), Gaussian kernel sigma = 1:")
    print(f"  MMD^2 closed form 2/sqrt(3) (1 - e^(-m^2/6)) = {cf:.5f},  MMD = {math.sqrt(cf):.4f}")
    print(f"  MMD^2 by quadrature (p - q)^T K (p - q)      = {quad:.5f}")
    print(f"  MMD^2 from 500 + 500 samples, mean over 40 draws: unbiased {ub.mean():.4f} (s.e. {ub.std(ddof=1) / math.sqrt(40):.4f}), "
          f"biased {bi.mean():.4f}; one draw has s.d. {ub.std(ddof=1):.4f}")
    print("  MMD(m, sigma=1) for m = 0.5, 1, 2, 4, 8:",
          "  ".join(f"{mmd_shift(mm):.4f}" for mm in (0.5, 1, 2, 4, 8)), f" (limit {math.sqrt(2 / math.sqrt(3)):.4f} = sqrt(2 sig / sqrt(sig^2 + 2 s^2)))")

    # the witness and the unit-ball sup on the four-point example
    print("\nFour-point example: X = -1, 0, 0.5, 2 and Y = 0.5, 1.5, 2, 3.5, sigma = 1")
    mb = mmd2_biased(TX, TY, 1.0)
    print(f"  biased MMD^2 = {mb:.6f}, MMD = {math.sqrt(mb):.6f};   unbiased MMD^2 = {mmd2_unbiased(TX, TY, 1.0):.6f}")
    pool = np.concatenate([TX, TY])
    beta = np.concatenate([np.full(4, 0.25), np.full(4, -0.25)])  # mu_P - mu_Q as weights on k(pool_i, .)
    Kp = gk(pool, pool, 1.0)
    nrm = math.sqrt(beta @ Kp @ beta)
    fstar = beta / nrm
    wP = np.concatenate([np.full(4, 0.25), np.zeros(4)])
    wQ = np.concatenate([np.zeros(4), np.full(4, 0.25)])
    gap_star = float((wP - wQ) @ Kp @ fstar)  # E_P f* - E_Q f*, f*(z) = sum_i fstar_i k(pool_i, z)
    print(f"  ||mu_P - mu_Q|| = {nrm:.6f};  the witness judge f* = (mu_P - mu_Q)/||.|| earns E_P f* - E_Q f* = {gap_star:.6f}")
    rng = np.random.default_rng(4)
    best = 0.0
    for _ in range(200000):
        al = rng.normal(size=8)
        al = al / math.sqrt(al @ Kp @ al)
        gap = abs((wP - wQ) @ Kp @ al)
        best = max(best, gap)
    print(f"  200000 random unit-norm RKHS judges: best gap {best:.6f} (never above {nrm:.6f})")
    t = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    print("  witness w(t) at t = -1, 0, 1, 2, 3:", witness(TX, TY, 1.0, t))
    wx = witness(TX, TY, 1.0, TX).mean() - witness(TX, TY, 1.0, TY).mean()
    print(f"  identity: mean w over X minus mean w over Y = {wx:.6f} = MMD^2 = {mb:.6f}")

    # MMD <= 2 TV and MMD <= (1 + 1/sigma) Dudley
    print("\nSandwich: unit ball of the Gaussian-kernel RKHS sits inside {|h| <= 1, Lip <= 1/sigma}")
    print("  m     MMD(sig=1)   2TV      (1+1/sig) Dudley     MMD(sig=0.5)   (1+2) Dudley")
    for mm in (0.25, 0.5, 1.0, 2.0, 4.0):
        d = dudley_shift(mm)
        print(f"{mm:5g}   {mmd_shift(mm, 1, 1.0):.4f}     {2 * tv_shift(mm):.4f}   {2 * d:.4f}              "
              f"{mmd_shift(mm, 1, 0.5):.4f}       {3 * d:.4f}")

    # bandwidth: huge sigma sees only means, tiny sigma sees nothing
    print("\nBandwidth. Pair A: N(0,1) vs N(1,1) (means differ). Pair B: N(0,1) vs N(0,2^2) (same mean, different width).")
    print("  sigma      MMD A      MMD B      sigma*MMD A    sigma*MMD B")
    for sg in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        a_ = math.sqrt(mmd2_gauss(1.0, 1.0, sg))
        b_ = math.sqrt(max(mmd2_pair_widths(1.0, 2.0, sg), 0.0))
        print(f"{sg:7g}   {a_:.5f}   {b_:.5f}    {sg * a_:.5f}       {sg * b_:.5f}")
    print("  large sigma: MMD -> |difference of means| / sigma  (sigma * MMD A -> 1), and pair B, equal means, goes to 0 faster")
    # sample version of the tiny-sigma blindness
    xs2, ys2 = np.random.default_rng(6).normal(0, 1, 40), np.random.default_rng(7).normal(1.5, 1, 40)
    xs3, ys3 = np.random.default_rng(6).normal(0, 1, 40), np.random.default_rng(8).normal(0, 1, 40)
    print(f"  40 + 40 samples, tiny sigma = 0.001: biased MMD^2 = {mmd2_biased(xs2, ys2, 0.001):.5f} (shifted pair) and "
          f"{mmd2_biased(xs3, ys3, 0.001):.5f} (same law); the constant 1/n + 1/m = {1 / 40 + 1 / 40:.5f}")


def check_presets():
    section("8b. MMD^2 against the bandwidth for the three pairs of the interactive page (population, Gaussian kernel)")
    sigs = np.logspace(-1.3, 1.3, 400)
    print("  sigma:        " + "  ".join(f"{v:8g}" for v in (0.05, 0.3, 1.0, 3.0, 20.0)))
    for name, (Pm, Qm) in PRESETS.items():
        vals = [mmd2_pop_mix(Pm, Qm, v) for v in (0.05, 0.3, 1.0, 3.0, 20.0)]
        curve = np.array([mmd2_pop_mix(Pm, Qm, v) for v in sigs])
        k = int(np.argmax(curve))
        print(f"  {name:6s} MMD^2 " + "  ".join(f"{v:8.5f}" for v in vals) + f"   peak {curve[k]:.5f} at sigma = {sigs[k]:.2f}")
    print("  (means: same as the closed form 2/sqrt(3) (1 - exp(-m^2/6)) at sigma = 1 with m = 1.5:",
          f"{mmd2_gauss(1.5, 1.0, 1.0):.5f})")
    P, Q = PRESETS["humps"]
    mv = sum(w * m for w, m, s_ in P), sum(w * m for w, m, s_ in Q)
    vv = sum(w * (m * m + s_ * s_) for w, m, s_ in P), sum(w * (m * m + s_ * s_) for w, m, s_ in Q)
    print(f"  humps pair: means {mv[0]:.3f} and {mv[1]:.3f}; second moments {vv[0]:.4f} and {vv[1]:.4f} (equal)")


def mmd2_pair_widths(s1, s2, sig):
    """MMD^2, Gaussian kernel, between N(0,s1^2) and N(0,s2^2)."""
    def kk(v):
        return sig / math.sqrt(sig * sig + v)
    return kk(2 * s1 * s1) + kk(2 * s2 * s2) - 2 * kk(s1 * s1 + s2 * s2)


# ------------------------------------------------------------------ 9. panels

def check_panels():
    section("9. Panels, lower bounds by a single judge, and what a panel cannot see")
    # bigger panel, bigger sup: N(0,1) vs N(1,1)
    m = 1.0
    print(f"nested panels, N(0,1) vs N(1,1): Dudley {dudley_shift(m):.4f} <= 2TV {2 * tv_shift(m):.4f} and Dudley <= W1 {m:.4f}; "
          f"TV {tv_shift(m):.4f} <= (|h|<=1 gives 2TV) {2 * tv_shift(m):.4f}")
    print(f"  MMD (sigma = 1) {mmd_shift(m):.4f} <= (1 + 1/sigma) Dudley {2 * dudley_shift(m):.4f}")

    # cos judge
    print("\nOne judge is enough for a lower bound: h(y) = cos(y)/2 has ||h||_L = 1/2 and ||h||_inf = 1/2, so ||h||_BL = 1.")
    sg = 1.0
    for f_, g_ in ((0.0, math.pi), (0.0, 2.0), (0.0, 1.0)):
        gap = math.exp(-sg * sg / 2) * abs(math.cos(f_) - math.cos(g_)) / 2
        true = dudley_shift(g_ - f_, sg)
        # verify E cos(y) = exp(-sigma^2/2) cos(f) by quadrature
        xq = np.linspace(-12, 12, 200001)
        Ecos = float(np.sum(np.cos(xq) * pdf((xq - f_) / sg) / sg) * (xq[1] - xq[0]))
        print(f"  N({f_:g},1) vs N({g_:.4f},1): gap of cos/2 = {gap:.4f}  (E cos(y) = {Ecos:.5f} vs e^(-1/2) cos f = {math.exp(-0.5) * math.cos(f_):.5f});"
              f" true Dudley {true:.4f}, so cos/2 recovers {gap / true:.0%}")

    # the bound printed in Tahir et al. (2025), eq. (29), against what the cos judge can earn
    gs_ = np.linspace(0.0, 60.0, 600001)
    gaps = math.exp(-0.5) * np.abs(1 - np.cos(gs_)) / 2
    k = int(np.argmax(gaps))
    print(f"\nScan over g in [0, 60], f = 0, sigma = 1: the largest gap the judge cos(y)/2 can earn is {gaps[k]:.4f} "
          f"(= e^(-1/2), reached whenever cos g = -1, i.e. g = pi, 3 pi, ...); the true Dudley distance at g = 60 is {dudley_shift(60.0):.4f} (ceiling 2).")
    lhs_max = math.exp(-0.5)
    rhs = math.exp(-0.5) / 2 * (0 ** 2 + 10.0 ** 2)
    print(f"  Eq. (29) as printed, e^(-sigma^2/2)/2 * int (f^2 + g^2) p dx, at f = 0 and the constant g = 10: {rhs:.4f}; "
          f"but the quantity it is meant to bound below, e^(-sigma^2/2)/2 * |int (cos f - cos g) p dx|, is at most {lhs_max:.4f} "
          f"because |cos f - cos g| <= 2, and Dudley <= 2 always.")

    # embedding coordinate the task ignores
    print("\nEmbeddings: P = N(0, I) and Q = P shifted by m along coordinate 1, in R^2.")
    rng = np.random.default_rng(9)
    n, mshift = 4000, 3.0
    Z = rng.normal(size=(n, 2))
    Zq = Z + np.array([mshift, 0.0])
    print(f"  judge h(z) = z_1 (1-Lipschitz): gap {abs(Z[:, 0].mean() - Zq[:, 0].mean()):.4f}; the identity pairing z_i -> z_i + (m,0) costs exactly {mshift:g}, so W1 = {mshift:g}")
    print(f"  panel of functions of z_2 only: any h(z_2) has the same mean under both (same z_2 values): gap "
          f"{abs(np.tanh(Z[:, 1]).mean() - np.tanh(Zq[:, 1]).mean()):.4f} for tanh(z_2), 0 for every h")

    # a bounded panel cannot see a rare far-away mass
    print("\nA rare far-away mass: P = delta_0, Q = (1 - eps) delta_0 + eps delta_R")
    for eps, R in ((0.01, 1000.0), (0.01, 10.0), (0.1, 1000.0)):
        print(f"  eps = {eps:g}, R = {R:g}: TV = {eps:g}, Dudley = eps * 2R/(R+2) = {eps * dudley_points(R):.5f}, W1 = eps R = {eps * R:g}; "
              f"the task f(x) = x has E_Q f - E_P f = {eps * R:g}")


# ------------------------------------------------------------------ 10. summary table and test vectors

def check_table():
    section("10. Summary: delta masses at 0 and D = 2, and N(0,1) vs N(1,1)")
    D, m = 2.0, 1.0
    print(f"{'metric':22s} {'delta_0 vs delta_D=2':>22s} {'N(0,1) vs N(1,1)':>20s}")
    print(f"{'KL':22s} {'inf':>22s} {m * m / 2:>20.4f}")
    print(f"{'TV':22s} {1.0:>22.4f} {tv_shift(m):>20.4f}")
    print(f"{'Kolmogorov':22s} {1.0:>22.4f} {tv_shift(m):>20.4f}")
    print(f"{'Wasserstein-1':22s} {D:>22.4f} {m:>20.4f}")
    print(f"{'Dudley (sum norm)':22s} {dudley_points(D):>22.4f} {dudley_shift(m):>20.4f}")
    print(f"{'Dudley (max norm)':22s} {min(D, 2.0):>22.4f} {'(not computed)':>20s}")
    print(f"{'MMD, sigma = 1':22s} {math.sqrt(2 - 2 * math.exp(-D * D / 2)):>22.4f} {mmd_shift(m):>20.4f}")


def test_vectors():
    section("Test vectors for figures/interactive.html (it recomputes these and reports the largest deviation)")
    v = {}
    for name, m, s in (("gs_m1_s1", 1.0, 1.0), ("gs_m3_s0p5", 3.0, 0.5), ("gs_m0p3_s2", 0.3, 2.0)):
        dud, a = dudley_shift(m, s, return_a=True)
        v[name + "_tv"] = tv_shift(m, s)
        v[name + "_kl"] = kl_gauss(0, s, m, s)
        v[name + "_w1"] = m
        v[name + "_dudley"] = dud
        v[name + "_mmd_sig1"] = mmd_shift(m, s, 1.0)
        v[name + "_mmd_sig0p4"] = mmd_shift(m, s, 0.4)
    v["pts_D1_dudley"] = dudley_points(1.0)
    v["pts_D5_dudley"] = dudley_points(5.0)
    v["pts_D5_mmd_sig1"] = math.sqrt(2 - 2 * math.exp(-25 / 2))
    v["four_pt_mmd2_biased_sig1"] = mmd2_biased(TX, TY, 1.0)
    v["four_pt_mmd2_unbiased_sig1"] = mmd2_unbiased(TX, TY, 1.0)
    v["four_pt_mmd2_biased_sig0p5"] = mmd2_biased(TX, TY, 0.5)
    # ramp gap at fixed parameters, Gaussian pair with different widths
    v["ramp_gap_P0_1_Q1p5_0p6_c0p7_a0p5"] = ramp_gap_gauss(0, 1, 1.5, 0.6, 0.7, 0.5, 0.5)
    # bimodal Q = 0.5 N(-1.5, 0.5^2) + 0.5 N(1.5, 0.5^2) vs P = N(0,1): quadrature on a fine grid
    x = np.linspace(-14, 14, 2000001)
    dx = x[1] - x[0]
    p = pdf(x)
    q = 0.5 * pdf((x + 1.5) / 0.5) / 0.5 + 0.5 * pdf((x - 1.5) / 0.5) / 0.5
    F = np.cumsum(p) * dx
    G = np.cumsum(q) * dx
    v["bimodal_tv"] = float(0.5 * np.sum(np.abs(p - q)) * dx)
    v["bimodal_w1"] = float(np.sum(np.abs(F - G)) * dx)
    mask = p > 1e-300
    v["bimodal_kl_pq"] = float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.maximum(q[mask], 1e-300)))) * dx)
    # the same pair on the DP grid: best ramp and best any-shape judge (a = 0.45), and the shift pair at a = 0.57
    xg, dxg = GX, GDX
    d2 = (pdf(xg) - bimodal_q(xg)) * dxg
    v["bimodal_grid_ramp_a0p45"] = ramp_best_grid(d2, xg, 0.45)[0]
    v["bimodal_grid_dp_a0p45"] = best_judge_dp(d2, dxg, 0.45, 0.55)
    v["shift_m1_grid_dp_a0p57"] = best_judge_dp((pdf(xg) - pdf(xg - 1.0)) * dxg, dxg, 0.57, 0.43)
    v["shift_m1_grid_ramp_a0p57"] = ramp_best_grid((pdf(xg) - pdf(xg - 1.0)) * dxg, xg, 0.57)[0]
    for nm, (Pm, Qm) in PRESETS.items():
        v[f"preset_{nm}_pop_mmd2_sig1"] = mmd2_pop_mix(Pm, Qm, 1.0)
        v[f"preset_{nm}_pop_mmd2_sig0p3"] = mmd2_pop_mix(Pm, Qm, 0.3)
    wv = witness(TX, TY, 1.0, np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))
    for i_, t_ in enumerate((-1, 0, 1, 2, 3)):
        v[f"four_pt_witness_sig1_t{t_}"] = float(wv[i_])
    v["four_pt_witness_sig0p5_t1"] = float(witness(TX, TY, 0.5, np.array([1.0]))[0])
    for k, val in v.items():
        print(f"  {k:36s} {val:.6f}")
    return v


# ------------------------------------------------------------------ figures

def write_figures(data):
    import svgkit
    from svgkit import Axes, legend

    # ---- Figure 1: ramp window
    m = 1.0
    dud, a_star = dudley_shift(m, 1.0, return_a=True)
    L = 1 - a_star
    hw = a_star / L
    x0, x1 = -3.5, 4.5
    xs = np.linspace(x0, x1, 400)
    p, q = pdf(xs), pdf(xs - m)
    diff = Phi(xs) - Phi(xs - m)
    body = []
    top = Axes(60, 30, 560, 130, (x0, x1), (0, 0.5))
    body.append(top.frame([-2, 0, 2, 4], [0, 0.2, 0.4], "x", "density", yfmt="{:g}"))
    body.append(f'<polygon class="bb" points="' + " ".join(f"{top.X(a):.1f},{top.Y(b):.1f}" for a, b in zip(xs, p)) +
                f' {top.X(x1):.1f},{top.Y(0):.1f} {top.X(x0):.1f},{top.Y(0):.1f}"/>')
    body.append(f'<polygon class="ob" points="' + " ".join(f"{top.X(a):.1f},{top.Y(b):.1f}" for a, b in zip(xs, q)) +
                f' {top.X(x1):.1f},{top.Y(0):.1f} {top.X(x0):.1f},{top.Y(0):.1f}"/>')
    body.append(top.path(xs, p, "b"))
    body.append(top.path(xs, q, "o"))
    body.append(top.text(-0.9, 0.30, "P = N(0,1)", "lab", "end"))
    body.append(top.text(2.0, 0.30, "Q = N(1,1)", "lab", "start"))
    # the judge, drawn on the density axis scaled to fit: h from -a..a mapped to 0.05..0.45
    c = m / 2
    hs = -np.clip(L * (xs - c), -a_star, a_star)  # decreasing ramp so P scores higher
    hmap = 0.25 + hs * 0.18 / a_star
    body.append(top.path(xs, hmap, "g"))
    body.append(top.text(-3.4, 0.47, "judge h, drawn scaled to fit (it runs from -a to +a)", "lab"))
    bot = Axes(60, 205, 560, 130, (x0, x1), (0, 0.4))
    body.append(bot.frame([-2, 0, 2, 4], [0, 0.2, 0.4], "x", "F(x) - G(x)"))
    lo, hi = c - hw, c + hw
    wmask = (xs >= lo) & (xs <= hi)
    wx = xs[wmask]
    body.append('<polygon class="shade" points="' + " ".join(f"{bot.X(a):.1f},{bot.Y(b):.1f}" for a, b in zip(xs, diff)) +
                f' {bot.X(x1):.1f},{bot.Y(0):.1f} {bot.X(x0):.1f},{bot.Y(0):.1f}"/>')
    body.append('<polygon class="ob" points="' + " ".join(f"{bot.X(a):.1f},{bot.Y(b):.1f}" for a, b in zip(wx, diff[wmask])) +
                f' {bot.X(wx[-1]):.1f},{bot.Y(0):.1f} {bot.X(wx[0]):.1f},{bot.Y(0):.1f}"/>')
    body.append(bot.path(xs, diff, "k0"))
    body.append(bot.text(-3.3, 0.36, f"whole area under F - G = W1 = {m:.2f}", "lab"))
    body.append(bot.text(-3.3, 0.30, f"orange part x L = {dud:.3f} = Dudley distance", "lab"))
    body.append(bot.text(-3.3, 0.24, f"(window {2 * hw:.2f} wide, L = {L:.2f}, a = {a_star:.2f})", "tiny"))
    area = dud / L
    desc = (f"Two Gaussians N(0,1) and N(1,1) with the best bounded-Lipschitz ramp judge, and the gap F minus G between their CDFs. "
            f"The ramp has slope L = {L:.2f} and sup norm a = {a_star:.2f}, so it slopes over a window of width {2 * hw:.2f} centred at 0.5. "
            f"Its gap equals L times the area of F minus G over the window, {dud:.3f}. The whole area under F minus G is W1 = 1.")
    svgkit.write("ramp-window.svg", svgkit.svg(660, 370, "A ramp judge reads the area between the CDFs", desc, body))

    # ---- Figure 2: distance vs separation (Gaussian shift, point masses)
    body = []
    ms = np.linspace(0.02, 6, 120)
    ax1 = Axes(50, 34, 270, 240, (0, 6), (0, 3))
    body.append('<text class="hd" x="50" y="20">Two Gaussians, N(0,1) vs N(m,1)</text>')
    body.append(ax1.frame([0, 2, 4, 6], [0, 1, 2, 3], "shift m", ""))
    tv = [tv_shift(v) for v in ms]
    dd = [dudley_shift(v) for v in ms]
    kl = [v * v / 2 for v in ms]
    mm = [mmd_shift(v) for v in ms]
    body.append(f'<line class="k" x1="{ax1.x0}" x2="{ax1.x0 + ax1.w}" y1="{ax1.Y(2):.1f}" y2="{ax1.Y(2):.1f}"/>')
    body.append(ax1.path(ms, np.minimum(ms, 3.0), "o"))
    body.append(ax1.path(ms, [min(v, 3.0) for v in kl], "p"))
    body.append(ax1.path(ms, dd, "g"))
    body.append(ax1.path(ms, tv, "b"))
    body.append(ax1.path(ms, mm, "r"))
    Ds = np.logspace(-1.5, 1.7, 120)
    ax2 = Axes(370, 34, 270, 240, (10 ** -1.5, 10 ** 1.7), (0, 3), logx=True)
    body.append('<text class="hd" x="370" y="20">Two point masses, delta_0 vs delta_D</text>')
    body.append(ax2.frame([0.1, 1, 10], [0, 1, 2, 3], "distance D", "", xfmt="{:g}"))
    body.append(f'<line class="k" x1="{ax2.x0}" x2="{ax2.x0 + ax2.w}" y1="{ax2.Y(2):.1f}" y2="{ax2.Y(2):.1f}"/>')
    body.append(ax2.path(Ds, np.minimum(Ds, 3.0), "o"))
    body.append(ax2.path(Ds, [dudley_points(v) for v in Ds], "g"))
    body.append(ax2.path(Ds, [1.0] * len(Ds), "b"))
    body.append(ax2.path(Ds, [math.sqrt(2 - 2 * math.exp(-v * v / 2)) for v in Ds], "r"))
    body.append(f'<text class="tiny" x="{ax2.x0 + 6}" y="{ax2.Y(2.9):.1f}">KL = infinity for every D &gt; 0</text>')
    body.append(f'<text class="tiny" x="{ax1.X(5.9):.1f}" y="{ax1.Y(2) - 4:.1f}" text-anchor="end">2: ceiling for Dudley</text>')
    body.append(legend(60, 322, [("o", "W1 (capped at 3)"), ("p", "KL (capped at 3)"), ("g", "Dudley")], 15))
    body.append(legend(230, 322, [("b", "TV"), ("r", "MMD, Gaussian kernel, bandwidth 1")], 15))
    desc = ("Left: total variation, Wasserstein-1, Dudley, KL and Gaussian-kernel MMD for N(0,1) against N(m,1) as the shift m grows. "
            "TV saturates at 1, Dudley at 2 (very slowly), MMD at about 1.07, W1 grows linearly and KL quadratically. "
            "Right: the same for two point masses a distance D apart on a log axis: TV is 1, Dudley is 2D/(D+2), MMD tends to sqrt 2, "
            "W1 equals D and KL is infinite.")
    svgkit.write("separation.svg", svgkit.svg(680, 350, "Distance versus separation", desc, body))

    # ---- Figure 3: Dudley by two routes (left) and beyond ramps (right)
    body = []
    (aval, ramp, ot) = data["dudley_samples"]["quantile grids of N(0,1), N(1,1), n = 60"]
    (aval2, ramp2, ot2) = data["dudley_samples"]["random Gaussian samples, n = 60, seed 3"]
    ax = Axes(50, 44, 280, 200, (0, 1), (0, 0.4))
    body.append('<text class="hd" x="50" y="16">Dudley on n = 60 samples, two ways</text>')
    body.append(ax.frame([0, 0.25, 0.5, 0.75, 1], [0, 0.1, 0.2, 0.3, 0.4], "sup-norm budget a (Lipschitz budget L = 1 - a)", "best gap at this a"))
    body.append(ax.path(aval2, ot2, "o"))
    body.append(ax.path(aval2, ramp2, "b"))
    body.append(ax.path(aval, ot, "g dash"))
    body.append(ax.path(aval, ramp, "p dash"))
    kr, ko = int(np.argmax(ramp2)), int(np.argmax(ot2))
    body.append(ax.text(0.62, 0.385, f"exact maximum {ot2[ko]:.4f}", "lab"))
    body.append(ax.text(0.62, 0.355, f"best ramp {ramp2[kr]:.4f}", "lab"))
    body.append(legend(52, 288, [("o", "exact (OT route), random samples"), ("b", "best ramp, random samples")], 15))
    body.append(legend(52, 322, [("g dash", "exact, quantile grids"), ("p dash", "best ramp, quantile grids")], 15))

    xg, dxg = GX, GDX
    d2 = (pdf(xg) - bimodal_q(xg)) * dxg
    a_b = 0.45
    val_dp, h_dp = best_judge_dp(d2, dxg, a_b, 1 - a_b, path=True)
    r_val, r_c = ramp_best_grid(d2, xg, a_b)
    h_r = np.clip((1 - a_b) * (xg - r_c), -a_b, a_b)
    if h_r @ d2 < 0:
        h_r = -h_r
    bx = Axes(400, 44, 260, 200, (-4, 4), (0, 0.6))
    body.append('<text class="hd" x="400" y="16">P = N(0,1) against a two-humped Q</text>')
    body.append(bx.frame([-4, -2, 0, 2, 4], [0, 0.2, 0.4, 0.6], "x", "density"))
    sel = (xg >= -4) & (xg <= 4)
    for arr, cls in ((pdf(xg), "bb"), (bimodal_q(xg), "ob")):
        pts = " ".join(f"{bx.X(a_):.1f},{bx.Y(b_):.1f}" for a_, b_ in zip(xg[sel][::4], arr[sel][::4]))
        body.append(f'<polygon class="{cls}" points="{pts} {bx.X(4):.1f},{bx.Y(0):.1f} {bx.X(-4):.1f},{bx.Y(0):.1f}"/>')
    body.append(bx.path(xg[sel][::4], pdf(xg)[sel][::4], "b"))
    body.append(bx.path(xg[sel][::4], bimodal_q(xg)[sel][::4], "o"))
    sc = lambda h_: 0.3 + h_ / a_b * 0.22
    body.append(bx.path(xg[sel][::4], sc(h_dp)[sel][::4], "p"))
    body.append(bx.path(xg[sel][::4], sc(h_r)[sel][::4], "g"))
    body.append(legend(402, 288, [("g", f"best ramp, a = 0.45: gap {r_val:.3f}"), ("p", f"best judge of any shape: gap {val_dp:.3f}")], 15))
    body.append(legend(402, 322, [("b", "P"), ("o", "Q")], 15))
    desc = ("Left: for each sup-norm budget a, the best gap over bounded-Lipschitz judges on n = 60 samples of N(0,1) and N(1,1), computed two independent ways: "
            "a search over ramp judges, and the exact optimal-transport cost with truncated ground cost min(|x - y|, 2a/(1-a)) times 1 - a, solved by the Hungarian algorithm. "
            f"On the random samples the exact maximum is {ot2[ko]:.4f} and the best ramp reaches {ramp2[kr]:.4f}; on smooth quantile grids the two curves nearly coincide. "
            f"Right: P = N(0,1) against a two-humped Q with the same mean. At a = 0.45 the best ramp earns {r_val:.3f}, the best judge of any shape earns {val_dp:.3f}: it goes up at P's centre and down at each hump of Q.")
    svgkit.write("beyond-ramps.svg", svgkit.svg(680, 350, "Dudley by two routes, and what a ramp cannot see", desc, body))

    # ---- Figure 4: MMD bandwidth
    body = []
    sigs = np.logspace(-1.3, 2, 100)
    curves = {k: np.array([math.sqrt(max(mmd2_pop_mix(P_, Q_, s_), 0.0)) for s_ in sigs]) for k, (P_, Q_) in PRESETS.items()}
    ax = Axes(56, 30, 560, 210, (10 ** -1.3, 10 ** 2), (0, 0.5), logx=True)
    body.append(ax.frame([0.1, 1, 10, 100], [0, 0.1, 0.2, 0.3, 0.4, 0.5], "kernel bandwidth sigma", "MMD"))
    body.append(ax.path(sigs, curves["means"], "b"))
    body.append(ax.path(sigs, curves["width"], "o"))
    body.append(ax.path(sigs, curves["humps"], "p"))
    keep = [(s_, 1.5 / s_) for s_ in sigs if 1.5 / s_ <= 0.5]
    body.append(ax.path([k[0] for k in keep], [k[1] for k in keep], "k"))
    body.append(legend(72, 46, [("b", "N(0,1) vs N(1.5,1): means differ"), ("o", "N(0,1) vs N(0,2^2): same mean, wider"),
                                ("p", "same mean and variance, Q two-humped"), ("k", "|difference of means| / sigma (means pair)")], 15))
    desc = ("Population Gaussian-kernel MMD as a function of the bandwidth sigma for three pairs of one-dimensional distributions: "
            "one differing in mean, one in width only, one in shape only (a Gaussian against a two-humped mixture with the same mean and variance). "
            "For small sigma every curve falls towards 0; for large sigma the mean-differing pair follows the difference of means divided by sigma "
            "while the other two fall much faster, so a huge bandwidth cannot see a difference in width or shape.")
    svgkit.write("mmd-bandwidth.svg", svgkit.svg(660, 285, "What the MMD bandwidth lets the judges see", desc, body))
    print("\nfigures: wrote", ", ".join(sorted(p_.name for p_ in svgkit.OUT.glob("*.svg"))))


if __name__ == "__main__":
    check_tv()
    check_points()
    check_ramp_identity()
    check_gauss_shift()
    check_bumps()
    check_w1_samples()
    dud_samples = check_dudley_samples()
    check_all_shapes()
    check_mmd()
    check_presets()
    check_panels()
    check_table()
    test_vectors()
    if "--figures" in sys.argv:
        write_figures({"dudley_samples": dud_samples})
