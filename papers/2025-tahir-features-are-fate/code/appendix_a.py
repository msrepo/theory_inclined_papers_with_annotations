#!/usr/bin/env python3
"""Appendix A of Tahir, Ganguli & Rotskoff (ICML 2025): "dataset similarity is not predictive of transfer".

The appendix proves that for any source function f in a feature space Phi there is a target g in Phi whose
joint law p_g is arbitrarily far from p_f in the Dudley metric and in KL. Together with Section 3 (targets
in the same feature space transfer well) this is the paper's argument against distribution-level distances.
Checked here, in the order of the notes:

  1. The KL half: KL(p_f || p_g) = ||f - g||^2 / (2 sigma^2), by quadrature; and the step "choose alpha >
     sigma sqrt(delta) / ||f||" gives only KL > delta/2, which is not always enough.
  2. The Dudley half, line by line: the test function cos(y)/2 has BL norm 1 (fine), but it can never
     produce more than exp(-sigma^2/2) < 1, and inequality (29) fails for every g = f and for all large g.
  3. Dudley is bounded by 2. A judge that does work: h(y) = a(1 - 2 min(|y|/K, 1)), a = K/(K+2), for
     g = c f. The bound climbs towards 2 as c grows, but never beyond, so "any delta" must mean "any
     delta < 2" (and the proof would need this judge, not cos).
  4. Inside the Section-3 family (||beta|| = 1) the KL is (1 - cos theta)/sigma^2, and T_lt depends on
     theta only through sin^2 theta. Sweeping theta over [0, pi] reproduces the U shape of Figure 4(a); over
     [0, pi/2], the range of the phase diagrams, KL orders T_lt perfectly.
  5. Along g = c f (a norm not equal to 1) transferability *grows* with the distance.
  6. Figure 4(b): the finite-sample Wasserstein-1 between samples of N(0, I_500) x noise is ~527 at every
     theta, including theta = 0 where the two laws are identical; the population W1 is at most 1.6.

Standard library and numpy only.

Run:  python3 appendix_a.py             (prints every number quoted in the notes)
      python3 appendix_a.py --figures   (also rewrites the SVGs in ../figures/)
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import theory as T

PI = math.pi
XS, WS = np.polynomial.hermite_e.hermegauss(80)
WS = WS / WS.sum()                       # E over N(0,1) as sum(WS * h(XS))


def header(s):
    print("\n" + s + "\n" + "-" * len(s))


# ------------------------------------------------------------------ 1. KL
def kl_quadrature(a, c, sig):
    """KL(p_f || p_g) for f(x) = a x, g(x) = c x, x ~ N(0,1), by Gauss-Hermite in x and in y."""
    tot = 0.0
    for x, w in zip(XS, WS):
        y = a * x + sig * XS                                        # y ~ N(a x, sig^2)
        lp = -(y - a * x) ** 2 / (2 * sig ** 2)
        lq = -(y - c * x) ** 2 / (2 * sig ** 2)
        tot += w * np.sum(WS * (lp - lq))
    return tot


def part1():
    header("1. The KL half of Theorem A.2")
    sig = 0.2
    for a, c in [(1, -1), (1, 3), (1, 10)]:
        print("   f = %gx, g = %gx, sigma = %.1f: quadrature %.4f, ||f-g||^2/(2 sigma^2) = %.4f"
              % (a, c, sig, kl_quadrature(a, c, sig), (a - c) ** 2 / (2 * sig ** 2)))
    print("   The paper picks g = -alpha f with alpha > sigma sqrt(delta)/||f||. Then KL = (1+alpha)^2 ||f||^2/(2 sigma^2) > alpha^2 ||f||^2/(2 sigma^2) > delta/2.")
    print("   That is delta/2, not delta. It is enough when ||f|| >= 0.42 sigma sqrt(delta) and can fail otherwise:")
    f_norm, delta = 0.01, 100.0
    alpha = sig * math.sqrt(delta) / f_norm * 1.0000001
    print("   ||f|| = %.2f, sigma = %.1f, delta = %g: alpha = %.1f, KL = %.1f  (< delta)" % (f_norm, sig, delta, alpha, (1 + alpha) ** 2 * f_norm ** 2 / (2 * sig ** 2)))
    print("   Fix: alpha > sigma sqrt(2 delta)/||f||. A harmless slip; the KL half is otherwise exact.")


# ------------------------------------------------------------------ 2. the Dudley proof
def part2():
    header("2. The Dudley half: lines (26)-(31)")
    sig = 0.2
    ebound = math.exp(-sig ** 2 / 2)
    print("   Test function h(x,y) = cos(y)/2: ||h||_L = 1/2, ||h||_inf = 1/2, so ||h||_BL = 1. Allowed.")
    print("   E over p_f of cos(y) = cos(f(x)) exp(-sigma^2/2), so line (28) = (exp(-sigma^2/2)/2) |E cos f(x) - E cos g(x)|.")
    print("   That is at most exp(-sigma^2/2) = %.4f whatever g is (each cosine is in [-1,1])." % ebound)
    print("   So (29), which needs it to be >= (exp(-sigma^2/2)/2) E[f^2 + g^2] and hence arbitrarily large, cannot hold for large g.")
    print("   Numbers, f(x) = a x and g(x) = c x with x ~ N(0,1), where E cos(a x) = exp(-a^2/2):")
    for a, c in [(1, 1), (1, 3), (1, 10)]:
        lhs = ebound / 2 * abs(math.exp(-a * a / 2) - math.exp(-c * c / 2))
        rhs = ebound / 2 * (a * a + c * c)
        print("     f = %gx, g = %gx: line (28) = %.4f;  line (29)'s right side = %.4f  ->  (29) %s"
              % (a, c, lhs, rhs, "holds" if lhs >= rhs else "FAILS"))
    print("   The identity 'cos x + x^2 >= cos z - z^2' is true, but it rearranges to cos x - cos z >= -(x^2 + z^2):")
    print("   a lower bound on the difference by a negative number, which says nothing about its absolute value.")
    best = max(ebound / 2 * abs(math.exp(-0.5) - math.exp(-c * c / 2)) for c in np.linspace(0, 50, 5001))
    print("   For f = x the cos judge gives at most %.4f over all g = c x (attained as c -> infinity, when E cos(cx) -> 0): the ceiling exp(-sigma^2/2) is not approached." % best)


# ------------------------------------------------------------------ 3. a judge that works
def e_ramp(var, K):
    """E[1 - 2 min(|y|/K, 1)] for y ~ N(0, var)."""
    s = math.sqrt(var)
    phi = math.exp(-(K / s) ** 2 / 2) / math.sqrt(2 * PI)
    Q = 0.5 * (1 - math.erf(K / s / math.sqrt(2)))
    emin = s * math.sqrt(2 / PI) - 2 * (s * phi - K * Q)             # E min(|y|, K)
    return 1 - 2 * emin / K


def dudley_lower_bound(c, sig, fnorm=1.0):
    """Lower bound on the Dudley distance between p_f and p_{cf}, f(x) = fnorm x with x ~ N(0,1).

    Judge h(y) = a (1 - 2 min(|y|/K, 1)) depends on y only, has sup norm a and Lipschitz constant 2a/K,
    so ||h||_BL = a (1 + 2/K) = 1 for a = K/(K+2). The y-marginals are N(0, fnorm^2 + sig^2) and N(0, c^2 fnorm^2 + sig^2)."""
    best = 0.0
    for K in np.exp(np.linspace(math.log(0.3), math.log(5000), 500)):
        a = K / (K + 2)
        v = a * (e_ramp(fnorm ** 2 + sig ** 2, K) - e_ramp(c * c * fnorm ** 2 + sig ** 2, K))
        best = max(best, v)
    return best


def part3():
    header("3. Dudley is bounded by 2; a judge that works")
    sig = 0.2
    print("   Every h in the BL unit ball has |h| <= 1, so |E_P h - E_Q h| <= 2 for any P, Q. 'For every delta > 0' is false for delta >= 2.")
    print("   Lower bound from the ramp judge, g = c f with f(x) = x, sigma = 0.2 (KL = (c-1)^2/(2 sigma^2)):")
    for c in [1, 2, 3, 5, 10, 30, 100, 1000, 100000]:
        print("     c = %-7g Dudley >= %.4f    KL = %.4g" % (c, dudley_lower_bound(c, sig), (c - 1) ** 2 / (2 * sig ** 2)))
    print("   For this judge the gap to 2 shrinks like c^(-1/2) (0.39 at c = 100, 0.13 at c = 1000, 0.013 at c = 1e5); it never reaches 2, while KL grows without bound.")


# ------------------------------------------------------------------ 4. the Section-3 family
def part4():
    header("4. Inside the Section-3 family: KL(theta) and T_lt(theta), gamma = 0.5, sigma = 0.2 (Figure 4a)")
    g, sig = 0.5, 0.2
    s2 = sig ** 2
    ths = np.linspace(0, PI, 41)
    kl = (1 - np.cos(ths)) / s2
    tt = T.t_lt(g, ths, s2)
    print("   KL(theta) = ||beta_s - beta_t||^2/(2 sigma^2) = (1 - cos theta)/sigma^2, from 0 to %.1f at theta = pi (axis of Fig. 4a: 0..50)." % kl[-1])
    print("   T_lt: theta=0 -> %+.3f;  theta=pi/2 -> %+.3f;  theta=pi -> %+.3f   (Fig. 4a: about +0.54, -0.46, +0.54)" % (tt[0], tt[20], tt[-1]))
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return np.corrcoef(rx, ry)[0, 1]
    print("   Over theta in [0, pi]:   Pearson(KL, T) = %+.3f,  Spearman = %+.3f" % (np.corrcoef(kl, tt)[0, 1], spearman(kl, tt)))
    m = ths <= PI / 2 + 1e-12
    print("   Over theta in [0, pi/2]: Pearson(KL, T) = %+.3f,  Spearman = %+.3f   (the range of the phase diagrams)"
          % (np.corrcoef(kl[m], tt[m])[0, 1], spearman(kl[m], tt[m])))
    print("   T_lt depends on theta only through sin^2(theta), which is symmetric about pi/2, while KL is monotone: hence the U.")
    print("   KL(pi - x) = 2/sigma^2 - KL(x): the anti-aligned target (theta = pi, beta_t = -beta_s) is the farthest and transfers exactly as well as theta = 0.")
    rng = np.random.default_rng(0)
    d, reps = 200, 300
    n = int(g * d)
    print("   Monte Carlo at d = %d, %d draws (T_lt = R_sc - R_lt):" % (d, reps))
    for th in [0.0, PI / 2, 3 * PI / 4, PI]:
        sc, lt = [], []
        for _ in range(reps):
            bs = rng.standard_normal(d); bs /= np.linalg.norm(bs)
            nu = rng.standard_normal(d); nu -= (nu @ bs) * bs; nu /= np.linalg.norm(nu)
            bt = math.cos(th) * bs + math.sin(th) * nu
            X = rng.standard_normal((n, d)); y = X @ bt + sig * rng.standard_normal(n)
            sc.append(np.sum((np.linalg.pinv(X) @ y - bt) ** 2))
            z = X @ bs
            lt.append(np.sum((((z @ y) / (z @ z)) * bs - bt) ** 2))
        print("     theta = %.3f: KL = %5.1f   T_lt MC %+.3f   eq.(11)-exact %+.3f" % (th, (1 - math.cos(th)) / s2, np.mean(sc) - np.mean(lt), T.t_lt_exact(n, d, th, s2)))
    print("   Other axes: at fixed theta the same pair has one KL and many T. theta = pi/4, sigma = 0.2: KL = %.2f, and T_lt changes sign at gamma = 0.549, 0.911, 1.080 (transfer_theory.py 3d)."
          % ((1 - math.cos(PI / 4)) / s2))


# ------------------------------------------------------------------ 5. norm c targets
def part5():
    header("5. Targets g = c f with c > 1 (outside the unit-norm family)")
    g, sig, d = 0.5, 0.2, 200
    s2 = sig ** 2
    n = int(g * d)
    rng = np.random.default_rng(1)
    print("   R_sc(c) = c^2 (1-gamma) + gamma sigma^2/(1-gamma); linear transfer on the pretrained feature has R_lt = sigma^2/(n-2) for every c.")
    print("   c   KL(p_f||p_g)   T_lt MC     T_lt formula")
    for c in [1, 2, 3, 10]:
        sc, lt = [], []
        for _ in range(200):
            bs = rng.standard_normal(d); bs /= np.linalg.norm(bs)
            bt = c * bs
            X = rng.standard_normal((n, d)); y = X @ bt + sig * rng.standard_normal(n)
            sc.append(np.sum((np.linalg.pinv(X) @ y - bt) ** 2))
            z = X @ bs
            lt.append(np.sum((((z @ y) / (z @ z)) * bs - bt) ** 2))
        form = c * c * (1 - g) + g * s2 / (1 - g) - s2 / (n - 2)
        print("   %-3g %-14.4g %-11.3f %.3f" % (c, (c - 1) ** 2 / (2 * s2), np.mean(sc) - np.mean(lt), form))
    print("   Along this family T grows with the distance: the far target is the *easier* one to beat scratch on, since scratch has more to learn.")


# ------------------------------------------------------------------ 6. W1 in Figure 4(b)
def hungarian_mean_cost(C):
    """Mean cost of the optimal perfect matching (Jonker-Volgenant / e-maxx shortest augmenting path), O(n^3), numpy.
    For two empirical measures with n atoms of mass 1/n this is the Wasserstein-1 cost for the ground cost C."""
    n = C.shape[0]
    INF = 1e18
    u = np.zeros(n + 1); v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int); way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = C[i0 - 1, :] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], INF)
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
    return sum(C[p[j] - 1, j - 1] for j in range(1, n + 1)) / n


def sample_joint(rng, n, beta, sig):
    X = rng.standard_normal((n, len(beta)))
    y = X @ beta + sig * rng.standard_normal(n)
    return np.hstack([X, y[:, None]])


def part6():
    header("6. Figure 4(b): what a finite-sample W1 measures at d = 500")
    rng = np.random.default_rng(3)
    sig = 0.2
    def unit(d):
        v = rng.standard_normal(d); return v / np.linalg.norm(v)
    d, n = 500, 250
    bs = unit(d); nu = unit(d); nu -= (nu @ bs) * bs; nu /= np.linalg.norm(nu)
    print("   gamma = 0.5 (n = %d), d = %d, sigma = %.1f, exact assignment solver, samples drawn independently for p_s and p_t:" % (n, d, sig))
    for th in [0.0, PI / 2, PI]:
        bt = math.cos(th) * bs + math.sin(th) * nu
        A = sample_joint(rng, n, bs, sig); B = sample_joint(rng, n, bt, sig)
        C2 = np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0))
        C1 = np.abs(A[:, None, :] - B[None, :, :]).sum(2)
        print("     theta = %.3f: W1 with l1 ground metric = %7.2f,  with l2 = %6.2f    (population W1 <= sqrt(2/pi) ||beta_s - beta_t|| = %.2f)"
              % (th, hungarian_mean_cost(C1), hungarian_mean_cost(C2), math.sqrt(2 / PI) * 2 * math.sin(th / 2)))
    print("   The l1 value is ~527 at theta = 0, where the two laws are the same; Figure 4(b)'s axis runs 526-529. The plotted W1 is sampling noise of the shared x-marginal.")
    print("   Population bound: couple the same x and the same noise, then |y_s - y_t| = |(beta_s - beta_t).x| ~ ||beta_s - beta_t|| |N(0,1)|, mean sqrt(2/pi) ||beta_s - beta_t|| <= 1.6.")

    print("   In d = 2 the estimator is informative. (x1, x2, y) in R^3, n = 500, l2 ground metric, sigma = 0.2:")
    d, n = 2, 500
    bs = np.array([1.0, 0.0])
    floor = None
    for th in [0.0, PI / 4, PI / 2, 3 * PI / 4, PI]:
        bt = np.array([math.cos(th), math.sin(th)])
        vals = []
        for _ in range(3):
            A = sample_joint(rng, n, bs, sig); B = sample_joint(rng, n, bt, sig)
            C = np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0))
            vals.append(hungarian_mean_cost(C))
        w = float(np.mean(vals))
        floor = w if floor is None else floor
        print("     theta = %.3f: W1 = %.3f  (minus theta = 0 floor: %.3f;  bound sqrt(2/pi) 2 sin(theta/2) = %.3f;  KL = %5.1f)"
              % (th, w, w - floor, math.sqrt(2 / PI) * 2 * math.sin(th / 2), (1 - math.cos(th)) / sig ** 2))
    print("   There W1 does rise monotonically with theta, like KL. So Figure 4(a)'s U shape is the whole message about the *population* distances; 4(b) adds nothing.")


# ------------------------------------------------------------------ figures
def figures():
    from svgkit import Axes, legend, svg, write
    g, sig = 0.5, 0.2
    s2 = sig ** 2

    # ---- U shape and fixed pair
    A = Axes(60, 40, 300, 240, (0, 50), (-0.6, 0.7))
    B = Axes(440, 40, 290, 240, (0, 2), (-1, 1))
    body = ['<text class="hd" x="60" y="20">Figure 4(a) reproduced: T against KL, θ swept over [0, π]</text>']
    body.append(A.frame([0, 10, 20, 30, 40, 50], [-0.5, 0, 0.5], "KL(p_s ‖ p_t) = (1 − cos θ)/σ²", "T_lt", yfmt="{:+.1f}"))
    body.append(f'<line class="k" x1="{A.X(0)}" x2="{A.X(50)}" y1="{A.Y(0):.1f}" y2="{A.Y(0):.1f}"/>')
    ths = np.linspace(0, PI / 2, 60)
    ths2 = np.linspace(PI / 2, PI, 60)
    for arr, cls in [(ths, "b"), (ths2, "o")]:
        body.append(A.path((1 - np.cos(arr)) / s2, T.t_lt(g, arr, s2), cls))
    body.append(A.text(2, 0.5, "θ = 0", "tiny", dx=5))
    body.append(A.text(25, -0.46, "θ = π/2", "tiny", anchor="middle", dy=14))
    body.append(A.text(50, 0.54, "θ = π", "tiny", anchor="end", dx=-4, dy=-6))
    body.append(legend(A.x0, A.y0 + A.h + 50, [("b", "θ ≤ π/2 (the phase-diagram range)"), ("o", "θ > π/2 (target turned against the source)")]))
    body.append(B.frame([0, 0.5, 1, 1.5, 2], [-1, -0.5, 0, 0.5, 1], "γ = n/d", "T_lt", yfmt="{:+.1f}"))
    body.append(f'<line class="k" x1="{B.X(0)}" x2="{B.X(2)}" y1="{B.Y(0):.1f}" y2="{B.Y(0):.1f}"/>')
    th = PI / 4
    for seg in (np.linspace(0.003, 0.999, 600), np.linspace(1.001, 2, 600)):
        v = T.t_lt(seg, th, s2)
        m = np.abs(v) <= 1
        body.append(B.path(seg[m], v[m], "o"))
    body.append('<text class="sub" x="440" y="326">One pair (θ = π/4, σ = 0.2): KL = 7.3 for every γ,</text>')
    body.append('<text class="sub" x="440" y="342">yet T changes sign three times as n grows.</text>')
    write("kl-vs-transfer.svg", svg(760, 360, "KL against transferability",
          "Left: T_lt against KL when theta sweeps 0 to pi gives a U. Right: for a single pair the sign of T depends on n.", body))

    # ---- Dudley saturation
    C = Axes(60, 40, 400, 240, (1, 1e5), (0, 2.2), logx=True)
    body = ['<text class="hd" x="60" y="20">“Arbitrarily far” in the Dudley metric: g = c f, σ = 0.2</text>']
    cs = np.exp(np.linspace(0, math.log(1e5), 60))
    body.append(C.frame([1, 10, 100, 1000, 10000, 100000], [0, 0.5, 1, 1.5, 2], "c   (KL = (c−1)²/(2σ²): 12.5 at c = 2, 1012 at c = 10, 1.05·10⁴ at c = 30, 1.2·10⁵ at c = 100)", "Dudley lower bound", xfmt="{:g}"))
    body.append(f'<line class="r dash" x1="{C.X(1):.1f}" x2="{C.X(1e5):.1f}" y1="{C.Y(2):.1f}" y2="{C.Y(2):.1f}"/>')
    body.append(C.text(1, 2, "ceiling: 2", "tiny", dx=6, dy=-6))
    body.append(C.path(cs, [dudley_lower_bound(c, sig) for c in cs], "b"))
    body.append('<text class="sub" x="490" y="80">The judge is</text>')
    body.append('<text class="sub" x="490" y="98">h(y) = a (1 − 2 min(|y|/K, 1)),</text>')
    body.append('<text class="sub" x="490" y="116">a = K/(K+2), so ‖h‖_BL = 1.</text>')
    body.append('<text class="sub" x="490" y="146">The distance saturates at 2,</text>')
    body.append('<text class="sub" x="490" y="164">while KL keeps growing like c².</text>')
    write("dudley-saturation.svg", svg(820, 320, "Dudley distance saturates",
          "Lower bound on the Dudley distance between p_f and p_cf against c, with the ceiling at 2.", body))


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        figures()
        sys.exit(0)
    t0 = time.time()
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    print("\n(%.0f s)" % (time.time() - t0))
    if "--figures" in sys.argv:
        figures()
