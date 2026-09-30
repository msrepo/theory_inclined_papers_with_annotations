#!/usr/bin/env python3
"""Sections 2.2 and 3.4 of Gao & Chaudhari: how the task is moved from source to target.

Checked here, on 1-D point clouds where optimal transport can be done exactly (sort the points):

  1. McCann's interpolation (7): with the exact optimal coupling, W2(p_s, p_tau) = tau W2(p_s, p_t), a
     constant-speed geodesic. The mixture (9), (1-tau) p_s + tau p_t, is not: W2^2 is convex along mixtures, so
     its distance from p_s is at most sqrt(tau) W2(p_s, p_t), and for the clouds used here it lies between tau
     and sqrt(tau) times W2; it leaves the source much faster than the geodesic does.
  2. Entropic regularisation (6). Sinkhorn's coupling is not a map, so the "interpolant" (7) built from it
     is a blurred, contracted cloud: for large epsilon it is the mixup of independent pairs. Its distance from
     the source is no longer tau W2 and its spread shrinks.
  3. Mixup with lambda ~ Beta(tau, 1 - tau) (Section 3.4): the mean is tau, but the law is U-shaped, so most
     interpolated images are close to one of the two endpoints and only some are really blended.

Standard library and numpy only.

Run:  python3 interpolation.py             (prints every number quoted in the notes)
      python3 interpolation.py --figures   (also rewrites ../figures/interpolation.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np


def norm_ppf(u):
    """Inverse standard normal CDF (Acklam's rational approximation, refined by one Newton step)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    out = np.empty_like(u, dtype=float)
    lo, hi = 0.02425, 1 - 0.02425
    m = (u >= lo) & (u <= hi)
    q = u[m] - 0.5
    r = q * q
    out[m] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    for mask, sign in ((u < lo, 1), (u > hi, -1)):
        q = np.sqrt(-2 * np.log(u[mask] if sign == 1 else 1 - u[mask]))
        v = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        out[mask] = v if sign == 1 else -v
    for _ in range(2):  # Newton refinement
        cdf = 0.5 * (1 + np.array([math.erf(t / math.sqrt(2)) for t in out]))
        out -= (cdf - u) / (np.exp(-0.5 * out ** 2) / math.sqrt(2 * math.pi))
    return out


def w2_1d(xa, wa, xb, wb, m=400000):
    """Wasserstein-2 distance between two weighted atomic measures on the line, from their quantile functions."""
    def quant(x, w):
        o = np.argsort(x)
        c = np.cumsum(w[o])
        c /= c[-1]
        u = (np.arange(m) + 0.5) / m
        return x[o][np.minimum(np.searchsorted(c, u), len(x) - 1)]

    return math.sqrt(float(np.mean((quant(xa, wa) - quant(xb, wb)) ** 2)))


def logsumexp(a, axis):
    mx = a.max(axis=axis, keepdims=True)
    return (mx + np.log(np.exp(a - mx).sum(axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn_plan(x, y, eps, iters=3000):
    """Entropic OT plan of eq. (6) between two uniform clouds, log-domain Sinkhorn."""
    n = len(x)
    C = (x[:, None] - y[None, :]) ** 2
    la = lb = -math.log(n)
    f = np.zeros(n)
    g = np.zeros(n)
    for _ in range(iters):
        f = -eps * logsumexp((g[None, :] - C) / eps + lb, axis=1)
        g = -eps * logsumexp((f[:, None] - C) / eps + la, axis=0)
    return np.exp((f[:, None] + g[None, :] - C) / eps + la + lb)


N = 200
U = (np.arange(N) + 0.5) / N
XS = norm_ppf(U)  # a Gaussian cloud, N(0, 1), as N quantiles
GAP = 6.0
XT = XS + GAP  # the target: the same cloud moved by 6


def check_geodesic():
    print("1. McCann interpolation against the mixture (source N(0,1), target N(6,1), 200 atoms each)")
    w = np.full(N, 1.0 / N)
    W = w2_1d(XS, w, XT, w)
    print(f"   W2(p_s, p_t) = {W:.4f}   (exact: {GAP})")
    print("   tau | displacement: W2(p_s,p_tau)/W2(p_s,p_t) | mixture: same ratio | sqrt(tau)")
    rows = []
    for t in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
        xd = (1 - t) * XS + t * XT  # exact OT pairs sorted with sorted
        rd = w2_1d(XS, w, xd, w) / W
        xm = np.concatenate([XS, XT])
        wm = np.concatenate([(1 - t) * w, t * w])
        wsrc = np.concatenate([w, np.zeros(N)])
        rm = w2_1d(np.concatenate([XS, XT]), wsrc, xm, wm) / W
        rows.append((t, rd, rm))
        print(f"   {t:4} |        {rd:.4f}                          |      {rm:.4f}         | {math.sqrt(t):.4f}")
    print()
    return rows


def check_sinkhorn():
    print("2. Entropic OT (6): the interpolant (7) built from Sinkhorn's coupling")
    w = np.full(N, 1.0 / N)
    W = w2_1d(XS, w, XT, w)
    print("   epsilon | mass off the diagonal band | sd of interpolant at tau = 1/2 (exact 1) | W2(p_s,p_1/2)/(W2/2)")
    rows = []
    for eps in (0.05, 0.5, 2.0, 8.0, 30.0):
        G = sinkhorn_plan(XS, XT, eps)
        # interpolated atoms with the plan's masses
        t = 0.5
        pts = ((1 - t) * XS[:, None] + t * XT[None, :]).ravel()
        wts = G.ravel()
        mean = float(np.sum(wts * pts))
        sd = math.sqrt(float(np.sum(wts * (pts - mean) ** 2)))
        r = w2_1d(XS, w, pts, wts) / (t * W)
        off = float(G[np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) > 0.05 * N].sum())
        rows.append((eps, sd, r))
        print(f"   {eps:6}  |          {off:6.3f}             |          {sd:6.3f}                        |   {r:6.3f}")
    print("   independent coupling (epsilon -> infinity): sd = sqrt(1/4 + 1/4) = 0.7071, i.e. plain mixup of unrelated pairs")
    print()
    return rows


def check_beta():
    print("3. Mixup with lambda ~ Beta(tau, 1 - tau) (Section 3.4)")
    rng = np.random.default_rng(0)
    print("   tau | mean lambda | P(lambda<0.1 or >0.9) | P(0.25<lambda<0.75) | (fixed lambda = tau would give 0 and 1[.25<tau<.75])")
    for t in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
        lam = rng.beta(t, 1 - t, size=400000)
        near = float(np.mean((lam < 0.1) | (lam > 0.9)))
        mid = float(np.mean((lam > 0.25) & (lam < 0.75)))
        print(f"   {t:4} |   {lam.mean():.3f}     |        {near:.3f}          |       {mid:.3f}        |")
    print("   at tau = 1/2 the law is the arcsine law: P(lambda < 0.1) = (2/pi) arcsin(sqrt(0.1)) = "
          f"{2 / math.pi * math.asin(math.sqrt(0.1)):.4f}")
    print()


def figures(geo_rows, sk_rows):
    from svgkit import Axes, legend, svg, write

    W, H = 640, 320
    ax = Axes(60, 30, 300, 240, (0, 1), (0, 1))
    body = [ax.frame([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], "tau", "W2(p_s, p_tau) / W2(p_s, p_t)")]
    ts = np.linspace(0.0, 1.0, 101)
    body.append(ax.path(ts, ts, "b"))
    body.append(ax.path(ts, np.sqrt(ts), "o"))
    for t, rd, rm in geo_rows:
        body.append(ax.dot(t, rd, "bf", 2.5))
        body.append(ax.dot(t, rm, "of", 2.5))
    body.append(legend(400, 60, [("b", "displacement (exact OT)"), ("o", "sqrt(tau): upper bound for the mixture")], 18))
    body.append('<text class="tiny" x="400" y="110">dots: computed on 200-atom clouds,</text>')
    body.append('<text class="tiny" x="400" y="124">blue displacement, orange mixture</text>')
    body.append('<text class="hd" x="400" y="170">Blur from entropic OT</text>')
    body.append('<text class="tiny" x="400" y="188">sd of the tau = 1/2 interpolant</text>')
    for i, (eps, sd, r) in enumerate(sk_rows):
        body.append(f'<text class="lab" x="400" y="{208 + 16 * i}">eps = {eps:g}: sd {sd:.2f}</text>')
    body.append('<text class="tiny" x="400" y="{}">(exact OT: 1.00)</text>'.format(208 + 16 * len(sk_rows)))
    write("interpolation.svg", svg(W, H, "Distance from the source along two interpolations",
                                   "Displacement interpolation moves at constant W2 speed; the mixture moves like the square root of tau.", body))


if __name__ == "__main__":
    g = check_geodesic()
    s = check_sinkhorn()
    check_beta()
    if "--figures" in sys.argv:
        figures(g, s)
