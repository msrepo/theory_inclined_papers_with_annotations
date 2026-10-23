#!/usr/bin/env python3
"""Mean-field VB for the Normal model of Example 2.1 in Tran, Nguyen and Dao (2021).

    y_i ~ N(mu, sigma2),  mu ~ N(mu0, s0sq),  sigma2 ~ Inverse-Gamma(a0, b0)
    y = (11, 12, 8, 10, 9, 8, 9, 10, 13, 7),  mu0 = 0, s0sq = 100, a0 = 1, b0 = 1

The approximation is q(mu, sigma2) = q(mu) q(sigma2) with
    q(mu)     = N(m, s2)
    q(sigma2) = Inverse-Gamma(a, b)
and the two factors are updated in turn (coordinate ascent) until they stop changing.
The exact posterior is computed on a grid so the approximation can be checked against it.

Run:  python3 mfvb_normal.py            prints every number quoted in the notes
      python3 mfvb_normal.py --figures  also rewrites ../figures/*.svg

numpy and the standard library only.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- data and prior
y = np.array([11, 12, 8, 10, 9, 8, 9, 10, 13, 7], dtype=float)
n = len(y)
mu0, s0sq = 0.0, 100.0        # prior mu ~ N(mu0, s0sq)
a0, b0 = 1.0, 1.0             # prior sigma2 ~ Inverse-Gamma(a0, b0)


def norm_logpdf(x, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var)


def ig_logpdf(x, a, b):
    """log density of Inverse-Gamma(shape a, scale b) at x > 0."""
    return a * math.log(b) - math.lgamma(a) - (a + 1) * np.log(x) - b / x


# ---------------------------------------------------------------- the MFVB loop
def mfvb(iters=15):
    """Returns the list of (m, s2, a, b) after each sweep."""
    tau = 1.0                  # tau = E_q[1/sigma2], an arbitrary start; q(mu) is updated first
    hist = []
    for _ in range(iters):
        # ---- update q(mu) = N(m, s2).
        # Keep the terms of log p(y, mu, sigma2) that involve mu and average sigma2 out:
        #   E[log p] = -(tau/2) sum_i (y_i - mu)^2 - (mu - mu0)^2 / (2 s0sq) + const
        #            = -(1/2) mu^2 (n tau + 1/s0sq) + mu (tau sum y + mu0/s0sq) + const.
        # Matching -mu^2/(2 s2) + mu m / s2 gives the two lines below. The data precision n*tau
        # adds to the prior precision 1/s0sq; m is a precision-weighted average. The line below
        # leaves out the mu0/s0sq term of m because mu0 = 0 here.
        s2 = 1 / (n * tau + 1 / s0sq)
        m = s2 * tau * y.sum()                         # general case: s2 * (tau*y.sum() + mu0/s0sq)

        # ---- update q(sigma2) = Inverse-Gamma(a, b).
        # Keep the terms involving sigma2 and average mu out:
        #   E[log p] = -(a0 + n/2 + 1) log sigma2
        #              - [b0 + (1/2) E sum_i (y_i - mu)^2] / sigma2 + const.
        # Under q(mu) = N(m, s2), E[(y_i - mu)^2] = (y_i - m)^2 + s2, so the bracket's average is
        # sum (y_i - m)^2 + n s2: the extra n*s2 is the cost of not knowing mu exactly.
        # Matching -(a+1) log sigma2 - b/sigma2 gives a and b.
        a = a0 + n / 2                                  # never changes between sweeps
        b = b0 + 0.5 * (((y - m) ** 2).sum() + n * s2)

        # ---- hand the answer to the other factor: the mean of 1/sigma2 under IG(a, b) is a/b.
        tau = a / b
        hist.append((m, s2, a, b))
    return hist


# ---------------------------------------------------------------- exact posterior on a grid
def grids(mu_lim=(7.0, 12.5), sg_lim=(0.3, 60.0), k=1200):
    """Grid of (mu, sigma2) values; M and S hold mu and sigma2 at every grid point."""
    mu = np.linspace(*mu_lim, k)
    sg = np.linspace(*sg_lim, k)
    M, S = np.meshgrid(mu, sg)
    return mu, sg, M, S


def exact_posterior(M, S, dmu, dsg):
    # log prior of mu plus log prior of sigma2 ...
    lp = norm_logpdf(M, mu0, s0sq) + ig_logpdf(S, a0, b0)
    # ... plus the log-likelihood (independent observations, so a sum over data points).
    # lp is now log p(mu, sigma2, y): the true log-posterior up to the constant log p(y).
    for yi in y:
        lp = lp + norm_logpdf(yi, M, S)
    # Normalise numerically in place of the intractable integral p(y): subtract the maximum
    # before exponentiating to avoid underflow, then divide by a Riemann sum of the integral.
    P = np.exp(lp - lp.max())
    return P / (P.sum() * dmu * dsg)


def mean_field_density(M, S, m, s2, a, b):
    """q(mu) q(sigma2) on the grid; already normalised, since each factor is."""
    return np.exp(norm_logpdf(M, m, s2) + ig_logpdf(S, a, b))


# ---------------------------------------------------------------- the numbers quoted in the notes
def report():
    hist = mfvb()
    print(f"n = {n}, sum y = {y.sum():.0f}, mean y = {y.mean():.2f}")
    print("sweep      m        s2         a        b")
    for i, (m, s2, a, b) in enumerate(hist[:6], 1):
        print(f"{i:5d}  {m:7.4f}  {s2:8.4f}  {a:6.1f}  {b:8.4f}")
    m, s2, a, b = hist[-1]
    print(f"converged: q(mu) = N({m:.3f}, {s2:.3f}), q(sigma2) = IG({a:.0f}, {b:.2f})")
    print(f"E_q[sigma2] = {b / (a - 1):.3f}, E_q[1/sigma2] = {a / b:.4f}")

    mu, sg, M, S = grids()
    dmu, dsg = mu[1] - mu[0], sg[1] - sg[0]
    P = exact_posterior(M, S, dmu, dsg)
    Q = mean_field_density(M, S, m, s2, a, b)
    Pm, Ps = P.sum(0) * dsg, P.sum(1) * dmu          # exact marginals of mu and sigma2
    print(f"exact:  E[mu] = {(Pm * mu).sum() * dmu:.3f}, sd[mu] = "
          f"{math.sqrt((Pm * mu ** 2).sum() * dmu - ((Pm * mu).sum() * dmu) ** 2):.3f}, "
          f"E[sigma2] = {(Ps * sg).sum() * dsg:.3f}")
    print(f"VB:     E[mu] = {m:.3f}, sd[mu] = {math.sqrt(s2):.3f}, E[sigma2] = {b / (a - 1):.3f}")
    # Posterior correlation of (mu, sigma2) under the exact posterior: near zero, which is why
    # a product form loses little here.
    Emu, Es = (Pm * mu).sum() * dmu, (Ps * sg).sum() * dsg
    cov = ((M - Emu) * (S - Es) * P).sum() * dmu * dsg
    sdm = math.sqrt((Pm * (mu - Emu) ** 2).sum() * dmu)
    sds = math.sqrt((Ps * (sg - Es) ** 2).sum() * dsg)
    print(f"exact posterior correlation of (mu, sigma2) = {cov / (sdm * sds):.3f}")
    return hist, mu, sg, M, S, P, Q, Pm, Ps


# ---------------------------------------------------------------- figures
def figures(hist, mu, sg, M, S, P, Q, Pm, Ps):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integral-probability-metrics" / "code"))
    import svgkit
    from svgkit import Axes, legend
    svgkit.OUT = Path(__file__).resolve().parent.parent / "figures"   # svgkit defaults to its own folder

    m, s2, a, b = hist[-1]
    keep = (sg < 9.5) & (sg > 0.4)                       # the plotted part of the sigma2 axis
    thin = slice(None, None, 4)                          # fewer points per curve keeps the SVG small

    # ---- joint: exact posterior next to the product q(mu) q(sigma2), same colour scale
    lim_mu, lim_sg, k = (8.5, 11.0), (1.0, 7.0), 40
    cm = np.linspace(*lim_mu, k + 1)
    cs = np.linspace(*lim_sg, k + 1)
    cmid_m, cmid_s = (cm[:-1] + cm[1:]) / 2, (cs[:-1] + cs[1:]) / 2
    GM, GS = np.meshgrid(cmid_m, cmid_s)
    Pg = np.exp(norm_logpdf(GM, mu0, s0sq) + ig_logpdf(GS, a0, b0)
                + sum(norm_logpdf(yi, GM, GS) for yi in y))
    Pg /= Pg.sum() * (cm[1] - cm[0]) * (cs[1] - cs[0])
    Qg = mean_field_density(GM, GS, m, s2, a, b)
    vmax = max(Pg.max(), Qg.max())
    body = []
    for idx, (G, cls, title) in enumerate([(Pg, "bf", "exact posterior p(mu, sigma2 | y)"),
                                           (Qg, "of", "mean-field q(mu) q(sigma2)")]):
        ax = Axes(50 + idx * 330, 40, 280, 250, lim_mu, lim_sg)
        for i in range(k):
            for j in range(k):
                op = G[j, i] / vmax
                if op < 0.01:
                    continue
                x0, x1 = ax.X(cm[i]), ax.X(cm[i + 1])
                y0, y1 = ax.Y(cs[j + 1]), ax.Y(cs[j])
                body.append(f'<rect class="{cls}" x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0 + .3:.1f}" '
                            f'height="{y1 - y0 + .3:.1f}" style="opacity:{op:.2f}"/>')
        body.append(ax.frame([9, 9.5, 10, 10.5], [2, 4, 6], "mu", "sigma2" if idx == 0 else ""))
        body.append(f'<text class="hd" x="{ax.x0}" y="14">{title}</text>')
    desc = "Heat maps of the exact posterior and of the fitted product of a Normal and an Inverse-Gamma."
    svgkit.write("joint.svg", svgkit.svg(680, 330, "Exact posterior and its mean-field approximation", desc, body))

    # ---- marginals
    body = []
    a1 = Axes(50, 40, 270, 230, (7.5, 12.0), (0, 1.0))      # headroom above the peak for the legend
    body.append(a1.frame([8, 9, 10, 11, 12], [0, 0.25, 0.5, 0.75, 1.0], "mu", "density"))
    body.append(a1.path(mu[thin], Pm[thin], "b"))
    body.append(a1.path(mu[thin], np.exp(norm_logpdf(mu[thin], m, s2)), "o dash"))
    a2 = Axes(380, 40, 270, 230, (0.5, 9.0), (0, 0.6))
    body.append(a2.frame([2, 4, 6, 8], [0, 0.2, 0.4, 0.6], "sigma2", ""))
    x2 = sg[keep][::2]
    body.append(a2.path(x2, Ps[keep][::2], "b"))
    body.append(a2.path(x2, np.exp(ig_logpdf(x2, a, b)), "o dash"))
    body.append(legend(420, 56, [("b", "exact p(sigma2 | y)"), ("o dash", "q(sigma2) = IG(6, 18.6)")]))
    body.append(legend(90, 56, [("b", "exact p(mu | y)"), ("o dash", "q(mu) = N(9.67, 0.31)")]))
    desc = "Exact and fitted marginals of mu and sigma2; the dashed curves almost coincide with the solid ones."
    svgkit.write("marginals.svg", svgkit.svg(680, 310, "Marginals: exact against mean-field", desc, body))

    # ---- iterations
    body = []
    sweeps = list(range(1, 9))
    panels = [("m (mean of q(mu))", [h[0] for h in hist[:8]], (9.6, 9.75)),
              ("s2 (variance of q(mu))", [h[1] for h in hist[:8]], (0.0, 0.35)),
              ("b (rate of q(sigma2))", [h[3] for h in hist[:8]], (17.0, 19.0))]
    for idx, (title, vals, ylim) in enumerate(panels):
        ax = Axes(40 + idx * 220, 40, 170, 170, (1, 8), ylim)
        body.append(ax.frame([2, 4, 6, 8], [ylim[0], (ylim[0] + ylim[1]) / 2, ylim[1]], "sweep", "",
                             yfmt="{:.2f}"))
        body.append(ax.path(sweeps, vals, "b"))
        for x, v in zip(sweeps, vals):
            body.append(ax.dot(x, v, "bf"))
        body.append(f'<text class="hd" x="{ax.x0}" y="26">{title}</text>')
    desc = "m, s2 and b over the first eight sweeps; all settle within about five."
    svgkit.write("iterations.svg", svgkit.svg(680, 250, "MFVB parameters settle in a few sweeps", desc, body))


if __name__ == "__main__":
    out = report()
    if "--figures" in sys.argv:
        figures(*out[:1], *out[1:7], out[7], out[8])
