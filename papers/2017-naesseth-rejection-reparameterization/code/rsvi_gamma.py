#!/usr/bin/env python3
"""RSVI on the gamma distribution: the three claims that carry the paper.

Naesseth et al. argue that you can get a reparameterization gradient through a
rejection sampler by integrating out the accept/reject coin. Three things have
to be true for that to be useful, and all three are checkable on Gamma(alpha,1)
with the Marsaglia-Tsang sampler:

  1. Equation 4 is the right density for the accepted epsilon. Equivalently,
     and more revealingly, pi(eps;theta) = q(h(eps)) |dh/deps| -- the pullback
     of the target through the transformation.

  2. The decomposition grad = g_rep + g_cor is unbiased, and dropping g_cor
     (which is what naive autodiff through the sampler silently does) leaves a
     real bias.

  3. g_cor shrinks as the sampler gets more efficient, so RSVI's variance sits
     far below the score function estimator's -- and shape augmentation buys
     more of the same.

Standard library and numpy only, so digamma is implemented here.

Run:  python3 rsvi_gamma.py
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- digamma
def digamma(x):
    """psi(x) by upward recurrence onto the asymptotic series."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    while np.any(x < 6):
        m = x < 6
        out[m] -= 1.0 / x[m]
        x = np.where(m, x + 1, x)
    inv2 = 1.0 / x ** 2
    return out + (np.log(x) - 0.5 / x
                  - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 / 240))))


# ------------------------------------------- Marsaglia-Tsang, in epsilon space
def h(eps, alpha):
    """z = h(eps, alpha), Eq. 10. Valid where 1 + eps/sqrt(9*alpha-3) > 0."""
    d = alpha - 1.0 / 3.0
    return d * (1.0 + eps / np.sqrt(9.0 * d)) ** 3


def dh_deps(eps, alpha):
    d = alpha - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    return 3.0 * d * c * (1.0 + c * eps) ** 2


def dh_dalpha(eps, alpha):
    """Total derivative of h in alpha, with eps held fixed."""
    d = alpha - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    v = 1.0 + c * eps
    # dv/dalpha = eps * dc/dalpha, with dc/dalpha = -1/(2*sqrt(9)*d^{3/2}) = -c/(2d)
    return v ** 3 + d * 3.0 * v ** 2 * (-eps * c / (2.0 * d))


def sample_eps(alpha, n, rng):
    """Accepted epsilon from the rejection sampler (Algorithm 1)."""
    out = np.empty(n)
    filled, d = 0, alpha - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    while filled < n:
        eps = rng.standard_normal(n)
        v = 1.0 + c * eps
        u = rng.random(n)
        ok = v > 0
        vv = np.where(ok, v, 1.0) ** 3
        # log u < 0.5 eps^2 + d - d*v^3 + d*log(v^3)
        ok &= np.log(u) < 0.5 * eps ** 2 + d - d * vv + d * np.log(vv)
        take = eps[ok][: n - filled]
        out[filled:filled + take.size] = take
        filled += take.size
    return out


def acceptance_rate(alpha, n, rng):
    d, tot = alpha - 1.0 / 3.0, 0
    c = 1.0 / np.sqrt(9.0 * d)
    eps = rng.standard_normal(n)
    v = 1.0 + c * eps
    u = rng.random(n)
    ok = v > 0
    vv = np.where(ok, v, 1.0) ** 3
    ok &= np.log(u) < 0.5 * eps ** 2 + d - d * vv + d * np.log(vv)
    return ok.mean()


# ----------------------------------------------------------------- densities
def log_gamma_pdf(z, alpha):
    """log Gamma(z; alpha, 1) up to the normaliser's alpha-dependence handled below."""
    return (alpha - 1.0) * np.log(z) - z - log_gamma_fn(alpha)


def log_gamma_fn(a):
    """log Gamma(a) by Lanczos."""
    g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
         -176.61502916214059, 12.507343278686905, -0.13857109526572012,
         9.9843695780195716e-6, 1.5056327351493116e-7]
    a = np.asarray(a, dtype=float)
    x = np.full_like(a, 0.99999999999980993)
    for i, c in enumerate(g):
        x = x + c / (a + i)
    t = a + len(g) - 1.5
    return 0.5 * np.log(2 * np.pi) + (a - 0.5) * np.log(t) - t + np.log(x)


def log_pi(eps, alpha):
    """Eq. 4, in the pullback form pi(eps) = q(h(eps)) |dh/deps|."""
    return log_gamma_pdf(h(eps, alpha), alpha) + np.log(dh_deps(eps, alpha))


# ------------------------------------------------------- the two gradient parts
def grad_log_ratio(eps, alpha, step=1e-6):
    """d/dalpha log( q(h)/r(h) ) = d/dalpha [ log q(h(eps,a);a) + log|dh/deps| ]  (Eq. 8).

    That bracket is exactly log_pi, so the correction term's weight is
    d/dalpha log pi(eps; alpha) -- which is what Eq. 5 says it should be.
    """
    return (log_pi(eps, alpha + step) - log_pi(eps, alpha - step)) / (2 * step)


def rsvi_grad(f, df, eps, alpha):
    """(g_rep, g_cor) per sample, for E_{Gamma(alpha,1)}[f(z)]."""
    z = h(eps, alpha)
    g_rep = df(z) * dh_dalpha(eps, alpha)
    g_cor = f(z) * grad_log_ratio(eps, alpha)
    return g_rep, g_cor


def score_grad(f, z, alpha):
    """E_q[f(z) d/dalpha log q(z;alpha)] -- the score function estimator."""
    return f(z) * (np.log(z) - digamma(np.array([alpha]))[0])


# ---------------------------------------------------------------------- tests
def check_density(rng):
    print("1. Equation 4 gives the right density for the accepted epsilon\n")
    print("   pi(eps;a) = s(eps) q(h)/r(h) = q(h(eps,a);a) |dh/deps|")
    print("   (the two forms agree because r(h)|dh/deps| = s(eps))\n")
    print(f"   {'alpha':>6}  {'accept':>8}  {'max |hist - pi|':>16}  {'KL(hist||pi)':>13}")
    for alpha in (1.0, 2.0, 5.0, 10.0):
        eps = sample_eps(alpha, 400_000, rng)
        lo, hi = np.percentile(eps, [0.5, 99.5])
        edges = np.linspace(lo, hi, 61)
        mid, width = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
        # Compare as densities on the SAME window: histogram only the in-range
        # samples, and renormalise pi over the window. Otherwise the truncated
        # 1% of mass shows up as a fixed offset and swamps the real error.
        inside = eps[(eps >= lo) & (eps <= hi)]
        emp, _ = np.histogram(inside, bins=edges, density=True)
        pred = np.exp(log_pi(mid, alpha))
        pred /= np.sum(pred * width)
        m = emp > 0
        kl = float(np.sum(emp[m] * np.log(emp[m] / pred[m]) * width[m]))
        acc = acceptance_rate(alpha, 400_000, rng)
        print(f"   {alpha:>6.1f}  {acc:>8.3f}  {np.abs(emp - pred).max():>16.5f}  {kl:>13.2e}")
    print()


def check_unbiased(rng, n=400_000):
    print("2. g_rep + g_cor is unbiased; g_rep alone (naive autodiff) is not\n")
    print("   target: d/dalpha E[z^2] = d/dalpha alpha(alpha+1) = 2*alpha + 1\n")
    f, df = lambda z: z ** 2, lambda z: 2.0 * z
    print(f"   {'alpha':>6}  {'truth':>8}  {'g_rep only':>11}  {'g_rep+g_cor':>12}  {'score fn':>10}")
    for alpha in (1.0, 2.0, 5.0, 10.0):
        eps = sample_eps(alpha, n, rng)
        g_rep, g_cor = rsvi_grad(f, df, eps, alpha)
        z = h(eps, alpha)
        print(f"   {alpha:>6.1f}  {2*alpha+1:>8.3f}  {g_rep.mean():>11.3f}"
              f"  {(g_rep+g_cor).mean():>12.3f}  {score_grad(f, z, alpha).mean():>10.3f}")
    print("\n   Dropping g_cor is a bias, not noise: it does not shrink with n.\n")


def check_variance(rng, n=200_000):
    print("3. The correction shrinks as the sampler improves, and RSVI wins on variance\n")
    f, df = lambda z: z ** 2, lambda z: 2.0 * z
    print(f"   {'alpha':>6}  {'accept':>8}  {'|E g_cor| / |grad|':>19}"
          f"  {'Var RSVI':>11}  {'Var score':>11}  {'ratio':>8}")
    for alpha in (1.0, 2.0, 5.0, 10.0, 20.0):
        eps = sample_eps(alpha, n, rng)
        g_rep, g_cor = rsvi_grad(f, df, eps, alpha)
        z = h(eps, alpha)
        v_rsvi = float(np.var(g_rep + g_cor))
        v_score = float(np.var(score_grad(f, z, alpha)))
        share = abs(g_cor.mean()) / abs((g_rep + g_cor).mean())
        print(f"   {alpha:>6.1f}  {acceptance_rate(alpha, n, rng):>8.3f}  {share:>19.4f}"
              f"  {v_rsvi:>11.3f}  {v_score:>11.3f}  {v_score/v_rsvi:>7.1f}x")
    print()


def check_augmentation(rng, n=200_000):
    print("4. Shape augmentation: run the sampler at alpha+B, shrink back with uniforms\n")
    print("   z = zbar * prod_i u_i^{1/(alpha+i-1)},  zbar ~ Gamma(alpha+B,1)")
    print("   the shrink step is smooth in alpha, so it moves weight out of g_cor\n")
    alpha = 1.0
    f, df = lambda z: z ** 2, lambda z: 2.0 * z
    print(f"   {'B':>4}  {'sampler shape':>14}  {'accept':>8}  {'|E g_cor|/|grad|':>17}  {'Var':>10}")
    for B in (0, 1, 4, 10):
        a_s = alpha + B
        eps = sample_eps(a_s, n, rng)
        u = rng.random((B, n)) if B else np.ones((0, n))
        powers = np.array([1.0 / (alpha + i) for i in range(B)])[:, None]
        shrink = np.prod(u ** powers, axis=0) if B else np.ones(n)

        # z = h(eps, alpha+B) * shrink; differentiate the whole map in alpha
        zbar = h(eps, a_s)
        z = zbar * shrink
        dzbar = dh_dalpha(eps, a_s)
        # d/dalpha of prod u_i^{1/(alpha+i)} = shrink * sum_i -log(u_i)/(alpha+i)^2
        dshrink = shrink * np.sum(-np.log(np.clip(u, 1e-300, None)) * powers ** 2, axis=0) if B \
            else np.zeros(n)
        g_rep = df(z) * (dzbar * shrink + zbar * dshrink)
        g_cor = f(z) * grad_log_ratio(eps, a_s)
        g = g_rep + g_cor
        share = abs(g_cor.mean()) / abs(g.mean())
        print(f"   {B:>4}  {a_s:>14.1f}  {acceptance_rate(a_s, n, rng):>8.3f}"
              f"  {share:>17.4f}  {np.var(g):>10.3f}")
    print(f"\n   truth at alpha = {alpha}: 2*alpha + 1 = {2*alpha+1}\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("RSVI on Gamma(alpha, 1), Marsaglia-Tsang sampler")
    print("=" * 74, "\n")
    check_density(rng)
    check_unbiased(rng)
    check_variance(rng)
    check_augmentation(rng)
