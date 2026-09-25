#!/usr/bin/env python3
"""Normality normalization: the power transform, its one-step lambda, and Lemma B.1.

Everything the paper derives in Appendices B-D is checkable without a network,
because the layer only ever sees one channel's minibatch of pre-activations.

Checked here:

  * Eq 3 (Yeo-Johnson) is the identity at lambda = 1 and is continuous across the
    lambda = 0 and lambda = 2 special cases;
  * Eq 13, the profile NLL, including the h < 0 Jacobian term the paper omits
    "by symmetry": (lambda - 1) * sign(h) * log(1 + |h|);
  * Eqs 18 and 21 (d psi / d lambda at lambda = 1) against finite differences,
    plus the h < 0 branch, which the paper does not write out;
  * Eqs 15-21 collapse, after standardisation, to three batch moments -- the mu'
    terms in Eqs 16/19 are identically zero;
  * the one Newton step of Eq 5 against the exact minimiser of the NLL, on
    skewed, heavy-tailed and Gaussian batches, and the skewness it leaves behind;
  * Lemma B.1: with N(0,1) marginals and a fixed correlation, the Gaussian joint
    has the least mutual information -- a non-Gaussian joint with the same rho
    has strictly more, and at rho = 0 it can be uncorrelated yet dependent;
  * a per-unit monotone transform leaves mutual information unchanged while it
    moves Pearson correlation, so the power transform by itself cannot make two
    units more independent;
  * the noise scale s of Sec 4.2 is sqrt(2/pi) for a standard normal channel.

Standard library and numpy only.

Run:  python3 normality_norm.py
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- Eq 3
def psi(h, lam):
    """Yeo-Johnson power transform, Eq 3, vectorised over h."""
    h = np.asarray(h, float)
    out = np.empty_like(h)
    pos = h >= 0
    if abs(lam) > 1e-12:
        out[pos] = ((1 + h[pos]) ** lam - 1) / lam
    else:
        out[pos] = np.log1p(h[pos])
    if abs(lam - 2) > 1e-12:
        out[~pos] = -((1 - h[~pos]) ** (2 - lam) - 1) / (2 - lam)
    else:
        out[~pos] = -np.log1p(-h[~pos])
    return out


# ---------------------------------------------------------------- Eq 13
def nll(h, lam):
    """Profile NLL (Eq 13), with the Jacobian term for both signs of h."""
    x = psi(h, lam)
    jac = np.mean(np.sign(h) * np.log1p(np.abs(h)))
    return 0.5 * (np.log(2 * np.pi) + 1) + 0.5 * np.log(x.var()) - (lam - 1) * jac


# ---------------------------------------------------------------- Eqs 18, 21
def dpsi(h):
    """d psi / d lambda at lambda = 1. Eq 18 for h >= 0; the h < 0 branch derived here."""
    t = np.abs(h)
    return (1 + t) * np.log1p(t) - t                     # even in h


def d2psi(h):
    """d^2 psi / d lambda^2 at lambda = 1. Eq 21 for h >= 0; odd extension for h < 0."""
    t = np.abs(h)
    g2 = (1 + t) * np.log1p(t) ** 2 - 2 * ((1 + t) * np.log1p(t) - t)
    return np.sign(h) * g2


def newton_lambda(h):
    """Eq 5 with Eqs 15-21 reduced to moments. Assumes h standardised (mean 0, var 1)."""
    p1, p2 = dpsi(h), d2psi(h)
    ds2 = 2 * np.mean(h * p1)                             # Eq 17, mu' term drops
    d2s2 = 2 * (np.mean(h * p2) + p1.var())               # Eq 20, mu'' term drops
    jac = np.mean(np.sign(h) * np.log1p(np.abs(h)))
    L1 = 0.5 * ds2 - jac                                  # Eq 15 with sigma^2(1) = 1
    L2 = -0.5 * ds2 ** 2 + 0.5 * d2s2
    return 1 - L1 / L2, L1, L2


def newton_lambda_paper(h):
    """Eqs 15-21 transcribed literally (mu' and mu'' terms kept), for comparison."""
    p1, p2 = dpsi(h), d2psi(h)
    mu, s2 = h.mean(), h.var()
    mu1, mu2 = p1.mean(), p2.mean()
    ds2 = 2 * np.mean((h - mu) * (p1 - mu1))
    d2s2 = 2 * np.mean((h - mu) * (p2 - mu2) + (p1 - mu1) ** 2)
    jac = np.mean(np.sign(h) * np.log1p(np.abs(h)))
    L1 = ds2 / (2 * s2) - jac
    L2 = -ds2 ** 2 / (2 * s2 ** 2) + d2s2 / (2 * s2)
    return 1 - L1 / L2


def exact_lambda(h, lo=-3.0, hi=5.0):
    """Minimiser of Eq 13 by a coarse grid, then a fine grid around the best point."""
    for _ in range(3):
        grid = np.linspace(lo, hi, 161)
        best = grid[np.argmin([nll(h, l) for l in grid])]
        step = grid[1] - grid[0]
        lo, hi = best - step, best + step
    return best


def standardise(u):
    return (u - u.mean()) / np.sqrt(u.var() + 1e-5)


def skew(u):
    u = (u - u.mean()) / u.std()
    return np.mean(u ** 3)


# ---------------------------------------------------------------- Lemma B.1
def bvn(x, y, r):
    q = (x * x - 2 * r * x * y + y * y) / (1 - r * r)
    return np.exp(-q / 2) / (2 * np.pi * np.sqrt(1 - r * r))


def mi_mixture(r1, r2, w=0.5, L=7.0, n=1401):
    """MI of w*N2(r1) + (1-w)*N2(r2). Both components have N(0,1) marginals, so the
    mixture does too, with correlation w*r1 + (1-w)*r2 -- but it is not Gaussian."""
    g = np.linspace(-L, L, n)
    dx = g[1] - g[0]
    X, Y = np.meshgrid(g, g)
    p = w * bvn(X, Y, r1) + (1 - w) * bvn(X, Y, r2)
    phi = np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi)
    q = phi[None, :] * phi[:, None]
    m = p > 1e-300
    return float(np.sum(p[m] * np.log(p[m] / q[m])) * dx * dx)


def mi_gauss(rho):
    return -0.5 * np.log(1 - rho ** 2) + 0.0


# ---------------------------------------------------------------- main
def main():
    print("== Eq 3: identity at lambda = 1, continuity at the special cases ==")
    h = np.linspace(-4, 4, 17)
    print(f"  max |psi(h;1) - h|              = {np.abs(psi(h, 1) - h).max():.1e}")
    print(f"  max |psi(h;1e-7) - psi(h;0)|    = {np.abs(psi(h, 1e-7) - psi(h, 0)).max():.1e}")
    print(f"  max |psi(h;2-1e-7) - psi(h;2)|  = {np.abs(psi(h, 2 - 1e-7) - psi(h, 2)).max():.1e}")
    d = np.diff(psi(np.linspace(-4, 4, 2001), 0.3))
    print(f"  psi strictly increasing (lambda=0.3): {bool((d > 0).all())}")

    print("\n== Eqs 18 and 21 vs central finite differences (both signs of h) ==")
    h = np.array([-3.0, -1.0, -0.2, 0.2, 1.0, 3.0])
    e = 1e-4
    fd1 = (psi(h, 1 + e) - psi(h, 1 - e)) / (2 * e)
    fd2 = (psi(h, 1 + e) - 2 * psi(h, 1) + psi(h, 1 - e)) / e ** 2
    print(f"  max |dpsi  - FD| = {np.abs(dpsi(h) - fd1).max():.1e}   dpsi  = {np.round(dpsi(h), 4)}")
    print(f"  max |d2psi - FD| = {np.abs(d2psi(h) - fd2).max():.1e}   d2psi = {np.round(d2psi(h), 4)}")
    print("  -> d psi/d lambda is EVEN in h, d^2 psi/d lambda^2 is ODD")

    print("\n== Eqs 15-21: literal transcription == moment form (mu' terms vanish) ==")
    u = standardise(RNG.lognormal(0, 0.6, 4096))
    a, b = newton_lambda(u)[0], newton_lambda_paper(u)
    print(f"  lambda_hat moment form = {a:.6f}, literal = {b:.6f}, diff = {abs(a-b):.1e}")

    print("\n== Eq 13 NLL: analytic L', L'' at lambda = 1 vs finite differences of nll() ==")
    lam_hat, L1, L2 = newton_lambda(u)
    e = 1e-4
    fL1 = (nll(u, 1 + e) - nll(u, 1 - e)) / (2 * e)
    fL2 = (nll(u, 1 + e) - 2 * nll(u, 1) + nll(u, 1 - e)) / e ** 2
    print(f"  L'(1)  analytic {L1:+.5f}  FD {fL1:+.5f}")
    print(f"  L''(1) analytic {L2:+.5f}  FD {fL2:+.5f}")

    print("\n== One Newton step (Eq 5) vs exact NLL minimiser, N = 4096 ==")
    batches = {
        "gaussian":            RNG.normal(size=4096),
        "lognormal s=0.4":     RNG.lognormal(0, 0.4, 4096),
        "lognormal s=0.8":     RNG.lognormal(0, 0.8, 4096),
        "ReLU-ish gamma(2)":   RNG.gamma(2.0, 1.0, 4096),
        "left-skew -gamma(3)": -RNG.gamma(3.0, 1.0, 4096),
        "student-t (5 dof)":   RNG.standard_t(5, 4096),
    }
    print(f"  {'batch':22s} {'skew in':>8s} {'lam_newton':>10s} {'lam_exact':>9s} {'skew out':>8s} {'L2(1)>0':>7s}")
    for name, raw in batches.items():
        hb = standardise(raw)
        ln, _, l2 = newton_lambda(hb)
        le = exact_lambda(hb)
        print(f"  {name:22s} {skew(hb):8.3f} {ln:10.3f} {le:9.3f} {skew(psi(hb, ln)):8.3f} {str(l2 > 0):>7s}")
    print("  -> right skew gives lambda < 1, left skew lambda > 1; symmetric data stays near 1.")
    print("     Symmetric heavy tails (t) are NOT fixed: Yeo-Johnson only fights skew.")

    print("\n== small-skew rule of thumb: lambda_hat ~ 1 - c * skewness ==")
    for s in (0.1, 0.2, 0.3):
        hb = standardise(RNG.lognormal(0, s, 200000))
        ln = newton_lambda(hb)[0]
        print(f"  lognormal s={s}: skew {skew(hb):.3f}, 1 - lambda_hat = {1-ln:.3f}, ratio {(1-ln)/skew(hb):.3f}")

    print("\n== Lemma B.1: N(0,1) marginals, fixed rho -> Gaussian joint has least MI ==")
    print(f"  {'mixture r1, r2':16s} {'rho':>6s} {'MI mixture':>11s} {'MI gaussian':>12s}")
    for r1, r2 in [(0.6, -0.6), (0.9, -0.9), (0.9, 0.1), (0.8, 0.4), (0.5, 0.5)]:
        rho = 0.5 * r1 + 0.5 * r2
        print(f"  ({r1:+.1f}, {r2:+.1f})      {rho:6.2f} {mi_mixture(r1, r2):11.4f} {mi_gauss(rho):12.4f}")
    print("  -> rho = 0 rows: uncorrelated, normal marginals, yet dependent (MI > 0).")

    print("\n== monotone per-unit transforms keep MI, move Pearson rho ==")
    rho = 0.6
    z = RNG.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 400000)
    r_after = np.corrcoef(z[:, 0], np.exp(z[:, 1]))[0, 1]
    print(f"  Gaussian pair, rho = {rho}: MI = {mi_gauss(rho):.4f} nats")
    print(f"  after X2 -> exp(X2): Pearson = {r_after:.4f} (theory rho/sqrt(e-1) = {rho/np.sqrt(np.e-1):.4f}); MI unchanged")

    print("\n== Sec 4.2 noise scale: s = mean |x - xbar| for a N(0,1) channel ==")
    x = RNG.normal(size=10 ** 6)
    print(f"  s = {np.mean(np.abs(x - x.mean())):.4f}   sqrt(2/pi) = {np.sqrt(2/np.pi):.4f}")


if __name__ == "__main__":
    main()
