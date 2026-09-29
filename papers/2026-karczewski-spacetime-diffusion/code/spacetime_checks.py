#!/usr/bin/env python3
"""The spacetime of diffusion models: every identity in the paper, checked on a toy.

The paper's own toy (App G.1) is a 1-D mixture of three Gaussians under a
variance-preserving schedule. For a Gaussian-mixture prior every denoising
distribution p(x0 | x_t) is again a Gaussian mixture, so eta, mu, psi, KL and
the Fisher-Rao metric are all available in closed form, and every claim in the
paper can be checked without a neural network. This script prints every
number quoted in ../notes.md.

Checked here:

  1. Prop C.2: p(x0|x_t) = q(x0) exp(eta.T - psi), with eta, T, psi as in
     Eqs 52-55, against brute-force Bayes on a grid.
  2. Eq 16 / 57: E[x0^2 | x_t] from the denoiser and its divergence (Tweedie).
  3. Prop C.1: G = (d eta/dz)^T (d mu/dz) equals the Fisher information
     E[score score^T], and equals J_eta^T Cov[T] J_eta.
  4. Lemma D.2: (eta1-eta2).(mu1-mu2) = 2 KL^S exactly, near and far; and
     the local approximation 1/2 dz^T G dz.
  5. Eq 14: the discrete energy converges to the continuous one as N grows.
  6. Memorylessness (Eq 11): the metric collapses as t -> T.
  7. The geometry does not depend on the noise schedule (VP versus VE).
  8. The Fig. 1 geodesic, and the straight spacetime line for comparison.
  9. Gaussian data: the spacetime is the hyperbolic half-plane, with a
     closed-form DiffED that grows like log(distance / noise).
 10. PF-ODE versus geodesic (Fig. 3) in the Gaussian case.
 11. Prop D.1: KL to a fixed point, accumulated along a curve.
 12. Hutchinson's trick (Eq 16) in D = 256: unbiased energy, slightly biased length.
 13. Prop B.1: under a bijective decoder, pullback geodesics decode to
     straight lines.
 14. Sec 6.3 on a 2-D toy: the Fisher-Rao geodesic between two wells moves
     mass by reweighting them, not through the channel; annealed Langevin
     along it still finds the channel.

Standard library and numpy only.

Run:  python3 spacetime_checks.py
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------ the toy (App G.1)
MU = np.array([-2.5, 0.5, 2.5])      # component means
PI = np.array([0.275, 0.45, 0.275])  # component weights
S2 = 0.75 ** 2                       # component variance
LMAX, LMIN = 10.0, -10.0             # log-SNR runs linearly from 10 (t=0) to -10 (t=T=1)


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def logsnr(t):
    return LMAX + (LMIN - LMAX) * t


def vp(t):
    """Variance preserving: alpha^2 = sigmoid(lambda), sigma^2 = sigmoid(-lambda)."""
    lam = logsnr(t)
    return np.sqrt(sigmoid(lam)), np.sqrt(sigmoid(-lam))


def ve(t):
    """Variance exploding with the same log-SNR: alpha = 1, sigma^2 = exp(-lambda)."""
    return np.ones_like(np.asarray(t, float)), np.exp(-logsnr(t) / 2)


def eta_of(x, t, sched=vp):
    """Eq 52: natural parameter of p(x0 | x_t)."""
    a, s = sched(t)
    return np.stack(np.broadcast_arrays(a * x / s ** 2, -(a ** 2) / (2 * s ** 2)), -1)


# ------------------------------------------------------------ the tilted family
def tilt(eta, mu=MU, pi=PI, s2=S2):
    """q(x) exp(eta1 x + eta2 x^2), normalised, as a Gaussian mixture.

    Each component N(mu_k, s2) times exp(e1 x + e2 x^2) is again Gaussian with
    precision 1/s2 - 2 e2 and mean v (mu_k/s2 + e1); the leftover constants
    become the new weights. Returns weights, means, variances and psi.
    """
    eta = np.asarray(eta, float)
    e1, e2 = eta[..., 0, None], eta[..., 1, None]
    v = 1.0 / (1.0 / s2 - 2.0 * e2)
    m = v * (mu / s2 + e1)
    logw = np.log(pi) + 0.5 * np.log(v / s2) + 0.5 * m ** 2 / v - 0.5 * mu ** 2 / s2
    top = logw.max(-1, keepdims=True)
    w = np.exp(logw - top)
    Z = w.sum(-1, keepdims=True)
    return w / Z, m, v, (np.log(Z) + top)[..., 0]


def raw_moments(eta, **kw):
    w, m, v, psi = tilt(eta, **kw)
    E1 = (w * m).sum(-1)
    E2 = (w * (m ** 2 + v)).sum(-1)
    E3 = (w * (m ** 3 + 3 * m * v)).sum(-1)
    E4 = (w * (m ** 4 + 6 * m ** 2 * v + 3 * v ** 2)).sum(-1)
    return E1, E2, E3, E4, psi


def mu_of_eta(eta, **kw):
    """Expectation parameter mu = E[T(x0)], T(x0) = (x0, x0^2)."""
    E1, E2, *_ = raw_moments(eta, **kw)
    return np.stack([E1, E2], -1)


def cov_T(eta, **kw):
    """Cov[T(x0)]: the Hessian of psi in natural coordinates."""
    E1, E2, E3, E4, _ = raw_moments(eta, **kw)
    c11, c12, c22 = E2 - E1 ** 2, E3 - E1 * E2, E4 - E2 ** 2
    return np.stack([np.stack([c11, c12], -1), np.stack([c12, c22], -1)], -2)


def psi_of_eta(eta, **kw):
    return tilt(eta, **kw)[3]


def kl(eta1, eta2, **kw):
    """Lemma D.1: KL(p_eta1 || p_eta2) = (eta1 - eta2).mu1 - psi1 + psi2."""
    return (((eta1 - eta2) * mu_of_eta(eta1, **kw)).sum(-1)
            - psi_of_eta(eta1, **kw) + psi_of_eta(eta2, **kw))


# ------------------------------------------------------------ a grid for brute force
GRID = np.linspace(-12, 12, 40001)
DX = GRID[1] - GRID[0]


def q_grid(mu=MU, pi=PI, s2=S2):
    return (pi * np.exp(-(GRID[:, None] - mu) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)).sum(-1)


def posterior_grid(x, t, sched=vp):
    """Bayes' rule on the grid: q(x0) N(x_t; alpha x0, sigma^2) / p_t(x_t)."""
    a, s = sched(t)
    un = q_grid() * np.exp(-(x - a * GRID) ** 2 / (2 * s ** 2))
    return un / (un.sum() * DX)


def marginal_logpdf(x, t, sched=vp):
    a, s = sched(t)
    var = a ** 2 * S2 + s ** 2
    return np.log((PI * np.exp(-(x - a * MU) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)).sum(-1))


def denoiser(x, t, sched=vp):
    """E[x0 | x_t], the quantity a trained network approximates."""
    return mu_of_eta(eta_of(x, t, sched))[..., 0]


# ------------------------------------------------------------ curves and geodesics
# Coordinates y = (xhat, ell): xhat = x_t / alpha_t (where the noisy point says x0 is),
# ell = log SNR. Then eta = (e^ell xhat, -e^ell / 2), for any schedule.
def eta_y(Y):
    el = np.exp(Y[..., 1])
    return np.stack([el * Y[..., 0], -el / 2], -1)


def jac_y(Y):
    el = np.exp(Y[..., 1])
    zero = np.zeros_like(el)
    return np.stack([np.stack([el, el * Y[..., 0]], -1), np.stack([zero, -el / 2], -1)], -2)


def y_of_z(x, t, sched=vp):
    a, _ = sched(t)
    return np.array([x / a, logsnr(t)])


def z_of_y(Y, sched=vp):
    t = (LMAX - Y[..., 1]) / (LMAX - LMIN)
    a, _ = sched(t)
    return np.stack([a * Y[..., 0], t], -1)


def chords(E, **kw):
    """(eta_{n+1} - eta_n).(mu_{n+1} - mu_n) for consecutive points, = 2 KL^S."""
    M = mu_of_eta(E, **kw)
    return ((E[1:] - E[:-1]) * (M[1:] - M[:-1])).sum(-1)


def energy_eta(E, **kw):
    """Eq 14 / 44."""
    return (len(E) - 1) / 2 * chords(E, **kw).sum()


def length_eta(E, **kw):
    """Eq 45."""
    return np.sqrt(np.maximum(chords(E, **kw), 0)).sum()


def reparam(Y, N, **kw):
    """Resample a curve at N points equally spaced in Fisher-Rao arc length."""
    c = np.sqrt(np.maximum(chords(eta_y(Y), **kw), 0))
    L = np.concatenate([[0], np.cumsum(c)])
    u = np.linspace(0, L[-1], N)
    return np.stack([np.interp(u, L, Y[:, 0]), np.interp(u, L, Y[:, 1])], -1)


def geodesic(ya, yb, N=64, iters=4000, init=None, **kw):
    """Minimise Eq 14 over the interior points of a discretised curve.

    The gradient of Eq 14 with respect to eta_n is exact:
      (N-1)/2 [ (mu_n - mu_{n-1}) - (mu_{n+1} - mu_n)
                + H_n ((eta_n - eta_{n-1}) - (eta_{n+1} - eta_n)) ],  H_n = Cov[T] at n,
    chained to y coordinates and preconditioned by the metric itself (a natural
    gradient step), which makes the iteration behave like averaging neighbours.
    """
    if init is None:
        s = np.linspace(0, 1, N)[:, None]
        Y = (1 - s) * ya + s * yb
    else:
        Y = init.copy()
    for _ in range(iters):
        E = eta_y(Y)
        M = mu_of_eta(E, **kw)
        H = cov_T(E, **kw)
        dE, dM = E[1:] - E[:-1], M[1:] - M[:-1]
        gE = np.zeros_like(E)
        gE[:-1] -= dM + np.einsum('nij,nj->ni', H[:-1], dE)
        gE[1:] += dM + np.einsum('nij,nj->ni', H[1:], dE)
        gE *= (N - 1) / 2
        J = jac_y(Y)
        g = np.einsum('nij,ni->nj', J, gE)
        G = np.einsum('nki,nkl,nlj->nij', J, H, J)
        step = np.linalg.solve(G[1:-1], g[1:-1][..., None])[..., 0] / (N - 1)
        Y[1:-1] -= 0.4 * np.clip(step, -0.3, 0.3)
    return Y


# ------------------------------------------------------------ D-dimensional isotropic mixtures
class Mixture:
    """q(x) = sum_k w_k N(x; c_k, v_k I_D), with closed-form tilts, as in the 1-D code above."""

    def __init__(self, C, W, V):
        self.C, self.W, self.V = np.asarray(C, float), np.asarray(W, float), np.asarray(V, float)
        self.D = self.C.shape[1]

    def tilt(self, E):                                    # E: (..., D+1)
        D, C, V = self.D, self.C, self.V
        a, b = E[..., None, :D], E[..., None, D]
        v = 1.0 / (1.0 / V - 2.0 * b)
        m = v[..., None] * (C / V[:, None] + a)
        lw = (np.log(self.W) + D / 2 * np.log(v / V) + (m ** 2).sum(-1) / (2 * v)
              - (C ** 2).sum(-1) / (2 * V))
        w = np.exp(lw - lw.max(-1, keepdims=True))
        return w / w.sum(-1, keepdims=True), m, v

    def stats(self, E):
        """mu = E[T] and H = Cov[T] for T = (x, |x|^2), batched over E."""
        D = self.D
        w, m, v = self.tilt(E)
        mm = (m ** 2).sum(-1)
        e2 = mm + D * v
        mu = np.concatenate([(w[..., None] * m).sum(-2), (w * e2).sum(-1, keepdims=True)], -1)
        S = np.zeros(E.shape[:-1] + (D + 1, D + 1))
        S[..., :D, :D] = (np.einsum('...k,...ki,...kj->...ij', w, m, m)
                          + np.eye(D) * (w * v).sum(-1)[..., None, None])
        S[..., :D, D] = (w[..., None] * m * (mm + (D + 2) * v)[..., None]).sum(-2)
        S[..., D, :D] = S[..., :D, D]
        S[..., D, D] = (w * (e2 ** 2 + 2 * D * v ** 2 + 4 * v * mm)).sum(-1)
        return mu, S - mu[..., :, None] * mu[..., None, :]

    def logq(self, X):
        d2 = ((X[..., None, :] - self.C) ** 2).sum(-1)
        l = np.log(self.W) - self.D / 2 * np.log(2 * np.pi * self.V) - d2 / (2 * self.V)
        top = l.max(-1, keepdims=True)
        return (top + np.log(np.exp(l - top).sum(-1, keepdims=True)))[..., 0]

    def grad_logq(self, X):
        d2 = ((X[..., None, :] - self.C) ** 2).sum(-1)
        l = np.log(self.W) - self.D / 2 * np.log(self.V) - d2 / (2 * self.V)
        r = np.exp(l - l.max(-1, keepdims=True))
        r /= r.sum(-1, keepdims=True)
        return (r[..., None] * (self.C - X[..., None, :]) / self.V[:, None]).sum(-2)


def eta_yD(Y):
    el = np.exp(Y[..., -1:])
    return np.concatenate([el * Y[..., :-1], -el / 2], -1)


def jac_yD(Y):
    D = Y.shape[-1] - 1
    el = np.exp(Y[..., -1])
    J = np.zeros(Y.shape[:-1] + (D + 1, D + 1))
    J[..., :D, :D] = np.eye(D) * el[..., None, None]
    J[..., :D, D] = el[..., None] * Y[..., :-1]
    J[..., D, D] = -el / 2
    return J


def geodesicD(fam, ya, yb, N=40, iters=1500):
    """The same natural-gradient minimisation of Eq 14 as geodesic(), for any D."""
    s = np.linspace(0, 1, N)[:, None]
    Y = (1 - s) * ya + s * yb
    for _ in range(iters):
        E = eta_yD(Y)
        M, H = fam.stats(E)
        dE, dM = E[1:] - E[:-1], M[1:] - M[:-1]
        gE = np.zeros_like(E)
        gE[:-1] -= dM + np.einsum('nij,nj->ni', H[:-1], dE)
        gE[1:] += dM + np.einsum('nij,nj->ni', H[1:], dE)
        gE *= (N - 1) / 2
        J = jac_yD(Y)
        g = np.einsum('nij,ni->nj', J, gE)
        G = np.einsum('nki,nkl,nlj->nij', J, H, J)
        step = np.linalg.solve(G[1:-1], g[1:-1][..., None])[..., 0] / (N - 1)
        Y[1:-1] -= 0.4 * np.clip(step, -0.3, 0.3)
    return Y


# ------------------------------------------------------------ Gaussian data
def fisher_rao_normal(m1, s1, m2, s2, D=1):
    """Closed-form Fisher-Rao distance between N(m1, s1^2 I_D) and N(m2, s2^2 I_D).

    The metric is (|dm|^2 + 2D ds^2)/s^2: a scaled hyperbolic half-plane.
    """
    dm2 = np.sum((np.asarray(m1) - np.asarray(m2)) ** 2)
    return np.sqrt(2 * D) * np.arccosh(1 + (dm2 / (2 * D) + (s1 - s2) ** 2) / (2 * s1 * s2))


def section(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def main():
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ 1
    section('1. Prop C.2: denoising distributions are an exponential family')
    worst = 0.0
    for x, t in [(-2.3, 0.35), (2.0, 0.4), (0.3, 0.1), (1.0, 0.7), (-4.0, 0.25)]:
        brute = posterior_grid(x, t)
        eta = eta_of(x, t)
        a, s = vp(t)
        # Eq 55 with the marginal log-density: psi = log p_t + 1/2 log(2 pi s^2) + x^2 / 2 s^2
        psi55 = marginal_logpdf(x, t) + 0.5 * np.log(2 * np.pi * s ** 2) + x ** 2 / (2 * s ** 2)
        fam = q_grid() * np.exp(eta[0] * GRID + eta[1] * GRID ** 2 - psi55)
        err = np.abs(fam - brute).max() / brute.max()
        worst = max(worst, err, abs(psi55 - psi_of_eta(eta)))
        print(f'  (x_t, t) = ({x:+.1f}, {t:.2f})  eta = ({eta[0]:+8.3f}, {eta[1]:+8.3f})'
              f'  psi(Eq 55) = {psi55:+.5f}  psi(mixture) = {psi_of_eta(eta):+.5f}'
              f'  max rel. density error {err:.1e}')
    print(f'  worst discrepancy {worst:.1e}')

    # ------------------------------------------------------------------ 2
    section('2. Eq 16 / 57: the second moment from the denoiser and its divergence')
    for x, t in [(-2.3, 0.35), (0.3, 0.1), (1.0, 0.7)]:
        a, s = vp(t)
        h = 1e-5
        div = (denoiser(x + h, t) - denoiser(x - h, t)) / (2 * h)
        tweedie = denoiser(x, t) ** 2 + s ** 2 / a * div
        brute = (posterior_grid(x, t) * GRID ** 2).sum() * DX
        mean_brute = (posterior_grid(x, t) * GRID).sum() * DX
        # Tweedie's mean from the score, Eq 58
        hs = 1e-5
        score = (marginal_logpdf(x + hs, t) - marginal_logpdf(x - hs, t)) / (2 * hs)
        mean58 = (x + s ** 2 * score) / a
        print(f'  (x_t, t) = ({x:+.1f}, {t:.2f})  E[x0|x_t]: Eq 58 {mean58:+.6f}, grid {mean_brute:+.6f}'
              f'   E[x0^2|x_t]: Eq 16 {tweedie:.6f}, grid {brute:.6f}')

    # ------------------------------------------------------------------ 3
    section('3. Prop C.1: the Fisher-Rao metric is J_eta^T J_mu')
    for x, t in [(-2.3, 0.35), (0.3, 0.1), (1.0, 0.7)]:
        z = np.array([x, t])
        h = 1e-6

        def eta_z(zz):
            return eta_of(zz[0], zz[1])

        def mu_z(zz):
            return mu_of_eta(eta_z(zz))

        Jeta = np.stack([(eta_z(z + h * e) - eta_z(z - h * e)) / (2 * h) for e in np.eye(2)], -1)
        Jmu = np.stack([(mu_z(z + h * e) - mu_z(z - h * e)) / (2 * h) for e in np.eye(2)], -1)
        G_prop = Jeta.T @ Jmu
        G_cov = Jeta.T @ cov_T(eta_z(z)) @ Jeta
        # Fisher information by brute force: score of log p(x0 | z) in z, averaged over x0
        logp = lambda zz: np.log(posterior_grid(zz[0], zz[1]) + 1e-300)
        sc = np.stack([(logp(z + 1e-5 * e) - logp(z - 1e-5 * e)) / 2e-5 for e in np.eye(2)], -1)
        p = posterior_grid(x, t)
        G_fisher = (p[:, None, None] * sc[:, :, None] * sc[:, None, :]).sum(0) * DX
        print(f'  (x_t, t) = ({x:+.1f}, {t:.2f})')
        for name, G in [('J_eta^T J_mu      ', G_prop), ('J_eta^T Cov[T] J  ', G_cov),
                        ('E[score score^T]  ', G_fisher)]:
            print(f'     {name} [[{G[0,0]:10.4f} {G[0,1]:10.4f}] [{G[1,0]:10.4f} {G[1,1]:10.4f}]]')
        print(f'     asymmetry of J_eta^T J_mu: {abs(G_prop[0,1]-G_prop[1,0]):.1e}')

    # ------------------------------------------------------------------ 4
    section('4. Lemma D.2: the chord (d eta).(d mu) is exactly 2 KL^S')
    pairs = [((-2.3, 0.35), (-2.25, 0.36)), ((-2.3, 0.35), (-1.0, 0.45)), ((-2.3, 0.35), (2.0, 0.40)),
             ((0.5, 0.15), (1.5, 0.15))]
    for (x1, t1), (x2, t2) in pairs:
        e1, e2 = eta_of(x1, t1), eta_of(x2, t2)
        chord = ((e1 - e2) * (mu_of_eta(e1) - mu_of_eta(e2))).sum()
        klf, klb = kl(e1, e2), kl(e2, e1)
        # brute-force KLs on the grid
        p1, p2 = posterior_grid(x1, t1), posterior_grid(x2, t2)
        ok = (p1 > 1e-300) & (p2 > 1e-300)
        kf = (p1[ok] * np.log(p1[ok] / p2[ok])).sum() * DX
        kb = (p2[ok] * np.log(p2[ok] / p1[ok])).sum() * DX
        # local quadratic approximation 1/2 dz^T G dz, at z1
        z1 = np.array([x1, t1]); dz = np.array([x2 - x1, t2 - t1]); h = 1e-6
        Je = np.stack([(eta_of(*(z1 + h * e)) - eta_of(*(z1 - h * e))) / (2 * h) for e in np.eye(2)], -1)
        G = Je.T @ cov_T(e1) @ Je
        print(f'  z1=({x1:+.2f},{t1:.2f}) z2=({x2:+.2f},{t2:.2f}): chord {chord:10.4f}'
              f' | KL fwd+bwd {klf + klb:10.4f} (grid {kf + kb:10.4f})'
              f' | dz^T G dz {dz @ G @ dz:10.4f}')

    # ------------------------------------------------------------------ 5
    section('5. Eq 14: discrete energy of a fixed curve converges as N grows')
    # the straight spacetime segment from (-2.3, 0.35) to (2.0, 0.40)
    def curve(N):
        s = np.linspace(0, 1, N)
        return np.stack([-2.3 + 4.3 * s, 0.35 + 0.05 * s], -1)

    def cont_energy(Nq=20001):
        s = np.linspace(0, 1, Nq)
        Z = curve(Nq)
        a, sg = vp(Z[:, 1])
        E = eta_of(Z[:, 0], Z[:, 1])
        dE = np.gradient(E, s, axis=0)
        H = cov_T(E)
        integrand = np.einsum('ni,nij,nj->n', dE, H, dE)
        return 0.5 * np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(s))

    ref = cont_energy()
    for N in [4, 8, 16, 32, 64, 128, 256]:
        Z = curve(N)
        E = eta_of(Z[:, 0], Z[:, 1])
        print(f'  N = {N:4d}  energy {energy_eta(E):10.4f}   length {length_eta(E):8.4f}')
    print(f'  continuous energy (quadrature) {ref:10.4f}')

    # ------------------------------------------------------------------ 6
    section('6. Memorylessness: the metric collapses as t -> T')
    for t in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]:
        z = np.array([1.0, t]); h = 1e-7
        Je = np.stack([(eta_of(*(z + h * e)) - eta_of(*(z - h * e))) / (2 * h) for e in np.eye(2)], -1)
        G = Je.T @ cov_T(eta_of(*z)) @ Je
        ev = np.linalg.eigvalsh(G)
        e = eta_of(1.0, t)
        print(f'  t = {t:4.2f}  log SNR {logsnr(t):+6.1f}  eta(1, t) = ({e[0]:+.2e}, {e[1]:+.2e})'
              f'  G_xx = {G[0,0]:.3e}  eigenvalues {ev[0]:.2e}, {ev[1]:.2e}')
    # all x_T give almost the same distribution: KL^S between x_T = -3 and +3
    e1, e2 = eta_of(-3.0, 1.0), eta_of(3.0, 1.0)
    print(f'  KL^S between p(x0 | x_T=-3) and p(x0 | x_T=+3): '
          f'{0.5 * ((e1 - e2) * (mu_of_eta(e1) - mu_of_eta(e2))).sum():.2e}')

    # ------------------------------------------------------------------ 7
    section('7. The geometry does not depend on the noise schedule')
    s = np.linspace(0, 1, 200)
    xs_vp = -2.3 + 4.3 * s
    ts = 0.35 + 0.05 * s + 0.3 * np.sin(np.pi * s)
    E_vp = eta_of(xs_vp, ts, vp)
    # the same curve of *distributions* in VE coordinates: x_ve = x_vp / alpha_vp
    a, _ = vp(ts)
    E_ve = eta_of(xs_vp / a, ts, ve)
    print(f'  max |eta_VP - eta_VE| along the curve: {np.abs(E_vp - E_ve).max():.1e}')
    print(f'  energy in VP coordinates {energy_eta(E_vp):.6f}, in VE coordinates {energy_eta(E_ve):.6f}')
    print(f'  but the coordinates differ: x_t ranges over [{xs_vp.min():+.2f}, {xs_vp.max():+.2f}] (VP)'
          f' and [{(xs_vp / a).min():+.2f}, {(xs_vp / a).max():+.2f}] (VE)')

    # ------------------------------------------------------------------ 8
    section('8. The Fig. 1 geodesic: z1 = (-2.3, 0.35), z2 = (2.0, 0.40)')
    ya, yb = y_of_z(-2.3, 0.35), y_of_z(2.0, 0.40)
    for N in [16, 32, 64]:
        Y = geodesic(ya, yb, N=N, iters=6000)
        E = eta_y(Y)
        Z = z_of_y(Y)
        i = Z[:, 1].argmax()
        print(f'  N = {N:3d}  energy {energy_eta(E):8.4f}  length {length_eta(E):7.4f}'
              f'  2*energy/length^2 {2 * energy_eta(E) / length_eta(E) ** 2:.4f}'
              f'  highest noise t = {Z[i, 1]:.3f} at x_t = {Z[i, 0]:+.3f}')
    for (x1, t1), (x2, t2) in [((-2.3, 0.35), (-1.9, 0.35)), ((-2.3, 0.35), (2.0, 0.40))]:
        Yp = geodesic(y_of_z(x1, t1), y_of_z(x2, t2), N=32, iters=6000)
        up = lambda y: length_eta(eta_y(np.stack([np.full(400, y[0]), np.linspace(y[1], LMIN, 400)], -1)))
        via = up(y_of_z(x1, t1)) + up(y_of_z(x2, t2))
        print(f'  ({x1:+.1f},{t1:.2f}) -> ({x2:+.1f},{t2:.2f}): geodesic length {length_eta(eta_y(Yp)):.3f},'
              f' via complete noise {via:.3f} (ratio {length_eta(eta_y(Yp)) / via:.3f})')
    Zs = np.stack([np.linspace(-2.3, 2.0, 64), np.linspace(0.35, 0.40, 64)], -1)
    Es = eta_of(Zs[:, 0], Zs[:, 1])
    print(f'  straight segment in (x_t, t), N = 64: energy {energy_eta(Es):8.4f}  length {length_eta(Es):7.4f}')
    # the "noise everything, then regenerate" path: straight up to t = T in y coordinates,
    # across at t = T, straight down. It is an upper bound on the geodesic length.
    top = LMIN
    legs = [np.stack([np.full(400, ya[0]), np.linspace(ya[1], top, 400)], -1),
            np.stack([np.linspace(ya[0], yb[0], 400), np.full(400, top)], -1),
            np.stack([np.full(400, yb[0]), np.linspace(top, yb[1], 400)], -1)]
    parts = [length_eta(eta_y(L)) for L in legs]
    print(f'  noise-everything path (up to t = T, across, down): length {sum(parts):7.4f}'
          f' = {parts[0]:.3f} + {parts[1]:.3f} + {parts[2]:.3f}')

    # Fig. 3 left: PF-ODE trajectories versus geodesics with the same endpoints
    print('  Fig. 3: PF-ODE from x_T in {1, 0, -1} to t_min = 0.1 (Euler, 512 steps), vs geodesic:')
    def score(x, t):
        a, s = vp(t)
        var = a ** 2 * S2 + s ** 2
        comp = PI * np.exp(-(x - a * MU) ** 2 / (2 * var)) / np.sqrt(var)
        return (comp * (-(x - a * MU) / var)).sum(-1) / comp.sum(-1)
    def f_g2(t, h=1e-6):
        a1, s1 = vp(t - h); a2, s2_ = vp(t + h); a, s = vp(t)
        f = (np.log(a2) - np.log(a1)) / (2 * h)
        g2 = (s2_ ** 2 - s1 ** 2) / (2 * h) - 2 * f * s ** 2
        return f, g2
    for xT in [1.0, 0.0, -1.0]:
        ts = np.linspace(1.0, 0.1, 513)
        xs = [xT]
        for k in range(512):
            f, g2 = f_g2(ts[k])
            xs.append(xs[-1] + (f * xs[-1] - 0.5 * g2 * score(xs[-1], ts[k])) * (ts[k + 1] - ts[k]))
        xs = np.array(xs)
        # both curves at 65 points, equally spaced in arc length, so lengths are comparable
        init = reparam(np.stack([xs / vp(ts)[0], logsnr(ts)], -1), 65)
        Y = geodesic(init[0], init[-1], N=65, iters=3000, init=init)
        Zg = z_of_y(Y)
        xg = np.interp(0.5, Zg[::-1, 1], Zg[::-1, 0])
        k_half = np.argmin(np.abs(ts - 0.5))
        Li, Lg = length_eta(eta_y(init)), length_eta(eta_y(Y))
        print(f'   x_T = {xT:+.0f} -> x_tmin = {xs[-1]:+.3f}: length PF-ODE {Li:.4f}, geodesic {Lg:.4f}'
              f' ({100 * (Lg / Li - 1):+.2f}%);  x at t = 0.5: PF-ODE {xs[k_half]:+.3f}, geodesic {xg:+.3f}')

    # ------------------------------------------------------------------ 9
    section('9. Gaussian data q = N(0, c^2): the spacetime is a hyperbolic strip')
    c2 = 1.0
    kw = dict(mu=np.array([0.0]), pi=np.array([1.0]), s2=c2)
    # map spacetime to (m, s) and back: posterior is N(m, s^2)
    for (ma, sa), (mb, sb) in [((-0.5, 0.2), (0.5, 0.2)), ((-1.0, 0.1), (1.0, 0.1)), ((-0.3, 0.05), (0.4, 0.3))]:
        # natural parameters of N(m, s^2) as a tilt of N(0, c^2): e2 = (1/c2 - 1/s^2)/2, e1 = m/s^2
        ea = np.array([ma / sa ** 2, 0.5 * (1 / c2 - 1 / sa ** 2)])
        eb = np.array([mb / sb ** 2, 0.5 * (1 / c2 - 1 / sb ** 2)])
        # geodesic in y = (xhat, ell) coordinates of the same family
        yA = np.array([ea[0] / (-2 * ea[1]), np.log(-2 * ea[1])])
        yB = np.array([eb[0] / (-2 * eb[1]), np.log(-2 * eb[1])])
        Y = geodesic(yA, yB, N=64, iters=8000, **kw)
        E = eta_y(Y)
        w, m, v, _ = tilt(E, **kw)
        # closed-form semi-ellipse through both points: (m - m0)^2 / 2 + s^2 = R^2
        m0 = ((ma ** 2 - mb ** 2) / 2 + sa ** 2 - sb ** 2) / (ma - mb)
        R = np.sqrt((ma - m0) ** 2 / 2 + sa ** 2)
        dev = np.abs((m[:, 0] - m0) ** 2 / 2 + v[:, 0] - R ** 2).max()
        print(f'  N({ma:+.2f}, {sa:.2f}^2) -> N({mb:+.2f}, {sb:.2f}^2):'
              f'  closed form {fisher_rao_normal(ma, sa, mb, sb):.4f}'
              f'  numerical geodesic {length_eta(E, **kw):.4f}'
              f'  apex s: predicted {R:.4f}, found {np.sqrt(v[:, 0].max()):.4f}'
              f'  (off the ellipse by {dev:.1e})')
    print('  DiffED between two points a distance d apart, both anchored at noise s,')
    print('  against the log law 2 sqrt2 log(d / (sqrt2 s)), valid once d >> s:')
    for s_ in [0.1, 0.03, 0.01]:
        row = []
        for d in [0.1, 0.3, 1.0]:
            row.append(f'd={d}: {fisher_rao_normal(0, s_, d, s_):6.3f} '
                       f'(log law {2 * np.sqrt(2) * np.log(d / (np.sqrt(2) * s_)):6.3f})')
        print(f'   s = {s_:4.2f}  ' + '   '.join(row))
    print('  apex of the geodesic between (0, s) and (d, s) for small s: d / (2 sqrt 2) = '
          + ', '.join(f'{d / (2 * np.sqrt(2)):.3f} (d={d})' for d in [0.1, 0.3, 1.0]))

    # ------------------------------------------------------------------ 10
    section('10. PF-ODE versus geodesic for standard Gaussian data (VP)')
    # For q = N(0, 1) under VP the PF-ODE keeps x_t constant, and p(x0|x_t) = N(alpha x, sigma^2).
    # In (m, s) that traces m^2/x^2 + s^2 = 1; a geodesic is (m - m0)^2/2 + s^2 = R^2.
    for x in [0.5, 1.0, np.sqrt(2), 2.0]:
        print(f'  x = {x:.3f}: PF-ODE ellipse semi-axes (m, s) = ({x:.3f}, 1); '
              f'geodesic ellipses need ratio sqrt 2 = 1.414; this one has {x:.3f}'
              + ('  -> the PF-ODE path is a geodesic' if abs(x - np.sqrt(2)) < 1e-9 else ''))

    # ------------------------------------------------------------------ 11
    section('11. Prop D.1: KL(gamma_s || z*) accumulated along a curve')
    s = np.linspace(0, 1, 4001)
    Z = np.stack([-2.3 + 4.3 * s, 0.35 + 0.05 * s + 0.2 * np.sin(np.pi * s)], -1)
    E = eta_of(Z[:, 0], Z[:, 1])
    M = mu_of_eta(E)
    estar = eta_of(0.0, 0.5)
    dM = np.gradient(M, s, axis=0)
    integ = (dM * (E - estar)).sum(-1)
    cum = np.concatenate([[0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(s))])
    for k in [1000, 2000, 4000]:
        print(f'  s = {s[k]:.2f}: KL(gamma_0||z*) + integral = {kl(E[0], estar) + cum[k]:.5f}'
              f'   direct KL(gamma_s||z*) = {kl(E[k], estar):.5f}')

    # ------------------------------------------------------------------ 12
    section("12. Hutchinson's trick in D = 256: energy unbiased, length barely biased")
    D = 256
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    lam = np.exp(rng.normal(0, 0.8, D))
    Sigma = (Q * lam) @ Q.T                       # q = N(0, Sigma), not axis-aligned

    def denoise_matrix(a, s):                     # E[x0 | x_t] = A x_t for Gaussian data
        return (Q * (a * lam / (a ** 2 * lam + s ** 2))) @ Q.T

    x_a, x_b = Q @ (np.sqrt(lam) * rng.normal(size=D)), Q @ (np.sqrt(lam) * rng.normal(size=D))
    Nc = 32
    ss = np.linspace(0, 1, Nc)
    tt = 0.3 + 0.2 * np.sin(np.pi * ss)           # a curve that rises in noise and comes back
    aa, sg = vp(tt)
    As = [denoise_matrix(aa[i], sg[i]) for i in range(Nc)]
    traces = np.array([np.trace(A) for A in As])

    for label, end in [('two different samples', x_b), ('one sample, noise up and back down', x_a)]:
        xt = [aa[i] * ((1 - u) * x_a + u * end) for i, u in enumerate(ss)]
        etas = np.array([np.concatenate([aa[i] * xt[i] / sg[i] ** 2, [-(aa[i] ** 2) / (2 * sg[i] ** 2)]])
                         for i in range(Nc)])
        means = np.array([As[i] @ xt[i] for i in range(Nc)])

        def chords_of(div):
            m2 = sg ** 2 / aa * div + (means ** 2).sum(-1)
            M = np.concatenate([means, m2[:, None]], -1)
            return ((etas[1:] - etas[:-1]) * (M[1:] - M[:-1])).sum(-1)

        ch = chords_of(traces)
        E_true, L_true = (Nc - 1) / 2 * ch.sum(), np.sqrt(ch).sum()
        print(f'  curve between {label}: exact energy {E_true:.2f}, exact length {L_true:.3f}')
        for probes in [1, 4, 16]:
            reps = 1000
            Es, Ls, neg = np.empty(reps), np.empty(reps), 0
            for r in range(reps):
                eps = rng.choice([-1.0, 1.0], size=(Nc, probes, D))
                div = np.einsum('npd,nde,npe->np', eps, np.array(As), eps).mean(-1)
                chh = chords_of(div)
                neg += (chh < 0).sum()
                Es[r] = (Nc - 1) / 2 * chh.sum()
                Ls[r] = np.sqrt(np.maximum(chh, 0)).sum()
            print(f'    {probes:2d} probe(s): energy bias {100 * (Es.mean() / E_true - 1):+6.2f}%'
                  f' (+- {100 * Es.std() / np.sqrt(reps) / E_true:.2f}%),'
                  f' length bias {100 * (Ls.mean() / L_true - 1):+6.2f}%'
                  f' (+- {100 * Ls.std() / np.sqrt(reps) / L_true:.2f}%),'
                  f' negative chords {100 * neg / (reps * (Nc - 1)):4.1f}%')

    # ------------------------------------------------------------------ 13
    section('13. Prop B.1: pullback geodesics decode to straight lines')
    Rm = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])

    def f(z):                      # a bijective, strongly nonlinear 2-D decoder
        w = z + 0.6 * np.sin(z[..., ::-1])
        return w @ Rm.T + 0.3 * w ** 3 / (1 + w ** 2)

    def pb_length(Zc):
        X = f(Zc)
        return np.linalg.norm(np.diff(X, axis=0), axis=-1).sum()

    za_, zb_ = np.array([-1.5, -0.5]), np.array([1.2, 1.4])
    s = np.linspace(0, 1, 2001)[:, None]
    straight_latent = (1 - s) * za_ + s * zb_
    # preimage of the straight data segment, by Newton from the latent line
    target = (1 - s) * f(za_) + s * f(zb_)
    Zp = straight_latent.copy()
    for _ in range(50):
        h = 1e-6
        J = np.stack([(f(Zp + h * e) - f(Zp - h * e)) / (2 * h) for e in np.eye(2)], -1)
        Zp -= np.linalg.solve(J, (f(Zp) - target)[..., None])[..., 0]
    print(f'  |x_b - x_a| = {np.linalg.norm(f(zb_) - f(za_)):.5f}')
    print(f'  pullback length of the straight latent line   {pb_length(straight_latent):.5f}')
    print(f'  pullback length of the preimage of the chord  {pb_length(Zp):.5f}'
          f'  (latent curve bends: max distance from the latent line {np.abs(Zp - straight_latent).max():.3f})')

    # ------------------------------------------------------------------ 14
    section('14. Sec 6.3 on a 2-D toy: two wells, a barrier, a channel over the top')
    A, B = np.array([-1.8, -0.6]), np.array([1.8, -0.6])
    C, W, V = [A, B], [0.2, 0.2], [0.05, 0.05]
    for k in range(11):
        ang = np.pi * (150 - 12 * k) / 180
        C.append([1.9 * np.cos(ang), -0.6 + 1.9 * np.sin(ang)]); W.append(0.6 / 11); V.append(0.1)
    fam = Mixture(C, W, V)
    U = lambda X: -fam.logq(X)
    # lower bound: the lowest level at which A and B are connected in {U <= level} (flood fill)
    GN = 220
    gx, gy = np.linspace(-3.2, 3.2, GN), np.linspace(-2.2, 2.2, GN)
    UG = U(np.stack(np.meshgrid(gx, gy), -1))
    ia, ib = (np.abs(gy - A[1]).argmin(), np.abs(gx - A[0]).argmin()), (np.abs(gy - B[1]).argmin(), np.abs(gx - B[0]).argmin())

    def connected(level):
        ok, seen, stack = UG <= level, np.zeros(UG.shape, bool), [ia]
        seen[ia] = True
        while stack:
            i, j = stack.pop()
            if (i, j) == ib:
                return True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < GN and 0 <= b < GN and ok[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        return False

    lo, hi = float(U(A)), float(UG.max())
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if connected(mid) else (mid, hi)
    print(f'  U at the wells {float(U(A)):.2f}, at the midpoint of the straight line {float(U(np.array([0, -0.6]))):.2f},'
          f' lowest possible barrier (flood fill) {hi:.2f}')
    ELL = 4.0
    Y = geodesicD(fam, np.r_[A, ELL], np.r_[B, ELL])
    M, _ = fam.stats(eta_yD(Y))
    print(f'  geodesic: length {np.sqrt(np.maximum(((eta_yD(Y)[1:] - eta_yD(Y)[:-1]) * (M[1:] - M[:-1])).sum(-1), 0)).sum():.3f},'
          f' peak noise SNR^-1/2 = {np.exp(-Y[:, 2].min() / 2):.3f}')
    print(f'  posterior mean E[x0|gamma_s]: highest y = {M[:, 1].max():+.3f} (the wells sit at y = -0.6);'
          f' x_t/alpha reaches y = {Y[:, 1].min():+.2f}')
    w_mid, _, _ = fam.tilt(eta_yD(Y[len(Y) // 2]))
    print(f'  posterior at s = 0.5: mass on the two wells {w_mid[:2].sum():.3f}, on the channel {w_mid[2:].sum():.3f}')
    straight = np.array([np.r_[(1 - u) * A + u * B, ELL] for u in np.linspace(0, 1, len(Y))])
    rng2 = np.random.default_rng(1)

    def annealed_langevin(Yc, K, dt=0.003, n=32):
        """Algorithm 1: K Langevin steps on p(x | gamma_n) per point, carrying the state along."""
        x = np.tile(A, (n, 1))
        top, ycross = U(x), np.full(n, np.nan)       # ycross: height where x1 first passes 0
        for e in eta_yD(Yc):
            for _ in range(K):
                xn = x + (fam.grad_logq(x) + e[:2] + 2 * e[2] * x) * dt + np.sqrt(2 * dt) * rng2.normal(size=x.shape)
                first = np.isnan(ycross) & (x[:, 0] < 0) & (xn[:, 0] >= 0)
                ycross[first] = 0.5 * (x[first, 1] + xn[first, 1])
                x = xn
                top = np.maximum(top, U(x))
        return top, ycross

    # ablation: the straight line in x_t / alpha, but with the geodesic's own noise schedule
    same_noise = straight.copy()
    same_noise[:, 2] = Y[:, 2]
    for K in [40, 120]:
        mg, yg = annealed_langevin(Y, K)
        mb, yb = annealed_langevin(straight, K)
        ms, ys_ = annealed_langevin(same_noise, K)
        print(f'  K = {K:3d} Langevin steps per point, 32 paths, MaxEnergy (share crossing x = 0 in the channel):')
        print(f'      spacetime geodesic                         {mg.mean():5.2f} +- {mg.std():.2f}  ({np.mean(yg > 0.3):.0%})')
        print(f'      straight line, fixed log-SNR 4             {mb.mean():5.2f} +- {mb.std():.2f}  ({np.mean(yb > 0.3):.0%})')
        print(f'      straight line, geodesic noise schedule     {ms.mean():5.2f} +- {ms.std():.2f}  ({np.mean(ys_ > 0.3):.0%})')


if __name__ == '__main__':
    main()
