#!/usr/bin/env python3
"""Fisher information, checked by hand: every number quoted in the foundations notes.

Small exact computations and simulations on one- and two-parameter models,
with nothing trained beyond a logistic regression.

Checked here, in the order the notes use them:

  1. a coin: the score has mean zero, E[score^2] = -E[d^2 log p] = 1/(theta(1-theta)),
     and N Var(MLE) approaches 1/I;
  2. Cramer-Rao: the sample mean of Gaussian data attains the bound, the sample
     median has efficiency 2/pi;
  3. two parameters: the Fisher matrix of N(mu, sigma^2) is diag(1/s^2, 2/s^2), the
     MLE's covariance is F^-1 / N, and a redundant parameterisation (a b x) has a
     singular Fisher matrix;
  4. the local KL: KL(p_theta || p_theta+d) / (d^T F d) -> 1/2;
  5. reparameterisation: the Fisher information changes by the Jacobian squared,
     and the Fisher-Rao length of a path does not; Bernoulli's closed form
     2 |arcsin sqrt(t1) - arcsin sqrt(t2)|;
  6. Gaussians form a hyperbolic half-plane: the closed-form Fisher-Rao distance,
     the semicircle geodesic, and why the path between two means widens first;
  7. natural gradient: for a coin, one step of size 1 lands on the MLE in any
     parameterisation, while plain gradient descent needs many and depends on
     the parameterisation; the same on a Gaussian fit;
  8. Fisher, Hessian and empirical Fisher: equal for a model linear in its
     parameters at the optimum, and how far the empirical Fisher strays away from it;
  9. the Fisher and the neural tangent kernel share their non-zero eigenvalues;
 10. Monte Carlo estimates of the Fisher.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only.

Run:  python3 fisher.py            (checks)
      python3 fisher.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


def trapz(y, x):
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2)


# ------------------------------------------------------------------ 1. a coin

def bern_I(t):
    return 1 / (t * (1 - t))


def check_coin():
    print("1. A coin: the Fisher information of Bernoulli(theta)")
    for t in (0.5, 0.2, 0.1, 0.01):
        # exact expectations over x in {0, 1}
        s1, s0 = 1 / t, -1 / (1 - t)                   # score d/dtheta log p at x = 1 and x = 0
        mean = t * s1 + (1 - t) * s0
        sq = t * s1 ** 2 + (1 - t) * s0 ** 2
        curv = -(t * (-1 / t ** 2) + (1 - t) * (-1 / (1 - t) ** 2))
        print(f"   theta = {t:4.2f}: E[score] = {mean:+.1e}; E[score^2] = {sq:.3f}; -E[d^2 log p] = {curv:.3f};"
              f" 1/(theta(1-theta)) = {bern_I(t):.3f}; one-flip standard error 1/sqrt(I) = {1 / math.sqrt(bern_I(t)):.3f}")
    rng = np.random.default_rng(1)
    rows = []
    for t in (0.5, 0.2):
        for N in (10, 50, 500):
            est = rng.binomial(N, t, size=200000) / N
            rows.append((t, N, N * est.var() * bern_I(t)))
            print(f"   theta = {t}, N = {N:3d}: N Var(MLE) x I(theta) = {N * est.var() * bern_I(t):.3f}")
    return rows


# ------------------------------------------------------------------ 2. Cramer-Rao

def check_cramer_rao():
    print("\n2. Cramer-Rao: the best possible spread of an unbiased estimator is 1/(N I)")
    rng = np.random.default_rng(2)
    N, reps, sigma = 101, 40000, 1.0
    X = rng.normal(0.0, sigma, size=(reps, N))
    mean, med = X.mean(axis=1), np.median(X, axis=1)
    bound = sigma ** 2 / N
    print(f"   Gaussian mean, sigma = 1, N = {N}: Cramer-Rao bound sigma^2/N = {bound:.5f}")
    print(f"   sample mean: variance {mean.var():.5f} (ratio to the bound {mean.var() / bound:.3f})")
    print(f"   sample median: variance {med.var():.5f} (ratio {med.var() / bound:.3f}; theory pi/2 = {math.pi / 2:.3f},"
          f" efficiency {bound / med.var():.3f} against 2/pi = {2 / math.pi:.3f})")
    return mean, med, bound


# ------------------------------------------------------------------ 3. two parameters

def check_two_params():
    print("\n3. Two parameters: the Fisher matrix and the uncertainty ellipse")
    rng = np.random.default_rng(3)
    mu, s = 1.0, 2.0
    x = rng.normal(mu, s, size=400000)
    g = np.stack([(x - mu) / s ** 2, -1 / s + (x - mu) ** 2 / s ** 3], axis=1)   # score in (mu, sigma)
    F = g.T @ g / x.size
    print(f"   N(mu = {mu}, sigma = {s}): Monte Carlo Fisher [[{F[0, 0]:.4f}, {F[0, 1]:+.4f}], [{F[1, 0]:+.4f}, {F[1, 1]:.4f}]];"
          f" exact diag(1/sigma^2, 2/sigma^2) = diag({1 / s ** 2:.4f}, {2 / s ** 2:.4f})")
    N, reps = 50, 40000
    X = rng.normal(mu, s, size=(reps, N))
    m_hat, s_hat = X.mean(axis=1), X.std(axis=1)
    C = np.cov(np.stack([m_hat, s_hat]))
    print(f"   MLE over {reps} datasets of N = {N}: N Cov = [[{N * C[0, 0]:.3f}, {N * C[0, 1]:+.3f}], [{N * C[1, 0]:+.3f}, {N * C[1, 1]:.3f}]];"
          f" F^-1 = diag({s ** 2:.3f}, {s ** 2 / 2:.3f})")
    # redundant parameterisation: logistic regression with logit a * b * x
    xs = rng.normal(size=2000)
    a, b = 1.5, 0.8
    p = 1 / (1 + np.exp(-a * b * xs))
    J = np.stack([b * xs, a * xs], axis=1)
    Fr = (J * (p * (1 - p))[:, None]).T @ J / xs.size
    ev = np.linalg.eigvalsh(Fr)
    print(f"   logit a*b*x at a = {a}, b = {b}: Fisher eigenvalues {ev[0]:.1e} and {ev[1]:.4f}: singular,"
          f" because only the product ab is visible; null direction (a, -b) = ({a}, {-b})")
    return (mu, s, N, m_hat, s_hat)


# ------------------------------------------------------------------ 4. local KL

def kl_gauss(m1, s1, m2, s2):
    return math.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5


def check_local_kl():
    print("\n4. The Fisher matrix is the local shape of the KL divergence")
    mu, s = 0.0, 1.5
    F = np.diag([1 / s ** 2, 2 / s ** 2])
    rng = np.random.default_rng(4)
    v = rng.normal(size=2)
    v /= np.linalg.norm(v)
    for eps in (0.3, 0.1, 0.01, 0.001):
        d = eps * v
        kl = kl_gauss(mu, s, mu + d[0], s + d[1])
        print(f"   |d| = {eps:5.3f}: KL(p || p_(theta+d)) / (d^T F d) = {kl / (d @ F @ d):.4f}")
    for t, dt in ((0.5, 0.01), (0.01, 0.001)):
        kl = t * math.log(t / (t + dt)) + (1 - t) * math.log((1 - t) / (1 - t - dt))
        print(f"   coin at theta = {t}, step {dt}: KL = {kl:.3e}, (1/2) I dt^2 = {0.5 * bern_I(t) * dt ** 2:.3e}")


# ------------------------------------------------------------------ 5. reparameterisation

def fr_bern(t1, t2):
    return 2 * abs(math.asin(math.sqrt(t1)) - math.asin(math.sqrt(t2)))


def check_reparam():
    print("\n5. Change the parameterisation: the numbers change, the geometry does not")
    t = 0.2
    eta = math.log(t / (1 - t))
    dth = t * (1 - t)                                   # d theta / d eta
    print(f"   coin at theta = {t} (log-odds eta = {eta:.3f}): I_theta = {bern_I(t):.3f};"
          f" I_eta = (dtheta/deta)^2 I_theta = {dth ** 2 * bern_I(t):.4f} = theta(1 - theta) = {t * (1 - t):.4f}")
    t1, t2 = 0.05, 0.5
    th = np.linspace(t1, t2, 200001)
    L_th = trapz(np.sqrt(bern_I(th)), th)
    e = np.linspace(math.log(t1 / (1 - t1)), math.log(t2 / (1 - t2)), 200001)
    te = 1 / (1 + np.exp(-e))
    L_eta = trapz(np.sqrt(te * (1 - te)), e)
    print(f"   from theta = {t1} to {t2}: Fisher-Rao length integrated in theta {L_th:.6f}, in eta {L_eta:.6f},"
          f" closed form 2|arcsin sqrt t1 - arcsin sqrt t2| = {fr_bern(t1, t2):.6f}")
    print(f"   plain distances disagree: |theta difference| = {t2 - t1:.3f}, |eta difference| = {abs(e[-1] - e[0]):.3f}")
    for a, b in ((0.5, 0.6), (0.01, 0.11)):
        print(f"   theta {a} -> {b} (a step of 0.1 either way): Fisher-Rao distance {fr_bern(a, b):.3f}")


# ------------------------------------------------------------------ 6. Gaussians as a hyperbolic half-plane

def fr_gauss(m1, s1, m2, s2):
    """Fisher-Rao distance between N(m1, s1^2) and N(m2, s2^2): sqrt(2) times the Poincare half-plane distance
    in coordinates (mu / sqrt 2, sigma)."""
    return math.sqrt(2) * math.acosh(1 + ((m1 - m2) ** 2 / 2 + (s1 - s2) ** 2) / (2 * s1 * s2))


def path_length(mus, sigmas):
    dm, ds = np.diff(mus), np.diff(sigmas)
    sm = (sigmas[1:] + sigmas[:-1]) / 2
    return float(np.sum(np.sqrt(dm ** 2 + 2 * ds ** 2) / sm))


def geodesic(m1, m2, s, n=4001):
    """For two Gaussians with equal sigma: the semicircle centred on the axis in (mu/sqrt2, sigma) coordinates."""
    u1, u2 = m1 / math.sqrt(2), m2 / math.sqrt(2)
    c = (u1 + u2) / 2
    r = math.hypot(u1 - c, s)
    p0 = math.atan2(s, u1 - c)
    ph = np.linspace(p0, math.pi - p0, n)[::-1] if u1 < u2 else np.linspace(p0, math.pi - p0, n)
    return math.sqrt(2) * (c + r * np.cos(ph)), r * np.sin(ph)


def check_gauss_geometry():
    print("\n6. Gaussians: the Fisher metric turns (mu, sigma) into a hyperbolic half-plane")
    m1, m2, s = -2.0, 2.0, 1.0
    straight = path_length(np.linspace(m1, m2, 4001), np.full(4001, s))
    gm, gs = geodesic(m1, m2, s)
    print(f"   N(-2, 1) to N(2, 1): closed-form Fisher-Rao distance {fr_gauss(m1, s, m2, s):.4f};"
          f" length of the semicircle path {path_length(gm, gs):.4f}; of the straight path at sigma = 1 {straight:.4f}")
    print(f"   the geodesic's midpoint is N(0, {gs.max():.3f}^2): the shortest route widens the Gaussian"
          f" (to sqrt(3) = {math.sqrt(3):.3f}) before moving it")
    rng = np.random.default_rng(6)
    worst = min(path_length(gm, gs + a * np.sin(np.linspace(0, math.pi, gs.size)) * np.sin(k * np.linspace(0, math.pi, gs.size)))
                for a, k in zip(rng.uniform(-0.3, 0.3, 200), rng.integers(1, 5, 200)))
    print(f"   200 random wiggles of the semicircle: every one is longer; the shortest by {worst - fr_gauss(m1, s, m2, s):.1e}")
    print(f"   compare KL(N(-2,1) || N(2,1)) = {kl_gauss(m1, s, m2, s):.3f}; for mean gaps of 0.1, 1, 4 at sigma = 1 the"
          f" Fisher-Rao distances are {fr_gauss(0, 1, 0.1, 1):.4f}, {fr_gauss(0, 1, 1, 1):.4f}, {fr_gauss(0, 1, 4, 1):.4f}")
    print(f"   the same mean gap of 1 at sigma = 0.1: {fr_gauss(0, 0.1, 1, 0.1):.3f}; at sigma = 10: {fr_gauss(0, 10, 1, 10):.4f}")
    w2 = math.hypot(m1 - m2, s - s)                     # 2-Wasserstein distance between 1-D Gaussians
    print(f"   optimal transport instead: W2(N(-2,1), N(2,1)) = sqrt((mu1-mu2)^2 + (sigma1-sigma2)^2) = {w2:.3f},"
          f" and its geodesic keeps sigma = 1 all the way")
    return gm, gs


# ------------------------------------------------------------------ 7. natural gradient

def check_natural_gradient():
    print("\n7. Natural gradient: steepest descent measured in KL, not in parameters")
    xbar, t0 = 0.8, 0.1                                 # fraction of heads in the data; starting guess
    # natural gradient on the average log-likelihood, step 1, in theta: theta + I^-1 * score
    score = xbar / t0 - (1 - xbar) / (1 - t0)
    print(f"   coin with 80% heads, start theta = {t0}: one natural-gradient step of size 1 gives {t0 + score / bern_I(t0):.4f}")
    e0 = math.log(t0 / (1 - t0))
    th0 = 1 / (1 + math.exp(-e0))
    e1 = e0 + (xbar - th0) / (th0 * (1 - th0))          # in log-odds: gradient xbar - theta, Fisher theta(1-theta)
    print(f"   the same step taken in log-odds coordinates lands on theta = {1 / (1 + math.exp(-e1)):.4f}"
          f" (not exactly the MLE: the step is exact only to first order, the direction is what is invariant)")

    def steps(update, x0, target, tol=1e-3, cap=100000):
        x = x0
        for k in range(1, cap):
            x = update(x)
            if abs(target(x) - xbar) < tol:
                return k
        return cap
    lr = 0.02
    n_th = steps(lambda t: min(max(t + lr * (xbar / t - (1 - xbar) / (1 - t)), 1e-6), 1 - 1e-6), t0, lambda t: t)
    n_eta = steps(lambda e: e + lr * (xbar - 1 / (1 + math.exp(-e))), e0, lambda e: 1 / (1 + math.exp(-e)))
    n_ng = steps(lambda t: t + 0.5 * (xbar / t - (1 - xbar) / (1 - t)) / bern_I(t), t0, lambda t: t)
    print(f"   steps to reach |theta - 0.8| < 0.001 with learning rate {lr}: plain gradient in theta {n_th},"
          f" plain gradient in log-odds {n_eta}; natural gradient with step 0.5: {n_ng}")
    # Gaussian fit: trajectories for the figure
    rng = np.random.default_rng(7)
    data = rng.normal(2.0, 0.5, size=200)
    m_hat, s_hat = data.mean(), data.std()

    def grad(m, s):                                     # gradient of the average negative log-likelihood
        return np.array([-(m_hat - m) / s ** 2, 1 / s - (s_hat ** 2 + (m_hat - m) ** 2) / s ** 3])

    start = np.array([-1.0, 3.0])
    gd, ng = [start.copy()], [start.copy()]
    p = start.copy()
    for _ in range(4000):
        p = p - 0.05 * grad(*p)
        p[1] = max(p[1], 0.05)
        gd.append(p.copy())
    p = start.copy()
    for _ in range(4000):
        F = np.diag([1 / p[1] ** 2, 2 / p[1] ** 2])
        p = p - 0.05 * np.linalg.solve(F, grad(*p))
        ng.append(p.copy())
    gd, ng = np.array(gd), np.array(ng)
    # the same two methods in coordinates (mu, log sigma), mapped back to (mu, sigma)
    def grad_log(m, r):                                 # r = log sigma
        g = grad(m, math.exp(r))
        return np.array([g[0], g[1] * math.exp(r)])
    q = np.array([start[0], math.log(start[1])])
    gd2 = [q.copy()]
    for _ in range(4000):
        q = q - 0.05 * grad_log(*q)
        gd2.append(q.copy())
    q = np.array([start[0], math.log(start[1])])
    ng2 = [q.copy()]
    for _ in range(4000):
        F = np.diag([math.exp(-2 * q[1]), 2.0])         # Fisher in (mu, log sigma)
        q = q - 0.05 * np.linalg.solve(F, grad_log(*q))
        ng2.append(q.copy())
    gd2, ng2 = np.array(gd2), np.array(ng2)
    gd2[:, 1], ng2[:, 1] = np.exp(gd2[:, 1]), np.exp(ng2[:, 1])

    def gap(a, b, k=60):
        return float(np.max(np.hypot(a[:k, 0] - b[:k, 0], a[:k, 1] - b[:k, 1])))
    print(f"   Gaussian fit to 200 points (mean {m_hat:.3f}, sd {s_hat:.3f}) from (mu, sigma) = (-1, 3), step 0.05:")
    print(f"     plain gradient run in (mu, sigma) and in (mu, log sigma): paths up to {gap(gd, gd2):.2f} apart over the first 60 steps")
    print(f"     natural gradient run in the two coordinate systems: paths up to {gap(ng, ng2):.3f} apart (they agree to first order in the step)")
    print(f"     after 60 steps: plain gradient at ({gd[60, 0]:.2f}, {gd[60, 1]:.2f}), natural at ({ng[60, 0]:.2f}, {ng[60, 1]:.2f})")
    return gd, ng, (m_hat, s_hat), gd2


# ------------------------------------------------------------------ 8. Fisher, Hessian, empirical Fisher

def check_hessian_empirical():
    print("\n8. Fisher, Hessian and the empirical Fisher")
    rng = np.random.default_rng(8)
    N, d, sig = 5000, 3, 0.5
    X = rng.normal(size=(N, d))
    w_true = np.array([1.0, -2.0, 0.5])
    y = X @ w_true + sig * rng.normal(size=N)
    F = X.T @ X / (N * sig ** 2)                         # Gaussian likelihood: Fisher = Hessian of the NLL, at every w
    w_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    for name, w in (("at the least-squares fit", w_hat), ("at w = 0", np.zeros(d)), ("at w = 3 w_true", 3 * w_true)):
        r = y - X @ w
        EF = (X * (r ** 2)[:, None]).T @ X / (N * sig ** 4)   # mean of g g^T with the observed labels
        print(f"   linear regression, {name}: largest eigenvalue of the empirical Fisher / of the Fisher ="
              f" {np.linalg.eigvalsh(EF)[-1] / np.linalg.eigvalsh(F)[-1]:.2f}")
    # logistic regression: H = F exactly; the empirical Fisher at the optimum of a misspecified model
    x = rng.normal(size=4000)
    yb = (rng.random(4000) < 1 / (1 + np.exp(-3 * np.tanh(2 * x)))).astype(float)   # true logit is not linear
    A = np.stack([np.ones_like(x), x], axis=1)
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ w))
        Fm = (A * (p * (1 - p))[:, None]).T @ A / x.size
        w = w + np.linalg.solve(Fm, A.T @ (yb - p) / x.size)
    p = 1 / (1 + np.exp(-A @ w))
    Fm = (A * (p * (1 - p))[:, None]).T @ A / x.size
    EF = (A * ((yb - p) ** 2)[:, None]).T @ A / x.size
    print(f"   logistic regression on data whose true logit is 3 tanh(2x): at the maximum likelihood fit the Fisher"
          f" (= Hessian) has eigenvalues {np.round(np.linalg.eigvalsh(Fm), 4).tolist()}, the empirical Fisher"
          f" {np.round(np.linalg.eigvalsh(EF), 4).tolist()}")


# ------------------------------------------------------------------ 9. Fisher and NTK

def check_ntk():
    print("\n9. The Fisher and the neural tangent kernel are two views of one matrix")
    rng = np.random.default_rng(9)
    N, P = 20, 50
    J = rng.normal(size=(N, P))                          # d logit_i / d w, one row per example
    lam = rng.uniform(0.05, 0.25, size=N)                # p_i (1 - p_i)
    F = (J * lam[:, None]).T @ J / N                     # P x P, parameter space
    K = (np.sqrt(lam)[:, None] * J) @ (np.sqrt(lam)[:, None] * J).T / N   # N x N, sample space
    ef = np.sort(np.linalg.eigvalsh(F))[::-1][:N]
    ek = np.sort(np.linalg.eigvalsh(K))[::-1]
    print(f"   {N} examples, {P} weights: the Fisher (P x P) has {np.sum(np.linalg.eigvalsh(F) > 1e-10)} non-zero eigenvalues;"
          f" max difference from the weighted kernel's (N x N) eigenvalues {np.abs(ef - ek).max():.1e}")


# ------------------------------------------------------------------ 10. Monte Carlo

def check_monte_carlo():
    print("\n10. Estimating the Fisher: exact, sampled labels, observed labels")
    rng = np.random.default_rng(10)
    x = rng.normal(size=(300, 2))
    w = np.array([1.0, -1.0])
    p = 1 / (1 + np.exp(-x @ w))
    F = (x * (p * (1 - p))[:, None]).T @ x / x.shape[0]
    for S in (1, 10, 100):
        errs = []
        for _ in range(200):
            ys = rng.random((S, x.shape[0])) < p          # labels drawn from the model
            g = (ys - p)[:, :, None] * x[None, :, :]      # score for each sample and example
            Fs = np.einsum("sni,snj->ij", g, g) / (S * x.shape[0])
            errs.append(np.linalg.norm(Fs - F) / np.linalg.norm(F))
        print(f"   {S:3d} sampled label(s) per example: relative error {np.mean(errs):.3f}")


# ------------------------------------------------------------------ figures

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:10.5px;fill:#57564f} .v{font-size:11px;fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s4{stroke:#eda100} .s0{stroke:#8a8880}
  .f1{fill:#2a78d6} .f2{fill:#eb6834} .f3{fill:#1baf7a} .f4{fill:#eda100} .f0{fill:#8a8880}
  .ln{stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
  .thin{stroke-width:1;fill:none}
  .dash{stroke-width:1.4;fill:none;stroke-dasharray:5 3}
  .bar{opacity:.35} .pt{opacity:.35}
  .ring{stroke:#fdfdfc;stroke-width:1.5}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500} .s0{stroke:#85837b}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500} .f0{fill:#85837b}
    .ring{stroke:#161615}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


def fmt(v):
    return f"{v:g}".replace("-", "−")


class Panel:
    def __init__(self, body, x0, y0, w, h, xr, yr):
        self.b, self.x0, self.y0, self.w, self.h, self.xr, self.yr = body, x0, y0, w, h, xr, yr

    def X(self, x):
        return self.x0 + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def Y(self, y):
        return self.y0 + self.h - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def inside(self, x, y):
        return self.xr[0] - 1e-9 <= x <= self.xr[1] + 1e-9 and self.yr[0] - 1e-9 <= y <= self.yr[1] + 1e-9

    def frame(self, xt, yt, xlab, ylab, title):
        b = self.b
        for y in yt:
            b.append(f'<line class="gd" x1="{self.x0}" y1="{self.Y(y):.1f}" x2="{self.x0 + self.w}" y2="{self.Y(y):.1f}"/>')
            b.append(f'<text class="sm" x="{self.x0 - 6}" y="{self.Y(y) + 3.5:.1f}" text-anchor="end">{fmt(y)}</text>')
        for x in xt:
            b.append(f'<text class="sm" x="{self.X(x):.1f}" y="{self.y0 + self.h + 15}" text-anchor="middle">{fmt(x)}</text>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0 + self.h}" x2="{self.x0 + self.w}" y2="{self.y0 + self.h}"/>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0}" x2="{self.x0}" y2="{self.y0 + self.h}"/>')
        b.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 32}" text-anchor="middle">{xlab}</text>')
        b.append(f'<text class="lab" transform="translate({self.x0 - 40},{self.y0 + self.h / 2}) rotate(-90)" text-anchor="middle">{ylab}</text>')
        b.append(f'<text class="hd" x="{self.x0}" y="{self.y0 - 12}">{title}</text>')

    def line(self, xs, ys, cls):
        segs, cur = [], []
        for x, y in zip(xs, ys):
            if np.isfinite(y) and self.inside(x, y):
                cur.append(f"{self.X(x):.1f},{self.Y(y):.1f}")
            elif cur:
                segs.append(cur); cur = []
        if cur:
            segs.append(cur)
        for s in segs:
            if len(s) > 1:
                self.b.append(f'<polyline class="{cls}" points="{" ".join(s)}"/>')

    def dot(self, x, y, cls, r=4.5):
        self.b.append(f'<circle class="{cls} ring" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>')

    def pts(self, xs, ys, cls, r=1.4):
        for x, y in zip(xs, ys):
            if self.inside(x, y):
                self.b.append(f'<circle class="{cls} pt" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>')

    def bars(self, edges, vals, cls):
        for a, c, v in zip(edges[:-1], edges[1:], vals):
            v = min(v, self.yr[1])
            self.b.append(f'<rect class="{cls}" x="{self.X(a) + 0.4:.1f}" y="{self.Y(v):.1f}" width="{max(self.X(c) - self.X(a) - 0.8, 0.5):.1f}" height="{self.Y(0) - self.Y(v):.1f}"/>')

    def text(self, x, y, s, cls="v", anchor="start", dx=0, dy=0):
        self.b.append(f'<text class="{cls}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" text-anchor="{anchor}">{s}</text>')


def fig_coin(out, rows):
    W, H = 920, 320
    body = []
    L = Panel(body, 70, 45, 330, 220, (0, 1), (0, 1.6))
    L.frame([0, 0.25, 0.5, 0.75, 1], [0, 0.5, 1, 1.5], "θ (true probability of heads)", "average log-likelihood, shifted",
            "Sharp peak = much information")
    th = np.linspace(0.001, 0.999, 999)
    for t0, cls in ((0.5, "s1"), (0.1, "s2")):
        ll = t0 * np.log(th) + (1 - t0) * np.log(1 - th)
        ll0 = t0 * math.log(t0) + (1 - t0) * math.log(1 - t0)
        L.line(th, ll - ll0 + 1.5, f"ln {cls}")
        L.line(th, 1.5 - 0.5 * bern_I(t0) * (th - t0) ** 2, f"dash {cls}")
        L.text(t0, 1.5, f"θ = {t0}: I = {bern_I(t0):.1f}", "sm", "start" if t0 < 0.3 else "middle", 6 if t0 < 0.3 else 0, -6 if t0 > 0.3 else 18)
    R = Panel(body, 530, 45, 350, 220, (0, 1), (0, 16))
    R.frame([0, 0.25, 0.5, 0.75, 1], [0, 4, 8, 12, 16], "θ", "Fisher information I(θ)", "I(θ) = 1/(θ(1 − θ)), and simulated 1/(N Var)")
    R.line(th, bern_I(th), "ln s1")
    for t, N, r in rows:
        R.dot(t, bern_I(t) / r, "f2" if N == 500 else "f0", 4)
    R.text(0.5, 4, "minimum 4 at θ = ½", "sm", "middle", 0, 16)
    desc = ("Left: the average log-likelihood of a coin as a function of the guessed probability, for data with true "
            "probability 0.5 (blue) and 0.1 (orange), shifted to a common peak, with the parabola of curvature I(θ) "
            "dashed. The θ = 0.1 curve is much sharper: 11.1 against 4. Right: I(θ) = 1/(θ(1 − θ)), smallest at a fair "
            "coin, with dots at 1/(N Var(MLE)) from simulations at θ = 0.5 and 0.2, which sit on the curve.")
    out.write_text(svg(W, H, "Fisher information of a coin", desc, body))


def fig_efficiency(out, mean, med, bound):
    W, H = 920, 300
    body = []
    P = Panel(body, 90, 45, 780, 200, (-0.4, 0.4), (0, 5.2))
    P.frame([-0.4, -0.2, 0, 0.2, 0.4], [0, 1, 2, 3, 4, 5], "estimate of μ (true value 0)", "density",
            "Two estimators of a Gaussian mean, N = 101")
    edges = np.linspace(-0.4, 0.4, 81)
    hm, _ = np.histogram(mean, bins=edges, density=True)
    hd, _ = np.histogram(med, bins=edges, density=True)
    P.bars(edges, hm, "f1 bar")
    P.bars(edges, hd, "f2 bar")
    x = np.linspace(-0.4, 0.4, 400)
    P.line(x, np.exp(-x ** 2 / (2 * bound)) / math.sqrt(2 * math.pi * bound), "ln s1")
    P.text(0.05, 4.0, f"sample mean: variance = the Cramér–Rao bound 1/(N I) (curve)", "sm", "start", 0, 0)
    P.text(0.12, 2.6, f"sample median: {med.var() / bound:.2f}× the bound (≈ π/2)", "sm", "start", 0, 0)
    desc = ("Histograms of 40000 sample means (blue) and sample medians (orange) of 101 standard normal draws. The mean's "
            "spread matches the Cramér–Rao curve 1/(N I) exactly; the median's is about π/2 times wider, an efficiency of "
            "2/π.")
    out.write_text(svg(W, H, "Cramér–Rao and efficiency", desc, body))


def fig_ellipse(out, two):
    mu, s, N, m_hat, s_hat = two
    W, H = 920, 320
    body = []
    L = Panel(body, 70, 45, 330, 220, (mu - 1.2, mu + 1.2), (s - 0.8, s + 0.8))
    L.frame([0, 0.5, 1, 1.5, 2], [1.4, 1.8, 2.2, 2.6], "estimated μ", "estimated σ", f"MLE of N(1, 2²) from N = {N}")
    L.pts(m_hat[:1500], s_hat[:1500], "f1")
    t = np.linspace(0, 2 * math.pi, 300)
    L.line(mu + 2 * math.sqrt(s ** 2 / N) * np.cos(t), s + 2 * math.sqrt(s ** 2 / 2 / N) * np.sin(t), "ln s2")
    L.text(mu + 0.9, s + 0.62, "2-sd ellipse of F⁻¹/N", "sm", "end", 0, 0)
    R = Panel(body, 530, 45, 350, 220, (0, 3), (0, 2))
    R.frame([0, 1, 2, 3], [0, 0.5, 1, 1.5, 2], "a", "b", "Redundant: logit = a·b·x")
    a = np.linspace(0.1, 3, 300)
    R.line(a, 1.2 / a, "ln s0")
    R.dot(1.5, 0.8, "f2", 5)
    R.line([1.5 - 0.6, 1.5 + 0.6], [0.8 + 0.6 * 0.8 / 1.5, 0.8 - 0.6 * 0.8 / 1.5], "ln s2")
    R.text(0.15, 0.2, "every (a, b) on the curve ab = 1.2 makes the same predictions", "sm", "start", 0, 0)
    R.text(1.5, 0.8, "the Fisher's null direction", "sm", "start", 12, -12)
    desc = ("Left: 1500 maximum-likelihood estimates of (μ, σ) for a normal with μ = 1 and σ = 2, each from 50 draws, "
            "inside the two-standard-deviation ellipse predicted by the inverse Fisher matrix divided by N. Right: for a "
            "logistic model whose logit is a·b·x, every (a, b) on the curve ab = 1.2 gives the same predictions, and the "
            "Fisher matrix at (1.5, 0.8) is singular along the curve's tangent: the data cannot see that direction.")
    out.write_text(svg(W, H, "The Fisher ellipse, and a blind direction", desc, body))


def fig_metric(out):
    W, H = 920, 250
    body = []
    L = Panel(body, 60, 55, 800, 40, (0, 1), (0, 1))
    body.append('<text class="hd" x="60" y="30">Ten equal Fisher–Rao steps from θ = 0.01 to θ = 0.99</text>')
    body.append(f'<line class="ax" x1="{L.X(0)}" y1="{L.Y(0.5)}" x2="{L.X(1)}" y2="{L.Y(0.5)}"/>')
    a0, a1 = math.asin(math.sqrt(0.01)), math.asin(math.sqrt(0.99))
    for k in range(11):
        th = math.sin(a0 + (a1 - a0) * k / 10) ** 2
        body.append(f'<circle class="f1 ring" cx="{L.X(th):.1f}" cy="{L.Y(0.5):.1f}" r="5"/>')
        body.append(f'<text class="sm" x="{L.X(th):.1f}" y="{L.Y(0.5) + 22:.1f}" text-anchor="middle">{th:.2f}</text>')
    L2 = Panel(body, 60, 160, 800, 40, (-5, 5), (0, 1))
    body.append('<text class="hd" x="60" y="140">The same eleven coins on the log-odds axis η = log(θ/(1 − θ))</text>')
    body.append(f'<line class="ax" x1="{L2.X(-5)}" y1="{L2.Y(0.5)}" x2="{L2.X(5)}" y2="{L2.Y(0.5)}"/>')
    for k in range(11):
        th = math.sin(a0 + (a1 - a0) * k / 10) ** 2
        e = math.log(th / (1 - th))
        body.append(f'<circle class="f2 ring" cx="{L2.X(e):.1f}" cy="{L2.Y(0.5):.1f}" r="5"/>')
        body.append(f'<text class="sm" x="{L2.X(e):.1f}" y="{L2.Y(0.5) + 22:.1f}" text-anchor="middle">{fmt(round(e, 1) + 0.0)}</text>')
    desc = ("Eleven coins spaced equally in Fisher–Rao distance, from θ = 0.01 to 0.99. On the θ axis (top) they bunch up "
            "near 0 and 1, where a small change in θ is easy to detect; on the log-odds axis (bottom) they bunch up near 0 "
            "instead. Neither axis is right: the spacing is a property of the distributions, and each parameterisation "
            "stretches it differently.")
    out.write_text(svg(W, H, "Equal steps in Fisher–Rao distance", desc, body))


def fig_geodesic(out, gm, gs):
    W, H = 920, 340
    body = []
    L = Panel(body, 70, 45, 380, 240, (-3.2, 3.2), (0, 2.2))
    L.frame([-3, -2, -1, 0, 1, 2, 3], [0, 0.5, 1, 1.5, 2], "mean μ", "standard deviation σ",
            "From N(−2, 1) to N(2, 1) in the (μ, σ) plane")
    L.line([-2, 2], [1, 1], "dash s0")
    L.line(gm, gs, "ln s1")
    L.dot(-2, 1, "f2", 5)
    L.dot(2, 1, "f2", 5)
    L.text(0, 1.0, f"straight: length {path_length(np.linspace(-2, 2, 400), np.ones(400)):.2f}", "sm", "middle", 0, 16)
    L.text(0, gs.max(), f"geodesic: length {fr_gauss(-2, 1, 2, 1):.2f}", "sm", "middle", 0, -8)
    R = Panel(body, 520, 45, 360, 240, (-6, 6), (0, 0.45))
    R.frame([-6, -4, -2, 0, 2, 4, 6], [0, 0.1, 0.2, 0.3, 0.4], "x", "density", "The distributions along the geodesic")
    x = np.linspace(-6, 6, 600)
    idx = np.linspace(0, gm.size - 1, 7).astype(int)
    for k, i in enumerate(idx):
        m, s = gm[i], gs[i]
        cls = "s2" if k in (0, 6) else ("s1" if k == 3 else "s0")
        R.line(x, np.exp(-(x - m) ** 2 / (2 * s ** 2)) / (s * math.sqrt(2 * math.pi)), f"{'ln' if k in (0, 3, 6) else 'thin'} {cls}")
    R.text(0, 1 / (gs.max() * math.sqrt(2 * math.pi)), f"midpoint N(0, {gs.max():.2f}²)", "sm", "middle", 0, -8)
    desc = ("Left: in the plane of mean and standard deviation, the straight path at σ = 1 between N(−2, 1) and N(2, 1) has "
            "Fisher–Rao length 4, while the geodesic, a half-ellipse rising to σ = √3, has length 3.24. Right: seven "
            "distributions along the geodesic: the shortest way to move a Gaussian is to widen it, slide it, and narrow it "
            "again.")
    out.write_text(svg(W, H, "The Gaussian half-plane", desc, body))


def fig_natgrad(out, gd, ng, hat, gd2):
    W, H = 920, 360
    body = []
    m_hat, s_hat = hat
    P = Panel(body, 90, 45, 780, 260, (-1.5, 3.0), (0.2, 3.2))
    P.frame([-1, 0, 1, 2, 3], [0.5, 1, 1.5, 2, 2.5, 3], "μ", "σ", "Fitting a Gaussian: plain gradient against natural gradient")
    mus = np.linspace(-1.5, 3.0, 181)
    sis = np.linspace(0.2, 3.2, 121)
    M, S = np.meshgrid(mus, sis)
    NLL = np.log(S) + (s_hat ** 2 + (m_hat - M) ** 2) / (2 * S ** 2)
    base = math.log(s_hat) + 0.5
    for lev in (0.05, 0.2, 0.5, 1, 2, 4, 8):
        # trace each contour column by column: for each mu, the sigmas where NLL crosses the level
        for branch in (0, 1):
            xs, ys = [], []
            for j, m in enumerate(mus):
                col = NLL[:, j] - base - lev
                idx = np.where(np.diff(np.sign(col)) != 0)[0]
                if len(idx) > branch:
                    i = idx[branch if branch < len(idx) else -1]
                    t = col[i] / (col[i] - col[i + 1])
                    xs.append(m); ys.append(sis[i] + t * (sis[i + 1] - sis[i]))
                else:
                    xs.append(m); ys.append(np.nan)
            P.line(xs, ys, "thin s0")
    P.line(gd[:, 0], gd[:, 1], "ln s2")
    P.line(gd2[:, 0], gd2[:, 1], "dash s2")
    P.line(ng[:, 0], ng[:, 1], "ln s1")
    P.dot(gd[0, 0], gd[0, 1], "f0", 5)
    P.dot(m_hat, s_hat, "f3", 5)
    P.text(-1.0, 3.0, "start", "sm", "start", 8, 4)
    P.text(m_hat, s_hat, "maximum likelihood", "sm", "start", 8, 14)
    for k, (cls, lab) in enumerate((("ln s2", "plain gradient in (μ, σ)"), ("dash s2", "plain gradient in (μ, log σ)"),
                                    ("ln s1", "natural gradient, in either coordinates"))):
        y = P.y0 + 14 + 18 * k
        body.append(f'<line class="{cls}" x1="{P.X(1.35):.1f}" y1="{y:.1f}" x2="{P.X(1.55):.1f}" y2="{y:.1f}"/>')
        body.append(f'<text class="sm" x="{P.X(1.6):.1f}" y="{y + 3.5:.1f}">{lab}</text>')
    desc = ("Contours of the average negative log-likelihood of a normal model for 200 points with mean about 2 and "
            "standard deviation about 0.44, in the (μ, σ) plane. From the start (−1, 3), plain gradient descent in (μ, σ) "
            "(solid orange) and in (μ, log σ) (dashed orange) take visibly different paths to the same answer. Natural "
            "gradient descent (blue), which rescales the gradient by the inverse Fisher matrix, takes the same path in both "
            "coordinate systems: it keeps the Gaussian wide while it moves the mean, and narrows it at the end.")
    out.write_text(svg(W, H, "Natural gradient", desc, body))


def figures(coin_rows, cr, two, gm, gs, gd, ng, hat, gd2):
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    fig_coin(d / "coin.svg", coin_rows)
    fig_efficiency(d / "efficiency.svg", *cr)
    fig_ellipse(d / "ellipse.svg", two)
    fig_metric(d / "metric.svg")
    fig_geodesic(d / "gaussian-geodesic.svg", gm, gs)
    fig_natgrad(d / "natural-gradient.svg", gd, ng, hat, gd2)
    print("\nfigures: wrote", ", ".join(sorted(p.name for p in d.glob("*.svg"))))


def main():
    coin_rows = check_coin()
    cr = check_cramer_rao()
    two = check_two_params()
    check_local_kl()
    check_reparam()
    gm, gs = check_gauss_geometry()
    gd, ng, hat, gd2 = check_natural_gradient()
    check_hessian_empirical()
    check_ntk()
    check_monte_carlo()
    if "--figures" in sys.argv:
        figures(coin_rows, cr, two, gm, gs, gd, ng, hat, gd2)


if __name__ == "__main__":
    main()
