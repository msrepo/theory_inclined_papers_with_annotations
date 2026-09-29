#!/usr/bin/env python3
"""Achille, Mbeng & Soatto 2019: dynamics and reachability of learning tasks, checked by hand.

Every number the notes quote is printed by this script, from small exact or
nearly exact computations: Gaussian integrals, 1-D and 2-D quadrature, a
matrix-exponential Fokker-Planck solver, the exact mean-first-passage-time
formula, and one short Langevin simulation. Nothing is trained.

Checked here, in the order the notes use them:

  1. Eq. (1) as printed is full-batch gradient descent (the noise cancels); the
     SGD step is w - eta grad L - sqrt(eta) T. The diffusion constant is
     D = eta sigma^2 / (2B), checked on SGD for a toy loss;
  2. the Fisher expansion of the KL has a factor 1/2: KL = (1/2) dw^T F dw;
  3. the Gaussian KL(Q || P) has -log|Sigma|, not +log|Sigma|;
  4. the optimal posterior covariance is Sigma* = beta (H + beta/lambda^2 I)^-1
     and C_beta(w0) = L + (beta/2)[|w0|^2/lambda^2 + log|lambda^2 H / beta + I|];
     the paper's Sigma* and eq. (4) are what one gets from dropping the 1/2 in
     the second-order expansion of the loss;
  5. the Gibbs identity behind Section 6: SGD's stationary density
     exp(-U/D) is exactly the minimiser of C_beta over ALL Q when beta = D and
     lambda^2 = D / gamma. The paper's beta = 2 lambda^2 gamma doubles the
     weight decay;
  6. the effective potential along a valley is U + (D/2) log b, not
     U + D log b and not U - D log b: exact quadrature, basin probabilities,
     the temperature at which a sharp minimum stops being a minimum, and a
     Langevin simulation;
  7. the "static x reachability" split of eq. (10) is detailed balance: the
     reachability factor is symmetric in the endpoints (exact for the OU
     process; to discretisation error for a double well), and
     p_t(y|x) / p_t(x|y) = exp(-dU/D) at every t;
  8. on the most likely downhill path the static and dynamic parts of the
     action cancel exactly; on the uphill path they add to dU/D;
  9. Kramers: the escape time follows the barrier to the saddle, not the
     difference between the two minima that eq. (16) uses;
 10. time to reach a loss threshold against batch size in a 100-dimensional
     quadratic: a noise-floor effect reproduces the shape of Figure 1
     (centre) without any complexity barrier;
 11. Figure 2 re-read from the numbers printed in it: pooled and within-target
     correlations of log epochs with the static distance, direction
     agreement, negative distances, and which way the matrices are oriented.

With --figures it also regenerates the SVGs in ../figures/ from these same
computations.

Standard library and numpy only.

Run:  python3 reachability.py            (checks)
      python3 reachability.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


def trapz(y, x):
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2)


def cumtrapz(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum((y[1:] + y[:-1]) * np.diff(x) / 2)
    return out


def expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring with an order-20 Taylor series."""
    nrm = np.abs(M).sum(axis=0).max()
    s = max(0, int(math.ceil(math.log2(nrm))) + 1) if nrm > 0 else 0
    A = M / 2 ** s
    E = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, 21):
        term = term @ A / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


# ------------------------------------------------------------------ 1. eq. (1)

def check_sgd_update():
    print("1. Eq. (1): SGD as a gradient step plus noise")
    rng = np.random.default_rng(1)
    k, eta = 6, 0.1
    w, g = rng.normal(size=k), rng.normal(size=k)
    gb = g + rng.normal(size=k)                    # a mini-batch gradient
    T = math.sqrt(eta) * (gb - g)                  # the paper's noise term
    printed = w - eta * gb + math.sqrt(eta) * T
    print(f"   printed update minus the full-batch GD step w - eta grad L: {np.abs(printed - (w - eta * g)).max():.1e}"
          "  (the noise cancels)")
    fixed = w - eta * g - math.sqrt(eta) * T
    print(f"   w - eta grad L - sqrt(eta) T minus the SGD step w - eta grad L_hat: {np.abs(fixed - (w - eta * gb)).max():.1e}")

    # D = eta sigma^2 / (2B) on the per-sample loss (1/2)(w - z_i)^2, Hessian h = 1:
    # the stationary variance of w is D / h in the SDE, eta sigma^2 / (B (2 - eta)) exactly.
    z = rng.normal(size=5000) * 1.7
    sig2 = z.var()
    eta, chains, steps, burn = 0.05, 100, 3000, 300
    print(f"   SGD on (1/2)(w - z_i)^2, eta = {eta}, per-sample gradient variance sigma^2 = {sig2:.3f}:")
    for B in (4, 16, 64):
        wc = np.zeros(chains)
        acc = []
        for t in range(steps):
            m = z[rng.integers(0, z.size, size=(chains, B))].mean(axis=1)
            wc = wc - eta * (wc - m)
            if t >= burn:
                acc.append(wc - z.mean())
        v = np.var(np.array(acc))
        print(f"     B = {B:3d}: stationary Var(w) = {v:.5f}; D/h = eta sigma^2/(2B) = {eta * sig2 / (2 * B):.5f};"
              f" exact AR(1) value {eta * sig2 / (B * (2 - eta)):.5f}")


# ------------------------------------------------------------------ 2. Fisher

def check_fisher():
    print("\n2. The Fisher expansion of the KL (Section 2)")
    rng = np.random.default_rng(2)
    C, d, n = 3, 4, 4000
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(C, d)) * 0.7

    def probs(W):
        Z = X @ W.T
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    P = probs(W)
    F = np.zeros((C * d, C * d))
    for c in range(C):                             # y ~ p_w(y|x): the model's own labels
        G = ((np.eye(C)[c] - P)[:, :, None] * X[:, None, :]).reshape(n, -1)
        F += (G * P[:, c:c + 1]).T @ G
    F /= n
    for eps in (1e-1, 1e-2, 1e-3):
        dW = rng.normal(size=(C, d))
        dW *= eps / np.linalg.norm(dW)
        kl = np.mean(np.sum(P * np.log(P / probs(W + dW)), axis=1))
        print(f"   |dw| = {eps:.0e}: E_x KL(p_w || p_(w+dw)) / (dw^T F dw) = {kl / (dW.ravel() @ F @ dW.ravel()):.4f}")


# ------------------------------------------------------------------ 3. Gaussian KL

def gauss_kl(w0, S, lam):
    k = w0.size
    return 0.5 * (np.trace(S) / lam ** 2 + w0 @ w0 / lam ** 2 - k + k * math.log(lam ** 2) - np.linalg.slogdet(S)[1])


def check_gauss_kl():
    print("\n3. KL between the Gaussian posterior and the Gaussian prior (Section 4)")
    rng = np.random.default_rng(3)
    k, lam = 4, 1.5
    w0 = rng.normal(size=k)
    A = rng.normal(size=(k, k))
    S = A @ A.T / k + 0.2 * np.eye(k)
    logdet = np.linalg.slogdet(S)[1]
    correct = gauss_kl(w0, S, lam)
    paper = correct + logdet                       # +log|S| instead of -log|S|
    Lc = np.linalg.cholesky(S)
    w = w0 + rng.normal(size=(400_000, k)) @ Lc.T
    Si = np.linalg.inv(S)
    logq = -0.5 * np.einsum("ij,jk,ik->i", w - w0, Si, w - w0) - 0.5 * logdet - 0.5 * k * math.log(2 * math.pi)
    logp = -0.5 * np.sum(w ** 2, axis=1) / lam ** 2 - k * math.log(lam) - 0.5 * k * math.log(2 * math.pi)
    mc = logq - logp
    print(f"   Monte Carlo KL = {mc.mean():.4f} +- {mc.std() / math.sqrt(mc.size):.4f};"
          f" formula with -log|S| = {correct:.4f}; as printed (+log|S|) = {paper:.4f}")


# ------------------------------------------------------------------ 4. Sigma* and eq. (4)

def check_sigma_star():
    print("\n4. The optimal posterior covariance and eq. (4)")
    rng = np.random.default_rng(4)
    k, beta, lam, L0 = 5, 0.5, 2.0, 0.3
    Qm, _ = np.linalg.qr(rng.normal(size=(k, k)))
    H = Qm @ np.diag([0.01, 0.1, 1.0, 3.0, 10.0]) @ Qm.T
    w0 = rng.normal(size=k)
    I = np.eye(k)

    def C(S, half=True):                           # exact for a quadratic loss around w0
        return L0 + (0.5 if half else 1.0) * np.trace(H @ S) + beta * gauss_kl(w0, S, lam)

    def grad(S, half=True):
        return (0.5 if half else 1.0) * H + 0.5 * beta * (I / lam ** 2 - np.linalg.inv(S))

    S_c = beta * np.linalg.inv(H + beta / lam ** 2 * I)
    S_p = beta / 2 * np.linalg.inv(H + beta / (2 * lam ** 2) * I)
    print(f"   |dC/dSigma| at beta (H + beta/lambda^2 I)^-1 = {np.linalg.norm(grad(S_c)):.1e};"
          f" at the paper's (beta/2)(H + beta/(2 lambda^2) I)^-1 = {np.linalg.norm(grad(S_p)):.3f}")
    print(f"   C at the two: {C(S_c):.5f} (corrected) < {C(S_p):.5f} (paper's)")
    print(f"   with the 1/2 dropped from E_Q[L] = L + tr(H Sigma) the paper's Sigma* has |dC/dSigma| ="
          f" {np.linalg.norm(grad(S_p, half=False)):.1e}")
    corr4 = L0 + beta / 2 * (w0 @ w0 / lam ** 2 + np.linalg.slogdet(lam ** 2 * H / beta + I)[1])
    paper4 = L0 + beta / 2 * (w0 @ w0 / lam ** 2 + np.linalg.slogdet(2 * lam ** 2 * H / beta + I)[1])
    print(f"   eq. (4) corrected, log|lambda^2 H/beta + I|: {corr4:.5f} (= min C {C(S_c):.5f});"
          f" as printed, log|2 lambda^2 H/beta + I|: {paper4:.5f}")


# ------------------------------------------------------------------ 5. Gibbs identity

def loss_1d(w):
    return 0.25 * (w ** 2 - 1) ** 2 + 0.3 * w


def tv(p, q, x):
    return 0.5 * trapz(np.abs(p - q), x)


def check_gibbs():
    print("\n5. The Gibbs identity behind Section 6")
    w = np.linspace(-7, 7, 28001)
    L = loss_1d(w)
    D, gamma = 0.2, 0.5
    beta, lam2 = D, D / gamma
    P = np.exp(-w ** 2 / (2 * lam2)) / math.sqrt(2 * math.pi * lam2)

    def Cfun(Q):
        m = Q > 1e-300
        return trapz(Q * L, w) + beta * trapz(np.where(m, Q * np.log(np.where(m, Q, 1) / P), 0.0), w)

    g = P * np.exp(-L / beta)
    Z = trapz(g, w)
    Qs = g / Z
    print(f"   D = {D}, gamma = {gamma}, beta = D, lambda^2 = D/gamma:"
          f" C_beta(Gibbs) = {Cfun(Qs):.6f}, -beta log Z = {-beta * math.log(Z):.6f}")
    for t in (-0.3, 0.1, 0.3):
        Qt = Qs * np.exp(t * np.sin(3 * w))
        Qt /= trapz(Qt, w)
        pos = Qs > 1e-300
        klt = trapz(np.where(pos, Qt * np.log(np.where(pos, Qt, 1) / np.where(pos, Qs, 1)), 0.0), w)
        print(f"     tilt t = {t:+.1f}: C(Q_t) - C(Gibbs) = {Cfun(Qt) - Cfun(Qs):.6f} = beta KL(Q_t||Gibbs) = {beta * klt:.6f}")
    U = L + gamma / 2 * w ** 2
    pi = np.exp(-(U - U.min()) / D)
    pi /= trapz(pi, w)
    print(f"   SGD stationary exp(-U/D), U = L + (gamma/2) w^2, against the Gibbs posterior:"
          f" max density gap {np.abs(pi - Qs).max():.1e}")
    qp = np.exp(-(L + gamma * w ** 2 - (L + gamma * w ** 2).min()) / D)   # beta = 2 lambda^2 gamma, beta = D
    qp /= trapz(qp, w)
    print(f"   the paper's beta = 2 lambda^2 gamma gives exp(-(L + gamma w^2)/beta): TV distance to the SGD"
          f" stationary law {tv(pi, qp, w):.3f} (E w^2 {trapz(w ** 2 * pi, w):.3f} vs {trapz(w ** 2 * qp, w):.3f})")


# ------------------------------------------------------------------ 6. effective potential

VALLEY = dict(A=0.5, c=0.05, peak=99.0, width=0.15)


def a_fun(u, A=VALLEY["A"], c=VALLEY["c"]):
    return A * (u ** 2 - 1) ** 2 + c * u


def a_prime(u, A=VALLEY["A"], c=VALLEY["c"]):
    return 4 * A * u * (u ** 2 - 1) + c


def b_fun(u, peak=VALLEY["peak"], s=VALLEY["width"]):
    return 1 + peak * np.exp(-(u + 1) ** 2 / (2 * s ** 2))


def b_prime(u, peak=VALLEY["peak"], s=VALLEY["width"]):
    return -peak * (u + 1) / s ** 2 * np.exp(-(u + 1) ** 2 / (2 * s ** 2))


U_GRID = np.linspace(-2.2, 2.2, 8801)


def f_eff(D, coef=0.5, u=U_GRID):
    """a + coef * D * log b; coef = 1/2 is the exact u-marginal, 1 is the paper's."""
    return a_fun(u) + coef * D * np.log(b_fun(u))


def p_sharp(D, coef=0.5):
    if D == 0:
        return 1.0
    F = f_eff(D, coef)
    g = np.exp(-(F - F.min()) / D)
    return trapz(np.where(U_GRID < 0, g, 0), U_GRID) / trapz(g, U_GRID)


def spike_is_bump(D, coef=0.5):
    """True once F has a local maximum within 0.1 of the sharp well's centre u = -1."""
    m = np.abs(U_GRID + 1) < 0.1
    F = f_eff(D, coef)[m]
    return bool(np.any((F[1:-1] > F[:-2]) & (F[1:-1] > F[2:])))


def bisect(fn, lo, hi, it=60):
    flo = fn(lo)
    for _ in range(it):
        mid = (lo + hi) / 2
        if fn(mid) == flo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate_valley(D, particles=2000, T=150.0, dt=0.005, seed=6):
    """Langevin in (u, v) for U = a(u) + b(u) v^2 / 2: Euler in u, the exact OU step in v at frozen u.
    All particles start in the flat well at u = +1."""
    rng = np.random.default_rng(seed)
    u = np.ones(particles)
    v = np.zeros(particles)
    steps = int(T / dt)
    frac = []
    for t in range(steps):
        b = b_fun(u)
        force = -a_prime(u) - 0.5 * b_prime(u) * v ** 2
        u = u + force * dt + math.sqrt(2 * D * dt) * rng.normal(size=particles)
        e = np.exp(-b * dt)
        v = v * e + np.sqrt(D / b * (1 - e ** 2)) * rng.normal(size=particles)
        if t * dt >= T / 3 and t % 20 == 0:
            frac.append(np.mean(u < 0))
    return float(np.mean(frac))


def check_valley():
    print("\n6. The effective potential along a valley (Sections 5.2 and 6)")
    print(f"   U(u, v) = a(u) + b(u) v^2/2, a = {VALLEY['A']}(u^2 - 1)^2 + {VALLEY['c']} u,"
          f" b = 1 + {VALLEY['peak']:.0f} exp(-(u+1)^2 / (2 * {VALLEY['width']}^2)): the deeper well at u = -1 is sharp")
    worst = 0.0
    for D in (0.05, 0.2):
        v = np.linspace(-8 * math.sqrt(D), 8 * math.sqrt(D), 4001)
        for u0 in np.linspace(-2, 2, 41):
            ex = -D * math.log(trapz(np.exp(-(a_fun(u0) + 0.5 * b_fun(u0) * v ** 2) / D), v))
            pred = a_fun(u0) + 0.5 * D * math.log(b_fun(u0)) - 0.5 * D * math.log(2 * math.pi * D)
            worst = max(worst, abs(ex - pred))
    print(f"   -D log (integral of exp(-U/D) dv) against a + (D/2) log b - (D/2) log(2 pi D): max gap {worst:.1e}")
    print("   probability of the sharp basin u < 0:   D    exact (D/2)   paper (D log b)   no curvature (a only)")
    for D in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        print(f"                                        {D:4.2f}   {p_sharp(D):.3f}         {p_sharp(D, 1.0):.3f}"
              f"             {p_sharp(D, 0.0):.3f}")
    d_half = bisect(lambda D: p_sharp(D) > 0.5, 0.01, 0.5)
    d_half_p = bisect(lambda D: p_sharp(D, 1.0) > 0.5, 0.005, 0.5)
    print(f"   the sharp basin loses its majority at D = {d_half:.3f} (paper's coefficient: {d_half_p:.3f})")
    d_min = bisect(spike_is_bump, 0.01, 0.6)
    d_min_p = bisect(lambda D: spike_is_bump(D, 1.0), 0.005, 0.6)
    print(f"   the bottom of the sharp well turns into a bump of F (two side minima) at D = {d_min:.3f}"
          f" (paper's coefficient: {d_min_p:.3f}); F''(-1) = 8A - (D/2) 0.99/s^2 = 4 - 22 D vanishes at {4 / 22:.3f}")
    Dsim = 0.15
    sim = simulate_valley(Dsim)
    print(f"   Langevin, D = {Dsim}, 2000 particles started in the flat well: fraction in the sharp basin"
          f" {sim:.3f} (exact {p_sharp(Dsim):.3f}, paper {p_sharp(Dsim, 1.0):.3f}, a only {p_sharp(Dsim, 0.0):.3f})")
    return sim, Dsim


# ------------------------------------------------------------------ 7. detailed balance

def tilted(w):
    return (w ** 2 - 1) ** 2 - 0.25 * w


def tilted_p(w):
    return 4 * w * (w ** 2 - 1) - 0.25


def fp_generator(n, D, lo=-2.0, hi=2.0, U=tilted, Up=tilted_p):
    """Central-difference Fokker-Planck operator dp/dt = d/dw (U' p) + D d^2p/dw^2, zero outside."""
    x = np.linspace(lo, hi, n)
    h = x[1] - x[0]
    A = np.zeros((n, n))
    f = Up(x)
    for i in range(n):
        A[i, i] = -2 * D / h ** 2
        if i > 0:
            A[i, i - 1] = D / h ** 2 - f[i - 1] / (2 * h)
        if i < n - 1:
            A[i, i + 1] = D / h ** 2 + f[i + 1] / (2 * h)
    return x, h, A


def check_detailed_balance():
    print("\n7. Eq. (10) is detailed balance: the reachability factor is symmetric")
    h_, D, t = 1.3, 0.2, 0.7
    x = np.linspace(-2, 2, 41)
    X, Y = np.meshgrid(x, x, indexing="ij")
    var = D * (1 - math.exp(-2 * h_ * t)) / h_
    p = np.exp(-(Y - X * math.exp(-h_ * t)) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)
    K = np.exp((0.5 * h_ * Y ** 2 - 0.5 * h_ * X ** 2) / (2 * D)) * p
    print(f"   OU process, exact kernel: max |K(x,y) - K(y,x)| / max K = {np.abs(K - K.T).max() / K.max():.1e}")
    D = 0.25
    for n in (101, 201, 401):
        xg, hg, A = fp_generator(n, D)
        Pt = expm(10.0 * A)                        # Pt[j, i] = mass at x_j at t = 10 from x_i
        Ug = tilted(xg)
        K = np.exp((Ug[:, None] - Ug[None, :]) / (2 * D)) * Pt / hg
        m = slice(n // 8, n - n // 8)
        Km = K[m, m]
        print(f"   tilted double well, finite differences n = {n}: max relative asymmetry of K = "
              f"{np.abs(Km - Km.T).max() / Km.max():.1e}")
    xg, hg, A = fp_generator(401, D)
    Ug = tilted(xg)
    i0, i1 = np.argmin(np.abs(xg + 1.03)), np.argmin(np.abs(xg - 0.97))
    near1 = np.abs(xg - xg[i1]) <= 0.1
    dU = Ug[i1] - Ug[i0]
    print(f"   start at w = {xg[i0]:.2f} (U = {Ug[i0]:.3f}), target w = {xg[i1]:.2f} (U = {Ug[i1]:.3f}), D = {D}:"
          f" static factor exp(-dU/2D) = {math.exp(-dU / (2 * D)):.2f}")
    rows = []
    for t in (0.5, 5.0, 50.0, 500.0):
        Pt = expm(t * A)
        mass = Pt[near1, i0].sum()
        ratio = Pt[i1, i0] / Pt[i0, i1]
        rows.append((t, mass))
        print(f"     t = {t:6.1f}: P(|w - {xg[i1]:.2f}| <= 0.1) = {mass:.2e};"
              f"  p_t(target|start)/p_t(start|target) = {ratio:.3f}  (exp(-dU/D) = {math.exp(-dU / D):.3f})")
    return rows


# ------------------------------------------------------------------ 8. action on gradient paths

def check_action():
    print("\n8. The action along the most likely paths (Section 5.2)")
    ws = [r.real for r in np.roots([4, 0, -4, -0.25]) if abs(r.imag) < 1e-12]
    ws.sort()
    w_left, w_sad, w_right = ws
    out = {}
    for name, start, stop in (("downhill, saddle -> right well", w_sad + 1e-3, w_right),
                              ("downhill, saddle -> left well", w_sad - 1e-3, w_left)):
        w, dt, path = start, 1e-4, [start]
        while abs(w - stop) > 1e-3:
            k1 = -tilted_p(w); k2 = -tilted_p(w + dt / 2 * k1); k3 = -tilted_p(w + dt / 2 * k2); k4 = -tilted_p(w + dt * k3)
            w += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            path.append(w)
        path = np.array(path)
        wd = np.gradient(path, dt)
        dyn = trapz(0.5 * wd ** 2 + 0.5 * tilted_p(path) ** 2, np.arange(path.size) * dt)
        dU = tilted(path[-1]) - tilted(path[0])
        out[name] = (dU, dyn)
        print(f"   {name}: dU = {dU:+.4f}; 2D x action: static {dU:+.4f}, dynamic {dyn:+.4f},"
              f" total {dU + dyn:+.1e}")
        print(f"     the same path run backwards (uphill): static {-dU:+.4f}, dynamic {dyn:+.4f},"
              f" total {dyn - dU:+.4f} = 2|dU|, so the path weight is exp(-|dU|/D)")
    # a straight constant-speed path from the left well to the saddle, best duration
    x = np.linspace(w_left, w_sad, 4001)
    Lx = w_sad - w_left
    rms = math.sqrt(trapz(tilted_p(x) ** 2, x) / Lx)
    best = Lx * rms                                 # min over T of Lx^2/(2T) + (T/2) mean(U'^2)
    bar = tilted(w_sad) - tilted(w_left)
    print(f"   left well -> saddle (barrier {bar:.4f}): best constant-speed path has dynamic part {best:.4f}"
          f" + static {bar:.4f} = {best + bar:.4f} > 2 x barrier = {2 * bar:.4f} for the reversed gradient path")
    return out, (w_left, w_sad, w_right)


# ------------------------------------------------------------------ 9. Kramers

def kr_U(w, h, s):
    return h * (w ** 2 - 1) ** 2 + s * (3 * w - w ** 3) / 2


def kr_saddle(h, s):
    return 3 * s / (8 * h)


def kr_barrier(h, s):
    return kr_U(kr_saddle(h, s), h, s) - kr_U(-1.0, h, s)


def solve_h(s, E=1.0):
    lo, hi = 0.2, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if kr_barrier(mid, s) < E:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mfpt(U, D, a=-1.0, b=1.0, lo=-3.0, n=40001):
    """Exact mean first-passage time from a to b > a, absorbing at b, reflecting far to the left."""
    y = np.linspace(lo, b, n)
    Uy = U(y) - U(np.array(a))
    inner = cumtrapz(np.exp(-Uy / D), y)
    m = y >= a
    return trapz(np.exp(Uy[m] / D) * inner[m], y[m]) / D


KR_S = (0.2, 0.0, -0.25, -0.5, -0.75)


def check_kramers():
    print("\n9. Kramers: escape follows the barrier, not the end-to-end difference (Section 6, eq. 16)")
    print("   U = h (w^2 - 1)^2 + s (3w - w^3)/2 with h chosen so the barrier from w = -1 is exactly 1")
    rows = []
    for s in KR_S:
        h = solve_h(s)
        U = lambda w, h=h, s=s: kr_U(w, h, s)
        dU = kr_U(1.0, h, s) - kr_U(-1.0, h, s)
        ws = kr_saddle(h, s)
        Ua = 8 * h - 3 * s * (-1.0)                 # U''(-1)
        Us = 4 * h * (3 * ws ** 2 - 1) - 3 * s * ws  # U''(w_s) < 0
        taus = {D: mfpt(U, D) for D in (0.05, 0.1)}
        kr = 2 * math.pi / math.sqrt(Ua * abs(Us)) * math.exp(1.0 / 0.05)
        slope = (math.log(taus[0.05]) - math.log(taus[0.1])) / (1 / 0.05 - 1 / 0.1)
        rows.append((s, h, dU, taus))
        print(f"   s = {s:+.2f}, h = {h:.3f}: dU(end - start) = {dU:+.2f}; D ln tau = {0.05 * math.log(taus[0.05]):.3f}"
              f" (D = 0.05), {0.1 * math.log(taus[0.1]):.3f} (D = 0.1); d ln tau / d(1/D) = {slope:.3f};"
              f" tau / Kramers = {taus[0.05] / kr:.3f}")
    spread = [0.1 * math.log(r[3][0.1]) for r in rows]
    print(f"   across dU from {rows[0][2]:+.1f} to {rows[-1][2]:+.1f}, D ln tau at D = 0.1 moves by"
          f" {max(spread) - min(spread):.3f}; eq. (16) read as ln tau = dU/D + const would move it by"
          f" {rows[0][2] - rows[-1][2]:.1f}")
    return rows


# ------------------------------------------------------------------ 10. batch size

BATCH = dict(k=100, h=1.0, r0=10.0, loss=1.5, c=2.0)


def radial_mfpt(D, k=BATCH["k"], h=BATCH["h"], r0=BATCH["r0"], loss=BATCH["loss"], n=40001):
    """Time for |w| to fall from r0 to rho = sqrt(2 loss / h) under dw = -h w dt + sqrt(2D) dW in k dims."""
    rho = math.sqrt(2 * loss / h)
    z = np.linspace(rho, r0 + 2, n)
    dz = z[1] - z[0]
    logf = (k - 1) * np.log(z) - h * z ** 2 / (2 * D)
    # each cell integrated exactly for log-linear f, so a decay length below dz is still handled
    a, b = logf[:-1], logf[1:]
    g = np.abs(a - b)
    cell = np.maximum(a, b) + np.log(np.where(g < 1e-12, 1.0, -np.expm1(-g) / np.where(g < 1e-12, 1.0, g))) + math.log(dz)
    logI = np.append(np.logaddexp.accumulate(cell[::-1])[::-1], -np.inf)
    m = z <= r0
    return trapz(np.exp(logI[m] - logf[m]), z[m]) / D


def check_batch():
    print("\n10. Time to a loss threshold against batch size (Figure 1, centre)")
    k, h, r0, loss, c = (BATCH[x] for x in ("k", "h", "r0", "loss", "c"))
    gf = math.log(r0 / math.sqrt(2 * loss / h)) / h
    print(f"   quadratic loss (h/2)|w|^2 in k = {k} dims from loss {h * r0 ** 2 / 2:.0f} to {loss}, D = {c}/B;"
          f" gradient-flow time {gf:.3f}; the noise floor k D / 2 hits the threshold at B = {c * k / (2 * loss):.1f}")
    rows = []
    for B in (70, 80, 100, 150, 200, 300, 400, 1000):
        T = radial_mfpt(c / B)
        rows.append((B, T))
        print(f"     B = {B:4d}: D = {c / B:.4f}, floor {k * c / B / 2:.3f}, mean time to threshold {T:.3f}")
    print("   eq. (16) with D = k/B gives tau ~ exp(dC B / k): growing in B if dC > 0, and tending to 0"
          " (not to the gradient-flow time) if dC < 0")
    lo, hi = 2900.0, 4700.0
    print(f"   Figure 1 (left), ends read off the plot at about {lo:.0f} and {hi:.0f} steps: at mid-range a straight"
          f" line gives {(lo + hi) / 2:.0f}, an exponential {math.sqrt(lo * hi):.0f} ({((lo + hi) / 2 / math.sqrt(lo * hi) - 1) * 100:.1f}% apart)")
    return rows, gf


# ------------------------------------------------------------------ 11. Figure 2

NAMES = ["cifar10", "mnist", "fashion", "ifashion", "letters", "cifar100", "natural", "artificial"]
_n = float("nan")
DIST = np.array([
    [0.00, 0.29, 0.31, 0.28, 0.10, 0.01, -0.08, -0.19],
    [0.15, 0.00, 0.20, 0.15, 0.04, -0.01, -0.06, -0.01],
    [0.21, 0.25, 0.00, 0.07, 0.12, 0.06, -0.03, -0.04],
    [0.24, 0.25, 0.12, 0.00, 0.10, 0.07, 0.02, 0.01],
    [0.61, 0.68, 0.72, 0.64, 0.00, 0.41, 0.32, 0.33],
    [0.52, 0.64, 0.67, 0.62, 0.42, 0.00, 0.12, 0.07],
    [0.41, 0.57, 0.55, 0.54, 0.30, 0.10, 0.00, -0.14],
    [0.25, 0.58, 0.49, 0.50, 0.27, 0.00, -0.18, 0.00]])
EPOCHS = np.array([
    [_n, 32.85, 30.12, 27.30, 28.84, 17.23, 19.24, 25.78],
    [0.57, _n, 0.70, 0.75, 0.32, 0.48, 0.57, 0.76],
    [18.75, 33.72, _n, 11.51, 26.33, 18.75, 21.75, 24.87],
    [19.35, 35.38, 17.72, _n, 29.38, 18.73, 21.47, 25.41],
    [55.80, _n, 79.72, 78.43, _n, 53.43, 58.24, 74.87],
    [42.29, 54.99, 61.52, 51.92, 48.85, _n, 24.84, 39.86],
    [27.60, 36.85, 36.68, 34.41, 31.22, 12.74, _n, 27.17],
    [14.60, 21.20, 20.03, 19.06, 17.75, 7.57, 11.28, _n]])


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def fig2_mask(drop_mnist=True):
    m = ~np.isnan(EPOCHS) & ~np.eye(8, dtype=bool)
    if drop_mnist:
        m[1] = False
    return m


def check_fig2():
    print("\n11. Figure 2 re-read from the numbers printed in its two matrices (rows = target, columns = source)")
    mall, m = fig2_mask(False), fig2_mask(True)
    y = np.log2(EPOCHS)
    print(f"   pooled r(distance, log2 epochs): all {mall.sum()} converged pairs {corr(DIST[mall], y[mall]):.2f};"
          f" without the MNIST-target row (epochs < 1, off the plot) {m.sum()} pairs {corr(DIST[m], y[m]):.2f}")
    slope = np.polyfit(DIST[m], y[m], 1)[0]
    print(f"   least-squares slope {slope:.2f} doublings per unit distance, i.e. epochs ~ exp(d / {1 / (slope * math.log(2)):.2f})")
    xs, ys, row = [], [], []
    for i in range(8):
        k = mall[i]
        xi, yi = DIST[i, k], y[i, k]
        xs += list(xi - xi.mean()); ys += list(yi - yi.mean()); row += [i] * k.sum()
        print(f"     target {NAMES[i]:10s} ({k.sum()} sources): r = {corr(xi, yi):+.2f}, Spearman {corr(rank(xi), rank(yi)):+.2f}")
    xs, ys, row = np.array(xs), np.array(ys), np.array(row)
    wr = row != 1
    print(f"   within-target (each row centred): all {len(xs)} pairs r = {corr(xs, ys):.2f};"
          f" without the MNIST row {wr.sum()} pairs r = {corr(xs[wr], ys[wr]):.2f}")
    rm = np.array([DIST[i, m[i]].mean() for i in range(8) if m[i].any()])
    re = np.array([y[i, m[i]].mean() for i in range(8) if m[i].any()])
    print(f"   between targets (row means, 7 rows): r = {corr(rm, re):.2f}")
    agree = tot = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if np.isnan(EPOCHS[i, j]) or np.isnan(EPOCHS[j, i]):
                continue
            tot += 1
            agree += np.sign(DIST[i, j] - DIST[j, i]) == np.sign(EPOCHS[i, j] - EPOCHS[j, i])
    print(f"   direction: the larger of d(A->B), d(B->A) is also the slower fine-tune in {agree} of {tot} pairs")
    off = DIST[~np.eye(8, dtype=bool)]
    print(f"   negative 'distances': {(off < 0).sum()} of {off.size} off-diagonal entries, down to {off.min():.2f}")
    print(f"   orientation: row cifar10 / column cifar100 is d = {DIST[0, 5]:.2f}, {EPOCHS[0, 5]:.2f} epochs;"
          f" row cifar100 / column cifar10 is d = {DIST[5, 0]:.2f}, {EPOCHS[5, 0]:.2f} epochs")
    return slope


# ------------------------------------------------------------------ figures

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:10.5px;fill:#57564f} .v{font-size:11px;fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s4{stroke:#eda100} .s0{stroke:#8a8880}
  .f1{fill:#2a78d6} .f2{fill:#eb6834} .f3{fill:#1baf7a} .f4{fill:#eda100} .f0{fill:#8a8880}
  .ln{stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
  .dash{stroke-width:1.4;fill:none;stroke-dasharray:5 3}
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


class Panel:
    """A plotting frame: data (x, y) -> pixels, with grid, ticks and labels."""

    def __init__(self, body, x0, y0, w, h, xr, yr, logx=False):
        self.b, self.x0, self.y0, self.w, self.h, self.xr, self.yr, self.logx = body, x0, y0, w, h, xr, yr, logx

    def X(self, x):
        lo, hi = self.xr
        if self.logx:
            x, lo, hi = math.log10(x), math.log10(lo), math.log10(hi)
        return self.x0 + (x - lo) / (hi - lo) * self.w

    def Y(self, y):
        lo, hi = self.yr
        return self.y0 + self.h - (y - lo) / (hi - lo) * self.h

    def frame(self, xt, yt, xlab, ylab, title, fmt_x=str, fmt_y=str):
        b = self.b
        for y in yt:
            b.append(f'<line class="gd" x1="{self.x0}" y1="{self.Y(y):.1f}" x2="{self.x0 + self.w}" y2="{self.Y(y):.1f}"/>')
            b.append(f'<text class="sm" x="{self.x0 - 6}" y="{self.Y(y) + 3.5:.1f}" text-anchor="end">{fmt_y(y)}</text>')
        for x in xt:
            b.append(f'<text class="sm" x="{self.X(x):.1f}" y="{self.y0 + self.h + 15}" text-anchor="middle">{fmt_x(x)}</text>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0 + self.h}" x2="{self.x0 + self.w}" y2="{self.y0 + self.h}"/>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0}" x2="{self.x0}" y2="{self.y0 + self.h}"/>')
        b.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 32}" text-anchor="middle">{xlab}</text>')
        b.append(f'<text class="lab" transform="translate({self.x0 - 40},{self.y0 + self.h / 2}) rotate(-90)" text-anchor="middle">{ylab}</text>')
        b.append(f'<text class="hd" x="{self.x0}" y="{self.y0 - 12}">{title}</text>')

    def line(self, xs, ys, cls):
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys)
                       if self.yr[0] - 1e-9 <= y <= self.yr[1] + 1e-9)
        self.b.append(f'<polyline class="{cls}" points="{pts}"/>')

    def dot(self, x, y, cls, r=4.5):
        self.b.append(f'<circle class="{cls} ring" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>')

    def text(self, x, y, s, cls="v", anchor="start", dx=0, dy=0):
        self.b.append(f'<text class="{cls}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" text-anchor="{anchor}">{s}</text>')


def fmt(v):
    return f"{v:g}".replace("-", "−")


def fig_kramers(out: Path, rows):
    W, H = 920, 330
    body = []
    L = Panel(body, 70, 45, 330, 220, (-1.6, 1.6), (-1.75, 2.0))
    L.frame([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], "weight w", "U(w) − U(−1)",
            "Same barrier from w = −1, different end points", fmt, fmt)
    pick = [(0, "s1"), (2, "s2"), (4, "s3")]
    w = np.linspace(-1.6, 1.6, 400)
    for idx, cls in pick:
        s, h, dU, _ = rows[idx]
        L.line(w, kr_U(w, h, s) - kr_U(-1.0, h, s), f"ln {cls}")
        L.text(1.0, dU, f"ΔU = {fmt(round(dU, 2))}", "sm", "start", 8, 4)
    L.line([-1.6, 1.6], [1, 1], "dash s0")
    L.text(-1.55, 1, "barrier = 1", "sm", "start", 0, -5)
    R = Panel(body, 530, 45, 350, 220, (-1.6, 0.5), (-1.0, 1.6))
    R.frame([-1.5, -1, -0.5, 0, 0.5], [-1, -0.5, 0, 0.5, 1, 1.5], "ΔU = U(+1) − U(−1)", "D · ln τ",
            "Exact escape time against ΔU", fmt, fmt)
    for D, cls in ((0.05, "f1"), (0.1, "f2")):
        xs = [r[2] for r in rows]
        ys = [D * math.log(r[3][D]) for r in rows]
        R.line(xs, ys, f"ln {cls.replace('f', 's')}")
        for x, y in zip(xs, ys):
            R.dot(x, y, cls)
    R.text(rows[0][2], 0.1 * math.log(rows[0][3][0.1]), "exact, D = 0.05 and D = 0.1 (they overlap)", "sm", "end", 0, -12)
    x0 = rows[1][2]
    y0 = 0.1 * math.log(rows[1][3][0.1])
    R.line([-1.6, 0.5], [y0 + (-1.6 - x0), y0 + (0.5 - x0)], "dash s3")
    R.text(-1.2, y0 + (-1.2 - x0), "eq. (16): D ln τ = ΔU + const", "sm", "start", 8, 14)
    desc = ("Left: three one-dimensional potentials with the same barrier of height 1 above the starting well at "
            "w = −1 but end wells at ΔU = +0.4, −0.5 and −1.5. Right: the exact mean first-passage time τ from −1 to +1, "
            "plotted as D ln τ against ΔU for D = 0.05 and 0.1. The dots lie on nearly flat lines near the barrier "
            "height, while eq. (16), which puts the end-to-end difference in the exponent, predicts a line of slope 1.")
    out.write_text(svg(W, H, "Escape time follows the barrier", desc, body))


def fig_effective(out: Path, sim, Dsim):
    W, H = 920, 330
    body = []
    L = Panel(body, 70, 45, 330, 220, (-2.0, 2.0), (-0.2, 1.2))
    L.frame([-2, -1, 0, 1, 2], [0, 0.4, 0.8, 1.2], "u (along the valley)", "F(u) − F(+1)",
            "Effective potential a(u) + (D/2) log b(u)", fmt, fmt)
    u = np.linspace(-2, 2, 801)
    i1 = np.argmin(np.abs(u - 1))
    for D, cls in ((0.0, "s0"), (0.1, "s1"), (0.2, "s2"), (0.3, "s3")):
        F = f_eff(D, 0.5, u)
        L.line(u, F - F[i1], f"ln {cls}")
        L.text(-1.0, F[np.argmin(np.abs(u + 1))] - F[i1], f"D = {fmt(D)}", "sm", "start", 12 if D else -44, -2)
    L.text(-1, -0.2, "sharp well", "sm", "middle", 0, -6)
    L.text(1, -0.2, "flat well", "sm", "middle", 0, -6)
    R = Panel(body, 530, 45, 350, 220, (0.0, 0.4), (0.0, 1.0))
    R.frame([0, 0.1, 0.2, 0.3, 0.4], [0, 0.25, 0.5, 0.75, 1], "temperature D", "P(sharp basin)",
            "Probability of the sharp basin", fmt, fmt)
    Ds = np.linspace(0.005, 0.4, 80)
    R.line(Ds, [p_sharp(D) for D in Ds], "ln s1")
    R.line(Ds, [p_sharp(D, 1.0) for D in Ds], "dash s2")
    R.line(Ds, [p_sharp(D, 0.0) for D in Ds], "dash s0")
    R.dot(Dsim, sim, "f1", 5)
    R.text(0.4, p_sharp(0.4), "exact: (D/2) log b", "sm", "end", 0, -8)
    R.text(0.4, p_sharp(0.4, 1.0), "paper: D log b", "sm", "end", 0, 14)
    R.text(0.4, p_sharp(0.4, 0.0), "no curvature term", "sm", "end", 0, -8)
    R.text(Dsim, sim, "Langevin", "sm", "start", 9, -6)
    desc = ("Left: the effective potential along a valley U = a(u) + b(u) v²/2 whose deeper well at u = −1 is a hundred "
            "times sharper across the valley, for D = 0, 0.1, 0.2, 0.3. The log b term lifts the sharp well until, "
            "near D = 0.18, it is no longer a minimum. Right: the probability of the sharp basin against D, exact (the "
            "(D/2) log b correction), with the paper's D log b coefficient, and with no curvature term; a Langevin "
            "simulation at D = 0.15 sits on the exact curve.")
    out.write_text(svg(W, H, "Sharp minima lose to flat ones", desc, body))


def fig_action(out: Path, act):
    W, H = 920, 300
    body = []
    dU, dyn = act["downhill, saddle -> right well"]
    P = Panel(body, 90, 45, 780, 190, (0, 6), (-1.5, 3.0))
    P.frame([], [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], "", "2D × action (units of U)",
            "Static and dynamic parts of the path weight, saddle ↔ right well", fmt, fmt)
    groups = [("downhill (gradient flow)", dU, dyn), ("uphill (reversed)", -dU, dyn)]
    for gi, (name, st, dy) in enumerate(groups):
        base = 0.5 + 3 * gi
        for j, (val, cls, lab) in enumerate(((st, "f1", "static ΔU"), (dy, "f2", "dynamic ∫½ẇ²+½U′²"),
                                             (st + dy, "f3", "total"))):
            x = base + j * 0.75
            y0, y1 = P.Y(0), P.Y(val)
            top = min(y0, y1)
            hgt = max(abs(y1 - y0), 0.8)
            body.append(f'<rect class="{cls}" x="{P.X(x):.1f}" y="{top:.1f}" width="{P.X(0.6) - P.X(0):.1f}" height="{hgt:.1f}" rx="3"/>')
            lab_v = "0" if abs(val) < 5e-4 else f"{val:+.3f}".replace("-", "−")
            P.text(x + 0.3, val, lab_v, "v", "middle", 0, -6 if val >= 0 else 14)
            P.text(x + 0.3, -1.5, lab, "sm", "middle", 0, 16)
        P.text(base + 1.05, 3.0, name, "lab", "middle", 0, 12)
    P.line([0, 6], [0, 0], "ax")
    desc = ("Bars of the two parts of 2D times the Onsager–Machlup action for the tilted double well, along the gradient-"
            f"flow path from the saddle to the right well (ΔU = {dU:.3f}) and along the same path reversed. Downhill the "
            "static part is negative and the dynamic part cancels it exactly, so the total is zero; uphill both are "
            "positive and add to twice the climb, giving the Arrhenius weight exp(−ΔU/D).")
    out.write_text(svg(W, H, "Static and dynamic parts cancel downhill", desc, body))


def fig_fig2(out: Path):
    W, H = 920, 340
    body = []
    m = fig2_mask(True)
    y = np.log2(EPOCHS)
    group = {0: 0, 5: 0, 6: 0, 7: 0, 2: 1, 3: 1, 4: 2}
    cls = ["f1", "f2", "f3"]
    labels = ["CIFAR targets", "Fashion targets", "Letters target"]
    Lp = Panel(body, 70, 50, 340, 220, (-0.25, 0.8), (3.0, 6.5))
    Lp.frame([-0.2, 0, 0.2, 0.4, 0.6, 0.8], [3, 4, 5, 6], "static distance d (Fig. 2 left)", "log₂ epochs (Fig. 2 centre)",
             f"Pooled: r = {corr(DIST[m], y[m]):.2f}", fmt, fmt)
    Rp = Panel(body, 540, 50, 340, 220, (-0.45, 0.45), (-1.5, 1.5))
    xs, ys, gs = [], [], []
    for i in range(8):
        k = m[i]
        if not k.any():
            continue
        xi, yi = DIST[i, k], y[i, k]
        for a, b in zip(xi, yi):
            Lp.dot(a, b, cls[group[i]], 4)
        xs += list(xi - xi.mean()); ys += list(yi - yi.mean()); gs += [group[i]] * len(xi)
    Rp.frame([-0.4, -0.2, 0, 0.2, 0.4], [-1.5, -1, -0.5, 0, 0.5, 1, 1.5], "d minus its target-row mean",
             "log₂ epochs minus row mean", f"Within each target: r = {corr(np.array(xs), np.array(ys)):.2f}", fmt, fmt)
    for a, b, g in zip(xs, ys, gs):
        Rp.dot(a, b, cls[g], 4)
    s, c = np.polyfit(DIST[m], y[m], 1)
    Lp.line([-0.25, 0.8], [c - 0.25 * s, c + 0.8 * s], "dash s0")
    s2, c2 = np.polyfit(xs, ys, 1)
    Rp.line([-0.45, 0.45], [c2 - 0.45 * s2, c2 + 0.45 * s2], "dash s0")
    for j, lab in enumerate(labels):
        body.append(f'<circle class="{cls[j]} ring" cx="{80 + 150 * j}" cy="322" r="4.5"/>')
        body.append(f'<text class="sm" x="{90 + 150 * j}" y="326">{lab}</text>')
    desc = ("Scatter plots of the 48 converged off-diagonal fine-tuning pairs printed in Figure 2 (the MNIST-target row, "
            "whose epochs are below 1, is left out as in the paper's own scatter). Left: log2 epochs against the static "
            f"distance, r = {corr(DIST[m], y[m]):.2f}. Right: both centred within each target row, r = "
            f"{corr(np.array(xs), np.array(ys)):.2f}, so the distance predicts which "
            "source is faster for a fixed target, not only which targets are hard. Colours group the targets into "
            "CIFAR (cifar10, cifar100, natural, artificial), Fashion-MNIST (fashion, ifashion) and Letters.")
    out.write_text(svg(W, H, "Figure 2 re-read", desc, body))


def fig_batch(out: Path, rows, gf):
    W, H = 920, 320
    body = []
    P = Panel(body, 90, 45, 780, 210, (60, 1000), (0, 4), logx=True)
    P.frame([60, 100, 200, 400, 1000], [0, 1, 2, 3, 4], "batch size B (log scale), D = 2/B",
            "mean time to loss ≤ 1.5", "Quadratic loss in 100 dimensions: time to a fixed loss threshold", fmt, fmt)
    Bs = np.exp(np.linspace(math.log(68), math.log(1000), 60))
    P.line(Bs, [radial_mfpt(2.0 / B) for B in Bs], "ln s1")
    for B, T in rows:
        P.dot(B, T, "f1", 4)
    P.line([60, 1000], [gf, gf], "dash s0")
    P.text(1000, gf, f"gradient flow, {gf:.2f}", "sm", "end", 0, -6)
    Bf = BATCH["c"] * BATCH["k"] / (2 * BATCH["loss"])
    body.append(f'<line class="dash s2" x1="{P.X(Bf):.1f}" y1="{P.y0}" x2="{P.X(Bf):.1f}" y2="{P.y0 + P.h}"/>')
    P.text(Bf, 4, "noise floor k·D/2 = threshold", "sm", "start", 6, 12)
    desc = ("The exact mean time for a 100-dimensional Ornstein–Uhlenbeck process (SGD on a quadratic loss with "
            "D = 2/B) to bring the loss from 50 down to 1.5, against batch size. The time falls steeply as B grows "
            "past the point where the stationary loss k D / 2 equals the threshold, then levels off at the "
            "gradient-flow time 1.75: the shape of Figure 1 (centre), produced by a noise floor rather than a barrier.")
    out.write_text(svg(W, H, "Batch size and a noise floor", desc, body))


def figures(sim, Dsim, act, kr_rows, batch_rows, gf):
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    fig_kramers(d / "kramers-barrier.svg", kr_rows)
    fig_effective(d / "effective-potential.svg", sim, Dsim)
    fig_action(d / "action-split.svg", act)
    fig_fig2(d / "figure2-reread.svg")
    fig_batch(d / "batch-size.svg", batch_rows, gf)
    print("\nfigures: wrote", ", ".join(sorted(p.name for p in d.glob("*.svg"))))


def main():
    check_sgd_update()
    check_fisher()
    check_gauss_kl()
    check_sigma_star()
    check_gibbs()
    sim, Dsim = check_valley()
    check_detailed_balance()
    act, _ = check_action()
    kr_rows = check_kramers()
    batch_rows, gf = check_batch()
    check_fig2()
    if "--figures" in sys.argv:
        figures(sim, Dsim, act, kr_rows, batch_rows, gf)


if __name__ == "__main__":
    main()
