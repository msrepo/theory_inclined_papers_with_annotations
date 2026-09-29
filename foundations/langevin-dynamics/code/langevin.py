#!/usr/bin/env python3
"""Langevin dynamics, checked by hand: every number quoted in the foundations notes.

Small simulations and exact computations (quadrature, a reversible grid
generator for the Fokker-Planck equation, the mean-first-passage-time integral,
Lyapunov equations), with nothing trained.

Checked here, in the order the notes use them:

  1. Brownian motion: Var(W_t) = 2 D t, and the same answer for any step size
     once the kick per step is sqrt(2 D eta);
  2. Ornstein-Uhlenbeck: stationary variance D / h, and equipartition,
     E[h w^2 / 2] = D / 2 in every direction whatever its curvature;
  3. the Gibbs law exp(-U/D) / Z on a tilted double well, against a Langevin
     simulation;
  4. depth against width: the occupation of a deep narrow well against a
     shallow wide one, exact and by the Laplace (free-energy) rule;
  5. the Fokker-Planck equation on a grid: relaxation to Gibbs, and the
     slowest rate equal to the sum of the two Kramers rates;
  6. Kramers' law: the exact mean first-passage time, the Kramers formula,
     and a simulation;
  7. detailed balance and when it fails: a rotating drift keeps the Gibbs
     snapshot but makes the movie run in circles; noise shaped like the
     Hessian gives a round stationary cloud instead of D H^-1;
  8. the path weight: downhill along gradient flow costs nothing, uphill
     costs dU / D;
  9. sampling: the unadjusted Langevin algorithm's step-size bias
     1 / (1 - eta/2), the Metropolis-adjusted version (MALA) without it, and
     SGLD's extra temperature from mini-batch noise;
 10. SGD as Langevin: D = eta sigma^2 / (2B);
 11. annealed Langevin with the exact score of a Gaussian mixture: plain
     Langevin gets the mode weights wrong, annealing gets them right.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only.

Run:  python3 langevin.py            (checks)
      python3 langevin.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)


def trapz(y, x):
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2)


def cumtrapz(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum((y[1:] + y[:-1]) * np.diff(x) / 2)
    return out


def expm_sym(S, t):
    """exp(t S) for a symmetric matrix S."""
    lam, V = np.linalg.eigh(S)
    return (V * np.exp(t * lam)) @ V.T


# ------------------------------------------------------------------ potentials

TILT = 0.25


def dw_U(w, c=TILT):
    """Tilted double well: the deeper well is on the left."""
    return (w ** 2 - 1) ** 2 + c * w


def dw_Up(w, c=TILT):
    return 4 * w * (w ** 2 - 1) + c


def dw_Upp(w):
    return 12 * w ** 2 - 4


def dw_crit(c=TILT):
    r = np.sort(np.roots([4, 0, -4, c]).real)
    return r[0], r[1], r[2]          # left minimum, saddle, right minimum


def gibbs(U, x, D):
    g = np.exp(-(U - U.min()) / D)
    return g / trapz(g, x)


def langevin_1d(Up, w0, D, dt, steps, rng, record_every=0, clip=3.0):
    w = np.array(w0, dtype=float)
    rec = []
    s = math.sqrt(2 * D * dt)
    for t in range(steps):
        w = w - Up(w) * dt + s * rng.normal(size=w.shape)
        np.clip(w, -clip, clip, out=w)
        if record_every and t % record_every == 0:
            rec.append(w.copy())
    return w, rec


# ------------------------------------------------------------------ 1. Brownian motion

def check_brownian():
    print("1. Brownian motion: variance grows like 2 D t")
    rng = np.random.default_rng(1)
    D = 0.5
    for eta in (0.1, 0.01):
        n = 20000
        w = np.zeros(n)
        out = {}
        for k in range(1, int(round(16 / eta)) + 1):
            w += math.sqrt(2 * D * eta) * rng.normal(size=n)
            t = k * eta
            for T in (1, 4, 16):
                if abs(t - T) < eta / 2:
                    out[T] = w.var()
        print(f"   step {eta}: Var(W_t) / (2 D t) at t = 1, 4, 16: "
              + ", ".join(f"{out[T] / (2 * D * T):.3f}" for T in (1, 4, 16)))
    # a kick of size eta (not sqrt(eta)) per step: the spread vanishes as the step shrinks
    for eta in (0.1, 0.01, 0.001):
        n = 4000
        steps = int(round(1 / eta))
        w = (2 * D * eta * rng.normal(size=(n, steps))).sum(axis=1) if steps <= 1000 else None
        if w is not None:
            print(f"   kicks of size 2D*eta instead of sqrt(2D*eta), step {eta}: Var(W_1) = {w.var():.4f}")


# ------------------------------------------------------------------ 2. OU and equipartition

def check_ou():
    print("\n2. Ornstein-Uhlenbeck: stationary variance D/h, equipartition")
    rng = np.random.default_rng(2)
    D, h, dt = 0.2, 2.0, 0.01
    w = np.full(4000, 3.0)
    acc = []
    for t in range(3000):
        w = w - h * w * dt + math.sqrt(2 * D * dt) * rng.normal(size=w.size)
        if t > 1000 and t % 10 == 0:
            acc.append(w.copy())
    v = np.var(np.array(acc))
    print(f"   h = {h}, D = {D}, Euler-Maruyama dt = {dt}: stationary Var(w) = {v:.4f};"
          f" D/h = {D / h:.4f}; with the step-size factor 1/(1 - h dt/2): {D / h / (1 - h * dt / 2):.4f}")
    k = 50
    hs = np.geomspace(0.1, 10, k)
    chains, steps, tau = 400, 400, 0.05
    W = np.zeros((chains, k))
    E = []
    for t in range(steps):                          # exact OU transitions, all directions at once
        e = np.exp(-hs * tau)
        W = W * e + np.sqrt(D / hs * (1 - e ** 2)) * rng.normal(size=W.shape)
    for t in range(steps):
        e = np.exp(-hs * tau)
        W = W * e + np.sqrt(D / hs * (1 - e ** 2)) * rng.normal(size=W.shape)
        E.append(0.5 * hs * W ** 2)
    E = np.array(E).mean(axis=(0, 1))
    print(f"   {k} directions with curvatures 0.1 .. 10: mean energy in the flattest {E[0]:.4f},"
          f" in the stiffest {E[-1]:.4f} (D/2 = {D / 2}); total {E.sum():.3f} (k D / 2 = {k * D / 2})")
    print(f"   spread: std of w in the flattest direction {math.sqrt(D / hs[0]):.3f}, stiffest {math.sqrt(D / hs[-1]):.3f}")


# ------------------------------------------------------------------ 3. Gibbs law on a double well

def check_gibbs():
    print("\n3. Where the particles end up: the Gibbs law exp(-U/D)/Z")
    rng = np.random.default_rng(3)
    x = np.linspace(-2.2, 2.2, 4401)
    U = dw_U(x)
    wl, ws, wr = dw_crit()
    print(f"   U = (w^2 - 1)^2 + {TILT} w: minima {wl:.3f} (U = {dw_U(wl):.3f}), {wr:.3f} (U = {dw_U(wr):.3f});"
          f" saddle {ws:.3f} (U = {dw_U(ws):.3f})")
    rows = []
    for D in (0.1, 0.3, 1.0):
        p = gibbs(U, x, D)
        rows.append((D, trapz(np.where(x < ws, p, 0), x)))
    print("   P(left well) under Gibbs: " + ", ".join(f"D = {D}: {P:.3f}" for D, P in rows))
    D = 0.4
    w0 = rng.uniform(-1.5, 1.5, size=3000)
    w, rec = langevin_1d(dw_Up, w0, D, 0.005, 40000, rng, record_every=40)
    samples = np.concatenate(rec[len(rec) // 2:])
    edges = np.linspace(-2.2, 2.2, 45)
    hist, _ = np.histogram(samples, bins=edges, density=True)
    mids = (edges[1:] + edges[:-1]) / 2
    p = gibbs(U, x, D)
    pb = np.array([trapz(np.where((x >= a) & (x <= b), p, 0), x) / (b - a) for a, b in zip(edges[:-1], edges[1:])])
    tv = 0.5 * np.sum(np.abs(hist - pb) * np.diff(edges))
    print(f"   Langevin, D = {D}, 3000 particles, 200 time units: P(left) = {np.mean(samples < ws):.3f}"
          f" (Gibbs {trapz(np.where(x < ws, p, 0), x):.3f}); total-variation distance of the histogram {tv:.3f}")
    return mids, hist, D


# ------------------------------------------------------------------ 4. depth against width

DW = dict(m1=-1.5, h1=25.0, a1=-0.3, m2=1.5, h2=1.0, a2=0.0)


def dwidth_U(w, p=DW):
    e1 = -(p["a1"] + p["h1"] * (w - p["m1"]) ** 2 / 2)
    e2 = -(p["a2"] + p["h2"] * (w - p["m2"]) ** 2 / 2)
    m = np.maximum(e1, e2)
    return -(m + np.log(np.exp(e1 - m) + np.exp(e2 - m)))


def p_narrow(D, x=np.linspace(-5, 7, 24001)):
    U = dwidth_U(x)
    g = np.exp(-(U - U.min()) / D)
    return trapz(np.where(x < 0, g, 0), x) / trapz(g, x)


def p_narrow_laplace(D, p=DW):
    r = math.exp(-(p["a1"] - p["a2"]) / D) * math.sqrt(p["h2"] / p["h1"])
    return r / (1 + r)


def p_narrow_depth(D, p=DW):
    r = math.exp(-(p["a1"] - p["a2"]) / D)
    return r / (1 + r)


def check_depth_width():
    print("\n4. Depth against width: the free energy U + (D/2) log h")
    p = DW
    print(f"   narrow well at {p['m1']} (curvature {p['h1']:.0f}, depth {p['a1']}), wide well at {p['m2']}"
          f" (curvature {p['h2']:.0f}, depth {p['a2']})")
    print("   D      P(narrow) exact   Laplace e^(-dU/D) sqrt(h2/h1)   depth only e^(-dU/D)")
    for D in (0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"   {D:4.2f}   {p_narrow(D):.3f}             {p_narrow_laplace(D):.3f}"
              f"                          {p_narrow_depth(D):.3f}")
    lo, hi = 0.05, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_narrow(mid) > 0.5 else (lo, mid)
    print(f"   the narrow well loses its majority at D = {lo:.3f}; the Laplace rule says"
          f" dU / log sqrt(h1/h2) = {0.3 / math.log(math.sqrt(p['h1'] / p['h2'])):.3f}")
    x = np.linspace(-3.5, 4.5, 80001)
    U = dwidth_U(x)
    ridge = U[(x > p["m1"]) & (x < p["m2"])].max()
    esc = {D: mfpt(U, x, D, p["m1"], p["m2"]) for D in (0.3, 0.8)}
    print(f"   escape from the narrow well over its barrier ({ridge - p['a1']:.2f} high): mean time {esc[0.3]:.0f} at D = 0.3,"
          f" {esc[0.8]:.1f} at D = 0.8 (Gibbs P(narrow) = {p_narrow(0.3):.3f} and {p_narrow(0.8):.3f})")


# ------------------------------------------------------------------ 5. Fokker-Planck on a grid

def grid_generator(U, D, h):
    """Reversible nearest-neighbour jump rates on a grid: stationary law exp(-U/D) exactly,
    drift -U' and diffusion D in the limit h -> 0. Returns the symmetrised generator."""
    n = U.size
    up = D / h ** 2 * np.exp(-(U[1:] - U[:-1]) / (2 * D))     # i -> i+1
    dn = D / h ** 2 * np.exp(-(U[:-1] - U[1:]) / (2 * D))     # i+1 -> i
    Q = np.zeros((n, n))
    Q[np.arange(n - 1), np.arange(1, n)] = up
    Q[np.arange(1, n), np.arange(n - 1)] = dn
    Q[np.arange(n), np.arange(n)] = -Q.sum(axis=1)
    pi = np.exp(-(U - U.min()) / D)
    pi /= pi.sum()
    s = np.sqrt(pi)
    S = (s[:, None] * Q) / s[None, :]
    return S, pi, Q


def mfpt(U, x, D, a, b):
    """Exact mean first-passage time from a to b under dw = -U' dt + sqrt(2D) dW on the grid x
    (reflecting at the end of x behind a, absorbing at b)."""
    if b > a:
        m = (x >= a) & (x <= b)
        inner = cumtrapz(np.exp(-(U - U.min()) / D), x)
        return trapz(np.exp((U[m] - U.min()) / D) * inner[m], x[m]) / D
    xr, Ur = -x[::-1], U[::-1]
    return mfpt(Ur, xr, D, -a, -b)


def check_fokker_planck():
    print("\n5. The Fokker-Planck equation: how the density moves")
    D, h = 0.25, 0.01
    x = np.arange(-2.2, 2.2 + h / 2, h)
    U = dw_U(x)
    S, pi, Q = grid_generator(U, D, h)
    lam = np.sort(-np.linalg.eigvalsh(S))
    wl, ws, wr = dw_crit()
    t_lr = mfpt(U, x, D, wl, wr)
    t_rl = mfpt(U, x, D, wr, wl)
    print(f"   D = {D}: slowest decay rates of the density (eigenvalues): {lam[0]:.1e}, {lam[1]:.5f}, {lam[2]:.3f}, {lam[3]:.3f}")
    print(f"   exact mean passage times: left -> right {t_lr:.1f}, right -> left {t_rl:.2f};"
          f" 1/tau_lr + 1/tau_rl = {1 / t_lr + 1 / t_rl:.5f} (vs lambda_1 = {lam[1]:.5f})")
    print(f"   curvatures: U''(left) = {dw_Upp(wl):.2f}, U''(right) = {dw_Upp(wr):.2f}, |U''(saddle)| = {abs(dw_Upp(ws)):.2f};"
          f" every rate after lambda_1 is of that order: a gap of {lam[2] / lam[1]:.0f}x")
    xq = np.arange(-2, 2 + h / 2, h)
    Sq, _, _ = grid_generator(0.5 * 8 * xq ** 2, D, h)
    print(f"   sanity check on a quadratic U = 4 w^2 (curvature 8): rates {np.round(np.sort(-np.linalg.eigvalsh(Sq))[1:4], 3).tolist()}"
          " (exact 8, 16, 24)")
    Ds, hs_ = 0.05, 0.005
    xs_ = np.arange(-2.2, 2.2 + hs_ / 2, hs_)
    Ss, _, _ = grid_generator(dw_U(xs_), Ds, hs_)
    ls = np.sort(-np.linalg.eigvalsh(Ss))
    near = ls[np.argmin(np.abs(ls - dw_Upp(wr)))]
    print(f"   at D = {Ds} the fastest-but-one rate is {ls[2]:.2f}, a mode on the saddle (|U''(s)| = {abs(dw_Upp(ws)):.2f}),"
          f" and there is a rate {near:.2f} beside the right well's curvature {dw_Upp(wr):.2f}")
    pi_c = pi / h
    i0 = np.argmin(np.abs(x - 1.0))
    p0 = np.zeros_like(x); p0[i0] = 1.0
    s = np.sqrt(pi)
    snaps = {}
    for t in (0.02, 0.3, 3.0, 30.0, 300.0):
        pt = s * (expm_sym(S, t) @ (p0 / s))
        snaps[t] = pt / h
        print(f"   start at w = +1: t = {t:6.2f}: mass in the left well {abs(pt[x < ws].sum()):.3f}"
              f" (equilibrium {pi[x < ws].sum():.3f}); max |p - Gibbs| = {np.abs(pt / h - pi_c).max():.3f}")
    return x, snaps, pi_c, lam


# ------------------------------------------------------------------ 6. Kramers

def sym_U(w):
    return (w ** 2 - 1) ** 2


def sym_Up(w):
    return 4 * w * (w ** 2 - 1)


def kramers_tau(D):
    return 2 * math.pi / math.sqrt(8 * 4) * math.exp(1.0 / D)


def check_kramers():
    print("\n6. Waiting to cross: Kramers' law on U = (w^2 - 1)^2 (barrier 1)")
    x = np.linspace(-3, 1, 40001)
    U = sym_U(x)
    rows = []
    for D in (0.1, 0.15, 0.2, 0.25, 0.35, 0.5):
        rows.append((D, mfpt(U, x, D, -1.0, 1.0), kramers_tau(D)))
    for D, ex, kr in rows:
        print(f"   D = {D:4.2f}: exact tau = {ex:10.2f}, Kramers 2 pi / sqrt(U''(a)|U''(s)|) e^(1/D) = {kr:10.2f},"
              f" ratio {ex / kr:.3f}")
    rng = np.random.default_rng(6)
    sims = []
    for D in (0.35, 0.5):
        n, dt = 4000, 0.002
        w = np.full(n, -1.0)
        hit = np.full(n, np.nan)
        t = 0.0
        s = math.sqrt(2 * D * dt)
        while np.isnan(hit).any() and t < 400:
            w = w - sym_Up(w) * dt + s * rng.normal(size=n)
            t += dt
            new = np.isnan(hit) & (w >= 1.0)
            hit[new] = t
        sims.append((D, np.nanmean(hit)))
        print(f"   simulation, D = {D}: mean first-passage time over {n} particles {np.nanmean(hit):.2f}"
              f" +- {np.nanstd(hit) / math.sqrt(n):.2f} (exact {mfpt(U, x, D, -1.0, 1.0):.2f})")
    slope = (math.log(rows[0][1]) - math.log(rows[2][1])) / (1 / rows[0][0] - 1 / rows[2][0])
    print(f"   slope of ln tau against 1/D between D = 0.1 and 0.2: {slope:.3f} (the barrier is 1)")
    return rows, sims


# ------------------------------------------------------------------ 7. detailed balance and when it fails

def lyapunov(A, C):
    """Solve A S + S A^T = 2 C for the stationary covariance of dw = -A w dt + sqrt(2C) dW."""
    k = A.shape[0]
    I = np.eye(k)
    M = np.kron(I, A) + np.kron(A, I)
    return np.linalg.solve(M, (2 * C).reshape(-1)).reshape(k, k)


def check_reversibility():
    print("\n7. Detailed balance, and two ways to break it")
    D, Om = 1.0, 2.0
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    A = np.eye(2) - Om * J                          # drift -(w) + Om J w: a gradient part plus a rotation
    S = lyapunov(A, D * np.eye(2))
    lam, V = np.linalg.eig(-A)
    s = 0.5
    E = (V @ np.diag(np.exp(lam * s)) @ np.linalg.inv(V)).real   # exp(-A s)
    C = E @ S                                                    # E[w(t+s) w(t)^T]
    print(f"   rotating drift, Omega = {Om}: stationary covariance {np.round(S, 6).tolist()} = D I, the Gibbs law of |w|^2/2")
    print(f"   but E[w1(t+s) w2(t)] = {C[0, 1]:+.4f} and E[w2(t+s) w1(t)] = {C[1, 0]:+.4f} at s = {s}:"
          " the movie is not the same played backwards")
    th = 0.5
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    H = R @ np.diag([10.0, 0.1]) @ R.T
    Dbar = 0.05
    S_iso = lyapunov(H, Dbar * np.eye(2))
    c = Dbar * 2 / np.trace(H)                       # noise covariance c H with the same total size tr = 2 Dbar
    S_hess = lyapunov(H, c * H)
    ev_iso = np.linalg.eigvalsh(S_iso)
    ev_hess = np.linalg.eigvalsh(S_hess)
    print(f"   quadratic loss with curvatures 10 and 0.1: isotropic noise D = {Dbar} gives stationary spreads"
          f" (variances) {ev_iso[1]:.4f}, {ev_iso[0]:.4f} = D/h (Gibbs)")
    print(f"   noise shaped like the Hessian, c H with the same trace: {ev_hess[1]:.5f}, {ev_hess[0]:.5f} = c I, c = {c:.5f}"
          " (a round cloud)")
    H10 = 10 * H
    ex_iso = [0.5 * np.trace(M @ lyapunov(M, Dbar * np.eye(2))) for M in (H, H10)]
    ex_hes = [0.5 * np.trace(M @ lyapunov(M, c * M)) for M in (H, H10)]
    print(f"   expected excess loss tr(H Sigma)/2 at this minimum and at one 10x sharper: isotropic"
          f" {ex_iso[0]:.4f} -> {ex_iso[1]:.4f} (k D / 2 either way); Hessian-shaped {ex_hes[0]:.4f} -> {ex_hes[1]:.4f} (c tr H / 2)")
    return H, S_iso, S_hess


# ------------------------------------------------------------------ 8. the path weight

def check_paths():
    print("\n8. The weight of a path: downhill is free, uphill costs dU/D")
    wl, ws, wr = dw_crit()
    dt = 1e-4
    w = ws - 1e-3
    path = [w]
    while abs(w - wl) > 1e-3:
        k1 = -dw_Up(w); k2 = -dw_Up(w + dt / 2 * k1); k3 = -dw_Up(w + dt / 2 * k2); k4 = -dw_Up(w + dt * k3)
        w += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        path.append(w)
    path = np.array(path)
    t = np.arange(path.size) * dt
    wd = np.gradient(path, dt)
    down = trapz((wd + dw_Up(path)) ** 2, t) / 4       # (1/4D) int |w' + U'|^2 dt, times D
    up = trapz((-wd + dw_Up(path)) ** 2, t) / 4        # the same path run backwards: velocity -w'
    dU = dw_U(path[0]) - dw_U(path[-1])
    print(f"   saddle -> left well along gradient descent: D x action = {down:.2e}")
    print(f"   left well -> saddle along the reversed path: D x action = {up:.4f}; climb dU = {dU:.4f}")
    x = np.linspace(wl, ws, 4001)
    L = ws - wl
    best = L * math.sqrt(trapz(dw_Up(x) ** 2, x) / L)
    straight = (best + dU) / 2                        # D x (1/4D) int |v + U'|^2 = (1/4)(L^2/T + T mean U'^2) + dU/2
    print(f"   the best straight constant-speed climb: D x action = {straight:.4f} > {up:.4f}")


# ------------------------------------------------------------------ 9. sampling: ULA, MALA, SGLD

ETAS = (0.05, 0.2, 0.5, 1.0, 1.5, 1.9)


def run_ula_mala(eta, rng, chains=2000, steps=3000, burn=500):
    """Target N(0, 1): U = w^2/2, D = 1."""
    w_u = rng.normal(size=chains)
    w_m = rng.normal(size=chains)
    su, sm, acc = [], [], 0
    s = math.sqrt(2 * eta)
    for t in range(steps):
        w_u = w_u - eta * w_u + s * rng.normal(size=chains)
        prop = w_m - eta * w_m + s * rng.normal(size=chains)
        # log pi(prop) + log q(w | prop) - log pi(w) - log q(prop | w)
        lq_back = -((w_m - (prop - eta * prop)) ** 2) / (4 * eta)
        lq_fwd = -((prop - (w_m - eta * w_m)) ** 2) / (4 * eta)
        la = -0.5 * prop ** 2 + lq_back + 0.5 * w_m ** 2 - lq_fwd
        ok = np.log(rng.random(chains)) < la
        w_m = np.where(ok, prop, w_m)
        if t >= burn:
            su.append(w_u.copy()); sm.append(w_m.copy()); acc += ok.sum()
    return np.var(np.array(su)), np.var(np.array(sm)), acc / (chains * (steps - burn))


def check_samplers():
    print("\n9. Sampling with Langevin: the step-size bias and how to remove it (target N(0, 1))")
    rng = np.random.default_rng(9)
    rows = []
    for eta in ETAS:
        vu, vm, a = run_ula_mala(eta, rng)
        rows.append((eta, vu, vm, a))
        print(f"   eta = {eta:4.2f}: ULA variance {vu:.3f} (formula 1/(1 - eta/2) = {1 / (1 - eta / 2):.3f});"
              f" MALA variance {vm:.3f}, acceptance {a:.3f}")
    print("   eta = 2 and beyond: ULA's update w <- (1 - eta) w + noise no longer contracts; the chain diverges")
    # SGLD on the mean of N = 1000 Gaussian observations
    N, B, prior = 1000, 10, 10.0
    y = rng.normal(1.0, 1.0, size=N)
    Upp = N + 1 / prior ** 2
    v_post = 1 / Upp
    g2 = N ** 2 / B * y.var()                          # variance of the mini-batch gradient estimate
    print(f"   SGLD, N = {N}, batch {B}: exact posterior variance {v_post:.3e}; mini-batch gradient variance {g2:.3g}")
    for eta in (3e-4, 1e-4, 1e-5, 1e-6):
        a = 1 - eta * Upp
        v = (2 * eta + eta ** 2 * g2) / (1 - a ** 2)
        print(f"     step {eta:.0e}: stationary variance / posterior variance = {v / v_post:.3f};"
              f" effective temperature 1 + eta g^2 / 2 = {1 + eta * g2 / 2:.3f}")
    eta = 1e-4
    th = np.zeros(200)
    acc = []
    m_post = y.sum() / Upp
    for t in range(6000):
        idx = rng.integers(0, N, size=(200, B))
        grad = th / prior ** 2 + N / B * (th[:, None] - y[idx]).sum(axis=1)
        th = th - eta * grad + math.sqrt(2 * eta) * rng.normal(size=200)
        if t > 1000:
            acc.append(th.copy())
    acc = np.array(acc)
    print(f"     simulation at step 1e-04: variance ratio {acc.var() / v_post:.3f}, mean {acc.mean():.4f}"
          f" (posterior mean {m_post:.4f})")
    return rows


# ------------------------------------------------------------------ 10. SGD as Langevin

def check_sgd():
    print("\n10. SGD as Langevin: D = eta sigma^2 / (2B)")
    rng = np.random.default_rng(10)
    z = rng.normal(size=5000) * 1.5
    sig2 = z.var()
    eta, chains, steps, burn = 0.05, 100, 3000, 300
    for B in (4, 16, 64):
        w = np.zeros(chains)
        acc = []
        for t in range(steps):
            w = w - eta * (w - z[rng.integers(0, z.size, size=(chains, B))].mean(axis=1))
            if t >= burn:
                acc.append(w - z.mean())
        print(f"   loss (1/2)(w - z_i)^2, eta = {eta}, B = {B:2d}: Var(w) = {np.var(np.array(acc)):.5f};"
              f" eta sigma^2 / (2B) = {eta * sig2 / (2 * B):.5f}")


# ------------------------------------------------------------------ 11. annealed Langevin

MIX = dict(w=(0.8, 0.2), mu=(-4.0, 4.0), s=1.0)


def mix_score(x, sigma):
    """Exact score d/dx log p_sigma of the mixture smoothed by N(0, sigma^2)."""
    v = MIX["s"] ** 2 + sigma ** 2
    comps = [wk * np.exp(-(x - mk) ** 2 / (2 * v)) for wk, mk in zip(MIX["w"], MIX["mu"])]
    tot = comps[0] + comps[1] + 1e-300
    return sum(c * (mk - x) / v for c, mk in zip(comps, MIX["mu"])) / tot


def mix_pdf(x, sigma=0.0):
    v = MIX["s"] ** 2 + sigma ** 2
    return sum(wk * np.exp(-(x - mk) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v) for wk, mk in zip(MIX["w"], MIX["mu"]))


SIGMAS = np.geomspace(10, 0.1, 10)


def plain_langevin(rng, n=5000, eta=0.01, steps=2000):
    x = rng.uniform(-8, 8, size=n)
    for _ in range(steps):
        x = x + eta * mix_score(x, 0.0) + math.sqrt(2 * eta) * rng.normal(size=n)
    return x


def annealed_langevin(rng, n=5000, steps=200, eps=0.1):
    x = rng.uniform(-8, 8, size=n)
    for sg in list(SIGMAS) + [0.0]:
        eta = eps * (MIX["s"] ** 2 + sg ** 2)
        for _ in range(steps):
            x = x + eta * mix_score(x, sg) + math.sqrt(2 * eta) * rng.normal(size=n)
    return x


def check_annealed():
    print("\n11. Score-based sampling: plain against annealed Langevin on 0.8 N(-4, 1) + 0.2 N(4, 1)")
    rng = np.random.default_rng(11)
    xp = plain_langevin(rng)
    xa = annealed_langevin(rng)
    print(f"   plain Langevin from uniform[-8, 8], 2000 steps of 0.01: weight of the left mode {np.mean(xp < 0):.3f}")
    print(f"   annealed, sigma from 10 down to 0.1 then 0 (200 steps each): weight of the left mode {np.mean(xa < 0):.3f}")
    print(f"   true weight {MIX['w'][0]}; the barrier of -log p between the modes is"
          f" {-math.log(mix_pdf(np.array(0.0))) + math.log(mix_pdf(np.array(4.0))):.2f} from the small mode"
          f" (so crossings take ~ e^{-math.log(mix_pdf(np.array(0.0))) + math.log(mix_pdf(np.array(4.0))):.1f} time units)")
    return xp, xa


# ------------------------------------------------------------------ figures

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:10.5px;fill:#57564f} .v{font-size:11px;fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s4{stroke:#eda100} .s0{stroke:#8a8880}
  .f1{fill:#2a78d6} .f2{fill:#eb6834} .f3{fill:#1baf7a} .f4{fill:#eda100} .f0{fill:#8a8880}
  .ln{stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
  .thin{stroke-width:1.1;fill:none;stroke-linejoin:round;opacity:.85}
  .dash{stroke-width:1.4;fill:none;stroke-dasharray:5 3}
  .band{opacity:.14} .bar{opacity:.35}
  .ring{stroke:#fdfdfc;stroke-width:1.5}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500} .s0{stroke:#85837b}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500} .f0{fill:#85837b}
    .ring{stroke:#161615} .band{opacity:.22}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


def fmt(v):
    return f"{v:g}".replace("-", "−")


class Panel:
    def __init__(self, body, x0, y0, w, h, xr, yr, logy=False):
        self.b, self.x0, self.y0, self.w, self.h, self.xr, self.yr, self.logy = body, x0, y0, w, h, xr, yr, logy

    def X(self, x):
        return self.x0 + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def Y(self, y):
        lo, hi = self.yr
        if self.logy:
            y, lo, hi = math.log10(max(y, 1e-300)), math.log10(lo), math.log10(hi)
        return self.y0 + self.h - (y - lo) / (hi - lo) * self.h

    def frame(self, xt, yt, xlab, ylab, title, fx=fmt, fy=fmt):
        b = self.b
        for y in yt:
            b.append(f'<line class="gd" x1="{self.x0}" y1="{self.Y(y):.1f}" x2="{self.x0 + self.w}" y2="{self.Y(y):.1f}"/>')
            b.append(f'<text class="sm" x="{self.x0 - 6}" y="{self.Y(y) + 3.5:.1f}" text-anchor="end">{fy(y)}</text>')
        for x in xt:
            b.append(f'<text class="sm" x="{self.X(x):.1f}" y="{self.y0 + self.h + 15}" text-anchor="middle">{fx(x)}</text>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0 + self.h}" x2="{self.x0 + self.w}" y2="{self.y0 + self.h}"/>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0}" x2="{self.x0}" y2="{self.y0 + self.h}"/>')
        b.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 32}" text-anchor="middle">{xlab}</text>')
        b.append(f'<text class="lab" transform="translate({self.x0 - 40},{self.y0 + self.h / 2}) rotate(-90)" text-anchor="middle">{ylab}</text>')
        b.append(f'<text class="hd" x="{self.x0}" y="{self.y0 - 12}">{title}</text>')

    def line(self, xs, ys, cls):
        """A polyline, cut into pieces wherever it leaves the frame (never flattened along an edge)."""
        segs, cur = [], []
        for x, y in zip(xs, ys):
            inside = (self.xr[0] - 1e-9 <= x <= self.xr[1] + 1e-9 and (y > 0 or not self.logy)
                      and self.y0 - 1 <= self.Y(y) <= self.y0 + self.h + 1)
            if inside:
                cur.append(f"{self.X(x):.1f},{self.Y(y):.1f}")
            elif cur:
                segs.append(cur); cur = []
        if cur:
            segs.append(cur)
        for seg in segs:
            if len(seg) > 1:
                self.b.append(f'<polyline class="{cls}" points="{" ".join(seg)}"/>')

    def area(self, xs, lo, hi, cls):
        up = [f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, hi)]
        dn = [f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs[::-1], lo[::-1])]
        self.b.append(f'<polygon class="{cls}" points="{" ".join(up + dn)}"/>')

    def dot(self, x, y, cls, r=4.5):
        self.b.append(f'<circle class="{cls} ring" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>')

    def bars(self, edges, vals, cls):
        for a, b_, v in zip(edges[:-1], edges[1:], vals):
            y0, y1 = self.Y(self.yr[0]), self.Y(min(v, self.yr[1]))
            self.b.append(f'<rect class="{cls}" x="{self.X(a) + 0.5:.1f}" y="{y1:.1f}" width="{max(self.X(b_) - self.X(a) - 1, 0.5):.1f}" height="{max(y0 - y1, 0):.1f}"/>')

    def text(self, x, y, s, cls="v", anchor="start", dx=0, dy=0):
        self.b.append(f'<text class="{cls}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" text-anchor="{anchor}">{s}</text>')


def fig_paths(out):
    rng = np.random.default_rng(21)
    W, H = 920, 320
    body = []
    D, dt, T = 0.5, 0.01, 8.0
    t = np.arange(0, T + dt / 2, dt)
    L = Panel(body, 70, 45, 330, 220, (0, 8), (-7, 7))
    L.frame([0, 2, 4, 6, 8], [-6, -4, -2, 0, 2, 4, 6], "time t", "w", "Pure noise: a random walk spreads like √t")
    env = 2 * np.sqrt(2 * D * t)
    L.area(t, -env, env, "f1 band")
    for k in range(6):
        w = np.concatenate([[0], np.cumsum(math.sqrt(2 * D * dt) * rng.normal(size=t.size - 1))])
        L.line(t, w, f"thin {'s1' if k % 2 == 0 else 's0'}")
    L.text(8, env[-1], "±2√(2Dt)", "sm", "end", -4, -6)
    R = Panel(body, 530, 45, 350, 220, (0, 8), (-2, 4))
    R.frame([0, 2, 4, 6, 8], [-2, -1, 0, 1, 2, 3, 4], "time t", "w", "Noise plus a pull to 0: it settles in a band")
    h = 1.0
    sd = np.sqrt(D / h * (1 - np.exp(-2 * h * t)))
    mean = 3 * np.exp(-h * t)
    R.area(t, mean - 2 * sd, mean + 2 * sd, "f2 band")
    R.line(t, mean, "dash s0")
    for k in range(6):
        w = [3.0]
        for _ in range(t.size - 1):
            w.append(w[-1] - h * w[-1] * dt + math.sqrt(2 * D * dt) * rng.normal())
        R.line(t, w, f"thin {'s2' if k % 2 == 0 else 's0'}")
    R.text(8, 2 * math.sqrt(D / h), "±2√(D/h)", "sm", "end", -4, -6)
    R.text(1.2, 3 * math.exp(-1.2), "gradient flow 3e^(−ht)", "sm", "start", 6, -6)
    desc = ("Left: six Brownian paths with D = 0.5 from w = 0, inside the band ±2√(2Dt), which keeps widening. "
            "Right: six Ornstein–Uhlenbeck paths, dw = −w dt + √(2D) dW, from w = 3: they follow the gradient-flow curve "
            "3e^(−t) down and then jitter inside a band of fixed width ±2√(D/h) around 0.")
    out.write_text(svg(W, H, "Random walk and Langevin paths", desc, body))


def fig_gibbs(out, mids, hist, Dsim):
    W, H = 920, 320
    body = []
    x = np.linspace(-2.0, 2.0, 801)
    U = dw_U(x)
    L = Panel(body, 70, 45, 330, 220, (-2, 2), (-0.5, 2.0))
    L.frame([-2, -1, 0, 1, 2], [-0.5, 0, 0.5, 1, 1.5, 2], "w", "U(w)", f"The potential U = (w² − 1)² + {TILT}w")
    L.line(x, U, "ln s0")
    R = Panel(body, 530, 45, 350, 220, (-2, 2), (0, 4.2))
    R.frame([-2, -1, 0, 1, 2], [0, 1, 2, 3, 4], "w", "density", "The Gibbs law e^(−U/D) / Z at three temperatures")
    edges = np.concatenate([[mids[0] - (mids[1] - mids[0]) / 2], (mids[1:] + mids[:-1]) / 2, [mids[-1] + (mids[1] - mids[0]) / 2]])
    R.bars(edges, hist, "f2 bar")
    for D, cls, lx in ((0.1, "s1", -1.03), (0.4, "s2", -0.85), (1.0, "s3", 0.0)):
        p = gibbs(U, x, D)
        R.line(x, p, f"ln {cls}")
        i = np.argmin(np.abs(x - lx))
        R.text(x[i], p[i], f"D = {D}", "sm", "start", 8, -6)
    R.text(2, 1.6, f"bars: Langevin, D = {Dsim}", "sm", "end", 0, 0)
    desc = ("Left: a tilted double well whose left well is deeper. Right: the stationary density e^(−U/D)/Z for D = 0.1 "
            "(almost all mass in the deeper well), 0.4 and 1.0 (spread over both wells), with the histogram of a Langevin "
            "simulation at D = 0.4 as bars, sitting on the D = 0.4 curve.")
    out.write_text(svg(W, H, "Gibbs law at three temperatures", desc, body))


def fig_depth_width(out):
    W, H = 920, 320
    body = []
    x = np.linspace(-3.5, 4.5, 1601)
    L = Panel(body, 70, 45, 330, 220, (-3.5, 4.5), (-0.5, 3.0))
    L.frame([-3, -2, -1, 0, 1, 2, 3, 4], [-0.5, 0, 1, 2, 3], "w", "U(w)", "A deep narrow well and a shallow wide one")
    L.line(x, dwidth_U(x), "ln s0")
    L.text(DW["m1"], DW["a1"], "deeper, 25× stiffer", "sm", "end", -10, 4)
    L.text(DW["m2"], DW["a2"], "wide", "sm", "middle", 0, 16)
    R = Panel(body, 530, 45, 350, 220, (0, 0.6), (0, 1))
    R.frame([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 0.25, 0.5, 0.75, 1], "temperature D", "P(narrow well)",
            "Who wins: depth at low D, width at high D")
    Ds = np.linspace(0.01, 0.6, 90)
    R.line(Ds, [p_narrow_depth(D) for D in Ds], "dash s0")
    R.line(Ds, [p_narrow_laplace(D) for D in Ds], "dash s2")
    R.line(Ds, [p_narrow(D) for D in Ds], "ln s1")
    R.text(0.6, p_narrow_depth(0.6), "depth only, e^(−ΔU/D)", "sm", "end", 0, -8)
    R.text(0.6, p_narrow(0.6), "exact", "sm", "end", 0, -8)
    R.text(0.12, 0.16, "orange dashed: free energy U + (D/2) log h", "sm", "start", 0, 0)
    desc = ("Left: a potential with a narrow well of curvature 25 that is 0.3 deeper and a wide well of curvature 1. "
            "Right: the Gibbs probability of the narrow well against D. Counting depth only, it would keep a majority at every "
            "temperature; exactly (solid) and by the free-energy rule (dashed orange) it drops below one half near D = 0.19 "
            "and towards 1/6 as D grows.")
    out.write_text(svg(W, H, "Depth against width", desc, body))


def fig_fp(out, x, snaps, pi_c):
    W, H = 920, 320
    body = []
    P = Panel(body, 90, 45, 780, 210, (-2, 2), (0, 4.5))
    P.frame([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], [0, 1, 2, 3, 4], "w", "density p(w, t)",
            "The density from a start at w = +1 (Fokker–Planck, D = 0.25)")
    cls = {0.02: "s0", 0.3: "s4", 3.0: "s3", 30.0: "s2", 300.0: "s1"}
    for t, p in snaps.items():
        m = np.abs(x) <= 2
        P.line(x[m], p[m], f"ln {cls[t]}")
        i = np.argmax(np.where(x > 0, p, 0))
        P.text(x[i], p[i], f"t = {fmt(t)}", "sm", "start", 12, 4)
    P.line(x, pi_c, "dash s0")
    i = np.argmax(pi_c)
    P.text(x[i], pi_c[i], "Gibbs", "sm", "end", -8, -4)
    desc = ("Snapshots of the density of a particle started at w = +1 in the tilted double well at D = 0.25: at t = 0.02 a "
            "narrow spike, by t = 0.3 settled into the right well's local shape, and only by t = 30 to 300 flowing into the "
            "deeper left well until it matches the Gibbs law (dashed). Two time scales: the fast one is the well's curvature, "
            "the slow one the Kramers rate over the barrier.")
    out.write_text(svg(W, H, "Two time scales of the Fokker–Planck equation", desc, body))


def fig_kramers(out, rows, sims):
    W, H = 920, 320
    body = []
    P = Panel(body, 90, 45, 780, 210, (1.5, 10.5), (1, 1e5), logy=True)
    P.frame([2, 4, 6, 8, 10], [1, 10, 100, 1e3, 1e4, 1e5], "1 / D", "mean crossing time τ (log scale)",
            "Kramers' law: ln τ grows like barrier / D", fx=fmt, fy=lambda v: f"{v:g}")
    inv = np.linspace(1.5, 10.5, 80)
    xg = np.linspace(-3, 1, 20001)
    P.line(inv, [mfpt(sym_U(xg), xg, 1 / i, -1.0, 1.0) for i in inv], "ln s1")
    P.line(inv, [kramers_tau(1 / i) for i in inv], "dash s2")
    for D, tsim in sims:
        P.dot(1 / D, tsim, "f2", 5)
    P.text(10.5, kramers_tau(1 / 10.5), "Kramers: (2π/√(U″(a)·|U″(s)|)) e^(ΔE/D), dashed", "sm", "end", 0, 18)
    P.text(6.5, mfpt(sym_U(xg), xg, 1 / 6.5, -1.0, 1.0), "exact first-passage time", "sm", "end", -8, -8)
    P.text(1 / sims[0][0], sims[0][1], "simulation", "sm", "start", 9, 14)
    desc = ("The mean time to cross from w = −1 to +1 in U = (w² − 1)², barrier 1, on a log scale against 1/D. The exact "
            "first-passage time (solid) becomes a straight line of slope 1, the barrier; Kramers' formula (dashed) matches it "
            "for small D and overestimates it slightly at D = 0.5; two simulations sit on the exact curve.")
    out.write_text(svg(W, H, "Kramers' law", desc, body))


def fig_samplers(out, rows):
    W, H = 920, 320
    body = []
    L = Panel(body, 70, 45, 330, 220, (0, 2), (0.8, 3.0))
    L.frame([0, 0.5, 1, 1.5, 2], [1, 1.5, 2, 2.5, 3], "step size η", "variance of the samples",
            "Sampling N(0, 1): the step-size bias")
    e = np.linspace(0, 1.95, 100)
    L.line(e, 1 / (1 - e / 2), "ln s2")
    L.line([0, 2], [1, 1], "dash s0")
    for eta, vu, vm, a in rows:
        L.dot(eta, vu, "f2", 4.5)
        L.dot(eta, vm, "f1", 4.5)
    L.text(1.6, 1 / (1 - 1.6 / 2), "ULA: 1/(1 − η/2)", "sm", "end", -8, -4)
    L.text(1.95, 1, "MALA", "sm", "end", 0, 16)
    R = Panel(body, 530, 45, 350, 220, (0, 2), (0, 1))
    R.frame([0, 0.5, 1, 1.5, 2], [0, 0.25, 0.5, 0.75, 1], "step size η", "acceptance rate",
            "MALA's price: rejected moves")
    R.line([r[0] for r in rows], [r[3] for r in rows], "ln s1")
    for eta, vu, vm, a in rows:
        R.dot(eta, a, "f1", 4.5)
    desc = ("Left: the variance of samples from the unadjusted Langevin algorithm (orange dots) on a standard normal target "
            "rises with the step size exactly as 1/(1 − η/2) and blows up at η = 2; the Metropolis-adjusted version (blue dots) "
            "stays at 1. Right: MALA's acceptance rate falls from near 1 at small steps as the step grows.")
    out.write_text(svg(W, H, "ULA bias and MALA", desc, body))


def fig_noise_shape(out, H_, S_iso, S_hess):
    rng = np.random.default_rng(23)
    W, Hh = 920, 330
    body = []
    for j, (S, cls, title) in enumerate(((S_iso, "f1", "Isotropic noise: Gibbs, spread D/h"),
                                         (S_hess, "f2", "Noise shaped like H: a round cloud"))):
        P = Panel(body, 70 + 460 * j, 45, 330, 230, (-1.6, 1.6), (-1.0, 1.0))
        P.frame([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], [-1, -0.5, 0, 0.5, 1], "w₁", "w₂", title)
        # contours of the loss
        t = np.linspace(0, 2 * math.pi, 200)
        lam, V = np.linalg.eigh(H_)
        for lev in (0.02, 0.1, 0.4):
            pts = np.array([V @ np.array([math.sqrt(2 * lev / lam[0]) * math.cos(a), math.sqrt(2 * lev / lam[1]) * math.sin(a)]) for a in t])
            P.line(pts[:, 0], pts[:, 1], "thin s0")
        Lc = np.linalg.cholesky(S)
        pts = rng.normal(size=(500, 2)) @ Lc.T
        for p in pts:
            if abs(p[0]) < 1.6 and abs(p[1]) < 1.0:
                body.append(f'<circle class="{cls}" cx="{P.X(p[0]):.1f}" cy="{P.Y(p[1]):.1f}" r="1.6" opacity="0.6"/>')
    desc = ("Stationary clouds of 500 samples for SGD-like dynamics on a quadratic loss whose curvatures are 10 and 0.1 "
            "(grey contours). Left: with isotropic noise the cloud is the Gibbs law, stretched along the flat direction "
            "(variances D/h). Right: with noise covariance proportional to the Hessian and the same total size, the cloud "
            "is round.")
    out.write_text(svg(W, Hh, "Noise shape and the stationary cloud", desc, body))


def fig_annealed(out, xp, xa):
    W, H = 920, 320
    body = []
    x = np.linspace(-9, 9, 721)
    L = Panel(body, 70, 45, 330, 220, (-9, 9), (0, 0.34))
    L.frame([-8, -4, 0, 4, 8], [0, 0.1, 0.2, 0.3], "x", "density", "The target, smoothed by noise of size σ")
    for sg, cls, lx in ((0.0, "ln s1", -4.0), (1.0, "ln s3", -4.0), (3.0, "ln s4", -4.0), (10.0, "ln s0", 6.0)):
        L.line(x, mix_pdf(x, sg), cls)
        L.text(lx, float(mix_pdf(np.array(lx), sg)), f"σ = {fmt(sg)}", "sm", "start", 8, -6)
    edges = np.linspace(-9, 9, 55)
    for j, (xs, title, cls) in enumerate(((xp, f"Plain Langevin: left-mode weight {np.mean(xp < 0):.2f}", "f2 bar"),
                                          (xa, f"Annealed: left-mode weight {np.mean(xa < 0):.2f}", "f3 bar"))):
        P = Panel(body, 500 + 215 * j, 45, 180, 220, (-9, 9), (0, 0.34))
        P.frame([-8, 0, 8], [0, 0.1, 0.2, 0.3], "x", "" if j else "density", "")
        hst, _ = np.histogram(xs, bins=edges, density=True)
        P.bars(edges, hst, cls)
        P.line(x, mix_pdf(x), "dash s1")
        body.append(f'<text class="hd" x="{P.x0}" y="{P.y0 - 12}">{"Plain" if j == 0 else "Annealed"}: {np.mean(xs < 0):.2f} left</text>')
    desc = ("Left: the mixture 0.8 N(−4, 1) + 0.2 N(4, 1) and its versions smoothed by Gaussian noise of size σ = 1, 3 and "
            "10, which merge the two modes. Right: histograms of 5000 samples from plain Langevin started uniformly on "
            "[−8, 8], which splits them about evenly between the modes, and from annealed Langevin, which recovers the 0.8 "
            "weight. The dashed line is the target.")
    out.write_text(svg(W, H, "Plain and annealed Langevin", desc, body))


def figures(mids, hist, Dsim, x, snaps, pi_c, kr_rows, sims, samp_rows, H_, S_iso, S_hess, xp, xa):
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    fig_paths(d / "paths.svg")
    fig_gibbs(d / "gibbs-temperatures.svg", mids, hist, Dsim)
    fig_depth_width(d / "depth-vs-width.svg")
    fig_fp(d / "fokker-planck.svg", x, snaps, pi_c)
    fig_kramers(d / "kramers.svg", kr_rows, sims)
    fig_samplers(d / "ula-mala.svg", samp_rows)
    fig_noise_shape(d / "noise-shape.svg", H_, S_iso, S_hess)
    fig_annealed(d / "annealed.svg", xp, xa)
    print("\nfigures: wrote", ", ".join(sorted(p.name for p in d.glob("*.svg"))))


def main():
    check_brownian()
    check_ou()
    mids, hist, Dsim = check_gibbs()
    check_depth_width()
    x, snaps, pi_c, _ = check_fokker_planck()
    kr_rows, sims = check_kramers()
    H_, S_iso, S_hess = check_reversibility()
    check_paths()
    samp_rows = check_samplers()
    check_sgd()
    xp, xa = check_annealed()
    if "--figures" in sys.argv:
        figures(mids, hist, Dsim, x, snaps, pi_c, kr_rows, sims, samp_rows, H_, S_iso, S_hess, xp, xa)


if __name__ == "__main__":
    main()
