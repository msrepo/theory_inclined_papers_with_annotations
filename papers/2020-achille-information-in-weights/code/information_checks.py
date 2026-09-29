#!/usr/bin/env python3
"""Checks of the algebra in Achille, Paolini & Soatto, "Where is the Information
in a Deep Neural Network?" (arXiv:1905.12213v5). Every number quoted in the notes
from sections 1-7 of the annotation is printed here.

  1. Proposition 2.3: E_D KL(Q(w|D) || P) = I(w; D) + KL(Qbar || P).
  2. Definition 2.1 is minimised by a Gibbs distribution, and d/dbeta of the
     optimum is the Information in the Weights (a rate-distortion identity).
  3. Proposition 2.5 and eq (8): the optimal covariance is beta (H + beta/lam^2 I)^-1,
     not (beta/2) (H + beta/(2 lam^2) I)^-1; eq (8) has -k where -k/2 belongs.
  4. The beta of Definition 2.1 is the PAC-Bayes beta divided by N.
  5. Proposition 2.8: with equal-loss minima, the preference of isotropic
     Langevin dynamics for the flatter one does not change with temperature
     (exact 1-D mean first-passage times and Gibbs masses).
  6. Proposition 2.9: flatness and stability are not separate knobs. At an exact
     minimiser the product (grad_D w*)^T H (grad_D w*) is reparameterisation
     invariant; for least squares it is the hat matrix.
  7. Proposition 3.2: the Fisher of the perturbed activations is
     grad_x f^T (beta J F^-1 J^T)^-1 grad_x f, not (1/beta) grad_x f^T J F J^T grad_x f;
     and eq (10) needs dim z >= dim x.

Standard library and numpy only.

Run:  python3 information_checks.py             (checks)
      python3 information_checks.py --figures   (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np


def kl_discrete(p, q):
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def kl_gauss(m1, S1, m2, S2):
    k = len(m1)
    S2i = np.linalg.inv(S2)
    d = m2 - m1
    return 0.5 * (np.trace(S2i @ S1) + d @ S2i @ d - k
                  + np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1])


# ---------------------------------------------------------------- 1
def check_prop_2_3():
    print("1. Proposition 2.3 (the adapted prior is the marginal)")
    rng = np.random.default_rng(0)
    nD, nW = 5, 6
    pD = rng.dirichlet(np.ones(nD))
    Q = rng.dirichlet(np.ones(nW) * 0.7, size=nD)          # Q(w | D), rows
    qbar = pD @ Q
    mi = sum(pD[d] * kl_discrete(Q[d], qbar) for d in range(nD))
    P = rng.dirichlet(np.ones(nW))
    lhs = sum(pD[d] * kl_discrete(Q[d], P) for d in range(nD))
    rhs = mi + kl_discrete(qbar, P)
    print(f"   E_D KL(Q||P) = {lhs:.6f}   I(w;D) + KL(Qbar||P) = {rhs:.6f}")
    print(f"   at P = Qbar the expected coding length is I(w;D) = {mi:.4f} nats; "
          f"at a random P it is {lhs:.4f}")
    print()


# ---------------------------------------------------------------- 2
def check_gibbs():
    print("2. Definition 2.1: the minimiser is Gibbs, Q* proportional to P exp(-L/beta)")
    w = np.linspace(-4, 4, 40_001)
    dw = w[1] - w[0]
    L = 0.25 * (w ** 2 - 1) ** 2 + 0.1 * w           # a tilted double well
    lam = 1.5
    P = np.exp(-w ** 2 / (2 * lam ** 2)); P /= P.sum() * dw

    def C(q, beta):
        m = q > 1e-300
        return float((q * L).sum() * dw + beta * (q[m] * np.log(q[m] / P[m])).sum() * dw)

    def gibbs(beta):
        q = P * np.exp(-(L - L.min()) / beta)
        return q / (q.sum() * dw)

    beta = 0.2
    qs = gibbs(beta)
    rng = np.random.default_rng(1)
    worse = []
    for _ in range(200):
        bump = np.exp(-(w - rng.uniform(-2, 2)) ** 2 / (2 * rng.uniform(0.05, 0.5) ** 2))
        q = qs * (1 + rng.uniform(-0.5, 0.5) * bump)
        q /= q.sum() * dw
        worse.append(C(q, beta) - C(qs, beta))
    # Gaussian candidates centred at the deeper minimum
    gauss_best = min(C(np.exp(-(w - mu) ** 2 / (2 * s ** 2)) / (math.sqrt(2 * math.pi) * s), beta)
                     for mu in np.linspace(-1.2, -0.8, 21) for s in np.linspace(0.05, 0.6, 56))
    print(f"   beta = {beta}: C(Gibbs) = {C(qs, beta):.5f}; 200 random perturbations are all worse "
          f"(min excess {min(worse):.2e}); best Gaussian is worse by {gauss_best - C(qs, beta):.4f}")
    # envelope identity: d/dbeta min_Q C = KL(Q*||P)
    Fm = lambda b: C(gibbs(b), b)
    h = 1e-4
    kl = (lambda q: float((q * np.log(np.maximum(q, 1e-300) / P)).sum() * dw))(qs)
    print(f"   d/dbeta min C = {(Fm(beta + h) - Fm(beta - h)) / (2 * h):.5f}   KL(Q*||P) = {kl:.5f}")
    print("   (so the Information in the Weights is the slope of the optimal Lagrangian in beta)")
    print()


# ---------------------------------------------------------------- 3
def check_prop_2_5():
    print("3. Proposition 2.5 and eq (8), quadratic loss L = L0 + (1/2)(w-w*)' H (w-w*)")
    rng = np.random.default_rng(2)
    k = 6
    A = rng.standard_normal((k, k))
    H = A @ A.T / k + 0.05 * np.eye(k)
    wstar = rng.standard_normal(k)
    lam, beta = 2.0, 0.01
    I = np.eye(k)

    def C(S):
        return 0.5 * np.trace(H @ S) + beta * kl_gauss(wstar, S, np.zeros(k), lam ** 2 * I)

    S_paper = (beta / 2) * np.linalg.inv(H + beta / (2 * lam ** 2) * I)
    S_true = beta * np.linalg.inv(H + beta / lam ** 2 * I)
    # gradient of the exact C at each candidate
    g = lambda S: 0.5 * H + beta / (2 * lam ** 2) * I - 0.5 * beta * np.linalg.inv(S)
    print(f"   ||dC/dSigma|| at the paper's Sigma* = {np.linalg.norm(g(S_paper)):.3e}, "
          f"at beta (H + beta/lam^2 I)^-1 = {np.linalg.norm(g(S_true)):.3e}")
    # the exact Gibbs posterior of a quadratic loss with a Gaussian prior has this covariance
    S_gibbs = np.linalg.inv(H / beta + I / lam ** 2)
    print(f"   Gibbs covariance (H/beta + I/lam^2)^-1 equals it: max diff {np.abs(S_gibbs - S_true).max():.2e}")
    ratio = np.trace(S_paper) / np.trace(S_true)
    print(f"   tr(paper) / tr(correct) = {ratio:.3f} (tends to 1/2 as beta -> 0)")
    print(f"   excess Lagrangian at the paper's Sigma*: C(paper) - C(correct) = {C(S_paper) - C(S_true):.4e} "
          f"(beta * {(C(S_paper) - C(S_true)) / beta:.3f} nats)")
    # the paper's C (tr(H Sigma) without the 1/2) is minimised by the paper's Sigma*
    g_paper = lambda S: H + beta / (2 * lam ** 2) * I - 0.5 * beta * np.linalg.inv(S)
    print(f"   the paper's own objective L0 + tr(H Sigma) + beta KL is stationary at its Sigma*: "
          f"||grad|| = {np.linalg.norm(g_paper(S_paper)):.1e}  -> the 1/2 of the Taylor term was dropped")
    # eq (8) as printed vs KL evaluated at the paper's Sigma*
    Hb = H + beta / (2 * lam ** 2) * I
    eq8 = (0.5 * np.linalg.slogdet(Hb)[1] + 0.5 * k * math.log(2 * lam ** 2 / beta) - k
           + (wstar @ wstar + np.trace(S_paper)) / (2 * lam ** 2))
    kl_direct = kl_gauss(wstar, S_paper, np.zeros(k), lam ** 2 * I)
    print(f"   eq (8) as printed = {eq8:.4f}, KL at the same Sigma* = {kl_direct:.4f}, "
          f"difference {kl_direct - eq8:.4f} = k/2 = {k / 2}")
    # uninformative limit
    for b in [1e-2, 1e-3]:
        St = b * np.linalg.inv(H + b / lam ** 2 * I)
        print(f"   beta = {b:g}: correct Sigma* vs beta H^-1: rel. diff "
              f"{np.linalg.norm(St - b * np.linalg.inv(H)) / np.linalg.norm(St):.2e}; the paper's is (beta/2) H^-1")
    print("   Definition 3.1 and Proposition 2.9 both use beta F^-1, i.e. the corrected formula")
    print()


# ---------------------------------------------------------------- 4
def check_beta_scale():
    print("4. Two betas: Definition 2.1 vs Theorem 2.2")
    print("   PAC-Bayes bracket: E_Q L_D + (beta_PAC / N)(KL + log 1/delta); Definition 2.1: E_Q L_D + beta KL")
    for N in [1000, 50_000]:
        print(f"   N = {N:>6}: beta_PAC > 1/2 means beta > {0.5 / N:.1e}; the ELBO is beta = 1/N = {1 / N:.1e}, not beta = 1")
    print()


# ---------------------------------------------------------------- 5
def potential(w, dL=0.0):
    a, b, ha, hb = -1.0, 1.5, 100.0, 1.0
    qa = 0.5 * ha * (w - a) ** 2
    qb = 0.5 * hb * (w - b) ** 2
    V = qa * qb / (qa + qb + 1e-300)
    return V + dL * np.exp(-(w - b) ** 2 / 0.08)       # optionally lift the flat minimum by dL


def mfpt(V, w, T, start, end):
    """Exact mean first-passage time for dX = -V'(X) dt + sqrt(2T) dW in 1-D,
    from `start` to `end` (> start), reflecting on the far left."""
    dw = w[1] - w[0]
    e_minus = np.exp(-(V - V.min()) / T)
    inner = np.cumsum(e_minus) * dw
    i0, i1 = np.searchsorted(w, start), np.searchsorted(w, end)
    e_plus = np.exp((V - V.min()) / T)
    return float((e_plus[i0:i1] * inner[i0:i1]).sum() * dw / T)


def check_kramers():
    print("5. Proposition 2.8: sharp minimum (h = 100) vs flat minimum (h = 1), same loss")
    w = np.linspace(-3.0, 4.5, 150_001)
    rows = []
    for dL, tag in [(0.0, "equal loss"), (0.25, "flat minimum 0.25 higher")]:
        V = potential(w, dL)
        sad = w[np.argmax(np.where((w > -1) & (w < 1.5), V, -1))]
        print(f"   {tag}: saddle at w = {sad:.3f}, barrier from the sharp side {V[np.searchsorted(w, sad)] - V[np.searchsorted(w, -1.0)]:.3f}")
        for T in [0.1, 0.2, 0.3]:
            t_sharp = mfpt(V, w, T, -1.0, sad + 0.3)            # leave the sharp basin
            t_flat = mfpt(V[::-1], -w[::-1], T, -1.5, -(sad - 0.3))   # leave the flat basin (mirror)
            p = np.exp(-(V - V.min()) / T)
            mass_sharp = p[w < sad].sum() / p.sum()
            rows.append((tag, T, t_sharp, t_flat, mass_sharp))
            print(f"     T = {T:4.2f}: E tau(leave sharp) = {t_sharp:9.3g}, E tau(leave flat) = {t_flat:9.3g}, "
                  f"ratio {t_sharp / t_flat:.4f};  Gibbs mass in sharp basin {mass_sharp:.4f}")
    print(f"   Laplace prediction for the equal-loss mass: sqrt(1/100) / (1 + sqrt(1/100)) = {0.1 / 1.1:.4f} at every T")
    print("   With equal loss the ratio does not move with T; T matters only through loss differences.")
    print()
    return rows


# ---------------------------------------------------------------- 6
def check_prop_2_9():
    print("6. Proposition 2.9: flatness and stability move together at a minimiser")
    rng = np.random.default_rng(3)
    N, p, beta = 40, 5, 0.1
    X = rng.standard_normal((N, p))
    H = 2 / N * X.T @ X                                   # Hessian of the mean squared loss
    J = np.linalg.solve(X.T @ X, X.T)                     # grad_y w*,  p x N
    Fw = J.T @ H @ J / beta
    hat = X @ np.linalg.solve(X.T @ X, X.T)
    print(f"   least squares, D parameterised by the labels y: (1/beta) J' H J = (2 / (N beta)) * hat matrix, "
          f"max diff {np.abs(Fw - 2 / (N * beta) * hat).max():.1e}")
    A = np.diag([0.1] * p) @ np.linalg.qr(rng.standard_normal((p, p)))[0]   # w = A^-1 v ... v = A w
    # reparameterise v = A^-1 w with A shrinking curvature 100x: H_v = A' H A, J_v = A^-1 J
    Hv = A.T @ H @ A
    Jv = np.linalg.solve(A, J)
    Fv = Jv.T @ Hv @ Jv / beta
    print(f"   after v = A^-1 w: Hessian trace {np.trace(H):.3f} -> {np.trace(Hv):.4f} (100x flatter), "
          f"||grad_y w*|| {np.linalg.norm(J):.3f} -> {np.linalg.norm(Jv):.3f} (10x less stable), "
          f"F_(w|D) unchanged: max diff {np.abs(Fv - Fw).max():.1e}")
    print(f"   rank of F_(w|D) = {np.linalg.matrix_rank(Fw)} for dim D = {N}: its determinant is 0 unless "
          f"dim D <= dim w, and the Brunel-Nadal formula is then -infinity")
    print()


# ---------------------------------------------------------------- 7
def gauss_fisher(mean_fn, cov_fn, x, h=1e-5):
    """Fisher of N(m(x), S(x)) with respect to x, including the covariance term."""
    d = len(x)
    S = cov_fn(x)
    Si = np.linalg.inv(S)
    dm, dS = [], []
    for i in range(d):
        e = np.zeros(d); e[i] = h
        dm.append((mean_fn(x + e) - mean_fn(x - e)) / (2 * h))
        dS.append((cov_fn(x + e) - cov_fn(x - e)) / (2 * h))
    Fmean = np.array([[dm[i] @ Si @ dm[j] for j in range(d)] for i in range(d)])
    Fcov = np.array([[0.5 * np.trace(Si @ dS[i] @ Si @ dS[j]) for j in range(d)] for i in range(d)])
    return Fmean, Fcov


def mc_fisher(mean_fn, cov_fn, x, n=4000, h=1e-3, seed=0):
    """Monte Carlo E_z[-Hess_x log p(z|x)], z ~ p(z|x), by finite differences."""
    rng = np.random.default_rng(seed)
    S = cov_fn(x)
    z = mean_fn(x) + rng.standard_normal((n, len(S))) @ np.linalg.cholesky(S).T

    def logp(xx):
        m, C = mean_fn(xx), cov_fn(xx)
        Ci = np.linalg.inv(C)
        r = z - m
        return -0.5 * np.einsum("ni,ij,nj->n", r, Ci, r) - 0.5 * np.linalg.slogdet(C)[1]

    d = len(x)
    Hm = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei = np.zeros(d); ei[i] = h
            ej = np.zeros(d); ej[j] = h
            Hm[i, j] = -np.mean(logp(x + ei + ej) - logp(x + ei - ej) - logp(x - ei + ej) + logp(x - ei - ej)) / (4 * h * h)
    return Hm


def check_prop_3_2():
    print("7. Proposition 3.2: Fisher of the perturbed activations z_n = (W + noise) x")
    rng = np.random.default_rng(4)
    dz, dx, beta = 3, 5, 0.05
    W = rng.standard_normal((dz, dx))
    Aa = rng.standard_normal((dx, dx)); A = Aa @ Aa.T / dx + 0.2 * np.eye(dx)
    Bb = rng.standard_normal((dz, dz)); B = Bb @ Bb.T / dz + 0.2 * np.eye(dz)
    Fw = np.kron(A, B)                                    # Fisher of vec(W) (column-major), K-FAC form
    Fwi = np.linalg.inv(Fw)
    Jf = lambda x: np.kron(x[None, :], np.eye(dz))        # d z / d vec(W)
    mean_fn = lambda x: W @ x
    cov_fn = lambda x: beta * Jf(x) @ Fwi @ Jf(x).T
    x = rng.standard_normal(dx)
    Fmean, Fcov = gauss_fisher(mean_fn, cov_fn, x)
    Fmc = mc_fisher(mean_fn, cov_fn, x)
    paper = W.T @ Jf(x) @ Fw @ Jf(x).T @ W / beta
    print(f"   exact Gaussian Fisher (mean + covariance terms) vs Monte Carlo: rel. diff "
          f"{np.linalg.norm(Fmean + Fcov - Fmc) / np.linalg.norm(Fmc):.3f}")
    print(f"   trace: mean term {np.trace(Fmean):.3f}, covariance term {np.trace(Fcov):.3f} "
          f"(O(1) in beta, the mean term is O(1/beta)), paper's formula {np.trace(paper):.3f}")
    kant = (x @ A @ x) * (x @ np.linalg.solve(A, x)) / (x @ x) ** 2
    print(f"   paper / correct mean term = {np.trace(paper) / np.trace(Fmean):.3f} = "
          f"(x'Ax)(x'A^-1x) = {kant * (x @ x) ** 2:.3f}; normalised by |x|^4 this is {kant:.3f}, "
          f"inside Kantorovich's [1, {(np.linalg.cond(A) + 1) ** 2 / (4 * np.linalg.cond(A)):.3f}]")
    F2m, _ = gauss_fisher(mean_fn, cov_fn, 2 * x)
    paper2 = W.T @ Jf(2 * x) @ Fw @ Jf(2 * x).T @ W / beta
    print(f"   doubling the input: correct mean term x{np.trace(F2m) / np.trace(Fmean):.3f}, "
          f"paper's formula x{np.trace(paper2) / np.trace(paper):.3f}  (units: J F J' is z^2/w^4, not 1/z^2)")
    # the two formulas also disagree on which directions of weight space matter
    print("   two weights, one activation z = w1 x, Fisher [[1, rho], [rho, 1]]:")
    for rho in [0.0, 0.9, 0.99]:
        F2 = np.array([[1, rho], [rho, 1.0]])
        j = np.array([[1.0, 0.0]])
        corr = 1 / (j @ np.linalg.inv(F2) @ j.T)[0, 0]
        pap = (j @ F2 @ j.T)[0, 0]
        print(f"     rho = {rho:4.2f}: (J F^-1 J')^-1 = {corr:.4f}   J F J' = {pap:.4f}")
    print("   Scaling F_w -> c F_w scales both by c, so the paper's qualitative claim (i) survives.")
    ev = np.linalg.eigvalsh(Fmean + Fcov)
    print(f"   eigenvalues of F_(z|x) (dim z = {dz}, dim x = {dx}): " + ", ".join(f"{e:.2e}" for e in ev))
    print(f"   the mean term has rank {np.linalg.matrix_rank(Fmean, tol=1e-8)} (<= dim z), the covariance term rank "
          f"{np.linalg.matrix_rank(Fcov, tol=1e-8)}, the sum rank {np.linalg.matrix_rank(Fmean + Fcov, tol=1e-8)} < dim x: "
          "|F_(z|x)| = 0 and eq (10) returns -infinity.")
    print("   The Brunel-Nadal step needs a representation from which x can be recovered, the opposite of an invariant one.")
    print()


def main():
    check_prop_2_3()
    check_gibbs()
    check_prop_2_5()
    check_beta_scale()
    rows = check_kramers()
    check_prop_2_9()
    check_prop_3_2()
    if "--figures" in sys.argv:
        figures(rows)


def figures(rows):
    from pathlib import Path
    import svgplot as sp

    out = Path(__file__).resolve().parent.parent / "figures"
    # ---- factor-of-two figure: 1-D exact Gibbs vs the two covariance formulas
    w = np.linspace(-0.6, 0.6, 481)
    h, lam, beta = 4.0, 1.0, 0.05
    s_true = beta / (h + beta / lam ** 2)
    s_pap = (beta / 2) / (h + beta / (2 * lam ** 2))
    m = (h / beta) * 0.0 / (h / beta + 1 / lam ** 2)
    g = lambda s: np.exp(-(w - m) ** 2 / (2 * s)) / math.sqrt(2 * math.pi * s)
    ax = sp.Axes(60, 70, 280, 180, (-0.6, 0.6), (0, 5.2))
    body = [sp.title(20, 22, "Proposition 2.5: the optimal perturbation is twice as wide as stated"),
            sp.sub(20, 39, "Quadratic loss (h = 4), prior N(0, 1), β = 0.05. Left: densities. Right: the Lagrangian in σ².")]
    body.append(ax.frame([-0.6, -0.3, 0, 0.3, 0.6], [0, 1, 2, 3, 4, 5], "w − w*", "density"))
    body.append(ax.path(w, g(s_true), "c2"))
    body.append(ax.path(w, g(s_pap), "c1 dash"))
    body.append(ax.path(w, 0.9 * np.exp(-0) * (0.5 * h * w ** 2) / 0.36 * 1.0, "k"))
    body.append(sp.legend(70, 300, [("c2", "exact Gibbs = β(h + β/λ²)⁻¹"), ("c1", "paper: (β/2)(h + β/2λ²)⁻¹"),
                                    ("k", "the loss, rescaled")]))
    # Lagrangian vs sigma^2
    s2 = np.linspace(0.002, 0.03, 200)
    Cs = 0.5 * h * s2 + beta * 0.5 * (s2 / lam ** 2 - 1 + math.log(lam ** 2) - np.log(s2))
    Cp = h * s2 + beta * 0.5 * (s2 / lam ** 2 - 1 + math.log(lam ** 2) - np.log(s2))
    ax2 = sp.Axes(420, 70, 270, 180, (0, 0.03), (float(Cs.min()) - 0.005, float(Cs.min()) + 0.06))
    body.append(ax2.frame([0, 0.01, 0.02, 0.03], [], "σ² of the post-distribution", "C_β − const"))
    body.append(ax2.path(s2, np.minimum(Cs, Cs.min() + 0.06), "c2"))
    body.append(ax2.path(s2, np.minimum(Cp - (Cp.min() - Cs.min()), Cs.min() + 0.06), "c1 dash"))
    body.append(ax2.vline(s_true, "c2"))
    body.append(ax2.vline(s_pap, "c1 dash"))
    body.append(sp.legend(430, 300, [("c2", "½ tr(HΣ) + β KL (correct Taylor term)"),
                                     ("c1", "tr(HΣ) + β KL (the paper's, shifted)")]))
    desc = ("Two Gaussian densities around a quadratic minimum: the exact minimiser of the Information Lagrangian, "
            "which is also the Gibbs distribution, and the covariance stated in Proposition 2.5, which is about half "
            "as wide. Right: the Lagrangian as a function of the variance, with the correct second-order term and "
            "with the paper's, which drops the one-half and so moves the minimum to half the variance.")
    (out / "factor-two.svg").write_text(sp.svg(720, 345, "Proposition 2.5 factor of two", desc, "\n".join(body)))

    # ---- Kramers figure
    wv = np.linspace(-2.0, 3.8, 1200)
    V0 = potential(wv, 0.0)
    ax = sp.Axes(60, 70, 280, 170, (-2.0, 3.8), (0, 2.0))
    body = [sp.title(20, 22, "Proposition 2.8: temperature does not tilt the flat/sharp preference"),
            sp.sub(20, 39, "Two minima of equal loss, curvature 100 and 1. Exact 1-D Langevin (dX = −V′dt + √(2T) dW).")]
    body.append(ax.frame([-2, -1, 0, 1, 2, 3], [0, 0.5, 1, 1.5, 2], "w", "V(w), and Gibbs densities scaled to peak 1.5"))
    body.append(ax.path(wv, np.minimum(V0, 2.0), "k0"))
    for T, cls in [(0.1, "c0"), (0.3, "c1")]:
        p = np.exp(-V0 / T); p /= p.max()
        body.append(ax.path(wv, 1.5 * p, cls))
    body.append(sp.legend(70, 292, [("k0", "V(w)"), ("c0", "Gibbs density, T = 0.1"), ("c1", "Gibbs density, T = 0.3")]))
    ax2 = sp.Axes(420, 70, 270, 170, (0.08, 0.32), (0.0, 0.5))
    body.append(ax2.frame([0.1, 0.2, 0.3], [0, 0.1, 0.2, 0.3, 0.4, 0.5], "temperature T", "share in the sharp basin"))
    for tag, cls in [("equal loss", "c2"), ("flat minimum 0.25 higher", "c3")]:
        rr = [r for r in rows if r[0] == tag]
        body.append(ax2.path([r[1] for r in rr], [r[4] for r in rr], cls))
        body.append(ax2.dots([r[1] for r in rr], [r[4] for r in rr], cls + "f"))
    body.append(ax2.hline(0.1 / 1.1, "k"))
    body.append(sp.legend(430, 292, [("c2", "equal loss"), ("c3", "flat minimum 0.25 higher"),
                                     ("k", "√(h_flat/h_sharp) rule, 0.091")]))
    desc = ("Left: a potential with a narrow, sharp well and a wide, flat well at the same depth, and its Gibbs "
            "density at two temperatures; the share in the sharp well stays near 0.09 at both. Right: that share "
            "against temperature is flat when the minima have equal loss and changes only when the flat minimum "
            "is lifted, so temperature trades loss against volume and does not by itself penalise curvature.")
    (out / "kramers.svg").write_text(sp.svg(720, 340, "Kramers and flat minima", desc, "\n".join(body)))
    print("wrote figures/factor-two.svg, figures/kramers.svg")


if __name__ == "__main__":
    main()
