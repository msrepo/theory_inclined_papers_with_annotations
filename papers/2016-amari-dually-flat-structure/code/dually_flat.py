#!/usr/bin/env python3
"""Amari, Information Geometry and Its Applications, Chapter 1, checked by hand.

Every number quoted in the notes comes from here. The running example is a three-outcome
categorical distribution p = (p0, p1, p2), because there the objects of the chapter are things
a deep-learning reader already owns: the natural parameters are logits (relative to outcome 0),
the convex function psi is log-sum-exp, the dual coordinates are the softmax probabilities, and
the Legendre dual of psi is the negative entropy.

Checked here, in the order the notes use them:

  1. a divergence is a local squared distance: KL ~ (1/2) g dxi dxi, its asymmetry, and how badly
     the triangle inequality fails (even for the square root);
  2. Bregman divergences: the book's examples, the Itakura-Saito and generalised-KL cases, the matrix
     divergences (1.32)-(1.34) including a Gaussian meaning for (1.33) and the alpha -> -1/+1 limits
     of (1.34); strict convexity versus a positive-definite Hessian;
  3. exponential families: grad psi = mean, Hessian psi = covariance, and
     D_psi[theta : theta'] = KL[p_theta' : p_theta] (note the reversal), for the categorical and for
     the Gaussian in the book's coordinates (1.8)-(1.10);
  4. the Legendre dual: psi* is the negative entropy, G* = G^-1, D_psi*[a : b] = D_psi[b : a], the
     self-dual expression of Theorem 1.1, and the book's pairs (1.71)-(1.75);
  5. two flat structures, one metric: e- and m-geodesics between two categoricals and where the
     Fisher-Rao (Levi-Civita) midpoint falls; the dual bases; what parallel transport changes and
     what it keeps;
  6. the generalised Pythagorean theorem, both versions, and the printed identity (1.114), whose
     indices do not match the proof's own (1.119);
  7. projections onto an e-flat and an m-flat submanifold (the independence model), the
     necessary-versus-sufficient warning, and the alternating (em) minimisation with (1.124);
  8. coordinates: the tensor law (1.130) on the Gaussian, convexity lost under a change of
     coordinates (1.80), affine changes preserved, and natural gradient = gradient in dual coordinates.

With --figures it also regenerates the SVGs in ../figures/.

Standard library and numpy only.

Run:  python3 dually_flat.py            (checks)
      python3 dually_flat.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ the categorical model on {0, 1, 2}


def p_of_theta(th):
    z = np.concatenate([[0.0], th])
    e = np.exp(z - z.max())
    return e / e.sum()


def theta_of_p(p):
    return np.log(p[1:] / p[0])


def psi(th):
    return float(np.log1p(np.sum(np.exp(th))))          # log-sum-exp with the baseline logit fixed at 0


def eta_of_theta(th):
    return p_of_theta(th)[1:]                            # grad psi = (p1, p2)


def G_theta(th):
    e = eta_of_theta(th)
    return np.diag(e) - np.outer(e, e)                   # Hessian of psi = covariance of the indicator vector


def psi_star(eta):
    p0 = 1 - eta.sum()
    return float(np.sum(eta * np.log(eta)) + p0 * np.log(p0))   # negative entropy


def G_eta(eta):
    p0 = 1 - eta.sum()
    return np.diag(1 / eta) + 1 / p0                     # Hessian of psi* (adds 1/p0 everywhere)


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sum(p * np.log(p / q)))


def D_psi(a, b):
    """Bregman divergence D_psi[a : b] on theta coordinates, definition (1.44) with xi_0 = b."""
    return psi(a) - psi(b) - float(eta_of_theta(b) @ (a - b))


def D_star(ea, eb):
    return psi_star(ea) - psi_star(eb) - float(theta_of_p(np.concatenate([[1 - eb.sum()], eb])) @ (ea - eb))


def head(s):
    print("\n" + s)


P3 = np.array([0.7, 0.2, 0.1])
Q3 = np.array([0.1, 0.3, 0.6])


# ------------------------------------------------------------------ 1. divergence = local squared distance

def check_divergence():
    head("1. A divergence is a local squared distance (Definition 1.1, eq 1.24)")
    p = np.array([0.5, 0.3, 0.2])
    g = np.diag(1 / p[1:]) + 1 / p[0]                    # Fisher metric in the coordinates xi = (p1, p2)
    print(f"   p = {p}; metric g in xi = (p1, p2): {np.array2string(g, precision=3)}")
    for eps in (0.1, 0.01, 0.001):
        dxi = eps * np.array([1.0, -1.0])
        q = np.array([p[0] - dxi.sum(), p[1] + dxi[0], p[2] + dxi[1]])
        ratio = kl(p, q) / (0.5 * dxi @ g @ dxi)
        print(f"   step {eps:5.3f} along (+1, -1): KL[p : p+dxi] = {kl(p, q):.3e}, (1/2) g dxi dxi = {0.5 * dxi @ g @ dxi:.3e}, ratio {ratio:.4f}")
    head("   asymmetry and the triangle inequality")
    print(f"   P = {P3}, Q = {Q3}: KL[P:Q] = {kl(P3, Q3):.4f}, KL[Q:P] = {kl(Q3, P3):.4f}, symmetrised (1.26) = {0.5 * (kl(P3, Q3) + kl(Q3, P3)):.4f}")
    a, b, c = np.array([0.99, 0.01]), np.array([0.5, 0.5]), np.array([0.01, 0.99])
    d_ac, d_ab, d_bc = kl(a, c), kl(a, b), kl(b, c)
    print(f"   three coins (0.99, 0.5, 0.01 heads): KL[a:c] = {d_ac:.3f} versus KL[a:b] + KL[b:c] = {d_ab:.3f} + {d_bc:.3f} = {d_ab + d_bc:.3f}")
    print(f"   even the square root fails: sqrt KL[a:c] = {math.sqrt(d_ac):.3f} versus {math.sqrt(d_ab):.3f} + {math.sqrt(d_bc):.3f} = {math.sqrt(d_ab) + math.sqrt(d_bc):.3f}")


# ------------------------------------------------------------------ 2. Bregman divergences

def bregman(f, grad, x, y):
    return f(x) - f(y) - grad(y) @ (x - y)


def mat_fun(A, f):
    w, V = np.linalg.eigh(A)
    return (V * f(w)) @ V.T


def rand_pd(rng, n):
    M = rng.normal(size=(n, n))
    return M @ M.T + 0.5 * np.eye(n)


def check_bregman():
    head("2. Bregman divergences (1.44) and the book's examples")
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0.3, 2.0, 4), rng.uniform(0.3, 2.0, 4)
    d_log = bregman(lambda z: -np.sum(np.log(z)), lambda z: -1 / z, x, y)
    d_log_formula = float(np.sum(np.log(y / x) + x / y - 1))
    print(f"   psi = -sum log xi (1.46): Bregman {d_log:.6f}, closed form sum[log(xi'/xi) + xi/xi' - 1] = {d_log_formula:.6f} (1.48; the Itakura-Saito divergence)")
    d_ent = bregman(lambda z: np.sum(z * np.log(z)), lambda z: np.log(z) + 1, x, y)
    d_gkl = float(np.sum(x * np.log(x / y) - x + y))
    print(f"   phi = sum xi log xi (1.49): Bregman {d_ent:.6f}, generalised KL (1.31)/(1.50) {d_gkl:.6f}")
    px, py = x / x.sum(), y / y.sum()
    print(f"   on probability vectors it is plain KL: {bregman(lambda z: np.sum(z * np.log(z)), lambda z: np.log(z) + 1, px, py):.6f} against {kl(px, py):.6f}")
    head("   strictly convex is not the same as a positive-definite Hessian")
    f4 = lambda z: z ** 4
    print(f"   psi(x) = x^4 is strictly convex, so D[x : 0] = x^4 = {f4(0.5):.4f} > 0 for x = 0.5, but its Hessian 12 x^2 is 0 at x = 0:"
          f" criterion (3) of Definition 1.1 (positive-definite g) fails there, so x^4 gives a divergence-like gap but not a Riemannian metric on all of R")
    head("   matrix divergences (1.32)-(1.34) on random positive-definite 3x3 matrices")
    n = 3
    logm = lambda A: mat_fun(A, np.log)
    d32 = lambda P, Q: float(np.trace(P @ logm(P) - P @ logm(Q) - P + Q))
    d33 = lambda P, Q: float(np.trace(P @ np.linalg.inv(Q)) - np.linalg.slogdet(P @ np.linalg.inv(Q))[1] - n)
    def d34(P, Q, a):
        return float(4 / (1 - a ** 2) * np.trace(-mat_fun(P, lambda w: w ** ((1 - a) / 2)) @ mat_fun(Q, lambda w: w ** ((1 + a) / 2)) + (1 - a) / 2 * P + (1 + a) / 2 * Q))
    mins = {"1.32": 1e9, "1.33": 1e9, "1.34 a=0.3": 1e9, "1.34 a=-0.7": 1e9}
    for _ in range(2000):
        P, Q = rand_pd(rng, n), rand_pd(rng, n)
        mins["1.32"] = min(mins["1.32"], d32(P, Q)); mins["1.33"] = min(mins["1.33"], d33(P, Q))
        mins["1.34 a=0.3"] = min(mins["1.34 a=0.3"], d34(P, Q, 0.3)); mins["1.34 a=-0.7"] = min(mins["1.34 a=-0.7"], d34(P, Q, -0.7))
    print("   smallest value over 2000 random pairs (all must be >= 0): " + ", ".join(f"{k}: {v:.4f}" for k, v in mins.items()))
    P = rand_pd(rng, n)
    print(f"   D[P:P] = 0 for each: {d32(P, P):.1e}, {d33(P, P):.1e}, {d34(P, P, 0.3):.1e}")
    P, Q = rand_pd(rng, n), rand_pd(rng, n)
    print(f"   alpha -> -1 limit of (1.34) is (1.32): D(alpha = -0.999999) = {d34(P, Q, -0.999999):.5f} against (1.32) = {d32(P, Q):.5f}")
    print(f"   alpha -> +1 limit is (1.32) with the arguments swapped: {d34(P, Q, 0.999999):.5f} against {d32(Q, P):.5f}")
    # (1.33) is twice the KL divergence between zero-mean Gaussians N(0, P) and N(0, Q): check in 2-D by quadrature
    P2, Q2 = rand_pd(rng, 2), rand_pd(rng, 2)
    xs = np.linspace(-9, 9, 721); h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs); Z = np.stack([X.ravel(), Y.ravel()], axis=1)
    def logpdf(S):
        Si = np.linalg.inv(S)
        return -0.5 * np.einsum("ni,ij,nj->n", Z, Si, Z) - 0.5 * np.linalg.slogdet(2 * math.pi * S)[1]
    lp, lq = logpdf(P2), logpdf(Q2)
    kl_num = float(np.sum(np.exp(lp) * (lp - lq)) * h * h)
    d33_2 = float(np.trace(P2 @ np.linalg.inv(Q2)) - np.linalg.slogdet(P2 @ np.linalg.inv(Q2))[1] - 2)
    print(f"   n = 2: KL[N(0,P) : N(0,Q)] by quadrature = {kl_num:.5f}; (1.33) = {d33_2:.5f}; ratio {d33_2 / kl_num:.4f}  ->  (1.33) is twice the Gaussian KL")


# ------------------------------------------------------------------ 3. exponential families

def gauss_psi(t):
    return -t[0] ** 2 / (4 * t[1]) + 0.5 * math.log(-math.pi / t[1])


def gauss_eta(t):
    return np.array([-t[0] / (2 * t[1]), t[0] ** 2 / (4 * t[1] ** 2) - 1 / (2 * t[1])])   # (E x, E x^2)


def gauss_theta(mu, sig):
    return np.array([mu / sig ** 2, -1 / (2 * sig ** 2)])


def check_exp_family():
    head("3. Exponential families: psi is the cumulant function (1.51)-(1.58)")
    th = np.array([0.4, -0.8])
    e, G = eta_of_theta(th), G_theta(th)
    p = p_of_theta(th)
    # covariance of the indicator vector x = (1[x=1], 1[x=2]) under p
    X = np.array([[0, 0], [1, 0], [0, 1]], float)
    mean = p @ X
    cov = (X - mean).T @ np.diag(p) @ (X - mean)
    num_grad = np.array([(psi(th + 1e-6 * np.eye(2)[i]) - psi(th - 1e-6 * np.eye(2)[i])) / 2e-6 for i in range(2)])
    print(f"   categorical, theta = {th}: p = {np.round(p, 4)}")
    print(f"   grad psi (finite differences) = {np.round(num_grad, 6)}; E[x] = {np.round(mean, 6)} (1.54)")
    print(f"   Hessian psi = {np.round(G, 5).tolist()}; Cov[x] = {np.round(cov, 5).tolist()} (1.56)")
    th2 = np.array([-0.5, 0.9])
    d = D_psi(th, th2)
    print(f"   D_psi[theta : theta'] = {d:.6f}; KL[p_theta' : p_theta] = {kl(p_of_theta(th2), p_of_theta(th)):.6f}; the other order KL[p_theta : p_theta'] = {kl(p_of_theta(th), p_of_theta(th2)):.6f}  (1.57-1.58: the arguments reverse)")
    head("   Gaussian in the book's coordinates: theta = (mu/s^2, -1/(2 s^2)), eta = (mu, mu^2 + s^2) = (m1, m2) of (1.8)-(1.10)")
    mu, s = 1.0, 2.0
    t = gauss_theta(mu, s)
    num_grad = np.array([(gauss_psi(t + 1e-6 * np.eye(2)[i]) - gauss_psi(t - 1e-6 * np.eye(2)[i])) / 2e-6 for i in range(2)])
    print(f"   (mu, sigma) = ({mu}, {s}): theta = {t}; grad psi = {np.round(num_grad, 6)}; (mu, mu^2+sigma^2) = {(mu, mu ** 2 + s ** 2)}")
    xs = np.linspace(-30, 30, 200001); dx = xs[1] - xs[0]
    def pdf(m, sg): return np.exp(-(xs - m) ** 2 / (2 * sg ** 2)) / (math.sqrt(2 * math.pi) * sg)
    Hnum = np.zeros((2, 2)); f = pdf(mu, s); m1, m2 = np.sum(f * xs) * dx, np.sum(f * xs ** 2) * dx
    feats = [xs, xs ** 2]
    for i in range(2):
        for j in range(2):
            Hnum[i, j] = np.sum(f * (feats[i] - [m1, m2][i]) * (feats[j] - [m1, m2][j])) * dx
    Hfd = np.array([[(gauss_psi(t + 1e-4 * (np.eye(2)[i] + np.eye(2)[j])) - gauss_psi(t + 1e-4 * (np.eye(2)[i] - np.eye(2)[j]))
                      - gauss_psi(t - 1e-4 * (np.eye(2)[i] - np.eye(2)[j])) + gauss_psi(t - 1e-4 * (np.eye(2)[i] + np.eye(2)[j]))) / 4e-8 for j in range(2)] for i in range(2)])
    print(f"   Hessian psi (finite differences) = {np.round(Hfd, 4).tolist()}; Cov[(x, x^2)] by quadrature = {np.round(Hnum, 4).tolist()}")
    mu2, s2 = -0.5, 1.2
    t2 = gauss_theta(mu2, s2)
    d = gauss_psi(t) - gauss_psi(t2) - gauss_eta(t2) @ (t - t2)
    g2 = pdf(mu2, s2)
    kl_num = float(np.sum(g2 * np.log(g2 / f + 1e-300)) * dx)
    print(f"   D_psi[theta : theta'] = {d:.6f}; KL[N({mu2},{s2}^2) : N({mu},{s}^2)] by quadrature = {kl_num:.6f}")


# ------------------------------------------------------------------ 4. Legendre duality

def check_legendre():
    head("4. The Legendre dual (1.59)-(1.79)")
    th = np.array([0.4, -0.8]); e = eta_of_theta(th); p = p_of_theta(th)
    print(f"   theta = {th} -> eta = grad psi = {np.round(e, 5)};  psi(theta) = {psi(th):.6f}, psi*(eta) = {psi_star(e):.6f}, theta.eta = {th @ e:.6f}, psi + psi* = {psi(th) + psi_star(e):.6f}")
    print(f"   psi* equals the negative entropy sum p log p over all three outcomes: {np.sum(p * np.log(p)):.6f}  (1.78)")
    G, Gs = G_theta(th), G_eta(e)
    print(f"   G = {np.round(G, 4).tolist()}, G* = {np.round(Gs, 4).tolist()}, G G* = {np.round(G @ Gs, 10).tolist()}  (1.66)")
    th2 = np.array([-0.5, 0.9]); e2 = eta_of_theta(th2)
    print(f"   D_psi[theta : theta'] = {D_psi(th, th2):.6f}; dual divergence D_psi*[eta' : eta] = {D_star(e2, e):.6f}  (1.68); self-dual form psi(theta) + psi*(eta') - theta.eta' = {psi(th) + psi_star(e2) - th @ e2:.6f}  (1.69)")
    print(f"   Fenchel-Young: psi(theta) + psi*(eta') - theta.eta' >= 0, and equals 0 only when eta' = grad psi(theta): at eta' = eta it is {psi(th) + psi_star(e) - th @ e:.1e}")
    head("   one coin: the picture in figures/legendre.svg")
    t0, t1 = -0.4, 2.0
    f1 = lambda x: math.log1p(math.exp(x)); fs1 = lambda e_: e_ * math.log(e_) + (1 - e_) * math.log(1 - e_)
    e0, e1 = 1 / (1 + math.exp(-t0)), 1 / (1 + math.exp(-t1))
    g1 = f1(t1) - f1(t0) - e0 * (t1 - t0)
    g2 = fs1(e0) - fs1(e1) - t1 * (e0 - e1)
    print(f"   psi(theta) = log(1 + e^theta): theta0 = {t0}, theta1 = {t1}; eta0 = psi'(theta0) = {e0:.3f}, eta1 = {e1:.3f}")
    print(f"   D_psi[theta1 : theta0] = {g1:.4f}; the dual gap psi*(eta0) - psi*(eta1) - theta1 (eta0 - eta1) = {g2:.4f}  (1.68: equal, arguments swapped)")
    head("   the book's example pairs (1.71)-(1.75)")
    xi = np.array([0.7, 1.9, 0.4]); xs_ = -1 / xi                       # grad of -sum log is -1/xi
    print(f"   psi = -sum log xi: xi* = -1/xi = {np.round(xs_, 4)}; psi* = xi.xi* - psi = {xi @ xs_ + np.sum(np.log(xi)):.6f}; -sum[1 + log(-xi*)] = {-np.sum(1 + np.log(-xs_)):.6f}  (1.73)")
    xs2 = np.log(xi) + 1                                                  # grad of sum xi log xi
    print(f"   phi = sum xi log xi: xi* = log xi + 1 = {np.round(xs2, 4)}; phi* = xi.xi* - phi = {xi @ xs2 - np.sum(xi * np.log(xi)):.6f}; sum exp(xi* - 1) = {np.sum(np.exp(xs2 - 1)):.6f}  (1.74)")
    print(f"   gradients of the duals give back xi: {np.round(-1 / xs_, 4)} and {np.round(np.exp(xs2 - 1), 4)}  (1.75)")


# ------------------------------------------------------------------ 5. two flat structures, one metric

def fisher_rao_mid(p, q, t=0.5):
    """Levi-Civita (Fisher-Rao) geodesic on the simplex: a great-circle arc after the map p -> sqrt(p)."""
    a, b = np.sqrt(p), np.sqrt(q)
    om = math.acos(float(np.clip(a @ b, -1, 1)))
    s = (math.sin((1 - t) * om) * a + math.sin(t * om) * b) / math.sin(om)
    return s ** 2


def e_path(p, q, t):
    return p_of_theta((1 - t) * theta_of_p(p) + t * theta_of_p(q))


def m_path(p, q, t):
    return (1 - t) * p + t * q


def check_flat_structures():
    head("5. Two notions of straight line, one Riemannian metric")
    print(f"   P = {P3}, Q = {Q3}")
    for t in (0.25, 0.5, 0.75):
        print(f"   t = {t}: m-geodesic (straight in eta = p) {np.round(m_path(P3, Q3, t), 4)};  e-geodesic (straight in theta) {np.round(e_path(P3, Q3, t), 4)};  Fisher-Rao arc {np.round(fisher_rao_mid(P3, Q3, t), 4)}")
    m, e_, f = m_path(P3, Q3, .5), e_path(P3, Q3, .5), fisher_rao_mid(P3, Q3)
    print(f"   at the midpoint p0 is {e_[0]:.4f} (e) < {f[0]:.4f} (Fisher-Rao) < {m[0]:.4f} (m): the Riemannian geodesic sits between the two flat ones")
    print(f"   the e-geodesic is the normalised geometric mixture p^(1-t) q^t: midpoint {np.round(np.sqrt(P3 * Q3) / np.sqrt(P3 * Q3).sum(), 4)}")
    head("   dual bases at p = (0.5, 0.3, 0.2): theta = log(p/p0), eta = (p1, p2)")
    th = theta_of_p(np.array([0.5, 0.3, 0.2])); e = eta_of_theta(th); G, Gs = G_theta(th), G_eta(e)
    print(f"   G = {np.round(G, 4).tolist()}, G* = G^-1 = {np.round(Gs, 4).tolist()}")
    A_up = np.array([1.0, 0.0])                                         # components A^i in the basis e_i (theta directions)
    A_dn = G @ A_up                                                      # components A_i in the dual basis e*^i (eta directions)
    print(f"   a step of one unit along theta_1: theta-components A^i = {A_up}, eta-components A_i = G A = {np.round(A_dn, 4)}; |A|^2 = A^i A_i = {A_up @ A_dn:.4f} = A^T G A")
    q_th = theta_of_p(np.array([0.2, 0.2, 0.6]))
    print(f"   same theta-components carried to p = (0.2, 0.2, 0.6): |A|^2 becomes {A_up @ G_theta(q_th) @ A_up:.4f} (length is not preserved by parallel transport)")
    B_up = np.array([G[1, 0], -G[0, 0]])                                  # B^T G A = 0, i.e. B is orthogonal to A at this point
    print(f"   choose B with B^T G A = {B_up @ G @ A_up:.1e} at the first point (orthogonal)")
    Gq = G_theta(q_th)
    print(f"   transport both by the same connection (keep theta-components): <A,B> at the new point = {A_up @ Gq @ B_up:+.4f}  (orthogonality lost)")
    B_dn = G @ B_up
    print(f"   transport A with nabla (keep A^i) and B with the dual connection (keep B_i): <A,B> = A^i B_i = {A_up @ B_dn:+.1e}  (orthogonality kept, as the book says)")


# ------------------------------------------------------------------ 6. generalised Pythagorean theorem

def orth_triangle(th_p, th_q, t, flip=False):
    """R on the e-geodesic through Q, orthogonal to the m-geodesic PQ (Theorem 1.2)."""
    d_eta = eta_of_theta(th_q) - eta_of_theta(th_p)
    v = np.array([-d_eta[1], d_eta[0]])
    return th_q + t * v


def check_pythagoras():
    head("6. Generalised Pythagorean theorem (Theorems 1.2 and 1.3) and the identity (1.114)")
    rng = np.random.default_rng(5)
    thp, thq = theta_of_p(P3), theta_of_p(Q3)
    print(f"   P = {P3}, Q = {Q3}; D_psi(Q:P) = {D_psi(thq, thp):.6f}")
    for t in (0.3, -0.5, 2.2):
        thr = orth_triangle(thp, thq, t)
        lhs, rhs = D_psi(thr, thp), D_psi(thq, thp) + D_psi(thr, thq)
        print(f"   t = {t:+.1f}: R = {np.round(p_of_theta(thr), 4)}; D(R:Q) = {D_psi(thr, thq):.6f}; D(R:P) = {lhs:.8f}; D(Q:P) + D(R:Q) = {rhs:.8f}; difference {lhs - rhs:+.1e}")
    thr = orth_triangle(thp, thq, 2.2)
    kl_form = (kl(P3, p_of_theta(thr)), kl(P3, Q3) + kl(Q3, p_of_theta(thr)))
    print(f"   in KL language, since D_psi(R:P) = KL[P:R]: KL[P:R] = {kl_form[0]:.6f} = KL[P:Q] + KL[Q:R] = {kl_form[1]:.6f}")
    head("   the identity printed as (1.114)")
    worst_mine = worst_book = 0.0; shown = 0
    for _ in range(200):
        a, b, c = (theta_of_p(rng.dirichlet(np.ones(3))) for _ in range(3))      # P, Q, R at random
        lhs = D_psi(b, a) + D_psi(c, b) - D_psi(c, a)                              # D(Q:P) + D(R:Q) - D(R:P)
        ea, eb, ec = eta_of_theta(a), eta_of_theta(b), eta_of_theta(c)
        mine = (b - c) @ (eb - ea)                                                 # (theta_Q - theta_R) . (theta*_Q - theta*_P)
        book = (a - b) @ (eb - ec)                                                 # (theta_P - theta_Q) . (theta*_Q - theta*_R), as printed
        worst_mine = max(worst_mine, abs(lhs - mine)); worst_book = max(worst_book, abs(lhs - book))
        if shown < 2:
            print(f"   random P, Q, R: left side {lhs:+.6f}; (theta_Q - theta_R).(eta_Q - eta_P) = {mine:+.6f}; as printed (theta_P - theta_Q).(eta_Q - eta_R) = {book:+.6f}")
            shown += 1
    print(f"   over 200 random triples: largest error of the corrected form {worst_mine:.1e}; of the printed form {worst_book:.2f}")
    print("   the corrected form vanishes exactly when (theta*_P - theta*_Q).(theta_Q - theta_R) = 0, i.e. the book's own (1.119)")
    head("   the dual theorem (1.120): geodesic PQ orthogonal to dual geodesic QR, then D_psi*(R:P) = D_psi*(Q:P) + D_psi*(R:Q)")
    eq, ep = eta_of_theta(thq), eta_of_theta(thp)
    d_th = thq - thp
    v = np.array([-d_th[1], d_th[0]])
    for t in (0.02, -0.05):
        er = eq + t * v
        lhs = D_star(er, ep); rhs = D_star(eq, ep) + D_star(er, eq)
        print(f"   t = {t:+.2f}: D*(R:P) = {lhs:.8f}; D*(Q:P) + D*(R:Q) = {rhs:.8f}; difference {lhs - rhs:+.1e}")


# ------------------------------------------------------------------ 7. projections and alternating minimisation

def golden(f, lo, hi, iters=200):
    for _ in range(iters):
        m1, m2 = lo + 0.381966 * (hi - lo), hi - 0.381966 * (hi - lo)
        lo, hi = (lo, m2) if f(m1) < f(m2) else (m1, hi)
    return (lo + hi) / 2


def check_projection():
    head("7. Projection theorem (1.4, 1.5): the independence model is e-flat")
    P = np.array([[0.4, 0.1], [0.2, 0.3]])
    ra, cb = P.sum(1), P.sum(0)
    Ph = np.outer(ra, cb)
    mi = kl(P.ravel(), Ph.ravel())
    print(f"   P = {P.tolist()}, marginals {ra} and {cb}; product of marginals {Ph.tolist()}; KL[P : product] (the mutual information) = {mi:.6f}")
    best = (1e9, None)
    grid = np.linspace(0.02, 0.98, 241)
    for a in grid:
        for b in grid:
            v = kl(P.ravel(), np.outer([a, 1 - a], [b, 1 - b]).ravel())
            if v < best[0]: best = (v, (a, b))
    print(f"   brute-force minimum of KL[P : Q] over products Q: {best[0]:.6f} at Q = (a, b) = ({best[1][0]:.3f}, {best[1][1]:.3f}); the marginals are ({ra[0]:.1f}, {cb[0]:.1f})")
    rng = np.random.default_rng(11); worst = 0.0
    for _ in range(2000):
        a, b = rng.uniform(0.02, 0.98, 2)
        Q = np.outer([a, 1 - a], [b, 1 - b])
        worst = max(worst, abs(kl(P.ravel(), Q.ravel()) - (mi + kl(Ph.ravel(), Q.ravel()))))
    print(f"   Pythagoras KL[P:Q] = KL[P:P-hat] + KL[P-hat:Q] for 2000 random products Q: largest error {worst:.1e}")
    print(f"   at a product Q = (0.5, 0.5) x (0.5, 0.5): KL[P:Q] = {kl(P.ravel(), np.full(4, .25)):.6f} = {mi:.6f} + {kl(Ph.ravel(), np.full(4, .25)):.6f}")
    head("   the dual statement: the I-projection onto the m-flat set K of joints with the marginals of P")
    p_prod = np.outer([0.7, 0.3], [0.2, 0.8])
    # stationarity of KL[Q(s) : p] along K: the odds ratio of Q(s) equals the odds ratio of p; solve by bisection
    odds = lambda M: math.log(M[0, 0] * M[1, 1] / (M[0, 1] * M[1, 0]))
    lo, hi = 0.1 + 1e-12, 0.5 - 1e-12
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if odds(joint_k(mid)) < odds(p_prod) else (lo, mid)
    s_best = (lo + hi) / 2; val = kl(joint_k(s_best).ravel(), p_prod.ravel())
    print("   K = joints with marginals (0.5, 0.5) and (0.6, 0.4): a line, parametrised by s = Q[0,0] in (0.1, 0.5); target p = product (0.7, 0.3) x (0.2, 0.8)")
    print(f"   min over K of KL[Q : p] = {val:.6f} at s = {s_best:.4f}")
    worst = 0.0
    for s in rng.uniform(0.11, 0.49, 500):
        worst = max(worst, abs(kl(joint_k(s).ravel(), p_prod.ravel()) - (val + kl(joint_k(s).ravel(), joint_k(s_best).ravel()))))
    print(f"   Pythagoras KL[R:p] = KL[R:Q-hat] + KL[Q-hat:p] over 500 random R in K: largest error {worst:.1e}")
    head("   the same on the three-outcome simplex (the example drawn in figures/projection.svg)")
    Pp, Sline, s_hat, Ph, d_min, gap, orth = projection_example()
    print(f"   P = {Pp}; S = the line theta(s) = (0.3, -0.6) + s (0.819, 0.573), straight in theta (e-flat); projection at s = {s_hat:.6f}, P-hat = {np.round(Ph, 4)}")
    print(f"   KL[P : P-hat] = {d_min:.6f}; over s in [-4, 4] the largest |KL[P:Q(s)] - (KL[P:P-hat] + KL[P-hat:Q(s)])| = {gap:.1e}")
    print(f"   orthogonality at P-hat: (eta_P - eta_P-hat) . (tangent of S in theta) = {orth:+.1e}")
    head("   which divergence does each projection minimise? (the pairing in Theorem 1.4 against the calculus)")
    thP = theta_of_p(Pp); dirv = Sline(1.0) - Sline(0.0)
    f_first = lambda s_: D_psi(Sline(s_), thP)             # D_psi[R : P], R varies in the first slot (= KL[P:R])
    f_second = lambda s_: D_psi(thP, Sline(s_))            # D_psi[P : R], R varies in the second slot (= KL[R:P])
    s1, s2 = golden(f_first, -6, 6), golden(f_second, -6, 6)
    def orth_dual(s_):                                     # eta-straight segment P->R against S's tangent (theta components)
        return float((eta_of_theta(thP) - eta_of_theta(Sline(s_))) @ dirv)
    def orth_geo(s_):                                      # theta-straight segment P->R against S's tangent (eta components)
        return float((Sline(s_) - thP) @ (G_theta(Sline(s_)) @ dirv))
    print(f"   S = the same e-flat line. argmin_R D_psi[R : P] is s = {s1:.6f}; argmin_R D_psi[P : R] is s = {s2:.6f}")
    print(f"   at the first, the dual-geodesic (eta-straight) segment is orthogonal to S: {orth_dual(s1):+.1e}; the geodesic one is not: {orth_geo(s1):+.4f}")
    print(f"   at the second, the geodesic (theta-straight) segment is orthogonal to S: {orth_geo(s2):+.1e}; the dual-geodesic one is not: {orth_dual(s2):+.4f}")
    Ph2 = Sline(s2)
    gap_first = max(abs(D_psi(Sline(t), thP) - (D_psi(Sline(s1), thP) + D_psi(Sline(t), Sline(s1)))) for t in np.linspace(-4, 4, 801))
    gap_second = max(abs(D_psi(thP, Sline(t)) - (D_psi(thP, Ph2) + D_psi(Ph2, Sline(t)))) for t in np.linspace(-4, 4, 801))
    print(f"   Pythagoras along S around the first minimiser: largest gap {gap_first:.1e}; around the second (S is flat, not dual flat): {gap_second:.3f}")
    print(f"   so on a flat S it is the DUAL geodesic projection that minimises D_psi[R : P] (R in the first slot), whereas Theorem 1.4 as printed says it minimises D_psi[P : R]")
    head("   necessary but not sufficient: a curved S")
    Pp = np.array([0.5, 0.0])
    ang = np.array([0.0, math.pi])
    pts = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    for a, pt in zip(ang, pts):
        tang = np.array([-math.sin(a), math.cos(a)])
        print(f"   unit circle S, P = (0.5, 0): point ({pt[0]:+.0f}, {pt[1]:+.0f}): (point - P) . tangent = {(pt - Pp) @ tang:+.1e} (projection condition holds); D = (1/2)|point - P|^2 = {0.5 * np.sum((pt - Pp) ** 2):.3f}")
    print("   both points satisfy the orthogonality condition; one is the minimum (0.125) and the other the maximum (1.125) of D over S")


def joint_k(s):
    """2x2 joint with marginals (0.5, 0.5) and (0.6, 0.4), parametrised by Q[0,0] = s."""
    return np.array([[s, 0.5 - s], [0.6 - s, s - 0.1]])


def em_iterations(Q0, steps=12):
    """K: joints with P00 = P11 = 0.4 (m-flat). S: independence model (e-flat). min KL[P:Q] over both."""
    vals, Q = [], Q0
    for _ in range(steps):
        s = 0.2 * Q[0, 1] / (Q[0, 1] + Q[1, 0])                         # I-projection of Q onto K: geodesic projection
        P = np.array([[0.4, s], [0.2 - s, 0.4]])
        v1 = kl(P.ravel(), Q.ravel())
        Q = np.outer(P.sum(1), P.sum(0))                                  # m-projection of P onto S: product of marginals
        v2 = kl(P.ravel(), Q.ravel())
        vals += [v1, v2]
    return vals, P, Q


def check_em():
    head("   alternating minimisation between K = {P00 = P11 = 0.4} (m-flat) and S = independence model (e-flat) (1.123-1.124)")
    ss = np.linspace(1e-4, 0.2 - 1e-4, 200001)
    def mi_s(s):
        P = np.array([[0.4, s], [0.2 - s, 0.4]]); return kl(P.ravel(), np.outer(P.sum(1), P.sum(0)).ravel())
    vals_ = np.array([mi_s(s) for s in ss[::40]])
    k = int(np.argmin(vals_)); s_star = ss[::40][k]; d_star = vals_[k]
    print(f"   brute force over K of KL[P : product of its marginals]: minimum {d_star:.6f} at P01 = {s_star:.4f} (= P10 by symmetry)")
    rng = np.random.default_rng(2)
    finals = []
    for trial in range(5):
        Q0 = np.outer(*rng.dirichlet(np.ones(2), 2))
        vals, P, Q = em_iterations(Q0)
        mono = all(vals[i + 1] <= vals[i] + 1e-14 for i in range(len(vals) - 1))
        finals.append(vals[-1])
        print(f"   start {trial}: Q0 = {np.round(Q0.ravel(), 3)}; D after steps 1, 2, 3, 12: {vals[0]:.5f}, {vals[2]:.5f}, {vals[4]:.5f}, {vals[-1]:.6f}; non-increasing: {mono}")
    print(f"   all five starts end at {max(finals):.6f} .. {min(finals):.6f}; brute-force minimum {d_star:.6f}")
    return vals


# ------------------------------------------------------------------ 8. coordinates

def check_coordinates():
    head("8. Coordinates (1.80), (1.125)-(1.130) and natural gradient")
    mu, s = 1.0, 2.0
    t = gauss_theta(mu, s)
    Hth = np.array([[(gauss_psi(t + 1e-4 * (np.eye(2)[i] + np.eye(2)[j])) - gauss_psi(t + 1e-4 * (np.eye(2)[i] - np.eye(2)[j]))
                      - gauss_psi(t - 1e-4 * (np.eye(2)[i] - np.eye(2)[j])) + gauss_psi(t - 1e-4 * (np.eye(2)[i] + np.eye(2)[j]))) / 4e-8 for j in range(2)] for i in range(2)])
    J = np.array([[1 / s ** 2, -2 * mu / s ** 3], [0, 1 / s ** 3]])     # d theta / d (mu, sigma)
    g_musig = J.T @ Hth @ J
    print(f"   Gaussian at (mu, sigma) = (1, 2): Hessian of psi in theta = {np.round(Hth, 4).tolist()}")
    print(f"   transformed by the tensor law (1.130) with J = d theta / d(mu, sigma): {np.round(g_musig, 5).tolist()}; the Fisher information is diag(1/sigma^2, 2/sigma^2) = {np.diag([1 / s ** 2, 2 / s ** 2]).tolist()}")
    ds2_a = 1e-3 ** 2 * np.array([0.6, -0.8]) @ Hth @ np.array([0.6, -0.8])
    dmusig = np.linalg.solve(J, 1e-3 * np.array([0.6, -0.8]))
    print(f"   one small step: ds^2 = {ds2_a:.4e} in theta and {dmusig @ g_musig @ dmusig:.4e} in (mu, sigma); the length does not depend on the chart")
    head("   (1.80): the same psi in (mu, sigma) is not convex")
    f = lambda m, sg: m ** 2 / (2 * sg ** 2) + math.log(sg) + 0.5 * math.log(2 * math.pi)
    print(f"   psi~(mu, sigma) = mu^2/(2 sigma^2) + log sigma + const equals gauss_psi: {f(mu, s):.6f} vs {gauss_psi(t):.6f}")
    hs = 1e-4
    Hms = np.array([[(f(0 + hs, 1) - 2 * f(0, 1) + f(0 - hs, 1)) / hs ** 2, (f(hs, 1 + hs) - f(hs, 1 - hs) - f(-hs, 1 + hs) + f(-hs, 1 - hs)) / (4 * hs ** 2)],
                    [0, (f(0, 1 + hs) - 2 * f(0, 1) + f(0, 1 - hs)) / hs ** 2]])
    print(f"   at (mu, sigma) = (0, 1): d^2/d sigma^2 = {Hms[1, 1]:+.4f} < 0, so psi~ is not convex in (mu, sigma) although it is convex in theta")
    head("   affine changes of theta keep everything (1.81)")
    rng = np.random.default_rng(8)
    A = rng.normal(size=(2, 2)) + 2 * np.eye(2); b = rng.normal(size=2)
    th = np.array([0.4, -0.8]); th_new = A @ th + b
    psi_new = lambda tn: psi(np.linalg.solve(A, tn - b))
    grad_new = np.array([(psi_new(th_new + 1e-6 * np.eye(2)[i]) - psi_new(th_new - 1e-6 * np.eye(2)[i])) / 2e-6 for i in range(2)])
    print(f"   theta' = A theta + b; new dual coordinates: grad psi' = {np.round(grad_new, 6)}; A^-T eta = {np.round(np.linalg.solve(A.T, eta_of_theta(th)), 6)}")
    th2 = np.array([-0.5, 0.9])
    print(f"   divergence unchanged: D_psi'[theta' : theta2'] = {psi_new(th_new) - psi_new(A @ th2 + b) - grad_new_at(psi_new, A @ th2 + b) @ (th_new - (A @ th2 + b)):.6f} against {D_psi(th, th2):.6f}")
    head("   natural gradient is the plain gradient in the dual coordinates")
    target = np.array([0.2, 0.5, 0.3])
    L = lambda th_: kl(target, p_of_theta(th_))                        # a loss on the logits
    th = np.array([0.4, -0.8])
    gr_theta = np.array([(L(th + 1e-6 * np.eye(2)[i]) - L(th - 1e-6 * np.eye(2)[i])) / 2e-6 for i in range(2)])
    Lh = lambda e_: L(theta_of_p(np.concatenate([[1 - e_.sum()], e_])))   # the same loss as a function of eta
    e0 = eta_of_theta(th)
    gr_eta = np.array([(Lh(e0 + 1e-6 * np.eye(2)[i]) - Lh(e0 - 1e-6 * np.eye(2)[i])) / 2e-6 for i in range(2)])
    print(f"   cross-entropy loss on logits theta = {th}: gradient in theta = {np.round(gr_theta, 5)} (= eta - target = {np.round(e0 - target[1:], 5)})")
    print(f"   gradient in eta = {np.round(gr_eta, 5)}; G^-1 (gradient in theta) = {np.round(np.linalg.solve(G_theta(th), gr_theta), 5)}")


def grad_new_at(f, x):
    return np.array([(f(x + 1e-6 * np.eye(2)[i]) - f(x - 1e-6 * np.eye(2)[i])) / 2e-6 for i in range(2)])


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
  .con{stroke-width:0.9;fill:none;opacity:.75}
  .fillS{fill:#2a78d6;opacity:.08}
  .ring{stroke:#fdfdfc;stroke-width:1.5}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500} .s0{stroke:#85837b}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500} .f0{fill:#85837b}
    .fillS{fill:#3987e5}
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

    def frame(self, xt, yt, xlab, ylab, title, grid=True, xtl=None):
        b = self.b
        for y in yt:
            if grid: b.append(f'<line class="gd" x1="{self.x0}" y1="{self.Y(y):.1f}" x2="{self.x0 + self.w}" y2="{self.Y(y):.1f}"/>')
            b.append(f'<text class="sm" x="{self.x0 - 6}" y="{self.Y(y) + 3.5:.1f}" text-anchor="end">{fmt(y)}</text>')
        for x in xt:
            if grid: b.append(f'<line class="gd" x1="{self.X(x):.1f}" y1="{self.y0}" x2="{self.X(x):.1f}" y2="{self.y0 + self.h}"/>')
            b.append(f'<text class="sm" x="{self.X(x):.1f}" y="{self.y0 + self.h + 15}" text-anchor="middle">{xtl[x] if xtl else fmt(x)}</text>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0 + self.h}" x2="{self.x0 + self.w}" y2="{self.y0 + self.h}"/>')
        b.append(f'<line class="ax" x1="{self.x0}" y1="{self.y0}" x2="{self.x0}" y2="{self.y0 + self.h}"/>')
        b.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 32}" text-anchor="middle">{xlab}</text>')
        b.append(f'<text class="lab" transform="translate({self.x0 - 40},{self.y0 + self.h / 2}) rotate(-90)" text-anchor="middle">{ylab}</text>')
        b.append(f'<text class="hd" x="{self.x0}" y="{self.y0 - 12}">{title}</text>')

    def line(self, xs, ys, cls):
        keep = [(x, y) for x, y in zip(xs, ys) if self.xr[0] - 1e-9 <= x <= self.xr[1] + 1e-9 and self.yr[0] - 1e-9 <= y <= self.yr[1] + 1e-9]
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in keep)
        self.b.append(f'<polyline class="{cls}" points="{pts}"/>')

    def dot(self, x, y, cls, r=4.5):
        self.b.append(f'<circle class="{cls} ring" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>')

    def text(self, x, y, s, cls="v", anchor="start", dx=0, dy=0):
        self.b.append(f'<text class="{cls}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" text-anchor="{anchor}">{s}</text>')


# simplex chart: a point p = (p0, p1, p2) is drawn at p0 V0 + p1 V1 + p2 V2
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]])


def simplex_xy(p):
    return np.asarray(p) @ V


class SimplexPanel(Panel):
    def __init__(self, body, x0, y0, w, title, ty=None):
        super().__init__(body, x0, y0, w, w * math.sqrt(3) / 2, (0, 1), (0, math.sqrt(3) / 2))
        self.title, self.ty = title, (y0 - 34 if ty is None else ty)

    def frame_simplex(self):
        b = self.b
        pts = " ".join(f"{self.X(v[0]):.1f},{self.Y(v[1]):.1f}" for v in V)
        b.append(f'<polygon class="ax fillS" points="{pts}"/>')
        for k, name in enumerate(["outcome 0", "outcome 1", "outcome 2"]):
            anchor = ["start", "end", "middle"][k]
            dy = [16, 16, -8][k]
            b.append(f'<text class="sm" x="{self.X(V[k][0]):.1f}" y="{self.Y(V[k][1]) + dy:.1f}" text-anchor="{anchor}">{name}</text>')
        b.append(f'<text class="hd" x="{self.x0}" y="{self.ty}">{self.title}</text>')

    def curve(self, ps, cls):
        xy = np.array([simplex_xy(p) for p in ps])
        self.line(xy[:, 0], xy[:, 1], cls)

    def pt(self, p, cls, r=4.5):
        xy = simplex_xy(p); self.dot(xy[0], xy[1], cls, r)

    def label(self, p, s, dx=8, dy=-8, cls="v"):
        xy = simplex_xy(p)
        self.b.append(f'<text class="{cls}" x="{self.X(xy[0]) + dx:.1f}" y="{self.Y(xy[1]) + dy:.1f}">{s}</text>')


def theta_panel(body, x0, y0, w, h, title, xr=(-3, 3), yr=(-3, 3)):
    P_ = Panel(body, x0, y0, w, h, xr, yr)
    P_.frame([-2, 0, 2], [-2, 0, 2], "theta1 = log(p1/p0)", "theta2 = log(p2/p0)", title)
    return P_


def fig_charts(out):
    ts = np.linspace(0, 1, 120)
    thp, thq = theta_of_p(P3), theta_of_p(Q3)
    e_th = np.array([(1 - t) * thp + t * thq for t in ts])
    m_th = np.array([theta_of_p(m_path(P3, Q3, t)) for t in ts])
    e_p = [e_path(P3, Q3, t) for t in ts]; m_p = [m_path(P3, Q3, t) for t in ts]
    body = []
    L = theta_panel(body, 56, 46, 290, 290, "θ chart (logits): straight = e-geodesic")
    L.line(e_th[:, 0], e_th[:, 1], "ln s1"); L.line(m_th[:, 0], m_th[:, 1], "ln s2")
    for t in (0.25, 0.5, 0.75):
        a, b_ = (1 - t) * thp + t * thq, theta_of_p(m_path(P3, Q3, t)); L.dot(a[0], a[1], "f1", 3.5); L.dot(b_[0], b_[1], "f2", 3.5)
    L.dot(thp[0], thp[1], "f0"); L.dot(thq[0], thq[1], "f0")
    L.text(thp[0], thp[1], "P", "v", dx=-12, dy=-7); L.text(thq[0], thq[1], "Q", "v", dx=7, dy=-7)
    R = SimplexPanel(body, 452, 46 + 29, 290, "η chart (probabilities): straight = m-geodesic", ty=34)
    R.frame_simplex()
    R.curve(e_p, "ln s1"); R.curve(m_p, "ln s2")
    for t in (0.25, 0.5, 0.75):
        R.pt(e_path(P3, Q3, t), "f1", 3.5); R.pt(m_path(P3, Q3, t), "f2", 3.5)
    R.pt(P3, "f0"); R.pt(Q3, "f0"); R.label(P3, "P", -14, -4); R.label(Q3, "Q", 8, -6)
    body.append('<line class="ln s1" x1="60" y1="392" x2="92" y2="392"/><text class="sm" x="98" y="396">e-geodesic: straight in θ (normalised geometric mixture of P and Q)</text>')
    body.append('<line class="ln s2" x1="60" y1="412" x2="92" y2="412"/><text class="sm" x="98" y="416">m-geodesic: straight in η (ordinary mixture (1−t)P + tQ)</text>')
    body.append('<text class="sm" x="60" y="436">dots mark t = 0.25, 0.5, 0.75: the two paths join P to Q but pass through different points, so \"straight\" depends on the chart</text>')
    (out / "charts.svg").write_text(svg(780, 450, "One pair of distributions, two straight lines",
        "Two charts of the same three-outcome probability simplex with P = (0.7, 0.2, 0.1) and Q = (0.1, 0.3, 0.6). In the logit chart the e-geodesic is a straight blue line and the m-geodesic an orange curve; in the probability triangle the roles swap. The t = 0.5 midpoints differ: p0 is 0.351 on the e-geodesic and 0.400 on the m-geodesic.", body))


def fig_pythagoras(out):
    thp, thq = theta_of_p(P3), theta_of_p(Q3)
    thr = orth_triangle(thp, thq, 2.2); Rp = p_of_theta(thr)
    ts = np.linspace(0, 1, 120)
    m_th = np.array([theta_of_p(m_path(P3, Q3, t)) for t in ts])
    tt = np.linspace(-0.2, 1.2, 160)
    e_th = np.array([thq + t * (thr - thq) for t in tt])
    e_p = [p_of_theta(x) for x in e_th]; m_p = [m_path(P3, Q3, t) for t in ts]
    dqp, drq, drp = D_psi(thq, thp), D_psi(thr, thq), D_psi(thr, thp)
    body = []
    L = theta_panel(body, 56, 46, 290, 290, "θ chart: QR is straight", xr=(-3, 3.5), yr=(-3, 3.5))
    L.line(m_th[:, 0], m_th[:, 1], "ln s2"); L.line(e_th[:, 0], e_th[:, 1], "ln s1")
    for nm, th_ in (("P", thp), ("Q", thq), ("R", thr)):
        L.dot(th_[0], th_[1], "f0"); L.text(th_[0], th_[1], nm, "v", dx=7, dy=-7)
    R_ = SimplexPanel(body, 452, 75, 290, "η chart: PQ is straight", ty=34)
    R_.frame_simplex(); R_.curve(m_p, "ln s2"); R_.curve(e_p, "ln s1")
    for nm, p_ in (("P", P3), ("Q", Q3), ("R", Rp)):
        R_.pt(p_, "f0"); R_.label(p_, nm, 8, -6)
    body.append('<line class="ln s2" x1="60" y1="392" x2="92" y2="392"/><text class="sm" x="98" y="396">m-geodesic P→Q (dual geodesic)</text>')
    body.append('<line class="ln s1" x1="330" y1="392" x2="362" y2="392"/><text class="sm" x="368" y="396">e-geodesic Q→R, chosen orthogonal to PQ at Q</text>')
    body.append(f'<text class="v" x="60" y="424">D(Q:P) = {dqp:.4f}  ·  D(R:Q) = {drq:.4f}  ·  sum = {dqp + drq:.4f}  ·  D(R:P) = {drp:.4f}</text>')
    body.append('<text class="sm" x="60" y="444">the right angle is invisible in both charts (neither is orthonormal); it is a statement about the metric G, not about the picture</text>')
    (out / "pythagoras.svg").write_text(svg(780, 460, "A right triangle in a dually flat manifold",
        f"P, Q, R in the three-outcome simplex drawn in the logit chart and in the probability triangle. The m-geodesic from P to Q and the e-geodesic from Q to R meet at a right angle in the Fisher metric. D(Q:P) = {dqp:.4f}, D(R:Q) = {drq:.4f}, their sum {dqp + drq:.4f} equals D(R:P) = {drp:.4f}.", body))


def contour_segments(Z, xs, ys, level):
    segs = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            c = [Z[i, j], Z[i + 1, j], Z[i + 1, j + 1], Z[i, j + 1]]
            if any(math.isnan(v) for v in c): continue
            pts_ = [(xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])]
            hits = []
            for k in range(4):
                a, b_ = c[k], c[(k + 1) % 4]
                if (a - level) * (b_ - level) < 0:
                    f = (level - a) / (b_ - a)
                    hits.append((pts_[k][0] + f * (pts_[(k + 1) % 4][0] - pts_[k][0]), pts_[k][1] + f * (pts_[(k + 1) % 4][1] - pts_[k][1])))
            if len(hits) == 2: segs.append(hits)
            elif len(hits) == 4: segs += [hits[:2], hits[2:]]
    return segs


def projection_example():
    """P off an e-flat line S in the three-outcome simplex; P-hat found to machine precision."""
    Pp = np.array([0.15, 0.25, 0.60])
    a0 = np.array([0.3, -0.6]); dirv = np.array([1.0, 0.7]) / math.hypot(1.0, 0.7)
    Sline = lambda s: a0 + s * dirv                                             # e-flat submanifold S(s)
    f = lambda s: kl(Pp, p_of_theta(Sline(s)))
    s_hat = golden(f, -6.0, 6.0); Ph = p_of_theta(Sline(s_hat)); d_min = f(s_hat)
    ss = np.linspace(-4, 4, 4001)
    gap = max(abs(f(s) - (d_min + kl(Ph, p_of_theta(Sline(s))))) for s in ss)
    # orthogonality at P-hat in the Fisher metric: the m-geodesic direction P - P-hat (in eta) against the tangent of S (in theta)
    th_hat = Sline(s_hat)
    return Pp, Sline, s_hat, Ph, d_min, gap, float((Pp[1:] - Ph[1:]) @ dirv)


def fig_projection(out):
    Pp, Sline, s_hat, Ph, d_min, max_err, _ = projection_example()
    ss = np.linspace(-4, 4, 4001)
    # sample on an equilateral grid: parametrise by (x, y) in the triangle
    xs = np.linspace(0, 1, 110); ys = np.linspace(0, math.sqrt(3) / 2, 96)
    Zg = np.full((len(xs), len(ys)), np.nan)
    inv = np.linalg.inv(np.array([[V[1, 0] - V[0, 0], V[2, 0] - V[0, 0]], [V[1, 1] - V[0, 1], V[2, 1] - V[0, 1]]]))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            u, v = inv @ np.array([x - V[0, 0], y - V[0, 1]]); p0 = 1 - u - v
            if min(u, v, p0) > 1e-4: Zg[i, j] = kl(Pp, np.array([p0, u, v]))
    body = []
    R_ = SimplexPanel(body, 40, 60, 330, "level sets of KL[P : Q] over Q, and the line S", ty=34)
    R_.frame_simplex()
    for lev in (0.08, 0.2, 0.4, 0.7, 1.1):
        Zl = Zg
        d = "".join(f"M{R_.X(x1):.1f} {R_.Y(y1):.1f}L{R_.X(x2):.1f} {R_.Y(y2):.1f}" for (x1, y1), (x2, y2) in contour_segments(Zl, xs, ys, lev))
        R_.b.append(f'<path class="con s1" d="{d}"/>')
    sline = np.linspace(-3.2, 3.2, 200)
    Spts = [p_of_theta(Sline(s)) for s in sline]
    xy_S = np.array([simplex_xy(p) for p in Spts])
    R_.line(xy_S[:, 0], xy_S[:, 1], "ln s2")
    tt = np.linspace(0, 1, 100)
    mpath = [m_path(Pp, Ph, t) for t in tt]
    R_.curve(mpath, "dash s0")
    R_.pt(Pp, "f0"); R_.label(Pp, "P", 8, -6); R_.pt(Ph, "f2"); R_.label(Ph, "P̂", 8, 14)
    # profile along S
    Q = Panel(body, 470, 70, 280, 200, (-4, 4), (0, 3))
    Q.frame([-4, -2, 0, 2, 4], [0, 1, 2, 3], "position s along S", "divergence", "along S: KL[P:Q(s)] and KL[P:P̂] + KL[P̂:Q(s)]")
    prof = np.array([kl(Pp, p_of_theta(Sline(s))) for s in ss]); pyth = np.array([d_min + kl(Ph, p_of_theta(Sline(s))) for s in ss])
    Q.line(ss, np.clip(prof, 0, 3), "ln s1"); Q.line(ss, np.clip(pyth, 0, 3), "dash s2")
    Q.dot(s_hat, d_min, "f2")
    body.append('<line class="ln s2" x1="48" y1="430" x2="78" y2="430"/><text class="sm" x="84" y="434">S: a straight line in θ (e-flat)</text>')
    body.append('<line class="dash s0" x1="48" y1="450" x2="78" y2="450"/><text class="sm" x="84" y="454">m-geodesic from P to its projection P̂, meeting S at a right angle (in the Fisher metric)</text>')
    body.append(f'<text class="sm" x="470" y="330">the two curves coincide: largest gap {max_err:.1e}</text>')
    body.append(f'<text class="sm" x="470" y="348">because S is e-flat, Pythagoras holds for every Q on S,</text>')
    body.append('<text class="sm" x="470" y="364">so the only critical point is the minimum</text>')
    (out / "projection.svg").write_text(svg(780, 470, "Projecting onto an e-flat submanifold",
        f"Left: level sets of KL[P:Q] in the probability triangle, an e-flat line S (a straight line in the logit chart, curved here) and the m-geodesic from P to its projection P-hat on S. Right: KL[P:Q(s)] along S and KL[P:P-hat] + KL[P-hat:Q(s)] lie on top of each other (largest gap {max_err:.1e}); the minimum is {d_min:.4f}.", body))
    return d_min, max_err


def fig_critical(out):
    body = []
    L = Panel(body, 56, 46, 290, 290, (-1.4, 1.4), (-1.4, 1.4))
    L.frame([-1, 0, 1], [-1, 0, 1], "ξ1", "ξ2", "Euclidean case: S is a circle, P = (0.5, 0)")
    th = np.linspace(0, 2 * math.pi, 200)
    L.line(np.cos(th), np.sin(th), "ln s2")
    Pp = np.array([0.5, 0.0])
    for ang in (0.0, math.pi):
        pt = np.array([math.cos(ang), math.sin(ang)]); L.line([Pp[0], pt[0]], [Pp[1], pt[1]], "dash s0"); L.dot(pt[0], pt[1], "f2" if ang == 0 else "f4")
    L.dot(0.5, 0, "f0"); L.text(0.5, 0, "P", "v", dx=-4, dy=-9)
    L.text(1.0, 0.0, "min", "sm", dx=-6, dy=-10); L.text(-1.0, 0.0, "max", "sm", dx=6, dy=-10)
    Q = Panel(body, 452, 46, 290, 290, (0, 2 * math.pi), (0, 1.3))
    Q.frame([0, math.pi, 2 * math.pi], [0, 0.5, 1.0], "angle of the point on S", "D = ½ |point − P|²", "D along the circle", xtl={0: "0", math.pi: "π", 2 * math.pi: "2π"})
    D = 0.5 * ((np.cos(th) - 0.5) ** 2 + np.sin(th) ** 2)
    Q.line(th, D, "ln s1"); Q.dot(0, 0.125, "f2"); Q.dot(math.pi, 1.125, "f4")
    Q.text(0, 0.125, "0.125", "sm", dx=10, dy=-6); Q.text(math.pi, 1.125, "1.125", "sm", dx=8, dy=-8)
    body.append('<text class="sm" x="60" y="400">Both marked points satisfy the projection condition (the segment to P is orthogonal to S there). One is the closest point, the other the farthest:</text>')
    body.append('<text class="sm" x="60" y="418">the orthogonality condition of Theorem 1.4 is necessary for a minimum, not sufficient. Theorem 1.5 adds flatness of S to repair this.</text>')
    (out / "critical-points.svg").write_text(svg(780, 430, "Orthogonality is necessary, not sufficient",
        "Left: a unit circle S in the plane with P = (0.5, 0); the closest point (1, 0) and the farthest point (-1, 0) both have the segment to P orthogonal to the circle. Right: half the squared distance along the circle has a minimum 0.125 and a maximum 1.125, two critical points.", body))


def fig_legendre(out):
    th0, th1 = -0.4, 2.0
    f = lambda x: np.log1p(np.exp(x)); fs = lambda e: e * np.log(e) + (1 - e) * np.log(1 - e)
    e0, e1 = 1 / (1 + math.exp(-th0)), 1 / (1 + math.exp(-th1))                # eta0 = psi'(theta0), eta1 = psi'(theta1)
    body = []
    A = Panel(body, 56, 46, 290, 240, (-4, 4), (0, 4.2))
    A.frame([-4, -2, 0, 2, 4], [0, 1, 2, 3, 4], "θ", "ψ(θ)", "ψ(θ) = log(1 + e^θ), one coin")
    xs = np.linspace(-4, 4, 400)
    A.line(xs, f(xs), "ln s1")
    tang = lambda x: f(th0) + e0 * (x - th0)
    A.line(xs, tang(xs), "dash s0")
    A.dot(th0, f(th0), "f2"); A.dot(th1, f(th1), "f1")
    A.b.append(f'<line class="s2" stroke-width="3" x1="{A.X(th1):.1f}" y1="{A.Y(tang(th1)):.1f}" x2="{A.X(th1):.1f}" y2="{A.Y(f(th1)):.1f}"/>')
    gap = f(th1) - tang(th1)
    A.text(th1, (f(th1) + tang(th1)) / 2, f"gap = {gap:.4f}", "v", dx=-8, dy=4, anchor="end")
    A.text(th0, f(th0), "θ₀", "sm", dx=-4, dy=18, anchor="end"); A.text(th1, f(th1), "θ₁", "sm", dx=-8, dy=-6, anchor="end")
    B = Panel(body, 452, 46, 290, 240, (0, 1), (-1.5, 0.1))
    B.frame([0, 0.25, 0.5, 0.75, 1], [-1.2, -0.8, -0.4, 0], "η", "ψ*(η)", "ψ*(η) = η log η + (1−η) log(1−η)")
    es = np.linspace(0.002, 0.998, 500)
    B.line(es, fs(es), "ln s1")
    tangs = lambda e: fs(e1) + th1 * (e - e1)                                   # tangent of psi* at eta1 has slope theta1
    B.line(es, tangs(es), "dash s0")
    B.dot(e0, fs(e0), "f2"); B.dot(e1, fs(e1), "f1")
    gap2 = fs(e0) - tangs(e0)
    B.b.append(f'<line class="s2" stroke-width="3" x1="{B.X(e0):.1f}" y1="{B.Y(tangs(e0)):.1f}" x2="{B.X(e0):.1f}" y2="{B.Y(fs(e0)):.1f}"/>')
    B.text(e0, (fs(e0) + tangs(e0)) / 2, f"gap = {gap2:.4f}", "v", dx=8, dy=4)
    B.text(e0, fs(e0), "η₀", "sm", dx=-6, dy=-8, anchor="end"); B.text(e1, fs(e1), "η₁", "sm", dx=8, dy=0)
    body.append('<text class="sm" x="60" y="338">Left: the gap above the tangent at θ₀, measured at θ₁, is D_ψ[θ₁ : θ₀] — the fall of ψ below its tangent (Fig. 1.4 of the book).</text>')
    body.append(f'<text class="sm" x="60" y="356">Right: the same two coins in η = ψ′(θ), η₀ = {e0:.3f}, η₁ = {e1:.3f}. The gap above the tangent at η₁, measured at η₀, is D_ψ*[η₀ : η₁].</text>')
    body.append('<text class="sm" x="60" y="374">The two gaps are equal: the Legendre dual swaps the order of the two points and changes nothing else.</text>')
    (out / "legendre.svg").write_text(svg(780, 392, "The Legendre dual swaps the two points",
        f"Left: the convex function log(1 + e^theta) for one coin, its tangent at theta0 = -0.4, and the gap {gap:.4f} at theta1 = 2. Right: the dual function, the negative entropy of a coin, its tangent at eta1 and the gap {gap2:.4f} at eta0; the two divergences are equal with their arguments exchanged.", body))
    return gap, gap2


def fig_em(out):
    body = []
    A = Panel(body, 56, 46, 380, 230, (0, 24), (-15.5, 0))
    A.frame([0, 6, 12, 18, 24], [-15, -10, -5, 0], "half-step (projection onto K, then onto S)", "log₁₀(D_t − D*)", "alternating projections, five random starts")
    def mi_s(s):
        P = np.array([[0.4, s], [0.2 - s, 0.4]]); return kl(P.ravel(), np.outer(P.sum(1), P.sum(0)).ravel())
    d_star = mi_s(0.1)
    rng = np.random.default_rng(2)
    for trial, cls in zip(range(5), ("s1", "s2", "s3", "s4", "s0")):
        Q0 = np.outer(*rng.dirichlet(np.ones(2), 2))
        vals, _, _ = em_iterations(Q0, steps=12)
        ys = [math.log10(max(v - d_star, 1e-15)) for v in vals]
        A.line(np.arange(1, len(ys) + 1), ys, f"ln {cls}")
    for k, line in enumerate(("K = {P₀₀ = P₁₁ = 0.4} is m-flat; S = the independence model is e-flat.",
                              "Every start lowers D at every half-step (1.124) and all five reach the same",
                              "value; straight lines mean a constant shrink factor per step.")):
        body.append(f'<text class="sm" x="60" y="{338 + 18 * k}">{line}</text>')
    (out / "alternating.svg").write_text(svg(560, 392, "The em algorithm converges, and here to one value",
        f"Five random starts of the alternating projection between an m-flat set K and the e-flat independence model S. The divergence decreases at every half-step and all runs reach the minimum {d_star:.6f}.", body))
    return d_star


def make_figures():
    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig_legendre(out); fig_charts(out); fig_pythagoras(out); fig_projection(out); fig_critical(out); fig_em(out)
    print("\nwrote", ", ".join(sorted(p.name for p in out.glob("*.svg"))))


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    check_divergence(); check_bregman(); check_exp_family(); check_legendre(); check_flat_structures()
    check_pythagoras(); check_projection(); check_em(); check_coordinates()
    if "--figures" in sys.argv:
        make_figures()
