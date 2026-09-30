#!/usr/bin/env python3
"""The closed forms of Tahir, Ganguli & Rotskoff (ICML 2025), in one place.

Everything here is a plain function of (gamma, theta, sigma) so that the checks, the figures and the
interactive page all quote the same formulas. Notation follows the paper:

    gamma = n/d          target sample size relative to dimension
    theta                angle between the source and target weight vectors, cos(theta) = beta_s . beta_t
    sigma                label-noise standard deviation; s2 = sigma^2

"asym" functions are the n, d -> infinity limits printed in the paper (eq. 8, 11, 14, 15).
"exact" functions are the finite-n, finite-d expectations for Gaussian data, which the proofs in
Appendix D actually give before the limit is taken (derived in the notes, checked in transfer_theory.py).

Standard library and numpy only. Running this file prints a one-line smoke test.
"""
from __future__ import annotations

import math

import numpy as np


# ------------------------------------------------------------------ scratch training, eq. (8)
def r_scratch(g, s2):
    """E R_sc, eq. (8): minimum-norm least squares, ||beta_t|| = 1, n, d -> infinity, gamma = n/d.
    Diverges at gamma = 1 (the interpolation threshold)."""
    g = np.asarray(g, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        under = ((1 - g) ** 2 + g * s2) / (1 - g)      # gamma < 1: overparameterised, bias (1-g) + variance
        over = s2 / (g - 1)                            # gamma > 1: ordinary least squares
        return np.where(g < 1, under, over)


def r_scratch_exact(n, d, s2):
    """Exact for Gaussian X: 1 - n/d + s2 n/(d-n-1) if n < d-1, s2 d/(n-d-1) if n > d+1."""
    if n < d - 1:
        return 1 - n / d + s2 * n / (d - n - 1)
    if n > d + 1:
        return s2 * d / (n - d - 1)
    return math.inf


# ------------------------------------------------------------------ linear transfer, eq. (10)-(11)
def r_lt_exact(n, th, s2):
    """Theorem 3.7, valid for every n > 2: sin^2(th) + (s2 + sin^2 th)/(n-2)."""
    return np.sin(th) ** 2 + (s2 + np.sin(th) ** 2) / (n - 2)


def r_lt_asym(th):
    """The n -> infinity limit of eq. (10): the irreducible term sin^2(theta)."""
    return np.sin(th) ** 2


def t_lt(g, th, s2):
    """Eq. (11): T_lt = R_sc - sin^2(theta)."""
    return r_scratch(g, s2) - np.sin(th) ** 2


def t_lt_exact(n, d, th, s2):
    return r_scratch_exact(n, d, s2) - r_lt_exact(n, th, s2)


def neg_transfer_interval(th, s):
    """For gamma < 1: T_lt < 0 exactly for gamma in (g_minus, g_plus); None if empty.

    Solving ((1-g)^2 + g s2)/(1-g) < sin^2 th for g in (0,1) is the quadratic
        g^2 - (1 + cos^2 th - s2) g + cos^2 th < 0.
    Real roots need 1 - cos th > s, i.e. theta > arccos(1 - s).
    """
    c2 = math.cos(th) ** 2
    b = 1 + c2 - s * s
    disc = b * b - 4 * c2
    if disc <= 0 or b <= 0:
        return None
    r = math.sqrt(disc)
    return (b - r) / 2, (b + r) / 2


def neg_transfer_over(th, s):
    """For gamma > 1: T_lt < 0 exactly when gamma > 1 + s2 / sin^2(theta) (never for theta = 0)."""
    sn = math.sin(th) ** 2
    return math.inf if sn == 0 else 1 + s * s / sn


# ------------------------------------------------------------------ ridge linear transfer, eq. (13)
def r_ridge_limit(th, lam):
    """Eq. (13): n -> infinity limit of the ridge-regularised linear-transfer risk."""
    return 1 - (1 + 2 * lam) / (1 + lam) ** 2 * np.cos(th) ** 2


# ------------------------------------------------------------------ fine-tuning, eq. (14)-(15)
def t_ft(g, th):
    """Eq. (15): T_ft = (1-g)(2 cos th - 1) for g <= 1, and 0 for g > 1."""
    g = np.asarray(g, dtype=float)
    return np.where(g <= 1, (1 - g) * (2 * np.cos(th) - 1), 0.0)


# ------------------------------------------------------------------ fine-tuning, with the balancedness factor
def c_balance(g, s2):
    """The scale c on the null-space part of the fine-tuned solution (derived in the notes).

    Fine-tuning a two-layer linear net from the pretrained state keeps W1 = u v', W2 = c v with the
    conserved quantity |u|^2 - c^2 = 0, and ends at beta_ft = beta_sc + c (I-P) beta_s. Writing
    a = |(I-P) beta_s|^2 -> 1-g and b = |beta_sc|^2 -> g + s2 g/(1-g), c solves c^4 - a c^2 - b = 0.
    The paper's Theorem 3.9 is the case c = 1, which is exact only when s2 = 0 (then a + b = 1).
    Valid for g < 1.
    """
    g = np.asarray(g, dtype=float)
    a = 1 - g
    b = g + s2 * g / (1 - g)
    return np.sqrt((a + np.sqrt(a * a + 4 * b)) / 2)


def t_ft_balanced(g, th, s2):
    """T_ft with the balancedness factor: (1-g)(2 c cos th - c^2). Equals eq. (15) when c = 1."""
    g = np.asarray(g, dtype=float)
    gg = np.minimum(g, 0.999999)
    c = c_balance(gg, s2)
    return np.where(g <= 1, (1 - g) * (2 * c * np.cos(th) - c * c), 0.0)


def t_ft_finite_source(gs, gt, th, s2s):
    """Eq. (22): fine-tuning with a finite source set of gs*d noisy points (noise variance s2s).

    Both rows are (gt - 1) * E[ ||beta_hat_s||^2 - 2 <beta_t, beta_hat_s> ] with
    E||beta_hat_s||^2 = gs + s2s gs/(1-gs) (gs < 1) or 1 + s2s/(gs-1) (gs > 1)   [eq. 152]
    E<beta_t, beta_hat_s> = cos(th) min(1, gs).
    """
    if gt > 1:
        return 0.0
    if gs <= 1:
        return (gt - 1) * gs * (1 - 2 * math.cos(th) + s2s / (1 - gs))
    return (gt - 1) * (1 - 2 * math.cos(th) + s2s / (gs - 1))


def cos_theta_star(gs, s2s):
    """Eq. (23): fine-tuning helps iff cos(theta) > cos_theta_star."""
    return 0.5 * (s2s + 1 - gs) / (1 - gs) if gs < 1 else 0.5 * (s2s + gs - 1) / (gs - 1)


if __name__ == "__main__":
    print("theory.py: T_lt(gamma=0.5, theta=pi/4, sigma=0.2) = %.4f, T_ft = %.4f"
          % (t_lt(0.5, math.pi / 4, 0.04), t_ft(0.5, math.pi / 4)))
