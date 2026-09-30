#!/usr/bin/env python3
"""Section 4 and Appendix C of Gao & Chaudhari: an audit of the Rademacher argument (Theorems 5, 6, 7).

The chain of Theorem 6's proof is

    gap_k <= phi_k := sup_{w in ball_k} ( E_p loss - (1/N) sum loss )        (16)-(17)
    E phi_k <= 2 R_N(ball_k)                                                 symmetrisation
    E exp(lam (phi_k - E phi_k)) <= exp(lam^2 M^2 / 8)                       "Hoeffding's lemma"   (18)
    P( sum_k phi_k > K eps ) <= exp( -lam K eps + sum_k 2 lam R_k + K lam^2 M^2 / 8 )   Markov       (19)
    lam = 4 K (eps - 2 Rbar) / M^2 ;  result  exp( -2K (eps - 2 Rbar)^2 / M^2 )          (15)

Checked here:
  1. The choice of lambda. The minimiser of the exponent in (19) is lam* = 4 (eps - 2 Rbar) / M^2; the printed
     value has an extra factor K, and at the printed value the exponent is positive.
  2. The bound (15) has no N in the exponent. McDiarmid's inequality, applied to phi_k (which changes by at most
     M/N when one sample is replaced), gives exp(-2 N K (eps - 2 Rbar)^2 / M^2). Table of the two.
  3. A Monte-Carlo check of the chain on a finite class (thresholds with 0-1 loss): symmetrisation holds
     with room to spare, both bounds are valid, and the printed one is far above the truth.
  4. Theorem 7's complexity term along the minimiser path of transfer_toy.py: sum_k dtau_k E|dl_k| is
     T_K / K, which tends to 0, while sum_k E|dl_k| = T_K tends to the total variation of the loss (the
     Fisher-Rao length under the empirical-Fisher reading of (21)).

Standard library and numpy only.

Run:  python3 bounds.py             (prints every number quoted in the notes)
      python3 bounds.py --figures   (also rewrites ../figures/bound-in-N.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np


# ------------------------------------------------------------------------------ 1. lambda
def check_lambda():
    print("1. The optimiser lambda in (19)")
    K, M, eps, Rbar = 10, 1.0, 0.5, 0.1
    A = K * eps - 2 * K * Rbar  # K eps - 2 sum R_k

    def expo(lam):
        return -lam * A + K * lam ** 2 * M ** 2 / 8

    lam_star = 4 * A / (K * M ** 2)
    lam_paper = 4 * K * (eps - 2 * Rbar) / M ** 2
    print(f"   K = {K}, M = {M}, eps = {eps}, Rbar = {Rbar}  (so K eps - 2 sum R = {A})")
    print(f"   minimiser of the exponent:  lam* = 4 (eps - 2 Rbar) / M^2 = {lam_star:.3f}, exponent {expo(lam_star):.3f},"
          f" bound exp = {math.exp(expo(lam_star)):.4f}")
    print(f"   -2 K (eps - 2 Rbar)^2 / M^2 (eq. 15)                                  = {-2 * K * (eps - 2 * Rbar) ** 2 / M ** 2:.3f}")
    print(f"   printed value  lam = 4 K (eps - 2 Rbar) / M^2 = {lam_paper:.3f}, exponent {expo(lam_paper):+.3f}"
          f"   (closed form (2K - 4) A^2 / M^2 = {(2 * K - 4) * A ** 2 / M ** 2:.3f})")
    print(f"   at the printed lambda the 'bound' is exp({expo(lam_paper):+.1f}), which is not a bound at all;"
          f" the displayed result (15) is the one obtained with lam*")
    print()


# ------------------------------------------------------------------------------ 2. N in the exponent
def bound_printed(K, M, eps, Rbar):
    x = eps - 2 * Rbar
    return math.exp(-2 * K * x * x / M ** 2) if x > 0 else 1.0


def bound_mcdiarmid(K, N, M, eps, Rbar):
    x = eps - 2 * Rbar
    return math.exp(-2 * N * K * x * x / M ** 2) if x > 0 else 1.0


def check_scaling_in_N():
    print("2. The printed bound (15) against the same argument with McDiarmid in place of Hoeffding's lemma")
    print("   M = 1, Rbar = 0 (the most favourable case), K independent time points")
    print("      K    eps |  printed (15) | N=100: exp(-2NK eps^2) | N=1000")
    for K in (5, 20):
        for eps in (0.05, 0.1, 0.2):
            print(f"   {K:4}  {eps:5} |   {bound_printed(K, 1, eps, 0):8.4f}    |     {bound_mcdiarmid(K, 100, 1, eps, 0):10.3e}      | {bound_mcdiarmid(K, 1000, 1, eps, 0):10.3e}")
    d = 0.05
    print(f"   smallest eps - 2 Rbar the bound certifies at confidence {1 - d:.0%} (M = 1):")
    for K in (5, 20, 100):
        eps_p = math.sqrt(math.log(1 / d) / (2 * K))
        print(f"      K = {K:3}: printed {eps_p:.3f};  McDiarmid, N = 1000: {eps_p / math.sqrt(1000):.4f};  N = 50000: {eps_p / math.sqrt(50000):.5f}")
    print("   the printed bound cannot certify a gap below M sqrt(ln(1/delta) / 2K) however large N is; Hoeffding's lemma is applied")
    print("   to phi_k as if it were a single [0, M]-valued variable, whereas phi_k moves by at most M/N per sample")
    print()


# ------------------------------------------------------------------------------ 3. Monte Carlo
TS = np.linspace(0, 1, 51)  # thresholds
FLIP = 0.1
T0 = 0.4


def true_risk():
    return 0.1 + 0.8 * np.abs(TS - T0)  # P(error) of the threshold t: flips everywhere, plus the sliver between t and 0.4


def draw(rng, N, size):
    z = rng.random((size, N))
    y = ((z > T0) ^ (rng.random((size, N)) < FLIP)).astype(np.int8)
    return z, y


def loss_matrix(z, y):
    """z, y: (size, N).  returns (size, len(TS), N) 0-1 losses of every threshold."""
    pred = (z[:, None, :] > TS[None, :, None]).astype(np.int8)
    return (pred != y[:, None, :]).astype(np.float32)


def check_montecarlo():
    print("3. Monte Carlo on a finite class: 51 threshold classifiers, 0-1 loss (M = 1), N = 50 samples per time point")
    rng = np.random.default_rng(0)
    N, K = 50, 5
    R = true_risk()
    # phi = sup_t ( R(t) - empirical risk )
    reps = 40000
    phis = np.empty(reps)
    gaps_fixed = np.empty(reps)  # gap of one fixed hypothesis, t = 0.4 (index 20)
    fixed = int(np.argmin(np.abs(TS - T0)))
    for a in range(0, reps, 4000):
        z, y = draw(rng, N, 4000)
        L = loss_matrix(z, y)
        emp = L.mean(axis=2)
        phis[a:a + 4000] = (R[None, :] - emp).max(axis=1)
        gaps_fixed[a:a + 4000] = R[fixed] - emp[:, fixed]
    # Rademacher complexity R_N = E_S E_sigma sup_t (1/N) sum sigma_i l_t(z_i, y_i)
    rn = []
    for _ in range(400):
        z, y = draw(rng, N, 50)
        L = loss_matrix(z, y)
        sig = rng.choice([-1.0, 1.0], size=(50, 1, N)).astype(np.float32)
        rn.append(((L * sig).mean(axis=2)).max(axis=1))
    Rn = float(np.mean(rn))
    print(f"   E phi = {phis.mean():.4f}    2 R_N = {2 * Rn:.4f}    (symmetrisation: E phi <= 2 R_N {'holds' if phis.mean() <= 2 * Rn else 'FAILS'})")
    # tail of the average of K independent phi's
    idx = rng.integers(0, reps, size=(200000, K))
    mean_phi = phis[idx].mean(axis=1)
    print(f"   average of K = {K} independent phi_k: mean {mean_phi.mean():.4f}, sd {mean_phi.std():.4f}, max seen {mean_phi.max():.3f}")
    print("      eps  | P(avg phi > eps) | printed (15) | McDiarmid version")
    for eps in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
        emp_p = float(np.mean(mean_phi > eps))
        print(f"     {eps:5} |    {emp_p:9.5f}     |   {bound_printed(K, 1, eps, Rn):7.4f}    |   {bound_mcdiarmid(K, N, 1, eps, Rn):9.5f}")
    print(f"   (2 R_N = {2 * Rn:.3f}: both bounds are trivial, = 1, below this eps.)")
    print("   a single fixed hypothesis (t = 0.4), no supremum needed:")
    mean_gap = np.array([gaps_fixed[rng.integers(0, reps, size=K)].mean() for _ in range(20000)])
    print("      eps  | P(avg gap > eps) | Hoeffding exp(-2 N K eps^2)")
    for eps in (0.02, 0.05, 0.08, 0.1):
        print(f"     {eps:5} |    {float(np.mean(mean_gap > eps)):9.5f}     |   {math.exp(-2 * N * K * eps ** 2):9.5f}")
    print("   for a hypothesis chosen in advance the complexity term and the sup are unnecessary; they are only needed")
    print("   when w(tau_k) is picked using the very sample it is evaluated on, which is the case in the paper")
    print()
    return Rn


# ------------------------------------------------------------------------------ 4. Theorem 7 scaling
def check_thm7_scaling():
    print("4. Theorem 7: what the complexity term sum_k dtau_k E|dl_k| tends to")
    from transfer_toy import SOURCE, displacement, lengths, minimiser_path, shifted

    tgt = shifted(SOURCE, 4.0)
    n = 1024
    ts, ws = minimiser_path(SOURCE, tgt, displacement, n=n)
    A, B, TV = lengths(SOURCE, tgt, displacement, ts, ws)
    print(f"   translated task, s = 4, minimiser path: Fisher-Rao lengths A = {A:.3f}, B = {B:.3f}; total variation of the loss TV = {TV:.3f}")
    print("      K |  T_K = sum_k E|dl_k|  |  S_K = sum_k dtau_k E|dl_k| = T_K / K | 2 S_K")
    out = []
    for K in (2, 4, 16, 64, 256, 1024):
        step = n // K
        T = 0.0
        for k in range(1, K + 1):
            tk, tprev = ts[k * step], ts[(k - 1) * step]
            X, Y, W = displacement(SOURCE, tgt, tk).nodes()
            phi = np.stack([X, np.ones_like(X)], axis=1)
            sgn = np.where(Y == 1, 1.0, -1.0)
            l_now = np.logaddexp(0.0, -sgn * (phi @ ws[k * step]))  # -log p_w(y | x), computed stably
            l_prev = np.logaddexp(0.0, -sgn * (phi @ ws[(k - 1) * step]))
            T += float(np.sum(W * np.abs(l_now - l_prev)))
        out.append((K, T, T / K))
        print(f"   {K:5} |       {T:8.4f}        |            {T / K:9.5f}                   | {2 * T / K:9.5f}")
    print("   S_K falls like 1/K; only T_K, without the extra dtau_k, converges to the length-like quantity")
    print()
    return out


def figures(Rn):
    from svgkit import Axes, legend, svg, write

    W, H = 640, 300
    K, eps = 5, 0.03
    ax = Axes(60, 30, 520, 200, (10, 10000), (1e-30, 1.0), logx=True, logy=True)
    body = [ax.frame([10, 100, 1000, 10000], [1e-30, 1e-20, 1e-10, 1], "N (samples per time point)", "bound on P(average gap > eps)",
                     yfmt="{:.0e}")]
    Ns = np.logspace(1, 4, 60)
    body.append(ax.path(Ns, [bound_printed(K, 1, eps, 0) for _ in Ns], "r"))
    body.append(ax.path(Ns, [max(bound_mcdiarmid(K, int(n), 1, eps, 0), 1e-30) for n in Ns], "b"))
    body.append(legend(300, 100, [("r", "printed (15): no N in the exponent"), ("b", "with McDiarmid: exp(-2NK eps^2 / M^2)")], 16))
    body.append('<text class="tiny" x="300" y="140">K = 5 time points, eps = 0.03, M = 1, Rbar = 0</text>')
    write("bound-in-N.svg", svg(W, H, "The printed bound does not improve with N",
                                "With Hoeffding's lemma applied to phi the bound is flat in N.", body))


if __name__ == "__main__":
    check_lambda()
    check_scaling_in_N()
    rn = check_montecarlo()
    check_thm7_scaling()
    if "--figures" in sys.argv:
        figures(rn)
