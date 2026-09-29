#!/usr/bin/env python3
"""Sections 3-4 of Achille, Paolini, Mbeng & Soatto: checks of the Kolmogorov-level claims.

Kolmogorov complexity is not computable, so nothing here computes K. What can be
computed is the cost of explicit coding schemes (an upper bound on K) and the
probability statements that the proofs rely on. Every quantity is in nats;
a description of t bits costs t ln 2 nats.

Checked here:

  1. Example 3.5 (structure function of random labels, S(t) ~ N log|Y| - t).
     The paper's mechanism, "memorise floor(t / log|Y|) points", reaches slope
     -1 only with the index oracle of Prop 3.2.4. Without it, saying *which*
     points are memorised costs log C(N, m) extra. A different scheme, trying
     2^t random labelings and keeping the best, does reach slope -1.
  2. Lemma A.1 is Wilks' theorem: with finitely many inputs,
     2 (L_D(p) - L_D(p_hat)) ~ chi^2 with |X|(|Y|-1) degrees of freedom, so the
     gap stays bounded as N grows. With distinct inputs (images) p_hat memorises
     and the gap grows like N H(y|x); the lemma, and the proof of Prop 3.3.2
     that uses it, need a finite input space.
  3. A proof of Prop 3.3.2 that needs no finite X: Markov's inequality on the
     likelihood ratio plus Kraft's inequality over all programs. Simulated on a
     family of 2^10 x 8 "memorising" candidates: the bound holds, with room.
  4. The beta-Lagrangian on the random-label structure function: for beta < 1
     memorising everything is optimal, for beta > 1 the constant is. This is why
     Theorem 4.4 (which cites Prop 3.3, a beta = 1 statement) needs beta >= 1
     on noisy labels.

Standard library and numpy only.

Run:  python3 kolmogorov_toys.py             (prints every number quoted in the notes)
      python3 kolmogorov_toys.py --figures   (also rewrites ../figures/structure-function.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np

LN2 = math.log(2)
rng = np.random.default_rng(0)


def H(q):
    """Binary entropy in nats."""
    q = np.clip(q, 1e-300, 1 - 1e-16)
    return -(q * np.log(q) + (1 - q) * np.log1p(-q))


def log2_binom(n, m):
    return (math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1)) / LN2


# ----------------------------------------------------------------------- 1
def seed_search_saving(N, t_bits, reps):
    """Best of 2^t random binary labelings against N random labels; the saving
    (in nats) of the optimal tilted predictor p(y = g(x)) = q over uniform."""
    M = 2 ** t_bits
    out = []
    for _ in range(reps):
        a = rng.binomial(N, 0.5, size=M).max()
        q = a / N
        out.append(N * (LN2 - H(q)))
    return float(np.mean(out))


def memorise_cost_nats(N, m, oracle):
    """Cost of the paper's scheme: m label bits, plus naming the m points."""
    bits = m + (math.log2(m + 1) if oracle else log2_binom(N, m))
    return bits * LN2


def check_random_labels():
    N = 2000
    print(f"1. Example 3.5: random binary labels, N = {N}, S(0) = N ln 2 = {N * LN2:.1f} nats")
    print("   seed search: try 2^t random labelings, keep the best, predict it with confidence q")
    print("     t (bits)  cost t ln2  saving   saving / cost")
    rows = []
    for t in (2, 4, 8, 12, 16):
        s = seed_search_saving(N, t, 40 if t < 16 else 10)
        rows.append((t, s))
        print(f"     {t:8d}  {t * LN2:9.2f}  {s:7.2f}   {s / (t * LN2):.3f}")
    print("     (the O(log N) bits that name q are left out; the large-deviation limit of the")
    print("      ratio is exactly 1: max over 2^t draws reaches N KL(q || 1/2) = t ln 2)")
    print("   the paper's scheme: memorise m labels exactly, uniform elsewhere; saving m ln 2")
    print("     m      cost with oracle   cost without     slope without oracle")
    for m in (10, 100, 1000, 1900):
        c1, c2 = memorise_cost_nats(N, m, True), memorise_cost_nats(N, m, False)
        print(f"     {m:5d}  {c1:14.1f}   {c2:12.1f}     {m * LN2 / c2:.3f}")
    return rows


# ----------------------------------------------------------------------- 2
def check_wilks():
    print("\n2. Lemma A.1 is Wilks' theorem (finite X), and fails for distinct inputs")
    nx, ny = 20, 3
    p = rng.dirichlet(np.ones(ny), size=nx)
    Hyx = float(np.mean(-(p * np.log(p)).sum(1)))
    df = nx * (ny - 1)
    print(f"   |X| = {nx}, |Y| = {ny}, random p(y|x), H(y|x) = {Hyx:.3f} nats; "
          f"Wilks predicts E[L(p) - L(p_hat)] -> df/2 = {df / 2:.0f}")
    for N in (600, 2400, 9600):
        gaps = []
        for _ in range(300):
            x = rng.integers(nx, size=N)
            u = rng.random(N)
            y = (u[:, None] > np.cumsum(p[x], 1)).sum(1)
            cnt = np.zeros((nx, ny))
            np.add.at(cnt, (x, y), 1)
            phat = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
            Lp = -np.log(p[x, y]).sum()
            Lh = -np.log(phat[x, y]).sum()
            gaps.append(Lp - Lh)
        gaps = np.array(gaps)
        print(f"   N = {N:5d}: mean gap {gaps.mean():6.2f}, sd {gaps.std():5.2f}, "
              f"95th pct {np.percentile(gaps, 95):6.2f}")
    print("   distinct inputs (every x_i different, as with images): p_hat memorises, L(p_hat) = 0,")
    for N in (600, 2400, 9600):
        print(f"   N = {N:5d}: gap = L(p) ~ N H(y|x) = {N * Hyx:8.1f}")


# ----------------------------------------------------------------------- 3
def check_markov_kraft():
    print("\n3. Prop 3.3.2 without finite X: Markov + Kraft")
    N, tb, Q = 400, 10, 8
    qs = np.linspace(0.52, 0.66, Q)       # 8 tilt levels -> 3 more bits
    K_bits = tb + 3                        # every candidate has a 13-bit description
    kraft = (2 ** tb) * Q * 2.0 ** (-K_bits)
    print(f"   true p = uniform on 2 labels, N = {N}, L_D(p) = N bits exactly")
    print(f"   candidates: 2^{tb} seeds x {Q} tilts, K = {K_bits} bits each, Kraft sum = {kraft:.3f}")
    print("   c (bits)   P(some candidate has L + K <= L(p) - c)   bound 2^-c x Kraft sum")
    reps = 4000
    best = np.empty(reps)
    for r in range(reps):
        a = rng.binomial(N, 0.5, size=2 ** tb).astype(float)[:, None]
        L = -(a * np.log2(qs) + (N - a) * np.log2(1 - qs))
        best[r] = L.min() + K_bits
    res = []
    for c in (0, 1, 2, 3, 4):
        pr = float(np.mean(best <= N - c))
        res.append((c, pr))
        print(f"   {c:8d}   {pr:39.4f}   {kraft * 2.0 ** -c:.4f}")
    return res


# ----------------------------------------------------------------------- 4
def check_lagrangian():
    N = 2000
    print("\n4. C_beta on the random-label structure function S(t) = (N - t) ln 2, t in bits")
    for beta in (0.5, 1.0, 2.0):
        c0 = N * LN2
        cN = beta * N * LN2
        pick = "memorise all (t = N)" if cN < c0 else ("tie" if cN == c0 else "constant (t = 0)")
        print(f"   beta = {beta:3.1f}: C at t = 0 is {c0:7.1f}, at t = N is {cN:7.1f}  ->  {pick}")


# ----------------------------------------------------------------- figures
def figures(seed_rows):
    from svgkit import Axes, legend, svg, write
    N = 2000
    body = ['<text class="hd" x="20" y="22">Example 3.5: the structure function of 2000 random binary labels</text>',
            '<text class="sub" x="20" y="38">Loss left after spending t nats of description. No scheme can go below the dashed line (Kolmogorov).</text>']
    ms = np.unique(np.round(np.geomspace(1, N, 400)).astype(int))
    cost_o = np.array([memorise_cost_nats(N, m, True) for m in ms])
    cost_n = np.array([memorise_cost_nats(N, m, False) for m in ms])
    loss = (N - ms) * LN2
    keep = cost_n <= N * LN2
    # main panel
    ax = Axes(62, 70, 330, 230, (0, N * LN2), (0, N * LN2))
    ticks = [0, 400, 800, 1200]
    body.append(ax.frame(ticks, ticks, "description length t (nats)", "loss S(t) (nats)"))
    tt = np.linspace(0, N * LN2, 50)
    body.append(ax.path(tt, N * LN2 - tt, "k"))
    body.append(ax.path(cost_n[keep], loss[keep], "o"))
    body.append(ax.path(cost_o, loss, "b dash"))
    body.append(f'<rect class="shade" x="{ax.X(0):.1f}" y="{ax.Y(N * LN2):.1f}" width="{ax.X(14) - ax.X(0) + 2:.1f}" height="{ax.Y(N * LN2 - 14) - ax.Y(N * LN2) + 2:.1f}"/>')
    body.append(legend(80, 232, [("k", "lower bound  S(t) ≥ N ln 2 − t"),
                                 ("b dash", "memorise m points, index oracle"),
                                 ("o", "memorise m points, no oracle")]))
    body.append(f'<circle class="gf" cx="90" cy="{232 + 48}" r="3.5"/>')
    body.append(f'<text class="lab" x="106" y="{232 + 52}">best of 2^t random labelings</text>')
    # zoom on the first 14 nats, where the simulation lives
    zx = Axes(470, 70, 190, 230, (0, 14), (N * LN2 - 14, N * LN2))
    body.append(zx.frame([0, 4, 8, 12], [1375, 1380, 1385], "t (nats), zoomed", "", yfmt="{:g}"))
    body.append(zx.path([0, 14], [N * LN2, N * LN2 - 14], "k"))
    zz = cost_n <= 14
    for cz, lz in zip(cost_n[zz], loss[zz]):
        body.append(zx.dot(cz, lz, "of", 3))
    body.append(zx.text(cost_n[zz][0], loss[zz][0], "one point, no oracle", "tiny", "end", dx=-6, dy=-4))
    oz = cost_o <= 14
    body.append(zx.path(np.r_[0, cost_o[oz]], np.r_[N * LN2, loss[oz]], "b dash"))
    for t, sv in seed_rows:
        if t * LN2 <= 14:
            body.append(zx.dot(t * LN2, N * LN2 - sv, "gf", 3.5))
    body.append(f'<text class="tiny" x="{zx.x0}" y="{zx.y0 - 8}">the shaded corner, enlarged</text>')
    half = N * LN2 / 2
    saved = N * LN2 - np.interp(half, cost_n[keep], loss[keep])
    body.append(f'<text class="lab" x="20" y="344">Without the oracle, memorising is far off the line: spending t = {half:.0f} nats saves only {saved:.0f}. '
                'Searching random labelings</text>')
    body.append('<text class="lab" x="20" y="359">stays close to it (and reaches it as t grows); that, not memorisation, is what makes S(t) ≈ N log|Y| − t true.</text>')
    write("structure-function.svg", svg(690, 372, "Structure function of random labels",
                                        "Loss against description length for 2000 random binary labels: the Kolmogorov lower "
                                        "bound, the paper's memorisation scheme with and without the index oracle, and random search.",
                                        body))


if __name__ == "__main__":
    rows = check_random_labels()
    check_wilks()
    check_markov_kraft()
    check_lagrangian()
    if "--figures" in sys.argv:
        figures(rows)
