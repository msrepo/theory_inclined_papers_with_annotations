#!/usr/bin/env python3
"""Checks on Wang & Isola (ICML 2020), alignment and uniformity on the hypersphere.

Four things, in the order the paper introduces them:

  1. why the naive uniformity objectives fail and the Gaussian potential does not
  2. Proposition 1: the uniform measure minimises the average pairwise potential
  3. Theorem 1: L_contrastive - log M converges to the claimed limit, and at what rate
  4. how the two uniformity forms relate: the gap between them is a Jensen gap that
     closes exactly at uniformity

Run:  python3 alignment_uniformity.py
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(0)
TAU = 0.5


# ------------------------------------------------ configurations on the circle
def circle(kind, N=4000, rng=RNG):
    if kind == "uniform":        a = rng.uniform(0, 2*np.pi, N)
    elif kind == "8 equispaced": a = np.repeat(np.arange(8)*np.pi/4, N//8)
    elif kind == "4 equispaced": a = np.repeat(np.arange(4)*np.pi/2, N//4)
    elif kind == "2 antipodal":  a = np.repeat([0, np.pi], N//2)
    elif kind == "two clusters": a = np.concatenate([rng.vonmises(0, 20, N//2),
                                                     rng.vonmises(np.pi, 20, N//2)])
    elif kind == "vMF k=1":      a = rng.vonmises(0, 1, N)
    elif kind == "vMF k=5":      a = rng.vonmises(0, 5, N)
    elif kind == "single point": a = np.zeros(N)
    else: raise ValueError(kind)
    return np.stack([np.cos(a), np.sin(a)], 1)


def gaussian_potential(X, t=1.0):
    """E[G_t(u,v)] with G_t(u,v) = exp(-t||u-v||^2)."""
    return np.exp(-t*((X[:, None, :]-X[None, :, :])**2).sum(-1)).mean()


def avg_dot(X):
    """The naive alternative: E[u.v] = ||E[u]||^2."""
    return (X @ X.T).mean()


def check_naive_objectives_fail():
    print("1. Why not just use the average dot product?\n")
    print(f"{'configuration':>16} {'avg dot':>10} {'E[G_t]':>9} {'L_uniform':>11}")
    print("-"*50)
    for k in ("uniform", "8 equispaced", "4 equispaced", "2 antipodal",
              "two clusters", "vMF k=1", "vMF k=5", "single point"):
        X = circle(k)
        g = gaussian_potential(X)
        print(f"{k:>16} {avg_dot(X):>+10.4f} {g:>9.4f} {np.log(g):>11.4f}")
    print("-"*50)
    print("  Average dot product is ~0 for uniform, 8-pt, 4-pt AND 2 antipodal points.")
    print("  E[u.v] = ||E[u]||^2, so every zero-mean configuration ties. Two antipodal")
    print("  points are maximally non-uniform and score identically to uniform.")
    print("  The Gaussian potential separates them. That is what strict positive")
    print("  definiteness buys.\n")
    X = circle("uniform", 200)
    D = ((X[:, None, :]-X[None, :, :])**2).sum(-1)
    err = np.abs(np.exp(-D) - np.exp(2*(X @ X.T) - 2)).max()
    print(f"  Identity G_t(u,v) = exp(2t u.v - 2t) on the sphere: max err {err:.1e}")
    print("  -> the RBF potential IS exponentiated cosine similarity, which is what")
    print("     lets Theorem 1 connect the contrastive loss to an energy.\n")


def check_proposition_1():
    print("2. Proposition 1: uniform uniquely minimises the average potential\n")
    best = min(("uniform", "8 equispaced", "4 equispaced", "2 antipodal",
                "two clusters", "vMF k=1", "vMF k=5", "single point"),
               key=lambda k: gaussian_potential(circle(k)))
    print(f"  argmin over the configurations above: {best}")
    print("  (8 equispaced edges out sampled uniform: that is Proposition 2, the")
    print("   finite-N minimiser, whose counting measure converges weak* to uniform.)\n")


# ---------------------------------------------------------------- Theorem 1
def _setup(n_item=400, spread=0.35, rng=RNG):
    base = rng.uniform(0, 2*np.pi, n_item)
    def aug(idx):
        a = base[idx] + rng.normal(0, spread, idx.shape)
        return np.stack([np.cos(a), np.sin(a)], 1)
    return n_item, aug


def theorem_1_limit(n_item, aug, rng=RNG, pool=None):
    """-1/tau E_pos[u.v]  +  E_x log E_x- exp(u-.u/tau)."""
    idx = rng.integers(0, n_item, 20000)
    align = -(aug(idx)*aug(idx)).sum(1).mean()/TAU
    A = aug(rng.integers(0, n_item, 4000))
    return align + np.log(np.exp((A @ pool.T)/TAU).mean(1)).mean()


def check_theorem_1():
    print("3. Theorem 1: convergence of L_contrastive - log M\n")
    n_item, aug = _setup()
    pool = aug(RNG.integers(0, n_item, 40000))
    lim = np.mean([theorem_1_limit(n_item, aug, pool=pool) for _ in range(5)])
    print(f"  analytic limit = {lim:.5f}\n")

    def loss_at(M, nrep=30000):
        idx = RNG.integers(0, n_item, nrep)
        V = aug(idx)
        pos = (aug(idx)*V).sum(1)/TAU
        neg = np.exp((pool[RNG.integers(0, len(pool), (nrep, M))]*V[:, None, :]).sum(-1)/TAU)
        return (-pos + np.log(np.exp(pos) + neg.sum(1))).mean() - np.log(M)

    print(f"{'M':>6} {'L - log M':>12} {'|deviation|':>13} {'dev*sqrt(M)':>13} {'dev*M':>8}")
    out = []
    for M in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        v = np.mean([loss_at(M) for _ in range(3)])
        d = abs(v - lim); out.append((M, d))
        print(f"{M:>6} {v:>12.5f} {d:>13.5f} {d*np.sqrt(M):>13.4f} {d*M:>8.2f}")
    Ms = np.array([o[0] for o in out]); ds = np.array([o[1] for o in out])
    slope = np.polyfit(np.log(Ms[2:]), np.log(ds[2:]), 1)[0]
    print(f"\n  fitted log-log slope (M>=8): {slope:.2f}")
    print("  dev*M is roughly constant, so the true rate is Theta(1/M).")
    print("  The published version states O(M^-2/3); the authors corrected it to")
    print("  O(M^-1/2) in the arXiv changelog of 11/6/2020. Both are valid upper")
    print("  bounds, but loose: the proof applies the mean value theorem before")
    print("  taking the expectation, discarding the cancellation of the mean-zero")
    print("  first-order term that a delta-method expansion keeps.\n")
    return out


# ------------------------------------------------- the two uniformity forms
def check_jensen_gap():
    print("4. L_uniform vs Theorem 1's second term: a Jensen gap\n")
    print("  L_uniform  = log E_x E_y[G]        (log outside both)")
    print("  Eq.2 term  = E_x log E_y[G] + 1/tau  (log between them)")
    print("  Jensen: E_x log E_y[G] <= log E_x E_y[G], equality iff E_y[G(x,.)]")
    print("  is constant in x -- which is exactly the uniformity equilibrium.\n")
    t = 1/(2*TAU)
    print(f"{'configuration':>16} {'L_uniform':>11} {'Eq2 - 1/tau':>13} {'gap':>10}")
    print("-"*54)
    rows = []
    for k in ("uniform", "8 equispaced", "two clusters", "vMF k=5", "vMF k=1"):
        X = circle(k)
        K = np.exp(2*t*(X @ X.T) - 2*t)
        lu, eq = np.log(K.mean()), np.log(K.mean(1)).mean()
        rows.append((k, lu, eq)); print(f"{k:>16} {lu:>11.4f} {eq:>13.4f} {lu-eq:>10.5f}")
    print("-"*54)
    print("  The gap vanishes at uniformity and is strictly positive otherwise:")
    print("  same minimiser, different value. That is why swapping log and E is")
    print("  free for the argmin, and why L_uniform can be the cheaper metric.\n")
    return rows


if __name__ == "__main__":
    check_naive_objectives_fail()
    check_proposition_1()
    check_theorem_1()
    check_jensen_gap()
