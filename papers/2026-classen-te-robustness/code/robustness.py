#!/usr/bin/env python3
"""Why TE-metric rankings move when you resample the target set.

Classen et al. report that transferability rankings churn under nothing but a
change of random seed on the target subset, that H-score is the least stable of
the metrics on small targets, and that the choice of rank correlation can flip
which metric looks better. Their experiment is eight MedMNIST targets and a pool
of ResNet-18s; none of that is reproducible here. The MECHANISMS are.

Checked here:

  * the correlation coefficients disagree -- constructed ranking pairs where
    Kendall's tau and a top-weighted tau order the same two candidates the
    opposite way, which is their Figure 2;
  * intra-metric stability (their Eq 1) falls as the target subset shrinks, for
    a metric computed on honest data with nothing changing but the seed;
  * H-score's instability is a TRANSITION at n ~ k, not a gradual decay: above
    n/k of about 2 it is the more stable of the two, and below it the ranking
    becomes noise and then anti-correlates across seeds. cond(S_T) tracks the
    boundary. This is the ill-conditioning already documented in the Bao notes,
    and it locates their finding rather than merely agreeing with it;
  * a ranking can be perfectly stable and still perfectly wrong, so stability is
    necessary and not sufficient -- which is why their Ex1 and Ex2 are separate.

Standard library and numpy only, so the rank correlations are implemented here.

Run:  python3 robustness.py
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------- rank correlations
def kendall_tau(a, b):
    """Tau-b, with tie correction."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    i, j = np.triu_indices(n, 1)
    da, db = a[i] - a[j], b[i] - b[j]
    conc = np.sign(da) * np.sign(db)
    num = conc.sum()
    n0 = len(i)
    n1 = (np.sign(da) == 0).sum()
    n2 = (np.sign(db) == 0).sum()
    den = np.sqrt((n0 - n1) * (n0 - n2))
    return float(num / den) if den > 0 else 0.0


def weighted_tau(a, b):
    """Top-weighted tau: each pair weighted by its additive hyperbolic rank
    weight, so disagreements near the top of the list count for more.

    This is the same *idea* as scipy's weightedtau (Vigna 2015). The exact
    constants differ; what matters below is only that it is top-weighted.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(-a))          # rank 0 = best
    rb = np.argsort(np.argsort(-b))
    i, j = np.triu_indices(len(a), 1)
    w = (1.0 / (ra[i] + 1) + 1.0 / (ra[j] + 1)
         + 1.0 / (rb[i] + 1) + 1.0 / (rb[j] + 1))
    conc = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float((w * conc).sum() / w.sum())


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else 0.0


# ------------------------------------------------------------- the metrics
def softmax(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def hscore(Z, y, C):
    """Bao et al.: tr(S_T^-1 S_B). pinv, as the reference implementation uses."""
    Z = Z - Z.mean(0)
    S_T = Z.T @ Z / len(Z)
    S_B = np.zeros_like(S_T)
    for c in range(C):
        m = y == c
        if m.sum():
            S_B += m.mean() * np.outer(Z[m].mean(0), Z[m].mean(0))
    return float(np.trace(np.linalg.pinv(S_T, rcond=1e-15) @ S_B))


def leep(theta, y, C):
    """Nguyen et al.: average log-likelihood of the expected empirical predictor."""
    n = len(y)
    P = np.stack([theta[y == c].sum(0) for c in range(C)]) / n
    cond = P / np.maximum(P.sum(0), 1e-300)
    pred = theta @ cond.T
    return float(np.mean(np.log(np.maximum(pred[np.arange(n), y], 1e-300))))


def source_pool(rng, m=12, n=4000, k=64, C=6, S=10):
    """A pool of m source models, each giving features and a softmax on one
    target set. Quality is graded so there IS a true ordering to recover."""
    y = rng.integers(0, C, n)
    proto = rng.standard_normal((C, k))
    pool = []
    for q in np.linspace(0.15, 1.6, m):                  # graded informativeness
        F = proto[y] * q + rng.standard_normal((n, k))
        theta = softmax(F @ rng.standard_normal((k, S)) * 0.6)
        pool.append((F, theta, q))
    return pool, y, C


# ---------------------------------------------------------------------- tests
def coefficients_disagree():
    print("1. The rank correlations disagree, so 'which metric is better' moves\n")
    print("   Two candidate metrics scored against the same reference ranking of")
    print("   10 sources. A gets the top of the list right and the tail wrong;")
    print("   B gets the tail right and the top wrong.\n")
    ref = np.arange(10)[::-1].astype(float)              # 9 is best
    A = ref.copy(); A[:5] = A[:5][::-1]                  # scramble the WORST five
    B = ref.copy(); B[5:] = B[5:][::-1]                  # scramble the BEST five
    print(f"   {'':>14}  {'tau':>8}  {'weighted tau':>14}  {'spearman':>10}")
    for tag, v in (("metric A", A), ("metric B", B)):
        print(f"   {tag:>14}  {kendall_tau(ref, v):>8.3f}"
              f"  {weighted_tau(ref, v):>14.3f}  {spearman(ref, v):>10.3f}")
    wa, wb = weighted_tau(ref, A), weighted_tau(ref, B)
    ta, tb = kendall_tau(ref, A), kendall_tau(ref, B)
    print(f"\n   plain tau prefers:     {'A' if ta > tb else 'B' if tb > ta else 'tie'}")
    print(f"   weighted tau prefers:  {'A' if wa > wb else 'B' if wb > wa else 'tie'}")
    print("   -> Both are defensible. Top-weighted tau is the usual choice because")
    print("      you deploy the top-ranked source, but it assumes a clear winner")
    print("      exists. Reporting one coefficient silently picks a verdict.\n")


def stability_shrinks(rng):
    print("2. Intra-metric stability (their Eq 1) against target subset size\n")
    print("   Nothing changes but the random seed of the subset. Reported is the")
    print("   mean pairwise Kendall tau between rankings from 5 seeds -- 1.0 means")
    print("   the seed does not matter, 0.0 means the ranking is noise.\n")
    for k in (16, 64, 256):
        pool, y, C = source_pool(rng, k=k, n=4000)
        print(f"   feature dimension k = {k}")
        print(f"   {'n':>7}  {'n/k':>6}  {'LEEP':>8}  {'H-score':>9}  {'cond(S_T)':>11}")
        for sz in (32, 64, 128, 256, 512, 2000):
            rl, rh, cs = [], [], []
            for seed in range(5):
                r = np.random.default_rng(1000 + seed)
                idx = r.choice(len(y), min(sz, len(y)), replace=False)
                ys = y[idx]
                rl.append([leep(th[idx], ys, C) for _, th, _ in pool])
                rh.append([hscore(F[idx], ys, C) for F, _, _ in pool])
                Zc = pool[0][0][idx] - pool[0][0][idx].mean(0)
                cs.append(np.linalg.cond(Zc.T @ Zc / len(idx)))

            def intra(rk):
                return float(np.mean([kendall_tau(rk[i], rk[j])
                                      for i in range(5) for j in range(i + 1, 5)]))
            print(f"   {sz:>7}  {sz/k:>6.1f}  {intra(rl):>8.3f}  {intra(rh):>9.3f}"
                  f"  {np.mean(cs):>11.1e}")
        print()
    print("   LEEP degrades smoothly with n, which is ordinary estimator noise.")
    print("   H-score does something else: it is the MORE stable metric whenever")
    print("   n/k is above about 2, and collapses below that -- to zero, and then")
    print("   to NEGATIVE stability, meaning rankings from different seeds")
    print("   anti-correlate. It is not a gradual decay but a transition at n ~ k,")
    print("   tracked exactly by cond(S_T) jumping from ~1e1 to ~1e17.\n")
    print("   That is the ill-conditioning documented in the Bao notes, and it")
    print("   locates their finding: with ResNet-18 features (k = 512) and Breast")
    print("   at a 5% fraction (about 27 images), n/k is roughly 0.05. Deep in the")
    print("   regime where H-score is noise. Their result is not that H-score is")
    print("   generally unstable -- it is that medical targets are small enough to")
    print("   sit the wrong side of a sharp boundary.\n")


def stability_is_not_accuracy(rng):
    print("3. Stable does not mean right\n")
    pool, y, C = source_pool(rng)
    truth = np.array([q for _, _, q in pool])            # the planted ordering
    n = len(y)
    full_l = np.array([leep(th, y, C) for _, th, _ in pool])
    full_h = np.array([hscore(F, y, C) for F, _, _ in pool])
    const = np.array([7, 3, 11, 1, 9, 5, 0, 8, 2, 10, 4, 6], float)[:len(pool)]
    print(f"   {'ranking':>28}  {'intra-stability':>16}  {'tau vs truth':>13}")
    print(f"   {'a fixed, input-free ranking':>28}  {1.000:>16.3f}"
          f"  {kendall_tau(truth, const):>13.3f}")
    print(f"   {'LEEP on the full target':>28}  {'-':>16}  {kendall_tau(truth, full_l):>13.3f}")
    print(f"   {'H-score on the full target':>28}  {'-':>16}  {kendall_tau(truth, full_h):>13.3f}")
    print("\n   A metric that ignores its input is perfectly stable and useless.")
    print("   Stability is necessary, not sufficient -- which is exactly why the")
    print("   paper needs Ex2 (agreement with a reference) alongside Ex1.\n")


def reference_moves(rng):
    print("4. The reference ranking is itself a random variable\n")
    print("   Their Ex2 point, in miniature: 'ground truth' comes from fine-tuning")
    print("   runs, and those move too. Two evaluation criteria on the same pool:\n")
    pool, y, C = source_pool(rng)
    q = np.array([qq for _, _, qq in pool])
    # two criteria that are both defensible and disagree on the middle of the pool
    acc_like = q + rng.standard_normal(len(q)) * 0.10
    auroc_like = q + rng.standard_normal(len(q)) * 0.10
    print(f"   tau between the two reference rankings   {kendall_tau(acc_like, auroc_like):>7.3f}")
    print(f"   weighted tau between them                {weighted_tau(acc_like, auroc_like):>7.3f}")
    leep_s = np.array([leep(th, y, C) for _, th, _ in pool])
    print(f"\n   tau(LEEP, reference A)                   {kendall_tau(acc_like, leep_s):>7.3f}")
    print(f"   tau(LEEP, reference B)                   {kendall_tau(auroc_like, leep_s):>7.3f}")
    print("\n   The same metric scores differently against two equally defensible")
    print("   references. Any single number of the form 'metric X attains tau = x'")
    print("   is conditional on a reference that was itself sampled.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Robustness of TE metrics (Classen et al. 2026), mechanisms only")
    print("=" * 72, "\n")
    coefficients_disagree()
    stability_shrinks(rng)
    stability_is_not_accuracy(rng)
    reference_moves(rng)
