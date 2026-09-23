#!/usr/bin/env python3
"""PAS: what the score is, what it cannot see, and what it depends on.

PAS scores an unlabelled target set against a labelled source set in a frozen
embedding: build source class centroids, and for each target sample report how
much closer it is to its nearest centroid than to the runner-up. It never sees a
target label, which is the whole point -- unsupervised domain adaptation has
none.

Everything below runs on synthetic embeddings, because PAS only ever consumes
unit vectors and a source partition. No network is needed to check it.

Checked here:

  * Equation 1 is the argmax of mean cosine similarity over unit vectors, which
    the paper asserts and does not derive -- one line of Cauchy-Schwarz;
  * the range [0,1] and both endpoints, plus the fact that the NULL is not zero;
  * PAS >= Oracle always, with equality exactly when the nearest centroid is the
    true class, so the gap between them measures misassignment;
  * the failure mode this forces: PAS is a measure of DECISIVENESS, not of
    correctness, and is maximal on a confidently wrong embedding;
  * PAS falls with the number of classes at FIXED embedding quality, which
    matters because their headline correlation pools four benchmarks whose
    label-space sizes differ by a factor of 29.

Standard library and numpy only.

Run:  python3 pas.py
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- the score
def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def centroids(ZS, yS, C):
    """Eq 1: mu_c = sum f(x) / ||sum f(x)||, the spherical (Dhillon-Modha) mean."""
    return np.stack([unit(ZS[yS == c].sum(0)) for c in range(C)])


def distances(ZT, mu):
    """Eq 2: cosine distance from each target sample to each source centroid."""
    return 1.0 - ZT @ mu.T


def pas(ZT, mu):
    """Eq 3. Note (d2 - d1)/d2 = 1 - d1/d2, so PAS = 1 - mean(d1/d2): one minus
    the average nearest-to-second-nearest ratio, i.e. Lowe's ratio test."""
    D = distances(ZT, mu)
    s = np.sort(D, axis=1)
    return float(np.mean((s[:, 1] - s[:, 0]) / s[:, 1])), D


def oracle(ZT, mu, yT_true):
    """Their oracle: d1 to the TRUE centroid, d2 to the nearest non-true one.
    The max{} in the denominator is back, because d1 can now exceed d2."""
    D = distances(ZT, mu)
    d1 = D[np.arange(len(ZT)), yT_true]
    Dm = D.copy()
    Dm[np.arange(len(ZT)), yT_true] = np.inf
    d2 = Dm.min(1)
    return float(np.mean((d2 - d1) / np.maximum(d1, d2)))


# ---------------------------------------------------------------------- tests
def equation_one(rng):
    print("1. Equation 1 is the argmax of mean cosine similarity (asserted, not shown)\n")
    print("   max_mu (1/n) sum_i x_i . mu  s.t. ||mu|| = 1.  By Cauchy-Schwarz")
    print("   (sum_i x_i) . mu <= ||sum_i x_i||, with equality iff mu is parallel")
    print("   to the sum -- which is exactly Eq 1.\n")
    X = unit(rng.standard_normal((200, 8)))
    star = unit(X.sum(0))
    best = max((unit(rng.standard_normal(8)) for _ in range(200_000)),
               key=lambda v: (X @ v).mean())
    print(f"   mean cosine to  sum/||sum||        {(X @ star).mean():.8f}")
    print(f"   best of 200,000 random unit vecs   {(X @ best).mean():.8f}\n")


def range_and_endpoints(rng):
    print("2. The range is [0,1] -- but the null is not 0\n")
    print("   d1 <= d2 by construction (they are the two smallest of a sorted")
    print("   list), so max{d1,d2} = d2 always and the Silhouette denominator")
    print("   collapses. That is why Eq 3 looks asymmetric: it is forced.\n")
    C, d, n = 5, 16, 3000
    mu = unit(rng.standard_normal((C, d)))
    print(f"   target sitting exactly on the centroids   PAS = "
          f"{pas(mu[rng.integers(0, C, n)], mu)[0]:.6f}")
    amb = unit(mu[0] + mu[1] + 1e-9 * rng.standard_normal((n, d)))
    print(f"   target equidistant between two           PAS = {pas(amb, mu)[0]:.6f}")
    print(f"   target unrelated to the source structure PAS = "
          f"{pas(unit(rng.standard_normal((n, d))), mu)[0]:.6f}")
    print("\n   That last row is the point: a structureless embedding does not")
    print("   score 0. PAS carries no absolute meaning, only a ranking.\n")


def against_the_oracle(rng):
    print("3. PAS >= Oracle, with equality iff the nearest centroid IS the true class\n")
    C, d, n = 5, 16, 3000
    mu = unit(rng.standard_normal((C, d)))
    print(f"   {'shift':>14}  {'PAS':>8}  {'Oracle':>9}  {'nearest-centroid acc':>21}")
    for tag, spread in (("clean", 0.15), ("moderate", 0.6), ("severe", 1.4)):
        yT = rng.integers(0, C, n)
        ZT = unit(mu[yT] + spread * rng.standard_normal((n, d)))
        p, D = pas(ZT, mu)
        print(f"   {tag:>14}  {p:>8.4f}  {oracle(ZT, mu, yT):>+9.4f}"
              f"  {(D.argmin(1) == yT).mean():>21.3f}")
    print("\n   In the clean row the two are numerically identical. So PAS - Oracle")
    print("   is a direct measure of how often the nearest-centroid assignment is")
    print("   wrong -- which is exactly the quantity PAS cannot see for itself.\n")


def the_blind_spot(rng):
    print("4. The blind spot: PAS measures decisiveness, not correctness\n")
    C, d, n = 5, 16, 3000
    mu = unit(rng.standard_normal((C, d)))
    yT = rng.integers(0, C, n)
    ZT = unit(mu[(yT + 1) % C] + 0.12 * rng.standard_normal((n, d)))  # always WRONG class
    p, D = pas(ZT, mu)
    print("   Every target sample placed tightly around a wrong-class centroid:\n")
    print(f"     PAS                            {p:>8.4f}   (high)")
    print(f"     true nearest-centroid accuracy {(D.argmin(1) == yT).mean():>8.4f}"
          f"   (chance = {1/C:.2f})")
    print(f"     Oracle                         {oracle(ZT, mu, yT):>+8.4f}   (sees it)")
    print("\n   Silhouette can go negative -- that is how it says 'this point is in")
    print("   the wrong cluster'. By defining the nearest cluster to BE the true")
    print("   one, PAS gave that up along with the labels it did not have. This is")
    print("   the paper's own ImageCLEF failure: a sample close to the centroid of")
    print("   an object that IS in the image, but is not the labelled one.\n")


def class_count_confound(rng):
    print("5. PAS depends on the number of classes, at fixed embedding quality\n")
    d, n = 64, 2000
    print(f"   {'C':>6}  {'PAS':>9}  {'PAS, random emb':>17}  {'nearest-centroid acc':>21}")
    for C in (12, 31, 65, 345):
        mu = unit(rng.standard_normal((C, d)))
        yT = rng.integers(0, C, n)
        ZT = unit(mu[yT] + 0.5 * rng.standard_normal((n, d)))   # SAME noise every row
        p, D = pas(ZT, mu)
        pr, _ = pas(unit(rng.standard_normal((n, d))), mu)
        print(f"   {C:>6}  {p:>9.4f}  {pr:>17.4f}  {(D.argmin(1) == yT).mean():>21.3f}")
    print("\n   Embedding quality is identical in every row; only the label-space")
    print("   size changes. PAS falls anyway, because more clusters crowd the")
    print("   nearest and second-nearest together and d1/d2 -> 1.\n")
    print("   Their benchmarks have C = 12 (ImageCLEF), 31 (Office-31), 65")
    print("   (Office-Home) and 345 (DomainNet), and accuracy falls with C too.")
    print("   Pooling them mixes a real signal with this artefact -- which is the")
    print("   likely reason their Total Pearson of 0.83 exceeds every one of the")
    print("   per-benchmark values (0.76, 0.63, 0.44, 0.53) that make it up.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("PAS (Diniz et al. 2026), on synthetic embeddings")
    print("=" * 70, "\n")
    equation_one(rng)
    range_and_endpoints(rng)
    against_the_oracle(rng)
    the_blind_spot(rng)
    class_count_confound(rng)
