#!/usr/bin/env python3
"""Principal angles between subspaces: the identities, and the accuracy trap.

Two subspaces of R^n do not meet at a single angle. They meet at min(p,q) of
them, and almost everything one wants to say about "how close are these two
subspaces" is a function of that list.

Checked here:

  * the SVD characterisation (Bjorck-Golub): singular values of Q_U^T Q_V are
    the cosines, and they do not depend on which orthonormal bases you picked;
  * the projector identities, tr(P_U P_V) = sum cos^2 and
    ||P_U - P_V||_2 = sin(theta_max);
  * the degenerate cases -- nested subspaces give all zeros, orthogonal ones
    all right angles;
  * canonical correlations from CCA are exactly these cosines;
  * and the trap: arccos of a cosine loses half your digits on small angles.
    The sine route recovers them.

Standard library and numpy only.

Run:  python3 angles.py
"""
from __future__ import annotations

import numpy as np


def orth(A):
    """An orthonormal basis for the column space of A."""
    Q, _ = np.linalg.qr(A)
    return Q


def cosines(QU, QV):
    """Cosines of the principal angles, largest first (smallest angle first)."""
    return np.clip(np.linalg.svd(QU.T @ QV, compute_uv=False), 0.0, 1.0)


def angles_via_cos(QU, QV):
    return np.arccos(cosines(QU, QV))


def angles_via_sin(QU, QV):
    """Bjorck-Golub: singular values of (I - P_U) Q_V are the SINES.

    Reversed, because descending sines correspond to descending angles while
    descending cosines correspond to ascending ones.
    """
    R = QV - QU @ (QU.T @ QV)
    s = np.clip(np.linalg.svd(R, compute_uv=False), 0.0, 1.0)
    # There are only min(p, q) principal angles. When dim V > dim U the extra
    # singular values are 1 (the part of V orthogonal to all of U) and are not
    # principal angles, so drop them after sorting ascending.
    return np.arcsin(s)[::-1][:min(QU.shape[1], QV.shape[1])]


def angles_stable(QU, QV):
    """Cosines for the wide angles, sines for the narrow ones."""
    tc, ts = angles_via_cos(QU, QV), angles_via_sin(QU, QV)
    return np.where(tc > np.pi / 4, tc, ts)


# ---------------------------------------------------------------------- tests
def svd_characterisation(rng):
    print("1. The cosines are a property of the SUBSPACES, not of the bases\n")
    n, p, q = 40, 5, 4
    U, V = rng.standard_normal((n, p)), rng.standard_normal((n, q))
    QU, QV = orth(U), orth(V)
    c = cosines(QU, QV)
    print(f"   cos(theta) = {np.round(c, 6)}")

    # re-span each subspace with a completely different basis
    QU2 = orth(U @ rng.standard_normal((p, p)))
    QV2 = orth(V @ rng.standard_normal((q, q)))
    print(f"   after re-basing both        max |change| "
          f"{np.abs(np.sort(cosines(QU2, QV2)) - np.sort(c)).max():.2e}")

    # the largest cosine is the largest achievable inner product
    best = 0.0
    for _ in range(20000):
        a, b = rng.standard_normal(p), rng.standard_normal(q)
        u, v = QU @ a, QV @ b
        best = max(best, abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    print(f"   cos(theta_1)                {c[0]:.6f}")
    print(f"   best of 20000 random pairs  {best:.6f}   (must not exceed it)\n")

    print("   The cosine and sine routes agree, at every shape:")
    for nn, pp, qq in ((40, 5, 5), (40, 6, 3), (40, 3, 7), (60, 2, 9)):
        a, b = orth(rng.standard_normal((nn, pp))), orth(rng.standard_normal((nn, qq)))
        tc, ts = angles_via_cos(a, b), angles_via_sin(a, b)
        print(f"     p={pp}, q={qq}  ->  min(p,q)={min(pp,qq)} angles,"
              f"  max |cos route - sin route| = {np.abs(tc - ts).max():.1e}")
    print()


def projector_identities(rng):
    print("2. Everything you want to say about two subspaces is a function of them\n")
    n, p = 50, 6
    QU, QV = orth(rng.standard_normal((n, p))), orth(rng.standard_normal((n, p)))
    c = cosines(QU, QV)
    s2 = 1 - c ** 2
    PU, PV = QU @ QU.T, QV @ QV.T

    print(f"   tr(P_U P_V)            {np.trace(PU @ PV):.10f}")
    print(f"   sum cos^2(theta)       {(c ** 2).sum():.10f}")
    print(f"   ||Q_U^T Q_V||_F^2      {np.linalg.norm(QU.T @ QV) ** 2:.10f}\n")

    print(f"   ||P_U - P_V||_2        {np.linalg.norm(PU - PV, 2):.10f}")
    print(f"   sin(theta_max)         {np.sqrt(s2.max()):.10f}\n")

    print(f"   ||P_U - P_V||_F        {np.linalg.norm(PU - PV):.10f}")
    print(f"   sqrt(2 * sum sin^2)    {np.sqrt(2 * s2.sum()):.10f}")
    print("   -> the chordal distance; note it needs dim U = dim V.\n")


def degenerate(rng):
    print("3. The two extremes\n")
    n, p = 30, 4
    W = orth(rng.standard_normal((n, 8)))
    nested_u, nested_v = W[:, :p], W[:, :6]          # U is contained in V
    print(f"   U contained in V,  via arccos  {np.round(angles_via_cos(nested_u, nested_v), 12)}")
    print(f"   U contained in V,  via arcsin  {np.round(angles_via_sin(nested_u, nested_v), 12)}")
    print("   -> even here arccos cannot return a clean zero. Foreshadows test 4.")
    print(f"\n   orthogonal subspaces, angles   "
          f"{np.round(angles_via_cos(W[:, :3], W[:, 5:8]), 12)}   (pi/2 = {np.pi/2:.6f})\n")


def accuracy_trap(rng):
    print("4. The trap: arccos of a cosine throws away half your digits\n")
    print("   Rotate a subspace by a known tiny angle and try to recover it.\n")
    n, p = 60, 3
    W = orth(rng.standard_normal((n, 2 * p)))
    A, Bdir = W[:, :p], W[:, p:2 * p]                 # orthogonal complements
    print(f"   {'true theta':>12}  {'via arccos':>14}  {'rel err':>9}"
          f"  {'via arcsin':>14}  {'rel err':>9}")
    for t in (1e-2, 1e-4, 1e-6, 1e-8):
        QV = orth(A * np.cos(t) + Bdir * np.sin(t))   # each direction tilted by t
        tc = angles_via_cos(A, QV).max()
        ts = angles_via_sin(A, QV).max()
        print(f"   {t:>12.0e}  {tc:>14.3e}  {abs(tc-t)/t:>9.1e}"
              f"  {ts:>14.3e}  {abs(ts-t)/t:>9.1e}")
    print("\n   cos(t) = 1 - t^2/2, so for t = 1e-8 the cosine differs from 1 by")
    print("   5e-17 -- below double precision. The angle is unrecoverable from it.")
    print("   The sine is 1e-8 and carries full relative precision. Use cosines")
    print("   above pi/4 and sines below; that is the Bjorck-Golub recipe.\n")


def canonical_correlations(rng):
    print("5. CCA's canonical correlations ARE these cosines\n")
    m, p, q = 4000, 4, 3
    X = rng.standard_normal((m, p))
    Y = X[:, :q] @ rng.standard_normal((q, q)) * 0.8 + rng.standard_normal((m, q))
    X, Y = X - X.mean(0), Y - Y.mean(0)

    # textbook CCA: whiten each block, then SVD the cross-covariance
    Sxx, Syy = X.T @ X / m, Y.T @ Y / m
    Sxy = X.T @ Y / m
    iw = lambda M: np.linalg.inv(np.linalg.cholesky(M))
    rho = np.linalg.svd(iw(Sxx) @ Sxy @ iw(Syy).T, compute_uv=False)

    print(f"   canonical correlations   {np.round(rho, 8)}")
    print(f"   cos(principal angles)    {np.round(cosines(orth(X), orth(Y)), 8)}")
    print("   -> CCA is principal angles between the column spans of the data.\n")


def hscore_instance(rng):
    print("6. The instance this page was written for: H-score\n")
    m, k, C = 5000, 10, 5
    y = rng.integers(0, C, m)
    Z = rng.standard_normal((C, k))[y] * 0.6 + rng.standard_normal((m, k))
    Z = Z - Z.mean(0)
    cnt = np.bincount(y, minlength=C)
    G = np.zeros((m, C)); G[np.arange(m), y] = 1.0
    G = G / np.sqrt(cnt)                              # orthonormal columns

    S_T = Z.T @ Z / m
    mu = np.stack([Z[y == c].mean(0) for c in range(C)])
    S_B = (mu * (cnt / m)[:, None]).T @ mu
    H = np.trace(np.linalg.solve(S_T, S_B))

    c = cosines(orth(Z), G)
    print(f"   tr(S_T^-1 S_B)                {H:.10f}")
    print(f"   sum cos^2(principal angle)    {(c ** 2).sum():.10f}")
    print(f"   cosines                       {np.round(c, 4)}")
    print("   -> one angle is exactly 90 degrees: Z is centred so the all-ones")
    print("      vector is orthogonal to span(Z), yet it lies in span(G). That is")
    print("      why rank(S_B) = C-1 and H < min(k, C-1).\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Principal angles between subspaces")
    print("=" * 68, "\n")
    svd_characterisation(rng)
    projector_identities(rng)
    degenerate(rng)
    accuracy_trap(rng)
    canonical_correlations(rng)
    hscore_instance(rng)
