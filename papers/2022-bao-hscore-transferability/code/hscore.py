#!/usr/bin/env python3
"""H-score: the exact identities, and the limit of the approximate one.

Bao et al. chain four claims together. Three of them are exact linear algebra
and hold to machine precision; the fourth is a second-order approximation and
is the one worth probing.

  Eq 2   given features, the classifier is an OLS solve
  Eq 3   ||Btilde||^2 splits into (captured by features) + (residual loss)
  Eq 4   the captured piece equals tr(cov(f)^-1 cov(E[f|Y])), no Btilde needed
  Eq 1   ... and minimising log-loss is that same low-rank problem, to o(eps^2)

The first three are checked here to machine precision. The fourth carries a
local assumption -- X and Y weakly dependent -- and the last test asks whether
that assumption actually binds in practice. It does not, for ranking.

Also checked: H-score is exactly the Fisher discriminant ratio tr(S_T^-1 S_B),
it is invariant to invertible linear reparameterisation of the features, and
duplicating a feature does not inflate it.

Everything runs on a small discrete joint P_XY, where Btilde can actually be
built -- which on real data it cannot. That is the point of Eq 4.

Run:  python3 hscore.py
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- the objects
def make_joint(nx, ny, eps, rng):
    """P_XY = P_X P_Y (1 + eps * noise). eps -> 0 is the local regime."""
    px = rng.random(nx) + 0.5
    px /= px.sum()
    py = rng.random(ny) + 0.5
    py /= py.sum()
    pert = rng.standard_normal((ny, nx))
    pert -= (pert * px).sum(axis=1, keepdims=True)          # keep P_Y marginal
    joint = np.outer(py, px) * (1.0 + eps * pert)
    joint = np.clip(joint, 1e-12, None)
    return joint / joint.sum()


def dtm(joint):
    """Definition 1: Btilde[y,x] = P(x,y)/(sqrt(Px) sqrt(Py)) - sqrt(Py) sqrt(Px)."""
    px, py = joint.sum(axis=0), joint.sum(axis=1)
    return joint / np.sqrt(np.outer(py, px)) - np.sqrt(np.outer(py, px))


def center(F, px):
    """Subtract E[f(X)], as Eq 4 assumes."""
    return F - (px[:, None] * F).sum(axis=0)


def phi(F, px):
    """Phi = [sqrt(Px)] F -- the feature matrix in probability coordinates."""
    return np.sqrt(px)[:, None] * F


def inv_sqrt(M):
    w, V = np.linalg.eigh(M)
    return V @ np.diag(w ** -0.5) @ V.T


# ------------------------------------------------------- the two forms of H
def h_projection(F, joint):
    """||Btilde Phi (Phi^T Phi)^{-1/2}||_F^2 -- the Eq 3 form, needs Btilde."""
    px = joint.sum(axis=0)
    P = phi(center(F, px), px)
    return float(np.linalg.norm(dtm(joint) @ P @ inv_sqrt(P.T @ P)) ** 2)


def h_score(F, joint, pinv=False):
    """Eq 4: tr(cov(f(X))^-1 cov(E[f(X)|Y])) -- no Btilde anywhere.

    pinv=True uses the pseudo-inverse, which is what you need when the
    features are linearly dependent and cov(f(X)) is singular.
    """
    px, py = joint.sum(axis=0), joint.sum(axis=1)
    Fc = center(F, px)
    cov_f = Fc.T @ (px[:, None] * Fc)                       # E[f f^T]
    cond = (joint / py[:, None]) @ Fc                       # rows: E[f|Y=y]
    cov_c = cond.T @ (py[:, None] * cond)
    M = np.linalg.pinv(cov_f) @ cov_c if pinv else np.linalg.solve(cov_f, cov_c)
    return float(np.trace(M))


def h_sample(Z, y, ny):
    """The same thing from samples: tr(S_T^-1 S_B), the Fisher ratio."""
    Z = Z - Z.mean(axis=0)
    S_T = Z.T @ Z / len(Z)
    S_B = np.zeros_like(S_T)
    for c in range(ny):
        m = y == c
        if m.sum():
            mu = Z[m].mean(axis=0)
            S_B += (m.mean()) * np.outer(mu, mu)
    return float(np.trace(np.linalg.solve(S_T, S_B)))


def optimal_logloss(F, joint, steps=3000, lr=0.5):
    """min_theta E[-log softmax(theta_y . f(x))], exactly (no sampling)."""
    px = joint.sum(axis=0)
    Fc = center(F, px)
    ny, k = joint.shape[0], Fc.shape[1]
    th = np.zeros((ny, k))
    for _ in range(steps):
        logits = Fc @ th.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        grad = (p * px[:, None] - joint.T).T @ Fc           # d/dtheta of the loss
        th -= lr * grad
    logits = Fc @ th.T
    logits -= logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(logits).sum(axis=1))
    return float(-(joint.T * (logits - lse[:, None])).sum())


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


# ---------------------------------------------------------------------- tests
def exact_identities(rng):
    print("1. The exact identities (no local assumption needed)\n")
    nx, ny, k = 60, 5, 3
    joint = make_joint(nx, ny, 0.3, rng)
    px = joint.sum(axis=0)
    F = rng.standard_normal((nx, k))
    B, P = dtm(joint), phi(center(F, px), px)

    # Eq 2 as the normal equations, and Eq 3 as Pythagoras
    Psi = B @ P @ np.linalg.inv(P.T @ P)
    normal_eq = np.abs(B @ P - Psi @ (P.T @ P)).max()
    captured = np.linalg.norm(B @ P @ inv_sqrt(P.T @ P)) ** 2
    residual = np.linalg.norm(B - Psi @ P.T) ** 2
    total = np.linalg.norm(B) ** 2

    print(f"   Eq 2 residual of  B~Phi = Psi Phi^T Phi        {normal_eq:.2e}")
    print(f"   Eq 3  |B~|^2 - (captured + residual)           "
          f"{abs(total - captured - residual):.2e}")
    print(f"   Eq 4  h_score - h_projection                   "
          f"{abs(h_score(F, joint) - h_projection(F, joint)):.2e}")

    # chi-square reading of ||B~||^2
    chi2 = float(((joint - np.outer(joint.sum(1), px)) ** 2
                  / np.outer(joint.sum(1), px)).sum())
    print(f"   |B~|^2 - chi2(P_XY || P_X P_Y)                 {abs(total - chi2):.2e}")

    # Eckart-Young: the optimal rank-k feature IS the top-k SVD of B~.
    # Phi's columns should span the top-k right singular vectors, so take
    # F = [sqrt(Px)]^-1 V_k and check H hits the bound exactly.
    _, sv, Vt = np.linalg.svd(B)
    F_opt = Vt[:k].T / np.sqrt(px)[:, None]
    best_rand = max(h_score(rng.standard_normal((nx, k)), joint) for _ in range(300))
    print(f"\n   H of the top-{k} SVD feature                    {h_score(F_opt, joint):.6f}")
    print(f"   sum of top-{k} squared singular values of B~    {(sv[:k]**2).sum():.6f}")
    print(f"   best of 300 RANDOM rank-{k} features            {best_rand:.6f}")
    print(f"   |B~|^2, the bound once k >= rank(B~)           {total:.6f}")
    print("   -> Eq 3 is Eckart-Young: the optimum is the truncated SVD,")
    print("      and random subspaces land nowhere near it.\n")


def invariance_and_redundancy(rng):
    print("2. H is scale-free; and what 'redundancy is penalised' really means\n")
    nx, ny, k = 60, 5, 3
    joint = make_joint(nx, ny, 0.3, rng)
    px, py = joint.sum(axis=0), joint.sum(axis=1)
    F = rng.standard_normal((nx, k))

    A = rng.standard_normal((k, k))
    print("   Invariance to invertible linear reparameterisation:")
    print(f"     H(f)                                         {h_score(F, joint):.6f}")
    print(f"     H(A f), A random invertible                  {h_score(F @ A.T, joint):.6f}")
    print(f"     H(100 f)                                     {h_score(100 * F, joint):.6f}")
    print("   -> this is why cov(f)^-1 is there: without it you could inflate")
    print("      the score by rescaling. H measures the SUBSPACE.\n")

    # unnormalised between-class scatter, for contrast
    def scatter(M):
        Mc = center(M, px)
        cond = (joint / py[:, None]) @ Mc
        return float(np.trace(cond.T @ (py[:, None] * cond)))

    exact = np.hstack([F, F[:, :1]])                       # an exact duplicate
    print("   Adding an exactly duplicated feature:")
    print(f"     tr(between-class scatter)   {scatter(F):.6f} -> {scatter(exact):.6f}")
    print(f"     H (pseudo-inverse)          {h_score(F, joint, pinv=True):.6f}"
          f" -> {h_score(exact, joint, pinv=True):.6f}")
    print("   -> the raw scatter double-counts the duplicate; H does not move,")
    print("      because a repeated basis vector adds no new subspace.\n")

    print("   The practical hazard: a NEAR-duplicate with a plain inverse")
    print(f"   {'noise scale':>13}  {'cond(cov f)':>13}  {'H (solve)':>11}  {'H (pinv)':>10}")
    for tol in (1e-2, 1e-4, 1e-7):
        near = np.hstack([F, F[:, :1] + tol * rng.standard_normal((nx, 1))])
        c = np.linalg.cond(center(near, px).T @ (px[:, None] * center(near, px)))
        print(f"   {tol:>13.0e}  {c:>13.2e}  {h_score(near, joint):>11.6f}"
              f"  {h_score(near, joint, pinv=True):>10.6f}")
    print("   -> as cov(f) becomes ill-conditioned the plain inverse AMPLIFIES a")
    print("      direction carrying almost no signal. At k=2048 on real features")
    print("      this is a live concern, and the paper does not discuss it.\n")


def sample_estimator(rng):
    print("3. The sample estimator is the Fisher ratio, and converges\n")
    nx, ny, k = 40, 4, 3
    joint = make_joint(nx, ny, 0.4, rng)
    F = rng.standard_normal((nx, k))
    flat = joint.T.ravel()
    print(f"   population H (Eq 4)                            {h_score(F, joint):.5f}\n")
    print(f"   {'m':>10}  {'sample tr(S_T^-1 S_B)':>22}")
    for m in (500, 5_000, 50_000, 500_000):
        idx = rng.choice(nx * ny, size=m, p=flat)
        xs, ys = idx // ny, idx % ny
        print(f"   {m:>10}  {h_sample(F[xs], ys, ny):>22.5f}")
    print()


def locality(rng):
    print("4. Does the o(eps^2) in Eq 1 actually bind?\n")
    print("   Eq 1 is derived only to second order, for X and Y weakly dependent.")
    print("   Rank 40 random features by H, rank them by the log-loss a trained")
    print("   linear head actually reaches, and compare. -1 = perfect agreement.\n")
    nx, ny, k = 40, 4, 3
    print(f"   {'eps':>8}  {'dependence |B~|^2':>18}  {'Spearman(H, logloss)':>21}")
    for eps in (0.05, 0.2, 0.5, 1.0, 2.0):
        joint = make_joint(nx, ny, eps, rng)
        feats = [rng.standard_normal((nx, k)) for _ in range(40)]
        hs = [h_score(F, joint) for F in feats]
        ll = [optimal_logloss(F, joint) for F in feats]
        print(f"   {eps:>8.2f}  {np.linalg.norm(dtm(joint))**2:>18.4f}"
              f"  {spearman(hs, ll):>21.3f}")
    print("\n   The agreement does not decay as the dependence grows -- if anything")
    print("   it tightens, because at tiny eps the log-loss differences between")
    print("   features are themselves tiny and the ranking is resolution-limited.")
    print("   So the local assumption buys the DERIVATION, not the RANKING, at")
    print("   least for random features on a random joint. That is more than the")
    print("   paper claims, and less than it would need for real learned features.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("H-score (Bao et al. 2022), checked on a discrete joint")
    print("=" * 66, "\n")
    exact_identities(rng)
    invariance_and_redundancy(rng)
    sample_estimator(rng)
    locality(rng)
