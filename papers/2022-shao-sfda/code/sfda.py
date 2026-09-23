#!/usr/bin/env python3
"""SFDA: the two mechanisms, and the lambda the paper states three ways.

SFDA scores a pre-trained model by projecting its frozen features into a
regularised Fisher space, classifying with LDA there, and reporting the average
log-likelihood -- then doing it a second time on features deliberately made
harder by ConfMix. Both stages are checkable without a network, because SFDA
only ever consumes features and a label partition.

Checked here:

  * ConfMix (Eq 7) is a difficulty amplifier: each point moves toward the mean of
    the other classes by (1 - p_n), its own current difficulty;
  * D' = min(D, C-1), so a BINARY target gives a ONE-dimensional Fisher space --
    which is where the benchmarks report ties and then NaN correlations;
  * the regularisation strength is specified three mutually inconsistent ways
    (Eq 3, the text after it, and Algorithm 1), and at the stated a = 4 the two
    written formulas saturate at OPPOSITE ends on realistic scatter magnitudes;
  * what lambda = 1 costs: S~_W = I exactly, so within-class scatter leaves the
    method and U becomes the eigenbasis of S_B alone.

Standard library and numpy only.

Run:  python3 sfda.py
"""
from __future__ import annotations

import numpy as np


def scatters(X, y, C):
    """S_B and S_W as defined below Eq 2."""
    mu = X.mean(0)
    SB = np.zeros((X.shape[1],) * 2)
    SW = np.zeros_like(SB)
    for c in range(C):
        Xc = X[y == c]
        mc = Xc.mean(0)
        d = (mc - mu)[:, None]
        SB += len(Xc) * (d @ d.T)
        Z = Xc - mc
        SW += Z.T @ Z
    return SB, SW


def lam_eq3(SW, a=4.0):
    """Eq 3 of the main text: lambda = exp(-a * sigma(S_W))."""
    return float(np.exp(-np.clip(a * np.linalg.eigvalsh(SW).max(), 0, 700)))


def lam_alg1(SB, a=4.0):
    """Algorithm 1, line 11: lambda = 1/(1 + exp(-a * sigma(S_B))).
    A sigmoid, of a different matrix, moving the opposite way."""
    return float(1.0 / (1.0 + np.exp(-np.clip(a * np.linalg.eigvalsh(SB).max(),
                                              -700, 700))))


def fisher_U(SB, SW, lam, k):
    """Eq 2/Eq 4: the top-k generalised eigenvectors of (S_B, S~_W)."""
    SWt = (1 - lam) * SW + lam * np.eye(len(SW))
    w, V = np.linalg.eigh(np.linalg.solve(SWt, SB))
    return V[:, np.argsort(w)[::-1][:k]]


def principal_angles(A, B):
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def ncm_accuracy(X, y, U, C):
    Z = X @ U
    mu = np.stack([Z[y == c].mean(0) for c in range(C)])
    return float((((Z[:, None, :] - mu[None]) ** 2).sum(-1).argmin(1) == y).mean())


# ---------------------------------------------------------------------- tests
def confmix_is_a_difficulty_amplifier():
    print("1. ConfMix (Eq 7) scales each perturbation by the sample's own difficulty\n")
    print("   x~_n = p_n x^_n + (1 - p_n) mu_{c != y_n}\n")
    print(f"   {'p_n (confidence)':>17}  {'keeps of its own x':>19}  {'pulled to others':>17}")
    for p in (0.99, 0.9, 0.7, 0.5, 0.3):
        print(f"   {p:>17.2f}  {p:>19.2f}  {1 - p:>17.2f}")
    print("\n   Confident points barely move; ambiguous ones are dragged across, so")
    print("   the second Reg-FDA pass must separate a deliberately harder problem.")
    print("   See figures/confmix.svg.\n")


def fisher_space_dimension():
    print("2. D' = min(D, C-1), because rank(S_B) <= C-1\n")
    print(f"   {'classes C':>10}  {'Fisher dimensions D0':>21}")
    for C in (2, 5, 10, 100):
        print(f"   {C:>10}  {min(512, C - 1):>21}")
    print("\n   A BINARY target therefore gets a ONE-dimensional Fisher space: every")
    print("   model is scored through a single direction. That is the setting in")
    print("   which both medical benchmarks report SFDA assigning every source the")
    print("   same score, and then NaN when a correlation is taken.\n")


def lambda_is_stated_three_ways(rng):
    print("3. The regularisation strength, three mutually inconsistent ways\n")
    print("   Eq 3 (main text)      lambda = exp(-a sigma(S_W))")
    print("   text after Eq 3       refers to 'sigma(S_B) in Eqn.(3)'")
    print("   Algorithm 1, line 11  lambda = 1/(1 + exp(-a sigma(S_B))),  a = 4\n")
    print("   The first decreases in its argument; the third increases, in a")
    print("   different matrix. They are not variants of one formula.\n")
    print(f"   {'setting':>20}  {'sigma(S_W)':>11}  {'sigma(S_B)':>11}"
          f"  {'Eq 3':>9}  {'Alg 1':>7}")
    for tag, D, npc, C in (("small, low-dim", 16, 500, 5),
                           ("typical CNN", 512, 200, 10),
                           ("binary medical", 512, 30, 2),
                           ("wide backbone", 2048, 50, 5)):
        y = np.repeat(np.arange(C), npc)
        X = rng.standard_normal((C, D))[y] + rng.standard_normal((C * npc, D))
        SB, SW = scatters(X, y, C)
        print(f"   {tag:>20}  {np.linalg.eigvalsh(SW).max():>11.1f}"
              f"  {np.linalg.eigvalsh(SB).max():>11.1f}"
              f"  {lam_eq3(SW):>9.1e}  {lam_alg1(SB):>7.4f}")
    print("\n   At a = 4 they saturate at opposite ends. Eq 3 gives 0, so S~_W = S_W")
    print("   and a singular S_W stays singular. Algorithm 1 gives 1, so S~_W = I.")
    print("   See figures/lambda.svg.\n")


def what_lambda_one_costs(rng):
    print("4. lambda = 1 removes the within-class scatter from the method\n")
    print("   S~_W = (1-1)S_W + 1*I = I, so Eq 2 becomes |U^T S_B U| / |U^T U| and U")
    print("   is just the eigenbasis of S_B. Whether that matters depends on how")
    print("   anisotropic the within-class scatter is:\n")
    C, D, npc, k = 5, 64, 300, 4
    print(f"   {'anisotropy':>11}  {'cos angles, lam=1 vs Reg-FDA':>30}"
          f"  {'NCM acc Reg-FDA':>16}  {'NCM acc lam=1':>14}")
    for aniso in (1.0, 5.0, 25.0, 100.0):
        y = np.repeat(np.arange(C), npc)
        scale = np.linspace(1.0, aniso, D)
        X = rng.standard_normal((C, D))[y] + rng.standard_normal((C * npc, D)) * scale
        SB, SW = scatters(X, y, C)
        Ut, U1 = fisher_U(SB, SW, 0.05, k), fisher_U(SB, SW, 1.0, k)
        print(f"   {aniso:>11.0f}  {str(np.round(principal_angles(U1, Ut), 3)):>30}"
              f"  {ncm_accuracy(X, y, Ut, C):>16.3f}"
              f"  {ncm_accuracy(X, y, U1, C):>14.3f}")
    # lambda=1 is literally S_B's eigenbasis
    y = np.repeat(np.arange(C), npc)
    X = rng.standard_normal((C, D))[y] + rng.standard_normal((C * npc, D))
    SB, SW = scatters(X, y, C)
    cos = principal_angles(fisher_U(SB, SW, 1.0, k),
                           np.linalg.eigh(SB)[1][:, ::-1][:, :k])
    print(f"\n   lambda=1 subspace vs S_B's eigenbasis: cos = {np.round(cos, 6)}")
    print("\n   Identical, as the algebra says. With isotropic noise lambda hardly")
    print("   matters; as the within-class scatter becomes anisotropic -- the case")
    print("   Reg-FDA exists to handle -- the subspaces diverge and accuracy in the")
    print("   lambda=1 space falls away.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("SFDA (Shao et al. 2022)")
    print("=" * 70, "\n")
    confmix_is_a_difficulty_amplifier()
    fisher_space_dimension()
    lambda_is_stated_three_ways(rng)
    what_lambda_one_costs(rng)
