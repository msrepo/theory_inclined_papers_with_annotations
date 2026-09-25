#!/usr/bin/env python3
"""Column space, null space, residual space: every number quoted in notes.md.

A matrix A (m x n) carves its input space R^n and its output space R^m into
four subspaces. Checked here, in the order the notes use them:

  1. the running example -- fitting a line through three points -- where the
     least-squares residual comes out as a multiple of (1, -2, 1), the one
     direction the columns of A cannot reach;
  2. the four subspaces read off one SVD, their dimensions (rank-nullity) and
     the two orthogonality relations;
  3. the projector onto the column space and the one onto the residual space:
     symmetric, idempotent, traces p and m - p;
  4. "residual degrees of freedom": E||r||^2 = sigma^2 (m - p), measured;
  5. a wide system: gradient descent from zero lands on the pseudo-inverse
     (minimum-norm) solution, and from anywhere else keeps its null-space
     component unchanged -- the NTK "Delta^0 never moves" statement in miniature;
  6. more features than samples: the column space is all of R^m, so the
     least-squares residual is zero whatever the features are (the LogME table);
  7. centring puts the all-ones vector into the left null space, which is why
     H-score always has one principal angle of exactly 90 degrees;
  8. rank caps on scatter matrices: rank(S_b) <= g - 1, and S_w singular
     once N < n + g (LDA, SFDA);
  9. the null space of a graph Laplacian counts connected components
     (HaoChen et al.'s "multiplicity of eigenvalue 1");
 10. PCA residual subspace as an OOD score: two test points at the same
     distance from the mean, very different residuals;
 11. numerical rank: a matrix that is rank 2 on paper and rank 3 to the
     computer, and how the tolerance decides.

Standard library and numpy only.

Run:  python3 subspaces.py
"""
from __future__ import annotations

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def fmt(v):
    return "[" + ", ".join(f"{x:.4f}" for x in np.ravel(v)) + "]"


def four_subspaces(A, tol=None):
    """Orthonormal bases for col(A), N(A^T), row(A), N(A), from one SVD."""
    U, s, Vt = np.linalg.svd(A)
    if tol is None:
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    r = int(np.sum(s > tol))
    return dict(col=U[:, :r], left_null=U[:, r:], row=Vt[:r].T, null=Vt[r:].T, rank=r, s=s)


def section(title):
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------- 1
def running_example():
    section("1. Fitting a line through (0,1), (1,3), (2,2)")
    t = np.array([0.0, 1.0, 2.0])
    A = np.column_stack([np.ones(3), t])         # columns: intercept, slope
    b = np.array([1.0, 3.0, 2.0])
    x_hat = np.linalg.solve(A.T @ A, A.T @ b)     # normal equations
    p = A @ x_hat
    r = b - p
    print("A^T A          =", A.T @ A.tolist())
    print("x_hat          =", fmt(x_hat), " (intercept 3/2, slope 1/2)")
    print("p = A x_hat    =", fmt(p), " (3/2, 2, 5/2)")
    print("r = b - p      =", fmt(r), " = -(1, -2, 1)/2")
    print("A^T r          =", fmt(A.T @ r), " (the normal equations, read as orthogonality)")
    print("||b||^2        = %.4f  = ||p||^2 + ||r||^2 = %.4f + %.4f" % (b @ b, p @ p, r @ r))
    S = four_subspaces(A)
    print("left null space basis:", fmt(S["left_null"][:, 0] / S["left_null"][0, 0]),
          "(normalised so the first entry is 1)")
    assert np.allclose(r * -2, [1, -2, 1])
    assert np.allclose(A.T @ r, 0)
    return A, b


# --------------------------------------------------------------------- 2
def four_from_svd():
    section("2. Four subspaces from one SVD (a random 5 x 4 matrix of rank 2)")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 2)) @ rng.standard_normal((2, 4))
    S = four_subspaces(A)
    m, n = A.shape
    r = S["rank"]
    print("singular values:", fmt(S["s"]))
    print(f"m = {m}, n = {n}, rank r = {r}")
    print(f"dim col(A) = {S['col'].shape[1]}, dim N(A^T) = {S['left_null'].shape[1]}  "
          f"(sum = {S['col'].shape[1] + S['left_null'].shape[1]} = m)")
    print(f"dim row(A) = {S['row'].shape[1]}, dim N(A)   = {S['null'].shape[1]}  "
          f"(sum = {S['row'].shape[1] + S['null'].shape[1]} = n)")
    print("max |A  x| over null-space basis:      %.1e" % np.abs(A @ S["null"]).max())
    print("max |A^T y| over left-null basis:      %.1e" % np.abs(A.T @ S["left_null"]).max())
    print("max |row . null| (orthogonality in R^n): %.1e" % np.abs(S["row"].T @ S["null"]).max())
    print("max |col . left_null| (in R^m):          %.1e" % np.abs(S["col"].T @ S["left_null"]).max())
    assert r == 2
    assert np.abs(A @ S["null"]).max() < 1e-12


# --------------------------------------------------------------------- 3
def projectors(A, b):
    section("3. The two projectors of the running example")
    P = A @ np.linalg.solve(A.T @ A, A.T)
    M = np.eye(3) - P
    print("P (the 'hat' matrix) =\n", P)
    print("I - P (the residual maker) =\n", M, " = (1/6) * outer((1,-2,1), (1,-2,1))")
    print("(I - P) b =", fmt(M @ b), " -- the residual again, now as a projection of b")
    print("symmetric: %.1e   idempotent: %.1e   P(I-P) = 0: %.1e"
          % (np.abs(P - P.T).max(), np.abs(P @ P - P).max(), np.abs(P @ M).max()))
    print("trace P = %.4f (= p = 2)   trace (I-P) = %.4f (= m - p = 1)" % (np.trace(P), np.trace(M)))
    assert np.allclose(M, np.outer([1, -2, 1], [1, -2, 1]) / 6)


# --------------------------------------------------------------------- 4
def degrees_of_freedom():
    section("4. Residual degrees of freedom: E||r||^2 = sigma^2 (m - p)")
    rng = np.random.default_rng(1)
    m, sigma, trials = 50, 0.7, 20000
    print(f"m = {m}, sigma^2 = {sigma**2:.2f}, {trials} noise draws each")
    print(" p | mean ||r||^2 | sigma^2 (m-p) | naive ||r||^2/m | ||r||^2/(m-p)")
    for p in (1, 5, 20, 45):
        A = rng.standard_normal((m, p))
        M = np.eye(m) - A @ np.linalg.pinv(A)
        E = sigma * rng.standard_normal((trials, m))
        rss = np.sum((E @ M) ** 2, axis=1)
        print(f"{p:2d} | {rss.mean():12.3f} | {sigma**2 * (m - p):13.3f} | "
              f"{rss.mean() / m:15.3f} | {rss.mean() / (m - p):12.3f}")


# --------------------------------------------------------------------- 5
def gradient_descent_bias():
    section("5. One equation, two unknowns: x1 + 2 x2 = 4")
    a = np.array([[1.0, 2.0]])
    b = np.array([4.0])
    x_pinv = np.linalg.pinv(a) @ b
    n_hat = np.array([2.0, -1.0]) / np.sqrt(5)       # null-space direction

    def gd(x0, steps=200, lr=0.1):
        x = x0.astype(float).copy()
        for _ in range(steps):
            x -= lr * a.T @ (a @ x - b)
        return x

    x_from_0 = gd(np.zeros(2))
    x0 = np.array([3.0, 3.0])
    x_from_x0 = gd(x0)
    print("pseudo-inverse solution   =", fmt(x_pinv), " (0.8, 1.6), norm %.4f" % np.linalg.norm(x_pinv))
    print("GD from (0, 0) lands at   =", fmt(x_from_0))
    print("GD from (3, 3) lands at   =", fmt(x_from_x0), " (2, 1)")
    print("null-space coordinate of x0 = %.4f, of the landing point = %.4f"
          % (n_hat @ x0, n_hat @ x_from_x0))
    print("landing - pinv =", fmt(x_from_x0 - x_pinv), " = 0.6 * (2, -1): the null-space part of x0")
    assert np.allclose(x_from_0, x_pinv)
    assert np.allclose(x_from_x0, [2, 1])
    assert np.isclose(n_hat @ x0, n_hat @ x_from_x0)


# --------------------------------------------------------------------- 6
def more_features_than_samples():
    section("6. D > n: the column space is everything")
    rng = np.random.default_rng(2)
    n, D = 40, 60
    F_true = rng.standard_normal((n, D))
    y = F_true @ rng.standard_normal(D) / np.sqrt(D) + 0.3 * rng.standard_normal(n)
    candidates = {
        "true features": F_true,
        "half true, half noise": np.hstack([F_true[:, :D // 2], rng.standard_normal((n, D - D // 2))]),
        "pure noise": rng.standard_normal((n, D)),
    }
    print(f"n = {n} samples, D = {D} features, labels generated from the true features")
    print("features              | rank | dim N(F) | train R^2")
    for name, F in candidates.items():
        w = np.linalg.pinv(F) @ y
        r2 = 1 - np.sum((y - F @ w) ** 2) / np.sum((y - y.mean()) ** 2)
        S = four_subspaces(F)
        print(f"{name:21s} | {S['rank']:4d} | {S['null'].shape[1]:8d} | {r2:.6f}")
        assert r2 > 1 - 1e-10
    print("(rank n means col(F) = R^n: every label vector is reachable, so the fit is perfect)")


# --------------------------------------------------------------------- 7
def centring_and_ones():
    section("7. Centring puts 1 in the left null space (H-score's forced 90 degrees)")
    rng = np.random.default_rng(3)
    m, k, C = 300, 6, 4
    y = rng.integers(0, C, m)
    Z = rng.standard_normal((m, k)) + 0.8 * np.eye(C, k)[y]
    Z = Z - Z.mean(axis=0)
    ones = np.ones(m)
    print("max |Z^T 1| after centring = %.1e" % np.abs(Z.T @ ones).max())
    Q, _ = np.linalg.qr(Z)
    G = np.zeros((m, C))
    G[np.arange(m), y] = 1
    G /= np.sqrt(G.sum(axis=0))
    cos = np.linalg.svd(G.T @ Q, compute_uv=False)
    print("cosines of principal angles, span(Z) vs span(indicators):", fmt(cos))
    print("H-score = sum cos^2 = %.4f  <  min(k, C-1) = %d" % (np.sum(cos ** 2), min(k, C - 1)))
    assert abs(cos[-1]) < 1e-10


# --------------------------------------------------------------------- 8
def scatter_rank_caps():
    section("8. Rank caps on scatter matrices (LDA, SFDA)")
    rng = np.random.default_rng(4)
    print("  n  |  g  |  N  | rank S_b | rank S_w | dim N(S_w)")
    for n, g, N in ((10, 3, 300), (10, 2, 300), (50, 3, 30), (50, 5, 40)):
        y = np.arange(N) % g
        X = rng.standard_normal((N, n)) + 2.0 * rng.standard_normal((g, n))[y]
        mu = X.mean(axis=0)
        Sb = sum((y == c).sum() * np.outer(X[y == c].mean(0) - mu, X[y == c].mean(0) - mu)
                 for c in range(g))
        Sw = sum((X[y == c] - X[y == c].mean(0)).T @ (X[y == c] - X[y == c].mean(0))
                 for c in range(g))
        rb = np.linalg.matrix_rank(Sb)
        rw = np.linalg.matrix_rank(Sw)
        print(f" {n:3d} | {g:3d} | {N:3d} | {rb:8d} | {rw:8d} | {n - rw:10d}")
        assert rb == min(g - 1, n)
        assert rw == min(N - g, n)


# --------------------------------------------------------------------- 9
def laplacian_components():
    section("9. Null space of a graph Laplacian counts connected components")
    blocks = [3, 4, 2]
    N = sum(blocks)
    W = np.zeros((N, N))
    i = 0
    for size in blocks:
        W[i:i + size, i:i + size] = 1.0
        i += size
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    Dm = np.diag(1 / np.sqrt(d))
    A_bar = Dm @ W @ Dm                    # normalised adjacency
    L = np.eye(N) - A_bar                  # normalised Laplacian
    ev = np.linalg.eigvalsh(A_bar)[::-1]
    S = four_subspaces(L, tol=1e-9)
    print("blocks:", blocks)
    print("top eigenvalues of normalised adjacency:", fmt(ev[:4]))
    print("dim N(I - A_bar) =", S["null"].shape[1], "(= number of components)")
    # add one weak bridge: the third eigenvalue drops just below 1
    W2 = W.copy()
    W2[2, 3] = W2[3, 2] = 0.05
    d2 = W2.sum(1)
    A2 = np.diag(1 / np.sqrt(d2)) @ W2 @ np.diag(1 / np.sqrt(d2))
    print("with one weak bridge between blocks 1 and 2:",
          fmt(np.linalg.eigvalsh(A2)[::-1][:4]))
    assert S["null"].shape[1] == len(blocks)


# --------------------------------------------------------------------- 10
def pca_residual_score():
    section("10. PCA residual subspace as an out-of-distribution score")
    rng = np.random.default_rng(5)
    D, k, N = 20, 3, 2000
    basis, _ = np.linalg.qr(rng.standard_normal((D, k)))
    X = rng.standard_normal((N, k)) * [3.0, 2.0, 1.5] @ basis.T + 0.1 * rng.standard_normal((N, D))
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Vk = Vt[:k].T                          # principal subspace
    R = np.eye(D) - Vk @ Vk.T              # projector onto the residual subspace

    ood_dir = R @ rng.standard_normal(D)
    ood_dir /= np.linalg.norm(ood_dir)
    in_dir = Vk[:, 0]
    x_in, x_out = mu + 4.0 * in_dir, mu + 4.0 * ood_dir
    typical = np.linalg.norm((X - mu) @ R, axis=1)
    for name, x in (("along the data", x_in), ("off the data  ", x_out)):
        z = x - mu
        print(f"{name}: ||x - mu|| = {np.linalg.norm(z):.3f}   principal part = "
              f"{np.linalg.norm(Vk.T @ z):.3f}   residual part = {np.linalg.norm(R @ z):.3f}")
    print("residual norm of training points: median %.3f, 99th percentile %.3f"
          % (np.median(typical), np.percentile(typical, 99)))


# --------------------------------------------------------------------- 11
def numerical_rank():
    section("11. Numerical rank: the tolerance decides")
    rng = np.random.default_rng(6)
    A = rng.standard_normal((6, 2)) @ rng.standard_normal((2, 3))
    A_noisy = A + 1e-9 * rng.standard_normal(A.shape)
    s = np.linalg.svd(A_noisy, compute_uv=False)
    print("singular values:", "  ".join(f"{v:.3e}" for v in s))
    print("matrix_rank default tol:", np.linalg.matrix_rank(A_noisy),
          "  with tol = 1e-6:", np.linalg.matrix_rank(A_noisy, tol=1e-6))
    print("exact matrix (no noise), default tol:", np.linalg.matrix_rank(A))


if __name__ == "__main__":
    A, b = running_example()
    four_from_svd()
    projectors(A, b)
    degrees_of_freedom()
    gradient_descent_bias()
    more_features_than_samples()
    centring_and_ones()
    scatter_rank_caps()
    laplacian_components()
    pca_residual_score()
    numerical_rank()
    print("\nall checks passed")
