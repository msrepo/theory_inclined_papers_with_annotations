#!/usr/bin/env python3
"""Gram matrices: every number quoted in notes.md.

  1. the running example: three points in the plane, their Gram matrix,
     and distances read straight off it;
  2. rotation invariance: rotating the points leaves G unchanged;
  3. positive semidefinite, rank = rank(X), and the zero eigenvalue;
  4. X X^T and X^T X share their nonzero eigenvalues (the "dual" view);
  5. recovering the points from G alone (classical MDS), up to a rotation;
  6. centring: H G H is the Gram matrix of the mean-subtracted points;
  7. an RBF kernel Gram matrix and how the bandwidth moves it between
     "identity" and "all ones";
  8. squaring the condition number: cond(X^T X) = cond(X)^2.

Standard library and numpy only.

Run:  python3 gram.py
"""
from __future__ import annotations

import numpy as np

np.set_printoptions(precision=4, suppress=True)

X = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])  # rows are points x1, x2, x3


def section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    section("1. running example")
    G = X @ X.T
    print("G = X X^T =\n", G)
    d2 = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
    print("squared distances from G =\n", d2)
    print("||x1 - x3||^2 directly =", np.sum((X[0] - X[2]) ** 2))
    cos = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
    print("cosine matrix =\n", cos)

    section("2. rotation invariance")
    t = 0.7
    Q = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    print("max |G(XQ) - G(X)| =", np.abs((X @ Q) @ (X @ Q).T - G).max())

    section("3. PSD and rank")
    ev = np.linalg.eigvalsh(G)[::-1]
    print("eigenvalues of G =", ev)
    print("rank G =", np.linalg.matrix_rank(G), " rank X =", np.linalg.matrix_rank(X))
    v = np.array([2.0, -1.0, 0.5])
    print("v^T G v =", v @ G @ v, " = ||X^T v||^2 =", np.sum((X.T @ v) ** 2))

    section("4. dual view")
    C = X.T @ X
    print("X^T X =\n", C)
    print("eigenvalues of X^T X =", np.linalg.eigvalsh(C)[::-1])
    print("(7 +- sqrt 13)/2 =", (7 + np.sqrt(13)) / 2, (7 - np.sqrt(13)) / 2)

    section("5. points from G alone (MDS)")
    w, U = np.linalg.eigh(G)
    w, U = w[::-1][:2], U[:, ::-1][:, :2]
    Y = U * np.sqrt(w)
    print("recovered Y =\n", Y)
    print("max |Y Y^T - G| =", np.abs(Y @ Y.T - G).max())
    R, *_ = np.linalg.lstsq(Y, X, rcond=None)
    print("Y maps onto X by an orthogonal matrix? R^T R =\n", R.T @ R)

    section("6. centring")
    n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    Xc = X - X.mean(0)
    print("max |H G H - Xc Xc^T| =", np.abs(H @ G @ H - Xc @ Xc.T).max())
    print("row sums of H G H =", (H @ G @ H).sum(1))

    section("7. RBF Gram")
    for s in (0.3, 1.0, 5.0):
        K = np.exp(-d2 / (2 * s * s))
        print(f"sigma={s}: K =\n", K)

    section("8. condition number")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 5)) @ np.diag([1, 1, 1, 1, 1e-3])
    print("cond(A) =", f"{np.linalg.cond(A):.4g}", " cond(A^T A) =", f"{np.linalg.cond(A.T @ A):.4g}",
          " cond(A)^2 =", f"{np.linalg.cond(A) ** 2:.4g}")


if __name__ == "__main__":
    main()
