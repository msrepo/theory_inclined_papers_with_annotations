#!/usr/bin/env python3
"""LEEP: its two limits, the bound that is definitional, and the one that is not.

LEEP evaluates a single hand-built classifier -- the Expected Empirical
Predictor -- on the target data. Everything interesting about it follows from
that, and most of it is checkable without touching a real network: LEEP only
ever sees the source model's softmax output, so a synthetic softmax is enough.

Checked here:

  * Property 1, LEEP <= the best achievable log-likelihood. True, and nearly
    definitional -- but the decomposition below shows where the slack actually
    goes, which the property itself does not.
  * the two exact limits: an uninformative source scores exactly -H(Y), and a
    perfectly confident source collapses LEEP onto NCE.
  * Property 2, LEEP >= NCE + mean log confidence. It holds everywhere tested,
    but the obvious derivation does NOT reach it -- see below.
  * temperature: a transformation that changes no decision the source model
    makes moves LEEP substantially, while NCE and H-score do not budge.

Standard library and numpy only.

Run:  python3 leep.py
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- the measures
def softmax(L, T=1.0):
    L = L / T
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def leep(theta, y, C):
    """T(theta, D) = (1/n) sum_i log sum_z Phat(y_i|z) theta(x_i)_z."""
    n = len(y)
    P = np.stack([theta[y == c].sum(0) for c in range(C)]) / n   # soft joint Phat(y,z)
    cond = P / np.maximum(P.sum(0), 1e-300)                      # Phat(y|z)
    pred = theta @ cond.T                                        # the EEP, a LINEAR map
    return float(np.mean(np.log(np.maximum(pred[np.arange(n), y], 1e-300))))


def nce(theta, y, C):
    """Tran et al.'s negative conditional entropy, on the HARD dummy labels."""
    z = theta.argmax(1)
    J = np.zeros((C, theta.shape[1]))
    np.add.at(J, (y, z), 1.0)
    J /= len(y)
    Pz = J.sum(0)
    nz = J > 0
    return float((J[nz] * np.log(J[nz] / np.broadcast_to(Pz, J.shape)[nz])).sum())


def hscore(Z, y, C):
    """Bao et al., for contrast: tr(S_T^-1 S_B) on whatever representation."""
    Z = Z - Z.mean(0)
    S_T = Z.T @ Z / len(Z)
    S_B = np.zeros_like(S_T)
    for c in range(C):
        m = y == c
        if m.sum():
            S_B += m.mean() * np.outer(Z[m].mean(0), Z[m].mean(0))
    return float(np.trace(np.linalg.solve(S_T, S_B)))


def best_loglik(feat, y, C, steps=4000, lr=0.3):
    """max over linear heads of the average log-likelihood: Property 1's RHS.

    Whitened first -- a linear head absorbs any invertible map, so this leaves
    the optimum untouched while keeping fixed-step descent well conditioned.
    """
    X = feat - feat.mean(0)
    S = X.T @ X / len(X) + 1e-9 * np.eye(X.shape[1])
    X = X @ np.linalg.inv(np.linalg.cholesky(S)).T
    X = np.hstack([X, np.ones((len(X), 1))])
    W = np.zeros((C, X.shape[1]))
    for _ in range(steps):
        p = softmax(X @ W.T)
        W -= lr * ((p - np.eye(C)[y]).T @ X / len(X))
    p = softmax(X @ W.T)
    return float(np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-300))))


def task(rng, n=4000, k=16, S=12, C=5, scale=1.2, T=1.0):
    """A synthetic source model and a target task sharing some structure."""
    y = rng.integers(0, C, n)
    H = rng.standard_normal((C, k))[y] * 0.9 + rng.standard_normal((n, k))
    logits = H @ (rng.standard_normal((S, k)) * scale).T
    return H, logits, softmax(logits, T), y, C


# ---------------------------------------------------------------------- tests
def property_one(rng):
    print("1. Property 1: LEEP lower-bounds the best achievable log-likelihood\n")
    H, _, theta, y, C = task(rng)
    L = leep(theta, y, C)
    on_theta, on_feat = best_loglik(theta, y, C), best_loglik(H, y, C)
    print(f"   LEEP                                  {L:+.5f}")
    print(f"   best linear head on theta(x)          {on_theta:+.5f}")
    print(f"   best linear head on features h(x)     {on_feat:+.5f}\n")
    print(f"   cost of the hand-built head           {on_theta - L:.5f}")
    print(f"   cost of the softmax bottleneck        {on_feat - on_theta:.5f}")
    print("   -> Property 1 holds, but it is nearly definitional. What it does")
    print("      not say is where the slack goes, and the larger share is not")
    print("      the head -- it is routing through the source classifier at all.\n")


def limits(rng):
    print("2. The two exact limits, which fix what LEEP is measured against\n")
    _, logits, _, y, C = task(rng)
    n, S = len(y), logits.shape[1]

    flat = np.full((n, S), 1.0 / S)
    negH = float(np.mean(np.log(np.bincount(y, minlength=C)[y] / n)))
    print(f"   uninformative source (uniform theta)  LEEP = {leep(flat, y, C):+.6f}")
    print(f"   negative empirical entropy  -H(Y)          = {negH:+.6f}")
    print("   -> the floor. So LEEP + H(Y) is the quantity that means something,")
    print("      and raw LEEP is NOT comparable across targets with different |Y|.\n")

    onehot = np.eye(S)[logits.argmax(1)]
    print(f"   perfectly confident source (one-hot)  LEEP = {leep(onehot, y, C):+.6f}")
    print(f"   NCE on the same hard labels                = {nce(onehot, y, C):+.6f}")
    print("   -> LEEP collapses onto NCE. The entire delta between the two")
    print("      methods is the softmax uncertainty that NCE argmaxes away.\n")


def property_two(rng):
    print("3. Property 2: LEEP >= NCE(Y|Z) + (1/n) sum_i log theta(x_i)_{z_i}\n")
    print(f"   {'setting':>20}  {'LEEP':>9}  {'NCE':>9}  {'conf':>9}  {'RHS':>9}  {'holds':>6}")
    cases = {"well matched": {}, "many source labels": dict(S=60),
             "many target labels": dict(C=30), "weak source": dict(scale=0.25),
             "hot softmax": dict(T=8.0), "cold softmax": dict(T=0.2)}
    for tag, kw in cases.items():
        _, _, theta, y, C = task(rng, **kw)
        z = theta.argmax(1)
        L, N = leep(theta, y, C), nce(theta, y, C)
        conf = float(np.mean(np.log(np.maximum(theta[np.arange(len(y)), z], 1e-300))))
        print(f"   {tag:>20}  {L:>9.4f}  {N:>9.4f}  {conf:>9.4f}  {N+conf:>9.4f}"
              f"  {'yes' if L >= N + conf - 1e-12 else 'NO':>6}")

    print("\n   It holds everywhere tested. But the obvious derivation does not")
    print("   reach it. Dropping all but the argmax term inside the log gives")
    print("       LEEP >= (1/n) sum log Phat(y_i|z_i) + conf,")
    print("   with Phat the SOFT conditional -- and that first term is strictly")
    print("   BELOW NCE, because NCE uses the hard conditional, which is the")
    print("   maximum-likelihood conditional for hard assignments:\n")
    print(f"   {'setting':>20}  {'(1/n)S log Phat':>16}  {'NCE':>10}  {'soft < hard?':>13}")
    for tag, kw in (("well matched", {}), ("weak source", dict(scale=0.25)),
                    ("hot softmax", dict(T=8.0))):
        _, _, theta, y, C = task(rng, **kw)
        z = theta.argmax(1)
        n = len(y)
        P = np.stack([theta[y == c].sum(0) for c in range(C)]) / n
        cond = P / np.maximum(P.sum(0), 1e-300)
        soft = float(np.mean(np.log(np.maximum(cond[y, z], 1e-300))))
        N = nce(theta, y, C)
        print(f"   {tag:>20}  {soft:>16.4f}  {N:>10.4f}  {'yes' if soft < N else 'NO':>13}")
    print("\n   So the supplement must argue differently. It is not in the PDF.\n")


def temperature(rng):
    print("4. Temperature: LEEP reads calibration, not only discriminability\n")
    H, logits, _, y, C = task(rng)
    print(f"   {'T':>6}  {'LEEP':>10}  {'NCE':>10}  {'H on theta':>11}  {'H on h(x)':>10}")
    for T in (0.25, 0.5, 1.0, 2.0, 8.0):
        th = softmax(logits, T)
        print(f"   {T:>6.2f}  {leep(th, y, C):>10.4f}  {nce(th, y, C):>10.4f}"
              f"  {hscore(th, y, C):>11.4f}  {hscore(H, y, C):>10.4f}")
    print("\n   Temperature changes no decision the source model makes: same argmax,")
    print("   same ranking, same discriminative content. NCE is pinned because it")
    print("   is argmax-based; H-score on the raw features is pinned because it is")
    print("   invariant to any invertible linear map. LEEP is not, and the swing is")
    print("   comparable to the gaps it is asked to resolve between source models.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("LEEP (Nguyen et al. 2020), on a synthetic source softmax")
    print("=" * 70, "\n")
    property_one(rng)
    limits(rng)
    property_two(rng)
    temperature(rng)
