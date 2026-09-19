#!/usr/bin/env python3
"""The D-cancellation that makes the Jacobian-free NTK usable on an LLM.

Appendix E.5 of Wang et al. claims that summing the logits before
differentiating costs you a relative error of order d^{-1/2}, where d is the
hidden width -- and, crucially, that the vocabulary size D does not appear.
That is the whole reason the trick is affordable on a model with 128k logits.

The claim is a statement about how two sums accumulate:

    signal  S = sum_k     w_k^T B w_k     D terms, all positive, add coherently
    error   E = sum_{k!=m} w_k^T B w_m    D^2 terms, zero mean, random signs

D terms adding coherently give D. D^2 terms adding incoherently give
sqrt(D^2) = D. The two growth rates match exactly, D divides out, and what is
left is the per-term ratio std(X_km)/E[X_kk] = d^{-1/2}.

Everything rests on the off-diagonal terms having *random signs*, which is
Assumption E.1 (LM-head rows independent and mean-zero). The last experiment
here breaks that assumption on purpose, to show how much work it is doing.

Nothing is differentiated: B stands in for the backbone Jacobian Gram matrix
J_psi(x) J_psi(x')^T and W for the LM head, which is all the argument uses.

Run:  python3 cross_output_cancellation.py
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- the kernels
def backbone_gram(d, rank, beta, lam, rng):
    """B = beta*I_d + U Lambda U^T -- the 'bulk plus spikes' of Assumption E.2."""
    B = (beta * np.eye(d)).astype(np.float32)
    if rank > 0:
        U, _ = np.linalg.qr(rng.standard_normal((d, rank), dtype=np.float32))
        B += ((U * (lam * np.ones(rank, dtype=np.float32))) @ U.T).astype(np.float32)
    return B


def head(D, d, rng, sigma_w=1.0, shared=0.0):
    """LM head with isotropic rows (Assumption E.1), optionally correlated.

    `shared` > 0 adds one common direction to every row, which destroys the
    zero-mean property the cancellation needs.
    """
    W = rng.standard_normal((D, d), dtype=np.float32) * np.float32(sigma_w / np.sqrt(d))
    if shared:
        mu = rng.standard_normal(d, dtype=np.float32)
        mu /= np.linalg.norm(mu)
        W += np.float32(shared * sigma_w / np.sqrt(d)) * mu
    return W


def signal_and_error(W, B):
    """S = sum of diagonal terms, E = sum of off-diagonal terms.

    Both are computed without ever forming the D x D grid:
        sum over the whole grid = (1^T W) B (W^T 1)
        diagonal sum            = tr(W B W^T)
    """
    WB = W @ B
    S = float(np.sum(WB * W))            # sum_k w_k^T B w_k
    s = W.sum(axis=0)                    # W^T 1
    total = float(s @ B @ s)             # sum_{k,m} w_k^T B w_m
    return S, total - S


def n_trials(D, d, budget=4_000_000, lo=30, hi=200):
    """Draw roughly `budget` head entries per configuration, so the big-D rows
    do not dominate the runtime. The estimator's spread barely depends on D."""
    return int(min(hi, max(lo, budget // (D * d))))


def ratio(D, d, rng, rank=0, beta=1.0, lam=0.0, shared=0.0, trials=None):
    """Median |E| / S over many draws.

    |E| is itself a sum of D^2 random terms, so a single draw is a draw from a
    roughly half-normal distribution and fluctuates a lot. The median over a
    few hundred draws is what the rate is a statement about.

    B is drawn once and held fixed across the draws: Assumption E.1 conditions
    on the backbone parameters, so the randomness that matters is in W.
    """
    trials = n_trials(D, d) if trials is None else trials
    B = backbone_gram(d, rank, beta, lam, rng)
    out = []
    for _ in range(trials):
        W = head(D, d, rng, shared=shared)
        S, E = signal_and_error(W, B)
        out.append(abs(E) / abs(S))
    return float(np.median(out))


# --------------------------------------------------------------------- tests
def sweep_D(rng):
    print("1. Vocabulary size D should NOT matter (d fixed at 512)")
    print("   the D^2 error terms grow like D, the D signal terms grow like D\n")
    print(f"   {'D':>8}  {'|E|/S':>10}  {'vs d^-1/2':>10}")
    d = 512
    ref = d ** -0.5
    for D in (64, 256, 1024, 4096):
        r = ratio(D, d, rng)
        print(f"   {D:>8}  {r:>10.4f}  {r/ref:>9.2f}x")
    print(f"\n   d^-1/2 = {ref:.4f}   -- the column is flat in D, as claimed\n")


def sweep_d(rng):
    print("2. Hidden width d is what controls the error (D fixed at 512)")
    print("   halving the error should take a 4x wider model\n")
    print(f"   {'d':>8}  {'|E|/S':>10}  {'d^-1/2':>10}  {'ratio':>8}")
    for d in (128, 512, 2048, 8192):
        r = ratio(512, d, rng)
        pred = d ** -0.5
        print(f"   {d:>8}  {r:>10.4f}  {pred:>10.4f}  {r/pred:>7.2f}x")
    print()


def real_models():
    print("3. What the rate predicts for the models in the paper\n")
    print(f"   {'model':<22} {'d':>6}  {'d^-1/2':>8}")
    for name, d in (("Llama3-8B-Instruct", 4096), ("Qwen3-8B", 4096)):
        print(f"   {name:<22} {d:>6}  {d ** -0.5:>8.4f}")
    print("\n   Figure 2b measures a mean cross-output gradient similarity of 0.016.")
    print("   The prediction is 1/sqrt(4096) = 0.0156. Two significant figures,")
    print("   on a quantity that could have come out anywhere in [0, 1].\n")


def low_rank(rng):
    print("4. Adding the low-rank spikes of Assumption E.2 (d = 512, D = 2048)")
    print("   these are NOT suppressed by d^-1/2 on their own; they stay small")
    print("   only because the isotropic bulk dominates the denominator\n")
    print(f"   {'rank r':>8}  {'lambda':>8}  {'|E|/S':>10}")
    for rank, lam in ((0, 0.0), (8, 1.0), (8, 10.0), (64, 10.0)):
        r = ratio(2048, 512, rng, rank=rank, beta=1.0, lam=lam)
        print(f"   {rank:>8}  {lam:>8.1f}  {r:>10.4f}")
    print()


def break_the_assumption(rng):
    print("5. The counterfactual: what if the head rows were NOT mean-zero?")
    print("   a shared direction makes the off-diagonal terms add coherently too,")
    print("   and then the error grows LINEARLY in D instead of cancelling\n")
    print(f"   {'D':>8}  {'isotropic':>10}  {'correlated':>11}")
    for D in (64, 256, 1024, 4096):
        iso = ratio(D, 512, rng)
        cor = ratio(D, 512, rng, shared=0.5)
        print(f"   {D:>8}  {iso:>10.4f}  {cor:>11.4f}")
    print("\n   With D = 128256 and d = 4096 a bias of that size would give a")
    print("   relative error of order D/sqrt(d) ~ 2000, i.e. pure noise.")
    print("   Assumption E.1 is not a technical convenience; it is the result.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(__doc__.split("Run:")[0].strip().split("\n")[0])
    print("=" * 66, "\n")
    sweep_D(rng)
    sweep_d(rng)
    real_models()
    low_rank(rng)
    break_the_assumption(rng)
