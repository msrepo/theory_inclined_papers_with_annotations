#!/usr/bin/env python3
"""Conditional entropy as a transferability bound, and the term people drop.

Tran et al. bound the transferred log-likelihood below by

    Trf(Z -> Y)  >=  l_Z(w_Z, h_Z)  -  H(Y|Z)                        (Thm 1)

with H(Y|Z) computed by counting on two label sequences over the same inputs.
The measure needs no model at all; the BOUND needs a trained source model, and
that asymmetry is where the trouble is.

Checked here:

  * the identity the proof turns on -- (1/n) sum_i log Phat(y_i|z_i) is
    EXACTLY -H(Y|Z) when Phat is the hard empirical conditional. This is why
    Tran's one-term-drop closes and the same step in LEEP's Property 2 does
    not: LEEP's conditional is soft, and the hard one dominates it;
  * Theorem 1 itself, on a trained source and a transferred head;
  * the hardness bound Hard(Z) <= H(Z|C), which is just H(Z);
  * the toy taxonomy of Figure 1, where H(Y|Z) separates the five cases;
  * and the term that gets dropped: with the source FIXED, l_Z is constant and
    ranking by -H(Y|Z) is valid. For source SELECTION -- which is what the
    later papers use NCE for -- l_Z varies and dropping it misorders.

Standard library and numpy only.

Run:  python3 nce.py
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ measures
def joint(y, z, C, S):
    """Phat(y, z) by HARD counting -- Eq 6. Both labels are given."""
    J = np.zeros((C, S))
    np.add.at(J, (y, z), 1.0)
    return J / len(y)


def cond_entropy(y, z, C, S):
    """H(Y|Z), Eq 7."""
    J = joint(y, z, C, S)
    Pz = J.sum(0)
    nz = J > 0
    return float(-(J[nz] * np.log(J[nz] / np.broadcast_to(Pz, J.shape)[nz])).sum())


def softmax(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def fit_head(R, lab, C, steps=3000, lr=0.5):
    """Max-likelihood linear head on a FIXED representation R. Returns (l, W)."""
    X = np.hstack([R, np.ones((len(R), 1))])
    W = np.zeros((C, X.shape[1]))
    for _ in range(steps):
        W -= lr * ((softmax(X @ W.T) - np.eye(C)[lab]).T @ X / len(X))
    p = softmax(X @ W.T)
    return float(np.mean(np.log(np.maximum(p[np.arange(len(lab)), lab], 1e-300)))), W


def whiten(F):
    F = F - F.mean(0)
    C = F.T @ F / len(F) + 1e-8 * np.eye(F.shape[1])
    return F @ np.linalg.inv(np.linalg.cholesky(C)).T


def train_source(F, z, S, D, rng, steps=4000, lr=0.1):
    """Learn a representation w_Z : R^k -> R^D and head h_Z, jointly, on Z.

    Inputs are whitened and the step is modest: joint descent on (Wr, Wh) is
    non-convex and happily diverges otherwise, which silently poisons l_Z.
    """
    F = whiten(F)
    k = F.shape[1]
    Wr = rng.standard_normal((k, D)) * (1.0 / np.sqrt(k))
    Wh = np.zeros((S, D + 1))
    for _ in range(steps):
        R = F @ Wr
        X = np.hstack([R, np.ones((len(R), 1))])
        G = softmax(X @ Wh.T) - np.eye(S)[z]
        gWh = G.T @ X / len(X)
        gWr = F.T @ (G @ Wh[:, :D]) / len(X)
        Wh -= lr * gWh
        Wr -= lr * gWr
    R = F @ Wr
    X = np.hstack([R, np.ones((len(R), 1))])
    p = softmax(X @ Wh.T)
    lZ = float(np.mean(np.log(np.maximum(p[np.arange(len(z)), z], 1e-300))))
    return R, p, lZ


def l_kbar(psrc, y, z, C, S):
    """Log-likelihood of the classifier kbar of Eq 9 -- the one the theorem
    assumes lives in K. Without it in K the bound simply does not apply."""
    J = joint(y, z, C, S)
    condp = J / np.maximum(J.sum(0), 1e-300)          # Phat(y|z)
    pred = psrc @ condp.T                             # sum_z Phat(y|z) p_Z(z|x)
    return float(np.mean(np.log(np.maximum(pred[np.arange(len(y)), y], 1e-300))))


def task(rng, n=3000, k=12, C=4, S=5, align=2.0, src_scale=1.5):
    """Two label sequences over the same inputs, coupled through a latent."""
    z = rng.integers(0, S, n)
    mu = rng.standard_normal((S, k)) * src_scale
    F = mu[z] + rng.standard_normal((n, k))
    # y depends on z with strength `align`: align -> large means near 1-1
    W = rng.standard_normal((S, C)) * align
    py = softmax(W)[z]
    y = np.array([rng.choice(C, p=r) for r in py])
    return F, y, z, C, S


# ---------------------------------------------------------------------- tests
def the_identity(rng):
    print("1. The identity Theorem 1 turns on (and LEEP cannot reuse)\n")
    _, y, z, C, S = task(rng)
    J = joint(y, z, C, S)
    condp = J / np.maximum(J.sum(0), 1e-300)
    plug = float(np.mean(np.log(np.maximum(condp[y, z], 1e-300))))
    H = cond_entropy(y, z, C, S)
    print(f"   (1/n) sum_i log Phat(y_i | z_i)      {plug:+.12f}")
    print(f"   -H(Y|Z)                              {-H:+.12f}")
    print(f"   difference                            {abs(plug + H):.2e}\n")
    print("   EXACT, because Phat here is the hard empirical conditional and")
    print("   (1/n) sum_i log Q(y_i|z_i) is maximised over conditionals Q by")
    print("   exactly that Phat. Tran's proof drops all but the z_i term inside")
    print("   the log and lands on -H(Y|Z) on the nose.\n")
    print("   LEEP runs the same argument with a SOFT conditional, and the same")
    print("   step then lands strictly BELOW NCE -- so it cannot reach LEEP's")
    print("   Property 2. Here is the gap, on this data:\n")
    # soft version: replace the one-hot z by a softmax over z
    theta = softmax(rng.standard_normal((len(y), S)) * 0.5 + np.eye(S)[z] * 2.0)
    Ps = np.stack([theta[y == c].sum(0) for c in range(C)]) / len(y)
    cs = Ps / np.maximum(Ps.sum(0), 1e-300)
    zs = theta.argmax(1)
    soft = float(np.mean(np.log(np.maximum(cs[y, zs], 1e-300))))
    print(f"     soft conditional, plugged in        {soft:+.6f}")
    print(f"     -H(Y|Z) on the same hard labels     {-cond_entropy(y, zs, C, S):+.6f}")
    print(f"     soft is below hard by               {-cond_entropy(y, zs, C, S) - soft:.6f}\n")


def theorem_one(rng):
    print("2. Theorem 1:  Trf(Z -> Y)  >=  l_Z  -  H(Y|Z)\n")
    print("   Trf is the max over K union {kbar}, as the paper's Discussion 2")
    print("   specifies -- the bound is not claimed for a K that excludes kbar.\n")
    print(f"   {'alignment':>10}  {'l_Z':>9}  {'H(Y|Z)':>9}  {'bound':>9}"
          f"  {'Trf (actual)':>13}  {'holds':>6}")
    for align in (0.3, 1.0, 2.0, 4.0):
        F, y, z, C, S = task(rng, align=align)
        R, psrc, lZ = train_source(F, z, S, 8, rng)
        H = cond_entropy(y, z, C, S)
        lin, _ = fit_head(R, y, C)                      # freeze w_Z, retrain head on Y
        kb = l_kbar(psrc, y, z, C, S)
        trf = max(lin, kb)
        print(f"   {align:>10.1f}  {lZ:>9.4f}  {H:>9.4f}  {lZ - H:>9.4f}"
              f"  {trf:>13.4f}  {'yes' if trf >= lZ - H - 1e-9 else 'NO':>6}")
    print("\n   Higher alignment means Y is more nearly a function of Z, so H(Y|Z)")
    print("   falls and the bound tightens.\n")


def hardness(rng):
    print("3. Hardness:  Hard(Z) = -l_Z  <=  H(Z|C),  and H(Z|C) is just H(Z)\n")
    print("   C is the trivial constant label sequence, so conditioning on it")
    print("   is no conditioning at all.\n")
    print(f"   {'|Z|':>6}  {'H(Z|C)':>9}  {'H(Z)':>9}  {'Hard(Z) = -l_Z':>16}  {'holds':>6}")
    for S in (2, 4, 8, 16):
        F, _, z, _, _ = task(rng, S=S, C=3)
        const = np.zeros(len(z), dtype=int)
        HzC = cond_entropy(z, const, S, 1)
        p = np.bincount(z, minlength=S) / len(z)
        Hz = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        _, _, lZ = train_source(F, z, S, 8, rng)
        print(f"   {S:>6}  {HzC:>9.4f}  {Hz:>9.4f}  {-lZ:>16.4f}"
              f"  {'yes' if -lZ <= HzC + 1e-9 else 'NO':>6}")
    print()


def toy_taxonomy():
    print("4. What H(Y|Z) actually measures, on constructed cases\n")
    print("   Three regimes, and they are the whole intuition:\n")
    n = 4096
    x = np.arange(n)
    rows = []
    # a source that is a bijection onto Y: nothing left to learn
    rows.append(("Z bijective with Y (|Y|=4)", x % 4, x % 4, 0.0))
    # each source class splits into k target classes
    for k in (2, 4):
        z = x % 4
        y = z * k + (x // 4) % k
        rows.append((f"each Z class splits into {k}", z, y, np.log(k)))
    # a trivial (constant) source: nothing to condition on, so H(Y|Z) = H(Y)
    for M in (4, 16):
        rows.append((f"Z trivial, |Y| = {M}", np.zeros(n, int), x % M, np.log(M)))

    print(f"   {'case':>30}  {'H(Y|Z)':>9}  {'predicted':>10}")
    for tag, z, y, pred in rows:
        H = cond_entropy(y, z, int(y.max()) + 1, int(z.max()) + 1)
        print(f"   {tag:>30}  {H:>9.4f}  {pred:>10.4f}")
    print(f"\n   log 2 = {np.log(2):.4f},  log 4 = {np.log(4):.4f},"
          f"  4 log 2 = log 16 = {4*np.log(2):.4f}")
    print("   A trivial source gives H(Y|Z) = H(Y): there is nothing to condition")
    print("   on, so the transfer must supply all of it. That is the paper's")
    print("   hardest case, and its 4 log 2 is this row with a 16-class target.\n")


def the_dropped_term(rng):
    print("5. The term the later papers drop\n")
    print("   Theorem 1 has TWO terms. With the source FIXED, l_Z is constant and")
    print("   ranking targets by -H(Y|Z) is licensed -- that is the paper's own")
    print("   stated use. NCE is later used for source SELECTION, where the target")
    print("   is fixed and the source varies, and then l_Z varies too.\n")
    print("   Whether that matters depends on whether the two terms move together.")
    print("   Vary them INDEPENDENTLY -- how noisily x carries Z (which drives l_Z)")
    print("   against how tightly Y follows Z (which drives H(Y|Z)) -- and they")
    print("   come apart:\n")
    n, k, C, S = 3000, 12, 4, 4
    print(f"   {'source':>24}  {'l_Z':>8}  {'H(Y|Z)':>8}  {'-H(Y|Z)':>9}"
          f"  {'l_Z-H':>8}  {'actual':>8}")
    rows = []
    for tag, align, noise in [("clean x, loose Y|Z", 0.8, 0.3),
                              ("clean x, tight Y|Z", 6.0, 0.3),
                              ("noisy x, tight Y|Z", 6.0, 9.0),
                              ("noisy x, loose Y|Z", 0.8, 9.0),
                              ("v.noisy x, tight Y|Z", 6.0, 25.0)]:
        lat = rng.standard_normal((n, k))
        z = np.array([rng.choice(S, p=r)
                      for r in softmax(lat @ (rng.standard_normal((k, S)) * 1.5))])
        y = np.array([rng.choice(C, p=r)
                      for r in softmax(rng.standard_normal((S, C)) * align)[z]])
        F = lat + rng.standard_normal((n, k)) * noise      # x carries z only noisily
        R, psrc, lZ = train_source(F, z, S, 8, rng)
        H = cond_entropy(y, z, C, S)
        trf = max(fit_head(R, y, C)[0], l_kbar(psrc, y, z, C, S))
        rows.append((tag, -H, lZ - H, trf))
        print(f"   {tag:>24}  {lZ:>8.4f}  {H:>8.4f}  {-H:>9.4f}"
              f"  {lZ - H:>8.4f}  {trf:>8.4f}")

    def spear(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))

    nh = [r[1] for r in rows]; full = [r[2] for r in rows]; tr = [r[3] for r in rows]
    print(f"\n   Spearman(-H(Y|Z), actual)        {spear(nh, tr):+.3f}")
    print(f"   Spearman(l_Z - H(Y|Z), actual)   {spear(full, tr):+.3f}\n")
    best_nce = max(rows, key=lambda r: r[1])
    best_true = max(rows, key=lambda r: r[3])
    if best_nce[0] != best_true[0]:
        print(f"   NCE's top pick:      {best_nce[0]:>22}"
              f"   (-H = {best_nce[1]:+.4f}, actual {best_nce[3]:+.4f})")
        print(f"   actually best:       {best_true[0]:>22}"
              f"   (-H = {best_true[1]:+.4f}, actual {best_true[3]:+.4f})")
        print("   -> NCE picks the wrong source. Training on an unpredictable Z")
        print("      yields a representation that learned nothing, and H(Y|Z)")
        print("      cannot see that: it never looks at x.\n")
    print("   Keeping l_Z is what the theorem licenses. Dropping it assumes source")
    print("   hardness and label alignment move together -- where they do, NCE is")
    print("   fine, and a generator that varies them in step shows no disagreement")
    print("   at all. The assumption is real, and it is unstated.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Conditional entropy as a transferability bound (Tran et al. 2019)")
    print("=" * 72, "\n")
    the_identity(rng)
    theorem_one(rng)
    hardness(rng)
    toy_taxonomy()
    the_dropped_term(rng)
