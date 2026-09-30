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
    later papers use NCE for -- l_Z varies and dropping it misorders;
  * Figure 1 counted as drawn: the paper's "4 log 2" for panels (a), (b) needs
    16 balanced target labels, and the figure has 2;
  * the one-term-drop step of the proof, and exactly how much it gives away;
  * Eq. 14 without Theorem 1: the input-blind frequency predictor already has
    loss H(Z);
  * for a binary target, H(Y|Z) squeezes the error of the best Z-only guesser
    into [h^-1(H), H / (2 ln 2)] (Fano and concavity), both ends attained;
  * the paper's own Tables 3 and 4 (40 CelebA attributes): what the reported
    correlation is made of, which attributes fall outside that window, and
    how much the label entropy H(Y) alone explains;
  * the downward bias of the plug-in H(Y|Z) with ~18 images per identity.

Standard library and numpy only.

Run:  python3 nce.py            (checks, prints every number the notes quote)
      python3 nce.py --figures  (checks, then rewrites ../figures/*.svg)
"""
from __future__ import annotations

import sys
from pathlib import Path

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
        H = cond_entropy(y, z, int(y.max()) + 1, int(z.max()) + 1) + 0.0
        print(f"   {tag:>30}  {H:>9.4f}  {pred:>10.4f}")
    print(f"\n   log 2 = {np.log(2):.4f},  log 4 = {np.log(4):.4f},"
          f"  4 log 2 = log 16 = {4*np.log(2):.4f}")
    print("   A trivial source gives H(Y|Z) = H(Y): there is nothing to condition")
    print("   on, so the transfer must supply all of it. Reaching the paper's")
    print("   4 log 2 that way needs a 16-class balanced target (last row); the")
    print("   paper's Figure 1(a,b) has a BINARY target -- see section 6.\n")


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


# ------------------------------------------------- the paper's own numbers
# Tables 3 and 4 of the arXiv v1 PDF (appendix E and F), CelebA, 40 attributes.
#   ce   : H(Y | identity), Table 3 row 9 ("Conditional Entropy")
#   hy   : H(Y), the "hardness" of Table 4, which Eq. 14 shows is the plain label entropy
#   trf  : accuracy of the linear SVM on face-recognition features, Table 3 row 11
#   ded  : accuracy of a dedicated ResNet-18 trained per attribute, Table 3 row 10
# name, ce, hy, trf, ded
CELEBA = [
    ("Male", .017, .679, .992, .985), ("Bald", .026, .107, .991, .990),
    ("Gray Hair", .052, .174, .981, .980), ("Mustache", .062, .173, .968, .968),
    ("Double Chin", .083, .189, .963, .959), ("Chubby", .087, .220, .957, .951),
    ("Sideburns", .088, .217, .976, .976), ("Goatee", .089, .235, .973, .974),
    ("Young", .095, .535, .899, .879), ("Wearing Hat", .107, .194, .988, .991),
    ("Eyeglasses", .109, .241, .996, .997), ("Pale Skin", .122, .177, .958, .970),
    ("Necktie", .131, .261, .941, .963), ("Blurry", .139, .201, .956, .963),
    ("No Beard", .141, .448, .958, .961), ("Receding Hairline", .141, .278, .933, .936),
    ("5 o'clock Shadow", .145, .349, .937, .942), ("Rosy Cheeks", .152, .242, .939, .950),
    ("Blond Hair", .160, .419, .949, .961), ("Big Lips", .161, .552, .710, .715),
    ("Bushy Eyebrows", .192, .409, .919, .927), ("Lipstick", .202, .692, .940, .935),
    ("Big Nose", .232, .545, .845, .828), ("Bangs", .236, .425, .950, .961),
    ("Narrow Eyes", .252, .357, .863, .875), ("Necklace", .252, .373, .865, .859),
    ("Heavy Makeup", .270, .667, .897, .916), ("Black Hair", .286, .550, .869, .901),
    ("Earrings", .291, .485, .853, .896), ("Arched Eyebrows", .306, .580, .822, .834),
    ("Brown Hair", .315, .508, .854, .886), ("Bags Under Eyes", .324, .507, .838, .834),
    ("Oval Face", .339, .597, .733, .752), ("Straight Hair", .339, .512, .812, .836),
    ("Pointy Nose", .341, .591, .769, .769), ("Attractive", .361, .693, .820, .823),
    ("Wavy Hair", .381, .627, .800, .842), ("High Cheekbones", .476, .689, .859, .878),
    ("Smiling", .521, .693, .909, .933), ("Mouth Open", .551, .693, .901, .943),
]
CELEBA_N, CELEBA_Z = 182_626, 10_177        # training images, identities (Sec. 5)

# The Pearson r printed on every panel of the paper's CE-versus-error figures.
FIG7_CELEBA = [.79, .88, .88, .86, .73, .85, .85, .86, .86, .85, .85, .85, .90, .86, .87, .82, .83, .78, .93, .83,
               .93, .86, .76, .84, .86, .85, .93, .90, .84, .84, .74, .87, .80, .84, .83, .78, .94, .80, .80, .82]
FIG8_10_AWA2 = [.96, .94, .97, .96, .96, .97, .97, .96, .96, .96, .96, .96, .97, .95, .95, .95, .94, .94, .97, .97,
                .94, .95, .92, .95, .95, .95, .96, .91, .96, .96, .95, .93, .97, .95, .95, .97, .97, .97, .97, .97,
                .92, .95, .95, .95, .95, .98, .92, .94, .95, .95, .92, .97, .92, .96, .96, .96, .95, .93, .93, .93,
                .96, .93, .95, .96, .96, .97, .97, .96, .97, .97, .96, .96, .96, .98, .98, .97, .95, .95, .93, .93,
                .94, .93, .94, .95, .95]
FIG2_MAIN_CELEBA = {18: .93, 20: .93, 26: .93, 36: .94}        # source attributes shown in Fig. 2(a-d)
FIG2_MAIN_ALL = [.93, .93, .93, .94, .97, .95, .92, .95, .95, .94, .94, .95]   # all twelve panels


def h_bin(p):
    """Binary entropy in nats, vectorised, 0 at the ends."""
    p = np.clip(np.asarray(p, float), 1e-300, 1 - 1e-16)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def h_bin_inv(H):
    """The p in [0, 1/2] with h(p) = H, by bisection (H in [0, ln 2])."""
    H = np.atleast_1d(np.asarray(H, float))
    lo, hi = np.zeros_like(H), np.full_like(H, .5)
    for _ in range(60):
        mid = (lo + hi) / 2
        below = h_bin(mid) < H
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return (lo + hi) / 2


def minority_rate(H):
    return h_bin_inv(H)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)


def resid(y, x):
    A = np.c_[x, np.ones_like(x)]
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def celeba_arrays():
    names = [r[0] for r in CELEBA]
    ce, hy, trf, ded = (np.array([r[i] for r in CELEBA]) for i in (1, 2, 3, 4))
    return names, ce, hy, 1 - trf, 1 - ded


def null_plugin_ce(p, m):
    """E[plug-in H(Y|Z)] when Y ~ Bernoulli(p) is INDEPENDENT of Z and every
    Z class has m examples: the exact binomial expectation of h(k/m)."""
    from math import lgamma
    k = np.arange(m + 1)
    lp = np.array([lgamma(m + 1) - lgamma(j + 1) - lgamma(m - j + 1) for j in k], float)
    lp += np.where(k > 0, k * np.log(p), 0.0) + np.where(m - k > 0, (m - k) * np.log1p(-p), 0.0)
    return float((np.exp(lp) * h_bin(k / m)).sum())


def null_plugin_ce_celeba(p):
    m = CELEBA_N / CELEBA_Z                         # 17.94 images per identity
    f = m - int(m)
    return (1 - f) * null_plugin_ce(p, int(m)) + f * null_plugin_ce(p, int(m) + 1)


# ------------------------------------------------------ the added sections
def figure_one_as_drawn():
    print("6. Figure 1 counted as drawn\n")
    print("   Eight examples per panel, equal runs, exactly as the figure draws")
    print("   them: cyan is Y, red is Z. (a) and (b) have a constant Z.\n")
    z0 = [0] * 8
    panels = [
        ("(a)", z0, [0, 0, 0, 0, 1, 1, 1, 1]),
        ("(b)", z0, [0, 0, 1, 1, 0, 0, 1, 1]),
        ("(c)", [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]),
        ("(d)", [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1]),
        ("(e)", [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2, 3, 3]),
    ]
    paper = {"(a)": "4 log 2", "(b)": "4 log 2", "(c)": "0", "(d)": "log 2", "(e)": "log 2"}
    print(f"   {'panel':>6}  {'|Y|':>4}  {'H(Y)':>7}  {'H(Y|Z)':>8}  {'I(Y;Z)':>7}   paper says")
    out = []
    for tag, z, y in panels:
        z, y = np.array(z), np.array(y)
        C, S = int(y.max()) + 1, int(z.max()) + 1
        H = cond_entropy(y, z, C, S) + 0.0            # + 0.0 turns -0.0 into 0.0
        py = np.bincount(y, minlength=C) / len(y)
        Hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
        print(f"   {tag:>6}  {C:>4}  {Hy:>7.4f}  {H:>8.4f}  {Hy - H:>7.4f}   {paper[tag]}")
        out.append((tag, z.tolist(), y.tolist(), H))
    print(f"\n   log 2 = {np.log(2):.4f}   4 log 2 = {4 * np.log(2):.4f}")
    print("   A binary target has H(Y) <= log 2, so a trivial source cannot give")
    print("   4 log 2 for it: the text's number needs 16 balanced target labels.")
    print("   As drawn, (a), (b), (d) and (e) all tie at log 2 and only (c) differs.")
    print("   (a), (b), (d) share I(Y;Z) = 0; (e) reaches log 2 by a different route,")
    print("   log 4 of target entropy minus log 2 the source supplies.\n")
    return out


def toy_drop(c):
    """Six points, Z = (0,0,0,1,1,1), Y = (0,0,1,1,2,2); a source that puts c on
    the right z. Returns (l_Z, l_Y(kbar) with the full sum, bound, gap, H(Y|Z))."""
    z = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 0, 1, 1, 2, 2])
    C, S = 3, 2
    J = joint(y, z, C, S)
    condp = J / J.sum(0)                            # Phat(y | z), columns are z
    H = cond_entropy(y, z, C, S)
    pz = np.where(np.arange(S)[None, :] == z[:, None], c, 1 - c)       # p(z | x_i)
    lZ = float(np.mean(np.log(pz[np.arange(6), z])))
    lY = float(np.mean(np.log((condp[y] * pz).sum(1))))                # full sum, Eq. 11 line 1
    return lZ, lY, lZ - H, lY - (lZ - H), H


def one_term_drop():
    print("7. The one-term-drop step, and exactly what it gives away\n")
    print("   Six training points: Z = (0,0,0,1,1,1), Y = (0,0,1,1,2,2). The source")
    print("   model puts probability c on the right z and 1-c on the other one.\n")
    print(f"   H(Y|Z) = {toy_drop(0.9)[4]:.4f}")
    print(f"   {'c':>7}  {'l_Z':>8}  {'l_Y(kbar)':>10}  {'bound':>8}  {'gap':>8}  {'(1/3)log(1/c)':>14}")
    rows = []
    for c in (0.5, 0.8, 0.95, 0.999):
        lZ, lY, bound, gap, _ = toy_drop(c)
        rows.append((c, lZ, lY, bound, gap))
        print(f"   {c:>7.3f}  {lZ:>8.4f}  {lY:>10.4f}  {bound:>8.4f}  {gap:>8.4f}  {np.log(1 / c) / 3:>14.4f}")
    print("\n   The gap is (1/3) log(1/c), and the 1/3 is the fraction of points (the two")
    print("   with y = 1) whose label is possible under BOTH source classes: only those")
    print("   have a second term to throw away. Where P(y_i | z) is zero for every")
    print("   other z, or the source is certain, the step loses nothing.\n")
    return rows


def eq14_without_theorem(rng):
    print("8. Eq. 14 needs no theorem: the input-blind frequency predictor\n")
    print(f"   {'|Z|':>5}  {'H(Z)':>8}  {'loss of Phat(z)':>16}  {'best of 2000 rivals':>20}")
    for S in (2, 5, 16):
        z = rng.integers(0, S, 4000)
        f = np.bincount(z, minlength=S) / len(z)
        Hz = float(-(f[f > 0] * np.log(f[f > 0])).sum())
        loss = float(-np.mean(np.log(f[z])))
        best = min(float(-np.mean(np.log(q[z]))) for q in rng.dirichlet(f * 200 + 1, 2000))
        print(f"   {S:>5}  {Hz:>8.4f}  {loss:>16.4f}  {best:>20.4f}")
    print("\n   A head that ignores x and outputs the label frequencies has training")
    print("   cross-entropy exactly H(Z), and no other input-blind head does better.")
    print("   So Hard(T^Z) <= H(Z) holds for any model class that can output a bias.\n")


def fano_window(rng):
    print("9. A binary target: H(Y|Z) squeezes the error of the best Z-only guesser\n")
    print("   Guess the majority y within each z. Its error is P_e = sum_z P(z) min(q_z, 1-q_z)")
    print("   and H(Y|Z) = sum_z P(z) h(q_z), with h the binary entropy. Jensen on h")
    print("   gives H <= h(P_e); h(q) >= 2 ln2 min(q, 1-q) gives P_e <= H / (2 ln 2).\n")
    lo_slack, up_slack, n = 1.0, 1.0, 0
    for trial in range(2000):
        S = int(rng.integers(1, 40))
        pz = rng.dirichlet(np.ones(S) * rng.choice([.3, 1, 5]))
        q = rng.beta(*rng.choice([[.15, .15], [1, 1], [.5, 4], [4, .5]]), S)
        Pe = float((pz * np.minimum(q, 1 - q)).sum())
        H = float((pz * h_bin(q)).sum())
        lo = float(h_bin_inv(H)[0]); up = H / (2 * np.log(2))
        lo_slack = min(lo_slack, Pe - lo); up_slack = min(up_slack, up - Pe)
        n += 1
    print(f"   {n} random binary tables: smallest P_e - h^-1(H) = {lo_slack:.1e}, smallest H/(2 ln 2) - P_e = {up_slack:.1e}")
    print("   (non-negative up to floating-point round-off, so both ends hold)\n")
    print("   Both ends are attained:")
    print("     lower: one z class with q = 0.1        -> P_e = 0.1,  H = h(0.1)  = %.4f, h^-1(H) = %.4f"
          % (float(h_bin(0.1)), float(h_bin_inv(h_bin(0.1))[0])))
    print("     upper: half the mass at q = 1/2, rest at 0 -> P_e = 0.25, H = (1/2) ln 2 = %.4f, H/(2 ln 2) = %.4f\n"
          % (0.5 * np.log(2), 0.5 * np.log(2) / (2 * np.log(2))))
    print(f"   {'H(Y|Z)':>8}  {'P_e >=':>8}  {'P_e <=':>8}")
    for H in (0.017, 0.1, 0.3, 0.5, 0.6931):
        print(f"   {H:>8.3f}  {float(h_bin_inv(H)[0]):>8.4f}  {H / (2 * np.log(2)):>8.4f}")
    print("\n   So for a binary target, the label-only number pins the ideal error to a")
    print("   window whose top is 2.4x its bottom at H = 0.3. That is most of what a")
    print("   correlation between CE and error, across binary targets, is made of.\n")


def paper_tables():
    names, ce, hy, err, errd = celeba_arrays()
    print("10. The paper's own Tables 3 and 4 (CelebA, 40 attributes, identity as source)\n")
    print(f"   Pearson(CE, transfer error)              {pearson(ce, err):.3f}   (Fig. 3 prints 0.78)")
    print(f"   Spearman(CE, transfer error)             {spearman(ce, err):.3f}")
    i = [k for k, n in enumerate(names) if n != "Big Lips"]
    print(f"   Pearson without Big Lips                 {pearson(ce[i], err[i]):.3f}")
    print(f"   Pearson(H(Y), transfer error)            {pearson(hy, err):.3f}   <- no source used at all")
    print(f"   Spearman(H(Y), transfer error)           {spearman(hy, err):.3f}   <- rank correlation: CE is ahead here")
    print(f"   Pearson(CE, H(Y))                        {pearson(ce, hy):.3f}")
    print(f"   Pearson(CE/H(Y), transfer error)         {pearson(ce / hy, err):.3f}")
    print(f"   partial r(CE, err | H(Y))                {pearson(resid(ce, hy), resid(err, hy)):.3f}")
    print(f"   partial r(H(Y), err | CE)                {pearson(resid(hy, ce), resid(err, ce)):.3f}")
    print(f"   Pearson(CE, dedicated-network error)     {pearson(ce, errd):.3f}")
    print(f"   Pearson(H(Y), dedicated-network error)   {pearson(hy, errd):.3f}   (Fig. 6a prints 0.58: the tables and this transcription agree)")
    pm = minority_rate(hy)
    I = hy - ce
    print(f"   Pearson(H(Y) - CE, base error - transfer error)   {pearson(I, pm - err):.3f}"
          "   (plain-accuracy reading)")
    print(f"\n   Smallest CE values are {', '.join(f'{n} {c:.3f}' for n, c in zip(names[:3], ce[:3]))};"
          f" H(Y) for them {', '.join(f'{h:.3f}' for h in hy[:3])}\n")
    lo = h_bin_inv(ce); up = ce / (2 * np.log(2))
    inside = [(l - 1e-9 <= e <= u) for e, l, u in zip(err, lo, up)]
    above = [(n, e, u) for n, e, u, ok in zip(names, err, up, inside) if e > u]
    below = [(n, e, l) for n, e, l, ok in zip(names, err, lo, inside) if e < l - 1e-9]
    print(f"   The Fano window [h^-1(CE), CE/(2 ln 2)] of item 9, against the SVM's test error:")
    print(f"     inside  {sum(inside)} of 40")
    print(f"     above   {len(above)}: " + ", ".join(f"{n} (err {e:.3f} > {u:.3f})" for n, e, u in above))
    print(f"     below   {len(below)}: " + ", ".join(f"{n} (err {e:.3f} < {l:.3f})" for n, e, l in below))
    print("   'above' = the linear head does worse than an identity-lookup could.")
    print("   'below' = the frozen features know more than the identity label does.\n")
    print("   Selected attributes (base error = the rarer class's rate, from H(Y)):")
    print(f"   {'attribute':>12}  {'CE':>6}  {'H(Y)':>6}  {'CE/H(Y)':>8}  {'base error':>10}  {'SVM error':>9}")
    for n in ("Male", "Bald", "Pale Skin", "Blurry", "Big Lips", "Smiling", "Mouth Open"):
        k = names.index(n)
        print(f"   {n:>12}  {ce[k]:>6.3f}  {hy[k]:>6.3f}  {ce[k] / hy[k]:>8.2f}  {pm[k]:>10.3f}  {err[k]:>9.3f}")
    print()
    print("   Rare attributes and what transfer bought over always saying 'no':")
    print(f"   {'attribute':>12}  {'CE':>6}  {'rank by CE':>10}  {'base error':>10}  {'SVM error':>9}")
    order = np.argsort(np.argsort(ce)) + 1
    for n in ("Pale Skin", "Blurry", "Necktie", "Bald"):
        k = names.index(n)
        print(f"   {n:>12}  {ce[k]:>6.3f}  {order[k]:>7d}/40  {pm[k]:>10.3f}  {err[k]:>9.3f}")
    print("   (base error = the rarer class's rate, from H(Y); it assumes the tables report")
    print("   plain accuracy. The paper does not say whether they are plain or class-balanced.)\n")


def coarse_target():
    print("13. A coarse target from a fine source: where the bound is loose\n")
    print("   Z has 2g classes and Y = Z // g, so Y is a function of Z: H(Y|Z) = 0. The source")
    print("   is confident about the group but spreads its probability evenly over the g")
    print("   classes inside it -- a confusion that cannot affect Y at all.\n")
    print(f"   {'g':>3}  {'l_Z':>8}  {'H(Y|Z)':>7}  {'bound':>8}  {'l_Y(kbar)':>10}  {'gap':>7}  {'ln g':>7}")
    for g in (1, 2, 4, 8):
        S = 2 * g
        z = np.repeat(np.arange(S), 50)
        y = z // g
        psrc = np.zeros((len(z), S))
        for i, zi in enumerate(z):
            psrc[i, (zi // g) * g:(zi // g + 1) * g] = 1.0 / g
        lZ = float(np.mean(np.log(psrc[np.arange(len(z)), z])))
        H = cond_entropy(y, z, 2, S) + 0.0
        lY = l_kbar(psrc, y, z, 2, S)
        print(f"   {g:>3}  {lZ:>8.4f}  {H:>7.4f}  {lZ - H:>8.4f}  {lY:>10.4f}  {lY - (lZ - H):>7.4f}  {np.log(g):>7.4f}")
    print("\n   The target log-likelihood is 0 (nothing to get wrong) but the bound is -ln g,")
    print("   the source's own within-group confusion. The bound is tight for a bijection")
    print("   and charges a coarse target for every distinction the source cannot make.\n")


def printed_correlations():
    print("14. The correlations the paper prints, all of them\n")
    f7, aw = np.array(FIG7_CELEBA), np.array(FIG8_10_AWA2)
    print(f"   main-text Fig. 2, the twelve panels shown:   r from {min(FIG2_MAIN_ALL):.2f} to {max(FIG2_MAIN_ALL):.2f}")
    print(f"   appendix Fig. 7, all {len(f7)} CelebA sources:      r from {f7.min():.2f} to {f7.max():.2f}, "
          f"mean {f7.mean():.3f}, median {np.median(f7):.2f}")
    print(f"   appendix Figs. 8-10, all {len(aw)} AwA2 sources:    r from {aw.min():.2f} to {aw.max():.2f}, "
          f"mean {aw.mean():.3f}, median {np.median(aw):.2f}")
    top4 = sorted(range(len(f7)), key=lambda i: -f7[i])[:4]
    shown = sorted(FIG2_MAIN_CELEBA)
    print(f"   the four CelebA sources shown in Fig. 2(a-d): attributes {shown}, r = {[FIG2_MAIN_CELEBA[k] for k in shown]}")
    print(f"   the four highest of the 40 in Fig. 7:          attributes {sorted(top4)}, r = {[float(f7[k]) for k in sorted(top4)]}")
    print(f"   same set: {sorted(top4) == shown}")
    print(f"   identity source (Fig. 3): 0.78; {int((f7 > 0.78).sum())} of the 40 binary-source values in Fig. 7 are higher\n")


def eight_images():
    print("12. The eight-image example the notes use to introduce H(Y|Z)\n")
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    J = joint(y, z, 2, 2)
    print("   Counts (rows y = 0, 1; columns z = A, B):", J.astype(float).__mul__(8).astype(int).tolist())
    P = J / J.sum(0)
    print(f"   P(y | z = A) = {P[:, 0].round(4).tolist()},  P(y | z = B) = {P[:, 1].round(4).tolist()}")
    h = lambda q: float(-(q[q > 0] * np.log(q[q > 0])).sum())    # noqa: E731
    print(f"   H(Y | z = A) = {h(P[:, 0]):.4f},  H(Y | z = B) = {h(P[:, 1]):.4f},  H(Y | Z) = {cond_entropy(y, z, 2, 2):.4f}")
    print(f"   H(Y) = {h(J.sum(1)):.4f} = ln 2, so knowing z removes {h(J.sum(1)) - cond_entropy(y, z, 2, 2):.4f} nats")
    sur = -np.log(P[y, z])
    print(f"   surprise -ln P(y_i | z_i) per image: {sur.round(4).tolist()}; average {sur.mean():.4f}\n")


def plugin_bias(rng):
    names, ce, hy, err, errd = celeba_arrays()
    print("11. Counting with ~18 images per identity: the plug-in H(Y|Z) is low\n")
    m = CELEBA_N / CELEBA_Z
    mm = (2 - 1) * CELEBA_Z / (2 * CELEBA_N)
    print(f"   n = {CELEBA_N:,} images, |Z| = {CELEBA_Z:,} identities: {m:.2f} per identity;")
    print(f"   the first-order (Miller-Madow) bias (|Y|-1)|Z|/(2n) = {mm:.4f} nats\n")
    print("   Y independent of identity, so the TRUE H(Y|Z) equals H(Y). Exact binomial")
    print("   expectation of the plug-in, and a Monte Carlo of the whole data set:\n")
    print(f"   {'p':>6}  {'H(Y)':>7}  {'exact E[plug-in]':>17}  {'shortfall':>10}  {'Monte Carlo':>12}")
    for p in (0.022, 0.1, 0.24, 0.5):
        H = float(h_bin(p)); nul = null_plugin_ce_celeba(p)
        sizes = np.full(CELEBA_Z, int(m)); sizes[: CELEBA_N - sizes.sum()] += 1
        zz = np.repeat(np.arange(CELEBA_Z), sizes)
        yy = (rng.random(CELEBA_N) < p).astype(int)
        mc = cond_entropy(yy, zz, 2, CELEBA_Z)
        print(f"   {p:>6.3f}  {H:>7.4f}  {nul:>17.4f}  {H - nul:>10.4f}  {mc:>12.4f}")
    print(f"\n   The shortfall (~0.03 nats) is as large as the smallest reported CEs:")
    print(f"   Male {ce[0]:.3f}, Bald {ce[1]:.3f}. The order of the top few attributes is")
    print("   inside the estimator's own bias.\n")
    nul = np.array([null_plugin_ce_celeba(float(minority_rate(h)[0])) for h in hy])
    ratio = ce / nul
    o = np.argsort(-ratio)[:4]
    print("   Reported CE as a fraction of what pure noise would give (1 = identity tells")
    print("   the plug-in nothing): " + ", ".join(f"{names[k]} {ratio[k]:.2f}" for k in o) + ".")
    print(f"   Smallest: {names[int(np.argmin(ratio))]} {ratio.min():.2f}. Pearson(noise CE, H(Y)) = {pearson(nul, hy):.4f}.\n")
    print(f"   The window's top CE/(2 ln 2) after adding the {mm:.3f} shortfall to each CE, for the three 'above' attributes:")
    for n in ("Young", "Big Lips", "Oval Face"):
        k = names.index(n)
        print(f"     {n:>10}: SVM error {err[k]:.3f}, top {ce[k] / (2 * np.log(2)):.3f} -> {(ce[k] + mm) / (2 * np.log(2)):.3f}")
    print()


# ------------------------------------------------------------------ figures
STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:10.5px;fill:#57564f} .v{font-size:10.5px;fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .box{fill:none;stroke:#c9c7bf;stroke-width:1}
  .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s4{stroke:#eda100} .s5{stroke:#8a8880}
  .f1{fill:#2a78d6} .f2{fill:#eb6834} .f3{fill:#1baf7a} .f4{fill:#eda100}
  .ln{stroke-width:2.2;fill:none;stroke-linecap:round;stroke-linejoin:round}
  .seg{stroke-width:4;stroke-linecap:round;fill:none}
  .dash{stroke-width:1.3;fill:none;stroke-dasharray:5 3}
  .win{fill:#2a78d6;opacity:.13} .gap{fill:#eb6834;opacity:.18}
  .ring{stroke:#fdfdfc;stroke-width:1.2} .arr{fill:#6f6d66}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .box{stroke:#4a4844} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500} .s5{stroke:#85837b}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500}
    .win{fill:#3987e5;opacity:.18} .gap{fill:#d95926;opacity:.24} .ring{stroke:#161615} .arr{fill:#a3a19a}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


def _runs(seq):
    out, s = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            out.append((s, i, seq[s])); s = i
    return out


def fig_figure1(panels):
    body = ['<text x="14" y="20" class="hd">The paper\'s Figure 1, counted as drawn: eight examples per panel, equal runs</text>']
    w, gap = 128, 12
    claim = {"(a)": ("text says 4 log 2 = 2.773", True), "(b)": ("text says 4 log 2 = 2.773", True),
             "(c)": ("text says 0", False), "(d)": ("text says log 2", False), "(e)": ("text says log 2", False)}
    body.append('<text x="14" y="70" class="sm">Y</text><text x="14" y="140" class="sm">Z</text>')
    for i, (tag, z, y, H) in enumerate(panels):
        x0 = 34 + i * (w + gap)
        sx = lambda k: x0 + k * w / 8            # noqa: E731
        for a, b, v in _runs(y):
            yy = 92 - 15 * v
            body.append(f'<path d="M {sx(a) + 2:.1f},{yy} H {sx(b) - 2:.1f}" class="seg s1"/>')
        for a, b, v in _runs(z):
            yy = 152 - 15 * v
            body.append(f'<path d="M {sx(a) + 2:.1f},{yy} H {sx(b) - 2:.1f}" class="seg s2"/>')
        body.append(f'<path d="M {x0},176 H {x0 + w - 6}" class="ax"/>'
                    f'<path d="M {x0 + w - 6},172 L {x0 + w + 2},176 L {x0 + w - 6},180 Z" class="arr"/>')
        body.append(f'<text x="{x0 + w / 2:.0f}" y="196" class="hd" text-anchor="middle">{tag}</text>')
        body.append(f'<text x="{x0 + w / 2:.0f}" y="216" class="v" text-anchor="middle">counted {H:.3f}</text>')
        txt, wrong = claim[tag]
        style = ' style="fill:#eb6834"' if wrong else ""
        body.append(f'<text x="{x0 + w / 2:.0f}" y="234" class="sm" text-anchor="middle"{style}>{txt}</text>')
    body.append('<text x="14" y="262" class="sm">Blue is Y, orange is Z; height is the label. A binary Y has H(Y) at most log 2 = 0.693, so a constant Z cannot give 4 log 2.</text>')
    return svg(760, 274, "Figure 1 of Tran et al. counted as drawn",
               "Five toy panels. Counted H(Y|Z) is 0.693 in panels a, b, d, e and 0 in c; the text gives a and b as 4 log 2.", body)


def _scatter_panel(body, ox, oy, w, h, xs, ys, xmax, ymax, xlabel, title, names, label_set):
    X = lambda v: ox + v / xmax * w                  # noqa: E731
    Y = lambda v: oy + h - v / ymax * h              # noqa: E731
    body.append(f'<text x="{ox}" y="{oy - 14}" class="hd">{title}</text>')
    for v in np.arange(0, ymax + 1e-9, 0.05):
        body.append(f'<path d="M {ox},{Y(v):.1f} H {ox + w}" class="gd"/>')
        body.append(f'<text x="{ox - 6}" y="{Y(v) + 4:.1f}" class="sm" text-anchor="end">{v:.2f}</text>')
    for v in np.arange(0, xmax + 1e-9, 0.1):
        body.append(f'<text x="{X(v):.1f}" y="{oy + h + 15}" class="sm" text-anchor="middle">{v:.1f}</text>')
    body.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" class="box"/>')
    A = np.c_[xs, np.ones_like(xs)]
    m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    body.append(f'<path d="M {X(0):.1f},{Y(c):.1f} L {X(xmax):.1f},{Y(m * xmax + c):.1f}" class="dash s5"/>')
    for n, x, y in zip(names, xs, ys):
        body.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.6" class="f1 ring"/>')
    for n, x, y in zip(names, xs, ys):
        if n in label_set:
            dx, anchor, txt = label_set[n]
            body.append(f'<text x="{X(x) + dx:.1f}" y="{Y(y) + 3.5:.1f}" class="sm" text-anchor="{anchor}">{txt}</text>')
    body.append(f'<text x="{ox + w / 2}" y="{oy + h + 32}" class="lab" text-anchor="middle">{xlabel}</text>')


def fig_scatter(names, ce, hy, err):
    body = []
    lab1 = {"Big Lips": (7, "start", "Big Lips"), "Male": (7, "start", "Male"),
            "Smiling": (-7, "end", "Smiling, Mouth Open")}
    lab2 = {"Big Lips": (7, "start", "Big Lips"), "Male": (-7, "end", "Male"),
            "Smiling": (-8, "end", "Smiling, Mouth Open")}
    r1, r2 = pearson(ce, err), pearson(hy, err)
    _scatter_panel(body, 56, 46, 300, 230, ce, err, 0.6, 0.30, "H(Y | identity), the paper's measure",
                   f"Test error vs the conditional entropy: r = {r1:.2f}", names, lab1)
    _scatter_panel(body, 466, 46, 300, 230, hy, err, 0.75, 0.30, "H(Y), the target's own label entropy: no source at all",
                   f"Test error vs the label entropy alone: r = {r2:.2f}", names, lab2)
    body.append(f'<text x="14" y="160" class="lab" transform="rotate(-90 14 160)" text-anchor="middle">transfer error, 1 - accuracy</text>')
    body.append(f'<text x="424" y="160" class="lab" transform="rotate(-90 424 160)" text-anchor="middle">transfer error</text>')
    return svg(800, 332, "Error against conditional entropy and against label entropy",
               f"Two scatter plots over 40 CelebA attributes; Pearson r is {r1:.2f} for the conditional entropy and {r2:.2f} for the label entropy alone.", body)


def fig_window(names, ce, err):
    ox, oy, w, h = 64, 40, 460, 300
    xmax, ymax = 0.6, 0.40
    X = lambda v: ox + v / xmax * w                  # noqa: E731
    Y = lambda v: oy + h - v / ymax * h              # noqa: E731
    body = [f'<text x="{ox}" y="24" class="hd">Where the SVM\'s test error falls against the window H(Y|Z) allows</text>',
            f'<clipPath id="pl"><rect x="{ox}" y="{oy}" width="{w}" height="{h}"/></clipPath>']
    for v in np.arange(0, ymax + 1e-9, 0.05):
        body.append(f'<path d="M {ox},{Y(v):.1f} H {ox + w}" class="gd"/>')
        body.append(f'<text x="{ox - 6}" y="{Y(v) + 4:.1f}" class="sm" text-anchor="end">{v:.2f}</text>')
    for v in np.arange(0, xmax + 1e-9, 0.1):
        body.append(f'<text x="{X(v):.1f}" y="{oy + h + 15}" class="sm" text-anchor="middle">{v:.1f}</text>')
    g = np.linspace(0, xmax, 121)
    lo = h_bin_inv(g); up = g / (2 * np.log(2))
    poly = "M " + " L ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(g, up)) + " L " + \
           " L ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(g[::-1], lo[::-1])) + " Z"
    body.append(f'<path d="{poly}" class="win" clip-path="url(#pl)"/>')
    body.append(f'<path d="M {X(0):.1f},{Y(0):.1f} L {X(xmax):.1f},{Y(xmax / (2 * np.log(2))):.1f}" class="dash s1" clip-path="url(#pl)"/>')
    body.append(f'<path d="' + "M " + " L ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(g, lo)) + '" class="dash s1" clip-path="url(#pl)"/>')
    body.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" class="box"/>')
    lo_ce, up_ce = h_bin_inv(ce), ce / (2 * np.log(2))
    lab = {"Big Lips": (8, "start", "Big Lips"), "Young": (8, "start", "Young"),
           "Oval Face": (-8, "end", "Oval Face"), "Smiling": (-8, "end", "Smiling, Mouth Open"),
           "Eyeglasses": (8, "start", "Eyeglasses, Wearing Hat"),
           "High Cheekbones": (-8, "end", "High Cheekbones")}
    for n, x, e, l, u in zip(names, ce, err, lo_ce, up_ce):
        cls = "f2" if e > u else ("f3" if e < l - 1e-9 else "f1")
        body.append(f'<circle cx="{X(x):.1f}" cy="{Y(e):.1f}" r="4" class="{cls} ring"/>')
    for n, x, e in zip(names, ce, err):
        if n in lab:
            dx, an, txt = lab[n]
            body.append(f'<text x="{X(x) + dx:.1f}" y="{Y(e) + 3.5:.1f}" class="sm" text-anchor="{an}">{txt}</text>')
    body.append(f'<text x="{X(0.40):.1f}" y="{Y(0.365):.1f}" class="sm" style="fill:#2a78d6" text-anchor="middle">error &lt;= CE / (2 ln 2)</text>')
    body.append(f'<text x="{X(0.42):.1f}" y="{Y(0.045):.1f}" class="sm" style="fill:#2a78d6" text-anchor="middle">error &gt;= h⁻¹(CE)</text>')
    body.append(f'<text x="{ox + w / 2}" y="{oy + h + 34}" class="lab" text-anchor="middle">CE = H(Y | identity)</text>')
    body.append(f'<text x="18" y="{oy + h / 2}" class="lab" transform="rotate(-90 18 {oy + h / 2})" text-anchor="middle">transfer error, 1 - accuracy</text>')
    lx, ly = ox + w + 26, oy + 30
    for j, (cls, txt) in enumerate([("f1", "31 inside the window"), ("f2", "3 above: the linear head does"),
                                    (None, "   worse than an identity lookup"), ("f3", "6 below: the features know"),
                                    (None, "   more than the identity label")]):
        if cls:
            body.append(f'<circle cx="{lx}" cy="{ly + 20 * j - 3}" r="4.5" class="{cls} ring"/>')
        body.append(f'<text x="{lx + 12}" y="{ly + 20 * j}" class="sm">{txt}</text>')
    body.append(f'<text x="{lx}" y="{ly + 128}" class="sm">Shaded: what the best guesser that</text>')
    body.append(f'<text x="{lx}" y="{ly + 143}" class="sm">sees only the identity can reach</text>')
    body.append(f'<text x="{lx}" y="{ly + 158}" class="sm">(Fano and concavity, both</text>')
    body.append(f'<text x="{lx}" y="{ly + 173}" class="sm">ends attained).</text>')
    return svg(760, 400, "Test error against the Fano window of the conditional entropy",
               "Forty CelebA attributes plotted by conditional entropy and linear SVM test error, with the window a Z-only guesser can reach shaded.", body)


def fig_drop():
    ox, oy, w, h = 64, 40, 420, 240
    cs = np.linspace(0.3, 0.999, 120)
    vals = np.array([toy_drop(c) for c in cs])
    lZ, lY, bd = vals[:, 0], vals[:, 1], vals[:, 2]
    ymin, ymax = -2.0, 0.0
    X = lambda v: ox + (v - 0.3) / (0.999 - 0.3) * w  # noqa: E731
    Y = lambda v: oy + (ymax - v) / (ymax - ymin) * h  # noqa: E731
    body = [f'<text x="{ox}" y="24" class="hd">What the one-term drop gives away, on six points (H(Y|Z) = {vals[0, 4]:.3f})</text>']
    for v in (-2.0, -1.5, -1.0, -0.5, 0.0):
        body.append(f'<path d="M {ox},{Y(v):.1f} H {ox + w}" class="gd"/>')
        body.append(f'<text x="{ox - 6}" y="{Y(v) + 4:.1f}" class="sm" text-anchor="end">{v:.1f}</text>')
    for v in (0.3, 0.5, 0.7, 0.9, 1.0):
        body.append(f'<text x="{X(min(v, 0.999)):.1f}" y="{oy + h + 15}" class="sm" text-anchor="middle">{v:g}</text>')
    up = " L ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in zip(cs, lY))
    dn = " L ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in zip(cs[::-1], bd[::-1]))
    body.append(f'<path d="M {up} L {dn} Z" class="gap"/>')
    body.append(f'<path d="M ' + " L ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in zip(cs, lY)) + '" class="ln s1"/>')
    body.append(f'<path d="M ' + " L ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in zip(cs, bd)) + '" class="ln s2"/>')
    body.append(f'<path d="M ' + " L ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in zip(cs, lZ)) + '" class="dash s5"/>')
    body.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" class="box"/>')
    body.append(f'<text x="{ox + w + 8}" y="{Y(lY[-1]) - 4:.1f}" class="v" style="fill:#2a78d6">ℓ<tspan dy="3" font-size="8.5">Y</tspan><tspan dy="-3">(k̄), the full sum</tspan></text>')
    body.append(f'<text x="{ox + w + 8}" y="{Y(bd[-1]) + 14:.1f}" class="v" style="fill:#eb6834">ℓ<tspan dy="3" font-size="8.5">Z</tspan><tspan dy="-3"> − H(Y|Z), the bound</tspan></text>')
    body.append(f'<text x="{ox + w + 8}" y="{Y(lZ[-1]) - 2:.1f}" class="sm">ℓ<tspan dy="3" font-size="8.5">Z</tspan><tspan dy="-3"> = log c</tspan></text>')
    body.append(f'<text x="{X(0.66):.1f}" y="{Y(-1.55):.1f}" class="sm">shaded gap = (1/3) log(1/c)</text>')
    body.append(f'<text x="{ox + w / 2}" y="{oy + h + 34}" class="lab" text-anchor="middle">c, the source model\'s probability on the true z</text>')
    body.append(f'<text x="18" y="{oy + h / 2}" class="lab" transform="rotate(-90 18 {oy + h / 2})" text-anchor="middle">nats</text>')
    return svg(760, 330, "The one-term drop in the proof of Theorem 1",
               "The exact target log-likelihood of the constructed classifier and the theorem's bound as the source confidence c grows; the gap is (1/3) log(1/c).", body)


def fig_bias(ce):
    ox, oy, w, h = 64, 40, 440, 240
    ms = np.arange(2, 61)
    ymax = 0.36
    X = lambda m: ox + (np.log(m) - np.log(2)) / (np.log(60) - np.log(2)) * w   # noqa: E731
    ymin = -0.03
    Y = lambda v: oy + h - (v - ymin) / (ymax - ymin) * h                        # noqa: E731
    body = [f'<text x="{ox}" y="24" class="hd">How far below the truth the plug-in H(Y|Z) lands when Y ignores Z</text>']
    for v in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        body.append(f'<path d="M {ox},{Y(v):.1f} H {ox + w}" class="gd"/>')
        body.append(f'<text x="{ox - 6}" y="{Y(v) + 4:.1f}" class="sm" text-anchor="end">{v:.2f}</text>')
    for m in (2, 5, 10, 20, 40, 60):
        body.append(f'<text x="{X(m):.1f}" y="{oy + h + 15}" class="sm" text-anchor="middle">{m}</text>')
    for p, cls in ((0.5, "s1"), (0.1, "s3"), (0.022, "s4")):
        yy = [float(h_bin(p)) - null_plugin_ce(p, int(m)) for m in ms]
        body.append(f'<path d="M ' + " L ".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in zip(ms, yy)) + f'" class="ln {cls}"/>')
    mmv = [1 / (2 * m) for m in ms]
    body.append(f'<path d="M ' + " L ".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in zip(ms, mmv)) + '" class="dash s5"/>')
    for name, v in (("Male", ce[0]), ("Bald", ce[1])):
        body.append(f'<path d="M {ox},{Y(v):.1f} H {ox + w}" class="dash s2"/>')
        body.append(f'<text x="{ox + 6}" y="{Y(v) + (-4 if name == "Bald" else 13):.1f}" class="sm" style="fill:#eb6834">reported CE of {name}: {v:.3f}</text>')
    mm = CELEBA_N / CELEBA_Z
    body.append(f'<path d="M {X(mm):.1f},{oy} V {oy + h}" class="dash s5"/>')
    body.append(f'<text x="{X(mm) + 5:.1f}" y="{oy + 12}" class="sm">CelebA: {mm:.1f} images per identity</text>')
    body.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" class="box"/>')
    lx = ox + w + 14
    for j, (cls, txt) in enumerate([("s1", "P(Y=1) = 0.5"), ("s3", "P(Y=1) = 0.1"), ("s4", "P(Y=1) = 0.022")]):
        body.append(f'<path d="M {lx},{oy + 20 + 18 * j} h 18" class="ln {cls}"/><text x="{lx + 24}" y="{oy + 24 + 18 * j}" class="sm">{txt}</text>')
    body.append(f'<path d="M {lx},{oy + 74} h 18" class="dash s5"/><text x="{lx + 24}" y="{oy + 78}" class="sm">1/(2m), first order</text>')
    body.append(f'<text x="{ox + w / 2}" y="{oy + h + 34}" class="lab" text-anchor="middle">images per identity m (log scale)</text>')
    body.append(f'<text x="18" y="{oy + h / 2}" class="lab" transform="rotate(-90 18 {oy + h / 2})" text-anchor="middle">H(Y) minus expected plug-in, nats</text>')
    return svg(760, 330, "Downward bias of the plug-in conditional entropy",
               "Shortfall of the plug-in estimate against images per identity for three base rates, with the smallest reported CEs as horizontal lines.", body)


def write_figures(fig1_panels):
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    names, ce, hy, err, errd = celeba_arrays()
    files = {
        "figure1-as-drawn.svg": fig_figure1(fig1_panels),
        "error-vs-entropy.svg": fig_scatter(names, ce, hy, err),
        "fano-window.svg": fig_window(names, ce, err),
        "one-term-drop.svg": fig_drop(),
        "plug-in-bias.svg": fig_bias(ce),
    }
    for name, text in files.items():
        (d / name).write_text(text)
    print("figures: wrote", ", ".join(sorted(files)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Conditional entropy as a transferability bound (Tran et al. 2019)")
    print("=" * 72, "\n")
    the_identity(rng)
    theorem_one(rng)
    hardness(rng)
    toy_taxonomy()
    the_dropped_term(rng)
    fig1 = figure_one_as_drawn()
    drop_rows = one_term_drop()
    eq14_without_theorem(np.random.default_rng(1))
    fano_window(np.random.default_rng(2))
    paper_tables()
    plugin_bias(np.random.default_rng(3))
    eight_images()
    coarse_target()
    printed_correlations()
    if "--figures" in sys.argv:
        write_figures(fig1)
