#!/usr/bin/env python3
"""Zhang, Tanno et al. 2023: the trace-regularised annotator model, checked by hand.

Everything in the paper's Section 2 can be checked at the level of one pixel, or of
a pile of independent pixels, without a U-Net.

Checked here, in the order the notes use them:

  1. Eq 3: the negative log-likelihood of Eqs 1-2 is the sum of per-annotator
     cross-entropies against A_hat @ p_hat;
  2. Eq 1 keeps each annotator's marginal only ("product of mixtures"). Unlike
     STAPLE's "mixture of products", it lets two perfect annotators disagree,
     by blaming the pixel for being 50/50. The two agree when p is one-hot;
  3. non-identifiability: relabelling the true classes (permuting the columns of
     A_hat and the entries of p_hat) leaves every product A_hat @ p_hat unchanged;
  4. one pixel, two classes, one label distribution q = (1 - t, t): every pair
     (A_hat, p_hat) with A_hat @ p_hat = q, the trace on that set, and where it is
     smallest with no constraint, under row dominance (the condition printed in
     Theorem 1) and under column dominance (the condition the proof uses);
  5. Lemma 1 as printed is false: A_eps = (1 - eps) q 1^T + eps I with p_hat = q is
     strictly row-dominant, reproduces q, and has a smaller trace than the truth;
  6. with column dominance added, no feasible point beats q_k + (L-1)/L, and the
     near-minimisers have p_hat close to e_k (random search, L = 3);
  7. under row dominance tr(A) >= 1, with equality exactly for the rank-one
     matrices u 1^T, i.e. annotators whose label ignores the truth;
  8. the trace and HSIC: with the classes weighted equally,
     tr(A) - 1 = L * tr(C) where C is the cross-covariance of one-hot labels,
     and HSIC with delta kernels is ||C||_F^2. HSIC does not change when the
     classes are relabelled; the trace does;
  9. the "mean trace = probability of a correct label" reading needs a uniform
     class prior; with small lesions it is far off;
 10. Section 2.5: the parameter and FLOP counts of Table 4 (BraTS, 192 x 192,
     L = 4, rank 1); a column-normalised rank-1 CM ignores p_hat entirely; the
     released code adds a learned multiple of the identity before normalising;
 11. a label-fusion toy (free per-pixel p_hat, one global CM per annotator,
     one careful and two sloppy annotators): cross-entropy alone prefers
     "every annotator perfect, the pixel uncertain"; with the warm-up and a
     large enough lambda, training passes through the right answer and then
     swaps every label, which is where the trace is lowest; without the
     warm-up it swaps straight away.

With --figures it also regenerates the SVGs in ../figures/ from these same
computations, so every number drawn is a real one.

Standard library and numpy only.

Run:  python3 trace_identifiability.py            (checks)
      python3 trace_identifiability.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def rowdom(A, strict=True):
    L = len(A)
    ok = (lambda a, b: a > b) if strict else (lambda a, b: a >= b)
    return all(ok(A[i, i], A[i, j]) for i in range(L) for j in range(L) if j != i)


def coldom(A, strict=True):
    L = len(A)
    ok = (lambda a, b: a > b) if strict else (lambda a, b: a >= b)
    return all(ok(A[j, j], A[i, j]) for i in range(L) for j in range(L) if j != i)


# ------------------------------------------------------------------ 1. Eq 3

def check_likelihood():
    section("1. Eq 3: NLL of Eqs 1-2 = sum of cross-entropies")
    rng = np.random.default_rng(0)
    L, R, P = 3, 2, 5                                  # classes, annotators, pixels
    p = rng.dirichlet(np.ones(L), size=P)              # p_hat at each pixel
    A = rng.dirichlet(np.ones(L), size=(R, P, L))      # A[r, pix, j, :] = column j
    A = np.transpose(A, (0, 1, 3, 2))                  # A[r, pix, i, j] = P(say i | true j)
    y = rng.integers(0, L, size=(R, P))                # observed noisy labels
    lik = 1.0
    for r in range(R):
        for n in range(P):
            lik *= sum(A[r, n, y[r, n], j] * p[n, j] for j in range(L))   # Eq 2 inside Eq 1
    ce = 0.0
    for r in range(R):
        q = np.einsum("nij,nj->ni", A[r], p)           # A_hat . p_hat, pixel by pixel
        ce += -np.log(q[np.arange(P), y[r]]).sum()
    print(f"  -log p(labels | x) = {-np.log(lik):.10f}")
    print(f"  sum_r CE(A_hat p_hat, y_r) = {ce:.10f}")


# ------------------------------------------------------------------ 2. product of mixtures

def check_product_of_mixtures():
    section("2. Eq 1 is a product of mixtures, STAPLE a mixture of products")
    I = np.eye(2)
    for p in (np.array([0.5, 0.5]), np.array([1.0, 0.0])):
        # two perfect annotators (A = I), one says background (0), the other lesion (1)
        pom = (I @ p)[0] * (I @ p)[1]                     # prod_r sum_y
        mop = sum(p[k] * I[0, k] * I[1, k] for k in range(2))   # sum_y prod_r
        print(f"  p_hat = {p}: product of mixtures {pom:.3f}   mixture of products {mop:.3f}")
    print("  -> Eq 1 lets two *perfect* annotators disagree, by making the pixel 50/50.")
    print("     STAPLE / Dawid-Skene give that event probability 0. With one-hot p_hat the two coincide.")


# ------------------------------------------------------------------ 3. relabelling

def check_permutation():
    section("3. relabelling the true classes leaves A_hat p_hat unchanged")
    rng = np.random.default_rng(1)
    A = rng.dirichlet(np.ones(3), size=3).T
    p = rng.dirichlet(np.ones(3))
    Pm = np.eye(3)[[2, 0, 1]]
    print(f"  A p          = {A @ p}")
    print(f"  (A P^T)(P p) = {(A @ Pm.T) @ (Pm @ p)}   (columns of A permuted, not rows)")
    print(f"  traces: {np.trace(A):.4f} vs {np.trace(A @ Pm.T):.4f}")


# ------------------------------------------------------------------ 4. the fibre, L = 2

def fibre_point(t, s, b):
    """On {A p = q}, q = (1-t, t): given s = p_hat(lesion) and b = A[lesion, bg] = P(say lesion | bg),
    the remaining free entry a = A[bg, lesion] = P(say bg | lesion) is fixed by the constraint."""
    a = 1 - b - (t - b) / s
    A = np.array([[1 - b, a], [b, 1 - a]])
    return a, A, 2 - a - b


def fibre_grid(t, n=1201):
    s = np.linspace(1e-4, 1, n)[None, :]
    b = np.linspace(0, 1, n)[:, None]
    a = 1 - b - (t - b) / s
    tr = 1 + (t - b) / s                              # = 2 - a - b
    feas = (a >= 0) & (a <= 1)
    row = feas & (a + b < 1)                          # for L = 2, row dominance <=> a + b < 1
    col = feas & (a < 0.5) & (b < 0.5)                # column dominance <=> a, b < 1/2
    return s, b, a, tr, feas, row, col


def constrained_minima(t):
    s, b, a, tr, feas, row, col = fibre_grid(t)
    out = {}
    for name, m in (("none", feas), ("row dominance", row), ("column dominance", col)):
        i = np.unravel_index(np.argmin(np.where(m, tr, np.inf)), tr.shape)
        out[name] = (tr[i], s[0, i[1]], b[i[0], 0], a[i])
    return out


def check_fibre():
    section("4. one pixel, L = 2: every (A_hat, p_hat) that reproduces q = (1 - t, t)")
    t = 0.8
    print(f"  t = {t}: the pixel is lesion and the annotator marks it with probability {t}")
    print("  on the fibre, tr(A_hat) = 1 + (t - b)/s: every level set is a line through (s, b) = (0, t)")
    pts = {"copy the label (A = I, p = q)": (t, 0.0),
           "truth (p = e_lesion, unused column uniform)": (1.0, 0.5),
           "annotator ignores the truth (A = q 1^T)": (0.5, t),
           "labels swapped (A anti-diagonal)": (1 - t, 1.0)}
    for name, (s, b) in pts.items():
        a, A, tr = fibre_point(t, s, b)
        print(f"  {name:44s} s={s:.2f} b={b:.2f} a={a:.2f}  A p = {A @ np.array([1 - s, s])}  tr = {tr:.3f}")
    for tt in (0.8, 0.6, 0.4):
        mins = constrained_minima(tt)
        print(f"  t = {tt}: " + "; ".join(f"{k}: min tr {v[0]:.3f} at s = {v[1]:.2f}, b = {v[2]:.2f}"
                                           for k, v in mins.items()))
    print("  -> no constraint: the label swap (tr 0). Row dominance: the rank-one line b = t (tr -> 1,")
    print("     s arbitrary). Column dominance: s = 1 when t > 1/2 (the truth, tr = t + 1/2), s -> 0 when t < 1/2.")


# ------------------------------------------------------------------ 5. Lemma 1 counterexample

def check_counterexample():
    section("5. Lemma 1 as printed: a strictly row-dominant A_hat with a smaller trace than the truth")
    for q in (np.array([0.2, 0.8]), np.array([0.15, 0.7, 0.15])):
        L, k = len(q), int(np.argmax(q))
        trA = q[k] + (L - 1) / L                     # true CM: column k = q, other columns uniform
        for eps in (0.1,):
            A = (1 - eps) * np.outer(q, np.ones(L)) + eps * np.eye(L)
            print(f"  q = {q}, eps = {eps}: A_hat =\n{A}")
            print(f"    columns sum to {A.sum(0)}, A_hat q = {A @ q}, strictly row-dominant: {rowdom(A)}, "
                  f"column-dominant: {coldom(A)}")
            print(f"    tr(A_hat) = {np.trace(A):.3f} < tr(A) = {trA:.3f}, and p_hat = q, not e_{k}")


# ------------------------------------------------------------------ 6. column dominance restores the bound

def check_column_bound(n=300000):
    section("6. with column dominance the bound q_k + (L-1)/L holds (random search, L = 3)")
    rng = np.random.default_rng(2)
    q = np.array([0.15, 0.7, 0.15]); L = 3; k = 1
    bound = q[k] + (L - 1) / L
    P = rng.dirichlet(np.ones(L), size=n)
    C = np.transpose(rng.dirichlet(np.full(L, 0.7), size=(n, L)), (0, 2, 1))   # C[m, :, j] = column j
    best = {"row only": (np.inf, None), "row + column": (np.inf, None)}
    feasible = 0
    for m in range(n):
        p, A = P[m], C[m]
        js = int(np.argmax(p))
        rest = sum(p[j] * A[:, j] for j in range(L) if j != js)
        col = (q - rest) / p[js]                     # solve A p = q for the largest-weight column
        if (col < 0).any():
            continue
        A = A.copy(); A[:, js] = col
        feasible += 1
        tr = np.trace(A)
        if rowdom(A):
            if tr < best["row only"][0]:
                best["row only"] = (tr, p)
            if coldom(A) and tr < best["row + column"][0]:
                best["row + column"] = (tr, p)
    print(f"  {feasible} feasible samples; bound q_k + (L-1)/L = {bound:.4f}")
    for name, (tr, p) in best.items():
        print(f"  {name:13s}: smallest trace found {tr:.4f} with p_hat = {p}")


# ------------------------------------------------------------------ 7. the trace floor

def check_trace_floor():
    section("7. row dominance => tr >= 1, with equality exactly at rank one (label independent of truth)")
    rng = np.random.default_rng(3)
    trs = []
    for _ in range(20000):
        A = rng.dirichlet(np.ones(3), size=3).T
        if rowdom(A):
            trs.append(np.trace(A))
    print(f"  {len(trs)} random row-dominant 3x3 column-stochastic matrices: min trace {min(trs):.4f}")
    u = np.array([0.2, 0.5, 0.3]); A = np.outer(u, np.ones(3))
    print(f"  A = u 1^T with u = {u}: trace {np.trace(A):.3f}; A e_1 = {A[:, 0]}, A e_3 = {A[:, 2]} (same for every truth)")


# ------------------------------------------------------------------ 8. trace and HSIC

def cross_cov(A, pi):
    """C_ij = P(y_tilde = i, y = j) - P(y_tilde = i) P(y = j): the delta-kernel cross-covariance."""
    return A * pi[None, :] - np.outer(A @ pi, pi)


def check_trace_hsic():
    section("8. the trace and HSIC are two functions of the same cross-covariance matrix")
    rng = np.random.default_rng(4)
    L = 4
    pi = np.full(L, 1 / L)
    for _ in range(2):
        A = rng.dirichlet(np.ones(L), size=L).T
        C = cross_cov(A, pi)
        print(f"  tr(A) - 1 = {np.trace(A) - 1:.6f}   L tr(C) = {L * np.trace(C):.6f}   HSIC = ||C||_F^2 = {(C ** 2).sum():.6f}")
    A = np.array([[0.9, 0.2], [0.1, 0.8]]); pi2 = np.array([0.5, 0.5])
    F = A[:, ::-1]                                   # the same annotator with the true classes swapped
    for name, M in (("faithful", A), ("swapped ", F)):
        C = cross_cov(M, pi2)
        print(f"  {name} annotator {M.tolist()}: trace {np.trace(M):.2f}, HSIC {(C ** 2).sum():.4f}")
    print("  -> HSIC cannot tell agreement from anti-agreement; the trace can, and prefers anti-agreement.")


# ------------------------------------------------------------------ 9. mean trace vs accuracy

def check_mean_trace():
    section("9. 'mean trace = probability of a correct label' needs a uniform class prior")
    fpr, fnr = 0.02, 0.40
    A = np.array([[1 - fpr, fnr], [fpr, 1 - fnr]])
    for pi_fg in (0.5, 0.02):
        acc = (1 - pi_fg) * (1 - fpr) + pi_fg * (1 - fnr)
        print(f"  lesion fraction {pi_fg:.2f}: accuracy {acc:.4f}, tr(A)/L = {np.trace(A) / 2:.4f}")


# ------------------------------------------------------------------ 10. low rank

def check_low_rank():
    section("10. Section 2.5: low-rank CMs")
    W = H = 192; L = 4; l = 1
    print(f"  BraTS 192x192, L = 4: full CM entries W H L^2 = {W * H * L * L:,}; rank 1: 2 W H L l = {2 * W * H * L * l:,}")
    print(f"  FLOPs W H (2L - 1) L = {W * H * (2 * L - 1) * L:,}; low rank W H (4L(l - 0.25) - l) = {int(W * H * (4 * L * (l - 0.25) - l)):,}")
    print(f"  per pixel, B2^T p then B1 (.): l(2L - 1) + L(2l - 1) = {l * (2 * L - 1) + L * (2 * l - 1)} = 4L(l - 0.25) - l = {4 * L * (l - 0.25) - l:g}")
    rng = np.random.default_rng(5)
    u, v = rng.random(3), rng.random(3)
    M = np.outer(u, v); M = M / M.sum(0, keepdims=True)
    print(f"  column-normalised u v^T:\n{M}\n  M e_1 = {M[:, 0]}, M e_3 = {M[:, 2]}: identical, so the product ignores p_hat")
    M2 = np.outer(u, v) + 0.7 * np.eye(3); M2 = M2 / M2.sum(0, keepdims=True)
    print(f"  released code, normalise(u v^T + s I) with s = 0.7:\n{M2}")


# ------------------------------------------------------------------ 11. the label-fusion toy

S = 48
FUSION_RATES = [(0.02, 0.02), (0.40, 0.05), (0.40, 0.05)]    # (FPR, FNR): one careful, two sloppy
FUSION_NAMES = ["careful", "sloppy 1", "sloppy 2"]


def fusion_truth():
    yy, xx = np.mgrid[0:S, 0:S]
    blob = ((xx - 17) ** 2 + (yy - 19) ** 2 <= 100) | ((xx - 33) ** 2 + (yy - 31) ** 2 <= 64)
    return blob.astype(int).reshape(-1)


def fusion_labels(y, rates, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for fpr, fnr in rates:
        pf = np.where(y == 1, 1 - fnr, fpr)
        out.append((rng.random(y.size) < pf).astype(int))
    return np.array(out)


def dice(pred, y):
    return 2 * (pred * y).sum() / (pred.sum() + y.sum())


def fusion_ce(lab, s, a10, a11):
    q = a10[:, None] * (1 - s)[None, :] + a11[:, None] * s[None, :]
    return -np.log(np.where(lab == 1, q, 1 - q) + 1e-12).mean(1).sum()


def train_fusion(y, lab, lam, init, steps=3000, lr=0.05, warm=300, seed=0, every=0):
    """Free logit per pixel, one global 2x2 CM per annotator; Adam on sum_r mean CE + lam * sum_r tr.

    init: 'warm-up'  -- identity-like start, then lam = -1 (maximise the trace) for `warm` steps
          'identity' -- identity-like start, lam from the first step
          'uniform'  -- every CM entry equal at the start, as a softplus head gives
    every > 0 also records (step, Dice, sum of traces, CM error) every `every` steps.
    """
    R, N = lab.shape
    rates = np.array(FUSION_RATES)
    rs = np.random.default_rng(seed)
    z = rs.normal(0, 0.01, N)
    U = rs.normal(0, 0.01, (R, 2))                    # logits of a10 = P(say lesion | bg), a11 = P(say lesion | lesion)
    if init in ("warm-up", "identity"):
        U[:, 0] -= 3.0; U[:, 1] += 3.0
    params = [z, U]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    hist = []

    def state():
        s = 1 / (1 + np.exp(-z))
        a10 = 1 / (1 + np.exp(-U[:, 0])); a11 = 1 / (1 + np.exp(-U[:, 1]))
        return s, a10, a11

    for it in range(steps):
        lt = -1.0 if (init == "warm-up" and it < warm) else lam
        s, a10, a11 = state()
        if every and it % every == 0:
            est = np.stack([a10, 1 - a11], 1)
            hist.append((it, dice((s > 0.5).astype(int), y), float((1 - a10 + a11).sum()),
                         float(np.abs(est - rates).mean())))
        q = a10[:, None] * (1 - s) + a11[:, None] * s
        dq = np.where(lab == 1, -1 / (q + 1e-12), 1 / (1 - q + 1e-12)) / N
        da10 = (dq * (1 - s)).sum(1) - lt              # tr = (1 - a10) + a11
        da11 = (dq * s).sum(1) + lt
        ds = (dq * (a11 - a10)[:, None]).sum(0)
        grads = [ds * s * (1 - s), np.stack([da10 * a10 * (1 - a10), da11 * a11 * (1 - a11)], 1)]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** (it + 1))) / (np.sqrt(v[i] / (1 - 0.999 ** (it + 1))) + 1e-8)
    s, a10, a11 = state()
    est = np.stack([a10, 1 - a11], 1)
    if every:
        hist.append((steps, dice((s > 0.5).astype(int), y), float((1 - a10 + a11).sum()), float(np.abs(est - rates).mean())))
    return dict(dice=dice((s > 0.5).astype(int), y), cm_err=float(np.abs(est - rates).mean()),
                soft=float(np.abs(s - y).mean()), est=est, s=s, ce=fusion_ce(lab, s, a10, a11),
                tr=float((1 - a10 + a11).sum()), hist=hist)


def fusion_references(y, lab):
    maj = (lab.mean(0) > 0.5).astype(int)
    ll1 = np.zeros(y.size); ll0 = np.zeros(y.size)
    for (fpr, fnr), l in zip(FUSION_RATES, lab):
        ll1 += np.where(l == 1, np.log(1 - fnr), np.log(fnr))
        ll0 += np.where(l == 1, np.log(fpr), np.log(1 - fpr))
    wv = (ll1 > ll0).astype(int)
    return dice(maj, y), dice(wv, y)


def check_fusion():
    section("11. label-fusion toy: one careful and two sloppy annotators, 48x48 pixels")
    y = fusion_truth(); lab = fusion_labels(y, FUSION_RATES)
    maj, wv = fusion_references(y, lab)
    print(f"  lesion fraction {y.mean():.3f}; annotators (FPR, FNR) = {FUSION_RATES}")
    print(f"  majority vote Dice {maj:.3f}; reliability-weighted vote with the true CMs Dice {wv:.3f}")
    # cross-entropy of two explanations of the same labels
    frac = lab.mean(0)
    ce_naive = fusion_ce(lab, frac, np.zeros(3) + 1e-9, np.ones(3) - 1e-9)
    t10 = np.array([r[0] for r in FUSION_RATES]); t11 = np.array([1 - r[1] for r in FUSION_RATES])
    ce_true = fusion_ce(lab, y.astype(float), t10, t11)
    print(f"  CE of 'annotators perfect, p_hat = vote fraction' = {ce_naive:.4f}")
    print(f"  CE of 'true CMs, p_hat = one-hot truth'           = {ce_true:.4f}   (higher: CE alone prefers the first)")
    marks = (1000, 2000, 3000, 4000, 5000)
    print("  Dice along training (warm-up = identity start, then 300 steps maximising the trace):")
    for lam in (0.0, 0.3, 0.55, 0.7):
        r = train_fusion(y, lab, lam, "warm-up", steps=5000, every=500)
        at = {h[0]: h for h in r["hist"]}
        print(f"   lambda {lam:.2f}: " + "  ".join(f"step {k}: {at[k][1]:.3f}" for k in marks)
              + f"   | CM error at 3000: {at[3000][3]:.3f}, sum tr at 3000: {at[3000][2]:.2f}")
    for init in ("identity", "uniform"):
        r = train_fusion(y, lab, 0.55, init, steps=3000, every=500)
        at = {h[0]: h for h in r["hist"]}
        print(f"   lambda 0.55, {init} start, no warm-up: " + "  ".join(f"step {k}: {at[k][1]:.3f}" for k in (1000, 2000, 3000)))
    r = train_fusion(y, lab, 0.55, "warm-up", steps=3000)
    print(f"   warm-up, lambda 0.55, stopped at step 3000: (FPR, FNR) estimates {r['est'].round(2).tolist()}")
    return y, lab, maj, wv


# ------------------------------------------------------------------ figures

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:10.5px;fill:#57564f} .v{font-size:10.5px;fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .box{fill:none;stroke:#c9c7bf;stroke-width:1}
  .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .infeas{fill:#e9e7e0} .band{fill:#ffffff} .row{fill:#2a78d6;opacity:.08} .col{fill:#2a78d6;opacity:.16}
  .iso{stroke:#8a8880;stroke-width:0.8;fill:none;stroke-dasharray:3 3}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s4{stroke:#eda100}
  .f1{fill:#2a78d6} .f2{fill:#eb6834} .f3{fill:#1baf7a} .f4{fill:#eda100}
  .ln{stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
  .dash{stroke-width:1.2;fill:none;stroke-dasharray:5 3}
  .ring{stroke:#fdfdfc;stroke-width:2}
  .pt{fill:#1a1a19} .apex{fill:none;stroke:#1a1a19;stroke-width:1.2}
  .tn{fill:#f4f3ee} .tp{fill:#6f6d66} .fp{fill:#eb6834} .fn{fill:#2a78d6}
  .ref{stroke:#8a8880;stroke-width:1;stroke-dasharray:4 3;fill:none}
  .node{fill:#ffffff;stroke:#b9b7ae;stroke-width:1} .node2{fill:#f2f0ea;stroke:#b9b7ae;stroke-width:1}
  .arr{stroke:#6f6d66;stroke-width:1.2;fill:none} .arrh{fill:#6f6d66}
  .acc{stroke:#6f6d66;stroke-width:1.2;fill:none;stroke-dasharray:4 3}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax{stroke:#85837b} .box{stroke:#4a4844} .gd{stroke:#33312e}
    .infeas{fill:#2a2c30} .band{fill:#1b1d21} .row{fill:#3987e5;opacity:.10} .col{fill:#3987e5;opacity:.20}
    .iso{stroke:#85837b}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70} .f4{fill:#c98500}
    .ring{stroke:#161615} .pt{fill:#eceae3} .apex{stroke:#eceae3}
    .tn{fill:#1f2124} .tp{fill:#a3a19a} .fp{fill:#d95926} .fn{fill:#3987e5}
    .ref{stroke:#85837b} .node{fill:#1b1d21;stroke:#4a4844} .node2{fill:#23262b;stroke:#4a4844}
    .arr{stroke:#a3a6ae} .arrh{fill:#a3a6ae} .acc{stroke:#a3a6ae}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


def fig_fibre(out, t=0.8):
    x0, y0, w, h = 60, 34, 380, 380
    X = lambda s: x0 + s * w          # noqa: E731
    Y = lambda b: y0 + (1 - b) * h    # noqa: E731
    body = [f'<text x="{x0}" y="20" class="hd">One lesion pixel, labelled "lesion" with probability t = {t}: every (Â, p̂) with Â p̂ = q</text>']
    ss = np.linspace(1e-3, 1, 400)
    lo = np.clip((t - ss) / (1 - ss + 1e-12), 0, 1); hi = np.clip(t / (1 - ss + 1e-12), 0, 1)
    top = " ".join(f"L {X(s):.1f},{Y(v):.1f}" for s, v in zip(ss, hi))
    bot = " ".join(f"L {X(s):.1f},{Y(v):.1f}" for s, v in zip(ss[::-1], lo[::-1]))
    body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" class="infeas"/>')
    band = f"M {X(0):.1f},{Y(t):.1f} {top} {bot} Z"
    body.append(f'<path d="{band}" class="band"/>')
    # row-dominant part (b < t) and column-dominant part (b < 1/2) of the band
    body.append(f'<clipPath id="bandclip"><path d="{band}"/></clipPath>')
    body.append(f'<rect x="{x0}" y="{Y(t):.1f}" width="{w}" height="{Y(0) - Y(t):.1f}" class="row" clip-path="url(#bandclip)"/>')
    body.append(f'<rect x="{x0}" y="{Y(0.5):.1f}" width="{w}" height="{Y(0) - Y(0.5):.1f}" class="col" clip-path="url(#bandclip)"/>')
    # iso-trace lines through the apex (0, t): b = t - (c - 1) s
    for c in (0.0, 0.5, 1.0, 1.3, 1.6, 2.0):
        sl = np.linspace(0, 1, 200); bl = t - (c - 1) * sl
        seg = [(s_, b_) for s_, b_ in zip(sl, bl) if 0 <= b_ <= 1]
        if len(seg) < 2:
            continue
        d = "M " + " L ".join(f"{X(s_):.1f},{Y(b_):.1f}" for s_, b_ in seg)
        body.append(f'<path d="{d}" class="iso" clip-path="url(#bandclip)"/>')
    body.append(f'<text x="{X(1) + 4:.1f}" y="{Y(t - 0.6) + 4:.1f}" class="sm">tr 1.6</text>')
    # the rank-one line b = t (trace exactly 1)
    body.append(f'<path d="M {X(0):.1f},{Y(t):.1f} H {X(1):.1f}" class="ln s4"/>')
    body.append(f'<text x="{X(1) + 4:.1f}" y="{Y(t) + 4:.1f}" class="sm">tr 1: Â = q 1ᵀ</text>')
    body.append(f'<path d="M {X(0):.1f},{Y(0.5):.1f} H {X(1):.1f}" class="dash s1"/>')
    body.append(f'<text x="{X(1) + 10:.1f}" y="{Y(0.5) + 4:.1f}" class="sm">b = ½</text>')
    body.append(f'<circle cx="{X(0):.1f}" cy="{Y(t):.1f}" r="4" class="apex"/>')
    # special points
    body.append(f'<circle cx="{X(1):.1f}" cy="{Y(0.5):.1f}" r="5" class="f3 ring"/>')
    body.append(f'<text x="{X(1) - 8:.1f}" y="{Y(0.5) - 9:.1f}" class="v" text-anchor="end">T  truth, tr {t + 0.5:g}</text>')
    body.append(f'<circle cx="{X(1 - t):.1f}" cy="{Y(1):.1f}" r="5" class="f2 ring"/>')
    body.append(f'<text x="{X(1 - t) + 9:.1f}" y="{Y(1) + 15:.1f}" class="v">F  labels swapped, tr 0</text>')
    body.append(f'<circle cx="{X(t):.1f}" cy="{Y(0):.1f}" r="4.5" class="pt ring"/>')
    body.append(f'<text x="{X(t) - 6:.1f}" y="{Y(0) - 8:.1f}" class="sm" text-anchor="end">N  Â = I, p̂ = q, tr 2</text>')
    # axes
    body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" class="box"/>')
    for v in (0, 0.25, 0.5, 0.75, 1):
        body.append(f'<text x="{X(v):.1f}" y="{y0 + h + 15}" class="sm" text-anchor="middle">{v:g}</text>')
        body.append(f'<text x="{x0 - 6}" y="{Y(v) + 4:.1f}" class="sm" text-anchor="end">{v:g}</text>')
    body.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 32}" class="lab" text-anchor="middle">s = p̂(lesion), what the segmentation network says</text>')
    body.append(f'<text x="{x0 - 38}" y="{y0 + h / 2}" class="lab" text-anchor="middle" transform="rotate(-90 {x0 - 38} {y0 + h / 2})">b = Â(says lesion | background)</text>')
    # right-hand legend
    lx = x0 + w + 100
    legend = [("hd", "Where the trace is smallest"),
              ("sm", "Grey: impossible (the third entry of Â"), ("sm", "would leave [0, 1]). White band: all"),
              ("sm", "pairs that fit the data exactly."),
              ("sm", ""),
              ("sm", "Dashed: equal-trace lines (0, 0.5, 1.3,"), ("sm", "1.6, 2). All pass through the circle"),
              ("sm", "(0, t), so lowering the trace swings"), ("sm", "the line anticlockwise about it."),
              ("sm", ""),
              ("f2", "No constraint: F, the label swap."),
              ("f4", "Row dominance (Theorem 1's words):"), ("", "slide to the yellow line, tr → 1,"),
              ("", "any s. The segmentation is free."),
              ("f3", "Column dominance (what the proof"), ("", "uses): T, the truth, s = 1."),
              ("sm", ""),
              ("sm", "Light blue: row-dominant (b &lt; t)."), ("sm", "Darker: also column-dominant (b &lt; ½).")]
    for j, (cls, txt) in enumerate(legend):
        if not txt:
            continue
        yy = y0 + 10 + 17 * j
        if cls in ("f2", "f3", "f4"):
            body.append(f'<circle cx="{lx - 9}" cy="{yy - 4}" r="4" class="{cls}"/>')
        body.append(f'<text x="{lx}" y="{yy}" class="{"hd" if cls == "hd" else "sm"}">{txt}</text>')
    desc = (f"The set of all confusion matrices and segmentation outputs that reproduce one pixel's label distribution "
            f"q = (0.2, 0.8), drawn in the plane of s, the predicted lesion probability, and b, the estimated false-positive "
            f"rate. The feasible set is a curved band starting at (0, {t}). Lines of equal trace are straight lines through "
            f"that point. The label-swapped solution F at (0.2, 1) has trace 0; the rank-one line b = {t} has trace 1; the "
            f"true solution T at (1, 0.5) has trace 1.3; copying the label, N at ({t}, 0), has trace 2. Minimising the trace "
            f"over the whole band gives F, over its row-dominant part gives the rank-one line, and over its column-dominant "
            f"part gives T.")
    out.write_text(svg(760, 460, "Where the trace is minimised on one pixel", desc, body))


def rle_rects(mask, x0, y0, c, cls, value=None):
    """One rect per horizontal run of True pixels (of equal `value`, when given), in a crisp-edged group."""
    out = ['<g shape-rendering="crispEdges">']
    for i in range(mask.shape[0]):
        j = 0
        while j < mask.shape[1]:
            if mask[i, j]:
                k = j
                while k < mask.shape[1] and mask[i, k] and (value is None or value[i, k] == value[i, j]):
                    k += 1
                op = "" if value is None else f' opacity="{0.12 + 0.83 * min(1.0, value[i, j] / 0.9):.2f}"'
                out.append(f'<rect x="{x0 + j * c:.2f}" y="{y0 + i * c:.2f}" width="{(k - j) * c:.2f}" height="{c:.2f}" class="{cls}"{op}/>')
                j = k
            else:
                j += 1
    out.append("</g>")
    return out


def fig_spatial(out):
    n = 40
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.sqrt((xx - 19.5) ** 2 + (yy - 19.5) ** 2) - 11
    y = (d <= 0)
    rng = np.random.default_rng(6)
    ann = [
        ("over-segmenter", np.where(y, 0.02, np.where(d <= 3, 0.9, 0.02))),
        ("under-segmenter", np.where(y, np.where(d > -3, 0.8, 0.02), 0.02)),
        ("careless everywhere", np.full(y.shape, 0.2)),
    ]
    c = 3.2
    body = ['<text x="20" y="20" class="hd">What a pixel-wise confusion matrix looks like</text>']
    for k, (name, perr) in enumerate(ann):
        x0 = 20 + k * 168
        lab_err = rng.random(y.shape) < perr
        lab = np.where(lab_err, ~y, y)
        body.append(f'<text x="{x0 + n * c / 2:.1f}" y="44" class="lab" text-anchor="middle">{name}</text>')
        y1 = 52
        body.append(f'<rect x="{x0}" y="{y1}" width="{n * c}" height="{n * c}" class="tn"/>')
        body += rle_rects(lab & y, x0, y1, c, "tp")
        body += rle_rects(lab & ~y, x0, y1, c, "fp")
        body += rle_rects(~lab & y, x0, y1, c, "fn")
        body.append(f'<rect x="{x0}" y="{y1}" width="{n * c}" height="{n * c}" class="box"/>')
        y2 = y1 + n * c + 14
        body.append(f'<rect x="{x0}" y="{y2}" width="{n * c}" height="{n * c}" class="tn"/>')
        # probability of a wrong label at each pixel, coloured by the kind of error it would be
        body += rle_rects(~y, x0, y2, c, "fp", value=perr)
        body += rle_rects(y, x0, y2, c, "fn", value=perr)
        body.append(f'<rect x="{x0}" y="{y2}" width="{n * c}" height="{n * c}" class="box"/>')
    yb = 52 + 2 * n * c + 34
    caption = ["Top: one label from each simulated annotator. Grey: lesion marked correctly.",
               "Orange: background marked as lesion. Blue: lesion missed.",
               "Bottom: the column of each pixel's true CM that its true class uses, shown as",
               "P(wrong label | true class, x); stronger colour means a likelier error. The first two",
               "annotators err only in a ring at the boundary, which a single CM per annotator cannot",
               "express. A CM that depends on the image and the pixel, Â⁽ʳ⁾(x), can."]
    for j, line in enumerate(caption):
        body.append(f'<text x="20" y="{yb + 15 * j}" class="sm">{line}</text>')
    desc = ("Three columns, one per simulated annotator of a disc-shaped lesion. Top row: a sampled label map, with false "
            "positives in orange and false negatives in blue. The over-segmenter has an orange ring just outside the disc, the "
            "under-segmenter a blue ring just inside it, and the careless annotator scattered orange and blue pixels "
            "everywhere. Bottom row: the probability of an error at each pixel for that pixel's true class, which is where "
            "the per-pixel confusion matrix departs from the identity.")
    out.write_text(svg(524, yb + 15 * len(caption) + 8, "Pixel-wise confusion matrices of three annotators", desc, body))


def arrow(body, x1, y1, x2, y2, cls="arr"):
    body.append(f'<path d="M {x1},{y1} L {x2},{y2}" class="{cls}"/>')
    ang = np.arctan2(y2 - y1, x2 - x1)
    ax, ay = x2 - 7 * np.cos(ang - 0.4), y2 - 7 * np.sin(ang - 0.4)
    bx, by = x2 - 7 * np.cos(ang + 0.4), y2 - 7 * np.sin(ang + 0.4)
    body.append(f'<path d="M {x2:.1f},{y2:.1f} L {ax:.1f},{ay:.1f} L {bx:.1f},{by:.1f} Z" class="arrh"/>')


def box(body, x, y, w, h, lines, cls="node"):
    body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" class="{cls}"/>')
    for j, (c, t) in enumerate(lines):
        body.append(f'<text x="{x + w / 2}" y="{y + 18 + 15 * j}" class="{c}" text-anchor="middle">{t}</text>')


def fig_model(out):
    body = [f'<text x="20" y="22" class="hd">The model of Fig. 1, one pixel at a time</text>']
    box(body, 20, 110, 92, 52, [("lab", "image x"), ("sm", "W × H × C")])
    box(body, 150, 100, 130, 72, [("lab", "shared U-Net"), ("sm", "4 stages, 32–256 ch,"), ("sm", "instance norm")], "node2")
    box(body, 330, 40, 170, 62, [("lab", "segmentation head θ"), ("sm", "1×1 conv, softmax"), ("sm", "p̂(x): L numbers per pixel")])
    box(body, 330, 168, 170, 74, [("lab", "annotator heads φ, r = 1..R"), ("sm", "2 conv blocks, 1×1 conv,"), ("sm", "softplus, columns normalised"), ("sm", "Â⁽ʳ⁾(x): L × L per pixel")])
    box(body, 552, 104, 150, 62, [("lab", "per pixel"), ("v", "p̂⁽ʳ⁾ = Â⁽ʳ⁾ · p̂"), ("sm", "annotator r's guess")])
    box(body, 748, 60, 150, 62, [("lab", "cross-entropy"), ("sm", "against annotator r's"), ("sm", "label ỹ⁽ʳ⁾ (Eq 3)")])
    box(body, 748, 150, 150, 62, [("lab", "+ λ · tr Â⁽ʳ⁾"), ("sm", "push every modelled"), ("sm", "annotator to be unreliable")])
    arrow(body, 112, 136, 148, 136)
    arrow(body, 280, 124, 328, 78)
    arrow(body, 280, 148, 328, 198)
    arrow(body, 500, 78, 550, 124)
    arrow(body, 500, 200, 550, 150)
    arrow(body, 702, 128, 746, 96)
    arrow(body, 500, 226, 746, 190)
    body.append('<path d="M 500,56 H 560" class="acc"/>')
    body.append('<text x="566" y="52" class="sm">test time: output p̂(x),</text>')
    body.append('<text x="566" y="65" class="sm">annotator heads unused</text>')
    body.append('<text x="20" y="276" class="sm">Â⁽ʳ⁾ is column-stochastic: entry (i, j) is P(annotator r says i | true class j, image x, pixel). The product is a mixture of its columns,</text>')
    body.append('<text x="20" y="291" class="sm">weighted by p̂. Only the products are compared with data, so Â⁽ʳ⁾ and p̂ are not separately determined; the trace term chooses between them.</text>')
    desc = ("Block diagram. The image goes through a shared U-Net. A segmentation head outputs p-hat, a probability vector "
            "over L classes at every pixel. R annotator heads output a column-stochastic L by L matrix at every pixel. For "
            "each annotator the per-pixel product A-hat times p-hat is compared with that annotator's label by cross-entropy, "
            "and lambda times the trace of A-hat is added. At test time only p-hat is used.")
    out.write_text(svg(920, 300, "The two-headed model and its loss", desc, body))


def fig_dynamics(out, y, lab, maj, wv, steps=8000, every=100):
    runs = [(0.0, "s1", "λ = 0"), (0.3, "s2", "λ = 0.3"), (0.55, "s3", "λ = 0.55"), (0.7, "s4", "λ = 0.7")]
    res = {lam: train_fusion(y, lab, lam, "warm-up", steps=steps, every=every)["hist"] for lam, *_ in runs}
    x0, y0, w, h = 58, 44, 300, 190
    X = lambda it: x0 + it / steps * w   # noqa: E731
    body = [f'<text x="20" y="20" class="hd">Label-fusion toy: training passes the right answer on its way to swapping the labels</text>']
    for p, (k, lo, hi, ylab) in enumerate(((1, 0, 1, "Dice against the truth"), (2, 0, 6, "sum of the three traces"))):
        px = x0 + p * (w + 76)
        Yv = lambda v: y0 + h - (v - lo) / (hi - lo) * h   # noqa: E731
        body.append(f'<rect x="{px}" y="{y0}" width="{w}" height="{h}" class="box"/>')
        for v in np.linspace(lo, hi, 5 if k == 1 else 4):
            body.append(f'<path d="M {px},{Yv(v):.1f} H {px + w}" class="gd"/>')
            body.append(f'<text x="{px - 5}" y="{Yv(v) + 4:.1f}" class="sm" text-anchor="end">{v:.3g}</text>')
        for it in range(0, steps + 1, 2000):
            body.append(f'<text x="{px + it / steps * w:.1f}" y="{y0 + h + 14}" class="sm" text-anchor="middle">{it}</text>')
        body.append(f'<text x="{px + w / 2}" y="{y0 + h + 30}" class="lab" text-anchor="middle">training step</text>')
        body.append(f'<text x="{px}" y="{y0 - 8}" class="lab">{ylab}</text>')
        if k == 1:
            for v, name, dy in ((maj, "majority vote", 13), (wv, "weighted vote with the true CMs", -5)):
                body.append(f'<path d="M {px},{Yv(v):.1f} H {px + w}" class="ref"/>')
                body.append(f'<text x="{px + w - 4}" y="{Yv(v) + dy:.1f}" class="sm" text-anchor="end">{name} {v:.3f}</text>')
        for lam, lc, _ in runs:
            d = "M " + " L ".join(f"{px + hh[0] / steps * w:.1f},{Yv(hh[k]):.1f}" for hh in res[lam])
            body.append(f'<path d="{d}" class="ln {lc}"/>')
        if k == 1:
            for lam, lc, name in runs[2:]:
                best = max(res[lam], key=lambda t_: t_[1])
                bx, by = px + best[0] / steps * w, Yv(best[1])
                body.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4" class="{lc.replace("s", "f")} ring"/>')
    def story(lam):
        hh = res[lam]
        best = max(hh, key=lambda t_: t_[1])
        crash = next((t_[0] for t_ in hh if t_[0] > best[0] and t_[1] < 0.1), None)
        if crash is None:
            return f"stays at {hh[-1][1]:.3f}"
        return f"peak {best[1]:.3f} at step {best[0]}, below 0.1 by step {crash}"
    for j, (lam, lc, name) in enumerate(runs):
        xx, yy = x0 + 376 * (j % 2), y0 + h + 50 + 17 * (j // 2)
        body.append(f'<path d="M {xx},{yy - 4} h 22" class="ln {lc}"/><text x="{xx + 28}" y="{yy}" class="sm">{name}: {story(lam)}</text>')
    body.append(f'<text x="{x0}" y="{y0 + h + 94}" class="sm">48 × 48 pixels; (FPR, FNR) = (0.02, 0.02), (0.40, 0.05), (0.40, 0.05). Free p̂ per pixel, one CM per annotator, Adam (lr 0.05);</text>')
    body.append(f'<text x="{x0}" y="{y0 + h + 109}" class="sm">every run starts at the identity and maximises the trace for 300 steps, as the paper describes. Dice 0 = every label swapped.</text>')

    def summary(lam):
        hh = res[lam]
        best = max(hh, key=lambda t: t[1])
        return f"lambda {lam}: best Dice {best[1]:.2f} at step {best[0]}, final Dice {hh[-1][1]:.2f}"
    desc = ("Two line charts over 8000 training steps for four trace weights, each started with the paper's warm-up. "
            + "; ".join(summary(l) for l, *_ in runs) + f". Reference lines: majority vote {maj:.3f}, weighted vote "
            f"with the true confusion matrices {wv:.3f}. The right panel shows the summed trace, which falls to 0 when "
            f"the labels swap.")
    out.write_text(svg(760, y0 + h + 120, "Dice and trace during training for four trace weights", desc, body))
    return res


def write_figures(y, lab, maj, wv):
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    fig_fibre(d / "fibre.svg")
    fig_spatial(d / "spatial-cm.svg")
    fig_model(d / "model.svg")
    res = fig_dynamics(d / "fusion-dynamics.svg", y, lab, maj, wv)
    print("\n  fusion runs for the figure (warm-up, 8000 steps):")
    for lam, hh in res.items():
        best = max(hh, key=lambda t: t[1])
        crash = next((t[0] for t in hh if t[0] > best[0] and t[1] < 0.1), None)
        print(f"   lambda {lam:.2f}: best Dice {best[1]:.3f} at step {best[0]}; first step with Dice < 0.1 after it: {crash}; final {hh[-1][1]:.3f}")
    print("\nfigures: wrote", ", ".join(sorted(p.name for p in d.glob("*.svg"))))


if __name__ == "__main__":
    check_likelihood()
    check_product_of_mixtures()
    check_permutation()
    check_fibre()
    check_counterexample()
    check_column_bound()
    check_trace_floor()
    check_trace_hsic()
    check_mean_trace()
    check_low_rank()
    y, lab, maj, wv = check_fusion()
    if "--figures" in sys.argv:
        write_figures(y, lab, maj, wv)
