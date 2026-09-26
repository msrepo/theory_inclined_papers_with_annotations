#!/usr/bin/env python3
"""HSIC, the Hilbert-Schmidt independence criterion: every number in notes.md.

Checked here, in the order the notes use them:

  1. the five-point parabola: Pearson correlation exactly 0 although y = x^2;
  2. with a linear kernel HSIC is the squared covariance, and for vectors the
     squared Frobenius norm of the cross-covariance matrix (the sum of its
     squared singular values);
  3. adding the feature x^2 exposes the parabola: Cov(x^2, y) = 2.8, and the
     feature-kernel HSIC is the sum of the squared cross-covariances;
  4. the same five points with Gaussian kernels: the centred Gram matrices,
     their elementwise product, and an exact test over all 120 shuffles;
  5. the trace formula equals the three-term "pairs" expansion, and the MMD^2
     between the joint sample and the product of its marginals;
  6. five data sets of n = 200: Pearson r and permutation p-values under a
     linear and a Gaussian kernel;
  7. polynomial features: which cross-correlation lights up for each set;
  8. the bandwidth: a huge sigma collapses Gaussian HSIC onto linear HSIC,
     a tiny sigma makes every shuffle score the same, and a fine checkerboard
     is missed at the median bandwidth but found at a fifth of it;
  9. the null mean of the biased estimator shrinks like 1/n but is never 0;
 10. CKA: invariant to rotation and isotropic scaling, not to stretching, and
     dominated by high-variance directions.

With --figures it also regenerates the SVGs in ../figures/ from these same
computations, so every number drawn is a real one.

Standard library and numpy only.

Run:  python3 hsic.py            (checks)
      python3 hsic.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

np.set_printoptions(precision=4, suppress=True)

N_PERM = 999  # shuffles per permutation test; smallest reportable p is 1/1000


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ------------------------------------------------------------------ core

def centre(K):
    """H K H with H = I - 11^T/n: remove row and column means."""
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()


def hsic(K, L):
    """Biased estimator (1/n^2) tr(K H L H) = (1/n^2) sum_ij Kc_ij Lc_ij."""
    return float(np.sum(centre(K) * centre(L))) / len(K) ** 2


def as2d(X):
    X = np.asarray(X, float)
    return X.reshape(len(X), -1)


def sqdist(X):
    X = as2d(X)
    s = np.sum(X * X, 1)
    return np.maximum(s[:, None] + s[None, :] - 2 * X @ X.T, 0.0)


def linear(X):
    X = as2d(X)
    return X @ X.T


def rbf(X, sigma):
    return np.exp(-sqdist(X) / (2 * sigma ** 2))


def median_sigma(X):
    """Median heuristic: the median distance between two different samples."""
    d = np.sqrt(sqdist(X))
    return float(np.median(d[np.triu_indices(len(d), 1)]))


def perm_test(K, L, n_perm=N_PERM, seed=0):
    """Shuffle y (i.e. rows and columns of L together) and recompute HSIC.

    Permuting y commutes with centring, so the centred L can be permuted
    directly. p = (1 + #{shuffles >= observed}) / (1 + n_perm).
    """
    rng = np.random.default_rng(seed)
    Kc, Lc = centre(K), centre(L)
    n = len(K)
    obs = float(np.sum(Kc * Lc)) / n ** 2
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(n)
        null[b] = np.sum(Kc * Lc[np.ix_(p, p)]) / n ** 2
    pval = (1 + np.sum(null >= obs - 1e-9 * abs(obs))) / (1 + n_perm)  # relative tolerance: HSIC can be ~1e-15
    return obs, null, float(pval)


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def cka(K, L):
    return hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L))


KINDS = ("line", "parabola", "circle", "funnel", "independent")


def make(kind, n=200, seed=1):
    r = np.random.default_rng(seed)
    x = r.uniform(-1, 1, n)
    if kind == "line":
        y = x + 0.4 * r.standard_normal(n)
    elif kind == "parabola":
        y = x ** 2 + 0.1 * r.standard_normal(n)
    elif kind == "circle":
        t = r.uniform(0, 2 * np.pi, n)
        x = np.cos(t) + 0.08 * r.standard_normal(n)
        y = np.sin(t) + 0.08 * r.standard_normal(n)
    elif kind == "funnel":
        y = x * r.standard_normal(n)
    elif kind == "independent":
        y = r.uniform(-1, 1, n)
    elif kind == "checker":  # 5 x 5 board; keep points on squares whose indices sum to an even number
        pts = []
        while len(pts) < n:
            a, b = r.uniform(-1, 1, 2)
            i, j = min(int((a + 1) * 2.5), 4), min(int((b + 1) * 2.5), 4)
            if (i + j) % 2 == 0:
                pts.append((a, b))
        x, y = np.array(pts).T
    else:
        raise ValueError(kind)
    return x, y


def tests(x, y, seed=0):
    """Pearson r, and (HSIC, p) under a linear and a median-bandwidth kernel."""
    lin = perm_test(linear(x), linear(y), seed=seed)
    gau = perm_test(rbf(x, median_sigma(x)), rbf(y, median_sigma(y)), seed=seed)
    return pearson(x, y), lin, gau


# ------------------------------------------------------------------ checks

X5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
Y5 = X5 ** 2


def check_parabola_five():
    section("1. five-point parabola")
    print("x =", X5, " y =", Y5)
    cov = np.mean(X5 * Y5) - X5.mean() * Y5.mean()
    print("Cov(x, y) = mean(x^3) - mean(x) mean(y) =", cov, "  Pearson r =", pearson(X5, Y5))


def check_linear_kernel():
    section("2. linear kernel = squared covariance")
    print("five-point parabola, linear kernels: HSIC =", hsic(linear(X5), linear(Y5)))
    r = np.random.default_rng(4)
    x, y = r.standard_normal(50), r.standard_normal(50)
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    print("random 1-D pair: HSIC_lin =", f"{hsic(linear(x), linear(y)):.6f}", " Cov^2 =", f"{cov ** 2:.6f}")
    X = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    y = np.array([1.0, 3.0, 2.0])
    Xc, yc = X - X.mean(0), y - y.mean()
    C = Xc.T @ yc / len(X)
    print("vector example: X rows (1,0),(1,1),(0,2); y = (1,3,2)")
    print("  centred X =\n", Xc, "\n  centred y =", yc)
    print("  cross-covariance C = X~^T y~ / n =", C, " ||C||_F^2 =", C @ C)
    print("  HSIC_lin = (1/n^2) tr(K~ L~) =", hsic(linear(X), linear(y)))
    r = np.random.default_rng(5)
    A, B = r.standard_normal((40, 3)), r.standard_normal((40, 2))
    B = B + A[:, :2] @ np.array([[1.0, 0.0], [0.5, -0.3]])
    Cab = (A - A.mean(0)).T @ (B - B.mean(0)) / 40
    s = np.linalg.svd(Cab, compute_uv=False)
    print("  3-D vs 2-D random: HSIC_lin =", f"{hsic(linear(A), linear(B)):.6f}",
          " ||C||_F^2 =", f"{np.sum(Cab ** 2):.6f}", " sum of squared singular values =", f"{np.sum(s ** 2):.6f}",
          " top singular value (COCO) =", f"{s[0]:.4f}", f" s1^2 = {s[0] ** 2:.4f} ({s[0] ** 2 / np.sum(s ** 2):.1%} of HSIC)")


def check_features():
    section("3. features (x, x^2) against y")
    Phi = np.stack([X5, X5 ** 2], 1)
    Phic, yc = Phi - Phi.mean(0), Y5 - Y5.mean()
    C = Phic.T @ yc / 5
    print("Cov(x, y), Cov(x^2, y) =", C, "  sum of squares =", C @ C)
    K = np.outer(X5, X5) + np.outer(X5 ** 2, X5 ** 2)  # k(x,x') = x x' + x^2 x'^2
    print("HSIC with kernel k(x,x') = x x' + x^2 x'^2 :", hsic(K, linear(Y5)))


def check_gaussian_five():
    section("4. five points, Gaussian kernels")
    sx, sy = median_sigma(X5), median_sigma(Y5)
    print("median-heuristic bandwidths: sigma_x =", sx, " sigma_y =", sy)
    K, L = rbf(X5, sx), rbf(Y5, sy)
    Kc, Lc = centre(K), centre(L)
    print("K~ =\n", Kc.round(2), "\nL~ =\n", Lc.round(2), "\nK~ * L~ (elementwise) =\n", (Kc * Lc).round(2))
    S = np.sum(Kc * Lc)
    print("sum of all 25 cells =", round(S, 4), "  HSIC = sum / 25 =", round(S / 25, 4))
    vals = np.array([np.sum(Kc * Lc[np.ix_(p, p)]) / 25 for p in itertools.permutations(range(5))])
    obs = S / 25
    ge = int(np.sum(vals >= obs - 1e-12))
    print("all 120 orderings of y: mean HSIC =", round(vals.mean(), 4), " max =", round(vals.max(), 4))
    print("orderings scoring >= observed:", ge, " -> exact p =", f"{ge}/120 = {ge / 120:.4f}")
    same = sum(1 for p in itertools.permutations(range(5)) if np.array_equal(Y5[list(p)], Y5))
    print("orderings that give the same pairing as observed (ties in y):", same)
    best = 120
    for bx in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for by in (0.5, 1.0, 1.5, 2.0, 3.0, 4.5):
            Kb, Lb = centre(rbf(X5, bx)), centre(rbf(Y5, by))
            o = np.sum(Kb * Lb)
            best = min(best, sum(np.sum(Kb * Lb[np.ix_(p, p)]) >= o - 1e-12 for p in itertools.permutations(range(5))))
    print("best case over sigma_x in 0.5..3, sigma_y in 0.5..4.5: orderings >= observed =", best, "of 120")
    return Kc, Lc, obs, vals


def check_expansion():
    section("5. trace formula = pairs expansion = MMD^2(joint, product)")
    x, y = make("parabola", n=5, seed=3)
    K, L = rbf(x, median_sigma(x)), rbf(y, median_sigma(y))
    n = len(K)
    t1 = np.sum(K * L) / n ** 2
    t2 = K.sum() * L.sum() / n ** 4
    t3 = 2 * np.sum(K.sum(1) * L.sum(1)) / n ** 3
    print("trace formula      :", f"{hsic(K, L):.10f}")
    print("t1 + t2 - t3       :", f"{t1 + t2 - t3:.10f}", f"(t1={t1:.4f}, t2={t2:.4f}, t3={t3:.4f})")
    sx, sy = median_sigma(x), median_sigma(y)
    J = [(x[i], y[i]) for i in range(n)]
    P = [(x[a], y[b]) for a in range(n) for b in range(n)]

    def kap(u, v):
        return np.exp(-(u[0] - v[0]) ** 2 / (2 * sx ** 2)) * np.exp(-(u[1] - v[1]) ** 2 / (2 * sy ** 2))

    def mean_k(A, B):
        return np.mean([kap(a, b) for a in A for b in B])

    mmd2 = mean_k(J, J) - 2 * mean_k(J, P) + mean_k(P, P)
    print("MMD^2 with 5 joint points vs 25 product points:", f"{mmd2:.10f}")


def check_datasets():
    section("6. five data sets, n = 200")
    out = {}
    for kind in KINDS:
        x, y = make(kind)
        r, lin, gau = tests(x, y)
        out[kind] = (x, y, r, lin, gau)
        print(f"{kind:12s} Pearson r = {r:+.3f}   linear HSIC p = {lin[2]:.3f}   Gaussian HSIC p = {gau[2]:.3f}"
              f"   (Gaussian HSIC = {gau[0]:.5f}, null mean {gau[1].mean():.5f}, null max {gau[1].max():.5f})")
    return out


def check_polynomial():
    section("7. polynomial features: correlations between (x, x^2, x^3) and (y, y^2, y^3)")
    for kind in ("parabola", "circle", "funnel"):
        x, y = make(kind)
        F = np.stack([x, x ** 2, x ** 3], 1)
        G = np.stack([y, y ** 2, y ** 3], 1)
        R = np.corrcoef(F.T, G.T)[:3, 3:]
        print(f"{kind}: rows x, x^2, x^3; columns y, y^2, y^3\n", R.round(2))
    x, y = make("funnel", n=1_000_000, seed=2)
    print("funnel, n = 10^6: Corr(x^2, y^2) =", f"{np.corrcoef(x ** 2, y ** 2)[0, 1]:.3f}",
          " (population value sqrt(2/11) =", f"{np.sqrt(2 / 11):.3f})",
          " Corr(x, y^2) =", f"{np.corrcoef(x, y ** 2)[0, 1]:.3f}", " Corr(x^3, y^2) =", f"{np.corrcoef(x ** 3, y ** 2)[0, 1]:.3f}")


def check_bandwidth():
    section("8. bandwidth limits (parabola, n = 200)")
    x, y = make("parabola")
    sx, sy = median_sigma(x), median_sigma(y)
    print(f"median bandwidths sigma_x = {sx:.4f}, sigma_y = {sy:.4f}")
    hl = hsic(linear(x), linear(y))
    for m in (0.001, 0.01, 0.1, 1, 10, 100):
        K, L = rbf(x, m * sx), rbf(y, m * sy)
        obs, null, p = perm_test(K, L)
        z = (obs - null.mean()) / null.std() if null.std() > 0 else 0.0
        ratio = obs * (m * sx) ** 2 * (m * sy) ** 2 / hl
        print(f"sigma = {m:>6} x median: z = {z:7.2f}  p = {p:.3f}  null sd/mean = {null.std() / null.mean():.2e}"
              f"  HSIC*sx^2*sy^2 / HSIC_lin = {ratio:.4f}")
    _, _, pl = perm_test(linear(x), linear(y))
    print(f"linear-kernel p for the same data = {pl:.3f}")
    x, y = make("checker", n=150)
    print(f"5 x 5 checkerboard, n = 150: Pearson r = {pearson(x, y):+.3f}, median sigma_x = {median_sigma(x):.3f}")
    for m in (0.2, 1.0):
        obs, null, p = perm_test(rbf(x, m * median_sigma(x)), rbf(y, m * median_sigma(y)))
        print(f"  sigma = {m} x median: p = {p:.3f}  z = {(obs - null.mean()) / null.std():.2f}")


def check_null_mean():
    section("9. null mean of the biased estimator")
    for n in (50, 100, 200, 400):
        x, y = make("independent", n=n, seed=11)
        _, null, _ = perm_test(rbf(x, median_sigma(x)), rbf(y, median_sigma(y)), n_perm=200)
        print(f"n = {n:3d}: mean HSIC under shuffling = {null.mean():.5f}   n x mean = {n * null.mean():.3f}")


def check_cka():
    section("10. CKA")
    r = np.random.default_rng(7)
    n = 500
    X = r.standard_normal((n, 10)) * np.linspace(2, 0.5, 10)
    Q, _ = np.linalg.qr(r.standard_normal((10, 10)))
    D = np.diag(np.r_[np.ones(5), 0.1 * np.ones(5)])
    lin = linear
    print("linear CKA(X, X Q) rotation       =", f"{cka(lin(X), lin(X @ Q)):.4f}")
    print("linear CKA(X, 3 X) isotropic scale =", f"{cka(lin(X), lin(3 * X)):.4f}")
    print("linear CKA(X, X D) stretch         =", f"{cka(lin(X), lin(X @ D)):.4f}")
    print("linear CKA(X, tanh X)              =", f"{cka(lin(X), lin(np.tanh(X))):.4f}")
    for scale in (1.0, 3.0):
        s = r.standard_normal(n)
        A = np.column_stack([scale * s, r.standard_normal((n, 9))])
        B = np.column_stack([scale * s, r.standard_normal((n, 9))])
        print(f"one shared direction of ten, shared std {scale:.0f}: linear CKA = {cka(lin(A), lin(B)):.3f}")
    Xc = X - X.mean(0)
    print("linear CKA via features, ||X~^T Y~||_F^2 / (||X~^T X~||_F ||Y~^T Y~||_F):",
          f"{np.sum((Xc.T @ (Xc @ D)) ** 2) / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm((Xc @ D).T @ (Xc @ D))):.4f}")


# ------------------------------------------------------------------ figures

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:12px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:11px;fill:#57564f} .v{font-size:11px;fill:#1a1a19}
  .ax{stroke:#c3c2b7;stroke-width:0.75;fill:none}
  .box{fill:none;stroke:#c3c2b7;stroke-width:0.8}
  .dot{fill:#2f6fb5;opacity:.55}
  .yes{fill:#0f7a5a;font-weight:500} .no{fill:#8a8880}
  .pos{fill:#2f6fb5} .neg{fill:#c2571a} .bg{fill:#f4f3ee}
  .bar{fill:#8a8880;opacity:.55}
  .obs{stroke:#b42318;stroke-width:2} .obst{fill:#b42318;font-weight:500}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd,.v{fill:#eceae3} .ax,.box{stroke:#4a4844}
    .dot{fill:#7fb2e8;opacity:.6} .yes{fill:#4cc79a} .no{fill:#85837b}
    .pos{fill:#7fb2e8} .neg{fill:#f0a070} .bg{fill:#23262b}
    .bar{fill:#85837b} .obs{stroke:#ff7a6b} .obst{fill:#ff7a6b}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


def fmt_p(p):
    if p <= 0.001 + 1e-12:
        return "p = 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def fig_datasets(out, results):
    body, W = [], 140
    for k, kind in enumerate(KINDS):
        x, y, r, lin, gau = results[kind]
        x0, y0, s = 12 + k * W, 34, 116
        body.append(f'<text x="{x0 + s / 2}" y="22" class="hd" text-anchor="middle">{kind}</text>')
        body.append(f'<rect x="{x0}" y="{y0}" width="{s}" height="{s}" class="box"/>')
        lo_x, hi_x = x.min(), x.max()
        lo_y, hi_y = y.min(), y.max()
        for a, b in zip(x, y):
            px = x0 + 6 + (a - lo_x) / (hi_x - lo_x) * (s - 12)
            py = y0 + s - 6 - (b - lo_y) / (hi_y - lo_y) * (s - 12)
            body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.6" class="dot"/>')
        rows = [(f"Pearson r = {r:+.2f}", ""),
                (f"linear HSIC {fmt_p(lin[2])}", "yes" if lin[2] < 0.05 else "no"),
                (f"Gaussian HSIC {fmt_p(gau[2])}", "yes" if gau[2] < 0.05 else "no")]
        for j, (t, c) in enumerate(rows):
            body.append(f'<text x="{x0 + s / 2}" y="{y0 + s + 20 + 17 * j}" class="sm {c}" text-anchor="middle">{t}</text>')
    body.append('<text x="12" y="248" class="sm">n = 200 each, 999 shuffles per test. Green: p &lt; 0.05. The linear-kernel test is exactly a permutation test of Pearson r.</text>')
    desc = ("Five scatter plots of 200 points: a noisy line, a parabola, a circle, a funnel whose spread grows with |x|, "
            "and an independent cloud. Under each: Pearson r and permutation-test p-values for HSIC with a linear and a "
            "Gaussian kernel. " + "; ".join(
                f"{k}: r = {results[k][2]:+.2f}, linear p = {results[k][3][2]:.3f}, Gaussian p = {results[k][4][2]:.3f}"
                for k in KINDS) + ".")
    out.write_text(svg(712, 258, "Correlation versus HSIC on five data sets", desc, body))


def heat(body, M, x0, y0, c, vmax, labels_r, labels_c):
    n = len(M)
    for i in range(n):
        body.append(f'<text x="{x0 - 6}" y="{y0 + c * i + c / 2 + 4}" class="sm" text-anchor="end">{labels_r[i]}</text>')
        body.append(f'<text x="{x0 + c * i + c / 2}" y="{y0 - 6}" class="sm" text-anchor="middle">{labels_c[i]}</text>')
        for j in range(n):
            v = M[i, j]
            body.append(f'<rect x="{x0 + c * j}" y="{y0 + c * i}" width="{c}" height="{c}" class="bg"/>')
            body.append(f'<rect x="{x0 + c * j}" y="{y0 + c * i}" width="{c}" height="{c}" '
                        f'class="{"pos" if v >= 0 else "neg"}" opacity="{0.08 + 0.62 * min(1, abs(v) / vmax):.2f}"/>')
            t = f"{v:.2f}".replace("-0.00", "0.00")
            body.append(f'<text x="{x0 + c * j + c / 2}" y="{y0 + c * i + c / 2 + 4}" class="v" text-anchor="middle">{t}</text>')
    body.append(f'<rect x="{x0}" y="{y0}" width="{c * n}" height="{c * n}" class="box"/>')


def fig_agreement(out, Kc, Lc, obs, vals):
    body, c = [], 36
    lx = [f"{v:g}" for v in X5]
    ly = [f"{v:g}" for v in Y5]
    P = Kc * Lc
    specs = [(Kc, "K̃: unusually close in x?", lx, lx), (Lc, "L̃: unusually close in y?", ly, ly),
             (P, "K̃ ⊙ L̃: do they agree?", [f"{a}, {b}" for a, b in zip(lx, ly)], lx)]
    for k, (M, title, lr, lc) in enumerate(specs):
        x0 = 40 + k * 236 + (18 if k == 2 else 0)
        body.append(f'<text x="{x0 - 30}" y="20" class="hd">{title}</text>')
        heat(body, M, x0, 48, c, max(abs(M).max(), 1e-9), lr, lc)
    body.append('<text x="20" y="262" class="lab">Rows and columns are the five samples, labelled by their x or y value. '
                'Blue: above average, orange: below.</text>')
    S = P.sum()
    body.append(f'<text x="20" y="282" class="lab">Sum of all 25 products = {S:.3f}, so HSIC = {S:.3f} / 25 = {obs:.4f}. '
                f'Yet the 120 shuffles of y average {vals.mean():.4f}: five points are not enough.</text>')
    desc = (f"Three 5 by 5 heatmaps for x = (-2,-1,0,1,2) and y = x squared with Gaussian kernels at the median bandwidths. "
            f"Left: the centred x Gram matrix. Middle: the centred y Gram matrix. Right: their elementwise product, "
            f"which is mostly positive: pairs that are unusually close in x are also unusually close in y. The 25 products "
            f"sum to {S:.3f}, giving HSIC = {obs:.4f}; the average over all 120 shuffles of y is {vals.mean():.4f}.")
    out.write_text(svg(720, 292, "HSIC as the agreement of two centred Gram matrices", desc, body))


def fig_permutation(out, results):
    body = []
    for k, kind in enumerate(("parabola", "independent")):
        x, y, r, lin, gau = results[kind]
        obs, null, p = gau
        x0, y0, w, h = 40 + k * 350, 40, 290, 150
        lo, hi = 0.0, 1.08 * max(null.max(), obs)  # the biased estimator is never negative
        bins = np.linspace(null.min(), null.max(), 31)
        cnt, edges = np.histogram(null, bins)
        X = lambda v: x0 + (v - lo) / (hi - lo) * w  # noqa: E731
        for cc, a, b in zip(cnt, edges[:-1], edges[1:]):
            bh = cc / cnt.max() * (h - 20)
            body.append(f'<rect x="{X(a):.1f}" y="{y0 + h - bh:.1f}" width="{max(0.5, X(b) - X(a) - 0.6):.1f}" height="{bh:.1f}" class="bar"/>')
        body.append(f'<path d="M {x0},{y0 + h} H {x0 + w}" class="ax"/>')
        body.append(f'<line x1="{X(obs):.1f}" y1="{y0 + 4}" x2="{X(obs):.1f}" y2="{y0 + h}" class="obs"/>')
        anchor = "end" if X(obs) > x0 + w * 0.6 else "start"
        dx = -5 if anchor == "end" else 5
        body.append(f'<text x="{X(obs) + dx:.1f}" y="{y0 + 14}" class="sm obst" text-anchor="{anchor}">observed {obs:.4f}</text>')
        body.append(f'<text x="{x0}" y="24" class="hd">{kind}: {fmt_p(p)}</text>')
        body.append(f'<text x="{x0}" y="{y0 + h + 16}" class="sm">0</text>')
        body.append(f'<text x="{x0 + w}" y="{y0 + h + 16}" class="sm" text-anchor="end">{hi:.4f}</text>')
        body.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 16}" class="sm" text-anchor="middle">HSIC</text>')
        above = int(np.sum(null >= obs))
        body.append(f'<text x="{x0}" y="{y0 + h + 36}" class="lab">{above} of {len(null)} shuffles score at least the observed value</text>')
    body.append('<text x="40" y="250" class="sm">Grey: HSIC after shuffling y (999 shuffles, Gaussian kernels, median bandwidth, n = 200). Red: the unshuffled data.</text>')
    pp, pi = results["parabola"][4], results["independent"][4]
    desc = (f"Two histograms of HSIC under 999 random shuffles of y. For the parabola the observed value {pp[0]:.4f} lies far to "
            f"the right of every shuffle, p = {pp[2]:.3f}. For the independent cloud the observed value {pi[0]:.4f} sits inside "
            f"the shuffled values, p = {pi[2]:.3f}.")
    out.write_text(svg(700, 262, "The permutation test for HSIC", desc, body))


def write_figures(results, five):
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    fig_datasets(d / "datasets.svg", results)
    fig_agreement(d / "agreement.svg", *five)
    fig_permutation(d / "permutation.svg", results)
    print("\nfigures: wrote", ", ".join(sorted(p.name for p in d.glob("*.svg"))))


if __name__ == "__main__":
    check_parabola_five()
    check_linear_kernel()
    check_features()
    five = check_gaussian_five()
    check_expansion()
    results = check_datasets()
    check_polynomial()
    check_bandwidth()
    check_null_mean()
    check_cka()
    if "--figures" in sys.argv:
        write_figures(results, five)
