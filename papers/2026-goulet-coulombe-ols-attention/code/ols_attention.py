#!/usr/bin/env python3
"""OLS as an attention mechanism: every identity in the paper, checked, plus the
places where the wording is looser than the algebra.

Paper: Goulet Coulombe, "Ordinary Least Squares as an Attention Mechanism"
(arXiv:2504.09663, NeurIPS 2026).

Checked here:

  1. Eq 2 = Eq 6: X_test beta_hat equals F_test F_train' y, with F = X U Lambda^(-1/2),
     F_train' F_train = I, and F_test' F_test != I (the paper says both);
  2. Eq 7: each prediction is sum_i <F_j, F_i> y_i, the weights sum to ONE when X has
     an intercept, are SIGNED (affine, not convex), and their squared length is the
     leverage x_j'(X'X)^(-1) x_j -- so the weights are large exactly when extrapolating;
  3. Eq 8: raw inner products <x_j, x_i> double-count correlated predictors;
  4. Eq 17 and Appendix A.1: for ONE realised y the minimiser Omega is NOT unique
     (a P(P-1)-dimensional affine set); requiring it for EVERY y gives S^(-1);
  5. "Gradient descent on a linear attention module converges to OLS predictions": true
     for the predictions, false for the embedding -- GD from zero lands on a rank-one
     Omega, not on S^(-1);
  6. Eqs 19-23: PCR, ridge and OLS are the same module with D = diag of (1/l_i for the
     top L, 1/(l_i + lambda), 1/l_i), and the best rank-L approximation of S^(-1)
     (Eckart-Young) keeps the WRONG directions;
  7. Section 3.3 / A.6: a stack of linear attention layers is still X times a matrix,
     and the OLS fit is a fixed point, H^2 = H;
  8. A.4: masking the weights does not remove look-ahead, because Omega = S^(-1) is
     estimated on the whole sample;
  9. Table 1: the column means and win counts behind the sentence "lands in the
     neighbourhood of Random Forest".

Standard library and numpy only.

Run:  python3 ols_attention.py             (checks, prints every number in the notes)
      python3 ols_attention.py --figures   (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)


def hdr(s):
    print("\n" + s + "\n" + "-" * len(s))


def ols_pieces(X):
    S = X.T @ X
    lam, U = np.linalg.eigh(S)
    return S, lam, U


# ------------------------------------------------------------------ 1. Eq 2 = Eq 6
def check_identity():
    hdr("1. Eq 2 equals Eq 6, and the factor scores are orthonormal")
    N, J, P = 200, 5, 4
    X = rng.standard_normal((N, P)) @ rng.standard_normal((P, P))   # correlated columns
    Xt = rng.standard_normal((J, P)) @ rng.standard_normal((P, P))
    y = X @ rng.standard_normal(P) + rng.standard_normal(N)
    S, lam, U = ols_pieces(X)
    beta = np.linalg.solve(S, X.T @ y)
    W = U @ np.diag(lam ** -0.5)                     # Eq 10: W = U Lambda^(-1/2)
    F, Ft = X @ W, Xt @ W
    p_ols, p_att = Xt @ beta, Ft @ F.T @ y
    print(f"   max |Eq 2 - Eq 6|                      {np.abs(p_ols - p_att).max():.2e}")
    print(f"   max |F'F - I|                           {np.abs(F.T @ F - np.eye(P)).max():.2e}")
    print(f"   max |Ft'Ft - I| (test, NOT orthonormal) {np.abs(Ft.T @ Ft - np.eye(P)).max():.2f}")
    # rotation freedom: any orthogonal Q gives the same inner products
    Q, _ = np.linalg.qr(rng.standard_normal((P, P)))
    print(f"   same predictions after a random rotation of F: "
          f"{np.abs((Ft @ Q) @ (F @ Q).T @ y - p_att).max():.2e}  (the embedding is a rotation class)")
    # a different square root of S^-1 (the symmetric one) also works
    M = U @ np.diag(lam ** -0.5) @ U.T
    print(f"   symmetric square root S^(-1/2) also works:      "
          f"{np.abs((Xt @ M) @ (X @ M).T @ y - p_att).max():.2e}")
    return X, y


# ------------------------------------------------------------------ 2. weights
def check_weights():
    hdr("2. The weights omega_ji: sum to 1, can be negative, squared length = leverage")
    N, P = 120, 5
    x = rng.uniform(-1, 1, N)
    X = np.vander(x, P, increasing=True)             # 1, x, x^2, x^3, x^4: has an intercept
    S, lam, U = ols_pieces(X)
    Sinv = np.linalg.inv(S)
    for xj in (0.0, 0.8, 1.3):
        xt = np.vander([xj], P, increasing=True)[0]
        w = X @ Sinv @ xt                            # omega_ji = x_j' S^-1 x_i
        print(f"   test x={xj:4.1f}: sum w = {w.sum():.6f}   min w = {w.min():+.4f}   "
              f"negative share = {np.mean(w < 0):.2f}   sum w^2 = {w @ w:.4f}   "
              f"x'S^-1 x = {xt @ Sinv @ xt:.4f}")
    # on the training points the weight matrix is the hat matrix
    H = X @ Sinv @ X.T
    print(f"   F F' = H: symmetric {np.allclose(H, H.T)}, idempotent {np.allclose(H @ H, H)}, "
          f"trace {np.trace(H):.4f} (= P = {P}), rows sum to {H.sum(1).min():.4f}..{H.sum(1).max():.4f}")
    print(f"   share of negative entries of H on training points: {np.mean(H < -1e-12):.2f}")
    # the weight for a *noise* outcome: prediction variance = sigma^2 * sum w^2
    sig, reps = 1.0, 40000
    xt = np.vander([1.3], P, increasing=True)[0]
    w = X @ Sinv @ xt
    sim = (rng.standard_normal((reps, N)) * sig) @ w
    print(f"   Var(prediction at x=1.3) by simulation {sim.var():.3f}  vs  sigma^2 sum w^2 = {w @ w:.3f}")
    return x


# ------------------------------------------------------------------ 3. Eq 8
def check_double_count():
    hdr("3. Why whiten first: raw inner products count a duplicated predictor twice")
    N = 400
    a = rng.standard_normal(N)
    X1 = np.c_[a, rng.standard_normal(N)]
    X2 = np.c_[a, a + 0.05 * rng.standard_normal(N), X1[:, 1]]      # column 1 duplicated
    t1, t2 = np.array([1.0, 0.0]), np.array([1.0, 1.0, 0.0])
    s1, s2 = X1 @ t1, X2 @ t2
    # raw similarity of the test point with itself: ||x||^2 jumps from 1 to 2
    print(f"   raw  <x,x>:  without duplicate {t1 @ t1:.1f}   with duplicate {t2 @ t2:.1f}")
    for nm, X, t in (("without", X1, t1), ("with   ", X2, t2)):
        Sinv = np.linalg.inv(X.T @ X)
        print(f"   whitened <F,F>, {nm} duplicate: {t @ Sinv @ t:.5f}   (x'S^-1 x)")


# ------------------------------------------------------------------ 4. uniqueness
def check_uniqueness():
    hdr("4. Eq 17 / A.1: one y does not identify Omega; every y does")
    N, P = 80, 4
    X = rng.standard_normal((N, P))
    S = X.T @ X
    Sinv = np.linalg.inv(S)

    def sse(Om, y):
        return float(np.sum((y - X @ Om @ X.T @ y) ** 2))

    y = X @ rng.standard_normal(P) + rng.standard_normal(N)
    z = X.T @ y
    # Delta with Delta z = 0: project a random matrix off z
    D = rng.standard_normal((P, P))
    D = D - np.outer(D @ z, z) / (z @ z)
    print(f"   |Delta z| = {np.linalg.norm(D @ z):.1e},  Delta != 0: |Delta|_F = {np.linalg.norm(D):.2f}")
    print(f"   SSE at S^-1          {sse(Sinv, y):.6f}")
    print(f"   SSE at S^-1 + Delta  {sse(Sinv + D, y):.6f}   (same minimum, different Omega)")
    print(f"   dimension of the minimiser set for one y: P(P-1) = {P * (P - 1)}")
    # every y: stack the conditions Omega z_k = S^-1 z_k for many y_k
    for K in (1, 2, P - 1, P, 3 * P):
        Z = X.T @ rng.standard_normal((N, K))                        # z_1..z_K as columns
        # unknown Omega (P*P); equations Omega Z = S^-1 Z; rank of the map Omega -> Omega Z
        A = np.kron(Z.T, np.eye(P))
        rank = np.linalg.matrix_rank(A)
        print(f"   K={K:2d} targets: identified directions {rank:2d} of {P * P:2d}"
              f"  -> free dimension {P * P - rank}")


# ------------------------------------------------------------------ 5. GD
def gd_trace(X, y, steps=4000, factored=False):
    N, P = X.shape
    S = X.T @ X
    Sinv = np.linalg.inv(S)
    z = X.T @ y
    lr = 0.5 / (np.linalg.eigvalsh(S).max() * (z @ z))
    if factored:
        A = 0.3 * rng.standard_normal((P, P))
        B = 0.3 * rng.standard_normal((P, P))
        lr *= 2.0
    else:
        Om = np.zeros((P, P))
    rows = []
    Xt = rng.standard_normal((6, P))
    target = Xt @ Sinv @ z
    for t in range(steps + 1):
        if factored:
            Om = A @ B.T
        r = y - X @ Om @ z
        if t % 20 == 0:
            rows.append((t, float(r @ r), float(np.linalg.norm(Xt @ Om @ z - target)),
                         float(np.linalg.norm(Om - Sinv) / np.linalg.norm(Sinv))))
        G = -2 * X.T @ np.outer(r, z)                                # dL/dOmega
        if factored:
            A, B = A - lr * G @ B, B - lr * G.T @ A
        else:
            Om = Om - lr * G
    return rows, Om, Sinv, z


def check_gd():
    hdr("5. Gradient descent on a linear attention module: predictions yes, embedding no")
    N, P = 100, 4
    X = rng.standard_normal((N, P)) @ np.diag([2.0, 1.0, 0.7, 0.4])
    y = X @ rng.standard_normal(P) + rng.standard_normal(N)
    out = {}
    for fac in (False, True):
        rows, Om, Sinv, z = gd_trace(X, y, steps=60000 if fac else 6000, factored=fac)
        t, loss, perr, oerr = rows[-1]
        nm = "factored W_Q W_K'" if fac else "full Omega from 0   "
        rk = np.linalg.matrix_rank(Om, tol=1e-6)
        print(f"   {nm}: loss {loss:.4f} (OLS {float(np.sum((y - X @ np.linalg.solve(X.T@X, X.T@y))**2)):.4f}), "
              f"test-prediction error {perr:.1e}, relative |Omega - S^-1| {oerr:.2f}, rank(Omega)={rk}")
        out[fac] = rows
    return out


# ------------------------------------------------------------------ 6. PCR / ridge
def check_pcr_ridge():
    hdr("6. OLS, ridge and PCR are one module with different D (Eq 23); Eckart-Young keeps the wrong ones")
    N, P, L, lamb = 150, 6, 3, 5.0
    X = rng.standard_normal((N, P)) @ np.diag([3, 2.5, 2, 1, 0.6, 0.3])
    y = X @ rng.standard_normal(P) + rng.standard_normal(N)
    Xt = rng.standard_normal((4, P)) @ np.diag([3, 2.5, 2, 1, 0.6, 0.3])
    S, lam, U = ols_pieces(X)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    z = X.T @ y
    Ds = {"OLS": 1 / lam, "ridge": 1 / (lam + lamb), "PCR": np.r_[1 / lam[:L], np.zeros(P - L)]}
    refs = {
        "OLS": Xt @ np.linalg.solve(S, z),
        "ridge": Xt @ np.linalg.solve(S + lamb * np.eye(P), z),
        "PCR": (Xt @ U[:, :L]) @ np.linalg.solve(np.diag(lam[:L]), (X @ U[:, :L]).T @ y),
    }
    for k, D in Ds.items():
        Om = U @ np.diag(D) @ U.T
        print(f"   {k:5s}: |attention - estimator| = {np.abs(Xt @ Om @ z - refs[k]).max():.2e}")
    # Eckart-Young on S^-1: keep largest eigenvalues of S^-1 = the smallest lam
    Dey = np.zeros(P)
    Dey[-L:] = 1 / lam[-L:]
    Om_ey = U @ np.diag(Dey) @ U.T
    Om_pc = U @ np.diag(Ds["PCR"]) @ U.T

    def fit_sse(Om):
        return float(np.sum((y - X @ Om @ z) ** 2))

    print(f"   rank-{L} embeddings, in-sample SSE:  OLS {fit_sse(U @ np.diag(Ds['OLS']) @ U.T):.1f}   "
          f"PCR (top-{L} directions) {fit_sse(Om_pc):.1f}   "
          f"Eckart-Young truncation of S^-1 (bottom-{L} directions) {fit_sse(Om_ey):.1f}")
    print(f"   eigenvalues of X'X: {np.round(lam, 1)}")


# ------------------------------------------------------------------ 7. stacking
def check_stacking():
    hdr("7. Linear stacks collapse; the OLS fit is a fixed point")
    N, P = 90, 4
    X = rng.standard_normal((N, P))
    y = rng.standard_normal(N)
    H = X @ np.linalg.solve(X.T @ X, X.T)
    print(f"   |H^2 - H| = {np.abs(H @ H - H).max():.1e}")
    X1 = X
    R = np.eye(P)
    for l in range(3):
        Om = 0.1 * rng.standard_normal((P, P))
        Xn = X1 + X1 @ Om @ (X1.T @ X1) * 0.001        # residual linear attention block
        # Xn = X1 R_l with R_l = I + Om (X1'X1) * 0.001
        R = R @ (np.eye(P) + 0.001 * Om @ (X1.T @ X1))
        X1 = Xn
    print(f"   X_3 = X R_eff exactly: {np.abs(X1 - X @ R).max():.1e}")
    # regressing y on X_3 gives the same fitted values as regressing on X
    fit = lambda A: A @ np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"   |fit(X_3) - fit(X)| = {np.abs(fit(X1) - fit(X)).max():.1e}  (same column space)")


# ------------------------------------------------------------------ 8. masking
def check_masking():
    hdr("8. Appendix A.4: masking the weights does not remove look-ahead")
    T = 400
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.9 * y[t - 1] + rng.standard_normal()
    ylag, yy = y[:-1], y[1:]
    t0 = 150
    a_full = lambda i, j: ylag[i] * ylag[j] / (ylag @ ylag)         # a_{t tau}, full sample S
    a_trunc = lambda i, j: ylag[i] * ylag[j] / (ylag[:t0] @ ylag[:t0])
    # fitted value at time i=100 with mask 1(tau<=i) but S from the full vs the first t0 obs
    i = 100
    f_full = sum(a_full(i, j) * yy[j] for j in range(i + 1))
    f_trunc = sum(a_trunc(i, j) * yy[j] for j in range(i + 1))
    print(f"   fitted value at t=100, mask applied, S from whole sample: {f_full:+.4f}")
    print(f"   same, S estimated only on observations up to t=150:       {f_trunc:+.4f}")
    print(f"   -> the masked fit still moves when later data change S (relative change "
          f"{abs(f_full / f_trunc - 1):.1%}); see the paper's own March-1974 argument")


# ------------------------------------------------------------------ 9. Table 1
TABLE1 = {
    "California": (0.597, 0.777, 0.763, 0.764, 0.738, 0.766),
    "Yacht":      (0.562, 0.979, 0.970, 0.988, 0.958, 0.989),
    "Energy":     (0.913, 0.996, 0.994, 0.994, 0.995, 0.995),
    "Concrete":   (0.624, 0.892, 0.895, 0.898, 0.837, 0.907),
    "Airfoil":    (0.497, 0.910, 0.917, 0.924, 0.757, 0.927),
    "Abalone":    (0.260, 0.322, 0.323, 0.324, 0.312, 0.329),
    "Kin8nm":     (0.430, 0.664, 0.912, 0.914, 0.851, 0.920),
    "Protein":    (0.285, 0.521, 0.486, 0.426, 0.361, 0.455),
}
COLS = ("OLS", "RF", "MLP", "FT-T", "AttReg", "RegBlock")


def check_table1():
    hdr("9. Table 1 as printed: what 'in the neighbourhood of Random Forest' means")
    T = np.array(list(TABLE1.values()))
    print("   column means:  " + "  ".join(f"{c} {m:.3f}" for c, m in zip(COLS, T.mean(0))))
    d_rf = T[:, 4] - T[:, 1]
    print(f"   AttReg - RF per dataset: {np.round(d_rf, 3)}   (ahead on {int((d_rf > 0).sum())} of 8)")
    print(f"   AttReg - RF mean {d_rf.mean():+.3f}, median {np.median(d_rf):+.3f}")
    d_ft = T[:, 5] - T[:, 3]
    print(f"   RegBlock - FT-T per dataset: {np.round(d_ft, 3)}   mean {d_ft.mean():+.4f}")
    gap = T[:, 1] - T[:, 0]
    pay = T[:, 5] - T[:, 0]
    print(f"   OLS->RF gap vs OLS->RegBlock gain, correlation {np.corrcoef(gap, pay)[0, 1]:.3f}")
    best = [COLS[int(np.argmax(r))] for r in T]
    print("   best method per dataset: " + ", ".join(f"{k} {b}" for k, b in zip(TABLE1, best)))


# ------------------------------------------------------------------ figures
CSS = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s4{stroke:#eda100}
  .f1{fill:#2a78d6} .f2{fill:#eb6834} .f3{fill:#1baf7a}
  .ln{stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
  .dash{stroke-width:1.3;fill:none;stroke-dasharray:5 3}
  @media (prefers-color-scheme: dark){
    .lab{fill:#b6b4ab} .hd{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s4{stroke:#c98500}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70}
  }
</style>"""


class Panel:
    """A little axes box mapping data coordinates to SVG coordinates."""

    def __init__(self, x0, y0, w, h, xl, yl):
        self.x0, self.y0, self.w, self.h, self.xl, self.yl = x0, y0, w, h, xl, yl

    def X(self, v):
        return self.x0 + (v - self.xl[0]) / (self.xl[1] - self.xl[0]) * self.w

    def Y(self, v):
        return self.y0 + self.h - (v - self.yl[0]) / (self.yl[1] - self.yl[0]) * self.h

    def frame(self, xt, yt, xlab, ylab, fmt="{:g}"):
        s = []
        for v in yt:
            s.append(f'<path class="gd" d="M{self.x0:.1f} {self.Y(v):.1f}H{self.x0 + self.w:.1f}"/>')
            s.append(f'<text class="lab" x="{self.x0 - 6:.1f}" y="{self.Y(v) + 4:.1f}" text-anchor="end">{fmt.format(v)}</text>')
        for v in xt:
            s.append(f'<text class="lab" x="{self.X(v):.1f}" y="{self.y0 + self.h + 16:.1f}" text-anchor="middle">{v:g}</text>')
        s.append(f'<path class="ax" d="M{self.x0:.1f} {self.y0:.1f}V{self.y0 + self.h:.1f}H{self.x0 + self.w:.1f}"/>')
        s.append(f'<text class="lab" x="{self.x0 + self.w / 2:.1f}" y="{self.y0 + self.h + 34:.1f}" text-anchor="middle">{xlab}</text>')
        s.append(f'<text class="lab" x="{self.x0 - 40:.1f}" y="{self.y0 - 10:.1f}">{ylab}</text>')
        return "\n".join(s)

    def line(self, xs, ys, cls):
        pts = " ".join(f"{self.X(a):.1f},{self.Y(b):.1f}" for a, b in zip(xs, ys))
        return f'<polyline class="ln {cls}" points="{pts}"/>'


def fig_weights(d):
    N, P = 400, 6
    x = np.sort(rng.uniform(-1, 1, N))
    X = np.vander(x, P, increasing=True)
    Sinv = np.linalg.inv(X.T @ X)
    pan = Panel(60, 56, 780, 230, (-1, 1), (-0.04, 0.12))
    s = [f'<svg viewBox="0 0 900 366" xmlns="http://www.w3.org/2000/svg" role="img">',
         "<title>OLS attention weights are signed and wiggly</title>",
         "<desc>Weights omega_i = x_j' (X'X)^-1 x_i that a degree-5 polynomial OLS gives each of 400 training points, for three test points x_j = 0, 0.8 and 1.05. At x_j = 0 the weights are a broad low bump with small negative side lobes; at 0.8 the bump moves toward the edge; at 1.05, just outside the training range, weights are several times larger, alternate in sign across the whole interval, and rise steeply at the right edge, nearest the test point, while also putting weight on points far from it. The dashed grey line is the smooth Gaussian-kernel weight for x_j = 0 with bandwidth 0.15, which is never negative.</desc>",
         CSS, '<text class="hd" x="20" y="26">Who does each test point attend to?  Degree-5 polynomial OLS, N = 400</text>']
    s.append(pan.frame([-1, -0.5, 0, 0.5, 1], [-0.04, 0, 0.04, 0.08, 0.12], "training input x_i",
                       "weight omega_ji", "{:+.2f}"))
    for xj, cls in ((0.0, "s1"), (0.8, "s3"), (1.05, "s2")):
        w = X @ Sinv @ np.vander([xj], P, increasing=True)[0]
        s.append(pan.line(x, w, cls))
    g = np.exp(-0.5 * (x / 0.15) ** 2)
    g = g / g.sum()
    s.append(f'<polyline class="dash ax" points="' + " ".join(f"{pan.X(a):.1f},{pan.Y(b):.1f}" for a, b in zip(x, g)) + '"/>')
    for i, (lab, cls) in enumerate((("test x = 0", "s1"), ("test x = 0.8", "s3"),
                                    ("test x = 1.05 (just outside)", "s2"))):
        s.append(f'<path class="ln {cls}" d="M{70 + i * 190} 346h22"/><text class="lab" x="{96 + i * 190}" y="350">{lab}</text>')
    s.append('<path class="dash ax" d="M650 346h22"/><text class="lab" x="676" y="350">Gaussian kernel, x = 0 (never negative)</text>')
    s.append("</svg>")
    (d / "ols-weights.svg").write_text("\n".join(s))


def fig_gd(d, rows):
    rows = {False: [r for r in rows[False] if r[0] <= 600]}
    t = np.array([r[0] for r in rows[False]])
    pan = Panel(60, 56, 780, 230, (0, t.max()), (-4, 0.5))
    s = ['<svg viewBox="0 0 900 366" xmlns="http://www.w3.org/2000/svg" role="img">',
         "<title>Gradient descent reaches OLS predictions but not the OLS embedding</title>",
         "<desc>Log10 error against gradient-descent step (first 600 steps), for one fixed outcome vector. The error of the test-set predictions relative to OLS falls steadily toward zero (blue). The relative distance of the learned embedding Omega from the inverse Gram matrix S^-1 stays near one (orange) because gradient descent from zero stops at a rank-one embedding that merely agrees with S^-1 along the single direction X'y.</desc>",
         CSS, '<text class="hd" x="20" y="26">Gradient descent on a linear attention module, one y, Omega started at zero</text>']
    s.append(pan.frame([0, t.max() / 2, t.max()], [-4, -3, -2, -1, 0], "gradient step",
                       "log10 of error (full-Omega GD)", "{:.0f}"))
    perr = np.log10(np.maximum([r[2] for r in rows[False]], 1e-4))
    oerr = np.log10(np.maximum([r[3] for r in rows[False]], 1e-4))
    s.append(pan.line(t, perr, "s1"))
    s.append(pan.line(t, oerr, "s2"))
    s.append('<path class="ln s1" d="M70 346h22"/><text class="lab" x="96" y="350">test-prediction error vs OLS</text>')
    s.append('<path class="ln s2" d="M330 346h22"/><text class="lab" x="356" y="350">relative distance of Omega from S^-1</text>')
    s.append("</svg>")
    (d / "gd-embedding.svg").write_text("\n".join(s))



def fig_eig(d):
    """What S^-1 = U Lambda^-1 U' does: rotate to the eigen-axes, then divide each axis by its spread."""
    r = np.random.default_rng(3)
    th = np.deg2rad(32.0)
    U = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])    # columns u1, u2
    sd = np.array([1.9, 0.6])                                              # sqrt(lambda_k / N)
    Z = r.standard_normal((70, 2))
    Z[0] = (1.3, 0.0); Z[1] = (0.0, 1.3)                                    # pair marked i and j below
    Xp = (Z * sd) @ U.T                                                    # training cloud, original axes
    k = 24.0
    cx = [150, 450, 750]; cy = 170
    def pt(P, c):
        return c + k * P[0], cy - k * P[1]
    s = ['<svg viewBox="0 0 900 392" xmlns="http://www.w3.org/2000/svg" role="img">',
         "<title>What the eigendecomposition of the inverse Gram matrix does to the data</title>",
         "<desc>Three panels of the same two-dimensional training cloud. Left: the cloud is an elongated tilted ellipse in the original (x1, x2) axes; its long axis u1 has spread sqrt(lambda_1) and its short axis u2 has spread sqrt(lambda_2). Middle: after rotating coordinates with U transpose the ellipse lies along the horizontal axis and each coordinate is the data's position along an eigen-direction. Right: after dividing each coordinate by its spread, Lambda to the minus one half, the cloud is a round disc and an ordinary dot product measures similarity. Two marked points, a training point and a test point, are carried through all three panels.</desc>",
         CSS,
         '<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" class="mk"/></marker></defs>',
         '<style>.mk{fill:#57564f}.dot{fill:#8a8880;opacity:.55}.ell{fill:none;stroke-width:1.2}.big{font-size:12.5px;fill:#1a1a19;font-weight:500}'
         '@media (prefers-color-scheme: dark){.mk{fill:#b6b4ab}.dot{fill:#85837b}.big{fill:#eceae3}}</style>']
    def cloud(P, c):
        out = []
        for a in P:
            x, y = pt(a, c); out.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="2.6"/>')
        return out
    def ellipse(c, A, rad, cls="s3"):
        t = np.linspace(0, 2 * np.pi, 90)
        E = np.c_[rad[0] * np.cos(t), rad[1] * np.sin(t)] @ A.T
        pts = " ".join(f"{pt(e, c)[0]:.1f},{pt(e, c)[1]:.1f}" for e in E)
        return f'<polyline class="ell {cls}" points="{pts}"/>'
    def arrow(c, v, cls, lab, dx=6, dy=-6):
        x, y = pt(v, c)
        return (f'<path class="ln {cls}" style="stroke-width:2.2" d="M{c:.1f} {cy}L{x:.1f} {y:.1f}" marker-end="url(#ah)"/>'
                f'<text class="big" x="{x + dx:.1f}" y="{y + dy:.1f}">{lab}</text>')
    def axes(c, lab1, lab2):
        return (f'<path class="gd" d="M{c-130} {cy}H{c+130}M{c} {cy-110}V{cy+100}"/>'
                f'<text class="lab" x="{c+130}" y="{cy+14}" text-anchor="end">{lab1}</text>'
                f'<text class="lab" x="{c+6}" y="{cy-100}">{lab2}</text>')
    # panel a: original coordinates
    s.append('<text class="hd" x="20" y="24">1  Training cloud in the original coordinates</text>')
    s.append(axes(cx[0], "x1", "x2"))
    s += cloud(Xp, cx[0])
    s.append(ellipse(cx[0], U, 2 * sd))
    s.append(arrow(cx[0], U[:, 0] * 2 * sd[0], "s1", "u1"))
    s.append(arrow(cx[0], U[:, 1] * 2 * sd[1], "s2", "u2", dx=-22, dy=-4))
    # panel b: rotate
    Xr = Xp @ U
    s.append('<text class="hd" x="320" y="24">2  Rotate:  coordinates = position along u1, u2</text>')
    s.append(axes(cx[1], "u1 coordinate", "u2 coordinate"))
    s += cloud(Xr, cx[1])
    s.append(ellipse(cx[1], np.eye(2), 2 * sd))
    s.append(arrow(cx[1], np.array([2 * sd[0], 0]), "s1", "spread √λ₁"))
    s.append(arrow(cx[1], np.array([0, 2 * sd[1]]), "s2", "spread √λ₂", dx=10, dy=-14))
    # panel c: whiten
    Xw = Xr / sd
    s.append('<text class="hd" x="620" y="24">3  Divide each axis by its spread: round</text>')
    s.append(axes(cx[2], "u1 / √λ₁", "u2 / √λ₂"))
    s += cloud(Xw, cx[2])
    s.append(ellipse(cx[2], np.eye(2), (2.0, 2.0)))
    # marked pair in all three panels
    for P, c in ((Xp, cx[0]), (Xr, cx[1]), (Xw, cx[2])):
        for idx, cls, lab in ((0, "f1", "xᵢ"), (1, "f2", "xⱼ")):
            x, y = pt(P[idx], c)
            s.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="5.2" style="stroke:var(--none,#fff);stroke-width:1"/>'
                     f'<text class="big" x="{x + (8 if idx == 0 else -22):.1f}" y="{y - 7 if idx == 0 else y + 14:.1f}">{lab}</text>')
    # bottom explanation, in three columns
    notes = [
        ["Eigenvectors U = [u1 u2] of XᵀX are the", "tilted axes the cloud is stretched along;", "λ₁, λ₂ say how stretched (λ ∝ spread²)."],
        ["Uᵀ only re-expresses each point in those", "axes. Nothing is stretched yet; xᵢ·xⱼ is", "unchanged by a rotation."],
        ["Λ^(-1/2) shrinks the long axis and grows the", "short one until both have the same spread.", "Now the plain dot product is a fair similarity."],
    ]
    for c, ls in zip(cx, notes):
        for n, line in enumerate(ls):
            s.append(f'<text class="lab" x="{c-135}" y="{300+n*14}">{line}</text>')
    s.append('<text class="hd" x="20" y="366">xⱼᵀ (XᵀX)⁻¹ xᵢ  =  xⱼᵀ U Λ⁻¹ Uᵀ xᵢ  =  (Λ^(-1/2) Uᵀ xⱼ) · (Λ^(-1/2) Uᵀ xᵢ)  =  dot product in panel 3</text>')
    s.append("</svg>")
    (d / "eigen-whitening.svg").write_text("\n".join(s))


def write_figures(rows):
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    fig_weights(d)
    fig_gd(d, rows)
    fig_eig(d)
    print("\nfigures: wrote ols-weights.svg, gd-embedding.svg, eigen-whitening.svg in", d)


def main():
    check_identity()
    check_weights()
    check_double_count()
    check_uniqueness()
    rows = check_gd()
    check_pcr_ridge()
    check_stacking()
    check_masking()
    check_table1()
    if "--figures" in sys.argv:
        write_figures(rows)


if __name__ == "__main__":
    main()
