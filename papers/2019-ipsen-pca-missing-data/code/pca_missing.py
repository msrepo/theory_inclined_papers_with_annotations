#!/usr/bin/env python3
"""Phase transition in PCA with missing data (Ipsen & Hansen, ICML 2019).

Claim: for one spike with signal-to-noise S = ||a0||^2 / sigma^2 and sample ratio
alpha = N/D, the expected squared overlap R^2 of the top principal direction with
the true one is, in the large-D limit,

    R^2(alpha, S) = 0                          if alpha S^2 <  1
                  = (alpha S^2 - 1)/(S + alpha S^2)   otherwise            (Eq 1)

and with entries missing completely at random at rate m, S -> (1-m) S         (Eq 2).

Checked here:
  * Eq 1 is the Baik-Ben Arous-Peche / Paul overlap formula, rewritten;
  * the large-alpha expansion 1 - (S+1)/(S^2 alpha) + O(alpha^-2);
  * the appendix's stationary-point step: with 1 - r^2 (not 1 + r^2) in the first
    term, extremising (sqrt(1-r^2) + sqrt(alpha(1+S r^2)))^2 over r gives Eq 28;
    with the printed 1 + r^2 there is no interior stationary point;
  * Eq 1 against simulated top eigenvectors (D = 600);
  * Eq 2 against PCA on zero-filled data, and against EM probabilistic PCA run
    on the observed entries only;
  * that Eq 2 is NOT the same curve as an effective sample size, N(1-m) (the
    paper's Fig 2c) or N(1-m)^2, even though alpha S(m)^2 = alpha (1-m)^2 S^2;
  * where the transition sits for the paper's Fig 2a parameters;
  * the plateau: what the two readings predict far above threshold.

Standard library and numpy only.

Run:  python3 pca_missing.py            (checks, prints every number the notes quote)
      python3 pca_missing.py --figures  (checks, then rewrites ../figures/*.svg)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ theory
def r2(alpha, S):
    """Eq 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(alpha * S * S < 1, 0.0, (alpha * S * S - 1) / (S + alpha * S * S))


def r2_bbp(alpha, S):
    """The same thing in random-matrix form, with c = D/N and spike strength S."""
    c = 1.0 / alpha
    return np.where(S * S < c, 0.0, (1 - c / S**2) / (1 + c / S))


def r2_missing(alpha, S, m):
    """Eq 2."""
    return r2(alpha, (1 - m) * S)


def m_critical(alpha, S):
    """Missing rate at which Eq 2 hits zero: (1-m) S sqrt(alpha) = 1."""
    return 1 - 1 / (S * np.sqrt(alpha))


# ------------------------------------------------------------------ simulation
def make(N, D, S):
    a0 = RNG.standard_normal(D)
    a0 /= np.linalg.norm(a0)
    sigma = 1 / np.sqrt(S)                      # ||a0|| = 1, so S = 1/sigma^2
    X = np.outer(RNG.standard_normal(N), a0) + sigma * RNG.standard_normal((N, D))
    return X, a0


def top_dir(X):
    return np.linalg.svd(X, full_matrices=False)[2][0]


def sim_zero_filled(N, D, S, m, reps):
    out = []
    for _ in range(reps):
        X, a0 = make(N, D, S)
        M = RNG.random((N, D)) >= m
        out.append((top_dir(X * M) @ a0) ** 2)
    return float(np.mean(out))


def em_ppca(X, M, iters=200):
    """One-component probabilistic PCA fitted by EM on the observed entries."""
    Xm = X * M
    v = top_dir(Xm)
    w = v * np.sqrt(np.var(Xm) * X.shape[1] * 0.5)
    s2 = np.var(Xm[M]) * 0.5
    for _ in range(iters):
        prec = 1 + (M * w**2).sum(1) / s2
        Ez = (Xm @ w) / s2 / prec
        Vz = 1 / prec
        w = (Xm * Ez[:, None]).sum(0) / (M * (Ez**2 + Vz)[:, None]).sum(0)
        res = (M * (X - np.outer(Ez, w)) ** 2).sum() + (M * (w**2)[None, :] * Vz[:, None]).sum()
        s2 = res / M.sum()
    return w / np.linalg.norm(w)


def sim_em(N, D, S, m, reps):
    out = []
    for _ in range(reps):
        X, a0 = make(N, D, S)
        M = RNG.random((N, D)) >= m
        out.append((em_ppca(X, M) @ a0) ** 2)
    return float(np.mean(out))


# ------------------------------------------------------------------ checks
def check_algebra():
    print("== Eq 1 against the random-matrix form and its expansion")
    worst = max(abs(float(r2(a, S) - r2_bbp(a, S))) for a in (0.3, 1, 2, 5, 40) for S in (0.5, 1, 2, 7))
    print(f"   max |Eq1 - BBP form| over a grid: {worst:.1e}")
    for alpha in (10, 100, 1000):
        S = 2.0
        exact = float(r2(alpha, S))
        approx = 1 - (S + 1) / (S**2 * alpha)
        print(f"   S=2 alpha={alpha:4d}: exact {exact:.6f}  1-(S+1)/(S^2 alpha) {approx:.6f}"
              f"  diff*alpha^2 = {(exact - approx) * alpha**2:+.3f}")
    print("   (the remainder is O(alpha^-2), not o(alpha^-2) as the text writes)")


def check_stationary():
    print("== Appendix: the stationary point over r")
    for alpha, S in ((3.0, 2.0), (0.4, 2.0), (1.0, 0.5)):
        r = np.linspace(0, 1, 200001)
        g = np.sqrt(1 - r**2) + np.sqrt(alpha * (1 + S * r**2))
        r_star = r[np.argmax(g)]
        want = float(r2(alpha, S))
        gp = np.sqrt(1 + r**2) + np.sqrt(alpha * (1 + S * r**2))
        print(f"   alpha={alpha} S={S}: argmax r^2 of (sqrt(1-r^2)+...) = {r_star**2:.4f}, Eq 28 = {want:.4f};"
              f" with 1+r^2 the maximiser is r = {r[np.argmax(gp)]:.2f} (the boundary)")


def check_complete(D=600, reps=4):
    print(f"== Eq 1 against simulation (D={D}, {reps} reps)")
    rows = []
    for S, alpha in ((2, 0.2), (2, 0.5), (2, 1), (2, 2), (2, 5), (0.7, 3), (0.7, 6)):
        s = sim_zero_filled(int(alpha * D), D, S, 0.0, reps)
        t = float(r2(alpha, S))
        rows.append((S, alpha, s, t))
        print(f"   S={S:<4} alpha={alpha:<4} sim {s:.3f}   Eq1 {t:.3f}")
    return rows


def check_missing(D=600, reps=4):
    print(f"== Eq 2 against zero-filled PCA, and two effective-sample-size readings (D={D}, N=2D)")
    N = 2 * D
    err = {"Eq 2 (S -> (1-m)S)": [], "N(1-m)": [], "N(1-m)^2": []}
    rows = []
    for S, m in ((4, 0.3), (4, 0.5), (4, 0.7), (4, 0.8), (8, 0.7), (8, 0.8), (8, 0.9)):
        s = sim_zero_filled(N, D, S, m, reps)
        a = N / D
        t = float(r2_missing(a, S, m))
        n1 = float(r2(a * (1 - m), S))
        n2 = float(r2(a * (1 - m) ** 2, S))
        rows.append((S, m, s, t, n1, n2))
        err["Eq 2 (S -> (1-m)S)"].append(abs(s - t))
        err["N(1-m)"].append(abs(s - n1))
        err["N(1-m)^2"].append(abs(s - n2))
        print(f"   S={S} m={m}: sim {s:.3f} | Eq2 {t:.3f} | N(1-m) {n1:.3f} | N(1-m)^2 {n2:.3f}")
    for k, v in err.items():
        print(f"   mean |sim - prediction|  {k:<20} {np.mean(v):.3f}")
    return rows


def check_em(D=400, reps=3):
    print(f"== EM probabilistic PCA (observed entries only), the estimator the paper actually ran (D={D}, N=2D)")
    N = 2 * D
    for S, m in ((3, 0.3), (4, 0.5), (3, 0.6)):
        e = sim_em(N, D, S, m, reps)
        z = sim_zero_filled(N, D, S, m, reps)
        print(f"   S={S} m={m}: EM-PPCA {e:.3f} | zero-filled PCA {z:.3f} | Eq 2 {float(r2_missing(N / D, S, m)):.3f}")


def check_fig2a():
    print("== Where Fig 2a's two transitions sit (N=2000, D=3000, sigma^2=0.05, ||a||=1, 0.5)")
    alpha = 2000 / 3000
    for S in (20.0, 5.0):
        print(f"   S={S:>4}: critical missing rate {m_critical(alpha, S):.3f};"
              f" with no missing data alpha S^2 = {alpha * S * S:.1f}")


def check_plateau():
    print("== Far above threshold: what each reading predicts at m=0.5, S=4")
    S, m = 4.0, 0.5
    for alpha in (2, 5, 20, 100):
        print(f"   alpha={alpha:>3}: no missing {float(r2(alpha, S)):.3f} | Eq 2 {float(r2_missing(alpha, S, m)):.3f}"
              f" | N(1-m) {float(r2(alpha * (1 - m), S)):.3f} | N(1-m)^2 {float(r2(alpha * (1 - m) ** 2, S)):.3f}")
    print("   Same threshold only for N(1-m)^2; the plateau 1-(1+S')/(S'^2 alpha) separates them.")


# ------------------------------------------------------------------ figures
CSS = """
<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:10.5px;fill:#57564f}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s5{stroke:#8a8880}
  .f1{fill:#2a78d6} .f2{fill:#eb6834} .f3{fill:#1baf7a}
  .ln{stroke-width:2.2;fill:none;stroke-linecap:round;stroke-linejoin:round}
  .dash{stroke-width:1.6;fill:none;stroke-dasharray:5 3}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd{fill:#eceae3} .ax{stroke:#85837b} .gd{stroke:#33312e}
    .s1{stroke:#3987e5} .s2{stroke:#d95926} .s3{stroke:#199e70} .s5{stroke:#85837b}
    .f1{fill:#3987e5} .f2{fill:#d95926} .f3{fill:#199e70}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>{CSS}\n{body}\n</svg>\n")


class Axes:
    def __init__(self, x0, y0, w, h, xmax, ymax=1.0, xmin=0.0):
        self.x0, self.y0, self.w, self.h, self.xmax, self.ymax, self.xmin = x0, y0, w, h, xmax, ymax, xmin

    def X(self, x):
        return self.x0 + (x - self.xmin) / (self.xmax - self.xmin) * self.w

    def Y(self, y):
        return self.y0 + self.h - y / self.ymax * self.h

    def frame(self, xticks, yticks, xlabel, ylabel):
        o = []
        for t in yticks:
            o.append(f'<path class="gd" d="M{self.x0},{self.Y(t):.1f}H{self.x0 + self.w}"/>')
            o.append(f'<text class="sm" x="{self.x0 - 6}" y="{self.Y(t) + 3.5:.1f}" text-anchor="end">{t:g}</text>')
        for t in xticks:
            o.append(f'<text class="sm" x="{self.X(t):.1f}" y="{self.y0 + self.h + 15}" text-anchor="middle">{t:g}</text>')
        o.append(f'<path class="ax" d="M{self.x0},{self.y0}V{self.y0 + self.h}H{self.x0 + self.w}"/>')
        o.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 32}" text-anchor="middle">{xlabel}</text>')
        o.append(f'<text class="lab" transform="translate({self.x0 - 38},{self.y0 + self.h / 2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
        return "\n".join(o)

    def path(self, xs, ys, cls):
        d = "M" + "L".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))
        return f'<path class="{cls}" d="{d}"/>'


def fig_missing_curves(rows_sim):
    A = Axes(64, 44, 470, 270, 1.0)
    ms = np.linspace(0, 1, 400)
    body = [f'<text class="hd" x="64" y="24">Overlap with the true direction as entries go missing (N = 2D)</text>',
            A.frame([0, .2, .4, .6, .8, 1], [0, .25, .5, .75, 1], "missing rate m", "R²")]
    cols = {4: "s1", 8: "s2"}
    for S, cls in cols.items():
        body.append(A.path(ms, [float(r2_missing(2, S, m)) for m in ms], f"{cls} ln"))
        body.append(A.path(ms, [float(r2(2 * (1 - m), S)) for m in ms], f"{cls} dash"))
        for (S_, m, s, *_rest) in rows_sim:
            if S_ == S:
                body.append(f'<circle class="f{cls[1]}" cx="{A.X(m):.1f}" cy="{A.Y(s):.1f}" r="3.6"/>')
    body.append('<text class="sm" x="548" y="70">solid: Eq 2</text>')
    body.append('<text class="sm" x="548" y="86">dashed: N(1−m)</text>')
    body.append('<text class="sm" x="548" y="102">dots: simulation</text>')
    body.append('<text class="sm f1" x="548" y="126">S = 4</text><text class="sm f2" x="548" y="142">S = 8</text>')
    return svg(660, 360, "Overlap against missing rate, theory and simulation",
               "R squared against missing rate for S=4 and S=8 with N=2D. Solid curves are Eq 2, dashed curves treat missing data as a smaller sample N(1-m); simulation dots lie on the solid curves.",
               "\n".join(body))


def fig_plateau():
    A = Axes(64, 44, 470, 270, 100, xmin=0)
    al = np.linspace(0.05, 100, 600)
    S, m = 4.0, 0.5
    body = [f'<text class="hd" x="64" y="24">Same threshold, different plateau (S = 4, m = 0.5)</text>',
            A.frame([0, 20, 40, 60, 80, 100], [0, .25, .5, .75, 1], "samples per dimension α = N/D", "R²")]
    for f, cls, dash in ((lambda a: r2(a, S), "s5", ""), (lambda a: r2_missing(a, S, m), "s1", ""),
                         (lambda a: r2(a * (1 - m), S), "s2", " dash"), (lambda a: r2(a * (1 - m) ** 2, S), "s3", " dash")):
        body.append(A.path(al, [float(f(a)) for a in al], f"{cls}{dash if dash else ' ln'}"))
    body += ['<text class="sm" x="548" y="64">grey: no missing data</text>',
             '<text class="sm f1" x="548" y="82">Eq 2: S → (1−m)S</text>',
             '<text class="sm f2" x="548" y="100">N(1−m)</text>',
             '<text class="sm f3" x="548" y="118">N(1−m)²</text>']
    return svg(660, 360, "Plateau separates signal-to-noise scaling from sample scaling",
               "R squared against samples per dimension at S=4 and m=0.5. N(1-m)^2 has the same threshold as Eq 2 but approaches one faster; N(1-m) crosses the threshold earlier.",
               "\n".join(body))


def write_figures(rows_sim):
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    (d / "missing-curves.svg").write_text(fig_missing_curves(rows_sim))
    (d / "plateau.svg").write_text(fig_plateau())
    print("figures: wrote missing-curves.svg, plateau.svg")


if __name__ == "__main__":
    check_algebra()
    check_stationary()
    check_complete()
    rows = check_missing()
    check_em()
    check_fig2a()
    check_plateau()
    if "--figures" in sys.argv:
        write_figures(rows)
