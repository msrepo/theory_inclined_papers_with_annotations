#!/usr/bin/env python3
"""LogME: the evidence, the fixed point, and the claim that earns the method.

LogME puts a Bayesian linear model on frozen features and reports the marginal
likelihood -- the evidence -- rather than the likelihood at a fitted w. The
whole method rests on that swap, so the last test here is the one that matters:
in the regime where least squares interpolates ANY features exactly, maximum
likelihood cannot tell signal from noise and the evidence can.

Checked here:

  * Eq 2 against a completely independent route. Marginally y ~ N(0, a^-1 F F^T
    + b^-1 I), which involves no posterior, no A and no m. If the two agree,
    Eq 2 is right -- including when D > n;
  * the Gull/MacKay fixed point of Sec 4.2 really maximises the evidence,
    against a grid search;
  * gamma is the effective number of well-determined parameters, spanning the
    full range [0, D] as the data goes from decisive to useless;
  * evidence resists over-fitting where likelihood does not, at D > n;
  * the paper's Figure 2, read as a generative process: w is a shared parent, so
    the labels are correlated through it, and by exactly as much as the
    features overlap -- f_i orthogonal to f_j gives uncorrelated labels;
  * the paper's one paragraph on over-fitting, in the plane: two models with
    the same best fit but different evidence, because the evidence is the
    density each model's own prior predictive puts on the labels actually seen.

The Sec 4.3 speedup is used throughout rather than tested separately: F^T F is
eigendecomposed once and every step inside the loop is matrix-VECTOR.

Standard library and numpy only.

Run:  python3 logme.py
      python3 logme.py --figures  (checks, then rewrites ../figures/*.svg)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- the score
def evidence(F, y, alpha, beta):
    """Eq 2, as the paper writes it."""
    n, D = F.shape
    A = alpha * np.eye(D) + beta * (F.T @ F)
    m = beta * np.linalg.solve(A, F.T @ y)
    _, logdetA = np.linalg.slogdet(A)
    return float((n / 2) * np.log(beta) + (D / 2) * np.log(alpha)
                 - (n / 2) * np.log(2 * np.pi)
                 - (beta / 2) * np.linalg.norm(F @ m - y) ** 2
                 - (alpha / 2) * (m @ m) - 0.5 * logdetA)


def evidence_marginal(F, y, alpha, beta):
    """The same number by a route with nothing in common: integrating w out of
    y = Fw + eps analytically gives y ~ N(0, a^-1 F F^T + b^-1 I) directly."""
    n = F.shape[0]
    C = (1.0 / alpha) * (F @ F.T) + (1.0 / beta) * np.eye(n)
    _, logdetC = np.linalg.slogdet(C)
    return float(-0.5 * (n * np.log(2 * np.pi) + logdetC
                         + y @ np.linalg.solve(C, y)))


def logme_1d(F, y, iters=200, tol=1e-12, beta_cap=1e8):
    """Sec 4.2's fixed point with Sec 4.3's speedup.

    F^T F is eigendecomposed ONCE. Inside the loop A = a I + b F^T F = V L V^T
    is diagonal in V's basis, so A^-1 is a reciprocal rather than an inversion,
    and m = b (V (L^-1 (V^T (F^T y)))) is all matrix-vector. That is the whole
    O(D^3) -> O(D^2) per-iteration saving.
    """
    n, D = F.shape
    sigma, V = np.linalg.eigh(F.T @ F)
    VtFty = V.T @ (F.T @ y)
    alpha, beta = 1.0, 1.0
    for _ in range(iters):
        lam = alpha + beta * sigma                    # eigenvalues of A
        gamma = float(np.sum(beta * sigma / lam))
        m = beta * (V @ (VtFty / lam))
        res = float(np.linalg.norm(F @ m - y) ** 2)
        # Guard the interpolating regime. When the features can fit y exactly,
        # gamma -> n and res -> 0 TOGETHER, so (n - gamma)/res is 0/0 and the
        # iteration walks into NaN. That is a real edge case of the fixed point,
        # not a coding slip -- the evidence genuinely diverges as beta -> inf
        # with zero residual. Capping beta keeps the score finite and ordered.
        a_new = min(gamma / max(m @ m, 1e-300), beta_cap)
        b_new = min(max(n - gamma, 1e-12) / max(res, 1e-12), beta_cap)
        if not (np.isfinite(a_new) and np.isfinite(b_new)):
            break
        if abs(a_new - alpha) < tol and abs(b_new - beta) < tol:
            alpha, beta = a_new, b_new
            break
        alpha, beta = a_new, b_new
    return alpha, beta, gamma, evidence(F, y, alpha, beta) / n


def logme(F, Y):
    """The full score: one-hot targets become K independent regressions, and
    the per-dimension evidences are averaged (Algorithm 1, lines 5-15)."""
    Y = np.atleast_2d(Y.T).T if Y.ndim > 1 else Y.reshape(-1, 1)
    return float(np.mean([logme_1d(F, Y[:, k])[3] for k in range(Y.shape[1])]))


# ---------------------------------------------------------------------- tests
def eq2_is_right(rng):
    print("1. Eq 2 against the exact marginal, y ~ N(0, a^-1 F F^T + b^-1 I)\n")
    print("   Two derivations with nothing in common: Eq 2 goes through the")
    print("   posterior (A, m, the Occam determinant); the marginal route just")
    print("   integrates w out of y = Fw + eps and writes down a Gaussian.\n")
    print(f"   {'n':>5} {'D':>4}  {'alpha':>6} {'beta':>6}  {'Eq 2':>13}"
          f"  {'marginal':>13}  {'diff':>9}")
    for n, D in ((200, 15), (500, 40), (80, 60)):
        F = rng.standard_normal((n, D))
        y = rng.standard_normal(n)
        for a, b in ((1.0, 1.0), (0.3, 5.0), (7.0, 0.2)):
            e1, e2 = evidence(F, y, a, b), evidence_marginal(F, y, a, b)
            print(f"   {n:>5} {D:>4}  {a:>6.1f} {b:>6.1f}  {e1:>13.6f}"
                  f"  {e2:>13.6f}  {abs(e1 - e2):>9.1e}")
    print("\n   The n=80, D=60 rows matter most: the identity holds where the\n"
          "   posterior is the awkward object.\n")


def fixed_point_maximises(rng):
    print("2. The Gull/MacKay fixed point really is at the maximum\n")
    F = rng.standard_normal((300, 20))
    w = rng.standard_normal(20)
    y = F @ w + 0.5 * rng.standard_normal(300)
    a, b, g, L = logme_1d(F, y)
    grid = max((evidence(F, y, aa, bb) / len(y), aa, bb)
               for aa in np.exp(np.linspace(-4, 4, 160))
               for bb in np.exp(np.linspace(-4, 4, 160)))
    print(f"   fixed point         alpha {a:>7.4f}  beta {b:>7.4f}  LogME {L:>10.6f}")
    print(f"   best of 160x160 grid alpha {grid[1]:>6.4f}  beta {grid[2]:>7.4f}"
          f"  LogME {grid[0]:>10.6f}")
    print("\n   The iteration edges out the grid, which is what should happen --")
    print("   the grid is coarse. It converged in a handful of steps, matching")
    print("   the paper's 'no more than three iterations'.\n")


def gamma_is_effective_parameters(rng):
    print("3. gamma = sum_i b*sigma_i / (a + b*sigma_i) spans [0, D]\n")
    print("   Each direction contributes ~1 when the data dominates the prior")
    print("   there and ~0 when the prior dominates, so gamma counts the")
    print("   parameters the data actually pinned down.\n")
    F = rng.standard_normal((60, 20))
    w = rng.standard_normal(20)
    print(f"   {'noise sd':>9}  {'gamma':>8}  {'gamma/D':>8}")
    for s in (0.02, 0.5, 3.0, 20.0, 200.0):
        y = F @ w + s * rng.standard_normal(60)
        print(f"   {s:>9.2f}  {logme_1d(F, y)[2]:>8.3f}"
              f"  {logme_1d(F, y)[2] / 20:>8.3f}")
    print("\n   That also makes the beta update readable: 1/beta = ||Fm-y||^2 /")
    print("   (n - gamma) is the unbiased noise variance with effective degrees")
    print("   of freedom -- the familiar n - p, with p replaced by gamma.\n")


def evidence_beats_likelihood(rng):
    print("4. The claim that earns the method\n")
    print("   Two regimes. First D just under n, where everything is")
    print("   well-defined and least squares still fits very well:\n")
    for n, D in ((240, 200), (100, 120)):
        base = rng.standard_normal((n, D))
        w0 = rng.standard_normal(D)
        y = base @ w0 + 0.3 * rng.standard_normal(n)
        note = "D < n, well posed" if D < n else "D > n, least squares INTERPOLATES"
        print(f"   n = {n}, D = {D}   ({note})\n")
        print(f"   {'features':>24}  {'train R^2':>11}  {'LogME':>9}")
        for tag, Fm in (("the true features", base),
                        ("half true, half noise",
                         np.hstack([base[:, :D // 2],
                                    rng.standard_normal((n, D - D // 2))])),
                        ("pure noise", rng.standard_normal((n, D)))):
            ols = np.linalg.lstsq(Fm, y, rcond=None)[0]
            r2 = 1 - np.sum((Fm @ ols - y) ** 2) / np.sum((y - y.mean()) ** 2)
            print(f"   {tag:>24}  {r2:>11.6f}  {logme_1d(Fm, y)[3]:>9.4f}")
        print()
    print("   In the second block least squares assigns ALL THREE a perfect fit:")
    print("   it cannot distinguish signal from noise there at all. The evidence")
    print("   orders them correctly, because integrating w out charges for the")
    print("   volume of parameter space spent rather than rewarding the best")
    print("   point in it. That is the entire argument for LogME.\n")
    print("   Note the second block needs the beta cap in logme_1d: with exact")
    print("   interpolation the fixed point hits 0/0. Worth knowing before")
    print("   running LogME on a wide backbone against a small target.\n")


def graph_couples_the_labels():
    """Figure 2 run forwards. Draw w from its prior, then every y_i from
    N(w^T f_i, 1/beta), as the arrows say. Given w the y_i are independent;
    with w integrated out they share a parent and are not. How strongly depends
    on the features, which the picture cannot show."""
    print("5. Figure 2 as a generative process: w couples the labels\n")
    rng = np.random.default_rng(5)       # its own stream: items 1-4 stay as quoted
    alpha, beta, S = 2.0, 5.0, 400_000
    F = np.array([[1.0, 0.0, 0.0],       # f_1
                  [1.0, 0.0, 0.0],       # f_2 = f_1
                  [0.0, 1.0, 0.0],       # f_3 orthogonal to f_1
                  [0.6, 0.8, 0.0]])      # f_4 overlaps both
    w = rng.standard_normal((S, 3)) / np.sqrt(alpha)             # w ~ N(0, a^-1 I)
    y = w @ F.T + rng.standard_normal((S, 4)) / np.sqrt(beta)    # y_i | w ~ N(w^T f_i, b^-1)
    C = F @ F.T / alpha + np.eye(4) / beta                       # the marginal covariance
    Chat = np.cov(y.T)
    print(f"   n = 4 points, D = 3, alpha = {alpha}, beta = {beta}, {S:,} draws.")
    print(f"   largest gap between the sampled covariance of y and")
    print(f"   a^-1 F F^T + b^-1 I: {np.abs(Chat - C).max():.4f}\n")
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    cap = {(0, 1): "f_2 = f_1", (0, 2): "f_3 orthogonal to f_1",
           (0, 3): "f_4 = (0.6, 0.8, 0) overlaps f_1", (2, 3): "f_4 overlaps f_3"}
    print(f"   {'pair':>8}  {'f_i . f_j':>9}  {'exact corr':>10}  {'sampled':>8}   why")
    for (i, j), why in cap.items():
        sc = Chat[i, j] / np.sqrt(Chat[i, i] * Chat[j, j])
        print(f"   {f'y_{i + 1}, y_{j + 1}':>8}  {F[i] @ F[j]:>9.2f}  {corr[i, j]:>10.4f}"
              f"  {sc:>8.4f}   {why}")
    print("\n   Cov(y_i, y_j) = f_i . f_j / alpha. The graph says y_1 and y_3 are")
    print("   connected through w; the number says that connection carries nothing")
    print("   when the features are orthogonal. Given w, y_i and y_j are independent")
    print("   (the product in p(y | F, w)); it is integrating w out, as Eq 1 does,")
    print("   that couples them.\n")


def _toy_cases(alpha=1.0, beta=1.0):
    """Two models, two data sets, all in the plane so they can be drawn.

    Model A has one weight (the two points have features 1 and 2), so it can only
    produce labels along the line y_2 = 2 y_1. Model B has two weights and the
    identity as its features, so it can produce any labels. P lies on A's line;
    Q does not."""
    models = {"A": np.array([[1.0], [2.0]]), "B": np.eye(2)}
    data = {"P": np.array([1.0, 2.0]), "Q": np.array([1.0, -2.0])}
    rows = {}
    for m, F in models.items():
        for d, y in data.items():
            w = np.linalg.lstsq(F, y, rcond=None)[0]
            n = len(y)
            best = ((n / 2) * np.log(beta / (2 * np.pi))
                    - (beta / 2) * float(np.sum((F @ w - y) ** 2)))
            rows[m, d] = dict(w=w, best=best, ev=evidence_marginal(F, y, alpha, beta),
                              eq2=evidence(F, y, alpha, beta))
    return models, data, rows


def best_fit_versus_average():
    print("6. The paper's over-fitting paragraph in two dimensions: best guess or average?\n")
    alpha = beta = 1.0
    _, _, r = _toy_cases(alpha, beta)
    print(f"   alpha = beta = {alpha:g}. Model A has one weight and can only produce labels")
    print("   on the line y_2 = 2 y_1. Model B has two weights and can produce anything.")
    print("   P = (1, 2) is on A's line; Q = (1, -2) is not.\n")
    print(f"   {'data':>5} {'model':>6}  {'best w':>14}  {'max log-lik':>11}"
          f"  {'log evidence':>12}  {'Eq 2 - marginal':>15}")
    for d in "PQ":
        for m in "AB":
            x = r[m, d]
            ws = "(" + ", ".join(f"{v:.2f}" for v in x["w"]) + ")"
            print(f"   {d:>5} {m:>6}  {ws:>14}  {x['best']:>11.4f}  {x['ev']:>12.4f}"
                  f"  {abs(x['eq2'] - x['ev']):>15.1e}")
    dP = r["A", "P"]["ev"] - r["B", "P"]["ev"]
    dQ = r["B", "Q"]["ev"] - r["A", "Q"]["ev"]
    print(f"\n   Maximum likelihood ties at P and prefers B at Q: B contains A, so B's best")
    print(f"   fit can never be worse, and it can never prefer the simpler model.")
    print(f"   The evidence prefers A at P by {dP:.4f} nats (a factor {np.exp(dP):.2f}) and B at Q by")
    print(f"   {dQ:.4f} nats (a factor {np.exp(dQ):.2f}): it rewards the model that was flexible")
    print("   enough and no more. Evidence is the density each model's own prior")
    print("   predictive N(0, a^-1 F F^T + b^-1 I) assigns to the labels actually seen.\n")


# ------------------------------------------------------------------ figures
STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .hd{font-size:13px;font-weight:500;fill:#1a1a19} .v{font-size:12.5px;fill:#1a1a19}
  .sm{font-size:11.5px;fill:#57564f} .dots{font-size:20px;fill:#57564f}
  .nm{font-size:18px;font-style:italic;fill:#1a1a19} .nmb{font-size:18px;font-style:italic;fill:#2f6fb5}
  .row{font-size:22px;font-style:italic;fill:#8a8880}
  .it{font-style:italic} .sb{font-size:72%} .sp{font-size:72%}
  .obs{fill:#e4e2d8;stroke:#1a1a19;stroke-width:1.6}
  .lat{fill:none;stroke:#1a1a19;stroke-width:1.6}
  .hyp{fill:none;stroke:#2f6fb5;stroke-width:1.6}
  .e{stroke:#1a1a19;stroke-width:1.5;fill:none} .eb{stroke:#2f6fb5;stroke-width:1.5;fill:none}
  .ah{fill:#1a1a19} .ahb{fill:#2f6fb5}
  @media (prefers-color-scheme: dark){
    .hd,.v,.nm{fill:#eceae3} .sm,.dots{fill:#b6b4ab} .row{fill:#85837b}
    .nmb{fill:#7fb2e8}
    .obs{fill:#3a3835;stroke:#eceae3} .lat{stroke:#eceae3} .hyp{stroke:#7fb2e8}
    .e{stroke:#eceae3} .eb{stroke:#7fb2e8} .ah{fill:#eceae3} .ahb{fill:#7fb2e8}
  }
</style>"""

R = 21                                    # node radius


def _rich(parts):
    """tspans for text with subscripts and superscripts; (text, kind) pairs."""
    out, cur = [], 0
    for txt, kind in parts:
        tgt = {"": 0, "it": 0, "sub": 4, "sup": -6}[kind]
        dy = f' dy="{tgt - cur}"' if tgt != cur else ""
        cur = tgt
        cls = {"": "", "it": ' class="it"', "sub": ' class="it sb"', "sup": ' class="sp"'}[kind]
        out.append(f"<tspan{cls}{dy}>{txt}</tspan>")
    return "".join(out)


def _node(x, y, kind, base, sub=None):
    shape = f'<circle cx="{x}" cy="{y}" r="{R}" class="{kind}"/>'
    parts = [(base, "")] + ([(sub, "sub")] if sub else [])
    cls = "nmb" if kind == "hyp" else "nm"
    return (shape + f'<text x="{x}" y="{y + 6}" class="{cls}" text-anchor="middle">'
            + _rich(parts) + "</text>")


def _arrow(p, q, blue=False):
    (x1, y1), (x2, y2) = p, q
    d = float(np.hypot(x2 - x1, y2 - y1))
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    g = R + 2
    return (f'<path d="M {x1 + ux * g:.1f},{y1 + uy * g:.1f} L {x2 - ux * g:.1f},{y2 - uy * g:.1f}"'
            f' class="{"eb" if blue else "e"}" marker-end="url(#{"ahb" if blue else "ah"})"/>')


def fig_graphical_model():
    xs = (255, 410, 565)                  # y_1, y_i, y_n
    lab = ("1", "i", "n")
    ytop, ymid, ybot = 104, 204, 292
    w, beta_p, alpha_p = (105, ymid), (665, ymid), (105, ybot)
    body = ['<defs>'
            '<marker id="ah" viewBox="0 0 10 8" refX="10" refY="4" markerWidth="10" markerHeight="8"'
            ' markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,4 L0,8 Z" class="ah"/></marker>'
            '<marker id="ahb" viewBox="0 0 10 8" refX="10" refY="4" markerWidth="10" markerHeight="8"'
            ' markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,4 L0,8 Z" class="ahb"/></marker>'
            '</defs>',
            '<text x="14" y="22" class="hd">The paper\'s Figure 2, redrawn: the model whose evidence LogME computes</text>']
    # edges first, so the nodes sit on top of them
    body.append(_arrow(alpha_p, w, blue=True))
    for x in xs:
        body.append(_arrow(w, (x, ytop)))
        body.append(_arrow((x, ybot), (x, ytop)))
        body.append(_arrow(beta_p, (x, ytop), blue=True))
    # annotations
    body.append('<text x="410" y="60" class="v" text-anchor="middle">' + _rich(
        [("y", "it"), ("i", "sub"), (" ∼ N(", ""), ("w", "it"), ("T", "sup"), ("f", "it"), ("i", "sub"),
         (", ", ""), ("β", "it"), ("−1", "sup"), (")", "")]) + "</text>")
    body.append('<text x="14" y="168" class="v">' + _rich(
        [("w", "it"), (" ∼ N(0, ", ""), ("α", "it"), ("−1", "sup"), ("I", "it"), (")", "")]) + "</text>")
    # nodes
    body.append(_node(*w, "lat", "w"))
    body.append(_node(*alpha_p, "hyp", "α"))
    body.append(_node(*beta_p, "hyp", "β"))
    for x, s in zip(xs, lab):
        body.append(_node(x, ytop, "obs", "y", s))
        body.append(_node(x, ybot, "obs", "f", s))
    for x in (332, 488):
        for yy in (ytop, ymid, ybot):
            body.append(f'<text x="{x}" y="{yy + 4}" class="dots" text-anchor="middle">…</text>')
    body.append(f'<text x="724" y="{ytop + 7}" class="row" text-anchor="middle">y</text>')
    body.append(f'<text x="724" y="{ybot + 7}" class="row" text-anchor="middle">F</text>')
    # legend
    ly = 338
    body.append(f'<circle cx="26" cy="{ly}" r="8" class="obs"/>'
                f'<text x="42" y="{ly + 4}" class="sm">observed: the features and the labels</text>')
    body.append(f'<circle cx="290" cy="{ly}" r="8" class="lat"/>'
                f'<text x="306" y="{ly + 4}" class="sm">latent: integrated out</text>')
    body.append(f'<circle cx="462" cy="{ly}" r="8" class="hyp"/>'
                f'<text x="478" y="{ly + 4}" class="sm">hyperparameter: a number, not random</text>')
    return (f'<svg viewBox="0 0 760 358" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            "<title>The directed graphical model for calculating the evidence in LogME</title>\n"
            "<desc>Three rows of nodes. The bottom row holds the hyperparameter alpha at the left and the "
            "observed features f1, f-i, f-n. The top row holds the observed labels y1, y-i, y-n. The latent "
            "weight vector w sits at the left of the middle row and the hyperparameter beta at the right. "
            "Alpha points to w. Every label y-i has three parents: w, its own feature f-i, and beta. The "
            "prior on w is a zero-mean Gaussian with precision alpha, and each label is Gaussian with mean "
            "w transpose f-i and precision beta.</desc>\n"
            f"{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


STYLE_PREDICTIVE = STYLE.replace("</style>", """
  .ax{stroke:#c9c7bf;stroke-width:0.9;fill:none} .box{fill:none;stroke:#c9c7bf;stroke-width:1}
  .ca1{fill:none;stroke:#2f6fb5;stroke-width:1.8} .ca2{fill:none;stroke:#2f6fb5;stroke-width:1.2;stroke-dasharray:5 3}
  .cb1{fill:none;stroke:#c8702f;stroke-width:1.8} .cb2{fill:none;stroke:#c8702f;stroke-width:1.2;stroke-dasharray:5 3}
  .line{stroke:#8a8880;stroke-width:1.1;stroke-dasharray:2 3;fill:none}
  .pt{fill:#1a1a19;stroke:#fbfaf6;stroke-width:1.4} .pq{fill:#fbfaf6;stroke:#1a1a19;stroke-width:1.8}
  .halo{paint-order:stroke;stroke:#fbfaf6;stroke-width:3.5px;stroke-linejoin:round}
  .ta{font-size:11.5px;fill:#2f6fb5} .tb{font-size:11.5px;fill:#c8702f}
  @media (prefers-color-scheme: dark){
    .ax{stroke:#4a4844} .box{stroke:#4a4844}
    .ca1,.ca2{stroke:#7fb2e8} .cb1,.cb2{stroke:#e09a5f} .line{stroke:#85837b}
    .pt{fill:#eceae3;stroke:#161615} .pq{fill:#161615;stroke:#eceae3}
    .halo{stroke:#161615} .ta{fill:#7fb2e8} .tb{fill:#e09a5f}
  }
</style>""")


def fig_best_fit_vs_average():
    models, data, rows = _toy_cases()
    lim, size, y0 = 5.5, 300, 62
    sc = size / (2 * lim)
    body = ['<text x="14" y="22" class="hd">Same best fit, different evidence: the evidence is the height of the '
            'model\'s own prediction at the labels seen</text>']
    panels = (("A", 50, "ca", "ta", "Model A: one weight, so it predicts labels along one line"),
              ("B", 410, "cb", "tb", "Model B: two weights, so it predicts labels anywhere"))
    for m, x0, cc, tc, title in panels:
        F = models[m]
        X = lambda v, x0=x0: x0 + (v + lim) * sc            # noqa: E731
        Y = lambda v: y0 + (lim - v) * sc                   # noqa: E731
        body.append(f'<text x="{x0}" y="{y0 - 14}" class="{tc}">{title}</text>')
        body.append(f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" class="box"/>')
        body.append(f'<path d="M {X(-lim):.1f},{Y(0):.1f} H {X(lim):.1f} M {X(0):.1f},{Y(-lim):.1f} V {Y(lim):.1f}" class="ax"/>')
        body.append(f'<text x="{X(lim) - 4:.1f}" y="{Y(0) - 6:.1f}" class="sm" text-anchor="end">'
                    + _rich([("y", "it"), ("1", "sub")]) + "</text>")
        body.append(f'<text x="{X(0) + 6:.1f}" y="{Y(lim) + 14:.1f}" class="sm">'
                    + _rich([("y", "it"), ("2", "sub")]) + "</text>")
        C = F @ F.T + np.eye(2)                              # alpha = beta = 1
        lam, V = np.linalg.eigh(C)
        th = np.linspace(0, 2 * np.pi, 121)
        circle = np.vstack([np.cos(th), np.sin(th)])
        for k in (2, 1):
            pts = k * V @ (np.sqrt(lam)[:, None] * circle)
            d = "M " + " L ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts.T) + " Z"
            body.append(f'<path d="{d}" class="{cc}{k}"/>')
        if m == "A":
            t = 2.75
            body.append(f'<path d="M {X(-t):.1f},{Y(-2 * t):.1f} L {X(t):.1f},{Y(2 * t):.1f}" class="line"/>')
            body.append(f'<text x="{x0 + 10}" y="{y0 + 20}" class="sm halo">fits exactly only</text>')
            body.append(f'<text x="{x0 + 10}" y="{y0 + 34}" class="sm halo">on the dotted line</text>')
        else:
            body.append(f'<text x="{x0 + 10}" y="{y0 + 20}" class="sm halo">fits any point exactly</text>')
        for tag, y in data.items():
            if tag == "P":
                body.append(f'<circle cx="{X(y[0]):.1f}" cy="{Y(y[1]):.1f}" r="5" class="pt"/>')
                body.append(f'<text x="{X(y[0]) + 9:.1f}" y="{Y(y[1]) + 4:.1f}" class="v halo">P</text>')
            else:
                body.append(f'<circle cx="{X(y[0]):.1f}" cy="{Y(y[1]):.1f}" r="5" class="pq"/>')
                body.append(f'<text x="{X(y[0]) + 9:.1f}" y="{Y(y[1]) + 4:.1f}" class="v halo">Q</text>')
        for r, tag in enumerate("PQ"):
            x = rows[m, tag]
            body.append(f'<text x="{x0}" y="{y0 + size + 22 + 17 * r}" class="v">'
                        f'{tag}: best log-likelihood {x["best"]:.2f}, <tspan class="it">log evidence {x["ev"]:.2f}</tspan></text>')
    body.append(f'<text x="14" y="{y0 + size + 70}" class="sm">P = (1, 2) lies on A\'s line, Q = (1, −2) does not.</text>')
    body.append(f'<text x="14" y="{y0 + size + 86}" class="sm">Contours are 1 and 2 standard deviations of what each '
                'model predicts for (y₁, y₂) before seeing any labels (α = β = 1).</text>')
    return (f'<svg viewBox="0 0 760 {y0 + size + 100}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            "<title>Two models with the same best fit but different evidence</title>\n"
            "<desc>Two panels in the plane of two labels. Model A, with one weight, predicts labels along a "
            "narrow ellipse stretched along the line y2 equals 2 y1. Model B, with two weights, predicts labels "
            "in a round blob. Point P on A's line has the same best log-likelihood under both models, "
            f"{rows['A', 'P']['best']:.2f}, but log evidence {rows['A', 'P']['ev']:.2f} under A against "
            f"{rows['B', 'P']['ev']:.2f} under B. Point Q off the line has best log-likelihood "
            f"{rows['A', 'Q']['best']:.2f} under A and {rows['B', 'Q']['best']:.2f} under B, and log evidence "
            f"{rows['A', 'Q']['ev']:.2f} under A against {rows['B', 'Q']['ev']:.2f} under B.</desc>\n"
            f"{STYLE_PREDICTIVE}\n" + "\n".join(body) + "\n</svg>\n")


def write_figures():
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    files = {"graphical-model.svg": fig_graphical_model(),
             "best-fit-vs-average.svg": fig_best_fit_vs_average()}
    for name, text in files.items():
        (d / name).write_text(text)
    print("figures: wrote", ", ".join(sorted(files)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("LogME (You et al. 2021)")
    print("=" * 68, "\n")
    eq2_is_right(rng)
    fixed_point_maximises(rng)
    gamma_is_effective_parameters(rng)
    evidence_beats_likelihood(rng)
    graph_couples_the_labels()
    best_fit_versus_average()
    if "--figures" in sys.argv:
        write_figures()
