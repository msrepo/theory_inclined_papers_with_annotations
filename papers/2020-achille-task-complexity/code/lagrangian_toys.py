#!/usr/bin/env python3
"""Sections 3.2, 6.1 and 7 of Achille, Paolini, Mbeng & Soatto: when does the
beta-minimiser jump, and what does Figure 2's "phase transition" look like in a
model small enough to solve?

Checked here:

  1. A double well in one weight, P = N(0, lam^2), Q = N(mu, sigma^2), with a
     wide shallow basin and a narrow deep one. The structure function S(t) is
     not convex; C_beta only sees its convex hull, so a whole interval of t is
     never beta-sufficient for any beta, and the minimiser jumps at beta_c.
     Annealing beta with epsilon-local steps (Definitions 6.2-6.3) starting
     from the global minimiser follows the wide basin past beta_c and ends at a
     local minimum: Proposition 6.4's hypothesis fails exactly at the jump.
  2. The convex case. For logistic regression with Gaussian Q, E_Q L + beta KL
     is jointly convex in (mu, sigma), so S(t) is convex, t*(beta) moves
     continuously, and the minimiser is never the constant: beta* of
     Proposition 3.6 does not exist for Definition 5.1.
  3. Figure 2 in miniature: random-features logistic regression, Laplace Q,
     real vs random labels. Real labels hold their fit to a larger beta than
     random ones, and at large beta the expected loss sits *above* the uniform
     loss ln 2, because Q is then close to the wide prior (Figure 2's plateaus
     are likewise above ln|Y|).

Standard library and numpy only.

Run:  python3 lagrangian_toys.py             (prints every number quoted in the notes)
      python3 lagrangian_toys.py --figures   (also rewrites ../figures/double-well.svg, ../figures/fig2-toy.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np

rng = np.random.default_rng(2)

# ----------------------------------------------------------------------- 1
N_DW, LAM_DW = 50, 3.0
WELLS = ((0.30, 1.0, 0.8), (0.60, 3.5, 0.08))     # (depth, centre, width)


def dw_loss(w):
    """Per-sample loss of the double well."""
    out = 1.0
    for A, c, s in WELLS:
        out = out - A * np.exp(-(w - c) ** 2 / (2 * s**2))
    return out


def dw_expected_loss(mu, sig):
    """N E_{w ~ N(mu, sig^2)} loss(w), in closed form."""
    out = 1.0
    for A, c, s in WELLS:
        v = s**2 + sig**2
        out = out - A * s / np.sqrt(v) * np.exp(-(mu - c) ** 2 / (2 * v))
    return N_DW * out


def kl1(mu, sig, lam):
    return np.log(lam / sig) + (sig**2 + mu**2) / (2 * lam**2) - 0.5


def dw_grid():
    mu = np.linspace(-1.0, 5.0, 1201)
    sig = np.exp(np.linspace(math.log(0.01), math.log(4.0), 400))
    M, S = np.meshgrid(mu, sig, indexing="ij")
    return M, S, dw_expected_loss(M, S), kl1(M, S, LAM_DW)


def hull_lower(t, s):
    """Lower convex hull of points (t, s), t increasing."""
    h = []
    for p in zip(t, s):
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
            h.pop()
        h.append(p)
    return np.array(h)


def intervals(t, mask):
    """Maximal runs of True in mask, as (t_start, t_end)."""
    out, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        if not m and start is not None:
            out.append((start, prev)); start = None
        prev = ti
    if start is not None:
        out.append((start, prev))
    return out


def structure_function(EL, KL, tgrid):
    order = np.argsort(KL.ravel())
    kl_sorted = KL.ravel()[order]
    el_cummin = np.minimum.accumulate(EL.ravel()[order])
    idx = np.searchsorted(kl_sorted, tgrid, side="right") - 1
    return el_cummin[np.clip(idx, 0, None)]


def check_double_well():
    print("1. Double well: one weight, P = N(0, 3^2), Q = N(mu, sigma^2), N = 50")
    print(f"   wells: depth 0.30 at w = 1 (width 0.8), depth 0.60 at w = 3.5 (width 0.08)")
    M, S, EL, KL = dw_grid()
    betas = np.geomspace(20, 0.3, 400)
    glob = []
    for b in betas:
        C = EL + b * KL
        i = np.unravel_index(np.argmin(C), C.shape)
        glob.append((b, M[i], S[i], KL[i], EL[i]))
    glob = np.array(glob)
    jumps = np.where(np.abs(np.diff(glob[:, 1])) > 0.5)[0]
    for j in jumps:
        b0, b1 = glob[j, 0], glob[j + 1, 0]
        print(f"   global minimiser jumps between beta = {b0:.3f} and {b1:.3f}: "
              f"mu {glob[j, 1]:.2f} -> {glob[j + 1, 1]:.2f}, sigma {glob[j, 2]:.3f} -> {glob[j + 1, 2]:.3f}, "
              f"KL {glob[j, 3]:.2f} -> {glob[j + 1, 3]:.2f} nats")
    # the structure function and its hull
    tg = np.linspace(0, 8, 1601)
    St = structure_function(EL, KL, tg)
    hl = hull_lower(tg, St)
    gap = St - np.interp(tg, hl[:, 0], hl[:, 1])
    for a, b in intervals(tg, gap > 0.2):
        g = gap[(tg >= a) & (tg <= b)].max()
        print(f"   S(t) lies above its convex hull for t in [{a:.2f}, {b:.2f}] nats (max gap {g:.2f} nats of loss): "
              f"no beta selects these complexities")
    # annealing with epsilon-local steps
    b_final = 0.5
    j = np.argmin(np.abs(glob[:, 0] - b_final))
    print(f"   at beta = {b_final}: global minimiser mu = {glob[j, 1]:.2f}, sigma = {glob[j, 2]:.3f}, "
          f"C = {glob[j, 4] + b_final * glob[j, 3]:.2f}")
    sched = np.geomspace(20, b_final, 60)
    res = {}
    for eps in (0.1, 0.5, 1.0, 2.0, 3.0):
        C0 = EL + sched[0] * KL
        i = np.unravel_index(np.argmin(C0), C0.shape)
        cur = np.array([M[i], S[i]])
        for b in sched:
            C = EL + b * KL
            for _ in range(200):                  # epsilon-local steps until they stop moving
                near = (M - cur[0]) ** 2 + (S - cur[1]) ** 2 <= eps**2
                Cn = np.where(near, C, np.inf)
                i = np.unravel_index(np.argmin(Cn), C.shape)
                new = np.array([M[i], S[i]])
                if np.allclose(new, cur):
                    break
                cur = new
        Cend = dw_expected_loss(cur[0], cur[1]) + b_final * kl1(cur[0], cur[1], LAM_DW)
        res[eps] = (cur, Cend)
        print(f"   annealed, epsilon = {eps:3.1f}: ends at mu = {cur[0]:.2f}, sigma = {cur[1]:.3f}, C = {Cend:.2f}"
              + ("   (global)" if abs(cur[0] - glob[j, 1]) < 0.1 else "   (stuck in the wide basin)"))
    return glob, (tg, St, hl), res


# ----------------------------------------------------------------------- 2
def check_convex():
    print("\n2. The convex case: one-weight logistic regression, Gaussian Q")
    n, lam = 40, 3.0
    x = rng.standard_normal(n)
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-2.0 * x)), 1.0, -1.0)
    gx, gw = np.polynomial.hermite_e.hermegauss(60)
    gw = gw / gw.sum()
    mus = np.linspace(-1, 6, 351)
    sigs = np.exp(np.linspace(math.log(0.01), math.log(3.0), 240))
    Mg, Sg = np.meshgrid(mus, sigs, indexing="ij")
    W = Mg[..., None] + Sg[..., None] * gx
    EL = (np.logaddexp(0, -W[..., None] * (y * x)).sum(-1) * gw).sum(-1)
    KL = kl1(Mg, Sg, lam)
    betas = np.geomspace(1e3, 1e-2, 120)
    ts, ms = [], []
    for b in betas:
        i = np.unravel_index(np.argmin(EL + b * KL), EL.shape)
        ts.append(KL[i]); ms.append(Mg[i])
    ts, ms = np.array(ts), np.array(ms)
    print(f"   t*(beta) over 120 betas from 1e3 to 1e-2: monotone {bool(np.all(np.diff(ts) >= -1e-9))}, "
          f"largest step between neighbours {np.diff(ts).max():.3f} nats (grid-limited)")
    print(f"   mu* at beta = 1e3 is {ms[0]:.2f}, not 0: the gradient of the loss at w = 0 is "
          f"{-(y * x).sum() / 2:.2f}, and KL is quadratic there, so some move always pays")
    tg = np.linspace(0, 6, 601)
    St = structure_function(EL, KL, tg)
    hl = hull_lower(tg, St)
    gap = St - np.interp(tg, hl[:, 0], hl[:, 1])
    print(f"   S(t) for t in [0, 6]: falls from {St[0]:.2f} to {St[-1]:.2f}; largest gap above its convex hull "
          f"{gap.max():.3f} nats (grid error; the double well's was 6.35)")


# ----------------------------------------------------------------------- 3
def rf_problem(n, D, labels, seed):
    r = np.random.default_rng(seed)
    X0 = r.standard_normal((n, 5))
    V = r.standard_normal((5, D))
    Phi = np.maximum(X0 @ V, 0) / math.sqrt(D)
    if labels == "real":
        y = np.sign(X0[:, 0] + 0.5 * X0[:, 1] ** 2 - 0.5)
        flip = r.random(n) < 0.05
        y[flip] *= -1
    else:
        y = np.where(r.random(n) < 0.5, 1.0, -1.0)
    return Phi, y


def laplace_curve(Phi, y, betas, lam=3.0, mc=300, seed=0):
    r = np.random.default_rng(seed)
    n, D = Phi.shape
    w = np.zeros(D)
    out = []
    for b in betas:                               # from large beta to small: warm starts
        prec = b / lam**2
        for _ in range(60):
            m = y * (Phi @ w)
            p = 1 / (1 + np.exp(-m))
            gr = -(Phi * ((1 - p) * y)[:, None]).sum(0) + prec * w
            Hh = (Phi * (p * (1 - p))[:, None]).T @ Phi
            step = np.linalg.solve(Hh + prec * np.eye(D), gr)
            w = w - step
            if np.linalg.norm(step) < 1e-9:
                break
        m = y * (Phi @ w)
        p = 1 / (1 + np.exp(-m))
        Hh = (Phi * (p * (1 - p))[:, None]).T @ Phi
        A = Hh + prec * np.eye(D)
        S = b * np.linalg.inv(A)
        L = np.linalg.cholesky(S)
        Ws = w + r.standard_normal((mc, D)) @ L.T
        EL = np.logaddexp(0, -(Ws @ Phi.T) * y).sum(1).mean()
        kl = 0.5 * (w @ w / lam**2 + np.trace(S) / lam**2 + D * math.log(lam**2) - np.linalg.slogdet(S)[1] - D)
        out.append((b, EL / n, kl))
    return np.array(out)


def crossing(curve, level):
    """Largest beta... the beta where per-sample loss first exceeds `level`, scanning up in beta."""
    b, l = curve[::-1, 0], curve[::-1, 1]
    k = np.argmax(l > level)
    return math.exp(np.interp(level, [l[k - 1], l[k]], [math.log(b[k - 1]), math.log(b[k])]))


def check_fig2_toy():
    print("\n3. Figure 2 in miniature: random-features logistic regression, Laplace Q, lam = 3")
    betas = np.geomspace(100, 1e-3, 36)
    curves = {}
    for lab in ("real", "random"):
        Phi, y = rf_problem(200, 400, lab, 5)
        curves[lab] = laplace_curve(Phi, y, betas)
    print("   beta      real: loss/sample   KL      random: loss/sample   KL")
    for i in range(0, len(betas), 5):
        print(f"   {betas[i]:8.3g}  {curves['real'][i, 1]:12.3f}  {curves['real'][i, 2]:8.1f}"
              f"   {curves['random'][i, 1]:16.3f}  {curves['random'][i, 2]:8.1f}")
    ln2 = math.log(2)
    for lab in ("real", "random"):
        c = curves[lab]
        print(f"   {lab:6s}: loss at beta = 100 is {c[0, 1]:.3f} (ln 2 = {ln2:.3f}); "
              f"loss reaches ln 2 / 2 at beta = {crossing(c, ln2 / 2):.3f}")
    print("   random labels, n = 100, 200, 400 with D = 800 fixed: beta where loss reaches ln 2 / 2")
    for n in (100, 200, 400):
        Phi, y = rf_problem(n, 800, "random", 7 + n)
        c = laplace_curve(Phi, y, betas, mc=100)
        print(f"     n = {n}: {crossing(c, ln2 / 2):.3f}")
    return betas, curves


# ----------------------------------------------------------------- figures
def figures(glob, sf, curves_pack):
    from svgkit import Axes, legend, svg, write
    tg, St, hl = sf
    body = ['<text class="hd" x="20" y="22">A double well: the β-minimiser jumps, and annealing misses the jump</text>',
            '<text class="sub" x="20" y="38">One weight; P = N(0, 3²); Q = N(µ, σ²); N = 50. Left: the per-sample loss. Right: the structure function.</text>']
    ax = Axes(52, 72, 270, 200, (-1, 5), (0.35, 1.05))
    body.append(ax.frame([-1, 0, 1, 2, 3, 4, 5], [0.4, 0.6, 0.8, 1.0], "weight w", "loss per sample"))
    ww = np.linspace(-1, 5, 1200)
    body.append(ax.path(ww, dw_loss(ww), "k0"))
    body.append(ax.text(1.0, 0.66, "wide, shallow", "tiny", "middle", dy=4))
    body.append(ax.text(3.5, 0.40, "narrow, deep", "tiny", "start", dx=8))
    # mark the global minimisers on either side of the jump
    j = np.where(np.abs(np.diff(glob[:, 1])) > 0.5)[0][-1]
    for jj, cls in ((j, "bf"), (j + 1, "of")):
        body.append(ax.dot(glob[jj, 1], 0.37, cls, 4))
    body.append(ax.text(glob[j, 1], 0.37, f"β &gt; {glob[j, 0]:.2f}", "tiny", "middle", dy=-8))
    body.append(ax.text(glob[j + 1, 1], 0.37, f"β &lt; {glob[j + 1, 0]:.2f}", "tiny", "end", dx=-8, dy=3))
    ax2 = Axes(392, 72, 270, 200, (0, 7), (15, 52))
    body.append(ax2.frame([0, 1, 2, 3, 4, 5, 6, 7], [20, 30, 40, 50], "t = KL(Q ‖ P) (nats)", "S(t) = min E_Q L (nats)"))
    keep = tg <= 7
    body.append(ax2.path(tg[keep], St[keep], "b"))
    hk = hl[hl[:, 0] <= 7]
    body.append(ax2.path(hk[:, 0], hk[:, 1], "o dash"))
    gap = St - np.interp(tg, hl[:, 0], hl[:, 1])
    for a, b in intervals(tg, (gap > 0.2) & keep):
        body.append(f'<rect class="shader" x="{ax2.X(a):.1f}" y="{ax2.y0}" width="{ax2.X(b) - ax2.X(a):.1f}" height="{ax2.h}"/>')
        if b - a > 1:
            body.append(ax2.text((a + b) / 2, 50, "never selected", "tiny", "middle"))
    body.append(legend(545, 100, [("b", "S(t)"), ("o dash", "convex hull")]))
    body.append(ax2.text(5.6, 41, "C_β sees only the hull", "tiny", "middle"))
    body.append('<text class="lab" x="20" y="322">A line of slope −β touches the hull; at β_c it touches two points at once, and the optimum jumps across the red band.</text>')
    body.append('<text class="lab" x="20" y="337">Annealing from large β with ε-local steps stays in the wide basin unless ε covers the whole jump.</text>')
    write("double-well.svg", svg(700, 350, "Double-well Lagrangian",
                                 "Left: a one-weight loss with a wide shallow basin and a narrow deep basin, with the global "
                                 "beta-minimisers on each side of the jump. Right: the structure function, its convex hull, "
                                 "and the band of complexities that no beta selects.", body))

    betas, curves = curves_pack
    body = ['<text class="hd" x="20" y="22">Figure 2 in miniature: loss per sample against β</text>',
            '<text class="sub" x="20" y="38">Random-features logistic regression, n = 200, D = 400, Laplace Q, λ = 3.</text>']
    ax = Axes(62, 66, 560, 220, (1e-3, 100), (0, 1.2), logx=True)
    body.append(ax.frame([1e-3, 1e-2, 1e-1, 1, 10, 100], [0, 0.3, 0.6, 0.9, 1.2], "β (log scale)",
                         "E_Q L / n (nats)", xfmt="{:g}"))
    body.append(f'<line class="k" x1="{ax.x0}" x2="{ax.x0 + ax.w}" y1="{ax.Y(math.log(2)):.1f}" y2="{ax.Y(math.log(2)):.1f}"/>')
    body.append(ax.text(1.2e-3, math.log(2), "ln 2: predict ½ everywhere", "tiny", dy=-5))
    body.append(ax.path(curves["real"][:, 0], curves["real"][:, 1], "b"))
    body.append(ax.path(curves["random"][:, 0], curves["random"][:, 1], "o dash"))
    body.append(legend(90, 96, [("b", "real labels (5% flipped)"), ("o dash", "random labels")]))
    body.append('<text class="lab" x="20" y="330">Real labels keep a low loss up to a larger β, as in the paper, but nothing is sharp: this model is convex, so the optimum</text>')
    body.append('<text class="lab" x="20" y="345">moves continuously. At large β both curves pass ln 2: Q is close to the wide prior and noisy logits cost more than predicting ½.</text>')
    write("fig2-toy.svg", svg(700, 358, "Loss against beta, miniature",
                              "Per-sample expected loss against beta for real and random labels in a random-features "
                              "logistic regression with a Laplace post-distribution.", body))


if __name__ == "__main__":
    glob, sf, _ = check_double_well()
    check_convex()
    pack = check_fig2_toy()
    if "--figures" in sys.argv:
        figures(glob, sf, pack)
