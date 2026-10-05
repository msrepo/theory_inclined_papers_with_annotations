#!/usr/bin/env python3
"""Draws the static SVG figures for the Sinkhorn / entropic-OT part of the optimal transport notes.

Writes into ../figures/. It only draws pictures; the notes quote no numbers from it and it
prints nothing. Needs numpy.
"""
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures"

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .hd{font-size:14px;font-weight:500;fill:#1a1a19} .t{font-size:13px;fill:#1a1a19}
  .sm{font-size:12px;fill:#57564f} .orn{font-size:13px;fill:#c2571a} .blt{font-size:13px;fill:#2f6fb5} .grt{font-size:13px;fill:#0f7a5a}
  .ax{stroke:#c3c2b7;stroke-width:1;fill:none} .axs{stroke:#8a8880;stroke-width:1;fill:none}
  .mut{stroke:#57564f;stroke-width:1.3;fill:none;stroke-dasharray:4 3}
  .cb{fill:#2f6fb5} .tick{stroke:#1a1a19;stroke-width:2.2} .barr{fill:#c2571a;fill-opacity:.55} .barc{fill:#0f7a5a;fill-opacity:.55}
  .l1{stroke:#7f77dd;stroke-width:2.4;fill:none} .l2{stroke:#2f6fb5;stroke-width:2.4;fill:none}
  .l3{stroke:#0f7a5a;stroke-width:2.4;fill:none} .l4{stroke:#c2571a;stroke-width:2.4;fill:none}
  .ex{stroke:#1a1a19;stroke-width:1.6;fill:none;stroke-dasharray:5 3}
  @media (prefers-color-scheme: dark){
    .hd,.t{fill:#eceae3} .sm{fill:#b6b4ab} .orn{fill:#f0a070} .blt{fill:#7fb2e8} .grt{fill:#4cc79a}
    .ax{stroke:#4a4844} .axs{stroke:#85837b} .mut{stroke:#b6b4ab}
    .cb{fill:#7fb2e8} .tick{stroke:#eceae3} .barr{fill:#f0a070} .barc{fill:#4cc79a}
    .l1{stroke:#b79df0} .l2{stroke:#7fb2e8} .l3{stroke:#4cc79a} .l4{stroke:#f0a070} .ex{stroke:#eceae3}
  }
</style>"""
DEFS = ('<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')


def f(v):
    return f"{v:.1f}"


def text(x, y, s, cls, anchor="start"):
    return f'<text x="{f(x)}" y="{f(y)}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def line(x1, y1, x2, y2, cls, extra=""):
    return f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" class="{cls}" {extra}/>'


def rect(x, y, w, h, cls, extra=""):
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" class="{cls}" {extra}/>'


def poly(xs, ys, cls):
    return f'<polyline points="{" ".join(f(a)+","+f(b) for a, b in zip(xs, ys))}" class="{cls}"/>'


def write(name, w, h, title, desc, body):
    svg = (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">'
           f'<title>{title}</title><desc>{desc}</desc>{STYLE}{DEFS}{"".join(body)}</svg>\n')
    (OUT / name).write_text(svg)


def heat(x0, y0, M, cell, vmax, flip=False):
    """Cells of matrix M (row = from, column = to). With flip, the 'to' axis points up."""
    n, m = M.shape
    g = [rect(x0, y0, m * cell, n * cell, "ax")]
    for i in range(n):
        for j in range(m):
            v = M[i, j] / vmax
            if v < 0.01:
                continue
            x, y = (x0 + i * cell, y0 + (m - 1 - j) * cell) if flip else (x0 + j * cell, y0 + i * cell)
            g.append(rect(x, y, cell, cell, "cb", f'fill-opacity="{min(1.0, v) ** 0.6:.2f}"'))
    return g


# ---------------- Sinkhorn, log domain, numpy ----------------
def lse(z, axis):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def potentials(a, b, C, eps, iters):
    la, lb = np.log(a), np.log(b)
    fp, gp = np.zeros(len(a)), np.zeros(len(b))
    for _ in range(iters):
        fp = -eps * lse(lb[None, :] + (gp[None, :] - C) / eps, 1)
        gp = -eps * lse(la[:, None] + (fp[:, None] - C) / eps, 0)
    return fp, gp


def plan_of(a, b, C, eps, iters=3000):
    fp, gp = potentials(a, b, C, eps, iters)
    return a[:, None] * b[None, :] * np.exp((fp[:, None] + gp[None, :] - C) / eps)


def ot_eps(a, b, C, eps, iters=800):
    fp, gp = potentials(a, b, C, eps, iters)
    return (a * fp).sum() + (b * gp).sum()


def monotone_plan(a, b):
    """Exact optimal plan in 1-D for squared cost: match by sorted order (north-west corner rule)."""
    P = np.zeros((len(a), len(b)))
    a, b, i, j = a.copy(), b.copy(), 0, 0
    while i < len(a) and j < len(b):
        m = min(a[i], b[j]); P[i, j] += m; a[i] -= m; b[j] -= m
        if a[i] < 1e-15: i += 1
        if b[j] < 1e-15: j += 1
    return P


# ---------------- figure 1: alternate rescaling ----------------
def scaling():
    n, eps = 5, 0.08
    pos = np.arange(n) / (n - 1)
    C = (pos[:, None] - pos[None, :]) ** 2
    a = np.array([.35, .30, .15, .10, .10]); b = np.array([.10, .10, .15, .30, .35])
    K = np.exp(-C / eps)
    u, v = np.ones(n), np.ones(n)
    snaps = [("start: K", K / K.sum())]
    u = a / (K @ v); snaps.append(("rescale rows", u[:, None] * K * v[None, :]))
    v = b / (K.T @ u); snaps.append(("rescale columns", u[:, None] * K * v[None, :]))
    for _ in range(500):
        u = a / (K @ v); v = b / (K.T @ u)
    snaps.append(("after many rounds", u[:, None] * K * v[None, :]))
    cell, g = 20, [text(30, 28, "Sinkhorn: rescale the rows, then the columns, and repeat", "hd")]
    for k, (title, M) in enumerate(snaps):
        x0, y0 = 30 + k * 160, 74
        g += heat(x0, y0, M, cell, 0.30)
        g.append(text(x0 + n * cell / 2, y0 - 12, title, "t", "middle"))
        for i in range(n):                                   # row sums (orange) vs target a (tick)
            L = M[i].sum() * 80
            g.append(rect(x0 + n * cell + 6, y0 + i * cell + 4, L, 12, "barr"))
            g.append(line(x0 + n * cell + 6 + a[i] * 80, y0 + i * cell + 2, x0 + n * cell + 6 + a[i] * 80, y0 + i * cell + 18, "tick"))
        for j in range(n):                                   # column sums (green) vs target b (tick)
            L = M[:, j].sum() * 80
            g.append(rect(x0 + j * cell + 4, y0 + n * cell + 6, 12, L, "barc"))
            g.append(line(x0 + j * cell + 2, y0 + n * cell + 6 + b[j] * 80, x0 + j * cell + 18, y0 + n * cell + 6 + b[j] * 80, "tick"))
    g.append(text(30, 232, "Orange bars: current row sums.  Green bars: current column sums.", "sm"))
    g.append(text(30, 250, "Dark tick: the target (the mass of P for rows, the mass of Q for columns).", "sm"))
    g.append(text(30, 268, "Each step makes one set of sums exact and disturbs the other; the disturbance shrinks every round.", "sm"))
    write("sinkhorn-scaling.svg", 680, 282, "Sinkhorn alternately rescales rows and columns",
          "Four heatmaps of a small table. Starting from a kernel that is large near the diagonal, rescaling the rows "
          "makes the row sums match the targets, rescaling the columns then makes the column sums match, and after "
          "many rounds both match.", g)


# ---------------- figure 2: the plan for shrinking epsilon ----------------
def bumpy():
    n = 24
    grid = np.arange(n)
    gs = lambda m, s: np.exp(-0.5 * ((grid - m) / s) ** 2)
    a = gs(4, 1.8) + 0.7 * gs(10, 1.6); a /= a.sum()
    b = 0.6 * gs(9, 1.9) + gs(18, 2.1); b /= b.sum()
    pos = grid / (n - 1)
    return a, b, (pos[:, None] - pos[None, :]) ** 2


def epsilon_plans():
    a, b, C = bumpy()
    n = len(a); cell = 4.2
    panels = [("large ε", plan_of(a, b, C, 0.5)), ("smaller ε", plan_of(a, b, C, 0.05)),
              ("small ε", plan_of(a, b, C, 0.01)), ("tiny ε", plan_of(a, b, C, 0.002)),
              ("exact plan (ε = 0)", monotone_plan(a, b))]
    vmax = max(P.max() for _, P in panels[1:])
    g = [text(24, 26, "The entropic plan sharpens into the exact optimal plan as ε shrinks", "hd")]
    for k, (title, P) in enumerate(panels):
        x0, y0 = 24 + k * 128, 62
        g += heat(x0, y0, P, cell, vmax, flip=True)
        g.append(text(x0 + n * cell / 2, y0 - 10, title, "t", "middle"))
        g.append(text(x0 + n * cell / 2, y0 + n * cell + 16, "from (P) →", "sm", "middle"))
    g.append(text(24, 62 + n * cell + 38, "Vertical axis: where the mass goes in Q (up = larger position).", "sm"))
    g.append(text(24, 62 + n * cell + 56, "Large ε spreads each source over many destinations; as ε → 0 the plan becomes a thin curve.", "sm"))
    write("sinkhorn-epsilon.svg", 680, 62 + int(n * cell) + 70, "The Sinkhorn plan for decreasing epsilon",
          "Five heatmaps of a transport plan between two bumpy histograms. For large epsilon the plan is a set of "
          "blurry blobs; as epsilon shrinks it concentrates along a thin monotone curve that matches the exact "
          "optimal plan shown last.", g)


# ---------------- figure 3: epsilon as a temperature ----------------
def temperature():
    x = np.linspace(-6, 10, 80)
    C = (x[:, None] - x[None, :]) ** 2
    nm = lambda m: (lambda w: w / w.sum())(np.exp(-0.5 * (x - m) ** 2))
    a, b = nm(0), nm(4)
    i0 = int(np.argmin(abs(x - 0)))
    x0, y0, W, H = 60, 50, 560, 190
    g = [text(30, 26, "ε acts like a temperature: where does the mass at one source point go?", "hd"),
         line(x0, y0 + H, x0 + W, y0 + H, "axs"), line(x0, y0, x0, y0 + H, "axs")]
    curves = [(0.05, "l1"), (0.5, "l2"), (2.0, "l3"), (8.0, "l4")]
    rows = [plan_of(a, b, C, e, 1500)[i0] for e, _ in curves]
    top = max(r.max() / r.sum() for r in rows)
    for (e, cls), r in zip(curves, rows):
        r = r / r.sum()
        g.append(poly(x0 + (x + 6) / 16 * W, y0 + H - r / top * (H - 8), cls))
    g.append(text(x0 + W / 2, y0 + H + 40, "destination position in Q", "sm", "middle"))
    g.append(text(x0 - 38, y0 + H / 2, "share of the mass", "sm", "middle").replace("<text", f'<text transform="rotate(-90 {x0-38} {y0+H/2})"', 1))
    g.append(line(x0 + 6 / 16 * W, y0 + H, x0 + 6 / 16 * W, y0 + H + 6, "axs"))
    g.append(text(x0 + 6 / 16 * W, y0 + H + 19, "source point", "sm", "middle"))
    g.append(text(x0 + 14, y0 + 20, "Sharpest curve: smallest ε (almost one destination).", "sm"))
    g.append(text(x0 + 14, y0 + 38, "Widest curve: largest ε (spread over many).", "sm"))
    write("sinkhorn-temperature.svg", 680, 310, "How epsilon spreads the mass from one source point",
          "Four curves showing where the mass at one source point is sent in the target distribution. The smallest "
          "epsilon concentrates it near one destination; larger epsilon spreads it over a wide range.", g)


# ---------------- figure 4: removing the bias ----------------
def divergence():
    x = np.linspace(-6, 10, 80)
    C = (x[:, None] - x[None, :]) ** 2
    nm = lambda m: (lambda w: w / w.sum())(np.exp(-0.5 * (x - m) ** 2))
    shifts = np.linspace(0, 5, 11)
    a0 = nm(0)
    out = {}
    for e in (0.5, 5.0):
        oaa = ot_eps(a0, a0, C, e)
        plain, div = [], []
        for s in shifts:
            bs = nm(s)
            oab = ot_eps(a0, bs, C, e)
            plain.append(oab); div.append(oab - 0.5 * oaa - 0.5 * ot_eps(bs, bs, C, e))
        out[e] = (np.array(plain), np.array(div))
    ymax = shifts.max() ** 2 * 1.04
    g = [text(30, 26, "Plain entropic OT is biased; subtracting the two self-terms removes the bias", "hd")]
    for p, (title, which) in enumerate([("plain entropic OT", 0), ("Sinkhorn divergence", 1)]):
        x0, y0, W, H = 60 + p * 320, 70, 270, 190
        g += [text(x0 + W / 2, y0 - 16, title, "t", "middle"),
              line(x0, y0 + H, x0 + W, y0 + H, "axs"), line(x0, y0, x0, y0 + H, "axs"),
              text(x0 + W / 2, y0 + H + 24, "shift between the two distributions", "sm", "middle")]
        g.append(poly(x0 + shifts / shifts.max() * W, y0 + H - shifts ** 2 / ymax * H, "ex"))
        for e, cls in ((0.5, "l2"), (5.0, "l4")):
            g.append(poly(x0 + shifts / shifts.max() * W, y0 + H - out[e][which] / ymax * H, cls))
    g.append(text(60, 308, "Dashed: the exact squared Wasserstein distance (the squared shift).  Blue: small ε.  Orange: large ε.", "sm"))
    g.append(text(60, 326, "Left: at zero shift the curves start above zero, and the offset grows with ε.  Right: both start at zero and lie on the dashed curve.", "sm"))
    write("sinkhorn-divergence.svg", 680, 344, "Sinkhorn divergence removes the entropic bias",
          "Two plots against the shift between two Gaussians. Left, plain entropic optimal transport lies above the exact "
          "curve and does not start at zero, more so for larger epsilon. Right, the Sinkhorn divergence starts at zero.", g)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    scaling(); epsilon_plans(); temperature(); divergence()
