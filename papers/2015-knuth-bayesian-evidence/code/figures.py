#!/usr/bin/env python3
"""Draws the themed SVG pictures used in the Knuth et al. (2015) notes.

These are drawings, not experiments: every curve is a smooth shape chosen to
show an idea, and the notes quote no number read off them.

Run:  python3 figures.py --figures     # rewrite ../figures/*.svg
      python3 figures.py               # does nothing (so `make verify` stays quiet)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures"

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:12px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .ax{stroke:#c3c2b7;stroke-width:0.8;fill:none}
  .a{fill:none;stroke:#2f6fb5;stroke-width:2}
  .b{fill:none;stroke:#c2642a;stroke-width:2}
  .k{fill:none;stroke:#1a1a19;stroke-width:1;stroke-dasharray:3 3}
  .fa{fill:#2f6fb5;opacity:0.18;stroke:none} .fb{fill:#c2642a;opacity:0.18;stroke:none}
  .fg{fill:#8a8880;opacity:0.16;stroke:#8a8880;stroke-width:0.6}
  .dot{fill:#1a1a19;stroke:none}
  @media (prefers-color-scheme: dark){
    .lab{fill:#b6b4ab} .hd{fill:#eceae3} .ax{stroke:#4a4844}
    .a{stroke:#7fb2e8} .b{stroke:#e0955c} .k{stroke:#eceae3}
    .fa{fill:#7fb2e8} .fb{fill:#e0955c} .fg{fill:#b6b4ab;stroke:#b6b4ab} .dot{fill:#eceae3}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n{body}\n</svg>\n")


def path(pts, cls, close=False):
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + (" Z" if close else "")
    return f'<path d="{d}" class="{cls}"/>'


def text(x, y, s, cls="lab", anchor="middle"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def gauss(x, mu, s):
    return math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def occam():
    """Left: prior width vs likelihood width. Right: evidence over possible data sets."""
    b = []
    # --- left panel
    x0, x1, base, top = 40, 360, 230, 70
    b.append(text(200, 28, "One parameter: evidence = Lmax x (delta / Delta)", "hd"))
    b.append(f'<path d="M {x0},{base} H {x1}" class="ax"/>')
    b.append(f'<rect x="{x0+20}" y="{base-34}" width="{x1-x0-40}" height="34" class="fg"/>')
    b.append(text(200, base-14, "prior: flat over the width Delta", "lab"))
    pts = [(x, base - 150 * gauss(x, 220, 14) / gauss(220, 220, 14)) for x in range(x0 + 20, x1 - 19, 2)]
    b.append(path(pts, "a"))
    b.append(path(pts + [(x1 - 20, base), (x0 + 20, base)], "fa", close=True))
    b.append(f'<path d="M {x0+20},{base+22} H {x1-20}" class="k"/>')
    b.append(text(200, base+40, "Delta: everything the prior allowed", "lab"))
    b.append(f'<path d="M 190,{base-165} H 250" class="b"/>')
    b.append(text(220, base-172, "delta: where the data agree", "lab"))
    b.append(text(70, base-155, "Lmax", "lab"))
    b.append(f'<path d="M {x0+20},{base-150} H 150" class="k"/>')
    # --- right panel
    r0, r1, rb = 440, 760, 230
    b.append(text(600, 28, "Each model's evidence over all data sets sums to one", "hd"))
    b.append(f'<path d="M {r0},{rb} H {r1}" class="ax"/>')
    simple = [(x, rb - 190 * gauss(x, 560, 22) / gauss(560, 560, 22)) for x in range(r0, r1 + 1, 2)]
    flex = [(x, rb - 190 * 0.34 * gauss(x, 600, 70) / gauss(600, 600, 70)) for x in range(r0, r1 + 1, 2)]
    b.append(path(simple, "a")); b.append(path(flex, "b"))
    b.append(text(500, rb-170, "simple model", "lab")); b.append(text(690, rb-100, "flexible model", "lab"))
    b.append(f'<path d="M 565,{rb} V {rb-205}" class="k"/>')
    b.append(text(560, rb+18, "d*: simple wins", "lab", "end"))
    b.append(f'<path d="M 690,{rb} V {rb-30}" class="k"/>')
    b.append(text(695, rb+18, "d**: flexible wins", "lab", "start"))
    b.append(text(600, rb+44, "the flexible model spends its probability on many data sets,", "lab"))
    b.append(text(600, rb+60, "so it can never be the best explanation of a typical one", "lab"))
    return svg(800, 290, "Where the Occam factor comes from",
               "Left: a flat prior of width Delta and a likelihood bump of width delta; the evidence is the peak likelihood times delta over Delta. Right: a simple model puts a tall narrow evidence curve over data sets, a flexible one a low wide curve; both have unit area.",
               "\n".join(b))


def thermo():
    b = []
    x0, x1, top, base = 70, 700, 40, 260
    b.append(text(385, 24, "Thermodynamic integration: the area under the average log-likelihood is log Z", "hd"))
    b.append(f'<path d="M {x0},{top} H {x1}" class="ax"/><path d="M {x0},{top} V {base}" class="ax"/>')
    def f(t):  # nondecreasing: average log-likelihood under the tempered posterior
        return top + (base - top) * (1 - t) ** 2.2 * 0.92 + 14
    pts = [(x0 + (x1 - x0) * t / 100, f(t / 100)) for t in range(0, 101)]
    b.append(path([(x0, top)] + pts + [(x1, top)], "fa", close=True))
    steps = 8
    for i in range(steps):
        xa = x0 + (x1 - x0) * i / steps
        xb = x0 + (x1 - x0) * (i + 1) / steps
        y = f(i / steps)
        b.append(f'<rect x="{xa:.1f}" y="{top}" width="{xb-xa:.1f}" height="{y-top:.1f}" class="fg"/>')
    b.append(path(pts, "a"))
    b.append(text(x0-8, top+4, "0", "lab", "end"))
    b.append(text(x0-8, base, "very negative", "lab", "end"))
    b.append(text(x0-48, (top+base)/2, "<tspan>average of log L</tspan>", "lab"))
    b.append(text(x0, base+24, "beta = 0 (prior)", "lab", "start"))
    b.append(text(x1, base+24, "beta = 1 (posterior)", "lab", "end"))
    b.append(text(430, 150, "grey boxes: the finite sum, eq. (49)", "lab"))
    b.append(text(430, 168, "curve never falls (its slope is the variance of log L)", "lab"))
    return svg(780, 300, "Thermodynamic integration",
               "A rising curve of the tempered average of the log-likelihood against inverse temperature beta from 0 to 1, with the area between the curve and zero shaded and approximated by eight rectangles.",
               "\n".join(b))


def nested():
    b = []
    b.append(text(190, 24, "Shrinking the prior volume", "hd"))
    cx, cy = 190, 150
    b.append(f'<rect x="60" y="50" width="260" height="200" class="ax"/>')
    for i, r in enumerate([95, 70, 50, 34, 22, 12]):
        b.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{r*1.25:.0f}" ry="{r:.0f}" class="{"a" if i%2==0 else "k"}"/>')
    for (dx, dy) in [(-30, 10), (5, -20), (30, 5), (-8, 18), (12, 30), (-40, -30)]:
        b.append(f'<circle cx="{cx+dx}" cy="{cy+dy}" r="2.6" class="dot"/>')
    b.append(text(190, 272, "live points; each step discards the lowest-likelihood one", "lab"))
    b.append(text(560, 24, "Z = area under L(X), X = prior mass above a likelihood level", "hd"))
    x0, x1, base, top = 400, 740, 250, 50
    b.append(f'<path d="M {x0},{base} H {x1}" class="ax"/><path d="M {x0},{top} V {base}" class="ax"/>')
    def L(X): return top + (base - top) * (1 - math.exp(-5 * X)) * 0.93
    N = 4
    edges = [math.exp(-i / N) for i in range(0, 14)]
    for xa, xb in zip(edges[1:], edges[:-1]):
        b.append(f'<rect x="{x0+(x1-x0)*xa:.1f}" y="{L(xa):.1f}" width="{(x1-x0)*(xb-xa):.1f}" height="{base-L(xa):.1f}" class="fg"/>')
    pts = [(x0 + (x1 - x0) * t / 100, L(t / 100)) for t in range(1, 101)]
    b.append(path(pts, "a"))
    b.append(text(x0, base+20, "X = 0 (best fit)", "lab", "start")); b.append(text(x1, base+20, "X = 1 (whole prior)", "lab", "end"))
    b.append(text(x0-6, top+4, "high L", "lab", "end"))
    b.append(text(600, 150, "strips are wide on the right and bunch up on the left:", "lab"))
    b.append(text(600, 168, "X shrinks by about the same factor every step", "lab"))
    return svg(780, 290, "Nested sampling",
               "Left: nested elliptical likelihood contours around a peak with live points. Right: the likelihood as a decreasing function of the enclosed prior mass X, approximated by strips whose widths shrink geometrically towards X equals zero.",
               "\n".join(b))


def vb():
    b = []
    b.append(text(280, 24, "log Z = F(Q) + KL(Q || posterior): the bound is tight only when Q is the posterior", "hd"))
    base, top = 250, 60
    b.append(f'<path d="M 70,{top} H 520" class="k"/>')
    b.append(text(60, top+4, "log Z", "lab", "end"))
    b.append(f'<path d="M 70,{base} H 520" class="ax"/>')
    for i, (gap, name) in enumerate([(110, "Q far off"), (55, "Q better"), (0, "Q = posterior")]):
        x = 110 + i * 150
        b.append(f'<rect x="{x}" y="{top+gap}" width="70" height="{base-top-gap}" class="fa"/>')
        if gap:
            b.append(f'<rect x="{x}" y="{top}" width="70" height="{gap}" class="fb"/>')
            b.append(text(x+35, top+gap/2+4, "KL", "lab"))
        b.append(text(x+35, (top+gap+base)/2+4, "F (ELBO)", "lab"))
        b.append(text(x+35, base+20, name, "lab"))
    return svg(580, 290, "The variational Bayes decomposition",
               "Three bars of equal total height log Z, each split into a lower part F, the negative free energy, and an upper part, the KL divergence. The gap shrinks to zero when Q equals the posterior.",
               "\n".join(b))


def importance():
    b = []
    x0, x1 = 60, 720
    b.append(text(390, 22, "Importance sampling: the weight p/q explodes where q is thin", "hd"))
    def X(v): return x0 + (x1 - x0) * (v + 4) / 8
    b.append(f'<path d="M {x0},150 H {x1}" class="ax"/>')
    p = [(X(v / 20), 150 - 110 * gauss(v / 20, 0, 1.25) / gauss(0, 0, 1.25)) for v in range(-80, 81)]
    q = [(X(v / 20), 150 - 110 * gauss(v / 20, 0, 0.75) / gauss(0, 0, 0.75)) for v in range(-80, 81)]
    b.append(path(p, "a")); b.append(path(q, "b"))
    b.append(text(X(-3.2), 100, "target p (wide)", "lab")); b.append(text(X(2.1), 95, "proposal q (narrow)", "lab"))
    b.append(f'<path d="M {x0},300 H {x1}" class="ax"/>')
    r = [(X(v / 20), 300 - 85 * min(1.0, (gauss(v/20, 0, 1.25) / gauss(v/20, 0, 0.75)) / 3.4)) for v in range(-80, 81)]
    b.append(path(r, "k"))
    b.append(text(X(0), 330, "weight p/q (drawn clipped at the top): smallest at the centre, enormous in the tails, where almost no samples land", "lab"))
    return svg(780, 345, "Importance weights",
               "Top: a wide target density and a narrow proposal density. Bottom: the ratio of target to proposal, small near the centre and growing without bound in the tails.",
               "\n".join(b))


if __name__ == "__main__":
    if "--figures" in sys.argv:
        OUT.mkdir(exist_ok=True)
        for name, fn in [("occam", occam), ("thermo", thermo), ("nested", nested), ("vb-gap", vb), ("importance", importance)]:
            (OUT / f"{name}.svg").write_text(fn())
