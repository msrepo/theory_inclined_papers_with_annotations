#!/usr/bin/env python3
"""Draws figures/concrete-subspaces.svg: the four subspaces of the running 3x2 matrix.

A = [[1,0],[1,1],[1,2]]. Input space R^2 (left): the row space is all of it and the null
space is only the origin. Output space R^3 (right): the column space is the plane spanned
by (1,1,1) and (0,1,2), and the residual space is the line along (1,-2,1) through the
origin, perpendicular to that plane. Standard library only; it prints nothing and the
notes quote no numbers from it.
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures" / "concrete-subspaces.svg"
S = 46.0

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .hd{font-size:13px;font-weight:500;fill:#1a1a19} .sm{font-size:11px;fill:#57564f} .lab{font-size:12px;fill:#57564f}
  .ax{stroke:#c3c2b7;stroke-width:0.8;fill:none} .axf{fill:#c3c2b7}
  .cs{stroke:#2f6fb5;stroke-width:1.8;fill:none} .csf{fill:#2f6fb5} .cst{fill:#2f6fb5}
  .ns{stroke:#c2571a;stroke-width:2.4;fill:none} .nsf{fill:#c2571a} .nst{fill:#c2571a}
  .plane{fill:#2f6fb5;opacity:.12;stroke:#2f6fb5;stroke-opacity:.55;stroke-width:0.9}
  .box{fill:#f4f3ee;stroke:#c3c2b7;stroke-width:0.8}
  .ra{stroke:#57564f;stroke-width:0.9;fill:none} .ar{stroke:#57564f;stroke-width:1.3;fill:none;stroke-dasharray:5 4}
  .arf{fill:#57564f}
  @media (prefers-color-scheme: dark){
    .hd{fill:#eceae3} .sm,.lab{fill:#b6b4ab} .ax{stroke:#4a4844} .axf{fill:#4a4844}
    .cs{stroke:#7fb2e8} .csf,.cst{fill:#7fb2e8} .ns{stroke:#f0a070} .nsf,.nst{fill:#f0a070}
    .plane{fill:#7fb2e8;opacity:.16;stroke:#7fb2e8} .box{fill:#23262b;stroke:#4a4844}
    .ra{stroke:#b6b4ab} .ar{stroke:#b6b4ab} .arf{fill:#b6b4ab}
  }
</style>"""


def proj(ox, oy):
    def p(x, y, z):
        return ox + S * (0.95 * y - 0.55 * x), oy - S * z + S * 0.35 * x
    return p


def f(v):
    return f"{v:.1f}"


def poly(pts, cls):
    return f'<polygon points="{" ".join(f(a)+","+f(b) for a, b in pts)}" class="{cls}"/>'


def line(a, b, cls):
    return f'<line x1="{f(a[0])}" y1="{f(a[1])}" x2="{f(b[0])}" y2="{f(b[1])}" class="{cls}"/>'


def arrow(a, b, cls, fcls, head=8):
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    tip = b
    base = (tip[0] - head * math.cos(ang), tip[1] - head * math.sin(ang))
    w = head * 0.42
    l = (base[0] + w * math.sin(ang), base[1] - w * math.cos(ang))
    r = (base[0] - w * math.sin(ang), base[1] + w * math.cos(ang))
    return line(a, base, cls) + poly([tip, l, r], fcls)


def text(x, y, s, cls, anchor="start"):
    return f'<text x="{f(x)}" y="{f(y)}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def build():
    g = []
    g.append(text(20, 24, "Input space ℝ²: what you feed in", "hd"))
    g.append(text(380, 24, "Output space ℝ³: what comes out", "hd"))

    # left: the plane z=0 drawn as ℝ² (x1 toward viewer-left, x2 to the right)
    pl = proj(150, 215)
    g.append(poly([pl(1.5, -1.6, 0), pl(1.5, 2.0, 0), pl(-1.5, 2.0, 0), pl(-1.5, -1.6, 0)], "plane"))
    g.append(arrow(pl(0, -1.6, 0), pl(0, 2.1, 0), "ax", "axf"))
    g.append(arrow(pl(1.6, 0, 0), pl(-1.6, 0, 0), "ax", "axf"))
    g.append(text(pl(0, 2.1, 0)[0] + 6, pl(0, 2.1, 0)[1] + 4, "x₂", "sm"))
    g.append(text(pl(1.6, 0, 0)[0] - 4, pl(1.6, 0, 0)[1] + 12, "x₁", "sm", "end"))
    o = pl(0, 0, 0)
    g.append(f'<circle cx="{f(o[0])}" cy="{f(o[1])}" r="5.5" class="nsf"/>')
    g.append(text(20, 52, "row space row(A) = the whole plane", "sm cst"))
    g.append(text(20, 67, "every input direction is “seen” (dimension 2)", "sm"))
    g.append(text(o[0] + 14, o[1] + 34, "null space N(A) = {0}", "sm nst"))
    g.append(text(o[0] + 14, o[1] + 49, "only the origin: no other input gives 0", "sm"))

    # right: R^3 viewed so the column plane is the floor and the residual line stands on it.
    # Coordinates (p, q, h) are the ambient vector's components along the orthonormal
    # directions u = a1/|a1|, v = (-1,0,1)/sqrt2 (both in the plane) and n = (1,-2,1)/sqrt6.
    R3, R2 = math.sqrt(3), math.sqrt(2)
    s2 = 40.0
    ox, oy = 490, 215

    def pr(p, q, h):
        return ox + s2 * (p + 0.45 * q), oy - s2 * h - s2 * 0.5 * q

    corners = [pr(-0.9, -0.9, 0), pr(2.5, -0.9, 0), pr(2.5, 1.9, 0), pr(-0.9, 1.9, 0)]
    g.append(poly(corners, "plane"))
    g.append(line(pr(0, 0, -1.2), pr(0, 0, 0), "ns"))
    g.append(arrow(pr(0, 0, 0), pr(R3, 0, 0), "cs", "csf", 9))
    g.append(arrow(pr(0, 0, 0), pr(R3, R2, 0), "cs", "csf", 9))
    g.append(arrow(pr(0, 0, 0), pr(0, 0, 2.0), "ns", "nsf", 9))
    g.append(f'<polyline points="{f(pr(0.28,0,0)[0])},{f(pr(0.28,0,0)[1])} {f(pr(0.28,0,0.28)[0])},{f(pr(0.28,0,0.28)[1])} {f(pr(0,0,0.28)[0])},{f(pr(0,0,0.28)[1])}" class="ra"/>')
    pa1, pa2, pn = pr(R3, 0, 0), pr(R3, R2, 0), pr(0, 0, 2.0)
    g.append(text(pa1[0] + 6, pa1[1] + 14, "a₁ = (1,1,1)", "sm cst"))
    g.append(text(pa2[0] + 6, pa2[1] - 2, "a₂ = (0,1,2)", "sm cst"))
    g.append(text(pn[0] + 8, pn[1] + 4, "(1, −2, 1)", "sm nst"))
    g.append(text(380, 52, "column space col(A) = the plane spanned by a₁, a₂", "sm cst"))
    g.append(text(380, 67, "every output A can make (dimension 2)", "sm"))
    g.append(text(380, 300, "residual space N(Aᵀ) = the line along (1, −2, 1)", "sm nst"))
    g.append(text(380, 315, "through the origin, perpendicular to the plane (dim 1)", "sm nst"))
    g.append(text(380, 338, "axes turned so the plane lies flat; angles are true", "sm"))

    # the map
    g.append(arrow((278, 160), (352, 160), "ar", "arf", 8))
    g.append(text(315, 150, "A", "hd", "middle"))
    g.append(text(315, 178, "plane onto plane", "sm", "middle"))
    return g


svg = ('<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" role="img">'
       '<title>The four subspaces of the 3×2 running example</title>'
       '<desc>Left: input space R^2. The row space is the whole plane (dimension 2) and the null space is only the origin. '
       'Right: output space R^3. The column space is the plane spanned by a1=(1,1,1) and a2=(0,1,2), and the residual space is the line along (1,-2,1) through the origin, perpendicular to that plane. '
       'An arrow labelled A carries the input plane one to one onto the column space plane.</desc>'
       + STYLE + "".join(build()) + "</svg>\n")
OUT.write_text(svg)
