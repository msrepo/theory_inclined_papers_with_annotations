#!/usr/bin/env python3
"""Draws the static SVG figures for the Hilbert space notes (standard library only).

Writes into ../figures/. It only draws pictures; the notes quote no numbers from it and it
prints nothing.
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures"

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .hd{font-size:14px;font-weight:500;fill:#1a1a19} .t{font-size:13px;fill:#1a1a19}
  .sm{font-size:12px;fill:#57564f} .orn{font-size:13px;fill:#c2571a} .blt{font-size:13px;fill:#2f6fb5} .grt{font-size:13px;fill:#0f7a5a}
  .ax{stroke:#c3c2b7;stroke-width:1;fill:none} .axs{stroke:#8a8880;stroke-width:1;fill:none}
  .dot{fill:#1a1a19} .mut{stroke:#57564f;stroke-width:1.3;fill:none;stroke-dasharray:4 3} .mutf{fill:#57564f}
  .blue{stroke:#2f6fb5;stroke-width:2.4;fill:none} .bluef{fill:#2f6fb5}
  .org{stroke:#c2571a;stroke-width:2.4;fill:none} .orgf{fill:#c2571a}
  .grn{stroke:#0f7a5a;stroke-width:2.4;fill:none} .grnf{fill:#0f7a5a}
  .ballo{fill:#c2571a;fill-opacity:.13;stroke:#c2571a;stroke-width:2}
  .ball1{fill:#7f77dd;fill-opacity:.18;stroke:#534ab7;stroke-width:2.2}
  .ball2{fill:#1d9e75;fill-opacity:.18;stroke:#0f6e56;stroke-width:2.2}
  .ball3{fill:#d85a30;fill-opacity:.15;stroke:#993c1d;stroke-width:2.2}
  .plane{fill:#2f6fb5;fill-opacity:.13;stroke:#2f6fb5;stroke-opacity:.6;stroke-width:1}
  .ra{stroke:#57564f;stroke-width:1.2;fill:none}
  .pos{fill:#2f6fb5;fill-opacity:.35} .neg{fill:#c2571a;fill-opacity:.35}
  .curve{fill:none;stroke:#1a1a19;stroke-width:2.2} .bg{fill:#f4f3ee}
  @media (prefers-color-scheme: dark){
    .hd,.t{fill:#eceae3} .sm{fill:#b6b4ab} .orn{fill:#f0a070} .blt{fill:#7fb2e8} .grt{fill:#4cc79a}
    .ax{stroke:#4a4844} .axs{stroke:#85837b} .dot{fill:#eceae3} .mut{stroke:#b6b4ab} .mutf{fill:#b6b4ab}
    .blue{stroke:#7fb2e8} .bluef{fill:#7fb2e8} .org{stroke:#f0a070} .orgf{fill:#f0a070}
    .grn{stroke:#4cc79a} .grnf{fill:#4cc79a}
    .ballo{fill:#f0a070;stroke:#f0a070} .ball1{fill:#b79df0;stroke:#b79df0} .ball2{fill:#4cc79a;stroke:#4cc79a} .ball3{fill:#f0a070;stroke:#f0a070}
    .plane{fill:#7fb2e8;stroke:#7fb2e8} .ra{stroke:#b6b4ab} .pos{fill:#7fb2e8} .neg{fill:#f0a070}
    .curve{stroke:#eceae3} .bg{fill:#23262b}
  }
</style>"""
DEFS = ('<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')


def f(v):
    return f"{v:.1f}"


def line(x1, y1, x2, y2, cls, extra=""):
    return f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" class="{cls}" {extra}/>'


def arr(x1, y1, x2, y2, cls):
    return line(x1, y1, x2, y2, cls, 'marker-end="url(#a)"')


def circ(x, y, r, cls):
    return f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}" class="{cls}"/>'


def text(x, y, s, cls, anchor="start"):
    return f'<text x="{f(x)}" y="{f(y)}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def poly(pts, cls):
    return f'<polygon points="{" ".join(f(a)+","+f(b) for a, b in pts)}" class="{cls}"/>'


def path(pts, cls, close=False):
    d = "M" + " L".join(f"{f(a)} {f(b)}" for a, b in pts) + (" Z" if close else "")
    return f'<path d="{d}" class="{cls}"/>'


def write(name, w, h, title, desc, body):
    svg = (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">'
           f'<title>{title}</title><desc>{desc}</desc>{STYLE}{DEFS}{"".join(body)}</svg>\n')
    (OUT / name).write_text(svg)


def no_holes():
    g = [text(30, 28, "Not complete: the rational numbers (fractions)", "hd")]
    xs = [70, 300, 430, 490, 515]
    labs = ["1", "1.4", "1.41", "1.414", ""]
    for yl, hole in ((110, True), (310, False)):
        if not hole:
            g.append(text(30, 228, "Complete: the real numbers", "hd"))
        g.append(line(40, yl, 640, yl, "axs"))
        for x in xs:
            g.append(circ(x, yl, 5, "dot"))
        for a, b in zip(xs, xs[1:]):
            mid = (a + b) / 2
            g.append(f'<path d="M{a} {yl-12} Q{mid} {yl-12-(b-a)*0.25:.0f} {b} {yl-12}" class="mut" marker-end="url(#a)" style="stroke-dasharray:none"/>')
        if hole:
            for x, s in zip(xs, labs):
                if s:
                    g.append(text(x, yl + 26, s, "sm", "middle"))
            g.append(f'<circle cx="528" cy="{yl}" r="7" class="org" style="fill:none"/>')
            g.append(text(528, yl + 50, "a hole: the point they close in on", "orn", "middle"))
            g.append(text(528, yl + 67, "(√2) is not a fraction", "orn", "middle"))
        else:
            g.append(f'<circle cx="528" cy="{yl}" r="7" class="grnf"/>')
            g.append(text(528, yl + 38, "the limit is here, inside the space", "grt", "middle"))
    g.append(line(30, 200, 650, 200, "ax"))
    g.append(text(30, 352, "Spacing not to scale.", "sm"))
    write("no-holes.svg", 680, 360, "Completeness: a sequence that closes in on a point",
          "Top: on the rational numbers a sequence closes in on a spot that is not in the space, a hole. Bottom: on the real numbers the same sequence has its limit inside the space.", g)


def projection():
    g = []
    g.append(poly([(68.5, 279.6), (350.5, 279.6), (418, 231.6), (136, 231.6)], "plane"))
    g.append(text(90, 300, "subspace: the vectors you are allowed to use", "sm"))
    g.append(arr(100, 270, 268, 132.7, "ax"))
    g.append(line(100, 270, 268, 132.7, "mut", 'style="stroke-dasharray:none;stroke:#8a8880"'))
    g.append(arr(100, 270, 268, 252.7, "blue"))
    g.append(arr(268, 252.7, 268, 132.7, "org"))
    g.append(f'<polyline points="268,252.7 279,252.7 279,241.7 268,241.7" class="ra"/>')
    g.append(line(268, 252.7, 321, 266, "mut"))
    g.append(line(268, 132.7, 321, 266, "mut"))
    g.append(circ(100, 270, 4, "dot"))
    g.append(circ(268, 132.7, 5, "dot"))
    g.append(circ(268, 252.7, 5, "bluef"))
    g.append(circ(321, 266, 5, "mutf"))
    g.append(text(278, 130, "b  (the target)", "t"))
    g.append(text(180, 252, "p", "blt"))
    g.append(text(274, 190, "r = b − p", "orn"))
    g.append(text(328, 270, "q", "sm"))
    g.append(text(86, 266, "0", "sm", "end"))
    g.append(text(450, 42, "What the picture says", "hd"))
    rows = [("1. p is the closest point to b", "t"), ("inside the subspace.", "sm"), ("", ""),
            ("2. The miss r meets the subspace", "t"), ("at a right angle. That right angle", "sm"), ("is what makes p the closest.", "sm"), ("", ""),
            ("3. Pythagoras: any other q in the", "t"), ("subspace is farther from b than p,", "sm"), ("since b to q is the hypotenuse of a", "sm"), ("right triangle with r as one side.", "sm")]
    y = 72
    for s, c in rows:
        if s:
            g.append(text(450, y, s, c))
        y += 18
    write("projection.svg", 680, 330, "Projection onto a subspace",
          "A target vector b above a plane. Its projection p on the plane is the closest point, and the residual from p to b is at a right angle to the plane. Another point q in the plane is farther from b.", g)


def unit_balls():
    g = []
    spec = [(120, "Sum of absolute values", "length = |x| + |y|", "a diamond, corners on the axes", "ball1", (155, 95)),
            (340, "Ordinary length", "length = √(x² + y²)", "a circle, the only one with angles", "ball2", (389.5, 80.5)),
            (560, "Largest coordinate", "length = max(|x|, |y|)", "a square, flat sides", "ball3", (630, 60))]
    for cx, title, l1, l2, cls, dot in spec:
        g.append(line(cx - 90, 130, cx + 90, 130, "ax"))
        g.append(line(cx, 40, cx, 220, "ax"))
        if cls == "ball1":
            g.append(poly([(cx, 60), (cx + 70, 130), (cx, 200), (cx - 70, 130)], cls))
        elif cls == "ball2":
            g.append(circ(cx, 130, 70, cls))
        else:
            g.append(f'<rect x="{cx-70}" y="60" width="140" height="140" class="{cls}"/>')
        g.append(line(cx, 130, dot[0], dot[1], "mut"))
        g.append(circ(dot[0], dot[1], 5, "dot"))
        g.append(text(cx, 248, title, "hd", "middle"))
        g.append(text(cx, 266, l1, "sm", "middle"))
        g.append(text(cx, 284, l2, "sm", "middle"))
    g.append(text(340, 316, "Each shape is every point at length 1. The dot is the diagonal direction, reaching the edge at a different spot in each.", "sm", "middle"))
    write("unit-balls.svg", 680, 330, "Unit balls for three different lengths in the plane",
          "All points at length 1 from the origin: a diamond for the sum of absolute values, a circle for the ordinary length, a square for the largest coordinate.", g)


def closest_point():
    S = 50
    g = []

    def panel(px, py, mode):
        X = lambda x: px + 35 + S * x
        Y = lambda y: py + 175 - S * y
        g.append(line(X(-0.6), Y(0), X(3.3), Y(0), "ax"))
        g.append(line(X(0), Y(-0.4), X(0), Y(3.6), "ax"))
        g.append(line(X(-0.6), Y(-0.3), X(3.3), Y(1.65), "blue"))
        b = (1, 2)
        if mode == "l1":
            r = 1.5
            g.append(poly([(X(b[0] + r), Y(b[1])), (X(b[0]), Y(b[1] + r)), (X(b[0] - r), Y(b[1])), (X(b[0]), Y(b[1] - r))], "ballo"))
            p = (1, 0.5)
        elif mode == "l2":
            g.append(circ(X(b[0]), Y(b[1]), S * math.sqrt(1.8), "ballo"))
            p = (1.6, 0.8)
        else:
            g.append(f'<rect x="{f(X(0))}" y="{f(Y(3))}" width="{S*2}" height="{S*2}" class="ballo"/>')
            p = (2, 1)
        g.append(line(X(b[0]), Y(b[1]), X(p[0]), Y(p[1]), "mut"))
        g.append(circ(X(b[0]), Y(b[1]), 5, "dot"))
        g.append(circ(X(p[0]), Y(p[1]), 6, "bluef"))
        g.append(text(X(b[0]) + 8, Y(b[1]) - 8, "b", "sm"))

    panel(10, 30, "l1")
    panel(230, 30, "l2")
    panel(450, 30, "li")
    for cx, t1, t2 in ((120, "Diamond (sum of absolute values)", "touches the line at its corner"),
                       (340, "Circle (ordinary length)", "touches at the perpendicular foot"),
                       (560, "Square (largest coordinate)", "touches at its corner")):
        g.append(text(cx, 255, t1, "hd", "middle"))
        g.append(text(cx, 273, t2, "sm", "middle"))
    g.append(text(340, 300, "Same line, same target b. The ball grows until it first touches the line; that point is the closest.", "sm", "middle"))
    g.append(line(30, 318, 650, 318, "ax"))
    px, py = 60, 335
    X = lambda x: px + 35 + S * x
    Y = lambda y: py + 130 - S * y
    g.append(line(X(-0.5), Y(0), X(4), Y(0), "ax"))
    g.append(f'<rect x="{f(X(0.2))}" y="{f(Y(2))}" width="{S*2}" height="{S*2}" class="ballo"/>')
    g.append(f'<line x1="{f(X(0.2))}" y1="{f(Y(0))}" x2="{f(X(2.2))}" y2="{f(Y(0))}" class="blue" style="stroke-width:5;stroke-linecap:round"/>')
    g.append(circ(X(1.2), Y(1), 5, "dot"))
    g.append(text(X(1.2) + 8, Y(1) - 8, "b", "sm"))
    g.append(text(px + 35 + S * 1.2, py + 168, "Square on a flat line", "hd", "middle"))
    tx = 360
    g.append(text(tx, 370, "When the ball is flat against the line", "hd"))
    notes = ["The square lies flat on the line, so it touches along a",
             "whole stretch, not one point. Every point of that stretch",
             "is equally close to b: the closest point is not unique.", "",
             "A round ball always touches in exactly one place, the",
             "foot of the perpendicular. That is why projection onto",
             "a subspace is well defined in a Hilbert space."]
    for i, s in enumerate(notes):
        if s:
            g.append(text(tx, 395 + i * 18, s, "sm"))
    write("closest-point.svg", 680, 545, "Closest point on a line under three different lengths",
          "A target point above a tilted line. Growing a ball around it until it first touches the line gives the closest point. The diamond, circle and square touch at three different points. A fourth panel shows a square resting flat on a horizontal line, which touches along a whole segment.", g)


def sine_orthogonal():
    L, W = 70, 560
    X = lambda x: L + W * x / math.pi
    g = []

    def panel(cy, amp, fn, label, sub, shade):
        g.append(line(L, cy, L + W, cy, "axs"))
        for xv in (0, math.pi / 2, math.pi):
            g.append(line(X(xv), cy - 3, X(xv), cy + 3, "axs"))
        N = 200
        if shade:
            h = math.pi / 2
            pu = [(X(0), cy)] + [(X(h * i / N), cy - amp * fn(h * i / N)) for i in range(N + 1)] + [(X(h), cy)]
            pd = [(X(h), cy)] + [(X(h + h * i / N), cy - amp * fn(h + h * i / N)) for i in range(N + 1)] + [(X(math.pi), cy)]
            g.append(path(pu, "pos", True))
            g.append(path(pd, "neg", True))
        pts = [(X(math.pi * i / N), cy - amp * fn(math.pi * i / N)) for i in range(N + 1)]
        g.append(path(pts, "curve"))
        g.append(text(L - 10, cy + 4, label, "hd", "end"))
        if sub:
            g.append(text(L + W, cy - amp - 8, sub, "sm", "end"))

    panel(70, 45, math.sin, "sin x", "one hump", False)
    panel(185, 45, lambda x: math.sin(2 * x), "sin 2x", "a hump, then a dip", False)
    panel(325, 75, lambda x: math.sin(x) * math.sin(2 * x), "product", "", True)
    g.append(text(X(math.pi / 4) + 10, 392, "area above: positive", "blt", "middle"))
    g.append(text(X(3 * math.pi / 4), 392, "area below: negative", "orn", "middle"))
    g.append(text(L + W / 2, 418, "Multiply the two waves at every point. The positive and negative areas are equal,", "t", "middle"))
    g.append(text(L + W / 2, 436, "so they add to zero: the inner product is 0, and the two functions are perpendicular.", "t", "middle"))
    write("sine-orthogonal.svg", 680, 450, "sin x and sin 2x are perpendicular",
          "Three stacked graphs on the interval from 0 to pi: sin x, sin 2x, and their product. The product has equal positive and negative areas, so the inner product is zero.", g)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (no_holes, projection, unit_balls, closest_point, sine_orthogonal):
        fn()
