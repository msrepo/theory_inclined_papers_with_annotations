"""Draws the static SVG figures for the Lagrange multiplier notes (standard library only).

Run from anywhere: it writes next to this folder, into ../figures/. It only draws
pictures; the notes quote no numbers from it.
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures"

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .t{font-size:13px;fill:#1a1a19} .sm{font-size:12px;fill:#57564f}
  .bg{fill:#f1efe8} .ct{fill:none;stroke:#b4b2a9;stroke-width:1} .trail{fill:none;stroke:#534ab7;stroke-width:2.5}
  .hot{fill:none;stroke:#eb6834;stroke-width:2} .gf{stroke:#eb6834;stroke-width:2.2} .gt{stroke:#534ab7;stroke-width:2.2}
  .ax{stroke:#8a8880;stroke-width:1} .dot{fill:#1a1a19}
  @media (prefers-color-scheme: dark){
    .t{fill:#eceae3} .sm{fill:#b6b4ab} .bg{fill:#2a2926} .ct{stroke:#5a5953} .trail{stroke:#8f88e6}
    .hot{stroke:#f0997b} .gf{stroke:#f0997b} .gt{stroke:#8f88e6} .ax{stroke:#85837b} .dot{fill:#eceae3}
  }
</style>"""
DEFS = ('<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')


def F(x, y):
    return math.exp(-(x * x / 1.6 + y * y / 0.9))


def grad(x, y):
    f = F(x, y)
    return -f * 2 * x / 1.6, -f * 2 * y / 0.9


def tx(s):
    return -2 + 4 * s


def ty(s):
    return 0.7 + 0.3 * math.sin(1.3 * tx(s))


def arrow(x1, y1, x2, y2, cls):
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" marker-end="url(#a)"/>'


def trail():
    best = max((F(tx(i / 4000), ty(i / 4000)), i / 4000) for i in range(4001))[1]
    K, W, H = 48, 200, 180
    titles = ["before the peak", "at the peak", "after the peak"]
    subs = ["crossing up to higher contours", "touching one contour",
            "crossing down to lower contours"]
    g = []
    for idx, sp in enumerate([best - 0.22, best, best + 0.22]):
        ox, oy = 15 + idx * 222, 15
        cx, cy = ox + W / 2, oy + H / 2 + 6
        mx = lambda x: cx + K * x
        my = lambda y: cy - K * y
        g.append(f'<rect class="bg" x="{ox}" y="{oy}" width="{W}" height="{H}" rx="8"/>')
        for h in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
            q = -math.log(h)
            g.append(f'<ellipse class="ct" cx="{cx}" cy="{cy}" rx="{K*math.sqrt(1.6*q):.1f}" ry="{K*math.sqrt(0.9*q):.1f}"/>')
        pts = " ".join(f"{mx(tx(s)):.1f},{my(ty(s)):.1f}" for s in [0.15 + 0.7 * i / 120 for i in range(121)])
        g.append(f'<polyline class="trail" points="{pts}"/>')
        x, y = tx(sp), ty(sp)
        q = -math.log(F(x, y))
        g.append(f'<ellipse class="hot" cx="{cx}" cy="{cy}" rx="{K*math.sqrt(1.6*q):.1f}" ry="{K*math.sqrt(0.9*q):.1f}"/>')
        e = 1e-3
        dx, dy = tx(sp + e) - tx(sp - e), ty(sp + e) - ty(sp - e)
        L = math.hypot(dx, dy)
        gx, gy = grad(x, y)
        gl = math.hypot(gx, gy)
        px, py = mx(x), my(y)
        g.append(arrow(px, py, px + 36 * dx / L, py - 36 * dy / L, "gt"))
        g.append(arrow(px, py, px + 36 * gx / gl, py - 36 * gy / gl, "gf"))
        g.append(f'<circle class="dot" cx="{px:.1f}" cy="{py:.1f}" r="5"/>')
        g.append(f'<text class="t" x="{ox+W/2}" y="{oy+H+20}" text-anchor="middle">{titles[idx]}</text>')
        g.append(f'<text class="sm" x="{ox+W/2}" y="{oy+H+38}" text-anchor="middle">{subs[idx]}</text>')
    g.append('<text class="sm" x="15" y="262">orange: the contour of F through the hiker, and the gradient across it. Purple: the trail and the direction walked.</text>')
    return ("trail.svg", 680, 275, "Trail crossing contours before the peak, touching one at the peak, crossing back after",
            "Three small contour maps of the same trail. Before the peak the trail crosses the hiker's contour upward, at the peak it touches without crossing, after it crosses downward.",
            "".join(g))


def tangency():
    S, OX, OY = 120, 150, 300
    X = lambda x: OX + S * x
    Y = lambda y: OY - S * y
    c = 1.0
    g = [f'<rect class="bg" x="0" y="0" width="680" height="330" rx="8"/>']
    for r in (0.5, 1, 1.5, 2, 2.5):
        g.append(f'<circle class="ct" cx="{X(0)}" cy="{Y(0)}" r="{S*r:.1f}"/>')
    g.append(f'<line class="ax" x1="20" y1="{OY}" x2="660" y2="{OY}"/><line class="ax" x1="{OX}" y1="15" x2="{OX}" y2="320"/>')
    g.append(f'<line class="trail" x1="{X(-0.6):.1f}" y1="{Y(c+0.6):.1f}" x2="{X(c+0.6):.1f}" y2="{Y(-0.6):.1f}"/>')
    r = c / math.sqrt(2)
    g.append(f'<circle class="hot" cx="{X(0)}" cy="{Y(0)}" r="{S*r:.1f}"/>')
    px, py = X(c / 2), Y(c / 2)
    s2 = math.sqrt(0.5)
    g.append(arrow(px, py, px + 80 * s2, py - 80 * s2, "gf"))
    g.append(arrow(px, py, px + 80 * s2 * 0.999, py - 80 * s2 * 0.999, "gt"))
    g.append(f'<circle class="dot" cx="{px:.1f}" cy="{py:.1f}" r="6"/>')
    g.append(f'<text class="t" x="330" y="60">Minimize the squared distance, staying on the line</text>')
    g.append(f'<text class="sm" x="330" y="82">Circles are contours of F; the purple line is the rule G = c.</text>')
    g.append(f'<text class="sm" x="330" y="104">At the best point the line only touches a circle,</text>')
    g.append(f'<text class="sm" x="330" y="124">so the two gradients lie on one line: grad F = λ grad G.</text>')
    return ("tangency.svg", 680, 330, "A line touching a circle at the nearest point to the origin",
            "Circles around the origin are contours of squared distance. The constraint line touches one circle at the nearest point, where the two gradient arrows point along the same direction.",
            "".join(g))


def fence():
    g = ['<rect class="bg" x="0" y="0" width="680" height="260" rx="8"/>']
    L = 40.0
    k = 5
    x, y = 20.0, 10.0
    g.append('<rect x="20" y="30" width="310" height="8" fill="#85b7eb"/>')
    g.append('<text class="sm" x="25" y="24">river (needs no fence)</text>')
    g.append(f'<rect x="20" y="38" width="{k*x}" height="{k*y}" rx="2" fill="none" class="trail"/>')
    g.append(f'<text class="t" x="{20+k*x/2}" y="{38+k*y+18}" text-anchor="middle">x (along the river)</text>')
    g.append(f'<text class="t" x="{20+k*x+8}" y="{38+k*y/2+4}">y</text>')
    g.append('<text class="sm" x="20" y="150">best shape: the river side is twice each of the others</text>')
    g.append('<text class="sm" x="20" y="170">the rule is x + 2y = fence length</text>')
    ox, base = 370, 215
    A = lambda xx: xx * (L - xx) / 2
    sx, sy = 6.5, 150 / 200
    g.append(f'<line class="ax" x1="{ox}" y1="{base}" x2="{ox+sx*L:.0f}" y2="{base}"/><line class="ax" x1="{ox}" y1="40" x2="{ox}" y2="{base}"/>')
    pts = " ".join(f"{ox+sx*xx:.1f},{base-sy*A(xx):.1f}" for xx in [L * i / 100 for i in range(101)])
    g.append(f'<polyline class="trail" points="{pts}"/>')
    g.append(f'<circle class="dot" cx="{ox+sx*20:.1f}" cy="{base-sy*A(20):.1f}" r="5"/>')
    ex = L / 3
    g.append(f'<circle fill="#eb6834" cx="{ox+sx*ex:.1f}" cy="{base-sy*A(ex):.1f}" r="4"/>')
    g.append(f'<text class="sm" x="{ox+sx*ex-6:.1f}" y="{base-sy*A(ex)+20:.1f}" text-anchor="middle">even split</text>')
    g.append(f'<text class="sm" x="{ox+sx*20:.1f}" y="{base-sy*A(20)-12:.1f}" text-anchor="middle">best</text>')
    g.append(f'<text class="sm" x="{ox+sx*L/2:.0f}" y="245" text-anchor="middle">side along the river</text>')
    g.append(f'<text class="sm" x="{ox}" y="32">area</text>')
    return ("fence.svg", 680, 260, "Fenced field by a river and its area curve",
            "A rectangle beside a river with the best shape drawn, and the curve of area against the side along the river with its peak and the even split marked.",
            "".join(g))


def write(name, w, h, title, desc, body):
    svg = (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">'
           f'<title>{title}</title><desc>{desc}</desc>{STYLE}{DEFS}{body}</svg>\n')
    (OUT / name).write_text(svg)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (trail, tangency, fence):
        write(*fn())
