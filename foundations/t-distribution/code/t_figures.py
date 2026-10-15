"""Draws the three static figures on the t-distribution page.

    python3 t_figures.py            # prints nothing, writes ../figures/*.svg

Standard library only. The figures are pictures of known formulas; nothing here is a result that the
notes quote.
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures"

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:12px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .ax{stroke:#c3c2b7;stroke-width:0.6;fill:none}
  .gd{stroke:#e2e1da;stroke-width:0.5;fill:none;stroke-dasharray:4 4}
  .a{fill:none;stroke:#2f6fb5;stroke-width:2} .b{fill:none;stroke:#c2642a;stroke-width:2}
  .g{fill:none;stroke:#2a8a5c;stroke-width:2} .n{fill:none;stroke:#8a8880;stroke-width:2;stroke-dasharray:6 4}
  .fa{fill:#2f6fb5} .fb{fill:#c2642a} .fg{fill:#2a8a5c} .fn{fill:#8a8880}
  .sph{fill:none;stroke:#8a8880;stroke-width:1.5} .rg{fill:none;stroke:#c2642a;stroke-width:2}
  .vec{stroke:#2f6fb5;stroke-width:2;fill:none} .ref{stroke:#57564f;stroke-width:1.2;fill:none;stroke-dasharray:5 4}
  @media (prefers-color-scheme: dark){
    .lab{fill:#b6b4ab} .hd{fill:#eceae3} .ax{stroke:#4a4844} .gd{stroke:#33312e}
    .a{stroke:#7fb2e8} .b{stroke:#e0955c} .g{stroke:#5fcf96} .n{stroke:#8a8880}
    .fa{fill:#7fb2e8} .fb{fill:#e0955c} .fg{fill:#5fcf96} .fn{fill:#8a8880}
    .sph{stroke:#8a8880} .rg{stroke:#e0955c} .vec{stroke:#7fb2e8} .ref{stroke:#b6b4ab}
  }
</style>"""


def lg(x):
    return math.lgamma(x)


def tpdf(x, v):
    return math.exp(lg((v + 1) / 2) - lg(v / 2)) / math.sqrt(v * math.pi) * (1 + x * x / v) ** (-(v + 1) / 2)


def npdf(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n<title>{title}</title>\n'
            f'<desc>{desc}</desc>\n{STYLE}\n{body}\n</svg>\n')


def axes(x0, x1, y0, y1, xs, ys, xticks, yticks, xlabel):
    s = []
    for t in yticks:
        y = y1 - (t - ys[0]) / (ys[1] - ys[0]) * (y1 - y0)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="gd"/>'
                 f'<text class="lab" x="{x0-8}" y="{y+4:.1f}" text-anchor="end">{t}</text>')
    s.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="ax"/>')
    for t in xticks:
        x = x0 + (t - xs[0]) / (xs[1] - xs[0]) * (x1 - x0)
        s.append(f'<text class="lab" x="{x:.1f}" y="{y1+18}" text-anchor="middle">{t}</text>')
    s.append(f'<text class="lab" x="{(x0+x1)/2}" y="{y1+40}" text-anchor="middle">{xlabel}</text>')
    return s


def curve(f, xs, ys, x0, x1, y0, y1, cls, n=200):
    pts = []
    for i in range(n + 1):
        x = xs[0] + (xs[1] - xs[0]) * i / n
        y = f(x)
        pts.append(f"{x0+(x-xs[0])/(xs[1]-xs[0])*(x1-x0):.1f},{y1-(min(y,ys[1])-ys[0])/(ys[1]-ys[0])*(y1-y0):.1f}")
    return f'<polyline class="{cls}" points="{" ".join(pts)}"/>'


def fig_density():
    x0, x1, y0, y1 = 70, 740, 50, 260
    xs, ys = (-6, 6), (0, 0.45)
    b = [f'<text class="hd" x="{x0}" y="26">t densities against the normal: smaller ν, fatter tails</text>']
    b += axes(x0, x1, y0, y1, xs, ys, [-6, -4, -2, 0, 2, 4, 6], [0.0, 0.1, 0.2, 0.3, 0.4], "t")
    b.append(curve(npdf, xs, ys, x0, x1, y0, y1, "n"))
    for v, c in ((1, "b"), (3, "g"), (10, "a")):
        b.append(curve(lambda x, v=v: tpdf(x, v), xs, ys, x0, x1, y0, y1, c))
    lx = 590
    for i, (c, t) in enumerate((("fb", "ν = 1 (Cauchy)"), ("fg", "ν = 3"), ("fa", "ν = 10"), ("fn", "normal"))):
        b.append(f'<rect class="{c}" x="{lx}" y="{58+i*18}" width="10" height="10" rx="2"/>'
                 f'<text class="lab" x="{lx+16}" y="{67+i*18}">{t}</text>')
    return svg(780, 312, "t densities against the normal",
               "Bell curves for ν equal to 1, 3 and 10 and the normal. All are centred at zero; the smaller ν is, the lower the peak and the more mass sits far from zero.",
               "\n".join(b))


def fig_loss():
    x0, x1, y0, y1 = 70, 740, 50, 260
    xs, ys = (0, 1), (0, 1.1)
    b = [f'<text class="hd" x="{x0}" y="26">Per-pixel T-Loss against the error δ, rescaled to 1 at δ = 1</text>']
    b += axes(x0, x1, y0, y1, xs, ys, [0, 0.25, 0.5, 0.75, 1], [0.0, 0.5, 1.0], "error δ = |y − f(x)|")
    b.append(curve(lambda d: d * d, xs, ys, x0, x1, y0, y1, "b"))
    b.append(curve(lambda d: d, xs, ys, x0, x1, y0, y1, "n"))
    for v, c in ((1.0, "a"), (0.05, "g")):
        L = lambda d, v=v: (v + 1) / 2 * math.log(1 + d * d / v)
        b.append(curve(lambda d, L=L: L(d) / L(1), xs, ys, x0, x1, y0, y1, c))
    lx = 110
    for i, (c, t) in enumerate((("fb", "MSE"), ("fn", "MAE"), ("fa", "T-Loss, ν = 1"), ("fg", "T-Loss, ν = 0.05"))):
        b.append(f'<rect class="{c}" x="{lx}" y="{58+i*18}" width="10" height="10" rx="2"/>'
                 f'<text class="lab" x="{lx+16}" y="{67+i*18}">{t}</text>')
    return svg(780, 312, "T-Loss compared with MSE and MAE",
               "Loss against error. MSE is a parabola. MAE is a straight line. T-Loss with small ν rises steeply near zero error, then flattens: large errors cost little more than moderate ones.",
               "\n".join(b))


def fig_sphere():
    cx, cy, r = 260, 170, 120
    th = math.radians(55)
    px, py = cx + r * math.sin(th) * 0.0, cy - r * math.cos(th)
    # point on the sphere at polar angle th from the vertical reference axis, seen from the side
    ex = r * math.sin(th)
    ry = 24
    b = [f'<text class="hd" x="30" y="26">A uniform point on a sphere and its angle θ to a reference axis</text>',
         f'<circle class="sph" cx="{cx}" cy="{cy}" r="{r}"/>',
         f'<ellipse class="sph" cx="{cx}" cy="{cy}" rx="{r}" ry="{ry}" style="stroke-dasharray:4 4"/>',
         f'<line class="ref" x1="{cx}" y1="{cy+r+16}" x2="{cx}" y2="{cy-r-18}"/>',
         f'<text class="lab" x="{cx+8}" y="{cy-r-6}">reference axis</text>',
         f'<ellipse class="rg" cx="{cx}" cy="{cy-r*math.cos(th)}" rx="{ex:.1f}" ry="{ex*ry/r:.1f}"/>',
         f'<line class="vec" x1="{cx}" y1="{cy}" x2="{cx+ex*0.8:.1f}" y2="{cy-r*math.cos(th)+ex*ry/r*0.6:.1f}"/>',
         f'<circle class="fa" cx="{cx+ex*0.8:.1f}" cy="{cy-r*math.cos(th)+ex*ry/r*0.6:.1f}" r="4.5"/>',
         f'<path d="M {cx} {cy-44} A 44 44 0 0 1 {cx+44*math.sin(th):.1f} {cy-44*math.cos(th):.1f}" class="vec" style="stroke-width:1.2"/>',
         f'<text class="lab" x="{cx+14}" y="{cy-50}">θ</text>',
         f'<text class="lab" x="{cx+r+16}" y="{cy-r*math.cos(th)-ex*ry/r-4:.1f}">all points with the same θ</text>',
         f'<text class="lab" x="{cx+r+16}" y="{cy-r*math.cos(th)-ex*ry/r+12:.1f}">form a ring: a sphere one</text>',
         f'<text class="lab" x="{cx+r+16}" y="{cy-r*math.cos(th)-ex*ry/r+28:.1f}">dimension lower, radius sin θ</text>',
         f'<text class="lab" x="{cx+r+16}" y="{cy-r*math.cos(th)-ex*ry/r+48:.1f}">ring size ∝ sin<tspan dy="-5" font-size="10">ν−1</tspan><tspan dy="5">θ</tspan></text>']
    return svg(780, 340, "A sphere cut into rings of constant angle",
               "A sphere with a vertical reference axis, a random point and the angle theta between its radius and the axis. The points at the same angle form a horizontal ring, a sphere one dimension lower with radius sine theta.",
               "\n".join(b))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "densities.svg").write_text(fig_density())
    (OUT / "tloss.svg").write_text(fig_loss())
    (OUT / "sphere.svg").write_text(fig_sphere())
