#!/usr/bin/env python3
"""Draws figures/design-matrix.svg: where the rows of the running matrix come from.

Left: three measurements at t = 0, 1, 2 and a straight line c0 + c1 t through them, with
the line's three heights marked. Right: the same heights as the product A x, one row of A
per measurement, row i being (1, t_i). The line drawn is only an illustration; the figure
is symbolic and prints nothing. Standard library only.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures" / "design-matrix.svg"

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .hd{font-size:13px;font-weight:500;fill:#1a1a19} .sm{font-size:11px;fill:#57564f} .lab{font-size:12px;fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:1;fill:none} .gd{stroke:#c3c2b7;stroke-width:1;stroke-dasharray:4 4;fill:none}
  .ln{stroke:#57564f;stroke-width:2;fill:none} .st{stroke:#57564f;stroke-width:1.2;fill:none}
  .c1{fill:#2f6fb5} .c2{fill:#0f7a5a} .c3{fill:#8a4fb0}
  .t1{fill:#2f6fb5} .t2{fill:#0f7a5a} .t3{fill:#8a4fb0}
  .br{stroke:#57564f;stroke-width:1.4;fill:none}
  .box{fill:#f4f3ee}
  @media (prefers-color-scheme: dark){
    .hd,.lab{fill:#eceae3} .sm{fill:#b6b4ab} .ax{stroke:#85837b} .gd{stroke:#4a4844}
    .ln,.st,.br{stroke:#b6b4ab} .c1{fill:#7fb2e8} .c2{fill:#4cc79a} .c3{fill:#b79df0}
    .t1{fill:#7fb2e8} .t2{fill:#4cc79a} .t3{fill:#b79df0} .box{fill:#23262b}
  }
</style>"""


def tx(x, y, s, cls="sm", anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def build():
    g = []
    g.append('<rect class="box" x="10" y="10" width="330" height="300" rx="8"/>')
    g.append(tx(25, 32, "Three measurements, one straight line", "hd"))
    X = {0: 100, 1: 190, 2: 280}
    base = 255
    Y = {0: 190, 1: 150, 2: 110}
    g.append(f'<line class="ax" x1="55" y1="{base}" x2="325" y2="{base}"/><line class="ax" x1="55" y1="{base}" x2="55" y2="55"/>')
    g.append(tx(325, base + 28, "t", "lab", "end"))
    g.append(tx(62, 62, "height", "sm"))
    for t in (0, 1, 2):
        g.append(f'<line class="gd" x1="{X[t]}" y1="{base}" x2="{X[t]}" y2="{Y[t]}"/>')
        g.append(tx(X[t], base + 16, f"t = {t}", "sm", "middle"))
    g.append(f'<line class="ln" x1="55" y1="{Y[0]+22.5:.1f}" x2="320" y2="{Y[2]-27:.1f}"/>')
    g.append(f'<line class="st" x1="{X[0]-18}" y1="{base}" x2="{X[0]-18}" y2="{Y[0]}"/>')
    g.append(tx(X[0] - 24, (base + Y[0]) / 2 + 4, "c₀", "lab", "end"))
    g.append(f'<polyline class="st" points="{X[0]},{Y[0]} {X[1]},{Y[0]} {X[1]},{Y[1]}"/>')
    g.append(tx(X[1] + 6, (Y[0] + Y[1]) / 2 + 4, "c₁", "lab"))
    g.append(tx((X[0] + X[1]) / 2, Y[0] + 14, "one step in t", "sm", "middle"))
    for t, cls in ((0, "c1"), (1, "c2"), (2, "c3")):
        g.append(f'<circle class="{cls}" cx="{X[t]}" cy="{Y[t]}" r="6"/>')
    g.append(tx(X[0] + 4, Y[0] - 12, "c₀", "sm t1"))
    g.append(tx(X[1] + 4, Y[1] - 12, "c₀ + c₁", "sm t2"))
    g.append(tx(X[2] - 4, Y[2] - 12, "c₀ + 2c₁", "sm t3", "end"))
    g.append(tx(25, 298, "line: height = c₀ + c₁ · t. c₀ is the start, c₁ the rise per step.", "sm"))

    g.append('<rect class="box" x="350" y="10" width="340" height="300" rx="8"/>')
    g.append(tx(365, 32, "The same three heights as one product A x", "hd"))
    ys = (95, 135, 175)
    rows = ((1, 0), (1, 1), (1, 2))
    cl = ("c1", "c2", "c3")
    for (a, b), y, c in zip(rows, ys, cl):
        g.append(f'<rect class="{c}" x="368" y="{y-18}" width="5" height="26" rx="2"/>')
        g.append(tx(405, y, str(a), "lab", "middle"))
        g.append(tx(445, y, str(b), "lab", "middle"))
    g.append(f'<polyline class="br" points="391,{ys[0]-22} 385,{ys[0]-22} 385,{ys[2]+16} 391,{ys[2]+16}"/>')
    g.append(f'<polyline class="br" points="459,{ys[0]-22} 465,{ys[0]-22} 465,{ys[2]+16} 459,{ys[2]+16}"/>')
    g.append(tx(405, ys[2] + 38, "1s", "sm", "middle"))
    g.append(tx(445, ys[2] + 38, "t", "sm", "middle"))
    g.append(tx(485, ys[1], "×", "lab", "middle"))
    g.append(tx(520, ys[0] + 12, "c₀", "lab", "middle"))
    g.append(tx(520, ys[1] + 12, "c₁", "lab", "middle"))
    g.append(f'<polyline class="br" points="506,{ys[0]-6} 500,{ys[0]-6} 500,{ys[1]+22} 506,{ys[1]+22}"/>')
    g.append(f'<polyline class="br" points="534,{ys[0]-6} 540,{ys[0]-6} 540,{ys[1]+22} 534,{ys[1]+22}"/>')
    g.append(tx(560, ys[1], "=", "lab", "middle"))
    for lab, y, t in (("c₀", ys[0], "t1"), ("c₀ + c₁", ys[1], "t2"), ("c₀ + 2c₁", ys[2], "t3")):
        g.append(tx(580, y, lab, f"lab {t}"))
    g.append(tx(365, 238, "One row per measurement: row i is (1, tᵢ).", "sm"))
    g.append(tx(365, 256, "Column a₁ = (1,1,1): the 1s, the intercept's share.", "sm"))
    g.append(tx(365, 274, "Column a₂ = (0,1,2): the t values, the slope's share.", "sm"))
    g.append(tx(365, 296, "Colours match the three dots on the left.", "sm"))
    return g


svg = ('<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" role="img">'
       '<title>Where the design matrix comes from</title>'
       '<desc>Left: three measurements at t=0, 1, 2 and a straight line c0 + c1 t through them, with the line heights c0, c0+c1 and c0+2c1 marked as three coloured dots. '
       'Right: the same heights as the product of the matrix with rows (1,0), (1,1), (1,2) and the vector (c0, c1). Each row of the matrix is one measurement; the first column is all ones and the second column is the t values.</desc>'
       + STYLE + "".join(build()) + "</svg>\n")
OUT.write_text(svg)
