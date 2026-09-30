#!/usr/bin/env python3
"""Tiny SVG helper shared by the --figures branches of the other scripts.

Light and dark themes via prefers-color-scheme, the same palette as the rest of
the collection. Standard library only. Running it directly does nothing useful;
it only prints its own name so `make verify` has something to show.
"""
from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures"

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sub{font-size:11.5px;fill:#57564f} .tiny{font-size:10px;fill:#57564f} .ink{fill:#1a1a19}
  .ax{stroke:#c3c2b7;stroke-width:0.8;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .b{stroke:#2f6fb5;stroke-width:2;fill:none} .bf{fill:#2f6fb5} .bb{fill:#2f6fb5;opacity:.28}
  .o{stroke:#c2571a;stroke-width:2;fill:none} .of{fill:#c2571a} .ob{fill:#c2571a;opacity:.28}
  .g{stroke:#0f7a5a;stroke-width:2;fill:none} .gf{fill:#0f7a5a}
  .p{stroke:#7c3aed;stroke-width:2;fill:none} .pf{fill:#7c3aed}
  .r{stroke:#b42318;stroke-width:1.6;fill:none} .rf{fill:#b42318}
  .k{stroke:#1a1a19;stroke-width:1.1;fill:none;stroke-dasharray:4 3} .k0{stroke:#1a1a19;stroke-width:1.3;fill:none}
  .dash{stroke-dasharray:5 3} .dot{stroke-dasharray:1.5 3}
  .shade{fill:#8a8880;opacity:.16} .shader{fill:#b42318;opacity:.12}
  .cell{stroke:#ffffff;stroke-width:1}
  @media (prefers-color-scheme: dark){
    .lab,.sub,.tiny{fill:#b6b4ab} .hd,.ink{fill:#eceae3} .ax{stroke:#4a4844} .gd{stroke:#33312e}
    .b{stroke:#3d8ad8} .bf{fill:#3d8ad8} .bb{fill:#3d8ad8;opacity:.36}
    .o{stroke:#d8733a} .of{fill:#d8733a} .ob{fill:#d8733a;opacity:.36}
    .g{stroke:#4cc79a} .gf{fill:#4cc79a} .p{stroke:#b48cff} .pf{fill:#b48cff}
    .r{stroke:#ff7a6b} .rf{fill:#ff7a6b}
    .k,.k0{stroke:#eceae3} .shade{fill:#b6b4ab;opacity:.16} .shader{fill:#ff7a6b;opacity:.16}
    .cell{stroke:#131417}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n" + "\n".join(body) + "\n</svg>\n")


def write(name, text):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"wrote figures/{name}")


class Axes:
    """A rectangle with data limits; logx/logy take base-10 logs of the data."""

    def __init__(self, x0, y0, w, h, xlim, ylim, logx=False, logy=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.logx, self.logy = logx, logy
        self.xlim = tuple(math.log10(v) for v in xlim) if logx else xlim
        self.ylim = tuple(math.log10(v) for v in ylim) if logy else ylim

    def X(self, x):
        x = math.log10(x) if self.logx else x
        return self.x0 + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def Y(self, y):
        y = math.log10(y) if self.logy else y
        return self.y0 + self.h - (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def path(self, xs, ys, cls):
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))
        return f'<polyline class="{cls}" points="{pts}"/>'

    def dot(self, x, y, cls, r=3):
        return f'<circle class="{cls}" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>'

    def text(self, x, y, s, cls="tiny", anchor="start", dx=0, dy=0):
        return (f'<text class="{cls}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" '
                f'text-anchor="{anchor}">{s}</text>')

    def frame(self, xticks, yticks, xlab, ylab, xfmt="{:g}", yfmt="{:g}"):
        out = []
        for t in yticks:
            out.append(f'<line class="gd" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.Y(t):.1f}" y2="{self.Y(t):.1f}"/>')
            out.append(f'<text class="tiny" x="{self.x0 - 5}" y="{self.Y(t) + 3:.1f}" text-anchor="end">{yfmt.format(t)}</text>')
        for t in xticks:
            out.append(f'<text class="tiny" x="{self.X(t):.1f}" y="{self.y0 + self.h + 13}" text-anchor="middle">{xfmt.format(t)}</text>')
        out.append(f'<line class="ax" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.y0 + self.h}" y2="{self.y0 + self.h}"/>')
        out.append(f'<line class="ax" x1="{self.x0}" x2="{self.x0}" y1="{self.y0}" y2="{self.y0 + self.h}"/>')
        out.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 28}" text-anchor="middle">{xlab}</text>')
        if ylab:
            out.append(f'<text class="lab" x="{self.x0}" y="{self.y0 - 8}">{ylab}</text>')
        return "\n".join(out)


def legend(x, y, items, step=16):
    """items: (css class of a line, label)."""
    out = []
    for i, (cls, lab) in enumerate(items):
        yy = y + i * step
        out.append(f'<line class="{cls}" x1="{x}" x2="{x + 20}" y1="{yy}" y2="{yy}"/>')
        out.append(f'<text class="lab" x="{x + 26}" y="{yy + 4}">{lab}</text>')
    return "\n".join(out)


if __name__ == "__main__":
    print("svgkit.py: SVG helper for the --figures branches; nothing to check here.")
