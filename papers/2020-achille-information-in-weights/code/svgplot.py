#!/usr/bin/env python3
"""Tiny SVG helpers shared by the --figures branches of the scripts here.

Light and dark themes via prefers-color-scheme; no dependencies. Running this
file on its own does nothing but say so (make verify runs every code/*.py).
"""
from __future__ import annotations

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sub{font-size:11.5px;fill:#57564f} .tiny{font-size:10px;fill:#57564f} .ink{fill:#1a1a19}
  .ax{stroke:#c3c2b7;stroke-width:0.8;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .c0{stroke:#2f6fb5;stroke-width:2;fill:none} .c0f{fill:#2f6fb5} .c0b{fill:#2f6fb5;opacity:.45}
  .c1{stroke:#c2571a;stroke-width:2;fill:none} .c1f{fill:#c2571a} .c1b{fill:#c2571a;opacity:.45}
  .c2{stroke:#0f7a5a;stroke-width:2;fill:none} .c2f{fill:#0f7a5a} .c2b{fill:#0f7a5a;opacity:.45}
  .c3{stroke:#7c3aed;stroke-width:2;fill:none} .c3f{fill:#7c3aed} .c3b{fill:#7c3aed;opacity:.45}
  .k{stroke:#1a1a19;stroke-width:1.1;fill:none;stroke-dasharray:4 3} .kf{fill:#1a1a19}
  .k0{stroke:#1a1a19;stroke-width:1.2;fill:none}
  .dash{stroke-dasharray:5 3}
  @media (prefers-color-scheme: dark){
    .lab,.sub,.tiny{fill:#b6b4ab} .hd,.ink{fill:#eceae3} .ax{stroke:#4a4844} .gd{stroke:#33312e}
    .c0{stroke:#7fb2e8} .c0f{fill:#7fb2e8} .c0b{fill:#7fb2e8;opacity:.5}
    .c1{stroke:#f0a070} .c1f{fill:#f0a070} .c1b{fill:#f0a070;opacity:.5}
    .c2{stroke:#4cc79a} .c2f{fill:#4cc79a} .c2b{fill:#4cc79a;opacity:.5}
    .c3{stroke:#b48cff} .c3f{fill:#b48cff} .c3b{fill:#b48cff;opacity:.5}
    .k,.k0{stroke:#eceae3} .kf{fill:#eceae3}
  }
</style>"""


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{esc(title)}</title>\n<desc>{esc(desc)}</desc>\n{STYLE}\n{body}\n</svg>\n")


def title(x, y, s):
    return f'<text class="hd" x="{x}" y="{y}">{esc(s)}</text>'


def sub(x, y, s):
    return f'<text class="sub" x="{x}" y="{y}">{esc(s)}</text>'


def text(x, y, s, cls="lab", anchor="start"):
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{esc(s)}</text>'


class Axes:
    def __init__(self, x0, y0, w, h, xlim, ylim, logx=False, logy=False):
        import math
        self.m = math
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.logx, self.logy = logx, logy
        self.xlim = tuple(math.log10(v) for v in xlim) if logx else xlim
        self.ylim = tuple(math.log10(v) for v in ylim) if logy else ylim

    def X(self, x):
        if self.logx:
            x = self.m.log10(x)
        return self.x0 + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def Y(self, y):
        if self.logy:
            y = self.m.log10(max(y, 1e-300))
        return self.y0 + self.h - (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def path(self, xs, ys, cls):
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))
        return f'<polyline class="{cls}" points="{pts}"/>'

    def dots(self, xs, ys, cls, r=2.8):
        return "".join(f'<circle class="{cls}" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>'
                       for x, y in zip(xs, ys))

    def bars(self, edges, heights, cls):
        out = []
        base = self.Y(self.ylim[0] if not self.logy else 10 ** self.ylim[0])
        for a, b, hgt in zip(edges[:-1], edges[1:], heights):
            if hgt <= 0:
                continue
            y = self.Y(hgt)
            out.append(f'<rect class="{cls}" x="{self.X(a):.1f}" y="{y:.1f}" '
                       f'width="{max(self.X(b) - self.X(a), 0.6):.1f}" height="{base - y:.1f}"/>')
        return "".join(out)

    def hline(self, y, cls="k"):
        return f'<line class="{cls}" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.Y(y):.1f}" y2="{self.Y(y):.1f}"/>'

    def vline(self, x, cls="k"):
        return f'<line class="{cls}" x1="{self.X(x):.1f}" x2="{self.X(x):.1f}" y1="{self.y0}" y2="{self.y0 + self.h}"/>'

    def frame(self, xticks, yticks, xlab, ylab, xfmt="{:g}", yfmt="{:g}"):
        out = []
        for t in yticks:
            out.append(f'<line class="gd" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.Y(t):.1f}" y2="{self.Y(t):.1f}"/>')
            out.append(f'<text class="tiny" x="{self.x0 - 5}" y="{self.Y(t) + 3:.1f}" text-anchor="end">{yfmt.format(t)}</text>')
        for t in xticks:
            out.append(f'<text class="tiny" x="{self.X(t):.1f}" y="{self.y0 + self.h + 13}" text-anchor="middle">{xfmt.format(t)}</text>')
        out.append(f'<line class="ax" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.y0 + self.h}" y2="{self.y0 + self.h}"/>')
        out.append(f'<line class="ax" x1="{self.x0}" x2="{self.x0}" y1="{self.y0}" y2="{self.y0 + self.h}"/>')
        out.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 28}" text-anchor="middle">{esc(xlab)}</text>')
        if ylab:
            out.append(f'<text class="lab" x="{self.x0}" y="{self.y0 - 8}">{esc(ylab)}</text>')
        return "".join(out)


def tick_label(ax, x, s):
    return f'<text class="tiny" x="{ax.X(x):.1f}" y="{ax.y0 + ax.h + 13}" text-anchor="middle">{esc(s)}</text>'


def legend(x, y, items, cols=1, colw=150):
    out = []
    for i, (cls, lab) in enumerate(items):
        cx = x + (i % cols) * colw
        cy = y + (i // cols) * 15
        if cls.endswith("b"):
            out.append(f'<rect class="{cls}" x="{cx}" y="{cy - 6}" width="16" height="8"/>')
        else:
            out.append(f'<line class="{cls}" x1="{cx}" x2="{cx + 16}" y1="{cy - 2}" y2="{cy - 2}"/>')
        out.append(f'<text class="tiny" x="{cx + 21}" y="{cy + 1}">{esc(lab)}</text>')
    return "".join(out)


if __name__ == "__main__":
    print("svgplot.py: SVG helpers only, nothing to check")
