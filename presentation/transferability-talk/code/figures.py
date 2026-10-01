#!/usr/bin/env python3
"""Figures for the transferability talk, and the numbers behind the two charts.

The talk quotes numbers from the annotated notes and never recomputes them, so
this script's job is to keep the *plots* from drifting away from those notes:

  * the H-score `n ~ k` chart is drawn from the table in the Claßen notes
    ("H-score's instability is a transition at n ~ k, not a decay");
  * the statistical-power chart is drawn from the two tables in the Chaves
    notes ("The instrument's resolution").

Both tables are parsed out of the notes' markdown, checked for shape and range,
and printed here, so a change in a note that would change the chart fails loudly.
The two diagrams (the problem pipeline and the progression from NCE to PAS) hold
no data, only structure. One claim from the Claßen notes is recomputed here from
scratch, in plain Python, as an independent check: of two rankings with the same
tau and rho, the weighted tau prefers the one that gets the TOP of the list right
(see `rank_agreement`). Two figures from the individual papers' notes are
copied in so the deck can be previewed from this folder alone; each copy gets an
opaque card behind it, because a Marp slide is always white while the figures
follow the viewer's dark mode.

Standard library only.

Run:  python3 figures.py
      python3 figures.py --figures  (checks, then rewrites ../figures/*.svg)
"""
from __future__ import annotations

import html
import math
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
PAPERS = TOPIC.parent.parent / "papers"
FIGS = TOPIC / "figures"

# existing figures from the papers' notes, copied under a deck-local name
REUSED = {
    "logme-graphical-model.svg": "2021-you-logme/figures/graphical-model.svg",
    "sfda-confmix.svg": "2022-shao-sfda/figures/confmix.svg",
}


# ------------------------------------------------------- numbers from the notes
def _cells(line):
    # the notes typeset negatives with U+2212, which float() does not accept
    return [c.strip().replace("*", "").replace("−", "-").strip()
            for c in line.strip().strip("|").split("|")]


def tables_after(text, heading_prefix):
    """Every markdown table between `heading_prefix` and the next heading of the
    same or a higher level, as lists of rows of cells (separator rows dropped)."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(heading_prefix))
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    out, cur = [], []
    for l in lines[start + 1:]:
        if re.match(r"#{1,%d} " % level, l):
            break
        if l.lstrip().startswith("|"):
            if not re.fullmatch(r"[|\s:\-]+", l.strip()):
                cur.append(_cells(l))
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def read_note(slug):
    return (PAPERS / slug / "notes.md").read_text(encoding="utf-8")


def classen_sweep():
    """(k, n, n/k, LEEP stability, H-score stability, cond(S_T)) per row."""
    t = tables_after(read_note("2026-classen-te-robustness"),
                     "### H-score's instability is a transition")[0]
    rows = [(int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            for r in t[1:]]
    assert len(rows) == 7, rows
    for k, n, ratio, leep, h, cond in rows:
        # the note prints n/k rounded (2000/64 = 31.25 appears as 31)
        assert abs(n / k - ratio) <= 0.02 * ratio and -1 <= leep <= 1 and -1 <= h <= 1 and cond >= 1
    return rows


def chaves_power():
    """({n: |tau| needed for p<0.05}, {n: power at true tau = 0.3}, {n: null sd})."""
    ts = tables_after(read_note("2023-chaves-medical-transferability"),
                      "## The instrument's resolution")
    sd_t = next(t for t in ts if "sd under null" in t[0][1])
    pw_t = next(t for t in ts if t[1][0].startswith("power at true"))
    sd = {int(r[0]): float(r[1]) for r in sd_t[1:]}
    need = {int(r[0]): float(r[2]) for r in sd_t[1:]}
    ns = [int(c) for c in pw_t[0][1:]]
    power = dict(zip(ns, [float(c) for c in pw_t[1][1:]]))
    assert list(need) == [10, 20, 50] and list(power) == [10, 20, 30, 50, 80]
    assert all(0 <= v <= 1 for v in power.values())
    return need, power, sd


def reported_tau():
    """The H-score tau of 0.270 that the Chaves notes call 'barely one standard
    error from zero'; asserted to still be in the note, not retyped from memory."""
    assert "0.270" in read_note("2023-chaves-medical-transferability")
    return 0.270


def rank_agreement():
    """Two rankings of ten sources against one reference, labelled by what they get
    right. Position 0 holds the best source. These are the Claßen notes' metrics A (top
    right) and B (tail right), recomputed here without the notes' code."""
    n = 10
    ref = [n - 1 - i for i in range(n)]
    top_right = ref[:5] + ref[5:][::-1]        # best five in place, worst five reversed
    tail_right = ref[:5][::-1] + ref[5:]       # best five reversed, worst five in place

    def tau(a, b):
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        s = lambda x: (x > 0) - (x < 0)                                       # noqa: E731
        return sum(s(a[i] - a[j]) * s(b[i] - b[j]) for i, j in pairs) / len(pairs)

    def weighted_tau(a, b):
        ra = [sorted(a, reverse=True).index(x) for x in a]                    # 0 = best
        rb = [sorted(b, reverse=True).index(x) for x in b]
        s = lambda x: (x > 0) - (x < 0)                                       # noqa: E731
        num = den = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                w = 1 / (ra[i] + 1) + 1 / (ra[j] + 1) + 1 / (rb[i] + 1) + 1 / (rb[j] + 1)
                num += w * s(a[i] - a[j]) * s(b[i] - b[j])
                den += w
        return num / den

    def rho(a, b):
        ra = [sorted(a).index(x) for x in a]
        rb = [sorted(b).index(x) for x in b]
        d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
        return 1 - 6 * d2 / (n * (n * n - 1))

    out = {}
    for tag, v in (("top right, tail scrambled", top_right), ("tail right, top scrambled", tail_right)):
        out[tag] = (tau(ref, v), weighted_tau(ref, v), rho(ref, v))
    (t1, w1, r1), (t2, w2, r2) = out.values()
    assert abs(t1 - t2) < 1e-12 and abs(r1 - r2) < 1e-12 and w1 > w2
    return out


# ------------------------------------------------------------------ svg basics
STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .hd{font-size:14.5px;font-weight:500;fill:#1a1a19} .v{font-size:13px;fill:#1a1a19}
  .b{font-weight:500} .sm{font-size:12px;fill:#57564f} .tiny{font-size:11px;fill:#57564f}
  .it{font-style:italic} .sb{font-size:72%}
  .bgcard{fill:#fbfaf6}
  .box{fill:#f1efe7;stroke:#8a8880;stroke-width:1.3}
  .slow{fill:#f7e7d8;stroke:#c8702f;stroke-width:1.5} .fast{fill:#dfeaf6;stroke:#2f6fb5;stroke-width:1.5}
  .need{fill:#dfeaf6;stroke:#2f6fb5;stroke-width:1.8} .free{fill:#f7e7d8;stroke:#c8702f;stroke-width:1.8}
  .e{stroke:#1a1a19;stroke-width:1.5;fill:none} .ah{fill:#1a1a19}
  .ax{stroke:#8a8880;stroke-width:0.9;fill:none} .gd{stroke:#e2e1da;stroke-width:0.7;fill:none}
  .frame{fill:none;stroke:#c9c7bf;stroke-width:1}
  .sb1{stroke:#2f6fb5;stroke-width:2.2;fill:none} .sb2{stroke:#c8702f;stroke-width:2.2;fill:none}
  .db1{stroke:#2f6fb5;stroke-width:2.2;fill:none;stroke-dasharray:6 4}
  .db2{stroke:#c8702f;stroke-width:2.2;fill:none;stroke-dasharray:6 4}
  .p1{fill:#2f6fb5;stroke:#fbfaf6;stroke-width:1.2} .p2{fill:#c8702f;stroke:#fbfaf6;stroke-width:1.2}
  .band{fill:#c8702f;opacity:.10} .ref{stroke:#8a8880;stroke-width:1.1;stroke-dasharray:3 3;fill:none}
  .refd{stroke:#1a1a19;stroke-width:1.3;stroke-dasharray:6 4;fill:none}
  .blue{fill:#2f6fb5} .orange{fill:#c8702f}
  @media (prefers-color-scheme: dark){
    .hd,.v{fill:#eceae3} .sm,.tiny{fill:#b6b4ab} .bgcard{fill:#161615}
    .box{fill:#25241f;stroke:#85837b}
    .slow{fill:#3a2a1c;stroke:#e09a5f} .fast{fill:#1d2f45;stroke:#7fb2e8}
    .need{fill:#1d2f45;stroke:#7fb2e8} .free{fill:#3a2a1c;stroke:#e09a5f}
    .e{stroke:#eceae3} .ah{fill:#eceae3}
    .ax{stroke:#85837b} .gd{stroke:#33312e} .frame{stroke:#4a4844}
    .sb1,.db1{stroke:#7fb2e8} .sb2,.db2{stroke:#e09a5f}
    .p1{fill:#7fb2e8;stroke:#161615} .p2{fill:#e09a5f;stroke:#161615}
    .band{fill:#e09a5f;opacity:.14} .ref{stroke:#85837b} .refd{stroke:#eceae3}
    .blue{fill:#7fb2e8} .orange{fill:#e09a5f}
  }
</style>"""

DEFS = ('<defs><marker id="ah" viewBox="0 0 10 8" refX="10" refY="4" markerWidth="10" markerHeight="8" '
        'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,4 L0,8 Z" class="ah"/></marker></defs>')


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n{DEFS}\n"
            f'<rect width="{w}" height="{h}" rx="6" class="bgcard"/>\n' + "\n".join(body) + "\n</svg>\n")


def rich(parts):
    """tspans for text with subscripts; (text, kind) pairs, kind in '', 'it', 'sub'."""
    out, cur = [], 0
    for txt, kind in parts:
        tgt = 4 if kind == "sub" else 0
        dy = f' dy="{tgt - cur}"' if tgt != cur else ""
        cur = tgt
        cls = {"": "", "it": ' class="it"', "sub": ' class="it sb"'}[kind]
        out.append(f"<tspan{cls}{dy}>{html.escape(txt, quote=False)}</tspan>")
    return "".join(out)


def text(x, y, content, cls="v", anchor="middle"):
    inner = rich(content) if isinstance(content, list) else html.escape(content, quote=False)
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{inner}</text>'


def box(x, y, w, h, cls, lines, first_bold=True, lh=17):
    """A rounded box with centred lines of text (str or rich parts)."""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>']
    y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + 4.5
    for i, ln in enumerate(lines):
        c = "v b" if (i == 0 and first_bold) else "sm"
        out.append(text(x + w / 2, y0 + i * lh, ln, c))
    return "\n".join(out)


def arrow(d):
    return f'<path d="{d}" class="e" marker-end="url(#ah)"/>'


# -------------------------------------------------------------------- diagrams
def fig_pipeline():
    body = [text(14, 24, "The problem in one picture: rank the pool without fine-tuning every model",
                 "hd", "start")]
    # inputs, as two stacked cards behind the pool
    for dx in (12, 6):
        body.append(f'<rect x="{20 + dx}" y="{52 - dx}" width="170" height="74" rx="8" class="box"/>')
    body.append(box(20, 58, 170, 74, "box", [
        "M pre-trained models",
        [("θ", "it"), ("m", "sub"), (" = (", ""), ("φ", "it"), ("m", "sub"), (", ", ""), ("h", "it"), ("m", "sub"), (")", "")],
        "extractor + optional head"]))
    body.append(box(20, 168, 170, 74, "box", [
        "Target data",
        [("𝒟", ""), ("T", "sub"), (", n examples", "")],
        "labelled or not"]))
    body.append('<path d="M 204,62 V 238" class="e"/>')
    body.append(arrow("M 204,150 H 236 V 92 H 268"))
    body.append(arrow("M 204,150 H 236 V 208 H 268"))
    # the two paths
    body.append(box(270, 52, 214, 80, "slow", ["Fine-tune each model", "M training runs,", "hours to days apiece"]))
    body.append(box(270, 168, 214, 80, "fast", [
        [("Score ", ""), ("S", "it"), ("m", "sub"), (" = S(", ""), ("θ", "it"), ("m", "sub"), (", ", ""), ("𝒟", ""), ("T", "sub"), (")", "")],
        "one forward pass,", "no gradient descent"]))
    body.append(arrow("M 484,92 H 540"))
    body.append(arrow("M 484,208 H 540"))
    body.append(box(542, 52, 102, 80, "slow", ["ground truth", [("T", "it"), ("1", "sub"), (" … ", ""), ("T", "it"), ("M", "sub")]]))
    body.append(box(542, 168, 102, 80, "fast", ["estimate", [("S", "it"), ("1", "sub"), (" … ", ""), ("S", "it"), ("M", "sub")]]))
    body.append(arrow("M 644,92 H 660 V 128 H 674"))
    body.append(arrow("M 644,208 H 660 V 172 H 674"))
    body.append(box(676, 105, 76, 90, "box", ["compare", [("τ", "it"), ("w", "sub"), (", regret", "")]], lh=18))
    body.append(text(14, 278, "Success means the cheap ranking (blue) reproduces the expensive one (orange).", "sm", "start"))
    return svg(760, 296, "The transferability estimation problem",
               "A pool of M pre-trained models and a target dataset feed two paths. The expensive path "
               "fine-tunes every model to get ground-truth performance T1 to TM. The cheap path scores each "
               "model in one forward pass to get estimates S1 to SM. The two rankings are compared with a "
               "weighted Kendall tau and the regret of the top pick.", body)


LADDER = [  # name, who, how it differs from the step before, reads, target labels needed?
    ("NCE", "Tran 2019", None, ["two label lists", "no model at all"], True),
    ("LEEP", "Nguyen 2020", ["swap true source labels", "for the model's", "soft predictions"],
     ["source softmax", "+ target labels"], True),
    ("H-score", "Bao 2019", ["drop the source head;", "optimise the head", "away in closed form"],
     ["features", "+ target labels"], True),
    ("LogME", "You 2021", ["integrate the head out", "(evidence); add", "regression"],
     ["features", "+ target labels"], True),
    ("SFDA", "Shao 2022", ["imitate fine-tuning:", "make the task", "harder first"],
     ["features", "+ target labels"], True),
    ("PAS", "Diniz 2026", ["drop the target labels;", "use the source", "partition instead"],
     ["features + source data", "no target labels"], False),
]


def fig_ladder():
    xs = [78 + i * 148 for i in range(len(LADDER))]
    yc, r = 138, 33
    body = [text(14, 24, "From the tractable case to the label-free one: what changes at each step",
                 "hd", "start")]
    for i, (name, who, change, reads, needs) in enumerate(LADDER):
        x = xs[i]
        body.append(f'<circle cx="{x}" cy="{yc}" r="{r}" class="{"need" if needs else "free"}"/>')
        body.append(text(x, yc + 5, name, "v b"))
        body.append(text(x, yc + r + 20, who, "sm"))
        for j, ln in enumerate(reads):
            body.append(text(x, yc + r + 44 + j * 15, ln, "tiny"))
        if change:
            xm = (xs[i - 1] + x) / 2
            body.append(arrow(f"M {xs[i - 1] + r + 3},{yc} H {x - r - 5}"))
            for j, ln in enumerate(change):
                body.append(text(xm, 70 + j * 15, ln, "tiny"))
    body.append('<circle cx="470" cy="270" r="8" class="need"/>'
                + text(484, 274, "needs target labels", "sm", "start")
                + '<circle cx="640" cy="270" r="8" class="free"/>'
                + text(654, 274, "needs no target labels", "sm", "start"))
    body.append(text(14, 300, "A conceptual order, not a chronological one: NCE and H-score are both 2019, LEEP 2020, "
                              "LogME 2021, SFDA 2022, PAS 2026.", "sm", "start"))
    return svg(890, 316, "From NCE to PAS",
               "Six circles in a row: NCE, LEEP, H-score, LogME, SFDA and PAS, joined by arrows labelled with "
               "what changes at each step. NCE reads two label lists. LEEP reads the source softmax. H-score, "
               "LogME and SFDA read frozen features, with target labels. PAS reads features and labelled source "
               "data but needs no target labels.", body)


# ---------------------------------------------------------------------- charts
def _tick(v):
    return f"{v:g}"


def fig_hscore_transition(rows):
    x0, y0, w, h = 78, 52, 640, 250
    lo, hi = math.log10(0.03), math.log10(60)
    ylo, yhi = -0.25, 1.1
    X = lambda v: x0 + (math.log10(v) - lo) / (hi - lo) * w                   # noqa: E731
    Y = lambda v: y0 + h - (v - ylo) / (yhi - ylo) * h                        # noqa: E731
    body = [text(14, 24, "Seed-to-seed stability against target images per feature dimension (the Claßen sweep)",
                 "hd", "start")]
    # the ill-conditioned region, then grid and frame
    body.append(f'<rect x="{x0}" y="{y0}" width="{X(2) - x0:.1f}" height="{h}" class="band"/>')
    for v in (0, 0.5, 1):
        body.append(f'<path d="M {x0},{Y(v):.1f} H {x0 + w}" class="{"ref" if v == 0 else "gd"}"/>')
        body.append(text(x0 - 8, Y(v) + 4, _tick(v), "tiny", "end"))
    for v in (0.05, 0.1, 0.5, 1, 2, 5, 10, 30):
        body.append(f'<path d="M {X(v):.1f},{y0 + h} v 5" class="ax"/>')
        body.append(text(X(v), y0 + h + 19, _tick(v), "tiny"))
    body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" class="frame"/>')
    # markers for the boundary and for the paper's own regime
    body.append(f'<path d="M {X(2):.1f},{y0} V {y0 + h}" class="refd"/>')
    body.append(text(X(2) + 6, y0 + 14, "n / k ≈ 2", "sm", "start"))
    body.append(f'<path d="M {X(0.05):.1f},{y0} V {y0 + h}" class="ref"/>')
    body.append(text(X(0.05) + 6, y0 + 14, "Breast, 5% subset, ResNet-18: n / k ≈ 0.05", "sm", "start"))
    below = [r[5] for r in rows if r[2] < 2]
    above = [r[5] for r in rows if r[2] >= 2]
    lo_c, hi_c = f"{min(below):.0e}".replace("e+", "e"), f"{max(below):.0e}".replace("e+", "e")
    body.append(text(X(0.07), Y(0.42), "cond(S_T) ≈", "sm", "start"))
    body.append(text(X(0.07), Y(0.42) + 15, f"{lo_c} to {hi_c}", "sm", "start"))
    body.append(text(X(8), Y(0.72), f"cond(S_T) ≈ {min(above):.0f}–{max(above):.0f}", "sm"))
    # series
    for k, ls_leep, ls_h in ((64, "sb1", "sb2"), (256, "db1", "db2")):
        pts = [r for r in rows if r[0] == k]
        for col, cls, mk in ((3, ls_leep, "p1"), (4, ls_h, "p2")):
            d = "M " + " L ".join(f"{X(r[2]):.1f},{Y(r[col]):.1f}" for r in pts)
            body.append(f'<path d="{d}" class="{cls}"/>')
            for r in pts:
                body.append(f'<circle cx="{X(r[2]):.1f}" cy="{Y(r[col]):.1f}" r="4.2" class="{mk}"/>')
    # axes labels and legend
    body.append(text(x0 + w / 2, y0 + h + 40, "n / k   (target images per feature dimension, log scale)", "sm"))
    body.append(f'<text transform="translate(22,{y0 + h / 2}) rotate(-90)" class="sm" text-anchor="middle">'
                "seed-to-seed stability</text>")
    lx, ly = x0 + 8, y0 + h + 62
    for i, (cls, mk, lab) in enumerate((("sb1", "p1", "LEEP, k = 64"), ("sb2", "p2", "H-score, k = 64"),
                                         ("db1", "p1", "LEEP, k = 256"), ("db2", "p2", "H-score, k = 256"))):
        x = lx + i * 158
        body.append(f'<path d="M {x},{ly} h 26" class="{cls}"/><circle cx="{x + 13}" cy="{ly}" r="4" class="{mk}"/>')
        body.append(text(x + 34, ly + 4, lab, "sm", "start"))
    return svg(760, 392, "H-score stability collapses below n over k of about two",
               "Line chart on a log axis of target images per feature dimension. LEEP's seed-to-seed stability "
               "falls smoothly as n over k shrinks. H-score's stays near one above n over k of two, then drops to "
               "about zero and slightly below, where the covariance matrix is ill-conditioned. A marker shows that "
               "the Breast target at a five percent subset, with ResNet-18 features, sits at about 0.05.", body)


def fig_power(need, power, sd, tau_reported):
    body = [text(14, 24, "What a rank correlation over a handful of models can resolve (the Chaves tables)",
                 "hd", "start")]
    # left: power at true tau = 0.3
    x0, y0, w, h = 70, 60, 270, 210
    X = lambda n: x0 + (n - 0) / 90 * w                                       # noqa: E731
    Y = lambda p: y0 + h - p * h                                              # noqa: E731
    body.append(text(x0, y0 - 14, "Power to detect a true τ = 0.3", "v b", "start"))
    for v in (0, 0.25, 0.5, 0.75, 1):
        body.append(f'<path d="M {x0},{Y(v):.1f} H {x0 + w}" class="gd"/>' + text(x0 - 7, Y(v) + 4, _tick(v), "tiny", "end"))
    for n in power:
        body.append(text(X(n), y0 + h + 17, str(n), "tiny"))
    body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" class="frame"/>')
    d = "M " + " L ".join(f"{X(n):.1f},{Y(p):.1f}" for n, p in power.items())
    body.append(f'<path d="{d}" class="sb1"/>')
    for n, p in power.items():
        body.append(f'<circle cx="{X(n):.1f}" cy="{Y(p):.1f}" r="4.2" class="p1"/>')
        body.append(text(X(n), Y(p) - 10, f"{p:.2f}", "tiny"))
    body.append(text(x0 + w / 2, y0 + h + 38, "number of models ranked, n", "sm"))
    # right: |tau| needed for p < 0.05
    x1, w1 = 430, 270
    X1 = lambda n: x1 + (n - 0) / 60 * w1                                      # noqa: E731
    Y1 = lambda t: y0 + h - t / 0.6 * h                                        # noqa: E731
    body.append(text(x1, y0 - 14, "Smallest |τ| that clears p < 0.05", "v b", "start"))
    for v in (0, 0.2, 0.4, 0.6):
        body.append(f'<path d="M {x1},{Y1(v):.1f} H {x1 + w1}" class="gd"/>' + text(x1 - 7, Y1(v) + 4, _tick(v), "tiny", "end"))
    for n in need:
        body.append(text(X1(n), y0 + h + 17, str(n), "tiny"))
    body.append(f'<rect x="{x1}" y="{y0}" width="{w1}" height="{h}" class="frame"/>')
    d = "M " + " L ".join(f"{X1(n):.1f},{Y1(t):.1f}" for n, t in need.items())
    body.append(f'<path d="{d}" class="sb2"/>')
    for n, t in need.items():
        body.append(f'<circle cx="{X1(n):.1f}" cy="{Y1(t):.1f}" r="4.2" class="p2"/>')
        body.append(text(X1(n) + 8, Y1(t) - 9, f"{t:.3f}", "tiny", "start"))
    body.append(f'<path d="M {x1},{Y1(tau_reported):.1f} H {x1 + w1}" class="refd"/>')
    body.append(text(x1 + 8, Y1(tau_reported) + 17, f"a reported τ = {tau_reported:.3f}", "sm", "start"))
    body.append(text(x1 + w1 / 2, y0 + h + 38, "number of models ranked, n", "sm"))
    body.append(text(14, 330, f"At n = 10 the null standard deviation of τ is {sd[10]:.3f}; "
                              "the 42 cells of Table 2 were each computed from ten models.", "sm", "start"))
    return svg(760, 348, "Statistical resolution of a rank correlation over ten models",
               "Two panels. On the left, the power to detect a true Kendall tau of 0.3 rises from 0.14 with ten "
               "models to 0.74 with eighty. On the right, the smallest absolute tau that reaches p below 0.05 "
               "falls from 0.467 with ten models to 0.192 with fifty. A dashed line marks a reported tau of "
               "0.270, which is below the threshold at ten models.", body)


# ------------------------------------------------------------------ reused figs
def with_card(src_text):
    """Insert an opaque, theme-aware card behind an existing themed SVG."""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', src_text)
    assert m, "no viewBox"
    w, h = m.groups()
    card = f'<rect x="0" y="0" width="{w}" height="{h}" rx="6" class="bgcard"/>'
    css = ("  .bgcard{fill:#fbfaf6}\n  @media (prefers-color-scheme: dark){.bgcard{fill:#161615}}\n")
    out = src_text.replace("</style>", css + "</style>", 1)
    return re.sub(r"(</style>\s*)", lambda mo: mo.group(1) + card + "\n", out, count=1)


def write_figures(rows, need, power, sd, tau):
    FIGS.mkdir(exist_ok=True)
    files = {
        "pipeline.svg": fig_pipeline(),
        "ladder.svg": fig_ladder(),
        "hscore-n-over-k.svg": fig_hscore_transition(rows),
        "power-n10.svg": fig_power(need, power, sd, tau),
    }
    for name, src in REUSED.items():
        files[name] = with_card((PAPERS / src).read_text(encoding="utf-8"))
    for name, text_ in files.items():
        (FIGS / name).write_text(text_, encoding="utf-8")
    print("figures: wrote", ", ".join(sorted(files)))


if __name__ == "__main__":
    print("Transferability talk: the numbers behind its two charts")
    print("=" * 64, "\n")
    rows = classen_sweep()
    print("1. Claßen sweep (parsed from the note): intra-metric stability against n/k\n")
    print(f"   {'k':>4} {'n':>5} {'n/k':>6}  {'LEEP':>6}  {'H-score':>7}  {'cond(S_T)':>9}")
    for k, n, ratio, leep, h, cond in rows:
        print(f"   {k:>4} {n:>5} {ratio:>6.1f}  {leep:>6.3f}  {h:>7.3f}  {cond:>9.1e}")
    below = [r for r in rows if r[2] < 2]
    above = [r for r in rows if r[2] >= 2]
    print(f"\n   Below n/k = 2: H-score stability in [{min(r[4] for r in below):.3f}, "
          f"{max(r[4] for r in below):.3f}], cond(S_T) {min(r[5] for r in below):.1e}-{max(r[5] for r in below):.1e}.")
    print(f"   At n/k >= 2:   H-score stability in [{min(r[4] for r in above):.3f}, "
          f"{max(r[4] for r in above):.3f}], cond(S_T) {min(r[5] for r in above):.1e}-{max(r[5] for r in above):.1e}.")
    print("   ResNet-18 has k = 512 and the Breast 5% subset is about 27 images: "
          f"n/k = {27 / 512:.3f}, an order of magnitude below the lowest ratio swept.\n")

    need, power, sd = chaves_power()
    tau = reported_tau()
    print("2. Chaves resolution (parsed from the note)\n")
    print(f"   {'n':>4}  {'null sd':>8}  {'|tau| for p<0.05':>17}")
    for n in need:
        print(f"   {n:>4}  {sd[n]:>8.3f}  {need[n]:>17.3f}")
    print(f"\n   power at true tau = 0.3: " + ", ".join(f"n={n}: {p:.2f}" for n, p in power.items()))
    print(f"   a reported tau of {tau:.3f} is {tau / sd[10]:.2f} null standard deviations from zero at n = 10, "
          f"below the {need[10]:.3f} that p < 0.05 needs.\n")

    print("3. Two rankings with the same tau and rho (recomputed here)\n")
    print(f"   {'':>28}  {'tau':>6}  {'weighted tau':>12}  {'rho':>6}")
    for tag, (a, b, c) in rank_agreement().items():
        print(f"   {tag:>28}  {a:>6.3f}  {b:>12.3f}  {c:>6.3f}")
    print("\n   The weighted tau prefers the ranking that gets the TOP right (0.804 against")
    print("   0.307), as a top-weighted coefficient should; this matches the Claßen notes' table.\n")

    if "--figures" in sys.argv:
        write_figures(rows, need, power, sd, tau)
