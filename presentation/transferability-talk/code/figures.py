#!/usr/bin/env python3
"""Figures for the transferability talk.

Two diagrams hold no data, only structure: the problem pipeline and the progression
from NCE to PAS. Two figures from the individual papers' notes are copied in so the
deck can be previewed from this folder alone; each copy gets an opaque card behind it,
because a Marp slide is always white while the figures follow the viewer's dark mode.

One figure is cut from a paper rather than drawn: Figure 3 of Chaves et al. 2023
(arXiv:2308.07444), a 3 x 7 grid of scatter plots whose points exist only in the paper's
image. It is cropped from papers/2023-chaves-medical-transferability/paper.pdf with
pdftoppm and gitignored, so it follows the repo's rule that paper PDFs and anything
lifted from them are never committed. A fresh clone needs `make fetch` and a re-run of
this script before that slide shows its figure.

Standard library only (pdftoppm and pdftotext, from poppler, for the crop).

Run:  python3 figures.py --figures  (rewrites ../figures/*)
"""
from __future__ import annotations

import html
import re
import subprocess
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
    "hscore-between-vs-within.svg": "2019-bao-hscore-transferability/figures/between_vs_within.svg",
}


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
  .cen{stroke:#1a1a19;stroke-width:1.6}
  .off{fill:none;stroke:#b3b1a8;stroke-width:1.3;stroke-dasharray:5 4}
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
    .blue{fill:#7fb2e8} .orange{fill:#e09a5f} .off{stroke:#5d5b55} .cen{stroke:#eceae3}
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


def fig_nce_counts():
    """Three joint count tables P(y,z) for 18 examples: Z determines Y, partly, not at all."""
    panels = [
        ("Z determines Y", [[6, 0, 0], [0, 6, 0], [0, 0, 6]], "0", "each row hits one column"),
        ("Z partly informative", [[4, 2, 0], [0, 4, 2], [2, 0, 4]], "0.92", "rows are concentrated"),
        ("Z useless", [[2, 2, 2], [2, 2, 2], [2, 2, 2]], "1.58", "every row looks like the Y marginal"),
    ]
    cell, gx, gy = 50, 62, 92
    body = [text(14, 24, "NCE counts the joint of two label lists: how much of Y is left once Z is known",
                 "hd", "start")]
    for i, (title, grid, h, note) in enumerate(panels):
        x0 = 20 + i * 290
        body.append(text(x0 + gx + 1.5 * cell, 56, title, "v b"))
        body.append(text(x0 + gx + 1.5 * cell, 74, note, "tiny"))
        for j in range(3):
            body.append(text(x0 + gx + (j + 0.5) * cell, gy - 6, [("y", "it"), (str(j + 1), "sub")], "sm"))
            body.append(text(x0 + gx - 10, gy + (j + 0.5) * cell + 4.5, [("z", "it"), (str(j + 1), "sub")], "sm", "end"))
        for r in range(3):
            for c in range(3):
                n = grid[r][c]
                x, y = x0 + gx + c * cell, gy + r * cell
                body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" class="frame"/>')
                if n:
                    body.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" '
                                f'class="blue" fill-opacity="{0.12 + 0.88 * n / 6:.2f}"/>')
                    body.append(text(x + cell / 2, y + cell / 2 + 5, str(n), "v b"))
        body.append(text(x0 + gx + 1.5 * cell, gy + 3 * cell + 32,
                         [("H(Y | Z) = ", ""), (h + " bits", "")], "hd"))
    body.append(text(14, 322, "The same 18 examples in every panel; a cell counts the examples with source label z and "
                              "target label y. No model, no features.", "sm", "start"))
    return svg(890, 340, "Joint count tables for NCE",
               "Three 3 by 3 tables of counts of source label Z against target label Y over 18 examples. "
               "Left: Z determines Y, one filled cell per row, H(Y|Z) is 0. Middle: partly informative, "
               "H(Y|Z) is 0.92 bits. Right: Z is useless, all cells equal, H(Y|Z) equals H(Y), 1.58 bits.", body)


LEEP_EX = [  # (target label y, softmax over source classes z)
    (1, (0.7, 0.2, 0.1)), (1, (0.6, 0.3, 0.1)), (2, (0.1, 0.8, 0.1)), (3, (0.2, 0.1, 0.7)),
]


def leep_tables():
    hard = [[0] * 3 for _ in range(3)]
    soft = [[0.0] * 3 for _ in range(3)]
    for y, th in LEEP_EX:
        hard[th.index(max(th))][y - 1] += 1
        for z in range(3):
            soft[z][y - 1] += th[z]
    cond = [[c / sum(row) for c in row] for row in soft]  # P(y | z)
    return hard, soft, cond


def count_grid(body, x0, y0, grid, cell, fmt, top):
    for j in range(3):
        body.append(text(x0 + (j + 0.5) * cell, y0 - 6, [("y", "it"), (str(j + 1), "sub")], "sm"))
        body.append(text(x0 - 8, y0 + (j + 0.5) * cell + 4.5, [("z", "it"), (str(j + 1), "sub")], "sm", "end"))
    for r in range(3):
        for c in range(3):
            n = grid[r][c]
            x, y = x0 + c * cell, y0 + r * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" class="frame"/>')
            if n > 1e-9:
                body.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" class="blue" '
                            f'fill-opacity="{0.12 + 0.88 * min(n, top) / top:.2f}"/>')
                body.append(text(x + cell / 2, y + cell / 2 + 5, fmt(n), "v b"))


def mini_bars(body, x0, ybase, vals, hmax=40, w=16, gap=5, labels=None):
    for k, v in enumerate(vals):
        x = x0 + k * (w + gap)
        body.append(f'<rect x="{x}" y="{ybase - v * hmax:.1f}" width="{w}" height="{v * hmax:.1f}" class="blue" '
                    f'fill-opacity="0.75"/>')
    body.append(f'<path d="M {x0 - 3},{ybase} H {x0 + len(vals) * (w + gap) - gap + 3}" class="ax"/>')
    if labels:
        for k, lab in enumerate(labels):
            body.append(text(x0 + k * (w + gap) + w / 2, ybase + 13, lab, "tiny"))


def fig_leep_soft_counts():
    """Option A: NCE drops a whole count in one cell; LEEP spreads it by the softmax."""
    hard, soft, _ = leep_tables()
    cell = 46
    body = [text(14, 24, "LEEP counts like NCE, but each example spreads its count over the source classes by its softmax",
                 "hd", "start")]
    body.append(text(120, 56, "target examples", "v b"))
    body.append(text(120, 72, [("target label y, softmax ", ""), ("θ", "it"), ("(x)", "")], "tiny"))
    for i, (y, th) in enumerate(LEEP_EX):
        yb = 96 + i * 64
        body.append(f'<rect x="20" y="{yb}" width="200" height="54" rx="8" class="box"/>')
        body.append(text(34, yb + 31, [("x", "it"), (str(i + 1), "sub")], "v b", "start"))
        body.append(text(70, yb + 31, [("y = " + str(y), "")], "sm", "start"))
        mini_bars(body, 128, yb + 40, th, 30, 16, 5, ["z1", "z2", "z3"])
    body.append(text(393, 56, "NCE: hard count", "v b"))
    body.append(text(393, 72, "whole count to the top class", "tiny"))
    body.append(text(693, 56, "LEEP: soft count", "v b"))
    body.append(text(693, 72, "count split by the softmax", "tiny"))
    count_grid(body, 324, 96, hard, cell, lambda n: str(int(n)), 2)
    count_grid(body, 624, 96, soft, cell, lambda n: f"{n:.1f}", 2)
    body.append(arrow("M 232,165 H 290"))
    body.append(arrow("M 476,165 H 596"))
    body.append(text(536, 150, "soften", "tiny"))
    body.append(text(393, 262, "each column sums to the number of examples with that y", "tiny"))
    body.append(text(693, 262, "same totals; counts are now fractional", "tiny"))
    body.append(text(14, 374, "Normalising each row of the soft table gives P(y | z): the fixed linear map from "
                              "the source softmax to a target distribution.", "sm", "start"))
    return svg(890, 392, "From hard to soft counts",
               "Four target examples, each with a target label and a softmax over three source classes. Left table: "
               "NCE puts each example's whole count in the row of its top source class. Right table: LEEP splits the "
               "count across the rows by the softmax, giving fractional counts with the same column totals.", body)


def fig_leep_predictor():
    """Option B: the Expected Empirical Predictor as a fixed linear layer on the softmax."""
    _, _, cond = leep_tables()
    th = LEEP_EX[0][1]
    p = [sum(th[z] * cond[z][y] for z in range(3)) for y in range(3)]
    body = [text(14, 24, "LEEP's predictor: nothing is trained, the softmax goes through one counted matrix",
                 "hd", "start")]
    xs = [17 + i * 145 for i in range(6)]
    bw, by, bh = 118, 66, 78
    labels = [
        ("Target input", [[("x", "it"), ("i", "sub")]], "box"),
        ("Source model", ["frozen"], "box"),
        ("Source softmax", [[("θ", "it"), ("(x", ""), ("i", "sub"), (")", "")]], "need"),
        ("Fixed matrix", [[("P̂(y | z)", "")], "counted once"], "free"),
        ("Target prediction", [[("p(y | x", ""), ("i", "sub"), (")", "")]], "need"),
        ("Score term", [[("log p(y", ""), ("i", "sub"), (" | x", ""), ("i", "sub"), (")", "")]], "box"),
    ]
    for x, (title, lines, cls) in zip(xs, labels):
        body.append(box(x, by, bw, bh, cls, [title] + lines))
    for i in range(5):
        body.append(arrow(f"M {xs[i] + bw + 3},{by + bh / 2} H {xs[i + 1] - 5}"))
    yb = 214
    mini_bars(body, xs[2] + 30, yb, th, 40, 18, 6, ["z1", "z2", "z3"])
    mini_bars(body, xs[4] + 30, yb, p, 40, 18, 6, ["y1", "y2", "y3"])
    body.append(text(xs[2] + bw / 2, yb + 34, "e.g. (0.7, 0.2, 0.1)", "tiny"))
    body.append(text(xs[4] + bw / 2, yb + 34, "(" + ", ".join(f"{v:.2f}" for v in p) + ")", "tiny"))
    for x in (xs[2], xs[4]):
        body.append(f'<path d="M {x + bw / 2},{by + bh + 4} V {yb - 52}" class="ref"/>')
    body.append(text(xs[5] + bw / 2, yb - 6, [("true y", ""), ("i", "sub"), (" = 1:", "")], "sm"))
    body.append(text(xs[5] + bw / 2, yb + 14, f"log {p[0]:.2f} = {__import__('math').log(p[0]):.2f}", "v b"))
    body.append(text(xs[5] + bw / 2, yb + 34, "LEEP = average over i", "tiny"))
    body.append(text(14, 296, "Expected Empirical Predictor: p(y | x) = Σ_z P̂(y | z) θ(x)_z. The only fitted object is "
                              "the table P̂, and it is counted, not trained.", "sm", "start"))
    return svg(890, 314, "LEEP as a fixed linear layer",
               "Flow from a target input through the frozen source model to a source softmax, through a fixed counted "
               "matrix P-hat of y given z, to a target prediction, and finally the log probability of the true label, "
               "averaged over examples to give the LEEP score.", body)


PREMISE_SLOTS = ["Source data", "Target inputs", "Extractor", "Head", "Target labels"]

# per score: headline, and for each of the five slots (used?, title, sub-lines), the reads box, the flows
PREMISES = {
    "nce": ("Premise: two tasks labelled on the same inputs, and nothing else",
            [(1, "Source labels z", ["on the shared x"]),
             (1, "Same inputs x", ["both tasks use them"]),
             (0, "Extractor", ["φm", "no model"]),
             (0, "Head", ["hm", "no model"]),
             (1, "Target labels y", ["on the same x"])],
            ["Reads", "label lists z and y", "nothing else"],
            [("adj", 4), ("arc", 0, 5, 20)]),
    "leep": ("Premise loosened: source and target inputs may differ; the head's softmax stands in for z",
             [(0, "Source data", ["not needed"]),
              (1, "Target inputs x", ["not the source's"]),
              (1, "Extractor φm", ["frozen"]),
              (1, "Head hm", ["frozen; gives θ(x)"]),
              (1, "Target labels y", [])],
             ["Reads", "θ(x) as a dummy z", "and the labels y"],
             [("adj", 1), ("adj", 2), ("arc", 3, 5, 20), ("adj", 4)]),
    "hscore": ("Premise loosened again: only the feature extractor φm, no head",
               [(0, "Source data", ["not needed"]),
                (1, "Target inputs x", []),
                (1, "Extractor φm", ["features f(x)"]),
                (0, "Head hm", ["not needed"]),
                (1, "Target labels y", [])],
               ["Reads", "features f(x)", "and labels y"],
               [("adj", 1), ("arc", 2, 5, 20), ("adj", 4)]),
    "logme": ("Same premise as H-score: φm and target labels, which may be real-valued",
              [(0, "Source data", ["not needed"]),
               (1, "Target inputs x", []),
               (1, "Extractor φm", ["features f(x)"]),
               (0, "Head hm", ["not needed"]),
               (1, "Target labels y", ["class or real value"])],
              ["Reads", "features f(x)", "and labels y"],
              [("adj", 1), ("arc", 2, 5, 20), ("adj", 4)]),
    "sfda": ("Same premise as H-score: φm, target inputs and target labels",
             [(0, "Source data", ["not needed"]),
              (1, "Target inputs x", []),
              (1, "Extractor φm", ["features f(x)"]),
              (0, "Head hm", ["not needed"]),
              (1, "Target labels y", [])],
             ["Reads", "features f(x)", "and labels y"],
             [("adj", 1), ("arc", 2, 5, 20), ("adj", 4)]),
    "pas": ("Premise flipped: labels on the source side, none on the target",
            [(1, "Source data", ["(x, z), labelled"]),
             (1, "Target inputs x", ["unlabelled"]),
             (1, "Extractor φm", ["embeddings f(x)"]),
             (0, "Head hm", ["not needed"]),
             (0, "Target labels y", ["none"])],
            ["Reads", "embeddings f(x)", "+ source centroids"],
            [("arc", 0, 2, 20), ("adj", 1), ("arc", 2, 5, 38)]),
}


def sub_m(s):
    """Turn the tokens φm and hm into φ_m and h_m with a real subscript."""
    parts = []
    for tok in re.split(r"(φm|hm)", s):
        if tok in ("φm", "hm"):
            parts += [(tok[0], "it"), ("m", "sub")]
        elif tok:
            parts.append((tok, ""))
    return parts


def fig_premise(key):
    """A premise strip: which of the five inputs a score uses (blue), which it ignores (dashed)."""
    head, slots, reads, flows = PREMISES[key]
    bw, bh, by, step, x0 = 124, 62, 38, 148, 14
    xs = [x0 + i * step for i in range(6)]
    body = [text(14, 24, sub_m(head), "hd", "start")]

    def draw(i, used, title, subs, cls=None):
        c = cls or ("need" if used else "off")
        out = [f'<rect x="{xs[i]}" y="{by}" width="{bw}" height="{bh}" rx="8" class="{c}"/>']
        lines = [title] + subs
        y0 = by + bh / 2 - (len(lines) - 1) * 16 / 2 + 4.5
        for k, ln in enumerate(lines):
            c2 = ("v b" if used else "sm") if k == 0 else ("sm" if used else "tiny")
            out.append(text(xs[i] + bw / 2, y0 + k * 16, sub_m(ln), c2))
        return out

    for i, (used, title, subs) in enumerate(slots):
        body += draw(i, used, title, subs)
    body += draw(5, 1, reads[0], reads[1:], "free")
    ymid = by + bh / 2
    for f in flows:
        if f[0] == "adj":
            i = f[1]
            body.append(arrow(f"M {xs[i] + bw + 3},{ymid} H {xs[i + 1] - 5}"))
        else:
            _, i, j, d = f
            body.append(arrow(f"M {xs[i] + bw / 2 + 18},{by + bh + 3} V {by + bh + d} "
                             f"H {xs[j] + bw / 2 - 18} V {by + bh + 5}"))
    body.append(f'<rect x="14" y="146" width="14" height="10" rx="2" class="need"/>')
    body.append(text(34, 155, "used", "tiny", "start"))
    body.append(f'<rect x="84" y="146" width="14" height="10" rx="2" class="off"/>')
    body.append(text(104, 155, "not used", "tiny", "start"))
    body.append(f'<rect x="170" y="146" width="14" height="10" rx="2" class="free"/>')
    body.append(text(190, 155, "what the score reads", "tiny", "start"))
    return svg(890, 166, f"Premise of {key}", head + ". Boxes: source data, target inputs, feature extractor, head, "
               "target labels; used ones are filled, unused ones dashed; a final box shows what the score reads.", body)


def fig_hscore_whitening():
    """Two cases with the same centroid gap, raw and then divided by the total spread."""
    import math
    import random
    rnd = random.Random(7)
    # (title, centroid axis, within-class std (x, y), centroid offset on that axis)
    cases = [("Classes differ along a quiet direction", "y", (3.0, 0.5), 1.0),
             ("Classes differ along a noisy direction", "x", (3.0, 0.5), 1.0)]
    cx = [225, 665]
    sr, sw = 22, 38
    ry, wy = 128, 312
    body = [text(14, 24, "H-score: what cov(f(X))⁻¹ does: measure the class gap in units of the total spread along it",
                 "hd", "start")]
    for (title, axis, (wx_, wy_), off), c in zip(cases, cx):
        ia = 0 if axis == "x" else 1
        within = [wx_, wy_]
        tot = [math.sqrt(within[0] ** 2 + (off ** 2 if ia == 0 else 0)),
               math.sqrt(within[1] ** 2 + (off ** 2 if ia == 1 else 0))]
        body.append(text(c, 46, title, "v b"))
        for row, (yc, scale, lab) in enumerate([(ry, sr, "raw features f(x)"), (wy, sw, "divided by the total spread")]):
            top, hh = (yc - 70, 140) if row == 0 else (yc - 85, 170)
            body.append(f'<rect x="{c - 205}" y="{top}" width="410" height="{hh}" rx="6" class="frame"/>')
            body.append(text(c - 197, top + hh - 7, lab, "tiny", "start"))
        pts = {}
        for k, sign in enumerate((-1, 1)):
            for _ in range(32):
                v = [rnd.gauss(0, within[0]), rnd.gauss(0, within[1])]
                v[ia] += sign * off
                v[0] = max(-8.8, min(8.8, v[0]))
                pts.setdefault(sign, []).append(v)
        for sign, cls in ((-1, "p1"), (1, "p2")):
            for v in pts[sign]:
                body.append(f'<circle cx="{c + v[0] * sr:.1f}" cy="{ry - v[1] * sr:.1f}" r="2.6" class="{cls}"/>')
                body.append(f'<circle cx="{c + v[0] / tot[0] * sw:.1f}" cy="{wy - max(-80, min(80, v[1] / tot[1] * sw)):.1f}" '
                            f'r="2.6" class="{cls}"/>')
        # total-spread ellipse (raw) and unit circle (whitened), then the centroids and their gap
        body.append(f'<ellipse cx="{c}" cy="{ry}" rx="{tot[0] * sr:.1f}" ry="{tot[1] * sr:.1f}" class="refd"/>')
        body.append(f'<circle cx="{c}" cy="{wy}" r="{sw}" class="refd"/>')
        for sign, cls in ((-1, "p1"), (1, "p2")):
            a, b = (sign * off, 0) if ia == 0 else (0, sign * off)
            body.append(f'<circle cx="{c + a * sr:.1f}" cy="{ry - b * sr:.1f}" r="5.5" class="{cls} cen"/>')
            body.append(f'<circle cx="{c + a / tot[0] * sw:.1f}" cy="{wy - b / tot[1] * sw:.1f}" r="5.5" '
                        f'class="{cls} cen"/>')
        gap_raw = 2 * off
        gap_w = 2 * off / tot[ia]
        body.append(text(c + 197, ry - 70 + 16, f"gap = {gap_raw:.0f}", "sm", "end"))
        body.append(text(c + 197, wy - 85 + 16, f"gap = {gap_w:.2f} spreads", "sm", "end"))
        body.append(arrow(f"M {c},{ry + 74} V {wy - 89}"))
        body.append(text(c + 10, (ry + wy) / 2 - 8, "× Σ^(−1/2)", "sm", "start"))
        body.append(text(c, wy + 108, f"H = (gap / 2)² = {(gap_w / 2) ** 2:.2f}", "hd"))
    body.append(text(14, 444, "Dashed: one standard deviation of the total spread (class means included). After dividing "
                              "by it that spread is 1 in every direction,", "sm", "start"))
    body.append(text(14, 462, "so the class gap is read directly as a fraction of the spread along it.", "sm", "start"))
    return svg(890, 478, "What the inverse covariance does in the H-score",
               "Two columns, each with the same centroid gap of 2 between two classes. Left: the gap lies along a "
               "quiet direction; right: along a noisy one. Top: raw features with the total-spread ellipse. Bottom: "
               "after dividing by the total spread, the ellipse is a unit circle and the gap is 1.79 versus 0.63, "
               "giving H of 0.80 versus 0.10.", body)


def fig_hscore_worked():
    """The two four-point datasets of the supplementary worked example, drawn exactly."""
    import math
    cases = [
        ("A: split along the quiet x₂", [(-3, -1.5), (3, -0.5)], [(-3, 1.5), (3, 0.5)],
         "diag(0, 1)", "diag(9, 1.25)", "H = 0/9 + 1/1.25 = 0.80", (3.0, math.sqrt(1.25)), 1),
        ("B: split along the noisy x₁", [(-4, -0.5), (2, 0.5)], [(-2, 0.5), (4, -0.5)],
         "diag(1, 0)", "diag(10, 0.25)", "H = 1/10 + 0/0.25 = 0.10", (math.sqrt(10), 0.5), 0),
    ]
    cx, sx_px, sy_px, cy = [225, 665], 34, 46, 150
    body = [text(14, 24, "H-score worked example: same centroid gap, different spread along it", "hd", "start")]
    for (title, c0, c1, sb, sg, h, (sdx, sdy), axis), c in zip(cases, cx):
        X = lambda x: c + x * sx_px
        Y = lambda y: cy - y * sy_px
        body.append(text(c, 50, title, "v b"))
        body.append(f'<rect x="{c - 205}" y="62" width="410" height="176" rx="6" class="frame"/>')
        body.append(f'<path d="M {c - 190},{cy} H {c + 190}" class="ax"/><path d="M {c},{cy - 82} V {cy + 82}" class="ax"/>')
        body.append(text(c + 192, cy + 14, [("x", "it"), ("1", "sub")], "tiny", "end"))
        body.append(text(c + 8, 80, [("x", "it"), ("2", "sub")], "tiny", "start"))
        body.append(f'<ellipse cx="{c}" cy="{cy}" rx="{sdx * sx_px:.1f}" ry="{sdy * sy_px:.1f}" class="refd"/>')
        mus = []
        for pts in (c0, c1):
            mx = sum(q[0] for q in pts) / 2
            my = sum(q[1] for q in pts) / 2
            mus.append((mx, my))
            for (x, y) in pts:
                body.append(f'<path d="M {X(x):.1f},{Y(y):.1f} L {X(mx):.1f},{Y(my):.1f}" class="ref"/>')
        for pts, cls, (mx, my) in ((c0, "p1", mus[0]), (c1, "p2", mus[1])):
            for (x, y) in pts:
                body.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" class="{cls}"/>')
                right = x >= 0
                body.append(text(X(x) + (8 if right else -8), Y(y) + (-8 if y > 0 else 15),
                                 f"({x:g}, {y:g})", "tiny", "start" if right else "end"))
        for (mx, my), cls in zip(mus, ("p1", "p2")):
            body.append(f'<circle cx="{X(mx):.1f}" cy="{Y(my):.1f}" r="6.5" class="{cls} cen"/>')
        (ax_, ay_), (bx_, by_) = mus
        body.append(f'<path d="M {X(ax_):.1f},{Y(ay_):.1f} L {X(bx_):.1f},{Y(by_):.1f}" class="e"/>')
        if axis == 1:
            body.append(text(c + 12, cy - 6, "gap = 2", "sm", "start"))
        else:
            body.append(text(c, cy + 38, "gap = 2", "sm"))
        body.append(text(c, 262, [("Σ", "it"), ("B", "sub"), (" = " + sb + "     ·     Σ = " + sg, "")], "sm"))
        body.append(text(c, 288, h, "hd"))
    body.append(text(14, 322, "Large dots: class centroids; thin lines: each point to its centroid (the within-class "
                              "scatter); dashed ellipse: one standard deviation of the", "sm", "start"))
    body.append(text(14, 340, "total spread Σ. The vertical scale is stretched so the small spread along x₂ is visible.",
                     "sm", "start"))
    return svg(890, 356, "The H-score worked example",
               "Two four-point datasets with the same centroid gap of 2. In A the classes split along x2, where the "
               "total spread is small, giving H of 0.80. In B they split along x1, where the spread is large, giving 0.10.",
               body)


def fig_logme_worked():
    """LogME by hand: the evidence N(0, C) in the plane of two examples, with the target y marked."""
    import math
    cases = [
        ("Good feature F = (1, −1)", 45, "2/3", "−2.72", "−1.36"),
        ("Bad feature F = (1, 1)", -45, "2", "−3.39", "−1.69"),
    ]
    cx, cy, sc = [225, 665], 150, 26
    body = [text(14, 24, "LogME worked example: where the target y falls under the evidence N(0, C), C = FFᵀ + I",
                 "hd", "start")]
    for (title, rot, q, lp, per), c in zip(cases, cx):
        body.append(text(c, 50, title, "v b"))
        body.append(f'<rect x="{c - 205}" y="62" width="410" height="176" rx="6" class="frame"/>')
        body.append(f'<path d="M {c - 190},{cy} H {c + 190}" class="ax"/><path d="M {c},{cy - 82} V {cy + 82}" class="ax"/>')
        body.append(text(c + 192, cy + 14, [("y", "it"), ("1", "sub")], "tiny", "end"))
        body.append(text(c + 8, 80, [("y", "it"), ("2", "sub")], "tiny", "start"))
        # long axis of C has eigenvalue 3 along F, short has 1; rings at 1 and 2 standard deviations
        for d, lab in ((1, "1σ"), (2, "2σ")):
            body.append(f'<ellipse cx="{c}" cy="{cy}" rx="{math.sqrt(3) * d * sc:.1f}" ry="{d * sc:.1f}" '
                        f'transform="rotate({rot} {c} {cy})" class="refd"/>')
        ang = math.radians(rot)
        dx, dy = math.cos(ang), math.sin(ang)
        body.append(f'<path d="M {c - dx * 100:.1f},{cy - dy * 100:.1f} L {c + dx * 100:.1f},{cy + dy * 100:.1f}" '
                    f'class="ref"/>')
        body.append(text(c + dx * 100 + 6, cy + dy * 100 + (12 if dy > 0 else -4), "F", "sm", "start"))
        px, py = c + 1 * sc, cy + 1 * sc            # y = (1, -1) in data coordinates
        body.append(f'<circle cx="{px}" cy="{py}" r="6" class="p2 cen"/>')
        body.append(text(px + 10, py + 22, "y = (1, −1)", "v b", "start"))
        body.append(text(c, 262, [("y", "it"), ("ᵀ", ""), ("C", "it"), ("⁻¹", ""), ("y = " + q + "     ·     log p(y) = " + lp, "")],
                         "sm"))
        body.append(text(c, 288, "LogME (α = β = 1) = " + per, "hd"))
    body.append(text(14, 322, "Dashed rings: 1 and 2 standard deviations of the evidence N(0, C); thin line: the feature "
                              "direction F, the long axis of the", "sm", "start"))
    body.append(text(14, 340, "ring (variance 3; the short axis has variance 1). y sits inside the 1σ ring when F is "
                              "aligned with it, and outside it otherwise.", "sm", "start"))
    return svg(890, 356, "LogME worked example",
               "Two panels in the plane of two examples. The ellipses are the Gaussian evidence of the target y given "
               "a feature. With the aligned feature, y lies on the long axis inside the 1 sigma ring and LogME is -1.36; "
               "with the orthogonal feature, y lies on the short axis outside it and LogME is -1.69.", body)


def fig_logme_evidence_vs_likelihood():
    """LogME: a rigid and a flexible model fit y = (1, -1) equally well, but the evidence tells them apart."""
    import math
    # (title, long-axis rotation or None for a circle, ring radii in units of d, rows: (max-likelihood, evidence))
    models = [("Model A, rigid: one feature F = (1, −1)", 45, (math.sqrt(3), 1.0),
               [("−1.84", "−2.72"), ("−2.84", "−3.39")]),
              ("Model B, flexible: two features F = I", None, (math.sqrt(2), math.sqrt(2)),
               [("−1.84", "−3.03"), ("−1.84", "−3.03")])]
    cx, cy, sc = [225, 665], 140, 26
    body = [text(14, 24, "LogME: the same best-case fit, but different evidence (α = β = 1, n = 2)", "hd", "start")]
    targets = [("p2", "y = (1, −1)", (1, -1)), ("p1", "y = (1, 1)", (1, 1))]
    for (title, rot, (rl, rs), rows), c in zip(models, cx):
        body.append(text(c, 50, title, "v b"))
        body.append(f'<rect x="{c - 205}" y="62" width="410" height="156" rx="6" class="frame"/>')
        body.append(f'<path d="M {c - 190},{cy} H {c + 190}" class="ax"/><path d="M {c},{cy - 72} V {cy + 72}" class="ax"/>')
        body.append(text(c + 192, cy + 14, [("y", "it"), ("1", "sub")], "tiny", "end"))
        body.append(text(c + 8, 78, [("y", "it"), ("2", "sub")], "tiny", "start"))
        for d in (1, 2):
            tr = f' transform="rotate({rot} {c} {cy})"' if rot is not None else ""
            body.append(f'<ellipse cx="{c}" cy="{cy}" rx="{rl * d * sc:.1f}" ry="{rs * d * sc:.1f}"{tr} class="refd"/>')
        for cls, lab, (a, b) in targets:
            body.append(f'<circle cx="{c + a * sc}" cy="{cy - b * sc}" r="6" class="{cls} cen"/>')
        body.append(text(c + 1 * sc + 10, cy + 1 * sc + 18, "y = (1, −1)", "sm", "start"))
        body.append(text(c + 1 * sc + 10, cy - 1 * sc - 8, "y = (1, 1)", "sm", "start"))
        # a small table: best-case likelihood against evidence, for each target
        body.append(text(c - 20, 244, "best-case likelihood", "tiny", "end"))
        body.append(text(c + 120, 244, "evidence", "tiny", "end"))
        for k, ((cls, lab, _), (ml, ev)) in enumerate(zip(targets, rows)):
            yy = 264 + k * 22
            body.append(f'<circle cx="{c - 180}" cy="{yy - 4}" r="5" class="{cls} cen"/>')
            body.append(text(c - 168, yy, lab, "sm", "start"))
            body.append(text(c - 20, yy, ml, "sm", "end"))
            body.append(text(c + 120, yy, ev, "v b", "end"))
    body.append(text(14, 328, "Dashed: 1 and 2 standard deviations of each model's evidence density N(0, C). For "
                              "y = (1, −1) the likelihood ties,", "sm", "start"))
    body.append(text(14, 346, "because both models fit it exactly; the evidence prefers the rigid model, which predicted it. "
                              "For y = (1, 1) it prefers B.", "sm", "start"))
    return svg(890, 362, "LogME: evidence versus likelihood",
               "Two panels, a rigid model with one feature and a flexible model with two. Each shows the evidence "
               "density as rings and two targets, y equal (1, -1) and (1, 1). The best-case likelihood is minus 1.84 "
               "for both models on the first target, but the evidence is minus 2.72 for the rigid model and minus "
               "3.03 for the flexible one.", body)


def fig_logme_integral():
    """LogME: the prior, the likelihood as a function of w, their product, and the area that is the evidence."""
    import math
    prior = lambda w: math.exp(-w * w / 2) / math.sqrt(2 * math.pi)                  # N(w; 0, 1)
    lik = lambda w: math.exp(-(w - 1) ** 2) / (2 * math.pi)                          # N(y; Fw, I), F = y = (1, -1)
    prod = lambda w: prior(w) * lik(w)
    lo, hi, n = -3.0, 4.0, 200
    ws = [lo + (hi - lo) * i / n for i in range(n + 1)]
    area = sum(prod(w) for w in ws) * (hi - lo) / n
    assert abs(math.log(area) + 2.7205) < 5e-3 and abs(math.log(lik(1)) + 1.8379) < 1e-3
    pw, gap, x0, base, top = 262, 40, 14, 206, 84
    panels = [("Prior  p(w) = N(w; 0, 1)", prior, "sb1", "blue", 0.4, ("favours weights near 0", "N(0, α⁻¹) with α = 1")),
              ("Likelihood  p(y | w), as a function of w", lik, "sb2", "orange", lik(1), ("peaks at the best fit w = 1", "height 0.159 = e^(−1.84)")),
              ("Product  p(w) · p(y | w)", prod, "sb1", "blue", prod(2 / 3), ("shaded area = the evidence", "0.066 = e^(−2.72)"))]
    body = [text(14, 24, "LogME: the evidence integral adds up prior × fit over every possible weight w "
                         "(toy case: D = 1, α = β = 1, F = y = (1, −1))", "hd", "start")]
    for k, (title, f, sk, fill, peak, note) in enumerate(panels):
        x = x0 + k * (pw + gap)
        sx = (pw - 20) / (hi - lo)
        X = lambda w: x + 10 + (w - lo) * sx
        Y = lambda v: base - v / peak * top
        body.append(text(x + pw / 2, 50, title, "v b"))
        body.append(f'<rect x="{x}" y="62" width="{pw}" height="168" rx="6" class="frame"/>')
        body.append(f'<path d="M {x + 8},{base} H {x + pw - 8}" class="ax"/>')
        pts = " L ".join(f"{X(w):.1f},{Y(f(w)):.1f}" for w in ws)
        body.append(f'<path d="M {X(lo):.1f},{base} L {pts} L {X(hi):.1f},{base} Z" class="{fill}" '
                    f'fill-opacity="{0.30 if k == 2 else 0.14}" stroke="none"/>')
        body.append(f'<path d="M {pts}" class="{sk}"/>')
        for w0 in (-2, -1, 0, 1, 2, 3):
            body.append(f'<path d="M {X(w0):.1f},{base} v 4" class="ax"/>')
            body.append(text(X(w0), base + 16, str(w0), "tiny"))
        body.append(text(x + pw - 10, base - 6, [("w", "it")], "tiny", "end"))
        body.append(text(x + pw / 2, 78, note[0], "tiny"))
        body.append(text(x + pw / 2, 95, note[1], "sm" if k == 0 else "v b"))
    # markers: best-fit weight (likelihood) and posterior mean (product)
    x2 = x0 + 1 * (pw + gap); X2 = lambda w: x2 + 10 + (w - lo) * (pw - 20) / (hi - lo)
    body.append(f'<path d="M {X2(1):.1f},{base} V {base - top}" class="ref"/>')
    body.append(text(X2(1) + 5, base - 6, "w* = 1", "tiny", "start"))
    x3 = x0 + 2 * (pw + gap); X3 = lambda w: x3 + 10 + (w - lo) * (pw - 20) / (hi - lo)
    body.append(f'<path d="M {X3(2 / 3):.1f},{base} V {base - top}" class="ref"/>')
    body.append(text(X3(2 / 3) + 5, base - 6, "m = 2/3", "tiny", "start"))
    body.append(text(x0 + pw + gap / 2, 150, "×", "hd"))
    body.append(text(x0 + 2 * pw + gap * 1.5, 150, "=", "hd"))
    body.append(text(14, 262, "Maximum likelihood takes the peak of the middle curve. The evidence takes the area under the "
                              "right curve: every weight votes,", "sm", "start"))
    body.append(text(14, 280, "in proportion to how plausible it is a priori (left) and how well it explains y (middle). "
                              "Each panel has its own vertical scale.", "sm", "start"))
    return svg(890, 298, "LogME: what the evidence integral does",
               "Three panels over the weight w. Left: the Gaussian prior, centred at 0. Middle: the likelihood of y as a "
               "function of w, a Gaussian peaked at the best-fit weight 1 with height 0.159. Right: their product, peaked "
               "at the posterior mean 2/3; the area under it is the evidence 0.066, which is e to the minus 2.72.", body)


def fig_sfda_worked():
    """SFDA's ConfMix on a 1-D binary toy: two models tie on the first pass and separate on the second."""
    import math
    sig = lambda z: 1 / (1 + math.exp(-z))

    def run(a):
        mu = sum(a) / len(a)
        p = [sig(2 * mu * x) for x in a]
        xt = [pi * x + (1 - pi) * (-mu) for x, pi in zip(a, p)]
        mut = sum(xt) / len(xt)
        pt = [sig(2 * mut * x) for x in xt]
        return p, xt, pt, sum(map(math.log, p)) / len(a), sum(map(math.log, pt)) / len(a)

    models = [("Model X: four equally moderate samples", [-1, -1, -1, -1]),
              ("Model Y: one very confident, one ambiguous", [-3.5, -1, -0.25, -1])]
    r1, r2 = run(models[0][1]), run(models[1][1])
    assert abs(r1[3] - r2[3]) < 1e-3 and abs(r1[4] + 0.273) < 1e-3 and abs(r2[4] + 0.339) < 1e-3
    cx, sc, ya, yb = [224, 666], 45, 112, 196
    body = [text(14, 24, "SFDA worked example: ConfMix on a binary toy; the two models tie on the first pass", "hd", "start")]
    for (title, a), c, res in zip(models, cx, (r1, r2)):
        p, xt, pt, l1, l2 = res
        X = lambda v: c + v * sc
        body.append(text(c, 50, title, "v b"))
        body.append(f'<rect x="{c - 210}" y="62" width="420" height="168" rx="6" class="frame"/>')
        body.append(text(c - 206, ya + 14, "before", "tiny", "start"))
        body.append(text(c - 204, yb - 8, "after", "tiny", "start"))
        for yy in (ya, yb):
            body.append(f'<path d="M {c - 198},{yy} H {c + 198}" class="ax"/>')
        body.append(f'<path d="M {c},{ya - 22} V {yb + 26}" class="ref"/>')
        seen = {}
        for x, pi, x2, p2 in zip(a, p, xt, pt):
            k = seen.get(round(x, 3), 0)
            seen[round(x, 3)] = k + 1
            for sgn, cls in ((1, "p1"), (-1, "p2")):
                body.append(f'<path d="M {X(sgn * x):.1f},{ya - 6 * k} L {X(sgn * x2):.1f},{yb - 6 * k}" class="ref"/>')
                body.append(f'<circle cx="{X(sgn * x):.1f}" cy="{ya - 6 * k}" r="4" class="{cls}"/>')
                body.append(f'<circle cx="{X(sgn * x2):.1f}" cy="{yb - 6 * k}" r="4" class="{cls}"/>')
            if k == 0:
                n = sum(1 for q in a if round(q, 3) == round(x, 3))
                body.append(text(X(x), ya - 6 * (n - 1) - 11, f"{pi:.2f}", "tiny"))
                body.append(text(X(x2), yb + 18, f"{p2:.2f}", "tiny"))
        body.append(text(c, 248, [("mean log p:  " + f"{l1:.3f}".replace("-", "−") + "  →  " + f"{l2:.3f}".replace("-", "−"), "")], "hd"))
    body.append(text(14, 282, "Blue: class A, orange: class B (its mirror image); numbers: the confidence p of the class-A "
                              "samples before and after the move.", "sm", "start"))
    body.append(text(14, 300, "Both models score −0.127 on the first pass; Y's ambiguous sample (0.67) is dragged across "
                              "the dashed boundary (0.32).", "sm", "start"))
    return svg(890, 316, "SFDA worked example",
               "Two one-dimensional binary toy models. Each shows class A and mirrored class B before and after "
               "ConfMix, with the confidence of the class A samples. Both tie at minus 0.127 on the first pass; after "
               "ConfMix model X scores minus 0.273 and model Y minus 0.339, because its ambiguous sample crossed "
               "the decision boundary.", body)


def fig_sfda_lda():
    """LDA on the SFDA toy (Model Y, class A samples): two Gaussians, one boundary, one sigmoid."""
    import math
    mu = -1.4375
    pts = [-3.5, -1.0, -0.25, -1.0]
    sig = lambda z: 1 / (1 + math.exp(-z))
    c, sc = 445, 74
    X = lambda v: c + v * sc
    ya, ph, yb, pb = 168, 100, 316, 92
    gauss = lambda v, m: math.exp(-(v - m) ** 2 / 2) / math.sqrt(2 * math.pi)
    xs = [-5 + 10 * i / 200 for i in range(201)]
    body = [text(14, 24, "LDA on the toy (Model Y): each class is a Gaussian with the same width; "
                         "the posterior p(A | x) follows from Bayes' rule", "hd", "start")]
    body.append(f'<rect x="14" y="40" width="862" height="140" rx="6" class="frame"/>')
    body.append(f'<rect x="14" y="196" width="862" height="140" rx="6" class="frame"/>')
    body.append(text(24, 56, "class-conditional densities  p(x | A), p(x | B)", "tiny", "start"))
    body.append(f'<text transform="rotate(-90 36 110)" x="36" y="110" class="tiny" text-anchor="middle">density p(x | class)</text>')
    body.append(f'<text transform="rotate(-90 36 266)" x="36" y="266" class="tiny" text-anchor="middle">probability p(A | x)</text>')
    body.append(text(24, 212, "posterior  p(A | x)", "tiny", "start"))
    # top panel: the two class Gaussians
    for m, fillc, sk in ((mu, "blue", "sb1"), (-mu, "orange", "sb2")):
        pts_ = " L ".join(f"{X(v):.1f},{ya - gauss(v, m) * ph:.1f}" for v in xs)
        body.append(f'<path d="M {X(-5):.1f},{ya} L {pts_} L {X(5):.1f},{ya} Z" class="{fillc}" fill-opacity="0.16" stroke="none"/>')
        body.append(f'<path d="M {pts_}" class="{sk}"/>')
        body.append(f'<path d="M {X(m):.1f},{ya - gauss(m, m) * ph:.1f} V {ya}" class="ref"/>')
    body.append(f'<path d="M 24,{ya} H 866" class="ax"/>')
    body.append(text(X(mu) - 6, ya - gauss(mu, mu) * ph - 8, "class A, mean μ_A = −1.44", "sm", "end"))
    body.append(text(X(-mu) + 6, ya - gauss(mu, mu) * ph - 8, "class B, mean μ_B = +1.44 (mirror)", "sm", "start"))
    for v in pts:
        body.append(f'<circle cx="{X(v):.1f}" cy="{ya}" r="5" class="p1"/>')
    # bottom panel: the posterior is a sigmoid in x
    zs = " L ".join(f"{X(v):.1f},{pb + 196 + 44 - sig(2 * mu * v) * 88:.1f}" for v in xs)
    yb0 = 196 + 44 + 88 + 0   # unused
    base = 322                  # p = 0 line
    top = base - 88             # p = 1 line
    curve = " L ".join(f"{X(v):.1f},{base - sig(2 * mu * v) * 88:.1f}" for v in xs)
    body.append(f'<path d="M 24,{base} H 866" class="ax"/><path d="M 24,{top} H 866" class="gd"/>')
    body.append(f'<path d="M 24,{(base + top) / 2} H 866" class="ref"/>')
    for val, yy in (("1", top), ("0.5", (base + top) / 2), ("0", base)):
        body.append(text(20, yy + 4, val, "tiny", "end"))
    body.append(f'<path d="M {curve}" class="sb1"/>')
    for v in pts:
        pv = sig(2 * mu * v)
        body.append(f'<path d="M {X(v):.1f},{ya} V {base - pv * 88:.1f}" class="gd"/>')
        body.append(f'<circle cx="{X(v):.1f}" cy="{base - pv * 88:.1f}" r="5" class="p1"/>')
    for v, pv, dx in ((-3.5, sig(2 * mu * -3.5), 0), (-1.0, sig(2 * mu * -1.0), -8), (-0.25, sig(2 * mu * -0.25), 20)):
        lab = f"x = {v:g}, p = {pv:.2f}".replace("-", "−") + (" (two samples)" if v == -1.0 else "")
        if dx > 0:
            body.append(text(X(v) - 10, base - pv * 88 + 20, lab, "tiny", "end"))
        else:
            body.append(text(X(v) + dx, base - pv * 88 - 10, lab, "tiny", "end"))
    # decision boundary through both panels
    body.append(f'<path d="M {X(0):.1f},48 V 330" class="refd"/>')
    body.append(text(X(0) + 6, 58, "boundary: δ_A = δ_B at x = 0", "sm", "start"))
    body.append(text(X(1.0), 248, "p(A | x) = σ(2μ_A·x) = σ(−2.875·x)", "v b", "start"))
    body.append(text(X(1.0), 266, "steeper when the class means are farther apart", "sm", "start"))
    for v in (-4, -2, 0, 2, 4):
        body.append(text(X(v), 350, str(v), "tiny"))
    body.append(text(866, 362, "feature x (after the Fisher projection)", "tiny", "end"))
    body.append(text(866, 192, "feature x", "tiny", "end"))
    return svg(890, 372, "LDA on the SFDA toy example",
               "Top: two Gaussian class densities of equal width, class A centred at minus 1.44 and its mirror class B "
               "at plus 1.44, with the four class A samples marked on the axis. Bottom: the posterior probability of "
               "class A as a function of x, a sigmoid that equals one half at the decision boundary x equals 0, with "
               "the four samples marked on it.", body)


def fig_lda_derivation():
    """LDA in 1-D, step by step: weighted densities -> linear discriminants -> sigmoid posterior (priors 0.75 / 0.25)."""
    import math
    muA, muB, qA, qB = -1.4375, 1.4375, 0.75, 0.25
    sig = lambda z: 1 / (1 + math.exp(-z))
    g = lambda x, m: math.exp(-(x - m) ** 2 / 2) / math.sqrt(2 * math.pi)
    dA = lambda x: x * muA - muA ** 2 / 2 + math.log(qA)
    dB = lambda x: x * muB - muB ** 2 / 2 + math.log(qB)
    xb = (math.log(qA) - math.log(qB)) / (muB - muA)          # where delta_A = delta_B
    assert abs(dA(xb) - dB(xb)) < 1e-9 and abs(xb - 0.3821) < 1e-3
    pw, gap, x0, top, base = 260, 32, 14, 92, 252
    sx = 28
    xs = [-4 + 8 * i / 160 for i in range(161)]
    heads = ["① prior-weighted densities", "② log of them: straight lines", "③ normalise: the posterior"]
    subs = ["q_c · N(x; μ_c, 1)", "δ_c(x) = x·μ_c − ½μ_c² + log q_c", "p(A | x) = σ(δ_A − δ_B)"]
    body = [text(14, 24, "LDA from Bayes' rule, step by step (toy classes, but with unequal priors q_A = 0.75, q_B = 0.25)",
                 "hd", "start")]
    for k in range(3):
        x = x0 + k * (pw + gap)
        c = x + pw / 2
        X = lambda v: c + v * sx
        body.append(text(c, 50, heads[k], "v b"))
        body.append(text(c, 68, subs[k], "sm"))
        body.append(f'<rect x="{x}" y="76" width="{pw}" height="208" rx="6" class="frame"/>')
        body.append(f'<path d="M {x + 8},{base} H {x + pw - 8}" class="ax"/>')
        for v in (-4, -2, 0, 2, 4):
            body.append(text(X(v), base + 14, str(v).replace("-", "−"), "tiny"))
        body.append(text(x + pw - 8, base + 27, "feature x", "tiny", "end"))
        ylab = ["density × prior", "log score δ_c(x)", "probability p(A | x)"][k]
        body.append(f'<text transform="rotate(-90 {x + 11} 174)" x="{x + 11}" y="174" class="tiny" text-anchor="middle">{ylab}</text>')
    # panel 1: weighted densities
    c1 = x0 + pw / 2
    X1 = lambda v: c1 + v * sx
    peak = qA * g(muA, muA)
    for m, q, fc, sk in ((muA, qA, "blue", "sb1"), (muB, qB, "orange", "sb2")):
        pts = " L ".join(f"{X1(v):.1f},{base - q * g(v, m) / peak * 120:.1f}" for v in xs)
        body.append(f'<path d="M {X1(-4):.1f},{base} L {pts} L {X1(4):.1f},{base} Z" class="{fc}" fill-opacity="0.16" stroke="none"/>')
        body.append(f'<path d="M {pts}" class="{sk}"/>')
    body.append(text(X1(muA), base - 130, "class A, area 0.75", "tiny", "middle"))
    body.append(text(X1(0.8), base - 52, "class B, area 0.25", "tiny", "start"))
    body.append(f'<path d="M {X1(xb):.1f},82 V {base}" class="refd"/>')
    # panel 2: the two discriminants are straight lines
    c2 = x0 + 1 * (pw + gap) + pw / 2
    X2 = lambda v: c2 + v * sx
    lo, hi = -8.2, 4.6
    Y2 = lambda val: base - (val - lo) / (hi - lo) * 150
    for f, sk, lab in ((dA, "sb1", "δ_A"), (dB, "sb2", "δ_B")):
        body.append(f'<path d="M {X2(-4):.1f},{Y2(f(-4)):.1f} L {X2(4):.1f},{Y2(f(4)):.1f}" class="{sk}"/>')
    body.append(text(X2(-3.9) + 4, Y2(dA(-3.9)) - 6, "δ_A", "sm", "start"))
    body.append(text(X2(3.9) - 4, Y2(dB(3.9)) - 6, "δ_B", "sm", "end"))
    body.append(f'<circle cx="{X2(xb):.1f}" cy="{Y2(dA(xb)):.1f}" r="4.5" class="p2 cen"/>')
    body.append(f'<path d="M {X2(xb):.1f},82 V {base}" class="refd"/>')
    body.append(text(X2(xb) + 6, 92, "δ_A = δ_B here", "tiny", "start"))
    # panel 3: the posterior, with the equal-priors curve for comparison
    c3 = x0 + 2 * (pw + gap) + pw / 2
    X3 = lambda v: c3 + v * sx
    Y3 = lambda pv: base - pv * 150
    for f, cls, dash in ((lambda v: sig(2 * muA * v + math.log(qA / qB)), "sb1", ""),
                         (lambda v: sig(2 * muA * v), "ref", "")):
        pts = " L ".join(f"{X3(v):.1f},{Y3(f(v)):.1f}" for v in xs)
        body.append(f'<path d="M {pts}" class="{cls}"/>')
    body.append(f'<path d="M {X3(-4):.1f},{Y3(0.5):.1f} H {X3(4):.1f}" class="gd"/>')
    body.append(f'<path d="M {X3(xb):.1f},82 V {base}" class="refd"/>')
    body.append(f'<circle cx="{X3(xb):.1f}" cy="{Y3(0.5):.1f}" r="4.5" class="p2 cen"/>')
    body.append(text(X3(xb) + 8, Y3(0.5) - 8, "0.5 at x = 0.38", "tiny", "start"))
    body.append(text(X3(-0.5), Y3(0.5) + 34, "grey: equal priors", "tiny", "end"))
    body.append(text(14, 304, "The dropped term −½x² is the same for every class, so it cancels in the normalisation. A larger prior "
                              "gives a class a higher line δ_c, so the boundary", "sm", "start"))
    body.append(text(14, 322, "moves toward the rarer class: from x = 0 (equal priors) to x = 0.38 here, because log(q_A / q_B) = "
                              "log 3 shifts δ_A − δ_B by 1.10.", "sm", "start"))
    return svg(890, 338, "LDA derived from Bayes' rule in one dimension",
               "Three panels over x. First, two prior-weighted Gaussian densities, class A with area 0.75 and class B with "
               "area 0.25. Second, the logs of those densities, which are the straight lines delta A and delta B, crossing at "
               "x 0.38. Third, the posterior probability of class A, a sigmoid equal to one half at x 0.38, shifted right "
               "of the equal-prior curve.", body)


def fig_lda_2d():
    """LDA in 2-D with a shared, correlated covariance: the boundary is a straight line, tilted by Sigma^-1."""
    import math
    S = [[1.0, 0.7], [0.7, 1.0]]
    det = S[0][0] * S[1][1] - S[0][1] ** 2
    Si = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]]
    muA, muB = (-1.5, 0.0), (1.5, 0.0)
    d = (muA[0] - muB[0], muA[1] - muB[1])
    w = (Si[0][0] * d[0] + Si[0][1] * d[1], Si[1][0] * d[0] + Si[1][1] * d[1])
    sc, cx, cy = 52, 300, 168
    X = lambda v: cx + v * sc
    Y = lambda v: cy - v * sc
    body = [text(14, 24, "LDA in two dimensions: with one shared, correlated covariance Σ the boundary is still a straight line, "
                         "tilted by Σ⁻¹", "hd", "start")]
    body.append(f'<rect x="14" y="40" width="560" height="270" rx="6" class="frame"/>')
    body.append(f'<path d="M 24,{cy} H 564" class="ax"/><path d="M {cx},48 V 302" class="ax"/>')
    for v in (-4, -2, 2, 4):
        body.append(text(X(v), cy + 15, str(v).replace("-", "−"), "tiny"))
    for v in (-2, -1, 1, 2):
        body.append(text(cx - 6, Y(v) + 4, str(v).replace("-", "−"), "tiny", "end"))
    body.append(text(562, cy - 6, [("feature x", ""), ("1", "sub")], "sm", "end"))
    body.append(text(cx + 8, 60, [("feature x", ""), ("2", "sub")], "sm", "start"))
    for m, cls, sk in ((muA, "p1", "sb1"), (muB, "p2", "sb2")):
        for kk in (1, 2):
            body.append(f'<ellipse cx="{X(m[0]):.1f}" cy="{Y(m[1]):.1f}" rx="{math.sqrt(1.7) * kk * sc * 0.75:.1f}" '
                        f'ry="{math.sqrt(0.3) * kk * sc * 0.75:.1f}" transform="rotate(-45 {X(m[0]):.1f} {Y(m[1]):.1f})" class="{sk}" '
                        f'fill="none" opacity="{0.9 if kk == 1 else 0.45}"/>')
        body.append(f'<circle cx="{X(m[0]):.1f}" cy="{Y(m[1]):.1f}" r="6" class="{cls} cen"/>')
    # the naive boundary (perpendicular bisector of the means) and the LDA boundary w.x = 0
    body.append(f'<path d="M {X(0):.1f},48 V 302" class="ref"/>')
    t = 2.6
    nrm = math.hypot(*w)
    ux, uy = -w[1] / nrm, w[0] / nrm
    body.append(f'<path d="M {X(-t * ux):.1f},{Y(-t * uy):.1f} L {X(t * ux):.1f},{Y(t * uy):.1f}" class="refd"/>')
    body.append(f'<path d="M {X(0):.1f},{Y(0):.1f} L {X(w[0] / nrm * 1.4):.1f},{Y(w[1] / nrm * 1.4):.1f}" class="e" marker-end="url(#ah)"/>')
    body.append(text(X(muA[0]), Y(muA[1]) - 14, "μ_A", "sm"))
    body.append(text(X(muB[0]), Y(muB[1]) - 14, "μ_B", "sm"))
    body.append(text(X(w[0] / nrm * 1.4) - 6, Y(w[1] / nrm * 1.4) - 8, "w", "sm", "end"))
    body.append(text(26, 78, "ellipses: 1σ and 2σ of the shared covariance", "tiny", "start"))
    body.append(text(X(0) + 6, 296, "perpendicular bisector (what Σ = I would give)", "tiny", "start"))
    tx = 590
    body.append(text(tx, 66, "Boundary:  wᵀx = 0,   w = Σ⁻¹(μ_A − μ_B)", "v b", "start"))
    lines = [
        f"Σ = [[1, 0.7], [0.7, 1]],   μ_A − μ_B = (−3, 0)",
        f"Σ⁻¹(μ_A − μ_B) = ({w[0]:.2f}, {w[1]:.2f})".replace("-", "−"),
        "",
        "The classes are stretched along the diagonal, so",
        "the most reliable separating direction is not the",
        "line joining the means. Σ⁻¹ corrects for the spread:",
        "it turns the boundary from the dotted vertical line",
        "to the dashed tilted one.",
        "",
        "With Σ = I (SFDA after its Fisher projection)",
        "w is parallel to μ_A − μ_B, and the boundary is the",
        "exact perpendicular bisector.",
    ]
    for i, ln in enumerate(lines):
        body.append(text(tx, 92 + i * 19, ln, "sm", "start"))
    return svg(890, 326, "LDA in two dimensions",
               "Two class means at minus 1.5 and plus 1.5 on the horizontal axis, each surrounded by tilted elliptical "
               "contours of the shared covariance. A dotted vertical line is the perpendicular bisector of the means; "
               "the LDA boundary, a dashed line, is tilted because of the inverse covariance, and the arrow w is normal to it.",
               body)


def fig_fisher_criterion():
    """Fisher's criterion: the same two classes projected on three directions; J(u) = (u.d)^2 / (u' S u)."""
    import math
    S = [[1.0, 0.7], [0.7, 1.0]]
    mu = (1.5, 0.0)                       # class B at +mu, class A at -mu
    d = (2 * mu[0], 0.0)
    det = S[0][0] * S[1][1] - S[0][1] ** 2
    Si = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]]
    wv = (Si[0][0] * d[0], Si[1][0] * d[0])
    n = math.hypot(*wv)
    dirs = [("along the line joining the means", (1.0, 0.0)),
            ("along the long axis of the spread", (math.sqrt(0.5), math.sqrt(0.5))),
            ("the Fisher direction, parallel to S⁻¹(μ_B − μ_A)", (wv[0] / n, wv[1] / n))]

    def J(u):
        var = u[0] ** 2 * S[0][0] + 2 * u[0] * u[1] * S[0][1] + u[1] ** 2 * S[1][1]
        return (u[0] * d[0] + u[1] * d[1]) ** 2 / var, var
    Js = [J(u)[0] for _, u in dirs]
    assert abs(Js[2] - (d[0] * Si[0][0] * d[0])) < 1e-6 and Js[2] > Js[0] > Js[1]
    sc, cx, cy = 46, 190, 186
    X = lambda v: cx + v * sc
    Y = lambda v: cy - v * sc
    body = [text(14, 24, "Fisher's criterion: choose the direction u where the projected classes are farthest apart "
                         "relative to their projected spread", "hd", "start")]
    body.append('<rect x="14" y="40" width="352" height="292" rx="6" class="frame"/>')
    for m, cls, sk in ((-mu[0], "p1", "sb1"), (mu[0], "p2", "sb2")):
        for kk in (1, 2):
            body.append(f'<ellipse cx="{X(m):.1f}" cy="{Y(0):.1f}" rx="{math.sqrt(1.7) * kk * sc * 0.7:.1f}" '
                        f'ry="{math.sqrt(0.3) * kk * sc * 0.7:.1f}" transform="rotate(-45 {X(m):.1f} {Y(0):.1f})" '
                        f'class="{sk}" fill="none" opacity="{0.9 if kk == 1 else 0.4}"/>')
        body.append(f'<circle cx="{X(m):.1f}" cy="{Y(0):.1f}" r="5.5" class="{cls} cen"/>')
    body.append(f'<path d="M 24,{cy} H 356" class="ax"/><path d="M {cx},48 V 324" class="ax"/>')
    body.append(text(354, cy + 18, [("feature x", ""), ("1", "sub")], "sm", "end"))
    body.append(text(cx + 8, 60, [("feature x", ""), ("2", "sub")], "sm", "start"))
    marks = ["①", "②", "③"]
    for (nm, u), mk in zip(dirs, marks):
        body.append(f'<path d="M {X(-2.4 * u[0]):.1f},{Y(-2.4 * u[1]):.1f} L {X(2.4 * u[0]):.1f},{Y(2.4 * u[1]):.1f}" '
                    f'class="{"refd" if mk != "③" else "e"}"/>')
        body.append(text(X(2.55 * u[0]), Y(2.55 * u[1]) + 4, mk, "v b"))
    body.append(text(24, 322, "two classes, shared spread; lines: candidate directions u", "tiny", "start"))
    # right: the projected classes on each direction
    rx0, rw = 392, 484
    scp = 40
    maxden = max(1 / math.sqrt(J(u)[1]) for _, u in dirs)
    for k, ((nm, u), mk, Jv) in enumerate(zip(dirs, marks, Js)):
        yy = 52 + k * 94
        body.append(f'<rect x="{rx0}" y="{yy}" width="{rw}" height="84" rx="6" class="frame"/>')
        j, var = J(u)
        body.append(text(rx0 + 10, yy + 16, f"{mk} {nm}", "sm", "start"))
        body.append(text(rx0 + rw - 10, yy + 16, f"J(u) = {Jv:.1f}".replace(".0", ".0"), "v b", "end"))
        cc = rx0 + rw / 2
        tA, tB = -(u[0] * mu[0]), (u[0] * mu[0])
        sd = math.sqrt(var)
        base = yy + 74
        for m, sk, fc in ((tA, "sb1", "blue"), (tB, "sb2", "orange")):
            pts = " L ".join(f"{cc + (m + z) * scp:.1f},{base - 44 * (1 / sd) / maxden * math.exp(-z * z / (2 * var)):.1f}"
                             for z in [-4 + 8 * i / 120 for i in range(121)])
            body.append(f'<path d="M {pts}" class="{sk}"/>')
        body.append(f'<path d="M {rx0 + 10},{base} H {rx0 + rw - 10}" class="ax"/>')
        body.append(text(rx0 + 24, yy + 34, "horizontal axis: t = u·x", "tiny", "start"))
        body.append(f'<text transform="rotate(-90 {rx0 + 11} {yy + 50})" x="{rx0 + 11}" y="{yy + 50}" class="tiny" text-anchor="middle">density</text>')
    body.append(text(14, 350, "J(u) = (u·(μ_B − μ_A))² / (uᵀ S u): squared gap of the projected means over the projected variance. "
                              "The Fisher direction ③ maximises it,", "sm", "start"))
    body.append(text(14, 368, "and the maximum equals (μ_B − μ_A)ᵀ S⁻¹ (μ_B − μ_A) = 17.6, against 9.0 for ① and 2.6 for ②.",
                     "sm", "start"))
    return svg(890, 386, "Fisher's criterion on two projected classes",
               "Left: two classes with a shared correlated spread and three candidate directions. Right: the same classes "
               "projected on each direction as two one-dimensional Gaussians. The Fisher direction separates them best, "
               "with J of 17.6 against 9.0 for the direction along the means and 2.6 for the long axis of the spread.", body)


def fig_pas_idea():
    """PAS in a simplified, 2-D cosine picture: three source centroids on the unit circle, target points on it."""
    import math
    cents = [90, 210, 330]                                    # source class centroids, as angles
    dist = lambda t, c: 1 - math.cos(math.radians(t - c))   # cosine distance between unit vectors
    margins = lambda t: (lambda d: ((d[1] - d[0]) / d[1], d[0], d[1]))(sorted(dist(t, c) for c in cents))
    near = lambda t: min(range(3), key=lambda i: dist(t, cents[i]))
    scen = [("Embedding A", "target points land right next to one source centroid",
             [c + o for c in cents for o in (-12, -5, 4, 11)], 102),
            ("Embedding B", "target points fall between the source centroids",
             [c + o for c in (150, 270, 30) for o in (-14, -6, 5, 13)] + [90 + 22, 210 - 24, 330 + 20], 152)]
    cls = ["p1", "p2", "ah"]
    cx, cy, R = [224, 666], 178, 78
    body = [text(14, 24, "PAS in a nutshell: how decisively does each target point pick one source class centroid?",
                 "hd", "start")]
    results = []
    for (name, sub, angs, hl), c in zip(scen, cx):
        pas = sum(margins(t)[0] for t in angs) / len(angs)
        m_hl, d1, d2 = margins(hl)
        results.append(pas)
        P = lambda t, r=R: (c + r * math.cos(math.radians(t)), cy - r * math.sin(math.radians(t)))
        body.append(text(c, 50, name + ": " + sub, "v b"))
        body.append(f'<rect x="{c - 210}" y="60" width="420" height="232" rx="6" class="frame"/>')
        body.append(f'<circle cx="{c}" cy="{cy}" r="{R}" class="gd"/>')
        # the highlighted point's two nearest centroids
        order = sorted(range(3), key=lambda i: dist(hl, cents[i]))
        hx, hy = P(hl)
        for rank, i in enumerate(order[:2]):
            x2, y2 = P(cents[i])
            body.append(f'<path d="M {hx:.1f},{hy:.1f} L {x2:.1f},{y2:.1f}" class="{"e" if rank == 0 else "refd"}"/>')
        for t in angs:
            x, y = P(t)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" class="{cls[near(t)]}" fill-opacity="0.75"/>')
        for k, t in enumerate(cents):
            x, y = P(t)
            body.append(f'<path d="M {c},{cy} L {x:.1f},{y:.1f}" class="ref"/>')
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="{cls[k]} cen"/>')
            lx, ly = P(t, R + 22)
            body.append(text(lx, ly + 4, [("μ", "it"), (str(k + 1), "sub")], "sm"))
        body.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="6.5" fill="none" class="e"/>')
        body.append(text(c, 276, f"circled point: d₁ = {d1:.2f}, d₂ = {d2:.2f}, margin (d₂ − d₁)/d₂ = {m_hl:.2f}".replace("0.", "0."),
                         "sm"))
        body.append(text(c, 312, f"PAS = average margin = {pas:.2f}", "hd"))
    assert results[0] > results[1] + 0.3
    body.append(text(14, 346, "Unit-length embeddings, so every point sits on the circle and distance is cosine distance 1 − cos θ. Dots take the "
                              "colour of their nearest centroid,", "sm", "start"))
    body.append(text(14, 364, "which is computed, not given: no target labels are used. Solid line: nearest centroid (d₁); dashed: "
                              "second nearest (d₂).", "sm", "start"))
    return svg(890, 380, "PAS: the core idea",
               "Two panels with three source class centroids on a unit circle. In the first, target points sit right next to "
               "one centroid each, so the nearest centroid is much closer than the second nearest and PAS is high. In the "
               "second, target points fall between centroids, the two nearest distances are similar, and PAS is low.", body)


def with_card(src_text):
    """Insert an opaque, theme-aware card behind an existing themed SVG."""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', src_text)
    assert m, "no viewBox"
    w, h = m.groups()
    card = f'<rect x="0" y="0" width="{w}" height="{h}" rx="6" class="bgcard"/>'
    css = ("  .bgcard{fill:#fbfaf6}\n  @media (prefers-color-scheme: dark){.bgcard{fill:#161615}}\n")
    out = src_text.replace("</style>", css + "</style>", 1)
    return re.sub(r"(</style>\s*)", lambda mo: mo.group(1) + card + "\n", out, count=1)


def crop_chaves_fig3(dest):
    """Crop Fig. 3 of Chaves et al. (page 6 of the arXiv PDF) to `dest`. Returns False,
    with a message, if the PDF is not there or the page is not the one expected."""
    pdf = PAPERS / "2023-chaves-medical-transferability" / "paper.pdf"
    if not pdf.exists():
        print("chaves-fig3.png: skipped, paper.pdf is missing (run `make fetch`)")
        return False
    page = subprocess.run(["pdftotext", "-f", "6", "-l", "6", "-layout", str(pdf), "-"],
                          capture_output=True, text=True, check=True).stdout
    if "Fig. 3:" not in page:
        print("chaves-fig3.png: skipped, page 6 does not carry Fig. 3 in this PDF version")
        return False
    # pixels at 300 dpi: the figure's grid, its row labels and the legend, without the caption
    subprocess.run(["pdftoppm", "-r", "300", "-f", "6", "-l", "6", "-x", "540", "-y", "465",
                    "-W", "1480", "-H", "720", "-png", "-singlefile", str(pdf),
                    str(dest.with_suffix(""))], check=True)
    return True


def write_figures():
    FIGS.mkdir(exist_ok=True)
    files = {"pipeline.svg": fig_pipeline(), "ladder.svg": fig_ladder(),
             "nce-joint-counts.svg": fig_nce_counts(),
             "leep-soft-counts.svg": fig_leep_soft_counts(),
             "leep-predictor.svg": fig_leep_predictor(),
             "hscore-whitening.svg": fig_hscore_whitening(),
             "hscore-worked-example.svg": fig_hscore_worked(),
             "logme-worked-example.svg": fig_logme_worked(),
             "logme-evidence-vs-likelihood.svg": fig_logme_evidence_vs_likelihood(),
             "logme-integral.svg": fig_logme_integral(),
             "sfda-worked-example.svg": fig_sfda_worked(),
             "sfda-lda.svg": fig_sfda_lda(),
             "pas-idea.svg": fig_pas_idea(),
             "lda-derivation.svg": fig_lda_derivation(),
             "lda-2d.svg": fig_lda_2d(),
             "fisher-criterion.svg": fig_fisher_criterion(),
             **{f"premise-{k}.svg": fig_premise(k) for k in PREMISES}}
    for name, src in REUSED.items():
        files[name] = with_card((PAPERS / src).read_text(encoding="utf-8"))
    for name, text_ in files.items():
        (FIGS / name).write_text(text_, encoding="utf-8")
    print("figures: wrote", ", ".join(sorted(files)))
    if crop_chaves_fig3(FIGS / "chaves-fig3.png"):
        print("figures: cropped chaves-fig3.png (gitignored)")


if __name__ == "__main__":
    if "--figures" in sys.argv:
        write_figures()
    else:
        print("Transferability talk: nothing to check. Run with --figures to redraw ../figures/.")
