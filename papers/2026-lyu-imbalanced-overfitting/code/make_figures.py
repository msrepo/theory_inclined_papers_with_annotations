#!/usr/bin/env python3
"""Writes the three SVG figures in ../figures from asymptotics.py and
svm_simulation.py. Seeded, so re-running reproduces the same files.

  eld-tld.svg      training vs test logits per class (Figure 1 of the paper,
                   theory curves plus a simulated SVM histogram)
  errors-vs-pi.svg limiting test errors against pi, tau = 1 and tau = tau_opt
                   (Figure 4 of the paper, theory only), with the tau_opt < 0 band
  calibration.svg  the limiting reliability curve at tau_opt, and the same curve
                   once the prior log-odds are added back

Standard library and numpy only.

Run:  python3 make_figures.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from asymptotics import Phi, bisect, solve, tau_opt
from svm_simulation import fit, sample

OUT = Path(__file__).resolve().parent.parent / "figures"

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:11px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sub{font-size:11.5px;fill:#57564f} .tiny{font-size:10px;fill:#57564f} .ink{fill:#1a1a19}
  .ax{stroke:#c3c2b7;stroke-width:0.8;fill:none} .gd{stroke:#e2e1da;stroke-width:0.6;fill:none}
  .mn{stroke:#2f6fb5;stroke-width:2;fill:none} .mnf{fill:#2f6fb5} .mnb{fill:#2f6fb5;opacity:.30}
  .mj{stroke:#c2571a;stroke-width:2;fill:none} .mjf{fill:#c2571a} .mjb{fill:#c2571a;opacity:.30}
  .dash{stroke-dasharray:5 3} .shade{fill:#8a8880;opacity:.16} .shade2{fill:#b42318;opacity:.16}
  .k{stroke:#1a1a19;stroke-width:1.1;fill:none;stroke-dasharray:3 3} .k0{stroke:#1a1a19;stroke-width:1.3;fill:none}
  .band{fill:#8a8880;opacity:.14}
  .s0{stroke:#9cc3ea;stroke-width:2;fill:none} .s1{stroke:#5b93d0;stroke-width:2;fill:none}
  .s2{stroke:#2f6fb5;stroke-width:2;fill:none} .s3{stroke:#184f95;stroke-width:2;fill:none}
  @media (prefers-color-scheme: dark){
    .lab,.sub,.tiny{fill:#b6b4ab} .hd,.ink{fill:#eceae3} .ax{stroke:#4a4844} .gd{stroke:#33312e}
    .mn{stroke:#3d8ad8} .mnf{fill:#3d8ad8} .mnb{fill:#3d8ad8;opacity:.38}
    .mj{stroke:#d8733a} .mjf{fill:#d8733a} .mjb{fill:#d8733a;opacity:.38}
    .shade{fill:#b6b4ab;opacity:.16} .shade2{fill:#ff7a6b;opacity:.18}
    .k,.k0{stroke:#eceae3} .band{fill:#b6b4ab;opacity:.14}
    .s0{stroke:#2a5e9e} .s1{stroke:#3d7fc6} .s2{stroke:#6aa6e6} .s3{stroke:#a9cdf3}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n{body}\n</svg>\n")


class Axes:
    def __init__(self, x0, y0, w, h, xlim, ylim):
        self.x0, self.y0, self.w, self.h, self.xlim, self.ylim = x0, y0, w, h, xlim, ylim

    def X(self, x):
        return self.x0 + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def Y(self, y):
        return self.y0 + self.h - (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def path(self, xs, ys, cls):
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))
        return f'<polyline class="{cls}" points="{pts}"/>'

    def frame(self, xticks, yticks, xlab, ylab, xfmt="{:g}", yfmt="{:g}"):
        out = []
        for t in yticks:
            out.append(f'<line class="gd" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.Y(t):.1f}" y2="{self.Y(t):.1f}"/>')
            out.append(f'<text class="tiny" x="{self.x0 - 5}" y="{self.Y(t) + 3:.1f}" text-anchor="end">{yfmt.format(t)}</text>')
        for t in xticks:
            out.append(f'<text class="tiny" x="{self.X(t):.1f}" y="{self.y0 + self.h + 13}" text-anchor="middle">{xfmt.format(t)}</text>')
        out.append(f'<line class="ax" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.y0 + self.h}" y2="{self.y0 + self.h}"/>')
        out.append(f'<text class="lab" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 28}" text-anchor="middle">{xlab}</text>')
        if ylab:
            out.append(f'<text class="lab" x="{self.x0}" y="{self.y0 - 7}">{ylab}</text>')
        return "\n".join(out)


def gauss(x, mu):
    return np.exp(-0.5 * (x - mu) ** 2) / math.sqrt(2 * math.pi)


# ----------------------------------------------------------------------------
def fig_eld_tld():
    m, pi, n, d = 1.75, 0.15, 3000, 1200
    th = solve(m, pi, n / d)
    X, y = sample(n, d, m, pi, np.random.default_rng(11))
    beta, b0, kappa, _, proj = fit(X, y)
    marg = y * (proj + b0)
    body = [f'<text class="hd" x="20" y="22">Training logits are the test logits, truncated at the margin</text>',
            f'<text class="sub" x="20" y="38">Hard-margin SVM, n = {n}, d = {d}, ‖µ‖ = 1.75, π = 0.15. Bars: one run’s training margins. Curves: Theorem 2.1.</text>']
    xs = np.linspace(-3.0, 5.0, 400)
    k = th['kappa']
    for col, (cls, lab, c) in enumerate(((1.0, "minority (y = +1)", "mn"), (-1.0, "majority (y = −1)", "mj"))):
        ax = Axes(52 + col * 345, 78, 290, 190, (-3, 5), (0, 0.72))
        c_mu = th['rho'] * m + cls * th['b0']
        tld = gauss(xs, c_mu)
        atom = Phi(k - c_mu)
        err = Phi(-c_mu)
        body.append(ax.frame(range(-3, 6), (0, 0.2, 0.4, 0.6), "y · f(x)   (distance past the boundary, signed)",
                             "density" if col == 0 else ""))
        # shaded TLD mass below kappa and below 0
        below = xs <= k
        pts = [(ax.X(xs[0]), ax.Y(0))] + [(ax.X(x), ax.Y(t)) for x, t in zip(xs[below], tld[below])] + [(ax.X(k), ax.Y(0))]
        body.append('<polygon class="shade" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
        neg = xs <= 0
        pts = [(ax.X(xs[0]), ax.Y(0))] + [(ax.X(x), ax.Y(t)) for x, t in zip(xs[neg], tld[neg])] + [(ax.X(0), ax.Y(0))]
        body.append('<polygon class="shade2" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
        # histogram of simulated training margins (continuous part), bin width 0.2
        vals = marg[y == cls]
        edges = np.arange(-3, 5.0001, 0.2)
        on = vals <= kappa + 1e-6
        h, _ = np.histogram(vals[~on], bins=edges)
        h = h / (len(vals) * 0.2)
        for lo, hv in zip(edges[:-1], h):
            if hv > 0:
                body.append(f'<rect class="{c}b" x="{ax.X(lo) + 1:.1f}" y="{ax.Y(hv):.1f}" width="{ax.X(lo + 0.2) - ax.X(lo) - 2:.1f}" height="{ax.Y(0) - ax.Y(hv):.1f}" rx="1"/>')
        # theory: continuous part of the ELD (x > kappa) and the atom
        keep = xs > k
        body.append(ax.path(xs[keep], tld[keep], c))
        body.append(ax.path(xs, tld, f"{c} dash"))
        # atom drawn as a spike whose label gives its mass
        sim_atom = on.mean()
        body.append(f'<line class="{c}" x1="{ax.X(k):.1f}" x2="{ax.X(k):.1f}" y1="{ax.Y(0):.1f}" y2="{ax.Y(0.66):.1f}" stroke-width="3"/>')
        body.append(f'<text class="lab" x="{ax.X(k) + 5:.1f}" y="{ax.Y(0.66) + 4:.1f}">atom at κ* = {k:.2f}</text>')
        body.append(f'<text class="tiny" x="{ax.X(k) + 5:.1f}" y="{ax.Y(0.66) + 17:.1f}">{atom:.0%} of the class (run: {sim_atom:.0%})</text>')
        body.append(f'<line class="k0" x1="{ax.X(0):.1f}" x2="{ax.X(0):.1f}" y1="{ax.Y(0):.1f}" y2="{ax.Y(0.5):.1f}"/>')
        body.append(f'<text class="tiny" x="{ax.X(0) - 4:.1f}" y="{ax.Y(0.5) - 4:.1f}" text-anchor="end">boundary</text>')
        body.append(f'<text class="hd" x="{ax.x0}" y="{ax.y0 - 22}">{lab}</text>')
        body.append(f'<text class="tiny" x="{ax.X(-2.9):.1f}" y="{ax.Y(0.45):.1f}">test error {err:.1%}:</text>')
        body.append(f'<text class="tiny" x="{ax.X(-2.9):.1f}" y="{ax.Y(0.45) + 12:.1f}">red area, erased</text>')
        body.append(f'<text class="tiny" x="{ax.X(-2.9):.1f}" y="{ax.Y(0.45) + 24:.1f}">in training</text>')
    body.append('<text class="lab" x="20" y="330">Dashed: the test-logit density N(ρ*‖µ‖ + yβ₀*, 1). Solid: its part above κ*, kept unchanged in training.</text>')
    body.append('<text class="lab" x="20" y="345">Shaded: test mass below κ*, moved onto the margin. β₀* = '
                f'{th["b0"]:.2f} puts the minority centre at {th["rho"] * m + th["b0"]:.2f}, the majority’s at {th["rho"] * m - th["b0"]:.2f}.</text>')
    desc = ("Two panels, minority and majority. In each, a dashed Gaussian is the test logit density; to the right of the margin kappa* = "
            f"{k:.2f} the solid curve and the simulated histogram agree with it. Everything the Gaussian puts below kappa is moved to a spike at "
            f"kappa: {Phi(k - th['rho'] * m - th['b0']):.0%} of the minority and {Phi(k - th['rho'] * m + th['b0']):.0%} of the majority. "
            f"The mass below zero, the test error, is {th['err_p']:.1%} for the minority and {th['err_m']:.1%} for the majority.")
    (OUT / "eld-tld.svg").write_text(svg(700, 360, "Empirical and testing logit distributions", desc, "\n".join(body)))
    return th, kappa, b0, beta[0]


def fig_errors_vs_pi():
    m, delta = 1.5, 0.5
    pis = np.concatenate([np.linspace(0.006, 0.05, 30), np.linspace(0.05, 0.5, 60)])
    ep, em, eb, eo = [], [], [], []
    for p in pis:
        s = solve(m, p, delta)
        ep.append(s['err_p']); em.append(s['err_m']); eb.append(s['err_b']); eo.append(Phi(-s['rho'] * m))
    pc = bisect(lambda p: (lambda z: z['b0'] + z['kappa'])(solve(m, p, delta)), 0.005, 0.3)
    ax = Axes(60, 60, 600, 230, (0, 0.5), (0, 1.0))
    body = ['<text class="hd" x="20" y="22">Imbalance hurts the minority, and rebalancing the margin repairs most of it</text>',
            f'<text class="sub" x="20" y="38">Limits from Theorem 2.1 and Eq 38, ‖µ‖ = {m}, n/d = {delta} (the setting of the paper’s Figure 4).</text>',
            f'<rect class="band" x="{ax.X(0):.1f}" y="{ax.y0}" width="{ax.X(pc) - ax.X(0):.1f}" height="{ax.h}"/>',
            ax.frame((0, 0.1, 0.2, 0.3, 0.4, 0.5), (0, 0.25, 0.5, 0.75, 1.0), "π  (minority fraction)", "test error")]
    body.append(ax.path(pis, ep, "mn"))
    body.append(ax.path(pis, em, "mj"))
    body.append(ax.path(pis, eb, "k"))
    body.append(ax.path(pis, eo, "k0"))
    body.append(f'<text class="lab" x="{ax.X(0.13):.1f}" y="{ax.Y(0.75):.1f}">minority error, τ = 1</text>')
    body.append(f'<text class="lab" x="{ax.X(0.36):.1f}" y="{ax.Y(0.08):.1f}">majority error, τ = 1</text>')
    body.append(f'<text class="lab" x="{ax.X(0.052):.1f}" y="{ax.Y(0.53):.1f}">balanced error, τ = 1</text>')
    body.append(f'<text class="lab" x="{ax.X(0.17):.1f}" y="{ax.Y(0.13):.1f}">all three errors, τ = τopt: Φ(−ρ*‖µ‖)</text>')
    body.append(f'<text class="tiny" x="{ax.X(pc) + 4:.1f}" y="{ax.y0 + 12}">← π &lt; {pc:.3f}:</text>')
    body.append(f'<text class="tiny" x="{ax.X(pc) + 4:.1f}" y="{ax.y0 + 24}">τopt &lt; 0</text>')
    legend = [("mn", "minority, τ = 1"), ("mj", "majority, τ = 1"), ("k", "balanced, τ = 1"), ("k0", "any class, τ = τopt")]
    for i, (c, t) in enumerate(legend):
        x = 60 + i * 150
        body.append(f'<line class="{c}" x1="{x}" x2="{x + 22}" y1="325" y2="325"/><text class="lab" x="{x + 28}" y="329">{t}</text>')
    desc = (f"Line chart of test error against pi from 0 to 0.5. At tau = 1 the minority error rises from {ep[-1]:.2f} at pi = 0.5 to 1 as pi goes to 0, "
            f"the majority error falls to 0, and the balanced error rises to 0.5. With tau = tau_opt all three errors equal Phi(-rho* ||mu||), which rises "
            f"only from {eo[-1]:.2f} to {eo[0]:.2f}. A grey band marks pi below {pc:.3f}, where tau_opt is negative.")
    (OUT / "errors-vs-pi.svg").write_text(svg(700, 345, "Test errors against imbalance", desc, "\n".join(body)))
    return pc


def fig_calibration():
    m, delta = 1.0, 2.0
    pis = (0.5, 0.25, 0.1, 0.05)
    cls = ("s0", "s1", "s2", "s3")
    body = ['<text class="hd" x="20" y="22">The limiting reliability diagram at τ = τopt is one explicit curve per π</text>',
            f'<text class="sub" x="20" y="38">‖µ‖ = 1, n/d = 2 (the paper’s Figure 6). True P(y = +1 | p̂) = σ(2ρ*‖µ‖·logit p̂ + log π/(1−π)).</text>']
    q = np.linspace(0.005, 0.995, 200)
    lg = np.log(q / (1 - q))
    for col, (title, add_prior) in enumerate((("confidence p̂ = σ(f)", False),
                                               ("confidence σ(f + log π/(1−π))", True))):
        ax = Axes(55 + col * 340, 72, 270, 220, (0, 1), (0, 1))
        body.append(ax.frame((0, 0.25, 0.5, 0.75, 1), (0, 0.25, 0.5, 0.75, 1), title,
                             "P(y = +1 | confidence)" if col == 0 else ""))
        body.append(ax.path([0, 1], [0, 1], "k"))
        for p, c in zip(pis, cls):
            rho = solve(m, p, delta)['rho']
            prior = math.log(p / (1 - p))
            f = lg - prior if add_prior else lg          # the logit that produces confidence q
            true = 1 / (1 + np.exp(-(2 * rho * m * f + prior)))
            body.append(ax.path(q, true, c))
            if not add_prior:
                y1 = float(1 / (1 + np.exp(-(2 * rho * m * math.log(0.7 / 0.3) + prior))))
                body.append(f'<text class="tiny" x="{ax.X(0.7) + 6:.1f}" y="{ax.Y(y1) + 12:.1f}">π = {p}</text>')
    for i, (p, c) in enumerate(zip(pis, cls)):
        x = 55 + i * 120
        body.append(f'<line class="{c}" x1="{x}" x2="{x + 22}" y1="335" y2="335"/><text class="lab" x="{x + 28}" y="339">π = {p}</text>')
    body.append('<line class="k" x1="535" x2="557" y1="335" y2="335"/><text class="lab" x="563" y="339">perfect</text>')
    desc = ("Two reliability panels. Left: the confidence sigma(f) of the rebalanced max-margin classifier against the true probability; as pi falls "
            "from 0.5 to 0.05 the curve drops further below the diagonal, i.e. the minority probability is inflated. Right: the same classifier "
            "with the prior log-odds log(pi/(1-pi)) added to the logit; all four curves lie almost on the diagonal.")
    (OUT / "calibration.svg").write_text(svg(700, 350, "Calibration at the optimal margin ratio", desc, "\n".join(body)))


def main():
    OUT.mkdir(exist_ok=True)
    th, kappa, b0, rho = fig_eld_tld()
    print(f"eld-tld.svg: simulated kappa {kappa:.3f} b0 {b0:+.3f} rho {rho:.3f};"
          f" theory {th['kappa']:.3f} {th['b0']:+.3f} {th['rho']:.3f}")
    pc = fig_errors_vs_pi()
    print(f"errors-vs-pi.svg: tau_opt < 0 below pi = {pc:.4f}")
    fig_calibration()
    print("calibration.svg written")


if __name__ == "__main__":
    main()
