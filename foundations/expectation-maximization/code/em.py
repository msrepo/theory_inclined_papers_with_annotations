#!/usr/bin/env python3
"""EM for a Gaussian mixture, with the claims in notes.md checked numerically.

Checked here:

  * the worked six-point example in notes.md §5 (responsibilities and the
    parameters after one M-step), printed so the table can be compared;
  * log p(X|theta) = L(q, theta) + KL(q || p(z|x, theta)) for an arbitrary q,
    and that the KL gap is exactly zero once q is the posterior (E-step);
  * the log-likelihood never decreases over 200 EM iterations;
  * the k-means limit: with shared variance s^2 -> 0, responsibilities
    become 0/1 nearest-centre assignments;
  * the singularity: parking one component on a data point and shrinking its
    variance sends the likelihood to +infinity.

With --figures it also regenerates the SVGs in ../figures/ from these same
computations, so every curve in the notes is a real one.

Standard library and numpy only.

Run:  python3 em.py            (checks)
      python3 em.py --figures  (checks, then rewrite ../figures/*.svg)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LOG2PI = np.log(2 * np.pi)


def log_normal(x, mu, var):
    """log N(x | mu, var) for 1-D x (shape n) against K components -> (n, K)."""
    x = np.asarray(x, float)[:, None]
    return -0.5 * (LOG2PI + np.log(var) + (x - mu) ** 2 / var)


def log_joint(x, pi, mu, var):
    """log [pi_k N(x_i | mu_k, var_k)], shape (n, K)."""
    return np.log(pi) + log_normal(x, mu, var)


def logsumexp(a, axis=1):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def loglik(x, pi, mu, var):
    return logsumexp(log_joint(x, pi, mu, var)).sum()


def e_step(x, pi, mu, var):
    lj = log_joint(x, pi, mu, var)
    return np.exp(lj - logsumexp(lj)[:, None])


def m_step(x, gamma):
    nk = gamma.sum(0)
    mu = (gamma * x[:, None]).sum(0) / nk
    var = (gamma * (x[:, None] - mu) ** 2).sum(0) / nk
    return nk / len(x), mu, var


def lower_bound(x, q, pi, mu, var):
    """L(q, theta) = sum_i sum_k q_ik [log p(x_i, z=k | theta) - log q_ik]."""
    lj = log_joint(x, pi, mu, var)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.where(q > 0, q * np.log(q), 0.0)
    return (q * lj).sum() - ent.sum()


def kl_to_posterior(x, q, pi, mu, var):
    post = e_step(x, pi, mu, var)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(q > 0, q * (np.log(q) - np.log(post)), 0.0)
    return t.sum()


# ---------------------------------------------------------------- data sets

WORKED_X = np.array([1.0, 2.0, 3.0, 6.0, 7.0, 8.0])
WORKED_INIT = (np.array([0.5, 0.5]), np.array([2.0, 4.0]), np.array([1.0, 1.0]))


def blob_data(seed=3):
    rng = np.random.default_rng(seed)
    a = rng.normal(-2.0, 0.8, 45)
    b = rng.normal(1.8, 1.1, 75)
    return np.sort(np.concatenate([a, b]))


# ------------------------------------------------------------------- checks

def check_worked():
    pi, mu, var = WORKED_INIT
    g = e_step(WORKED_X, pi, mu, var)
    print("worked example: responsibilities gamma_i1 (for component 1, mu=2)")
    for xi, gi in zip(WORKED_X, g[:, 0]):
        print(f"  x={xi:>4.1f}  gamma1={gi:.3f}  gamma2={1 - gi:.3f}")
    pi2, mu2, var2 = m_step(WORKED_X, g)
    print(f"  N_k   = {g.sum(0).round(3)}")
    print(f"  pi    = {pi2.round(3)}")
    print(f"  mu    = {mu2.round(3)}")
    print(f"  var   = {var2.round(3)}")
    print(f"  loglik before {loglik(WORKED_X, pi, mu, var):.3f}"
          f"  after {loglik(WORKED_X, pi2, mu2, var2):.3f}")
    p, m, v = pi, mu, var
    for _ in range(50):
        p, m, v = m_step(WORKED_X, e_step(WORKED_X, p, m, v))
    print(f"  after 50 iterations: pi={p.round(3)} mu={m.round(3)} var={v.round(3)}")


def check_decomposition():
    x = blob_data()
    rng = np.random.default_rng(0)
    pi, mu, var = np.array([0.3, 0.7]), np.array([-1.0, 0.5]), np.array([2.0, 1.5])
    q = rng.dirichlet([1, 1], size=len(x))
    ll = loglik(x, pi, mu, var)
    L = lower_bound(x, q, pi, mu, var)
    kl = kl_to_posterior(x, q, pi, mu, var)
    assert abs(ll - (L + kl)) < 1e-8, (ll, L, kl)
    assert kl > 0
    post = e_step(x, pi, mu, var)
    assert abs(kl_to_posterior(x, post, pi, mu, var)) < 1e-10
    assert abs(lower_bound(x, post, pi, mu, var) - ll) < 1e-8
    print(f"decomposition: log p = {ll:.4f} = L {L:.4f} + KL {kl:.4f}; KL after E-step = 0  ok")


def check_monotone():
    x = blob_data()
    pi, mu, var = np.array([0.5, 0.5]), np.array([3.0, 3.5]), np.array([1.0, 1.0])
    lls = [loglik(x, pi, mu, var)]
    for _ in range(200):
        pi, mu, var = m_step(x, e_step(x, pi, mu, var))
        lls.append(loglik(x, pi, mu, var))
    d = np.diff(lls)
    assert (d > -1e-9).all(), d.min()
    print(f"monotone: log-likelihood {lls[0]:.2f} -> {lls[-1]:.2f}, never decreased over 200 steps  ok")


def check_kmeans_limit():
    x = blob_data()
    mu = np.array([-1.0, 1.0])
    nearest = np.abs(x[:, None] - mu).argmin(1)
    for s2 in (1.0, 0.1, 1e-3):
        g = e_step(x, np.array([0.5, 0.5]), mu, np.array([s2, s2]))
        hard = (g.argmax(1) == nearest).all()
        crisp = np.mean(np.max(g, 1) > 0.999)
        print(f"k-means limit: var={s2:<6} argmax=nearest centre: {hard}; "
              f"fraction of points with gamma>0.999: {crisp:.2f}")
    assert crisp > 0.98  # only points almost exactly midway between centres stay unsure


def check_singularity():
    x = blob_data()
    # Only the one point under the spike gains, by 1/2 log(1/var): slow, but unbounded.
    lls = []
    for v in (1e-4, 1e-16, 1e-64, 1e-256):
        lls.append(loglik(x, np.array([0.5, 0.5]), np.array([x[0], 0.0]), np.array([v, 4.0])))
        print(f"singularity: component 1 sitting on x_1 with var={v:<7} -> log-lik {lls[-1]:9.2f}")
    assert np.all(np.diff(lls) > 0)


# ------------------------------------------------------------------ figures

STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:12px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .sm{font-size:11px;fill:#57564f}
  .ax{stroke:#c3c2b7;stroke-width:0.75;fill:none}
  .gd{stroke:#e2e1da;stroke-width:0.5;fill:none;stroke-dasharray:4 4}
  .a{fill:none;stroke:#2f6fb5;stroke-width:1.6} .af{fill:#2f6fb5}
  .b{fill:none;stroke:#c2571a;stroke-width:1.6} .bf{fill:#c2571a}
  .mix{fill:none;stroke:#1a1a19;stroke-width:2}
  .ll{fill:none;stroke:#b42318;stroke-width:2.2}
  .bd{fill:none;stroke-width:1.6}
  .tick{stroke:#8a8880;stroke-width:1}
  .drop{stroke:#8a8880;stroke-width:0.8;stroke-dasharray:2 3}
  .dot{fill:#1a1a19}
  .bar{stroke:none}
  .kl{fill:#b42318;opacity:.18} .lb{fill:#2f6fb5;opacity:.22}
  .edge{stroke:#1a1a19;stroke-width:1.4;fill:none}
  .arr{stroke:#0f7a5a;stroke-width:1.6;fill:none}
  .arrf{fill:#0f7a5a}
  @media (prefers-color-scheme: dark){
    .lab,.sm{fill:#b6b4ab} .hd{fill:#eceae3} .ax{stroke:#4a4844} .gd{stroke:#33312e}
    .a{stroke:#7fb2e8} .af{fill:#7fb2e8} .b{stroke:#f0a070} .bf{fill:#f0a070}
    .mix{stroke:#eceae3} .ll{stroke:#ff7a6b} .tick{stroke:#85837b} .drop{stroke:#85837b}
    .dot{fill:#eceae3} .kl{fill:#ff7a6b;opacity:.22} .lb{fill:#7fb2e8;opacity:.25}
    .edge{stroke:#eceae3} .arr{stroke:#4cc79a} .arrf{fill:#4cc79a}
  }
</style>"""


def svg(w, h, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}\n{body}\n</svg>\n")


def path(xs, ys, cls, extra=""):
    pts = " L ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    return f'<path d="M {pts}" class="{cls}"{extra}/>'


class Frame:
    """Maps data coordinates into a pixel box."""

    def __init__(self, x0, y0, w, h, xlim, ylim):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim = xlim, ylim

    def X(self, v):
        a, b = self.xlim
        return self.x0 + (np.asarray(v) - a) / (b - a) * self.w

    def Y(self, v):
        a, b = self.ylim
        return self.y0 + self.h - (np.asarray(v) - a) / (b - a) * self.h

    def axis(self):
        y = self.y0 + self.h
        return f'<path d="M {self.x0},{y} H {self.x0 + self.w}" class="ax"/>'


def mix_colour(g):
    """Blend blue (component 1) and orange (component 2) by responsibility."""
    a, b = np.array([0x2f, 0x6f, 0xb5]), np.array([0xc2, 0x57, 0x1a])
    c = (g * a + (1 - g) * b).round().astype(int)
    return "#%02x%02x%02x" % tuple(c)


def fig_mixture(out):
    x = blob_data()
    pi, mu, var = m_step(x, e_step(x, *em_run(x, 300)))
    grid = np.linspace(-5, 6, 400)
    comp = np.exp(log_joint(grid, pi, mu, var))
    F = Frame(40, 30, 620, 170, (-5, 6), (0, comp.sum(1).max() * 1.1))
    body = [F.axis()]
    body.append(path(F.X(grid), F.Y(comp[:, 0]), "a", ' stroke-dasharray="5 3"'))
    body.append(path(F.X(grid), F.Y(comp[:, 1]), "b", ' stroke-dasharray="5 3"'))
    body.append(path(F.X(grid), F.Y(comp.sum(1)), "mix"))
    g = e_step(x, pi, mu, var)[:, 0]
    for xi, gi in zip(x, g):
        body.append(f'<line x1="{F.X(xi):.1f}" x2="{F.X(xi):.1f}" y1="{F.y0 + F.h + 4}" '
                    f'y2="{F.y0 + F.h + 16}" stroke="{mix_colour(gi)}" stroke-width="1.3"/>')
    for t in range(-4, 7, 2):
        body.append(f'<text x="{F.X(t):.1f}" y="{F.y0 + F.h + 32}" class="sm" text-anchor="middle">{t}</text>')
    k = np.argmax(comp[:, 0])
    body.append(f'<text x="{F.X(grid[k]) - 8:.1f}" y="{F.Y(comp[k, 0]) - 8:.1f}" class="lab" '
                f'text-anchor="end">π₁·N(x | μ₁, σ₁²)</text>')
    k = np.argmax(comp[:, 1])
    body.append(f'<text x="{F.X(grid[k]) + 40:.1f}" y="{F.Y(comp[k, 1]) + 26:.1f}" class="lab">π₂·N(x | μ₂, σ₂²)</text>')
    k = np.argmin(np.abs(grid - 3.2))
    body.append(f'<text x="{F.X(3.3):.1f}" y="{F.Y(comp.sum(1)[k]) - 10:.1f}" class="hd">p(x) = sum of the two</text>')
    body.append(f'<text x="{F.x0}" y="{F.y0 + F.h + 50}" class="sm">tick colour = responsibility: '
                f'blue = surely component 1, orange = surely component 2, in between = unsure</text>')
    desc = ("A one-dimensional two-component Gaussian mixture fitted by EM to 120 points. Two dashed "
            "bell curves, each scaled by its mixing weight, add up to a solid bimodal density. Below the "
            "axis each data point is a tick coloured by its responsibility: blue on the left, orange on "
            "the right, with purple-ish ticks in the overlap region between the humps.")
    out.write_text(svg(700, 290, "A two-component Gaussian mixture and its responsibilities", desc,
                       "\n".join(body)))


def em_run(x, n, init=(np.array([0.5, 0.5]), np.array([3.0, 3.5]), np.array([1.0, 1.0]))):
    pi, mu, var = init
    for _ in range(n):
        pi, mu, var = m_step(x, e_step(x, pi, mu, var))
    return pi, mu, var


def fig_iterations(out):
    x = blob_data()
    init = (np.array([0.5, 0.5]), np.array([-0.5, 0.5]), np.array([4.0, 4.0]))
    snaps = {}
    pi, mu, var = init
    for it in range(41):
        if it in (0, 10, 15, 40):
            snaps[it] = (pi, mu, var, loglik(x, pi, mu, var))
        pi, mu, var = m_step(x, e_step(x, pi, mu, var))
    grid = np.linspace(-5, 6, 300)
    body = []
    W, H = 330, 120
    ymax = 0.32
    for n, (it, (p, m, v, ll)) in enumerate(snaps.items()):
        cx, cy = 20 + (n % 2) * 345, 30 + (n // 2) * 185
        F = Frame(cx, cy, W, H, (-5, 6), (0, ymax))
        comp = np.exp(log_joint(grid, p, m, v))
        body.append(F.axis())
        body.append(path(F.X(grid), F.Y(np.minimum(comp[:, 0], ymax)), "a"))
        body.append(path(F.X(grid), F.Y(np.minimum(comp[:, 1], ymax)), "b"))
        g = e_step(x, p, m, v)[:, 0]
        for xi, gi in zip(x, g):
            body.append(f'<line x1="{F.X(xi):.1f}" x2="{F.X(xi):.1f}" y1="{cy + H + 3}" '
                        f'y2="{cy + H + 14}" stroke="{mix_colour(gi)}" stroke-width="1.2"/>')
        name = "start (before any step)" if it == 0 else f"after {it} iteration{'s' if it > 1 else ''}"
        body.append(f'<text x="{cx}" y="{cy - 10}" class="hd">{name}</text>')
        body.append(f'<text x="{cx + W}" y="{cy - 10}" class="sm" text-anchor="end">log-lik {ll:.1f}</text>')
    desc = ("Four snapshots of EM on 120 one-dimensional points. At the start both components are wide "
            "and almost on top of each other, and every tick is a muddy purple (unsure). After ten "
            "iterations they have barely moved and the log-likelihood has crept from -263 to -260: EM is "
            "crawling off a near-symmetric plateau. By fifteen they have split, and after forty the left "
            "bump is blue, the right bump orange, and the log-likelihood has settled at -243.")
    out.write_text(svg(700, 390, "EM iterations on one-dimensional data", desc, "\n".join(body)))


def fig_lower_bound(out):
    """ll(mu_1) with the EM lower bounds that touch it, for an EM run on mu_1 alone."""
    x = blob_data()
    pi, var = np.array([0.5, 0.5]), np.array([1.0, 1.0])
    mu2 = 1.8
    xlim = (-7.0, 0.5)
    grid = np.linspace(*xlim, 400)
    ll = np.array([loglik(x, pi, np.array([m, mu2]), var) for m in grid])
    m1 = -6.2
    steps = [m1]
    bounds = []
    for _ in range(3):
        q = e_step(x, pi, np.array([m1, mu2]), var)
        bounds.append(np.array([lower_bound(x, q, pi, np.array([m, mu2]), var) for m in grid]))
        m1 = (q[:, 0] * x).sum() / q[:, 0].sum()  # M-step for mu_1 only
        steps.append(m1)
    lo = loglik(x, pi, np.array([steps[0], mu2]), var) - 20
    F = Frame(60, 20, 600, 250, xlim, (lo, ll.max() + 12))
    body = [F.axis(), f'<path d="M {F.x0},{F.y0} V {F.y0 + F.h}" class="ax"/>']
    body.append('<clipPath id="c"><rect x="{}" y="{}" width="{}" height="{}"/></clipPath>'
                .format(F.x0, F.y0, F.w, F.h))
    shades = ["#6c4bc7", "#2f6fb5", "#0f7a5a"]
    for b, col in zip(bounds, shades):
        body.append(path(F.X(grid), F.Y(np.maximum(b, lo - 50)), "bd",
                         f' stroke="{col}" clip-path="url(#c)"'))
    body.append(path(F.X(grid), F.Y(ll), "ll", ' clip-path="url(#c)"'))
    anchors = ["middle", "middle", "end", "start"]
    for t, s in enumerate(steps):
        y = loglik(x, pi, np.array([s, mu2]), var)
        body.append(f'<line x1="{F.X(s):.1f}" x2="{F.X(s):.1f}" y1="{F.Y(y):.1f}" y2="{F.y0 + F.h}" class="drop"/>')
        body.append(f'<circle cx="{F.X(s):.1f}" cy="{F.Y(y):.1f}" r="3.5" class="dot"/>')
        dx = {"end": -3, "start": 3}.get(anchors[t], 0)
        body.append(f'<text x="{F.X(s) + dx:.1f}" y="{F.y0 + F.h + 16}" class="lab" '
                    f'text-anchor="{anchors[t]}">θ{"⁰¹²³"[t]}</text>')
    j = np.argmin(np.abs(grid + 0.4))
    body.append(f'<text x="{F.X(grid[j]):.1f}" y="{F.Y(ll[j]) - 12:.1f}" class="hd">log p(X | θ)</text>')
    # legend, top left, clear of the curves
    lx, ly = F.x0 + 16, F.y0 + 14
    names = ["ℒ(q⁰, θ): bound built at θ⁰", "ℒ(q¹, θ): bound built at θ¹", "ℒ(q², θ): bound built at θ²"]
    for i, (col, name) in enumerate(zip(shades, names)):
        body.append(f'<path d="M {lx},{ly + i * 18} h 22" class="bd" stroke="{col}"/>')
        body.append(f'<text x="{lx + 28}" y="{ly + i * 18 + 4}" class="lab">{name}</text>')
    body.append(f'<path d="M {lx},{ly + 54} h 22" class="ll"/>')
    body.append(f'<text x="{lx + 28}" y="{ly + 58}" class="lab">log-likelihood (what we actually want)</text>')
    body.append(f'<text x="{F.x0 + F.w}" y="{F.y0 + F.h + 38}" class="sm" text-anchor="end">'
                f'θ = μ₁, the mean of the first component (the others held fixed)</text>')
    body.append(f'<text x="{F.x0 - 8}" y="{F.y0 + 10}" class="sm" text-anchor="end">high</text>')
    body.append(f'<text x="{F.x0 - 8}" y="{F.y0 + F.h}" class="sm" text-anchor="end">low</text>')
    desc = ("A red curve shows the log-likelihood as a function of one parameter, the first component's "
            "mean. Three coloured hump-shaped curves are the EM lower bounds built at successive "
            "iterates theta0, theta1, theta2. Each lies entirely below the red curve and touches it "
            "exactly at the point where it was built; the next iterate is the peak of that bound, so the "
            "dots climb the red curve step by step, from theta0 on the far left toward the maximum.")
    out.write_text(svg(700, 320, "EM as repeatedly maximizing a lower bound that touches the log-likelihood",
                       desc, "\n".join(body)))
    return steps


def fig_gap(out):
    """Stacked bars: log p = L + KL, before the E-step, after it, after the M-step."""
    x = blob_data()
    pi, mu, var = np.array([0.2, 0.8]), np.array([3.0, -0.25]), np.array([3.0, 3.0])
    # "some old q": the posterior under a slightly different theta, e.g. left over from earlier
    q_old = e_step(x, np.array([0.35, 0.65]), np.array([2.0, 0.3]), np.array([3.0, 3.0]))
    s = []
    s.append((lower_bound(x, q_old, pi, mu, var), loglik(x, pi, mu, var)))
    q = e_step(x, pi, mu, var)
    s.append((lower_bound(x, q, pi, mu, var), loglik(x, pi, mu, var)))
    pi2, mu2, var2 = m_step(x, q)
    s.append((lower_bound(x, q, pi2, mu2, var2), loglik(x, pi2, mu2, var2)))
    base = min(a for a, _ in s) - 12
    top = max(b for _, b in s) + 6
    Y = lambda v: 250 - (v - base) / (top - base) * 210
    body = ['<path d="M 30,250 H 670" class="ax"/>']
    titles = ["1. some old q", "2. after the E-step", "3. after the M-step"]
    notes = ["q ≠ posterior: a gap", "q := p(z | x, θ): gap closes", "θ moves: ℒ rises, gap reopens"]
    for i, ((L, ll), t, n) in enumerate(zip(s, titles, notes)):
        cx = 70 + i * 210
        body.append(f'<rect x="{cx}" y="{Y(L):.1f}" width="70" height="{250 - Y(L):.1f}" class="lb"/>')
        if ll - L > 1e-6:
            body.append(f'<rect x="{cx}" y="{Y(ll):.1f}" width="70" height="{Y(L) - Y(ll):.1f}" class="kl"/>')
        body.append(f'<path d="M {cx - 8},{Y(ll):.1f} H {cx + 78}" class="ll"/>')
        body.append(f'<path d="M {cx},{Y(L):.1f} H {cx + 70}" stroke="#2f6fb5" stroke-width="2" class="a"/>')
        body.append(f'<text x="{cx + 84}" y="{Y(ll) + 4:.1f}" class="lab">log p = {ll:.1f}</text>')
        body.append(f'<text x="{cx + 84}" y="{Y(L) + 18 if Y(L) - Y(ll) < 16 else Y(L) + 4:.1f}" class="lab">ℒ = {L:.1f}</text>')
        if Y(L) - Y(ll) > 16:
            body.append(f'<text x="{cx + 35}" y="{(Y(L) + Y(ll)) / 2 + 4:.1f}" class="lab" text-anchor="middle">KL</text>')
        body.append(f'<text x="{cx}" y="{272}" class="hd">{t}</text>')
        body.append(f'<text x="{cx}" y="{290}" class="sm">{n}</text>')
    for i in range(2):
        x0 = 70 + i * 210 + 150
        body.append(f'<path d="M {x0},140 h 36" class="arr"/><path d="M {x0 + 36},135 l 8,5 l -8,5 z" class="arrf"/>')
    body.append('<text x="30" y="20" class="sm">vertical axis: log-likelihood units, truncated '
                '(bars start at a common floor, not at zero)</text>')
    desc = ("Three stacked bars. In each, the blue part is the lower bound L and the red part on top is "
            "the KL gap, and together they reach the red line, the log-likelihood. Bar one: an arbitrary q "
            "leaves a visible KL gap. Bar two: after the E-step the gap is zero and L equals the "
            "log-likelihood. Bar three: after the M-step L has grown, and the log-likelihood has grown by "
            "at least as much because a new KL gap has opened above it.")
    out.write_text(svg(700, 300, "log p = L + KL through one EM iteration", desc, "\n".join(body)))
    return s


def fig_jensen(out):
    """log of an average >= average of the logs."""
    F = Frame(60, 20, 330, 200, (0.1, 5.2), (-2.4, 1.8))
    grid = np.linspace(0.12, 5.2, 300)
    a, b, w = 0.4, 4.5, 0.5
    m = w * a + (1 - w) * b
    body = [F.axis(), path(F.X(grid), F.Y(np.log(grid)), "mix")]
    body.append(f'<path d="M {F.X(a):.1f},{F.Y(np.log(a)):.1f} L {F.X(b):.1f},{F.Y(np.log(b)):.1f}" class="a"/>')
    for v in (a, b):
        body.append(f'<circle cx="{F.X(v):.1f}" cy="{F.Y(np.log(v)):.1f}" r="3.5" class="dot"/>')
    avg_log = w * np.log(a) + (1 - w) * np.log(b)
    body.append(f'<circle cx="{F.X(m):.1f}" cy="{F.Y(avg_log):.1f}" r="4" class="af"/>')
    body.append(f'<circle cx="{F.X(m):.1f}" cy="{F.Y(np.log(m)):.1f}" r="4" class="bf"/>')
    body.append(f'<line x1="{F.X(m):.1f}" x2="{F.X(m):.1f}" y1="{F.Y(np.log(m)):.1f}" y2="{F.y0 + F.h}" class="drop"/>')
    body.append(f'<text x="{F.X(m) - 8:.1f}" y="{F.Y(np.log(m)) - 8:.1f}" class="lab" text-anchor="end">log(average)</text>')
    body.append(f'<text x="{F.X(m) + 8:.1f}" y="{F.Y(avg_log) + 14:.1f}" class="lab">average of logs</text>')
    body.append(f'<text x="{F.X(a):.1f}" y="{F.y0 + F.h + 16}" class="lab" text-anchor="middle">a</text>')
    body.append(f'<text x="{F.X(b):.1f}" y="{F.y0 + F.h + 16}" class="lab" text-anchor="middle">b</text>')
    body.append(f'<text x="{F.X(m):.1f}" y="{F.y0 + F.h + 16}" class="lab" text-anchor="middle">(a+b)/2</text>')
    body.append(f'<text x="{F.X(5.0):.1f}" y="{F.Y(np.log(5.0)) - 10:.1f}" class="hd" text-anchor="end">log x</text>')
    tx = 430
    lines = [
        ("hd", "Jensen's inequality for log"),
        ("lab", "log bends downward (it is concave), so the"),
        ("lab", "straight chord between two points on it"),
        ("lab", "lies below the curve. At the midpoint:"),
        ("lab", ""),
        ("lab", "  log( ½a + ½b )  ≥  ½ log a + ½ log b"),
        ("lab", ""),
        ("lab", "Same for any weights q₁…q_K summing to 1:"),
        ("lab", "  log Σ q_k r_k  ≥  Σ q_k log r_k"),
        ("lab", ""),
        ("lab", "EM uses this to pull the log inside the sum."),
        ("lab", "Equality holds when all the r_k are equal."),
    ]
    for i, (c, t) in enumerate(lines):
        body.append(f'<text x="{tx}" y="{40 + i * 18}" class="{c}" xml:space="preserve">{t}</text>')
    desc = ("The curve log x with two points a and b joined by a straight chord that lies below the curve. "
            "Above their midpoint, an orange dot on the curve (log of the average) sits higher than a blue "
            "dot on the chord (average of the logs). Text beside it states log of a weighted sum is at "
            "least the weighted sum of logs, with equality when all terms are equal.")
    out.write_text(svg(720, 260, "Jensen's inequality: log of an average is at least the average of logs",
                       desc, "\n".join(body)))


def write_figures():
    d = Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    fig_mixture(d / "mixture.svg")
    fig_iterations(d / "iterations.svg")
    steps = fig_lower_bound(d / "lower-bound.svg")
    gap = fig_gap(d / "kl-gap.svg")
    fig_jensen(d / "jensen.svg")
    print("figures: wrote", ", ".join(sorted(p.name for p in d.glob("*.svg"))))
    print("  lower-bound iterates mu_1:", np.round(steps, 3))
    print("  gap bars (L, log p):", [(round(a, 1), round(b, 1)) for a, b in gap])


if __name__ == "__main__":
    check_worked()
    check_decomposition()
    check_monotone()
    check_kmeans_limit()
    check_singularity()
    if "--figures" in sys.argv:
        write_figures()
