#!/usr/bin/env python3
"""Mode connectivity on a small network, from scratch in numpy.

Three notions of "two trained networks are connected", checked on a one-hidden-layer tanh network
trained on a two-class 2-D problem (two noisy half-moons):

  1. linear connectivity     the loss on the straight line between two independently trained solutions
  2. curved connectivity     a path with one trained bend point (Garipov et al. 2018)
  3. linear after alignment  permute (and sign-flip) the hidden units of one network to match the other,
                             then take the straight line (the permutation story of Entezari et al. 2022,
                             Ainsworth et al. 2023)

It also sweeps the hidden width, and draws the loss on the plane through the two solutions and the bend point.
The barrier height of Entezari et al. 2021 is  BH = max_a [ L(a A + (1-a) B) - (a L(A) + (1-a) L(B)) ].

Only numpy. Run:  python3 connectivity.py            (prints every number quoted in the notes)
                  python3 connectivity.py --figures   (also rewrites ../figures/*.svg)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


# ----------------------------------------------------------------- data and network
def moons(n, seed):
    r = np.random.default_rng(seed)
    t = r.uniform(0, np.pi, n // 2)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.r_[a, b] + 0.15 * r.standard_normal((n, 2))
    return X, np.r_[np.zeros(n // 2), np.ones(n // 2)]


class Net:
    """2 -> H (tanh) -> 1 (logit); parameters live in one flat vector."""

    def __init__(self, H):
        self.H = H
        self.dim = 4 * H + 1

    def unpack(self, th):
        H = self.H
        return th[:2 * H].reshape(H, 2), th[2 * H:3 * H], th[3 * H:4 * H], th[4 * H]

    def forward(self, th, X):
        W1, b1, w2, b2 = self.unpack(th)
        Z = np.tanh(X @ W1.T + b1)
        return Z, Z @ w2 + b2

    def loss_grad(self, th, X, y):
        W1, b1, w2, b2 = self.unpack(th)
        Z, o = self.forward(th, X)
        p = 1 / (1 + np.exp(-o))
        L = np.mean(np.logaddexp(0, o) - y * o)
        do = (p - y) / len(y)
        dZ = np.outer(do, w2) * (1 - Z ** 2)
        return L, np.r_[(dZ.T @ X).ravel(), dZ.sum(0), Z.T @ do, do.sum()]

    def loss(self, th, X, y):
        return self.loss_grad(th, X, y)[0]

    def acc(self, th, X, y):
        return float(np.mean((self.forward(th, X)[1] > 0) == y))


def adam_step(state, g, lr, t):
    m, v = state
    m[:] = 0.9 * m + 0.1 * g
    v[:] = 0.999 * v + 0.001 * g * g
    return lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def train(net, X, y, seed, steps=4000, lr=0.02):
    th = np.random.default_rng(seed).standard_normal(net.dim) * 0.8
    st = (np.zeros_like(th), np.zeros_like(th))
    for t in range(1, steps + 1):
        th = th - adam_step(st, net.loss_grad(th, X, y)[1], lr, t)
    return th


# ----------------------------------------------------------------- paths
def line_losses(net, a, b, X, y, ts):
    return np.array([net.loss(((1 - t) * a + t * b), X, y) for t in ts])


def barrier_height(Ls, ts):
    return float(np.max(Ls - ((1 - ts) * Ls[0] + ts * Ls[-1])))


def bend_point(th_at, a, b, net, X, y, steps=600, lr=0.02, seed=0):
    """Garipov et al.: train the bend point m so the polygonal chain a -> m -> b has low expected loss."""
    m = (a + b) / 2
    st = (np.zeros_like(m), np.zeros_like(m))
    r = np.random.default_rng(seed)
    for t in range(1, steps + 1):
        g = np.zeros_like(m)
        for s in r.uniform(0, 1, 8):
            c = 2 * s if s <= 0.5 else 2 * (1 - s)
            g += c * net.loss_grad(th_at(m, s), X, y)[1] / 8
        m = m - adam_step(st, g, lr, t)
    return m


def chain(a, b):
    def at(m, s):
        return 2 * s * m + (1 - 2 * s) * a if s <= 0.5 else 2 * (1 - s) * m + (2 * s - 1) * b
    return at


def hungarian(C):
    """Minimum-cost assignment (Kuhn-Munkres with potentials), O(n^3)."""
    n = C.shape[0]
    INF = 1e18
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, int), np.zeros(n + 1, int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(n + 1, INF), np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    asg = np.zeros(n, int)
    for j in range(1, n + 1):
        asg[p[j] - 1] = j - 1
    return asg


def align(net, a, b):
    """Re-order (and sign-flip, since tanh is odd) the hidden units of b to match a. Same function as b."""
    H = net.H
    Fa = np.c_[net.unpack(a)[0], net.unpack(a)[1], net.unpack(a)[2]]
    Fb = np.c_[net.unpack(b)[0], net.unpack(b)[1], net.unpack(b)[2]]
    C, S = np.zeros((H, H)), np.ones((H, H))
    for i in range(H):
        for j in range(H):
            c1, c2 = ((Fa[i] - Fb[j]) ** 2).sum(), ((Fa[i] + Fb[j]) ** 2).sum()
            C[i, j], S[i, j] = min(c1, c2), (1 if c1 <= c2 else -1)
    asg = hungarian(C)
    Fn = np.array([S[i, asg[i]] * Fb[asg[i]] for i in range(H)])
    return np.r_[Fn[:, :2].ravel(), Fn[:, 2], Fn[:, 3], net.unpack(b)[3]]


# ----------------------------------------------------------------- experiments
def main_experiment():
    print("== One pair of solutions (hidden width 10, 300 training points, 2000 test points) ==")
    X, y = moons(300, 0)
    Xt, yt = moons(2000, 1)
    net = Net(10)
    sols = [train(net, X, y, s) for s in range(6)]
    A, B = sols[0], sols[2]
    ts = np.linspace(0, 1, 101)
    raw = line_losses(net, A, B, X, y, ts)
    Bal = align(net, A, B)
    same_fn = np.abs(net.forward(Bal, Xt)[1] - net.forward(B, Xt)[1]).max()
    ali = line_losses(net, A, Bal, X, y, ts)
    at = chain(A, B)
    m = bend_point(at, A, B, net, X, y)
    bent = np.array([net.loss(at(m, s), X, y) for s in ts])
    print(f"train loss at the two solutions: {net.loss(A, X, y):.1e}, {net.loss(B, X, y):.1e}; "
          f"test accuracy {net.acc(A, Xt, yt):.3f}, {net.acc(B, Xt, yt):.3f}")
    print(f"alignment changes the function of B by at most {same_fn:.1e} (it is a symmetry)")
    print(f"{'path':28s}{'max train loss':>16s}{'barrier height':>16s}{'test acc at midpoint':>24s}")
    for name, Ls, mid in (("straight line", raw, 0.5 * (A + B)),
                          ("straight line after align", ali, 0.5 * (A + Bal)),
                          ("bend (Garipov)", bent, at(m, 0.5))):
        print(f"{name:28s}{Ls.max():16.4f}{barrier_height(Ls, ts):16.4f}{net.acc(mid, Xt, yt):24.3f}")
    print()
    return net, X, y, A, B, m, ts, raw, ali, bent


def all_pairs():
    print("== Five pairs against one reference solution: barrier height on the straight line ==")
    X, y = moons(300, 0)
    net = Net(10)
    sols = [train(net, X, y, s) for s in range(6)]
    ts = np.linspace(0, 1, 41)
    print(f"{'pair':>6s}{'raw':>10s}{'aligned':>10s}")
    raws, alis = [], []
    for i in range(1, 6):
        r = barrier_height(line_losses(net, sols[0], sols[i], X, y, ts), ts)
        a = barrier_height(line_losses(net, sols[0], align(net, sols[0], sols[i]), X, y, ts), ts)
        raws.append(r); alis.append(a)
        print(f"  0-{i:<3d}{r:10.3f}{a:10.4f}")
    print(f"{'median':>6s}{np.median(raws):10.3f}{np.median(alis):10.4f}")
    print()


def width_sweep():
    print("== Hidden width against the straight-line barrier (median over 7 pairs) ==")
    X, y = moons(300, 0)
    ts = np.linspace(0, 1, 41)
    out = []
    print(f"{'width':>6s}{'raw':>10s}{'aligned':>10s}")
    for H in (2, 3, 4, 6, 10, 20):
        net = Net(H)
        sols = [train(net, X, y, s, steps=3000) for s in range(8)]
        raws, alis = [], []
        for i in range(1, 8):
            raws.append(barrier_height(line_losses(net, sols[0], sols[i], X, y, ts), ts))
            alis.append(barrier_height(line_losses(net, sols[0], align(net, sols[0], sols[i]), X, y, ts), ts))
        out.append((H, float(np.median(raws)), float(np.median(alis))))
        print(f"{H:6d}{out[-1][1]:10.3f}{out[-1][2]:10.4f}")
    print()
    return out


# ----------------------------------------------------------------- figures
STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:12px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .ax{stroke:#c3c2b7;stroke-width:0.6;fill:none}
  .gd{stroke:#e2e1da;stroke-width:0.5;fill:none;stroke-dasharray:4 4}
  .a{fill:none;stroke:#2f6fb5;stroke-width:2} .b{fill:none;stroke:#c2642a;stroke-width:2}
  .g{fill:none;stroke:#2a8a5c;stroke-width:2}
  .ia{fill:#2f6fb5} .ib{fill:#c2642a} .ig{fill:#2a8a5c}
  @media (prefers-color-scheme: dark){
    .lab{fill:#b6b4ab} .hd{fill:#eceae3} .ax{stroke:#4a4844} .gd{stroke:#33312e}
    .a{stroke:#7fb2e8} .b{stroke:#e0955c} .g{stroke:#5fcf96}
    .ia{fill:#7fb2e8} .ib{fill:#e0955c} .ig{fill:#5fcf96}
  }
</style>"""


def frame(W, H, title, desc, head):
    return [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img">',
            f'<title>{title}</title>', f'<desc>{desc}</desc>', STYLE,
            f'<text class="hd" x="70" y="24">{head}</text>']


def fig_paths(ts, raw, ali, bent):
    W, H, L, R, T, Bt = 780, 320, 70, 20, 40, 50
    ymax = 2.0
    X = lambda t: L + (W - L - R) * t
    Y = lambda v: T + (H - T - Bt) * (1 - min(v, ymax) / ymax)
    p = frame(W, H, "Training loss along three paths between two independently trained networks",
              f"The straight line between the two networks rises to a loss of {raw.max():.1f}. The straight line after re-ordering the hidden units of one network stays near zero, and so does the path with one trained bend, below {bent.max():.2f}.",
              "Training loss between two trained networks (width 10)")
    for v in (0, 0.5, 1.0, 1.5, 2.0):
        p.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" class="gd"/>'
                 f'<text class="lab" x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.1f}</text>')
    p.append(f'<line x1="{L}" y1="{H-Bt}" x2="{W-R}" y2="{H-Bt}" class="ax"/>')
    for t in (0, 0.5, 1):
        p.append(f'<text class="lab" x="{X(t):.1f}" y="{H-Bt+18}" text-anchor="middle">{t}</text>')
    p.append(f'<text class="lab" x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">position along the path (0 = network A, 1 = network B)</text>')
    for arr, cls in ((raw, "b"), (ali, "a"), (bent, "g")):
        p.append(f'<polyline class="{cls}" points="' + " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(ts, arr)) + '"/>')
    for j, (cls, nm) in enumerate((("b", f"straight line (max {raw.max():.2f})"), ("a", f"straight line after aligning units (max {ali.max():.3f})"),
                                    ("g", f"path with one trained bend (max {bent.max():.3f})"))):
        p.append(f'<line x1="400" y1="{T+6+18*j}" x2="425" y2="{T+6+18*j}" class="{cls}"/><text class="lab" x="432" y="{T+10+18*j}">{nm}</text>')
    p.append('</svg>')
    return "\n".join(p)


def fig_width(res):
    W, H, L, R, T, Bt = 780, 300, 70, 20, 40, 50
    X = lambda h: L + (W - L - R) * (np.log2(h) - 1) / (np.log2(20) - 1)
    Y = lambda v: T + (H - T - Bt) * (1 - (np.log10(max(v, 1e-4)) + 4) / 4.5)
    p = frame(W, H, "Straight-line barrier height against hidden width, before and after aligning units",
              "The raw barrier is not monotone: it grows from 0.4 at width two to about 1.7 at width six and falls to about 0.6 at width twenty. After aligning the hidden units it is near zero at every width from three up, and 0.17 at width two.",
              "Straight-line barrier height vs hidden width (median of 7 pairs)")
    for e in range(-4, 1):
        p.append(f'<line x1="{L}" y1="{Y(10.0**e):.1f}" x2="{W-R}" y2="{Y(10.0**e):.1f}" class="gd"/>'
                 f'<text class="lab" x="{L-8}" y="{Y(10.0**e)+4:.1f}" text-anchor="end">1e{e}</text>')
    p.append(f'<line x1="{L}" y1="{H-Bt}" x2="{W-R}" y2="{H-Bt}" class="ax"/>')
    for h in (2, 3, 4, 6, 10, 20):
        p.append(f'<text class="lab" x="{X(h):.1f}" y="{H-Bt+18}" text-anchor="middle">{h}</text>')
    p.append(f'<text class="lab" x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">hidden units</text>')
    for idx, cls, nm in ((1, "b", "raw straight line"), (2, "a", "after aligning units")):
        p.append(f'<polyline class="{cls}" points="' + " ".join(f"{X(r[0]):.1f},{Y(r[idx]):.1f}" for r in res) + '"/>')
        for r in res:
            p.append(f'<circle cx="{X(r[0]):.1f}" cy="{Y(r[idx]):.1f}" r="3.5" class="i{cls}"/>')
    for j, (cls, nm) in enumerate((("b", "raw straight line"), ("a", "after aligning units"))):
        yy = Y(0.06) + 18 * j
        p.append(f'<line x1="330" y1="{yy:.1f}" x2="355" y2="{yy:.1f}" class="{cls}"/><text class="lab" x="362" y="{yy+4:.1f}">{nm}</text>')
    p.append('</svg>')
    return "\n".join(p)


def fig_plane(net, X, y, A, B, m):
    """Training loss on the plane through A, B and the bend point m (Garipov et al., Fig. 1)."""
    u = (B - A) / np.linalg.norm(B - A)
    w = (m - A) - ((m - A) @ u) * u
    v = w / np.linalg.norm(w)
    Lab, xm, ym = np.linalg.norm(B - A), (m - A) @ u, (m - A) @ v
    xs = np.linspace(-0.35 * Lab, 1.35 * Lab, 64)
    ys = np.linspace(min(0, ym) - 0.5 * abs(ym) - 0.3 * Lab, max(0, ym) + 0.5 * abs(ym) + 0.3 * Lab, 40)
    G = np.array([[net.loss(A + a * u + b * v, X, y) for a in xs] for b in ys])
    W, H, L, T = 780, 360, 30, 40
    cw, ch = (W - 2 * L) / len(xs), (H - T - 20) / len(ys)

    def col(val):                       # log scale, 1e-3 .. 3: light teal -> orange -> deep red
        f = np.clip((np.log10(max(val, 1e-3)) + 3) / (np.log10(3) + 3), 0, 1)
        r, g, b = (np.interp(f, [0, .5, 1], [220, 240, 150]), np.interp(f, [0, .5, 1], [240, 170, 40]),
                   np.interp(f, [0, .5, 1], [235, 110, 40]))
        return f"rgb({int(r)},{int(g)},{int(b)})"
    P = lambda a, b: (L + (a - xs[0]) / (xs[-1] - xs[0]) * (W - 2 * L), T + (ys[-1] - b) / (ys[-1] - ys[0]) * (H - T - 20))
    p = frame(W, H, "Training loss on the plane through two trained networks and the bend point",
              "A heat map of the training loss. Both networks sit in low-loss valleys; the straight line between them crosses a high-loss ridge, while the two-segment path through the bend point stays in low loss.",
              "Training loss on the plane through A, B and the trained bend point (log colour scale)")
    for i, b in enumerate(ys[::-1]):
        for j, a in enumerate(xs):
            p.append(f'<rect x="{L + j*cw:.1f}" y="{T + i*ch:.1f}" width="{cw+0.6:.1f}" height="{ch+0.6:.1f}" fill="{col(G[len(ys)-1-i, j])}"/>')
    pa, pb, pm = P(0, 0), P(Lab, 0), P(xm, ym)
    p.append(f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="#1a1a19" stroke-width="2" stroke-dasharray="6 4"/>')
    p.append(f'<polyline points="{pa[0]:.1f},{pa[1]:.1f} {pm[0]:.1f},{pm[1]:.1f} {pb[0]:.1f},{pb[1]:.1f}" fill="none" stroke="#1a1a19" stroke-width="2.4"/>')
    for q, nm in ((pa, "A"), (pb, "B"), (pm, "bend")):
        p.append(f'<circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="5" fill="#1a1a19" stroke="#fff" stroke-width="1.5"/>'
                 f'<text x="{q[0]+9:.1f}" y="{q[1]-8:.1f}" font-size="13" fill="#1a1a19" stroke="#fff" stroke-width="3" paint-order="stroke">{nm}</text>')
    p.append(f'<text class="lab" x="{L}" y="{H-4}">dashed: straight line; solid: path through the bend point. Colour: light = low loss, deep orange-red = loss of 3 or more.</text>')
    p.append('</svg>')
    return "\n".join(p), float(G.min()), float(G.max())


def main():
    net, X, y, A, B, m, ts, raw, ali, bent = main_experiment()
    all_pairs()
    wid = width_sweep()
    if "--figures" in sys.argv:
        FIG_DIR.mkdir(exist_ok=True)
        (FIG_DIR / "paths.svg").write_text(fig_paths(ts, raw, ali, bent))
        (FIG_DIR / "width.svg").write_text(fig_width(wid))
        svg, lo, hi = fig_plane(net, X, y, A, B, m)
        (FIG_DIR / "plane.svg").write_text(svg)
        print(f"plane: loss range {lo:.1e} .. {hi:.2f}")
        print("wrote figures/paths.svg, width.svg, plane.svg")


if __name__ == "__main__":
    main()
