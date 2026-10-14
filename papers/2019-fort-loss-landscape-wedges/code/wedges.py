#!/usr/bin/env python3
"""The wedge model of Fort & Jastrzebski (NeurIPS 2019), from scratch in numpy.

Section 3 of the paper builds a toy loss landscape: take all n-subsets of the D coordinate axes and
put an "n-wedge" (long in those n directions, thin in the other D-n) on each. The surrogate loss is
the distance to the nearest wedge:

    L_toy(P) = sqrt( sum of the (D-n) smallest P_i^2 ).

This script runs the paper's experiments on that toy and checks the claims drawn from it:

  1. two optimised random points sit on different wedges, and the straight line between them leaves
     the low-loss set (the midpoint has fewer zero coordinates than a wedge needs)
  2. the paper's tunnel algorithm (hyperplanes normal to the line, optimise inside each) builds a
     zero-loss connector; the deviations from the line are aligned within each half and orthogonal
     across the halves; the connector passes through intermediate wedges
  3. optimising in a random d-dimensional hyperplane works once d >= D - n  (intrinsic dimension)
  4. m-connectors: points in the convex hull of m+1 optima can be driven to zero loss inside a
     (D-m)-dimensional hyperplane
  5. weight-averaging two points works iff they share enough short directions (the SWA prediction)

Only numpy. Run:  python3 wedges.py             (prints every number quoted in the notes)
                  python3 wedges.py --figures   (also rewrites ../figures/*.svg)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
D, N_LONG = 100, 90                       # ambient dimension, wedge (long) dimension
S = D - N_LONG                            # number of short directions
TOL = 1e-6


def toy_loss(P):
    idx = np.argsort(np.abs(P))[:S]
    return float(np.sqrt((P[idx] ** 2).sum()))


def toy_grad(P):
    """Gradient of 0.5 * L_toy^2: only the S smallest coordinates feel it."""
    idx = np.argsort(np.abs(P))[:S]
    g = np.zeros_like(P)
    g[idx] = P[idx]
    return g


def optimise(P, project=None, steps=500, lr=0.5):
    P = P.copy()
    for _ in range(steps):
        g = toy_grad(P)
        if project is not None:
            g = project(g)
        P -= lr * g
    return P


def zero_set(P):
    return set(np.flatnonzero(np.abs(P) < TOL))


def part1_two_optima(rng):
    print(f"== 1. Two optima on different wedges (D={D}, wedge dimension n={N_LONG}, {S} short directions) ==")
    A = optimise(rng.standard_normal(D))
    B = optimise(rng.standard_normal(D))
    ZA, ZB = zero_set(A), zero_set(B)
    ts = np.linspace(0, 1, 101)
    lin = np.array([toy_loss((1 - t) * A + t * B) for t in ts])
    mid = (A + B) / 2
    print(f"loss at the two optima: {toy_loss(A):.1e}, {toy_loss(B):.1e};  ||A||={np.linalg.norm(A):.2f}")
    print(f"zero coordinates: |Z_A|={len(ZA)}, |Z_B|={len(ZB)}, shared {len(ZA & ZB)} "
          f"(a random pair shares about S^2/D = {S * S / D:.0f})")
    print(f"zero coordinates at the midpoint: {len(zero_set(mid))} (a wedge point needs {S})")
    print(f"loss on the straight line: max {lin.max():.3f} at t={ts[lin.argmax()]:.2f}; midpoint {toy_loss(mid):.3f}")
    print()
    return A, B, ts, lin


def part2_tunnel(A, B, ts_lin, lin):
    print("== 2. The paper's tunnel algorithm (Sec. 4.2) ==")
    u = (B - A) / np.linalg.norm(B - A)
    proj = lambda g: g - (g @ u) * u                      # stay inside the hyperplane normal to the line
    K = 100
    ts = np.linspace(0, 1, K + 1)
    base = [(1 - t) * A + t * B for t in ts]
    pts = [optimise(b, proj) for b in base]
    dev = np.array([p - b for p, b in zip(pts, base)])
    seg_max = 0.0
    for i in range(K):
        for s in np.linspace(0, 1, 6):
            seg_max = max(seg_max, toy_loss((1 - s) * pts[i] + s * pts[i + 1]))
    print(f"loss at the {K + 1} optimised points: max {max(toy_loss(p) for p in pts):.1e}")
    print(f"loss along the polyline joining them: max {seg_max:.3f}   (straight line: {lin.max():.3f})")
    first, second = slice(1, 40), slice(60, 100)
    C = np.array([[a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12) for b in dev] for a in dev])
    print(f"cosine between deviations from the line: within first half {C[first, first].mean():.2f}, "
          f"within second half {C[second, second].mean():.2f}, across halves {C[first, second].mean():+.2f}")
    ZA, ZB = zero_set(A), zero_set(B)
    ov = [(len(zero_set(p) & ZA), len(zero_set(p) & ZB), len(zero_set(p))) for p in pts]
    print("overlap of each point's zero set with Z_A / Z_B at t = 0, .1, .2, ..., 1:")
    print("   Z_A:", [ov[i][0] for i in range(0, K + 1, 10)])
    print("   Z_B:", [ov[i][1] for i in range(0, K + 1, 10)])
    print(f"   number of zero coordinates at every point: {sorted(set(o[2] for o in ov))}")
    print()
    return ov, seg_max


def hyper_loss(d, seed):
    r = np.random.default_rng(seed)
    M = np.linalg.qr(r.standard_normal((D, d)))[0]        # D x d orthonormal columns
    P = r.standard_normal(D) * 3
    for _ in range(3000):
        P -= 0.5 * (M @ (M.T @ toy_grad(P)))
    return toy_loss(P)


def part3_hyperplanes():
    print(f"== 3. Optimising inside a random d-dimensional hyperplane (D - n = {S}) ==")
    ds = [2, 5, 8, 9, 10, 11, 12, 15, 30]
    out = []
    print(f"{'d':>4s}{'median final loss':>20s}")
    for d in ds:
        v = float(np.median([hyper_loss(d, s) for s in range(5)]))
        out.append((d, v))
        print(f"{d:4d}{v:20.2e}")
    print()
    return out


def part4_mconnectors():
    print("== 4. m-connectors: m+1 optima, hyperplane of dimension D - m (5 random hull points each) ==")
    print(f"{'m':>4s}{'loss before':>14s}{'loss after':>14s}")
    for m in (1, 2, 4, 6, 10):
        r = np.random.default_rng(m)
        opts = [optimise(r.standard_normal(D)) for _ in range(m + 1)]
        Q = np.linalg.qr(np.array([o - opts[0] for o in opts[1:]]).T)[0]
        proj = lambda g: g - Q @ (Q.T @ g)
        res = []
        for _ in range(5):
            w = r.dirichlet(np.ones(m + 1))
            b = sum(wi * o for wi, o in zip(w, opts))
            res.append((toy_loss(b), toy_loss(optimise(b, proj, steps=1500))))
        res = np.mean(res, axis=0)
        print(f"{m:4d}{res[0]:14.3f}{res[1]:14.1e}")
    print()


def part5_averaging():
    print("== 5. Averaging two optima: loss of the midpoint against the number of shared short directions ==")
    print(f"{'shared zeros':>13s}{'midpoint loss (median of 20)':>32s}")
    out = []
    for shared in (10, 8, 6, 4, 2, 0):
        vals = []
        for seed in range(20):
            r = np.random.default_rng(seed)
            perm = r.permutation(D)
            ZA = perm[:S]
            ZB = np.concatenate([ZA[:shared], perm[S:S + (S - shared)]])
            A, B = r.standard_normal(D) * 2, r.standard_normal(D) * 2
            A[ZA] = 0.0
            B[ZB] = 0.0
            vals.append(toy_loss((A + B) / 2))
        v = float(np.median(vals))
        out.append((shared, v))
        print(f"{shared:13d}{v:32.3f}")
    print()
    return out


# ------------------------------------------------------------------ figures
STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:12px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .ax{stroke:#c3c2b7;stroke-width:0.6;fill:none}
  .gd{stroke:#e2e1da;stroke-width:0.5;fill:none;stroke-dasharray:4 4}
  .a{fill:none;stroke:#2f6fb5;stroke-width:2} .b{fill:none;stroke:#c2642a;stroke-width:2}
  .g{fill:none;stroke:#2a8a5c;stroke-width:2} .k{fill:none;stroke:#1a1a19;stroke-width:1.2;stroke-dasharray:5 3}
  .ia{fill:#2f6fb5} .ib{fill:#c2642a} .ig{fill:#2a8a5c}
  @media (prefers-color-scheme: dark){
    .lab{fill:#b6b4ab} .hd{fill:#eceae3} .ax{stroke:#4a4844} .gd{stroke:#33312e}
    .a{stroke:#7fb2e8} .b{stroke:#e0955c} .g{stroke:#5fcf96} .k{stroke:#eceae3}
    .ia{fill:#7fb2e8} .ib{fill:#e0955c} .ig{fill:#5fcf96}
  }
</style>"""


def frame(W, H, title, desc, head):
    return [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img">',
            f'<title>{title}</title>', f'<desc>{desc}</desc>', STYLE,
            f'<text class="hd" x="70" y="24">{head}</text>']


def fig_line(ts, lin, seg_max):
    W, H, L, R, T, Bt = 780, 300, 70, 20, 40, 50
    ymax = 0.25
    X = lambda t: L + (W - L - R) * t
    Y = lambda v: T + (H - T - Bt) * (1 - v / ymax)
    p = frame(W, H, "Toy loss along the straight line between two optima and along the tunnel",
              "The straight line between two optima rises to a loss of about 0.29 near one end, while the tunnel stays at zero at its optimised points and below about 0.07 between them.",
              "Toy loss along the path between two optima (D=100, n=90)")
    for v in (0, 0.05, 0.1, 0.15, 0.2, 0.25):
        p.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" class="gd"/>'
                 f'<text class="lab" x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
    p.append(f'<line x1="{L}" y1="{H-Bt}" x2="{W-R}" y2="{H-Bt}" class="ax"/>')
    for t in (0, 0.5, 1):
        p.append(f'<text class="lab" x="{X(t):.1f}" y="{H-Bt+18}" text-anchor="middle">{t}</text>')
    p.append(f'<text class="lab" x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">position along the path (0 = optimum A, 1 = optimum B)</text>')
    p.append('<polyline class="b" points="' + " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(ts, lin)) + '"/>')
    p.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{W-R}" y2="{Y(0):.1f}" class="g"/>')
    p.append(f'<line x1="470" y1="{T+6}" x2="495" y2="{T+6}" class="b"/><text class="lab" x="502" y="{T+10}">straight line (max {lin.max():.2f})</text>')
    p.append(f'<line x1="470" y1="{T+24}" x2="495" y2="{T+24}" class="g"/><text class="lab" x="502" y="{T+28}">tunnel (segments between points: max {seg_max:.2f})</text>')
    p.append('</svg>')
    return "\n".join(p)


def fig_hyper(res):
    W, H, L, R, T, Bt = 780, 300, 70, 20, 40, 50
    X = lambda d: L + (W - L - R) * np.log10(d / 2) / np.log10(30 / 2)
    Y = lambda v: T + (H - T - Bt) * (1 - (np.log10(max(v, 1e-12)) + 12) / 12.5)
    p = frame(W, H, "Final toy loss against the dimension of a random hyperplane",
              "The final loss stays large for hyperplane dimensions below ten and falls to numerically zero by fifteen, with the edge at D minus n equal to ten.",
              "Optimising in a random d-dimensional hyperplane (D - n = 10)")
    for e in range(-12, 1, 3):
        p.append(f'<line x1="{L}" y1="{Y(10.0**e):.1f}" x2="{W-R}" y2="{Y(10.0**e):.1f}" class="gd"/>'
                 f'<text class="lab" x="{L-8}" y="{Y(10.0**e)+4:.1f}" text-anchor="end">1e{e}</text>')
    p.append(f'<line x1="{L}" y1="{H-Bt}" x2="{W-R}" y2="{H-Bt}" class="ax"/>')
    for d in (2, 5, 10, 15, 30):
        p.append(f'<text class="lab" x="{X(d):.1f}" y="{H-Bt+18}" text-anchor="middle">{d}</text>')
    p.append(f'<text class="lab" x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">hyperplane dimension d</text>')
    p.append(f'<line x1="{X(S):.1f}" y1="{T}" x2="{X(S):.1f}" y2="{H-Bt}" class="k"/><text class="lab" x="{X(S)+6:.1f}" y="{T+14}">d = D - n</text>')
    p.append('<polyline class="a" points="' + " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in res) + '"/>')
    for d, v in res:
        p.append(f'<circle cx="{X(d):.1f}" cy="{Y(v):.1f}" r="3.5" class="ia"/>')
    p.append('</svg>')
    return "\n".join(p)


def fig_zero(ov):
    W, H, L, R, T, Bt = 780, 300, 70, 20, 40, 50
    K = len(ov) - 1
    X = lambda i: L + (W - L - R) * i / K
    Y = lambda v: T + (H - T - Bt) * (1 - v / 10.5)
    p = frame(W, H, "Which short directions each tunnel point shares with the two optima",
              "Along the tunnel, the zero coordinates shared with optimum A fall from ten to one, then the zero coordinates shared with optimum B rise from zero to ten, with a stretch in the middle that matches neither.",
              "Zero coordinates of each tunnel point shared with Z_A and Z_B")
    for v in (0, 5, 10):
        p.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" class="gd"/>'
                 f'<text class="lab" x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end">{v}</text>')
    p.append(f'<line x1="{L}" y1="{H-Bt}" x2="{W-R}" y2="{H-Bt}" class="ax"/>')
    for i in (0, K // 2, K):
        p.append(f'<text class="lab" x="{X(i):.1f}" y="{H-Bt+18}" text-anchor="middle">{i/K:.1f}</text>')
    p.append(f'<text class="lab" x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">position along the tunnel</text>')
    for j, cls in ((0, "a"), (1, "b")):
        p.append('<polyline class="' + cls + '" points="' + " ".join(f"{X(i):.1f},{Y(o[j]):.1f}" for i, o in enumerate(ov)) + '"/>')
    p.append(f'<line x1="520" y1="{T+6}" x2="545" y2="{T+6}" class="a"/><text class="lab" x="552" y="{T+10}">shared with Z_A</text>')
    p.append(f'<line x1="520" y1="{T+24}" x2="545" y2="{T+24}" class="b"/><text class="lab" x="552" y="{T+28}">shared with Z_B</text>')
    p.append('</svg>')
    return "\n".join(p)


def fig_avg(res):
    W, H, L, R, T, Bt = 780, 300, 70, 20, 40, 50
    shared = [r[0] for r in res]
    X = lambda s: L + (W - L - R) * (10 - s) / 10
    Y = lambda v: T + (H - T - Bt) * (1 - v / 0.45)
    p = frame(W, H, "Loss of the weight average against the number of shared short directions",
              "The average of two points has zero loss when they share all ten short directions and rises steadily as they share fewer, reaching about 0.32 when they share none.",
              "Loss at the midpoint of two optima vs shared short directions (median of 20)")
    for v in (0, 0.1, 0.2, 0.3, 0.4):
        p.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" class="gd"/>'
                 f'<text class="lab" x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.1f}</text>')
    p.append(f'<line x1="{L}" y1="{H-Bt}" x2="{W-R}" y2="{H-Bt}" class="ax"/>')
    for s in (10, 8, 6, 4, 2, 0):
        p.append(f'<text class="lab" x="{X(s):.1f}" y="{H-Bt+18}" text-anchor="middle">{s}</text>')
    p.append(f'<text class="lab" x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">zero coordinates shared by the two optima (10 = same wedge)</text>')
    p.append('<polyline class="b" points="' + " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in res) + '"/>')
    for s, v in res:
        p.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3.5" class="ib"/>')
    p.append('</svg>')
    return "\n".join(p)


def main():
    rng = np.random.default_rng(0)
    A, B, ts, lin = part1_two_optima(rng)
    ov, seg_max = part2_tunnel(A, B, ts, lin)
    hyp = part3_hyperplanes()
    part4_mconnectors()
    avg = part5_averaging()
    if "--figures" in sys.argv:
        FIG_DIR.mkdir(exist_ok=True)
        (FIG_DIR / "line_vs_tunnel.svg").write_text(fig_line(ts, lin, seg_max))
        (FIG_DIR / "hyperplane_dimension.svg").write_text(fig_hyper(hyp))
        (FIG_DIR / "tunnel_zero_sets.svg").write_text(fig_zero(ov))
        (FIG_DIR / "averaging.svg").write_text(fig_avg(avg))
        print("wrote figures/line_vs_tunnel.svg, hyperplane_dimension.svg, tunnel_zero_sets.svg, averaging.svg")


if __name__ == "__main__":
    main()
