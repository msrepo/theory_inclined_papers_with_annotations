#!/usr/bin/env python3
"""Section 2.1 and 3 of Gao & Chaudhari: what the Fisher-Rao length in (2)-(5), (11), (12b) and (21) measures.

Checked here, all on softmax / logistic models small enough to do exactly:

  1. Eq. (2): 2 KL(p_w || p_{w+dw}) = dw^T g dw to second order, with g the Fisher matrix (3).
  2. The Fisher-Rao geometry of a categorical distribution is a sphere of radius 2:
     d_FR(p, q) = 2 arccos sum_i sqrt(p_i q_i), at most pi. The length of a curve does not
     depend on the coordinates it is written in (logits or probabilities), but it does depend on
     the curve: a straight line in logit space is longer than the geodesic.
  3. Eq. (4)/(5) against Eq. (11). The FIM (3) averages the KL over x *inside* the square root;
     Definition 2 averages the square root over x. By Jensen the second is never larger, and
     it is not the length in any Riemannian metric (it is a Finsler norm).
  4. Eq. (12b) against the sentence that introduces it ("the Fisher-Rao distance between
     p_{w(0)}(.|x_s^i) and p_{w(1)}(.|x_t^j)"). Those are different numbers.
  5. Eq. (21): replacing the true Fisher by the empirical one (labels as Dirac deltas) rescales
     each input's contribution to the length by sqrt((1-p_y)/p_y).

Standard library and numpy only.

Run:  python3 fisher_rao.py             (prints every number quoted in the notes)
      python3 fisher_rao.py --figures   (also rewrites ../figures/fisher-balls.svg, empirical-fisher.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np


def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max())
    return e / e.sum()


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


# --------------------------------------------------------------------------- 1. KL is the Fisher
def check_kl_hessian():
    print("1. Eq. (2): 2 KL(p_z || p_{z+dz}) against dz^T F dz, softmax with 4 classes")
    rng = np.random.default_rng(0)
    z = np.array([1.3, -0.4, 0.2, 0.9])
    p = softmax(z)
    F = np.diag(p) - np.outer(p, p)  # Fisher matrix in logit coordinates
    u = rng.normal(size=4)
    u /= np.linalg.norm(u)
    print("   eps      2KL          dz^T F dz    ratio")
    for eps in (1.0, 0.3, 0.1, 0.03, 0.01):
        dz = eps * u
        two_kl = 2 * kl(p, softmax(z + dz))
        quad = float(dz @ F @ dz)
        print(f"   {eps:5.2f}   {two_kl:.6e}  {quad:.6e}  {two_kl / quad:.5f}")
    print()


# --------------------------------------------------------------------------- 2. the sphere
def bc_distance(p, q):
    """Fisher-Rao distance between two categorical distributions: 2 arccos(Bhattacharyya coefficient)."""
    bc = float(np.sum(np.sqrt(p * q)))
    return 2 * math.acos(min(1.0, bc))


def curve_length_prob(ps):
    """Length of a polyline of probability vectors under the metric sum dp^2 / p (midpoint rule)."""
    tot = 0.0
    for a, b in zip(ps[:-1], ps[1:]):
        m = 0.5 * (a + b)
        tot += math.sqrt(float(np.sum((b - a) ** 2 / m)))
    return tot


def curve_length_logit(zs):
    """Same curve, measured in logit coordinates with the metric dz^T F(z) dz."""
    tot = 0.0
    for a, b in zip(zs[:-1], zs[1:]):
        m = 0.5 * (a + b)
        p = softmax(m)
        F = np.diag(p) - np.outer(p, p)
        d = b - a
        tot += math.sqrt(float(d @ F @ d))
    return tot


def geodesic_curve(p0, p1, n=4000):
    u0, u1 = np.sqrt(p0), np.sqrt(p1)
    th = math.acos(min(1.0, float(u0 @ u1)))
    ts = np.linspace(0, 1, n + 1)
    return [((math.sin((1 - t) * th) * u0 + math.sin(t * th) * u1) / math.sin(th)) ** 2 for t in ts]


def logit_line(p0, p1, n=4000):
    z0, z1 = np.log(p0), np.log(p1)
    ts = np.linspace(0, 1, n + 1)
    return [(1 - t) * z0 + t * z1 for t in ts]


def prob_line(p0, p1, n=4000):
    ts = np.linspace(0, 1, n + 1)
    return [(1 - t) * p0 + t * p1 for t in ts]


def check_sphere():
    print("2. Categorical Fisher-Rao geometry: a sphere of radius 2")
    p0 = np.array([0.90, 0.05, 0.05])
    p1 = np.array([0.05, 0.05, 0.90])
    d = bc_distance(p0, p1)
    geo = curve_length_prob(geodesic_curve(p0, p1))
    lin = curve_length_prob(prob_line(p0, p1))
    zs = logit_line(p0, p1)
    lz_logit = curve_length_logit(zs)
    lz_prob = curve_length_prob([softmax(z) for z in zs])
    print(f"   p0 = {p0.tolist()},  p1 = {p1.tolist()}")
    print(f"   closed form 2 arccos(BC)                 = {d:.5f}")
    print(f"   length of the great circle, numerically  = {geo:.5f}")
    print(f"   straight line in probability space       = {lin:.5f}")
    print(f"   straight line in logit space, measured in logit coordinates      = {lz_logit:.5f}")
    print(f"   the same curve, measured in probability coordinates              = {lz_prob:.5f}")
    print(f"   -> the last two agree to {abs(lz_logit - lz_prob):.1e}: length does not know the coordinates")
    print(f"   largest possible distance between two categorical laws = pi = {math.pi:.5f}"
          f" (disjoint supports); here {d / math.pi:.0%} of it")
    # the long way round: confident and opposite
    print("   distance between the logit-line and the geodesic as the pair becomes more confident:")
    print("      conf     d_FR     logit-line   ratio")
    for c in (0.6, 0.9, 0.99, 0.999, 0.99999):
        e = (1 - c) / 2
        a = np.array([c, e, e])
        b = np.array([e, e, c])
        dd = bc_distance(a, b)
        ll = curve_length_logit(logit_line(a, b, 6000))
        print(f"      {c:<8} {dd:6.3f}   {ll:8.3f}     {ll / dd:5.2f}")
    print()
    return p0, p1


# --------------------------------------------------------------------------- 3. Jensen
def sig(t):
    return 1.0 / (1.0 + np.exp(-t))


def bern_two_kl(p, q):
    return 2 * (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


def path_lengths_logistic(w0, w1, xs, wts, n=2000):
    """Straight weight path w(t) = (1-t) w0 + t w1 for a logistic model sigma(w . (x, 1)).

    A = int sqrt( E_x 2KL_x )   the metric of eq. (3): the average sits inside the square root
    B = E_x int sqrt( 2KL_x )   Definition 2 / eq. (11): the average sits outside
    """
    phi = np.stack([xs, np.ones_like(xs)], axis=1)
    ts = np.linspace(0, 1, n + 1)
    A = B = 0.0
    for t0, t1 in zip(ts[:-1], ts[1:]):
        p0 = sig(phi @ ((1 - t0) * w0 + t0 * w1))
        p1 = sig(phi @ ((1 - t1) * w0 + t1 * w1))
        two_kl = bern_two_kl(p0, p1)
        A += math.sqrt(float(np.sum(wts * two_kl)))
        B += float(np.sum(wts * np.sqrt(two_kl)))
    return A, B


def check_jensen():
    print("3. Eq. (5) against Eq. (11): where the average over x sits")
    # half the inputs sit on the decision boundary (they feel every weight change),
    # half are far from it (saturated, they feel almost nothing)
    xs = np.array([0.0, 6.0])
    wts = np.array([0.5, 0.5])
    w0 = np.array([1.0, 0.0])
    print("   two input populations: x = 0 (on the boundary) and x = 6 (saturated), half each;")
    print("   the weight path is a straight line from w0 = (1, 0) to w1 = (slope, 0)")
    print("   w1 slope     A = int sqrt(E_x 2KL)     B = E_x int sqrt(2KL)     B / A")
    for s in (1.5, 2.0, 4.0):
        A, B = path_lengths_logistic(w0, np.array([s, 0.0]), xs, wts)
        print(f"   {s:7.1f}      {A:12.4f}             {B:12.4f}          {B / A:.3f}")
    # a spread of inputs
    rng = np.random.default_rng(1)
    xs = rng.normal(0, 2.0, size=400)
    wts = np.full(400, 1 / 400)
    A, B = path_lengths_logistic(w0, np.array([3.0, 1.0]), xs, wts)
    print(f"   400 inputs ~ N(0, 2^2), w1 = (3, 1): A = {A:.4f}, B = {B:.4f}, B/A = {B / A:.3f}")
    print("   B <= A always (Jensen); B is E_x of a norm of dw, so it is a Finsler length, not Riemannian")
    print()


# --------------------------------------------------------------------------- 4. (12b) vs endpoint FR
def check_12b_vs_endpoint():
    print("4. Eq. (12b) against 'the Fisher-Rao distance between p_w(0)(.|x_s^i) and p_w(1)(.|x_t^j)'")
    a, s = 1.5, 4.0  # source classifier sigma(a x); the target task is the source translated by s
    # translation-equivariant transfer: w(tau) = (a, -a tau s), so p_w(tau)(1 | x_i + tau s) = sigma(a x_i) for all tau
    xi = 0.7
    xj = xi + s
    p_start = sig(a * xi)
    p_end = sig(a * xj - a * s)  # p_{w(1)}(1 | x_j), w(1) = (a, -a s)
    d_end = bc_distance(np.array([p_start, 1 - p_start]), np.array([p_end, 1 - p_end]))
    n = 20000
    C = 0.0
    for k in range(n):
        t = k / n
        xt = (1 - t) * xi + t * xj
        pa = sig(a * xt - a * t * s)
        pb = sig(a * xt - a * (t + 1 / n) * s)
        C += math.sqrt(bern_two_kl(pa, pb))
    closed = a * s * math.sqrt(p_start * (1 - p_start))
    print(f"   perfectly tracking model, pair (x_i = {xi}, x_j = {xj}), source slope a = {a}, shift s = {s}")
    print(f"   endpoint Fisher-Rao distance d_FR(p_w(0)(.|x_i), p_w(1)(.|x_j)) = {d_end:.5f}")
    print(f"   eq. (12b): int sqrt(2KL[p_w(tau)(.|x_tau), p_w(tau+dtau)(.|x_tau)]) = {C:.5f}")
    print(f"   closed form a s sqrt(p(1-p))                                       = {closed:.5f}")
    print("   (12b) is the speed of the weights seen through a frozen input, not a distance between predictions")
    print()


# --------------------------------------------------------------------------- 5. empirical Fisher
def check_empirical_fisher():
    print("5. Eq. (21): the empirical Fisher speed |dl/dtau| against the true Fisher speed")
    print("   For a binary model with P(y=1|x) = p and dz/dtau = logit speed v:")
    print("     true Fisher speed        sqrt(p(1-p)) |v|")
    print("     empirical, label y = 1   (1 - p) |v|        (|dl/dtau|, l = -log p)")
    print("     ratio empirical/true     sqrt((1-p_y)/p_y)")
    print("   p_y (prob. of the observed label)   ratio")
    for py in (0.01, 0.1, 0.5, 0.9, 0.99, 0.999):
        print(f"     {py:<8}                         {math.sqrt((1 - py) / py):8.4f}")
    print()


# --------------------------------------------------------------------------- figures
def figures():
    from svgkit import Axes, svg, write

    # Fisher-Rao balls on the 3-class simplex (ternary plot)
    W, H = 720, 400
    A = np.array([110.0, 350.0])
    B = np.array([430.0, 350.0])
    Cc = np.array([270.0, 73.0])

    def tern(p):
        return p[0] * A + p[1] * B + p[2] * Cc

    def ball(p, r, n=240):
        """Points at Fisher-Rao distance r from p (a circle of angle r/2 on the unit sphere of u = sqrt p)."""
        u0 = np.sqrt(p)
        e1 = np.cross(u0, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(u0, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u0, e1)
        th = r / 2
        runs, cur = [], []
        for ph in np.linspace(0, 2 * math.pi, n + 1):
            u = math.cos(th) * u0 + math.sin(th) * (math.cos(ph) * e1 + math.sin(ph) * e2)
            if np.all(u >= 0):
                cur.append(tern(u ** 2))
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        return runs

    body = [f'<polygon points="{A[0]},{A[1]} {B[0]},{B[1]} {Cc[0]},{Cc[1]}" class="ax" style="fill:none"/>']
    for i, (pt, lab) in enumerate(((A, "class 1"), (B, "class 2"), (Cc, "class 3"))):
        dx = -8 if i == 0 else (8 if i == 1 else 0)
        dy = 18 if i < 2 else -10
        anc = "end" if i == 0 else ("start" if i == 1 else "middle")
        body.append(f'<text class="lab" x="{pt[0] + dx}" y="{pt[1] + dy}" text-anchor="{anc}">{lab}</text>')
    r = 0.35
    centres = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.6, 0.3, 0.1]), np.array([0.8, 0.15, 0.05]),
               np.array([0.94, 0.04, 0.02]), np.array([0.45, 0.5, 0.05])]
    for p in centres:
        for run in ball(p, r):
            pts = " ".join(f"{q[0]:.1f},{q[1]:.1f}" for q in run)
            body.append(f'<polyline class="b" points="{pts}"/>')
        t = tern(p)
        body.append(f'<circle class="ink" cx="{t[0]:.1f}" cy="{t[1]:.1f}" r="2.5"/>')
    for i, (cls, txt) in enumerate((("hd", "Every curve is a circle"), ("hd", "of Fisher-Rao radius 0.35"),
                                    ("tiny", ""), ("tiny", "Same radius, very different size"),
                                    ("tiny", "in probability. Near an edge or"), ("tiny", "corner, a small change in a rare"),
                                    ("tiny", "class costs a lot: the ball is"), ("tiny", "squashed against it and cut off."))):
        if txt:
            body.append(f'<text class="{cls}" x="470" y="{110 + 17 * i}">{txt}</text>')
    write("fisher-balls.svg", svg(W, H, "Fisher-Rao balls on the three-class simplex",
                                  "Circles of equal Fisher-Rao radius around five predictions; they shrink and are squashed near the boundary.", body))

    # empirical vs true Fisher
    W, H = 640, 330
    ax = Axes(70, 30, 520, 240, (0.0, 1.0), (0.01, 100.0), logy=True)
    body = [ax.frame([0, 0.25, 0.5, 0.75, 1.0], [0.01, 0.1, 1, 10, 100],
                     "probability the model gives to the observed label, p_y",
                     "empirical / true Fisher speed", yfmt="{:g}")]
    pys = np.linspace(0.005, 0.995, 300)
    body.append(ax.path(pys, [math.sqrt((1 - p) / p) for p in pys], "b"))
    body.append(f'<line class="k" x1="{ax.x0}" x2="{ax.x0 + ax.w}" y1="{ax.Y(1):.1f}" y2="{ax.Y(1):.1f}"/>')
    body.append(ax.text(0.5, 1.0, "equal at p_y = 1/2", "tiny", "start", 6, -6))
    body.append(ax.text(0.66, 0.06, "confident and right: the empirical", "tiny", "middle", 0, 0))
    body.append(ax.text(0.66, 0.042, "Fisher under-counts", "tiny", "middle", 0, 0))
    body.append(ax.text(0.14, 25, "confident and wrong: it", "tiny", "middle", 0, 0))
    body.append(ax.text(0.14, 17, "over-counts", "tiny", "middle", 0, 0))
    write("empirical-fisher.svg", svg(W, H, "Empirical against true Fisher speed",
                                     "The ratio sqrt((1-p_y)/p_y) as a function of the probability of the observed label.", body))


if __name__ == "__main__":
    check_kl_hessian()
    p0, p1 = check_sphere()
    check_jensen()
    check_12b_vs_endpoint()
    check_empirical_fisher()
    if "--figures" in sys.argv:
        figures()
