#!/usr/bin/env python3
"""A transfer problem small enough to solve exactly, to test Remarks 1, 3 and 4 of Gao & Chaudhari.

A task is a 1-D binary classification problem: each class is a Gaussian, x | (class c) ~ N(m_c, s_c^2),
two classes with equal weight. The model is logistic regression, p_w(y=1 | x) = sigma(w1 x + w0), a convex
model with two weights. Expectations over x are done by Gauss-Hermite quadrature, so every number is
deterministic and there is no sampling noise.

Two ways to interpolate from a source task to a target task (Section 3):
  displacement  each class is moved by optimal transport (for Gaussians in 1-D: means and standard
                deviations interpolate linearly), the labels travel with their points. This is (7)-(8)
                for a coupling that pairs each class with its counterpart.
  mixture       (1 - tau) p_source + tau p_target, the "uncoupled" interpolation (9).

Two ways to follow the tasks with the weights:
  minimiser path  w*(tau) = argmin of the cross-entropy on the tau-task. This is the limit of (10) when the weights
                  relax infinitely fast ("quasi-static").
  gradient flow   dw/dtau = -kappa grad L_tau(w), started at w*(0). kappa is the number of SGD steps
                  times the learning rate: it is the ratio of the learning speed to the speed at which the
                  task moves. Eq. (10) with dtau = learning rate is kappa = 1.

Three lengths of a weight path w(tau) (each x-average taken under the task at time tau):
  A = int sqrt( E_x [ 2KL ] )         the Riemannian length of eq. (2)-(4): average inside the root
  B = E_x int sqrt( 2KL )             Definition 2, eq. (11): average outside the root
  TV = E int |dl/dtau| dtau           the total variation of the per-sample loss, labels held fixed;
                                      this is what (21)-(22) turn the length into

Checked here:
  1. Remark 1 (mixture is longer than displacement): a table over shifts, with all three lengths.
  2. The displacement path for a translated task is a straight line in weight space and A has a closed form.
  3. Remark 3 (asymmetry): the minimiser path is exactly reversible, so forward and backward lengths agree;
     the gradient flow is not reversible, and forward and backward lengths differ.
  4. The length of the gradient flow depends on kappa, and is small when the weights cannot keep up.

Standard library and numpy only.

Run:  python3 transfer_toy.py             (prints every number quoted in the notes)
      python3 transfer_toy.py --figures   (also rewrites ../figures/remark1.svg, kappa.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np

_gx, _gw = np.polynomial.hermite_e.hermegauss(40)  # nodes for weight exp(-x^2/2)
_gw = _gw / _gw.sum()


class Task:
    """A list of Gaussian clusters (mean, std, label, weight)."""

    def __init__(self, clusters):
        self.cl = list(clusters)

    def nodes(self):
        xs, ys, ws = [], [], []
        for m, s, y, pr in self.cl:
            xs.append(m + s * _gx)
            ys.append(np.full_like(_gx, y))
            ws.append(pr * _gw)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def displacement(src, tgt, t):
    return Task([((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1], a[2], a[3]) for a, b in zip(src.cl, tgt.cl)])


def mixture(src, tgt, t):
    return Task([(m, s, y, (1 - t) * pr) for m, s, y, pr in src.cl] + [(m, s, y, t * pr) for m, s, y, pr in tgt.cl])


def shifted(task, s):
    return Task([(m + s, sd, y, pr) for m, sd, y, pr in task.cl])


SOURCE = Task([(-2.0, 1.0, 0, 0.5), (2.0, 1.0, 1, 0.5)])
SHARP = Task([(-1.0, 0.5, 0, 0.5), (1.0, 0.5, 1, 0.5)])  # a second target: closer classes, narrower


def sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def loss_grad_hess(w, X, Y, W):
    phi = np.stack([X, np.ones_like(X)], axis=1)
    p = sig(phi @ w)
    eps = 1e-300
    L = -np.sum(W * (Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps)))
    g = phi.T @ (W * (p - Y))
    H = (phi * (W * p * (1 - p))[:, None]).T @ phi
    return L, g, H


def minimise(task, w0):
    X, Y, W = task.nodes()
    w = np.array(w0, float)
    for _ in range(200):
        L, g, H = loss_grad_hess(w, X, Y, W)
        if np.linalg.norm(g) < 1e-12:
            break
        step = np.linalg.solve(H + 1e-13 * np.eye(2), g)
        a = 1.0
        while a > 1e-8 and loss_grad_hess(w - a * step, X, Y, W)[0] > L + 1e-14:
            a /= 2
        w = w - a * step
    return w


def minimiser_path(src, tgt, interp, n=600):
    ts = np.linspace(0, 1, n + 1)
    w = minimise(src, np.zeros(2))
    ws = []
    for t in ts:
        w = minimise(interp(src, tgt, t), w)
        ws.append(w.copy())
    return ts, np.array(ws)


def lengths(src, tgt, interp, ts, ws):
    """A, B, TV of a discretised weight path (midpoint rule in tau)."""
    A = B = TV = 0.0
    for k in range(len(ts) - 1):
        t = 0.5 * (ts[k] + ts[k + 1])
        wm = 0.5 * (ws[k] + ws[k + 1])
        dw = ws[k + 1] - ws[k]
        X, Y, Wt = interp(src, tgt, t).nodes()
        phi = np.stack([X, np.ones_like(X)], axis=1)
        p = sig(phi @ wm)
        F = (phi * (Wt * p * (1 - p))[:, None]).T @ phi
        A += math.sqrt(max(float(dw @ F @ dw), 0.0))
        B += float(np.sum(Wt * np.sqrt(p * (1 - p)) * np.abs(phi @ dw)))
        TV += float(np.sum(Wt * np.abs((Y - p) * (phi @ dw))))
    return A, B, TV


def flow(src, tgt, interp, kappa, n=None):
    """RK4 for dw/dtau = -kappa grad L_tau(w), w(0) = minimiser on the source."""
    n = n or int(max(800, 12 * kappa))
    dt = 1.0 / n
    w = minimise(src, np.zeros(2))
    ts = np.linspace(0, 1, n + 1)
    ws = [w.copy()]

    def grad(w, t):
        X, Y, W = interp(src, tgt, t).nodes()
        return loss_grad_hess(w, X, Y, W)[1]

    for k in range(n):
        t = ts[k]
        k1 = -kappa * grad(w, t)
        k2 = -kappa * grad(w + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = -kappa * grad(w + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = -kappa * grad(w + dt * k3, t + dt)
        w = w + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ws.append(w.copy())
    return ts, np.array(ws)


def excess_loss(src, tgt, interp, ts, ws, every):
    """L_tau(w(tau)) - min_w L_tau(w), sampled every `every` steps."""
    out = []
    for k in range(0, len(ts), every):
        task = interp(src, tgt, ts[k])
        X, Y, W = task.nodes()
        out.append(loss_grad_hess(ws[k], X, Y, W)[0] - loss_grad_hess(minimise(task, ws[k]), X, Y, W)[0])
    return np.array(out)


def subsample(ts, ws, m=401):
    idx = np.linspace(0, len(ts) - 1, m).astype(int)
    return ts[idx], ws[idx]


# ------------------------------------------------------------------------------ 1. Remark 1
SHIFTS = (0.5, 1, 2, 3, 4, 6, 8)


def check_remark1():
    print("1. Remark 1: is the mixture (uncoupled) path longer than the displacement (coupled) path?")
    print("   source: classes at -2 and +2, sd 1. target: the same task translated by s. Minimiser paths.")
    print("   lengths A (average inside the root), B (outside, eq. 11) and TV (loss variation):")
    print("      s |  disp: A      B     TV  |  mix: A      B     TV  |  mix/disp: A     B     TV")
    rows = []
    for s in SHIFTS:
        tgt = shifted(SOURCE, s)
        d = lengths(SOURCE, tgt, displacement, *minimiser_path(SOURCE, tgt, displacement))
        m = lengths(SOURCE, tgt, mixture, *minimiser_path(SOURCE, tgt, mixture))
        rows.append((s, d, m))
        print(f"   {s:4} |  {d[0]:6.3f} {d[1]:6.3f} {d[2]:6.3f} |  {m[0]:6.3f} {m[1]:6.3f} {m[2]:6.3f} |"
              f"  {m[0] / d[0]:6.2f} {m[1] / d[1]:6.2f} {m[2] / d[2]:6.2f}")
    print("   B / A along the path:  s  displacement  mixture     smallest slope of the mixture's minimiser")
    for s, d, m in rows:
        slope = float(min(w[0] for w in minimiser_path(SOURCE, shifted(SOURCE, s), mixture, n=300)[1]))
        print(f"                         {s:4}   {d[1] / d[0]:8.3f}   {m[1] / m[0]:8.3f}      {slope:8.3f}   (displacement keeps slope 4.000)")
    print()
    return rows


def check_translation_closed_form():
    print("2. Displacement path of a translated task: a straight line in weight space, and A in closed form")
    s = 4.0
    tgt = shifted(SOURCE, s)
    ts, ws = minimiser_path(SOURCE, tgt, displacement)
    print(f"   w*(0) = ({ws[0][0]:.4f}, {ws[0][1]:.4f}),  w*(1) = ({ws[-1][0]:.4f}, {ws[-1][1]:.4f})"
          f"   (slope stays {ws[0][0]:.4f}; bias moves by -slope * s = {-ws[0][0] * s:.4f})")
    dev = max(abs(ws[k][0] - ws[0][0]) + abs(ws[k][1] - (ws[0][1] - ws[0][0] * ts[k] * s)) for k in range(len(ts)))
    X, Y, W = SOURCE.nodes()
    p = sig(ws[0][0] * X + ws[0][1])
    Ep = float(np.sum(W * p * (1 - p)))
    closed = abs(ws[0][0]) * s * math.sqrt(Ep)
    A = lengths(SOURCE, tgt, displacement, ts, ws)[0]
    print(f"   largest deviation from the straight line: {dev:.1e}")
    print(f"   A = |slope| * s * sqrt(E[p(1-p)]) = {ws[0][0]:.4f} * {s} * sqrt({Ep:.5f}) = {closed:.4f};  numerically {A:.4f}")
    print(f"   so A grows linearly with the translation: A / s = {A / s:.4f}")
    print()


# ------------------------------------------------------------------------------ 3. symmetry
def check_symmetry():
    print("3. Remark 3: forward and backward, quasi-static and with finite kappa")
    print("   pair: SOURCE (classes at -2, +2, sd 1) and SHARP (classes at -1, +1, sd 0.5); displacement interpolation")
    print("   minimiser path:")
    for name, f in (("displacement", displacement), ("mixture", mixture)):
        fw = lengths(SOURCE, SHARP, f, *minimiser_path(SOURCE, SHARP, f))
        bw = lengths(SHARP, SOURCE, f, *minimiser_path(SHARP, SOURCE, f))
        print(f"     {name:12s} forward  A={fw[0]:.5f} B={fw[1]:.5f} TV={fw[2]:.5f}   backward  A={bw[0]:.5f} B={bw[1]:.5f} TV={bw[2]:.5f}")
    print("   gradient flow, displacement interpolation:")
    print("     kappa |  forward A     B   final excess loss |  backward A     B   final excess loss | A fwd / bwd")
    out = []
    for kappa in (10, 30, 100, 300, 1000, 3000):
        r = []
        for a, b in ((SOURCE, SHARP), (SHARP, SOURCE)):
            ts, ws = flow(a, b, displacement, kappa)
            A, B, TV = lengths(a, b, displacement, *subsample(ts, ws))
            ex = excess_loss(a, b, displacement, ts, ws, every=len(ts) - 1)[-1]
            r.append((A, B, ex))
        out.append((kappa, r))
        print(f"     {kappa:5} |  {r[0][0]:8.4f} {r[0][1]:8.4f}   {r[0][2]:8.4f}     |  {r[1][0]:8.4f} {r[1][1]:8.4f}   {r[1][2]:8.4f}     |  {r[0][0] / r[1][0]:6.2f}")
    print()
    return out


# ------------------------------------------------------------------------------ 4. time-scale
def check_kappa():
    print("4. The length of the gradient flow depends on kappa (translated task, s = 4, displacement interpolation)")
    tgt = shifted(SOURCE, 4.0)
    ts, ws = minimiser_path(SOURCE, tgt, displacement)
    qs = lengths(SOURCE, tgt, displacement, ts, ws)
    print(f"   quasi-static minimiser path: A={qs[0]:.3f} B={qs[1]:.3f} TV={qs[2]:.3f}")
    print("   kappa |     A       B      TV   | w(1)              | max excess loss   excess loss at tau=1")
    print("   (quasi-static w(1) = (4.00, -16.00))")
    rows = []
    for kappa in (1, 3, 10, 30, 100, 300, 1000, 3000):
        ts, ws = flow(SOURCE, tgt, displacement, kappa)
        A, B, TV = lengths(SOURCE, tgt, displacement, *subsample(ts, ws))
        ex = excess_loss(SOURCE, tgt, displacement, ts, ws, every=max(1, len(ts) // 40))
        rows.append((kappa, A, B, TV, ex.max(), ex[-1]))
        print(f"   {kappa:5} | {A:6.3f}  {B:6.3f}  {TV:6.3f}  | ({ws[-1][0]:5.2f}, {ws[-1][1]:6.2f})  |  {ex.max():7.3f}           {ex[-1]:7.3f}")
    # solver convergence: halve the step at kappa = 30
    t1, w1 = flow(SOURCE, tgt, displacement, 30)
    t2, w2 = flow(SOURCE, tgt, displacement, 30, n=2 * (len(t1) - 1))
    a1 = lengths(SOURCE, tgt, displacement, *subsample(t1, w1))
    a2 = lengths(SOURCE, tgt, displacement, *subsample(t2, w2))
    print(f"   solver check at kappa = 30, step halved: A {a1[0]:.4f} -> {a2[0]:.4f},  B {a1[1]:.4f} -> {a2[1]:.4f}")
    print()
    return qs, rows


# ------------------------------------------------------------------------------ figures
def figures(remark1_rows, kappa_out):
    from svgkit import Axes, legend, svg, write

    # Remark 1 figure: A and B against the shift
    W, H = 640, 300
    body = []
    for col, (idx, lab) in enumerate(((0, "A: average inside the root (eq. 4)"), (1, "B: average outside (eq. 11, Definition 2)"))):
        ax = Axes(50 + col * 300, 40, 240, 190, (0, 8), (0, 4.5 if idx == 0 else 3.0))
        body.append(ax.frame([0, 2, 4, 6, 8], [0, 1, 2, 3, 4] if idx == 0 else [0, 1, 2, 3], "translation s of the target", ""))
        body.append(f'<text class="hd" x="{ax.x0}" y="{ax.y0 - 14}" style="font-size:11.5px">{lab}</text>')
        ss = [r[0] for r in remark1_rows]
        body.append(ax.path(ss, [r[1][idx] for r in remark1_rows], "b"))
        body.append(ax.path(ss, [r[2][idx] for r in remark1_rows], "o"))
        for s, d, m in remark1_rows:
            body.append(ax.dot(s, d[idx], "bf", 2.5))
            body.append(ax.dot(s, m[idx], "of", 2.5))
    body.append(legend(60, 276, [("b", "displacement (coupled)")], 16))
    body.append(legend(260, 276, [("o", "mixture (uncoupled)")], 16))
    write("remark1.svg", svg(W, H, "Length of the minimiser path against the shift",
                             "Displacement path length grows linearly in the shift; the mixture path is longer by definition B but saturates and is shorter by A for large shifts.", body))

    # kappa figure
    qs, rows = kappa_out
    W, H = 640, 300
    ax = Axes(60, 30, 520, 200, (1, 3000), (0, 3.5), logx=True)
    body = [ax.frame([1, 10, 100, 1000], [0, 1, 2, 3], "kappa  (learning speed / task speed)", "path length A, and worst excess loss (same axis)")]
    ks = [r[0] for r in rows]
    body.append(ax.path(ks, [r[1] for r in rows], "b"))
    body.append(ax.path(ks, [r[4] for r in rows], "r"))
    for r in rows:
        body.append(ax.dot(r[0], r[1], "bf", 2.5))
        body.append(ax.dot(r[0], r[4], "rf", 2.5))
    body.append(f'<line class="k" x1="{ax.x0}" x2="{ax.x0 + ax.w}" y1="{ax.Y(qs[0]):.1f}" y2="{ax.Y(qs[0]):.1f}"/>')
    body.append(ax.text(2800, qs[0], "quasi-static (kappa = infinity): A = %.2f" % qs[0], "tiny", "end", 0, -5))
    body.append(legend(330, 150, [("b", "length A of the SGD path"), ("r", "worst excess loss along the way")], 16))
    write("kappa.svg", svg(W, H, "Path length against learning speed",
                           "For a small kappa the weights barely move: the path is short and the excess loss large.", body))


if __name__ == "__main__":
    r1 = check_remark1()
    check_translation_closed_form()
    check_symmetry()
    ko = check_kappa()
    if "--figures" in sys.argv:
        figures(r1, ko)
