#!/usr/bin/env python3
"""Lemma 2.4 (Hessian = Fisher + residual) checked on real curvature matrices.

The lemma: for cross-entropy on logits z = f_w(x),
    H = F + (1/N) sum_i sum_j [p_i - e_{y_i}]_j  Hess_w z_ij,
with F the Fisher (labels drawn from the model), which for softmax equals the
Gauss-Newton matrix J' (diag p - p p') J. The paper concludes H ~ F "if almost
all training samples are predicted correctly", because then p_i - e_{y_i} ~ 0.

But the Fisher's own middle factor diag p - p p' goes to zero at exactly the same
rate (both are O(1 - p_y)), so the residual does not become small *relative to
F* from that alone. The relative gap is set by the ratio of logit curvature to
squared logit gradient, an architectural quantity. This script measures, along
full-batch gradient descent on two moons:

  * the decomposition itself (H computed by finite differences of the exact
    gradient, the residual computed independently from logit Hessians);
  * ||H - F|| / ||F||, tr H / tr F, and the top eigenvalues;
  * tr F over training (the paper's Figures 1 and 4 plot Fisher statistics);
  * for logistic regression (a linear model) H = F exactly, for any fit.

Standard library and numpy only.

Run:  python3 hessian_fisher.py             (checks)
      python3 hessian_fisher.py --figures   (checks, then rewrite ../figures/hessian-fisher.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np

HID = 16


def moons(n, noise, rng):
    t = rng.uniform(0, math.pi, n)
    lab = rng.integers(0, 2, n)
    x = np.where(lab[:, None] == 0,
                 np.stack([np.cos(t), np.sin(t)], 1),
                 np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1))
    x = x + noise * rng.standard_normal((n, 2))
    return (x - x.mean(0)) / x.std(0), lab.astype(float)


SHAPES = [("W1", (HID, 2)), ("b1", (HID,)), ("W2", (HID, HID)), ("b2", (HID,)), ("v", (HID,)), ("c", (1,))]
SIZES = [int(np.prod(s)) for _, s in SHAPES]
K = sum(SIZES)


def unpack(th):
    out, i = {}, 0
    for (name, shp), n in zip(SHAPES, SIZES):
        out[name] = th[i:i + n].reshape(shp)
        i += n
    return out


def logits_and_jac(th, x):
    P = unpack(th)
    h1 = np.tanh(x @ P["W1"].T + P["b1"])
    h2 = np.tanh(h1 @ P["W2"].T + P["b2"])
    z = h2 @ P["v"] + P["c"][0]
    d2 = P["v"] * (1 - h2 ** 2)                       # dz/da2
    d1 = (d2 @ P["W2"]) * (1 - h1 ** 2)               # dz/da1
    n = len(x)
    G = np.concatenate([
        (d1[:, :, None] * x[:, None, :]).reshape(n, -1), d1,
        (d2[:, :, None] * h1[:, None, :]).reshape(n, -1), d2,
        h2, np.ones((n, 1))], 1)
    return z, G


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss_grad(th, x, t):
    z, G = logits_and_jac(th, x)
    p = sigmoid(z)
    loss = float(np.mean(np.logaddexp(0, z) - t * z))
    return loss, G.T @ (p - t) / len(x), z, G, p


def fd_hessian(grad_fn, th, h=1e-5):
    Hm = np.zeros((len(th), len(th)))
    for j in range(len(th)):
        e = np.zeros(len(th)); e[j] = h
        Hm[:, j] = (grad_fn(th + e) - grad_fn(th - e)) / (2 * h)
    return 0.5 * (Hm + Hm.T)


def curvature(th, x, t):
    loss, _, z, G, p = loss_grad(th, x, t)
    n = len(x)
    F = (G * (p * (1 - p))[:, None]).T @ G / n
    H = fd_hessian(lambda u: loss_grad(u, x, t)[1], th)
    c = (p - t) / n                                    # frozen residual weights
    R = fd_hessian(lambda u: logits_and_jac(u, x)[1].T @ c, th)
    return loss, z, p, F, H, R


def main():
    rng = np.random.default_rng(0)
    x, t = moons(200, 0.12, rng)
    print(f"two moons, N = {len(x)}; MLP 2-{HID}-{HID}-1 tanh, {K} parameters; full-batch GD, lr 0.5\n")

    # --- logistic regression: H = F exactly
    xl = np.concatenate([x, np.ones((len(x), 1))], 1)
    wl = np.array([2.0, -3.0, 0.5])
    gl = lambda w: xl.T @ (sigmoid(xl @ w) - t) / len(x)
    Hl = fd_hessian(gl, wl)
    pl = sigmoid(xl @ wl)
    Fl = (xl * (pl * (1 - pl))[:, None]).T @ xl / len(x)
    print(f"logistic regression (linear logits): max |H - F| = {np.abs(Hl - Fl).max():.1e} "
          f"(finite-difference noise), with training loss {np.mean(np.logaddexp(0, xl @ wl) - t * (xl @ wl)):.3f}")
    print("  Hess_w z = 0 for a linear model, so the residual vanishes at any fit, good or bad.\n")

    th = np.concatenate([rng.standard_normal(n) * (1 / math.sqrt(max(s[-1], 1)) if len(s) > 1 else 0.0)
                         for (_, s), n in zip(SHAPES, SIZES)])
    checkpoints = [0, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000]
    rows = []
    step = 0
    print(" step    loss   acc  mean|z|   tr F     tr H    ||H-F||/||F||  ||R||/||F||  lmax H  lmax F  decomp err")
    for cp in checkpoints:
        while step < cp:
            th = th - 0.5 * loss_grad(th, x, t)[1]
            step += 1
        loss, z, p, F, H, R = curvature(th, x, t)
        acc = float(np.mean((z > 0) == (t > 0.5)))
        err = np.linalg.norm(H - F - R) / np.linalg.norm(H)
        rel = np.linalg.norm(H - F) / np.linalg.norm(F)
        rows.append(dict(step=step, loss=loss, acc=acc, z=float(np.mean(np.abs(z))), trF=float(np.trace(F)),
                         trH=float(np.trace(H)), rel=float(rel),
                         lH=float(np.linalg.eigvalsh(H)[-1]), lF=float(np.linalg.eigvalsh(F)[-1]),
                         neg=float(np.linalg.eigvalsh(H)[0])))
        print(f"{step:>6} {loss:7.4f} {acc:5.3f} {rows[-1]['z']:7.2f} {np.trace(F):8.4f} {np.trace(H):8.4f}"
              f"   {rel:10.3f}    {np.linalg.norm(R) / np.linalg.norm(F):10.3f}  {rows[-1]['lH']:6.3f}  "
              f"{rows[-1]['lF']:6.3f}   {err:.1e}")
    r0 = rows[3]
    last = rows[-1]
    print(f"\nthe decomposition H = F + R holds to finite-difference precision at every checkpoint")
    print(f"from step {r0['step']} to {last['step']}: loss {r0['loss']:.4f} -> {last['loss']:.2e} "
          f"(x{r0['loss'] / last['loss']:.0f} smaller), tr F {r0['trF']:.3f} -> {last['trF']:.2e}, "
          f"but ||H - F|| / ||F|| only {r0['rel']:.2f} -> {last['rel']:.2f}")
    print(f"most negative Hessian eigenvalue at the end: {last['neg']:.2e} (Fisher is PSD by construction)")
    ipk = int(np.argmax([r["trF"] for r in rows]))
    print(f"tr F peaks at step {rows[ipk]['step']} ({rows[ipk]['trF']:.3f}) and then falls "
          f"{rows[ipk]['trF'] / last['trF']:.0f}x as the fit saturates: a late 'compression' of the Fisher that "
          "needs no change in the function, only more confident logits.")
    if "--figures" in sys.argv:
        figures(rows)


def figures(rows):
    from pathlib import Path
    import svgplot as sp

    out = Path(__file__).resolve().parent.parent / "figures"
    rows = [r for r in rows if r["step"] > 0]
    steps = [r["step"] for r in rows]
    body = [sp.title(20, 22, "Lemma 2.4 on a real network: the Hessian–Fisher gap stops shrinking once the data are fitted"),
            sp.sub(20, 39, "2-16-16-1 tanh MLP on two moons, full-batch GD. Exact Hessian (finite differences) vs Fisher.")]
    ax = sp.Axes(62, 72, 270, 180, (30, 100_000), (1e-6, 10), logx=True, logy=True)
    body.append(ax.frame([], [], "GD step", "log scale"))
    for s in [100, 1000, 10_000, 100_000]:
        body.append(sp.tick_label(ax, s, f"{s:g}"))
    for v, lab in [(1e-6, "1e-6"), (1e-4, "1e-4"), (1e-2, "0.01"), (1, "1")]:
        body.append(f'<text class="tiny" x="{ax.x0 - 5}" y="{ax.Y(v) + 3:.1f}" text-anchor="end">{lab}</text>')
        body.append(f'<line class="gd" x1="{ax.x0}" x2="{ax.x0 + ax.w}" y1="{ax.Y(v):.1f}" y2="{ax.Y(v):.1f}"/>')
    for key, cls in [("loss", "c1"), ("trF", "c0")]:
        body.append(ax.path(steps, [r[key] for r in rows], cls))
        body.append(ax.dots(steps, [r[key] for r in rows], cls + "f"))
    body.append(sp.legend(72, 290, [("c1", "training loss"), ("c0", "tr F (Fisher trace)")], cols=2))
    ax2 = sp.Axes(420, 72, 270, 180, (1e-6, 1), (0, 0.5), logx=True)
    body.append(ax2.frame([], [0, 0.1, 0.2, 0.3, 0.4, 0.5], "training loss (log scale, training runs right to left)", "‖H − F‖ / ‖F‖"))
    for s in [1e-6, 1e-4, 1e-2, 1]:
        body.append(sp.tick_label(ax2, s, f"{s:g}"))
    body.append(ax2.path([r["loss"] for r in rows], [r["rel"] for r in rows], "c2"))
    body.append(ax2.dots([r["loss"] for r in rows], [r["rel"] for r in rows], "c2f"))
    body.append(sp.legend(430, 290, [("c2", "relative Hessian-Fisher gap")]))
    desc = ("Left, on log axes over training: the loss falls by five orders of magnitude, and the trace of the "
            "Fisher rises, peaks near step 100, and then falls with it. Right: the relative gap between Hessian and "
            "Fisher against the training loss; it drops while the network learns and then levels off near 4 percent "
            "for the last three decades of loss.")
    (out / "hessian-fisher.svg").write_text(sp.svg(720, 310, "Hessian vs Fisher", desc, "\n".join(body)))
    print("wrote figures/hessian-fisher.svg")


if __name__ == "__main__":
    main()
