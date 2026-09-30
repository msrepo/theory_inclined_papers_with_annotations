#!/usr/bin/env python3
"""Section 4 and Appendix E of Tahir, Ganguli & Rotskoff: the out-of-RKHS part of a ReLU teacher (Figure 3d).

The paper measures how much of the target function f_t = (1/m*) sum_i c_i relu(w_i . x) lies outside the
feature space of the pretrained network, ||P_perp f_t||^2 / ||f_t||^2, as a fraction mu of the teacher's
neurons is ablated from the source. For x ~ N(0, I_d) and unit weight vectors the kernel is the arc-cosine
kernel of Cho & Saul (eq. 167),

    <relu(w.x) relu(v.x)> = (sqrt(1 - u^2) + u (pi - arccos u)) / (2 pi),   u = cos angle(w, v),

so every projection is a linear solve with a Gram matrix (eqs. 172-173).

Two idealisations of "the pretrained network's feature space" are computed, because the paper does not say
which one its panel uses:
  (a) the span of the (1 - mu) m* source-teacher neurons only (the paper's claim that the trained kernel
      equals the source kernel, Fig. 3a-b);
  (b) that span plus the m - (1 - mu) m* untouched random hidden neurons of the m = 1000 student.
Figure 3d's axis tops out near 0.045; neither idealisation reproduces that scale (see the notes).

Standard library and numpy only.

Run:  python3 relu_projection.py
"""
from __future__ import annotations

import numpy as np


def arccos_kernel(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    u = np.clip(A @ B.T, -1, 1)
    return (np.sqrt(1 - u ** 2) + u * (np.pi - np.arccos(u))) / (2 * np.pi)


def main(d=100, mstar=100, m=1000, reps=10, seed=5):
    rng = np.random.default_rng(seed)
    print("Fraction of ||f_t||^2 outside the span of the source features (d = %d, m* = %d, m = %d, %d draws)" % (d, mstar, m, reps))
    print("  mu    (a) source neurons only     (b) plus untouched random neurons")
    for mu in [0.1, 0.3, 0.5, 0.7, 0.9]:
        fa, fb = [], []
        for _ in range(reps):
            Wt = rng.standard_normal((mstar, d))
            c = rng.standard_normal(mstar)
            k = int(round(mu * mstar))
            keep = Wt[k:]
            tot = c @ arccos_kernel(Wt, Wt) @ c
            for basis, acc in [(keep, fa), (np.vstack([keep, rng.standard_normal((m - len(keep), d))]), fb)]:
                Kb = arccos_kernel(basis, basis)
                Kx = arccos_kernel(basis, Wt)
                v = Kx @ c
                proj = v @ np.linalg.solve(Kb + 1e-9 * np.eye(len(basis)), v)
                acc.append((tot - proj) / tot)
        print("  %.1f   %.4f                      %.4f" % (mu, np.mean(fa), np.mean(fb)))
    print("  Both rise with mu and are convex, like Figure 3d; the paper's axis reads 0 to 0.045.")


if __name__ == "__main__":
    main()
