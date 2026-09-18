#!/usr/bin/env python3
"""The Lorentz-model primitives HFM is built on, and two checks on its claims.

Implements Section 3.1 of Li et al., then uses it to test the geometric
argument the paper rests on: that hyperbolic space has room to keep transport
trajectories apart.

The second check is the interesting one. The advantage is real at the boundary
and gone at the origin, which is why the diameter-based stopping rule of
Section 3.4 is structurally necessary rather than an efficiency tweak.

Run:  python3 lorentz.py
"""
from __future__ import annotations

import math

import numpy as np


# ------------------------------------------- Lorentz model L^{n,kappa}
# A point is x = (x0, x~) on the upper sheet of <x,x>_L = -1/kappa,
# with <x,y>_L = -x0 y0 + <x~,y~>_E.
def linner(x, y):
    return -x[..., 0]*y[..., 0] + (x[..., 1:]*y[..., 1:]).sum(-1)


def to_manifold(xt, k):
    """Lift a space-like vector onto the hyperboloid."""
    return np.concatenate([np.sqrt((xt**2).sum(-1) + 1.0/k)[..., None], xt], -1)


def dist(x, y, k):
    """Geodesic distance: d_L(x,y) = arcosh(-kappa <x,y>_L)/sqrt(kappa)."""
    return np.arccosh(np.clip(-k*linner(x, y), 1.0, None))/np.sqrt(k)


def proj_tangent(z, v, k):
    """Their Pi_{T_z L}(x) = x + kappa <z,x>_L z."""
    return v + k*linner(z, v)[..., None]*z


def expmap(z, u, k):
    """exp_z(u) for u in T_z L."""
    nu = np.sqrt(np.clip(k*linner(u, u), 1e-30, None))
    return np.cosh(nu)[..., None]*z + np.sinh(nu)[..., None]*u/nu[..., None]


def logmap(z, x, k):
    """Their Eq. 2."""
    a = linner(z, x)
    num = np.arccosh(np.clip(-k*a, 1.0, None))
    den = np.sqrt(np.clip(a**2 - 1.0/k**2, 1e-30, None))
    return (num/den)[..., None]*proj_tangent(z, x, k)


def geodesic(x0, x1, t, k):
    """Their ground-truth path, Algorithm 1 line 5: x_t = exp_{x0}(t log_{x0}(x1))."""
    return expmap(x0, t*logmap(x0, x1, k), k)


# --------------------------------------------------------------- checks
def check_primitives(k=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = to_manifold(rng.standard_normal((5, 3)), k)
    y = to_manifold(rng.standard_normal((5, 3)), k)
    v = logmap(x, y, k)
    xt = geodesic(x, y, 0.37, k)
    print("Section 3.1 primitives")
    for name, err in [
        ("points satisfy <x,x>_L = -1/kappa", np.abs(linner(x, x) + 1/k).max()),
        ("log_x(y) lies in T_x L          ", np.abs(linner(x, v)).max()),
        ("exp_x(log_x(y)) == y            ", np.abs(expmap(x, v, k) - y).max()),
        ("||log_x(y)||_L == d_L(x,y)      ", np.abs(np.sqrt(linner(v, v)) - dist(x, y, k)).max()),
        ("geodesic point stays on manifold", np.abs(linner(xt, xt) + 1/k).max()),
        ("constant speed: d(x,x_t)/d(x,y)=t", np.abs(dist(x, xt, k)/dist(x, y, k) - 0.37).max()),
    ]:
        print(f"  {name}  max err {err:.2e}")
    print()


def check_capacity():
    """How much room is there at radius R? The paper's volume-growth argument."""
    print("Points at mutual distance >= 1 on a sphere of radius R (2-D slice)")
    print(f"{'R':>4} {'Euclidean':>12} {'hyperbolic':>13} {'ratio':>9}")
    for R in (1, 2, 3, 5, 8):
        e, h = 2*math.pi*R, 2*math.pi*math.sinh(R)
        print(f"{R:>4} {e:>12.1f} {h:>13.1f} {h/e:>8.1f}x")
    print()


def check_separation(k=1.0, ang=0.3, target_r=0.3):
    """Absolute separation between two class trajectories along the flow.

    Sources at radius R, targets near the origin, same angular separation
    throughout, so the two geometries are compared like for like.
    """
    def pt(rad, a):
        return to_manifold(np.array([math.cos(a), math.sin(a)])*math.sinh(rad), k)

    ts = np.array([0.0, 0.5, 0.8, 0.95, 1.0])
    print("Separation between two class trajectories (absolute geodesic distance)")
    print(f"{'R':>4} | {'geometry':>10} | " + " ".join(f"{f't={t:g}':>7}" for t in ts))
    print("-"*66)
    for R in (2.0, 4.0):
        srcH, tgtH = [pt(R, 0.0), pt(R, ang)], [pt(target_r, 0.0), pt(target_r, ang)]
        trH = [np.stack([geodesic(s, g, t, k) for t in ts]) for s, g in zip(srcH, tgtH)]
        dH = np.array([dist(trH[0][i], trH[1][i], k) for i in range(len(ts))])
        srcE = [np.array([R*math.cos(a), R*math.sin(a)]) for a in (0.0, ang)]
        tgtE = [np.array([target_r*math.cos(a), target_r*math.sin(a)]) for a in (0.0, ang)]
        trE = [np.stack([(1-t)*s + t*g for t in ts]) for s, g in zip(srcE, tgtE)]
        dE = np.linalg.norm(trE[0]-trE[1], axis=1)
        print(f"{R:>4.0f} | {'Euclidean':>10} | " + " ".join(f"{v:>7.3f}" for v in dE))
        print(f"{'':>4} | {'hyperbolic':>10} | " + " ".join(f"{v:>7.3f}" for v in dH))
        print(f"{'':>4} | {'ratio H/E':>10} | " + " ".join(f"{h/e:>7.2f}" for h, e in zip(dH, dE)))
        print("-"*66)
    print("  -> the hyperbolic advantage is large at the source, ~2x at the midpoint,")
    print("     and gone by the target. The corridors merge near the crowded origin,")
    print("     which is exactly what the diameter-based stopping rule works around.")


if __name__ == "__main__":
    check_primitives()
    check_capacity()
    check_separation()
