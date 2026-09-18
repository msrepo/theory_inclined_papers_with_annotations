#!/usr/bin/env python3
"""The arithmetic behind STEEL: why a finite hypothesis class certifies where a
continuous one cannot.

Rezk et al. compare two risk certificates of the form  R <= r + complexity:

  Eq. 3   finite class    complexity = C sqrt( log(|Theta|/eps) / 2n )
  Eq. 7   continuous      complexity = C sqrt( (d + 2 log d + log(1/eps)) / 2n )

Eq. 7 is the best case of the Lotfi et al. (2024) quantisation bound, assuming
one bit per parameter, so it flatters the competing method.

Eq. 3 is not exotic: it is Hoeffding plus a union bound over |Theta|, which is
why it holds uniformly over the class and lets you pick the best candidate after
looking at the support set. That uniformity is what you buy with log|Theta|.

Run:  python3 bounds.py
"""
from __future__ import annotations

import math


def finite_class(n, n_hyp, eps=0.05, C=1.0):
    """Eq. 3. Hoeffding + union bound over n_hyp candidates."""
    return C*math.sqrt(math.log(n_hyp/eps)/(2*n))


def continuous(n, d, eps=0.05, C=1.0):
    """Eq. 7, the best case of the quantisation bound at one bit per parameter."""
    return C*math.sqrt((d + 2*math.log(d) + math.log(1/eps))/(2*n))


def union_bound_is_hoeffding(n, n_hyp, eps=0.05):
    """Rederive Eq. 3 so the mechanism is visible rather than asserted.

    Hoeffding for one fixed hypothesis: P(R - r > t) <= exp(-2 n t^2 / C^2).
    Union over n_hyp of them and set the total to eps, then solve for t.
    """
    t = finite_class(n, n_hyp)
    return n_hyp*math.exp(-2*n*t**2)          # should equal eps


if __name__ == "__main__":
    n, d, M = 80, 1000, 20000                 # 16-shot 5-way; LoRA-XS-sized adapter
    print("Their 16-shot 5-way setting: n = 80 support examples\n")
    print(f"{'bound':>34} {'complexity':>12} {'verdict':>12}")
    c = continuous(n, d)
    f = finite_class(n, M)
    print(f"{'continuous, d = %d params' % d:>34} {c:>12.2f} "
          f"{'vacuous' if c >= 1 else 'usable':>12}")
    print(f"{'finite, |Theta| = %d' % M:>34} {f:>12.3f} "
          f"{'vacuous' if f >= 1 else 'usable':>12}")
    print(f"\n  ratio {c/f:.1f}x.  The complexity term went from d = {d} to "
          f"log|Theta| = {math.log(M/0.05):.1f}.")

    n_needed = int((d + 2*math.log(d) + math.log(1/0.05))/2)
    print(f"  The continuous bound only drops below 1 at n = {n_needed}, "
          f"far outside few-shot.\n")

    print("Eq. 3 really is Hoeffding + a union bound:")
    print(f"  n_hyp * exp(-2 n t^2) = {union_bound_is_hoeffding(n, M):.4f}  (eps = 0.05)\n")

    print("Complexity vs support size")
    print(f"{'n':>6} {'continuous':>12} {'finite':>10}")
    for nn in (10, 25, 80, 250, 508, 2000):
        print(f"{nn:>6} {continuous(nn, d):>12.2f} {finite_class(nn, M):>10.3f}")
    print("\nSensitivity: the finite bound is barely affected by how many candidates"
          "\nyou draw, because the count enters through a log.")
    print(f"{'|Theta|':>10} {'complexity at n=80':>20}")
    for m in (100, 1000, 10000, 100000, 1000000):
        print(f"{m:>10} {finite_class(n, m):>20.3f}")
