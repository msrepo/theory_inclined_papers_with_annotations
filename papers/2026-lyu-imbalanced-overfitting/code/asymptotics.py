#!/usr/bin/env python3
"""A statistical theory of overfitting for imbalanced classification: the
limiting (n, d -> infinity, n/d -> delta) quantities, computed from the paper's
own equations, plus checks of the algebra the notes lean on.

Every number quoted in ../notes.md that is not a simulation comes from here;
the simulations are in svm_simulation.py.

Checked here:

  1. g1(t) = E[(G+t)_+] = t Phi(t) + phi(t) and g2(t) = E[(G+t)_+^2]
     = (t^2+1) Phi(t) + t phi(t), against Monte Carlo.
  2. The separability threshold delta*(0) (Eq 28) by direct maximisation,
     and Cover's 2 at mu = 0, pi = 1/2.
  3. The system of Lemma E.9 (Eqs 76a-c) for (rho*, beta0*, kappa*), and
     that its solution maximises kappa in Eq 33: the "budget" constraint
     E[(1 - rho^2) xi^2] = (1 - rho^2)/delta is tight.
  4. The per-point push: g1 of the minority gap is (1 - pi)/pi times g1 of
     the majority gap (the two first-order conditions, Eq 79).
  5. Corollary E.10: tau changes beta0* and kappa* only, and tau_opt (Eq 38)
     is exactly the tau that sets beta0* = 0.
  6. The limiting test errors (Theorem 2.1a) against pi, with and without
     rebalancing (the curves of Figure 4).
  7. Calibration at tau_opt (Proposition D.9): how much of CalErr* is the
     missing prior log-odds log(pi/(1 - pi)).
  8. The high-imbalance orders of Theorem 3.2 / Lemma H.1 (margin ~ half the
     distance between the two training class means).
  9. The optimal-transport claim (Proposition D.2): the monotone map from
     the TLD to the ELD is x -> max(kappa, x), and the pointwise Pythagorean
     inequality that makes truncation the cheapest way to spend the budget.

Standard library and numpy only.

Run:  python3 asymptotics.py
"""
from __future__ import annotations

import math

import numpy as np

SQ2 = math.sqrt(2.0)


# ----------------------------------------------------------------------------
# Gaussian helpers
# ----------------------------------------------------------------------------
def Phi(t: float) -> float:
    return 0.5 * math.erfc(-t / SQ2)


def phi(t: float) -> float:
    return math.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)


def g1(t: float) -> float:
    """E[(G + t)_+], G ~ N(0,1): the mean amount a Gaussian overshoots -t."""
    return t * Phi(t) + phi(t)


def g2(t: float) -> float:
    """E[(G + t)_+^2]."""
    return (t * t + 1.0) * Phi(t) + t * phi(t)


def g1_inv(y: float) -> float:
    """Inverse of the strictly increasing g1 : R -> (0, inf), by bisection."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g1(mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bisect(f, lo: float, hi: float, it: int = 200) -> float:
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# The separability threshold delta*(0), Eq 28
# ----------------------------------------------------------------------------
def H(kappa: float, rho: float, b0: float, m: float, pi: float, tau: float = 1.0) -> float:
    """H_kappa(rho, beta0) = (1 - rho^2) / E[(s(Y) kappa - rho m - G - beta0 Y)_+^2]."""
    den = pi * g2(tau * kappa - rho * m - b0) + (1 - pi) * g2(kappa - rho * m + b0)
    return (1 - rho * rho) / den


def best_b0(kappa: float, rho: float, m: float, pi: float, tau: float = 1.0) -> float:
    """For fixed rho, the beta0 maximising H solves pi g1(tau k - rho m - b0) = (1-pi) g1(k - rho m + b0)."""
    f = lambda b: (1 - pi) * g1(kappa - rho * m + b) - pi * g1(tau * kappa - rho * m - b)
    return bisect(f, -30.0, 30.0)


def delta_star(kappa: float, m: float, pi: float, tau: float = 1.0) -> tuple[float, float, float]:
    """max over (rho, beta0) of H_kappa; returns (value, rho, beta0)."""
    grid = np.linspace(0.0, 0.999, 400)
    vals = [H(kappa, r, best_b0(kappa, r, m, pi, tau), m, pi, tau) for r in grid]
    k = int(np.argmax(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    f = lambda r: H(kappa, r, best_b0(kappa, r, m, pi, tau), m, pi, tau)
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(80):
        if f(c) > f(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    r = 0.5 * (a + b)
    b0 = best_b0(kappa, r, m, pi, tau)
    return H(kappa, r, b0, m, pi, tau), r, b0


# ----------------------------------------------------------------------------
# Lemma E.9: the limiting parameters
# ----------------------------------------------------------------------------
def solve(m: float, pi: float, delta: float, tau: float = 1.0) -> dict:
    """(rho*, beta0*, kappa*) from Eqs 76a-c; kappa* > 0 iff delta < delta*(0)."""
    ap = lambda r: r / (2 * pi * m * delta)          # g1 of the minority gap
    am = lambda r: r / (2 * (1 - pi) * m * delta)    # g1 of the majority gap
    g = lambda y: g2(g1_inv(y))
    f = lambda r: pi * delta * g(ap(r)) + (1 - pi) * delta * g(am(r)) - (1 - r * r)
    rho = bisect(f, 1e-9, 1 - 1e-12)
    tp, tm = g1_inv(ap(rho)), g1_inv(am(rho))
    # Eq 76b: -b0 + tau k = rho m + tp ;  Eq 76c: b0 + k = rho m + tm
    kappa = (2 * rho * m + tp + tm) / (tau + 1)
    b0 = rho * m + tm - kappa
    err_p, err_m = Phi(-rho * m - b0), Phi(-rho * m + b0)
    return dict(rho=rho, b0=b0, kappa=kappa, tp=tp, tm=tm, err_p=err_p, err_m=err_m,
                err_b=0.5 * (err_p + err_m), tau=tau, m=m, pi=pi, delta=delta)


def tau_opt(m: float, pi: float, delta: float) -> float:
    """Eq 38."""
    s = solve(m, pi, delta)
    return (s["tp"] + s["rho"] * m) / (s["tm"] + s["rho"] * m)


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


# ----------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(0)
    np.set_printoptions(precision=4, suppress=True)

    print("1. g1, g2 closed forms against Monte Carlo (1e6 draws)")
    G = rng.standard_normal(1_000_000)
    for t in (-2.0, -0.5, 0.0, 1.0):
        mc1, mc2 = np.maximum(G + t, 0).mean(), (np.maximum(G + t, 0) ** 2).mean()
        print(f"   t={t:+.1f}: g1 {g1(t):.4f} (MC {mc1:.4f})   g2 {g2(t):.4f} (MC {mc2:.4f})")

    print("\n2. separability threshold delta*(0)")
    v, r, b = delta_star(0.0, 0.0, 0.5)
    print(f"   mu = 0, pi = 0.5: delta*(0) = {v:.4f} at rho={r:.3f}, b0={b:+.3f}   (Cover: 2)")
    for m, pi in ((0.0, 0.15), (1.0, 0.5), (1.75, 0.15), (1.75, 0.5), (1.0, 0.15)):
        v, r, b = delta_star(0.0, m, pi)
        print(f"   ||mu|| = {m:.2f}, pi = {pi:.2f}: delta*(0) = {v:.3f}  (rho={r:.3f}, b0={b:+.3f})")

    print("\n3. Lemma E.9 in the Figure 1 setting (||mu|| = 1.75, pi = 0.15, delta = 2.5)")
    m, pi, delta = 1.75, 0.15, 2.5
    s = solve(m, pi, delta)
    print(f"   rho* = {s['rho']:.4f}  beta0* = {s['b0']:+.4f}  kappa* = {s['kappa']:.4f}")
    print(f"   Err+ = {s['err_p']:.4f}  Err- = {s['err_m']:.4f}  Err_b = {s['err_b']:.4f}")
    bayes = Phi(-m + math.log((1 - pi) / pi) / (2 * m)), Phi(-m - math.log((1 - pi) / pi) / (2 * m))
    print(f"   (Bayes classifier, for scale: Err+ = {bayes[0]:.4f}, Err- = {bayes[1]:.4f};"
          f" balanced-optimal Phi(-||mu||) = {Phi(-m):.4f})")
    budget = pi * g2(s['kappa'] - m * s['rho'] - s['b0']) + (1 - pi) * g2(s['kappa'] - m * s['rho'] + s['b0'])
    print(f"   budget: E[(1-rho^2) xi^2] = {budget:.5f}  vs (1-rho^2)/delta = {(1 - s['rho']**2) / delta:.5f}")
    dk = delta_star(s['kappa'], m, pi)
    print(f"   cross-check: delta*(kappa*) = {dk[0]:.4f} (should equal delta = {delta}),"
          f" at rho={dk[1]:.4f}, b0={dk[2]:+.4f}")
    fp, fm = Phi(s['kappa'] - m * s['rho'] - s['b0']), Phi(s['kappa'] - m * s['rho'] + s['b0'])
    print(f"   mass pushed onto the margin: minority {fp:.3f}, majority {fm:.3f}")
    print(f"   mass pushed across the boundary (test errors wiped out in training): "
          f"minority {s['err_p']:.3f}, majority {s['err_m']:.3f}")
    frac_sv = pi * fp + (1 - pi) * fm
    print(f"   fraction of all training points on the margin = {frac_sv:.4f};"
          f" times delta = {frac_sv * delta:.4f} support vectors per dimension")

    print("\n4. per-point push (Eq 79): g1(minority gap) / g1(majority gap) = (1-pi)/pi")
    gp = g1(s['kappa'] - m * s['rho'] - s['b0'])
    gm = g1(s['kappa'] - m * s['rho'] + s['b0'])
    print(f"   mean push per minority point {gp:.4f}, per majority point {gm:.4f},"
          f" ratio {gp / gm:.3f} vs (1-pi)/pi = {(1 - pi) / pi:.3f}")
    print(f"   total push per class: pi*{gp:.4f} = {pi * gp:.4f}, (1-pi)*{gm:.4f} = {(1 - pi) * gm:.4f}")
    print(f"   2*beta0* = g1^-1(a-) - g1^-1(a+) = {s['tm'] - s['tp']:+.4f} = {2 * s['b0']:+.4f}  (< 0 because a+ > a-)")

    print("\n5. margin rebalancing (Corollary E.10, Eq 38)")
    to = tau_opt(m, pi, delta)
    s1 = solve(m, pi, delta, 1.0)
    for tau in (1.0, 2.0, to, 8.0):
        st = solve(m, pi, delta, tau)
        pred_b0 = s1['b0'] + (tau - 1) / (tau + 1) * s1['kappa']
        print(f"   tau={tau:6.3f}: rho*={st['rho']:.4f} beta0*={st['b0']:+.4f} (E.10: {pred_b0:+.4f})"
              f" kappa*={st['kappa']:.4f} (E.10: {2 / (tau + 1) * s1['kappa']:.4f})"
              f"  Err+={st['err_p']:.4f} Err-={st['err_m']:.4f} Err_b={st['err_b']:.4f}")
    print(f"   tau_opt = {to:.4f}; 1/pi = {1 / pi:.3f}; the range of beta0*(tau) over tau>0 is"
          f" ({s1['b0'] - s1['kappa']:+.3f}, {s1['b0'] + s1['kappa']:+.3f}), so beta0*=0 is reachable"
          f" iff beta0*(1) + kappa*(1) > 0: {s1['b0'] + s1['kappa']:+.4f}")

    print("\n6. limiting test errors against pi (||mu|| = 1.5, delta = 0.5, as in Figure 4's n/d)")
    m2, d2 = 1.5, 0.5
    print("      pi    rho*   Err+(tau=1)  Err-(tau=1)  Err_b(tau=1)  tau_opt  Err(tau_opt)  2||mu||sqrt(pi delta)")
    for p in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
        st = solve(m2, p, d2)
        to2 = tau_opt(m2, p, d2)
        print(f"   {p:5.2f}  {st['rho']:.4f}   {st['err_p']:.4f}       {st['err_m']:.4f}       "
              f"{st['err_b']:.4f}     {to2:7.3f}   {Phi(-st['rho'] * m2):.4f}        {2 * m2 * math.sqrt(p * d2):.4f}")
    v, _, _ = delta_star(0.0, m2, 0.01)
    print(f"   (delta*(0) at pi = 0.01 is {v:.3f}, so every row is separable)")

    print("\n7. calibration at tau_opt (Prop D.9): at beta0* = 0 the true posterior of the test logit f is")
    print("   p0(f) = sigma(2 rho* ||mu|| f + log(pi/(1-pi))), while the confidence is sigma(f)")
    G = rng.standard_normal(400_000)
    print("   setting            pi    tau_opt  slope 2rho*||mu||  CalErr*   CalErr* after adding log(pi/(1-pi))")
    for (mm, dd, lab) in ((1.0, 2.0, "Fig 6  ||mu||=1  "), (2.0, 10.0, "Fig 10 ||mu||=2  "),
                          (0.5, 2.0, "Fig 11 ||mu||=0.5")):
        for p in (0.5, 0.25, 0.1, 0.05):
            st = solve(mm, p, dd)
            rho = st['rho']
            Y = np.where(rng.random(G.size) < p, 1.0, -1.0)
            f = rho * mm * Y + G                   # test logit at beta0* = 0
            p0 = sigmoid(2 * rho * mm * f + math.log(p / (1 - p)))
            cal = np.mean((sigmoid(f) - p0) ** 2)
            cal_prior = np.mean((sigmoid(f + math.log(p / (1 - p))) - p0) ** 2)
            print(f"   {lab}  {p:5.2f}  {tau_opt(mm, p, dd):7.2f}   {2 * rho * mm:6.3f}            "
                  f"{cal:.4f}    {cal_prior:.4f}")
    for mm in (1.25, 1.5, 1.75):
        pc = bisect(lambda p: (lambda z: z['b0'] + z['kappa'])(solve(mm, p, 0.5)), 0.005, 0.3)
        print(f"   Figure 4/9 setting (delta = 0.5), ||mu|| = {mm}: tau_opt > 0 only for pi > {pc:.4f}")
    pc = bisect(lambda p: (lambda z: z['b0'] + z['kappa'])(solve(1.0, p, 2.0)), 0.01, 0.5)
    print(f"   Figure 6 setting (delta = 2), ||mu|| = 1: tau_opt > 0 only for pi > {pc:.4f}")

    print("\n8. high imbalance (Lemma H.1): margin <= half the distance between training class means")
    for d in (500, 2000):
        a, b, c = 0.5, 0.3, 0.1
        pi_d, m_d, n_d = d ** -a, math.sqrt(0.75 * d ** b), int(d ** (c + 1))
        npos = max(int(round(pi_d * n_d)), 1)
        mu = np.zeros(d); mu[0] = m_d
        Xp = mu + rng.standard_normal((npos, d)); Xm = -mu + rng.standard_normal((n_d - npos, d))
        half = 0.5 * np.linalg.norm(Xp.mean(0) - Xm.mean(0))
        pred = math.sqrt(m_d ** 2 + d / (4 * npos) + d / (4 * (n_d - npos)))
        print(f"   d={d}: n={n_d}, n+={npos}, ||mu||^2={m_d**2:.2f}; half-distance of means {half:.3f}"
              f" vs sqrt(||mu||^2 + d/(4n+)) = {pred:.3f};  d/(4n+) = {d / (4 * npos):.2f}")
    print("   phases for (b, c) = (0.3, 0.1): a - c < 0.3 high, 0.3 < a - c < 0.6 moderate, > 0.6 low")

    print("\n9. optimal transport and the Pythagorean inequality behind truncation")
    s = solve(1.75, 0.15, 2.5)
    k = s['kappa']
    X = rng.standard_normal(200_000) + s['rho'] * 1.75 + s['b0']      # minority TLD in y*f units
    T = np.maximum(k, X)
    # monotone rearrangement: sort X, sort target sample, compare
    target = np.sort(np.maximum(k, rng.standard_normal(X.size) + s['rho'] * 1.75 + s['b0']))
    print(f"   max |sorted(max(k,X)) - sorted(target)| on the 1%..99% quantiles: "
          f"{np.max(np.abs(np.sort(T) - target)[2000:-2000]):.3f}  (Monte Carlo noise level)")
    print(f"   W2^2(TLD, ELD) for the minority = E[(k - X)_+^2] = {np.mean((k - X).clip(0) ** 2):.4f}"
          f" = g2(k - rho m - b0) = {g2(k - s['rho'] * 1.75 - s['b0']):.4f}")
    V = X + np.abs(rng.standard_normal(X.size)) * (X < k) * 3 + np.maximum(k - X, 0)   # any V >= k
    lhs = np.mean((V - X) ** 2)
    rhs = np.mean((V - np.maximum(k, X)) ** 2) + np.mean(np.maximum(k - X, 0) ** 2)
    print(f"   for a V >= kappa: E(V-X)^2 = {lhs:.4f} >= E(V-max(k,X))^2 + E(k-X)_+^2 = {rhs:.4f}")


if __name__ == "__main__":
    main()
