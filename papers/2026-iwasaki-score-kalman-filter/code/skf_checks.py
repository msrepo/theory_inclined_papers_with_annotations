#!/usr/bin/env python3
"""The Score Kalman Filter: the algebra that holds up, and the two steps that do not.

Everything the SKF does is a linear solve built out of moments. This script
checks each solve on small cases where the exact answer is known, and prints
every number quoted in ../notes.md.

Checked here:

  1. Prop 1 on a Gaussian (App D.3): A and b by hand, lambda = (-mu/P, 1/2P).
  2. Theorem 1: score matching recovers lambda exactly when the moments come from
     a member of the family, in 1-D (quartic) and 2-D (14 unknowns).
  3. Prop 2 (Stein's identity): residuals on the same 2-D density, the Gaussian
     moment recursion, and a one-layer closure m_7 from m_0..m_6.
  4. Off the model: score matching versus MaxEnt on a Gaussian mixture. Score
     matching can return a non-normalisable fit, or a density whose mass sits
     in a spurious well, from perfectly realisable moments.
  5. Odd r: a cubic energy is never normalisable.
  6. App B.5 equation counts.
  7. Sec 5.2 moment recovery implemented as written, where the right answer is
     the Kalman update: it moves the posterior mean the wrong way.
  8. The Sec 5.2 / App E refinement: with exact recovery it re-applies the
     measurement on every pass; with the truncated recovery it returns the
     posterior mean to the prior mean.

Standard library and numpy only.

Run:  python3 skf_checks.py
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np


# ------------------------------------------------------------ multi-indices
def multi_indices(n, lo, hi):
    """All alpha in N^n with lo <= |alpha| <= hi, graded by degree."""
    out = []
    for d in range(lo, hi + 1):
        for c in itertools.combinations_with_replacement(range(n), d):
            a = [0] * n
            for i in c:
                a[i] += 1
            out.append(tuple(a))
    return out


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def unit(i, n):
    v = [0] * n
    v[i] = 1
    return tuple(v)


def grid_moments(logp, grids, maxdeg):
    """E[x^alpha], |alpha| <= maxdeg, for the density exp(logp) on a tensor grid."""
    mesh = np.meshgrid(*grids, indexing="ij")
    lp = logp(*mesh)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    m = {}
    for al in multi_indices(len(grids), 0, maxdeg):
        f = np.ones_like(w)
        for i, ai in enumerate(al):
            if ai:
                f = f * mesh[i] ** ai
        m[al] = float((f * w).sum())
    return m


# ------------------------------------------------ Prop 1: the linear system
def score_matching_system(m, n, r):
    """Eq 8: A_ab = sum_i a_i b_i m_{a+b-2e_i},  b_a = sum_i a_i (a_i - 1) m_{a-2e_i}.
    Non-constant basis only (the constant row and column are identically zero)."""
    basis = multi_indices(n, 1, r)
    A = np.zeros((len(basis), len(basis)))
    b = np.zeros(len(basis))
    for p, al in enumerate(basis):
        for q, be in enumerate(basis):
            A[p, q] = sum(al[i] * be[i] * m[sub(add(al, be), add(unit(i, n), unit(i, n)))]
                          for i in range(n) if al[i] and be[i])
        b[p] = sum(al[i] * (al[i] - 1) * m[sub(al, add(unit(i, n), unit(i, n)))]
                   for i in range(n) if al[i] >= 2)
    return A, b, basis


def score_match(m, n, r):
    mm = dict(m)
    mm[tuple([0] * n)] = 1.0
    A, b, basis = score_matching_system(mm, n, r)
    return dict(zip(basis, np.linalg.solve(A, b)))


def check_prop1_gaussian():
    print("1. Prop 1 on a Gaussian (App D.3)")
    mu, P = 1.0, 0.5
    A, b, _ = score_matching_system({(0,): 1.0, (1,): mu, (2,): mu**2 + P}, 1, 2)
    lam = np.linalg.solve(A, b)
    print(f"   mu = {mu}, P = {P}:  A = {A.tolist()},  b = {b.tolist()}")
    print(f"   lambda = ({lam[0]:+.4f}, {lam[1]:+.4f});  -mu/P = {-mu / P:+.4f}, 1/(2P) = {1 / (2 * P):+.4f}\n")


QUARTIC_1D = {(1,): 0.3, (2,): -2.0, (3,): 0.1, (4,): 1.0}          # two wells, slightly tilted
QUARTIC_2D = {(4, 0): 0.25, (0, 4): 0.25, (2, 0): -0.5, (0, 2): -0.5,
              (1, 1): 0.3, (1, 0): 0.1, (2, 1): 0.05}


def energy(lam, *xs):
    return sum(v * np.prod([x**a for x, a in zip(xs, al)], axis=0) for al, v in lam.items())


def check_theorem1():
    print("2. Theorem 1: exact recovery on the model class")
    x = np.linspace(-4, 4, 40001)
    m = grid_moments(lambda X: -energy(QUARTIC_1D, X), [x], 6)
    A, b, basis = score_matching_system(m, 1, 4)
    lam = np.linalg.solve(A, b)
    print(f"   1-D quartic: true {[QUARTIC_1D[k] for k in basis]}, recovered {np.round(lam, 6).tolist()}, "
          f"cond(A) = {np.linalg.cond(A):.0f}")
    g = np.linspace(-3.5, 3.5, 701)
    m2 = grid_moments(lambda X, Y: -energy(QUARTIC_2D, X, Y), [g, g], 6)
    A, b, basis = score_matching_system(m2, 2, 4)
    lam = np.linalg.solve(A, b)
    err = np.abs(lam - np.array([QUARTIC_2D.get(k, 0.0) for k in basis])).max()
    print(f"   2-D quartic: {len(basis)} unknowns, max |lambda - true| = {err:.1e}, cond(A) = {np.linalg.cond(A):.0f}\n")
    return m, m2


def stein_residual(lam, m, n, beta, i):
    """Eq 12 left minus right:  sum_a lambda_a a_i m_{a+beta-e_i} - beta_i m_{beta-e_i}."""
    lhs = sum(v * al[i] * m[sub(add(al, beta), unit(i, n))] for al, v in lam.items() if al[i])
    rhs = beta[i] * m[sub(beta, unit(i, n))] if beta[i] else 0.0
    return lhs - rhs


def check_prop2(m1, m2):
    print("3. Prop 2: Stein's identity")
    res = max(abs(stein_residual(QUARTIC_2D, m2, 2, be, i))
              for be in multi_indices(2, 0, 3) for i in range(2))
    print(f"   2-D density, all |beta| <= 3, both coordinates: max residual {res:.1e}")
    mu, s2, mg = 1.0, 1.0, [1.0, 1.0]
    for k in range(1, 4):
        mg.append(mu * mg[k] + k * s2 * mg[k - 1])            # m_{k+1} = mu m_k + k s^2 m_{k-1}
    print(f"   Gaussian recursion, mu = 1, var = 1: m_0..m_4 = {mg}")
    x = np.linspace(-4, 4, 40001)
    mm = grid_moments(lambda X: -energy(QUARTIC_1D, X), [x], 7)
    lam, r = {k[0]: v for k, v in QUARTIC_1D.items()}, 4
    mk = [mm[(k,)] for k in range(8)]
    m7 = (r * mk[r - 1] - sum(j * lam[j] * mk[r + j - 1] for j in range(1, r))) / (r * lam[r])
    print(f"   closure layer 1 (1-D quartic): m_7 from lambda and m_0..m_6 = {m7:.6f}, by quadrature {mk[7]:.6f}\n")


# -------------------------------------------------- off the model: SM vs ME
def mixture_logpdf(x):
    return np.log(0.6 * np.exp(-0.5 * ((x + 1.2) / 0.5) ** 2) / 0.5
                  + 0.4 * np.exp(-0.5 * ((x - 1.5) / 0.8) ** 2) / 0.8)


def energy_1d(lam, x):
    return sum(lam[k - 1] * x**k for k in range(1, len(lam) + 1))


def density_1d(lam, x):
    E = energy_1d(lam, x)
    w = np.exp(-(E - E.min()))
    return w / (w.sum() * (x[1] - x[0]))


def maxent_1d(m, r, x, iters=200):
    """Moment matching: Newton with backtracking on the convex dual  lambda.m + log Z(lambda)."""
    Phi = np.stack([x**k for k in range(1, r + 1)])
    lam = np.zeros(r)
    lam[1] = 0.5

    def dual(l):
        E = l @ Phi
        c = E.min()
        return l @ m[1:r + 1] + np.log(np.exp(-(E - c)).sum()) - c

    for _ in range(iters):
        E = lam @ Phi
        w = np.exp(-(E - E.min()))
        w /= w.sum()
        mp = Phi @ w
        g = m[1:r + 1] - mp                                   # gradient of the dual
        if np.abs(g).max() < 1e-12:
            break
        d = -np.linalg.solve((Phi * w) @ Phi.T - np.outer(mp, mp) + 1e-12 * np.eye(r), g)
        t, f0 = 1.0, dual(lam)
        while dual(lam + t * d) > f0 + 1e-4 * t * (g @ d) and t > 1e-10:
            t *= 0.5
        lam = lam + t * d
    return lam


def check_off_model():
    print("4. Off the model: a two-Gaussian mixture, in standardised coordinates")
    xs = np.linspace(-12, 12, 240001)
    w = np.exp(mixture_logpdf(xs))
    w /= w.sum()
    mu = (xs * w).sum()
    sd = np.sqrt(((xs - mu) ** 2 * w).sum())
    x = (xs - mu) / sd
    p = np.exp(mixture_logpdf(x * sd + mu))
    p /= p.sum() * (x[1] - x[0])
    m = np.array([(x**k * p).sum() * (x[1] - x[0]) for k in range(17)])
    dx = x[1] - x[0]
    print(f"   mixture mean {mu:.3f}, sd {sd:.3f}")
    print(f"   {'r':>2}  {'SM leading':>10}  {'SM normalisable':>15}  {'L1(SM,p)':>8}  {'L1(ME,p)':>8}  {'cond(A)':>8}")
    for r in (2, 4, 6, 8):
        A = np.array([[a * b * m[a + b - 2] for b in range(1, r + 1)] for a in range(1, r + 1)])
        b = np.array([a * (a - 1) * m[a - 2] if a >= 2 else 0.0 for a in range(1, r + 1)])
        lsm = np.linalg.solve(A, b)
        lme = maxent_1d(m, r, x)
        l1_me = np.abs(density_1d(lme, x) - p).sum() * dx
        ok = lsm[-1] > 0
        l1_sm = f"{np.abs(density_1d(lsm, x) - p).sum() * dx:8.3f}" if ok else "     n/a"
        print(f"   {r:>2}  {lsm[-1]:>+10.4f}  {'yes' if ok else 'NO':>15}  {l1_sm}  {l1_me:8.3f}  {np.linalg.cond(A):8.1e}")
        if r == 8:
            E = energy_1d(lsm, x)
            bulk = np.abs(x) < 2
            print(f"   r = 8 SM energy: lowest value in the bulk (|x| < 2) {E[bulk].min():.2f}, "
                  f"global minimum {E.min():.2f} at x = {x[E.argmin()]:.2f} sd, "
                  f"where the target density is {p[E.argmin()]:.1e}")
    print()


def check_odd_r():
    print("5. Odd r: exp(-E) for the cubic energy E = x^2/2 + 0.05 x^3")
    for L in (5, 10, 15, 20):
        x = np.linspace(-L, L, 200001)
        E = 0.5 * x**2 + 0.05 * x**3
        print(f"   integral over [-{L}, {L}] = {np.trapezoid(np.exp(-E), x):.3e}")
    print()


def check_counts():
    print("6. App B.5 equation counts")
    U = lambda n, r: comb(2 * r + n - 2, n - 1)
    R = lambda n, r: sum(n * comb(r + j + n - 2, n - 1) for j in range(r - 1))
    Rext = lambda n, r: sum(n * comb(r + j + n - 2, n - 1) for j in range(r))
    print("   r = 3, R/U at n = 10, 14, 15, 16, 40: "
          + ", ".join(f"{R(n, 3) / U(n, 3):.3f}" for n in (10, 14, 15, 16, 40)))
    print(f"   r = 3, n = 40: R = {R(40, 3):,}, R_ext = {Rext(40, 3):,}, U = {U(40, 3):,}")
    first = next(n for n in range(2, 200) if R(n, 4) < U(n, 4))
    print(f"   r = 4: first n with R < U is {first}")
    print(f"   n = 2 first layer square for r = 2..9: {all(2 * r == U(2, r) for r in range(2, 10))}")
    print(f"   n = 20: C(24,5) = {comb(24, 5):,}, M(r=3) = C(23,3) = {comb(23, 3):,}, "
          f"moments to degree 4 = C(24,4) = {comb(24, 4):,}\n")


# ------------------------------------------- Sec 5.2 recovery, as written
def poly_affine(P, c, s):
    """Coefficients of P(c + s*w) as a polynomial in w; constant term dropped."""
    out, n = {}, len(c)
    for al, v in P.items():
        per_dim = [[(k, comb(al[i], k) * c[i] ** (al[i] - k) * s[i] ** k) for k in range(al[i] + 1)]
                   for i in range(n)]
        for combo in itertools.product(*per_dim):
            g = tuple(k for k, _ in combo)
            cf = v
            for _, f in combo:
                cf *= f
            out[g] = out.get(g, 0.0) + cf
    out.pop(tuple([0] * n), None)
    return out


def quadratic_energy(Om, eta):
    """exp(-lambda.phi) with E(x) = 1/2 x'Om x - eta'x, in monomial coordinates."""
    n, L = len(eta), {}
    for i in range(n):
        L[unit(i, n)] = -eta[i]
        for j in range(i, n):
            L[add(unit(i, n), unit(j, n))] = (0.5 if i == j else 1.0) * Om[i, j]
    return L


def stein_recover(L, n, r, zero_rows=False):
    """Eq 15 as Sec 5.2 describes it: directional rows (beta, i) with beta_i >= 1 and
    1 <= |beta| <= K = 2r-2, unknowns m_gamma with 1 <= |gamma| <= K, m_0 = 1, and every
    moment of degree > K truncated to zero. zero_rows adds the beta_i = 0 rows as well."""
    K = 2 * r - 2
    unknowns = multi_indices(n, 1, K)
    col = {g: j for j, g in enumerate(unknowns)}
    rows, rhs = [], []
    for be in multi_indices(n, 0 if zero_rows else 1, K):
        for i in range(n):
            if be[i] == 0 and not zero_rows:
                continue
            row, const = np.zeros(len(unknowns)), 0.0
            for al, v in L.items():
                if al[i]:
                    g = sub(add(al, be), unit(i, n))
                    if sum(g) == 0:
                        const += v * al[i]
                    elif sum(g) <= K:
                        row[col[g]] += v * al[i]
            if be[i]:
                g = sub(be, unit(i, n))
                if sum(g) == 0:
                    const -= be[i]
                else:
                    row[col[g]] -= be[i]
            rows.append(row)
            rhs.append(-const)
    sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return {g: sol[j] for g, j in col.items()}, (len(rows), len(unknowns))


def update_and_recover(n, r, mu0, P0, C, Rv, z, centre, passes=0, zero_rows=False, scale=3.5):
    """Gaussian prior, linear-Gaussian measurement: the conjugate update lambda+ = lambda- + lambda_lik
    is exact, so any error in the posterior mean comes from the recovery (and the refinement)."""
    s = scale * np.sqrt(np.diag(P0))
    Lprior = poly_affine(quadratic_energy(np.linalg.inv(P0), np.linalg.inv(P0) @ mu0), centre, s)
    Ri = np.linalg.inv(Rv)
    Llik = poly_affine(quadratic_energy(C.T @ Ri @ C, C.T @ Ri @ z), centre, s)
    keys = multi_indices(n, 1, r)
    L = {k: Lprior.get(k, 0.0) + Llik.get(k, 0.0) for k in keys}
    means = []
    for _ in range(passes + 1):
        m, shape = stein_recover(L, n, r, zero_rows)
        means.append(centre + s * np.array([m[unit(i, n)] for i in range(n)]))
        refit = score_match(m, n, r)                          # (i) re-fit lambda to the recovered m+
        L = {k: refit.get(k, 0.0) + Llik.get(k, 0.0) for k in keys}   # (ii) re-apply the measurement
    Pp = np.linalg.inv(np.linalg.inv(P0) + C.T @ Ri @ C)
    return means, Pp @ (np.linalg.inv(P0) @ mu0 + C.T @ Ri @ z), shape


def check_recovery():
    print("7. Sec 5.2 moment recovery, where the right answer is the Kalman update")
    one = np.eye(1)
    for r in (2, 3, 4):
        args = (1, r, np.zeros(1), one, one, one, np.array([2.0]))
        lit, exact, shape = update_and_recover(*args, centre=np.zeros(1))
        zr, _, _ = update_and_recover(*args, centre=np.zeros(1), zero_rows=True)
        oracle, _, _ = update_and_recover(*args, centre=exact)
        print(f"   1-D r={r} ({shape[0]} rows, {shape[1]} unknowns), prior N(0,1), z = 2, R = 1: "
              f"recovered mean {lit[0][0]:+.3f} (Kalman {exact[0]:+.3f}); "
              f"with beta_i = 0 rows {zr[0][0]:+.3f}; centred at the posterior mean {oracle[0][0]:+.3f}")
    P0 = np.array([[0.04, 0.02], [0.02, 0.04]])
    C, Rv = np.array([[1.0, 0.0]]), np.array([[0.09]])
    for r in (2, 3, 4):
        means, exact, shape = update_and_recover(2, r, np.zeros(2), P0, C, Rv, np.array([0.3]), np.zeros(2))
        print(f"   2-D r={r} ({shape[0]} rows, {shape[1]} unknowns), observe x1, z = 0.3: "
              f"recovered mean {np.round(means[0], 4)}, Kalman {np.round(exact, 4)}, "
              f"gain on x1 {means[0][0] / exact[0]:+.2f}")
    # a genuinely bimodal posterior: prior N(0,1), quadratic sensor z = x^2 + v, z = 2, R = 1, r = 4
    lam = {(1,): 0.0, (2,): -1.5, (3,): 0.0, (4,): 0.5}
    x = np.linspace(-7, 7, 70001)
    p = np.exp(-(0.5 * x**4 - 1.5 * x**2))
    p /= p.sum()
    ex = {k: float((x**k * p).sum()) for k in (1, 2, 4, 6, 8)}
    print(f"   bimodal posterior (prior N(0,1), z = x^2 + v, z = 2, R = 1, r = 4): exact mean {ex[1]:+.3f}, "
          f"variance {ex[2]:.3f}, m4 {ex[4]:.3f}, m6 {ex[6]:.3f}, m8 {ex[8]:.3f}")
    for label, s, zr in (("as written", 3.5, False), ("+ beta_i = 0 rows", 3.5, True),
                         ("scaled by the posterior sd", 3.5 * np.sqrt(ex[2]), False)):
        m, _ = stein_recover(poly_affine(lam, np.zeros(1), np.array([s])), 1, 4, zr)
        mean = s * m[(1,)]
        print(f"      {label:<27} recovered mean {mean:+.3f}, variance {s * s * m[(2,)] - mean**2:+.3f}")
    print()


def check_refinement():
    print("8. The refinement of Sec 5.2 / App E: lambda <- SM(m+) + lambda_lik, then recover again")
    Om, eta = np.array([[1.0]]), np.array([0.0])               # prior N(0, 1)
    lik = quadratic_energy(np.array([[1.0]]), np.array([2.0]))  # z = 2, R = 1
    L = {k: v + lik[k] for k, v in quadratic_energy(Om, eta).items()}
    line = []
    for _ in range(5):
        P = 1.0 / (2 * L[(2,)])
        mean = -L[(1,)] * P                                   # exact recovery (Gaussian closed form)
        line.append(f"N({mean:.3f}, {P:.3f})")
        refit = score_match({(1,): mean, (2,): P + mean**2}, 1, 2)
        L = {k: refit[k] + lik[k] for k in refit}
    print("   exact recovery, 1-D, prior N(0,1), z = 2, R = 1, passes 0..4: " + " -> ".join(line))
    P0 = np.array([[0.04, 0.02], [0.02, 0.04]])
    C, Rv = np.array([[1.0, 0.0]]), np.array([[0.09]])
    print("   truncated recovery, 2-D, gain on x1 after 0, 1 and 8 passes (Kalman = +1):")
    for r in (2, 4):
        for label, scale in (("scaled 3.5 sd", 3.5), ("scaled 1 sd", 1.0), ("centred only", 1 / np.sqrt(P0[0, 0]))):
            means, exact, _ = update_and_recover(2, r, np.zeros(2), P0, C, Rv, np.array([0.3]),
                                                 np.zeros(2), passes=8, scale=scale)
            g = [mm[0] / exact[0] for mm in means]
            print(f"      r={r}, {label:<13}  {g[0]:+.2f}  {g[1]:+.2f}  {g[8]:+.2f}")
    print()


if __name__ == "__main__":
    print("The Score Kalman Filter (Iwasaki, Bloch, Lee & Ghaffari 2026)")
    print("=" * 68, "\n")
    check_prop1_gaussian()
    m1, m2 = check_theorem1()
    check_prop2(m1, m2)
    check_off_model()
    check_odd_r()
    check_counts()
    check_recovery()
    check_refinement()
