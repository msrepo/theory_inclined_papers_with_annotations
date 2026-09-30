#!/usr/bin/env python3
"""Sections 3 and C of Tahir, Ganguli & Rotskoff (ICML 2025), checked independently.

The paper derives closed forms for the transferability T = E[R_sc - R_tx] of a deep linear network,
with R_sc the risk of training on the target alone. Here every one of them is tested three ways:

  1. Monte Carlo of the *limit objects* the proofs reduce to (minimum-norm least squares for
     scratch training, a rank-one regression for linear transfer, and beta_sc + (I-P)beta_s for
     fine-tuning) at d = 200 with 300 draws.
  2. Real gradient descent on two-layer linear networks at d = 20 from initial scale alpha = 0.005,
     pretraining on the population loss and then transferring (Theorems 3.4, 3.5, 3.7, 3.9).
  3. The algebra: negative-transfer boundaries, the sigma = 0 limit, the ridge limit (which needs
     only a law of large numbers), and the finite-source fine-tuning formula (22)-(23).

Everything printed here is quoted in the notes.

Standard library and numpy only.

Run:  python3 transfer_theory.py             (prints every number quoted in the notes)
      python3 transfer_theory.py --figures   (also rewrites the SVGs in ../figures/)
      python3 transfer_theory.py --figures-only   (just the SVGs, no checks)
"""
from __future__ import annotations

import math
import sys

import numpy as np

import theory as T

PI = math.pi


# ------------------------------------------------------------------ helpers
def unit(rng, d):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def make_betas(rng, d, th):
    """Unit vectors beta_s, beta_t with beta_s . beta_t = cos(th)."""
    bs = unit(rng, d)
    nu = unit(rng, d)
    nu -= (nu @ bs) * bs
    nu /= np.linalg.norm(nu)
    return bs, math.cos(th) * bs + math.sin(th) * nu


def header(s):
    print("\n" + s + "\n" + "-" * len(s))


# ------------------------------------------------------------------ 1. Monte Carlo of the limit objects
def monte_carlo(d=200, s=0.2, reps=300, seed=0):
    header("1. Monte Carlo of the closed forms, d = %d, sigma = %.1f, %d draws per cell" % (d, s, reps))
    rng = np.random.default_rng(seed)
    s2 = s * s
    print("   gamma   theta   |  R_sc  MC / exact   |  R_lt  MC / exact   |  T_lt  MC / eq.11-exact  |  T_ft  MC / eq.15-exact  |  T_ft with c: MC / formula")
    for g in [0.25, 0.5, 0.9, 1.5]:
        n = int(g * d)
        for th in [PI / 8, PI / 3, PI / 2]:
            sc, lt, ft, ftc = [], [], [], []
            for _ in range(reps):
                bs, bt = make_betas(rng, d, th)
                X = rng.standard_normal((n, d))
                y = X @ bt + s * rng.standard_normal(n)
                pinv = np.linalg.pinv(X)
                b_sc = pinv @ y                                  # Theorem 3.5: minimum-norm solution
                sc.append(np.sum((b_sc - bt) ** 2))
                z = X @ bs
                b = (z @ y) / (z @ z)                            # linear transfer: one scalar on the direction beta_s
                lt.append(np.sum((b * bs - bt) ** 2))
                P = pinv @ X                                     # projector onto row(X)
                ft.append(np.sum((b_sc + (np.eye(d) - P) @ bs - bt) ** 2))   # eq. (143)
                ns = (np.eye(d) - P) @ bs                        # null-space part of beta_s
                a_, b_ = ns @ ns, b_sc @ b_sc
                c_ = math.sqrt((a_ + math.sqrt(a_ * a_ + 4 * b_)) / 2)       # balancedness factor, see theory.c_balance
                ftc.append(np.sum((b_sc + c_ * ns - bt) ** 2))
            sc, lt, ft, ftc = map(np.mean, (sc, lt, ft, ftc))
            ex_sc = T.r_scratch_exact(n, d, s2)
            ex_lt = T.r_lt_exact(n, th, s2)
            ex_ft = ex_sc + ((1 - n / d) * (1 - 2 * math.cos(th)) if n <= d else 0.0)
            tfc = float(T.t_ft_balanced(n / d, th, s2)) if n < d else 0.0
            print("   %.2f   %.3f  |  %.3f / %.3f     |  %.3f / %.3f     |  %+.3f / %+.3f          |  %+.3f / %+.3f          |  %+.3f / %+.3f"
                  % (n / d, th, sc, ex_sc, lt, ex_lt, sc - lt, ex_sc - ex_lt, sc - ft, ex_sc - ex_ft, sc - ftc, tfc))


# ------------------------------------------------------------------ 2. real gradient descent
def gd_check(d=20, alpha=0.005, lr=0.1, s=0.2, reps=100, seed=7):
    header("2. Real gradient descent on two-layer linear networks (d = %d, alpha = %g, %d draws)" % (d, alpha, reps))
    rng = np.random.default_rng(seed)

    def init():
        return (alpha * rng.standard_normal((d, d)) / math.sqrt(d) + 0.5 * alpha * np.eye(d),
                alpha * rng.standard_normal(d) / math.sqrt(d))

    def train(W1, W2, resid, steps, tol):
        for k in range(steps):
            r = resid(W1 @ W2)
            if r @ r < tol:
                break
            W1, W2 = W1 - lr * np.outer(r, W2), W2 - lr * (W1.T @ r)
        return W1, W2, k

    # feature sparsification and the conserved quantity, once
    bs = unit(rng, d)
    W1, W2 = init()
    D0 = W1.T @ W1 - np.outer(W2, W2)
    W1, W2, k = train(W1, W2, lambda b: b - bs, 200000, 1e-26)
    D1 = W1.T @ W1 - np.outer(W2, W2)
    U, S, Vt = np.linalg.svd(W1)
    print("   pretraining converged in %d steps; ||W1 W2 - beta_s|| = %.1e" % (k, np.linalg.norm(W1 @ W2 - bs)))
    print("   Theorem 3.4: top singular value of W1 = %.4f, next = %.4f (= %.1f alpha), |cos(u1, beta_s)| = %.10f"
          % (S[0], S[1], S[1] / alpha, abs(U[:, 0] @ bs)))
    print("   Theorem 3.4 needs D = W1'W1 - W2 W2' to stay put under gradient *flow*; discrete steps of size lr break it by O(lr):")
    for lr2 in [0.1, 0.01, 0.001]:
        U0, V0 = init()
        Wa, Wb = U0.copy(), V0.copy()
        Dref = Wa.T @ Wa - np.outer(Wb, Wb)
        t = 0.0
        while t < 16.0:                                       # the same *time* t = steps * lr for every lr
            r = Wa @ Wb - bs
            Wa, Wb = Wa - lr2 * np.outer(r, Wb), Wb - lr2 * (Wa.T @ r)
            t += lr2
        Dnow = Wa.T @ Wa - np.outer(Wb, Wb)
        print("     lr = %-6g  fit at t = 16: ||W1 W2 - beta_s|| = %.1e   ||D(16) - D(0)||_F = %.2e" % (lr2, np.linalg.norm(Wa @ Wb - bs), np.linalg.norm(Dnow - Dref)))

    s2 = s * s
    print("   Fine-tuning is run with a small step (lr_ft), because the drift of D per step is ~ lr^2 |r|^2 and |r| is large at the start.")
    print("   n   theta |  scratch  sim / exact   |  lin. transfer  sim / exact  |  fine-tune: sim | paper limit | with c   (paired means over the same draws)")
    lr_ft = 0.005
    for n, th in [(5, PI / 2), (10, PI / 3), (15, PI / 8)]:
        sc, lt, ft, lim, cm = [], [], [], [], []
        for _ in range(reps):
            bs, bt = make_betas(rng, d, th)
            W1, W2 = init()
            W1, W2, _ = train(W1, W2, lambda b: b - bs, 200000, 1e-26)
            X = rng.standard_normal((n, d))
            y = X @ bt + s * rng.standard_normal(n)
            res = lambda b: X.T @ (X @ b - y) / n
            S1, S2 = init()
            S1, S2, _ = train(S1, S2, res, 400000, 1e-24)
            sc.append(np.sum((S1 @ S2 - bt) ** 2))
            U, S, Vt = np.linalg.svd(W1)
            F1 = S[0] * np.outer(U[:, 0], Vt[0])             # the learned feature map is (numerically) rank one
            w = np.linalg.pinv(X @ F1) @ y
            lt.append(np.sum((F1 @ w - bt) ** 2))
            lr_save, lr = lr, lr_ft
            F1, F2, _ = train(W1.copy(), W2.copy(), res, 4000000, 1e-20)
            lr = lr_save
            ft.append(np.sum((F1 @ F2 - bt) ** 2))
            pinv = np.linalg.pinv(X)
            b_sc = pinv @ y
            ns = (np.eye(d) - pinv @ X) @ bs
            lim.append(np.sum((b_sc + ns - bt) ** 2))
            a_, b_ = ns @ ns, b_sc @ b_sc
            c_ = math.sqrt((a_ + math.sqrt(a_ * a_ + 4 * b_)) / 2)
            cm.append(np.sum((b_sc + c_ * ns - bt) ** 2))
        ex_sc = T.r_scratch_exact(n, d, s2)
        ex_lt = T.r_lt_exact(n, th, s2)
        m = lambda v: "%.3f±%.3f" % (np.mean(v), np.std(v) / math.sqrt(len(v)))
        pd = lambda u, v: "%+.4f±%.4f" % (np.mean(np.array(u) - np.array(v)), np.std(np.array(u) - np.array(v)) / math.sqrt(len(u)))
        print("   %2d  %.3f |  %s / %.3f  |  %s / %.3f | GD %s | paper %.3f | c-model %.3f  ;  GD - paper %s,  GD - c-model %s"
              % (n, th, m(sc), ex_sc, m(lt), ex_lt, m(ft), np.mean(lim), np.mean(cm), pd(ft, lim), pd(ft, cm)))
    # discretisation: the same draws with a coarse step
    n, th = 5, PI / 2
    rows = []
    for lr_try in [0.1, 0.02, 0.005]:
        rng2 = np.random.default_rng(99)
        diff = []
        for _ in range(40):
            bs, bt = make_betas(rng2, d, th)
            W1 = alpha * rng2.standard_normal((d, d)) / math.sqrt(d) + 0.5 * alpha * np.eye(d)
            W2 = alpha * rng2.standard_normal(d) / math.sqrt(d)
            W1, W2, _ = train(W1, W2, lambda b: b - bs, 200000, 1e-26)
            X = rng2.standard_normal((n, d))
            y = X @ bt + s * rng2.standard_normal(n)
            res = lambda b: X.T @ (X @ b - y) / n
            lr_save, lr = lr, lr_try
            F1, F2, _ = train(W1.copy(), W2.copy(), res, 4000000, 1e-20)
            lr = lr_save
            pinv = np.linalg.pinv(X)
            b_sc = pinv @ y
            ns = (np.eye(d) - pinv @ X) @ bs
            a_, b_ = ns @ ns, b_sc @ b_sc
            c_ = math.sqrt((a_ + math.sqrt(a_ * a_ + 4 * b_)) / 2)
            diff.append(np.sum((F1 @ F2 - bt) ** 2) - np.sum((b_sc + c_ * ns - bt) ** 2))
        rows.append("lr_ft = %g: %+.4f" % (lr_try, np.mean(diff)))
    print("   mean(GD - c-model) at n = 5, theta = pi/2 shrinks with the step, as gradient flow should: " + ";  ".join(rows))



# ------------------------------------------------------------------ 2b. noisy targets: the balancedness factor matters
def noisy_finetune(seed=21):
    header("2b. Fine-tuning at large label noise (sigma = 1): eq. (15) versus the balancedness factor c")
    rng = np.random.default_rng(seed)
    d, reps, s = 200, 200, 1.0
    print("   Monte Carlo at d = %d, %d draws. T_ft = R_sc - R_ft:" % (d, reps))
    print("   gamma   theta |  paper's beta_sc + (I-P)beta_s: MC / eq.(15)  |  beta_sc + c (I-P)beta_s: MC / formula (1-g)(2c cos th - c^2)")
    for g, th in [(0.5, PI / 3), (0.5, PI / 4), (0.75, PI / 3)]:
        n = int(g * d)
        a_list, b_list = [], []
        for _ in range(reps):
            bs, bt = make_betas(rng, d, th)
            X = rng.standard_normal((n, d))
            y = X @ bt + s * rng.standard_normal(n)
            pinv = np.linalg.pinv(X)
            b_sc = pinv @ y
            ns = (np.eye(d) - pinv @ X) @ bs
            a_, b_ = ns @ ns, b_sc @ b_sc
            c_ = math.sqrt((a_ + math.sqrt(a_ * a_ + 4 * b_)) / 2)
            r_sc = np.sum((b_sc - bt) ** 2)
            a_list.append(r_sc - np.sum((b_sc + ns - bt) ** 2))
            b_list.append(r_sc - np.sum((b_sc + c_ * ns - bt) ** 2))
        print("   %.2f   %.3f |  %+.3f / %+.3f                          |  %+.3f / %+.3f"
              % (g, th, np.mean(a_list), T.t_ft(g, th), np.mean(b_list), float(T.t_ft_balanced(g, th, s * s))))
    d, alpha, n, th, reps = 20, 0.005, 10, PI / 3, 60
    def init():
        return (alpha * rng.standard_normal((d, d)) / math.sqrt(d) + 0.5 * alpha * np.eye(d),
                alpha * rng.standard_normal(d) / math.sqrt(d))
    def train(W1, W2, resid, steps, tol, lr):
        for k in range(steps):
            r = resid(W1 @ W2)
            if r @ r < tol:
                break
            W1, W2 = W1 - lr * np.outer(r, W2), W2 - lr * (W1.T @ r)
        return W1, W2
    gd, cm, pp = [], [], []
    for _ in range(reps):
        bs, bt = make_betas(rng, d, th)
        W1, W2 = init()
        W1, W2 = train(W1, W2, lambda b: b - bs, 200000, 1e-26, 0.1)
        X = rng.standard_normal((n, d))
        y = X @ bt + s * rng.standard_normal(n)
        F1, F2 = train(W1.copy(), W2.copy(), lambda b: X.T @ (X @ b - y) / n, 4000000, 1e-20, 0.003)
        pinv = np.linalg.pinv(X)
        b_sc = pinv @ y
        ns = (np.eye(d) - pinv @ X) @ bs
        a_, b_ = ns @ ns, b_sc @ b_sc
        c_ = math.sqrt((a_ + math.sqrt(a_ * a_ + 4 * b_)) / 2)
        gd.append(np.sum((F1 @ F2 - bt) ** 2)); cm.append(np.sum((b_sc + c_ * ns - bt) ** 2)); pp.append(np.sum((b_sc + ns - bt) ** 2))
    gd, cm, pp = map(np.array, (gd, cm, pp))
    se = lambda v: np.std(v) / math.sqrt(len(v))
    print("   Real gradient descent (lr 0.003), d = 20, n = 10, theta = pi/3, %d draws, paired with the same data:" % reps)
    print("     GD - (beta_sc + c (I-P) beta_s) = %+.4f ± %.4f;   GD - (beta_sc + (I-P) beta_s) = %+.4f ± %.4f"
          % (np.mean(gd - cm), se(gd - cm), np.mean(gd - pp), se(gd - pp)))


# ------------------------------------------------------------------ 3. algebra
def algebra():
    header("3a. Eq. (11) at the paper's Figure 1 settings (sigma = 0.2)")
    s = 0.2
    s2 = s * s
    gs = [0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 2.0]
    print("   gamma:           " + "  ".join("%6.2f" % g for g in gs))
    print("   R_sc (eq. 8):    " + "  ".join("%6.3f" % T.r_scratch(g, s2) for g in gs))
    for th, name in [(PI / 8, "pi/8 "), (PI / 4, "pi/4 "), (3 * PI / 8, "3pi/8")]:
        print("   T_lt, th=%s: " % name + "  ".join("%+6.3f" % T.t_lt(g, th, s2) for g in gs)
              + "     floor sin^2 = %.3f" % math.sin(th) ** 2)

    header("3b. Where T_lt changes sign (sigma = 0.2)")
    print("   theta_min = arccos(1 - sigma) = %.4f rad = %.1f deg: below it there is no negative transfer for gamma < 1"
          % (math.acos(1 - s), math.degrees(math.acos(1 - s))))
    for th_deg in [20, 30, 45, 60, 75, 90]:
        th = math.radians(th_deg)
        iv = T.neg_transfer_interval(th, s)
        ov = T.neg_transfer_over(th, s)
        # brute-force check of the roots against eq. (11)
        gg = np.linspace(0.001, 0.999, 200000)
        neg = gg[T.t_lt(gg, th, s * s) < 0]
        bf = (neg.min(), neg.max()) if len(neg) else None
        print("   theta = %2d deg: gamma < 1 negative on %s (brute force %s);  gamma > 1 negative beyond %.3f"
              % (th_deg,
                 "(%.4f, %.4f)" % iv if iv else "nowhere",
                 "(%.4f, %.4f)" % bf if bf else "nowhere", ov))
    print("   sigma -> 0: roots of g^2 - (1+cos^2 th) g + cos^2 th are 1 and cos^2 th; e.g. th = 60 deg -> (%.3f, %.3f)"
          % T.neg_transfer_interval(PI / 3, 1e-9))
    print("   sigma >= 1: negative transfer for gamma < 1 disappears at every theta <= pi/2: %s"
          % all(T.neg_transfer_interval(math.radians(t), 1.0) is None for t in range(0, 91, 5)))

    header("3c. Anomalous positive transfer at theta = pi/2 (orthogonal tasks), sigma = 0.2")
    th = PI / 2
    lo = 1 - s2
    hi = 1 + s2 / math.sin(th) ** 2
    print("   T_lt > 0 only for gamma in (%.2f, %.2f), the double-descent spike around gamma = 1;" % (lo, hi))
    print("   in between, R_sc > 1 = the risk of predicting zero. E.g. R_sc(0.99) = %.2f, R_sc(1.01) = %.2f."
          % (T.r_scratch(0.99, s2), T.r_scratch(1.01, s2)))

    header("3d. One pair of tasks, many sample sizes (theta = pi/4, sigma = 0.2)")
    th = PI / 4
    iv = T.neg_transfer_interval(th, s)
    print("   T_lt > 0 for gamma < %.4f, negative on (%.4f, %.4f), positive again on (%.4f, %.4f), negative beyond %.4f"
          % (iv[0], iv[0], iv[1], iv[1], T.neg_transfer_over(th, s), T.neg_transfer_over(th, s)))
    print("   The sign of T flips three times as n grows, while the two data distributions do not change.")

    header("3e. Noiseless case sigma = 0 (Appendix G Fig. 7)")
    for g in [0.5, 1.5, 3.0]:
        print("   gamma = %.1f: R_sc = %.3f;  T_lt(th=pi/4) = %+.3f   (sin^2 th = %.3f)" % (g, T.r_scratch(g, 0.0), T.t_lt(g, PI / 4, 0.0), 0.5))
    print("   For gamma > 1 scratch is exact, so T_lt = -sin^2(theta) < 0 at every theta > 0. For gamma < 1 the sign is that of (1-gamma) - sin^2 th.")
    print("   Negative on (cos^2 th, 1): at th = 60 deg, (0.25, 1).")

    header("3f. Fine-tuning, eq. (15)")
    print("   T_ft = (1-gamma)(2 cos th - 1). Zero crossing at cos th = 1/2, i.e. th = 60 deg = %.4f rad; independent of gamma and sigma." % (PI / 3))
    for g in [0.0, 0.25, 0.5, 0.9]:
        print("   gamma = %.2f:  T_ft(th=pi/8) = %+.3f   T_ft(th=pi/4) = %+.3f   T_ft(th=3pi/8) = %+.3f"
              % (g, T.t_ft(g, PI / 8), T.t_ft(g, PI / 4), T.t_ft(g, 3 * PI / 8)))
    print("   Null-space picture: scratch guesses 0 where the data are silent (error (1-g)), fine-tuning guesses beta_s")
    print("   (error (1-g)||beta_t - beta_s||^2 = (1-g)(2 - 2cos th)); the difference is (1-g)(2cos th - 1).")
    print("   The two are equal at 60 degrees because ||beta_t - beta_s|| = 2 sin(th/2) = 1 there: %.6f" % (2 * math.sin(PI / 6)))

    print("   With the balancedness factor c (notes, 'the null-space part carries a scale'): c^4 - (1-g)c^2 - g(1 + s2/(1-g)) = 0,")
    print("   T_ft = (1-g)(2c cos th - c^2), break-even cos th = c/2. c = 1 exactly when sigma = 0.")
    for g_, s_ in [(0.25, 0.2), (0.5, 0.2), (0.5, 0.5), (0.5, 1.0), (0.75, 1.0)]:
        c_ = float(T.c_balance(g_, s_ * s_))
        print("   gamma = %.2f, sigma = %.1f: c = %.4f, break-even angle %.1f deg (paper: 60.0), T_ft(th=pi/3) = %+.4f (paper: 0)"
              % (g_, s_, c_, math.degrees(math.acos(c_ / 2)), float(T.t_ft_balanced(g_, PI / 3, s_ * s_))))

    header("3g. Ridge linear transfer, eq. (13): a law of large numbers, no saddle point needed")
    rng = np.random.default_rng(3)
    n, th, s = 20000, PI / 4, 0.2
    print("   n = %d, theta = pi/4. As n grows, b = z.y/(|z|^2 + n lam) -> cos(th)/(1+lam), so the risk is" % n)
    print("   (b - cos th)^2 + sin^2 th = sin^2 th + cos^2 th (lam/(1+lam))^2 = 1 - cos^2 th (1+2lam)/(1+lam)^2.")
    for lam in [0.0, 0.1, 0.3, 0.5]:
        R = []
        for _ in range(40):
            z = rng.standard_normal(n)
            w = rng.standard_normal(n)
            y = math.cos(th) * z + math.sin(th) * w + s * rng.standard_normal(n)
            b = (z @ y) / (z @ z + n * lam)
            R.append((b - math.cos(th)) ** 2 + math.sin(th) ** 2)
        print("   lam = %.1f: MC %.4f   eq.(13) %.4f" % (lam, np.mean(R), T.r_ridge_limit(th, lam)))
    print("   The derivative of (1+2lam)/(1+lam)^2 is -2 lam/(1+lam)^3 < 0, so the risk rises with lam for every theta < pi/2.")
    print("   Ridge helps a *finite-n* linear probe only through the variance term (sigma^2 + sin^2 th)/(n-2), which is 0 in the limit.")

    header("3h. Fine-tuning with a finite source set, eqs. (22)-(23)")
    d, reps, s_t, s_s = 300, 150, 0.2, 0.3
    rng = np.random.default_rng(11)
    for gs, gt, th in [(0.5, 0.5, PI / 6), (0.5, 0.5, PI / 3), (0.25, 0.7, PI / 4), (2.0, 0.5, PI / 4)]:
        ns, nt = int(gs * d), int(gt * d)
        diffs = []
        for _ in range(reps):
            bs, bt = make_betas(rng, d, th)
            Xs = rng.standard_normal((ns, d)); ys = Xs @ bs + s_s * rng.standard_normal(ns)
            Xt = rng.standard_normal((nt, d)); yt = Xt @ bt + s_t * rng.standard_normal(nt)
            bhat = np.linalg.pinv(Xs) @ ys
            pinv = np.linalg.pinv(Xt)
            b_sc = pinv @ yt
            P = pinv @ Xt
            b_ft = b_sc + (np.eye(d) - P) @ bhat
            diffs.append(np.sum((b_sc - bt) ** 2) - np.sum((b_ft - bt) ** 2))
        print("   gs=%.2f gt=%.2f th=%.3f: T_ft MC %+.3f   eq.(22) %+.3f   cos th* = %.3f (cos th = %.3f)"
              % (gs, gt, th, np.mean(diffs), T.t_ft_finite_source(gs, gt, th, s_s ** 2),
                 T.cos_theta_star(gs, s_s ** 2), math.cos(th)))
    print("   Source noise raises the bar: at gs = 0.5, sigma_s = 0.3, fine-tuning needs cos th > %.3f (th < %.1f deg), not 0.5 (60 deg)."
          % (T.cos_theta_star(0.5, 0.09), math.degrees(math.acos(T.cos_theta_star(0.5, 0.09)))))

    header("3i. Beyond rank one (an extension, not in the paper): linear transfer on a k-dimensional feature space")
    print("   Same proof: regress y on k Gaussian features Z = X U. The part of the target outside the span acts as extra noise of variance rho = |P_perp beta_t|^2, and")
    print("   E R_lt = rho + (sigma^2 + rho) k/(n-k-1)   (inverse-Wishart mean; k = 1 recovers Theorem 3.7). d = 100, sigma = 0.3, 600 draws:")
    rng = np.random.default_rng(31)
    d, sg = 100, 0.3
    for k, n, rho in [(1, 40, 0.5), (5, 40, 0.5), (5, 120, 0.2), (20, 60, 0.3)]:
        R = []
        for _ in range(600):
            Q, _ = np.linalg.qr(rng.standard_normal((d, k + 1)))
            U, v = Q[:, :k], Q[:, k]
            a = rng.standard_normal(k); a /= np.linalg.norm(a)
            bt = math.sqrt(1 - rho) * (U @ a) + math.sqrt(rho) * v
            X = rng.standard_normal((n, d))
            y = X @ bt + sg * rng.standard_normal(n)
            w = np.linalg.lstsq(X @ U, y, rcond=None)[0]
            R.append(np.sum((U @ w - bt) ** 2))
        print("   k = %2d, n = %3d, rho = %.1f: MC %.4f ± %.4f   formula %.4f" % (k, n, rho, np.mean(R), np.std(R) / math.sqrt(600), rho + (sg ** 2 + rho) * k / (n - k - 1)))


# ------------------------------------------------------------------ figures
class PiFmt:
    """Tick labels 0, pi/6, pi/3, pi/2 for a numeric tick."""
    def format(self, x):
        k = round(x / (PI / 6))
        return {0: "0", 1: "π/6", 2: "π/3", 3: "π/2"}.get(k, "%.2f" % x)


def figures():
    from svgkit import Axes, legend, svg, write

    s = 0.2
    s2 = s * s

    def heat(ax, fn, nx, ny, cmax, xr, yr):
        """Diverging heat map: blue for v > 0, red for v < 0, opacity ~ |v|, so it works on light and dark pages.
        Vertical runs of equal (sign, quantised opacity) are merged into one rect to keep the file small."""
        out = []
        for i in range(nx):
            x0 = xr[0] + (xr[1] - xr[0]) * i / nx
            x1 = xr[0] + (xr[1] - xr[0]) * (i + 1) / nx
            run = None                                   # (cls, level, j_start)
            def flush(j_end):
                if run is None:
                    return
                cls, lev, j0 = run
                ya = yr[0] + (yr[1] - yr[0]) * j0 / ny
                yb = yr[0] + (yr[1] - yr[0]) * j_end / ny
                px0, px1 = ax.X(x0), ax.X(x1)
                out.append(f'<rect class="{cls}" opacity="{lev * 0.05:.2f}" x="{px0:.1f}" y="{ax.Y(yb):.1f}" '
                           f'width="{px1 - px0 + 0.3:.1f}" height="{ax.Y(ya) - ax.Y(yb) + 0.3:.1f}"/>')
            for j in range(ny):
                ymid = yr[0] + (yr[1] - yr[0]) * (j + 0.5) / ny
                v = float(fn((x0 + x1) / 2, ymid))
                if not math.isfinite(v):
                    v = cmax
                lev = int(round(min(abs(v) / cmax, 1.0) * 17))
                key = None if lev == 0 else ("bf" if v > 0 else "rf", lev, j)
                if run is not None and key is not None and key[0] == run[0] and key[1] == run[1]:
                    continue
                flush(j)
                run = key
            flush(ny)
        return "\n".join(out)

    # ---- Fig A: eq. (11), phase diagram and slices
    A1 = Axes(50, 40, 300, 250, (0, PI / 2), (0, 2))
    A2 = Axes(430, 40, 300, 250, (0, 2), (-1, 1))
    body = [f'<text class="hd" x="50" y="20">Eq. (11): linear-transfer phase diagram (σ = 0.2)</text>']
    body.append(heat(A1, lambda th, g: T.t_lt(g, th, s2), 60, 80, 1.0, (0, PI / 2), (0, 2)))
    # boundaries from the closed-form roots
    ths = np.linspace(math.acos(1 - s) + 1e-6, PI / 2, 200)
    lo = [T.neg_transfer_interval(t, s)[0] for t in ths]
    hi = [T.neg_transfer_interval(t, s)[1] for t in ths]
    body.append(A1.path(ths, lo, "k0"))
    body.append(A1.path(ths, hi, "k0"))
    ths2 = np.linspace(0.05, PI / 2, 400)
    ov = np.array([T.neg_transfer_over(t, s) for t in ths2])
    keep = ov <= 2.0
    body.append(A1.path(ths2[keep], ov[keep], "k0"))
    body.append(A1.frame([0, PI / 6, PI / 3, PI / 2], [0, 0.5, 1, 1.5, 2], "θ", "γ = n/d", xfmt=PiFmt()))
    body.append(A1.text(0.15, 1.75, "negative", "hd", dy=0))
    body.append(A1.text(0.15, 0.45, "positive", "hd"))
    body.append(A1.text(1.05, 0.15, "negative", "hd"))
    body.append(A1.text(0.05, 1.11, "anomalous positive (γ ≈ 1)", "tiny"))
    body.append(f'<text class="tiny" x="{A1.x0}" y="{A1.y0 + A1.h + 42}">blue T &gt; 0, red T &lt; 0; black: T = 0</text>')
    gg = np.concatenate([np.linspace(0.005, 0.98, 200), np.linspace(1.02, 2, 200)])
    body.append(A2.frame([0, 0.5, 1, 1.5, 2], [-1, -0.5, 0, 0.5, 1], "γ = n/d", "T_lt", yfmt="{:+.1f}"))
    body.append(f'<line class="k" x1="{A2.X(0)}" x2="{A2.X(2)}" y1="{A2.Y(0):.1f}" y2="{A2.Y(0):.1f}"/>')
    for th, cls, name in [(PI / 8, "b", "θ = π/8"), (PI / 4, "o", "θ = π/4"), (3 * PI / 8, "g", "θ = 3π/8")]:
        for seg in (gg[gg < 1], gg[gg > 1]):
            v = np.clip(T.t_lt(seg, th, s2), -1, 1)
            body.append(A2.path(seg, v, cls))
    body.append(legend(A2.x0 + 200, A2.y0 + 40, [("b", "θ = π/8"), ("o", "θ = π/4"), ("g", "θ = 3π/8")]))
    body.append(A2.text(0.94, 0.92, "→ ∞ at γ = 1", "tiny", anchor="end"))
    write("phase-linear.svg", svg(760, 340, "Linear-transfer phase diagram",
          "Heat map of eq. (11) over angle and sample ratio, with the three zero contours and slices at three angles.", body))

    # ---- Fig B: R_sc against the flat floor, why T falls with n
    B = Axes(50, 40, 420, 250, (0, 3), (0, 1.6))
    body = [f'<text class="hd" x="50" y="20">Why T falls with more target data: scratch keeps improving, transfer sits on a floor</text>']
    body.append(B.frame([0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 0.5, 1, 1.5], "γ = n/d", "test risk"))
    for seg in (np.linspace(0.002, 0.999, 600), np.linspace(1.001, 3, 600)):
        v = T.r_scratch(seg, s2)
        m = v <= 1.6
        body.append(B.path(seg[m], v[m], "b"))
    for th, cls, name in [(PI / 8, "o", "π/8"), (PI / 4, "g", "π/4"), (3 * PI / 8, "p", "3π/8")]:
        f = math.sin(th) ** 2
        body.append(f'<line class="{cls} dash" x1="{B.X(0)}" x2="{B.X(3)}" y1="{B.Y(f):.1f}" y2="{B.Y(f):.1f}"/>')
        body.append(B.text(3.0, f, f"sin²θ, θ = {name}", "tiny", anchor="end", dy=-3))
    body.append(B.text(1.0, 1.55, "scratch: double descent", "tiny", dx=8))
    body.append(legend(500, 60, [("b", "R_sc, eq. (8)"), ("o dash", "R_lt → sin²θ (n → ∞)")]))
    body.append('<text class="sub" x="500" y="120">T is the vertical gap:</text>')
    body.append('<text class="sub" x="500" y="136">scratch above the floor, T &gt; 0;</text>')
    body.append('<text class="sub" x="500" y="152">scratch below it, T &lt; 0.</text>')
    body.append('<text class="sub" x="500" y="176">The floor does not move with n;</text>')
    body.append('<text class="sub" x="500" y="192">the blue curve does.</text>')
    body.append('<text class="sub" x="500" y="222">Beyond γ = 1 scratch falls like σ²/(γ−1);</text>')
    body.append('<text class="sub" x="500" y="238">it crosses each floor and T turns negative.</text>')
    write("scratch-vs-floor.svg", svg(760, 330, "Scratch risk against the linear-transfer floor",
          "Eq. (8) for sigma = 0.2 with the floors sin^2 theta of eq. (10) for three angles.", body))

    # ---- Fig C: eq. (15) phase diagram and geometric picture
    C1 = Axes(50, 40, 300, 250, (0, PI / 2), (0, 2))
    C2 = Axes(430, 40, 300, 250, (0, 1), (-1.2, 1.2))
    body = [f'<text class="hd" x="50" y="20">Eq. (15): fine-tuning transferability</text>']
    body.append(heat(C1, lambda th, g: float(T.t_ft(g, th)), 60, 80, 1.0, (0, PI / 2), (0, 2)))
    body.append(f'<line class="k0" x1="{C1.X(PI / 3):.1f}" x2="{C1.X(PI / 3):.1f}" y1="{C1.Y(1):.1f}" y2="{C1.Y(0):.1f}"/>')
    body.append(f'<line class="k0" x1="{C1.X(0):.1f}" x2="{C1.X(PI / 2):.1f}" y1="{C1.Y(1):.1f}" y2="{C1.Y(1):.1f}"/>')
    body.append(C1.frame([0, PI / 6, PI / 3, PI / 2], [0, 0.5, 1, 1.5, 2], "θ", "γ = n/d", xfmt=PiFmt()))
    body.append(C1.text(PI / 6, 0.5, "positive", "hd", anchor="middle"))
    body.append(C1.text(PI / 3 + 0.25, 0.5, "negative", "hd", anchor="middle"))
    body.append(C1.text(PI / 4, 1.5, "T = 0 (no memory of the init)", "sub", anchor="middle"))
    body.append(C1.text(PI / 3, 1.06, "60°", "lab", anchor="middle"))
    body.append(C2.frame([0, 0.25, 0.5, 0.75, 1.0], [-1, -0.5, 0, 0.5, 1], "γ = n/d", "T_ft", yfmt="{:+.1f}"))
    body.append(f'<line class="k" x1="{C2.X(0)}" x2="{C2.X(1)}" y1="{C2.Y(0):.1f}" y2="{C2.Y(0):.1f}"/>')
    gg = np.linspace(0, 1, 50)
    for th, cls in [(PI / 8, "b"), (PI / 4, "o"), (3 * PI / 8, "g"), (PI / 2, "r")]:
        body.append(C2.path(gg, T.t_ft(gg, th), cls))
    body.append(legend(C2.x0 + 150, C2.y0 + 12, [("b", "θ = π/8"), ("o", "θ = π/4"), ("g", "θ = 3π/8"), ("r", "θ = π/2")]))
    write("phase-finetune.svg", svg(760, 340, "Fine-tuning transferability",
          "Heat map of eq. (15) and slices; the sign is decided by cos theta against one half, independent of sample size.", body))

    ang = np.linspace(0, 2 * PI, 200)

    # ---- Fig D: geometry of the null-space guess, exact
    G = Axes(30, 34, 330, 250, (-1.35, 2.15), (-1.3, 1.3))
    body = ['<text class="hd" x="30" y="20">Where the data are silent, scratch guesses 0; fine-tuning guesses β_s</text>']
    body.append(G.path(1 + np.cos(ang), np.sin(ang), "k"))
    body.append(G.path(np.cos(ang), np.sin(ang), "gd"))
    body.append(f'<line class="ax" x1="{G.X(-1.3):.1f}" x2="{G.X(2.1):.1f}" y1="{G.Y(0):.1f}" y2="{G.Y(0):.1f}"/>')
    body.append(G.dot(1, 0, "bf", 5))
    body.append(G.text(1, 0, "β_t", "hd", dx=8, dy=16))
    body.append(G.dot(0, 0, "rf", 5))
    body.append(G.text(0, 0, "0  (scratch)", "lab", anchor="middle", dy=17))
    for th_deg, cls, nm in [(40, "gf", "θ = 40°: β_s inside the circle, fine-tuning wins"),
                            (80, "of", "θ = 80°: β_s outside, scratch wins")]:
        th = math.radians(th_deg)
        x, y = math.cos(th), math.sin(th)
        body.append(G.dot(x, y, cls, 5))
        body.append(f'<line class="{cls[0]} dot" x1="{G.X(1):.1f}" x2="{G.X(x):.1f}" y1="{G.Y(0):.1f}" y2="{G.Y(y):.1f}"/>')
        body.append(G.text(x, y, f"β_s, {th_deg}°", "lab", dx=8, dy=-4))
    body.append(G.text(1.0, 0.0, "", "tiny"))
    body.append('<text class="sub" x="390" y="80">distance from β_t to a guess:</text>')
    body.append('<text class="sub" x="390" y="100">scratch: ‖β_t − 0‖ = 1</text>')
    body.append('<text class="sub" x="390" y="118">fine-tune: ‖β_t − β_s‖ = 2 sin(θ/2)</text>')
    body.append('<text class="sub" x="390" y="148">equal at θ = 60°  (2 sin 30° = 1)</text>')
    body.append('<text class="sub" x="390" y="182">Both are then scaled by the (1 − γ) of</text>')
    body.append('<text class="sub" x="390" y="200">directions the data cannot see:</text>')
    body.append('<text class="ink" x="390" y="226" style="font-size:13px">T_ft = (1−γ)(1 − 4 sin²(θ/2))</text>')
    body.append('<text class="tiny" x="390" y="244">and 1 − 4 sin²(θ/2) = 2 cos θ − 1.</text>')
    write("null-space-guess.svg", svg(760, 320, "Null-space guess",
          "Unit circle geometry behind eq. (15): fine-tuning wins when the source vector is closer to the target than the origin is.", body))


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        figures()
        sys.exit(0)
    monte_carlo()
    gd_check()
    noisy_finetune()
    algebra()
    if "--figures" in sys.argv:
        figures()
