#!/usr/bin/env python3
"""Section 5 of Achille, Paolini, Mbeng & Soatto: the information in the weights.

Checked here:

  1. Theorem 5.4. With P = N(0, lam^2 I) and Q = N(w*, Sigma), the minimiser is
     Sigma* = beta (H + beta/lam^2 I)^{-1} (checked by perturbation), and the
     KL formula is right (checked by Monte Carlo).
  2. The finite-lam value, which the theorem's lam -> infinity limit hides:
        C_beta* = L(w*) + beta |w*|^2 / (2 lam^2) + (beta/2) log det(I + lam^2 H / beta)
        KL*     = |w*|^2 / (2 lam^2) + (1/2) sum_j g(lam^2 h_j / beta),
        g(s)    = log(1 + s) - s / (1 + s) >= 0,  g(0) = 0.
     A flat direction (h_j = 0) carries exactly zero information; the theorem's
     (1/2) log|F| is -infinity as soon as one eigenvalue of F is zero.
  3. What the theorem's "O(1)" contains: (k/2) log(N / beta) - k/2. For an
     AllCNN-sized k this is millions of nats, and it moves with beta and N.
  4. Minimising E_Q L + beta KL(Q||P) over *all* Q gives the Gibbs posterior
     Q ~ P exp(-L/beta) and the value -beta log E_P exp(-L/beta): a free
     energy; at beta = 1 the negative log marginal likelihood (the evidence).
     Exact for linear-Gaussian regression; an upper bound for Gaussian Q in
     logistic regression, with the gap measured.
  5. "H = N F because w* is a critical point": true for logistic regression at
     every w (canonical link), false in general: the Hessian carries an extra
     sum_i (p_i - y_i) Hess z_i that a critical point does not remove. A
     two-parameter tanh model, well specified and misspecified: at the minimum
     H and the Fisher differ, most in the low-curvature direction.
  6. Proposition 5.3 on a discrete toy: E_D KL(Q(w|D) || P) is minimised by the
     marginal and equals I(w; D) there; any other P pays KL(marginal || P).
  7. Theorem 5.5 (PAC-Bayes). The bound needs beta > 1/2 (after rescaling the
     loss to [0, 1]: beta > L_max / 2), and it diverges as lam -> infinity, the
     limit Theorem 5.4 is stated in. Logistic regression, Laplace Q.

Standard library and numpy only.

Run:  python3 gaussian_information.py             (prints every number quoted in the notes)
      python3 gaussian_information.py --figures   (also rewrites ../figures/information-per-direction.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np

rng = np.random.default_rng(1)


def logdet(A):
    s, v = np.linalg.slogdet(A)
    assert s > 0
    return v


def kl_gauss(mu, S, lam):
    """KL( N(mu, S) || N(0, lam^2 I) )."""
    k = len(mu)
    return 0.5 * (mu @ mu / lam**2 + np.trace(S) / lam**2 + k * math.log(lam**2) - logdet(S) - k)


def g(s):
    return np.log1p(s) - s / (1 + s)


# ----------------------------------------------------------------------- 1
def check_theorem_54():
    print("1. Theorem 5.4: the optimal covariance and the KL formula")
    N, k, lam, beta = 1000, 3, 10.0, 0.5
    Qo, _ = np.linalg.qr(rng.standard_normal((k, k)))
    F = Qo @ np.diag([2.0, 0.3, 0.004]) @ Qo.T
    H = N * F
    w = rng.standard_normal(k)

    def f(S):   # C_beta minus L(w*), quadratic approximation of the loss
        return 0.5 * np.trace(H @ S) + beta * kl_gauss(w, S, lam)

    S_star = beta * np.linalg.inv(H + beta / lam**2 * np.eye(k))
    f0 = f(S_star)
    worst = np.inf
    for _ in range(2000):
        E = rng.standard_normal((k, k)); E = (E + E.T) / 2
        E *= 1e-3 * np.linalg.norm(S_star) / np.linalg.norm(E)
        worst = min(worst, f(S_star + E) - f0)
    print(f"   N = {N}, k = {k}, lam = {lam}, beta = {beta}; eig(F) = 2, 0.3, 0.004")
    print(f"   2000 random perturbations of Sigma*: smallest increase of C_beta = {worst:.2e} (> 0: a minimum)")
    S_paper = beta * np.linalg.inv(H)
    rel = np.linalg.norm(S_paper - S_star) / np.linalg.norm(S_star)
    print(f"   beta H^-1 vs Sigma*: relative difference {rel:.3f} at lam = {lam:g} (the 0.004 direction: "
          f"lam^2 h / beta = {lam**2 * N * 0.004 / beta:.0f})")
    # Monte Carlo check of the KL formula
    L = np.linalg.cholesky(S_star)
    z = rng.standard_normal((200_000, k))
    ws = w + z @ L.T
    logq = -0.5 * (z**2).sum(1) - 0.5 * logdet(2 * math.pi * S_star)
    logp = -0.5 * (ws**2).sum(1) / lam**2 - 0.5 * k * math.log(2 * math.pi * lam**2)
    print(f"   KL formula {kl_gauss(w, S_star, lam):.4f}  vs  Monte Carlo {np.mean(logq - logp):.4f}")


# ----------------------------------------------------------------------- 2-3
def check_closed_form():
    print("\n2. The finite-lam closed form, and flat directions")
    N, k, beta = 1000, 4, 1.0
    h = N * np.array([2.0, 0.3, 0.004, 0.0])        # one exactly flat direction
    H = np.diag(h)
    w = np.array([0.5, -1.0, 2.0, 0.0])
    for lam in (1.0, 10.0):
        S = beta * np.linalg.inv(H + beta / lam**2 * np.eye(k))
        direct = 0.5 * np.trace(H @ S) + beta * kl_gauss(w, S, lam)
        closed = beta * w @ w / (2 * lam**2) + 0.5 * beta * logdet(np.eye(k) + lam**2 * H / beta)
        kl_closed = w @ w / (2 * lam**2) + 0.5 * g(lam**2 * h / beta).sum()
        print(f"   lam = {lam:4g}: C_beta - L(w*) direct {direct:.6f}, closed form {closed:.6f}; "
              f"KL {kl_gauss(w, S, lam):.6f} vs {kl_closed:.6f}")
        print(f"             per-direction information (1/2) g(lam^2 h/beta): "
              + ", ".join(f"{0.5 * v:.3f}" for v in g(lam**2 * h / beta)))
    print("   the paper's (1/2) log|F| with eig(F) = 2, 0.3, 0.004, 0: -infinity")

    print("\n3. What Theorem 5.4's O(1) hides: KL* - [(1/2) log|F| + (k/2) log lam^2]")
    k, beta = 3, 0.5
    fe = np.array([2.0, 0.3, 0.004])
    pred = 0.5 * k * math.log(N / beta) - k / 2
    for lam in (10.0, 100.0, 1e3, 1e4):
        kl = 0.5 * g(lam**2 * N * fe / beta).sum()     # w* = 0 for clarity
        paper = 0.5 * np.log(fe).sum() + 0.5 * k * math.log(lam**2)
        print(f"   lam = {lam:7g}: remainder {kl - paper:8.4f}   (limit (k/2) log(N/beta) - k/2 = {pred:.4f})")
    kA, NA = 1.4e6, 50_000
    print(f"   AllCNN-sized: k = {kA:.1e}, N = {NA}: (k/2) log N - k/2 = {0.5 * kA * math.log(NA) - kA / 2:.3e} nats at beta = 1")
    print(f"   sweeping beta from 1 to 1e-3 adds (k/2) log 1000 = {0.5 * kA * math.log(1000):.3e} nats with H unchanged")
    print(f"   doubling N adds (k/2) log 2 = {0.5 * kA * math.log(2):.3e} nats")


# ----------------------------------------------------------------------- 4
def check_free_energy():
    print("\n4. min over all Q of E_Q L + beta KL(Q||P) = -beta log E_P exp(-L/beta)")
    # (a) linear-Gaussian regression: exact
    n, k, sig, lam = 40, 5, 0.5, 2.0
    X = rng.standard_normal((n, k))
    y = X @ rng.standard_normal(k) + sig * rng.standard_normal(n)
    H = X.T @ X / sig**2
    b = X.T @ y / sig**2
    c = y @ y / (2 * sig**2) + 0.5 * n * math.log(2 * math.pi * sig**2)   # L(w) = w'Hw/2 - b'w + c
    for beta in (0.5, 1.0, 2.0):
        A = H + beta / lam**2 * np.eye(k)
        mu = np.linalg.solve(A, b)
        S = beta * np.linalg.inv(A)
        EL = 0.5 * mu @ H @ mu - b @ mu + c + 0.5 * np.trace(H @ S)
        gauss_opt = EL + beta * kl_gauss(mu, S, lam)
        A2 = H / beta + np.eye(k) / lam**2
        free = c + 0.5 * beta * logdet(lam**2 * A2) - 0.5 / beta * b @ np.linalg.solve(A2, b)
        line = f"   linear-Gaussian, beta = {beta}: best Gaussian Q {gauss_opt:.6f}, free energy {free:.6f}"
        if beta == 1.0:
            Cy = sig**2 * np.eye(n) + lam**2 * X @ X.T
            ev = 0.5 * y @ np.linalg.solve(Cy, y) + 0.5 * logdet(2 * math.pi * Cy)
            line += f", -log evidence {ev:.6f}"
        print(line)
    # (b) one-weight logistic regression: Gaussian Q is an upper bound
    n = 30
    x = rng.standard_normal(n)
    yy = np.where(rng.random(n) < 1 / (1 + np.exp(-1.5 * x)), 1.0, -1.0)
    lam = 3.0
    grid = np.linspace(-15, 15, 30001)
    dw = grid[1] - grid[0]
    Lg = np.logaddexp(0, -np.outer(grid, yy * x)).sum(1)
    logP = -0.5 * grid**2 / lam**2 - 0.5 * math.log(2 * math.pi * lam**2)
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60)
    gh_w = gh_w / gh_w.sum()
    for beta in (0.5, 1.0, 2.0):
        a = logP - Lg / beta
        m = a.max()
        free = -beta * (m + math.log(np.exp(a - m).sum() * dw))
        best = np.inf
        for mu in np.linspace(-1, 4, 251):
            for ls in np.linspace(-3, 1, 161):
                s = math.exp(ls)
                wq = mu + s * gh_x
                EL = (np.logaddexp(0, -np.outer(wq, yy * x)).sum(1) * gh_w).sum()
                kl = math.log(lam / s) + (s**2 + mu**2) / (2 * lam**2) - 0.5
                best = min(best, EL + beta * kl)
        print(f"   logistic, n = {n}, beta = {beta}: best Gaussian Q {best:.4f} >= free energy {free:.4f} "
              f"(gap {best - free:.4f} = beta KL(Q || Gibbs))")


# ----------------------------------------------------------------------- 5
def check_hessian_fisher():
    print("\n5. Is the Hessian at a minimum N times the Fisher?")
    n = 400
    x = 2 * rng.standard_normal(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-3 * np.sin(2 * x)))).astype(float)

    # (a) logistic regression: H(w) = sum p(1-p) x x' whatever y is
    Xl = np.c_[x, np.ones(n)]
    wl = np.array([0.7, -0.2])
    p = 1 / (1 + np.exp(-Xl @ wl))
    Hl = (Xl * (p * (1 - p))[:, None]).T @ Xl
    eps = 1e-5
    def nll(w):
        return (np.logaddexp(0, Xl @ w) - y * (Xl @ w)).sum()
    Hn = np.array([[(nll(wl + eps * (np.eye(2)[i] + np.eye(2)[j])) - nll(wl + eps * np.eye(2)[i])
                     - nll(wl + eps * np.eye(2)[j]) + nll(wl)) / eps**2 for j in range(2)] for i in range(2)])
    print(f"   logistic regression at an arbitrary w: |H_numeric - Fisher| / |Fisher| = "
          f"{np.linalg.norm(Hn - Hl) / np.linalg.norm(Hl):.1e}")

    # (b) z = a tanh(b x): fit, then compare, once well specified and once not
    def fit_tanh(truth):
        xx = 2 * rng.standard_normal(n)
        yy = (rng.random(n) < 1 / (1 + np.exp(-truth(xx)))).astype(float)

        def parts(th):
            a, bb = th
            t = np.tanh(bb * xx); s2 = 1 - t**2
            z = a * t
            pz = 1 / (1 + np.exp(-z))
            J = np.c_[t, a * xx * s2]
            r = pz - yy
            G = (J * (pz * (1 - pz))[:, None]).T @ J          # Fisher = Gauss-Newton
            d2 = np.zeros((n, 2, 2))
            d2[:, 0, 1] = d2[:, 1, 0] = xx * s2
            d2[:, 1, 1] = -2 * a * xx**2 * s2 * t
            Hs = G + np.einsum("i,ijk->jk", r, d2)             # + sum_i (p_i - y_i) Hess z_i
            return (np.logaddexp(0, z) - yy * z).sum(), J.T @ r, Hs, G

        best = None
        for a0 in (0.5, 1.0, 2.0, 4.0):                        # multi-start Newton with backtracking
            for b0 in (0.2, 0.5, 1.0, 2.0):
                th = np.array([a0, b0])
                for _ in range(300):
                    loss, gr, Hs, _ = parts(th)
                    step = np.linalg.solve(Hs, gr) if np.all(np.linalg.eigvalsh(Hs) > 0) else 0.01 * gr
                    t = 1.0
                    while parts(th - t * step)[0] > loss and t > 1e-10:
                        t /= 2
                    th = th - t * step
                cand = (parts(th), th)
                if best is None or cand[0][0] < best[0][0]:
                    best = cand
        return best

    for name, truth in (("well specified, logit 3 tanh(0.7x)", lambda u: 3 * np.tanh(0.7 * u)),
                        ("misspecified, logit 2x exp(-x^2/8)", lambda u: 2 * u * np.exp(-u**2 / 8))):
        (loss, gr, Hs, G), th = fit_tanh(truth)
        eh, ef = np.linalg.eigvalsh(Hs), np.linalg.eigvalsh(G)
        print(f"   z = a tanh(bx), n = {n}, {name}: minimum (a, b) = ({th[0]:.3f}, {th[1]:.3f}), |grad| = {np.linalg.norm(gr):.0e}")
        print(f"     eig(Hessian) = {eh[0]:.2f}, {eh[1]:.2f};  eig(Fisher) = {ef[0]:.2f}, {ef[1]:.2f};  "
              f"|H - F| / |F| = {np.linalg.norm(Hs - G) / np.linalg.norm(G):.3f};  "
              f"(1/2)(log det H - log det F) = {0.5 * (logdet(Hs) - logdet(G)):.3f} nats")


# ----------------------------------------------------------------------- 6
def check_prop_53():
    print("\n6. Proposition 5.3 on a discrete toy (4 datasets, 6 weight values)")
    piD = rng.dirichlet(np.ones(4))
    Q = rng.dirichlet(0.5 * np.ones(6), size=4)
    marg = piD @ Q
    def ekl(P):
        return float((piD[:, None] * Q * np.log(Q / P)).sum())
    joint = piD[:, None] * Q
    I = float((joint * np.log(joint / (piD[:, None] * marg[None, :]))).sum())
    print(f"   E_D KL(Q || marginal) = {ekl(marg):.6f},  I(w; D) from the joint = {I:.6f}")
    for _ in range(3):
        P = rng.dirichlet(np.ones(6))
        kl_mp = float((marg * np.log(marg / P)).sum())
        print(f"   another P: E_D KL = {ekl(P):.6f} = I + KL(marginal || P) = {I + kl_mp:.6f}")


# ----------------------------------------------------------------------- 7
def check_pac_bayes():
    print("\n7. Theorem 5.5 (PAC-Bayes, McAllester's Theorem 2) against lam")
    print("   rescaling a loss bounded by L_max to [0, 1] turns beta > 1/2 into beta > L_max / 2:")
    for name, Lm in (("0-1 loss", 1.0), ("CE clipped at ln 10", math.log(10)), ("CE clipped at ln 100", math.log(100))):
        print(f"     {name:22s} L_max = {Lm:.2f}  ->  beta > {Lm / 2:.2f}")
    n, d, ntest, delta, beta = 500, 10, 20000, 0.05, 1.0
    wt = rng.standard_normal(d)
    def sample(m):
        X = rng.standard_normal((m, d))
        yy = np.where(rng.random(m) < 1 / (1 + np.exp(-X @ wt)), 1.0, -1.0)
        return X, yy
    X, yy = sample(n)
    Xt, yt = sample(ntest)
    print(f"   logistic regression, d = {d}, n = {n}, beta = {beta}, delta = {delta}; Q = Laplace at the MAP")
    print("       lam      KL(Q||P)   train Gibbs err   test Gibbs err   bound")
    out = []
    for lam in (0.3, 1.0, 3.0, 10.0, 100.0, 1e4, 1e8):
        w = np.zeros(d)
        for _ in range(100):
            p = 1 / (1 + np.exp(-yy * (X @ w)))
            gr = -(X * ((1 - p) * yy)[:, None]).sum(0) + w / lam**2
            Hh = (X * (p * (1 - p))[:, None]).T @ X + np.eye(d) / lam**2
            w = w - np.linalg.solve(Hh, gr)
        p = 1 / (1 + np.exp(-yy * (X @ w)))
        Hh = (X * (p * (1 - p))[:, None]).T @ X
        S = beta * np.linalg.inv(Hh + beta / lam**2 * np.eye(d))
        kl = kl_gauss(w, S, lam)
        W = w + rng.standard_normal((400, d)) @ np.linalg.cholesky(S).T
        tr = float(np.mean(np.sign(X @ W.T) != yy[:, None]))
        te = float(np.mean(np.sign(Xt @ W.T) != yt[:, None]))
        bound = (tr + beta * (kl + math.log(1 / delta)) / n) / (1 - 1 / (2 * beta))
        out.append((lam, kl, tr, te, bound))
        print(f"   {lam:8g}   {kl:8.2f}   {tr:15.3f}   {te:14.3f}   {bound:6.3f}")
    return out


# ----------------------------------------------------------------- figures
def figures():
    from svgkit import Axes, legend, svg, write
    body = ['<text class="hd" x="20" y="22">Information carried by one weight direction, at the optimal Gaussian Q</text>',
            '<text class="sub" x="20" y="38">s = λ² h / β: prior variance times curvature over β. Theorem 5.4 keeps only the large-s asymptote.</text>']
    ax = Axes(62, 66, 560, 230, (1e-3, 1e5), (-2, 6), logx=True)
    body.append(ax.frame([1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5], [-2, 0, 2, 4, 6],
                         "s = λ² h / β   (log scale)", "nats", xfmt="{:.0e}"))
    s = np.geomspace(1e-3, 1e5, 400)
    body.append(ax.path(s, 0.5 * g(s), "b"))
    keep = 0.5 * np.log(s) - 0.5 >= -2
    body.append(ax.path(s[keep], (0.5 * np.log(s) - 0.5)[keep], "o dash"))
    body.append(f'<line class="k" x1="{ax.x0}" x2="{ax.x0 + ax.w}" y1="{ax.Y(0):.1f}" y2="{ax.Y(0):.1f}"/>')
    body.append(legend(90, 90, [("b", "exact: ½ [log(1+s) − s/(1+s)]  → 0 for flat directions"),
                                ("o dash", "λ → ∞ form: ½ log s − ½  → −∞ for flat directions")]))
    body.append(ax.text(1e-2, 0.25, "a flat direction (h = 0) costs nothing", "tiny"))
    body.append('<text class="lab" x="20" y="340">Summing the blue curve over the eigenvalues of H gives the information in the weights, finite for any λ.</text>')
    body.append('<text class="lab" x="20" y="355">Summing the orange one gives ½ log|F| + ½ k log λ² + ½ k log(N/β) − k/2: the last two terms are the theorem’s “O(1)”.</text>')
    write("information-per-direction.svg", svg(680, 368, "Information per weight direction",
                                               "The exact per-direction information at the optimal Gaussian post-distribution, "
                                               "against the large-lambda asymptote used by Theorem 5.4.", body))


if __name__ == "__main__":
    check_theorem_54()
    check_closed_form()
    check_free_energy()
    check_hessian_fisher()
    check_prop_53()
    check_pac_bayes()
    if "--figures" in sys.argv:
        figures()
