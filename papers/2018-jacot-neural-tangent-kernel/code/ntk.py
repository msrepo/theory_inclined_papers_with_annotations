#!/usr/bin/env python3
"""The Neural Tangent Kernel, from scratch, in numpy.

Checks the two central claims of Jacot, Gabriel & Hongler (NeurIPS 2018):

  Theorem 1  the empirical NTK converges to a deterministic limit as width grows
  Theorem 2  that kernel stays put during training

Uses the paper's own parametrisation (Section 2), which is the thing that makes
the limit exist:

    a~(l+1) = (1/sqrt(n_l)) W(l) a(l) + beta b(l),   W, b ~ N(0, 1)
    a(l)    = relu(a~(l)),    f(x) = a~(L)   (the output is a preactivation)

Only numpy: the backward pass is written out rather than autodiffed, which is
short here and makes the NTK assembly explicit.

Run:  python3 ntk.py
"""
from __future__ import annotations

import numpy as np

BETA = 0.1                      # their beta, balancing bias against weights


# --------------------------------------------------------------- the network
def init(dims, rng):
    return [(rng.standard_normal((dims[i+1], dims[i])), rng.standard_normal(dims[i+1]))
            for i in range(len(dims)-1)]


def forward(params, X, dims):
    """X: (B, n0) -> scalar output (B,), plus activations and preactivations."""
    a = X                                           # a(0) = x, no nonlinearity here
    acts, pre = [a], []
    for l, (W, b) in enumerate(params):
        z = a @ W.T / np.sqrt(dims[l]) + BETA * b
        pre.append(z)
        a = np.maximum(z, 0.0) if l < len(params)-1 else z
        acts.append(a)
    return pre[-1][:, 0], acts, pre


def empirical_ntk(params, X, dims):
    """Theta(x,x') = <grad_theta f(x), grad_theta f(x')>.

    Assembled per layer from the backprop deltas, so the Jacobian is never
    materialised. For layer l the weight block contributes
    (delta.delta') * (a(l).a(l)') / n_l and the bias block beta^2 * (delta.delta').
    """
    B = X.shape[0]
    _, acts, pre = forward(params, X, dims)
    L = len(params)
    deltas = [None]*L
    d = np.ones((B, dims[-1]))                      # df / d a~(L) = 1
    deltas[L-1] = d
    for l in range(L-1, 0, -1):
        d = (d @ params[l][0]) / np.sqrt(dims[l]) * (pre[l-1] > 0)
        deltas[l-1] = d
    K = np.zeros((B, B))
    for l in range(L):
        G = deltas[l] @ deltas[l].T
        K += G * (acts[l] @ acts[l].T) / dims[l] + BETA**2 * G
    return K


def grads(params, X, y, dims):
    """Gradients of C = (1/2N) sum (f - y)^2, for plain gradient descent."""
    f, acts, pre = forward(params, X, dims)
    N, L = X.shape[0], len(params)
    d = ((f - y)/N)[:, None]
    gs = [None]*L
    for l in range(L-1, -1, -1):
        gs[l] = (d.T @ acts[l] / np.sqrt(dims[l]), BETA * d.sum(0))
        if l > 0:
            d = (d @ params[l][0]) / np.sqrt(dims[l]) * (pre[l-1] > 0)
    return gs, f


# ------------------------------------------------- the infinite-width limit
def relu_moments(a, b, c):
    """For (u,v) ~ N(0, [[a,c],[c,b]]): E[relu(u)relu(v)] and E[relu'(u)relu'(v)].

    The arccos kernels. These are what make Theorem 1's recursion computable in
    closed form for relu.
    """
    ab = np.sqrt(np.clip(a*b, 1e-30, None))
    th = np.arccos(np.clip(c/ab, -1.0, 1.0))
    return ab/(2*np.pi)*(np.sin(th) + (np.pi-th)*np.cos(th)), (np.pi-th)/(2*np.pi)


def limiting_ntk(X, depth, n0):
    """Theorem 1:  Theta(1) = Sigma(1),  Theta(L+1) = Theta(L) Sigmadot(L+1) + Sigma(L+1)."""
    S = X @ X.T / n0 + BETA**2                      # Sigma(1)
    T = S.copy()                                    # Theta(1)
    for _ in range(depth-1):
        d = np.diag(S).copy()
        Snew, Sdot = relu_moments(d[:, None], d[None, :], S)
        Snew = Snew + BETA**2
        T = T*Sdot + Snew
        S = Snew
    return T


# ------------------------------------------------------------------ checks
def check_convergence(depth=4, n0=2, seeds=10):
    g = np.linspace(-np.pi, np.pi, 41)
    X = np.stack([np.cos(g), np.sin(g)], 1)
    lim = limiting_ntk(X, depth, n0)
    print("Theorem 1 - empirical NTK vs the limiting recursion")
    print(f"{'width':>8} {'mean rel err':>14} {'std over seeds':>16}")
    for n in (50, 200, 1000, 5000):
        dims = [n0] + [n]*(depth-1) + [1]
        errs = [np.abs(empirical_ntk(init(dims, np.random.default_rng(100+s)), X, dims)[0]
                       - lim[0]).mean()/np.abs(lim[0]).mean() for s in range(seeds)]
        print(f"{n:>8} {np.mean(errs):>14.4f} {np.std(errs):>16.4f}")
    print("  -> decays like 1/sqrt(n)\n")


def check_constancy(depth=4, n0=2, steps=200, lr=1.0, seeds=5):
    print("Theorem 2 - does the kernel move while the network learns?")
    print(f"{'width':>8} {'||K(T)-K(0)||/||K(0)||':>24} {'loss 0':>9} {'loss T':>10}")
    g = np.linspace(-np.pi, np.pi, 41)
    Xp = np.stack([np.cos(g), np.sin(g)], 1)
    for n in (50, 200, 1000, 4000):
        dims = [n0] + [n]*(depth-1) + [1]
        out = []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            p = [list(t) for t in init(dims, rng)]
            Xtr = rng.standard_normal((20, n0))
            Xtr /= np.linalg.norm(Xtr, axis=1, keepdims=True)
            ytr = Xtr[:, 0]*Xtr[:, 1]               # their f*(x) = x1 x2
            K0 = empirical_ntk(p, Xp, dims)
            l0 = None
            for _ in range(steps):
                gs, f = grads(p, Xtr, ytr, dims)
                if l0 is None:
                    l0 = 0.5*np.mean((f-ytr)**2)
                for l, (gW, gb) in enumerate(gs):
                    p[l][0] -= lr*gW
                    p[l][1] -= lr*gb
            lT = 0.5*np.mean((forward(p, Xtr, dims)[0]-ytr)**2)
            KT = empirical_ntk(p, Xp, dims)
            out.append((np.linalg.norm(KT-K0)/np.linalg.norm(K0), l0, lT))
        d, l0, lT = np.mean(out, 0)
        print(f"{n:>8} {d:>24.4f} {l0:>9.4f} {lT:>10.5f}")
    print("  -> drift decays faster than the initialisation error (roughly 1/n),")
    print("     and the loss falls just as far at every width: the wide network")
    print("     learns while its kernel is effectively frozen.\n")


if __name__ == "__main__":
    check_convergence()
    check_constancy()
