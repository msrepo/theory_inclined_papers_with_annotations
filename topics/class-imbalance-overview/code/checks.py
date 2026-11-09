"""Numeric checks for the class-imbalance notes (numpy + stdlib only).

  python3 checks.py

1. Effective number of samples (Cui 2019): recursion == closed form, and its limits.
2. LDAM margin allocation (Cao 2019): gamma_j ~ n_j^(-1/4) minimises sum_j n_j^(-1/2)/gamma_j.
3. Logit adjustment (Menon 2021): on two Gaussians, argmax_y [log p(y|x) - log pi_y] maximises
   balanced accuracy; plain argmax p(y|x) does not.
"""
import math
import numpy as np

# 1 ---------------------------------------------------------------------------------------
beta = 0.99
E = 0.0
for n in range(1, 501):
    E = 1.0 + beta * E            # E_n = 1 + beta * E_{n-1}   (new point is "new" w.p. beta^(n-1))
    if n in (1, 10, 100, 500):
        closed = (1 - beta**n) / (1 - beta)
        print(f"beta={beta} n={n:4d} recursion={E:8.3f} closed form={closed:8.3f}")
        assert abs(E - closed) < 1e-9
print("beta->0 gives E_n =", (1 - 1e-12**100) / (1 - 1e-12), "; beta->1 gives E_n ~ n:",
      (1 - (1 - 1e-9)**100) / 1e-9)
ns = np.array([5000, 500, 50])
for b in (0.9, 0.99, 0.999, 0.9999):
    w = (1 - b) / (1 - b**ns); w = w / w.sum() * len(ns)
    print(f"class-balanced weights, beta={b}: {np.round(w, 3)}   (counts {ns.tolist()})")

# 2 ---------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
def cost(g): return float(np.sum(ns ** -0.5 / g))
g_star = ns ** -0.25; g_star /= g_star.sum()          # budget sum(gamma) = 1
best_random = min(cost(g) for g in rng.dirichlet(np.ones(3), 200_000))
print(f"LDAM: gamma*={np.round(g_star, 4)} cost={cost(g_star):.4f}; best of 200k random={best_random:.4f}")
assert cost(g_star) <= best_random
print("ratio gamma_minority/gamma_majority =", round(g_star[2] / g_star[0], 3), "= (5000/50)^(1/4) =", round(100 ** 0.25, 3))

# 3 ---------------------------------------------------------------------------------------
pi = 0.05                                         # P(y=+1), minority; x|y ~ N(y*mu, 1)
mu = 1.0
def Phi(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))
def balanced_acc(t):                              # predict +1 iff x > t
    return 0.5 * ((1 - Phi(t - mu)) + Phi(t + mu))
# plain Bayes: p(+|x) > 1/2  <=>  2*mu*x + log(pi/(1-pi)) > 0
t_plain = -math.log(pi / (1 - pi)) / (2 * mu)
# logit-adjusted: log p(+|x) - log pi > log p(-|x) - log(1-pi)  <=>  2*mu*x > 0
t_adj = 0.0
grid = np.linspace(-6, 6, 120001)
t_best = grid[np.argmax([balanced_acc(t) for t in grid[::10]]) * 10]
print(f"threshold plain={t_plain:.3f} adjusted={t_adj:.3f} brute-force best balanced accuracy={t_best:.3f}")
print(f"balanced accuracy plain={balanced_acc(t_plain):.4f} adjusted={balanced_acc(t_adj):.4f}")
print(f"minority recall plain={1 - Phi(t_plain - mu):.4f} adjusted={1 - Phi(t_adj - mu):.4f}")
assert abs(t_best - t_adj) < 1e-2
