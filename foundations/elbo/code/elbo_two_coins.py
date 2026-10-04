"""Two-coin example for the ELBO discussion (standard library only).

z = which coin (0: fair, 1: biased), x = heads.
Checks: marginal, posterior, Jensen's inequality, and log p(x) = ELBO + KL.
"""
from math import log

prior = [0.5, 0.5]            # p(z)
p_heads = [0.5, 0.9]          # p(x = heads | z)

# Joint p(x = heads, z), marginal p(x = heads), posterior p(z | x = heads)
joint = [prior[z] * p_heads[z] for z in range(2)]
marginal = sum(joint)
posterior = [j / marginal for j in joint]

print(f"prior      = {prior}")
print(f"p_heads    = {p_heads}")
print(f"joint      = {joint}")
print(f"marginal   = {marginal:.4f}   log p(x) = {log(marginal):.4f}")
print(f"posterior  = {[round(p, 4) for p in posterior]}")


def elbo(q):
    """E_q[log p(x, z) - log q(z)]."""
    return sum(q[z] * (log(joint[z]) - log(q[z])) for z in range(2) if q[z] > 0)


def kl(q, p):
    return sum(q[z] * (log(q[z]) - log(p[z])) for z in range(2) if q[z] > 0)


def log_of_average(q):
    """log E_q[p(x, z) / q(z)]  (the 'blue dot'); always equals log p(x)."""
    return log(sum(q[z] * joint[z] / q[z] for z in range(2) if q[z] > 0))


def average_of_log(q):
    """E_q[log(p(x, z) / q(z))]  (the 'orange dot'); this is the ELBO."""
    return sum(q[z] * log(joint[z] / q[z]) for z in range(2) if q[z] > 0)


print("\n  q(biased)   ELBO     KL(q||post)   ELBO+KL   log p(x)   Jensen gap")
for qb in [0.05, 0.25, 0.5, posterior[1], 0.8, 0.95]:
    q = [1 - qb, qb]
    e, k = elbo(q), kl(q, posterior)
    gap = log_of_average(q) - average_of_log(q)
    print(f"  {qb:8.4f}  {e:8.4f}  {k:11.4f}  {e + k:8.4f}  {log(marginal):8.4f}  {gap:9.4f}")

def reconstruction(q):
    """E_q[log p(x | z)]."""
    return sum(q[z] * log(p_heads[z]) for z in range(2) if q[z] > 0)


print("\n  q(biased)   reconstruction   KL(q||prior)   recon - KL   ELBO")
for qb in [0.05, 0.25, 0.5, posterior[1], 0.8, 0.95]:
    q = [1 - qb, qb]
    r, kp = reconstruction(q), kl(q, prior)
    print(f"  {qb:8.4f}  {r:14.4f}  {kp:13.4f}  {r - kp:11.4f}  {elbo(q):8.4f}")

# Scan q on a fine grid: the maximizer of the ELBO should be the posterior.
grid = [i / 1000 for i in range(1, 1000)]
best = max(grid, key=lambda qb: elbo([1 - qb, qb]))
print(f"\nELBO-maximizing q(biased) on grid = {best:.3f}   posterior = {posterior[1]:.3f}")
