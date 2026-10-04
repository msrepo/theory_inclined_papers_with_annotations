"""Continuous-latent version of the two-coin example (standard library only).

Model:  z ~ N(0, 1)            (prior)
        x | z ~ N(z, s2_lik)   (likelihood), observed x = X_OBS
Everything has a closed form, so the exact answers can be compared with the
ELBO and with sampling estimates of the marginal.
"""
import random
from math import exp, log, pi, sqrt

X_OBS = 2.0
S2_LIK = 0.25          # likelihood variance


def log_norm(v, mean, var):
    return -0.5 * log(2 * pi * var) - (v - mean) ** 2 / (2 * var)


# ---- Exact answers (marginal x ~ N(0, 1 + S2_LIK); Gaussian posterior) ----
log_px = log_norm(X_OBS, 0.0, 1.0 + S2_LIK)
post_mean = X_OBS / (1.0 + S2_LIK)
post_var = S2_LIK / (1.0 + S2_LIK)
print(f"prior       z ~ N(0, 1)")
print(f"likelihood  x | z ~ N(z, {S2_LIK}),  observed x = {X_OBS}")
print(f"log p(x)    = {log_px:.4f}")
print(f"posterior   z | x ~ N({post_mean:.3f}, {post_var:.3f})")


# ---- ELBO for q = N(m, s2): closed form ----
def reconstruction(m, s2):
    """E_q[log p(x | z)]."""
    return -0.5 * log(2 * pi * S2_LIK) - ((X_OBS - m) ** 2 + s2) / (2 * S2_LIK)


def kl_to_prior(m, s2):
    """KL( N(m, s2) || N(0, 1) )."""
    return 0.5 * (s2 + m * m - 1.0 - log(s2))


def kl_to_posterior(m, s2):
    """KL( N(m, s2) || N(post_mean, post_var) )."""
    return 0.5 * (log(post_var / s2) + (s2 + (m - post_mean) ** 2) / post_var - 1.0)


print("\n   q = N(m, s2)        recon    KL(q||prior)    ELBO    KL(q||post)   ELBO+KL")
for m, s2 in [(0.0, 1.0), (1.0, 0.5), (1.6, 1.0), (1.6, 0.05), (post_mean, post_var)]:
    r, kp = reconstruction(m, s2), kl_to_prior(m, s2)
    e, kq = r - kp, kl_to_posterior(m, s2)
    print(f"  N({m:4.2f}, {s2:5.3f})  {r:10.4f}  {kp:12.4f}  {e:9.4f}  {kq:11.4f}  {e + kq:9.4f}")
print(f"  (log p(x) = {log_px:.4f}; the last column should equal it)")


# ---- Sampling the marginal: blind (prior) vs guided (q) ----
def estimate_log_px(d, sampler, log_q, n, rng):
    """log of the importance-sampling average of p(x, z) / q(z) over n draws,
    in d independent latent dimensions (each with the same model as above)."""
    logs = []
    for _ in range(n):
        lw = 0.0
        for _ in range(d):
            z = sampler(rng)
            lw += log_norm(z, 0.0, 1.0) + log_norm(X_OBS, z, S2_LIK) - log_q(z)
        logs.append(lw)
    mx = max(logs)
    return mx + log(sum(exp(v - mx) for v in logs) / n)


proposals = {
    "prior N(0,1)  (blind)":
        (lambda r: r.gauss(0.0, 1.0), lambda z: log_norm(z, 0.0, 1.0)),
    "posterior (guided)":
        (lambda r: r.gauss(post_mean, sqrt(post_var)), lambda z: log_norm(z, post_mean, post_var)),
}

N_SAMPLES, REPEATS = 200, 30
print(f"\nEstimating log p(x) with {N_SAMPLES} samples, repeated {REPEATS} times")
print("  d   proposal                  exact log p(x)   mean estimate    std over repeats")
for d in [1, 5, 20]:
    exact = d * log_px
    for name, (sampler, log_q) in proposals.items():
        rng = random.Random(0)
        ests = [estimate_log_px(d, sampler, log_q, N_SAMPLES, rng) for _ in range(REPEATS)]
        mean = sum(ests) / REPEATS
        std = sqrt(sum((e - mean) ** 2 for e in ests) / REPEATS)
        print(f"  {d:2d}  {name:24s}  {exact:12.3f}  {mean:14.3f}  {std:14.3f}")
