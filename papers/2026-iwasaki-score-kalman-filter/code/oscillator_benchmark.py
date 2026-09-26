#!/usr/bin/env python3
"""The coupled-oscillator benchmark of Sec 7.3 / App J.5, re-created with the paper's settings.

The claim under test is Table A1: the SKF's RMSE is 4-300x below EKF, UKF, EnKF and a
500k-particle bootstrap PF. No code is released, so this rebuilds the benchmark from
App G.3 and J.5 (gamma = 0.3, alpha = 1, beta = 0.6, kappa = 0.3, sigma = 0.4,
dt_pred = 0.15, dt_ode = 0.005, 25 steps, odd positions observed with R = 0.3^2 I,
prior N(m0, 0.15^2 I) with positions cycling 0.3, -0.2, 0.1, -0.3, ...).

Checked here:

  1. Fig 3's "ground truth" is the noise-free ODE started exactly at the prior mean:
     the deterministic trajectory reproduces the plotted curves, while genuine
     sigma = 0.4 sample paths from the same start wander far from it.
  2. With that truth, a correctly tuned EKF and a bootstrap PF land on the paper's
     EKF / PF numbers, and the same EKF told to trust its measurements 100x less
     beats the reported SKF. A truth that never meets the process noise rewards a
     filter for ignoring its data.
  3. With a truth drawn from the model every filter assumes, the ranking flips:
     the correctly tuned filters are best and all errors are 3-4x larger (and heavy-tailed:
     occasionally the true path crosses the quadratic spring's barrier).

Standard library and numpy only.  Takes about two minutes.

Run:  python3 oscillator_benchmark.py
"""
from __future__ import annotations

import numpy as np

GAMMA, ALPHA, BETA, KAPPA, SIGMA = 0.3, 1.0, 0.6, 0.3, 0.4
R_STD, P0_STD = 0.3, 0.15
DT_PRED, DT_ODE, STEPS = 0.15, 0.005, 25
SUB = int(round(DT_PRED / DT_ODE))
CYCLE = [0.3, -0.2, 0.1, -0.3, 0.15, 0.25, -0.1, 0.2]
SEEDS = 10                # noise-free truth: only the measurement noise changes between seeds
SEEDS_MODEL = 40          # model-drawn truth: heavy-tailed errors (some paths cross the barrier), so more seeds
PARTICLES = 10_000
PAPER = {  # Table A2, mean RMSE over 10 seeds
    4: dict(SKF=0.0103, EKF=0.0676, UKF=0.0725, EnKF=0.0762, PF=0.0727),
    8: dict(SKF=0.0181, EKF=0.0713, UKF=0.0775, EnKF=0.0806, PF=0.0747),
}


def drift(X, N):
    """App G.3 with free ends (q_0 = q_1, q_{N+1} = q_N). X is (..., 2N) = (q, p)."""
    q, p = X[..., :N], X[..., N:]
    qm = np.concatenate([q[..., :1], q[..., :-1]], -1)
    qp = np.concatenate([q[..., 1:], q[..., -1:]], -1)
    return np.concatenate([p, -GAMMA * p - ALPHA * q - BETA * q**2 + KAPPA * (qp - 2 * q + qm)], -1)


def jacobian(x, N):
    F = np.zeros((2 * N, 2 * N))
    F[:N, N:] = np.eye(N)
    lap = -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
    lap[0, 0] = lap[-1, -1] = -1                              # free ends
    F[N:, :N] = np.diag(-ALPHA - 2 * BETA * x[:N]) + KAPPA * lap
    F[N:, N:] = -GAMMA * np.eye(N)
    return F


def rk4(x, dt, N):
    k1 = drift(x, N)
    k2 = drift(x + dt / 2 * k1, N)
    k3 = drift(x + dt / 2 * k2, N)
    k4 = drift(x + dt * k3, N)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def prior_mean(N):
    return np.array([CYCLE[i % 8] for i in range(N)] + [0.0] * N)


def observation_matrix(N):
    obs = np.arange(0, N, 2)                                  # q1, q3, ...
    C = np.zeros((len(obs), 2 * N))
    C[np.arange(len(obs)), obs] = 1
    return C


def simulate_truth(N, noise_free, rng):
    """noise_free=True: the ODE from the prior mean (what Fig 3 plots).
    noise_free=False: x0 ~ prior, then Euler-Maruyama with sigma = 0.4 (a draw from the model)."""
    x = prior_mean(N) if noise_free else prior_mean(N) + P0_STD * rng.standard_normal(2 * N)
    xs = []
    for _ in range(STEPS):
        for _ in range(SUB):
            if noise_free:
                x = rk4(x, DT_ODE, N)
            else:
                x = x + DT_ODE * drift(x, N)
                x[N:] += SIGMA * np.sqrt(DT_ODE) * rng.standard_normal(N)
        xs.append(x.copy())
    xs = np.array(xs)
    C = observation_matrix(N)
    return xs, xs @ C.T + R_STD * rng.standard_normal((STEPS, C.shape[0]))


def ekf(N, zs, trust=1.0):
    """Continuous-discrete EKF with the paper's model. trust < 1 inflates R by 1/trust."""
    C = observation_matrix(N)
    x, P = prior_mean(N), P0_STD**2 * np.eye(2 * N)
    Qc = np.zeros((2 * N, 2 * N))
    Qc[N:, N:] = SIGMA**2 * np.eye(N)
    R = R_STD**2 / trust * np.eye(C.shape[0])
    out = []
    for z in zs:
        for _ in range(SUB):
            F = jacobian(x, N)
            x = rk4(x, DT_ODE, N)
            P = P + DT_ODE * (F @ P + P @ F.T + Qc)
        K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
        x = x + K @ (z - C @ x)
        P = (np.eye(2 * N) - K @ C) @ P
        out.append(x.copy())
    return np.array(out)


def bootstrap_pf(N, zs, rng, n=PARTICLES):
    C = observation_matrix(N)
    X = prior_mean(N) + P0_STD * rng.standard_normal((n, 2 * N))
    out = []
    for z in zs:
        for _ in range(SUB):
            X = X + DT_ODE * drift(X, N)
            X[:, N:] += SIGMA * np.sqrt(DT_ODE) * rng.standard_normal((n, N))
        logw = -0.5 * np.sum((z - X @ C.T) ** 2, 1) / R_STD**2
        w = np.exp(logw - logw.max())
        w /= w.sum()
        out.append(w @ X)
        u = (rng.random() + np.arange(n)) / n                 # systematic resampling
        X = X[np.minimum(np.searchsorted(np.cumsum(w), u), n - 1)]
    return np.array(out)


def rmse(est, xs, N):
    """Position RMSE at each step, averaged over the 25 steps."""
    return float(np.mean(np.sqrt(np.mean((est[:, :N] - xs[:, :N]) ** 2, 1))))


def ground_truth_is_the_ode():
    print("1. Fig 3 (n = 8): the noise-free ODE from the prior mean")
    N = 4
    x, traj = prior_mean(N), [prior_mean(N)]
    for _ in range(STEPS * SUB):
        x = rk4(x, DT_ODE, N)
        traj.append(x.copy())
    traj = np.array(traj)
    for t in (0.0, 1.0, 2.0, 2.5, 3.0, 3.75):
        q = traj[int(round(t / DT_ODE)), :N]
        print(f"   t = {t:4.2f}  q1..q4 = " + "  ".join(f"{v:+.3f}" for v in q))
    rng = np.random.default_rng(123)
    M = 5000
    X = np.tile(prior_mean(N), (M, 1))
    worst = np.zeros(M)
    for k in range(STEPS):
        for _ in range(SUB):
            X = X + DT_ODE * drift(X, N)
            X[:, N:] += SIGMA * np.sqrt(DT_ODE) * rng.standard_normal((M, N))
        worst = np.maximum(worst, np.abs(X[:, :N] - traj[(k + 1) * SUB, :N]).max(1))
    print(f"   {M} sigma = 0.4 paths from the same start: largest |q - q_ode| over the run has "
          f"median {np.median(worst):.2f}, 1st percentile {np.percentile(worst, 1):.2f}; "
          f"{np.mean(worst < 0.03):.4f} of paths stay within 0.03\n")


def table():
    print(f"2-3. Position RMSE (PF: {PARTICLES:,} particles for the noise-free truth, 5,000 for the model-drawn one)")
    runs = [(2, True, SEEDS), (4, True, SEEDS), (2, False, SEEDS_MODEL)]    # N oscillators, state dimension n = 2N
    barrier = -ALPHA / BETA                                    # the quadratic spring's potential peaks here
    for N, noise_free, seeds in runs:
        res = {k: [] for k in ("EKF", "PF", "EKF, R x100")}
        crossed = 0
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            xs, zs = simulate_truth(N, noise_free, rng)
            crossed += bool(xs[:, :N].min() < barrier)
            res["EKF"].append(rmse(ekf(N, zs), xs, N))
            res["EKF, R x100"].append(rmse(ekf(N, zs, trust=0.01), xs, N))
            res["PF"].append(rmse(bootstrap_pf(N, zs, rng, PARTICLES if noise_free else 5000), xs, N))
        truth = "noise-free from the prior mean" if noise_free else "drawn from the model"
        print(f"   n = {2 * N}, truth {truth}, {seeds} seeds:")
        print("      ours, mean:   " + "   ".join(f"{k} {np.mean(v):.4f}±{np.std(v):.4f}" for k, v in res.items()))
        if noise_free:
            print("      paper, mean:  " + "   ".join(f"{k} {v:.4f}" for k, v in PAPER[2 * N].items()))
        else:
            print("      ours, median: " + "   ".join(f"{k} {np.median(v):.4f}" for k, v in res.items()))
            print(f"      true paths that cross the barrier q = {barrier:.2f} within 3.75 s: {crossed} of {seeds}")
    print()


def trust_curve():
    print("4. EKF RMSE against trust in the measurements (R divided by trust), n = 4")
    trusts = [100, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.001]
    for noise_free in (True, False):
        vals = []
        for tr in trusts:
            e = []
            for seed in range(SEEDS if noise_free else SEEDS_MODEL):
                xs, zs = simulate_truth(2, noise_free, np.random.default_rng(seed))
                e.append(rmse(ekf(2, zs, trust=tr), xs, 2))
            vals.append(np.mean(e))
        label = "noise-free truth" if noise_free else f"model-drawn truth ({SEEDS_MODEL} seeds)"
        print(f"   {label:>17}: " + "  ".join(f"{tr:g}:{v:.4f}" for tr, v in zip(trusts, vals)))
    print()


if __name__ == "__main__":
    print("SKF coupled-oscillator benchmark, re-created (Iwasaki et al. 2026, Sec 7.3)")
    print("=" * 76, "\n")
    ground_truth_is_the_ode()
    table()
    trust_curve()
