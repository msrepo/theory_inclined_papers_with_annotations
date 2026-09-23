#!/usr/bin/env python3
"""What a rank correlation over 10 architectures can actually resolve.

Chaves et al. rank 10 ImageNet-pretrained architectures on 3 medical tasks (plus
3 out-of-distribution test sets) with 7 transferability scores, and report a
Kendall tau per cell: 42 numbers. Their conclusion is negative -- no score
reliably predicts target performance in medical contexts.

Their experiment is 2250 fine-tuned models and is not reproducible here. What is
checkable is the resolution of the instrument: a rank correlation over n = 10
items is very noisy, and that determines which of their 42 cells carry
information.

Checked here:

  * the null distribution of Kendall's tau at n = 10, against the analytic
    sd = sqrt(2(2n+5)/(9n(n-1)));
  * how many of their reported cells clear the 5% threshold, against how many
    would be expected by chance;
  * what survives a multiple-comparison correction -- which turns out to be
    exactly their POSITIVE finding, not their negative one.

Standard library and numpy only.

Run:  python3 power.py
"""
from __future__ import annotations

import numpy as np


def kendall_tau(a, b):
    n = len(a)
    i, j = np.triu_indices(n, 1)
    return float(np.sign(a[i] - a[j]) @ np.sign(b[i] - b[j]) / (n * (n - 1) / 2))


def null_taus(n, reps, rng):
    base = np.arange(n)
    return np.array([kendall_tau(base, rng.permutation(n)) for _ in range(reps)])


# Table 2 of the paper. Rows are (dataset, is_out_of_distribution); columns are
# H-Score, NCE, LEEP, N-LEEP, LogME, Regularised H-Score, GBC.
SCORERS = ["H-Score", "NCE", "LEEP", "N-LEEP", "LogME", "Reg. H-Score", "GBC"]
TABLE2 = [
    ("BrainTumor-cheng", False, [0.270, -0.180, 0.090, 0.494, 0.584, 0.405, 0.135]),
    ("NINS", True,             [-0.333, 0.156, 0.200, -0.333, -0.289, -0.422, 0.200]),
    ("BreakHis", False,         [0.600, -0.156, 0.200, -0.244, 0.378, 0.200, 0.022]),
    ("ICIAR2018", True,         [0.333, 0.778, 0.778, 0.289, 0.289, 0.378, 0.156]),
    ("ISIC2019", False,        [-0.244, 0.022, 0.333, -0.111, -0.067, -0.289, 0.022]),
    ("PAD-UFES-20", True,      [-0.156, 0.911, 0.422, -0.156, -0.022, -0.022, 0.067]),
]


def resolution(rng):
    print("1. What Kendall's tau can resolve, by sample size\n")
    print("   Their n is 10: ten pre-trained architectures per cell.\n")
    print(f"   {'n':>5}  {'sd under the null':>18}  {'analytic sd':>12}"
          f"  {'|tau| for p<0.05':>17}")
    for n in (10, 20, 50, 100):
        null = null_taus(n, 60_000, rng)
        analytic = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
        crit = np.quantile(np.abs(null), 0.95)
        print(f"   {n:>5}  {null.std():>18.3f}  {analytic:>12.3f}  {crit:>17.3f}")
    print("\n   At n = 10 a tau of 0.27 is barely one standard error from zero.\n")


def what_survives(rng):
    print("2. Applying that to their Table 2\n")
    null = null_taus(10, 200_000, rng)
    crit05 = float(np.quantile(np.abs(null), 0.95))
    n_cells = sum(len(r[2]) for r in TABLE2)
    crit_bonf = float(np.quantile(np.abs(null), 1 - 0.05 / n_cells))
    print(f"   {n_cells} cells.  |tau| for p<0.05: {crit05:.3f}"
          f"   Bonferroni over {n_cells}: {crit_bonf:.3f}\n")
    print(f"   {'dataset':>18}  {'':>4}  {'clears p<0.05':>34}")
    hits05, hitsb = [], []
    for name, ood, row in TABLE2:
        tag = "OOD" if ood else "in-d"
        names = [f"{SCORERS[k]} {v:+.3f}" for k, v in enumerate(row)
                 if abs(v) >= crit05]
        hitsb += [(name, ood, SCORERS[k], v) for k, v in enumerate(row)
                  if abs(v) >= crit_bonf]
        hits05 += names
        print(f"   {name:>18}  {tag:>4}  {', '.join(names) if names else '-':>34}")
    print(f"\n   {len(hits05)} of {n_cells} clear p<0.05; "
          f"~{0.05*n_cells:.1f} expected by chance alone.")
    print(f"   {len(hitsb)} clear Bonferroni:")
    for name, ood, sc, v in hitsb:
        print(f"     {sc:>12} on {name:<14} {v:+.3f}   "
              f"{'OUT-of-distribution' if ood else 'in-distribution'}")
    print("\n   Note what survives: every one is an out-of-distribution cell, and")
    print("   every one is a LABEL-based scorer (NCE, LEEP). That is precisely")
    print("   the paper's positive finding. Their headline NEGATIVE finding --")
    print("   that no scorer works in-distribution -- rests on cells the study")
    print("   is not powered to distinguish from zero either way.\n")
    print("   Underpowered null results are the safe direction to err in: failing")
    print("   to detect a correlation is not evidence that none exists. The")
    print("   recommendation not to rely on these scores in medical imaging is")
    print("   reasonable; the evidence is thinner than 42 numbers make it look.\n")


def what_would_it_take(rng):
    print("3. How many architectures would be needed\n")
    print("   To call a true tau of 0.3 significant at p<0.05 with 80% power:\n")
    true_tau = 0.3
    print(f"   {'n':>5}  {'|tau| for p<0.05':>17}  {'power at true tau=0.3':>22}")
    for n in (10, 20, 30, 50, 80):
        null = null_taus(n, 40_000, rng)
        crit = np.quantile(np.abs(null), 0.95)
        # simulate an alternative with roughly the target concordance
        hits = 0
        for _ in range(3000):
            x = rng.standard_normal(n)
            y = true_tau * x + np.sqrt(max(1 - true_tau ** 2, 0)) * rng.standard_normal(n)
            hits += abs(kendall_tau(x, y)) >= crit
        print(f"   {n:>5}  {crit:>17.3f}  {hits/3000:>22.2f}")
    print("\n   Ten architectures is not enough to see a moderate effect. Any future")
    print("   benchmark of this kind needs a model pool several times larger, or")
    print("   should report confidence intervals rather than point estimates.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Statistical resolution of Chaves et al. 2023's Table 2")
    print("=" * 70, "\n")
    resolution(rng)
    what_survives(rng)
    what_would_it_take(rng)
