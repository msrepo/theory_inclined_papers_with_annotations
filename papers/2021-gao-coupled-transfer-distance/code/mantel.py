#!/usr/bin/env python3
"""Section 5.3 of Gao & Chaudhari: re-running the Mantel tests on the distance matrices printed in Figures 2 and 3.

The matrices below are transcribed from the figures (row = source task, column = target task). The paper
reports, for each method, a normalised statistic r = 1/(n^2 - n - 1) sum_ij (a_ij - abar)(b_ij - bbar) / (s_a s_b)
against the fine-tuning matrix, and a p-value from a Mantel test. With n = 4 tasks there are only 4! = 24
relabellings of the rows and columns, and with n = 5 only 120, so exact permutation p-values are cheap.

Checked here:
  1. Whether any convention for r reproduces the paper's r for Figures 2 and 3: Pearson correlation on the
     off-diagonal entries, on all n^2 entries, on symmetrised matrices, and then a search over 640 ways of
     choosing the entries used for the means, the standard deviations and the sum, the divisor and a log
     of the fine-tuning matrix. None matches all five reported values.
  2. The exact permutation p-value, one- and two-sided, and the smallest p-value any 4x4 test can return.
  3. Rank correlations, and whether the log scale matters (fine-tuning entries span three orders of magnitude).
  4. The asymmetry of Figure 2's coupled matrix, entry against transpose.
  5. Figure 3: is the uncoupled distance larger than the coupled one for every pair, as the text says?

Standard library and numpy only.

Run:  python3 mantel.py             (prints every number quoted in the notes)
      python3 mantel.py --figures   (also rewrites ../figures/mantel.svg)
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np

# ---- Figure 2: CIFAR-100, CIFAR-10, animals, vehicles (rows = source, columns = target)
FIG2 = {
    "coupled": np.array([[0, .17, .17, .15], [.24, 0, .084, .081], [.3, .099, 0, .14], [.31, .14, .23, 0]]),
    "task2vec": np.array([[0, .037, .047, .059], [.037, 0, .044, .056], [.047, .044, 0, .071], [.059, .056, .071, 0]]),
    "finetune": np.array([[0, 180, 120, 23], [930, 0, 18, 7.6], [1100, 39, 0, 45], [1100, 200, 380, 0]]),
}
FIG2_REPORTED = {"coupled": (0.428, 0.13), "task2vec": (0.03, 0.98)}

# ---- Figure 3: herbivores, carnivores, vehicles 1, vehicles 2, flowers
FIG3 = {
    "coupled": np.array([[0, .23, .18, .17, .2], [.23, 0, .18, .17, .2], [.2, .21, 0, .18, .2], [.21, .19, .2, 0, .19], [.2, .19, .17, .16, 0]]),
    "task2vec": np.array([[0, .066, .069, .07, .068], [.066, 0, .066, .07, .07], [.069, .066, 0, .073, .07], [.07, .07, .073, 0, .077], [.068, .07, .07, .077, 0]]),
    "finetune": np.array([[0, 87, 87, 29, 68], [82, 0, 52, 40, 110], [120, 78, 0, 26, 67], [72, 44, 57, 0, 60], [99, 41, 34, 32, 0]]),
    "uncoupled": np.array([[0, .29, .18, .18, .28], [.37, 0, .2, .2, .32], [.22, .22, 0, .3, .24], [.24, .24, .27, 0, .31], [.25, .33, .19, .21, 0]]),
}
FIG3_REPORTED = {"coupled": (0.14, 0.05), "task2vec": (0.07, 0.17), "uncoupled": (0.12, 0.47)}


def r_stat(a, b, mode):
    n = a.shape[0]
    if mode == "offdiag":
        m = ~np.eye(n, dtype=bool)
        x, y = a[m], b[m]
    elif mode == "all":
        x, y = a.ravel(), b.ravel()
    elif mode == "upper-sym":  # symmetrise by averaging, then use the strict upper triangle
        iu = np.triu_indices(n, 1)
        x, y = (0.5 * (a + a.T))[iu], (0.5 * (b + b.T))[iu]
    else:
        raise ValueError(mode)
    x = x - x.mean()
    y = y - y.mean()
    return float(np.sum(x * y) / (math.sqrt(np.sum(x * x)) * math.sqrt(np.sum(y * y))))  # Pearson correlation


def paper_r(a, b, mode):
    """The paper's normalisation: (1/(n^2 - n - 1)) sum (a - abar)(b - bbar) / (sigma_a sigma_b) over the entries used."""
    n = a.shape[0]
    if mode == "offdiag":
        m = ~np.eye(n, dtype=bool)
        x, y = a[m], b[m]
    else:
        x, y = a.ravel(), b.ravel()
    k = len(x)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (k - 1) / (x.std(ddof=1) * y.std(ddof=1)))


def perm_test(a, b, mode="offdiag"):
    n = a.shape[0]
    r0 = r_stat(a, b, mode)
    rs = []
    for p in itertools.permutations(range(n)):
        p = list(p)
        rs.append(r_stat(a[np.ix_(p, p)], b, mode))
    rs = np.array(rs)
    one = float(np.mean(rs >= r0 - 1e-12))
    two = float(np.mean(np.abs(rs) >= abs(r0) - 1e-12))
    return r0, one, two, rs


def search_conventions():
    """Try every combination of entry sets for mean / std / sum, divisor, ddof and an optional log of the
    fine-tuning matrix, and report how close the best one gets to the five reported r values."""
    targets = []
    for mats, rep in ((FIG2, FIG2_REPORTED), (FIG3, FIG3_REPORTED)):
        for k, v in rep.items():
            targets.append((mats[k], mats["finetune"], v[0]))

    def entries(a, sel):
        n = a.shape[0]
        return {"all": a.ravel(), "off": a[~np.eye(n, dtype=bool)],
                "lower": a[np.tril_indices(n, -1)], "upper": a[np.triu_indices(n, 1)]}[sel]

    def stat(a, b, ms, ss, us, div, ddof, log):
        n = a.shape[0]
        if log:
            b = b.copy()
            m = ~np.eye(n, dtype=bool)
            b[m] = np.log(b[m])
        am, bm = entries(a, ms).mean(), entries(b, ms).mean()
        asd, bsd = entries(a, ss).std(ddof=ddof), entries(b, ss).std(ddof=ddof)
        x, y = entries(a, us), entries(b, us)
        d = {"n2-n-1": n * n - n - 1, "n2-1": n * n - 1, "n2-n": n * n - n, "n2": n * n, "k-1": len(x) - 1}[div]
        return float(np.sum((x - am) * (y - bm)) / (asd * bsd) / d)

    sels = ["all", "off", "lower", "upper"]
    res = []
    for ms, ss, us in itertools.product(sels, sels, sels):
        for div in ("n2-n-1", "n2-1", "n2-n", "n2", "k-1"):
            for ddof in (0, 1):
                for log in (False, True):
                    vals = [stat(a, b, ms, ss, us, div, ddof, log) for a, b, _ in targets]
                    err = max(abs(v - t) for v, (_, _, t) in zip(vals, targets))
                    res.append((err, (ms, ss, us, div, ddof, log), vals))
    res.sort(key=lambda t: t[0])
    n_plain = min(r[0] for r in res if not r[1][5])
    print(f"   searched {len(res)} conventions against the five reported values {[t[2] for t in targets]}")
    print(f"   best over all (needs a log of the fine-tuning matrix): max |error| = {res[0][0]:.3f}, "
          f"values {[round(v, 3) for v in res[0][2]]}")
    print(f"   best without a log: max |error| = {n_plain:.3f}")


def main():
    print("1. Which convention reproduces the reported r?  (Pearson r of the off-diagonal entries unless stated)")
    for fig, mats, rep in (("Fig 2", FIG2, FIG2_REPORTED), ("Fig 3", FIG3, FIG3_REPORTED)):
        ft = mats["finetune"]
        for name, (r_rep, p_rep) in rep.items():
            a = mats[name]
            print(f"   {fig} {name:9s} vs fine-tuning: reported r = {r_rep:5.3f}, p = {p_rep:4.2f}   |"
                  f"  off-diag {r_stat(a, ft, 'offdiag'):6.3f}   all n^2 {r_stat(a, ft, 'all'):6.3f}   symmetrised {r_stat(a, ft, 'upper-sym'):6.3f}")
    search_conventions()
    print()

    print("2. Exact permutation p-values (rows and columns of one matrix relabelled jointly)")
    rows = {}
    for fig, mats in (("Fig 2", FIG2), ("Fig 3", FIG3)):
        ft = mats["finetune"]
        n = ft.shape[0]
        print(f"   {fig}: n = {n}, {math.factorial(n)} relabellings; smallest possible one-sided p = 1/{math.factorial(n)} = "
              f"{1 / math.factorial(n):.4f}, two-sided 2/{math.factorial(n)} = {2 / math.factorial(n):.4f}")
        for name in mats:
            if name == "finetune":
                continue
            r0, one, two, rs = perm_test(mats[name], ft)
            rows[(fig, name)] = (r0, one, two, rs)
            print(f"      {name:9s}: r = {r0:6.3f}   p(one-sided) = {one:5.3f}   p(two-sided) = {two:5.3f}"
                  f"   null r in [{rs.min():.2f}, {rs.max():.2f}]")
    print()

    print("3. Rank correlation and the log scale (the fine-tuning entries span three orders of magnitude)")
    for fig, mats in (("Fig 2", FIG2), ("Fig 3", FIG3)):
        ft = mats["finetune"]
        n = ft.shape[0]
        m = ~np.eye(n, dtype=bool)
        lf = np.log(ft[m])
        for name in mats:
            if name == "finetune":
                continue
            a = mats[name][m]
            ra = np.argsort(np.argsort(a))
            rf = np.argsort(np.argsort(ft[m]))
            sp = float(np.corrcoef(ra, rf)[0, 1])
            lp = float(np.corrcoef(a, lf)[0, 1])
            print(f"   {fig} {name:9s}: Spearman {sp:6.3f}   Pearson with log(fine-tuning) {lp:6.3f}")
    print()

    print("4. Asymmetry in Figure 2's coupled matrix: d(i->j) against d(j->i)")
    a = FIG2["coupled"]
    names = ["CIFAR100", "CIFAR10", "animals", "vehicles"]
    for i in range(4):
        for j in range(i + 1, 4):
            print(f"   {names[i]:9s}->{names[j]:9s} {a[i, j]:5.3f}   {names[j]:9s}->{names[i]:9s} {a[j, i]:5.3f}   ratio {max(a[i, j], a[j, i]) / min(a[i, j], a[j, i]):4.2f}")
    print()
    ft2 = FIG2["finetune"]
    r2 = [max(ft2[i, j], ft2[j, i]) / min(ft2[i, j], ft2[j, i]) for i in range(4) for j in range(i + 1, 4)]
    print(f"   entry/transpose ratio in Figure 2: coupled {min(max(a[i, j], a[j, i]) / min(a[i, j], a[j, i]) for i in range(4) for j in range(i + 1, 4)):.2f} to "
          f"{max(max(a[i, j], a[j, i]) / min(a[i, j], a[j, i]) for i in range(4) for j in range(i + 1, 4)):.2f}, fine-tuning {min(r2):.2f} to {max(r2):.2f}")
    m4 = ~np.eye(4, dtype=bool)
    print(f"   off-diagonal range in Figure 2a: {a[m4].min():.3f} to {a[m4].max():.3f}")
    print()

    print("5. Figure 3: uncoupled (3d) against coupled (3a), pair by pair")
    a, d = FIG3["coupled"], FIG3["uncoupled"]
    m = ~np.eye(5, dtype=bool)
    diff = d[m] - a[m]
    ratio = d[m] / a[m]
    print(f"   {m.sum()} ordered pairs: uncoupled larger in {(diff > 1e-9).sum()}, equal in {(abs(diff) <= 1e-9).sum()}, smaller in {(diff < -1e-9).sum()}")
    print(f"   ratio uncoupled / coupled: min {ratio.min():.2f}, median {np.median(ratio):.2f}, max {ratio.max():.2f}")
    ft = FIG3["finetune"]
    print(f"   off-diagonal range in Figure 3: coupled {a[m].min():.2f} to {a[m].max():.2f}, fine-tuning {ft[m].min():g} to {ft[m].max():g}")
    asym = lambda x: max(max(x[i, j], x[j, i]) / min(x[i, j], x[j, i]) for i in range(5) for j in range(i + 1, 5))
    print(f"   largest entry/transpose ratio in Figure 3: coupled {asym(a):.2f}, uncoupled {asym(d):.2f}, fine-tuning {asym(ft):.2f}")
    print()

    return rows


def figures(rows):
    from svgkit import Axes, svg, write

    r0, one, two, rs = rows[("Fig 2", "coupled")]
    r0t, _, _, rst = rows[("Fig 2", "task2vec")]
    W, H = 640, 250
    ax = Axes(60, 30, 520, 150, (-1, 1), (0, 4))
    body = [ax.frame([-1, -0.5, 0, 0.5, 1], [0, 1, 2, 3, 4], "Mantel r", "number of the 24 relabellings")]
    vals, counts = np.unique(np.round(rs, 4), return_counts=True)
    for v, c in zip(vals, counts):
        body.append(f'<rect class="bb" x="{ax.X(v) - 7:.1f}" y="{ax.Y(c):.1f}" width="14" height="{ax.Y(0) - ax.Y(c):.1f}"/>')
    body.append(f'<line class="o" x1="{ax.X(r0):.1f}" x2="{ax.X(r0):.1f}" y1="{ax.y0}" y2="{ax.y0 + ax.h}"/>')
    body.append(f'<text class="lab" x="{ax.X(r0) - 5:.1f}" y="{ax.y0 + 12}" text-anchor="end">coupled vs fine-tuning: r = {r0:.2f}</text>')
    body.append(f'<line class="r dash" x1="{ax.X(r0t):.1f}" x2="{ax.X(r0t):.1f}" y1="{ax.y0}" y2="{ax.y0 + ax.h}"/>')
    body.append(f'<text class="lab" x="{ax.X(r0t) - 5:.1f}" y="{ax.y0 + 12}" text-anchor="end">Task2Vec: r = {r0t:.2f}</text>')
    write("mantel.svg", svg(W, H, "Mantel null distribution for Figure 2",
                            "With four tasks there are only 24 relabellings, so the smallest p-value is 1/24.", body))


if __name__ == "__main__":
    rows = main()
    if "--figures" in sys.argv:
        figures(rows)
