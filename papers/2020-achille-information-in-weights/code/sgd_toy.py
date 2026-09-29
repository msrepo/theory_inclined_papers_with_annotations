#!/usr/bin/env python3
"""The one-parameter toy of Section 4 / Figure 3, rebuilt and taken apart.

The paper's setup: a dataset D = {x_i}, N = 100, x_i ~ N(mu, 1) with
mu ~ Unif[-1, 1]; the loss L_D(theta) = (1/N) sum_i (x_i - phi(theta))^2 with a
fixed, redundant zigzag phi that is steep near theta = 0 and shallow far from
it (their Figure 3, right). Every linear piece of phi sweeps the whole range
[-1, 1], so every piece holds exactly one global minimiser phi(theta) = x-bar,
all with the same loss. The pieces differ only in their slope s, hence in their
curvature 2 s^2 and their Fisher information N s^2.

Choices the paper does not state, fixed here and reported in the notes:
  * phi has 10 pieces on each side of 0 on [-1.5, 1.5], widths growing by a
    factor 1.4, so slopes run from about 93 down to 4.5;
  * mu ~ Unif[-0.8, 0.8], which keeps x-bar inside [-1, 1] with high
    probability, so that every minimiser sits inside a piece and not on a kink;
  * two initialisations: theta_0 ~ Unif[-1.5, 1.5], and theta_0 ~ Unif[-0.5, 0.5]
    (the paper's GD histogram sits inside |theta| < 0.7, which suggests the
    latter); reflecting boundary; learning rate 1e-4, just stable in the
    steepest piece (eta * 2 s^2 = 1.73 < 2); 20 000 steps;
  * the minibatch mean is drawn as x-bar plus Gaussian noise with the exact
    variance of sampling B of N points without replacement.

Measured, for each batch size B and for isotropic Langevin at three
temperatures:
  * which piece SGD ends in, and its entropy H(piece);
  * mean log Fisher, E log(N s^2), the paper's "log-determinant of Fisher";
  * the paper's estimator of Shannon information:
        E_{D, run} KL( N(theta*, 1/F) || Qbar ),  Qbar = mixture of all of them;
  * the same with the SGD run averaged out first, which is I(theta; D) proper:
        E_D KL( (1/R) sum_r N(theta*_r, 1/F_r) || Qbar );
  * the Gaussian IW with the best proper Gaussian pre-distribution.

Standard library and numpy only.

Run:  python3 sgd_toy.py             (prints every number quoted in the notes)
      python3 sgd_toy.py --figures   (also rewrites ../figures/sgd-toy.svg)
"""
from __future__ import annotations

import math
import sys

import numpy as np

N = 100
HALF = 1.5
NPIECE = 10          # per side
GROWTH = 1.4
ETA = 1e-4
STEPS = 20_000
M_DATA = 300         # datasets
R_RUNS = 12          # SGD runs per dataset
BATCHES = [1, 4, 16, 64, 100]
TEMPS = [0.01, 0.04, 0.16]

w0 = HALF * (GROWTH - 1) / (GROWTH ** NPIECE - 1)
WIDTHS = w0 * GROWTH ** np.arange(NPIECE)
KINKS = np.concatenate([[0.0], np.cumsum(WIDTHS)])      # 0 .. 1.5
KINKS[-1] = HALF
SLOPES = 2.0 / WIDTHS


def piece_of(theta):
    """Signed piece index: 0..NPIECE-1 for theta >= 0, -1..-NPIECE for theta < 0."""
    a = np.abs(theta)
    j = np.clip(np.searchsorted(KINKS, a, side="right") - 1, 0, NPIECE - 1)
    return np.where(theta >= 0, j, -j - 1)


def phi_and_slope(theta):
    """phi is even; phi(kink_j) = (-1)^j, linear in between."""
    a = np.abs(theta)
    j = np.clip(np.searchsorted(KINKS, a, side="right") - 1, 0, NPIECE - 1)
    start = np.where(j % 2 == 0, 1.0, -1.0)
    ds = np.where(j % 2 == 0, -1.0, 1.0) * SLOPES[j]      # d phi / d|theta|
    val = start + ds * (a - KINKS[j])
    return val, ds * np.sign(np.where(theta == 0, 1.0, theta))


def minimiser_in_piece(p, xbar):
    """The theta in piece p with phi(theta) = xbar (clipped into the piece)."""
    j = np.where(p >= 0, p, -p - 1)
    start = np.where(j % 2 == 0, 1.0, -1.0)
    ds = np.where(j % 2 == 0, -1.0, 1.0) * SLOPES[j]
    a = KINKS[j] + np.clip((xbar - start) / ds, 0.0, WIDTHS[j])
    return np.where(p >= 0, a, -a)


def reflect(theta):
    theta = np.where(theta > HALF, 2 * HALF - theta, theta)
    return np.where(theta < -HALF, -2 * HALF - theta, theta)


def make_data(rng):
    mu = rng.uniform(-0.8, 0.8, M_DATA)
    x = mu[:, None] + rng.standard_normal((M_DATA, N))
    return mu, x.mean(1), x.var(1, ddof=1)


def run(xbar, s2, theta0, rng, batch=None, temp=None):
    """Vectorised over all (dataset, run) pairs. Returns the final iterate."""
    xb = np.repeat(xbar, R_RUNS)
    sd = np.sqrt(np.repeat(s2, R_RUNS))
    th = theta0.copy()
    if batch is not None:
        noise_sd = sd * math.sqrt((N - batch) / (batch * (N - 1)))
    for _ in range(STEPS):
        ph, sl = phi_and_slope(th)
        if batch is not None:
            target = xb + noise_sd * rng.standard_normal(th.shape)
            th = th + ETA * 2 * (target - ph) * sl
        else:
            th = th + ETA * 2 * (xb - ph) * sl + math.sqrt(2 * ETA * temp) * rng.standard_normal(th.shape)
        th = reflect(th)
    return th


GRID = np.linspace(-HALF - 0.05, HALF + 0.05, 34_001)
DG = GRID[1] - GRID[0]


def gauss_on_grid(m, v):
    return np.exp(-(GRID[None, :] - m[:, None]) ** 2 / (2 * v[:, None])) / np.sqrt(2 * np.pi * v[:, None])


def information(theta_star, fisher, xbar_rep, mu_rep):
    """Paper's estimator, the proper I(theta; D), and the best Gaussian IW."""
    v = 1.0 / fisher
    qbar = np.zeros_like(GRID)
    for c in range(0, len(v), 400):
        qbar += gauss_on_grid(theta_star[c:c + 400], v[c:c + 400]).sum(0)
    qbar /= len(v)
    logq = np.log(np.maximum(qbar, 1e-300))
    # paper: average KL of each single-run Gaussian to the mixture
    kl = []
    for c in range(0, len(v), 400):
        g = gauss_on_grid(theta_star[c:c + 400], v[c:c + 400])
        cross = (g * logq).sum(1) * DG
        kl.append(-0.5 * np.log(2 * np.pi * np.e * v[c:c + 400]) - cross)
    i_paper = float(np.mean(np.concatenate(kl)))
    # proper: Q(theta | D) must average out the SGD run. With only R runs per
    # dataset the empirical mixture is biased (about (K - 1) / (2R) nats), so the
    # piece probabilities are pooled over all datasets in the same mu-bin and
    # each dataset gets sum_b pi(b | bin) N(theta*_b(xbar), 1 / F_b).
    klp = []
    xb = xbar_rep.reshape(M_DATA, R_RUNS)[:, 0]
    bins = np.digitize(mu_rep.reshape(M_DATA, R_RUNS)[:, 0], np.linspace(-0.8, 0.8, 7)[1:-1])
    pieces = np.concatenate([-np.arange(1, NPIECE + 1)[::-1], np.arange(NPIECE)])
    pr = piece_of(theta_star).reshape(M_DATA, R_RUNS)
    for d in range(M_DATA):
        same = pr[bins == bins[d]].ravel()
        pi = np.array([(same == p).mean() for p in pieces])
        keep = pi > 0
        pk = pieces[keep]
        jk = np.where(pk >= 0, pk, -pk - 1)
        g = gauss_on_grid(minimiser_in_piece(pk, np.full(len(pk), xb[d])), 1.0 / (N * SLOPES[jk] ** 2))
        q = (pi[keep][:, None] * g).sum(0)
        mask = q > 1e-300
        klp.append(float((q[mask] * (np.log(q[mask]) - logq[mask])).sum() * DG))
    i_proper = float(np.mean(klp))
    # best proper Gaussian pre-distribution N(m, lam^2)
    m = theta_star.mean()
    lam2 = np.mean((theta_star - m) ** 2 + v)
    i_gauss = float(np.mean(0.5 * (np.log(lam2 / v) - 1 + (v + (theta_star - m) ** 2) / lam2)))
    return i_paper, i_proper, i_gauss, qbar


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def summarise(label, theta_final, xbar_rep, mu_rep):
    p = piece_of(theta_final)
    ts = minimiser_in_piece(p, xbar_rep)
    j = np.where(p >= 0, p, -p - 1)
    fisher = N * SLOPES[j] ** 2
    counts = np.bincount(j, minlength=NPIECE) / len(j)
    h_piece = entropy(np.bincount(p + NPIECE, minlength=2 * NPIECE) / len(p))
    # does the piece carry information about the data? plug-in I(piece; mu-bin)
    bins = np.digitize(mu_rep, np.linspace(-0.8, 0.8, 7)[1:-1])
    joint = np.zeros((6, 2 * NPIECE))
    np.add.at(joint, (bins, p + NPIECE), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    i_piece_data = float((joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])).sum())
    i_paper, i_proper, i_gauss, qbar = information(ts, fisher, xbar_rep, mu_rep)
    row = dict(label=label, logF=float(np.mean(np.log(fisher))), h=h_piece, i_paper=i_paper,
               i_proper=i_proper, i_gauss=i_gauss, i_pd=i_piece_data, counts=counts,
               theta=ts, qbar=qbar)
    print(f"  {label:<12} E logF {row['logF']:5.2f}   H(piece) {h_piece:4.2f}   "
          f"paper-I {i_paper:4.2f}   I(theta;D) {i_proper:4.2f}   Gaussian IW {i_gauss:4.2f}   "
          f"I(piece;mu) {i_piece_data:5.3f}")
    return row


def within_piece_prediction():
    """I(xbar; xbar + N(0, 1/N)) for xbar = mu + N(0,1/N), mu ~ U[-.8,.8]: the
    information one piece carries, the same for every piece (in phi-coordinates
    every Gaussian N(theta*, 1/(N s^2)) becomes N(xbar, 1/N))."""
    u = np.linspace(-1.6, 1.6, 20_001)
    du = u[1] - u[0]
    sd = math.sqrt(2.0 / N)             # xbar noise plus the coding noise
    dens = (np.vectorize(lambda t: 0.5 * (math.erf((0.8 - t) / (sd * math.sqrt(2))) -
                                          math.erf((-0.8 - t) / (sd * math.sqrt(2)))))(u)) / 1.6
    h = -float((dens[dens > 0] * np.log(dens[dens > 0])).sum() * du)
    return h - 0.5 * math.log(2 * math.pi * math.e / N)


def main():
    rng = np.random.default_rng(0)
    mu, xbar, s2 = make_data(rng)
    xbar_rep = np.repeat(xbar, R_RUNS)
    mu_rep = np.repeat(mu, R_RUNS)

    print("zigzag phi: 10 pieces per side, widths "
          f"{WIDTHS[0]:.4f} .. {WIDTHS[-1]:.3f}, slopes {SLOPES[0]:.1f} .. {SLOPES[-1]:.2f}, "
          f"Fisher N s^2 from {N * SLOPES[-1] ** 2:.0f} to {N * SLOPES[0] ** 2:.0f}")
    print(f"stability of GD in the steepest piece: eta * 2 s^2 = {ETA * 2 * SLOPES[0] ** 2:.2f} (< 2)")
    print(f"{M_DATA} datasets x {R_RUNS} runs, {STEPS} steps\n")

    pred = within_piece_prediction()
    print(f"information inside one piece (same for every piece): {pred:.2f} nats\n")

    allrows = {}
    for init, lo in [("wide", HALF), ("centred", 0.5)]:
        theta0 = np.random.default_rng(1).uniform(-lo, lo, M_DATA * R_RUNS)
        rows = allrows[init] = {}
        print(f"=== initialisation theta_0 ~ Unif[-{lo}, {lo}] ===")
        print("SGD, by batch size")
        for b in BATCHES:
            th = run(xbar, s2, theta0, np.random.default_rng(100 + b), batch=b)
            rows[f"B={b}"] = summarise(f"B = {b}", th, xbar_rep, mu_rep)
        print("isotropic Langevin, dtheta = -L' dt + sqrt(2T) dW")
        for t in TEMPS:
            th = run(xbar, s2, theta0, np.random.default_rng(7), temp=t)
            rows[f"T={t}"] = summarise(f"T = {t}", th, xbar_rep, mu_rep)

        print("share of runs ending in each piece (piece 0 = steepest, both sides pooled)")
        print("  width share " + " ".join(f"{c:5.3f}" for c in WIDTHS / WIDTHS.sum()))
        for k in ["B=100", "B=16", "B=4", "B=1"] + [f"T={t}" for t in TEMPS]:
            print(f"  {k:<11} " + " ".join(f"{c:5.3f}" for c in rows[k]["counts"]))

        gd, sgd1 = rows["B=100"], rows["B=1"]
        print("from B = 100 to B = 1:")
        print(f"  drop in the paper's Shannon estimate  {gd['i_paper'] - sgd1['i_paper']:.2f} nats")
        print(f"  drop in the entropy of the piece      {gd['h'] - sgd1['h']:.2f} nats")
        print(f"  drop in I(theta; D) proper            {gd['i_proper'] - sgd1['i_proper']:.2f} nats")
        print("  paper-I minus H(piece), by batch:     " +
              " ".join(f"{rows[f'B={b}']['i_paper'] - rows[f'B={b}']['h']:.2f}" for b in BATCHES))
        print()

    gs = [r["i_gauss"] for rows in allrows.values() for r in rows.values()]
    print("the paper quotes 4000-5000 nats of Gaussian IW for this toy; here the best proper "
          f"Gaussian code gives {min(gs):.2f}-{max(gs):.2f} nats, never below the Shannon value "
          "(Proposition 2.3).")
    rows = allrows["centred"]

    if "--figures" in sys.argv:
        figures(rows)


def figures(rows):
    from pathlib import Path
    import svgplot as sp

    out = Path(__file__).resolve().parent.parent / "figures"
    W, Hh = 720, 600
    body = [sp.title(20, 22, "Figure 3 rebuilt: what small batches actually change"),
            sp.sub(20, 39, "One parameter, 20 zigzag pieces, each holding one global minimum with the same loss. "
                           "Seeded; code/sgd_toy.py --figures.")]
    # panel A: density of final theta (histogram), B = 100 vs B = 1, with Fisher step curve
    ax = sp.Axes(62, 70, 620, 170, (-1.5, 1.5), (0, 3.2))
    body.append(ax.frame([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], [0, 1, 2, 3],
                         "final θ (snapped to the minimiser of its piece)", "density of final θ"))
    edges = np.linspace(-1.5, 1.5, 151)
    for key, cls in [("B=100", "c0b"), ("B=1", "c1b")]:
        h, _ = np.histogram(rows[key]["theta"], bins=edges, density=True)
        body.append(ax.bars(edges, np.minimum(h, 3.2), cls))
    xs = np.linspace(-1.5, 1.5, 1201)
    j = np.where(piece_of(xs) >= 0, piece_of(xs), -piece_of(xs) - 1)
    lf = np.log(N * SLOPES[j] ** 2)
    body.append(ax.path(xs, (lf - 7.5) / (13.8 - 7.5) * 3.0, "k"))
    body.append(sp.legend(470, 82, [("c0b", "B = 100 (plain GD)"), ("c1b", "B = 1"),
                                    ("k", "log Fisher of the piece (rescaled)")]))
    # panel B: information vs batch size
    ax2 = sp.Axes(62, 320, 280, 170, (0, 4.6), (0, 4.5))
    xpos = {1: 0, 4: 1, 16: 2, 64: 3, 100: 4.6}
    body.append(ax2.frame([], [0, 1, 2, 3, 4], "batch size B", "nats"))
    for b, xp in xpos.items():
        body.append(sp.tick_label(ax2, xp, str(b)))
    series = [("i_paper", "c0", "paper's estimator"), ("h", "c1", "entropy of the piece"),
              ("i_proper", "c2", "I(θ; D), runs averaged")]
    for key, cls, _ in series:
        body.append(ax2.path([xpos[b] for b in BATCHES], [rows[f"B={b}"][key] for b in BATCHES], cls))
        body.append(ax2.dots([xpos[b] for b in BATCHES], [rows[f"B={b}"][key] for b in BATCHES], cls + "f"))
    body.append(sp.legend(70, 540, [(c, l) for _, c, l in series]))
    # panel C: piece shares
    ax3 = sp.Axes(420, 320, 262, 170, (-0.5, 9.5), (0, 0.45))
    body.append(ax3.frame([], [0, 0.1, 0.2, 0.3, 0.4], "piece, steepest (0) to shallowest (9)", "share of runs"))
    for k in range(10):
        body.append(sp.tick_label(ax3, k, str(k)))
    ws = WIDTHS / WIDTHS.sum()
    body.append(ax3.path(np.arange(10), ws, "k"))
    for key, cls in [("B=100", "c0"), ("B=1", "c1"), ("T=0.16", "c3")]:
        body.append(ax3.path(np.arange(10), rows[key]["counts"], cls))
        body.append(ax3.dots(np.arange(10), rows[key]["counts"], cls + "f"))
    body.append(sp.legend(430, 540, [("k", "width share"), ("c0", "B = 100 (GD)"), ("c1", "B = 1"),
                                     ("c3", "Langevin T = 0.16")], cols=2, colw=130))
    desc = ("Top: histograms of the final parameter for plain gradient descent and batch size 1, over a step curve of "
            "the log Fisher of each piece. Batch size 1 empties the steep central pieces. Bottom left: the paper's "
            "Shannon estimate falls with batch size, but so does the entropy of which piece is reached, by about the "
            "same amount; the information about the data itself stays flat. Bottom right: plain GD and isotropic "
            "Langevin both land in each piece in proportion to its width; only SGD noise shifts runs outward.")
    (out / "sgd-toy.svg").write_text(sp.svg(W, Hh, "Figure 3 toy, rebuilt", desc, "\n".join(body)))
    print("wrote figures/sgd-toy.svg")


if __name__ == "__main__":
    main()
