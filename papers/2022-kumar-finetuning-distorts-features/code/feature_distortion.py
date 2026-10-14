#!/usr/bin/env python3
"""Fine-tuning distorts pretrained features: a numpy check of Kumar et al. (ICLR 2022).

Section 3 of the paper studies the two-layer linear model f(x) = v^T B x, trained on
L(v, B) = || X B^T v - Y ||^2 by gradient flow from the pretrained feature extractor B0. This script
integrates that flow (RK4, small step) and checks, on random instances:

  1. Lemma A.3   B_ft(t) x = B0 x for every x orthogonal to the training span, at every t
  2. Lemma A.4   the balancedness invariant  v v^T - B B^T  is constant along the flow
  3. Step 1 of the proof of Theorem 3.3   ||v_ft - v*|| <= (Delta + eps ||v*||) / c
     and the identity (A.72)  (v_ft^T v*)^2 - (v0^T v*)^2 = ||B_ft^T v*||^2 - ||B0^T v*||^2
  4. Theorem 3.5 / Proposition 3.7   OOD error of LP, FT and LP-FT, along the whole trajectory
  5. Proposition 3.6   ID error of LP vs FT
  6. the toy 2-D picture of Figure 2

Only numpy. Run:   python3 feature_distortion.py            (prints every number quoted in the notes)
                   python3 feature_distortion.py --figures  (also rewrites ../figures/*.svg)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


# ------------------------------------------------------------------ the setting
def orth_rows(M):
    """Nearest matrix with orthonormal rows (polar factor)."""
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def make_problem(d=24, k=2, m=10, n=30, eps=0.1, seed=0, vnorm=1.0):
    """Well-specified overparameterised linear problem of Section 3.1.

    B* has orthonormal rows; the ID inputs live in an m-dim subspace S (columns of F); B0 is B*
    perturbed to distance ~eps and re-orthonormalised. Sigma = I on the OOD side.
    """
    rng = np.random.default_rng(seed)
    Bs = orth_rows(rng.standard_normal((k, d)))
    vs = rng.standard_normal(k)
    vs *= vnorm / np.linalg.norm(vs)
    F = np.linalg.qr(rng.standard_normal((d, m)))[0]          # basis of S
    X = rng.standard_normal((n, m)) @ F.T / np.sqrt(n)        # n training inputs, rows in S
    B0 = orth_rows(Bs + eps * orth_rows(rng.standard_normal((k, d))))
    return dict(d=d, k=k, m=m, n=n, Bs=Bs, vs=vs, ws=Bs.T @ vs, F=F, X=X, B0=B0,
                Y=X @ (Bs.T @ vs), eps_actual=np.linalg.norm(B0 - Bs, 2), rng=rng)


def loss(P, v, B):
    r = P["X"] @ B.T @ v - P["Y"]
    return r @ r


def grads(P, v, B):
    r = P["X"] @ B.T @ v - P["Y"]                  # n-vector
    return 2 * B @ (P["X"].T @ r), 2 * np.outer(v, r @ P["X"])     # dL/dv, dL/dB


def flow(P, v0, B0, train_B=True, T=300.0, h=0.01, record=20):
    """Gradient flow dv/dt = -dL/dv, dB/dt = -dL/dB (RK4). Returns the recorded path."""
    v, B = v0.copy(), B0.copy()

    def f(v, B):
        gv, gB = grads(P, v, B)
        return -gv, (-gB if train_B else np.zeros_like(B))

    path = [(0.0, v.copy(), B.copy())]
    for s in range(1, int(T / h) + 1):
        a1, b1 = f(v, B)
        a2, b2 = f(v + h / 2 * a1, B + h / 2 * b1)
        a3, b3 = f(v + h / 2 * a2, B + h / 2 * b2)
        a4, b4 = f(v + h * a3, B + h * b3)
        v = v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        B = B + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        if s % record == 0:
            path.append((s * h, v.copy(), B.copy()))
    return path


def l_ood(P, v, B):
    e = P["ws"] - B.T @ v
    return float(e @ e)                            # Sigma = I


def l_id(P, v, B):
    e = P["F"].T @ (P["ws"] - B.T @ v)             # error restricted to S, isotropic on S
    return float(e @ e)


def linear_probe(P, B):
    """Closed-form LP solution (B0 X^T X B0^T)^-1 B0 X^T Y."""
    A = P["X"] @ B.T
    return np.linalg.solve(A.T @ A, A.T @ P["Y"])


def cos_theta_max(R, S):
    """cos of the largest principal angle between row spaces R (k x d) and S (columns of S, d x s)."""
    E = np.linalg.qr(R.T)[0]
    return np.linalg.svd(E.T @ S, compute_uv=False)[-1]


def perp_basis(F):
    d, m = F.shape
    U = np.linalg.svd(F, full_matrices=True)[0]
    return U[:, m:]


# ------------------------------------------------------------------ experiments
def run_instance(P, seed_head=1, T=300.0, h=0.01, with_lp_flow=True):
    rng = np.random.default_rng(seed_head)
    v0 = rng.standard_normal(P["k"])
    vlp = linear_probe(P, P["B0"])
    out = {}
    out["FT"] = flow(P, v0, P["B0"], True, T, h)
    out["LP-FT"] = flow(P, vlp, P["B0"], True, T, h)
    if with_lp_flow:                       # LP is convex: its flow ends at the closed form vlp
        out["LP"] = flow(P, v0, P["B0"], False, T, h)
    return out, v0, vlp


def part1_invariants():
    print("== 1-3. Lemmas A.3, A.4 and the Step-1 inequality of Theorem 3.3 ==")
    P = make_problem(eps=0.2, seed=3)
    paths, v0, _ = run_instance(P)
    ft = paths["FT"]
    Sp = perp_basis(P["F"])
    dev_perp = max(np.linalg.norm((B - P["B0"]) @ Sp, 2) for _, _, B in ft)
    dev_S = max(np.linalg.norm((B - P["B0"]) @ P["F"], 2) for _, _, B in ft)
    print(f"max_t ||(B_ft(t)-B0) x|| over x in S-perp (unit):   {dev_perp:.2e}   (Lemma A.3: 0)")
    print(f"max_t ||(B_ft(t)-B0) x|| over x in S (unit):        {dev_S:.3f}")
    inv0 = np.outer(v0, v0) - P["B0"] @ P["B0"].T
    drift = max(np.abs(np.outer(v, v) - B @ B.T - inv0).max() for _, v, B in ft)
    print(f"max_t |v v^T - B B^T - (v0 v0^T - B0 B0^T)|:        {drift:.2e}   (Lemma A.4: 0)")
    c = cos_theta_max(P["B0"], Sp)
    eps = P["eps_actual"]
    worst = -1e9
    ident = 0.0
    for _, v, B in ft:
        Delta = np.linalg.norm(P["ws"] - B.T @ v)
        lhs = np.linalg.norm(v - P["vs"])
        rhs = (Delta + eps * np.linalg.norm(P["vs"])) / c
        worst = max(worst, lhs - rhs)
        ident = max(ident, abs((v @ P["vs"]) ** 2 - (v0 @ P["vs"]) ** 2
                               - (np.linalg.norm(B.T @ P["vs"]) ** 2 - np.linalg.norm(P["B0"].T @ P["vs"]) ** 2)))
    print(f"cos theta_max(R0, S-perp) = {c:.3f},  eps = ||B0 - B*|| = {eps:.3f}")
    print(f"max_t [ ||v-v*|| - (Delta + eps||v*||)/c ]:          {worst:.3f}   (Step 1: <= 0)")
    print(f"max_t |identity (A.72) residual|:                   {ident:.2e}   (0)")
    print()


def part2_trajectories():
    print("== 4-5. LP vs FT vs LP-FT on one instance (d=24, k=2, m=10, n=30, eps=0.1) ==")
    P = make_problem(eps=0.1, seed=0)
    paths, v0, vlp = run_instance(P)
    ood = {k_: np.array([l_ood(P, v, B) for _, v, B in p]) for k_, p in paths.items()}
    idl = {k_: np.array([l_id(P, v, B) for _, v, B in p]) for k_, p in paths.items()}
    print(f"eps = {P['eps_actual']:.3f},  ||w*|| = {np.linalg.norm(P['ws']):.2f},  "
          f"alignment error phi^2 = {abs((v0 @ P['vs'])**2 - (P['vs'] @ P['vs'])**2):.3f}")
    print(f"{'':7s}{'OOD at t=0':>12s}{'min_t OOD':>12s}{'OOD at end':>12s}{'ID at end':>12s}")
    for name in ("LP", "FT", "LP-FT"):
        print(f"{name:7s}{ood[name][0]:12.4f}{ood[name].min():12.4f}{ood[name][-1]:12.4f}{idl[name][-1]:12.2e}")
    print(f"LP min-OOD / FT min-OOD = {ood['LP'].min() / ood['FT'].min():.3f}")
    # features moved on ID vs OOD inputs
    Sp = perp_basis(P["F"])
    for name in ("FT", "LP-FT"):
        B = paths[name][-1][2]
        dID = np.linalg.norm((B - P["B0"]) @ P["F"], 2)
        dOOD = np.linalg.norm((B - P["B0"]) @ Sp, 2)
        print(f"{name:6s} feature change: on S {dID:.4f}, on S-perp {dOOD:.1e}")
    print()
    return P, paths, ood


SWEEP_EPS = (0.4, 0.2, 0.1, 0.05, 0.02)


def sweep_data(seeds=range(10)):
    """Per eps: medians over random problems of L_ood(LP), min_t L_ood(FT), L_ood(LP-FT at the end)."""
    res = []
    for eps in SWEEP_EPS:
        lp, ft, lf, ratio = [], [], [], []
        for s in seeds:
            P = make_problem(eps=eps, seed=100 + s)
            paths, v0, vlp = run_instance(P, seed_head=s, T=100.0, h=0.02, with_lp_flow=False)
            a = l_ood(P, vlp, P["B0"])
            b = min(l_ood(P, v, B) for _, v, B in paths["FT"])
            c = l_ood(P, paths["LP-FT"][-1][1], paths["LP-FT"][-1][2])
            lp.append(a); ft.append(b); lf.append(c); ratio.append(a / b)
        res.append((eps, float(np.median(lp)), float(np.median(ft)), float(np.median(lf)),
                    float(np.median(ratio))))
    return res


def part3_epsilon_sweep():
    print("== 4. Theorem 3.5: L_ood(LP) / min_t L_ood(FT) as the pretrained features improve ==")
    print("   (median over 10 random problems per row)")
    print(f"{'eps':>6s}{'LP':>11s}{'FT, best t':>13s}{'LP-FT, end':>13s}{'LP/FT ratio':>13s}")
    res = sweep_data()
    for eps, lp, ft, lf, ratio in res:
        print(f"{eps:6.2f}{lp:11.5f}{ft:13.5f}{lf:13.5f}{ratio:13.4f}")
    print()
    return res


def part4_perfect_features():
    print("== 4. Proposition 3.7: perfect features B0 = B* ==")
    P = make_problem(eps=0.0, seed=5)
    P["B0"] = P["Bs"].copy()
    paths, v0, vlp = run_instance(P)
    ood = {k_: np.array([l_ood(P, v, B) for _, v, B in p]) for k_, p in paths.items()}
    print(f"LP solution error ||v_lp - v*|| = {np.linalg.norm(vlp - P['vs']):.1e}")
    print(f"FT   (random head): OOD at t=0 {ood['FT'][0]:.3f}, min over t {ood['FT'].min():.4f}, at end {ood['FT'][-1]:.4f}")
    print(f"LP-FT             : max over t of OOD {ood['LP-FT'].max():.1e}")
    print(f"LP (head only)    : OOD at the end {ood['LP'][-1]:.1e}   (it starts at {ood['LP'][0]:.2f}, the random head)")
    print()


def toy_problem(a0_deg, astar_deg=45.0, scale=2.0):
    a0, ast = np.deg2rad(a0_deg), np.deg2rad(astar_deg)
    B0 = np.array([[np.cos(a0), np.sin(a0)]])
    ws = scale * np.array([np.cos(ast), np.sin(ast)])
    P = dict(X=np.array([[1.0, 0.0]]), Y=np.array([ws[0]]), ws=ws, F=np.array([[1.0], [0.0]]),
             vs=np.array([scale]), Bs=np.array([[np.cos(ast), np.sin(ast)]]), k=1)
    return P, B0, ws


def toy_errors(a0_deg, v0, astar_deg=45.0):
    P, B0, ws = toy_problem(a0_deg, astar_deg)
    ft = flow(P, np.array([v0]), B0, True, T=80.0, h=0.01)
    vf, Bf = ft[-1][1], ft[-1][2]
    wlp = B0.T @ linear_probe(P, B0)
    return abs(ws[1] - wlp[1]), abs(ws[1] - (Bf.T @ vf)[1]), wlp, Bf.T @ vf, Bf, ft


def part5_toy_2d():
    print("== 6. The 2-D toy of Figure 2 (d=2, k=1, ID data on the x-axis, w* at 45 deg, |w*| = 2) ==")
    err_lp, err_ft, wlp, wft, Bf, ft = toy_errors(50.0, 0.5)
    P, B0, ws = toy_problem(50.0)
    print(f"B0 at 50 deg, head v0 = 0.5:")
    print(f"  w* = ({ws[0]:.3f}, {ws[1]:.3f});  w_lp = ({wlp[0]:.3f}, {wlp[1]:.3f});  w_ft = ({wft[0]:.3f}, {wft[1]:.3f})")
    print(f"  error along the y-axis (OOD direction):  LP {err_lp:.3f}   FT {err_ft:.3f}")
    print(f"  B_ft row = ({Bf[0,0]:.3f}, {Bf[0,1]:.3f});  B0 row = ({B0[0,0]:.3f}, {B0[0,1]:.3f})  (y-coordinate untouched)")
    print("  y-error of LP / FT for other tilts of B0 and other initial heads (FT depends on v0, LP does not):")
    print(f"  {'B0 angle':>9s}{'LP':>8s}" + "".join(f"{'FT v0='+str(v):>12s}" for v in (0.0, 0.5, 1.0, 2.0, 3.0)))
    for a0 in (46, 50, 55, 60):
        row = [toy_errors(a0, v)[:2] for v in (0.0, 0.5, 1.0, 2.0, 3.0)]
        print(f"  {a0:7d}\u00b0{row[0][0]:8.3f}" + "".join(f"{r[1]:12.3f}" for r in row))
    print()
    return dict(B0=B0, ws=ws, wlp=wlp, wft=wft, Bf=Bf, ft=ft)


def part6_head_alignment():
    print("== 4. Theorem 3.3: the head's alignment error phi matters (eps = 0.1, 10 problems) ==")
    print("   head init v0 = (1-s) * random + s * v_lp;  s = 1 is LP-FT")
    print(f"{'s':>6s}{'phi^2 (median)':>16s}{'FT min_t OOD (median)':>24s}{'OOD at end (median)':>22s}")
    for s_ in (0.0, 0.25, 0.5, 0.75, 1.0):
        phis, best, endv = [], [], []
        for seed in range(10):
            P = make_problem(eps=0.1, seed=100 + seed)
            rng = np.random.default_rng(seed)
            vr = rng.standard_normal(P["k"])
            v0 = (1 - s_) * vr + s_ * linear_probe(P, P["B0"])
            path = flow(P, v0, P["B0"], True, 100.0, 0.02)
            vals = [l_ood(P, v, B) for _, v, B in path]
            phis.append(abs((v0 @ P["vs"]) ** 2 - (P["vs"] @ P["vs"]) ** 2))
            best.append(min(vals)); endv.append(vals[-1])
        print(f"{s_:6.2f}{np.median(phis):16.4f}{np.median(best):24.5f}{np.median(endv):22.5f}")
    print()


def part7_other_losses():
    """Lemmas A.3 and A.4 use only the shape  L = sum_i l(v^T B x_i):  re-check them with the logistic loss."""
    print("== 7. The two invariants for a different loss (logistic, labels sign(w*^T x)) ==")
    P = make_problem(eps=0.2, seed=3)
    y = np.sign(P["X"] @ P["ws"])
    X = P["X"]
    rng = np.random.default_rng(1)
    v, B = rng.standard_normal(P["k"]), P["B0"].copy()
    inv0 = np.outer(v, v) - B @ B.T
    B0 = B.copy()
    Sp = perp_basis(P["F"])

    def f(v, B):
        z = y * (X @ B.T @ v)
        g = -y / (1.0 + np.exp(z))                       # dl/d(output)
        return -(B @ (X.T @ g)), -np.outer(v, g @ X)

    h, drift, dev = 0.01, 0.0, 0.0
    for _ in range(3000):
        a1, b1 = f(v, B); a2, b2 = f(v + h/2*a1, B + h/2*b1)
        a3, b3 = f(v + h/2*a2, B + h/2*b2); a4, b4 = f(v + h*a3, B + h*b3)
        v = v + h/6*(a1 + 2*a2 + 2*a3 + a4); B = B + h/6*(b1 + 2*b2 + 2*b3 + b4)
        drift = max(drift, np.abs(np.outer(v, v) - B @ B.T - inv0).max())
        dev = max(dev, np.linalg.norm((B - B0) @ Sp, 2))
    print(f"after 30 time units: max drift of v v^T - B B^T = {drift:.1e};  max change of B on S-perp = {dev:.1e};  "
          f"change of B on S = {np.linalg.norm((B - B0) @ P['F'], 2):.3f}")
    print()


# ------------------------------------------------------------------ figures (themed SVG)
STYLE = """<style>
  text{font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  .lab{font-size:12px;fill:#57564f} .hd{font-size:13px;font-weight:500;fill:#1a1a19}
  .ax{stroke:#c3c2b7;stroke-width:0.6;fill:none}
  .gd{stroke:#e2e1da;stroke-width:0.5;fill:none;stroke-dasharray:4 4}
  .a{fill:none;stroke:#2f6fb5;stroke-width:2} .b{fill:none;stroke:#c2642a;stroke-width:2}
  .g{fill:none;stroke:#2a8a5c;stroke-width:2} .k{fill:none;stroke:#1a1a19;stroke-width:1.4;stroke-dasharray:5 3}
  .ia{fill:#2f6fb5} .ib{fill:#c2642a} .ig{fill:#2a8a5c} .ik{fill:#1a1a19}
  .vec{stroke-width:2.2;fill:none}
  @media (prefers-color-scheme: dark){
    .lab{fill:#b6b4ab} .hd{fill:#eceae3} .ax{stroke:#4a4844} .gd{stroke:#33312e}
    .a{stroke:#7fb2e8} .b{stroke:#e0955c} .g{stroke:#5fcf96} .k{stroke:#eceae3}
    .ia{fill:#7fb2e8} .ib{fill:#e0955c} .ig{fill:#5fcf96} .ik{fill:#eceae3}
  }
</style>"""


def _arrow(x0, y0, x1, y1, cls, label, dx=6, dy=-6):
    return (f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" class="vec {cls}"/>'
            f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="3.5" class="i{cls}"/>'
            f'<text class="lab" x="{x1+dx:.1f}" y="{y1+dy:.1f}">{label}</text>')


def fig_toy(T):
    W, H, s = 780, 330, 120.0
    ox, oy = 90.0, 250.0                       # origin; x to the right, y up
    px = lambda v: (ox + s * v[0], oy - s * v[1])
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img">',
             '<title>Two-dimensional toy: fine-tuning rotates the feature direction only along the training axis</title>',
             '<desc>The training data lies on the horizontal axis. The pretrained feature direction B0 is tilted from the true direction. Linear probing rescales B0 so the horizontal coordinate is right and keeps the vertical coordinate of B0. Fine-tuning instead slides the feature direction horizontally, so the vertical coordinate of its predictor is further from the truth.</desc>',
             STYLE]
    # panel A: LP
    for (i, (name, w, B, cls)) in enumerate([("(a) linear probing", T["wlp"], T["B0"], "a"),
                                             ("(b) fine-tuning", T["wft"], T["Bf"], "b")]):
        ox_i = ox + i * 360
        pxi = lambda v: (ox_i + s * v[0], oy - s * v[1])
        parts.append(f'<text class="hd" x="{ox_i-60}" y="30">{name}</text>')
        parts.append(f'<line x1="{ox_i-40}" y1="{oy}" x2="{ox_i+250}" y2="{oy}" class="ax"/>')
        parts.append(f'<line x1="{ox_i}" y1="{oy+20}" x2="{ox_i}" y2="{oy-210}" class="ax"/>')
        parts.append(f'<line x1="{ox_i}" y1="{oy}" x2="{ox_i+230}" y2="{oy}" stroke-width="5" class="ax" style="stroke:#2a8a5c;opacity:.5"/>')
        parts.append(f'<text class="lab" x="{ox_i+150}" y="{oy+18}">training data (x-axis)</text>')
        x0, y0 = pxi(T["B0"][0] * 1.0)
        parts.append(_arrow(ox_i, oy, x0, y0, "k", "B0", 6, -4) if i == 0 else
                     f'<line x1="{ox_i}" y1="{oy}" x2="{x0:.1f}" y2="{y0:.1f}" class="k"/><text class="lab" x="{x0+6:.1f}" y="{y0-4:.1f}">B0</text>')
        if i == 1:
            xb, yb = pxi(B[0] / np.linalg.norm(B[0]))
            parts.append(_arrow(ox_i, oy, xb, yb, "b", "B_ft (rotated)", 6, -4))
        xs, ys = pxi(T["ws"])
        xw, yw = pxi(w)
        parts.append(_arrow(ox_i, oy, xs, ys, "g", "w*", 6, 16 if i == 0 else -6))
        parts.append(_arrow(ox_i, oy, xw, yw, cls, "w_lp" if i == 0 else "w_ft", -40 if i == 0 else 6, -6 if i == 0 else 16))
        # vertical dotted line at the shared x coordinate
        parts.append(f'<line x1="{xs:.1f}" y1="{oy}" x2="{xs:.1f}" y2="{min(ys,yw)-6:.1f}" class="gd"/>')
        gap = abs(T["ws"][1] - w[1])
        parts.append(f'<line x1="{xs+14:.1f}" y1="{ys:.1f}" x2="{xs+14:.1f}" y2="{yw:.1f}" stroke="#b03030" stroke-width="2"/>')
        parts.append(f'<text class="lab" x="{xs+20:.1f}" y="{(ys+yw)/2+4:.1f}">error {gap:.2f}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def fig_curves(ood):
    W, H = 780, 340
    L, R, Tp, Bt = 70, 20, 40, 50
    tmax = 30.0                                       # all the action is in the first few time units
    tt = np.linspace(0, 300.0, len(ood["FT"]))
    keep = tt <= tmax
    tt = tt[keep]
    ood = {k_: np.asarray(v)[keep] for k_, v in ood.items()}
    ymin, ymax = -6.0, 0.6
    X = lambda v: L + (W - L - R) * v / tmax
    Y = lambda v: Tp + (H - Tp - Bt) * (ymax - np.log10(max(v, 1e-6))) / (ymax - ymin)
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img">',
             '<title>Out-of-distribution error along the training path of linear probing, fine-tuning and LP-FT</title>',
             '<desc>Log-scale squared error on out-of-distribution inputs against training time. Fine-tuning starts near the error of a random head, dips, and settles above the flat linear-probing level. LP-FT starts from the linear-probing head and stays close to that level.</desc>',
             STYLE, '<text class="hd" x="70" y="24">OOD error along the gradient-flow path (log scale)</text>']
    for e in range(-6, 1):
        y = Y(10.0 ** e)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" class="gd"/>')
        parts.append(f'<text class="lab" x="{L-8}" y="{y+4:.1f}" text-anchor="end">1e{e}</text>')
    parts.append(f'<line x1="{L}" y1="{H-Bt}" x2="{W-R}" y2="{H-Bt}" class="ax"/>')
    for tv in (0, 10, 20, 30):
        parts.append(f'<text class="lab" x="{X(tv):.1f}" y="{H-Bt+18}" text-anchor="middle">{tv}</text>')
    parts.append(f'<text class="lab" x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">training time t (first 30 of 300 units)</text>')
    for name, cls in (("FT", "b"), ("LP", "a"), ("LP-FT", "g")):
        pts = " ".join(f"{X(a):.1f},{Y(max(v,1e-6)):.1f}" for a, v in zip(tt, ood[name]))
        parts.append(f'<polyline class="{cls}" points="{pts}"/>')
    ly = Tp + 8
    for j, (name, cls) in enumerate((("fine-tuning (random head)", "b"), ("linear probing", "a"), ("LP-FT", "g"))):
        parts.append(f'<line x1="{480+j*0}" y1="{ly+j*18}" x2="{505}" y2="{ly+j*18}" class="{cls}"/>')
        parts.append(f'<text class="lab" x="512" y="{ly+j*18+4}">{name}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def fig_sweep(res):
    W, H = 780, 330
    L, R, Tp, Bt = 70, 20, 40, 50
    xs = np.log10([r[0] for r in res]); xlo, xhi = xs.min() - 0.1, xs.max() + 0.1
    ylo, yhi = -6.0, 0.0
    X = lambda e: L + (W - L - R) * (np.log10(e) - xlo) / (xhi - xlo)
    Y = lambda v: Tp + (H - Tp - Bt) * (yhi - np.log10(max(v, 1e-6))) / (yhi - ylo)
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img">',
             '<title>OOD error of linear probing and the best fine-tuning iterate against feature quality</title>',
             '<desc>Median over twenty random problems, log-log. As the pretrained features approach the truth the linear-probing error falls like the square of the distance, while the best fine-tuning error over the whole path falls much more slowly or stalls.</desc>',
             STYLE, '<text class="hd" x="70" y="24">Median OOD error vs. pretrained-feature error eps (median of 10 random problems)</text>']
    for e in range(-6, 1):
        y = Y(10.0 ** e)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" class="gd"/>')
        parts.append(f'<text class="lab" x="{L-8}" y="{y+4:.1f}" text-anchor="end">1e{e}</text>')
    parts.append(f'<line x1="{L}" y1="{H-Bt}" x2="{W-R}" y2="{H-Bt}" class="ax"/>')
    for r in res:
        parts.append(f'<text class="lab" x="{X(r[0]):.1f}" y="{H-Bt+18}" text-anchor="middle">{r[0]}</text>')
    parts.append(f'<text class="lab" x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">eps = ||B0 - B*||  (smaller = better features)</text>')
    for idx, cls, nm in ((1, "a", "linear probing"), (2, "b", "fine-tuning, best over the path"), (3, "g", "LP-FT")):
        pts = " ".join(f"{X(r[0]):.1f},{Y(r[idx]):.1f}" for r in res)
        parts.append(f'<polyline class="{cls}" points="{pts}"/>')
        for r in res:
            parts.append(f'<circle cx="{X(r[0]):.1f}" cy="{Y(r[idx]):.1f}" r="3.5" class="i{cls}"/>')
    for j, (cls, nm) in enumerate((("a", "linear probing"), ("b", "fine-tuning, best over the path"), ("g", "LP-FT"))):
        parts.append(f'<line x1="100" y1="{H-Bt-70+j*18}" x2="125" y2="{H-Bt-70+j*18}" class="{cls}"/>'
                     f'<text class="lab" x="132" y="{H-Bt-66+j*18}">{nm}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main():
    part1_invariants()
    P, paths, ood = part2_trajectories()
    res = part3_epsilon_sweep()
    part4_perfect_features()
    part6_head_alignment()
    part7_other_losses()
    T = part5_toy_2d()
    if "--figures" in sys.argv:
        FIG_DIR.mkdir(exist_ok=True)
        (FIG_DIR / "toy_2d.svg").write_text(fig_toy(T))
        (FIG_DIR / "ood_curves.svg").write_text(fig_curves(ood))
        (FIG_DIR / "epsilon_sweep.svg").write_text(fig_sweep(res))
        print("wrote figures/toy_2d.svg, ood_curves.svg, epsilon_sweep.svg")


if __name__ == "__main__":
    main()
