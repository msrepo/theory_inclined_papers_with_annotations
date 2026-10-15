---
title: "Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms"
category: "Misc"
short_title: "Beg 2005 — LDDMM"
authors: "Mirza Faisal Beg, Michael I. Miller, Alain Trouvé, Laurent Younes"
venue: "IJCV 61(2):139–157"
year: 2005
url: "https://doi.org/10.1023/B:VISI.0000043755.93987.AA"
pdf_url: ""
tags: [image-registration, diffeomorphism, lddmm, geodesic, euler-lagrange, rkhs, sobolev-gradient, computational-anatomy]
status: read
---

## Links

- **[DOI 10.1023/B:VISI.0000043755.93987.AA](https://doi.org/10.1023/B:VISI.0000043755.93987.AA)**: the paper (Springer, paywalled; no open copy is linked here).
- **[Interactive version](figures/interactive.html)**: a displacement map against a flow of the same velocity field (watch folding appear and disappear),
  the frequency factors that separate the L² gradient from the Hilbert gradient, and why constant speed minimises energy.
- **Earlier theory the paper builds on**: Dupuis, Grenander & Miller 1998; Trouvé 1995 and 1998 (existence of the flow and of minimisers).
- **The method it is compared with**: Christensen, Rabbitt & Miller 1996 (viscous-fluid flow, called "GEC" or "Christensen" in the paper).
- **Used in this repo by**: [Diffeomorphic registration with Lipschitz ResNets](../2022-joshi-lipschitz-resnet-registration/index.html), which compares its flow with LDDMM.
- **Background**: [the matrix exponential](../matrix-exponential/index.html) for the flow idea in its simplest, linear form;
  [Hilbert spaces](../hilbert-space/index.html) for what the space $V$ is.

Beg, Miller, Trouvé and Younes, *Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms*, IJCV 2005.

## In one paragraph

Registration looks for a warp $\varphi$ that bends a template image $I_0$ onto a target $I_1$. Finding $\varphi$ directly allows folds and tears,
and a single displacement field only works for small motions. The paper builds $\varphi$ as the end point of a flow: pick a time-varying
velocity field $v_t$, let every point ride it from time 0 to 1, and take $\varphi=\phi_1$. If $v_t$ is smooth, the flow cannot fold, so $\varphi$ is
a diffeomorphism by construction. The velocity is chosen to minimise "total smoothness cost along the way" plus "mismatch at the end".
The authors derive the equation the best velocity must satisfy (Euler–Lagrange), turn it into a gradient algorithm, and show that the
cost of the best path is a genuine distance between images.

## The picture, intuition first

**A warp as a journey.** Mark every point of the template. Over time $t\in[0,1]$ each point moves, and at each time its motion is read off a
velocity field: $\dot\phi_t(x)=v_t(\phi_t(x))$. After the journey the template has been carried to $I_0\circ\varphi^{-1}$. Smooth velocity means
neighbouring points move almost alike, so neighbours stay neighbours and nothing crosses. Compare the "big step" in section 1 of the
[interactive page](figures/interactive.html): the same field applied once, as $x+v(x)$, can turn cells inside out; applied as a flow it cannot.

**The cost of a journey.** Moving fast and roughly is expensive. Each moment costs $\lVert v_t\rVert_V^2$, where the norm $\lVert\cdot\rVert_V$ is large for
rough fields. The total cost $\int_0^1\lVert v_t\rVert_V^2\,dt$ is the energy of the path. The cheapest way of reaching a given warp is a
shortest path, a **geodesic**, and its length $\int_0^1\lVert v_t\rVert_V\,dt$ measures how far the warp is from doing nothing.

**The compromise.** Zero cost means no warp; perfect matching may cost a lot. So minimise
$$E(v)=\int_0^1\lVert v_t\rVert_V^2\,dt+\frac1{\sigma^2}\big\lVert I_0\circ\varphi_1^{-1}-I_1\big\rVert_{L^2}^2 \qquad(\text{their Eq. 1 and 8}).$$
$\sigma$ sets how much a leftover mismatch is tolerated: small $\sigma$ means matching dominates.

## Notation in plain words

- $\Omega\subset\mathbb R^n$: the image domain (2D or 3D). An image is a function $I:\Omega\to\mathbb R^d$.
- $\phi_t$: where each point is at time $t$ (a map $\Omega\to\Omega$), $\phi_0=\mathrm{id}$. $\varphi=\phi_1$ is the end warp.
- $\phi_{s,t}=\phi_t\circ\phi_s^{-1}$: "where a particle that is at $y$ at time $s$ is at time $t$". So $\phi_{t,1}$ takes a point at time $t$ to its end position,
  and $\phi_{t,0}$ takes a point at time $t$ back to where it started.
- $I_0\circ\varphi^{-1}$: the warped template. Reading $I_0$ at $\varphi^{-1}(x)$ means each output pixel $x$ asks "which template point ended up here?"
- $D\phi$: the Jacobian matrix of $\phi$ (how a small patch is stretched and turned at each point); $|D\phi|$: its determinant, the local volume factor
  (negative means the warp has flipped orientation, which is folding).
- $V$: the space of allowed velocity fields, with an inner product $\langle f,g\rangle_V=\langle Lf,Lg\rangle_{L^2}$. $L$ is a differential operator,
  usually $L=-\alpha\nabla^2+\gamma$ (Cauchy–Navier type). Rough fields have a large $\lVert Lf\rVert$, so they are penalised.
- $K$: the smoothing operator that undoes $L^\dagger L$ ($K(L^\dagger L)f=f$, Eq. 7). Applied to a force field it returns a smooth velocity field.
  It is built so that $\langle a,b\rangle_{L^2}=\langle Ka,b\rangle_V$ (Eq. 6): $K$ turns an $L^2$ inner product into a $V$ inner product.
  In practice $K$ is convolution with a smooth bump (a kernel); this is the reproducing-kernel idea behind the RKHS in the Joshi note.
- **Gateaux variation** $\partial_hE(v)$: the rate at which $E$ changes if you nudge the whole velocity path in direction $h$,
  $\lim_{\epsilon\to0}[E(v+\epsilon h)-E(v)]/\epsilon$. The **gradient** $\nabla_vE$ is the field for which
  $\partial_hE=\langle\nabla_vE,h\rangle$ for every nudge $h$. Which inner product you use (here $V$, not $L^2$) decides what the gradient looks like.
- $J^0_t=I_0\circ\phi_{t,0}$: the template carried forward to time $t$. $J^1_t=I_1\circ\phi_{t,1}$: the target pulled back to time $t$.
- **Right-invariant metric**: $\rho_G(\varphi_0,\varphi_1)=\rho_G(\varphi_0\circ\varphi,\varphi_1\circ\varphi)$, so the distance does not depend on how the
  image domain is relabelled by a further warp.

## Derivation of the Euler–Lagrange equation (Section 2), step by step

**Step 1: what a nudge to the velocity does to the flow (Lemma 2.1).** Nudge $v$ by $\epsilon h$ and ask how $\phi_{s,t}$ changes. The result (Eq. 3) is
$$\partial_h\phi^v_{s,t}=D\phi^v_{s,t}\int_s^t\big(D\phi^v_{s,u}\big)^{-1}\,h_u\circ\phi^v_{s,u}\,du.$$
Reading it: a nudge $h_u$ applied at time $u$ shifts the particle at that moment; $(D\phi_{s,u})^{-1}$ and $D\phi_{s,t}$ transport that shift
along the rest of the flow (a push at time $u$ is stretched by how the flow stretches space between $u$ and $t$). Adding up over $u$ gives the total shift.
The proof checks that this expression solves the linear ODE (their Eq. 4) satisfied by the derivative of the flow equation with respect to $\epsilon$.

**Step 2: the smoothness term.** $E^1(v)=\int\lVert v_t\rVert_V^2dt$ has $\partial_hE^1=2\int\langle v_t,h_t\rangle_V\,dt$, the derivative of a square.

**Step 3: the matching term.** With $E^2=\frac1{\sigma^2}\lVert I_0\circ\phi_{1,0}-I_1\rVert^2$, the chain rule gives
$\frac2{\sigma^2}\langle I_0\circ\phi_{1,0}-I_1,\ D(I_0\circ\phi_{1,0})\,\partial_h\phi_{1,0}\rangle$. Substitute Step 1, then change variables so that the integral
over space is expressed at time $t$ (the factor $|D\phi_{t,1}|$ is the Jacobian of that change of variables). The result is
$$\partial_hE^2=-\int_0^1\Big\langle\frac2{\sigma^2}\,|D\phi_{t,1}|\,\big(J^0_t-J^1_t\big)\,\nabla J^0_t,\ h_t\Big\rangle_{L^2}dt.$$

**Step 4: move the pairing from $L^2$ to $V$.** Because $\langle a,b\rangle_{L^2}=\langle Ka,b\rangle_V$, the same expression equals a $V$ pairing with $K$ applied to the force. So (Eq. 10)
$$\nabla_vE_t=2v_t-K\Big[\frac2{\sigma^2}|D\phi_{t,1}|\,\nabla J^0_t\,\big(J^0_t-J^1_t\big)\Big],$$
and the optimum satisfies $\nabla_vE_t=0$, which is Eq. 9: $v_t=\frac1{\sigma^2}K[\ldots]$.

**What the force means.** At time $t$, compare the template carried forward ($J^0_t$) with the target carried back ($J^1_t$): they are two images that
should coincide if the path is right. Where they differ, the term $\nabla J^0_t\,(J^0_t-J^1_t)$ is the classical optical-flow direction that moves template edges toward the target;
$|D\phi_{t,1}|$ rescales for volume change between $t$ and the end. Applying $K$ smooths this rough force into an admissible velocity. So the best path at every
instant is "move along a smoothed version of the correction needed to make the two time-$t$ images agree", and it meets in the middle rather than chasing the target.

## The algorithm (Section 3), in outline

Discretise time into $N$ steps with a piecewise-constant velocity. Repeat: compute $\phi_{t,0}$ forward and $\phi_{t,1}$ backward by transporting the identity
along $v$ (a **semi-Lagrangian** scheme: follow each grid point backwards along the field for one step and interpolate), form $J^0_t,J^1_t$, build the force,
apply $K$ by division in the Fourier domain (the symbol of $L^\dagger L$ is $A(k)^2$ with $A(k)=\gamma+2\alpha\sum_i(1-\cos(2\pi k_i/N_i))/\Delta x_i^2$), and take a gradient step.
Every 10 iterations, **re-time** the path to constant speed (below). Stop when the gradient is small.

**Why constant speed.** Cauchy–Schwarz gives $\int_0^1\lVert v_t\rVert^2dt\ge(\int_0^1\lVert v_t\rVert dt)^2$ with equality only for constant speed. Re-timing the path changes how fast it is
travelled but not where it goes, so $\phi_1$, the matching term and the length are all unchanged, while the energy drops to the minimum for that path. Geodesics are constant-speed,
so this speeds convergence. Section 3 of the [interactive page](figures/interactive.html) shows the gap as the variance of the speed.

## Why the gradient is taken in $V$ (Section 6.1)

The $L^2$ gradient is $2L^\dagger Lv_t+b_t$ and the $V$ (Hilbert or Sobolev) gradient is $K$ applied to it, $2v_t+Kb_t$, where $b_t$ is the matching force. In a basis of eigenfunctions of
$L^\dagger L$ with eigenvalues $\lambda_i\to\infty$, the $L^2$ gradient multiplies the high-frequency parts of $v_t$ by $\lambda_i$ (amplifying them), whereas the $V$ gradient divides the force's
parts by $\lambda_i$ (smoothing them). Large $\lambda_i$ make the $L^2$ problem stiff: a step size small enough for the highest frequency makes progress on the low ones crawl.
Section 2 of the interactive page plots the two factors.

## Metric, and comparison with Christensen's algorithm

**Metric.** With the right-invariant $\rho_G(\varphi_0,\varphi_1)=\inf\{\int_0^1\lVert v_t\rVert_V\,dt:\ \varphi_1=\phi^v_1\circ\varphi_0\}$ the paper defines
$\rho_I(I_0,I_1)=\inf\{\rho_G(\mathrm{id},\varphi):I_1=I_0\circ\varphi^{-1}\}$ on the orbit of images. Symmetry and the triangle inequality follow from right-invariance
(if $I_1=I_0\circ\varphi^{-1}$ and $I_2=I_1\circ\psi^{-1}$ then $I_2=I_0\circ(\psi\varphi)^{-1}$ and $\rho_G(\mathrm{id},\psi\varphi)\le\rho_G(\mathrm{id},\varphi)+\rho_G(\varphi,\psi\varphi)$,
with $\rho_G(\varphi,\psi\varphi)=\rho_G(\mathrm{id},\psi)$); positivity needs that the infimum is attained, which is cited to Younes 1999.

**GEC (Section 5).** Christensen's method solves, at each time step independently, the local problem $L^\dagger Lv_t+b_t=0$ with force $b_t=-\zeta(J^0_t-J^1_1)\nabla J^0_t$, then integrates forward. The authors show it is a
Riemannian gradient descent on the matching term alone along the manifold of diffeomorphisms, taking the locally best step at each moment (greedy). It gives a path to a matching warp but not the shortest one,
and it is not the solution of a global variational problem, so it cannot give the metric.

## Results reported

| Experiment (2D) | $L$ | Image error left | Estimated distance | Iterations | Time |
|---|---|---|---|---|---|
| Parallel translation | $-0.01\nabla^2+0.1I$ | 1.97% | 0.769 | 70 | 0.2 min |
| Heart (canine, normal vs failing) | $-0.01\nabla^2+I$ | 7.12% | 6.122 | 159 | 3.7 min |
| Schizophrenia hippocampus | $-0.01\nabla^2+I$ | 3.01% | 4.891 | 277 | 4.2 min |
| Alzheimer hippocampus | $-0.01\nabla^2+I$ | 2.69% | 5.592 | 255 | 3.9 min |
| Macaque cortex | $-0.01\nabla^2+I$ | 3.64% | 6.869 | 134 | 3.2 min |

| LDDMM against GEC at comparable final image error | Final error LDDMM / GEC | Path length LDDMM / GEC | Time (s) LDDMM / GEC |
|---|---|---|---|
| Parallel translation | 5.648% / 5.65% | 0.7223 / 0.7228 | 12 / 1 |
| Heart | 11.558% / 11.799% | 4.631 / 6.7145 | 120 / 2.4 |
| Macaque cortex | 9.55% / 9.53% | 4.636 / 5.45 | 98 / 1.2 |

The mitochondria experiment (Fig. 8) orders eight target shapes by estimated distance, which the authors say agrees with human judgements of near and far. 3D hippocampus volumes
(80×128×128) take half an hour to a few hours on 8 processors.

## Questions and doubts

- **The Euler–Lagrange equation is necessary, not sufficient, and its existence assumptions are outside this paper.** The paper "assumes sufficient smoothness of the norm"
  and cites Dupuis et al. and Trouvé for the flow being a diffeomorphism and for a minimiser existing. Theorem 2.1 also asks for a continuously differentiable template, which real
  (discretised, discontinuous) images are not, so the theorem is for an idealised $I_0$. Because the energy is non-convex in $v$, gradient descent finds a critical point, not necessarily the global minimum.
- **The smoothness the theory asks for is not the smoothness used in the experiments.** The introduction takes $L=(-\alpha\nabla^2+\gamma)^\beta$ with $\beta>1.5$ in 3D so that flows are diffeomorphic;
  every experiment uses $L=-\alpha\nabla^2+\gamma$, that is $\beta=1$ (including the 3D one). By my count $\beta=1$ gives a space of fields with only two derivatives, which in 3D is short of the three needed
  for a guaranteed $C^1$ flow, so the guarantee quoted does not directly cover the runs. On a finite grid the question is partly moot, but then the folding-free claim is empirical.
- **The distance in the tables is not a distance between $I_0$ and $I_1$.** Because $E$ allows a mismatch, the reported length is that of the path to $\varphi_1$ with $I_0\circ\varphi_1^{-1}\neq I_1$ (image errors of 2–12% remain).
  The paper says as much in Section 6.2 (it measures $I_0$ to $I_0\circ\phi_{1,0}$, an approximation of the exact-matching metric), but the figure captions simply say "distance from $I_0$" and the value depends on $\sigma$,
  which the paper never reports or varies.
- **The metric argument is a sketch.** Positivity ($\rho_I=0\Rightarrow I_0=I_1$) needs the infimum to be attained and is delegated to Younes 1999. The remark that triangle inequality needs invariance "to the stabilizer of the
  template" is stated, but the short argument above shows right-invariance is what is used.
- **The comparison with GEC is single-run and trades a lot of time.** LDDMM takes 10 to 80 times longer in Table 2 (12 s against 1 s, 120 against 2.4, 98 against 1.2). The length advantage is large for the heart (4.63 against 6.71) and
  macaque (4.64 against 5.45) but absent for the parallel translation (0.7223 against 0.7228). The final errors are only "comparable" (11.56% against 11.80% for the heart), and stopping when errors are similar
  means the path lengths depend on the stopping rule. Three 2D slices are all that is shown; no variability is given.
- **Smoothness in time is shown by eye.** The claim that LDDMM velocities are smooth in time while GEC's are not rests on superposed velocity plots (Figs. 6 and 11), with no measure.
- **"Much more stable" is asserted.** Section 6.1 says the authors implemented both gradients and found the Sobolev gradient more stable, but no numbers, curves or step sizes are given. The frequency argument is sound as far as it goes
  (the factor $\lambda_i\to\infty$ is exact), but how much it matters at the grid sizes used is not shown.
- **Theory uses compactly supported fields; the code uses periodic boundaries.** The Fourier inversion of $L^\dagger L$ assumes a periodic domain, so a flow can leave one side and re-enter the other, whereas the
  theory has fields that vanish at the boundary. Data are presumably padded so this does not matter; the paper does not say.
- **Time steps.** Section 4 says the flow is cut into 20 steps "of length $\delta t=0.1$", which would span $[0,2]$, not $[0,1]$; I take it to be a slip for 0.05. The figures show steps $j=6,12,19$ of 20.
- **"Non-dissipative" semi-Lagrangian scheme.** The introduction contrasts it with dissipative Eulerian schemes, but with bilinear or trilinear interpolation (which the paper uses) a semi-Lagrangian step averages neighbouring values,
  which smooths. How much that blurs $J^0_t$ and $J^1_t$ over 20 steps is not examined.
- **Rigid alignment is done outside.** All but the first experiment were rigidly aligned beforehand with other software, as the metric is not invariant to rigid motion; so reported distances include whatever that preprocessing left over.

## Takeaways

- A diffeomorphism is obtained for free by flowing a smooth velocity field; the work is in choosing the field.
- The Euler–Lagrange equation says the optimal velocity is a smoothed correction force that compares the template carried forward with the target carried back, at every time.
- Taking the gradient in $V$ (apply $K$) is what makes the descent well-behaved, since the raw $L^2$ gradient amplifies high frequencies.
- The squared length of the optimal path is a distance between images, defined by the best warp rather than by pixel differences.
- Cost: many time steps and many iterations, which is why later work (stationary velocity fields, scaling and squaring, learned registration such as VoxelMorph and the Joshi paper) trades the path for speed.
