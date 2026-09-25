---
title: "Principal angles between subspaces"
category: "Foundations"
subcategory: "Linear algebra"
short_title: "Principal angles"
authors: "Background notes"
venue: "Foundations"
tags: [linear-algebra, svd, subspaces, canonical-correlation, grassmannian, numerical-stability, background]
status: living
---

## Links

- **[Runnable code](https://github.com/msrepo/theory_inclined_papers_with_annotations/blob/main/foundations/principal-angles-subspaces/code/angles.py)** —
  the SVD characterisation, the projector identities, the two degenerate cases, the
  arccos accuracy trap, and the CCA equivalence. `make verify` runs it.
- **[Eckart–Young / low-rank SVD](../eckart-young-lowrank-svd/index.html)** — the other place
  singular values answer a geometric question about subspaces.
- **[Bao 2022 — H-score](../2022-bao-hscore-transferability/index.html)** — where this page
  came from: H-score turns out to be $\sum_i\cos^2\theta_i$ between feature space and label
  indicators.
- **[Column, null & residual spaces](../four-fundamental-subspaces/index.html)** — the four fundamental subspaces, projectors
  and residuals; principal angles compare two of the column spaces defined there.

## In one paragraph

Two lines in $\mathbb{R}^3$ meet at an angle. Two *planes* do not — they meet at a whole list of
angles, one per dimension of the smaller. **Principal angles** are that list: a recursively
defined sequence $0\le\theta_1\le\dots\le\theta_m\le\pi/2$, $m=\min(p,q)$, which turns out to be
computed by a single SVD. Nearly everything one wants to say about "how close are these two
subspaces" — the norm of the difference of their projectors, the distance on the Grassmannian,
the canonical correlations of two data blocks, the conclusion of Davis–Kahan — is a function of
that list. The one thing to remember beyond the definition is that the obvious way to compute
them, `arccos` of the singular values, silently destroys half your digits on small angles.

## Definition

Let $\mathcal{U},\mathcal{V}\subseteq\mathbb{R}^n$ with $\dim\mathcal{U}=p$,
$\dim\mathcal{V}=q$, and $m=\min(p,q)$. The principal angles
$0\le\theta_1\le\dots\le\theta_m\le\pi/2$ and principal vectors $u_k\in\mathcal{U}$,
$v_k\in\mathcal{V}$ are defined recursively by

$$
\cos\theta_k = \max_{\substack{u\in\mathcal{U},\,v\in\mathcal{V} \\ \lVert u\rVert=\lVert v\rVert=1 \\ u\perp u_1..u_{k-1},\; v\perp v_1..v_{k-1}}} u^\top v .
$$

In words: find the closest pair of unit directions, one from each subspace; record the angle;
then throw both away and repeat in what is left. $\theta_1$ is the smallest angle at which the
subspaces approach each other; $\theta_m$ the largest.

## The SVD characterisation

Let $Q_U\in\mathbb{R}^{n\times p}$, $Q_V\in\mathbb{R}^{n\times q}$ have **orthonormal columns**
spanning $\mathcal{U}$ and $\mathcal{V}$. Then (Björck–Golub, 1973)

$$
\boxed{\;\sigma_i\big(Q_U^\top Q_V\big) = \cos\theta_i\;}
$$

with singular values in descending order matching angles in ascending order.

**Why.** The operator norm is $\lVert Q_U^\top Q_V\rVert_2=\max_{\lVert x\rVert=\lVert y\rVert=1}
x^\top Q_U^\top Q_V y$, and as $x,y$ range over unit vectors, $u=Q_Ux$ and $v=Q_Vy$ range over
unit vectors of $\mathcal{U}$ and $\mathcal{V}$ (orthonormal columns preserve norms). So
$\sigma_1=\max u^\top v=\cos\theta_1$. The SVD's deflation — remove the top singular pair, repeat
on the orthogonal complement — is exactly the recursion in the definition.

**It is a property of the subspaces, not the bases.** Re-spanning with $Q_U\mapsto Q_UR$ for
orthogonal $R$ sends $Q_U^\top Q_V\mapsto R^\top Q_U^\top Q_V$, which has the same singular
values. The code re-bases both subspaces with random invertible mixing and the cosines move by
$4\times10^{-16}$.

## What the list is good for

Write $P_U=Q_UQ_U^\top$, $P_V=Q_VQ_V^\top$ for the orthogonal projectors. Then:

| quantity | equals | needs $p=q$? |
|---|---|---|
| $\lVert Q_U^\top Q_V\rVert_F^2 = \operatorname{tr}(P_UP_V)$ | $\sum_i\cos^2\theta_i$ | no |
| $\lVert P_UP_V\rVert_2$ | $\cos\theta_1$ | no |
| $\lVert P_U-P_V\rVert_2$ | $\sin\theta_m$ | yes |
| $\lVert P_U-P_V\rVert_F$ (chordal distance) | $\sqrt{2\sum_i\sin^2\theta_i}$ | yes |
| geodesic distance on the Grassmannian | $\sqrt{\sum_i\theta_i^2}$ | yes |

The first row is the one that recurs most. It follows in a line from cyclicity:

$$
\operatorname{tr}(P_UP_V)=\operatorname{tr}(Q_UQ_U^\top Q_VQ_V^\top)
=\operatorname{tr}\big(Q_U^\top Q_V\,Q_V^\top Q_U\big)=\lVert Q_U^\top Q_V\rVert_F^2 .
$$

The chordal one needs only one more step: $\lVert P_U-P_V\rVert_F^2=\operatorname{tr}P_U
+\operatorname{tr}P_V-2\operatorname{tr}(P_UP_V)=p+q-2\sum\cos^2\theta_i$, which at $p=q=m$ is
$2\sum(1-\cos^2\theta_i)=2\sum\sin^2\theta_i$.

**The two extremes.** If $\mathcal{U}\subseteq\mathcal{V}$ every angle is $0$. If the subspaces
are orthogonal every angle is $\pi/2$. Both are in the code.

## The accuracy trap

The natural implementation is `arccos(svd(QU.T @ QV))`. It is wrong for small angles, and
quietly so.

Near $\theta=0$, $\cos\theta = 1-\theta^2/2$. So a *small angle* produces a cosine
*indistinguishable from 1*: at $\theta=10^{-8}$ the cosine differs from one by $5\times10^{-17}$,
which is below double-precision resolution. The information is not merely degraded, it is absent
from the number you computed. Equivalently, $\arccos$ has infinite derivative at 1, so it
amplifies whatever error the SVD left behind by $1/\sqrt{2\varepsilon}$.

Rotating a subspace by a known angle and trying to recover it:

| true $\theta$ | via `arccos` | rel. err | via `arcsin` | rel. err |
|---|---|---|---|---|
| 1e−2 | 1.000e−02 | 4.6e−12 | 1.000e−02 | 2.6e−15 |
| 1e−4 | 1.000e−04 | 4.2e−08 | 1.000e−04 | 3.9e−13 |
| 1e−6 | 1.000e−06 | 4.4e−05 | 1.000e−06 | 3.8e−11 |
| 1e−8 | **2.581e−08** | **1.6e+00** | 1.000e−08 | 2.2e−09 |

At $10^{-8}$ the cosine route is wrong by 160%. Note the pattern: it loses roughly *half* the
available digits at every scale, which is the signature of a squared quantity.

**The fix**, also Björck–Golub: compute the **sines** instead. The singular values of
$(I-P_U)Q_V$ — the part of $\mathcal{V}$ that projecting out $\mathcal{U}$ fails to kill — are
$\sin\theta_i$. Sines carry full relative precision near zero where cosines carry none. Use
cosines for $\theta>\pi/4$ and sines below; that is the standard recipe, and it is what
`angles_stable` in the code does.

Two implementation notes the textbooks state and libraries get wrong often enough to be worth
repeating. The sine route returns $q$ singular values but there are only $\min(p,q)$ principal
angles — when $\dim\mathcal{V}>\dim\mathcal{U}$ the surplus values are exactly 1, the part of
$\mathcal{V}$ orthogonal to all of $\mathcal{U}$, and must be discarded. And the sines come out
in the opposite order to the cosines, so one list needs reversing. With both handled, the two
routes agree to $2\times10^{-15}$ at every shape tested.

## Where this shows up

**Canonical correlation analysis.** The canonical correlations between two centred data blocks
$X$ and $Y$ are *exactly* the cosines of the principal angles between their column spans. The
code computes them the textbook way — whiten each block, SVD the cross-covariance — and gets
`[0.94431366, 0.76544267, 0.33085384]`, matching the principal-angle cosines to all printed
digits. CCA is not *like* principal angles; it is principal angles, in statistical costume.

**Davis–Kahan.** The eigenspace perturbation theorem is stated as a bound on
$\sin\Theta$ between the true and perturbed invariant subspaces, in terms of the perturbation
size over the eigenvalue gap. The $\sin$ is not decorative — it is there for exactly the
numerical reason above.

**[H-score](../2022-bao-hscore-transferability/index.html).** Bao et al.'s transferability
metric, written $\operatorname{tr}(\Sigma_T^{-1}\Sigma_B)$, is
$\sum_i\cos^2\theta_i$ between the span of the centred features and the span of the class
indicators. Reading it that way makes three of its properties immediate: invariance under
invertible linear maps of the features (the basis changes, the subspace does not), invariance
under permuting class labels (permutes the indicator basis, not its span), and the bound
$\mathcal{H}<\min(k,\lvert\mathcal{Y}\rvert-1)$ — centring forces the all-ones vector out of the
feature span while it stays inside the indicator span, so one angle is always exactly $\pi/2$.
The code reproduces this: cosines `[0.7702, 0.6841, 0.5749, 0.4660, 0.0000]`, with the trailing
zero being that forced right angle.

## Caveats

- **The angles do not determine the subspaces.** They are a complete invariant only up to a
  simultaneous orthogonal change of basis. Two very different pairs of subspaces can share an
  angle list.
- **$\lVert P_U-P_V\rVert$ needs equal dimensions.** For $p\ne q$ the projector difference has
  norm 1 regardless of how the subspaces sit, because one of them has directions the other
  cannot reach. The $\operatorname{tr}(P_UP_V)$ row is the one that survives unequal dimensions,
  which is why H-score uses it.
- **Angles near $\pi/2$ have the mirror-image problem**: there the sine saturates and the cosine
  is the accurate one. The $\pi/4$ switch handles both ends.
- **$\cos\theta_1=1$ means the subspaces intersect**, not that they are equal. Their
  intersection has dimension equal to the number of zero angles.
