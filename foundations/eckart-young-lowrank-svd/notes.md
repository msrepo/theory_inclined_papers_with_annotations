---
title: "Eckart–Young–Mirsky and low-rank approximation via truncated SVD"
authors: "Background notes"
venue: "Foundations"
tags: [linear-algebra, svd, matrix-norms, low-rank-approximation, background]
status: living
category: "Foundations"
subcategory: "Linear algebra"
short_title: "Eckart–Young / low-rank SVD"
---

## Links

- **[Source: §3.5 Low-rank approximation, MATH3030 lecture notes, Rich Wilkinson (Nottingham)](https://rich-d-wilkinson.github.io/MATH3030/3.5-lowrank.html)** —
  the section these notes summarise, part of a larger set of course notes on matrix
  decompositions.
- **[Column, null & residual spaces](../four-fundamental-subspaces/index.html)** — what the kept and discarded singular
  vectors span: the column space of $\mathbf A_k$ and the PCA residual subspace.

## In one paragraph

The SVD writes any matrix as a sum of rank-1 pieces, $\mathbf A = \sum_{i=1}^r \sigma_i
\mathbf u_i \mathbf v_i^\top$, ordered by decreasing singular value. Truncating that sum
after the top $k$ terms gives a rank-$k$ matrix $\mathbf A_k$. The Eckart–Young–Mirsky
theorem says this truncation is not just *a* reasonable rank-$k$ approximation — it is
provably the *best* one, in both the spectral (2-)norm and the Frobenius norm, among all
rank-$k$ matrices. This is the fact underneath every use of "keep the top few singular
values/eigenvalues and drop the rest" as a compression or denoising step, including the
HaoChen et al. spectral-contrastive-loss argument in `papers/2021-haochen-spectral-contrastive`
(minimising $\lVert \bar A - FF^\top\rVert_F^2$ is exactly this problem, so its minimiser is
the top eigenvectors by Eckart–Young).

## Setup and notation

| Symbol | Meaning |
|---|---|
| $\mathbf A \in \mathbb{R}^{n\times p}$ | the matrix being approximated |
| $r = \operatorname{rank}(\mathbf A)$ | rank of $\mathbf A$ |
| $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ | singular values of $\mathbf A$, decreasing |
| $\mathbf u_i, \mathbf v_i$ | left/right singular vectors, $\mathbf A = \sum_i \sigma_i \mathbf u_i \mathbf v_i^\top$ |
| $\mathbf A_k = \sum_{i=1}^k \sigma_i \mathbf u_i \mathbf v_i^\top$ | the rank-$k$ truncated SVD |
| $\lVert\mathbf A\rVert_F$ | Frobenius (Hilbert–Schmidt) norm |
| $\lVert\mathbf A\rVert_2$ | spectral norm (2-norm), the induced operator norm |

## Matrix norms: two ways to measure "how far apart"

Comparing $\mathbf A$ to an approximation $\mathbf B$ needs a notion of distance
$\lVert \mathbf A - \mathbf B\rVert$ between matrices. Two norms matter here:

**Frobenius norm** — the Euclidean norm applied entrywise:

$$
\lVert \mathbf A \rVert_F = \left(\sum_{i,j} a_{ij}^2\right)^{1/2} = \left(\operatorname{tr}(\mathbf A^\top \mathbf A)\right)^{1/2}.
$$

It is invariant under multiplying by an orthogonal matrix ($\lVert \mathbf A\mathbf
U\rVert_F = \lVert \mathbf A\rVert_F$ for orthogonal $\mathbf U$, by cyclicity of the
trace), and in terms of the SVD it is simply

$$
\lVert \mathbf A\rVert_F = \left(\sum_{i=1}^r \sigma_i^2\right)^{1/2}.
$$

**Spectral norm** — the operator norm induced by the vector 2-norm,
$\lVert \mathbf A\rVert_2 = \sup_{\mathbf x \ne 0} \lVert \mathbf A\mathbf x\rVert_2 /
\lVert \mathbf x\rVert_2$. This turns out to equal the largest singular value,
$\lVert \mathbf A\rVert_2 = \sigma_1$: the most a unit vector can be stretched by $\mathbf
A$ is exactly the top singular value, achieved along the corresponding right singular
vector $\mathbf v_1$.

So the Frobenius norm sees *all* the singular values (it is their $\ell_2$ norm), while
the spectral norm only sees the largest one — the difference matters below, where the two
norms give different-looking but equally clean answers for the approximation error.

## Eckart–Young–Mirsky theorem

> For either $\lVert \cdot \rVert_2$ or $\lVert \cdot \rVert_F$,
> $$
> \lVert \mathbf A - \mathbf A_k \rVert \le \lVert \mathbf A - \mathbf B \rVert
> \quad\text{for every rank-}k\text{ matrix } \mathbf B,
> $$
> and the truncated SVD's error has a closed form:
> $$
> \lVert \mathbf A - \mathbf A_k \rVert =
> \begin{cases}
> \sigma_{k+1} & \text{2-norm} \\[4pt]
> \left(\sum_{i=k+1}^r \sigma_i^2\right)^{1/2} & \text{Frobenius norm.}
> \end{cases}
> $$

In words: among *every* matrix of rank $k$ — not just SVD truncations — none gets closer to
$\mathbf A$ than $\mathbf A_k$ does, in either norm. The error is exactly the part of the
spectrum that got thrown away: the next singular value for the 2-norm, or the root-sum-square
of all the discarded singular values for the Frobenius norm.

### Why it's true (2-norm case)

The proof is a clean pigeonhole argument. Let $\mathbf B$ be any rank-$k$ matrix (competing
against $\mathbf A_k$). Its null space $\mathcal N(\mathbf B) \subset \mathbb{R}^p$ has
dimension $p - k$ by rank–nullity. Consider the span of the *first* $k+1$ right singular
vectors of $\mathbf A$, $\mathcal C(\mathbf V_{k+1})$, which has dimension $k+1$. Since

$$
\dim \mathcal N(\mathbf B) + \dim \mathcal C(\mathbf V_{k+1}) = (p-k) + (k+1) = p + 1 > p,
$$

these two subspaces of $\mathbb{R}^p$ must intersect: there is a unit vector $\mathbf w$
that is simultaneously in $\mathbf B$'s null space and a combination of $\mathbf A$'s top
$k+1$ right singular vectors, $\mathbf w = \sum_{i=1}^{k+1} w_i \mathbf v_i$ with
$\sum w_i^2 = 1$. Then

$$
\lVert \mathbf A - \mathbf B\rVert_2^2 \ge \lVert(\mathbf A - \mathbf B)\mathbf w\rVert_2^2
= \lVert \mathbf A\mathbf w\rVert_2^2
= \sum_{i=1}^{k+1}\sigma_i^2 w_i^2
\ge \sigma_{k+1}^2 \sum_{i=1}^{k+1} w_i^2 = \sigma_{k+1}^2 = \lVert \mathbf A - \mathbf A_k\rVert_2^2,
$$

where $\mathbf B\mathbf w = 0$ kills the $\mathbf B$ term, and the last inequality uses that
$\sigma_1 \ge \dots \ge \sigma_{k+1}$. No rank-$k$ $\mathbf B$ can beat $\sigma_{k+1}$, and
$\mathbf A_k$ attains it — so $\mathbf A_k$ is optimal. The trick is entirely about forcing a
direction $\mathbf w$ that $\mathbf B$ cannot see ($\mathbf B\mathbf w = 0$) but that still
carries a chunk of $\mathbf A$'s energy at least as large as $\sigma_{k+1}^2$; the dimension
count $p+1 > p$ is what guarantees such a $\mathbf w$ always exists, for *any* rank-$k$
competitor.

## Example: image compression

Applying this to a $512\times512$ image (three colour-channel matrices) and truncating each
channel's SVD to $k$ terms gives a visibly reasonable reconstruction by around $k=30$–$100$,
while storing only the top $k$ singular values and singular vectors rather than the full
matrix ($k{=}100$ costs roughly a fifth of the naive storage). The point is not compression
mechanics but that Eckart–Young guarantees this truncation is the *provably best possible*
rank-$k$ reconstruction in Frobenius norm — you cannot do better with any other rank-$k$
matrix, not just any other SVD-based one.

## Questions and doubts

- The theorem gives the optimal *rank-k matrix*, but says nothing about whether the
  individual truncated singular vectors are individually meaningful or stable — in
  particular under perturbation of $\mathbf A$, which is where Weyl's inequality and
  Davis–Kahan (see `foundations/inequalities-and-concentration`) pick up the story.
- The 2-norm error depends only on $\sigma_{k+1}$ (the *next* singular value), while the
  Frobenius error depends on the whole discarded tail — so "how good is rank-$k$ good
  enough" can look very different depending on which norm the application actually cares
  about, and the source notes don't dwell on when that distinction bites in practice.

## Takeaways

- Truncating the SVD at $k$ terms, $\mathbf A_k = \sum_{i=1}^k \sigma_i \mathbf u_i
  \mathbf v_i^\top$, is the globally optimal rank-$k$ approximation to $\mathbf A$ in both
  the spectral and Frobenius norms — not merely a convenient heuristic.
- The optimal error is read directly off the discarded singular values: $\sigma_{k+1}$ for
  the 2-norm, $\big(\sum_{i>k}\sigma_i^2\big)^{1/2}$ for the Frobenius norm.
- This is the general fact behind any argument of the shape "the minimiser of $\lVert
  \mathbf M - \mathbf X\mathbf X^\top\rVert_F^2$ over low-rank $\mathbf X$ is given by the
  top eigenvectors of $\mathbf M$" — e.g. HaoChen et al.'s spectral contrastive loss.

---

*Notes started 2026-09-22.*
