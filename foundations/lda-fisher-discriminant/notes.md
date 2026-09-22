---
title: "Linear (Fisher) Discriminant Analysis"
authors: "Background notes"
venue: "Foundations"
tags: [classification, dimensionality-reduction, scatter-matrices, generalised-eigenproblem, background]
status: living
category: "Foundations"
short_title: "LDA / Fisher discriminant"
---

## Links

- **[Source: Lecture 15, DOC493 Intelligent Data Analysis and Probabilistic Inference, Imperial College London](https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/old_IDAPILecture15.pdf)** —
  the lecture these notes summarise. A local copy also sits in `Theory_oriented_papers/` in
  this repo's parent folder.

## In one paragraph

PCA finds the projection that best represents the data as a whole, by diagonalising the
overall covariance matrix. Linear Discriminant Analysis (LDA), due to Fisher, instead finds
the projection that best *separates* labelled classes: it maximises the spread between
class means relative to the spread within each class. Both are eigenproblems on scatter
matrices; the difference is which scatter matrix is being diagonalised, and against what.
LDA is the projection step several papers in this repo lean on implicitly whenever they
talk about "class means well separated relative to within-class variance" (e.g. the
alignment/uniformity and spectral-contrastive-loss lines of argument, which optimise for
something structurally similar without calling it LDA).

## Setup and notation

| Symbol | Meaning |
|---|---|
| $g$ | number of classes |
| $N_i$ | number of samples in class $\pi_i$; $N = \sum_i N_i$ |
| $\overline{x_i}$ | mean of class $i$ |
| $\overline{x}$ | grand mean over all samples |
| $\Sigma_i$ | sample covariance of class $i$ |
| $S_b$ | between-class scatter matrix |
| $S_w$ | within-class (pooled) scatter matrix |
| $\Phi$ | projection basis (columns are projection directions) |
| $\Lambda$ | diagonal matrix of eigenvalues |

## From PCA to LDA

PCA projects mean-centred data $U$ (rows are samples) through an orthonormal basis
$\Phi = [\phi_1, \dots, \phi_m]$ so that the projected covariance $\Phi^\top \Sigma \Phi$ is
diagonal. The basis is the eigenvectors of $\Sigma$, and it is chosen without reference to
any class labels — it is optimal for *representing* the data, not for telling classes apart.

LDA asks a different question: given labelled classes, which projection keeps class means
far apart while keeping each class tight around its own mean? A projection that is good for
PCA can be useless for classification (it may point straight along the direction the two
classes overlap most), and vice versa.

## Between- and within-class scatter

For $g$ classes, define the between-class scatter

$$
S_b = \sum_{i=1}^g N_i (\overline{x_i} - \overline{x})(\overline{x_i} - \overline{x})^\top
$$

and the within-class (pooled) scatter

$$
S_w = \sum_{i=1}^g (N_i - 1)\Sigma_i = \sum_{i=1}^g \sum_{j=1}^{N_i} (x_{i,j} - \overline{x_i})(x_{i,j} - \overline{x_i})^\top .
$$

$S_b$ has rank at most $g - 1$ (it is built from only $g$ class means), which is why LDA can
never produce more than $g-1$ discriminant directions, no matter how high-dimensional the
data is. $S_w$ pools the individual class covariances and has rank at most $N - g$.

## Fisher's criterion

LDA picks the projection $\Phi_{\text{lda}}$ maximising

$$
\Phi_{\text{lda}} = \arg\max_\Phi \frac{|\Phi^\top S_b \Phi|}{|\Phi^\top S_w \Phi|}.
$$

The determinant of a covariance-like matrix is a proxy for how spread out the data is along
the axes it defines (for a diagonal matrix it is literally the product of per-axis
variances, and it is invariant to which orthonormal basis you diagonalise in). So this
ratio is exactly "spread of the class means" over "spread within each class" — maximise the
former, minimise the latter.

The solution is the generalised eigenproblem

$$
S_b \Phi = S_w \Phi \Lambda,
$$

and, when $S_w$ is invertible, this is equivalent to the ordinary eigenproblem

$$
S_w^{-1} S_b \, \Phi = \Phi \Lambda,
$$

so $\Phi_{\text{lda}}$ is the (generalised) eigenvectors of $S_w^{-1}S_b$, ordered by
eigenvalue. Because $S_b$ has rank at most $g-1$, only $g-1$ of these eigenvalues are
non-zero — LDA collapses an $n$-dimensional problem to at most $g-1$ discriminant axes,
regardless of $n$.

## The small-sample failure mode, and the fix

$S_w$ is singular whenever $N < n + g$ (too few samples relative to dimensionality — the
classic "more variables than data points" regime), which makes $S_w^{-1}$ undefined. The
standard fix is a two-stage projection: first reduce dimensionality with PCA to some
$p$-dimensional subspace $\Phi_{\text{pca}}$, then run LDA inside that subspace,

$$
\Phi_{\text{lda}} = \arg\max_\Phi
\frac{|\Phi^\top \Phi_{\text{pca}}^\top S_b \Phi_{\text{pca}} \Phi|}
     {|\Phi^\top \Phi_{\text{pca}}^\top S_w \Phi_{\text{pca}} \Phi|}.
$$

Choosing $g \le p \le N - g$ principal components keeps $\Phi_{\text{pca}}^\top S_w
\Phi_{\text{pca}}$ a $p\times p$ matrix estimated from $N - g$ independent observations,
which is invertible — so an LDA solution always exists, at the cost of discarding some
variance in the initial PCA step.

## Questions and doubts

- Fisher's criterion implicitly assumes all classes share the same true covariance (the
  homoscedastic assumption baked into $S_w$ being a single pooled matrix); it is not stated
  how badly LDA degrades when that assumption is wrong, only that it is a "main limitation."
- The two-stage PCA+LDA fix trades away exactly the low-variance directions PCA discards —
  but those could in principle be exactly the directions that best separate two classes with
  similar overall variance and only a small mean shift. The notes don't address this
  tension.

## Takeaways

- PCA and LDA are both eigenproblems on scatter matrices, but PCA diagonalises the
  unlabelled covariance while LDA solves a generalised eigenproblem $S_b\Phi = S_w\Phi\Lambda$
  that trades off between-class against within-class scatter.
- LDA yields at most $g-1$ useful directions because $S_b$ has rank $g-1$ — a hard cap set
  purely by the number of classes, independent of the ambient dimension.
- Small-sample, high-dimension regimes make $S_w$ singular; PCA-then-LDA is the standard
  patch, valid whenever the PCA subspace dimension $p$ satisfies $g \le p \le N-g$.

---

*Notes started 2026-09-22.*
