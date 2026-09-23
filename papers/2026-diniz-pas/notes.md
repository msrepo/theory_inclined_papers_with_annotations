---
title: "PAS: Estimating the target accuracy before domain adaptation"
category: "Transferability"
subcategory: "Theory"
short_title: "Diniz 2026 — PAS"
authors: "Raphaella Diniz, Jackson de Faria Júnior, Martin Ester (Simon Fraser University)"
venue: "ICLR"
year: 2026
url: "https://arxiv.org/abs/2604.09863"
pdf_url: "https://arxiv.org/pdf/2604.09863"
tags: [transferability, domain-adaptation, unsupervised, silhouette, cosine-similarity, source-selection, nearest-centroid]
status: read
---

## Links

- **[arXiv:2604.09863](https://arxiv.org/abs/2604.09863)** — preprint; ICLR 2026.
- The measures it cannot use, and says so:
  [Bao 2022 — H-score](../2022-bao-hscore-transferability/index.html),
  [Nguyen 2020 — LEEP](../2020-nguyen-leep/index.html),
  [Tran 2019 — NCE](../2019-tran-nce-hardness/index.html) — all three need target labels.
- **[You 2021 — LogME](../2021-you-logme/index.html)** — the Bayesian-evidence measure,
  which needs only a feature extractor and so also covers regression and contrastive or
  language-model sources.
- **[Claßen 2026 — TE robustness](../2026-classen-te-robustness/index.html)** — on how
  fragile rankings from measures of this kind are under resampling.
- **[Runnable code](https://github.com/msrepo/theory_inclined_papers_with_annotations/blob/main/papers/2026-diniz-pas/code/pas.py)** —
  Eq. 1 as an argmax, the range and its non-zero null, the exact PAS/Oracle relationship, the
  blind spot that relationship implies, and the class-count dependence. `make verify` runs it.

## In one paragraph

Unsupervised domain adaptation has no target labels, so every transferability measure in the
[Theory](../2022-bao-hscore-transferability/index.html) section is unavailable — each of them
needs a target partition. PAS builds one out of the *source* instead: form a centroid per source
class in a frozen embedding, and for each unlabelled target sample measure how much closer it is
to its nearest centroid than to the runner-up. Average that relative margin and you have the
score. It is a Silhouette score with the unavailable "own cluster" replaced by "nearest cluster",
and it is deliberately **asymmetric**, unlike MMD or Wasserstein, because transferring easy→hard
is not the same problem as hard→easy. The construction is clean and the empirical correlations
are real; the cost of the substitution is that the score can no longer represent a target sample
being in the *wrong* place, which is precisely its reported failure case.

## The spine of the argument

1. A good pre-trained embedding puts same-class samples together regardless of domain.
2. So if target samples land decisively inside single source class clusters, the embedding has
   found domain-invariant discriminative features and adaptation should go well.
3. Measure that decisiveness as a relative margin: nearest versus second-nearest source centroid.
4. Average over the target set. Rank candidate (source, backbone) pairs by it.

## Setup and notation

| Symbol | Meaning |
|---|---|
| $\mathcal{D}^S=\{(x^S_i,y^S_i)\}$ | **labelled** source set |
| $\mathcal{D}^T=\{x^T_j\}$ | **unlabelled** target set — no $y^T$ anywhere |
| $f_\theta:\mathcal{X}\to\mathcal{Z}$ | frozen pre-trained feature extractor |
| $S^S_c$ | source samples of class $c$; $C$ classes, shared by both domains |
| $\mu_c$ | the source class-$c$ centroid |
| $d_{1i}, d_{2i}$ | smallest and second-smallest distance from target $i$ to any centroid |

All embeddings are normalised to unit length, so everything happens in cosine geometry.

## The construction

**Equation 1 — spherical class centroids.**

$$
\mu_c = \frac{\sum_{x^S_i\in S^S_c} f_\theta(x^S_i)}{\big\lVert\sum_{x^S_i\in S^S_c} f_\theta(x^S_i)\big\rVert}
$$

The paper cites Dhillon & Modha and asserts this "represents the vector that, on average, has the
highest cosine similarity to all the samples in the cluster". It does not derive it, and the
derivation is one line of Cauchy–Schwarz:

$$
\max_{\lVert\mu\rVert=1}\ \tfrac1n\sum_i x_i\cdot\mu
\;\Longleftrightarrow\;
\max_{\lVert\mu\rVert=1}\ \Big(\sum_i x_i\Big)\cdot\mu ,
\qquad
\Big(\sum_i x_i\Big)\cdot\mu \le \Big\lVert\sum_i x_i\Big\rVert
$$

with equality iff $\mu\parallel\sum_i x_i$. Checked against 200,000 random unit vectors; none
beats it.

**Equation 2 — cosine distance to each centroid.**

$$
\operatorname{dist}(f_\theta(x^T_i),\mu_c) = 1 - f_\theta(x^T_i)\cdot\mu_c \;\in[0,2]
$$

**Equation 3 — the score.** Sort the $C$ distances for target sample $i$ ascending, take the two
smallest as $d_{1i}$ and $d_{2i}$, and

$$
\boxed{\;\mathrm{PAS}(\theta,\mathcal{D}^S,\mathcal{D}^T)
= \frac{1}{\lvert\mathcal{D}^T\rvert}\sum_i \frac{d_{2i}-d_{1i}}{d_{2i}}\;}
$$

### Rewriting it makes the meaning obvious

$$
\frac{d_2-d_1}{d_2} = 1-\frac{d_1}{d_2}
\qquad\Longrightarrow\qquad
\mathrm{PAS} = 1 - \mathbb{E}_i\!\left[\frac{d_{1i}}{d_{2i}}\right]
$$

**One minus the average nearest-to-second-nearest distance ratio.** That statistic has a name in
another field: it is **Lowe's ratio test** from SIFT matching, where a correspondence is accepted
when $d_1/d_2$ falls below a threshold. PAS is its mean complement over the target set.

So operationally: *build a nearest-class-mean classifier from the source classes, run it on the
unlabelled target data, and report its average relative margin.* It is a **confidence** measure
for a nearest-centroid classifier — with no check on correctness, which is the crux.

## The Silhouette derivation, and what the modification costs

The paper says PAS is "inspired by" the Silhouette score. Spelling out the modification is worth
it, because one consequence is the method's central limitation.

$$
s(i) = \frac{b(i)-a(i)}{\max\{a(i),b(i)\}}\in[-1,1],
\quad
\begin{cases}
a(i) &= \text{mean distance within } i\text{'s \textit{own} cluster}\\
b(i) &= \text{min over other clusters}
\end{cases}
$$

| | Silhouette | PAS |
|---|---|---|
| clusters | the points' own | **source** clusters, scoring **target** points |
| $a$ | the true class — needs labels | $d_1$, the **nearest** cluster, *assumed* correct |
| $b$ | nearest other cluster | $d_2$, the second nearest |
| distance | mean pairwise within cluster | to the centroid |
| range | $[-1,1]$ | $[0,1]$ |

The third row is load-bearing. Because $d_1,d_2$ are the two smallest of a *sorted* list,
$d_1\le d_2$ **by construction**, so $\max\{a,b\}=d_2$ always and the denominator collapses.
Equation 3's lopsided look is not a design choice — it is forced by the substitution.

And the consequence, which the paper notes only as a range restriction:

> **Silhouette can go negative, and that is precisely how it says "this point is in the wrong
> cluster". PAS cannot. In defining the nearest cluster to be the true one, it discarded the
> ability to represent misassignment along with the labels it never had.**

## Range, endpoints, and a null that is not zero

| target configuration | PAS |
|---|---|
| sitting exactly on the source centroids | 1.000000 |
| equidistant between two centroids | 0.000000 |
| **unrelated to the source structure** | **0.186** |

The first two are the interpretable endpoints. The third is the one to remember: a structureless
embedding does not score zero. **PAS carries no absolute meaning** — only a ranking against a
fixed target — and the paper never states the null. This resurfaces below.

## The oracle, and what the gap between them measures

Their oracle baseline swaps "nearest" for "true":

$$
\mathrm{Oracle} = \frac{1}{\lvert\mathcal{D}^T\rvert}\sum_i
\frac{d_{2i}-d_{1i}}{\max\{d_{1i},d_{2i}\}},
\quad
\begin{cases}
d_{1i} &= \text{distance to the \textit{true} class centroid}\\
d_{2i} &= \text{distance to the nearest non-true one}
\end{cases}
$$

The $\max$ returns because $d_1>d_2$ is now possible. The relationship, asserted by the paper and
confirmed here:

| shift | PAS | Oracle | nearest-centroid accuracy |
|---|---|---|---|
| clean | 0.8271 | **+0.8271** | 1.000 |
| moderate | 0.2894 | +0.1749 | 0.712 |
| severe | 0.2081 | **−0.0606** | 0.418 |

Numerically identical when every nearest centroid is the true class, and strictly below
otherwise. So **$\mathrm{PAS}-\mathrm{Oracle}$ is a direct measure of how often the
nearest-centroid assignment is wrong** — exactly the quantity PAS cannot see for itself.

## The blind spot, made concrete

Place every target sample tightly around a *wrong*-class centroid:

```
PAS                              0.8505    high
true nearest-centroid accuracy   0.0000    chance would be 0.20
Oracle                          -0.8838    sees it immediately
```

PAS is confidently, maximally wrong. This is not a contrived case; it is the paper's own
ImageCLEF result, reported honestly:

> "the sample is very close to the centroid of one class that is indeed present in the image, but
> the true label is related to another object in the scene. In these cases, the PAS for the
> sample is high... but the final accuracy is low."

PAS measures **decisiveness**, not **correctness**, and the two separate exactly when a confident
embedding is confidently mistaken. ImageCLEF's 0.44 Pearson — their weakest benchmark — is this.

## Where it sits relative to the other three

| | [H-score](../2022-bao-hscore-transferability/index.html) | [LEEP](../2020-nguyen-leep/index.html) / [NCE](../2019-tran-nce-hardness/index.html) | **PAS** |
|---|---|---|---|
| needs target labels | yes | yes | **no** |
| setting | supervised transfer | supervised transfer | **unsupervised DA** |
| centroids from | **target** classes | — | **source** classes |
| what is measured | between/total scatter ratio | likelihood of a fixed head | margin to nearest source centroid |
| symmetric | — | — | no, deliberately |

The structural link worth seeing: H-score *is* the Fisher ratio, which is a nearest-class-mean
statistic computed on target labels. **PAS is what that idea becomes when the target labels
vanish and the source partition stands in for them.** Both ask "how well separated are class
centroids relative to spread"; they differ only in whose partition supplies the centroids.

One thing PAS never uses, though it is freely available: **the target's own cluster structure.**
It never checks whether target points that ought to share a class actually group together. An
unsupervised clustering term is the obvious extension and is not tried.

## Questions and doubts

- **The headline pooled correlation looks confounded by label-space size.** Table 3 reports
  per-benchmark Pearson of 0.76, 0.63, 0.44, 0.53 and a **Total of 0.83** — higher than every
  component, which is the signature of pooling across groups differing in a third variable. Here
  that variable is $C$, and PAS depends on it mechanically at fixed embedding quality:

  | $C$ | 12 | 31 | 65 | 345 |
  |---|---|---|---|---|
  | PAS | 0.1179 | 0.0909 | 0.0782 | 0.0580 |

  Only the label-space size changes across those rows. More clusters crowd the nearest and
  second-nearest together and drive $d_1/d_2\to1$. Their four benchmarks have exactly
  $C=12,31,65,345$, and accuracy falls with $C$ too, so the pooled number mixes a real signal
  with an artefact. The per-benchmark values are the honest ones and they are considerably
  weaker.
- **It cannot separate confident-correct from confident-wrong**, by construction. The paper
  frames this as an ImageCLEF quirk; it is the defining consequence of dropping Silhouette's
  negative range.
- **The null is not zero and is never stated**, so absolute PAS values are uninterpretable and
  not comparable across benchmarks — the same point as the first bullet from the other side.
- **No comparison against the obvious label-free baseline**: the entropy or max-softmax
  confidence of a source-trained classifier on the target data. That is also unsupervised, also a
  decisiveness measure, and would isolate how much the centroid geometry adds over simply asking
  the source model how sure it is.
- **The §4.4 ablation is three numbers.** Cosine versus Euclidean versus mean-pairwise distance
  is reported as a table of correlations with no analysis of *when* each choice matters, and the
  justification for cosine (magnitude invariance, curse of dimensionality) is asserted.
- **Robustness is untested in the sense that matters.** Figure 5 varies the number of samples and
  finds the ranking stable, which is good — but it is one sampling procedure, not the repeated
  reseeding that [Claßen et al.](../2026-classen-te-robustness/index.html) show is where these
  rankings actually come apart.

## Takeaways

- **PAS is a nearest-centroid margin in cosine geometry, with source classes standing in for the
  missing target labels.** Everything good and everything limited about it follows from that one
  substitution.
- The clean rewriting to remember is $\mathrm{PAS}=1-\mathbb{E}[d_1/d_2]$ — the mean complement
  of Lowe's ratio test. It makes the "confidence, not correctness" reading immediate.
- **$\mathrm{PAS}-\mathrm{Oracle}$ quantifies misassignment**, which is a usable diagnostic in
  any setting where a few target labels *are* available for spot-checking, even though the score
  itself is designed for none.
- The genuine contribution is filling a real gap: the established measures all need target
  labels, and unsupervised domain adaptation has none. Asymmetry is the right call and the
  construction is about as simple as it could be. The reservations are about how the evidence is
  aggregated, not about the idea.
