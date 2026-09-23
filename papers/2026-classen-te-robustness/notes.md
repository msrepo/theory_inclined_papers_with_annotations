---
title: "Robustness of transferability estimation metrics for medical imaging"
category: "Transferability"
subcategory: "Applications"
short_title: "Claßen 2026 — TE robustness"
authors: "Niclas Claßen, Théo Sourget, Dovile Juodelyte, Rob van der Goot, Veronika Cheplygina (IT University of Copenhagen)"
venue: "arXiv"
year: 2026
url: "https://arxiv.org/abs/2608.09999"
pdf_url: "https://arxiv.org/pdf/2608.09999"
tags: [transferability, medical-imaging, benchmark, robustness, rank-correlation, medmnist, reproducibility]
status: read
---

## Links

- **[arXiv:2608.09999](https://arxiv.org/abs/2608.09999)** — preprint. Code, checkpoints and
  data splits are released.
- The three measures it benchmarks:
  [Tran 2019 — NCE](../2019-tran-nce-hardness/index.html),
  [Nguyen 2020 — LEEP](../2020-nguyen-leep/index.html),
  [Bao 2022 — H-score](../2022-bao-hscore-transferability/index.html).
- **[Chaves 2023 — medical TE](../2023-chaves-medical-transferability/index.html)** —
  the study this one follows up, and whose open question about the evaluation metric it
  takes up directly.
- **[Runnable code](https://github.com/msrepo/theory_inclined_papers_with_annotations/blob/main/papers/2026-classen-te-robustness/code/robustness.py)** —
  not a reproduction, which is not possible here, but the mechanisms: where the rank
  correlations disagree, how stability falls with subset size, and the sharp $n\approx k$
  boundary that locates their H-score result. `make verify` runs it.

## In one paragraph

Transferability estimation promised to replace "fine-tune every candidate source and see" with a
cheap score. This paper asks whether the scores are stable enough to use, on medical imaging
targets where data scarcity is not hypothetical. It holds everything fixed except the *target
subset* — same size, same class distribution, different random seed — and finds the rankings
move. It then shows the **reference** ranking moves too, depending on whether you fine-tune
optimising accuracy or AUROC. And it reports that agreement between the TE metrics and either
reference is low across all eight targets, including at 100% of the data. The contribution is
negative and methodological: the existing comparisons in this literature, which mostly use a
single subset from a single seed, do not support the conclusions drawn from them.

## What was actually run

| | |
|---|---|
| targets | 8 of the 2D MedMNIST v2 datasets, 224×224 |
| sources | the other MedMNIST 2D sets (12 total) plus ImageNet |
| architecture | ResNet-18 only |
| TE metrics | H-score, LEEP, $\mathcal{N}$LEEP, LogME, NCTI, PARC, SFDA |
| subsets | 5%, 10%, 25%, 50%, 75% of target train, stratified and **nested**, ×5 seeds |
| reference | full fine-tuning, hyperparameter-tuned separately for accuracy and for AUROC |
| correlations | Spearman $\rho$, Kendall $\tau$, weighted Kendall $\tau_w$ — all three |

Two design choices are better than the norm in this literature and worth singling out. The
subsets are **nested**, so a larger subset contains the smaller ones and size is isolated from
composition. And the reference is computed **twice**, under two evaluation criteria, which is
what makes Ex2 possible at all.

## The three findings

**1. Rankings move under nothing but the seed.** Their intra-metric stability (Eq. 1) is the
mean pairwise rank correlation between rankings from equal-sized subsets drawn with different
seeds. It falls as subsets shrink, badly on the small targets — *Breast* has 546 training
images, so its 5% fraction is about 27. Figure 5 is the memorable one: on *Breast* at 5%,
LogME ranks *Blood* as both the best and the worst source depending on the seed.

**2. The reference ranking is not a fixed thing either.** Fine-tuning the same pool while
optimising for accuracy versus AUROC produces different orderings — "almost none of the source
models are ranked the same", and one source moves between 2nd and 11th. So the "ground truth"
that every TE paper reports correlation against is itself a random variable with a choice baked
in.

**3. Agreement with the reference is low regardless.** Including at 100% of the target data,
where the TE metric and the fine-tuning run see identical data. This replicates Chaves et al.
in the medical domain and adds that it is not an artefact of the evaluation criterion, since
both accuracy and AUROC references give the same verdict.

Plus two practical notes worth having: $\mathcal{N}$LEEP and SFDA produce **NaNs** on these
targets — SFDA because it assigns every source the same score — so their applicability is
limited by the target dataset itself. And the three rank correlations differ by as much as
**0.227** on individual cases, which is why reporting one is a choice.

## Reproduced — the mechanisms, not the experiment

Eight MedMNIST targets and a pool of fine-tuned ResNet-18s are not reproducible here. The
mechanisms are, and one of them turns out to sharpen the paper's own result.

### The correlation coefficients genuinely disagree

Their Figure 2 asserts this; it is easy to construct. Take a reference ranking of 10 sources,
and two candidate metrics: **A** gets the top half right and scrambles the tail, **B** gets the
tail right and scrambles the top.

| | $\tau$ | weighted $\tau$ | $\rho$ |
|---|---|---|---|
| metric A | 0.556 | 0.307 | 0.758 |
| metric B | 0.556 | **0.804** | 0.758 |

Plain $\tau$ and $\rho$ call it a tie; top-weighted $\tau$ prefers B decisively. Both verdicts
are defensible — $\tau_w$ is the usual choice because you deploy the top-ranked source, but it
presumes a clear winner exists, which the paper explicitly notes is not guaranteed. Reporting a
single coefficient silently picks a verdict.

### H-score's instability is a transition at $n\approx k$, not a decay

This is where the checking pays for itself. The paper reports H-score as "the least stable
across all target representations" on small targets. That is true but it undersells what is
happening. Sweeping the subset size $n$ against the feature dimension $k$, with five seeds and
nothing else changing:

| $k$ | $n$ | $n/k$ | LEEP | H-score | $\operatorname{cond}(S_T)$ |
|---|---|---|---|---|---|
| 64 | 32 | 0.5 | 0.618 | 0.178 | 1.5e18 |
| 64 | 64 | 1.0 | 0.782 | −0.024 | 7.8e16 |
| 64 | 128 | 2.0 | 0.891 | **0.988** | 3.0e1 |
| 64 | 2000 | 31 | 1.000 | **1.000** | 2.2e0 |
| 256 | 128 | 0.5 | 0.836 | −0.092 | 1.6e18 |
| 256 | 256 | 1.0 | 0.879 | 0.012 | 1.0e17 |
| 256 | 512 | 2.0 | 0.897 | **1.000** | 3.6e1 |

LEEP degrades smoothly, which is ordinary estimator noise. H-score does something categorically
different: **above $n/k\approx2$ it is the *more* stable of the two, and below it the ranking
becomes noise and then anti-correlates across seeds.** Negative intra-stability means two seeds
on the same data disagree worse than chance. The boundary is tracked exactly by
$\operatorname{cond}(S_T)$ jumping from $\sim10^1$ to $\sim10^{17}$.

That is the ill-conditioning already documented in
[the H-score notes](../2022-bao-hscore-transferability/index.html), where forming
$S_T=Z^\top Z$ squares the condition number. And it **locates** their finding rather than merely
agreeing with it: ResNet-18 gives $k=512$, and *Breast* at a 5% fraction is about 27 images, so
$n/k\approx0.05$ — an order of magnitude the wrong side of the boundary. Their result is not
that H-score is generally unstable. It is that medical targets are small enough, relative to
feature dimension, to sit deep in the regime where it is noise.

This also implies the cheap fix the paper does not try: computing H-score by a thin QR of the
centred features rather than by inverting their covariance keeps ten correct digits where the
textbook form loses three, because it never squares the conditioning. It will not manufacture
information that 27 images do not contain, but it would separate "the estimator is broken" from
"the data is insufficient", which the current experiment cannot distinguish.

### Stability is necessary, not sufficient

| ranking | intra-stability | $\tau$ vs truth |
|---|---|---|
| a fixed, input-free ranking | **1.000** | −0.030 |
| LEEP on the full target | — | 0.848 |
| H-score on the full target | — | 1.000 |

A metric that ignores its input entirely is perfectly stable and perfectly useless. Which is
exactly why Ex1 and Ex2 have to be separate experiments, and why the paper's Ex1 results should
not be read as a ranking of the metrics.

## Questions and doubts

- **The negative result is partly a resolution problem, and the paper cannot separate the two.**
  At 5% of *Breast* — 27 images, 2 classes — no estimator of any kind has much to work with. Low
  agreement there is evidence about the sample size at least as much as about the metric. The
  100% rows are the load-bearing ones for the "TE metrics do not work in medical imaging" claim,
  and they are one column of the experiment rather than its bulk.
- **Estimator conditioning is treated as a property of the metric.** H-score is scored as
  unstable when a large part of what is measured is an avoidable numerical choice. The paper
  cites Ibrahim et al., who proposed shrinkage for exactly this, and does not apply it — so the
  benchmark measures reference implementations rather than the methods.
- **One architecture.** ResNet-18 throughout, which the limitations section concedes. Since the
  $n/k$ boundary above depends on the feature dimension, architecture is not a free variable for
  at least one of the metrics benchmarked.
- **Five seeds** is enough to show instability exists and thin for quantifying it; the pairwise
  stability at 5 seeds is an average over 10 pairs.
- **Nested subsets are the right design for isolating size**, but they induce dependence between
  the fractions, so the per-fraction results are not independent observations.
- **"Low agreement" is never given a null.** What $\tau$ *would* two independent fine-tuning
  runs of the same source achieve against each other? Figure 7 gestures at this by showing the
  accuracy and AUROC references disagree, but it is not turned into the baseline it should be —
  and without it, "agreement is low" has no scale.
- **MedMNIST is 224×224 standardised**, which the authors flag as making the setting less
  realistic than clinical data. Worth keeping when reading the headline claim.

## Takeaways

- **The methodological point generalises well beyond medical imaging**: if a comparison of TE
  metrics rests on one subset from one seed, it does not support a conclusion about which metric
  is better. That is a statement about the whole literature, and it is the paper's real
  contribution.
- **The reference ranking is a random variable.** Every "our metric attains $\tau=x$" in this
  area is conditional on a fine-tuning run and an evaluation criterion, both of which move.
  Reporting the reference's own variability should be standard and is not.
- **H-score's failure here has a diagnosable cause and a sharp boundary** at $n\approx k$, which
  is more actionable than "least stable": it tells you when not to use it, and suggests that
  computing it differently would move the boundary rather than remove it.
- **Read this against the three method papers together.** Each was validated in a regime — NCE
  with shared inputs, LEEP with ImageNet-to-CIFAR semantic overlap, H-score with $n \gg k$ — and
  medical imaging violates a different assumption for each. The metrics are not failing at
  random; they are being used outside what their derivations covered.
