---
title: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
category: "Class imbalance"
subcategory: "Methods"
short_title: "Cao 2019 — LDAM-DRW"
authors: "Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, Tengyu Ma"
venue: "NeurIPS 2019"
year: 2019
url: "https://arxiv.org/abs/1906.07413"
pdf_url: "https://arxiv.org/pdf/1906.07413"
tags: [class-imbalance, long-tail, margin, generalization-bound, ldam, drw]
status: skimmed
summary: "a margin-based bound gives class-j error ∝ 1/(γ_j√n_j); minimising the total under a fixed margin budget gives γ_j ∝ n_j^(−1/4), so rare classes get larger margins (checked: 3.16× for a 100× count ratio); deferred re-weighting then reweights only after a first stage"
---

## Links

- [arXiv:1906.07413](https://arxiv.org/abs/1906.07413). Overview: [class imbalance](../class-imbalance-overview/index.html).

## In one paragraph

For a class with $n_j$ training points and margin $\gamma_j$, a margin generalization bound for the
class-conditional error behaves like $\frac{1}{\gamma_j\sqrt{n_j}}$ (up to terms shared by all classes).
Choosing margins to minimise the *sum* of these bounds, subject to a budget $\sum_j\gamma_j=\beta$, is a
one-line Lagrange problem: set $\partial/\partial\gamma_j\,[c_j/\gamma_j]=-c_j/\gamma_j^2=-\lambda$ with
$c_j=n_j^{-1/2}$, so $\gamma_j\propto\sqrt{c_j}=n_j^{-1/4}$. The **LDAM loss** implements this by subtracting
a class-dependent offset $\Delta_j=C/n_j^{1/4}$ from the true-class logit inside the softmax.

**DRW** (deferred re-weighting): train with plain (or LDAM) loss first, so the features are learned on the
natural distribution, then switch on class-balanced weights near the end of the schedule.

## Checked

With counts $5000/500/50$ the optimum is $\gamma^*=(0.168,\,0.299,\,0.532)$ for a unit budget; no one of
200k random feasible allocations beats its cost ($0.4991$). The minority/majority margin ratio is
$(5000/50)^{1/4}=3.16$ (`checks.py`).

## Questions and doubts

- The bound is a heuristic in two steps: the $1/(\gamma\sqrt n)$ form drops constants and the shared
  complexity term, and the budget $\sum\gamma_j=\beta$ is imposed, not derived. The $-1/4$ exponent is only
  as good as that bound.
- A class-dependent margin on the logit is the same kind of object as a logit offset; at test time LDAM
  does *not* undo it, unlike [logit adjustment](../2021-menon-logit-adjustment/index.html), which
  has a consistency argument for the balanced error.
- DRW's switch epoch is a hyper-parameter; the benefit is empirical.
- Written from the abstract and the headline derivation, not a full reading.

## Takeaways

- Margin $\propto n_j^{-1/4}$ comes from minimising $\sum_j n_j^{-1/2}/\gamma_j$ under a fixed total margin.
- Learn the representation first, rebalance second (the same idea as
  [Kang 2020](../2020-kang-decoupling-long-tail/index.html)).
