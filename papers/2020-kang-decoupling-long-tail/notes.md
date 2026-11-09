---
title: "Decoupling Representation and Classifier for Long-Tailed Recognition"
category: "Class imbalance"
subcategory: "Methods"
short_title: "Kang 2020 — Decoupling"
authors: "Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, Yannis Kalantidis"
venue: "ICLR 2020"
year: 2020
url: "https://arxiv.org/abs/1910.09217"
pdf_url: "https://arxiv.org/pdf/1910.09217"
tags: [class-imbalance, long-tail, representation-learning, classifier-rebalancing]
status: skimmed
summary: "train features with plain instance-balanced sampling, then fix only the classifier (retrain it class-balanced, or shrink weight norms by ‖w_j‖^τ); the finding is that imbalance hurts the classifier's decision boundary more than the representation, which beats specialised losses on ImageNet-LT, Places-LT and iNaturalist"
---

## Links

- [arXiv:1910.09217](https://arxiv.org/abs/1910.09217). Code: facebookresearch/classifier-balancing.
  Overview: [class imbalance](../class-imbalance-overview/index.html).

## In one paragraph

Two stages. **Stage 1:** learn the backbone with ordinary instance-balanced sampling. **Stage 2:** keep it
frozen and rebalance only the linear classifier, in one of three ways: *cRT* (re-train the classifier with
class-balanced sampling), *τ-normalisation* (replace the class weight $w_j$ by $w_j/\lVert w_j\rVert^{\tau}$,
$\tau\in[0,1]$) or learnable weight scaling. The observation behind it: head classes end up with larger
weight norms $\lVert w_j\rVert$, which inflates their logits; $\tau$-normalisation removes exactly that
size effect and has one parameter.

## Questions and doubts

- τ-normalisation works only if logit magnitude is the main bias; a bias term or a feature-norm effect would
  not be removed by it. [Menon 2021](../2021-menon-logit-adjustment/index.html) corrects by a $\log\pi_j$
  offset instead and explains why the right correction is additive in the logit.
- "Imbalance does not hurt representations" is shown on these benchmarks with instance-balanced sampling
  as the winner; it is not a statement about very extreme ratios, and [Fang 2021](../2021-fang-minority-collapse/index.html)
  predicts the features themselves degrade then.
- $\tau$ is chosen on validation data from the long-tailed distribution.
- Written from the abstract and method summary, not a full reading.

## Takeaways

- Imbalance shows up most in the classifier's weight norms; fixing only the last layer recovers most of the loss.
- A clean baseline to compare any new loss against.
