---
title: "Exploring Deep Neural Networks via Layer-Peeled Model: Minority Collapse in Imbalanced Training"
category: "Class imbalance"
subcategory: "Theory"
short_title: "Fang 2021 — Minority collapse"
authors: "Cong Fang, Hangfeng He, Qi Long, Weijie J. Su"
venue: "PNAS 2021"
year: 2021
url: "https://arxiv.org/abs/2101.12699"
pdf_url: "https://arxiv.org/pdf/2101.12699"
tags: [class-imbalance, neural-collapse, layer-peeled-model, etf, minority-collapse]
status: skimmed
summary: "treat the last-layer features and classifier as free variables under norm constraints (Layer-Peeled Model); balanced data gives the simplex ETF of neural collapse, and as the imbalance ratio grows the minority classifiers are pushed onto each other (Minority Collapse); not re-derived here"
---

## Links

- [arXiv:2101.12699](https://arxiv.org/abs/2101.12699). Neural collapse itself:
  Papyan, Han, Donoho (2020), downloaded in `class_imbalance/`. Overview: [class imbalance](../class-imbalance-overview/index.html).

## In one paragraph

The **Layer-Peeled Model** keeps only the last two stages of the network: the features $h_{k,i}$ of every
training point and the classifier rows $w_k$, both free, with norm constraints $\lVert h\rVert^2\le E_H$,
$\lVert w_k\rVert^2\le E_W$, and minimises the cross-entropy. With *equal* class sizes the minimisers are the
**simplex equiangular tight frame (ETF)**: $K$ class means at equal angles $\cos\theta=-\tfrac1{K-1}$, which
is the neural collapse seen in practice. With *unequal* class sizes the paper finds that, as the ratio
$R=n_{\max}/n_{\min}$ of majority to minority class size grows, the minority classifiers $w_k$ become
closer and closer in angle and in the limit coincide: the model can no longer tell the minority classes
apart (**Minority Collapse**), which is a statement about the geometry of the optimum rather than about
under-training.

## Questions and doubts

- The model removes everything below the last layer, so it explains the geometry at the optimum of
  a network that can fit anything, not the path gradient descent takes or finite-width effects.
- The norm constraints are what produce the ETF; with weight decay in place of a hard constraint the
  details change.
- If this is the whole story, oversampling the minority and logit adjustment should help for different
  reasons ([Menon 2021](../2021-menon-logit-adjustment/index.html) corrects the intercept, this
  paper moves the minority classifiers apart); I have not checked which of the two the paper's experiments show.
- Not re-derived: the collapse condition and the value of $R$ at which it sets in are taken from the
  abstract, not verified. This note needs a proper reading before it is cited.

## Takeaways

- Balanced: ETF. Strongly imbalanced: minority classifiers merge in angle.
- A geometric reason why the last-layer fix of [Kang 2020](../2020-kang-decoupling-long-tail/index.html)
  may not be enough at extreme ratios.
