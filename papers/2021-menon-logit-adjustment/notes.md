---
title: "Long-tail learning via logit adjustment"
category: "Class imbalance"
subcategory: "Theory"
short_title: "Menon 2021 — Logit adjustment"
authors: "Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar"
venue: "ICLR 2021"
year: 2021
url: "https://arxiv.org/abs/2007.07314"
pdf_url: "https://arxiv.org/pdf/2007.07314"
tags: [class-imbalance, long-tail, logit-adjustment, balanced-error, fisher-consistency]
status: skimmed
summary: "the balanced-error Bayes rule is argmax_y p(y|x)/π_y, i.e. subtract log π_y from the logit; do it after training or put +τ log π_y inside the softmax during training; checked on two Gaussians: plain Bayes gets balanced accuracy 0.656 and minority recall 0.32, the adjusted rule 0.841 and 0.84"
---

## Links

- [arXiv:2007.07314](https://arxiv.org/abs/2007.07314). Overview: [class imbalance](../class-imbalance-overview/index.html).

## In one paragraph

Plain softmax cross-entropy learns $p(y\mid x)\propto\pi_y\,p(x\mid y)$, which bakes the class prior $\pi_y$
into the logit. The metric on a long-tailed benchmark is the **balanced error**, the average of per-class
errors, whose optimal rule uses the prior-free quantity $p(x\mid y)\propto p(y\mid x)/\pi_y$:

$$\hat y(x)=\arg\max_y\ \big[f_y(x)-\tau\log\pi_y\big]\qquad(\tau=1\ \text{is the Bayes rule}).$$

*Post-hoc*: apply this to a trained model's logits. *In the loss*: train with
$\ell(y,f)=-\log\frac{e^{f_y+\tau\log\pi_y}}{\sum_{y'}e^{f_{y'}+\tau\log\pi_{y'}}}$, so the network is pushed to a
larger raw margin on rare classes, and at test time use $f$ directly. The two are consistent for the
balanced error, which the paper proves; LDAM and similar margin losses are special cases.

## Checked

Two unit-variance Gaussians at $\pm1$ with minority prior $0.05$: the plain Bayes threshold is $1.47$
(balanced accuracy $0.656$, minority recall $0.318$); the adjusted threshold is $0$ (balanced accuracy
$0.841$, recall $0.841$), and brute force over thresholds agrees. See `checks.py`.

## Questions and doubts

- Consistency is an infinite-data statement. In the overparameterised regime the training logits are
  nearly separable and the learned $f$ is not the posterior;
  [Lyu 2026](../2026-lyu-imbalanced-overfitting/index.html) shows the minority intercept is then pushed the
  wrong way and the right $\tau$ can be negative.
- The balanced error is one target; if a deployment cares about minority recall at a fixed false-positive
  rate, $\tau=1$ is not obviously right.
- The loss form changes training, the post-hoc form does not; the paper finds the loss form better but the
  gap is empirical.
- Written from the abstract and the standard derivation, not a full reading.

## Takeaways

- Balanced-error optimal rule: subtract $\log\pi_y$ from the logit.
- Post-hoc is free; the training-time version trades a hyper-parameter $\tau$ for better features.
