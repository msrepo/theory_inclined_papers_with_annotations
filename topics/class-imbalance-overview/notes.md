---
title: "Class imbalance: a map of the problem"
category: "Class imbalance"
subcategory: "Overview"
short_title: "Class imbalance overview"
tags: [class-imbalance, long-tail, balanced-error, logit-adjustment]
status: reading
summary: "what goes wrong with a skewed label distribution, three families of fixes (reweight/resample, margin and logit offsets, rebalance only the classifier), and a numeric check of the three formulas that recur: effective number, LDAM margins and the logit-adjusted Bayes rule"
---

## Links

Code: [`checks.py`](https://github.com/msrepo/theory_inclined_papers_with_annotations/tree/main/topics/class-imbalance-overview/code).

## The problem

With class priors $\pi_y$ very unequal, a classifier trained on the average loss is rewarded for the
head classes. Three separate things go wrong, and papers differ on which they target:

1. **The decision rule.** $p(y\mid x)\propto\pi_y p(x\mid y)$ bakes the prior into the logit, so the rule
   optimal for plain accuracy is not the one for balanced accuracy. (Two Gaussians, minority prior $0.05$:
   balanced accuracy $0.656$ vs $0.841$ once $\log\pi_y$ is subtracted.)
2. **The classifier weights.** Head classes get larger weight norms.
3. **The features.** With few examples the minority is memorised ([Lyu 2026](../2026-lyu-imbalanced-overfitting/index.html)),
   and at extreme ratios the minority class means collapse ([Fang 2021](../2021-fang-minority-collapse/index.html)).

## Reading order

| Paper | What it fixes | Core object |
|---|---|---|
| [Menon 2021](../2021-menon-logit-adjustment/index.html) | the rule | $f_y-\tau\log\pi_y$ |
| [Cao 2019](../2019-cao-ldam-drw/index.html) | the margin | $\gamma_j\propto n_j^{-1/4}$ |
| [Cui 2019](../2019-cui-class-balanced-loss/index.html) | the weights | $E_n=\frac{1-\beta^n}{1-\beta}$ |
| [Kang 2020](../2020-kang-decoupling-long-tail/index.html) | the classifier only | $w_j/\lVert w_j\rVert^\tau$ |
| [Fang 2021](../2021-fang-minority-collapse/index.html) | (explains) feature geometry | simplex ETF, Minority Collapse |
| [Lyu 2026](../2026-lyu-imbalanced-overfitting/index.html) | (explains) overfitting in a linear probe | truncated logits |

## Also downloaded for this section (not yet annotated)

Local PDFs in `Theory_oriented_papers/class_imbalance/`: Huang et al. universal features (2019) and ISIT 2017,
Hastie–Tibshirani–Buja FDA (1994), Wan et al. L-GM loss (2018), Andrew et al. Deep CCA (2013), Papyan et al.
neural collapse (2020), and three evaluation-methodology papers (Demšar 2006, Bouthillier 2021, Agarwal 2021).

## Questions and doubts

- Several fixes are the same offset in different clothes (a $\log\pi_y$ shift, a margin $\Delta_j$, a weight
  norm); I have not worked out when they are exactly equivalent.
- The five new paper notes are short and written from abstracts plus the headline formulas; only the
  formulas in `checks.py` were checked.

## Takeaways

- Identify which of rule, weights or features is broken before choosing a fix.
- Report balanced accuracy, and compare against post-hoc logit adjustment first: it costs nothing.
