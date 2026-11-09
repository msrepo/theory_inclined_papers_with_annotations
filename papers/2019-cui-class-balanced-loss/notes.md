---
title: "Class-Balanced Loss Based on Effective Number of Samples"
category: "Class imbalance"
subcategory: "Methods"
short_title: "Cui 2019 — Class-balanced loss"
authors: "Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie"
venue: "CVPR 2019"
year: 2019
url: "https://arxiv.org/abs/1901.05555"
pdf_url: "https://arxiv.org/pdf/1901.05555"
tags: [class-imbalance, long-tail, reweighting, effective-number]
status: skimmed
summary: "weight each class by 1/E_n with E_n=(1−β^n)/(1−β), the 'effective' number of samples when new points overlap old ones; β→0 is no reweighting, β→1 is inverse frequency; checking it shows the weights barely move for β≤0.9 and saturate for β≥0.9999, so β is the whole method"
---

## Links

- [arXiv:1901.05555](https://arxiv.org/abs/1901.05555). Overview: [class imbalance](../class-imbalance-overview/index.html).

## In one paragraph

Inverse-frequency weights $1/n_y$ over-correct because the 1000th photo of a cat adds less than the
first. The paper models each sample as covering a small region of the space; a new sample is "new" with
probability $\beta^{\,n-1}$ (the chance it lands outside the already-covered part). The **effective number**
is then $E_n=\sum_{j=1}^{n}\beta^{j-1}=\frac{1-\beta^n}{1-\beta}$, and the class-balanced loss weights
class $y$ by $1/E_{n_y}$, normalised to sum to the number of classes.

## The one derivation

$E_n=E_{n-1}+\beta^{n-1}$, i.e. $E_n=1+\beta E_{n-1}$ (checked in
[`checks.py`](https://github.com/msrepo/theory_inclined_papers_with_annotations/tree/main/topics/class-imbalance-overview/code)).
Limits: $\beta\to0$ gives $E_n=1$ for every class (no reweighting); $\beta\to1$ gives $E_n\to n$
(inverse frequency). With counts $5000/500/50$ the normalised weights are $(0.998,\,0.998,\,1.003)$ at
$\beta=0.9$, $(0.66,\,0.67,\,1.67)$ at $0.99$, $(0.13,\,0.32,\,2.56)$ at $0.999$ and $(0.03,\,0.28,\,2.69)$ at
$0.9999$. So $\beta$ interpolates between "off" and "inverse frequency", and a tail class is capped at
roughly $1/E_\infty=1-\beta$ times a head class.

## Questions and doubts

- The "random cover" model is a heuristic, not a theorem: nothing in it connects $E_n$ to test error. It
  is a one-parameter family justified by experiments.
- $\beta$ is tuned on the same long-tailed validation distribution that is being evaluated, and the best
  value is close to $1$ (e.g. $0.999$–$0.9999$) where the weights are almost inverse frequency anyway.
- Reweighting changes the loss but not the model's logit offset in a principled way; compare
  [Menon 2021](../2021-menon-logit-adjustment/index.html), where the correction is derived from the
  balanced error.
- These notes were written from the abstract and the headline formula, not a line-by-line reading.

## Takeaways

- Effective number $=\frac{1-\beta^n}{1-\beta}$; weight $=1/E_n$.
- Everything lives in $\beta$: below $0.9$ nothing happens at these counts; near $1$ it is inverse frequency.
