---
title: "SACP: Spatially-Adaptive Conformal Prediction in Uncertainty Quantification of Medical Image Segmentation"
category: "Misc"
subcategory: "Uncertainty Quantification"
short_title: "Bereska 2025 — SACP"
authors: "Jacqueline Isabel Bereska, Hamed Karimi, Reza Samavi"
venue: "MIDL (PMLR 301)"
year: 2025
url: "https://proceedings.mlr.press/v301/bereska26a.html"
pdf_url: "https://raw.githubusercontent.com/mlresearch/v301/main/assets/bereska26a/bereska26a.pdf"
tags: [conformal-prediction, uncertainty, segmentation, coverage-guarantee, class-conditional, pancreatic-cancer, exchangeability]
status: read
---

## Links

- **[PMLR 301:127–152](https://proceedings.mlr.press/v301/bereska26a.html)**: the paper (MIDL 2025, full paper).
- **Code**: [tailabTMU/SACP](https://github.com/tailabTMU/SACP).
- **Conformal prediction primer**: Angelopoulos & Bates, *A gentle introduction to conformal prediction* (2023).
- **Class-conditional CP**: Shi, Ong & Leckie 2013; Sadinle, Lei & Wasserman 2019 (least ambiguous set-valued classifiers).
- **Data**: PANORAMA challenge CTs (pancreatic cancer) and the Medical Segmentation Decathlon pancreas set.
- **Related in this repo**: [PHiSeg](../2019-baumgartner-phiseg/notes.md) and [Probabilistic U-Net](../2018-kohl-probabilistic-unet/notes.md) for a different way of getting uncertainty in segmentation.

Bereska, Karimi and Samavi, *SACP*, MIDL 2025 (proceedings volume dated 2026).

## In one paragraph

Conformal prediction (CP) turns any classifier into one that outputs a *set* of labels per item, with a promise that
the true label is inside the set at least $1-\alpha$ of the time. Applied voxel by voxel to a segmentation, the
promise is the same everywhere in the image. A surgeon deciding whether a pancreatic tumour can be cut out cares
much more about the voxels where the tumour touches a major blood vessel than about the middle of the tumour.
SACP multiplies the tumour's non-conformity score by a weight below 1 that is smaller near vessels, so the tumour
label enters the prediction set more readily there, and the paper argues the original coverage guarantee still holds.

## The picture, intuition first

**Conformal prediction as a "how weird is this?" test.** Fit a model. On a held-out *calibration* set, record for
each item a non-conformity score $S$: how badly the model's answer fits the true label (a common choice is
$S=1-p$, one minus the probability the model gave the true class). Sort the scores and take the
threshold $\tau$ below which $1-\alpha$ of them fall. For a new item, put into its prediction set every label
whose score is below $\tau$. If the new item behaves like the calibration items, the true label's score is below $\tau$
about $1-\alpha$ of the time. That is the whole guarantee; it needs no assumption about the data's distribution.

**What "behaves like" means: exchangeability.** The calibration items and the test item must be
interchangeable: if you shuffled them, the joint probability would not change (independent draws from one
distribution satisfy this). This is what lets the new item's rank among the calibration scores be uniformly random,
and that is what makes the $1-\alpha$ promise exact.

**Class-conditional version.** Tumour voxels are a tiny fraction of the scan, so a single threshold is dominated by
background voxels. Class-conditional CP computes one threshold $\tau_{\hat y}$ per class from the calibration voxels
that truly belong to that class, so each class gets its own $1-\alpha$ promise (their Eq. 1 and 2).

**The SACP twist.** For the tumour label $l$ only, replace the score by
$S_{\text{SACP}}=w\cdot S_{\text{base}}$ with weight $w\in[0.5,1)$ (Eq. 3). Since $w<1$ the score shrinks, so it is more
often below the threshold $\tau_l$ and the tumour label is included in more voxels. The weight (Eq. 4) is
$w=\sigma\!\big(\tfrac1\gamma(\phi+\delta\,I)\big)$ where

- $\delta$: distance from the voxel to the nearest vessel; $\phi$: distance to the predicted tumour;
- $I=-\log p(\text{tumour}\mid x)$: the "surprisal", large when the model gives the tumour little probability;
- $\gamma\in(0,1]$: a hand-set importance of that vessel (0.8 for arteries, 0.6 for veins);
- $\sigma$: the sigmoid squashing function, steepened with gain $\beta=10$, mapping any non-negative input into
  $[0.5,1)$. Input near 0 (close to vessel and tumour) gives $w\approx0.5$; large input gives $w\approx1$ (no change).

Close to a vessel and tumour, $w$ is small, the tumour label is included generously, and the set grows.

## Notation in plain words

- **Coverage**: the share of voxels whose prediction set contains the true label. Higher than $1-\alpha$ means
  more cautious, lower means broken.
- **Quantile**: the value with a given fraction of the sorted scores below it.
- **Indicator** $\mathbf 1\{y\in C(x)\}$: 1 if the truth is in the set, 0 otherwise; averaging it gives coverage.
- **RWR** (relative width ratio, Eq. 6): average set size at voxels farther than $r$ from a vessel divided by
  the average set size at voxels within $r$. (See the doubts below about its direction.)
- **SSCV**: split voxels into bins by set size, compute coverage in each bin, report the worst gap from $1-\alpha$.

## The theory claimed

Theorem 3 and its proof (Appendix C) say $C_{\text{base}}(x)\subseteq C_{\text{SACP}}(x)$: since $0.5\le w<1$,
$S_{\text{SACP}}<S_{\text{base}}$, so any label passing the base test passes the SACP test. Hence SACP sets are
never smaller, and coverage is at least that of class-conditional CP.

## Results (30 CTs from PANORAMA; 10 calibration, 20 test; $\alpha=0.05$)

| Quantity | CCCP (class-conditional CP) | SACP |
|---|---|---|
| Overall coverage | 0.968 | 0.981 (voxel-pooled 0.987) |
| Coverage within 2 mm of a vessel | 0.954 | 0.981 |
| RWR within 2 mm / beyond 20 mm | 2.887 / 2.509 | 2.762 / 2.525 |

Per-vessel coverage within 2 mm for CCCP is 0.925 (SMA) and 0.927 (PV), below the 0.95 target, and SACP brings these
to 0.975 and 0.953. A second dataset (MSK) gives similar overall coverage (0.980 against 0.979).

## Questions and doubts

- **The guarantee is inherited, not earned.** Because $C_{\text{SACP}}\supseteq C_{\text{base}}$, coverage can only
  go up, so "SACP keeps the CP guarantee" is immediate and Theorem 3 is a one-line fact. What the guarantee does *not*
  give is anything about coverage *near vessels specifically*. The introduction motivates the work with
  "error rate 5% near vessels and 10% elsewhere", but nothing in the method lets one set or certify a local $\alpha$;
  the weight is a heuristic that inflates the set, and near-vessel coverage is only observed in the experiment.
- **Higher coverage is not "improved" coverage.** The headline comparison is SACP 0.981 against CCCP 0.968 with a
  paired $t$-test. As sets are supersets by construction, coverage rising is guaranteed; the test only measures by how
  much. Both are well above the 0.95 target (the paper tests that 0.987 "significantly exceeds" 0.95), so the
  real cost is larger sets, and that cost is not reported in absolute terms.
- **Over-coverage at all distances points at the base method.** Appendix D says over-coverage should be at most
  $1/(n+1)$, which is tiny with millions of calibration voxels. Observing 0.968 to 0.987 against 0.95 contradicts it. Three
  suspects: voxels inside a scan are strongly correlated, so they are far from exchangeable (the effective
  sample is about 10 scans, not millions of voxels); the softmax probabilities saturate near 0 and 1, giving many
  tied scores, which the $1/(n+1)$ bound assumes away; and coverage is pooled over all 11 classes, dominated by background.
  The paper does not say which it is, and it matters for what "valid" means here.
- **Where is the finite-sample correction?** Standard CP uses the $\lceil(n+1)(1-\alpha)\rceil/n$ quantile; Eq. 1
  uses a plain $(1-\alpha)$-quantile. With very many voxels this barely matters, but it is the very thing the guarantee
  rests on.
- **The weight points the "wrong" way for low confidence.** $I=-\log p$ is large when the model is *unsure* the
  voxel is tumour, which pushes $w$ towards 1 and *removes* the inflation. So the set grows most where the model is
  confident, near a vessel, and not where it hesitates. The text says this is intended ("more conservative near
  high-confidence tumour regions") but it is odd for an uncertainty method; part of the reason may be that $S_{\text{base}}$
  is already large where confidence is low, though the main text never defines $S_{\text{base}}$ (Table 1 mentions the "least
  ambiguous" score).
- **Units and scale of $w$ are unspecified.** $\phi$ and $\delta$ are distances (mm or voxels?) added to a
  surprisal in nats, with distances clipped at the 95th percentile, then fed through a sigmoid with gain 10. How
  quickly $w$ rises from 0.5 to 1 as a function of millimetres depends entirely on these choices, and $\gamma$ is
  "fine-tuned depending on the application", so the amount of adaptivity is a tuning knob. The abstract
  speaks of weights that "exponentially decay with distance", while the method uses a sigmoid.
- **The RWR direction looks inconsistent.** By Eq. 6, $\rho=\mu(C\mid\delta>r)/\mu(C\mid\delta\le r)$ is the far
  size over the near size, and Table 1 reports 2.5 to 2.9, meaning far-away voxels have *larger* sets than
  near-vessel voxels, opposite to the claim that sets are "wider near vessels". A likely reason is that sets
  count all 11 classes and far voxels sit among several organs, but then RWR measures organ mix as much as vessel
  adaptation. Appendix E describes "increasing width with vessel proximity" for numbers that rise with distance
  (1.141 to 1.655). Also, set size over all classes mostly reflects how many labels pass, not the tumour label SACP edits.
- **Small, single-centre-mix evaluation.** 20 test scans, 10 calibration scans, and "the number of distinct
  tumour–vessel interfaces is constrained", as the authors admit. How calibration and test cases were split across
  the five centres is not stated, and CP's guarantee needs them exchangeable. Near-vessel numbers for a single
  vessel at 2 mm rest on very few tumours.
- **Segmentation quality is mixed in.** Vessel distances come from another network whose errors near a tumour are
  likely larger (the paper itself shows lower Dice near vessels). Errors in the distance map change $w$ but not the
  validity of the base guarantee, since $w$ depends only on $x$; the reliability of the clinical *interpretation* is a
  different matter and untested.

## Takeaways

- CP's guarantee is about a threshold on the scores; SACP only makes the threshold for one class vary with position,
  which can only enlarge sets, so validity is automatic and the benefit is a heuristic one.
- The clinically interesting part is that plain class-conditional CP under-covers near some vessels
  (SMA, PV within 2 mm); inflating the tumour label there repairs it in these 20 scans.
- A genuine locally-valid guarantee would need local calibration (for example, thresholds per region), not
  a weight on top of a global threshold.
