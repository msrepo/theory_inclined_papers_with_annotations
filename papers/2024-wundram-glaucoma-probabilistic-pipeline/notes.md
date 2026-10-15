---
title: "Leveraging Probabilistic Segmentation Models for Improved Glaucoma Diagnosis: A Clinical Pipeline Approach"
category: "Misc"
subcategory: "Uncertainty Quantification"
short_title: "Wundram 2024 — glaucoma pipeline"
authors: "Anna M. Wundram, Paul Fischer, Stephan Wunderlich, Hanna Faber, Lisa M. Koch, Philipp Berens, Christian F. Baumgartner"
venue: "MIDL (PMLR 250)"
year: 2024
url: "https://proceedings.mlr.press/v250/wundram24a.html"
pdf_url: "https://raw.githubusercontent.com/mlresearch/v250/main/assets/wundram24a/wundram24a.pdf"
tags: [uncertainty, uncertainty-propagation, segmentation, glaucoma, multiple-annotators, marginalisation, medical-imaging]
status: read
---

## Links

- **[PMLR 250:1725–1740](https://proceedings.mlr.press/v250/wundram24a.html)**: the paper (MIDL 2024).
- **Code**: [annawundram/glaucoma-diagnosis-pipeline](https://github.com/annawundram/glaucoma-diagnosis-pipeline).
- **Segmenters compared**: Probabilistic U-Net (Kohl et al. 2018), [PHiSeg](../2019-baumgartner-phiseg/index.html),
  MC Dropout, and ensembles.
- **Related in this repo**: [Zhang et al. 2023](../2023-zhang-multiple-annotators/index.html) on learning
  from several annotators.

## In one paragraph

The cup and disc of the optic nerve head are blurry, and experts disagree on where the cup ends. A
single segmentation hides that. The paper samples many plausible cup and disc masks, turns each into a
rim thickness curve, classifies each curve, and averages the votes. The averaged vote is both the
diagnosis and a measure of uncertainty. It is a small applied paper with one formula, a marginalisation.

## The pipeline

1. **Region of interest.** A U-Net on the full fundus photo finds the disc; the image is cropped to
   320×320 around it.
2. **Probabilistic segmentation.** Draw samples $s_i\sim p(s\mid x)$ of the cup and disc masks.
3. **Rim thickness curve (RTC), $r=g(s)$.** Rotate a ray from the cup centre in 0.5° steps; at each angle
   record the distance between the cup border and the disc border. That is a 720-number vector, a
   clinically motivated alternative to the cup-to-disc ratio (a single number).
4. **Classifier, $d=f(r)$.** Average the RTC into 72 bins and fit logistic regression to predict
   "glaucoma suspect". The threshold is chosen by the Youden index (sensitivity + specificity − 1) on
   the validation set.
5. **Marginalise.**

$$p(d\mid x)=\int p(s\mid x)\,f\big(g(s)\big)\,ds\;\approx\;\frac1N\sum_{i=1}^{N}f\big(g(s_i)\big)$$

In words: run every sampled mask through the same deterministic steps and take the fraction of samples
that vote "suspect". Thresholding at 0.5 is a majority vote. If the cup is ambiguous the masks disagree,
the votes split, and $p(d\mid x)$ lands near 0.5. That is the propagated uncertainty.

## How the segmenters are trained

Two datasets have several expert masks per image: Cháksu (1345 images, 5 experts, plus a consensus mask
from STAPLE and each expert's diagnosis) and RIGA (750 images, 6 experts, no diagnosis). That makes 11
annotators. In every training batch each image is paired with one randomly chosen expert's mask, so the
network sees all the experts' versions across epochs.

| Method | How it produces samples | Model selection |
|---|---|---|
| Probabilistic U-Net, PHiSeg | draw the latent code from the prior and decode | generalised energy distance |
| MC Dropout | keep dropout (0.2) on at test time | generalised energy distance |
| Ensemble, seeds | 10 U-Nets with different seeds | Dice |
| Ensemble, experts | one U-Net per annotator (11) | Dice |

The paper groups the first two methods under aleatoric uncertainty (the ambiguity in the data) and the
other two under epistemic uncertainty (what the model does not know). Since the seed ensemble and MC
Dropout are also trained with a random expert per batch, they probably absorb some annotator variation
as well; the paper does not separate the two.

The classifier is trained on Cháksu only, with 100 RTC samples per training image (10 or 11 for the
ensembles).

## Results (their Table 1)

| Segmenter | AUROC with RTC | AUROC with CDR | Dice |
|---|---|---|---|
| U-Net (deterministic) | 0.857 | 0.888 | 0.919 |
| Ensemble, experts | 0.863 | 0.890 | 0.907 |
| Ensemble, seeds | 0.899 | 0.886 | 0.926 |
| Probabilistic U-Net | 0.876 | 0.869 | 0.871 |
| PHiSeg | 0.884 | 0.882 | 0.900 |
| MC Dropout | 0.885 | 0.887 | 0.894 |
| Expert annotations | 0.957 | 0.930 | — |
| ResNet50 (black box) | 0.884 | — | — |

The paper's claims are that probabilistic segmentation helps downstream (the deterministic U-Net has the
best-in-class Dice yet lower AUROC), that RTC beats CDR, that the pipeline matches a black-box ResNet50,
and that the vote fraction correlates with expert disagreement (0.59 to 0.65, best for PHiSeg).

## Questions and doubts

- **"Consistently improves" is too strong.** With RTC, the experts ensemble is only 0.006 above the
  U-Net and three probabilistic methods land at 0.876 to 0.885. With CDR, the plain U-Net (0.888) is as
  good as or better than every probabilistic method except the experts ensemble (0.890). The clear win
  is the seed ensemble with RTC.
- **No confidence intervals or repeated runs.** The test set is small, so gaps of 0.01 to 0.02 AUROC may
  be noise, and the ranking of methods is not established.
- **RTC does not always beat CDR.** For the U-Net it is worse (0.857 vs 0.888) and for MC Dropout and
  PHiSeg it is a tie. The advantage depends on the segmenter.
- **The "probability" is a vote fraction, not a calibrated probability.** The paper says calibration is
  future work. The seed ensemble has 10 voters, so its output takes only 11 values.
- **Possible train/test mismatch.** The classifier is trained on samples from segmenters that were
  trained on the same images, so the spread seen at training time may be tighter than at test time.
  This is not addressed.
- **The ceiling is far away.** Classifying expert masks gives 0.957, so the segmentation stage still
  costs about 0.06 AUROC.
- **Best on paper is not best at uncertainty.** PHiSeg tracks expert disagreement best but does not win
  on AUROC, so modelling the annotation distribution well and diagnosing well are different things.

## Takeaways

- Propagating samples (not just a variance map) through a pipeline lets any downstream model, not only a
  network, consume the uncertainty.
- Averaging decisions over plausible upstream outputs is a cheap, simple way to be robust to an
  ambiguous first stage, but the evidence here is modest.
- A good Dice score does not guarantee a good diagnosis.
