---
title: "On the Pitfalls of Using the Residual Error as Anomaly Score"
category: "Medical image analysis"
subcategory: "Anomaly detection"
short_title: "Meissen 2022 — residual anomaly score"
authors: "Felix Meissen, Benedikt Wiestler, Georgios Kaissis, Daniel Rueckert"
venue: "MIDL (PMLR 172)"
year: 2022
url: "https://proceedings.mlr.press/v172/meissen22a.html"
pdf_url: "https://proceedings.mlr.press/v172/meissen22a/meissen22a.pdf"
tags: [anomaly-detection, autoencoder, reconstruction-error, brain-mri, average-precision, unsupervised-segmentation]
status: read
summary: "a lesion is found only if its residual beats the reconstruction error at healthy pixels, so the score is blind to lesions near common tissue intensity; a sharper autoencoder also reconstructs the lesion; the one inequality it rests on is loose and the evidence is simulated discs"
ext_links: [code|https://github.com/FeliMe/residual-score-pitfalls]
---

## Links

- **[PMLR 172:914–928](https://proceedings.mlr.press/v172/meissen22a.html)**: the paper (MIDL 2022, full paper).
- **Code and experiments**: [FeliMe/residual-score-pitfalls](https://github.com/FeliMe/residual-score-pitfalls).
- **Precursor**: Meissen, Kaissis & Rueckert 2021, *Challenging current semi-supervised anomaly segmentation methods for brain MRI* ([arXiv:2109.06023](https://arxiv.org/abs/2109.06023)), the "white object detection" result.
- **Data**: MOOD challenge 2020 (800 healthy T2 brain MRIs).
- **Comparative study of the methods being criticised**: Baur et al. 2021, *Autoencoders for unsupervised anomaly segmentation in brain MR images*.

Meissen, Wiestler, Kaissis and Rueckert, *On the Pitfalls of Using the Residual Error as Anomaly Score*, MIDL 2022.

## In one paragraph

A popular recipe for finding lesions without lesion labels: train an autoencoder on healthy scans only, feed it a
scan that may contain a lesion, and flag the pixels where the output differs most from the input. The hope is that
the network "cannot imagine" the lesion, repaints healthy tissue instead, and the difference (the **residual**)
lights up exactly the lesion. The paper argues this hope ignores a second source of residual: the network is never a
perfect copy machine, so *healthy* pixels also have non-zero residual, and those can be bigger than the residual
of a lesion whose brightness is close to normal tissue. The paper shows this with simulated lesions on healthy
brain MRI.

## The picture, intuition first

Think of the residual map as a noisy alarm. The alarm should ring loudly at the lesion and quietly everywhere
else. There are two things that make it ring:

1. **Real signal.** The lesion pixel has a value different from what healthy tissue would have there. The ring
   strength is that difference.
2. **Reconstruction noise.** The network smooths edges, loses fine texture, and mis-draws small structures. This
   rings the alarm at healthy edges even with no lesion at all.

A lesion is found only if its signal rings louder than the noise at the healthy pixels it is being compared with.
A lesion that is only slightly brighter or darker than the tissue it replaced gives a quiet signal, and healthy
edges drown it out. A very bright lesion on a mostly mid-grey brain gives a loud signal, which is why these
methods look good on hyperintense lesions (bright on FLAIR or T2) and are accused of being "white object detectors".

The authors' point about the **fix** is the less obvious one: making the autoencoder reconstruct better does not
help much, because a stronger autoencoder also reconstructs the lesion (it generalises), so the lesion signal
shrinks along with the noise.

## Notation in plain words

- $x$: the (possibly abnormal) image, $\tilde x$: what the same image would look like if healthy (known only
  because the authors paste the lesion in themselves), $\hat x$: the reconstruction. Subscript $i$ is a pixel.
- Residual map: $a_i=|x_i-\hat x_i|$, the size of the disagreement at pixel $i$ (their Eq. 1).
- Reconstruction error $e_i$: how far the healthy reconstruction is from the truth at pixel $i$; small in flat
  regions, large at edges.
- Their condition (Section 3) says, roughly: if some healthy pixel $j$ has error $|e_j|$ larger than the lesion
  residual at every lesion pixel, the healthy pixel will be ranked above the lesion and the score makes false alarms.
- **Average precision (AP)**: rank all pixels by score. Walk down the ranking; at each true lesion pixel note
  the fraction of pixels seen so far that are lesion (the precision); average those. Equals the area under the
  precision–recall curve. It is 1 for a perfect ranking and, for a random ranking, equals the fraction of lesion
  pixels.
- **Gaussian blur with width $\sigma$**: the stand-in for an imperfect network. Larger $\sigma$ means a blurrier,
  worse reconstruction. Blur has almost no effect for $2\sigma<0.5$ pixels.

## What they do (MOOD brain MRI, 256×256 slices, lesion = disc of radius 20 pixels)

- **Experiment 1, intensity.** Paste a flat disc of brightness $I\in[0,1]$ into a healthy slice. Pretend the
  network removes it perfectly but blurs the rest ($\hat x$ = blurred healthy image). Plot AP against $I$ for
  several $\sigma$. AP collapses already at small $\sigma$ for $I$ between about 0.2 and 0.6, which is where most
  brain pixels lie (average 0.446); it stays high for very bright discs.
- **Experiment 2, texture.** Keep brightness roughly unchanged but change the pattern: sink and source
  deformations (pixels pushed away from or towards the centre) and pixel shuffling. AP behaves like the constant
  $I=0.44$ disc, so texture alone is not detected by the residual.
- **Experiment 3, real autoencoders.** Train vanilla and spatial autoencoders with different latent sizes and a
  VQ-VAE-2. (3.1) Reconstruct the *healthy* image: curves match a blurred image with some best-fit $\sigma$.
  (3.2) Reconstruct the *anomalous* image, the realistic case: now bigger latent spaces give worse AP, and the
  VQ-VAE, the best reconstructor, drops most because it reconstructs the lesion too.
- **Appendices.** Histogram-equalising the slices makes the blind spot spread evenly over all intensities,
  supporting the idea that it sits wherever many healthy pixels live. Histograms of BraTS and MS FLAIR datasets
  show lesions are brighter than normal tissue.

## Questions and doubts

- **The "theory" is one inequality, and it is stated loosely.** Section 3 writes the lesion residual as bounded
  above by $|\tilde x_i-(x_i+e_i)|$, mixing the healthy value, the anomalous value and an error term. My reading is
  that the lesion residual is the gap between the lesion value and the healthy value, shifted by an error of size at most
  $|e_i|$, so the real statement is "a lesion is invisible when this gap is smaller than the error at healthy pixels".
  That is plausible but it is a heuristic, not a theorem; nothing bounds how often it happens.
- **One bad pixel is not enough to lose AP.** The condition says an error larger than the lesion residual at *any*
  healthy pixel $j$ causes trouble. AP depends on how *many* healthy pixels outrank the lesion pixels, so what is
  needed is the fraction of healthy pixels with large error, not the existence of one. The paper never states that
  quantity; the experiments are what do the work.
- **Intensity or contrast?** The $x$-axis is absolute brightness $I$, but the residual at a lesion pixel depends on
  $|I-\tilde x_i|$, the contrast against the healthy value *at that spot*, which differs between white matter, grey
  matter and fluid. A disc of brightness 0.45 placed in tissue that is already 0.45 is invisible even to a perfect
  reconstruction; so some of the blind spot at 0.2 to 0.6 is simply "the lesion looks like its surroundings",
  which no residual method could fix. The equalisation experiment points to the pixel-mass explanation, but it does
  not separate it from this one.
- **The blur stand-in is an oracle.** The simulated generator removes the lesion perfectly and only blurs the
  healthy image, so Experiments 1 and 2 describe an upper bound for residual scoring. That makes the negative
  result stronger, but it also means the intensity and texture curves say nothing about how real networks fail on
  the lesion itself (that is Experiment 3.2).
- **Texture is irrelevant only to this score.** Experiment 2 shows the residual ignores texture, not that
  texture is undetectable; feature-space or likelihood-based scores can use it. The conclusion should be read as
  about the pixel residual, which is what the title says.
- **No baseline or error bars.** The disc is about 1,250 of 65,536 pixels (under 2% of the slice, more of the brain), so a random score has AP
  of roughly 2 to 10% depending on whether the background is scored; the figures do not mark this, so the size of the drop cannot be read in "how far from chance".
  Whether the background is scored is not stated; if it is, brain-edge residuals add false alarms. Neural
  networks are one training run each, so the reversed ordering in Experiment 3.2 rests on a handful of models.
- **The "better reconstruction hurts" claim is shown, not explained.** Figure 4 gives one example (an enlarged
  ventricle reconstructed by the larger autoencoder) and a scatter of reconstruction error against AP across
  different architectures. Different architectures also differ in other ways, so the correlation could have
  other causes. It does agree with Baur et al., who found that higher error on normal pixels went with better
  segmentation, so the observation is not new; what is new is the simulated setting.
- **Flat discs are not pathology.** Real lesions are heterogeneous and have borders. The authors note this
  themselves; the conclusion that texture and size do not matter is for the artificial cases only.

## Takeaways

- A residual score is the sum of a real signal and reconstruction noise; the signal must beat the noise at healthy
  pixels, and the noise is largest at edges and in the intensity ranges where tissue is common.
- On brain MRI this explains why the methods mostly find hyperintense lesions.
- A sharper autoencoder is not a fix: it also reconstructs the lesion. The authors point towards likelihood-based
  and feature-based scores instead.
