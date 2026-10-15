---
title: "PHiSeg: Capturing Uncertainty in Medical Image Segmentation"
category: "Misc"
subcategory: "Uncertainty Quantification"
short_title: "Baumgartner 2019 — PHiSeg"
authors: "Christian F. Baumgartner, Kerem C. Tezcan, Krishna Chaitanya, Andreas M. Hötker, Urs J. Muehlematter, Khoschy Schawkat, Anton S. Becker, Olivio Donati, Ender Konukoglu"
venue: "MICCAI"
year: 2019
url: "https://arxiv.org/abs/1906.04045"
pdf_url: "https://arxiv.org/pdf/1906.04045"
tags: [uncertainty, segmentation, variational-autoencoder, hierarchical-latent, multiple-annotators, medical-imaging]
status: read
---

## Links

- **[arXiv:1906.04045](https://arxiv.org/abs/1906.04045)**: the paper (MICCAI 2019).
- **Code**: [baumgach/PHiSeg-code](https://github.com/baumgach/PHiSeg-code).
- **Builds on**: [Kohl et al. 2018, the Probabilistic U-Net](../2018-kohl-probabilistic-unet/index.html).
- **Used by**: [Wundram et al. 2024](../2024-wundram-glaucoma-probabilistic-pipeline/index.html), which
  uses PHiSeg as one of four segmenters in a glaucoma pipeline.
- **[Interactive companion](figures/interactive.html)**: a toy cup outline built from a coarse random
  draw plus finer residuals, with a slider for how random each level is.

## In one paragraph

A normal segmentation network returns one mask. When experts disagree, or the border is blurry, there
are many plausible masks, and PHiSeg learns to produce them. It is a conditional variational
autoencoder (a VAE whose random code is also told about the image) with one random variable per
resolution level. The coarsest variable decides the rough shape, and each finer one adds a residual
correction at twice the resolution. Drawing a new sample means drawing a new set of codes and decoding.

## What problem it fixes

The baseline is the Probabilistic U-Net, which injects one random code at the top (full-resolution) end
of a U-Net. The authors name two weaknesses:

1. Randomness enters at one place only, so variation at coarse scales ("is this lesion big or small?")
   has to be squeezed through a single code.
2. The code is concatenated as extra channels, so the network is free to ignore it. They report that,
   trained on a single annotator, the Probabilistic U-Net sometimes collapses to near-deterministic
   output.

PHiSeg's answer is to make every skip connection of the U-Net a random sample. No information can reach
the output from the image without passing through a sampling step, so the noise cannot be bypassed.

## The model, intuition first

Sampling goes top-down:

1. Draw the coarse code $z_L$ from a distribution that depends on the image.
2. A small network looks at that draw and the image features at the next finer resolution, and outputs
   a mean and a spread. Draw $z_{L-1}$ from that Gaussian.
3. Repeat until $z_1$ (full resolution).
4. The decoder turns the codes into a mask: a rough mask from the coarsest level, then a residual added
   at each finer level.

Because each step sees the coarser draw, the fine detail stays consistent with the coarse choice.

Each $z_\ell$ is a small feature map (image size divided by $2^{\ell-1}$, with $D=2$ channels per
location), not one vector. The paper reports this keeps the spatial structure and improves accuracy. The
best model used 7 resolution levels, of which $L=5$ carry latents.

### The formulas

$$p(s\mid x)=\int p(s\mid z_1,\dots,z_L)\,p(z_1\mid z_2,x)\cdots p(z_{L-1}\mid z_L,x)\,p(z_L\mid x)\,dz$$

Read left to right: the chance of mask $s$ given image $x$ is the average, over all code draws, of the
chance the decoder produces $s$ from those codes. Each conditional factor is an axis-aligned Gaussian
(independent per location and channel):

$$p(z_\ell\mid z_{\ell+1},x)=\mathcal N\big(\phi^{(\mu)}_\ell(z_{\ell+1},x),\ \phi^{(\sigma)}_\ell(z_{\ell+1},x)\big)$$

where $\phi^{(\mu)},\phi^{(\sigma)}$ are neural networks. The posterior $q(z_\ell\mid z_{\ell+1},s,x)$
has the same form but also sees the ground-truth mask $s$.

## Training

The integral cannot be computed, so the model maximises a lower bound on $\log p(s\mid x)$ (the ELBO; see
the [ELBO page](../elbo/index.html)):

$$\mathcal L=\mathbb E_q\big[\log p(s\mid z)\big]-\alpha_L\,\mathrm{KL}\big[q(z_L\mid s,x)\,\|\,p(z_L\mid x)\big]-\sum_{\ell<L}\alpha_\ell\,\mathbb E_{q(z_{\ell+1}\mid s,x)}\Big[\mathrm{KL}\big[q(z_\ell\mid z_{\ell+1},s,x)\,\|\,p(z_\ell\mid z_{\ell+1},x)\big]\Big]$$

- The first term is cross-entropy: reconstruct the expert's mask.
- Each KL term pulls the image-only prior toward the posterior that has seen the answer. The prior has
  to learn what annotation variation looks like from the image alone, because at test time there is no
  mask.
- **Which expert?** In each batch one annotator's mask is picked at random per image. Over many steps the
  posterior sees every expert's version of the same image, so the prior has to cover all of them.
- At test time only the prior and the decoder are used.

Two departures from the clean theory were needed for stable training:

- **Weights $\alpha_\ell=2^{\ell-1}$.** A finer latent has more dimensions and so a larger KL, so the
  weights rebalance. The objective is therefore a reweighted bound, not the exact ELBO.
- **Deep supervision.** Each level's coarse output is also trained with cross-entropy against the
  downsampled ground truth. Without it a latent can match its prior and posterior and do nothing else.

## How they judge it

- **Generalised energy distance (GED).** $D^2_{GED}=2\,\mathbb E[d(s,y)]-\mathbb E[d(s,s')]-\mathbb E[d(y,y')]$
  with $d=1-\mathrm{IoU}$, $s,s'$ model samples and $y,y'$ expert masks. The first term rewards samples
  that are near experts; the other two subtract each set's own spread. A model cannot win by being only
  accurate (no spread) or only diverse. Lower is better.
- **$S_{NCC}$.** The correlation between a map of where samples disagree with their mean and the map of
  where the mean is wrong. It asks whether the model knows where it will be wrong.

Data: LIDC-IDRI lung nodules (1018 CT patches, 4 radiologists) and an in-house prostate MR set (68
patients, 6 annotators).

## Results

| Setup | Metric | Prob. U-Net | PHiSeg, $L=5$ |
|---|---|---|---|
| LIDC, all experts | $D^2_{GED}$ (lower is better) | 0.2393 | 0.2248 |
| Prostate, all experts | $D^2_{GED}$ | 0.1322 | 0.0864 |
| LIDC, one expert | $D^2_{GED}$ | 0.4452 | 0.3225 |

Dice of the mean prediction is not significantly different from a plain U-Net, so the stochastic parts
cost no accuracy. $L=1$ (no hierarchy) is worse, which is the evidence for the multi-level design.

## Questions and doubts

- **The gain depends on the dataset.** On LIDC the GED improvement over the Probabilistic U-Net is small
  (0.2248 vs 0.2393); the large gain is on the 68-patient prostate set.
- **Slices are not independent.** The significance test is a paired t-test over test slices, but slices
  from one patient are correlated, so the effective sample is smaller than it looks.
- **Capacity is only loosely matched.** $L=5$ with 7 resolution levels has far more latent dimensions
  than a single-code model. The $L=1$ ablation helps but is not a capacity-matched control.
- **The reweighted objective is not a bound any more.** With $\alpha_\ell\neq1$ the training objective no
  longer lower-bounds $\log p(s\mid x)$, so the usual reading ("maximising this raises the likelihood")
  does not strictly hold.
- **Independence within a level.** Given the coarser level, locations are sampled independently. All
  spatial correlation in the fine detail has to travel through the coarse levels; that is the argument
  for the hierarchy, but it is not tested directly.
- **A small slip in the appendix.** The appendix's Eq. 7 writes $\mathbb E_q[p(s\mid z,x)]$ without the
  log; the main text has it right.
- **The exact wiring of the skip connections** (how the coarse code reaches the finer prior network) is
  in the architecture figure and the released code, and was not checked against either here.

## Takeaways

- One random variable per resolution, each conditioned on the coarser one, forces the model to put its
  randomness where the data actually varies, and stops the decoder from ignoring it.
- GED is the right kind of metric for "is my set of samples like the experts' set", because it rewards
  both closeness and the right amount of spread.
- Training on one random expert per batch is how annotator disagreement becomes a learning signal.
