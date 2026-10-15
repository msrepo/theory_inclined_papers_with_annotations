---
title: "A Probabilistic U-Net for Segmentation of Ambiguous Images"
category: "Misc"
subcategory: "Uncertainty Quantification"
short_title: "Kohl 2018 — Probabilistic U-Net"
authors: "Simon A. A. Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R. Ledsam, Klaus H. Maier-Hein, S. M. Ali Eslami, Danilo Jimenez Rezende, Olaf Ronneberger"
venue: "NeurIPS"
year: 2018
url: "https://arxiv.org/abs/1806.05034"
pdf_url: "https://arxiv.org/pdf/1806.05034"
tags: [uncertainty, segmentation, conditional-vae, u-net, multiple-annotators, generalised-energy-distance, medical-imaging]
status: read
---

## Links

- **[arXiv:1806.05034](https://arxiv.org/abs/1806.05034)**: the paper (NeurIPS 2018).
- **Code**: [SimonKohl/probabilistic_unet](https://github.com/SimonKohl/probabilistic_unet), an open-source
  re-implementation linked from the paper.
- **Extended by**: [PHiSeg (Baumgartner et al. 2019)](../2019-baumgartner-phiseg/index.html), which replaces the single
  latent code by a hierarchy.
- **Used by**: [Wundram et al. 2024](../2024-wundram-glaucoma-probabilistic-pipeline/index.html), as one of four segmenters.
- **Background**: the [ELBO page](../elbo/index.html) for the training bound.

## In one paragraph

When an image is ambiguous, a group of experts draws several different but plausible masks. This paper
trains a U-Net together with a conditional VAE so that one small random vector $z$ selects *which* plausible
mask the network draws. Sampling a different $z$ gives a different, internally consistent segmentation, at
the cost of re-running only the last few layers.

## What the model gives that others do not

| Approach | What it outputs | Why it falls short here |
|---|---|---|
| Plain U-Net | one mask | hides the ambiguity |
| Dropout U-Net | per-pixel probabilities | each sample mixes variants pixel by pixel, so no single sample is a coherent hypothesis |
| Ensemble | several masks | members are trained independently and each learns the most likely variant; the count is fixed |
| M-heads | several masks | the count is fixed in advance and the frequencies of variants are not learned |
| Probabilistic U-Net | unlimited coherent masks, with learned frequencies | |

The key phrase is *joint* over pixels. A per-pixel probability of "50% cat, 50% dog" cannot say that
the fur is either all cat or all dog; one draw of $z$ commits to a single consistent reading.

## The architecture

There are three networks and one latent space $\mathbb R^N$ (the paper found $N=6$ best on Cityscapes; 3
to 5 dimensions did similarly well).

- **U-Net** (weights $\theta$): produces image features.
- **Prior net** (weights $\omega$): looks only at the image $X$ and outputs a mean and variance in
  latent space, so $P(z\mid X)=\mathcal N(\mu_{prior}(X),\mathrm{diag}\,\sigma_{prior}(X))$. Each position in
  latent space is one segmentation variant, and the prior says how likely each variant is for this image.
- **Posterior net** (weights $\nu$): used only in training. It sees the image *and* one ground-truth mask
  $Y$ and outputs $Q(z\mid X,Y)$, which says where in latent space this particular expert's mask lives.

**Sampling.**

1. Run the U-Net and the prior net once.
2. Draw $z_i\sim P(\cdot\mid X)$ and broadcast it to an $N$-channel map the size of the mask.
3. Concatenate it to the last activation map of the U-Net.
4. Three $1\times1$ convolutions ($f_{comb}$, weights $\psi$) turn that into class scores:
   $S_i=f_{comb}(f_{U\text{-}Net}(X;\theta),z_i;\psi)$.

For $m$ samples only step 4 is repeated, which is why sampling is cheap. Injecting late is the deliberate
choice. The Image2Image VAE baseline injects into the encoder instead, which forces the latent
information to travel through the whole U-Net.

## Training

Standard conditional-VAE training. The loss is

$$\mathcal L(Y,X)=\mathbb E_{z\sim Q(\cdot\mid Y,X)}\big[-\log P_c(Y\mid S(X,z))\big]+\beta\,D_{KL}\big(Q(z\mid Y,X)\,\|\,P(z\mid X)\big)$$

- First term: cross-entropy. A $z$ drawn from the posterior, combined with the U-Net features, must
  reproduce the expert mask $Y$.
- Second term: KL from posterior to prior. The posterior knows the answer and the prior does not, and this
  term drags them together.
- The paper used $\beta=1$ for the Probabilistic U-Net.

Why it learns the spread: for the same image the posterior lands in different places for different experts'
masks, and the KL asks the one prior to cover all those places. On average over the data, the prior spreads its
mass over the variants and in proportion to how often they occur. At test time only the prior is used.

## How it is judged

The generalised energy distance (GED) with $d=1-\mathrm{IoU}$:

$$D^2_{GED}(P_{gt},P_{out})=2\,\mathbb E[d(S,Y)]-\mathbb E[d(S,S')]-\mathbb E[d(Y,Y')]$$

$Y,Y'$ are independent expert masks and $S,S'$ independent model samples. The first term rewards samples near
experts' masks. The other two subtract each set's own spread, so a model cannot win by being only accurate
(all samples identical) or only diverse. It is also a proper metric, and it penalises *missing* variants and
variants produced at the wrong frequency. Lower is better.

## Experiments

- **LIDC-IDRI lung lesions.** 1018 CT scans with four expert masks each, cropped to 180 by 180 pixel
  patches around lesions. Up to three of the four masks can be empty, because the experts disagree on
  whether the lesion is abnormal at all. About 8.9k training and 2k test patches.
- **Cityscapes with planted ambiguity.** Five classes are randomly relabelled to a twin class (for example
  sidewalk to "sidewalk 2" with probability 8/17), giving $2^5=32$ modes whose true probabilities run from 10.9%
  (nothing flipped) down to 0.5% (all flipped). Since the true distribution is known, calibration can be
  checked directly.

## Results

Mean $\hat D^2_{GED}$ for the Probabilistic U-Net (their Tables 2 and 3):

| Samples drawn | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| LIDC | 0.811 | 0.388 | 0.321 | 0.287 |
| Cityscapes | 0.874 | 0.337 | 0.248 | 0.206 |

It beat every baseline at 4, 8 and 16 samples (Wilcoxon signed-rank test at 16 samples, p of order $10^{-13}$
on LIDC and $10^{-77}$ on Cityscapes). On Cityscapes it is the only model whose frequency of predicting each of the 32
modes lies close to the true probabilities (their Fig. 5), down to the rare ones. The Image2Image VAE finds all the
modes but with the wrong frequencies. Ablations (fixed prior; fixed prior with no image context in the posterior;
injecting at the start of the U-Net) each did worse.

The latent space can be inspected. In a 2D slice for one lung example, one axis roughly encodes lesion size
(including a transition to "no lesion") and the other shape variation, and the three experts who saw a lesion lie
within one standard deviation of the prior while the expert who saw none lies near its edge.

## Questions and doubts

- **Calibration is shown on a toy.** The 32 modes are artificial, independent class flips with a
  fixed probability regardless of the image. Real ambiguity depends on the image and is not so cleanly
  discrete, so the Cityscapes calibration plot shows the capability but not that it holds in clinical data.
- **Calibration is not quantified on LIDC.** Four experts per image is too few to estimate frequencies of
  variants. Only the energy distance, which mixes closeness, spread and frequency into one number, is reported there.
- **The paper itself notes an under-conservative prior.** Where the experts agree a lesion is present, the model
  sometimes still gives non-trivial probability to "absent".
- **A trained prior can ignore the code.** The latent is concatenated as extra channels, so the decoder
  can learn to ignore it. [PHiSeg](../2019-baumgartner-phiseg/index.html) reports exactly this collapse
  to near-deterministic output when trained with one annotator per image.
- **Latent transitions give mixtures.** Between two well-separated modes the latent space has transition
  regions, which produce blends of variants (the paper says these are "as sharp as possible" but they exist).
- **$\beta$ was tuned per model.** The Image2Image VAE baseline got its own $\beta$ and latent size, which is fair, but
  the reported edge is partly a result of that tuning on a validation split.
- **Numbers do not carry across papers.** PHiSeg's paper reports 0.2393 for this same model on LIDC (100
  samples, its own split), against 0.287 at 16 samples here. They are not comparable.

## Takeaways

- Put the randomness into a latent code that is *conditioned on the image* (the prior) and *trained against the
  answer* (the posterior). That is what makes the sample frequencies mean something.
- Inject it late so that sampling more hypotheses costs almost nothing.
- Compare *sets* of masks with an energy distance, not one mask against one mask.
