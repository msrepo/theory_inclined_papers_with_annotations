---
title: "Towards IID representation learning and its application on biomedical data"
category: "Misc"
short_title: "Wu 2022 — IID representation"
authors: "Jiqing Wu, Inti Zlobec, Maxime Lafarge, Yukun He, Viktor H. Koelzer"
venue: "MIDL (PMLR 172)"
year: 2022
url: "https://proceedings.mlr.press/v172/wu22a.html"
pdf_url: "https://proceedings.mlr.press/v172/wu22a/wu22a.pdf"
tags: [out-of-distribution, causality, invariance, representation-learning, instance-normalization, pathology]
status: read
---

## Links

- **[PMLR 172:1–28](https://proceedings.mlr.press/v172/wu22a.html)**: the paper (MIDL 2022, full paper).
- **Code**: [CTPLab/IID_representation_learning](https://github.com/CTPLab/IID_representation_learning).
- **Benchmark**: WILDS (Koh et al. 2021), the out-of-distribution benchmark the experiments follow.
- **Builds on**: invariant causal prediction (Peters et al. 2016), invariant risk minimisation (Arjovsky et al.
  2019), and the causal representation learning programme of Schölkopf et al. 2021.

## In one paragraph

The usual IID assumption says every data point comes from the same distribution. Real hospital data breaks it
(different scanners, staining, sampling sites). The paper's slogan is that IID should not be an assumption but a
*property you learn*: find a function $\phi$ so that the *transformed* data $Z_i=\phi(X_i)$ do have the same
distribution. It argues that several causal-inference problems are instances of this, and then tests one cheap
version, an autoencoder with instance normalisation, on two pathology datasets with built-in distribution shift.

## The definition, in plain terms

Take several independent random vectors $X_1,\dots,X_k$ (think: data from $k$ hospitals, each its own
random draw). Pick a function $\phi$ and a "question distribution" $Q_{X_i}$ for each $X_i$, meaning whatever
distribution you care about, such as the conditional distribution of the outcome given treatment. Pushing $Q_{X_i}$
through $\phi$ gives a distribution $Q_{Z_i}$ for each hospital.

- **IID symmetry (Definition 1).** $X_1,\dots,X_k$ have a $\phi$-IID symmetry if the $k$ pushed-through
  distributions are all the same. It is *generalisable* to new data $X_{k+1},\dots,X_{k+n}$ if it stays the
  same when they are included.
- **Remark 1.** The $Z_i$ are automatically independent, because applying one fixed function to independent
  variables keeps them independent. The proof is a short expectation calculation:
  $E[f(Z_1)f(Z_2)]=E[(f\circ\phi)(X_1)\,(f\circ\phi)(X_2)]=E[f(Z_1)]\,E[f(Z_2)]$.
  So independence is free, and "identical distribution" is the only real requirement.
- **Remark 2.** A constant $\phi$ (everything maps to the same value) satisfies the definition trivially, so
  the definition is only useful when combined with a requirement that the representation keeps something
  (for example, can reconstruct the image).

## Three causal problems recast as "find $\phi$"

| Section | Causal question | The $\phi$ that induces IID |
|---|---|---|
| 3.1 | Which variables are plausible *causes* of the outcome $x_0$, given several randomised trials? | projection $\phi(x_0,\ldots,x_l)=(x_0, X_{PA(0)})$ onto the outcome and its parents, since the outcome's mechanism $f_0$ is unchanged by interventions on other variables |
| 3.2 | Can a causal effect measured in trials be *transported* to a new observational cohort? | a $\phi$ that keeps the treatment (and any variables, such as age, that must be held fixed) so the conditional outcome distribution matches across cohorts |
| 3.3 | Can we learn a *reusable feature representation* of pixels? | an encoder $\phi$ with a decoder $\phi'$, $\phi'\circ\phi=\mathrm{id}$, such that training, validation and test images have the same latent distribution |

Intuition for the first row: in a structural causal model each variable is computed from its parents plus noise,
$x_j=f_j(X_{PA_j},\delta_j)$. An intervention on one variable replaces *its* equation with a constant and leaves
the others alone, so the outcome's equation given its parents is the same in every trial. Throwing away
everything except the parents therefore makes the trials look identical. This is the invariant causal prediction
idea restated, not new theory.

## The experiment (Section 4)

- **Data (WILDS protocol).** RxRx1, fluorescence images of cells with batch effects, predicting which gene was
  perturbed; and a Swiss colorectal cancer tissue-microarray set (SCRC) where the training regions and test
  region differ (front, micro-environment, centre), predicting the molecular subtype.
- **The learned representation.** An autoencoder with a Restyle encoder and a StyleGAN decoder. To push the
  latent distributions towards identical, the encoder uses *instance normalisation*, which rescales each
  channel of each image to the same mean and standard deviation. Latents split into "morphology" (noise
  inputs) and "stain" (style inputs). Loss: pixel $L_2$ plus a perceptual loss plus a cosine-similarity term.
- **Downstream.** Freeze the encoder and add its latents into a standard classifier (ResNet, DenseNet, MobileNet,
  MnasNet), trained with cross-entropy plus an ArcFace loss.
- **Comparison.** ERM (plain training), IRM, CORAL and GroupDRO as implemented in WILDS. The paper reports that
  ERM beat the other three, and "ERM plus IID representation" beat all of them on RxRx1 and the SCRC test
  regions, across all four backbones.

## Questions and doubts

- **The theory is thin.** Remark 1 (independence is preserved by a fixed function) is a standard fact, and
  the three "reformulations" restate known ideas: invariant causal prediction, transportability and
  autoencoding. The paper itself says that a theoretical guarantee is future work.
- **Identical distributions are never checked.** The whole definition rests on $Q_{Z_1}=\dots=Q_{Z_k}$, but the
  method only uses instance normalisation, which equalises the mean and standard deviation per channel for each
  image. That makes first and second moments match; it does not make distributions identical, and a
  style-heavy shift can survive it. The paper says "approximately identical", without testing it.
- **"IID is a property to learn" is doing little work in the experiment.** Instance normalisation is an
  existing, widely used trick for removing stain and style differences. There is no ablation comparing the
  proposed encoder with a plain instance-normalised encoder or with a strong augmentation baseline, so it is not clear
  which part is responsible for the gain.
- **The comparison is not like for like.** The proposed model adds a pretrained-style StyleGAN
  encoder-decoder pipeline in front of the classifier, so it has far more capacity and pretraining than ERM,
  IRM or CORAL on the same backbone.
- **ERM beating IRM, CORAL and GroupDRO** is consistent with the literature on OOD benchmarks, but it also hints
  the baselines were not well tuned for these data.
- **A degenerate solution exists.** As Remark 2 notes, a constant map is IID. Reconstruction forces the
  representation to be informative, but nothing forces it to keep the *causal* information, which is what the
  causal motivation needs.
- **Results tables** (accuracies with four-run standard deviations) are in images in the PDF; I did not
  transcribe them, and the notes make no claim about the size of the gains.

## Takeaways

- A clean way to say what many OOD methods are after: find a representation under which the distribution you
  care about is the same across domains.
- A reminder that independence of transformed data is free; it is the *identical distribution* that needs
  to be earned.
- As a method, this is an instance-normalised autoencoder; the causal framing is the paper's argument, not
  something the experiments test.
