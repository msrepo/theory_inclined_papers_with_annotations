---
title: "Is Exchangeability better than I.I.D. to handle Data Distribution Shifts while Pooling Data for Data-scarce Medical Image Segmentation?"
category: "Medical image analysis"
subcategory: "Domain adaptation / generalization"
short_title: "Roy 2026 — exchangeability and pooling"
authors: "Ayush Roy, Samin Enam, Jun Xia, Won Hwa Kim, Vishnu Suresh Lokhande"
venue: "MIDL (PMLR 315)"
year: 2026
url: "https://proceedings.mlr.press/v315/roy26a.html"
pdf_url: "https://arxiv.org/pdf/2507.19575"
tags: [exchangeability, de-finetti, data-pooling, distribution-shift, domain-adaptation, invariance, segmentation, causality, ultrasound, pathology]
status: read
summary: "a layer-wise foreground/background feature-separation loss plus a cross-dataset version for pooled data; exchangeability is invoked as a weaker assumption than i.i.d., but is best read as a goal for the features (class-conditional invariance across sites), which the loss only half enforces: it separates across datasets but never aligns them"
related: [midl2026-overview, 2022-wu-iid-representation, 2025-bereska-sacp, lda-fisher-discriminant, integral-probability-metrics]
---

## Links

- **[PMLR 315](https://proceedings.mlr.press/v315/roy26a.html)**: the paper (MIDL 2026).
- **[arXiv:2507.19575](https://arxiv.org/abs/2507.19575)**: v1 (July 2025, WACV format), read first. The **PMLR camera-ready** (30 pages, from the PMLR assets) was then checked for the lemmas: Lemma 1 is unchanged apart from a footnote, and it adds a proof of Lemma 2 and a new Axiom 3 on exchangeability. See [What the camera-ready changes](#what-the-camera-ready-changes).
- **Related in this repo**: [Wu et al. 2022, *Towards IID representation learning*](../2022-wu-iid-representation/index.html), which also treats "the samples share one distribution" as something to *learn* rather than assume; [SACP](../2025-bereska-sacp/index.html), whose conformal guarantee rests on exchangeability; the [MIDL 2026 overview](../midl2026-overview/index.html), where this paper is listed as an outlier.

These notes came out of a question-and-answer session; the questions are kept as section headings.

## In one paragraph

Pooling segmentation data from several sites can make a model *worse* on the original site, the
"data addition dilemma". The paper adds a loss at every U-Net layer that pushes the summed features
of foreground pixels away from those of background pixels, and a cross-dataset version that pushes
one dataset's foreground away from the other dataset's background. The title's probability idea,
exchangeability, is offered as a weaker and more realistic assumption than i.i.d. for pooled data.
In the method it does one job: it licenses pairing foreground and background across datasets. Read
as an *assumption about the data* it undermines itself; read as a *goal for the features*, "within
each class, the site should not matter", it makes sense, and the loss enforces only half of it.

## The method

**Within-dataset loss, at every layer.** With the ground-truth mask max-pooled to each layer's
resolution, sum the layer's features over foreground pixels ($F_g$) and over background pixels
($B_g$), and penalise

$$
\mathcal L_{fd} = -\log \lVert F_g - B_g\rVert^2 .
$$

Each layer has a learnable weight $\alpha$, warm-started at 0 so that early training uses only Dice
plus binary cross-entropy. In spirit this is deep supervision with a contrastive-style separation term.

**Cross-dataset loss, "under assumed exchangeability".**

$$
\mathcal L^{exch}_{fd} = -\log\Big(\lVert F_g(\mathcal D_{base})-B_g(\mathcal D_{novel})\rVert^2+\lVert F_g(\mathcal D_{novel})-B_g(\mathcal D_{base})\rVert^2\Big).
$$

**Experiments.** Histopathology (TNBC, MoNuSeg, an Alzheimer's tau dataset) and breast ultrasound
(UDIAT and a new 15-image US-TNBC set). Gains of +0.5 to +5 Dice. In the data-addition experiment
(MoNuSeg added to TNBC, UDIAT added to US-TNBC) plain training, a pixel-contrastive loss and deep
supervision lose accuracy as data is added, while the cross-dataset loss gains.

## Q1. What is exchangeability, concretely?

A sequence is **exchangeable** if the order carries no information: shuffling it gives the same
joint distribution,

$$
(X_1,\dots,X_n)\ \overset{d}{=}\ (X_{\pi(1)},\dots,X_{\pi(n)})\quad\text{for every permutation }\pi .
$$

It does **not** require independence. Every i.i.d. sequence is exchangeable; the reverse fails.

**Drawing without replacement.** An urn of red and blue balls, drawn without putting them back.
Any order of the same colours is equally likely, so the draws are exchangeable, but a red draw makes
the next red *less* likely, so they are dependent. Exchangeable, not i.i.d.

**A coin of unknown bias.** You are handed one of two coins, one mostly heads and one mostly tails,
without being told which. The flips are exchangeable (same coin throughout) and positively dependent
(a run of heads suggests the heads-heavy coin). Given the coin, they are i.i.d. **De Finetti's
theorem** says every infinite exchangeable sequence is of this form: a hidden parameter $\theta$ is
drawn once, then the samples are i.i.d. given $\theta$. The Bayesian prior is the uncertainty about
which coin you hold.

**Pólya's urn.** Each drawn ball goes back with an extra ball of its colour, so early draws shape
later ones. The sequence is still exchangeable: a sequence's probability depends only on how many
of each colour it contains. Strong dependence and exchangeability can coexist.

**One hospital's scans.** The hospital (scanner, protocol, population) plays the coin. Its scans are
correlated but interchangeable, so they are exchangeable. This is the paper's US-TNBC argument, that
batches depend on earlier batches, and on this point the paper is right.

**Not exchangeable.** Time series with trend or memory; longitudinal scans of one patient; data
collected across a scanner upgrade; actively selected or cherry-picked samples; training and test
sets from different sites. A quick test: if someone secretly shuffled the order, would anything you
believe change?

**Useful facts.** Exchangeable variables share one marginal distribution, and every pair has the same
correlation. In an infinite exchangeable sequence that correlation is non-negative; a finite one can
be negatively correlated (the urn without replacement), and then de Finetti does not apply.

**Where it matters in ML.** Conformal prediction needs only exchangeability of calibration and test
points, which is why its guarantee breaks under site shift. Permutation tests need exchangeability
under the null. Shuffled cross-validation assumes order does not matter, which is why grouped,
leave-one-site-out splits exist. Hierarchical Bayesian models are de Finetti's theorem put to work.

## Q2. Is pooled multi-site data exchangeable?

**Not fully.** Exchangeable samples share one marginal distribution, so assuming the whole pooled
set is exchangeable assumes that which site a sample came from carries no information, the opposite
of a distribution shift. The assumption that fits is **partial exchangeability**: exchangeable
*within* each site, with a parameter per site ($\theta_A$, $\theta_B$), tied together in a
hierarchical model. Full exchangeability holds only if each image's site is chosen at random and
never recorded, in which case every image is a draw from the mixture.

## Q3. How do they link exchangeability to the two losses?

Through two separate justifications, neither of which feeds a derivation.

- **Layer-wise $\mathcal L_{fd}$** is justified causally. Confounders $U$ (scanner, protocol) affect
  both image $X$ and mask $Y$; the features $Z$ are called a mediator in $X\to Z\to Y$, and a good
  mediator should track $Y$, hence foreground/background separation. Proposition 1,
  $P(Y\mid X)=\int P(Y\mid Z)P(Z\mid X)$, is the Markov-chain property, not front-door adjustment,
  which concerns $P(Y\mid do(X))$ and is not used.
- **Cross-dataset $\mathcal L^{exch}_{fd}$** is justified by exchangeability. If the datasets are
  exchangeable, "base" and "novel" are arbitrary labels, so a foreground feature from one dataset and
  a background feature from the other form as legitimate a pair as two from the same dataset.

**The tension.** If base and novel really were exchangeable, their features would have the same
distribution and the cross term would add nothing beyond the within-dataset term. The cross term
changes training only when the datasets *differ*, that is, when exchangeability fails.

## Q4. So what are they really doing? Enforcing exchangeability

The reading that makes sense: **exchangeability is a goal for the features, not an assumption about
the data.** The data is only partially exchangeable; the loss tries to make the representation
behave as if it were fully exchangeable. The useful target is exchangeability *within each class,
across datasets*:

$$
P(Z\mid Y=\text{fg},\ D=\text{base})=P(Z\mid Y=\text{fg},\ D=\text{novel}),\quad\text{and likewise for bg},
$$

that is, $Z\perp D\mid Y$: shuffling dataset labels among foreground features changes nothing, and
the same for background. Domain adaptation calls this **class-conditional invariance**; fairness calls
the same condition **separation** (equalised odds). Conditioning on the class matters: forcing the
*marginal* feature distributions to match can hurt when foreground proportions differ between sites
(Zhao et al., ICML 2019, *On Learning Invariant Representations for Domain Adaptation*).

<figure>
<img src="figures/separation-vs-alignment.svg" alt="Three panels of feature space with orange foreground and blue background marks, filled for site A and hollow for site B. A: site B's background overlaps site A's foreground. B: after the paper's loss there is no overlap, but A and B foreground form separate clusters. C: under class-conditional exchangeability, foreground from both sites forms one mixed cluster and background another.">
<figcaption><b>Separation is not exchangeability.</b> Orange = foreground, blue = background;
filled = site A, hollow = site B. <b>A:</b> pooling without the loss lets one site's background fall
on the other's foreground. <b>B:</b> the paper's cross-dataset loss removes that crossover, but the
site can still be read off the features. <b>C:</b> class-conditional exchangeability: within each
class the two sites are mixed, so the site cannot be read off.</figcaption>
</figure>

**Does the loss enforce it?** Only partly.

| Requirement | The paper's cross-dataset loss |
|---|---|
| fg(base) ≈ fg(novel), pulled together | no term |
| bg(base) ≈ bg(novel), pulled together | no term |
| fg(base) far from bg(novel), and the reverse | yes, this is all it does |

It also works on summed activations, so at best it matches first moments, and a single $-\log$ of a
sum lets one large distance hide a small one.

**A loss that would enforce it.** Keep the separation terms and add within-class alignment: pull terms
on $\lVert F_g(\text{base})-F_g(\text{novel})\rVert$ and $\lVert B_g(\text{base})-B_g(\text{novel})\rVert$;
per-class MMD or Wasserstein matching (see [integral probability metrics](../integral-probability-metrics/index.html));
or a dataset discriminator per class that the encoder learns to fool, so that chance accuracy means
exchangeable.

**How to check it.** Train a small classifier to predict the dataset from foreground features, and
separately from background features; chance accuracy means exchangeable. Or run a permutation test
that shuffles dataset labels within each class. The paper reports neither: it shows that Dice held up
as data was added, not that the features became exchangeable.

**The price.** Invariance shrinks the divergence term of the Ben-David bound but can raise $\lambda$,
the error of the best joint classifier, if it removes information the task needs. A stain or texture
difference that helps locate boundaries is an example. That is why the loss weights should be
warm-started and tuned.

## Q5. How would the idea carry over to classification?

The target becomes $Z\perp D\mid Y$ for $C$ image-level classes. The pieces map as follows.

| Segmentation (this paper) | Classification |
|---|---|
| fg / bg pixel features | per-image features, grouped by class |
| $F_g$, $B_g$ | class prototypes $\mu_{c,d}$: the mean feature of class $c$ at site $d$ |
| fg far from bg within a dataset | class means far apart within each site |
| fg(A) far from bg(B) | $\mu_{c,A}$ far from $\mu_{c',B}$ for $c\neq c'$ |
| (missing) | **alignment:** $\mu_{c,A}\approx\mu_{c,B}$ |

**Losses.** Cross-entropy plus, on running-average class-by-site prototypes at a few layers, an
alignment term $\sum_c\sum_{d\neq d'}\lVert\mu_{c,d}-\mu_{c,d'}\rVert^2$ and a per-pair hinge
separation term $\sum_{c\neq c'}\sum_{d,d'}\max(0,m-\lVert\mu_{c,d}-\mu_{c',d'}\rVert)$. A one-term
alternative is **supervised contrastive learning with cross-site positives**: same class from
different sites as positives, which is the literal form of "shuffle the site label within a class".
Distribution-level versions: per-class MMD or CORAL, or a class-conditional site discriminator (CDAN).

**What is harder than in segmentation.**

1. *Few samples.* One label per image, not thousands of pixels, so per-class, per-site prototypes are
   noisy; use running averages and batches stratified by site and class.
2. *Different prevalence across sites.* A referral hospital sees more disease; condition on the class
   and weight sites equally.
3. *A class missing at a site.* Align only the pairs that exist.
4. *Site confounded with label.* If one hospital supplies most positives, the network can learn the
   scanner as a proxy for the diagnosis. Enforcing $Z\perp D\mid Y$ removes that shortcut, which is
   the strongest reason to use the idea.

**Checks.** A per-class site classifier at chance; a permutation test within classes; a data-addition
curve on both the base and the new site's test sets; and a weight-space check that the classifier head
is uncorrelated with a site head (Wong et al., *Weight Space Correlation Analysis*, MIDL 2026).

**For regression,** such as gestational age from blind-sweep ultrasound, condition on bins of the
target: $Z\perp\text{site/device}\mid\text{GA bin}$, so that alignment does not erase genuine
differences in the GA distribution between sites.

## The theory claimed, read critically

| Claim | What it actually says |
|---|---|
| Front-door adjustment with mediator $Z$ | Proposition 1 is the Markov-chain identity; no $do(\cdot)$ step is taken, and $Z$ is built using the label |
| Lemma 1: $-\log(\text{Dice}\cdot(k+1))\le\mathcal L_{fd}$ | a lower $\mathcal L_{fd}$ raises a floor under Dice, meaningful only if $k$ is fixed, which the paper does not pin down |
| Lemma 2: minimising $\mathcal L_{fd}$ bounds $\lVert W\rVert_2$ | looks backwards: with $F=Wx$, $-\log\lVert W(x_g-x_b)\rVert^2$ falls as $W$ grows, unless normalisation layers stop it |
| "Axiom 2": SGD on convex Lipschitz functions | a textbook theorem, called an axiom, assuming convexity that networks lack |
| Features "channel-averaged" | the formula sums over pixels (and, for $F_g$, channels), so the loss depends on object size |

## Lemma 1, step by step

**The claim:** $-\log\big(\text{Dice}\times(k+1)\big)\le\mathcal L_{fd}$, that is,
Dice ≥ $e^{-\mathcal L_{fd}}/(k+1)$: more feature separation should guarantee a higher Dice.

**Notation.** $y$ is the predicted mask and $\tilde y$ the ground-truth mask. Summed over pixels
$i,j$ and channels $k$, write $S=\sum F$ for the total feature sum and $S_{fg}=\sum F\,\tilde y$
for the foreground feature sum. Because the sums run over channels too, each is a single number, so
every $\lVert\cdot\rVert_2$ in the proof is an absolute value.

| Step | What the proof writes | Verdict |
|---|---|---|
| Eq. 9 | $\sum\tilde y = \frac{\text{Dice}}{2}\cdot\frac{\sum y+\sum\tilde y}{\sum y}$, "from the Dice definition" | **False.** Dice gives the *overlap*, $\sum y\tilde y=\frac{\text{Dice}}{2}(\sum y+\sum\tilde y)$. Eq. 9 amounts to replacing that overlap with a product of sums, $\sum y\cdot\sum\tilde y$, a different and much larger quantity. A perfect prediction of a 100-pixel object (Dice = 1) gives 100 on the left and 1 on the right. |
| Eq. 10 | $FD=\lvert 2S_{fg}-S\rvert/\lvert S\rvert$ | Correct algebra, since $F_g-B_g=2S_{fg}-S$. But this FD is **normalised and not squared**, so it is not the quantity inside $\mathcal L_{fd}$. |
| Eq. 11 | $FD\le 2\lvert S_{fg}\rvert/\lvert S\rvert+1$ | **Valid** (triangle inequality). |
| Eq. 12 | substitutes Eq. 9 into $S_{fg}$ | **Invalid twice.** It uses the false Eq. 9, and Eq. 9 concerns the mask's *area* $\sum\tilde y$, whereas $S_{fg}$ is a *feature-weighted* sum, not a multiple of the area. The result is $FD-1\le\text{Dice}\,(1+k')$ with $k'=\sum\tilde y/\sum y$, the true area over the predicted area. |
| Eq. 13 | "taking $-\log$": $-\log FD\ge-\log(\text{Dice}(k+1))$ | **Drops a term.** From $FD\le1+\text{Dice}(k+1)$ only $-\log FD\ge-\log\big(1+\text{Dice}(k+1)\big)$ follows. Dropping the 1 makes the stated inequality stronger than what was shown. |
| Eq. 14 | so $\mathcal L_{fd}\ge-\log(\text{Dice}(k+1))$ | **Wrong quantity.** $-\log FD$ (normalised, unsquared) is not $\mathcal L_{fd}=-\log\lVert F_g-B_g\rVert^2$. |

**Why the lemma cannot hold as stated.** The proof never uses how the prediction $y$ is computed from
the features $F$; the only link between them is the false Eq. 9. So the lemma claims the inequality
for *any* features and *any* prediction, and that fails at once. Take features that separate
foreground from background well, so $\mathcal L_{fd}$ is finite, and a final mask that misses the
object entirely, so Dice = 0. The left side, $-\log 0$, is $+\infty$ and cannot be ≤ a finite
$\mathcal L_{fd}$. With the ε smoothing in Dice the left side is huge rather than infinite, and the
inequality still fails.

**What a correct version would need.**

1. A link from features to prediction, for example that the mask is a thresholded linear read-out of
   the last layer.
2. Spread as well as separation. A large gap between class means does not prevent overlap if each
   class is widely spread; the right quantity is a Fisher-type ratio, the gap between means over the
   spread within classes (see the [LDA / Fisher discriminant note](../lda-fisher-discriminant/index.html)).
   Under assumptions such as Gaussian classes, that ratio bounds the error of the best linear
   read-out, and pixel error bounds Dice.
3. A constant that does not depend on the output. Their $k'$, true area over predicted area, changes
   with every prediction.

What remains is the empirical evidence: across samples, $\mathcal L_{fd}$ correlates with Dice
(Figure 1(a)). That is plausible and useful; Lemma 1 does not explain it.

## What the camera-ready changes

**Lemma 1: unchanged.** The statement and the proof (Appendix C.1, Eqs. 9–14) are word for word the
same as in v1, including the same $\hat y$ typo in Eq. 12. Every problem above still applies. The one
addition is footnote 3: the bound "may not be tight", but experiments "show a strict upper-lower
bound relationship, indicating that minimizing $\mathcal L_{fd}$ directly improves the Dice score".
That concedes looseness, not invalidity, and a correlation in experiments cannot establish that a
bound holds.

**Lemma 2: a proof is added (Appendix C.2), and it shows the opposite direction.** The lemma still
claims that minimising $\mathcal L_{fd}$ puts an *upper* bound on $\lVert W\rVert_2$. The proof:

1. minimising the loss requires a minimum separation, $\lVert W\Delta x\rVert^2\ge\gamma$;
2. by Cauchy–Schwarz, $\lVert W\rVert_2\ge\sqrt\gamma/\lVert\Delta x\rVert$, which is a **lower**
   bound on the weight norm;
3. the upper-bound claim then rests on an informal "adaptive damping" argument: the gradient carries a
   factor $1/\lVert W\Delta x\rVert^2$, so steps shrink as separation grows.

Shrinking steps slow the growth of $W$ but do not cap it. Every step still points towards larger
$W$, so $\lVert W\rVert$ keeps growing, roughly logarithmically, unless normalisation or weight decay
stops it.

**Axiom 3: exchangeability of pooled data (Appendix C.3), new.** It is stated over triples
$(X_i,Y_i,S_i)$ with $S_i$ the source site: permuting whole triples leaves the joint distribution
unchanged.

*What is right about it.* This is a sensible, weak assumption. Shuffling the rows of a pooled
dataset, each row keeping its site label, changes nothing, and sites may still differ ($P(X\mid S)$
can vary). It corresponds to "each sample's site is drawn at random and recorded". It answers the
objection in [Q2](#q2.-is-pooled-multi-site-data-exchangeable) that full exchangeability of the
images alone would assume the shift away.

*Where it breaks.* The proof sketch then concludes that "permuting the source labels does not change
the joint distribution of (X, Y, Z)", which is a different and much stronger statement.

| | Axiom 3 as stated | What the sketch concludes |
|---|---|---|
| What is permuted | whole rows $(X,Y,S)$ | the site labels $S$ alone |
| What it means | the order of samples does not matter | the site carries no information, $S\perp(X,Y,Z)$ |
| Site shift allowed? | yes | no |

The cross-dataset loss needs the second, or more precisely $Z\perp S\mid Y$, to treat base and novel
features as interchangeable, and Axiom 3 does not imply it. The sketch also argues that, given $Z$,
the confounders are independent of $S$, and that front-door identification makes the effect of $X$
on $Y$ "invariant across sources". Being able to identify the effect *within* each site does not make
the effect, or the feature distributions, the same *across* sites.

This supports the reading in [Q4](#q4.-so-what-are-they-really-doing-enforcing-exchangeability): the
axiom they can defend is too weak to justify the loss, and the property the loss needs, invariance
across sites within each class, is something to *enforce* in the features rather than assume of the
data.

## Questions and doubts

- The test sets are tiny (TNBC 50, MoNuSeg 44, US-TNBC 15 images in total), and the "worst-off"
  samples are chosen from the baseline's own scores, so part of their gain may be regression to the
  mean.
- The data-addition experiments pair two H&E nuclei sets and two breast-ultrasound sets, a mild shift,
  one pairing per modality.
- Would adding within-class alignment terms improve the data-addition curve, or does the separation
  term already do most of the work?
- Figure 2c of the camera-ready, the causal graph Axiom 3 appeals to, was not examined; the
  appendix text is enough to see the gap between permuting rows and permuting site labels.

## Takeaways

- **Exchangeability is weaker than i.i.d.**: it allows dependence but requires that order carry no
  information, and it implies identical marginals.
- **Pooled multi-site data is partially exchangeable,** not fully: exchangeable within sites, with
  parameters per site.
- **The paper's real idea is to make the representation exchangeable across sites, within each
  class.** Its loss enforces the weakest part of that, no crossover between classes, and leaves out
  within-class alignment and any test of whether exchangeability was reached.
- **The idea carries over to classification and regression** as $Z\perp D\mid Y$, enforced by
  alignment plus separation on class-by-site prototypes, and checked with a per-class site classifier.
