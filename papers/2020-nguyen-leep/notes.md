---
title: "LEEP: A New Measure to Evaluate Transferability of Learned Representations"
category: "Transferability"
subcategory: "Theory"
short_title: "Nguyen 2020 — LEEP"
authors: "Cuong V. Nguyen, Tal Hassner, Matthias Seeger, Cedric Archambeau (Amazon Web Services, Facebook AI)"
venue: "ICML"
year: 2020
url: "https://arxiv.org/abs/2002.12462"
pdf_url: "https://arxiv.org/pdf/2002.12462"
tags: [transfer-learning, transferability, log-likelihood, conditional-entropy, source-model-selection, calibration]
status: read
---

## Links

- **[arXiv:2002.12462](https://arxiv.org/abs/2002.12462)** — preprint. ICML 2020, PMLR 119.
  The supplementary material carrying the proofs of Properties 1 and 2 is **not** in the
  conference PDF, which ends at the references; see the caveat under Property 2.
- **[Tran 2019 — NCE](../2019-tran-nce-hardness/index.html)** — the predecessor this
  measure generalises, and the source of Property 2. Its proof turns on the empirical
  conditional being *hard*, which is exactly why the same step does not close here.
- **[Bao 2022 — H-score](../2022-bao-hscore-transferability/index.html)** — the measure this
  one is competing with, and the more interesting comparison. Read that first.
- **[Diniz 2026 — PAS](../2026-diniz-pas/index.html)** — the unsupervised counterpart:
  scores a source/backbone pair without any target labels, which this measure requires.
- **[You 2021 — LogME](../2021-you-logme/index.html)** — the Bayesian-evidence measure,
  which needs only a feature extractor and so also covers regression and contrastive or
  language-model sources.
- **[Chaves 2023 — medical TE](../2023-chaves-medical-transferability/index.html)** —
  evaluates this score on medical targets, including out-of-distribution ones.
- **[Claßen 2026 — TE robustness](../2026-classen-te-robustness/index.html)** — benchmarks
  this measure against the others on medical imaging targets, and finds the rankings move
  under nothing but a change of random seed.
- **[Shao 2022 — SFDA](../2022-shao-sfda/index.html)** — the measure that imitates
  fine-tuning dynamics rather than scoring a static representation.
- **[Runnable code](https://github.com/msrepo/theory_inclined_papers_with_annotations/blob/main/papers/2020-nguyen-leep/code/leep.py)** —
  the two exact limits, where Property 1's slack actually goes, Property 2 across six regimes
  together with why the obvious derivation misses it, and the temperature sensitivity.
  `make verify` runs it.

## In one paragraph

Push the target data through a frozen source classifier and keep the **softmax outputs** —
"dummy" label distributions over source classes, semantically meaningless on target inputs but
not uninformative. Build a soft empirical joint between source and target labels by pouring
each example's whole softmax vector into the row of its true target label, normalise to get
$\hat P(y\mid z)$, and use it to build one specific classifier: draw a source label from the
softmax, then a target label from the conditional. LEEP is that classifier's average
log-likelihood on the target data. No training, no optimisation, one forward pass. The
contrast with H-score is the thing worth understanding: H-score estimates how good the *best*
head would be and needs a local approximation to do it; LEEP evaluates *one* hand-built head
exactly, and is therefore a lower bound rather than an approximation.

## The spine of the argument

1. The source softmax on a target input is a usable signature even though its labels are
   meaningless for that input.
2. Group those signatures by true target label to get a soft joint $\hat P(y,z)$; condition it.
3. Compose: source model to dummy label, conditional to target label. Call it the Expected
   Empirical Predictor. Report its log-likelihood.
4. Since it is *a* classifier, its likelihood lower-bounds the best classifier's (Property 1).
5. It relates to Tran et al.'s NCE, which is the same construction with the softmax argmaxed
   away (Property 2).

## Setup and notation

| Symbol | Meaning |
|---|---|
| $\mathcal{Z}$ | source label set (e.g. 1000 ImageNet classes) |
| $\mathcal{Y}$ | target label set |
| $\theta$ | the frozen source model; $\theta(x)\in\Delta(\mathcal{Z})$ is its **softmax output** |
| $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n$ | the target data, with labels |
| $\hat P(y,z)$ | the soft empirical joint between target and source labels |
| $z_i = \arg\max_z\theta(x_i)_z$ | the *hard* dummy label, used only by NCE |
| $h(x)$ | the penultimate feature, which LEEP never sees directly |

The essential thing to notice before anything else: **$\theta(x)$ is the softmax output, not the
feature.** H-score works on $h(x)$; LEEP works on $\mathrm{softmax}(Wh(x))$. Almost every
difference between them descends from that one choice.

## The construction

**Step 1.** $\theta(x_i)\in\Delta(\mathcal{Z})$ for each target input — a distribution over
*source* labels. On a CIFAR image from an ImageNet model these are nonsense as labels, which is
the point: they are being used as a signature, not a prediction.

**Step 2.** Each example pours its entire softmax vector into the row of its true target label:

$$
\hat P(y,z) = \frac1n\sum_{i\,:\,y_i=y}\theta(x_i)_z ,
\qquad
\hat P(z)=\sum_y \hat P(y,z),
\qquad
\hat P(y\mid z)=\frac{\hat P(y,z)}{\hat P(z)} .
\tag{1}
$$

It is a genuine joint distribution: $\sum_{y,z}\hat P(y,z)=\frac1n\sum_i\sum_z\theta(x_i)_z=1$.

**Step 3.** The **Expected Empirical Predictor**: draw $z\sim\theta(x)$, then $y\sim\hat P(y\mid z)$.
Marginalising out $z$,

$$
p(y\mid x) = \sum_{z\in\mathcal{Z}}\hat P(y\mid z)\,\theta(x)_z ,
\qquad
T(\theta,\mathcal{D}) = \frac1n\sum_{i=1}^n\log\Big(\sum_{z}\hat P(y_i\mid z)\,\theta(x_i)_z\Big).
\tag{2}
$$

### The one structural fact

Read $\hat P$ as a $\lvert\mathcal{Z}\rvert\times\lvert\mathcal{Y}\rvert$ matrix and Eq. 2 says

$$
p(\cdot\mid x) = \hat P^{\top}\theta(x).
$$

**The EEP is a fixed linear map applied to the source softmax**, the map being column-stochastic
and obtained in closed form by counting. Everything below is a consequence: LEEP is the exact
log-likelihood of *one particular linear-in-$\theta(x)$ classifier*, chosen without optimisation.

## Property 1, and where the slack really goes

> $T(\theta,\mathcal{D}) \le l(w,k^*)$, with $k^*$ the head that maximises average
> log-likelihood over a class $\mathcal{K}$.

The proof is one line: the EEP is some element of $\mathcal{K}$, and $k^*$ maximises over
$\mathcal{K}$. **It is essentially definitional** — evaluating a specific classifier gives you no
more than the best classifier — and the paper all but concedes this, proposing to satisfy the
"$\mathcal{K}$ contains the EEP" assumption by *defining* $\mathcal{K}=\bar{\mathcal{K}}\cup\{k_{\mathrm{EEP}}\}$.
It carries no information about tightness, which is the only thing that would explain why LEEP
predicts transfer accuracy.

What the property does not say is where the gap goes, and that is worth decomposing. On the
synthetic setup in the code:

| | average log-likelihood |
|---|---|
| LEEP | −1.15561 |
| best linear head on $\theta(x)$ | −0.99808 |
| best linear head on the features $h(x)$ | −0.09325 |

$$
\underbrace{0.158}_{\text{cost of the hand-built head}}
\qquad\text{versus}\qquad
\underbrace{0.905}_{\text{cost of the softmax bottleneck}}
$$

**The larger share by far is not the head — it is routing through the source classifier at
all.** That is the structural price of LEEP's design, and it is the place where H-score is
better founded.

## Property 2, and a caveat about its proof

> $T(\theta,\mathcal{D}) \ge \mathrm{NCE}(Y\mid Z) + \frac1n\sum_i\log\theta(x_i)_{z_i}$

with $z_i$ the hard dummy label and NCE $=-H(Y\mid Z)$ on the *hard* empirical joint
(Tran et al. 2019). Tested across six regimes — well matched, many source labels, many target
labels, weak source, hot and cold softmax — **it held in all of them**.

But the obvious derivation does not reach it, and it is worth knowing why. Since every term
inside the log is non-negative, dropping all but the $z_i$ one gives

$$
T(\theta,\mathcal{D}) \;\ge\; \frac1n\sum_i\log\hat P(y_i\mid z_i) \;+\; \frac1n\sum_i\log\theta(x_i)_{z_i},
$$

and that first term is **not** NCE. It uses the *soft* conditional $\hat P$, whereas NCE uses the
hard one $\tilde P$ — and $\tilde P$ is the maximum-likelihood conditional for hard-assigned
data, so it dominates any other conditional, $\hat P$ included. The code confirms
$\frac1n\sum\log\hat P(y_i\mid z_i) < \mathrm{NCE}$ in every case tried. So this route yields a
strictly *weaker* bound than Property 2 claims, and the supplement must argue differently. The
supplement is not in the conference PDF, so I cannot check it.

Where the step *does* close is [Tran et al.](../2019-tran-nce-hardness/index.html), whose
Theorem 1 this is modelled on: there $\hat P$ is the hard empirical conditional, and
$\frac1n\sum_i\log\hat P(y_i\mid z_i) = -H(Y\mid Z)$ exactly. Moving to soft assignments is
what breaks the inheritance.

## The two exact limits, which say what LEEP is measured against

Both verified to six decimals in the code.

**An uninformative source scores exactly $-\hat H(Y)$.** If $\theta(x)$ is the same for every
input then $\hat P(y\mid z)=\hat P(y)$, the EEP predicts the target marginal, and
$T = \frac1n\sum_i\log\hat P(y_i) = -\hat H(Y)$ (measured: $-1.608927$ both ways).

This fixes the zero point, and the paper never states it. Two consequences follow immediately:

- **$T+\hat H(Y)$ is the quantity that means something** — an empirical mutual information
  between the source-derived prediction and the target label. Raw LEEP is a cross-entropy.
- **Raw LEEP is not comparable across target tasks with different $\lvert\mathcal{Y}\rvert$**,
  because the floor moves. The paper's remark that "when the target task contains more classes,
  LEEP scores tend to be smaller" is this effect, unnormalised. LEEP ranks source models against
  a *fixed* target — exactly the restriction H-score's transferability ratio has.

**A perfectly confident source collapses LEEP onto NCE.** If $\theta(x)$ is one-hot then
$\hat P=\tilde P$ and $\log\theta(x_i)_{z_i}=0$, so $T=\mathrm{NCE}$ exactly (measured:
$-1.111528$ both ways). So **the whole difference between LEEP and NCE is the softmax
uncertainty that NCE argmaxes away**, and Property 2's confidence term
$\frac1n\sum\log\theta(x_i)_{z_i}$ is precisely the measure of how much there is.

## The relationship with H-score

Both answer the same question with one forward pass and no training, and differ on essentially
every design axis.

| | [**H-score**](../2022-bao-hscore-transferability/index.html) | **LEEP** |
|---|---|---|
| what plays the role of feature | penultimate $h(x)\in\mathbb{R}^k$ | source softmax $\theta(x)\in\Delta(\mathcal{Z})$ |
| the head | optimal linear head, solved in closed form then **eliminated** | one specific head $\hat P(y\mid z)$, by counting |
| what it estimates | the **supremum** over heads | the **value at one head** |
| relation to the optimum | second-order *approximation* | *lower bound* (Property 1) |
| approximation error | $o(\varepsilon^2)$, local assumption | none — an exact log-likelihood |
| matrix inverse | yes, $k\times k$ | none |
| cost | $O(mk^2)$ | $O(n\lvert\mathcal{Z}\rvert)$ |
| conditioning hazard | real | none |
| invariant to | any invertible linear map of $f$ | permutations of $\mathcal{Z},\mathcal{Y}$ only |
| sensitive to calibration | no | **yes** |
| sees what the source head discarded | yes | no |
| exploits source-label semantics | no | yes |

### Supremum versus evaluation

H-score's derivation is *solve for the best head, substitute back, watch it cancel*; what
survives estimates how good the **best** linear head would be. LEEP writes down **one**
head and reports its likelihood. They are not competing estimators of one quantity so much as
approaches to the optimum from opposite sides, with opposite error characters: H-score is an
approximation with no guaranteed direction, LEEP a guaranteed-direction bound with no guaranteed
tightness.

### They condition in opposite directions

$$
\text{H-score's numerator: }\operatorname{cov}\big(\mathbb{E}[f(X)\mid Y]\big)
\qquad\text{versus}\qquad
\text{LEEP's head: } \hat P(Y\mid Z)
$$

Both are group-and-average operations, run the opposite way round. H-score groups **features by
target label** and asks whether the group means spread out; LEEP groups **target labels by
source label** and asks whether the resulting conditional is peaked. Same joint distribution,
approached from the two different margins.

### The common floor: both estimate $I(\text{source representation};\,\text{target label})$

H-score's $\lVert\tilde B\rVert_F^2=\chi^2(P_{XY}\Vert P_XP_Y)\approx 2I(X;Y)$, and H-score is
the share of it a $k$-dimensional feature subspace captures — a **local, second-order** mutual
information in a continuous space. LEEP's $\mathrm{NCE}=-H(Y\mid Z)=I(Y;Z)-H(Y)$, and
$T+\hat H(Y)$ is the analogous quantity — an **empirical plug-in** mutual information in a
discrete label space. Same target, two estimators: a $\chi^2$ expansion valid under weak
dependence, versus counting on a discretised representation.

### The softmax bottleneck

$\theta(x)=\mathrm{softmax}(Wh(x))$, so LEEP sees the features only after the source classifier
has compressed them. If $W$ discards a direction the target needs, LEEP is structurally blind to
it; H-score, working on $h(x)$, is not. The paper addresses this in one paragraph — "the dummy
source label distribution indirectly contains information about the input features" — which is
true but weak, since the map is a non-injective compression. The decomposition under Property 1
puts a number on it: $0.905$ nats against $0.158$ for the head choice.

Conversely LEEP gets something H-score cannot. $\hat P(y\mid z)$ is **label-level semantics**: it
encodes directly that a target class lines up with a particular group of source classes.
H-score's subspace geometry has no notion of which source class is which, and would see such an
alignment only insofar as it surfaces as feature-space variance.

### Calibration, which is a real weakness and goes unmentioned

Temperature-scaling the source logits changes nothing the model decides — same argmax, same
ranking, same discriminative content. Yet:

| $T$ | 0.25 | 0.5 | 1.0 | 2.0 | 8.0 |
|---|---|---|---|---|---|
| **LEEP** | −0.9618 | −0.9393 | **−0.9209** | −0.9621 | −1.4138 |
| NCE | −0.9947 | −0.9947 | −0.9947 | −0.9947 | −0.9947 |
| H-score on $h(x)$ | 2.6600 | 2.6600 | 2.6600 | 2.6600 | 2.6600 |

NCE is pinned because it is argmax-based; H-score on the features is pinned because it is
invariant to any invertible linear map. **LEEP reads calibration, not only discriminability**,
and swings by about half a nat over a transformation that changes no decision. Comparing source
models trained with different label smoothing, temperature, or degrees of overfitting means
comparing their calibration as much as their transferability. It is the flip side of the soft
information LEEP keeps, and the paper does not raise it.

## When each should do better

- **LEEP** when source and target label sets correspond semantically, the source model is
  well calibrated, and $\lvert\mathcal{Z}\rvert$ is large enough to be an expressive code. Its
  $\hat P(y\mid z)$ then does real work no subspace geometry can express.
- **H-score** when the target needs feature structure the source head threw away — a different
  granularity, a target orthogonal to the source labelling, or a source whose head is badly
  matched while its features are fine.
- **Neither** when $\lvert\mathcal{Y}\rvert$ is large relative to $n$: H-score's $\Sigma_T$ goes
  ill-conditioned and LEEP's $\hat P(y\mid z)$ becomes a sparse, high-variance estimate. The
  paper notes the latter without quantifying it.

## Questions and doubts

- **Property 1 is definitional** and says nothing about tightness, so it cannot explain the
  empirical correlation with transfer accuracy. That correlation is the paper's real result and
  it is entirely empirical.
- **Property 2's proof is absent from the conference PDF**, and the natural derivation provably
  does not reach it. It holds in every regime I tested; I cannot confirm the argument.
- **Calibration sensitivity is unaddressed**, and it is a confound in precisely the comparison
  the measure exists for.
- **The softmax bottleneck is dismissed in a paragraph** when it is the dominant structural
  cost, and the one place a features-based measure is clearly better founded.
- **The $-\hat H(Y)$ floor is never stated**, so the normalisation $T+\hat H(Y)$ that would make
  scores comparable across target tasks is left on the table, and the observed dependence on
  $\lvert\mathcal{Y}\rvert$ is reported as a curiosity rather than as the known offset it is.
- **"No assumptions on the data" is oversold.** LEEP drops Tran et al.'s shared-input-sample
  requirement, which is real. But it substitutes a hard structural assumption of its own: that
  the source *label set* is a sufficient code for the target task. That is not weaker, only
  different, and it is the assumption the bottleneck number above is measuring.

## Takeaways

- The construction reduces to one line worth remembering: **the EEP is a fixed stochastic matrix
  applied to the source softmax**, $p(\cdot\mid x)=\hat P^\top\theta(x)$. Everything — the bound
  direction, the cost, the limits — follows from its being an evaluation rather than an
  optimisation.
- **LEEP is NCE with the softmax uncertainty kept**, and collapses onto it exactly in the
  confident limit. That is the whole delta, and it is why the soft version wins.
- **$-\hat H(Y)$ is the floor.** Report $T+\hat H(Y)$ if you want a number that means anything
  across tasks.
- Against H-score the division is clean: **H-score picks the better feature space and the better
  head but pays an approximation; LEEP picks a worse feature space and a worse head but computes
  exactly.** Which wins is an empirical question about whether the source label set happens to
  be a good code for your target — not a question either paper's theory settles.
