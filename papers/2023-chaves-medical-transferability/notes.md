---
title: "The Performance of Transferability Metrics does not Translate to Medical Tasks"
category: "Transferability"
subcategory: "Applications"
short_title: "Chaves 2023 — medical TE"
authors: "Levy Chaves, Alceu Bissoto, Eduardo Valle, Sandra Avila (University of Campinas, Valeo.ai)"
venue: "arXiv"
year: 2023
url: "https://arxiv.org/abs/2308.07444"
pdf_url: "https://arxiv.org/pdf/2308.07444"
tags: [transferability, medical-imaging, benchmark, out-of-distribution, statistical-power, source-model-selection]
status: read
---

## Links

- **[arXiv:2308.07444](https://arxiv.org/abs/2308.07444)** — preprint. Code at
  [VirtualSpaceman/transfer-estimation-medical](https://github.com/VirtualSpaceman/transfer-estimation-medical).
- The scores it evaluates:
  [Tran 2019 — NCE](../2019-tran-nce-hardness/index.html),
  [Nguyen 2020 — LEEP](../2020-nguyen-leep/index.html),
  [You 2021 — LogME](../2021-you-logme/index.html),
  [Bao 2022 — H-score](../2022-bao-hscore-transferability/index.html).
- **[Claßen 2026 — TE robustness](../2026-classen-te-robustness/index.html)** — the direct
  follow-up, which takes up this paper's own call for robustness work and separates the
  evaluation-metric question this one leaves open.
- **[Runnable code](https://github.com/msrepo/theory_inclined_papers_with_annotations/blob/main/papers/2023-chaves-medical-transferability/code/power.py)** —
  not a reproduction, which is impossible here, but the statistical resolution of their Table 2:
  what a rank correlation over ten architectures can and cannot detect. `make verify` runs it.

## In one paragraph

Seven transferability scores, ten ImageNet-pretrained architectures, three medical
classification tasks — skin lesion, breast histopathology, brain tumour — and 2250 fine-tuned
models with proper hyperparameter search behind every reference number. The conclusion is
negative: no score reliably or consistently predicts target performance in a medical context,
and the authors recommend practitioners not rely on them yet. The distinctive contribution is
the **out-of-distribution axis**: each task gets a second, genuinely shifted test set, and the
scores are evaluated against OOD performance too — the first study to do that. The surprise is
that the OOD column is where the scores work *best*, and specifically where the label-based ones
do. The reservation is about resolution: ten architectures is a small pool for a rank
correlation, and that turns out to decide which of their forty-two numbers carry information.

## What was run

| | |
|---|---|
| scores | H-Score, NCE, LEEP, $\mathcal{N}$-LEEP, LogME, Regularised H-Score, GBC |
| architectures | 10 ImageNet-pretrained (ResNet-18/34/50, MobileNetV2 ×2, DenseNet-121/161/169, EfficientNet-B0, ViT-Small) |
| in-distribution targets | BrainTumor-Cheng, BreakHis, ISIC2019 |
| **out-of-distribution** targets | NINS, ICIAR2018, PAD-UFES-20 |
| target metric | **balanced accuracy** |
| tuning | 75 quasi-random (Halton) configs per architecture; SGD, cosine, 100 epochs |
| total | **2250 models trained** |
| correlation | Kendall's $\tau$ |

Two things are done better than the norm here and deserve saying. The hyperparameter search is
real — 75 configurations per architecture, so the reference ranking reflects each model at its
best rather than at default settings, which is the usual shortcut. And **balanced accuracy** is
used rather than plain accuracy, which is the right choice for imbalanced medical data. Both
matter because the reference ranking is what everything is measured against.

## The taxonomy that turns out to matter

They split the scores in two:

- **feature-based (fb)**: need only the source *feature extractor* — H-Score, $\mathcal{N}$-LEEP,
  LogME, Regularised H-Score, GBC.
- **label-based (lb)**: need the source *classification head* — NCE, LEEP.

This is the same split that runs through the [Theory](../2021-you-logme/index.html) section, and
it is exactly the axis along which their results separate.

## The findings

**In-distribution: unstable, and inconsistent across datasets.** LogME is among the best on
BrainTumor-Cheng ($\tau=0.584$) and BreakHis ($0.378$), and **negative** on ISIC2019 ($-0.067$).
H-Score goes $0.270 \to 0.600 \to -0.244$ across the three. No score keeps its sign, let alone
its rank. Their hypothesis is domain dissimilarity: unlike the general-purpose benchmarks these
scores were validated on, medical targets share neither classes nor low-level statistics with
ImageNet.

**Out-of-distribution: the label-based scores do well.** NCE reaches $0.911$ on PAD-UFES-20 and
$0.778$ on ICIAR2018; LEEP reaches $0.778$ on ICIAR2018. This inverts the in-distribution
picture, where NCE is often negative.

Their explanation is worth quoting because it is a real mechanism and not hand-waving: for
**binary** targets with few classes, the source model's predicted distribution concentrates on a
single class, which **inflates** the label-based score. So the OOD success may be an artefact of
task shape rather than genuine OOD sensitivity. They flag this themselves.

**Two candidate causes, tested.** They fine-tuned on each medical dataset and re-scored against
the validation set: only label-based methods did well, so domain shift "helps to degrade the
efficiency of such scores, but it is not the main reason". They then collapsed OxfordPets from
37 classes to binary cats-vs-dogs: correlations dropped but stayed high. So neither shift nor
class count alone explains the failure — a genuinely useful negative result that narrows the
space.

## The instrument's resolution

The experiment is 2250 models and cannot be reproduced here. What can be checked is how much
signal a rank correlation over **ten** items can carry, and it decides how to read Table 2.

Under the null, Kendall's $\tau$ at $n=10$ has standard deviation
$\sqrt{2(2n+5)/(9n(n-1))} = 0.248$ — confirmed by simulation to three decimals. So:

| $n$ | sd under null | $\lvert\tau\rvert$ for $p<0.05$ |
|---|---|---|
| **10** | **0.248** | **0.467** |
| 20 | 0.163 | 0.316 |
| 50 | 0.097 | 0.192 |

At $n=10$, a reported $\tau$ of $0.270$ is barely one standard error from zero. Applying the
threshold to all 42 cells of their Table 2:

- **6 of 42** clear $p<0.05$, against **~2.1 expected by chance** with no correction.
- **3 clear Bonferroni** ($\lvert\tau\rvert \ge 0.733$), and they are:

| scorer | dataset | $\tau$ | |
|---|---|---|---|
| NCE | PAD-UFES-20 | +0.911 | **OOD** |
| NCE | ICIAR2018 | +0.778 | **OOD** |
| LEEP | ICIAR2018 | +0.778 | **OOD** |

**Every survivor is an out-of-distribution cell, and every one is a label-based scorer.** That is
exactly the paper's *positive* finding, and it is the part that is statistically solid.

The headline *negative* finding is the underpowered part: the in-distribution cells are not
distinguishable from zero in either direction. Power to detect a true $\tau=0.3$ at $n=10$ is
**14%**:

| $n$ | 10 | 20 | 30 | 50 | 80 |
|---|---|---|---|---|---|
| power at true $\tau=0.3$ | **0.14** | 0.23 | 0.35 | 0.51 | 0.74 |

This is a criticism of the evidence, not of the conclusion. Underpowered null results err in the
safe direction — failing to detect a correlation is not evidence that none exists — so
"don't rely on these scores in medical imaging yet" remains sound advice. But it is supported by
far less than forty-two numbers suggest, and the paper reports point estimates with no interval
anywhere.

## Questions and doubts

- **Ten architectures is the binding constraint.** Every conclusion is a rank correlation over
  ten items, where the null standard deviation is 0.248. Confidence intervals would have made
  the strength of each claim visible; none are reported.
- **The OOD result may be a task-shape artefact**, as they say themselves: binary targets
  concentrate the source model's predictions and inflate label-based scores. Since all three OOD
  sets are binary or near-binary, the cleanest interpretation of their strongest result is also
  the least flattering to it.
- **Single seed, single subset.** [Claßen et al.](../2026-classen-te-robustness/index.html)
  later showed that rankings churn under reseeding alone, which means each of these 42 numbers
  carries an unreported sampling variance on top of the null variance above.
- **Balanced accuracy is the right choice but creates a mismatch** that goes unexamined: most of
  these scores were designed and validated against plain accuracy. Claßen et al. make this the
  explicit subject of their second experiment and find the reference ranking does depend on the
  criterion — so part of what is measured here as "the score fails" may be "the score optimises
  something else".
- **Table 2's caption says $\tau_w$ while the text says Kendall's $\tau$.** A small
  inconsistency, but the two weight disagreements differently and the distinction matters when
  the whole result is a table of correlations.
- **The in-distribution instability is reported but not diagnosed.** LogME swinging from $+0.584$
  to $-0.067$ across datasets is the most interesting number in the paper, and the two causes
  they test (shift, class count) are both ruled out without a third being proposed.

## Takeaways

- **The OOD axis is the contribution.** Being first to ask whether transferability scores predict
  *out-of-distribution* performance is the durable idea here, and it is the axis on which the
  results are cleanest — the only cells that survive correction are OOD ones.
- **Feature-based versus label-based is the split that separates the results**, and it separates
  them in the opposite direction from the general-purpose literature: label-based scores are the
  ones that work here, on OOD targets, possibly for the wrong reason.
- **Read Table 2 with the null in mind.** At $n=10$ the resolution is $\pm 0.25$, so most cells
  say nothing. The three that survive correction all point the same way, which is more
  informative than the forty-two taken at face value.
- Together with [Claßen et al.](../2026-classen-te-robustness/index.html) the picture is
  consistent: these scores were validated where source and target overlap, and medical targets
  break a different assumption for each of them. Neither paper shows the scores are worthless —
  they show the evidence base for using them in this domain does not exist yet.
