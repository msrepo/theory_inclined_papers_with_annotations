---
title: "ANNPR 2026 Position Paper Track: the three papers and what to take from them"
authors: "Topic notes"
venue: "ANNPR 2026 (Position Papers Track)"
tags: [annpr, position-papers, conference-overview, continual-learning, histopathology, explainability, baselines]
status: living
category: "Research practice & policy"
subcategory: "Conference overviews"
short_title: "ANNPR 2026 — position papers"
summary: "the three papers in the first ANNPR position papers track (Milan, October 2026): a strong forgotten baseline for continual learning, cell-centric molecular prediction from histopathology, and explainability as a bridge to space law"
related: [ijcnn2026-position-overview, icml2026-position-overview]
---

## Why this page exists

The 12th IAPR TC3 Workshop on Artificial Neural Networks in Pattern Recognition (ANNPR 2026, University
of Milan, 7–9 October 2026) launched a **Position Papers Track** alongside its main track. The
programme has one position-paper session, chaired by Francesco Pistolesi, with **three papers**. They
are published in the Springer LNAI proceedings
([Artificial Neural Networks in Pattern Recognition, ANNPR 2026](https://link.springer.com/book/10.1007/978-3-032-39028-8)),
as chapters 37–39. The notes below are from the abstracts only.

With three papers there are no topic groups; each paper is summarised on its own, followed by what it
offers a theory-inclined, medical-imaging reader.

## The papers

### How Streaming LDA Questions our Progress in Continual Learning with Pretrained Models

[Springer chapter](https://link.springer.com/chapter/10.1007/978-3-032-39028-8_38)

**Position.** Reported progress in class-incremental learning (CIL) with pretrained vision models
(prompt pools, adapters, expandable subspaces, mixtures of experts) is overstated, for two reasons.
First, Streaming Linear Discriminant Analysis (SLDA) on frozen pretrained features, an older baseline
that is hyperparameter-free, single-pass and order-invariant, is almost never reported. Where a fair
comparison can be rebuilt, it beats the nearest-class-mean baseline the field uses, beats every
prompt-based method and is competitive with the strongest exemplar-free methods; on a recent benchmark
of 300+ tasks it has the second-best final accuracy of all published methods without any backbone
tuning. Second, model selection is under-specified: hyperparameters appear as fixed constants without a
validation protocol, and where one can be inferred it often uses end-of-stream or test-set performance,
which contradicts the premise of continual learning. The paper asks for SLDA as a standard baseline and
for transparent, early-task model selection.

**Why it is worth reading.** It is the most mathematical of the three. SLDA keeps running class means
and a shared covariance, and classifies with the linear discriminant; because the sufficient statistics
are sums, the result does not depend on the order in which data arrive. A good exercise is to write the
streaming updates for the class means and the pooled covariance, check order invariance, and see why
there is nothing to forget. The model-selection argument is the same leakage problem that appears in
the [ICML 2026](../icml2026-position-overview/index.html) and
[NeurIPS 2026](../neurips2026-position-overview/index.html) evaluation groups.

### From Patch-Centric to Cell-Centric Vision-to-Omics: A Position Paper on Molecular Prediction from Histopathology

[Springer chapter](https://link.springer.com/chapter/10.1007/978-3-032-39028-8_39)

**Position.** Models that predict gene expression or cell composition from haematoxylin and eosin (H&E)
slides, a cheap alternative to transcriptomic assays, mostly work on patch-level representations, even
though molecular signals come from individual cells and their interactions. Future models should be
cell-centric and multi-scale, combining nucleus-level features with local and global tissue context.
The paper also asks the field to rethink its molecular targets and evaluation, which often rely on
model-derived rather than experimentally measured ground truth.

**Why it is worth reading.** It is the paper closest to medical imaging, and it continues the
"imaging as a stand-in for another measurement" theme in the MIDL overviews (for example spatial
transcriptomics from histology in the [MIDL 2025 overview](../midl2025-overview/index.html)). Its
warning about model-derived ground truth echoes "ground truth as a human construction" and
"measurement pipelines as inference" in the [ICML 2026 overview](../icml2026-position-overview/index.html).

### The Missing Link Between Space Law and Artificial Intelligence: A Position Paper on Explainability for Future Intelligent Space Systems

[Springer chapter](https://link.springer.com/chapter/10.1007/978-3-032-39028-8_37)

**Position.** Space law was written to be technology-neutral, long before AI systems were used in
space missions, and gives little guidance on how compliance should be assessed when algorithms make
decisions. Research on the problem has been mostly normative. The paper analyses explainable AI (XAI)
tools that can make autonomous decisions observable, understandable and auditable, and argues that XAI
is not a complete compliance solution but a supporting technology for accountability, liability,
transparency and human oversight. It sets out a research agenda for legally trustworthy AI in space.

**Why it is worth reading.** The structure of the argument (explanations as evidence for a legal
property, not as proof of compliance) carries over to medical devices, where liability and oversight
raise the same questions; compare "Explainability matters: the effect of liability rules on the
healthcare sector" in the [NeurIPS 2026 overview](../neurips2026-position-overview/index.html).

## Taken together

The three positions share one move: each says a field is measuring the wrong thing. Continual learning
compares methods without a strong simple baseline and selects hyperparameters on data it should not
see; vision-to-omics evaluates against model-made targets at the wrong spatial unit; and space-AI
compliance asks for explanations without saying what they are evidence of. For a medical-imaging group,
the first two give direct, checkable advice: report a closed-form baseline on frozen features, and
evaluate against measured, cell-level ground truth.

## Sources

- [ANNPR 2026 programme](https://annpr2026.unimi.it/program/) (position-paper session, 8 October 2026).
- [ANNPR 2026 call for papers](https://annpr2026.unimi.it/important-dates/call-for-papers/) (main track and position papers track).
- [ANNPR 2026 proceedings, Springer LNAI](https://link.springer.com/book/10.1007/978-3-032-39028-8).
