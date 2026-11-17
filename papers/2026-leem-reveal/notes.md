---
title: "REVEAL: Multimodal Vision–Language Alignment of Retinal Morphometry and Clinical Risks for Incident AD and Dementia Prediction"
category: "Medical image analysis"
subcategory: "Methods"
short_title: "Leem 2026 — REVEAL (retina and dementia)"
authors: "Seowung Leem, Lin Gu, Chenyu You, Kuang Gong, Ruogu Fang"
venue: "MIDL (PMLR 315)"
year: 2026
url: "https://proceedings.mlr.press/v315/leem26a.html"
pdf_url: "https://raw.githubusercontent.com/mlresearch/v315/main/assets/leem26a/leem26a.pdf"
tags: [retina, oculomics, alzheimers, dementia, vision-language, contrastive-learning, clip, tabular-to-text, uk-biobank, replicability]
status: read
summary: "fundus photographs aligned CLIP-style with risk-factor questionnaires rewritten as text, using 'group-aware' positives (similar retinal morphometry or similar reports), to predict incident AD and dementia about 8 years ahead; the paper's own ablation shows the questionnaire, including cognitive tests, carries most of the signal, REVEAL ties a simple RETFound + GatorTron baseline, and the test set has only 17 AD cases"
related: [midl2026-overview, 2026-roy-exchangeability-pooling]
---

## Links

- **[PMLR 315:1869–1889](https://proceedings.mlr.press/v315/leem26a.html)**: the paper (MIDL 2026, full paper), read in its camera-ready form.
- **Code**: no link in the paper.
- **Related in this repo**: the [MIDL 2026 overview](../midl2026-overview/index.html), where this paper is in the *Imaging as a stand-in for another measurement* outliers; the [Roy et al. notes](../2026-roy-exchangeability-pooling/index.html) for the related question of who counts as interchangeable with whom in a contrastive or pooled setting.

These notes came out of a question-and-answer session; the questions are kept as section headings.

## In one paragraph

REVEAL tries to predict who will develop Alzheimer's disease (AD) or dementia, on average about 8
years before diagnosis, from a colour fundus photograph and a questionnaire of risk factors (age,
blood pressure, lifestyle, diet and cognitive test scores). Because pretrained vision–language models
expect text, an LLM first rewrites each questionnaire as a short "clinical report" from a fixed
template. Image and report are then aligned CLIP-style, with "group-aware" contrastive learning: two
people count as a positive pair if their retinas are structurally similar or their reports are
similar, not only when image and text belong to the same person. A support vector machine (SVM) then
predicts future AD or dementia from the learned embeddings. The idea is appealing, but the paper's own
ablation shows that most of the signal comes from the questionnaire rather than the eye.

## Q1. What does the paper do, and what do the results support?

**Turning tables into text.** Llama 3.1 fills a fixed template (Appendix A) from about 48 UK Biobank
variables: demographics; HbA1c, cholesterol and blood pressure; cognitive tests (numeric memory, fluid
intelligence, Trail-Making times and errors); depression, sleep, smoking, alcohol and cannabis;
physical activity; social life; and diet. The prompt asks for a one-to-one mapping, so the text holds
the same information as the table.

**Encoders.** RETFound for images and GatorTron for text, both frozen apart from small projection
layers, trained on about 30,000 participants who never developed AD or dementia.

**Group-aware contrastive learning (GACL).** Ordinary CLIP treats every other person's report as a
negative, even when two people are clinically almost identical. Within each batch GACL computes two
similarity matrices,

$$
S^{(F)} = F F^\top \ \ \text{(17 AutoMorph retinal measurements)},\qquad
S^{(T)} = T T^\top \ \ \text{(report embeddings)},
$$

thresholds them at $\tau_F$ and $\tau_T$, and marks a pair positive if **either** passes (logical
OR), on the argument that risk factors may appear before retinal change. The loss is a sigmoid
multi-positive contrastive loss (from Bulat et al., 2024) with a bias term $\beta$.

**Prediction.** An RBF-kernel SVM on the concatenated image and text embeddings, with controls sampled
to about 12% prevalence and matched on age and sex.

**Results for incident AD** (mean over 10 runs; dementia is similar):

| Model | AUROC |
|---|---|
| General medical CLIP models (BiomedCLIP, PMC-CLIP, KeepFIT, RET-CLIP) | 0.47–0.56 |
| SVM on the raw table plus retinal measurements | 0.593 |
| RETFound + GatorTron, concatenated | 0.655 |
| REVEAL without GACL | 0.654 |
| REVEAL with GACL | 0.658 ± 0.095 |

Three things stand out.

1. **REVEAL ties the simple baseline.** Against RETFound + GatorTron the AUROC difference is 0.003,
   p = 0.92 (p = 0.53 for dementia). "Substantially outperforms" holds only against the general CLIP
   models, which sit near chance.
2. **GACL adds almost nothing:** 0.654 to 0.658, against a standard deviation of about 0.095.
3. **The eye contributes little** (Appendix H):

   | | AD AUROC | Dementia AUROC |
   |---|---|---|
   | Image only | 0.561 | 0.518 |
   | Text (questionnaire) only | 0.630 | 0.641 |
   | Image + text | 0.658 | 0.659 |

**Why the questionnaire works so well.** It includes cognitive test scores, and weaker memory or slower
Trail-Making years before diagnosis are early signs of the disease itself, not independent risk
factors. Part of what the model learns may therefore be prodromal cognitive decline rather than "the
retina predicting dementia".

**Statistical caveats.** The held-out test set has 17 AD and 19 dementia cases, so an AUROC of 0.66
carries a confidence interval of roughly ±0.13; the 10 seeds measure training variation, not test
sampling. The tabular baseline is an RBF SVM with simple imputation, so "text beats table" (with the
same information) is more likely about representation than content; logistic regression or gradient
boosting would be fairer. One cohort (UK Biobank, mostly British and White), and results depend on the
GACL thresholds, which the authors acknowledge.

**Worth keeping.** GACL is a clean way to fix CLIP's false negatives with clinically meaningful
similarity, and measurement-based similarity edges out embedding-based similarity (Table 4). Rewriting
tables as text is a practical trick for language-model encoders. And the paper is a good exercise in
reading a headline claim against its own ablation.

## Q2. What part of this paper is replicable?

**In short:** with UK Biobank access the pipeline can be rebuilt fairly closely, but the headline
ranking cannot be reproduced in any meaningful sense, because the differences are far smaller than the
noise from a 17-case test set. The ablation finding (questionnaire ≫ retina) is the part most likely to
replicate.

**Available.**

| Ingredient | Status |
|---|---|
| Data | UK Biobank: approved application and fees. Dementia outcome fields given (42018, 42020, 42022, 42024). |
| Image encoder | RETFound, public weights. |
| Text encoder | GatorTron, public weights. |
| Retinal measurements | AutoMorph, public; the 17 measurements listed in Appendix F. |
| Report template | Given in full (Appendix A). |
| Loss | Formula given, from Bulat et al. (2024), with $\beta = -0.63$. |
| Hyperparameters | Given (Appendix B): learning rate, eps, weight decay, $\tau_F = 0.948$, $\tau_T = 0.981$, batch size 128, projection size 1024. |
| Code | No link. |

**Missing or underspecified.**

- *Cohort construction.* Image quality control ended with a manual review (66,251 of 136,994 images
  kept). There are 66,251 images from 39,242 people, and how both eyes are handled is not stated.
  Risk-factor field IDs, codings and missing-value handling are not given (only cannabis is described).
  Control sampling is described but the drawn controls are not listed.
- *The LLM step.* "LLaMA-3.1 API", with no model size, provider or decoding settings. Because the
  template is one-to-one, plain string formatting would give the same text deterministically, which a
  replication should use. GatorTron reads at most 512 tokens, and truncation of the long report is not
  discussed.
- *Training and evaluation.* Epochs, optimiser, image resolution and augmentation are missing; what the
  10 seeds randomise is not stated; SVM search ranges are not given.

**A useful replication design.**

1. Generate reports deterministically from the Appendix A template, with no LLM.
2. Pool cases and use repeated cross-validation over all of them, reporting a confidence interval for
   AUROC (bootstrap or DeLong), rather than one hold-out of 17.
3. Run the baselines that matter: logistic regression or gradient boosting on the table; the same
   without the cognitive tests; then adding AutoMorph measurements.
4. Run the representation ablations (concatenated encoders, REVEAL without and with GACL) on the same
   splits with paired tests.
5. Vary the GACL thresholds.

## Q3. Which public datasets beyond UK Biobank could help?

None found is fully open with retinal images and incident dementia labels. The best fits need an
application; the openly accessible one has only cross-sectional cognition. Access terms were not
confirmed and change over time, so check with each data owner.

| Dataset | What it has | Fit | Access |
|---|---|---|---|
| [AlzEye](https://pmc.ncbi.nlm.nih.gov/articles/PMC8928293) (Moorfields + NHS Hospital Episode Statistics) | about 6.3 million retinal images from 154,830 patients, 13,363 with all-cause dementia | Best match and much larger than REVEAL's 86 AD cases; but an eye-clinic population, with dementia from hospital records | Not open; linked through UCL as a trusted third party; contact the team |
| [CLSA](https://www.clsa-elcv.ca/resource-types/baseline-physical-assessments/) (Canadian Longitudinal Study on Aging) | population cohort, retinal photographs, cognitive tests, rich metadata | A check of REVEAL's ablation: on 14,711 participants, fundus images explained about 9% of the variance in cognitive scores and metadata about 20% ([Sci Rep 2022](https://www.nature.com/articles/s41598-022-09719-3)) | Application; check whether images are released or analysed in CLSA's environment |
| [ARIC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6408967) (Atherosclerosis Risk in Communities) | retinal photos at visit 3 (1993–95) and visit 5 (2011–13), dementia follow-up | Longest horizon; earlier work used graded retinal features | ARIC proposal process; a BioLINCC listing was not confirmed |
| [AI-READI](https://retinalphysician.com/issues/2025/june/ai-readi-a-multimodal-data-set-for-diabetic-eye-research/) (Bridge2AI) | type 2 diabetes cohort; colour fundus, OCT, OCT angiography; MoCA cognitive scores | Most accessible; cross-sectional cognition only, diabetes only; good for testing the method | [FAIRhub](https://fairhub.io/datasets/2), public and controlled parts |
| [AREDS / AREDS2](https://clinicaltrials.gov/study/NCT00345176) | fundus photographs, cognitive batteries (about 3,500 tested in AREDS2) | Retina and cognition in macular degeneration; the two use different instruments ([cognition study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4863274)) | Usually dbGaP; contents not confirmed |

**How to use them.** To test the clinical claim, AlzEye (many cases) or ARIC (long follow-up). For a
quick second opinion on eye versus questionnaire, CLSA, whose published result already points the same
way. To test the method without a long application, AI-READI: deterministic reports, AutoMorph
measurements, GACL versus standard CLIP, predicting MoCA. Fully open fundus datasets generally lack
cognitive or dementia outcomes and are only useful for testing the method.

## Questions and doubts

- How much of the questionnaire signal is the cognitive tests? Removing them is the single most
  informative ablation.
- Does GACL help with more data, or is its effect within noise at any scale?
- Is "text beats table" about language-model representations, or about a weak tabular baseline?

## Takeaways

- The retina alone is close to chance here; the questionnaire, including cognitive tests, carries most
  of the signal, and the retina adds about 0.02–0.03 AUROC within the noise.
- REVEAL ties a simple concatenation of two foundation models; GACL's gain is negligible at this scale.
- With 17 test cases, headline rankings are not replicable; the ablation pattern is.
- GACL, positives defined by clinical similarity, is the reusable idea, best tested on a larger or
  more accessible cohort.
