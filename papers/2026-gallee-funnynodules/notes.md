---
title: "FunnyNodules: A Customizable Medical Dataset Tailored for Evaluating Explainable AI"
category: "Medical image analysis"
subcategory: "Uncertainty & annotator noise"
short_title: "Gallée 2026 — FunnyNodules"
authors: "Luisa Gallée, Yiheng Xiong, Meinrad Beer, Michael Götz"
venue: "MIDL (PMLR 315)"
year: 2026
url: "https://proceedings.mlr.press/v315/gallee26a.html"
pdf_url: "https://raw.githubusercontent.com/mlresearch/v315/main/assets/gallee26a/gallee26a.pdf"
tags: [explainable-ai, synthetic-data, ground-truth-explanations, concept-bottleneck, prototypes, counterfactuals, shortcut-learning, ti-rads, thyroid-ultrasound]
status: read
summary: "a generator of synthetic nodule images whose class follows a known attribute rule, giving perfect ground truth for attributes and attribute regions; good for testing whether a model is right for the right reasons, though it conflates that with testing explanation faithfulness, and its Trust Index is non-zero even when attribute and target accuracy match; plus a sketch of a TI-RADS version for thyroid ultrasound"
related: [midl2026-overview, 2026-ren-shapley-uncertainty, 2026-lux-gradient-surgery]
ext_links: [code|https://github.com/XRad-Ulm/FunnyNodules]
---

## Links

- **[PMLR 315:198–214](https://proceedings.mlr.press/v315/gallee26a.html)**: the paper (MIDL 2026, full paper), read in its camera-ready form.
- **Code**: [XRad-Ulm/FunnyNodules](https://github.com/XRad-Ulm/FunnyNodules), the generator and the experiments.
- **Ancestors**: FunnyBirds (Hesse et al., ICCV 2023) for natural images, and the Elements dataset.
- **Related in this repo**: [Ren et al.](../2026-ren-shapley-uncertainty/index.html), which validates explanations against clinical rankings; the [MIDL 2026 overview](../midl2026-overview/index.html), where this paper is listed among the papers that critique the field.

These notes came out of a question-and-answer session; the questions are kept as section headings.

## In one paragraph

FunnyNodules is a generator of synthetic, lung-nodule-like images in which everything is known. Each
nodule has six controllable attributes (roundness, spiculation, edge sharpness, size and intensity on
a 1–5 scale, and internal structure present or absent), and a fixed rule turns the attributes into a
1–5 "malignancy" class. Because the images are drawn by a program, the class, every attribute and a
mask showing where each attribute lives in the image all come for free. The point is to test whether a
model, and its explanation, is right for the right reasons. The idea is useful; the paper mixes up two
different questions, and its Trust Index formula has a flaw.

## Q1. What does the paper do, and what should a reader watch for?

**The decision rule (Algorithm 1).** A score is summed and binned into classes 1 to 5:

| Attribute | Rule |
|---|---|
| Spiculation | high +2, low −2 |
| Edge sharpness | soft −2, sharp +2 |
| Size | big +2, small −2 |
| Intensity | bright −1, dark +1 |
| Roundness | depends on internal structure: without it, oval +2 and round −2; with it, the signs flip |

The roundness row is a deliberate interaction, and it is the hard part for models.

**What the paper uses it for.**

1. *"What if" sweeps.* Keep everything fixed, vary one attribute from 1 to 5, and compare the change in
   prediction with the truth through a contrastive difference
   $\Delta_{target}(a)=\mathbb{E}\big[f(x_{a=\max})-f(x_{a=\min})\big]$; for example, HierViT's
   spiculation sweep gives 1.63 against a true 1.4. Models follow the simple attributes but get the
   roundness × internal-structure interaction right only when internal structure is absent: they learn a
   main effect and miss the sign flip.
2. *A Trust Index* comparing target accuracy with attribute accuracy: high target accuracy with poor
   attributes suggests "right answer, wrong reasons".
3. *Attention against attribute regions.* The generator outputs a mask per attribute (spikes, rim,
   inner patch). HierViT's attribute attention covers the nodule outline but does not line up with
   these regions; the comparison is qualitative.
4. *Prototype checks.* Does the prototype shown for an attribute have that attribute's value? Mostly yes
   (0.86–1.0 within-1 accuracy).
5. *Data size.* Generate as many images as needed and see how explanations degrade with less data; the
   concept bottleneck model is very sensitive.

**Things to watch.**

1. **The Trust Index does not do what the text says.** As printed, and consistent with Table 2,
   $\mathrm{TI}=P_{target}-\overline{A}/P_{target}$. It is meant to be about 0 when attribute and target
   performance match, but it is zero only when both equal 1: if both are 0.8,
   $\mathrm{TI}=0.8-0.8/0.8=-0.2$, reporting "better attributes than target mapping" for a model whose two
   accuracies are equal. The −0.22 to −0.30 values for every model at 100 training images are therefore
   largely an artefact. A plain difference $P_{target}-\overline{A}$, or a ratio, would behave as
   described. (The alternative reading $(P-\overline{A})/P$ would imply attribute accuracies above 1.)
2. **Testing the model's reasoning is not testing the explanation's faithfulness.** The paper adopts
   Nauta et al.'s *correctness* (an explanation reflects what the model actually does) but measures
   agreement with the data-generating rule. A model that learned the wrong rule, with an explanation
   that faithfully reports it, would be marked as having an incorrect explanation. The known rule tests
   whether the model is right for the right reasons; faithfulness needs interventions on the model,
   as FunnyBirds does by removing parts. Attention that misses an attribute's region may mean the model
   uses other evidence, and for size and intensity the "correct" region is itself assumed.
3. **The default setting is too easy.** With 1,800 training images every model except the concept
   bottleneck reaches about 1.0 target accuracy; differences appear only with less data or harder rules.
4. **The metric is lenient.** Within-1 accuracy on a 5-point scale is about 0.52 by chance under uniform
   labels, so the concept bottleneck's 0.48–0.50 is roughly chance.
5. **No spurious correlations by default.** Attributes are independent, so there is no shortcut to
   find unless one is added; planting a known shortcut (for example background tied to class) and
   testing whether explanation and shortcut-detection methods find it would be one of the framework's
   most valuable uses.

**Why it is worth discussing.** It turns "right for the right reasons" into something measurable, which
is the missing ingredient in evaluating explanations. Good questions: what exactly a known rule lets
you test; how a Trust Index should be defined; and which planted shortcut would be most revealing.

## Q2. Would a similar idea work for TI-RADS in thyroid ultrasound?

Yes, and ACR TI-RADS may fit better than FunnyNodules' invented rule, because it is a real,
points-based rule that radiologists already use. A synthetic version would test whether models reason
the way the guideline does, with clinically meaningful features.

**The rule** (from memory; check against the ACR lexicon chart before building):

| Category | Options and points |
|---|---|
| Composition | cystic or spongiform 0; mixed cystic/solid 1; solid 2 |
| Echogenicity | anechoic 0; hyper- or isoechoic 1; hypoechoic 2; very hypoechoic 3 |
| Shape (transverse view) | wider-than-tall 0; taller-than-wide 3 |
| Margin | smooth or ill-defined 0; lobulated or irregular 2; extrathyroidal extension 3 |
| Echogenic foci (add all that apply) | none or large comet-tail 0; macrocalcifications 1; peripheral (rim) 2; punctate 3 |

Total points map to TR1–TR5, and the recommendation (biopsy, follow-up or nothing) then depends on TR
level × nodule size. This gives a mainly additive rule (sensitivity to each feature can be checked
against its points), real interactions (size thresholds by TR level; foci points that add), and, if the
lexicon is recalled correctly, an override: spongiform nodules are scored low without adding points from
other categories, the kind of rule models miss, like the roundness sign flip.

**What ultrasound adds.** Features where appearance and meaning diverge:

- *Bright dots with opposite meanings.* Large comet-tail artefacts (reverberation, typically benign
  colloid) score 0; punctate echogenic foci, which can look similar, score 3.
- *Echogenicity is relative*, judged against surrounding thyroid tissue (and strap muscles for "very
  hypoechoic"); a global gain change tests whether the model uses the reference tissue.
- *Shape depends on the scan plane*: taller-than-wide is defined in the transverse view.
- *Calcifications cast shadows*, partly hiding the margin, so evidence for one feature degrades another.
- *Margins need anatomy*: extrathyroidal extension means crossing the thyroid capsule.

**Shortcuts to plant.** Caliper marks and on-screen annotations (which tend to appear on measured,
suspicious nodules), machine or vendor, preset, depth and gain, cropping and text overlays. Tie any of
them to TR level at a chosen strength and measure whether models use them and whether explanation or
shortcut-detection methods notice.

**Two levels of realism.**

| | Procedural (FunnyNodules-style) | Physics-based |
|---|---|---|
| How | draw the nodule on textured background, add speckle-like noise, place bright foci, draw shadows | scatterer maps (density sets echogenicity, strong reflectors make calcifications) simulated with tools such as Field II or k-Wave |
| Ground truth | features, points, per-feature masks | the same, plus physically correct speckle, shadowing and comet-tail artefacts |
| Cost | seconds per image | minutes per image or more |
| Best for | fast experiments, explanation methods, planted shortcuts | checking that findings carry over to realistic image formation |

A sensible route is procedural first, with a smaller physics-simulated test set.

**Limits to state up front.**

- *TI-RADS is a risk score, not a diagnosis.* The dataset tests guideline-consistent reasoning, not
  cancer detection.
- *Synthetic does not transfer automatically.* Models can learn the generator's quirks; pair it with real
  data that has TI-RADS feature labels (public thyroid ultrasound sets exist, for example DDTI and Stanford
  AIMI's thyroid cine-clip collection; check which have per-feature labels and on what terms).
- *Real labels are noisy.* Echogenicity and margin have weak agreement between readers; adding controlled
  label noise to the perfect labels allows rater disagreement to be studied with the true answer known,
  which links to the multi-rater discussion in the [Lux et al. notes](../2026-lux-gradient-surgery/index.html).

**A first version.**

1. *Generator:* the five categories with their options, nodule size, transverse and longitudinal views,
   speckle level, gain, depth, and optional calipers or text overlays.
2. *Outputs per image:* feature labels, points, TR level, the size-based recommendation, and a mask per
   feature (margin rim, foci locations, shadow region, reference tissue patch).
3. *Evaluation suite:* single-feature sweeps against point values; the spongiform override and the size ×
   TR interaction; attention against feature masks; comet-tail versus punctate confusion; gain changes for
   echogenicity; planted-caliper shortcut tests.
4. *A small real-data check* on a public thyroid set.

The same "known factors, planted shortcuts" approach could later serve the blind-sweep work: synthetic
fetal sweeps with controllable, known factors for testing interpretability of gestational-age models.

## Questions and doubts

- What should a Trust Index be: a difference, a ratio, or something calibrated to chance?
- How should explanation faithfulness, as opposed to model correctness, be tested on a known-rule dataset?
- Which planted shortcut would most clearly separate good from poor explanation methods?

## Takeaways

- A known decision rule makes "right for the right reasons" measurable for the model; faithfulness of
  explanations still needs interventions on the model itself.
- The Trust Index as printed is non-zero when attribute and target accuracy are equal.
- At default difficulty the dataset does not separate models; harder rules, less data or planted
  shortcuts are where it becomes informative.
- A TI-RADS version for thyroid ultrasound is a natural extension, with a real clinical rule and
  ultrasound-specific confounders such as comet-tail versus punctate foci and gain-dependent echogenicity.
