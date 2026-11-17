---
title: "MultiPersistence Topological Fusion with Vision Transformers for Skin Cancer Detection"
category: "Medical image analysis"
subcategory: "Methods"
short_title: "Tastan 2026 — multipersistence for skin lesions"
authors: "Fulya Tastan, Sayoni Chakraborty, Sangyeon Lee, Baris Coskunuzer"
venue: "MIDL (PMLR 315)"
year: 2026
url: "https://proceedings.mlr.press/v315/tastan26a.html"
pdf_url: "https://raw.githubusercontent.com/mlresearch/v315/main/assets/tastan26a/tastan26a.pdf"
tags: [topological-data-analysis, persistent-homology, multiparameter-persistence, cubical-complex, betti-numbers, hilbert-function, supervised-contrastive, dermoscopy]
status: read
summary: "a 20 × 20 grid of Betti counts over a red–green bifiltration, fused with a frozen Swin Transformer through a supervised contrastive loss; the 'multipersistence' is the Hilbert function (counts computed per threshold pair, nothing tracked across scales), and the in-distribution gains are confounded with a different training recipe, while cross-dataset F1 drops below Swin-T"
related: [midl2026-overview]
ext_links: [code|https://github.com/sayoni-c98/MIDL2026-TopoConMP]
---

## Links

- **[Interactive version](figures/interactive.html)**: slide red and green thresholds over a synthetic lesion and watch the islands, holes and the 20 × 20 Betti grid change.
- **[PMLR 315:2071–2096](https://proceedings.mlr.press/v315/tastan26a.html)**: the paper (MIDL 2026, full paper), read in its camera-ready form.
- **Code**: [sayoni-c98/MIDL2026-TopoConMP](https://github.com/sayoni-c98/MIDL2026-TopoConMP), a demo that computes the features and runs on random tensors; no data pipelines, training or evaluation scripts for the reported tables.
- **Background**: Botnan and Lesnick, *An introduction to multiparameter persistence* (arXiv:2203.14289), cited by the paper.
- **Related in this repo**: the [MIDL 2026 overview](../midl2026-overview/index.html), where this paper is listed under unusual mathematics.

These notes came out of a question-and-answer session; the questions are kept as section headings.

## In one paragraph

TopoCon-MP classifies dermoscopy images (skin-lesion photographs) from two inputs: ordinary image
features from a frozen Swin Transformer, and a topological summary of the image. For the summary it
thresholds the red and green channels on a 20 × 20 grid of cut-offs and, at every grid point, counts
the connected components ($\beta_0$) and holes ($\beta_1$) of the resulting binary image, plus the
number of switched-on pixels, giving a 3 × 20 × 20 "topological image". A small network encodes it, and
a supervised contrastive loss pulls image and topology embeddings of the same class together. The
topology is a good teaching example; the "multipersistence" is a much simpler object than the name
suggests, and the main experiment does not separate the topology's contribution from a change in
training recipe.

## Q1. What is the topology, step by step?

**1. Cubical filtration of one channel.** Treat the image as a grid of pixel squares. For a threshold
$t$, keep the pixels with value $\le t$. Raising $t$ switches on more pixels, giving nested binary
images $X_1\subset X_2\subset\cdots$.

**2. Betti numbers.** In each binary image count $\beta_0$, the connected components ("islands"), and
$\beta_1$, the holes (background regions completely enclosed by foreground).

**3. Persistent homology versus Betti curves.** Persistent homology tracks individual features: when
each island or hole is born and when it dies, giving a barcode or persistence diagram in which long
bars are robust structure and short bars are usually noise. A Betti curve only records the count at
each threshold, $\beta_k(t)$: cheaper, but it cannot tell one long-lived hole from many brief ones. In
the paper's 5 × 5 example, $\beta_0=[4,2,1,1,1]$ and $\beta_1=[0,1,2,2,0]$.

**4. Two parameters.** Threshold red at $s$ and green at $t$ together: a pixel is on if $R\le s$ and
$G\le t$. This gives a 2D grid of nested binary images, a bifiltration, in which each row and column is
an ordinary one-parameter filtration.

**5. Why multiparameter persistence is hard.** Grid points are only partially ordered, since $(s_1,t_2)$
and $(s_2,t_1)$ cannot be compared, so there is no barcode in general: a feature can be born in several
incomparable ways and the algebra does not decompose into simple bars (Carlsson and Zomorodian).
Partial summaries are used instead: the Hilbert function (the Betti count at each grid point), the rank
invariant (how many features survive from one grid point to a later one), multiparameter landscapes and
others.

**6. What the paper computes is the Hilbert function.** It is the Betti count at each of the 20 × 20
grid points, as Appendix A states. The released code computes it by labelling the connected regions of
each binary mask separately (`scipy.ndimage`), so nothing is tracked across thresholds: there is no
persistence in the sense of births and deaths, only 400 independent counts of islands and holes, one per
colour-threshold pair. A reasonable and cheap descriptor, a 2D generalisation of a Betti curve, but
"multipersistence" overstates it.

## Q2. How does the learning part work, and what do the results show?

**Model.** Image branch: Swin-T, ImageNet-pretrained and frozen, with a linear head. Topology branch: an
MLP on the 3 × 20 × 20 tensor. Loss: cross-entropy plus $0.1\times$ a supervised contrastive loss, in
which the image and topology embeddings of a case act as two "views" and any two samples with the same
label are positives. The motivation is sensible: heavy augmentations can distort lesion borders and
colour, so the topological summary serves as a label-preserving second view instead.

**In-distribution** (Table 2, AUC):

| Dataset | Swin-T | TopoCon-MP |
|---|---|---|
| DermaMNIST | 93.3 | 94.9 |
| MILK-10K | 87.7 | 94.5 |
| PAD-UFES-20 | 88.6 | 93.0 |

**Cross-dataset** (train on DermaMNIST + MILK-10K, test on PAD-UFES-20; Table 3): ViT-B/16 is best (AUC
78.9, F1 45.5). TopoCon-MP has AUC 78.5 but F1 36.7, below Swin-T's 42.4, and sensitivity 42.9 against
46.3. Under shift, adding topology to Swin lowered F1 and sensitivity, which does not support "helps
stabilize performance under domain shift".

**Topology alone** (Table 4, XGBoost): on DermaMNIST the single-parameter features on all channels beat
the red–green bifiltration (AUC 92.8 against 87.1); the two-parameter version wins only on MILK-10K AUC
and on PAD-UFES-20 F1 and sensitivity.

**What the experiments cannot separate.**

1. *The training recipe changes along with the topology.* Baselines: Adam, 15 epochs, frozen backbone,
   no augmentation. TopoCon-MP: AdamW with weight decay, 30 epochs, cosine schedule, a fusion head with
   LayerNorm and dropout, plus the contrastive loss. The missing ablation is Swin-T trained with exactly
   this recipe and loss but no topology branch.
2. *"MP+SupCon" in Table 5 is the full model with images*, while its comparison rows use topology only,
   so the table compares image + topology with topology alone.
3. *The baselines are described two ways:* frozen and trained without augmentation, and also fine-tuned
   "with identical optimization and augmentation settings".
4. *The interpretability claim goes further than the descriptor.* The introduction says the topological
   channels co-localise with pigment networks and irregular borders, but the tensor's axes are colour
   thresholds, not pixel positions; the appendix shows class-average curves and heatmaps.
5. *One run, no error bars*, on test sets of about 230–1,000 images.
6. *The released code is a demo* and does not settle which DermaMNIST resolution was used; the original
   MedMNIST version is 28 × 28, and counting islands and holes on an upscaled 28 × 28 image would mean
   little.

## Q3. Is there a visualisation for the intuition?

The **[interactive explorer](figures/interactive.html)** runs the paper's computation on a made-up
64 × 64 lesion. Two sliders set the red and green thresholds; the mask shows which pixels are on; three
numbers give islands, holes and pixels on at that pair; three heatmaps give those numbers at all 20 × 20
pairs (the paper's topological image); and a curve panel shows one row of the grid, the one-parameter
Betti curve.

Things to try:

1. **Compact lesion, slide red upwards.** One island appears in the dark centre and grows; at high
   thresholds the background floods in. $\beta_0$ stays small, holes are rare, and the heatmaps show one
   tidy block.
2. **Switch to the irregular lesion.** Many islands appear at middle thresholds, enclosed gaps show up as
   holes, and the heatmaps spread out with hotspots in particular red–green combinations: the class
   signature the classifier learns from, and the reason the appendix finds broader, shifted peaks for
   irregular lesions.
3. **Compare a row with the grid.** Two lesions can have similar red-only curves but different patterns
   across red–green pairs, which is the case for two parameters.
4. **New random seed.** Exact counts move, but the overall heatmap shape is fairly stable, which is what
   makes the counts usable as features.

What it makes visible: each grid cell is computed independently, so it is the Hilbert function, not
persistence; and the heatmap axes are colour thresholds, not positions, which is why the co-localisation
claim does not follow. The lesions are synthetic and only illustrate the computation.

## Questions and doubts

- How much of the in-distribution gain survives a Swin-T baseline trained with the same recipe and loss?
- Would a true multiparameter invariant (the rank invariant or multiparameter landscapes), which tracks
  features across scales, beat the Hilbert function here?
- Which DermaMNIST resolution was used?

## Takeaways

- A clear path into cubical persistence: filtrations, Betti numbers, persistence versus Betti curves,
  and why two parameters break the barcode.
- The descriptor is the Hilbert function, the weakest common invariant of a multiparameter module; it is
  worth knowing which invariant a "multipersistence" paper actually uses.
- Topology as a label-preserving contrastive view is a good idea, still to be tested against a baseline
  trained with the same recipe, with repeated runs.
