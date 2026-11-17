---
title: "MIDL 2025: topic map, outliers and theory picks"
authors: "Topic notes"
venue: "MIDL 2025"
tags: [midl, survey, conference-overview, segmentation, foundation-models, uncertainty, registration]
status: living
category: "Medical image analysis"
subcategory: "Conference overviews"
short_title: "MIDL 2025 — overview"
summary: "a high-level map of the ~108 MIDL 2025 full papers (PMLR v301) in eleven topic groups, with an idea-level synthesis of each, the outlier papers that fall outside them, and a shortlist of the most mathematically foundational ones to read"
related: [midl2024-overview, midl2026-overview, 2025-bereska-sacp]
---

## Why this page exists

MIDL 2025 (Salt Lake City, 9–11 July 2025) was published as
[PMLR volume 301](https://proceedings.mlr.press/v301/), with about 108 full papers. This page
is the companion to the [MIDL 2024](../midl2024-overview/index.html) and
[MIDL 2026](../midl2026-overview/index.html) overviews: which topics
dominate, which papers do not fit the pattern, and which are worth a close, theory-first read.

The grouping and the syntheses are from **titles and abstracts only**; the PDFs have not been
read. Shares are approximate and many papers sit in more than one group, so they add up to
more than 100%.

A metadata quirk: the volume's entry labelled "Preface"
([anderson26a](https://proceedings.mlr.press/v301/anderson26a.html)) actually carries the abstract of
a paper on predicting visual acuity after treatment for diabetic macular oedema, fusing optical
coherence tomography with clinical data in small cohorts. It belongs with the longitudinal and
multimodal papers but is not counted here.

## Topic groups at a glance

| Group | Share | What's in it |
|---|---|---|
| [Segmentation](#segmentation) | ~29% | Semi- and weakly supervised learning, label budgets, adapting SAM, efficient and anatomy-aware architectures, distribution shift, quality estimates |
| [Foundation models and representation learning](#foundation-models-and-representation-learning) | ~22% | Self-supervised recipes adjusted for medical data (retina, chest X-ray, surgery), domain models, cheap adaptation, efficient backbones |
| [Vision-language models and reports](#vision-language-models-and-reports) | ~13% | Report generation and dialogue, change-aware VQA, visual and text prompting, reports as a source of labels |
| [Computational pathology and microscopy](#computational-pathology-and-microscopy) | ~16% | Cell-level segmentation and counting, noisy and lay labels, stain and site shift, graphs and biological constraints, virtual staining |
| [Generative models and synthesis](#generative-models-and-synthesis) | ~22% | Diffusion, GANs and VQ models for missing modalities, harmonisation, shape generation, counterfactuals and checks on generators |
| [Reconstruction and super-resolution](#reconstruction-and-super-resolution) | ~6% | MRI reconstruction without full targets, deep image priors, blind super-resolution, ultra-low-field MRI, CT perfusion curves |
| [Registration, shape and neural fields](#registration-shape-and-neural-fields) | ~12% | Forward splatting, log-Euclidean regularisation, neural displacement fields, mesh prompts, vessel-tree and point-shape representations |
| [Trustworthiness](#trustworthiness) | ~18% | Conformal prediction, built-in explanations, counterfactual shortcut tests, fairness and privacy, federated learning |
| [Longitudinal and multimodal clinical prediction](#longitudinal-and-multimodal-clinical-prediction) | ~14% | Image–tabular fusion, follow-up scans, disease trajectories and world models, fMRI connectivity graphs |
| [Video, surgery and ultrasound](#video-surgery-and-ultrasound) | ~11% | Self-supervised surgical video, temporal ultrasound models, labels from clinical notes, 2D-to-3D alignment and outpainting |
| [Benchmarks, data and validation](#benchmarks-data-and-validation) | ~14% | Multi-centre comparisons, annotation budgets, cheap label sources, federated heterogeneity, data streaming |

**Modalities, roughly by frequency:** MRI, CT, histopathology and microscopy, chest X-ray,
surgical video and ultrasound, then retina, mammography and others.

**Compared with MIDL 2026:** a volume about half the size, with no agents or reinforcement-learning
fine-tuning of language models yet; SAM adaptation and DINOv2-style pretraining play the role
that SAM 2/3 and DINOv3 play a year later; registration and shape representation are relatively
more prominent, and several papers question common assumptions (scale, augmentations, active
learning, medical fine-tuning of SAM).

## Topic groups

Each group lists a few representative papers, not all of them.

### Segmentation

**The ideas behind this group** (a synthesis from the abstracts of about 31 papers in this group; the individual papers have not been read in full).

*The shared problem is that a segmentation model is expensive to supervise and fragile once deployed.* Voxel labels cost expert time, small or under-represented structures are segmented poorly, and models degrade when scanner, sequence or site changes. The papers fall into five threads, and a few of them question whether the usual fixes work at all.

*The first thread gets more from fewer, weaker or noisier labels.* Semi-supervised methods pair two branches and keep them from collapsing onto the same features ([Dual Mask Ensemble](https://proceedings.mlr.press/v301/nguyen26a.html)) or pretrain on unlabelled data first, with a student–teacher model on an extended long short-term memory (xLSTM) backbone ([3D-SegSync](https://proceedings.mlr.press/v301/mazher26a.html)) or a masked denoising autoencoder for laparoscopic surgery ([SurgicalSemiSeg](https://proceedings.mlr.press/v301/zhou26a.html)). Weak supervision comes as scribbles regularised by an anatomy distribution prior ([MedCL](https://proceedings.mlr.press/v301/zhang26c.html)), image-level labels turned into pixel classifiers ([PixelCAM](https://proceedings.mlr.press/v301/guichemerre26a.html)), labels from lay annotators reweighted by a consensus matrix ([CASC-AI](https://proceedings.mlr.press/v301/deng26a.html)), no labels at all with rotation-equivariant kernels ([equivariant biomarkers](https://proceedings.mlr.press/v301/chen26a.html)), and automatic T1-based parcellations used as targets for FLAIR-only models ([FLAIRBrainSeg](https://proceedings.mlr.press/v301/edern26a.html)). Two papers question the budget itself. One finds that annotating fewer slices in more volumes is almost always better, and that unsupervised active learning does not beat random or fixed-interval selection ([slice selection](https://proceedings.mlr.press/v301/zhang26a.html)). The other prunes training sets without any training ([PRIME](https://proceedings.mlr.press/v301/rahman26a.html)).

*The second thread adapts promptable foundation segmenters.* Segment Anything Model (SAM) is retrained for nuclei ([PathoSAM](https://proceedings.mlr.press/v301/griebel26a.html)), tuned with parameter-efficient layers whose placement matters more than their type ([PEFT for SAM](https://proceedings.mlr.press/v301/teuber26a.html)), and benchmarked zero-shot on bone CT, where general SAM and SAM2 beat medically fine-tuned variants ([SAM-family bone](https://proceedings.mlr.press/v301/magg26a.html)). The prompt itself is redesigned: a memory of past clicks ([MAIS](https://proceedings.mlr.press/v301/orbes26a.html)), or a canonical mesh registered to the image as a shape prior ([mesh prompts](https://proceedings.mlr.press/v301/su26a.html)). A ControlNet-style diffusion transformer is pretrained for transfer ([MedDelinea](https://proceedings.mlr.press/v301/deshmukh26a.html)).

*The third thread changes the architecture for efficiency or anatomy.* State-space models cut training memory ([SegResMamba](https://proceedings.mlr.press/v301/das26a.html)), two decoder heads replace three for nuclei and the model is deployed in eight hospitals ([DualU-Net](https://proceedings.mlr.press/v301/anglada-rotger26a.html)), and binary self-attention targets long sequences ([BinaryFormer](https://proceedings.mlr.press/v301/heinrich26b.html)). Anatomy enters through anisotropic stages for MRI ([RCSegNeXt](https://proceedings.mlr.press/v301/bo26a.html)), organ-specific decoders for small organs ([disjoint learning](https://proceedings.mlr.press/v301/tomar26a.html)), reformatting into cardiac planes before fine segmentation ([CCTA planes](https://proceedings.mlr.press/v301/vigneault26a.html)), auxiliary zonal segmentation for prostate cancer detection ([U-Mamba MTL](https://proceedings.mlr.press/v301/larsen26a.html)), and a diffusion segmenter for lumbar spine ([SpineSegDiff](https://proceedings.mlr.press/v301/monzon26a.html)).

*The fourth thread targets distribution shift without naming it in advance.* MixUp and auxiliary Fourier augmentation are added to nnU-Net ([augmentations for the unknown](https://proceedings.mlr.press/v301/vaish26a.html)), and a conditional diffusion model simulates other acquisition parameters while keeping anatomy fixed ([counterfactual acquisition](https://proceedings.mlr.press/v301/morao26a.html)). A nine-site breast MRI comparison finds standard nnU-Net sensitive but with too many false positives for clinical use ([breast MRI](https://proceedings.mlr.press/v301/geissler26a.html)).

*The fifth thread asks for a quality estimate with the mask.* Confidence predictors are trained on adversarial perturbations so that they generalise out of distribution ([adversarial confidence](https://proceedings.mlr.press/v301/lennartz26a.html)), and conformal prediction sets widen near critical interfaces such as vessels ([SACP](https://proceedings.mlr.press/v301/bereska26a.html)).

*Taken together*, the group spends most of its effort on cutting supervision and compute, with SAM-style models as a new default starting point. Still open: many results are on in-house or single datasets with few direct comparisons to nnU-Net; the slice-selection finding suggests that clever sampling may not beat simple rules, which few label-efficient papers test; and label efficiency and quality estimation are developed in separate papers rather than together.

**Representative papers**

- [How to select slices for annotation to train best-performing deep learning segmentation models for cross-sectional medical images?](https://proceedings.mlr.press/v301/zhang26a.html)
- [Zero-shot capability of 2D SAM-family models for bone segmentation in CT scans](https://proceedings.mlr.press/v301/magg26a.html)
- [Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging](https://proceedings.mlr.press/v301/teuber26a.html)
- [Mesh-Prompted Anatomy Segmentation](https://proceedings.mlr.press/v301/su26a.html)
- [MedCL: Learn Consistent Anatomy Distribution for Scribble-supervised Medical Image Segmentation](https://proceedings.mlr.press/v301/zhang26c.html)
- [Data-Agnostic Augmentations for Unknown Variations: Out-of-Distribution Generalisation in MRI Segmentation](https://proceedings.mlr.press/v301/vaish26a.html)
- [Adversarial Perturbations Improve Generalization of Confidence Prediction in Medical Image Segmentation](https://proceedings.mlr.press/v301/lennartz26a.html)
- [Two Heads Are Enough: DualU-Net, a Fast and Efficient Architecture for Nuclei Instance Segmentation](https://proceedings.mlr.press/v301/anglada-rotger26a.html)

### Foundation models and representation learning

**The ideas behind this group** (a synthesis from the abstracts of about 24 papers in this group; the individual papers have not been read in full).

*The shared problem is how to get general-purpose features for medical images when labels are scarce and natural-image recipes do not transfer cleanly.* The papers split between designing pretraining for medical data, building domain models at scale, adapting existing models cheaply, and making backbones efficient enough to train at all.

*The first thread argues that self-supervised recipes need changing for medical images.* Retinal images sit densely in latent space, so weaker augmentations help contrastive learning ([augmentation scales](https://proceedings.mlr.press/v301/cheng26a.html)). Random masking leaks information between similar neighbouring patches, so masks follow attention, with a noisy teacher keeping attention heads diverse ([DAGMaN](https://proceedings.mlr.press/v301/jiang26a.html)). For laparoscopy, pixel-level objectives work best ([SurgicalSemiSeg](https://proceedings.mlr.press/v301/zhou26a.html)). In video, masks follow optical flow so that the model attends to tool–tissue interaction ([surgical flow MAE](https://proceedings.mlr.press/v301/mostafa26a.html)), and a Joint-Embedding Predictive Architecture (JEPA) predicts latent features instead of pixels for cataract surgery ([JHU-VPT](https://proceedings.mlr.press/v301/shah26b.html)). Adapting DINOv2 to chest X-rays, registers, teacher centring and head count matter, and the larger model performs about the same as the smaller one ([CXformer](https://proceedings.mlr.press/v301/al-mahrooqi26a.html)).

*The second thread builds domain models and reuses them as supervision.* A multi-scale foundation model pretrained on single-time-point prostate MRI from 10 institutions supports a small longitudinal active-surveillance cohort ([prostate progression](https://proceedings.mlr.press/v301/wang26b.html)). Large 2D and 3D autoencoders compress images into latents that downstream models train on faster ([MedVAE](https://proceedings.mlr.press/v301/varma26a.html)). Universal segmentation models supply "free" anatomical masks as an auxiliary training signal for classifiers ([scaling supervision](https://proceedings.mlr.press/v301/li26d.html)), and a student–teacher volumetric pretraining feeds segmentation ([3D-SegSync](https://proceedings.mlr.press/v301/mazher26a.html)).

*The third thread adapts existing foundation models rather than training new ones.* For SAM, where parameter-efficient layers go matters more than which kind ([PEFT for SAM](https://proceedings.mlr.press/v301/teuber26a.html)); retraining gives a nucleus model ([PathoSAM](https://proceedings.mlr.press/v301/griebel26a.html)); zero-shot prompting is benchmarked on bone ([SAM-family bone](https://proceedings.mlr.press/v301/magg26a.html)); few-shot and in-domain pretraining are compared for orthodontic staging ([cervical maturation](https://proceedings.mlr.press/v301/schneider26a.html)). Frozen features are combined with learned constraints so that multi-label blood-cell predictions stay biologically possible ([DinoBloom constraints](https://proceedings.mlr.press/v301/mouadden26a.html)), or with nearest-neighbour distances for cellular anomaly detection ([toxicological anomalies](https://proceedings.mlr.press/v301/juturu26a.html)). Ensembling several pathology foundation models improves out-of-distribution performance ([FM ensemble](https://proceedings.mlr.press/v301/chung26a.html)). Two papers warn of costs: contrastive text alignment drives classification accuracy towards zero unless the objectives are balanced ([classification–retrieval trade-off](https://proceedings.mlr.press/v301/lefkes26a.html)), and zero-shot results vary widely with prompt wording, which embedding ensembles reduce ([MedImageInsight deployment](https://proceedings.mlr.press/v301/fay26a.html)).

*The fourth thread makes backbones cheaper.* Options include state-space segmenters ([SegResMamba](https://proceedings.mlr.press/v301/das26a.html)), 1-bit self-attention for training and inference ([BinaryFormer](https://proceedings.mlr.press/v301/heinrich26b.html)), a convolutional neural network (CNN) feeding hierarchical tokens to a transformer ([DuoFormer](https://proceedings.mlr.press/v301/tang26a.html)), and global–local attention that mimics radiologists scrolling through CT without heavy pretraining ([CT-Scroll](https://proceedings.mlr.press/v301/di-piazza26a.html)).

*Taken together*, the group treats foundation models less as finished products and more as components to retune, ensemble, constrain or reuse as label generators. Still open: evidence on scaling is thin and partly negative (CXformer's size result); the prompt-sensitivity and ensemble findings suggest that a single foundation representation is not yet reliable across sites; and few papers compare pretraining recipes on equal compute.

**Representative papers**

- [Empirical Analysis of Scaling Vision Foundation Models for Chest X-rays](https://proceedings.mlr.press/v301/al-mahrooqi26a.html)
- [Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales](https://proceedings.mlr.press/v301/cheng26a.html)
- [Co-distilled attention guided masked image modeling with noisy teacher for self-supervised learning on medical images](https://proceedings.mlr.press/v301/jiang26a.html)
- [A Vision Foundation Model for Cataract Surgery Using Joint-Embedding Predictive Architecture](https://proceedings.mlr.press/v301/shah26b.html)
- [MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders](https://proceedings.mlr.press/v301/varma26a.html)
- [Scaling Supervision for Free: Leveraging Universal Segmentation Models for Enhanced Medical Image Diagnosis](https://proceedings.mlr.press/v301/li26d.html)
- [A Balancing Act: Optimizing Classification and Retrieval in Cross-Modal Vision Models](https://proceedings.mlr.press/v301/lefkes26a.html)
- [Foundation Model Ensemble for Out-of-Distribution Generalization: Predicting Lymph Node Metastasis in Early Gastric Cancer Using Whole-Slide Imaging](https://proceedings.mlr.press/v301/chung26a.html)

### Vision-language models and reports

**The ideas behind this group** (a synthesis from the abstracts of about 14 papers in this group; the individual papers have not been read in full).

*The shared problem is that hospitals hold vast amounts of text paired with images, but the text is hard to use well.* Almost every scan or slide comes with a report. Those reports are free text written for clinicians, and they are rarely tied to image regions. The papers here use language in three roles. It is an output (reports and answers), an interface (prompts that steer a pretrained vision–language model, VLM) and a cheap source of labels for conventional image models.

*The first thread generates reports or answers and tries to make them clinically correct, not just fluent.* [RaDialog](https://proceedings.mlr.press/v301/pellegrini26a.html) adds a secondary image branch for structured clinical findings and an instruction dataset of dialogue tasks. One large VLM can then write chest X-ray reports, correct them interactively and answer follow-up questions. Several papers make comparison explicit. [DiffRGenNet](https://proceedings.mlr.press/v301/bian26a.html) retrieves similar reports and models how the query image differs from them, aiming at rare diseases. A [difference VQA model](https://proceedings.mlr.press/v301/marhuenda26a.html) answers questions about changes between two chest X-rays. For whole slides, [PathGenIC](https://proceedings.mlr.press/v301/liu26a.html) retrieves similar slide–report pairs as in-context examples. Retrieving neighbours as context is a recurring device. Evaluation, however, often still rests on text-overlap metrics such as BLEU and ROUGE.

*The second thread steers frozen or lightly tuned VLMs through prompts.* [Visual prompt engineering](https://proceedings.mlr.press/v301/denner26a.html) draws arrows, boxes and circles onto chest X-rays before zero-shot classification with Contrastive Language–Image Pretraining (CLIP). The model's attention moves towards the marked region. On the text side, [deploying MedImageInsight](https://proceedings.mlr.press/v301/fay26a.html) finds that zero-shot pneumonia detection varies widely with prompt wording. It also finds that ensembles of text and image embeddings beat fine-tuning and linear probes with little data, for both adults and children. [Curriculum learning for Grounding DINO](https://proceedings.mlr.press/v301/heidrich26a.html) applies language-guided detection to tumours in MRI and CT. [Generate to Ground](https://proceedings.mlr.press/v301/nutzel26a.html) argues that a text-to-image diffusion model, fine-tuned with a frozen domain language model, grounds report phrases better than contrastive models do, through its cross-attention maps.

*The third thread treats reports as a source of labels.* [Chest-OMDL](https://proceedings.mlr.press/v301/bai26a.html) trains organ-level disease detection in 3D chest CT from two sources: labels extracted by RadBERT and organ masks from a text-prompted segmentation model. [LLM-based extraction](https://proceedings.mlr.press/v301/vasylechko26a.html) pulls fracture type, location and severity from paediatric wrist reports to fine-tune a detector without expert boxes. [Surgical Aggregation](https://proceedings.mlr.press/v301/shah26a.html) trains a report labeller by federated learning, even though sites label different, partly overlapping sets of diseases.

*The fourth thread asks what alignment with text costs, and what else language can do.* [A balancing act](https://proceedings.mlr.press/v301/lefkes26a.html) shows that contrastive tuning for cross-modal retrieval drives a strong classifier's accuracy towards zero. Keeping both objectives in the loss avoids this, both on natural images and on prostate grading in whole slides. [PRISM](https://proceedings.mlr.press/v301/kumar26a.html) uses language-guided Stable Diffusion to add or remove devices and disease features, producing counterfactuals for training more robust classifiers. [Patient-level contrastive pretraining](https://proceedings.mlr.press/v301/li26c.html) aligns longitudinal CT with clinical variables encoded by a text model, not with reports.

*Taken together*, language earns its place here mostly as structure and supervision: structured findings, retrieved neighbours, prompts and extracted labels. Free generation alone is rarely enough. Still open: metrics that measure clinical correctness, robustness to small changes in wording, the trade-off between alignment and discrimination, and reasoning over 3D volumes and change over time.

**Representative papers**

- [RaDialog: Large Vision-Language Models for X-Ray Reporting and Dialog-Driven Assistance](https://proceedings.mlr.press/v301/pellegrini26a.html)
- [Generate to Ground: Multimodal Text Conditioning Boosts Phrase Grounding in Medical Vision-Language Models](https://proceedings.mlr.press/v301/nutzel26a.html)
- [Beyond the Prompt: Deploying Medical Foundation Models on Diverse Chest X-ray Populations](https://proceedings.mlr.press/v301/fay26a.html)
- [Visual Prompt Engineering for Vision Language Models in Radiology](https://proceedings.mlr.press/v301/denner26a.html)
- [A Balancing Act: Optimizing Classification and Retrieval in Cross-Modal Vision Models](https://proceedings.mlr.press/v301/lefkes26a.html)
- [Chest-OMDL: Organ-specific Multidisease Detection and Localization in Chest Computed Tomography using Weakly Supervised Deep Learning from Free-text Radiology Report](https://proceedings.mlr.press/v301/bai26a.html)
- [Enhancing Wrist Fracture Detection through LLM-Powered Data Extraction and Knowledge-Based Ensemble Learning](https://proceedings.mlr.press/v301/vasylechko26a.html)

### Computational pathology and microscopy

**The ideas behind this group** (a synthesis from the abstracts of about 17 papers in this group; the individual papers have not been read in full).

*The shared problem is that cells and tissue are abundant, but reliable labels and consistent appearance are not.* Pathology and microscopy images contain thousands of nuclei per field. Expert annotation is slow, staining and scanners vary between sites, and the biology imposes relationships that pixel classifiers ignore. Unlike work dominated by slide-level multiple instance learning, most papers in this volume work at the level of cells and regions.

*The first thread makes cell-level segmentation and counting faster, more general or more legible.* [DualU-Net](https://proceedings.mlr.press/v301/anglada-rotger26a.html) cuts HoVer-Net's three decoders to two, one for segmentation and one for centroid density. It reports faster inference and has been deployed across eight hospitals. [PathoSAM](https://proceedings.mlr.press/v301/griebel26a.html) trains the Segment Anything Model (SAM) on diverse nucleus data for automatic and interactive segmentation. A study of [parameter-efficient fine-tuning of SAM](https://proceedings.mlr.press/v301/teuber26a.html) finds that where the adapter layers go matters more than which type is used. [CountXplain](https://proceedings.mlr.press/v301/mohammed26a.html) adds prototypes to density-map counting, and biologists judged the learned patterns relevant.

*The second thread learns from weak, noisy or absent labels.* [CASC-AI](https://proceedings.mlr.press/v301/deng26a.html) trains on lay annotations. It weights regions by agreement between model and annotators, then corrects disagreements by similarity to the agreed regions. [PixelCAM](https://proceedings.mlr.press/v301/guichemerre26a.html) trains classification and pixel-level localisation together from image-level labels. [Cellular anomaly detection](https://proceedings.mlr.press/v301/juturu26a.html) in toxicology finds that a simple nearest-neighbour distance works well once features come from an encoder pretrained on histopathology. [Equivariant biomarkers](https://proceedings.mlr.press/v301/chen26a.html) use symmetric kernels in unsupervised segmentation, so features do not depend on orientation.

*The third thread tackles stain and site variation.* [Foundation-model ensembles](https://proceedings.mlr.press/v301/chung26a.html) combine pathology encoders trained on different data, and improve lymph node metastasis prediction in and out of distribution. [SAIC](https://proceedings.mlr.press/v301/qi26a.html) composes abnormal cells into backgrounds in matching styles for cytology detection, using a large VLM to filter what it synthesises. DualU-Net and PixelCAM also report robustness to stain variation or domain shift.

*The fourth thread puts biological structure into the model.* [HIEGNet](https://proceedings.mlr.press/v301/kormann26a.html) builds a heterogeneous graph of glomeruli and the immune cells around them. A [biologically constrained classifier](https://proceedings.mlr.press/v301/mouadden26a.html) adds learnable constraint matrices to a haematology foundation model to rule out contradictory neutrophil labels. [MagNet](https://proceedings.mlr.press/v301/zhu26b.html) predicts spatial transcriptomics at 8 µm resolution from multi-resolution image patches.

*A fifth thread replaces a physical measurement with a predicted one.* [Virtual multiplex staining](https://proceedings.mlr.press/v301/arbel26a.html) for CD68 raised pathologists' agreement and accuracy in identifying macrophages on PD-L1 slides. [Sequence models](https://proceedings.mlr.press/v301/leger26a.html) predict continuous cell-cycle reporter signals from label-free brightfield time-lapse images, and outperform single-frame models.

*Taken together*, the group favours efficient, deployable cell-level tools and borrows priors from biology or pretrained encoders in place of dense labels. Language appears only at the edges, in [slide report generation](https://proceedings.mlr.press/v301/liu26a.html) and [retrieval](https://proceedings.mlr.press/v301/lefkes26a.html). Still open: prospective validation across many sites, whether prototypes and constraints match what the model actually uses, and when a virtual stain is reliable enough to replace a physical one.

**Representative papers**

- [Two Heads Are Enough: DualU-Net, a Fast and Efficient Architecture for Nuclei Instance Segmentation](https://proceedings.mlr.press/v301/anglada-rotger26a.html)
- [Segment Anything for Histopathology](https://proceedings.mlr.press/v301/griebel26a.html)
- [CASC-AI: Consensus-aware Self-corrective Learning for Cell Segmentation with Noisy Labels](https://proceedings.mlr.press/v301/deng26a.html)
- [Foundation Model Ensemble for Out-of-Distribution Generalization: Predicting Lymph Node Metastasis in Early Gastric Cancer Using Whole-Slide Imaging](https://proceedings.mlr.press/v301/chung26a.html)
- [HIEGNet: A Heterogenous Graph Neural Network Including the Immune Environment in Glomeruli Classification](https://proceedings.mlr.press/v301/kormann26a.html)
- [Evaluation of Virtual Stain Multiplexed CD68 for Macrophage Detection in NSCLC PD-L1 Slides](https://proceedings.mlr.press/v301/arbel26a.html)
- [MagNet: Multi-Level Attention Graph Network for Predicting High-Resolution Spatial Transcriptomics](https://proceedings.mlr.press/v301/zhu26b.html)

### Generative models and synthesis

**The ideas behind this group** (a synthesis from the abstracts of about 24 papers in this group; the individual papers have not been read in full).

*The shared problem is producing useful images where real ones are scarce, inconsistent across sites or cannot be shared, without the generator quietly changing anatomy.* Diffusion models are the default tool. Generative adversarial networks (GANs), vector-quantised autoencoders and composition methods that need no training also appear. The papers use them to make missing data, to translate between domains, to generate shapes rather than images, and to probe other models. A smaller set checks the generators themselves.

*The first thread creates data or modalities that are hard to acquire.* Synthetic pairs of depth images and radiographs computed from CT are used to pretrain a patient-pose assessor ([pose from CT](https://proceedings.mlr.press/v301/laufer26a.html)). Abnormal cells are taken from a bank and composited into backgrounds of matching stain style, then filtered by a vision–language model, to help rare categories ([SAIC](https://proceedings.mlr.press/v301/qi26a.html)). Counterfactual MRI simulating different acquisition parameters, with anatomy unchanged, serves as augmentation for out-of-distribution segmentation ([acquisition counterfactuals](https://proceedings.mlr.press/v301/morao26a.html)). Whole modalities are synthesised too. PET is generated from MRI, demographics and cerebrospinal-fluid biomarkers using the causal relations among them ([Causal PETS](https://proceedings.mlr.press/v301/li26a.html)). A "world model" evolves VQ-GAN latents with a neural ordinary differential equation to produce fibrotic lung CT at any time point, validated by survival analysis on the generated scans ([4D-VQ-GAN](https://proceedings.mlr.press/v301/zhao26a.html)). Virtual stain multiplexing improves pathologists' agreement on macrophage detection ([virtual CD68](https://proceedings.mlr.press/v301/arbel26a.html)). Completion methods fill in missing surgical scene context ([SSDD-GAN](https://proceedings.mlr.press/v301/zhang26b.html)) or outpaint sector-shaped 3D ultrasound so that stitching is easier ([LOTUS](https://proceedings.mlr.press/v301/yao26a.html)).

*The second thread translates and harmonises, with explicit guards against hallucination.* Lung CT reconstruction kernels are harmonised by a multi-path CycleGAN regularised with anatomical labels and a tissue-statistic loss ([kernel harmonisation](https://proceedings.mlr.press/v301/krishnan26a.html)). A single lightweight generator guided by a pretrained CLIP model replaces one generator–discriminator pair per mapping ([I2I-Galip](https://proceedings.mlr.press/v301/korkmaz26a.html)). The idea of "style" also extends beyond scanners: differences between functional MRI (fMRI) analysis pipelines are treated as style and transferred between statistic maps ([pipeline style transfer](https://proceedings.mlr.press/v301/germani26a.html)).

*The third thread generates geometry rather than intensities.* Examples are a diffusion model that keeps point correspondences across hippocampal shapes ([correspondence-preserving diffusion](https://proceedings.mlr.press/v301/zhu26a.html)), latent diffusion over semantic signed distance fields for cerebral vessel trees ([vessel-tree SDF](https://proceedings.mlr.press/v301/kuipers26a.html)), and conditional diffusion on the sphere for cortical normative modelling ([surface diffusion](https://proceedings.mlr.press/v301/zhang26d.html)).

*The fourth thread turns generators into instruments, and some papers check the generators themselves.* Class-conditional diffusion models act as classifiers, with explanations and uncertainty as by-products ([diffusion classifiers](https://proceedings.mlr.press/v301/favero26a.html)). Diffusion also drives segmentation ([SpineSegDiff](https://proceedings.mlr.press/v301/monzon26a.html), [MedDelinea](https://proceedings.mlr.press/v301/deshmukh26a.html)). Counterfactuals edit devices and disease features through language ([PRISM](https://proceedings.mlr.press/v301/kumar26a.html)) or remove protected attributes to measure how much a classifier relies on them as shortcuts ([shortcut counterfactuals](https://proceedings.mlr.press/v301/vigneshwaran26a.html)). On the checking side, planted artificial fingerprints test both memorisation and neglect of rare features ([privacy–fairness](https://proceedings.mlr.press/v301/dombrowski26a.html)). A knowledge-based detector flags anatomically implausible synthetic mammograms, supported by a reader study ([shape artefacts](https://proceedings.mlr.press/v301/deshpande26a.html)). Supporting infrastructure includes large autoencoders whose latents stand in for high-resolution images ([MedVAE](https://proceedings.mlr.press/v301/varma26a.html)) and binary self-attention that makes 3D diffusion cheaper ([BinaryFormer](https://proceedings.mlr.press/v301/heinrich26b.html)).

*Taken together*, generators are increasingly conditioned on anatomy, acquisition parameters or causal structure, and are judged by downstream use rather than by appearance. Still open: anatomical fidelity is enforced mostly through loss terms rather than verified. Only a couple of papers test whether synthetic images are plausible or leak training data. Usefulness is usually shown on the authors' own downstream task, so results are hard to compare.

**Representative papers**

- [Can Diffusion Models Generalize? Privacy and Fairness Trade-offs for Medical Data Sharing.](https://proceedings.mlr.press/v301/dombrowski26a.html)
- [Data Augmentation for Medical Imaging: Counterfactual Simulation of Acquisition Parameters via Conditional Diffusion Model](https://proceedings.mlr.press/v301/morao26a.html)
- [Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free](https://proceedings.mlr.press/v301/favero26a.html)
- [4D-VQ-GAN: A World Model for Synthesizing Medical Scans at Any Time Point for Personalized Disease Progression Modeling of Idiopathic Pulmonary Fibrosis](https://proceedings.mlr.press/v301/zhao26a.html)
- [Anatomy-Guided Multi-Path CycleGAN for Lung CT Kernel Harmonization](https://proceedings.mlr.press/v301/krishnan26a.html)
- [A knowledge-based method for detecting network-induced shape artifacts in synthetic images](https://proceedings.mlr.press/v301/deshpande26a.html)
- [PRISM: High-Resolution & Precise Counterfactual Medical Image Generation using Language-guided Stable Diffusion](https://proceedings.mlr.press/v301/kumar26a.html)

### Reconstruction and super-resolution

**The ideas behind this group** (a synthesis from the abstracts of about 7 papers in this group; the individual papers have not been read in full).

*The shared problem is recovering a clinically usable image or signal from fewer or poorer measurements.* Examples are undersampled k-space, a low field strength or a handful of time points, often with no clean ground truth to train against. The group is small, and its main question is less which architecture to use than where the supervision comes from.

*The first thread trains reconstruction without fully sampled targets.* Joint supervised and self-supervised learning trains on subsampled target data in a self-supervised way and on a proxy dataset with fully sampled k-space in a supervised way at the same time. It also offers rule-of-thumb guidelines for training ([JSSL](https://proceedings.mlr.press/v301/yiasemis26a.html)). Deep image prior (DIP) methods need no training data at all, and one study compares four DIP variants for accelerated isotropic 3D late gadolinium enhancement MRI of the left atrium. It reports that the self-guided variant beats compressed sensing in 10 subjects ([DIP family](https://proceedings.mlr.press/v301/hisham26a.html)). For super-resolution, a score-based diffusion model trained only on high-resolution MRI is steered by data consistency with the low-resolution scan ([SRMRI](https://proceedings.mlr.press/v301/poudel26a.html)).

*The second thread questions simulated training pairs.* The same super-resolution paper argues that low-resolution images simulated with fixed kernels bias supervised models. It concedes that its unsupervised method does not beat supervised baselines on peak signal-to-noise ratio (PSNR) or structural similarity (SSIM), and it releases low- and high-resolution pairs acquired on a scanner ([SRMRI](https://proceedings.mlr.press/v301/poudel26a.html)). This is one of the few attempts in the group at real rather than simulated degradation.

*The third thread is about access as much as accuracy.* Enhancing paediatric ultra-low-field MRI is framed as a way to bring neurodevelopmental imaging to low- and middle-income settings. The model is a hybrid of a convolutional network and a state-space model, with a 3D-to-1D serialisation (the "generalised Hilbert" scan of the title) that captures long-range context without losing local precision ([GAMBAS](https://proceedings.mlr.press/v301/baljer26a.html)).

*The fourth thread reconstructs signals in time, not images in space.* Conditional neural fields recover full tissue attenuation curves from 4 of the 30 time points of CT perfusion. This could lower radiation dose and motion sensitivity, and it enables perfusion analysis from multiphase CT angiography ([STAR](https://proceedings.mlr.press/v301/vries26a.html)). Two neighbouring papers deal with acquisition quality rather than reconstruction itself. One scores MR artefact severity without labels, using contrastive learning on simulated bias, noise, anisotropy and ghosting ([artefact scoring](https://proceedings.mlr.press/v301/hays26a.html)). The other harmonises CT reconstruction kernels so that emphysema measurements agree ([kernel harmonisation](https://proceedings.mlr.press/v301/krishnan26a.html)).

*Taken together*, the papers spend more effort on the supervision gap than on network design: proxy datasets, untrained priors, unconditional generative priors with data consistency, and scanner-acquired pairs. Still open: cohorts are small, and the DIP study has 10 subjects. Evaluation still rests on PSNR and SSIM, which one paper explicitly doubts, and downstream diagnostic value is rarely measured. Compared with the larger reconstruction literature, joint design of the acquisition, explicit physics models and uncertainty are largely absent here.

**Representative papers**

- [Joint Supervised and Self-supervised Learning for MRI Reconstruction](https://proceedings.mlr.press/v301/yiasemis26a.html)
- [SRMRI: A Diffusion-Based Super-resolution Framework and Open Dataset for Blind MRI Super-Resolution](https://proceedings.mlr.press/v301/poudel26a.html)
- [GAMBAS: Generalised-Hilbert Mamba for Super-resolution of Paediatric Ultra-Low-Field MRI](https://proceedings.mlr.press/v301/baljer26a.html)
- [Family of Deep Image Prior Networks for Accelerated 3D LGE-MRI Acquisition with Enhanced Reconstruction](https://proceedings.mlr.press/v301/hisham26a.html)
- [Neural fields for tissue attenuation curve reconstruction in sparsely sampled time-resolved CT](https://proceedings.mlr.press/v301/vries26a.html)
- [An Unsupervised Approach for Artifact Severity Scoring in Multi-Contrast MR Images](https://proceedings.mlr.press/v301/hays26a.html)

### Registration, shape and neural fields

**The ideas behind this group** (a synthesis from the abstracts of about 13 papers in this group; the individual papers have not been read in full).

*The shared problem is describing anatomy by its geometry: how one image maps onto another, how a structure changes over time, and how a shape or vessel tree can be encoded so that it can be compared, generated or segmented.* The papers swap the usual dense voxel grid for other representations: forward-splatted point densities, log-domain deformations, coordinate networks, meshes, point sets with correspondence and tree-structured codes.

*The first thread revisits how deformable registration is formulated and regularised.* For brain registration with large deformations, a symmetric multi-level network with gradient-inverse consistency removes the need for pre-registration ([SM-GICNet](https://proceedings.mlr.press/v301/bai26b.html)). A more radical change replaces backward warping with forward splatting. Volumes are rasterised from intermediate point clouds, for example from vessel filters, and registration becomes density alignment ([forward splatting](https://proceedings.mlr.press/v301/heinrich26a.html)). Regularisation can also come from the population. A log-Euclidean autoencoder learns a linear latent space of deformations that respects inverse consistency and steers registration networks towards anatomically plausible transforms ([MORPH-LER](https://proceedings.mlr.press/v301/karanam26a.html)).

*The second thread treats deformation and signal as continuous functions of time.* An implicit neural representation (INR) takes space and time as input and gives longitudinal brain deformation at observed and unobserved time points. Its analytic derivatives supply a regulariser that enforces a monotonic rate of change ([temporal neural displacement fields](https://proceedings.mlr.press/v301/shuaibu26a.html)). Conditional neural fields similarly fill in tissue attenuation curves from sparse CT time points ([STAR](https://proceedings.mlr.press/v301/vries26a.html)).

*The third thread uses alignment as a means to another end.* Mesh-prompted segmentation registers the signed distance function of a canonical mesh to the target image, one deformation per structure. This gives a stronger shape prior than point prompts and preserves surface correspondence ([mesh prompting](https://proceedings.mlr.press/v301/su26a.html)). High-resolution 2D fetal echocardiography is placed in a 3D reference space, trained on slices simulated from 3D volumes ([FERN](https://proceedings.mlr.press/v301/gilliland26a.html)). Cardiac CT angiography is resampled into predicted standard planes before the left ventricle is segmented ([CCTA planes](https://proceedings.mlr.press/v301/vigneault26a.html)). Generative outpainting reshapes the sector-shaped field of view of 3D ultrasound so that stitching does not get stuck in local minima ([LOTUS](https://proceedings.mlr.press/v301/yao26a.html)).

*The fourth thread builds representations of shape itself.* Vessel trees become single vectors through two transformer autoencoders, one for segments and one for topology, with recursive decoding that guarantees a valid tree ([VeTTA](https://proceedings.mlr.press/v301/batten26a.html)). They can also be generated as semantic signed distance fields, which avoids a separate topology-reconstruction step ([vessel-tree SDF](https://proceedings.mlr.press/v301/kuipers26a.html)). Point-based hippocampal shapes are generated with their correspondences preserved, which allows counterfactual disease progression ([correspondence-preserving diffusion](https://proceedings.mlr.press/v301/zhu26a.html)). Cortical feature maps are generated on the sphere, conditioned on each subject's segmentation, to reduce the mismatch in folding between subjects ([surface diffusion](https://proceedings.mlr.press/v301/zhang26d.html)).

*Taken together*, the common move is to put geometric structure into the representation: inverse consistency, diffeomorphic log maps, analytic derivatives, correspondence and tree validity. Leaving it to the loss is the alternative they avoid. Still open: the abstracts give little evidence on when splatting, neural fields or learned population priors beat well-tuned dense methods across tasks. Shape generators are judged mainly by realism and interpolation. Several pipelines are trained on simulated slices or synthetic trees, and transfer to messy clinical data is only partly tested.

**Representative papers**

- [Fast forward: Rephrasing 3D deformable image registration through density alignment and splatting](https://proceedings.mlr.press/v301/heinrich26a.html)
- [MORPH-LER: Log-Euclidean Regularization for Population-Aware Image Registration](https://proceedings.mlr.press/v301/karanam26a.html)
- [Capturing Longitudinal Changes in Brain Morphology Using Temporally Parameterized Neural Displacement Fields.](https://proceedings.mlr.press/v301/shuaibu26a.html)
- [Mesh-Prompted Anatomy Segmentation](https://proceedings.mlr.press/v301/su26a.html)
- [Vector Representations of Vessel Trees](https://proceedings.mlr.press/v301/batten26a.html)
- [Point-Based Shape Representation Generation with a Correspondence-Preserving Diffusion Model](https://proceedings.mlr.press/v301/zhu26a.html)
- [FERN: A Fetal Echocardiography Registration Network for 2D-to-3D Alignment](https://proceedings.mlr.press/v301/gilliland26a.html)

### Trustworthiness

**The ideas behind this group** (a synthesis from the abstracts of about 19 papers in this group; the individual papers have not been read in full).

*The shared problem is that a correct-looking output does not show whether the model is confident for the right reasons, fair across groups, or safe with patient data.* The papers cover four properties: calibrated confidence, built-in explanation, freedom from shortcuts, and fairness and privacy. Only a few test a property against an external reference.

*The first thread wants confidence with a guarantee attached.* Conformal prediction is the common tool. For regression, predicting a Gaussian variance and then rescaling it conformally gives the tightest intervals with coverage ([instance-dependent intervals](https://proceedings.mlr.press/v301/nizhar26a.html)). For segmentation, nonconformity scores are weighted by distance to critical interfaces, so prediction sets widen where errors matter ([SACP](https://proceedings.mlr.press/v301/bereska26a.html)). Learned quality predictors are made robust by training on adversarial images that degrade the mask ([adversarial confidence](https://proceedings.mlr.press/v301/lennartz26a.html)). Generative models offer uncertainty as a by-product: class-conditional diffusion classifiers ([diffusion classifiers](https://proceedings.mlr.press/v301/favero26a.html)) and diffusion segmentation of the spine ([SpineSegDiff](https://proceedings.mlr.press/v301/monzon26a.html)).

*The second thread prefers explanations that are part of the model.* Prototype layers in a density-estimation network show which visual patterns are counted as cells, and biologists judged them relevant in a survey ([CountXplain](https://proceedings.mlr.press/v301/mohammed26a.html)). A pixel classifier trained alongside the image classifier gives localisation ([PixelCAM](https://proceedings.mlr.press/v301/guichemerre26a.html)), and diffusion classifiers are described as intrinsically explainable ([diffusion classifiers](https://proceedings.mlr.press/v301/favero26a.html)). For post hoc attribution, one paper defines feature importance as the fraction of explained variance and validates it on a synthetic dataset with known ground truth ([variance decomposition](https://proceedings.mlr.press/v301/jin26a.html)). Visual markers drawn on the image steer a vision–language model's attention to the pathology ([visual prompts](https://proceedings.mlr.press/v301/denner26a.html)).

*The third thread measures shortcuts and implausible outputs instead of assuming them.* A causal generative model removes protected attributes from a classifier's activations: removing site lowers Parkinson's disease AUROC from 0.74 to 0.65, whereas removing sex barely changes it ([shortcut counterfactuals](https://proceedings.mlr.press/v301/vigneshwaran26a.html)). Language-guided counterfactuals remove medical devices to train more robust classifiers ([PRISM](https://proceedings.mlr.press/v301/kumar26a.html)). Other papers constrain outputs to what is possible: rotation-equivariant kernels for orientation-free histology ([equivariant biomarkers](https://proceedings.mlr.press/v301/chen26a.html)), learned constraint matrices against biologically impossible label combinations ([DinoBloom constraints](https://proceedings.mlr.press/v301/mouadden26a.html)), and a knowledge-based detector of anatomically unrealistic synthetic images, checked in a reader study ([shape artefacts](https://proceedings.mlr.press/v301/deshpande26a.html)).

*The fourth thread links fairness and privacy.* Planted fingerprints in training data test a diffusion model both ways: reproduced verbatim is a privacy leak, ignored is a fairness failure ([privacy–fairness trade-off](https://proceedings.mlr.press/v301/dombrowski26a.html)). Prompt and embedding ensembles narrow subgroup gaps by sex, age and race in zero-shot chest X-ray ([MedImageInsight deployment](https://proceedings.mlr.press/v301/fay26a.html)). Federated learning keeps data local but struggles with heterogeneous ordinal labels ([federated ordinal](https://proceedings.mlr.press/v301/corbetta26a.html)), and is adapted to clients with different label sets ([surgical aggregation](https://proceedings.mlr.press/v301/shah26a.html)).

*Taken together*, the stronger papers attach a test to the property: a coverage guarantee, a known ground truth, a reader study, or a counterfactual that removes one attribute. Still open: conformal guarantees are marginal and say little about subgroups; the shortcut and fairness analyses each examine one attribute or one dataset; and no paper here checks whether calibrated or explained outputs change clinical decisions.

**Representative papers**

- [SACP: Spatially-Adaptive Conformal Prediction in Uncertainty Quantification of Medical Image Segmentation](https://proceedings.mlr.press/v301/bereska26a.html)
- [Clinical Measurements with Calibrated Instance-Dependent Confidence Interval](https://proceedings.mlr.press/v301/nizhar26a.html)
- [Evaluating Shortcut Utilization in Deep Learning Disease Classification through Counterfactual Analysis](https://proceedings.mlr.press/v301/vigneshwaran26a.html)
- [Can Diffusion Models Generalize? Privacy and Fairness Trade-offs for Medical Data Sharing.](https://proceedings.mlr.press/v301/dombrowski26a.html)
- [Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free](https://proceedings.mlr.press/v301/favero26a.html)
- [CountXplain: Interpretable Cell Counting with Prototype-Based Density Map Estimation](https://proceedings.mlr.press/v301/mohammed26a.html)
- [Feature Attribution for Deep Learning Models through Total Variance Decomposition](https://proceedings.mlr.press/v301/jin26a.html)
- [A knowledge-based method for detecting network-induced shape artifacts in synthetic images](https://proceedings.mlr.press/v301/deshpande26a.html)

### Longitudinal and multimodal clinical prediction

**The ideas behind this group** (a synthesis from the abstracts of about 15 papers in this group; the individual papers have not been read in full).

*The shared problem is that clinical decisions rest on more than one image at one time.* Clinicians combine scans with patient records and compare a scan with earlier ones. Models built on a single image throw this away. Paired multimodal or longitudinal data, however, is scarce, irregular and of uneven quality. The papers here ask how to fuse heterogeneous inputs, how to use earlier time points, and how to model a trajectory rather than a single state.

*The first thread fuses images with tabular data, and increasingly declines to treat modalities as equals.* [Total knee replacement timing](https://proceedings.mlr.press/v301/cigdem26a.html) fuses MRI with a self-supervised pretrained tabular encoder through cross-modal attention. It outperforms concatenation and other fusion schemes. [DAFTED](https://proceedings.mlr.press/v301/stym-popper26a.html) argues against symmetric fusion. It starts from a primary modality, echocardiographic time series, and adds tabular information after separating shared from modality-specific content. [DeFusion](https://proceedings.mlr.press/v301/ouyang26a.html) similarly splits related from unrelated information when combining time-lapse embryo images with fertility indicators to predict pregnancy. [Causal PETS](https://proceedings.mlr.press/v301/li26a.html) takes fusion into synthesis: it builds causal relations between MRI, demographics and cerebrospinal fluid biomarkers into PET generation.

*The second thread uses earlier scans to judge the current one.* [LiFE-Net](https://proceedings.mlr.press/v301/yassine26a.html) fuses baseline and follow-up liver CT features through self-attention, trained without labels on synthetic lesions. [Intelligent lesion selection](https://proceedings.mlr.press/v301/qahqaie26a.html) tracks lung metastases by registration and classifies lesion pairs to estimate tumour load in the style of RECIST (Response Evaluation Criteria in Solid Tumours). For prostate active surveillance, [progression prediction](https://proceedings.mlr.press/v301/wang26b.html) gets around a small longitudinal cohort by first training a foundation model on a much larger single-time-point one. [Patient-level contrastive pretraining](https://proceedings.mlr.press/v301/li26c.html) aligns longitudinal chest CT with clinical variables and improves lung cancer risk prediction over conventional fusion.

*The third thread models the trajectory itself.* [4D-VQ-GAN](https://proceedings.mlr.press/v301/zhao26a.html) learns latent dynamics of lung CT with a neural ordinary differential equation and synthesises scans at any time point. It checks them clinically through survival analysis on the generated images. [Temporally parameterised neural displacement fields](https://proceedings.mlr.press/v301/shuaibu26a.html) represent brain change as a continuous function of time, with a regulariser that enforces a monotonic rate of change. Normative and counterfactual generators do related work: a [spherical surface diffusion model](https://proceedings.mlr.press/v301/zhang26d.html) generates anatomy-conditioned cortical norms for Alzheimer's disease, and a [correspondence-preserving shape diffusion model](https://proceedings.mlr.press/v301/zhu26a.html) predicts hippocampal change.

*The fourth thread treats the brain as a graph built from functional MRI.* [STNAGNN](https://proceedings.mlr.press/v301/wang26a.html) supplements sparse functional connectivity with dense, data-driven spatio-temporal connections. [ABFR-KAN](https://proceedings.mlr.press/v301/ward26a.html) avoids a predefined atlas and uses Kolmogorov–Arnold network blocks to diagnose autism. A [graph framework for adolescent psychiatric disorders](https://proceedings.mlr.press/v301/yu26a.html) adds behavioural characteristics to connectivity to address variability between subjects.

*Taken together*, two ideas recur: separate what modalities share from what each adds, and borrow strength from larger single-time-point or synthetic data when longitudinal cohorts are small. Still open: irregular and missing visits, external validation of small-cohort fusion models, and how to judge generated future scans beyond downstream surrogates.

**Representative papers**

- [DAFTED: Decoupled Asymmetric Fusion of Tabular and Echocardiographic Data for Cardiac Hypertension Diagnosis](https://proceedings.mlr.press/v301/stym-popper26a.html)
- [Contrastive Patient-level Pretraining Enables Longitudinal and Multimodal Fusion for Lung Cancer Risk Prediction](https://proceedings.mlr.press/v301/li26c.html)
- [Predicting Prostate Cancer Progression During Active Surveillance Using Longitudinal bpMRI Scans and A Multi-scale Foundation Model](https://proceedings.mlr.press/v301/wang26b.html)
- [4D-VQ-GAN: A World Model for Synthesizing Medical Scans at Any Time Point for Personalized Disease Progression Modeling of Idiopathic Pulmonary Fibrosis](https://proceedings.mlr.press/v301/zhao26a.html)
- [LiFE-Net: Longitudinal information Fusion for Enhanced lesion detection in unsupervised learning contexts](https://proceedings.mlr.press/v301/yassine26a.html)
- [Predicting the Year of Total Knee Replacement: A Transformer-Based Multimodal Approach](https://proceedings.mlr.press/v301/cigdem26a.html)
- [STNAGNN: Data-driven Spatio-temporal Brain Connectivity beyond FC](https://proceedings.mlr.press/v301/wang26a.html)

### Video, surgery and ultrasound

**The ideas behind this group** (a synthesis from the abstracts of about 12 papers in this group; the individual papers have not been read in full).

*The shared problem is that the signal unfolds in time or depends on an operator, while labels are scarce and local.* Surgical video and ultrasound are long, operator-dependent and expensive to annotate. Their meaning often lies in motion, such as tool–tissue interaction, cardiac phase or artefacts across frames, rather than in any single frame. Ultrasound adds a narrow, sector-shaped field of view with no fixed spatial reference.

*The first thread pretrains on unlabelled surgical data and makes the pretraining task fit the domain.* [JHU-VPT(JEPA)](https://proceedings.mlr.press/v301/shah26b.html) learns from multi-site cataract videos by predicting latent features rather than reconstructing pixels. It works well with a frozen encoder, which the authors present as a privacy advantage. The [surgical flow masked autoencoder](https://proceedings.mlr.press/v301/mostafa26a.html) replaces random masking with masks guided by optical flow, so the network attends to tool–tissue interaction. This helps phase and event recognition in cataract and neurosurgical video. For laparoscopic cholecystectomy, [SurgicalSemiSeg](https://proceedings.mlr.press/v301/zhou26a.html) finds that pixel-level objectives suit deformable anatomy best and tunes a masked denoising autoencoder accordingly. [Organ-specific decoders](https://proceedings.mlr.press/v301/tomar26a.html) address the bias against small organs in abdominal surgical scenes.

*The second thread models time explicitly in ultrasound.* [LUV-Net](https://proceedings.mlr.press/v301/lee26a.html) combines pattern-specific spatial attention with a module linking adjacent frames to label four lung ultrasound patterns. [FA-DETR](https://proceedings.mlr.press/v301/qin26a.html) uses a lightweight spatio-temporal attention module for real-time breast lesion detection in ultrasound video, aiming for both accuracy and speed. [DAFTED](https://proceedings.mlr.press/v301/stym-popper26a.html) compresses echocardiographic sequences into time series before fusing them with patient records. Outside ultrasound, [sequence models for the cell cycle](https://proceedings.mlr.press/v301/leger26a.html) make the same point: causal and transformer sequence models beat single-frame ones.

*The third thread manufactures labels from clinical routine.* [PLAX ejection fraction](https://proceedings.mlr.press/v301/gao26a.html) matches echocardiographic videos to clinical notes by time and uses view classifiers and proxy labels. This yields a large labelled set for a parasternal long-axis view that public datasets barely cover, with errors the authors place in the range reported for the clinical-standard apical view.

*The fourth thread restores spatial context that a probe or microscope does not give.* [FERN](https://proceedings.mlr.press/v301/gilliland26a.html) aligns high-resolution 2D fetal echocardiography into a 3D reference space defined by lower-resolution 3D acquisitions. It is trained on simulated slices and validated on real scans. [LOTUS](https://proceedings.mlr.press/v301/yao26a.html) outpaints the sector-shaped field of view of 3D ultrasound into a rectangle with a latent diffusion model, which makes stitching registration less prone to local minima. [SSDD-GAN](https://proceedings.mlr.press/v301/zhang26b.html) completes synthetic microscope views of mastoidectomy, trained on real surgical data and applied zero-shot.

*Taken together*, the group's answer to scarce labels is domain-shaped self-supervision (flow-guided masks, latent prediction, pixel-level objectives), proxy labels and simulation. Speed and spatial registration are treated as first-class requirements. Still open: multi-site external validation, since several studies rely on single in-house datasets; long-range temporal reasoning across whole procedures or examinations; and whether simulated or synthetic training transfers to real acquisitions at scale.

**Representative papers**

- [A Vision Foundation Model for Cataract Surgery Using Joint-Embedding Predictive Architecture](https://proceedings.mlr.press/v301/shah26b.html)
- [Surgical Flow Masked Autoencoder for Event Recognition](https://proceedings.mlr.press/v301/mostafa26a.html)
- [SurgicalSemiSeg: A Semi-Supervised Framework for Laparoscopic Image Segmentation](https://proceedings.mlr.press/v301/zhou26a.html)
- [LUV-Net: Multi-Pattern Lung Ultrasound Video Classification through Pattern-Specific Attention with Efficient Temporal Feature Extraction](https://proceedings.mlr.press/v301/lee26a.html)
- [Real-time Breast Lesion Detection in Videos via Spatial-temporal Feature Aggregation](https://proceedings.mlr.press/v301/qin26a.html)
- [Machine Learning with Scarce Data: Ejection Fraction Prediction Using PLAX View](https://proceedings.mlr.press/v301/gao26a.html)
- [FERN: A Fetal Echocardiography Registration Network for 2D-to-3D Alignment](https://proceedings.mlr.press/v301/gilliland26a.html)

### Benchmarks, data and validation

**The ideas behind this group** (a synthesis from the abstracts of about 15 papers in this group; the individual papers have not been read in full).

*The shared problem is that progress depends as much on data, labels and fair comparisons as on new models.* These papers ask which existing method works where, how to spend a limited annotation budget, how to train across institutions that cannot pool data, and how to move large datasets around at all. Many of their findings are negative or deflating, which is what makes them useful.

*The first thread compares methods under realistic conditions.* On multi-centre breast MRI from nine sites, a standard nnU-Net reaches high lesion sensitivity but produces too many false positives per case to be used clinically, which motivates adapted variants ([breast MRI comparison](https://proceedings.mlr.press/v301/geissler26a.html)). For zero-shot bone segmentation in CT, general Segment Anything Model (SAM) variants beat medically fine-tuned ones, and a box combined with a centre point is the best prompt ([SAM bone benchmark](https://proceedings.mlr.press/v301/magg26a.html)). For parameter-efficient fine-tuning of SAM, where the adapter layers go matters more than which kind is used ([PEFT for SAM](https://proceedings.mlr.press/v301/teuber26a.html)). Results of the same kind challenge common assumptions about scale. Training choices matter more than model size in a chest X-ray foundation model ([CXformer](https://proceedings.mlr.press/v301/al-mahrooqi26a.html)), and zero-shot performance varies widely with the wording of text prompts ([prompting across populations](https://proceedings.mlr.press/v301/fay26a.html)). Generic augmentations such as MixUp and auxiliary Fourier augmentation are evaluated systematically for robustness in MRI segmentation ([augmentations for the unknown](https://proceedings.mlr.press/v301/vaish26a.html)).

*The second thread is about where labels come from and how many are needed.* Under a fixed budget it is almost always better to annotate fewer slices in more volumes. Unsupervised active learning does no better than random or evenly spaced slices, and interpolating masks rarely helps ([slice selection](https://proceedings.mlr.press/v301/zhang26a.html)). Pruning by community detection on an image-similarity network, without any training, cuts the annotation needed for polyp segmentation ([PRIME](https://proceedings.mlr.press/v301/rahman26a.html)). Labels can also be had cheaply from elsewhere: timing links between clinical notes and echocardiography videos ([PLAX ejection fraction](https://proceedings.mlr.press/v301/gao26a.html)), report extraction with a large language model for wrist fractures ([LLM labels](https://proceedings.mlr.press/v301/vasylechko26a.html)), or anatomical masks from universal segmentation models used as auxiliary supervision ([free anatomical supervision](https://proceedings.mlr.press/v301/li26d.html)). Where data stay small, few-shot learning and in-domain pretraining are tested directly ([CVM few-shot](https://proceedings.mlr.press/v301/schneider26a.html)).

*The third thread studies federated learning when clients differ.* One study adds ordinal classification to four federated frameworks and stresses them with increasing label heterogeneity ([federated ordinal learning](https://proceedings.mlr.press/v301/corbetta26a.html)). Another merges shared class weights and appends client-specific ones, so that institutions with different label sets can train one report labeller ([Surgical Aggregation](https://proceedings.mlr.press/v301/shah26a.html)).

*The fourth thread is infrastructure.* A format-agnostic streaming database serves several resolutions and formats from a single high-resolution copy, reducing storage and bandwidth ([MIST](https://proceedings.mlr.press/v301/kulkarni26a.html)).

*Taken together*, the evidence favours strong, well-configured baselines, careful use of annotation effort and cheap secondary sources of labels over bigger models or clever sampling. Still open: most comparisons cover one task or one modality, so the guidelines may not transfer. The federated studies rely on simulated heterogeneity. Labels drawn from reports, notes or other models carry their own errors, which are rarely quantified.

**Representative papers**

- [How to select slices for annotation to train best-performing deep learning segmentation models for cross-sectional medical images?](https://proceedings.mlr.press/v301/zhang26a.html)
- [Multi-centric Comparison of Deep Learning Models for Lesion Detection in Breast MRI](https://proceedings.mlr.press/v301/geissler26a.html)
- [Zero-shot capability of 2D SAM-family models for bone segmentation in CT scans](https://proceedings.mlr.press/v301/magg26a.html)
- [Empirical Analysis of Scaling Vision Foundation Models for Chest X-rays](https://proceedings.mlr.press/v301/al-mahrooqi26a.html)
- [Understanding the Impact of Client Heterogeneity on Ordinal Classification in Federated Medical Image Analysis](https://proceedings.mlr.press/v301/corbetta26a.html)
- [Training-Free Dataset Pruning for Polyp Segmentation via Community Detection in Similarity Networks](https://proceedings.mlr.press/v301/rahman26a.html)
- [Towards Resource-Efficient Streaming of Large-Scale Medical Image Datasets for Deep Learning](https://proceedings.mlr.press/v301/kulkarni26a.html)

## Outlier papers

Papers that sit outside the main groups in topic, method or framing.

### Unusual subjects or data

- [Sequence models for continuous cell cycle stage prediction from brightfield images](https://proceedings.mlr.press/v301/leger26a.html): basic cell biology rather than the clinic; 1.3 million images of dividing cells, resolving cell-cycle transitions that cannot be seen by eye.
- [Unsupervised Cellular Anomaly Detection in Toxicological Histopathology](https://proceedings.mlr.press/v301/juturu26a.html): rat kidney and liver tissue from drug-safety studies, where the comparison is between dose groups rather than patients.
- [DeFusion: An Effective Decoupling Fusion Network for Multi-Modal Pregnancy Prediction](https://proceedings.mlr.press/v301/ouyang26a.html): time-lapse embryo images and parental fertility records, used to predict whether in vitro fertilisation embryo transfer leads to pregnancy.
- [Learning from a Few Shots: Data-efficient Cervical Vertebral Maturation Assessment](https://proceedings.mlr.press/v301/schneider26a.html): orthodontics, where the scan is used to judge a growth phase so that treatment can be timed.
- [SSDD-GAN: Single-Step Denoising Diffusion GAN for Cochlear Implant Surgical Scene Completion](https://proceedings.mlr.press/v301/zhang26b.html): fills in the surroundings of synthetic surgical-microscope views of mastoidectomy, the ear surgery done for cochlear implants.
- [Synthetic Data Generated from CT Scans for Patient Pose Assessment](https://proceedings.mlr.press/v301/laufer26a.html): depth-camera images of the ankle, made from CT scans, to check the patient's position before an X-ray is taken.
- [Towards Resource-Efficient Streaming of Large-Scale Medical Image Datasets for Deep Learning](https://proceedings.mlr.press/v301/kulkarni26a.html): data infrastructure; one high-resolution copy of each image is streamed at whatever resolution and format each researcher needs.

### Social or ethical findings

- [Can Diffusion Models Generalize? Privacy and Fairness Trade-offs for Medical Data Sharing](https://proceedings.mlr.press/v301/dombrowski26a.html): artificial fingerprints are planted in the training data; if the model copies them exactly privacy is breached, and if it ignores them rare subgroups are left out.
- [Evaluating Shortcut Utilization in Deep Learning Disease Classification through Counterfactual Analysis](https://proceedings.mlr.press/v301/vigneshwaran26a.html): in a Parkinson's disease classifier, counterfactually removing scanner site lowers AUROC from 0.74 to 0.65, so site is used as a shortcut; removing sex barely changes it.
- [Beyond the Prompt: Deploying Medical Foundation Models on Diverse Chest X-ray Populations](https://proceedings.mlr.press/v301/fay26a.html): the wording of the text prompt alone shifts zero-shot performance by up to 46% in Matthews correlation coefficient; results are also reported by sex, age and race.

### Unusual mathematics or methods

- [BinaryFormer: 1-bit self-attention for long-range transformers in medical image segmentation and 3D diffusion models](https://proceedings.mlr.press/v301/heinrich26b.html): self-attention computed from bitwise Hamming distances, trained end to end through learnable query and key weights and explained by Bernoulli sampling.
- [Training-Free Dataset Pruning for Polyp Segmentation via Community Detection in Similarity Networks](https://proceedings.mlr.press/v301/rahman26a.html): network science rather than a trained model decides which images to keep.
- [Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free](https://proceedings.mlr.press/v301/favero26a.html): a generative model used as a classifier, by comparing reconstruction errors under each class label.
- [Vector Representations of Vessel Trees](https://proceedings.mlr.press/v301/batten26a.html): a whole vascular tree encoded as one vector, with a recursive decoder that guarantees a valid tree comes back.
- [Biologically-Constrained Multi-Label Classification with Learnable Domain Knowledge](https://proceedings.mlr.press/v301/mouadden26a.html): learnable constraint matrices rule out biologically impossible label combinations.
- [Improving brain disorder diagnosis with advanced brain function representation and Kolmogorov-Arnold Networks](https://proceedings.mlr.press/v301/ward26a.html): Kolmogorov–Arnold network blocks in place of the usual multi-layer perceptrons.

### Papers that critique the field

- [How to select slices for annotation to train best-performing deep learning segmentation models for cross-sectional medical images?](https://proceedings.mlr.press/v301/zhang26a.html): with a fixed budget, annotate fewer slices across more volumes; unsupervised active learning does no better than random or evenly spaced slices, and interpolating masks rarely helps.
- [A Balancing Act: Optimizing Classification and Retrieval in Cross-Modal Vision Models](https://proceedings.mlr.press/v301/lefkes26a.html): as contrastive fine-tuning improves vision–text alignment, classification accuracy falls rapidly towards zero.
- [Empirical Analysis of Scaling Vision Foundation Models for Chest X-rays](https://proceedings.mlr.press/v301/al-mahrooqi26a.html): the larger model is no better than the small one, so training choices matter more than size.
- [Zero-shot capability of 2D SAM-family models for bone segmentation in CT scans](https://proceedings.mlr.press/v301/magg26a.html): the general-purpose SAM and SAM2 beat their medically fine-tuned versions.
- [Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales](https://proceedings.mlr.press/v301/cheng26a.html): strong augmentations borrowed from natural images hurt retinal pretraining; weaker ones work better.
- [Generate to Ground: Multimodal Text Conditioning Boosts Phrase Grounding in Medical Vision-Language Models](https://proceedings.mlr.press/v301/nutzel26a.html): the cross-attention maps of a generative diffusion model beat discriminative contrastive models at linking report phrases to image regions.

### Unusual clinical framing

- [Imitating Radiological Scrolling: A Glocal-Lobal Attention Model for 3D Chest CT Volumes Multi-Label Anomaly Classification](https://proceedings.mlr.press/v301/di-piazza26a.html): the model is designed to copy how radiologists scroll through CT slices.
- [4D-VQ-GAN: A World Model for Synthesizing Medical Scans at Any Time Point for Personalized Disease Progression Modeling of Idiopathic Pulmonary Fibrosis](https://proceedings.mlr.press/v301/zhao26a.html): a "world model" for lung fibrosis in which a neural ordinary differential equation evolves the scans' latent codes, and biomarkers from generated scans are tested against survival.
- [Self-Supervised Synthetic Cerebral Vessel Tree Generation using Semantic Signed Distance Fields](https://proceedings.mlr.press/v301/kuipers26a.html): virtual patient populations for in-silico trials of stroke treatments.
- [Anatomy-Guided Surface Diffusion Model for Alzheimer's Disease Normative Modeling](https://proceedings.mlr.press/v301/zhang26d.html): normative modelling (how far each person lies from a healthy reference) carried out on the cortical surface.

### Imaging as a stand-in for another measurement

Papers that predict, from routine or cheap imaging, something normally measured in a more invasive, expensive or different way.

- [Mapping Functional Language Areas with non-Functional Brain MRI](https://proceedings.mlr.press/v301/leshem26a.html): locates Broca's and Wernicke's areas for surgical planning without functional MRI, which needs the patient's active cooperation.
- [MagNet: Multi-Level Attention Graph Network for Predicting High-Resolution Spatial Transcriptomics](https://proceedings.mlr.press/v301/zhu26b.html): gene expression at 8 µm resolution predicted from histology images.
- [Causal PETS: Causality-Informed PET Synthesis from Multi-modal Data](https://proceedings.mlr.press/v301/li26a.html): PET synthesised from MRI, demographics and cerebrospinal fluid biomarkers, avoiding the cost and radiation of PET.
- [Neural fields for tissue attenuation curve reconstruction in sparsely sampled time-resolved CT](https://proceedings.mlr.press/v301/vries26a.html): full perfusion curves rebuilt from a few of the roughly 30 time points CT perfusion normally needs, opening perfusion analysis from multiphase CT angiography.
- [Evaluation of Virtual Stain Multiplexed CD68 for Macrophage Detection in NSCLC PD-L1 Slides](https://proceedings.mlr.press/v301/arbel26a.html): a virtual stain replaces an extra physical stain; agreement between pathologists (Fleiss' kappa) rises from −0.1 to 0.62.
- [Staging Liver Fibrosis with Hepatic Perivascular Adipose Tissue as a CT Biomarker](https://proceedings.mlr.press/v301/chan26a.html): fat around the portal vein on CT as a marker of fibrosis, with opportunistic screening in mind.
- [Machine Learning with Scarce Data: Ejection Fraction Prediction Using PLAX View](https://proceedings.mlr.press/v301/gao26a.html): ejection fraction from the parasternal long-axis view instead of the standard apical view, with labels taken from clinical notes matched to the videos by time.

The [cell-cycle paper](https://proceedings.mlr.press/v301/leger26a.html) under *Unusual subjects or data* fits here too: label-free brightfield images stand in for genetically engineered fluorescent reporters.

## Theory picks

The most mathematically foundational papers, ranked for a close reading (from abstracts only).

1. [Fast forward: registration through density alignment and splatting](https://proceedings.mlr.press/v301/heinrich26a.html): replaces the usual backward warping (interpolation) with forward splatting (extrapolation), so deformable registration becomes the alignment of rasterised point-cloud densities. A clean change of formulation; it is worth working out why the two directions behave differently.
2. [MORPH-LER: Log-Euclidean Regularization](https://proceedings.mlr.press/v301/karanam26a.html): computes the principal matrix logarithm of deformation fields through repeated square roots, giving a linear latent space that respects composition and inverse consistency of diffeomorphisms. A good entry point into Log-Euclidean statistics on transformation groups.
3. [Feature Attribution through Total Variance Decomposition](https://proceedings.mlr.press/v301/jin26a.html): a feature's importance is the share of the variance in the model's decisions explained by conditioning on that feature (the law of total variance), with a diffusion model sampling the conditional distribution when data are scarce.
4. [SACP: Spatially-Adaptive Conformal Prediction](https://proceedings.mlr.press/v301/bereska26a.html): a distance-weighted non-conformity score in class-conditional conformal calibration, so prediction sets widen near critical anatomy while coverage holds; already annotated in [this repo](../2025-bereska-sacp/index.html).
5. [Clinical Measurements with Calibrated Instance-Dependent Confidence Interval](https://proceedings.mlr.press/v301/nizhar26a.html): a network predicts a Gaussian variance per case and conformal prediction rescales it, giving case-adaptive intervals with a coverage guarantee, compared directly with variance scaling and conformalised quantile regression.
6. [BinaryFormer: 1-bit self-attention](https://proceedings.mlr.press/v301/heinrich26b.html): attention from Hamming distances between binary codes, made differentiable through learnable query and key weights, justified by Bernoulli sampling and distance interpolation, an argument worth checking for yourself.

Also worth a look: [Equivariant Imaging Biomarkers](https://proceedings.mlr.press/v301/chen26a.html) (symmetric kernels for rotation and reflection), [Temporally Parameterized Neural Displacement Fields](https://proceedings.mlr.press/v301/shuaibu26a.html) (a regulariser forcing monotonic change, from the network's analytic derivatives), [Anatomy-Guided Surface Diffusion](https://proceedings.mlr.press/v301/zhang26d.html) (diffusion on the sphere), [Correspondence-Preserving Diffusion](https://proceedings.mlr.press/v301/zhu26a.html) (point correspondences as in statistical shape models), [SRMRI](https://proceedings.mlr.press/v301/poudel26a.html) (a score-based stochastic differential equation with data consistency, treating super-resolution as an inverse problem) and [Shortcut Utilization through Counterfactual Analysis](https://proceedings.mlr.press/v301/vigneshwaran26a.html) (causal counterfactuals in activation space).

## Sources

- *Proceedings of the 8th International Conference on Medical Imaging with Deep Learning*,
  PMLR volume 301: <https://proceedings.mlr.press/v301/>
