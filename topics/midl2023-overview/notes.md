---
title: "MIDL 2023: topic map, outliers and theory picks"
authors: "Topic notes"
venue: "MIDL 2023"
tags: [midl, survey, conference-overview, segmentation, equivariance, reconstruction, reproducibility]
status: living
category: "Medical image analysis"
subcategory: "Conference overviews"
short_title: "MIDL 2023 — overview"
summary: "a high-level map of the 112 MIDL 2023 full papers (PMLR v227) in fourteen topic groups, with an idea-level synthesis of each, the outlier papers that fall outside them, and a shortlist of the most mathematically foundational ones to read"
related: [midl2024-overview, midl2025-overview, midl2026-overview, 2023-xu-federated-cross-learning]
---

## Why this page exists

MIDL 2023 (Nashville, 10–12 July 2023) was published as
[PMLR volume 227](https://proceedings.mlr.press/v227/), with 112 full papers. The volume
appeared in 2024, so its paper keys end in "24a", "24b" and so on. This page is the companion to
the [MIDL 2024](../midl2024-overview/index.html), [MIDL 2025](../midl2025-overview/index.html)
and [MIDL 2026](../midl2026-overview/index.html) overviews: which topics dominate, which papers do
not fit the pattern, and which are worth a close, theory-first read.

The grouping and the syntheses are from **titles and abstracts only**; the PDFs have not been
read. Shares are approximate and many papers sit in more than one group, so they add up to
more than 100%.

## Topic groups at a glance

| Group | Share | What's in it |
|---|---|---|
| [Segmentation](#segmentation) | ~23% | Transformer and diffusion segmenters, size- and geometry-aware losses, semi-supervised and clinical labels, missing modalities, test-time and source-free adaptation |
| [Self-supervised and representation learning](#self-supervised-and-representation-learning) | ~13% | Augmentation studies for chest X-ray, masked ViT autoencoders, video pretraining for CT, nuisance-aware contrastive objectives, text as a second view |
| [Computational pathology and microscopy](#computational-pathology-and-microscopy) | ~18% | Weakly supervised MIL, stain invariance and optimal transport, stain decomposition and virtual staining, cell graphs, connectomics |
| [Generative models and synthesis](#generative-models-and-synthesis) | ~16% | Diffusion and GAN translation and harmonisation, synthetic data for training and validation, generators as segmenters and posterior samplers |
| [Reconstruction and acquisition](#reconstruction-and-acquisition) | ~12% | Deep image prior, data-consistent motion correction, posterior sampling, learned trajectories, adversarial robustness of reconstruction |
| [Registration, geometry and equivariance](#registration-geometry-and-equivariance) | ~14% | Incompressible and geometry-informed deformations, cortical registration, shape graphs and SSMs, E(3)×SO(3) and rotation-scale equivariance |
| [Trustworthiness](#trustworthiness) | ~11% | Interpretable-by-design models, confidence for grading, GNN calibration, OOD detection, projection bias, fairness of uncertainty |
| [Unsupervised anomaly detection](#unsupervised-anomaly-detection) | ~4% | Reversed autoencoders, patched diffusion, one-class SVM on learned embeddings, reverse distillation for slides |
| [Brain networks and neuroimaging](#brain-networks-and-neuroimaging) | ~9% | Dynamic brain graphs, fMRI fingerprinting, incomplete connectomes, image decoding from cortex, tractography |
| [Vision-language models and reports](#vision-language-models-and-reports) | ~6% | CLIP-style pretraining with little data, ECG zero-shot learning, retrieval-based report generation, structured reporting |
| [Domain shift, federated and continual learning](#domain-shift-federated-and-continual-learning) | ~14% | Harmonisation, silver-standard labels, normalisation layers under shift, round-robin federated training, coreset replay |
| [Multimodal and longitudinal clinical prediction](#multimodal-and-longitudinal-clinical-prediction) | ~6% | Image–tabular fusion (concatenation often wins), survival prediction, temporal ordering as supervision |
| [Surgery, video and ultrasound](#surgery-video-and-ultrasound) | ~7% | Ultrasound NeRF, transcranial ultrasound CT, operating-room activity, fluorescence video, fetal segmentation quality |
| [Benchmarks, reproducibility and evaluation](#benchmarks-reproducibility-and-evaluation) | ~14% | Reproducibility audits of MIDL code and training, clinically motivated metrics, systematic comparisons, annotation strategy |

**Modalities, roughly by frequency:** MRI (brain above all), histopathology and microscopy, CT,
chest X-ray, then ultrasound, surgical video, PET and others.

**Across four years:** 2023 is the last volume before promptable foundation models. There is no
SAM, segmentation papers still argue about the backbone, and pathology relies on weak supervision
and explicit stain invariance rather than pretrained encoders. Language is a small, CLIP-style
group. Diffusion models are arriving, as segmenters, harmonisers and memory-efficient 3D models.
Brain-network and fMRI papers form a group of their own, which shrinks in later years. The volume
is also unusually rich in explicit mathematics (equivariance under E(3)×SO(3), incompressible
deformations, posterior sampling, singular-value fine-tuning) and in self-criticism: two papers
audit the reproducibility of MIDL itself.

## Topic groups

Each group lists a few representative papers, not all of them.

### Segmentation

**The ideas behind this group** (a synthesis from the abstracts of about 26 papers in this group; the individual papers have not been read in full).

*The shared problem is that a segmentation network is trained on one curated, fully labelled dataset, but it meets incomplete labels, missing inputs and unfamiliar scanners.* Labels in clinical archives are noisy or partial. Small instances get lost under overlap losses. Sites differ in scanner, sequence and stain. Unlike two years later, there is no promptable foundation segmenter yet, so the papers still argue about the backbone itself.

*The first thread argues about the architecture.* Transformers are made multi-scale ([MERIT](https://proceedings.mlr.press/v227/rahman24a.html)), cheaper in 3D through axial attention ([3D MAT](https://proceedings.mlr.press/v227/liu24b.html)), or real-time and tested out of distribution on six polyp centres ([TransNetR](https://proceedings.mlr.press/v227/jha24a.html)). When modalities are missing, auxiliary tokens link the full and missing-modality paths ([MMCFormer](https://proceedings.mlr.press/v227/karimijafarbigloo24b.html)). Diffusion appears as a segmenter, with conditions that change at each sampling step ([MedSegDiff](https://proceedings.mlr.press/v227/wu24a.html)), and is trained on patches but run on whole 3D volumes to save memory ([PatchDDM](https://proceedings.mlr.press/v227/bieder24a.html)). Other papers build structure into the layers. Rotation-scale equivariant steerable filters need fewer parameters ([RSESF](https://proceedings.mlr.press/v227/yang24a.html)), and a kaleidoscope transform reduces overfitting when the training set is extremely small ([KAN3D](https://proceedings.mlr.press/v227/woo24a.html)).

*The second thread changes what the loss rewards.* An instance-wise and centre-of-instance loss tackles the imbalance that the Dice loss creates between large and small lesions, improving on it by 1.7-3.7% Dice ([ICI loss](https://proceedings.mlr.press/v227/rachmadi24a.html)). Label smoothing follows the geodesic distance transform of the image rather than Euclidean distance ([GeoLS](https://proceedings.mlr.press/v227/vasudeva24a.html)). A topology loss encodes what is known about catheter shape ([catheters](https://proceedings.mlr.press/v227/hwang24a.html)).

*The third thread learns from fewer or messier labels.* Semi-supervised methods align class representations of labelled and unlabelled data ([ICL](https://proceedings.mlr.press/v227/zhu24a.html)) or of unpaired computed tomography (CT) and MRI ([unpaired multi-modal](https://proceedings.mlr.press/v227/yang24c.html)). In both, the helper module is dropped at test time. A teacher–student model trains on routine clinical contours with missing annotations, and its output is judged as clinically acceptable as manual contours ([organs at risk](https://proceedings.mlr.press/v227/grewal24a.html)). Other papers add further sources of supervision. Clustering alone yields masks ([MS-Former](https://proceedings.mlr.press/v227/karimijafarbigloo24a.html)). Registration and segmentation reinforce each other to give high-resolution masks from unaligned low-resolution views ([SuperMask](https://proceedings.mlr.press/v227/gu24b.html)). Joint reconstruction and segmentation shows a correlation of ρ=0.92 between the two tasks ([MTLRS](https://proceedings.mlr.press/v227/karkalousos24a.html)).

*The fourth thread handles a shift between sites.* Test-time adaptation without back-propagation feeds a domain code into adaptive batch normalisation ([on-the-fly TTA](https://proceedings.mlr.press/v227/valanarasu24a.html)). Source-free adaptation needs neither source data nor new labels ([MATS](https://proceedings.mlr.press/v227/chen24b.html)). Pretraining on target-domain silver-standard masks and then fine-tuning on gold-standard masks reaches mean DSC = 0.89 for the ventricles ([silver standard](https://proceedings.mlr.press/v227/crystal24a.html)). Other papers try intensity standardisation ([WML standardisation](https://proceedings.mlr.press/v227/ghazvanchahi24a.html)), stain-adversarial training ([stain invariance](https://proceedings.mlr.press/v227/abutalip24a.html)) and an nnU-Net adapted to pathology ([nnUNet for pathology](https://proceedings.mlr.press/v227/spronck24a.html)). In federated learning, a theoretical analysis of model aggregation motivates round-robin training with no aggregation step ([FedCross](https://proceedings.mlr.press/v227/xu24a.html)), and radiomics clusters define the groups for personalised models ([clustered personalisation](https://proceedings.mlr.press/v227/manthe24a.html)).

*Taken together*, the group treats segmentation as a problem of priors: geometric, topological, anatomical and cross-task. Only one paper asks whether a given mask can be trusted: a classifier separates good from poor fetal masks with over 90% accuracy. Gestational age estimated from good masks differs from the doctors' value by 1.45 days, against up to 7.73 days from poor masks ([FUSQA](https://proceedings.mlr.press/v227/cengiz24a.html)). Still open: most architecture claims rest on one or two benchmarks with no repeated runs, priors such as equivariance and geodesics are not compared with each other, and quality estimation is not combined with the label-efficient or adaptive methods.

**Representative papers**

- [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://proceedings.mlr.press/v227/wu24a.html)
- [GeoLS: Geodesic Label Smoothing for Image Segmentation](https://proceedings.mlr.press/v227/vasudeva24a.html)
- [Improving Segmentation of Objects with Varying Sizes in Biomedical Images using Instance-wise and Center-of-Instance Segmentation Loss Function](https://proceedings.mlr.press/v227/rachmadi24a.html)
- [On-the-Fly Test-time Adaptation for Medical Image Segmentation](https://proceedings.mlr.press/v227/valanarasu24a.html)
- [Learning Clinically Acceptable Segmentation of Organs at Risk in Cervical Cancer Radiation Treatment from Clinically Available Annotations](https://proceedings.mlr.press/v227/grewal24a.html)
- [Federated Cross Learning for Medical Image Segmentation](https://proceedings.mlr.press/v227/xu24a.html)
- [SuperMask: Generating High-resolution object masks from multi-view, unaligned low-resolution MRIs](https://proceedings.mlr.press/v227/gu24b.html)
- [FUSQA: Fetal Ultrasound Segmentation Quality Assessment](https://proceedings.mlr.press/v227/cengiz24a.html)

### Self-supervised and representation learning

**The ideas behind this group** (a synthesis from the abstracts of about 15 papers in this group; the individual papers have not been read in full).

*The shared problem is that labels are scarce, and the self-supervised learning (SSL) recipes built for natural images may not encode the right invariances for medical data.* An augmentation that is harmless on a photograph can erase a finding on a chest X-ray. Nuisance factors such as batch effects, scan contrast and ageing are domain-specific. The papers ask which pretext task, which augmentations and which pretraining data actually transfer.

*The first thread tests recipes imported from natural images, and the answers do not always agree.* A systematic study of augmentations for Siamese networks on chest X-rays identifies a set that generalises out of distribution and beats supervised baselines by up to 20% using only zero-shot transfer and linear probes ([Siamese augmentations](https://proceedings.mlr.press/v227/sluijs24a.html)). The masked vision transformer autoencoder gains a structured reconstruction loss, a contrastive loss between two masked views and a 3D version ([ViT-AE++](https://proceedings.mlr.press/v227/prabhakar24b.html)). For mortality risk from chest X-rays, a semi-supervised autoencoder beats contrastive and transfer learning ([SSL for risk](https://proceedings.mlr.press/v227/chen24c.html)). For 3D CT, pretraining on natural video improves seven 3D models on two chest CT datasets, and the out-of-domain Kinetics dataset helps more than typically sized in-domain CT pretraining ([video pretraining](https://proceedings.mlr.press/v227/ke24a.html)).

*The second thread writes the domain's nuisance factors into the objective.* In high-content screening, metadata tells the model which signal is a batch effect to ignore ([CDCL](https://proceedings.mlr.press/v227/haslum24a.html)). Brain MRI re-identification adds transformations for scan contrast, age and disease progression ([DeepBrainPrint](https://proceedings.mlr.press/v227/puglisi24a.html)). A triplet loss that pulls together functional MRI (fMRI) sequences from the same subject is the key to better stress prediction ([MetricFMRI](https://proceedings.mlr.press/v227/rosenman24a.html)). Ordering two scans of the same person in time is a pretext task that forces the model to learn changes that cannot run backwards and to ignore nuisance variation ([PaIRNet](https://proceedings.mlr.press/v227/kim24a.html)).

*The third thread uses text as a second view.* With 1% and 10% labelled data, joint multi-modal and self-supervised learning beats SSL alone and matches multi-modal learning, and multi-modal learning is more robust out of distribution ([reports improve representations](https://proceedings.mlr.press/v227/huang24a.html)). Combining unimodal SSL, local and global contrastive losses and extra positive pairs is roughly equivalent to training with 10× the data ([low-data VLM](https://proceedings.mlr.press/v227/windsor24a.html)). Tabular clinical data can also be turned into text prompts for contrastive learning on kidney transplant MRI ([MEDIMP](https://proceedings.mlr.press/v227/milecki24a.html)).

*The fourth thread uses learned representations to choose, create or label data.* Contrastive features solve the cold-start problem of active learning, where the first queries are worse than random ([HaCon](https://proceedings.mlr.press/v227/chen24a.html)). A variational autoencoder that encodes neighbouring cell-maturation stages generates samples for few-shot classes ([RelationVAE](https://proceedings.mlr.press/v227/nguyen24a.html)). Segmentation maps serve as prompts for parameter-efficient tuning ([SegPrompt](https://proceedings.mlr.press/v227/zhu24b.html)), and self-supervised clustering produces masks ([MS-Former](https://proceedings.mlr.press/v227/karimijafarbigloo24a.html)).

*Taken together*, the group moves from copying natural-image SSL towards objectives that name the nuisance factors to ignore. Still open: the comparisons disagree on whether contrastive, autoencoding or video pretraining is best, and none explains why out-of-domain data beats in-domain data. Each study uses one or two datasets. No paper says which invariances a medical task should require before training starts.

**Representative papers**

- [Exploring Image Augmentations for Siamese Representation Learning with Chest X-Rays](https://proceedings.mlr.press/v227/sluijs24a.html)
- [Video pretraining advances 3D deep learning on chest CT tasks](https://proceedings.mlr.press/v227/ke24a.html)
- [A comparison of self-supervised pretraining approaches for predicting disease risk from chest radiograph images](https://proceedings.mlr.press/v227/chen24c.html)
- [ViT-AE++: Improving Vision Transformer Autoencoder for Self-supervised Medical Image Representations](https://proceedings.mlr.press/v227/prabhakar24b.html)
- [Metadata-guided Consistency Learning for High Content Images](https://proceedings.mlr.press/v227/haslum24a.html)
- [Learning to Compare Longitudinal Images](https://proceedings.mlr.press/v227/kim24a.html)
- [Making Your First Choice: To Address Cold Start Problem in Medical Active Learning](https://proceedings.mlr.press/v227/chen24a.html)

### Computational pathology and microscopy

**The ideas behind this group** (a synthesis from the abstracts of about 20 papers in this group; the individual papers have not been read in full).

*The shared problem is that slides are gigapixel images with labels at the wrong level, made in labs whose staining differs.* Whole slide images (WSIs) usually come with a slide-level or even endoscopic label. Cell-level annotation is expensive. Every laboratory's protocol and scanner shifts the colour distribution. Multiplex stains add a further problem: separating overlapping signals.

*The first thread turns weak labels into fine-grained outputs.* Multiple instance learning (MIL) with top-k pooling trains a detector whose pseudo-labels then supervise pixel-level segmentation, on more than 100k whole slide images ([breast neoplasms](https://proceedings.mlr.press/v227/casson24a.html)). Trainable prototypes softly cluster all patches and replace attention scores as the explanation ([TPMIL](https://proceedings.mlr.press/v227/yang24d.html)). Reverse knowledge distillation matches prior metastasis detection with less than ten percent of the training samples on Camelyon16 ([limited tumorous data](https://proceedings.mlr.press/v227/he24a.html)). Endoscopic labels alone train biopsy models (CD vs UC AUC=0.87) whose attention a pathologist found consistent with inflammation ([IBD biopsies](https://proceedings.mlr.press/v227/mokhtari24a.html)). For ordinal grading, a confidence score needing no extra training separates mispredicted slides, with a gap in AUC of 17.1% ([grading confidence](https://proceedings.mlr.press/v227/lubrano24a.html)).

*The second thread attacks stain and site variation in several different ways.* Channel attention finds stain-specific features early in the network and passes them to a stain-adversarial branch ([stain invariance](https://proceedings.mlr.press/v227/abutalip24a.html)). An optimal-transport loss compares batches rather than single samples, and only it reliably classifies a phenotype unseen in training ([optimal transport](https://proceedings.mlr.press/v227/falahkheirkhah24a.html)). Silver-standard labels from the target laboratory give 95.9% accuracy on the Ki-67 proliferation index ([Ki-67 silver standard](https://proceedings.mlr.press/v227/dy24a.html)). Rotation and scale are built into the filters ([RSESF](https://proceedings.mlr.press/v227/yang24a.html)). nnU-Net gains pathology-specific colour augmentation and a WSI inference pipeline ([nnUNet for pathology](https://proceedings.mlr.press/v227/spronck24a.html)).

*The third thread treats stains as signals to decompose or synthesise.* Unsupervised decomposition of 7-plexed immunohistochemistry (IHC) needs only colour samples, and an inversion regulariser removes most of the solutions that the underdetermined problem allows ([stain decomposition](https://proceedings.mlr.press/v227/abousamra24a.html)). Virtual haematoxylin and eosin (H&E) is made from multiplex immunofluorescence by a CycleGAN with segmentation constraints. Random sliding-window inference reduces tiling seams with 50% fewer computation resources ([MxIF to H&E](https://proceedings.mlr.press/v227/bao24a.html)). A comparison of twelve stain-transfer approaches supports a reasoned choice between them ([stain transfer comparison](https://proceedings.mlr.press/v227/zingman24a.html)).

*The fourth thread works at the level of cells and builds biology into the model.* A mean teacher for cell detection mixes in labelled images to suppress noisy pseudo-labels ([robust mean teacher](https://proceedings.mlr.press/v227/wen24a.html)). Crowd-sourced labels prove valuable for mitosis detection ([data acquisition strategies](https://proceedings.mlr.press/v227/ji24b.html)). Graphs of tumour buds and T-cells over 626 hotspots from 575 patients raise specificity by 20% without losing sensitivity ([tumour budding graphs](https://proceedings.mlr.press/v227/studer24a.html)). Maturation-stage neighbourhoods guide few-shot haematopoietic classification ([RelationVAE](https://proceedings.mlr.press/v227/nguyen24a.html)). Other papers handle neuron tracking in electron microscopy with biology-aware run-length metrics ([II-CATS](https://proceedings.mlr.press/v227/zhai24a.html)), batch effects in high-content screening ([CDCL](https://proceedings.mlr.press/v227/haslum24a.html)), and single-cell images generated from interpretable CellProfiler features ([CP2Image](https://proceedings.mlr.press/v227/ji24a.html)).

*Taken together*, this is a pre-foundation-model pathology programme: weak supervision, explicit invariance and domain-specific structure carry the load that pretrained pathology encoders carry later. Still open: the stain methods are rarely compared with each other on shared multi-centre data, and the confidence scores and prototype explanations are not validated prospectively. It is unclear whether virtual stains preserve diagnostic content beyond the downstream task used to test them.

**Representative papers**

- [Joint Breast Neoplasm Detection and Subtyping using Multi-Resolution Network Trained on Large-Scale H&E Whole Slide Images with Weak Labels](https://proceedings.mlr.press/v227/casson24a.html)
- [Unsupervised Stain Decomposition via Inversion Regulation for Multiplex Immunohistochemistry Images](https://proceedings.mlr.press/v227/abousamra24a.html)
- [Domain adaptation using optimal transport for invariant learning using histopathology datasets](https://proceedings.mlr.press/v227/falahkheirkhah24a.html)
- [Trainable Prototype Enhanced Multiple Instance Learning for Whole Slide Image Classification](https://proceedings.mlr.press/v227/yang24d.html)
- [Tumor Budding T-cell Graphs: Assessing the Need for Resection in pT1 Colorectal Cancer Patients](https://proceedings.mlr.press/v227/studer24a.html)
- [A comparative evaluation of image-to-image translation methods for stain transfer in histopathology](https://proceedings.mlr.press/v227/zingman24a.html)
- [Simple and Efficient Confidence Score for Grading Whole Slide Images](https://proceedings.mlr.press/v227/lubrano24a.html)
- [nnUNet meets pathology: Bridging the gap for application to whole slide images and computational biomarkers](https://proceedings.mlr.press/v227/spronck24a.html)

### Generative models and synthesis

**The ideas behind this group** (a synthesis from the abstracts of about 18 papers in this group; the individual papers have not been read in full).

*The shared problem is changing one factor of an image, such as contrast, stain, resolution, modality or acquisition site, while leaving anatomy untouched, and proving the result is useful.* Generative adversarial networks (GANs) and diffusion models appear in roughly equal numbers. Variational autoencoders and normalising flows fill the gaps. The cost of 3D diffusion is still an obstacle.

*The first thread translates and harmonises with explicit structure guidance.* A diffusion model maps brain MRI between 1.5 T and 3 T contrast and uses the source image to keep anatomy fixed ([diffusion harmonisation](https://proceedings.mlr.press/v227/durrer24a.html)). A GAN with explicit attenuation constraints corrects poorly timed contrast in 1,179 coronary CT angiography scans ([CCTA correction](https://proceedings.mlr.press/v227/herten24a.html)). Translation from T1- to T2-weighted MRI is combined with super-resolution and kept as a distribution, so that several plausible outputs can be sampled ([MTSR-MRI](https://proceedings.mlr.press/v227/dey24a.html)). In pathology, segmentation constraints guide a virtual stain ([MxIF to H&E](https://proceedings.mlr.press/v227/bao24a.html)), and twelve stain-transfer methods are compared head to head ([stain transfer comparison](https://proceedings.mlr.press/v227/zingman24a.html)). Harmonisation can also act on features: a causal normalising flow asks how MRI features would look at another site, raising weighted f1 from 0.36 to 0.68 at an unseen site ([counterfactual harmonisation](https://proceedings.mlr.press/v227/vigneshwaran24a.html)).

*The second thread synthesises data for training, teaching or validation.* Stable diffusion conditioned on text and on another sequence generates prostate MRI with lesions. A radiologist identifies real versus fake with only 59.4% accuracy, and adding the synthetic images raises a lesion classifier from 70.4% to 76.2% ([prostate synthesis](https://proceedings.mlr.press/v227/saeed24a.html)). A 3D GAN inpaints cerebrospinal fluid in deep sulci so that synthetic atrophy has a known ground truth for validating cortical thickness ([CSF inpainting](https://proceedings.mlr.press/v227/wang24a.html)). Virtual cohorts of left ventricles come from a graph-based shape matcher with a β-VAE, which needs no prior correspondence and beats registration-PCA models ([virtual ventricles](https://proceedings.mlr.press/v227/kalaie24a.html)). Hand-crafted CellProfiler features prove sufficient to generate realistic cells ([CP2Image](https://proceedings.mlr.press/v227/ji24a.html)), and class relationships guide few-shot sample synthesis ([RelationVAE](https://proceedings.mlr.press/v227/nguyen24a.html)).

*The third thread uses the generator as a solver.* Diffusion segments images ([MedSegDiff](https://proceedings.mlr.press/v227/wu24a.html)), is made memory-efficient by training on patches ([PatchDDM](https://proceedings.mlr.press/v227/bieder24a.html)), and estimates healthy anatomy patch by patch for anomaly detection, with a relative improvement of 25.1% ([patched diffusion](https://proceedings.mlr.press/v227/behrendt24a.html)). In inverse problems, a conditional GAN approximates posterior sampling for PET reconstruction ([PET posterior](https://proceedings.mlr.press/v227/vlas-ic24a.html)), and amortised normalising flows give calibrated uncertainty for transcranial ultrasound ([amortised flows](https://proceedings.mlr.press/v227/orozco24a.html)). Generators also decode natural images from fMRI ([Cortex2Image](https://proceedings.mlr.press/v227/gu24a.html)) and render view-dependent ultrasound from a neural radiance field ([Ultra-NeRF](https://proceedings.mlr.press/v227/wysocki24a.html)).

*Taken together*, the 2023 generators are mostly conditioned translators, and they are judged by a downstream task such as segmentation, centreline extraction or classification rather than by image statistics. A few papers move towards modelling a distribution of outputs instead of a single image. Still open: anatomical fidelity is enforced by guidance or loss terms but rarely tested directly. Only the prostate study uses a blind expert reader, and none checks for memorisation. Whether sampled outputs are calibrated is shown only for the flow-based reconstruction.

**Representative papers**

- [Bi-parametric prostate MR image synthesis using pathology and sequence-conditioned stable diffusion](https://proceedings.mlr.press/v227/saeed24a.html)
- [Diffusion Models for Contrast Harmonization of Magnetic Resonance Images](https://proceedings.mlr.press/v227/durrer24a.html)
- [Memory-Efficient 3D Denoising Diffusion Models for Medical Image Processing](https://proceedings.mlr.press/v227/bieder24a.html)
- [MTSR-MRI: Combined Modality Translation and Super-Resolution of Magnetic Resonance Images](https://proceedings.mlr.press/v227/dey24a.html)
- [Generative Adversarial Networks for Coronary CT Angiography Acquisition Protocol Correction with Explicit Attenuation Constraints](https://proceedings.mlr.press/v227/herten24a.html)
- [A Geometric Deep Learning Framework for Generation of Virtual Left Ventricles as Graphs](https://proceedings.mlr.press/v227/kalaie24a.html)
- [Improved multi-site Parkinson's disease classification using neuroimaging data with counterfactual inference](https://proceedings.mlr.press/v227/vigneshwaran24a.html)
- [Estimating Uncertainty in PET Image Reconstruction via Deep Posterior Sampling](https://proceedings.mlr.press/v227/vlas-ic24a.html)

### Reconstruction and acquisition

**The ideas behind this group** (a synthesis from the abstracts of about 13 papers in this group; the individual papers have not been read in full).

*The shared problem is that a medical image is an estimate, not a direct measurement.* The raw data may be undersampled k-space, sparse-angle or low-dose projections, or noisy ultrasound and positron emission tomography (PET) signals. The inverse problem is ill-posed. The papers ask four things: how to keep learned reconstruction tied to the measurements, how to report its uncertainty, how to design acquisition together with reconstruction, and how to read or correct acquisition after the scan.

*The first thread builds learned regularisers that stay anchored to the data.* The deep image prior (DIP) overfits noise unless it is stopped early. [SVD-DIP](https://proceedings.mlr.press/v227/nittscher24a.html) restricts fine-tuning of a pretrained DIP to the singular values of its layers, which makes the optimisation stable. A [zero-shot self-supervised method](https://proceedings.mlr.press/v227/zhang24a.html) learns a regulariser in a linear latent space from a single undersampled fast spin-echo (FSE) time series, and estimates coil sensitivity maps at the same time. For motion, a network reconstructs an image given the corrupted k-space and the motion parameters. Correction then becomes a search over rigid motion parameters alone, scored by a data-consistency loss ([data-consistent motion correction](https://proceedings.mlr.press/v227/singh24a.html)). Attention transfers texture from reference data within a single sequence ([texture transformer](https://proceedings.mlr.press/v227/guo24a.html)). For digital breast tomosynthesis (DBT), a classical reconstruction is cleaned of limited-angle artefacts by a network trained on synthetic breast phantoms, because artefact-free real data do not exist ([DBT reconstruction](https://proceedings.mlr.press/v227/quillent24a.html)).

*The second thread treats reconstruction as posterior sampling rather than a single answer.* For PET, a conditional generative adversarial network (GAN) generator approximates sampling from the Bayesian posterior. It is conditioned on a low-dose reconstruction and a magnetic resonance image ([PET posterior sampling](https://proceedings.mlr.press/v227/vlas-ic24a.html)). For transcranial ultrasound computed tomography, amortised normalising flows are trained on a physics-informed summary statistic that compresses the observations. They give fast reconstructions whose uncertainty is calibrated with the error, though so far only in silico ([amortised flows](https://proceedings.mlr.press/v227/orozco24a.html)).

*The third thread co-designs acquisition, reconstruction and the downstream task.* [Multi PILOT](https://proceedings.mlr.press/v227/shor24a.html) extends trajectory learning to dynamic MRI. It jointly learns a feasible k-space trajectory for each frame together with the reconstruction network. [MTLRS](https://proceedings.mlr.press/v227/karkalousos24a.html) couples a recurrent reconstruction network and a segmentation network through hidden states, on data undersampled approximately 7.5 times. Reconstruction and per-lesion segmentation quality correlate ($\rho=0.92$). [SFT-KD-Recon](https://proceedings.mlr.press/v227/matcha24a.html) trains the teacher with the student's structure in view before distillation, and narrows the teacher–student gap from 0.53 dB to 0.03 dB.

*The fourth thread works on acquisition after the fact.* A network predicts image acquisition parameters from a breast MRI in one forward pass. These include field strength, echo and repetition times, scanner model and even contrast agent type, and the goal is to judge whether a model applies to an image ([reverse engineering breast MRIs](https://proceedings.mlr.press/v227/konz24a.html)). A GAN with explicit attenuation constraints corrects poorly timed contrast enhancement in 1,179 coronary computed tomography angiography (CCTA) scans ([CCTA protocol correction](https://proceedings.mlr.press/v227/herten24a.html)).

*Taken together*, data consistency is the common anchor, and uncertainty is starting to appear as a reconstruction output. One paper undercuts the anchor. Deep low-dose computed tomography (CT) reconstructors are vulnerable to untargeted and localised attacks that change how lesions look, and the attacked reconstructions still fit the measurements well ([adversarial low-dose CT](https://proceedings.mlr.press/v227/gandikota24a.html)). Still open: data consistency alone does not certify a reconstruction; the posterior-sampling methods are validated in silico or on single datasets, with no shared test of calibration; and learned acquisition has not yet been combined with posterior uncertainty.

**Representative papers**

- [SVD-DIP: Overcoming the Overfitting Problem in DIP-based CT Reconstruction](https://proceedings.mlr.press/v227/nittscher24a.html)
- [Estimating Uncertainty in PET Image Reconstruction via Deep Posterior Sampling](https://proceedings.mlr.press/v227/vlas-ic24a.html)
- [Amortized Normalizing Flows for Transcranial Ultrasound with Uncertainty Quantification](https://proceedings.mlr.press/v227/orozco24a.html)
- [Data Consistent Deep Rigid MRI Motion Correction](https://proceedings.mlr.press/v227/singh24a.html)
- [Multi PILOT: Feasible Learned Multiple Acquisition Trajectories For Dynamic MRI](https://proceedings.mlr.press/v227/shor24a.html)
- [Evaluating Adversarial Robustness of Low dose CT Recovery](https://proceedings.mlr.press/v227/gandikota24a.html)
- [Reverse Engineering Breast MRIs: Predicting Acquisition Parameters Directly from Images](https://proceedings.mlr.press/v227/konz24a.html)
- [Zero-Shot Self-Supervised Joint Temporal Image and Sensitivity Map Reconstruction via Linear Latent Space](https://proceedings.mlr.press/v227/zhang24a.html)

### Registration, geometry and equivariance

**The ideas behind this group** (a synthesis from the abstracts of about 16 papers in this group; the individual papers have not been read in full).

*The shared problem is that anatomy has geometry (shape, correspondence, symmetry) that pixel-grid networks do not represent.* The papers write that geometry into the model in four ways: as priors on deformations, as explicit shape representations, as built-in symmetries, and as physical models of how the image forms.

*The first thread constrains deformation with anatomical or physical priors.* For tagged MRI of the tongue, a loss based on the Jacobian determinant encourages incompressible, approximately diffeomorphic motion fields ([DRIMET](https://proceedings.mlr.press/v227/bian24a.html)). An implicit neural representation (INR) of bowel motion uses a coordinate system curved along the centrelines of the small intestine, so that the coordinates follow the expected motion ([geometry-informed INR](https://proceedings.mlr.press/v227/harten24a.html)). Cortical registration aligns function as well as folding and learns an atlas at the same time. Training is semi-supervised, so functional data are not needed at inference ([JOSA](https://proceedings.mlr.press/v227/li24b.html)). Two papers find correspondence without a dedicated registration step. In one, a template-based graph neural network (GNN) segmenter keeps vertex correspondence and gives target registration errors of 2.2$\pm$1.3 mm and 2.7$\pm$1.4 mm on vertebrae ([segmentation in lieu of registration](https://proceedings.mlr.press/v227/li24c.html)). In the other, segmentation and registration regularise each other to fuse unaligned low-resolution views into high-resolution masks ([SuperMask](https://proceedings.mlr.press/v227/gu24b.html)). For surgery, a learned mesh-recovery network moves the computational cost to before the operation, so that registration in liver resection runs in real time ([liver mesh recovery](https://proceedings.mlr.press/v227/labrunie24a.html)).

*The second thread represents shape directly as graphs, meshes or point models.* Attention-based graph matching combined with a $\beta$-VAE generates virtual left-ventricle cohorts without first establishing correspondence ([virtual left ventricles](https://proceedings.mlr.press/v227/kalaie24a.html)). Statistical shape models are estimated from unsegmented 3D images, including rigid pose ([Image2SSM](https://proceedings.mlr.press/v227/ukey24a.html)). A radius token yields complete 3D vessel graphs ([Vesselformer](https://proceedings.mlr.press/v227/prabhakar24a.html)). Regional pooling on brain meshes stages mild cognitive impairment ([RegionPool](https://proceedings.mlr.press/v227/guo24b.html)). A coarse-to-fine model uses the fact that physicians annotate correlated landmarks on the same slice ([prior-guided landmarks](https://proceedings.mlr.press/v227/pang24a.html)).

*The third thread builds symmetry into the layers.* For diffusion MRI (dMRI), layers equivariant to $E(3)\times SO(3)$ respect both spatial roto-translations and voxel-wise spherical rotations in sparse deconvolution ([RT-ESD](https://proceedings.mlr.press/v227/elaldi24a.html)). Steerable filters built from Gaussian derivatives, with trainable scales, combine rotation and scale equivariance for histology. They use fewer parameters ([RSESF](https://proceedings.mlr.press/v227/yang24a.html)).

*The fourth thread puts geometry into how the image forms.* A neural radiance field with physics-inspired ray tracing models view-dependent ultrasound appearance ([Ultra-NeRF](https://proceedings.mlr.press/v227/wysocki24a.html)). A convolutional-recurrent network learns streamline tractography from T1-weighted MRI alone. Its measures differ by approximately 5–15% from dMRI tractography, which the authors place at the level of scan–rescan variability ([tractography from T1w](https://proceedings.mlr.press/v227/cai24a.html)). Synthetic atrophy with inpainted cerebrospinal fluid gives ground truth for validating cortical thickness measurement ([CSF inpainting](https://proceedings.mlr.press/v227/wang24a.html)).

*Taken together*, the group uses hand-chosen geometric priors as an inductive bias in place of data. Still open: each prior is designed for one anatomy, and it is unclear how to choose or learn one in general; the equivariant models are tested on narrow tasks without comparison to augmentation at equal cost; and the tractography result asks whether tractography is really a microstructural phenomenon, a question this group does not answer.

**Representative papers**

- [E(3) x SO(3)-Equivariant Networks for Spherical Deconvolution in Diffusion MRI](https://proceedings.mlr.press/v227/elaldi24a.html)
- [DRIMET: Deep Registration-based 3D Incompressible Motion Estimation in Tagged-MRI with Application to the Tongue](https://proceedings.mlr.press/v227/bian24a.html)
- [Deformable Image Registration with Geometry-informed Implicit Neural Representations](https://proceedings.mlr.press/v227/harten24a.html)
- [Joint cortical registration of geometry and function using semi-supervised learning](https://proceedings.mlr.press/v227/li24b.html)
- [Spatial Correspondence between Graph Neural Network-Segmented Images](https://proceedings.mlr.press/v227/li24c.html)
- [Rotation-Scale Equivariant Steerable Filters](https://proceedings.mlr.press/v227/yang24a.html)
- [A Geometric Deep Learning Framework for Generation of Virtual Left Ventricles as Graphs](https://proceedings.mlr.press/v227/kalaie24a.html)
- [Convolutional-recurrent neural networks approximate diffusion tractography from T1-weighted MRI and associated anatomical context](https://proceedings.mlr.press/v227/cai24a.html)

### Trustworthiness

**The ideas behind this group** (a synthesis from the abstracts of about 12 papers in this group; the individual papers have not been read in full).

*The shared problem is that a model can be accurate on average and still be untrustworthy.* Its reasoning may be hidden, its confidence miscalibrated, its errors concentrated in particular subgroups, or its output easy to manipulate. The papers take these four failures in turn. Several of them find that fixing one failure makes another worse.

*The first thread makes models interpretable by design instead of explaining them after the fact.* A bag-of-local-features network (BagNet) with a sparsity constraint gives class evidence maps that, by construction, reflect the decision for diabetic retinopathy grading ([sparse BagNet](https://proceedings.mlr.press/v227/donteu24a.html)). [Attri-Net](https://proceedings.mlr.press/v227/sun24a.html) generates class-specific counterfactual attribution maps and classifies with logistic regression on those maps alone. It targets the multi-label case, where the authors show post-hoc methods do poorly. Case-based prototype networks are extended to 3D multi-parametric MRI, with a best activation precision of $0.713\pm0.058$ ([MProtoNet](https://proceedings.mlr.press/v227/wei24a.html)). In pathology, distances to trained prototypes replace attention scores as explanations ([TPMIL](https://proceedings.mlr.press/v227/yang24d.html)). Attention maps are checked against a pathologist's descriptions ([IBD biopsies](https://proceedings.mlr.press/v227/mokhtari24a.html)).

*The second thread makes confidence mean something.* A confidence score for ordinal grading needs no extra training or inference. The AUC gap between high- and low-confidence slides is 17.1% ([ordinal confidence](https://proceedings.mlr.press/v227/lubrano24a.html)). Graph neural networks (GNNs) tend to be underconfident rather than overconfident. Platt or temperature scaling still helps them, and helps more for multiclass than for binary tasks ([GNN calibration](https://proceedings.mlr.press/v227/vos24a.html)). For out-of-distribution (OOD) detectors, the space in which inliers and outliers are synthesised matters as much as the augmentation. Latent-space inliers with diverse pixel-space outliers give 15%–35% gains in AUROC ([know your space](https://proceedings.mlr.press/v227/narayanaswamy24a.html)).

*The third thread deals with bias and shortcuts.* A DenseNet-121 classifies chest X-ray projection better than any of fourteen findings, without being trained to. Label-conditional gradient reversal removes that signal ([projection bias](https://proceedings.mlr.press/v227/pereira24a.html)). Popular fairness methods such as data balancing and distributionally robust optimisation can close performance gaps between groups, but at the cost of worse uncertainty estimates ([fairness of uncertainty](https://proceedings.mlr.press/v227/mehta24a.html)). A causal normalising flow asks how features would look if acquired at another site. On an unseen site it reaches weighted F1 = 0.68, against 0.36 without it and 0.5 for standard harmonisation ([counterfactual harmonisation](https://proceedings.mlr.press/v227/vigneshwaran24a.html)).

*The fourth thread probes adversarial robustness.* In low-dose CT, small localised perturbations change how lesions look while the result still fits the measurements ([adversarial CT recovery](https://proceedings.mlr.press/v227/gandikota24a.html)).

*Taken together*, the group prefers mechanisms built into the model (sparse evidence, prototypes, counterfactual maps) to post-hoc saliency, and it begins to measure trade-offs between properties rather than one property at a time. Still open: interpretability is judged by localisation metrics or expert agreement rather than by whether it helps decisions; the fairness–uncertainty trade-off is documented but not resolved; and calibration, OOD detection and fairness are not evaluated together on shared benchmarks.

**Representative papers**

- [Inherently Interpretable Multi-Label Classification Using Class-Specific Counterfactuals](https://proceedings.mlr.press/v227/sun24a.html)
- [Sparse Activations for Interpretable Disease Grading](https://proceedings.mlr.press/v227/donteu24a.html)
- [Know Your Space: Inlier and Outlier Construction for Calibrating Medical OOD Detectors](https://proceedings.mlr.press/v227/narayanaswamy24a.html)
- [Evaluating the Fairness of Deep Learning Uncertainty Estimates in Medical Image Analysis](https://proceedings.mlr.press/v227/mehta24a.html)
- [Addressing Chest Radiograph Projection Bias in Deep Classification Models](https://proceedings.mlr.press/v227/pereira24a.html)
- [Calibration techniques for node classification using graph neural networks on medical image data](https://proceedings.mlr.press/v227/vos24a.html)
- [MProtoNet: A Case-Based Interpretable Model for Brain Tumor Classification with 3D Multi-parametric Magnetic Resonance Imaging](https://proceedings.mlr.press/v227/wei24a.html)
- [Improved multi-site Parkinson’s disease classification using neuroimaging data with counterfactual inference](https://proceedings.mlr.press/v227/vigneshwaran24a.html)

### Unsupervised anomaly detection

**The ideas behind this group** (a synthesis from the abstracts of about 4 papers in this group; the individual papers have not been read in full).

*The shared problem is to find pathology using only healthy, or mostly healthy, training data.* The usual approach models the normal distribution, then flags whatever the model cannot reproduce. The papers question both halves of that recipe.

*The first thread repairs the reconstruction model.* Autoencoders (AEs) can reconstruct some anomalies better than healthy samples. In some settings, simple intensity thresholding also beats them on hyperintense lesions in fluid-attenuated inversion recovery (FLAIR) MRI. Reversed autoencoders instead produce pseudo-healthy reconstructions and target non-hyperintense T1-weighted findings such as atrophy, oedema and resections. They raise global detection from 73.1 to 89.4 AUROC and local detection from 52.6% to 86.0% ([reversed autoencoders](https://proceedings.mlr.press/v227/bercea24a.html)). Diffusion models are recast as patch-wise estimators of healthy anatomy that use the surrounding context, with a relative improvement of 25.1% on tumours and multiple sclerosis lesions ([patched diffusion](https://proceedings.mlr.press/v227/behrendt24a.html)).

*The second thread replaces reconstruction error with outlier detection in a learned space.* A siamese patch autoencoder builds a latent space, and a one-class support vector machine (SVM) detects white matter hyperintensities, on par with the two best methods ([one-class SVM](https://proceedings.mlr.press/v227/pinon24a.html)). In pathology, reverse knowledge distillation with a focal loss on representation similarity localises metastases. It matches prior work with less than ten percent of the training samples on Camelyon16 ([reverse distillation for WSIs](https://proceedings.mlr.press/v227/he24a.html)).

*Taken together*, the group moves from "reconstruct and subtract" towards models that preserve healthy context or score anomalies in feature space. Still open: evaluation remains centred on brain MRI lesions, and the thresholding baseline that [bercea24a](https://proceedings.mlr.press/v227/bercea24a.html) mentions is rarely reported; detection of subtle, non-hyperintense or global change is only starting to be tested.

**Representative papers**

- [Generalizing Unsupervised Anomaly Detection: Towards Unbiased Pathology Screening](https://proceedings.mlr.press/v227/bercea24a.html)
- [Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI](https://proceedings.mlr.press/v227/behrendt24a.html)
- [One-Class SVM on siamese neural network latent space for Unsupervised Anomaly Detection on brain MRI White Matter Hyperintensities](https://proceedings.mlr.press/v227/pinon24a.html)
- [Whole-slide-imaging Cancer Metastases Detection and Localization with Limited Tumorous Data](https://proceedings.mlr.press/v227/he24a.html)

### Brain networks and neuroimaging

**The ideas behind this group** (a synthesis from the abstracts of about 10 papers in this group; the individual papers have not been read in full).

*The shared problem is that brain data are noisy, multimodal and structured, and the structure is usually fixed by hand before learning starts.* Functional MRI (fMRI) has a low signal-to-noise ratio. Connectivity graphs depend on the chosen connectivity measure. Modalities are often missing, and sites differ. The papers either learn the structure or carry it over into learned representations.

*The first thread learns brain graphs that change over time instead of assuming a fixed one.* Most GNNs on fMRI take a static, known adjacency matrix. [DBGSL](https://proceedings.mlr.press/v227/campbell24a.html) learns the time-varying dependency structure needed for a downstream prediction. [DBGDGM](https://proceedings.mlr.press/v227/campbell24b.html) is generative: it learns node embeddings drawn from communities that evolve over time. In both papers, the learnt structure is checked against known functional connectivity networks and the literature.

*The second thread learns representations from limited or incomplete subject data.* Pretraining a transformer to fingerprint subjects with a triplet loss is the key step for predicting stress from fMRI ([MetricFMRI](https://proceedings.mlr.press/v227/rosenman24a.html)). Modal-mixup synthesises complete samples from incomplete ones for multi-modal connectome diagnosis, with deep supervision to limit noise from unpaired modalities ([incomplete connectomes](https://proceedings.mlr.press/v227/yang24b.html)). A causal normalising flow produces counterfactual site features across six sites and 415 subjects for Parkinson's disease classification ([counterfactual harmonisation](https://proceedings.mlr.press/v227/vigneshwaran24a.html)).

*The third thread works on the cortical surface and white matter.* A surface-based convolutional network maps cortical responses to semantic and then fine-grained image features, decoding natural images seen by the subject ([Cortex2Image](https://proceedings.mlr.press/v227/gu24a.html)). Registration aligns function and folding together ([JOSA](https://proceedings.mlr.press/v227/li24b.html)). Mesh-based regional pooling stages mild cognitive impairment. It also suggests that current empirically derived subtypes miss heterogeneous atrophy patterns ([RegionPool](https://proceedings.mlr.press/v227/guo24b.html)). For white matter, equivariant spherical deconvolution recovers crossing fibres ([RT-ESD](https://proceedings.mlr.press/v227/elaldi24a.html)), and tractography is approximated from T1-weighted MRI ([CoRNN](https://proceedings.mlr.press/v227/cai24a.html)).

*Taken together*, the common move is to make structure learnable, whether graphs, communities, embeddings or correspondences, and to validate it by agreement with known neuroscience. Still open: agreement with prior literature is a weak test of whether learned structure is correct; benchmark tasks such as sex classification say little about clinical use; and multi-site and missing-modality problems are handled separately, not together.

**Representative papers**

- [DBGSL: Dynamic Brain Graph Structure Learning](https://proceedings.mlr.press/v227/campbell24a.html)
- [DBGDGM: Dynamic Brain Graph Deep Generative Model](https://proceedings.mlr.press/v227/campbell24b.html)
- [Decoding natural image stimuli from fMRI data with a surface-based convolutional network](https://proceedings.mlr.press/v227/gu24a.html)
- [Pre-Training Transformers for Fingerprinting to Improve Stress Prediction in fMRI](https://proceedings.mlr.press/v227/rosenman24a.html)
- [Incomplete learning of multi-modal connectome for brain disorder diagnosis via modal-mixup and deep supervision](https://proceedings.mlr.press/v227/yang24b.html)
- [Stage Detection of Mild Cognitive Impairment: Region-dependent Graph Representation Learning on Brain Morphable Meshes](https://proceedings.mlr.press/v227/guo24b.html)

### Vision-language models and reports

**The ideas behind this group** (a synthesis from the abstracts of about 7 papers in this group; the individual papers have not been read in full).

*The shared problem is that hospitals have plenty of images paired with text but few labelled examples, and the text is unstructured.* Chest X-rays come with radiology reports, electrocardiograms (ECGs) with machine-generated statements, and transplant patients with tables of clinical variables. The papers here ask how to turn that text into supervision for an image encoder, and how to get text back out in a form clinicians can check. The work predates large language models (LLMs) as the main tool. Most papers rely on contrastive image–text embedding in the style of Contrastive Language–Image Pretraining (CLIP).

*The first thread asks how to train vision–language models (VLMs) when paired data are scarce.* [Low-data VLMs](https://proceedings.mlr.press/v227/windsor24a.html) combine three ingredients: unimodal self-supervision to adapt generic encoders to medical images and reports, local and global contrastive losses, and extra positive pairs from augmentation and nearest-neighbour search. Together they improve text-to-image retrieval about as much as training with 10× the data would. They also beat CLIP and BioViL in zero-shot and linear-probe classification. [Reports improve visual representations](https://proceedings.mlr.press/v227/huang24a.html) compares multimodal, self-supervised and joint learning for chest radiograph classification. With 1% and 10% labelled data, joint learning beats self-supervision and matches multimodal learning, and multimodal learning holds up better out of distribution.

*The second thread uses language embeddings as an open label space, which allows zero-shot and few-shot prediction.* [METS](https://proceedings.mlr.press/v227/li24a.html) pairs a trainable ECG encoder with a frozen language model that embeds automatically generated reports. Without annotated data, its zero-shot classification improves by around 10% on supervised and self-supervised baselines, including on a dataset whose classes differ from pretraining. [FlexR](https://proceedings.mlr.press/v227/keicher24a.html) turns each sentence of a structured reporting template into a text prompt and trains a classifier on a contrastive model. This fills templates for cardiomegaly severity and pathology localisation from few image-level labels. [MEDIMP](https://proceedings.mlr.press/v227/milecki24a.html) converts tabular clinical and biological data into text prompts, augmented by LLMs, and aligns them contrastively with dynamic contrast-enhanced MRI of renal transplants.

*The third thread generates one modality from the other.* [X-REM](https://proceedings.mlr.press/v227/jeong24a.html) generates chest X-ray reports by retrieval. It ranks candidate reports with an image–text matching score rather than cosine similarity, which captures finer interactions. It improves both language and clinical metrics, and human readers found more zero-error reports. Going the other way, [prostate MR synthesis](https://proceedings.mlr.press/v227/saeed24a.html) conditions Stable Diffusion on text to control lesion presence and sequence. A radiologist told real from synthetic images with only 59.4% accuracy, where chance is 50%.

*Taken together*, text serves here as cheap, flexible supervision and as an open vocabulary of labels. Retrieval and templates are preferred to free generation because they keep the output clinically grounded. Still open: evaluating generated text beyond language-overlap scores, checking the auto-generated or LLM-augmented text that serves as supervision, and testing whether the low-data recipes transfer beyond chest X-rays and ECGs to other modalities and to 3D.

**Representative papers**

- [Vision-Language Modelling For Radiological Imaging and Reports In The Low Data Regime](https://proceedings.mlr.press/v227/windsor24a.html)
- [Frozen Language Model Helps ECG Zero-Shot Learning](https://proceedings.mlr.press/v227/li24a.html)
- [Multimodal Image-Text Matching Improves Retrieval-based Chest X-Ray Report Generation](https://proceedings.mlr.press/v227/jeong24a.html)
- [FlexR: Few-shot Classification with Language Embeddings for Structured Reporting of Chest X-rays](https://proceedings.mlr.press/v227/keicher24a.html)
- [Radiology Reports Improve Visual Representations Learned from Radiographs](https://proceedings.mlr.press/v227/huang24a.html)
- [MEDIMP: 3D Medical Images and clinical Prompts for renal transplant representation learning](https://proceedings.mlr.press/v227/milecki24a.html)

### Domain shift, federated and continual learning

**The ideas behind this group** (a synthesis from the abstracts of about 16 papers in this group; the individual papers have not been read in full).

*The shared problem is that a model trained at one site, on one scanner or with one stain fails elsewhere, and the data needed to fix it usually cannot be pooled or relabelled.* Shift has several sources: scanner and field strength, acquisition parameters, staining protocol, laboratory batch effects, and differences between institutions. Privacy rules often prevent source data from leaving its site. The papers attack the problem at four levels: the input, the training labels, the network's internals, and the training protocol across sites or over time.

*The first thread removes the shift before the model sees it.* [Intensity standardisation](https://proceedings.mlr.press/v227/ghazvanchahi24a.html) compares several normalisation methods as preprocessing for white matter lesion segmentation in multi-centre fluid-attenuated inversion recovery (FLAIR) MRI. A FLAIR-specific method (IAMLAB) and an ensemble do best on 128 unseen volumes. [Diffusion harmonisation](https://proceedings.mlr.press/v227/durrer24a.html) maps contrast between 1.5 T and 3 T scans with a diffusion model guided by the original image, so that anatomy is preserved. [Counterfactual harmonisation](https://proceedings.mlr.press/v227/vigneshwaran24a.html) works on features rather than pixels. A causal normalising flow asks how brain MRI features would look at another site, and this raises weighted F1 on an unseen site to 0.68, against 0.36 without harmonisation. [Reverse engineering breast MRIs](https://proceedings.mlr.press/v227/konz24a.html) predicts acquisition parameters from the image alone, which turns hidden shift into something that can be measured.

*The second thread adapts to the target without target annotations.* Two papers share one recipe for [Ki-67 scoring](https://proceedings.mlr.press/v227/dy24a.html) and [lateral ventricle segmentation](https://proceedings.mlr.press/v227/crystal24a.html). They pretrain on noisy silver-standard labels generated automatically in the target domain, then fine-tune on gold-standard source labels. This order beats the reverse order and beats source-only training. [MATS](https://proceedings.mlr.press/v227/chen24b.html) adapts a tooth segmentation model without access to source data, using prototypes and progressive pseudo-labels. [On-the-fly adaptation](https://proceedings.mlr.press/v227/valanarasu24a.html) adapts to one test image at a time without back-propagation, through batch normalisation conditioned on a domain code. Other papers align feature distributions: an [optimal transport loss](https://proceedings.mlr.press/v227/falahkheirkhah24a.html) compares batches rather than single samples and keeps a rare cancer phenotype unseen in training. [Channel attention](https://proceedings.mlr.press/v227/abutalip24a.html) targets early-layer style covariances for stain-adversarial training. [Consistency learning](https://proceedings.mlr.press/v227/haslum24a.html) suppresses batch effects in high-content screening.

*The third thread finds that the shift lives partly in the architecture.* [Normalisation under shift](https://proceedings.mlr.press/v227/madaan24a.html) identifies batch normalisation as a major cause of degradation under MRI artefacts, and group or layer normalisation as more robust. [TransNetR](https://proceedings.mlr.press/v227/jha24a.html) makes out-of-distribution testing on multi-centre colonoscopy data part of the evaluation protocol.

*The fourth thread trains across sites or over time.* [FedCross](https://proceedings.mlr.press/v227/xu24a.html) (annotated in [this repo](../2023-xu-federated-cross-learning/index.html)) starts from a theoretical analysis of aggregation on non-independent and identically distributed (non-IID) data. It then removes aggregation altogether, training one global model in turn across clients. [Radiomics-based personalisation](https://proceedings.mlr.press/v227/manthe24a.html) clusters scans by global texture to handle feature shift both between and within institutions. For lifelong learning, [coreset compression](https://proceedings.mlr.press/v227/zheng24a.html) shrinks experience replay buffers by 10x with no significant loss (p=0.28) on whole-body landmarks.

*Taken together*, the group treats shift as something to measure and engineer around, at every stage, rather than as a single domain-adaptation problem. Silver-standard pretraining and cheap per-site calibration stand out as practical recipes. Still open: theory for when pseudo-labels from the target domain help rather than hurt, comparisons across these levels on shared benchmarks, and federated methods tested on real rather than simulated heterogeneity.

**Representative papers**

- [Federated Cross Learning for Medical Image Segmentation](https://proceedings.mlr.press/v227/xu24a.html)
- [On-the-Fly Test-time Adaptation for Medical Image Segmentation](https://proceedings.mlr.press/v227/valanarasu24a.html)
- [Domain Adaptation using Silver Standard Labels for Ki-67 Scoring in Digital Pathology A Step Closer to Widescale Deployment](https://proceedings.mlr.press/v227/dy24a.html)
- [Domain adaptation using optimal transport for invariant learning using histopathology datasets](https://proceedings.mlr.press/v227/falahkheirkhah24a.html)
- [On Sensitivity and Robustness of Normalization Schemes to Input Distribution Shifts in Automatic MR Image Diagnosis](https://proceedings.mlr.press/v227/madaan24a.html)
- [Improved multi-site Parkinson's disease classification using neuroimaging data with counterfactual inference](https://proceedings.mlr.press/v227/vigneshwaran24a.html)
- [Reverse Engineering Breast MRIs: Predicting Acquisition Parameters Directly from Images](https://proceedings.mlr.press/v227/konz24a.html)
- [Selective experience replay compression using coresets for lifelong deep reinforcement learning in medical imaging](https://proceedings.mlr.press/v227/zheng24a.html)

### Multimodal and longitudinal clinical prediction

**The ideas behind this group** (a synthesis from the abstracts of about 7 papers in this group; the individual papers have not been read in full).

*The shared problem is predicting what happens to a patient, not just what an image shows.* The targets are recurrence, survival, mortality, graft outcome, need for surgery or change over time. These depend on more than one scan: on clinical variables, earlier examinations and, sometimes, modalities that are missing for some patients. Outcome labels are expensive because they need follow-up, so datasets are small.

*The first thread fuses images with tabular clinical data and asks how much the fusion design matters.* For prostate cancer in 3800 subjects, [fused MRI and tabular data](https://proceedings.mlr.press/v227/bashkanov24a.html) raises receiver operating characteristic area under the curve (ROC AUC) from 0.736 to 0.765 with demographics. Naïve concatenation performs as well as, or better than, state-of-the-art fusion modules. Adding the last three prostate-specific antigen values lifts AUC to 0.794 and quadratic weighted kappa from 0.342 to 0.464. [TransRP](https://proceedings.mlr.press/v227/ma24a.html) predicts recurrence-free survival in oropharyngeal cancer from PET/CT and clinical data, reaching a test concordance index of 0.698. Here too, simpler late fusion of clinical features works better than feeding them through the Transformer. [MEDIMP](https://proceedings.mlr.press/v227/milecki24a.html) rewrites clinical variables as text prompts and aligns them contrastively with renal transplant MRI, to support prognosis at 2, 3 and 4 years.

*The second thread uses time itself as supervision.* [PaIRNet](https://proceedings.mlr.press/v227/kim24a.html) learns to compare pairs of longitudinal images. In its self-supervised form it is trained to put scans in temporal order, which forces it to recognise irreversible changes while ignoring nuisance variation. This replaces hand-built preprocessing and statistical modelling, and it is tested on four datasets.

*The third thread works around scarce outcome labels and missing inputs.* [Self-supervised pretraining for risk](https://proceedings.mlr.press/v227/chen24c.html) compares approaches for predicting mortality from chest X-rays and finds that a semi-supervised autoencoder beats contrastive and transfer learning in both internal and external validation. For brain disorders, [modal-mixup](https://proceedings.mlr.press/v227/yang24b.html) synthesises complete multimodal connectomes from incomplete samples, so that single-modality data are not discarded. [Tumour budding T-cell graphs](https://proceedings.mlr.press/v227/studer24a.html) encode the spatial layout of cells as a graph to decide whether patients with pT1 colorectal cancer need resection. They raise specificity by 20% over current risk factors without losing sensitivity, on 626 hotspots from 575 patients.

*Taken together*, the clinical variables carry much of the signal. Fusion architecture matters less than which variables, and how much history, are included. Still open: principled fusion that beats concatenation, handling inputs that are irregularly sampled or missing in time, and external validation of prognostic claims, since most studies use a single cohort or challenge dataset.

**Representative papers**

- [Automatic Patient-level Diagnosis of Prostate Disease with Fused 3D MRI and Tabular Clinical Data](https://proceedings.mlr.press/v227/bashkanov24a.html)
- [TransRP: Transformer-based PET/CT feature extraction incorporating clinical data for recurrence-free survival prediction in oropharyngeal cancer](https://proceedings.mlr.press/v227/ma24a.html)
- [Learning to Compare Longitudinal Images](https://proceedings.mlr.press/v227/kim24a.html)
- [A comparison of self-supervised pretraining approaches for predicting disease risk from chest radiograph images](https://proceedings.mlr.press/v227/chen24c.html)
- [Tumor Budding T-cell Graphs: Assessing the Need for Resection in pT1 Colorectal Cancer Patients](https://proceedings.mlr.press/v227/studer24a.html)
- [Incomplete learning of multi-modal connectome for brain disorder diagnosis via modal-mixup and deep supervision](https://proceedings.mlr.press/v227/yang24b.html)

### Surgery, video and ultrasound

**The ideas behind this group** (a synthesis from the abstracts of about 8 papers in this group; the individual papers have not been read in full).

*The shared problem is that interventional and ultrasound imaging is real-time, depends on the operator and is governed by physics that standard vision models ignore.* Surgery needs answers within the procedure. Ultrasound images change with probe direction, and fluorescence video is captured at very low light. Labelled data are scarce, so several papers build structure into the model rather than learning it from many examples.

*The first thread puts imaging physics into the model.* [Ultra-NeRF](https://proceedings.mlr.press/v227/wysocki24a.html) extends implicit neural representations (INRs) to ultrasound with ray-tracing-based rendering. It models how tissue appearance depends on viewing direction, so that synthesised B-mode frames from unseen views give consistent volume compounding. For transcranial ultrasound computed tomography, [amortised normalising flows](https://proceedings.mlr.press/v227/orozco24a.html) compress raw observations with a physics-informed summary statistic. A normalising flow then gives fast reconstruction with calibrated Bayesian uncertainty across transducer configurations, so far only in silico.

*The second thread borrows information from a second, easier source.* [OFDVDnet](https://proceedings.mlr.press/v227/seets24a.html) denoises low-radiance fluorescence video in fluorescence-guided surgery using motion and texture from a simultaneously captured, noise-free visible-light video. [SegPrompt](https://proceedings.mlr.press/v227/zhu24b.html) feeds segmentation maps into an endoscopic kidney-stone classifier as prompts. This makes it attend to the stone and reduces trainable parameters when data are limited.

*The third thread reasons about objects and geometry rather than pixels.* [ST(OR)²](https://proceedings.mlr.press/v227/hamoud24a.html) recognises operating room activity from the spatial arrangement and interactions of clinicians and devices over time. It is sample-efficient for long videos in a low-data setting. [Liver Mesh Recovery](https://proceedings.mlr.press/v227/labrunie24a.html) does 3D/2D deformable registration for augmented reality in minimally invasive liver resection. It moves the computational cost to preoperative, patient-specific training, which matches optimisation-based accuracy while running in real time.

*The fourth thread targets deployment: speed and knowing when outputs are bad.* [TransNetR](https://proceedings.mlr.press/v227/jha24a.html) segments colonoscopy polyps in real time (a Dice coefficient of 0.8706 on Kvasir-SEG) and is tested out of distribution on six centres. [FUSQA](https://proceedings.mlr.press/v227/cengiz24a.html) recasts segmentation quality assessment in fetal ultrasound as classifying masks as good or poor, with over 90% accuracy on an unseen hospital's data. The clinical stakes are concrete. Gestational age from crown–rump length differs from doctors' estimates by 1.45 days with good masks and by up to 7.73 days with poor ones.

*Taken together*, the group relies on physics, auxiliary signals and object-level structure where labels are missing, and treats real-time operation and failure detection as requirements. Still open: validation beyond simulation and phantoms (Ultra-NeRF, transcranial reconstruction), uncertainty that propagates from imaging to downstream measurements, and multi-site surgical datasets.

**Representative papers**

- [Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging](https://proceedings.mlr.press/v227/wysocki24a.html)
- [Amortized Normalizing Flows for Transcranial Ultrasound with Uncertainty Quantification](https://proceedings.mlr.press/v227/orozco24a.html)
- [ST(OR)$^2$: Spatio-Temporal Object Level Reasoning for Activity Recognition in the Operating Room](https://proceedings.mlr.press/v227/hamoud24a.html)
- [OFDVDnet: A Sensor Fusion Approach for Video Denoising in Fluorescence-Guided Surgery](https://proceedings.mlr.press/v227/seets24a.html)
- [Automatic 3D/2D Deformable Registration in Minimally Invasive Liver Resection using a Mesh Recovery Network](https://proceedings.mlr.press/v227/labrunie24a.html)
- [FUSQA: Fetal Ultrasound Segmentation Quality Assessment](https://proceedings.mlr.press/v227/cengiz24a.html)

### Benchmarks, reproducibility and evaluation

**The ideas behind this group** (a synthesis from the abstracts of about 16 papers in this group; the individual papers have not been read in full).

*The shared problem is whether reported gains are real, reproducible and measured in a way that matters clinically.* These papers do not mainly propose models. They retrain other people's models, audit code, compare many methods under one protocol, or design metrics. Several of their findings question how the field, and this conference, reports results.

*The first thread is about reproducibility.* [Reproducibility of MIDL](https://proceedings.mlr.press/v227/simko24a.html) audits every accepted Medical Imaging with Deep Learning (MIDL) full paper from 2018 to 2022. Merely 22% of submissions had a repository judged repeatable, and the authors propose guidelines. [Reproducibility of training](https://proceedings.mlr.press/v227/bosma24a.html) retrains published top algorithms for five tasks several times. For all three diagnostic tasks, variation from retraining is significant compared with data variance. Comparing runs of the same method on the same data, which should follow the null hypothesis, gave p below 0.05 in 15% of comparisons under paired bootstrapping. Leaderboards and significance claims that ignore training randomness are therefore suspect.

*The second thread designs metrics that fit clinical needs.* [RoDeO](https://proceedings.mlr.press/v227/meissen24a.html) replaces natural-image detection metrics for chest X-ray pathology with a score that assesses different error types separately. In connectomics, [II-CATS](https://proceedings.mlr.press/v227/zhai24a.html) extends expected run length with biologically constrained variants. [Fairness of uncertainty](https://proceedings.mlr.press/v227/mehta24a.html) adds a dimension to fairness evaluation. Data balancing and distributionally robust optimisation can close performance gaps between subgroups but worsen uncertainty estimates.

*The third thread runs systematic comparisons that often deflate assumptions.* Twelve [stain-transfer methods](https://proceedings.mlr.press/v227/zingman24a.html), three traditional and nine based on generative adversarial networks (GANs), are compared on image quality, downstream grading and pathologist review. Natural video pretraining helps seven 3D models on two chest CT datasets, and it beats in-domain CT pretraining ([video pretraining](https://proceedings.mlr.press/v227/ke24a.html)). Augmentations for Siamese learning on chest X-rays are assessed one by one ([augmentation study](https://proceedings.mlr.press/v227/sluijs24a.html)). Calibration methods designed for convolutional networks are tested on underconfident graph neural networks ([GNN calibration](https://proceedings.mlr.press/v227/vos24a.html)). Deep networks for low-dose CT recovery, including data-consistent ones, are more vulnerable to adversarial attacks than classical methods ([adversarial robustness](https://proceedings.mlr.press/v227/gandikota24a.html)). Simple choices often win: naïve concatenation for fusing images and tables ([prostate fusion](https://proceedings.mlr.press/v227/bashkanov24a.html)), group or layer normalisation over batch normalisation ([normalisation under shift](https://proceedings.mlr.press/v227/madaan24a.html)), and a semi-supervised autoencoder for risk prediction ([pretraining for risk](https://proceedings.mlr.press/v227/chen24c.html)). Intensity standardisation methods are likewise benchmarked across centres ([intensity standardisation](https://proceedings.mlr.press/v227/ghazvanchahi24a.html)).

*The fourth thread studies how data are acquired.* For [mitosis detection](https://proceedings.mlr.press/v227/ji24b.html), crowd-sourced labels prove valuable, alongside uncertainty quantification and negative samples. Active learning suffers a [cold-start problem](https://proceedings.mlr.press/v227/chen24a.html): its first queries are worse than random selection. Using contrastive features and pseudo-labels for diversity fixes this.

*Taken together*, the group argues that variance across runs, metric choice and annotation strategy can matter as much as architecture, and that strong simple baselines are routinely underrated. Still open: standard practice for reporting variance across training runs, metrics validated against clinical outcomes rather than proposed by fiat, and whether single-task comparisons generalise to other tasks.

**Representative papers**

- [Reproducibility of the Methods in Medical Imaging with Deep Learning.](https://proceedings.mlr.press/v227/simko24a.html)
- [Reproducibility of Training Deep Learning Models for Medical Image Analysis](https://proceedings.mlr.press/v227/bosma24a.html)
- [Robust Detection Outcome: A Metric for Pathology Detection in Medical Images](https://proceedings.mlr.press/v227/meissen24a.html)
- [Evaluating the Fairness of Deep Learning Uncertainty Estimates in Medical Image Analysis](https://proceedings.mlr.press/v227/mehta24a.html)
- [A comparative evaluation of image-to-image translation methods for stain transfer in histopathology](https://proceedings.mlr.press/v227/zingman24a.html)
- [Video pretraining advances 3D deep learning on chest CT tasks](https://proceedings.mlr.press/v227/ke24a.html)
- [Making Your First Choice: To Address Cold Start Problem in Medical Active Learning](https://proceedings.mlr.press/v227/chen24a.html)
- [Considerations for data acquisition and modeling strategies: Mitosis detection in computational pathology](https://proceedings.mlr.press/v227/ji24b.html)

## Outlier papers

Papers that sit outside the main groups in topic, method or framing.

### Unusual subjects or data

- [DRIMET: Deep Registration-based 3D Incompressible Motion Estimation in Tagged-MRI with Application to the Tongue](https://proceedings.mlr.press/v227/bian24a.html): measures how the tongue moves during speech using tagged magnetic resonance imaging (MRI), in healthy speakers and in patients who have had part of the tongue removed (glossectomy).
- [Model Adaptive Tooth Segmentation](https://proceedings.mlr.press/v227/chen24b.html): digital dentistry, segmenting teeth on 3D intraoral scans and adapting to a new dental centre without moving any data.
- [Metadata-guided Consistency Learning for High Content Images](https://proceedings.mlr.press/v227/haslum24a.html): drug discovery rather than diagnosis; microscopy assays of compound treatments, where the nuisance is laboratory batch effects and the target is a drug's mechanism of action.
- [Intra- and Inter-Cellular Awareness for 3D Neuron Tracking and Segmentation in Large-Scale Connectomics](https://proceedings.mlr.press/v227/zhai24a.html): electron-microscopy connectomics in diverse animals, with new run-length metrics measured from the nucleus to the spines.
- [CP2Image: Generating high-quality single-cell images using CellProfiler representations](https://proceedings.mlr.press/v227/ji24a.html): runs the usual pipeline backwards, generating cell images from hand-crafted CellProfiler features.
- [Frozen Language Model Helps ECG Zero-Shot Learning](https://proceedings.mlr.press/v227/li24a.html): the signal is a one-dimensional electrocardiogram (ECG), not an image; machine-generated reports guide self-supervised pretraining, with about 10% improvement in zero-shot classification.
- [Decoding natural image stimuli from fMRI data with a surface-based convolutional network](https://proceedings.mlr.press/v227/gu24a.html): the medical image is the input and an ordinary photograph is the output; the paper rebuilds what a person was looking at from responses on the cortical surface measured by functional MRI (fMRI).

### Social or ethical findings

- [Evaluating the Fairness of Deep Learning Uncertainty Estimates in Medical Image Analysis](https://proceedings.mlr.press/v227/mehta24a.html): data balancing and distributionally robust optimisation narrow performance gaps between subgroups but can leave poorer uncertainty estimates, a trade-off seen across three clinical tasks.
- [Addressing Chest Radiograph Projection Bias in Deep Classification Models](https://proceedings.mlr.press/v227/pereira24a.html): a DenseNet-121 disease classifier, never trained to do so, recognises the X-ray projection (posteroanterior or anteroposterior) better than any of the fourteen findings it was trained on, so projection can act as a shortcut.
- [DeepBrainPrint: A Novel Contrastive Framework for Brain MRI Re-Identification](https://proceedings.mlr.press/v227/puglisi24a.html): treats finding earlier brain scans of the same patient as a goal, which also shows how identifiable "anonymised" scans can be.
- [Pre-Training Transformers for Fingerprinting to Improve Stress Prediction in fMRI](https://proceedings.mlr.press/v227/rosenman24a.html): learning to tell individual subjects apart from their fMRI (a "fingerprint") turns out to be key to predicting stress.
- [Considerations for data acquisition and modeling strategies: Mitosis detection in computational pathology](https://proceedings.mlr.press/v227/ji24b.html): weighs expert against crowd-sourced labellers, and single against aggregated labels, finding value in crowd-sourced labels.

### Unusual mathematics or methods

- [SVD-DIP: Overcoming the Overfitting Problem in DIP-based CT Reconstruction](https://proceedings.mlr.press/v227/nittscher24a.html): only the singular values of the pretrained convolution layers are fine-tuned, with the singular vectors held fixed, so the deep image prior (DIP) no longer overfits noise; tested on micro-computed tomography (μCT) of a lotus root.
- [Semantic Segmentation of 3D Medical Images Through a Kaleidoscope: Data from the Osteoarthritis Initiative](https://proceedings.mlr.press/v227/woo24a.html): the kaleidoscope transform rearranges the volume into a batch of downsampled copies before convolution and inverts the rearrangement afterwards.
- [Selective experience replay compression using coresets for lifelong deep reinforcement learning in medical imaging](https://proceedings.mlr.press/v227/zheng24a.html): landmark localisation as reinforcement learning, with replay buffers compressed by coresets that preserve the reward distribution, at ratios of 10x to 40x.
- [Few Shot Hematopoietic Cell Classification](https://proceedings.mlr.press/v227/nguyen24a.html): a graphical model links neighbouring stages of cell maturation, and a variational autoencoder (VAE) is trained on a lower bound of the pairwise data likelihood.
- [Learning to Compare Longitudinal Images](https://proceedings.mlr.press/v227/kim24a.html): self-supervision by putting scans in time order, which forces the network to learn changes that cannot be reversed in time.
- [One-Class SVM on siamese neural network latent space for Unsupervised Anomaly Detection on brain MRI White Matter Hyperintensities](https://proceedings.mlr.press/v227/pinon24a.html): a classical one-class support vector machine (SVM) does the anomaly detection on a learned patch embedding.
- [Alleviating tiling effect by random walk sliding window in high-resolution histological whole slide image synthesis](https://proceedings.mlr.press/v227/bao24a.html): patch seams in synthesised whole-slide images are reduced by moving the inference window in a random walk, using 50% fewer computation resources.

### Papers that critique the field

- [Reproducibility of the Methods in Medical Imaging with Deep Learning.](https://proceedings.mlr.press/v227/simko24a.html): an audit of every accepted MIDL full paper from 2018 to 2022 finds that merely 22% had a repository judged repeatable.
- [Reproducibility of Training Deep Learning Models for Medical Image Analysis](https://proceedings.mlr.press/v227/bosma24a.html): retraining the same method on the same data and testing with paired bootstrapping gives p below 0.05 in 15% of comparisons where the null hypothesis should hold, enough to question leaderboards.
- [Generalizing Unsupervised Anomaly Detection: Towards Unbiased Pathology Screening](https://proceedings.mlr.press/v227/bercea24a.html): notes that simple intensity thresholding has overtaken autoencoders on hyperintense lesions, and that state-of-the-art autoencoders miss atrophy, oedema and resections on T1-weighted scans.
- [Making Your First Choice: To Address Cold Start Problem in Medical Active Learning](https://proceedings.mlr.press/v227/chen24a.html): in its first few queries, active learning selects data less efficiently than random selection.
- [Automatic Patient-level Diagnosis of Prostate Disease with Fused 3D MRI and Tabular Clinical Data](https://proceedings.mlr.press/v227/bashkanov24a.html): on 3800 subjects, plain concatenation of image and tabular features does as well as or better than state-of-the-art fusion modules.
- [Video pretraining advances 3D deep learning on chest CT tasks](https://proceedings.mlr.press/v227/ke24a.html): pretraining on out-of-domain Kinetics action videos helps more than pretraining on a typically sized in-domain computed tomography (CT) dataset.
- [Evaluating Adversarial Robustness of Low dose CT Recovery](https://proceedings.mlr.press/v227/gandikota24a.html): attacked reconstructions stay highly consistent with the measurements, so data consistency is not a reliable safeguard.

### Unusual clinical framing

- [Tumor Budding T-cell Graphs: Assessing the Need for Resection in pT1 Colorectal Cancer Patients](https://proceedings.mlr.press/v227/studer24a.html): the aim is to avoid unnecessary bowel surgery; graphs of tumour buds and T-cells (626 hotspots from 575 patients) raise specificity by 20% over current risk stratification without losing any sensitivity.
- [FUSQA: Fetal Ultrasound Segmentation Quality Assessment](https://proceedings.mlr.press/v227/cengiz24a.html): judges segmentation quality by its effect on gestational age; good masks differ from the doctors' estimate by 1.45 days, poor masks by up to 7.73 days.
- [Bi-parametric prostate MR image synthesis using pathology and sequence-conditioned stable diffusion](https://proceedings.mlr.press/v227/saeed24a.html): a visual Turing test in which a radiologist with 4 years' experience tells real from synthetic images with only 59.4% accuracy (chance is 50%), with radiology training in mind.
- [Generative Adversarial Networks for Coronary CT Angiography Acquisition Protocol Correction with Explicit Attenuation Constraints](https://proceedings.mlr.press/v227/herten24a.html): corrects badly timed contrast injections after the scan has been taken, in 1,179 coronary CT angiography scans.
- [Self-Supervised CSF Inpainting for Improved Accuracy Validation of Cortical Surface Analyses](https://proceedings.mlr.press/v227/wang24a.html): creates synthetic atrophy with a known ground truth, to test cortical-thickness tools rather than patients.
- [A Geometric Deep Learning Framework for Generation of Virtual Left Ventricles as Graphs](https://proceedings.mlr.press/v227/kalaie24a.html): virtual cohorts of left ventricles from UK Biobank for in-silico trials of medical devices.

### Imaging as a stand-in for another measurement

Papers that predict, from routine or cheap imaging, something normally measured in a more invasive, expensive or different way.

- [Convolutional-recurrent neural networks approximate diffusion tractography from T1-weighted MRI and associated anatomical context](https://proceedings.mlr.press/v227/cai24a.html): white-matter streamlines from structural T1-weighted MRI alone, about 5-15% from diffusion tractography, which is on the level of scan-rescan variability.
- [Joint cortical registration of geometry and function using semi-supervised learning](https://proceedings.mlr.press/v227/li24b.html): aligns functional areas of the cortex at inference without any functional data.
- [Estimating Uncertainty in PET Image Reconstruction via Deep Posterior Sampling](https://proceedings.mlr.press/v227/vlas-ic24a.html): standard-dose positron emission tomography (PET) estimated from a low-dose scan plus MRI, given as posterior samples rather than a single image.
- [Alleviating tiling effect by random walk sliding window in high-resolution histological whole slide image synthesis](https://proceedings.mlr.press/v227/bao24a.html): virtual haematoxylin and eosin (H&E) slides made from multiplex immunofluorescence with 27 markers, because a physical H&E stain cannot be done on the same section; a 56% gain on downstream cell segmentation.
- [Reverse Engineering Breast MRIs: Predicting Acquisition Parameters Directly from Images](https://proceedings.mlr.press/v227/konz24a.html): the image stands in for its own metadata; field strength, echo and repetition times, scanner model and even contrast agent type are predicted from pixels.
- [MTSR-MRI: Combined Modality Translation and Super-Resolution of Magnetic Resonance Images](https://proceedings.mlr.press/v227/dey24a.html): a low-resolution scan in one contrast becomes several plausible high-resolution scans in the other contrast, to save scan time.
- [Interpretable histopathology-based prediction of disease relevant features in Inflammatory Bowel Disease biopsies using weakly-supervised deep learning](https://proceedings.mlr.press/v227/mokhtari24a.html): biopsy histology predicts endoscopic labels such as Crohn's disease against ulcerative colitis (AUC=0.87).

The [tractography paper](https://proceedings.mlr.press/v227/cai24a.html) also belongs under *Papers that critique the field*: it ends by asking whether tractography is truly a microstructural phenomenon at all.

## Theory picks

The most mathematically foundational papers, ranked for a close reading (from abstracts only).

1. [E(3) x SO(3)-Equivariant Networks for Spherical Deconvolution in Diffusion MRI](https://proceedings.mlr.press/v227/elaldi24a.html): each voxel holds a signal on the sphere, so the data are 6D and carry two symmetries: rotations, reflections and translations of space (E(3)), and a separate rotation of each voxel's sphere (SO(3)). The layers are built to be equivariant to both. It is worth working out how the two group actions combine, and why sparse deconvolution benefits from respecting both.
2. [Rotation-Scale Equivariant Steerable Filters](https://proceedings.mlr.press/v227/yang24a.html): filters are linear combinations of Gaussian derivatives, so they can be steered in direction, while trainable scale parameters are constrained to span disjoint scales in successive layers. A compact meeting of steerable-filter theory and scale-space theory; check what joint rotation-scale equivariance the construction actually guarantees.
3. [Amortized Normalizing Flows for Transcranial Ultrasound with Uncertainty Quantification](https://proceedings.mlr.press/v227/orozco24a.html): Bayesian inversion for ultrasound computed tomography, with a normalising flow amortised over observations that are compressed by a physics-informed summary statistic. The question to work through is what the summary statistic preserves of the posterior, and how calibration of uncertainty against error is checked.
4. [DRIMET: Deep Registration-based 3D Incompressible Motion Estimation in Tagged-MRI](https://proceedings.mlr.press/v227/bian24a.html): a loss on the Jacobian determinant pushes the deformation to be incompressible, and a sinusoidal transform of the harmonic phase removes the need for phase interpolation. Good for working out the geometry of volume-preserving, approximately diffeomorphic flows and why the phase wrapping matters.
5. [SVD-DIP: Overcoming the Overfitting Problem in DIP-based CT Reconstruction](https://proceedings.mlr.press/v227/nittscher24a.html): the deep image prior is fine-tuned only through the singular values of its convolution layers, with the left and right singular vectors frozen. This is spectral regularisation by restricting the parameter space; work out why limiting optimisation to singular values stops the network fitting noise.
6. [Reproducibility of Training Deep Learning Models for Medical Image Analysis](https://proceedings.mlr.press/v227/bosma24a.html): treats variation between training runs as a source of variance that standard tests ignore, and measures p below 0.05 in 15% of comparisons that should follow the null hypothesis. A clean statistics exercise: write down the variance decomposition (data against retraining) and derive how far the false-positive rate is inflated.

Also worth a look: [Geometry-informed Implicit Neural Representations](https://proceedings.mlr.press/v227/harten24a.html) (deformations in a coordinate system curved along the centrelines of the small intestine), [Domain adaptation using optimal transport](https://proceedings.mlr.press/v227/falahkheirkhah24a.html) (an optimal transport loss between batches instead of single samples), [Multi-site Parkinson's disease classification with counterfactual inference](https://proceedings.mlr.press/v227/vigneshwaran24a.html) (a causal normalising flow asking what features would look like at another site), [Deep Posterior Sampling for PET](https://proceedings.mlr.press/v227/vlas-ic24a.html) (a conditional generative adversarial network approximating the Bayesian posterior), [Federated Cross Learning](https://proceedings.mlr.press/v227/xu24a.html) (a theoretical analysis of model aggregation on non-iid data, replaced by round-robin training) and [Adversarial Robustness of Low dose CT Recovery](https://proceedings.mlr.press/v227/gandikota24a.html) (data-consistent attacks used to explore the solution space of the inverse problem).

## Sources

- *Proceedings of the 6th International Conference on Medical Imaging with Deep Learning*,
  PMLR volume 227: <https://proceedings.mlr.press/v227/>
