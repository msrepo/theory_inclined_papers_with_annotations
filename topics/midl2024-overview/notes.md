---
title: "MIDL 2024: topic map, outliers and theory picks"
authors: "Topic notes"
venue: "MIDL 2024"
tags: [midl, survey, conference-overview, segmentation, self-supervised-learning, anomaly-detection, registration]
status: living
category: "Medical image analysis"
subcategory: "Conference overviews"
short_title: "MIDL 2024 — overview"
summary: "a high-level map of the 117 MIDL 2024 full papers (PMLR v250) in thirteen topic groups, with an idea-level synthesis of each, the outlier papers that fall outside them, and a shortlist of the most mathematically foundational ones to read"
related: [midl2025-overview, midl2026-overview, 2024-wundram-glaucoma-probabilistic-pipeline]
---

## Why this page exists

MIDL 2024 (Paris, 3–5 July 2024) was published as
[PMLR volume 250](https://proceedings.mlr.press/v250/), with 117 full papers. This page is the
companion to the [MIDL 2025](../midl2025-overview/index.html) and
[MIDL 2026](../midl2026-overview/index.html) overviews: which topics dominate, which papers do not
fit the pattern, and which are worth a close, theory-first read.

The grouping and the syntheses are from **titles and abstracts only**; the PDFs have not been
read. Shares are approximate and many papers sit in more than one group, so they add up to
more than 100%.

## Topic groups at a glance

| Group | Share | What's in it |
|---|---|---|
| [Segmentation](#segmentation) | ~27% | Partial and weak labels, semi-supervision, active learning, SAM prompting and adapters, domain shift, mask quality and uncertainty |
| [Self-supervised pretraining and efficient adaptation](#self-supervised-pretraining-and-efficient-adaptation) | ~15% | Masked autoencoders adapted to medical data, dense and equivariant contrastive learning, what to pretrain on, PEFT benchmarks |
| [Computational pathology and microscopy](#computational-pathology-and-microscopy) | ~13% | Fast cell detection, mitosis, MIL from slide labels, label-free scoring, virtual staining, spatial transcriptomics |
| [Generative models and synthesis](#generative-models-and-synthesis) | ~16% | Diffusion and GANs for data sharing, controlled generation, restoration, counterfactuals, checks on anatomical fidelity |
| [Unsupervised anomaly detection](#unsupervised-anomaly-detection) | ~5% | Pseudo-healthy reconstruction with MAEs and diffusion, feature-space matching, separating healthy from pathological variation |
| [Registration, shape, neural fields and physics](#registration-shape-neural-fields-and-physics) | ~17% | Multi-objective and segmentation-guided registration, INRs, signed distance shape models, physics-informed networks, motion and pose |
| [Reconstruction, denoising and acquisition](#reconstruction-denoising-and-acquisition) | ~9% | Simulated forward models, diffusion priors, physics-based losses, federated denoising, unlearning |
| [Trustworthiness](#trustworthiness) | ~14% | Shortcuts and confounders, counterfactuals, natural-language explanations, uncertainty evaluation, bias attacks, privacy |
| [Federated, continual and collaborative learning](#federated-continual-and-collaborative-learning) | ~7% | Non-IID federations, malfunctioning clients, lifelong learning, joint training on mismatched databases |
| [Vision-language models and reports](#vision-language-models-and-reports) | ~8% | LLM-written report impressions, difference VQA, VLM benchmarks, text in segmentation, reports as labels |
| [Longitudinal and multimodal clinical prediction](#longitudinal-and-multimodal-clinical-prediction) | ~9% | Disease progression, missing modalities, image–metadata fusion, digital twins, outcome prediction |
| [Surgery, video and ultrasound](#surgery-video-and-ultrasound) | ~9% | Self-supervised surgical video, optical flow as supervision, ultrasound pretraining, fetal motion |
| [Benchmarks, data and meta-research](#benchmarks-data-and-meta-research) | ~14% | Pretraining and PEFT benchmarks, evaluation protocols, humans in the comparison, dataset citation practices |

**Modalities, roughly by frequency:** MRI (brain above all), histopathology and microscopy, CT,
chest X-ray, retina (OCT and fundus), surgical video, then ultrasound and others.

**Across three years:** in 2024 unsupervised anomaly detection and federated or continual learning
still form groups of their own, and language models are a small, chest X-ray-centred group
(an LLM writing report impressions, difference VQA, one benchmark). By 2025 SAM adaptation and
foundation-model reuse have grown, and by 2026 vision–language models, agents and reinforcement-learning
fine-tuning are a major group. Throughout, a steady minority of papers question evaluation
practice; in 2024 these include structure size as a confounder of uncertainty–Dice correlations,
cropping that hides domain shift, and LLM judges that disagree with radiologists.

## Topic groups

Each group lists a few representative papers, not all of them.

### Segmentation

**The ideas behind this group** (a synthesis from the abstracts of about 32 papers in this group; the individual papers have not been read in full).

*The shared problem is that dense masks are expensive to obtain and a segmenter trained at one site rarely keeps its quality at another.* The papers attack the label bottleneck, the transfer problem and the question of how far a given mask can be trusted. The Segment Anything Model (SAM) is now a common starting point.

*The first thread gets more from fewer, partial or weaker labels.* Partially labelled and region-based annotations are handled in two ways. One conditions the whole network with Feature-wise Linear Modulation (FiLM) layers ([CoNeMOS](https://proceedings.mlr.press/v250/billot24a.html)). The other distils several anatomy-specific teachers into one real-time student ([multi-teacher distillation](https://proceedings.mlr.press/v250/maack24a.html)). Joint training across 7 databases with different pathologies and modalities turns out to be feasible ([joint learning](https://proceedings.mlr.press/v250/xu24a.html)). In the semi-supervised setting, uncertain teacher voxels are pseudo-labelled through embedding correspondence, with results on just 4 CT scans ([embedding matching](https://proceedings.mlr.press/v250/xie24a.html)). Uncertain regions are also pulled towards class prototypes contrastively ([contrastive uncertainty](https://proceedings.mlr.press/v250/jarimijafarbigloo24a.html)). Weak circle scribbles with a size-constraint loss match fully supervised baselines ([weak prostate](https://proceedings.mlr.press/v250/trombetta24a.html)). In active learning with nnU-Net, most selection strategies beat random sampling ([active nnU-Net](https://proceedings.mlr.press/v250/follmer24a.html)). Dense pretraining that enforces equivariance of pixel-level representations gives a 5-fold reduction in annotation burden ([Pix2Rep](https://proceedings.mlr.press/v250/seince24a.html)).

*The second thread treats the prompt as SAM's main cost.* Box prompts are propagated through bone CT without any training, from one initial box per class, with Dice between 54.22% and 88.26% ([prompt propagation](https://proceedings.mlr.press/v250/magg24a.html)). Other papers slide a three-slice window through the volume ([Slide-SAM](https://proceedings.mlr.press/v250/quan24a.html)), or let an in-context learning model write the prompts, with over a 10% Dice increase over the in-context model alone ([ICL-SAM](https://proceedings.mlr.press/v250/hu24a.html)). Two more use self-similarity maps and test-time fine-tuning ([sparse-prompt SAM](https://proceedings.mlr.press/v250/shaharabany24a.html)), or let a small supervised model prompt MedSAM to produce weak labels ([auto weak labels](https://proceedings.mlr.press/v250/deshpande24a.html)). SAM is also used for continual learning ([UnCLe SAM](https://proceedings.mlr.press/v250/ranem24a.html)). Adapters cut tuned parameters by up to 4x ([Med-Tuning](https://proceedings.mlr.press/v250/shen24a.html)). Vision-language segmenters transfer, but image features dominate the text ([VLSM transfer](https://proceedings.mlr.press/v250/poudel24a.html)).

*The third thread targets domain shift.* One paper feeds image edges in as a domain-invariant input for single-source generalisation ([edge-guided SDG](https://proceedings.mlr.press/v250/jiang24a.html)). Another pseudo-labels 18.000 unlabelled images with a Noisy Student and shows that cropping around the optic disc hides most of the shift ([Noisy Student](https://proceedings.mlr.press/v250/moris24a.html)). A third adapts without source data in two stages ([source-free](https://proceedings.mlr.press/v250/vs24a.html)). Adult-to-newborn skull stripping combines adversarial features with Gaussian-mixture synthetic images ([newborn](https://proceedings.mlr.press/v250/omidi24a.html)), and style randomisation makes breast segmentation robust to fat suppression ([breast density](https://proceedings.mlr.press/v250/yuksel24a.html)).

*The fourth thread asks how good a given mask is.* Structure size confounds the usual correlation between uncertainty and Dice. A partial correlation lowers the estimates ([structure size](https://proceedings.mlr.press/v250/geissler24a.html)). Ensemble uncertainty picks which patches to review and predicts the gain in Dice ([patch review](https://proceedings.mlr.press/v250/kohler24a.html)). Other papers model uncertainty with probabilistic signed distance functions ([retinal SDF](https://proceedings.mlr.press/v250/islam24a.html)), carry segmentation uncertainty into glaucoma diagnosis ([probabilistic pipeline](https://proceedings.mlr.press/v250/wundram24a.html)), or use diffusion to model the gap between a U-Net and the ground truth ([Re-DiffiNet](https://proceedings.mlr.press/v250/ren24a.html)). A dose predictor judges contours by their effect on dose and outperforms radiation oncologists ([dosimetric QC](https://proceedings.mlr.press/v250/kamath24a.html)).

*Taken together*, label efficiency and SAM prompting take up most of the effort, and quality control is treated as a separate problem. Still open: SAM is still run slice by slice on volumes; both weakly and fully supervised models drop on unseen domains ([weak prostate](https://proceedings.mlr.press/v250/trombetta24a.html)); and if structure size confounds uncertainty, many reported uncertainty–quality correlations may be inflated.

**Representative papers**

- [Training-free Prompt Placement by Propagation for SAM Predictions in Bone CT Scans](https://proceedings.mlr.press/v250/magg24a.html)
- [ICL-SAM: Synergizing In-context Learning Model and SAM in Medical Image Segmentation](https://proceedings.mlr.press/v250/hu24a.html)
- [Structure Size as Confounder in Uncertainty Based Segmentation Quality Prediction](https://proceedings.mlr.press/v250/geissler24a.html)
- [Active Learning with the nnUNet and Sample Selection with Uncertainty-Aware Submodular Mutual Information Measure](https://proceedings.mlr.press/v250/follmer24a.html)
- [Network conditioning for synergistic learning on partial annotations](https://proceedings.mlr.press/v250/billot24a.html)
- [Semi-supervised learning with Noisy Students improves domain generalization in optic disc and cup segmentation in uncropped fundus images](https://proceedings.mlr.press/v250/moris24a.html)
- [Efficiently correcting patch-based segmentation errors to control image-level performance in retinal images](https://proceedings.mlr.press/v250/kohler24a.html)
- [Weakly supervised deep learning model with size constraint for prostate cancer detection in multiparametric MRI and generalization to unseen domains](https://proceedings.mlr.press/v250/trombetta24a.html)

### Self-supervised pretraining and efficient adaptation

**The ideas behind this group** (a synthesis from the abstracts of about 17 papers in this group; the individual papers have not been read in full).

*The shared problem is that ImageNet initialisation is still the default, although medical images differ from natural ones and labels are scarce.* The papers ask which pretext task suits medical data, which data to pretrain on, and how to adapt a large model cheaply.

*The first thread adapts masked reconstruction to medical images.* Disruptive Autoencoders mask across channel embeddings rather than tokens and add noise and downsampling, so that low-level, local features are learned. The method tops the BTCV leaderboard ([Disruptive AE](https://proceedings.mlr.press/v250/valanarasu24a.html)). UltraMAE uses an ultrasound-specific confidence map to guide masking over 100,000+ videos and images ([UltraMAE](https://proceedings.mlr.press/v250/rahman24a.html)). Surface masked autoencoders on icosahedral grids improve cortical phenotype prediction by ≥ 26% ([sMAE](https://proceedings.mlr.press/v250/dahan24b.html)). Starting from a video-pretrained MAE and then pretraining on unlabelled task data improves F1 by 8.3 with a training speedup of factor 2 ([Video-CT MAE](https://proceedings.mlr.press/v250/buess24a.html)).

*The second thread changes the contrastive objective or its augmentations.* Pix2Rep asks pixel representations to be equivariant, not invariant, to spatial transformations ([Pix2Rep](https://proceedings.mlr.press/v250/seince24a.html)). Laparoflow-SSL weights pixel pairs by optical-flow similarity and stays competitive with over 92% fewer annotated samples ([Laparoflow-SSL](https://proceedings.mlr.press/v250/moens24a.html)). Dimensionality Driven Augmentation Search (DDA) uses the local dimensionality of representations as a differentiable proxy to search for augmentations, and finds that hue, useful on natural images, does not help in laparoscopy ([DDA](https://proceedings.mlr.press/v250/zhou24b.html)). Barlow Twins followed by self-distillation and fine-tuning reaches 75.6% balanced accuracy for dementia differential diagnosis ([Triplet Training](https://proceedings.mlr.press/v250/li24a.html)).

*The third thread asks what to pretrain on.* A benchmark of 10 supervised and self-supervised models for multi-view chest X-ray trains at 1%, 10%, and 100% of the data ([multi-view benchmark](https://proceedings.mlr.press/v250/afzal24a.html)). Pretraining on large task-unrelated data ("wild") makes Swin models more robust to contrast, slice thickness and reconstruction kernel than pretraining on the task data. Centred kernel alignment (CKA) shows less feature reuse near the final encoder layers ([wild pretraining](https://proceedings.mlr.press/v250/jiang24b.html)).

*The fourth thread adapts cheaply.* A benchmark of 17 parameter-efficient fine-tuning (PEFT) algorithms over 700 experiments finds gains of up to 22%, mainly when data are scarce ([PEFT benchmark](https://proceedings.mlr.press/v250/dutt24a.html)). Adapters serve volumetric segmentation ([Med-Tuning](https://proceedings.mlr.press/v250/shen24a.html)) and natural-language explanations, where training 12% of parameters beats full fine-tuning ([NLE PEFT](https://proceedings.mlr.press/v250/rio-torto24a.html)). Textual Inversion adapts Stable Diffusion from 100 samples per modality ([textual inversion](https://proceedings.mlr.press/v250/wilde24a.html)). On the architecture side, VariViT handles variable image sizes by resizing positional embeddings and cuts computation by up to 30% ([VariViT](https://proceedings.mlr.press/v250/varma24a.html)). A multiscale surface transformer brings shifted windows to cortical meshes ([MS-SiT](https://proceedings.mlr.press/v250/dahan24a.html)). Contrastive transformers are applied to retrieval ([retrieval](https://proceedings.mlr.press/v250/susmitha24a.html)).

*Taken together*, the recipes move away from natural-image defaults: local masking, equivariance, and augmentations tuned to the domain. Still open: there is no shared benchmark across modalities; few papers explain why a given inductive bias helps (CKA and local dimensionality are early tools); and it is not settled whether large generic pretraining beats domain-specific pretraining.

**Representative papers**

- [Disruptive Autoencoders: Leveraging Low-level features for 3D Medical Image Pre-training](https://proceedings.mlr.press/v250/valanarasu24a.html)
- [Dense Self-Supervised Learning for Medical Image Segmentation](https://proceedings.mlr.press/v250/seince24a.html)
- [DDA: Dimensionality Driven Augmentation Search for Contrastive Learning in Laparoscopic Surgery](https://proceedings.mlr.press/v250/zhou24b.html)
- [Self-supervised pretraining in the wild imparts image acquisition robustness to medical image transformers: an application to lung cancer segmentation](https://proceedings.mlr.press/v250/jiang24b.html)
- [Parameter-Efficient Fine-Tuning for Medical Image Analysis: The Missed Opportunity](https://proceedings.mlr.press/v250/dutt24a.html)
- [A Comprehensive Benchmark of Supervised and Self-supervised Pre-training on Multi-view Chest X-ray Classification](https://proceedings.mlr.press/v250/afzal24a.html)
- [UltraMAE: Multi-modal Masked Autoencoder for Ultrasound Pre-training](https://proceedings.mlr.press/v250/rahman24a.html)
- [VariViT: A Vision Transformer for Variable Image Sizes](https://proceedings.mlr.press/v250/varma24a.html)

### Computational pathology and microscopy

**The ideas behind this group** (a synthesis from the abstracts of about 15 papers in this group; the individual papers have not been read in full).

*The shared problem is that slides are gigapixel-sized, cell-level labels are tedious, and stain or scanner changes break models.* The papers make cell-level analysis fast, learn from slide-level or no labels, and connect tissue appearance to molecular data.

*The first thread detects and classifies cells at scale.* HoVer-NeXt is 5× faster than the CellViT pipeline and 17× faster than the HoVer-Net pipeline, and adds a mitosis class to a public dataset ([HoVer-NeXt](https://proceedings.mlr.press/v250/baumann24a.html)). Cell-DETR argues that detection, not segmentation, is the clinically relevant task and reports x3.4 faster inference on whole-slide images ([Cell-DETR](https://proceedings.mlr.press/v250/pina24a.html)). For mitoses, annotated glass slides are physically rescanned and trained with a loss for partially matching labels, and atypical-mitosis subtyping is added as an auxiliary task, giving F1 0.80 ([mitosis rescanning](https://proceedings.mlr.press/v250/fick24a.html)). When labels are sparse, pseudo-labels support multiplex immunofluorescence detection with F1 84.75% ([mIF pseudo-labels](https://proceedings.mlr.press/v250/shokrollahi24a.html)). A boundary-aware contrastive module denoises nuclei pseudo-labels ([boundary-aware nuclei](https://proceedings.mlr.press/v250/zhang24a.html)). A specialised U-Net3+ argues that dataset-specific models remain the baseline that universal ones must beat ([U-Net3+](https://proceedings.mlr.press/v250/swain24a.html)).

*The second thread learns from slide-level labels with multiple instance learning (MIL).* NcIEMIL argues that decoupled MIL inflates instance representations into a high-rank space and filters them back towards a low-rank category space ([NcIEMIL](https://proceedings.mlr.press/v250/qiehe24a.html)). Slide labels alone classify lupus nephritis subtypes with AUC 91.0% ([LupusNet](https://proceedings.mlr.press/v250/sharma24a.html)). They also predict aneuploidy in Barrett's oesophagus with 0.93 AUC, where a feature-level augmentation for MIL drives most of the gain ([ploidy MIL](https://proceedings.mlr.press/v250/ercan24a.html)).

*The third thread removes the need for ground truth or for chemistry.* An unsupervised generative adversarial network (GAN) scores Ki67 with accuracy 0.97 on 2,136 cases ([IHCScoreGAN](https://proceedings.mlr.press/v250/molnar24a.html)). Virtual staining of unprocessed thick tissue takes less than a minute for an image with 100 million pixels ([ThickV-Stain](https://proceedings.mlr.press/v250/shi24a.html)). An unsupervised pipeline detects, localises and inpaints artefacts, and pathologists cannot tell the restored images from clean ones ([HARP](https://proceedings.mlr.press/v250/fuchs24a.html)).

*The fourth thread links morphology to molecular and structural data.* Self-supervised morphology features reassign cell types after spatial-transcriptomics deconvolution ([MHAST](https://proceedings.mlr.press/v250/chelebian24a.html)). A GAN maps gene expression to images and synthesises a 106496 × 53248 mouse pup that can be edited ([IST-editing](https://proceedings.mlr.press/v250/wu24a.html)). Conditional GANs turn neuron tractograms into labelled microscopy images ([neuron synthesis](https://proceedings.mlr.press/v250/decroocq24a.html)).

*Taken together*, the group values throughput and freedom from labels as much as accuracy. Still open: unsupervised and generative pipelines are validated mostly downstream or by visual judgement, not by checks that they preserve morphology; the low-rank argument in MIL is stated but not tested formally; and robustness to scanners is engineered (rescanning, stain normalisation) more than modelled.

**Representative papers**

- [HoVer-NeXt: A Fast Nuclei Segmentation and Classification Pipeline for Next Generation Histopathology](https://proceedings.mlr.press/v250/baumann24a.html)
- [Cell-DETR: Efficient cell detection and classification in WSIs with transformers](https://proceedings.mlr.press/v250/pina24a.html)
- [Improving CNN-Based Mitosis Detection through Rescanning Annotated Glass Slides and Atypical Mitosis Subtyping](https://proceedings.mlr.press/v250/fick24a.html)
- [NcIEMIL: Rethinking Decoupled Multiple Instance Learning Framework for Histopathological Slide Classification](https://proceedings.mlr.press/v250/qiehe24a.html)
- [IHCScoreGAN: An unsupervised generative adversarial network for end-to-end ki67 scoring for clinical breast cancer diagnosis](https://proceedings.mlr.press/v250/molnar24a.html)
- [ThickV-Stain: Unprocessed Thick Tissues Virtual Staining for Rapid Intraoperative Histology](https://proceedings.mlr.press/v250/shi24a.html)
- [Learned morphological features guide cell type assignment of deconvolved spatial transcriptomics](https://proceedings.mlr.press/v250/chelebian24a.html)
- [HARP: Unsupervised Histopathology Artifact Restoration](https://proceedings.mlr.press/v250/fuchs24a.html)

### Generative models and synthesis

**The ideas behind this group** (a synthesis from the abstracts of about 19 papers in this group; the individual papers have not been read in full).

*The shared problem is that medical data are scarce, private and heterogeneous, and generators promise more data, controlled variation or cleaner images, but the proof that they work is usually indirect.*

*The first thread synthesises data to augment or share.* One method synthesises, shares and segments across sites without manual hyperparameter tuning ([HyFree-S3](https://proceedings.mlr.press/v250/chebykin24a.html)). Textual Inversion adapts Stable Diffusion from 100 samples per modality into embeddings of less than 1 MB, and prostate cancer AUC rises from 0.78 to 0.80 ([textual inversion](https://proceedings.mlr.press/v250/wilde24a.html)). A vector-quantised GAN (VQGAN) with a masked transformer generates 3D tumour regions rather than whole volumes ([tumour ROIs](https://proceedings.mlr.press/v250/zhou24a.html)). Conditional GANs generate labelled neuron microscopy ([neuron synthesis](https://proceedings.mlr.press/v250/decroocq24a.html)). Set diffusion generates cerebral vessel trees for in-silico trials, conditioned on an occlusion ([vessel trees](https://proceedings.mlr.press/v250/kuipers24a.html)).

*The second thread controls geometry, covariates and misalignment.* A latent diffusion model conditioned on resolution, field of view and orientation trains on mixed whole-body CT and shows emergent super-resolution ([resolution-invariant LDM](https://proceedings.mlr.press/v250/patel24a.html)). Multi-Source StyleGAN learns from sources that record different covariates ([MSSG](https://proceedings.mlr.press/v250/lai24a.html)). DA-GAN corrects substantial misalignment between training pairs through inverse consistency and deformation-aware discriminators ([DA-GAN](https://proceedings.mlr.press/v250/xin24a.html)). Gene-expression conditioning edits a generated gigapixel tissue ([IST-editing](https://proceedings.mlr.press/v250/wu24a.html)). Radiologists' gaze conditions diffusion, and the resulting densities are used for zero-shot classification ([GazeDiff](https://proceedings.mlr.press/v250/bhattacharya24a.html)).

*The third thread uses generators as image operators.* Diffusion super-resolution is trained on simulated degradations at scaling factors larger than 6× without paired data ([spine super-resolution](https://proceedings.mlr.press/v250/graf24a.html)). Diffusion also removes artefacts while prioritising disease regions ([chest X-ray SISR](https://proceedings.mlr.press/v250/roy24a.html)), denoises planar X-ray ([X-ray denoising](https://proceedings.mlr.press/v250/sanderson24a.html)) and inpaints histology artefacts ([HARP](https://proceedings.mlr.press/v250/fuchs24a.html)). Other papers stain virtually ([ThickV-Stain](https://proceedings.mlr.press/v250/shi24a.html)), score Ki67 from unpaired synthetic data ([IHCScoreGAN](https://proceedings.mlr.press/v250/molnar24a.html)), and model the gap between a segmentation and the ground truth ([Re-DiffiNet](https://proceedings.mlr.press/v250/ren24a.html)). A detector-guided diffusion model produces counterfactuals that change pathology while leaving support devices in place ([DeCoDEx](https://proceedings.mlr.press/v250/fathi24a.html)).

*The fourth thread asks whether synthetic images are anatomically right.* A visual Turing test on causally controlled StyleGAN3 brain MRI turns out to be inadequate. Segmented hallmarks show slight hippocampus differences (P = 5.7e-2) but significant discrepancies in the lateral ventricles ([anatomical consistency](https://proceedings.mlr.press/v250/yassin24a.html)).

*Taken together*, generators are now general tools for augmentation, sharing, restoration and explanation, and diffusion is overtaking GANs. Still open: evaluation relies on downstream accuracy or expert viewing, which the Turing-test paper shows can miss anatomical error; privacy claims for shared synthetic data or embeddings are argued, not measured; and hallucination in diffusion restoration is rarely quantified.

**Representative papers**

- [Evaluating Age-Related Anatomical Consistency in Synthetic Brain MRI against Real-World Alzheimer's Disease Data.](https://proceedings.mlr.press/v250/yassin24a.html)
- [Medical diffusion on a budget: Textual Inversion for medical image generation](https://proceedings.mlr.press/v250/wilde24a.html)
- [Resolution and Field of View Invariant Generative Modelling with Latent Diffusion Models](https://proceedings.mlr.press/v250/patel24a.html)
- [Deformation-aware GAN for Medical Image Synthesis with Substantially Misaligned Pairs](https://proceedings.mlr.press/v250/xin24a.html)
- [Generating Cerebral Vessel Trees of Acute Ischemic Stroke Patients using Conditional Set-Diffusion](https://proceedings.mlr.press/v250/kuipers24a.html)
- [DeCoDEx: Confounder Detector Guidance for Improved Diffusion-based Counterfactual Explanations](https://proceedings.mlr.press/v250/fathi24a.html)
- [Hyperparameter-Free Medical Image Synthesis for Sharing Data and Improving Site-Specific Segmentation](https://proceedings.mlr.press/v250/chebykin24a.html)
- [Heterogeneous Medical Data Integration with Multi-Source StyleGAN](https://proceedings.mlr.press/v250/lai24a.html)

### Unsupervised anomaly detection

**The ideas behind this group** (a synthesis from the abstracts of about 6 papers in this group; the individual papers have not been read in full).

*The shared problem is that pathology is too varied and rare to label exhaustively, so the model learns what healthy looks like and flags departures from it.* This only works if the model reliably fails to reproduce the abnormal parts, and if those departures are pathology rather than ordinary healthy variation or noise. Reconstruction-based unsupervised anomaly detection (UAD) still segments less accurately than supervised models, and the papers try to close that gap from three directions.

*The first thread makes the pseudo-healthy reconstruction more trustworthy.* A Swin-transformer masked autoencoder (MAE) is trained on healthy brains. At test time it masks segments of each slice in turn and inpaints them, so pathological features are not copied through, and the errors are accumulated into an anomaly map ([Ano-swinMAE](https://proceedings.mlr.press/v250/rashmi24a.html)). A diffusion model produces pseudo-healthy reconstructions, and a supervised U-Net learns to delineate anomalies from the input–reconstruction differences. Synthetic anomalies or a small annotated set improve in-domain segmentation while the reconstruction stage keeps generalisation to pathologies unseen in training ([reconstruction plus segmentation](https://proceedings.mlr.press/v250/behrendt24a.html)).

*The second thread moves the comparison into feature or latent space.* Student–teacher feature pyramid matching (STFPM) is applied to 3D brain MRI with patch-based self-supervised pretraining and axial–coronal–sagittal (ACS) convolutions, and this top-down approach outperforms the bottom-up f-AnoGAN ([3D STFPM](https://proceedings.mlr.press/v250/schwarz24a.html)). Contrastive-analysis variational autoencoders (VAEs) split the latent space into features shared with healthy subjects and features found only in patients. A disentangling term and a classification term stop information leaking between the two parts ([SepVAE](https://proceedings.mlr.press/v250/louiset24a.html)). This states the "pathological versus healthy variation" question most directly.

*The third thread tests the idea outside curated benchmarks.* A β-VAE is pretrained on over 10,000 healthy UK Biobank FLAIR images and fine-tuned on nearly 700 images from a clinical data warehouse. Detected lesion volumes correlate with Fazekas scores taken from radiology reports, and the model is robust to image quality, but it is not ready for clinical routine because poor VAE reconstructions make it miss some lesions ([β-VAE in clinical routine](https://proceedings.mlr.press/v250/loizillon24a.html)). The same logic moves to histopathology, where anomaly detection, unsupervised localisation and diffusion inpainting remove slide artefacts that pathologists then cannot tell from clean images ([HARP](https://proceedings.mlr.press/v250/fuchs24a.html)).

*Taken together*, the group agrees that the reconstruction or feature model is the bottleneck and is moving towards hybrids that add a small amount of supervision. Still open:
- None of the abstracts gives a principled reason why masking or diffusion should erase pathology but keep healthy anatomy.
- Evaluation rests on a few brain datasets (BraTS, MSLUB, WMH, MSSEG).
- Only SepVAE models healthy variation explicitly, so the rest cannot tell rare-but-normal anatomy from disease.

**Representative papers**

- [Combining Reconstruction-based Unsupervised Anomaly Detection with Supervised Segmentation for Brain MRIs](https://proceedings.mlr.press/v250/behrendt24a.html)
- [Ano-swinMAE: Unsupervised Anomaly Detection in Brain MRI using swin Transformer based Masked Auto Encoder](https://proceedings.mlr.press/v250/rashmi24a.html)
- [A Patch-based Student-Teacher Pyramid Matching Approach to Anomaly Detection in 3D Magnetic Resonance Imaging](https://proceedings.mlr.press/v250/schwarz24a.html)
- [SepVAE: a contrastive VAE to separate pathological patterns from healthy ones](https://proceedings.mlr.press/v250/louiset24a.html)
- [Detecting Brain Anomalies in Clinical Routine with the β-VAE: Feasibility Study on Age-Related White Matter Hyperintensities](https://proceedings.mlr.press/v250/loizillon24a.html)
- [HARP: Unsupervised Histopathology Artifact Restoration](https://proceedings.mlr.press/v250/fuchs24a.html)

### Registration, shape, neural fields and physics

**The ideas behind this group** (a synthesis from the abstracts of about 20 papers in this group; the individual papers have not been read in full).

*The shared problem is turning images into geometry or physics: deformations, shapes, poses, motion and simulated physiology.* Learned methods have to match iterative solvers in accuracy and plausibility while being fast enough for clinical use. The papers fall into four threads.

*The first thread changes what registration optimises when intensity correspondence fails.* New retinal fluid has no counterpart in the earlier scan, so a "fluid seed" of sparse intensity offsets is placed in the moving image and registered to the follow-up ([FluidRegNet](https://proceedings.mlr.press/v250/andresen24a.html)). For multimodal pairs with few segmentations in one modality, segmentation is transferred to the other modality and registration is done on the predicted segmentations ([segmentation-guided registration](https://proceedings.mlr.press/v250/demir24a.html)). Deformable image registration (DIR) is cast as multi-objective learning, giving a spread of outputs along the trade-off front rather than a single weighting ([multi-objective DIR](https://proceedings.mlr.press/v250/grewal24a.html)). Registration quality itself is scored by a Siamese network trained on synthetic misalignments ([SiamRegQC](https://proceedings.mlr.press/v250/kulkarni24b.html)). Registration also appears as a tool inside other tasks: symmetric registration with inverse consistency inside a generative adversarial network (GAN) for misaligned synthesis pairs ([DA-GAN](https://proceedings.mlr.press/v250/xin24a.html)), and deformable alignment of brain contours to remove a shape confounder ([brain shape](https://proceedings.mlr.press/v250/dauchelle24a.html)).

*The second thread uses implicit neural representations (INRs) as continuous fields.* Displacement INRs fold in physiologically implausible ways, so the field is parameterised with free-form deformations ([SINR](https://proceedings.mlr.press/v250/sideri-lampretsa24a.html)). Per-pair optimisation is slow, so meta-learned templates infused with image embeddings give a better starting point ([REINDIR](https://proceedings.mlr.press/v250/harten24a.html)). Implicit networks fitted at test time add temporal consistency to video polyp segmentation ([implicit polyp](https://proceedings.mlr.press/v250/dahan24c.html)). An INR overfitted to a batch of X-rays, with parameters more than 6x smaller than the images, doubles as a way to obfuscate patient identity ([neural obfuscation](https://proceedings.mlr.press/v250/heinrich24a.html)).

*The third thread represents shape and physiology as functions or simulations.* Cardiac anatomy is learned as a Lipschitz-regular deep signed distance function (SDF) that can be reconstructed from partial point clouds or electroanatomical maps ([Shape of my heart](https://proceedings.mlr.press/v250/verhulsdonk24a.html)). Retinal layers get a probabilistic SDF with Gaussian uncertainty ([probabilistic SDF](https://proceedings.mlr.press/v250/islam24a.html)), and set-diffusion generates vessel trees for in-silico trials ([vessel trees](https://proceedings.mlr.press/v250/kuipers24a.html)). On the physics side:
- A finite-volume discretisation of Darcy's equations serves as the loss of a graph neural network for myocardial perfusion ([finite-volume GNN](https://proceedings.mlr.press/v250/chou24a.html)).
- Meta-learning makes physics-informed neural fields for CT perfusion produce full-volume maps in 1.2 minutes rather than over 40 minutes ([ReSPPINN](https://proceedings.mlr.press/v250/vries24a.html)).
- Cardiac digital twins of 1000 virtual patients supply labels for treatment-outcome prediction ([digital twins](https://proceedings.mlr.press/v250/zolotarev24a.html)).

*The fourth thread estimates motion and pose directly.* Rigid fetal motion is learned through a closed-form update and estimated jointly with diffeomorphic distortion correction ([fetal EPI](https://proceedings.mlr.press/v250/wang24a.html)). A recurrent network reads single k-space spokes and estimates motion at more than 230 Hz ([k-space motion](https://proceedings.mlr.press/v250/krusen24a.html)). Bone pose is inferred from non-orthogonal biplanar X-rays ([forearm pose](https://proceedings.mlr.press/v250/gu24a.html)), and optical flow localises catheter electrodes ([OFELIA](https://proceedings.mlr.press/v250/wang24b.html)).

*Taken together*, the group handles plausibility through parameterisation (free-form deformations, diffeomorphisms, Lipschitz SDFs, physics losses) rather than through guarantees. Still open:
- Much of the validation is synthetic: the XCAT phantom, simplified meshes and in-silico cohorts.
- The trade-off between amortised and per-pair INR optimisation is not settled.
- How to judge registration quality is itself still an open question, as SiamRegQC shows.

**Representative papers**

- [SINR: Spline-enhanced implicit neural representation for multi-modal registration](https://proceedings.mlr.press/v250/sideri-lampretsa24a.html)
- [REINDIR: Repeated Embedding Infusion for Neural Deformable Image Registration](https://proceedings.mlr.press/v250/harten24a.html)
- [Multi-Objective Learning for Deformable Image Registration](https://proceedings.mlr.press/v250/grewal24a.html)
- [FluidRegNet: Longitudinal registration of retinal OCT images with new pathological fluids](https://proceedings.mlr.press/v250/andresen24a.html)
- [Shape of my heart: Cardiac models through learned signed distance functions](https://proceedings.mlr.press/v250/verhulsdonk24a.html)
- [Finite Volume Informed Graph Neural Network for Myocardial Perfusion Simulation](https://proceedings.mlr.press/v250/chou24a.html)
- [Accelerating physics-informed neural fields for fast CT perfusion analysis in acute ischemic stroke](https://proceedings.mlr.press/v250/vries24a.html)
- [Real-time MR-based 3D motion monitoring using raw k-space data](https://proceedings.mlr.press/v250/krusen24a.html)

### Reconstruction, denoising and acquisition

**The ideas behind this group** (a synthesis from the abstracts of about 11 papers in this group; the individual papers have not been read in full).

*The shared problem is an inverse problem without clean paired targets.* Low dose, coarse slices, signal saturation, bias fields and artefacts degrade the measurement, and a matched clean version is rarely available for training. The papers either simulate the degradation, rely on a generative prior, or write an unsupervised objective, and two of them ask who should contribute training data at all.

*The first thread learns from a simulated forward model of the acquisition.* Spine MRI super-resolution builds synthetic pairs by modelling the k-space point spread function, the interslice gap, bias fields and motion artefacts. This trains diffusion models on scaling factors larger than 6× without real paired data ([axial–sagittal shift](https://proceedings.mlr.press/v250/graf24a.html)). Saturated arterial input functions in perfusion cardiac MRI are corrected by blind deconvolution trained on over 3×10^6 simulated curves. In 43 patients, R2 against the unsaturated reference rises from 0.26 to 0.86 ([deep blind AIF](https://proceedings.mlr.press/v250/rebbah24a.html)). Motion is estimated straight from k-space spokes without reconstructing an image first ([k-space motion](https://proceedings.mlr.press/v250/krusen24a.html)).

*The second thread uses diffusion models as priors for restoration.* They denoise Poisson noise in planar X-ray in real time ([diffusion X-ray denoising](https://proceedings.mlr.press/v250/sanderson24a.html)), super-resolve chest X-rays while removing artefacts and prioritising disease regions ([anomaly-focused SISR](https://proceedings.mlr.press/v250/roy24a.html)), and inpaint histopathology artefacts ([HARP](https://proceedings.mlr.press/v250/fuchs24a.html)). Super-resolution even emerges from a latent diffusion model conditioned on resolution and field of view ([resolution-invariant LDM](https://proceedings.mlr.press/v250/patel24a.html)).

*The third thread writes the physics or image statistics into the loss.* MR bias correction minimises a cost built from image entropy, bias-field derivatives and corrected-image statistics, and is compared with N4 ([unsupervised bias correction](https://proceedings.mlr.press/v250/perez-caballero24a.html)). Physics-informed neural fields fit noisy CT perfusion data directly ([ReSPPINN](https://proceedings.mlr.press/v250/vries24a.html)).

*The fourth thread asks which data a reconstruction model should learn from.* For low-dose CT (LDCT) across organs, federated learning shares only high-frequency components and keeps low-frequency characteristics local ([FedFDD](https://proceedings.mlr.press/v250/chen24a.html)). In MRI reconstruction, combining organ datasets can cause hallucinations. Machine unlearning removes them without full retraining, even with only a subset of the retained data ([Erase to Enhance](https://proceedings.mlr.press/v250/xue24a.html)).

*Taken together*, the group prefers simulated degradations and generative priors to classical inverse-problem formulations, and few abstracts mention data consistency with the measurement. Still open:
- The unlearning result shows that mixing training data can produce hallucinations, yet the diffusion restorers rarely test for them.
- Evaluation leans on image-quality metrics rather than downstream diagnostic effect.
- The simulated forward models are not checked against real acquisitions except in the perfusion study.

**Representative papers**

- [Modeling the acquisition shift between axial and sagittal MRI for diffusion superresolution to enable axial spine segmentation](https://proceedings.mlr.press/v250/graf24a.html)
- [Deep blind arterial input function: signal correction in perfusion cardiac magnetic resonance](https://proceedings.mlr.press/v250/rebbah24a.html)
- [Erase to Enhance: Data-Efficient Machine Unlearning in MRI Reconstruction](https://proceedings.mlr.press/v250/xue24a.html)
- [FedFDD: Federated Learning with Frequency Domain Decomposition for Low-Dose CT Denoising](https://proceedings.mlr.press/v250/chen24a.html)
- [Diffusion X-ray image denoising](https://proceedings.mlr.press/v250/sanderson24a.html)
- [Anomaly-focused Single Image Super-resolution with Artifact Removal for Chest X-rays using Distribution-aware Diffusion Model](https://proceedings.mlr.press/v250/roy24a.html)
- [Unsupervised Deep Learning Method for Bias Correction](https://proceedings.mlr.press/v250/perez-caballero24a.html)

### Trustworthiness

**The ideas behind this group** (a synthesis from the abstracts of about 16 papers in this group; the individual papers have not been read in full).

*The shared problem is that good aggregate test performance does not show that a model is right for the right reasons, equally safe for every patient group, honest about its errors, or safe to train on shared data.* The papers fall into five threads.

*The first thread finds and removes shortcuts and confounders.* Prediction depth combined with KL divergence locates the layers where a shortcut feature appears. Shortcuts that are visually more complex appear deeper in the network, and the learning rate interacts with how much the model relies on them ([shortcut layers](https://proceedings.mlr.press/v250/boland24a.html)). A two-step test shows that brain shape left after skull stripping acts as a confounder, and deformable registration removes it ([brain shape](https://proceedings.mlr.press/v250/dauchelle24a.html)). Models trained on CheXpert and MIMIC-CXR have image-level prediction tendencies that can be inferred above chance, and attributes such as position or sex do not explain them ([dataset tendencies](https://proceedings.mlr.press/v250/hoebel24a.html)). A pretrained artefact detector guides diffusion counterfactuals so that pathology markers change while support devices stay the same, and augmenting with these images helps empirical risk minimisation (ERM) and Group-DRO on under-represented groups ([DeCoDEx](https://proceedings.mlr.press/v250/fathi24a.html)).

*The second thread looks for explanations other than saliency maps.* Natural-language explanations are generated with parameter-efficient fine-tuning (PEFT) that trains 12% of the parameters, and supervision in the encoder latent space points towards adversarial training without reinforcement learning ([NLE with PEFT](https://proceedings.mlr.press/v250/rio-torto24a.html)). Explanation methods for transformer retrieval are compared with insertion–deletion metrics ([retrieval transformers](https://proceedings.mlr.press/v250/susmitha24a.html)), and saliency maps are registered to a template to give group-level maps across a cohort ([cohort saliency](https://proceedings.mlr.press/v250/ahmad24a.html)).

*The third thread questions how uncertainty is evaluated and used.* Structure size confounds the correlation between uncertainty and Dice, and partial correlation gives lower and more honest estimates ([structure size](https://proceedings.mlr.press/v250/geissler24a.html)). Ensemble uncertainty picks the patches to review by hand so that each image reaches a target Dice ([patch correction](https://proceedings.mlr.press/v250/kohler24a.html)). Probabilistic segmentations fed into a clinical glaucoma pipeline improve diagnosis ([probabilistic glaucoma](https://proceedings.mlr.press/v250/wundram24a.html), annotated in [this repo](../2024-wundram-glaucoma-probabilistic-pipeline/index.html)), and retinal layer shapes carry Gaussian uncertainty ([probabilistic SDF](https://proceedings.mlr.press/v250/islam24a.html)).

*The fourth thread treats fairness as something an attacker can exploit.* Label poisoning aimed at one demographic group causes underdiagnosis bias that does not show in overall performance and carries over to external data ([adversarial bias attacks](https://proceedings.mlr.press/v250/kulkarni24a.html)). Fusing skin images with patient metadata reduces, but does not remove, the performance gap across skin tones ([skin fusion fairness](https://proceedings.mlr.press/v250/wang24c.html)).

*The fifth thread is privacy.* Implicit neural obfuscation with k-anonymity mixing gives up to 3 times reduced privacy risks while keeping segmentation quality ([neural obfuscation](https://proceedings.mlr.press/v250/heinrich24a.html)). Other routes are hyperparameter-free synthetic data sharing ([HyFree-S3](https://proceedings.mlr.press/v250/chebykin24a.html)) and machine unlearning ([Erase to Enhance](https://proceedings.mlr.press/v250/xue24a.html)).

*Taken together*, the group's recurring finding is that summary metrics hide failures: shortcuts, subgroup harm, and uncertainty scores that only look calibrated because of structure size. Still open:
- Confounder detection is post hoc and tied to particular datasets.
- Privacy is measured with re-identification proxies rather than formal guarantees.
- No paper combines explanation, uncertainty and fairness into a single test.

**Representative papers**

- [There Are No Shortcuts to Anywhere Worth Going: Identifying Shortcuts in Deep Learning Models for Medical Image Analysis](https://proceedings.mlr.press/v250/boland24a.html)
- [An unexpected confounder: how brain shape can be used to classify MRI scans ?](https://proceedings.mlr.press/v250/dauchelle24a.html)
- [DeCoDEx: Confounder Detector Guidance for Improved Diffusion-based Counterfactual Explanations](https://proceedings.mlr.press/v250/fathi24a.html)
- [Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations](https://proceedings.mlr.press/v250/kulkarni24a.html)
- [Structure Size as Confounder in Uncertainty Based Segmentation Quality Prediction](https://proceedings.mlr.press/v250/geissler24a.html)
- [Parameter-Efficient Generation of Natural Language Explanations for Chest X-ray Classification](https://proceedings.mlr.press/v250/rio-torto24a.html)
- [Implicit neural obfuscation for privacy preserving medical image sharing](https://proceedings.mlr.press/v250/heinrich24a.html)
- [Beyond Structured Attributes: Image-Based Predictive Trends for Chest X-Ray Classification](https://proceedings.mlr.press/v250/hoebel24a.html)

### Federated, continual and collaborative learning

**The ideas behind this group** (a synthesis from the abstracts of about 8 papers in this group; the individual papers have not been read in full).

*The shared problem is that medical data sit in separate, heterogeneous silos and keep arriving over time, while privacy rules prevent pooling them.* The papers handle heterogeneity between sites, unreliable participants, learning in sequence, and joint training on databases that do not share a label or modality set.

*The first thread measures and exploits non-identically distributed (non-IID) data rather than just suffering from it.* Distances in image-embedding space, combined with metadata statistics, quantify how far each client is from the rest. Federated averaging then down-weights the most distant client as an outlier, and distance-based clustering gives personalised models. This is tested on FeTS, Prostate and Fed-KITS2019 ([distance-aware FL](https://proceedings.mlr.press/v250/alekseenko24a.html)). For LDCT denoising, heterogeneity appears as different frequency patterns. Sharing high frequencies and keeping low frequencies local beats both local and centralised models ([FedFDD](https://proceedings.mlr.press/v250/chen24a.html)).

*The second thread defends the aggregate against bad clients.* Clients whose updates are malfunctioning, whether from technical faults, poor data or attacks, are excluded by angular distance. The method needs no hyperparameters and no knowledge of how many clients are bad ([ASMR](https://proceedings.mlr.press/v250/konstantin24a.html)).

*The third thread adds time.* Asynchronous decentralised federated lifelong learning lets agents learn continually from their own past and from peers, applied to landmark localisation with deep reinforcement learning ([ADFLL](https://proceedings.mlr.press/v250/zheng24a.html)). SAM's natural-image pretraining is used to balance rigidity against plasticity in continual prostate segmentation, compared with Lifelong nnU-Net ([UnCLe SAM](https://proceedings.mlr.press/v250/ranem24a.html)).

*The fourth thread trains jointly on databases that do not line up.* One model is trained on 7 databases covering 5 brain pathologies with different MRI modality sets, which is shown to be feasible and helpful ([joint MRI databases](https://proceedings.mlr.press/v250/xu24a.html)). Data can also be shared indirectly: through hyperparameter-free distributed synthesis that improves site-specific segmentation ([HyFree-S3](https://proceedings.mlr.press/v250/chebykin24a.html)), or through a StyleGAN that learns from several sources with different available covariates ([Multi-Source StyleGAN](https://proceedings.mlr.press/v250/lai24a.html)).

*Taken together*, the group moves from treating heterogeneity as noise to measuring it and building it into aggregation or model structure. Still open:
- None of the abstracts gives convergence or generalisation analysis under the measured heterogeneity.
- Federations are small and usually simulated from public datasets.
- Continual learning is tested on narrow task sequences.
- Federated and continual settings are combined only in ADFLL.

**Representative papers**

- [Distance-Aware Non-IID Federated Learning for Generalization and Personalization in Medical Imaging Segmentation](https://proceedings.mlr.press/v250/alekseenko24a.html)
- [ASMR: Angular Support for Malfunctioning Client Resilience in Federated Learning](https://proceedings.mlr.press/v250/konstantin24a.html)
- [FedFDD: Federated Learning with Frequency Domain Decomposition for Low-Dose CT Denoising](https://proceedings.mlr.press/v250/chen24a.html)
- [Towards a Collective Medical Imaging AI: Enabling Continual Learning from Peers](https://proceedings.mlr.press/v250/zheng24a.html)
- [UnCLe SAM: Unleashing SAM's Potential for Continual Prostate MRI Segmentation](https://proceedings.mlr.press/v250/ranem24a.html)
- [Feasibility and benefits of joint learning from MRI databases with different brain diseases and modalities for segmentation](https://proceedings.mlr.press/v250/xu24a.html)

### Vision-language models and reports

**The ideas behind this group** (a synthesis from the abstracts of about 9 papers in this group; the individual papers have not been read in full).

*The shared problem is that language now sits beside the image, but nobody is sure how much it adds or how to check it.* Radiology reports, questions about pairs of images, text prompts and natural-language explanations all promise richer supervision and output than a class label. Each one also brings an evaluation problem: free text is hard to score, and a model may ignore the text it is given. The papers fall into four threads.

*The first thread generates clinical text or judges it.* A fine-tuned ChatGPT writes the 'Impression' section of Dutch CT-thorax reports. Two radiologists rate the original impressions higher on correctness, completeness and conciseness, and GPT-4 used as a judge favours the AI-generated text ([ChatGPT impressions](https://proceedings.mlr.press/v250/builtjes24a.html)). Natural-language explanations (NLEs) for chest X-ray classification come from an encoder–decoder tuned with parameter-efficient fine-tuning (PEFT). Feeding visual information into the generator layers beats full fine-tuning while training only 12% of the parameters, and generating in the continuous encoder latent space is a step towards adversarial training without reinforcement learning ([NLE with PEFT](https://proceedings.mlr.press/v250/rio-torto24a.html)).

*The second thread builds vision-language models (VLMs) for comparing images.* Difference visual question answering (diff-VQA) asks what has changed between two chest X-rays of the same patient. A VLM is pretrained first on natural image–text pairs and then on longitudinal X-ray pairs with question–answer sets and reports. It beats task-specific architectures on diff-VQA and also on single-image VQA ([PLURAL](https://proceedings.mlr.press/v250/cho24a.html)).

*The third thread asks whether text actually helps image tasks.* A systematic transfer study of vision-language segmentation models (VLSMs) on 11 curated 2D datasets finds them competitive with image-only models after fine-tuning. However, not all of them use the language prompt, and image features dominate ([VLSM transfer](https://proceedings.mlr.press/v250/poudel24a.html)). Text-to-image diffusion is adapted cheaply: Textual Inversion trains only the text embeddings of Stable Diffusion, on 100 samples per modality, and the embeddings are under 1 MB, which makes sharing easier ([Textual Inversion](https://proceedings.mlr.press/v250/wilde24a.html)). Radiologists' eye-gaze patterns condition a text-to-image diffusion model whose density estimates are used for zero-shot classification of pneumonia and tuberculosis ([GazeDiff](https://proceedings.mlr.press/v250/bhattacharya24a.html)). For evaluation, an open toolkit covers six multi-modal tasks over 23 datasets, spanning over 11 medical domains ([MultiMedEval](https://proceedings.mlr.press/v250/royer24a.html)).

*The fourth thread uses reports as a cheap but coarse source of labels.* Natural-language processing (NLP) extracts binary labels from reports to train classifiers. The classifiers' raw outputs rank finding severity with Spearman's rank coefficients between 0.563 - 0.848 against manual rankings, and a semi-supervised mix with a few fully ranked images does better still ([severity ranking](https://proceedings.mlr.press/v250/macpherson24a.html)). Fazekas scores extracted from reports serve as a weak reference for a brain anomaly detector in a clinical data warehouse ([β-VAE in clinical routine](https://proceedings.mlr.press/v250/loizillon24a.html)).

*Taken together*, the group is early and centred on chest X-ray. Language enters as an output, a prompt, a conditioning signal or a label source, but rarely as something the model is shown to depend on. Still open:
- Generated text is judged by small reader studies or by large language model (LLM) judges that turn out to be biased.
- The finding that segmentation models ignore their prompts is not tested in the other papers.
- No paper links generated text to a downstream clinical decision.

**Representative papers**

- [Exploring Transfer Learning in Medical Image Segmentation using Vision-Language Models](https://proceedings.mlr.press/v250/poudel24a.html)
- [Evaluating ChatGPT's Performance in Generating and Assessing Dutch Radiology Report Impressions](https://proceedings.mlr.press/v250/builtjes24a.html)
- [Pretraining Vision-Language Model for Difference Visual Question Answering in Longitudinal Chest X-rays](https://proceedings.mlr.press/v250/cho24a.html)
- [MultiMedEval: A Benchmark and a Toolkit for Evaluating Medical Vision-Language Models](https://proceedings.mlr.press/v250/royer24a.html)
- [Parameter-Efficient Generation of Natural Language Explanations for Chest X-ray Classification](https://proceedings.mlr.press/v250/rio-torto24a.html)
- [GazeDiff: A radiologist visual attention guided diffusion model for zero-shot disease classification](https://proceedings.mlr.press/v250/bhattacharya24a.html)
- [Automated ranking of chest x-ray radiological finding severity in a binary label setting](https://proceedings.mlr.press/v250/macpherson24a.html)

### Longitudinal and multimodal clinical prediction

**The ideas behind this group** (a synthesis from the abstracts of about 10 papers in this group; the individual papers have not been read in full).

*The shared problem is that a clinical question rarely rests on one image taken at one time.* Disease changes between visits, and other data such as metadata, physiological signals or simulations also bear on the answer. Some modalities are often missing, and the target is usually a future outcome rather than a present label. The papers fall into three threads.

*The first thread models change over time.* For age-related macular degeneration (AMD), a stage classifier is trained jointly with a neural ordinary differential equation (Neural-ODE) that models the disease trajectory in embedding space. The scan's distance from the decision hyperplane is tied inversely to time-to-conversion, and risk scores must be consistent across scans of the same eye. On 235 eyes (3,534 OCT scans) with 40 converters, the model reaches an average area under the ROC of 0.84 and a Concordance Index of 0.78 ([AMD progression](https://proceedings.mlr.press/v250/chakravarty24a.html)). When fluid appears between optical coherence tomography (OCT) visits, a "fluid seed" restores correspondence for longitudinal registration ([FluidRegNet](https://proceedings.mlr.press/v250/andresen24a.html)). Pairs of chest X-rays are compared through question answering ([PLURAL](https://proceedings.mlr.press/v250/cho24a.html)).

*The second thread fuses heterogeneous inputs and tolerates missing ones.* One framework aligns every modality to the space of the richest "anchor" modality and uses a masked multimodal transformer to handle missing inputs. It is applied to stress detection and to fighter pilots' g-force-induced loss of consciousness ([ADAPT](https://proceedings.mlr.press/v250/mordacq24a.html)). A brain-lesion segmenter is trained jointly on 7 databases with 5 types of brain pathologies and different sets of MRI modalities ([joint learning across databases](https://proceedings.mlr.press/v250/xu24a.html)). For skin malignancy, fusing images with patient metadata, including skin tone, works best in multiplicative form. It reduces, but does not remove, the performance gap between skin tones ([skin image and metadata fusion](https://proceedings.mlr.press/v250/wang24c.html)). Multiple focal-plane images of an embryo are fused to predict implantation outcome ([MFIF-Net](https://proceedings.mlr.press/v250/cheng24a.html)).

*The third thread predicts outcomes from simulation or from routine data.* Cardiac digital twins of 1000 virtual patients supply features before therapy and outcome labels after it. A Siamese fusion network then predicts whether atrial fibrillation stops under each treatment strategy ([digital twins](https://proceedings.mlr.press/v250/zolotarev24a.html)). In Barrett's oesophagus, multiple instance learning (MIL) on routine H&E slides predicts aneuploidy with 0.93 AUC, and the predicted status correlates with progression to adenocarcinoma (p=6.55e-06) ([aneuploidy from histology](https://proceedings.mlr.press/v250/ercan24a.html)). Severity, rather than presence, is recovered from classifiers trained on binary labels ([severity ranking](https://proceedings.mlr.press/v250/macpherson24a.html)).

*Taken together*, the group moves from "what is in this image" towards "what will happen to this patient", usually by adding a second scan or a second modality rather than a full trajectory. Still open:
- Time is mostly handled as pairs of visits; only the Neural-ODE models a continuous trajectory.
- Digital-twin labels are simulated, so in-silico accuracy says little yet about real patients.
- Missing-modality and metadata-fusion methods rarely ask whether the gains are shared fairly across groups; the skin-tone paper is the exception.

**Representative papers**

- [Predicting Age-related Macular Degeneration Progression from Retinal Optical Coherence Tomography with Intra-Subject Temporal Consistency](https://proceedings.mlr.press/v250/chakravarty24a.html)
- [ADAPT: Multimodal Learning for Detecting Physiological Changes under Missing Modalities](https://proceedings.mlr.press/v250/mordacq24a.html)
- [Predicting Atrial Fibrillation Treatment Outcome with Siamese Multi-modal Fusion and Cardiac Digital Twins](https://proceedings.mlr.press/v250/zolotarev24a.html)
- [Skin Malignancy Classification Using Patients' Skin Images and Meta-data: Multimodal Fusion for Improving Fairness](https://proceedings.mlr.press/v250/wang24c.html)
- [Feasibility and benefits of joint learning from MRI databases with different brain diseases and modalities for segmentation](https://proceedings.mlr.press/v250/xu24a.html)
- [FluidRegNet: Longitudinal registration of retinal OCT images with new pathological fluids](https://proceedings.mlr.press/v250/andresen24a.html)
- [Predicting DNA Content Abnormalities in Barrett's Esophagus: A Weakly Supervised Learning Paradigm](https://proceedings.mlr.press/v250/ercan24a.html)

### Surgery, video and ultrasound

**The ideas behind this group** (a synthesis from the abstracts of about 10 papers in this group; the individual papers have not been read in full).

*The shared problem is that video and other moving images produce many frames but few labels, and often have to run in real time.* Surgical, endoscopic, ultrasound and fluoroscopic sequences are cheap to record, expensive to annotate, and differ from natural images and video. Motion between frames is both a nuisance and a free signal. The papers fall into four threads.

*The first thread learns from unlabelled surgical video.* Optical flow weights pixel pairs in a contrastive InfoNCE loss, so that pixels moving together get similar representations. The resulting features stay competitive with over 92% fewer annotated samples, and with as little as 99.73% fewer for segmentation ([Laparoflow-SSL](https://proceedings.mlr.press/v250/moens24a.html)). Augmentation policies for contrastive learning are searched using the local dimensionality of the representation as a proxy. Hue, which helps on natural images, turns out not to help on laparoscopic images ([DDA](https://proceedings.mlr.press/v250/zhou24b.html)). For real-time use, several anatomy-specific teachers are distilled into one small student on partially labelled data ([multi-teacher distillation](https://proceedings.mlr.press/v250/maack24a.html)).

*The second thread uses motion for temporal consistency.* Video polyp segmentation combines a pretrained 2D model at test time with an implicit network and optical-flow temporal losses ([implicit video polyp segmentation](https://proceedings.mlr.press/v250/dahan24c.html)). A frame-level pyramid-attention network reaches 94.2% Dice and 89.7% IoU on Kvasir ([PAAN](https://proceedings.mlr.press/v250/yi24a.html)). In X-ray sequences during catheter ablation, fine-tuned optical-flow maps help localise electrodes, with a mean radial error (MRE) of 0.95 mm for radiofrequency catheters and 0.71 mm for coronary sinus catheters ([OFELIA](https://proceedings.mlr.press/v250/wang24b.html)).

*The third thread builds pretraining for modalities that ImageNet does not cover.* A masked autoencoder (MAE) trained jointly on ultrasound images and videos uses 100,000+ videos and images across many organs. An ultrasound-specific confidence map guides the masking ([UltraMAE](https://proceedings.mlr.press/v250/rahman24a.html)). In the other direction, video-pretrained vision transformers are adapted to CT through task-specific self-supervised pretraining. This improves vertebral fracture detection by 8.3 in F1, with a training speedup of factor 2 over naive transfer from Video MAE ([Video-CT MAE](https://proceedings.mlr.press/v250/buess24a.html)).

*The fourth thread handles fetal imaging, where the subject moves.* One network jointly estimates rigid fetal motion and local geometric distortion in fetal echo-planar images, using a closed-form rigid update and a diffeomorphic deformation model ([fetal motion correction](https://proceedings.mlr.press/v250/wang24a.html)). Feature-wise linear modulation (FiLM) conditions a multi-organ network on partial, region-based labels for low-resolution fetal MRI ([CoNeMOS](https://proceedings.mlr.press/v250/billot24a.html)).

*Taken together*, optical flow keeps returning as the cheap supervisory signal, both for pretraining and for consistency between frames. Pretraining recipes tuned to the domain beat generic ones. Still open:
- Fetal work is MRI only, with no obstetric ultrasound.
- Ultrasound pretraining is not compared with the surgical-video SSL recipes.
- Real-time claims are made for architectures, and few abstracts report latency on deployment hardware.
- Temporal models are tested mostly on short public benchmarks.

**Representative papers**

- [Laparoflow-SSL: Image Analysis From a Tiny Dataset Through Self-Supervised Transformers Leveraging Unlabeled Surgical Video](https://proceedings.mlr.press/v250/moens24a.html)
- [DDA: Dimensionality Driven Augmentation Search for Contrastive Learning in Laparoscopic Surgery](https://proceedings.mlr.press/v250/zhou24b.html)
- [UltraMAE: Multi-modal Masked Autoencoder for Ultrasound Pre-training](https://proceedings.mlr.press/v250/rahman24a.html)
- [Video-CT MAE: Self-supervised Video-CT Domain Adaptation for Vertebral Fracture Diagnosis](https://proceedings.mlr.press/v250/buess24a.html)
- [Video Polyp Segmentation using Implicit Networks](https://proceedings.mlr.press/v250/dahan24c.html)
- [OFELIA: Optical Flow-based Electrode LocalIzAtion](https://proceedings.mlr.press/v250/wang24b.html)
- [Joint Motion Estimation with Geometric Deformation Correction for Fetal Echo Planar Images Via Deep Learning](https://proceedings.mlr.press/v250/wang24a.html)
- [Efficient Anatomy Segmentation in Laparoscopic Surgery using Multi-Teacher Knowledge Distillation](https://proceedings.mlr.press/v250/maack24a.html)

### Benchmarks, data and meta-research

**The ideas behind this group** (a synthesis from the abstracts of about 16 papers in this group; the individual papers have not been read in full).

*The shared problem is that many reported gains depend on how the comparison was set up.* The choice of pretraining, fine-tuning method, loss, evaluation crop, correlation measure or human reference can change the conclusion. A sizeable part of MIDL 2024 tests these choices directly instead of proposing a new model. The papers fall into four threads.

*The first thread benchmarks pretraining and adaptation.* Ten supervised and self-supervised models are compared for multi-view chest X-ray at 1%, 10% and 100% of the training data. They beat earlier multi-view methods by as much as 9.9%, 8.8% and 1.6% respectively ([multi-view chest X-ray benchmark](https://proceedings.mlr.press/v250/afzal24a.html)). Seventeen parameter-efficient fine-tuning (PEFT) algorithms are tested in over 700 controlled experiments on six datasets, with gains of up to 22% in low-data regimes ([PEFT, the missed opportunity](https://proceedings.mlr.press/v250/dutt24a.html)). Pretraining "in the wild" on large unrelated data makes Swin transformers more robust to acquisition differences than pretraining on the task's own data. Centred kernel alignment (CKA) shows less feature reuse near the final encoder layers ([pretraining in the wild](https://proceedings.mlr.press/v250/jiang24b.html)). Similar systematic studies cover vision-language segmentation ([VLSM transfer](https://proceedings.mlr.press/v250/poudel24a.html)), VLM evaluation ([MultiMedEval](https://proceedings.mlr.press/v250/royer24a.html)) and transformers for retrieval ([retrieval analysis](https://proceedings.mlr.press/v250/susmitha24a.html)).

*The second thread re-examines evaluation protocols.* Domain-generalisation methods for optic disc and cup segmentation are usually tested on images cropped around the optic nerve head. On uncropped images they degrade markedly, and a Noisy Student model trained on 18.000 pseudo-labelled images does better ([Noisy Students](https://proceedings.mlr.press/v250/moris24a.html)). Structure size confounds the usual correlation between uncertainty and Dice, and a partial correlation coefficient gives lower, more honest estimates ([structure size as confounder](https://proceedings.mlr.press/v250/geissler24a.html)). With nnU-Net as the model, most active-learning strategies do beat random sampling ([nnU-Net active learning](https://proceedings.mlr.press/v250/follmer24a.html)). Losses derived from Matthews correlation coefficient (MCC) and F1, combined with cross-entropy, give up to +12% balanced accuracy and up to +51% minority-class F1 ([imbalance-aware losses](https://proceedings.mlr.press/v250/scholz24a.html)). Models trained on CheXpert and on MIMIC-CXR perform similarly overall but differ systematically, image by image, in ways that image statistics do not explain ([predictive trends](https://proceedings.mlr.press/v250/hoebel24a.html)).

*The third thread puts humans in the comparison.* A dose predictor beats three radiation oncologists at judging the dosimetric impact of 54 contour variants ([oncologists versus dose predictor](https://proceedings.mlr.press/v250/kamath24a.html)). A visual Turing test with radiologists is found inadequate for judging synthetic brain MRI, while segmenting hallmarks reveals lateral-ventricle discrepancies ([synthetic MRI consistency](https://proceedings.mlr.press/v250/yassin24a.html)). GPT-4 as a judge disagrees with radiologists ([ChatGPT impressions](https://proceedings.mlr.press/v250/builtjes24a.html)). A β-variational autoencoder (β-VAE) anomaly detector is tried on a heterogeneous clinical data warehouse and judged not ready for clinical routine ([β-VAE in clinical routine](https://proceedings.mlr.press/v250/loizillon24a.html)).

*The fourth thread audits the data itself.* Tracking 20 public datasets in MICCAI and MIDL papers between 2013 and 2023 shows that usage is concentrated on a few datasets. Citation practices are inconsistent enough to make automatic tracking difficult ([Citation needed](https://proceedings.mlr.press/v250/sourget24a.html)).

*Taken together*, the group makes the conference more self-critical than its method papers suggest: cropping, confounded correlations and LLM judges can each inflate results. Still open:
- Most benchmarks fix one modality or task family, so their recommendations may not transfer.
- Human references are small: two or three experts at most.
- Few method papers in the other groups adopt the corrected protocols proposed here.

**Representative papers**

- [[Citation needed] Data usage and citation practices in medical imaging conferences](https://proceedings.mlr.press/v250/sourget24a.html)
- [Parameter-Efficient Fine-Tuning for Medical Image Analysis: The Missed Opportunity](https://proceedings.mlr.press/v250/dutt24a.html)
- [A Comprehensive Benchmark of Supervised and Self-supervised Pre-training on Multi-view Chest X-ray Classification](https://proceedings.mlr.press/v250/afzal24a.html)
- [Structure Size as Confounder in Uncertainty Based Segmentation Quality Prediction](https://proceedings.mlr.press/v250/geissler24a.html)
- [Semi-supervised learning with Noisy Students improves domain generalization in optic disc and cup segmentation in uncropped fundus images](https://proceedings.mlr.press/v250/moris24a.html)
- [Comparing the Performance of Radiation Oncologists versus a Deep Learning Dose Predictor to Estimate Dosimetric Impact of Segmentation Variations for Radiotherapy](https://proceedings.mlr.press/v250/kamath24a.html)
- [Imbalance-aware loss functions improve medical image classification](https://proceedings.mlr.press/v250/scholz24a.html)
- [Beyond Structured Attributes: Image-Based Predictive Trends for Chest X-Ray Classification](https://proceedings.mlr.press/v250/hoebel24a.html)

## Outlier papers

Papers that sit outside the main groups in topic, method or framing.

### Unusual subjects or data

- [ADAPT: Multimodal Learning for Detecting Physiological Changes under Missing Modalities](https://proceedings.mlr.press/v250/mordacq24a.html): not about patients at all; it detects induced stress and fighter pilots' loss of consciousness under g-forces, fusing video with biomedical signals.
- [IST-editing: Infinite spatial transcriptomic editing in a generated gigapixel mouse pup](https://proceedings.mlr.press/v250/wu24a.html): a whole generated mouse pup at 106496 × 53248 resolution, in which edited gene expression produces cell-, tissue- and animal-level changes in form.
- [Improving CNN-Based Mitosis Detection through Rescanning Annotated Glass Slides and Atypical Mitosis Subtyping](https://proceedings.mlr.press/v250/fick24a.html): a mitosis detector trained only on animal data but competitive on human breast cancer, with scanner robustness gained by physically rescanning the glass slides.
- [MFIF-Net: A Multi-Focal Image Fusion Network for Implantation Outcome Prediction of Blastocyst](https://proceedings.mlr.press/v250/cheng24a.html): in vitro fertilisation, where several focal-plane images of one embryo are fused to predict whether it implants.
- [Multi-scale Stochastic Generation of Labelled Microscopy Images for Neuron Segmentation](https://proceedings.mlr.press/v250/decroocq24a.html): basic neuroscience; synthetic microscopy images of neurons are generated from single-neuron tractograms to train dendrite tracing.
- [Lupus Nephritis Subtype Classification with only Slide Level Labels](https://proceedings.mlr.press/v250/sharma24a.html): a rarely studied kidney disease, with a multi-stained slide dataset from the Indian population described as the largest of its kind.

### Social or ethical findings

- [Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations](https://proceedings.mlr.press/v250/kulkarni24a.html): label poisoning aimed at one demographic group causes underdiagnosis in that group without changing overall performance, so the attack goes unnoticed.
- [Implicit neural obfuscation for privacy preserving medical image sharing](https://proceedings.mlr.press/v250/heinrich24a.html): a chest X-ray can serve as a fingerprint even after anonymisation; overfitting a network more than 6x smaller than the images, plus k-anonymity mixing between patients, cuts privacy risk up to 3 times.
- [Skin Malignancy Classification Using Patients' Skin Images and Meta-data: Multimodal Fusion for Improving Fairness](https://proceedings.mlr.press/v250/wang24c.html): adding skin tone as an input narrows, but does not remove, the performance gap between patients with different skin tones.
- [Erase to Enhance: Data-Efficient Machine Unlearning in MRI Reconstruction](https://proceedings.mlr.press/v250/xue24a.html): machine unlearning, driven by privacy regulation, is applied to image reconstruction and used to remove the hallucinations caused by mixing organs in training.

### Unusual mathematics or methods

- [Finite Volume Informed Graph Neural Network for Myocardial Perfusion Simulation](https://proceedings.mlr.press/v250/chou24a.html): a finite volume discretisation of Darcy's equations, used as the loss of a graph network on myocardium meshes.
- [Multi-Objective Learning for Deformable Image Registration](https://proceedings.mlr.press/v250/grewal24a.html): registration treated as a multi-objective problem; one network returns a whole set of trade-offs between the objectives for each patient instead of a single answer.
- [Active Learning with the nnUNet and Sample Selection with Uncertainty-Aware Submodular Mutual Information Measure](https://proceedings.mlr.press/v250/follmer24a.html): batches for annotation chosen with submodular mutual information combined with uncertainty.
- [DDA: Dimensionality Driven Augmentation Search for Contrastive Learning in Laparoscopic Surgery](https://proceedings.mlr.press/v250/zhou24b.html): the local dimensionality of the learned representations is the target of a differentiable search for augmentation policies.
- [ASMR: Angular Support for Malfunctioning Client Resilience in Federated Learning](https://proceedings.mlr.press/v250/konstantin24a.html): faulty federated clients excluded by the angular distance between their updates, with no hyperparameters and no need to know how many clients are faulty.
- [Learned morphological features guide cell type assignment of deconvolved spatial transcriptomics](https://proceedings.mlr.press/v250/chelebian24a.html): a hierarchical permutation framework that uses tissue morphology to reassign cell types to nuclei, which existing methods assign at random within each spot.

### Papers that critique the field

- [[Citation needed] Data usage and citation practices in medical imaging conferences](https://proceedings.mlr.press/v250/sourget24a.html): tracks how 20 public datasets were used in MICCAI (Medical Image Computing and Computer Assisted Intervention) and MIDL papers from 2013 to 2023; use is concentrated on a few datasets, and inconsistent citation makes it hard to track automatically.
- [Structure Size as Confounder in Uncertainty Based Segmentation Quality Prediction](https://proceedings.mlr.press/v250/geissler24a.html): both Dice and the usual uncertainty measures depend on structure size, so their reported correlation is inflated; partial correlation gives lower estimates.
- [An unexpected confounder: how brain shape can be used to classify MRI scans ?](https://proceedings.mlr.press/v250/dauchelle24a.html): after skull stripping, brain shape alone lets classifiers tell databases apart; only deformable, not affine, registration removes the effect.
- [Semi-supervised learning with Noisy Students improves domain generalization in optic disc and cup segmentation in uncropped fundus images](https://proceedings.mlr.press/v250/moris24a.html): cropping images around the optic nerve head removes most domain variation, so published domain-generalisation results overstate how well methods cope; on uncropped images they degrade badly.
- [Exploring Transfer Learning in Medical Image Segmentation using Vision-Language Models](https://proceedings.mlr.press/v250/poudel24a.html): across 11 datasets, not all vision-language segmentation models make use of the text prompt; image features dominate.
- [Evaluating ChatGPT's Performance in Generating and Assessing Dutch Radiology Report Impressions](https://proceedings.mlr.press/v250/builtjes24a.html): radiologists rate the original impressions above AI-generated ones, while GPT-4 used as a judge favours the AI-generated text.
- [Evaluating Age-Related Anatomical Consistency in Synthetic Brain MRI against Real-World Alzheimer's Disease Data.](https://proceedings.mlr.press/v250/yassin24a.html): radiologists cannot tell synthetic from real scans, yet segmentation shows significant lateral ventricle discrepancies (Ps < 5.0e-2); a visual Turing test alone is not enough.

### Unusual clinical framing

- [Comparing the Performance of Radiation Oncologists versus a Deep Learning Dose Predictor to Estimate Dosimetric Impact of Segmentation Variations for Radiotherapy](https://proceedings.mlr.press/v250/kamath24a.html): contours judged by their effect on the radiation dose rather than by geometry; on 54 contour variants the model beats three expert radiation oncologists.
- [Predicting Atrial Fibrillation Treatment Outcome with Siamese Multi-modal Fusion and Cardiac Digital Twins](https://proceedings.mlr.press/v250/zolotarev24a.html): 1000 virtual patients with simulated electrophysiology; the model predicts whether each ablation strategy would end atrial fibrillation without running the slow simulations.
- [Generating Cerebral Vessel Trees of Acute Ischemic Stroke Patients using Conditional Set-Diffusion](https://proceedings.mlr.press/v250/kuipers24a.html): synthetic vessel-tree populations, conditioned on a middle cerebral artery occlusion, as input to in-silico clinical trials.
- [Automated ranking of chest x-ray radiological finding severity in a binary label setting](https://proceedings.mlr.press/v250/macpherson24a.html): asks whether a classifier trained only on present/absent labels encodes severity; raw outputs reach Spearman's rank coefficients between 0.563 and 0.848 against human rankings.
- [Efficiently correcting patch-based segmentation errors to control image-level performance in retinal images](https://proceedings.mlr.press/v250/kohler24a.html): sets a quality standard for every image rather than on average, and says how many patches a person must review to reach it.
- [Detecting Brain Anomalies in Clinical Routine with the β-VAE: Feasibility Study on Age-Related White Matter Hyperintensities](https://proceedings.mlr.press/v250/loizillon24a.html): tested on a heterogeneous hospital clinical data warehouse against Fazekas scores taken from reports, and honestly concludes the method is not yet ready for clinical routine.

### Imaging as a stand-in for another measurement

Papers that predict, from routine or cheap imaging, something normally measured in a more invasive, expensive or different way.

- [Predicting DNA Content Abnormalities in Barrett's Esophagus: A Weakly Supervised Learning Paradigm](https://proceedings.mlr.press/v250/ercan24a.html): aneuploidy predicted from routine haematoxylin and eosin slides instead of flow cytometry, saving biopsy tissue; 0.93 AUC, and the predictions correlate with progression to cancer (p=6.55e-06), close to flow cytometry (p=2.84e-7).
- [ThickV-Stain: Unprocessed Thick Tissues Virtual Staining for Rapid Intraoperative Histology](https://proceedings.mlr.press/v250/shi24a.html): virtual haematoxylin and eosin staining of unsectioned, unstained thick tissue in less than a minute for 100 million pixels, for use during surgery.
- [Predicting 3D forearm fracture angle from biplanar Xray images with rotational bone pose estimation](https://proceedings.mlr.press/v250/gu24a.html): the true 3D fracture angle recovered from two ordinary X-rays, even when the views are far from orthogonal, instead of from CT.
- [Deep blind arterial input function: signal correction in perfusion cardiac magnetic resonance](https://proceedings.mlr.press/v250/rebbah24a.html): blind deconvolution corrects the saturated arterial input function, standing in for a dual-saturation sequence; agreement (R2) rises from 0.26 to 0.86 in 43 patients.
- [Deformation-aware GAN for Medical Image Synthesis with Substantially Misaligned Pairs](https://proceedings.mlr.press/v250/xin24a.html): synthesises modalities that are "costly, invasive or harmful to acquire" from lung MRI–CT pairs misaligned by breathing.

The [IST-editing paper](https://proceedings.mlr.press/v250/wu24a.html) under *Unusual subjects or data* runs the other way: gene expression is the input and the image is the output.

## Theory picks

The most mathematically foundational papers, ranked for a close reading (from abstracts only).

1. [Finite Volume Informed Graph Neural Network for Myocardial Perfusion Simulation](https://proceedings.mlr.press/v250/chou24a.html): a graph neural network solves Darcy's equations on myocardium meshes, trained with the residual of a finite volume discretisation as the physics-informed loss. It is worth working out how flux conservation across mesh cells maps onto message passing, and why this should generalise from simplified synthetic shapes to a patient mesh. Being a differentiable solver, it is also a step towards the inverse problem for oxygen-15 water (H₂O) PET (positron emission tomography) perfusion imaging.
2. [Multi-Objective Learning for Deformable Image Registration](https://proceedings.mlr.press/v250/grewal24a.html): registration is posed as multi-objective optimisation, and one network learns a whole trade-off front rather than one weighted sum. The claim that it covers the front better than a grid of scalarisation weights is the place to think about convexity of the front and why linear weighting misses parts of it.
3. [Structure Size as Confounder in Uncertainty Based Segmentation Quality Prediction](https://proceedings.mlr.press/v250/geissler24a.html): a short statistics argument; Dice and structure-wise uncertainty measures both depend on structure size, so their raw correlation overstates how well uncertainty predicts quality, and partial correlation given size corrects for it. Worth checking when partial correlation is the right adjustment, and whether size acts as a confounder or as a mediator here.
4. [SINR: Spline-enhanced implicit neural representation for multi-modal registration](https://proceedings.mlr.press/v250/sideri-lampretsa24a.html): instead of letting an implicit neural representation output displacements at sampled coordinates, it predicts free-form deformation control points, so the field inherits B-spline smoothness and folds less. A good case for comparing the regularity a spline basis builds in with the regularity a penalty term only encourages.
5. [Active Learning with the nnUNet and Sample Selection with Uncertainty-Aware Submodular Mutual Information Measure](https://proceedings.mlr.press/v250/follmer24a.html): batch selection maximises a submodular mutual information measure weighted by uncertainty, aiming for samples that are uncertain, diverse and representative at once. Worth working out whether the uncertainty weighting keeps the function submodular, which is what the (1 − 1/e) guarantee for greedy selection relies on.
6. [DDA: Dimensionality Driven Augmentation Search for Contrastive Learning in Laparoscopic Surgery](https://proceedings.mlr.press/v250/zhou24b.html): local intrinsic dimensionality of the learned representations is used as a differentiable proxy objective when searching for augmentation policies. The open question is why local dimensionality should track representation quality, and how the estimator is made differentiable.

Also worth a look: [Shape of my heart](https://proceedings.mlr.press/v250/verhulsdonk24a.html) (deep signed distance functions with Lipschitz regularity, which allow changes of topology that linear shape models cannot), [Probabilistic signed distance functions for retinal layers](https://proceedings.mlr.press/v250/islam24a.html) (Gaussian distributions over a level-set shape parameterisation), [Predicting AMD progression with intra-subject temporal consistency](https://proceedings.mlr.press/v250/chakravarty24a.html) (a neural ordinary differential equation whose distance to the decision hyperplane is tied inversely to time-to-conversion), [Dense Self-Supervised Learning](https://proceedings.mlr.press/v250/seince24a.html) (pixel-level equivariance rather than image-level invariance under augmentations), [Deformation-aware GAN](https://proceedings.mlr.press/v250/xin24a.html) (three levels of inverse consistency for symmetric registration inside synthesis) and [Spatio-Temporal Encoding with Surface Masked Autoencoders](https://proceedings.mlr.press/v250/dahan24b.html) (masked autoencoding of cortical signals on regular icosahedral grids).

## Sources

- *Proceedings of the 7th International Conference on Medical Imaging with Deep Learning*,
  PMLR volume 250: <https://proceedings.mlr.press/v250/>
