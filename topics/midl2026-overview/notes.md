---
title: "MIDL 2026: topic map, outliers and theory picks"
authors: "Topic notes"
venue: "MIDL 2026"
tags: [midl, survey, conference-overview, segmentation, foundation-models, uncertainty, reconstruction]
status: living
category: "Medical image analysis"
subcategory: "Conference overviews"
short_title: "MIDL 2026 — overview"
summary: "a high-level map of the ~230 MIDL 2026 papers (PMLR v315) in ten topic groups, the outlier papers that fall outside them, and a shortlist of the most mathematically foundational ones to read"
---

## Why this page exists

MIDL 2026 (Chientan, Taipei, 8–10 July 2026) was published as
[PMLR volume 315](https://proceedings.mlr.press/v315/), with about 230 papers. This page
is a quick map of the whole volume: which topics dominate, which papers do not fit the
pattern, and which are worth a close, theory-first read.

The grouping is from **titles only**, with abstracts read for the shortlist, so the counts are
approximate and some papers belong to more than one group. The PDFs have not been read yet.

## Topic groups at a glance

| Group | Share | What's in it |
|---|---|---|
| [Segmentation](#segmentation) | ~25% | The biggest group: semi-supervised and weakly supervised methods, noisy labels, topology-aware losses for vessels, Mamba / state-space models, adapting SAM and DINO, calibration and uncertainty |
| [Foundation models and adaptation](#foundation-models-and-adaptation) | ~12% | Adapting DINOv3, SAM 2/3 and CLIP; pretraining on CT, pathology and endoscopy; domain-specific distillation; PEFT; hypernetworks |
| [Vision-language models, LLMs and agents](#vision-language-models-llms-and-agents) | ~10% | Radiology report generation (mostly chest X-ray), VQA, GRPO-style RL fine-tuning, multi-agent reasoning, clinical decision support |
| [Computational pathology](#computational-pathology) | ~10% | Whole-slide images with MIL, stain and domain shift, slide graphs, prototypes, slide-level vision-language models |
| [Generative models](#generative-models) | ~10% | Diffusion, flow matching and consistency models for synthesis, augmentation, harmonisation, super-resolution and modality translation |
| [Reconstruction and physics](#reconstruction-and-physics) | ~8% | Accelerated MRI (k-space sampling, diffusion priors, learned iterative schemes), slice-to-volume reconstruction, T1/T2 mapping, physically grounded rendering |
| [3D scene representations](#d-scene-representations) | ~7% | INRs, neural fields, Gaussian and triangle splatting for registration, cardiac motion, endoscopy and novel ultrasound views |
| [Trustworthiness](#trustworthiness) | ~10% | Uncertainty, calibration, explainability (concept bottlenecks, prototypes, Shapley values), fairness and bias, failure-mode discovery, shortcut learning |
| [Registration and geometry](#registration-and-geometry) | ~5% | Deformable, surface-based and longitudinal registration, pose estimation, landmarks |
| [Benchmarks and validation](#benchmarks-and-validation) | ~8% | Cross-scanner reproducibility, multi-site validation, independent re-evaluations of challenges, new datasets |

**Modalities, roughly by frequency:** MRI, CT, chest X-ray, histopathology, endoscopy and
surgical video, then ultrasound, retina and others.

**Overall trend:** much of the volume is "take a foundation model and adapt it", with a strong
counter-current asking whether the models can be trusted: uncertainty, fairness and failure modes.

## Topic groups

Each group lists a few representative papers, not all of them.

### Segmentation

- [Topological-Aware Regularization for Semi-Supervised Intracranial Aneurysm Vessel Segmentation](https://proceedings.mlr.press/v315/xiao26a.html)
- [Generalizing Abstention for Noise-Robust Learning in Medical Image Segmentation](https://proceedings.mlr.press/v315/moustafa26a.html)
- [Beyond scalar losses: calibrating segmentation models via gradient vector field surgery](https://proceedings.mlr.press/v315/lux26a.html)
- [Hyperbolic U-Net for Robust Medical Image Segmentation](https://proceedings.mlr.press/v315/mishra26a.html)
- [SegMaST: Mamba-based Spatio-Temporal Modeling for Longitudinal Disease Detection and Segmentation](https://proceedings.mlr.press/v315/varma26a.html)
- [ECT-3DMedSAM: Efficient Cross Teaching Using SAM for Semi-Supervised 3D Segmentation](https://proceedings.mlr.press/v315/huang26a.html)
- [Breaking the Memory Barrier: Efficient Multi-Class 3D Segmentation for Hundreds of Classes](https://proceedings.mlr.press/v315/jaubert26a.html)
- [Calibration-Aware Semi-Supervised Fetal Head Segmentation with Boundary-Positive Contrast](https://proceedings.mlr.press/v315/khan26a.html)

### Foundation models and adaptation

- [Incentivizing DINOv3 Adaptation for Medical Vision Tasks via Feature Disentanglement](https://proceedings.mlr.press/v315/he26a.html)
- [Domain-Constrained Distillation of DINOv3 into a Lightweight Foundation Model Toward Point-of-Care Ultrasound](https://proceedings.mlr.press/v315/nahian26a.html)
- [TAP-CT: 3D Task-Agnostic Pretraining of CT Foundation Models](https://proceedings.mlr.press/v315/veenboer26a.html)
- [StainNet: Scaling Self-Supervised Foundation Models on IHC and Special Stains](https://proceedings.mlr.press/v315/li26c.html)
- [HyperCT: Low-Rank Hypernet for Unified Chest CT Analysis](https://proceedings.mlr.press/v315/liu26c.html)
- [Task-Conditioned 3D U-Nets via Hypernetworks for Data-Scarce Medical Segmentation](https://proceedings.mlr.press/v315/hagen26a.html)
- [Comparing SAM 2 and SAM 3 for Zero-Shot Segmentation of 3D Medical Data](https://proceedings.mlr.press/v315/chakrabarty26a.html)

### Vision-language models, LLMs and agents

- [RadVLM-GRPO: Chest X-ray Report Generation and Visual Grounding via Reinforcement Learning](https://proceedings.mlr.press/v315/gundersen26a.html)
- [Does Grounding Improve Radiology Report Generation? An Empirical Study on PadChest-GR](https://proceedings.mlr.press/v315/aas-alas26a.html)
- [RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation](https://proceedings.mlr.press/v315/zhang26b.html)
- [CardAIc-Agents: A Multimodal Framework for Cardiac Care Support](https://proceedings.mlr.press/v315/zhang26a.html)
- [RadImageNet-VQA: A Large-Scale CT and MRI Dataset for Radiologic VQA](https://proceedings.mlr.press/v315/butsanets26a.html)
- [Can Vision Language Models Track a Heartbeat? Frame-Level Echocardiogram Understanding](https://proceedings.mlr.press/v315/liu26e.html)

### Computational pathology

- [Context-Aware Patch Representations for Multiple Instance Learning](https://proceedings.mlr.press/v315/lolos26a.html)
- [MoA: Mixture of Aggregators Improves Slide-Level Diagnosis](https://proceedings.mlr.press/v315/ozlugedik26a.html)
- [Counterfactual Intervention in Attention MIL for Digital Pathology](https://proceedings.mlr.press/v315/chraki26a.html)
- [Quantifying and Mitigating Hospital Domain Bias in Pathology Foundation Models](https://proceedings.mlr.press/v315/zhang26d.html)
- [Democratising Pathology Co-Pilots: An Open Pipeline and Dataset for Whole-Slide VLMs](https://proceedings.mlr.press/v315/moonemans26a.html)
- [More is more: multi-rater information for WSI grading via a virtual expert panel](https://proceedings.mlr.press/v315/grove26a.html)

### Generative models

- [Simplex-Aligned Diffusion with Cross-Granularity Interaction for Robust Classification](https://proceedings.mlr.press/v315/wu26a.html)
- [Flow Matching for 3D Craniofacial Skeletal Data Generation](https://proceedings.mlr.press/v315/melacini26a.html)
- [WFM: 3D Wavelet Flow Matching for Ultrafast Multi-Modal MRI Synthesis](https://proceedings.mlr.press/v315/tur26a.html)
- [MetaVoxel: Joint Diffusion Modeling of Imaging and Clinical Metadata](https://proceedings.mlr.press/v315/liu26a.html)
- [LabelG: Consistent Pairwise 3D CT Image and Mask Generation](https://proceedings.mlr.press/v315/wang26c.html)
- [A Pragmatic Note on Evaluating Generative Models with FID for Retinal Image Synthesis](https://proceedings.mlr.press/v315/wu26c.html)

### Reconstruction and physics

- [Conditional Learned Reconstruction for Medical Imaging](https://proceedings.mlr.press/v315/moriakov26a.html)
- [Beyond Diffusion: Consistency Models for One-Step MRI Reconstruction](https://proceedings.mlr.press/v315/akoda26a.html)
- [Sparse Subspace Diffusion Model for Physically Consistent Accelerated MRI Reconstruction](https://proceedings.mlr.press/v315/deng26a.html)
- [End-to-End Co-Optimization of Adaptive k-space Sampling and Reconstruction for Dynamic MRI](https://proceedings.mlr.press/v315/yiasemis26a.html)
- [T1-PILOT: Physics-Informed Learned Optimized Trajectories for T1 Mapping](https://proceedings.mlr.press/v315/shor26a.html)
- [Fast and Explicit: Slice-to-Volume Reconstruction via 3D Gaussians with Analytic PSF](https://proceedings.mlr.press/v315/dannecker26a.html)

### 3D scene representations

- [MedFuncta: A Unified Framework for Learning Efficient Medical Neural Fields](https://proceedings.mlr.press/v315/friedrich26a.html)
- [Geometry-Aware Cardiac MRI Representation Learning with Equivariant Neural Fields](https://proceedings.mlr.press/v315/wiers26a.html)
- [NISF++: Implicit representations of 3D+time cardiac function from 2D MR views](https://proceedings.mlr.press/v315/stolt-anso26a.html)
- [UltraG-Ray: Physics-Based Gaussian Ray Casting for Novel Ultrasound View Synthesis](https://proceedings.mlr.press/v315/duelmer26a.html)
- [Rendering with a Gut Feeling: Depth-Guided Triangle Splatting for Colonoscopic Reconstruction](https://proceedings.mlr.press/v315/hardy26a.html)
- [Endo-4DTS: Monocular 4D Scene Synthesis for Endoscopy](https://proceedings.mlr.press/v315/salort-benejam26a.html)

### Trustworthiness

- [Can You Trust Your Model? Uncertainty Approximations Guaranteeing Validity of Glioma Segmentation Explanations](https://proceedings.mlr.press/v315/ren26a.html)
- [CARE: Confidence-aware Ratio Estimation for Medical Biomarkers](https://proceedings.mlr.press/v315/li26b.html)
- [Testing the Trust: Verification and Validation of Bayesian Segmentation under Uncertainty](https://proceedings.mlr.press/v315/carannante26a.html)
- [Weight Space Correlation Analysis: Quantifying Feature Utilization](https://proceedings.mlr.press/v315/wong26a.html)
- [Revealing Hidden Failure Modes in Chest X-ray Classification via Spectral Domain Analysis](https://proceedings.mlr.press/v315/halimi26a.html)
- [SEG4SEG: Identifying Systematic Failure Modes in Segmentation by Subgroup Discovery](https://proceedings.mlr.press/v315/weng26a.html)
- [OxEnsemble: Fair Ensembles for Low-Data Classification](https://proceedings.mlr.press/v315/rystrom26a.html)
- [SoftCAM: Making black box models self-explainable](https://proceedings.mlr.press/v315/djoumessi26a.html)

### Registration and geometry

- [GeoReg: Biplanar DSA-to-CTA registration with geodesic consistency](https://proceedings.mlr.press/v315/herten26a.html)
- [Robust Multi-Scale INRs for Large-Deformation Lung Registration](https://proceedings.mlr.press/v315/gebauer26a.html)
- [Anatomical Longitudinal Cortical Surface Registration](https://proceedings.mlr.press/v315/saboo26a.html)
- [CSVR: Combined Surface and Volume Registration for Neonatal Brain MRI](https://proceedings.mlr.press/v315/masui26a.html)
- [FutureMorph: Toward Predicting Future Deformation Fields in Longitudinal Imaging](https://proceedings.mlr.press/v315/khawaled26a.html)
- [nnLandmark: A Self-Configuring Method for 3D Medical Landmark Detection](https://proceedings.mlr.press/v315/ertl26a.html)

### Benchmarks and validation

- [Benchmarking the Reproducibility of Brain Tissue Segmentation Across MRI Scanners](https://proceedings.mlr.press/v315/kondrateva26a.html)
- [Harmonizing MR Images Across 100+ Scanners: Multi-site Validation with Traveling Subjects](https://proceedings.mlr.press/v315/hays26a.html)
- [CrossPan: A Benchmark for Cross-Sequence Pancreas MRI Segmentation](https://proceedings.mlr.press/v315/peng26a.html)
- [Evaluating the Impact of Medical Image Reconstruction on Downstream AI Fairness and Performance](https://proceedings.mlr.press/v315/wohlrapp26a.html)
- [A benchmark of GNNs, graph kernels and classical ML on rs-fMRI brain graphs](https://proceedings.mlr.press/v315/mhanna26a.html)
- [Evaluation of 3D Ultrasound Reconstruction and 2D/3D Segmentation for Neonatal Hip Dysplasia Screening](https://proceedings.mlr.press/v315/heyer26a.html)

## Outlier papers

Papers that sit outside the main groups in topic, method or framing.

### Unusual subjects or data

- [Quantitative Pose-Based Analysis of Movement Disorders in Pediatric NGLY1 and SLC13A5 Patients](https://proceedings.mlr.press/v315/dai26a.html): video pose estimation for ultra-rare diseases.
- [Multicenter Morphometric Analysis of Stratum Corneum Nanotexture for Skin Barrier Assessment](https://proceedings.mlr.press/v315/wang26h.html): nanoscale skin texture.
- [Learning Structure-Aware Foundational Representation of Rat Testicular Tubules](https://proceedings.mlr.press/v315/kshirsagar26a.html): animal histology.
- [OmniNet: Remote Respiratory Rate Measurement from Facial Video](https://proceedings.mlr.press/v315/lin26a.html): no medical imaging involved.
- [GenVOG-DiT: Pose-Driven Nystagmus VOG Video Generation](https://proceedings.mlr.press/v315/rahman26a.html): generates eye-tracking video.
- [Deep Learning Based Emboli Detection Using Ultrasound Doppler Imaging](https://proceedings.mlr.press/v315/mushunuri26a.html)
- [Orientation Normalization of Multi-Stain Skin Tissue Cross-Sections](https://proceedings.mlr.press/v315/topolnjak26a.html)
- [SCR²-ST: Single Cell with Spatial Transcriptomics for Active Sampling via RL](https://proceedings.mlr.press/v315/zhu26a.html)
- [BETA: Resting-state fMRI Biotypes for tDCS Efficacy in Anxiety Among Older Adults](https://proceedings.mlr.press/v315/stolte26a.html)

### Social or ethical findings

- [Algorithms Trained on Normal Chest X-rays Can Predict Health Insurance Types](https://proceedings.mlr.press/v315/chen26c.html): socioeconomic leakage in images.
- [No Evidence of Disease: Clinically-Risky Adversarial Chest CT Report Generation](https://proceedings.mlr.press/v315/irshad26a.html)
- [On the Feasibility of Fréchet Radiomic Distance–Constrained Adversarial Examples](https://proceedings.mlr.press/v315/mahmoud26a.html)

### Unusual mathematics or methods

- [Is Exchangeability better than I.I.D. … for Data-scarce Medical Image Segmentation?](https://proceedings.mlr.press/v315/roy26a.html): a probability-theory framing.
- [Hyperbolic U-Net for Robust Medical Image Segmentation](https://proceedings.mlr.press/v315/mishra26a.html): a fully non-Euclidean network.
- [Quantum-Inspired Orthonormal CNN for Energy-Efficient Medical Image Denoising](https://proceedings.mlr.press/v315/dutta26a.html)
- [MedKamba: State-Space Models and Fractional Kolmogorov–Arnold Networks](https://proceedings.mlr.press/v315/shakya26a.html)
- [MultiPersistence Topological Fusion with Vision Transformers for Skin Cancer Detection](https://proceedings.mlr.press/v315/tastan26a.html): multiparameter TDA.
- [RNCA: Self-Repairing Segmentation Masks](https://proceedings.mlr.press/v315/silbernagel26a.html): neural cellular automata.
- [Beyond scalar losses: gradient vector field surgery](https://proceedings.mlr.press/v315/lux26a.html)

### Papers that critique the field

- [A Pragmatic Note on Evaluating Generative Models with FID for Retinal Image Synthesis](https://proceedings.mlr.press/v315/wu26c.html): FID does not track downstream usefulness.
- [The LUMirage: An independent evaluation of zero-shot performance in the LUMIR challenge](https://proceedings.mlr.press/v315/jena26a.html)
- [ReX-MLE: The Autonomous Agent Benchmark for Medical Imaging Challenges](https://proceedings.mlr.press/v315/kenia26a.html)
- [Test-Time Scaling in Clinical Decision Making](https://proceedings.mlr.press/v315/byun26a.html): an LLM-era idea brought to the clinic.

### Unusual clinical framing

- [A Deep Learning-Enabled Digital Twin for Fast Online Adaptive Proton Therapy](https://proceedings.mlr.press/v315/chang26a.html)
- [From Cross-Sectional CT to Dynamic Insights: Pseudotime-Based Modeling of Lung Nodule Progression](https://proceedings.mlr.press/v315/zhuang26a.html): borrows pseudotime from single-cell biology.
- [Brain Tumor Growth Inversion via Differentiable Neural Surrogates](https://proceedings.mlr.press/v315/weidner26a.html): solving a PDE inverse problem through a learned surrogate.

## Theory picks

The most mathematically foundational papers, ranked for a close reading (from abstracts only).

1. [Beyond scalar losses: gradient vector field surgery](https://proceedings.mlr.press/v315/lux26a.html): explains Dice-loss overconfidence through the loss gradient, then scales the partial derivative linearly with the prediction error. Short enough to derive on a whiteboard.
2. [Simplex-Aligned Diffusion](https://proceedings.mlr.press/v315/wu26a.html): unbounded Gaussian noise on a bounded probability simplex is a geometric mismatch; diffusion moves to the logit space instead.
3. [Is Exchangeability better than I.I.D.?](https://proceedings.mlr.press/v315/roy26a.html): exchangeability (de Finetti) in place of i.i.d. for multi-site pooling, and the "Data Addition Dilemma".
4. [Fast and Explicit: SVR via 3D Gaussians with Analytic PSF](https://proceedings.mlr.press/v315/dannecker26a.html): Gaussians stay Gaussian under convolution, giving a closed-form PSF in place of Monte Carlo sampling.
5. [Can You Trust Your Model?](https://proceedings.mlr.press/v315/ren26a.html): uniform convergence gives a confidence set of plausibly optimal models and valid intervals for Shapley values.
6. [CARE: Confidence-aware Ratio Estimation](https://proceedings.mlr.press/v315/li26b.html): error propagation from segmentation to a ratio biomarker, split into bias and variance versus calibration error.

Also worth a look: [Equivariant Neural Fields for cardiac MRI](https://proceedings.mlr.press/v315/wiers26a.html),
[GeoReg](https://proceedings.mlr.press/v315/herten26a.html) (consistency on SO(3)),
[Sparse Subspace Diffusion](https://proceedings.mlr.press/v315/deng26a.html), and, for ultrasound,
[Weight Space Correlation Analysis](https://proceedings.mlr.press/v315/wong26a.html) and
[Automated Quality Assessment of Blind Sweep Obstetric Ultrasound](https://proceedings.mlr.press/v315/bhandari26a.html).

## Sources

- *Proceedings of the 9th International Conference on Medical Imaging with Deep Learning*,
  PMLR volume 315: <https://proceedings.mlr.press/v315/>
