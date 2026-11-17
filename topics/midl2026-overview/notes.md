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

**The ideas behind this group** (a synthesis from the abstracts of about 55 papers in this group; the individual papers have not been read in full).

*The shared problem is that a good Dice score on one test set is not the same as a usable mask.* Voxel-level labels are expensive, thin and branching structures break easily, models fail when the scanner or sequence changes, and a mask with no indication of confidence is hard to act on clinically. The papers attack these gaps from four directions, and a set of benchmark papers checks how big the gaps really are.

*The first thread tries to get more from fewer or weaker labels.* Semi-supervised methods pair networks or teacher–student models and decide which pseudo-labels to trust: cross-supervision with uncertainty-guided pseudo-labels for endoscopic video ([Endo-SemiS](https://proceedings.mlr.press/v315/li26e.html)), a temperature-calibrated gate on pseudo-labels for fetal head ultrasound ([UltraSemiNet](https://proceedings.mlr.press/v315/khan26a.html)), and frozen MedSAM-2 encoders shared by two Low-Rank Adaptation (LoRA) decoders ([ECT-3DMedSAM](https://proceedings.mlr.press/v315/huang26a.html)). Others add a structural regulariser: vessel topology for aneurysms ([TAR](https://proceedings.mlr.press/v315/xiao26a.html)), or an axial continuity loss that bridges sparsely annotated slices in light-sheet microscopy ([3D coherence](https://proceedings.mlr.press/v315/hashisho26a.html)). Weakly supervised work turns image-level labels into masks, using higher-resolution class activation maps ([elaborated CAMs](https://proceedings.mlr.press/v315/chen26a.html)), diffusion-autoencoder "healthy edits" scored region by region ([Hide-and-Seek Attribution](https://proceedings.mlr.press/v315/atad26a.html)), and text-initialised prototypes from a pathology vision–language model ([ConStruct](https://proceedings.mlr.press/v315/le26b.html)). Label noise is handled by letting the model abstain on doubtful pixels ([generalised abstention](https://proceedings.mlr.press/v315/moustafa26a.html)). Supervision is also generated: paired CT images and masks from a diffusion foundation model ([LabelG](https://proceedings.mlr.press/v315/wang26c.html)), teachers trained on immunohistochemical restaining and distilled into one student ([PAGET](https://proceedings.mlr.press/v315/komura26a.html)), and U-Net weights produced by a hypernetwork from a single annotated volume ([task-conditioned U-Nets](https://proceedings.mlr.press/v315/hagen26a.html)). Labels can also be redefined, by an anatomical protocol checked by a language-guided evaluator ([NeuroLangSeg](https://proceedings.mlr.press/v315/liu26b.html)) or by guideline-filtered pathology reports for radiotherapy target volumes ([ReaCT](https://proceedings.mlr.press/v315/wu26b.html)).

*The second thread treats shape and topology as part of the target.* For vessels the argument is that "segment, then fix" is the wrong order: one paper predicts voxel-wise direction vectors and extracts a vessel graph from them ([Vesselpose](https://proceedings.mlr.press/v315/palaniappan26a.html)), another repairs broken masks with a neural cellular automaton ([RNCA](https://proceedings.mlr.press/v315/silbernagel26a.html)), others label 19 pulmonary vessel classes on a graph ([PaSAL](https://proceedings.mlr.press/v315/eppink26a.html)) or score branching organoids with centreline Dice as well as overlap ([Inception-UNet](https://proceedings.mlr.press/v315/andrusca26a.html)). Priors also enter through prompts and losses: tubular-morphology prompts for SAM on intracranial arteries ([A-ADAPT](https://proceedings.mlr.press/v315/tan26a.html)), prompts generated through a 3D volume from one click ([automatic prompts](https://proceedings.mlr.press/v315/zhang26c.html)), an ellipse loss for fetal heart chambers ([Ultra-ECP](https://proceedings.mlr.press/v315/le26a.html)), nesting constraints between glioma subregions ([MCMA-Net++](https://proceedings.mlr.press/v315/alameddin26a.html)) and boundary-guided dynamic convolution ([SGDC](https://proceedings.mlr.press/v315/shi26a.html)). Consistency across views and time is a related prior: one representation shared by all cardiac planes ([NISF++](https://proceedings.mlr.press/v315/stolt-anso26a.html)), baseline and follow-up scans segmented jointly ([SegMaST](https://proceedings.mlr.press/v315/varma26a.html)), and longer-range memory for surgical video ([temporal memory](https://proceedings.mlr.press/v315/gao26a.html)).

*The third thread changes the network or the training distribution.* Architectural proposals include state-space blocks ([MedKamba](https://proceedings.mlr.press/v315/shakya26a.html)), test-time training layers ([TTT-UNet](https://proceedings.mlr.press/v315/zhou26a.html)), a Swin–DINOv3 dual encoder with mixture-of-experts ([D²-Former](https://proceedings.mlr.press/v315/mia26a.html)), a U-Net running entirely in hyperbolic space ([Hyperbolic U-Net](https://proceedings.mlr.press/v315/mishra26a.html)), fixed centre–surround filters ([IB-nnU-Net](https://proceedings.mlr.press/v315/bhandary26a.html)), separate module branches ([expert branches](https://proceedings.mlr.press/v315/guo26a.html)), and a low-rank class projection that decouples memory from the number of classes ([hundreds of classes](https://proceedings.mlr.press/v315/jaubert26a.html)). For domain shift there are style and content augmentation ([StyCona](https://proceedings.mlr.press/v315/shen26a.html)), inter-organ contrastive adaptation ([AICL](https://proceedings.mlr.press/v315/sun26a.html)), test-time adaptation from intermediate optical coherence tomography reconstructions ([IRTTA](https://proceedings.mlr.press/v315/pinetz26a.html)), continual learning over varying MRI modality sets ([CLMU-Net](https://proceedings.mlr.press/v315/sadegheih26a.html)), and an exchangeability view of pooling datasets ([Roy et al.](../2026-roy-exchangeability-pooling/index.html)). Foundation-model segmenters are compared directly in [SAM 2 vs SAM 3](https://proceedings.mlr.press/v315/chakrabarty26a.html), [cell foundation models](https://proceedings.mlr.press/v315/archit26a.html), [ICL-NoiseUNet](https://proceedings.mlr.press/v315/charisiadis26a.html) and [Dual-DINOv3](https://proceedings.mlr.press/v315/lin26b.html).

*The fourth thread asks whether the mask comes with a usable confidence, where the group overlaps with Trustworthiness.* Region losses such as Dice make models over-confident, and [gradient surgery](../2026-lux-gradient-surgery/index.html) rescales the loss gradient by the prediction error to fix this. Other papers produce uncertainty in a single pass with evidential heads ([Evidential DualU-Net](https://proceedings.mlr.press/v315/anglada-rotger26a.html)), heteroscedastic boundary heatmaps ([pectoral muscle](https://proceedings.mlr.press/v315/zech26a.html)) or graph-decoder latents ([CheXmask-U](https://proceedings.mlr.press/v315/cosarinsky26a.html)), or tie uncertainty to topological complexity ([TUNE++](https://proceedings.mlr.press/v315/dhor26a.html)). Further papers test whether uncertainty helps downstream: under corruptions ([Bayesian V&V](https://proceedings.mlr.press/v315/carannante26a.html)), in clinicians' contour editing ([uncertainty maps](https://proceedings.mlr.press/v315/maruccio26a.html)), propagated into confidence intervals for ratio biomarkers ([CARE](https://proceedings.mlr.press/v315/li26b.html)), and in searching for systematic failure subgroups ([SEG4SEG](https://proceedings.mlr.press/v315/weng26a.html)).

*The benchmark papers supply the reality check.* Models collapse when moved across pancreas MRI sequences ([CrossPan](https://proceedings.mlr.press/v315/peng26a.html)), brain volumes vary between scanners even in test–retest data ([reproducibility](https://proceedings.mlr.press/v315/kondrateva26a.html)), and curated multi-site comparisons cover breast ultrasound ([sonomammograms](https://proceedings.mlr.press/v315/gunawardhana26a.html)), haemorrhage ([IPH](https://proceedings.mlr.press/v315/duarte26a.html)), hip ultrasound ([hip dysplasia](https://proceedings.mlr.press/v315/heyer26a.html)), axon histology ([ASTIH](https://proceedings.mlr.press/v315/collin26a.html)) and malaria ([YOLOv12](https://proceedings.mlr.press/v315/issah26a.html)).

*Taken together*, segmentation research is moving from "a better encoder" to "a better contract": cheaper supervision, anatomically valid outputs and an honest statement of confidence. Still open: topology metrics are not yet standard alongside Dice; calibration is usually measured per pixel rather than on the downstream quantity; physics-driven shifts such as a change of MRI sequence still defeat most domain-generalisation methods; and few papers test whether their uncertainty changes a clinician's decision.

**Representative papers**

- [Topological-Aware Regularization for Semi-Supervised Intracranial Aneurysm Vessel Segmentation](https://proceedings.mlr.press/v315/xiao26a.html)
- [Generalizing Abstention for Noise-Robust Learning in Medical Image Segmentation](https://proceedings.mlr.press/v315/moustafa26a.html)
- [Beyond scalar losses: calibrating segmentation models via gradient vector field surgery](https://proceedings.mlr.press/v315/lux26a.html)
- [Hyperbolic U-Net for Robust Medical Image Segmentation](https://proceedings.mlr.press/v315/mishra26a.html)
- [SegMaST: Mamba-based Spatio-Temporal Modeling for Longitudinal Disease Detection and Segmentation](https://proceedings.mlr.press/v315/varma26a.html)
- [ECT-3DMedSAM: Efficient Cross Teaching Using SAM for Semi-Supervised 3D Segmentation](https://proceedings.mlr.press/v315/huang26a.html)
- [Breaking the Memory Barrier: Efficient Multi-Class 3D Segmentation for Hundreds of Classes](https://proceedings.mlr.press/v315/jaubert26a.html)
- [Calibration-Aware Semi-Supervised Fetal Head Segmentation with Boundary-Positive Contrast](https://proceedings.mlr.press/v315/khan26a.html)

### Foundation models and adaptation

**The ideas behind this group** (a synthesis from the abstracts of about 25 papers in this group and
the ones next to it; the individual papers have not been read in full).

*The starting point is a domain gap.* General-purpose foundation models (DINOv2/v3 for image
features, the SAM family for promptable segmentation, CLIP-style vision–language models) learn
excellent representations from natural images and web text, but medical data differs in almost every
way they assume: it is often 3D rather than 2D, physical rather than photographic (CT intensity values,
PET uptake, ultrasound speckle, histology stains), and the signal often lies in subtle texture rather
than object shape. The papers respond with two strategies, often combined: re-train the model on the
medical domain, or keep it and adapt it cheaply.

*The first strategy treats the training recipe, not the weights, as the reusable asset.* Several papers
rerun a proven self-supervised method on large in-domain data: DINOv2 adapted to 3D CT with
depth-aware patches and positional encodings
([TAP-CT](https://proceedings.mlr.press/v315/veenboer26a.html)); a PET-only DINOv3 encoder combined
with a separately pretrained CT encoder, so paired scans are not needed for pretraining
([Dual-DINOv3](https://proceedings.mlr.press/v315/lin26b.html)); pathology models for
immunohistochemistry and special stains ([StainNet](https://proceedings.mlr.press/v315/li26c.html));
DINO variants pretrained on 4.7 million surgical video frames
([DINO for surgery](https://proceedings.mlr.press/v315/jong26a.html)); 3D masked-autoencoder priors for
low-dose lung CT ([ILD screening](https://proceedings.mlr.press/v315/mcconnell26a.html)) and
self-supervised pretraining for prostate MRI
([UMamba-ProSSL](https://proceedings.mlr.press/v315/abbas26a.html)); and federated pretraining across
hospitals where privacy rules stop data from being pooled
([Federated EndoViT](https://proceedings.mlr.press/v315/kirchner26a.html)). The consistent finding is
that in-domain pretraining helps reliably, and the general recipe transfers better than the general
weights. The medical-specific work goes into handling 3D input, augmentations that respect each
modality's physics, and data governance.

*The second strategy keeps a large model frozen, or nearly so, and adapts it cheaply.* The mechanisms
vary: parameter-efficient fine-tuning such as LoRA, sometimes with a step that separates task-relevant
features from task-irrelevant noise ([DINOv3-FD](https://proceedings.mlr.press/v315/he26a.html));
hypernetworks that generate task-specific weights, either LoRA updates for each chest-CT task
([HyperCT](https://proceedings.mlr.press/v315/liu26c.html)) or small 3D U-Nets from a single annotated
volume ([task-conditioned U-Nets](https://proceedings.mlr.press/v315/hagen26a.html)); in-context and
prototype-based learning, where a frozen encoder labels new cases by comparison with a handful of
examples ([PIKACHU](https://proceedings.mlr.press/v315/kumar26b.html),
[RandP](https://proceedings.mlr.press/v315/mao26a.html),
[ICL-NoiseUNet](https://proceedings.mlr.press/v315/charisiadis26a.html)); distillation of DINOv3 into a
ResNet-50 small enough for point-of-care ultrasound, where the student sometimes beats its teacher on
varied ultrasound data ([POCUS distillation](https://proceedings.mlr.press/v315/nahian26a.html)); and
using the model with no training at all, for example combining 2D features from three viewing axes into
3D tokens for zero-shot anomaly detection
([training-free 3D anomaly detection](https://proceedings.mlr.press/v315/le-gia26a.html)). Behind all of
these is one question: how little needs to change to make a general representation work for a specific
clinical task? One reality check: adapting a 3D vision–language model to a new hospital's liver-cancer
data, LoRA-style fine-tuning came close to full fine-tuning
([HCC adaptation](https://proceedings.mlr.press/v315/kartika26a.html)).

*The 2D-to-3D mismatch is its own sub-theme.* Most strong foundation models are 2D, while much clinical
imaging is volumetric. One group lifts 2D models to 3D: generating prompts slice by slice from a single
click for vessels ([automatic prompts](https://proceedings.mlr.press/v315/zhang26c.html)) or cells
([cell foundation models](https://proceedings.mlr.press/v315/archit26a.html)), and using SAM 2's
video-style tracking through a volume; a controlled comparison asks whether SAM 3 can simply replace
SAM 2 ([SAM 2 vs SAM 3](https://proceedings.mlr.press/v315/chakrabarty26a.html)). The other group
pretrains natively in 3D. Within the SAM line the recurring practical problem is sensitivity to prompts,
answered by generating prompts automatically, simulating imprecise clicks during training, and adding
anatomical shape priors such as an ellipse for fetal heart chambers
([Ultra-ECP](https://proceedings.mlr.press/v315/le26a.html)); frozen SAM encoders also anchor
semi-supervised cross-teaching ([ECT-3DMedSAM](https://proceedings.mlr.press/v315/huang26a.html)).

*The third thread asks what these models have actually learned.* Pathology foundation models turn out to
encode which hospital a slide came from, so scanner and lab can act as a shortcut; one paper measures
this with a robustness index and removes it with an adversarially trained adapter
([hospital bias](https://proceedings.mlr.press/v315/zhang26d.html)), and
[AdvDINO](https://proceedings.mlr.press/v315/su26a.html) builds the same idea into self-supervised
pretraining to suppress batch effects. Another uses radiomic features as an interpretable view of what
fine-tuning changes inside a prostate model
([radiomic lens](https://proceedings.mlr.press/v315/wang26e.html)). This is the same goal as the
exchangeability discussion in the [Roy et al. notes](../2026-roy-exchangeability-pooling/index.html):
make the representation carry no information about the site, given the label,
$Z\perp\text{site}\mid Y$.

*Taken together*, the field is converging on a three-step pattern: pretrain in-domain with a general
recipe; adapt lightly with LoRA, hypernetworks, prompts or prototypes; and audit the representation for
shortcuts such as site, scanner or stain. Still open: evaluation is fragmented, so it is hard to say
which pretrained model suits a new task, which is the transferability-estimation question, and none of
these papers uses a transferability score to choose a model; where self-configuring baselines such as
nnU-Net are included, foundation models do not always win for segmentation; and ultrasound, functional
imaging and non-H&E pathology remain the weakest areas, which is why several papers build their own
in-domain models.

**Representative papers**

- [Incentivizing DINOv3 Adaptation for Medical Vision Tasks via Feature Disentanglement](https://proceedings.mlr.press/v315/he26a.html)
- [Domain-Constrained Distillation of DINOv3 into a Lightweight Foundation Model Toward Point-of-Care Ultrasound](https://proceedings.mlr.press/v315/nahian26a.html)
- [TAP-CT: 3D Task-Agnostic Pretraining of CT Foundation Models](https://proceedings.mlr.press/v315/veenboer26a.html)
- [StainNet: Scaling Self-Supervised Foundation Models on IHC and Special Stains](https://proceedings.mlr.press/v315/li26c.html)
- [HyperCT: Low-Rank Hypernet for Unified Chest CT Analysis](https://proceedings.mlr.press/v315/liu26c.html)
- [Task-Conditioned 3D U-Nets via Hypernetworks for Data-Scarce Medical Segmentation](https://proceedings.mlr.press/v315/hagen26a.html)
- [Comparing SAM 2 and SAM 3 for Zero-Shot Segmentation of 3D Medical Data](https://proceedings.mlr.press/v315/chakrabarty26a.html)

### Vision-language models, LLMs and agents

**The ideas behind this group** (a synthesis from the abstracts of about 27 papers in this group; the individual papers have not been read in full).

*The shared problem is that language makes a model easy to talk to, not necessarily faithful to the image.* Vision–language models (VLMs) and large language models (LLMs) promise reports, answers to clinical questions and explanations in plain words. In medicine they meet three obstacles: paired image–text data is scarce outside chest X-ray, generated text drifts towards language priors and invents findings, and standard text metrics say little about clinical correctness. Most papers in this group try to tie the text more tightly to visual evidence or to clinical structure.

*The first thread constrains radiology report generation so that the words follow the image.* One approach contrasts each normal X-ray with a masked version, category by category, to counter spurious co-occurrences of pathologies that build up during decoding ([CWCD](https://proceedings.mlr.press/v315/srivastava26a.html)). Another feeds bounding boxes of anatomy and findings into the generator, and finds that even partial or noisy grounding helps ([grounded RRG](https://proceedings.mlr.press/v315/aas-alas26a.html)). A third turns free text into anatomy-level structure and oversamples abnormal regions, raising sensitivity to pathology without changing the model ([ALO](https://proceedings.mlr.press/v315/buess26a.html)). For brain tumours, [BTReport](https://proceedings.mlr.press/v315/heras-rivera26a.html) goes furthest: imaging features are extracted deterministically and the LLM only turns them into prose. Reinforcement-learning fine-tuning with Group Relative Policy Optimization and clinically grounded rewards adds gains on both report generation and visual grounding over strong supervised fine-tuning, while explicit "thinking" adds nothing further ([RadVLM-GRPO](https://proceedings.mlr.press/v315/gundersen26a.html)). A threat-model paper shows the other side: small perturbations to CT volumes and prompts can make a report omit or invent high-risk findings ([CRA-RG](https://proceedings.mlr.press/v315/irshad26a.html)).

*The second thread aligns images and text at a finer scale than whole image against whole report.* [MApLe](https://proceedings.mlr.press/v315/bader26a.html) links individual report sentences to image patches, separating anatomical region from finding; [HiPro-CT](https://proceedings.mlr.press/v315/lu26b.html) embeds 3D CT regions and text as Gaussian distributions, so uncertainty is explicit and alignment happens at organ level; a text-to-image latent diffusion model is fine-tuned so that report phrases attend to the right places ([phrase grounding](https://proceedings.mlr.press/v315/vilouras26a.html)); [WristMIR](https://proceedings.mlr.press/v315/sonmezer26a.html) mines region-level captions from reports to retrieve similar wrist fractures; and [REVEAL](https://proceedings.mlr.press/v315/leem26a.html) turns questionnaire risk factors into narratives so that a VLM can align them with fundus images. Language also becomes a specification: [NeuroLangSeg](https://proceedings.mlr.press/v315/liu26b.html) uses protocol-defined anatomical prompts to enforce a consistent definition of subcortical structures, and [ReaCT](https://proceedings.mlr.press/v315/wu26b.html) distils pathology reports into guideline-relevant attributes that steer radiotherapy target segmentation.

*The third thread answers data scarcity by building data and adapting cheaply.* Some papers synthesise image–text pairs: 100,000 synthetic OCT angiography image–reasoning pairs ([synthetic vasculature reasoning](https://proceedings.mlr.press/v315/li26d.html)), timestamped question–answer annotations for cataract surgery video ([CatVLM](https://proceedings.mlr.press/v315/paranjape26a.html)), and instruction data for whole slides ([ANTONI-α](https://proceedings.mlr.press/v315/moonemans26a.html)). Others release open data and models: [Aloe-Vision](https://proceedings.mlr.press/v315/guasch-marti26a.html) publishes data mixtures, weights and a contamination-resistant exam benchmark, and [RadImageNet-VQA](https://proceedings.mlr.press/v315/butsanets26a.html) is a CT and MRI visual question answering set designed so that text-only models fall to near chance. On the adaptation side, updating under 2% of a multimodal LLM's parameters ([LLaMA32-Med](https://proceedings.mlr.press/v315/dong26a.html)), or parameter-efficient fine-tuning of a 3D VLM at a new hospital ([HCC adaptation](https://proceedings.mlr.press/v315/kartika26a.html)), comes close to heavier alternatives.

*The fourth thread keeps perception and reasoning apart, through tools and agents.* In [Surg-SAGE](https://proceedings.mlr.press/v315/hamdy26a.html), specialist surgical models describe the scene in text and a frozen LLM reasons over it, with no multimodal training. [INFORM-CT](https://proceedings.mlr.press/v315/tankel26a.html) has an LLM write code that calls detectors to manage incidental findings according to guidelines. The [Tool Bottleneck Model](https://proceedings.mlr.press/v315/liu26d.html) argues that spatial tool outputs should be fused by a learned network rather than through text. Multi-agent systems encode a clinical workflow: [RadAgents](https://proceedings.mlr.press/v315/zhang26b.html) follows a radiologist's steps and checks for conflicts between tools, and [CardAIc-Agents](https://proceedings.mlr.press/v315/zhang26a.html) uses retrieval-based planning, stepwise plan updates and a simulated multidisciplinary discussion. Without any training, [test-time scaling](https://proceedings.mlr.press/v315/byun26a.html) samples many answers, aggregates them by self-consistency and derives scaling laws for when this helps.

*Taken together*, the common move is to put structure between the image and the generated text: anatomy-level categories, grounding boxes, extracted features, rewards or tool calls. Free-form generation is rarely trusted on its own. The reality checks are sobering: general VLMs cannot reliably localise end-diastole and end-systole frames in echocardiograms, sometimes doing no better than random ([echo VLM benchmark](https://proceedings.mlr.press/v315/liu26e.html)), and autonomous coding agents mostly rank at the 0th percentile against the top human entries on medical imaging challenges ([ReX-MLE](https://proceedings.mlr.press/v315/kenia26a.html)). Still open: evaluation that measures clinical correctness rather than text overlap, temporal and 3D reasoning, robustness to adversarial or misleading inputs, and whether multi-agent systems beat a well-grounded single model once compute is matched.

**Representative papers**

- [RadVLM-GRPO: Chest X-ray Report Generation and Visual Grounding via Reinforcement Learning](https://proceedings.mlr.press/v315/gundersen26a.html)
- [Does Grounding Improve Radiology Report Generation? An Empirical Study on PadChest-GR](https://proceedings.mlr.press/v315/aas-alas26a.html)
- [RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation](https://proceedings.mlr.press/v315/zhang26b.html)
- [CardAIc-Agents: A Multimodal Framework for Cardiac Care Support](https://proceedings.mlr.press/v315/zhang26a.html)
- [RadImageNet-VQA: A Large-Scale CT and MRI Dataset for Radiologic VQA](https://proceedings.mlr.press/v315/butsanets26a.html)
- [Can Vision Language Models Track a Heartbeat? Frame-Level Echocardiogram Understanding](https://proceedings.mlr.press/v315/liu26e.html)

### Computational pathology

**The ideas behind this group** (a synthesis from the abstracts of about 28 papers in this group; the individual papers have not been read in full).

*The shared problem is scale combined with weak labels.* A whole-slide image (WSI) is gigapixel-sized, usually carries only a slide-level label, and changes appearance with stain, scanner and hospital. The standard response is multiple instance learning (MIL): a frozen patch encoder produces features for tens of thousands of tiles, and an aggregator pools them into one slide prediction. Most papers here question one part of that pipeline: the aggregator, the representation of the slide, robustness to domain shift, or how far its attention can be trusted.

*The first thread moves effort out of the aggregator, or makes it cheaper.* [CAPRMIL](https://proceedings.mlr.press/v315/lolos26a.html) injects global context into patch embeddings before pooling, after which a plain mean aggregator matches state-of-the-art MIL with far fewer trainable parameters; [dynamic token compression](https://proceedings.mlr.press/v315/rahman26b.html) gradually merges redundant patches instead of pruning them early; and [eWSI](https://proceedings.mlr.press/v315/marikkar26a.html) makes end-to-end training feasible by combining parameter-efficient fine-tuning with MIL, matching MIL built on in-domain extractors while starting from ImageNet encoders. Other papers go the other way: a [mixture of aggregators](https://proceedings.mlr.press/v315/ozlugedik26a.html), with a router choosing two per slide, argues that different diseases need different pooling, and [CrossFusion](https://proceedings.mlr.press/v315/soraki26a.html) fuses several magnifications for survival prediction. The same machinery reaches other tissues and samples: testicular tubules in toxicology, with masked-instance pretraining ([TBA-MIL](https://proceedings.mlr.press/v315/kshirsagar26a.html)), and malaria microscopy, where sample-level labels become parasite detections ([MILCA](https://proceedings.mlr.press/v315/manescu26a.html)).

*The second thread replaces the bag of patches with structure.* [Tissue-boundary graphs](https://proceedings.mlr.press/v315/weers26a.html) build nodes from natural tissue regions, coarsen homogeneous areas and attach interpretable features; the authors report results competitive with a large foundation model from over 300 times less training data. [ResGAT](https://proceedings.mlr.press/v315/lin26c.html) runs residual graph attention on k-nearest-neighbour patch graphs and adapts to a new site with few labels on a rare appendiceal cancer cohort.

*The third thread deals with stain and site.* [StainNet](https://proceedings.mlr.press/v315/li26c.html) pretrains foundation models on immunohistochemistry and special stains, since most pathology models have only seen haematoxylin and eosin. [One-by-One Stainer](https://proceedings.mlr.press/v315/moens26a.html) normalises stain with a purely $1\times1$ convolutional network, so it cannot hallucinate texture. Two papers attach a gradient reversal layer to remove site information: an adapter on a frozen pathology foundation model, with the bias quantified by a robustness index ([hospital-bias disentanglement](https://proceedings.mlr.press/v315/zhang26d.html)), and [AdvDINO](https://proceedings.mlr.press/v315/su26a.html), which removes slide-specific batch effects in multiplex immunofluorescence. Smaller preprocessing steps such as [orientation normalisation](https://proceedings.mlr.press/v315/topolnjak26a.html) of skin sections serve the same aim. A cautionary note comes from microscopy: in [malaria detection](https://proceedings.mlr.press/v315/issah26a.html), stain normalisation alone did not recover performance on external sites.

*The fourth thread asks whether explanations and labels can be trusted.* [Counterfactual attention MIL](https://proceedings.mlr.press/v315/chraki26a.html) questions attention weights as explanations and supervises them causally; [ConStruct](https://proceedings.mlr.press/v315/le26b.html) builds segmentation prototypes from a pathology vision–language model initialised with text descriptions; and [manifold-aware diffusion](https://proceedings.mlr.press/v315/yang26a.html) edits pathomics features while respecting their correlations. On the label side, a [virtual expert panel](https://proceedings.mlr.press/v315/grove26a.html) keeps every rater's Barrett's oesophagus grade rather than a consensus, improving performance and calibration; [PAGET](https://proceedings.mlr.press/v315/komura26a.html) distils teachers trained on immunohistochemically restained data, aggregated along the cell-type taxonomy; [Evidential DualU-Net](https://proceedings.mlr.press/v315/anglada-rotger26a.html) gives single-pass uncertainty for nuclei; and [3D light-sheet segmentation](https://proceedings.mlr.press/v315/hashisho26a.html) enforces continuity between sparsely labelled slices.

*A smaller fifth thread links slides to language and molecules.* [ANTONI-α](https://proceedings.mlr.press/v315/moonemans26a.html) is an open slide-level visual question answering model trained on synthetic instructions; the [Tool Bottleneck Model](https://proceedings.mlr.press/v315/liu26d.html) composes histopathology tools; [SCR²-ST](https://proceedings.mlr.press/v315/zhu26a.html) uses single-cell priors to decide where to spend a spatial-transcriptomics budget; and [HAF](https://proceedings.mlr.press/v315/zheng26a.html) fuses histopathology with other modalities for survival classification when some are missing.

*Taken together*, the field is moving from "a better aggregator" towards better instance representations, explicit tissue structure and control of what the frozen encoder has learned, including spurious hospital signatures. Shared resources such as [cell foundation-model comparisons](https://proceedings.mlr.press/v315/archit26a.html) and the [ASTIH](https://proceedings.mlr.press/v315/collin26a.html) axon datasets support this. Still open: whether attention or prototypes reflect what the model actually uses, how to validate across hospitals at scale, and how slide-level language models should be evaluated beyond question-answering accuracy.

**Representative papers**

- [Context-Aware Patch Representations for Multiple Instance Learning](https://proceedings.mlr.press/v315/lolos26a.html)
- [MoA: Mixture of Aggregators Improves Slide-Level Diagnosis](https://proceedings.mlr.press/v315/ozlugedik26a.html)
- [Counterfactual Intervention in Attention MIL for Digital Pathology](https://proceedings.mlr.press/v315/chraki26a.html)
- [Quantifying and Mitigating Hospital Domain Bias in Pathology Foundation Models](https://proceedings.mlr.press/v315/zhang26d.html)
- [Democratising Pathology Co-Pilots: An Open Pipeline and Dataset for Whole-Slide VLMs](https://proceedings.mlr.press/v315/moonemans26a.html)
- [More is more: multi-rater information for WSI grading via a virtual expert panel](https://proceedings.mlr.press/v315/grove26a.html)

### Generative models

**The ideas behind this group** (a synthesis from the abstracts of about 29 papers in this group; the individual papers have not been read in full).

*The shared problem is that medical data is scarce, unevenly distributed and hard to share, and generative models are asked to make up the difference without inventing anatomy.* Diffusion models, flow matching and their faster relatives dominate, and the papers use them in four ways: to create labelled training data, to translate between acquisitions, to remove unwanted variation, and as tools for analysis rather than image factories.

*The first thread generates training data that comes with its labels attached.* The aim is an image–annotation pair, not a realistic image on its own: paired 3D CT volumes and segmentation masks from one sampling pass of a pretrained CT diffusion model ([LabelG](https://proceedings.mlr.press/v315/wang26c.html)); findings inpainted into healthy chest X-rays at anatomically plausible positions, giving bounding boxes for free ([SemiSynCXR](https://proceedings.mlr.press/v315/posada26a.html)); lung nodules whose size, shape and radiological characteristics are each controlled by their own low-rank adapter, merged under an orthogonality loss ([nodule synthesis](https://proceedings.mlr.press/v315/singh26a.html)); and synthetic retinal vasculature with diabetic-retinopathy features plus matching reasoning text, used to train a vision–language model ([SVR](https://proceedings.mlr.press/v315/li26d.html)). Others target fairness, generating samples for under-represented groups and hard cases ([fair glaucoma](https://proceedings.mlr.press/v315/wang26d.html)), or privacy, with synthetic craniofacial skeletons from optimal-transport flow matching ([craniofacial OTFM](https://proceedings.mlr.press/v315/melacini26a.html)) and synthetic nystagmus eye-movement videos ([GenVOG-DiT](https://proceedings.mlr.press/v315/rahman26a.html)). A non-learned counterpart projects CT into synthetic radiographs that inherit CT-derived labels ([DRRs for calcification](https://proceedings.mlr.press/v315/saeed26a.html)). A sobering note on evaluation: in retinal imaging, Fréchet Inception Distance and its variants can disagree with downstream classification and segmentation performance ([FID note](https://proceedings.mlr.press/v315/wu26c.html); see also [What FID sees](../fid-geometry/index.html)).

*The second thread is translation, and its central idea is to start from the source image rather than from noise.* Because source and target share anatomy and differ mainly in contrast, several papers learn only the difference: a flow from the mean of the available MRI sequences in wavelet space, needing one or two integration steps ([WFM](https://proceedings.mlr.press/v315/tur26a.html)); a residual diffusion from non-contrast prostate MRI to dynamic contrast-enhanced phases in about 15 steps ([ResDCE-diff](https://proceedings.mlr.press/v315/kumar26a.html)); residual error-shifting super-resolution from 1.5T to 3T-like fetal MRI, trained on pairs simulated from a phantom ([fetal super-resolution](https://proceedings.mlr.press/v315/zhong26a.html)); a Schrödinger bridge that completes a partially rendered ultrasound view ([novel-view ultrasound](https://proceedings.mlr.press/v315/shi26b.html)); and time-conditioned latent diffusion that fills in cardiac phases between end-diastole and end-systole ([DiffTAC](https://proceedings.mlr.press/v315/banerjee26a.html)). Translation also crosses modalities, from EEG to fMRI time series ([UnEBOLT](https://proceedings.mlr.press/v315/li26g.html)). A useful counterpoint: for predicting post-acetazolamide perfusion, a plain conditional autoencoder beat three diffusion formulations ([post-ACZ CBF](https://proceedings.mlr.press/v315/goyal26a.html)).

*The third thread removes unwanted variation while trying hard not to hallucinate.* MRI harmonisation separates anatomy from contrast, with the target contrast given by an image or by DICOM metadata through CLIP encoders ([DIST-CLIP](https://proceedings.mlr.press/v315/avci26a.html)), or is validated with travelling subjects across more than 100 scanners ([HACA3+](https://proceedings.mlr.press/v315/hays26a.html)). The safeguards are architectural: a fully $1\times1$ convolutional stain normaliser cannot alter texture ([One-by-One Stainer](https://proceedings.mlr.press/v315/moens26a.html)); a device-removal refiner predicts residuals only inside the mask and leaves every other pixel untouched ([LightRefine-PCXR](https://proceedings.mlr.press/v315/jiang26a.html)); and a frozen Stable Diffusion prior is anchored to a classical enhancement result by gradient guidance ([hybrid enhancement](https://proceedings.mlr.press/v315/singh26b.html)).

*The fourth thread uses generative models as instruments for analysis.* Counterfactual "healthy" edits localise lesions from vertebra-level labels ([Hide-and-Seek](https://proceedings.mlr.press/v315/atad26a.html)) or quantify midline shift through reversed disease progression ([VQ flow matching](https://proceedings.mlr.press/v315/chen26b.html)); feature edits are kept on the manifold of correlated pathomics features ([MAD](https://proceedings.mlr.press/v315/yang26a.html)); diffusion trained on normal anatomy detects anomalies in chest CT ([SAC-Diff](https://proceedings.mlr.press/v315/zheng26b.html)); a chest X-ray latent diffusion model is tuned for phrase grounding ([anatomy-grounded prompt tuning](https://proceedings.mlr.press/v315/vilouras26a.html)); one diffusion process over images and clinical metadata replaces separate predictors ([MetaVoxel](https://proceedings.mlr.press/v315/liu26a.html)); diffusion over logits instead of the probability simplex targets calibration ([Simplex-Aligned Diffusion](../2026-wu-simplex-aligned-diffusion/index.html)); and forecasting is posed as predicting future deformation fields rather than future intensities ([FutureMorph](https://proceedings.mlr.press/v315/khawaled26a.html)).

*Taken together*, the trend is away from unconditional sampling and towards generators that are tightly conditioned, start from an informative prior, take few steps, and are judged by what they do downstream. The same models reappear as priors in reconstruction. Still open: there is no agreed metric of usefulness, since FID is known to mislead and most papers report task gains on their own setups; hallucination is handled by design choices rather than guarantees; and privacy claims for synthetic data are asserted rather than measured.

**Representative papers**

- [Simplex-Aligned Diffusion with Cross-Granularity Interaction for Robust Classification](https://proceedings.mlr.press/v315/wu26a.html)
- [Flow Matching for 3D Craniofacial Skeletal Data Generation](https://proceedings.mlr.press/v315/melacini26a.html)
- [WFM: 3D Wavelet Flow Matching for Ultrafast Multi-Modal MRI Synthesis](https://proceedings.mlr.press/v315/tur26a.html)
- [MetaVoxel: Joint Diffusion Modeling of Imaging and Clinical Metadata](https://proceedings.mlr.press/v315/liu26a.html)
- [LabelG: Consistent Pairwise 3D CT Image and Mask Generation](https://proceedings.mlr.press/v315/wang26c.html)
- [A Pragmatic Note on Evaluating Generative Models with FID for Retinal Image Synthesis](https://proceedings.mlr.press/v315/wu26c.html)

### Reconstruction and physics

**The ideas behind this group** (a synthesis from the abstracts of about 17 papers in this group; the individual papers have not been read in full).

*The shared problem is an inverse problem.* The scanner measures $y = A x + \varepsilon$, where $A$ is undersampled k-space, a blurred and motion-corrupted slice stack or a decaying echo train, and the task is to recover $x$ fast enough for clinical use. The papers differ in where they put the learning: in what is measured, in the prior over images, or in a differentiable model of the physics; a final set asks whether better reconstructions actually help.

*The first strategy designs the acquisition and the reconstruction together.* Rather than a fixed or random sampling mask, sampling is learned jointly with the reconstruction network: case-specific, optionally frame-specific, patterns for dynamic cardiac MRI ([E2E-ADS-Recon](https://proceedings.mlr.press/v315/yiasemis26a.html)); non-Cartesian trajectories for cardiac $T_1$ mapping with the relaxation model built into the optimisation ([T1-PILOT](https://proceedings.mlr.press/v315/shor26a.html)); and a reinforcement-learning agent that chooses k-space lines to help a metal-artefact-reduction network, trained on physically simulated pairs with and without implants ([MASC](https://proceedings.mlr.press/v315/lu26a.html)). The shared idea is that what to measure depends on the case and on what the reconstruction does with the data.

*The second strategy improves the prior while keeping data consistency.* Learned iterative schemes usually ignore acquisition settings, so one paper makes the weights functions of parameters such as acceleration factor, field strength, tube current and projection count ([conditional learned reconstruction](https://proceedings.mlr.press/v315/moriakov26a.html)). Generative priors are the other route: diffusion in sparse subspaces from convolutional dictionary learning under measurement-consistency constraints ([SSDM](https://proceedings.mlr.press/v315/deng26a.html)), and a one-step consistency model with measurement-guided encoding and physics-based refinement that generalises from cardiac to knee data without retraining ([C-MORE](https://proceedings.mlr.press/v315/akoda26a.html)). The latter answers the cost of diffusion sampling raised in the generative-models group.

*The third strategy makes the forward model explicit and differentiable.* In slice-to-volume reconstruction of fetal MRI, representing the volume as anisotropic Gaussians turns the point-spread function into a closed-form covariance sum, $\Sigma_{obs}=\Sigma_{HR}+\Sigma_{PSF}$, replacing Monte Carlo sampling ([Gaussian SVR](https://proceedings.mlr.press/v315/dannecker26a.html)). A neural surrogate of a slow tumour-growth solver makes patient-specific inversion gradient-based, from days to seconds ([tumour growth inversion](../2026-weidner-tumor-growth-surrogate/index.html)). In quantitative MRI, bootstrapping subsets of the echo train turns a deterministic $T_2$-distribution network into an ensemble for low signal-to-noise pancreas imaging ([bootstrapped P2T2](https://proceedings.mlr.press/v315/ben-atya26a.html)). In radiotherapy, fluence-map prediction is treated as an ill-posed inverse problem with a loss that includes beam-wise energy conservation ([FluenceFormer](https://proceedings.mlr.press/v315/mgboh26a.html)). Physics also supplies training data: simulated microbubbles for ultrasound localisation microscopy ([GraphULM](https://proceedings.mlr.press/v315/sabih26a.html)) and phantom-simulated pairs for fetal super-resolution ([fetal super-resolution](https://proceedings.mlr.press/v315/zhong26a.html)).

*The fourth thread asks what reconstruction does downstream.* One evaluation runs U-Net, GAN and diffusion reconstructions ahead of diagnostic models and finds that peak signal-to-noise ratio poorly tracks task performance, while reconstruction sometimes amplifies sex-related bias, modestly compared with the bias already in the diagnostic models ([reconstruction and fairness](https://proceedings.mlr.press/v315/wohlrapp26a.html)). Another uses the intermediate iterates of an optical coherence tomography reconstruction for test-time adaptation of a segmentation model, with uncertainty from their disagreement ([IRTTA](https://proceedings.mlr.press/v315/pinetz26a.html)). Clinical validation of freehand 3D ultrasound reconstruction for hip dysplasia ([hip 3D ultrasound](https://proceedings.mlr.press/v315/heyer26a.html)) and lightweight restoration models for denoising ([QICNN](https://proceedings.mlr.press/v315/dutta26a.html)) and enhancement ([hybrid enhancement](https://proceedings.mlr.press/v315/singh26b.html)) belong here too.

*Taken together*, the common move is to put the measurement physics inside the learned system, through a learned sampler, a measurement-consistent prior or an analytic forward model, rather than learning a generic image-to-image map. Still open: most work is on cardiac and brain MRI, often with retrospective undersampling or simulation, so prospective acquisition is rarely tested; few papers report uncertainty; and, as the fairness study shows, pixel metrics are a weak guide to clinical value yet remain the main yardstick.

**Representative papers**

- [Conditional Learned Reconstruction for Medical Imaging](https://proceedings.mlr.press/v315/moriakov26a.html)
- [Beyond Diffusion: Consistency Models for One-Step MRI Reconstruction](https://proceedings.mlr.press/v315/akoda26a.html)
- [Sparse Subspace Diffusion Model for Physically Consistent Accelerated MRI Reconstruction](https://proceedings.mlr.press/v315/deng26a.html)
- [End-to-End Co-Optimization of Adaptive k-space Sampling and Reconstruction for Dynamic MRI](https://proceedings.mlr.press/v315/yiasemis26a.html)
- [T1-PILOT: Physics-Informed Learned Optimized Trajectories for T1 Mapping](https://proceedings.mlr.press/v315/shor26a.html)
- [Fast and Explicit: Slice-to-Volume Reconstruction via 3D Gaussians with Analytic PSF](https://proceedings.mlr.press/v315/dannecker26a.html)

### 3D scene representations

**The ideas behind this group** (a synthesis from the abstracts of about 16 papers in this group and the ones next to it; the individual papers have not been read in full).

*The shared problem is that anatomy is continuous in space and time but is observed as sparse 2D slices, frames or views.* Cardiac MRI comes as short- and long-axis planes, fetal MRI as motion-corrupted slice stacks, endoscopy as a monocular video with moving tissue, and freehand ultrasound as tracked B-mode frames. The papers replace the voxel grid with a representation that can be queried anywhere: an implicit neural representation (INR) or neural field, a network mapping coordinates to values, or an explicit set of Gaussian or triangle primitives that can be rendered.

*The first thread uses INRs as continuous deformation and motion fields.* For registration, a dual-branch multi-scale INR separates global from local lung motion to handle large deformations ([multi-scale lung INR](https://proceedings.mlr.press/v315/gebauer26a.html)), and an INR recovers the susceptibility field for echo-planar imaging distortion correction from a single b0 image, with higher-order regularisers computed analytically because the field is smooth ([EPINR](https://proceedings.mlr.press/v315/spears26a.html)). For the heart, one model builds a single 3D+time representation from all planes and learns each plane's pose to correct slice misalignment ([NISF++](https://proceedings.mlr.press/v315/stolt-anso26a.html)). Deformation codes from an INR also expose morphological biases that CLIP-type features miss ([INR bias detection](https://proceedings.mlr.press/v315/engelson26a.html)).

*The second thread attacks the main weakness of INRs, that each subject needs its own slow optimisation.* Three answers appear: warm-start cine cardiac MRI motion from a population prior averaged over INRs fitted to tagged MRI, reaching near-optimal performance in about half the steps ([tagged-informed prior](https://proceedings.mlr.press/v315/alvarez-florez26a.html)); meta-learn one shared field modulated by a per-instance latent vector, with a layer-dependent SIREN frequency and sparse supervision to scale to large datasets ([MedFuncta](https://proceedings.mlr.press/v315/friedrich26a.html)); and replace the single global latent with a geometry-aware latent point cloud, giving local, transformation-robust codes for downstream prediction ([equivariant neural fields](https://proceedings.mlr.press/v315/wiers26a.html)). Together these turn neural fields from a per-scan fitting tool into a representation for whole datasets.

*The third thread switches to explicit primitives, Gaussian and triangle splatting, mostly for endoscopy.* Explicit primitives render fast and can carry geometric priors. A colonoscopy Gaussian map is given tubular coordinates $(s,r,\theta)$ around an online centreline, which regularises the map towards a hollow tube and yields coverage statistics ([centreline Gaussians](https://proceedings.mlr.press/v315/dunn-beltran26a.html)). Triangles enforce surface continuity, combined with depth supervision invariant to per-frame scale and shift and a coaxial-spotlight lighting model so that specular highlights are not mistaken for geometry ([GutSee](https://proceedings.mlr.press/v315/hardy26a.html)), and a canonical triangle set with a deformation network handles deforming tissue in 4D ([Endo-4DTS](https://proceedings.mlr.press/v315/salort-benejam26a.html)). Neighbouring papers supply the inputs these methods assume: depth ([EndoStreamDepth](https://proceedings.mlr.press/v315/li26f.html)), camera poses ([RVO-MIS](https://proceedings.mlr.press/v315/wang26b.html)) and metric scale ([polyp size](https://proceedings.mlr.press/v315/lawrence26a.html)).

*The fourth thread builds the sensor's physics into the renderer.* For ultrasound, a 3D Gaussian field stores attenuation and reflection, and B-mode images are formed by ray casting, capturing view-dependent attenuation ([UltraG-Ray](https://proceedings.mlr.press/v315/duelmer26a.html)); an alternative renders a partial view from a point cloud and completes it with a diffusion bridge ([novel-view ultrasound](https://proceedings.mlr.press/v315/shi26b.html)). For fetal MRI, Gaussians make the slice point-spread function analytic ([Gaussian SVR](https://proceedings.mlr.press/v315/dannecker26a.html)). The colonoscopy lighting model above is the optical counterpart.

*Taken together*, the field is moving from implicit networks fitted one scan at a time towards priors, shared fields and explicit primitives, with rendering models that respect how each modality forms an image. Still open: endoscopic results are largely on phantoms with known or estimated poses, novel ultrasound views are judged by image similarity rather than clinical use, and it is unclear when a neural field beats a well-tuned grid-based method on clinical tasks rather than on reconstruction quality.

**Representative papers**

- [MedFuncta: A Unified Framework for Learning Efficient Medical Neural Fields](https://proceedings.mlr.press/v315/friedrich26a.html)
- [Geometry-Aware Cardiac MRI Representation Learning with Equivariant Neural Fields](https://proceedings.mlr.press/v315/wiers26a.html)
- [NISF++: Implicit representations of 3D+time cardiac function from 2D MR views](https://proceedings.mlr.press/v315/stolt-anso26a.html)
- [UltraG-Ray: Physics-Based Gaussian Ray Casting for Novel Ultrasound View Synthesis](https://proceedings.mlr.press/v315/duelmer26a.html)
- [Rendering with a Gut Feeling: Depth-Guided Triangle Splatting for Colonoscopic Reconstruction](https://proceedings.mlr.press/v315/hardy26a.html)
- [Endo-4DTS: Monocular 4D Scene Synthesis for Endoscopy](https://proceedings.mlr.press/v315/salort-benejam26a.html)

### Trustworthiness

**The ideas behind this group** (a synthesis from the abstracts of about 40 papers in this group; the individual papers have not been read in full).

*The shared problem is that accuracy on a held-out set says little about whether a model should be trusted.* A model can be right with the wrong confidence, right for the wrong reason, right on average but wrong for a subgroup, or right until someone perturbs the input. The papers split along these four failure types, and one pattern runs through them: building the check into the model works better than adding it afterwards.

*The first thread makes confidence a quantity that is produced, measured and then used.* Several papers change the training objective so that uncertainty comes out in one pass: evidential heads for cell instance segmentation ([Evidential DualU-Net](https://proceedings.mlr.press/v315/anglada-rotger26a.html)) and image retrieval ([Evidential Retriever](https://proceedings.mlr.press/v315/arvapalli26a.html)), diffusion on logits instead of the bounded probability simplex ([Simplex-Aligned Diffusion](../2026-wu-simplex-aligned-diffusion/index.html)), heteroscedastic ([pectoral muscle](https://proceedings.mlr.press/v315/zech26a.html)) or topology-linked ([TUNE++](https://proceedings.mlr.press/v315/dhor26a.html)) heads, and probabilistic imputation of missing modalities in federated learning ([P-FIN](https://proceedings.mlr.press/v315/shahid26a.html)). Calibration is attacked through the gradient of region-based losses ([gradient surgery](../2026-lux-gradient-surgery/index.html)), by keeping disagreement between raters instead of collapsing to one label ([virtual expert panel](https://proceedings.mlr.press/v315/grove26a.html)), and by calibrated gating of pseudo-labels ([UltraSemiNet](https://proceedings.mlr.press/v315/khan26a.html)). The more distinctive papers ask what uncertainty is *for*: propagating segmentation miscalibration into confidence intervals for ratio biomarkers ([CARE](https://proceedings.mlr.press/v315/li26b.html)), weighting landmarks by their uncertainty in a Perspective-n-Point pose solve ([pelvic pose](https://proceedings.mlr.press/v315/suh26a.html)), and releasing per-node uncertainty for a large chest X-ray dataset ([CheXmask-U](https://proceedings.mlr.press/v315/cosarinsky26a.html)). Three papers test whether it holds up: whether Bayesian segmenters' uncertainty tracks error under corruption ([Bayesian V&V](https://proceedings.mlr.press/v315/carannante26a.html)), whether uncertainty maps change how clinicians edit contours ([uncertainty maps](https://proceedings.mlr.press/v315/maruccio26a.html)), and whether Vision Transformer gains survive once training variance is reported ([ViT stability](https://proceedings.mlr.press/v315/navet26a.html)).

*The second thread prefers explanations that are part of the model to explanations added after it.* [SoftCAM](https://proceedings.mlr.press/v315/djoumessi26a.html) replaces global pooling with a convolutional evidence layer, so the class map is the prediction. Concept bottlenecks route the decision through named concepts: Liver Imaging Reporting and Data System features ([Li-RADS bottleneck](https://proceedings.mlr.press/v315/monnin26a.html)), or predefined concepts complemented by mined ones ([concept complement](https://proceedings.mlr.press/v315/wang26a.html)). Further designs use shared prototypes ([UM–ProtoShare](https://proceedings.mlr.press/v315/golbaf26a.html)), tissue-region graphs ([pixels to histopathology](https://proceedings.mlr.press/v315/weers26a.html)) and composition of tool outputs ([tool bottleneck](https://proceedings.mlr.press/v315/liu26d.html)). In multiple instance learning, attention is questioned as an explanation and replaced by a counterfactual intervention ([counterfactual MIL](https://proceedings.mlr.press/v315/chraki26a.html)). Generative editing explains features by changing them: correlated pathomics features ([MAD](https://proceedings.mlr.press/v315/yang26a.html)) and reversed disease progression ([vector quantisation](https://proceedings.mlr.press/v315/chen26b.html)); radiomics serves as an interpretable map of what fine-tuning changes ([radiomic lens](https://proceedings.mlr.press/v315/wang26e.html)). Two papers ask how explanations themselves should be validated: a synthetic dataset with a known decision rule ([FunnyNodules](https://proceedings.mlr.press/v315/gallee26a.html)), and intervals for Shapley values of MRI contrasts ([glioma explanations](../2026-ren-shapley-uncertainty/index.html)).

*The third thread hunts for shortcuts and hidden failure modes without needing labels for them.* Chest X-ray models can predict health-insurance type, a proxy for socio-economic status, from normal images ([insurance type](https://proceedings.mlr.press/v315/chen26c.html)), and pathology foundation models encode the hospital a slide came from ([hospital bias](https://proceedings.mlr.press/v315/zhang26d.html)). Proposed tools for finding such problems compare classifier weights for the clinical task and for metadata tasks ([weight space correlation](https://proceedings.mlr.press/v315/wong26a.html)), cluster images by frequency spectrum ([spectral failure modes](https://proceedings.mlr.press/v315/halimi26a.html)), adapt slice-discovery methods to segmentation ([SEG4SEG](https://proceedings.mlr.press/v315/weng26a.html)), and search for subgroups in deformation space rather than image-feature space ([morphological bias](https://proceedings.mlr.press/v315/engelson26a.html)). Restricting the input can expose or remove shortcuts: a liver mask on the input redirects attention away from subcutaneous fat ([UKBB liver](https://proceedings.mlr.press/v315/al-belmpeisi26a.html)), and strict CT quality control lowers the performance of some models, which the authors read as reliance on context ([Virtual-Eyes](https://proceedings.mlr.press/v315/hoq26a.html)).

*The fourth thread covers fairness and adversarial robustness.* Fairness methods enforce constraints inside small ensembles ([OxEnsemble](https://proceedings.mlr.press/v315/rystrom26a.html)), weight active-learning entropy by group performance ([fair active learning](https://proceedings.mlr.press/v315/danaee26a.html)), or synthesise data for groups the model handles badly ([fair glaucoma](https://proceedings.mlr.press/v315/wang26d.html)). One study finds that AI image reconstruction sometimes amplifies demographic bias, though only modestly ([reconstruction fairness](https://proceedings.mlr.press/v315/wohlrapp26a.html)). The adversarial papers make attacks clinically realistic: perturbations constrained to stay radiomically plausible ([FRD attacks](https://proceedings.mlr.press/v315/mahmoud26a.html)), and attacks designed to produce clinically dangerous CT reports ([CRA-RG](https://proceedings.mlr.press/v315/irshad26a.html)).

*Taken together*, the threads move the same way: from reporting a property afterwards to building it into the model and testing it against something external, such as a known decision rule, a clinician's behaviour or a downstream biomarker. Still open: few papers report more than one trust property at once, for example whether a calibrated model is also fair; shortcut detection still mostly needs the confounder to be named in advance; and evidence that uncertainty or explanations improve clinical decisions is limited to small user studies.

**Representative papers**

- [Can You Trust Your Model? Uncertainty Approximations Guaranteeing Validity of Glioma Segmentation Explanations](https://proceedings.mlr.press/v315/ren26a.html)
- [CARE: Confidence-aware Ratio Estimation for Medical Biomarkers](https://proceedings.mlr.press/v315/li26b.html)
- [Testing the Trust: Verification and Validation of Bayesian Segmentation under Uncertainty](https://proceedings.mlr.press/v315/carannante26a.html)
- [Weight Space Correlation Analysis: Quantifying Feature Utilization](https://proceedings.mlr.press/v315/wong26a.html)
- [Revealing Hidden Failure Modes in Chest X-ray Classification via Spectral Domain Analysis](https://proceedings.mlr.press/v315/halimi26a.html)
- [SEG4SEG: Identifying Systematic Failure Modes in Segmentation by Subgroup Discovery](https://proceedings.mlr.press/v315/weng26a.html)
- [OxEnsemble: Fair Ensembles for Low-Data Classification](https://proceedings.mlr.press/v315/rystrom26a.html)
- [SoftCAM: Making black box models self-explainable](https://proceedings.mlr.press/v315/djoumessi26a.html)

### Registration and geometry

**The ideas behind this group** (a synthesis from the abstracts of about 20 papers in this group; the individual papers have not been read in full).

*The shared problem is correspondence: deciding which point in one image, surface or video frame corresponds to which point in another.* The mapping should be accurate where data exist, plausible where they do not, and stable across scans and patients. The papers differ mainly in what the mapping is (a dense deformation, a map between surfaces, a rigid pose or a sparse set of landmarks) and in where the prior that keeps it plausible comes from.

*The first thread revisits dense deformable registration and often favours optimisation for each image pair over a single trained network.* Implicit neural representations model the deformation as a continuous function fitted to one pair: a dual-branch, multi-scale INR separates global from local motion for large lung deformations ([multi-scale INR](https://proceedings.mlr.press/v315/gebauer26a.html)), an INR corrects echo-planar distortion in diffusion MRI without an opposite phase-encoded scan ([EPINR](https://proceedings.mlr.press/v315/spears26a.html)), and for cine cardiac MRI a population prior built from tagged MRI warm-starts the fit ([tagged-informed prior](https://proceedings.mlr.press/v315/alvarez-florez26a.html)). A learned network followed by pairwise instance optimisation is the recipe for longitudinal cortical surfaces ([anatomical longitudinal registration](https://proceedings.mlr.press/v315/saboo26a.html)). Registration also becomes a building block elsewhere: it drives daily anatomy updates in adaptive proton therapy ([digital twin](https://proceedings.mlr.press/v315/chang26a.html)), a metadata-conditioned network predicts *future* diffeomorphic deformations from a baseline scan ([FutureMorph](https://proceedings.mlr.press/v315/khawaled26a.html)), and INR deformation fields serve as features for bias discovery ([morphological bias](https://proceedings.mlr.press/v315/engelson26a.html)). An independent re-evaluation of the LUMIR challenge checks the trend towards learned methods and finds that deep methods match iterative optimisation in distribution but degrade on unseen MRI contrasts and high resolutions, and are sensitive to preprocessing ([LUMirage](https://proceedings.mlr.press/v315/jena26a.html)).

*The second thread treats anatomy as a surface or shape, not a grid of voxels.* [CSVR](https://proceedings.mlr.press/v315/masui26a.html) combines surface-based cortical alignment with volumetric registration of the subcortex for neonatal brains. [SuD-CoTAN](https://proceedings.mlr.press/v315/grigorescu26a.html) fits fetal cortical meshes and aligns them to age-matched templates in one step, guided by sulcal depth, and, like the longitudinal surface registration above, treats topological integrity as a success criterion. Shape priors also predict hidden anatomy: a mean organ shape is deformed to fit a patient's body-surface point cloud ([surface to viscera](https://proceedings.mlr.press/v315/atici26a.html)). Cardiac planes are given learnable rigid poses inside a shared implicit representation to correct motion between slices ([NISF++](https://proceedings.mlr.press/v315/stolt-anso26a.html)).

*The third thread is rigid pose, where the geometry of rotations matters.* Biplanar digital subtraction angiography is registered directly to CT angiography with soft consistency constraints on the rotation group $SO(3)$, so vessel segmentation is not needed ([GeoReg](https://proceedings.mlr.press/v315/herten26a.html)). 3D ultrasound volumes are stitched without training by combining point-cloud geometry with DINOv3 descriptors ([VFMStitch](https://proceedings.mlr.press/v315/yao26a.html)). Surgical visual odometry solves for absolute camera poses first, using robust matching ([RVO-MIS](https://proceedings.mlr.press/v315/wang26b.html)). Two papers model periodic quantities explicitly: tissue rotation angle as classification over a divided circle ([orientation normalisation](https://proceedings.mlr.press/v315/topolnjak26a.html)), and cardiac phase regressed on the unit circle ([CyclePhase](https://proceedings.mlr.press/v315/ben-haddou26a.html)).

*The fourth thread works with landmarks and keypoints, which are sparse but easy to interpret.* [nnLandmark](https://proceedings.mlr.press/v315/ertl26a.html) brings nnU-Net-style self-configuration to 3D landmark heatmaps to address inconsistent baselines. Landmark uncertainty from Monte Carlo dropout weights a Perspective-n-Point solve for 2D/3D pelvic pose ([uncertainty-weighted PnP](https://proceedings.mlr.press/v315/suh26a.html)), landmark-based chest X-ray segmentation gains per-node uncertainty ([CheXmask-U](https://proceedings.mlr.press/v315/cosarinsky26a.html)), and keypoints also describe surgical instruments ([MID-POSE](https://proceedings.mlr.press/v315/wei26a.html)) and children's movements ([pose-based analysis](https://proceedings.mlr.press/v315/dai26a.html)).

*Taken together*, the group prefers structure to scale: explicit regularity (diffeomorphisms, topology-preserving meshes, $SO(3)$ constraints), priors from populations or foundation features, and fitting each case at test time when a single network does not generalise. Still open: the LUMIR re-evaluation suggests that claims of zero-shot learned registration need stricter protocols; uncertainty is almost never reported for dense deformations, although the landmark papers show how it could feed into downstream pose; and few papers test whether better alignment changes a clinical measurement.

**Representative papers**

- [GeoReg: Biplanar DSA-to-CTA registration with geodesic consistency](https://proceedings.mlr.press/v315/herten26a.html)
- [Robust Multi-Scale INRs for Large-Deformation Lung Registration](https://proceedings.mlr.press/v315/gebauer26a.html)
- [Anatomical Longitudinal Cortical Surface Registration](https://proceedings.mlr.press/v315/saboo26a.html)
- [CSVR: Combined Surface and Volume Registration for Neonatal Brain MRI](https://proceedings.mlr.press/v315/masui26a.html)
- [FutureMorph: Toward Predicting Future Deformation Fields in Longitudinal Imaging](https://proceedings.mlr.press/v315/khawaled26a.html)
- [nnLandmark: A Self-Configuring Method for 3D Medical Landmark Detection](https://proceedings.mlr.press/v315/ertl26a.html)

### Benchmarks and validation

**The ideas behind this group** (a synthesis from the abstracts of about 30 papers in this group; the individual papers have not been read in full). Of the 221 papers, 23 were accepted in MIDL's separate Validation Studies track; they are the last 23 in the volume, from *The LUMirage* to *Harmonizing MR Images Across 100+ Scanners*, and most of them appear below.

*The shared question is how far a reported number can be trusted.* These papers mostly do not propose new methods. They measure how results change with scanner, site, sequence, preprocessing, random seed or evaluation protocol, and they build the datasets needed to do so. The recurring finding is that the main source of error is often not the architecture.

*The first thread measures variability in the imaging itself.* Even under controlled test–retest conditions, FreeSurfer-based pipelines show volume variation of up to 7–8% in small subcortical structures, comparable to the longitudinal changes researchers hope to detect ([brain segmentation reproducibility](https://proceedings.mlr.press/v315/kondrateva26a.html)). [HACA3⁺](https://proceedings.mlr.press/v315/hays26a.html) validates MR harmonisation on data from more than 100 scanners using travelling subjects. [CrossPan](https://proceedings.mlr.press/v315/peng26a.html) finds that switching between MRI sequences is far more damaging than switching centres: pancreas models with Dice above 0.85 in-domain collapse to near zero across sequences, and domain-generalisation methods barely help. External validation tells similar stories for haemorrhage segmentation across 17 sites ([IPH benchmark](https://proceedings.mlr.press/v315/duarte26a.html)), malaria detection that loses over 80% on unseen data ([malaria head-to-head](https://proceedings.mlr.press/v315/issah26a.html)), and an interstitial lung disease model trained on one screening cohort and tested on another without adaptation ([ILD screening](https://proceedings.mlr.press/v315/mcconnell26a.html)). Preprocessing matters too: a lung CT quality-control pipeline helped a generalist foundation model but degraded specialist models, which the authors read as reliance on contextual shortcuts ([Virtual-Eyes](https://proceedings.mlr.press/v315/hoq26a.html)).

*The second thread re-examines published claims.* An [independent re-evaluation of the LUMIR challenge](https://proceedings.mlr.press/v315/jena26a.html) finds that its zero-shot registration claims do not hold on out-of-distribution contrasts or at high resolution, and calls for protocols that reflect real workflows. Others find that a simpler model does as well: combining imaging and clinical data for stroke outcome only matches logistic regression on a multi-centre test set ([stroke outcome](https://proceedings.mlr.press/v315/abouyoussef26a.html)); tuned classical models and graph kernels compete with graph neural networks on resting-state fMRI connectomes ([rs-fMRI benchmark](https://proceedings.mlr.press/v315/mhanna26a.html)); and reported vision-transformer gains for dementia classification shrink once seed variability is reported ([ViT stability](https://proceedings.mlr.press/v315/navet26a.html)). Metrics are questioned as well: peak signal-to-noise ratio poorly tracks downstream diagnostic accuracy and fairness after reconstruction ([reconstruction and fairness](https://proceedings.mlr.press/v315/wohlrapp26a.html)), Fréchet Inception Distance can disagree with task-based evaluation of synthetic retinal images ([FID note](https://proceedings.mlr.press/v315/wu26c.html); see [What FID sees](../fid-geometry/index.html)), Bayesian segmentation is tested under corruption rather than only on clean data ([verification and validation](https://proceedings.mlr.press/v315/carannante26a.html)), and normal chest X-rays turn out to predict health-insurance type ([insurance from CXR](https://proceedings.mlr.press/v315/chen26c.html)).

*The third thread runs controlled head-to-head comparisons of models people are about to adopt*: [SAM 2 versus SAM 3](https://proceedings.mlr.press/v315/chakrabarty26a.html) across 16 datasets, with named failure modes; [cell segmentation foundation models](https://proceedings.mlr.press/v315/archit26a.html); eleven architectures on a curated breast ultrasound set with patient-level splits ([sonomammogram benchmark](https://proceedings.mlr.press/v315/gunawardhana26a.html)); 2D and 3D segmentation for neonatal hip ultrasound ([hip dysplasia](https://proceedings.mlr.press/v315/heyer26a.html)); general VLMs on echocardiogram frame localisation ([echo VLM benchmark](https://proceedings.mlr.press/v315/liu26e.html)); and LLM coding agents on 20 medical imaging challenges ([ReX-MLE](https://proceedings.mlr.press/v315/kenia26a.html)).

*The fourth thread builds the resources validation needs*: a fully parameterised synthetic dataset with known decision rules for testing explainability methods ([FunnyNodules](https://proceedings.mlr.press/v315/gallee26a.html)); semi-synthetic chest X-ray localisation data made by inpainting findings into healthy images ([SemiSynCXR](https://proceedings.mlr.press/v315/posada26a.html)); landmark segmentations with per-node uncertainty ([CheXmask-U](https://proceedings.mlr.press/v315/cosarinsky26a.html)); multi-species axon datasets ([ASTIH](https://proceedings.mlr.press/v315/collin26a.html)); a shortcut-resistant question-answering set ([RadImageNet-VQA](https://proceedings.mlr.press/v315/butsanets26a.html)); and a two-country skin nanotexture cohort ([stratum corneum](https://proceedings.mlr.press/v315/wang26h.html)). A user study adds the human side: simulated uncertainty maps did not change clinicians' editing of radiotherapy contours, and trust in them was low ([uncertainty maps in editing](https://proceedings.mlr.press/v315/maruccio26a.html)).

*Taken together*, the group argues that validation design deserves the same care as model design: separate sequence, scanner and site effects; report variance across seeds and use paired tests; check that proxy metrics track the clinical task; and re-test headline claims independently. Still open: who funds and maintains independent re-evaluations, how to build shared multi-site test sets that are not already in pretraining corpora, and how to measure whether an "improvement" changes what clinicians actually do.

**Representative papers**

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
- [SCR²-ST: Single Cell with Spatial Transcriptomics for Active Sampling via RL](https://proceedings.mlr.press/v315/zhu26a.html)
- [BETA: Resting-state fMRI Biotypes for tDCS Efficacy in Anxiety Among Older Adults](https://proceedings.mlr.press/v315/stolte26a.html)
- [Beyond Machine Interpretation: Learning from Expert Over-Reads Improves ECG Diagnosis](https://proceedings.mlr.press/v315/kwak26a.html): not imaging at all; over two million ECGs, each with a machine reading and a cardiologist's correction, so the model learns from the gap between the two.
- [Unpaired Multimodal Learning for Biological Datasets](https://proceedings.mlr.press/v315/ji26a.html): molecular biology, where measuring a sample often destroys it (RNA sequencing), so paired data cannot exist.
- [GEMCONT: Genetics-based Multimodal Contrastive Learning](https://proceedings.mlr.press/v315/sens26a.html): genetic variation shapes the image embeddings, aimed at genetic discovery rather than diagnosis.
- [From Surface to Viscera: 3D Estimation of Internal Anatomy from Body Surface Point Clouds](https://proceedings.mlr.press/v315/atici26a.html): organ positions predicted from a body-surface point cloud before any scan, to help position the patient.
- [Flow Matching for 3D Craniofacial Skeletal Data Generation](https://proceedings.mlr.press/v315/melacini26a.html): synthetic skeletal shape data, motivated by privacy.

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
- [Chronological Contrastive Learning: Few-Shot Progression Assessment in Irreversible Diseases](https://proceedings.mlr.press/v315/watzenbock26a.html): the order of scans in time is the supervision, since in an irreversible disease a later scan cannot be less severe.
- [CyclePhase](https://proceedings.mlr.press/v315/ben-haddou26a.html) and [Orientation Normalization of Multi-Stain Skin Tissue Cross-Sections](https://proceedings.mlr.press/v315/topolnjak26a.html): periodic targets (cardiac phase, rotation angle) treated on the circle rather than as ordinary numbers.
- [GeoReg](https://proceedings.mlr.press/v315/herten26a.html) and [Geometry-Aware Cardiac MRI Representation Learning with Equivariant Neural Fields](https://proceedings.mlr.press/v315/wiers26a.html): constraints on the rotation group $SO(3)$ and equivariance.

### Papers that critique the field

- [A Pragmatic Note on Evaluating Generative Models with FID for Retinal Image Synthesis](https://proceedings.mlr.press/v315/wu26c.html): FID does not track downstream usefulness.
- [The LUMirage: An independent evaluation of zero-shot performance in the LUMIR challenge](https://proceedings.mlr.press/v315/jena26a.html)
- [ReX-MLE: The Autonomous Agent Benchmark for Medical Imaging Challenges](https://proceedings.mlr.press/v315/kenia26a.html)
- [Test-Time Scaling in Clinical Decision Making](https://proceedings.mlr.press/v315/byun26a.html): an LLM-era idea brought to the clinic.
- [Validating the Benefit of Combining Imaging and Clinical Data for Ischemic Stroke Outcome Prediction](https://proceedings.mlr.press/v315/abouyoussef26a.html): on a multi-centre test set of over 600 cases, adding imaging only matches logistic regression on clinical data (AUROC 0.72), and the imaging features carry little weight.
- [On the Stability and Robustness of Vision Transformers for Neurodegenerative Disease Classification](https://proceedings.mlr.press/v315/navet26a.html): reported gains shrink once variation across random seeds is reported.
- [Impact of uncertainty maps on manual editing of rectal cancer segmentation in radiotherapy](https://proceedings.mlr.press/v315/maruccio26a.html): a user study in which uncertainty maps did not change how clinicians edited contours, and trust in them was low.
- [FunnyNodules: A Customizable Medical Dataset Tailored for Evaluating Explainable AI](https://proceedings.mlr.press/v315/gallee26a.html): a synthetic dataset with a known decision rule, built to test whether explanation methods find the right reasons.

### Unusual clinical framing

- [A Deep Learning-Enabled Digital Twin for Fast Online Adaptive Proton Therapy](https://proceedings.mlr.press/v315/chang26a.html)
- [From Cross-Sectional CT to Dynamic Insights: Pseudotime-Based Modeling of Lung Nodule Progression](https://proceedings.mlr.press/v315/zhuang26a.html): borrows pseudotime from single-cell biology.
- [Brain Tumor Growth Inversion via Differentiable Neural Surrogates](https://proceedings.mlr.press/v315/weidner26a.html): solving a PDE inverse problem through a learned surrogate.

### Imaging as a stand-in for another measurement

Papers that predict, from routine or cheap imaging, something normally measured in a more invasive,
expensive or different way.

- [Non-invasive estimation of haemodynamic parameters in pulmonary hypertension](https://proceedings.mlr.press/v315/cheng26a.html): echocardiography cine loops instead of right-heart catheterisation.
- [Generating Post-Acetazolamide Cerebral Blood Flow MRI for High-Risk Stroke Patients](https://proceedings.mlr.press/v315/goyal26a.html): predicting the response to a drug challenge without giving the drug.
- [UnEBOLT: A Unified Model for EEG-to-BOLD Translation](https://proceedings.mlr.press/v315/li26g.html): cheap, portable EEG standing in for fMRI.
- [REVEAL: Retinal Morphometry and Clinical Risks for Incident AD and Dementia Prediction](https://proceedings.mlr.press/v315/leem26a.html): the eye as a window onto brain disease.
- [X-Cardia: Opportunistic Cardiac Screening on Routine Chest CT](https://proceedings.mlr.press/v315/nizam26a.html): cardiac assessment from scans taken for other reasons.
- [Machine-Learning Based Detection of Coronary Artery Calcification Using Synthetic Chest X-Rays](https://proceedings.mlr.press/v315/saeed26a.html): CT-derived labels carried over to cheap X-rays through simulated radiographs.

The [health-insurance paper](https://proceedings.mlr.press/v315/chen26c.html) under *Social or ethical findings* is the cautionary twin of this group: imaging can stand in for things nobody intended, too.

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
