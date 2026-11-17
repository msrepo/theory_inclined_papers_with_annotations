---
title: "IJCNN 2026 Position Paper Track: topic map, outliers and theory picks"
authors: "Topic notes"
venue: "IJCNN 2026 (IEEE WCCI 2026, Position Paper Track)"
tags: [ijcnn, wcci, position-papers, conference-overview, uncertainty, edge-ai, robustness, class-imbalance]
status: living
category: "Research practice & policy"
subcategory: "Conference overviews"
short_title: "IJCNN 2026 — position papers"
summary: "a map of the 11 papers in the first IJCNN Position Paper Track (IEEE WCCI 2026), plus the two companion position papers at CEC and FUZZ-IEEE, with a synthesis of each group, theory picks and relevance to medical imaging"
related: [icml2026-position-overview, neurips2026-position-overview, class-imbalance-overview]
---

## Why this page exists

IJCNN 2026, held within the IEEE World Congress on Computational Intelligence (WCCI 2026) in
Maastricht in June 2026, ran its **first Position Paper Track**. The call asked for papers that take
a clear stance on neural networks and learning systems, back it with reasoned argument, and are "not
standard empirical papers in disguise"; titles had to begin with "Position Paper:". The track was
shared across the three WCCI conferences (IJCNN, CEC and FUZZ-IEEE), and its chairs included Manuel
Roveri and Fabrizio Pittorino.

The papers are in IEEE Xplore, in the *2026 International Joint Conference on Neural Networks
(IJCNN)* proceedings. Searching Xplore for 2026 titles beginning "Position Paper:" finds **11 IJCNN
papers**, plus **one at CEC and one at FUZZ-IEEE**, listed separately below. This is far smaller than
the [ICML 2026](../icml2026-position-overview/index.html) (212) or
[NeurIPS 2026](../neurips2026-position-overview/index.html) (137) tracks. The track may also have
accepted papers whose titles lack the prefix; those would not show up in this search.

The grouping and syntheses are from **titles and abstracts only**. Links go to each paper's
IEEE Xplore page, where the abstract is free and the full text needs a subscription.

## Topic groups at a glance

| Group | Papers | What's in it |
|---|---|---|
| [Reliability in deployment](#reliability-in-deployment) | 5 | Uncertainty for clinical decisions, robust and shared compressed models, adaptive edge AI, post-solve robustness of optimisers, imbalance in semi-supervised streams |
| [New foundations for neural computation](#new-foundations-for-neural-computation) | 2 | Neural computing in Hilbert lattices of interval numbers, neuromodulation as a third axis of learning |
| [Large language models](#large-language-models) | 2 | "Vaccinating" LLMs against falsehoods, LLMs as the cognitive core of robots |
| [Research practice and AI for science](#research-practice-and-ai-for-science) | 2 | Managing risk in AI projects, the place of AI researchers in scientific discovery |
| [Companion papers at CEC and FUZZ-IEEE](#companion-papers-at-cec-and-fuzz-ieee) | 2 | LLM advice on benchmarking evolutionary multi-objective optimisation, neuro-fuzzy fusion for deepfake detection |

**Overall pattern:** unlike the ICML and NeurIPS tracks, which argue mostly about evaluation,
governance and large models, this track is about **systems that have to keep working after
deployment**: under drift, on constrained hardware, with scarce labels, or as inputs to a downstream
decision. It also keeps IJCNN's interest in brain-inspired and non-standard mathematical
foundations.

## Topic groups

### Reliability in deployment

**The ideas behind this group** (a synthesis from the abstracts of 5 papers; the papers have not been read in full).

*The shared problem is that a model, or a solver, is validated once and then deployed into conditions
that the validation did not cover.* Each paper names a place where that gap is ignored and argues
that reliability must become a reported output, not an assumption.

*The first thread concerns uncertainty and decisions.* Common uncertainty quantification (UQ)
techniques and their evaluation metrics judge reliability within simple measurement frameworks, and
cannot easily absorb the contextual factors of complex medical decisions; even where those factors can
be quantified, the methods do not deliver every aspect of reliability that clinical use needs. The
paper sets out these shortcomings and proposes research priorities
([UQ is unsatisfactory for clinical use](https://ieeexplore.ieee.org/document/11730012)). A parallel
argument outside machine learning proper treats mixed-integer linear programming (MILP) decision
engines: a nominally optimal plan can lose feasibility or jump to a qualitatively different solution
under small perturbations of costs, demands or resources. The paper proposes a "post-solve robustness"
layer that audits a solved incumbent, built on two formal objects: an ε-near-optimal feasible
neighbourhood in parameter space, and solution smoothness in decision space. It ends with a reporting
template and evaluation protocol ([post-solve robustness](https://ieeexplore.ieee.org/document/11728950)).

*The second thread is about models that must change after deployment.* Edge AI is usually framed as
compression for tight budgets; the counter-thesis is that realistic edge AI is necessarily adaptive.
Over long horizons a fixed configuration must either violate time-varying budgets (latency, energy,
thermal, connectivity, privacy) or lose predictive reliability, especially calibration, with risk
concentrated in transient regimes rather than in averages. The paper proposes an
Agent–System–Environment (ASE) lens and ten research challenges
([adaptive edge AI](https://ieeexplore.ieee.org/document/11729144)). A related call asks that models
compressed by pruning, quantisation or distillation be evaluated for robustness (uncertainty,
adversarial resilience, sensitivity) before release, and that the compressed models themselves be
shared, not only the techniques ([robust, shared compressed models](https://ieeexplore.ieee.org/document/11729170)).

*The third thread is about labels that arrive late and unevenly.* In semi-supervised online learning
from data streams, class imbalance is made worse by missing labels, because unlabelled examples can
propagate the bias towards the majority class. The paper argues that this makes imbalance a more
serious problem than in supervised stream learning and that little work addresses it
([imbalance in semi-supervised streams](https://ieeexplore.ieee.org/document/11729505)).

*Taken together*, the group asks for reliability under change to be specified (what drifts, what is
observed, what can be reconfigured) and measured over the deployment lifetime. Still open: none of the
abstracts says how to validate calibration or feasibility guarantees when the labels or ground truth
needed to check them are themselves scarce.

**Papers**

- [Uncertainty Quantification in Deep Learning Is Unsatisfactory for Clinical Applications and Complex Decision Making](https://ieeexplore.ieee.org/document/11730012)
- [Post-Solve Robustness in Decision Engines: Feasible Regions and Smoothness Under Perturbations](https://ieeexplore.ieee.org/document/11728950)
- [From Edge AI to Adaptive Edge AI](https://ieeexplore.ieee.org/document/11729144)
- [Compressed Models Should Be Robust and Publicly Shared: A Call for Responsible Model Optimization](https://ieeexplore.ieee.org/document/11729170)
- [The Significance of Class Imbalance in Online Semi-Supervised Data Stream Learning](https://ieeexplore.ieee.org/document/11729505)

### New foundations for neural computation

**The ideas behind this group** (a synthesis from the abstracts of 2 papers).

*The shared problem is that standard neural networks are built on one kind of space and one kind of
learning rule, and both papers argue for widening them.*

One paper proposes two necessary and sufficient conditions for neural computing: a Hilbert space, and
non-linear transformations on it. It then extends deep learning from the Euclidean space of real
vectors to a Hilbert space built from Generalised Intervals' Numbers (GINs), whose convex cone of
Intervals' Numbers (INs) can represent probability or possibility distributions as well as real
numbers. Because these form a lattice whose partial order carries semantics, logic and reasoning can be
applied inside the network, which the paper presents as a route to explainable AI
([Hilbert lattices](https://ieeexplore.ieee.org/document/11728911)). The other paper argues that
artificial networks lack the adaptivity and long-term stability of brains because they use uniform,
local, gradient-based updates. It proposes neuromodulation by neurotransmitters as a third axis of
learning, alongside neural activity and synaptic plasticity, especially for continual learning
([neurotransmitters](https://ieeexplore.ieee.org/document/11729214)).

*Taken together*, both papers want structure (order and logic, or modulatory signals) that gradient
descent on real-valued weights does not provide. Still open: neither abstract shows a task on which the
extended framework does something a standard network cannot.

**Papers**

- [Deep Learning in Hilbert Lattices](https://ieeexplore.ieee.org/document/11728911)
- [Neurotransmitters as a Missing Dimension in Artificial Neural Networks](https://ieeexplore.ieee.org/document/11729214)

### Large language models

**The ideas behind this group** (a synthesis from the abstracts of 2 papers).

*The shared problem is how to make LLMs dependable components, as sources of facts and as controllers.*
One paper argues that LLMs reproduce misinformation by learning the persuasive patterns of falsehoods
(hedging, false presuppositions, fabricated citations), not just false facts. It proposes "model
immunisation": supervised fine-tuning with small "vaccine doses" of (false claim, correction) pairs,
5–10% of tokens, alongside truthful data. Across four open-weight model families this improves
TruthfulQA accuracy by 12 points and misinformation rejection by 30 points with negligible capability
loss, and the paper calls for standard vaccine corpora and benchmarks
([model immunisation](https://ieeexplore.ieee.org/document/11728502)). The other paper surveys LLMs as
the "cognitive core" of robots, integrating multimodal perception, commonsense reasoning and action
generation through unified semantic interfaces and metacognitive coordination
([LLMs as robot cognitive core](https://ieeexplore.ieee.org/document/11728577)).

*Taken together*, one paper treats the LLM as something to be inoculated and the other as something
to build around. Still open: whether immunisation generalises to falsehoods unlike the training doses,
which the first paper itself raises.

**Papers**

- [Just as Humans Need Vaccines, So Do Models - Immunization to Combat Falsehoods](https://ieeexplore.ieee.org/document/11728502)
- [A New Paradigm for Robot Multimodal Understanding and Decision-Making with Large Language Models as the Cognitive Core](https://ieeexplore.ieee.org/document/11728577)

### Research practice and AI for science

**The ideas behind this group** (a synthesis from the abstracts of 2 papers).

*The shared problem is that technical skill alone does not make AI work succeed.* One paper lists the
main risks in AI projects (data availability, budgets and deadlines, stakeholders' misunderstanding of
how AI works) and argues that knowledge of innovation, scientific method and project management must
sit beside knowledge of algorithms ([risks in AI projects](https://ieeexplore.ieee.org/document/11728545)).
The other looks at AI for science: AI4Science publications in 145 Nature Index journals have grown
fifteen-fold in a decade but remain under 3% of the total, and nearly 95% of AI-driven research there
is led by experimental scientists. Using Diffusion of Innovation theory it projects the share rising
from 2.72% in 2024 to about 20% by 2050, and proposes pathways for AI researchers to lead discovery
([AI researchers in science](https://ieeexplore.ieee.org/document/11729527)).

**Papers**

- [Minimizing Risks in Artificial Intelligence Projects](https://ieeexplore.ieee.org/document/11728545)
- [Unlocking the Potential of AI Researchers in Scientific Discovery—What Is Missing?](https://ieeexplore.ieee.org/document/11729527)

### Companion papers at CEC and FUZZ-IEEE

Two more "Position Paper:" titles appeared in the other WCCI 2026 proceedings, apparently through the
same shared track.

- [Suggestions from Large Language Models about Experimental Settings for Benchmarking Evolutionary Multi-objective Optimization Algorithms](https://ieeexplore.ieee.org/document/11727422)
  (CEC 2026): repeats a CEC 2025 study with the same prompts, asking two LLMs which algorithms, test
  problems and performance indicators to use for evolutionary multi-objective optimisation (EMO)
  benchmarks. Suggestions are mostly consistent with common practice but show variations and
  inaccuracies, so the authors stress established benchmarking guidelines.
- [Neuro-Fuzzy Fusion for Dataset-Agnostic Deepfake Detection](https://ieeexplore.ieee.org/document/11626363)
  (FUZZ-IEEE 2026): argues that neuro-fuzzy systems can fuse heterogeneous deepfake detectors
  interpretably, and supports this with a frequency-based ensemble fused by an Adaptive Neuro-Fuzzy
  Inference System (ANFIS), reporting a mean AUC of 0.8705 and mean F1 of 0.9312 with linguistic rules.

## Outlier papers

- [Deep Learning in Hilbert Lattices](https://ieeexplore.ieee.org/document/11728911): the only
  single-author mathematical-foundations paper, proposing a different number system (lattices of
  interval numbers) for neural computing rather than a new architecture.
- [Neurotransmitters as a Missing Dimension](https://ieeexplore.ieee.org/document/11729214): the only
  paper arguing from neuroscience for a new learning signal, in keeping with IJCNN's brain-inspired
  tradition.
- [Post-Solve Robustness in Decision Engines](https://ieeexplore.ieee.org/document/11728950): about
  operations research solvers rather than neural networks, treating learning-enabled decision systems
  as one case.
- [Minimizing Risks in AI Projects](https://ieeexplore.ieee.org/document/11728545): a project-management
  position, closer to engineering practice than to research.
- [Neuro-Fuzzy Fusion for Deepfake Detection](https://ieeexplore.ieee.org/document/11626363): a
  "position" supported mainly by a new model and its scores, the kind of paper the call warned against.

## Theory picks

The positions with the most mathematical content to work through (from abstracts only).

1. [Post-Solve Robustness in Decision Engines](https://ieeexplore.ieee.org/document/11728950): defines
   an ε-near-optimal feasible neighbourhood of a solved incumbent in parameter space, and solution
   smoothness in decision space. Exercise: for a linear programme min cᵀx subject to Ax = b, x ≥ 0 with
   optimal basis B, write the region of cost vectors c for which B stays optimal (reduced costs
   non-negative) and the region of right-hand sides b for which it stays feasible (B⁻¹b ≥ 0); both are
   polyhedra. Then see why integrality breaks this picture for MILP, where small changes can move the
   optimum to a different combinatorial solution, which is the gap the paper wants audited.
2. [Deep Learning in Hilbert Lattices](https://ieeexplore.ieee.org/document/11728911): claims that a
   Hilbert space plus non-linear maps is necessary and sufficient for neural computing, and builds such a
   space from equivalence classes of generalised interval numbers ordered as a lattice. Worth checking
   what "necessary and sufficient" is relative to (which class of computations), and how an inner
   product is defined on interval-valued objects so that the space is complete.
3. [UQ Is Unsatisfactory for Clinical Applications](https://ieeexplore.ieee.org/document/11730012):
   argues that UQ metrics evaluate reliability in frameworks too simple for clinical decisions. A useful
   exercise is to compare a calibration metric such as expected calibration error with a decision-
   theoretic one: for a binary treat/do-not-treat decision with asymmetric costs, show that two models
   with the same calibration error can have different expected cost, because the decision depends on the
   probability only near the cost-ratio threshold.
4. [From Edge AI to Adaptive Edge AI](https://ieeexplore.ieee.org/document/11729144): the claim that a
   fixed configuration must either violate a time-varying budget or lose reliability is a dilemma about
   non-stationary processes. Worth formalising: with a constraint budget B(t) and a data distribution
   P_t that both drift, state when a single fixed model can satisfy cost ≤ B(t) and a calibration bound
   for all t, and why the failure concentrates in transients.
5. [Class Imbalance in Semi-Supervised Streams](https://ieeexplore.ieee.org/document/11729505): the
   claim that unlabelled data amplify majority bias can be checked with a toy self-training model: if a
   classifier's pseudo-labels have minority recall r < 1, each round of self-training on its own
   pseudo-labels shrinks the effective minority prior, and the shrinkage compounds over time.

## Relevance to medical imaging

- **Uncertainty for clinical decisions**: the [UQ position](https://ieeexplore.ieee.org/document/11730012)
  is the most directly relevant paper in the track. It complements the decision-theoretic defence of the
  aleatoric–epistemic split in the [NeurIPS 2026 overview](../neurips2026-position-overview/index.html)
  and the uncertainty papers in the [ICML 2026 overview](../icml2026-position-overview/index.html).
- **Imbalance with scarce labels**: rare findings in screening streams (for example in ultrasound or
  pathology) are exactly the setting of the [semi-supervised stream paper](https://ieeexplore.ieee.org/document/11729505);
  see also the [class imbalance overview](../class-imbalance-overview/index.html) in this repo.
- **Compressed and edge models**: point-of-care ultrasound and other portable devices run compressed
  models under drift. The [compression](https://ieeexplore.ieee.org/document/11729170) and
  [adaptive edge](https://ieeexplore.ieee.org/document/11729144) papers argue that calibration and
  robustness must be re-checked after compression and over the device's lifetime.

## Sources

- *2026 International Joint Conference on Neural Networks (IJCNN)*, IEEE Xplore; papers found by
  searching 2026 titles beginning "Position Paper:".
- [IJCNN 2026 Position Papers call](https://attend.ieee.org/wcci-2026/ijcnn-2026-position-papers)
  and [WCCI 2026 Call for Position Papers](https://attend.ieee.org/wcci-2026/?p=1467).
