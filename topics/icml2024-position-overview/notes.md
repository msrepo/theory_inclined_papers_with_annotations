---
title: "ICML 2024 Position Paper Track: topic map, outliers and theory picks"
authors: "Topic notes"
venue: "ICML 2024 (Position Paper Track)"
tags: [icml, position-papers, conference-overview, learning-theory, evaluation, alignment, ai-policy]
status: living
category: "Research practice & policy"
subcategory: "Conference overviews"
short_title: "ICML 2024 — position papers"
summary: "a map of the 75 papers in the first ICML Position Paper Track (2024, PMLR v235) in seven groups of arguments, with an idea-level synthesis of each, the outliers, the positions with the most mathematical content, and their relevance to medical imaging"
related: [icml2025-position-overview, icml2026-position-overview, neurips2025-position-overview]
---

## Why this page exists

ICML 2024 (Vienna, 21–27 July 2024) ran its **first Position Paper Track**. Position papers argue for a claim about the field rather than presenting a new method.
They are published in the main proceedings, [PMLR volume 235](https://proceedings.mlr.press/v235/),
where every title begins "Position:".

The track has **75 papers: 15 orals, 11 spotlights and 49 posters**, identified from the PMLR
titles and matched one-to-one to their decisions in the ICML 2024 schedule data. Titles below drop
the "Position:" prefix. The grouping and syntheses are from **titles and abstracts only**; each
paper is listed in one group, though the syntheses mention related papers from other groups.

Companion pages: the [ICML 2025](../icml2025-position-overview/index.html), [ICML 2026](../icml2026-position-overview/index.html) and
[NeurIPS 2025](../neurips2025-position-overview/index.html) position tracks.

## Topic groups at a glance

| Group | Papers | What's in it |
|---|---|---|
| [Theory and mathematical foundations](#theory-and-mathematical-foundations) | 14 | No free lunch and Kolmogorov complexity, non-identifiability of LLMs, ICL vs gradient descent, Platonic representations, Rashomon sets, category theory, C*-algebras, topology, Bayesian DL |
| [Research practice, evaluation and the science of ML](#research-practice-evaluation-and-the-science-of-ml) | 13 | Exploratory vs confirmatory research, unscientific AGI claims, negative results, benchmarking limits, measuring dataset diversity, ML in the natural sciences |
| [Foundation models, agents and open-endedness](#foundation-models-agents-and-open-endedness) | 13 | Open-endedness and levels of AGI, LLM-Modulo planning, compositional generative models, foundation agents, video as language, environment shaping |
| [Data modalities and application domains](#data-modalities-and-application-domains) | 7 | Relational, graph, tabular and satellite data as their own modalities, fusion energy, RL for treatment regimes, decentralised training |
| [Trustworthiness: explanation, safety and privacy](#trustworthiness-explanation-safety-and-privacy) | 11 | What explanation is for, context in vision XAI, limits of censorship, AI-text detection, guardrails, DP with public pretraining, a safe harbour for red-teaming |
| [Alignment, social choice and society](#alignment-social-choice-and-society) | 9 | Social choice for RLHF, pluralistic alignment, agency preservation, the reasonable-person standard, randomised allocation, cascading disparities |
| [Open models, data and governance](#open-models-data-and-governance) | 8 | Marginal risk of open models, data provenance, behavioural-use licences, running out of data, technical governance, autonomous weapons |

**Compared with the 2025 tracks:** the first ICML track already has the same backbone (theory of
learning, research practice, governance), but its centre of gravity is different. In 2024 the
debates are about foundation models as a paradigm (open-endedness, levels of AGI, foundation
agents, whether open weights are safe) and about mathematical languages for deep learning (category
theory, C*-algebras, topology, Kolmogorov complexity). By 2025 measurement and evaluation have
become the largest group at both ICML and NeurIPS, and agents are discussed as deployed systems
rather than as a research programme.

## Topic groups

Every paper is listed under its group.

### Theory and mathematical foundations

**The ideas behind this group** (a synthesis from the abstracts of about 14 papers; the papers have not been read in full).

*The shared problem is that deep learning works better than its theory can explain, and the tools used to explain it may be the wrong ones.* The papers in this group argue in two ways. Some say that a familiar theoretical lens, such as no-free-lunch arguments, generalisation bounds or gradient-descent analogies, gives the wrong picture. Others propose a new branch of mathematics as the missing language for the field.

*The first thread revisits what generalisation and inductive bias really mean.* No free lunch theorems are often read as saying that every problem needs a tailored learner. One paper argues instead that real-world data, like neural networks, prefers low Kolmogorov complexity. On this view a few general models can cover many domains, and architectures built for vision can even compress unrelated data ([no free lunch](https://proceedings.mlr.press/v235/goldblum24a.html), spotlight). Statistical generalisation is not enough to explain large language models (LLMs). Autoregressive models are non-identifiable: two models whose Kullback–Leibler (KL) divergence is zero or near zero have the same test loss but can behave very differently. Three case studies make the point: rule extrapolation, in-context learning and fine-tunability ([beyond statistical generalisation](https://proceedings.mlr.press/v235/reizinger24a.html), spotlight). A popular explanation of in-context learning (ICL) as implicit gradient descent (GD) does not hold up on LLaMa-7B. ICL and GD react differently to the order of demonstrations, so the equivalence "remains an open hypothesis" ([ICL as gradient descent](https://proceedings.mlr.press/v235/shen24d.html), oral). Two papers look at what many models share. Representations are said to be converging across architectures and modalities towards a shared statistical model of reality ([Platonic representation](https://proceedings.mlr.press/v235/huh24a.html), oral). For noisy tabular data, the Rashomon effect, the existence of many equally good models, is presented as mostly an opportunity: it makes room for simple models and for fairness constraints, and it shapes how uncertainty and variable importance should be reported ([Rashomon effect](https://proceedings.mlr.press/v235/rudin24a.html), spotlight).

*The second thread proposes new mathematical languages.* Category theory, specifically monads valued in a 2-category of parametric maps, is put forward as one algebraic theory that covers both the constraints of geometric deep learning and concrete architectures such as recurrent neural networks (RNNs) ([categorical deep learning](https://proceedings.mlr.press/v235/gavranovic24a.html)). C*-algebras generalise the complex numbers and are proposed as a way to unify kernel methods and neural networks ([C*-algebraic ML](https://proceedings.mlr.press/v235/hashimoto24a.html)). Topological deep learning (TDL) is offered as the next step beyond graph and geometric learning ([topological deep learning](https://proceedings.mlr.press/v235/papamarkou24a.html)). Tensor networks are argued to support Green AI because of their logarithmic compression potential ([tensor networks](https://proceedings.mlr.press/v235/memmel24a.html)). In scientific machine learning (SciML), optimisation should "first optimize, then discretize", deriving algorithms in function space ([function-space optimisation](https://proceedings.mlr.press/v235/muller24d.html)). Theory for graph neural networks (GNNs) should move beyond combinatorial expressivity and study expressivity, generalisation and optimisation together ([graph ML theory](https://proceedings.mlr.press/v235/morris24a.html)).

*The third thread is statistical and inferential.* Bayesian deep learning (BDL) is defended as necessary for uncertainty, active and continual learning, and scientific data, including when combined with foundation models ([Bayesian deep learning](https://proceedings.mlr.press/v235/papamarkou24b.html)). Quantifying realism is said to be harder than generating it. Drawing on algorithmic information theory, one paper proposes a "universal critic" as a north star for realism metrics ([what makes an image realistic](https://proceedings.mlr.press/v235/theis24a.html), spotlight). Causal methods, another paper argues, are stuck between scientific perfectionism and computer science's taste for assumption-free algorithms, and need "scientific pragmatism" to escape ([causal pragmatism](https://proceedings.mlr.press/v235/loftus24a.html)).

*Taken together*, the group treats the theory of deep learning as something to be rebuilt around simplicity bias, identifiability and structure, not just test loss. Still open: whether the new algebraic, topological and operator languages will produce predictions that can be tested, or will mainly re-describe architectures that already exist.

**Papers**

- [The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning](https://proceedings.mlr.press/v235/goldblum24a.html) (spotlight)
- [Categorical Deep Learning is an Algebraic Theory of All Architectures](https://proceedings.mlr.press/v235/gavranovic24a.html)
- [$C^*$-Algebraic Machine Learning $-$ Moving in a New Direction](https://proceedings.mlr.press/v235/hashimoto24a.html)
- [The Platonic Representation Hypothesis](https://proceedings.mlr.press/v235/huh24a.html) (oral)
- [Do pretrained Transformers Learn In-Context by Gradient Descent?](https://proceedings.mlr.press/v235/shen24d.html) (oral)
- [Understanding LLMs Requires More Than Statistical Generalization](https://proceedings.mlr.press/v235/reizinger24a.html) (spotlight)
- [Amazing Things Come From Having Many Good Models](https://proceedings.mlr.press/v235/rudin24a.html) (spotlight)
- [Optimization in SciML Should Employ the Function Space Geometry](https://proceedings.mlr.press/v235/muller24d.html)
- [Future Directions in the Theory of Graph Machine Learning](https://proceedings.mlr.press/v235/morris24a.html)
- [Topological Deep Learning is the New Frontier for Relational Learning](https://proceedings.mlr.press/v235/papamarkou24a.html)
- [Bayesian Deep Learning is Needed in the Age of Large-Scale AI](https://proceedings.mlr.press/v235/papamarkou24b.html)
- [Tensor Networks are a Valuable Asset for Green AI](https://proceedings.mlr.press/v235/memmel24a.html)
- [What makes an image realistic?](https://proceedings.mlr.press/v235/theis24a.html) (spotlight)
- [The Causal Revolution Needs Scientific Pragmatism](https://proceedings.mlr.press/v235/loftus24a.html)

### Research practice, evaluation and the science of ML

**The ideas behind this group** (a synthesis from the abstracts of about 13 papers; the papers have not been read in full).

*The shared problem is that machine learning (ML) research presents its findings as more confirmed than its methods allow.* Benchmark wins, striking claims about large language models (LLMs) and loosely defined dataset properties all travel further than the evidence behind them. This group asks what kind of science ML is, and how its incentives and reporting should change.

*The first thread is epistemological.* Most empirical ML research is written up as confirmatory when it should be treated as exploratory, and this mismatch produces results that do not replicate ([rethink empirical research](https://proceedings.mlr.press/v235/herrmann24b.html)). Probing random projections, matrix decompositions, autoencoders and transformers shows that all of them encode variables that can be decoded from their latent spaces. Such patterns are therefore no evidence of artificial general intelligence (AGI), and the paper links claims otherwise to the human tendency to anthropomorphise ([unscientific AGI claims](https://proceedings.mlr.press/v235/altmeyer24a.html)). A related paper offers a definition of LLMs and re-examines five common claims about them, including "emergent properties" ([long tail of footnotes](https://proceedings.mlr.press/v235/rogers24a.html)). From physics, one paper argues that ML's data-only ontology and held-out-performance epistemology conflict with the natural sciences. It finds ML useful for modelling nuisance confounders, but harmful when emulators amplify confirmation bias or when ML-generated labels feed into downstream analyses ([ML for the natural sciences](https://proceedings.mlr.press/v235/hogg24a.html)).

*The second thread shows how benchmarking misleads, field by field.* In reinforcement learning (RL), rigorous benchmarking is shown to have often prohibitive computational cost, so the paper calls for an additional experimental paradigm ([RL benchmarking](https://proceedings.mlr.press/v235/jordan24a.html)). For large traveling salesman problems (TSP), a simple baseline beats ML-generated heatmaps, and heatmap-guided Monte Carlo tree search (MCTS) loses to the LKH-3 heuristic ([TSP heatmaps](https://proceedings.mlr.press/v235/xia24f.html), oral). In time series anomaly detection (TAD), flawed metrics hide the fact that state-of-the-art deep detectors effectively learn linear mappings ([time series anomaly detection](https://proceedings.mlr.press/v235/sarfraz24a.html)). Dataset "diversity", studied across 135 image and text datasets, is claimed but seldom defined. The paper applies measurement theory to how it should be conceptualised, operationalised and validated ([measure diversity](https://proceedings.mlr.press/v235/zhao24a.html), oral). Survey methodology, the science of data collection, is offered as a way to improve training and feedback data ([survey methodology](https://proceedings.mlr.press/v235/eckman24a.html)).

*The third thread is about incentives and community structure.* If predictive performance is the main criterion for publication, research is wasted and incentives are skewed, so negative results should become normal to publish ([negative results](https://proceedings.mlr.press/v235/karl24a.html), oral). Application-driven research is undervalued by reviewing, hiring and teaching, even though it can drive methodological innovation ([application-driven ML](https://proceedings.mlr.press/v235/rolnick24a.html)). Automated machine learning (AutoML) has optimised predictive performance at the cost of its original goals, and should become human-centred ([human-centred AutoML](https://proceedings.mlr.press/v235/lindauer24a.html)). A causal analysis of over 8,000 papers shared on social media, against controls matched on 9 covariates, finds median citation counts 2-3 times higher for papers that influencers promoted. The authors ask influencers to curate responsibly ([AI/ML influencers](https://proceedings.mlr.press/v235/weissburg24a.html)).

*Taken together*, the group asks ML to state what its experiments can and cannot show, to measure the properties it claims, and to reward work that does not top a leaderboard. Still open: who pays for the rigour that, on the RL paper's own evidence, is often computationally out of reach, and how review processes can tell exploratory findings from confirmatory ones.

**Papers**

- [Stop Making Unscientific AGI Performance Claims](https://proceedings.mlr.press/v235/altmeyer24a.html)
- [Why We Must Rethink Empirical Research in Machine Learning](https://proceedings.mlr.press/v235/herrmann24b.html)
- [Embracing Negative Results in Machine Learning](https://proceedings.mlr.press/v235/karl24a.html) (oral)
- [Benchmarking is Limited in Reinforcement Learning Research](https://proceedings.mlr.press/v235/jordan24a.html)
- [Key Claims in LLM Research Have a Long Tail of Footnotes](https://proceedings.mlr.press/v235/rogers24a.html)
- [Measure Dataset Diversity, Don't Just Claim It](https://proceedings.mlr.press/v235/zhao24a.html) (oral)
- [AI/ML Influencers Have a Place in the Academic Process](https://proceedings.mlr.press/v235/weissburg24a.html)
- [Insights from Survey Methodology can Improve Training Data](https://proceedings.mlr.press/v235/eckman24a.html)
- [Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale Traveling Salesman Problems](https://proceedings.mlr.press/v235/xia24f.html) (oral)
- [Quo Vadis, Unsupervised Time Series Anomaly Detection?](https://proceedings.mlr.press/v235/sarfraz24a.html)
- [A Call to Action for a Human-Centered AutoML Paradigm](https://proceedings.mlr.press/v235/lindauer24a.html)
- [Application-Driven Innovation in Machine Learning](https://proceedings.mlr.press/v235/rolnick24a.html)
- [Is machine learning good or bad for the natural sciences?](https://proceedings.mlr.press/v235/hogg24a.html)

### Foundation models, agents and open-endedness

**The ideas behind this group** (a synthesis from the abstracts of about 13 papers; the papers have not been read in full).

*The shared problem is what foundation models should become once next-token prediction has been shown to scale.* This group is the most programmatic in the track. Most papers set out a roadmap: for agents, for generalist intelligence or for using pretrained models in new domains. They disagree about whether a large model is the system itself or only one component of it.

*The first thread defines the destination.* Open-endedness is given a formal definition in terms of novelty and learnability with respect to a human observer. The paper argues that it is an essential property of any artificial superhuman intelligence (ASI) and that the ingredients now exist ([open-endedness](https://proceedings.mlr.press/v235/hughes24a.html), oral). A second paper distils six principles from existing definitions of artificial general intelligence (AGI). It then proposes "Levels of AGI" based on performance (depth) and generality (breadth), with autonomy treated separately as a deployment consideration ([levels of AGI](https://proceedings.mlr.press/v235/morris24b.html), spotlight). Embodied AI is offered as the next step towards AGI, using a cognitive architecture aligned with Friston's active inference principle ([embodied AI](https://proceedings.mlr.press/v235/paolo24a.html)).

*The second thread argues against monoliths.* Autoregressive large language models (LLMs) cannot plan or verify their own outputs, but they can act as approximate knowledge sources inside "LLM-Modulo" frameworks with external model-based verifiers ([LLM-Modulo](https://proceedings.mlr.press/v235/kambhampati24a.html), spotlight). Composing smaller generative models is argued to be more data-efficient and to generalise to unseen parts of the distribution ([compositional generative modelling](https://proceedings.mlr.press/v235/du24d.html)). Agents should be aligned at the same time with human intentions, environment dynamics and self-constraints such as budget. A retrofitted WebShop benchmark illustrates the point ([unified alignment for agents](https://proceedings.mlr.press/v235/yang24p.html)). Fully automated data-driven discovery with large generative models is difficult. Using DataVoyager, a GPT-4 proof of concept, one paper argues for tool integration and user moderation ([data-driven discovery](https://proceedings.mlr.press/v235/majumder24a.html)).

*The third thread carries the foundation-model recipe into new settings.* Foundation agents pretrained on large interactive data are proposed as a paradigm shift for decision making ([foundation agents](https://proceedings.mlr.press/v235/liu24aq.html)). Video generation could serve as a planner, agent and environment simulator, as language models already do ([video as language](https://proceedings.mlr.press/v235/yang24z.html)). Black-box optimisation and experimental design should be reframed around sequence-based foundation models ([foundation models for optimisation](https://proceedings.mlr.press/v235/song24h.html)). LLMs could also move time series analysis beyond domain-specific prediction ([LLMs for time series](https://proceedings.mlr.press/v235/jin24i.html)). Robotics pushes back. In sim-to-real reinforcement learning (RL), the bottleneck is shaping the environment (observations, actions, rewards and dynamics), not policy optimisation, so environment shaping should be automated ([environment shaping](https://proceedings.mlr.press/v235/park24i.html), oral). Scaling simulation is argued to be neither necessary nor sufficient for real-world manipulation ([against scaling simulation](https://proceedings.mlr.press/v235/bharadhwaj24a.html)).

*Taken together*, the group agrees that foundation models are a starting point and that their value depends on what surrounds them: verifiers, compositional structure, interaction data or real environments. Still open: whether open-endedness and AGI levels can be measured rigorously enough to guide research, given the concerns about [unscientific AGI claims](https://proceedings.mlr.press/v235/altmeyer24a.html) raised in the research-practice group.

**Papers**

- [Open-Endedness is Essential for Artificial Superhuman Intelligence](https://proceedings.mlr.press/v235/hughes24a.html) (oral)
- [LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks](https://proceedings.mlr.press/v235/kambhampati24a.html) (spotlight)
- [Foundation Agents as the Paradigm Shift for Decision Making](https://proceedings.mlr.press/v235/liu24aq.html)
- [Video as the New Language for Real-World Decision Making](https://proceedings.mlr.press/v235/yang24z.html)
- [Towards Unified Alignment Between Agents, Humans, and Environment](https://proceedings.mlr.press/v235/yang24p.html)
- [Compositional Generative Modeling: A Single Model is Not All You Need](https://proceedings.mlr.press/v235/du24d.html)
- [Automatic Environment Shaping is the Next Frontier in RL](https://proceedings.mlr.press/v235/park24i.html) (oral)
- [Levels of AGI for Operationalizing Progress on the Path to AGI](https://proceedings.mlr.press/v235/morris24b.html) (spotlight)
- [A Call for Embodied AI](https://proceedings.mlr.press/v235/paolo24a.html)
- [Scaling Simulation is Neither Necessary Nor Sufficient for In-the-Wild Robot Manipulation](https://proceedings.mlr.press/v235/bharadhwaj24a.html)
- [Leverage Foundational Models for Black-Box Optimization](https://proceedings.mlr.press/v235/song24h.html)
- [Data-driven Discovery with Large Generative Models](https://proceedings.mlr.press/v235/majumder24a.html)
- [What Can Large Language Models Tell Us about Time Series Analysis](https://proceedings.mlr.press/v235/jin24i.html)

### Data modalities and application domains

**The ideas behind this group** (a synthesis from the abstracts of about 7 papers; the papers have not been read in full).

*The shared problem is that machine learning (ML) methods built for images and text are being applied to data whose structure they ignore.* Most valuable data sits in tables, graphs, satellite archives and clinical records, not in web text. These papers argue that each kind of data needs models and evaluation designed for its own structure. Without that, applications stay on the margins of the field.

*The first thread argues that structured data deserves its own foundation-scale models.* Relational databases are tables linked by primary–foreign keys, and no current algorithm learns from several connected tables directly. Practitioners therefore fall back on manual joins and feature engineering, which is slow and error-prone. Relational Deep Learning (RDL) instead represents the database as a temporal, heterogeneous graph with one node per row and edges along key links, and lets graph neural networks (GNNs) learn end to end. The paper also introduces the RelBench benchmark ([relational deep learning](https://proceedings.mlr.press/v235/fey24a.html)). For graphs in general, the central difficulty is getting positive transfer across very diverse graph data. One paper proposes a "graph vocabulary" of transferable units that encode invariances on graphs, grounded in network analysis, expressiveness and stability, and argues that this fits with neural scaling laws ([graph foundation models](https://proceedings.mlr.press/v235/mao24a.html), spotlight). Tabular data is the dominant modality in many fields but gets little research attention. A third paper calls for Large Tabular Models (LTMs) that read a dataset in the context of related datasets instead of in isolation. It points to few-shot tabular learning, automated data science and out-of-distribution synthetic data as possible gains ([tabular foundation models](https://proceedings.mlr.press/v235/van-breugel24a.html)).

*The second thread argues that some application areas should become research disciplines in their own right.* Satellite data is presented as a distinct modality, not just an application area, and the paper argues that recognising it as such would improve theory, methods and deployment ([satellite data](https://proceedings.mlr.press/v235/rolf24a.html), spotlight). Magnetic confinement fusion offers six problems that are ripe for ML: disruption prediction, simulation and dynamics modelling, partially observed data, control, optimal experimental design and materials discovery. The paper also notes the field's changing data ecosystem ([fusion energy](https://proceedings.mlr.press/v235/spangher24a.html), oral). The closest paper to medical imaging is a critique of offline reinforcement learning (RL) for dynamic treatment regimes (DTRs). In more than 17,000 evaluation experiments on a public sepsis dataset, performance changed substantially with the evaluation metric and the Markov decision process (MDP) formulation. In some cases RL was beaten by random baselines, and the paper notes that naive and supervised baselines are usually missing ([RL for treatment regimes](https://proceedings.mlr.press/v235/luo24f.html), spotlight).

*The third thread concerns infrastructure, not data.* Decentralised training based on pipeline parallelism is meant to make large-model training accessible to more people, but a malicious pipeline stage can poison it. The paper demonstrates such an attack and proposes a robust framework with a 100% detection strategy ([decentralised training](https://proceedings.mlr.press/v235/lu24c.html)).

*Taken together*, the group argues that a data type's structure (relational keys, graph invariances, sensor physics, sequential clinical decisions) should determine the inductive bias and the evaluation protocol, rather than being forced into image or text pipelines. Still open: whether the transferable units these papers propose, such as graph vocabularies and cross-dataset tabular context, actually exist and can be identified, and how a domain community should agree on evaluation choices like the MDP formulations that the sepsis study shows can reverse conclusions.

**Papers**

- [Relational Deep Learning - Graph Representation Learning on Relational Databases](https://proceedings.mlr.press/v235/fey24a.html)
- [Graph Foundation Models Are Already Here](https://proceedings.mlr.press/v235/mao24a.html) (spotlight)
- [Why Tabular Foundation Models Should Be a Research Priority](https://proceedings.mlr.press/v235/van-breugel24a.html)
- [Mission Critical – Satellite Data is a Distinct Modality in Machine Learning](https://proceedings.mlr.press/v235/rolf24a.html) (spotlight)
- [Opportunities Exist for Machine Learning in Magnetic Fusion Energy](https://proceedings.mlr.press/v235/spangher24a.html) (oral)
- [Reinforcement Learning in Dynamic Treatment Regimes Needs Critical Reexamination](https://proceedings.mlr.press/v235/luo24f.html) (spotlight)
- [Exploring the Robustness of Pipeline-Parallelism-Based Decentralized Training](https://proceedings.mlr.press/v235/lu24c.html)

### Trustworthiness: explanation, safety and privacy

**The ideas behind this group** (a synthesis from the abstracts of about 11 papers; the papers have not been read in full).

*The shared problem is that the tools meant to make models trustworthy (explanations, filters, detectors, privacy guarantees) promise more than they can deliver.* Several papers in this group give formal or empirical arguments that a safeguard has a hard limit. Others say what explanation should be for in the first place.

*The first thread asks what explanation is for.* Explainable artificial intelligence (XAI) is said to mix two cultures. "BLUE XAI" serves human values, while "RED XAI" exists to validate models: extracting knowledge from good models and finding bugs in faulty ones. The paper argues that RED XAI is under-explored and matters for safety ([explain to question](https://proceedings.mlr.press/v235/biecek24a.html)). Explanations of vision models ignore context. Whether a stethoscope makes the person next to it a doctor or a patient depends on how the two relate, and the paper argues that saliency should move from asking *where* to asking *how* ([context in vision XAI](https://proceedings.mlr.press/v235/tomaszewska24a.html)). Inner (mechanistic) interpretability faces critiques that cognitive neuroscience has already dealt with. One paper borrows that field's conceptual framework and methods for building mechanistic explanations ([inner interpretability](https://proceedings.mlr.press/v235/vilas24a.html)).

*The second thread gives impossibility and possibility results for safeguards.* Censoring large language model (LLM) outputs on semantic grounds is argued to be undecidable. Semantic properties cannot be verified if the model can produce "encrypted" outputs, and attackers can rebuild impermissible content from pieces that are each permissible. The paper therefore suggests syntactic censorship, which can come with guarantees ([limits of censorship](https://proceedings.mlr.press/v235/glukhov24a.html)). Detecting AI-generated text gets the opposite verdict. An information-theoretic argument says detection stays possible unless the human and machine text distributions are indistinguishable over their entire support, and sample-complexity bounds say how much text (sample size or sequence length) a reliable detector needs ([AI-text detection](https://proceedings.mlr.press/v235/chakraborty24a.html)). On the engineering side, one paper asks for guardrails built through systematic, socio-technical design with neural-symbolic implementation and verification, instead of the current open-source filters ([guardrails](https://proceedings.mlr.press/v235/dong24c.html)). TrustLLM proposes principles across eight dimensions and benchmarks 16 mainstream LLMs on six of them, using over 30 datasets. It finds that trustworthiness and capability are generally positively related, and that some models are so cautious that they refuse benign prompts ([TrustLLM](https://proceedings.mlr.press/v235/huang24x.html)). Implicit prompts, which hint at a target without naming it, get around text-to-image (T2I) safety constraints. The ImplicitBench benchmark of more than 2,000 such prompts shows privacy leakage and bypassed not-safe-for-work (NSFW) filters ([implicit prompts](https://proceedings.mlr.press/v235/yang24o.html)).

*The third thread is about privacy and access to models.* Differentially private (DP) learning gets much better results when it builds on models pretrained on public web data. One paper questions whether scraped "public" data should count as privacy-preserving at all, whether current benchmarks reflect sensitive target domains, and whether depending on large models loses a different kind of privacy, because the data has to be sent to a third party with more compute ([DP with public pretraining](https://proceedings.mlr.press/v235/tramer24a.html), oral). Independent red-teaming is discouraged by companies' terms of service. Another paper asks developers to commit to a legal and technical safe harbour for good-faith safety research ([safe harbour](https://proceedings.mlr.press/v235/longpre24a.html), oral). The last paper is an outlier. It proposes enforced amnesia, meaning memory resets, as a hedge against the possibility of unobserved suffering in AI that might be conscious ([enforced amnesia](https://proceedings.mlr.press/v235/tkachenko24a.html)).

*Taken together*, the group treats trustworthiness as a set of properties that need their own definitions and, where possible, provable limits: decidability for censorship, sample complexity for detection, and formal privacy accounting for pretraining. Still open: what formal guarantee DP still gives when the "public" pretraining data itself contains sensitive records, and whether explanation methods can be validated against ground truth, the RED XAI aim, rather than judged by how plausible they look.

**Papers**

- [Explain to Question not to Justify](https://proceedings.mlr.press/v235/biecek24a.html)
- [Do Not Explain Vision Models Without Context](https://proceedings.mlr.press/v235/tomaszewska24a.html)
- [An Inner Interpretability Framework for AI Inspired by Lessons from Cognitive Neuroscience](https://proceedings.mlr.press/v235/vilas24a.html)
- [On the Possibilities of AI-Generated Text Detection](https://proceedings.mlr.press/v235/chakraborty24a.html)
- [Fundamental Limitations of LLM Censorship Necessitate New Approaches](https://proceedings.mlr.press/v235/glukhov24a.html)
- [Building Guardrails for Large Language Models Requires Systematic Design](https://proceedings.mlr.press/v235/dong24c.html)
- [TrustLLM: Trustworthiness in Large Language Models](https://proceedings.mlr.press/v235/huang24x.html)
- [Considerations for Differentially Private Learning with Large-Scale Public Pretraining](https://proceedings.mlr.press/v235/tramer24a.html) (oral)
- [A Safe Harbor for AI Evaluation and Red Teaming](https://proceedings.mlr.press/v235/longpre24a.html) (oral)
- [Towards Implicit Prompt For Text-To-Image Models](https://proceedings.mlr.press/v235/yang24o.html)
- [Enforced Amnesia as a Way to Mitigate the Potential Risk of Silent Suffering in the Conscious AI](https://proceedings.mlr.press/v235/tkachenko24a.html)

### Alignment, social choice and society

**The ideas behind this group** (a synthesis from the abstracts of about 9 papers; the papers have not been read in full).

*The shared problem is that "align the model with human preferences" assumes a single, stable human whose preferences can be read off.* In practice people disagree, their intentions can be manipulated, and the decisions involved are collective ones. These papers bring in social choice theory, political philosophy and law to reformulate what alignment and fairness should aim at.

*The first thread treats alignment as aggregation.* Reinforcement learning from human feedback (RLHF) and constitutional AI both take inputs from many people that may conflict. One paper argues that social choice theory, the study of turning individual preferences into collective decisions, is the right framework for deciding what the model should do ([social choice for alignment](https://proceedings.mlr.press/v235/conitzer24a.html)). Pluralistic alignment formalises three targets. *Overton* models present the range of reasonable answers. *Steerable* models can adopt a given perspective. *Distributionally* pluralistic models are calibrated to a population. Matching these are three kinds of benchmark: multi-objective, trade-off steerable and jury-pluralistic. The paper cites evidence that standard alignment may *reduce* distributional pluralism ([pluralistic alignment](https://proceedings.mlr.press/v235/sorensen24a.html)). Social Environment Design extends the aggregation view to policy-making. It is a framework that combines reinforcement learning (RL), economics and computation (EconCS) and computational social choice, with voting on policy objectives inside simulated economies ([social environment design](https://proceedings.mlr.press/v235/zhang24cl.html)).

*The second thread asks what the target of alignment should be.* Aligning a model with a user's intent is not enough when that intent can itself be manipulated. One paper argues for explicitly optimising the preservation of long-term human agency, and gives a formal definition of agency-preserving interaction ([agency preservation](https://proceedings.mlr.press/v235/mitelut24a.html), spotlight). Another contrasts a *mechanistic* view of agency, which AI research usually assumes, with a *volitional* one. On the volitional view AI is not an ethical agent, and the paper argues that ethically significant AI behaviour should be seen as the outcome of political processes ([beyond personhood](https://proceedings.mlr.press/v235/dai24a.html), oral). Law already has a well-tested behavioural standard, the "reasonable person" of American law. One paper proposes it as a source of concrete targets for developing and stress-testing model behaviour ([reasonable person standard](https://proceedings.mlr.press/v235/rane24a.html)).

*The third thread is about distributional justice and collective dynamics.* Disparities in performance, representation, privacy, robustness, interpretability and safety are argued to cascade together in foundation models. Their combined effect falls on marginalised communities, and the paper traces it back through data creation, training and deployment ([cascading disparity](https://proceedings.mlr.press/v235/farnadi24a.html)). In contrast to deterministic notions of fairness, another paper argues that allocating scarce resources fairly with ML often *requires* randomisation, because a lottery can respect each individual's claim in a way a deterministic ranking cannot ([randomised allocation](https://proceedings.mlr.press/v235/jain24a.html)). Finally, interacting LLMs may "program" one another into collectives. A small simulated community suggests that such collectives could widen diversity and reduce toxic behaviour, which motivates AI cross-moderation ([AI collectives](https://proceedings.mlr.press/v235/lai24b.html)).

*Taken together*, the group recasts alignment as a problem of collective choice and institutional design, not of estimating one reward function. It brings in Arrow-type aggregation problems, calibration to a population, and legal and political standards of accountability. Still open: which social choice axioms preference-learning pipelines can satisfy, or are forced to violate, and how to measure "agency preservation" or "reasonableness" well enough to optimise for them.

**Papers**

- [Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback](https://proceedings.mlr.press/v235/conitzer24a.html)
- [A Roadmap to Pluralistic Alignment](https://proceedings.mlr.press/v235/sorensen24a.html)
- [Intent-aligned AI Systems Must Optimize for Agency Preservation](https://proceedings.mlr.press/v235/mitelut24a.html) (spotlight)
- [Evolving AI Collectives Enhance Human Diversity and Enable Self-Regulation](https://proceedings.mlr.press/v235/lai24b.html)
- [Beyond Personhood: Agency, Accountability, and the Limits of Anthropomorphic Ethical Analysis](https://proceedings.mlr.press/v235/dai24a.html) (oral)
- [The Reasonable Person Standard for AI](https://proceedings.mlr.press/v235/rane24a.html)
- [Cracking the Code of Cascading Disparity Towards Marginalized Communities](https://proceedings.mlr.press/v235/farnadi24a.html)
- [Scarce Resource Allocations That Rely On Machine Learning Should Be Randomized](https://proceedings.mlr.press/v235/jain24a.html)
- [Social Environment Design Should be Further Developed for AI-based Policy-Making](https://proceedings.mlr.press/v235/zhang24cl.html)

### Open models, data and governance

**The ideas behind this group** (a synthesis from the abstracts of about 8 papers; the papers have not been read in full).

*The shared problem is that regulators are making rules about AI faster than the evidence and technical tools they would need.* These papers ask what openness actually costs, where training data comes from and whether it will run out, and what governance needs from ML researchers. Four of the eight were orals, which makes this the most decorated group in the track.

*The first thread is the debate about open weights.* Open foundation models have five distinctive properties, such as greater customisability and poor monitoring. One paper proposes judging their misuse risk by its *marginal* risk relative to technologies that already exist. On that basis, current research cannot yet characterise the marginal risk across misuse routes such as cyberattacks and bioweapons ([open foundation models](https://proceedings.mlr.press/v235/kapoor24a.html), oral). A companion paper introduces an openness taxonomy, applies it to 40 current large language models, and argues for responsible open-sourcing in the near and medium term. Its worry is that regulation driven by the leading companies could stifle open-source generative AI ([open-source generative AI](https://proceedings.mlr.press/v235/eiras24b.html), oral).

*The second thread concerns the supply of training data and the rules attached to it.* Practices for tracing data authenticity, consent and provenance are described as broken. On the basis of a large-scale analysis of training data, one paper calls for universal data-provenance standards ([data provenance](https://proceedings.mlr.press/v235/longpre24b.html), spotlight). Responsible AI licences with behavioural-use clauses had been adopted by on the order of 40,000 repositories by the end of 2023. Using interviews, clustering of licence clauses and adoption data, another paper argues for "standardised customisation": common clauses, with room for domain-specific restrictions, for example in medicine ([behavioural-use licences](https://proceedings.mlr.press/v235/mcduff24a.html)). A forecasting paper estimates that if current trends continue, models will be trained on datasets roughly equal to the whole stock of public human text between 2026 and 2032, or slightly earlier if models are overtrained. It names synthetic data, transfer from data-rich domains and better data efficiency as ways forward ([running out of data](https://proceedings.mlr.press/v235/villalobos24a.html)).

*The third thread asks what governance needs from ML.* A survey of policy documents from the EU, US and China finds gaps between what the policies assume is technically possible and the actual state of the art. The paper calls for technical research and technical staff inside governing institutions ([technical AI governance](https://proceedings.mlr.press/v235/reuel24a.html), oral). ML-driven autonomous weapons are presented as a nearer-term risk than superintelligence. Replacing soldiers lowers the political cost of war, could fuel an arms race, and could bring national-security restrictions on AI research ([autonomous weapons](https://proceedings.mlr.press/v235/simmons-edler24a.html), oral). In the other direction, ML could be used to evaluate regulation itself. One paper proposes quantifying the effects of the European Union's (EU) Digital Services Act (DSA) on online harms, which requires solving data access, isolating marginal regulatory effects and generalising across contexts ([evaluating the DSA](https://proceedings.mlr.press/v235/bonel24a.html)).

*Taken together*, the group asks for governance grounded in evidence: counterfactual (marginal) risk instead of absolute risk, documented provenance instead of opaque scraping, and causal estimates of what a policy actually did. Still open: how to identify marginal risk or a regulation's causal effect from observational data with no clean control group, and who should hold the provenance and licensing infrastructure.

**Papers**

- [On the Societal Impact of Open Foundation Models](https://proceedings.mlr.press/v235/kapoor24a.html) (oral)
- [Near to Mid-term Risks and Opportunities of Open-Source Generative AI](https://proceedings.mlr.press/v235/eiras24b.html) (oral)
- [Data Authenticity, Consent, & Provenance for AI are all broken: what will it take to fix them?](https://proceedings.mlr.press/v235/longpre24b.html) (spotlight)
- [Standardization of Behavioral Use Clauses is Necessary for the Adoption of Responsible Licensing of AI](https://proceedings.mlr.press/v235/mcduff24a.html)
- [Will we run out of data? Limits of LLM scaling based on human-generated data](https://proceedings.mlr.press/v235/villalobos24a.html)
- [Technical Research and Talent is Needed for Effective AI Governance](https://proceedings.mlr.press/v235/reuel24a.html) (oral)
- [AI-Powered Autonomous Weapons Risk Geopolitical Instability and Threaten AI Research](https://proceedings.mlr.press/v235/simmons-edler24a.html) (oral)
- [Machine Learning-powered Assessments of the EU Digital Services Act Aid Quantify Policy Impacts on Online Harms](https://proceedings.mlr.press/v235/bonel24a.html)

## Outlier papers

Positions that stand out from the rest of the track in subject, form or reach.

- [Enforced Amnesia as a Way to Mitigate the Potential Risk of Silent Suffering in the Conscious AI](https://proceedings.mlr.press/v235/tkachenko24a.html):
  the only paper about the welfare of possibly conscious systems. It proposes resetting or blocking an AI's
  memory to interrupt suffering, without first having to decide whether the system is conscious at all.
- [Opportunities Exist for Machine Learning in Magnetic Fusion Energy](https://proceedings.mlr.press/v235/spangher24a.html):
  the only paper set in experimental physics. It lists six problems, from disruption prediction to materials
  discovery, that it wants machine learning to take on as research priorities.
- [AI-Powered Autonomous Weapons Risk Geopolitical Instability and Threaten AI Research](https://proceedings.mlr.press/v235/simmons-edler24a.html):
  the only military paper. Its risk needs no superhuman AI: replacing soldiers with machines lowers the
  political cost of going to war, and an arms race could bring national-security restrictions on open research.
- [AI/ML Influencers Have a Place in the Academic Process](https://proceedings.mlr.press/v235/weissburg24a.html):
  a causal study of the field itself. It uses over 8,000 papers, with controls matched on 9 covariates, and
  finds that papers shared by influencers reach median citation counts 2 to 3 times higher.
- [The Reasonable Person Standard for AI](https://proceedings.mlr.press/v235/rane24a.html): takes a
  doctrine from American law, illustrated with cases, and proposes "what a reasonable person would do" as a
  technical target for AI behaviour.
- [Scarce Resource Allocations That Rely On Machine Learning Should Be Randomized](https://proceedings.mlr.press/v235/jain24a.html):
  goes against deterministic notions of algorithmic fairness, arguing that a fair allocation of scarce
  goods often needs a lottery.
- [Beyond Personhood](https://proceedings.mlr.press/v235/dai24a.html): a philosophy and political-science
  argument that AI should be seen not as an ethical agent but as the outcome of political processes.
- [Evolving AI Collectives Enhance Human Diversity and Enable Self-Regulation](https://proceedings.mlr.press/v235/lai24b.html):
  studies "societies" of language models that shape one another's behaviour. Using a small community of
  models, it argues that such collectives could widen human diversity and reduce toxic behaviour online,
  where most of the track treats this as a risk.

## Theory picks

The positions with the most mathematical or statistical content to work through (from abstracts only).

1. [The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases](https://proceedings.mlr.press/v235/goldblum24a.html):
   no-free-lunch theorems average over a uniform distribution on problems, and almost every problem drawn
   that way has high Kolmogorov complexity. Real data, and neural networks (even randomly initialised ones),
   both prefer low-complexity sequences. Work through the counting argument that fewer than a 2^(−k) fraction
   of binary strings of length n can be compressed by k bits. Then see how a prior proportional to
   2^(−K(h)) over hypotheses, where K(h) is the Kolmogorov complexity of h, turns this into an Occam-style
   generalisation bound.
2. [Understanding LLMs Requires More Than Statistical Generalization](https://proceedings.mlr.press/v235/reizinger24a.html):
   autoregressive (AR) probabilistic models are not identifiable. Two models whose Kullback–Leibler (KL)
   divergence is zero or near zero, and which therefore have the same test loss, can still behave very
   differently. A good exercise is to build two next-token distributions that agree wherever the data has
   probability mass and disagree on prefixes the data never produces. Then check that KL(p_data ‖ q) cannot
   tell them apart, which leaves rule extrapolation unconstrained.
3. [On the Possibilities of AI-Generated Text Detection](https://proceedings.mlr.press/v235/chakraborty24a.html):
   human and machine text can be told apart unless the two distributions coincide over their whole support.
   Detection gets harder as the distributions get closer, but more samples or longer sequences restore it,
   with sample complexity bounds to say how much is needed. Derive how the total variation (TV) between the
   n-fold product distributions approaches 1 by tensorising the Hellinger affinity. From that, read off the
   number of samples needed for a given detection error, as a function of the Hellinger distance between the
   two distributions.
4. [Optimization in SciML Should Employ the Function Space Geometry](https://proceedings.mlr.press/v235/muller24d.html):
   for scientific machine learning (SciML), the paper argues "first optimise, then discretise": pick an
   algorithm in function space first, then discretise it. It says recent state-of-the-art SciML optimisers
   can be derived this way. Take a linear model u_θ = Σ θ_i φ_i and a least-squares partial differential
   equation (PDE) residual loss. Check that projecting the function-space Newton step onto span{φ_i} gives
   a Gauss–Newton step in θ, while the plain parameter gradient implicitly uses the Euclidean metric on θ.
5. [What makes an image realistic?](https://proceedings.mlr.press/v235/theis24a.html): measuring realism
   is said to be harder than generating it, and a good generative model alone is not enough. The paper
   offers a "universal critic", built from algorithmic information theory, as a guide. A short exercise
   shows why likelihood fails: for a standard Gaussian in d dimensions the density peaks at the origin, but
   the norm of samples concentrates near √d. Then compare the critic with the randomness deficiency
   −log p(x) − K(x).
6. [Amazing Things Come From Having Many Good Models](https://proceedings.mlr.press/v235/rudin24a.html):
   when the Rashomon set {f : L(f) ≤ L* + ε} is large, it brings simple but accurate models, room for
   constraints such as fairness and monotonicity, and honest uncertainty about variable importance. The
   paper also gives a theory of when this happens, especially for noisy data. For logistic regression,
   approximate the set by the Hessian ellipsoid around θ*. Work out the range of each coefficient,
   θ*_j ± √(2ε (H⁻¹)_jj), and so how unstable a single model's variable importance can be.

Also worth a look: [Do pretrained Transformers Learn In-Context by Gradient Descent?](https://proceedings.mlr.press/v235/shen24d.html)
(tests whether the claimed equivalence between in-context learning (ICL) and gradient descent (GD) holds
in a real pretrained model, finding different sensitivity to the order of demonstrations),
[The Platonic Representation Hypothesis](https://proceedings.mlr.press/v235/huh24a.html) (vision and
language models measure distances between datapoints more alike as they grow, which it reads as convergence
on a shared statistical model of reality),
[Categorical Deep Learning is an Algebraic Theory of All Architectures](https://proceedings.mlr.press/v235/gavranovic24a.html)
(monads valued in a 2-category of parametric maps, recovering the constraints of geometric deep learning
and architectures such as recurrent networks),
[C*-Algebraic Machine Learning](https://proceedings.mlr.press/v235/hashimoto24a.html) (replaces the complex
numbers with a C*-algebra in kernel methods and neural networks),
[Future Directions in the Theory of Graph Machine Learning](https://proceedings.mlr.press/v235/morris24a.html)
(argues that the theory of graph neural networks (GNNs) should study expressive power, generalisation and
optimisation together, not expressivity alone) and
[Fundamental Limitations of LLM Censorship](https://proceedings.mlr.press/v235/glukhov24a.html) (semantic
censorship of large language model (LLM) outputs seen as an undecidable problem, and attackers rebuilding a
forbidden output from permissible pieces).

## Relevance to medical imaging

- **Explanations that use anatomy**: [Do Not Explain Vision Models Without Context](https://proceedings.mlr.press/v235/tomaszewska24a.html)
  asks whether a stethoscope makes the person next to it a doctor or a patient. It argues that saliency
  methods ignore spatial context and should explain *how* features are used, not just *where* they are. In
  ultrasound, a finding often means something only in relation to nearby landmarks.
  [Explain to Question not to Justify](https://proceedings.mlr.press/v235/biecek24a.html) argues for
  explainable AI (XAI) aimed at catching bugs in models, which fits the hunt for shortcut features in
  clinical models.
- **Evaluation and baselines**: [Reinforcement Learning in Dynamic Treatment Regimes Needs Critical Reexamination](https://proceedings.mlr.press/v235/luo24f.html)
  runs more than 17,000 evaluation experiments on a public sepsis dataset. In some settings, random
  baselines beat reinforcement learning (RL) algorithms once policy evaluation and reward design change.
  [Quo Vadis, Unsupervised Time Series Anomaly Detection?](https://proceedings.mlr.press/v235/sarfraz24a.html)
  finds that state-of-the-art deep detectors effectively learn linear maps. Both suggest checking imaging
  methods against simple baselines first.
- **Labels made by models**: [Is machine learning good or bad for the natural sciences?](https://proceedings.mlr.press/v235/hogg24a.html)
  warns that labels produced by expressive regressions carry uncontrolled biases into later joint
  analyses. The same applies when automated segmentations or biometric measurements, such as fetal
  biometry from ultrasound, feed population studies. [Measure Dataset Diversity, Don't Just Claim It](https://proceedings.mlr.press/v235/zhao24a.html)
  reviews 135 datasets and asks for "diverse cohort" claims (across sites, scanners and populations) to be
  measured, not just asserted.
- **Privacy with pretrained models**: [Considerations for Differentially Private Learning with Large-Scale Public Pretraining](https://proceedings.mlr.press/v235/tramer24a.html)
  questions whether web-scraped pretraining data should count as privacy-preserving, and whether current
  benchmarks show how well models transfer to sensitive domains. It also notes that relying on large models
  can mean sending data to a third party with more compute, which matters for hospitals using differential
  privacy (DP).
- **Synthetic images and realism**: [What makes an image realistic?](https://proceedings.mlr.press/v235/theis24a.html)
  explains why a good generative model does not by itself give a reliable realism score. It is the
  theoretical background to the doubts about synthetic-data metrics in the
  [ICML 2025 position overview](../icml2025-position-overview/index.html).
- **Uncertainty, annotation and drift**: [Bayesian Deep Learning is Needed in the Age of Large-Scale AI](https://proceedings.mlr.press/v235/papamarkou24b.html)
  names uncertainty, active learning and continual learning as settings neglected by accuracy-driven
  research. These match calibrated triage, annotation on a budget and adapting to new scanners.
  [Amazing Things Come From Having Many Good Models](https://proceedings.mlr.press/v235/rudin24a.html)
  adds that a large set of equally good models makes room for interpretable clinical risk scores.

## Sources

- *Proceedings of the 41st International Conference on Machine Learning*, PMLR volume 235:
  <https://proceedings.mlr.press/v235/>
- Oral and spotlight decisions from the ICML 2024 schedule data (`icml-2024-orals-posters.json` on icml.cc).
