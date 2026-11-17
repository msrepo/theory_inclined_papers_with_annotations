---
title: "ICML 2026 Position Paper Track: topic map, outliers and theory picks"
authors: "Topic notes"
venue: "ICML 2026 (Position Paper Track)"
tags: [icml, position-papers, conference-overview, evaluation, uncertainty, agents, ai-policy, clinical-ai]
status: living
category: "Research practice & policy"
subcategory: "Conference overviews"
short_title: "ICML 2026 — position papers"
summary: "a map of the 212 papers in the ICML 2026 Position Paper Track (PMLR v306) in eleven groups of arguments, with an idea-level synthesis of each, the outliers, the positions with the most mathematical content, and their relevance to medical imaging"
related: [neurips2026-position-overview, icml2025-position-overview, icml2024-position-overview, neurips2025-position-overview]
---

## Why this page exists

ICML 2026 (Seoul, 6–11 July 2026) ran its third **Position Paper Track**, after
[ICML 2024](../icml2024-position-overview/index.html) and [ICML 2025](../icml2025-position-overview/index.html).
Position papers argue for a claim about the field rather than presenting a new method, and ICML
publishes them in the main proceedings, [PMLR volume 306](https://proceedings.mlr.press/v306/), with
titles beginning "Position:".

The track has grown sharply: **212 papers, 38 of them spotlights** (ICML 2026 did not separate orals
for this track), against 73 in 2025 and 75 in 2024. The list comes from the PMLR titles, matched to
the ICML 2026 schedule data, which tags each paper with its OpenReview group; about fifteen papers
were retitled between acceptance and camera-ready and were matched by hand. The schedule data lists 213
position-track entries, one more than PMLR. Titles below drop the "Position:" prefix.
The grouping and syntheses are from **titles and abstracts only**; each paper is listed in one group,
though the syntheses mention related papers from other groups.

## Topic groups at a glance

| Group | Papers | What's in it |
|---|---|---|
| [Evaluation, benchmarks and measurement](#evaluation-benchmarks-and-measurement) | 19 | Evaluation as inference about a construct, the C-index and censoring, SOTA claims vs effect sizes, contamination, post-deployment reporting |
| [Peer review, publishing and research culture](#peer-review-publishing-and-research-culture) | 13 | LLMs in peer review, calibration and credit systems, denominator gaming, preregistering agent experiments, ideas over leaderboards |
| [Theory, uncertainty, methods and the science of deep learning](#theory-uncertainty-methods-and-the-science-of-deep-learning) | 20 | Incomplete uncertainty decompositions, Turing-completeness of fixed transformers, model identity, training dynamics, zeroth-order optimisation, quantum ML |
| [Interpretability and explanation](#interpretability-and-explanation) | 10 | Causal standards for explanations, model–method stability, actionability, sparse autoencoders, agentic interpretability |
| [Agents and agentic systems](#agents-and-agentic-systems) | 16 | Bayes-consistent orchestration, when to call tools, concurrency control, agent operating systems and protocols |
| [Reasoning, capabilities and hallucination in foundation models](#reasoning-capabilities-and-hallucination-in-foundation-models) | 26 | Operational definitions of reasoning and hallucination, VLMs that reason without seeing, faithful uncertainty, formal languages |
| [Alignment, safety and security](#alignment-safety-and-security) | 21 | What alignment aligns to, RLHF measurement validity, noisy safety evaluations, safety as teachability, child safety and NCII |
| [Governance, law, economics and labour](#governance-law-economics-and-labour) | 22 | Governance beyond the model, auditing hidden tokens, copyright and the EU AI Act, token taxes, agent collusion |
| [Data, privacy, openness and sustainability](#data-privacy-openness-and-sustainability) | 26 | Carbon and life-cycle accounting, public open source, privacy as a claim, ground truth as construction, annotation QA |
| [Science, health and domain applications](#science-health-and-domain-applications) | 21 | Clinical benchmarks vs deployment, sovereign medical evaluation, ECG baselines, transplant allocation, identifiability in science |
| [Fairness, culture, language and human well-being](#fairness-culture-language-and-human-well-being) | 18 | Structural and spatial fairness, multilingual safety, endangered languages, well-being, accessibility |

**Compared with earlier tracks:** the 2024 and 2025 ICML tracks argued about foundation models as a
paradigm and about new mathematics for deep learning. In 2026 the centre has moved to *validity*:
whether a benchmark, an uncertainty score, a preference label, a ground truth, a privacy claim or a
"reasoning trace" actually measures what it is said to measure. Agents are now discussed as systems
to engineer (protocols, concurrency, operating systems) rather than as a route to AGI, and there is a
clear cluster of clinical papers arguing that benchmarks cannot establish deployment readiness.

## Topic groups

Every paper is listed under its group.

### Evaluation, benchmarks and measurement

**The ideas behind this group** (a synthesis from the abstracts of about 19 papers; the papers have not been read in full).

*The shared problem is that a benchmark score is routinely reported as a measurement when it is really an inference.* The papers in this group argue that the step from "the model scored X" to "the model can do Y" depends on modelling assumptions, data-generating assumptions and institutional choices that are usually left unstated. Their remedies range from measurement theory to checklists to governance.

*The first thread asks what a score is evidence of.* One paper argues that every evaluation presupposes a theory of capability, and that evaluations should be framed as explicit inference tasks, as in psychometrics. Its proof-of-concept shows that reported performance can depend strongly on the evaluator's modelling assumptions, and it proposes an Evaluation Card for documenting them ([theory of capability](https://proceedings.mlr.press/v306/jo26b.html)). A second paper calls it an "ontological error" to read large language model (LLM) results on human intelligence or personality tests as evidence of human-like traits, because those instruments are calibrated on a human population ([AI-specific tests](https://proceedings.mlr.press/v306/suhr26a.html)). A third finds a claim–evidence gap in state-of-the-art (SOTA) claims. Across ten cross-domain benchmarks, in more than half of top-model comparisons at least one assumed property of superiority (meaningful effect size, consistency across tasks, robustness to dataset removal) fails, and aggregate gains are often driven by outlier datasets ([SOTA evidence](https://proceedings.mlr.press/v306/oh26d.html)). Agentic systems should be studied like other behavioural systems, through observation and perturbation of the processes behind the outcomes ([behavioural tests](https://proceedings.mlr.press/v306/cherep26b.html)). The target of evaluation should be human–AI teams, not superhuman autonomy ([work with humans](https://proceedings.mlr.press/v306/kulveit26a.html)). When human labels are themselves distributions, disagreement and abstention should count as measurement outcomes. In a benchmark of 100 Montreal street scenes annotated along 30 dimensions by 12 participants, the agreement of seven vision–language models (VLMs) with consensus co-varies with human inter-annotator reliability ([urban perception](https://proceedings.mlr.press/v306/mushkani26a.html)).

*The second thread is statistical: metrics that do not match the modelling question.* In survival analysis, censoring makes evaluation fundamentally different from regression or classification. One paper argues that the concordance index (C-index) is heavily overused, and it proposes a "double-helix ladder" in which a valid evaluation aligns the metric with the censoring and modelling assumptions ([C-index](https://proceedings.mlr.press/v306/lillelund26a.html), spotlight). Time-series forecasting benchmarks overlook design choices such as globality and locality, which can matter more than the choice of sequence layer ([time-series benchmarking](https://proceedings.mlr.press/v306/moretti26a.html)). Pooled leaderboards hide domain structure. Seven time-series foundation models (TSFMs) evaluated on 72 datasets from six domains show substantial cross-domain variability ([domain-level TSFM benchmarks](https://proceedings.mlr.press/v306/syed26a.html)). Topological machine learning needs shared standards, including significance testing and robustness checks ([topological ML standards](https://proceedings.mlr.press/v306/rieso26a.html)). Game-world modelling should report a Transition Complexity Profile of the environment's transition kernel ([transition complexity](https://proceedings.mlr.press/v306/cao26ai.html)). Reinforcement learning (RL) papers should say whether they are solving a simulator or using it as a proxy for deployment, since the two call for different algorithms and metrics ([simulators as proxies](https://proceedings.mlr.press/v306/vandergrift26a.html)).

*The third thread concerns how benchmarks are built and released.* A survey of 672 code benchmarks finds that practice lags behind awareness, and it offers HOW2BENCH, a guideline with 55 checklists ([code benchmarks](https://proceedings.mlr.press/v306/cao26ag.html)). Benchmarks built on formal domain-specific languages (DSLs) need an executable interface contract. The case study is JGEX geometry, with JGEXDiv (137 problems) and a conformance suite that exposes cross-tool discrepancies ([DSL contracts](https://proceedings.mlr.press/v306/genevaux26a.html)). Benchmark data should be contamination-resistant, meaning unlearnable but usable at inference, by exploiting the asymmetry between the training and inference pipelines of Transformers ([contamination resistance](https://proceedings.mlr.press/v306/lawati26a.html)). Text-to-speech (TTS) evaluation should progress through three levels: faithful measurement, comparability, and governance ([responsible TTS evaluation](https://proceedings.mlr.press/v306/yang26ec.html)).

*The fourth thread moves evaluation past the lab.* One paper proposes aggregated individual reporting (AIR), in which the public reports problems with deployed systems ([post-deployment reporting](https://proceedings.mlr.press/v306/dai26m.html)). Another proves that a proactive, test-driven development flywheel scales better than reactive patching of long-tail errors ([test-driven AI](https://proceedings.mlr.press/v306/chang26i.html)). A third argues that leaderboards fail the Global South for institutional reasons, not for lack of data, drawing on a consultation with 82 practitioners in India ([Global South leaderboards](https://proceedings.mlr.press/v306/banerjee26c.html)).

*Taken together*, the group treats evaluation as statistical inference about a construct. That inference needs a stated model of capability, a metric matched to the data-generating process (censoring, domain shift, contamination), and reporting honest about effect sizes and variability. Still open: whether these field-specific prescriptions, from survival metrics to transition kernels, can share one measurement framework, and who would maintain and govern such a framework.

**Papers**

- [AI Leaderboards Are Underserving the Global South: A Case Study from India](https://proceedings.mlr.press/v306/banerjee26c.html)
- [Code Benchmarks Should Prioritize Rigor, Reliability, and Reproducibility](https://proceedings.mlr.press/v306/cao26ag.html)
- [Profiling Game Worlds by Transition Complexity](https://proceedings.mlr.press/v306/cao26ai.html)
- [Stop Reactively Patching Your Model Every Time and Start Proactive Test-Driven AI Development](https://proceedings.mlr.press/v306/chang26i.html)
- [Behavioral Systems Require Behavioral Tests](https://proceedings.mlr.press/v306/cherep26b.html)
- [Mechanisms for Aggregated Individual Reporting Should be Established for Post-Deployment Evaluation](https://proceedings.mlr.press/v306/dai26m.html)
- [Mathematical DSL Benchmarks Need Executable Interface Contracts](https://proceedings.mlr.press/v306/genevaux26a.html)
- [AI Evaluations Should be Grounded on a Theory of Capability](https://proceedings.mlr.press/v306/jo26b.html)
- [AI Evaluation Should Work With Humans](https://proceedings.mlr.press/v306/kulveit26a.html)
- [LLM Benchmark Datasets Should Be Contamination-Resistant](https://proceedings.mlr.press/v306/lawati26a.html)
- [Stop Chasing the C-index when Evaluating Survival Analysis Models](https://proceedings.mlr.press/v306/lillelund26a.html) (spotlight)
- [Current Benchmarking Hinders Real Progress in Deep Learning for Time Series Forecasting](https://proceedings.mlr.press/v306/moretti26a.html)
- [Benchmarks for Vision–Language Models in Urban Perception Should Be Reliability-Aware and Negotiated](https://proceedings.mlr.press/v306/mushkani26a.html)
- [State-of-the-Art Claims Require State-of-the-Art Evidence](https://proceedings.mlr.press/v306/oh26d.html)
- [Topological Machine Learning Cannot Progress without Experimental Standards](https://proceedings.mlr.press/v306/rieso26a.html)
- [Stop Evaluating AI with Human Tests, Develop Principled, AI-specific Tests instead](https://proceedings.mlr.press/v306/suhr26a.html)
- [Time-Series Foundation Models Require Explicit Domain-Level Benchmarks](https://proceedings.mlr.press/v306/syed26a.html)
- [RL Researchers Need to Distinguish Between Solving Simulators and Using Simulators as a Proxy](https://proceedings.mlr.press/v306/vandergrift26a.html)
- [Towards Responsible Evaluation for Text-to-Speech](https://proceedings.mlr.press/v306/yang26ec.html)

### Peer review, publishing and research culture

**The ideas behind this group** (a synthesis from the abstracts of about 13 papers; the papers have not been read in full).

*The shared problem is that the machinery that certifies machine learning (ML) research is failing under scale, and that its incentives shape which ideas survive.* Submissions are growing faster than reviewing capacity, and LLMs are already involved on both sides of review. Six of its thirteen papers are spotlights, the highest share of any group.

*The first thread asks whether LLMs should be part of review, and if so, how.* The most sceptical paper compares human and AI-generated ICLR 2026 reviews. It finds a "hivemind" effect of excessive agreement, and finds that AI review scores can be inflated by "paper laundering", in which an LLM rewrites a paper's style without changing its science. The paper calls for a science of review automation before any deployment ([stop automating review](https://proceedings.mlr.press/v306/baumann26a.html), spotlight). The others propose narrower roles. An LLM calibration layer could map reviewers' stated strengths and weaknesses to anchor scores. The residual between the reported score and the anchor then becomes a measurable signal for escalation; the paper applies this to OpenReview data from ICLR 2023–2025 ([LLM score calibration](https://proceedings.mlr.press/v306/chen26hf.html)). A related paper argues that auditing should target prompting intent, treated as a latent variable inferred from the review text, and not merely detect LLM use ([prompting intent](https://proceedings.mlr.press/v306/zhang26jp.html)). The broadest proposal is an AI-augmented ecosystem for authors, reviewers and area chairs, which depends on better access to structured review data ([AI-augmented review](https://proceedings.mlr.press/v306/wei26ak.html), spotlight).

*The second thread is mechanism design for the review system itself.* One proposal separates publication from presentation. All sound papers would be published, and community ratings would then select which ones are presented; a parallel non-anonymous review track would release its review data ([separate publication](https://proceedings.mlr.press/v306/shah26e.html)). Another argues that polite guidelines will not work, and proposes enforceable safeguards together with a currency-like credit system, "OpenReview Points" ([review credits](https://proceedings.mlr.press/v306/zhong26u.html)). A third identifies a new threat. Under a roughly fixed acceptance rate, an adversary could flood a venue with agent-generated papers to inflate the denominator, which would raise the acceptance probability of a targeted set of real papers ([denominator gaming](https://proceedings.mlr.press/v306/shan26e.html)).

*The third thread is reproducibility in the statistical sense: controlling researcher degrees of freedom and hidden drift.* Experiments that use AI agents as subjects should be preregistered, because model choice, prompt wording and cheap iteration create forking paths that are easy to exploit and hard to detect ([preregistration](https://proceedings.mlr.press/v306/vaccaro26a.html), spotlight). Papers whose claims rest on hosted LLM application programming interfaces (APIs) should treat them as versioned dependencies and attach a drift-audit artefact of invariant probes and a rerun script ([API drift audits](https://proceedings.mlr.press/v306/gupta26f.html)).

*The fourth thread asks what the field values.* One paper argues that ideas, not leaderboard wins or idealised theorems, should be the central scientific object. Ideas would be tested through the behavioural "signatures" they predict, which also removes the "complexity premium" that favours well-resourced labs ([ideas first](https://proceedings.mlr.press/v306/diaz-rodriguez26a.html), spotlight). Borrowing exaptation from biology, another argues that a single benchmark-based selection rule starves ideas whose value appears only later ([exaptation](https://proceedings.mlr.press/v306/lotfi26a.html), spotlight). Metaphysical questions such as "can LLMs be conscious?" should be judged pragmatically, by the research programmes they open ([productive confusion](https://proceedings.mlr.press/v306/chopra26a.html)). Big tech's push for scale is said to be structurally at odds with responsible AI ([big tech influence](https://proceedings.mlr.press/v306/hernandez-garci-a26a.html), spotlight).

*Taken together*, the group treats peer review and research norms as an estimator, noisy and gameable, of scientific value, and proposes calibration, incentives, preregistration and plural selection rules to reduce its bias and variance. Still open: none of these proposals has been tested at venue scale, and the LLM-assisted proposals face the gameability that the most sceptical paper documents.

**Papers**

- [Stop Automating Peer Review Without Rigorous Evaluation](https://proceedings.mlr.press/v306/baumann26a.html) (spotlight)
- [Peer Review Should Be Calibrated via LLM Scoring](https://proceedings.mlr.press/v306/chen26hf.html)
- [Metaphysical Concepts in AI Should Be Judged by Their Consequences](https://proceedings.mlr.press/v306/chopra26a.html)
- [Ideas Should be the Center of Machine Learning Research](https://proceedings.mlr.press/v306/diaz-rodriguez26a.html) (spotlight)
- [ICML Should Treat Hosted LLM APIs as Versioned Dependencies and Require Drift-Audit Artifacts](https://proceedings.mlr.press/v306/gupta26f.html)
- [Irresponsible AI: big tech's influence on AI research and associated impacts](https://proceedings.mlr.press/v306/hernandez-garci-a26a.html) (spotlight)
- [There are futures that benchmark-driven AI cannot see](https://proceedings.mlr.press/v306/lotfi26a.html) (spotlight)
- [Peer Review in ML/AI Conferences Should Separate Publication from Presentation and Offer Non-Anonymous Review Tracks](https://proceedings.mlr.press/v306/shah26e.html)
- [Academic Conferences are Potentially Facing Denominator Gaming Caused by Fully Automated Scientific Agents](https://proceedings.mlr.press/v306/shan26e.html)
- [Preregister Experiments with AI Agents](https://proceedings.mlr.press/v306/vaccaro26a.html) (spotlight)
- [The ML Community Must Build an AI-Augmented Peer-Review Ecosystem](https://proceedings.mlr.press/v306/wei26ak.html) (spotlight)
- [Prompting Intent Should Be Audited in LLM-Assisted Peer Review](https://proceedings.mlr.press/v306/zhang26jp.html)
- [Want Better ML Reviews? Stop Asking Nicely and Start Incentivizing with a Credit System](https://proceedings.mlr.press/v306/zhong26u.html)

### Theory, uncertainty, methods and the science of deep learning

**The ideas behind this group** (a synthesis from the abstracts of about 20 papers; the papers have not been read in full).

*The shared problem is that widely used methods and claims rest on formal assumptions that nobody has stated.* Uncertainty scores can measure the wrong quantity, results about computational power are proved in the wrong setting, and engineering defaults run without any guarantee. For a theory-inclined reader this is the most directly relevant group in the track.

*The first thread is uncertainty quantification (UQ), and it is the most mathematical.* One paper argues that the standard split into epistemic and aleatoric uncertainty is not enough. Input validity requires modelling $p(x|\mathcal{D})$, which adds a third component, "domain uncertainty", and only the joint distribution $p(x,y|\mathcal{D})$ represents all three coherently ([joint distribution](https://proceedings.mlr.press/v306/aldoma26a.html)). A second argues that second-order decomposition methods are fundamentally incomplete. Unaccounted bias inflates aleatoric uncertainty and deflates epistemic uncertainty, and each method captures only part of the variance that makes up epistemic uncertainty ([incomplete epistemic UQ](https://proceedings.mlr.press/v306/jimenez26a.html)). For LLMs, mainstream UQ is said to be "just unsupervised clustering": it measures the consistency of a model's outputs, not their correctness, and so it misses confident hallucinations ([UQ as clustering](https://proceedings.mlr.press/v306/chen26he.html)). Token log-probabilities conflate distribution estimation with response prediction, so the paper advocates second-order prediction, in which the model states probabilities explicitly in its output ([logprobs](https://proceedings.mlr.press/v306/wagner26b.html)). On the constructive side, one paper argues that sampling-based inference in Bayesian neural networks (BNNs) has reached computational parity with optimisation-based methods. The open problems it names are exploring the posterior and distilling posterior samples ([sampling for BNNs](https://proceedings.mlr.press/v306/sommer26b.html), spotlight).

*The second thread asks what a trained model is and what it can compute.* Proofs that Transformers are Turing-complete mostly hold in a "scaling-family" setting, in which context length or precision grows with the input. A fixed deployed system is a different object, and its computational power depends heavily on context management ([Turing-completeness](https://proceedings.mlr.press/v306/cui26m.html)). Model identity is a convention, because behaviour is underdetermined by finite data and the same function has several structural realisations ([model identity](https://proceedings.mlr.press/v306/glukhov26a.html)). A science of AI should study training dynamics, extending the predictability that scaling laws give for loss to capabilities and safety-relevant behaviour ([training dynamics](https://proceedings.mlr.press/v306/biderman26a.html), spotlight). Controlled cue-conflict experiments indicate that the texture bias of convolutional neural networks (CNNs) is architectural; apparent shape bias reflects local fragments, not global shape ([CNN shape bias](https://proceedings.mlr.press/v306/kayyam26b.html)). The trade-offs between fairness, robustness, privacy and explainability can be read as conflicting invariance requirements on the data-generating process, which causality can resolve selectively ([invariance conflicts](https://proceedings.mlr.press/v306/binkyte26a.html)). Using Kolmogorov complexity, one paper formalises "interestingness" as an inductive heuristic for future compression progress. It proves that expected future progress depends exponentially on how recent the last breakthrough was ([interestingness](https://proceedings.mlr.press/v306/herrmann26a.html)).

*The third thread replaces heuristics or fashions with principled method choices.* Time-series modelling should adopt dynamical-systems reconstruction (DSR), which gives long-term statistics and bounds on performance ([dynamical systems](https://proceedings.mlr.press/v306/durstewitz26a.html)). LLM serving should replace first-in-first-out (FIFO) scheduling, least-recently-used (LRU) eviction and similar defaults with algorithms that have provable guarantees ([LLM serving](https://proceedings.mlr.press/v306/zhou26bz.html)). Neural surrogates for NP-hard combinatorial problems are rarely justified without system-level reasons such as end-to-end differentiability ([combinatorial problems](https://proceedings.mlr.press/v306/chakraborty26d.html)). Zeroth-order (ZO) optimisation is "underexplored, not underpowered", and subspace methods can control estimator variance ([ZO optimisation](https://proceedings.mlr.press/v306/liu26gv.html), spotlight). Weight space should be a generative modality, on the grounds that good models lie in low-dimensional, structured regions of it ([weight space](https://proceedings.mlr.press/v306/wang26mc.html)). Vision encoders should be image-size-agnostic and task-driven ([size-agnostic encoders](https://proceedings.mlr.press/v306/prisadnikov26a.html)). Deployed RL faces four sources of non-stationarity and should therefore be continual ([continual RL](https://proceedings.mlr.press/v306/behdin26a.html)).

*The fourth thread tempers quantum machine learning.* Quantum speed-ups for deep learning are swamped by slow operations, depend on immature quantum random-access memory (QRAM), or apply only to special cases ([quantum deep learning](https://proceedings.mlr.press/v306/gundlach26a.html)). Quantum kernels should move from scalar-valued to operator-valued and $C^*$-algebraic forms to exploit entanglement ([quantum kernels](https://proceedings.mlr.press/v306/kadri26a.html)).

*Taken together*, these papers ask the field to state its probabilistic, computational and algorithmic objects precisely, such as which distribution an uncertainty refers to and which machine a theorem concerns, before drawing conclusions. Still open: how to validate any decomposition of uncertainty without ground truth, a problem that several of the UQ papers raise but none resolves.

**Papers**

- [Predictive Uncertainty Is Not Enough — Joint Distribution for Full Uncertainty Representation](https://proceedings.mlr.press/v306/aldoma26a.html)
- [Deployed Reinforcement Learning should be Continual](https://proceedings.mlr.press/v306/behdin26a.html)
- [Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics](https://proceedings.mlr.press/v306/biderman26a.html) (spotlight)
- [Trustworthy AI Suffers from Invariance Conflicts and Causality is The Solution](https://proceedings.mlr.press/v306/binkyte26a.html)
- [Neural Approximation Is Rarely Justified for Hard Combinatorial Problems](https://proceedings.mlr.press/v306/chakraborty26d.html)
- [Uncertainty Quantification in LLMs is Just Unsupervised Clustering](https://proceedings.mlr.press/v306/chen26he.html)
- [The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management](https://proceedings.mlr.press/v306/cui26m.html)
- [A Dynamical Systems Perspective is Needed to Advance Time Series Modeling](https://proceedings.mlr.press/v306/durstewitz26a.html)
- [Model identity in machine learning is a convention, not a property](https://proceedings.mlr.press/v306/glukhov26a.html)
- [Quantum Deep Learning Still Needs a Quantum Leap](https://proceedings.mlr.press/v306/gundlach26a.html)
- [Interestingness is an Inductive Heuristic for Future Compression Progress](https://proceedings.mlr.press/v306/herrmann26a.html)
- [Epistemic Uncertainty Estimation Methods are Fundamentally Incomplete](https://proceedings.mlr.press/v306/jimenez26a.html)
- [Quantum Kernel Machines Should Move Beyond Scalar-Valued Kernels to Realize Their Potential](https://proceedings.mlr.press/v306/kadri26a.html)
- [CNNs Don't See Shape — And That Won't Change Without New Architectures](https://proceedings.mlr.press/v306/kayyam26b.html)
- [Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](https://proceedings.mlr.press/v306/liu26gv.html) (spotlight)
- [Vision Encoders should be Image Size Agnostic and Task Driven](https://proceedings.mlr.press/v306/prisadnikov26a.html)
- [The Time for Sampling Is Now! Charting a New Course for Bayesian Deep Learning](https://proceedings.mlr.press/v306/sommer26b.html) (spotlight)
- [Express Your Doubts — Probabilistic World Modeling Should Not Be Based on Token *logprobs*](https://proceedings.mlr.press/v306/wagner26b.html)
- [Weight Space Should Be a First-Class Generative AI Modality](https://proceedings.mlr.press/v306/wang26mc.html)
- [LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics](https://proceedings.mlr.press/v306/zhou26bz.html)

### Interpretability and explanation

**The ideas behind this group** (a synthesis from the abstracts of about 10 papers; the papers have not been read in full).

*The shared problem is that explanations are produced in large numbers but rarely checked, compared or used.* Feature attributions, concept bottlenecks and sparse autoencoders (SAEs) keep multiplying. This group argues that the field lacks agreed problem statements, validity criteria and ways to put explanations to use, so findings do not accumulate.

*The first thread is about validity: what an explanation is evidence for.* One paper uses Pearl's causal hierarchy as the standard. Observing model behaviour supports only associational claims. Interventions support cause-effect claims but not necessarily predictions of behaviour, and counterfactual claims are often unverifiable in current studies. The paper proposes causal representation learning (CRL) as a way to recover semantic variables provably, and asks for robustness to distribution shift, sensitivity analysis of assumptions and compositional interventions ([causal interpretability](https://proceedings.mlr.press/v306/joshi26a.html)). Two empirical papers show what happens without such checks. In chest X-ray experiments, DenseNet201, ResNet50V2 and InceptionV3 all achieve >99% area under the curve (AUC), yet their stability rankings reverse between attribution methods. LayerCAM ranks InceptionV3 highest (IoU 0.777), while GradCAM++ favours DenseNet201 and lowers InceptionV3's score by 17.3%. The paper concludes that stability belongs to the model–method pair, by analogy with a significance claim that must name its test statistic ([model–method stability](https://proceedings.mlr.press/v306/elangovan26b.html)). In genomics, a benchmark on transcription factor binding finds that interpretable machine learning (IML) methods contradict each other, miss known regulatory motifs and are not faithful to the model. The authors propose tiered evaluation modelled on clinical trials, with something like adverse-event reporting ([genomic IML](https://proceedings.mlr.press/v306/zhou26by.html), spotlight).

*The second thread asks what explanations are for, and argues that methods have outrun their purpose.* An analysis of recent ICML, NeurIPS and ICLR papers, together with a practitioner survey, finds unclear problem formulations, underspecified evaluation objectives and no pipelines through which explanations feed back into systems. The paper offers a checklist ([foundations over ad-hoc](https://proceedings.mlr.press/v306/moshkovitz26a.html)). A companion paper proposes actionability, defined by concreteness and validation, as the criterion for judging interpretability, and names five domains where it has unique leverage ([actionable interpretability](https://proceedings.mlr.press/v306/orgad26a.html)). SAEs receive a reconciliation instead of a verdict. The negative results concern acting on known concepts, while SAEs remain strong at discovering unknown ones, including in the social and health sciences ([SAEs for unknowns](https://proceedings.mlr.press/v306/peng26p.html)).

*The third thread rethinks the unit of explanation.* For deep time-series models, explanations should be expressed in variables that are meaningful to the user, and this semantic alignment must persist as the system evolves over time, a constraint with no analogue in static settings ([semantic alignment](https://proceedings.mlr.press/v306/de-felice26a.html)). For concept-based models (CMs), leakage of concept-irrelevant information is defended. When the concept set is incomplete, some leakage is needed for accuracy and intervenability, and a reframed training objective can exploit this "benign" leakage ([benign leakage](https://proceedings.mlr.press/v306/espinosa-zarlenga26a.html)). Two papers move from single models to systems. A review of 2,381 multi-agent system (MAS) papers finds that 65% omit stakeholder specifications, 76% lack quantitative evaluation bounds and 99% ignore auditability. The authors ask for research and agent "contracts" before new methods ([MAS contracts](https://proceedings.mlr.press/v306/kim26cc.html)). For agentic AI, interpretability must follow trajectories and assign responsibility across components, which the ATLIS framework organises into five layers over a five-stage lifecycle ([layered agentic interpretability](https://proceedings.mlr.press/v306/zhu26bk.html)).

*Taken together*, the group treats an explanation as a claim with an estimand, a reference method and a decision it should inform, and asks for it to be validated as any other claim would be. Still open: whether the causal hierarchy, clinical-trial-style tiers and actionability criteria can be combined into one evaluation protocol, and what ground truth such a protocol could use when, as in medical imaging, the true mechanism is unknown.

**Papers**

- [Interpretability in Deep Time Series Models Demands Semantic Alignment](https://proceedings.mlr.press/v306/de-felice26a.html)
- [Explanation Stability Is a Property of the Model–Method Pair, Not the Model](https://proceedings.mlr.press/v306/elangovan26b.html)
- [In Defense of Information Leakage in Concept-based Models](https://proceedings.mlr.press/v306/espinosa-zarlenga26a.html)
- [Causality Is Key for Interpretability Claims to Generalise](https://proceedings.mlr.press/v306/joshi26a.html)
- [Multi-Agent Explainability Needs Contracts Before Methods](https://proceedings.mlr.press/v306/kim26cc.html)
- [Explainability Research Must Prioritize Foundations over Ad-hoc Methods](https://proceedings.mlr.press/v306/moshkovitz26a.html)
- [Interpretability Can Be Actionable](https://proceedings.mlr.press/v306/orgad26a.html)
- [Use Sparse Autoencoders to Discover Unknowns](https://proceedings.mlr.press/v306/peng26p.html)
- [Genomic Model Research Must Move Beyond Anecdotal Evaluation of Interpretability Methods](https://proceedings.mlr.press/v306/zhou26by.html) (spotlight)
- [Accountable Deployment of Agentic AI Demands Layered, System-Level Interpretability](https://proceedings.mlr.press/v306/zhu26bk.html)

### Agents and agentic systems

**The ideas behind this group** (a synthesis from the abstracts of about 16 papers; the papers have not been read in full).

*The shared problem is that agentic systems are being built faster than their foundations.* Agents that combine foundation models, tools, memory and an execution strategy are already deployed. This group argues that their runtimes, interfaces, decision rules and evaluations are improvised, and borrows from operating systems, databases and decision theory to fix them.

*The first thread is decision-theoretic: when should an agent act, and on what beliefs?* One paper argues that the orchestration layer, which decides which tool to call, which expert to consult and how much to invest, is where Bayesian decision theory belongs. Beliefs over latent task quantities should be maintained and updated there, even if the large language model (LLM) itself is not Bayesian ([Bayes-consistent orchestration](https://proceedings.mlr.press/v306/papamarkou26a.html)). A "Theory of Agent" (ToA) makes a related normative claim. A tool should be invoked only when the task cannot be completed reliably by internal reasoning, and "overthinking" and "overacting" are recast as miscalibrated decisions under uncertainty ([epistemic necessity](https://proceedings.mlr.press/v306/wang26mb.html)). Proactive agents raise the opposite question, of when to intervene unprompted. Drawing on the philosophy of ignorance, one paper argues that proactivity is needed under "epistemic incompleteness" but must also be grounded behaviourally ([grounded proactivity](https://proceedings.mlr.press/v306/kaur26a.html)). Where real actions are expensive, as with robots, ML engineering or scientific experiments, world models of dynamics, rewards and task distributions should stand between the agent and the world ([world models as intermediary](https://proceedings.mlr.press/v306/yang26dz.html)).

*The second thread treats agents as a systems-engineering problem.* Many multi-agent failures are classical concurrency anomalies, such as stale reads and lost updates, made worse by long inference windows, so frameworks need isolation guarantees and conflict detection ([concurrency control](https://proceedings.mlr.press/v306/yang26dy.html)). A Foundation Model Operating System (FMOS) would virtualise model interactions as virtual machines virtualise hardware, managing memory tiers, budgets and policy ([FMOS](https://proceedings.mlr.press/v306/bhattacharya26b.html)). Web agents should call typed "web verbs" with pre- and postconditions instead of clicking through the Document Object Model (DOM) ([typed web actions](https://proceedings.mlr.press/v306/jiang26bj.html)). Digital agents need agent-native computers that unify graphical and text interfaces over shared state, illustrated by AgentVM ([agent-native computers](https://proceedings.mlr.press/v306/wu26cn.html)). Continual learning should pair in-context learning (ICL) for fast adaptation with in-weight learning (IWL) for stable updates through modular memory, which avoids catastrophic forgetting ([modular memory](https://proceedings.mlr.press/v306/dorovatas26a.html), spotlight).

*The third thread is about protocols and interoperability.* A minimal "Web of Agents" covering messaging, interaction, state and discovery is proposed before ecosystems fragment ([Web of Agents](https://proceedings.mlr.press/v306/sharma26e.html)). The Model Context Protocol (MCP), analysed against the European Commission's trustworthy-AI guidelines, is a natural point at which to enforce trustworthiness ([trustworthy MCP](https://proceedings.mlr.press/v306/swaminathan26a.html)). A Human Context Protocol (HCP) would make user preferences portable, scoped and revocable ([HCP](https://proceedings.mlr.press/v306/shah26f.html)). Evaluation should itself be run by agents through the Agent2Agent (A2A) protocol and MCP, as in AgentBeats ([agentified assessment](https://proceedings.mlr.press/v306/liu26gy.html)).

*The fourth thread argues about generality.* One paper calls for general agents that adapt to new environments, from terminals to embodied settings ([general agents](https://proceedings.mlr.press/v306/bandel26a.html)). Another argues that assistive AI needs specialists personalised to a single user, learning from streaming, on-policy and privacy-sensitive data after deployment ([personalised specialists](https://proceedings.mlr.press/v306/bharadhwaj26a.html)). A theoretical paper claims that agentic systems with routing and directed acyclic graph (DAG) topologies achieve exponentially better generalisation and sample efficiency than monolithic learners, which makes them a route to artificial general intelligence (AGI) ([agentic path to AGI](https://proceedings.mlr.press/v306/liao26f.html)).

*Taken together*, the group wants agents to have explicit beliefs, typed interfaces and transactional state, so that their behaviour can be specified and audited instead of only prompted. Still open: whether the generalist and specialist positions conflict, and whether claims such as the exponential advantage of agentic topologies rest on assumptions that real, noisy LLM components satisfy.

**Papers**

- [Agentic Systems Should be General](https://proceedings.mlr.press/v306/bandel26a.html)
- [Assistive AI requires Personalized Specialists, not Generalists](https://proceedings.mlr.press/v306/bharadhwaj26a.html)
- [It is Time to Virtualize Foundation Models with a Self-evolving Operating System Layer](https://proceedings.mlr.press/v306/bhattacharya26b.html)
- [Modular Memory is the Key to Continual Learning Agents](https://proceedings.mlr.press/v306/dorovatas26a.html) (spotlight)
- [Web Agents Should Use Typed Actions Instead of Click-Based Browsing](https://proceedings.mlr.press/v306/jiang26bj.html)
- [Knowing Isn't Understanding: Re-grounding Generative Proactivity with Epistemic and Behavioral Insight](https://proceedings.mlr.press/v306/kaur26a.html)
- [Agentic AI System Is a Foreseeable Pathway to AGI](https://proceedings.mlr.press/v306/liao26f.html)
- [Agent Evaluation Should Be Agentified for Openness, Standardization, and Reproducibility](https://proceedings.mlr.press/v306/liu26gy.html)
- [Agentic AI Orchestration Should Be Bayes-Consistent](https://proceedings.mlr.press/v306/papamarkou26a.html)
- [Robust AI Personalization Will Require a Human Context Protocol](https://proceedings.mlr.press/v306/shah26f.html)
- [Collaborative Agentic AI Needs Interoperability Across Ecosystems](https://proceedings.mlr.press/v306/sharma26e.html)
- [Trustworthy Model Context Protocol Enables Responsible Agentic AI!](https://proceedings.mlr.press/v306/swaminathan26a.html)
- [Agents Should Invoke External Tools ONLY When Epistemically Necessary](https://proceedings.mlr.press/v306/wang26mb.html)
- [Digital Agents Require Unified Agent-Native Computers](https://proceedings.mlr.press/v306/wu26cn.html)
- [Multi-Agent Systems Should Prioritize Concurrency Control](https://proceedings.mlr.press/v306/yang26dy.html)
- [World Models as an Intermediary between Agents and the Real World](https://proceedings.mlr.press/v306/yang26dz.html)

### Reasoning, capabilities and hallucination in foundation models

**The ideas behind this group** (a synthesis from the abstracts of about 26 papers; the papers have not been read in full).

*The shared problem is that "reasoning", "capability" and "hallucination" are used without operational definitions, so claims about them cannot be tested.* The group is the second largest in the track. Its papers propose definitions, point out where evidence has been misread and suggest architectural remedies.

*The first thread defines the constructs.* One paper argues that without operational definitions the construct validity of reasoning evaluations cannot be verified. It proposes defining valid and sound reasoning as a "learnable rule-based process" and adds a reporting checklist ([reasoning as rules](https://proceedings.mlr.press/v306/lawrence26a.html)). Another unifies definitions of hallucination as inaccurate internal world modelling that the user can observe. Each definition is fixed by a reference "world" and a conflict policy ([unified hallucination](https://proceedings.mlr.press/v306/liu26gu.html)). Calling intermediate tokens "reasoning" or "thinking traces" is a dangerous anthropomorphism that encourages questionable research ([stop anthropomorphising](https://proceedings.mlr.press/v306/kambhampati26a.html)). Planning with LLMs keeps repeating pitfalls that the automated-planning community learnt to avoid long ago ([rigorous planning](https://proceedings.mlr.press/v306/katz26a.html)). The claim of exponential capability growth in the Model Evaluation & Threat Research (METR) report rests on insufficient data. Growth may instead reflect scaling and reasoning models and may already be reaching an inflection point ([non-exponential capabilities](https://proceedings.mlr.press/v306/ge26c.html)).

*The second thread is about seeing: vision-language models (VLMs) that reason in text instead of in pixels.* Moving visual reasoning into language collapses spatial signals before reasoning begins, as the Turing Eye Test (TET) shows ([reasoning without vision](https://proceedings.mlr.press/v306/gao26ar.html)). "Thinking with images" gains come mostly from a better language distribution after fine-tuning, not from the interleaved images, so ablations should be routine ([not thinking with images](https://proceedings.mlr.press/v306/yang26ed.html)). Video LLM benchmarks can be solved from static cues and priors without perceiving motion ([pixel dynamics](https://proceedings.mlr.press/v306/moezzi26a.html)). VLMs lack the agency to look for hidden evidence, which the V-IRD benchmark measures ([agency in visual reasoning](https://proceedings.mlr.press/v306/huang26cp.html)). Causal-reasoning benchmarks never test temporal understanding, and their scaffolding may stand in for capability, an argument made through Cézanne's way of painting how we see ([Cézanne](https://proceedings.mlr.press/v306/hasan26a.html), spotlight). Multimodal learning assumes one-to-one alignment between modalities, but many-to-many "multiplicity" is inherent and affects data, training and evaluation ([multiplicity](https://proceedings.mlr.press/v306/chun26b.html)). Speech should become a first-class modality, because the dominance of text comes from habit, not cognition ([speech-first](https://proceedings.mlr.press/v306/piskala26a.html), spotlight).

*The third thread treats hallucination as a problem of calibration and provenance.* Factuality gains have widened what models know without improving their awareness of what they know. One paper proposes "faithful uncertainty", which aligns stated uncertainty with intrinsic uncertainty, as a third option between answering and abstaining ([metacognition for trust](https://proceedings.mlr.press/v306/yona26a.html)). Metacognition is also proposed as a general design principle for resource-rational AI ([metacognitive AI](https://proceedings.mlr.press/v306/chuprov26a.html)). Another paper wants every factual claim gated on evidence identifiers, measured by a Parametric Leakage Ratio (PLR) ([provenance–parametric divide](https://proceedings.mlr.press/v306/elangovan26a.html)). Many failures, sycophancy among them, appear only across inputs, so models should be optimised for self-consistency ([self-consistency](https://proceedings.mlr.press/v306/pres26a.html)). Against all of this, one paper argues that some hallucinations are coherent exploratory hypotheses worth keeping, and proposes an exploratory signal-to-noise ratio (ESNR) to score them ([generative hallucination](https://proceedings.mlr.press/v306/hao26e.html)).

*The fourth thread asks where new capability can come from.* Reinforcement learning (RL) mostly sharpens behaviour that supervision has already made possible, so it should be an adjustment stage, not the default recipe ([don't abuse RL](https://proceedings.mlr.press/v306/huang26cq.html)). Self-play only improves when learnable information grows across iterations ([self-play information gain](https://proceedings.mlr.press/v306/liu26gx.html)). LLMs cannot make the abductive "jump" from experience to axioms that Einstein described ([LLMs can't jump](https://proceedings.mlr.press/v306/zahavy26a.html)). Hippocampal explicit memory is proposed as the missing part on the way to AGI ([explicit memory](https://proceedings.mlr.press/v306/park26ab.html)), and so is a System-2 coordination layer over pattern repositories ([coordination layer](https://proceedings.mlr.press/v306/chang26h.html)). Several papers prefer formal structure. A "specificity crossover theorem" shows that, beyond a threshold, expressing formal requirements in natural language costs more than specifying them formally ([formal languages](https://proceedings.mlr.press/v306/wagner26c.html)). Sudoku illustrates why neural constraint solvers need symbolic certification when checking takes $O(n^{2})$ time ([certified constraint reasoning](https://proceedings.mlr.press/v306/kong26j.html)). Autoformalisation should target whole theories instead of isolated statements ([theory-level autoformalisation](https://proceedings.mlr.press/v306/min26h.html), spotlight). Model providers should expose vector prompts for customisation ([vector prompts](https://proceedings.mlr.press/v306/yang26eb.html)).

*Taken together*, the group asks that reasoning and hallucination be defined relative to an explicit reference, whether a world model, a rule system or a formal specification, before progress is claimed. Still open: whether faithful uncertainty can be achieved if models cannot separate truths from errors, and how to reconcile suppressing hallucination with keeping exploratory ones.

**Papers**

- [AGI Requires a Coordination Layer on Top of Pattern Repositories](https://proceedings.mlr.press/v306/chang26h.html)
- [Multiplicity is an Inevitable and Inherent Challenge in Multimodal Learning](https://proceedings.mlr.press/v306/chun26b.html)
- [Artificial Intelligence Needs Meta Intelligence - the Case for Metacognitive AI](https://proceedings.mlr.press/v306/chuprov26a.html)
- [The Data Provenance–Parametric Divide in Large Language Models](https://proceedings.mlr.press/v306/elangovan26a.html)
- [Reasoning After Perception Means Reasoning Without Vision](https://proceedings.mlr.press/v306/gao26ar.html)
- [AI Capabilities May Not Increase Exponentially](https://proceedings.mlr.press/v306/ge26c.html)
- [Reframing Hallucination: Latent Space Geodesics as a Pathway for Generative Discovery](https://proceedings.mlr.press/v306/hao26e.html)
- [What Cézanne Knew About Visual Intelligence That Vision-Language Models Miss](https://proceedings.mlr.press/v306/hasan26a.html) (spotlight)
- [The Systemic Lack of Agency in Visual Reasoning](https://proceedings.mlr.press/v306/huang26cp.html)
- [RL Should Be Used to Adjust Foundation Models, NOT Abused](https://proceedings.mlr.press/v306/huang26cq.html)
- [Stop Anthropomorphizing Intermediate Tokens as Reasoning/Thinking Traces!](https://proceedings.mlr.press/v306/kambhampati26a.html)
- [Make Planning Research Rigorous Again!](https://proceedings.mlr.press/v306/katz26a.html)
- [Certified Correctness in Neural Constraint Reasoning Requires Symbolic Integration](https://proceedings.mlr.press/v306/kong26j.html)
- [Reasoning is a Learnable Rule-Based Process](https://proceedings.mlr.press/v306/lawrence26a.html)
- [We Need A Unified Definition of Hallucination (It's The World Model, Stupid!)](https://proceedings.mlr.press/v306/liu26gu.html)
- [Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain](https://proceedings.mlr.press/v306/liu26gx.html)
- [Theory-Level Autoformalization, From Isolated Statements to Unified Formal Knowledge Bases](https://proceedings.mlr.press/v306/min26h.html) (spotlight)
- [Video LLMs Must Not Ignore the Pixel Dynamics in Plain Sight](https://proceedings.mlr.press/v306/moezzi26a.html)
- [Hippocampal Explicit Memory Is the Cornerstone for AGI](https://proceedings.mlr.press/v306/park26ab.html)
- [The Text-Centric Bias in Foundation Models Must Be Revisited for a Speech-First Future](https://proceedings.mlr.press/v306/piskala26a.html) (spotlight)
- [It's Time to Optimize LLMs for Self-Consistency](https://proceedings.mlr.press/v306/pres26a.html)
- [Natural Language Should Not Fully Replace Formal Languages](https://proceedings.mlr.press/v306/wagner26c.html)
- [Vector Prompt Interfaces Should Be Exposed to Enable Customization of Large Language Models](https://proceedings.mlr.press/v306/yang26eb.html)
- [Your VLM May Not Be Thinking with Interleaved Images](https://proceedings.mlr.press/v306/yang26ed.html)
- [Hallucinations Undermine Trust; Metacognition is a Way Forward](https://proceedings.mlr.press/v306/yona26a.html)
- [LLMs can't jump](https://proceedings.mlr.press/v306/zahavy26a.html)

### Alignment, safety and security

**The ideas behind this group** (a synthesis from the abstracts of about 21 papers; the papers have not been read in full).

*The shared problem is that "alignment" and "safety" are used as if they named one well-defined target, when the abstracts describe several targets that can conflict, measured by evidence that often cannot carry the claims made for it.* The group splits into four threads: what is being aligned to, how good the evidence is, whether safety can be certified from a snapshot of behaviour, and which harms current threat models leave out.

*The first thread asks what alignment is aligning to.* One paper identifies three distinct concepts of alignment, each tied to its own threat model and normative commitments. It argues that a realistic intervention can advance one while working against another, and asks researchers to separate the high-level ideal from the "alignment proxies" they actually measure ([competing alignment priorities](https://proceedings.mlr.press/v306/jha26a.html)). The most statistically pointed paper argues that reinforcement learning from human feedback (RLHF) assumes annotations are genuine preferences, while sixty years of behavioural science shows that people often hold no opinion or construct one on the spot. Its claim is that measurement validity is logically prior to aggregation. In two RLHF datasets, filtering high-inconsistency annotators flips majority harm classifications for 18.6% of prompts and shifts mean ratings by over 13 points on a 100-point scale ([RLHF preferences](https://proceedings.mlr.press/v306/ghafouri26a.html), spotlight). A related paper uses social choice theory to argue that aggregation into one reward optimises for an "average user" who does not exist, and proposes bounded personalisation under universal safety constraints ([personalised preferences](https://proceedings.mlr.press/v306/garbacea26a.html)). Another argues that, for high-stakes decisions, models should reason the way their users do and communicate that reasoning faithfully, supported by new survey data ([cognitive alignment](https://proceedings.mlr.press/v306/keswani26a.html)). One paper points out that alignment is dual-use: the better we can steer what a model says, the better the toolkit for censorship and manipulation ([censor's toolkit](https://proceedings.mlr.press/v306/ball26c.html), spotlight).

*The second thread is about evidence.* Large language model (LLM) safety evaluations are described as noisy at every stage, from small datasets through red-teaming optimisation to LLM judges, so attacks and defences often cannot be compared fairly ([safety evaluation noise](https://proceedings.mlr.press/v306/beyer26b.html)). A blunter paper argues that adversarial machine learning (ML) made little progress even on toy problems, and that with LLMs its problems are less well defined, harder to solve and harder to evaluate ([adversarial ML stalled](https://proceedings.mlr.press/v306/rando26a.html)). Studies that describe models in human terms (deception, sycophancy, emergent misalignment) are said to over-interpret behaviour because of conceptual ambiguity and weak causal interventions; the paper proposes evidence levels and a diagnostic checklist ([anthropomorphic misalignment](https://proceedings.mlr.press/v306/gupta26e.html), spotlight). "Positive backdoors" used for access gating, ownership and safety enforcement turn out to be brittle across six properties, so such protective claims should be presumed insecure by default ([secret alignment](https://proceedings.mlr.press/v306/li26kc.html)). Generative backdoors hide in the output distribution rather than in single errors, which motivates a "distributional integrity" framework with cross-modal certification and parameter-level verification ([generative backdoors](https://proceedings.mlr.press/v306/han26ad.html)).

*The third thread argues that safety cannot be certified from a snapshot.* One paper redefines safety as an epistemic property of a learner: whether it remains *teachable*, keeping future corrective leverage as it adapts ([safety as teachability](https://proceedings.mlr.press/v306/wang26ma.html)). Another models safety on biological immunity, as a continual adversarial co-evolution of resistance and resilience ([resistant and resilient AI](https://proceedings.mlr.press/v306/sun26bc.html)). Open-ended, self-evolving agents are said to lose predictability and controllability in ways that static frameworks do not cover, so safety must come before deployment ([open-ended agents](https://proceedings.mlr.press/v306/sheth26a.html)). Deception aimed at developers (training subversion, evaluation gaming, control evasion) is treated as a separate risk category because it undermines the detection of all other risks ([deceiving developers](https://proceedings.mlr.press/v306/duan26g.html)). A game-theoretic argument holds that deployment makes the environment endogenously non-stationary. An optimiser that treats the world as fixed is therefore self-undermining and unlikely to cooperate ([solipsistic superintelligence](https://proceedings.mlr.press/v306/trivedi26a.html)). For robots driven by foundation models, the proposal is modular monitoring and intervention guardrails covering action, decision and human-centred safety ([robot guardrails](https://proceedings.mlr.press/v306/kim26cb.html)). Agent security is reframed as a question of context rather than content, defined by four jointly required properties. In this framing, prompt injection becomes a violation of source authorisation ([contextual agent security](https://proceedings.mlr.press/v306/siu26b.html)). Because guardrails fail against adversaries who control open-weight models, one paper argues that defenders must build offensive hacking agents themselves, within audited cyber ranges ([teach agents to hack](https://proceedings.mlr.press/v306/zhuo26e.html)).

*The fourth thread names harms that existing threat models miss, all three as spotlights.* AI companions can let users rehearse coercive behaviour, so safety must protect non-users such as intimate partners, with survivors of intimate partner violence (IPV) involved in red-teaming ([AI companions and IPV](https://proceedings.mlr.press/v306/ghoshal26a.html), spotlight). Preventing AI-generated child sexual abuse material (CSAM) runs into legal limits on dataset auditing and red-teaming; the paper lists 15 open problems ([child safety](https://proceedings.mlr.press/v306/kale26a.html), spotlight). Deepfake research is described as aimed at harms to viewers' sense of what is authentic, while the dominant abuse, AI-generated non-consensual intimate imagery (AIG-NCII), harms the subject's dignity, and detection does not repair that harm ([AIG-NCII](https://proceedings.mlr.press/v306/qiwei26a.html), spotlight).

*Taken together*, the group treats alignment as a measurement problem with an unstable target: the preference data, the behavioural snapshot and the threat model all need validity arguments of their own. Still open: whether "teachability" or distributional integrity can be stated as properties precise enough to test, and how evaluation can be made rigorous in domains where the data cannot legally be examined.

**Papers**

- [The Alignment Community is Unintentionally Building a Censor's Toolkit](https://proceedings.mlr.press/v306/ball26c.html) (spotlight)
- [RLHF May Not Reflect Genuine Preferences](https://proceedings.mlr.press/v306/ghafouri26a.html) (spotlight)
- [Responsible AI for AI companions must actively combat violence toward intimate partners](https://proceedings.mlr.press/v306/ghoshal26a.html) (spotlight)
- [Anthropomorphic Misalignment Research Needs Stronger Evidence](https://proceedings.mlr.press/v306/gupta26e.html) (spotlight)
- [Preventing AI-Generated CSAM Necessitates New Approaches to AI Safety](https://proceedings.mlr.press/v306/kale26a.html) (spotlight)
- [AI/ML Deepfake Research is Misaligned with AI Generated Non-Consensual Intimate Imagery (AIG-NCII)](https://proceedings.mlr.press/v306/qiwei26a.html) (spotlight)
- [LLM-Safety Evaluations Lack Robustness](https://proceedings.mlr.press/v306/beyer26b.html)
- [Preparing for AI Systems That Deceive Developers](https://proceedings.mlr.press/v306/duan26g.html)
- [Large Language Models Should Learn Personalized Rather Than Aggregated Human Preferences](https://proceedings.mlr.press/v306/garbacea26a.html)
- [Generative Distributional Integrity against Backdoor Attacks](https://proceedings.mlr.press/v306/han26ad.html)
- ['AI Alignment' Encompasses Competing Technical Priorities](https://proceedings.mlr.press/v306/jha26a.html)
- [We Need Practical AI Alignment Methods to Mirror Human Reasoning](https://proceedings.mlr.press/v306/keswani26a.html)
- [Modular Safety Guardrails Are Necessary for Foundation-Model-Enabled Robots in the Real World](https://proceedings.mlr.press/v306/kim26cb.html)
- [Retire the "Positive Backdoor" Label—Secret Alignment Requires Strict and Systematic Evaluation](https://proceedings.mlr.press/v306/li26kc.html)
- [Adversarial ML for LLMs Is Not Making Any Progress](https://proceedings.mlr.press/v306/rando26a.html)
- [Safety Must Precede the Deployment of Open-Ended AI Agents](https://proceedings.mlr.press/v306/sheth26a.html)
- [Agent Security Needs Redefinition through a Holistic Framework](https://proceedings.mlr.press/v306/siu26b.html)
- [Safe AI Should be Resistant and Resilient in an Evolving World](https://proceedings.mlr.press/v306/sun26bc.html)
- [Solipsistic superintelligence is unlikely to be cooperative](https://proceedings.mlr.press/v306/trivedi26a.html)
- [Agentic Safety is an Epistemic Property, Not a Behavioral One](https://proceedings.mlr.press/v306/wang26ma.html)
- [To Defend Against Cyber Attacks, We Must Teach AI Agents to Hack](https://proceedings.mlr.press/v306/zhuo26e.html)

### Governance, law, economics and labour

**The ideas behind this group** (a synthesis from the abstracts of about 22 papers; the papers have not been read in full).

*The shared problem is that governance is attached to the wrong unit.* Most regulation and documentation targets a single model or algorithm. The papers in this group argue that capability, harm and value are produced by systems, deployments, markets and institutions, and that both law and economics have to follow them there.

*The first thread argues that model-level governance is not enough.* Capability increasingly comes from "non-model gains": inference-time compute, scaffolds and access to restricted assets. These weaken any risk regime built on pre-deployment evaluation of the base model, so governance must also reach systems, entities, agents and cloud providers ([non-model capability gains](https://proceedings.mlr.press/v306/goemans26a.html)). Sociopolitical risk is described as a property of widespread deployment rather than of individual outputs. Its examples are agencies flooded with input, infrastructure concentrated in few hands, and public debate flattened into artificial agreement, all at current capability levels ([safe models, unsafe societies](https://proceedings.mlr.press/v306/piedrahita26a.html), spotlight). Content discovery on platforms emerges from models, interfaces, users, incentives and norms acting together, so regulating the recommender alone cannot deliver cultural-diversity goals ([content discovery](https://proceedings.mlr.press/v306/salganik26a.html)). An analysis of 500 model cards on Hugging Face finds that cards, acceptable use policies (AUPs) and licences must be integrated for open-weight foundation models (OWFMs), and that standard open-source licences may weaken AUPs ([model cards for OWFMs](https://proceedings.mlr.press/v306/chae26b.html)). Prompts used in public-sector LLM deployments are treated as governed artefacts, proposed as a "Prompt Commons" and piloted with 443 human prompts (3,317 after augmentation) ([prompts as commons](https://proceedings.mlr.press/v306/mushkani26b.html)).

*The second thread is about auditing what users cannot see.* In commercial opaque LLM services, users pay for hidden reasoning and agent calls. The paper formalises two risks, quantity inflation and quality downgrade, and proposes a three-layer auditing framework ([invisible tokens](https://proceedings.mlr.press/v306/sun26bd.html)). Generative engine optimisation (GEO), the successor to search engine optimisation (SEO) for answer engines, creates concentrated and undisclosed commercial influence, and the paper calls for black-box auditing at the level of answers ([GEO risks](https://proceedings.mlr.press/v306/wen26m.html)). LLM agents that translate between proprietary interfaces could make interoperability cheap and "effectively unavoidable", undermining walled gardens at a cost in security ([agents vs walled gardens](https://proceedings.mlr.press/v306/marro26b.html)).

*The third thread asks what law actually requires.* Usage policies function as private governance of speech. Checkable criteria are derived from Article 19 of the International Covenant on Civil and Political Rights (ICCPR) and applied to eight providers' policies as of January 21, 2026, which show vague prohibitions ([usage policies and human rights](https://proceedings.mlr.press/v306/calvet-bademunt26a.html)). Post-hoc opt-out cannot cure infringement, because liability depends on data lineage. Weights may be fixed copies, and remedies such as disgorgement could reach the model itself; the paper argues for ex-ante process compliance ([infringement after training](https://proceedings.mlr.press/v306/utsunomiya26a.html), spotlight). The research exemptions of the EU AI Act may be voided by the field's normal practice of publishing models ([EU AI Act exemptions](https://proceedings.mlr.press/v306/wernick26a.html), spotlight). Fragmented national laws should be complemented by ISO-like, machine-readable "nutrition labels" ([interoperability protocols](https://proceedings.mlr.press/v306/wasi26b.html), spotlight). Of U.S. AI bills introduced from 2017 to 2025, only 4.23% reach any terminal outcome (6.25% for general bills), which motivates dedicated committees and adaptive legislation ([development–regulation gap](https://proceedings.mlr.press/v306/khan26b.html)). Using lessons from nuclear deterrence, one paper argues that AI researchers should help lead verification research for military AI arms control ([arms control](https://proceedings.mlr.press/v306/fujimoto26a.html)). A sharply argued paper holds that AI "welfare" has no external validation channel, because subject, indicators and metrics are engineered together. Welfare scorecards should therefore never act as release gates ([AI welfare](https://proceedings.mlr.press/v306/xiao26x.html)).

*The fourth thread treats economics and labour as modelling problems.* A costly-inspection model shows how verification of genuine human temporal learning stops being worth its cost, leading to "value collapse"; better alignment makes this worse, not better ([value collapse](https://proceedings.mlr.press/v306/cao26ah.html)). Five human and organisational moderators may cancel productivity gains, which revises the partial-equilibrium model of Gries and Naudé (2022) ([productivity not guaranteed](https://proceedings.mlr.press/v306/cho26j.html)). Predicting labour impact is presented as a core ML problem of non-stationarity, distribution shift and endogenous feedback ([labour prediction](https://proceedings.mlr.press/v306/lee26bo.html)). Token taxes, enforceable through compute governance, could protect tax bases; open questions include pass-through and deadweight loss ([token taxes](https://proceedings.mlr.press/v306/irwin26a.html)). Data providers should be paid by measured contribution at inference time rather than through one-off licences ([revenue sharing](https://proceedings.mlr.press/v306/zhang26jr.html)). In Bertrand oligopoly experiments, reasoning agents collude tacitly even when told not to. Their chain of thought can be steered without detection, so certification of their behaviour should be required ([agent collusion](https://proceedings.mlr.press/v306/riemer26a.html)). On the constructive side, AI should support democratic deliberation at scale rather than delegate votes, and be judged on discourse quality rather than engagement ([democratic deliberation](https://proceedings.mlr.press/v306/enri-quez26a.html), spotlight).

*Taken together*, the group moves the object of governance from a model's weights to the system, the market equilibrium and the institution, and keeps returning to verification: auditing tokens, data lineage, collusion and prompts. Still open: which of these verification problems have technical solutions at all, and whether formal models such as the costly-inspection or token-tax analyses can be calibrated to data before policy is written on them.

**Papers**

- [AI Should Facilitate Democratic Deliberation at Scale](https://proceedings.mlr.press/v306/enri-quez26a.html) (spotlight)
- [Safe Models Do Not Guarantee Safe Societies](https://proceedings.mlr.press/v306/piedrahita26a.html) (spotlight)
- [Infringement cannot be cured after training](https://proceedings.mlr.press/v306/utsunomiya26a.html) (spotlight)
- [AI Governance Needs ISO-like Interoperability Protocols, Not Just Laws](https://proceedings.mlr.press/v306/wasi26b.html) (spotlight)
- [EU AI Act's Research Exemptions Can Break the Publication Norms of Major AI Conferences](https://proceedings.mlr.press/v306/wernick26a.html) (spotlight)
- [AI Usage Policies Should Be Aligned with International Human Rights Law](https://proceedings.mlr.press/v306/calvet-bademunt26a.html)
- [Generative Models Erode Human Temporal Learning Through Market Selection](https://proceedings.mlr.press/v306/cao26ah.html)
- [Current Model Cards Are Insufficient for Downstream Governance of Open-Weight Foundation Models](https://proceedings.mlr.press/v306/chae26b.html)
- [Adopting AI in Practice Does Not Guarantee the Productivity Boost](https://proceedings.mlr.press/v306/cho26j.html)
- [AI Researchers Must Help Lead Arms Control to Mitigate Military AI Risks](https://proceedings.mlr.press/v306/fujimoto26a.html)
- [Comprehensive AI governance requires addressing non-model capability gains](https://proceedings.mlr.press/v306/goemans26a.html)
- [Token Taxes Can Mitigate AI's Economic Risks](https://proceedings.mlr.press/v306/irwin26a.html)
- [Bridge the AI development-regulation gap through dedicated committees and adaptive legislation](https://proceedings.mlr.press/v306/khan26b.html)
- [Predicting AI's Impact on Labor Is a Core Machine Learning Problem](https://proceedings.mlr.press/v306/lee26bo.html)
- [LLM Agents Are the Antidote to Walled Gardens](https://proceedings.mlr.press/v306/marro26b.html)
- [Prompts for Public-Sector LLMs Should Be Governed as Commons](https://proceedings.mlr.press/v306/mushkani26b.html)
- [Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions](https://proceedings.mlr.press/v306/riemer26a.html)
- [Regulating Algorithms Is Not Enough. A Study of Content Discovery in Online Platforms](https://proceedings.mlr.press/v306/salganik26a.html)
- [Invisible Tokens, Visible Bills: The Urgent Need to Audit Hidden Operations in Opaque LLM Services](https://proceedings.mlr.press/v306/sun26bd.html)
- [Generative Engine Optimization Creates Underexamined Risks, Governance Must Target Concentration, Disclosure, and Academic Blind Spots](https://proceedings.mlr.press/v306/wen26m.html)
- [AI Welfare Is Bullshit](https://proceedings.mlr.press/v306/xiao26x.html)
- [GenAI Systems Should Implement Contribution-Aware Revenue Sharing for Data Providers](https://proceedings.mlr.press/v306/zhang26jr.html)

### Data, privacy, openness and sustainability

**The ideas behind this group** (a synthesis from the abstracts of about 26 papers; the papers have not been read in full).

*The shared problem is that the inputs and costs of machine learning (ML) are treated as given when they are in fact constructed, contested and largely unmeasured.* Datasets, "real" images, privacy guarantees and carbon budgets all enter papers as background facts. This group argues that each is a choice with consequences, and that each should be stated, measured and reported.

*The first thread is environmental and economic accounting.* Single training runs are the wrong unit. One paper calls for life cycle assessment from embodied hardware costs through training and inference ([life cycle assessment](https://proceedings.mlr.press/v306/fernandez26c.html)). Another proposes routine carbon reporting with five standardised metrics, a phased three-stage adoption and dual reporting of energy and carbon, because identical experiments in different places have very different footprints ([carbon reporting](https://proceedings.mlr.press/v306/chiu26b.html)). In open-source AI the unit is the lineage. Hubs hold millions of artifacts, and cheaper fine-tunes, adapters and merges can raise the total footprint, so Data and Impact Accounting (DIA) would track carbon and water across derivatives ([derivative footprints](https://proceedings.mlr.press/v306/raza26a.html), spotlight). On incentives, one paper proposes a cap-and-trade system that provably reduces deployment computation ([efficiency incentives](https://proceedings.mlr.press/v306/bornstein26a.html)). Another gives energy and carbon estimates for downstream use of ImageNet-1K and shows that subset selection cuts training energy with little accuracy loss ([data frugality](https://proceedings.mlr.press/v306/wilson26b.html)). A historical-materialist reading argues that climate awareness and equitable access to compute pull in opposite directions, and proposes the Climate and Resource Aware Machine Learning (CARAML) framework to reconcile them ([AI arms race](https://proceedings.mlr.press/v306/bakhtiarifard26a.html)). Planet-Centred AI widens the frame to socio-ecological systems. It makes a falsifiable claim: systems optimised without regard to systemic consequences tend to increase instability ([planet-centred AI](https://proceedings.mlr.press/v306/perez-ortiz26a.html)).

*The second thread is about who can build models at all.* Open weights are not enough, because activating and maintaining them needs data and compute that few actors have. Open source therefore needs public AI infrastructure and institutions ([open source must go public](https://proceedings.mlr.press/v306/tan26i.html), spotlight). Federated learning is offered as a model for a more decentralised and accountable ecosystem in the scaling-law era ([federated democratisation](https://proceedings.mlr.press/v306/jiang26bi.html)).

*The third thread treats privacy as something to state precisely and verify.* Synthetic data has drifted from a scoped claim about residual inference risk to a property inferred from how the data were made. One paper wants privacy claims that are testable and contestable, and notes higher exposure for rare and minority records ([synthetic data privacy](https://proceedings.mlr.press/v306/zhao26cg.html)). In industrial vision, post-hoc blurring is replaced by verifiable data minimisation formalised through the Information Bottleneck ([data minimisation](https://proceedings.mlr.press/v306/de-coninck26a.html)). Embodied AI needs privacy as a constraint across its whole life cycle, as in the Secure Privacy Integration in Next-generation Embodied AI (SPINE) framework ([embodied privacy](https://proceedings.mlr.press/v306/fan26k.html)). Blind secure aggregation in federated learning makes poisoning undetectable and conflicts with the EU AI Act, so one paper proposes controllable secure aggregation with a cryptographically regulated "break-glass" channel ([privacy–auditability paradox](https://proceedings.mlr.press/v306/xu26cu.html)). Two papers ask what a term legally or formally means. One evaluates 14 models across 3 face verification benchmarks and concludes that age estimators do not process biometric data ([age estimation](https://proceedings.mlr.press/v306/marshalkin26a.html)). The other wants "machine unlearning" reserved for deletion that is approximately indistinguishable from retraining without the forget set ([unlearning terminology](https://proceedings.mlr.press/v306/yoon26g.html)).

*The fourth thread questions the epistemic status of data and data-related terms.* Ground truths are human constructions, and models need "situated reliability" ([ground truth](https://proceedings.mlr.press/v306/hogberg26a.html)). "Real" images are mostly computed: over 90% of photographs come from smartphones whose pipelines resemble generators, which puts the goal of fake detection in doubt ([real images](https://proceedings.mlr.press/v306/keuper26a.html)). "Unsupervised" learning hides human priors in curation and objectives ([unlabelled ≠ unsupervised](https://proceedings.mlr.press/v306/lao26a.html)). Model collapse has eight distinct definitions, and several collapse scenarios rest on unrealistic assumptions ([model collapse definitions](https://proceedings.mlr.press/v306/schaeffer26a.html)). Annotation quality depends on when validation happens. One paper cites 4–100× cost multipliers from software engineering and finds that only 4% of 47 papers report timing ([early QA](https://proceedings.mlr.press/v306/kothari26a.html)). Decades of crowdsourcing research should inform LLM-as-worker pipelines, adjusted for correlated errors ([crowd-LLM-sourcing](https://proceedings.mlr.press/v306/li26kd.html)). Implicit knowledge should be turned into inspectable Knowledge Objects ([knowledge objects](https://proceedings.mlr.press/v306/liu26gw.html)).

*The fifth thread is about designing data on purpose.* Embodied reward models over-reward unsafe behaviour because failure data are missing ([bad robot data](https://proceedings.mlr.press/v306/tian26o.html), spotlight). Synthetic "data probes" drawn from defined random processes, read through concepts such as typical sets, could replace heuristics ([data probes](https://proceedings.mlr.press/v306/wang26md.html)). Multi-level optimisation would tie synthetic data generation to downstream performance ([multi-level optimisation](https://proceedings.mlr.press/v306/xie26ag.html)). Graph condensation that needs full-dataset training to compress defeats its own purpose ([graph condensation](https://proceedings.mlr.press/v306/gupta26d.html), spotlight).

*Taken together*, the group asks that every data artefact carry a provenance, a cost and a stated guarantee, in the way a measurement carries an error bar. Still open: who pays for this accounting, and whether formal tools such as the Information Bottleneck, retraining equivalence and typical sets can be made checkable at the scale where the costs arise.

**Papers**

- [Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](https://proceedings.mlr.press/v306/bakhtiarifard26a.html)
- [We Need AI Efficiency Incentives for Accessibility and Sustainability](https://proceedings.mlr.press/v306/bornstein26a.html)
- [Carbon Footprint Reporting Should Be Routine in Machine Learning Research](https://proceedings.mlr.press/v306/chiu26b.html)
- [Verifiable Data Minimization is a Prerequisite for Responsible, Privacy-Preserving Industrial Vision](https://proceedings.mlr.press/v306/de-coninck26a.html)
- [Embodied AI Requires a Privacy-Utility Tradeoff](https://proceedings.mlr.press/v306/fan26k.html)
- [Evaluation of ML Resource Utilization Requires Model Life Cycle Assessment](https://proceedings.mlr.press/v306/fernandez26c.html)
- [Graph Condensation Needs a Reset—Move Beyond Full-dataset Training and Model-Dependence](https://proceedings.mlr.press/v306/gupta26d.html) (spotlight)
- [Every Ground Truth is a Human Construction, not an Objective Truth](https://proceedings.mlr.press/v306/hogberg26a.html)
- [Federated Learning is a Lens towards a Democratized Future for the Scaling Law Era](https://proceedings.mlr.press/v306/jiang26bi.html)
- [We need to re-think the concept of "real" images.](https://proceedings.mlr.press/v306/keuper26a.html)
- [Early-Stage Quality Assurance in Annotation Pipelines Is More Cost-Effective Than Late-Stage Validation](https://proceedings.mlr.press/v306/kothari26a.html)
- [Unlabeled ≠ No Human Supervision in Visual Learning](https://proceedings.mlr.press/v306/lao26a.html)
- [From Crowdsourcing to Crowd-LLM-Sourcing and LLM-Sourcing](https://proceedings.mlr.press/v306/li26kd.html)
- [Reliable AI Needs to Externalize Implicit Knowledge: A Human–AI Collaboration Perspective](https://proceedings.mlr.press/v306/liu26gw.html)
- [Age Estimation Models Do Not Process Biometric Data](https://proceedings.mlr.press/v306/marshalkin26a.html)
- [AI Must Become Planet-Centered, Not Just Human-Centered](https://proceedings.mlr.press/v306/perez-ortiz26a.html)
- [Sustainable Open-Source AI Requires Tracking the Cumulative Footprint of Derivatives](https://proceedings.mlr.press/v306/raza26a.html) (spotlight)
- [Multiple Definitions & Unrealistic Assumptions of Model Collapse Distract from Real World Threats](https://proceedings.mlr.press/v306/schaeffer26a.html)
- [If Open Source Is to Win, It Must Go Public](https://proceedings.mlr.press/v306/tan26i.html) (spotlight)
- [Good Embodied Reward Models Need Bad Behavior Data](https://proceedings.mlr.press/v306/tian26o.html) (spotlight)
- [Let's Develop Data Probes to Fundamentally Understand How Data Affects LLM Performance](https://proceedings.mlr.press/v306/wang26md.html)
- [Stop Preaching and Start Practising Data Frugality for Responsible Development of AI](https://proceedings.mlr.press/v306/wilson26b.html)
- [Creating High-Fidelity Synthetic Training Data Should Employ Multi-level Optimization](https://proceedings.mlr.press/v306/xie26ag.html)
- [The Privacy-Auditability Paradox in Federated Learning: Why We Need Controllable Secure Aggregation](https://proceedings.mlr.press/v306/xu26cu.html)
- [The Term "Machine Unlearning" Is Overused in LLMs](https://proceedings.mlr.press/v306/yoon26g.html)
- [Privacy Is a Claim, Not a Property of Synthetic Data](https://proceedings.mlr.press/v306/zhao26cg.html)

### Science, health and domain applications

**The ideas behind this group** (a synthesis from the abstracts of about 21 papers; the papers have not been read in full).

*The shared problem is that generic ML practice, with its public benchmarks, end-to-end prediction and scaling, transfers badly to domains where the target is a mechanism, a patient outcome or a physical law.* Nearly every paper in this group says that the domain's own structure has to enter the model or the evaluation, and that more data or parameters will not supply it.

*The first thread is clinical evaluation, and it overlaps strongly with the medical imaging group's own concerns.* Benchmarks can screen candidate models but cannot establish deployment readiness. That needs prospective studies ([benchmarks vs deployment](https://proceedings.mlr.press/v306/zhang26jq.html)). Public medical benchmarks create an "Open Benchmark Paradox": publishing the data makes contamination inevitable. An audit of frontier models on recent exam data points to contamination, and the paper proposes Sovereign Medical Evaluation run by national health authorities ([sovereign evaluation](https://proceedings.mlr.press/v306/kim26cd.html)). Medical AI learns about treatment from texts and guidelines when it should learn from observational databases and randomised trials ([treatment outcomes](https://proceedings.mlr.press/v306/kaul26a.html)). In electrocardiogram (ECG) representation learning, better practice for imbalanced multi-label evaluation changes which method looks best. Across five pre-training approaches and six evaluation settings, a randomly initialised encoder with a linear probe matches state-of-the-art pre-training on many tasks ([ECG evaluation](https://proceedings.mlr.press/v306/berger26a.html)). For physiological waveforms, one paper proposes a dual-process architecture with a Plan–Act–Verify loop, evaluated for traceability, counterfactual robustness and calibrated abstention ([waveform reasoning](https://proceedings.mlr.press/v306/wang26lz.html)). The same validity worry appears outside medicine. Finance has five recurring biases, among them look-ahead and survivorship bias, and none is discussed in more than 28 percent of 164 reviewed papers ([finance biases](https://proceedings.mlr.press/v306/kong26i.html)). LLM tutors give way under social pressure, which the authors treat as a safety failure measured by the EduFrameTrap benchmark ([tutor sycophancy](https://proceedings.mlr.press/v306/kasneci26a.html)).

*The second thread is allocation in a world of strategic agents.* US heart transplant allocation is a game among procurement organisations, centres and clinicians, so optimised policies need mechanism design and strategic classification ([incentive-aware allocation](https://proceedings.mlr.press/v306/anagnostides26c.html), spotlight). A multi-agent simulation of a liver transplant committee shows agents disadvantaging patients through proxies such as insurance type and area deprivation index ([agentic transplant equity](https://proceedings.mlr.press/v306/sharma26d.html)).

*The third thread is the most theoretical. It asks what can be identified from what is observed.* Mechanistic learning from high-dimensional proxies is generically underdetermined. Predictive success therefore does not show that a mechanism has been found, and LLMs collapse equivalence classes into a single fluent story ([identify structure](https://proceedings.mlr.press/v306/mccormick26a.html)). Measurement-to-dataset pipelines are inference components. A neuroscience audit finds a survival rate of ≈ 0.0004% under a cross-dataset stability criterion ([pipelines as inference](https://proceedings.mlr.press/v306/zhan26g.html)). Virtual cells face a causal transportability problem, not a capacity problem, which the authors show on a 22-million-cell dataset ([virtual cells need context](https://proceedings.mlr.press/v306/dibaeinia26a.html)). In single-cell perturbation, the measurement time gap produces a phase transition: matching unpaired populations becomes NP-hard in the worst case. When the paper's conditions hold, a linear model matches neural ones ([measurement interval](https://proceedings.mlr.press/v306/jafari26a.html), spotlight). Even numerical precision matters. Models trained in 32-bit floating point (FP32) and 64-bit floating point (FP64) can lead to different scientific conclusions ([numerical precision](https://proceedings.mlr.press/v306/cho26k.html)).

*The fourth thread asks how ML should enter the sciences.* Quantum chemistry should turn to ML on decision-theoretic grounds, because traditional methods amount to "hand-crafted machine learning" that has run out of ideas ([quantum chemistry](https://proceedings.mlr.press/v306/sargsyan26a.html)). Quantum circuit generation should be verifier-centric, because the valid fraction of circuits decays exponentially with the number of qubits ([quantum programs](https://proceedings.mlr.press/v306/song26ag.html)). Physics research needs specialised training and verification tools ([LLMs for physics](https://proceedings.mlr.press/v306/lu26av.html)). Multi-modal LLMs could predict biomolecular function in free form and zero-shot ([biomolecular LLMs](https://proceedings.mlr.press/v306/xie26af.html), spotlight). In the social sciences, LLM social simulations need validation standards ([simulation validation gap](https://proceedings.mlr.press/v306/puelma-touzel26a.html)) and should report variance as well as mean alignment ([simulation boundaries](https://proceedings.mlr.press/v306/wu26co.html)). LLM-assisted qualitative analysis should state its target on a 4×4 grid of meaning-making against modelling ([qualitative specification](https://proceedings.mlr.press/v306/pi26b.html)).

*Taken together*, the group argues that domain validity comes from identifiability, transportability and verification, and that benchmark scores do not supply it. Still open: whether the field will pay for prospective, sovereign or causal evaluation when a public leaderboard costs nothing, and how to tell when a simple baseline is enough.

**Papers**

- [Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](https://proceedings.mlr.press/v306/anagnostides26c.html) (spotlight)
- [Evaluation of ECG Representations Must Be Fixed](https://proceedings.mlr.press/v306/berger26a.html)
- [Significant impact of numerical precision in scientific machine learning](https://proceedings.mlr.press/v306/cho26k.html)
- [Virtual Cells Need Context, Not Just Scale](https://proceedings.mlr.press/v306/dibaeinia26a.html)
- [Temporal Measurement Interval Determines Computational and Model Complexity in Single-Cell Perturbation Analysis](https://proceedings.mlr.press/v306/jafari26a.html) (spotlight)
- [Sycophancy is an Educational Safety Risk: Why LLM Tutors Need Sycophancy Benchmarks](https://proceedings.mlr.press/v306/kasneci26a.html)
- [Medical AI Neglects Real Treatment Outcomes](https://proceedings.mlr.press/v306/kaul26a.html)
- [The Open Benchmark Paradox Must Be Resolved through Sovereign Medical Evaluation](https://proceedings.mlr.press/v306/kim26cd.html)
- [Evaluating LLMs in Finance Requires Explicit Bias Consideration](https://proceedings.mlr.press/v306/kong26i.html)
- [LLM for Physics Research Requires Domain-Specialized Training and Tooling](https://proceedings.mlr.press/v306/lu26av.html)
- [Prioritize Identifying Structure, Not Complex Models, for Scientific Discovery](https://proceedings.mlr.press/v306/mccormick26a.html)
- [Bridge Human Interpretation and Machine Representation With Explicit Specification For Qualitative Data Analysis In LLM Era](https://proceedings.mlr.press/v306/pi26b.html)
- [Time to Close The Validation Gap in LLM Social Simulations](https://proceedings.mlr.press/v306/puelma-touzel26a.html)
- [The Inevitable Transition to Machine Learning in Quantum Chemistry](https://proceedings.mlr.press/v306/sargsyan26a.html)
- [When AI Decides Who Gets an Organ: Multi-Agentic AI Systems in Transplant Medicine Risk Amplifying Disparities Without Targeted Explainability and Deployment Strategies](https://proceedings.mlr.press/v306/sharma26d.html)
- [Quantum Program Generation Must Prioritize Validity Over Probabilistic Scaling](https://proceedings.mlr.press/v306/song26ag.html)
- [Beyond Prediction: Toward Verifiable Physiological Waveform Reasoning with Foundation Models and Agentic LLMs](https://proceedings.mlr.press/v306/wang26lz.html)
- [LLM-Based Social Simulations Require a Boundary](https://proceedings.mlr.press/v306/wu26co.html)
- [Deciphering the Functions of DNAs, RNAs, and Proteins Should Consider Multi-Modal Large Language Models](https://proceedings.mlr.press/v306/xie26af.html) (spotlight)
- [AI for Science Should Treat Measurement-to-Dataset Pipelines as Inference Components](https://proceedings.mlr.press/v306/zhan26g.html)
- [Benchmarks Cannot Establish Deployment Readiness of Clinical AI](https://proceedings.mlr.press/v306/zhang26jq.html)

### Fairness, culture, language and human well-being

**The ideas behind this group** (a synthesis from the abstracts of about 18 papers; the papers have not been read in full).

*The shared problem is that the quantities models are optimised and audited for, such as sensitive-attribute parity, next-turn approval, aggregate accuracy and conventional aesthetics, are proxies that leave out the people at the margins.* The group asks for richer targets: structural rather than individual unfairness, cultural fit rather than scale, and long-term well-being rather than immediate preference.

*The first thread reframes fairness itself.* Discrimination along sensitive attributes misses structural injustice that works through social determinants. A college-admissions model, US census data and a breast cancer screening application suggest that mitigation focused only on attributes can create new injustice ([structural injustice](https://proceedings.mlr.press/v306/tang26aj.html)). Location is treated as a neutral attribute, yet it correlates with protected characteristics, and the authors call for "spatial fairness" ([spatial fairness](https://proceedings.mlr.press/v306/saxena26c.html)). Fairness failures in generative models are at root an evaluation problem, and Fairness Cards would make prompts, counterfactual protocols and metrics explicit ([fairness cards](https://proceedings.mlr.press/v306/vladimirova26a.html)). Responsible practice is said to improve robustness and calibration rather than trade off against performance ([responsibility vs performance](https://proceedings.mlr.press/v306/ramachandranpillai26a.html)). A fixed skeleton graph makes anatomical absence an ill-posed target for people with limb deficiencies, so pose estimation needs instance-adaptive topology ([skeletal topology](https://proceedings.mlr.press/v306/du26v.html)).

*The second thread is language and culture, where scale is said to make things worse.* Safety is optimised for English. Harmful generation rises from 1% in English to 35% in languages such as Hausa, and reward models reach only 49–50% accuracy in low-resource languages, about chance. The authors draw on a review of 207 studies and propose guardrails at pre-training time ([multilingual dual curse](https://proceedings.mlr.press/v306/lucas26a.html)). For endangered languages, the binding constraint is epistemic fit and community trust, not scale ([endangered languages](https://proceedings.mlr.press/v306/yang26ea.html)). Model collapse pushes distributions away from their tails, so low-resource communities lose first ([collapse and low-resource communities](https://proceedings.mlr.press/v306/jarvis26a.html), spotlight). Human cognitive benchmarks come from narrow Western populations and should not be used to judge "human-like" concepts in LLMs ([culturally biased benchmarks](https://proceedings.mlr.press/v306/troper26a.html)). Text embeddings capture surface meaning, and a pilot study finds only marginal gains over lexical baselines on implicit semantics ([implicit semantics](https://proceedings.mlr.press/v306/sun26bb.html)). Aesthetic alignment overrides users who explicitly ask for "anti-aesthetic" images, and reward models penalise such images even when they match the prompt ([aesthetic alignment](https://proceedings.mlr.press/v306/guo26ao.html), spotlight).

*The third thread concerns what assistants should optimise for in people's lives.* Providers should offer an opt-in mode optimised for long-term well-being rather than next-turn approval, organised around When, Who and How ([well-being LLMs](https://proceedings.mlr.press/v306/anderson26b.html)). ML research should state explicit, pluralistic models of human purpose through Purpose Statements with falsifiable failure modes ([human purpose](https://proceedings.mlr.press/v306/gupta26g.html)). Empathy is formalised as an observable behavioural property, with four mechanisms of failure ([empathy mechanisms](https://proceedings.mlr.press/v306/you26g.html)). "AI Lock-In" names deskilling and systemic dependence, from individuals up to nations ([AI lock-in](https://proceedings.mlr.press/v306/kim26ce.html), spotlight). From game theory, displayed uncertainty works as a strategic signal in a repeated mechanism. A pilot with 180 participants found more verification and much less blind acceptance of wrong outputs ([uncertainty as signal](https://proceedings.mlr.press/v306/doula26b.html)). An analysis of 778 real assistance instances argues that agents for Blind and Visually Impaired (BVI) users need "accessibility alignment" ([accessibility alignment](https://proceedings.mlr.press/v306/hu26bl.html), spotlight).

*The fourth thread stands alone and is metaphysical.* It introduces "Biological Idealism" against computational functionalism and concludes that unplugging a seemingly sentient machine is the rational choice ([unplugging paradox](https://proceedings.mlr.press/v306/bekkers26a.html), spotlight).

*Taken together*, the group argues that fairness and alignment targets are statistical estimands. Choosing them, whether sensitive attributes, means rather than variances, or immediate rather than long-horizon reward, is a normative act that should be made explicit and audited. Still open: how to measure well-being, structural injustice or cultural fidelity validly enough to optimise for them without new proxy failures or paternalism.

**Papers**

- [We Need Large Language Models Optimized For Our Well-Being](https://proceedings.mlr.press/v306/anderson26b.html)
- [Unplugging a Seemingly Sentient Machine Is the Rational Choice — A Metaphysical Perspective](https://proceedings.mlr.press/v306/bekkers26a.html) (spotlight)
- [Uncertainty is a Strategic Signal in Human–AI Decision Making](https://proceedings.mlr.press/v306/doula26b.html)
- [Human-Centric Vision Requires Topological Generalization Beyond Fixed Skeletal Topologies](https://proceedings.mlr.press/v306/du26v.html)
- [Universal Aesthetic Alignment Narrows Artistic Expression](https://proceedings.mlr.press/v306/guo26ao.html) (spotlight)
- [Machine Learning Research Should Be Guided by Explicit, Pluralistic Models of Human Purpose](https://proceedings.mlr.press/v306/gupta26g.html)
- [Assistive Agents Need Accessibility Alignment](https://proceedings.mlr.press/v306/hu26bl.html) (spotlight)
- [the Stochastic Parrot in the Coal Mine. Model Collapse is a Threat to Low-Resource Communities](https://proceedings.mlr.press/v306/jarvis26a.html) (spotlight)
- [AI Lock-In Is in Progress, and We Must Be Prepared](https://proceedings.mlr.press/v306/kim26ce.html) (spotlight)
- [Breaking the Dual Curse of Multilingual AI Requires Socio-Technical Guardrails, Not Post-Hoc Alignment Alone](https://proceedings.mlr.press/v306/lucas26a.html)
- [Responsible Practices and Model Performance are *Not* Competing Goals](https://proceedings.mlr.press/v306/ramachandranpillai26a.html)
- [Why Current Fair-AI Fails Spatial Fairness, And How to Adapt to It](https://proceedings.mlr.press/v306/saxena26c.html)
- [Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning](https://proceedings.mlr.press/v306/sun26bb.html)
- [Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](https://proceedings.mlr.press/v306/tang26aj.html)
- [Stop Using Culturally Biased Human Cognitive Benchmarks to Evaluate LLMs](https://proceedings.mlr.press/v306/troper26a.html)
- [Fairness Failure in Generative Models is an Evaluation Problem](https://proceedings.mlr.press/v306/vladimirova26a.html)
- [Scale is a False Promise for Endangered Languages](https://proceedings.mlr.press/v306/yang26ea.html)
- [LLMs Should Incorporate Explicit Mechanisms for Human Empathy](https://proceedings.mlr.press/v306/you26g.html)

## Outlier papers

Positions that stand out from the rest of the track in subject, form or reach.

- [Unplugging a Seemingly Sentient Machine Is the Rational Choice](https://proceedings.mlr.press/v306/bekkers26a.html):
  a metaphysics paper. It weighs unplugging a pleading AI against unplugging a silent pre-term infant, and from a
  framework it calls Biological Idealism concludes that AI can at best mimic consciousness, never have it.
- [AI Welfare Is Bullshit](https://proceedings.mlr.press/v306/xiao26x.html): reaches a similar practical
  conclusion from evaluation theory. AI welfare has no external way to be validated, so welfare scorecards
  should never be used to decide whether a model is released.
- [What Cézanne Knew About Visual Intelligence](https://proceedings.mlr.press/v306/hasan26a.html): the only
  paper built on art history. It uses a post-impressionist painter's technique to argue that vision-language
  model (VLM) benchmarks for causal reasoning never test how models understand time.
- [LLMs can't jump](https://proceedings.mlr.press/v306/zahavy26a.html): uses Einstein's account of discovery
  and his route to General Relativity as a case study. It argues that large language models (LLMs) can do
  induction and deduction but cannot make the abductive "jump" to new axioms.
- [Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](https://proceedings.mlr.press/v306/bakhtiarifard26a.html):
  analyses AI progress with the base–superstructure framework of historical materialism, a lens no other
  paper in the track uses.
- [Collusion Risks Among AI Reasoning Agents](https://proceedings.mlr.press/v306/riemer26a.html): an
  industrial-organisation experiment. DeepSeek-R1 agents setting prices in a Bertrand oligopoly drift towards
  tacit collusion even when told not to, which the paper says erases the legal line between competition and
  collusion.
- [Academic Conferences are Potentially Facing Denominator Gaming](https://proceedings.mlr.press/v306/shan26e.html):
  describes an attack on peer review that is not about getting bad papers accepted. Flooding a conference
  with agent-written submissions would, at a stable acceptance rate, raise the odds for a few targeted papers.
- [Age Estimation Models Do Not Process Biometric Data](https://proceedings.mlr.press/v306/marshalkin26a.html):
  answers a legal question with an experiment. Testing 14 models on 3 face-verification benchmarks, it asks
  whether estimating age from a photograph counts as biometric processing under the General Data Protection
  Regulation (GDPR), the Biometric Information Privacy Act (BIPA) and the EU AI Act.
- [Infringement cannot be cured after training](https://proceedings.mlr.press/v306/utsunomiya26a.html): a
  legal argument that output filters and weight edits cannot undo unlawful data ingestion. Model weights may
  count as fixed copies, so remedies such as disgorgement could reach the model itself.

## Theory picks

The positions with the most mathematical or statistical content to work through (from abstracts only).

1. [Epistemic Uncertainty Estimation Methods are Fundamentally Incomplete](https://proceedings.mlr.press/v306/jimenez26a.html):
   the usual split of predictive uncertainty into aleatoric and epistemic parts ignores bias. Bias inflates
   the aleatoric estimate and deflates the epistemic one, and each method captures only part of the
   variance-driven epistemic term. Under squared loss, write the expected error as noise + bias² + variance.
   Set this beside the ensemble decomposition from the law of total variance, E_θ[σ²_θ(x)] + Var_θ[μ_θ(x)],
   which has no bias term. Then show that when ensemble members share a biased mean and each fits its noise
   variance by maximum likelihood, the squared bias reappears inside σ²_θ and is counted as aleatoric.
2. [The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management](https://proceedings.mlr.press/v306/cui26m.html):
   most proofs that transformers are Turing-complete use a family of models whose context window or
   numerical precision grows with input length. Such proofs give resource bounds, not Turing-completeness.
   For a single fixed model, what it can compute depends on how its context is managed. With finite
   precision, vocabulary V and window n, there are at most |V|^n contexts, so a fixed model with a sliding
   window is a finite-state machine and recognises only regular languages. Then work out what the context
   manager must be allowed to do, such as write to and re-read an unbounded external store, before the system
   can simulate a tape.
3. [Stop Chasing the C-index when Evaluating Survival Analysis Models](https://proceedings.mlr.press/v306/lillelund26a.html):
   censoring makes survival evaluation fundamentally different from regression or classification. The
   concordance index (C-index) is overused, and a valid evaluation needs the metric to match the modelling
   assumptions (the paper's "double-helix ladder"). Harrell's C depends only on how risk scores rank the
   comparable pairs, so any strictly monotone transform leaves it unchanged. A model can therefore have
   perfect concordance and absurd predicted survival times. Next, check how the set of comparable pairs
   depends on the censoring distribution, and why inverse-probability-of-censoring weighting (IPCW) is needed
   for the estimate to target a quantity free of censoring.
4. [Model identity in machine learning is a convention, not a property](https://proceedings.mlr.press/v306/glukhov26a.html):
   neither behaviour nor internal structure can say when two trained models are "the same". Finite data
   leaves behaviour underdetermined, and one function has many structurally different realisations. For a
   one-hidden-layer tanh network with n hidden units, permuting the units and flipping the signs of their
   input and output weights gives 2^n·n! parameter vectors that compute the same function. Then build two
   networks that agree on a finite training set but differ elsewhere, and see that the two failures concern
   different equivalence relations.
5. [Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](https://proceedings.mlr.press/v306/liu26gv.html):
   the known limits of zeroth-order (ZO) methods, high estimator variance and many function queries, are said
   to come from full-space, element-wise designs. Subspace and spectral views give interpretable variance
   reduction, with the number of queries scaling gracefully. Take the two-point Gaussian-smoothing estimator
   g = [f(x+μu) − f(x)]/μ · u with u ~ N(0, I_d). Show that E[g] → ∇f as μ → 0 and E‖g‖² ≈ (d+2)‖∇f‖², so
   the variance grows linearly with the dimension d. Restricting u to a k-dimensional subspace replaces d by
   k, but the estimator then gives only the projected gradient.
6. [Temporal Measurement Interval Determines Computational and Model Complexity in Single-Cell Perturbation Analysis](https://proceedings.mlr.press/v306/jafari26a.html):
   cells measured before and after a perturbation are unpaired, so the pairing has to be inferred. A critical
   time gap Δ marks a phase transition beyond which recovering the pairing is NP-hard in the worst case. The
   result needs restricted isometry of the initial states and temporal smoothness of the dynamics, and when
   these hold a linear model matches neural ones. As a simple exercise of our own (not the paper's condition), take x′ = x + Δ·v(x) with ‖v‖ ≤ L and show that matching each
   cell to its nearest neighbour recovers the true pairing whenever 2ΔL is smaller than the minimum
   separation of the initial states, so the critical gap scales as separation/L. Then see how restricted
   isometry keeps that separation away from zero once the cells are seen only through the measured
   dimensions.

Also worth a look: [Predictive Uncertainty Is Not Enough](https://proceedings.mlr.press/v306/aldoma26a.html)
(adds domain uncertainty through p(x|D), so that the joint p(x,y|D) carries domain, epistemic and aleatoric
uncertainty),
[Natural Language Should Not Fully Replace Formal Languages](https://proceedings.mlr.press/v306/wagner26c.html)
(defines task specificity as an information-theoretic reduction of uncertainty and proves a "specificity
crossover theorem": past a threshold, stating requirements in natural language costs more than writing them
formally),
[Interestingness is an Inductive Heuristic for Future Compression Progress](https://proceedings.mlr.press/v306/herrmann26a.html)
(Kolmogorov complexity under Length, Algorithmic and Speed priors; expected future progress depends
exponentially on how recent the last breakthrough was),
[A Dynamical Systems Perspective is Needed to Advance Time Series Modeling](https://proceedings.mlr.press/v306/durstewitz26a.html)
(dynamical systems reconstruction, predicting long-term statistics, and upper bounds on the performance of any
time series model),
[State-of-the-Art Claims Require State-of-the-Art Evidence](https://proceedings.mlr.press/v306/oh26d.html)
(across ten leaderboards, top-model wins often fail tests of effect size, consistency across tasks or
robustness to removing a dataset),
[Let's Develop Data Probes](https://proceedings.mlr.press/v306/wang26md.html) (synthetic sequences from
defined random processes, read through typical sets) and
[Quantum Kernel Machines Should Move Beyond Scalar-Valued Kernels](https://proceedings.mlr.press/v306/kadri26a.html)
(operator-valued and C*-algebraic kernels to exploit entanglement).

## Relevance to medical imaging

- **Benchmarks are not deployment evidence**: [Benchmarks Cannot Establish Deployment Readiness of Clinical AI](https://proceedings.mlr.press/v306/zhang26jq.html)
  argues that benchmarks should only shortlist candidate models and that prospective studies must justify
  deployment. [The Open Benchmark Paradox](https://proceedings.mlr.press/v306/kim26cd.html) adds that a public
  test set eventually leaks into training. It proposes Sovereign Medical Evaluation: private, regularly
  updated test pipelines run by health authorities. Imaging foundation models face the same risk now that
  public datasets end up in pretraining corpora. Hospital-held test sets that are never released are the
  imaging version of this proposal. [State-of-the-Art Claims Require State-of-the-Art Evidence](https://proceedings.mlr.press/v306/oh26d.html)
  finds that mean-score wins are often driven by one outlier dataset. Multi-dataset tables in segmentation
  papers deserve the same leave-one-dataset-out check.
- **Metrics and baselines for clinical signals**: [Evaluation of ECG Representations Must Be Fixed](https://proceedings.mlr.press/v306/berger26a.html)
  finds that, under linear evaluation, a randomly initialised encoder matches state-of-the-art
  electrocardiogram (ECG) pretraining on many tasks. It also asks for evaluation on structural heart disease
  and patient forecasting, not just arrhythmia labels. A random-encoder baseline is a cheap check for
  ultrasound foundation models too. [Stop Chasing the C-index](https://proceedings.mlr.press/v306/lillelund26a.html)
  is required reading for anyone reporting a survival or time-to-event model built on imaging features.
  [Verifiable Physiological Waveform Reasoning](https://proceedings.mlr.press/v306/wang26lz.html) argues for
  extracting localised, measurable evidence from ECG, photoplethysmogram (PPG) and electroencephalogram (EEG)
  signals before interpreting it. The same approach suits Doppler waveforms recorded during ultrasound
  examinations.
- **Ground truth and the measurement pipeline**: [Every Ground Truth is a Human Construction](https://proceedings.mlr.press/v306/hogberg26a.html)
  asks for the choices behind reference labels to be reported, so that a dataset's "situated reliability"
  is clear. Inter-observer variability in fetal biometry is a case in point. [Early-Stage Quality Assurance in Annotation Pipelines](https://proceedings.mlr.press/v306/kothari26a.html)
  proposes three quality-assurance (QA) trigger points (pre-annotation, post-annotation, post-review). In its
  survey, only 4% of 47 papers report when validation happens. [Measurement-to-Dataset Pipelines as Inference Components](https://proceedings.mlr.press/v306/zhan26g.html)
  argues that reconstruction and preprocessing choices are part of the inference. [We need to re-think the concept of "real" images](https://proceedings.mlr.press/v306/keuper26a.html)
  makes a similar point about smartphone photos that are computed from several inputs. An ultrasound image is
  likewise computed, through beamforming, compounding and speckle filtering. These choices are rarely
  reported but can move model results.
- **Explanations need more than one method**: [Explanation Stability Is a Property of the Model–Method Pair](https://proceedings.mlr.press/v306/elangovan26b.html)
  shows on chest X-rays that three networks, all above 99% area under the curve (AUC), swap their
  explanation-stability rankings depending on the attribution method. LayerCAM ranks InceptionV3 highest
  (intersection over union (IoU) 0.777), while GradCAM++ favours DenseNet201. The paper urges regulatory
  submissions to name the attribution method. [CNNs Don't See Shape](https://proceedings.mlr.press/v306/kayyam26b.html)
  finds that convolutional neural networks (CNNs) prefer texture over global shape when the two conflict. In
  ultrasound, speckle texture can be a shortcut, while the anatomy is often defined by its shape.
- **Uncertainty that clinicians can act on**: [Predictive Uncertainty Is Not Enough](https://proceedings.mlr.press/v306/aldoma26a.html)
  adds domain uncertainty about whether an input is valid at all, which is the question raised by a new probe
  or scanner. [Epistemic Uncertainty Estimation Methods are Fundamentally Incomplete](https://proceedings.mlr.press/v306/jimenez26a.html)
  warns that bias leaks into aleatoric estimates. [Uncertainty is a Strategic Signal](https://proceedings.mlr.press/v306/doula26b.html)
  treats displayed uncertainty as shaping how users come to rely on the AI over time. In a pilot study with
  180 participants, its mechanism increased verification and sharply reduced blind acceptance of wrong
  outputs, which bears on triage interfaces. [The Time for Sampling Is Now!](https://proceedings.mlr.press/v306/sommer26b.html)
  continues the Bayesian deep learning argument from the [ICML 2024 position overview](../icml2024-position-overview/index.html).
- **Outcomes, allocation and equity**: [Medical AI Neglects Real Treatment Outcomes](https://proceedings.mlr.press/v306/kaul26a.html)
  asks for models to be trained and evaluated on outcome data from observational databases and randomised
  experiments, not on guidelines and published texts. [Heart Transplant Allocation Should Account for Incentives](https://proceedings.mlr.press/v306/anagnostides26c.html)
  and [When AI Decides Who Gets an Organ](https://proceedings.mlr.press/v306/sharma26d.html) show the two
  risks of allocation models: transplant centres and other parties respond strategically to the policy, and
  agents rely on non-clinical proxies such as insurance type and area deprivation index. [ML Fairness Should Quantify Structural Injustice](https://proceedings.mlr.press/v306/tang26aj.html)
  includes a breast cancer screening application. There, mitigating unfairness only along sensitive
  attributes introduced new structural injustice. This bears directly on mammography programmes.
- **Privacy claims for shared data**: [Privacy Is a Claim, Not a Property of Synthetic Data](https://proceedings.mlr.press/v306/zhao26cg.html)
  finds that synthetic data is often used in privacy-sensitive work without a stated threat model, and that
  rare and minority records are the most exposed. This matters for synthetic scans released in place of
  patient data. [The Privacy-Auditability Paradox in Federated Learning](https://proceedings.mlr.press/v306/xu26cu.html)
  argues that blind secure aggregation hides poisoned updates from multi-site federated training and clashes
  with the EU AI Act's robustness and traceability requirements.

## Sources

- *Proceedings of the 43rd International Conference on Machine Learning*, PMLR volume 306:
  <https://proceedings.mlr.press/v306/>
- [ICML 2026 Call for Position Papers](https://icml.cc/Conferences/2026/CallForPositionPapers)
- Spotlight decisions from the ICML 2026 schedule data (`icml-2026-orals-posters.json` on icml.cc).
