---
title: "ICML 2025 Position Paper Track: topic map, outliers and theory picks"
authors: "Topic notes"
venue: "ICML 2025 (Position Paper Track)"
tags: [icml, position-papers, conference-overview, evaluation, measurement, learning-theory, causal-inference, ai-policy]
status: living
category: "Research practice & policy"
subcategory: "Conference overviews"
short_title: "ICML 2025 — position papers"
summary: "a map of the 73 papers in the ICML 2025 Position Paper Track (PMLR v267) in seven groups of arguments, with an idea-level synthesis of each, the outliers, the positions with the most mathematical content, and their relevance to medical imaging"
related: [icml2024-position-overview, neurips2025-position-overview, midl2025-overview]
---

## Why this page exists

ICML 2025 (Vancouver, 13–19 July 2025) ran its second **Position Paper Track**; the first was at
ICML 2024 ([overview](../icml2024-position-overview/index.html)). Position papers argue for a claim about the field rather than presenting a new method.
Unlike NeurIPS, ICML publishes them in the main proceedings,
[PMLR volume 267](https://proceedings.mlr.press/v267/), where every title begins "Position:".

The track has **73 papers: 12 orals, 12 spotlights and 49 posters**, identified from the PMLR
titles and cross-checked against the ICML 2025 schedule data, which tags each paper with the
OpenReview group it came from. Titles below drop the "Position:" prefix. The grouping and syntheses
are from **titles and abstracts only**; each paper is listed in one group, though the syntheses
mention related papers from other groups.

As with the [NeurIPS 2025 position track](../neurips2025-position-overview/index.html), the useful
questions are whether the argument is valid, whether the evidence supports the claim, and what
would change one's mind.

## Topic groups at a glance

| Group | Papers | What's in it |
|---|---|---|
| [Evaluation, measurement and peer review](#evaluation-measurement-and-peer-review) | 18 | Construct validity, measurement theory, psychometrics and animal cognition for evals, CLT error bars, train–test overlap, broken benchmarks, peer-review reform |
| [Understanding deep learning: theory and science](#understanding-deep-learning-theory-and-science) | 12 | Soft inductive biases and PAC-Bayes, compositional sparsity, neuroalgebraic geometry, solvable linear models, identifiability, OOD detection |
| [Probabilistic, causal and statistical foundations](#probabilistic-causal-and-statistical-foundations) | 8 | Causal inference as probabilistic modelling, prior-fitted networks, synthetic causal experiments, regret constants, uncertainty for agents, XAI as statistics |
| [Agents, reasoning and general intelligence](#agents-reasoning-and-general-intelligence) | 11 | Reward-based pretraining, metacognition, asymptotic analysis of agents, formal methods and proofs, authenticated delegation, AGI as a goal |
| [Scale, data and the model supply chain](#scale-data-and-the-model-supply-chain) | 8 | Downscaling, the cost of training data, annotation incentives, signing models, licences and copyright |
| [Safety and security](#safety-and-security) | 5 | What certified robustness guarantees, knowledge-editing attacks, antifragile safety, future of work, gradual disempowerment |
| [Society, politics and governance](#society-politics-and-governance) | 11 | Lessons from social-media regulation, consumer protection, due process, democratic participation, political neutrality, race categories, health communication |

**Compared with NeurIPS 2025:** both tracks are dominated by evaluation and measurement, and both
borrow heavily from the social sciences. ICML's track is larger and much more technical: it has a
whole group of positions on the theory of deep learning (compositional sparsity, algebraic
geometry, PAC-Bayes, solvable linear models) and another on probabilistic and statistical
foundations, which NeurIPS largely lacks. NeurIPS leans more towards regulation, privacy and
concrete harms.

## Topic groups

Every paper is listed under its group.

### Evaluation, measurement and peer review

**The ideas behind this group** (a synthesis from the abstracts of about 18 papers; the papers have not been read in full).

*The shared problem is that capability claims rest on numbers whose meaning nobody has checked.* Leaderboards are built from convenient test items, error bars rest on asymptotic assumptions, and test sets leak into training data. The largest group in the track argues that machine learning (ML) evaluation must become a measurement science, and that the peer-review system that certifies results needs reform as well.

*The first thread asks what a benchmark actually measures, and borrows from fields that test minds.* Medical large language model (LLM) benchmarks are built from licensing-exam questions. One paper argues that they should be evaluated empirically for construct validity, in the sense used in psychological testing, and reports proof-of-concept experiments on real clinical data that show significant gaps ([medical construct validity](https://proceedings.mlr.press/v267/alaa25a.html), oral). More generally, evaluating generative AI is framed as a social-science measurement challenge, with a four-level framework for interrogating validity ([GenAI as measurement](https://proceedings.mlr.press/v267/wallach25a.html)). Other papers take their tools from other disciplines. Animal cognition supplies five evaluation principles, tested in a case study of transitive inference ([animal cognition](https://proceedings.mlr.press/v267/rane25a.html), oral). Psychometrics suggests adaptive testing that estimates item characteristics and latent ability instead of averaging over a fixed test set ([psychometric testing](https://proceedings.mlr.press/v267/zhuang25e.html)). Theory-of-mind benchmarks that copy human tests measure only "literal" prediction. They miss "functional" adaptation to a partner, and the two can come apart ([theory of mind](https://proceedings.mlr.press/v267/riemer25a.html)). Bias probes need criteria for when results generalise, which one paper supplies with the EcoLevels framework ([bias probing](https://proceedings.mlr.press/v267/morehouse25a.html), spotlight). A review of 115 human baselines finds that "super-human" claims are poorly supported ([human baselines](https://proceedings.mlr.press/v267/wei25s.html), spotlight). Centaur evaluations would score humans and AI working together instead of AI working alone ([centaur evaluations](https://proceedings.mlr.press/v267/haupt25a.html)).

*The second thread is statistical and procedural: leakage, uncertainty and broken metrics.* Error bars based on the central limit theorem (CLT) dramatically underestimate uncertainty on small specialised benchmarks, so the paper recommends frequentist and Bayesian alternatives ([CLT in evals](https://proceedings.mlr.press/v267/bowyer25a.html), spotlight). Only 9 of 30 documented models report train-test overlap, and one paper argues that developers should publish it ([train-test overlap](https://proceedings.mlr.press/v267/zhang25dt.html), spotlight). AI competitions have already learnt to fight leakage, and one paper calls them the gold standard for rigour ([AI competitions](https://proceedings.mlr.press/v267/sculley25a.html), oral). Sanity checks show that all current generative fidelity and diversity metrics are flawed ([generative metrics](https://proceedings.mlr.press/v267/raisa25a.html)). The same pattern appears in individual domains. Graph learning benchmarks favour narrow molecular tasks ([graph benchmarks](https://proceedings.mlr.press/v267/bechler-speicher25a.html)). Graph-matching datasets based on graph edit distance (GED) suffer train-test leakage, and properties of GED allow adversarial test generation ([graph matching](https://proceedings.mlr.press/v267/roy25a.html)). An analysis of 2 million user prompts shows that current benchmarks miss creative composition tasks ([creative composition](https://proceedings.mlr.press/v267/shen25r.html)). In-house testing is not enough, and flaws in general-purpose AI should be reported through standardised flaw reports and disclosure programmes with legal safe harbours ([third-party evaluation](https://proceedings.mlr.press/v267/longpre25a.html), spotlight).

*The third thread turns to the review system itself.* With more than 10,000 submissions per venue, one paper proposes a two-stage bi-directional review in which authors rate reviews and reviewers earn formal rewards ([reviewer rewards](https://proceedings.mlr.press/v267/kim25am.html), oral). Another draws on data from Paper Copilot to argue for more transparent and regulated open review ([transparent review](https://proceedings.mlr.press/v267/yang25ba.html)).

*Taken together*, the group argues that a benchmark score is an estimate of a construct, with a sampling distribution and a validity argument, and that both should be reported. Still open: whether any single validity framework can serve such different targets, from clinical reasoning to graph matching, and who will pay for proctored or adaptive evaluation at scale.

**Papers**

- [Medical Large Language Model Benchmarks Should Prioritize Construct Validity](https://proceedings.mlr.press/v267/alaa25a.html) (oral)
- [Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints](https://proceedings.mlr.press/v267/bowyer25a.html) (spotlight)
- [Evaluating Generative AI Systems Is a Social Science Measurement Challenge](https://proceedings.mlr.press/v267/wallach25a.html)
- [AI Competitions Provide the Gold Standard for Empirical Rigor in GenAI Evaluation](https://proceedings.mlr.press/v267/sculley25a.html) (oral)
- [In-House Evaluation Is Not Enough. Towards Robust Third-Party Evaluation and Flaw Disclosure for General-Purpose AI](https://proceedings.mlr.press/v267/longpre25a.html) (spotlight)
- [Human Baselines in Model Evaluations Need Rigor and Transparency (With Recommendations & Reporting Checklist)](https://proceedings.mlr.press/v267/wei25s.html) (spotlight)
- [Language model developers should report train-test overlap](https://proceedings.mlr.press/v267/zhang25dt.html) (spotlight)
- [Principles of Animal Cognition to Improve LLM Evaluations](https://proceedings.mlr.press/v267/rane25a.html) (oral)
- [AI Evaluation Should Learn from How We Test Humans](https://proceedings.mlr.press/v267/zhuang25e.html)
- [AI Should Not Be An Imitation Game: Centaur Evaluations](https://proceedings.mlr.press/v267/haupt25a.html)
- [Theory of Mind Benchmarks are Broken for Large Language Models](https://proceedings.mlr.press/v267/riemer25a.html)
- [Rethinking LLM Bias Probing Using Lessons from the Social Sciences](https://proceedings.mlr.press/v267/morehouse25a.html) (spotlight)
- [All Current Generative Fidelity and Diversity Metrics are Flawed](https://proceedings.mlr.press/v267/raisa25a.html)
- [Societal Impacts Research Requires Benchmarks for Creative Composition Tasks](https://proceedings.mlr.press/v267/shen25r.html)
- [Graph Learning Will Lose Relevance Due To Poor Benchmarks](https://proceedings.mlr.press/v267/bechler-speicher25a.html)
- [Graph Matching Systems Deserve Better Benchmarks](https://proceedings.mlr.press/v267/roy25a.html)
- [The AI Conference Peer Review Crisis Demands Author Feedback and Reviewer Rewards](https://proceedings.mlr.press/v267/kim25am.html) (oral)
- [The Artificial Intelligence and Machine Learning Community Should Adopt a More Transparent and Regulated Peer Review Process](https://proceedings.mlr.press/v267/yang25ba.html)

### Understanding deep learning: theory and science

**The ideas behind this group** (a synthesis from the abstracts of about 12 papers; the papers have not been read in full).

*The shared problem is that deep learning works without an agreed account of why.* These papers argue about what a satisfying explanation would look like: classical learning theory extended, new mathematics, or the reverse-engineering of mechanisms. Several of them also say that some popular explanations rest on assumptions that do not hold.

*The first thread says that deep learning is less mysterious than it looks, and that we should explain it with general principles instead of puzzles.* Benign overfitting, double descent and the success of overparametrisation can be characterised rigorously with probably approximately correct (PAC)-Bayes and countable-hypothesis bounds. On this view the unifying principle is a soft inductive bias: a flexible hypothesis space with a preference for simple solutions ([not so mysterious](https://proceedings.mlr.press/v267/wilson25a.html), spotlight). A companion argument holds that there is little evidence that phenomena such as grokking or the lottery ticket hypothesis appear in real-world applications. They should therefore serve as test beds for broad explanatory theories, not as isolated puzzles that each need a bespoke explanation ([deep learning phenomena](https://proceedings.mlr.press/v267/jeffares25a.html), oral). In the physicist's style, layerwise linear models are minimal solvable systems. Their "dynamical feedback principle" already accounts for neural collapse, emergence, lazy and rich regimes, and grokking ([layerwise linear models](https://proceedings.mlr.press/v267/nam25a.html)).

*The second thread proposes new mathematical lenses.* One paper argues that deep networks succeed because target functions are compositionally sparse, a property it shows is shared by all efficiently Turing-computable functions. Learnability and optimisation in this setting remain open ([compositional sparsity](https://proceedings.mlr.press/v267/danhofer25a.html)). Neuroalgebraic geometry treats networks with polynomial activations as parameterising semi-algebraic varieties. It then links dimension, degree and singularities to sample complexity, expressivity, training dynamics and implicit bias ([neuroalgebraic geometry](https://proceedings.mlr.press/v267/marchetti25a.html), spotlight). For self-supervised learning (SSL), identifiability theory can explain the Platonic Representation Hypothesis, and one paper proposes a broader Singular Identifiability Theory that covers finite samples and inductive biases ([SSL identifiability](https://proceedings.mlr.press/v267/reizinger25a.html)). Spectral graph neural networks (GNNs) may owe less to the graph Fourier basis than their theory claims, because the limits on expressive filters also hide whether that basis really behaves like a Fourier basis ([spectral GNNs](https://proceedings.mlr.press/v267/guo25w.html)).

*The third thread asks what trained models actually compute, and where the standard framing misleads.* AlgEval proposes to identify the algorithmic primitives that LLMs learn by forming top-down hypotheses and testing them at the level of circuits ([algorithmic understanding](https://proceedings.mlr.press/v267/eberle25a.html), spotlight). Interpretability can be treated as a communication problem that needs neologisms, new words for machine concepts ([neologisms](https://proceedings.mlr.press/v267/hewitt25a.html), spotlight). Supervised classifiers answer the wrong question for out-of-distribution (OOD) detection. They confuse high uncertainty or a large distance in feature space with being OOD, and the resulting errors cannot be removed by scaling or outlier exposure ([OOD detection](https://proceedings.mlr.press/v267/li25ec.html)). Generative 3D models fail to generalise beyond their training data to objects that can be manufactured ([manufacturing gap](https://proceedings.mlr.press/v267/kimmel25a.html)). Biological neural networks grown in vitro now offer a way to test theories of neural computation outside simulation ([in vitro computation](https://proceedings.mlr.press/v267/gressmann25a.html)).

*Taken together*, the group prefers general principles to bespoke explanations, and solvable or algebraic models to anecdotes. Several papers also warn that an elegant theoretical framing, such as the Fourier basis or predictive uncertainty, may not be what drives a method's behaviour. Still open: whether these lenses agree with each other, and which of them will make predictions that can be tested at the scale of real models.

**Papers**

- [A Theory of Deep Learning Must Include Compositional Sparsity](https://proceedings.mlr.press/v267/danhofer25a.html)
- [We Need An Algorithmic Understanding of Generative AI](https://proceedings.mlr.press/v267/eberle25a.html) (spotlight)
- [We Can't Understand AI Using our Existing Vocabulary](https://proceedings.mlr.press/v267/hewitt25a.html) (spotlight)
- [Not All Explanations for Deep Learning Phenomena Are Equally Valuable](https://proceedings.mlr.press/v267/jeffares25a.html) (oral)
- [Algebra Unveils Deep Learning - An Invitation to Neuroalgebraic Geometry](https://proceedings.mlr.press/v267/marchetti25a.html) (spotlight)
- [Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena (Neural Collapse, Emergence, Lazy/Rich Regime, and Grokking)](https://proceedings.mlr.press/v267/nam25a.html)
- [Deep Learning is Not So Mysterious or Different](https://proceedings.mlr.press/v267/wilson25a.html) (spotlight)
- [Spectral GNNs Rely Less on Graph Fourier Basis than Conceived](https://proceedings.mlr.press/v267/guo25w.html)
- [An Empirically Grounded Identifiability Theory Will Accelerate Self Supervised Learning Research](https://proceedings.mlr.press/v267/reizinger25a.html)
- [It Is Time We Test Neural Computation In Vitro](https://proceedings.mlr.press/v267/gressmann25a.html)
- [Supervised Classifiers Answer the Wrong Questions for OOD Detection](https://proceedings.mlr.press/v267/li25ec.html)
- [You Can't Manufacture a NeRF](https://proceedings.mlr.press/v267/kimmel25a.html)

### Probabilistic, causal and statistical foundations

**The ideas behind this group** (a synthesis from the abstracts of about 8 papers; the papers have not been read in full).

*The shared problem is that ML borrows the language of probability and statistics, but often uses it loosely or works around it.* These papers argue that existing probabilistic and statistical foundations already answer more than the field assumes, and that ML should adopt the habits of those disciplines as well as their notation.

*The first thread says that probabilistic modelling is enough, and is the future.* Every causal inference question can be answered by "writing down the probability of everything", without bespoke causal frameworks or notation. On this view causal tools are what standard probabilistic modelling and inference produce ([probabilistic causal inference](https://proceedings.mlr.press/v267/mlodozeniec25a.html), oral). Prior-data fitted networks (PFNs) are networks trained on synthetic datasets drawn from a prior, so they amount to amortised Bayesian inference. One paper argues that they are the future of Bayesian prediction, because they move pre-training compute into settings with little data ([prior-fitted networks](https://proceedings.mlr.press/v267/muller25d.html)). Another proposes that LLM reasoning should rest on a Bayesian meta-reasoning framework that draws on dual-process and metacognitive theories ([Bayesian meta-reasoning](https://proceedings.mlr.press/v267/yan25g.html)).

*The second thread questions how methods are assessed and quantified.* Synthetic experiments are often dismissed. One paper argues instead that they are essential for assessing causal ML, because only they reveal the ground truth, and it proposes principles for running them rigorously ([synthetic causal experiments](https://proceedings.mlr.press/v267/poinsot25a.html)). In reinforcement learning (RL) theory, matching a regret lower bound up to multiplicative constants is not enough. Tightening the bonus terms of the UCBVI algorithm improves its empirical performance noticeably, which suggests that constants matter ([regret constants](https://proceedings.mlr.press/v267/drago25c.html)). The aleatoric–epistemic split in uncertainty quantification has definitions that contradict each other, and the split loses its meaning for interactive LLM agents. One paper proposes three alternatives: underspecification uncertainty, interactive learning and output uncertainty ([agent uncertainty](https://proceedings.mlr.press/v267/kirchhof25b.html)).

*The third thread recasts explainability as statistics.* Explanations are statistics of high-dimensional functions, so the questions an applied statistician would ask apply to them, above all how a quantity relates to the question a human actually has ([XAI as statistics](https://proceedings.mlr.press/v267/bordt25b.html)). The empirical side is weak. An analysis of 607 explainable AI (XAI) papers with user studies finds that most lack reproducibility, baselines or placebo explanations ([XAI user studies](https://proceedings.mlr.press/v267/piculin25a.html)).

*Taken together*, the group argues for the probabilistic model as the common language and for statistical practice as the standard of evidence: state the estimand, design the experiment and keep the constants. Still open: whether amortised inference with PFNs inherits the guarantees of the Bayesian posterior it approximates when the prior is misspecified, and whether a single probabilistic language suits interactive and agentic settings in which the task itself is uncertain.

**Papers**

- [Probabilistic Modelling is Sufficient for Causal Inference](https://proceedings.mlr.press/v267/mlodozeniec25a.html) (oral)
- [The Future of Bayesian Prediction Is Prior-Fitted](https://proceedings.mlr.press/v267/muller25d.html)
- [LLMs Need a Bayesian Meta-Reasoning Framework for More Robust and Generalizable Reasoning](https://proceedings.mlr.press/v267/yan25g.html)
- [Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption](https://proceedings.mlr.press/v267/poinsot25a.html)
- [Constants are Critical in Regret Bounds for Reinforcement Learning](https://proceedings.mlr.press/v267/drago25c.html)
- [Uncertainty Quantification Needs Reassessment for Large Language Model Agents](https://proceedings.mlr.press/v267/kirchhof25b.html)
- [Rethinking Explainable Machine Learning as Applied Statistics](https://proceedings.mlr.press/v267/bordt25b.html)
- [Explainable AI Cannot Advance Without Better User Studies](https://proceedings.mlr.press/v267/piculin25a.html)

### Agents, reasoning and general intelligence

**The ideas behind this group** (a synthesis from the abstracts of about 11 papers; the papers have not been read in full).

*The shared problem is that large language models (LLMs) are being turned into agents faster than anyone can say what those agents can reliably do.* The papers in this group question how reasoning is learned, how agent systems should be analysed, and what the field should be aiming for at all.

*The first thread says today's reasoning is too entangled with knowledge to generalise.* Experiments on algorithmic tasks in esoteric programming languages suggest that LLM reasoning overfits to its training data. One proposal is to disentangle reasoning from knowledge by pretraining with reinforcement learning (RL) from scratch on a curriculum of synthetic tasks, with a small context window, and to keep knowledge in a retrieval system with a large external memory ([reward-based pretraining](https://proceedings.mlr.press/v267/han25n.html), spotlight). A second paper argues that self-improving agents stay rigid because their improvement loops are extrinsic and designed by humans. It offers a formal framework for *intrinsic* metacognitive learning made of three parts: metacognitive knowledge, planning and evaluation ([metacognitive learning](https://proceedings.mlr.press/v267/liu25cw.html)). A third asks for algorithmic rigour. Treat the LLM forward pass as the atomic unit of cost and use asymptotic analysis to judge how a problem is split across agents, rather than copying how a human team would assign roles ([asymptotic analysis](https://proceedings.mlr.press/v267/meyerson25a.html)).

*The second thread turns to formal guarantees.* Proof assistants can check reasoning and return automatic feedback. On that basis, formal mathematical reasoning is put forward as an indispensable part of AI for mathematics, verification and verifiable generation ([formal maths](https://proceedings.mlr.press/v267/yang25az.html), spotlight). The same logic applied to agents says that hallucination can only be contained by integrating LLMs with formal methods (FMs). LLMs bring adaptability, and FMs bring correctness guarantees but need LLMs to make them accessible ([LLMs + formal methods](https://proceedings.mlr.press/v267/zhang25ds.html)). In software engineering, one paper offers a taxonomy of tasks and measures and names the bottlenecks to full automation ([AI for software engineering](https://proceedings.mlr.press/v267/gu25e.html)).

*The third thread concerns how agents are deployed, evaluated and used.* Agents acting for users need authenticated, auditable delegation. One paper sketches this by extending OAuth 2.0 and OpenID Connect with agent-specific credentials and by translating natural-language permissions into scoping rules ([authenticated delegation](https://proceedings.mlr.press/v267/south25a.html), oral). Who controls the agent matters too. "Platform agents" owned by incumbents would deepen surveillance and lock-in, so the field should build user-controlled "agent advocates", backed by public compute, open standards and competition regulation ([agent advocates](https://proceedings.mlr.press/v267/kapoor25a.html)). On evaluation, continual RL is undermined by "lifetime tuning", where hyperparameters are chosen with access to the full lifetime, e.g. 200 million frames in Atari. Under lifetime tuning all algorithms succeed equally, and continual-RL methods only stand out when tuning is restricted to a fraction of the lifetime ([lifetime tuning](https://proceedings.mlr.press/v267/mesbahi25a.html)). LLM simulations of human research subjects are defended as a promising method, already fit for pilot studies, once five tractable challenges are addressed ([social simulations](https://proceedings.mlr.press/v267/anthis25a.html)).

*Taken together*, the group wants agents and reasoning treated as objects of analysis, with cost models, formal checks, honest evaluation protocols and explicit authority chains. The odd one out is a call to stop treating artificial general intelligence (AGI) as the field's north star at all. It names six traps the AGI framing worsens, among them the "Illusion of Consensus" and "Generality Debt", and asks for specific, pluralistic and inclusive goals instead ([stop AGI](https://proceedings.mlr.press/v267/blili-hamelin25a.html)). Still open: whether the proposed formal frameworks (metacognition, LLM-primitive complexity) yield theorems or experiments that could prove any of these positions wrong.

**Papers**

- [General Intelligence Requires Reward-based Pretraining](https://proceedings.mlr.press/v267/han25n.html) (spotlight)
- [Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](https://proceedings.mlr.press/v267/liu25cw.html)
- [Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives](https://proceedings.mlr.press/v267/meyerson25a.html)
- [Build Agent Advocates, Not Platform Agents](https://proceedings.mlr.press/v267/kapoor25a.html)
- [AI Agents Need Authenticated Delegation](https://proceedings.mlr.press/v267/south25a.html) (oral)
- [Trustworthy AI Agents Require the Integration of Large Language Models and Formal Methods](https://proceedings.mlr.press/v267/zhang25ds.html)
- [Formal Mathematical Reasoning—A New Frontier in AI](https://proceedings.mlr.press/v267/yang25az.html) (spotlight)
- [Stop treating 'AGI' as the north-star goal of AI research](https://proceedings.mlr.press/v267/blili-hamelin25a.html)
- [Lifetime tuning is incompatible with continual reinforcement learning](https://proceedings.mlr.press/v267/mesbahi25a.html)
- [Future Research and Challenges Remain Towards AI for Software Engineering](https://proceedings.mlr.press/v267/gu25e.html)
- [LLM Social Simulations Are a Promising Research Method](https://proceedings.mlr.press/v267/anthis25a.html)

### Scale, data and the model supply chain

**The ideas behind this group** (a synthesis from the abstracts of about 8 papers; the papers have not been read in full).

*The shared problem is that the inputs and outputs of large models (data, compute, weights and licences) are handled as if they were free and trustworthy.* These papers argue that the costs and risks have simply moved somewhere less visible.

*The first thread questions scale itself.* One paper says the field should shift from neural scaling laws to "downscaling", keeping performance while drastically cutting resource demands, because of computational inefficiency, environmental impact and deployment constraints ([downscaling](https://proceedings.mlr.press/v267/goel25c.html)). A broader framework says that Scaling Up faces inherent bottlenecks and that the future lies in Scaling Down (efficiency) and Scaling Out (collaboration across domains), with healthcare among the applications ([up, down and out](https://proceedings.mlr.press/v267/wang25fh.html)).

*The second thread prices the data.* Across 64 LLMs released between 2016 and 2024, paying people to write the training data from scratch would cost 10–1000 times more than training the model, even at conservative wage rates. On that basis the paper argues that compensation for data producers *should* be the largest cost of an LLM ([data should be expensive](https://proceedings.mlr.press/v267/kandpal25a.html)). A second paper says that LLM-generated content on annotation platforms points to a design flaw. Data collection systems that optimise speed and external incentives erode intrinsic motivation and with it data quality ([incentives backfire](https://proceedings.mlr.press/v267/santy25a.html)). For alignment, one paper argues that methods focus on algorithms and loss functions and neglect the feedback data. It catalogues unreliability, temporal drift and context dependence in human feedback, and the limits of AI feedback ([data-centric alignment](https://proceedings.mlr.press/v267/yeh25b.html)).

*The third thread treats released models as a software supply chain with legal and security exposure.* Open model hubs carry real, already-exploited risks: models swapped for malware, or trained with vulnerable frameworks or on poisoned data. Sigstore, built for open-source software, could let publishers sign models and attest to their datasets ([supply chain](https://proceedings.mlr.press/v267/meiklejohn25a.html)). Licensing is also a mess. A review of common licences (Apache-2.0, OpenRAIL, Creative Commons, Llama2, GPL-3.0) and of real Hugging Face repositories finds risks of noncompliance, and the paper drafts ModelGo Licenses as a standard ([model licensing](https://proceedings.mlr.press/v267/duan25d.html), oral). Copyright risk arises at several lifecycle stages that are optimised separately. One paper argues this separation creates theoretical bottlenecks and proposes iterative online-offline joint optimisation ([copyright joint optimisation](https://proceedings.mlr.press/v267/pan25i.html)).

*Taken together*, the group says that "bigger model" has hidden costs in data labour, provenance, licensing and energy, and that these should be measured and built into the pipeline. Still open: who would enforce provenance and compensation schemes, and whether the "theoretical bottlenecks" claimed for separate copyright fixes can be stated formally.

**Papers**

- [Enough of Scaling LLMs! Lets Focus on Downscaling](https://proceedings.mlr.press/v267/goel25c.html)
- [AI Scaling: From Up to Down and Out](https://proceedings.mlr.press/v267/wang25fh.html)
- [The Most Expensive Part of an LLM *should* be its Training Data](https://proceedings.mlr.press/v267/kandpal25a.html)
- [When Incentives Backfire, Data Stops Being Human](https://proceedings.mlr.press/v267/santy25a.html)
- [Machine Learning Models Have a Supply Chain Problem](https://proceedings.mlr.press/v267/meiklejohn25a.html)
- [Current Model Licensing Practices are Dragging Us into a Quagmire of Legal Noncompliance](https://proceedings.mlr.press/v267/duan25d.html) (oral)
- [Challenges and Future Directions of Data-Centric AI Alignment](https://proceedings.mlr.press/v267/yeh25b.html)
- [Iterative Online-Offline Joint Optimization is Needed to Manage Complex LLM Copyright Risks](https://proceedings.mlr.press/v267/pan25i.html)

### Safety and security

**The ideas behind this group** (a synthesis from the abstracts of about 5 papers; the papers have not been read in full).

*The shared problem is that AI safety is defined too narrowly, both in the guarantees it trusts and in the harms it watches for.* Two papers look at technical guarantees and tools. Three widen what counts as a safety risk.

*The first thread is about guarantees that promise more than they deliver.* Certified robustness is promoted as the answer to adversarial examples, yet it has a "paradox of detection without distinction" and gives practitioners no clear criteria for judging certification schemes. "Guaranteed" robustness claims can themselves create security risk when users over-trust them, so how certificates are presented does not match what they actually cover ([certified robustness](https://proceedings.mlr.press/v267/cullen25a.html), oral). Knowledge editing methods (KEs), built to update outdated facts, are cheap, effective and stealthy. Combined with model hubs that accept uploads without verification, that makes them an attractive tool for attackers. The paper asks for tamper-resistant models and a more secure ecosystem ([knowledge editing](https://proceedings.mlr.press/v267/youssef25a.html)).

*The second thread rejects static testing.* Static benchmarks and single-shot robustness tests miss drift into maladaptation: reward hacking, over-optimisation, over-alignment. AI safety should instead be "antifragile", using uncertainty so that a system's ability to handle rare and out-of-distribution (OOD) events grows with repeated exposure ([antifragile safety](https://proceedings.mlr.press/v267/jin25m.html)). For a statistically minded reader, this is an argument against evaluating once under a fixed distribution.

*The third thread moves the risk from catastrophic events to slow structural change.* "Gradual disempowerment" is offered as an alternative to abrupt-takeover scenarios. As AI replaces human labour and cognition, explicit control mechanisms (voting, consumer choice) and the implicit alignment that comes from systems needing human participation both weaken. Feedback between economy, culture and state could make this loss effectively irreversible ([gradual disempowerment](https://proceedings.mlr.press/v267/kulveit25a.html)). Read through economic theory, one paper argues that the future of work belongs in safety. AI reshapes labour markets and widens income inequality, closed-source development resembles rent-seeking, and the remedy is collective licensing and a pro-worker framework for global governance ([future of work](https://proceedings.mlr.press/v267/hazra25a.html), oral).

*Taken together*, the group says that a guarantee is only as good as the threat model and the deployment around it, and that the slowest harms may be the least studied. Still open: whether antifragility or gradual disempowerment can be made precise enough to measure, and what formal statement certified robustness would have to make before it implies security.

**Papers**

- [Certified Robustness Does Not (Yet) Imply Model Security](https://proceedings.mlr.press/v267/cullen25a.html) (oral)
- [Editing Large Language Models Poses Serious Safety Risks](https://proceedings.mlr.press/v267/youssef25a.html)
- [AI Safety Must Embrace an Antifragile Perspective](https://proceedings.mlr.press/v267/jin25m.html)
- [AI Safety should prioritize the Future of Work](https://proceedings.mlr.press/v267/hazra25a.html) (oral)
- [Humanity Faces Existential Risk from Gradual Disempowerment](https://proceedings.mlr.press/v267/kulveit25a.html)

### Society, politics and governance

**The ideas behind this group** (a synthesis from the abstracts of about 11 papers; the papers have not been read in full).

*The shared problem is that AI now makes or shapes decisions that societies used to govern through law, democratic process and professional norms.* These papers ask which existing institutions and concepts can carry over, and where ML practice breaks them.

*The first thread argues that existing law and regulatory experience already provide tools.* Generative AI regulation can learn from social media regulation. The paper gives four recommendations, from countering bias (and perceptions of it) to taking a global perspective ([social media lessons](https://proceedings.mlr.press/v267/appel25a.html), oral). In the United States, consumer protection law is put forward as an "inalienable defense" for AI safety, against both the push for new AI-specific laws and the view that it is red tape ([consumer protection](https://proceedings.mlr.press/v267/booth25a.html)). Building on two recent court decisions, another paper argues that current AI cannot guarantee due process. Without technical answers on interpretability and causality, and with strong trade-secret protection, it should not make decisions that require due process ([due process](https://proceedings.mlr.press/v267/rane25b.html)).

*The second thread is about participation and democratic legitimacy.* A "Right to AI", modelled on Lefebvre's "Right to the City", treats AI as societal infrastructure and data as socially produced. Drawing on Arnstein's Ladder of Citizen Participation and nine case studies, it builds a four-tier participation model ([Right to AI](https://proceedings.mlr.press/v267/mushkani25b.html)). A "Democracy Levels" framework defines milestones towards democratic governance and alignment of AI, citing early steps such as Meta's Community Forums and Anthropic's Collective Constitutional AI ([Democracy Levels](https://proceedings.mlr.press/v267/ovadya25a.html)). Responsible, application-driven (RAD) AI research asks for transdisciplinary teams, context-specific methods and metrics, and staged testbeds ([RAD AI](https://proceedings.mlr.press/v267/hartman25a.html)).

*The third thread criticises the concepts ML borrows.* Following Raz, true political neutrality is said to be impossible but achievable by degree. The paper proposes eight techniques at three levels of conceptualising AI and tests them on current LLMs ([political neutrality](https://proceedings.mlr.press/v267/fisher25a.html), oral). Rigid racial categories are U.S.-centric, erase mixed-race people and contradict the view of race as socially constructed, so ML should move beyond categorical race variables ([race categorisation](https://proceedings.mlr.press/v267/doh25a.html), spotlight). Contextual Integrity (CI), a privacy theory, is being applied to LLMs without its four fundamental tenets. The paper also points to experimental-hygiene problems such as prompt sensitivity and positional bias ([contextual integrity](https://proceedings.mlr.press/v267/shvartzshnaider25a.html)).

*The fourth thread, closest to the group's own field, is health.* Retrieval-augmented generation (RAG) and citation grounding can produce answers that are literally accurate yet misleading. A large-scale query analysis shows they decontextualise facts, omit key sources and reinforce patient misconceptions ([RAG in medicine](https://proceedings.mlr.press/v267/wong25b.html)). For mental health, LLMs are recast as co-creators rather than tools, under SAFE-i deployment guidelines and a HAAS-e evaluation framework whose metrics include empathy and cultural sensitivity ([mental-health co-creators](https://proceedings.mlr.press/v267/badawi25a.html)).

*Taken together*, the group says that legitimacy, contestability and context are properties of a deployed system and can be designed for, or measured, like accuracy. Still open: how proxies such as "approximate neutrality" or participation tiers would be validated, and whether interpretability research can supply the reasons that due process demands.

**Papers**

- [Generative AI Regulation Can Learn from Social Media Regulation](https://proceedings.mlr.press/v267/appel25a.html) (oral)
- [Strong Consumer Protection is an Inalienable Defense for AI Safety in the United States](https://proceedings.mlr.press/v267/booth25a.html)
- [The Categorization of Race in ML is a Flawed Premise](https://proceedings.mlr.press/v267/doh25a.html) (spotlight)
- [Political Neutrality in AI Is Impossible — But Here Is How to Approximate It](https://proceedings.mlr.press/v267/fisher25a.html) (oral)
- [The Right to AI](https://proceedings.mlr.press/v267/mushkani25b.html)
- [Democratic AI is Possible. The Democracy Levels Framework Shows How It Might Work.](https://proceedings.mlr.press/v267/ovadya25a.html)
- [AI's growing due process problem](https://proceedings.mlr.press/v267/rane25b.html)
- [Contextual Integrity is Inadequately Applied to Language Models](https://proceedings.mlr.press/v267/shvartzshnaider25a.html)
- [We Need Responsible, Application-Driven (RAD) AI Research](https://proceedings.mlr.press/v267/hartman25a.html)
- [Beyond Assistance – Reimagining LLMs as Ethical and Adaptive Co-Creators in Mental Health Care](https://proceedings.mlr.press/v267/badawi25a.html)
- [Retrieval-augmented systems can be dangerous medical communicators](https://proceedings.mlr.press/v267/wong25b.html)

## Outlier papers

Positions that stand out from the rest of the track in subject, form or reach.

- [It Is Time We Test Neural Computation In Vitro](https://proceedings.mlr.press/v267/gressmann25a.html): the
  only wet-lab paper, proposing artificial networks that interact with and control living neural cultures
  as a way to test theories of neural computation outside simulation.
- [You Can't Manufacture a NeRF](https://proceedings.mlr.press/v267/kimmel25a.html): the only paper about
  physical production, arguing that 3D generative models ignore milling, casting and injection moulding, and
  supplying new datasets on which existing models turn out to be badly overfit.
- [AI's growing due process problem](https://proceedings.mlr.press/v267/rane25b.html): a legal argument
  built on two United States court decisions, concluding that current AI cannot guarantee due process and
  should not make decisions that require it.
- [Humanity Faces Existential Risk from Gradual Disempowerment](https://proceedings.mlr.press/v267/kulveit25a.html):
  the widest reach in the track, replacing the sudden-takeover scenario with a slow, self-reinforcing loss of
  human influence over the economy, culture and the state.
- [Principles of Animal Cognition to Improve LLM Evaluations](https://proceedings.mlr.press/v267/rane25a.html):
  borrows experimental design from comparative psychology and tests it on a single capability, transitive
  inference.
- [We Can't Understand AI Using our Existing Vocabulary](https://proceedings.mlr.press/v267/hewitt25a.html):
  treats interpretability as a communication problem and proposes coining new words ("neologisms") for
  machine concepts, demonstrated with length and diversity neologisms that steer outputs.
- [The Most Expensive Part of an LLM *should* be its Training Data](https://proceedings.mlr.press/v267/kandpal25a.html):
  a costing exercise across 64 LLMs, which estimates that paying people to write the training data would cost
  10 to 1000 times more than training the model.
- [A More Transparent and Regulated Peer Review Process](https://proceedings.mlr.press/v267/yang25ba.html):
  draws its evidence from Paper Copilot, the authors' own website, which reports over 200,000 early-career
  users from 177 countries.

## Theory picks

The positions with the most mathematical or statistical content to work through (from abstracts only).

1. [Algebra Unveils Deep Learning: Neuroalgebraic Geometry](https://proceedings.mlr.press/v267/marchetti25a.html):
   networks with polynomial activations parameterise semi-algebraic varieties, and the paper proposes a
   dictionary between their invariants and learning: dimension with sample complexity, degree with
   expressivity, singularities with training dynamics and implicit bias. Worth computing the dimension of
   the function space of a small two-layer polynomial network and comparing it with the parameter count.
   The gap comes from parameter symmetries and is where "effective" capacity hides.
2. [A Theory of Deep Learning Must Include Compositional Sparsity](https://proceedings.mlr.press/v267/danhofer25a.html):
   the claim is that every efficiently Turing-computable function is compositionally sparse, meaning it
   is built from constituent functions that each depend on only a few inputs. Work through why a
   polynomial-size, bounded-fan-in circuit gives such a composition. Then ask why this breaks the curse of
   dimensionality for approximation. Learnability and optimisation remain open, as the authors say.
3. [Deep Learning is Not So Mysterious or Different](https://proceedings.mlr.press/v267/wilson25a.html):
   benign overfitting, double descent and overparameterisation are said to be explained by PAC-Bayes and
   countable-hypothesis bounds once one allows "soft inductive biases": a large hypothesis space with a
   prior preferring simple solutions. A good exercise is to derive the countable Occam bound and see how a
   prior over hypotheses lets a model interpolate the data and still generalise.
4. [Solve Layerwise Linear Models First](https://proceedings.mlr.press/v267/nam25a.html): linear networks
   with several layers are offered as solvable minimal models. They follow a "dynamical feedback
   principle", in which layers amplify each other's growth, and the paper says this explains neural collapse,
   emergence, lazy and rich regimes, and grokking. Derive the gradient-flow equations for a two-layer linear
   network, decoupled into singular modes. Check how much of each phenomenon really follows from the
   resulting sigmoidal, scale-dependent learning curves.
5. [Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints](https://proceedings.mlr.press/v267/bowyer25a.html):
   error bars based on the central limit theorem (CLT) are too narrow on small, specialised benchmarks, and
   the authors recommend simple frequentist and Bayesian alternatives. It takes an afternoon to simulate the
   coverage of the normal-approximation interval for a binomial accuracy. Compare it with exact and
   Beta-posterior intervals, especially when accuracy is close to 0 or 1.
6. [Probabilistic Modelling is Sufficient for Causal Inference](https://proceedings.mlr.press/v267/mlodozeniec25a.html):
   any causal question can be answered by "writing down the probability of everything", with no special
   causal notation, and existing causal tools re-read as ordinary inference. Check how interventions and
   counterfactuals are encoded as conditioning in an extended joint model. Then ask where identifiability
   enters, since that question does not go away when the notation does.

Also worth a look: [Constants are Critical in Regret Bounds](https://proceedings.mlr.press/v267/drago25c.html)
(an improved bonus analysis of the UCBVI algorithm for finite-horizon tabular Markov decision processes,
arguing that minimax optimality up to constants hides real differences in performance),
[Spectral GNNs Rely Less on Graph Fourier Basis than Conceived](https://proceedings.mlr.press/v267/guo25w.html)
(questions whether Laplacian eigenvectors really act as a Fourier basis, and suggests that limits on filter
complexity are what protect stability and generalisation),
[An Empirically Grounded Identifiability Theory](https://proceedings.mlr.press/v267/reizinger25a.html)
(identifiability as a route to the Platonic Representation Hypothesis in self-supervised learning (SSL)),
[The Future of Bayesian Prediction Is Prior-Fitted](https://proceedings.mlr.press/v267/muller25d.html)
(prior-data fitted networks (PFNs) as amortised posterior predictive inference, trained on data sampled from a
prior), [Supervised Classifiers Answer the Wrong Questions for OOD Detection](https://proceedings.mlr.press/v267/li25ec.html)
(irreducible errors when out-of-distribution (OOD) detection relies on a classifier's uncertainty or feature
distances) and [AI Evaluation Should Learn from How We Test Humans](https://proceedings.mlr.press/v267/zhuang25e.html)
(psychometric, latent-trait models of benchmark items and adaptive testing).

## Relevance to medical imaging

- **Construct validity** ([Alaa et al.](https://proceedings.mlr.press/v267/alaa25a.html)) is the closest
  match. The authors show with clinical data that benchmarks built from licensing-exam questions miss the
  real tasks, and the same question applies to imaging benchmarks whose labels stand in for a clinical
  decision. The measurement-theory papers by [Wallach et al.](https://proceedings.mlr.press/v267/wallach25a.html)
  and [Wei et al.](https://proceedings.mlr.press/v267/wei25s.html) give the vocabulary.
- **Small test sets** are the norm in imaging, so the CLT warning from
  [Bowyer et al.](https://proceedings.mlr.press/v267/bowyer25a.html) applies directly to per-case Dice and
  accuracy reported on a few dozen held-out scans. Train-test overlap reporting
  ([Zhang et al.](https://proceedings.mlr.press/v267/zhang25dt.html)) matters for foundation models pretrained
  on the same public datasets they are later tested on.
- **Out-of-distribution detection** under scanner, site or operator shift (in ultrasound, probe and sweep
  variation) often uses classifier confidence. [Li et al.](https://proceedings.mlr.press/v267/li25ec.html)
  argue that this cannot be fixed by scaling or outlier exposure. Read it alongside the post-deployment
  monitoring paper in the [NeurIPS 2025 position overview](../neurips2025-position-overview/index.html).
- **Communicating with patients and clinicians**: [Wong et al.](https://proceedings.mlr.press/v267/wong25b.html)
  show that retrieval-augmented answers can be accurate sentence by sentence and still mislead, and
  [Kirchhof et al.](https://proceedings.mlr.press/v267/kirchhof25b.html) argue that the aleatoric and epistemic
  split of uncertainty fails for interactive agents. Both bear on report-generation and question-answering
  tools built on images.
- **Synthetic data**: [Räisä et al.](https://proceedings.mlr.press/v267/raisa25a.html) conclude that every
  current fidelity and diversity metric is flawed. Their sanity checks are worth applying before trusting
  metrics such as Fréchet Inception Distance (FID) for synthetic medical images used in augmentation or data
  sharing.
- **Explanations** used in clinical studies: [Bordt et al.](https://proceedings.mlr.press/v267/bordt25b.html)
  treat saliency maps as statistics of a high-dimensional function that need interpretation. In their review
  of 607 papers, [Pičulin et al.](https://proceedings.mlr.press/v267/piculin25a.html) find that most user
  studies lack baselines and placebo explanations.

## Sources

- *Proceedings of the 42nd International Conference on Machine Learning*, PMLR volume 267:
  <https://proceedings.mlr.press/v267/>
- [OpenReview group for the ICML 2025 Position Paper Track](https://openreview.net/group?id=ICML.cc/2025/Position_Paper_Track)
- Oral and spotlight decisions from the ICML 2025 schedule data (`icml-2025-orals-posters.json` on icml.cc).
