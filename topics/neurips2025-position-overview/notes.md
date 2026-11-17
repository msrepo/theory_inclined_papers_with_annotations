---
title: "NeurIPS 2025 Position Paper Track: topic map, outliers and theory picks"
authors: "Topic notes"
venue: "NeurIPS 2025 (Position Paper Track)"
tags: [neurips, position-papers, conference-overview, evaluation, measurement, privacy, ai-policy, research-practice]
status: living
category: "Research practice & policy"
subcategory: "Conference overviews"
short_title: "NeurIPS 2025 — position papers"
summary: "a map of the 40 papers accepted to the first NeurIPS Position Paper Track (2025) in six groups of arguments, with an idea-level synthesis of each, the outliers, and the positions with the most mathematical content to engage with"
related: [icml2024-position-overview, icml2025-position-overview, midl2026-overview, midl2025-overview]
---

## Why this page exists

NeurIPS 2025 (San Diego, 2–7 December 2025) ran its first **Position Paper Track**. A position
paper argues for a claim about the field (what to measure, what to build, what to regulate) rather
than presenting a new method. The organisers report 657 submissions, 140 of them desk-rejected, and
a deliberately low acceptance rate. Accepted papers are part of the proceedings and were presented
as orals and posters.

The list here comes from the NeurIPS public schedule data: **40 accepted papers, 9 of them orals**.
Links go to each paper's OpenReview page, which has the PDF and the reviews. The grouping and
syntheses are from **titles and abstracts only**; shares add up to more than 100% because some
papers make arguments in two groups.

ICML has run a position track since 2024; see the
[ICML 2024](../icml2024-position-overview/index.html), [ICML 2025](../icml2025-position-overview/index.html) and [ICML 2026](../icml2026-position-overview/index.html) overviews for larger, more
theory-heavy companions.

Position papers are worth reading differently from method papers. The question is less "does the
method work?" and more "is the argument valid, is the evidence enough for the claim, and what would
change my mind?"

## Topic groups at a glance

| Group | Papers | What's in it |
|---|---|---|
| [Measurement, evaluation and research practice](#measurement-evaluation-and-research-practice) | 10 | Validity of attack-success rates and LLM judges, broken benchmarks, a refutations track, broader rigour, AI-written surveys, reporting standards |
| [Privacy, consent and unlearning](#privacy-consent-and-unlearning) | 6 | Measuring privacy in pose estimation, pedestrian data, the differential-privacy budget, nonconsensual nude images, what unlearning can and cannot do |
| [Data, compute and power](#data-compute-and-power) | 7 | Data deals and collective bargaining, democratising AI, capability per resource, adaptive sensing, analog models, an atlas of all models |
| [Regulation, risk and harm](#regulation-risk-and-harm) | 7 | Embodied and military AI, weapons, "digital heroin", regulation as a basis for innovation, NeurIPS as a consensus body, red-teaming in schools |
| [Agents, simulation and alignment](#agents-simulation-and-alignment) | 7 | LLM personas and social simulation, multi-agent systems, world models, bidirectional alignment, brain–agent collaboration, conflicting fairness metrics |
| [AI for science and health](#ai-for-science-and-health) | 4 | Foundation models and the scientific method, biology-informed ML, surrogate-model reporting, post-deployment monitoring of clinical AI |

**Overall pattern:** most positions are about *how the field knows things*, not about models. The
single most common move is to borrow measurement theory, statistics or another discipline's
institutions (the IPCC, standardised testing, pharmaceutical regulation, collective bargaining) and
argue that ML practice falls short of it. Only a handful of papers make a technical claim that could
be checked with a theorem or an experiment.

## Topic groups

Each group lists the papers it draws on, since the track is small.

### Measurement, evaluation and research practice

**The ideas behind this group** (a synthesis from the abstracts of about 10 papers; the papers have not been read in full).

*The shared problem is that ML's evidence is weaker than its claims.* Benchmarks leak, comparisons
are apples to oranges, evaluators are other models, and the publication system has no routine way
to correct mistakes. These papers argue that the field needs the measurement discipline that other
sciences take for granted.

*The first thread imports measurement theory from the social sciences.* Attack success rates in red
teaming can only be compared when they measure the same construct under comparable conditions;
many published comparisons of jailbreaks do not meet that bar
([ASR comparisons](https://openreview.net/forum?id=d7hqAhLvWG)). The same lens applied to LLMs as
judges questions four assumptions behind them: that they stand in for human judgment, that they are
capable evaluators, and that they scale and save money
([LLMs as judges](https://openreview.net/forum?id=yqKfMr0yvY)).

*The second thread says the evaluation system itself needs redesign.* Benchmarks suffer from
contamination and selective reporting, and one proposal is a proctored, community-governed layer
with sealed execution and rolling item banks, modelled on high-stakes human exams
([PeerBench](https://openreview.net/forum?id=vFae5rRman)). Evaluations also need to assume that a
capable system may reason strategically about being tested, and game theory can formalise the
reasoning in evaluation-based safety cases
([strategic testing](https://openreview.net/forum?id=OMc0BYxND4)).

*The third thread is about how research is published and corrected.* Conferences should run a
"Refutations and Critiques" track so that errors in accepted papers can be corrected with the same
visibility ([R&C track](https://openreview.net/forum?id=DS1XSAPvKs), oral). AI-generated survey
papers flood preprint servers in what one paper calls a "DDoS attack" on the research community
([survey DDoS](https://openreview.net/forum?id=R5uuqCAPf8), oral). Rigour should mean more than
correct methods: epistemic, normative, conceptual, reporting and interpretative rigour too
([broader rigour](https://openreview.net/forum?id=USqNoPVhxx)). Applied work for social impact is
pushed by review criteria to claim both deployment and methodological novelty, and should instead
be judged on either, with deployed impact actually evaluated
([AI for social impact](https://openreview.net/forum?id=dl5pvd5IgW)). Surrogate models need a shared
reporting standard ([SMRS](https://openreview.net/forum?id=V6DcL5L6CU)), and prompting should be
treated as behavioural science on a new kind of organism rather than as alchemy
([prompting as inquiry](https://openreview.net/forum?id=yZU8kdwafM)).

*Taken together*, the group's message is that validity, not only accuracy, should be reported, and
that the community should build institutions (tracks, proctored benchmarks, standards) to enforce it.
Still open: who pays for and governs such institutions, and whether stricter measurement slows the
field in the way its critics fear.

**Papers**

- [Comparison requires valid measurement: Rethinking attack success rate comparisons in AI red teaming](https://openreview.net/forum?id=d7hqAhLvWG)
- [Neither Valid nor Reliable? Investigating the Use of LLMs as Judges](https://openreview.net/forum?id=yqKfMr0yvY)
- [Position: Benchmarking is Broken - Don't Let AI be Its Own Judge](https://openreview.net/forum?id=vFae5rRman)
- [AI Testing Should Account for Sophisticated Strategic Behaviour](https://openreview.net/forum?id=OMc0BYxND4)
- [Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track](https://openreview.net/forum?id=DS1XSAPvKs) (oral)
- [Stop DDoS Attacking the Research Community with AI-Generated Survey Papers](https://openreview.net/forum?id=R5uuqCAPf8) (oral)
- [Rigor in AI: Doing Rigorous AI Work Requires a Broader, Responsible AI-Informed Conception of Rigor](https://openreview.net/forum?id=USqNoPVhxx)
- [Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](https://openreview.net/forum?id=dl5pvd5IgW)
- [SMRS: advocating a unified reporting standard for surrogate models in the artificial intelligence era](https://openreview.net/forum?id=V6DcL5L6CU)
- [Prompting as Scientific Inquiry](https://openreview.net/forum?id=yZU8kdwafM)

### Privacy, consent and unlearning

**The ideas behind this group** (a synthesis from the abstracts of about 6 papers).

*The shared problem is that privacy is claimed far more often than it is measured.* The papers
target vision datasets collected in public or scraped from the web, and the technical tools
(differential privacy, unlearning) that are offered as remedies.

*The first thread asks vision research to measure what it claims.* Human pose estimation is called
"privacy-preserving" or "human-centric" while being evaluated only on accuracy, with RGB input and
no benchmark tied to legal requirements
([pose estimation privacy](https://openreview.net/forum?id=upugtLPOxC)). Pedestrians are captured
without consent in driving and surveillance datasets, and current anonymisation either fails to
protect them or destroys utility ([pedestrian privacy](https://openreview.net/forum?id=rdeCalg68L),
oral). Nudity-detection research has collected and even redistributed nude images without consent,
sometimes with uncensored faces, which the authors call a continuation of image-based sexual abuse
([nonconsensual images](https://openreview.net/forum?id=Ev5xwr3vWh), oral).

*The second thread is about what the formal tools actually deliver.* The difficulty of choosing
$\varepsilon$ is not a flaw of differential privacy but the intrinsic difficulty of estimating
privacy risk in context; any sound risk method should be expressible within differential privacy or
explain why not ([setting ε](https://openreview.net/forum?id=tp94g4Vmad)). Machine unlearning is
asked to do two different things, remove information from parameters and suppress it in outputs,
and neither matches what policy expects
([unlearning doesn't do what you think](https://openreview.net/forum?id=mfd6GRW4Az), oral). For the
EU AI Act specifically, each potential use of unlearning has a technical gap that researchers would
need to close ([unlearning and regulation](https://openreview.net/forum?id=0ngi2StMwC), oral).

*Taken together*, the group asks for privacy as a measured, reported quantity, and for honesty about
the gap between a formal guarantee and a legal goal. Still open: privacy metrics for vision that
regulators would accept.

**Papers**

- [Don't call it privacy-preserving or human-centric pose estimation if you don't measure privacy](https://openreview.net/forum?id=upugtLPOxC)
- [More effort is needed to protect pedestrian privacy in the era of AI](https://openreview.net/forum?id=rdeCalg68L) (oral)
- [Stop the Nonconsensual Use of Nude Images in Research](https://openreview.net/forum?id=Ev5xwr3vWh) (oral)
- [Setting $\varepsilon$ is not the Issue in Differential Privacy](https://openreview.net/forum?id=tp94g4Vmad)
- [Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research](https://openreview.net/forum?id=mfd6GRW4Az) (oral)
- [Position: Bridge the Gaps between Machine Unlearning and AI Regulation](https://openreview.net/forum?id=0ngi2StMwC) (oral)

### Data, compute and power

**The ideas behind this group** (a synthesis from the abstracts of about 7 papers).

*The shared problem is concentration: of data value, of compute and of access to frontier models.*

*The first thread is economic.* An analysis of seventy-three public data deals finds that value
accrues to aggregators while documented creator royalties round to zero, and proposes a minimal
market framework ([data deals](https://openreview.net/forum?id=mdKzkjY1dM)). Small producers of
information, such as journalists and researchers, should be able to bargain collectively with AI
builders, helped by data-valuation and federated data-management tools
([collective bargaining](https://openreview.net/forum?id=SbfjBNlJE7)). Arguments against
"democratising AI" are sorted into eight traps that look plausible but fail
([democratising AI](https://openreview.net/forum?id=FjxyAotxtT)).

*The second thread is about efficiency over scale.* Progress should be measured as capability per
resource, with gradient-influence patterns used to decide which parameters and data to train
([capability per resource](https://openreview.net/forum?id=6plSmhBI33)). Adaptive sensing, which
adjusts exposure and other sensor settings at the input, lets small models such as EfficientNet-B0
beat much larger ones such as OpenCLIP-H ([adaptive sensing](https://openreview.net/forum?id=xnNHXepQ9h)).

*The third thread is about access to models.* Frontier labs should be required to release small
"analog" models distilled from their largest ones, so that outside researchers can study safety and
interpretability ([analog models](https://openreview.net/forum?id=xcdlSMYXxD)). The millions of
undocumented models in public repositories should be charted as a graph of models and the weight
transformations between them, which requires learning from weights while handling parameter
symmetries ([model atlas](https://openreview.net/forum?id=BzFMBNqg7R)).

*Taken together*, the papers treat data, compute and models as economic goods whose distribution the
research community can shape. Still open: whether the technical pieces they rely on (data valuation,
transfer from analog models, inference from weights) are reliable enough to carry policy.

**Papers**

- [A Sustainable AI Economy Needs Data Deals That Work for Generators](https://openreview.net/forum?id=mdKzkjY1dM)
- [Collective Bargaining in the Information Economy Can Address AI-Driven Power Concentration](https://openreview.net/forum?id=SbfjBNlJE7)
- [Don't Give Up on Democratizing AI for the Wrong Reasons](https://openreview.net/forum?id=FjxyAotxtT)
- [AI Progress Should Be Measured by Capability-Per-Resource, Not Scale Alone](https://openreview.net/forum?id=6plSmhBI33)
- [Position: AI Should Sense Better, Not Just Scale Bigger: Adaptive Sensing as a Paradigm Shift](https://openreview.net/forum?id=xnNHXepQ9h)
- [Position: Require Frontier AI Labs To Release Small "Analog" Models](https://openreview.net/forum?id=xcdlSMYXxD)
- [We Should Chart an Atlas of All the World's Models](https://openreview.net/forum?id=BzFMBNqg7R)

### Regulation, risk and harm

**The ideas behind this group** (a synthesis from the abstracts of about 7 papers).

*The shared problem is that deployment is outpacing governance, and ML researchers are largely
absent from the rules being written.*

*The first thread names specific harms.* Embodied AI brings physical, informational, economic and
social risks that existing robot and vehicle rules do not cover, and a review of US, UK and EU
policy finds gaps ([embodied AI](https://openreview.net/forum?id=fXiPp3qvrW)). AI-powered lethal
autonomous weapons need a behaviour-based definition so that regulation can be grounded in how the
models actually behave ([military AI](https://openreview.net/forum?id=gXTFLbGUQp)), and a shorter
paper calls for a moratorium on generative AI in any weapon control until hallucinations can be
modelled and predicted ([weapon control](https://openreview.net/forum?id=uEY7kQsiZz)). Real-time,
hyper-personalised generation could become addictive "digital heroin", and should be overseen like
addictive substances, especially for minors
([digital heroin](https://openreview.net/forum?id=1IpHkK5Q8F), oral).

*The second thread is about the relation between regulation and innovation.* Regulation done well
is the foundation of innovation, as in aviation and pharmaceuticals, and the EU AI Act's sandboxes
and impact assessments are offered as a model
([innovation and rights](https://openreview.net/forum?id=PFRandBfSz), oral). NeurIPS should build
scientific consensus on AI policy, following the IPCC model for climate
([NeurIPS and consensus](https://openreview.net/forum?id=a4oXTW1PW2), oral). The analog-models and
unlearning papers in other groups also make regulatory arguments.

*The third thread is about citizens.* School pupils should learn adversarial prompting and ethical
"red-teaming" as a civic skill ([right to red-team](https://openreview.net/forum?id=vlM9rLv5xB)).

*Taken together*, the papers argue that researchers' technical knowledge is needed to make rules that
work. Still open: how a scientific community without a tradition of consensus statements would
produce one, and how to state technical thresholds (reliability, predictability of hallucination)
that a regulator could check.

**Papers**

- [Emerging Risks from Embodied AI Require Urgent Policy Action](https://openreview.net/forum?id=fXiPp3qvrW)
- [Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications](https://openreview.net/forum?id=gXTFLbGUQp)
- [Prohibiting Generative AI in any Form of Weapon Control](https://openreview.net/forum?id=uEY7kQsiZz)
- [Real-Time Hyper-Personalized Generative AI Should Be Regulated to Prevent the Rise of "Digital Heroin"](https://openreview.net/forum?id=1IpHkK5Q8F) (oral)
- [Position: If Innovation in AI systematically Violates Fundamental Rights, Is It Innovation at All?](https://openreview.net/forum?id=PFRandBfSz) (oral)
- [NeurIPS should lead scientific consensus on AI policy](https://openreview.net/forum?id=a4oXTW1PW2) (oral)
- [The Right to Red-Team: Adversarial AI Literacy as a Civic Imperative in K-12 Education](https://openreview.net/forum?id=vlM9rLv5xB)

### Agents, simulation and alignment

**The ideas behind this group** (a synthesis from the abstracts of about 7 papers).

*The shared problem is that LLM-based agents are used to stand in for people and societies without
the theory those uses need.*

*The first thread concerns simulating people.* Experiments on presidential election forecasts and
opinion surveys show that ad hoc LLM-generated personas give systematically biased results, so a
science of persona generation is needed; about one million generated personas are released
([LLM personas](https://openreview.net/forum?id=qh9eGtMG4H)). Social simulation built on "demographics
in, behaviour out" lacks causal, traceable reasoning, and a cognitively grounded alternative with a
causal-path benchmark is proposed ([simulating thought](https://openreview.net/forum?id=EvXWexakZX)).

*The second thread says agent research is ignoring its own history.* Many "multi-agent" LLM systems
lack autonomy, social interaction and structured environments, and re-solve problems that the
multi-agent systems literature addressed long ago
([multi-agent mark](https://openreview.net/forum?id=FfsxgSZW0c)). World models should unify physical
and social dynamics instead of modelling them in separate silos
([world models](https://openreview.net/forum?id=aVwhTcSMl4)).

*The third thread rethinks alignment.* A review of over 400 papers proposes that alignment runs both
ways: AI aligned with humans, and humans supported in adapting to AI
([bidirectional alignment](https://openreview.net/forum?id=PgA9rZoMY8)). Contradictions between
responsible-AI metrics, such as incompatible fairness definitions, should be kept rather than
resolved, as a source of pluralism and even of regularisation
([embracing contradiction](https://openreview.net/forum?id=RT3Jby7v21)). Brain–computer interfaces
combined with LLM agents should be designed as collaborators rather than decoders
([brain–agent collaboration](https://openreview.net/forum?id=VZnOKzQ5qW)).

*Taken together*, the group asks for validity tests before simulated people or societies are used as
evidence. Still open: what would count as validating a simulated population, beyond matching a few
known aggregate outcomes.

**Papers**

- [LLM Generated Persona is a Promise with a Catch](https://openreview.net/forum?id=qh9eGtMG4H)
- [Simulating Society Requires Simulating Thought](https://openreview.net/forum?id=EvXWexakZX)
- [Large Language Models Miss the Multi-agent Mark](https://openreview.net/forum?id=FfsxgSZW0c)
- [World Models Should Prioritize the Unification of Physical and Social Dynamics](https://openreview.net/forum?id=aVwhTcSMl4)
- [Position: Towards Bidirectional Human-AI Alignment](https://openreview.net/forum?id=PgA9rZoMY8)
- [Embracing Contradiction: Theoretical Inconsistency Will Not Impede the Road of Building Responsible AI Systems](https://openreview.net/forum?id=RT3Jby7v21)
- [Embracing Trustworthy Brain-Agent Collaboration as Paradigm Extension for Intelligent Assistive Technologies](https://openreview.net/forum?id=VZnOKzQ5qW)

### AI for science and health

**The ideas behind this group** (a synthesis from the abstracts of about 4 papers).

*The shared problem is carrying ML into fields whose standards of evidence are stricter, or whose
prior knowledge is softer, than ML assumes.*

Foundation models are described as moving science through three stages, from enhancing existing
workflows to co-creation to autonomous discovery
([FMs for science](https://openreview.net/forum?id=PegEYWWXvx)). Physics-informed ML should evolve
into biology-informed ML, working with soft, probabilistic prior knowledge, partial observability and
noisy data, with uncertainty quantification as one of four pillars
([biology-informed ML](https://openreview.net/forum?id=HzGZVYi8fK)). Surrogate models need a reporting
standard covering sampling, model choice and evaluation
([SMRS](https://openreview.net/forum?id=V6DcL5L6CU)). Most relevant to medical imaging: only 9% of
FDA-registered AI health tools include a post-deployment surveillance plan, and monitoring should be
framed as two separate hypothesis tests, one for data change and one for performance degradation,
with explicit error-rate guarantees and few labels
([post-deployment monitoring](https://openreview.net/forum?id=mXBFoHDuil)).

*Taken together*: uncertainty and statistical validity are the price of entry into these fields.
Still open: label-efficient tests that keep their guarantees when labels arrive late and are biased
towards cases that were flagged.

**Papers**

- [Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition](https://openreview.net/forum?id=PegEYWWXvx)
- [Position: Biology is the Challenge Physics-Informed ML Needs to Evolve](https://openreview.net/forum?id=HzGZVYi8fK)
- [SMRS: advocating a unified reporting standard for surrogate models](https://openreview.net/forum?id=V6DcL5L6CU)
- [Statistically Valid Post-Deployment Monitoring Should Be Standard for AI-Based Digital Health](https://openreview.net/forum?id=mXBFoHDuil)

## Outlier papers

Positions that stand out from the rest of the track in subject, form or reach.

- [The Right to Red-Team](https://openreview.net/forum?id=vlM9rLv5xB): the only paper about school
  education, arguing that pupils should learn to attack AI systems as a civic skill.
- [Prompting as Scientific Inquiry](https://openreview.net/forum?id=yZU8kdwafM): a philosophy-of-science
  argument that LLMs are a new kind of organism and prompting is their behavioural science, the
  counterpart to mechanistic interpretability's neuroscience.
- [Prohibiting Generative AI in any Form of Weapon Control](https://openreview.net/forum?id=uEY7kQsiZz):
  the shortest and most absolute position, with a concrete lifting condition: the ban stays until
  hallucinations can be modelled and predicted.
- [Real-Time Hyper-Personalized Generative AI … "Digital Heroin"](https://openreview.net/forum?id=1IpHkK5Q8F):
  draws on addiction research and dopamine-driven feedback to argue for controls like those on
  addictive substances.
- [Embracing Contradiction](https://openreview.net/forum?id=RT3Jby7v21): argues against the common goal
  of resolving conflicts between fairness metrics, claiming that optimising contradictory objectives
  acts as implicit regularisation.
- [Stop DDoS Attacking the Research Community](https://openreview.net/forum?id=R5uuqCAPf8): a
  networking metaphor for a publishing problem, with a concrete fix in community-maintained,
  version-controlled "live surveys".
- [Embracing Trustworthy Brain-Agent Collaboration](https://openreview.net/forum?id=VZnOKzQ5qW): the
  only neurotechnology paper, moving from brain-computer interfaces that decode commands to agents
  that collaborate with users.

## Theory picks

The positions with the most mathematical or statistical content to work through (from abstracts
only).

1. [Setting $\varepsilon$ is not the Issue in Differential Privacy](https://openreview.net/forum?id=tp94g4Vmad):
   the strongest formal claim in the track, that any sound method of estimating privacy risk should be
   expressible within differential privacy or justify why not. Worth working out what "expressible"
   means (hypothesis-testing interpretations of DP and privacy profiles are the natural route) and
   whether empirical attacks such as membership inference fit.
2. [Comparison requires valid measurement](https://openreview.net/forum?id=d7hqAhLvWG): conditions
   under which two attack success rates estimate the same quantity. A clean exercise in measurement
   validity and inference: what has to be held fixed (threat model, judge, prompt distribution) before
   a difference in rates means anything.
3. [Statistically Valid Post-Deployment Monitoring](https://openreview.net/forum?id=mXBFoHDuil): data
   shift and performance degradation as two separate hypothesis tests with controlled Type I and II
   errors. Directly relevant to clinical imaging; the interesting part is sequential testing when
   labels are scarce and delayed.
4. [AI Testing Should Account for Sophisticated Strategic Behaviour](https://openreview.net/forum?id=OMc0BYxND4):
   evaluation as a game between tester and a system that may know it is being tested. Work through the
   stylised scenario and ask what an evaluation can still certify when the system can condition its
   behaviour on being observed.
5. [We Should Chart an Atlas of All the World's Models](https://openreview.net/forum?id=BzFMBNqg7R):
   learning from weights requires handling parameter symmetries (permutations of hidden units,
   rescalings). Good entry point into weight-space learning and why symmetries make it hard.
6. [Embracing Contradiction](https://openreview.net/forum?id=RT3Jby7v21): read alongside the known
   impossibility results for group fairness, and check the claim that optimising conflicting metrics
   regularises; it is a testable statement about multi-objective optimisation.

Also worth a look: [Capability per resource](https://openreview.net/forum?id=6plSmhBI33) (a claim
that updating only high-influence parameters strictly beats full fine-tuning per unit resource,
resting on heavy-tailed influence distributions; check the assumptions behind "strictly"),
[Neither Valid nor Reliable?](https://openreview.net/forum?id=yqKfMr0yvY) (validity and reliability of
LLM judges in measurement-theory terms) and [Biology-informed ML](https://openreview.net/forum?id=HzGZVYi8fK)
(soft, probabilistic priors in place of hard physical laws).

## Relevance to medical imaging

- **Post-deployment monitoring** is the most directly usable position: it frames the question that
  every deployed imaging model faces and points to testable, label-efficient methods.
- **Measurement validity** (red-teaming, LLM judges, benchmarks) carries over to imaging leaderboards,
  where the MIDL reproducibility papers make similar points; see the
  [MIDL 2023 overview](../midl2023-overview/index.html).
- **Privacy** papers on pedestrians and pose estimation apply equally to identifiable medical images,
  as in the re-identification work in the MIDL volumes.
- **Unlearning** limits matter for any request to remove a patient's data from a trained model.

## Sources

- [NeurIPS 2025 Position Paper Track announcement](https://blog.neurips.cc/2025/04/01/neurips-2025-position-paper-track/)
- [Reflecting on the inaugural NeurIPS Position Paper Track](https://blog.neurips.cc/2025/10/02/reflecting-on-the-inaugural-neurips-position-paper-track-a-pilot-year-journey/)
- [Call for Position Papers](https://neurips.cc/Conferences/2025/CallForPositionPapers)
- [OpenReview group for the track](https://openreview.net/group?id=NeurIPS.cc/2025/Position_Paper_Track)
- Paper list taken from the NeurIPS 2025 public schedule data (`neurips-2025-orals-posters.json` on neurips.cc), filtered to papers from the Position Paper Track.
