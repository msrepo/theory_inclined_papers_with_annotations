---
title: "MICCAI 2026: image quality assessment — the mathematical constructs"
authors: "Topic notes"
venue: "MICCAI 2026"
tags: [image-quality-assessment, weak-supervision, label-noise, few-shot, positive-unlabelled, miccai]
status: reading
category: "Misc"
subcategory: "Image Quality Assessment"
short_title: "MICCAI 2026 — IQA papers"
---

## Why this page exists

These are the 11 MICCAI 2026 posters and orals tagged *image quality assessment* in
`MICCAI-2026-IQA/miccai2026-radiology-papers-2.csv`. As on the
[domain adaptation page](../miccai2026-domain-adaptation/index.html), each gets a short reading of
**its core idea and the mathematical objects it is built from**: the score, loss or estimator, what
it assumes, and which known tool it is an instance of. Results are left out except where a number
makes a point. Each entry ends with *What to watch*, which names the assumption the maths leans on
or the step that is heuristic rather than derived.

The tag is broad. Six papers grade or localise image quality directly (T-PM-174, W-AM-041,
W-PM-008, W-PM-121, W-PM-139, W-PM-212). T-AM-063 grades *annotation* quality, T-AM-171 uses a
quality classifier to steer a generator, W-AM-042 filters the quality of *pseudo-labels*,
W-PM-123 identifies DICOM series as a quality-control step, and W-PM-194 has quality regression
as one of four downstream tasks. Each entry opens by saying what is being graded.

Every entry has a widget on the **[interactive page](figures/interactive.html)**. The widgets run
on small toy numbers in the browser, not on the papers' data, so they show mechanisms, not
results. Where a toy exposes a gap in a paper, the entry says so.

Equations were reconstructed from pdftotext output. Where a paper gives a construct only in words,
or its printed equation reads oddly, the entry says so rather than guessing silently.

## What is being graded, and from what

| Paper | What is graded | Supervision actually available | Output |
|---|---|---|---|
| T-AM-063 | vessel annotation masks (coronary CTA) | none: one mask per scan | patch noise score, voxel loss weights |
| T-AM-171 | susceptibility distortion in prostate DWI | image-level distorted / undistorted | a distortion generator, then a corrector |
| T-PM-174 | prostate bpMRI diagnostic quality (PI-QUAL) | DWI distortion labels; 5 PI-QUAL cases per class | binary quality call |
| W-AM-041 | quality perception and description by an MLLM | multiple-choice + text, humans in the loop | a fine-tuned MLLM |
| W-AM-042 | surgical-phase pseudo-labels | none: zero-shot foundation model | per-frame phase labels |
| W-PM-008 | echocardiography image quality grade | image-level grade and view | ordinal grade |
| W-PM-121 | whole-slide-image artefacts | pixel masks (benchmark only) | a comparison of 7 QC tools |
| W-PM-123 | DICOM series type (liver MRI) | series labels | series class |
| W-PM-139 | per-pixel fundus visibility | pixel labels for evaluation; pseudo-labels for training | a pixel quality map |
| W-PM-194 | ASL cerebral blood flow maps | unlabelled pretraining, small labelled sets | a pretrained encoder |
| W-PM-212 | atrial LGE-MRI quality grade | volume-level grade and 3 concept grades | grade plus concept attention maps |

The common thread is that quality labels are scarce, coarse or subjective, so most of the
mathematics is about **getting more out of weak labels**: prototypes built from a handful of
examples, positive-unlabelled risks, multiple-instance pooling, statistical filtering of noisy
labels, and generators that manufacture paired data.

## Construct map

A quick index by the kind of mathematics used. Several papers appear more than once.

| Construct | Papers |
|---|---|
| Prototypes and cosine similarity | T-AM-171 (quality prototypes, margin ranking), T-PM-174 (prototypical network), W-AM-041 (failure prototypes as retrieval anchors) |
| Dense or local quality from weak labels | W-PM-139 (non-negative PU learning), W-PM-212 (multiple-instance learning), T-AM-063 (self-consistency, no labels at all) |
| Statistical filtering of noisy labels | T-AM-063 (binned z-score and median), W-AM-042 (per-class loss GMM — see [EM](../expectation-maximization/index.html)) |
| Invariance by gradient reversal | T-PM-174 (b-value), W-PM-212 (concept erasure) |
| FiLM conditioning | T-PM-174 (on the b-value), W-PM-123 (on a metadata value) |
| Generators as paired-data factories | T-AM-171 (flow matching with minibatch OT — see [OT](../optimal-transport/index.html)) |
| Confidence-routed annotation | W-AM-041 (sequence NLL and agreement), W-AM-042 (GMM posterior) |
| Ordinal outputs and ordinal metrics | W-PM-212 (CORN, QWK), W-PM-139 (QWK), W-PM-008 (PLCC, SRCC, KRCC) |
| Pooling operators | W-PM-008 (generalised Lehmer mean), W-PM-212 (gated attention), W-PM-123 (mean over a set, gated slice weights) |
| Overlap metrics under class imbalance | W-PM-121, W-PM-139 |
| Masked reconstruction pretraining and LoRA | W-PM-194 |
| Directed "causal" matrices and hyperbolic embeddings | W-PM-008 |

## Interactive figures

**[The interactive page](figures/interactive.html)** has one widget per paper. Each has a
callout naming the one thing worth trying first.

| Widget | What you can do |
|---|---|
| [T-AM-063 · self-consistency](figures/interactive.html#t-am-063) | Give a query patch a mask error and give some of its twins wrong masks; watch the pair residuals and their median. The median blames a clean patch once noisy twins are the majority; a robust median/MAD baseline makes the same error score three times more negative. |
| [T-AM-171 · learning to distort](figures/interactive.html#t-am-171) | Drag a generated feature around the prototype circle and compare the prototype hinge with a logistic loss on the same cosine gap; the hinge's pull is exactly zero past the margin. Also shows the quadratic step weights. |
| [T-PM-174 · few-shot bpMRI IQA](figures/interactive.html#t-pm-174) | Tilt the embedding by the b-value, choose which b-value the support cases come from, and change ψ, the share of low quality caused by distortion; sensitivity caps near ψ. The readout shows that τ = 1 keeps every probability inside [0.12, 0.88]. |
| [W-AM-041 · MedQ-Engine](figures/interactive.html#w-am-041) | See the binomial tail that defines a persistent failure, split a budget by e_k^α, and route 1200 toy samples by confidence and agreement; shared student–oracle errors lower the review rate and raise the silently kept wrong labels. |
| [W-AM-042 · LaST](figures/interactive.html#w-am-042) | Fit a two-component loss GMM live and see that systematic, easy-to-fit wrong labels pass as clean; see how a uniform class prior pulls Cholec80-like phase shares towards 1/7. |
| [W-PM-008 · GeoCAN](figures/interactive.html#w-pm-008) | Move the pooling from min to mean to max; drag four factor blobs and watch the "causal" matrix, the tournament and the per-factor gains; the readout checks that Eq. (3)'s second factor is redundant and the multi-hop matrix is symmetric. |
| [W-PM-121 · SlideGuard](figures/interactive.html#w-pm-121) | See artefact Dice collapse and artefact-free Dice stay near 1 as artefacts get rarer, for a fixed detector; see which artefacts survive the 25% patch-coverage rule. |
| [W-PM-123 · DICOM series](figures/interactive.html#w-pm-123) | Toggle and edit five DICOM tags; a zero-imputed encoder cannot tell "missing" from "0", the set encoder can; sum pooling shows the tag count that mean pooling drops. |
| [W-PM-139 · FunPiQ](figures/interactive.html#w-pm-139) | Let a classifier memorise the labelled pixels and compare the PN, uPU and nnPU objectives with the true risk; switch the labelled positives to a strict threshold and watch the random-sampling assumption fail. |
| [W-PM-194 · ICHOR](figures/interactive.html#w-pm-194) | Hide patches of a toy CBF slice and fill them by plain interpolation; the curve shows how much of the masked-reconstruction task smoothness alone solves at each masking ratio. |
| [W-PM-212 · AC-MIL](figures/interactive.html#w-pm-212) | Compare CORN's chained heads with independent cumulative heads; move two attention maps and compare the cosine penalty with Jensen–Shannon divergence. |

## The papers

### T-AM-063 · Decoupled Single-Mask Annotation Noise Detection via Cross-Sectional Patch Self-Consistency
Yinheng Zhu, Xiaowei Xu — [arXiv:2607.05965](https://arxiv.org/abs/2607.05965) · [MICCAI page](https://papers.miccai.org/miccai-2026/0248-Paper1603.html) · oral

**What is graded.** The annotation, not the image: which parts of a single vessel mask are
unreliable.

**Core idea.** Coronary CT datasets carry one mask per scan, so there is no second rater to
disagree with. The paper builds the second rater from the dataset itself. Vessels look alike in
cross-section, so a 24 × 24 cross-sectional patch usually has near-identical twins elsewhere, in
the same scan or in other patients. If two patches look the same but their masks differ, at least
one mask is wrong. A per-bin z-score measures "differ more than usual", and a median over all
twins decides *which* patch is to blame. No network is trained to find the noise, so every flag
comes with its evidence: the twin patches themselves.

**Mathematical constructs.**

- *Self-consistency as an implication.* Classic multi-rater consistency says the same image
  must get the same mask, $d_I(I_i,I_j)=0 \Rightarrow d_M(M_i,M_j)\le\epsilon_M$. The
  single-rater version relaxes "the same image" to "a near-identical patch":
$$d_I\big(I^{(p)}_i, I^{(p)}_j\big)\le\epsilon_I \;\Longrightarrow\; d_M\big(M^{(p)}_i, M^{(p)}_j\big)\le\epsilon_M .$$
  A flagged pair is one where the left side holds and the right side fails. Here $d_I$ is the
  mean squared error between image patches and $d_M = 1-\mathrm{IoU}$ between mask patches. The
  threshold $\epsilon_I = 10^{-3}$ on intensities in $[0,1]$ is a PSNR of
  $10\log_{10}(1/10^{-3}) = 30$ dB, a difference the eye cannot see.
- *Bishop (rotation-minimising) frames.* To cut a cross-section you need two directions
  perpendicular to the centreline at every point. The Frenet–Serret frame takes them from the
  direction the curve bends; on a nearly straight segment that direction is undefined, and
  torsion makes it spin, so the same vessel gives patches rotated by arbitrary amounts. The
  Bishop frame carries the normals along by parallel transport. Its equations,
  $T' = k_1N_1 + k_2N_2$, $N_1' = -k_1T$, $N_2' = -k_2T$, have no $N_1\leftrightarrow N_2$ term,
  so the normals turn only as much as the tangent forces them to. More patches then line up
  pixel for pixel, which matters because $d_I$ is a raw MSE with no rotation search.
- *A baseline for how much correct masks disagree.* Even correct masks disagree a little, and
  more so when the images differ more. So the neighbour pairs are binned by $d_I$ into $K=100$
  equal-width bins over $[0,\epsilon_I)$, and each bin gets the mean and standard deviation of
  $d_M$: $\mu_k\approx\mathbb E[d_M\mid d_I\in B_k]$ and
  $\sigma_k\approx\sqrt{\operatorname{Var}[d_M\mid d_I\in B_k]}$. This is a binned regression of
  $d_M$ on $d_I$ whose spread is allowed to change with $d_I$.
- *Pair residual, a within-bin z-score.*
$$r_{ij} = \frac{\mu_k - d_M\big(M^{(p)}_i, M^{(p)}_j\big)}{\sigma_k + \varepsilon},\qquad d_I\big(I^{(p)}_i, I^{(p)}_j\big)\in B_k .$$
  The sign is flipped so that a strongly *negative* $r_{ij}$ means "these masks disagree far
  more than masks of equally similar images usually do".
- *Patch score by median: assigning blame.* $r_{ij}$ is symmetric in $i$ and $j$, so on its own
  it cannot say which mask is wrong. The patch score asks all of $i$'s near-twins,
  $\mathcal N_i=\{\ell : d_I(I_i,I_\ell)<\epsilon_I\}$, and takes the median:
$$R_i = \operatorname{median}\{r_{ij} : j\in\mathcal N_i\}.$$
  If patch $i$ is clean, most of its twins are clean too and the residuals scatter around 0. If
  $i$'s mask is wrong, it disagrees with nearly every twin and $R_i\ll 0$. The median tolerates
  up to half of the twins being noisy themselves (its breakdown point is 50%).
- *From patches to a loss weight.* Each voxel takes the score of its nearest centreline node (a
  Voronoi assignment), $q(v)=\sigma(R(v))$, and
$$Q(v) = 1-\big(1-q(v)\big)\,w(v),\qquad \mathcal L_{qw} = \frac{\sum_v Q(v)\,\ell\big(\hat y(v),y(v)\big)}{\sum_v Q(v)},$$
  where $w(v)$ decays away from the vessel so background voxels keep weight 1.
- *Scale.* About $3\times10^6$ patches, so about $10^{12}$ candidate pairs; a FAISS top-$k$
  search over flattened patches makes this a one-off 6-hour precomputation.

*Interactive:* [T-AM-063 widget](figures/interactive.html#t-am-063).

**What to watch.** $\sigma(0)=0.5$, so an ordinary clean patch ($R\approx0$) gets
$Q = 1-0.5\,w(v)$. Right next to the vessel, clean voxels weigh half as much as background, and
only better-than-average agreement ($R>0$) restores full weight. The map therefore also
re-balances vessel against background, not only noisy against clean. The baseline
$(\mu_k,\sigma_k)$ is estimated from all pairs, noisy ones included, so noise inflates its own
$\sigma_k$ and hides its outliers; the widget shows a median/MAD baseline scoring the same noisy
patch about three times more negative. Pairs are not independent, since a patch sits in many
pairs. The principle detects *inconsistency*: an error that is made the same way on every similar
patch leaves no disagreement and is invisible. Recall depends on retrieval: rotated or
window-shifted twins are missed (the paper's own failure cases), and rare geometry such as
bifurcations has few twins and so no score.

### T-AM-171 · Learning to Distort: Weakly-Supervised Image Quality Transfer for Prostate DWI Correction
Yucheng Tang, Wen Yan, Alexander Ng, … Shaheer Ullah Saeed, Veeru Kasivisvanathan, Yipeng Hu —
[arXiv:2606.18869](https://arxiv.org/abs/2606.18869) · [MICCAI page](https://papers.miccai.org/miccai-2026/0570-Paper4009.html)

**What is graded.** Image-level labels (distorted or not) are turned into supervision for a
generator; the paper's goal is correction.

**Core idea.** There are no paired distorted and undistorted DWI scans, so a corrector cannot be
trained directly. The paper turns the problem around. Learning to *add* susceptibility distortion
is safer than learning to remove it, because a hallucinated error in a degraded image does less
harm than a hallucinated lesion in a "corrected" one. A flow-matching model is trained to carry
undistorted volumes towards distorted ones. A quality classifier's feature space, with one
prototype per class, pushes the late part of each generated trajectory towards the "distorted"
prototype. The synthetic pairs then train an ordinary supervised corrector.

**Mathematical constructs.**

- *A quality feature space.* A 3D ResNet-18 $f_\phi$ maps a volume to $z\in\mathbb R^{512}$. It is
  trained with BCE plus a supervised contrastive loss, which pulls same-class features together
  on the unit sphere:
$$\mathcal L_{con} = \frac{1}{\lvert B\rvert}\sum_{i}\frac{-1}{\lvert P(i)\rvert}\sum_{p\in P(i)}\log\frac{\exp(\cos(z_i,z_p)/\tau)}{\sum_{j\neq i}\exp(\cos(z_i,z_j)/\tau)},\qquad \tau=0.1,$$
  where $P(i)$ holds the other samples of $i$'s class.
- *Two quality prototypes, by margin ranking.* With $f_\phi$ frozen, vectors $p_0$
  (undistorted) and $p_1$ (distorted) are learned so that every feature is at least $m$ more
  cosine-similar to its own prototype than to the other:
$$\mathcal L_{proto} = \frac1{\lvert B\rvert}\sum_i \max\big(0,\ \cos(z_i,p_{1-c_i}) - \cos(z_i,p_{c_i}) + m\big),\qquad m=0.2 .$$
- *Flow matching with minibatch OT pairing* (see [optimal transport](../optimal-transport/index.html)).
  Within each minibatch, undistorted $x_i$ and distorted $y_j$ are paired by an optimal
  transport plan. The straight path $\hat y^t = (1-t)x_i + t y_j$ has constant velocity
  $u = y_j - x_i$, and a network learns it:
$$\mathcal L_{FM} = \mathbb E_{(x_i,y_j),\,t\sim U(0,1)}\big\lVert \hat v_\theta(\hat y^t, t) - (y_j - x_i)\big\rVert_2^2 .$$
  Generation is Euler integration, $x^k = x^{k-1} + \hat v_\theta(x^{k-1}, t_{k-1})/K$, with
  $K=20$.
- *Prototype guidance on the generated trajectory.* During training the model is also rolled
  out, each late state $x^k$ ($k\ge k_0=10$) is encoded, $z^k=f_\phi(x^k)$, and pushed towards
  $p_1$ with the same hinge:
$$\mathcal L_{PG} = \sum_{k=k_0}^{K} w_k\,\max\big(0,\ \cos(z^k,p_0) - \cos(z^k,p_1) + m\big),\qquad w_k = \Big(\frac{k-k_0}{K-k_0}\Big)^2 .$$
  The quadratic weight grows from 0 at $k_0$ to 1 at $K$, because early states are still noisy
  and the encoder's reading of them is unreliable. The total loss is
  $\mathcal L_{FM} + \lambda_{PG}\mathcal L_{PG}$ with $\lambda_{PG}=1$, and gradients are
  accumulated every 3 steps to fit the rollout in memory.
- *Stage two.* The synthetic pairs $(x_i, \hat y_i = x_i^K)$ train a 3D U-Net corrector with
  $\mathcal L_{corr} = \mathbb E\lVert x_i - G_\psi(\hat y_i)\rVert_1$.
- *Realism, measured by harm.* A distortion counts as realistic if a frozen PI-RADS/Gleason
  classifier (ProFound) does worse on it: lower downstream accuracy and AUC are better for the
  generator.

*Interactive:* [T-AM-171 widget](figures/interactive.html#t-am-171).

**What to watch.** The hinge saturates too. Once $\cos(z,p_1)-\cos(z,p_0)\ge m$,
$\mathcal L_{PG}$ and its gradient are exactly zero, so prototype guidance stops at a fixed cosine
gap of 0.2. The ablation's explanation, that BCE stops pulling once a sample is classified while
PG "pulls deeper", holds only inside the margin band; a logistic loss decays but never reaches
zero. What PG changes is the *direction* of the pull (towards $p_1$ rather than along a
classifier normal) and where it stops. "Lower downstream accuracy" rewards any degradation that
confuses the classifier, including blur or destroyed lesions, so it is necessary but not
sufficient for realistic susceptibility geometry. The generator is trained against the same
encoder that defines "distorted", which invites encoder-specific shortcuts, as with adversarial
examples. And the corrector learns to invert the generator's distortion, not the scanner's; the
gap between the two is what the real-distortion results have to cover.

### T-PM-174 · Bridging Single Distortion Artifacts and Multifactorial Clinical Quality: Few-Shot Biparametric MRI Quality Assessment via Distortion-Trained Prototypical Networks
Yucheng Tang, Alexander Ng, Wen Yan, … Veeru Kasivisvanathan, Yipeng Hu —
[arXiv:2606.18872](https://arxiv.org/abs/2606.18872) · [MICCAI page](https://papers.miccai.org/miccai-2026/0133-Paper3982.html)

**What is graded.** Whether a prostate bpMRI exam is diagnostic (PI-QUAL ≥ 4), learned from DWI
distortion labels.

**Core idea.** Low-quality exams are rare (6% of the PRIME trial has PI-QUAL < 4), and most of
the rare DWI problems are distortion (87%). So the paper trains an embedding on the plentiful,
comparatively objective distortion labels by episodic prototypical meta-learning. It then freezes
the embedding and recomputes the two class prototypes from just five curated examples per class
of the new task: distortion at a new site, or PI-QUAL itself. T2WI enters as a second branch so
the network can tell real anatomy from DWI warping. FiLM plus a gradient-reversal adversary make
the DWI features insensitive to the b-value.

**Mathematical constructs.**

- *Episodes.* A 2-way 5-shot 5-query episode draws a support set (5 labelled examples per class)
  and a query set, and the network is trained so that prototypes built from the support classify
  the queries. Training on many tiny tasks rehearses deployment, where a new task arrives with
  five examples per class.
- *Masked instance normalisation.* With $m=\mathbb 1[x>\bar x]$ and $\bar x$ the global mean
  intensity, voxels inside $m$ are standardised to zero mean and unit variance and the rest are
  set to 0, so background air and rectal gas cannot drive the prototypes.
- *FiLM conditioning on the b-value.* The b-value (1400–2000 s/mm²) is one-hot encoded as $b$,
  embedded as $e_b = E^\top b$, and turned into a per-channel scale and shift,
  $[\gamma(b),\beta(b)] = W_E e_b + b_E$, $F' = \gamma(b)\odot F + \beta(b)$, initialised at
  $\gamma=1,\beta=0$. In plain words: one learned affine correction of every feature channel per
  b-value.
- *Gradient reversal for invariance.* A discriminator reads $F'$ and predicts the b-value,
  $\hat b = \operatorname{softmax}(W_D F' + b_D)$, with $\mathcal L_{adv} = \mathrm{CE}(b,\hat b)$.
  The gradient reversal layer is the identity going forward and multiplies the gradient by
  $-\alpha$ going back. So the discriminator gets better at reading $b$, while everything
  upstream, FiLM included, is pushed to make $b$ unreadable: a minimax game played in a single
  backward pass. FiLM supplies the capacity (a per-b affine correction), the adversary supplies
  the objective (remove what the discriminator can see).
- *Cosine prototypes.* Support embeddings are $\ell_2$-normalised and averaged per class,
  $P_c = \frac{1}{\lvert S_c\rvert}\sum_{i\in S_c} z_i/\lVert z_i\rVert_2$, and a query is scored
  by a softmax over scaled cosines,
$$P(y=c\mid x) = \frac{\exp\big(\tau\cos(z,P_c)\big)}{\sum_{c'}\exp\big(\tau\cos(z,P_{c'})\big)},\qquad \tau = 1,$$
  trained with $\mathcal L = \mathrm{CE} + \lambda\,\mathcal L_{adv}$, $\lambda=0.5$.
- *Adaptation is non-parametric.* At test time nothing is trained. Five curated support cases per
  class give new prototypes, and every later scan goes to the nearer one. For PI-QUAL the two
  "classes" are simply PI-QUAL ≥ 4 and < 4.
- *Metrics that ignore the prior.* Balanced accuracy, sensitivity at 80% specificity and
  specificity at 80% sensitivity are sensible with 31 positives in 483.

*Interactive:* [T-PM-174 widget](figures/interactive.html#t-pm-174).

**What to watch.** With $\tau=1$ and two classes,
$P(y=1\mid x)=\sigma\big(\cos(z,P_1)-\cos(z,P_0)\big)$. The cosine difference lies in $[-2,2]$,
so every probability lies in $[\sigma(-2),\sigma(2)] = [0.119, 0.881]$ and the query loss never
falls below $-\ln 0.881 = 0.127$. Every query keeps pulling on the embedding however well it is
classified, which acts like strong label smoothing; cosine prototypes usually use a scale of
10–30. The transfer to PI-QUAL has a ceiling. A frozen embedding trained only on distortion
places a low-PI-QUAL exam with another cause (T2 motion, noise) near the "good" prototype, so
sensitivity is capped by the share of low PI-QUAL that is distortion; the widget's ψ slider shows
the cap. The support set is hand-curated for diversity, so that choice is part of the method. And
a linear discriminator enforces only what a linear read-out can see, roughly equal feature means
per b-value.

### W-AM-041 · MedQ-Engine: A Closed-Loop Data Engine for Evolving MLLMs in Medical Image Quality Assessment
Jiyao Liu, Junzhi Ning, Wanying Qu, Lihao Liu, Chenglong Ma, Junjun He, Ningsheng Xu —
[arXiv:2603.19863](https://arxiv.org/abs/2603.19863) · [MICCAI page](https://papers.miccai.org/miccai-2026/0640-Paper1141.html)

**What is graded.** Quality perception (multiple-choice questions) and quality description (free
text) by a multimodal LLM, across MRI, CT, endoscopy, fundus and histopathology.

**Core idea.** MLLM errors on medical IQA are concentrated in a few capability × modality cells,
so annotating more data uniformly is wasteful. The engine repeats *evaluate → explore → evolve*.
It finds the dev-set questions the model gets *persistently* wrong, clusters them into failure
prototypes, retrieves look-alike images from a million-image pool, annotates them cheaply (GPT-4o
drafts, human review only where needed), fine-tunes, and repeats. Most of the mathematics is in
the selection and routing rules.

**Mathematical constructs.**

- *The target problem, stated and set aside.* Choose $D_{train}\subset U$ with
  $\lvert D_{train}\rvert\le B$ to maximise test performance of the fine-tuned model. This
  subset-selection problem is combinatorial, and the loop approximates it greedily, one round at
  a time.
- *Persistent failures.* Each dev sample is answered $R=5$ times (sampling makes answers vary)
  and enters the failure pool when its error rate exceeds $\gamma=0.6$, that is, when at least 4
  of 5 answers are wrong. If a sample's per-answer error probability is $p$, the chance of being
  flagged is a binomial tail,
$$\Pr(\text{flag}\mid p) = \Pr\big(\mathrm{Bin}(5,p)\ge 4\big) = 5p^4(1-p) + p^5,$$
  which is 0.19 at $p=0.5$ and 0.74 at $p=0.8$. A coin-flip question is rarely flagged; a
  consistently wrong one usually is.
- *Failure prototypes.* Failures are embedded (image plus question/answer features) and
  clustered agglomeratively. The number of clusters is chosen by the silhouette
  $s(i) = \frac{b(i)-a(i)}{\max\{a(i),b(i)\}}$, where $a$ is the mean distance to the own cluster
  and $b$ to the nearest other cluster. The centroids $p_j$ are the prototypes.
- *Where the model is weak.* For each capability dimension $k$,
  $e_k = \frac{\lvert\{b\in\mathcal B : c_k(b)=1\}\rvert}{\lvert\{s\in D_{dev}: c_k(s)=1\}\rvert}$,
  the dev error rate restricted to questions that test $k$.
- *Retrieval and budget.* Candidates are pool images whose BiomedCLIP embedding is close to a
  prototype's visual part, $\mathcal N(p_j) = \{x : \cos(p_j^{vis}, f_{enc}(x)) > 0.75\}$. The
  budget is split with weights $w_k\propto e_k^{\alpha}$: $\alpha=0$ is uniform, $\alpha=1$
  proportional to error, and large $\alpha$ spends everything on the worst dimension.
- *Routing by self-confidence.* The model's confidence in its own answer $\hat y^{self}$ is the
  per-token average negative log-likelihood,
$$H^{traj}(x,q) = -\frac{1}{\lvert\hat y^{self}\rvert}\sum_{l}\log M_{\theta}\big(\hat y^{self}_l\mid x,q,\hat y^{self}_{<l}\big),$$
  the log-perplexity of its own answer (a one-sample estimate of per-token entropy). With the
  agreement $\delta = R(\hat y^{self},\hat y^{GPT})$ there are three routes: if $H\ge\tau_H$, use
  GPT-4o's label; if $H<\tau_H$ but $\delta<\tau_{ann}$, send the sample to an expert; otherwise
  keep the model's own label.
- *Evolve.* Deduplicate by perceptual hash, drop near-duplicate texts by TF-IDF, and fine-tune
  with the usual token-level NLL,
  $\mathcal L_{SFT} = -\mathbb E\sum_i\log p_\theta(y_i\mid y_{<i},x,q)$.

*Interactive:* [W-AM-041 widget](figures/interactive.html#w-am-041).

**What to watch.** Two of the three routes skip humans. Route (a) adopts GPT-4o's label exactly
where the student is unsure, and GPT-4o scores only 64.8% on the perception benchmark. Route (c)
passes errors that student and oracle share, and sharing grows as the student is trained on the
oracle's labels; in the widget, raising the shared-error share lowers the review rate while the
silently kept wrong labels rise. The reported 18% review rate follows from the thresholds, and
$\tau_H$, $\tau_{ann}$, $\alpha$ and the agreement function $R$ are not reported. A low $H^{traj}$
means fluent, not correct, so confidently wrong answers go to route (c). And retrieval uses the
visual half of a centroid built from joint features; a centroid need not look like any real
image.

### W-AM-042 · Large-Small Model Collaboration for Zero-Shot Surgical Phase Recognition
Yiyi Zhang, Ying Zheng, Wenxin Fan, … Zheng Li, Pheng-Ann Heng —
[arXiv:2608.22879](https://arxiv.org/abs/2608.22879) · [MICCAI page](https://papers.miccai.org/miccai-2026/0546-Paper1153.html)

**What is graded.** The quality of pseudo-labels, not of images: a "dynamic quality control" step
decides which frame labels to trust, which is why the paper carries the IQA tag.

**Core idea.** A surgical vision–language model labels frames zero-shot but has no sense of time,
so its labels flicker. A small temporal network (MS-TCN) models time well but needs labels. LaST
lets the large model label, and two small models learn from the labels they find trustworthy,
cross-teaching each other, with a smoothness loss and a class-balance prior. Their ensembled
output replaces the labels, and the loop runs three times. Adaptation and evaluation both use the
unlabelled test videos, so the setting is transductive.

**Mathematical constructs.**

- *Zero-shot labels.*
  $\hat y_t^{(0)} = \arg\max_{c}\ \mathrm{Sim}\big(M_L(x_t), \text{text}(P_c)\big)$: the phase
  whose text prompt is closest to the frame embedding.
- *Small-loss selection with a per-class two-component GMM.* Networks fit clean labels before
  they memorise wrong ones, so early in training a wrong label tends to have a large loss.
  Within each pseudo-class $c$, the per-frame losses are modelled as
$$f_c(\ell) = \sum_{k=1}^{2}\pi_k\,\mathcal N(\ell\mid\mu_k,\sigma_k^2),$$
  fitted by EM (see [EM and Gaussian mixtures](../expectation-maximization/index.html)). The
  component with the smaller mean is "clean", and each frame gets the responsibility
  $w_t = \Pr(\text{clean}\mid\ell_t) = \pi_1\mathcal N(\ell_t\mid\mu_1,\sigma_1^2)/f_c(\ell_t)$.
  Frames with $w_t>\tau=0.9$ form the labelled set $X_L$. Fitting per class stops a rare phase,
  whose frames all have larger losses, from being declared noise wholesale.
- *Cross-learning (co-teaching, DivideMix style).* Model $\theta_1$ trains on the clean set chosen
  by $\theta_2$ and vice versa, so one network's confident mistakes are not fed straight back into
  itself.
- *Three losses per model.*
$$\mathcal L^{\theta_i}_{sparse} = -\frac{1}{\lvert X_L^{\theta_j}\rvert}\sum_{x_t\in X_L^{\theta_j}}\log p_{\theta_i}(\hat y_t\mid x_t),\qquad \mathcal L^{\theta_i}_{temp} = \frac{1}{T-1}\sum_{t=2}^{T}\big\lVert\log p_{\theta_i}(\cdot\mid x_t) - \mathrm{sg}[\log p_{\theta_i}(\cdot\mid x_{t-1})]\big\rVert_2^2,$$
$$\mathcal L^{\theta_i}_{reg} = D_{KL}(\phi\,\Vert\,\bar p_{\theta_i}) = \sum_{c}\frac1C\log\frac{1/C}{\bar p_{\theta_i,c}},\qquad \bar p_{\theta_i} = \frac{1}{\lvert V\rvert}\sum_{x\in V}p_{\theta_i}(\cdot\mid x).$$
  The temporal term pulls each frame towards its predecessor, and the stop-gradient makes the
  pull one-way, forwards in time. The KL term compares the video-average prediction with a
  uniform prior. It grows without bound if any phase's average probability goes to 0, which is
  what prevents collapse onto one phase. The unlabelled set $X_U$ enters no loss, hence "sparse".
- *Cycle replay.* After a cycle, the ensemble of $\theta_1$ and $\theta_2$ relabels the video,
  $\hat Y^{(r)}$ replaces $\hat Y^{(r-1)}$, and the GMM split is redone on the better labels, for
  $r=3$ cycles.

*Interactive:* [W-AM-042 widget](figures/interactive.html#w-am-042).

**What to watch.** The class-balance prior is uniform, but surgical phases are not: in a
cholecystectomy two dissection phases take most of the time. A toy calculation makes the pull
visible. If the only other pressure were a cross-entropy to the true phase shares $q$, minimising
$-\sum_c q_c\log\bar p_c + \lambda D_{KL}(u\Vert\bar p)$ gives $\bar p = (q+\lambda u)/(1+\lambda)$,
so at $\lambda = 1$ the predicted shares sit halfway between truth and uniform. The real losses
act per frame, so the pull is weaker, but it points the same way: short phases get lengthened.
Loss-based selection finds labels that are *hard to fit*, not labels that are *wrong*. A confusion
the foundation model makes systematically (two phases that look alike) is easy for the small model
to fit, has a low loss, and is selected as clean; the widget shows the kept set's clean ratio
falling as such labels increase. And a per-class GMM on a rare class has few points to fit.

### W-PM-008 · GeoCAN: Nonlinear Causal-Geometric Learning for Echocardiography Quality Assessment
Yiran Li, Kai Zheng, Shuo Li — [MICCAI page](https://papers.miccai.org/miccai-2026/0419-Paper0202.html)
(no preprint listed in the paper list)

**What is graded.** An echocardiography image's quality grade (11 levels on CACTUS, 5 on the
other datasets), jointly with its view.

**Core idea.** Quality should be read from the relations between visual factors (chambers, walls,
valves), not from appearance that merely co-occurs with good images. GeoCAN builds a directed
factor-by-factor matrix from how much of each factor map's activation is covered by another's.
It turns the asymmetric part of that matrix into a per-factor gain and adds a hyperbolic ranking
loss on multi-hop relations. The prediction head is a Kolmogorov–Arnold-style spline network.
"Causal" here means directed co-activation; no intervention is modelled.

**Mathematical constructs.**

- *A tunable pooling operator (generalised Lehmer mean).* For a spatial map $F_k$ with $S$
  positions,
$$\mathrm{GLM}_{\alpha,\beta}(F_k) = \frac{1}{\ln\alpha}\ln\frac{\sum_s\alpha^{(\beta+1)F_k(s)}}{\sum_s\alpha^{\beta F_k(s)}} .$$
  Two readings help. First, with $x_s=\alpha^{F_k(s)}$ it is $\log_\alpha$ of the Lehmer mean
  $L_p(x)=\sum_s x_s^p/\sum_s x_s^{p-1}$ at $p=\beta+1$, and Lehmer means run from the minimum
  ($p\to-\infty$) through the arithmetic mean ($p=1$) to the maximum ($p\to\infty$). Second, with
  $\kappa=\ln\alpha$ it equals $\frac1\kappa\ln\sum_s\pi_s e^{\kappa F_k(s)}$ with weights
  $\pi=\operatorname{softmax}(\beta\kappa F_k)$. So $\beta$ decides which positions the pool
  listens to, $\kappa$ how max-like the average over them is, and the result always lies between
  $\min_s F_k$ and $\max_s F_k$. It tends to the max as $\kappa\beta\to+\infty$, to the min as
  $\kappa\beta\to-\infty$, and to the mean as $\kappa\to0$.
- *A "conditional" matrix.* By analogy with $P(A\mid B) = P(A\cap B)/P(B)$, with GLM playing the
  role of a soft measure,
$$C_{ij} = \frac{\mathrm{GLM}_{\alpha,\beta}\big(F^{(i)}\odot F^{(j)}\big)}{\mathrm{GLM}_{\alpha,\beta}\big(F^{(j)}\big)+\varepsilon} .$$
  Near mean pooling this is "how much of $j$'s activation also lies under $i$". If a large factor
  $i$ covers a small factor $j$, $C_{ij}$ is large and $C_{ji}$ small: the matrix is asymmetric.
- *A tournament from the asymmetric part.* With $A = C - C^\top$ (antisymmetric) and
  $P=\sigma(A/\tau)$, every pair satisfies $P_{ij}+P_{ji}=1$, so $P_{ij}$ reads as the
  probability that factor $i$ "beats" factor $j$ in a round-robin tournament. The per-factor
  score is
$$g = \mathrm{Norm}\Big(\sigma\big(\tfrac{A}{\tau}\big)\big(1-\sigma(\tfrac{A}{\tau})\big)^{\!\top}\mathbf 1\Big) .$$
  Because $1-\sigma(a)=\sigma(-a)$ and $A^\top=-A$, the second factor equals $\sigma(A/\tau)$
  itself, so $g=\mathrm{Norm}(P^2\mathbf 1)$: each factor's wins, weighted by how many wins its
  victims have (a Kendall–Wei-style second-order tournament score). The features are then
  rescaled factor by factor, $X_{cgl} = X\odot(1+r\tanh g)$, a gain between $1-r$ and $1+r$.
- *Hyperbolic embedding and multi-hop ranking.* Factor embeddings are mapped into the Poincaré
  ball by $h_i=\exp_0^c(z_i) = \tanh(\sqrt c\lVert z_i\rVert)\,z_i/(\sqrt c\lVert z_i\rVert)$.
  Multi-hop relations come from the SVD $C=USV^\top$. Even powers give
  $(CC^\top)^n = US^{2n}U^\top$, and a $\lambda$-weighted series of them gives
  $\mathrm{hop} = U\operatorname{diag}(S')U^\top$. A softplus (BPR-style) ranking loss asks
  dominant pairs to score above suppressed ones,
  $\mathcal L_{geo}=\sum\log\big(1+\exp(\mathrm{hop}(i,j^-)-\mathrm{hop}(i,j^+))\big)$.
- *Spline aggregation (KAN-style).* Each token channel passes through learnable univariate spline
  functions, $\mathcal F(t)=\sum_{j=1}^{D}\sum_{m=1}^{M}w_{jm}\phi_{jm}(t_j)$, which are added
  residually, $f' = f+\beta\sum_i\psi_i\mathcal F(t_i)$, and the score is
  $\hat y = \sum_m u_m\psi_m(f')$. A Kolmogorov–Arnold network replaces the usual MLP head.

*Interactive:* [W-PM-008 widget](figures/interactive.html#w-pm-008).

**What to watch.** Nothing here is causal in the do-calculus sense. $C_{ij}$ is a co-activation
ratio, so it cannot tell a cause from a confounder, which is the problem the paper sets out to
solve; the widget also shows the "direction" changing with the pooling knob alone. The
$(1-\sigma)^\top$ factor in Eq. (3) adds nothing beyond $\sigma$ for an antisymmetric argument.
The multi-hop matrix $U\operatorname{diag}(S')U^\top$ is symmetric, so the direction that
$C-C^\top$ was built to keep is discarded by the ranking loss. How $j^+$ and $j^-$ are chosen is
not stated, and as printed Eq. (5) uses $\mathrm{hop}$ rather than hyperbolic distances between
the $h_i$, so the link between the hyperbolic embedding and the loss is not written down. With a
learnable $\alpha$, $\ln\alpha\to0$ is a removable but numerically delicate singularity.

### W-PM-121 · SlideGuard: WSI Artifact Detection Benchmark
Gabriela Kaczmarek, Zuzanna Krawczyk-Borysiak, Mateusz Miller, … Zaneta Swiderska-Chadaj —
[MICCAI page](https://papers.miccai.org/miccai-2026/0969-Paper3442.html) · oral ·
[dataset](https://doi.org/10.5281/zenodo.20339462) · [benchmark](https://www.codabench.org/competitions/16457)

**What is graded.** Seven existing whole-slide-image QC tools, on 53 H&E slides (4 sources, 2
species, 6 tissues) with 2,116 pixel-level annotations of 11 artefact types.

**Core idea.** A benchmark, not a method. Every tool runs with its default settings. Pixel-mask
tools are scored by Dice inside the tissue region, and patch-classification tools against a
ground truth discretised to their own grid. The headline finding, that no tool exceeds 0.6
artefact Dice while artefact-free Dice is high, is largely a property of the metric under heavy
class imbalance, which is why the metric is worth unpacking.

**Mathematical constructs.**

- *Evaluation region.* Tissue is the pixels darker than 200 (8-bit), cleaned by morphological
  opening and closing and checked by hand. Every metric is computed inside it, so tools are not
  scored on their own tissue detection.
- *Two Dice scores from one confusion matrix.* Artefact Dice is
  $\frac{2TP}{2TP+FP+FN}$, and artefact-free Dice $\frac{2TN}{2TN+FP+FN}$ is the Dice of the
  inverted masks. Write the artefact share of tissue as $\pi$, and a detector's sensitivity and
  false-positive rate as $s$ and $f$. Per unit of tissue,
$$\mathrm{Dice}_{art} = \frac{2\pi s}{2\pi s + (1-\pi)f + \pi(1-s)},\qquad \mathrm{Dice}_{free} = \frac{2(1-\pi)(1-f)}{2(1-\pi)(1-f) + (1-\pi)f + \pi(1-s)} .$$
  At $\pi = 2\%$, a detector with 80% sensitivity and 98% specificity scores
  $\mathrm{Dice}_{art}=0.58$ and $\mathrm{Dice}_{free}=0.99$. Its false positives are only 2% of
  the clean tissue, but that is as much area as the artefacts themselves. The same detector at
  $\pi=20\%$ scores 0.85. The "detection asymmetry" the paper reports is what this formula
  predicts.
- *Patch discretisation.* For patch-level tools a patch counts as artefact if annotations cover
  more than 25% of it, predictions are thresholded at 0.5, and Dice becomes the patch F1. Thin
  artefacts (knife marks, scratches, narrow folds) rarely fill a quarter of a patch, so they drop
  out of the patch ground truth. That is why pixel- and patch-mode scores cannot be compared.
- *Label-agnostic recall.* Each tool has its own artefact taxonomy, so per-type precision is
  undefined. The paper reports, per annotated type, the share of its pixels flagged as *any*
  artefact.

*Interactive:* [W-PM-121 widget](figures/interactive.html#w-pm-121).

**What to watch.** Recall without precision rewards over-flagging: a tool that marks everything
scores recall 1 on every type (PathProfiler reaches 1.00 on air bubbles). Per-slide Dice is
unstable when a slide's artefact area is tiny, which is why the near-artefact-free "HQ" slides
switch to artefact-free Dice. All annotations come from one histotechnologist, so there is no
inter-rater ceiling to compare 0.6 against, and for soft-edged artefacts the boundary itself is
uncertain. The intensity-200 tissue mask can clip pale artefacts such as bubbles or faint folds.
Running every tool at its defaults is impartial but also measures configuration fit; as the
authors point out, a nominal "20x" means anything from 0.25 to 0.50 µm per pixel across scanners.

### W-PM-123 · Revisiting Integration of Image and Metadata for DICOM Series Classification: Cross-Attention and Dictionary Learning
Tuan Truong, Melanie Dohmen, Sara Lorio, Matthias Lenga —
[arXiv:2602.23833](https://arxiv.org/abs/2602.23833) · [MICCAI page](https://papers.miccai.org/miccai-2026/0896-Paper3515.html)

**What is graded.** Which MRI series a DICOM series is (T1, T2, DWI, ADC, Dixon, plane, contrast
phase). This is a prerequisite for automated QC and protocol harmonisation rather than image
quality itself.

**Core idea.** The DICOM header is informative but patchy: tags are vendor-specific, edited by
hand, or missing. Instead of imputing missing tags, each slice's metadata is treated as a *set* of
the (tag, value) pairs that are actually present. Each pair is embedded as a learnable per-tag
vector modulated by the value, and the set is averaged. Image tokens (a 2.5D encoder over 10
equidistant slices) and metadata tokens then attend to each other in both directions before a
learned pooling.

**Mathematical constructs.**

- *A series as a set of slices.* From $N$ slices, $S=10$ equidistant ones are kept, giving
  images $x\in\mathbb R^{S\times H\times W}$ and metadata $y\in\mathbb R^{S\times F}$ with NaN for
  missing entries. Slice tokens attend to each other across slices.
- *Sparse metadata encoder: a set encoder with FiLM.* For slice $s$ with observed tags $O_s$,
  each tag $f$ has a learnable embedding $e_f\in\mathbb R^d$, and a small value network reads the
  value and the tag and returns a scale and a shift:
$$(\alpha_{s,f},\beta_{s,f}) = g_\theta\big([v_{s,f}, e_f]\big),\qquad \tilde e_{s,f} = e_f\odot(1+\alpha_{s,f})+\beta_{s,f},\qquad m_s = \mathrm{MLP}\Big(\frac{1}{\lvert O_s\rvert}\sum_{f\in O_s}\tilde e_{s,f}\Big).$$
  A missing tag contributes nothing, rather than a zero, and the mean works for any number of
  observed tags in any order (a Deep Sets-style encoder). The tag identity says *what* the
  number is (echo time, flip angle); FiLM lets the same number mean different things for
  different tags.
- *Bi-directional cross-attention.* After projection to a shared width,
  $V' = \mathrm{MHA}(Q=\tilde V, K=\tilde M, V=\tilde M)$ and
  $M' = \mathrm{MHA}(Q=\tilde M, K=\tilde V, V=\tilde V)$, each followed by a residual
  connection, a feed-forward block and LayerNorm, and then
  $F=\mathrm{GELU}(\mathrm{LN}([V'',M'']W_f))$.
- *Learned slice pooling.* $z=\sum_{s=1}^S w(F)_s F_s$ with $w:\mathbb R^{S\times d_o}\to[0,1]^S$.
  These are gates rather than softmax weights, so they need not sum to one.

*Interactive:* [W-PM-123 widget](figures/interactive.html#w-pm-123).

**What to watch.** Mean pooling forgets how many tags were present. When the pattern of missing
tags is itself informative (vendor, sequence family), only the identities of the present tags
carry that signal; a sum or an explicit count would keep it. The preprocessing section also
appends a binary missingness indicator per categorical tag, which re-introduces missingness as an
observed value, and which encoders see it is not spelled out. Most header tags are constant within
a series, so the $S$ metadata tokens are near-copies, and cross-attention over them adds little
beyond the slice-varying tags. The pooling gates are unnormalised, so the series embedding's scale
grows with $S$. Weighted F1 is dominated by the frequent classes, and the out-of-domain drops
(Dixon opposed-phase 74%, portal venous 65%) look as much like label-definition shift as like
covariate shift.

### W-PM-139 · FunPiQ: A New Benchmark for Pixel-Level Quality Assessment in Fundus Images
Pengwei Wang, José Morano, Virginia Mares, Hrvoje Bogunović —
[arXiv:2606.25915](https://arxiv.org/abs/2606.25915) · [MICCAI page](https://papers.miccai.org/miccai-2026/0410-Paper3819.html) ·
[code and data](https://github.com/penway/FunPiQ)

**What is graded.** Every pixel of a colour fundus photograph, as good, usable or bad, according
to whether the anatomy under it is visible.

**Core idea.** Image-level quality labels disagree across datasets because "gradable" depends on
the downstream task. Pixel-level visibility is task-agnostic, and a task-specific rule ("is the
optic disc in a good region?") can be layered on top. The paper releases 300 pixel-annotated
images from EyeQ, BRSET and mBRSET, two thirds from Brazil and some from mobile cameras, as an
evaluation-only benchmark. It also proposes EFIQA-CP, which trains a small convolutional adapter
on frozen DINOv3 features from pseudo-labels that say "no vessels are visible here". Only
confidently bad pixels are trusted and everything else is treated as *unlabelled*, so it trains
with non-negative positive-unlabelled (nnPU) learning.

**Mathematical constructs.**

- *Positive-unlabelled learning in plain words.* We have some pixels known to be bad (positives)
  and many we know nothing about (unlabelled, a mix of bad and good). Treating every unlabelled
  pixel as good would teach the model that the hidden bad pixels are good. PU learning instead
  uses the class prior $\pi$ (the share of bad pixels, set to 0.05) to subtract the bad pixels'
  expected contribution from the unlabelled set.
- *The unbiased PU risk.* The unlabelled density is the mixture
  $p(x)=\pi p_+(x)+(1-\pi)p_-(x)$, so the loss on true negatives, which we cannot observe, can be
  written with quantities we can:
  $(1-\pi)\,\mathbb E_-[\ell(g(x),-1)] = \mathbb E_u[\ell(g(x),-1)] - \pi\,\mathbb E_+[\ell(g(x),-1)]$.
  The classification risk becomes
$$\hat R_{pu}(g) = \pi\hat R_p^+ + \hat R_u^- - \pi\hat R_p^-,$$
  where $\hat R_p^+$ and $\hat R_p^-$ are the losses on labelled positives scored as positive and
  as negative, and $\hat R_u^-$ is the loss on unlabelled pixels scored as negative.
- *Why it must be clamped.* The last two terms estimate a loss, which cannot be negative, but
  they are a *difference* of two sample averages. A flexible network can drive the difference
  below zero by memorising the labelled positives, and minimising $\hat R_{pu}$ then rewards
  exactly that overfitting. nnPU clamps the estimate (the paper's Eq. 1, with a softplus
  surrogate loss),
$$\mathcal L = \pi\hat R_p^+ + \max\big(0,\ \hat R_u^- - \pi\hat R_p^-\big),$$
  and the training rule of Kiryo et al. (2017) takes a step that *raises* the negative-class term
  whenever it dips below zero.
- *Adapter with context.* Five ConvNeXt blocks with 7 × 7 depthwise convolutions and MLP ratio 1
  (instead of 4) give a large receptive field with little capacity. The fovea, which has few
  vessels by anatomy, can then be told apart from a region where vessels are hidden.
- *Ordinal agreement: quadratic weighted kappa.* With observed confusion counts $O_{ij}$, the
  table $E_{ij}$ expected from the two marginals alone, and weights $w_{ij}=(i-j)^2/(C-1)^2$,
$$\kappa_w = 1-\frac{\sum_{ij}w_{ij}O_{ij}}{\sum_{ij}w_{ij}E_{ij}} .$$
  Calling a bad pixel good costs four times as much as calling it usable, and chance agreement
  scores 0. A binary "reject" task (bad against the rest) adds Dice, AUROC, AUPRC and sensitivity
  at 95% specificity.

*Interactive:* [W-PM-139 widget](figures/interactive.html#w-pm-139).

**What to watch.** PU theory assumes the labelled positives are a *random* sample of all positives
("selected completely at random"). Here they are the pixels the pseudo-labeller was most certain
about, the extreme tail. The classifier then learns what *very* bad looks like, and the
$\pi\hat R_p^-$ correction subtracts the wrong amount; in the widget even the best possible
classifier gets a negative uPU value under strict selection. The prior $\pi=0.05$ is set, not
estimated, and a wrong $\pi$ moves the decision threshold. Each method's metrics are reported at
its own best threshold, which inflates absolute numbers for everyone. Labels come from expert
readers under one ophthalmologist's supervision, with multi-reader agreement left for future work.

### W-PM-194 · ICHOR: A Robust Representation Learning Approach for ASL CBF Maps with Self-Supervised Masked Autoencoders
Xavier Beltran-Urbano, Yiran Li, Xinglin Zeng, … John A. Detre, Sudipto Dolui —
[arXiv:2603.05247](https://arxiv.org/abs/2603.05247) · [MICCAI page](https://papers.miccai.org/miccai-2026/0487-Paper5216.html) · spotlight

**What is graded.** One of the four downstream tasks regresses a quality score in $[0,1]$ for ASL
cerebral-blood-flow (CBF) maps; the paper itself is a pretraining method.

**Core idea.** No pretrained backbone exists for perfusion MRI, and models pretrained on
structural MRI learn edges and tissue boundaries that CBF maps lack. ICHOR pretrains a 3D ViT-B
as a masked autoencoder on 11,405 CBF maps from 14 studies, then adapts it to each small labelled
task with LoRA.

**Mathematical constructs.**

- *Patches and masking.* A $96^3$ map with patch size $P=12$ gives $N=(96/12)^3=512$ patches of
  $12^3=1728$ voxels. A random half ($\rho=0.5$) is hidden, and the encoder sees only the 256
  visible tokens, which is what makes MAE pretraining cheap.
- *Asymmetric encoder and decoder.* The encoder is a ViT-B (12 blocks, width 768, sinusoidal
  positions). A light decoder (4 blocks, width 384) receives all 512 positions: the encoded
  visible tokens, plus one shared learnable mask token at every hidden position. A linear head
  predicts each patch's 1728 voxels.
- *Loss only where the answer was hidden.*
$$\mathcal L_{MSE} = \frac{1}{\lvert\mathcal M\rvert}\sum_{i\in\mathcal M}\lVert\hat p_i - p_i\rVert_2^2 .$$
  Scoring the visible patches would reward copying the input.
- *LoRA adaptation.* Each attention projection $W_0\in\mathbb R^{768\times768}$ (query, key,
  value, output) becomes $W_0 + \frac{\alpha}{r}BA$ with $B\in\mathbb R^{768\times r}$,
  $A\in\mathbb R^{r\times768}$, $r=8$ and $\alpha=16$, and only $A$ and $B$ are trained. That is
  $12\times4\times2\times768\times8 = 589{,}824$ parameters, about 0.7% of the roughly 86M in the
  encoder. Features are global-average-pooled into a LayerNorm and a linear head.
- *The masking-ratio trade-off.* With too little masking a hidden patch is predictable from its
  neighbours; with too much there is not enough context. On the amyloid task $\rho=0.5$ beat 0.25
  and 0.75 (AUC 78.9 against 76.7 and 71.9), lower than the 0.75 usual for natural images.

*Interactive:* [W-PM-194 widget](figures/interactive.html#w-pm-194).

**What to watch.** CBF maps are smooth at the 2 mm scale, so part of the pixel-MSE pretext task
can be solved by interpolating from visible neighbours, without learning anything about perfusion
patterns. In the widget's toy, interpolation alone explains about 80% of the hidden-patch variance
at $\rho=0.5$. The ρ comparison uses one task of 147 subjects with five folds, where a 2-point AUC
gap is within noise. The pretraining set was filtered by an automated QC tool, so the encoder has
mostly seen acceptable maps, which may be why its advantage is smallest on the quality task.
Cohorts of 35 and 63 subjects (AUC 100% on AD against bvFTD) make the classification numbers
fragile. Adapting the structural-MRI baselines with LoRA rather than full fine-tuning partly
confounds the comparison, as the authors note.

### W-PM-212 · AC-MIL: Weakly-Supervised Atrial LGE-MRI Quality Assessment via Adversarial Concept Disentanglement
K M Arefeen Sultan, Kaysen Hansen, Benjamin Orkild, … Ed DiBella, Shireen Elhabian —
[arXiv:2604.10303](https://arxiv.org/abs/2604.10303) · [MICCAI page](https://papers.miccai.org/miccai-2026/0025-Paper5702.html)

**What is graded.** A 4-point quality grade for atrial LGE-MRI volumes, decomposed into three
clinical concepts (sharpness, myocardium nulling, aorta and valve enhancement) plus a residual,
from volume-level labels only.

**Core idea.** Multiple-instance learning can grade a volume from a single label, but it squeezes
all the evidence into one opaque vector. AC-MIL splits that vector into four concept slots, each
with its own attention over patches. Three slots are trained only by their own concept grades,
and a stop-gradient keeps the task loss out of them. The fourth, residual slot carries whatever
else the task needs. Adversaries stop it from re-learning the three concepts, and a penalty stops
the localised concepts' attention maps from overlapping.

**Mathematical constructs.**

- *Nested bags.* A volume is a bag of the $M$ slices that contain the left atrium. Training
  samples $N=8$ slices, and each slice is a bag of $K=80$ random 64 × 64 patches. Only the volume
  carries labels.
- *Concept slots with gated attention.* A shared ResNet gives $h_k$, and four projection heads
  give $h_k^{(c)}$ for $c\in\{sh,nu,ao,un\}$. Each slot pools the patches with its own gated
  attention (Ilse et al.),
$$a_k^{(c)} = w_c^\top\big(\tanh(V_ch_k^{(c)})\odot\sigma(U_ch_k^{(c)})\big),\qquad \alpha^{(c)} = \operatorname{softmax}_k\big(a^{(c)}\big),\qquad Z^{(c)} = \sum_k\alpha_k^{(c)}h_k^{(c)} .$$
- *Asymmetric stop-gradient.* Slices are fused as
  $V_m=[\mathrm{sg}(Z_{sh}),\mathrm{sg}(Z_{nu}),\mathrm{sg}(Z_{ao}),Z_{un}]$, pooled over slices by
  a second attention, $V_{vol} = \sum_m\beta_mV_m$, and graded. The task gradient reaches only the
  residual slot, so the three concept slots cannot quietly absorb task information ("concept
  leakage"). It still reaches the shared ResNet through that slot.
- *CORN ordinal loss.* For $K$ ordered grades, $K-1$ binary heads estimate *conditional*
  probabilities $f_k(x) = \Pr(y>k\mid y>k-1)$, and head $k$ is trained only on samples with
  $y>k-1$. The cumulative probabilities are products, $\Pr(y>k)=\prod_{j\le k}f_j(x)$, so they can
  only decrease as $k$ grows (rank consistency by construction). The predicted grade is
  $1+\sum_k\mathbb 1[\Pr(y>k)>0.5]$. CORN is used for the task and for each concept.
- *Adversarial erasure.* Adversaries $D_c$ try to predict each concept grade from $Z_{un}$,
  $\mathcal L_{adv} = \sum_c\mathcal L_{CORN}(D_c(Z_{un}),y_c)$, through a gradient reversal layer:
  the adversaries get better, and the residual slot learns to defeat them.
- *Spatial attention diversity.* For the three localised slots (sharpness is global and
  excluded),
$$\mathcal L_{SAD} = \sum_{c_i\neq c_j\in\{nu,ao,un\}}\frac{\langle\alpha^{(c_i)},\alpha^{(c_j)}\rangle}{\lVert\alpha^{(c_i)}\rVert_2\lVert\alpha^{(c_j)}\rVert_2} .$$
- *Total.* $\mathcal L = \mathcal L_{task} + 1.0\,\mathcal L_{CBM} + 0.5\,\mathcal L_{adv} + 0.1\,\mathcal L_{SAD}$,
  evaluated by QWK and class-averaged MAE.

*Interactive:* [W-PM-212 widget](figures/interactive.html#w-pm-212).

**What to watch.** Attention weights are non-negative, so the cosine between two attention maps is
at least 0 and reaches 0 only when their supports are disjoint. Two near-uniform maps have cosine
near 1 whatever they attend to, so $\mathcal L_{SAD}$ pushes towards peaky, disjoint attention, not
just different attention. Training uses the cosine, while the evaluation reports Jensen–Shannon
divergence. An adversary at chance shows that *that* adversary cannot decode the concepts from
$Z_{un}$, not that $Z_{un}$ is "statistically independent" of them. Adding the diversity term to
the concept-plus-adversary model lowers QWK from 0.68 to 0.66, so the more interpretable variant
is not the most accurate one. Concept grades have an inter-rater ICC of 0.62–0.72, which bounds
how well the concept slots can be supervised.

## Threads across the papers

- **Prototypes are the default way to use few labels.** T-AM-171 learns two prototypes as targets
  for a generator, T-PM-174 recomputes them from five cases per class, and MedQ-Engine uses
  cluster centroids as retrieval anchors. All three score by cosine similarity. In two of them a margin or a scale
  decides whether a well-placed sample ever stops being pulled: T-AM-171's hinge stops exactly at
  the margin, and T-PM-174's τ = 1 never lets the pull stop. That detail is worth checking
  whenever a paper argues about "pulling deeper" or "saturating".
- **Weak labels need a model of how they are wrong.** FunPiQ's nnPU assumes random selection of
  the labelled positives, LaST's GMM assumes wrong labels are hard to fit, and T-AM-063's median
  assumes most twins are clean. Each fails in a specific, predictable way (extreme-tail labels,
  systematic confusions, majority-noisy neighbourhoods), and each paper's own setting comes close
  to its failure case.
- **Gradient reversal appears twice with the same overclaim risk.** In T-PM-174 and AC-MIL, a
  discriminator at chance is read as invariance or independence. It shows only that a model of
  that capacity cannot read the nuisance.
- **The metric shapes the headline.** SlideGuard's sub-0.6 ceiling is largely class imbalance in
  Dice; FunPiQ reports each method at its best threshold; T-AM-171 measures realism by how much a
  classifier is harmed. Reading the metric's formula first changes how the results read.

## Takeaways

- The strongest ideas here are the *statistical* ones: T-AM-063's model-free consistency test with
  explicit evidence, FunPiQ's move to a positive-unlabelled risk, and T-PM-174's honest use of a
  cheap proxy label (distortion) for an expensive one (PI-QUAL).
- The weakest claims are about *causality* and *independence* (GeoCAN, AC-MIL): the mathematics
  shown measures co-activation or adversary failure, not either of those.
- For applied IQA work, the most transferable pieces are pixel-level quality maps trained from
  anatomy-visibility pseudo-labels with nnPU, few-shot prototypes on a proxy-trained embedding,
  and a failure-driven data loop, each with the caveats listed above.
