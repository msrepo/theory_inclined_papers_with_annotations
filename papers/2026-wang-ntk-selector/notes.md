---
title: "Mining Useful General Data for Low-Resource Domain Adaptation"
category: "NTK & function space"
subcategory: "Applications"
short_title: "Wang 2026 — NTK-Selector"
authors: "Pingjie Wang, Hongcheng Liu, Yusheng Liao, Ziqing Fan, Yaxin Du, Shuo Tang, Yanfeng Wang, Yu Wang (Shanghai Jiao Tong University)"
venue: "ICML"
year: 2026
url: "https://github.com/applewpj/NTK-Selector"
tags: [ntk, function-space, data-selection, domain-adaptation, llm, lora, low-resource, empirical]
status: read
---

## Links

- **[Official repository](https://github.com/applewpj/NTK-Selector)** — the only link the paper
  gives. Proceedings of the 43rd ICML, PMLR 306, 2026. No arXiv id at time of writing, so
  `make fetch` has nothing to pull; drop the PDF in as `paper.pdf` by hand.
- **[Jacot et al. 2018](../2018-jacot-neural-tangent-kernel/index.html)** — the kernel this
  paper borrows. Read that first.
- **[Fort et al. 2020](../2020-fort-deep-vs-kernel/index.html)** — measures how far real
  training departs from the NTK limit. Worth reading *against* this one: Fort finds the kernel
  moves fast early, this paper needs it to sit still.
- **[Runnable code](https://github.com/msrepo/theory_inclined_papers_with_annotations/blob/main/papers/2026-wang-ntk-selector/code/cross_output_cancellation.py)** —
  the $D$-cancellation of Appendix E.5, checked numerically, including what happens when the
  assumption it rests on is broken. `make verify` runs it.

## In one paragraph

You have 1K domain examples and fine-tuning on them alone makes the model *worse* — Llama3-8B
loses 18.8 points on ContractNLI. You also have 1.8M general-domain chain-of-thought examples.
The paper's opening observation is that mixing in 9K of those **chosen at random** already
beats domain-only fine-tuning, so general data carries usable signal; the question is how to
pick the good ones. Their answer is to score each candidate by how well its gradient aligns
with the domain data's gradients, using the NTK — a criterion about *training dynamics* rather
than semantic similarity, which is what lets it find useful data across a topic gap. Making
that affordable on an LLM takes three moves: an argument that fine-tuning is close enough to
linear that the kernel at the initial checkpoint is a valid proxy (Theorem 3.2), an
approximation that replaces a $128{,}000 \times 128{,}000$ kernel matrix with a scalar
(Definition 3.3), and an error analysis showing the approximation's cost is governed by hidden
width and **not** by vocabulary size (Appendix E.5). The result is +8.7 points against +0.8
for domain-only fine-tuning.

## The spine of the argument

1. The honest objective is bi-level and combinatorial, so it is unusable. Find a surrogate.
2. Under gradient flow the NTK is exactly the object converting per-sample gradients into model
   change, so "will this sample help that sample?" is a kernel entry. That is the right
   surrogate — *if* the kernel is stable.
3. It is not constant, but it is nearly **collinear** with where it started ($\cos_F \ge 0.99$).
   Show that collinearity is enough: a kernel that only changes in length is a clock change,
   and a clock change does not alter which samples influence which.
4. The exact kernel is a $D \times D$ matrix needing $D$ backward passes. Sum the logits first,
   then differentiate once: a scalar, one backward pass.
5. Bound what that costs. The error is the sum of the off-diagonal entries, and it is small
   because those entries have random signs — giving $O(d^{-1/2})$ with $D$ cancelling out.
6. Score candidates by mean kernel alignment with the domain set, take the top $N$.

Steps 3 and 5 are the two places something non-obvious happens, and they are where most of
this page goes.

## Setup and notation

| Symbol | Meaning |
|---|---|
| $\mathcal{D}$ | the small domain dataset, $\lvert\mathcal{D}\rvert \approx 1000$ |
| $\mathcal{C}$ | the general candidate corpus, $\lvert\mathcal{C}\rvert = 1.8\text{M}$ |
| $\mathcal{S} \subseteq \mathcal{C}$ | the subset to select, $\lvert\mathcal{S}\rvert = N = 9000$ |
| $\mathcal{S}_{\mathrm{pre}}$ | coarse pre-selected pool, size $M$ (default $4N$) |
| $D$ | output dimension = **vocabulary size**, $\approx 128\text{k}$ |
| $d$ | hidden width, 4096 for both models studied |
| $P$ | parameter count entering the gradient (LoRA only, $\approx 0.5\%$ of full) |
| $h(x;\psi) \in \mathbb{R}^d$ | final hidden state; $\psi$ = backbone parameters |
| $W \in \mathbb{R}^{D\times d}$ | LM head, rows $w_k^\top$; $\theta = (W,\psi)$ |
| $\Theta(x,x';\theta_t)$ | the NTK; a $D\times D$ matrix in general |
| $\widetilde\Theta(x,x';\theta_t)$ | the **Jacobian-free** approximation; a scalar |
| $J_\psi(x) = \partial h/\partial \psi$ | backbone Jacobian |
| $B(x,x') = J_\psi(x)J_\psi(x')^\top \in \mathbb{R}^{d\times d}$ | backbone Gram matrix |
| $\gamma(t) = \nabla_f \mathcal{L}$ | loss gradient in function space |
| $\varepsilon$ | collinearity slack in Definition 3.1; measured $\approx 0.01$ |
| $\Pi \in \mathbb{R}^{P\times p}$ | Rademacher random projection, $p = 8192$ |

## The objective nobody can solve

$$
\mathcal{S}^* = \arg\max_{\mathcal{S}\subseteq\mathcal{C},\,|\mathcal{S}|=N}
T\big(f(\cdot;\theta_{\mathcal{S}}), \mathcal{T}_{\text{test}}\big),
\qquad
\theta_{\mathcal{S}} = \arg\min_\theta \mathcal{L}(\theta;\mathcal{D}\cup\mathcal{S})
$$

An optimisation inside an optimisation: the inner one is ordinary training, the outer one
searches subsets. Evaluating a single candidate $\mathcal{S}$ means fine-tuning a model, and
there are $\binom{1.8\text{M}}{9000}$ of them.

Worth being blunt about the status of what follows: **the paper never derives its score from
this objective.** It says the score "serves as a practical surrogate". The chain is
*NTK-like* $\Rightarrow$ *fixed-kernel dynamics* $\Rightarrow$ *per-sample influence is additive
and computable once* $\Rightarrow$ *high alignment plausibly means high usefulness*. The last
arrow is a heuristic backed by experiments, not an argument. Equation 1 is framing, not a
premise anything is deduced from.

## Why a kernel shows up at all

The paper states the function-space dynamics without deriving them, which hides where the
kernel comes from. Under gradient flow,

$$
\dot\theta = -\eta\nabla_\theta\mathcal{L} = -\eta J(x)^\top \nabla_f \mathcal{L},
\qquad
\dot f = J\dot\theta = -\eta\, \underbrace{J J^\top}_{\Theta}\, \nabla_f\mathcal{L}
= -\eta\,\Theta\gamma .
$$

So $\Theta = JJ^\top$ is not an analogy — it is *exactly* the operator turning loss gradients
into function change. Entry $\Theta(x,x')$ says how much training on $x$ moves the prediction
at $x'$. That is the data-selection question, stated in the model's own dynamics.

Two things follow. First, the criterion is about **optimisation geometry, not semantics**: two
examples can be topically unrelated and strongly NTK-aligned if they share a reasoning format
or answer structure, which is precisely the regime a cross-domain miner needs. Second, if
$\Theta$ were constant, the ODE is linear, influence is additive over samples, and each
candidate can be scored **once** with no retraining and no subset enumeration. That is the
property the whole method is buying.

## Theorem 3.2: why collinearity is as good as constancy

Jacot's constancy holds at infinite width from random init. A pretrained LLM under LoRA is
neither, and Fort et al. already showed the kernel moves. So the paper measures what actually
happens and finds a weaker but sufficient property.

**Definition 3.1 (NTK-like).** With $\cos_F(A,B) = \langle A,B\rangle / (\lVert A\rVert_F
\lVert B\rVert_F)$ — the ordinary cosine between the matrices flattened into vectors —

$$
\cos_F\big(\Theta(\cdot,\cdot;\theta_t),\,\Theta(\cdot,\cdot;\theta_0)\big) \ge 1-\varepsilon
\quad \text{for all } t\in[0,T].
$$

Measured above $0.99$ over 20 epochs (their Fig. 2a). The kernel's *magnitude* drifts; its
*direction* barely moves. Read as a claim about learning: fine-tuning amplifies directions the
pretrained model already had rather than building new orthogonal ones — a believable thing to
say about adapting a pretrained model, and notably *not* what Fort et al. observe when training
from scratch.

**The theorem.** If NTK-like on $[0,T]$, then after a change of time variable

$$
\dot f(\cdot;\theta_{t(u)}) = -\eta\,\Theta(\cdot,\cdot;\theta_0)\,\gamma(t(u)) + \Delta(u),
\qquad
\lVert\Delta(u)\rVert \le \eta\lVert\Theta_0\rVert\frac{\sqrt{2\varepsilon}}{1-\varepsilon}\lVert\gamma(t(u))\rVert .
$$

### The proof, in five steps

**1. Split the kernel along $\Theta_0$.** Least-squares projection onto the line through
$\Theta_0$:

$$
\Theta(t) = a^*(t)\Theta_0 + R(t),
\qquad a^*(t) = \frac{\langle\Theta(t),\Theta_0\rangle}{\lVert\Theta_0\rVert^2},
\qquad R(t) = \Theta(t)-a^*(t)\Theta_0 .
$$

$a^*$ is a scale, $R$ is the part pointing somewhere else.

**2. The residual is controlled by the cosine.** Writing
$S(t) = \langle\Theta(t),\Theta_0\rangle/(\lVert\Theta(t)\rVert\lVert\Theta_0\rVert)$, the
decomposition is orthogonal, so Pythagoras gives

$$
\lVert\Theta(t)\rVert^2 = \frac{\langle\Theta(t),\Theta_0\rangle^2}{\lVert\Theta_0\rVert^2} + \lVert R(t)\rVert^2
\;\Longrightarrow\;
\lVert R(t)\rVert = \lVert\Theta(t)\rVert\sqrt{1-S(t)^2}.
$$

Sanity check: $S=1$ gives $R=0$.

**3. Rescale and bound the leftover.** Define $\Theta_{\mathrm{eq}}(t) = \Theta(t)/a^*(t) =
\Theta_0 + E(t)$. Substituting $a^*(t) = \lVert\Theta(t)\rVert S(t)/\lVert\Theta_0\rVert$, the
$\lVert\Theta(t)\rVert$ cancels:

$$
\lVert E(t)\rVert = \frac{\lVert R(t)\rVert}{a^*(t)} = \lVert\Theta_0\rVert\frac{\sqrt{1-S^2}}{S}
\;\le\; \lVert\Theta_0\rVert\frac{\sqrt{2\varepsilon-\varepsilon^2}}{1-\varepsilon}
\;\le\; \lVert\Theta_0\rVert\frac{\sqrt{2\varepsilon}}{1-\varepsilon},
$$

using $S \ge 1-\varepsilon \Rightarrow 1-S^2 \le 2\varepsilon-\varepsilon^2$.

**4. Change the clock.** Substitute the decomposition into $\dot f = -\eta\Theta\gamma$ and let
$u(t) = \int_0^t a^*(\tau)\,d\tau$, so $du/dt = a^*(t)$. By the chain rule,

$$
\frac{df}{du} = \frac{-\eta a^*\Theta_0\gamma - \eta R\gamma}{a^*}
= -\eta\Theta_0\gamma(t(u)) + \Delta(u), \qquad \Delta(u) = -\eta E(t(u))\gamma(t(u)).
$$

**This is the whole idea.** The term along $\Theta_0$ had its scale factor divided out exactly
by the reparameterisation. Only the orthogonal residual survives.

**5. Bound $\Delta$** with $\lVert E\gamma\rVert \le \lVert E\rVert_{\mathrm{op}}\lVert\gamma\rVert
\le \lVert E\rVert_F\lVert\gamma\rVert$ and step 3. $\blacksquare$

The intuition worth keeping: **a kernel that grows or shrinks without turning is just a clock
running fast or slow, and the clock does not change who influences whom.** Only turning
matters, and turning is what $\cos_F$ measures.

## The Jacobian-free kernel

For a multi-output model the NTK is a matrix, $\Theta(x,x';\theta) = J_\theta(f(x))
J_\theta(f(x'))^\top \in \mathbb{R}^{D\times D}$. Building it needs $D$ backward passes per
input and $D\times P$ storage. With $D \approx 128\text{k}$ and $P$ in the billions, that is
not a cost problem, it is an impossibility.

**Definition 3.3.** Sum the logits *first*, then take one gradient:

$$
\widetilde\Theta(x,x';\theta_t) = \Big\langle \nabla_{\theta_t}\sum_{k=1}^D f_k(x;\theta_t),\;
\nabla_{\theta_t}\sum_{k=1}^D f_k(x';\theta_t) \Big\rangle .
$$

Linearity of $\nabla$ makes the inner argument a scalar, so this is **one** backward pass and
the output is a scalar. Expanding shows what you actually get:

$$
\widetilde\Theta(x,x') = \underbrace{\sum_{k} \langle \nabla f_k(x),\nabla f_k(x')\rangle}_{\text{trace of the true NTK}}
\;+\; \underbrace{\sum_{k\neq m} \langle \nabla f_k(x),\nabla f_m(x')\rangle}_{\text{cross-output error}} .
$$

This is exactly the pseudo-NTK of Mohamadi & Sutherland (2022) up to the constant $D$, which is
irrelevant here because selection only uses the *ranking* of scores. Their result covers wide
ReLU networks at random init; Appendix E extends it to finite-width pretrained LLMs.

## The $D$-cancellation

Both quantities are sums over the same $D\times D$ grid of terms $X_{km} = w_k^\top B(x,x')\,w_m$.
The signal is the diagonal; the error is everything else.

The naive worry is severe: there are $D^2-D \approx 1.6\times 10^{10}$ error terms against
$D \approx 128{,}000$ signal terms, outnumbering the signal by a factor of $D$. The resolution
is that **the two sums accumulate by different mechanisms.**

**The diagonal is coherent.** With $B = \beta I_d$, $\;X_{kk} = \beta\lVert w_k\rVert^2 \ge 0$.
A sum of squares — every term positive, nothing cancels:

$$
\mathbb{E}[S] = \sum_{k=1}^D \mathbb{E}\big[\beta\lVert w_k\rVert^2\big] = D\beta\sigma_w^2 \quad\sim\; D .
$$

**The off-diagonal is incoherent.** Assumption E.1 makes $w_k, w_m$ independent and mean-zero,
so $\mathbb{E}[X_{km}] = 0$ for $k\neq m$: each term is as likely positive as negative.
Adding $N$ zero-mean terms of standard deviation $s$ gives $s\sqrt{N}$, not $sN$. With

$$
\mathbb{E}[X_{km}^2] = \beta^2\sum_{i=1}^d \mathbb{E}[w_{k,i}^2]\mathbb{E}[w_{m,i}^2]
= \beta^2 d\Big(\frac{\sigma_w^2}{d}\Big)^2 = \frac{\beta^2\sigma_w^4}{d}
\;\Longrightarrow\; \operatorname{std}(X_{km}) = \frac{\beta\sigma_w^2}{\sqrt d},
$$

the $D^2$ terms give $\operatorname{std}(\Delta) \approx (\beta\sigma_w^2/\sqrt d)\cdot\sqrt{D^2}
= (\beta\sigma_w^2/\sqrt d)\, D \;\sim\; D$ as well. Hence

$$
\frac{\operatorname{std}(\Delta)}{\mathbb{E}[S]} \;\lesssim\;
\sqrt{\frac{D-1}{D}}\cdot d^{-1/2} = O(d^{-1/2}).
$$

$D$ terms marching give $D$; $D^2$ terms stumbling give $\sqrt{D^2} = D$. The growth rates tie,
$D$ divides out, and only the per-term ratio survives. **The vocabulary size — the thing that
made the exact kernel impossible — has no effect on the approximation quality.** That is the
result, and it is specific to the shape of an LLM: huge $D$, moderate $d$.

### Is the $\lesssim$ hiding anything?

The $X_{km}$ are not independent — $X_{km}$ and $X_{kn}$ share $w_k$ — and the paper waves at
"constants arising from the covariance terms". It survives inspection. For $m\neq n$, both
$\neq k$:

$$
\mathbb{E}[X_{km}X_{kn}] = \beta^2\operatorname{Tr}\big(\mathbb{E}[w_kw_k^\top]\,\mathbb{E}[w_mw_n^\top]\big) = 0,
$$

since $\mathbb{E}[w_mw_n^\top] = 0$. Terms sharing one index are **uncorrelated**, which is the
case that matters. Only the transposed pairs $(k,m)$ and $(m,k)$ are correlated, and for
symmetric $B$ they are equal, giving $\operatorname{Var}(\Delta) = 2D(D-1)\operatorname{Var}(X_{km})$
— a factor of 2, not a change of scaling.

### The low-rank part is weaker than it looks

Under the full Assumption E.2, $B = \beta I_d + U\Lambda U^\top$. Setting $u_k = U^\top w_k \sim
\mathcal{N}(0,(\sigma_w^2/d)I_r)$, the spike contribution has

$$
\frac{\operatorname{std}(X^{\mathrm{low}}_{km})}{\mathbb{E}[Y^{\mathrm{low}}_k]}
= \frac{\lVert\Lambda\rVert_F}{\operatorname{Tr}(\Lambda)} = O(1) .
$$

**No $d$ anywhere.** The low-rank cross terms are not individually suppressed at all; they stay
small only because the isotropic bulk dominates the denominator. The paper is honest about
this, and it is where the second term in the final bound comes from:

$$
\frac{\lvert\widetilde\Theta(x,x') - \Theta(x,x')\rvert}{\lvert\Theta(x,x')\rvert}
= O_{\mathbb{P}}\Big(d^{-1/2} + \frac{\sqrt r}{d}\Big).
$$

One free gain worth noting: the LM-head contribution $D\langle h,h'\rangle$ appears identically
in $\Theta$ and $\widetilde\Theta$, so it **cancels from the numerator** while remaining in the
denominator. It can only shrink the relative error.

### Reproduced

`code/cross_output_cancellation.py` builds $W$ and $B$ directly — nothing is differentiated,
since the argument only uses their shapes and distributions — and measures $\lvert\Delta\rvert/S$.
Both sums are computed without ever forming the $D\times D$ grid, via
$\sum_{k,m}X_{km} = (\mathbf{1}^\top W)B(W^\top\mathbf{1})$ and
$\sum_k X_{kk} = \operatorname{Tr}(WBW^\top)$.

Sweeping $D$ over a 64-fold range at $d=512$ ($d^{-1/2} = 0.0442$):

| $D$ | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| $\lvert\Delta\rvert/S$ | 0.0529 | 0.0415 | 0.0324 | 0.0366 |

Flat, to within the sampling noise of a median-of-medians estimate. Sweeping $d$ instead tracks
the predicted rate across a 64-fold range, staying within $0.77$–$1.02\times$ of $d^{-1/2}$:

| $d$ | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|
| measured | 0.0795 | 0.0433 | 0.0170 | 0.0113 |
| $d^{-1/2}$ | 0.0884 | 0.0442 | 0.0221 | 0.0110 |

### What the cancellation is resting on

Everything above needs $\mathbb{E}[X_{km}] = 0$, i.e. Assumption E.1. Give the head rows a
shared direction and the off-diagonal terms become coherent too, at which point the error grows
like $D^2/D = D$. The same sweep with a modest common component added to every row:

| $D$ | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| isotropic | 0.0340 | 0.0454 | 0.0437 | 0.0467 |
| correlated | 0.0514 | 0.1182 | 0.5397 | **1.9938** |

Flat versus a 40-fold climb over the same range. Extrapolated to the real numbers
($D = 128{,}256$, $d = 4096$) a bias of that size gives a relative error of order
$D/\sqrt d \approx 2000$ — the score would be pure noise.

So the gap between "this works" and "this is meaningless" is entirely whether cross-output
gradients have random signs, and no amount of theory settles that. Which is why **Fig. 2b is
the load-bearing measurement in the paper**, not a decoration: predicted $d^{-1/2} =
1/\sqrt{4096} = 0.0156$, measured mean cross-output cosine similarity $0.016$. Two significant
figures on a quantity that could have landed anywhere in $[0,1]$.

## The algorithm

**Stage 1 — coarse pre-selection**, $1.8\text{M} \to M = 4N$. Warm-up LoRA on $\mathcal{D}$
first, embed with $\phi(x)$ = mean of final-layer hidden states, then for each domain point take
its $K$ nearest candidates and score each candidate by **how many domain points voted for it**:

$$
r(x_j) = \sum_{i=1}^{|\mathcal{D}|}\mathbf{1}\big(j \in N_K(d_i)\big).
$$

A reverse-$k$NN vote rather than a mean similarity, which favours candidates close to *many*
domain points over candidates very close to one — a mild coverage effect.

**Stage 2 — NTK scoring**, $M \to N$:

$$
s_j = \frac{1}{|\mathcal{D}|}\sum_{x_i\in\mathcal{D}} \widetilde\Theta(x_i,x_j;\theta),
\qquad \mathcal{S} = \operatorname{Top-}N.
$$

Two engineering moves make this affordable. Gradients are restricted to **LoRA parameters**
($\approx 0.5\%$ of the full gradient dimension), then **randomly projected** to $p = 8192$ with
a Rademacher matrix ($\Pi_{ij} = \pm 1$). The projection is justified by Johnson–Lindenstrauss:
random low-dimensional projections approximately preserve inner products, and the entire score
*is* an inner product, so this is the right tool rather than a convenience. Same seed across
all samples. Gradients are normalised by sequence length so long documents do not win by
default.

Total cost is about **31 A100-hours** for 1.8M candidates, dominated by embedding and gradient
computation, both linear in $\lvert\mathcal{C}\rvert$ and $M$.

## Results at a glance

| | Llama3-8B-Instruct | Qwen3-8B |
|---|---|---|
| Base | 67.9 | 73.5 |
| Domain-only (1K) | 68.7 (+0.8) | 74.4 (+0.9) |
| Random 9K auxiliary | 74.3 | 72.9 *(below base)* |
| LESS | 74.8 | 73.7 |
| TSDS | 74.8 | 75.0 |
| **NTK-Selector** | **76.6 (+8.7)** | **78.6 (+5.1)** |

Four things worth carrying away. Domain-only fine-tuning can be actively harmful (ContractNLI
$59.9 \to 41.1$). Random auxiliary data helps the weaker model and hurts the stronger one, and
methods built for in-domain coreset selection can fail badly when repurposed here — LESS costs
Qwen3 **49.8 points** on FPB. Gains saturate around $\times 10$ auxiliary data, consistent with
the supply of genuinely useful cross-domain samples being finite. And the relative gain is
largest exactly where you would want it: $\lvert\mathcal{D}\rvert = 100$ gives $54.6 \to 68.6$
(+25.6%), against +5.2% at $\lvert\mathcal{D}\rvert = 2000$.

## Table 3, which is better evidence than Figure 2a

$\cos_F \ge 0.99$ is **self-referential** — it compares the kernel to itself at another time. A
kernel can be perfectly stable and tell you nothing about the model's behaviour. Table 3 tests
the downstream consequence instead: if the fixed-kernel picture is right, then literally doing
kernel ridge regression with $\Theta_0$ should land where fine-tuning lands.

The regression is standard. Minimising
$\frac1n\sum_i (g(x_i)-y_i)^2 + \lambda\lVert g\rVert^2_{\mathcal{H}_\Theta}$ over the RKHS —
informally, the space of functions built by stacking copies of the kernel, with the norm
penalising functions that vary between inputs the kernel calls similar — the Representer
Theorem collapses an infinite-dimensional search to $n$ numbers, $g^*(x) = \sum_i
\alpha_i\Theta(x,x_i)$, and setting the gradient to zero gives

$$
\alpha = (\Theta + n\lambda I)^{-1}\mathbf{y},
\qquad \hat y_{\text{test}} = \Theta_{\text{test}}(x_{\text{test}},\cdot)\,\alpha .
$$

No gradient descent anywhere; one matrix inversion at $n=1000$.

| | Financial | Legal | Psychological |
|---|---|---|---|
| Fine-tuning | 77.27 | 52.35 | 85.10 |
| eNTK regression | 71.57 | **54.45** | **88.40** |

The kernel analogue *exceeds* fine-tuning on two of three domains. The right reading is
probably not "kernels beat training" but **"fine-tuning on 1K samples is a weak and unstable
baseline"** — which is this paper's own finding elsewhere. Kernel ridge regression has an
explicit regulariser, a closed form and no optimiser dynamics to destabilise; on 1K samples
that is a real advantage. Which is itself the NTK-like claim, reached from the other direction:
fine-tuning here barely moves the model beyond what a linearisation of it achieves.

## Questions and doubts

- **The theorem is descriptive, not prescriptive.** It says *if* NTK-like *then* fixed-kernel
  dynamics hold up to $\Delta$. It does not establish that fine-tuning *is* NTK-like — that is
  one empirical curve. And at the measured $\varepsilon = 0.01$ the bound is
  $\sqrt{2\varepsilon}/(1-\varepsilon) \approx 0.143$, a **14% relative perturbation**, scaling
  with an uncontrolled $\lVert\gamma\rVert$. It licenses "reasonable proxy", not "tight".
- **$\cos_F$ may be the wrong stability statistic.** Selection depends on the *relative ordering*
  of $\widetilde\Theta(x_i,x_j)$ across candidates. A matrix with a large common component can
  hold $\cos_F \approx 1$ while that ordering shifts. Rank correlation of the scores over
  training is the directly relevant measurement and is not reported.
- **Figure 2b reports magnitudes, not signs.** It appears to plot $\lvert\cos\rvert$ (all values
  $\ge 0$, spiking at zero, mean $\approx$ std). That confirms the per-term *size*, which is
  what $\operatorname{std}(X_{km})$ needs — but a distribution of $\lvert\cos\rvert$ looks
  identical whether signs are balanced or all positive, and the $D$-cancellation lives or dies
  on balance. Reporting $\mathbb{E}[\cos]$ would have cost nothing and closed the gap. Given
  the table above, this is the single measurement I would most want added.
- **Assumption E.1 at a pretrained checkpoint is a strong idealisation.** The paper says so
  itself ("a high-dimensional regularity hypothesis, not an exact statistical description").
  The Neural Collapse justification is suggestive; the outlier-dimension literature it cites
  elsewhere cuts the other way.
- **The score is essentially label-agnostic.** $\widetilde\Theta$ is built from the gradient of
  *summed logits*, not of the loss at the target. It measures whether an input moves the model
  in a direction that matters for domain inputs, not whether the input–output pair teaches the
  right thing. That is closer to a sophisticated similarity than to utility, which may be why
  the plain embedding baseline already reaches 74.9 of the 76.6.
- **No diversity or redundancy control.** $s_j$ is a per-sample mean and selection is greedy
  top-$N$; nothing stops it picking 9K near-duplicates. Given that gains saturate at $\times 10$,
  a submodular objective looks like obvious headroom.
- **The stages are not disentangled.** With $M = 4N$ the kernel stage only picks 1 in 4, so the
  embedding vote is doing much of the work. There is an ablation on $M$ but none isolating
  "pre-selection then random" from the full pipeline.
- **Table 3's tasks were reshaped to make it possible.** FPB's seven sentiment classes were
  collapsed to three for label alignment, so its 77.27 is not comparable to the 81.7/85.5 in
  Table 1. MedMCQA and MMLU-Med were dropped entirely because answer options vary per instance
  — the two most knowledge-intensive tasks, and arguably the ones where fine-tuning would be
  *least* kernel-like. Testing the linearisation only where it is most likely to hold is a real
  selection effect.
- **No fixed-compute comparison.** 31 A100-hours of selection is not free, and "train on 4×
  more random data for the same total budget" is the baseline a practitioner actually faces.

## Takeaways

- The useful reframing: **data selection is a question about training dynamics, not semantics**,
  and the NTK is the object that makes it one. Topic overlap is a proxy; gradient alignment is
  the thing itself.
- **Collinearity is enough where constancy is assumed.** A kernel that changes length but not
  direction is a clock change, and a clock change does not alter who influences whom. That is a
  cheap, reusable weakening of the NTK assumption, and it fits pretrained models far better than
  the infinite-width story does.
- The $D$-cancellation is the transferable piece of maths: **coherent sums grow like $N$,
  incoherent sums like $\sqrt N$**, so a quadratic pile of random-sign terms can be exactly as
  large as a linear pile of positive ones. It is why the trick works on LLMs specifically, where
  $D \gg d$.
- Read against [Fort et al.](../2020-fort-deep-vs-kernel/index.html), the two are consistent
  rather than contradictory. Fort finds the kernel moves violently for two to three epochs when
  training from scratch; this paper fine-tunes a pretrained model for three epochs with LoRA
  and finds it barely turns. The NTK's validity window is a statement about *regime*, and
  fine-tuning sits inside it in a way that training does not.
- The most practical finding is the one the theory does not predict: **auxiliary data helps most
  when domain data is scarcest**, and badly-chosen auxiliary data can cost a strong model 50
  points. Selection matters more as the base model gets better, not less.
