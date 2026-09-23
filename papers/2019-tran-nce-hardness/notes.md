---
title: "Transferability and Hardness of Supervised Classification Tasks"
category: "Transferability"
subcategory: "Theory"
short_title: "Tran 2019 — NCE"
authors: "Anh T. Tran, Cuong V. Nguyen, Tal Hassner (VinAI Research, Amazon Web Services, Facebook AI)"
venue: "ICCV"
year: 2019
url: "https://arxiv.org/abs/1908.08142"
pdf_url: "https://arxiv.org/pdf/1908.08142"
tags: [transfer-learning, transferability, conditional-entropy, task-hardness, information-theory, source-model-selection]
status: read
---

## Links

- **[arXiv:1908.08142](https://arxiv.org/abs/1908.08142)** — preprint; ICCV 2019. The
  supplemental with the full derivation of Theorem 1 is not in the conference PDF, but the
  proof sketch in the main text is complete enough to follow and to check.
- **[Nguyen 2020 — LEEP](../2020-nguyen-leep/index.html)** — the direct successor, which
  replaces the ground-truth source labels here with a source model's soft predictions.
- **[Bao 2022 — H-score](../2022-bao-hscore-transferability/index.html)** — the third member
  of the set, approaching the same question from feature-space geometry.
- **[Diniz 2026 — PAS](../2026-diniz-pas/index.html)** — the unsupervised counterpart:
  scores a source/backbone pair without any target labels, which this measure requires.
- **[Claßen 2026 — TE robustness](../2026-classen-te-robustness/index.html)** — benchmarks
  this measure against the others on medical imaging targets, and finds the rankings move
  under nothing but a change of random seed.
- **[Runnable code](https://github.com/msrepo/theory_inclined_papers_with_annotations/blob/main/papers/2019-tran-nce-hardness/code/nce.py)** —
  the identity Theorem 1 turns on, the theorem itself, the hardness bound, what $H(Y\mid Z)$
  measures on constructed cases, and a source-selection setting where dropping the
  source-hardness term picks the wrong model. `make verify` runs it.

## In one paragraph

Take two classification tasks over **the same input instances**, each with its own ground-truth
label sequence — $Z$ for the source, $Y$ for the target. Count the joint, compute the
conditional entropy $H(Y\mid Z)$, and that number alone bounds how well a representation trained
on $Z$ will transfer to $Y$. The striking thing is what it does *not* need: no model, no
features, no input data. It is **solution-agnostic** — two lists of labels suffice. The same
quantity, applied with a trivial constant source, estimates task *hardness*. The catch, which
the paper states and the follow-up literature then quietly drops, is that the bound has **two**
terms: the conditional entropy and the source model's own log-likelihood. Only when the source
is held fixed does the second one disappear.

## The spine of the argument

1. Define transferability as the log-likelihood on the target after freezing the source
   representation $w_Z$ and retraining only the head.
2. Build a classifier $\bar k$ that composes the source model's output with the empirical
   conditional $\hat P(y\mid z)$, and assume it lives in the head space $K$.
3. Its log-likelihood lower-bounds the best head's. Expand it, drop all but one term inside
   the log, and what falls out is exactly $l_Z - H(Y\mid Z)$.
4. Hardness is transferability from a *trivial* task, which gives $\mathrm{Hard}(T^Z)\le H(Z)$.

## Setup and notation

| Symbol | Meaning |
|---|---|
| $X=(x_1,\dots,x_n)$ | one input sequence, **shared by both tasks** |
| $Y, Z$ | target and source label sequences over those same inputs, both ground truth |
| $w_Z:\mathcal{X}\to\mathbb{R}^D$ | the representation learned on the source task |
| $h_Z$ | the source head; $k_Y$ the head retrained on the target |
| $l_Z(w_Z,h_Z)$ | average log-likelihood of the source model on $Z$ |
| $\widetilde{\mathrm{Trf}}(T^Z\to T^Y) = l_Y(w_Z,k_Y)$ | transferability, Eq. 5 |
| $\hat P(y,z)$ | empirical joint by **hard counting**, Eq. 6 |
| $H(Y\mid Z)$ | the conditional entropy, Eq. 7 — the measure itself |

The setting assumption is the strong one and worth stating up front: **both tasks are labelled
on the same inputs.** That is what makes counting $\hat P(y,z)$ possible at all, and it is
exactly what LEEP later removes.

## Theorem 1, and the step it turns on

$$
\widetilde{\mathrm{Trf}}(T^Z\to T^Y)\;\ge\; l_Z(w_Z,h_Z)\;-\;H(Y\mid Z)
\tag{10}
$$

The proof constructs a specific classifier $\bar k$ (Eq. 9): push $x$ through the source model
to get $p_Z = h_Z(w_Z(x))$, then convert to a target distribution by averaging the empirical
conditional against it,

$$
p_Y(y) = \mathbb{E}_{z\sim p_Z}\big[\hat P(y\mid z)\big] = \sum_z \hat P(y\mid z)\,p_Z(z).
\tag{9}
$$

Then, assuming $\bar k\in K$ so that $l_Y(w_Z,k_Y)\ge l_Y(w_Z,\bar k)$,

$$
l_Y(w_Z,\bar k) = \frac1n\sum_i\log\Big(\sum_z\hat P(y_i\mid z)P(z\mid x_i)\Big)
\;\ge\; \frac1n\sum_i\log\Big(\hat P(y_i\mid z_i)P(z_i\mid x_i)\Big)
$$

$$
= \underbrace{\frac1n\sum_i\log\hat P(y_i\mid z_i)}_{=\,-H(Y\mid Z)}
\;+\;\underbrace{\frac1n\sum_i\log P(z_i\mid x_i)}_{=\,l_Z(w_Z,h_Z)} .
\tag{11–12}
$$

### Why that first term is *exactly* $-H(Y\mid Z)$ — and why LEEP cannot reuse it

This is the hinge, and it is worth being precise because it resolves something left open in the
LEEP notes. Because $\hat P$ here is the **hard** empirical conditional,

$$
\frac1n\sum_i\log\hat P(y_i\mid z_i)
= \sum_{y,z}\hat P(y,z)\log\hat P(y\mid z)
= -H(Y\mid Z)
$$

on the nose — the code confirms it to $0.00\mathrm{e}{+}00$. The step closes because
$\frac1n\sum_i\log Q(y_i\mid z_i)$ is maximised over conditionals $Q$ by precisely that $\hat P$.

[LEEP](../2020-nguyen-leep/index.html) runs the same argument with a **soft** conditional built
from a source model's softmax. The identity then fails in the wrong direction — the soft
conditional is *dominated* by the hard one, measured at $0.138$ nats below on the same data — so
the one-term-drop lands strictly below NCE and cannot establish LEEP's Property 2. That property
still holds empirically, but by a route this proof does not supply.

### Checking the theorem honestly

The bound is stated for a head space $K$ that **contains $\bar k$**, and the paper's Discussion 2
says how to arrange that: take $K = K'\cup\{\bar k\}$ and pick whichever scores better. That
assumption is load-bearing. Checking the bound against a plain linear head on the frozen
representation — a $K$ that does *not* contain $\bar k$ — produces violations, and they are the
checker's fault, not the theorem's. Done correctly:

| alignment | $l_Z$ | $H(Y\mid Z)$ | bound | actual Trf | holds |
|---|---|---|---|---|---|
| 0.3 | −0.0097 | 1.3355 | −1.3452 | −1.3357 | yes |
| 1.0 | −0.0313 | 1.1113 | −1.1425 | −1.1145 | yes |
| 2.0 | −0.0378 | 1.0104 | −1.0482 | −1.0204 | yes |
| 4.0 | −0.0069 | 0.2827 | −0.2896 | −0.2875 | yes |

Tight throughout, and tightening further as $Y$ becomes more nearly a function of $Z$.

## What $H(Y\mid Z)$ actually measures

Three regimes, all exact in the code:

| case | $H(Y\mid Z)$ | |
|---|---|---|
| $Z$ bijective with $Y$ | 0.0000 | nothing left to learn; transfer is free |
| each $Z$ class splits into 2 | 0.6931 | $\log 2$ |
| each $Z$ class splits into 4 | 1.3863 | $\log 4$ |
| $Z$ trivial, $\lvert\mathcal{Y}\rvert=4$ | 1.3863 | $=H(Y)$ |
| $Z$ trivial, $\lvert\mathcal{Y}\rvert=16$ | 2.7726 | $=H(Y)=4\log 2$ |

**A trivial source gives $H(Y\mid Z)=H(Y)$** — there is nothing to condition on, so the transfer
must supply all of the information. That is the paper's hardest case, and its quoted $4\log 2$ is
the last row: a 16-class target. The measure is, straightforwardly, *how much information about
$Y$ is still missing once you know $Z$.*

## Task hardness

Hardness is the optimal loss, $\mathrm{Hard}(T^Z)=\min_{w,h}\mathcal{L}_Z(w,h)=-l_Z(w_Z,h_Z)$
(Eq. 13). Applying Theorem 1 with a **trivial** constant source $C$ gives

$$
\mathrm{Hard}(T^Z)\;\le\;H(Z\mid C)
\tag{14}
$$

and since conditioning on a constant is no conditioning at all, $H(Z\mid C)=H(Z)$ — the plain
label entropy. So the paper's hardness estimate is, after unwinding, **the entropy of the source
label distribution**. Verified across $\lvert\mathcal{Z}\rvert\in\{2,4,8,16\}$: $H(Z\mid C)$ and
$H(Z)$ agree to four decimals and the bound holds every time.

That is a genuinely cheap estimator and it correlates well in their experiments. It is also
worth seeing plainly for what it is: a task with more, more balanced labels is estimated as
harder, with no reference to the inputs. The paper concedes the point — "this result does not
imply that the input domain has no impact on task hardness; only that the distribution of
training labels already provides a strong predictor" — but the estimator cannot distinguish a
4-class problem that is trivially separable from a 4-class problem that is not.

## The term the later papers drop

Theorem 1 has **two** terms, and this matters more than it looks.

The paper's own stated use is: *fix the source*, and then $l_Z(w_Z,h_Z)$ is a constant, so
ranking multiple **targets** by $-H(Y\mid Z)$ alone is licensed. Explicitly: "when the source
task $T^Z$ is fixed, the log-likelihood $l_Z(w_Z,h_Z)$ is a constant. In this case, the
transferability only depends on the CE $H(Y\mid Z)$."

But NCE is subsequently used — by [LEEP](../2020-nguyen-leep/index.html), by
[H-score](../2022-bao-hscore-transferability/index.html), and as a baseline generally — for
**source selection**: fixed target, varying source. There $l_Z$ is not constant, and dropping it
is an extra, unstated assumption.

Whether it bites depends on whether the two terms move together. Varying them *independently* —
how noisily $x$ carries $Z$ (which drives $l_Z$) against how tightly $Y$ follows $Z$ (which
drives $H(Y\mid Z)$) — they come apart:

| source | $l_Z$ | $H(Y\mid Z)$ | $-H(Y\mid Z)$ | $l_Z-H$ | actual Trf |
|---|---|---|---|---|---|
| clean $x$, loose $Y\mid Z$ | −0.4985 | 1.1940 | −1.1940 | −1.6925 | −1.2562 |
| clean $x$, tight $Y\mid Z$ | −0.5049 | 0.4891 | −0.4891 | −0.9940 | **−0.6710** |
| noisy $x$, tight $Y\mid Z$ | −1.3592 | 0.6126 | −0.6126 | −1.9718 | −1.1275 |
| noisy $x$, loose $Y\mid Z$ | −1.3637 | 1.1254 | −1.1254 | −2.4891 | −1.2914 |
| v. noisy $x$, tight $Y\mid Z$ | −1.3633 | 0.2004 | **−0.2004** | −1.5637 | −0.9732 |

Spearman against actual transfer: $-H(Y\mid Z)$ gives $+0.800$, the full $l_Z-H(Y\mid Z)$ gives
$+0.900$. And NCE's top pick is the *wrong* source — it prefers the very noisy one on the
strength of its low conditional entropy, while the clean source with looser alignment actually
transfers best. The reason is structural: **training on an unpredictable $Z$ yields a
representation that learned nothing, and $H(Y\mid Z)$ cannot see that, because it never looks at
$x$.** Solution-agnosticism is the selling point and the blind spot at once.

To be fair about the strength of this: where source hardness and label alignment *do* move
together — which is common, since a source with many fine-grained classes tends to be both
harder and more informative — a generator varying them in step shows no disagreement at all.
The assumption is real, it is unstated, and it is satisfied often enough that NCE works in
practice.

## How the three measures relate

| | **NCE** (this paper) | [**LEEP**](../2020-nguyen-leep/index.html) | [**H-score**](../2022-bao-hscore-transferability/index.html) |
|---|---|---|---|
| what it reads | two ground-truth label sequences | source model's softmax $\theta(x)$ | penultimate features $h(x)$ |
| needs a trained source model | **no** (for the measure) | yes | yes |
| needs source labels on target inputs | **yes** | no | no |
| the head | $\hat P(y\mid z)$, hard counting | $\hat P(y\mid z)$, soft counting | optimal linear head, eliminated |
| what it estimates | a lower bound, with a second term | a lower bound at one head | a second-order approximation to the supremum |
| assumption | shared input instances | source labels code the target | weak dependence, $o(\varepsilon^2)$ |

The lineage is clean. **NCE needs the source labels but no model; LEEP needs the model but no
source labels** — it replaces the ground-truth $z_i$ with the source model's prediction, and
that single substitution is the entire difference. H-score comes at the same target from the
other side, working in feature space and optimising the head away rather than constructing one.
Read the three together and the design space is essentially: *which representation do you read
(labels, softmax, features), and do you evaluate one head or the best one?*

## Questions and doubts

- **The measure is model-free; the bound is not.** $H(Y\mid Z)$ needs only labels, which is the
  headline. But Theorem 1's right-hand side contains $l_Z(w_Z,h_Z)$, which requires a trained
  source model. The advertised cheapness belongs to the measure alone, and only survives to the
  bound when the source is fixed.
- **The $\bar k\in K$ assumption is load-bearing** and easy to miss. Check the bound against a
  head space that excludes $\bar k$ and it fails; the paper handles this in a Discussion
  paragraph rather than in the theorem statement.
- **Source selection is outside what is licensed**, as the table above shows. Every later paper
  that benchmarks against NCE uses it in exactly that regime.
- **Hardness reduces to label entropy.** Eq. 14 unwinds to $\mathrm{Hard}(T^Z)\le H(Z)$, so the
  estimator cannot separate two tasks with the same label distribution and different intrinsic
  difficulty. The correlations reported are real but the mechanism is blunter than "hardness"
  suggests.
- **Shared input instances is a strong setting assumption**, and the one the paper is most
  candid about — the conclusion flags extending beyond it as future work. LEEP's main
  contribution is removing it.
- **Everything is empirical log-likelihood on the training set**, justified by "the
  non-overfitting assumption holds even for large networks". Discussion 3 sketches a McDiarmid
  route to the expected log-likelihood but does not carry it out.

## Takeaways

- **The measure is two lists of labels and a counting argument.** That it predicts transfer at
  all — no model, no features, no inputs — is the surprising result, and it is why this paper
  seeded the two that followed.
- **$H(Y\mid Z)$ is how much about $Y$ is still missing once you know $Z$.** A bijective source
  gives zero, a trivial source gives $H(Y)$, a $k$-way split gives $\log k$. Nothing subtler is
  going on.
- **The hinge identity is that the hard empirical conditional makes the one-term-drop exact.**
  That is why this proof works and why LEEP's soft version cannot inherit it — the cleanest
  thing to carry between the two papers.
- **Read Theorem 1's two terms, not one.** The paper is explicit that dropping $l_Z$ requires a
  fixed source; the field then used it for source selection anyway. Where source hardness and
  label alignment happen to move together that is harmless, and where they do not, NCE picks the
  wrong model.
