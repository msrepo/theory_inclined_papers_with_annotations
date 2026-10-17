---
title: "Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse"
category: "Generative models"
subcategory: "Theory"
short_title: "Lucas 2019 — linear VAE and posterior collapse"
authors: "James Lucas, George Tucker, Roger Grosse, Mohammad Norouzi (University of Toronto, Google Brain)"
venue: "NeurIPS 2019"
year: 2019
url: "https://arxiv.org/abs/1911.02469"
pdf_url: "https://arxiv.org/pdf/1911.02469"
tags: [vae, posterior-collapse, probabilistic-pca, elbo, linear-vae, observation-noise, identifiability]
status: read
summary: "a linear VAE is probabilistic PCA, so collapse can be read off the likelihood: it is a local maximum that a too-large noise level creates, not a product of the KL term"
---

## Links

- **[arXiv:1911.02469](https://arxiv.org/abs/1911.02469)**: the paper, with proofs in appendices A to C.
- **[Interactive version](figures/interactive.html)**: a single column that sits in a valley or on a hill at zero depending on the noise,
  how many latent units survive under fixed and under learned noise, and a decoder you can rotate to see the likelihood stay put while the ELBO moves.
- **[Probabilistic PCA](../probabilistic-pca/index.html)**: the model, its maximum-likelihood solution and its posterior,
  all used here without re-deriving them.
- **[The ELBO](../elbo/index.html)**: the identity $\log p=\mathrm{ELBO}+\mathrm{KL}$ that drives §3.

J. Lucas, G. Tucker, R. Grosse, M. Norouzi, *Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse*, NeurIPS 2019.

## In one paragraph

A VAE sometimes learns to ignore some of its latent units: for those units the encoder outputs the prior, whatever the input. This is
*posterior collapse*, and the usual blame falls on the KL term of the ELBO, which pulls the encoder toward the prior. The paper tests that
blame in the one case where everything can be solved exactly, a VAE whose decoder is linear with Gaussian noise. That VAE is probabilistic PCA
with an inference network attached. Three findings follow. (1) The best linear VAE is the best pPCA model, and a *diagonal* posterior covariance even
pins down the individual principal components, not just their span. (2) The ELBO adds no new local maxima: whatever bad solutions exist were already in the
log-likelihood. (3) Those bad solutions are real. If the noise variance $\sigma^2$ is held at a value larger than the data's variance
along some direction, the model prefers to call that direction noise, and a zero column in the decoder is a stable maximum. Learning $\sigma^2$ makes such points
saddles. So collapse is a property of the probabilistic model and its noise level, and the KL term is not what causes it. The paper then shows in deep Gaussian VAEs that the starting value of $\sigma^2$
strongly changes how many units collapse.

## Setup and notation

| Symbol | Meaning |
|---|---|
| $x\in\mathbb R^n$, $z\in\mathbb R^k$ | data and latent code. The paper uses $n$ and $k$ where the PPCA notes use $d$ and $q$ |
| $W$ | decoder, an $n\times k$ matrix whose columns are directions in data space |
| $\sigma^2$ | observation noise variance, one number for all coordinates |
| $S$, $\lambda_1\ge\dots\ge\lambda_n$ | sample covariance of the data and its eigenvalues |
| $U_k,\Lambda_k$ | top $k$ eigenvectors and eigenvalues of $S$ |
| $M=W^\top W+\sigma^2I$ | the $k\times k$ matrix that appears in the pPCA posterior |
| $V,\;D$ | encoder: $q(z\mid x)=\mathcal N(V(x-\mu),D)$ with $D$ **diagonal** and the same for every $x$ |
| $\tilde M=\mathrm{diag}(W^\top W)+\sigma^2I$ | $M$ with its off-diagonal entries removed (my notation, to keep §3 short) |

The model is $p(z)=\mathcal N(0,I)$ and $p(x\mid z)=\mathcal N(Wz+\mu,\sigma^2I)$, as in pPCA.

## 1. Posterior collapse, and what is being claimed

Posterior collapse for a latent unit means the encoder's answer for that unit is (nearly) the prior for (nearly) every input. Then the unit carries no information about
$x$, and the decoder gets nothing from it. The paper's sentence is "linear VAEs can suffer collapse": a decoder column that is exactly zero is a collapsed unit.

The three possible culprits are the KL term (which pulls $q$ to the prior), the decoder being too powerful (so it ignores $z$), and the likelihood itself. The paper's
claim is that for linear decoders it is the third, which is a stronger and stranger claim than the other two.

## 2. The linear VAE and what it recovers

pPCA's posterior is exactly Gaussian, $p(z\mid x)=\mathcal N(M^{-1}W^\top(x-\mu),\,\sigma^2M^{-1})$. A Gaussian encoder can therefore represent it exactly, apart from one restriction:
the paper uses a *diagonal* covariance $D$. That is only possible without loss when $M$ is diagonal, that is when the columns of $W$ are perpendicular.

**Lemma 1** (paper, §4.2). The global maximum of the ELBO equals the global maximum of the log-likelihood.

*Proof, line by line.* The pPCA maximum has $W=U_k(\Lambda_k-\sigma^2I)^{1/2}R$ for any rotation $R$, and every $R$ gives the same likelihood. Choose $R=I$. Then
$W^\top W=\Lambda_k-\sigma^2I$, so $M=\Lambda_k$, which is diagonal. Setting $V=M^{-1}W^\top$ and $D=\sigma^2M^{-1}=\sigma^2\Lambda_k^{-1}$ gives the true posterior, and a diagonal one. The KL gap in
$\log p=\mathrm{ELBO}+\mathrm{KL}$ is then zero, so ELBO equals likelihood there. And the ELBO can never exceed the likelihood, so this is also the global maximum of the ELBO. $\square$

The weak point of the lemma is only that $D$ is one matrix for all $x$. That is allowed because the pPCA posterior covariance $\sigma^2M^{-1}$ does not depend on $x$ (this is the "same spread for every point" of the PPCA notes).

### Identifiability: why a diagonal $D$ gives the principal components themselves

pPCA alone returns $W$ only up to a rotation $R$ (the PPCA notes, §4): same data distribution, different columns. The linear VAE removes this freedom because its posterior is forced to be diagonal.
At a stationary point the encoder solves $V^*=M^{-1}W^\top$ and $D^*=\sigma^2\tilde M^{-1}$ (appendix C, Eqs. 21 and 22). Whenever $M$ is *not* diagonal,
this $D^*$ cannot be the true posterior covariance, and the gap is positive. Rotating the columns changes $M$ but not the likelihood, so the ELBO prefers the rotation that makes $M$ diagonal: orthogonal
columns, pointing along principal directions, with lengths $\sqrt{\lambda_j-\sigma^2}$. The third widget of the interactive page draws exactly this tradeoff.

## 3. Theorem 1: the ELBO adds no new local maxima

This is the paper's central technical result. I rebuild the proof from the appendix.

**Step 1: the gap at a stationary point.** At a stationary point the encoder mean is the true posterior mean, and its covariance is $D^*=\sigma^2\tilde M^{-1}$ while the truth is $\sigma^2M^{-1}$.
For two Gaussians with the same mean,
$$\mathrm{KL}\bigl(\mathcal N(m,D)\,\Vert\,\mathcal N(m,\Sigma)\bigr)=\tfrac12\Bigl[\operatorname{tr}(\Sigma^{-1}D)-k+\log\det\Sigma-\log\det D\Bigr].$$
With $\Sigma=\sigma^2M^{-1}$ and $D=\sigma^2\tilde M^{-1}$, the trace is $\operatorname{tr}(M\tilde M^{-1})=\sum_i M_{ii}/\tilde M_{ii}=k$, because $M$ and $\tilde M$ have the same diagonal. What is left is
$$\mathrm{KL}=\tfrac12\bigl[\log\det\tilde M-\log\det M\bigr]\qquad\text{(the paper's Eq. 27)}.$$

**Step 2: this is never negative, and is zero only for a diagonal $M$.** $\det\tilde M$ is the product of the diagonal entries of $M$, and Hadamard's inequality says a positive-definite matrix has a determinant at most the product of its diagonal,
with equality exactly when it is diagonal. The paper does not name this inequality, but it is what makes the formula meaningful.

**Step 3: rotate.** Replace $W$ by $WR$. Then $M\mapsto R^\top MR$: $\det M$ is unchanged, $\log p(x)$ is unchanged, but the diagonal of $M$ changes. A small rotation can make the diagonal entries less equal, which
lowers $\prod_iM_{ii}$ whenever $M$ is not already diagonal (this is a Jacobi rotation, and it reduces the product of the diagonal because it keeps the trace but spreads the entries). So at a stationary point whose columns are not orthogonal, a small rotation lowers the gap with
the likelihood fixed, so the ELBO rises. A point you can improve cannot be a local maximum: it is a saddle. The paper's Lie-group argument with $\exp\{B-B^\top\}$ is the careful way of saying that such a small rotation exists.

**Step 4: the orthogonal case.** If the columns are orthogonal, the gap is zero at stationary points and ELBO and likelihood coincide. Hence **every extra stationary point is a saddle, and every local maximum of the ELBO sits where the gap is zero.**

## 4. The real cause: what noise level does to a zero column

This is the part to remember. The paper looks at stationary points of the *likelihood* of pPCA that are built from some eigenvectors of $S$, with other columns set to zero, and at their stability for a *fixed* $\sigma^2$.

**A one-column version I can check by hand.** Let one decoder column point along an eigenvector with data variance $\lambda$ and have length $s$. Along that direction the model's variance is $\sigma^2+s^2$. Up to terms that do not
depend on $s$, the negative log-likelihood per point (times 2) is
$$f(s)=\ln(\sigma^2+s^2)+\frac{\lambda}{\sigma^2+s^2}.$$
Write $t=\sigma^2+s^2\ge\sigma^2$. Then $\frac{d}{dt}(\ln t+\lambda/t)=\frac1t-\frac\lambda{t^2}$, which vanishes at $t=\lambda$, is negative before and positive after. So:

- if $\lambda>\sigma^2$, the best $t$ is $\lambda$, that is $s=\sqrt{\lambda-\sigma^2}$, and $s=0$ (that is, $t=\sigma^2$) is on the *wrong side* of the minimum of $f$: growing the column helps;
- if $\lambda\le\sigma^2$, $f$ is increasing for every allowed $t$, so the minimum is at the boundary $s=0$: **the zero column is the best**, and a small push away from zero makes things worse.

This is the paper's Eq. 13, where the sign of $\lambda_j/k_i-1$ decides stability, specialised to $k_i=\sigma^2$. It is also Figure 1 of the paper, and the first widget of the interactive page. Notice that nothing here involves an encoder, a KL term or a decoder's power: it is only the Gaussian model of the data saying that variance below the noise floor is noise.

**What this means for many columns.** For fixed $\sigma^2$, directions with $\lambda_j<\sigma^2$ are unreachable. A zero column along such a direction is a **stable local maximum** of the likelihood. If $\sigma^2$ is too large, many columns collapse, however many you provide.

**Why learning $\sigma^2$ helps.** When $\sigma^2$ is learned, its optimal value for a given set of $k$ columns is the average of the $n-k$ eigenvalues not covered, which is never above $\lambda_{k+1}$. A zero column pointing along $u_{k+1}$ then sees $\lambda_{k+1}\ge\sigma^2$ and is unstable: the noise can fall, and the column can grow. The paper writes this as "when $\sigma^2$ is learned, the local maxima become saddle points". The second widget plays this out.

**A side calculation that explains "how much a unit carries".** At the optimum for one unit, the encoder is $q(z\mid x)=\mathcal N\bigl(sx/\lambda,\;\sigma^2/\lambda\bigr)$ and the data variance along the direction is $\lambda$. Its average KL to the prior comes out as
$$\mathbb E_x\,\mathrm{KL}\bigl(q(z\mid x)\Vert p(z)\bigr)=\tfrac12\ln\frac{\lambda}{\sigma^2},$$
which is half the log signal-to-noise ratio and falls to zero exactly when $\lambda$ reaches $\sigma^2$. (It is the same as the mutual information of a Gaussian channel.) So here **collapse is the KL going to zero because the model has decided the unit has nothing to say**, which is why it looks like the KL's fault from the outside.

## 5. From the linear model to deep Gaussian VAEs

For a Gaussian decoder with output $D_\theta(z)$,
$$\mathcal L=-\mathrm{KL}(q\Vert p)-\frac1{2\sigma^2}\,\mathbb E\bigl\|D_\theta(z)-x\bigr\|^2-\frac n2\log(2\pi\sigma^2).$$
Multiplying by $2\sigma^2$ gives reconstruction error plus $2\sigma^2$ times the KL, which is a $\beta$-VAE with $\beta=2\sigma^2$ (up to the learning rate). So fixing a large $\sigma^2$ is
the same as a large KL weight, the opposite of the usual annealing remedy. The last term is what stops a *learned* $\sigma^2$ from simply shrinking to zero to cheat.

The paper's experiments, in words: deep MNIST and CelebA VAEs started at large $\sigma^2$ end with many collapsed units even when $\sigma^2$ is learned, and starting small collapses far fewer. Even after retuning $\sigma^2$ at the end,
the poorly started models stay worse. KL annealing relieves collapse early in training but it returns if $\sigma^2$ is not learned at the same time, and combining annealing with a learned $\sigma^2$ works better. In the linear case the stochastic
(sampled) ELBO converges more slowly and to a worse value than the exact analytic ELBO. I do not quote the tables because their numbers are the paper's runs, not something derived here; see the paper's Tables 1 to 3.

## Questions and doubts

- **A gap in the "no new maxima" sketch.** Steps 3 and 4 above show that a stationary point with non-orthogonal columns is never a local maximum, and that with orthogonal columns ELBO equals the likelihood.
  They do not by themselves show that an ELBO local maximum with orthogonal columns is also a likelihood local maximum. At such a point both objectives agree and have zero gradient, but the ELBO's second derivative is the likelihood's minus that of the gap, which is positive semidefinite, so the ELBO can only curve
  *further down*. A likelihood saddle whose unstable direction also breaks orthogonality could, in principle, become an ELBO maximum. For the zero-column saddles that matter here this does not happen: growing a zero column along an unused eigenvector keeps the columns perpendicular, the gap stays zero along that path, and the ELBO rises as the likelihood does. I have not checked other saddle types and did not find the general argument written down.
- **"Unique up to ordering" (appendix B).** The invariance $W\mapsto WA$ with a diagonal $A$ is the one the paper discusses, but $A$ with $\pm1$ entries leaves everything unchanged, so signs of the columns are also free. And if two eigenvalues are tied, $M$ is a multiple of the identity on that pair and any rotation inside the pair keeps $D$ exact, so the individual directions are not identified. The third widget shows this with $l_1=l_2$.
- **One global $D$.** The identifiability argument needs the posterior covariance not to depend on $x$. Real VAEs predict a variance per input, and the paper itself notes that Theorem 1 and the corollary fail once the encoder is nonlinear (even a deep linear encoder).
- **Bernoulli and categorical likelihoods.** Posterior collapse is best known for text models with softmax decoders and for binary images, but appendix C.3 gives up on those: the logit-normal integral has no closed form. Nothing here proves anything about them, and the deep experiments use a Gaussian likelihood on logit-transformed, dequantised MNIST, which the authors themselves call uncommon.
- **Correlation, not mechanism, in the deep case.** The deep results show that large initial $\sigma^2$ goes with more collapse. They do not show that the stable-zero-column mechanism of §4 is what is at work, and the paper notes the linear theory does not predict the gap that remains between initialisations when $\sigma^2$ is learned.
- **The title overreaches slightly.** The theory shows the *ELBO adds no new local maxima* and the collapse points exist in the likelihood. It does not show the KL term never contributes to collapse in dynamics, for example through the path an optimiser takes. The "lagging inference network" account of He et al. is a separate, dynamical story and is not refuted.

## Takeaways

- A linear VAE with a diagonal Gaussian posterior is probabilistic PCA; its global optimum has orthogonal columns of length $\sqrt{\lambda_j-\sigma^2}$ (up to signs and ties).
- The ELBO adds no new local maxima in the linear case. Any bad stationary point was already bad for the likelihood.
- With $\sigma^2$ fixed, a latent unit along a direction with variance below $\sigma^2$ stays at zero: a stable collapse, caused by the Gaussian likelihood and not by the KL term.
- Learning $\sigma^2$ turns these points into saddles. The information a live unit carries is $\tfrac12\ln(\lambda/\sigma^2)$, which vanishes exactly at collapse.
- For a Gaussian decoder, a large $\sigma^2$ acts like a large KL weight $\beta=2\sigma^2$, so the initial noise level is a first-order choice, not a detail.
- The proofs are for linear Gaussian models with one shared posterior covariance; the deep results are suggestive evidence only.
