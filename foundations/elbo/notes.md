---
title: "The ELBO: a computable lower bound on the log-likelihood"
authors: "Background notes"
venue: "Foundations"
tags: [elbo, variational-inference, vae, jensen, kl-divergence, latent-variables, background]
status: living
category: "Foundations"
subcategory: "Probability"
short_title: "ELBO"
---

## Links

- **[Interactive version](figures/interactive.html)**: drag $q$ in the two-coin example and watch
  the ELBO, the ceiling $\log p(x)$ and the gap between them; change the prior and the biased
  coin's heads probability and watch the posterior (and the peak of the curve) move.
- **[EM and Gaussian mixtures](../expectation-maximization/index.html)**: the same
  lower-bound-that-touches argument, with the E-step choosing $q$ exactly; §10 there steps to the ELBO.
- **[Inequalities and concentration](../inequalities-and-concentration/index.html)**: Jensen's
  inequality, which §3 below uses.
- `code/elbo_two_coins.py` and `code/elbo_gaussian.py` print the numbers quoted below.

## In one paragraph

We want to fit a model with a hidden variable $z$ by maximising $\log p(x)$, but
$p(x)=\int p(x,z)\,dz$ is a sum or integral we cannot evaluate when $z$ is large. Pick any easy
distribution $q(z)$. Jensen's inequality turns "log of an average" into "average of a log", which
gives a number we *can* compute, the **ELBO**, that never exceeds $\log p(x)$. The shortfall is
exactly $\mathrm{KL}(q\,\|\,p(z\mid x))$, so raising the ELBO both improves the model and pushes
$q$ toward the true posterior, without ever computing the intractable normaliser.

## 1. The running example: two coins

A bag holds a fair coin (heads with probability $0.5$) and a biased coin (heads with probability
$0.9$). You pick one at random without looking and flip it. You see **heads**.

- hidden choice $z$ = which coin, prior $\tfrac12,\tfrac12$;
- observation $x$ = heads;
- routes to heads: fair $0.5\times0.5=0.25$, biased $0.5\times0.9=0.45$.

Every quantity here can be computed exactly, so each idea can be checked. In a VAE, $z$ is
continuous and high-dimensional and none of them can.

## 2. Questions, in plain words and in symbols

| # | Plain-language question | Mathematical question |
|---|---|---|
| 1 | How likely is what I saw, if I don't know which coin I picked? Why is that hard when the hidden thing is large? | What are the marginal $p(x)=\int p(x,z)\,dz$ and the log-likelihood $\sum_i\log p(x_i)$? Why is the log of a sum (integral) intractable for continuous, high-dimensional $z$? |
| 2 | Is squashing first and then averaging the same as averaging first and then squashing? If not, which is bigger? | State Jensen's inequality: for concave $f$, $f(\mathbb E X)\ge\mathbb E f(X)$. When is equality attained? |
| 3 | Can I get a number I *can* compute that is guaranteed never to exceed the true answer? | Derive $\mathrm{ELBO}(q)=\mathbb E_q[\log p(x,z)-\log q(z)]$ and show $\log p(x)\ge\mathrm{ELBO}(q)$ for any $q$. |
| 4 | How far below the true answer is that number, and how do I close the gap? Does it tell me which coin it probably was? | Show $\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}(q\,\|\,p(z\mid x))$. Why does maximising the ELBO over $q$ minimise the KL, and when is the bound tight? |
| 5 | What is the number rewarding and what is it penalising? | Show $\mathrm{ELBO}=\mathbb E_q[\log p(x\mid z)]-\mathrm{KL}(q\,\|\,p(z))$ and interpret the two terms. |

## Notation

| Symbol | Meaning | Coin example |
|---|---|---|
| $x$ | observed data | heads |
| $z$ | latent variable | which coin |
| $p(z)$ | prior | $\tfrac12,\tfrac12$ |
| $p(x\mid z)$ | likelihood | $0.5,\ 0.9$ |
| $p(x,z)$ | joint $=p(x\mid z)p(z)$ | $0.25,\ 0.45$ |
| $p(x)$ | marginal ("evidence") $=\int p(x,z)\,dz$ | $0.70$ |
| $p(z\mid x)$ | posterior | $0.36,\ 0.64$ |
| $q(z)$ | our tractable guess for the posterior | any distribution |
| $\mathbb E_q[\cdot]$ | average with $z$ drawn from $q$ | |
| $\mathrm{KL}(q\|p)$ | $\mathbb E_q[\log q-\log p]\ge0$, zero only if $q=p$ | |
| $\mathrm{ELBO}(q)$ | $\mathbb E_q[\log p(x,z)-\log q(z)]$ | |

## 3. Question 1: the marginal, and why it is hard

*In plain words:* add up every hidden route that ends in what you saw: $0.25+0.45=0.70$. The
hidden choice has been summed out ("marginalised out"). Read as a function of the model's
parameters, with the data fixed, this is the **likelihood**. For a dataset of independent
observations, the log-likelihood is a sum, which is easier to handle than a product of tiny numbers.

*In symbols:*
$$p(x)=\sum_z p(x,z)\ \text{ or }\ \int p(x,z)\,dz,\qquad
\ell(\theta)=\sum_i\log p(x_i;\theta)=\sum_i\log\sum_z p(x_i,z;\theta).$$

Each term is a **log of a sum**. With two coins that is two terms, so it is harmless. With a
continuous, high-dimensional $z$ it is not, for two reasons:

1. A grid with 10 points per dimension needs $10^d$ points in $d$ dimensions.
2. For one image $x$, $p(x\mid z)$ is large only on a thin sliver of $z$-space and nearly zero
   elsewhere. Guessing $z$ from the prior almost never lands in the sliver, so a Monte Carlo
   estimate is dominated by luck.

## 4. Question 2: Jensen's inequality

Take $f=\log$, which bends downwards. Average the two inputs $1$ and $4$ with equal weight.

- average first, then log: $\log 2.5\approx0.92$;
- log first, then average: $\tfrac12(\log1+\log4)\approx0.69$.

The curve lies above its chord, so the first is larger. In general, for concave $f$,
$$f(\mathbb E[X])\ \ge\ \mathbb E[f(X)],$$
with the inequality reversed for convex $f$. Equality holds only when $f$ is linear or $X$ is not
random. The more the curve bends, or the more $X$ spreads, the bigger the gap.

## 5. Question 3: the bound

Multiply and divide by $q$ (assuming $q(z)>0$ wherever $p(x,z)>0$), turning the integral into an
average:
$$p(x)=\int q(z)\,\frac{p(x,z)}{q(z)}\,dz=\mathbb E_q\!\left[\frac{p(x,z)}{q(z)}\right].$$
Take the log, then apply Jensen with $f=\log$:
$$\log p(x)=\log\mathbb E_q\!\left[\frac{p(x,z)}{q(z)}\right]\ \ge\ \mathbb E_q\!\left[\log\frac{p(x,z)}{q(z)}\right]=\mathbb E_q[\log p(x,z)-\log q(z)]=\mathrm{ELBO}(q).$$

The log now sits inside the average, so we can estimate it by sampling $z$ from $q$ and
differentiate it. In the coin example with $q=(\tfrac12,\tfrac12)$ the ratios $p(x,z)/q(z)$ are
$0.5$ and $0.9$: two "points" like the $1$ and $4$ above, whose average is the marginal $0.70$.

The bound is tight exactly when the ratio $p(x,z)/q(z)$ is the same for every $z$, that is, when
$q(z)\propto p(x,z)$, which means $q(z)=p(z\mid x)$.

## 6. Question 4: the exact gap, and the posterior

Bayes' rule gives the **posterior**
$$p(z\mid x)=\frac{p(x,z)}{p(x)}.$$
In the coin example, each route's share of the total: fair $0.25/0.70\approx0.36$, biased
$0.45/0.70\approx0.64$. Seeing heads is evidence for the biased coin. The numerator is easy;
the denominator is the marginal, the intractable integral. So the exact posterior is as
out of reach as $p(x)$ itself.

Using the definition of KL and Bayes' rule,
$$\mathrm{KL}(q\|p(z\mid x))=\mathbb E_q[\log q(z)-\log p(x,z)+\log p(x)]=-\mathrm{ELBO}(q)+\log p(x),$$
because $\log p(x)$ does not depend on $z$ and comes out of the expectation. Hence
$$\boxed{\ \log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}\big(q\,\|\,p(z\mid x)\big)\ }$$

The left side does not depend on $q$, so the two terms on the right trade off one for one:
raising the ELBO lowers the KL. Since $\mathrm{KL}\ge0$ with equality only at $q=p(z\mid x)$, the
best $q$ is the posterior and the ELBO then equals $\log p(x)$. The ELBO contains only $p(x,z)$
and $q$, never $p(x)$, so we minimise the unreachable KL by maximising the reachable ELBO.
The Jensen gap of §5 is this same KL.

The [interactive page](figures/interactive.html) shows this directly: the ELBO curve peaks at the
posterior and touches the ceiling $\log p(x)$; the vertical gap at any other $q$ is the KL.

## 7. Question 5: reconstruction and prior matching

With $p(x,z)=p(x\mid z)\,p(z)$:
$$\mathrm{ELBO}=\underbrace{\mathbb E_q[\log p(x\mid z)]}_{\text{reconstruction}}\;-\;\underbrace{\mathrm{KL}\big(q(z)\,\|\,p(z)\big)}_{\text{prior matching}}.$$

- **Reconstruction** rewards $z$'s that make $x$ likely. Alone it would collapse $q$ onto the single best-fitting $z$.
- **Prior matching** penalises $q$ for straying from the prior. Alone it would set $q$ equal to the prior and ignore $x$.

The ELBO is the compromise: fit the data, but only as far as the evidence justifies. In the coin
example, leaning toward the biased coin explains heads better but moves away from the 50-50
belief; the best compromise is the posterior. This is the form used in VAEs: the decoder wants
good reconstructions, and the KL keeps the latent space organised like the prior.

| Form | Terms | Good for |
|---|---|---|
| joint | $\mathbb E_q[\log p(x,z)]+H(q)$ | seeing the Jensen derivation |
| gap | $\log p(x)-\mathrm{KL}(q\|p(z\mid x))$ | seeing why maximising gives the posterior |
| reconstruction | $\mathbb E_q[\log p(x\mid z)]-\mathrm{KL}(q\|p(z))$ | training VAEs: every piece is computable |

## 8. A continuous latent

The same story with a Gaussian model, where everything has a closed form: $z\sim N(0,1)$,
$x\mid z\sim N(z,0.25)$, observed $x=2$. Then $x\sim N(0,1.25)$ and the posterior is
$N(1.6,\,0.2)$. For $q=N(m,s^2)$ the ELBO is the closed-form reconstruction minus KL, and
$\mathrm{ELBO}+\mathrm{KL}(q\|\text{posterior})=\log p(x)$ for every $q$ (`code/elbo_gaussian.py`).
A $q$ with the right mean but too wide or too narrow a variance falls short; $q=$ posterior
reaches the ceiling.

The thin-sliver problem also shows up. Estimating $\log p(x)$ from 200 samples drawn blindly from
the prior is fine in one latent dimension, drifts at five, and at twenty is off by a large
amount; sampling from the posterior is exact at every dimension, because the ratio
$p(x,z)/q(z)$ is then constant (the Jensen equality case). This is an idealised
case: here the posterior is known exactly. In a real VAE, $q$ is only an approximation, so the
ELBO sits below $\log p(x)$ by the KL gap.

## Questions and doubts

- **The bound is only as tight as the family of $q$ allows.** If $q$ must be a diagonal Gaussian
  and the true posterior is skewed or multimodal, the best $q$ still leaves a positive KL, and
  the ELBO is strictly below $\log p(x)$ even at the optimum. Nothing in the derivation says how
  large that leftover gap is.
- **Which KL?** The ELBO minimises $\mathrm{KL}(q\|p(z\mid x))$, the "reverse" direction. That
  tends to make $q$ cover one mode and under-estimate spread, rather than cover all of the
  posterior. I have not checked this on an example here.
- **A higher ELBO is not obviously a better model.** Raising the ELBO can come from a better
  decoder or from a better $q$; the two are mixed in one number, so a rising ELBO does not say
  which improved.
- **Single-sample estimates.** The training objective estimates $\mathbb E_q[\cdot]$ with one or
  few samples. Whether that noise matters, and how tighter bounds with several samples
  (importance-weighted) behave, is not covered here.
- **The two-coin and Gaussian cases are friendly.** Both have a posterior we can write down,
  so every claim could be checked. They do not show how loose the bound is in a deep latent model.

## Cheat sheet

- Bayes: $p(z\mid x)=p(x,z)/p(x)$.
- Jensen (concave $f$): $f(\mathbb E X)\ge\mathbb E f(X)$.
- Bound: $\log p(x)\ge\mathrm{ELBO}(q)=\mathbb E_q[\log p(x,z)-\log q(z)]$.
- Exact gap: $\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}(q\|p(z\mid x))$.
- VAE form: $\mathrm{ELBO}=\mathbb E_q[\log p(x\mid z)]-\mathrm{KL}(q\|p(z))$.
- Tight iff $q(z)=p(z\mid x)$, i.e. $p(x,z)/q(z)$ is constant in $z$.
