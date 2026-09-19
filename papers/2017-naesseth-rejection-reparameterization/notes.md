---
title: "Reparameterization Gradients through Acceptance-Rejection Sampling Algorithms"
category: "Gradient estimation"
subcategory: "Theory"
short_title: "Naesseth 2017 — RSVI"
authors: "Christian A. Naesseth, Francisco J. R. Ruiz, Scott W. Linderman, David M. Blei (Linköping, Columbia, Cambridge)"
venue: "AISTATS"
year: 2017
url: "https://arxiv.org/abs/1610.05683"
pdf_url: "https://arxiv.org/pdf/1610.05683"
tags: [variational-inference, reparameterization-gradient, rejection-sampling, gradient-estimation, gamma, dirichlet, variance-reduction]
status: read
---

## Links

- **[arXiv:1610.05683](https://arxiv.org/abs/1610.05683)** — preprint. `make fetch` pulls the
  PDF from here. Published at AISTATS 2017, JMLR W&CP volume 54.
- **[Authors' code](https://github.com/blei-lab/ars-reparameterization)** — experiments.
- **[Runnable code](https://github.com/msrepo/theory_inclined_papers_with_annotations/blob/main/papers/2017-naesseth-rejection-reparameterization/code/rsvi_gamma.py)** —
  Equation 4 checked against a real Marsaglia–Tsang sampler, the unbiasedness of
  $g_{\text{rep}}+g_{\text{cor}}$ against the bias of dropping $g_{\text{cor}}$, and the
  variance comparison. `make verify` runs it.

## In one paragraph

The reparameterization trick needs $z = h(\varepsilon,\theta)$ with $\varepsilon$ drawn from a
$\theta$-free distribution and $h$ differentiable. Gamma and Dirichlet variables are simulated
by **rejection sampling**, whose accept/reject branch is not differentiable, so they were stuck
with the high-variance score function estimator. The paper's move is to refuse to differentiate
the sampler at all: instead ask what distribution the *accepted* $\varepsilon$ has once the
accept/reject coin is integrated out analytically. That marginal is smooth and has a
one-line form, and differentiating through it splits the gradient into the ordinary
reparameterization term plus a correction whose size is governed by **how well the proposal
matches the target** — which is exactly what rejection samplers have been engineered to
optimise for sixty-five years. The method inherits all of that work for free, and gradient
variance drops three to five orders of magnitude against the previous best.

## The spine of the argument

1. Return the accepted $\varepsilon$, not the accepted $z$. This is the whole trick and it
   costs nothing.
2. Write down the joint density of a proposed $(\varepsilon,u)$ conditioned on acceptance, and
   integrate out $u$. The indicator integrates to an interval length, the envelope constant
   $M_\theta$ cancels, and what is left is smooth (Eq. 4).
3. Differentiate that expectation. $\theta$ sits in both the integrand and the density, so the
   product rule gives two terms (Eq. 5).
4. Observe that the second term depends only on the *ratio* $q/r$, so a tight envelope kills it.
5. Use change of variables to remove the proposal density from the expression entirely, leaving
   only things you already have (Eq. 8).
6. Instantiate on the gamma; get the Dirichlet free; inflate the shape parameter to make the
   sampler better than it needs to be.

## Setup and notation

| Symbol | Meaning |
|---|---|
| $p(x,z)$ | the model; $x$ data, $z$ latent |
| $q(z;\theta)$ | variational family, **the target to sample from** |
| $f(z) = \log p(x,z)$ | the log joint |
| $\mathcal{L}(\theta)$ | ELBO, $\mathbb{E}_{q}[f(z)] + \mathbb{H}[q]$ |
| $r(z;\theta)$ | **proposal** distribution of the rejection sampler |
| $M_\theta$ | envelope constant, $q \le M_\theta r$; acceptance probability is $1/M_\theta$ |
| $s(\varepsilon)$ | base distribution, free of $\theta$ (here $\mathcal{N}(0,1)$) |
| $h(\varepsilon,\theta)$ | differentiable map with $\varepsilon\sim s \Rightarrow h(\varepsilon,\theta)\sim r$ |
| $\pi(\varepsilon;\theta)$ | **density of the accepted $\varepsilon$** — the object the paper is about |
| $g_{\text{rep}}, g_{\text{cor}}$ | the two halves of Eq. 5 |
| $B$ | shape-augmentation steps |

The one structural assumption: **the proposal is reparameterizable.** Not needed for rejection
sampling to be valid, but true of essentially every practical sampler, because they are built
out of transformed normals and uniforms.

## Why the naive thing fails, precisely

Rejection sampling draws $z\sim r$, draws $u\sim\mathcal{U}[0,1]$, accepts if
$u < q(z;\theta)/(M_\theta r(z;\theta))$, and otherwise starts over. Written in
$\varepsilon$-space (Algorithm 1) it consumes a *random number* $I$ of proposals and returns
$\varepsilon_I$.

The usual diagnosis is "the indicator is discontinuous, and $I$ depends on $\theta$". True, but
the sharper statement is about what autodiff actually computes. Run it on the accepted path and
you get $\nabla_\theta h(\varepsilon_I,\theta)$ with $\varepsilon_I$ treated as a constant. But
$\varepsilon_I$ **is not a draw from $s$** — conditioning on acceptance has reshaped its
distribution, and the reshaping depends on $\theta$.

So naive autodiff does not crash or produce nonsense. It computes $g_{\text{rep}}$ and
**silently drops $g_{\text{cor}}$**, returning a quietly biased gradient. The paper's real
contribution is naming exactly what is missing and showing when it is negligible. The code
measures the bias: at $\alpha=1$ the truth is $3.000$ and $g_{\text{rep}}$ alone gives $2.911$,
which does not improve with more samples.

## Equation 4: the density of the accepted $\varepsilon$

Write the joint density of a proposed pair $(\varepsilon,u)$ *given that it was accepted*.
$\varepsilon$ comes from $s$, $u$ is uniform so contributes density 1, acceptance is the event
below, and the overall acceptance probability is $1/M_\theta$, so conditioning multiplies by
$M_\theta$:

$$
\pi(\varepsilon,u;\theta) = M_\theta\, s(\varepsilon)\,
\mathbb{1}\!\left[0 < u < \frac{q(h(\varepsilon,\theta);\theta)}{M_\theta\, r(h(\varepsilon,\theta);\theta)}\right].
$$

Marginalising out $u$ integrates an indicator, which just returns the interval's length:

$$
\boxed{\;\pi(\varepsilon;\theta)
= s(\varepsilon)\,\frac{q(h(\varepsilon,\theta);\theta)}{r(h(\varepsilon,\theta);\theta)}\;}
\tag{4}
$$

**$M_\theta$ cancels** — the one genuinely awkward $\theta$-dependent object disappears before
anything is differentiated.

It is a density: substituting $z = h(\varepsilon,\theta)$ with $s(\varepsilon)d\varepsilon =
r(z;\theta)dz$ gives $\int r(z;\theta)\frac{q}{r}dz = \int q = 1$.

### A more revealing form

The same change of variables says $r(h(\varepsilon,\theta);\theta)\lvert dh/d\varepsilon\rvert =
s(\varepsilon)$, so $s(\varepsilon)/r(h) = \lvert dh/d\varepsilon\rvert$ and Eq. 4 becomes

$$
\pi(\varepsilon;\theta) = q\big(h(\varepsilon,\theta);\theta\big)\,
\left\lvert\frac{dh}{d\varepsilon}(\varepsilon,\theta)\right\rvert .
$$

**$\pi$ is just the pullback of the target through $h$** — the density $\varepsilon$ must have
in order for $h(\varepsilon,\theta)$ to be exactly $q$-distributed. Stated this way, Eq. 4 stops
being a calculation and becomes a tautology: *of course* that is the law of the accepted
$\varepsilon$, since rejection sampling produces exact draws from $q$. Proposition 1 is then
immediate, and Eq. 8 below is obvious rather than a derivation. The paper does not put it this
way, and it is the form the code checks.

The reading that matters for the rest: $\pi$ is the base distribution $s$, **reweighted by how
far the target exceeds the proposal**. If $r = q$ the ratio is constant, $\pi = s$, and there is
no $\theta$ in the density at all — ordinary reparameterization. Everything below is a
quantification of the gap between $r$ and $q$.

And crucially, $\pi$ is *smooth*. All the discontinuity went into that one $\mathrm{d}u$.

## Equation 5: where the two terms come from

Rewrite the ELBO over $\varepsilon$ (Eq. 3) and differentiate. $\theta$ appears **twice** — in
the integrand through $h$, and in the density $\pi$ — so the product rule gives

$$
\nabla_\theta\!\int f(h(\varepsilon,\theta))\,\pi(\varepsilon;\theta)\,\mathrm{d}\varepsilon
= \underbrace{\int \nabla_\theta[f(h)]\,\pi\,\mathrm{d}\varepsilon}_{\theta\text{ in the function}}
+ \underbrace{\int f(h)\,\nabla_\theta\pi\,\mathrm{d}\varepsilon}_{\theta\text{ in the density}} .
$$

The first is already an expectation. The second becomes one via the log-derivative trick
$\nabla_\theta\pi = \pi\nabla_\theta\log\pi$. Then take logs in Eq. 4:
$\log\pi = \log s(\varepsilon) + \log q(h) - \log r(h)$, and **$\log s(\varepsilon)$ carries no
$\theta$, so it dies**, leaving only the log-ratio:

$$
\nabla_\theta\mathbb{E}_{q(z;\theta)}[f(z)] = g_{\text{rep}} + g_{\text{cor}},
\tag{5}
$$

$$
g_{\text{rep}} = \mathbb{E}_{\pi}\big[\nabla_z f(z)\big|_{z=h(\varepsilon,\theta)}\nabla_\theta h(\varepsilon,\theta)\big],
\qquad
g_{\text{cor}} = \mathbb{E}_{\pi}\!\left[f(h(\varepsilon,\theta))\,
\nabla_\theta\log\frac{q(h(\varepsilon,\theta);\theta)}{r(h(\varepsilon,\theta);\theta)}\right].
$$

$g_{\text{rep}}$ is the standard reparameterization gradient, computed as if the sampler were
exact. $g_{\text{cor}}$ is score-function-flavoured, and it is the price of the proposal not
being the target. The full estimator is these two plus $\nabla_\theta\mathbb{H}[q]$ (Eq. 6–7),
unbiased because Eq. 5 is exact and Algorithm 1 returns an exact draw from $\pi$.

## Why $g_{\text{cor}}$ is small — the actual point

$g_{\text{cor}}$ contains **only the ratio** $q/r$, never $q$ or $r$ alone. A rejection sampler
is built so the envelope is tight, $q \approx M_\theta r$ with $M_\theta$ near 1 — which is
precisely what "high acceptance probability" means, since acceptance is $1/M_\theta$. Then
$q/r \approx M_\theta$ is nearly constant in $\varepsilon$, its log-gradient is nearly zero,
and $g_{\text{cor}}\approx 0$.

**So the high-variance term is suppressed in exact proportion to the efficiency of the
sampler** — a quantity people have optimised for sixty-five years for entirely unrelated
reasons. That is the "removing the lid" idea, and it is the good idea in the paper.

The contrast with G-REP (Ruiz et al. 2016), the prior state of the art, is the conceptual core:

| | strategy |
|---|---|
| **G-REP** | search for a transformation of $z$ making $\varepsilon$'s law *weakly depend* on $\theta$ (standardisation, $\varepsilon = (z-\mu(\theta))/\sigma(\theta)$) |
| **RSVI** | the opposite — transform a *simple* $\varepsilon$ so that $h(\varepsilon,\theta)$ is *almost exactly* $q$, reusing transformations the sampling literature already derived |

G-REP needs a new transformation per distribution. RSVI reads one off the shelf.

## Equation 8: deleting the proposal density

Evaluating $r$ is awkward, since a sampler is usually code rather than a formula. If $h$ is
invertible in $\varepsilon$ it is unnecessary. From
$r(h(\varepsilon,\theta);\theta)\lvert dh/d\varepsilon\rvert = s(\varepsilon)$, take logs and
differentiate; $s(\varepsilon)$ has no $\theta$, so
$\nabla_\theta\log r(h) = -\nabla_\theta\log\lvert dh/d\varepsilon\rvert$, and

$$
\nabla_\theta\log\frac{q(h(\varepsilon,\theta);\theta)}{r(h(\varepsilon,\theta);\theta)}
= \nabla_\theta\log q(h(\varepsilon,\theta);\theta)
+ \nabla_\theta\log\left\lvert\frac{dh}{d\varepsilon}(\varepsilon,\theta)\right\rvert .
\tag{8}
$$

In practice you need only **the target's log-density** and **the Jacobian of $h$**, both of
which you have. Note that the right-hand side is exactly $\nabla_\theta\log\pi$ in the pullback
form above, which is what Eq. 5 said the weight should be — the code uses this as its
definition and the unbiasedness check confirms the two agree.

A trap worth flagging: $\nabla_\theta$ here is a **total** derivative with $\varepsilon$ held
fixed, so $\nabla_\theta\log q(h(\varepsilon,\theta);\theta)$ picks up both the explicit
$\theta$ in $q(\cdot\,;\theta)$ and the $\theta$ routed through $h$.

### The importance-sampling variant, and why it is not used

Substituting $\pi = s\cdot(q/r)$ turns everything into expectations under $s$, avoiding the
sampler entirely — but the $q/r$ factors are then **importance weights**, whose variance
explodes with dimension. The paper keeps it as an intermediate step and notes it is viable only
in low dimensions. Running Algorithm 1 is cheaper anyway.

## The gamma, and shape augmentation

Marsaglia–Tsang (2000) for $\mathrm{Gamma}(\alpha,1)$, $\alpha\ge 1$:

$$
z = h_{\mathrm{Gamma}}(\varepsilon,\alpha) := \Big(\alpha-\tfrac13\Big)
\Big(1+\tfrac{\varepsilon}{\sqrt{9\alpha-3}}\Big)^3,
\qquad \varepsilon\sim\mathcal{N}(0,1).
\tag{10}
$$

A cubed, shifted, rescaled normal — perfectly differentiable. Rate $\beta\neq1$ divides through;
$\alpha<1$ uses $z = u^{1/\alpha}\bar z$ with $\bar z\sim\mathrm{Gamma}(\alpha+1,\beta)$.
Dirichlet comes free: $\bar z_k\sim\mathrm{Gamma}(\alpha_k,1)$ independently gives
$(\sum_\ell \bar z_\ell)^{-1}\bar z_{1:K}\sim\mathrm{Dirichlet}(\alpha_{1:K})$.

**Shape augmentation.** Since everything improves with $\alpha$, inflate it:

$$
z = \bar z \prod_{i=1}^B u_i^{1/(\alpha+i-1)},\qquad
\bar z\sim\mathrm{Gamma}(\alpha+B,1),\quad u_i\overset{\text{iid}}\sim\mathcal{U}[0,1].
$$

This is Stuart's (1962) identity applied $B$ times: $X\sim\mathrm{Gamma}(a+1,1)$ and
$U\sim\mathcal{U}(0,1)$ give $XU^{1/a}\sim\mathrm{Gamma}(a,1)$, peeling the shape down one unit
at a time. The point is that you run the *rejection sampler* at the inflated shape, where
acceptance is near 1, and shrink back with a map that is **smooth in $\alpha$ and involves no
rejection** — so its $\theta$-dependence lands in $g_{\text{rep}}$ instead of $g_{\text{cor}}$.
Weight is moved out of the badly-behaved term into the well-behaved one.

### Reproduced

`code/rsvi_gamma.py` implements the real Marsaglia–Tsang accept/reject loop and checks three
things. **Equation 4** against the empirical law of the accepted $\varepsilon$ (400k draws,
comparing densities on a common window so the histogram's truncation does not masquerade as
error):

| $\alpha$ | 1.0 | 2.0 | 5.0 | 10.0 |
|---|---|---|---|---|
| acceptance | 0.952 | 0.982 | 0.994 | 0.997 |
| $\max\lvert\text{hist}-\pi\rvert$ | 0.0067 | 0.0062 | 0.0072 | 0.0056 |
| $\mathrm{KL}$ | 4.8e−5 | 6.8e−5 | 8.2e−5 | 6.6e−5 |

**Unbiasedness**, on $\nabla_\alpha\mathbb{E}[z^2] = 2\alpha+1$:

| $\alpha$ | truth | $g_{\text{rep}}$ only | $g_{\text{rep}}+g_{\text{cor}}$ | score function |
|---|---|---|---|---|
| 1.0 | 3.000 | 2.911 | **2.998** | 2.991 |
| 5.0 | 11.000 | 10.989 | **11.005** | 11.046 |

**Variance, and the shrinking correction.** As the sampler gets better the correction's share of
the gradient collapses and the advantage over the score function estimator widens:

| $\alpha$ | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 |
|---|---|---|---|---|---|
| acceptance | 0.952 | 0.981 | 0.994 | 0.997 | 0.999 |
| $\lvert\mathbb{E}g_{\text{cor}}\rvert/\lvert\text{grad}\rvert$ | 0.0263 | 0.0076 | 0.0014 | 0.0004 | 0.0001 |
| Var(score)/Var(RSVI) | 3.9× | 6.5× | 13.7× | 29.8× | 77.2× |

That middle row is the paper's central claim, measured: **the correction term's share tracks the
rejection rate.** Shape augmentation at $\alpha=1$ does the same thing deliberately, driving the
share from $0.0294$ at $B=0$ to below $10^{-4}$ at $B=10$, with variance falling from $26.7$ to
$21.7$ — a real but much more modest variance gain than the correction-shrinkage suggests.

## Results in the paper

On a sparse gamma DEF over NIPS documents, gradient variance (median across dimensions):

| | RSVI $B=1$ | RSVI $B=4$ | G-REP |
|---|---|---|---|
| at initialisation | 9.0e7 | **2.9e7** | 1.6e12 |
| at iteration 2600 | 1.8e4 | **4.5e3** | 1.5e7 |

Roughly five orders of magnitude at init, three to four later. On Olivetti faces RSVI reaches a
high ELBO far faster in wall-clock than ADVI, BBVI or G-REP, and is about **2× faster per
iteration** than G-REP — because rejection-sampling transformations were optimised for speed by
the literature they came from.

## Questions and doubts

- **There is no theorem bounding $g_{\text{cor}}$.** The whole argument for low variance is
  "good sampler $\Rightarrow q/r\approx M_\theta \Rightarrow$ small correction", supported by
  plots. The missing result is something like
  $\operatorname{Var}(g_{\text{cor}}) \lesssim \psi(1-1/M_\theta)$ — a variance bound in terms
  of acceptance probability. The table above suggests the relationship is clean enough to be
  provable, and it would turn the paper's intuition into a guarantee.
- **$g_{\text{cor}}$ still carries $f$.** It is $f(h(\varepsilon,\theta))$ times a small
  log-ratio gradient. It is small because the *second* factor is small, not because $f$ is
  controlled, and $f$ is the full log joint. A badly scaled model could still let it dominate.
  The paper's own Table 1 shows maxima of 1e14–1e17, and the choice to report **medians** across
  dimensions rather than means is the tell: the gradients are heavy-tailed even after the fix.
- **The "mild assumptions" are never stated.** That $\pi(\varepsilon;\theta)$ is continuous
  "under mild assumptions" is load-bearing — it is what makes the construction legitimate — and
  the conditions never appear.
- **Equation 8 needs $h$ invertible in $\varepsilon$**, which holds for their examples but is
  not argued generally. Without it you are back to evaluating $r$.
- **The entropy gradient is assumed analytic** (footnote 1), which narrows scope. More
  interesting: they remark in passing that a *Monte Carlo* entropy gradient might have **lower**
  variance than the analytic one, and then decline to investigate. That is a surprising claim to
  leave dangling.
- **Not automatic.** You need a reparameterizable rejection sampler for your family, derived by
  hand. Gamma and Dirichlet are done here, more in the supplement, but this is not a drop-in the
  way the score function estimator is. Contrast with ADVI, which is fully automatic and
  correspondingly worse.
- **No principled choice of $B$.** Augmentation clearly helps and clearly costs $B$ extra
  uniforms per sample; guidance stops at "more was better in these experiments". My own numbers
  suggest sharply diminishing returns in variance past $B=1$ even as the correction keeps
  shrinking, so the two are not as tightly coupled as the framing implies.

## Takeaways

- The transferable idea, well beyond variational inference: **when a stochastic program has a
  discontinuous branch, do not differentiate the program. Find the marginal law of its output
  with the branch integrated out, and differentiate that.** Rejection sampling is one instance;
  the conclusion lists adaptive rejection, importance sampling, SMC and MCMC as others.
- **Naive autodiff through a rejection sampler fails silently**, returning $g_{\text{rep}}$ and
  dropping $g_{\text{cor}}$. It does not error, and the bias does not shrink with more samples.
  Worth remembering for any sampler wrapped in an autodiff framework.
- The elegant part is that the residual error is controlled by *proposal quality*, which is the
  thing an enormous existing literature already optimises. Reusing someone else's objective as
  your variance bound is a good trick to have seen.
- $\pi(\varepsilon;\theta) = q(h(\varepsilon,\theta);\theta)\lvert dh/d\varepsilon\rvert$ is the
  form to remember. It makes Equation 4 obvious rather than clever, and Equation 8 falls out of
  it in one line.
