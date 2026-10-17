---
title: "Probabilistic PCA: a line, blurred"
authors: "Background notes (after Tipping and Bishop, 1999)"
venue: "Foundations"
tags: [pca, probabilistic-pca, latent-variable, gaussian, em, factor-analysis, background]
status: living
category: "Foundations"
subcategory: "Probability"
short_title: "Probabilistic PCA"
pdf_url: "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-18.pdf"
summary: "a short hidden Gaussian vector carried along a few directions plus equal noise; the data as a Gaussian with covariance $WW^\top+\sigma^2I$ and the rotation freedom of $W$; the maximum-likelihood solution (PCA subspace, directions shortened to $\sqrt{\lambda_j-\sigma^2}$, noise equal to the average of the discarded eigenvalues); the Gaussian posterior over a point's hidden coordinate as a shrunken PCA projection that tends to PCA as the noise goes to zero; EM without forming the covariance, missing data, factor analysis and the linear VAE; three explorers (build and fit a cloud, drag a point and see its posterior, cut a spectrum to see the noise floor)"
---

## Links

- **[Interactive version](figures/interactive.html)**: grow a data cloud from a hidden coordinate and fit it back; drag a point and watch
  its hidden coordinate become a bell curve instead of a single number; and cut an eigenvalue spectrum to see where the noise level comes from.
- **[Expectation–Maximization](../expectation-maximization/index.html)**: the algorithm used in §8, and the
  chicken-and-egg idea behind it.
- **[Eckart–Young and low-rank SVD](../eckart-young-lowrank-svd/index.html)**: the non-probabilistic reason the top
  eigenvectors give the best low-rank summary.
- **[The ELBO](../elbo/index.html)**: where the latent-variable model here reappears in a VAE (§10).

## In one paragraph

Ordinary PCA finds a flat, low-dimensional sheet that sits close to the data. It does not say what kind of data
that sheet would produce. **Probabilistic PCA (PPCA)** tells a story: a short hidden vector $z$ is drawn from a bell curve,
it is carried into the full data space along a few fixed directions, and then every coordinate gets a little
independent noise of the same size. The data is the blurred sheet. Fitting this story by maximum likelihood gives back the
**same subspace as PCA**, and it also gives a noise level (the average of the variances PCA throws away) and, for any
point, a bell-curve answer for where it sits on the sheet. Having a likelihood is the point: it allows missing values,
a fast EM algorithm that never builds the big covariance, comparison between different numbers of components,
mixtures, and sampling.

## 1. The picture to hold

Take two measurements per point, but suppose they are really governed by one hidden quantity (say a person's overall size
driving both their height and their arm length). Points then lie near a line. They do not lie *on* it, because each
measurement also has its own small wobble.

PPCA is exactly this, with a bell curve for the hidden quantity and equal wobble in every direction. The first widget of
the [interactive page](figures/interactive.html) lets you build such a cloud and fit it back.

## 2. The model

The data are vectors $x\in\mathbb R^d$. We look for a hidden vector $z\in\mathbb R^q$ with $q<d$:

$$z\sim\mathcal N(0,I_q),\qquad x=Wz+\mu+\epsilon,\qquad \epsilon\sim\mathcal N(0,\sigma^2 I_d).$$

Symbol by symbol:

- $z$ is the **latent coordinate**: $q$ numbers that say where the point sits on the sheet. The prior $\mathcal N(0,I)$ says most points
  sit near the middle of the sheet and few far out, with no preferred direction inside it.
- $W$ is a $d\times q$ table of numbers. Its $j$-th **column is a direction in data space**, and $z_j$ says how far to move along it.
  The span of the columns is the sheet.
- $\mu$ is the centre of the sheet.
- $\epsilon$ is the **noise**: independent in each coordinate and with the same variance $\sigma^2$ everywhere. The "same everywhere"
  is the whole difference from factor analysis (§9) and is what makes the answer an eigenproblem.

Equivalently, in terms of the conditional distribution, $x\mid z\sim\mathcal N(Wz+\mu,\sigma^2I)$.

## 3. What the data look like: a Gaussian with a special covariance

We never see $z$, so what matters is the distribution of $x$ alone. A sum of independent Gaussians is Gaussian, so $x$ is Gaussian, and
we only need its mean and covariance.

- **Mean.** $\mathbb E[x]=W\,\mathbb E[z]+\mu+\mathbb E[\epsilon]=\mu$.
- **Covariance.** Subtract the mean: $x-\mu=Wz+\epsilon$. Then
  $\mathbb E[(Wz+\epsilon)(Wz+\epsilon)^\top]=W\,\mathbb E[zz^\top]W^\top+\mathbb E[\epsilon\epsilon^\top]$ (the cross terms vanish because $z$ and $\epsilon$ are independent
  with mean zero), which is $WW^\top+\sigma^2I$.

$$x\sim\mathcal N(\mu,\,C),\qquad C=WW^\top+\sigma^2 I.$$

Read $C$ geometrically: the ellipse of the data is **long along the columns of $W$** (the noise adds $\sigma^2$ on top of
what $W$ contributes) and is a **round blur of radius $\sigma$ in every other direction**. The solid ellipses in the first widget are exactly this.

A general Gaussian in $d$ dimensions has about $d^2/2$ free numbers in its covariance. $C$ has about $dq$. So PPCA is a Gaussian with a
deliberately restricted covariance: a few strong directions plus a flat floor. That is a useful assumption when $d$ is large
and the data really are low-dimensional, and it is exactly the kind of structure PCA already assumes.

## 4. Why $W$ is not pinned down: the rotation freedom

Replace $W$ by $WR$ for any $q\times q$ rotation (or reflection) $R$. Then $WR(WR)^\top=WRR^\top W^\top=WW^\top$, so $C$ is unchanged and
the data distribution is identical. The reason is that rotating the latent coordinates changes nothing about the standard bell curve for $z$.

So the data determine the **sheet** (the span of the columns) and the **scale along it**, but not which axes inside the sheet you call
"first" and "second". PCA fixes the axes by also demanding they be ordered by variance. PPCA does not, and if you want PCA's axes you apply one
more rotation at the end.

## 5. Fitting by maximum likelihood

Collect $N$ points and let $S$ be the sample covariance. The log-likelihood of the Gaussian depends on the data only through $S$:

$$\ell=-\frac N2\Bigl\{d\ln 2\pi+\ln|C|+\operatorname{tr}(C^{-1}S)\Bigr\}.$$

Now use the structure of $C$. Write $W=U\,L\,R$ with $U$ having orthonormal columns, $L=\mathrm{diag}(s_j)$ and $R$ a rotation. Then $C$ has the
columns of $U$ as eigenvectors with eigenvalues $s_j^2+\sigma^2$, and every direction perpendicular to them has eigenvalue exactly $\sigma^2$.

Setting the gradient to zero, the stationary points have $U$ made of **any $q$ eigenvectors of $S$**, with matching eigenvalues
$\lambda_j$, and
$$s_j^2=\lambda_j-\sigma^2 .$$
So each kept direction is stretched until the model's variance along it, $s_j^2+\sigma^2$, equals the data's $\lambda_j$. That is
the matching you would expect, and it is only possible when $\lambda_j\ge\sigma^2$.

Two choices remain: **which** $q$ eigenvectors, and **what** $\sigma^2$.

- **$\sigma^2$.** Given the chosen set, the noise is the average of the eigenvalues of the *other* $d-q$ directions:
  $$\sigma^2_{\mathrm{ML}}=\frac1{d-q}\sum_{i=q+1}^{d}\lambda_i .$$
  The model has only one noise number to spend on all the directions it did not keep, so it takes their average.
- **Which eigenvectors.** Substituting back, the log-likelihood becomes
  $$\ell=-\frac N2\Bigl\{d\ln2\pi+\sum_{j\in\text{kept}}\ln\lambda_j+(d-q)\ln\sigma^2+d\Bigr\},$$
  and this is largest when the **top $q$** eigenvalues are the kept ones. The intuition: the kept eigenvalues are matched exactly, while the
  discarded ones are replaced by their average, and that replacement costs the gap between the log of their average and the average of their logs
  (Jensen's gap). Tipping and Bishop show the cost is smallest when what is discarded is the bottom of the spectrum.

The result:
$$W_{\mathrm{ML}}=U_q\,(\Lambda_q-\sigma^2_{\mathrm{ML}}I)^{1/2}R .$$

Three things to read from it, all on the third widget of the interactive page:

1. **The subspace is the PCA subspace.** The columns of $U_q$ are the top eigenvectors of $S$.
2. **The scale is not the PCA scale.** Each direction is shortened to $\sqrt{\lambda_j-\sigma^2}$, because the noise accounts for $\sigma^2$ of the variance
   along it. The columns of $W_{\mathrm{ML}}$ are not unit vectors, and they are not the eigenvectors.
3. **The noise is the discarded variance.** The larger the pieces PCA throws away, the larger the noise the model believes in.

## 6. From a point to its hidden coordinate: the posterior

Given an observed $x$, what do we believe about $z$? Since $z$ and $x$ are jointly Gaussian, the conditional is Gaussian too (Bayes' rule for Gaussians):

$$z\mid x\sim\mathcal N\!\bigl(M^{-1}W^\top(x-\mu),\;\sigma^2M^{-1}\bigr),\qquad M=W^\top W+\sigma^2 I_q .$$

- **Centre.** $M^{-1}W^\top(x-\mu)$ is a *shrunken* version of "project $x$ onto the sheet". The projection alone would be
  $(W^\top W)^{-1}W^\top(x-\mu)$; the extra $\sigma^2I$ inside $M$ pulls the answer toward the centre. The pull is there because the prior
  says $z$ is probably small, and the more noise there is, the less the position of $x$ can be trusted to say otherwise.
- **Spread.** $\sigma^2M^{-1}$ is the uncertainty about $z$. It is the same for every $x$ (a Gaussian special property), it grows with the noise,
  and it is smaller along directions where $W$ is long.
- **The noise-free limit.** As $\sigma^2\to0$ the centre becomes exactly PCA's projection and the spread vanishes. **PCA is PPCA with the noise turned off.**

The second widget draws this for one hidden coordinate: the PCA foot, the shrunken posterior centre and the two-standard-deviation bar, with sliders for the noise and the arrow length.

## 7. What having a likelihood buys you

- **Missing values.** The likelihood of a point with some coordinates missing is obtained by integrating those coordinates out, which just deletes
  rows and columns from $C$. So the model can be fitted on incomplete data, whereas PCA needs a complete table.
- **Choosing $q$.** Different numbers of components can be compared by the likelihood on held-out data (or by Bayesian versions of it), instead of reading a scree plot by eye.
- **Mixtures.** Several local PPCA models, each with its own sheet, can be fitted together as a mixture; a mixture of PCAs has no such natural definition.
- **Sampling and anomaly scores.** The model generates data, and $\ln p(x)$ says how surprising a new point is.

## 8. Fitting without forming the covariance: EM

The latent $z$ is hidden, which is exactly the [EM](../expectation-maximization/index.html) setting. Each round:

- **E-step.** With the current $W,\sigma^2$, compute for each point $\langle z_n\rangle=M^{-1}W^\top(x_n-\mu)$ and the second moment
  $\langle z_nz_n^\top\rangle=\sigma^2M^{-1}+\langle z_n\rangle\langle z_n\rangle^\top$ (the posterior of §6).
- **M-step.**
  $$W_{\text{new}}=\Bigl[\sum_n(x_n-\mu)\langle z_n\rangle^\top\Bigr]\Bigl[\sum_n\langle z_nz_n^\top\rangle\Bigr]^{-1},$$
  and $\sigma^2_{\text{new}}$ is the average remaining squared error per coordinate under the new $W$.

Every quantity is a $d\times q$ or $q\times q$ table, so a round costs about $N\,d\,q$ operations and the $d\times d$ sample covariance
never has to be built. This is why PPCA's EM is attractive when $d$ is large and $q$ small. The likelihood of this model has no bad local maxima:
the only maxima are the ones in §5 (all choices of eigenvectors other than the top $q$ are saddles), so EM converges to the PCA subspace.

## 9. Relatives

- **Factor analysis (FA)** is the same model with noise $\epsilon\sim\mathcal N(0,\Psi)$ where $\Psi$ is diagonal and *different* for each
  coordinate. That is more flexible, since coordinates measured with different precision can have different noise, but it has no closed
  form, and its answer changes if you rescale a coordinate. PPCA's equal noise is what makes it an eigenproblem, and it is also what makes it
  not invariant to rescaling the coordinates, just like PCA.
- **PCA** is the limit $\sigma^2\to0$.
- **Bayesian PCA** puts a prior on the columns of $W$ so that unneeded columns shrink to zero, which chooses $q$ automatically.
- **Kernel and nonlinear versions** replace the linear map $Wz$ with a flexible function of $z$; see the next section.

## 10. Where it reappears

- **Linear VAE.** A VAE with a linear Gaussian decoder $x=Wz+\mu+\epsilon$ and a standard-normal prior is exactly PPCA. Its exact posterior is the one in §6, so the
  ELBO's gap closes, and the optimum recovers the PCA subspace. A nonlinear decoder is "PPCA with a bent sheet".
  See Lucas, Tucker, Grosse and Norouzi, ["Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse"](https://arxiv.org/abs/1911.02469) (NeurIPS 2019).
- **Anomaly and OOD scores.** The sheet is "normal", and the distance to it, plus the position on it, is the score. The posterior spread above
  tells you how much of that distance is expected from noise alone.
- **Data with missing entries.** Whenever a table has holes, the model of §7 is the principled replacement for filling in the mean first.

## Questions and doubts

- **How should one pick $q$?** The maximum likelihood on training data always improves with $q$, so a held-out score, a Bayesian
  version, or a gap in the spectrum (third widget) is needed. A gradual spectrum has no honest answer.
- **Is equal noise believable?** If coordinates have different units or precisions, PPCA treats the high-variance ones as signal. Standardising first changes the answer, and
  neither choice is the "correct" one.
- **Does the posterior spread matter in practice?** It is the same for every point in this model, which is a limitation of the Gaussian assumption.
  A nonlinear decoder breaks this and gives a point-dependent spread.
- **Where is the statement "all other stationary points are saddles" proved?** Tipping and Bishop state it; I have not worked through that proof here.

## Takeaways

- PPCA = a hidden short Gaussian vector, a linear map into data space, and equal noise on every coordinate.
- The data are Gaussian with covariance $WW^\top+\sigma^2I$: a few strong directions on a flat floor.
- Maximum likelihood gives the **PCA subspace**, with each direction shortened to $\sqrt{\lambda_j-\sigma^2}$ and $\sigma^2$ equal to the **average of the discarded eigenvalues**.
- Each point gets a **Gaussian posterior** over its coordinate, a shrunken version of PCA's projection, which becomes PCA's projection as the noise goes to zero.
- The reward for the extra structure is a likelihood: missing data, EM in $O(Ndq)$, model comparison, mixtures and sampling.
- Equal noise is both the reason the answer is an eigenproblem and the reason PPCA, like PCA, depends on how the coordinates are scaled.

## Reference

M. E. Tipping and C. M. Bishop, "Probabilistic Principal Component Analysis", *Journal of the Royal Statistical Society, Series B*
61(3), 611–622, 1999 (technical report NCRG/97/010, 1997/1998).
