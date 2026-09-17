---
title: "InfoNCE Induces Gaussian Distribution"
authors: "Roy Betser, Eyal Gofer, Meir Yossef Levi, Guy Gilboa (Technion)"
venue: "ICLR"
year: 2026
url: ""
pdf_url: ""
tags: [contrastive-learning, infonce, self-supervised, high-dimensional-probability, theory]
status: read
---

## In one paragraph

Contrastive training with InfoNCE is usually described geometrically: positive pairs
are pulled together, everything else is pushed apart, and the representations end up
"spread out" on the hypersphere. This paper asks the sharper probabilistic question —
*which distribution* do the representations actually converge to — and answers:
asymptotically Gaussian. The argument is that data augmentation puts a hard ceiling on
how much alignment the loss can buy; once alignment saturates at that ceiling, only the
uniformity term is still doing work, and its unique minimiser is the uniform law on the
sphere. Classical high-dimensional probability then converts "uniform on $S^{d-1}$"
into "Gaussian low-dimensional projections". Two routes to the same conclusion are given:
one leaning on observed training dynamics, one on a vanishing convex regulariser.

## The spine of the argument

1. InfoNCE $=$ $-$alignment $+$ uniformity potential (Eq. 4).
2. Augmentation strength caps alignment at $\eta_2$, via HGR maximal correlation (Prop. 1).
   Beating that cap requires breaking centredness.
3. With alignment capped and constant, only the uniformity potential varies — and it is
   *uniquely* minimised by the uniform law $\sigma$ on $S^{d-1}$.
4. Uniform on $S^{d-1}$ with $d$ large $\Rightarrow$ Gaussian $k$-dimensional projections
   (Maxwell–Poincaré spherical CLT, total-variation error $O(1/d)$).
5. Add thin-shell norm concentration and the result lifts from the sphere to the raw
   unnormalised embeddings in $\mathbb{R}^d$.
6. The regularised route replaces the dynamics-dependent assumptions (3 and 5) with a
   vanishing convex regulariser, reaching the same conclusion by a competing-quadratics
   argument in $\lVert m(\mu) \rVert^2$.

Steps 1–2 are the genuinely new contribution. Steps 3–4 are classical results being
connected up. Steps 5–6 are the robustness argument.

## Setup and notation

| Symbol | Meaning |
|---|---|
| $(\mathcal{X}, \mathcal{B}(\mathcal{X}))$ | standard Borel data space |
| $p_{\text{base}}$ | distribution of raw data items; $X_0 \sim p_{\text{base}}$ |
| $A(\cdot \mid X_0)$ | augmentation channel (a Markov kernel) |
| $X, Y \sim A(\cdot \mid X_0)$ | two independent views of the same base item |
| $f : \mathcal{X} \to \mathbb{R}^d$ | encoder; $\hat f = f/\lVert f \rVert$ its normalised version |
| $\mu = \hat f_* p_X$ | law of normalised embeddings, on $S^{d-1}$ |
| $\rho = f_* p_X$ | law of unnormalised embeddings, in $\mathbb{R}^d$ |
| $\pi = (\hat f, \hat f)_* p_{XY}$ | joint law of embedded positive pairs |
| $\sigma$ | uniform distribution on $S^{d-1}$ |
| $\alpha = 1/\tau$ | inverse temperature |
| $\eta_2$ | augmentation mildness, $= \rho_m^2(X, X_0)$ |
| $m(\mu) = \mathbb{E}[u]$ | mean embedding |

### Why a standard Borel space and a base probability

$p_{\text{base}}$ is simply "the distribution the data comes from" — one raw image before
augmentation. Everything downstream is built on it.

The *standard Borel* qualifier is a technical licence that buys three things the paper
genuinely uses later:

1. **Conditional distributions exist.** The augmentation channel $A(\cdot \mid X_0)$ is a
   Markov kernel. In general measure spaces well-behaved conditional distributions are
   not guaranteed; on a standard Borel space they always exist.
2. **Disintegration works.** Proposition 3 writes $\rho(dz) = \mu(du)\,\kappa(dr \mid u)$ —
   splitting a law on $\mathbb{R}^d$ into direction times radius-given-direction. That is
   the disintegration theorem, which needs standard Borel.
3. **The key move of Section 4.2.** The paper states it outright: *because* the space is
   standard Borel with nonatomic $p_X$, every $\rho \in \mathcal{P}(B)$ equals $g_* p_X$
   for some encoder $g$. This is what licenses switching from optimising over encoders $f$
   to optimising over distributions $\mu$ directly. Without it, "let us simply choose the
   uniform distribution on the sphere" would be an illegal move — one would first have to
   exhibit a network realising it.

**Concrete instance.** $\mathcal{X} = [0,1]^{3072} \subset \mathbb{R}^{3072}$, the CIFAR-10
image space with its Borel $\sigma$-algebra (a closed subset of $\mathbb{R}^n$ is
automatically standard Borel, so this costs nothing). $p_{\text{base}}$ is the CIFAR-10
image distribution, $X_0$ one image, $A(\cdot\mid X_0)$ random crop $+$ flip $+$ colour
jitter, and $f$ a ResNet-18 into $\mathbb{R}^{128}$ — Borel-measurable because continuous.
The nonatomic assumption on $p_X$ says no exact pixel array has positive probability;
with a finite dataset that is technically false, which is why the paper mentions
*infinitesimal dither* as the practical repair.

### Pushforward measures, informally

The pushforward of $p$ by $f$, written $f_* p$, is nothing more than **the distribution of
$f(X)$ when $X \sim p$**. Formally

$$
(f_* p)(B) = p\big(f^{-1}(B)\big),
$$

that is, the probability the output lands in $B$ equals the probability the input landed
in whatever maps into $B$.

*Toy case.* $X \sim \mathrm{Uniform}[0,1]$ and $f(x) = x^2$. Then $f_* p$ is the law of
$X^2$, with density $1/(2\sqrt{y})$ on $[0,1]$. This is the ordinary change-of-variables
rule for random variables; "pushforward" is just the name for the resulting object.

*Practical picture.* Embed the whole dataset, collect every $f(x)$ into an array, and
histogram it — that histogram estimates $f_* p_X$.

The notation is used rather than "the distribution of $f(X)$" because it names the map
$\text{encoder} \mapsto \text{distribution}$ explicitly, and the paper's entire strategy
is to treat the resulting $\mu$ as a free variable.

## The population objective (Eq. 4)

Splitting $\log(a/b) = \log a - \log b$ in the empirical loss (Eq. 2):

$$
\begin{aligned}
\mathcal{L}_{\text{InfoNCE}}
&= -\underbrace{\frac{1}{\tau}\cdot\frac{1}{N}\sum_i \langle u_i, v_i\rangle}_{\text{alignment}} \\[4pt]
&\quad + \underbrace{\frac{1}{N}\sum_i \log \sum_j \exp\!\big(\langle u_i, v_j\rangle/\tau\big)}_{\text{uniformity}}
\end{aligned}
$$

Letting $N \to \infty$, setting $\alpha = 1/\tau$ and dropping the $\log N$ offset gives

$$
\mathcal{L}(\mu, \pi) = -\alpha\, \mathbb{E}_{(u,v)\sim\pi}[u \cdot v] + \Phi(\mu),
\qquad
\Phi(\mu) := \mathbb{E}_{u\sim\mu} \log \mathbb{E}_{v\sim\mu} \exp(\alpha\, u\cdot v).
$$

**Alignment.** Carries a minus sign, so minimising the loss maximises it: pull positive
pairs together. Since $u, v$ are unit vectors this is cosine similarity, and
$\mathbb{E}[u\cdot v] = 1 - \tfrac12 \mathbb{E}\lVert u - v\rVert^2$, so maximising
alignment is the same as minimising squared distance between views.

**Uniformity.** $\exp(\alpha\, u \cdot v)$ is large when $v$ points roughly where $u$ does,
so $\mathbb{E}_v \exp(\alpha\, u\cdot v)$ measures *how crowded the neighbourhood of $u$ is*.
$\Phi$ is the average log-crowdedness over all points; minimising it spreads points out.
It is a Gaussian-kernel energy on the sphere, and Wang & Isola proved it is **uniquely
minimised by the uniform law $\sigma$ on $S^{d-1}$**. That fact is the engine of the paper.

**The structural point.** Alignment depends on the *pairing* $\pi$; uniformity depends
**only on the marginal** $\mu$. The two pull in opposite directions — perfect alignment
would place every embedding at one point, the worst possible $\Phi$. The paper's story is
that augmentation strength caps what the alignment term can pay, after which only $\Phi$
matters, and $\Phi$ wants uniform.

## HGR maximal correlation

**The problem it fixes.** Pearson correlation sees only linear relationships. If $Y = X^2$
with $X$ symmetric, Pearson is exactly $0$ even though $Y$ is a deterministic function of
$X$. Useless as a dependence measure.

**The fix.** Reshape both variables arbitrarily first, then correlate:

$$
\rho_m(X,Y) \;=\; \sup_{\substack{\mathbb{E}[\varphi(X)] = \mathbb{E}[\psi(Y)] = 0 \\ \operatorname{Var}(\varphi) = \operatorname{Var}(\psi) = 1}} \mathbb{E}\big[\varphi(X)\,\psi(Y)\big] \;\in\; [0,1].
$$

Over all ways of transforming $X$ and all ways of transforming $Y$, what is the best
correlation obtainable?

Properties that matter here:

- $\rho_m = 0$ **if and only if** $X \perp Y$. Pearson cannot do this.
- $\rho_m = 1$ when one variable determines the other.
- For **jointly Gaussian** $X, Y$: $\rho_m = |\text{Pearson}|$ — nonlinear transforms buy
  nothing (Appendix A.2 of the paper).
- **Multiplicative data-processing inequality**: if $X - Y - Z$ is a Markov chain then
  $\rho_m(X,Z) \le \rho_m(X,Y)\,\rho_m(Y,Z)$. Dependence decays along a chain.

The equivalent *explained-variance* form (Eqs. 5 and 22) reads as a nonlinear $R^2$:

$$
\rho_m^2(X,Y) = \sup_{g \in L^2(p_X),\, \operatorname{Var}(g) > 0}
\frac{\operatorname{Var}\big(\mathbb{E}[g(X)\mid Y]\big)}{\operatorname{Var}\big(g(X)\big)}
$$

— over all real-valued features $g$ of $X$, what fraction of that feature's variance is
predictable from $Y$?

**In this paper**, $\eta_2 := \rho_m^2(X, X_0)$ is called the **augmentation mildness**,
with $X$ an augmented view and $X_0$ the base item.

| $\eta_2$ | Meaning |
|---|---|
| $\to 1$ | views barely differ from the source; **mild** augmentation |
| $\to 0$ | augmentation destroys the source; views effectively independent of $X_0$; **aggressive** augmentation |

So a single number in $[0,1]$ summarises augmentation strength.

**Why HGR rather than mutual information?** The quantity being bounded,
$\mathbb{E}[u\cdot v]$, is a *correlation between $L^2$ functions* of the two views, and
HGR is precisely the sharp constant for that kind of bound across a Markov chain. Mutual
information would produce a bound of the wrong shape.

**The line that does the work.** The two views satisfy $X \leftarrow X_0 \rightarrow Y$,
i.e. the Markov chain $X - X_0 - Y$, so

$$
\rho_m(X,Y) \;\le\; \rho_m(X,X_0)\,\rho_m(X_0,Y) \;=\; \sqrt{\eta_2}\cdot\sqrt{\eta_2} \;=\; \eta_2 .
$$

The two views are at most $\eta_2$-correlated **whatever encoder is used**. That is the
hard ceiling.

## Proposition 1: the alignment bound

$$
\mathbb{E}_{(u,v)\sim\pi}[u\cdot v] \;\le\; \eta_2 + (1-\eta_2)\,\lVert m(\mu) \rVert^2,
\qquad m(\mu) := \mathbb{E}[u] = \mathbb{E}[v].
$$

**Read the right-hand side as two sources of alignment.**

- $\eta_2$ — alignment earned *honestly*, by genuinely recognising that two views share a
  source. Capped by augmentation strength.
- $(1-\eta_2)\lVert m(\mu)\rVert^2$ — alignment obtained *by cheating*: if all embeddings
  drift toward one common direction, any two have high cosine similarity whether or not
  they are a real pair. $\lVert m\rVert = 0$ means perfectly centred (e.g. uniform on the
  sphere); $\lVert m\rVert = 1$ means total collapse to a point.

### Proof in four steps

**Step 1 — split off the common mean.** $u$ and $v$ share the marginal $\mu$, hence the
mean $m$. With residuals $\tilde u = u - m$, $\tilde v = v - m$:

$$
\mathbb{E}[u\cdot v] = \lVert m\rVert^2 + \mathbb{E}[\tilde u \cdot \tilde v],
$$

the cross terms vanishing because the residuals are mean-zero. This separates *bias
alignment* from *genuine co-variation*.

**Step 2 — bound the genuine part with HGR.** For coordinate $k$, put $g_k(X) := \tilde u_k$
and $h_k(Y) := \tilde v_k$, both mean-zero and square-integrable. The definition of HGR
plus Cauchy–Schwarz gives
$\mathbb{E}[g(X)h(Y)] \le \rho_m(X,Y)\sqrt{\operatorname{Var}(g)\operatorname{Var}(h)}$,
and the DPI gives $\rho_m(X,Y) \le \eta_2$. Summing over coordinates and applying
Cauchy–Schwarz again over the sequence:

$$
\mathbb{E}[\tilde u \cdot \tilde v] \;\le\; \eta_2 \sqrt{\textstyle\sum_k \operatorname{Var}(\tilde u_k)} \sqrt{\textstyle\sum_k \operatorname{Var}(\tilde v_k)} .
$$

**Step 3 — the unit-norm constraint pins the variance exactly.** Because $\lVert u\rVert = 1$,

$$
\sum_k \operatorname{Var}(\tilde u_k) = \mathbb{E}\lVert u - m \rVert^2 = \mathbb{E}\lVert u\rVert^2 - \lVert m\rVert^2 = 1 - \lVert m\rVert^2 .
$$

This is where normalisation earns its keep: the total variance is not an unknown, it is
exactly $1 - \lVert m\rVert^2$.

**Step 4 — combine.** $\mathbb{E}[\tilde u\cdot\tilde v] \le \eta_2(1 - \lVert m\rVert^2)$, so

$$
\mathbb{E}[u \cdot v] \le \lVert m\rVert^2 + \eta_2\big(1 - \lVert m\rVert^2\big) = \eta_2 + (1-\eta_2)\lVert m\rVert^2 . \qquad \blacksquare
$$

### Sanity checks

- $\eta_2 = 1$ (no augmentation) $\Rightarrow$ bound is $1$, vacuous. Correct: with
  identical views one can align perfectly.
- $\eta_2 = 0$ (independent views) $\Rightarrow$ bound is $\lVert m\rVert^2$. Only collapse
  buys alignment.
- $\lVert m \rVert = 0$ (centred / uniform) $\Rightarrow$ bound is exactly $\eta_2$. **This
  is the case Theorem 1 uses.**

### Why this is the pivot of the paper

Alignment and uniformity are in direct competition, and this proposition sets the exchange
rate. Alignment can exceed $\eta_2$ *only* by breaking uniformity — growing
$\lVert m\rVert$. In Theorem 1 the extra alignment gained is $\alpha(1-\eta_2)\lVert m\rVert^2$
while Lemma 1 says the KL penalty paid is at least $\beta C(d-1)\lVert m\rVert^2$. Both are
quadratic in $\lVert m\rVert$, but the penalty's coefficient **grows with dimension**. In
high $d$ the penalty always wins, so $m = 0$ — uniform — is optimal. That is Eq. 19 in one
sentence.

## The two empirical assumptions (Section 4.1)

### Alignment plateau (Assumption 1)

$$
\mathbb{E}_{(u,v)\sim\pi}[u\cdot v] = \eta_2 + r_{\text{plat}}, \qquad r_{\text{plat}} \le 0 \text{ constant}.
$$

Training alignment rises quickly, then flattens: it reaches a ceiling at or below $\eta_2$
and stops improving, while uniformity keeps improving with larger $d$ and larger batches
(Fig. 2 of the paper).

**The operative word is "constant".** Once $\mathbb{E}[u\cdot v]$ no longer depends on
$\mu$, it is an additive constant and drops out of the optimisation:
$\mathcal{L}(\mu,\pi) = -\alpha \cdot \text{const} + \Phi(\mu)$. So minimising InfoNCE
$\equiv$ minimising $\Phi(\mu)$ $\equiv$ $\mu = \sigma$ (Lemma 2). The Maxwell–Poincaré
spherical CLT then converts "uniform on $S^{d-1}$" into "any fixed $k$ coordinates,
rescaled by $\sqrt{d}$, are standard Gaussian", with the explicit rate

$$
d_{\mathrm{TV}}\Big(\sqrt{d}\,(U_{d,1},\dots,U_{d,k}),\; \mathcal{N}(0, I_k)\Big) \;\le\; \frac{2(k+3)}{d-k-3}, \qquad 1 \le k \le d-4,
$$

i.e. $O(1/d)$. The authors are explicit that the plateau is a modelling choice rather than
a theorem, and note the plateau value must be feasible at $\mu = \sigma$ — hence
$r_{\text{plat}} \le 0$.

### Thin-shell concentration (Assumption 2)

$$
\frac{r}{r_0} \xrightarrow[d\to\infty]{} 1, \qquad r = \lVert z \rVert,\; r_0 \in (0,\infty) \text{ deterministic}.
$$

All unnormalised embeddings end up with essentially the same norm: they occupy a thin
spherical shell, not the interior of a ball. Measured empirically by the coefficient of
variation

$$
\mathrm{CV} = \frac{\operatorname{std}\{\lVert z_i\rVert\}}{\operatorname{mean}\{\lVert z_i\rVert\}} \longrightarrow 0 .
$$

Three forces push this way: it is a generic high-dimensional concentration phenomenon
(Klartag's thin-shell); InfoNCE normalises embeddings, so nothing in the loss rewards
spreading norms apart; and weight decay actively suppresses norm inflation.

**Why it is needed.** InfoNCE only ever sees *direction*, so Corollary 1 speaks only about
$u$. To say anything about the raw $z = r u$ one needs control of $r$. With $r \approx r_0$
deterministic, Slutsky's theorem gives

$$
\sqrt{d}\, z_k = r \cdot \big(\sqrt{d}\, u_k\big) \;\Rightarrow\; \mathcal{N}\big(0, r_0^2 I_k\big).
$$

Without it $r$ would be random and the limit would be a *scale mixture* of Gaussians —
heavy-tailed, and exactly what the paper's normality tests would reject.

## The regularised route (Section 4.2)

### The truncated Gaussian (Eq. 12)

$$
\gamma_\lambda^B(dz) = c_{B,\lambda}\, e^{-\lambda \lVert z\rVert^2}\, \mathbf{1}_B(z)\, dz,
\qquad c_{B,\lambda}^{-1} = \int_B e^{-\lambda\lVert z\rVert^2}\, dz .
$$

Three pieces, left to right:

- $e^{-\lambda\lVert z\rVert^2}$ — an isotropic Gaussian shape. Matching against
  $e^{-\lVert z\rVert^2/2\varsigma^2}$ gives $\varsigma^2 = 1/(2\lambda)$, so this is
  $\mathcal{N}\big(0, (2\lambda)^{-1} I\big)$. Larger $\lambda$ means tighter.
- $\mathbf{1}_B(z)$ — zero out everything outside $B$, which is either a closed ball around
  the origin or all of $\mathbb{R}^d$.
- $c_{B,\lambda}$ — renormalise so the whole thing integrates to $1$. That is all the
  integral in the definition does.

So: **take a Gaussian, cut off everything outside the ball, scale what remains back up.**
With $B = \mathbb{R}^d$ there is no cutting and it is exactly
$\mathcal{N}\big(0,(2\lambda)^{-1}I\big)$ — which is Corollary 2.

*Why this object?* The regulariser in Eq. 11 is
$\beta\big(-H(\rho) + \lambda\,\mathbb{E}\lVert Z\rVert^2\big)$, i.e. "high entropy, small
norm". The maximum-entropy distribution subject to a second-moment budget *is* a Gaussian;
on a bounded domain it is a truncated Gaussian. So $\gamma_\lambda^B$ is precisely what the
regulariser is implicitly asking for, and Eq. 13 makes that exact.

*Useful aside.* In $d$ dimensions the **radius** of a truncated Gaussian has density
proportional to $r^{d-1} e^{-\lambda r^2}$ — the $r^{d-1}$ being the surface-area Jacobian.
It peaks at $r_0 = \sqrt{(d-1)/(2\lambda)}$ with relative width $O(1/\sqrt d)$. So the thin
shell of Assumption 2 is not an extra assumption in the regularised route; it falls out of
the reference measure for free. Even a plain isotropic Gaussian in high dimensions already
lives on a shell.

### Deriving Eq. 13

The standard "KL $=$ cross-entropy $-$ entropy" identity, specialised to a Gaussian
reference. Three moves.

**Move 1 — split the Radon–Nikodym derivative using Lebesgue measure as a common yardstick.**

$$
\frac{d\rho}{d\gamma} = \frac{d\rho/dz}{d\gamma/dz}
\quad\Longrightarrow\quad
\log \frac{d\rho}{d\gamma} = \log\frac{d\rho}{dz} - \log\frac{d\gamma}{dz},
$$

and integrating against $d\rho$ produces exactly the two integrals of Eq. 13:

$$
\mathrm{KL}(\rho \Vert \gamma) = \int \log\frac{d\rho}{dz}\, d\rho \;-\; \int \log\frac{d\gamma}{dz}\, d\rho .
$$

**Move 2 — the first integral is negative entropy.** Writing $p(z) = d\rho/dz$, that term is
$\int p \log p \, dz$, and differential entropy is $H(\rho) = -\int p\log p$. So the first
integral *is* $-H(\rho)$.

**Move 3 — the second integral is where the Gaussian shape pays off.** On $B$,

$$
\log \frac{d\gamma_\lambda^B}{dz} = \log c_{B,\lambda} - \lambda\lVert z\rVert^2
\quad\Longrightarrow\quad
-\int \log\frac{d\gamma}{dz}\, d\rho = \log c_{B,\lambda}^{-1} + \lambda\, \mathbb{E}_\rho\lVert Z\rVert^2 .
$$

Because the reference density is $\exp(-\lambda\lVert z\rVert^2)$, taking its log turns the
exponent into a polynomial, and integrating that polynomial against $\rho$ is just a
**second moment**. That is the whole trick: a Gaussian reference converts cross-entropy
into $\mathbb{E}\lVert Z\rVert^2$.

Adding up,

$$
\mathrm{KL}(\rho \Vert \gamma_\lambda^B) = -H(\rho) + \lambda\,\mathbb{E}_\rho \lVert Z\rVert^2 + \log c_{B,\lambda}^{-1},
$$

where the last term depends only on $B, \lambda, d$ — that is what "equality up to an
additive constant" means. The degenerate case is handled too: if $\rho \not\ll \gamma$
(mass outside $B$, atoms, or a singular part) then both $\mathrm{KL}$ and $-H(\rho)$ are
$+\infty$, so the identity holds unconditionally in the extended sense.

### From Eq. 13 to Eq. 14

Pure substitution. Starting from Eq. 11 and replacing the bracket using Eq. 13:

$$
J(f) = \Phi(\mu) - \alpha\,\mathbb{E}[u\cdot v] + \beta\,\mathrm{KL}(\rho \Vert \gamma_\lambda^B) - \underbrace{\beta \log c_{B,\lambda}^{-1}}_{\text{constant in } f} .
$$

Dropping the constant leaves the argmin unchanged, which is Eq. 14. The rewrite buys two
things beyond cosmetics:

1. **Interpretation.** The regulariser literally reads "pull the embedding distribution
   toward a truncated Gaussian in KL". Two apparently unrelated engineering penalties turn
   out to be one principled object.
2. **The domain constraint becomes automatic.** Finite $\mathrm{KL}(\rho\Vert\gamma_\lambda^B)$
   forces $\rho \ll \gamma_\lambda^B$, which forces $\rho(B) = 1$; no separate feasibility
   constraint is needed.

### Radial and angular decomposition (Proposition 3)

Any $z \ne 0$ in $\mathbb{R}^d$ splits uniquely as

$$
z = r\,u, \qquad r = \lVert z\rVert \in (0,\infty), \qquad u = z/\lVert z\rVert \in S^{d-1},
$$

so $\mathbb{R}^d \setminus \{0\} \cong (0,\infty) \times S^{d-1}$ — "how far out" times
"which direction". Applied to distributions this is the disintegration theorem (the payoff
from the standard Borel setup):

$$
\rho(dz) = \mu(du)\,\kappa(dr \mid u), \qquad \gamma_\lambda^B(dz) = \sigma(du)\,\xi(dr\mid u),
$$

with $\mu$ the **angular marginal** on the sphere and $\kappa(\cdot\mid u)$ the
**conditional radial law**. Since $\gamma_\lambda^B$ is isotropic, $\sigma$ is uniform and
$\xi$ does not depend on $u$ — it is the $r^{d-1}e^{-\lambda r^2}$ law noted above.

The KL chain rule then gives

$$
\mathrm{KL}(\rho \Vert \gamma_\lambda^B) = \underbrace{\mathrm{KL}(\mu \Vert \sigma)}_{\text{angular}} + \underbrace{\int \mathrm{KL}\big(\kappa(\cdot\mid u) \,\Vert\, \xi(\cdot\mid u)\big)\, \mu(du)}_{\text{radial}} .
$$

The radial term is $\ge 0$ and can be set to **exactly zero** by choosing $\kappa = \xi$ —
and, crucially, that choice costs nothing elsewhere, because $\Phi(\mu)$ and
$\mathbb{E}[u\cdot v]$ act on *normalised* vectors and therefore see only the angular part.
The radius can be optimised in isolation.

What this achieves:

- A hard $d$-dimensional optimisation over laws on $\mathbb{R}^d$ collapses to an
  optimisation on the sphere,
  $\tilde J(\mu) = \Phi(\mu) - \alpha\,\mathrm{Align}(\mu) + \beta\,\mathrm{KL}(\mu\Vert\sigma)$ (Eq. 16).
- The optimal radial law is obtained in **closed form**, not assumed. This fixes a real
  blind spot: plain InfoNCE is completely oblivious to embedding norms because it
  normalises them away; the regulariser supplies the missing radial law.
- Corollary 2 follows immediately: with $B = \mathbb{R}^d$ and $\mathrm{Align}(\sigma) = \eta_2$,
  the optimum is uniform angular $\times$ Gaussian radial, i.e. exactly
  $\mathcal{N}\big(0, (2\lambda)^{-1} I_d\big)$.

## Experiments, in brief

Three diagnostics: coefficient of variation of norms (radial concentration), the
Anderson–Darling statistic (accept normality below $0.752$), and D'Agostino–Pearson
($p > 0.05$ fails to reject). Three settings of increasing complexity: synthetic data
(Laplace, a 25-component Gaussian mixture, sparse binary) with linear encoders; CIFAR-10
with contrastive *and* supervised training, so the objective is isolated from data and
architecture; and pretrained foundation models including DINO. Norms concentrate and
coordinates pass the normality tests, with the effect strengthening in $d$ and batch size.

## Questions and doubts

- **Is the regularised route circular?** It regularises toward a Gaussian and then
  discovers a Gaussian. The defence is that the threshold
  $\beta_0 = \alpha(1-\eta_2)/\big(C(d-1)\big) \to 0$ as $d \to \infty$, so the nudge
  vanishes and the InfoNCE term carries the conclusion. Worth deciding for oneself how
  convincing that is.
- **Assumption 3** ($\alpha(\eta_2 - \mathrm{Align}(\sigma)) \to 0$) asserts the alignment
  ceiling is attainable *at uniformity*. The paper does not establish when this holds.
- **The plateau is asserted, not derived.** No optimisation dynamics are analysed; the
  results characterise population optima under the stated assumptions, and the paper says
  so plainly in its limitations.
- **$\eta_2$ is never measured.** The augmentation mildness is central to every bound but
  no experiment estimates it for real augmentation pipelines. Estimating HGR maximal
  correlation empirically is itself nontrivial.
- **Gaussianity is a marginal statement**, not a claim about class structure. The paper is
  explicit that well-separated class clusters can coexist with an approximately Gaussian
  overall embedding law — worth keeping in mind before reading anything into it about
  downstream separability.

## Takeaways

- Alignment is capped by augmentation strength; the cap is $\eta_2$ at centred
  distributions, and exceeding it requires partial collapse. This bound is the paper's
  own contribution and is independent of the Gaussianity story.
- Once alignment saturates, InfoNCE is a pure uniformity objective, whose minimiser is
  uniform on the sphere — and uniform-on-the-sphere in high dimension *is* Gaussian when
  viewed through any fixed low-dimensional projection.
- The Gaussian modelling assumptions already common in practice (likelihood scoring,
  OOD detection, uncertainty estimation on contrastive features) get a population-level
  justification.
- Explicit isotropy-promoting regularisers may be reasonable surrogates for InfoNCE's
  implicit bias, which is a concrete design suggestion rather than only an explanation.
