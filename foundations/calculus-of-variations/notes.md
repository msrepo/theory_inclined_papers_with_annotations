---
title: "Calculus of variations: finding the best function"
authors: "Background notes"
venue: "Foundations"
tags: [calculus-of-variations, functional, euler-lagrange, maximum-entropy, gaussian, background]
status: living
category: "Foundations"
subcategory: "Optimization"
short_title: "Calculus of variations"
summary: "a functional as a height over a space of functions; nudge-and-look-at-the-slope giving the Euler–Lagrange equation and the functional derivative; the shortest path as a worked case; a multiplier per constraint; the Gaussian as the maximum-entropy density for a given mean and variance (with an explorer mixing it toward other densities), great circles as the shortest paths on a sphere (with a globe where pushing the arc sideways always lengthens it), and the exponential family as the general pattern"
---

## Links

- **[Interactive version](figures/interactive.html)**: compare a Gaussian with a uniform, Laplace, logistic or two-bump density
  that has the same mean and spread, and slide along the path between them to watch the entropy fall away from the Gaussian; and a sphere where you push a great-circle arc sideways and watch it lengthen.
- **[Lagrange multiplier](../lagrange-multiplier/index.html)**: the same constrained-maximum idea for a point instead of a
  function. Its maximum-entropy section is the discrete version of the Gaussian derivation below.
- **[Fisher information](../fisher-information/index.html)**: where the exponential family at the end of this page lives.

## What a functional is

An ordinary function takes a number and returns a number. A **functional** takes a whole function and returns one number.
$$J[y]=\int_a^b F\bigl(x,\,y(x),\,y'(x)\bigr)\,dx$$
is a functional: hand it a curve $y$ and it hands back, say, the curve's length, or its area, or a distribution's entropy.

Think of a function as a single point in a space with a coordinate for every value of $x$. A functional is then a height
assigned to every such point, a landscape over a space of functions.

## The problem

**Calculus of variations asks for the function $y$ that makes $J[y]$ smallest (or largest).**

Ordinary calculus finds the number $x$ where $f(x)$ is lowest. Here the unknown is a whole curve. Classic cases:

- The shortest path between two points: minimise arc length $\int\sqrt{1+y'^2}\,dx$. The answer is a straight line.
- The fastest slide for a bead under gravity (brachistochrone). The answer is a cycloid.
- A soap film: the surface of least area across a wire frame.
- A particle in physics follows the path that makes the action $\int L\,dt$ stationary.
- The distribution with the most entropy under given constraints (below).

## The one idea: nudge and look at the slope

At a lowest point of $f(x)$ the slope is zero: a small nudge of $x$ changes $f$ only at second order.

Do the same for functions. Suppose $y$ is the best curve. Add a small bump $\varepsilon\,\eta(x)$, where the bump $\eta$ is zero at the
two ends so the endpoints stay fixed. The family $y+\varepsilon\eta$ is a one-parameter slice through the landscape, so
$J(\varepsilon)=J[y+\varepsilon\eta]$ is an ordinary function of the single number $\varepsilon$, lowest at $\varepsilon=0$. So
$$\frac{dJ}{d\varepsilon}\Big|_{\varepsilon=0}=0\qquad\text{for every bump }\eta.$$
"Every bump" is what makes this strong: it is the slope being zero in every direction of the landscape at once.

## The Euler–Lagrange equation

Differentiate under the integral:
$$\frac{dJ}{d\varepsilon}\Big|_0=\int_a^b\Bigl(\frac{\partial F}{\partial y}\,\eta+\frac{\partial F}{\partial y'}\,\eta'\Bigr)dx.$$
Integrate the second term by parts. The boundary term vanishes because $\eta(a)=\eta(b)=0$:
$$\int_a^b\Bigl(\frac{\partial F}{\partial y}-\frac{d}{dx}\frac{\partial F}{\partial y'}\Bigr)\eta\,dx=0.$$
Because $\eta$ can be any bump, the bracket must be zero at every $x$:
$$\boxed{\;\frac{\partial F}{\partial y}-\frac{d}{dx}\frac{\partial F}{\partial y'}=0\;}$$
This is the **Euler–Lagrange equation**. It turns "minimise a functional" into "solve a differential equation".
The left side is the **functional derivative** $\delta J/\delta y$, the gradient of the landscape. Stationary means that gradient is zero,
as $\nabla f=0$ does for a function of several numbers.

**Example: the shortest path.** Here $F=\sqrt{1+y'^2}$ does not contain $y$, so the equation reads
$\dfrac{d}{dx}\dfrac{y'}{\sqrt{1+y'^2}}=0$. The quantity $y'/\sqrt{1+y'^2}$ is constant, so $y'$ is constant, so $y$ is a straight line.

## Constraints: a Lagrange multiplier per constraint

If the best function must also satisfy $G[y]=c$ (for example, total probability equal to 1), make $J-\lambda G$ stationary and then choose
$\lambda$ so the constraint holds. This is exactly the [Lagrange multiplier](../lagrange-multiplier/index.html) rule, with the
functional derivative standing in for the gradient.

## Worked example: the Gaussian is the maximum-entropy density for a given mean and variance

Among all densities $p(x)$ on the real line, which has the largest entropy $H[p]=-\int p\log p\,dx$ with these three constraints?

1. $\int p\,dx=1$ (it is a density),
2. $\int x\,p\,dx=\mu$ (fixed mean),
3. $\int (x-\mu)^2p\,dx=\sigma^2$ (fixed variance).

**Step 1. Build the Lagrangian.** One multiplier per constraint:
$$\mathcal L[p]=\int\Bigl[-p\log p-\lambda_0\,p-\lambda_1\,x\,p-\lambda_2\,(x-\mu)^2p\Bigr]dx.$$

**Step 2. Take the functional derivative.** The integrand has no $p'$, so the Euler–Lagrange equation has no derivative term and
reduces to $\partial F/\partial p=0$:
$$-\log p-1-\lambda_0-\lambda_1x-\lambda_2(x-\mu)^2=0
\;\Longrightarrow\;
p(x)=\exp\bigl(-1-\lambda_0-\lambda_1x-\lambda_2(x-\mu)^2\bigr).$$
The best $p$ is the exponential of a quadratic in $x$. That is already a bell shape; the constraints only fix the constants.

**Step 3. Fix the multipliers.**

- The exponent's peak sits at $x=\mu-\lambda_1/(2\lambda_2)$. The mean must be $\mu$, so $\lambda_1=0$.
- Then $p=C\,e^{-\lambda_2(x-\mu)^2}$, and a variance of $\sigma^2$ needs $\lambda_2=1/(2\sigma^2)$.
- Normalisation gives $C=1/\sqrt{2\pi\sigma^2}$.

$$p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr).$$

**Why this is the maximum and not just a stationary point.** Entropy is concave in $p$ (because $-p\log p$ is concave) and the
constraints are linear in $p$. For a concave functional with linear constraints, any stationary point is the global maximum, so
no second-variation check is needed.

### See it

The [interactive page](figures/interactive.html) draws the Gaussian next to a uniform, Laplace, logistic or two-bump density with the same
mean and spread. Mix the two with the slider $t$: the mixture $(1-t)\,\text{Gaussian}+t\,\text{alternative}$ still has that mean and
variance for every $t$, so every point on the path obeys the constraints, and the entropy curve along the path peaks at $t=0$, the Gaussian.

## Worked example: the shortest path on a sphere is a great circle

Take a sphere of radius $R$ with polar angle $\theta$ (from the north pole) and longitude $\varphi$. A tiny step on the surface has
length $ds^2=R^2(d\theta^2+\sin^2\theta\,d\varphi^2)$. Describe a path as $\varphi(\theta)$; its length is the functional
$$J[\varphi]=R\int\sqrt{1+\sin^2\theta\;\varphi'^2}\,d\theta,\qquad F=\sqrt{1+\sin^2\theta\,\varphi'^2}.$$

**Euler–Lagrange.** $F$ contains $\varphi'$ but not $\varphi$ itself, so $\partial F/\partial\varphi=0$ and the equation says
$\dfrac{d}{d\theta}\dfrac{\partial F}{\partial\varphi'}=0$. Therefore
$$\frac{\sin^2\theta\,\varphi'}{\sqrt{1+\sin^2\theta\,\varphi'^2}}=c\quad(\text{a constant}).$$
This constant exists because turning the sphere about its polar axis changes no lengths. A symmetry of the problem gives a conserved quantity (Noether's theorem in its simplest form).

**Solve.** Square it and solve for $\varphi'$:
$$\varphi'=\frac{c}{\sin\theta\,\sqrt{\sin^2\theta-c^2}}\;\Longrightarrow\;\cot\theta=A\cos(\varphi-\varphi_0),\qquad A=\frac{c}{\sqrt{1-c^2}}$$
(substitute $u=\cot\theta$ to integrate).

**Why that is a great circle.** With $x=R\sin\theta\cos\varphi,\;y=R\sin\theta\sin\varphi,\;z=R\cos\theta$ we have $\cot\theta=z/\sqrt{x^2+y^2}$, and the
answer becomes
$$z=A\,(x\cos\varphi_0+y\sin\varphi_0),$$
the equation of a plane through the centre. A plane through the centre cuts the sphere in a **great circle**.

**A shortcut by symmetry.** Rotate the sphere so both points lie on the equator. Then the equator is a path that obeys the equation (the case $c=1$). Any two points can be rotated onto a common great circle, so the answer is always a great-circle arc.

Details:

- Two points split their great circle into a short arc and a long arc. Both are stationary, but only the short one is the minimum.
- If the points are antipodal, every great circle through them has the same length $\pi R$, so the shortest path is not unique.
- The shortest paths on any curved surface are called **geodesics**, and the Euler–Lagrange equation for length is the **geodesic equation**.
  The same calculation with a different length rule gives the geodesics on the Gaussians in the
  [Fisher information](../fisher-information/index.html) notes.

### See it

The second plot on the [interactive page](figures/interactive.html) puts two points on a sphere. Push the great-circle arc sideways by $\varepsilon$ and the path gets longer
on either side, with the extra length growing as the bump grows. That is the nudge argument above, drawn on a globe.

## The pattern behind it: the exponential family

Nothing special happened for the Gaussian. Maximising entropy under constraints $\mathbb E[T_k(x)]=c_k$ always gives
$$p(x)\;\propto\;\exp\Bigl(-\sum_k\lambda_kT_k(x)\Bigr),$$
the **exponential family**. The Gaussian is $T=(x,x^2)$. With no constraint beyond normalisation on a bounded interval you get the
uniform density, and with a fixed mean on $x\ge 0$ you get the exponential density. This is why exponential families are the natural
"least assuming" models, and why they are the home ground of information geometry.

## Caveats

- Euler–Lagrange finds only **stationary** functions. Each still has to be checked as a minimum, maximum or saddle; the concavity
  argument above is the shortcut for the entropy problem.
- It assumes a best function exists and is smooth enough. Proving that is the job of the *direct methods* in the calculus of variations.
