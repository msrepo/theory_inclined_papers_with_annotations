---
title: "The matrix exponential: following a linear flow"
authors: "Background notes"
venue: "Foundations"
tags: [linear-algebra, matrix-exponential, ode, flows, lie-group, background]
status: living
category: "Foundations"
subcategory: "Linear algebra"
short_title: "Matrix exponential"
---

## Links

- **[Interactive version](figures/interactive.html)**: pick a 2×2 matrix and slide time to watch the unit square
  travel along the flow; truncate the series term by term and see it converge; compare $e^{A+B}$ with
  $e^{A}e^{B}$ for commuting and non-commuting pairs.
- Related foundations pages:
  **[Gram matrix](../gram-matrix/index.html)**, **[Four fundamental subspaces](../four-fundamental-subspaces/index.html)**,
  **[Hilbert spaces](../hilbert-space/index.html)**.
- Where it shows up in the paper notes: the stationary velocity field $\exp(V)$ in
  [Lipschitz ResNet registration](../../papers/2022-joshi-lipschitz-resnet-registration/notes.md).

## In one paragraph

For a number $a$, $e^{a}=1+a+a^2/2!+\cdots$. Put a square matrix $A$ in the same series and you get the **matrix
exponential**
$$e^{A}=I+A+\frac{A^2}{2!}+\frac{A^3}{3!}+\cdots,$$
a matrix of the same size that always exists. Its meaning is geometric: $e^{tA}$ is the linear map that moves every point
forward for time $t$ along the flow whose velocity at $x$ is $Ax$.

## The picture, intuition first

Draw the plane and, at every point $x$, an arrow $Ax$. That is a **velocity field**: a point placed anywhere drifts along
the arrows. The path of a point is the solution of $\dfrac{dx}{dt}=Ax$. Because the arrows depend linearly on $x$, the point
reached at time $t$ is a linear function of the starting point:
$$x(t)=e^{tA}\,x(0).$$
So the matrix $e^{tA}$ is "the whole plane, carried along for time $t$". Section 1 of the interactive page shows exactly this:
the blue unit square is the start, the orange shape is its image under $e^{tA}$, and the dotted curves are the paths of its corners.

For a single number the same fact is $\dfrac{dx}{dt}=ax\Rightarrow x(t)=e^{at}x(0)$: growth at a rate proportional to the size.
The matrix case is that, with a direction-dependent rate.

## Where the series comes from

Guess $x(t)=x(0)+tAx(0)+\dots$ and fix it term by term. Starting at $x(0)$, a tiny time $h$ moves the point to about
$x(0)+hAx(0)=(I+hA)x(0)$. Repeating this $n$ times with $h=t/n$ gives $(I+tA/n)^n x(0)$, and as $n\to\infty$,
$$\Big(I+\frac{tA}{n}\Big)^n\to e^{tA}.$$
Expanding the binomial gives the series. So the series and "many tiny steps along the arrows" are the same statement,
which is also why a residual block $x\mapsto x+F(x)$ is an Euler step of a flow.

## Properties, and what they say about the flow

| Fact | Meaning |
|---|---|
| $e^{0}=I$ | zero time, nothing moves |
| $e^{(s+t)A}=e^{sA}e^{tA}$ | flowing for $s$ then $t$ equals flowing for $s+t$ |
| $(e^{A})^{-1}=e^{-A}$ | the flow can be run backwards, so $e^{A}$ is always invertible |
| $\det e^{A}=e^{\operatorname{tr}A}$ | area (volume) is scaled by $e^{t\,\mathrm{tr}A}$; trace 0 means area is preserved |
| $A^\top=-A\Rightarrow e^{A}$ is a rotation | antisymmetric $A$ gives a flow that turns without stretching |
| $A=PDP^{-1}\Rightarrow e^{A}=Pe^{D}P^{-1}$ | in the eigen-directions of $A$ each eigenvalue $\lambda$ becomes growth by $e^{\lambda t}$ |

Reading the eigenvalues of $A$ gives the behaviour: real positive means stretching away, real negative means squeezing in,
purely imaginary means rotation, complex with negative real part means spiralling in. The preset buttons on the interactive
page are one of each.

## When two matrices do not commute

$e^{A+B}=e^{A}e^{B}$ holds when $AB=BA$, and fails in general. The geometric reason: flowing along $A$ for a while then $B$
is not the same as flowing along $A+B$, because by the time you switch, the point has moved to where $B$'s arrows differ. To
first order the error is governed by the **commutator** $AB-BA$ (the Baker–Campbell–Hausdorff formula gives the
rest: $e^{A}e^{B}=e^{A+B+\frac12[A,B]+\cdots}$). Section 3 of the interactive page lets you see the gap open as the scale grows.

## Computing it

Summing the series directly is inaccurate when $A$ has large entries. The standard method is **scaling and squaring**: use
$e^{A}=(e^{A/2^k})^{2^k}$, compute $e^{A/2^k}$ from a few series terms (it is close to $I+A/2^k$), then square $k$ times
(`scipy.linalg.expm`). Section 2 of the interactive page shows how many terms are needed.

## Why it appears in the papers

- **Registration with stationary velocity fields**: the warp is $\exp(V)$, the flow of a fixed velocity field for unit
  time, which is a diffeomorphism because the flow is invertible (the table above, non-linear version).
- **Linear dynamics and continuous-time models** (Kalman filters, state-space layers): the transition over a time step $\Delta$ is $e^{\Delta A}$.
- **Lie groups**: the exponential maps a velocity (an element of the Lie algebra) to a group element such as a rotation.
