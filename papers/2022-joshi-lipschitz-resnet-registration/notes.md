---
title: "Diffeomorphic Image Registration using Lipschitz Continuous Residual Networks"
category: "Medical image analysis"
subcategory: "Registration"
short_title: "Joshi 2022 — Lipschitz ResNet registration"
authors: "Ankita Joshi, Yi Hong"
venue: "MIDL (PMLR 172)"
year: 2022
url: "https://proceedings.mlr.press/v172/joshi22a.html"
pdf_url: "https://proceedings.mlr.press/v172/joshi22a/joshi22a.pdf"
tags: [image-registration, diffeomorphism, ode, resnet, lipschitz, spectral-normalisation, rkhs, lddmm]
status: read
summary: "ResNet blocks as Euler steps of a velocity-field ODE, with spectral normalisation for Lipschitz control; fewer folded voxels than VoxelMorph at slightly worse matching"
---

## Links

- **[PMLR 172:1–13](https://proceedings.mlr.press/v172/joshi22a.html)**: the paper (MIDL 2022, full paper).
- **Classical background**: LDDMM (Beg et al. 2005); stationary velocity fields (Arsigny et al. 2006).
- **Deep baseline**: VoxelMorph (Dalca et al. 2018).
- **Source of the ResNet-as-ODE link**: Rousseau et al. 2020 (ResNets as discretised diffeomorphic flows).
- **Lipschitz tool**: spectral normalisation (Miyato et al. 2018).
- **Related in this repo**: [Hilbert spaces](../hilbert-space/index.html) for what an RKHS is.

## In one paragraph

Registration warps one image onto another with a smooth, invertible map (a diffeomorphism), so no tissue is torn
or folded. The classical way to build such a map is to integrate a velocity field over time. This paper notes that a
residual network is exactly an Euler discretisation of an ODE, so a stack of ResNet blocks can *be* the
integrator, and makes each block "Lipschitz" with spectral normalisation so the map stays well behaved.

## The picture, intuition first

Imagine every pixel carried along by a river. A **velocity field** $v_t(x)$ says how fast and in which direction
the water moves at each point at time $t$. Following the flow from $t=0$ to $t=1$ moves each pixel to a new
place. If the water never tears and never crosses itself, the end result is a smooth, invertible warp, a
diffeomorphism.

In symbols (their Eq. 2 and 3):

- Time-varying flow: $\dfrac{d}{dt}\phi_t=v_t\circ\phi_t,\ \ \phi_0=\mathrm{id}$.
- Stationary flow ($v_t=V$ for all $t$): $\phi(t)=\exp(tV)$.

The first is what LDDMM uses; the second is what most deep unsupervised methods (including VoxelMorph) use, and it
is evaluated with an expensive "scaling and squaring" step that is why those methods often work at half resolution.

**The link to ResNets (Section 3).** A residual block computes $x_{l+1}=x_l+F(x_l,\theta_l)$. That is one forward Euler
step of an ODE with step size one: "current position plus a small displacement." So $k$ residual blocks in a row
integrate the flow in $k$ steps.

- Blocks with *different* weights give a **time-varying** velocity (NSVF-R2Net).
- Blocks with the *same* weights repeated give a **stationary** velocity, whose flow is $\exp(V)$ (SVF-R2Net).

## The architecture (Fig. 1)

1. A U-Net takes the two images and outputs a mean and variance of an initial velocity field (as in VoxelMorph's
   variational set-up), from which a velocity field is sampled.
2. $k=7$ ResNet blocks integrate it. Each block is conv, leaky ReLU, conv, tanh, then divide by 2, added to the input.
3. The result deforms the source image by interpolation.

**Lipschitz constraint.** Every convolution inside a block is spectrally normalised: its weights are divided
by their largest singular value, so the layer cannot stretch any input by more than a factor of 1. Intuition: a
function $F$ is Lipschitz with constant $L$ if $\lVert F(a)-F(b)\rVert\le L\lVert a-b\rVert$, so nearby
inputs give nearby outputs. The paper's argument is that this controls smoothness, which (via the
Cauchy–Lipschitz theorem on ODEs) makes the flow well defined, and that it makes the space of admissible velocities
an RKHS, the setting LDDMM works in.

**Loss.** Image similarity ($L_2$) plus a KL term pulling the velocity distribution towards a standard normal,
plus a penalty on the number of places where the Jacobian determinant of the warp is negative (a negative
determinant means the warp has locally folded space). Weights $[2,1,1]$.

## Results (OASIS brain MRI, 414 subjects, 100 test pairs)

| Method | RMSE ($\times10^{-3}$) | Folded voxels | Dice | Inference per pair | Training memory |
|---|---|---|---|---|---|
| SyN (classical) | 1.08 | 47.7 ± 145.1 | 0.67 | 10 min (CPU) | n/a |
| VoxelMorph | 1.10 | 51.4 ± 83.8 | 0.72 | 450 ms + 1440 ms | 8.6 GB |
| SVF-R2Net | 1.54 | 28.4 ± 17.6 | 0.69 | 900 ms | 7.9 GB |
| NSVF-R2Net | 1.34 | 27.2 ± 47.4 | 0.70 | 900 ms | 8.5 GB |

The authors describe it as comparable image matching with smoother deformations and efficient computation.

## Questions and doubts

- **A Lipschitz bound on the network is not the same as a smooth velocity field.** My reading: the spectral
  normalisation bounds how much the block's output can change when its *input function* changes. The theory
  the paper invokes (Cauchy–Lipschitz) needs the velocity $v_t(x)$ to be Lipschitz in the *spatial* variable $x$,
  meaning smooth across the image. The two are related for convolutional layers but not identical, and the paper
  does not show the step.
- **"Lipschitz implies an RKHS" is asserted, not proved.** A space of functions with bounded Lipschitz constant
  is not by itself a reproducing kernel Hilbert space. The paper cites a result about convolutional networks with
  homogeneous activations falling under an RKHS (Bietti and Mairal 2019), but how that yields the *admissible
  velocity space* LDDMM needs is left implicit.
- **The resulting maps are not provably folding-free.** A true diffeomorphism would have zero folded voxels, but
  the table shows about 27 to 28 on average, compared with about 48 and 51 for the baselines. So the claim is
  "fewer foldings", not "diffeomorphic", and the counts have large standard deviations.
- **Image matching is worse, not equal.** RMSE rises from 1.10 to 1.34 or 1.54 and Dice drops by 0.02 to 0.03 against
  VoxelMorph. "Comparable" sits on a trade-off against smoothness and the paper does not show that the
  smoother field is more useful.
- **Efficiency is modest.** Training memory is 7.9 and 8.5 GB against 8.6 GB. Inference is faster than the
  quoted two-part VoxelMorph time, but the comparison mixes two timings I could not check against the code.
- **Small evaluation.** 350 training pairs, 100 test pairs and one dataset, with no repeated training runs, so differences
  of 0.01 Dice cannot be separated from noise.
- **The KL term and the Jacobian penalty also regularise.** The paper has no ablation separating the effect of
  spectral normalisation from those, so the credit for smoothness cannot be assigned to the Lipschitz constraint.

## Takeaways

- A residual block is one Euler step; shared weights give a stationary flow and unshared weights give a
  time-varying one, which is a neat way to get both LDDMM-style and SVF-style registration from one block.
- Spectral normalisation is a cheap way to keep each step well behaved, though the gap from "network is
  Lipschitz" to "deformation is diffeomorphic" is not fully closed here.
- Fewer folded voxels came at the cost of somewhat worse matching.
