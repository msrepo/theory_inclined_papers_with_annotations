---
title: "Simplex-Aligned Diffusion with Cross-Granularity Interaction for Robust Medical Image Classification"
category: "Medical image analysis"
subcategory: "Uncertainty & annotator noise"
short_title: "Wu 2026 — simplex-aligned diffusion"
authors: "Chao Wu, Mingchen Gao"
venue: "MIDL (PMLR 315)"
year: 2026
url: "https://proceedings.mlr.press/v315/wu26a.html"
pdf_url: "https://raw.githubusercontent.com/mlresearch/v315/main/assets/wu26a/wu26a.pdf"
tags: [diffusion, generative-classification, calibration, robustness, aitchison-geometry, centred-log-ratio, simplex, label-smoothing, signal-to-noise]
status: read
summary: "a diffusion classifier that generates centred log-ratio (CLR) logits instead of one-hot vectors; the claimed 'geometric conflict' between Gaussian noise and the simplex is weak, because the CLR of a smoothed one-hot is an affine rescaling of the one-hot (centred and stretched about 5×), so the change mainly raises the targets' signal-to-noise ratio and moves calibration into a softmax temperature; clean accuracy is a tie, and the robustness evidence rests on one baseline and noisy corruption tests"
related: [midl2026-overview, 2026-lux-gradient-surgery, optimal-transport, fid-geometry]
ext_links: [code|https://github.com/SamaritanW/simplex-aligned-diffusion]
---

## Links

- **[PMLR 315:121–152](https://proceedings.mlr.press/v315/wu26a.html)**: the paper (MIDL 2026, full paper), read in its camera-ready form.
- **Code**: [SamaritanW/simplex-aligned-diffusion](https://github.com/SamaritanW/simplex-aligned-diffusion).
- **Baselines it builds on**: CARD (Han et al., 2022) and DiffMIC / DiffMIC-v2 (Yang et al., 2023, 2025), which generate one-hot label vectors by diffusion.
- **Related in this repo**: [Lux et al. 2026](../2026-lux-gradient-surgery/index.html), another MIDL 2026 calibration paper, which locates calibration in the gradient rather than in geometry; the [MIDL 2026 overview](../midl2026-overview/index.html), where this is the second theory pick.

These notes came out of a question-and-answer session; the questions are kept as section headings.

## In one paragraph

Some classifiers treat "predict the label" as a diffusion problem: start from noise and denoise
towards the class vector, conditioned on the image. DiffMIC-v2 does this with one-hot vectors, and
this paper argues that adding unbounded Gaussian noise to vectors that should stay on the
probability simplex is a geometric mismatch that hurts calibration. Its fix maps each one-hot label to
centred log-ratio (CLR) logits, runs diffusion there, and maps back with a softmax. It also adds a
Transformer layer fusing global and local image features. Clean accuracy matches DiffMIC-v2; the
claims are better calibration and robustness under image corruption. Read closely, the CLR targets
are an affine rescaling of the one-hot targets, so the change is better described as raising the
targets' signal-to-noise ratio and decoding through a softmax temperature than as fixing a geometry.

## Q1. What does the paper do?

**Background: classification as label generation.** Take the true label as a vector, for example
$y=(0,0,1,0,0)$ for class 3 of 5. The forward process blurs it with noise,
$y_t=\sqrt{\bar\alpha_t}\,y+\sqrt{1-\bar\alpha_t}\,\epsilon$, until it is pure noise. A U-Net learns
to predict the noise, conditioned on image features. At inference it denoises from noise to a label
vector. The appeal is a *distribution* over labels rather than one point estimate.

**The claimed problem: two propositions (Appendix F).**

| Claim | What they argue | A closer look |
|---|---|---|
| Prop. 1, "probability leakage" | Noisy one-hot vectors leave the simplex; the Gaussian posterior puts mass on invalid values, so its mean is "biased" relative to a truncated Gaussian, pulling predictions off the vertices. | Intermediate diffusion states are *meant* to leave the data space; noisy images leave the valid pixel range too. $q(y_{t-1}\mid y_t,y_0)$ is the *exact* posterior of the forward process, not an approximation, and only the final output needs decoding. The truncated-Gaussian "bias" corrects a target DDPM never uses. |
| Prop. 2, "approximation gap" | Projecting onto the simplex (e.g. a softmax) at every step breaks the closed-form Gaussian marginals, so training uses a biased proxy (a Jensen-type gap). | True, but no baseline does this; CARD and DiffMIC do not project at each step. |

**The fix: CLR logits.**

$$
z_0=\frac1\lambda\Big(\log y_{\text{smooth}}-\tfrac1C\textstyle\sum_k\log y^{(k)}_{\text{smooth}}\Big),
$$

with label smoothing $\epsilon=0.001$ to avoid $\log 0$ and $\lambda=1.5$. Diffusion runs on $z$; at
inference the probabilities are $\text{softmax}(\lambda\hat z_0)$.

**The other components.** A cross-granularity interaction layer (one Transformer encoder layer over
the global token and $K$ salient-patch tokens) whose output conditions the denoising U-Net through
channel gating; a focal reweighting of the diffusion loss, $1+\alpha(1-p_y)^\gamma$, from the fused
prior; and a spatial guidance map inherited from DiffMIC-v2. So three things change relative to
DiffMIC-v2 at once.

## Q2. What is the geometry behind CLR, intuitively?

CLR comes from **compositional data analysis** (Aitchison geometry), where what matters about a
probability vector is the *ratios* between its parts. The simplex has its own vector-space structure
under this view, and CLR maps it isometrically onto the sum-zero hyperplane of $\mathbb R^C$.

<figure>
<img src="figures/simplex-to-clr.svg" alt="Left: the triangle of three-class probability vectors with two families of constant-ratio lines fanning out from the corners and crowding near them, a curved shortest path in Aitchison geometry next to a dashed straight segment, and three smoothed class-3 labels stacked on the class-3 corner. Right: the same objects after the centred log-ratio map, where the triangle becomes a flat plane centred on the uniform vector, both line families become evenly spaced parallel lines, the curved path becomes straight, and the three smoothed labels separate along the class-3 direction.">
<figcaption><b>The simplex and its CLR image, for three classes.</b> <b>Left:</b> blue lines hold
$p_1/p_2$ fixed (all through the class-3 corner) and purple lines hold $p_2/p_3$ fixed (all through
the class-1 corner). Equal steps in log-ratio crowd together near the corners, where a small shift in
position is a large change in ratio. The green curve is the shortest path in ratio geometry; it bends
away from the dashed straight segment. The smoothed labels for $\epsilon=10^{-2},10^{-3},10^{-4}$ all
sit on the class-3 corner. <b>Right:</b> after CLR the triangle is an entire flat plane with the
uniform vector at the origin. Both line families become evenly spaced parallel lines, the curved path
becomes straight, and the corners have no image (they contain $\log 0$). The smoothed labels separate
along the class-3 direction, further out the smaller $\epsilon$ is, by about $\log(1/\epsilon)$; a true
one-hot is infinitely far away. Computed from the maps, not sketched.</figcaption>
</figure>

What the picture says:

- **CLR straightens ratio geometry.** Distances and shortest paths measured in log-ratios become
  ordinary Euclidean distances and straight lines. This is the legitimate, foundational idea in the paper.
- **One-hot labels are where it breaks.** The corners of the simplex have no CLR image, so the
  paper must smooth the labels, and the smoothing level $\epsilon$ decides how far out the targets sit.
- **With $C$ classes** the CLR space is a $(C-1)$-dimensional flat space (4-D and 6-D for the paper's
  5 and 7 classes); the same rules hold.

**The key observation** (a derivation made in these notes, not in the paper). A smoothed one-hot vector
has only two distinct values, $u$ for the true class and $v$ for the rest, so

$$
\text{CLR}(y)=\log\!\frac uv\,\Big(e_k-\tfrac1C\mathbf 1\Big),\qquad
z_0=\frac{\log(u/v)}{\lambda}\Big(e_k-\tfrac1C\mathbf 1\Big).
$$

That is an **affine function of the one-hot vector**: subtract the uniform vector and multiply by a
constant. With $\epsilon=0.001$ and five classes, $\log(u/v)\approx8$, so the constant is about 5 after
dividing by $\lambda$. The class targets are still $C$ fixed points, now on a hyperplane rather than at
the simplex's corners; in the right panel, they are the dots on the class directions.

What that changes in practice:

1. **Signal-to-noise ratio.** Targets about 5× larger against the same unit-variance noise raise the
   signal-to-noise ratio about 25-fold at every step, which effectively changes the noise schedule:
   classes stay distinguishable for more of the reverse process.
2. **Decoding.** Probabilities now come from a softmax over generated logits with temperature
   $\lambda$, so calibration depends directly on $\lambda$ and on the spread of the generated logits.
3. **The "manifold" is not preserved.** Isotropic noise in $\mathbb R^C$ also leaves the sum-zero
   hyperplane; this is harmless only because softmax ignores a constant shift. (The text's forward
   equation also drops the $\sqrt{1-\bar\alpha_t}$ factor on the noise, probably a typo.)

**The paper's sensitivity table supports this reading** (Appendix H, Table 8).

| Setting | HAM10000 accuracy | APTOS accuracy |
|---|---|---|
| $\lambda$ = 1.0 / 1.5 / 2.0 | 0.876 / **0.894** / 0.886 | 0.818 / **0.848** / 0.842 |
| $\epsilon$ = 10⁻² / 10⁻³ / 10⁻⁴ | 0.882 / **0.894** / 0.873 | 0.833 / **0.848** / 0.844 |

Both settings affect accuracy only through the target scale $\log(u/v)/\lambda$, since the argmax at
decoding ignores $\lambda$. Accuracy moving by up to 3 points says the target scale does much of the
work. One-hot diffusion with targets centred and multiplied by 5 would test this directly.

## Q3. What is the main experiment? Classification accuracy?

**No.** Clean accuracy is roughly a tie; the headline claim is robustness and calibration under image
corruption, tested against one baseline, DiffMIC-v2.

| Experiment | What it measures | What it shows |
|---|---|---|
| 1. Clean benchmark (Table 1; 3 seeds in Table 9) | accuracy and macro-F1 on APTOS2019 (5 retinopathy grades) and HAM10000 (7 skin-lesion classes), against 9 methods | about a tie: accuracy +1 point (HAM 0.893 vs 0.883; APTOS 0.848 vs 0.839), F1 level or slightly lower (APTOS 0.666 vs 0.669) |
| 2. Gaussian-noise stress test (Figure 2) | κ, accuracy and ECE as added noise grows (σ 0.05 to 0.30), against DiffMIC-v2 only | the headline: on APTOS at σ = 0.3 DiffMIC-v2's κ falls to about 0.02 while theirs stays at 0.42; on HAM their ECE stays below 0.07 while DiffMIC-v2's reaches 0.17 |
| 3. "Clinical artifact" benchmark (Table 2) | accuracy, F1, κ, ECE under shot noise and motion blur (fundus) and defocus and motion blur (dermoscopy), severities 1, 3, 5, against DiffMIC-v2 only | mostly better but noisy; e.g. APTOS motion blur at severity 1, accuracy 0.811 vs 0.636 |
| 4. Ablation (Table 3) | clean and noisy (σ = 0.1) accuracy, adding each component | on APTOS the interaction module alone helps more than the logit diffusion alone (0.714 vs 0.662 noisy accuracy) |
| 5. Grad-CAM (Figure 3) | where the network looks | qualitative only |
| 6. Cost (Section 4.5) | runtime overhead | practicality only |

**How strong the evidence is.**

- **One comparison.** Every robustness and calibration result is against DiffMIC-v2; there is no
  ordinary classifier (ResNet, ViT) under the same corruptions and no temperature-scaling baseline. It
  shows "better than one diffusion classifier", not "a robust, calibrated classifier".
- **No calibration on clean data.** The paper says the one-hot mismatch causes overconfidence, which
  would show on clean images too, yet ECE appears only under corruption; NLL and reliability diagrams
  are missing.
- **High variance.** DiffMIC-v2's shot-noise accuracy on APTOS goes 0.140 → 0.499 → 0.324 as severity
  rises, and with 3 seeds their κ at severity 5 is 0.194 ± 0.233.
- **Several changes at once**, and the ablation does not favour the geometric story on APTOS.

**What the main experiment should have been.**

1. Clean-data calibration (ECE, NLL, reliability diagrams) for one-hot versus logit diffusion with
   everything else identical.
2. The rescaling control: one-hot diffusion with targets centred and scaled ×5.
3. Temperature scaling on DiffMIC-v2 and on a standard classifier.

## Related work not cited

Representing a category as a scaled logit vector and diffusing it is established in diffusion
language models: SSD-LM (Han, Kumar & Tsvetkov, ACL 2023) uses "almost-one-hot" logit vectors of
$\pm K$, and TESS (Mahabadi et al., EACL 2024) does the same. Principled diffusion on the simplex also
exists: *Categorical SDEs with Simplex Diffusion* (Richemond, Dieleman & Doucet, 2022), the
*Dirichlet Diffusion Score Model* (Avdeyev et al., ICML 2023) and *Dirichlet Flow Matching* (Stark et
al., ICML 2024). The claim to be first is plausible only within medical classification.

## Questions and doubts

- Is the "geometric conflict" real, given that intermediate diffusion states never need to be valid
  probabilities? Or is the gain the centring and scaling of the targets, as the $\lambda$ and
  $\epsilon$ sensitivity suggests?
- Where does calibration in a generative classifier come from: the geometry, the decoding temperature,
  or the spread of the samples? The paper does not separate these.
- How do the corruption results hold up with more seeds and with standard classifiers in the comparison?

## Takeaways

- The real mathematics is Aitchison geometry: CLR straightens ratio geometry, and one-hot corners are
  degenerate, which forces label smoothing.
- For smoothed one-hot labels, CLR is an affine map: the targets are the centred one-hot directions
  stretched by $\log(u/v)/\lambda$, about 5×. That raises the signal-to-noise ratio of label diffusion
  and moves calibration into a softmax temperature.
- Clean accuracy is a tie; the robustness and calibration claims rest on one baseline and noisy
  corruption tests, with no clean-data calibration or temperature-scaling comparison.
