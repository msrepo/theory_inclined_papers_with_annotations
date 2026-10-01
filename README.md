# Theory-inclined papers, with annotations

**Rendered notes: <https://msrepo.github.io/theory_inclined_papers_with_annotations/>**

Close readings of theory-heavy machine learning papers. One folder per paper, each with a
plain-language annotation in Markdown (LaTeX maths included) that can be rendered to a
styled HTML page with `make`.

The aim of each annotation is not a summary. It is to reconstruct the argument so that the
steps a first reading skips over — why a definition is stated the way it is, where an
inequality comes from, what an assumption is buying — are written down explicitly.

## Layout

```
.
├── Makefile                  build entry point
├── papers/
│   └── <slug>/
│       ├── notes.md          the annotation; YAML front matter + body
│       ├── figures/          optional; self-contained themed SVGs
│       ├── code/             optional; runnable checks, see `make verify`
│       └── paper.pdf         gitignored, fetched or dropped in by hand
├── foundations/
│   └── <slug>/notes.md       background maths the annotations lean on
├── topics/
│   └── <slug>/notes.md       themes spanning several papers
├── presentation/             slide decks (Marp); not built into the site
│   └── <slug>/               slides.md, figures/, code/
├── tools/
│   ├── build.py              notes.md -> build/<slug>/index.html, plus the index page
│   ├── new_paper.py          scaffolds a new papers/<slug>/notes.md
│   ├── fetch_pdfs.py         downloads PDFs listed in front matter
│   └── style.css             shared stylesheet (light and dark)
└── build/                    gitignored render output
```

Slugs are `<year>-<first-author-surname>-<short-topic>`, for example
`2026-betser-infonce-gaussian`. That keeps the folder listing chronological-ish and
readable without needing an index.

**PDFs are never committed.** Each `notes.md` records a `pdf_url` in its front matter and
`make fetch` pulls anything missing, so a fresh clone stays small and no publisher PDF is
redistributed. Dropping a file in as `papers/<slug>/paper.pdf` by hand works equally well.

## Requirements

- [`pandoc`](https://pandoc.org) — `brew install pandoc`
- Python 3.9+ (standard library only; no packages to install)

Maths is rendered with KaTeX loaded from a CDN, so the built pages need a network
connection the first time they are opened.

## Publishing

Every push to `main` triggers `.github/workflows/pages.yml`, which runs `make` and deploys `build/`
to GitHub Pages, except a push that only touches `README.md`, `.gitignore` or `presentation/`.

The workflow deliberately does **not** run `make fetch`, so no paper PDF is ever published
to the site. A paper whose front matter carries a `pdf_url` gets a link out to the
publisher instead; locally, `make fetch` still puts `paper.pdf` beside the notes and the
local render links to it directly.

## Usage

```sh
make                              # render everything into build/
make open                         # render, then open build/index.html
make serve                        # render, then serve on http://localhost:8000
make new SLUG=2026-lastname-topic # scaffold a new paper folder
make fetch                        # download any missing PDFs
make list                         # list the papers in the repo
make check                        # verify pandoc is present and notes parse
make verify                       # run every papers/*/code/*.py
make clean                        # remove build/
```

`make` is incremental: it only re-runs when a `notes.md`, the build script or the
stylesheet has changed.

## Adding a paper

```sh
make new SLUG=2027-lastname-topic
```

Every page declares a `category` (and optionally a `subcategory`) in its front matter, which
drives the sidebar shown on every page. The taxonomy and its display order live in `CATEGORIES`
in `tools/build.py` — the one place to edit when adding a section. A `short_title` keeps the
sidebar readable when the real title is long. Anything uncategorised lands under *Unsorted*
rather than vanishing.

Pages that are not about one paper live under `foundations/<slug>/notes.md` (background maths)
or `topics/<slug>/notes.md` (themes spanning several papers), with the same front matter minus
`year`, `url` and `pdf_url`. All roots render into a flat `build/<slug>/`, so slugs must be
unique across them and cross-links are always `../<slug>/index.html`. To add another
collection, append it to `COLLECTIONS` in `tools/build.py` and `ROOTS` in the `Makefile`.

Then fill in the front matter at the top of the new `notes.md`:

```yaml
---
title: "Paper title"
authors: "First Author, Second Author"
venue: "NeurIPS"
year: 2027
url: "https://openreview.net/forum?id=..."
pdf_url: "https://openreview.net/pdf?id=..."
tags: [optimisation, generalisation, theory]
status: reading
---
```

`status` is free text; `read` and `reading` get distinct colours on the index page.
`tags` is an inline list. The parser in `tools/build.py` deliberately supports only
scalars and inline lists — anything more structured belongs in the body.

Then `make fetch` to pull the PDF, and `make` to render.

A note on `pdf_url`: prefer an arXiv PDF link over an OpenReview one. OpenReview's
`/pdf?id=...` endpoint refuses non-browser requests with a 403, so `make fetch` cannot
retrieve it; arXiv serves it without complaint. Keep the venue page in `url` and the
fetchable PDF in `pdf_url` — they do not have to point at the same host.

It is also worth opening the notes with a short `## Links` section listing the venue page,
the preprint and any project page, so the Markdown source is useful on its own and not
only once rendered.

### Writing the body

Plain Markdown, plus:

- inline maths between single dollars: `$\eta_2 = \rho_m^2(X, X_0)$`
- display maths between double dollars on their own lines
- tables for symbol glossaries, which are worth including early in every annotation

### Code

Where a paper makes a claim worth checking, `papers/<slug>/code/` holds a small self-contained
script that checks it, with a `__main__` printing exactly the numbers quoted in the notes.
`make verify` runs them all, so a claim in the prose cannot quietly drift from the code that
produced it. Standard library and numpy only — these are meant to be read and poked at, not
reproduced at scale.

A loose section order that has worked so far: *In one paragraph* → *The spine of the
argument* → *Setup and notation* → the results, in the paper's own order → *Questions and
doubts* → *Takeaways*. The doubts section is the point of the exercise; it should not be
left empty.

## Foundations

Background pages that several annotations lean on, kept separate so they are not buried
inside one paper's notes.

| Page | Covers |
|---|---|
| *Inequalities and concentration: a working toolbox* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/inequalities-and-concentration/) · [source](foundations/inequalities-and-concentration/notes.md) | Cheeger and higher-order Cheeger, Markov/Chebyshev/Hoeffding, McDiarmid, Rademacher, KL chain rule, Donsker–Varadhan, HGR, Weyl, Davis–Kahan, Wigner, spherical CLT |
| *Linear (Fisher) Discriminant Analysis* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/lda-fisher-discriminant/) · [source](foundations/lda-fisher-discriminant/notes.md) | between/within-class scatter matrices, Fisher's criterion as a generalised eigenproblem $S_w^{-1}S_b$, the $g-1$ rank cap, and the PCA+LDA fix for small-sample singular $S_w$ |

**Linear algebra**

| Page | Covers |
|---|---|
| *Eckart–Young–Mirsky and low-rank approximation via truncated SVD* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/eckart-young-lowrank-svd/) · [source](foundations/eckart-young-lowrank-svd/notes.md) | Frobenius and spectral matrix norms, the Eckart–Young–Mirsky theorem (truncated SVD is the provably-best rank-$k$ approximation in both norms), the 2-norm proof, and an image-compression example |
| *Column space, null space and residuals: the four fundamental subspaces* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/four-fundamental-subspaces/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/four-fundamental-subspaces/figures/interactive.html) · [source](foundations/four-fundamental-subspaces/notes.md) · [code](foundations/four-fundamental-subspaces/code/) | what a matrix can reach (column space) and cannot see (null space), their perpendicular partners, least squares as a right angle with residuals in $N(A^\top)$, the hat matrix and $n-p$ residual degrees of freedom, centring as a residual, the pseudo-inverse and why gradient descent from zero finds the minimum-norm solution, the SVD bases, PCA residual subspaces for OOD, and where each one does work in H-score, LogME, LDA/SFDA, the NTK, spectral contrastive learning and the MICCAI notes |
| *The Gram matrix: a table of dot products* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/gram-matrix/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/gram-matrix/figures/interactive.html) · [source](foundations/gram-matrix/notes.md) · [code](foundations/gram-matrix/code/) | $G=XX^\top$ as lengths, angles and distances; rotation invariance and rebuilding points from $G$ (MDS); why it is always PSD and its rank; the dual view $XX^\top$ vs. $X^\top X$ and centring $HGH$; kernel Gram matrices and RBF bandwidth; the NTK as a Gram matrix of gradients; the InfoNCE logit matrix; where each appears in the paper notes; and why forming it squares the condition number |
| *HSIC: the Hilbert–Schmidt independence criterion* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/hsic/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/hsic/figures/interactive.html) · [source](foundations/hsic/notes.md) · [code](foundations/hsic/code/) | why uncorrelated is not independent (parabola, circle, funnel); squared covariance and $\lVert C\rVert_F^2$ as agreement of two centred Gram matrices; more features catch more dependence, and the Gaussian kernel as every polynomial degree at once; the estimator $\frac{1}{n^2}\operatorname{tr}(KHLH)$ symbol by symbol with a five-point example that honestly fails; the population form and HSIC as MMD between the joint and the product of marginals; bandwidth limits (too wide is linear, too narrow is blind, a checkerboard the median heuristic misses); the permutation test and the $1/n$ bias; CKA and its eigenvalue weighting; COCO, HGR and the H-score as relatives |
| *Principal angles between subspaces* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/principal-angles-subspaces/) · [source](foundations/principal-angles-subspaces/notes.md) · [code](foundations/principal-angles-subspaces/code/) | the SVD characterisation, the projector and distance identities, why `arccos` destroys small angles and the sine fix, and the equivalence with canonical correlations |

**Probability**

| Page | Covers |
|---|---|
| *Optimal transport and the Wasserstein distance* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/optimal-transport/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/optimal-transport/figures/interactive.html) · [source](foundations/optimal-transport/notes.md) | Monge maps vs. Kantorovich couplings, pushforward in plain words, $W_p$ and why it beats KL/JS without overlap (WGAN), Sinkhorn, duality, 1-D and Gaussian (FID) closed forms, minibatch OT for flow matching, and a symbol-by-symbol reading of the discrete OT definition used in data selection |
| *Integral probability metrics: a panel of judges (TV, Wasserstein-1, Dudley, MMD)* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/integral-probability-metrics/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/integral-probability-metrics/figures/interactive.html) · [source](foundations/integral-probability-metrics/notes.md) · [code](foundations/integral-probability-metrics/code/) | a distance as the best gap any judge in a panel can produce between two average scores; judges scored in $[0,1]$ give total variation, 1-Lipschitz judges give $W_1$, judges with height plus slope at most 1 give the Dudley (bounded-Lipschitz) metric, a kernel's unit ball gives MMD; closed forms for two point masses (TV = 1, $W_1 = D$, Dudley $= 2D/(D+2)$, saturating at 2) and for shifted Gaussians; the best ramp judge and why it is optimal for a single hump, cross-checked by an assignment solver; Dudley is not $W_1$ for close distributions unless they are narrow; why bounded panels cannot be "arbitrarily far"; IPMs against $\varphi$-divergences; what a supremum over a panel does not tell you about a downstream task |
| *Expectation–Maximization (EM) and Gaussian mixtures* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/expectation-maximization/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/expectation-maximization/figures/interactive.html) · [source](foundations/expectation-maximization/notes.md) · [code](foundations/expectation-maximization/code/) | the chicken-and-egg of latent labels, responsibilities by Bayes' rule (a softmax, and a sigmoid for two blobs), weighted M-step refits with a worked six-point example, why the likelihood never drops (Jensen, a touching lower bound, the $\ln p = \mathcal L + \mathrm{KL}$ gap), k-means as the hard limit, and the step to the ELBO of VAEs |
| *Langevin dynamics: gradient descent plus noise* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/langevin-dynamics/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/langevin-dynamics/figures/interactive.html) · [source](foundations/langevin-dynamics/notes.md) · [code](foundations/langevin-dynamics/code/) | why a kick in time dt has size √dt (Brownian motion spreads like √t), Euler–Maruyama as gradient descent plus noise, the Ornstein–Uhlenbeck bowl and equipartition (kD/2 of excess loss), the Gibbs law e^(−U/D) from a zero probability current, depth against width (the free energy U + (D/2) log det H), the Fokker–Planck equation and its two time scales, Kramers' law (the barrier sets the exponent, not the destination), detailed balance and two ways to break it (a rotating drift; Hessian-shaped noise gives a round cloud), the path weight (downhill free, uphill e^(−ΔU/D)), ULA's 1/(1 − η/2) bias against MALA, SGLD's extra temperature, SGD's D ≈ ησ²/2B, and annealed Langevin fixing mode weights in score-based sampling |

**Information geometry**

| Page | Covers |
|---|---|
| *The Fisher information matrix: how sharply data pin down parameters* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/fisher-information/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/fisher-information/figures/interactive.html) · [source](foundations/fisher-information/notes.md) · [code](foundations/fisher-information/code/) | the score and its variance on a coin (why a fair coin is least informative), Cramér–Rao and efficiency (the median's 2/π), the Fisher matrix as an inverted uncertainty ellipse and blind directions, the local KL expansion, reparameterisation and the Fisher–Rao distance, the Gaussians as a hyperbolic half-plane (geodesics widen, slide, narrow; contrasted with optimal transport), natural gradient and Fisher scoring, Hessian against Fisher against empirical Fisher, the Fisher and the NTK sharing eigenvalues, and how it is estimated in practice |

## Topics

Themes that cut across several papers.

| Page | Covers |
|---|---|
| *Applications of Gaussianity: uncertainty, dense prediction, test-time adaptation* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/gaussianity-in-practice/) · [source](topics/gaussianity-in-practice/notes.md) | the applied literature that assumes CLIP features are Gaussian, with close reads of Zhou et al. 2025 (CVPR), Venkataramanan et al. 2025 (UAI) and C. Huang et al. 2024 (IJCAI) |
| *MICCAI 2026: domain adaptation and generalisation — the mathematical constructs* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/miccai2026-domain-adaptation/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/miccai2026-domain-adaptation/figures/interactive.html) · [source](topics/miccai2026-domain-adaptation/notes.md) | the core mathematical construct of each of 26 MICCAI 2026 DA/DG/TTA/OOD papers (Mahalanobis residuals, deep EM, OT and Schrödinger bridges, flow-matching TTA, InfoMax, variational logit energies, GRPO), with a construct map, one toy-computed figure per paper, and a *What to watch* per paper |
| *MICCAI 2026: image quality assessment — the mathematical constructs* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/miccai2026-image-quality-assessment/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/miccai2026-image-quality-assessment/figures/interactive.html) · [source](topics/miccai2026-image-quality-assessment/notes.md) | the core mathematical construct of each of 11 MICCAI 2026 papers tagged image quality assessment (cross-sectional self-consistency for label noise, prototype-guided flow matching, few-shot prototypical networks with FiLM and gradient reversal, a failure-driven MLLM data engine, loss-GMM pseudo-label filtering, generalised Lehmer pooling, Dice under class imbalance, set encoders for missing metadata, non-negative PU learning, masked autoencoders with LoRA, CORN and concept MIL), with a what-is-graded table, a construct map, one interactive widget per paper, and a *What to watch* per paper |

## Presentations

Slide decks built from the notes, kept apart from the site: `tools/build.py` does not read `presentation/`, so
nothing here is rendered to HTML, and a push that only touches it does not run the Pages workflow. Each deck is a
[Marp](https://marp.app) Markdown file (`slides.md`; slides separated by `---`, HTML comments are presenter notes).
Preview it with the Marp extension for VS Code, or export it with `npx @marp-team/marp-cli slides.md`.

| Deck | Covers |
|---|---|
| *Transferability estimation: the problem, the scores, the gaps, the open problems* — [source](presentation/transferability-talk/slides.md) · [code](presentation/transferability-talk/code/) | the transferability literature as one talk: the problem in symbols with its labelled and unlabelled variants, then one slide each for NCE → LEEP → H-score → LogME → SFDA → PAS, the two medical-imaging benchmarks (Chaves 2023, Claßen 2026) and what they leave open, and seven open problems |

## Papers

| Year | Paper | Notes | Topic |
|---|---|---|---|
| 2026 | Lyu, Zhou & Zhong, *A Statistical Theory of Overfitting for Imbalanced Classification*, ICLR — [arXiv:2502.11323](https://arxiv.org/abs/2502.11323) · [code](https://github.com/jlyu55/Imbalanced_Classification) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-lyu-imbalanced-overfitting/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-lyu-imbalanced-overfitting/figures/interactive.html) · [source](papers/2026-lyu-imbalanced-overfitting/notes.md) · [code](papers/2026-lyu-imbalanced-overfitting/code/) | max-margin linear probes on an imbalanced Gaussian mixture with n/d → δ: training logits are the test Gaussian truncated at the margin, because the d free directions can lift the n training logits by a mean square of (1−ρ²)/δ and truncation is the cheapest way to spend it (Gordon/CGMT); both classes get the same total lift, so each minority point is lifted (1−π)/π times further and the intercept goes negative; rebalancing moves only the intercept and τ_opt has a closed form, but it is negative in several of the paper's own settings, and the π-trend in calibration is almost entirely the prior log-odds that rebalancing discards |
| 2026 | von Berg, Fono, Datres, Maskey & Kutyniok, *The Price of Robustness: Stable Classifiers Need Overparameterization*, ICLR — [OpenReview](https://openreview.net/forum?id=YnpiyoklHP) · [arXiv:2603.02806](https://arxiv.org/abs/2603.02806) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-vonberg-price-of-robustness/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-vonberg-price-of-robustness/figures/interactive.html) · [source](papers/2026-vonberg-price-of-robustness/notes.md) · [code](papers/2026-vonberg-price-of-robustness/code/) | a law of robustness for classifiers: swap the Lipschitz constant for the mean distance to the decision boundary, and memorisers need $p\gtrsim nd$ parameters for margins above the free $\sqrt{c/d}$; the ramp surrogate carries every proof, but Corollary 15 drops a covering term and loses its dependence on $p$, and the experiments never cross $p = nd$ on CIFAR-10 |
| 2026 | Nikolentzos & Skianis, *On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets*, ICLR — [arXiv:2505.24403](https://arxiv.org/abs/2505.24403) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-nikolentzos-set-lipschitz/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-nikolentzos-set-lipschitz/figures/interactive.html) · [source](papers/2026-nikolentzos-set-lipschitz/notes.md) · [code](papers/2026-nikolentzos-set-lipschitz/code/) | mean, sum and max are 1-, 1- and √d-Lipschitz for EMD, matching and Hausdorff respectively, and each distance's blind spot (duplication, multiplicity, zero vectors) explains five of the six failures at distance zero; the sixth, max against EMD, holds only for unbounded set sizes (constant in [M, √d·M] under a cap), attention fails EMD only on unbounded inputs, the NN_max bound needs the hidden width, and Proposition 3.6 is an equality |
| 2026 | Karczewski, Heinonen, Pouplin, Hauberg & Garg, *The Spacetime of Diffusion Models: An Information Geometry Perspective*, ICLR (Oral) — [arXiv:2505.17517](https://arxiv.org/abs/2505.17517) · [code](https://github.com/Aalto-QuML/spacetime-geometry) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-karczewski-spacetime-diffusion/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-karczewski-spacetime-diffusion/figures/interactive.html) · [source](papers/2026-karczewski-spacetime-diffusion/notes.md) · [code](papers/2026-karczewski-spacetime-diffusion/code/) | Fisher–Rao geometry of the denoising posteriors p(x₀ | x_t) over spacetime (x_t, t): an exponential family whose chord (Δη)ᵀ(Δμ) is exactly twice a symmetrised KL, so geodesic energies need only the denoiser and one JVP; spacetime is the natural-parameter space (schedule-free), and for Gaussian data the hyperbolic half-plane, which makes DiffED a log of Euclidean distance that saturates; and in a transition-path toy the geodesic reweights wells rather than transporting mass, so its noise schedule, not its route, does the work |
| 2026 | Iwasaki, Bloch, Lee & Ghaffari, *The Score Kalman Filter* — [arXiv:2605.16644](https://arxiv.org/abs/2605.16644) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-iwasaki-score-kalman-filter/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-iwasaki-score-kalman-filter/figures/interactive.html) · [source](papers/2026-iwasaki-score-kalman-filter/notes.md) · [code](papers/2026-iwasaki-score-kalman-filter/code/) | moment filtering without the partition function: score matching fits an exp-polynomial belief in one linear solve, Stein's identity closes the moment hierarchy; but the benchmark truth is noise-free, which rewards ignoring the data, and the update as written leaves the mean at the prior |
| 2026 | Diniz, de Faria Júnior & Ester, *PAS: Estimating the target accuracy before domain adaptation*, ICLR — [arXiv:2604.09863](https://arxiv.org/abs/2604.09863) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-diniz-pas/) · [source](papers/2026-diniz-pas/notes.md) · [code](papers/2026-diniz-pas/code/) | scoring a source domain with no target labels at all, via a nearest-centroid margin |
| 2026 | Claßen, Sourget, Juodelyte, van der Goot & Cheplygina, *Robustness of transferability estimation metrics for medical imaging* — [arXiv:2608.09999](https://arxiv.org/abs/2608.09999) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-classen-te-robustness/) · [source](papers/2026-classen-te-robustness/notes.md) · [code](papers/2026-classen-te-robustness/code/) | whether transferability rankings survive a change of random seed; mostly they do not |
| 2026 | Wang, Liu, Liao, Fan, Du, Tang, Wang & Wang, *Mining Useful General Data for Low-Resource Domain Adaptation*, ICML — [repository](https://github.com/applewpj/NTK-Selector) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-wang-ntk-selector/) · [source](papers/2026-wang-ntk-selector/notes.md) · [code](papers/2026-wang-ntk-selector/code/) | mining general-domain data by NTK alignment with a tiny domain set |
| 2026 | Li, Jiang, Ye, He, Li, Xiao, Cheng & Chen, *Path-Decoupled Hyperbolic Flow Matching for Few-Shot Adaptation*, ICML — [arXiv:2602.20479](https://arxiv.org/abs/2602.20479) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-li-hyperbolic-flow-matching/) · [source](papers/2026-li-hyperbolic-flow-matching/notes.md) · [code](papers/2026-li-hyperbolic-flow-matching/code/) | transporting CLIP features to text prototypes on the Lorentz manifold |
| 2026 | Rezk, Lee, Gouk, Hospedales & Kim, *Weight Space Learning for Certifiable Few-shot Transfer Learning*, ICML — [arXiv:2502.06970](https://arxiv.org/abs/2502.06970) (earlier version) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-rezk-steel-certifiable-fewshot/) · [source](papers/2026-rezk-steel-certifiable-fewshot/notes.md) · [code](papers/2026-rezk-steel-certifiable-fewshot/code/) | non-vacuous few-shot certificates from a finite hypothesis class |
| 2026 | Zhang, Cui, Li & Wang, *Difficult Examples Hurt Unsupervised Contrastive Learning*, ICLR — [OpenReview](https://openreview.net/forum?id=5LMdnUdAoy) · [arXiv:2501.01317](https://arxiv.org/abs/2501.01317) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-zhang-difficult-examples/) · [source](papers/2026-zhang-difficult-examples/notes.md) | why deleting boundary examples improves contrastive learning |
| 2026 | Betser, Gofer, Levi & Gilboa, *InfoNCE Induces Gaussian Distribution*, ICLR (Oral) — [OpenReview](https://openreview.net/forum?id=BlSH7gNQSq) · [arXiv:2602.24012](https://arxiv.org/abs/2602.24012) · [project page](https://rbetser.github.io/InfoNCE-induces-Gaussian-distribution/) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-betser-infonce-gaussian/) · [source](papers/2026-betser-infonce-gaussian/notes.md) | why contrastive representations come out approximately Gaussian |
| 2025 | Tahir, Ganguli & Rotskoff, *Features are fate: a theory of transfer learning in high-dimensional regression*, ICML — [arXiv:2410.08194](https://arxiv.org/abs/2410.08194) · [code](https://github.com/javantahir/features_are_fate) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2025-tahir-features-are-fate/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2025-tahir-features-are-fate/figures/interactive.html) · [source](papers/2025-tahir-features-are-fate/notes.md) · [code](papers/2025-tahir-features-are-fate/code/) | transfer in a deep linear network from a tiny initialisation: pretraining leaves one direction, so linear transfer has a floor $\sin^2\theta$ against scratch training's double-descent risk, and $\mathcal T_{lt}$ is a phase diagram (positive for few samples, negative for many, positive again in a thin band at $\gamma = 1$; eq. 11 derived in three lines, the same pair of tasks changes sign three times as $n$ grows); fine-tuning keeps the pretrained guess where the data are silent and wins iff $\theta < 60^\circ$ (eq. 15), but real gradient descent carries a scale factor $c = 1 + O(\sigma^2)$ the printed theorem omits (break-even 52° at $\sigma = 1$); Appendix A's Dudley proof fails at step (29) and "any $\delta$" is false beyond 2, Figure 4(a) is a sweep over $[0,\pi]$ and 4(b)'s $W_1$ is sampling noise (≈ 527 for identical laws) |
| 2025 | Eftekhari & Papyan, *On the Importance of Gaussianizing Representations*, ICML — [arXiv:2505.00685](https://arxiv.org/abs/2505.00685) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2025-eftekhari-normality-normalization/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2025-eftekhari-normality-normalization/figures/power-transform.html) · [source](papers/2025-eftekhari-normality-normalization/notes.md) · [code](papers/2025-eftekhari-normality-normalization/code/) | a normalization layer that Gaussianizes each unit with a one-step Yeo–Johnson power transform, plus scaled Gaussian noise |
| 2023 | Chaves, Bissoto, Valle & Avila, *The Performance of Transferability Metrics does not Translate to Medical Tasks* — [arXiv:2308.07444](https://arxiv.org/abs/2308.07444) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2023-chaves-medical-transferability/) · [source](papers/2023-chaves-medical-transferability/notes.md) · [code](papers/2023-chaves-medical-transferability/code/) | seven transferability scores on medical targets, in and out of distribution |
| 2023 | Zhang, Tanno, Xu, Huang, Bronik, Jin, Jacob, Zheng, Shao, Ciccarelli, Barkhof & Alexander, *Learning from multiple annotators for medical image segmentation*, Pattern Recognition — [PMC10533416](https://pmc.ncbi.nlm.nih.gov/articles/PMC10533416/) · [arXiv:2007.15963](https://arxiv.org/abs/2007.15963) (NeurIPS 2020 version) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2023-zhang-multiple-annotators/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2023-zhang-multiple-annotators/figures/interactive.html) · [source](papers/2023-zhang-multiple-annotators/notes.md) · [code](papers/2023-zhang-multiple-annotators/code/) | separating annotator error from the true segmentation with a trace penalty on per-pixel confusion matrices (no HSIC); the per-pixel theorem needs column dominance, and without it the trace is smallest at the label swap |
| 2022 | Shao, Zhao, Ge, Zhang, Yang, Wang, Shan & Luo, *Not All Models Are Equal: Predicting Model Transferability in a Self-challenging Fisher Space*, ECCV — [arXiv:2207.03036](https://arxiv.org/abs/2207.03036) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2022-shao-sfda/) · [source](papers/2022-shao-sfda/notes.md) · [code](papers/2022-shao-sfda/code/) | imitating fine-tuning by making the target task deliberately harder |
| 2021 | You, Liu, Wang & Long, *LogME: Practical Assessment of Pre-trained Models for Transfer Learning*, ICML — [arXiv:2102.11005](https://arxiv.org/abs/2102.11005) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2021-you-logme/) · [source](papers/2021-you-logme/notes.md) · [code](papers/2021-you-logme/code/) | scoring features by the marginal likelihood of the labels, not the best fit to them |
| 2021 | Gao & Chaudhari, *An Information-Geometric Distance on the Space of Tasks*, ICML — [arXiv:2011.00613](https://arxiv.org/abs/2011.00613) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2021-gao-coupled-transfer-distance/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2021-gao-coupled-transfer-distance/figures/interactive.html) · [source](papers/2021-gao-coupled-transfer-distance/notes.md) · [code](papers/2021-gao-coupled-transfer-distance/code/) | task distance as the Fisher–Rao length of a classifier that tracks a task moved from source to target by optimal transport; Definition 2 averages the square root of the KL over inputs ($B\le A$ by Jensen, a Finsler length, not the Riemannian length of Eqs. 4–5) and (12b) is the speed of the weights through a frozen input, not an endpoint distance (2.63 against 0 for a perfectly tracking model); in an exactly solvable logistic toy the minimiser path is exactly reversible, so Remark 3's asymmetry is dynamical and vanishes as the learning speed $\kappa\to\infty$, the SGD length creeps to its limit (6.7% short at $\kappa=3000$) and Remark 1 (uncoupled is longer) holds for $B$ and the loss variation but fails for the Riemannian $A$ at large shifts; Theorem 6's printed $\lambda$ has an extra factor $K$ (exponent $+144$ where the right one is $-1.8$), its exponent has no $N$ (McDiarmid gives $\exp\{-2NK(\epsilon-2\bar R)^2/M^2\}$), the complexity term the proof supports for Theorems 5/7 is $2\,\mathrm{TV}/K$ rather than the length, and the reported Mantel $r$ cannot be reproduced from the printed matrices (Pearson 0.93 against 0.43 in Figure 2; a 4-task test cannot return $p<1/24$) |
| 2020 | Nguyen, Hassner, Seeger & Archambeau, *LEEP: A New Measure to Evaluate Transferability of Learned Representations*, ICML — [arXiv:2002.12462](https://arxiv.org/abs/2002.12462) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-nguyen-leep/) · [source](papers/2020-nguyen-leep/notes.md) · [code](papers/2020-nguyen-leep/code/) | scoring a source model by the likelihood of one hand-built classifier, not the best one |
| 2021 | HaoChen, Wei, Gaidon & Ma, *Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss*, NeurIPS (Oral) — [arXiv:2106.04156](https://arxiv.org/abs/2106.04156) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2021-haochen-spectral-contrastive/) · [source](papers/2021-haochen-spectral-contrastive/notes.md) | contrastive learning is spectral clustering of the augmentation graph |
| 2020 | Fort, Dziugaite, Paul, Kharaghani, Roy & Ganguli, *Deep Learning versus Kernel Learning*, NeurIPS — [arXiv:2010.15110](https://arxiv.org/abs/2010.15110) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-fort-deep-vs-kernel/) · [source](papers/2020-fort-deep-vs-kernel/notes.md) · [code](papers/2020-fort-deep-vs-kernel/code/) | how far real training departs from the NTK limit, and when |
| 2020 | Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML — [arXiv:2005.10242](https://arxiv.org/abs/2005.10242) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-wang-isola-alignment-uniformity/) · [source](papers/2020-wang-isola-alignment-uniformity/notes.md) · [code](papers/2020-wang-isola-alignment-uniformity/code/) | the decomposition of InfoNCE into alignment plus a potential-theory energy |
| 2020 | Achille, Paolini, Mbeng & Soatto, *The Information Complexity of Learning Tasks, their Structure and their Distance*, Information and Inference (2021) — [arXiv:1904.03292](https://arxiv.org/abs/1904.03292) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-achille-task-complexity/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-achille-task-complexity/figures/interactive.html) · [source](papers/2020-achille-task-complexity/notes.md) · [code](papers/2020-achille-task-complexity/code/) | task complexity as a two-part code with a factorised model, $C=\min_p L_{\mathcal D}(p)+K(p)$, its structure function and $\beta$-Lagrangian, and an asymmetric distance $K(p_{12})-K(p_1)$; whenever the optima are unique the asymmetry is exactly a difference of complexities, and the paper's Figure 1 obeys this to its rounding ($R^2=0.99992$); over all $Q$ the computable Lagrangian is a free energy (the evidence at $\beta=1$), and Theorem 5.4's $O(1)$ is $\tfrac k2\log(N/\beta)-\tfrac k2$, millions of nats at network scale; Theorem 4.4 needs $\beta\ge1$, Lemma A.1 a finite input space and Example 3.5 an index oracle; and sharp phase transitions need a non-convex structure function, so a convex model cannot show them |
| 2020 | Achille, Paolini & Soatto, *Where is the Information in a Deep Neural Network?* — [arXiv:1905.12213](https://arxiv.org/abs/1905.12213) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-achille-information-in-weights/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-achille-information-in-weights/figures/interactive.html) · [source](papers/2020-achille-information-in-weights/notes.md) · [code](papers/2020-achille-information-in-weights/code/) | information in the weights as the optimum of $\mathbb E_Q L+\beta\,\mathrm{KL}(Q\Vert P)$, whose minimiser is the Gibbs distribution; Shannon and Fisher are special codes. The Gaussian optimum is $\beta(H+\beta/\lambda^2)^{-1}$, twice the stated one; the activation Fisher is $\nabla_x f^\top(\beta JF^{-1}J^\top)^{-1}\nabla_x f$, not $JFJ^\top/\beta$, and Eq 10 needs $\dim z\ge\dim x$; isotropic Kramers gives no temperature-dependent preference among equal-loss minima; and in the rebuilt Figure 3 toy the batch-size drop in "Shannon information" is the entropy of which equivalent minimum the seed reached, while $I(\theta;\mathcal D)$ stays flat |
| 2019 | Bao, Li, Huang, Zhang, Zheng, Zamir & Guibas, *An Information-Theoretic Approach to Transferability in Task Transfer Learning*, ICIP — [arXiv:2212.10082](https://arxiv.org/abs/2212.10082) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2019-bao-hscore-transferability/) · [source](papers/2019-bao-hscore-transferability/notes.md) · [code](papers/2019-bao-hscore-transferability/code/) | predicting transfer performance without training, via a projection onto the feature subspace |
| 2019 | Achille, Mbeng & Soatto, *Dynamics and Reachability of Learning Tasks* — [arXiv:1810.02440](https://arxiv.org/abs/1810.02440) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2019-achille-task-reachability/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2019-achille-task-reachability/figures/interactive.html) · [source](papers/2019-achille-task-reachability/notes.md) · [code](papers/2019-achille-task-reachability/code/) | SGD as a Langevin particle and a path integral over training runs: the static × reachability split is exactly detailed balance (the path factor is symmetric), and SGD's stationary law is exactly the minimiser of the complexity Lagrangian with β = D and λ² = D/γ (the paper's β = 2λ²γ doubles the weight decay); the curvature correction is (D/2) log|H|, which decides when a sharp minimum loses or turns into a bump; eq. (16) puts end points where Kramers puts the barrier, and on the most likely downhill path the static factor is cancelled exactly; Figure 2's printed matrices still show the static distance ranking sources within a target (r ≈ 0.6); plus a first-reading Q&A (transition probability, the information-theoretic distance, reachability, eqs. 2, 3, 5 and 9 and the SDE line by line, the Fisher in place of the Hessian, a solvable structure function, feature-only transferability scores, dataset against architecture) |
| 2019 | Tran, Nguyen & Hassner, *Transferability and Hardness of Supervised Classification Tasks*, ICCV — [arXiv:1908.08142](https://arxiv.org/abs/1908.08142) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2019-tran-nce-hardness/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2019-tran-nce-hardness/figures/interactive.html) · [source](papers/2019-tran-nce-hardness/notes.md) · [code](papers/2019-tran-nce-hardness/code/) | bounding transfer with the conditional entropy of two label sequences, no model needed; the proof's one-term drop costs $\ln(1+\text{other terms}/\text{kept term})$ per image, so the bound is tight for a one-to-one source and loose by $\ln g$ for a fine source with a coarse target; Figure 1 as drawn ties four panels at $\log 2$ (the text's $4\log 2$ needs sixteen labels); in the paper's own Tables 3 and 4, Figure 3's $r=0.78$ recomputes to $0.63$, the target's label entropy alone ties it on Pearson $r$ ($0.62$) though not on rank ($0.66$ against $0.83$), the four CelebA panels in Figure 2 are exactly the top four of forty, and the counted $H(Y\mid Z)$ is biased low by about $0.028$ nats at $17.9$ images per identity, as large as the smallest reported values |
| 2018 | Jacot, Gabriel & Hongler, *Neural Tangent Kernel: Convergence and Generalization in Neural Networks*, NeurIPS — [arXiv:1806.07572](https://arxiv.org/abs/1806.07572) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2018-jacot-neural-tangent-kernel/) · [source](papers/2018-jacot-neural-tangent-kernel/notes.md) · [code](papers/2018-jacot-neural-tangent-kernel/code/) | wide networks train as kernel regression in function space |
| 2017 | Naesseth, Ruiz, Linderman & Blei, *Reparameterization Gradients through Acceptance-Rejection Sampling Algorithms*, AISTATS — [arXiv:1610.05683](https://arxiv.org/abs/1610.05683) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2017-naesseth-rejection-reparameterization/) · [source](papers/2017-naesseth-rejection-reparameterization/notes.md) · [code](papers/2017-naesseth-rejection-reparameterization/code/) | differentiating through an accept/reject step by integrating out the coin |
| 2016 | Amari, *Information Geometry and Its Applications*, Springer, Chapter 1 — [DOI](https://doi.org/10.1007/978-4-431-55978-8) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2016-amari-dually-flat-structure/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2016-amari-dually-flat-structure/figures/interactive.html) · [source](papers/2016-amari-dually-flat-structure/notes.md) · [code](papers/2016-amari-dually-flat-structure/code/) | one convex function $\psi$ yields a divergence, a second (Legendre-dual) coordinate system, a metric and two flat structures; softmax logits and probabilities as the running example; the Pythagorean and projection theorems and the alternating (em) minimisation, each rechecked numerically. The printed identity (1.114) has permuted indices, and Theorem 1.4 and the em paragraph pair each projection with the opposite argument slot of $D_\psi$ to what the calculus gives |
