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

Every push to `main` that touches `papers/`, `tools/` or the `Makefile` triggers
`.github/workflows/pages.yml`, which runs `make` and deploys `build/` to GitHub Pages.

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
| *Principal angles between subspaces* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/principal-angles-subspaces/) · [source](foundations/principal-angles-subspaces/notes.md) · [code](foundations/principal-angles-subspaces/code/) | the SVD characterisation, the projector and distance identities, why `arccos` destroys small angles and the sine fix, and the equivalence with canonical correlations |

**Probability**

| Page | Covers |
|---|---|
| *Optimal transport and the Wasserstein distance* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/optimal-transport/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/optimal-transport/figures/interactive.html) · [source](foundations/optimal-transport/notes.md) | Monge maps vs. Kantorovich couplings, pushforward in plain words, $W_p$ and why it beats KL/JS without overlap (WGAN), Sinkhorn, duality, 1-D and Gaussian (FID) closed forms, minibatch OT for flow matching, and a symbol-by-symbol reading of the discrete OT definition used in data selection |
| *Expectation–Maximization (EM) and Gaussian mixtures* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/expectation-maximization/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/expectation-maximization/figures/interactive.html) · [source](foundations/expectation-maximization/notes.md) · [code](foundations/expectation-maximization/code/) | the chicken-and-egg of latent labels, responsibilities by Bayes' rule (a softmax, and a sigmoid for two blobs), weighted M-step refits with a worked six-point example, why the likelihood never drops (Jensen, a touching lower bound, the $\ln p = \mathcal L + \mathrm{KL}$ gap), k-means as the hard limit, and the step to the ELBO of VAEs |

## Topics

Themes that cut across several papers.

| Page | Covers |
|---|---|
| *Applications of Gaussianity: uncertainty, dense prediction, test-time adaptation* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/gaussianity-in-practice/) · [source](topics/gaussianity-in-practice/notes.md) | the applied literature that assumes CLIP features are Gaussian, with close reads of Zhou et al. 2025 (CVPR), Venkataramanan et al. 2025 (UAI) and C. Huang et al. 2024 (IJCAI) |
| *MICCAI 2026: domain adaptation and generalisation — the mathematical constructs* — [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/miccai2026-domain-adaptation/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/miccai2026-domain-adaptation/figures/interactive.html) · [source](topics/miccai2026-domain-adaptation/notes.md) | the core mathematical construct of each of 26 MICCAI 2026 DA/DG/TTA/OOD papers (Mahalanobis residuals, deep EM, OT and Schrödinger bridges, flow-matching TTA, InfoMax, variational logit energies, GRPO), with a construct map, one toy-computed figure per paper, and a *What to watch* per paper |

## Papers

| Year | Paper | Notes | Topic |
|---|---|---|---|
| 2026 | Diniz, de Faria Júnior & Ester, *PAS: Estimating the target accuracy before domain adaptation*, ICLR — [arXiv:2604.09863](https://arxiv.org/abs/2604.09863) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-diniz-pas/) · [source](papers/2026-diniz-pas/notes.md) · [code](papers/2026-diniz-pas/code/) | scoring a source domain with no target labels at all, via a nearest-centroid margin |
| 2026 | Claßen, Sourget, Juodelyte, van der Goot & Cheplygina, *Robustness of transferability estimation metrics for medical imaging* — [arXiv:2608.09999](https://arxiv.org/abs/2608.09999) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-classen-te-robustness/) · [source](papers/2026-classen-te-robustness/notes.md) · [code](papers/2026-classen-te-robustness/code/) | whether transferability rankings survive a change of random seed; mostly they do not |
| 2026 | Wang, Liu, Liao, Fan, Du, Tang, Wang & Wang, *Mining Useful General Data for Low-Resource Domain Adaptation*, ICML — [repository](https://github.com/applewpj/NTK-Selector) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-wang-ntk-selector/) · [source](papers/2026-wang-ntk-selector/notes.md) · [code](papers/2026-wang-ntk-selector/code/) | mining general-domain data by NTK alignment with a tiny domain set |
| 2026 | Li, Jiang, Ye, He, Li, Xiao, Cheng & Chen, *Path-Decoupled Hyperbolic Flow Matching for Few-Shot Adaptation*, ICML — [arXiv:2602.20479](https://arxiv.org/abs/2602.20479) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-li-hyperbolic-flow-matching/) · [source](papers/2026-li-hyperbolic-flow-matching/notes.md) · [code](papers/2026-li-hyperbolic-flow-matching/code/) | transporting CLIP features to text prototypes on the Lorentz manifold |
| 2026 | Rezk, Lee, Gouk, Hospedales & Kim, *Weight Space Learning for Certifiable Few-shot Transfer Learning*, ICML — [arXiv:2502.06970](https://arxiv.org/abs/2502.06970) (earlier version) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-rezk-steel-certifiable-fewshot/) · [source](papers/2026-rezk-steel-certifiable-fewshot/notes.md) · [code](papers/2026-rezk-steel-certifiable-fewshot/code/) | non-vacuous few-shot certificates from a finite hypothesis class |
| 2026 | Zhang, Cui, Li & Wang, *Difficult Examples Hurt Unsupervised Contrastive Learning*, ICLR — [OpenReview](https://openreview.net/forum?id=5LMdnUdAoy) · [arXiv:2501.01317](https://arxiv.org/abs/2501.01317) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-zhang-difficult-examples/) · [source](papers/2026-zhang-difficult-examples/notes.md) | why deleting boundary examples improves contrastive learning |
| 2026 | Betser, Gofer, Levi & Gilboa, *InfoNCE Induces Gaussian Distribution*, ICLR (Oral) — [OpenReview](https://openreview.net/forum?id=BlSH7gNQSq) · [arXiv:2602.24012](https://arxiv.org/abs/2602.24012) · [project page](https://rbetser.github.io/InfoNCE-induces-Gaussian-distribution/) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2026-betser-infonce-gaussian/) · [source](papers/2026-betser-infonce-gaussian/notes.md) | why contrastive representations come out approximately Gaussian |
| 2025 | Eftekhari & Papyan, *On the Importance of Gaussianizing Representations*, ICML — [arXiv:2505.00685](https://arxiv.org/abs/2505.00685) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2025-eftekhari-normality-normalization/) · [interactive](https://msrepo.github.io/theory_inclined_papers_with_annotations/2025-eftekhari-normality-normalization/figures/power-transform.html) · [source](papers/2025-eftekhari-normality-normalization/notes.md) · [code](papers/2025-eftekhari-normality-normalization/code/) | a normalization layer that Gaussianizes each unit with a one-step Yeo–Johnson power transform, plus scaled Gaussian noise |
| 2023 | Chaves, Bissoto, Valle & Avila, *The Performance of Transferability Metrics does not Translate to Medical Tasks* — [arXiv:2308.07444](https://arxiv.org/abs/2308.07444) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2023-chaves-medical-transferability/) · [source](papers/2023-chaves-medical-transferability/notes.md) · [code](papers/2023-chaves-medical-transferability/code/) | seven transferability scores on medical targets, in and out of distribution |
| 2022 | Bao, Li, Huang, Zhang, Zheng, Zamir & Guibas, *An Information-Theoretic Approach to Transferability in Task Transfer Learning* — [arXiv:2212.10082](https://arxiv.org/abs/2212.10082) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2022-bao-hscore-transferability/) · [source](papers/2022-bao-hscore-transferability/notes.md) · [code](papers/2022-bao-hscore-transferability/code/) | predicting transfer performance without training, via a projection onto the feature subspace |
| 2022 | Shao, Zhao, Ge, Zhang, Yang, Wang, Shan & Luo, *Not All Models Are Equal: Predicting Model Transferability in a Self-challenging Fisher Space*, ECCV — [arXiv:2207.03036](https://arxiv.org/abs/2207.03036) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2022-shao-sfda/) · [source](papers/2022-shao-sfda/notes.md) · [code](papers/2022-shao-sfda/code/) | imitating fine-tuning by making the target task deliberately harder |
| 2021 | You, Liu, Wang & Long, *LogME: Practical Assessment of Pre-trained Models for Transfer Learning*, ICML — [arXiv:2102.11005](https://arxiv.org/abs/2102.11005) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2021-you-logme/) · [source](papers/2021-you-logme/notes.md) · [code](papers/2021-you-logme/code/) | scoring features by the marginal likelihood of the labels, not the best fit to them |
| 2020 | Nguyen, Hassner, Seeger & Archambeau, *LEEP: A New Measure to Evaluate Transferability of Learned Representations*, ICML — [arXiv:2002.12462](https://arxiv.org/abs/2002.12462) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-nguyen-leep/) · [source](papers/2020-nguyen-leep/notes.md) · [code](papers/2020-nguyen-leep/code/) | scoring a source model by the likelihood of one hand-built classifier, not the best one |
| 2021 | HaoChen, Wei, Gaidon & Ma, *Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss*, NeurIPS (Oral) — [arXiv:2106.04156](https://arxiv.org/abs/2106.04156) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2021-haochen-spectral-contrastive/) · [source](papers/2021-haochen-spectral-contrastive/notes.md) | contrastive learning is spectral clustering of the augmentation graph |
| 2020 | Fort, Dziugaite, Paul, Kharaghani, Roy & Ganguli, *Deep Learning versus Kernel Learning*, NeurIPS — [arXiv:2010.15110](https://arxiv.org/abs/2010.15110) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-fort-deep-vs-kernel/) · [source](papers/2020-fort-deep-vs-kernel/notes.md) · [code](papers/2020-fort-deep-vs-kernel/code/) | how far real training departs from the NTK limit, and when |
| 2020 | Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML — [arXiv:2005.10242](https://arxiv.org/abs/2005.10242) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2020-wang-isola-alignment-uniformity/) · [source](papers/2020-wang-isola-alignment-uniformity/notes.md) · [code](papers/2020-wang-isola-alignment-uniformity/code/) | the decomposition of InfoNCE into alignment plus a potential-theory energy |
| 2019 | Tran, Nguyen & Hassner, *Transferability and Hardness of Supervised Classification Tasks*, ICCV — [arXiv:1908.08142](https://arxiv.org/abs/1908.08142) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2019-tran-nce-hardness/) · [source](papers/2019-tran-nce-hardness/notes.md) · [code](papers/2019-tran-nce-hardness/code/) | bounding transfer with the conditional entropy of two label sequences, no model needed |
| 2018 | Jacot, Gabriel & Hongler, *Neural Tangent Kernel: Convergence and Generalization in Neural Networks*, NeurIPS — [arXiv:1806.07572](https://arxiv.org/abs/1806.07572) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2018-jacot-neural-tangent-kernel/) · [source](papers/2018-jacot-neural-tangent-kernel/notes.md) · [code](papers/2018-jacot-neural-tangent-kernel/code/) | wide networks train as kernel regression in function space |
| 2017 | Naesseth, Ruiz, Linderman & Blei, *Reparameterization Gradients through Acceptance-Rejection Sampling Algorithms*, AISTATS — [arXiv:1610.05683](https://arxiv.org/abs/1610.05683) | [read online](https://msrepo.github.io/theory_inclined_papers_with_annotations/2017-naesseth-rejection-reparameterization/) · [source](papers/2017-naesseth-rejection-reparameterization/notes.md) · [code](papers/2017-naesseth-rejection-reparameterization/code/) | differentiating through an accept/reject step by integrating out the coin |
