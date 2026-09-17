# Theory-inclined papers, with annotations

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
│       └── paper.pdf         gitignored, fetched or dropped in by hand
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

## Usage

```sh
make                              # render everything into build/
make open                         # render, then open build/index.html
make serve                        # render, then serve on http://localhost:8000
make new SLUG=2026-lastname-topic # scaffold a new paper folder
make fetch                        # download any missing PDFs
make list                         # list the papers in the repo
make check                        # verify pandoc is present and notes parse
make clean                        # remove build/
```

`make` is incremental: it only re-runs when a `notes.md`, the build script or the
stylesheet has changed.

## Adding a paper

```sh
make new SLUG=2027-lastname-topic
```

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

### Writing the body

Plain Markdown, plus:

- inline maths between single dollars: `$\eta_2 = \rho_m^2(X, X_0)$`
- display maths between double dollars on their own lines
- tables for symbol glossaries, which are worth including early in every annotation

A loose section order that has worked so far: *In one paragraph* → *The spine of the
argument* → *Setup and notation* → the results, in the paper's own order → *Questions and
doubts* → *Takeaways*. The doubts section is the point of the exercise; it should not be
left empty.

## Papers

| Year | Paper | Topic |
|---|---|---|
| 2026 | Betser, Gofer, Levi & Gilboa — *InfoNCE Induces Gaussian Distribution* (ICLR) | why contrastive representations come out approximately Gaussian |
