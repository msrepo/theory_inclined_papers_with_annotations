#!/usr/bin/env python3
"""Render every papers/<slug>/notes.md to build/<slug>/index.html and write an index.

No third-party dependencies: the YAML front matter is restricted to a small
flat subset (scalars and inline lists) that is parsed here directly.
"""
from __future__ import annotations

import html
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Pin the KaTeX CDN explicitly. Bare `--katex` uses whatever the local pandoc
# build was configured with, and Debian's package points at the filesystem path
# /usr/share/javascript/katex/ -- which 404s once the site is served elsewhere.
KATEX_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/"
PAPERS = ROOT / "papers"
# Pages that are not annotations of one paper. Same front matter and same
# pipeline; separate roots only so the index can group them. Order here is the
# order they appear on the index page.
COLLECTIONS = [
    (ROOT / "foundations", "foundations"),   # background: the maths the papers run on
    (ROOT / "topics", "topic"),              # themes spanning several papers
    (PAPERS, "paper"),                       # one paper each
]
NON_PAPER_KINDS = {"foundations", "topic"}

# Sidebar taxonomy. Each page picks a `category` (and optionally a
# `subcategory`) in its front matter; this list fixes the display order and is
# the only place to edit when adding a section. Anything uncategorised is
# collected under "Unsorted" at the end rather than silently disappearing.
CATEGORIES: list[tuple[str, list[str]]] = [
    ("Foundations", []),
    ("NTK & function space", ["Theory", "Applications"]),
    ("Contrastive learning", ["Theory", "Applications"]),
    ("Misc", []),
]
UNSORTED = "Unsorted"
BUILD = ROOT / "build"
STYLE = ROOT / "tools" / "style.css"

FIELDS_SHOWN = ("authors", "venue", "year")


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a leading `---` fenced block off the document.

    Supports `key: value`, `key: "value"` and `key: [a, b, c]`. Anything more
    elaborate belongs in the body, not the metadata.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = next((i for i, l in enumerate(lines[1:], start=1) if l.strip() == "---"), None)
    if end is None:
        return {}, text
    meta: dict = {}
    for line in lines[1:end]:
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [v.strip().strip("\"'") for v in value[1:-1].split(",")]
            meta[key] = [v for v in items if v]
        else:
            meta[key] = value.strip("\"'")
    return meta, "\n".join(lines[end + 1:])


def discover() -> list[tuple[Path, dict]]:
    """All pages, from both source roots. Slugs must be unique across roots:
    everything renders into a flat build/<slug>/ so cross-links stay ../<slug>/."""
    found = []
    for root, kind in COLLECTIONS:
        for notes in sorted(root.glob("*/notes.md")):
            meta, _ = parse_front_matter(notes.read_text(encoding="utf-8"))
            meta.setdefault("title", notes.parent.name)
            meta["slug"] = notes.parent.name
            meta["kind"] = kind
            found.append((notes, meta))
    return found


def meta_line(meta: dict) -> str:
    bits = [html.escape(str(meta[f])) for f in FIELDS_SHOWN if meta.get(f)]
    return " &middot; ".join(bits)


def tag_html(meta: dict) -> str:
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)


def render_one(notes: Path, meta: dict, entries: list) -> None:
    slug = meta["slug"]
    out_dir = BUILD / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-paper figures are published beside the page so notes.md can reference
    # them at figures/<name>. They are self-contained themed SVGs rather than
    # inline markup, so the Markdown source stays readable on GitHub too.
    figures = notes.parent / "figures"
    if figures.is_dir():
        shutil.copytree(figures, out_dir / "figures", dirs_exist_ok=True)

    links = []
    pdf = notes.parent / "paper.pdf"
    if pdf.exists():
        shutil.copy2(pdf, out_dir / "paper.pdf")
        links.append('<a href="paper.pdf">pdf</a>')
    elif meta.get("pdf_url"):
        links.append(f'<a href="{html.escape(meta["pdf_url"])}">pdf (remote)</a>')
    if meta.get("url"):
        links.append(f'<a href="{html.escape(meta["url"])}">paper page</a>')
    links.append('<a href="../index.html">all notes</a>')

    header = (
        '<header class="paper-head">'
        f'<h1>{html.escape(str(meta["title"]))}</h1>'
        f'<p class="meta">{meta_line(meta)}</p>'
        f'<p class="tags">{tag_html(meta)}</p>'
        f'<p class="links">{" &middot; ".join(links)}</p>'
        "</header>"
    )
    # before-body opens the two-column layout and emits the shared nav;
    # after-body closes it. Pandoc drops the TOC and content in between, so
    # they land inside <main>.
    before = (sidebar_html(entries, slug, depth=1)
              + '<main>' + header)
    header_file = out_dir / ".header.html"
    header_file.write_text(f'<div class="layout">{before}', encoding="utf-8")
    after_file = out_dir / ".after.html"
    after_file.write_text("</main></div>", encoding="utf-8")

    # Feed pandoc the body only. Left in place, the front matter's `title`
    # would make pandoc emit its own title block on top of the header above.
    _, body = parse_front_matter(notes.read_text(encoding="utf-8"))
    body_file = out_dir / ".body.md"
    body_file.write_text(body, encoding="utf-8")

    cmd = [
        "pandoc", str(body_file),
        "--from", "markdown+tex_math_dollars+tex_math_single_backslash",
        "--to", "html5",
        "--standalone",
        f"--katex={KATEX_CDN}",
        "--toc", "--toc-depth=2",
        "--css", "../style.css",
        # pagetitle sets <title> without emitting pandoc's own title block,
        # which would duplicate the header we inject below.
        "--metadata", f"pagetitle={meta['title']}",
        "--include-before-body", str(header_file),
        "--include-after-body", str(after_file),
        "--output", str(out_dir / "index.html"),
    ]
    subprocess.run(cmd, check=True)
    header_file.unlink()
    after_file.unlink()
    body_file.unlink()
    print(f"  built  {slug}")


def _grouped(entries: list[tuple[Path, dict]]) -> list[tuple[str, list[tuple[str, list[dict]]]]]:
    """[(category, [(subcategory_or_empty, [meta, ...]), ...]), ...] in display order."""
    by_cat: dict[str, dict[str, list[dict]]] = {}
    for _, m in entries:
        by_cat.setdefault(m.get("category") or UNSORTED, {}).setdefault(
            m.get("subcategory") or "", []).append(m)

    def sort_key(m: dict) -> tuple:
        # newest first, then by title; undated pages (foundations, topics) lead
        return (-int(m["year"]) if str(m.get("year", "")).isdigit() else -9999,
                str(m["title"]).lower())

    order = [c for c, _ in CATEGORIES] + [UNSORTED]
    out = []
    for cat in order:
        if cat not in by_cat:
            continue
        subs = dict(CATEGORIES).get(cat, [])
        seen = by_cat[cat]
        keys = [k for k in subs if k in seen] + [k for k in seen if k not in subs]
        out.append((cat, [(k, sorted(seen[k], key=sort_key)) for k in keys]))
    return out


def sidebar_html(entries: list[tuple[Path, dict]], current: str | None, depth: int) -> str:
    """Nav shared by every page. `depth` is how many levels up the site root is."""
    up = "../"*depth
    parts = [f'<nav class="sidebar"><details open><summary>Contents</summary>',
             f'<a class="nav-home" href="{up}index.html">All notes</a>']
    for cat, subs in _grouped(entries):
        parts.append(f'<div class="nav-cat">{html.escape(cat)}</div>')
        for sub, metas in subs:
            if sub:
                parts.append(f'<div class="nav-sub">{html.escape(sub)}</div>')
            parts.append("<ul>")
            for m in metas:
                cur = ' aria-current="page"' if m["slug"] == current else ""
                short = m.get("short_title") or m["title"]
                parts.append(f'<li><a href="{up}{html.escape(m["slug"])}/index.html"{cur}>'
                             f'{html.escape(str(short))}</a></li>')
            parts.append("</ul>")
    parts.append("</details></nav>")
    return "".join(parts)


def _entry_html(meta: dict) -> str:
    status = meta.get("status", "")
    badge = (f'<span class="status status-{html.escape(status)}">{html.escape(status)}</span>'
             if status else "")
    return (
        f'<li><a class="title" href="{html.escape(meta["slug"])}/index.html">'
        f'{html.escape(str(meta["title"]))}</a>{badge}'
        f'<div class="meta">{meta_line(meta)}</div>'
        f'<div class="tags">{tag_html(meta)}</div></li>'
    )


def render_index(entries: list[tuple[Path, dict]]) -> None:
    rows = []
    for cat, subs in _grouped(entries):
        rows.append(f"<h2>{html.escape(cat)}</h2>")
        for sub, metas in subs:
            if sub:
                rows.append(f"<h3>{html.escape(sub)}</h3>")
            rows.append("<ul class='papers'>")
            rows += [_entry_html(m) for m in metas]
            rows.append("</ul>")

    n_paper = len([m for _, m in entries if m.get("kind") not in NON_PAPER_KINDS])
    n_other = len(entries) - n_paper
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Theory-inclined papers, with annotations</title>
<link rel="stylesheet" href="style.css"></head>
<body><div class="layout">
{sidebar_html(entries, None, depth=0)}
<main>
<header class="paper-head"><h1>Theory-inclined papers, with annotations</h1>
<p class="meta">{n_paper} paper(s), {n_other} background page(s)</p></header>
{"".join(rows)}
</main></div></body></html>
"""
    (BUILD / "index.html").write_text(doc, encoding="utf-8")
    print(f"  built  index.html ({len(entries)} page(s))")


def main() -> int:
    if shutil.which("pandoc") is None:
        sys.exit("pandoc not found. Install it: brew install pandoc")
    entries = discover()
    if not entries:
        sys.exit("No <slug>/notes.md found under papers/ or foundations/. Scaffold one with: make new SLUG=my-paper")
    BUILD.mkdir(exist_ok=True)
    shutil.copy2(STYLE, BUILD / "style.css")
    for notes, meta in entries:
        render_one(notes, meta, entries)
    render_index(entries)
    print(f"\nOpen {BUILD / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
