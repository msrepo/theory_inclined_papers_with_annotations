#!/usr/bin/env python3
"""Render every papers/<slug>/notes.md to build/<slug>/index.html and write an index.

No third-party dependencies: the YAML front matter is restricted to a small
flat subset (scalars and inline lists) that is parsed here directly.
"""
from __future__ import annotations

import html
import math
import re
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
    ("Foundations", ["Linear algebra", "Probability", "Optimization", "Information geometry"]),
    ("NTK & function space", ["Theory", "Applications"]),
    ("Contrastive learning", ["Theory", "Applications"]),
    ("Attention Mechanism", ["Theory"]),
    ("Gradient estimation", ["Theory"]),
    ("Transferability", ["Theory", "Applications"]),
    ("Generative models", ["Theory"]),
    ("Bayesian inference", ["Review", "Theory"]),
    ("Theory of deep learning", ["Theory"]),
    ("High-dimensional statistics", ["Theory"]),
    ("Medical image analysis", ["Uncertainty & annotator noise", "Registration", "Anomaly detection",
                                "Federated learning", "Domain adaptation / generalization",
                                "Image quality assessment"]),
    # "" first: untitled Misc pages must not render under the last subheading.
    ("Misc", [""]),
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
    attach_related(found)
    return found


def attach_related(found: list[tuple[Path, dict]]) -> None:
    """Fill meta["linked"] and meta["similar"] for every page.

    `linked`: pages this one cites by a relative link (../<slug>/...) in its body
    or lists under `related:` in the front matter, plus pages that cite it.
    `similar`: pages sharing tags, strongest first, for pages with no link between them.
    Nothing here needs hand-maintenance: a link written in the prose is enough.
    """
    slugs = {m["slug"] for _, m in found}
    out: dict[str, set[str]] = {}
    for notes, m in found:
        _, body = parse_front_matter(notes.read_text(encoding="utf-8"))
        cited = set(re.findall(r"\.\./([a-z0-9][a-z0-9-]*)/", body))
        cited |= set(m.get("related") or [])
        out[m["slug"]] = {c for c in cited if c in slugs and c != m["slug"]}
    linked = {s: set(c) for s, c in out.items()}
    for s_, cited in out.items():
        for c in cited:
            linked[c].add(s_)
    tag_count: dict[str, int] = {}
    for _, m in found:
        for t in set(m.get("tags") or []):
            tag_count[t] = tag_count.get(t, 0) + 1
    n = len(found)
    # rare tags carry more weight; "background" and the like carry almost none
    weight = {t: math.log(n / c) for t, c in tag_count.items()}
    for _, m in found:
        mine = set(m.get("tags") or [])
        scored = []
        for _, o in found:
            if o["slug"] == m["slug"] or o["slug"] in linked[m["slug"]]:
                continue
            shared = mine & set(o.get("tags") or [])
            # two shared tags, or one tag that few pages carry
            if len(shared) >= 2 or any(tag_count[t] <= 4 for t in shared):
                scored.append((sum(weight[t] for t in shared), o["slug"], sorted(shared)))
        scored.sort(key=lambda x: (-x[0], x[1]))
        m["linked"] = sorted(linked[m["slug"]])
        m["similar"] = [(sl, sh) for _, sl, sh in scored[:5]]


def meta_line(meta: dict) -> str:
    bits = [html.escape(str(meta[f])) for f in FIELDS_SHOWN if meta.get(f)]
    return " &middot; ".join(bits)


def tag_html(meta: dict) -> str:
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return "".join(f'<button type="button" class="tag" data-tag="{html.escape(t)}">{html.escape(t)}</button>'
                   for t in tags)


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
    head_file = out_dir / ".head.html"
    head_file.write_text(HEAD_SCRIPT, encoding="utf-8")
    header_file = out_dir / ".header.html"
    header_file.write_text(
        TOGGLE_BUTTON + '<div class="layout">'
        + sidebar_html(entries, slug, depth=1) + "<main>" + header, encoding="utf-8")
    after_file = out_dir / ".after.html"
    after_file.write_text(related_html(meta, entries) + "</main></div>" + TOGGLE_SCRIPT
                          + '<script src="../site.js" defer></script>', encoding="utf-8")

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
        "--include-in-header", str(head_file),
        "--include-before-body", str(header_file),
        "--include-after-body", str(after_file),
        "--output", str(out_dir / "index.html"),
    ]
    subprocess.run(cmd, check=True)
    head_file.unlink()
    header_file.unlink()
    after_file.unlink()
    body_file.unlink()
    print(f"  built  {slug}")


KIND_LABEL = {"paper": "paper", "foundations": "foundation", "topic": "topic"}


def related_html(meta: dict, entries: list[tuple[Path, dict]]) -> str:
    by_slug = {m["slug"]: m for _, m in entries}

    def item(slug: str, note: str = "") -> str:
        o = by_slug[slug]
        blurb = f'<span class="rel-note">{html.escape(note)}</span>' if note else ""
        return (f'<li><a href="../{html.escape(slug)}/index.html">{html.escape(str(o["title"]))}</a> '
                f'<span class="kind kind-{o["kind"]}">{KIND_LABEL.get(o["kind"], o["kind"])}</span>'
                f'<div class="meta">{meta_line(o)}</div>{blurb}</li>')

    parts = []
    if meta.get("linked"):
        parts.append("<h3>Linked from the notes</h3><ul>" + "".join(item(s) for s in meta["linked"]) + "</ul>")
    if meta.get("similar"):
        parts.append("<h3>Similar tags</h3><ul>" + "".join(
            item(s, "shares " + ", ".join(sh)) for s, sh in meta["similar"]) + "</ul>")
    if not parts:
        return ""
    return '<section class="related" id="related-notes"><h2>Related notes</h2>' + "".join(parts) + "</section>"


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


BURGER = ('<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">'
          '<path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" stroke-width="1.5" '
          'stroke-linecap="round" fill="none"/></svg>')

# Applied before first paint so a collapsed sidebar never flashes open. Defaults
# to collapsed on narrow viewports, expanded otherwise, then remembers the
# choice. Wrapped in try/catch because localStorage throws in private mode.
HEAD_SCRIPT = """<script>
(function(){try{
  var v=localStorage.getItem('sidebar-collapsed');
  if(v===null)v=window.matchMedia('(max-width: 66rem)').matches?'1':'0';
  if(v==='1')document.documentElement.classList.add('nav-collapsed');
}catch(e){}})();
</script>"""

TOGGLE_SCRIPT = """<script>
(function(){
  var b=document.querySelector('.nav-toggle');if(!b)return;
  function sync(){b.setAttribute('aria-expanded',
    String(!document.documentElement.classList.contains('nav-collapsed')));}
  sync();
  b.addEventListener('click',function(){
    var c=document.documentElement.classList.toggle('nav-collapsed');
    try{localStorage.setItem('sidebar-collapsed',c?'1':'0');}catch(e){}
    sync();
  });
})();
</script>"""

TOGGLE_BUTTON = ('<button class="nav-toggle" type="button" aria-controls="site-nav" '
                 f'aria-expanded="true" aria-label="Show or hide the contents sidebar">{BURGER}</button>')


def sidebar_html(entries: list[tuple[Path, dict]], current: str | None, depth: int) -> str:
    """Nav shared by every page. `depth` is how many levels up the site root is."""
    up = "../"*depth
    parts = [f'<nav class="sidebar" id="site-nav">',
             f'<a class="nav-home" href="{up}index.html">All notes</a>',
             '<input class="nav-filter" type="search" placeholder="Filter pages ( / )" '
             'aria-label="Filter the sidebar" autocomplete="off">']
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
    parts.append("</nav>")
    return "".join(parts)


def _entry_html(meta: dict) -> str:
    status = meta.get("status", "")
    badge = (f'<span class="status status-{html.escape(status)}">{html.escape(status)}</span>'
             if status else "")
    kind = meta.get("kind", "paper")
    kind_badge = f'<span class="kind kind-{kind}">{KIND_LABEL.get(kind, kind)}</span>' if kind != "paper" else ""
    tags = meta.get("tags") or []
    text = " ".join(str(meta.get(k, "")) for k in
                    ("title", "short_title", "authors", "venue", "year", "category", "subcategory", "summary"))
    summary = (f'<p class="summary" tabindex="0" title="Click to expand">{html.escape(str(meta["summary"]))}</p>'
               if meta.get("summary") else "")
    return (
        f'<li data-kind="{kind}" data-status="{html.escape(status)}" data-year="{html.escape(str(meta.get("year", "")))}" '
        f'data-title="{html.escape(str(meta["title"]).lower())}" data-tags="{html.escape("|".join(tags))}" '
        f'data-text="{html.escape((text + " " + " ".join(tags)).lower())}">'
        f'<a class="title" href="{html.escape(meta["slug"])}/index.html">'
        f'{html.escape(str(meta["title"]))}</a>{kind_badge}{badge}'
        f'<div class="meta">{meta_line(meta)}</div>'
        f'{summary}'
        f'<div class="tags">{tag_html(meta)}</div></li>'
    )


def _anchor(*parts: str) -> str:
    """Stable id for an index heading, so notes can link to a section: "Transferability" -> "transferability"."""
    return re.sub(r"[^a-z0-9]+", "-", "-".join(p for p in parts if p).lower()).strip("-")


def render_index(entries: list[tuple[Path, dict]]) -> None:
    rows = []
    for cat, subs in _grouped(entries):
        rows.append(f'<section class="cat"><h2 id="{_anchor(cat)}">{html.escape(cat)}</h2>')
        for sub, metas in subs:
            rows.append('<div class="sub">')
            if sub:
                rows.append(f'<h3 id="{_anchor(cat, sub)}">{html.escape(sub)}</h3>')
            rows.append("<ul class='papers'>")
            rows += [_entry_html(m) for m in metas]
            rows.append("</ul></div>")
        rows.append("</section>")

    counts: dict[str, int] = {}
    for _, m in entries:
        for t in set(m.get("tags") or []):
            counts[t] = counts.get(t, 0) + 1
    common = sorted((t for t, c in counts.items() if c >= 3), key=lambda t: (-counts[t], t))
    chips = "".join(f'<button type="button" class="chip" data-tag="{html.escape(t)}">{html.escape(t)} '
                    f'<small>{counts[t]}</small></button>' for t in common)
    statuses = sorted({m["status"] for _, m in entries if m.get("status")})
    status_opts = "".join(f'<option value="{html.escape(x)}">{html.escape(x)}</option>' for x in statuses)
    controls = f"""<div class="finder">
<input id="q" type="search" placeholder="Search titles, authors, tags, summaries ( / )" aria-label="Search" autocomplete="off">
<div class="finder-row">
<label>Show <select id="f-kind"><option value="">everything</option><option value="paper">papers</option>
<option value="foundations">foundations</option><option value="topic">topics</option></select></label>
<label>Status <select id="f-status"><option value="">any</option>{status_opts}</select></label>
<label>Sort <select id="f-sort"><option value="new">newest first</option><option value="old">oldest first</option>
<option value="title">title A&ndash;Z</option></select></label>
<span id="count" class="count" aria-live="polite"></span>
<button type="button" id="reset" class="reset" hidden>Clear filters</button>
</div>
<details class="chips"><summary>Tags used on three or more pages ({len(common)})</summary><div>{chips}</div></details>
</div>"""

    n_paper = len([m for _, m in entries if m.get("kind") not in NON_PAPER_KINDS])
    n_other = len(entries) - n_paper
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Theory-inclined papers, with annotations</title>
<link rel="stylesheet" href="style.css">
{HEAD_SCRIPT}</head>
<body>{TOGGLE_BUTTON}<div class="layout">
{sidebar_html(entries, None, depth=0)}
<main>
<header class="paper-head"><h1>Theory-inclined papers, with annotations</h1>
<p class="meta">{n_paper} paper(s), {n_other} background page(s)</p></header>
{controls}
<div id="results">{"".join(rows)}</div>
<p id="none" class="none" hidden>No page matches. Try fewer words or clear the filters.</p>
</main></div>{TOGGLE_SCRIPT}<script src="site.js" defer></script></body></html>
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
    shutil.copy2(STYLE.with_name("site.js"), BUILD / "site.js")
    for notes, meta in entries:
        render_one(notes, meta, entries)
    render_index(entries)
    print(f"\nOpen {BUILD / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
