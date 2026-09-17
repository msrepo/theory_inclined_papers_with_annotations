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
    found = []
    for notes in sorted(PAPERS.glob("*/notes.md")):
        meta, _ = parse_front_matter(notes.read_text(encoding="utf-8"))
        meta.setdefault("title", notes.parent.name)
        meta["slug"] = notes.parent.name
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


def render_one(notes: Path, meta: dict) -> None:
    slug = meta["slug"]
    out_dir = BUILD / slug
    out_dir.mkdir(parents=True, exist_ok=True)

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
    header_file = out_dir / ".header.html"
    header_file.write_text(header, encoding="utf-8")

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
        "--output", str(out_dir / "index.html"),
    ]
    subprocess.run(cmd, check=True)
    header_file.unlink()
    body_file.unlink()
    print(f"  built  {slug}")


def render_index(entries: list[tuple[Path, dict]]) -> None:
    by_year: dict[str, list[dict]] = {}
    for _, meta in entries:
        by_year.setdefault(str(meta.get("year", "undated")), []).append(meta)

    rows = []
    for year in sorted(by_year, reverse=True):
        rows.append(f"<h2>{html.escape(year)}</h2><ul class='papers'>")
        for meta in sorted(by_year[year], key=lambda m: str(m["title"]).lower()):
            status = meta.get("status", "")
            badge = f'<span class="status status-{html.escape(status)}">{html.escape(status)}</span>' if status else ""
            rows.append(
                f'<li><a class="title" href="{html.escape(meta["slug"])}/index.html">'
                f'{html.escape(str(meta["title"]))}</a>{badge}'
                f'<div class="meta">{meta_line(meta)}</div>'
                f'<div class="tags">{tag_html(meta)}</div></li>'
            )
        rows.append("</ul>")

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Theory-inclined papers, with annotations</title>
<link rel="stylesheet" href="style.css"></head>
<body><main>
<header class="paper-head"><h1>Theory-inclined papers, with annotations</h1>
<p class="meta">{len(entries)} paper(s)</p></header>
{"".join(rows)}
</main></body></html>
"""
    (BUILD / "index.html").write_text(doc, encoding="utf-8")
    print(f"  built  index.html ({len(entries)} paper(s))")


def main() -> int:
    if shutil.which("pandoc") is None:
        sys.exit("pandoc not found. Install it: brew install pandoc")
    entries = discover()
    if not entries:
        sys.exit("No papers/<slug>/notes.md found. Scaffold one with: make new SLUG=my-paper")
    BUILD.mkdir(exist_ok=True)
    shutil.copy2(STYLE, BUILD / "style.css")
    for notes, meta in entries:
        render_one(notes, meta)
    render_index(entries)
    print(f"\nOpen {BUILD / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
