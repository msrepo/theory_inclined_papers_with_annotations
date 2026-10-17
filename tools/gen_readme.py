#!/usr/bin/env python3
"""Regenerate the Foundations, Topics and Papers tables in README.md from front matter.

The tables sit between `<!-- BEGIN name -->` and `<!-- END name -->` markers and are
rewritten in full, so edit a page's front matter (`summary`, `ext_links`, ...) and run
`make readme` instead of touching the README rows by hand. Everything outside the
markers is left alone.

Run:  python3 tools/gen_readme.py           rewrite README.md
      python3 tools/gen_readme.py --check   exit 1 if README.md is out of date
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402

ROOT = build.ROOT
SITE = "https://msrepo.github.io/theory_inclined_papers_with_annotations/"
VENUE_LABEL = {"proceedings.mlr.press": "PMLR", "openreview.net": "OpenReview", "papers.nips.cc": "NeurIPS",
               "proceedings.neurips.cc": "NeurIPS", "pmc.ncbi.nlm.nih.gov": "PMC", "www.ncbi.nlm.nih.gov": "PMC"}


def short_authors(authors: str) -> str:
    names = [n.strip() for n in re.split(r",| and ", re.sub(r"\([^)]*\)", "", authors or "")) if n.strip()]
    sur = [n.split()[-1] for n in names]
    if not sur:
        return ""
    if len(sur) > 3:
        return f"{sur[0]} et al."
    return sur[0] if len(sur) == 1 else ", ".join(sur[:-1]) + " & " + sur[-1]


def paper_link(url: str) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc
    m = re.search(r"arxiv\.org/abs/([\w.]+)", url)
    if m:
        return f"[arXiv:{m.group(1)}]({url})"
    return f"[{VENUE_LABEL.get(host, host.removeprefix('www.'))}]({url})"


def note_links(notes: Path, meta: dict) -> str:
    d, slug = notes.parent, meta["slug"]
    out = [f"[read online]({SITE}{slug}/)"]
    if (d / "figures" / "interactive.html").exists():
        out.append(f"[interactive]({SITE}{slug}/figures/interactive.html)")
    out.append(f"[source]({d.relative_to(ROOT)}/notes.md)")
    if any((d / "code").glob("*.py")):
        out.append(f"[code]({d.relative_to(ROOT)}/code/)")
    return " · ".join(out)


def cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def table(header: str, rows: list[str]) -> list[str]:
    cols = header.count("|") - 1
    return [header, "|" + "---|" * cols] + rows


def papers_block(entries) -> str:
    rows = []
    items = [(n, m) for n, m in entries if m["kind"] == "paper"]
    items.sort(key=lambda x: (-int(x[1].get("year") or 0), x[1]["slug"]))
    for notes, m in items:
        ext = "".join(f" · [{lab}]({u})" for lab, u in (e.split("|", 1) for e in m.get("ext_links") or []))
        left = f"{short_authors(m.get('authors', ''))}, *{m['title']}*, {m.get('venue', '')}".replace(", ,", ",").rstrip(", ")
        link = paper_link(m.get("url", ""))
        paper = left + (f" — {link}" if link else "") + ext
        rows.append(f"| {m.get('year', '')} | {cell(paper)} | {note_links(notes, m)} | {cell(m.get('summary', ''))} |")
    return "\n".join(table("| Year | Paper | Notes | Topic |", rows))


def page_rows(items) -> list[str]:
    return [f"| *{cell(m['title'])}* — {note_links(n, m)} | {cell(m.get('summary', ''))} |"
            for n, m in sorted(items, key=lambda x: str(x[1]["title"]).lower())]


def foundations_block(entries) -> str:
    groups: dict[str, list] = {}
    for n, m in entries:
        if m["kind"] == "foundations":
            g = m.get("subcategory") if m.get("category") == "Foundations" else m.get("category")
            groups.setdefault(g or "Other", []).append((n, m))
    order = [s for s in dict(build.CATEGORIES)["Foundations"] if s in groups] + \
            [g for g in groups if g not in dict(build.CATEGORIES)["Foundations"]]
    out = []
    for g in order:
        out += [f"**{g}**", ""] + table("| Page | Covers |", page_rows(groups[g])) + [""]
    return "\n".join(out).rstrip()


def topics_block(entries) -> str:
    return "\n".join(table("| Page | Covers |", page_rows([(n, m) for n, m in entries if m["kind"] == "topic"])))


def replace(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN {name} -->).*?(<!-- END {name} -->)", re.S)
    if not pat.search(text):
        sys.exit(f"README.md has no <!-- BEGIN {name} --> ... <!-- END {name} --> markers")
    return pat.sub(lambda m: m.group(1) + "\n" + body + "\n" + m.group(2), text)


def main() -> int:
    entries = build.discover()
    readme = ROOT / "README.md"
    old = readme.read_text(encoding="utf-8")
    new = replace(old, "foundations", foundations_block(entries))
    new = replace(new, "topics", topics_block(entries))
    new = replace(new, "papers", papers_block(entries))
    if "--check" in sys.argv:
        if new != old:
            sys.exit("README.md is out of date: run `make readme`")
        print("README.md is up to date")
        return 0
    if new != old:
        readme.write_text(new, encoding="utf-8")
        print("README.md tables rewritten")
    else:
        print("README.md already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
