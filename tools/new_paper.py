#!/usr/bin/env python3
"""Scaffold papers/<slug>/notes.md with the front matter fields the build expects."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = """---
title: "TODO title"
authors: "TODO, first author et al."
venue: "TODO venue"
year: {year}
url: ""
pdf_url: ""
tags: [todo]
status: reading
---

## In one paragraph

TODO: what the paper claims, in plain language.

## Setup and notation

| Symbol | Meaning |
|---|---|
| $x$ | TODO |

## Main results

### Result 1

TODO. Inline math is `$a + b$`; display math is a `$$ ... $$` block:

$$
\\mathcal{{L}} = -\\alpha\\, \\mathbb{{E}}[u \\cdot v] + \\Phi(\\mu)
$$

## Questions and doubts

- TODO: what is still unclear, and what would settle it.

## Takeaways

- TODO.

---

*Notes started {today}.*
"""


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit("usage: make new SLUG=<year>-<firstauthor>-<short-topic>")
    slug = sys.argv[1].strip().strip("/")
    target = ROOT / "papers" / slug
    notes = target / "notes.md"
    if notes.exists():
        sys.exit(f"{notes.relative_to(ROOT)} already exists; nothing written.")
    target.mkdir(parents=True, exist_ok=True)
    today = date.today()
    notes.write_text(TEMPLATE.format(year=today.year, today=today.isoformat()), encoding="utf-8")
    print(f"Created {notes.relative_to(ROOT)}")
    print("Next: fill in the front matter, then `make fetch` to pull the PDF, then `make`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
