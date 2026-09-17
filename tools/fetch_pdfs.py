#!/usr/bin/env python3
"""Download each paper's PDF from the pdf_url in its front matter.

PDFs are gitignored, so this is how a fresh clone gets them. Papers that
already have a paper.pdf, or that have no pdf_url yet, are skipped.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import PAPERS, ROOT, parse_front_matter  # noqa: E402

UA = "Mozilla/5.0 (compatible; theory-notes-fetcher/1.0)"


def main() -> int:
    missing_url = []
    for notes in sorted(PAPERS.glob("*/notes.md")):
        pdf = notes.parent / "paper.pdf"
        rel = notes.parent.relative_to(ROOT)
        if pdf.exists():
            print(f"  have   {rel}")
            continue
        meta, _ = parse_front_matter(notes.read_text(encoding="utf-8"))
        url = (meta.get("pdf_url") or "").strip()
        if not url:
            missing_url.append(str(rel))
            continue
        print(f"  get    {rel}  <-  {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                pdf.write_bytes(resp.read())
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  FAILED {rel}: {exc}")

    if missing_url:
        print("\nNo pdf_url set (drop the PDF in by hand as paper.pdf, or fill the field in):")
        for rel in missing_url:
            print(f"  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
