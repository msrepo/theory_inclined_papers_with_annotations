#!/usr/bin/env python3
"""Write a copy of the deck without its Supplementary section.

The supplementary section starts at the lead slide titled "# Supplementary" and runs to the end
of the file, so the copy is everything before the `---` that opens that slide.

Usage:  python3 code/split_main.py slides.md slides-main.md
"""
import sys

src, dest = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
marker = text.index("# Supplementary")
cut = text.rfind("\n---\n", 0, marker)
if cut < 0:
    sys.exit("no slide separator found before '# Supplementary'")
open(dest, "w", encoding="utf-8").write(text[:cut].rstrip() + "\n")
print(f"{dest}: {text[:cut].count(chr(10) + '---' + chr(10)) + 1} slides, supplementary section left out")
