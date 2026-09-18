# Build rendered HTML notes from papers/*/notes.md
#
#   make                                render everything into build/
#   make new SLUG=2026-lastname-topic   scaffold a new paper folder
#   make fetch                          download PDFs named in the front matter
#   make serve                          render, then serve build/ on $(PORT)
#   make open                           render, then open build/index.html
#   make list                           list the papers in the repo
#   make check                          verify pandoc and that notes parse
#   make verify                         run every papers/*/code/*.py
#   make clean                          remove build/

PYTHON ?= python3
PORT   ?= 8000

TOOLS := tools
ROOTS := papers foundations topics
NOTES := $(foreach r,$(ROOTS),$(wildcard $(r)/*/notes.md))
FIGS  := $(foreach r,$(ROOTS),$(wildcard $(r)/*/figures/*))
CODE  := $(foreach r,$(ROOTS),$(wildcard $(r)/*/code/*.py))
STAMP := .build-stamp

.DEFAULT_GOAL := html
.PHONY: html new fetch serve open list check verify clean help

html: $(STAMP)

$(STAMP): $(NOTES) $(FIGS) $(TOOLS)/build.py $(TOOLS)/style.css
	@$(PYTHON) $(TOOLS)/build.py
	@touch $@

new:
	@test -n "$(SLUG)" || { echo "usage: make new SLUG=2026-lastname-topic"; exit 1; }
	@$(PYTHON) $(TOOLS)/new_paper.py "$(SLUG)"

fetch:
	@$(PYTHON) $(TOOLS)/fetch_pdfs.py

serve: html
	@echo "Serving http://localhost:$(PORT)  (ctrl-c to stop)"
	@cd build && $(PYTHON) -m http.server $(PORT)

open: html
	@open build/index.html 2>/dev/null || xdg-open build/index.html

list:
	@$(PYTHON) -c "import sys; sys.path.insert(0,'$(TOOLS)'); import build; \
	  [print('%-40s %s' % (m['slug'], m.get('title',''))) for _, m in build.discover()]"

check:
	@command -v pandoc >/dev/null || { echo "pandoc missing: brew install pandoc"; exit 1; }
	@$(PYTHON) -c "import sys; sys.path.insert(0,'$(TOOLS)'); import build; \
	  e = build.discover(); print('pandoc ok; %d note(s) parse' % len(e))"

verify:
	@test -n "$(CODE)" || { echo "no code to run"; exit 0; }
	@for f in $(CODE); do \
	  echo "=== $$f ==="; $(PYTHON) $$f || exit 1; echo; \
	done
	@echo "all code ran"

clean:
	@rm -rf build $(STAMP)
	@echo "removed build/"

help:
	@sed -n '2,10p' Makefile | sed 's/^#//'
