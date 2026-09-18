# Build rendered HTML notes from papers/*/notes.md
#
#   make                                render everything into build/
#   make new SLUG=2026-lastname-topic   scaffold a new paper folder
#   make fetch                          download PDFs named in the front matter
#   make serve                          render, then serve build/ on $(PORT)
#   make open                           render, then open build/index.html
#   make list                           list the papers in the repo
#   make check                          verify pandoc and that notes parse
#   make clean                          remove build/

PYTHON ?= python3
PORT   ?= 8000

TOOLS := tools
NOTES := $(wildcard papers/*/notes.md)
FIGS  := $(wildcard papers/*/figures/*)
STAMP := .build-stamp

.DEFAULT_GOAL := html
.PHONY: html new fetch serve open list check clean help

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

clean:
	@rm -rf build $(STAMP)
	@echo "removed build/"

help:
	@sed -n '2,10p' Makefile | sed 's/^#//'
