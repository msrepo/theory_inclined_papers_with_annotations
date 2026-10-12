MARP ?= npx --yes @marp-team/marp-cli
FLAGS ?= --allow-local-files

SRC := slides.md
FIGS := $(wildcard figures/*)

.PHONY: all pdf main html pptx figures clean

all: pdf

pdf: slides.pdf
main: slides-main.pdf
html: slides.html
pptx: slides.pptx

slides.pdf: $(SRC) $(FIGS)
	$(MARP) $(SRC) -o $@ $(FLAGS)

# the deck without the Supplementary section: `make main` -> slides-main.pdf
slides-main.md: $(SRC) code/split_main.py
	python3 code/split_main.py $(SRC) $@

slides-main.pdf: slides-main.md $(FIGS)
	$(MARP) slides-main.md -o $@ $(FLAGS)

slides.html: $(SRC) $(FIGS)
	$(MARP) $(SRC) -o $@ $(FLAGS)

slides.pptx: $(SRC) $(FIGS)
	$(MARP) $(SRC) -o $@ $(FLAGS)

# redraw the figures from code/figures.py
figures:
	python3 code/figures.py --figures

clean:
	rm -f slides.pdf slides.html slides.pptx slides-main.md slides-main.pdf
