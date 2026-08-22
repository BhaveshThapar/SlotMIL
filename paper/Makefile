# No latexmk on this box, so the passes are spelled out. bibtex runs between
# them because the bibliography is only known after the first pass.
#
# Figures and tables are inputs, not build products of this Makefile: regenerate
# them with `make figures` (needs the repo venv and the artefacts under runs/).

TEX      := pdflatex -interaction=nonstopmode -halt-on-error
PYTHON   := ../.venv/bin/python

.PHONY: all clean figures isbi veryclean

all: main.pdf supplement.pdf

isbi:
	$(MAKE) -C isbi

main.pdf: main.tex macros.tex refs.bib tables/verdicts.tex tables/h5_sensitivity.tex \
          tables/arm_in_lung.tex
	$(TEX) main
	-bibtex main
	$(TEX) main
	$(TEX) main

supplement.pdf: supplement.tex macros.tex tables/falsifiers.tex tables/verdicts.tex \
                tables/h5_sensitivity.tex
	$(TEX) supplement
	$(TEX) supplement

figures:
	cd .. && $(PYTHON) scripts/make_paper_figures.py \
		--out paper/figures --tables paper/tables

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc

veryclean: clean
	rm -f main.pdf supplement.pdf
