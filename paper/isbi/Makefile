# The 4-page ISBI cut. IEEEtran.cls/.bst are vendored here (CTAN v1.8b) because
# this box's TeX Live does not ship them; replace both from the official ISBI
# author kit when it posts. Figures and tables are shared with ../: regenerate
# via `make -C .. figures`.

TEX := pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean veryclean

all: isbi.pdf

isbi.pdf: isbi.tex ../macros.tex ../refs.bib IEEEtran.cls IEEEtran.bst
	$(TEX) isbi
	-bibtex isbi
	$(TEX) isbi
	$(TEX) isbi

clean:
	rm -f *.aux *.log *.bbl *.blg *.out

veryclean: clean
	rm -f isbi.pdf
