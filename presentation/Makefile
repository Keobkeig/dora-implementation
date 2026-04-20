LATEXMK ?= latexmk
MAIN := slides.tex

.PHONY: all clean

all:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN)

clean:
	$(LATEXMK) -C
