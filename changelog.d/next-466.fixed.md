The TikZ backend spells a colour that matplotlib draws and xcolor does not
know, e.g. `tab:blue`, `C0`, `0.5`, `navy` or an RGB tuple, by its RGB
values, with or without `use_tikzstyles`, so its drawing compiles; it was
written raw, which pdflatex rejects, and a tuple raised `TypeError`
([next#466](https://github.com/rel-int/discopy-next/issues/466)).
