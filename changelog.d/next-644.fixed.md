`draw(format="typst", doctest=path)` checks the drawing against a
baseline, written when missing, as `draw(doctest=...)` does for the other
backends: the Typst source for a path ending in `.typ`, compared byte for
byte, its SVG otherwise. It ignored `doctest` ([next#644](https://github.com/rel-int/discopy-next/pull/644)).
