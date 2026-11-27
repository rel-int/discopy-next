# The tool paper is written in Typst, its LaTeX transpiled

## Context

The tool paper for Compositionality ([#564](https://github.com/rel-int/discopy-next/issues/564))
needs LaTeX for the journal and arXiv, while the library had just gained a Typst drawing backend.
toumix, 2026-11-16: Typst and LaTeX "in the same Ci From the start", with "custom code that
transpiles this paper adhoc", and the snippets compiled and tested there.

## Decision

`paper/` is Typst source. `paper/transpile.py` writes its LaTeX, and `test/paper_build.py` runs its
listings and draws each figure with both the Typst and the TikZ backends
([#577](https://github.com/rel-int/discopy-next/pull/577),
[#581](https://github.com/rel-int/discopy-next/pull/581)).

## Consequences

- The paper can only use the part of Typst the transpiler reads.
- Each figure is drawn by two backends, which checks both.

## Status

Superseded by [`2026-11-17-the-tool-paper-in-latex.md`](2026-11-17-the-tool-paper-in-latex.md).
