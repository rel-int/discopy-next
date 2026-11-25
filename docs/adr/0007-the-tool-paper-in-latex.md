# 7. The tool paper is LaTeX only

## Context

[0006](0006-the-tool-paper-in-typst.md) made the paper Typst source with transpiled LaTeX. A day
later toumix reversed it, 2026-11-17, on
[#564](https://github.com/rel-int/discopy-next/issues/564): "I think we're going back on our typst
decision and moving back to just LaTeX for the paper". The same day he moved the draft to the
wiki: "it makes more sense to move the tool paper to the wiki for now".

## Decision

The paper's source is LaTeX, with no Typst source and no transpiler. Its figures are drawn by the
library's TikZ backend alone. The library keeps its Typst backend, which the paper describes as a
feature. The draft lives in its author's folder on the wiki, not in this repository.

## Consequences

- `paper/`, `test/paper_build.py`, the `paper` workflow and CONTRIBUTING.md's "Build the paper"
  are removed by [#597](https://github.com/rel-int/discopy-next/pull/597).
- The Typst backend is no longer checked by the paper's figures, only by its own tests.

## Status

Accepted, 2026-11-17. #597, which carries it out, is open.
