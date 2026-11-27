`paper/`, the source of the tool paper in Typst: `paper/transpile.py` writes
its LaTeX from the part of Typst it is written in, `paper/build.py` runs
its listings and draws each figure with the Typst and the TikZ backends,
`test/paper_build.py` checks both and the `paper` workflow uploads both
PDFs ([next#577](https://github.com/rel-int/discopy-next/pull/577)).
