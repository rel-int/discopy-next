`build`'s `test` and `docs` jobs, the two that install Graphviz, run on
`ubuntu-24.04`: `ubuntu-latest` resolved to two images inside one run and
graphviz is not in the sources the Ubuntu 26.04 one ships, so a job asking
for it passed or failed by the image it landed on. `.github/tests/workflows.py`
keeps the pin ([next#692](https://github.com/rel-int/discopy-next/pull/692)).
