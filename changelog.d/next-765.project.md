Read the Docs embeds the notebooks the `build` workflow renders on `main`
instead of linking to their sources, still running none itself: a
`notebooks` job pushes them to the `rendered-notebooks` branch, whose
tarball `docs/export_notebooks.py` downloads when `DISCOPY_RENDERED_NOTEBOOKS`
names it, as `docs/conf.py` does on Read the Docs ([next#765](https://github.com/rel-int/discopy-next/pull/765)).
