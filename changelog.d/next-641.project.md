`docs/export_notebooks.py` runs each notebook as the Python file
`marimo convert` writes, not its Markdown: marimo runs that file as
`__main__`, which a worker spawned by `multiprocessing`, e.g. by cotengra's
optimizer, imports again, and a Markdown one failed it with a `SyntaxError`
([next#641](https://github.com/rel-int/discopy-next/pull/641)). `optyx/readme_example`, its one exemption, now runs past the
pool and fails at `chi=1` on [next#442](https://github.com/rel-int/discopy-next/issues/442) instead, which it names ([next#641](https://github.com/rel-int/discopy-next/pull/641)).
