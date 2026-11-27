`docs/export_notebooks.py` exempts from `--check` only the notebooks an
**open** issue tracks. Three of its four entries named [next#247](https://github.com/rel-int/discopy-next/issues/247), closed as
completed on 2026-10-26: `optyx/feed_forward_example` executes clean and is
dropped, `optyx/photon_distinguishability` is fixed above, and
`optyx/readme_example` fails on [next#251](https://github.com/rel-int/discopy-next/issues/251)'s cotengra process pool, which it
now names. `--check` prints nothing for an exemption that has stopped being
needed, so the list could only grow ([next#435](https://github.com/rel-int/discopy-next/issues/435)).
