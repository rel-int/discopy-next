> USER, reading the discopy.abc docs: "bang is missing from the abc diagram."
>
> Add `bang` to the architecture diagram in `docs/api/architecture.html` (included from the `discopy.abc` docstring, rendered at `_api/discopy.abc.html`, links relative to `_api/`). The diagram is syntax-only since rel-int/discopy-next#726, merged as `9b688a3`. Keep it that way.
>
> `discopy.bang` is the free linear category: a monoidal category with a bang, needing no symmetry and no closure, so it sits right above `discopy.monoidal`, the way `discopy.para` and `discopy.monad` sit beside the hierarchy. Its own module docstring says so. Draw it branching off `monoidal`, the way `cmap` and `hypergraph` branch off `cat` and `spacial` and `biclosed` branch off `monoidal`. Link it to `../_api/discopy.bang.html`. Keep the table readable; do not add anything else.
>
> `discopy.bang` is already listed in `docs/api/syntax.rst`, so the page needs no change unless you find it missing. Update the abc prose only if it now misdescribes the diagram.
>
> Follow the repo's AGENTS.md and changelog convention. Open a pull request. Reply to https://ampcode.com/threads/T-01a2123a-e6ca-715d-8fa5-ea931602b4f3 with its URL, or with what blocked you.

- [WIP] @T-dc6d2cec-2026-11-25 17:03 Draw bang branching off monoidal in docs/api/architecture.html
- [ ] Changelog entry
