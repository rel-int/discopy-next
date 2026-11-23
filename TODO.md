# TODO

Review round on #697 (codex at ddf2a307, the commit before 493f7d27), the feedback verbatim:

> codex 4660948126 (`discopy/enriched.py:317`): Use the underlying functor when mapping objects too. For a monoidal underlying category, this fallback treats composite types as keys in `ob_map` instead of evaluating them with `C.functor_factory`. [...] `F(x @ x)` raises `KeyError` [...]

> codex 4660948130 (`discopy/enriched.py:222`): Normalize shorthand boundaries before tensoring them. [...] `Lists.id(int) @ Lists.id(int)` attempts `int @ int` and raises `TypeError`. [...]

> codex 4660948136 (`discopy/matrix.py:567`): Include the homset boundaries in the algebra identity. [...] `Matrix[int].algebra(List, 1, 1)` equals `Matrix[int].algebra(List, 2, 2)` [...]

- [ ] `enriched.Functor` maps objects with the `functor_factory` of the category, with a test
- [ ] `enriched.Arrow` casts its boundaries to objects of the category, with a test of the tensor
- [ ] the algebras of two homsets: already fixed by 493f7d27, checked on the head
- [ ] checks: pytest, pflake8, pylint, ruff, ty
