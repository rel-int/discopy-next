# TODO

Review round on #697, the feedback verbatim:

> cubic 4660898366: P2: This drops every input and output type except the first, but the accepted `discopy.python.multiplicative.Function` supports multi-argument and multi-result arrows. Preserve and pack the full boundaries when building the channel, or explicitly reject non-singleton boundaries.

> codex 4660899298: Preserve complete function boundaries when converting to channels. For `discopy.python.Function` arrows with zero or multiple input/output types, selecting `inside[0]` either raises `IndexError` or silently drops part of the boundary. [...]

> codex 4660899301: Evaluate enclosed diagrams with a matching base functor. When the underlying category is `monoidal.Diagram`, the hard-coded `cat.Functor.__call__` cannot evaluate composites from the documented mapping of generating boxes. [...] evaluating `[f >> f]` raises `KeyError` [...]

USER's instruction for the round: reject non-singleton boundaries with a clear error and a test, no packing; evaluate the inner arrows with the functor of the underlying category.

- [ ] `to_channel` rejects a boundary of zero or several types, with a message and a test
- [ ] `enriched.Functor` maps the inner arrows with `C.functor_factory`, with the `monoidal.Diagram` test
- [ ] checks: pytest, pflake8, pylint, ruff, ty
