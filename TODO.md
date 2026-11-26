# TODO

> Can you solve the library bug, and the 'still open' parts? You can delegate it to a smart opus subagent

The gate, on the first round:

> FAIL ty: 94 findings, base 93, 1 new; first: discopy/stream.py:900 warning[invalid-argument-type] Argument to
> `Stream.__init__` is incorrect: Expected `Ty[Unknown] | None`, found `List[Unknown]`

- [x] `monoidal.List.tensor` and `monoidal.List.__pow__` return `Self`, which they do (`type(self)(...)`), as those of
      `monoidal.Ty` say, so that the power a `stream.Ty` borrows is typed a `stream.Ty`
