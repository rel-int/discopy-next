# TODO

> Can you solve the library bug, and the 'still open' parts? You can delegate it to a smart opus subagent

`stream.Stream[markov.Diagram].copy(stream.Ty.sequence(markov.Ty("x")), 0)` raised `AttributeError: 'Ty' object has
no attribute 'dom'`, and so did `x ** 0` on a `stream.Ty`: its `__pow__` was that of a coloured `monoidal.Ty`, which
returns the identity on the colour `self.dom` for `n <= 0`, and a stream of types has no colour.

- [ ] `stream.Ty.__pow__` is the power of the free monoid, `monoidal.List.__pow__`: `x ** 0` is the unit `Ty()`
- [ ] `Stream.copy(x, 0)` is the discard; no doctest, since `stream.py` is at pylint's limit of 1000 lines
- [ ] tests in `test/stream.py`: powers of constant and non-constant streams, copies of 0 to 3 wires
- [ ] a `Fixed` bullet in `CHANGELOG.md`
- [ ] the stream cells of the property matrix and the tests of the touched modules
