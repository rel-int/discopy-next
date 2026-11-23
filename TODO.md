# TODO

Review round on #697 (cubic at 35160248), the feedback verbatim:

> cubic 4660964776 (`discopy/matrix.py:572`): P2: `Matrix.algebra` is inherited by `Tensor`, whose dom and cod are `Dim` objects. `index(dom)` raises TypeError there because `Dim` has no `__index__`, so `Tensor.algebra(List, Dim(1), Dim(2))` crashes instead of returning the sum. Either override `algebra` in `Tensor`, or avoid `index` in the name and add a Tensor test.

> cubic 4660964766 (`CHANGELOG.md:71`): P2: This entry presents `tensor` without its caveat. `Arrow.tensor` is monoidal only for commutative monads, and interchange fails for the `List` monad this PR adds. Add that qualification here.

- [ ] `Matrix.algebra` names its homset by the boundaries cast to `cls.ob`, so `Tensor` inherits it, with a `Tensor` test
- [ ] the changelog bullet says `tensor` is monoidal only for a commutative monad
- [ ] checks: pytest (with test/tensor.py), pflake8, pylint, ruff, ty
