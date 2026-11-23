# TODO

Review round on #697 (cubic's re-review at 068bb325), the feedback verbatim:

> cubic 4660928484 (`discopy/matrix.py:567`): P2: Matrix algebras for different `dom`/`cod` shapes compare equal because both use the name `"sum"` and carrier `cls`. Include the canonical dimensions in the algebra name so composability checks distinguish these homsets.

> cubic 4660928488 (`CHANGELOG.md:75`): P3: The `sum.Sum` claim applies only to `cat.Arrow`. `Matrix.algebra` sends a list of matrices to a `Matrix`, so the sentence should say that for arrows.

- [ ] `Matrix.algebra` and `cat.Arrow.algebra` name their homset, so `monad.algebra.Homomorphism` composability tells them apart, with a test
- [ ] the changelog bullet says `Matrix` adds the matrices
- [ ] checks: pytest, pflake8, pylint, ruff, ty
