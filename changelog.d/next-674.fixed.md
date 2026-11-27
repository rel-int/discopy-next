`closed.Abstraction.eval` curries its variable on the side that
`Application.eval` evaluates, so a β-redex and its reduct are sent to the
same morphism by a functor into a rigid category, e.g. tensors, where they
failed to compose or gave transposed results ([next#674](https://github.com/rel-int/discopy-next/pull/674), [next#676](https://github.com/rel-int/discopy-next/issues/676)).
