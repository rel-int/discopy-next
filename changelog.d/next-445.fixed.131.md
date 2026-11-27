`python.finset.Permutation` refuses a `Function` whose domain is not its
codomain, so that `then` and `tensor` with a discard or a projection raise
rather than return a permutation of the wrong size, and `Permutation.then`
raises `AxiomError` on a `Function` that does not compose with it, as
`Function.then` does ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
